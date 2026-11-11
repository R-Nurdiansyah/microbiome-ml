"""Core data models for microbiome data structures."""

import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Type, Union

import polars as pl

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.features import SampleFeatureSet
from microbiome_ml.wrangle.icc import taxon_icc

logger = logging.getLogger(__name__)

# How a profile's rows relate to each other (SingleM's glossary terms):
#   unfilled -- each row is the reads assigned to that taxon and no deeper,
#               so a taxon's total is the sum of every row under it. This is
#               SingleM's condensed output; one row per species is a special
#               case of it.
#   filled   -- each row already includes all of its descendants, so a
#               taxon's total is its own row.
FILLED = "filled"
UNFILLED = "unfilled"
LAYOUTS = ("auto", FILLED, UNFILLED)
# A relative-abundance table at one rank cannot sum above 1 per sample;
# float32 rounding stays well inside this.
_SUM_TOLERANCE = 1e-4
# Lineage depth, carried through a filled profile's rank rows so the
# shallowest of a taxon's rows can be kept (see _rank_long).
_DEPTH = "__rank_depth"
# _detect_layout calls a profile filled when at least this share of its
# rows has its parent row present...
_MIN_PARENT_SHARE = 0.5
# ...and no more than this share of parents falls short of its children.
_MAX_SHORT_SHARE = 0.01


def _tidy_taxonomy(expr: pl.Expr) -> pl.Expr:
    """`a; b` -> `a;b`, and drop trailing empty ranks (`o__X;f__;g__`).

    An empty rank is "not assigned this deep", so a row's depth is its
    deepest *named* rank. Without this, `o__X;f__;g__;s__` would look like
    a species row and never like the order row it is.
    """
    return expr.str.replace_all(r";\s+", ";").str.replace(
        r"(?:;[a-z]__)+$", ""
    )


def _ends_at(rank: TaxonomicRanks) -> str:
    """Regex for rows whose deepest named rank is `rank`."""
    return f"(?:^|;){rank.prefix}[^;]+$"


def _names(rank: TaxonomicRanks) -> str:
    """Regex for rows with a named `rank` anywhere in the lineage."""
    return f"(?:^|;){rank.prefix}[^;]+(?:;|$)"


# How a feature table's values are expressed. The key is canonical -- it
# is what the feature set name is built from -- and the tuple is what a
# config may say instead. Adding one here means adding its name suffix
# to _TRANSFORM_SUFFIX in wrangle/dataset.py.
TRANSFORMS = {
    "relabund": ("relabund",),
    "clr": ("clr",),
    "presence": ("presence", "pa", "binary", "presence_absence"),
}


def canonical_transform(transform: str) -> str:
    """Resolve a configured transform name to its canonical spelling."""
    text = str(transform).strip().lower()
    for canonical, aliases in TRANSFORMS.items():
        if text in aliases:
            return canonical
    raise ValueError(
        f"transform must be one of {', '.join(sorted(TRANSFORMS))}, "
        f"got {transform!r}"
    )


def _sink_or_collect(frame: pl.LazyFrame, path: Path) -> None:
    """Write a LazyFrame to CSV, streaming when the plan allows it.

    `sink_csv` keeps memory flat; it raises on plans polars cannot
    stream, in which case fall back to the eager write.

    Writes to a sibling temporary file and moves it into place, because
    the frame is usually a scan of a CSV and may be a scan of this one
    (saving a dataset back over the directory it was loaded from).
    Writing in place would truncate the source mid-read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        try:
            frame.sink_csv(tmp)
        except Exception as exc:  # pragma: no cover - version/plan dependent
            logger.debug(
                "sink_csv unavailable for %s (%s); collecting", path, exc
            )
            frame.collect().write_csv(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class Fields(Enum):
    "Base class for field definitions with validation properties."

    def __init__(
        self,
        column_name: str,
        dtype: Any,
        required: bool,
        description: str,
        alternatives: Optional[List[str]] = None,
    ):
        self.column_name = column_name
        self.dtype = dtype
        self.required = required
        self.description = description
        self.alternatives = alternatives or []
        self.all_names = [column_name] + self.alternatives

    def find_column_name(self, lf: pl.LazyFrame) -> Optional[str]:
        """Find the actual column name in the LazyFrame from possible
        alternatives."""
        schema_names = lf.collect_schema().names()
        for name in self.all_names:
            if name in schema_names:
                return name
        return None


class ProfilesFields(Fields):
    """Enumeration of profiles fields with validation properties."""

    SAMPLE = (
        "sample",
        pl.Utf8,
        True,
        "Sample identifier",
        ["acc", "sample_id"],
    )
    TAXONOMY = (
        "taxonomy",
        pl.Utf8,
        True,
        "Taxonomic classification string",
        ["tax"],
    )
    RELABUND = (
        "relabund",
        pl.Float64,
        False,
        "Relative abundance value",
        ["relative_abundance", "abundance", "proportion"],
    )
    COVERAGE = ("coverage", pl.Float64, False, "Coverage value", ["cov"])


class RootFields(Fields):
    """Enumeration of root fields with validation properties."""

    SAMPLE = (
        "sample",
        pl.Utf8,
        True,
        "Sample identifier",
        ["acc", "sample_id"],
    )
    COVERAGE = (
        "root_coverage",
        pl.Float64,
        True,
        "Coverage value",
        ["coverage", "cov"],
    )


class TaxonomicProfiles:
    """Long-form taxonomic abundance data container (LazyFrame only).

    Stores taxonomic profiles in long format with exactly 3 columns:
    - sample: Sample identifier
    - taxonomy: Taxonomic classification string
    - relabund or coverage: Relative abundance or coverage value

    Attributes:
        profiles: Polars LazyFrame
        root: Optional Polars LazyFrame
        is_filled: True for a filled profile, False for an unfilled one
            (see FILLED / UNFILLED above). `layout` gives the same as text.
    """

    def __init__(
        self,
        profiles: Union[Path, str, pl.LazyFrame, pl.DataFrame],
        root: Optional[Union[Path, str, pl.LazyFrame, pl.DataFrame]] = None,
        check_filled: bool = True,
        sample_size: int = 1000,
        layout: str = "auto",
    ):
        """Initialize TaxonomicProfiles with validation and standardization.

        The profile is kept in the layout it arrives in. Nothing is filled
        here: every consumer (the per-sample total, rank features, the
        dominated-sample filter) reads either layout directly, and filling
        a species-level table multiplies its rows by the number of ranks.

        Args:
            profiles: Path to profiles file or LazyFrame/DataFrame with profiles data
            root: Optional path to root file or LazyFrame/DataFrame with root coverage data
            check_filled: With `layout="auto"`, detect the layout from the
                data. False keeps the old behaviour of assuming filled.
            sample_size: Samples read when detecting the layout
            layout: "auto", "filled" or "unfilled". Anything but "auto"
                is taken as given and skips detection.
        """
        layout = str(layout).strip().lower()
        if layout not in LAYOUTS:
            raise ValueError(
                f"layout must be one of {list(LAYOUTS)}, got {layout!r}"
            )

        # Load and standardize profiles
        profiles_lf = self._load_and_standardize(profiles, ProfilesFields)
        profiles_lf = self._standardize_tax_strings(profiles_lf)

        self.profiles = profiles_lf

        if layout != "auto":
            self.is_filled = layout == FILLED
            logger.info("Profiles layout given as %s", layout)
        elif check_filled:
            self.is_filled = self._detect_layout(sample_size) == FILLED
        else:
            self.is_filled = True  # Assume filled if not checking

        # Load and standardize root if provided
        root_lf = None
        if root is not None:
            root_lf = self._load_and_standardize(root, RootFields)
        else:
            try:
                root_lf = self._create_root_lf()
            except ValueError:
                root_lf = None

        self.root = root_lf

        profiles_lf = self.profiles

        if "coverage" in profiles_lf.collect_schema().names():
            if root_lf is None:
                # create root if not provided
                try:
                    root_lf = self._create_root_lf()
                    self.root = root_lf
                except Exception as e:
                    raise ValueError(
                        f"Error extracting root coverage data: {e}"
                    )
            try:
                relabund_lf = self._to_relabund_lf()
                self.profiles = relabund_lf
                self.is_relabund = True
            except Exception as e:
                raise ValueError(
                    f"Error converting coverage to relative abundance: {e}"
                )
        else:
            self.is_relabund = True  # Already in relative abundance format

    @property
    def layout(self) -> str:
        """FILLED or UNFILLED, the same thing `is_filled` says."""
        return FILLED if self.is_filled else UNFILLED

    def _load_data(
        self, data_source: Union[Path, str, pl.LazyFrame, pl.DataFrame]
    ) -> pl.LazyFrame:
        """Load data from various sources into a LazyFrame.

        Args:
            data_source: Path to file, existing LazyFrame, or DataFrame

        Returns:
            LazyFrame
        """
        if isinstance(data_source, pl.LazyFrame):
            return data_source
        elif isinstance(data_source, pl.DataFrame):
            return data_source.lazy()
        elif isinstance(data_source, (str, Path)):
            return pl.scan_csv(data_source)
        else:
            raise ValueError(
                f"Unsupported data source type: {type(data_source)}"
            )

    def _validate_and_standardize_fields(
        self, lf: pl.LazyFrame, field_enum: Type[Enum]
    ) -> pl.LazyFrame:
        """Validate required fields and standardize column names.

        Args:
            lf: LazyFrame to validate and standardize
            field_enum: Enum class with field definitions

        Returns:
            LazyFrame with standardized column names
        """
        missing_required = []
        rename_mapping = {}

        # Find actual columns and build rename mapping
        for field in field_enum:  # type: ignore
            actual_column = field.find_column_name(lf)  # type: ignore
            if actual_column:
                # Only rename if the actual column name is different from standard
                if actual_column != field.column_name:  # type: ignore
                    rename_mapping[actual_column] = field.column_name  # type: ignore
            elif field.required:  # type: ignore
                missing_required.append(
                    f"{field.column_name} (tried: {field.all_names})"  # type: ignore
                )

        if missing_required:
            raise ValueError(f"Missing required fields: {missing_required}")

        # Rename columns to standard names
        if rename_mapping:
            lf = lf.rename(rename_mapping)

        return lf

    def _load_and_standardize(
        self,
        data_source: Union[Path, str, pl.LazyFrame, pl.DataFrame],
        field_enum: Type[Enum],
    ) -> pl.LazyFrame:
        """Load data and standardize column names.

        Args:
            data_source: Data source to load
            field_enum: Field enum for validation

        Returns:
            Standardized LazyFrame
        """
        lf = self._load_data(data_source)
        return self._validate_and_standardize_fields(lf, field_enum)

    def _standardize_tax_strings(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Standardize taxonomy strings to use ';{prefix}' (no space after
        semicolon).

        Args:
            lf: LazyFrame with taxonomy column

        Returns:
            LazyFrame with standardized taxonomy strings
        """
        return lf.with_columns(
            _tidy_taxonomy(pl.col("taxonomy")).alias("taxonomy")
        )

    @classmethod
    def scan(
        cls,
        profiles: Union[Path, str],
        root: Optional[Union[Path, str]] = None,
        check_filled: bool = True,
        sample_size: int = 1000,
        layout: str = "auto",
    ) -> "TaxonomicProfiles":
        """Lazily load TaxonomicProfiles from files.

        Args:
            profiles: Path to profiles CSV file or directory containing profiles.csv
            root: Optional path to root CSV file
            check_filled: Whether to check if profiles are in filled format
            sample_size: Sample size for filled format checking
            layout: "auto", "filled" or "unfilled"; see __init__

        Returns:
            TaxonomicProfiles instance
        """
        # Handle directory input (from save/load workflow)
        profiles_path = Path(profiles)
        if profiles_path.is_dir():
            profiles_path = profiles_path / "profiles.csv"
            root_path = Path(profiles) / "root.csv"
            root = str(root_path) if root_path.exists() else root
            profiles = str(profiles_path)

        return cls(
            profiles=profiles,
            root=root,
            check_filled=check_filled,
            sample_size=sample_size,
            layout=layout,
        )

    def save(self, path: Union[Path, str]) -> None:
        """Save TaxonomicProfiles to CSV files in a directory.

        Written by streaming (`sink_csv`): the profile table is the
        largest object in a dataset, and collecting it just to copy it to
        disk was the single biggest allocation in `prepare`.

        Args:
            path: Directory path to save files (will be created if doesn't exist)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        _sink_or_collect(self.profiles, path / "profiles.csv")
        if self.root is not None:
            _sink_or_collect(self.root, path / "root.csv")

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        check_filled: bool = True,
        layout: str = "auto",
    ) -> "TaxonomicProfiles":
        """Load TaxonomicProfiles from directory containing CSV files.

        Args:
            path: Directory path containing profiles.csv and optionally root.csv
            check_filled: Whether to check if profiles are in filled format
            layout: "auto", "filled" or "unfilled"; see __init__

        Returns:
            TaxonomicProfiles instance
        """
        path = Path(path)

        profiles_path = path / "profiles.csv"
        root_path = path / "root.csv"

        if not profiles_path.exists():
            raise FileNotFoundError(f"profiles.csv not found in {path}")

        root = root_path if root_path.exists() else None

        return cls.scan(
            profiles=str(profiles_path),
            root=str(root) if root else None,
            check_filled=check_filled,
            layout=layout,
        )

    def _detect_layout(self, sample_size: int = 1000) -> str:
        """FILLED or UNFILLED, judged from up to `sample_size` samples.

        Each row is compared with its *direct* parent row only: the same
        lineage one named rank shorter, in the same sample.

        - Filled means a parent row exists for most rows, and holds at
          least the sum of its direct children (descendants included).
        - Unfilled means parent rows are mostly absent -- one row per
          species is the extreme, with none at all -- or fall short of
          their children, because they hold only their own reads.

        The old check compared each parent with every row beneath it at
        any depth, so a filled profile always failed it, and a profile
        with no parent rows passed it by making no comparison.
        """
        lf = self.profiles
        value = (
            "relabund"
            if "relabund" in lf.collect_schema().names()
            else "coverage"
        )
        samples = lf.select("sample").unique().collect()
        if samples.height > sample_size:
            keep = samples.sample(n=sample_size, seed=42)
            lf = lf.join(keep.lazy(), on="sample", how="semi")
        rows = (
            lf.select(
                "sample",
                _tidy_taxonomy(pl.col("taxonomy")).alias("taxonomy"),
                pl.col(value).cast(pl.Float64).alias("value"),
            )
            .filter(pl.col("taxonomy").str.contains(r"(?:^|;)[a-z]__[^;]+$"))
            .collect()
        )
        # Rows below a named rank, with that rank's lineage as `parent`.
        # A parent of `Root` alone is not a taxon, so it does not count.
        children = rows.with_columns(
            pl.col("taxonomy").str.replace(r";[^;]+$", "").alias("parent")
        ).filter(
            pl.col("taxonomy").str.contains(";")
            & pl.col("parent").str.contains(r"(?:^|;)[a-z]__[^;]+$")
        )
        if not children.height:
            logger.info(
                "Profiles layout: unfilled (no lineage has more than one "
                "named rank)"
            )
            return UNFILLED

        parents = rows.rename({"taxonomy": "parent", "value": "parent_value"})
        with_parent = children.join(
            parents.select(["sample", "parent"]).unique(),
            on=["sample", "parent"],
            how="semi",
        ).height
        compared = (
            children.group_by(["sample", "parent"])
            .agg(pl.col("value").sum().alias("child_sum"))
            .join(parents, on=["sample", "parent"], how="inner")
        )
        short = compared.filter(
            pl.col("parent_value") < pl.col("child_sum") * (1 - 1e-6) - 1e-12
        ).height

        parent_share = with_parent / children.height
        short_share = short / compared.height if compared.height else 0.0
        layout = (
            FILLED
            if parent_share >= _MIN_PARENT_SHARE
            and short_share <= _MAX_SHORT_SHARE
            else UNFILLED
        )
        logger.info(
            "Profiles layout: %s (%.0f%% of %d rows have their parent row; "
            "%d of %d parents hold less than their children)",
            layout,
            100 * parent_share,
            children.height,
            short,
            compared.height,
        )
        return layout

    def _is_filled(self, sample_size: int = 1000) -> bool:
        """True when `_detect_layout` reads the profile as filled."""
        return self._detect_layout(sample_size) == FILLED

    def _filter_by_sample(
        self, samples: Union[pl.LazyFrame, pl.DataFrame]
    ) -> "TaxonomicProfiles":
        """Create new TaxonomicProfiles filtered by sample IDs.

        Args:
            samples: LazyFrame containing sample IDs to keep

        Returns:
            New TaxonomicProfiles instance with filtered data
        """
        logger.debug("Filtering profiles")
        if isinstance(samples, pl.DataFrame):
            samples = samples.lazy()

        filtered_profiles = self.profiles.join(
            samples, on="sample", how="semi"
        )
        filtered_root = (
            self.root.join(samples, on="sample", how="semi")
            if self.root is not None
            else None
        )

        # Create new instance
        new_instance = TaxonomicProfiles.__new__(TaxonomicProfiles)
        new_instance.profiles = filtered_profiles
        new_instance.root = filtered_root
        new_instance.is_filled = self.is_filled

        return new_instance

    def _get_sample_list(self) -> Set[str]:
        """Extract sample IDs from this profiles instance.

        Returns:
            Set of sample IDs
        """
        return set(
            self.profiles.select("sample")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )

    def _get_rank(
        self, lf: pl.LazyFrame, rank: Union[str, TaxonomicRanks]
    ) -> pl.LazyFrame:
        """Internal method to get entries for a specific taxonomic rank."""
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        rank_regex = rank.get_regex()
        return lf.filter(pl.col("taxonomy").str.contains(rank_regex))

    def get_rank(self, rank: Union[str, TaxonomicRanks]) -> pl.LazyFrame:
        """Get entries for a specific taxonomic rank.

        Args:
            rank: Taxonomic rank to filter by

        Returns:
            LazyFrame with entries for the specified rank
        """
        return self._get_rank(self.profiles, rank)

    def filter_by_coverage(
        self, cov_cutoff: float = 50.0
    ) -> "TaxonomicProfiles":
        """Filter samples based on coverage metrics.

        Args:
            cov_cutoff: Minimum coverage cutoff (default: 50.0)

        Returns:
            Filtered TaxonomicProfiles instance
        """
        if self.root is None:
            # Check if profiles are in coverage format before trying to create root
            profiles_lf = self.profiles
            schema_names = profiles_lf.collect_schema().names()

            if "coverage" not in schema_names:
                raise ValueError("Root coverage data not available")

            try:
                root_lf = self._create_root_lf()
                self.root = root_lf
            except ValueError:
                raise ValueError("Root coverage data not available")

        # Get root data for filtering
        root_data = self.root
        if root_data is None:
            raise ValueError("Root coverage data not available")

        passing_samples = root_data.filter(
            pl.col("root_coverage") > cov_cutoff
        ).select("sample")

        return self._filter_by_sample(passing_samples)

    def get_samples(self) -> List[str]:
        """Get list of unique sample identifiers.

        Returns:
            List of unique sample IDs
        """
        samples: List[str] = (
            self.profiles.select("sample")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        return samples

    def filter_dominated_samples(
        self,
        dominated_cutoff: float = 0.99,
        rank: Union[TaxonomicRanks, str] = TaxonomicRanks.ORDER,
    ) -> "TaxonomicProfiles":
        """Filter out samples dominated by a single taxonomic group.

        Args:
            dominated_cutoff: Maximum relative abundance for single group (default: 0.99)
            rank: Taxonomic rank to check for domination

        Returns:
            New TaxonomicProfiles instance with filtered data
        """
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        # Determine abundance column
        if "coverage" in self.profiles.collect_schema().names():
            raise ValueError(
                "Profiles must be in relative abundance format to filter dominated samples."
            )

        # Find samples that are NOT dominated, from each taxon's total at
        # this rank. Reading only rows that *end* at the rank found nothing
        # in an unfilled profile, so it dropped every sample.
        non_dominated_samples = (
            self._sum_rank_rows(self._rank_long(self.profiles, rank))
            .group_by("sample")
            .agg(pl.col("relabund").max().alias("max_abund"))
            .filter(pl.col("max_abund") < dominated_cutoff)
            .select("sample")
        )

        # Use central filtering method
        return self._filter_by_sample(non_dominated_samples)

    def default_qc(
        self,
        cov_cutoff: float = 50.0,
        dominated_cutoff: float = 0.99,
        rank: Union[TaxonomicRanks, str] = TaxonomicRanks.ORDER,
    ) -> "TaxonomicProfiles":
        """
        Apply default quality control: coverage and domination filters.

        Args:
            cov_cutoff: Minimum coverage cutoff (default: 50.0)
            dominated_cutoff: Maximum relative abundance for single group (default: 0.99)
            rank: Taxonomic rank to check for domination
        """
        return self.filter_by_coverage(cov_cutoff).filter_dominated_samples(
            dominated_cutoff, rank
        )

    def _fill_profiles_lf(self) -> pl.LazyFrame:
        """An unfilled profile as a filled one: a row per taxon per rank.

        Each rank is built straight from the unfilled rows: a taxon's
        filled value is the sum of every row whose lineage names it, its
        own reads included. Building each rank from the one below instead
        lost every row that stopped above that rank, and filled again a
        profile that was already filled.

        Rows naming no domain (a bare `Root`) have no place in a filled
        table and are left out, as SingleM's relative abundance leaves
        them out of its total.
        """
        profiles_lf = self.profiles.with_columns(
            _tidy_taxonomy(pl.col("taxonomy")).alias("taxonomy")
        )
        value = (
            "relabund"
            if "relabund" in profiles_lf.collect_schema().names()
            else "coverage"
        )
        ranks = []
        for rank in TaxonomicRanks.iter_from_domain():
            lineage = rf"^((?:[^;]+;)*?{rank.prefix}[^;]+)(?:;|$)"
            ranks.append(
                profiles_lf.filter(
                    pl.col("taxonomy").str.contains(_names(rank))
                )
                .with_columns(pl.col("taxonomy").str.extract(lineage))
                .group_by(["sample", "taxonomy"], maintain_order=True)
                .agg(pl.col(value).sum())
            )
        return pl.concat(ranks)

    def _create_root_lf(self) -> pl.LazyFrame:
        """Per-sample total for coverage -> relative abundance.

        SingleM's denominator: the domain totals. Filled, that is the rows
        ending at a domain; unfilled, every row with a named domain, since
        each holds reads no other row does.
        """
        profiles_lf = self.profiles

        schema_names = profiles_lf.collect_schema().names()
        if "relabund" in schema_names:
            raise ValueError(
                "Cannot create root from relative abundance profiles."
            )

        domain = TaxonomicRanks.DOMAIN
        pattern = _ends_at(domain) if self.is_filled else _names(domain)
        return (
            profiles_lf.filter(
                _tidy_taxonomy(pl.col("taxonomy")).str.contains(pattern)
            )
            .group_by("sample")
            .agg(pl.col("coverage").sum().alias("root_coverage"))
            .select(["sample", "root_coverage"])
        )

    def _to_relabund_lf(self) -> pl.LazyFrame:
        """Convert coverage profiles to relative abundance.

        Returns LazyFrame with relabund column.
        """
        # Get profiles and root as LazyFrames
        profiles_lf = self.profiles
        root_lf = self.root

        if root_lf is None:
            raise ValueError(
                "Root coverage is required to convert to relative abundance."
            )

        schema_names = profiles_lf.collect_schema().names()
        if "relabund" in schema_names:
            return profiles_lf  # Already relative abundance

        # Join root coverage to profiles
        relabund = (
            profiles_lf.join(
                root_lf.select(["sample", "root_coverage"]), on="sample"
            )
            .with_columns(
                [
                    (pl.col("coverage") / pl.col("root_coverage")).alias(
                        "relabund"
                    )
                ]
            )
            .select(["sample", "taxonomy", "relabund"])
        )

        return relabund

    def fill_profiles(self) -> "TaxonomicProfiles":
        """Fill the taxonomic profiles so that each parent's abundance is the
        sum of its children.

        Returns a new TaxonomicProfiles instance with filled profiles.
        """
        if self.is_filled:
            return self  # Already filled

        filled_lf = self._fill_profiles_lf()

        # Create new instance
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.profiles = filled_lf
        new_instance.root = self.root
        new_instance.is_filled = True
        return new_instance

    def create_root(self) -> "TaxonomicProfiles":
        """Create the root dataframe if not provided.

        Only possible if profiles are in coverage (not relabund) format.
        Returns a new TaxonomicProfiles instance with root set.
        """
        if hasattr(self, "root") and self.root is not None:
            return self

        root_lf = self._create_root_lf()

        # Create new instance
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.profiles = self.profiles
        new_instance.root = root_lf
        new_instance.is_filled = self.is_filled
        return new_instance

    def create_features(
        self,
        rank: Union[str, TaxonomicRanks],
        stream_path: Optional[Union[str, Path]] = None,
        chunk_samples: int = 5000,
        samples: Optional[List[str]] = None,
        min_prevalence: Optional[float] = None,
        transform: str = "relabund",
        clr_pseudocount: float = 1e-6,
        presence_threshold: float = 0.0,
        prevalence_groups: Optional[pl.DataFrame] = None,
        icc_groups: Optional[pl.DataFrame] = None,
        max_icc: Optional[float] = None,
        icc_report: Optional[Union[str, Path]] = None,
    ) -> SampleFeatureSet:
        """Create a SampleFeatureSet from the taxonomic profiles at a rank.

        With `stream_path`, the wide table is built a chunk of samples at
        a time and appended to that CSV, and the returned feature set
        scans the file. Peak memory is then `chunk_samples x n_taxa`
        instead of the whole matrix: polars has no lazy pivot, so the
        eager path has to hold the long-format profile and the full wide
        table at once -- the dominant cost of `prepare` on large data.

        Args:
            rank: Taxonomic rank to extract features from
            stream_path: Write the wide table here, chunk by chunk, and
                return a scan of it. None keeps the in-memory behaviour.
            chunk_samples: Samples per chunk when streaming.
            samples: Restrict to these samples (e.g. the labelled ones).
            min_prevalence: Drop taxa present (non-zero) in fewer than
                this fraction of samples, e.g. 0.01 for 1%. None (the
                default) keeps every taxon.
            transform: ``"relabund"`` (default) leaves the values as
                relative abundance. ``"clr"`` applies the centred
                log-ratio per sample across the taxa kept at this rank,
                which puts the features on an additive scale instead of a
                constant-sum one. ``"presence"`` keeps only whether each
                taxon is there, dropping abundance entirely.
            prevalence_groups: ``{sample, group}``. When given, prevalence
                counts **groups** (e.g. how many bioprojects carry the
                taxon) rather than samples, so one large project cannot
                make a taxon look common on its own.
            clr_pseudocount: Added before the log, since a 99%-sparse
                table is mostly zeros and log(0) is undefined.
            presence_threshold: Relative abundance a taxon must exceed to
                count as present; only used for ``"presence"``.
            icc_groups: ``{sample, group}`` for the per-taxon intraclass
                correlation. Needed by ``max_icc`` and ``icc_report``.
            max_icc: Drop taxa whose ICC against that grouping exceeds
                this, i.e. taxa that are mostly a property of which study
                the sample came from. None (the default) computes nothing
                and drops nothing. Read the report before setting it --
                when the label is itself largely between-group, a
                predictive taxon also scores a high ICC.
            icc_report: Write the full ranked ICC table here as TSV.
                Works without ``max_icc``, which is the intended first
                use: look before filtering.
        Returns:
            SampleFeatureSet instance with features at the specified rank
        """
        if self.profiles is None:
            raise ValueError("No profiles available to create features.")

        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        # Get profiles as LazyFrame
        profiles_lf = self.profiles

        if "relabund" not in profiles_lf.collect_schema().names():
            raise ValueError(
                "Profiles must be in relative abundance format to create features. Use _to_relabund_lf() first."
            )

        # Long-format view: the rows that make up each taxon's total at
        # this rank, already cut down to the rank name. Unfilled rows are
        # summed per sample x taxon before each pivot. Filled ones are
        # collapsed here, so prevalence and ICC see one row per taxon
        # rather than the taxon plus its descendants; filled tables are
        # small, so the extra pass is cheap.
        long_lf = self._rank_long(profiles_lf, rank)
        if self.is_filled:
            long_lf = self._sum_rank_rows(long_lf)
        long_lf = long_lf.select(["sample", "taxonomy", "relabund"])
        if samples is not None:
            long_lf = long_lf.filter(pl.col("sample").is_in(list(samples)))

        mode = canonical_transform(transform)

        all_samples: Optional[List[str]] = None
        if min_prevalence:
            long_lf, all_samples = self._filter_by_prevalence(
                long_lf,
                rank,
                float(min_prevalence),
                groups=prevalence_groups,
            )

        if icc_groups is not None and (
            max_icc is not None or icc_report is not None
        ):
            long_lf, all_samples = self._apply_icc(
                long_lf,
                rank,
                icc_groups,
                max_icc,
                icc_report,
                all_samples,
            )

        if stream_path is None:
            # One row per sample x taxon before the pivot, so no
            # aggregate is needed and a duplicate would raise.
            features = self._sum_rank_rows(long_lf.collect()).pivot(
                values="relabund",
                index="sample",
                on="taxonomy",
                aggregate_function=None,
            )
            features = self._pad_samples(features, all_samples)
            # Same shape as the streamed path: a sample without a taxon
            # has zero of it, not an unknown amount, and Float32 is what
            # the models consume. Without this the two paths disagree.
            taxa_cols = [c for c in features.columns if c != "sample"]
            if taxa_cols:
                features = features.with_columns(
                    pl.col(taxa_cols).fill_null(0.0).cast(pl.Float32)
                )
                over, top = self._rank_sum_excess(features, taxa_cols)
                self._warn_rank_sums(rank, over, features.height, top)
                if mode == "clr":
                    features = self._apply_clr(
                        features, taxa_cols, float(clr_pseudocount)
                    )
                elif mode == "presence":
                    features = self._apply_presence(
                        features, taxa_cols, float(presence_threshold)
                    )
            return SampleFeatureSet.from_df(
                df=features, name=f"{rank.name.lower()}_features"
            )

        return self._stream_features(
            long_lf=long_lf,
            rank=rank,
            stream_path=Path(stream_path),
            chunk_samples=max(1, int(chunk_samples)),
            sample_ids=all_samples,
            transform=mode,
            clr_pseudocount=float(clr_pseudocount),
            presence_threshold=float(presence_threshold),
        )

    def _rank_long(
        self, lf: pl.LazyFrame, rank: TaxonomicRanks
    ) -> pl.LazyFrame:
        """Every row naming `rank`, with the lineage cut to that name.

        What makes up a taxon's total depends on the layout, and
        `_sum_rank_rows` applies it:

        - unfilled: all of these rows. Each holds reads no other row does,
          the taxon's own (assigned no deeper) included.
        - filled: the shallowest of them. That is the taxon's own row,
          which already includes the rest; where a lineage has no row of
          its own at this rank, its shallowest rows, which in a filled
          table are disjoint. Their depth travels in `_DEPTH`.

        This replaces pivot(aggregate_function="first"), which was right
        on filled profiles only because parents came first, and on
        species-only profiles kept the most abundant species alone.
        """
        token = rf"(?:^|;)({rank.prefix}[^;]+)(?:;|$)"
        tidy = _tidy_taxonomy(pl.col("taxonomy"))
        out = lf.with_columns(tidy.alias("taxonomy")).filter(
            pl.col("taxonomy").str.contains(_names(rank))
        )
        if self.is_filled:
            out = out.with_columns(
                pl.col("taxonomy").str.count_matches(";").alias(_DEPTH)
            )
        return out.with_columns(pl.col("taxonomy").str.extract(token))

    @staticmethod
    def _sum_rank_rows(frame: Any) -> Any:
        """One value per sample x taxon, so the pivot needs no aggregate.

        Sums every row, or only the shallowest where `_DEPTH` is present
        (a filled profile). Takes a DataFrame or a LazyFrame.
        """
        names = (
            frame.collect_schema().names()
            if isinstance(frame, pl.LazyFrame)
            else frame.columns
        )
        value = pl.col("relabund")
        if _DEPTH in names:
            depth = pl.col(_DEPTH)
            value = value.filter(depth == depth.min())
        return frame.group_by(["sample", "taxonomy"], maintain_order=True).agg(
            value.sum()
        )

    @staticmethod
    def _rank_sum_excess(
        wide: pl.DataFrame, taxa: List[str]
    ) -> Tuple[int, float]:
        """Samples whose relative abundances at one rank sum above 1.

        Returns that count and the largest per-sample sum.
        """
        if not taxa or not wide.height:
            return 0, 0.0
        totals = (
            wide.select(pl.sum_horizontal(taxa).alias("t"))
            .get_column("t")
            .to_list()
        )
        over = sum(1 for t in totals if t > 1.0 + _SUM_TOLERANCE)
        return over, float(max(totals))

    @staticmethod
    def _warn_rank_sums(
        rank: TaxonomicRanks, over: int, n_samples: int, top: float
    ) -> None:
        """Warn when a rank's relative abundances exceed 1 in any sample.

        A warning, not an error: hand-built tables in tests use values
        that are not proportions. On real profiles it means rows are being
        counted more than once, most likely a filled profile read as
        unfilled; give the layout explicitly.
        """
        if not over:
            return
        logger.warning(
            "%d of %d sample(s) have %s relative abundances summing above 1 "
            "(largest %.4g). Proportions cannot, so rows are probably being "
            "counted more than once: a filled profile read as unfilled? Set "
            "features.profile_layout (or layout=) to say which it is.",
            over,
            n_samples,
            rank.name.lower(),
            top,
        )

    @staticmethod
    def _apply_clr(
        wide: pl.DataFrame, taxa: List[str], pseudocount: float
    ) -> pl.DataFrame:
        """Centred log-ratio across the taxa of one sample.

        clr(x)_j = ln(x_j + p) - mean_k ln(x_k + p). Subtracting the mean
        of the logs is the same as dividing by the geometric mean, with
        no separate product to overflow. The pseudocount is unavoidable:
        the table is mostly zeros and log(0) is undefined, so the choice
        of p sets how far below the smallest real value a zero lands.

        Row-wise, so a streamed chunk can be transformed on its own --
        but only because each chunk already carries every taxon column.
        It is a *subcomposition* of whatever taxa survived the prevalence
        filter, which is the usual compromise.
        """
        if not taxa:
            return wide
        logged = wide.with_columns(
            pl.col(taxa).cast(pl.Float64).add(pseudocount).log()
        )
        logged = logged.with_columns(
            pl.mean_horizontal(taxa).alias("__clr_gm")
        )
        return logged.with_columns(
            (pl.col(taxa) - pl.col("__clr_gm")).cast(pl.Float32)
        ).drop("__clr_gm")

    @staticmethod
    def _apply_presence(
        wide: pl.DataFrame, taxa: List[str], threshold: float = 0.0
    ) -> pl.DataFrame:
        """Presence/absence: 1 where the taxon is detected, 0 where not.

        Throws away abundance and keeps membership. Sequencing depth and
        library protocol move abundances a great deal more than they move
        which taxa are detectable at all, so a binary table carries less
        of the between-study artefact that a leave-one-bioproject-out
        split asks the model to survive.

        `threshold` is a floor a taxon's relative abundance must exceed
        before it counts as present, for shedding barcode bleed and other
        near-zero detections. The default 0.0 means any non-zero value is
        presence, which is also how the prevalence filter counts, so the
        two agree unless the threshold is raised.

        Row-wise and independent of the other samples, so a streamed
        chunk can be transformed on its own -- unlike CLR, this does not
        even need the full column list.

        Stays Float32 rather than becoming boolean or UInt8: every other
        path here writes Float32, streamed chunks must share one schema,
        and the models take floats regardless.
        """
        if not taxa:
            return wide
        return wide.with_columns((pl.col(taxa) > threshold).cast(pl.Float32))

    @staticmethod
    def _pad_samples(
        wide: pl.DataFrame, all_samples: Optional[List[str]]
    ) -> pl.DataFrame:
        """Add back samples the prevalence filter emptied, as blank rows.

        The caller fills the nulls with 0.0, so a diagonal concat is all
        that is needed here.
        """
        if not all_samples:
            return wide
        present = set(wide.get_column("sample").to_list())
        absent = [s for s in all_samples if s not in present]
        if not absent:
            return wide
        return pl.concat(
            [wide, pl.DataFrame({"sample": absent})], how="diagonal"
        )

    def _apply_icc(
        self,
        long_lf: pl.LazyFrame,
        rank: TaxonomicRanks,
        groups: pl.DataFrame,
        max_icc: Optional[float],
        report: Optional[Union[str, Path]],
        all_samples: Optional[List[str]],
    ) -> Tuple[pl.LazyFrame, Optional[List[str]]]:
        """Rank taxa by ICC, write the table, optionally drop the top.

        Runs *after* the prevalence filter, so the table describes the
        taxa that will actually reach the model rather than the long tail
        the pivot was never going to include.

        Unsupervised, like the prevalence filter: it reads abundances and
        the grouping, never a label, so it cannot leak the target. What
        it can do is remove a taxon that predicts the label *through* the
        study it came from -- which is the point, and also the risk.
        """
        # A dropped taxon can empty a sample, exactly as the prevalence
        # filter can, so the pre-filter sample list has to exist before
        # anything is removed or _pad_samples has nothing to restore.
        if all_samples is None:
            all_samples = sorted(
                long_lf.select("sample")
                .unique()
                .collect()
                .get_column("sample")
                .to_list()
            )

        table = taxon_icc(long_lf, groups, samples=all_samples)
        if report is not None:
            path = Path(report)
            path.parent.mkdir(parents=True, exist_ok=True)
            table.write_csv(path, separator="	")
            logger.info("Wrote %d-row ICC table to %s", table.height, path)

        if max_icc is None:
            return long_lf, all_samples

        if not table.height:
            logger.warning(
                "features.icc.max_icc is set but the ICC table is empty; "
                "nothing dropped"
            )
            return long_lf, all_samples

        # A null ICC means the design could not support the estimate, not
        # that the taxon is clean, so keep it and say so rather than
        # dropping on a missing value.
        unknown = int(table.filter(pl.col("icc").is_null()).height)
        if unknown:
            logger.warning(
                "%d of %d taxa have no ICC estimate (too few groups or "
                "samples); they are kept",
                unknown,
                table.height,
            )
        drop = (
            table.filter(pl.col("icc") > float(max_icc))
            .get_column("taxonomy")
            .to_list()
        )
        logger.info(
            "ICC filter max_icc=%.3g: dropping %d of %d %s taxa",
            float(max_icc),
            len(drop),
            table.height,
            rank.name.lower(),
        )
        if not drop:
            return long_lf, all_samples
        if len(drop) == table.height:
            logger.warning(
                "features.icc.max_icc=%.3g drops every %s taxon; raise it",
                float(max_icc),
                rank.name.lower(),
            )
        return long_lf.filter(~pl.col("taxonomy").is_in(drop)), all_samples

    def _filter_by_prevalence(
        self,
        long_lf: pl.LazyFrame,
        rank: TaxonomicRanks,
        min_prevalence: float,
        groups: Optional[pl.DataFrame] = None,
    ) -> Tuple[pl.LazyFrame, List[str]]:
        """Keep taxa present in at least *min_prevalence* of samples.

        Returns the filtered frame and the sample list as it was *before*
        filtering, so no sample is lost just because all of its taxa were.

        Unsupervised -- it reads abundances only, never a label, so it
        cannot leak the target. Counted on the long frame, where a row
        exists only for a taxon that is present, so this is a group_by
        over sparse data rather than a pass over the wide matrix.
        """
        before = (
            long_lf.select("sample").unique().collect().get_column("sample")
        )
        all_samples = sorted(before.to_list())
        n_samples = len(all_samples)
        n_taxa = int(
            long_lf.select(pl.col("taxonomy").n_unique()).collect().item()
        )
        if not n_samples:
            return long_lf, all_samples

        # Counting samples lets one big project make a taxon look common.
        # Counting groups asks the more useful question: in how many
        # independent studies does this taxon turn up at all?
        present = long_lf.filter(pl.col("relabund") > 0)
        if groups is None:
            unit, count_col = "sample", "sample"
            n_units = n_samples
        else:
            gcol = next(c for c in groups.columns if c != "sample")
            present = present.join(
                groups.lazy().select(["sample", gcol]),
                on="sample",
                how="inner",
            )
            unit, count_col = gcol, gcol
            n_units = int(
                groups.filter(pl.col("sample").is_in(all_samples))
                .get_column(gcol)
                .n_unique()
            )
            if not n_units:
                logger.warning(
                    "Prevalence grouping '%s' matched no sample; "
                    "falling back to counting samples",
                    gcol,
                )
                present = long_lf.filter(pl.col("relabund") > 0)
                unit, count_col, n_units = "sample", "sample", n_samples

        need = max(1, int(min_prevalence * n_units))
        kept = (
            present.group_by("taxonomy")
            .agg(pl.col(count_col).n_unique().alias("n"))
            .filter(pl.col("n") >= need)
            .select("taxonomy")
            .collect()
            .get_column("taxonomy")
            .to_list()
        )
        logger.info(
            "Prevalence filter %.3g%%: keeping %d of %d %s taxa "
            "(present in >= %d of %d %ss)",
            100 * min_prevalence,
            len(kept),
            n_taxa,
            rank.name.lower(),
            need,
            n_units,
            unit,
        )
        if not kept:
            logger.warning(
                "Prevalence filter removed every %s taxon; lower "
                "features.min_prevalence",
                rank.name.lower(),
            )
        filtered = long_lf.filter(pl.col("taxonomy").is_in(kept))

        # A sample whose every taxon was rare now has no rows at all. It
        # keeps its place in the matrix as a row of zeros -- the filter is
        # about columns, and "carries no common taxon" is itself a fact
        # about that sample, which still has a label worth predicting.
        remaining = (
            filtered.select("sample")
            .unique()
            .collect()
            .get_column("sample")
            .to_list()
        )
        emptied = n_samples - len(remaining)
        if emptied:
            logger.warning(
                "Prevalence filter left %d of %d sample(s) with no %s "
                "taxon above the threshold; they are kept as all-zero "
                "rows",
                emptied,
                n_samples,
                rank.name.lower(),
            )
        return filtered, all_samples

    def _stream_features(
        self,
        long_lf: pl.LazyFrame,
        rank: TaxonomicRanks,
        stream_path: Path,
        chunk_samples: int,
        sample_ids: Optional[List[str]] = None,
        transform: str = "relabund",
        clr_pseudocount: float = 1e-6,
        presence_threshold: float = 0.0,
    ) -> SampleFeatureSet:
        """Pivot a rank to `stream_path` a chunk of samples at a time.

        Two cheap passes first -- the taxa that become columns, and the
        samples that become rows -- then one chunk of the wide matrix is
        alive at a time. Each chunk is reindexed onto the full column
        list so every appended block has the same schema.
        """
        taxa = (
            long_lf.select("taxonomy")
            .unique()
            .collect()
            .get_column("taxonomy")
            .sort()
            .to_list()
        )
        # The prevalence filter passes the pre-filter list, so a sample
        # whose every taxon was dropped still gets its row of zeros.
        if sample_ids is None:
            sample_ids = (
                long_lf.select("sample")
                .unique()
                .collect()
                .get_column("sample")
                .sort()
                .to_list()
            )
        else:
            sample_ids = sorted(sample_ids)
        if not taxa or not sample_ids:
            logger.warning(
                "No %s features found; nothing to stream", rank.name.lower()
            )
            return SampleFeatureSet.from_df(
                df=pl.DataFrame({"sample": []}),
                name=f"{rank.name.lower()}_features",
            )

        logger.info(
            "Streaming %s features: %d samples x %d taxa, "
            "chunks of %d -> %s",
            rank.name.lower(),
            len(sample_ids),
            len(taxa),
            chunk_samples,
            stream_path,
        )
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["sample", *taxa]
        # Parquet cannot be appended to in place without pyarrow, so each
        # chunk becomes one row-group file in a directory -- the standard
        # layout, read back by scan_parquet("<dir>/*.parquet").
        parquet = stream_path.suffix == ".parquet"
        handle = None
        if parquet:
            shutil.rmtree(stream_path, ignore_errors=True)
            stream_path.mkdir(parents=True, exist_ok=True)
        else:
            handle = stream_path.open("wb")
        over, top = 0, 0.0
        try:
            for part, start in enumerate(
                range(0, len(sample_ids), chunk_samples)
            ):
                batch = sample_ids[start : start + chunk_samples]
                # A chunk holds every row of its samples, so summing here
                # gives the same cells as summing the whole frame.
                sub = self._sum_rank_rows(
                    long_lf.filter(pl.col("sample").is_in(batch)).collect()
                )
                wide = (
                    sub.pivot(
                        values="relabund",
                        index="sample",
                        on="taxonomy",
                        aggregate_function=None,
                    )
                    if sub.height
                    else pl.DataFrame({"sample": batch})
                )
                wide = self._pad_samples(wide, batch)
                # A chunk only carries the taxa its samples have; pad the
                # rest so every block written shares one schema. 0.0 and
                # not null: absent means zero abundance, and a null would
                # have to be filled again on every later read.
                missing = [c for c in taxa if c not in wide.columns]
                if missing:
                    wide = wide.with_columns(
                        [
                            pl.lit(0.0, dtype=pl.Float32).alias(c)
                            for c in missing
                        ]
                    )
                wide = wide.select(columns).with_columns(
                    pl.col(taxa).fill_null(0.0).cast(pl.Float32)
                )
                n_over, chunk_top = self._rank_sum_excess(wide, taxa)
                over, top = over + n_over, max(top, chunk_top)
                if transform == "clr":
                    wide = self._apply_clr(wide, taxa, clr_pseudocount)
                elif transform == "presence":
                    wide = self._apply_presence(wide, taxa, presence_threshold)
                if parquet:
                    wide.write_parquet(
                        stream_path / f"part_{part:05d}.parquet"
                    )
                else:
                    wide.write_csv(handle, include_header=part == 0)
                del wide
        finally:
            if handle is not None:
                handle.close()

        # Once for the whole table, not once per chunk.
        self._warn_rank_sums(rank, over, len(sample_ids), top)
        return SampleFeatureSet.scan(
            stream_path, name=f"{rank.name.lower()}_features"
        )
