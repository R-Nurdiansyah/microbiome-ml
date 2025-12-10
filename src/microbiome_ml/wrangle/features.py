"""Feature set management for machine learning workflows."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Union

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

from microbiome_ml.utils.taxonomy import TaxonomicRanks

# standardised error messages
ERR_FEATURESET_NAME_UNDEFINED = "FeatureSet name must be defined"


class FeatureSet(ABC):
    """Abstract base class for feature sets.

    Provides common interface for both species-wise and sample-wise features.
    """

    def __new__(
        cls,
        accessions: List[str],
        feature_names: List[str],
        features: Any = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "FeatureSet":
        """Create appropriate concrete FeatureSet subclass for backward
        compatibility."""
        if cls is FeatureSet:
            # Validate name before passing to subclass
            if name is None:
                raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)

            # Direct instantiation - create SampleFeatureSet for backward compatibility
            return SampleFeatureSet(
                accessions, feature_names, features, name, **kwargs
            )
        return super().__new__(cls)

    def __init__(
        self,
        accessions: List[str],
        feature_names: List[str],
        name: str,
    ):
        """Initialize FeatureSet with common attributes.

        Args:
            accessions: Ordered sample/species IDs
            feature_names: Ordered feature names
            name: Name for the FeatureSet

        Raises:
            ValueError: If name is missing
        """
        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)

        self.accessions = accessions
        self.feature_names = feature_names
        self.name = name

        # Cache indices for O(1) lookups
        self._accession_idx = {acc: i for i, acc in enumerate(accessions)}
        self._feature_idx = {fname: i for i, fname in enumerate(feature_names)}

    @property
    @abstractmethod
    def features(self) -> Any:
        """Feature data structure (implementation-specific)."""

    @abstractmethod
    def filter_samples(self, sample_ids: List[str]) -> "FeatureSet":
        """Filter to subset of samples/species."""

    @abstractmethod
    def collect(self) -> pl.DataFrame:
        """Collect data to DataFrame."""

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "FeatureSet":
        """Load features from file (creates SampleFeatureSet for backward
        compatibility)."""
        return SampleFeatureSet.load(path, **kwargs)

    @classmethod
    def scan(
        cls, path: Union[str, Path], name: str, **kwargs: Any
    ) -> "FeatureSet":
        """Scan features from file (creates SampleFeatureSet for backward
        compatibility)."""
        return SampleFeatureSet.scan(path, name=name, **kwargs)

    @classmethod
    def from_df(
        cls, df: pl.DataFrame, name: str, **kwargs: Any
    ) -> "FeatureSet":
        """Create from DataFrame (creates SampleFeatureSet for backward
        compatibility)."""
        return SampleFeatureSet.from_df(df, name=name, **kwargs)

    def _get_sample_list(self) -> Set[str]:
        """Extract sample IDs from this feature set.

        Returns:
            Set of sample IDs (accessions)
        """
        return set(self.accessions)

    def get_sample_accs(self) -> List[str]:
        """Get list of all sample/accession IDs."""
        return self.accessions.copy()

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names.copy()

    def to_df(self) -> pl.DataFrame:
        """Convert FeatureSet back to wide-form DataFrame."""
        return self.collect()

    def save(self, path: Union[Path, str]) -> None:
        """Save FeatureSet to disk as a .csv file."""
        path = Path(path)
        self.collect().write_csv(path)


class SampleFeatureSet(FeatureSet):
    """Sample-wise features stored as LazyFrame for memory efficiency.

    Used for traditional ML where rows=samples and columns=features.
    """

    def __init__(
        self,
        accessions: List[str],
        feature_names: List[str],
        features: Any,
        name: str,
    ):
        """Initialize SampleFeatureSet with LazyFrame backend.

        Args:
            accessions: Ordered sample IDs
            feature_names: Ordered feature names
            features: LazyFrame, DataFrame, or numpy array
            name: Name for the FeatureSet

        Raises:
            ValueError: If dimensions don't match
        """
        super().__init__(accessions, feature_names, name)

        if isinstance(features, pl.LazyFrame):
            self._features = features
        elif isinstance(features, pl.DataFrame):
            self._features = features.lazy()
        elif isinstance(features, np.ndarray):
            # Convert numpy array to LazyFrame
            if features.shape[0] != len(accessions):
                raise ValueError(
                    f"Features array rows ({features.shape[0]}) must match accessions length ({len(accessions)})"
                )
            if features.shape[1] != len(feature_names):
                raise ValueError(
                    f"Features array cols ({features.shape[1]}) must match feature_names length ({len(feature_names)})"
                )

            df = pl.DataFrame(features, schema=feature_names)
            df = df.with_columns(pl.Series("sample", accessions)).select(
                ["sample"] + feature_names
            )
            self._features = df.lazy()
        else:
            raise TypeError(f"Unsupported type for features: {type(features)}")

    @property
    def features(self) -> pl.LazyFrame:
        """Feature LazyFrame."""
        return self._features

    def filter_samples(self, sample_ids: List[str]) -> "SampleFeatureSet":
        """Filter to subset of samples.

        Args:
            sample_ids: List of sample IDs to keep

        Returns:
            New SampleFeatureSet with filtered data
        """
        # Filter accessions and features
        filtered_accessions = [
            acc for acc in self.accessions if acc in sample_ids
        ]

        # Filter LazyFrame
        sample_filter = pl.DataFrame({"sample": sample_ids}).lazy()
        filtered_features = self._features.join(
            sample_filter, on="sample", how="semi"
        )

        return SampleFeatureSet(
            accessions=filtered_accessions,
            feature_names=self.feature_names.copy(),
            features=filtered_features,
            name=self.name,
        )

    def collect(self) -> pl.DataFrame:
        """Collect the internal LazyFrame to a DataFrame.

        Returns:
            DataFrame containing features and accessions
        """
        return self._features.collect()

    @classmethod
    def scan(
        cls,
        path: Union[str, Path],
        name: str,
        **kwargs: Any,
    ) -> "SampleFeatureSet":
        """Lazily load sample features from a file.

        Args:
            path: Path to CSV file
            name: Name for the FeatureSet
            **kwargs: Additional arguments including acc_column

        Returns:
            SampleFeatureSet instance (lazy-loaded)
        """
        acc_column = kwargs.get("acc_column")
        lf = pl.scan_csv(path)
        schema = lf.collect_schema()

        # Auto-detect accession column
        if acc_column is None:
            if "sample" in schema.names():
                acc_column = "sample"
            elif "acc" in schema.names():
                acc_column = "acc"
            else:
                raise ValueError(
                    "No accession column found. Expected 'sample' or 'acc' column, or specify acc_column"
                )

        # Extract accessions (requires collecting just that column)
        accessions = lf.select(acc_column).collect().to_series().to_list()
        feature_names = [col for col in schema.names() if col != acc_column]

        return cls(
            accessions=accessions,
            feature_names=feature_names,
            features=lf,
            name=name,
        )

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        name: str,
        **kwargs: Any,
    ) -> "SampleFeatureSet":
        """Create SampleFeatureSet from wide-form DataFrame.

        Args:
            df: Wide-form DataFrame with features as columns
            name: Name for the FeatureSet
            **kwargs: Additional arguments including acc_column

        Returns:
            SampleFeatureSet instance
        """
        acc_column = kwargs.get("acc_column")

        # Auto-detect accession column
        if acc_column is None:
            if "sample" in df.columns:
                acc_column = "sample"
            elif "acc" in df.columns:
                acc_column = "acc"
            else:
                raise ValueError(
                    "No accession column found. Expected 'sample' or 'acc' column, or specify acc_column"
                )

        if acc_column not in df.columns:
            raise ValueError(
                f"Specified acc_column '{acc_column}' not found in DataFrame columns"
            )

        # Extract components
        accessions = df.select(acc_column).to_series().to_list()
        feature_names = [col for col in df.columns if col != acc_column]

        return cls(
            accessions=accessions,
            feature_names=feature_names,
            features=df,
            name=name,
        )

    @classmethod
    def from_profiles(
        cls,
        profiles: "TaxonomicProfiles",
        sample_ids: List[str],
        name: str,
        rank: "TaxonomicRanks",
    ) -> "SampleFeatureSet":
        """Create SampleFeatureSet from TaxonomicProfiles at specified rank.

        Args:
            profiles: TaxonomicProfiles instance
            sample_ids: List of sample IDs to include
            rank: Taxonomic rank to extract features from
            name: Name for the FeatureSet

        Returns:
            SampleFeatureSet instance with features at the specified rank
        """
        # Convert rank to enum if string
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        # Filter profiles to specified samples
        sample_filter = pl.DataFrame({"sample": sample_ids}).lazy()
        filtered_profiles = profiles._filter_by_sample(sample_filter)

        # Use the TaxonomicProfiles create_features method
        feature_set = filtered_profiles.create_features(rank)

        # Set name if provided
        feature_set.name = name
        return feature_set

    @classmethod
    def from_lf(cls, lf: pl.LazyFrame, name: str) -> "SampleFeatureSet":
        """Create SampleFeatureSet from a LazyFrame.

        Args:
            lf: Polars LazyFrame with features
            name: Name for the FeatureSet
        Returns:
            SampleFeatureSet instance
        """
        schema = lf.collect_schema()

        # Auto-detect accession column
        if "sample" in schema.names():
            acc_column = "sample"
        elif "acc" in schema.names():
            acc_column = "acc"
        else:
            raise ValueError(
                "No accession column found. Expected 'sample' or 'acc' column"
            )

        accessions = lf.select(acc_column).collect().to_series().to_list()
        feature_names = [col for col in schema.names() if col != acc_column]

        return cls(
            accessions=accessions,
            feature_names=feature_names,
            features=lf,
            name=name,
        )

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "SampleFeatureSet":
        """Load SampleFeatureSet from a .csv file.

        Args:
            path: Path to the .csv file to load from
            **kwargs: Additional arguments including name
        Returns:
            Loaded SampleFeatureSet instance
        """
        name = kwargs.get("name")
        path = Path(path)
        if name is None:
            name = path.stem
        return cls.scan(path, name=name)

    def get_samples(self, sample_ids: List[str]) -> np.ndarray:
        """Get features for specific samples.

        Args:
            sample_ids: List of sample IDs to retrieve

        Returns:
            numpy array of shape (len(sample_ids), n_features)
        """
        # Filter and collect
        df = self._features.filter(
            pl.col("sample").is_in(sample_ids)
        ).collect()

        # Check if all samples found
        found_samples = set(df["sample"].to_list())
        missing = set(sample_ids) - found_samples
        if missing:
            raise ValueError(f"Sample IDs {missing} not found in FeatureSet")

        # Reorder to match input sample_ids
        order_df = pl.DataFrame(
            {"sample": sample_ids, "order": range(len(sample_ids))}
        )
        df = df.join(order_df, on="sample").sort("order").drop("order")

        return df.drop("sample").to_numpy()

    def get_features(self, feature_names: List[str]) -> np.ndarray:
        """Get features for specific feature names.

        Args:
            feature_names: List of feature names to retrieve

        Returns:
            numpy array of shape (n_samples, len(feature_names))
        """
        # Check if features exist
        schema_names = self._features.collect_schema().names()
        missing = set(feature_names) - set(schema_names)
        if missing:
            raise ValueError(
                f"Feature names {missing} not found in FeatureSet"
            )

        df = self._features.select(["sample"] + feature_names).collect()
        return df.drop("sample").to_numpy()


class SpeciesFeatureSet(FeatureSet):
    """Species-wise features with numpy backend for external feature data.

    Used for species-level ML where rows=taxa and columns=features.
    """

    def __init__(
        self,
        accessions: List[str],  # taxonomy strings
        feature_names: List[str],
        features: np.ndarray,
        name: str,
    ):
        """Initialize SpeciesFeatureSet with numpy backend.

        Args:
            accessions: Ordered taxonomy strings
            feature_names: Ordered feature names
            features: 2D numpy array with shape (n_taxa, n_features)
            name: Name for the FeatureSet

        Raises:
            ValueError: If dimensions don't match
        """
        super().__init__(accessions, feature_names, name)

        if features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array, got {features.ndim}D"
            )
        if features.shape[0] != len(accessions):
            raise ValueError(
                f"Features array rows ({features.shape[0]}) must match accessions length ({len(accessions)})"
            )
        if features.shape[1] != len(feature_names):
            raise ValueError(
                f"Features array cols ({features.shape[1]}) must match feature_names length ({len(feature_names)})"
            )

        self._features = features

    @property
    def features(self) -> np.ndarray:
        """Feature numpy array."""
        return self._features

    def filter_samples(self, taxonomy_ids: List[str]) -> "SpeciesFeatureSet":
        """Filter to subset of taxonomy IDs.

        Args:
            taxonomy_ids: List of taxonomy strings to keep

        Returns:
            New SpeciesFeatureSet with filtered data
        """
        # Find indices to keep
        indices = [
            i for i, acc in enumerate(self.accessions) if acc in taxonomy_ids
        ]

        # Filter data
        filtered_accessions = [self.accessions[i] for i in indices]
        filtered_features = self._features[indices, :]

        return SpeciesFeatureSet(
            accessions=filtered_accessions,
            feature_names=self.feature_names.copy(),
            features=filtered_features,
            name=self.name,
        )

    def collect(self) -> pl.DataFrame:
        """Convert to DataFrame format.

        Returns:
            DataFrame with taxonomy column and feature columns
        """
        df = pl.DataFrame(self._features, schema=self.feature_names)
        return df.with_columns(pl.Series("taxonomy", self.accessions)).select(
            ["taxonomy"] + self.feature_names
        )

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        name: str,
        **kwargs: Any,
    ) -> "SpeciesFeatureSet":
        """Create SpeciesFeatureSet from wide-form DataFrame.

        Args:
            df: Wide-form DataFrame with features as columns
            name: Name for the FeatureSet
            **kwargs: Additional arguments including acc_column

        Returns:
            SpeciesFeatureSet instance
        """
        acc_column = kwargs.get("acc_column", "taxonomy")
        if acc_column not in df.columns:
            raise ValueError(
                f"Specified acc_column '{acc_column}' not found in DataFrame columns"
            )

        # Extract components
        accessions = df.select(acc_column).to_series().to_list()
        feature_names = [col for col in df.columns if col != acc_column]

        # Convert to numpy array
        feature_array = df.drop(acc_column).to_numpy()

        return cls(
            accessions=accessions,
            feature_names=feature_names,
            features=feature_array,
            name=name,
        )

    @classmethod
    def scan(
        cls,
        path: Union[str, Path],
        name: str,
        **kwargs: Any,
    ) -> "SpeciesFeatureSet":
        """Load SpeciesFeatureSet from CSV file.

        Args:
            path: Path to CSV file
            name: Name for the FeatureSet
            **kwargs: Additional arguments including acc_column

        Returns:
            SpeciesFeatureSet instance
        """
        acc_column = kwargs.get("acc_column", "taxonomy")
        df = pl.read_csv(path)
        return cls.from_df(df, name, acc_column=acc_column)

    @classmethod
    def load(
        cls, path: Union[str, Path], **kwargs: Any
    ) -> "SpeciesFeatureSet":
        """Load SpeciesFeatureSet from a .csv file.

        Args:
            path: Path to the .csv file to load from
            **kwargs: Additional arguments including name

        Returns:
            Loaded SpeciesFeatureSet instance
        """
        name = kwargs.get("name")
        path = Path(path)
        if name is None:
            name = path.stem
        return cls.scan(path, name=name)

    def get_features(self, feature_names: List[str]) -> np.ndarray:
        """Get features for specific feature names.

        Args:
            feature_names: List of feature names to retrieve

        Returns:
            numpy array of shape (n_taxa, len(feature_names))
        """
        # Check if features exist
        missing = set(feature_names) - set(self.feature_names)
        if missing:
            raise ValueError(
                f"Feature names {missing} not found in FeatureSet"
            )

        indices = [self.feature_names.index(name) for name in feature_names]
        return self._features[:, indices]

    def get_samples(self, taxonomy_ids: List[str]) -> np.ndarray:
        """Get features for specific taxonomy IDs.

        Args:
            taxonomy_ids: List of taxonomy strings to retrieve

        Returns:
            numpy array of shape (len(taxonomy_ids), n_features)
        """
        # Find indices of taxonomy IDs
        indices = []
        missing = []

        for tax_id in taxonomy_ids:
            try:
                indices.append(self.accessions.index(tax_id))
            except ValueError:
                missing.append(tax_id)

        if missing:
            raise ValueError(f"Taxonomy IDs {missing} not found in FeatureSet")

        return self._features[indices, :]
