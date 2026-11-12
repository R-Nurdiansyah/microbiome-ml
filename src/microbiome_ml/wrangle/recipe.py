"""Feature recipes: many feature sets from one prepared base table.

`prepare` reads the raw profile table, which on these datasets takes hours,
and bakes one feature set into the prepared dataset. A recipe instead
starts from a base table that one prepare built -- relative abundance at a
rank, nothing filtered (`tax_order`, say) -- and builds a feature set from
it in each CV job, in seconds at order level:

    prevalence  keep taxa present in at least `min_prevalence` of the
                samples, or of the groups with `prevalence_by`
                (bioproject: in how many studies does it turn up at all?)
    icc         drop taxa whose ICC against `icc.group` is above
                `icc.max_icc`. With `icc.rare_below`, only taxa present in
                fewer than that fraction of samples can be dropped: a rare
                taxon at a high ICC sits in a few studies' samples and
                marks the study, while a common one at a high ICC may be
                the signal itself (see wrangle/icc.py)
    variance    drop taxa whose relative-abundance variance is at or below
                `variance_threshold`, as sklearn's VarianceThreshold does
                (it keeps variance > threshold, population variance)
    transform   relabund, clr, presence or log (transforms/features.py),
                after the selection, so clr is taken over the taxa kept

in that order. The selection is fitted on the job's training pool -- the
samples its holdout split put in `train` -- so the test samples never
decide which taxa the model sees; a prepare-time filter counts them too.
The CV folds inside the pool share the one selection. It reads no label,
so what they share is which taxa are common, not anything about the
target.

Variance is measured on relative abundance, before the transform, so the
relabund and clr arms of a sweep get the same taxa and differ only in the
transform. Relative abundance is a fraction, so the threshold is small: a
taxon present in a fraction q of samples at a typical abundance a has a
variance of about q * a^2 * (1 - q); 1e-6 is roughly a taxon in 20% of
samples at 0.25%.

Config, under `features`:

    base_set: tax_order          # the prepared table to start from
    transform: clr
    min_prevalence: 0.2
    prevalence_by: bioproject
    icc: {enabled: true, group: bioproject, max_icc: 0.75, rare_below: 0.05}
    variance_threshold: 1.0e-6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

import polars as pl

from microbiome_ml.transforms import features as feature_transforms
from microbiome_ml.wrangle.features import SampleFeatureSet
from microbiome_ml.wrangle.icc import taxon_icc

logger = logging.getLogger(__name__)

# Columns of the per-taxon table `select` returns.
REPORT_COLUMNS = (
    "taxonomy",
    "prevalence",
    "n_units",
    "icc",
    "sample_prevalence",
    "variance",
    "dropped_by",
)


def _fraction(value: Any, name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} is a fraction from 0 to 1, got {number:g}")
    return number


@dataclass(frozen=True)
class Recipe:
    """The selection and transform that turn a base table into a set."""

    transform: str = "relabund"
    min_prevalence: Optional[float] = None
    prevalence_by: Optional[str] = None
    max_icc: Optional[float] = None
    icc_group: Optional[str] = None
    icc_rare_below: Optional[float] = None
    icc_report: bool = False
    variance_threshold: Optional[float] = None
    clr_pseudocount: float = 1e-6
    presence_threshold: float = 0.0

    @classmethod
    def from_features(cls, features: Mapping[str, Any]) -> "Recipe":
        """From a config's `features` section; prepare reads the same keys."""
        icc = features.get("icc") or {}
        icc_on = bool(icc.get("enabled", False))
        variance = features.get("variance_threshold")
        recipe = cls(
            transform=feature_transforms.canonical_transform(
                features.get("transform", "relabund")
            ),
            min_prevalence=_fraction(
                features.get("min_prevalence") or None,
                "features.min_prevalence",
            ),
            prevalence_by=features.get("prevalence_by") or None,
            max_icc=(
                _fraction(icc.get("max_icc"), "features.icc.max_icc")
                if icc_on
                else None
            ),
            icc_group=(icc.get("group") or "bioproject") if icc_on else None,
            icc_rare_below=(
                _fraction(icc.get("rare_below"), "features.icc.rare_below")
                if icc_on
                else None
            ),
            icc_report=icc_on and bool(icc.get("report", True)),
            variance_threshold=None if variance is None else float(variance),
            clr_pseudocount=float(features.get("clr_pseudocount", 1e-6)),
            presence_threshold=float(features.get("presence_threshold", 0.0)),
        )
        threshold = recipe.variance_threshold
        if threshold is not None and threshold < 0:
            raise ValueError(
                "features.variance_threshold cannot be negative, got "
                f"{threshold:g}"
            )
        return recipe

    @property
    def groupings_needed(self) -> List[str]:
        names = [self.prevalence_by, self.icc_group]
        return list(dict.fromkeys(n for n in names if n))


def _prevalence(
    fit: pl.DataFrame,
    taxa: List[str],
    groups: Optional[pl.DataFrame],
) -> Tuple[Dict[str, int], int]:
    """Per taxon, how many samples (or groups) it is present in, of how many.

    By group, a sample without a group counts towards none, and the
    denominator is the number of groups among the training samples.
    """
    if groups is None:
        n_units = fit.height
        counts = fit.select([(pl.col(t) > 0).sum().alias(t) for t in taxa])
    else:
        gcol = next(c for c in groups.columns if c != "sample")
        joined = fit.join(
            groups.select(["sample", gcol]).drop_nulls(),
            on="sample",
            how="inner",
        )
        n_units = int(joined.get_column(gcol).n_unique())
        counts = joined.select(
            [
                pl.col(gcol).filter(pl.col(t) > 0).n_unique().alias(t)
                for t in taxa
            ]
        )
    row = counts.row(0, named=True) if taxa else {}
    return {t: int(row[t] or 0) for t in taxa}, n_units


def select(
    fit: pl.DataFrame,
    recipe: Recipe,
    groupings: Optional[pl.DataFrame] = None,
) -> Tuple[List[str], pl.DataFrame]:
    """The taxa *recipe* keeps, judged on *fit*, and why each other went.

    *fit* is the training pool's rows of the base table: `sample` plus
    one relative-abundance column per taxon. Returns the kept taxa in the
    base table's order and one row per taxon (REPORT_COLUMNS), where
    `dropped_by` is `kept` or the step that removed it.
    """
    taxa = [c for c in fit.columns if c != "sample"]
    fit = fit.with_columns(pl.col(taxa).fill_null(0.0).cast(pl.Float64))
    dropped: Dict[str, str] = {}
    stats: Dict[str, Dict[str, Any]] = {t: {} for t in taxa}

    def groups_for(column: str) -> pl.DataFrame:
        if groupings is None or column not in groupings.columns:
            raise ValueError(
                f"grouping {column!r} is not in the dataset's groupings; "
                "add it under groupings.columns"
            )
        return groupings.select(["sample", column])

    # 1. prevalence, by sample or by group
    if recipe.min_prevalence is not None:
        groups = (
            groups_for(recipe.prevalence_by) if recipe.prevalence_by else None
        )
        counts, n_units = _prevalence(fit, taxa, groups)
        # The same rule as prepare's filter (wrangle/profiles.py).
        need = max(1, int(recipe.min_prevalence * n_units))
        for t in taxa:
            stats[t]["prevalence"] = counts[t] / n_units if n_units else 0.0
            stats[t]["n_units"] = n_units
            if counts[t] < need:
                dropped[t] = "prevalence"
        logger.info(
            "prevalence >= %g of %d %s: keeping %d of %d taxa",
            recipe.min_prevalence,
            n_units,
            recipe.prevalence_by or "sample",
            len(taxa) - len(dropped),
            len(taxa),
        )

    # 2. ICC against a grouping, optionally only for rare taxa
    if recipe.icc_group and (recipe.max_icc is not None or recipe.icc_report):
        alive = [t for t in taxa if t not in dropped]
        long = (
            fit.select(["sample", *alive])
            .unpivot(
                index="sample",
                on=alive,
                variable_name="taxonomy",
                value_name="relabund",
            )
            .filter(pl.col("relabund") > 0)
        )
        table = taxon_icc(
            long.lazy(),
            groups_for(recipe.icc_group),
            samples=fit.get_column("sample").to_list(),
        )
        for row in table.iter_rows(named=True):
            entry = stats.get(row["taxonomy"])
            if entry is not None:
                entry["icc"] = row["icc"]
                entry["sample_prevalence"] = row["prevalence"]
        if recipe.max_icc is not None:
            n_before = len(dropped)
            for row in table.iter_rows(named=True):
                icc, name = row["icc"], row["taxonomy"]
                # A null ICC is "could not tell", not "clean": kept.
                if icc is None or icc <= recipe.max_icc:
                    continue
                if (
                    recipe.icc_rare_below is not None
                    and row["prevalence"] >= recipe.icc_rare_below
                ):
                    continue
                dropped.setdefault(name, "icc")
            logger.info(
                "ICC > %g against %s%s: dropping %d of %d taxa",
                recipe.max_icc,
                recipe.icc_group,
                (
                    ""
                    if recipe.icc_rare_below is None
                    else f" among taxa in < {recipe.icc_rare_below:g} "
                    "of samples"
                ),
                len(dropped) - n_before,
                len(alive),
            )

    # 3. variance of the relative abundance
    variances = fit.select([pl.col(t).var(ddof=0).alias(t) for t in taxa])
    var_row = variances.row(0, named=True) if taxa else {}
    for t in taxa:
        stats[t]["variance"] = var_row[t]
    if recipe.variance_threshold is not None:
        alive = [t for t in taxa if t not in dropped]
        gone = [
            t
            for t in alive
            if not (var_row[t] or 0.0) > recipe.variance_threshold
        ]
        for t in gone:
            dropped[t] = "variance"
        logger.info(
            "variance > %g: dropping %d of %d taxa",
            recipe.variance_threshold,
            len(gone),
            len(alive),
        )

    kept = [t for t in taxa if t not in dropped]
    report = pl.DataFrame(
        {
            "taxonomy": taxa,
            "prevalence": [stats[t].get("prevalence") for t in taxa],
            "n_units": [stats[t].get("n_units") for t in taxa],
            "icc": [stats[t].get("icc") for t in taxa],
            "sample_prevalence": [
                stats[t].get("sample_prevalence") for t in taxa
            ],
            "variance": [stats[t].get("variance") for t in taxa],
            "dropped_by": [dropped.get(t, "kept") for t in taxa],
        },
        schema={
            "taxonomy": pl.Utf8,
            "prevalence": pl.Float64,
            "n_units": pl.Int64,
            "icc": pl.Float64,
            "sample_prevalence": pl.Float64,
            "variance": pl.Float64,
            "dropped_by": pl.Utf8,
        },
    )
    return kept, report


def build(
    base: pl.DataFrame,
    fit_samples: Collection[str],
    recipe: Recipe,
    groupings: Optional[pl.DataFrame] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """The recipe's feature table for every sample of *base*, and its report.

    Selection is judged on the *fit_samples* rows only; the table then
    covers every sample, so the holdout's test rows get the same columns
    and the same row-wise transform.
    """
    if "sample" not in base.columns:
        raise ValueError("the base feature table has no `sample` column")
    wanted = set(fit_samples)
    fit = base.filter(pl.col("sample").is_in(list(wanted)))
    if not fit.height:
        raise ValueError(
            "none of the training samples are in the base feature table"
        )
    kept, report = select(fit, recipe, groupings)
    if not kept:
        raise ValueError(
            "the feature recipe dropped every taxon; loosen min_prevalence, "
            "icc.max_icc or variance_threshold"
        )
    table = base.select(["sample", *kept]).with_columns(
        pl.col(kept).fill_null(0.0).cast(pl.Float32)
    )
    table = feature_transforms.apply(
        table,
        kept,
        recipe.transform,
        clr_pseudocount=recipe.clr_pseudocount,
        presence_threshold=recipe.presence_threshold,
    )
    return table, report


def apply_to_dataset(
    dataset: Any,
    features_cfg: Mapping[str, Any],
    fit_samples: Collection[str],
    name: str,
    report_path: Optional[Path] = None,
) -> Any:
    """Add the recipe's feature set to *dataset* as *name*, in memory.

    `features.base_set` names the prepared table to start from. Nothing is
    written to the prepared dataset, which every job of a sweep shares;
    the per-taxon report goes to *report_path* if given.
    """
    base_name = features_cfg.get("base_set")
    if not base_name:
        raise ValueError("features.base_set is not set")
    if base_name not in dataset.feature_sets:
        raise ValueError(
            f"features.base_set={base_name!r} is not in the prepared "
            f"dataset; it has {sorted(dataset.feature_sets)}"
        )
    recipe = Recipe.from_features(features_cfg)
    base_set = dataset.feature_sets[base_name]
    frame = getattr(base_set, "features", None)
    if isinstance(frame, pl.LazyFrame):
        base = frame.collect()
    else:
        base = base_set.to_df()
    table, report = build(base, fit_samples, recipe, dataset.groupings)
    logger.info(
        "feature recipe %s from %s: %d of %d taxa, transform %s",
        name,
        base_name,
        table.width - 1,
        base.width - 1,
        recipe.transform,
    )
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.write_csv(report_path, separator="\t")
    # Straight into the dict: add_feature_set would re-sync the sample
    # list, which re-scans the profiles, and the samples are the base's.
    dataset.feature_sets = {
        **{k: v for k, v in dataset.feature_sets.items() if k != base_name},
        name: SampleFeatureSet.from_df(table, name=name),
    }
    return dataset
