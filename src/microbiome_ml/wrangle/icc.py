"""Per-taxon intraclass correlation: how much of a taxon is its study?

ICC answers, for one taxon, "what fraction of its variance across
samples is explained by knowing which group (bioproject) the sample came
from". A taxon at ICC ~ 1 is flat within every study and differs between
them: knowing the study tells you the abundance. That is the signature of
a protocol artefact -- primer bias, extraction kit, depth, a lab's
contaminant -- and a tree that splits on it is learning which study it is
looking at, which earns nothing under a leave-one-bioproject-out split.
A taxon at ICC ~ 0 moves as much inside a study as across studies, so
whatever drives it is something other than batch.

Read as a diagnostic before it is used as a filter. When the label is
itself mostly between-group -- which it is here; every label in this
dataset carries 57-80% of its variance between bioprojects -- any taxon
that genuinely predicts the label *must* also vary between groups and so
must also score a high ICC. Variance structure alone cannot separate
batch from signal under that confound, which is why `max_icc` defaults to
off and the ranked table is the point.

Computed from the long, sparse frame, so absences are implicit: a taxon
missing from a sample is a zero in the variance, not a gap. That matters
a great deal -- a taxon in 2% of samples is 98% zeros, and those zeros
are most of its variance.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)

# Columns of the table taxon_icc returns, in order.
ICC_COLUMNS = (
    "taxonomy",
    "n_present",
    "prevalence",
    "n_groups_present",
    "mean",
    "var_between",
    "var_within",
    "icc",
    "icc_raw",
)


def _design(
    groups: pl.DataFrame,
    samples: Optional[List[str]],
) -> Tuple[str, pl.DataFrame, pl.DataFrame, int, int, float]:
    """The group sizes and the constants every taxon shares.

    N, k and n0 depend only on how samples are distributed across groups,
    not on any taxon, so they are computed once. n0 is the effective
    group size of the one-way random effects model; it reduces to the
    common group size when the groups are balanced.
    """
    gcol = next(c for c in groups.columns if c != "sample")
    design = groups.select(["sample", gcol]).drop_nulls()
    if samples is not None:
        design = design.filter(pl.col("sample").is_in(list(samples)))
    design = design.unique(subset=["sample"])

    sizes = design.group_by(gcol).agg(pl.len().alias("n_in_group"))
    n_total = int(design.height)
    k = int(sizes.height)
    if k < 2 or n_total <= k:
        # One group cannot have a between-group variance, and with every
        # group a singleton there is no within-group variance to compare
        # it against -- the ratio would read 1.0 for every taxon.
        return gcol, design, sizes, n_total, k, float("nan")

    sum_sq_sizes = int(sizes.select((pl.col("n_in_group") ** 2).sum()).item())
    n0 = (n_total - sum_sq_sizes / n_total) / (k - 1)
    return gcol, design, sizes, n_total, k, n0


def taxon_icc(
    long_lf: pl.LazyFrame,
    groups: pl.DataFrame,
    samples: Optional[List[str]] = None,
    value_col: str = "relabund",
) -> pl.DataFrame:
    """Per-taxon ICC against the grouping column in `groups`.

    Args:
        long_lf: `{sample, taxonomy, <value_col>}`, with rows only for taxa
            actually present. Absences are inferred from `groups`, so the
            frame does not need the zeros spelled out. Several rows for one
            taxon in one sample are summed.
        groups: `{sample, <group>}` -- the second column is the grouping.
        samples: Restrict to these samples. None uses every sample that
            `groups` covers.
        value_col: The abundance column of `long_lf`.

    Returns:
        One row per taxon, columns `ICC_COLUMNS`, sorted by `icc`
        descending so the study markers are at the top. `icc` is the
        variance-components estimate and `icc_raw` the plain
        sum-of-squares ratio; both are null when the design cannot
        support the estimate.

    The algebra, for one taxon, where group i has n_i samples in total
    (present or not) and S_i is the taxon's summed abundance in group i:

        A   = sum_i S_i^2 / n_i
        SSB = A - S^2 / N          (between groups)
        SSW = Q - A                (within groups)

    with S the overall sum and Q the overall sum of squares. Only present
    rows contribute to S_i, S and Q, so this is a group_by over the
    sparse frame -- the implicit zeros are carried by n_i and N alone,
    never materialised.
    """
    gcol, design, sizes, n_total, k, n0 = _design(groups, samples)
    if not n_total:
        logger.warning("ICC: no samples matched the grouping; empty table")
        return pl.DataFrame(schema={c: pl.Float64 for c in ICC_COLUMNS})

    # An inner join against the design both attaches the group and
    # drops any sample outside it, which is what `samples` asked for.
    #
    # Then one value per sample x taxon, because the algebra above needs
    # it: the frame can hold one row per lineage under a taxon, and squaring
    # those separately makes Q too small. SSW then clamps to zero and the
    # taxon reads ICC 1.0, with a prevalence above 1.
    present = (
        long_lf.filter(pl.col(value_col) != 0)
        .join(design.lazy(), on="sample", how="inner")
        .group_by(["sample", "taxonomy", gcol])
        .agg(pl.col(value_col).sum())
    )

    per_group = present.group_by(["taxonomy", gcol]).agg(
        pl.col(value_col).sum().alias("gsum"),
        (pl.col(value_col) ** 2).sum().alias("gsumsq"),
        pl.len().alias("n_in_group_present"),
    )
    per_taxon = (
        per_group.join(sizes.lazy(), on=gcol, how="inner")
        .group_by("taxonomy")
        .agg(
            (pl.col("gsum") ** 2 / pl.col("n_in_group")).sum().alias("A"),
            pl.col("gsum").sum().alias("S"),
            pl.col("gsumsq").sum().alias("Q"),
            pl.col(gcol).n_unique().alias("n_groups_present"),
            pl.col("n_in_group_present").sum().alias("n_present"),
        )
        .collect()
    )
    if not per_taxon.height:
        # Either every value really is zero, or -- far more likely -- the
        # grouping does not cover the samples in the profiles, so the
        # inner join emptied the frame.
        logger.warning(
            "ICC: nothing left after joining the profiles to the "
            "grouping; check that '%s' covers the samples in the "
            "profiles (%d sample(s) matched)",
            gcol,
            n_total,
        )
        return pl.DataFrame(schema={c: pl.Float64 for c in ICC_COLUMNS})

    # Float error can push a difference of two near-equal sums slightly
    # negative; a variance is not negative, so clamp rather than let a
    # -1e-18 propagate into a nonsense ratio.
    ss = per_taxon.with_columns(
        pl.max_horizontal(
            pl.col("A") - pl.col("S") ** 2 / n_total, pl.lit(0.0)
        ).alias("ssb"),
        pl.max_horizontal(pl.col("Q") - pl.col("A"), pl.lit(0.0)).alias("ssw"),
    )

    if k < 2 or n_total <= k or math.isnan(n0):
        logger.warning(
            "ICC needs at least 2 groups and more samples than groups; "
            "got %d sample(s) in %d group(s) of '%s'. Reporting the raw "
            "sum-of-squares ratio only.",
            n_total,
            k,
            gcol,
        )
        icc_expr = pl.lit(None, dtype=pl.Float64)
        between = pl.lit(None, dtype=pl.Float64)
        within = pl.lit(None, dtype=pl.Float64)
    else:
        msb = pl.col("ssb") / (k - 1)
        msw = pl.col("ssw") / (n_total - k)
        var_a = pl.max_horizontal((msb - msw) / n0, pl.lit(0.0))
        between, within = var_a, msw
        icc_expr = (
            pl.when(var_a + msw > 0)
            .then(var_a / (var_a + msw))
            .otherwise(None)
        )

    total_ss = pl.col("ssb") + pl.col("ssw")
    out = (
        ss.with_columns(
            (pl.col("n_present") / n_total).alias("prevalence"),
            (pl.col("S") / n_total).alias("mean"),
            between.alias("var_between"),
            within.alias("var_within"),
            icc_expr.alias("icc"),
            pl.when(total_ss > 0)
            .then(pl.col("ssb") / total_ss)
            .otherwise(None)
            .alias("icc_raw"),
        )
        .select(list(ICC_COLUMNS))
        .sort("icc", descending=True, nulls_last=True)
    )
    logger.info(
        "ICC over %d sample(s) in %d '%s' group(s): %d taxa, "
        "median %.3f, %d at >= 0.9",
        n_total,
        k,
        gcol,
        out.height,
        out.get_column("icc").median() or float("nan"),
        int(out.filter(pl.col("icc") >= 0.9).height),
    )
    return out
