"""Feature transforms: how a taxon table's values are expressed.

Applied once, when `prepare` builds a feature set. Every transform here
works on one sample's row and fits nothing, so applying it before the
split leaks nothing between training and test samples.

    relabund  the relative abundance as profiled (a fraction, 0 to 1)
    clr       centred log-ratio across a sample's taxa; compositional,
              and dense (zeros become negative numbers)
    presence  1 where a taxon is detected above a threshold, else 0
    log       log1p(relabund * 1e6): log of 1 + parts per million

Tree models (random forest, XGBoost) only ever ask "is this value above
a threshold", so a transform that keeps each taxon's order of samples
leaves them the same candidate splits, scored on the same label values:
on the training samples they split the same way. The cut itself sits
midway between two neighbouring values on the new scale, so an unseen
sample falling between them can go the other way -- small differences,
not a different model. So `log` and a plain rescale change little for
them. `clr` does change them (it mixes taxa
within a sample) and so does `presence` (it discards abundance). `log`
is for models that use distances and magnitudes -- the SVR, and any
linear model -- where relative abundances spanning five orders of
magnitude put all the weight on the few dominant taxa.

Why 1e6 and not log1p(relabund): non-zero relative abundances here are
mostly between 1e-5 and 1e-2 (16S marine genus: median 3.9e-4), where
log1p(x) is x to within a fraction of a percent -- no change at all.
Scaled to parts per million first, the same values land between about
2.8 and 9.2 and are spread out. Zero stays zero, so a sparse table stays
sparse.

Adding a transform means: an entry in TRANSFORMS (its config spellings),
one in SUFFIX (its feature-set name suffix, which must be unique), and a
branch in `apply`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import polars as pl

# canonical name -> what a config may say instead
TRANSFORMS: Dict[str, Tuple[str, ...]] = {
    "relabund": ("relabund",),
    "clr": ("clr",),
    "presence": ("presence", "pa", "binary", "presence_absence"),
    "log": ("log", "log1p", "log_ppm", "logppm"),
}

# canonical name -> what it adds to the feature set name after the rank:
# tax_genus, tax_genus_clr, tax_genus_pa, tax_genus_log
SUFFIX: Dict[str, str] = {
    "relabund": "",
    "clr": "_clr",
    "presence": "_pa",
    "log": "_log",
}

# `log` multiplies by this before log1p: relabund -> parts per million.
LOG_SCALE = 1e6


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


def apply(
    wide: pl.DataFrame,
    taxa: List[str],
    transform: str,
    clr_pseudocount: float = 1e-6,
    presence_threshold: float = 0.0,
) -> pl.DataFrame:
    """*wide* (one row per sample, one column per taxon) transformed.

    Row-wise, so the streamed build can transform each chunk on its own.
    `relabund` returns the table as it is.
    """
    mode = canonical_transform(transform)
    if not taxa or mode == "relabund":
        return wide
    if mode == "clr":
        return clr(wide, taxa, float(clr_pseudocount))
    if mode == "presence":
        return presence(wide, taxa, float(presence_threshold))
    if mode == "log":
        return log_ppm(wide, taxa)
    raise AssertionError(f"no branch for transform {mode!r}")


def clr(wide: pl.DataFrame, taxa: List[str], pseudocount: float) -> pl.DataFrame:
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
    logged = logged.with_columns(pl.mean_horizontal(taxa).alias("__clr_gm"))
    return logged.with_columns(
        (pl.col(taxa) - pl.col("__clr_gm")).cast(pl.Float32)
    ).drop("__clr_gm")


def presence(
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

    Stays Float32 rather than becoming boolean or UInt8: every other
    path here writes Float32, streamed chunks must share one schema,
    and the models take floats regardless.
    """
    if not taxa:
        return wide
    return wide.with_columns((pl.col(taxa) > threshold).cast(pl.Float32))


def log_ppm(wide: pl.DataFrame, taxa: List[str]) -> pl.DataFrame:
    """log1p(relabund * 1e6), Float32; zero stays zero (see module doc)."""
    if not taxa:
        return wide
    return wide.with_columns(
        (pl.col(taxa).cast(pl.Float64) * LOG_SCALE).log1p().cast(pl.Float32)
    )
