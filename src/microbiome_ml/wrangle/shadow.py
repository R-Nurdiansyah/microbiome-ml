"""Shadow features: a measured noise floor for the importance table.

Borrowed from Boruta, without Boruta's cost. A shadow column is a real
feature column with its values shuffled across samples: same marginal
distribution, same sparsity, same scale, but every relationship to the
label destroyed. Whatever importance it earns is what a tree hands out
for nothing.

This is the comparison the importance report currently cannot make. It
reports that a forest used 1,155 of 4,260 columns, but with no null to
read that against, "1,155 informative taxa" and "1,155 columns the tree
had to split on something" look identical. A handful of shadows settles
it: if the best shuffled column scores as high as your 200th real
taxon, everything below rank 200 is noise.

What this is NOT: it is not Boruta. Boruta shadows *every* feature and
repeats ~100 forest fits with a binomial test per feature. This adds a
sample of shadows to one ordinary run, which costs almost nothing and
calibrates the ranking, but gives no per-feature p-value.

One caveat belongs with any number read off this, and the report repeats
it. With `n` shadows against `m` real columns, the largest shadow
importance is the max of `n` draws from the null, not of `m`. For n < m
that maximum is too small, so counting real features above it
over-counts. Prefer a high quantile of the shadow distribution, and
raise `n` when the answer is close.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Marks a column as a shuffled copy. Real taxa carry a rank prefix
# (d__, p__, ... s__), so this cannot collide with one.
SHADOW_PREFIX = "shadow__"

# Appended to a feature set built with shadows, beside _clr / _pa / _bio.
SHADOW_SUFFIX = "shad"


def is_shadow(name: Any) -> bool:
    """True for a column this module created."""
    return str(name).startswith(SHADOW_PREFIX)


def split_shadows(names: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Partition column names into (real, shadow)."""
    real = [n for n in names if not is_shadow(n)]
    shade = [n for n in names if is_shadow(n)]
    return real, shade


def add_shadow_columns(
    features: pl.LazyFrame,
    acc_col: str,
    feature_names: Sequence[str],
    n_shadow: int,
    seed: int = 0,
) -> Tuple[pl.LazyFrame, List[Tuple[str, str]]]:
    """Append `n_shadow` shuffled copies of randomly chosen columns.

    Args:
        features: The wide feature frame, `{acc_col, <features...>}`.
        acc_col: The sample column, carried through untouched.
        feature_names: Columns eligible to be copied.
        n_shadow: How many shadows to add; clamped to the column count.
        seed: Fixes both which columns are copied and how each is
            shuffled, so a rerun reproduces the same null.

    Returns:
        `(augmented frame, [(shadow name, source column), ...])`.

    Sources are drawn uniformly from `feature_names` rather than from
    the abundant end, so the shadows inherit the marginal distribution
    of the feature pool as a whole -- mostly sparse, occasionally not,
    which is what the real columns look like.

    Only the chosen columns are read. `select` pushes down into the scan
    when the features came from disk, so this touches `n_shadow` columns
    of the table, not all of it, and the returned frame stays lazy.
    """
    names = list(feature_names)
    if not names:
        logger.warning("No feature columns to shadow; frame unchanged")
        return features, []

    collisions = [n for n in names if is_shadow(n)]
    if collisions:
        raise ValueError(
            f"{len(collisions)} existing column(s) already start with "
            f"'{SHADOW_PREFIX}' (e.g. {collisions[0]!r}); shadows would "
            "be indistinguishable from real features"
        )

    k = min(int(n_shadow), len(names))
    if k < int(n_shadow):
        logger.warning(
            "Asked for %d shadows but only %d feature column(s) exist; "
            "adding %d",
            int(n_shadow),
            len(names),
            k,
        )
    if k <= 0:
        return features, []

    rng = np.random.default_rng(seed)
    chosen = [names[i] for i in rng.choice(len(names), size=k, replace=False)]

    sub = features.select([acc_col, *chosen]).collect()
    columns = {acc_col: sub.get_column(acc_col)}
    mapping: List[Tuple[str, str]] = []
    for i, source in enumerate(chosen):
        name = f"{SHADOW_PREFIX}{i:04d}"
        # Shuffled across samples: the column keeps its own values and
        # dtype exactly, and loses only which sample each belonged to.
        values = sub.get_column(source).to_numpy()
        columns[name] = pl.Series(name, rng.permutation(values))
        mapping.append((name, source))

    logger.info(
        "Added %d shadow column(s) to %d real one(s), seed %d",
        k,
        len(names),
        seed,
    )
    # Joined on the sample key, so row order is irrelevant: everything
    # downstream (CV's feature/label join included) addresses samples by
    # name, never by position.
    return (
        features.join(pl.DataFrame(columns).lazy(), on=acc_col, how="left"),
        mapping,
    )


def shadow_floor(
    importances: pl.DataFrame,
    feature_col: str = "feature",
    value_col: str = "importance",
    quantiles: Optional[Sequence[float]] = None,
) -> Optional[pl.DataFrame]:
    """Summarise one combination's importances against its shadows.

    Returns None when the table carries no shadow column, so a caller can
    skip the section rather than print an empty one.
    """
    qs = list(quantiles) if quantiles else [0.5, 0.95, 1.0]
    shade = importances.filter(
        pl.col(feature_col).str.starts_with(SHADOW_PREFIX)
    )
    if not shade.height:
        return None
    real = importances.filter(
        ~pl.col(feature_col).str.starts_with(SHADOW_PREFIX)
    )
    values = shade.get_column(value_col)
    row: dict = {
        "n_real": real.height,
        "n_shadow": shade.height,
        "shadow_total_share": (
            float(values.sum() / importances.get_column(value_col).sum())
            if importances.get_column(value_col).sum()
            else None
        ),
    }
    for q in qs:
        cut = float(values.quantile(q) or 0.0)
        tag = "max" if q >= 1.0 else f"q{int(round(q * 100)):02d}"
        row[f"shadow_{tag}"] = cut
        row[f"n_real_above_{tag}"] = int(
            real.filter(pl.col(value_col) > cut).height
        )
    return pl.DataFrame([row])
