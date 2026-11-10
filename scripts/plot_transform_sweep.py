#!/usr/bin/env python3
"""Plot a transform sweep: one figure per label, three questions each.

Reads the tables `scripts/sweep_transforms.py` writes under
`<out>/summary/` and draws one figure per label, because the labels do
not share a y axis in any useful way -- temperature sits near 0.74 while
oxygen sits near 0.11, and putting them in one panel hides the only
comparison that matters inside each.

Each figure answers, left to right:

  (a) Which CV scheme?  Pooled R2 for every scheme, full range. This is
      drawn first and at full scale because it is usually the largest
      effect in the table by an order of magnitude -- a scheme that
      leaves the same studies on both sides of a fold scores well, and
      one that does not can go deeply negative. No transform rescues
      that, so reading (b) without (a) is how a sweep gets misread.

  (b) Which transform?  The same scores zoomed to the best-scoring
      scheme, with each transform's clean set beside its `_shad` set.

  (c) How stable?  Per-fold R2 at that scheme, one dot per fold, from
      `cross_val_scores`. A pooled score of 0.74 built from folds
      spanning 0.07 to 0.92 is a different claim from one built from
      twelve folds at 0.74.

On (b), clean against `_shad`: a shadow set holds every real column
*plus* the shuffled ones, so its score is expected to match the clean
set's. Equal bars mean the 200 noise columns did not derail the model,
which is a sanity check passing -- not evidence that the real columns
are worthless, and not a horse race either way. The question "did the
model lean on noise" is answered by importances, in
`<combo>/feature_report/shadow_floor.tsv`, not by these scores.

    pixi run python scripts/plot_transform_sweep.py \
        --summary ML_outputs/transform_sweep/summary \
        --out ML_outputs/transform_sweep/summary/figures --formats png svg

Scores here are validation folds, not the holdout, so a transform
winning is a candidate to take to the holdout, not a result. The caption
repeats that, because a figure outlives its README.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from microbiome_ml.visualise.visualisations import (  # noqa: E402
    _CATEGORICAL_HUES,
    Visualiser,
    _fmt_r2,
)
from microbiome_ml.wrangle.shadow import SHADOW_SUFFIX  # noqa: E402

LOGGER = logging.getLogger("plot-transform-sweep")

SCORE = "pooled_validation_r2"
# Identifies one trained combination. The collate step globs run
# directories at two depths, so the same row can arrive twice; these are
# the columns a duplicate shares.
CELL = ["combo", "transform", "feature_set", "label", "scheme", "model"]

# Hatch for the shadow-augmented bar, so the pair reads in greyscale too.
SHADOW_HATCH = "///"


def _shadow_set(name: Any) -> bool:
    return str(name).endswith("_" + SHADOW_SUFFIX)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_models(path: Path) -> pl.DataFrame:
    """`all_models.tsv`, deduplicated and with the score made numeric."""
    frame = pl.read_csv(path, separator="\t", infer_schema_length=10000)
    missing = [c for c in (*CELL, SCORE) if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {missing}; expected the table "
            "written by scripts/sweep_transforms.py"
        )
    before = frame.height
    frame = frame.unique(subset=CELL, keep="first")
    if frame.height != before:
        # Not a warning: collate globs `holdout_*/**/results_summary.csv`,
        # which matches a run's table at more than one depth. The copies
        # carry identical scores, but counting them twice would weight
        # these panels by how deep a directory happened to be.
        LOGGER.info(
            "dropped %d duplicate row(s) of %d from %s",
            before - frame.height,
            before,
            path.name,
        )
    return frame.with_columns(
        pl.col(SCORE).cast(pl.Float64, strict=False)
    ).filter(pl.col(SCORE).is_not_null() & pl.col(SCORE).is_finite())


def best_per_cell(frame: pl.DataFrame) -> pl.DataFrame:
    """Best model per transform x scheme x feature set.

    The model is a nuisance parameter here: the sweep asks about
    transforms, so each cell keeps whichever of rf/xgboost did better and
    carries its name along for the annotation.
    """
    return (
        frame.sort(SCORE, descending=True, nulls_last=True)
        .group_by(
            ["label", "transform", "scheme", "feature_set"],
            maintain_order=True,
        )
        .first()
    )


def scheme_order(block: pl.DataFrame) -> List[str]:
    """Schemes best-scoring first, so the collapse reads left to right."""
    ranked = (
        block.filter(~pl.col("feature_set").str.ends_with("_" + SHADOW_SUFFIX))
        .group_by("scheme")
        .agg(pl.col(SCORE).median().alias("mid"))
        .sort("mid", descending=True, nulls_last=True)
    )
    return [str(s) for s in ranked.get_column("scheme").to_list()]


def transform_order(block: pl.DataFrame) -> List[str]:
    """A fixed, meaningful order rather than whatever the file holds."""
    known = ["relabund", "clr", "presence"]
    present = set(str(t) for t in block.get_column("transform").unique())
    return [t for t in known if t in present] + sorted(present - set(known))


def hue_map(transforms: List[str]) -> Dict[str, str]:
    """One colour per transform, the same in every figure."""
    return {
        t: _CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)]
        for i, t in enumerate(transforms)
    }


def _score(
    block: pl.DataFrame, transform: str, scheme: str, shadow: bool
) -> Optional[Dict[str, Any]]:
    """One cell, or None when the sweep has no row for it."""
    rows = [
        r
        for r in block.iter_rows(named=True)
        if r["transform"] == transform
        and r["scheme"] == scheme
        and _shadow_set(r["feature_set"]) is shadow
    ]
    if not rows:
        return None
    return max(rows, key=lambda r: r[SCORE])


def _pad_limits(ax: plt.Axes, top: float = 0.1, bottom: float = 0.1) -> None:
    """Add proportional breathing room, keeping zero on the axis.

    Annotations and bars are drawn before this, so the padding is taken
    from the data range matplotlib already settled on.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    if span <= 0:
        return
    ax.set_ylim(lo - span * bottom, hi + span * top)


# ---------------------------------------------------------------------------
# (a) which scheme
# ---------------------------------------------------------------------------
def panel_schemes(
    ax: plt.Axes,
    block: pl.DataFrame,
    schemes: List[str],
    transforms: List[str],
    hues: Dict[str, str],
) -> None:
    """Pooled R2 per scheme x transform, clean sets, full range."""
    width = 0.8 / max(1, len(transforms))
    for i, transform in enumerate(transforms):
        xs, ys = [], []
        for j, scheme in enumerate(schemes):
            cell = _score(block, transform, scheme, shadow=False)
            if cell is None:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(cell[SCORE])
        ax.bar(
            xs,
            ys,
            width=width * 0.9,
            color=hues[transform],
            label=transform,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(
        [s.replace("_logo", "\nLOGO") for s in schemes], fontsize=8
    )
    ax.set_ylabel("pooled validation $R^2$", fontsize=9)
    ax.set_title("(a) which CV scheme — full range", fontsize=9, loc="left")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85, title=None)
    # An R2 below zero means "worse than predicting the mean"; saying so
    # on the axis saves the reader working out why a bar points down.
    bottom = ax.get_ylim()[0]
    if bottom < -0.05:
        ax.text(
            0.98,
            0.03,
            "below 0 = worse than predicting the mean",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            color="#666666",
        )


# ---------------------------------------------------------------------------
# (b) which transform
# ---------------------------------------------------------------------------
def panel_transforms(
    ax: plt.Axes,
    block: pl.DataFrame,
    scheme: str,
    transforms: List[str],
    hues: Dict[str, str],
) -> None:
    """Clean beside `_shad` for each transform, at one scheme."""
    width = 0.36
    annotated = False
    for i, transform in enumerate(transforms):
        for shadow, offset in ((False, -width / 2), (True, width / 2)):
            cell = _score(block, transform, scheme, shadow=shadow)
            if cell is None:
                continue
            ax.bar(
                [i + offset],
                [cell[SCORE]],
                width=width * 0.92,
                color=hues[transform],
                alpha=1.0 if not shadow else 0.55,
                hatch=SHADOW_HATCH if shadow else None,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.annotate(
                _fmt_r2(cell[SCORE]),
                (i + offset, cell[SCORE]),
                textcoords="offset points",
                xytext=(0, 3 if cell[SCORE] >= 0 else -11),
                ha="center",
                fontsize=7,
            )
            if not shadow:
                # Which model won is a footnote, not the comparison, so
                # it sits at the foot of the panel. Pinned to the axes
                # floor rather than to data y=0, which is partway up the
                # panel once the bars are all positive -- anchored there
                # the text was clipped by the spine.
                ax.annotate(
                    str(cell["model"])
                    .replace("Regressor", "")
                    .replace("Classifier", ""),
                    (i + offset, 0.0),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=6,
                    color="#666666",
                )
                annotated = True

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(range(len(transforms)))
    ax.set_xticklabels(transforms, fontsize=8)
    ax.set_ylabel("pooled validation $R^2$", fontsize=9)
    ax.set_title(
        f"(b) which transform — {scheme} only", fontsize=9, loc="left"
    )
    ax.grid(True, axis="y", alpha=0.18)
    # Headroom for the value labels: without it the tallest annotation
    # sits on the axis spine, and a legend placed here would land on the
    # bars. The keys live in one figure-level legend instead.
    _pad_limits(ax, top=0.18, bottom=0.08)
    if annotated:
        ax.tick_params(axis="x", pad=14)


# ---------------------------------------------------------------------------
# (c) how stable
# ---------------------------------------------------------------------------
def _folds(cell: Optional[Dict[str, Any]]) -> List[float]:
    """Per-fold R2 from `cross_val_scores`, or [] when unusable."""
    if cell is None:
        return []
    raw = cell.get("cross_val_scores")
    if raw in (None, ""):
        return []
    try:
        values = json.loads(str(raw))
    except (TypeError, ValueError):
        LOGGER.debug("could not parse cross_val_scores: %r", raw)
        return []
    if not isinstance(values, list):
        return []
    return [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]


def panel_folds(
    ax: plt.Axes,
    block: pl.DataFrame,
    scheme: str,
    transforms: List[str],
    hues: Dict[str, str],
) -> bool:
    """One dot per fold, clean sets, at one scheme. False if no data."""
    import numpy as np

    drew = False
    rng = np.random.default_rng(0)  # fixed jitter, so reruns match
    for i, transform in enumerate(transforms):
        cell = _score(block, transform, scheme, shadow=False)
        folds = _folds(cell)
        if not folds:
            continue
        drew = True
        jitter = rng.uniform(-0.12, 0.12, size=len(folds))
        ax.scatter(
            [i + j for j in jitter],
            folds,
            s=22,
            color=hues[transform],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        mid = float(np.median(folds))
        ax.plot(
            [i - 0.26, i + 0.26],
            [mid, mid],
            color=hues[transform],
            linewidth=2.0,
            zorder=4,
        )
        # The pooled score is what panels (a) and (b) rank on, so it is
        # marked here to show where it sits among the folds it came from.
        ax.plot(
            [i - 0.26, i + 0.26],
            [cell[SCORE]] * 2,
            color="#444444",
            linestyle=":",
            linewidth=1.2,
            zorder=4,
        )

    if not drew:
        return False
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(range(len(transforms)))
    ax.set_xticklabels(transforms, fontsize=8)
    ax.set_xlim(-0.5, len(transforms) - 0.5)
    ax.set_ylabel("per-fold validation $R^2$", fontsize=9)
    ax.set_title(f"(c) how stable — {scheme}", fontsize=9, loc="left")
    ax.grid(True, axis="y", alpha=0.18)
    _pad_limits(ax, top=0.12, bottom=0.12)
    return True


# ---------------------------------------------------------------------------
# One label's figure
# ---------------------------------------------------------------------------
def draw_label(
    label: str,
    block: pl.DataFrame,
    n_label: Optional[int],
    scheme: Optional[str] = None,
) -> Optional[plt.Figure]:
    """The three-panel figure for one label, or None if nothing to draw.

    Panels (b) and (c) zoom to `scheme`, or to the best-scoring scheme
    when it is None. Panel (a) always shows every scheme.
    """
    schemes = scheme_order(block)
    transforms = transform_order(block)
    if not schemes or not transforms:
        LOGGER.warning("%s: no scheme/transform to plot", label)
        return None
    hues = hue_map(transform_order(block))
    shown = scheme or schemes[0]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    panel_schemes(axes[0], block, schemes, transforms, hues)
    panel_transforms(axes[1], block, shown, transforms, hues)
    if not panel_folds(axes[2], block, shown, transforms, hues):
        axes[2].set_visible(False)
        LOGGER.info("%s: no cross_val_scores; panel (c) omitted", label)

    # One legend for the whole figure: the clean/shadow and
    # median/pooled keys describe the figure, not a single panel, and
    # inside an axes they landed on the bars they were explaining.
    fig.legend(
        handles=[
            Patch(facecolor="#888888", label="clean set"),
            Patch(
                facecolor="#888888",
                alpha=0.55,
                hatch=SHADOW_HATCH,
                label=f"+ shadows (_{SHADOW_SUFFIX})",
            ),
            plt.Line2D(
                [],
                [],
                color="#444444",
                linewidth=2.0,
                label="median fold (c)",
            ),
            plt.Line2D(
                [],
                [],
                color="#444444",
                linestyle=":",
                linewidth=1.2,
                label="pooled score (c)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    n = f", n={n_label:,}" if n_label else ""
    focus = f"scheme: {scheme}" if scheme else f"best scheme: {shown}"
    fig.suptitle(
        f"Transform sweep — {label}{n}  ({focus})",
        y=0.995,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.005,
        "Validation folds, not the holdout: a transform winning here is "
        "a candidate, not a result. A `_shad` set holds every real column "
        "PLUS shuffled ones, so matching bars in (b) mean the noise "
        "columns did not derail the model — read importances for whether "
        "it leaned on them.",
        ha="center",
        fontsize=7.5,
        color="#444444",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    return fig


def label_sizes(headline: Optional[Path]) -> Dict[str, int]:
    """`n_label` per label from headline.tsv, for the titles."""
    if headline is None or not headline.exists():
        return {}
    try:
        frame = pl.read_csv(
            headline, separator="\t", infer_schema_length=10000
        )
    except Exception as exc:
        LOGGER.warning("could not read %s: %s", headline, exc)
        return {}
    if "n_label" not in frame.columns or "label" not in frame.columns:
        return {}
    sizes: Dict[str, int] = {}
    for row in frame.iter_rows(named=True):
        value = row.get("n_label")
        if value is None:
            continue
        try:
            sizes[str(row["label"])] = int(float(value))
        except (TypeError, ValueError):
            continue
    return sizes


def plot_sweep(
    models: pl.DataFrame,
    out_dir: Path,
    formats: List[str],
    headline: Optional[Path] = None,
    labels: Optional[List[str]] = None,
    scheme: Optional[str] = None,
) -> List[Path]:
    """One figure per label. Returns the stems written.

    With `scheme`, panels (b) and (c) show that scheme instead of the
    best-scoring one, the file name ends in `__<scheme>`, and a label that
    never ran the scheme is skipped.
    """
    best = best_per_cell(models)
    sizes = label_sizes(headline)
    vis = Visualiser(out=out_dir, formats=formats)
    written: List[Path] = []

    wanted = labels or [
        str(v) for v in best.get_column("label").unique().sort().to_list()
    ]
    for label in wanted:
        block = best.filter(pl.col("label") == label)
        if not block.height:
            LOGGER.warning("no rows for label '%s'; skipped", label)
            continue
        if scheme and block.filter(pl.col("scheme") == scheme).is_empty():
            LOGGER.warning(
                "label '%s' has no scheme '%s'; skipped", label, scheme
            )
            continue
        fig = draw_label(label, block, sizes.get(label), scheme=scheme)
        if fig is None:
            continue
        suffix = f"__{scheme}" if scheme else ""
        target = vis.output_dir / f"transform_sweep_{label}{suffix}"
        vis._save_fig(fig, target)
        plt.close(fig)
        LOGGER.info("wrote %s", target)
        written.append(target)
    return written


def load_holdout(path: Path) -> Optional[pl.DataFrame]:
    """`summary/holdout.tsv`, or None when the final test never ran.

    Absent is the normal state for a CV-only sweep, so this is not an
    error: the caller simply draws no final-test figure.
    """
    if not path.exists():
        return None
    try:
        frame = pl.read_csv(path, separator="\t", infer_schema_length=10000)
    except Exception as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return None
    needed = {"combo", "label", "r2"}
    if not frame.height or not needed <= set(frame.columns):
        LOGGER.warning(
            "%s lacks %s; no final-test figure",
            path,
            sorted(needed - set(frame.columns)),
        )
        return None
    frame = frame.with_columns(
        pl.col("r2").cast(pl.Float64, strict=False)
    ).filter(pl.col("r2").is_not_null())
    return frame if frame.height else None


def _cv_by_combo(block: pl.DataFrame, scheme: str) -> Dict[str, float]:
    """Best CV score per combination, for the gap panel."""
    sub = block.filter(pl.col("scheme") == scheme)
    out: Dict[str, float] = {}
    for row in sub.iter_rows(named=True):
        value = row.get(SCORE)
        if value is None:
            continue
        combo = str(row.get("combo"))
        out[combo] = max(out.get(combo, float("-inf")), float(value))
    return out


def panel_holdout(
    ax: plt.Axes,
    block: pl.DataFrame,
    combos: List[str],
    hues: Dict[str, str],
) -> None:
    """Final-test R2 per combination, best first."""
    scores = [float(r["r2"]) for r in block.iter_rows(named=True)]
    colors = [
        hues.get(str(r.get("transform")), "#888888")
        for r in block.iter_rows(named=True)
    ]
    positions = list(range(len(combos)))
    bars = ax.bar(positions, scores, color=colors, width=0.68)
    ax.axhline(0.0, color="#333333", lw=1.0, zorder=0)
    for rect, value in zip(bars, scores):
        ax.annotate(
            f"{value:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, value),
            xytext=(0, 3 if value >= 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("holdout R$^2$")
    ax.set_title("(a) final test, ranked", loc="left", fontsize=10)
    _pad_limits(ax)


def panel_gap(
    ax: plt.Axes,
    combos: List[str],
    cv_scores: Dict[str, float],
    ho_scores: Dict[str, float],
) -> bool:
    """CV beside holdout per combination: did the CV winner hold up?"""
    pairs = [
        (c, cv_scores[c], ho_scores[c])
        for c in combos
        if c in cv_scores and c in ho_scores
    ]
    if not pairs:
        return False
    positions = list(range(len(pairs)))
    width = 0.38
    ax.bar(
        [p - width / 2 for p in positions],
        [cv for _, cv, _ in pairs],
        width=width,
        color="#9ecae1",
        label="CV (validation folds)",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [ho for _, _, ho in pairs],
        width=width,
        color="#2171b5",
        label="holdout (final test)",
    )
    ax.axhline(0.0, color="#333333", lw=1.0, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [c for c, _, _ in pairs], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("R$^2$")
    ax.set_title(
        "(b) CV vs final test, same combination", loc="left", fontsize=10
    )
    ax.legend(fontsize=8, frameon=False)
    _pad_limits(ax)
    return True


def draw_holdout(
    label: str,
    holdout: pl.DataFrame,
    models: pl.DataFrame,
    scheme: Optional[str] = None,
) -> Optional[plt.Figure]:
    """The final-test figure for one label, or None if nothing to draw."""
    block = holdout.filter(pl.col("label") == label)
    if not block.height:
        return None
    # One row per combination: a sweep can evaluate several schemes, and
    # stacking them in one bar chart would compare different test sets.
    if "scheme" in block.columns:
        schemes = (
            [scheme]
            if scheme
            else [str(v) for v in block.get_column("scheme").unique().sort()]
        )
        best_scheme = max(
            schemes,
            key=lambda s: block.filter(pl.col("scheme") == s)
            .get_column("r2")
            .max(),
        )
        block = block.filter(pl.col("scheme") == best_scheme)
        if not block.height:
            return None
    else:
        best_scheme = "?"
    block = block.sort("r2", descending=True, nulls_last=True)
    combos = [str(v) for v in block.get_column("combo")]
    hues = hue_map(transform_order(block))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    panel_holdout(axes[0], block, combos, hues)
    ho_scores = {
        str(r["combo"]): float(r["r2"]) for r in block.iter_rows(named=True)
    }
    cv_scores = _cv_by_combo(
        models.filter(pl.col("label") == label), best_scheme
    )
    if not panel_gap(axes[1], combos, cv_scores, ho_scores):
        axes[1].set_visible(False)
        LOGGER.info(
            "%s: no CV score matching scheme '%s'; panel (b) omitted",
            label,
            best_scheme,
        )
    fig.suptitle(
        f"{label} -- final test  (scheme: {best_scheme})",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.015,
        "Samples no fold ever saw. Where (b) shows holdout well below "
        "CV, the CV score was optimistic -- rank on the dark bars.",
        fontsize=8,
        color="#444444",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return fig


def plot_holdout(
    holdout: pl.DataFrame,
    models: pl.DataFrame,
    out_dir: Path,
    formats: List[str],
    labels: Optional[List[str]] = None,
    scheme: Optional[str] = None,
) -> List[Path]:
    """One final-test figure per label. Returns the stems written.

    With `scheme`, each figure shows that scheme instead of the one with
    the highest holdout score, and the file name ends in `__<scheme>`.
    """
    vis = Visualiser(out=out_dir, formats=formats)
    written: List[Path] = []
    wanted = labels or [
        str(v) for v in holdout.get_column("label").unique().sort().to_list()
    ]
    for label in wanted:
        fig = draw_holdout(label, holdout, models, scheme=scheme)
        if fig is None:
            LOGGER.warning("no holdout rows for label '%s'; skipped", label)
            continue
        suffix = f"__{scheme}" if scheme else ""
        target = vis.output_dir / f"transform_sweep_holdout_{label}{suffix}"
        vis._save_fig(fig, target)
        plt.close(fig)
        LOGGER.info("wrote %s", target)
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Figures across every combination, for the end-to-end sweep
# ---------------------------------------------------------------------------
def _panel_grid(
    n_panels: int, ncols: int, width: float, height: float
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """A grid of small multiples, with the unused axes hidden."""
    ncols = min(ncols, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat


def _save(
    fig: plt.Figure, out_dir: Path, formats: List[str], stem: str
) -> Path:
    vis = Visualiser(out=out_dir, formats=formats)
    target = vis.output_dir / stem
    vis._save_fig(fig, target)
    plt.close(fig)
    LOGGER.info("wrote %s", target)
    return target


def plot_icc_distribution(
    tables: List[Tuple[str, List[float]]],
    cutoffs: List[float],
    group: str,
    out_dir: Path,
    formats: List[str],
) -> Optional[Path]:
    """One ICC histogram per combination, with the cutoffs marked."""
    if not tables:
        return None
    fig, axes = _panel_grid(len(tables), 3, 4.6, 3.2)
    for ax, (name, values) in zip(axes, tables):
        ax.hist(
            values,
            bins=30,
            range=(0, 1),
            color=_CATEGORICAL_HUES[0],
            edgecolor="white",
        )
        for cut in cutoffs:
            above = sum(value > cut for value in values)
            ax.axvline(cut, color="#444444", linestyle="--", linewidth=1.0)
            ax.annotate(
                f"{above:,} of {len(values):,} above {cut:g}",
                (cut, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(-4, -12),
                textcoords="offset points",
                ha="right",
                fontsize=8,
            )
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_xlabel("ICC", fontsize=8)
        ax.set_ylabel("taxa", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.18)
    fig.suptitle(
        f"Between-study share of variation, per taxon (by {group})",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, out_dir, formats, "icc_distribution")


def plot_shadow_floor(
    floor: pl.DataFrame, out_dir: Path, formats: List[str]
) -> Optional[Path]:
    """Share of real taxa above the shuffled-noise floor, per combination."""
    if not floor.height:
        return None
    labels = floor.get_column("label").unique().sort().to_list()
    height = 0.5 * floor.get_column("combo").n_unique() + 1.6
    fig, axes = _panel_grid(len(labels), 2, 6.4, height)
    for ax, label in zip(axes, labels):
        block = floor.filter(pl.col("label") == label).sort("pct_above_noise")
        names = block.get_column("combo").to_list()
        values = block.get_column("pct_above_noise").to_list()
        ax.barh(names, values, color=_CATEGORICAL_HUES[0], height=0.62)
        for y, row in enumerate(block.iter_rows(named=True)):
            ax.annotate(
                f"{row['n_real_above_q95']:,} of {row['n_real']:,}",
                (row["pct_above_noise"], y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, max(max(values), 1) * 1.3)
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_xlabel("% of real taxa above the shadow q95", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="x", alpha=0.18)
    fig.suptitle("How many taxa beat shuffled noise", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, formats, "shadow_floor")


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(value)


def plot_scoreboard(
    frame: pl.DataFrame,
    out_dir: Path,
    formats: List[str],
    floor: float = -1.0,
) -> Optional[Path]:
    """One panel per label, one row per combination, three markers each."""
    if not frame.height:
        return None
    labels = frame.get_column("label").unique().sort().to_list()
    n_combos = frame.get_column("combo").n_unique()
    fig, axes = _panel_grid(len(labels), 2, 7.2, 0.46 * n_combos + 1.5)
    series = [
        ("CV, unpruned", "o", "none", _CATEGORICAL_HUES[0]),
        ("holdout, unpruned", "s", _CATEGORICAL_HUES[1], _CATEGORICAL_HUES[1]),
        ("holdout, pruned", "D", _CATEGORICAL_HUES[2], _CATEGORICAL_HUES[2]),
    ]
    clipped = False
    for ax, label in zip(axes, labels):
        block = frame.filter(pl.col("label") == label)
        by_combo: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in block.iter_rows(named=True):
            by_combo.setdefault(row["combo"], {})[row["version"]] = row

        def pruned_score(name: str) -> float:
            value = by_combo[name].get("pruned", {}).get("holdout_r2")
            return value if _finite(value) else float("-inf")

        # Best pruned holdout at the top.
        order = sorted(by_combo, key=pruned_score)
        plotted: List[float] = []
        for y, name in enumerate(order):
            before = by_combo[name].get("unpruned", {})
            after = by_combo[name].get("pruned", {})
            values = [
                before.get("cv_pooled_r2"),
                before.get("holdout_r2"),
                after.get("holdout_r2"),
            ]
            finite = [v for v in values if _finite(v)]
            plotted += finite
            if finite:
                ax.plot(
                    [max(min(finite), floor), max(finite)],
                    [y, y],
                    color="#bbbbbb",
                    linewidth=1.0,
                    zorder=1,
                )
            for value, (_, marker, face, edge) in zip(values, series):
                if _finite(value):
                    ax.scatter(
                        [value],
                        [y],
                        marker=marker,
                        s=46,
                        facecolors=face,
                        edgecolors=edge,
                        linewidths=1.4,
                        zorder=3,
                    )
            n_before, n_after = before.get("n_features"), after.get(
                "n_features"
            )
            if n_before and n_after:
                ax.annotate(
                    f"{n_before:,} → {n_after:,}",
                    (1.0, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color="#444444",
                )
        ax.axvline(0.0, color="#444444", linewidth=1.0, zorder=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_ylim(-0.6, len(order) - 0.4)
        low = min(plotted, default=0.0)
        if low < floor:
            clipped = True
        ax.set_xlim(
            max(min(low, 0.0), floor) - 0.05,
            max(max(plotted, default=0.0), 0.1) + 0.05,
        )
        scheme = ", ".join(sorted(set(block.get_column("scheme").to_list())))
        ax.set_title(f"{label}  ({scheme})", fontsize=10, loc="left")
        ax.set_xlabel("$R^2$", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", alpha=0.18)
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=7,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=1.4,
            label=name,
        )
        for name, marker, face, edge in series
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    note = (
        f" The axis stops at {floor:g}; see the table for lower scores."
        if clipped
        else ""
    )
    fig.suptitle(
        f"Benchmark: every combo, before and after pruning.{note}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 0.98, 0.96), w_pad=5.0)
    return _save(fig, out_dir, formats, "benchmark_scoreboard")


def plot_runtime(
    table: pl.DataFrame,
    stages: List[str],
    out_dir: Path,
    formats: List[str],
) -> Optional[Path]:
    """Minutes per stage for each combination, stacked."""
    if not table.height:
        return None
    names = table.get_column("combo").to_list()
    fig, ax = plt.subplots(figsize=(9.0, 0.46 * len(names) + 1.8))
    left = [0.0] * len(names)
    for i, stage in enumerate(stages):
        values = [float(v) for v in table.get_column(stage).to_list()]
        ax.barh(
            names,
            values,
            left=left,
            height=0.62,
            label=stage,
            color=_CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)],
            edgecolor="white",
            linewidth=1.0,
        )
        left = [a + b for a, b in zip(left, values)]
    for y, total in enumerate(left):
        ax.annotate(
            f"{total:.1f} min",
            (total, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(left) * 1.14 if max(left) else 1)
    ax.set_xlabel("minutes", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="x", alpha=0.18)
    ax.legend(
        fontsize=8,
        frameon=False,
        ncol=len(stages),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    ax.set_title("What each combo cost to run", fontsize=11, loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, formats, "benchmark_runtime")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="the sweep's summary/ directory (holds all_models.tsv)",
    )
    parser.add_argument(
        "--all-models",
        type=Path,
        default=None,
        help="all_models.tsv directly, instead of --summary",
    )
    parser.add_argument(
        "--headline",
        type=Path,
        default=None,
        help="headline.tsv, for n_label in the titles (optional)",
    )
    parser.add_argument(
        "--label",
        nargs="+",
        default=None,
        help="only these labels (default: every label in the table)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <summary>/figures)",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help=(
            "holdout.tsv, for the final-test figures "
            "(default: <summary>/holdout.tsv when it exists)"
        ),
    )
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.all_models is None and args.summary is None:
        raise SystemExit("pass --summary or --all-models")
    models_path = args.all_models or (args.summary / "all_models.tsv")
    if not models_path.exists():
        raise SystemExit(f"no such file: {models_path}")
    headline = args.headline
    if headline is None and args.summary is not None:
        candidate = args.summary / "headline.tsv"
        headline = candidate if candidate.exists() else None

    out_dir = args.out or (
        (args.summary / "figures")
        if args.summary is not None
        else models_path.parent / "figures"
    )
    models = load_models(models_path)
    written = plot_sweep(
        models,
        out_dir,
        list(args.formats),
        headline=headline,
        labels=args.label,
    )

    # Only when the final test actually ran. A CV-only sweep is the
    # normal case and says so once, rather than failing.
    holdout_path = args.holdout
    if holdout_path is None and args.summary is not None:
        holdout_path = args.summary / "holdout.tsv"
    holdout = load_holdout(holdout_path) if holdout_path is not None else None
    if holdout is None:
        LOGGER.info(
            "No holdout table, so no final-test figures. Rerun the "
            "sweep with --holdout to produce one."
        )
    else:
        written += plot_holdout(
            holdout,
            models,
            out_dir,
            list(args.formats),
            labels=args.label,
        )

    if not written:
        raise SystemExit("nothing was plotted; see the warnings above")
    LOGGER.info("Done. %d figure(s) in %s", len(written), out_dir)


if __name__ == "__main__":
    main()
