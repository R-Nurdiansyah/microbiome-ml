#!/usr/bin/env python3
"""Plot a pruning trajectory: does the score hold up as features go?

Reads one or more `prune_history.tsv` files written by
scripts/prune_iterate_cv.py and draws score against features retained,
which is the question the loop exists to answer -- not "is round 3
better than round 2" but "how far left can I go before the line drops".

The x axis is features retained, log-scaled and inverted, so reading
left to right follows the pruning. Each marker is one round. The dashed
line is the round-0 score: while the trajectory tracks it, pruning is
free; where it leaves it, pruning has started to cost something.

Nothing here is specific to a pruning criterion. The `prune` column
names the one that ran, so a `zero` run and a `coverage` run of the
same combination overlay as two lines on one panel -- pass both files
and the comparison draws itself: which criterion holds the score with
fewer features.

Scores on these axes come from the folds the pruning was derived from,
so the figure says how much is droppable, not that a pruned model is
better. The caption repeats that, because a figure outlives its README.

Example:
    python scripts/plot_prune_history.py \\
        --history out/prune_ph/prune_history.tsv \\
                  out/prune_ph_cov/prune_history.tsv \\
        --out out/prune_ph/figures --formats png svg
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from microbiome_ml.visualise.visualisations import (  # noqa: E402
    _CATEGORICAL_HUES,
    Visualiser,
    _fmt_r2,
)

LOGGER = logging.getLogger("plot-prune")

# One panel per combination of these; one line per SERIES_KEYS within it.
PANEL_KEYS = ["label", "scheme", "feature_set"]
SERIES_KEYS = ["model", "prune"]

SCORE = "pooled_validation_r2"

# Rounds are annotated individually up to this many, then only the ends,
# which keeps a long coverage trajectory from turning into a smudge.
MAX_ROUND_LABELS = 7

# A drawable series, with polars already behind us: the figure is built
# from plain lists so it can be exercised without a prune run.
Series = Dict[str, Any]
Panel = Tuple[Tuple[Any, ...], List[Series]]


def load_history(paths: List[Path]) -> pl.DataFrame:
    """Concatenate prune_history.tsv files written by the prune loop."""
    frames = []
    for path in paths:
        df = pl.read_csv(path, separator="\t", infer_schema_length=10000)
        if not df.height:
            LOGGER.warning("%s is empty; skipped", path)
            continue
        if "prune" not in df.columns:
            # Written before the criterion became a choice.
            df = df.with_columns(pl.lit("zero").alias("prune"))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no usable rows in {[str(p) for p in paths]}")
    history = pl.concat(frames, how="diagonal_relaxed")
    required = (*PANEL_KEYS, *SERIES_KEYS, SCORE, "iteration")
    missing = [c for c in required if c not in history.columns]
    if missing:
        raise ValueError(
            f"prune history is missing column(s) {missing}; expected the "
            "table written by scripts/prune_iterate_cv.py"
        )
    return history.sort([*PANEL_KEYS, *SERIES_KEYS, "iteration"])


def _short_model(name: Any) -> str:
    """`RandomForestRegressor` -> `RandomForest`."""
    text = str(name)
    for suffix in ("Regressor", "Classifier"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def series_label(model: Any, mode: Any, coverage: Optional[float]) -> str:
    """`RandomForest / zero` or `XGB / coverage 0.95`."""
    if str(mode) == "coverage" and coverage is not None:
        return f"{_short_model(model)} / coverage {float(coverage):g}"
    return f"{_short_model(model)} / {mode}"


def to_panels(history: pl.DataFrame) -> List[Panel]:
    """Turn the table into plain per-panel, per-series point lists.

    Rounds with no finite score are dropped here rather than in the
    drawing code, so a panel either has points or is reported missing.
    """
    panels: List[Panel] = []
    for keys, panel in history.group_by(PANEL_KEYS, maintain_order=True):
        series: List[Series] = []
        for skeys, block in panel.group_by(SERIES_KEYS, maintain_order=True):
            xs: List[int] = []
            ys: List[float] = []
            rounds: List[int] = []
            for row in block.iter_rows(named=True):
                score, n_in = row.get(SCORE), row.get("n_features_in")
                if score is None or n_in is None:
                    continue
                value = float(score)
                if not math.isfinite(value):
                    continue
                xs.append(int(n_in))
                ys.append(value)
                rounds.append(int(row["iteration"]))
            if not xs:
                LOGGER.warning(
                    "no scored round for %s / %s; series skipped",
                    tuple(keys),
                    tuple(skeys),
                )
                continue
            coverage = (
                block.get_column("coverage")[0]
                if "coverage" in block.columns
                else None
            )
            series.append(
                {
                    "label": series_label(skeys[0], skeys[1], coverage),
                    "xs": xs,
                    "ys": ys,
                    "rounds": rounds,
                    "status": str(block.get_column("status")[-1])
                    if "status" in block.columns
                    else "",
                }
            )
        if series:
            panels.append((tuple(keys), series))
    return panels


def _annotate_rounds(
    ax: plt.Axes,
    xs: List[int],
    ys: List[float],
    rounds: List[int],
    colour: str,
) -> None:
    """Number the markers, sparsely once there are many of them."""
    if len(rounds) <= MAX_ROUND_LABELS:
        show = set(range(len(rounds)))
    else:
        show = {0, len(rounds) - 1}
    for i in sorted(show):
        ax.annotate(
            str(rounds[i]),
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color=colour,
        )


def draw_panel(
    ax: plt.Axes, keys: Tuple[Any, ...], series: List[Series]
) -> int:
    """Draw one panel; returns how many series landed on it."""
    baselines: List[float] = []
    for i, spec in enumerate(series):
        xs, ys = spec["xs"], spec["ys"]
        colour = _CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)]
        suffix = f", {spec['status']}" if spec["status"] else ""
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            color=colour,
            label=(f"{spec['label']}  " f"({xs[0]:,}→{xs[-1]:,}{suffix})"),
        )
        _annotate_rounds(ax, xs, ys, spec["rounds"], colour)
        # Round 0 is the unpruned reference every later round is really
        # being compared against.
        ax.axhline(
            ys[0], color=colour, linestyle="--", linewidth=0.8, alpha=0.5
        )
        baselines.append(ys[0])

    if not baselines:
        return 0

    ax.set_xscale("log")
    # Limits come from the data, not from autoscale: a single-round
    # series leaves matplotlib padding a lone point out to a full decade
    # each side, which reads as a huge range that was never explored.
    all_x = [x for spec in series for x in spec["xs"]]
    lo, hi = float(min(all_x)), float(max(all_x))
    if lo == hi:
        lo, hi = lo / 1.5, hi * 1.5
    else:
        pad = (hi / lo) ** 0.06
        lo, hi = lo / pad, hi * pad
    # Inverted, so moving right along the axis means fewer features --
    # the direction the loop travels.
    ax.set_xlim(hi, lo)
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="both", alpha=0.18)
    ax.set_title(
        "  ".join(f"{k}={v}" for k, v in zip(PANEL_KEYS, keys)), fontsize=9
    )
    ax.set_xlabel("features retained (log, fewer to the right)", fontsize=8)
    ax.set_ylabel("pooled validation $R^2$", fontsize=8)
    ax.tick_params(labelsize=8)
    # The baseline explanation rides in the legend rather than as loose
    # text in a corner, which collided with the legend on a panel with
    # many rounds.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], color="#444444", linestyle="--", linewidth=0.8)
    )
    labels.append(
        "round 0, unpruned " f"({', '.join(_fmt_r2(b) for b in baselines)})"
    )
    ax.legend(handles, labels, fontsize=7, loc="best", framealpha=0.85)
    return len(baselines)


def draw_figure(panels: List[Panel]) -> Optional[plt.Figure]:
    """The whole figure, or None when no panel had anything to draw."""
    if not panels:
        LOGGER.warning("nothing to plot")
        return None
    ncols = 1 if len(panels) == 1 else 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows), squeeze=False
    )
    flat = [ax for row in axes for ax in row]

    drawn = 0
    for ax, (keys, series) in zip(flat, panels):
        drawn += 1 if draw_panel(ax, keys, series) else 0
    for ax in flat[len(panels) :]:
        ax.set_visible(False)

    if not drawn:
        plt.close(fig)
        LOGGER.warning("no series had a finite score; no figure written")
        return None

    fig.suptitle(
        "Pruning trajectory: score against features retained", y=0.995
    )
    fig.text(
        0.5,
        0.006,
        "Scores come from the folds the pruning was derived from, so this "
        "shows how much is droppable, not that pruning improves the "
        "model. Compare on the holdout.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def plot_history(
    history: pl.DataFrame,
    out_dir: Path,
    formats: List[str],
    stem: str = "prune_trajectory",
) -> Optional[Path]:
    """Draw and save; None when nothing was drawable."""
    fig = draw_figure(to_panels(history))
    if fig is None:
        return None
    vis = Visualiser(out=out_dir, formats=formats)
    target = vis.output_dir / stem
    vis._save_fig(fig, target)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--history",
        type=Path,
        nargs="+",
        required=True,
        help="one or more prune_history.tsv; several overlay on one panel",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <first --history dir>/figures)",
    )
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--stem", default="prune_trajectory")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    history = load_history(list(args.history))
    out_dir = args.out or args.history[0].parent / "figures"
    target = plot_history(history, out_dir, list(args.formats), str(args.stem))
    if target is None:
        raise SystemExit("nothing was plotted; see the warnings above")
    LOGGER.info("Done. Figure stem: %s", target)


if __name__ == "__main__":
    main()
