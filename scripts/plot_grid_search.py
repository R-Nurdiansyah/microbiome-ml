"""Figures from the grid search on several datasets.

Reads what the runs left under one root and draws three things:

    grid_report__*      Per dataset: for every label x scheme, the settings
                        each model chose, its CV score, and the holdout score
                        of the one model sent to the holdout.
    grid_overview       Per label, the best CV score of every dataset x scheme
                        beside what that model then scored on the holdout.
    grid_params__*      Per dataset x label x model, how pooled R2 moved with
                        each hyperparameter that was actually searched.

`grid_report.csv` and `grid_winners.csv` are the first two as tables.

    pixi run python scripts/plot_grid_search.py --root <grid search root>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import plot_transform_sweep as pts
import polars as pl
from matplotlib.lines import Line2D

LOGGER = logging.getLogger("plot-grid-search")

# Slots 1 and 2 of the validated default palette; marker shape carries the
# same meaning, so the two series never rely on colour alone.
CV_HUE = "#2a78d6"
HOLDOUT_HUE = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID_INK = "#d9d8d4"
SURFACE = "#ffffff"

_SCORES = ("pooled_r2", "mean_test_r2", "std_test_r2", "mean_fit_time")
_MODEL_SHORT = {
    "RandomForestRegressor": "RF",
    "XGBRegressor": "XGB",
    "ScaledSVR": "SVR",
}
_KEYS = ["dataset", "label", "scheme", "model"]
# The report's x axis stops here; a worse score is drawn at the edge with its
# value written beside it, so one collapse does not squash every other row.
FLOOR = -1.0


def load_models(root: Path) -> Optional[pl.DataFrame]:
    """Every dataset's summary/all_models.tsv, as written by the sweep."""
    frames = [
        pl.read_csv(path, separator="\t", infer_schema_length=0).with_columns(
            pl.lit(path.relative_to(root).parts[0]).alias("dataset")
        )
        for path in sorted(root.glob("*/summary/all_models.tsv"))
    ]
    if not frames:
        return None
    return pl.concat(frames, how="diagonal").with_columns(
        pl.col("avg_validation_r2", "pooled_validation_r2").cast(
            pl.Float64, strict=False
        ),
        pl.col("n_folds").cast(pl.Int64, strict=False),
    )


def _searched(block: pl.DataFrame) -> List[str]:
    """The param_* columns this block's model varied.

    A column that is null throughout belongs to another model. Within the
    model's own rows a null is a real value, None (max_samples: ~), so it
    counts as one of the values tried.
    """
    return [
        column
        for column in block.columns
        if column.startswith("param_")
        and block.get_column(column).null_count() < block.height
        and block.get_column(column).n_unique() > 1
    ]


def _varying_params(grid: Optional[pl.DataFrame]) -> Dict[str, Set[str]]:
    """Per model, the hyperparameters that took more than one value."""
    if grid is None:
        return {}
    return {
        str(model): {c.removeprefix("param_") for c in _searched(block)}
        for (model,), block in grid.group_by(["model"])
    }


def _settings(raw: Any, keep: Optional[Set[str]]) -> str:
    """`best_params` JSON as `name=value, ...`, searched settings only."""
    try:
        params = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return str(raw)
    items = [
        f"{name}={value}"
        for name, value in sorted((params or {}).items())
        if keep is None or name in keep
    ]
    return ", ".join(items) or "defaults"


def report_table(
    models: pl.DataFrame, grid: Optional[pl.DataFrame]
) -> pl.DataFrame:
    """One row per dataset x label x scheme x model, settings spelled out.

    all_models.tsv holds each winner up to three times: from cv_results/,
    from best_models/, and from holdout/. The holdout copy has a single
    "fold" whose score is the holdout R2, so it is split off here and
    attached to the model it belongs to, not read as a CV result.
    """
    cv = (
        models.filter(pl.col("n_folds") > 1)
        .unique(
            subset=[*_KEYS, "feature_set"], keep="first", maintain_order=True
        )
        .select(
            *_KEYS,
            "feature_set",
            pl.col("pooled_validation_r2").alias("cv_pooled_r2"),
            pl.col("avg_validation_r2").alias("cv_mean_r2"),
            "best_params",
        )
    )
    held = (
        models.filter(pl.col("n_folds") == 1)
        .unique(subset=_KEYS, keep="first", maintain_order=True)
        .select(*_KEYS, pl.col("avg_validation_r2").alias("holdout_r2"))
    )
    varying = _varying_params(grid)
    # Attached before the join: a join need not keep row order.
    cv = cv.with_columns(
        pl.Series(
            "settings",
            [
                _settings(params, varying.get(str(model)))
                for model, params in cv.select(
                    "model", "best_params"
                ).iter_rows()
            ],
            dtype=pl.String,
        )
    )
    return (
        cv.join(held, on=_KEYS, how="left")
        .with_columns(
            pl.col("holdout_r2").is_not_null().alias("sent_to_holdout")
        )
        .drop("best_params")
        .sort("dataset", "label", "scheme", "model")
    )


def plot_report(
    table: pl.DataFrame, out_dir: Path, formats: List[str], dataset: str
) -> Optional[Path]:
    """One panel per label; a row per scheme x model, settings in the label."""
    if not table.height:
        return None
    labels = table.get_column("label").unique().sort().to_list()
    n_rows = int(table.group_by("label").len().get_column("len").max() or 1)
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(7.4 * len(labels), 0.62 * n_rows + 1.9),
        squeeze=False,
    )
    for ax, label in zip(axes[0], labels):
        block = table.filter(pl.col("label") == label)
        names: List[str] = []
        for y, row in enumerate(block.iter_rows(named=True)):
            model = _MODEL_SHORT.get(row["model"], row["model"])
            names.append(f"{row['scheme']} · {model}\n{row['settings']}")
            for value, marker, face, edge in (
                (row["cv_pooled_r2"], "o", SURFACE, CV_HUE),
                (row["holdout_r2"], "s", HOLDOUT_HUE, SURFACE),
            ):
                if not pts._finite(value):
                    continue
                shown = max(value, FLOOR)
                ax.scatter(
                    [shown],
                    [y],
                    marker=marker if value >= FLOOR else "<",
                    s=64,
                    facecolors=face,
                    edgecolors=edge,
                    linewidths=2 if marker == "o" else 1,
                    zorder=3,
                )
                if value < FLOOR:
                    ax.annotate(
                        f"{value:.2f}",
                        (shown, y),
                        xytext=(8, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=8,
                        color=INK_MUTED,
                    )
            finite = [
                max(v, FLOOR)
                for v in (row["cv_pooled_r2"], row["holdout_r2"])
                if pts._finite(v)
            ]
            if len(finite) == 2:
                ax.plot(
                    finite, [y, y], color=GRID_INK, linewidth=1.2, zorder=1
                )
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, zorder=2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(left=min(0.0, ax.get_xlim()[0]) - 0.05)
        ax.set_xlabel("$R^2$", color=INK_MUTED, fontsize=9)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        _style(ax)
    has_holdout = bool(table.get_column("holdout_r2").is_not_null().any())
    fig.legend(
        handles=_legend(
            has_holdout, "Holdout, for the model the scheme sent to it"
        ),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.text(
        0.5,
        0.005,
        (
            "Every scheme of a label is tested on the same holdout, so "
            "schemes that chose the same model and settings get the same "
            "holdout score."
            if has_holdout
            else "No holdout scored yet: CV only."
        ),
        ha="center",
        fontsize=8,
        color=INK_MUTED,
    )
    fig.suptitle(
        f"{dataset}: settings each grid search chose, per CV scheme",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    return pts._save(fig, out_dir, formats, f"grid_report__{dataset}")


def load_grid(root: Path) -> Optional[pl.DataFrame]:
    """Every candidate of every run under `root`, with its dataset."""
    frames = [
        pl.read_csv(path, infer_schema_length=0).with_columns(
            pl.lit(path.relative_to(root).parts[0]).alias("dataset")
        )
        for path in sorted(
            root.glob("*/*/holdout_*/runs/*/*/cv_results/grid_results.csv")
        )
    ]
    if not frames:
        return None
    return pl.concat(frames, how="diagonal").with_columns(
        pl.col(list(_SCORES)).cast(pl.Float64, strict=False),
        (pl.col("selected") == "True").alias("selected"),
    )


def load_holdout(root: Path) -> Optional[pl.DataFrame]:
    """Holdout R2 per dataset x label x scheme."""
    frames = [
        pl.read_csv(path, infer_schema_length=0)
        .select("label", "scheme", pl.col("r2").alias("holdout_r2"))
        .with_columns(pl.lit(path.relative_to(root).parts[0]).alias("dataset"))
        for path in sorted(
            root.glob("*/*/holdout_*/runs/*/*/holdout/holdout_summary.csv")
        )
    ]
    if not frames:
        return None
    return pl.concat(frames).with_columns(
        pl.col("holdout_r2").cast(pl.Float64, strict=False)
    )


def winners(
    grid: pl.DataFrame, holdout: Optional[pl.DataFrame]
) -> pl.DataFrame:
    """The one model per dataset x label x scheme that went to the holdout.

    Ties on pooled R2 keep the first row, so the table does not depend on
    which model happened to be sorted first by the group_by.
    """
    best = (
        grid.filter(pl.col("selected"))
        .sort("pooled_r2", descending=True, nulls_last=True)
        .group_by(["dataset", "label", "scheme"], maintain_order=True)
        .first()
        .select(
            "dataset",
            "label",
            "scheme",
            "model",
            "feature_set",
            "pooled_r2",
            "mean_test_r2",
        )
    )
    if holdout is None:
        best = best.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("holdout_r2")
        )
    else:
        best = best.join(
            holdout, on=["dataset", "label", "scheme"], how="left"
        )
    return best.sort("label", "dataset", "scheme")


def _legend(has_holdout: bool, holdout_text: str) -> List[Line2D]:
    """CV marker always; the holdout one only when something was scored."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=SURFACE,
            markeredgecolor=CV_HUE,
            markeredgewidth=2,
            label="CV, pooled across folds",
        )
    ]
    if has_holdout:
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                markersize=8,
                color=HOLDOUT_HUE,
                label=holdout_text,
            )
        )
    return handles


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_INK)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=8)
    ax.grid(axis="x", color=GRID_INK, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def plot_overview(
    table: pl.DataFrame, out_dir: Path, formats: List[str]
) -> Optional[Path]:
    """One panel per label: CV beside holdout for each dataset x scheme."""
    if not table.height:
        return None
    labels = table.get_column("label").unique().sort().to_list()
    n_rows = int(table.group_by("label").len().get_column("len").max() or 1)
    fig, axes = pts._panel_grid(len(labels), 2, 6.4, 0.34 * n_rows + 1.3)
    for ax, label in zip(axes, labels):
        block = table.filter(pl.col("label") == label)
        names = []
        for y, row in enumerate(block.iter_rows(named=True)):
            model = _MODEL_SHORT.get(row["model"], row["model"])
            names.append(f"{row['dataset']}  {row['scheme']}  ({model})")
            cv, held = row["pooled_r2"], row["holdout_r2"]
            finite = [v for v in (cv, held) if pts._finite(v)]
            if finite:
                ax.plot(
                    [min(finite), max(finite)],
                    [y, y],
                    color=GRID_INK,
                    linewidth=1.2,
                    zorder=1,
                )
            if pts._finite(cv):
                ax.scatter(
                    [cv],
                    [y],
                    marker="o",
                    s=64,
                    facecolors=SURFACE,
                    edgecolors=CV_HUE,
                    linewidths=2,
                    zorder=3,
                )
            if pts._finite(held):
                ax.scatter(
                    [held],
                    [y],
                    marker="s",
                    s=64,
                    color=HOLDOUT_HUE,
                    edgecolors=SURFACE,
                    linewidths=1,
                    zorder=4,
                )
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, zorder=2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("$R^2$", color=INK_MUTED, fontsize=9)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        _style(ax)
    fig.legend(
        handles=_legend(
            bool(table.get_column("holdout_r2").is_not_null().any()),
            "Holdout",
        ),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Best grid-search model per dataset and CV scheme",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return pts._save(fig, out_dir, formats, "grid_overview")


def _ordered(values: List[str]) -> List[str]:
    """Numbers in numeric order, then everything else (e.g. None, sqrt)."""

    def key(value: str) -> tuple:
        try:
            return (0, float(value), value)
        except ValueError:
            return (1, 0.0, value)

    return sorted(values, key=key)


def plot_params(
    block: pl.DataFrame,
    out_dir: Path,
    formats: List[str],
    stem: str,
    title: str,
) -> Optional[Path]:
    """Schemes as rows, searched hyperparameters as columns, one y scale."""
    columns = _searched(block)
    schemes = block.get_column("scheme").unique().sort().to_list()
    if not columns or not schemes:
        return None
    fig, axes = plt.subplots(
        len(schemes),
        len(columns),
        figsize=(2.6 * len(columns) + 1.2, 1.9 * len(schemes) + 1.2),
        squeeze=False,
        sharey=True,
    )
    for i, scheme in enumerate(schemes):
        rows = block.filter(pl.col("scheme") == scheme)
        for j, column in enumerate(columns):
            ax = axes[i][j]
            values = _ordered(
                [str(v) for v in rows.get_column(column).unique().to_list()]
            )
            where = {value: x for x, value in enumerate(values)}
            ordinary = rows.filter(~pl.col("selected"))
            ax.scatter(
                [where[str(v)] for v in ordinary.get_column(column)],
                ordinary.get_column("pooled_r2").to_list(),
                marker="o",
                s=64,
                color=CV_HUE,
                alpha=0.55,
                edgecolors=SURFACE,
                linewidths=1,
                zorder=2,
            )
            chosen = rows.filter(pl.col("selected"))
            ax.scatter(
                [where[str(v)] for v in chosen.get_column(column)],
                chosen.get_column("pooled_r2").to_list(),
                marker="D",
                s=64,
                color=HOLDOUT_HUE,
                edgecolors=SURFACE,
                linewidths=1,
                zorder=3,
            )
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values)
            ax.set_xlim(-0.5, len(values) - 0.5)
            ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color(GRID_INK)
            ax.spines["bottom"].set_color(GRID_INK)
            ax.tick_params(colors=INK_MUTED, length=0, labelsize=8)
            if i == 0:
                ax.set_title(
                    column.removeprefix("param_"),
                    fontsize=9,
                    color=INK,
                )
            if j == 0:
                ax.set_ylabel(
                    f"{scheme}\npooled $R^2$",
                    fontsize=8,
                    color=INK_MUTED,
                )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                color=CV_HUE,
                label="Candidate",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                markersize=8,
                color=HOLDOUT_HUE,
                label="Selected",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return pts._save(fig, out_dir, formats, stem)


def plot_all(root: Path, formats: List[str]) -> List[Path]:
    """Draw everything the runs under `root` have finished so far."""
    grid = load_grid(root)
    models = load_models(root)
    if grid is None and models is None:
        LOGGER.warning("no grid_results.csv or all_models.tsv under %s", root)
        return []
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: List[Optional[Path]] = []

    if models is not None:
        report = report_table(models, grid)
        report.write_csv(root / "grid_report.csv")
        for (dataset,), block in report.group_by(["dataset"]):
            written.append(plot_report(block, figures, formats, str(dataset)))
    if grid is None:
        return [path for path in written if path is not None]

    table = winners(grid, load_holdout(root))
    table.write_csv(root / "grid_winners.csv")
    written.append(plot_overview(table, figures, formats))
    groups = ["dataset", "label", "model"]
    for keys, block in grid.group_by(groups, maintain_order=True):
        dataset, label, model = keys
        short = _MODEL_SHORT.get(str(model), str(model))
        written.append(
            plot_params(
                block,
                figures,
                formats,
                f"grid_params__{dataset}__{label}__{short}",
                f"{dataset}, {label}, {short}: pooled $R^2$ by hyperparameter",
            )
        )
    return [path for path in written if path is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not plot_all(args.root, args.formats):
        sys.exit(f"Nothing to draw under {args.root}")


if __name__ == "__main__":
    main()
