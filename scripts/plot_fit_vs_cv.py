"""Train vs CV vs holdout R² per CV scheme: is the model stable, and does it overfit?

Usage:
    python plot_fit_vs_cv.py -d SWEEP_DIR [-o OUTPUT_DIR] [--top N|all]

SWEEP_DIR is a sweep's base_dir, holding the tables collate writes:
    cv_sweep_all.tsv          CV fold scores per scheme x model (required)
    final_fit_all.tsv         R² of the final model on its own 80% (training fit)
    holdout_sweep_summary.tsv R² on the held-out 20% per scheme
At least one of the last two must exist; a missing one just drops its marker
(no holdout_evaluation -> train vs CV only, and the gap is train - CV).

One panel per holdout grouping x label, laid out like plot_cv_sweep_top3.py
(rows: base label; columns: plain / geo_ / comb_). One row per CV scheme, using
the model that scheme picked (the one the final fit and the holdout used):

    square   train R²    final model scored on the data it was fit on (a ceiling)
    circle   CV R²       mean over folds; the whisker spans the fold min..max
    diamond  holdout R²  the held-out 20%: the honest test score

Stability: a short whisker. Overfit: a long way from the square down to the
diamond. A CV scheme that predicts the holdout well puts its circle on the diamond.

Writes fit_vs_cv_{topN|all}.png (default: SWEEP_DIR/sweep_visualisations).
`collate` in pipeline_steps.py runs this with --top 3 and --top all.
"""
import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_order import label_key  # noqa: E402

VARIANTS = ("", "geo_", "comb_")
MODEL_ABBR = {"RandomForestRegressor": "RF", "XGBRegressor": "XGB", "ScaledSVR": "SVR"}
KEY = ["holdout_grouping", "label", "scheme"]
X_LO, X_HI = -1.0, 1.0  # R² below X_LO is drawn at the edge, real value printed

# first three categorical slots: they separate under every colour-vision type
TRAIN = dict(color="#1baf7a", marker="s", label="train R² (final fit, in-sample)")
CV = dict(color="#2a78d6", marker="o", label="CV R² (fold mean, whisker = min..max)")
TEST = dict(color="#eb6834", marker="D", label="holdout R² (test 20%)")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def top_arg(v):
    if v.lower() == "all":
        return None
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("--top must be >= 1 or 'all'")
    return n


def folds(cell):
    try:
        vals = ast.literal_eval(cell) if isinstance(cell, str) else []
        return [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    except (ValueError, SyntaxError):
        return []


def variant_of(label):
    for v in VARIANTS[1:]:
        if label.startswith(v):
            return v, label[len(v):]
    return "", label


def title_of(panel):
    g, lab = panel
    return g if g.endswith(f"__{lab}") else f"{g}__{lab}"


def prefix(panel):
    return panel[0].split("__")[0]


def layout(panels):
    """Same grid as plot_cv_sweep_top3.py: a row per base label, a column per family."""
    present = [v for v in VARIANTS if any(variant_of(lab)[0] == v for _, lab in panels)]
    by_base = {}
    for panel in panels:
        v, base = variant_of(panel[1])
        by_base.setdefault(base, {u: [] for u in present})[v].append(panel)
    grid = []
    # Rows always temperature, ph, oxy (label_order.py).
    for _base, cols in sorted(by_base.items(), key=lambda kv: label_key(kv[0])):
        cols = [list(cols[v]) for v in present]
        while any(cols):
            lead_i = next(i for i, c in enumerate(cols) if c)
            lead = cols[lead_i].pop(0)
            row = [None] * len(cols)
            row[lead_i] = lead
            for j, c in enumerate(cols):
                if j != lead_i and c:
                    match = next((p for p in c if prefix(p) == prefix(lead)), c[0])
                    c.remove(match)
                    row[j] = match
            grid.append(row)
    return grid, present


def _table(path, value, rename):
    """KEY (+ model, when recorded) and one value column; None if absent."""
    if not path.is_file():
        return None
    t = pd.read_csv(path, sep="\t")
    cols = KEY + (["model"] if "model" in t.columns else []) + [value]
    return t[cols].drop_duplicates(KEY).rename(columns={value: rename})


def load(d):
    cv = pd.read_csv(d / "cv_sweep_all.tsv", sep="\t").drop_duplicates()
    fit = _table(d / "final_fit_all.tsv", "in_sample_r2", "train_r2")
    test = _table(d / "holdout_sweep_summary.tsv", "r2", "test_r2")
    # The scheme's chosen model is the one its final fit / holdout used,
    # else the best by CV (they agree: the CV winner is what gets refit).
    chosen = (cv.sort_values("avg_validation_r2", ascending=False)
                .drop_duplicates(KEY)[KEY + ["model"]])
    for t in (test, fit):
        if t is not None and "model" in t.columns:
            named = chosen.merge(t[KEY + ["model"]], on=KEY, how="left", suffixes=("", "_t"))
            chosen["model"] = named["model_t"].fillna(named["model"]).to_numpy()
    rows = chosen.merge(cv, on=KEY + ["model"], how="left")
    for t, col in ((fit, "train_r2"), (test, "test_r2")):
        if t is None:
            rows[col] = np.nan
            continue
        on = KEY + (["model"] if "model" in t.columns and t["model"].notna().all() else [])
        rows = rows.merge(t[on + [col]], on=on, how="left")
    rows["folds"] = rows["cross_val_scores"].map(folds) if "cross_val_scores" in rows else [[]] * len(rows)
    return rows


def clip(v):
    return min(max(v, X_LO), X_HI)


def draw(rows, top, out):
    panels = list(dict.fromkeys(zip(rows["holdout_grouping"], rows["label"])))
    grid, present = layout(panels)
    picked = {}
    for panel, sub in rows.groupby(["holdout_grouping", "label"], sort=False):
        sub = sub.sort_values("avg_validation_r2", ascending=False)
        grouped = sub[sub["scheme"] != "random"]
        picked[panel] = pd.concat([sub[sub["scheme"] == "random"].head(1),
                                   grouped.head(top or len(grouped))])
    n_rows = max(len(p) for p in picked.values())
    has_train, has_test = rows["train_r2"].notna().any(), rows["test_r2"].notna().any()

    ncol, nrow = len(present), len(grid)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, (0.42 * n_rows + 1.3) * nrow),
                             squeeze=False)
    for ax, panel in zip(axes.flat, (p for r in grid for p in r)):
        if panel is None or panel not in picked:
            ax.set_visible(False)
            continue
        sub = picked[panel].iloc[::-1]  # best grouped scheme on top, random at the bottom
        y = np.arange(len(sub))
        right = blended_transform_factory(ax.transAxes, ax.transData)
        for yi, (_, r) in zip(y, sub.iterrows()):
            f = r["folds"]
            if f:
                ax.plot([clip(min(f)), clip(max(f))], [yi, yi], color=CV["color"], lw=2,
                        solid_capstyle="round", alpha=0.45, zorder=2)
            train, cvm, test = r["train_r2"], r["avg_validation_r2"], r["test_r2"]
            for spec, v in ((TRAIN, train), (CV, cvm), (TEST, test)):
                if pd.notna(v):
                    ax.scatter(clip(v), yi, s=46, color=spec["color"], marker=spec["marker"],
                               edgecolor="white", linewidth=1.2, zorder=4)
                    if v < X_LO:  # CV's real value above the row, holdout's below
                        up = spec is not TEST
                        ax.annotate(f"{v:.2f}", (X_LO, yi), xytext=(4, 5 if up else -5),
                                    textcoords="offset points", fontsize=7, color=spec["color"],
                                    va="bottom" if up else "top")
            parts = []
            if has_test:
                parts.append(f"test {test:.2f}" if pd.notna(test) else "test n/a")
            low = test if has_test else cvm  # gap down to the honest score present
            if pd.notna(train) and pd.notna(low):
                parts.append(f"gap {train - low:.2f}")
            ax.text(1.02, yi, "  ".join(parts), transform=right, fontsize=8, color=INK,
                    va="center")
        names = [f"{s} · {MODEL_ABBR.get(m, m)}" for s, m in zip(sub["scheme"], sub["model"])]
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        for tick, s in zip(ax.get_yticklabels(), sub["scheme"]):
            if s == "random":
                tick.set_color(MUTED)
                tick.set_style("italic")
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.set_xlim(X_LO - 0.05, X_HI + 0.02)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title_of(panel), fontsize=11, loc="left")
        ax.set_xlabel("R²", fontsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    shown = [s for s, on in ((TRAIN, has_train), (CV, True), (TEST, has_test)) if on]
    handles = [Line2D([], [], ls="", marker=s["marker"], color=s["color"], markeredgecolor="white",
                      markersize=8, label=s["label"]) for s in shown]
    fig_h = fig.get_figheight()
    which = "all schemes" if top is None else f"random + best {top} grouped schemes"
    if has_test:
        what = "Train vs CV vs holdout R²" if has_train else "CV vs holdout R²"
        note = "right: holdout R² and train − holdout gap" if has_train else "right: holdout R²"
    else:
        what, note = "Train vs CV R² (no holdout evaluation)", "right: train − CV gap"
    fig.suptitle(f"{what} ({which}; {note})", y=1 - 0.12 / fig_h, va="top")
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.45 / fig_h), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / fig_h))
    fig.subplots_adjust(wspace=0.75)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-d", "--dir", type=Path, required=True, help="sweep base_dir with the collated tables")
    p.add_argument("-o", "--output", type=Path, help="output directory (default: DIR/sweep_visualisations)")
    p.add_argument("--top", type=top_arg, default=3, help="grouped schemes per panel, or 'all' (default: 3)")
    a = p.parse_args()
    if not (a.dir / "cv_sweep_all.tsv").is_file():
        p.error(f"{a.dir} has no cv_sweep_all.tsv; run collate first")
    extra = ["final_fit_all.tsv", "holdout_sweep_summary.tsv"]
    if not any((a.dir / n).is_file() for n in extra):
        p.error(f"{a.dir} has neither {' nor '.join(extra)}; "
                "turn on final_fit or holdout_evaluation")
    name = f"fit_vs_cv_{'all' if a.top is None else f'top{a.top}'}.png"
    draw(load(a.dir), a.top, (a.output or a.dir / "sweep_visualisations") / name)


if __name__ == "__main__":
    main()
