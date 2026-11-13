"""Bar plot of a CV-sweep metric per holdout grouping x label:
best random-scheme run vs the best N non-random scheme_featureset_model runs.

Usage:
    python plot_cv_sweep_top3.py -i cv_sweep_all.tsv [-o OUTPUT] [--top N|all]
                                 [--metric NAME [NAME ...] | --metric all]

--metric picks the column to plot and rank by (default avg_validation_r2).
Known metrics: avg/pooled/median_validation_r2, iqr_validation_r2,
frac_folds_negative_r2, avg_validation_mse. 'all' plots every one the input has.

OUTPUT may be a file (.png/.pdf/.svg) or a directory; if it is a directory
(or omitted) each figure is written as cv_sweep_{topN|all}_{metric}.png inside it
(default: the input file's folder). With several metrics and a file OUTPUT, the
metric name is appended to the file stem.

One panel per holdout grouping x label, so a run that holds every label
(holdout_groupings: [bioproject]) and a label-paired run
(pairs: {geo_ph: bioproject}) both get a panel per label. Rows are one base
label; columns are its plain, geo_ and comb_ versions, whichever exist.

`collate` in pipeline_steps.py runs this with --top 3 and --top all.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_order import label_key  # noqa: E402

# name -> how to draw it.
#   higher: True if a larger value is better (sets the ranking direction)
#   limits: fixed y range shared by every panel; values outside are drawn to the
#           edge with a break mark and their real value printed. None = free axis
#           per panel (MSE is in label units, so ph and temperature can't share one)
METRICS = {
    "avg_validation_r2":      dict(ylabel="avg validation R²",        higher=True,  limits=(-1.0, 1.0)),
    "pooled_validation_r2":   dict(ylabel="pooled validation R²",     higher=True,  limits=(-1.0, 1.0)),
    "median_validation_r2":   dict(ylabel="median validation R²",     higher=True,  limits=(-1.0, 1.0)),
    "iqr_validation_r2":      dict(ylabel="IQR of fold R²",           higher=False, limits=(0.0, 2.0)),
    "frac_folds_negative_r2": dict(ylabel="fraction of folds R² < 0", higher=False, limits=(0.0, 1.0)),
    "avg_validation_mse":     dict(ylabel="avg validation MSE",       higher=False, limits=None),
}

# Label families, one figure column each, in this order.
VARIANTS = ("", "geo_", "comb_")


def top_arg(v):
    if v.lower() == "all":
        return None
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("--top must be >= 1 or 'all'")
    return n


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("-i", "--input", type=Path, required=True, help="cv_sweep_all.tsv from the CV sweep")
parser.add_argument("-o", "--output", type=Path,
                    help="output figure path or directory (default: input folder)")
parser.add_argument("--top", type=top_arg, default=3,
                    help="number of best non-random runs per panel, or 'all' (default: 3)")
parser.add_argument("--metric", nargs="+", default=["avg_validation_r2"],
                    help=f"column(s) to plot, or 'all'. Known: {', '.join(METRICS)}")
args = parser.parse_args()

src = args.input
if not src.is_file():
    parser.error(f"input file not found: {src}")

df = pd.read_csv(src, sep="\t").drop_duplicates()

if [m.lower() for m in args.metric] == ["all"]:
    metrics = [m for m in METRICS if m in df.columns]
else:
    metrics = args.metric
    unknown = [m for m in metrics if m not in METRICS]
    missing = [m for m in metrics if m in METRICS and m not in df.columns]
    if unknown:
        parser.error(f"unknown metric(s) {unknown}; known: {list(METRICS)}")
    if missing:
        parser.error(f"metric(s) {missing} not in {src.name}; it has: "
                     f"{[m for m in METRICS if m in df.columns]}")

top_txt = "all" if args.top is None else f"top{args.top}"


def out_path(metric):
    name = f"cv_sweep_{top_txt}_{metric}.png"
    if args.output is None:
        return src.with_name(name)
    if args.output.is_dir() or not args.output.suffix:
        return args.output / name
    if len(metrics) == 1:
        return args.output
    return args.output.with_name(f"{args.output.stem}_{metric}{args.output.suffix}")


MODEL_ABBR = {"RandomForestRegressor": "RF", "XGBRegressor": "XGB", "ScaledSVR": "SVR"}
df["combo"] = (df["scheme"] + "_" + df["feature_set"] + "_"
               + df["model"].map(lambda m: MODEL_ABBR.get(m, m)))
# flag runs where some folds could not be scored, e.g. "month_logo_tax_genus_RF (9/12)"
if {"n_folds_evaluable", "n_folds"} <= set(df.columns):
    short = df["n_folds_evaluable"] < df["n_folds"]
    df.loc[short, "combo"] += (" (" + df.loc[short, "n_folds_evaluable"].astype(int).astype(str)
                               + "/" + df.loc[short, "n_folds"].astype(int).astype(str) + ")")
# samples with this label, e.g. "month_logo_tax_genus_RF n: 15958" (newer sweeps only)
if "n_label" in df.columns:
    has_n = df["n_label"].notna()
    df.loc[has_n, "combo"] += " n: " + df.loc[has_n, "n_label"].astype(int).astype(str)

all_schemes = sorted(s for s in df["scheme"].unique() if s != "random")
RANDOM_COLOR = "#8a8984"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           # overflow hues, only used when more than 8 schemes reach the top N
           "#17becf", "#8c564b", "#bcbd22", "#7f7f7f", "#9467bd", "#aec7e8", "#ff9896", "#c5b0d5"]


def variant_of(label):
    """('geo_', 'ph') for geo_ph; ('', 'ph') for ph."""
    for v in VARIANTS[1:]:
        if label.startswith(v):
            return v, label[len(v):]
    return "", label


def title_of(panel):
    """A paired run is already named <grouping>__<label>; don't repeat it."""
    g, lab = panel
    return g if g.endswith(f"__{lab}") else f"{g}__{lab}"


def prefix(panel):
    return panel[0].split("__")[0]


# A panel is one holdout grouping x label. One row per base label (and per
# extra grouping of it); columns are its plain / geo_ / comb_ versions, so
# depth_category__oxy sits next to ecoregion__geo_oxy, and bioproject's ph
# next to bioproject's geo_ph. A partner with the same grouping prefix wins.
panels = list(dict.fromkeys(zip(df["holdout_grouping"], df["label"])))
present = [v for v in VARIANTS if any(variant_of(lab)[0] == v for _, lab in panels)]
by_base = {}
for panel in panels:
    v, base = variant_of(panel[1])
    by_base.setdefault(base, {u: [] for u in present})[v].append(panel)

# Rows always temperature, ph, oxy (label_order.py), whatever order the
# sweep wrote them in, so two figures compare row for row.
grid = []
for _base, cols in sorted(by_base.items(), key=lambda kv: label_key(kv[0])):
    cols = [list(cols[v]) for v in present]
    while any(cols):
        lead_i = next(i for i, c in enumerate(cols) if c)
        lead = cols[lead_i].pop(0)
        row = [None] * len(cols)
        row[lead_i] = lead
        for j, c in enumerate(cols):
            if j == lead_i or not c:
                continue
            match = next((p for p in c if prefix(p) == prefix(lead)), c[0])
            c.remove(match)
            row[j] = match
        grid.append(row)


def fmt_val(v):
    """Same formatting as microbiome_ml.visualise.visualisations._fmt_r2."""
    if not np.isfinite(v):
        return "n/a"
    return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"


def ink_on(color):
    """Dark or white text, whichever reads on this bar colour."""
    r, g, b = to_rgb(color)
    return "#0b0b0b" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def plot_metric(metric):
    spec = METRICS[metric]
    ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=not spec["higher"])

    # bars per panel: best random run + top N non-random runs
    bars = {}
    for panel, sub in ranked.groupby(["holdout_grouping", "label"], sort=False):
        bars[panel] = pd.concat([sub[sub["scheme"] == "random"].head(1),
                                 sub[sub["scheme"] != "random"].head(args.top or len(sub))])

    # Colour by every scheme in the file, so a scheme keeps its colour across the
    # metric figures; only if that overflows the palette, colour just the plotted ones.
    shown = sorted({s for b in bars.values() for s in b["scheme"] if s != "random"})
    pool = all_schemes if len(all_schemes) <= len(PALETTE) else shown
    if len(pool) > len(PALETTE):
        raise SystemExit(f"{len(shown)} schemes to colour, palette has {len(PALETTE)}; lower --top")
    scheme_color = {s: PALETTE[i] for i, s in enumerate(pool)}

    # Fixed-range metrics: as in Visualiser.plot_cv_bars, values past the range are
    # drawn to the edge with a break mark and their real value printed, so one blown-up
    # run can't flatten the rest. For R² the lower half is dropped when nothing is negative.
    plotted = pd.concat(bars.values())[metric] if bars else pd.Series(dtype=float)
    if spec["limits"]:
        lo, hi = spec["limits"]
        y_bottom = lo if lo < 0 and (plotted < 0).any() else max(lo, 0.0)
        y_top = hi
    span = (y_top - y_bottom) if spec["limits"] else None

    nrow, ncol = len(grid), len(present)
    # panel width grows with the number of bar slots, so --top all stays readable
    n_slots = max((len(b) for b in bars.values()), default=1) if args.top is None else args.top + 1
    many = n_slots > 6
    fs = 7 if many else 9
    panel_w = max(6.0, 0.45 * n_slots + 1.5)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * panel_w, (5.4 if many else 4.4) * nrow),
                             sharey=bool(spec["limits"]), squeeze=False)

    for ax, panel in zip(axes.flat, (p for row in grid for p in row)):
        if panel is None or panel not in bars:
            ax.set_visible(False)
            continue
        plot = bars[panel]
        colors = [RANDOM_COLOR if s == "random" else scheme_color[s] for s in plot["scheme"]]
        x = range(len(plot))
        vals = plot[metric].to_numpy(dtype=float)
        if spec["limits"]:
            drawn = np.clip(vals, y_bottom, y_top)
            ax.set_ylim(y_bottom, y_top)
            ax.set_yticks(np.linspace(y_bottom, y_top, 5))
        else:
            drawn = vals
            ax.margins(y=0.12)
        ax.bar(x, drawn, color=colors, width=0.7, zorder=2)

        for xi, v, dv, c in zip(x, vals, drawn, colors):
            clipped_low = spec["limits"] and v < y_bottom
            clipped_high = spec["limits"] and v > y_top
            if clipped_low or clipped_high:
                # break mark: two surface-coloured slashes near the clipped end
                edge, sign = (y_bottom, 1) if clipped_low else (y_top, -1)
                for off in (0.05, 0.09):
                    ax.plot([xi - 0.3, xi + 0.3],
                            [edge + sign * span * (off - 0.012), edge + sign * span * (off + 0.012)],
                            color="white", lw=2, solid_capstyle="butt", zorder=3)
                # real value inside the bar, rotated, clear of the break mark
                ax.annotate(fmt_val(v), xy=(xi, edge + sign * span * 0.13),
                            xytext=(0, 3 * sign), textcoords="offset points", ha="center",
                            va="bottom" if clipped_low else "top", rotation=90,
                            fontsize=fs, fontweight="bold", color=ink_on(c), zorder=5)
            else:
                up = v >= 0
                ax.annotate(fmt_val(v), xy=(xi, dv), xytext=(0, 3 if up else -3),
                            textcoords="offset points", ha="center",
                            va="bottom" if up else "top", fontsize=fs, color="#0b0b0b", zorder=5)
        ax.set_xticks(list(x))
        ax.set_xlim(-0.6, n_slots - 0.4)  # same bar width even when a panel has fewer bars
        ax.set_xticklabels(plot["combo"], rotation=90 if many else 25,
                           ha="center" if many else "right", fontsize=fs)
        ax.set_title(title_of(panel), fontsize=11, loc="left")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel(spec["ylabel"])
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=True)  # left column can be empty, keep y labels on the others too

    # random only gets a legend entry when some panel actually has a random bar
    has_random = any((b["scheme"] == "random").any() for b in bars.values())
    handles = ([plt.Rectangle((0, 0), 1, 1, color=RANDOM_COLOR)] if has_random else []) + \
              [plt.Rectangle((0, 0), 1, 1, color=scheme_color[s]) for s in shown]
    legend_labels = (["random (control)"] if has_random else []) + shown
    # fixed header space in inches, independent of how many rows there are
    fig_h = fig.get_figheight()
    legend_rows = -(-len(handles) // 6)
    header_in = 0.3 + 0.45 + 0.25 * legend_rows
    which = "all" if args.top is None else f"best {args.top}"
    order = "higher is better" if spec["higher"] else "lower is better"
    fig.suptitle(f"{spec['ylabel']}: random CV vs {which} grouped-CV runs ({order})",
                 y=1 - 0.15 / fig_h, va="top")
    fig.legend(handles, legend_labels, title="scheme",
               loc="upper center", ncol=max(1, min(len(handles), 6)), frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.5 / fig_h))
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig_h))
    out = out_path(metric)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out)


for m in metrics:
    plot_metric(m)
