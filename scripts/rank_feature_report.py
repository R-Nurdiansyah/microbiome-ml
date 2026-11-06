#!/usr/bin/env python3
"""Two questions about features, answered from CV output already on disk.

Nothing is refitted and no model is unpickled; this only reads the CSVs
the pipeline wrote, so it is safe to point at a finished (or still
running) sweep and cheap to re-run.

(a) Is finer taxonomic resolution earning its cost?
    For each label/scheme/model, line the feature sets up by rank and
    compare `pooled_validation_r2`. If genus's 59k features do not beat
    order's 1.5k, the extra resolution is noise -- and no feature
    selection machinery is needed to find that out.

(b) How concentrated is importance?
    For each CV combination, how many features carry 50/90/95% of the
    total importance: how many could be dropped, and which taxa would
    survive. This is the evidence for (or against) a feature cap.

IMPORTANT, about what `importance` is here. It is NOT averaged over the
CV folds. `run_cross_validation` calls `cross_validate` for the scores
and then refits the estimator once on every CV row (`estimator.fit`);
`feature_importances_` is read off that single refit, and the fold models
are discarded. So these numbers describe one model trained on all the CV
data, and say nothing about whether the same taxa would be picked in
every fold. Measuring that needs `return_estimator=True` -- see the
`--help` epilog.

Example:
    python scripts/rank_feature_report.py \\
        --root out/pipeline \\
        --out out/pipeline/feature_report \\
        --coverage 0.5 0.9 0.95 --top 25

Outputs under --out:
    rank_comparison.tsv            pooled R2 per rank x variant
    importance_concentration.tsv   how many features carry the signal
    importance_kept_taxa.tsv       the surviving taxa, named, with shares
    shadow_floor.tsv               the noise floor, when shadows were run
    feature_report.txt             both tables, readable
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import polars as pl

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.dataset import rank_from_feature_set
from microbiome_ml.wrangle.shadow import SHADOW_PREFIX, shadow_floor

LOGGER = logging.getLogger("feature-report")

# What identifies one CV combination in both CSVs the pipeline writes.
KEYS = ["holdout_grouping", "label", "scheme", "model", "feature_set"]

# A feature set not named for a taxonomic rank sorts last rather than
# breaking the report.
UNKNOWN_DEPTH = 99


def _grouping_from(path: Path) -> str:
    """The sweep's holdout grouping, read from the `holdout_<g>` parent."""
    for parent in path.parents:
        if parent.name.startswith("holdout_"):
            return parent.name[len("holdout_") :]
    return "-"


def _cv_exports(root: Path, filename: str) -> List[Path]:
    """Only the CV exports, which is what this report is about.

    The same basename is written in three places per (label,
    scheme): `cv_results/` holds every CV combination,
    `best_models/<label>/<scheme>/` repeats the winning row, and --
    once the holdout stage runs -- `holdout/` holds final-test rows.
    A plain `**/` glob takes all three, which double-counts the
    winner and mixes the holdout in. The holdout table's
    `pooled_validation_r2` is empty, an all-empty column parses as
    text, and a text column makes `max()` lexicographic -- so the
    report would name the wrong best rank even where it did not
    crash on the arithmetic.
    """
    paths = sorted(root.glob(f"**/{filename}"))
    canonical = [p for p in paths if p.parent.name == "cv_results"]
    if canonical:
        return canonical
    # An older or hand-assembled tree may not have cv_results/ at
    # all; take what there is, minus the holdout copies, rather
    # than reporting nothing.
    other = [p for p in paths if p.parent.name != "holdout"]
    if other and len(other) != len(paths):
        LOGGER.warning(
            "No cv_results/%s under %s; using %d other copy(ies) "
            "and ignoring the holdout tables.",
            filename,
            root,
            len(other),
        )
    return other


def _load(root: Path, filename: str) -> pl.DataFrame:
    """Every CV `filename` under `root`, stamped with its holdout grouping."""
    frames = []
    for path in _cv_exports(root, filename):
        df = pl.read_csv(path, infer_schema_length=10000)
        if df.height:
            frames.append(
                df.with_columns(
                    pl.lit(_grouping_from(path)).alias("holdout_grouping")
                )
            )
    if not frames:
        raise FileNotFoundError(f"no {filename} under {root}")
    LOGGER.info("Read %d x %s under %s", len(frames), filename, root)
    return pl.concat(frames, how="diagonal_relaxed")


def split_feature_set(name: str, prefix: str) -> Tuple[Optional[str], str]:
    """`tax_genus_clr` -> (`genus`, `clr`); unrecognised -> (None, '-')."""
    rank = rank_from_feature_set(name, prefix)
    if rank is None:
        return None, "-"
    tail = name[len(prefix) + 1 + len(rank) :].lstrip("_")
    return rank, tail or "-"


def _depth(rank: Optional[str]) -> int:
    if not rank:
        return UNKNOWN_DEPTH
    try:
        return int(TaxonomicRanks.from_name(rank).value)
    except ValueError:
        return UNKNOWN_DEPTH


def annotate_ranks(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Add `tax_rank`, `variant` and `rank_depth` from `feature_set`."""
    parsed = [
        split_feature_set(str(n), prefix)
        for n in df.get_column("feature_set").to_list()
    ]
    return df.with_columns(
        pl.Series("tax_rank", [r for r, _ in parsed], dtype=pl.Utf8),
        pl.Series("variant", [v for _, v in parsed], dtype=pl.Utf8),
        pl.Series("rank_depth", [_depth(r) for r, _ in parsed]),
    )


def _short_model(name: Any) -> str:
    """`RandomForestRegressor` -> `RandomForest`, to keep the table narrow.

    Only the task suffix is dropped, never the algorithm, so the two
    models of one combination stay distinguishable.
    """
    text = str(name)
    for suffix in ("Regressor", "Classifier"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _cell(value: Any, width: int, spec: str = "") -> str:
    """Right-aligned number, or a right-aligned 'n/a' when unusable."""
    if value is None:
        return "n/a".rjust(width)
    try:
        return f"{value:>{width}{spec}}"
    except (TypeError, ValueError):
        return str(value).rjust(width)


# ----------------------------------------------------------------- (a)


def rank_comparison(summaries: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Pooled R2 per feature set, ordered coarse -> fine within a scheme.

    A parameter grid puts several rows under one model name, so the best
    of them is taken: the question is what the rank can do, not what an
    arbitrary setting did.
    """
    score = (
        "pooled_validation_r2"
        if "pooled_validation_r2" in summaries.columns
        else "avg_validation_r2"
    )
    if score != "pooled_validation_r2":
        LOGGER.warning(
            "No pooled_validation_r2 column; falling back to %s. These "
            "results predate the pooled scorers -- re-run the collate.",
            score,
        )
    extras = [
        c
        for c in ("n_label", "n_folds_evaluable", "n_folds")
        if c in summaries.columns
    ]
    best = (
        annotate_ranks(summaries, prefix)
        # Cast before the max, not after: one CSV with an empty
        # score column is enough to make the whole concatenated
        # column text, and max() on text is alphabetical -- which
        # ranks "-6.3" above "0.2". strict=False sends anything
        # genuinely unparseable to null instead of raising.
        .with_columns(pl.col(score).cast(pl.Float64, strict=False))
        .group_by([*KEYS, "tax_rank", "variant", "rank_depth"])
        .agg(
            [pl.col(score).max().alias("score")]
            + [pl.col(c).first() for c in extras]
        )
    )
    return best.sort(
        ["holdout_grouping", "label", "scheme", "model", "rank_depth"]
    )


def _rank_report(table: pl.DataFrame, score_name: str) -> List[str]:
    out: List[str] = [
        "(a) Does finer taxonomic resolution help?",
        f"    score = {score_name}; coarse -> fine within each block.",
        "",
    ]
    group_cols = ["holdout_grouping", "label", "scheme", "model"]
    for keys, block in table.group_by(group_cols, maintain_order=True):
        out.append(
            "  " + "  ".join(f"{c}={k}" for c, k in zip(group_cols, keys))
        )
        seen = block.get_column("score").to_list()
        scores = [s for s in seen if s is not None]
        best = max(scores) if scores else None
        out.append(
            f"    {'rank':<10}{'variant':<10}{'score':>10}"
            f"{'n_label':>10}{'folds':>10}"
        )
        for row in block.iter_rows(named=True):
            value = row.get("score")
            folds = (
                f"{row.get('n_folds_evaluable')}/{row.get('n_folds')}"
                if row.get("n_folds") is not None
                else "-"
            )
            out.append(
                f"    {str(row.get('tax_rank') or '-'):<10}"
                + f"{str(row.get('variant')):<10}"
                + _cell(value, 10, ".3f")
                + _cell(row.get("n_label"), 10, ",")
                + folds.rjust(10)
                + ("  <- best" if best is not None and value == best else "")
            )
        ordered = block.filter(pl.col("score").is_not_null())
        if ordered.height >= 2:
            first = ordered.get_column("score")[0]
            last = ordered.get_column("score")[-1]
            out.append(
                f"    coarsest -> finest: {last - first:+.3f} "
                f"({ordered.get_column('tax_rank')[0]} -> "
                f"{ordered.get_column('tax_rank')[-1]})"
            )
        out.append("")
    return out


# ----------------------------------------------------------------- (b)


def concentration(
    importances: pl.DataFrame, coverages: List[float], prefix: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """How few features carry the importance, and which ones they are.

    Returns (per-combination summary, the taxa kept at the LARGEST
    coverage asked for). Importances are averaged when a parameter grid
    produced several models with the same name, which is the only way
    the key columns can repeat.

    Shadow columns are excluded here. They are deliberately meaningless,
    so counting them would inflate `n_zero` and shift every coverage
    threshold, and the numbers would stop being comparable with a run
    that had no shadows. They get their own section instead.
    """
    importances = importances.filter(
        ~pl.col("feature").str.starts_with(SHADOW_PREFIX)
    )
    per_feature = (
        importances.group_by([*KEYS, "feature"])
        .agg(
            pl.col("importance").mean().alias("importance"),
            pl.len().alias("n_models_merged"),
        )
        .sort([*KEYS, "importance"], descending=[False] * len(KEYS) + [True])
    )

    # The total comes from a window rather than a join: a join may
    # reorder rows, and the cumulative sum below depends on the sort.
    # A model that split on nothing has no distribution to describe,
    # and dividing by its total would make every share NaN.
    per_feature = per_feature.with_columns(
        pl.col("importance").sum().over(KEYS).alias("total_importance")
    )
    dead = (
        per_feature.filter(pl.col("total_importance") <= 0)
        .select(KEYS)
        .unique()
    )
    if dead.height:
        LOGGER.warning(
            "%d combination(s) have zero total importance, skipped: %s",
            dead.height,
            dead.rows(),
        )
    per_feature = per_feature.filter(
        pl.col("total_importance") > 0
    ).with_columns(
        (pl.col("importance").cum_sum() / pl.col("total_importance"))
        .over(KEYS)
        .alias("cum_share"),
        pl.col("importance").cum_count().over(KEYS).alias("imp_rank"),
    )

    summary = per_feature.group_by(KEYS).agg(
        pl.len().alias("n_features"),
        (pl.col("importance") > 0).sum().alias("n_nonzero"),
        (pl.col("importance") <= 0).sum().alias("n_zero"),
        pl.col("n_models_merged").max().alias("n_models_merged"),
    )
    for cov in sorted(coverages):
        tag = f"n_for_{int(round(cov * 100))}pct"
        reached = (
            per_feature.filter(pl.col("cum_share") >= cov)
            .group_by(KEYS)
            .agg(pl.col("imp_rank").min().alias(tag))
        )
        summary = summary.join(reached, on=KEYS, how="left")

    tag = f"n_for_{int(round(max(coverages) * 100))}pct"
    summary = summary.with_columns(
        (pl.col("n_features") - pl.col(tag)).alias("n_dropped"),
        (100.0 * (pl.col("n_features") - pl.col(tag)) / pl.col("n_features"))
        .round(1)
        .alias("pct_dropped"),
    )
    summary = annotate_ranks(summary, prefix).sort(
        ["holdout_grouping", "label", "scheme", "rank_depth"]
    )

    kept = (
        per_feature.join(summary.select([*KEYS, tag]), on=KEYS, how="inner")
        .filter(pl.col("imp_rank") <= pl.col(tag))
        .drop(tag)
        .sort([*KEYS, "imp_rank"])
    )
    return summary, kept


def shadow_report(
    importances: pl.DataFrame,
) -> Tuple[Optional[pl.DataFrame], List[str]]:
    """(c) What does a feature earn for nothing?

    A shadow is a real column shuffled across samples, so any importance
    it collects is what the model hands out to noise. Features ranked
    below the shadows are not evidence of anything.
    """
    frames = []
    for keys, block in importances.group_by(KEYS, maintain_order=True):
        row = shadow_floor(block)
        if row is None:
            continue
        frames.append(
            row.with_columns(
                [pl.lit(v).alias(k) for k, v in zip(KEYS, keys)]
            ).select([*KEYS, *row.columns])
        )
    if not frames:
        return None, []

    table = pl.concat(frames, how="diagonal_relaxed")
    lines = [
        "",
        "(c) What does a feature earn for nothing?",
        "",
        "    Shadow columns are real columns shuffled across samples: same",
        "    values, same sparsity, no relationship to the label. Whatever",
        "    importance they collect is the noise floor. `real>max` counts",
        "    real features scoring above the BEST shadow; `real>q95` above",
        "    the 95th percentile of them.",
        "",
        "    With fewer shadows than real columns the best shadow is the max",
        "    of fewer draws from the null, so `real>max` over-counts. Treat",
        "    `real>q95` as the conservative read, and raise features.shadow.n",
        "    if the two disagree.",
        "",
        f"    {'label':<16}{'scheme':<17}{'model':<14}"
        f"{'n_real':>8}{'n_shad':>8}{'shad_max':>10}"
        f"{'real>max':>10}{'real>q95':>10}{'shad_share':>12}",
    ]
    for row in table.sort(["label", "scheme", "model"]).iter_rows(named=True):
        share = row.get("shadow_total_share")
        lines.append(
            f"    {str(row['label'])[:15]:<16}"
            f"{str(row['scheme'])[:16]:<17}"
            f"{_short_model(row['model'])[:13]:<14}"
            f"{row['n_real']:>8,}{row['n_shadow']:>8,}"
            f"{row.get('shadow_max', 0.0):>10.5f}"
            f"{row.get('n_real_above_max', 0):>10,}"
            f"{row.get('n_real_above_q95', 0):>10,}"
            f"{(f'{share:.2%}' if share is not None else '-'):>12}"
        )
    lines += [
        "",
        "    Reading it: `real>max` far below the feature count means most",
        "    columns are indistinguishable from shuffled noise -- the useful",
        "    set is that small. `real>max` near the feature count, or a",
        "    shadow share well above n_shad/n_real, means the model is",
        "    spreading importance thinly over everything, which is what a",
        "    forest does when no feature predicts the target.",
    ]
    return table, lines


def _concentration_report(
    summary: pl.DataFrame,
    kept: pl.DataFrame,
    coverages: List[float],
    top: int,
    scored: Optional[pl.DataFrame],
) -> List[str]:
    tags = [f"n_for_{int(round(c * 100))}pct" for c in sorted(coverages)]
    out: List[str] = [
        "(b) How concentrated is importance?",
        "    From one model refit on all CV rows -- NOT averaged over "
        "folds.",
        "",
        "    "
        + f"{'label':<16}{'scheme':<17}{'model':<14}"
        + f"{'rank':<9}{'variant':<8}"
        + "".join(t.rjust(13) for t in tags)
        + f"{'n_features':>12}{'n_zero':>9}{'score':>9}",
    ]
    scores = {}
    if scored is not None:
        for row in scored.iter_rows(named=True):
            scores[tuple(row[k] for k in KEYS)] = row.get("score")
    for row in summary.iter_rows(named=True):
        score = scores.get(tuple(row[k] for k in KEYS))
        out.append(
            "    "
            + f"{str(row['label'])[:15]:<16}{str(row['scheme'])[:16]:<17}"
            + f"{_short_model(row['model'])[:13]:<14}"
            + f"{str(row.get('tax_rank') or '-'):<9}"
            + f"{str(row.get('variant')):<8}"
            + "".join(_cell(row.get(t), 13, ",") for t in tags)
            + _cell(row["n_features"], 12, ",")
            + _cell(row["n_zero"], 9, ",")
            + _cell(score, 9, ".3f")
            + ("" if score is None or score > 0 else "   not predictive")
        )
    out += [
        "",
        f"    Top {top} features per combination "
        "(full list in importance_kept_taxa.tsv):",
        "",
    ]
    for keys, block in kept.group_by(KEYS, maintain_order=True):
        out.append("      " + "  ".join(str(k) for k in keys))
        for row in block.head(top).iter_rows(named=True):
            out.append(
                f"        {row['imp_rank']:>4}. "
                f"{str(row['feature'])[:68]:<68} "
                f"{row['importance']:.5f}  cum {row['cum_share']:.3f}"
            )
        out.append("")
    return out


# ---------------------------------------------------------------- main


def run_report(
    root: Path,
    out_dir: Path,
    coverages: List[float],
    top: int,
    prefix: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load(root, "results_summary.csv")
    score_name = (
        "pooled_validation_r2"
        if "pooled_validation_r2" in summaries.columns
        else "avg_validation_r2"
    )
    ranks = rank_comparison(summaries, prefix)
    ranks.write_csv(out_dir / "rank_comparison.tsv", separator="\t")
    LOGGER.info(
        "Wrote %s (%d rows)", out_dir / "rank_comparison.tsv", ranks.height
    )

    lines = ["=" * 78, "Feature report", "=" * 78, ""]
    lines += _rank_report(ranks, score_name)

    try:
        importances = _load(root, "feature_importances.csv")
    except FileNotFoundError:
        LOGGER.warning(
            "No feature_importances.csv under %s -- part (b) skipped. "
            "Only models exposing feature_importances_ write one.",
            root,
        )
        lines.append("(b) skipped: no feature_importances.csv found.")
    else:
        summary, kept = concentration(importances, coverages, prefix)
        summary.write_csv(
            out_dir / "importance_concentration.tsv", separator="\t"
        )
        kept.write_csv(out_dir / "importance_kept_taxa.tsv", separator="\t")
        LOGGER.info(
            "Wrote %s (%d rows) and %s (%d rows)",
            out_dir / "importance_concentration.tsv",
            summary.height,
            out_dir / "importance_kept_taxa.tsv",
            kept.height,
        )
        lines += _concentration_report(summary, kept, coverages, top, ranks)

        floor, floor_lines = shadow_report(importances)
        if floor is None:
            LOGGER.info(
                "No shadow columns in feature_importances.csv -- part (c) "
                "skipped. Set features.shadow.enabled and run CV on the "
                "'_shad' feature set to get a noise floor."
            )
        else:
            floor.write_csv(out_dir / "shadow_floor.tsv", separator="	")
            LOGGER.info(
                "Wrote %s (%d rows)",
                out_dir / "shadow_floor.tsv",
                floor.height,
            )
            lines += floor_lines

    report = out_dir / "feature_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Not answered here: whether the same taxa come out on top in "
            "every CV fold. cross_validate discards its fold models, so "
            "that needs return_estimator=True in run_cross_validation -- "
            "the same plumbing a SHAP-stability analysis would need."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="sweep or run directory to search recursively",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <root>/feature_report)",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        nargs="+",
        default=[0.5, 0.9, 0.95],
        help="importance shares to report; the largest drives n_dropped "
        "and the kept-taxa list",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="features printed per combination",
    )
    parser.add_argument(
        "--prefix",
        default="tax",
        help="feature set prefix used at prepare time",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bad = [c for c in args.coverage if not 0 < c <= 1]
    if bad:
        parser.error(f"--coverage must be in (0, 1], got {bad}")

    out = run_report(
        root=args.root,
        out_dir=args.out or args.root / "feature_report",
        coverages=list(args.coverage),
        top=int(args.top),
        prefix=str(args.prefix),
    )
    LOGGER.info("Done. Report: %s", out)


if __name__ == "__main__":
    main()
