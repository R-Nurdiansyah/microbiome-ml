#!/usr/bin/env python3
"""Pipeline stages as separate CLI steps, for Snakemake (workflow/Snakefile).

Each step is independent so a scheduler can run the expensive part -- one
CV job per holdout grouping x label x scheme -- in parallel:

    prepare  build the dataset once (inputs, groupings, QC, features), save it
    split    for one holdout grouping: holdout split + CV folds -> CSVs
    cv       for one grouping/label/scheme: CV -> best model -> (holdout eval)
    collate  stack per-run summaries into the sweep TSVs

All steps take the same YAML as scripts/run_cv_to_final_eval.py plus the
same `--set KEY=VALUE` overrides. They reuse that script's functions; the
Snakefile is only orchestration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_to_final_eval as pipeline  # noqa: E402

# Token parsing lives in a dependency-free module so the Snakefile can use
# it without importing microbiome_ml (whose setup_logging() would duplicate
# every Snakemake log line). Re-exported here for callers and tests.
from sweep_tokens import (  # noqa: E402,F401
    RANDOM,
    TAXON_PREFIX,
    holdout_spec,
    spec_dir_name,
)

from microbiome_ml.train.cv import CrossValidator  # noqa: E402
from microbiome_ml.train.results import CV_Result  # noqa: E402
from microbiome_ml.train.trainer import ModelTrainer  # noqa: E402
from microbiome_ml.visualise.visualisations import Visualiser  # noqa: E402
from microbiome_ml.wrangle.dataset import Dataset  # noqa: E402
from microbiome_ml.wrangle.splits import SplitManager  # noqa: E402

LOGGER = logging.getLogger("steps")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = pipeline.load_yaml_config(Path(args.config))
    for item in getattr(args, "set", None) or []:
        key, value = pipeline._parse_override(item)
        pipeline._set_by_path(cfg, key, value)
        LOGGER.info("Config override: %s = %r", key, value)
    return cfg


def _grouping_value(name: str) -> Optional[str]:
    """Metadata grouping column for a token, or None (random / taxon)."""
    kind, value = holdout_spec(name)
    return value if kind == "column" else None


def _ensure_grouping(dataset: Dataset, grouping: Optional[str]) -> Dataset:
    """Pull *grouping* from metadata if it is not a grouping column yet."""
    if grouping is None:
        return dataset
    cols = (
        [c for c in dataset.groupings.columns if c != "sample"]
        if dataset.groupings is not None
        else []
    )
    if grouping in cols:
        return dataset
    LOGGER.info("Grouping '%s' not in groupings table; adding", grouping)
    return dataset.create_default_groupings(
        groupings=[grouping], force=False, strict=True
    )


def _attach_splits(
    dataset: Dataset, splits_dir: Path, label: str, scheme: str
) -> Dataset:
    """Load holdout.csv and cv_<scheme>.csv for *label* onto *dataset*."""
    label_dir = splits_dir / label
    holdout_path = label_dir / "holdout.csv"
    cv_path = label_dir / f"cv_{scheme}.csv"
    if not holdout_path.exists():
        raise FileNotFoundError(holdout_path)
    if not cv_path.exists():
        raise FileNotFoundError(cv_path)
    sm = SplitManager(label)
    sm.holdout = pl.read_csv(holdout_path, infer_schema_length=10000)
    sm.cv_schemes[scheme] = pl.read_csv(cv_path, infer_schema_length=10000)
    dataset.splits = {label: sm}
    LOGGER.info(
        "Attached splits label=%s scheme=%s (%d holdout rows, %d fold rows)",
        label,
        scheme,
        sm.holdout.height,
        sm.cv_schemes[scheme].height,
    )
    return dataset


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------
def step_prepare(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args)
    out = Path(args.out)
    # Feature tables are streamed straight into the dataset directory, so
    # the wide matrices are never held in memory and Dataset.save finds
    # them already in place.
    dataset = pipeline.build_dataset(cfg, stream_dir=out / "features")
    LOGGER.info("Saving prepared dataset to %s", out)
    dataset.save(out)
    LOGGER.info("prepare done: %s", out / "manifest.json")


def step_split(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args)
    sec = pipeline._sections(cfg)
    holdout_cfg = sec["split"].get("holdout") or {}
    split_cv_cfg = sec["split"].get("cv") or {}
    cv_cfg = sec["cv"]
    kind, value = holdout_spec(args.grouping)
    run_name = getattr(args, "run_name", None) or spec_dir_name(args.grouping)
    out = Path(args.out)

    dataset = Dataset.load(args.dataset)

    if kind == "taxon":
        dataset = dataset.create_taxon_holdout_split(
            rank=str(value),
            label=holdout_cfg.get("label"),
            test_size=float(holdout_cfg.get("test_size", 0.2)),
            presence_threshold=float(
                holdout_cfg.get("presence_threshold", 0.0)
            ),
            min_test_samples=int(holdout_cfg.get("min_test_samples", 2)),
            force=True,
            output_dir=out,
        )
    else:
        dataset = _ensure_grouping(dataset, value)
        dataset = dataset.create_holdout_split(
            label=holdout_cfg.get("label"),
            test_size=float(holdout_cfg.get("test_size", 0.2)),
            n_bins=int(holdout_cfg.get("n_bins", 5)),
            grouping=value,
            random_state=int(holdout_cfg.get("random_state", 42)),
            force=True,
            output_dir=out,
        )
    dataset = dataset.create_cv_folds(
        label=split_cv_cfg.get("label"),
        n_folds=int(split_cv_cfg.get("n_folds", 5)),
        n_bins=int(split_cv_cfg.get("n_bins", 5)),
        grouping=split_cv_cfg.get("grouping", "all"),
        random_state=int(split_cv_cfg.get("random_state", 42)),
        use_holdout=bool(split_cv_cfg.get("use_holdout", True)),
        force=True,
        strict=bool(split_cv_cfg.get("strict", True)),
    )
    dataset.save_cv_folds(out)

    # Label distribution either side of the holdout, per label.
    pipeline.plot_holdout_distributions(
        out,
        sec["visualise"],
        out.parent / "split_visualisations",
        holdout_grouping=run_name,
    )

    # What the cv step should actually train, honouring cv.label / cv.scheme.
    want_labels = cv_cfg.get("label")
    want_labels = (
        None
        if want_labels is None
        else set(
            want_labels if isinstance(want_labels, list) else [want_labels]
        )
    )
    want_schemes = cv_cfg.get("scheme")
    want_schemes = (
        None
        if want_schemes is None
        else set(
            want_schemes if isinstance(want_schemes, list) else [want_schemes]
        )
    )
    schemes: Dict[str, List[str]] = {}
    for label, sm in dataset.splits.items():
        if want_labels is not None and label not in want_labels:
            continue
        names = [
            s
            for s in sm.cv_schemes
            if want_schemes is None or s in want_schemes
        ]
        if names:
            schemes[label] = sorted(names)
    if not bool(cv_cfg.get("enabled", True)):
        # Record the schemes even when CV is off: the workflow gates on
        # cv.enabled itself, and writing an empty list here would make
        # schemes.json say "nothing to run" to a later run with CV on
        # (the file already exists, so split would not be redone).
        LOGGER.info(
            "cv.enabled is false: stopping after the split "
            "(schemes recorded for a later run)"
        )
    _write_json(out / "schemes.json", schemes)
    LOGGER.info("split done (grouping=%s): %s", args.grouping, schemes)


def step_cv(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args)
    sec = pipeline._sections(cfg)
    split_cv_cfg = sec["split"].get("cv") or {}
    cv_cfg, out_cfg = sec["cv"], sec["outputs"]
    vis_cfg, shap_cfg = sec["visualise"], sec["shap"]
    holdout_enabled = bool(sec["holdout_evaluation"].get("enabled", True))
    holdout_grouping = getattr(args, "run_name", None) or spec_dir_name(
        args.grouping
    )
    label, scheme = args.label, args.scheme
    run_dir = Path(args.out)
    cv_out, best_out = run_dir / "cv_results", run_dir / "best_models"
    holdout_out = run_dir / "holdout"
    status: Dict[str, Any] = {
        "holdout_grouping": holdout_grouping,
        "label": label,
        "scheme": scheme,
        "status": "ok",
        "error": "",
    }

    dataset = Dataset.load(args.dataset)
    dataset = _attach_splits(dataset, Path(args.splits), label, scheme)

    models = cv_cfg.get("models", ["rf"])
    if not isinstance(models, list):
        raise ValueError("cv.models must be a list (e.g., ['rf', 'xgboost'])")
    cv = CrossValidator(
        dataset=dataset,
        models=models,
        cv_folds=int(split_cv_cfg.get("n_folds", 5)),
        label=label,
        scheme=scheme,
        feature_set=cv_cfg.get("feature_set"),
    )
    mode = str(cv_cfg.get("mode", "run")).strip().lower()
    param_path = str(cv_cfg.get("param_path", "parameters.yaml"))
    n_jobs_raw = cv_cfg.get("n_jobs")
    n_jobs = None if n_jobs_raw is None else int(n_jobs_raw)
    if mode == "run_grid":
        results = cv.run_grid(
            param_path=param_path,
            n_jobs=n_jobs,
            params_per_job=int(cv_cfg.get("params_per_job", 2)),
        )
    elif mode == "run":
        results = cv.run(param_path=param_path, n_jobs=n_jobs)
    else:
        raise ValueError("cv.mode must be either 'run' or 'run_grid'")

    if not results or not cv.best_result_by_label_scheme:
        status["status"] = "empty"
        status["error"] = "CV produced no results (scheme skipped?)"
        LOGGER.warning("%s", status["error"])
        _write_json(run_dir / "status.json", status)
        return

    CV_Result.export_result(
        results,
        cv_out,
        save_models=bool(out_cfg.get("cv_save_models", False)),
        fold_table=bool(out_cfg.get("cv_fold_table", False)),
    )
    CV_Result.export_best_results_by_scheme(
        cv.best_result_by_label_scheme,
        best_out,
        best_result_key_by_label_scheme=cv.best_result_key_by_label_scheme,
    )
    pipeline._log_best_combinations(cv)

    vis_enabled = bool(vis_cfg.get("enabled", False))
    vis_formats = list(vis_cfg.get("formats", ["png"]))
    top_n = int(vis_cfg.get("top_n", 20))
    if vis_enabled:
        cv_vis = Visualiser(run_dir / "cv_visualisations", formats=vis_formats)
        cv_vis.plot_cv_bars(
            results=cv_out, out_dir=run_dir / "cv_visualisations"
        )

    if holdout_enabled:
        shap_enabled = bool(shap_cfg.get("enabled", False))
        shap_top_n = int(shap_cfg.get("top_n", 20))
        trainer = ModelTrainer(
            dataset=dataset,
            best_result=cv.best_result_by_label_scheme,
            output_model_path=holdout_out,
        )
        evaluation = trainer.train_and_evaluate(
            compute_shap=shap_enabled,
            max_shap_background=int(shap_cfg.get("max_background", 100)),
        )
        # One (label, scheme) per job -> a single HoldoutEvaluation.
        if isinstance(evaluation, dict):
            evaluation = next(iter(evaluation.values()))
        key = f"{label}/{scheme}"
        metrics_payload = {
            key: pipeline._metrics_with_shap(evaluation, shap_top_n)
        }
        _write_json(holdout_out / "holdout_metrics.json", metrics_payload)
        pipeline._write_holdout_summary(
            metrics_payload,
            holdout_out / "holdout_summary.csv",
            holdout_grouping,
        )
        if vis_enabled:
            vis = Visualiser(holdout_out, formats=vis_formats)
            tag = key.replace("/", "__")
            sch = evaluation.metrics.get("scheme")
            vis.visualise_model_performance(
                evaluation.predictions,
                evaluation.targets,
                title=f"Holdout diagnostics ({key})",
                groups=[sch] * len(evaluation.targets) if sch else None,
                file_name=f"holdout_diagnostics_{tag}",
            )
            vis.plot_feature_importances(
                evaluation,
                output=f"holdout_feature_importance_{tag}",
                top_n=top_n,
            )
            if evaluation.shap_result is not None:
                pipeline._plot_shap(
                    vis, evaluation.shap_result, shap_top_n, tag=tag, label=key
                )

    _write_json(run_dir / "status.json", status)
    LOGGER.info("cv done: %s", run_dir)


def _read_csv_or_none(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    df = pl.read_csv(path, infer_schema_length=10000)
    return df if df.height else None


def _concat_sorted(
    frames: List[pl.DataFrame],
    sort_by: str,
    out_path: Optional[Path] = None,
) -> Optional[pl.DataFrame]:
    """Concatenate and sort; write to *out_path* only when one is given."""
    if not frames:
        LOGGER.warning("Nothing to concatenate for %s", out_path or sort_by)
        return None
    try:
        df = pl.concat(frames, how="diagonal_relaxed")
    except (TypeError, ValueError):
        df = pl.concat(frames, how="diagonal")
    if sort_by in df.columns:
        df = df.sort(sort_by, descending=True, nulls_last=True)
    if out_path is not None:
        df.write_csv(out_path, separator="\t")
        LOGGER.info("Wrote %s (%d rows)", out_path, df.height)
    return df


def _fmt(value: Any) -> str:
    """Format an R2-ish value for the log; '-' when absent."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


_HEADLINE_NEEDS = (
    "holdout_grouping",
    "label",
    "scheme",
    "feature_set",
    "model",
    "avg_validation_r2",
)


def headline_table(best: pl.DataFrame) -> Optional[pl.DataFrame]:
    """One row per holdout grouping x label: `random` vs the best grouped CV
    scheme, with the feature set and model each of them chose.

    `random` is the positive control (it leaks group structure into the
    validation folds, so it should look good); `best_grouped_*` is the best
    honest scheme. `r2_gap` = control - grouped: large means the signal was
    mostly within-group.
    """
    missing = [c for c in _HEADLINE_NEEDS if c not in best.columns]
    if missing:
        LOGGER.warning("headline table skipped; missing columns %s", missing)
        return None

    df = best.with_columns(
        pl.col("avg_validation_r2").cast(pl.Float64, strict=False)
    )

    def _top(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
        picked = (
            frame.sort("avg_validation_r2", descending=True, nulls_last=True)
            .group_by(["holdout_grouping", "label"], maintain_order=True)
            .first()
        )
        rename = {
            "scheme": f"{prefix}_scheme",
            "feature_set": f"{prefix}_feature_set",
            "model": f"{prefix}_model",
            "avg_validation_r2": f"{prefix}_r2",
        }
        keep = ["holdout_grouping", "label", *rename]
        if "avg_validation_mse" in picked.columns:
            rename["avg_validation_mse"] = f"{prefix}_mse"
            keep.append("avg_validation_mse")
        return picked.select(keep).rename(rename)

    control = _top(df.filter(pl.col("scheme") == "random"), "control")
    grouped = _top(df.filter(pl.col("scheme") != "random"), "best_grouped")

    table = control.join(
        grouped, on=["holdout_grouping", "label"], how="full", coalesce=True
    )
    if {"control_r2", "best_grouped_r2"} <= set(table.columns):
        table = table.with_columns(
            (pl.col("control_r2") - pl.col("best_grouped_r2")).alias("r2_gap")
        )
    order = [
        c
        for c in (
            "holdout_grouping",
            "label",
            "control_scheme",
            "control_feature_set",
            "control_model",
            "control_r2",
            "control_mse",
            "best_grouped_scheme",
            "best_grouped_feature_set",
            "best_grouped_model",
            "best_grouped_r2",
            "best_grouped_mse",
            "r2_gap",
        )
        if c in table.columns
    ]
    return table.select(order).sort(["holdout_grouping", "label"])


# A grouped CV scheme is only interpretable if its folds are usable. A group
# cannot be split, so one oversized group forces a lopsided split: the
# iteration that validates on the big fold trains on whatever is left.
MAX_FOLD_PCT = 50.0  # one fold holding more than this -> lopsided
MIN_TRAIN_N = 30  # worst iteration training on fewer than this -> unusable


def load_sweep_best(root: Union[str, Path]) -> pl.DataFrame:
    """Gather every best_models_summary.csv under a sweep root.

    Handles both layouts, and a sweep that is still running:
        serial:    <root>/holdout_<g>/best_models/best_models_summary.csv
        snakemake: <root>/holdout_<g>/runs/<label>/<scheme>/best_models/...
    `holdout_grouping` is taken from the `holdout_<g>` path component. This is
    the table `write_summaries` needs; it is not written to disk itself
    (cv_sweep_all.tsv is its superset).
    """
    root = Path(root)
    frames: List[pl.DataFrame] = []
    for path in sorted(root.glob("**/best_models_summary.csv")):
        grouping = next(
            (
                p.name[len("holdout_") :]
                for p in path.parents
                if p.name.startswith("holdout_")
            ),
            "unknown",
        )
        df = pl.read_csv(path, infer_schema_length=10000)
        if df.height:
            frames.append(
                df.with_columns(pl.lit(grouping).alias("holdout_grouping"))
            )
    if not frames:
        raise FileNotFoundError(f"no best_models_summary.csv under {root}")
    out = pl.concat(frames, how="diagonal_relaxed")
    rest = [c for c in out.columns if c != "holdout_grouping"]
    return out.select(["holdout_grouping", *rest]).sort(
        "avg_validation_r2", descending=True, nulls_last=True
    )


def fold_profile(root: Union[str, Path]) -> Optional[pl.DataFrame]:
    """Per holdout grouping x label x scheme: how usable the CV folds were.

    Reads the ``cv_<scheme>.csv`` files already written under
    ``<root>/holdout_<g>/splits/<label>/``; nothing is recomputed.

    min_train_n:  samples available to the worst CV iteration (total minus
                  the largest fold, which is validation exactly once).
    max_fold_pct: share of samples in the largest fold.
    verdict:      "ok", or why the scheme's R2 should not be trusted.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(root).glob("holdout_*/splits/*/cv_*.csv")):
        grouping = path.parents[2].name[len("holdout_") :]
        label = path.parent.name
        scheme = path.stem[len("cv_") :]
        df = pl.read_csv(path)
        if "fold" not in df.columns or not df.height:
            continue
        sizes = df.group_by("fold").len().sort("fold")["len"].to_list()
        total = sum(sizes)
        min_train = total - max(sizes)
        max_pct = round(100.0 * max(sizes) / total, 1)
        reasons = []
        if len(sizes) < 2:
            reasons.append("single fold")
        if max_pct > MAX_FOLD_PCT:
            reasons.append(f"fold holds {max_pct}%")
        if min_train < MIN_TRAIN_N:
            reasons.append(f"worst iteration trains on {min_train}")
        rows.append(
            {
                "holdout_grouping": grouping,
                "label": label,
                "scheme": scheme,
                "n_folds": len(sizes),
                "fold_sizes": str(sizes),
                "n_total": total,
                "min_train_n": min_train,
                "max_fold_pct": max_pct,
                "verdict": "; ".join(reasons) if reasons else "ok",
            }
        )
    if not rows:
        LOGGER.warning("No cv_<scheme>.csv found under %s", root)
        return None
    return pl.DataFrame(rows).sort(["holdout_grouping", "label", "scheme"])


def _annotate_headline(
    headline: pl.DataFrame, profile: Optional[pl.DataFrame]
) -> pl.DataFrame:
    """Attach the best-grouped scheme's fold profile to the headline table."""
    if profile is None or "best_grouped_scheme" not in headline.columns:
        return headline
    cols = ["n_folds", "fold_sizes", "min_train_n", "max_fold_pct", "verdict"]
    right = profile.select(
        ["holdout_grouping", "label", "scheme", *cols]
    ).rename(
        {
            "scheme": "best_grouped_scheme",
            **{c: f"best_grouped_{c}" for c in cols},
        }
    )
    return headline.join(
        right,
        on=["holdout_grouping", "label", "best_grouped_scheme"],
        how="left",
    )


def write_summaries(
    best_df: pl.DataFrame, out: Union[str, Path]
) -> Optional[pl.DataFrame]:
    """Write cv_sweep_headline.tsv and cv_sweep_fold_profile.tsv."""
    out = Path(out)
    profile = fold_profile(out)
    if profile is not None:
        profile.write_csv(out / "cv_sweep_fold_profile.tsv", separator="\t")
        LOGGER.info(
            "Wrote %s (%d rows)",
            out / "cv_sweep_fold_profile.tsv",
            profile.height,
        )
    headline = headline_table(best_df)
    if headline is None:
        return None
    headline = _annotate_headline(headline, profile)
    headline.write_csv(out / "cv_sweep_headline.tsv", separator="\t")
    LOGGER.info(
        "Wrote %s (%d rows)", out / "cv_sweep_headline.tsv", headline.height
    )
    return headline


def _num(value: Any, width: int, spec: str) -> str:
    """Right-aligned number for the text report; 'n/a' when unusable.

    Values can arrive as strings when a per-run stats file was written
    with an all-null column, so never assume the type.
    """
    if value is None:
        return "n/a".rjust(width)
    try:
        number = int(value) if spec == "d" else float(value)
    except (TypeError, ValueError):
        return str(value).rjust(width)
    if number != number:  # NaN
        return "n/a".rjust(width)
    return f"{number:>{width}{spec}}"


def collate_split_report(root: Union[str, Path]) -> Optional[Path]:
    """One ranked text report of every holdout's train/test agreement.

    Reads the per-run holdout_distribution_stats.tsv files and writes
    `holdout_distribution_summary.txt` at the sweep root: per label, the
    holdout strategies sorted by KS (smallest first = train and test most
    alike), so the figures only need opening for the interesting ones.
    """
    root = Path(root)
    files = sorted(root.glob("holdout_*/split_visualisations/*_stats.tsv"))
    if not files:
        LOGGER.warning("No holdout_distribution_stats.tsv under %s", root)
        return None
    frames = [pl.read_csv(f, separator="\t") for f in files]
    df = pl.concat(frames, how="diagonal_relaxed")
    # A run whose stats are all-null reads back as String, and the relaxed
    # concat then widens the whole column to String -- so re-impose the
    # numeric types before sorting or formatting them.
    numeric = ["ks", "p", "median_train", "median_test", "sd_train", "sd_test"]
    counts = ["n_train", "n_test"]
    df = df.with_columns(
        [
            pl.col(c).cast(pl.Float64, strict=False)
            for c in numeric
            if c in df.columns
        ]
        + [
            pl.col(c).cast(pl.Int64, strict=False).fill_null(0)
            for c in counts
            if c in df.columns
        ]
    )
    tsv = root / "holdout_distribution_stats.tsv"
    df.write_csv(tsv, separator="\t")

    lines = [
        "Holdout train/test label agreement",
        "=" * 78,
        "KS = largest gap between the train and test cumulative label",
        "distributions (0 = identical). p = chance of a gap that large if",
        "both sides came from the same distribution.",
        "",
        "A random holdout SHOULD agree (small KS, large p). A grouped or",
        "taxon holdout is expected to disagree -- whole groups are held out,",
        "and groups differ in the label; there KS measures how far the model",
        "has to extrapolate, it is not a defect.",
        "",
    ]
    for label in sorted(df["label"].unique().to_list()):
        sub = df.filter(pl.col("label") == label).sort(
            "ks", descending=False, nulls_last=True
        )
        lines.append(f"--- {label} " + "-" * max(0, 74 - len(label)))
        lines.append(
            f"{'holdout':<22}{'KS':>8}{'p':>12}{'n_train':>9}"
            f"{'n_test':>8}{'med_train':>11}{'med_test':>10}"
        )
        for r in sub.to_dicts():
            lines.append(
                f"{str(r.get('holdout_grouping')):<22}"
                f"{_num(r.get('ks'), 8, '.3f')}"
                f"{_num(r.get('p'), 12, '.3g')}"
                f"{_num(r.get('n_train'), 9, 'd')}"
                f"{_num(r.get('n_test'), 8, 'd')}"
                f"{_num(r.get('median_train'), 11, '.3g')}"
                f"{_num(r.get('median_test'), 10, '.3g')}"
            )
        lines.append("")

    out = root / "holdout_distribution_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    LOGGER.info("\n%s", "\n".join(lines))
    return out


def step_split_report(args: argparse.Namespace) -> None:
    collate_split_report(Path(args.out))


def _grouping_from_path(path: Path) -> str:
    """Run name from a path inside <root>/holdout_<run>/..."""
    for parent in path.parents:
        if parent.name.startswith("holdout_"):
            return parent.name[len("holdout_") :]
    return "unknown"


def _gather(root: Path, pattern: str, stamp: bool) -> List[pl.DataFrame]:
    """Read every file matching *pattern* under the sweep root.

    Globbing the directories rather than reading only the runs of this
    invocation means a later sweep that adds a grouping extends the
    summaries instead of replacing them -- earlier runs are still on
    disk, so they are still collated.
    """
    frames: List[pl.DataFrame] = []
    for path in sorted(root.glob(pattern)):
        frame = _read_csv_or_none(path)
        if frame is None:
            continue
        if stamp and "holdout_grouping" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(_grouping_from_path(path)).alias("holdout_grouping")
            ).select(["holdout_grouping", *frame.columns])
        frames.append(frame)
    return frames


def step_collate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    collate_split_report(out)

    # Every run directory present, not just the ones this invocation
    # happened to execute, so reruns accumulate.
    statuses: List[Dict[str, Any]] = []
    for status_path in sorted(out.glob("holdout_*/**/status.json")):
        try:
            statuses.append(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", status_path, exc)
    if args.runs:
        LOGGER.info(
            "Collating %d run(s) on disk (%d ran in this invocation)",
            len(statuses),
            len(args.runs),
        )

    best_frames = _gather(out, "holdout_*/**/best_models_summary.csv", True)
    all_frames = _gather(out, "holdout_*/**/results_summary.csv", True)
    holdout_frames = _gather(out, "holdout_*/**/holdout_summary.csv", False)

    if statuses:
        pl.DataFrame(statuses).write_csv(
            out / "sweep_status.tsv", separator="\t"
        )
    # Not written out: it is cv_sweep_all.tsv reduced to the top row per
    # grouping/label/scheme, and cv_sweep_headline.tsv is built from it.
    best_df = _concat_sorted(best_frames, "avg_validation_r2")
    _concat_sorted(all_frames, "avg_validation_r2", out / "cv_sweep_all.tsv")
    if holdout_frames:
        _concat_sorted(holdout_frames, "r2", out / "holdout_sweep_summary.tsv")

    bad = [s for s in statuses if s.get("status") != "ok"]
    if bad:
        LOGGER.warning(
            "%d run(s) not ok (see sweep_status.tsv): %s",
            len(bad),
            [(s["holdout_grouping"], s["label"], s["scheme"]) for s in bad],
        )
    if best_df is not None:
        headline = write_summaries(best_df, out)
        if headline is not None:
            LOGGER.info(
                "random (control) vs best grouped scheme, per "
                "holdout grouping x label "
                "[folds: how usable that grouped split was]:"
            )
            for row in headline.iter_rows(named=True):
                LOGGER.info(
                    "  holdout=%-14s label=%-12s | control %s/%s r2=%s | "
                    "best grouped %s %s/%s r2=%s | gap=%s | folds=%s %s",
                    row.get("holdout_grouping"),
                    row.get("label"),
                    row.get("control_feature_set"),
                    row.get("control_model"),
                    _fmt(row.get("control_r2")),
                    row.get("best_grouped_scheme"),
                    row.get("best_grouped_feature_set"),
                    row.get("best_grouped_model"),
                    _fmt(row.get("best_grouped_r2")),
                    _fmt(row.get("r2_gap")),
                    row.get("best_grouped_fold_sizes"),
                    row.get("best_grouped_verdict"),
                )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Pipeline YAML")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override (same syntax as run_cv_to_final_eval.py)",
    )
    p.add_argument("--log-level", default="INFO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)

    p = sub.add_parser("prepare", help="build and save the dataset once")
    _add_common(p)
    p.add_argument("--out", required=True, help="directory for Dataset.save")
    p.set_defaults(func=step_prepare)

    p = sub.add_parser("split", help="holdout split + CV folds for a grouping")
    _add_common(p)
    p.add_argument("--dataset", required=True, help="prepared dataset dir")
    p.add_argument("--grouping", required=True, help="column or 'random'")
    p.add_argument(
        "--run-name",
        default=None,
        help="Name this run is stamped with in every summary table "
        "(default: the grouping). The sweep uses it to keep a "
        "label-paired run distinct from a plain grouping run.",
    )
    p.add_argument("--out", required=True, help="splits directory")
    p.set_defaults(func=step_split)

    p = sub.add_parser("cv", help="CV (+ holdout eval) for label x scheme")
    _add_common(p)
    p.add_argument("--dataset", required=True, help="prepared dataset dir")
    p.add_argument("--splits", required=True, help="splits directory")
    p.add_argument("--grouping", required=True, help="column or 'random'")
    p.add_argument("--run-name", default=None, help="stamp for the tables")
    p.add_argument("--label", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--out", required=True, help="run directory")
    p.set_defaults(func=step_cv)

    p = sub.add_parser("collate", help="stack run summaries into TSVs")
    p.add_argument("--out", required=True, help="sweep root directory")
    p.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="status.json of the runs this invocation produced. Used for "
        "ordering only -- the summaries are built from every run "
        "directory present, so earlier sweeps keep contributing.",
    )
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(func=step_collate)

    p = sub.add_parser(
        "split_report",
        help="rank every holdout by train/test label agreement (KS)",
    )
    p.add_argument("--out", required=True, help="sweep root directory")
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(func=step_split_report)

    args = parser.parse_args()
    pipeline.setup_logging(args.log_level)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)
    pipeline.LOGGER.setLevel(level)
    args.func(args)


if __name__ == "__main__":
    main()
