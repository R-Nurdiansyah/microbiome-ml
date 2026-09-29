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
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_to_final_eval as pipeline  # noqa: E402

from microbiome_ml.train.cv import CrossValidator  # noqa: E402
from microbiome_ml.train.results import CV_Result  # noqa: E402
from microbiome_ml.train.trainer import ModelTrainer  # noqa: E402
from microbiome_ml.visualise.visualisations import Visualiser  # noqa: E402
from microbiome_ml.wrangle.dataset import Dataset  # noqa: E402
from microbiome_ml.wrangle.splits import SplitManager  # noqa: E402

LOGGER = logging.getLogger("steps")

RANDOM = "random"  # directory/CLI name for "no holdout grouping"


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


TAXON_PREFIX = "taxon_"  # dir-safe form of the "taxon:<rank>" token


def holdout_spec(name: str) -> Tuple[str, Optional[str]]:
    """Interpret a sweep token as a holdout strategy.

    random | null | none  -> ("column", None)   stratified random
    taxon:family          -> ("taxon", "family") community membership
    taxon_family          -> ("taxon", "family") (dir-safe spelling)
    <anything else>       -> ("column", <name>) metadata grouping
    """
    text = name.strip()
    if text.lower() in {RANDOM, "null", "none", ""}:
        return ("column", None)
    if text.startswith("taxon:"):
        return ("taxon", text.split(":", 1)[1])
    if text.startswith(TAXON_PREFIX):
        return ("taxon", text[len(TAXON_PREFIX) :])
    return ("column", text)


def spec_dir_name(name: str) -> str:
    """Directory/wildcard-safe name for a sweep token."""
    kind, value = holdout_spec(name)
    if kind == "taxon":
        return f"{TAXON_PREFIX}{value}"
    return RANDOM if value is None else value


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
    dataset = pipeline.build_dataset(cfg)
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
    _write_json(out / "schemes.json", schemes)
    LOGGER.info("split done (grouping=%s): %s", args.grouping, schemes)


def step_cv(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args)
    sec = pipeline._sections(cfg)
    split_cv_cfg = sec["split"].get("cv") or {}
    cv_cfg, out_cfg = sec["cv"], sec["outputs"]
    vis_cfg, shap_cfg = sec["visualise"], sec["shap"]
    holdout_enabled = bool(sec["holdout_evaluation"].get("enabled", True))
    holdout_grouping = spec_dir_name(args.grouping)
    label, scheme = args.label, args.scheme
    run_dir = Path(args.out)
    cv_out, best_out = run_dir / "cv_results", run_dir / "best_models"
    holdout_out = run_dir / "holdout"
    status: Dict[str, Any] = {
        "holdout_grouping": args.grouping,
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


def step_collate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    statuses: List[Dict[str, Any]] = []
    best_frames: List[pl.DataFrame] = []
    all_frames: List[pl.DataFrame] = []
    holdout_frames: List[pl.DataFrame] = []

    for status_file in args.runs:
        status_path = Path(status_file)
        run_dir = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        statuses.append(status)
        if status.get("status") != "ok":
            continue
        grouping = status["holdout_grouping"]
        tag = pl.lit(grouping).alias("holdout_grouping")
        best = _read_csv_or_none(
            run_dir / "best_models" / "best_models_summary.csv"
        )
        if best is not None:
            best_frames.append(
                best.with_columns(tag).select(
                    ["holdout_grouping", *best.columns]
                )
            )
        every = _read_csv_or_none(
            run_dir / "cv_results" / "results_summary.csv"
        )
        if every is not None:
            all_frames.append(
                every.with_columns(tag).select(
                    ["holdout_grouping", *every.columns]
                )
            )
        hold = _read_csv_or_none(run_dir / "holdout" / "holdout_summary.csv")
        if hold is not None:
            holdout_frames.append(hold)

    pl.DataFrame(statuses).write_csv(out / "sweep_status.tsv", separator="\t")
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
    p.add_argument("--out", required=True, help="splits directory")
    p.set_defaults(func=step_split)

    p = sub.add_parser("cv", help="CV (+ holdout eval) for label x scheme")
    _add_common(p)
    p.add_argument("--dataset", required=True, help="prepared dataset dir")
    p.add_argument("--splits", required=True, help="splits directory")
    p.add_argument("--grouping", required=True, help="column or 'random'")
    p.add_argument("--label", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--out", required=True, help="run directory")
    p.set_defaults(func=step_cv)

    p = sub.add_parser("collate", help="stack run summaries into TSVs")
    p.add_argument("--out", required=True, help="sweep root directory")
    p.add_argument(
        "--runs", nargs="+", required=True, help="status.json of every run"
    )
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(func=step_collate)

    args = parser.parse_args()
    pipeline.setup_logging(args.log_level)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)
    pipeline.LOGGER.setLevel(level)
    args.func(args)


if __name__ == "__main__":
    main()
