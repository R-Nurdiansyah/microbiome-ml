#!/usr/bin/env python3
"""Run microbiome ML pipeline from CV to final holdout evaluation.

This script is designed for large-data runs outside notebooks.

Example:
    pixi run python scripts/run_cv_to_final_eval.py --config pipeline.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

try:
    import microbiome_ml  # noqa: F401
except ModuleNotFoundError:
    # Fallback for running from a checkout where the package has not been
    # installed into the environment yet (e.g. before `pixi install`).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microbiome_ml.train.cv import CrossValidator  # noqa: E402
from microbiome_ml.train.results import (  # noqa: E402
    CV_Result,
    HoldoutEvaluation,
)
from microbiome_ml.train.trainer import ModelTrainer  # noqa: E402
from microbiome_ml.utils.logging import setup_logging  # noqa: E402
from microbiome_ml.visualise.visualisations import Visualiser  # noqa: E402
from microbiome_ml.wrangle.dataset import Dataset  # noqa: E402

LOGGER = logging.getLogger("pipeline")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


_HOLDOUT_SUMMARY_METRICS = ("r2", "q2", "mse", "mae", "pcc", "pval", "n_test")


def _metrics_with_shap(ev: HoldoutEvaluation, top_n: int) -> Dict[str, Any]:
    """Return ``ev.metrics`` plus the top-N SHAP features when SHAP ran."""
    metrics: Dict[str, Any] = dict(ev.metrics)
    shap_result = getattr(ev, "shap_result", None)
    if shap_result is not None and hasattr(shap_result, "top_features"):
        metrics["shap_top_features"] = list(shap_result.top_features(top_n))
    if shap_result is not None and hasattr(shap_result, "top_features_signed"):
        # direction: Spearman(feature value, SHAP value); >0 means higher
        # abundance pushes the prediction up, <0 pushes it down.
        metrics["shap_top_features_direction"] = {
            name: (None if direction != direction else round(direction, 4))
            for name, direction in shap_result.top_features_signed(top_n)
        }
    return metrics


def _plot_shap(
    vis: Visualiser,
    shap_result: Any,
    top_n: int,
    tag: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """Write the direction-coloured bar chart and the beeswarm.

    The bar shows magnitude (mean |SHAP|) with sign from ``direction``; the
    beeswarm shows every sample so the sign and spread are visible directly.
    """
    suffix = f"_{tag}" if tag else ""
    title_tail = f" ({label})" if label else ""
    vis.plot_shap_summary(
        shap_result,
        style="bar",
        top_n=top_n,
        output=f"holdout_shap_bar{suffix}",
        title=f"SHAP feature importance{title_tail}",
    )
    vis.plot_shap_summary(
        shap_result,
        style="beeswarm",
        top_n=top_n,
        output=f"holdout_shap_beeswarm{suffix}",
        title=f"SHAP values per sample{title_tail}",
    )


def _log_best_combinations(cv: CrossValidator) -> None:
    """Log the CV winner per label/scheme, best -> worst, for quick reading."""
    if not cv.best_result_by_label_scheme:
        return
    LOGGER.info("Best CV combination per label/scheme (by avg validation R2):")
    for (label, scheme), result in CV_Result._sorted_by_avg_r2(
        cv.best_result_by_label_scheme
    ):
        model = result.model
        r2 = result.avg_validation_r2
        LOGGER.info(
            "  label=%-14s scheme=%-14s model=%-24s feature_set=%-14s "
            "avg_r2=%s",
            label,
            scheme,
            model.__class__.__name__ if model is not None else "None",
            result.feature_set,
            f"{r2:.4f}" if r2 is not None else "None",
        )


def _validate_grouping_references(
    cfg: Dict[str, Any], available: List[str]
) -> None:
    """Fail fast if split/cv config names a grouping that does not exist.

    Runs right after the groupings table is built, before the (slow)
    feature-engineering stage, so a typo costs seconds rather than minutes.
    """
    split_cfg = cfg.get("split") or {}
    holdout_g = (split_cfg.get("holdout") or {}).get("grouping")
    cv_g = (split_cfg.get("cv") or {}).get("grouping", "all")
    schemes = (cfg.get("cv") or {}).get("scheme")

    def _check(name: Any, where: str, extra_ok: Sequence[str] = ()) -> None:
        if name is None or name in extra_ok:
            return
        if not isinstance(name, str):
            raise ValueError(
                f"{where} must be a single column name, got {name!r}. "
                "To compare several groupings run the pipeline once per "
                "grouping (holdout) or list them under cv.scheme (CV)."
            )
        if name not in available:
            raise ValueError(
                f"{where} = '{name}' is not a grouping column. "
                f"Available: {available}. Add it via groupings.columns "
                "(a metadata column) or groupings.file (a CSV)."
            )

    _check(holdout_g, "split.holdout.grouping")
    _check(cv_g, "split.cv.grouping", extra_ok=("all", "random"))
    if schemes is not None:
        scheme_list = schemes if isinstance(schemes, list) else [schemes]
        for s in scheme_list:
            _check(s, "cv.scheme entry", extra_ok=("random",))


def _write_holdout_summary(
    metrics_payload: Dict[str, Any], path: Path
) -> None:
    """Write one row per evaluated model, sorted best -> worst by R²."""

    def _r2(item: Any) -> float:
        value = item[1].get("r2") if isinstance(item[1], dict) else None
        return float("-inf") if value is None else float(value)

    rows = sorted(metrics_payload.items(), key=_r2, reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "key",
                "label",
                "scheme",
                "feature_set",
                *_HOLDOUT_SUMMARY_METRICS,
            ]
        )
        for key, metrics in rows:
            m = metrics if isinstance(metrics, dict) else {}
            writer.writerow(
                [
                    key,
                    m.get("label"),
                    m.get("scheme"),
                    m.get("feature_set"),
                    *(m.get(name) for name in _HOLDOUT_SUMMARY_METRICS),
                ]
            )


def run_pipeline(
    cfg: Dict[str, Any], output_dir: Optional[Path] = None
) -> None:
    # Stage 1: Resolve and validate config sections.
    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}
    cv_cfg = cfg.get("cv") or {}
    out_cfg = cfg.get("outputs") or {}
    prep_cfg = cfg.get("preprocessing") or {}
    feature_cfg = cfg.get("features") or {}
    vis_cfg = cfg.get("visualise") or {}
    shap_cfg = cfg.get("shap") or {}
    holdout_eval_cfg = cfg.get("holdout_evaluation") or {}

    if not isinstance(data_cfg, dict):
        raise ValueError("Section 'data' must be a mapping")
    if not isinstance(split_cfg, dict):
        raise ValueError("Section 'split' must be a mapping")
    if not isinstance(cv_cfg, dict):
        raise ValueError("Section 'cv' must be a mapping")
    if not isinstance(out_cfg, dict):
        raise ValueError("Section 'outputs' must be a mapping")
    if not isinstance(prep_cfg, dict):
        raise ValueError("Section 'preprocessing' must be a mapping")
    if not isinstance(feature_cfg, dict):
        raise ValueError("Section 'features' must be a mapping")
    if not isinstance(vis_cfg, dict):
        raise ValueError("Section 'visualise' must be a mapping")
    if not isinstance(shap_cfg, dict):
        raise ValueError("Section 'shap' must be a mapping")
    if not isinstance(holdout_eval_cfg, dict):
        raise ValueError("Section 'holdout_evaluation' must be a mapping")

    # `groupings` section; older configs used features.create_default_groupings
    # and data.groupings (a CSV path) instead — still honoured as fallbacks.
    group_cfg = cfg.get("groupings")
    if group_cfg is None:
        group_cfg = {
            "defaults": feature_cfg.get("create_default_groupings", True),
            "columns": [],
            "file": data_cfg.get("groupings"),
        }
    if not isinstance(group_cfg, dict):
        raise ValueError("Section 'groupings' must be a mapping")

    if not data_cfg.get("metadata") or not data_cfg.get("attributes"):
        raise ValueError("data.metadata and data.attributes are required")
    if not data_cfg.get("profiles"):
        raise ValueError("data.profiles is required")
    labels = data_cfg.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("data.labels must be a non-empty mapping")

    base_dir = output_dir or Path(str(out_cfg.get("base_dir", "out/pipeline")))
    cv_out = base_dir / "cv_results"
    best_out = base_dir / "best_models"
    holdout_out = base_dir / "holdout"
    # Where the sample assignments are written: holdout.csv (train/test) and
    # cv_<scheme>.csv (fold membership, drawn from holdout-train only) per
    # label. Defaults to <base_dir>/splits; `outputs.splits_dir` overrides
    # (`holdout_splits_dir` is still accepted as the old name).
    splits_raw = out_cfg.get("splits_dir") or out_cfg.get("holdout_splits_dir")
    splits_out = Path(str(splits_raw)) if splits_raw else base_dir / "splits"

    # Stage 2: Build dataset and feature tables.
    LOGGER.info("Building dataset")
    dataset = (
        Dataset()
        .add_metadata(
            metadata=data_cfg["metadata"],
            attributes=data_cfg["attributes"],
            study_titles=data_cfg.get("study_titles"),
        )
        .add_profiles(
            profiles=data_cfg["profiles"],
            root=data_cfg.get("root"),
        )
        .add_labels(labels)
    )

    # Groupings: the columns that `split.holdout.grouping`, `split.cv.grouping`
    # and `cv.scheme` can refer to. Built in this order so nothing is lost:
    #   1. defaults from metadata (missing ones skipped),
    #   2. user-named metadata columns (missing ones are an error),
    #   3. user CSV of custom columns (merged; duplicates are an error).
    if group_cfg.get("defaults", True):
        dataset = dataset.create_default_groupings(force=True)

    extra_columns = group_cfg.get("columns") or []
    if isinstance(extra_columns, str):
        extra_columns = [extra_columns]
    if not isinstance(extra_columns, list):
        raise ValueError("groupings.columns must be a list of column names")
    if extra_columns:
        dataset = dataset.create_default_groupings(
            groupings=[str(c) for c in extra_columns],
            force=False,
            strict=True,
        )

    groupings_file = group_cfg.get("file")
    if groupings_file:
        dataset = dataset.add_groupings(groupings_file)

    available_groupings = (
        [c for c in dataset.groupings.columns if c != "sample"]
        if dataset.groupings is not None
        else []
    )
    LOGGER.info("Grouping columns available: %s", available_groupings)
    _validate_grouping_references(cfg, available_groupings)

    if prep_cfg.get("enabled", True):
        dataset = dataset.apply_preprocessing(
            metadata_qc=bool(prep_cfg.get("metadata_qc", True)),
            profiles_qc=bool(prep_cfg.get("profiles_qc", True)),
            sync_after=bool(prep_cfg.get("sync_after", True)),
            metadata_mbp_cutoff=int(prep_cfg.get("metadata_mbp_cutoff", 1000)),
            profiles_cov_cutoff=float(
                prep_cfg.get("profiles_cov_cutoff", 50.0)
            ),
            profiles_dominated_cutoff=float(
                prep_cfg.get("profiles_dominated_cutoff", 0.99)
            ),
            profiles_rank=prep_cfg.get("profiles_rank", "order"),
        )

    if feature_cfg.get("add_taxonomic_features", True):
        dataset = dataset.add_taxonomic_features(
            ranks=feature_cfg.get("ranks"),
            prefix=str(feature_cfg.get("prefix", "tax")),
            all=bool(feature_cfg.get("all", True)),
        )

    # Stage 3: Create holdout and CV splits.
    LOGGER.info("Creating holdout split and CV folds")
    holdout_cfg = split_cfg.get("holdout") or {}
    split_cv_cfg = split_cfg.get("cv") or {}
    if not isinstance(holdout_cfg, dict):
        raise ValueError("Section 'split.holdout' must be a mapping")
    if not isinstance(split_cv_cfg, dict):
        raise ValueError("Section 'split.cv' must be a mapping")

    dataset = dataset.create_holdout_split(
        label=holdout_cfg.get("label"),
        test_size=float(holdout_cfg.get("test_size", 0.2)),
        n_bins=int(holdout_cfg.get("n_bins", 5)),
        grouping=holdout_cfg.get("grouping"),
        random_state=int(holdout_cfg.get("random_state", 42)),
        force=bool(holdout_cfg.get("force", True)),
        output_dir=splits_out,
    )
    LOGGER.info("Saved holdout split(s) to %s", splits_out)
    dataset = dataset.create_cv_folds(
        label=split_cv_cfg.get("label"),
        n_folds=int(split_cv_cfg.get("n_folds", 5)),
        n_bins=int(split_cv_cfg.get("n_bins", 5)),
        grouping=split_cv_cfg.get("grouping", "all"),
        random_state=int(split_cv_cfg.get("random_state", 42)),
        use_holdout=bool(split_cv_cfg.get("use_holdout", True)),
        force=bool(split_cv_cfg.get("force", True)),
        strict=bool(split_cv_cfg.get("strict", True)),
    )
    dataset.save_cv_folds(splits_out)
    LOGGER.info("Saved CV fold assignments to %s", splits_out)

    save_dataset_path = out_cfg.get("save_dataset_path")
    if save_dataset_path:
        dataset.save(
            save_dataset_path,
            compress=bool(out_cfg.get("save_dataset_compress", False)),
        )
        LOGGER.info("Saved processed dataset to %s", save_dataset_path)

    # Stage 4: Run cross-validation and export CV artifacts.
    LOGGER.info("Running cross-validation")
    models = cv_cfg.get("models", ["rf"])
    if not isinstance(models, list):
        raise ValueError("cv.models must be a list (e.g., ['rf', 'xgboost'])")

    cv = CrossValidator(
        dataset=dataset,
        models=models,
        cv_folds=int(split_cv_cfg.get("n_folds", 5)),
        label=cv_cfg.get("label"),
        scheme=cv_cfg.get("scheme"),
        feature_set=cv_cfg.get("feature_set"),
    )

    mode = str(cv_cfg.get("mode", "run")).strip().lower()
    param_path = str(cv_cfg.get("param_path", "parameters.yaml"))
    n_jobs_raw = cv_cfg.get("n_jobs", None)
    n_jobs: Optional[int] = None if n_jobs_raw is None else int(n_jobs_raw)

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

    # By default only the tables that are not derivable elsewhere are kept:
    # per-combination model pickles are already covered by best_models/ and
    # holdout/, and results_folds.csv is an unpivot of results.ndjson.
    LOGGER.info("Exporting all CV results to %s", cv_out)
    CV_Result.export_result(
        results,
        cv_out,
        save_models=bool(out_cfg.get("cv_save_models", False)),
        fold_table=bool(out_cfg.get("cv_fold_table", False)),
    )

    # Keep the winner of every CV scheme per label (best_models/<label>/
    # <scheme>/), not just the overall best: the random scheme usually wins
    # CV because it leaks group structure, so schemes must be compared on
    # the holdout instead.
    LOGGER.info("Exporting best result per label/scheme to %s", best_out)
    best_for_holdout: Any
    if cv.best_result_by_label_scheme:
        CV_Result.export_best_results_by_scheme(
            cv.best_result_by_label_scheme,
            best_out,
            best_result_key_by_label_scheme=cv.best_result_key_by_label_scheme,
        )
        best_for_holdout = cv.best_result_by_label_scheme
    elif cv.best_result is not None:
        CV_Result.export_result(
            {cv.best_result_key or "best_result": cv.best_result}, best_out
        )
        best_for_holdout = cv.best_result
    else:
        raise RuntimeError("No best CV result was produced")

    _log_best_combinations(cv)

    vis_enabled = bool(vis_cfg.get("enabled", False))
    vis_formats = list(vis_cfg.get("formats", ["png"]))
    top_n = int(vis_cfg.get("top_n", 20))
    if vis_enabled:
        LOGGER.info("Generating CV bar visualisations from %s", cv_out)
        Visualiser(
            base_dir / "cv_visualisations", formats=vis_formats
        ).plot_cv_bars(results=cv_out, out_dir=base_dir / "cv_visualisations")

    # Optional stop: inspect CV / best_models before committing to the
    # one-shot holdout evaluation. The holdout split was still created and
    # saved under splits/, so a later run with the same split.* settings
    # evaluates against the identical test samples.
    if not bool(holdout_eval_cfg.get("enabled", True)):
        LOGGER.info(
            "holdout_evaluation.enabled is false: stopping after CV. "
            "Best model per label/scheme is under %s "
            "(see best_models_summary.csv). Outputs in %s",
            best_out,
            base_dir,
        )
        return

    # Stage 5: Train final holdout model(s) per label/scheme, write metrics.
    # SHAP (optional, needs the `shap` package / `pixi run -e shap`) runs on
    # each holdout test split; the trainer writes shap_summary.csv and
    # shap_values.csv next to that model and logs a warning if shap is
    # missing instead of failing the run.
    shap_enabled = bool(shap_cfg.get("enabled", False))
    shap_top_n = int(shap_cfg.get("top_n", 20))
    LOGGER.info("Training final holdout model(s) (shap=%s)", shap_enabled)
    trainer = ModelTrainer(
        dataset=dataset,
        best_result=best_for_holdout,
        output_model_path=holdout_out,
    )
    evaluation = trainer.train_and_evaluate(
        compute_shap=shap_enabled,
        max_shap_background=int(shap_cfg.get("max_background", 100)),
    )

    metrics_path = holdout_out / "holdout_metrics.json"
    if isinstance(evaluation, dict):
        metrics_payload: Dict[str, Any] = {
            key: _metrics_with_shap(ev, shap_top_n)
            for key, ev in evaluation.items()
        }
    elif isinstance(evaluation, HoldoutEvaluation):
        metrics_payload = {
            "result": _metrics_with_shap(evaluation, shap_top_n)
        }
    else:
        metrics_payload = {"result": str(type(evaluation))}

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    LOGGER.info("Wrote holdout metrics to %s", metrics_path)

    # Flat, sorted table so "which scheme generalises best" is one glance.
    summary_path = holdout_out / "holdout_summary.csv"
    _write_holdout_summary(metrics_payload, summary_path)
    LOGGER.info("Wrote holdout summary to %s", summary_path)

    # Stage 6: Generate optional holdout visualisations.
    if vis_enabled:
        vis = Visualiser(holdout_out, formats=vis_formats)

        if isinstance(evaluation, dict):
            # Keys are "<label>" or "<label>/<scheme>"; flatten for filenames.
            for key, ev in evaluation.items():
                tag = key.replace("/", "__")
                scheme = (
                    ev.metrics.get("scheme")
                    if isinstance(ev.metrics, dict)
                    else None
                )
                groups = [scheme] * len(ev.targets) if scheme else None
                vis.visualise_model_performance(
                    ev.predictions,
                    ev.targets,
                    title=f"Holdout diagnostics ({key})",
                    groups=groups,
                    file_name=f"holdout_diagnostics_{tag}",
                )
                vis.plot_feature_importances(
                    ev,
                    output=f"holdout_feature_importance_{tag}",
                    top_n=top_n,
                )
                if ev.shap_result is not None:
                    _plot_shap(
                        vis, ev.shap_result, shap_top_n, tag=tag, label=key
                    )
        elif isinstance(evaluation, HoldoutEvaluation):
            scheme = (
                evaluation.metrics.get("scheme")
                if isinstance(evaluation.metrics, dict)
                else None
            )
            groups = [scheme] * len(evaluation.targets) if scheme else None
            vis.visualise_model_performance(
                evaluation.predictions,
                evaluation.targets,
                title="Holdout diagnostics",
                groups=groups,
                file_name="holdout_diagnostics",
            )
            vis.plot_feature_importances(
                evaluation,
                output="holdout_feature_importance",
                top_n=top_n,
            )
            if evaluation.shap_result is not None:
                _plot_shap(vis, evaluation.shap_result, shap_top_n)

    LOGGER.info("Pipeline finished. Outputs in %s", base_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run microbiome CV + final holdout evaluation pipeline"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML pipeline config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override (defaults to outputs.base_dir in YAML)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    LOGGER.setLevel(getattr(logging, args.log_level.upper()))
    cfg = load_yaml_config(args.config)
    run_pipeline(cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
