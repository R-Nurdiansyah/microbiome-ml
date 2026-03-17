#!/usr/bin/env python3
"""Run microbiome ML pipeline from CV to final holdout evaluation.

This script is designed for large-data runs outside notebooks.

Example:
    pixi run python scripts/run_cv_to_final_eval.py --config pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from microbiome_ml.train.cv import CrossValidator
from microbiome_ml.train.results import CV_Result, HoldoutEvaluation
from microbiome_ml.train.trainer import ModelTrainer
from microbiome_ml.utils.logging import setup_logging
from microbiome_ml.visualise.visualisations import Visualiser
from microbiome_ml.wrangle.dataset import Dataset

LOGGER = logging.getLogger("pipeline")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


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

    groupings = data_cfg.get("groupings")
    if groupings is not None:
        dataset = dataset.add_groupings(groupings)

    if feature_cfg.get("create_default_groupings", True):
        dataset = dataset.create_default_groupings(force=True)

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
    )
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

    LOGGER.info("Exporting all CV results to %s", cv_out)
    CV_Result.export_result(results, cv_out)

    LOGGER.info("Exporting best result(s) to %s", best_out)
    CV_Result.export_best_results(
        cv.best_result_by_label,
        best_out,
        best_result_key_by_label=cv.best_result_key_by_label,
        fallback_best_result=cv.best_result,
        fallback_best_key=cv.best_result_key or "best_result",
    )

    # Stage 5: Train final holdout model(s) and write metrics.
    best_for_holdout: Any
    if cv.best_result_by_label:
        best_for_holdout = cv.best_result_by_label
    elif cv.best_result is not None:
        best_for_holdout = cv.best_result
    else:
        raise RuntimeError("No best CV result was produced")

    LOGGER.info("Training final holdout model(s)")
    trainer = ModelTrainer(
        dataset=dataset,
        best_result=best_for_holdout,
        output_model_path=holdout_out,
    )
    evaluation = trainer.train_and_evaluate()

    metrics_path = holdout_out / "holdout_metrics.json"
    if isinstance(evaluation, dict):
        metrics_payload: Dict[str, Any] = {
            label: ev.metrics for label, ev in evaluation.items()
        }
    elif isinstance(evaluation, HoldoutEvaluation):
        metrics_payload = {"result": evaluation.metrics}
    else:
        metrics_payload = {"result": str(type(evaluation))}

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    LOGGER.info("Wrote holdout metrics to %s", metrics_path)

    # Stage 6: Generate optional holdout visualisations.
    if bool(vis_cfg.get("enabled", False)):
        vis = Visualiser(
            holdout_out,
            formats=list(vis_cfg.get("formats", ["png"])),
        )
        top_n = int(vis_cfg.get("top_n", 20))

        LOGGER.info("Generating CV bar visualisations from %s", cv_out)
        vis.plot_cv_bars(
            results=cv_out,
            out_dir=base_dir / "cv_visualisations",
        )

        if isinstance(evaluation, dict):
            for label, ev in evaluation.items():
                scheme = (
                    ev.metrics.get("scheme")
                    if isinstance(ev.metrics, dict)
                    else None
                )
                groups = [scheme] * len(ev.targets) if scheme else None
                vis.visualise_model_performance(
                    ev.predictions,
                    ev.targets,
                    title=f"Holdout diagnostics ({label})",
                    groups=groups,
                    file_name=f"holdout_diagnostics_{label}",
                )
                vis.plot_feature_importances(
                    ev,
                    output=f"holdout_feature_importance_{label}",
                    top_n=top_n,
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
