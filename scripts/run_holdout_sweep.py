#!/usr/bin/env python3
"""Run the pipeline once per holdout grouping and collate the results.

One queue submission, one summary. The dataset (inputs, groupings, QC,
taxonomic features) is built ONCE; then for every grouping in the sweep the
holdout split, CV folds, cross-validation and (if enabled in the config)
holdout evaluation are re-run into their own sub-directory.

Example:
    pixi run python scripts/run_holdout_sweep.py --config pipeline.yaml \\
        --groupings random bioproject ecoregion month

Outputs, under outputs.base_dir (or --output-dir):
    holdout_<grouping>/...              one full pipeline output per grouping
    cv_sweep_headline.tsv               control vs best grouped scheme per
                                        holdout grouping x label (+ fold profile)
    cv_sweep_fold_profile.tsv           whether each scheme s folds were usable
    cv_sweep_all.tsv                    every CV combination, all groupings,
                                        sorted by avg_validation_r2
    holdout_sweep_summary.tsv           (only when holdout_evaluation.enabled)
                                        holdout metrics, sorted by r2
    sweep_status.tsv                    ok / failed per grouping
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

# Reuse the single-run pipeline (same directory) rather than duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_steps as steps  # noqa: E402
import run_cv_to_final_eval as pipeline  # noqa: E402

from microbiome_ml.wrangle.dataset import _as_rank  # noqa: E402

LOGGER = logging.getLogger("sweep")

# Sweep tokens are interpreted by pipeline_steps.holdout_spec:
#   random / null / none  -> stratified random holdout
#   taxon:family          -> holdout by community membership at that rank
#   <column>              -> group-aware holdout on a metadata column
_dir_name = steps.spec_dir_name


def _read_if_exists(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        LOGGER.warning("Missing %s; skipped in the collation", path)
        return None
    df = pl.read_csv(path, infer_schema_length=10000)
    return df if df.height else None


def _collate(
    frames: List[pl.DataFrame],
    sort_by: str,
    out_path: Optional[Path] = None,
) -> Optional[pl.DataFrame]:
    """Concatenate and sort; write only when an out_path is given."""
    if not frames:
        return None
    try:
        df = pl.concat(frames, how="diagonal_relaxed")
    except (TypeError, ValueError):  # older polars: no *_relaxed
        df = pl.concat(frames, how="diagonal")
    if sort_by in df.columns:
        df = df.sort(sort_by, descending=True, nulls_last=True)
    if out_path is not None:
        df.write_csv(out_path, separator="\t")
        LOGGER.info("Wrote %s (%d rows)", out_path, df.height)
    return df


def run_sweep(
    cfg: Dict[str, Any],
    groupings: List[str],
    output_dir: Optional[Path] = None,
) -> Path:
    out_cfg = cfg.get("outputs") or {}
    base_dir = output_dir or Path(str(out_cfg.get("base_dir", "out/pipeline")))
    base_dir.mkdir(parents=True, exist_ok=True)

    # Stage 2 once.
    dataset = pipeline.build_dataset(cfg)

    specs = [steps.holdout_spec(g) for g in groupings]

    # Taxon holdouts need the feature set for that rank; check before any CV.
    for _kind, _value in specs:
        if _kind != "taxon":
            continue
        try:
            dataset._rank_feature_set(_as_rank(str(_value)))
        except ValueError as exc:
            raise ValueError(f"taxon:{_value} cannot be used -- {exc}")

    # A requested grouping COLUMN not in the groupings table yet is pulled
    # from the metadata, exactly as `groupings.columns` would do. Only a
    # column that does not exist anywhere is an error -- raised now, not
    # after hours of CV on the other groupings.
    available = (
        [c for c in dataset.groupings.columns if c != "sample"]
        if dataset.groupings is not None
        else []
    )
    missing = [
        v
        for kind, v in specs
        if kind == "column" and v is not None and v not in available
    ]
    if missing:
        LOGGER.info(
            "Grouping(s) %s not in the groupings table; adding from metadata",
            missing,
        )
        dataset = dataset.create_default_groupings(
            groupings=missing, force=False, strict=True
        )
        available = [c for c in dataset.groupings.columns if c != "sample"]
    LOGGER.info("Grouping columns available: %s", available)

    holdout_enabled = bool(
        (cfg.get("holdout_evaluation") or {}).get("enabled", True)
    )
    LOGGER.info(
        "Sweeping %d holdout grouping(s): %s  (holdout evaluation %s)",
        len(groupings),
        [_dir_name(g) for g in groupings],
        "enabled" if holdout_enabled else "disabled -> CV only",
    )

    status: List[Dict[str, Any]] = []
    best_frames: List[pl.DataFrame] = []
    all_frames: List[pl.DataFrame] = []
    holdout_frames: List[pl.DataFrame] = []

    for token, (kind, value) in zip(groupings, specs):
        name = _dir_name(token)
        run_dir = base_dir / f"holdout_{name}"
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("split", {}).setdefault("holdout", {})
        if kind == "taxon":
            run_cfg["split"]["holdout"]["taxon_rank"] = value
            run_cfg["split"]["holdout"]["grouping"] = None
        else:
            run_cfg["split"]["holdout"]["grouping"] = value
            run_cfg["split"]["holdout"].pop("taxon_rank", None)
        # Per-run outputs live under run_dir; drop any absolute overrides
        # that would make the runs write on top of each other.
        run_cfg.setdefault("outputs", {})
        run_cfg["outputs"]["base_dir"] = str(run_dir)
        run_cfg["outputs"].pop("splits_dir", None)
        run_cfg["outputs"].pop("holdout_splits_dir", None)
        # The saved dataset only differs by its splits; keep one copy per run.
        if run_cfg["outputs"].get("save_dataset_path"):
            run_cfg["outputs"]["save_dataset_path"] = str(run_dir / "dataset")

        LOGGER.info("=" * 72)
        LOGGER.info("holdout grouping = %s  ->  %s", name, run_dir)
        LOGGER.info("=" * 72)
        try:
            pipeline.run_from_dataset(run_cfg, dataset, output_dir=run_dir)
        except Exception as exc:  # keep sweeping; report at the end
            LOGGER.error("holdout grouping %s FAILED: %s", name, exc)
            LOGGER.debug(traceback.format_exc())
            status.append(
                {
                    "holdout_grouping": name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        status.append({"holdout_grouping": name, "status": "ok", "error": ""})

        best = _read_if_exists(
            run_dir / "best_models" / "best_models_summary.csv"
        )
        if best is not None:
            best_frames.append(
                best.with_columns(
                    pl.lit(name).alias("holdout_grouping")
                ).select(["holdout_grouping", *best.columns])
            )
        every = _read_if_exists(run_dir / "cv_results" / "results_summary.csv")
        if every is not None:
            all_frames.append(
                every.with_columns(
                    pl.lit(name).alias("holdout_grouping")
                ).select(["holdout_grouping", *every.columns])
            )
        if holdout_enabled:
            hold = _read_if_exists(run_dir / "holdout" / "holdout_summary.csv")
            if hold is not None:
                holdout_frames.append(hold)

    # Collate.
    pl.DataFrame(status).write_csv(
        base_dir / "sweep_status.tsv", separator="\t"
    )
    # Not written out: cv_sweep_all.tsv is the superset and
    # cv_sweep_headline.tsv is built from this.
    best_df = _collate(best_frames, "avg_validation_r2")
    _collate(all_frames, "avg_validation_r2", base_dir / "cv_sweep_all.tsv")
    if holdout_enabled:
        _collate(holdout_frames, "r2", base_dir / "holdout_sweep_summary.tsv")

    failed = [s["holdout_grouping"] for s in status if s["status"] != "ok"]
    if failed:
        LOGGER.warning("Failed grouping(s): %s (see sweep_status.tsv)", failed)
    if best_df is not None:
        steps.write_summaries(best_df, base_dir)
        LOGGER.info("Best CV combination per grouping/label/scheme:")
        for row in best_df.head(15).iter_rows(named=True):
            LOGGER.info(
                "  holdout=%-12s label=%-12s scheme=%-12s model=%-22s "
                "feature_set=%-12s avg_r2=%s",
                row.get("holdout_grouping"),
                row.get("label"),
                row.get("scheme"),
                row.get("model"),
                row.get("feature_set"),
                row.get("avg_validation_r2"),
            )
    LOGGER.info("Sweep finished. Outputs in %s", base_dir)
    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CV(+holdout) pipeline once per holdout grouping and "
            "collate the results into TSV summaries."
        )
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config"
    )
    parser.add_argument(
        "--groupings",
        nargs="+",
        required=True,
        metavar="GROUPING",
        help=(
            "Holdout strategies to sweep: 'random' (stratified random, the "
            "control), a metadata column (e.g. bioproject), or "
            "'taxon:<rank>' (e.g. taxon:family) to split on community "
            "membership at that rank."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sweep root (default: outputs.base_dir). Each grouping writes "
        "to <root>/holdout_<grouping>/",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override applied to every run (same syntax as "
        "run_cv_to_final_eval.py --set).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    pipeline.setup_logging(args.log_level)
    level = getattr(logging, args.log_level.upper())
    LOGGER.setLevel(level)
    pipeline.LOGGER.setLevel(level)

    cfg = pipeline.load_yaml_config(args.config)
    for item in args.set:
        key, value = pipeline._parse_override(item)
        pipeline._set_by_path(cfg, key, value)
        LOGGER.info("Config override: %s = %r", key, value)

    seen: List[str] = []
    for g in args.groupings:
        canonical = _dir_name(g)
        if canonical not in [_dir_name(s) for s in seen]:
            seen.append(g.strip())
    run_sweep(cfg, seen, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
