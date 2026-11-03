#!/usr/bin/env python3
"""Drop the features a model leaned on least, re-run the same CV, repeat.

One round is: run CV, read `feature_importances_` off the refit model,
keep a subset, run the same CV again on that narrower matrix. Two ways
to pick the subset:

    zero      (default) keep every feature with importance above zero.
              A zero-importance feature was never chosen for a single
              split in a single tree, so this removes only columns the
              model demonstrably ignored -- round 1 is close to free.
    coverage  keep the smallest set reaching `coverage` of the total
              importance, the same count the feature report prints as
              n_for_95pct. Far more aggressive, and it does throw away
              features the model used.

The two are not the same kind of operation. `zero` settles: once the
unused columns are gone there is usually nothing left to drop, and the
run converges in a round or two. `coverage` keeps taking a fraction of
whatever remains, so with one dominant feature it shrinks round after
round until --min-features stops it. Set that floor deliberately when
using coverage; the default of 10 is a backstop, not a choice.

Rounds after the first are not free, and that is the point. Tree models
pick one feature per split out of all the candidates, so a feature can
score zero only because something else always beat it. Remove the
winners' dead competition and the next fit faces a different contest:
features that were never used may start being used, and new zeros appear
at the margin. The run stops on its own when a round drops nothing.

Each combination keeps its own feature list, because they disagree
sharply -- on 18S marine genus, RandomForest left 2,142 of 4,260 at zero
and XGBoost 3,696. So one CV call is made per model, not one per pair.

What neither mode does is make the score trustworthy. The importances
come from a model refit on every CV row, validation folds included, so
pruning on them and re-scoring on the same folds is selection on the
data being scored. Under `zero` the bias is mild, since a dropped
feature contributed nothing to the model that ranked it. Under
`coverage` it is not mild at all: that is textbook select-then-refit on
the validation folds, and the score will drift upward for reasons that
have nothing to do with the features being better. Read the trajectory
as "how much can I throw away before the score moves", never as a
better score. The holdout, untouched by this loop, is the honest read.

Example:
    python scripts/prune_iterate_cv.py \\
        --config pipeline.yaml \\
        --dataset out/pipeline/prepared \\
        --splits out/pipeline/holdout_bioproject__ph/splits \\
        --label ph --scheme bioproject_logo \\
        --feature-set tax_genus_bio \\
        --iterations 3 --out out/prune_ph

and the aggressive variant, keeping 95% of the importance each round:
    ... --prune coverage --coverage 0.95 --min-features 100

Outputs under --out:
    prune_history.tsv     one row per combination per round
    prune_summary.txt     the same, as a readable trajectory
    kept_features/        the surviving features, named, per round
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_steps as steps  # noqa: E402
import run_cv_to_final_eval as pipeline  # noqa: E402

from microbiome_ml.train.cv import CrossValidator  # noqa: E402
from microbiome_ml.wrangle.dataset import Dataset  # noqa: E402
from microbiome_ml.wrangle.features import SampleFeatureSet  # noqa: E402

LOGGER = logging.getLogger("prune-iterate")

PRUNE_MODES = ("zero", "coverage")

HISTORY_COLUMNS = [
    "iteration",
    "label",
    "scheme",
    "model",
    "feature_set",
    "prune",
    "coverage",
    "n_features_in",
    "n_pruned",
    "n_features_out",
    "pooled_validation_r2",
    "avg_validation_r2",
    "n_folds_evaluable",
    "n_folds",
    "n_label",
    "status",
]


def _feature_columns(feature_set: Any) -> Tuple[str, List[str]]:
    """The accession column and the feature columns of a feature set."""
    frame = getattr(feature_set, "features", None)
    if not isinstance(frame, pl.LazyFrame):
        frame = feature_set.to_df().lazy()
    names = frame.collect_schema().names()
    acc = next((c for c in ("sample", "acc") if c in names), None)
    if acc is None:
        raise RuntimeError(
            f"feature set '{feature_set.name}' has no 'sample' or 'acc' "
            "column"
        )
    return acc, [c for c in names if c != acc]


def restrict_features(dataset: Dataset, name: str, keep: List[str]) -> Dataset:
    """A shallow copy of *dataset* whose feature set *name* has only *keep*.

    Nothing is written to disk and the original is left alone: the CV
    path reads `feature_sets[name].features` lazily, so selecting
    columns on that LazyFrame is the whole change. Column order is the
    caller's, which is what makes `feature_names` line up with
    `feature_importances_` on the way back.
    """
    original = dataset.feature_sets[name]
    acc, _ = _feature_columns(original)
    frame = original.features
    if not isinstance(frame, pl.LazyFrame):
        frame = original.to_df().lazy()
    pruned = SampleFeatureSet(
        accessions=list(original.accessions),
        feature_names=list(keep),
        features=frame.select([acc, *keep]),
        name=name,
    )
    narrowed = copy.copy(dataset)
    narrowed.feature_sets = dict(dataset.feature_sets)
    narrowed.feature_sets[name] = pruned
    return narrowed


def _best_result(results: Dict[str, Any]) -> Optional[Any]:
    """The result a sweep would call best: highest `rank_score`.

    Only matters with a parameter grid; a default run yields one.
    """
    scored = [r for r in results.values() if r.rank_score is not None]
    if scored:
        return max(scored, key=lambda r: r.rank_score)
    return next(iter(results.values()), None)


def features_to_keep(
    result: Any,
    keep: List[str],
    mode: str = "zero",
    coverage: float = 0.95,
) -> Optional[List[str]]:
    """The features to carry into the next round, in *keep*'s order.

    `zero` keeps importance > 0; `coverage` keeps the smallest set whose
    importances sum to at least `coverage` of the total, which is the
    count rank_feature_report.py prints as n_for_<coverage>pct.

    Returns None when there is nothing to prune *on* -- no importances,
    or a total of zero -- which is a different situation from "it used
    nothing" and must not be read as a prune of everything.
    """
    model = getattr(result, "model", None)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    names = list(getattr(result, "feature_names", None) or [])
    # Matching lengths are not enough: names that are the right count but
    # the wrong columns would zip onto the wrong importances and prune
    # everything silently. The CV yields columns in the order selected
    # here, so falling back to `keep` positionally is sound.
    if len(names) != len(importances) or set(names) != set(keep):
        if len(keep) != len(importances):
            LOGGER.warning(
                "cannot align %d importances with %d requested features; "
                "nothing pruned this round",
                len(importances),
                len(keep),
            )
            return None
        LOGGER.warning(
            "result.feature_names does not match the features requested; "
            "aligning importances positionally instead"
        )
        names = list(keep)

    paired = [(n, float(imp)) for n, imp in zip(names, importances)]
    if mode == "zero":
        alive = {n for n, imp in paired if imp > 0.0}
    else:
        total = sum(imp for _, imp in paired if imp > 0.0)
        if total <= 0.0:
            LOGGER.warning(
                "total importance is zero; no coverage to apportion, "
                "nothing pruned this round"
            )
            return None
        # Same rule as the report: walk the features from most to least
        # important and stop on the first one that takes the running
        # share to `coverage`, that one included.
        alive = set()
        running = 0.0
        for name, imp in sorted(paired, key=lambda p: p[1], reverse=True):
            if imp <= 0.0:
                break
            alive.add(name)
            running += imp
            if running / total >= coverage:
                break
    return [c for c in keep if c in alive]


def _row(
    iteration: int,
    label: str,
    scheme: str,
    model_name: str,
    feature_set: str,
    mode: str,
    coverage: float,
    n_in: int,
    n_out: int,
    result: Optional[Any],
    status: str,
) -> Dict[str, Any]:
    return {
        "iteration": iteration,
        "label": label,
        "scheme": scheme,
        "model": model_name,
        "feature_set": feature_set,
        "prune": mode,
        # Only meaningful for coverage mode; null keeps the column from
        # implying zero-mode used a threshold.
        "coverage": coverage if mode == "coverage" else None,
        "n_features_in": n_in,
        "n_pruned": n_in - n_out,
        "n_features_out": n_out,
        "pooled_validation_r2": getattr(result, "pooled_validation_r2", None),
        "avg_validation_r2": getattr(result, "avg_validation_r2", None),
        "n_folds_evaluable": getattr(result, "n_folds_evaluable", None),
        "n_folds": getattr(result, "n_folds", None),
        "n_label": getattr(result, "n_label", None),
        "status": status,
    }


def prune_one_model(
    dataset: Dataset,
    label: str,
    scheme: str,
    feature_set: str,
    model: str,
    iterations: int,
    min_features: int,
    mode: str,
    coverage: float,
    param_path: str,
    n_jobs: Optional[int],
    kept_dir: Path,
) -> List[Dict[str, Any]]:
    """Run the prune/re-run loop for one model, newest round last."""
    _, keep = _feature_columns(dataset.feature_sets[feature_set])
    rows: List[Dict[str, Any]] = []
    model_name = str(model)

    for iteration in range(iterations + 1):
        narrowed = restrict_features(dataset, feature_set, keep)
        LOGGER.info(
            "iteration %d: label=%s scheme=%s model=%s features=%d "
            "prune=%s",
            iteration,
            label,
            scheme,
            model_name,
            len(keep),
            mode if mode == "zero" else f"coverage@{coverage:g}",
        )
        cv = CrossValidator(
            dataset=narrowed,
            models=[model],
            label=label,
            scheme=scheme,
            feature_set=feature_set,
        )
        results = cv.run(param_path=param_path, n_jobs=n_jobs)
        result = _best_result(results)
        if result is None:
            rows.append(
                _row(
                    iteration,
                    label,
                    scheme,
                    model_name,
                    feature_set,
                    mode,
                    coverage,
                    len(keep),
                    len(keep),
                    None,
                    "no_result",
                )
            )
            LOGGER.warning(
                "iteration %d produced no CV result; stopping this model",
                iteration,
            )
            break
        # The class name is what the CV exports record, so the history
        # joins to results_summary.csv without translating aliases.
        actual = getattr(result, "model", None)
        if actual is not None:
            model_name = actual.__class__.__name__

        stem = f"{label}__{scheme}__{model_name}__it{iteration}.txt"
        (kept_dir / stem).write_text("\n".join(keep) + "\n", encoding="utf-8")

        if iteration == iterations:
            rows.append(
                _row(
                    iteration,
                    label,
                    scheme,
                    model_name,
                    feature_set,
                    mode,
                    coverage,
                    len(keep),
                    len(keep),
                    result,
                    "iterations_done",
                )
            )
            break

        alive = features_to_keep(result, keep, mode, coverage)
        if alive is None:
            rows.append(
                _row(
                    iteration,
                    label,
                    scheme,
                    model_name,
                    feature_set,
                    mode,
                    coverage,
                    len(keep),
                    len(keep),
                    result,
                    "no_importances",
                )
            )
            LOGGER.warning(
                "%s exposes no feature_importances_; nothing to prune on",
                model_name,
            )
            break
        if len(alive) == len(keep):
            rows.append(
                _row(
                    iteration,
                    label,
                    scheme,
                    model_name,
                    feature_set,
                    mode,
                    coverage,
                    len(keep),
                    len(keep),
                    result,
                    "converged",
                )
            )
            LOGGER.info("iteration %d dropped nothing; converged", iteration)
            break
        if len(alive) < min_features:
            rows.append(
                _row(
                    iteration,
                    label,
                    scheme,
                    model_name,
                    feature_set,
                    mode,
                    coverage,
                    len(keep),
                    len(keep),
                    result,
                    "min_features_floor",
                )
            )
            LOGGER.info(
                "would leave %d features, below --min-features %d; stopping",
                len(alive),
                min_features,
            )
            break

        rows.append(
            _row(
                iteration,
                label,
                scheme,
                model_name,
                feature_set,
                mode,
                coverage,
                len(keep),
                len(alive),
                result,
                "pruned",
            )
        )
        keep = alive

    return rows


def _summary(history: pl.DataFrame) -> List[str]:
    out = [
        "=" * 78,
        "Iterative zero-importance pruning",
        "=" * 78,
        "",
        "Scores come from the folds the pruning was derived on, so read",
        "the trajectory as 'how much can I drop before it moves', not as",
        "a better score. The holdout is the honest read.",
        "",
    ]
    if "coverage" in set(history.get_column("prune").to_list()):
        out += [
            "coverage mode picks features using the same folds it then",
            "scores on, so a score that climbs here is expected and says",
            "little about the features. Compare on the holdout.",
            "",
        ]
    group = ["label", "scheme", "model", "feature_set", "prune"]
    for keys, block in history.group_by(group, maintain_order=True):
        out.append("  " + "  ".join(f"{c}={k}" for c, k in zip(group, keys)))
        out.append(
            f"    {'round':>6}{'features':>11}{'pruned':>9}"
            f"{'pooled_r2':>12}   status"
        )
        for row in block.iter_rows(named=True):
            score = row["pooled_validation_r2"]
            out.append(
                f"    {row['iteration']:>6}{row['n_features_in']:>11,}"
                f"{row['n_pruned']:>9,}"
                + (f"{score:>12.3f}" if score is not None else f"{'n/a':>12}")
                + f"   {row['status']}"
            )
        first = block.get_column("pooled_validation_r2")[0]
        last = block.get_column("pooled_validation_r2")[-1]
        n_first = block.get_column("n_features_in")[0]
        n_last = block.get_column("n_features_out")[-1]
        if first is not None and last is not None:
            out.append(
                f"    {n_first:,} -> {n_last:,} features "
                f"({100.0 * (n_first - n_last) / n_first:.1f}% dropped), "
                f"score {first:+.3f} -> {last:+.3f} ({last - first:+.3f})"
            )
        out.append("")
    return out


def run_loop(
    dataset_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    label: str,
    scheme: str,
    feature_set: Optional[str],
    models: List[str],
    iterations: int,
    min_features: int,
    mode: str,
    coverage: float,
    param_path: str,
    n_jobs: Optional[int],
) -> Path:
    if mode not in PRUNE_MODES:
        raise ValueError(
            f"prune must be one of {list(PRUNE_MODES)}, got {mode!r}"
        )
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_dir = out_dir / "kept_features"
    kept_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.load(dataset_dir)
    dataset = steps._attach_splits(dataset, splits_dir, label, scheme)

    available = list(dataset.feature_sets)
    if feature_set is None:
        if len(available) != 1:
            raise ValueError(
                "--feature-set is required when the dataset has more than "
                f"one: {available}"
            )
        feature_set = available[0]
    if feature_set not in dataset.feature_sets:
        raise ValueError(
            f"feature set '{feature_set}' not in the dataset. "
            f"Available: {available}"
        )

    rows: List[Dict[str, Any]] = []
    for model in models:
        rows.extend(
            prune_one_model(
                dataset=dataset,
                label=label,
                scheme=scheme,
                feature_set=feature_set,
                model=model,
                iterations=iterations,
                min_features=min_features,
                mode=mode,
                coverage=coverage,
                param_path=param_path,
                n_jobs=n_jobs,
                kept_dir=kept_dir,
            )
        )

    history = pl.DataFrame(rows).select(HISTORY_COLUMNS)
    history.write_csv(out_dir / "prune_history.tsv", separator="\t")
    LOGGER.info(
        "Wrote %s (%d rows)", out_dir / "prune_history.tsv", history.height
    )

    text = "\n".join(_summary(history)) + "\n"
    (out_dir / "prune_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return out_dir / "prune_summary.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="pipeline YAML; supplies cv.models, cv.feature_set, "
        "cv.param_path, cv.n_jobs and the feature_pruning section",
    )
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="prune/re-run rounds after the baseline; 0 = baseline only",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=None,
        help="stop before a round would leave fewer than this many",
    )
    parser.add_argument(
        "--prune",
        choices=PRUNE_MODES,
        default=None,
        help="zero (default) keeps importance > 0; coverage keeps the "
        "smallest set reaching --coverage of the total importance",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=None,
        help="share of total importance to keep when --prune coverage "
        "(default 0.95)",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--param-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    pipeline.setup_logging(args.log_level)
    LOGGER.setLevel(getattr(logging, str(args.log_level).upper(), 20))

    cv_cfg: Dict[str, Any] = {}
    prune_cfg: Dict[str, Any] = {}
    if args.config is not None:
        cfg = pipeline.load_yaml_config(args.config)
        cv_cfg = pipeline._sections(cfg)["cv"]
        prune_cfg = cfg.get("feature_pruning") or {}

    # The toggle lives in the config so a workflow can call this rule
    # unconditionally; the CLI equivalent is --iterations 0.
    if args.iterations is None and not prune_cfg.get("enabled", False):
        LOGGER.info(
            "feature_pruning.enabled is false and no --iterations given: "
            "nothing to do"
        )
        return

    iterations = (
        int(args.iterations)
        if args.iterations is not None
        else int(prune_cfg.get("iterations", 1))
    )
    if iterations < 0:
        parser.error("--iterations must be >= 0")
    min_features = (
        int(args.min_features)
        if args.min_features is not None
        else int(prune_cfg.get("min_features", 10))
    )
    mode = str(args.prune or prune_cfg.get("prune", "zero")).strip().lower()
    if mode not in PRUNE_MODES:
        parser.error(f"prune must be one of {list(PRUNE_MODES)}, got {mode!r}")
    coverage = (
        float(args.coverage)
        if args.coverage is not None
        else float(prune_cfg.get("coverage", 0.95))
    )
    if not 0 < coverage <= 1:
        parser.error(f"--coverage must be in (0, 1], got {coverage}")
    if args.coverage is not None and mode == "zero":
        LOGGER.warning("--coverage %g is ignored with --prune zero", coverage)
    models = args.models or cv_cfg.get("models") or ["rf"]
    if not isinstance(models, list):
        parser.error("cv.models must be a list (e.g. ['rf', 'xgboost'])")
    feature_set = args.feature_set or cv_cfg.get("feature_set")
    if isinstance(feature_set, (list, tuple)):
        if len(feature_set) != 1:
            parser.error(
                "--feature-set takes one name; cv.feature_set has "
                f"{list(feature_set)}"
            )
        feature_set = feature_set[0]
    n_jobs = args.n_jobs
    if n_jobs is None and cv_cfg.get("n_jobs") is not None:
        n_jobs = int(cv_cfg["n_jobs"])
    param_path = args.param_path or str(
        cv_cfg.get("param_path", "parameters.yaml")
    )

    out = run_loop(
        dataset_dir=args.dataset,
        splits_dir=args.splits,
        out_dir=args.out,
        label=args.label,
        scheme=args.scheme,
        feature_set=feature_set,
        models=list(models),
        iterations=iterations,
        min_features=min_features,
        mode=mode,
        coverage=coverage,
        param_path=param_path,
        n_jobs=n_jobs,
    )
    LOGGER.info("Done. Summary: %s", out)


if __name__ == "__main__":
    main()
