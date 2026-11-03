"""Diagnostics for the model fitted on the whole training pool.

Every score the CV stage reports comes from a model trained on k-1 folds
and scored on the one held out. The model that would actually be shipped
is trained on all of the training pool, and nothing in the pipeline
measured it -- `CrossValidator` already fits it (`estimator.fit(X, y)`
after `cross_validate`) but only to read feature importances off it.

This module scores that fit. Three numbers, and the distance between
them is the point:

    in_sample_r2    the model scored on the rows it was trained on. A
                    ceiling, not a performance estimate: a random forest
                    lands near 1.0 on almost anything.
    oob_r2          out-of-bag R2, available for a bagging model fitted
                    with `oob_score=True`. Genuinely held out -- each row
                    is scored by the trees that did not draw it -- and it
                    comes free from the fit that already happens.
    overfit_gap     in_sample_r2 minus the CV score. How much of the
                    in-sample fit is memorisation.

Read `oob_r2` with the same care as a random CV split. Bootstrap
sampling draws rows, not groups, so a row's own study is almost always
represented among the trees scoring it. On data where study identity
predicts the label -- which is this data -- OOB will look like the
`random` scheme and far better than `bioproject_logo`. It answers "did
this fit converge", not "will this generalise to a new study".

No sklearn import here on purpose: everything is duck-typed against the
estimator API and the arithmetic is numpy, so the metrics can be checked
without a model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Columns of the table `evaluate_full_fit` returns, in order.
FULL_FIT_COLUMNS = (
    "label",
    "scheme",
    "feature_set",
    "model",
    "n_fit_samples",
    "n_fit_features",
    "cv_pooled_r2",
    "in_sample_r2",
    "oob_r2",
    "overfit_gap",
    "in_sample_mse",
    "in_sample_mae",
    "residual_mean",
    "residual_std",
    "residual_max_abs",
)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """R2 around the mean of `y_true`; None when it is undefined.

    Spelled out rather than imported so this module needs no sklearn,
    and so the convention is visible: the denominator is the total sum
    of squares of the truth, which is zero for a constant target and
    makes R2 meaningless rather than zero.
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.size == 0 or truth.shape != pred.shape:
        return None
    sst = float(((truth - truth.mean()) ** 2).sum())
    if sst <= 0.0:
        return None
    sse = float(((truth - pred) ** 2).sum())
    return 1.0 - sse / sst


def supports_oob(estimator: Any) -> bool:
    """True when this estimator can produce an out-of-bag score.

    Needs both knobs: a forest with `bootstrap=False` trains every tree
    on every row, so there is nothing out of bag to score and sklearn
    raises rather than warns.
    """
    try:
        params = estimator.get_params()
    except Exception:  # pragma: no cover - not an sklearn-style estimator
        return False
    return "oob_score" in params and bool(params.get("bootstrap", False))


def enable_oob(estimator: Any) -> bool:
    """Ask for an OOB score on the next fit. True when it was set.

    Called *before* the full-pool fit that already happens, so the score
    costs nothing extra. Setting it afterwards would mean refitting.
    """
    if not supports_oob(estimator):
        return False
    try:
        estimator.set_params(oob_score=True)
    except Exception as exc:  # pragma: no cover
        logger.debug("could not enable oob_score: %s", exc)
        return False
    return True


def _residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid.std(ddof=0)),
        "residual_max_abs": float(np.abs(resid).max()),
    }


def full_fit_diagnostics(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Score an already-fitted estimator on the pool it was fitted to.

    Args:
        estimator: A fitted estimator exposing `predict`.
        X, y: The training pool it was fitted on.
        cv_score: The held-out score to compare against, for
            `overfit_gap`. Normally the pooled validation R2.

    Returns:
        A dict of the metric columns in `FULL_FIT_COLUMNS`. Any metric
        that cannot be computed is None rather than absent, so a table
        built from several of these keeps one shape.
    """
    truth = np.asarray(y, dtype=float)
    out: Dict[str, Any] = {
        "n_fit_samples": int(truth.shape[0]),
        "n_fit_features": int(np.asarray(X).shape[1])
        if np.asarray(X).ndim == 2
        else None,
        "cv_pooled_r2": None if cv_score is None else float(cv_score),
        "in_sample_r2": None,
        "oob_r2": None,
        "overfit_gap": None,
        "in_sample_mse": None,
        "in_sample_mae": None,
        "residual_mean": None,
        "residual_std": None,
        "residual_max_abs": None,
    }

    try:
        pred = np.asarray(estimator.predict(X), dtype=float)
    except Exception as exc:
        # A diagnostic must never be the thing that fails the run; the
        # scores it describes are already written by this point.
        logger.warning("full-fit prediction failed: %s", exc)
        return out

    if pred.shape != truth.shape:
        logger.warning(
            "full-fit prediction has shape %s for %d target(s); skipped",
            pred.shape,
            truth.shape[0],
        )
        return out

    out["in_sample_r2"] = r2(truth, pred)
    out["in_sample_mse"] = float(((truth - pred) ** 2).mean())
    out["in_sample_mae"] = float(np.abs(truth - pred).mean())
    out.update(_residuals(truth, pred))

    # Present only when the fit was asked for it; sklearn leaves the
    # attribute off entirely otherwise.
    oob = getattr(estimator, "oob_score_", None)
    if oob is not None:
        try:
            value = float(oob)
        except (TypeError, ValueError):
            value = float("nan")
        out["oob_r2"] = None if np.isnan(value) else value

    if out["in_sample_r2"] is not None and cv_score is not None:
        out["overfit_gap"] = float(out["in_sample_r2"]) - float(cv_score)
    return out


def top_importances(
    estimator: Any,
    feature_names: Sequence[str],
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """The `top_n` highest-importance features of the full-pool fit.

    Returns [] for an estimator that exposes none, or when the vector
    does not line up with the names -- a mislabelled importance is worse
    than a missing one.
    """
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return []
    array = np.asarray(values, dtype=float).ravel()
    names = [str(n) for n in feature_names]
    if array.size != len(names):
        logger.warning(
            "%d importance(s) for %d feature name(s); importances skipped",
            array.size,
            len(names),
        )
        return []
    order = np.argsort(array)[::-1][: max(0, int(top_n))]
    return [(names[i], float(array[i])) for i in order]


def _model_name(result: Any) -> Optional[str]:
    model = getattr(result, "model", None)
    return None if model is None else type(model).__name__


def collect_rows(
    results_by_label_scheme: Dict[Any, Any],
) -> List[Dict[str, Any]]:
    """One row per CV winner that carries full-fit diagnostics.

    Keyed by whatever the CrossValidator used (``(label, scheme)``), but
    the label and scheme are read off the result itself so the row does
    not depend on that key's shape.
    """
    rows: List[Dict[str, Any]] = []
    for result in results_by_label_scheme.values():
        diagnostics = getattr(result, "full_fit", None)
        if not diagnostics:
            continue
        rows.append(
            {
                "label": getattr(result, "label", None),
                "scheme": getattr(result, "scheme", None),
                "feature_set": getattr(result, "feature_set", None),
                "model": _model_name(result),
                **{
                    column: diagnostics.get(column)
                    for column in FULL_FIT_COLUMNS
                    if column
                    not in ("label", "scheme", "feature_set", "model")
                },
            }
        )
    # Worst overfit gap first: the row most worth explaining leads.
    rows.sort(
        key=lambda r: (
            r.get("overfit_gap") is None,
            -(r.get("overfit_gap") or 0.0),
        )
    )
    return rows


def write_full_fit_report(
    results_by_label_scheme: Dict[Any, Any],
    out_dir: Any,
    top_n: int = 20,
) -> Optional[Any]:
    """Write ``full_fit_metrics.csv`` (+ importances) for the winners.

    Returns the directory written, or None when no result carried
    diagnostics -- which is the normal state with the stage switched
    off, so it is not an error.
    """
    import csv
    from pathlib import Path

    rows = collect_rows(results_by_label_scheme)
    if not rows:
        logger.info("No full-fit diagnostics to write")
        return None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "full_fit_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FULL_FIT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d row(s))", metrics_path, len(rows))

    importance_rows: List[Dict[str, Any]] = []
    for result in results_by_label_scheme.values():
        if not getattr(result, "full_fit", None):
            continue
        model = getattr(result, "model", None)
        if model is None:
            continue
        for rank, (name, value) in enumerate(
            top_importances(
                model, getattr(result, "feature_names", []), top_n
            ),
            start=1,
        ):
            importance_rows.append(
                {
                    "label": getattr(result, "label", None),
                    "scheme": getattr(result, "scheme", None),
                    "feature_set": getattr(result, "feature_set", None),
                    "model": _model_name(result),
                    "rank": rank,
                    "feature": name,
                    "importance": value,
                }
            )
    if importance_rows:
        imp_path = out / "full_fit_importances.csv"
        with imp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(importance_rows[0])
            )
            writer.writeheader()
            writer.writerows(importance_rows)
        logger.info("Wrote %s (%d row(s))", imp_path, len(importance_rows))
    return out
