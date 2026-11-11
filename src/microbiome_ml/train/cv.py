"""Cross-validation helpers and command-line usage sample for CV exports.

Instantiate `CrossValidator` with a dataset and model(s), run `run()` or
`run_grid()` to evaluate every feature/label/scheme combination, then flush the
outputs via `CV_Result.export_result(results, "out/cv_results")` to capture the
NDJSON/CSV manifests plus per-combo pickled models.

Example:
    cv = CrossValidator(dataset, models="rf")
    results = cv.run(param_path="hyperparameters.yaml")
    CV_Result.export_result(results, "out/cv_results")
"""

import logging
import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import polars as pl
import yaml  # type: ignore[import]
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    PredefinedSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVR

# from sklearn.linear_model import ElasticNet, LinearRegression
# from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from microbiome_ml.train.full_fit import enable_oob, full_fit_diagnostics
from microbiome_ml.train.results import (
    CV_Result,
    LabelSchemeKey,
    best_candidate_index,
)
from microbiome_ml.wrangle.dataset import Dataset

logger = logging.getLogger(__name__)


class ComboMeta(NamedTuple):
    """Per-combination data the CV loops need besides X and y.

    Deliberately small: the joined DataFrame it replaces held a second
    copy of the whole feature matrix.
    """

    fold: np.ndarray
    feature_columns: List[str]


# Mapping of model short names to their constructors
# Canonical constructors keyed by short names (keeps compatibility with hyperparameters.yaml)
# For now works with regressor models only (random forest, gradient boosting, xgboost)
#
# Seeded: an unseeded forest moved holdout R2 by up to 0.07 between two runs of
# the same data and splits, more than the gaps between the models being ranked.
RANDOM_STATE = 42


class ScaledSVR(Pipeline):
    """SVR with a linear kernel behind a MaxAbsScaler, as one estimator.

    sklearn's SVR (libsvm), not LinearSVR (liblinear): the standard
    epsilon-SVR, which leaves the intercept unpenalised and can take
    another kernel from the grid (`svr__kernel`). The price is fit time,
    which grows with the square of the sample count or worse.

    The penalty on the coefficients is what needs the scaling. Unscaled, a
    rare taxon peaking at 0.001 needs a coefficient 300 times larger than an
    abundant one peaking at 0.3 to matter as much, and the penalty all but
    forbids it. MaxAbsScaler puts each taxon on [0, 1] and leaves zeros as
    zeros, so sparse input stays sparse.

    A subclass rather than a bare Pipeline so reports name it, and so another
    Pipeline is never mistaken for it. Grid keys take the step prefix:
    `svr__C`, `svr__epsilon`, `svr__kernel`.
    """

    def __init__(
        self,
        steps: Optional[List[Tuple[str, Any]]] = None,
        *,
        transform_input: Optional[List[str]] = None,
        memory: Any = None,
        verbose: bool = False,
    ) -> None:
        if steps is None:
            # cache_size: libsvm's kernel cache in MB. The default 200 is
            # spent within a few thousand samples; past that it recomputes.
            svr = SVR(kernel="linear", cache_size=1000)
            steps = [("scale", MaxAbsScaler()), ("svr", svr)]
        super().__init__(
            steps,
            transform_input=transform_input,
            memory=memory,
            verbose=verbose,
        )


_MODEL_CONSTRUCTORS = {
    "rf": lambda: RandomForestRegressor(random_state=RANDOM_STATE),
    "gb": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "xgb": lambda: XGBRegressor(random_state=RANDOM_STATE),
    # Deterministic, so no seed. Fit time grows with the square of the sample
    # count or worse: fine at tens of thousands, not at 150,000.
    "svm": lambda: ScaledSVR(),
}

# Alias lists map many human-friendly synonyms to the canonical short keys above
_MODEL_ALIASES = {
    "rf": [
        "rf",
        "r f",
        "rforest",
        "r forest",
        "randomforest",
        "random forest",
        "random forest regressor",
        "random-forest",
        "random_forest",
    ],
    "gb": [
        "gb",
        "gradient boosting",
        "gradient_boosting",
        "gradient-boosting",
        "gradientboosting",
    ],
    "xgb": [
        "xgb",
        "xgboost",
        "x gboost",
        "x g b",
    ],
    "svm": [
        "svm",
        "svr",
        "linear svr",
        "linear svm",
        "support vector",
        "support vector machine",
        "support vector regression",
        "support vector regressor",
    ],
}


def _set_threads(estimator: Any, n_threads: int) -> Any:
    """Set an estimator's own thread count, where it has one."""
    if "n_jobs" in estimator.get_params():
        estimator.set_params(n_jobs=n_threads)
    return estimator


def _coerce_param_grid(raw: Optional[object]) -> object:
    """Ensure ParameterGrid input is valid; fall back to [{}] when empty."""

    if raw is None:
        return [{}]
    if isinstance(raw, (list, tuple)) and not raw:
        return [{}]
    if isinstance(raw, dict) and not raw:
        return [{}]
    return raw


def _normalize_alias(name: str) -> str:
    # Normalize a model alias by lowercasing and removing non-alphanumeric
    # characters (except spaces), collapsing multiple spaces.
    s = (name or "").lower()
    # keep only alnum and spaces, collapse spaces
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s


def _sse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of squared error on a fold."""
    return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _sst(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Total sum of squares on a fold, around that fold's own mean.

    Recorded per fold because a mean of per-fold R2 cannot be reweighted
    afterwards: pooled R2 needs the numerator and denominator separately.
    Zero when the fold's label is constant, which is how a fold with no
    variance to explain is told apart from one the model got wrong.
    """
    y = np.asarray(y_true, dtype=float)
    return float(np.sum((y - y.mean()) ** 2))


def _resolve_model_alias(name: str) -> str:
    """Return canonical short key for a model alias (e.g. 'random forest' ->
    'rf').

    Raises ValueError if no match.
    """
    if not isinstance(name, str):
        raise ValueError(f"Model alias must be a string, got: {type(name)}")
    norm = _normalize_alias(name)
    # first try exact match against alias lists
    for canon, aliases in _MODEL_ALIASES.items():
        for a in aliases:
            if _normalize_alias(a) == norm:
                return canon
    # fallback: allow direct canonical key (already normalized)
    if norm in _MODEL_CONSTRUCTORS:
        return norm
    raise ValueError(f"Unknown model alias: {name}")


# Define a CrossValidator class to handle cross-validation process
def _cv_parallelism(
    n_jobs: Optional[int], max_combos: int, cpu_total: int
) -> Tuple[int, int]:
    """Workers across combinations, and across one combination's folds.

    Only one level fans out, to avoid nested over-subscription. A single
    combination gives its folds the whole budget: an explicit `n_jobs` is a
    limit, because other jobs may be sharing the node. With none set, -1
    lets joblib count the cores this job can actually use, which on a
    shared node can be far fewer than `cpu_total`.
    """
    requested = cpu_total if n_jobs is None else int(n_jobs)
    parallel_jobs = max(1, min(requested, max_combos))
    if parallel_jobs > 1:
        return parallel_jobs, 1
    return 1, -1 if n_jobs is None else max(1, requested)


def _model_threads(
    n_jobs: Optional[int], parallel_jobs: int, inner_cv_jobs: int
) -> int:
    """Threads each model may use, so the three levels share one budget.

    Combination workers x fold workers x model threads stays within
    `n_jobs`. Left to itself XGBoost takes every core the node reports
    -- not the job's allocation -- in every worker at once, while a
    random forest stays on one core.
    """
    # int(): joblib is untyped, so mypy sees its count as Any.
    budget = int(effective_n_jobs(-1 if n_jobs is None else n_jobs))
    folds = budget if inner_cv_jobs == -1 else max(1, inner_cv_jobs)
    return max(1, budget // (max(1, parallel_jobs) * folds))


class CrossValidator:
    """Base class to perform cross-validation on microbiome datasets using
    specified models and CV schemes.

    It supports multiple feature sets, labels, and CV schemes, and can handle
    hyperparameter tuning
    """

    def __init__(
        self,
        dataset: Dataset,
        models: Union[object, List[object]],
        label: Optional[Union[str, List[str]]] = None,
        scheme: Optional[Union[str, List[str]]] = None,
        feature_set: Optional[Union[str, List[str]]] = None,
        full_fit: bool = False,
        full_fit_oob: bool = True,
    ) -> None:
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            required = ("feature_sets", "labels", "splits")
            missing = [a for a in required if not hasattr(dataset, a)]
            if missing:
                raise TypeError(
                    "dataset must be an instance of Dataset or a dataset-like object with attributes: "
                    f"{', '.join(required)}. Missing: {', '.join(missing)}"
                )
            self.dataset = dataset
        self.models: List[object]
        if isinstance(models, list):
            self.models = models
        else:
            self.models = [models]
        self.label: Optional[Union[str, List[str]]] = label
        self.scheme: Optional[Union[str, List[str]]] = scheme
        self.feature_set: Optional[Union[str, List[str]]] = feature_set
        # Score the model that is already fitted on the whole CV pool.
        # Off by default: it costs one extra predict per combination,
        # and with `full_fit_oob` an out-of-bag pass on every fit.
        self.full_fit: bool = bool(full_fit)
        self.full_fit_oob: bool = bool(full_fit_oob)
        self._best_validation_r2: float = float("-inf")
        self.best_result_key: Optional[str] = None
        self.best_result: Optional[CV_Result] = None
        self.best_model_estimator: Optional[Any] = None
        self._best_validation_r2_by_label: Dict[Optional[str], float] = {}
        self.best_result_key_by_label: Dict[Optional[str], str] = {}
        self.best_result_by_label: Dict[Optional[str], CV_Result] = {}
        self.best_model_estimator_by_label: Dict[Optional[str], Any] = {}
        # Best per (label, scheme). Lets callers compare CV schemes (e.g.
        # random vs. grouped) for the same label instead of only seeing the
        # overall winner, which is usually the leakier random scheme.
        self._best_validation_r2_by_label_scheme: Dict[
            LabelSchemeKey, float
        ] = {}
        self.best_result_key_by_label_scheme: Dict[LabelSchemeKey, str] = {}
        self.best_result_by_label_scheme: Dict[LabelSchemeKey, CV_Result] = {}
        self.best_model_estimator_by_label_scheme: Dict[
            LabelSchemeKey, Any
        ] = {}

    def _reset_best_tracking(self) -> None:
        """Clear cached best-model selections before a fresh run."""
        self._best_validation_r2 = float("-inf")
        self.best_result_key = None
        self.best_result = None
        self.best_model_estimator = None
        self._best_validation_r2_by_label.clear()
        self.best_result_key_by_label.clear()
        self.best_result_by_label.clear()
        self.best_model_estimator_by_label.clear()
        self._best_validation_r2_by_label_scheme.clear()
        self.best_result_key_by_label_scheme.clear()
        self.best_result_by_label_scheme.clear()
        self.best_model_estimator_by_label_scheme.clear()

    def _print_best_models_by_label(self) -> None:
        """Print best model summary per label and per label/scheme."""
        for label, cv_result in sorted(
            self.best_result_by_label.items(),
            key=lambda item: "" if item[0] is None else str(item[0]),
        ):
            model = cv_result.model
            model_name = (
                model.__class__.__name__ if model is not None else "None"
            )
            avg_r2 = cv_result.rank_score
            avg_r2_text = (
                f"{float(avg_r2):.6f}" if avg_r2 is not None else "None"
            )
            label_text = "None" if label is None else str(label)
            best_key = self.best_result_key_by_label.get(label, "")
            print(
                f"[best][label={label_text}] model={model_name} r2={avg_r2_text} key={best_key}"
            )

        for (label, scheme), cv_result in sorted(
            self.best_result_by_label_scheme.items(),
            key=lambda item: (
                "" if item[0][0] is None else str(item[0][0]),
                "" if item[0][1] is None else str(item[0][1]),
            ),
        ):
            model = cv_result.model
            model_name = (
                model.__class__.__name__ if model is not None else "None"
            )
            avg_r2 = cv_result.rank_score
            avg_r2_text = (
                f"{float(avg_r2):.6f}" if avg_r2 is not None else "None"
            )
            label_text = "None" if label is None else str(label)
            scheme_text = "None" if scheme is None else str(scheme)
            best_key = self.best_result_key_by_label_scheme.get(
                (label, scheme), ""
            )
            print(
                f"[best][label={label_text}][scheme={scheme_text}] "
                f"model={model_name} r2={avg_r2_text} key={best_key}"
            )

    @staticmethod
    def load_param_grids(path: str) -> dict:
        # Load hyperparameter grids from a YAML file
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    def _update_best_model(
        self,
        key: str,
        cv_result: CV_Result,
        estimator: Optional[Any] = None,
    ) -> None:
        # Which result wins a label/scheme: pooled R2 when the run has it,
        # else the mean. Same property the sweep headline ranks on, so the
        # exported best model and the headline cannot disagree.
        avg_r2 = cv_result.rank_score
        if avg_r2 is None:
            return
        if avg_r2 > self._best_validation_r2:
            self._best_validation_r2 = avg_r2
            self.best_result_key = key
            self.best_result = cv_result
            self.best_model_estimator = estimator

        label_key = cv_result.label
        best_for_label = self._best_validation_r2_by_label.get(label_key)
        if best_for_label is None or avg_r2 > best_for_label:
            self._best_validation_r2_by_label[label_key] = avg_r2
            self.best_result_key_by_label[label_key] = key
            self.best_result_by_label[label_key] = cv_result
            self.best_model_estimator_by_label[label_key] = estimator

        ls_key: LabelSchemeKey = (label_key, cv_result.scheme)
        best_for_ls = self._best_validation_r2_by_label_scheme.get(ls_key)
        if best_for_ls is None or avg_r2 > best_for_ls:
            self._best_validation_r2_by_label_scheme[ls_key] = avg_r2
            self.best_result_key_by_label_scheme[ls_key] = key
            self.best_result_by_label_scheme[ls_key] = cv_result
            self.best_model_estimator_by_label_scheme[ls_key] = estimator

    def run(
        self,
        param_path: str = "parameters.yaml",
        n_jobs: Optional[int] = None,
    ) -> dict:
        """Manages the cross-validation process across different feature sets,
        labels, schemes, models, and hyperparameter combinations.

        returns a dictionary mapping each unique combination to its CV_Result.
        CV_Result contains per-fold R², MSE, and cross-validation scores.
        Inputs:
            label: Union[str, List[str]] = None (specific label(s) to use; defaults to self.label or all label attributes in dataset)
            scheme: Union[str, List[str]] = None (specific CV scheme(s) to use; defaults to all schemes in dataset)
            param_path: str = "hyperparameters.yaml" (path to YAML file with hyperparameter grids)
            n_jobs: Optional[int] = None (workers for this run. Combinations run in parallel when there are several, otherwise the folds of the one combination do; defaults to detected CPU cores)
        Outputs:
            Dict[str, CV_Result]: A dictionary mapping each unique combination of feature set, label, scheme, model, and hyperparameters to its CV_Result.
        """

        self._reset_best_tracking()
        results: Dict[str, CV_Result] = {}

        # Determine label and scheme to use
        labels_attr = getattr(self.dataset, "labels", None)
        if self.label is not None:
            label_to_use = self.label
        elif labels_attr is None:
            label_to_use = []
        else:
            cols = list(labels_attr.columns)
            label_to_use = [c for c in cols if c != "sample"]
        if self.scheme is not None:
            scheme_to_use = self.scheme
        else:
            schemes = [s for _, s, _ in self.dataset.iter_cv_folds()]
            unique_schemes = list(dict.fromkeys(schemes))
            scheme_to_use = unique_schemes

        mapping = self._prepare_inputs(
            self.dataset,
            fillna=0.0,
            label=label_to_use,
            scheme=scheme_to_use,
            feature_set=self.feature_set,
        )
        params_grid = self.load_param_grids(param_path)
        logger.info(
            "Starting cross-validation over feature/label/scheme "
            "combinations with models: %s",
            self.models,
        )

        scorers = {
            "r2": "r2",
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
            "sse": make_scorer(_sse, greater_is_better=True),
            "sst": make_scorer(_sst, greater_is_better=True),
        }

        def payloads() -> (
            Iterator[
                Tuple[
                    str,
                    Optional[str],
                    Optional[str],
                    Optional[str],
                    List[str],
                    np.ndarray,
                    np.ndarray,
                    PredefinedSplit,
                ]
            ]
        ):
            """Validate each combination and hand it straight to a worker.

            A generator, so joblib pulls payloads as workers free up
            (pre_dispatch) instead of every feature matrix being built
            and held before the first fit starts.
            """
            for key, X_np, y_np, meta in mapping:
                try:
                    feature_name, label_name, scheme_name = key.split("::")
                except Exception:
                    feature_name = key
                    label_name = None
                    scheme_name = None

                try:
                    fold_arr = np.asarray(meta.fold).ravel().astype(int)
                except Exception:
                    logging.warning(
                        "Fold assignments for %s could not be cast to integers; skipping",
                        key,
                    )
                    continue

                unique_folds, counts = np.unique(fold_arr, return_counts=True)
                # PredefinedSplit reads -1 as "always train, never validate";
                # leave-one-group-out parks single-sample groups there. It is
                # not a fold, so it must not be counted as one nor trip the
                # fold-size check below.
                _keep = unique_folds >= 0
                unique_folds, counts = unique_folds[_keep], counts[_keep]
                if len(unique_folds) < 2:
                    logging.warning(
                        "Not enough folds for %s (found %d); skipping",
                        key,
                        len(unique_folds),
                    )
                    continue
                if np.min(counts) < 2:
                    logging.getLogger(__name__).warning(
                        "One or more folds for %s contain fewer than 2 samples; skipping",
                        key,
                    )
                    continue

                yield (
                    key,
                    feature_name,
                    label_name,
                    scheme_name,
                    meta.feature_columns,
                    np.asarray(X_np),
                    np.asarray(y_np, dtype=float),
                    PredefinedSplit(test_fold=fold_arr),
                )

        cpu_total = os.cpu_count() or 1

        def _n(value: Any, default: int) -> int:
            if value is None:
                return default
            if isinstance(value, (list, tuple, set)):
                return max(1, len(value))
            return 1

        # Upper bound on combinations: the exact count is only known while
        # streaming, so bound it from the requested feature sets, labels
        # and schemes rather than materialising them to count.
        max_combos = (
            _n(
                self.feature_set,
                max(1, len(getattr(self.dataset, "feature_sets", {}) or {})),
            )
            * _n(label_to_use, 1)
            * _n(scheme_to_use, 1)
        )
        parallel_jobs, inner_cv_jobs = _cv_parallelism(
            n_jobs, max_combos, cpu_total
        )
        model_threads = _model_threads(n_jobs, parallel_jobs, inner_cv_jobs)

        logger.info(
            "Dispatching CV combinations across %d worker(s) "
            "(inner_cv_jobs=%d, model_threads=%d)",
            parallel_jobs,
            inner_cv_jobs,
            model_threads,
        )

        parallel = Parallel(n_jobs=parallel_jobs)
        job_outputs = parallel(
            delayed(CrossValidator._process_combo)(
                payload,
                self.models,
                params_grid,
                scorers,
                inner_cv_jobs,
                self.full_fit,
                self.full_fit_oob,
                model_threads,
            )
            for payload in payloads()
        )

        for combo_key, partial_results, best_info in job_outputs:
            results.update(partial_results)
            logger.info(
                "Finished combination %s with %d trained model(s)",
                combo_key,
                len(partial_results),
            )
            if best_info is not None:
                best_key, best_cv_result, best_estimator, _ = best_info
                self._update_best_model(
                    best_key, best_cv_result, estimator=best_estimator
                )

        self._print_best_models_by_label()
        return results

    @staticmethod
    def _process_combo(
        payload: Tuple[
            str,
            Optional[str],
            Optional[str],
            Optional[str],
            List[str],
            np.ndarray,
            np.ndarray,
            PredefinedSplit,
        ],
        models: List[object],
        params_grid: Dict[str, Any],
        scorers: Dict[str, Any],
        inner_cv_jobs: int,
        full_fit: bool = False,
        full_fit_oob: bool = True,
        model_threads: Optional[int] = None,
    ) -> Tuple[
        str,
        Dict[str, CV_Result],
        Optional[Tuple[str, CV_Result, Any, float]],
    ]:
        """CV every model and setting on one combination.

        `model_threads` sets each estimator's own n_jobs (see
        `_model_threads`); None leaves the library default.
        """
        (
            key,
            feature_name,
            label_name,
            scheme_name,
            feature_names,
            X_arr,
            y_arr,
            ps,
        ) = payload
        results: Dict[str, CV_Result] = {}
        best_info: Optional[Tuple[str, CV_Result, Any, float]] = None

        for model in models:
            if isinstance(model, str):
                model_key = _resolve_model_alias(model)
                grid = _coerce_param_grid(params_grid.get(model_key, [{}]))
                for params in ParameterGrid(grid):
                    estimator = _MODEL_CONSTRUCTORS[model_key]()
                    estimator.set_params(**params)
                    if model_threads:
                        _set_threads(estimator, model_threads)
                    cvres = cross_validate(
                        estimator,
                        X_arr,
                        y_arr,
                        cv=ps,
                        scoring=scorers,
                        return_train_score=False,
                        n_jobs=inner_cv_jobs,
                    )
                    per_r2 = list(cvres["test_r2"])
                    per_mse = [-m for m in cvres["test_mse"]]
                    per_sse = list(cvres["test_sse"])
                    per_sst = list(cvres["test_sst"])
                    result_key = (
                        f"{key}::{estimator.__class__.__name__}::{params}"
                    )
                    # Asked for before the fit, so the out-of-bag pass
                    # rides along with it instead of costing a refit.
                    if full_fit and full_fit_oob:
                        enable_oob(estimator)
                    estimator.fit(X_arr, y_arr)
                    cv_result = CV_Result(
                        feature_set=feature_name,
                        label=label_name,
                        scheme=scheme_name,
                        cross_val_scores=per_r2,
                        validation_r2_per_fold=per_r2,
                        validation_mse_per_fold=per_mse,
                        validation_sse_per_fold=per_sse,
                        validation_sst_per_fold=per_sst,
                        best_params=dict(params),
                        trained_model=estimator,
                        feature_names=feature_names,
                        n_label=len(y_arr),
                    )
                    results[result_key] = cv_result
                    avg_r2 = cv_result.rank_score
                    if avg_r2 is not None and (
                        best_info is None or avg_r2 > best_info[3]
                    ):
                        best_info = (result_key, cv_result, estimator, avg_r2)
                    logger.info(
                        f"Completed CV for {result_key} with params {params}"
                    )
                continue

            est_name = model.__class__.__name__.lower()
            grid = _coerce_param_grid(params_grid.get(est_name, [{}]))
            for params in ParameterGrid(grid):
                estimator = clone(model)
                estimator.set_params(**params)
                if model_threads:
                    _set_threads(estimator, model_threads)
                cvres = cross_validate(
                    estimator,
                    X_arr,
                    y_arr,
                    cv=ps,
                    scoring=scorers,
                    return_train_score=False,
                    n_jobs=inner_cv_jobs,
                )
                per_r2 = list(cvres["test_r2"])
                per_mse = [-v for v in cvres["test_mse"]]
                per_sse = list(cvres["test_sse"])
                per_sst = list(cvres["test_sst"])
                result_key = f"{key}::{estimator.__class__.__name__}::{params}"
                if full_fit and full_fit_oob:
                    enable_oob(estimator)
                estimator.fit(X_arr, y_arr)
                cv_result = CV_Result(
                    feature_set=feature_name,
                    label=label_name,
                    scheme=scheme_name,
                    cross_val_scores=per_r2,
                    validation_r2_per_fold=per_r2,
                    validation_mse_per_fold=per_mse,
                    validation_sse_per_fold=per_sse,
                    validation_sst_per_fold=per_sst,
                    best_params=dict(params),
                    trained_model=estimator,
                    feature_names=feature_names,
                    n_label=len(y_arr),
                )
                results[result_key] = cv_result
                avg_r2 = cv_result.rank_score
                if avg_r2 is not None and (
                    best_info is None or avg_r2 > best_info[3]
                ):
                    best_info = (result_key, cv_result, estimator, avg_r2)
                logger.info(
                    f"Completed CV for {result_key} with params {params}"
                )

        # Only the combination's own winner is diagnosed. Every config
        # was already fitted on the full pool above, but scoring all of
        # them would cost a predict over the whole training set per
        # config to describe models nothing downstream will look at.
        if full_fit and best_info is not None:
            _, best_cv_result, best_estimator, best_score = best_info
            best_cv_result.full_fit = full_fit_diagnostics(
                best_estimator, X_arr, y_arr, cv_score=best_score
            )

        return key, results, best_info

    def run_grid(
        self,
        param_path: str = "hyperparameters.yaml",
        n_jobs: Optional[int] = None,
    ) -> dict:
        """One sklearn GridSearchCV per feature set x label x scheme x model.

        Every candidate is scored by GridSearchCV on the predefined folds.
        Each estimator trains on one thread and GridSearchCV spreads the fits
        across `n_jobs` workers, so the two levels never multiply. The winner
        is the best pooled R2, the score the rest of the workflow ranks on,
        and is then refit once with all `n_jobs` threads.

        Inputs:
            param_path: YAML of grids, one top-level key per model.
            n_jobs: Workers for this run. None uses every core this job can
                actually use, as counted by joblib.
        Outputs:
            Dict[str, CV_Result]: the winner per feature set x label x scheme
            x model. `grid_results` on each holds sklearn's `cv_results_` for
            all its candidates; `CV_Result.export_result` writes those to
            `grid_results.csv`.
        """
        self._reset_best_tracking()
        workers = effective_n_jobs(-1 if n_jobs is None else n_jobs)

        # Determine label and scheme to use
        labels_attr = getattr(self.dataset, "labels", None)
        if self.label is not None:
            label_to_use = self.label
        elif labels_attr is None:
            label_to_use = []
        else:
            cols = list(labels_attr.columns)
            label_to_use = [c for c in cols if c != "sample"]
        if self.scheme is not None:
            scheme_to_use = self.scheme
        else:
            schemes = [s for _, s, _ in self.dataset.iter_cv_folds()]
            unique_schemes = list(dict.fromkeys(schemes))
            scheme_to_use = unique_schemes

        results: Dict[str, CV_Result] = {}
        mapping = self._prepare_inputs(
            self.dataset,
            fillna=0.0,
            label=label_to_use,
            scheme=scheme_to_use,
            feature_set=self.feature_set,
        )
        param_grids = self.load_param_grids(param_path)
        scorers = {
            "r2": "r2",
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
            "sse": make_scorer(_sse, greater_is_better=True),
            "sst": make_scorer(_sst, greater_is_better=True),
        }

        for key, X_np, y_np, meta in mapping:
            try:
                feature_name, label_name, scheme_name = key.split("::")
            except Exception:
                feature_name, label_name, scheme_name = key, None, None

            fold_arr = meta.fold

            # ensure integer fold array and check per-fold sample counts
            try:
                fold_arr = np.asarray(fold_arr).ravel().astype(int)
            except Exception:
                logging.warning(
                    "Fold assignments for %s could not be cast to integers; skipping",
                    key,
                )
                continue

            unique_folds, counts = np.unique(fold_arr, return_counts=True)
            # PredefinedSplit reads -1 as "always train, never validate";
            # leave-one-group-out parks single-sample groups there. It is
            # not a fold, so it must not be counted as one nor trip the
            # fold-size check below.
            _keep = unique_folds >= 0
            unique_folds, counts = unique_folds[_keep], counts[_keep]
            if len(unique_folds) < 2:
                logging.warning(
                    "Not enough folds for %s (found %d); skipping",
                    key,
                    len(unique_folds),
                )
                continue
            if np.min(counts) < 2:
                logging.warning(
                    "One or more folds for %s contain fewer than 2 samples; skipping",
                    key,
                )
                continue

            ps = PredefinedSplit(test_fold=fold_arr)
            X_arr = np.asarray(X_np)
            y_arr = np.asarray(y_np, dtype=float)
            feature_columns = meta.feature_columns

            for model in self.models:
                if isinstance(model, str):
                    model_key = _resolve_model_alias(model)
                    # An empty `rf:` means defaults on purpose. A missing key
                    # is a typo or a commented-out block, and searching
                    # nothing would still report a "winner".
                    if model_key not in param_grids:
                        raise ValueError(
                            f"No grid for model '{model}' in {param_path}: "
                            f"expected the key '{model_key}', found "
                            f"{sorted(param_grids)}"
                        )
                    grid = _coerce_param_grid(param_grids[model_key])
                    estimator = _MODEL_CONSTRUCTORS[model_key]()
                else:
                    # estimator instance: prefer class-name key, then canonical short keys
                    est_name = model.__class__.__name__.lower()
                    raw_grid = param_grids.get(est_name, None)
                    if raw_grid is None:
                        canon = None
                        for k, ctor in _MODEL_CONSTRUCTORS.items():
                            try:
                                if isinstance(model, ctor().__class__):
                                    canon = k
                                    break
                            except Exception:
                                continue
                        grid = (
                            _coerce_param_grid(param_grids.get(canon, [{}]))
                            if canon is not None
                            else [{}]
                        )
                    estimator = clone(model)
                    if raw_grid is not None:
                        grid = _coerce_param_grid(raw_grid)
                # refit=False, because sklearn's refit would run on one
                # thread and pick by mean R2. Refitting below uses the whole
                # budget, and asks for the out-of-bag score on this one fit
                # rather than on every candidate.
                gs = GridSearchCV(
                    estimator=_set_threads(estimator, 1),
                    param_grid=grid,
                    cv=ps,
                    scoring=scorers,
                    refit=False,
                    n_jobs=workers,
                )
                gs.fit(X_arr, y_arr)
                cvres = gs.cv_results_
                best_idx = best_candidate_index(cvres)
                best_params = dict(cvres["params"][best_idx])
                best_estimator = _set_threads(
                    clone(estimator).set_params(**best_params), workers
                )
                if self.full_fit and self.full_fit_oob:
                    enable_oob(best_estimator)
                best_estimator.fit(X_arr, y_arr)
                n_splits = ps.get_n_splits()
                per_r2 = [
                    cvres[f"split{i}_test_r2"][best_idx]
                    for i in range(n_splits)
                ]
                per_mse = [
                    -cvres[f"split{i}_test_mse"][best_idx]
                    for i in range(n_splits)
                ]  # negate because scorer is neg MSE
                per_sse = [
                    cvres[f"split{i}_test_sse"][best_idx]
                    for i in range(n_splits)
                ]
                per_sst = [
                    cvres[f"split{i}_test_sst"][best_idx]
                    for i in range(n_splits)
                ]

                result_key = f"{key}::{best_estimator.__class__.__name__}"
                cv_result = CV_Result(
                    feature_set=feature_name,
                    label=label_name,
                    scheme=scheme_name,
                    cross_val_scores=per_r2,
                    validation_r2_per_fold=per_r2,
                    validation_mse_per_fold=per_mse,
                    validation_sse_per_fold=per_sse,
                    validation_sst_per_fold=per_sst,
                    best_params=best_params,
                    trained_model=best_estimator,
                    feature_names=feature_columns,
                    n_label=len(y_arr),
                    grid_results=cvres,
                )
                if self.full_fit:
                    cv_result.full_fit = full_fit_diagnostics(
                        best_estimator,
                        X_arr,
                        y_arr,
                        cv_score=cv_result.rank_score,
                    )
                results[result_key] = cv_result
                self._update_best_model(
                    result_key, cv_result, estimator=best_estimator
                )
                logger.info(
                    "Grid search %s: best of %d candidate(s) is %s, pooled "
                    "R2 %s",
                    result_key,
                    len(cvres["params"]),
                    best_params,
                    cv_result.rank_score,
                )

        self._print_best_models_by_label()
        return results

    def _prepare_inputs(
        self,
        dataset: Dataset,
        fillna: float = 0.0,
        label: Optional[Union[str, List[str]]] = None,
        scheme: Optional[Union[str, List[str]]] = None,
        feature_set: Optional[Union[str, List[str]]] = None,
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray, ComboMeta]]:
        """Yield one (key, X float32, y, ComboMeta) per feature set x label
        x CV scheme.

        A generator on purpose: the feature matrices are the largest
        objects in the run, and returning them in a dict kept every
        combination alive at once. Callers iterate exactly once.
        """
        # 1. Check if the dataset has feature sets, labels, and splits
        feature_sets = getattr(dataset, "feature_sets", None)
        if feature_sets is None:
            raise RuntimeError("dataset has no attribute 'feature_sets'")
        labels_attr = getattr(dataset, "labels", None)
        if labels_attr is None and label is not None:
            raise RuntimeError("dataset has no attribute 'labels'")
        splits = getattr(dataset, "splits", None)
        if splits is None:
            raise RuntimeError(
                "dataset has no 'splits' attribute; cannot find cv_schemes"
            )

        if feature_set is not None:
            if isinstance(feature_set, (list, tuple, set)):
                requested_features = list(feature_set)
            else:
                requested_features = [feature_set]
            feature_names: List[str] = []
            for req in requested_features:
                if req in feature_sets:
                    feature_names.append(req)
                else:
                    logger.warning(
                        "Requested feature set '%s' not found on dataset; skipping.",
                        req,
                    )
                    logger.warning(
                        "Available feature sets: %s", list(feature_sets.keys())
                    )
            if not feature_names:
                return
        else:
            feature_names = list(feature_sets.keys())

        # 2. Materialize feature sets and labels into DataFrames
        for feat_name in feature_names:
            feat_obj = feature_sets[feat_name]
            # Stay lazy: collecting the whole feature table here would
            # hold every sample and taxon in memory for the entire
            # label x scheme loop, when each combination only needs the
            # rows of its own folds.
            feat_lf = getattr(feat_obj, "features", None)
            if not isinstance(feat_lf, pl.LazyFrame):
                feat_lf = feat_obj.to_df().lazy()
            feat_schema = feat_lf.collect_schema().names()
            acc_col = next(
                (c for c in ("sample", "acc") if c in feat_schema), None
            )
            if acc_col is None:
                raise RuntimeError(
                    "featureset missing accession column ('sample' or 'acc')"
                )
            if acc_col != "sample":
                feat_lf = feat_lf.rename({acc_col: "sample"})
            feature_cols = [c for c in feat_schema if c != acc_col]

            label_keys: List[str] = []

            # Determine label keys to iterate. support single label or list of labels
            if label is not None:
                if isinstance(label, (list, tuple, set)):
                    requested = list(label)
                else:
                    requested = [label]

                # available names from dataset.labels (if present)
                available_names = [name for name, _ in dataset.iter_labels()]
                label_keys = []
                for req in requested:
                    if req in available_names:
                        label_keys.append(req)
                    else:
                        logger.warning(
                            "Requested label '%s' not found on dataset; skipping.",
                            req,
                        )
                        logger.warning("Available labels: %s", available_names)
                # nothing found -> continue to next feature set
                if not label_keys:
                    continue
            else:
                label_keys = [name for name, _ in dataset.iter_labels()]

            # Check the label DataFrame for each label key using cutting logic in polars and then delete nulls
            for label_name in label_keys:
                # materialize label DataFrame using dataset API
                if labels_attr is not None:
                    label_df = labels_attr.select(["sample", label_name])
                else:
                    # fallback: find the label DataFrame from iter_labels()
                    label_df = None
                    for n, df in dataset.iter_labels():
                        if n == label_name:
                            label_df = df
                            break
                    if label_df is None:
                        raise RuntimeError(
                            f"Could not materialize label '{label_name}' from dataset"
                        )

                split_manager = None
                if isinstance(splits, dict) and label_name in splits:
                    split_manager = splits[label_name]
                elif not isinstance(splits, dict) and hasattr(
                    splits, "get_cv_schemes"
                ):
                    try:
                        cv_map = splits.get_cv_schemes(label_name)
                        split_manager = type(
                            "SM", (), {"cv_schemes": cv_map}
                        )()
                    except Exception:
                        split_manager = None
                elif not isinstance(splits, dict) and hasattr(splits, "get"):
                    try:
                        split_manager = splits.get(label_name)
                    except Exception:
                        split_manager = None
                else:
                    split_manager = getattr(splits, label_name, None)

                if split_manager is None:
                    logger.warning(
                        "No split manager found for label '%s'", label_name
                    )
                    continue

                available_cv_schemes: Optional[Dict[str, Any]] = None
                if isinstance(split_manager, dict):
                    available_cv_schemes = split_manager
                else:
                    available_cv_schemes = getattr(
                        split_manager, "cv_schemes", None
                    )
                if scheme is not None:
                    if isinstance(scheme, (list, tuple, set)):
                        requested_schemes = list(scheme)
                    else:
                        requested_schemes = [scheme]

                    # If the split manager exposes a mapping, filter it directly
                    if isinstance(available_cv_schemes, dict):
                        cv_schemes = {
                            k: v
                            for k, v in available_cv_schemes.items()
                            if k in requested_schemes
                        }
                        for req in requested_schemes:
                            if req not in available_cv_schemes:
                                logger.warning(
                                    "Requested scheme '%s' not found for label '%s'.",
                                    req,
                                    label_name,
                                )
                                logger.warning(
                                    "Available schemes: %s",
                                    list(available_cv_schemes.keys()),
                                )
                    else:
                        # Fallback: build available schemes from dataset.iter_cv_folds
                        try:
                            all_schemes = {
                                s: cv_df
                                for lbl, s, cv_df in dataset.iter_cv_folds(
                                    label=label_name
                                )
                            }
                        except Exception:
                            all_schemes = {}
                        cv_schemes = {
                            k: v
                            for k, v in all_schemes.items()
                            if k in requested_schemes
                        }
                        for req in requested_schemes:
                            if req not in all_schemes:
                                logger.warning(
                                    "Requested scheme '%s' not found for label '%s'.",
                                    req,
                                    label_name,
                                )
                else:
                    cv_schemes = getattr(split_manager, "cv_schemes", {})
                for scheme_name, scheme_table in cv_schemes.items():
                    if isinstance(scheme_table, dict):
                        items = [
                            (str(k), v)
                            for k, v in scheme_table.items()
                            if v is not None
                        ]
                        if not items:
                            logging.warning(
                                "CV scheme '%s' for label '%s' contains no assigned folds; skipping",
                                scheme_name,
                                label_name,
                            )
                            continue
                        samples = [k for k, _ in items]
                        try:
                            folds = [int(v) for _, v in items]
                        except Exception:
                            logging.warning(
                                "CV scheme '%s' for label '%s' has non-integer fold values; skipping",
                                scheme_name,
                                label_name,
                            )
                            continue
                        cv_df = pl.DataFrame(
                            {"sample": samples, "fold": folds}
                        )
                    elif isinstance(scheme_table, pl.DataFrame):
                        cv_df = scheme_table
                    else:
                        try:
                            cv_df = pl.DataFrame(scheme_table)
                        except Exception:
                            logging.warning(
                                "Unsupported cv scheme format for '%s' in label '%s'",
                                scheme_name,
                                label_name,
                            )
                            continue

                    try:
                        n_folds = (
                            cv_df.select(pl.col("fold"))
                            .drop_nulls()
                            .unique()
                            .height
                        )
                    except Exception:
                        folds_list = []
                        for r in cv_df.to_dicts():
                            if isinstance(r, dict):
                                folds_list.append(r.get("fold"))
                            elif isinstance(r, (list, tuple)) and len(r) > 1:
                                folds_list.append(r[1])
                            else:
                                folds_list.append(None)
                        n_folds = len(
                            set([f for f in folds_list if f is not None])
                        )

                    if n_folds < 2:
                        logging.warning(
                            "CV scheme '%s' for label '%s' has %d populated fold(s); skipping (need >=2)",
                            scheme_name,
                            label_name,
                            n_folds,
                        )
                        continue

                    # 3. Join labels and folds FIRST (both tiny), so the
                    # fold check can reject a combination before any
                    # feature data is read.
                    ldf = label_df
                    label_value_col = next(
                        (c for c in ldf.columns if c != "sample"), None
                    )
                    if label_value_col is None:
                        raise RuntimeError(
                            "label DataFrame has no value column after materialize"
                        )
                    keys = cv_df.join(ldf, on="sample", how="left").filter(
                        ~pl.col(label_value_col).is_null()
                    )
                    n_folds_after = (
                        keys.select(pl.col("fold"))
                        .drop_nulls()
                        .unique()
                        .height
                    )
                    if n_folds_after < 2:
                        logging.warning(
                            "After dropping null labels CV scheme '%s' for label '%s' has %d fold(s); skipping",
                            scheme_name,
                            label_name,
                            n_folds_after,
                        )
                        continue

                    # 4. Materialise the feature matrix ONCE, for these
                    # samples only. Joining, filling and casting inside one
                    # lazy plan means the full-width table is never held:
                    # polars can also push the sample filter and column
                    # selection into the scan when the features came from
                    # disk. float32 matches what sklearn's tree code uses
                    # internally, so float64 would only buy a conversion
                    # copy per fit.
                    joined = (
                        keys.lazy()
                        .join(feat_lf, on="sample", how="left")
                        # One expression over all the feature columns,
                        # not one per column: a genus table has ~60k of
                        # them, and polars' optimiser walking 60k
                        # expression nodes is itself a memory problem.
                        .with_columns(
                            pl.col(feature_cols)
                            .fill_null(fillna)
                            .cast(pl.Float32, strict=False)
                        )
                        .collect()
                    )

                    drop_cols = ["sample", "fold", label_value_col]
                    X_np = joined.drop(drop_cols).to_numpy()
                    y_np = np.asarray(
                        joined.select(pl.col(label_value_col))
                        .to_numpy()
                        .ravel(),
                        dtype=float,
                    )

                    # Keep only what the CV loops need; retaining the frame
                    # would double the cost of every combination.
                    meta = ComboMeta(
                        fold=joined.select("fold")
                        .to_numpy()
                        .ravel()
                        .astype(int),
                        feature_columns=[
                            c for c in joined.columns if c not in drop_cols
                        ],
                    )
                    logger.info(
                        "Prepared data for feature set '%s', label '%s', scheme '%s' with %d samples and %d features.",
                        feat_name,
                        label_name,
                        scheme_name,
                        X_np.shape[0],
                        X_np.shape[1],
                    )
                    # Streamed, not accumulated: only this combination's
                    # matrix is alive at a time.
                    yield (
                        f"{feat_name}::{label_name}::{scheme_name}",
                        X_np,
                        y_np,
                        meta,
                    )
