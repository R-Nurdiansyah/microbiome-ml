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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml  # type: ignore[import]
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    PredefinedSplit,
    cross_validate,
)

# from sklearn.linear_model import ElasticNet, LinearRegression
# from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from microbiome_ml.train.results import CV_Result
from microbiome_ml.wrangle.dataset import Dataset

logger = logging.getLogger(__name__)

# Mapping of model short names to their constructors
# Canonical constructors keyed by short names (keeps compatibility with hyperparameters.yaml)
_MODEL_CONSTRUCTORS = {
    "rf": lambda: RandomForestRegressor(),
    "gb": lambda: GradientBoostingRegressor(),
    "xgb": lambda: XGBRegressor(),
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
}


def _normalize_alias(name: str) -> str:
    s = (name or "").lower()
    # keep only alnum and spaces, collapse spaces
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = " ".join(s.split())
    return s


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
        cv_folds: int = 5,
        label: Optional[Union[str, List[str]]] = None,
        scheme: Optional[Union[str, List[str]]] = None,
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
        self.cv_folds = cv_folds
        self.label: Optional[Union[str, List[str]]] = label
        self.scheme: Optional[Union[str, List[str]]] = scheme
        self._best_validation_r2: float = float("-inf")
        self.best_result_key: Optional[str] = None
        self.best_result: Optional[CV_Result] = None
        self.best_model_estimator: Optional[Any] = None

    @staticmethod
    def load_param_grids(path: str) -> dict:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    def _update_best_model(
        self,
        key: str,
        cv_result: CV_Result,
        estimator: Optional[Any] = None,
    ) -> None:
        avg_r2 = cv_result.avg_validation_r2
        if avg_r2 is None:
            return
        if avg_r2 > self._best_validation_r2:
            self._best_validation_r2 = avg_r2
            self.best_result_key = key
            self.best_result = cv_result
            self.best_model_estimator = estimator

    def _select_n_jobs(
        self,
        grid: object,
        user_n_jobs: Optional[int],
        params_per_job: int = 2,
    ) -> int:
        if user_n_jobs is not None:
            return int(user_n_jobs)
        try:
            combos = len(list(ParameterGrid(grid)))
        except Exception:
            combos = 1
        cpus = os.cpu_count() or 1
        # determine number of parallel jobs: one job per `params_per_job` combos
        calc = max(1, combos // max(1, params_per_job))
        return min(cpus, calc)

    def run(
        self,
        param_path: str = "hyperparameters.yaml",
    ) -> dict:
        """Manages the cross-validation process across different feature sets,
        labels, schemes, models, and hyperparameter combinations.

        returns a dictionary mapping each unique combination to its CV_Result.
        CV_Result contains per-fold R², MSE, and cross-validation scores.
        Inputs:
            label: Union[str, List[str]] = None (specific label(s) to use; defaults to self.label or all label attributes in dataset)
            scheme: Union[str, List[str]] = None (specific CV scheme(s) to use; defaults to all schemes in dataset)
            param_path: str = "hyperparameters.yaml" (path to YAML file with hyperparameter grids)
        Outputs:
            Dict[str, CV_Result]: A dictionary mapping each unique combination of feature set, label, scheme, model, and hyperparameters to its CV_Result.
        """

        # Initialize results dictionary and needed inputs
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
            self.dataset, fillna=0.0, label=label_to_use, scheme=scheme_to_use
        )
        params_grid = self.load_param_grids(param_path)

        for key, (X_np, y_np, joined) in mapping.items():
            try:
                feature_name, label_name, scheme_name = key.split("::")
            except Exception:
                feature_name = key
                label_name = None
                scheme_name = None

            try:
                fold_arr = np.array(joined.select("fold").to_numpy()).ravel()
            except Exception:
                try:
                    fold_arr = np.array([r["fold"] for r in joined.to_dicts()])
                except Exception:
                    logging.warning(
                        "Could not extract fold assignments for %s; skipping",
                        key,
                    )
                    continue

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

            ps = PredefinedSplit(test_fold=fold_arr)  # uses your exact folds

            # scorers: r2 (higher better) and mse (we use neg mse and will negate back)
            scorers = {
                "r2": "r2",
                "mse": make_scorer(
                    mean_squared_error, greater_is_better=False
                ),
            }

            X_arr = np.asarray(X_np)
            y_arr = np.asarray(y_np, dtype=float)

            for model in self.models:
                if isinstance(model, str):
                    model_key = _resolve_model_alias(model)
                    grid = params_grid.get(model_key, [{}])
                    for params in ParameterGrid(grid):
                        base_model = _MODEL_CONSTRUCTORS[model_key]()
                        base_model.set_params(**params)
                        cvres = cross_validate(
                            base_model,
                            X_arr,
                            y_arr,
                            cv=ps,
                            scoring=scorers,
                            return_train_score=False,
                            n_jobs=-1,
                        )
                        per_r2 = list(cvres["test_r2"])
                        per_mse = [
                            -m for m in cvres["test_mse"]
                        ]  # negate because scorer is neg MSE
                        result_key = (
                            f"{key}::{base_model.__class__.__name__}::{params}"
                        )
                        base_model.fit(X_arr, y_arr)
                        cv_result = CV_Result(
                            feature_set=feature_name,
                            label=label_name,
                            scheme=scheme_name,
                            cross_val_scores=per_r2,
                            validation_r2_per_fold=per_r2,
                            validation_mse_per_fold=per_mse,
                            best_params=dict(params),
                            trained_model=base_model,
                        )
                        results[result_key] = cv_result
                        self._update_best_model(
                            result_key, cv_result, estimator=base_model
                        )
                    continue  # skip to next model after processing all param combos
                else:
                    # estimator instance provided by user
                    est_name = model.__class__.__name__.lower()
                    grid = params_grid.get(est_name, [{}])
                    for params in ParameterGrid(grid):
                        est = clone(model)
                        est.set_params(**params)
                        cvres = cross_validate(
                            est,
                            X_arr,
                            y_arr,
                            cv=ps,
                            scoring=scorers,
                            return_train_score=False,
                            n_jobs=-1,
                        )
                        per_r2 = list(cvres["test_r2"])
                        per_mse = [-v for v in cvres["test_mse"]]
                        result_key = (
                            f"{key}::{est.__class__.__name__}::{params}"
                        )
                        est.fit(X_arr, y_arr)
                        cv_result = CV_Result(
                            feature_set=feature_name,
                            label=label_name,
                            scheme=scheme_name,
                            cross_val_scores=per_r2,
                            validation_r2_per_fold=per_r2,
                            validation_mse_per_fold=per_mse,
                            best_params=dict(params),
                            trained_model=est,
                        )
                        results[result_key] = cv_result
                        self._update_best_model(
                            result_key, cv_result, estimator=est
                        )

        return results

    def run_grid(
        self,
        param_path: str = "hyperparameters.yaml",
        n_jobs: Optional[int] = None,
        params_per_job: int = 2,
    ) -> dict:
        """Performs grid search cross-validation using predefined splits.

        Inputs:
            label: Union[str, List[str]] = None (specific label(s) to use; defaults to self.label or all label attributes in dataset)
            scheme: Union[str, List[str]] = None (specific CV scheme(s) to use; defaults to all schemes in dataset)
            param_path: str = "hyperparameters.yaml" (path to YAML file with hyperparameter grids)
        Outputs:
            Dict[str, CV_Result]: A dictionary mapping each unique combination of feature set, label, scheme, model, and hyperparameters to its CV_Result.
        """
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
            self.dataset, fillna=0.0, label=label_to_use, scheme=scheme_to_use
        )
        param_grids = self.load_param_grids(param_path)
        scorers = {
            "r2": "r2",
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
        }

        for key, (X_np, y_np, joined) in mapping.items():
            try:
                feature_name, label_name, scheme_name = key.split("::")
            except Exception:
                feature_name, label_name, scheme_name = key, None, None

            try:
                fold_arr = np.array(joined.select("fold").to_numpy()).ravel()
            except Exception:
                try:
                    fold_arr = np.array([r["fold"] for r in joined.to_dicts()])
                except Exception:
                    logging.warning(
                        "Could not extract fold assignments for %s; skipping",
                        key,
                    )
                    continue

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

            for model in self.models:
                if isinstance(model, str):
                    model_key = _resolve_model_alias(model)
                    grid = param_grids.get(model_key, [{}])
                    estimator = _MODEL_CONSTRUCTORS[model_key]()
                else:
                    # estimator instance: prefer class-name key, then canonical short keys
                    est_name = model.__class__.__name__.lower()
                    grid = param_grids.get(est_name, None)
                    if grid is None:
                        canon = None
                        for k, ctor in _MODEL_CONSTRUCTORS.items():
                            try:
                                if isinstance(model, ctor().__class__):
                                    canon = k
                                    break
                            except Exception:
                                continue
                        grid = (
                            param_grids.get(canon, [{}])
                            if canon is not None
                            else [{}]
                        )
                    estimator = model
                n_jobs_use = self._select_n_jobs(grid, n_jobs, params_per_job)
                gs = GridSearchCV(
                    estimator=estimator,
                    param_grid=grid,
                    cv=ps,
                    scoring=scorers,
                    refit="r2",
                    n_jobs=n_jobs_use,
                )
                gs.fit(X_arr, y_arr)
                cvres = gs.cv_results_
                best_idx = gs.best_index_
                n_splits = ps.get_n_splits()
                per_r2 = [
                    cvres[f"split{i}_test_r2"][best_idx]
                    for i in range(n_splits)
                ]
                per_mse = [
                    -cvres[f"split{i}_test_mse"][best_idx]
                    for i in range(n_splits)
                ]  # negate because scorer is neg MSE

                result_key = f"{key}::{gs.best_estimator_.__class__.__name__}"
                best_params = getattr(gs, "best_params_", None)
                cv_result = CV_Result(
                    feature_set=feature_name,
                    label=label_name,
                    scheme=scheme_name,
                    cross_val_scores=per_r2,
                    validation_r2_per_fold=per_r2,
                    validation_mse_per_fold=per_mse,
                    best_params=best_params,
                    trained_model=gs.best_estimator_,
                )
                results[result_key] = cv_result
                self._update_best_model(
                    result_key, cv_result, estimator=gs.best_estimator_
                )
                logger.info(f"Best hyperparameters: {gs.best_params_}\n")
        return results

    def _prepare_inputs(
        self,
        dataset: Dataset,
        fillna: float = 0.0,
        label: Optional[Union[str, List[str]]] = None,
        scheme: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Any]]:
        """Prepares the input data for cross-validation by aligning feature
        sets, labels, and CV schemes.

        Returns a dictionary mapping each unique combination of feature set,
        label, and scheme to its (X, y, joined DataFrame).
        """

        import logging

        import polars as pl

        results: Dict[str, Tuple[np.ndarray, np.ndarray, Any]] = {}

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

        # 2. Materialize feature sets and labels into DataFrames
        for feat_name, feat_obj in list(feature_sets.items()):
            feat_df = feat_obj.to_df()
            acc_col = next(
                (c for c in ("sample", "acc") if c in feat_df.columns), None
            )
            if acc_col is None:
                raise RuntimeError(
                    "featureset missing accession column ('sample' or 'acc')"
                )

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

                # Determine CV schemes to use. Accept a single scheme name or a list/tuple/set of names.
                # If `split_manager` is a plain dict mapping scheme->table, use it directly.
                # Declare with an explicit Optional type to satisfy static type checkers.
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

                    # 3. Align and combine feature, label, and cv DataFrames
                    fdf = feat_df
                    ldf = label_df
                    feature_cols = [c for c in fdf.columns if c != "sample"]
                    if feature_cols:
                        fdf = fdf.with_columns(
                            [pl.col(c).fill_null(fillna) for c in feature_cols]
                        )

                    joined = cv_df.join(fdf, on="sample", how="left").join(
                        ldf, on="sample", how="left"
                    )

                    label_value_col = next(
                        (c for c in ldf.columns if c != "sample"), None
                    )
                    if label_value_col is None:
                        raise RuntimeError(
                            "label DataFrame has no value column after materialize"
                        )

                    # Delete null rows in label column
                    joined = joined.filter(~pl.col(label_value_col).is_null())

                    # Check number of folds after dropping null labels
                    n_folds_after = (
                        joined.select(pl.col("fold"))
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

                    # 4. Extract X and y as numpy arrays
                    drop_cols = ["sample", "fold", label_value_col]
                    feat_only = joined.drop(drop_cols)
                    try:
                        X_np = feat_only.to_numpy()
                    except Exception:
                        X_np = np.asarray(feat_only)
                    try:
                        y_np = np.asarray(
                            joined.select(pl.col(label_value_col))
                            .to_numpy()
                            .ravel(),
                            dtype=float,
                        )
                    except Exception:
                        y_np = np.asarray(
                            joined.select(pl.col(label_value_col))
                            .to_numpy()
                            .ravel(),
                            dtype=float,
                        )

                    results[f"{feat_name}::{label_name}::{scheme_name}"] = (
                        X_np,
                        y_np,
                        joined,
                    )
                    logger.info(
                        "Prepared data for feature set '%s', label '%s', scheme '%s' with %d samples and %d features.",
                        feat_name,
                        label_name,
                        scheme_name,
                        X_np.shape[0],
                        X_np.shape[1],
                    )

        return results
