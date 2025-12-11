# microbiome_ml/src/microbiome_ml/train/cv.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from microbiome_ml.wrangle.dataset import Dataset


# Define a Results class to store cross-validation results
class Results:
    """Store CV results and produce a single-row DataFrame with fold columns."""

    def __init__(
        self,
        feature_set: Optional[str] = None,
        label: Optional[str] = None,
        scheme: Optional[str] = None,
        cross_val_scores: Optional[List[float]] = None,
        validation_r2_per_fold: Optional[List[float]] = None,
        validation_mse_per_fold: Optional[List[float]] = None,
    ) -> None:
        self.feature_set = feature_set
        self.label = label
        self.scheme = scheme
        self.cross_val_scores = (
            list(cross_val_scores) if cross_val_scores is not None else []
        )
        self.validation_r2_per_fold = (
            list(validation_r2_per_fold)
            if validation_r2_per_fold is not None
            else []
        )
        self.validation_mse_per_fold = (
            list(validation_mse_per_fold)
            if validation_mse_per_fold is not None
            else []
        )
        self.avg_validation_r2: Optional[float] = None
        self.avg_validation_mse: Optional[float] = None
        self._compute_averages()

    def _compute_averages(self) -> None:
        """Compute and store average validation metrics.

        Computes mean R2 and mean MSE from the per-fold lists `validation_r2_per_fold`
        and `validation_mse_per_fold` and stores them in `avg_validation_r2` and
        `avg_validation_mse` respectively. If the lists are empty, the averages are
        set to `None`.
        """
        # Compute average R2 and MSE from per-fold lists -> work with any fold count
        if self.validation_r2_per_fold:
            self.avg_validation_r2 = float(
                sum(self.validation_r2_per_fold)
                / len(self.validation_r2_per_fold)
            )
        else:
            self.avg_validation_r2 = None

        if self.validation_mse_per_fold:
            self.avg_validation_mse = float(
                sum(self.validation_mse_per_fold)
                / len(self.validation_mse_per_fold)
            )
        else:
            self.avg_validation_mse = None

    def _serialize_value(self, v: Any) -> Any:
        """Normalize a Python value for JSON/Polars serialization.

        Handles common types used in Results objects: numpy scalars and arrays,
        pathlib `Path` objects, lists/tuples and native Python scalars. Returns a
        JSON-serializable Python value (e.g. `float`, `int`, `list`, `str` or
        `None`). This method is defensive and will fall back to `str()` for
        unknown objects.

        Args:
            v: Any Python object to normalize.

        Returns:
            A JSON/Polars-friendly representation of `v`.
        """

        if isinstance(v, Path):
            return str(v)

        if np is not None:
            if isinstance(v, (np.floating, np.float32, np.float64)):
                return float(v)
            if isinstance(v, (np.integer, np.int32, np.int64)):
                return int(v)
            if isinstance(v, np.ndarray):
                return [self._serialize_value(x) for x in v.tolist()]

        if isinstance(v, (list, tuple)):
            return [self._serialize_value(x) for x in v]

        if isinstance(v, (str, int, float, bool)) or v is None:
            return v

        try:
            return str(v)
        except Exception:
            return None

    def _to_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation of the Results.

        The dictionary contains identifying metadata (`feature_set`, `label`,
        `scheme`) and per-fold / aggregate metrics. Values are normalized using
        `_serialize_value` so the dict is safe to pass to `json.dumps` or to
        construct a Polars DataFrame.

        Returns:
            dict: a plain-Python mapping suitable for JSON serialization.
        """
        self._compute_averages()
        return {
            "feature_set": self._serialize_value(self.feature_set),
            "label": self._serialize_value(self.label),
            "scheme": self._serialize_value(self.scheme),
            "cross_val_scores": self._serialize_value(self.cross_val_scores),
            "validation_r2_per_fold": self._serialize_value(
                self.validation_r2_per_fold
            ),
            "avg_validation_r2": self._serialize_value(self.avg_validation_r2),
            "validation_mse_per_fold": self._serialize_value(
                self.validation_mse_per_fold
            ),
            "avg_validation_mse": self._serialize_value(
                self.avg_validation_mse
            ),
        }

    def to_dataframe(self) -> Any:
        """Return a single-row Polars DataFrame representing this Results object.

        Requires `polars` to be importable. Converts the internal dict produced
        by `_to_dict()` into a `polars.DataFrame` with one row.

        Raises:
            RuntimeError: if `polars` cannot be imported.
        """
        try:
            import polars as pl
        except Exception as e:
            raise RuntimeError("polars is required for to_dataframe()") from e
        return pl.DataFrame([self._to_dict()])

    def to_json(
        self, path: Optional[Union[str, Path]] = None, indent: int = 2
    ) -> str:
        """Return a JSON string for this Results and optionally write it to disk.

        Args:
            path: Optional path to write the JSON contents. If provided, the
                JSON string is also written to this location.
            indent: JSON indentation level used for pretty-printing.

        Returns:
            str: the JSON string representation of this Results.
        """
        j = json.dumps(self._to_dict(), indent=indent)
        if path:
            Path(path).write_text(j)
        return j

    def summary(self, save_json: Optional[Union[str, Path]] = None) -> dict:
        """Print a concise human-readable summary and optionally persist to JSON.

        The method prints the identifying fields and averaged metrics to
        stdout and returns the same information as a dictionary. If
        `save_json` is provided, the dictionary is written to that path as
        pretty JSON.

        Args:
            save_json: Optional path to write the single-result JSON object.

        Returns:
            dict: the same JSON-serializable dict produced by `_to_dict()`.
        """
        self._compute_averages()
        # Header
        print(f"Feature set: {self.feature_set}")
        print(f"Label: {self.label}")
        print(f"Scheme: {self.scheme}")
        # Metrics
        print(f"avg_validation_r2: {self.avg_validation_r2}")
        print(f"avg_validation_mse: {self.avg_validation_mse}")
        print(f"cross_val_scores: {self.cross_val_scores}")

        out = self._to_dict()
        if save_json:
            Path(save_json).write_text(json.dumps(out, indent=2))
        return out

    @staticmethod
    def results_to_dataframe(results_list: List["Results"]) -> Any:
        """Convert a list of `Results` objects into a Polars DataFrame.

        Each `Results` instance is converted to a dict via `_to_dict()` and the
        resulting list of rows is used to construct a `polars.DataFrame`.

        Args:
            results_list: list of `Results` objects.

        Returns:
            polars.DataFrame: one row per Results entry.

        Raises:
            RuntimeError: if `polars` is not importable.
        """
        try:
            import polars as pl
        except Exception as e:
            raise RuntimeError(
                "polars is required for results_to_dataframe()"
            ) from e
        rows = [r._to_dict() for r in results_list]
        return pl.DataFrame(rows)

    @staticmethod
    def results_to_json_file(
        results_list: List["Results"], path: Union[str, Path], indent: int = 2
    ) -> None:
        """Write a list of `Results` objects to disk as a JSON array.

        Args:
            results_list: list of `Results` instances.
            path: destination file path for the JSON array.
            indent: JSON indentation level for pretty output.
        """
        rows = [r._to_dict() for r in results_list]
        Path(path).write_text(json.dumps(rows, indent=indent))


# Define a CrossValidator class to handle cross-validation process
class CrossValidator:
    def __init__(
        self,
        dataset: Dataset,
        models: Union[object, List[object]],
        cv_folds: int = 5,
        # can have different scoring param, and can take list of scoring methods
        scoring: Union[str, List[str]] = "r2",
    ) -> None:
        """Create a CrossValidator bound to a dataset and list of models.

        Args:
            dataset: a `Dataset` instance or a dataset-like object exposing
                `feature_sets`, `labels`, and `splits` attributes. The
                CrossValidator uses this to materialize features, labels and
                precomputed CV schemes.
            models: either a single model (string shortcut or estimator) or a
                list of models. String shortcuts such as `"rf"` map to
                sensible defaults (RandomForestRegressor). Estimator
                instances (scikit-learn-like) are also accepted.
            cv_folds: default number of folds used for fallback KFold or
                GroupKFold strategies when constructing `cv` objects.
            scoring: currently unused placeholder; reserved for future
                customization of scoring metrics.
        """
        # Accept Dataset instances or dataset-like objects (duck-typing) for tests.
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            # require minimal attributes to be present so tests can use FakeDataset
            required = ("feature_sets", "labels", "splits")
            missing = [a for a in required if not hasattr(dataset, a)]
            if missing:
                raise TypeError(
                    "dataset must be an instance of Dataset or a dataset-like object with attributes: "
                    f"{', '.join(required)}. Missing: {', '.join(missing)}"
                )
            self.dataset = dataset
        # Normalize models to a list for consistent iteration and annotate as such
        self.models: List[object]
        if isinstance(models, list):
            self.models = models
        else:
            self.models = [models]
        self.cv_folds = cv_folds
        self.scoring = scoring

    def _is_lazy_polars(obj: Any) -> bool:
        """Detect whether `obj` is a Polars `LazyFrame`.

        This helper is used to decide whether to `collect()` lazy frames or
        treat the object as an already-materialized DataFrame.

        Args:
            obj: object to test.

        Returns:
            bool: True when `obj` is a `polars.LazyFrame`, False otherwise.
        """
        import polars as pl

        return (
            pl is not None
            and hasattr(pl, "LazyFrame")
            and isinstance(obj, pl.LazyFrame)
        )

    def run_all(
        self,
        # group_column: Optional[str] = None,
        # Use holdout cv folds defined in dataset splits if available
        # use_cv_folds: bool = True,
    ) -> dict:
        """Execute cross-validation for all prepared feature×label×scheme sets.

        This method consumes the mapping returned by `_prepare_inputs` and
        iterates each `feature::label::scheme` combination. For each model
        configured on this CrossValidator it trains and evaluates the model on
        each fold precomputed in the `joined['fold']` column and collects per-
        fold R2 and MSE metrics.

        The method returns a dictionary mapping keys of the form
        `feature::label::scheme::ModelName` to `Results` objects containing
        per-fold metrics and averages.

        Args:
            group_column: optional metadata column name to indicate groupings
                for GroupKFold strategy (not directly used here because the
                method relies on precomputed `fold` assignments in `joined`).
            use_cv_folds: when True (default) uses precomputed CV schemes from
                the dataset; otherwise behavior may fall back to KFold
                strategies (not implemented here).

        Returns:
            Dict[str, Results]: mapping result key -> Results object.
        """

        import logging

        results: Dict[str, Results] = {}

        # Prepare inputs for all feature×label×scheme combinations
        mapping = self._prepare_inputs(self.dataset, fillna_x=0.0)

        for key, (X_np, y_np, joined) in mapping.items():
            # key expected as "feature::label::scheme"
            try:
                feat_name, label_name, scheme_name = key.split("::")
            except Exception:
                feat_name = key
                label_name = None
                scheme_name = None

            # Extract fold assignments from the joined table in row order
            try:
                fold_arr = np.array(joined.select("fold").to_numpy()).ravel()
            except Exception:
                # Fallback: build from dicts
                try:
                    fold_arr = np.array([r["fold"] for r in joined.to_dicts()])
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Could not extract fold assignments for %s; skipping",
                        key,
                    )
                    continue

            # Unique folds present (sorted)
            unique_folds = np.unique(fold_arr)
            if len(unique_folds) < 2:
                logging.getLogger(__name__).warning(
                    "Not enough folds for %s (found %d); skipping",
                    key,
                    len(unique_folds),
                )
                continue

            # Ensure arrays are numpy and aligned
            X_arr = np.asarray(X_np)
            y_arr = np.asarray(y_np, dtype=float)

            for model in self.models:
                # Resolve string shortcuts to sklearn estimators
                if isinstance(model, str):
                    mname = model.lower()
                    # Flexible matching for common shortcuts
                    if any(
                        tok in mname
                        for tok in ("random", "rf", "random_forest")
                    ):
                        base_model = RandomForestRegressor(
                            n_estimators=100, random_state=42
                        )
                    elif any(
                        tok in mname
                        for tok in (
                            "linear",
                            "regress",
                            "lr",
                            "linear_regression",
                        )
                    ):
                        base_model = LinearRegression()
                    else:
                        raise ValueError(f"Unknown model string: {model}")
                else:
                    base_model = model

                model_name = base_model.__class__.__name__
                per_r2: List[float] = []
                per_mse: List[float] = []

                for f in sorted(unique_folds):
                    test_mask = fold_arr == f
                    train_mask = ~test_mask
                    if int(test_mask.sum()) == 0 or int(train_mask.sum()) == 0:
                        logging.getLogger(__name__).warning(
                            "Skipping fold %s for %s: empty train/test", f, key
                        )
                        continue

                    clf = clone(base_model)
                    clf.fit(X_arr[train_mask], y_arr[train_mask])
                    y_pred = clf.predict(X_arr[test_mask])
                    r = float(r2_score(y_arr[test_mask], y_pred))
                    m = float(mean_squared_error(y_arr[test_mask], y_pred))
                    per_r2.append(r)
                    per_mse.append(m)

                res = Results(
                    feature_set=feat_name,
                    label=label_name,
                    scheme=scheme_name,
                    cross_val_scores=per_r2,
                    validation_r2_per_fold=per_r2,
                    validation_mse_per_fold=per_mse,
                )

                results[f"{key}::{model_name}"] = res

        return results

    # compile results into a Results object
    def compile_results(
        self,
        r2_scores: Optional[List[float]] = None,
        mse_scores: Optional[List[float]] = None,
        scores: Optional[List[float]] = None,
    ) -> Results:
        """Helper to assemble a `Results` object from per-fold metrics.

        Accepts lists for R2 and MSE per-fold and returns a populated
        `Results` instance. This is a small convenience wrapper used by
        higher-level code paths to normalise results construction.

        Args:
            r2_scores: optional per-fold R2 values.
            mse_scores: optional per-fold MSE values.
            scores: optional list of primary scoring values (can mirror R2).

        Returns:
            Results: populated Results instance with averages computed.
        """

        # Backwards-compatible handling: if caller passed a single positional
        # argument previously, frameworks may still call compile_results(scores).
        # This method signature prefers explicit names; ensure lists are proper.
        r2_list = list(r2_scores) if r2_scores is not None else []
        mse_list = list(mse_scores) if mse_scores is not None else []
        scores_list = list(scores) if scores is not None else []

        # If only r2_list is present and scores_list empty, use r2 as scores
        if not scores_list and r2_list:
            scores_list = r2_list

        results = Results(
            cross_val_scores=scores_list if scores_list else None,
            validation_r2_per_fold=r2_list if r2_list else None,
            validation_mse_per_fold=mse_list if mse_list else None,
        )

        # Compute averaged metrics from validation lists
        results._compute_averages()
        return results

    # Note: _prepare_data removed per user request — run_all now passes
    # the raw X, aligned y_array, and groups directly to cross-validation.

    # Internal method to choose the cross-validation strategy
    def _get_cv_strategy(self, groups: Optional[Any]) -> object:
        """Return a scikit-learn cross-validation splitter based on groups.

        If `groups` is None this returns a shuffled `KFold(n_splits=self.cv_folds)`.
        If `groups` contains enough distinct non-null values (>= max(2, cv_folds))
        this returns a `GroupKFold(n_splits=self.cv_folds)`. Otherwise it falls
        back to a shuffled KFold and logs a warning.

        Args:
            groups: optional iterable of group identifiers aligned to samples.

        Returns:
            A scikit-learn CV splitter instance (KFold or GroupKFold).
        """
        # Decide whether to use GroupKFold or fall back to KFold.
        # Use GroupKFold only when groups is provided and there are
        # enough non-null unique groups to perform grouping. GroupKFold
        # also requires at least `n_splits` distinct groups; if that
        # is not the case we fallback to a shuffled KFold.
        if groups is None:
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # Compute unique, non-null groups robustly
        try:
            import numpy as np

            # Robustly count unique, non-null groups without pandas
            cleaned = []
            for g in groups:
                if g is None:
                    continue
                try:
                    if isinstance(g, float) and np.isnan(g):
                        continue
                except Exception:
                    pass
                cleaned.append(g)
            n_unique = len(set(map(str, cleaned)))
        except Exception:
            n_unique = 0

        # Need at least 2 groups to justify GroupKFold, and at least cv_folds groups
        if n_unique >= max(2, self.cv_folds):
            return GroupKFold(n_splits=self.cv_folds)

        # Not enough groups — fall back to KFold with a warning
        import logging

        logging.getLogger(__name__).warning(
            "Insufficient distinct groups for GroupKFold (found %d); falling back to KFold.",
            n_unique,
        )
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    def _prepare_inputs(
        self,
        dataset: Dataset,
        fillna_x: float = 0.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Any]]:
        """Materialize and align features, labels and CV folds into arrays.

        Iterates every feature set and label available in `dataset` and, for
        each CV scheme defined under `dataset.splits[label].cv_schemes`, it
        materializes the feature DataFrame and label DataFrame, joins them
        to the CV scheme (which must include a `fold` column) and returns a
        mapping of `feature::label::scheme` -> (X_numpy, y_numpy, joined_pl_df).

        The method fills missing feature values with `fillna_x`, drops rows
        where the label is null and skips schemes with fewer than 2 populated
        folds. The returned `joined` Polars DataFrame contains the `fold`
        column aligned to the rows of `X_numpy` and `y_numpy`.

        Args:
            dataset: dataset-like object exposing `feature_sets`, `labels`,
                and `splits`.
            fillna_x: value used to fill missing feature values before
                conversion to numpy arrays.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, polars.DataFrame]]: mapping
            keys to prepared arrays and the joined DataFrame.
        """

        # New behaviour: prepare per-cv-scheme X/y arrays
        # Returns: dict mapping scheme_name -> (X_numpy, y_numpy, joined_polars_df)
        import logging

        try:
            import polars as pl
        except Exception:
            raise RuntimeError("polars is required for _prepare_inputs")

        results: Dict[str, Tuple[np.ndarray, np.ndarray, Any]] = {}

        # Resolve feature sets and label from dataset in a few common patterns
        # feature_sets: mapping name->FeatureSet
        fsets = getattr(dataset, "feature_sets", None)
        if fsets is None:
            raise RuntimeError("dataset has no attribute 'feature_sets'")

        # We'll iterate all feature sets (caller can choose which to pass);
        # for now use the first feature set present to align sample order
        # but produce results per scheme for the provided label(s).
        # Determine label names available
        labels_attr = getattr(dataset, "labels", None)

        # Helper to materialize a FeatureSet to a Polars DataFrame
        def _materialize_featureset(fs_obj: Any) -> Any:
            """Convert a feature-set object to a materialized `polars.DataFrame`.

            Accepts several shapes: a custom FeatureSet with `.to_df()` or
            `.features` (which may be lazy), an already-materialized Polars
            DataFrame, or any object coercible to `pl.DataFrame`.

            Returns a `polars.DataFrame`.
            """
            # fs_obj may be FeatureSet with .features LazyFrame or a LazyFrame/DataFrame itself
            if hasattr(fs_obj, "to_df"):
                return fs_obj.to_df()
            if hasattr(fs_obj, "collect"):
                try:
                    return fs_obj.collect()
                except Exception:
                    pass
            if hasattr(fs_obj, "features"):
                cand = fs_obj.features
                if hasattr(cand, "collect"):
                    return cand.collect()
                try:
                    return cand
                except Exception:
                    pass
            # as last resort attempt to coerce to polars DataFrame
            return pl.DataFrame(fs_obj)

        # For label materialization: support dataset.get_label(name) or dataset.labels[name]
        def _materialize_label(label_name: str) -> Any:
            """Return a label DataFrame for the given label name.

            Supports dataset.get_label(name) or indexing `dataset.labels[name]`.
            If a scalar/sequence is returned it will be wrapped into a DataFrame
            with a single value column. The returned DataFrame may contain a
            `sample` column (preferred) or only a value column (in which case
            alignment with feature accessions will be attempted later).

            Args:
                label_name: name of the label to materialize.

            Returns:
                polars.DataFrame: label DataFrame with at least one value column.
            """
            if hasattr(dataset, "get_label"):
                lab = dataset.get_label(label_name)
                # if lab is polars DataFrame/Series
                if isinstance(lab, pl.DataFrame):
                    return lab
                # if lab is a Series-like or numpy -> wrap
                try:
                    return pl.DataFrame({"value": lab})
                except Exception:
                    return pl.DataFrame({"value": [lab]})
            if labels_attr is not None and label_name in labels_attr:
                lab = labels_attr[label_name]
                if isinstance(lab, pl.DataFrame):
                    return lab
                try:
                    return pl.DataFrame(lab)
                except Exception:
                    # try to wrap list/array into DataFrame
                    return pl.DataFrame({"value": lab})
            raise RuntimeError(f"label {label_name} not found on dataset")

        # Access split manager mapping
        splits = getattr(dataset, "splits", None)
        if splits is None:
            raise RuntimeError(
                "dataset has no 'splits' attribute; cannot find cv_schemes"
            )

        # Iterate feature sets (use explicit iteration so caller can choose a specific one later)
        for feat_name, feat_obj in list(fsets.items()):
            feat_df = _materialize_featureset(feat_obj)
            # ensure accession column exists ('sample' or 'acc')
            if "sample" in feat_df.columns:
                acc_col = "sample"
            elif "acc" in feat_df.columns:
                acc_col = "acc"
            else:
                raise RuntimeError(
                    "featureset missing accession column ('sample' or 'acc')"
                )

            # label keys available for this dataset (we'll iterate them)
            label_keys = []
            if labels_attr is not None:
                label_keys = list(labels_attr.keys())
            elif hasattr(dataset, "iter_labels"):
                label_keys = [name for name, _ in dataset.iter_labels()]

            # for each label present, get split manager and iterate schemes
            for label_name in label_keys:
                # materialize label
                try:
                    lab_df = _materialize_label(label_name)
                except RuntimeError:
                    continue

                # ensure label has 'sample' column or we must align by accession
                value_col: Optional[str] = None
                if "sample" not in lab_df.columns:
                    # if lab_df has one column only, name it 'value'
                    if len(lab_df.columns) == 1:
                        value_col = lab_df.columns[0]
                        lab_df = lab_df.with_columns(
                            [pl.col(value_col).alias("value")]
                        )
                        value_col = "value"
                    else:
                        # try to coerce by adding accession from features
                        value_col = lab_df.columns[0]
                else:
                    # find value column (first col not 'sample')
                    value_col = next(
                        (c for c in lab_df.columns if c != "sample"), None
                    )
                    if value_col is None:
                        raise RuntimeError(
                            f"label {label_name} DataFrame has no value column"
                        )

                # get split manager and schemes for this label
                split_manager = None
                # Common shapes: a dict mapping label->split_manager, or an
                # object exposing `get_cv_schemes` or a `get(label)` method, or
                # attributes per-label. Handle dict explicitly so mypy knows the
                # dict case does not reach attribute checks below.
                if isinstance(splits, dict) and label_name in splits:
                    split_manager = splits[label_name]
                elif not isinstance(splits, dict) and hasattr(
                    splits, "get_cv_schemes"
                ):
                    # some APIs expose get_cv_schemes
                    try:
                        cv_map = splits.get_cv_schemes(label_name)
                        split_manager = type(
                            "SM", (), {"cv_schemes": cv_map}
                        )()
                    except Exception:
                        split_manager = None
                elif not isinstance(splits, dict) and hasattr(splits, "get"):
                    # object implements get(label)
                    try:
                        split_manager = splits.get(label_name)
                    except Exception:
                        split_manager = None
                else:
                    split_manager = getattr(splits, label_name, None)

                if split_manager is None:
                    logging.getLogger(__name__).warning(
                        "No split manager found for label '%s'", label_name
                    )
                    continue

                cv_schemes = getattr(split_manager, "cv_schemes", {})
                for scheme_name, scheme_table in cv_schemes.items():
                    # Normalize scheme_table to Polars DataFrame with 'sample' and 'fold'
                    if isinstance(scheme_table, dict):
                        # Filter out entries where fold is None (these indicate unassigned)
                        items = [
                            (str(k), v)
                            for k, v in scheme_table.items()
                            if v is not None
                        ]
                        if not items:
                            logging.getLogger(__name__).warning(
                                "CV scheme '%s' for label '%s' contains no assigned folds; skipping",
                                scheme_name,
                                label_name,
                            )
                            continue
                        samples = [k for k, _ in items]
                        try:
                            folds = [int(v) for _, v in items]
                        except Exception:
                            # If fold values are not coercible to int, skip scheme
                            logging.getLogger(__name__).warning(
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
                        # try to coerce list-of-dicts or pandas
                        try:
                            cv_df = pl.DataFrame(scheme_table)
                        except Exception:
                            logging.getLogger(__name__).warning(
                                "Unsupported cv scheme format for '%s' in label '%s'",
                                scheme_name,
                                label_name,
                            )
                            continue

                    # compute number of unique non-null folds
                    try:
                        n_folds = (
                            cv_df.select(pl.col("fold"))
                            .drop_nulls()
                            .unique()
                            .height
                        )
                    except Exception:
                        # fall back to python: robustly extract fold values whether rows are dicts or sequence-like
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
                        logging.getLogger(__name__).warning(
                            "CV scheme '%s' for label '%s' has %d populated fold(s); skipping (need >=2)",
                            scheme_name,
                            label_name,
                            n_folds,
                        )
                        continue

                    # Build joined table: start from cv_df (sample, fold), join features and label on 'sample'
                    # Ensure feature df has sample column and label df has sample column
                    if acc_col not in feat_df.columns:
                        raise RuntimeError(
                            "featureset missing accession column after materialize"
                        )

                    # rename feature accession column to 'sample' if necessary
                    fdf = feat_df
                    if acc_col != "sample":
                        fdf = fdf.rename({acc_col: "sample"})

                    # ensure label df has 'sample' column
                    ldf = lab_df
                    if "sample" not in ldf.columns:
                        # attempt to attach sample ids from features (align by order)
                        try:
                            ldf = ldf.with_columns(
                                pl.Series(
                                    "sample", fdf.select("sample").to_series()
                                )
                            )
                        except Exception:
                            raise RuntimeError(
                                "Cannot align label DataFrame to sample ids"
                            )

                    # Do lazy joins: fill nulls in feature columns first (exclude sample)
                    # Use pl.col().exclude('sample') to fill
                    feature_cols = [c for c in fdf.columns if c != "sample"]
                    if feature_cols:
                        fdf = fdf.with_columns(
                            [
                                pl.col(c).fill_null(fillna_x)
                                for c in feature_cols
                            ]
                        )

                    # Now join cv_df <- fdf <- ldf
                    joined = cv_df.join(fdf, on="sample", how="left").join(
                        ldf, on="sample", how="left"
                    )

                    # Drop rows where label value is null
                    # determine label value column (first non-sample)
                    label_value_col = next(
                        (c for c in ldf.columns if c != "sample"), None
                    )
                    if label_value_col is None:
                        raise RuntimeError(
                            "label DataFrame has no value column after materialize"
                        )

                    joined = joined.filter(~pl.col(label_value_col).is_null())

                    # Recompute folds presence after filtering
                    n_folds_after = (
                        joined.select(pl.col("fold"))
                        .drop_nulls()
                        .unique()
                        .height
                    )
                    if n_folds_after < 2:
                        logging.getLogger(__name__).warning(
                            "After dropping null labels CV scheme '%s' for label '%s' has %d fold(s); skipping",
                            scheme_name,
                            label_name,
                            n_folds_after,
                        )
                        continue

                    # Build X (drop sample, fold, and label columns) and y
                    drop_cols = ["sample", "fold", label_value_col]
                    feat_only = joined.drop(drop_cols)
                    # Convert to numpy arrays
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

        return results

    # Internal method to cross-validate a single model
    def _cross_validate_model(
        self, model: Any, X: Any, y: Any, groups: Optional[Any]
    ) -> object:
        """Run scikit-learn `cross_val_score` for a single model.

        Chooses an appropriate CV splitter from `_get_cv_strategy` and runs
        `cross_val_score` with the default `r2` scoring. If `groups` is
        accepted by the splitter it is passed through, otherwise it is ignored.

        Args:
            model: a scikit-learn estimator (or compatible) to evaluate.
            X: feature matrix (array-like) aligned to `y`.
            y: label array-like.
            groups: optional grouping array for grouped CV splitters.

        Returns:
            ndarray: array of per-fold scores returned by `cross_val_score`.
        """
        # Choose an appropriate cv object
        cv_strategy = self._get_cv_strategy(groups)

        # Use regression-style scoring by default (r2). If user needs classification
        # scoring they can adjust later or pass different models.
        scoring = "r2"

        # Only pass `groups` to cross_val_score when the cv object accepts groups
        try:
            accepts_groups = (
                hasattr(cv_strategy, "split")
                and "groups" in cv_strategy.split.__code__.co_varnames
            )
        except Exception:
            accepts_groups = False

        if accepts_groups and groups is not None:
            scores = cross_val_score(
                model, X, y, cv=cv_strategy, groups=groups, scoring=scoring
            )
        else:
            scores = cross_val_score(
                model, X, y, cv=cv_strategy, scoring=scoring
            )

        return scores
