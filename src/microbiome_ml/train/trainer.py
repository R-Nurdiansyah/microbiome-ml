from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from microbiome_ml.train.results import CV_Result, HoldoutEvaluation
from microbiome_ml.wrangle.dataset import Dataset


class ModelTrainer:
    """Retrain the best CV combo on holdout splits and persist the final model.

    The trainer reuses the metadata stored on `CV_Result` (feature set name, label,
    estimator, best hyperparameters, CV scheme) so downstream callers only need a
    finished CV run before instantiating this class.

    After that, the trainer materializes the feature set, joins it to the holdout train/test splits,
    and retrains a fresh estimator instance on the holdout training samples. Finally, it evaluates
    the model on the holdout test samples and returns the evaluation metrics along with the trained estimator.

    input:
        - dataset: Dataset object containing the holdout splits and feature sets
        - best_result: CV_Result object representing the best CV result to retrain
        - output_model_path: directory where the retrained model should be written or a file path
        - model_name: optional file name (defaults to "holdout model.pkl")
        - fillna: value to impute for any missing features in the holdout splits
    """

    def __init__(
        self,
        dataset: Dataset,
        best_result: CV_Result,
        output_model_path: Union[str, Path],
        model_name: Optional[str] = None,
        fillna: float = 0.0,
    ) -> None:
        """Initialize the trainer with the best CV result and dataset."""
        if not isinstance(best_result, CV_Result):
            raise TypeError(
                "best_result must be a CV_Result instance (pass cross_validator.best_result)"
            )
        self.dataset = dataset
        self.best_result = best_result
        self.fillna = fillna

        provided_path = Path(output_model_path)
        if provided_path.exists() and provided_path.is_file():
            self.output_model_dir = provided_path.parent
            self.output_model_path = provided_path
            self._save_as_package = False
        elif provided_path.suffix and not provided_path.exists():
            # treat as explicit file path even if parent directory missing
            self.output_model_dir = (
                provided_path.parent
                if provided_path.parent != Path("")
                else Path(".")
            )
            self.output_model_path = provided_path
            self._save_as_package = False
        else:
            self.output_model_dir = provided_path
            default_name = "holdout model"
            chosen_name = (model_name or default_name).strip() or default_name
            file_name = Path(chosen_name).name
            if not file_name.lower().endswith(".pkl"):
                file_name = f"{file_name}.pkl"
            self.output_model_path = self.output_model_dir / file_name
            self._save_as_package = True

    def train_and_evaluate(self) -> HoldoutEvaluation:
        """Train/re-evaluate the CV winner on the holdout train/test split.

        Steps:
            1. Materialize the feature set referenced by `best_result.feature_set`.
            2. Join the holdout train/test tables with that feature set, filling missing
              values with `fillna`.
            3. Clone and configure the estimator stored on `best_result`, retrain it on
              the holdout training samples, and evaluate on the holdout test samples.
            4. Persist the retrained estimator. If `output_model_path` is file-like,
                save a single pickle model. If directory-like, export a full results
                package via `CV_Result.export_result`.
            5. Return regression metrics plus the estimator instance.
        """
        feature_set_name = self._require_feature_set()
        label_name = self._require_label()

        train_df = self.dataset.get_train_samples(
            label=label_name, metadata=False
        )
        test_df = self.dataset.get_test_samples(
            label=label_name, metadata=False
        )

        feature_df = self._materialize_feature_set(feature_set_name)

        X_train, y_train, feature_cols = self._assemble_split(
            feature_df, train_df, label_name
        )
        X_test, y_test, _ = self._assemble_split(
            feature_df, test_df, label_name
        )

        estimator = self._clone_estimator()
        estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_test)
        metrics = self._calculate_metrics(
            y_test.tolist(), predictions.tolist()
        )
        result_metrics: Dict[str, Any] = dict(metrics)
        result_metrics.update(
            {
                "feature_set": feature_set_name,
                "label": label_name,
                "scheme": self.best_result.scheme,
                "n_test": len(y_test),
            }
        )

        self.output_model_dir.mkdir(parents=True, exist_ok=True)
        if self._save_as_package:
            holdout_result = CV_Result(
                feature_set=feature_set_name,
                label=label_name,
                scheme=self.best_result.scheme,
                cross_val_scores=[float(result_metrics["r2"])]
                if result_metrics.get("r2") is not None
                else [],
                validation_r2_per_fold=[float(result_metrics["r2"])]
                if result_metrics.get("r2") is not None
                else [],
                validation_mse_per_fold=[float(result_metrics["mse"])]
                if result_metrics.get("mse") is not None
                else [],
                best_params=self.best_result.best_params,
                trained_model=estimator,
                feature_names=feature_cols,
            )
            CV_Result.export_result(
                {"holdout_final_model": holdout_result},
                self.output_model_dir,
            )
        else:
            CV_Result.save_model(estimator, self.output_model_path)

        return HoldoutEvaluation(
            metrics=result_metrics,
            estimator=estimator,
            predictions=np.asarray(predictions),
            targets=np.asarray(y_test),
            feature_names=feature_cols,
        )

    def _clone_estimator(self) -> Any:
        """Make a fresh estimator using the best CV-trained estimator +
        params."""
        model = self.best_result.model
        if model is None:
            raise ValueError(
                "Best CV result does not expose a trained estimator"
            )
        estimator = clone(model)
        if self.best_result.best_params:
            estimator.set_params(**self.best_result.best_params)
        return estimator

    def _materialize_feature_set(self, name: str) -> pl.DataFrame:
        """Load the feature set by name and guarantee a `sample` column for
        joins."""
        feature_sets = self.dataset.feature_sets
        if name not in feature_sets:
            raise ValueError(
                f"Feature set '{name}' is not registered on the dataset"
            )

        feature_df = feature_sets[name].collect()
        if "sample" not in feature_df.columns:
            if "acc" in feature_df.columns:
                feature_df = feature_df.rename({"acc": "sample"})
            else:
                raise ValueError(
                    f"Feature set '{name}' is missing an accession column"
                )
        return feature_df

    def _assemble_split(
        self, feature_df: pl.DataFrame, split_df: pl.DataFrame, label: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Produce the feature matrix and target array for a specific split."""
        feature_cols = [col for col in feature_df.columns if col != "sample"]
        if not feature_cols:
            raise ValueError(
                "Feature set must provide at least one feature column"
            )

        joined = split_df.join(feature_df, on="sample", how="left")
        joined = joined.with_columns(
            [pl.col(col).fill_null(self.fillna) for col in feature_cols]
        )
        joined = joined.filter(~pl.col(label).is_null())
        if joined.is_empty():
            raise ValueError(
                "Holdout split contains no samples after combining with features"
            )

        X = np.asarray(joined.select(feature_cols).to_numpy())
        y = np.asarray(
            joined.select(pl.col(label)).to_numpy().ravel(), dtype=float
        )
        return X, y, feature_cols

    def _require_feature_set(self) -> str:
        feature_set = self.best_result.feature_set
        if not feature_set:
            raise ValueError("Best CV result did not record a feature set")
        return feature_set

    def _require_label(self) -> str:
        label = self.best_result.label
        if not label:
            raise ValueError("Best CV result did not record a label")
        return label

    def _calculate_metrics(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> Dict[str, Optional[float]]:
        """Compute MAE/MSE/R²/Q² and Pearson correlation for the holdout test
        set."""
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.size == 0 or y_pred_arr.size == 0:
            return {
                "mae": None,
                "mse": None,
                "r2": None,
                "q2": None,
                "pcc": None,
                "pval": None,
            }

        metrics: Dict[str, Optional[float]] = {}
        metrics["mae"] = mean_absolute_error(y_true_arr, y_pred_arr)
        metrics["mse"] = mean_squared_error(y_true_arr, y_pred_arr)
        metrics["r2"] = r2_score(y_true_arr, y_pred_arr)

        denom = ((y_true_arr - np.mean(y_true_arr)) ** 2).sum()
        metrics["q2"] = (
            1 - ((y_true_arr - y_pred_arr) ** 2).sum() / denom
            if denom > 0
            else None
        )

        try:
            metrics["pcc"], metrics["pval"] = pearsonr(y_true_arr, y_pred_arr)
        except Exception:
            metrics["pcc"] = None
            metrics["pval"] = None

        return metrics
