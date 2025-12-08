# microbiome_ml/src/microbiome_ml/train/cv.py
"""Cross-validation utilities for microbiome ML training."""

from microbiome_ml.wrangle.dataset import Dataset
from typing import Union, List, Optional
from pathlib import Path
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

# Define a Results class to store cross-validation results
class Results:
    """Class to store cross-validation results."""
    def __init__(
        self,
        validation_r2_scores: List[float] = None,
        validation_mse_scores: List[float] = None,
        scores: List[float] = None,
    ):
        # Container for cross-validation outputs
        # needed: scores per fold, r2 per fold, mse per fold and then average them per fold
        # Preserve provided lists (make shallow copies) or create empty lists
        self.scores = list(scores) if scores is not None else []
        self.validation_r2_scores = list(validation_r2_scores) if validation_r2_scores is not None else []
        self.validation_mse_scores = list(validation_mse_scores) if validation_mse_scores is not None else []
        self.avg_val_r2: Optional[float] = None
        self.avg_val_mse: Optional[float] = None
    
    def _average_r2_mse(self):
        if len(self.validation_r2_scores) > 0:
            self.avg_val_r2 = float(sum(self.validation_r2_scores) / len(self.validation_r2_scores))
        else:
            self.avg_val_r2 = None

        if len(self.validation_mse_scores) > 0:
            self.avg_val_mse = float(sum(self.validation_mse_scores) / len(self.validation_mse_scores))
        else:
            self.avg_val_mse = None
    
    def summary(
        self,
        save_csv: Optional[Union[str, Path]] = None,
        ):
        """Return a summary dict and optionally save to CSV.

        Args:
            save_csv: path to write a CSV file with metrics (optional).
        """
        self._average_r2_mse()
        out = {
            "avg_val_r2": self.avg_val_r2,
            "avg_val_mse": self.avg_val_mse,
            "cross_val_scores": self.scores,
        }

        if save_csv:
            path = Path(save_csv)
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["avg_val_r2", self.avg_val_r2])
                writer.writerow(["avg_val_mse", self.avg_val_mse])
                # write per-fold scores as extra rows
                for i, s in enumerate(self.scores, start=1):
                    writer.writerow([f"fold_{i}_score", s])
        return out
    
# Define a CrossValidator class to handle cross-validation process
class CrossValidator:
    def __init__(
        self, 
        dataset: Dataset, 
        models: Union[object, List[object]],
        cv_folds: int = 5,
        ):

        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be an instance of Dataset")
        self.models = models
        self.cv_folds = cv_folds
        if not isinstance(self.models, list):
            self.models = [self.models]

    def run_all(self, group_column: str = None, dropna_y: bool = False, dropna_groups: bool = False):
        """Run cross-validation on the dataset with the specified models.
        
        Args:
            group_column: Name of metadata column to use for grouping (e.g., 'study', 'project')
                         If None, no grouping is used.
        """

        results = {}

        for feature_name, feature_set in self.dataset.iter_feature_sets():
            X = feature_set.features
            sample_ids = feature_set.accessions  # These are the sample IDs from features
            
            for y_name, y in self.dataset.iter_labels():
                # Convert polars DataFrame to numpy array for y, aligned with X sample order
                import polars as pl
                import numpy as np
                
                if isinstance(y, pl.DataFrame):
                    print(f"Label '{y_name}': {y.height} rows")
                    
                    # Find the value column (not 'sample')
                    value_cols = [col for col in y.columns if col != 'sample']
                    if len(value_cols) == 0:
                        raise ValueError(f"Label DataFrame for '{y_name}' has no value column")
                    
                    # Align y with X sample order (feature_set.accessions)
                    # feature_set uses 'accessions' which are sample IDs
                    # labels DataFrame uses 'sample' column
                    # Create a mapping DataFrame
                    sample_df = pl.DataFrame({'sample': sample_ids})
                    y_aligned = sample_df.join(y, on='sample', how='left')
                    
                    print(f"After alignment: {y_aligned.height} rows, {(~y_aligned[value_cols[0]].is_null()).sum()} non-null values")
                    
                    # Extract the value column
                    y_array = y_aligned.select(pl.col(value_cols[0])).to_numpy().flatten()
                    
                    # Convert to float, handle potential nulls
                    y_array = y_array.astype(float)
                    print(f"y_array: shape={y_array.shape}, non-nan count={np.sum(~np.isnan(y_array))}")
                else:
                    y_array = np.asarray(y, dtype=float)
                
                # Extract grouping information if specified
                groups = None
                if group_column is not None and self.dataset.metadata is not None:
                    # Get groups from metadata, matching sample order
                    metadata_df = self.dataset.metadata.metadata
                    if not self.dataset.metadata._is_lazy:
                        groups_data = metadata_df.filter(
                            pl.col('sample').is_in(sample_ids)
                        ).select(['sample', group_column])
                    else:
                        groups_data = metadata_df.collect().filter(
                            pl.col('sample').is_in(sample_ids)
                        ).select(['sample', group_column])
                    
                    # Align groups with sample order
                    sample_df = pl.DataFrame({'sample': sample_ids})
                    groups_aligned = sample_df.join(groups_data, on='sample', how='left')
                    groups = groups_aligned.select(pl.col(group_column)).to_numpy().flatten()
                
                # Prepare inputs once per (feature_set, label) pair to avoid
                # re-materializing or re-coercing X/y/groups for each model.
                X_prepared, y_prepared, groups_prepared = self._prepare_inputs(
                    X, y_array, groups, sample_ids=sample_ids, dropna_y=dropna_y, dropna_groups=dropna_groups
                )

                for model in self.models:
                    model_name = model.__class__.__name__
                    key = f"{feature_name}_{y_name}_{model_name}"
                    if group_column:
                        key += f"_grouped_{group_column}"
                    results[key] = self._cross_validate_model(model, X_prepared, y_prepared, groups_prepared)
        return results
    
    # compile results into a Results object
    def compile_results(
        self, 
        r2_scores: Optional[List[float]] = None, 
        mse_scores: Optional[List[float]] = None, 
        scores: Optional[List[float]] = None):
        
        """Compile cross-validation scores into a Results object.

        Accepts separate lists for R2 and MSE per-fold, plus an optional
        `scores` list (e.g., the primary scoring metric per fold). If only
        one list is provided positionally (legacy usage), it will be treated
        as `r2_scores` and also used for `scores`.
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
            validation_r2_scores=r2_list if r2_list else None,
            validation_mse_scores=mse_list if mse_list else None,
            scores=scores_list if scores_list else None,
        )

        # Compute averaged metrics from validation lists
        results._average_r2_mse()
        return results

    # Note: _prepare_data removed per user request — run_all now passes
    # the raw X, aligned y_array, and groups directly to cross-validation.

    # Internal method to choose the cross-validation strategy
    def _get_cv_strategy(self, groups):
        """Determine the cross-validation strategy based on groups."""
        # Decide whether to use GroupKFold or fall back to KFold.
        # Use GroupKFold only when groups is provided and there are
        # enough non-null unique groups to perform grouping. GroupKFold
        # also requires at least `n_splits` distinct groups; if that
        # is not the case we fallback to a shuffled KFold.
        if groups is None:
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # Compute unique, non-null groups robustly
        try:
            import numpy as _np
            import pandas as _pd
            grp_series = _pd.Series(groups)
            unique_nonnull = grp_series.dropna().unique()
            n_unique = len(unique_nonnull)
        except Exception:
            # Fallback to simple set-based counting if pandas unavailable
            try:
                cleaned = [g for g in groups if g is not None and not (isinstance(g, float) and _np.isnan(g))]
                n_unique = len(set(map(str, cleaned)))
            except Exception:
                n_unique = 0

        # Need at least 2 groups to justify GroupKFold, and at least cv_folds groups
        if n_unique >= max(2, self.cv_folds):
            return GroupKFold(n_splits=self.cv_folds)

        # Not enough groups — fall back to KFold with a warning
        import logging
        logging.getLogger(__name__).warning(
            "Insufficient distinct groups for GroupKFold (found %d); falling back to KFold.", n_unique
        )
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    def _prepare_inputs(self, X, y, groups, sample_ids: Optional[list] = None, dropna_y: bool = False, dropna_groups: bool = False):
        """Materialize and validate inputs for sklearn.

        - Convert Polars LazyFrame/DataFrame to concrete numpy arrays.
        - Ensure `y` is a 1-d float numpy array.
        - Ensure `X` is array-like (2-d) numeric numpy array.
        - Convert `groups` to a 1-d numpy array if provided.
        - Optionally drop samples where `y` or `groups` are null.

        Returns (X_array, y_array, groups_array_or_None).
        Raises informative TypeError/ValueError if conversion is not possible.
        """
        import numpy as np
        try:
            import polars as pl
            _has_polars = True
        except Exception:
            pl = None
            _has_polars = False

        # Handle X
        X_array = None
        # If Polars LazyFrame/DataFrame, collect/convert to pandas when possible
        try:
            import pandas as pd
            _has_pandas = True
        except Exception:
            pd = None
            _has_pandas = False

        if _has_polars and (isinstance(X, pl.LazyFrame) or isinstance(X, pl.DataFrame)):
            if isinstance(X, pl.LazyFrame):
                X = X.collect()
            # prefer converting Polars DataFrame to pandas for column inspection
            try:
                X = X.to_pandas()
            except Exception:
                # fall back to numpy
                try:
                    X_array = X.to_numpy()
                except Exception:
                    X_array = np.asarray(X)

        # If we have a pandas DataFrame, drop ID-like and non-numeric columns conservatively
        if _has_pandas and isinstance(X, pd.DataFrame):
            dropped_cols = []
            # Drop columns with very common id names
            id_names = ("sample", "accession", "id", "sample_id", "run_accession")
            for bad in id_names:
                if bad in X.columns:
                    dropped_cols.append(bad)

            # If sample_ids provided, drop columns whose values are (all) sample ids
            if sample_ids is not None:
                try:
                    acc_set = set(map(str, sample_ids))
                    for c in list(X.columns):
                        if c in dropped_cols:
                            continue
                        vals = X[c].dropna().astype(str).unique()
                        if len(vals) > 0 and all(v in acc_set for v in vals):
                            dropped_cols.append(c)
                except Exception:
                    pass

            if dropped_cols:
                import logging
                logging.getLogger(__name__).warning("Dropping ID-like columns from X: %s", dropped_cols)
                X = X.drop(columns=dropped_cols, errors='ignore')

            # Keep only numeric columns
            numeric_X = X.select_dtypes(include=[np.number])
            non_numeric = set(X.columns) - set(numeric_X.columns)
            if non_numeric:
                import logging
                logging.getLogger(__name__).warning("Dropping non-numeric columns from X: %s", sorted(list(non_numeric))[:20])
            X = numeric_X

            # Convert to numpy
            X_array = X.to_numpy()
        else:
            # If object exposes to_numpy (pandas-like) prefer that
            if hasattr(X, "to_numpy") and X_array is None:
                try:
                    X_array = X.to_numpy()
                except Exception:
                    X_array = np.asarray(X)
            elif X_array is None:
                X_array = np.asarray(X)

        # Ensure X is 2-d
        X_array = np.asarray(X_array)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        # Handle y: ensure 1-d float array
        # If y is a Polars LazyFrame/DataFrame, collect and extract first column
        if _has_polars and (isinstance(y, pl.LazyFrame) or isinstance(y, pl.DataFrame)):
            if isinstance(y, pl.LazyFrame):
                y = y.collect()
            # If DataFrame, select first non-'sample' column if present else first column
            cols = [c for c in y.columns if c != 'sample'] if hasattr(y, 'columns') else []
            if cols:
                y_vals = y.select(pl.col(cols[0])).to_numpy().flatten()
            else:
                # fallback: convert to numpy and ravel
                y_vals = y.to_numpy().ravel()
            y_array = np.asarray(y_vals, dtype=float)
        else:
            y_array = np.asarray(y, dtype=float)

        # Flatten y to 1-d
        y_array = y_array.ravel()

        # Handle groups if present
        groups_array = None
        if groups is not None:
            if _has_polars and (isinstance(groups, pl.LazyFrame) or isinstance(groups, pl.DataFrame)):
                if isinstance(groups, pl.LazyFrame):
                    groups = groups.collect()
                # if groups is a dataframe pick first column
                if hasattr(groups, 'columns') and len(groups.columns) >= 1:
                    groups_array = groups.select(pl.col(groups.columns[0])).to_numpy().ravel()
                else:
                    groups_array = np.asarray(groups).ravel()
            else:
                groups_array = np.asarray(groups).ravel()

        # Optionally drop samples with NaN y and/or NaN groups
        import pandas as pd

        mask = np.ones(y_array.shape[0], dtype=bool)
        if dropna_y:
            mask &= ~pd.isnull(y_array)
        if dropna_groups and groups_array is not None:
            mask &= ~pd.isnull(groups_array)

        if not mask.any():
            raise ValueError("No samples remain after applying dropna filters (dropna_y=%s, dropna_groups=%s)" % (dropna_y, dropna_groups))

        # Apply mask if any rows removed
        if mask.sum() != y_array.shape[0]:
            X_array = X_array[mask]
            y_array = y_array[mask]
            if groups_array is not None:
                groups_array = np.asarray(groups_array)[mask]

        # Final sanity checks
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f"Mismatched number of samples: X has {X_array.shape[0]} rows but y has {y_array.shape[0]} values after processing")

        return X_array, y_array.astype(float), (groups_array if groups is not None else None)

    # Internal method to cross-validate a single model
    def _cross_validate_model(self, model, X, y, groups):
        """Perform cross-validation for a single model."""
        # Choose an appropriate cv object
        cv_strategy = self._get_cv_strategy(groups)

        # Use regression-style scoring by default (r2). If user needs classification
        # scoring they can adjust later or pass different models.
        scoring = 'r2'

        # Only pass `groups` to cross_val_score when the cv object accepts groups
        try:
            accepts_groups = hasattr(cv_strategy, 'split') and 'groups' in cv_strategy.split.__code__.co_varnames
        except Exception:
            accepts_groups = False

        if accepts_groups and groups is not None:
            scores = cross_val_score(model, X, y, cv=cv_strategy, groups=groups, scoring=scoring)
        else:
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)

        return scores