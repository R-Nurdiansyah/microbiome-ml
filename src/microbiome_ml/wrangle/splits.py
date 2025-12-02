"""Split management for train/test and cross-validation."""

import logging
from typing import Dict, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class SplitManager:
    """Manages train/test splits and cross-validation folds for a single label.

    Coordinates holdout splits and multiple CV schemes (random, grouped, etc.)
    for one target variable.

    Attributes:
        label: Name of the target variable
        holdout: DataFrame with {sample, split} columns ("train"/"test")
        cv_schemes: Dict mapping scheme names to CV DataFrames {sample, fold}
    """

    def __init__(self, label: str):
        """Initialize SplitManager for a label.

        Args:
            label: Name of the target variable
        """
        self.label = label
        self.holdout: Optional[pl.DataFrame] = None
        self.cv_schemes: Dict[str, pl.DataFrame] = {}

    def create_holdout(
        self,
        data: pl.DataFrame,
        grouping: Optional[str] = None,
        test_size: float = 0.2,
        n_bins: int = 5,
        random_state: int = 42,
    ) -> None:
        """Create holdout train/test split.

        Args:
            data: DataFrame with sample, label, and optional grouping columns
            grouping: Column name for grouping samples (prevents group leakage)
            test_size: Fraction of samples for test set
            n_bins: Number of bins for continuous targets
            random_state: Random seed for reproducibility
        """
        self.holdout = self._stratified_split(
            data=data,
            target_col=self.label,
            grouping=grouping,
            test_size=test_size,
            n_bins=n_bins,
            random_state=random_state,
            split_col_name="split",
        )

    def create_cv_folds(
        self,
        data: pl.DataFrame,
        grouping: Optional[str] = None,
        n_folds: int = 5,
        n_bins: int = 5,
        random_state: int = 42,
        scheme_name: Optional[str] = None,
    ) -> None:
        """Create k-fold cross-validation splits.

        Args:
            data: DataFrame with sample, label, and optional grouping columns
            grouping: Column name for grouping samples (prevents group leakage)
            n_folds: Number of folds
            n_bins: Number of bins for continuous targets
            random_state: Random seed for reproducibility
            scheme_name: Name for this CV scheme (defaults to grouping value or "random")
        """
        # Default scheme_name based on grouping
        if scheme_name is None:
            scheme_name = grouping if grouping is not None else "random"

        # Shuffle data
        df = data.sample(fraction=1.0, shuffle=True, seed=random_state)

        # Determine if continuous or categorical
        is_continuous = df.schema[self.label] in [pl.Float32, pl.Float64]

        # Create bins/classes
        if is_continuous:
            min_val, max_val = df.select(
                pl.min(self.label).alias("min"),
                pl.max(self.label).alias("max"),
            ).row(0)
            breaks = np.linspace(min_val, max_val, n_bins + 1)
            df = df.with_columns(
                pl.col(self.label)
                .cut(breaks[1:-1], left_closed=True)
                .alias("bin")
            )
        else:
            df = df.with_columns(pl.col(self.label).cast(pl.Utf8).alias("bin"))

        # Get unique groups
        if grouping is not None:
            group_col = grouping
        else:
            # Use sample as group if no grouping specified
            df = df.with_columns(pl.col("sample").alias("_group"))
            group_col = "_group"

        group_list = df[group_col].unique(maintain_order=True).to_list()

        # Calculate target counts per fold per bin
        bin_counts = df.group_by("bin").len().sort("bin")
        fold_targets = {}
        for row in bin_counts.iter_rows(named=True):
            bin_val = row["bin"]
            total = row["len"]
            fold_targets[bin_val] = [total // n_folds] * n_folds
            # Distribute remainder
            for i in range(total % n_folds):
                fold_targets[bin_val][i] += 1

        # Assign groups to folds using greedy algorithm
        fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
        fold_remaining = {
            bin_val: counts.copy() for bin_val, counts in fold_targets.items()
        }

        for group in group_list:
            # Get bin distribution for this group
            group_bin_counts = (
                df.filter(pl.col(group_col) == group).group_by("bin").len()
            )
            group_bins = {
                row["bin"]: row["len"]
                for row in group_bin_counts.iter_rows(named=True)
            }

            # Find best fold (one with most remaining capacity for this group)
            best_fold = None
            best_score = -1

            for fold_idx in range(n_folds):
                # Check if group fits in this fold
                fits = True
                score = 0
                for bin_val, count in group_bins.items():
                    if count > fold_remaining[bin_val][fold_idx]:
                        fits = False
                        break
                    score += fold_remaining[bin_val][fold_idx]

                if fits and score > best_score:
                    best_fold = fold_idx
                    best_score = score

            # Assign to best fold (or first if none fit perfectly)
            if best_fold is None:
                best_fold = 0

            fold_groups[best_fold].append(group)

            # Update remaining counts
            for bin_val, count in group_bins.items():
                fold_remaining[bin_val][best_fold] = max(
                    0, fold_remaining[bin_val][best_fold] - count
                )

        # Create CV DataFrame with fold assignments
        fold_dfs = []
        for fold_idx, groups in enumerate(fold_groups):
            fold_samples = df.filter(pl.col(group_col).is_in(groups))[
                "sample"
            ].to_list()
            fold_df = pl.DataFrame(
                {
                    "sample": fold_samples,
                    "fold": [fold_idx] * len(fold_samples),
                }
            )
            fold_dfs.append(fold_df)

        cv_df = pl.concat(fold_dfs)

        # Store CV scheme
        self.cv_schemes[scheme_name] = cv_df

    def _stratified_split(
        self,
        data: pl.DataFrame,
        target_col: str,
        grouping: Optional[str],
        test_size: float,
        n_bins: int,
        random_state: int,
        split_col_name: str,
    ) -> pl.DataFrame:
        """Internal method to perform stratified group-aware split.

        Args:
            data: DataFrame with sample, target, and optional grouping columns
            target_col: Name of target column
            grouping: Name of grouping column (or None)
            test_size: Fraction for test set
            n_bins: Number of bins for continuous targets
            random_state: Random seed
            split_col_name: Name for split column in output

        Returns:
            DataFrame with {sample, split_col_name} columns
        """
        # Shuffle and prepare grouping
        df = data.sample(fraction=1.0, shuffle=True, seed=random_state)

        if grouping is not None:
            group_col = grouping
        else:
            df = df.with_columns(pl.col("sample").alias("_group"))
            group_col = "_group"

        group_list = df[group_col].unique(maintain_order=True).to_list()

        # Determine bins/classes
        is_continuous = df.schema[target_col] in [pl.Float32, pl.Float64]
        if is_continuous:
            min_val, max_val = df.select(
                pl.min(target_col).alias("min"),
                pl.max(target_col).alias("max"),
            ).row(0)
            breaks = np.linspace(min_val, max_val, n_bins + 1)
            df = df.with_columns(
                pl.col(target_col)
                .cut(breaks[1:-1], left_closed=True)
                .alias("bin")
            )
        else:
            df = df.with_columns(pl.col(target_col).cast(pl.Utf8).alias("bin"))

        # Calculate target test counts per bin
        target_test_counts_df = df.group_by("bin").agg(
            (pl.len() * test_size).round(0).alias("target_count")
        )
        target_test_counts = {
            row["bin"]: row["target_count"]
            for row in target_test_counts_df.iter_rows(named=True)
        }

        # Assign groups to train/test
        train_groups, test_groups = [], []
        for group in group_list:
            group_bin_counts_df = (
                df.filter(pl.col(group_col) == group).group_by("bin").len()
            )
            group_bin_counts = {
                row["bin"]: row["len"]
                for row in group_bin_counts_df.iter_rows(named=True)
            }

            can_add_to_test = True
            for bin_val, count in group_bin_counts.items():
                if count > target_test_counts.get(bin_val, 0):
                    can_add_to_test = False
                    break

            if can_add_to_test:
                test_groups.append(group)
                for bin_val, count in group_bin_counts.items():
                    target_test_counts[bin_val] -= count
            else:
                train_groups.append(group)

        # Create split DataFrame
        test_samples = df.filter(pl.col(group_col).is_in(test_groups))[
            "sample"
        ].to_list()
        train_samples = df.filter(pl.col(group_col).is_in(train_groups))[
            "sample"
        ].to_list()

        test_df = pl.DataFrame(
            {
                "sample": test_samples,
                split_col_name: ["test"] * len(test_samples),
            }
        )
        train_df = pl.DataFrame(
            {
                "sample": train_samples,
                split_col_name: ["train"] * len(train_samples),
            }
        )

        return pl.concat([test_df, train_df])
