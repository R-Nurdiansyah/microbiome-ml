"""Split management for train/test and cross-validation."""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from sklearn.model_selection import LeaveOneGroupOut

logger = logging.getLogger(__name__)


class SplitManager:
    """Manages train/test splits and cross-validation folds for a single label.

    Coordinates holdout splits and multiple CV schemes (random, grouped, etc.)
    for one target variable.

    Attributes:
        label: Name of the target variable
        holdout: DataFrame with {sample, split, label} plus any metadata columns for the holdout samples
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
        metadata: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
    ) -> None:
        """Create holdout train/test split.

        Args:
            data: DataFrame with sample, label, and optional grouping columns
            grouping: Column name for grouping samples (prevents group leakage)
            test_size: Fraction of samples for test set
            n_bins: Number of bins for continuous targets
            random_state: Random seed for reproducibility
            metadata: Optional metadata DataFrame/LazyFrame to attach to results
        """
        self.holdout = self._stratified_split(
            data=data,
            target_col=self.label,
            grouping=grouping,
            test_size=test_size,
            n_bins=n_bins,
            random_state=random_state,
            split_col_name="split",
            metadata=metadata,
        )

    def create_cv_folds(
        self,
        data: pl.DataFrame,
        grouping: Optional[str] = None,
        n_folds: int = 5,
        n_bins: int = 5,
        random_state: int = 42,
        scheme_name: Optional[str] = None,
        use_holdout: bool = False,
        strict: bool = False,
        method: str = "kfold",
        max_groups: int = 50,
    ) -> None:
        """Create cross-validation splits.

        Args:
            data: DataFrame with sample, label, and optional grouping columns
            grouping: Column name for grouping samples (prevents group leakage)
            n_folds: Number of folds (``method="kfold"`` only)
            n_bins: Number of bins for continuous targets (``kfold`` only)
            random_state: Random seed for reproducibility
            scheme_name: Name for this CV scheme (defaults to grouping value or "random")
            use_holdout: If True, only use training samples from holdout split, must be created first with create_holdout_split()
            strict: If True, skip creating the scheme when fewer than n_folds
                    folds contain samples, and log a warning instead.
                    ``kfold`` only -- LOGO's fold count comes from the data.
            method: ``"kfold"`` (default) packs the groups into ``n_folds``
                stratified folds. ``"logo"`` gives every group a fold of
                its own -- sklearn's ``LeaveOneGroupOut`` -- so each
                iteration validates on exactly one group. Needs ``grouping``.
            max_groups: ``logo`` only. Refuse the scheme above this many
                groups, since LOGO runs one CV iteration per group.
        """
        # Default scheme_name based on grouping
        if scheme_name is None:
            scheme_name = grouping if grouping is not None else "random"

        if method not in ("kfold", "logo"):
            raise ValueError(
                f"method must be 'kfold' or 'logo', got {method!r}"
            )
        if method == "logo":
            self._create_logo_folds(
                data=data,
                grouping=grouping,
                scheme_name=scheme_name,
                max_groups=max_groups,
            )
            return

        # Shuffle data
        df = data.sample(fraction=1.0, shuffle=True, seed=random_state)

        if df.height == 0:
            # No sample has a value for this label (or none survived the
            # holdout-train filter). Binning an empty column would fail on
            # a missing min/max, so record no scheme.
            logger.warning(
                "CV scheme '%s' for label '%s' has no samples; skipping",
                scheme_name,
                self.label,
            )
            return

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

        # Check if fewer folds than requested contain samples (can happen when
        # data/grouping diversity is limited and some folds remain empty).
        try:
            present_folds = set(cv_df.select("fold").to_series().to_list())
            n_present = len(present_folds)
        except Exception:
            present_folds = set()
            n_present = 0

        if n_present < n_folds:
            if strict:
                logger.warning(
                    "Skipping CV scheme '%s': requested n_folds=%d but only %d fold(s) "
                    "would contain samples (strict=True). Use strict=False to allow "
                    "schemes with fewer folds.",
                    scheme_name,
                    n_folds,
                    n_present,
                )
                return
            else:
                logger.warning(
                    "CV scheme '%s': requested n_folds=%d but only %d fold(s) contain "
                    "samples (this can occur with limited data or grouping).",
                    scheme_name,
                    n_folds,
                    n_present,
                )

        # Store CV scheme
        self.cv_schemes[scheme_name] = cv_df

    def _create_logo_folds(
        self,
        data: pl.DataFrame,
        grouping: Optional[str],
        scheme_name: str,
        max_groups: int,
    ) -> None:
        """Leave-one-group-out folds: one fold per group.

        The enumeration is sklearn's ``LeaveOneGroupOut``, so the fold
        table reproduces exactly what that splitter would do. It is stored
        as ``{sample, fold}`` like every other scheme, and replayed through
        ``PredefinedSplit`` by ``CrossValidator``.
        """
        if grouping is None:
            logger.warning(
                "CV scheme '%s': leave-one-group-out needs a grouping "
                "column (every sample would be its own fold); skipping",
                scheme_name,
            )
            return
        if data.height == 0:
            logger.warning(
                "CV scheme '%s' for label '%s' has no samples; skipping",
                scheme_name,
                self.label,
            )
            return

        # Dataset.create_cv_folds drops these already, but a caller using
        # SplitManager directly may not have: a null group is not a group,
        # and pooling the nulls into one fold would mix unrelated samples.
        usable = data.drop_nulls(subset=[grouping])
        if usable.height < data.height:
            logger.warning(
                "CV scheme '%s': dropped %d sample(s) with no '%s' value",
                scheme_name,
                data.height - usable.height,
                grouping,
            )
        if usable.height == 0:
            logger.warning(
                "CV scheme '%s' for label '%s': no sample has a '%s' "
                "value; skipping",
                scheme_name,
                self.label,
                grouping,
            )
            return

        samples = usable["sample"].to_list()
        groups = usable[grouping].cast(pl.Utf8).to_list()
        counts = Counter(groups)

        # A one-sample group cannot be a validation fold: R2 is undefined
        # on a single point, and CrossValidator skips a scheme that has a
        # one-sample fold. PredefinedSplit reads -1 as "always train, never
        # validate", so those samples still inform every iteration instead
        # of being thrown away.
        testable = sorted(g for g, n in counts.items() if n >= 2)
        singletons = len(counts) - len(testable)

        if len(testable) < 2:
            logger.warning(
                "CV scheme '%s' for label '%s': only %d group(s) of '%s' "
                "have 2+ samples, need 2; skipping",
                scheme_name,
                self.label,
                len(testable),
                grouping,
            )
            return
        if len(testable) > max_groups:
            logger.warning(
                "CV scheme '%s' for label '%s': '%s' has %d usable groups, "
                "over max_groups=%d. LOGO runs one CV iteration per group, "
                "so this would be %d fits per model and parameter set; "
                "skipping. Raise split.cv.logo_max_groups to allow it.",
                scheme_name,
                self.label,
                grouping,
                len(testable),
                max_groups,
                len(testable),
            )
            return

        group_arr = np.asarray(groups, dtype=object)
        keep = np.isin(group_arr, testable)
        kept_idx = np.flatnonzero(keep)

        fold = np.full(len(samples), -1, dtype=int)
        logo = LeaveOneGroupOut()
        for fold_id, (_, test_idx) in enumerate(
            logo.split(np.zeros((len(kept_idx), 1)), groups=group_arr[keep])
        ):
            fold[kept_idx[test_idx]] = fold_id

        self.cv_schemes[scheme_name] = pl.DataFrame(
            {"sample": samples, "fold": fold.tolist()}
        )
        logger.info(
            "CV scheme '%s' for label '%s': leave-one-group-out over '%s' "
            "-- %d folds; %d sample(s) in %d single-sample group(s) are "
            "training-only (fold -1)",
            scheme_name,
            self.label,
            grouping,
            len(testable),
            int((fold < 0).sum()),
            singletons,
        )

    def _stratified_split(
        self,
        data: pl.DataFrame,
        target_col: str,
        grouping: Optional[str],
        test_size: float,
        n_bins: int,
        random_state: int,
        split_col_name: str,
        metadata: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
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
            metadata: Optional metadata DataFrame/LazyFrame to append

        Returns:
            DataFrame with {sample, split_col_name} columns
        """
        # Shuffle and prepare grouping
        df = data.sample(fraction=1.0, shuffle=True, seed=random_state)

        if target_col not in df.columns:
            logger.error(
                "Target column '%s' not found in data for holdout split",
                target_col,
            )
            raise ValueError(
                f"Target column '{target_col}' not found in source data"
            )

        if df.height == 0:
            # Nothing to split -- e.g. a label that is null for every
            # sample. Binning an empty column would divide by a missing
            # min/max; return the empty split so callers see a holdout
            # that exists and is simply empty.
            logger.warning(
                "No samples with a value for '%s'; empty split",
                target_col,
            )
            return self._enrich_holdout(
                holdout_df=df.select(
                    "sample",
                    pl.lit("train", dtype=pl.Utf8).alias(split_col_name),
                ),
                data=data,
                target_col=target_col,
                metadata=metadata,
                split_col_name=split_col_name,
            )

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
        total_samples = len(test_samples) + len(train_samples)

        # Try to adjust group assignments to better match requested test_size
        desired_test_count = int(round(total_samples * test_size))
        achieved_test_count = len(test_samples)

        if achieved_test_count != desired_test_count:
            # Precompute group sizes
            group_counts = {
                row[group_col]: row["count"]
                for row in df.group_by(group_col)
                .agg(pl.len().alias("count"))
                .iter_rows(named=True)
            }

            # Helper to compute best single-group move
            def best_move(
                candidates: List[str], direction: int
            ) -> Tuple[Optional[str], int]:
                # direction = +1 means move group from train->test (increase achieved)
                # direction = -1 means move group from test->train (decrease achieved)
                best_g = None
                best_diff = abs(desired_test_count - achieved_test_count)
                for g in candidates:
                    size = group_counts.get(g, 0)
                    potential = achieved_test_count + direction * size
                    diff = abs(desired_test_count - potential)
                    if diff < best_diff:
                        best_diff = diff
                        best_g = g
                return best_g, best_diff

            # Iteratively move the single group that most improves the closeness
            moved = True
            # Use sets for faster membership updates
            test_set = set(test_groups)
            train_set = set(train_groups)

            while moved:
                moved = False
                current_diff = abs(desired_test_count - achieved_test_count)
                if achieved_test_count < desired_test_count and train_set:
                    # need more test samples -> consider moving from train to test
                    # Sorted: best_move keeps the first of equal candidates,
                    # and a set of strings iterates in a per-process order,
                    # which gave one seed several different holdouts.
                    g, new_diff = best_move(sorted(train_set), +1)
                    if g is not None and new_diff < current_diff:
                        train_set.remove(g)
                        test_set.add(g)
                        achieved_test_count += group_counts.get(g, 0)
                        moved = True
                elif achieved_test_count > desired_test_count and test_set:
                    # too many test samples -> move from test to train
                    g, new_diff = best_move(sorted(test_set), -1)
                    if g is not None and new_diff < current_diff:
                        test_set.remove(g)
                        train_set.add(g)
                        achieved_test_count -= group_counts.get(g, 0)
                        moved = True

            # Replace group lists after adjustments
            test_groups = list(test_set)
            train_groups = list(train_set)

        # Recreate sample lists after possible adjustments
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

        # Ensure consistent string dtypes even when one side is empty
        try:
            test_df = test_df.with_columns(
                pl.col("sample").cast(pl.Utf8),
                pl.col(split_col_name).cast(pl.Utf8),
            )
        except Exception:
            pass
        try:
            train_df = train_df.with_columns(
                pl.col("sample").cast(pl.Utf8),
                pl.col(split_col_name).cast(pl.Utf8),
            )
        except Exception:
            pass

        # Log achieved ratio vs requested
        final_test = len(test_samples)
        final_train = len(train_samples)
        final_total = final_test + final_train
        final_ratio = final_test / final_total if final_total > 0 else 0.0
        if abs(final_ratio - test_size) > 1e-6:
            logger.info(
                f"Requested test_size={test_size:.3f}, achieved test proportion={final_ratio:.3f} (n_test={final_test}, n_train={final_train}, n_total={final_total})"
            )

        holdout_df = pl.concat([test_df, train_df])

        return self._enrich_holdout(
            holdout_df=holdout_df,
            data=data,
            target_col=target_col,
            metadata=metadata,
            split_col_name=split_col_name,
        )

    def _enrich_holdout(
        self,
        holdout_df: pl.DataFrame,
        data: pl.DataFrame,
        target_col: str,
        metadata: Optional[Union[pl.DataFrame, pl.LazyFrame]],
        split_col_name: str,
    ) -> pl.DataFrame:
        """Attach label values and optional metadata to the split results."""

        label_df = data.select(["sample", target_col])
        result = holdout_df.join(
            label_df, on="sample", how="left", coalesce=True
        )
        metadata_cols: List[str] = []
        if metadata is not None:
            if isinstance(metadata, pl.LazyFrame):
                metadata_df = metadata.collect()
            else:
                metadata_df = metadata

            if "sample" in metadata_df.columns:
                metadata_df = metadata_df.unique(subset=["sample"])
                metadata_cols = [
                    col for col in metadata_df.columns if col != "sample"
                ]
                if metadata_cols:
                    metadata_subset = metadata_df.select(
                        ["sample"] + metadata_cols
                    )
                    result = result.join(
                        metadata_subset, on="sample", how="left", coalesce=True
                    )

        final_columns = ["sample", split_col_name, target_col] + metadata_cols
        available_columns = [
            col for col in final_columns if col in result.columns
        ]
        return result.select(available_columns)
