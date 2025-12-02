import numpy as np
import polars as pl
import pytest

from microbiome_ml.wrangle.dataset import Dataset


@pytest.fixture
def splitting_dataset():
    """Create a dataset with labels and groupings for splitting tests."""
    n_samples = 100
    samples = [f"s{i}" for i in range(n_samples)]

    # Continuous target (0 to 1)
    targets_cont = np.linspace(0, 1, n_samples)

    # Categorical target (A, B)
    targets_cat = ["A"] * 50 + ["B"] * 50

    # Groups (10 groups of 10 samples)
    groups = []
    for i in range(10):
        groups.extend([f"g{i}"] * 10)

    labels = pl.DataFrame(
        {
            "sample": samples,
            "target_cont": targets_cont,
            "target_cat": targets_cat,
        }
    )

    groupings = pl.DataFrame({"sample": samples, "group": groups})

    dataset = Dataset()
    dataset.add_labels(labels)
    dataset.add_groupings(groupings)

    return dataset


def test_continuous_splitting(splitting_dataset):
    """Test splitting with continuous target."""
    dataset = splitting_dataset
    dataset.create_holdout_split(
        label="target_cont", test_size=0.2, n_bins=5, random_state=42
    )

    assert "target_cont" in dataset.splits
    assert dataset.splits["target_cont"].holdout is not None

    splits = dataset.splits["target_cont"].holdout
    assert "split" in splits.columns
    assert "sample" in splits.columns

    train_count = splits.filter(pl.col("split") == "train").height
    test_count = splits.filter(pl.col("split") == "test").height

    assert train_count + test_count == 100
    # Allow some variance due to binning/rounding
    assert 15 <= test_count <= 25


def test_categorical_splitting(splitting_dataset):
    """Test splitting with categorical target."""
    dataset = splitting_dataset
    dataset.create_holdout_split(
        label="target_cat", test_size=0.2, random_state=42
    )

    splits = dataset.splits["target_cat"].holdout
    test_samples = splits.filter(pl.col("split") == "test")

    # Check stratification
    # Should have roughly equal A and B in test
    # Total test size ~20
    # A ~ 10, B ~ 10

    # Join with labels to check
    test_with_labels = test_samples.join(dataset.labels, on="sample")
    counts = test_with_labels["target_cat"].value_counts()

    count_a = counts.filter(pl.col("target_cat") == "A")["count"][0]
    count_b = counts.filter(pl.col("target_cat") == "B")["count"][0]

    assert abs(count_a - count_b) <= 4  # Should be close


def test_grouped_splitting(splitting_dataset):
    """Test splitting with groups."""
    dataset = splitting_dataset
    # Groups g0-g9. 10 groups.
    # Target continuous is correlated with group (since both are ordered)
    # g0 has low values, g9 has high values.

    dataset.create_holdout_split(
        label="target_cont",
        grouping="group",
        test_size=0.2,
        n_bins=2,  # Low/High
        random_state=42,
    )

    splits = dataset.splits["target_cont"].holdout

    # Check that groups are not split
    # Join with groupings
    splits_with_groups = splits.join(dataset.groupings, on="sample")

    # Check if any group has both train and test
    group_split_counts = splits_with_groups.group_by("group").agg(
        pl.col("split").n_unique().alias("n_splits")
    )

    assert group_split_counts.filter(pl.col("n_splits") > 1).height == 0

    # Check test size (should be 2 groups = 20 samples)
    test_count = splits.filter(pl.col("split") == "test").height
    assert test_count == 20


def test_null_handling():
    """Test that nulls are excluded."""
    dataset = Dataset()
    labels = pl.DataFrame(
        {"sample": ["s1", "s2", "s3", "s4"], "target": [1.0, 2.0, None, 4.0]}
    )
    dataset.add_labels(labels)

    dataset.create_holdout_split(
        label="target", test_size=0.5, n_bins=2, random_state=42
    )

    # s3 should be missing from splits or filtered out?
    # The implementation returns only train/test samples.
    # So s3 should not be in splits df.

    assert dataset.splits["target"].holdout.height == 3
    assert "s3" not in dataset.splits["target"].holdout["sample"].to_list()


def test_save_load_splits(splitting_dataset, tmp_path):
    """Test saving and loading splits."""
    dataset = splitting_dataset
    dataset.create_holdout_split(label="target_cont", test_size=0.2)

    save_path = tmp_path / "dataset"
    dataset.save(save_path)

    loaded = Dataset.load(save_path)
    assert "target_cont" in loaded.splits
    assert loaded.splits["target_cont"].holdout is not None
    assert loaded.splits["target_cont"].holdout.height == 100
    assert "split" in loaded.splits["target_cont"].holdout.columns


def test_cv_folds_creation(splitting_dataset):
    """Test creating k-fold cross-validation splits."""
    dataset = splitting_dataset
    dataset.create_cv_folds(label="target_cont", n_folds=5, random_state=42)

    assert "target_cont" in dataset.splits
    assert "random" in dataset.splits["target_cont"].cv_schemes

    cv_df = dataset.splits["target_cont"].cv_schemes["random"]
    assert "sample" in cv_df.columns
    assert "fold" in cv_df.columns
    assert cv_df.height == 100

    # Check that all folds are present
    fold_counts = cv_df["fold"].value_counts().sort("fold")
    assert fold_counts.height == 5
    # Each fold should have ~20 samples
    for count in fold_counts["count"]:
        assert 15 <= count <= 25


def test_cv_folds_grouped(splitting_dataset):
    """Test creating k-fold CV with grouping."""
    dataset = splitting_dataset
    dataset.create_cv_folds(
        label="target_cont", grouping="group", n_folds=5, random_state=42
    )

    assert "target_cont" in dataset.splits
    # Scheme should be named after grouping column
    assert "group" in dataset.splits["target_cont"].cv_schemes

    cv_df = dataset.splits["target_cont"].cv_schemes["group"]

    # Check that groups are not split across folds
    cv_with_groups = cv_df.join(dataset.groupings, on="sample")
    group_fold_counts = cv_with_groups.group_by("group").agg(
        pl.col("fold").n_unique().alias("n_folds")
    )

    # Each group should only be in one fold
    assert group_fold_counts.filter(pl.col("n_folds") > 1).height == 0


def test_iter_cv_folds(splitting_dataset):
    """Test iterating over CV folds."""
    dataset = splitting_dataset

    # Create multiple CV schemes
    dataset.create_cv_folds(
        label="target_cont", n_folds=5, grouping="random", random_state=42
    )
    dataset.create_cv_folds(
        label="target_cont", grouping="group", n_folds=5, random_state=42
    )
    dataset.create_cv_folds(label="target_cat", n_folds=3, random_state=42)

    # Iterate over all
    all_combos = list(dataset.iter_cv_folds())
    assert len(all_combos) == 4  # 2 for target_cont + 2 for target_cat

    # Iterate over specific label
    cont_schemes = list(dataset.iter_cv_folds(label="target_cont"))
    assert len(cont_schemes) == 2  # random and group schemes

    # Iterate over specific scheme
    random_schemes = list(dataset.iter_cv_folds(scheme_name="random"))
    assert (
        len(random_schemes) == 2
    )  # target_cont and target_cat both have random


def test_auto_iteration_holdout(splitting_dataset):
    """Test auto-iteration when creating splits for all labels."""
    dataset = splitting_dataset

    # Create splits for all labels at once
    dataset.create_holdout_split(test_size=0.2, random_state=42)

    # Should have created splits for both labels
    assert "target_cont" in dataset.splits
    assert "target_cat" in dataset.splits
    assert dataset.splits["target_cont"].holdout is not None
    assert dataset.splits["target_cat"].holdout is not None


def test_auto_iteration_cv(splitting_dataset):
    """Test auto-iteration when creating CV folds for all labels."""
    dataset = splitting_dataset

    # Create CV for all labels at once
    dataset.create_cv_folds(n_folds=3, random_state=42)

    # Should have created CV for both labels
    assert "target_cont" in dataset.splits
    assert "target_cat" in dataset.splits
    assert "random" in dataset.splits["target_cont"].cv_schemes
    assert "random" in dataset.splits["target_cat"].cv_schemes
