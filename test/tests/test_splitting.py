from pathlib import Path

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


def test_save_holdout_splits_single_label(splitting_dataset, tmp_path):
    """Holdout split is written to <output_dir>/<label>/holdout.csv."""
    dataset = splitting_dataset
    dataset.create_holdout_split(
        label="target_cont", test_size=0.2, random_state=42
    )

    written = dataset.save_holdout_splits(tmp_path, label="target_cont")

    expected = tmp_path / "target_cont" / "holdout.csv"
    assert written == {"target_cont": expected}
    assert expected.exists()

    saved = pl.read_csv(expected)
    assert saved.height == 100
    assert set(saved["split"].unique().to_list()) == {"train", "test"}
    assert saved.sort("sample").equals(
        dataset.splits["target_cont"].holdout.sort("sample")
    )


def test_create_holdout_split_output_dir(splitting_dataset, tmp_path):
    """Passing output_dir to create_holdout_split saves every label."""
    dataset = splitting_dataset
    out_dir = tmp_path / "holdout_splits"

    dataset.create_holdout_split(
        test_size=0.2, random_state=42, output_dir=out_dir
    )

    for label in ("target_cont", "target_cat"):
        path = out_dir / label / "holdout.csv"
        assert path.exists(), f"missing {path}"
        assert pl.read_csv(path).height == 100


def test_save_holdout_splits_errors(splitting_dataset, tmp_path):
    """Saving before any holdout exists (or a missing label) raises."""
    dataset = splitting_dataset

    with pytest.raises(ValueError):
        dataset.save_holdout_splits(tmp_path)

    dataset.create_holdout_split(label="target_cont", random_state=42)
    with pytest.raises(ValueError):
        dataset.save_holdout_splits(tmp_path, label="target_cat")


def test_save_cv_folds_writes_one_csv_per_scheme(splitting_dataset, tmp_path):
    """cv_<scheme>.csv is written per label with only holdout-train rows."""
    dataset = splitting_dataset
    dataset.create_holdout_split(
        label="target_cont", test_size=0.2, random_state=42
    )
    dataset.create_cv_folds(
        label="target_cont", n_folds=3, random_state=42, use_holdout=True
    )

    written = dataset.save_cv_folds(tmp_path, label="target_cont")

    assert "target_cont" in written
    schemes = written["target_cont"]
    assert schemes  # at least the 'random' scheme
    for scheme, path in schemes.items():
        assert path == tmp_path / "target_cont" / f"cv_{scheme}.csv"
        cv = pl.read_csv(path)
        assert {"sample", "fold"} <= set(cv.columns)
        # folds come from holdout-train only: no test sample may appear
        holdout = dataset.splits["target_cont"].holdout
        test_samples = set(
            holdout.filter(pl.col("split") == "test")["sample"].to_list()
        )
        assert not (set(cv["sample"].to_list()) & test_samples)


def test_save_cv_folds_errors_without_folds(splitting_dataset, tmp_path):
    dataset = splitting_dataset
    with pytest.raises(ValueError):
        dataset.save_cv_folds(tmp_path)
    dataset.create_holdout_split(label="target_cont", random_state=42)
    with pytest.raises(ValueError):
        dataset.save_cv_folds(tmp_path, label="target_cont")


@pytest.fixture
def taxon_dataset():
    """Dataset whose family feature set reproduces the worked example:

        s1: f__a, f__b, f__c
        s2:       f__c, f__d, f__e
        s3: f__x, f__b, f__c
        s4:       f__b, f__c, f__d

    f__a and f__e and f__x are each in exactly 1 of 4 samples (25%).
    """
    from microbiome_ml.wrangle.features import SampleFeatureSet

    samples = ["s1", "s2", "s3", "s4"]
    taxa = ["f__a", "f__b", "f__c", "f__d", "f__e", "f__x"]
    present = {
        "s1": {"f__a", "f__b", "f__c"},
        "s2": {"f__c", "f__d", "f__e"},
        "s3": {"f__x", "f__b", "f__c"},
        "s4": {"f__b", "f__c", "f__d"},
    }
    matrix = np.array(
        [[1.0 if t in present[s] else 0.0 for t in taxa] for s in samples]
    )

    dataset = Dataset()
    dataset.add_labels(
        pl.DataFrame({"sample": samples, "target": [1.0, 2.0, 3.0, 4.0]})
    )
    dataset.feature_sets["tax_family"] = SampleFeatureSet(
        accessions=samples,
        feature_names=taxa,
        features=matrix,
        name="tax_family",
    )
    return dataset


def test_taxon_prevalence(taxon_dataset):
    prev = taxon_dataset.taxon_prevalence("family")

    as_dict = dict(zip(prev["taxon"], prev["prevalence"]))
    assert as_dict["f__c"] == 1.0  # in all 4
    assert as_dict["f__b"] == 0.75
    assert as_dict["f__d"] == 0.5
    assert as_dict["f__a"] == 0.25
    assert prev["n_samples"][0] == 4


def test_taxon_holdout_picks_prevalence_nearest_test_size(taxon_dataset):
    """25% prevalence is nearest to test_size=0.2, so a singleton taxon
    defines the holdout and its one sample is the test set."""
    taxon_dataset.create_taxon_holdout_split(
        rank="family", label="target", test_size=0.2, min_test_samples=1
    )

    holdout = taxon_dataset.splits["target"].holdout
    assert holdout is not None
    assert holdout.height == 4
    test = holdout.filter(pl.col("split") == "test")
    assert test.height == 1

    taxon = holdout["holdout_taxon"][0]
    assert taxon in {"f__a", "f__e", "f__x"}  # the 25% taxa
    assert holdout["holdout_rank"][0] == "family"
    # per-sample flag: the single test sample carries the taxon, the rest
    # do not -- the two sides of the split differ by exactly this
    assert holdout.filter(pl.col("split") == "test")[
        "holdout_taxon_presence"
    ].to_list() == [True]
    assert not any(
        holdout.filter(pl.col("split") == "train")[
            "holdout_taxon_presence"
        ].to_list()
    )


def test_taxon_holdout_train_never_contains_the_taxon(taxon_dataset):
    taxon_dataset.create_taxon_holdout_split(
        rank="f__", label="target", test_size=0.2, min_test_samples=1
    )

    holdout = taxon_dataset.splits["target"].holdout
    taxon = holdout["holdout_taxon"][0]
    feature_df = taxon_dataset.feature_sets["tax_family"].to_df()
    carriers = set(feature_df.filter(pl.col(taxon) > 0)["sample"].to_list())
    test_samples = set(
        holdout.filter(pl.col("split") == "test")["sample"].to_list()
    )
    train_samples = set(
        holdout.filter(pl.col("split") == "train")["sample"].to_list()
    )
    assert test_samples == carriers
    assert not (train_samples & carriers)


def test_taxon_holdout_prefers_presence_on_ties(taxon_dataset):
    """Presence and absence are equidistant here (25% vs 75%); the rule is
    to prefer presence, so the test set carries the taxon."""
    taxon_dataset.create_taxon_holdout_split(
        rank="family", label="target", test_size=0.25, min_test_samples=1
    )
    holdout = taxon_dataset.splits["target"].holdout
    test_rows = holdout.filter(pl.col("split") == "test")
    assert test_rows.height == 1
    # presence (not absence) won the tie: the test sample is a carrier
    assert test_rows["holdout_taxon_presence"].to_list() == [True]


def test_taxon_holdout_skips_when_no_usable_taxon(taxon_dataset):
    """min_test_samples=3 cannot be met on both sides of any split here.

    The label is registered with an EMPTY holdout rather than left out:
    create_cv_folds(use_holdout=True) raises on a missing split manager,
    so one unusable label would otherwise abort the whole run.
    """
    taxon_dataset.create_taxon_holdout_split(
        rank="family", label="target", test_size=0.2, min_test_samples=3
    )
    holdout = taxon_dataset.splits["target"].holdout
    assert holdout is not None
    assert holdout.height == 0


def test_taxon_holdout_unknown_rank_raises(taxon_dataset):
    with pytest.raises(ValueError, match="Invalid taxonomic rank"):
        taxon_dataset.create_taxon_holdout_split(rank="nonsense")


def test_taxon_holdout_missing_feature_set_raises(taxon_dataset):
    with pytest.raises(ValueError, match="No feature set for rank"):
        taxon_dataset.create_taxon_holdout_split(rank="genus")


def test_holdout_spec_tokens():
    """Sweep tokens map to the right holdout strategy and directory name."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import pipeline_steps as steps

    assert steps.holdout_spec("random") == ("column", None)
    assert steps.holdout_spec("null") == ("column", None)
    assert steps.holdout_spec("bioproject") == ("column", "bioproject")
    assert steps.holdout_spec("taxon:family") == ("taxon", "family")
    assert steps.holdout_spec("taxon_family") == ("taxon", "family")

    assert steps.spec_dir_name("random") == "random"
    assert steps.spec_dir_name("bioproject") == "bioproject"
    assert steps.spec_dir_name("taxon:family") == "taxon_family"
    assert steps.spec_dir_name("taxon_family") == "taxon_family"


def test_taxon_holdout_ignores_zero_coverage_taxa(taxon_dataset):
    """A taxon with no coverage anywhere must never define the holdout.

    Splitting on an all-zero taxon would put every sample on one side, and
    "absence" of something nobody has is not a meaningful test set.
    """
    from microbiome_ml.wrangle.features import SampleFeatureSet

    samples = ["s1", "s2", "s3", "s4"]
    taxa = ["f__a", "f__b", "f__c", "f__d", "f__e", "f__x", "f__ghost"]
    present = {
        "s1": {"f__a", "f__b", "f__c"},
        "s2": {"f__c", "f__d", "f__e"},
        "s3": {"f__x", "f__b", "f__c"},
        "s4": {"f__b", "f__c", "f__d"},
    }
    matrix = np.array(
        [[1.0 if t in present[s] else 0.0 for t in taxa] for s in samples]
    )
    taxon_dataset.feature_sets["tax_family"] = SampleFeatureSet(
        accessions=samples,
        feature_names=taxa,
        features=matrix,
        name="tax_family",
    )

    prev = dict(
        zip(
            taxon_dataset.taxon_prevalence("family")["taxon"],
            taxon_dataset.taxon_prevalence("family")["prevalence"],
        )
    )
    assert prev["f__ghost"] == 0.0

    taxon_dataset.create_taxon_holdout_split(
        rank="family", label="target", test_size=0.2, min_test_samples=1
    )
    holdout = taxon_dataset.splits["target"].holdout
    chosen = holdout["holdout_taxon"][0]
    assert chosen != "f__ghost"

    # the chosen taxon really is carried by the test samples
    feature_df = taxon_dataset.feature_sets["tax_family"].to_df()
    test_samples = holdout.filter(pl.col("split") == "test")["sample"]
    carried = feature_df.filter(pl.col("sample").is_in(test_samples.to_list()))
    assert (carried[chosen] > 0).all()


def test_feature_set_reload_casts_sparse_columns(tmp_path):
    """A taxon that is empty in the first CSV rows but has values later must
    come back numeric, not String (the production failure mode)."""
    from microbiome_ml.wrangle.features import SampleFeatureSet

    n = 150  # more than polars' default 100-row inference window
    samples = [f"s{i}" for i in range(n)]
    # empty for the first 120 rows, non-zero afterwards
    sparse = [None] * 120 + [0.5] * (n - 120)
    dense = [float(i) for i in range(n)]

    df = pl.DataFrame(
        {"sample": samples, "f__dense": dense, "f__sparse": sparse}
    )
    path = tmp_path / "tax_family.csv"
    df.write_csv(path)

    # Plain scan_csv infers String for this column (its inference window is
    # all empty); not asserted here, as that is polars behaviour, not ours.
    loaded = SampleFeatureSet.load(path, name="tax_family").to_df()
    assert loaded["f__sparse"].dtype == pl.Float64
    assert loaded["f__dense"].dtype == pl.Float64
    # the real values survived the cast
    assert loaded["f__sparse"].sum() == pytest.approx(0.5 * (n - 120))
    # and a comparison against a float no longer raises
    assert (loaded["f__sparse"] > 0).sum() == n - 120


def test_holdout_taxon_presence_separates_the_two_sides(taxon_dataset):
    """holdout_taxon_presence is per sample and must be uniform within each
    side and opposite between them -- that is what the split encodes."""
    taxon_dataset.create_taxon_holdout_split(
        rank="family", label="target", test_size=0.2, min_test_samples=1
    )
    holdout = taxon_dataset.splits["target"].holdout

    test_flags = set(
        holdout.filter(pl.col("split") == "test")[
            "holdout_taxon_presence"
        ].to_list()
    )
    train_flags = set(
        holdout.filter(pl.col("split") == "train")[
            "holdout_taxon_presence"
        ].to_list()
    )
    assert len(test_flags) == 1
    assert len(train_flags) == 1
    assert test_flags != train_flags

    # and the flag agrees with the actual community profile
    taxon = holdout["holdout_taxon"][0]
    feature_df = taxon_dataset.feature_sets["tax_family"].to_df()
    carriers = set(feature_df.filter(pl.col(taxon) > 0)["sample"].to_list())
    flagged = set(
        holdout.filter(pl.col("holdout_taxon_presence"))["sample"].to_list()
    )
    assert flagged == carriers


def test_taxon_holdout_empty_label_does_not_break_cv(taxon_dataset):
    """A label with no usable samples must leave an empty holdout behind,
    not a missing one, or create_cv_folds(use_holdout=True) raises and the
    whole run dies because of one unusable label (e.g. oxygen in soil)."""
    # add an all-null label alongside the usable one
    labels = taxon_dataset.labels.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("oxy")
    )
    taxon_dataset.labels = labels

    taxon_dataset.create_taxon_holdout_split(
        rank="family", test_size=0.2, min_test_samples=1
    )

    # the unusable label is registered with a zero-row holdout ...
    assert "oxy" in taxon_dataset.splits
    assert taxon_dataset.splits["oxy"].holdout is not None
    assert taxon_dataset.splits["oxy"].holdout.height == 0
    # ... while the usable one is split normally
    assert taxon_dataset.splits["target"].holdout.height == 4

    # and CV over all labels no longer raises
    taxon_dataset.create_cv_folds(
        n_folds=2, use_holdout=True, force=True, strict=False
    )
    assert taxon_dataset.splits["target"].cv_schemes


def test_sweep_pairs_tokens_and_dir_names():
    """sweep.pairs maps label -> grouping; each pair gets its own run name
    so the summary tables can tell the runs apart."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import sweep_tokens as st

    pairs = {"geo_ph": "organism", "ph": "taxon:family", "oxy": None}
    names = {}
    for label, token in pairs.items():
        token = "random" if token is None else token
        names[label] = f"holdout_{st.spec_dir_name(token)}__{label}"

    assert names["geo_ph"] == "holdout_organism__geo_ph"
    assert names["ph"] == "holdout_taxon_family__ph"
    assert names["oxy"] == "holdout_random__oxy"
    # every run directory still matches the holdout_* glob the collation
    # and fold profile rely on
    assert all(n.startswith("holdout_") for n in names.values())


def test_all_null_numeric_label_does_not_crash_binning(splitting_dataset):
    """An all-null Float64 label used to raise in the quantile binning
    (min/max of nothing). Both split paths must degrade to empty."""
    dataset = splitting_dataset
    dataset.labels = dataset.labels.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("all_null")
    )

    dataset.create_holdout_split(test_size=0.2, random_state=42, force=True)
    assert dataset.splits["all_null"].holdout is not None
    assert dataset.splits["all_null"].holdout.height == 0

    dataset.create_cv_folds(
        n_folds=2, use_holdout=True, force=True, strict=False
    )
    assert not dataset.splits["all_null"].cv_schemes
    # the usable label is unaffected
    assert dataset.splits["target_cont"].cv_schemes


def test_sweep_pairs_list_expands_to_one_run_each():
    """A label may name several groupings; each becomes its own run, and
    two labels may share a grouping without colliding."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import sweep_tokens as st

    pairs = {
        "temperature": ["bioproject", "organism"],
        "ph": "bioproject",
        "oxy": None,
    }
    runs = []
    for label, tokens in pairs.items():
        if tokens is None or isinstance(tokens, str):
            tokens = [tokens]
        for token in tokens:
            token = "random" if token is None else token
            runs.append(f"holdout_{st.spec_dir_name(token)}__{label}")

    assert runs == [
        "holdout_bioproject__temperature",
        "holdout_organism__temperature",
        "holdout_bioproject__ph",
        "holdout_random__oxy",
    ]
    # distinct directories, so adding a grouping later never overwrites
    assert len(set(runs)) == len(runs)
