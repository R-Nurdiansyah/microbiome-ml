"""Tests for Dataset class."""

import warnings

import polars as pl
import pytest

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles


class TestDatasetInitialization:
    """Test Dataset initialization."""

    def test_empty_initialization(self):
        """Test creating empty Dataset."""
        dataset = Dataset()

        assert dataset.metadata is None
        assert dataset.profiles is None
        assert dataset.feature_sets == {}
        assert dataset.labels is None
        assert dataset.groupings is None
        assert dataset._sample_ids is None

    def test_initialization_with_metadata(self, sample_metadata):
        """Test initialization with metadata (eager and lazy)."""
        dataset = Dataset(metadata=sample_metadata)

        assert dataset.metadata is not None
        assert dataset._sample_ids is not None
        assert len(dataset._sample_ids) == 4

    def test_initialization_with_all_components(
        self, sample_metadata, sample_profiles
    ):
        """Test initialization with multiple components (eager and lazy)."""
        dataset = Dataset(metadata=sample_metadata, profiles=sample_profiles)

        assert dataset.metadata is not None
        assert dataset.profiles is not None
        assert dataset._sample_ids is not None


class TestDatasetBuilderPattern:
    """Test builder pattern methods."""

    def test_add_metadata(
        self, sample_dataset_empty, metadata_csv, attributes_csv
    ):
        """Test adding metadata via builder."""
        result = sample_dataset_empty.add_metadata(
            metadata=metadata_csv, attributes=attributes_csv
        )

        assert result is sample_dataset_empty  # Chaining works
        assert sample_dataset_empty.metadata is not None

    def test_add_profiles(self, sample_dataset_empty, profiles_csv, root_csv):
        """Test adding profiles via builder."""
        result = sample_dataset_empty.add_profiles(
            profiles_csv, root=root_csv, check_filled=False
        )

        assert result is sample_dataset_empty
        assert sample_dataset_empty.profiles is not None

    def test_add_features_from_rank(self, sample_dataset):
        """Test adding features from taxonomic rank."""
        result = sample_dataset.add_features(
            "genus_features", rank=TaxonomicRanks.GENUS
        )

        assert result is sample_dataset
        assert "genus_features" in sample_dataset.feature_sets

    def test_add_features_from_instance(
        self, sample_dataset_empty, sample_features
    ):
        """Test adding existing FeatureSet instance."""
        result = sample_dataset_empty.add_features(
            "my_features", features=sample_features
        )

        assert result is sample_dataset_empty
        assert "my_features" in sample_dataset_empty.feature_sets

    def test_method_chaining(self, metadata_csv, attributes_csv, profiles_csv):
        """Test chaining multiple builder methods."""
        dataset = (
            Dataset()
            .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
            .add_profiles(profiles_csv, check_filled=False)
        )

        assert dataset.metadata is not None
        assert dataset.profiles is not None


class TestDatasetBatchOperations:
    """Test batch addition methods."""

    def test_add_feature_set_dict(
        self, sample_dataset_empty, features_csv, tmp_path
    ):
        """Test adding multiple feature sets at once."""
        # Create second features file
        features_csv2 = tmp_path / "features2.csv"
        pl.DataFrame(
            {
                "sample": ["S1", "S2", "S3"],
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [4.0, 5.0, 6.0],
            }
        ).write_csv(features_csv2)

        result = sample_dataset_empty.add_feature_set(
            {"set1": features_csv, "set2": features_csv2}
        )

        assert result is sample_dataset_empty
        assert "set1" in sample_dataset_empty.feature_sets
        assert "set2" in sample_dataset_empty.feature_sets

    def test_add_feature_set_single_requires_name(
        self, sample_dataset_empty, features_csv
    ):
        """Test single feature set addition requires name."""
        with pytest.raises(ValueError, match="name"):
            sample_dataset_empty.add_feature_set(features_csv)

    def test_add_labels_dict(self, sample_dataset_empty, tmp_path):
        """Test adding multiple label sets at once."""
        labels_csv1 = tmp_path / "labels1.csv"
        pl.DataFrame(
            {"sample": ["S1", "S2", "S3"], "val1": [1, 0, 1]}
        ).write_csv(labels_csv1)

        labels_csv2 = tmp_path / "labels2.csv"
        pl.DataFrame(
            {"sample": ["S1", "S2", "S3"], "val2": [1, 0, 1]}
        ).write_csv(labels_csv2)

        result = sample_dataset_empty.add_labels(
            {"labels1": labels_csv1, "labels2": labels_csv2}
        )

        assert result is sample_dataset_empty
        assert "labels1" in sample_dataset_empty.labels.columns
        assert "labels2" in sample_dataset_empty.labels.columns

    def test_add_labels_single_no_name(self, sample_dataset_empty, labels_csv):
        """Test single label addition without name."""
        sample_dataset_empty.add_labels(labels_csv)
        assert "target" in sample_dataset_empty.labels.columns
        assert "category" in sample_dataset_empty.labels.columns


class TestDatasetTaxonomicFeatures:
    """Test taxonomic feature generation."""

    def test_add_taxonomic_features_all_ranks(self, sample_dataset):
        """Test generating features for all standard ranks."""
        result = sample_dataset.add_taxonomic_features()

        assert result is sample_dataset
        # Should have 6 standard ranks (phyla to species)
        assert (
            len(
                [
                    k
                    for k in sample_dataset.feature_sets.keys()
                    if k.startswith("tax_")
                ]
            )
            == 6
        )

    @pytest.mark.parametrize(
        "ranks",
        [
            [TaxonomicRanks.GENUS, TaxonomicRanks.PHYLUM],
            [TaxonomicRanks.SPECIES],
        ],
    )
    def test_add_taxonomic_features_specific_ranks(
        self, sample_dataset, ranks
    ):
        """Test generating features for specific ranks."""
        sample_dataset.add_taxonomic_features(ranks=ranks)

        for rank in ranks:
            assert f"tax_{rank.name}" in sample_dataset.feature_sets

    def test_add_taxonomic_features_custom_prefix(self, sample_dataset):
        """Test custom prefix for taxonomic features."""
        sample_dataset.add_taxonomic_features(
            ranks=[TaxonomicRanks.GENUS], prefix="custom"
        )

        assert "custom_genus" in sample_dataset.feature_sets

    def test_add_taxonomic_features_requires_profiles(
        self, sample_dataset_empty
    ):
        """Test that profiles are required for taxonomic features."""
        with pytest.raises(ValueError, match="Profiles must be added"):
            sample_dataset_empty.add_taxonomic_features()


class TestDatasetAccessionSync:
    """Test accession synchronization."""

    def test_sync_finds_intersection(
        self, metadata_csv, attributes_csv, profiles_csv, tmp_path
    ):
        """Test sync computes strict intersection."""
        # Create metadata with S1, S2, S3, S4
        # Create profiles with only S1, S2
        profiles_subset = tmp_path / "profiles_subset.csv"
        pl.DataFrame(
            {
                "sample": ["S1", "S1", "S2", "S2"],
                "taxonomy": ["d__Bacteria", "d__Bacteria;p__Proteobacteria"]
                * 2,
                "coverage": [100.0, 50.0, 120.0, 60.0],
            }
        ).write_csv(profiles_subset)

        dataset = (
            Dataset()
            .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
            .add_profiles(profiles_subset, check_filled=False)
        )

        # Should only have S1, S2 (intersection)
        assert set(dataset._sample_ids) == {"S1", "S2"}

    def test_sync_warning_on_large_drop(
        self, metadata_csv, attributes_csv, tmp_path
    ):
        """Test warning is issued when >10% samples dropped."""
        # Create profiles with only 1 sample (will drop 75% of metadata samples)
        profiles_single = tmp_path / "profiles_single.csv"
        pl.DataFrame(
            {
                "sample": ["S1", "S1"],
                "taxonomy": ["d__Bacteria", "d__Bacteria;p__Proteobacteria"],
                "coverage": [100.0, 50.0],
            }
        ).write_csv(profiles_single)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (
                Dataset()
                .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                .add_profiles(profiles_single, check_filled=False)
            )

            # Should have issued warning
            assert len(w) > 0
            assert any(
                "removed" in str(warning.message).lower() for warning in w
            )

    def test_get_sample_ids(self, sample_dataset):
        """Test getting canonical sample IDs."""
        sample_ids = sample_dataset.get_sample_ids()

        assert isinstance(sample_ids, list)
        assert len(sample_ids) > 0
        # Should be sorted
        assert sample_ids == sorted(sample_ids)


class TestDatasetPreprocessing:
    """Test preprocessing pipeline."""

    def test_apply_preprocessing_default(self, sample_dataset):
        """Test default preprocessing (eager and lazy)."""
        result = sample_dataset.apply_preprocessing()

        assert result is sample_dataset

    def test_apply_preprocessing_no_sync(self, sample_dataset):
        """Test preprocessing without sync."""
        result = sample_dataset.apply_preprocessing(sync_after=False)

        assert result is sample_dataset

    def test_apply_preprocessing_selective(self, sample_dataset):
        """Test selective preprocessing."""
        result = sample_dataset.apply_preprocessing(
            metadata_qc=True, profiles_qc=False, sync_after=True
        )

        assert result is sample_dataset

    def test_apply_preprocessing_metadata_qc_flag(
        self, low_quality_metadata_csv, low_quality_attributes_csv
    ):
        """Test metadata QC flag is set after preprocessing."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        dataset = Dataset(metadata=meta)

        assert dataset.metadata_qc_done is False

        dataset.apply_preprocessing(
            metadata_qc=True, profiles_qc=False, sync_after=False
        )

        assert dataset.metadata_qc_done is True

    def test_apply_preprocessing_profiles_qc_flag(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test profiles QC flag is set after preprocessing."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )
        dataset = Dataset(profiles=profiles)

        assert dataset.profiles_qc_done is False

        dataset.apply_preprocessing(
            metadata_qc=False, profiles_qc=True, sync_after=False
        )

        assert dataset.profiles_qc_done is True

    def test_apply_preprocessing_custom_metadata_params(
        self, low_quality_metadata_csv, low_quality_attributes_csv
    ):
        """Test custom metadata QC parameters are applied."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        dataset = Dataset(metadata=meta)

        # Before: 7 samples
        assert dataset.metadata.metadata.collect().height == 7

        # Apply with custom cutoff of 900
        dataset.apply_preprocessing(
            metadata_qc=True,
            metadata_mbp_cutoff=900,
            profiles_qc=False,
            sync_after=False,
        )

        # After: 5 samples
        assert dataset.metadata.metadata.collect().height == 5

    def test_apply_preprocessing_custom_profiles_params(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test custom profiles QC parameters are applied."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )
        dataset = Dataset(profiles=profiles)

        # Before: 4 samples
        initial_samples = len(
            set(
                dataset.profiles.profiles.select("sample")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
        )
        assert initial_samples == 4

        # Apply with very lenient domination cutoff
        dataset.apply_preprocessing(
            metadata_qc=False,
            profiles_qc=True,
            profiles_dominated_cutoff=1.0,  # Keep everything
            sync_after=False,
        )

        # After: all 4 samples remain (only coverage filter applied)
        final_samples = len(
            set(
                dataset.profiles.profiles.select("sample")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
        )
        assert final_samples == 4

    def test_apply_preprocessing_skips_when_flag_set(self, sample_dataset):
        """Test QC is skipped when flag is already True."""
        # Manually set flags
        sample_dataset.metadata_qc_done = True
        sample_dataset.profiles_qc_done = True

        initial_meta_height = sample_dataset.metadata.metadata.collect().height
        initial_profiles_samples = len(
            set(
                sample_dataset.profiles.profiles.select("sample")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
        )

        # Apply preprocessing - should be skipped
        sample_dataset.apply_preprocessing()

        # Verify no changes
        assert (
            sample_dataset.metadata.metadata.collect().height
            == initial_meta_height
        )
        assert (
            len(
                set(
                    sample_dataset.profiles.profiles.select("sample")
                    .unique()
                    .collect()
                    .to_series()
                    .to_list()
                )
            )
            == initial_profiles_samples
        )

    def test_apply_preprocessing_synchronizes_after_qc(
        self,
        low_quality_metadata_csv,
        low_quality_attributes_csv,
        dominated_profiles_csv,
        dominated_root_csv,
    ):
        """Test synchronization happens after QC."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Create dataset - automatically syncs on init
        dataset = Dataset(metadata=meta, profiles=profiles)

        # After init sync, should have intersection: DOM1, DOM2, HQ1, HQ2
        initial_meta_samples = set(
            dataset.metadata.metadata.select("sample")
            .collect()
            .to_series()
            .to_list()
        )
        initial_profile_samples = set(
            dataset.profiles.profiles.select("sample")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        assert initial_meta_samples == initial_profile_samples
        assert initial_meta_samples == {"DOM1", "DOM2", "HQ1", "HQ2"}

        # Apply preprocessing - DOM1 and DOM2 are dominated and should be filtered
        dataset.apply_preprocessing()

        # After QC, only high quality samples remain
        final_meta_samples = set(
            dataset.metadata.metadata.select("sample")
            .collect()
            .to_series()
            .to_list()
        )
        final_profile_samples = set(
            dataset.profiles.profiles.select("sample")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        assert final_meta_samples == final_profile_samples
        assert final_meta_samples == {"HQ1", "HQ2"}

    def test_apply_preprocessing_metadata_only(
        self, low_quality_metadata_csv, low_quality_attributes_csv
    ):
        """Test preprocessing with only metadata component."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        dataset = Dataset(metadata=meta)

        # Should not crash when profiles is None
        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True)

        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is False  # No profiles to process

    def test_apply_preprocessing_profiles_only(
        self, dominated_profiles_csv, dominated_root_csv
    ):
        """Test preprocessing with only profiles component."""
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )
        dataset = Dataset(profiles=profiles)

        # Should not crash when metadata is None
        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True)

        assert dataset.metadata_qc_done is False  # No metadata to process
        assert dataset.profiles_qc_done is True

    def test_apply_preprocessing_both_components(
        self,
        low_quality_metadata_csv,
        low_quality_attributes_csv,
        dominated_profiles_csv,
        dominated_root_csv,
    ):
        """Test preprocessing with both components."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )
        dataset = Dataset(metadata=meta, profiles=profiles)

        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True)

        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is True


class TestDatasetPreprocessingEdgeCases:
    """Test preprocessing edge cases and transformations."""

    def test_preprocessing_creates_root_if_missing(self, profiles_csv):
        """Test preprocessing creates root if not provided."""
        # Load profiles without root
        profiles = TaxonomicProfiles(
            profiles_csv, root=None, check_filled=False
        )
        dataset = Dataset(profiles=profiles)

        # Root should exist after __init__ (created automatically)
        assert (
            dataset.profiles.root is not None
            or dataset.profiles._lf_root is not None
        )

        # Preprocessing should work fine
        dataset.apply_preprocessing(profiles_qc=True, sync_after=False)

        assert dataset.profiles_qc_done is True

    def test_preprocessing_fills_unfilled_profiles(self, tmp_path):
        """Test preprocessing fills profiles if not in filled format."""
        # Create unfilled profiles (just species level)
        unfilled_data = {
            "sample": ["S1", "S2", "S3"] * 2,
            "taxonomy": [
                "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia_coli",
                "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__Lactobacillus_acidophilus",
                "d__Bacteria;p__Actinobacteria;c__Actinobacteria;o__Actinomycetales;f__Actinomycetaceae;g__Actinomyces;s__Actinomyces_odontolyticus",
            ]
            * 2,
            "relabund": [0.4, 0.35, 0.25] * 2,
        }
        unfilled_csv = tmp_path / "unfilled_profiles.csv"
        pl.DataFrame(unfilled_data).write_csv(unfilled_csv)

        root_data = {
            "sample": ["S1", "S2", "S3"],
            "root_coverage": [100.0, 120.0, 110.0],
        }
        root_csv = tmp_path / "unfilled_root.csv"
        pl.DataFrame(root_data).write_csv(root_csv)

        # Load without filling
        profiles = TaxonomicProfiles(
            str(unfilled_csv), root=str(root_csv), check_filled=False
        )
        profiles.is_filled = False  # Manually mark as unfilled
        dataset = Dataset(profiles=profiles)

        assert dataset.profiles.is_filled is False

        # Preprocessing should fill
        dataset.apply_preprocessing(profiles_qc=True, sync_after=False)

        assert dataset.profiles.is_filled is True

    def test_preprocessing_converts_coverage_to_relabund(self, tmp_path):
        """Test preprocessing converts coverage to relabund format."""
        # Create coverage profiles
        coverage_data = {
            "sample": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "taxonomy": [
                "d__Bacteria",
                "d__Bacteria;p__Proteobacteria",
                "d__Bacteria;p__Firmicutes",
            ]
            * 2,
            "coverage": [100.0, 60.0, 40.0, 120.0, 70.0, 50.0],
        }
        coverage_csv = tmp_path / "coverage_profiles.csv"
        pl.DataFrame(coverage_data).write_csv(coverage_csv)

        root_data = {"sample": ["S1", "S2"], "root_coverage": [100.0, 120.0]}
        root_csv = tmp_path / "coverage_root.csv"
        pl.DataFrame(root_data).write_csv(root_csv)

        profiles = TaxonomicProfiles(
            str(coverage_csv), root=str(root_csv), check_filled=False
        )
        dataset = Dataset(profiles=profiles)

        # it auto converts during init
        schema_names = dataset.profiles.profiles.collect_schema().names()
        assert "relabund" in schema_names

        # test the values are correct
        # test the values are correct
        relabund_df = dataset.profiles.profiles.select(
            ["sample", "taxonomy", "relabund"]
        ).collect()
        expected_relabund = {
            ("S1", "d__Bacteria"): 100.0 / 100.0,
            ("S1", "d__Bacteria;p__Proteobacteria"): 60.0 / 100.0,
            ("S1", "d__Bacteria;p__Firmicutes"): 40.0 / 100.0,
            ("S2", "d__Bacteria"): 120.0 / 120.0,
            ("S2", "d__Bacteria;p__Proteobacteria"): 70.0 / 120.0,
            ("S2", "d__Bacteria;p__Firmicutes"): 50.0 / 120.0,
        }

        # Convert to list of dictionaries for easier iteration
        for row in relabund_df.to_dicts():
            key = (row["sample"], row["taxonomy"])
            assert pytest.approx(row["relabund"]) == expected_relabund[key]

    def test_preprocessing_idempotency(self, sample_dataset_eager):
        """Test repeated preprocessing calls are idempotent via flags."""
        # First preprocessing
        sample_dataset_eager.apply_preprocessing()

        first_meta_height = (
            sample_dataset_eager.metadata.metadata.collect().height
        )
        first_profile_samples = set(
            sample_dataset_eager.profiles.profiles.collect()["sample"]
            .unique()
            .to_list()
        )

        # Second preprocessing - should be skipped due to flags
        sample_dataset_eager.apply_preprocessing()

        second_meta_height = (
            sample_dataset_eager.metadata.metadata.collect().height
        )
        second_profile_samples = set(
            sample_dataset_eager.profiles.profiles.collect()["sample"]
            .unique()
            .to_list()
        )

        # No changes from second call
        assert first_meta_height == second_meta_height
        assert first_profile_samples == second_profile_samples

    def test_preprocessing_custom_thresholds_filter_correctly(
        self, low_quality_metadata_csv, low_quality_attributes_csv
    ):
        """Test custom thresholds actually filter the expected samples."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        dataset = Dataset(metadata=meta)

        # Cutoff of 800 should keep LQ2 (800), LQ3 (950), HQ1 (1500), HQ2 (2000)
        dataset.apply_preprocessing(
            metadata_qc=True, metadata_mbp_cutoff=800, sync_after=False
        )

        final_samples = set(
            dataset.metadata.metadata.select("sample")
            .collect()
            .to_series()
            .to_list()
        )
        assert final_samples == {"DOM1", "DOM2", "LQ3", "HQ1", "HQ2"}
        assert "LQ1" not in final_samples  # 500 mbases, filtered out


class TestDatasetSynchronizationWarnings:
    """Test warning messages during sample synchronization."""

    def test_sync_warns_when_samples_dropped(
        self,
        low_quality_metadata_csv,
        low_quality_attributes_csv,
        dominated_profiles_csv,
        dominated_root_csv,
    ):
        """Test warning is issued when synchronization drops samples."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )

        # Different sample sets - synchronization will drop samples
        # Should issue warning during sync (which happens in __init__)
        with pytest.warns(UserWarning, match="Accession sync removed"):
            Dataset(metadata=meta, profiles=profiles)

    def test_preprocessing_warns_on_sync(
        self,
        low_quality_metadata_csv,
        low_quality_attributes_csv,
        dominated_profiles_csv,
        dominated_root_csv,
    ):
        """Test preprocessing issues warnings when QC drops samples during
        sync."""
        meta = SampleMetadata(
            low_quality_metadata_csv, low_quality_attributes_csv
        )
        profiles = TaxonomicProfiles(
            dominated_profiles_csv, root=dominated_root_csv, check_filled=False
        )
        dataset = Dataset(metadata=meta, profiles=profiles)

        # QC will filter different samples from each component, sync will drop some
        with pytest.warns(UserWarning):
            dataset.apply_preprocessing(
                metadata_qc=True, profiles_qc=True, sync_after=True
            )


class TestDatasetLargeScaleIntegration:
    """Test with large datasets (100 samples) for realistic QC pipeline."""

    def test_large_dataset_qc_pipeline(
        self,
        tmp_path,
        large_metadata_data,
        large_profiles_data,
        large_root_data,
    ):
        """Test full QC pipeline with 100 samples."""
        # Create CSV files
        metadata_csv = tmp_path / "large_metadata.csv"
        pl.DataFrame(large_metadata_data).write_csv(metadata_csv)

        attributes_data = {
            "sample": large_metadata_data["sample"] * 2,
            "key": ["pH", "temp"] * len(large_metadata_data["sample"]),
            "value": [
                str(i % 10)
                for i in range(len(large_metadata_data["sample"]) * 2)
            ],
        }
        attributes_csv = tmp_path / "large_attributes.csv"
        pl.DataFrame(attributes_data).write_csv(attributes_csv)

        profiles_csv = tmp_path / "large_profiles.csv"
        pl.DataFrame(large_profiles_data).write_csv(profiles_csv)

        root_csv = tmp_path / "large_root.csv"
        pl.DataFrame(large_root_data).write_csv(root_csv)

        # Create dataset
        meta = SampleMetadata(str(metadata_csv), str(attributes_csv))
        profiles = TaxonomicProfiles(
            str(profiles_csv), root=str(root_csv), check_filled=False
        )
        dataset = Dataset(metadata=meta, profiles=profiles)

        # Initial: 100 samples
        assert dataset.metadata.metadata.collect().height == 100
        initial_profile_samples = len(
            set(
                dataset.profiles.profiles.select("sample")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
        )
        assert initial_profile_samples == 100

        # Apply preprocessing with default thresholds
        dataset.apply_preprocessing(
            metadata_qc=True,
            profiles_qc=True,
            sync_after=True,
            metadata_mbp_cutoff=1000,
            profiles_cov_cutoff=50.0,
            profiles_dominated_cutoff=0.99,
        )

        # After QC:
        # - Metadata: ~70 samples (30 below 1000 mbases)
        # - Profiles coverage: ~75 samples (25 below 50 coverage)
        # - Profiles domination: ~80 samples (20 dominated)
        # - Intersection of all filters should leave ~50-60 samples

        final_meta_samples = dataset.metadata.metadata.collect().height
        final_profile_samples = len(
            set(
                dataset.profiles.profiles.select("sample")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
        )

        # Verify filtering happened
        assert final_meta_samples < 100
        assert final_profile_samples < 100

        # Verify synchronization (same samples in both)
        assert final_meta_samples == final_profile_samples

        # Verify flags are set
        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is True

    def test_large_dataset_custom_thresholds(
        self,
        tmp_path,
        large_metadata_data,
        large_profiles_data,
        large_root_data,
    ):
        """Test custom thresholds with large dataset."""
        # Create CSV files
        metadata_csv = tmp_path / "large_metadata2.csv"
        pl.DataFrame(large_metadata_data).write_csv(metadata_csv)

        attributes_data = {
            "sample": large_metadata_data["sample"] * 2,
            "key": ["pH", "temp"] * len(large_metadata_data["sample"]),
            "value": [
                str(i % 10)
                for i in range(len(large_metadata_data["sample"]) * 2)
            ],
        }
        attributes_csv = tmp_path / "large_attributes2.csv"
        pl.DataFrame(attributes_data).write_csv(attributes_csv)

        profiles_csv = tmp_path / "large_profiles2.csv"
        pl.DataFrame(large_profiles_data).write_csv(profiles_csv)

        root_csv = tmp_path / "large_root2.csv"
        pl.DataFrame(large_root_data).write_csv(root_csv)

        meta = SampleMetadata(str(metadata_csv), str(attributes_csv))
        profiles = TaxonomicProfiles(
            str(profiles_csv), root=str(root_csv), check_filled=False
        )
        dataset = Dataset(metadata=meta, profiles=profiles)

        # Very lenient thresholds - should keep most samples
        dataset.apply_preprocessing(
            metadata_qc=True,
            profiles_qc=True,
            sync_after=True,
            metadata_mbp_cutoff=500,
            profiles_cov_cutoff=20.0,
            profiles_dominated_cutoff=1.0,  # Don't filter dominated
        )

        final_samples = dataset.metadata.metadata.collect().height

        # Should keep > 80 samples with lenient thresholds
        assert final_samples > 80

    def test_large_dataset_lazy_mode(
        self,
        tmp_path,
        large_metadata_data,
        large_profiles_data,
        large_root_data,
    ):
        """Test large dataset QC in lazy mode."""
        # Create CSV files
        metadata_csv = tmp_path / "large_metadata_lazy.csv"
        pl.DataFrame(large_metadata_data).write_csv(metadata_csv)

        attributes_csv = tmp_path / "large_attributes_lazy.csv"
        attributes_data = {
            "sample": large_metadata_data["sample"] * 2,
            "key": ["pH", "temp"] * len(large_metadata_data["sample"]),
            "value": [
                str(i % 10)
                for i in range(len(large_metadata_data["sample"]) * 2)
            ],
        }
        pl.DataFrame(attributes_data).write_csv(attributes_csv)

        profiles_csv = tmp_path / "large_profiles_lazy.csv"
        pl.DataFrame(large_profiles_data).write_csv(profiles_csv)

        root_csv = tmp_path / "large_root_lazy.csv"
        pl.DataFrame(large_root_data).write_csv(root_csv)

        # Load in lazy mode
        meta = SampleMetadata.scan(str(metadata_csv), str(attributes_csv))
        profiles = TaxonomicProfiles.scan(
            str(profiles_csv), root=str(root_csv), check_filled=False
        )
        dataset = Dataset(metadata=meta, profiles=profiles)

        # Apply preprocessing - should maintain lazy mode through pipeline
        dataset.apply_preprocessing()

        # Flags should be set
        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is True


class TestDatasetIteration:
    """Test iteration methods."""

    def test_iter_feature_sets(self, sample_dataset_eager):
        """Test iterating over feature sets."""
        sample_dataset_eager.add_features(
            "test_features", rank=TaxonomicRanks.GENUS
        )

        feature_sets = list(sample_dataset_eager.iter_feature_sets())

        assert len(feature_sets) > 0
        for name, fs in feature_sets:
            assert isinstance(name, str)
            assert isinstance(fs, FeatureSet)

    def test_iter_feature_sets_filtered(self, sample_dataset_eager):
        """Test iterating over specific feature sets."""
        sample_dataset_eager.add_features("fs1", rank=TaxonomicRanks.GENUS)
        sample_dataset_eager.add_features("fs2", rank=TaxonomicRanks.PHYLUM)

        feature_sets = list(
            sample_dataset_eager.iter_feature_sets(names=["fs1"])
        )

        assert len(feature_sets) == 1
        assert feature_sets[0][0] == "fs1"

    def test_iter_labels(self, sample_dataset_empty, labels_csv):
        """Test iterating over labels."""
        sample_dataset_empty.add_labels(labels_csv, name="test_labels")

        labels = list(sample_dataset_empty.iter_labels())

        # labels_csv has 'target' and 'category' columns
        assert len(labels) == 2
        names = sorted([label[0] for label in labels])
        assert names == ["category", "target"]


class TestDatasetPersistence:
    """Test save/load/scan with directory and tar.gz."""

    def test_save_to_directory(self, sample_dataset_eager, tmp_path):
        """Test saving to directory."""
        save_dir = tmp_path / "dataset_save"
        sample_dataset_eager.save(save_dir)

        assert save_dir.exists()
        assert (save_dir / "manifest.json").exists()
        assert (save_dir / "metadata").exists()
        assert (save_dir / "profiles").exists()

    def test_save_with_compression(self, sample_dataset_eager, tmp_path):
        """Test saving with tar.gz compression."""
        save_path = tmp_path / "dataset.tar.gz"
        sample_dataset_eager.save(save_path, compress=True)

        assert save_path.exists()
        assert str(save_path).endswith(".tar.gz")

    def test_load_from_directory(self, sample_dataset_eager, tmp_path):
        """Test loading from directory."""
        save_dir = tmp_path / "dataset_load"
        sample_dataset_eager.save(save_dir)

        loaded = Dataset.load(save_dir, lazy=False)

        assert loaded.metadata is not None
        assert loaded.profiles is not None
        assert loaded._sample_ids is not None

    def test_load_from_tarfile(self, sample_dataset_eager, tmp_path):
        """Test loading from tar.gz file."""
        save_path = tmp_path / "dataset_load.tar.gz"
        sample_dataset_eager.save(save_path, compress=True)

        loaded = Dataset.load(save_path, lazy=False)

        assert loaded.metadata is not None
        assert loaded.profiles is not None

    def test_load_lazy(self, sample_dataset_eager, tmp_path):
        """Test loading in lazy mode."""
        save_dir = tmp_path / "dataset_lazy"
        sample_dataset_eager.save(save_dir)

        loaded = Dataset.load(save_dir, lazy=True)

        assert isinstance(loaded.metadata.metadata, pl.LazyFrame)
        assert isinstance(loaded.profiles.profiles, pl.LazyFrame)

    def test_scan_alias(self, sample_dataset_eager, tmp_path):
        """Test scan is alias for load(lazy=True)."""
        save_dir = tmp_path / "dataset_scan"
        sample_dataset_eager.save(save_dir)

        scanned = Dataset.scan(save_dir)

        assert isinstance(scanned.metadata.metadata, pl.LazyFrame)
        assert isinstance(scanned.profiles.profiles, pl.LazyFrame)

    def test_manifest_content(self, sample_dataset_eager, tmp_path):
        """Test manifest.json contains expected metadata."""
        import json

        save_dir = tmp_path / "dataset_manifest"
        sample_dataset_eager.save(save_dir)

        with open(save_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        assert "version" in manifest
        assert "created" in manifest
        assert "components" in manifest
        assert "sample_ids" in manifest
        assert manifest["version"] == "1.0"

    def test_roundtrip_preserves_data(self, sample_dataset_eager, tmp_path):
        """Test save/load preserves data integrity."""
        save_dir = tmp_path / "dataset_roundtrip"

        # Get original sample count
        original_samples = set(sample_dataset_eager._sample_ids)

        # Save and load
        sample_dataset_eager.save(save_dir)
        loaded = Dataset.load(save_dir, lazy=False)

        # Check sample IDs preserved
        loaded_samples = set(loaded._sample_ids)
        assert original_samples == loaded_samples

    def test_save_with_features_and_labels(
        self, sample_dataset_eager, labels_csv, tmp_path
    ):
        """Test saving dataset with features and labels."""
        # Add features and labels
        sample_dataset_eager.add_features(
            "test_features", rank=TaxonomicRanks.GENUS
        )
        sample_dataset_eager.add_labels(labels_csv, name="test_labels")

        save_dir = tmp_path / "dataset_full"
        sample_dataset_eager.save(save_dir)

        assert (save_dir / "features").exists()
        assert (save_dir / "labels").exists()

        # Load and verify
        loaded = Dataset.load(save_dir, lazy=False)
        assert "test_features" in loaded.feature_sets
        # labels_csv has 'target' and 'category' columns
        assert "target" in loaded.labels.columns
        assert "category" in loaded.labels.columns

    def test_save_load_labels_groupings(self, sample_dataset_eager, tmp_path):
        """Test saving and loading labels and groupings."""
        # Add labels and groupings
        labels_df = pl.DataFrame({"sample": ["S1", "S2"], "target": [0, 1]})
        groupings_df = pl.DataFrame(
            {"sample": ["S1", "S2"], "group": ["A", "B"]}
        )

        sample_dataset_eager.add_labels(labels_df)
        sample_dataset_eager.add_groupings(groupings_df)

        save_dir = tmp_path / "dataset_full"
        sample_dataset_eager.save(save_dir)

        # Check files
        assert (save_dir / "labels" / "labels.csv").exists()
        assert (save_dir / "groupings" / "groupings.csv").exists()

        # Load
        loaded = Dataset.load(save_dir, lazy=False)

        assert loaded.labels is not None
        assert "target" in loaded.labels.columns
        assert loaded.groupings is not None
        assert "group" in loaded.groupings.columns


class TestDatasetDefaultGroupings:
    """Test create_default_groupings method."""

    def test_create_default_groupings_basic(self, sample_metadata):
        """Test creating default groupings from metadata."""
        dataset = Dataset(metadata=sample_metadata)

        # Create default groupings
        dataset.create_default_groupings()

        assert dataset.groupings is not None
        assert "sample" in dataset.groupings.columns
        assert "bioproject" in dataset.groupings.columns
        assert "biome" in dataset.groupings.columns

        # Check that data is extracted correctly
        groupings_collected = (
            dataset.groupings.collect()
            if hasattr(dataset.groupings, "collect")
            else dataset.groupings
        )
        assert groupings_collected.height == 4  # Should have all samples

    def test_create_default_groupings_specific_fields(self, sample_metadata):
        """Test creating only specific groupings."""
        dataset = Dataset(metadata=sample_metadata)

        dataset.create_default_groupings(groupings=["bioproject", "biome"])

        assert dataset.groupings is not None
        assert "bioproject" in dataset.groupings.columns
        assert "biome" in dataset.groupings.columns
        # Should not have other fields
        assert (
            "domain" not in dataset.groupings.columns
            or dataset.groupings.select("domain").null_count().item() > 0
        )

    def test_create_default_groupings_missing_fields(self, sample_metadata):
        """Test that missing fields are skipped gracefully."""
        dataset = Dataset(metadata=sample_metadata)

        # Request fields that don't exist in our test metadata
        dataset.create_default_groupings(
            groupings=["bioproject", "biome", "ecoregion", "climate"]
        )

        assert dataset.groupings is not None
        # Should have the fields that exist
        assert "bioproject" in dataset.groupings.columns
        assert "biome" in dataset.groupings.columns

    def test_create_default_groupings_no_metadata(self):
        """Test that it fails gracefully without metadata."""
        dataset = Dataset()

        with pytest.raises(ValueError, match="Metadata must be added"):
            dataset.create_default_groupings()

    def test_create_default_groupings_merge_with_existing(
        self, sample_metadata
    ):
        """Test merging with existing groupings."""
        dataset = Dataset(metadata=sample_metadata)

        # Add custom groupings first
        custom_groupings = pl.DataFrame(
            {
                "sample": ["S1", "S2", "S3", "S4"],
                "custom_group": ["A", "A", "B", "B"],
            }
        )
        dataset.add_groupings(custom_groupings)

        # Now create default groupings (should merge)
        dataset.create_default_groupings(groupings=["bioproject"])

        assert dataset.groupings is not None
        # Should have both custom and default groupings
        assert "custom_group" in dataset.groupings.columns
        assert "bioproject" in dataset.groupings.columns

    def test_create_default_groupings_force_overwrite(self, sample_metadata):
        """Test force overwrite of existing groupings."""
        dataset = Dataset(metadata=sample_metadata)

        # Add custom groupings first
        custom_groupings = pl.DataFrame(
            {
                "sample": ["S1", "S2", "S3", "S4"],
                "custom_group": ["A", "A", "B", "B"],
            }
        )
        dataset.add_groupings(custom_groupings)

        # Create default groupings with force=True (should replace)
        dataset.create_default_groupings(groupings=["bioproject"], force=True)

        assert dataset.groupings is not None
        assert "bioproject" in dataset.groupings.columns
        # Custom groupings should be gone
        assert "custom_group" not in dataset.groupings.columns

    def test_create_default_groupings_preserves_nulls(self, tmp_path):
        """Test that null values are preserved in groupings."""
        # Create metadata with some null values
        metadata_df = pl.DataFrame(
            {
                "sample": ["S1", "S2", "S3", "S4"],
                "biosample": ["BS1", "BS2", "BS3", "BS4"],
                "bioproject": ["BP1", None, "BP2", "BP2"],  # Null in S2
                "lat": [10.0, 20.0, 30.0, 40.0],
                "lon": [40.0, 50.0, 60.0, 70.0],
                "collection_date": [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                ],
                "biome": ["soil", "marine", None, "freshwater"],  # Null in S3
                "mbases": [1500, 2000, 1200, 1800],
            }
        )
        attributes_df = pl.DataFrame(
            {
                "sample": ["S1", "S2"],
                "key": ["pH", "pH"],
                "value": ["6.5", "7.0"],
            }
        )

        metadata_csv = tmp_path / "metadata.csv"
        attributes_csv = tmp_path / "attributes.csv"
        metadata_df.write_csv(metadata_csv)
        attributes_df.write_csv(attributes_csv)

        metadata = SampleMetadata(
            metadata=str(metadata_csv), attributes=str(attributes_csv)
        )
        dataset = Dataset(metadata=metadata)

        dataset.create_default_groupings(groupings=["bioproject", "biome"])

        assert dataset.groupings is not None
        groupings_collected = (
            dataset.groupings.collect()
            if hasattr(dataset.groupings, "collect")
            else dataset.groupings
        )

        # Check that nulls are preserved
        bioproject_nulls = groupings_collected.select(
            pl.col("bioproject").is_null().sum()
        ).item()
        biome_nulls = groupings_collected.select(
            pl.col("biome").is_null().sum()
        ).item()

        assert bioproject_nulls == 1  # S2 has null bioproject
        assert biome_nulls == 1  # S3 has null biome
