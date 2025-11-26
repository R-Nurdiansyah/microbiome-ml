"""Tests for Dataset class."""

import pytest
import polars as pl
import warnings
from pathlib import Path

from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.utils.taxonomy import TaxonomicRanks


class TestDatasetInitialization:
    """Test Dataset initialization."""
    
    def test_empty_initialization(self):
        """Test creating empty Dataset."""
        dataset = Dataset()
        
        assert dataset.metadata is None
        assert dataset.profiles is None
        assert dataset.feature_sets == {}
        assert dataset.labels == {}
        assert dataset._sample_ids is None
        
    def test_initialization_with_metadata(self, sample_metadata_eager):
        """Test initialization with metadata."""
        dataset = Dataset(metadata=sample_metadata_eager)
        
        assert dataset.metadata is not None
        assert dataset._sample_ids is not None
        assert len(dataset._sample_ids) == 4
        
    def test_initialization_with_all_components(self, sample_metadata_eager, sample_profiles_eager):
        """Test initialization with multiple components."""
        dataset = Dataset(metadata=sample_metadata_eager, profiles=sample_profiles_eager)
        
        assert dataset.metadata is not None
        assert dataset.profiles is not None
        assert dataset._sample_ids is not None


class TestDatasetBuilderPattern:
    """Test builder pattern methods."""
    
    def test_add_metadata(self, sample_dataset_empty, metadata_csv, attributes_csv):
        """Test adding metadata via builder."""
        result = sample_dataset_empty.add_metadata(metadata=metadata_csv, attributes=attributes_csv)
        
        assert result is sample_dataset_empty  # Chaining works
        assert sample_dataset_empty.metadata is not None
        
    def test_add_profiles(self, sample_dataset_empty, profiles_csv, root_csv):
        """Test adding profiles via builder."""
        result = sample_dataset_empty.add_profiles(profiles_csv, root=root_csv, check_filled=False)
        
        assert result is sample_dataset_empty
        assert sample_dataset_empty.profiles is not None
        
    def test_add_features_from_rank(self, sample_dataset_eager):
        """Test adding features from taxonomic rank."""
        result = sample_dataset_eager.add_features("genus_features", rank=TaxonomicRanks.GENUS)
        
        assert result is sample_dataset_eager
        assert "genus_features" in sample_dataset_eager.feature_sets
        
    def test_add_features_from_instance(self, sample_dataset_empty, sample_features_eager):
        """Test adding existing FeatureSet instance."""
        result = sample_dataset_empty.add_features("my_features", features=sample_features_eager)
        
        assert result is sample_dataset_empty
        assert "my_features" in sample_dataset_empty.feature_sets
        
    def test_method_chaining(self, metadata_csv, attributes_csv, profiles_csv):
        """Test chaining multiple builder methods."""
        dataset = (Dataset()
                  .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                  .add_profiles(profiles_csv, check_filled=False))
        
        assert dataset.metadata is not None
        assert dataset.profiles is not None


class TestDatasetBatchOperations:
    """Test batch addition methods."""
    
    def test_add_feature_set_dict(self, sample_dataset_empty, features_csv, tmp_path):
        """Test adding multiple feature sets at once."""
        # Create second features file
        features_csv2 = tmp_path / "features2.csv"
        pl.DataFrame({
            "sample": ["S1", "S2", "S3"],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0]
        }).write_csv(features_csv2)
        
        result = sample_dataset_empty.add_feature_set({
            "set1": features_csv,
            "set2": features_csv2
        })
        
        assert result is sample_dataset_empty
        assert "set1" in sample_dataset_empty.feature_sets
        assert "set2" in sample_dataset_empty.feature_sets
        
    def test_add_feature_set_single_requires_name(self, sample_dataset_empty, features_csv):
        """Test single feature set addition requires name."""
        with pytest.raises(ValueError, match="name"):
            sample_dataset_empty.add_feature_set(features_csv)
            
    def test_add_labels_dict(self, sample_dataset_empty, labels_csv, tmp_path):
        """Test adding multiple label sets at once."""
        labels_csv2 = tmp_path / "labels2.csv"
        pl.DataFrame({
            "sample": ["S1", "S2", "S3"],
            "outcome": [1, 0, 1]
        }).write_csv(labels_csv2)
        
        result = sample_dataset_empty.add_labels({
            "labels1": labels_csv,
            "labels2": labels_csv2
        })
        
        assert result is sample_dataset_empty
        assert "labels1" in sample_dataset_empty.labels
        assert "labels2" in sample_dataset_empty.labels
        
    def test_add_labels_single_requires_name(self, sample_dataset_empty, labels_csv):
        """Test single label addition requires name."""
        with pytest.raises(ValueError, match="name"):
            sample_dataset_empty.add_labels(labels_csv)


class TestDatasetTaxonomicFeatures:
    """Test taxonomic feature generation."""
    
    def test_add_taxonomic_features_all_ranks(self, sample_dataset_eager):
        """Test generating features for all standard ranks."""
        result = sample_dataset_eager.add_taxonomic_features()
        
        assert result is sample_dataset_eager
        # Should have 7 standard ranks
        assert len([k for k in sample_dataset_eager.feature_sets.keys() if k.startswith("tax_")]) == 7
        
    def test_add_taxonomic_features_specific_ranks(self, sample_dataset_eager):
        """Test generating features for specific ranks."""
        result = sample_dataset_eager.add_taxonomic_features(
            ranks=[TaxonomicRanks.GENUS, TaxonomicRanks.PHYLUM]
        )
        
        assert "tax_genus" in sample_dataset_eager.feature_sets
        assert "tax_phylum" in sample_dataset_eager.feature_sets
        
    def test_add_taxonomic_features_custom_prefix(self, sample_dataset_eager):
        """Test custom prefix for taxonomic features."""
        result = sample_dataset_eager.add_taxonomic_features(
            ranks=[TaxonomicRanks.GENUS],
            prefix="custom"
        )
        
        assert "custom_genus" in sample_dataset_eager.feature_sets
        
    def test_add_taxonomic_features_requires_profiles(self, sample_dataset_empty):
        """Test that profiles are required for taxonomic features."""
        with pytest.raises(ValueError, match="Profiles must be added"):
            sample_dataset_empty.add_taxonomic_features()


class TestDatasetAccessionSync:
    """Test accession synchronization."""
    
    def test_sync_finds_intersection(self, metadata_csv, attributes_csv, profiles_csv, tmp_path):
        """Test sync computes strict intersection."""
        # Create metadata with S1, S2, S3, S4
        # Create profiles with only S1, S2
        profiles_subset = tmp_path / "profiles_subset.csv"
        pl.DataFrame({
            "sample": ["S1", "S1", "S2", "S2"],
            "taxonomy": ["d__Bacteria", "d__Bacteria;p__Proteobacteria"] * 2,
            "coverage": [100.0, 50.0, 120.0, 60.0]
        }).write_csv(profiles_subset)
        
        dataset = (Dataset()
                  .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                  .add_profiles(profiles_subset, check_filled=False))
        
        # Should only have S1, S2 (intersection)
        assert set(dataset._sample_ids) == {"S1", "S2"}
        
    def test_sync_warning_on_large_drop(self, metadata_csv, attributes_csv, tmp_path):
        """Test warning is issued when >10% samples dropped."""
        # Create profiles with only 1 sample (will drop 75% of metadata samples)
        profiles_single = tmp_path / "profiles_single.csv"
        pl.DataFrame({
            "sample": ["S1", "S1"],
            "taxonomy": ["d__Bacteria", "d__Bacteria;p__Proteobacteria"],
            "coverage": [100.0, 50.0]
        }).write_csv(profiles_single)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset = (Dataset()
                      .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                      .add_profiles(profiles_single, check_filled=False))
            
            # Should have issued warning
            assert len(w) > 0
            assert any("removed" in str(warning.message).lower() for warning in w)
            
    def test_get_sample_ids(self, sample_dataset_eager):
        """Test getting canonical sample IDs."""
        sample_ids = sample_dataset_eager.get_sample_ids()
        
        assert isinstance(sample_ids, list)
        assert len(sample_ids) > 0
        # Should be sorted
        assert sample_ids == sorted(sample_ids)


class TestDatasetPreprocessing:
    """Test preprocessing pipeline."""
    
    def test_apply_preprocessing_default(self, sample_dataset_eager):
        """Test default preprocessing."""
        result = sample_dataset_eager.apply_preprocessing()
        
        assert result is sample_dataset_eager
        
    def test_apply_preprocessing_no_sync(self, sample_dataset_eager):
        """Test preprocessing without sync."""
        result = sample_dataset_eager.apply_preprocessing(sync_after=False)
        
        assert result is sample_dataset_eager
        
    def test_apply_preprocessing_selective(self, sample_dataset_eager):
        """Test selective preprocessing."""
        result = sample_dataset_eager.apply_preprocessing(
            metadata_qc=True,
            profiles_qc=False,
            sync_after=True
        )
        
        assert result is sample_dataset_eager
        
    def test_apply_preprocessing_metadata_qc_flag(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test metadata QC flag is set after preprocessing."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        dataset = Dataset(metadata=meta)
        
        assert dataset.metadata_qc_done is False
        
        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=False, sync_after=False)
        
        assert dataset.metadata_qc_done is True
        
    def test_apply_preprocessing_profiles_qc_flag(self, dominated_profiles_csv, dominated_root_csv):
        """Test profiles QC flag is set after preprocessing."""
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        dataset = Dataset(profiles=profiles)
        
        assert dataset.profiles_qc_done is False
        
        dataset.apply_preprocessing(metadata_qc=False, profiles_qc=True, sync_after=False)
        
        assert dataset.profiles_qc_done is True
        
    def test_apply_preprocessing_custom_metadata_params(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test custom metadata QC parameters are applied."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        dataset = Dataset(metadata=meta)
        
        # Before: 5 samples
        assert dataset.metadata.metadata.collect().height == 7
        
        # Apply with custom cutoff of 900
        dataset.apply_preprocessing(
            metadata_qc=True,
            metadata_mbp_cutoff=900,
            profiles_qc=False,
            sync_after=False
        )

        # After: 5 samples
        assert dataset.metadata.metadata.collect().height == 5
        
    def test_apply_preprocessing_custom_profiles_params(self, dominated_profiles_csv, dominated_root_csv):
        """Test custom profiles QC parameters are applied."""
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        dataset = Dataset(profiles=profiles)
        
        # Before: 4 samples
        initial_samples = len(set(dataset.profiles.profiles.select("sample").unique().collect().to_series().to_list()))
        assert initial_samples == 4
        
        # Apply with very lenient domination cutoff
        dataset.apply_preprocessing(
            metadata_qc=False,
            profiles_qc=True,
            profiles_dominated_cutoff=1.0,  # Keep everything
            sync_after=False
        )
        
        # After: all 4 samples remain (only coverage filter applied)
        final_samples = len(set(dataset.profiles.profiles.select("sample").unique().collect().to_series().to_list()))
        assert final_samples == 4
        
    def test_apply_preprocessing_skips_when_flag_set(self, sample_dataset_eager):
        """Test QC is skipped when flag is already True."""
        # Manually set flags
        sample_dataset_eager.metadata_qc_done = True
        sample_dataset_eager.profiles_qc_done = True
        
        initial_meta_height = sample_dataset_eager.metadata.metadata.collect().height
        initial_profiles_samples = len(set(sample_dataset_eager.profiles.profiles.select("sample").unique().collect().to_series().to_list()))
        
        # Apply preprocessing - should be skipped
        sample_dataset_eager.apply_preprocessing()
        
        # Verify no changes
        assert sample_dataset_eager.metadata.metadata.collect().height == initial_meta_height
        assert len(set(sample_dataset_eager.profiles.profiles.select("sample").unique().collect().to_series().to_list())) == initial_profiles_samples
        
    def test_apply_preprocessing_synchronizes_after_qc(self, low_quality_metadata_csv, low_quality_attributes_csv,
                                                        dominated_profiles_csv, dominated_root_csv):
        """Test synchronization happens after QC."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        
        # Create dataset - automatically syncs on init
        dataset = Dataset(metadata=meta, profiles=profiles)
        
        # After init sync, should have intersection: DOM1, DOM2, HQ1, HQ2
        initial_meta_samples = set(dataset.metadata.metadata.select("sample").collect().to_series().to_list())
        initial_profile_samples = set(dataset.profiles.profiles.select("sample").unique().collect().to_series().to_list())
        assert initial_meta_samples == initial_profile_samples
        assert initial_meta_samples == {"DOM1", "DOM2", "HQ1", "HQ2"}
        
        # Apply preprocessing - DOM1 and DOM2 are dominated and should be filtered
        dataset.apply_preprocessing()
        
        # After QC, only high quality samples remain
        final_meta_samples = set(dataset.metadata.metadata.select("sample").collect().to_series().to_list())
        final_profile_samples = set(dataset.profiles.profiles.select("sample").unique().collect().to_series().to_list())
        assert final_meta_samples == final_profile_samples
        assert final_meta_samples == {"HQ1", "HQ2"}
        
    def test_apply_preprocessing_metadata_only(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test preprocessing with only metadata component."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        dataset = Dataset(metadata=meta)
        
        # Should not crash when profiles is None
        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True)
        
        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is False  # No profiles to process
        
    def test_apply_preprocessing_profiles_only(self, dominated_profiles_csv, dominated_root_csv):
        """Test preprocessing with only profiles component."""
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        dataset = Dataset(profiles=profiles)
        
        # Should not crash when metadata is None
        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True)
        
        assert dataset.metadata_qc_done is False  # No metadata to process
        assert dataset.profiles_qc_done is True
        
    def test_apply_preprocessing_both_components(self, low_quality_metadata_csv, low_quality_attributes_csv,
                                                   dominated_profiles_csv, dominated_root_csv):
        """Test preprocessing with both components."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        dataset = Dataset(metadata=meta, profiles=profiles)
        
        dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True)
        
        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is True


class TestDatasetPreprocessingEdgeCases:
    """Test preprocessing edge cases and transformations."""
    
    def test_preprocessing_creates_root_if_missing(self, profiles_csv):
        """Test preprocessing creates root if not provided."""
        # Load profiles without root
        profiles = TaxonomicProfiles(profiles_csv, root=None, check_filled=False)
        dataset = Dataset(profiles=profiles)
        
        # Root should exist after __init__ (created automatically)
        assert dataset.profiles.root is not None or dataset.profiles._lf_root is not None
        
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
                "d__Bacteria;p__Actinobacteria;c__Actinobacteria;o__Actinomycetales;f__Actinomycetaceae;g__Actinomyces;s__Actinomyces_odontolyticus"
            ] * 2,
            "relabund": [0.4, 0.35, 0.25] * 2
        }
        unfilled_csv = tmp_path / "unfilled_profiles.csv"
        pl.DataFrame(unfilled_data).write_csv(unfilled_csv)
        
        root_data = {
            "sample": ["S1", "S2", "S3"],
            "root_coverage": [100.0, 120.0, 110.0]
        }
        root_csv = tmp_path / "unfilled_root.csv"
        pl.DataFrame(root_data).write_csv(root_csv)
        
        # Load without filling
        profiles = TaxonomicProfiles(str(unfilled_csv), root=str(root_csv), check_filled=False)
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
                "d__Bacteria;p__Firmicutes"
            ] * 2,
            "coverage": [100.0, 60.0, 40.0, 120.0, 70.0, 50.0]
        }
        coverage_csv = tmp_path / "coverage_profiles.csv"
        pl.DataFrame(coverage_data).write_csv(coverage_csv)
        
        root_data = {
            "sample": ["S1", "S2"],
            "root_coverage": [100.0, 120.0]
        }
        root_csv = tmp_path / "coverage_root.csv"
        pl.DataFrame(root_data).write_csv(root_csv)
        
        profiles = TaxonomicProfiles(str(coverage_csv), root=str(root_csv), check_filled=False)
        dataset = Dataset(profiles=profiles)
        
        # it auto converts during init
        schema_names = dataset.profiles.profiles.collect_schema().names()
        assert "relabund" in schema_names
        
        # test the values are correct
        # test the values are correct
        relabund_df = dataset.profiles.profiles.select(["sample", "taxonomy", "relabund"]).collect()
        expected_relabund = {
            ("S1", "d__Bacteria"): 100.0 / 100.0,
            ("S1", "d__Bacteria;p__Proteobacteria"): 60.0 / 100.0,
            ("S1", "d__Bacteria;p__Firmicutes"): 40.0 / 100.0,
            ("S2", "d__Bacteria"): 120.0 / 120.0,
            ("S2", "d__Bacteria;p__Proteobacteria"): 70.0 / 120.0,
            ("S2", "d__Bacteria;p__Firmicutes"): 50.0 / 120.0
        }

        # Convert to list of dictionaries for easier iteration
        for row in relabund_df.to_dicts():
            key = (row["sample"], row["taxonomy"])
            assert pytest.approx(row["relabund"]) == expected_relabund[key]
            
        
    def test_preprocessing_idempotency(self, sample_dataset_eager):
        """Test repeated preprocessing calls are idempotent via flags."""
        # First preprocessing
        sample_dataset_eager.apply_preprocessing()
        
        first_meta_height = sample_dataset_eager.metadata.metadata.collect().height
        first_profile_samples = set(sample_dataset_eager.profiles.profiles.collect()["sample"].unique().to_list())
        
        # Second preprocessing - should be skipped due to flags
        sample_dataset_eager.apply_preprocessing()
        
        second_meta_height = sample_dataset_eager.metadata.metadata.collect().height
        second_profile_samples = set(sample_dataset_eager.profiles.profiles.collect()["sample"].unique().to_list())
        
        # No changes from second call
        assert first_meta_height == second_meta_height
        assert first_profile_samples == second_profile_samples
        
    def test_preprocessing_custom_thresholds_filter_correctly(self, low_quality_metadata_csv, 
                                                               low_quality_attributes_csv):
        """Test custom thresholds actually filter the expected samples."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        dataset = Dataset(metadata=meta)
        
        # Cutoff of 800 should keep LQ2 (800), LQ3 (950), HQ1 (1500), HQ2 (2000)
        dataset.apply_preprocessing(
            metadata_qc=True,
            metadata_mbp_cutoff=800,
            sync_after=False
        )
        
        final_samples = set(dataset.metadata.metadata.select("sample").collect().to_series().to_list())
        assert final_samples == {"DOM1", "DOM2", "LQ3", "HQ1", "HQ2"}
        assert "LQ1" not in final_samples  # 500 mbases, filtered out


class TestDatasetSynchronizationWarnings:
    """Test warning messages during sample synchronization."""
    
    def test_sync_warns_when_samples_dropped(self, low_quality_metadata_csv, low_quality_attributes_csv,
                                              dominated_profiles_csv, dominated_root_csv):
        """Test warning is issued when synchronization drops samples."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        
        # Different sample sets - synchronization will drop samples
        # Should issue warning during sync (which happens in __init__)
        with pytest.warns(UserWarning, match="Accession sync removed"):
            dataset = Dataset(metadata=meta, profiles=profiles)
            
    def test_preprocessing_warns_on_sync(self, low_quality_metadata_csv, low_quality_attributes_csv,
                                          dominated_profiles_csv, dominated_root_csv):
        """Test preprocessing issues warnings when QC drops samples during sync."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        profiles = TaxonomicProfiles(dominated_profiles_csv, root=dominated_root_csv, check_filled=False)
        dataset = Dataset(metadata=meta, profiles=profiles)
        
        # QC will filter different samples from each component, sync will drop some
        with pytest.warns(UserWarning):
            dataset.apply_preprocessing(metadata_qc=True, profiles_qc=True, sync_after=True)


class TestDatasetLargeScaleIntegration:
    """Test with large datasets (100 samples) for realistic QC pipeline."""
    
    def test_large_dataset_qc_pipeline(self, tmp_path, large_metadata_data, large_profiles_data, large_root_data):
        """Test full QC pipeline with 100 samples."""
        # Create CSV files
        metadata_csv = tmp_path / "large_metadata.csv"
        pl.DataFrame(large_metadata_data).write_csv(metadata_csv)
        
        attributes_data = {
            "sample": large_metadata_data["sample"] * 2,
            "key": ["pH", "temp"] * len(large_metadata_data["sample"]),
            "value": [str(i % 10) for i in range(len(large_metadata_data["sample"]) * 2)]
        }
        attributes_csv = tmp_path / "large_attributes.csv"
        pl.DataFrame(attributes_data).write_csv(attributes_csv)
        
        profiles_csv = tmp_path / "large_profiles.csv"
        pl.DataFrame(large_profiles_data).write_csv(profiles_csv)
        
        root_csv = tmp_path / "large_root.csv"
        pl.DataFrame(large_root_data).write_csv(root_csv)
        
        # Create dataset
        meta = SampleMetadata(str(metadata_csv), str(attributes_csv))
        profiles = TaxonomicProfiles(str(profiles_csv), root=str(root_csv), check_filled=False)
        dataset = Dataset(metadata=meta, profiles=profiles)
        
        # Initial: 100 samples
        assert dataset.metadata.metadata.collect().height == 100
        initial_profile_samples = len(set(dataset.profiles.profiles.select("sample").unique().collect().to_series().to_list()))
        assert initial_profile_samples == 100
        
        # Apply preprocessing with default thresholds
        dataset.apply_preprocessing(
            metadata_qc=True,
            profiles_qc=True,
            sync_after=True,
            metadata_mbp_cutoff=1000,
            profiles_cov_cutoff=50.0,
            profiles_dominated_cutoff=0.99
        )
        
        # After QC:
        # - Metadata: ~70 samples (30 below 1000 mbases)
        # - Profiles coverage: ~75 samples (25 below 50 coverage)
        # - Profiles domination: ~80 samples (20 dominated)
        # - Intersection of all filters should leave ~50-60 samples
        
        final_meta_samples = dataset.metadata.metadata.collect().height
        final_profile_samples = len(set(dataset.profiles.profiles.select("sample").unique().collect().to_series().to_list()))
        
        # Verify filtering happened
        assert final_meta_samples < 100
        assert final_profile_samples < 100
        
        # Verify synchronization (same samples in both)
        assert final_meta_samples == final_profile_samples
        
        # Verify flags are set
        assert dataset.metadata_qc_done is True
        assert dataset.profiles_qc_done is True
        
    def test_large_dataset_custom_thresholds(self, tmp_path, large_metadata_data, large_profiles_data, large_root_data):
        """Test custom thresholds with large dataset."""
        # Create CSV files
        metadata_csv = tmp_path / "large_metadata2.csv"
        pl.DataFrame(large_metadata_data).write_csv(metadata_csv)
        
        attributes_data = {
            "sample": large_metadata_data["sample"] * 2,
            "key": ["pH", "temp"] * len(large_metadata_data["sample"]),
            "value": [str(i % 10) for i in range(len(large_metadata_data["sample"]) * 2)]
        }
        attributes_csv = tmp_path / "large_attributes2.csv"
        pl.DataFrame(attributes_data).write_csv(attributes_csv)
        
        profiles_csv = tmp_path / "large_profiles2.csv"
        pl.DataFrame(large_profiles_data).write_csv(profiles_csv)
        
        root_csv = tmp_path / "large_root2.csv"
        pl.DataFrame(large_root_data).write_csv(root_csv)
        
        meta = SampleMetadata(str(metadata_csv), str(attributes_csv))
        profiles = TaxonomicProfiles(str(profiles_csv), root=str(root_csv), check_filled=False)
        dataset = Dataset(metadata=meta, profiles=profiles)
        
        # Very lenient thresholds - should keep most samples
        dataset.apply_preprocessing(
            metadata_qc=True,
            profiles_qc=True,
            sync_after=True,
            metadata_mbp_cutoff=500,
            profiles_cov_cutoff=20.0,
            profiles_dominated_cutoff=1.0  # Don't filter dominated
        )
        
        final_samples = dataset.metadata.metadata.collect().height
        
        # Should keep > 80 samples with lenient thresholds
        assert final_samples > 80
        
    def test_large_dataset_lazy_mode(self, tmp_path, large_metadata_data, large_profiles_data, large_root_data):
        """Test large dataset QC in lazy mode."""
        # Create CSV files
        metadata_csv = tmp_path / "large_metadata_lazy.csv"
        pl.DataFrame(large_metadata_data).write_csv(metadata_csv)
        
        attributes_csv = tmp_path / "large_attributes_lazy.csv"
        attributes_data = {
            "sample": large_metadata_data["sample"] * 2,
            "key": ["pH", "temp"] * len(large_metadata_data["sample"]),
            "value": [str(i % 10) for i in range(len(large_metadata_data["sample"]) * 2)]
        }
        pl.DataFrame(attributes_data).write_csv(attributes_csv)
        
        profiles_csv = tmp_path / "large_profiles_lazy.csv"
        pl.DataFrame(large_profiles_data).write_csv(profiles_csv)
        
        root_csv = tmp_path / "large_root_lazy.csv"
        pl.DataFrame(large_root_data).write_csv(root_csv)
        
        # Load in lazy mode
        meta = SampleMetadata.scan(str(metadata_csv), str(attributes_csv))
        profiles = TaxonomicProfiles.scan(str(profiles_csv), root=str(root_csv), check_filled=False)
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
        sample_dataset_eager.add_features("test_features", rank=TaxonomicRanks.GENUS)
        
        feature_sets = list(sample_dataset_eager.iter_feature_sets())
        
        assert len(feature_sets) > 0
        for name, fs in feature_sets:
            assert isinstance(name, str)
            assert isinstance(fs, FeatureSet)
            
    def test_iter_feature_sets_filtered(self, sample_dataset_eager):
        """Test iterating over specific feature sets."""
        sample_dataset_eager.add_features("fs1", rank=TaxonomicRanks.GENUS)
        sample_dataset_eager.add_features("fs2", rank=TaxonomicRanks.PHYLUM)
        
        feature_sets = list(sample_dataset_eager.iter_feature_sets(names=["fs1"]))
        
        assert len(feature_sets) == 1
        assert feature_sets[0][0] == "fs1"
        
    def test_iter_labels(self, sample_dataset_empty, labels_csv):
        """Test iterating over labels."""
        sample_dataset_empty.add_labels(labels_csv, name="test_labels")
        
        labels = list(sample_dataset_empty.iter_labels())
        
        assert len(labels) == 1
        assert labels[0][0] == "test_labels"


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
        
    def test_save_with_features_and_labels(self, sample_dataset_eager, labels_csv, tmp_path):
        """Test saving dataset with features and labels."""
        # Add features and labels
        sample_dataset_eager.add_features("test_features", rank=TaxonomicRanks.GENUS)
        sample_dataset_eager.add_labels(labels_csv, name="test_labels")
        
        save_dir = tmp_path / "dataset_full"
        sample_dataset_eager.save(save_dir)
        
        assert (save_dir / "features").exists()
        assert (save_dir / "labels").exists()
        
        # Load and verify
        loaded = Dataset.load(save_dir, lazy=False)
        assert "test_features" in loaded.feature_sets
        assert "test_labels" in loaded.labels
