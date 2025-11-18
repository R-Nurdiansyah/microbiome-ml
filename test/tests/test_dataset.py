"""Tests for Dataset class."""

import pytest
import polars as pl
import warnings
from pathlib import Path

from microbiome_ml.wrangle.samples import Dataset
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
            assert any("dropped" in str(warning.message).lower() for warning in w)
            
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
        
        assert loaded.metadata._is_lazy is True
        assert loaded.profiles._is_lazy is True
        
    def test_scan_alias(self, sample_dataset_eager, tmp_path):
        """Test scan is alias for load(lazy=True)."""
        save_dir = tmp_path / "dataset_scan"
        sample_dataset_eager.save(save_dir)
        
        scanned = Dataset.scan(save_dir)
        
        assert scanned.metadata._is_lazy is True
        assert scanned.profiles._is_lazy is True
        
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
