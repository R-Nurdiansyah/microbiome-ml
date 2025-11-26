"""Tests for FeatureSet class."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

from microbiome_ml.wrangle.features import FeatureSet


class TestFeatureSetInitialization:
    """Test FeatureSet initialization."""
    
    def test_direct_initialization(self):
        """Test direct initialization with numpy arrays."""
        accessions = ["S1", "S2", "S3"]
        feature_names = ["f1", "f2", "f3"]
        features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        
        fs = FeatureSet(accessions, feature_names, features, name="test")
        
        assert fs.accessions == accessions
        assert fs.feature_names == feature_names
        assert isinstance(fs.features, pl.LazyFrame)
        assert fs.name == "test"
        
        # Check data content
        df = fs.features.collect()
        assert df.shape == (3, 4)  # 3 samples, 3 features + 1 sample col
        
    def test_requires_name(self):
        """Test that name parameter is required."""
        accessions = ["S1", "S2"]
        feature_names = ["f1", "f2"]
        features = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        with pytest.raises(ValueError, match="name"):
            FeatureSet(accessions, feature_names, features, name=None)
            
    def test_dimension_validation(self):
        """Test dimension mismatch raises error."""
        accessions = ["S1", "S2"]
        feature_names = ["f1", "f2", "f3"]  # Mismatch
        features = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        with pytest.raises(ValueError):
            FeatureSet(accessions, feature_names, features, name="test")


class TestFeatureSetLoadScan:
    """Test loading from CSV files."""
    
    def test_load_eager(self, features_csv):
        """Test loading (always lazy internally)."""
        fs = FeatureSet.load(features_csv)
        
        assert isinstance(fs.features, pl.LazyFrame)
        assert len(fs.accessions) == 4
        assert len(fs.feature_names) == 3
        
    def test_scan_lazy(self, features_csv):
        """Test scanning (always lazy internally)."""
        fs = FeatureSet.scan(features_csv, name="lazy_test")
        
        assert isinstance(fs.features, pl.LazyFrame)
        assert fs.name == "lazy_test"


class TestFeatureSetSave:
    """Test saving to CSV."""
    
    def test_save_eager(self, sample_features_eager, tmp_path):
        """Test saving instance."""
        save_path = tmp_path / "features_save.csv"
        sample_features_eager.save(save_path)
        
        assert save_path.exists()
        
    def test_save_lazy(self, sample_features_lazy, tmp_path):
        """Test saving lazy instance."""
        save_path = tmp_path / "features_save_lazy.csv"
        sample_features_lazy.save(save_path)
        
        assert save_path.exists()
        
    def test_save_load_roundtrip(self, sample_features_eager, tmp_path):
        """Test save then load preserves data."""
        save_path = tmp_path / "features_roundtrip.csv"
        sample_features_eager.save(save_path)
        
        loaded = FeatureSet.load(save_path)
        
        assert loaded.accessions == sample_features_eager.accessions
        assert loaded.feature_names == sample_features_eager.feature_names
        
        # Compare data
        original_df = sample_features_eager.features.collect().sort("sample")
        loaded_df = loaded.features.collect().sort("sample")
        assert original_df.equals(loaded_df)


class TestFeatureSetFactoryMethods:
    """Test factory methods for creating FeatureSets."""
    
    def test_from_df(self, sample_feature_data):
        """Test creating from DataFrame."""
        df = pl.DataFrame(sample_feature_data)
        fs = FeatureSet.from_df(df, name="from_df_test")
        
        assert fs.name == "from_df_test"
        assert len(fs.accessions) == 4
        assert len(fs.feature_names) == 3
        
        # Check dimensions via collect
        assert fs.features.collect().shape == (4, 4) # 4 samples, 3 features + 1 sample col
        
    def test_from_taxonomic_profiles(self, sample_profiles_eager):
        """Test creating from TaxonomicProfiles."""
        from microbiome_ml.utils.taxonomy import TaxonomicRanks
        
        fs = sample_profiles_eager.create_features(TaxonomicRanks.PHYLUM)
        
        assert isinstance(fs, FeatureSet)
        assert fs.accessions is not None
        assert fs.feature_names is not None


class TestFeatureSetFiltering:
    """Test sample filtering."""
    
    def test_filter_samples_eager(self, sample_features_eager):
        """Test filtering."""
        samples_to_keep = ["S1", "S3"]
        
        filtered = sample_features_eager.filter_samples(samples_to_keep)
        
        assert isinstance(filtered.features, pl.LazyFrame)
        assert filtered.accessions == samples_to_keep
        
        # Check dimensions
        df = filtered.features.collect()
        assert df.height == 2
        
    def test_filter_samples_lazy(self, sample_features_lazy):
        """Test filtering lazy instance."""
        samples_to_keep = ["S2", "S4"]
        
        filtered = sample_features_lazy.filter_samples(samples_to_keep)
        
        assert isinstance(filtered.features, pl.LazyFrame)
        
        # collect to check results
        df = filtered.features.collect()
        assert filtered.accessions == samples_to_keep
        assert df.height == 2


class TestFeatureSetQueries:
    """Test query methods."""
    
    def test_get_samples(self, sample_features_eager):
        """Test getting features for specific samples."""
        samples = ["S1", "S2"]
        result = sample_features_eager.get_samples(samples)
        
        assert result.shape[0] == 2
        assert result.shape[1] == len(sample_features_eager.feature_names)
        
    def test_get_features(self, sample_features_eager):
        """Test getting specific features."""
        features = ["feature1", "feature3"]
        result = sample_features_eager.get_features(features)
        
        assert result.shape[0] == len(sample_features_eager.accessions)
        assert result.shape[1] == 2
        
    def test_to_df(self, sample_features_eager):
        """Test conversion to DataFrame."""
        df = sample_features_eager.to_df()
        
        assert isinstance(df, pl.DataFrame)
        assert "sample" in df.columns
        assert df.height == len(sample_features_eager.accessions)
        assert len(df.columns) == len(sample_features_eager.feature_names) + 1  # +1 for sample column
