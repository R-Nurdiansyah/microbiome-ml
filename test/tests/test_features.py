"""Tests for FeatureSet class."""

import numpy as np
import polars as pl
import pytest

from microbiome_ml.wrangle.features import FeatureSet


class TestFeatureSetInitialization:
    """Test FeatureSet initialization."""

    def test_direct_initialization(self):
        """Test direct initialization with numpy arrays."""
        accessions = ["S1", "S2", "S3"]
        feature_names = ["f1", "f2", "f3"]
        features = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )

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

    def test_save(self, sample_features, tmp_path):
        """Test saving instance (eager and lazy)."""
        save_path = tmp_path / "features_save.csv"
        sample_features.save(save_path)

        assert save_path.exists()

    def test_save_load_roundtrip(self, sample_features, tmp_path):
        """Test save then load preserves data."""
        save_path = tmp_path / "features_roundtrip.csv"
        sample_features.save(save_path)

        loaded = FeatureSet.load(save_path)

        assert loaded.accessions == sample_features.accessions
        assert loaded.feature_names == sample_features.feature_names

        # Compare data
        original_df = sample_features.features.collect().sort("sample")
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
        assert fs.features.collect().shape == (
            4,
            4,
        )  # 4 samples, 3 features + 1 sample col

    def test_from_taxonomic_profiles(self, sample_profiles):
        """Test creating from TaxonomicProfiles."""
        from microbiome_ml.utils.taxonomy import TaxonomicRanks

        fs = sample_profiles.create_features(TaxonomicRanks.PHYLUM)

        assert isinstance(fs, FeatureSet)
        assert fs.accessions is not None
        assert fs.feature_names is not None


class TestFeatureSetFiltering:
    """Test sample filtering."""

    def test_filter_samples(self, sample_features):
        """Test filtering (eager and lazy)."""
        samples_to_keep = ["S1", "S3"]
        # Ensure samples exist in the fixture data (S1, S2, S3, S4)
        # S1 and S3 are present.

        filtered = sample_features.filter_samples(samples_to_keep)

        assert isinstance(filtered.features, pl.LazyFrame)

        # Check accessions if available immediately (eager) or after collect
        # But FeatureSet.filter_samples updates accessions list immediately if possible?
        # Looking at implementation (implied), it probably does.
        assert filtered.accessions == samples_to_keep

        # Check dimensions
        df = filtered.features.collect()
        assert df.height == 2
        assert set(df["sample"].to_list()) == set(samples_to_keep)


class TestFeatureSetQueries:
    """Test query methods."""

    def test_get_samples(self, sample_features):
        """Test getting features for specific samples."""
        samples = ["S1", "S2"]
        result = sample_features.get_samples(samples)

        assert result.shape[0] == 2
        assert result.shape[1] == len(sample_features.feature_names)

    def test_get_features(self, sample_features):
        """Test getting specific features."""
        features = ["feature1", "feature3"]
        result = sample_features.get_features(features)

        assert result.shape[0] == len(sample_features.accessions)
        assert result.shape[1] == 2

    def test_to_df(self, sample_features):
        """Test conversion to DataFrame."""
        df = sample_features.to_df()

        assert isinstance(df, pl.DataFrame)
        assert "sample" in df.columns
        assert df.height == len(sample_features.accessions)
        assert (
            len(df.columns) == len(sample_features.feature_names) + 1
        )  # +1 for sample column
