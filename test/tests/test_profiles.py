"""Tests for TaxonomicProfiles class."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

from microbiome_ml.wrangle.profiles import TaxonomicProfiles
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.utils.taxonomy import TaxonomicRanks


class TestTaxonomicProfilesInitialization:
    """Test TaxonomicProfiles initialization and modes."""
    
    def test_eager_initialization(self, profiles_csv, root_csv):
        """Test eager mode initialization from CSV files."""
        profiles = TaxonomicProfiles(profiles_csv, root=root_csv, check_filled=False)
        
        assert profiles._is_lazy is False
        assert profiles.profiles is not None
        assert isinstance(profiles.profiles, pl.DataFrame)
        assert profiles.profiles.height > 0
        
    def test_lazy_initialization(self, profiles_csv, root_csv):
        """Test lazy mode initialization via scan."""
        profiles = TaxonomicProfiles.scan(profiles_csv, root=root_csv, check_filled=False)
        
        assert profiles._is_lazy is True
        assert profiles._lf_profiles is not None
        assert isinstance(profiles._lf_profiles, pl.LazyFrame)
        
    def test_with_root(self, profiles_csv, root_csv):
        """Test initialization with root coverage."""
        profiles = TaxonomicProfiles(profiles_csv, root=root_csv, check_filled=False)
        
        assert profiles.root is not None
        assert isinstance(profiles.root, pl.DataFrame)
        
    def test_without_root(self, profiles_csv):
        """Test initialization without root coverage."""
        profiles = TaxonomicProfiles(profiles_csv, check_filled=False)
        
        # Root should be created from domain-level coverage
        assert profiles.root is not None


class TestTaxonomicProfilesMaterialize:
    """Test lazy-to-eager conversion."""
    
    def test_materialize_lazy(self, sample_profiles_lazy):
        """Test materializing a lazy instance."""
        assert sample_profiles_lazy._is_lazy is True
        
        sample_profiles_lazy.materialize()
        
        assert sample_profiles_lazy._is_lazy is False
        assert sample_profiles_lazy.profiles is not None
        assert isinstance(sample_profiles_lazy.profiles, pl.DataFrame)
        assert sample_profiles_lazy._lf_profiles is None
        
    def test_materialize_eager_noop(self, sample_profiles_eager):
        """Test materializing an already-eager instance (no-op)."""
        assert sample_profiles_eager._is_lazy is False
        
        sample_profiles_eager.materialize()
        
        assert sample_profiles_eager._is_lazy is False
        assert sample_profiles_eager.profiles is not None


class TestTaxonomicProfilesSaveLoad:
    """Test save/load round-trips."""
    
    def test_save_eager(self, sample_profiles_eager, tmp_path):
        """Test saving eager instance."""
        save_dir = tmp_path / "profiles_save"
        sample_profiles_eager.save(save_dir)
        
        assert (save_dir / "profiles.csv").exists()
        assert (save_dir / "root.csv").exists()
        
    def test_save_lazy(self, sample_profiles_lazy, tmp_path):
        """Test saving lazy instance (should materialize)."""
        save_dir = tmp_path / "profiles_save_lazy"
        sample_profiles_lazy.save(save_dir)
        
        assert (save_dir / "profiles.csv").exists()
        
    def test_load_eager(self, sample_profiles_eager, tmp_path):
        """Test save then load in eager mode."""
        save_dir = tmp_path / "profiles_roundtrip"
        sample_profiles_eager.save(save_dir)
        
        loaded = TaxonomicProfiles.load(save_dir, lazy=False, check_filled=False)
        
        assert loaded._is_lazy is False
        assert loaded.profiles.shape == sample_profiles_eager.profiles.shape
        
    def test_load_lazy(self, sample_profiles_eager, tmp_path):
        """Test save then load in lazy mode."""
        save_dir = tmp_path / "profiles_roundtrip_lazy"
        sample_profiles_eager.save(save_dir)
        
        loaded = TaxonomicProfiles.load(save_dir, lazy=True, check_filled=False)
        
        assert loaded._is_lazy is True
        assert loaded._lf_profiles is not None
        
    def test_scan_alias(self, sample_profiles_eager, tmp_path):
        """Test scan is alias for load(lazy=True)."""
        save_dir = tmp_path / "profiles_scan"
        sample_profiles_eager.save(save_dir)
        
        scanned = TaxonomicProfiles.scan(save_dir, check_filled=False)
        
        assert scanned._is_lazy is True


class TestTaxonomicProfilesFiltering:
    """Test filtering methods preserve mode."""
    
    def test_filter_by_sample_eager(self, sample_profiles_eager):
        """Test filtering preserves eager mode."""
        samples_lf = pl.DataFrame({"sample": ["S1"]}).lazy()
        
        filtered = sample_profiles_eager._filter_by_sample(samples_lf)
        
        assert filtered._is_lazy is False
        assert filtered.profiles is not None
        unique_samples = set(filtered.profiles["sample"].unique().to_list())
        assert unique_samples == {"S1"}
        
    def test_filter_by_sample_lazy(self, sample_profiles_lazy):
        """Test filtering preserves lazy mode."""
        samples_lf = pl.DataFrame({"sample": ["S2"]}).lazy()
        
        filtered = sample_profiles_lazy._filter_by_sample(samples_lf)
        
        assert filtered._is_lazy is True
        assert filtered._lf_profiles is not None
        
        # Materialize to check results
        filtered.materialize()
        assert filtered._is_lazy is False
        assert filtered.profiles is not None
        unique_samples = set(filtered.profiles["sample"].unique().to_list())
        assert unique_samples == {"S2"}


class TestTaxonomicProfilesTaxonomy:
    """Test taxonomy operations."""
    
    def test_taxonomy_standardization(self, sample_profiles_eager):
        """Test taxonomy strings are properly formatted."""
        # Check that taxonomy strings don't have spaces after semicolons
        taxonomies = sample_profiles_eager.profiles["taxonomy"].to_list()
        for tax in taxonomies:
            assert "; " not in tax, f"Taxonomy has space after semicolon: {tax}"
            
    def test_is_filled_flag(self, sample_profiles_eager):
        """Test is_filled flag is set correctly."""
        # Our test data is in filled format
        assert sample_profiles_eager.is_filled is True
        
    def test_rank_extraction(self, sample_profiles_eager):
        """Test extracting specific taxonomic rank."""
        rank_lf = sample_profiles_eager.get_rank(TaxonomicRanks.PHYLUM)
        
        assert rank_lf is not None
        # Collect to DataFrame to inspect
        rank_df = rank_lf.collect()
        assert "taxonomy" in rank_df.columns
        # All taxonomies at phylum level should have exactly 2 levels (domain;phylum)
        taxonomies = rank_df["taxonomy"].to_list()
        for tax in taxonomies:
            levels = tax.split(";")
            assert len(levels) == 2, f"Expected 2 levels for phylum, got {len(levels)}: {tax}"


class TestTaxonomicProfilesFeatureCreation:
    """Test feature set creation from profiles."""
    
    def test_create_features_eager(self, sample_profiles_eager):
        """Test creating FeatureSet from eager profiles."""
        features = sample_profiles_eager.create_features(TaxonomicRanks.PHYLUM)
        
        assert isinstance(features, FeatureSet)
        assert features.accessions is not None
        assert features.feature_names is not None
        assert features.features is not None
        
    def test_create_features_lazy(self, sample_profiles_lazy):
        """Test creating FeatureSet from lazy profiles."""
        features = sample_profiles_lazy.create_features(TaxonomicRanks.PHYLUM)
        
        assert isinstance(features, FeatureSet)
        # create_features() always returns eager FeatureSet (calls .collect())
        assert features._is_lazy is False
        assert features.features is not None


class TestTaxonomicProfilesRelativeAbundance:
    """Test coverage to relative abundance conversion."""
    
    def test_to_relabund(self, sample_profiles_eager):
        """Test conversion to relative abundance."""
        # Profiles should already be in relabund format after initialization
        # Check that relabund column exists and values are normalized per-taxonomy
        relabund_values = sample_profiles_eager.profiles.filter(
            pl.col("sample") == "S1"
        )["relabund"].to_list()
        
        # Just verify relabund column exists and contains values
        assert len(relabund_values) > 0
        assert all(v >= 0 for v in relabund_values)
        
        # Individual values should be between 0 and 1
        for val in relabund_values:
            assert 0.0 <= val <= 1.0
