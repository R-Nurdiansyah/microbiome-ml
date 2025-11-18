"""Tests for SampleMetadata class."""

import pytest
import polars as pl
from pathlib import Path

from microbiome_ml.wrangle.metadata import SampleMetadata


class TestSampleMetadataInitialization:
    """Test SampleMetadata initialization and modes."""
    
    def test_eager_initialization(self, metadata_csv, attributes_csv):
        """Test eager mode initialization from CSV files."""
        meta = SampleMetadata(metadata_csv, attributes_csv)
        
        assert meta._is_lazy is False
        assert meta.metadata is not None
        assert meta.attributes is not None
        assert isinstance(meta.metadata, pl.DataFrame)
        assert isinstance(meta.attributes, pl.DataFrame)
        assert meta.metadata.height == 4
        
    def test_lazy_initialization(self, metadata_csv, attributes_csv):
        """Test lazy mode initialization via scan."""
        meta = SampleMetadata.scan(metadata_csv, attributes_csv)
        
        assert meta._is_lazy is True
        assert meta._lf_metadata is not None
        assert meta._lf_attributes is not None
        assert isinstance(meta._lf_metadata, pl.LazyFrame)
        assert isinstance(meta._lf_attributes, pl.LazyFrame)
        
    def test_with_study_titles(self, metadata_csv, attributes_csv, study_titles_csv):
        """Test initialization with optional study titles."""
        meta = SampleMetadata(metadata_csv, attributes_csv, study_titles_csv)
        
        assert meta.study_titles is not None
        assert isinstance(meta.study_titles, pl.DataFrame)
        assert meta.study_titles.height == 2
        
    def test_without_study_titles(self, metadata_csv, attributes_csv):
        """Test initialization without study titles."""
        meta = SampleMetadata(metadata_csv, attributes_csv)
        
        assert meta.study_titles is None


class TestSampleMetadataMaterialize:
    """Test lazy-to-eager conversion."""
    
    def test_materialize_lazy(self, sample_metadata_lazy):
        """Test materializing a lazy instance."""
        assert sample_metadata_lazy._is_lazy is True
        
        sample_metadata_lazy.materialize()
        
        assert sample_metadata_lazy._is_lazy is False
        assert sample_metadata_lazy.metadata is not None
        assert isinstance(sample_metadata_lazy.metadata, pl.DataFrame)
        assert sample_metadata_lazy._lf_metadata is None
        
    def test_materialize_eager_noop(self, sample_metadata_eager):
        """Test materializing an already-eager instance (no-op)."""
        assert sample_metadata_eager._is_lazy is False
        
        sample_metadata_eager.materialize()
        
        assert sample_metadata_eager._is_lazy is False
        assert sample_metadata_eager.metadata is not None


class TestSampleMetadataSaveLoad:
    """Test save/load round-trips."""
    
    def test_save_eager(self, sample_metadata_eager, tmp_path):
        """Test saving eager instance."""
        save_dir = tmp_path / "metadata_save"
        sample_metadata_eager.save(save_dir)
        
        assert (save_dir / "metadata.csv").exists()
        assert (save_dir / "attributes.csv").exists()
        assert (save_dir / "study_titles.csv").exists()
        
    def test_save_lazy(self, sample_metadata_lazy, tmp_path):
        """Test saving lazy instance (should materialize)."""
        save_dir = tmp_path / "metadata_save_lazy"
        sample_metadata_lazy.save(save_dir)
        
        assert (save_dir / "metadata.csv").exists()
        assert (save_dir / "attributes.csv").exists()
        
    def test_load_eager(self, sample_metadata_eager, tmp_path):
        """Test save then load in eager mode."""
        save_dir = tmp_path / "metadata_roundtrip"
        sample_metadata_eager.save(save_dir)
        
        loaded = SampleMetadata.load(save_dir, lazy=False)
        
        assert loaded._is_lazy is False
        assert loaded.metadata.shape == sample_metadata_eager.metadata.shape
        assert loaded.attributes.shape == sample_metadata_eager.attributes.shape
        
    def test_load_lazy(self, sample_metadata_eager, tmp_path):
        """Test save then load in lazy mode."""
        save_dir = tmp_path / "metadata_roundtrip_lazy"
        sample_metadata_eager.save(save_dir)
        
        loaded = SampleMetadata.load(save_dir, lazy=True)
        
        assert loaded._is_lazy is True
        assert loaded._lf_metadata is not None
        assert isinstance(loaded._lf_metadata, pl.LazyFrame)
        
    def test_scan_alias(self, sample_metadata_eager, metadata_csv, attributes_csv):
        """Test scan loads lazily from separate CSV files."""
        scanned = SampleMetadata.scan(metadata_csv, attributes_csv)
        
        assert scanned._is_lazy is True
        assert scanned._lf_metadata is not None
        assert scanned.metadata is None


class TestSampleMetadataFiltering:
    """Test filtering methods preserve mode."""
    
    def test_filter_by_sample_eager(self, sample_metadata_eager):
        """Test filtering preserves eager mode."""
        samples_lf = pl.DataFrame({"sample": ["S1", "S3"]}).lazy()
        
        filtered = sample_metadata_eager._filter_by_sample(samples_lf)
        
        assert filtered._is_lazy is False
        assert filtered.metadata.height == 2
        assert set(filtered.metadata["sample"].to_list()) == {"S1", "S3"}
        
    def test_filter_by_sample_lazy(self, sample_metadata_lazy):
        """Test filtering preserves lazy mode."""
        samples_lf = pl.DataFrame({"sample": ["S2", "S4"]}).lazy()
        
        filtered = sample_metadata_lazy._filter_by_sample(samples_lf)
        
        assert filtered._is_lazy is True
        assert filtered._lf_metadata is not None
        
        # Materialize to check results
        filtered.materialize()
        assert filtered.metadata.height == 2
        assert set(filtered.metadata["sample"].to_list()) == {"S2", "S4"}


class TestSampleMetadataFieldValidation:
    """Test field validation and column name handling."""
    
    def test_required_metadata_fields(self, sample_metadata_eager):
        """Test metadata has required fields."""
        required = {"sample", "biosample", "bioproject", "lat", "lon", 
                   "collection_date", "biome", "mbases"}
        
        columns = set(sample_metadata_eager.metadata.columns)
        assert required.issubset(columns)
        
    def test_attributes_structure(self, sample_metadata_eager):
        """Test attributes has correct long-format structure."""
        assert "sample" in sample_metadata_eager.attributes.columns
        assert "key" in sample_metadata_eager.attributes.columns
        assert "value" in sample_metadata_eager.attributes.columns
