"""Tests for SampleMetadata class."""

import pytest
import polars as pl
from pathlib import Path

from microbiome_ml.wrangle.metadata import SampleMetadata


class TestSampleMetadataInitialization:
    """Test SampleMetadata initialization and modes."""
    
    def test_eager_initialization(self, metadata_csv, attributes_csv):
        """Test initialization from CSV files."""
        meta = SampleMetadata(metadata_csv, attributes_csv)
        
        assert meta.metadata is not None
        assert meta.attributes is not None
        assert isinstance(meta.metadata, pl.LazyFrame)
        assert isinstance(meta.attributes, pl.LazyFrame)
        assert meta.metadata.collect().height == 4
        
    def test_lazy_initialization(self, metadata_csv, attributes_csv):
        """Test lazy mode initialization via scan."""
        meta = SampleMetadata.scan(metadata_csv, attributes_csv)
        
        assert isinstance(meta.metadata, pl.LazyFrame)
        assert isinstance(meta.attributes, pl.LazyFrame)
        
    def test_with_study_titles(self, metadata_csv, attributes_csv, study_titles_csv):
        """Test initialization with optional study titles."""
        meta = SampleMetadata(metadata_csv, attributes_csv, study_titles_csv)
        
        assert meta.study_titles is not None
        assert isinstance(meta.study_titles, pl.LazyFrame)
        assert meta.study_titles.collect().height == 2
        
    def test_without_study_titles(self, metadata_csv, attributes_csv):
        """Test initialization without study titles."""
        meta = SampleMetadata(metadata_csv, attributes_csv)
        
        assert meta.study_titles is None


class TestSampleMetadataSaveLoad:
    """Test save/load round-trips."""
    
    def test_save_eager(self, sample_metadata_eager, tmp_path):
        """Test saving instance."""
        save_dir = tmp_path / "metadata_save"
        sample_metadata_eager.save(save_dir)
        
        assert (save_dir / "metadata.csv").exists()
        assert (save_dir / "attributes.csv").exists()
        assert (save_dir / "study_titles.csv").exists()
        
    def test_save_lazy(self, sample_metadata_lazy, tmp_path):
        """Test saving lazy instance (should collect)."""
        save_dir = tmp_path / "metadata_save_lazy"
        sample_metadata_lazy.save(save_dir)
        
        assert (save_dir / "metadata.csv").exists()
        assert (save_dir / "attributes.csv").exists()
        
    def test_load_eager(self, sample_metadata_eager, tmp_path):
        """Test save then load."""
        save_dir = tmp_path / "metadata_roundtrip"
        sample_metadata_eager.save(save_dir)
        
        loaded = SampleMetadata.load(save_dir)
        
        assert loaded.metadata.collect().shape == sample_metadata_eager.metadata.collect().shape
        assert loaded.attributes.collect().shape == sample_metadata_eager.attributes.collect().shape
        
    def test_load_lazy(self, sample_metadata_eager, tmp_path):
        """Test save then load."""
        save_dir = tmp_path / "metadata_roundtrip_lazy"
        sample_metadata_eager.save(save_dir)
        
        loaded = SampleMetadata.load(save_dir)
        
        assert isinstance(loaded.metadata, pl.LazyFrame)
        
    def test_scan_alias(self, sample_metadata_eager, metadata_csv, attributes_csv):
        """Test scan loads lazily from separate CSV files."""
        scanned = SampleMetadata.scan(metadata_csv, attributes_csv)
        
        assert isinstance(scanned.metadata, pl.LazyFrame)


class TestSampleMetadataFiltering:
    """Test filtering methods preserve mode."""
    
    def test_filter_by_sample_eager(self, sample_metadata_eager):
        """Test filtering."""
        samples_lf = pl.DataFrame({"sample": ["S1", "S3"]}).lazy()
        
        filtered = sample_metadata_eager._filter_by_sample(samples_lf)
        
        assert isinstance(filtered.metadata, pl.LazyFrame)
        assert filtered.metadata.collect().height == 2
        assert set(filtered.metadata.collect()["sample"].to_list()) == {"S1", "S3"}
        
    def test_filter_by_sample_lazy(self, sample_metadata_lazy):
        """Test filtering lazy instance."""
        samples_lf = pl.DataFrame({"sample": ["S2", "S4"]}).lazy()
        
        filtered = sample_metadata_lazy._filter_by_sample(samples_lf)
        
        assert isinstance(filtered.metadata, pl.LazyFrame)
        
        # collect to check results
        df = filtered.metadata.collect()
        assert df.height == 2
        assert set(df["sample"].to_list()) == {"S2", "S4"}


class TestSampleMetadataFieldValidation:
    """Test field validation and column name handling."""
    
    def test_required_metadata_fields(self, sample_metadata_eager):
        """Test metadata has required fields."""
        required = {"sample", "biosample", "bioproject", "lat", "lon", 
                   "collection_date", "biome", "mbases"}
        
        columns = set(sample_metadata_eager.metadata.collect_schema().names())
        assert required.issubset(columns)
        
    def test_attributes_structure(self, sample_metadata_eager):
        """Test attributes has correct long-format structure."""
        columns = sample_metadata_eager.attributes.collect_schema().names()
        assert "sample" in columns
        assert "key" in columns
        assert "value" in columns


class TestSampleMetadataQC:
    """Test quality control methods."""
    
    def test_default_qc_default_behavior(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc filters samples with mbases < 1000."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        
        # Before QC: 7 samples (3 low quality, 4 high quality)
        assert meta.metadata.collect().height == 7
        
        qc_meta = meta.default_qc()
        
        # After QC: only 2 high quality samples remain
        assert qc_meta.metadata.collect().height == 4
        assert set(qc_meta.metadata.collect()["sample"].to_list()) == {"DOM1", "DOM2", "HQ1", "HQ2"}
        
    def test_default_qc_custom_cutoff(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc with custom mbp_cutoff."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        
        # Custom cutoff of 900 - should keep LQ3 (950 mbases)
        qc_meta = meta.default_qc(mbp_cutoff=900)
        
        assert qc_meta.metadata.collect().height == 5
        assert set(qc_meta.metadata.collect()["sample"].to_list()) == {"DOM1", "DOM2", "LQ3", "HQ1", "HQ2"}
        
    def test_default_qc_preserves_eager_mode(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc preserves eager mode."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        
        qc_meta = meta.default_qc()
        
        assert isinstance(qc_meta.metadata, pl.LazyFrame)
        
    def test_default_qc_preserves_lazy_mode(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc preserves lazy mode."""
        meta = SampleMetadata.scan(low_quality_metadata_csv, low_quality_attributes_csv)
        
        qc_meta = meta.default_qc()
        
        assert isinstance(qc_meta.metadata, pl.LazyFrame)
        
    def test_default_qc_immutability(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc returns new instance without modifying original."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        original_height = meta.metadata.collect().height
        
        qc_meta = meta.default_qc()
        
        # Original unchanged
        assert meta.metadata.collect().height == original_height
        # New instance is different
        assert qc_meta.metadata.collect().height != original_height
        assert qc_meta is not meta
        
    def test_default_qc_all_samples_filtered(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc when all samples are below threshold."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        
        # Very high cutoff - filters all samples
        qc_meta = meta.default_qc(mbp_cutoff=10000)
        
        # TODO: add syncing + checks for removal of all samples
        assert qc_meta.metadata.collect().height == 0
        assert qc_meta.attributes.collect().height == 0
        
    def test_default_qc_filters_all_components(self, low_quality_metadata_csv, low_quality_attributes_csv):
        """Test default_qc filters metadata, attributes, and study_titles consistently."""
        meta = SampleMetadata(low_quality_metadata_csv, low_quality_attributes_csv)
        
        qc_meta = meta.default_qc()
        
        # Check all components have same samples
        meta_samples = set(qc_meta.metadata.collect()["sample"].to_list())
        attr_samples = set(qc_meta.attributes.collect()["sample"].unique().to_list())
        
        assert meta_samples == attr_samples
