"""Integration tests for end-to-end workflows."""

import pytest
import polars as pl
from pathlib import Path

from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.utils.taxonomy import TaxonomicRanks


@pytest.mark.integration
class TestLazyWorkflow:
    """Test complete lazy-first workflows."""
    
    def test_lazy_load_filter_save(self, metadata_csv, attributes_csv, profiles_csv, tmp_path):
        """Test lazy load → filter → save workflow."""
        # Lazy load
        dataset = (Dataset()
                  .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                  .add_profiles(profiles_csv, check_filled=False))
        
        # Verify lazy mode
        assert isinstance(dataset.metadata.metadata, pl.LazyFrame)
        
        # Save
        save_dir = tmp_path / "lazy_workflow"
        dataset.save(save_dir)
        
        # Reload lazily
        loaded = Dataset.scan(save_dir)
        assert isinstance(loaded.metadata.metadata, pl.LazyFrame)
        assert isinstance(loaded.profiles.profiles, pl.LazyFrame)
        
    def test_lazy_to_eager_workflow(self, metadata_csv, attributes_csv, profiles_csv, tmp_path):
        """Test lazy → process → collect → save workflow."""
        # Build and save dataset
        dataset = (Dataset()
                  .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                  .add_profiles(profiles_csv, check_filled=False))
        
        save_dir = tmp_path / "lazy_eager"
        dataset.save(save_dir)
        
        # Reload lazily
        lazy_dataset = Dataset.scan(save_dir)
        
        # collect manually
        meta_df = lazy_dataset.metadata.metadata.collect()
        prof_df = lazy_dataset.profiles.profiles.collect()
        
        assert isinstance(meta_df, pl.DataFrame)
        assert isinstance(prof_df, pl.DataFrame)


@pytest.mark.integration
class TestTarGzWorkflow:
    """Test tar.gz compression/extraction workflows."""
    
    def test_compress_extract_roundtrip(self, sample_dataset_eager, tmp_path):
        """Test complete tar.gz save/load cycle."""
        archive_path = tmp_path / "dataset.tar.gz"
        
        # Save compressed
        sample_dataset_eager.save(archive_path, compress=True)
        assert archive_path.exists()
        
        # Load from archive
        loaded = Dataset.load(archive_path, lazy=False)
        
        assert loaded.metadata is not None
        assert loaded.profiles is not None
        assert set(loaded._sample_ids) == set(sample_dataset_eager._sample_ids)
        
    def test_compress_without_extension(self, sample_dataset_eager, tmp_path):
        """Test compression adds .tar.gz extension."""
        save_path = tmp_path / "dataset"
        
        sample_dataset_eager.save(save_path, compress=True)
        
        # Should create dataset.tar.gz
        assert (tmp_path / "dataset.tar.gz").exists()
        
    def test_lazy_load_from_archive(self, sample_dataset_eager, tmp_path):
        """Test lazy loading from tar.gz."""
        archive_path = tmp_path / "dataset_lazy.tar.gz"
        
        sample_dataset_eager.save(archive_path, compress=True)
        
        loaded = Dataset.scan(archive_path)
        
        assert isinstance(loaded.metadata.metadata, pl.LazyFrame)
        assert isinstance(loaded.profiles.profiles, pl.LazyFrame)


@pytest.mark.integration
class TestMultiComponentSync:
    """Test synchronization across multiple components."""
    
    def test_sync_all_components(self, tmp_path):
        """Test sync with metadata, profiles, features, and labels."""
        # Create data with overlapping samples
        # Metadata: S1, S2, S3, S4
        metadata_df = pl.DataFrame({
            "sample": ["S1", "S2", "S3", "S4"],
            "biosample": ["BS1", "BS2", "BS3", "BS4"],
            "bioproject": ["BP1", "BP1", "BP1", "BP1"],
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lon": [0.0, 0.0, 0.0, 0.0],
            "collection_date": ["2020-01-01"] * 4,
            "biome": ["soil"] * 4,
            "mbases": [1000] * 4
        })
        metadata_csv = tmp_path / "metadata_sync.csv"
        metadata_df.write_csv(metadata_csv)
        
        attributes_df = pl.DataFrame({
            "sample": ["S1", "S2", "S3", "S4"],
            "key": ["pH"] * 4,
            "value": ["7.0"] * 4
        })
        attributes_csv = tmp_path / "attributes_sync.csv"
        attributes_df.write_csv(attributes_csv)
        
        # Profiles: S1, S2, S3 (missing S4)
        profiles_df = pl.DataFrame({
            "sample": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "taxonomy": ["d__Bacteria", "d__Bacteria;p__Proteobacteria"] * 3,
            "coverage": [100.0, 50.0] * 3
        })
        profiles_csv = tmp_path / "profiles_sync.csv"
        profiles_df.write_csv(profiles_csv)
        
        # Features: S1, S2 (missing S3, S4)
        features_df = pl.DataFrame({
            "sample": ["S1", "S2"],
            "feat1": [0.5, 0.3],
            "feat2": [0.2, 0.4]
        })
        features_csv = tmp_path / "features_sync.csv"
        features_df.write_csv(features_csv)
        
        # Labels: S1, S2, S3 (missing S4)
        labels_df = pl.DataFrame({
            "sample": ["S1", "S2", "S3"],
            "target": [0, 1, 0]
        })
        labels_csv = tmp_path / "labels_sync.csv"
        labels_df.write_csv(labels_csv)
        
        # Build dataset - should sync to intersection (S1, S2)
        dataset = (Dataset()
                  .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                  .add_profiles(profiles_csv, check_filled=False)
                  .add_feature_set(features_csv, name="features")
                  .add_labels(labels_csv, name="labels"))
        
        # Should only have S1, S2 (strict intersection)
        assert set(dataset._sample_ids) == {"S1", "S2"}
        
        # Verify all components filtered
        assert set(dataset.metadata.metadata.collect()["sample"].to_list()) == {"S1", "S2"}


@pytest.mark.integration
class TestManifestValidation:
    """Test manifest.json generation and validation."""
    
    def test_manifest_tracks_components(self, sample_dataset_eager, tmp_path):
        """Test manifest accurately tracks all components."""
        import json
        
        # Add features
        sample_dataset_eager.add_features("test_features", rank=TaxonomicRanks.GENUS)
        
        save_dir = tmp_path / "manifest_test"
        sample_dataset_eager.save(save_dir)
        
        with open(save_dir / "manifest.json", "r") as f:
            manifest = json.load(f)
            
        # Check components
        assert "metadata" in manifest["components"]
        assert "profiles" in manifest["components"]
        assert "features" in manifest["components"]
        assert "test_features" in manifest["components"]["features"]
        
    def test_manifest_sample_count(self, sample_dataset_eager, tmp_path):
        """Test manifest tracks sample counts correctly."""
        import json
        
        save_dir = tmp_path / "manifest_count"
        sample_dataset_eager.save(save_dir)
        
        with open(save_dir / "manifest.json", "r") as f:
            manifest = json.load(f)
            
        # Check sample counts
        metadata_count = manifest["components"]["metadata"]["n_samples"]
        sample_ids_count = len(manifest["sample_ids"])
        
        assert metadata_count == sample_ids_count
        
    def test_manifest_version_and_timestamp(self, sample_dataset_eager, tmp_path):
        """Test manifest includes version and creation timestamp."""
        import json
        from datetime import datetime
        
        save_dir = tmp_path / "manifest_version"
        sample_dataset_eager.save(save_dir)
        
        with open(save_dir / "manifest.json", "r") as f:
            manifest = json.load(f)
            
        assert manifest["version"] == "1.0"
        assert "created" in manifest
        
        # Verify timestamp is valid ISO format
        created_dt = datetime.fromisoformat(manifest["created"])
        assert isinstance(created_dt, datetime)


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test realistic end-to-end workflows."""
    
    def test_complete_ml_workflow(self, metadata_csv, attributes_csv, profiles_csv, tmp_path):
        """Test complete workflow: load → process → generate features → save → reload."""
        # Step 1: Load data
        dataset = (Dataset()
                  .add_metadata(metadata=metadata_csv, attributes=attributes_csv)
                  .add_profiles(profiles_csv, check_filled=False))
        
        # Step 2: Generate taxonomic features
        dataset.add_taxonomic_features(ranks=[TaxonomicRanks.GENUS, TaxonomicRanks.PHYLUM])
        
        # Step 3: Apply preprocessing
        dataset.apply_preprocessing()
        
        # Step 4: Save
        save_path = tmp_path / "ml_workflow.tar.gz"
        dataset.save(save_path, compress=True)
        
        # Step 5: Reload and verify
        loaded = Dataset.load(save_path, lazy=False)
        
        assert loaded.metadata is not None
        assert loaded.profiles is not None
        assert "tax_genus" in loaded.feature_sets
        assert "tax_phylum" in loaded.feature_sets
        
        # Verify sample consistency
        assert len(loaded._sample_ids) > 0
        
    def test_incremental_build_workflow(self, tmp_path):
        """Test incremental dataset building."""
        # Start empty
        dataset = Dataset()
        
        # Add components one by one
        metadata_df = pl.DataFrame({
            "sample": ["S1", "S2"],
            "biosample": ["BS1", "BS2"],
            "bioproject": ["BP1", "BP1"],
            "lat": [0.0, 0.0],
            "lon": [0.0, 0.0],
            "collection_date": ["2020-01-01", "2020-01-01"],
            "biome": ["soil", "soil"],
            "mbases": [1000, 1000]
        })
        metadata_csv = tmp_path / "meta_inc.csv"
        metadata_df.write_csv(metadata_csv)
        
        attributes_df = pl.DataFrame({
            "sample": ["S1", "S2"],
            "key": ["pH", "pH"],
            "value": ["7.0", "7.0"]
        })
        attributes_csv = tmp_path / "attr_inc.csv"
        attributes_df.write_csv(attributes_csv)
        
        dataset.add_metadata(metadata=metadata_csv, attributes=attributes_csv)
        assert dataset.metadata is not None
        
        # Add profiles with genus-level taxa
        profiles_df = pl.DataFrame({
            "sample": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "taxonomy": [
                "d__Bacteria",
                "d__Bacteria;p__Proteobacteria",
                "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia",
                "d__Bacteria",
                "d__Bacteria;p__Proteobacteria",
                "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia"
            ],
            "coverage": [100.0, 50.0, 30.0, 120.0, 60.0, 40.0]
        })
        profiles_csv = tmp_path / "prof_inc.csv"
        profiles_df.write_csv(profiles_csv)
        
        dataset.add_profiles(profiles_csv, check_filled=False)
        assert dataset.profiles is not None
        
        # Add features
        dataset.add_taxonomic_features(ranks=[TaxonomicRanks.GENUS])
        assert "tax_genus" in dataset.feature_sets
        
        # Verify all components present
        assert dataset.metadata is not None
        assert dataset.profiles is not None
        assert len(dataset.feature_sets) > 0
