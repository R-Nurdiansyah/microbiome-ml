"""Shared pytest fixtures for microbiome_ml tests."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.wrangle.samples import Dataset


# Data fixtures - small synthetic datasets

@pytest.fixture
def sample_metadata_data():
    """Small metadata DataFrame for testing."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "biosample": ["BS1", "BS2", "BS3", "BS4"],
        "bioproject": ["BP1", "BP1", "BP2", "BP2"],
        "lat": [10.0, 20.0, 30.0, 40.0],
        "lon": [40.0, 50.0, 60.0, 70.0],
        "collection_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "biome": ["soil", "marine", "soil", "freshwater"],
        "mbases": [1500, 2000, 1200, 1800]
    }


@pytest.fixture
def sample_attributes_data():
    """Sample attributes in long format."""
    return {
        "sample": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
        "key": ["pH", "temp", "pH", "temp", "pH", "temp", "pH", "temp"],
        "value": ["6.5", "25", "7.2", "18", "6.8", "22", "7.0", "20"]
    }


@pytest.fixture
def sample_study_titles_data():
    """Study titles mapping."""
    return {
        "sample": ["BP1", "BP2"],  # Using 'sample' to match StudyMetadataFields expectation
        "study_title": ["Soil microbiome study", "Water microbiome study"],
        "abstract": ["Study of soil bacterial communities", "Study of aquatic microbiome"]
    }


@pytest.fixture
def sample_profiles_data():
    """Taxonomic profiles with coverage (filled format) - includes all 7 standard ranks."""
    return {
        "sample": ["S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2", "S2", "S2", "S2", "S2"],
        "taxonomy": [
            # Domain
            "d__Bacteria",
            # Phylum
            "d__Bacteria;p__Proteobacteria",
            # Class
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria",
            # Order
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales",
            # Family
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae",
            # Genus
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia",
            # Species
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
            # Another branch
            "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__acidophilus",
            # Sample 2
            "d__Bacteria",
            "d__Bacteria;p__Proteobacteria",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
            "d__Bacteria;p__Actinobacteriota;c__Actinomycetes;o__Actinomycetales;f__Micrococcaceae;g__Micrococcus;s__luteus"
        ],
        "coverage": [150.0, 100.0, 100.0, 90.0, 80.0, 50.0, 30.0, 25.0, 
                     140.0, 80.0, 80.0, 70.0, 60.0, 40.0, 25.0, 30.0]
    }


@pytest.fixture
def sample_root_data():
    """Root coverage data."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "root_coverage": [150.0, 140.0, 130.0, 120.0]
    }


@pytest.fixture
def sample_feature_data():
    """Feature matrix in wide format."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "feature1": [0.5, 0.3, 0.6, 0.4],
        "feature2": [0.2, 0.4, 0.1, 0.3],
        "feature3": [0.3, 0.3, 0.3, 0.3]
    }


@pytest.fixture
def sample_labels_data():
    """Label data for classification/regression."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "target": [0, 1, 0, 1],
        "category": ["A", "B", "A", "B"]
    }


# CSV file fixtures - write data to temporary files

@pytest.fixture
def metadata_csv(tmp_path, sample_metadata_data):
    """Write metadata to CSV and return path."""
    path = tmp_path / "metadata.csv"
    pl.DataFrame(sample_metadata_data).write_csv(path)
    return path


@pytest.fixture
def attributes_csv(tmp_path, sample_attributes_data):
    """Write attributes to CSV and return path."""
    path = tmp_path / "attributes.csv"
    pl.DataFrame(sample_attributes_data).write_csv(path)
    return path


@pytest.fixture
def study_titles_csv(tmp_path, sample_study_titles_data):
    """Write study titles to CSV and return path."""
    path = tmp_path / "study_titles.csv"
    pl.DataFrame(sample_study_titles_data).write_csv(path)
    return path


@pytest.fixture
def profiles_csv(tmp_path, sample_profiles_data):
    """Write profiles to CSV and return path."""
    path = tmp_path / "profiles.csv"
    pl.DataFrame(sample_profiles_data).write_csv(path)
    return path


@pytest.fixture
def root_csv(tmp_path, sample_root_data):
    """Write root coverage to CSV and return path."""
    path = tmp_path / "root.csv"
    pl.DataFrame(sample_root_data).write_csv(path)
    return path


@pytest.fixture
def features_csv(tmp_path, sample_feature_data):
    """Write features to CSV and return path."""
    path = tmp_path / "features.csv"
    pl.DataFrame(sample_feature_data).write_csv(path)
    return path


@pytest.fixture
def labels_csv(tmp_path, sample_labels_data):
    """Write labels to CSV and return path."""
    path = tmp_path / "labels.csv"
    pl.DataFrame(sample_labels_data).write_csv(path)
    return path


# Instance fixtures - SampleMetadata

@pytest.fixture
def sample_metadata_eager(metadata_csv, attributes_csv, study_titles_csv):
    """SampleMetadata instance in eager mode."""
    return SampleMetadata(metadata_csv, attributes_csv, study_titles_csv)


@pytest.fixture
def sample_metadata_lazy(metadata_csv, attributes_csv, study_titles_csv):
    """SampleMetadata instance in lazy mode."""
    return SampleMetadata.scan(metadata_csv, attributes_csv, study_titles_csv)


# Instance fixtures - TaxonomicProfiles

@pytest.fixture
def sample_profiles_eager(profiles_csv, root_csv):
    """TaxonomicProfiles instance in eager mode."""
    return TaxonomicProfiles(profiles_csv, root=root_csv, check_filled=False)


@pytest.fixture
def sample_profiles_lazy(profiles_csv, root_csv):
    """TaxonomicProfiles instance in lazy mode."""
    return TaxonomicProfiles.scan(profiles_csv, root=root_csv, check_filled=False)


# Instance fixtures - FeatureSet

@pytest.fixture
def sample_features_eager(features_csv):
    """FeatureSet instance in eager mode."""
    return FeatureSet.load(features_csv)


@pytest.fixture
def sample_features_lazy(features_csv):
    """FeatureSet instance in lazy mode."""
    return FeatureSet.scan(features_csv, name="test_features")


# Instance fixtures - Dataset

@pytest.fixture
def sample_dataset_eager(sample_metadata_eager, sample_profiles_eager):
    """Dataset instance with metadata and profiles in eager mode."""
    return Dataset(metadata=sample_metadata_eager, profiles=sample_profiles_eager)


@pytest.fixture
def sample_dataset_lazy(sample_metadata_lazy, sample_profiles_lazy):
    """Dataset instance with metadata and profiles in lazy mode."""
    return Dataset(metadata=sample_metadata_lazy, profiles=sample_profiles_lazy)


@pytest.fixture
def sample_dataset_empty():
    """Empty Dataset instance for builder pattern testing."""
    return Dataset()
