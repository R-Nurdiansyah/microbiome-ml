"""Shared pytest fixtures for microbiome_ml tests."""

import logging

import numpy as np
import polars as pl
import pytest

from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles

# Configure debug logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        "collection_date": [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
        ],
        "biome": ["soil", "marine", "soil", "freshwater"],
        "mbases": [1500, 2000, 1200, 1800],
    }


@pytest.fixture
def sample_attributes_data():
    """Sample attributes in long format."""
    return {
        "sample": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
        "key": ["pH", "temp", "pH", "temp", "pH", "temp", "pH", "temp"],
        "value": ["6.5", "25", "7.2", "18", "6.8", "22", "7.0", "20"],
    }


@pytest.fixture
def sample_study_titles_data():
    """Study titles mapping."""
    return {
        "sample": [
            "BP1",
            "BP2",
        ],  # Using 'sample' to match StudyMetadataFields expectation
        "study_title": ["Soil microbiome study", "Water microbiome study"],
        "abstract": [
            "Study of soil bacterial communities",
            "Study of aquatic microbiome",
        ],
    }


@pytest.fixture
def sample_profiles_data():
    """Taxonomic profiles with coverage (filled format) - includes all 7 standard ranks."""
    return {
        "sample": [
            "S1",
            "S1",
            "S1",
            "S1",
            "S1",
            "S1",
            "S1",
            "S1",
            "S2",
            "S2",
            "S2",
            "S2",
            "S2",
            "S2",
            "S2",
            "S2",
        ],
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
            "d__Bacteria;p__Actinobacteriota;c__Actinomycetes;o__Actinomycetales;f__Micrococcaceae;g__Micrococcus;s__luteus",
        ],
        "coverage": [
            150.0,
            100.0,
            100.0,
            90.0,
            80.0,
            50.0,
            30.0,
            25.0,
            140.0,
            80.0,
            80.0,
            70.0,
            60.0,
            40.0,
            25.0,
            30.0,
        ],
    }


@pytest.fixture
def sample_root_data():
    """Root coverage data."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "root_coverage": [150.0, 140.0, 130.0, 120.0],
    }


@pytest.fixture
def sample_feature_data():
    """Feature matrix in wide format."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "feature1": [0.5, 0.3, 0.6, 0.4],
        "feature2": [0.2, 0.4, 0.1, 0.3],
        "feature3": [0.3, 0.3, 0.3, 0.3],
    }


@pytest.fixture
def sample_labels_data():
    """Label data for classification/regression."""
    return {
        "sample": ["S1", "S2", "S3", "S4"],
        "target": [0, 1, 0, 1],
        "category": ["A", "B", "A", "B"],
    }


# CSV file fixtures - write data to temporary files


@pytest.fixture
def csv_factory(tmp_path):
    """Factory fixture to create CSV files from data."""

    def _create_csv(data, filename):
        path = tmp_path / filename
        pl.DataFrame(data).write_csv(path)
        return path

    return _create_csv


@pytest.fixture
def metadata_csv(csv_factory, sample_metadata_data):
    """Write metadata to CSV and return path."""
    return csv_factory(sample_metadata_data, "metadata.csv")


@pytest.fixture
def attributes_csv(csv_factory, sample_attributes_data):
    """Write attributes to CSV and return path."""
    return csv_factory(sample_attributes_data, "attributes.csv")


@pytest.fixture
def study_titles_csv(csv_factory, sample_study_titles_data):
    """Write study titles to CSV and return path."""
    return csv_factory(sample_study_titles_data, "study_titles.csv")


@pytest.fixture
def profiles_csv(csv_factory, sample_profiles_data):
    """Write profiles to CSV and return path."""
    return csv_factory(sample_profiles_data, "profiles.csv")


@pytest.fixture
def root_csv(csv_factory, sample_root_data):
    """Write root coverage to CSV and return path."""
    return csv_factory(sample_root_data, "root.csv")


@pytest.fixture
def features_csv(csv_factory, sample_feature_data):
    """Write features to CSV and return path."""
    return csv_factory(sample_feature_data, "features.csv")


@pytest.fixture
def labels_csv(csv_factory, sample_labels_data):
    """Write labels to CSV and return path."""
    return csv_factory(sample_labels_data, "labels.csv")


# Instance fixtures - SampleMetadata


@pytest.fixture(params=["eager", "lazy"])
def sample_metadata(request, metadata_csv, attributes_csv, study_titles_csv):
    """SampleMetadata instance in both eager and lazy modes."""
    if request.param == "eager":
        return SampleMetadata(metadata_csv, attributes_csv, study_titles_csv)
    return SampleMetadata.scan(metadata_csv, attributes_csv, study_titles_csv)


@pytest.fixture
def sample_metadata_eager(metadata_csv, attributes_csv, study_titles_csv):
    """SampleMetadata instance in eager mode."""
    return SampleMetadata(metadata_csv, attributes_csv, study_titles_csv)


@pytest.fixture
def sample_metadata_lazy(metadata_csv, attributes_csv, study_titles_csv):
    """SampleMetadata instance in lazy mode."""
    return SampleMetadata.scan(metadata_csv, attributes_csv, study_titles_csv)


# Instance fixtures - TaxonomicProfiles


@pytest.fixture(params=["eager", "lazy"])
def sample_profiles(request, profiles_csv, root_csv):
    """TaxonomicProfiles instance in both eager and lazy modes."""
    if request.param == "eager":
        return TaxonomicProfiles(
            profiles_csv, root=root_csv, check_filled=False
        )
    return TaxonomicProfiles.scan(
        profiles_csv, root=root_csv, check_filled=False
    )


@pytest.fixture
def sample_profiles_eager(profiles_csv, root_csv):
    """TaxonomicProfiles instance in eager mode."""
    return TaxonomicProfiles(profiles_csv, root=root_csv, check_filled=False)


@pytest.fixture
def sample_profiles_lazy(profiles_csv, root_csv):
    """TaxonomicProfiles instance in lazy mode."""
    return TaxonomicProfiles.scan(
        profiles_csv, root=root_csv, check_filled=False
    )


# Instance fixtures - FeatureSet


@pytest.fixture(params=["eager", "lazy"])
def sample_features(request, features_csv):
    """FeatureSet instance in both eager and lazy modes."""
    if request.param == "eager":
        return FeatureSet.load(features_csv)
    return FeatureSet.scan(features_csv, name="test_features")


@pytest.fixture
def sample_features_eager(features_csv):
    """FeatureSet instance in eager mode."""
    return FeatureSet.load(features_csv)


@pytest.fixture
def sample_features_lazy(features_csv):
    """FeatureSet instance in lazy mode."""
    return FeatureSet.scan(features_csv, name="test_features")


# Instance fixtures - Dataset


@pytest.fixture(params=["eager", "lazy"])
def sample_dataset(
    request,
    metadata_csv,
    attributes_csv,
    study_titles_csv,
    profiles_csv,
    root_csv,
):
    """Dataset instance with metadata and profiles in both eager and lazy
    modes."""
    mode = request.param
    if mode == "eager":
        meta = SampleMetadata(metadata_csv, attributes_csv, study_titles_csv)
        prof = TaxonomicProfiles(
            profiles_csv, root=root_csv, check_filled=False
        )
        return Dataset(metadata=meta, profiles=prof)
    else:
        meta = SampleMetadata.scan(
            metadata_csv, attributes_csv, study_titles_csv
        )
        prof = TaxonomicProfiles.scan(
            profiles_csv, root=root_csv, check_filled=False
        )
        return Dataset(metadata=meta, profiles=prof)


@pytest.fixture
def sample_dataset_eager(sample_metadata_eager, sample_profiles_eager):
    """Dataset instance with metadata and profiles in eager mode."""
    return Dataset(
        metadata=sample_metadata_eager, profiles=sample_profiles_eager
    )


@pytest.fixture
def sample_dataset_lazy(sample_metadata_lazy, sample_profiles_lazy):
    """Dataset instance with metadata and profiles in lazy mode."""
    return Dataset(
        metadata=sample_metadata_lazy, profiles=sample_profiles_lazy
    )


@pytest.fixture
def sample_dataset_empty():
    """Empty Dataset instance for builder pattern testing."""
    return Dataset()


@pytest.fixture
def sync_data_files(csv_factory):
    """Create data files for synchronization tests."""
    # Metadata: S1, S2, S3, S4
    metadata_data = {
        "sample": ["S1", "S2", "S3", "S4"],
        "biosample": ["BS1", "BS2", "BS3", "BS4"],
        "bioproject": ["BP1", "BP1", "BP1", "BP1"],
        "lat": [0.0, 0.0, 0.0, 0.0],
        "lon": [0.0, 0.0, 0.0, 0.0],
        "collection_date": ["2020-01-01"] * 4,
        "biome": ["soil"] * 4,
        "mbases": [1000] * 4,
    }
    metadata_csv = csv_factory(metadata_data, "metadata_sync.csv")

    attributes_data = {
        "sample": ["S1", "S2", "S3", "S4"],
        "key": ["pH"] * 4,
        "value": ["7.0"] * 4,
    }
    attributes_csv = csv_factory(attributes_data, "attributes_sync.csv")

    # Profiles: S1, S2, S3 (missing S4)
    profiles_data = {
        "sample": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "taxonomy": ["d__Bacteria", "d__Bacteria;p__Proteobacteria"] * 3,
        "coverage": [100.0, 50.0] * 3,
    }
    profiles_csv = csv_factory(profiles_data, "profiles_sync.csv")

    # Features: S1, S2 (missing S3, S4)
    features_data = {
        "sample": ["S1", "S2"],
        "feat1": [0.5, 0.3],
        "feat2": [0.2, 0.4],
    }
    features_csv = csv_factory(features_data, "features_sync.csv")

    # Labels: S1, S2, S3 (missing S4)
    labels_data = {"sample": ["S1", "S2", "S3"], "target": [0, 1, 0]}
    labels_csv = csv_factory(labels_data, "labels_sync.csv")

    return {
        "metadata": metadata_csv,
        "attributes": attributes_csv,
        "profiles": profiles_csv,
        "features": features_csv,
        "labels": labels_csv,
    }


# QC test fixtures - low quality samples


@pytest.fixture
def low_quality_metadata_data():
    """Metadata with samples below quality thresholds."""
    return {
        "sample": ["LQ1", "LQ2", "LQ3", "DOM1", "DOM2", "HQ1", "HQ2"],
        "biosample": [
            "BS_LQ1",
            "BS_LQ2",
            "BS_LQ3",
            "BS_DOM1",
            "BS_DOM2",
            "BS_HQ1",
            "BS_HQ2",
        ],
        "bioproject": ["BP1", "BP1", "BP2", "BP2", "BP2", "BP3", "BP3"],
        "lat": [15.0, 25.0, 35.0, 45.0, 50.0, 55.0, 60.0],
        "lon": [45.0, 55.0, 65.0, 75.0, 80.0, 85.0, 90.0],
        "collection_date": [
            "2020-05-01",
            "2020-06-01",
            "2020-07-01",
            "2020-08-01",
            "2020-08-15",
            "2020-09-01",
            "2020-09-15",
        ],
        "biome": [
            "soil",
            "marine",
            "soil",
            "marine",
            "marine",
            "freshwater",
            "freshwater",
        ],
        "mbases": [
            500,
            800,
            950,
            1500,
            1600,
            2000,
            2200,
        ],  # First 3 below 1000, DOM1/DOM2/HQ1/HQ2 above
    }


@pytest.fixture
def low_quality_metadata_csv(tmp_path, low_quality_metadata_data):
    """CSV file with low quality metadata."""
    csv_path = tmp_path / "low_quality_metadata.csv"
    pl.DataFrame(low_quality_metadata_data).write_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def low_quality_attributes_csv(tmp_path):
    """Attributes for low quality samples."""
    data = {
        "sample": [
            "LQ1",
            "LQ1",
            "LQ2",
            "LQ2",
            "LQ3",
            "LQ3",
            "DOM1",
            "DOM1",
            "DOM2",
            "DOM2",
            "HQ1",
            "HQ1",
            "HQ2",
            "HQ2",
        ],
        "key": ["pH", "temp"] * 7,
        "value": [
            "6.0",
            "20",
            "7.0",
            "21",
            "6.5",
            "22",
            "7.3",
            "25",
            "7.4",
            "26",
            "7.2",
            "23",
            "6.8",
            "24",
        ],
    }
    csv_path = tmp_path / "low_quality_attributes.csv"
    pl.DataFrame(data).write_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def low_coverage_profiles_data():
    """Profiles with samples below coverage threshold."""
    # Create profiles where some samples have low root coverage
    return {
        "sample": ["LC1", "LC1", "LC2", "LC2", "HC1", "HC1", "HC2", "HC2"] * 3,
        "taxonomy": [
            "d__Bacteria",
            "d__Bacteria;p__Proteobacteria",
            "d__Bacteria",
            "d__Bacteria;p__Firmicutes",
            "d__Bacteria",
            "d__Bacteria;p__Actinobacteria",
            "d__Bacteria",
            "d__Bacteria;p__Bacteroidetes",
        ]
        * 3,
        "relabund": [0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.8, 0.1] * 3,
    }


@pytest.fixture
def low_coverage_root_data():
    """Root coverage with some samples below threshold."""
    return {
        "sample": ["LC1", "LC2", "HC1", "HC2"],
        "root_coverage": [
            30.0,
            45.0,
            100.0,
            150.0,
        ],  # First 2 below 50 threshold
    }


@pytest.fixture
def dominated_profiles_data():
    """Profiles with some samples dominated by single taxon."""
    return {
        "sample": ["DOM1"] * 8 + ["DOM2"] * 8 + ["HQ1"] * 8 + ["HQ2"] * 8,
        "taxonomy": [
            "d__Bacteria",
            "d__Bacteria;p__Proteobacteria",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales",  # Dominant order
            "d__Bacteria;p__Firmicutes",
            "d__Bacteria;p__Firmicutes;c__Bacilli",
            "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales",
            "d__Bacteria;p__Actinobacteria",
        ]
        * 4,
        "relabund": (
            # DOM1: 99.5% dominated by Enterobacterales
            [1.0, 0.995, 0.995, 0.995, 0.003, 0.001, 0.001, 0.001]
            +
            # DOM2: 99% dominated by Lactobacillales
            [1.0, 0.01, 0.01, 0.01, 0.99, 0.99, 0.99, 0.0]
            +
            # HQ1: balanced (renamed from NORM1)
            [1.0, 0.4, 0.35, 0.35, 0.6, 0.5, 0.5, 0.0]
            +
            # HQ2: balanced (renamed from NORM2)
            [1.0, 0.5, 0.45, 0.45, 0.5, 0.4, 0.4, 0.0]
        ),
    }


@pytest.fixture
def dominated_profiles_csv(tmp_path, dominated_profiles_data):
    """CSV file with dominated samples."""
    csv_path = tmp_path / "dominated_profiles.csv"
    pl.DataFrame(dominated_profiles_data).write_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def dominated_root_csv(tmp_path):
    """Root coverage for dominated samples."""
    data = {
        "sample": ["DOM1", "DOM2", "HQ1", "HQ2"],
        "root_coverage": [100.0, 120.0, 110.0, 130.0],
    }
    csv_path = tmp_path / "dominated_root.csv"
    pl.DataFrame(data).write_csv(csv_path)
    return str(csv_path)


# Large integration test fixtures


@pytest.fixture
def large_metadata_data():
    """Large metadata with 100 samples and various QC issues."""
    np.random.seed(42)
    n_samples = 100

    return {
        "sample": [f"SAMPLE_{i:03d}" for i in range(n_samples)],
        "biosample": [f"BIOSAMPLE_{i:03d}" for i in range(n_samples)],
        "bioproject": [
            f"BP{i % 5}" for i in range(n_samples)
        ],  # 5 different projects
        "lat": np.random.uniform(-90, 90, n_samples).tolist(),
        "lon": np.random.uniform(-180, 180, n_samples).tolist(),
        "collection_date": [
            f"2020-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            for i in range(n_samples)
        ],
        "biome": np.random.choice(
            ["soil", "marine", "freshwater", "sediment"], n_samples
        ).tolist(),
        # Create varying quality: 30% below 1000, 70% above
        "mbases": (
            np.random.choice([500, 700, 900], size=30).tolist()
            + np.random.choice([1200, 1500, 2000, 2500], size=70).tolist()
        ),
    }


@pytest.fixture
def large_profiles_data():
    """Large profiles with 100 samples."""
    np.random.seed(42)
    n_samples = 100

    # Create taxonomic profiles for each sample
    taxa = [
        "d__Bacteria",
        "d__Bacteria;p__Proteobacteria",
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria",
        "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales",
        "d__Bacteria;p__Firmicutes",
        "d__Bacteria;p__Firmicutes;c__Bacilli",
        "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales",
        "d__Bacteria;p__Actinobacteria",
        "d__Bacteria;p__Actinobacteria;c__Actinobacteria",
        "d__Bacteria;p__Actinobacteria;c__Actinobacteria;o__Actinomycetales",
    ]

    samples = []
    taxonomies = []
    relabunds = []

    for i in range(n_samples):
        sample_id = f"SAMPLE_{i:03d}"

        # 20% of samples are dominated (> 99% single order)
        if i < 20:
            # Dominated samples
            abundances = [
                1.0,
                0.995,
                0.99,
                0.99,
                0.005,
                0.003,
                0.002,
                0.0,
                0.0,
                0.0,
            ]
        else:
            # Balanced samples - random but normalized
            abundances = np.random.dirichlet([1] * len(taxa))
            # Ensure parent >= sum of children for filled format
            abundances[0] = 1.0  # Root
            abundances[1] = abundances[2] + abundances[3] + 0.01
            abundances[4] = abundances[5] + abundances[6] + 0.01
            abundances[7] = abundances[8] + abundances[9] + 0.01
            abundances = abundances / abundances.sum()

        for tax, abund in zip(taxa, abundances):
            samples.append(sample_id)
            taxonomies.append(tax)
            relabunds.append(float(abund))

    return {"sample": samples, "taxonomy": taxonomies, "relabund": relabunds}


@pytest.fixture
def large_root_data():
    """Large root coverage with various quality levels."""
    np.random.seed(42)
    n_samples = 100

    # 25% below coverage threshold of 50
    low_cov = np.random.uniform(20, 49, 25).tolist()
    high_cov = np.random.uniform(60, 200, 75).tolist()

    return {
        "sample": [f"SAMPLE_{i:03d}" for i in range(n_samples)],
        "root_coverage": low_cov + high_cov,
    }
