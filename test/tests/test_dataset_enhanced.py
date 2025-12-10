"""Tests for enhanced Dataset functionality."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from microbiome_ml.core.config import AggregationMethod, WeightingStrategy
from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.wrangle.features import SpeciesFeatureSet
from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    # Create proper metadata with required fields
    metadata_df = pl.DataFrame(
        {
            "sample": ["sample1", "sample2", "sample3"],
            "biosample": ["BS1", "BS2", "BS3"],
            "bioproject": ["BP1", "BP1", "BP2"],
            "lat": [10.0, 20.0, 30.0],
            "lon": [40.0, 50.0, 60.0],
            "collection_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "biome": ["soil", "marine", "soil"],
            "mbases": [1500, 2000, 1200],
            "group": ["A", "B", "A"],
            "age": [25, 30, 35],
        }
    )

    attributes_df = pl.DataFrame(
        {
            "sample": [
                "sample1",
                "sample1",
                "sample2",
                "sample2",
                "sample3",
                "sample3",
            ],
            "key": ["group", "age", "group", "age", "group", "age"],
            "value": ["A", "25", "B", "30", "A", "35"],
        }
    )

    # Create taxonomic profiles with coverage data (will be converted to relabund automatically)
    profiles_df = pl.DataFrame(
        {
            "sample": [
                "sample1",
                "sample1",
                "sample2",
                "sample2",
                "sample3",
                "sample3",
            ],
            "taxonomy": [
                "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
                "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__acidophilus",
                "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
                "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__acidophilus",
                "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
                "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__acidophilus",
            ],
            "coverage": [70.0, 30.0, 40.0, 60.0, 80.0, 20.0],
        }
    )

    # Create temporary CSV files for SampleMetadata
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    metadata_csv = temp_dir / "metadata.csv"
    attributes_csv = temp_dir / "attributes.csv"

    metadata_df.write_csv(metadata_csv)
    attributes_df.write_csv(attributes_csv)

    sample_metadata = SampleMetadata(
        metadata=metadata_csv, attributes=attributes_csv
    )

    return {
        "metadata_df": metadata_df,
        "attributes_df": attributes_df,
        "profiles_df": profiles_df,
        "sample_metadata": sample_metadata,
        "species_features_df": pl.DataFrame(
            {
                "species": ["s__coli", "s__acidophilus"],
                "trait1": [1.5, 2.5],
                "trait2": [0.8, 1.2],
                "trait3": [3.0, 4.0],
            }
        ),
        "species_ids": ["s__coli", "s__acidophilus"],
        "feature_names": ["trait1", "trait2", "trait3"],
        "feature_array": np.array([[1.5, 0.8, 3.0], [2.5, 1.2, 4.0]]),
    }


@pytest.fixture
def base_dataset(sample_data, tmp_path):
    """Base dataset for testing."""
    # Write profiles to temporary file
    profiles_csv = tmp_path / "profiles.csv"
    sample_data["profiles_df"].write_csv(profiles_csv)

    # Create taxonomic profiles
    taxonomic_profiles = TaxonomicProfiles(
        profiles=profiles_csv,
        check_filled=True,  # Let it detect and handle the format properly
    )

    return Dataset(
        metadata=sample_data["sample_metadata"], profiles=taxonomic_profiles
    )


class TestAddSpeciesFeatures:
    """Test enhanced add_species_features method."""

    def test_add_from_dataframe(self, base_dataset, sample_data):
        """Test adding species features from DataFrame."""
        df = sample_data["species_features_df"]

        # Add species features from DataFrame
        dataset = base_dataset.add_species_features(
            "traits", data=df, accession_col="species"
        )

        # Check the feature set was added
        assert "traits" in dataset.species_feature_sets
        feature_set = dataset.species_feature_sets["traits"]

        assert feature_set.accessions == ["s__coli", "s__acidophilus"]
        assert feature_set.feature_names == ["trait1", "trait2", "trait3"]
        np.testing.assert_array_equal(
            feature_set.features, sample_data["feature_array"]
        )

    def test_add_from_lazyframe(self, base_dataset, sample_data):
        """Test adding species features from LazyFrame."""
        lf = sample_data["species_features_df"].lazy()

        dataset = base_dataset.add_species_features(
            "traits", data=lf, accession_col="species"
        )

        assert "traits" in dataset.species_feature_sets
        feature_set = dataset.species_feature_sets["traits"]
        assert feature_set.accessions == ["s__coli", "s__acidophilus"]

    def test_add_from_csv(self, base_dataset, sample_data):
        """Test adding species features from CSV file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            sample_data["species_features_df"].write_csv(f.name)
            csv_path = f.name

        try:
            dataset = base_dataset.add_species_features(
                "traits", data=csv_path, accession_col="species"
            )

            assert "traits" in dataset.species_feature_sets
            feature_set = dataset.species_feature_sets["traits"]
            assert len(feature_set.accessions) == 2
            assert len(feature_set.feature_names) == 3
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_add_from_path_object(self, base_dataset, sample_data):
        """Test adding species features from Path object."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            sample_data["species_features_df"].write_csv(f.name)
            csv_path = Path(f.name)

        try:
            dataset = base_dataset.add_species_features(
                "traits", data=csv_path
            )
            assert "traits" in dataset.species_feature_sets
        finally:
            csv_path.unlink(missing_ok=True)

    def test_custom_accession_column(self, base_dataset, sample_data):
        """Test using custom accession column name."""
        # Create DataFrame with different column name
        df = sample_data["species_features_df"].rename({"species": "organism"})

        dataset = base_dataset.add_species_features(
            "traits", data=df, accession_col="organism"
        )

        assert "traits" in dataset.species_feature_sets
        feature_set = dataset.species_feature_sets["traits"]
        assert feature_set.accessions == ["s__coli", "s__acidophilus"]

    def test_missing_accession_column_error(self, base_dataset, sample_data):
        """Test error when accession column is missing."""
        df = sample_data["species_features_df"]

        with pytest.raises(
            ValueError, match="Accession column 'organism' not found in data"
        ):
            base_dataset.add_species_features(
                "traits", data=df, accession_col="organism"
            )

    def test_no_feature_columns_error(self, base_dataset):
        """Test error when no feature columns found."""
        df = pl.DataFrame({"species": ["sp1", "sp2"]})  # Only accession column

        with pytest.raises(
            ValueError, match="No feature columns found in data"
        ):
            base_dataset.add_species_features("traits", data=df)

    def test_unsupported_data_type_error(self, base_dataset):
        """Test error for unsupported data type."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            base_dataset.add_species_features(
                "traits", data=["invalid", "data"]
            )

    def test_traditional_methods_still_work(self, base_dataset, sample_data):
        """Test that traditional parameter methods still work."""
        # From arrays
        dataset = base_dataset.add_species_features(
            "traits",
            species_ids=sample_data["species_ids"],
            feature_names=sample_data["feature_names"],
            feature_array=sample_data["feature_array"],
        )

        assert "traits" in dataset.species_feature_sets

        # From SpeciesFeatureSet
        species_fs = SpeciesFeatureSet(
            accessions=sample_data["species_ids"],
            feature_names=sample_data["feature_names"],
            features=sample_data["feature_array"],
            name="traits2",
        )

        dataset.add_species_features("traits2", species_features=species_fs)
        assert "traits2" in dataset.species_feature_sets

    def test_insufficient_parameters_error(self, base_dataset):
        """Test error when insufficient parameters provided."""
        with pytest.raises(ValueError, match="Must provide"):
            base_dataset.add_species_features("traits")


class TestAggregateSpeciesToSamples:
    """Test enhanced aggregate_species_to_samples method."""

    @pytest.fixture
    def dataset_with_species_features(self, base_dataset, sample_data):
        """Dataset with species features added."""
        # Create different features for traits2
        traits2_df = pl.DataFrame(
            {
                "species": ["s__coli", "s__acidophilus"],
                "trait1": [3.0, 5.0],  # Double the values manually
                "trait2": [1.6, 2.4],
                "trait3": [6.0, 8.0],
            }
        )

        return base_dataset.add_species_features(
            "traits1",
            species_ids=sample_data["species_ids"],
            feature_names=sample_data["feature_names"],
            feature_array=sample_data["feature_array"],
        ).add_species_features(
            "traits2", data=traits2_df, accession_col="species"
        )

    def test_single_aggregation_with_defaults(
        self, dataset_with_species_features
    ):
        """Test single aggregation with default parameters."""
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            species_feature_name="traits1"
        )

        # Should create feature set with auto-generated name
        expected_name = "traits1_arithmetic_mean_none"
        assert expected_name in dataset.feature_sets

    def test_single_aggregation_with_custom_name(
        self, dataset_with_species_features
    ):
        """Test single aggregation with custom output name."""
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            species_feature_name="traits1", output_name="custom_name"
        )

        assert "custom_name" in dataset.feature_sets

    def test_create_all_combinations(self, dataset_with_species_features):
        """Test creating all possible combinations."""
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            create_all=True
        )

        # Count created aggregations
        aggregated_features = [
            name
            for name in dataset.feature_sets.keys()
            if name.startswith(("traits1_", "traits2_"))
        ]

        # Should have many combinations (exact count depends on implementation)
        assert len(aggregated_features) >= 15  # At least 15 combinations

        # Check some specific combinations exist
        assert "traits1_arithmetic_mean_none" in dataset.feature_sets
        assert "traits1_geometric_mean_abundance" in dataset.feature_sets
        assert "traits2_median_sqrt_abundance" in dataset.feature_sets

    def test_create_all_for_specific_feature(
        self, dataset_with_species_features
    ):
        """Test creating all combinations for specific feature set."""
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            species_feature_name="traits1", create_all=True
        )

        # Should only create combinations for traits1
        traits1_features = [
            name
            for name in dataset.feature_sets.keys()
            if name.startswith("traits1_")
        ]
        traits2_features = [
            name
            for name in dataset.feature_sets.keys()
            if name.startswith("traits2_")
        ]

        assert len(traits1_features) >= 10  # Many combinations for traits1
        assert len(traits2_features) == 0  # No combinations for traits2

    def test_auto_create_all_when_no_args(self, dataset_with_species_features):
        """Test auto create_all when called with no arguments."""
        dataset = dataset_with_species_features.aggregate_species_to_samples()

        # Should automatically create all combinations
        aggregated_features = [
            name
            for name in dataset.feature_sets.keys()
            if name.startswith(("traits1_", "traits2_"))
        ]
        assert len(aggregated_features) >= 15

    def test_invalid_combinations_skipped(self, dataset_with_species_features):
        """Test that invalid combinations are skipped."""
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            create_all=True
        )

        # These combinations should not exist (presence_absence with weighting)
        assert "traits1_presence_absence_abundance" not in dataset.feature_sets
        assert (
            "traits1_presence_absence_sqrt_abundance"
            not in dataset.feature_sets
        )

    def test_existing_features_skipped(self, dataset_with_species_features):
        """Test that existing feature sets are skipped."""
        # Create one manually first
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            species_feature_name="traits1",
            output_name="traits1_arithmetic_mean_none",
        )

        initial_count = len(dataset.feature_sets)

        # Now create all - should skip the existing one
        dataset.aggregate_species_to_samples(create_all=True)

        # Should have more features but not duplicate the existing one
        assert len(dataset.feature_sets) > initial_count

        # Verify the original one still exists and wasn't overwritten
        assert "traits1_arithmetic_mean_none" in dataset.feature_sets

    def test_error_handling_in_batch_creation(
        self, dataset_with_species_features, monkeypatch
    ):
        """Test error handling during batch creation."""
        # Mock the aggregation to fail for certain combinations
        original_aggregate = (
            dataset_with_species_features.aggregate_species_to_samples
        )

        def mock_aggregate(*args, **kwargs):
            # Fail if method is geometric_mean
            if kwargs.get("method") == AggregationMethod.GEOMETRIC_MEAN:
                raise ValueError("Mock error")
            return original_aggregate(*args, **kwargs)

        monkeypatch.setattr(
            dataset_with_species_features,
            "aggregate_species_to_samples",
            mock_aggregate,
        )

        # Should handle errors gracefully and continue
        dataset = dataset_with_species_features._create_all_aggregations()

        # Should have created some features despite errors
        assert len(dataset.feature_sets) > 0

    def test_traditional_required_parameters_still_work(
        self, dataset_with_species_features
    ):
        """Test that traditional required parameters still work."""
        dataset = dataset_with_species_features.aggregate_species_to_samples(
            species_feature_name="traits1",
            output_name="manual_aggregation",
            method=AggregationMethod.MEDIAN,
            weighting=WeightingStrategy.ABUNDANCE,
        )

        assert "manual_aggregation" in dataset.feature_sets

    def test_missing_species_feature_error(
        self, dataset_with_species_features
    ):
        """Test error when species feature set not found."""
        with pytest.raises(
            ValueError, match="Species feature set 'nonexistent' not found"
        ):
            dataset_with_species_features.aggregate_species_to_samples(
                species_feature_name="nonexistent"
            )

    def test_missing_profiles_error(self, sample_data):
        """Test error when taxonomic profiles are missing."""
        dataset = Dataset(metadata=sample_data["sample_metadata"])

        with pytest.raises(
            ValueError, match="TaxonomicProfiles required for aggregation"
        ):
            dataset.aggregate_species_to_samples(
                species_feature_name="traits1"
            )

    def test_no_species_features_warning(self, base_dataset):
        """Test warning when no species feature sets exist."""
        # Should handle gracefully and return self
        dataset = base_dataset._create_all_aggregations()
        assert dataset is base_dataset
        assert len(dataset.feature_sets) == 0


class TestIntegration:
    """Integration tests for the enhanced functionality."""

    def test_complete_workflow_with_dataframe(self, sample_data):
        """Test complete workflow using DataFrame input."""
        dataset = (
            Dataset(
                metadata=sample_data["sample_metadata"],
                profiles=sample_data["profiles_df"].lazy(),
            )
            .add_species_features(
                "traits", data=sample_data["species_features_df"]
            )
            .aggregate_species_to_samples(create_all=True)
        )

        # Should have many aggregated feature sets
        assert len(dataset.feature_sets) >= 15

        # Check that we can access the features
        for name, feature_set in dataset.feature_sets.items():
            assert hasattr(feature_set, "features")
            assert hasattr(feature_set, "feature_names")
            # Convert LazyFrame to DataFrame to check shape
            features_df = (
                feature_set.features.collect()
                if hasattr(feature_set.features, "collect")
                else feature_set.features
            )
            assert len(features_df.shape) == 2  # samples x features

    def test_chaining_multiple_species_features(self, sample_data):
        """Test chaining multiple species feature additions."""
        df1 = sample_data["species_features_df"]
        df2 = (
            sample_data["species_features_df"]
            .select("species")
            .with_columns(
                pl.lit(1.0).alias("new_trait1"),
                pl.lit(2.0).alias("new_trait2"),
            )
        )

        dataset = (
            Dataset(
                metadata=sample_data["sample_metadata"],
                profiles=sample_data["profiles_df"].lazy(),
            )
            .add_species_features("traits1", data=df1)
            .add_species_features("traits2", data=df2)
            .aggregate_species_to_samples("traits1", "aggregated_traits1")
            .aggregate_species_to_samples("traits2", "aggregated_traits2")
        )

        assert "traits1" in dataset.species_feature_sets
        assert "traits2" in dataset.species_feature_sets
        assert "aggregated_traits1" in dataset.feature_sets
        assert "aggregated_traits2" in dataset.feature_sets

    def test_mixed_input_methods(self, base_dataset, sample_data):
        """Test mixing different input methods."""
        # Add via DataFrame
        df = sample_data["species_features_df"]
        dataset = base_dataset.add_species_features("from_df", data=df)

        # Add via traditional arrays
        dataset.add_species_features(
            "from_arrays",
            species_ids=sample_data["species_ids"],
            feature_names=["manual_trait"],
            feature_array=np.array([[1.0], [2.0]]),
        )

        # Add via SpeciesFeatureSet
        feature_set = SpeciesFeatureSet(
            accessions=sample_data["species_ids"],
            feature_names=["fs_trait"],
            features=np.array([[3.0], [4.0]]),
            name="from_fs",
        )
        dataset.add_species_features("from_fs", species_features=feature_set)

        assert len(dataset.species_feature_sets) == 3

        # Create aggregations for all
        dataset.aggregate_species_to_samples(create_all=True)

        # Should have aggregations from all three feature sets
        aggregated_names = list(dataset.feature_sets.keys())
        assert any("from_df_" in name for name in aggregated_names)
        assert any("from_arrays_" in name for name in aggregated_names)
        assert any("from_fs_" in name for name in aggregated_names)
