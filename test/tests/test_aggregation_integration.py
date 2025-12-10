"""Test aggregation system integration."""

import numpy as np
import polars as pl
import pytest

from microbiome_ml.core.config import (
    AggregationConfig,
    AggregationMethod,
    WeightingStrategy,
)
from microbiome_ml.wrangle.aggregation import FeatureAggregator
from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.wrangle.features import FeatureSet, SpeciesFeatureSet
from microbiome_ml.wrangle.profiles import TaxonomicProfiles


@pytest.fixture
def sample_aggregation_data():
    """Create test taxonomic profiles and species features."""

    # Create taxonomic profiles (4 samples, 6 species)
    samples = ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
    species = [
        "Bacteroides_fragilis",
        "Escherichia_coli",
        "Lactobacillus_acidophilus",
        "Bifidobacterium_longum",
        "Clostridium_difficile",
        "Akkermansia_muciniphila",
    ]

    # Abundance matrix (samples x species)
    abundance_data = np.array(
        [
            [0.30, 0.25, 0.20, 0.15, 0.05, 0.05],  # Sample_1
            [0.20, 0.35, 0.15, 0.20, 0.05, 0.05],  # Sample_2
            [0.25, 0.20, 0.30, 0.10, 0.10, 0.05],  # Sample_3
            [0.15, 0.15, 0.25, 0.25, 0.15, 0.05],  # Sample_4
        ]
    )

    # Create profiles DataFrame
    profile_df = pl.DataFrame(
        {
            "sample": samples * len(species),
            "taxonomy": species * len(samples),
            "abundance": abundance_data.flatten(),
        }
    )

    # Create species features (6 species, 3 features)
    feature_names = [
        "enzyme_activity",
        "metabolite_production",
        "biofilm_formation",
    ]
    species_features_data = np.array(
        [
            [0.8, 0.6, 0.9],  # Bacteroides_fragilis
            [0.9, 0.8, 0.7],  # Escherichia_coli
            [0.7, 0.9, 0.4],  # Lactobacillus_acidophilus
            [0.6, 0.7, 0.5],  # Bifidobacterium_longum
            [0.5, 0.4, 0.8],  # Clostridium_difficile
            [0.8, 0.5, 0.3],  # Akkermansia_muciniphila
        ]
    )

    return {
        "profile_df": profile_df,
        "species": species,
        "species_features_data": species_features_data,
        "feature_names": feature_names,
        "samples": samples,
    }


class TestAggregationIntegration:
    """Test aggregation system integration."""

    def test_species_feature_creation(self, sample_aggregation_data):
        """Test creating SpeciesFeatureSet."""
        data = sample_aggregation_data

        # Create SpeciesFeatureSet
        species_fs = SpeciesFeatureSet(
            accessions=data["species"],
            feature_names=data["feature_names"],
            features=data["species_features_data"],
            name="test_species_features",
        )

        assert len(species_fs.accessions) == 6
        assert len(species_fs.feature_names) == 3
        assert species_fs.features.shape == (6, 3)
        assert species_fs.accessions == data["species"]
        assert species_fs.feature_names == data["feature_names"]

    def test_aggregation_direct(self, sample_aggregation_data):
        """Test direct aggregation using FeatureAggregator."""
        data = sample_aggregation_data

        # Create profiles and species features
        profiles = TaxonomicProfiles(data["profile_df"], check_filled=False)
        species_fs = SpeciesFeatureSet(
            accessions=data["species"],
            feature_names=data["feature_names"],
            features=data["species_features_data"],
            name="test_species_features",
        )

        # Test different aggregation methods
        methods_to_test = [
            AggregationMethod.ARITHMETIC_MEAN,
            AggregationMethod.GEOMETRIC_MEAN,
            AggregationMethod.PRESENCE_ABSENCE,
        ]

        for method in methods_to_test:
            # Configure based on method
            if method == AggregationMethod.TOP_K_ABUNDANT:
                weighting = WeightingStrategy.ABUNDANCE
            else:
                weighting = WeightingStrategy.NONE

            config = AggregationConfig(
                method=method,
                weighting=weighting,
                k=3 if method == AggregationMethod.TOP_K_ABUNDANT else 10,
            )

            aggregator = FeatureAggregator()
            sample_fs = aggregator.get_sample_features(
                species_fs, profiles.profiles, config
            )

            # Verify results
            assert len(sample_fs.accessions) == len(data["samples"])
            assert len(sample_fs.feature_names) > 0

            # Check sample features can be collected
            sample_df = sample_fs.collect()
            assert "acc" in sample_df.columns
            assert len(sample_df) == len(data["samples"])

    def test_dataset_integration(self, sample_aggregation_data):
        """Test full Dataset integration."""
        data = sample_aggregation_data

        # Create dataset with profiles
        dataset = Dataset()
        profiles = TaxonomicProfiles(data["profile_df"], check_filled=False)
        dataset.add_profiles(profiles)

        # Add species features
        dataset.add_species_features(
            name="species_traits",
            species_ids=data["species"],
            feature_names=data["feature_names"],
            feature_array=data["species_features_data"],
        )

        assert "species_traits" in dataset.species_feature_sets

        # Aggregate to sample features
        dataset.aggregate_species_to_samples(
            species_feature_name="species_traits",
            output_name="sample_traits_mean",
            method=AggregationMethod.ARITHMETIC_MEAN,
            weighting=WeightingStrategy.NONE,
        )

        assert "sample_traits_mean" in dataset.feature_sets

        # Test different methods
        dataset.aggregate_species_to_samples(
            species_feature_name="species_traits",
            output_name="sample_traits_weighted",
            method=AggregationMethod.ARITHMETIC_MEAN,
            weighting=WeightingStrategy.ABUNDANCE,
        )

        dataset.aggregate_species_to_samples(
            species_feature_name="species_traits",
            output_name="sample_traits_presence",
            method=AggregationMethod.PRESENCE_ABSENCE,
        )

        # Verify all aggregations created
        expected_features = [
            "sample_traits_mean",
            "sample_traits_weighted",
            "sample_traits_presence",
        ]
        for feature_name in expected_features:
            assert feature_name in dataset.feature_sets

        # Compare results
        mean_fs = dataset.feature_sets["sample_traits_mean"]
        weighted_fs = dataset.feature_sets["sample_traits_weighted"]
        presence_fs = dataset.feature_sets["sample_traits_presence"]

        # Verify they can be collected and have expected structure
        mean_df = mean_fs.collect()
        weighted_df = weighted_fs.collect()
        presence_df = presence_fs.collect()

        for df in [mean_df, weighted_df, presence_df]:
            assert "acc" in df.columns
            assert len(df) == len(data["samples"])

    def test_backward_compatibility(self):
        """Test that existing FeatureSet functionality still works."""
        accessions = ["S1", "S2", "S3"]
        feature_names = ["f1", "f2", "f3"]
        features = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )

        fs = FeatureSet(accessions, feature_names, features, name="test")

        assert isinstance(fs, FeatureSet)
        assert fs.accessions == accessions
        assert fs.feature_names == feature_names

        # Test class methods
        df = pl.DataFrame(
            {
                "sample": accessions,
                "f1": [0.1, 0.4, 0.7],
                "f2": [0.2, 0.5, 0.8],
                "f3": [0.3, 0.6, 0.9],
            }
        )

        fs2 = FeatureSet.from_df(df, name="from_df_test")
        assert isinstance(fs2, FeatureSet)
