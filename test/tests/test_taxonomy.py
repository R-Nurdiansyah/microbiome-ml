"""Tests for taxonomy utilities and regex generation."""

import polars as pl
import pytest

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.profiles import TaxonomicProfiles


class TestTaxonomicRanksRegex:
    """Test regex generation for taxonomic ranks."""

    @pytest.fixture
    def mock_taxonomy_df(self):
        """Create mock taxonomy DataFrame with one entry per rank."""
        return pl.DataFrame(
            {
                "sample": ["S1"] * 7,
                "taxonomy": [
                    "d__Bacteria",
                    "d__Bacteria;p__Proteobacteria",
                    "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria",
                    "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales",
                    "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae",
                    "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia",
                    "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
                ],
                "coverage": [100.0] * 7,
            }
        )

    def test_regex_format_all_ranks(self):
        """Test regex format for all ranks."""
        expected = {
            TaxonomicRanks.DOMAIN: "d__[^;]+(?:;p__)?$",
            TaxonomicRanks.PHYLUM: "p__[^;]+(?:;c__)?$",
            TaxonomicRanks.CLASS: "c__[^;]+(?:;o__)?$",
            TaxonomicRanks.ORDER: "o__[^;]+(?:;f__)?$",
            TaxonomicRanks.FAMILY: "f__[^;]+(?:;g__)?$",
            TaxonomicRanks.GENUS: "g__[^;]+(?:;s__)?$",
            TaxonomicRanks.SPECIES: "s__[^;]+$",
        }

        for rank, expected_regex in expected.items():
            assert (
                rank.get_regex() == expected_regex
            ), f"Rank {rank.name} regex mismatch"

    def test_regex_filters_correctly(self, mock_taxonomy_df):
        """Test that each rank's regex matches exactly its level."""
        for rank in TaxonomicRanks.iter_from_domain():
            regex = rank.get_regex()
            filtered = mock_taxonomy_df.filter(
                pl.col("taxonomy").str.contains(regex)
            )

            # Should match exactly one entry (the one at this rank)
            assert (
                filtered.height == 1
            ), f"Rank {rank.name} should match 1 entry, got {filtered.height}"

            # The matched entry should contain this rank's prefix
            taxonomy = filtered["taxonomy"][0]
            assert (
                rank.prefix in taxonomy
            ), f"Matched taxonomy should contain {rank.prefix}"

    def test_regex_excludes_child_ranks(self, mock_taxonomy_df):
        """Test that regex excludes entries with child rank data."""
        # Phylum regex should match phylum-only, not class and below
        phylum_regex = TaxonomicRanks.PHYLUM.get_regex()
        phylum_filtered = mock_taxonomy_df.filter(
            pl.col("taxonomy").str.contains(phylum_regex)
        )

        # Should not match entries with class or deeper
        for taxonomy in phylum_filtered["taxonomy"].to_list():
            # Should have phylum but not class (unless it's just "c__" prefix)
            if "c__" in taxonomy:
                assert taxonomy.endswith(
                    "c__"
                ), f"Should not match taxonomy with class data: {taxonomy}"


class TestTaxonomicProfilesFeatureCreation:
    """Test feature creation from taxonomic profiles at different ranks."""

    @pytest.fixture
    def real_profiles(self):
        """Load real test data."""
        return TaxonomicProfiles(
            profiles="test/data/subset_real_filled_coverage.csv"
        )

    def test_create_features_all_ranks(self, real_profiles):
        """Test creating features for all ranks using iter_from_domain."""
        for rank in TaxonomicRanks.iter_from_domain():
            # Create features at this rank
            features = real_profiles.create_features(rank)

            # Get expected taxonomy entries using regex
            regex = rank.get_regex()
            expected = real_profiles.profiles.filter(
                pl.col("taxonomy").str.contains(regex)
            )

            # Get unique taxonomy strings at this rank
            expected_taxa = (
                expected.select("taxonomy")
                .unique()
                .sort("taxonomy")
                .collect()
                .to_series()
                .to_list()
            )

            # Feature names should match the unique taxonomy strings
            assert len(features.feature_names) == len(
                expected_taxa
            ), f"Rank {rank.name}: Expected {len(expected_taxa)} features, got {len(features.feature_names)}"

    def test_feature_names_match_filtered_taxa(self, real_profiles):
        """Test that feature names exactly match filtered taxonomy entries."""
        for rank in TaxonomicRanks.iter_from_domain():
            features = real_profiles.create_features(rank)

            # Get taxa at this rank using regex
            regex = rank.get_regex()
            truth = real_profiles.profiles.filter(
                pl.col("taxonomy").str.contains(regex)
            )

            # Get unique taxonomy strings
            expected_features = sorted(
                truth.select("taxonomy")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
            actual_features = sorted(features.feature_names)

            # Feature names should exactly match
            assert (
                actual_features == expected_features
            ), f"Rank {rank.name}: Feature names don't match filtered taxonomy"

    def test_create_features_sample_consistency(self, real_profiles):
        """Test that samples are consistent with the filtered data."""
        for rank in TaxonomicRanks.iter_from_domain():
            features = real_profiles.create_features(rank)

            # Get expected samples from filtered profiles
            regex = rank.get_regex()
            filtered = real_profiles.profiles.filter(
                pl.col("taxonomy").str.contains(regex)
            )
            expected_samples = set(
                filtered.select("sample")
                .unique()
                .collect()
                .to_series()
                .to_list()
            )
            actual_samples = set(features.accessions)

            # Samples in features should match samples in filtered data
            assert (
                actual_samples == expected_samples
            ), f"Rank {rank.name}: Sample mismatch between features and filtered data"

    def test_create_features_string_rank(self, real_profiles):
        """Test creating features using string rank name."""
        features_str = real_profiles.create_features("PHYLUM")
        features_enum = real_profiles.create_features(TaxonomicRanks.PHYLUM)

        assert sorted(features_str.feature_names) == sorted(
            features_enum.feature_names
        )
        assert sorted(features_str.accessions) == sorted(
            features_enum.accessions
        )

    def test_create_features_returns_featureset(self, real_profiles):
        """Test that create_features returns a FeatureSet instance."""
        from microbiome_ml.wrangle.features import FeatureSet

        features = real_profiles.create_features(TaxonomicRanks.GENUS)

        assert isinstance(features, FeatureSet)
        assert features.name == "genus_features"
        assert features.features is not None
        assert len(features.accessions) > 0
        assert len(features.feature_names) > 0
