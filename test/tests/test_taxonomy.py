"""Tests for taxonomy utilities and regex generation."""

import re
from typing import List

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
            TaxonomicRanks.DOMAIN: "d__[^;]+(?:;p__[^;]+)?",
            TaxonomicRanks.PHYLUM: "p__[^;]+(?:;c__[^;]+)?",
            TaxonomicRanks.CLASS: "c__[^;]+(?:;o__[^;]+)?",
            TaxonomicRanks.ORDER: "o__[^;]+(?:;f__[^;]+)?",
            TaxonomicRanks.FAMILY: "f__[^;]+(?:;g__[^;]+)?",
            TaxonomicRanks.GENUS: "g__[^;]+(?:;s__[^;]+)?",
            TaxonomicRanks.SPECIES: "s__[^;]+",
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

            # Should match at least one entry containing this rank
            assert (
                filtered.height >= 1
            ), f"Rank {rank.name} should match at least 1 entry, got {filtered.height}"

            # The entry representing exactly this rank should be present
            expected_taxonomy = mock_taxonomy_df["taxonomy"][rank.value]
            assert (
                expected_taxonomy in filtered["taxonomy"].to_list()
            ), f"Rank {rank.name} regex should match {expected_taxonomy}"

            # Every matched entry should contain the rank prefix
            for taxonomy in filtered["taxonomy"].to_list():
                assert (
                    rank.prefix in taxonomy
                ), f"Matched taxonomy should contain {rank.prefix}"

    def test_regex_excludes_child_ranks(self, mock_taxonomy_df):
        """Test that regex captures at most one child rank segment."""
        # Phylum regex should capture phylum with optional class only
        phylum_regex = TaxonomicRanks.PHYLUM.get_regex()
        phylum_filtered = mock_taxonomy_df.filter(
            pl.col("taxonomy").str.contains(phylum_regex)
        )

        # Extract matched segment and ensure it stops at optional class
        for taxonomy in phylum_filtered["taxonomy"].to_list():
            match = re.search(phylum_regex, taxonomy)
            assert match is not None, "Regex should match a phylum segment"
            segment = match.group(0)
            levels = [lvl for lvl in segment.split(";") if lvl]
            assert (
                1 <= len(levels) <= 2
            ), f"Matched segment should include phylum with optional class, got {segment}"
            if len(levels) == 2:
                assert levels[1].startswith(
                    "c__"
                ), f"Optional child segment should be class, got {levels[1]}"


class TestTaxonomicProfilesFeatureCreation:
    """Test feature creation from taxonomic profiles at different ranks."""

    @staticmethod
    def _expected_rank_taxa(
        profiles_lf: pl.LazyFrame, rank: TaxonomicRanks
    ) -> List[str]:
        """Mirror create_features extraction for test expectations."""
        pattern = rf"(?:^|;)((?:{rank.prefix}[^;]+))(?:;|$)"
        return (
            profiles_lf.filter(
                pl.col("taxonomy").str.contains(rank.get_regex())
            )
            .with_columns(pl.col("taxonomy").str.replace_all(r";\s+", ";"))
            .with_columns(
                pl.col("taxonomy").str.extract(pattern).alias("rank_taxonomy")
            )
            .filter(pl.col("rank_taxonomy").is_not_null())
            .select("rank_taxonomy")
            .unique()
            .sort("rank_taxonomy")
            .collect()
            .to_series()
            .to_list()
        )

    @pytest.fixture
    def real_profiles(self):
        """Load real test data."""
        return TaxonomicProfiles(
            profiles="test/data/subset_real_filled_coverage.csv.gz"
        )

    def test_create_features_all_ranks(self, real_profiles):
        """Test creating features for all ranks using iter_from_domain."""
        for rank in TaxonomicRanks.iter_from_domain():
            features = real_profiles.create_features(rank)

            expected = self._expected_rank_taxa(real_profiles.profiles, rank)

            assert len(features.feature_names) == len(
                expected
            ), f"Rank {rank.name}: Expected {len(expected)} features, got {len(features.feature_names)}"

    def test_feature_names_match_filtered_taxa(self, real_profiles):
        """Test that feature names exactly match filtered taxonomy entries."""
        for rank in TaxonomicRanks.iter_from_domain():
            features = real_profiles.create_features(rank)

            expected_features = self._expected_rank_taxa(
                real_profiles.profiles, rank
            )
            actual_features = sorted(features.feature_names)

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
