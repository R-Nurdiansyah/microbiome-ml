"""Feature aggregation functionality for converting species-wise to sample-wise
features."""

import logging
from typing import Any, List

import polars as pl

from microbiome_ml.core.config import (
    AggregationConfig,
    AggregationMethod,
    WeightingStrategy,
)
from microbiome_ml.wrangle.features import SampleFeatureSet, SpeciesFeatureSet

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Memory-efficient feature aggregation using polars LazyFrame.

    Note:
        All aggregation methods require taxonomic profiles with relative abundance
        data (relabund column). If you have coverage data, use TaxonomicProfiles
        to automatically convert coverage to relative abundance before aggregation.
    """

    def __init__(self) -> None:
        """Initialize aggregator."""

    def _get_feature_columns(
        self, species_features_lf: pl.LazyFrame
    ) -> List[str]:
        """Get feature column names (excluding 'taxonomy')."""
        return [
            col
            for col in species_features_lf.collect_schema().names()
            if col not in ("acc", "sample", "taxonomy")
        ]

    def _apply_weighting(
        self, joined_data: pl.LazyFrame, weighting: WeightingStrategy
    ) -> pl.LazyFrame:
        """Apply weighting strategy to joined data."""

        if weighting == WeightingStrategy.NONE:
            return joined_data.with_columns(pl.lit(1.0).alias("weight"))

        elif weighting == WeightingStrategy.ABUNDANCE:
            return joined_data.with_columns(pl.col("relabund").alias("weight"))

        elif weighting == WeightingStrategy.SQRT_ABUNDANCE:
            return joined_data.with_columns(
                pl.col("relabund").sqrt().alias("weight")
            )

        else:
            raise ValueError(f"Unknown weighting strategy: {weighting}")

    def aggregate_arithmetic_mean(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Arithmetic mean aggregation with weighting."""

        weighted_data = self._apply_weighting(joined_data, weighting)

        agg_exprs = [
            (pl.col(col) * pl.col("weight")).sum() / pl.col("weight").sum()
            for col in feature_cols
        ]

        return (
            weighted_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_geometric_mean(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Geometric mean aggregation with weighting."""

        pseudocount = kwargs.get("pseudocount", 1e-8)
        weighted_data = self._apply_weighting(joined_data, weighting)

        agg_exprs = [
            (
                ((pl.col(col) + pseudocount).log() * pl.col("weight")).sum()
                / pl.col("weight").sum()
            )
            .exp()
            .alias(col)
            for col in feature_cols
        ]

        return (
            weighted_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_harmonic_mean(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Harmonic mean aggregation with weighting."""

        pseudocount = kwargs.get("pseudocount", 1e-8)
        weighted_data = self._apply_weighting(joined_data, weighting)

        agg_exprs = [
            (
                pl.col("weight").sum()
                / (pl.col("weight") / (pl.col(col) + pseudocount)).sum()
            ).alias(col)
            for col in feature_cols
        ]

        return (
            weighted_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_median(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Median aggregation (weighted median is approximated)."""

        if weighting != WeightingStrategy.NONE:
            logger.warning(
                "Weighted median not implemented, falling back to weighted arithmetic mean"
            )
            return self.aggregate_arithmetic_mean(
                joined_data, feature_cols, weighting, **kwargs
            )

        # For unweighted median, we don't need the weight column
        agg_exprs = [pl.col(col).median() for col in feature_cols]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_presence_absence(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Presence/absence aggregation (weighting not applicable)."""

        min_abundance = kwargs.get("min_abundance", 0.001)

        return (
            joined_data.filter(pl.col("relabund") >= min_abundance)
            .group_by("sample")
            .agg(
                [
                    pl.col("taxonomy").count().alias("richness"),
                    pl.col("relabund").sum().alias("total_abundance"),
                ]
            )
            .rename({"sample": "acc"})
        )

    def aggregate_max_abundance(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Max abundance aggregation (weighting not applicable)."""

        agg_exprs = [
            pl.col(col)
            .filter(pl.col("relabund") == pl.col("relabund").max())
            .first()
            for col in feature_cols
        ]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_min_abundance(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Min abundance aggregation (weighting not applicable)."""

        agg_exprs = [
            pl.col(col)
            .filter(pl.col("relabund") == pl.col("relabund").min())
            .first()
            for col in feature_cols
        ]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_top_k_abundant(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Top-k abundant taxa aggregation."""

        k = kwargs.get("k", 10)

        # Get top-k most abundant taxa per sample from the joined data
        top_k_relabund = (
            joined_data.with_row_index()
            .group_by("sample")
            .agg(
                [
                    pl.col("taxonomy", "relabund", "row_nr").top_k_by(
                        "relabund", k=k
                    )
                ]
            )
            .explode(["taxonomy", "relabund", "row_nr"])
        )

        # Re-join with the original data to get features for top-k taxa
        filtered_data = top_k_relabund.join(
            joined_data, on=["sample", "taxonomy"], how="inner", coalesce=True
        )

        # Use arithmetic mean on the filtered data
        return self.aggregate_arithmetic_mean(
            filtered_data, feature_cols, weighting, **kwargs
        )

    def remove_uncommon_features(
        self, feature_set: pl.LazyFrame, uncommon_feature_cutoff: int = 10
    ) -> pl.LazyFrame:
        """Removes columns that only appear in a fraction of the samples."""

        # Get column names without loading full data
        all_columns = feature_set.collect_schema().names()

        # Check each column's null count individually (memory efficient)
        good_columns = []
        for col in all_columns:
            null_count = (
                feature_set.select(pl.col(col).null_count()).collect().item()
            )

            if null_count < uncommon_feature_cutoff:
                good_columns.append(col)

        return feature_set.select(good_columns)

    def get_sample_features(
        self,
        species_features: SpeciesFeatureSet,
        relabund_lf: pl.LazyFrame,
        config: AggregationConfig,
    ) -> SampleFeatureSet:
        """Convert species-wise features into sample-wise features using
        aggregation.

        Args:
            species_features: SpeciesFeatureSet with species-level features
            relabund_lf: LazyFrame with relative abundance data (sample, taxonomy, relabund)
                        Must contain 'relabund' column with relative abundance values [0-1]
            config: Aggregation configuration

        Returns:
            SampleFeatureSet with aggregated features

        Note:
            The relabund_lf must contain relative abundance data, not raw coverage.
            Use TaxonomicProfiles to convert coverage to relative abundance if needed.
            relabund_lf: LazyFrame with relative abundance data (sample, taxonomy, relabund)
            config: AggregationConfig with method and parameters

        Returns:
            SampleFeatureSet with aggregated sample-level features
        """
        # Convert species features to LazyFrame for joining
        species_df = species_features.collect()
        species_lf = species_df.lazy()

        feature_cols = self._get_feature_columns(species_lf)

        # Strategy mapping
        aggregation_strategies = {
            AggregationMethod.ARITHMETIC_MEAN: self.aggregate_arithmetic_mean,
            AggregationMethod.GEOMETRIC_MEAN: self.aggregate_geometric_mean,
            AggregationMethod.HARMONIC_MEAN: self.aggregate_harmonic_mean,
            AggregationMethod.MEDIAN: self.aggregate_median,
            AggregationMethod.PRESENCE_ABSENCE: self.aggregate_presence_absence,
            AggregationMethod.MAX_ABUNDANCE: self.aggregate_max_abundance,
            AggregationMethod.MIN_ABUNDANCE: self.aggregate_min_abundance,
            AggregationMethod.TOP_K_ABUNDANT: self.aggregate_top_k_abundant,
        }

        # Get the appropriate aggregation function
        if config.method not in aggregation_strategies:
            raise ValueError(f"Unknown aggregation method: {config.method}")

        aggregation_func = aggregation_strategies[config.method]

        # Join the data first for all methods (now they all use the same signature)
        joined_data = relabund_lf.join(
            species_lf, on="taxonomy", how="inner", coalesce=True
        )
        sample_features_lf = aggregation_func(
            joined_data,
            feature_cols,
            config.weighting,
            k=config.k,
            pseudocount=config.pseudocount,
            min_abundance=config.min_abundance,
        )

        # Remove uncommon features
        sample_features_lf = self.remove_uncommon_features(
            sample_features_lf, config.uncommon_feature_cutoff
        )

        # Convert to SampleFeatureSet
        sample_features_df = sample_features_lf.collect()

        return SampleFeatureSet.from_df(
            df=sample_features_df,
            name=f"{species_features.name}_{config.method.value}",
            acc_column="acc",
        )


def aggregate_species_to_samples(
    species_features: SpeciesFeatureSet,
    relabund_lf: pl.LazyFrame,
    config: AggregationConfig,
) -> SampleFeatureSet:
    """Convenience function to aggregate species features to sample features.

    Args:
        species_features: SpeciesFeatureSet with species-level features
        relabund_lf: LazyFrame with relative abundance data
        config: AggregationConfig with method and parameters

    Returns:
        SampleFeatureSet with aggregated sample-level features
    """
    aggregator = FeatureAggregator()
    return aggregator.get_sample_features(
        species_features, relabund_lf, config
    )
