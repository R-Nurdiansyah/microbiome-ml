"""High-level sample coordination class for ML workflows."""

import json
import logging
import tarfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles
from microbiome_ml.wrangle.splits import SplitManager

logger = logging.getLogger(__name__)


class Dataset:
    """Global entry point for microbiome ML workflows with flexible builder
    pattern.

    Can be initialized empty and components added iteratively, or with all data at once.
    Serves as the central data structure for ML workflows.

    Examples:
        # Empty initialization with builder pattern
        samples = (Dataset()
                  .add_metadata(metadata='path/to/metadata.csv', attributes='path/to/attributes.csv')
                  .add_profiles('path/to/profiles.csv')
                  .add_features('genus', rank=TaxonomicRanks.GENUS))

        # Full initialization
        samples = Dataset(
            metadata={'metadata': 'path/to/metadata.csv', 'attributes': 'path/to/attributes.csv'},
            profiles='path/to/profiles.csv'
        )

        # Mixed approach
        samples = Dataset(metadata=existing_metadata)
        samples.add_profiles('path/to/profiles.csv').add_features('genus', rank=TaxonomicRanks.GENUS)
    """

    def __init__(
        self,
        metadata: Optional[
            Union[SampleMetadata, Dict[str, Union[str, Path, pl.LazyFrame]]]
        ] = None,
        profiles: Optional[
            Union[TaxonomicProfiles, str, Path, pl.LazyFrame]
        ] = None,
        features: Optional[Dict[str, FeatureSet]] = None,
        labels: Optional[
            Union[pl.DataFrame, str, Path, Dict[str, Any]]
        ] = None,
        groupings: Optional[
            Union[pl.DataFrame, str, Path, Dict[str, Any]]
        ] = None,
    ):
        """Initialize Dataset with optional components.

        Args:
            metadata: SampleMetadata instance or dict with metadata sources
            profiles: TaxonomicProfiles instance or data source
            features: Dict of FeatureSets by name
            labels: Labels DataFrame or source
            groupings: Groupings DataFrame or source
        """
        # Initialize empty structure
        self.metadata: Optional[SampleMetadata] = None
        self.profiles: Optional[TaxonomicProfiles] = None
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.labels: Optional[pl.DataFrame] = None
        self.groupings: Optional[pl.DataFrame] = None
        self.splits: Dict[str, SplitManager] = {}
        self._sample_ids: Optional[List[str]] = None

        self.metadata_qc_done: bool = False
        self.profiles_qc_done: bool = False

        # Process provided components if given
        if metadata is not None:
            self.add_metadata(metadata)
        if profiles is not None:
            self.add_profiles(profiles)
        if features is not None:
            self.feature_sets.update(features)
        if labels is not None:
            self.add_labels(labels)
        if groupings is not None:
            self.add_groupings(groupings)

        self._sync_accessions()

    def add_metadata(
        self,
        data: Optional[
            Union[SampleMetadata, Dict[str, Union[str, Path, pl.LazyFrame]]]
        ] = None,
        metadata: Optional[
            Union[str, Path, pl.LazyFrame, pl.DataFrame]
        ] = None,
        attributes: Optional[
            Union[str, Path, pl.LazyFrame, pl.DataFrame]
        ] = None,
        study_titles: Optional[
            Union[str, Path, pl.LazyFrame, pl.DataFrame]
        ] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Add metadata component with flexible input handling.

        Args:
            data: SampleMetadata instance or dict with metadata sources
            metadata: Direct metadata source
            attributes: Direct attributes source
            study_titles: Direct study_titles source
            **kwargs: Additional metadata sources

        Returns:
            Self for chaining

        Examples:
            .add_metadata(metadata='path.csv', attributes='attr.csv')
            .add_metadata({'metadata': df, 'attributes': 'path.csv'})
            .add_metadata(existing_metadata_instance)
        """
        # Handle SampleMetadata instance
        if isinstance(data, SampleMetadata):
            self.metadata = data
        # Handle dict with named metadata sources
        elif isinstance(data, dict):
            self.metadata = SampleMetadata(
                metadata=data.get("metadata"),  # type: ignore
                attributes=data.get("attributes"),  # type: ignore
                study_titles=data.get("study_titles", None),  # type: ignore
            )
        # Handle direct keyword arguments
        else:
            # Use provided kwargs or direct parameters
            meta_source = metadata or kwargs.get("metadata")
            attr_source = attributes or kwargs.get("attributes")
            study_source = study_titles or kwargs.get("study_titles")

            if meta_source and attr_source:
                self.metadata = SampleMetadata(
                    metadata=meta_source,
                    attributes=attr_source,
                    study_titles=study_source,
                )

        self._sync_accessions()
        return self

    def add_profiles(
        self,
        profiles: Union[
            TaxonomicProfiles, str, Path, pl.LazyFrame, pl.DataFrame
        ],
        root: Optional[Union[str, Path, pl.LazyFrame, pl.DataFrame]] = None,
        check_filled: bool = True,
        sample_size: int = 1000,
        **kwargs: Any,
    ) -> "Dataset":
        """Add profiles component with flexible input handling.

        Args:
            profiles: TaxonomicProfiles instance or data source
            root: Optional root coverage data
            check_filled: Whether to check if profiles are filled
            sample_size: Sample size for filled format checking
            **kwargs: Additional TaxonomicProfiles parameters

        Returns:
            Self for chaining
        """
        # Handle TaxonomicProfiles instance
        if isinstance(profiles, TaxonomicProfiles):
            self.profiles = profiles
        # Handle file paths, DataFrames, LazyFrames
        else:
            self.profiles = TaxonomicProfiles(
                profiles=profiles,
                root=root,
                check_filled=check_filled,
                sample_size=sample_size,
                **kwargs,
            )

        self._sync_accessions()
        return self

    def add_features(
        self,
        name: str,
        features: Optional[Union[FeatureSet, pl.DataFrame]] = None,
        rank: Optional[Union[str, TaxonomicRanks]] = None,
        accessions: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        feature_array: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Add feature set with flexible input handling.

        Args:
            name: Name for the feature set
            features: FeatureSet instance or DataFrame
            rank: Taxonomic rank to create features from profiles
            accessions: Direct accessions list
            feature_names: Direct feature names list
            feature_array: Direct feature numpy array
            **kwargs: Additional parameters for FeatureSet creation

        Returns:
            Self for chaining

        Examples:
            .add_features('genus', rank=TaxonomicRanks.GENUS)  # From profiles
            .add_features('custom', features=dataframe)
            .add_features('existing', features=feature_set_instance)
            .add_features('direct', accessions=[...], feature_names=[...], feature_array=array)
        """
        # Handle FeatureSet instance
        if isinstance(features, FeatureSet):
            self.feature_sets[name] = features
        # Handle DataFrame conversion
        elif isinstance(features, pl.DataFrame):
            self.feature_sets[name] = FeatureSet.from_df(
                features, name=name, **kwargs
            )
        # Create from profiles using rank
        elif rank is not None:
            if self.profiles is None:
                raise ValueError(
                    "Profiles must be added before creating taxonomic features"
                )
            feature_set = self.profiles.create_features(rank)
            feature_set.name = name
            self.feature_sets[name] = feature_set
        # Create directly from components
        elif accessions and feature_names and feature_array is not None:
            self.feature_sets[name] = FeatureSet(
                accessions=accessions,
                feature_names=feature_names,
                features=feature_array,
                name=name,
            )
        else:
            raise ValueError(
                "Must provide FeatureSet instance, DataFrame, rank parameter, or direct components"
            )

        return self

    def iter_feature_sets(
        self, names: Optional[List[str]] = None
    ) -> Any:  # Returns generator
        """Iterate over feature sets.

        Args:
            names: List of feature set names to iterate over (all if None)

        Yields:
            Tuple of (name, FeatureSet) for each feature set
        """
        if names is None:
            names = list(self.feature_sets.keys())
        for name in names:
            yield name, self.feature_sets[name]

    def iter_labels(
        self, names: Optional[List[str]] = None
    ) -> Any:  # Returns generator
        """Iterate over label columns.

        Args:
            names: List of label names to iterate over (all if None)

        Yields:
            Tuple of (name, DataFrame) for each label set (sample + label column)
        """
        if self.labels is None:
            return

        available_names = [c for c in self.labels.columns if c != "sample"]
        if names is None:
            names = available_names

        for name in names:
            if name in available_names:
                yield name, self.labels.select(["sample", name])

    def iter_groupings(
        self, names: Optional[List[str]] = None
    ) -> Any:  # Returns generator
        """Iterate over grouping columns.

        Args:
            names: List of grouping names to iterate over (all if None)

        Yields:
            Tuple of (name, DataFrame) for each grouping set (sample + grouping column)
        """
        if self.groupings is None:
            return

        available_names = [c for c in self.groupings.columns if c != "sample"]
        if names is None:
            names = available_names

        for name in names:
            if name in available_names:
                yield name, self.groupings.select(["sample", name])

    def get_sample_ids(self) -> List[str]:
        """Get the canonical list of sample IDs after synchronization.

        Returns:
            List of sample IDs present in all components (strict intersection)
        """
        return self._sample_ids.copy() if self._sample_ids else []

    def _sync_accessions(self) -> None:
        """Synchronize accessions across all components using strict
        intersection via inner joins.

        This method:
        1. Collects sample lists from all components
        2. Computes strict intersection using Polars inner joins
        3. Uses built-in filter methods to remove samples not in intersection
        4. Updates self._sample_ids with canonical ordering

        Warnings are issued if synchronization drops significant samples.
        """
        logger.debug("Starting sample synchronization across components")
        # Collect sample DataFrames from all available components
        sample_dfs: List[Tuple[str, pl.DataFrame]] = []
        component_counts = {}

        def get_samples_df(source_obj: Any) -> pl.DataFrame:
            """Helper to safely get samples DataFrame from component."""
            if hasattr(source_obj, "_get_sample_list"):
                samples = source_obj._get_sample_list()
                if isinstance(samples, pl.LazyFrame):
                    return samples.select("sample").unique().collect()
                elif isinstance(samples, pl.DataFrame):
                    return samples.select("sample").unique()
                elif isinstance(samples, (set, list)):
                    return pl.DataFrame({"sample": list(samples)})
            return pl.DataFrame({"sample": []}, schema={"sample": pl.Utf8})

        if self.metadata is not None:
            metadata_samples = get_samples_df(self.metadata)
            sample_dfs.append(("metadata", metadata_samples))
            component_counts["metadata"] = metadata_samples.height
            logger.debug(
                f"Metadata component has {metadata_samples.height} samples"
            )

        if self.profiles is not None:
            profiles_samples = get_samples_df(self.profiles)
            sample_dfs.append(("profiles", profiles_samples))
            component_counts["profiles"] = profiles_samples.height
            logger.debug(
                f"Profiles component has {profiles_samples.height} samples"
            )

        if self.feature_sets:
            # Get intersection of all feature sets
            feature_accs_list = [
                fs._get_sample_list() for fs in self.feature_sets.values()
            ]
            if feature_accs_list:
                features_accs = set.intersection(*feature_accs_list)
                features_samples = pl.DataFrame(
                    {"sample": list(features_accs)}
                )
                sample_dfs.append(("feature_sets", features_samples))
                component_counts["feature_sets"] = features_samples.height
                logger.debug(
                    f"Feature sets (intersection) have {features_samples.height} samples"
                )

        # If no components, set to None and return
        if not sample_dfs:
            logger.debug("No components found, setting sample_ids to None")
            self._sample_ids = None
            return

        logger.debug(
            f"Computing intersection across {len(sample_dfs)} component(s)"
        )

        # Start with first component's samples
        canonical_df = sample_dfs[0][1]

        # Inner join with each subsequent component to get strict intersection
        for comp_name, comp_df in sample_dfs[1:]:
            canonical_df = canonical_df.join(comp_df, on="sample", how="inner", coalesce=True)

        # Sort for deterministic ordering and extract list
        self._sample_ids = (
            canonical_df.sort("sample").select("sample").to_series().to_list()
        )
        final_count = canonical_df.height
        logger.debug(f"Final canonical sample set: {final_count} samples")

        # Check for significant drops relative to each component
        for comp_name, comp_df in sample_dfs:
            comp_count = comp_df.height
            if comp_count == 0:
                continue

            dropped = comp_count - final_count
            pct_dropped = (dropped / comp_count) * 100

            if pct_dropped > 10:
                warnings.warn(
                    f"Accession sync removed {dropped} samples ({pct_dropped:.1f}%) from {comp_name}. "
                    f"Component has {comp_count}, but intersection has {final_count}.",
                    UserWarning,
                )

        # Warn if no samples remain after intersection
        if len(self._sample_ids) == 0:
            warnings.warn(
                "Accession synchronization resulted in empty sample set (no samples common to all components). "
                "No filtering will be performed. Check that your components have overlapping sample IDs.",
                UserWarning,
            )
            logger.warning(
                "No common samples found across components - synchronization aborted"
            )
            return

        # Filter components to canonical set using their built-in filter methods
        self._filter_components_to_canonical()

    def _filter_components_to_canonical(self) -> None:
        """Filter all components to match the canonical sample set using built-
        in filter methods."""
        if self._sample_ids is None or not self._sample_ids:
            logger.debug("No canonical sample IDs to filter to")
            return

        logger.debug(
            f"Filtering components to canonical set of {len(self._sample_ids)} samples"
        )
        # Create LazyFrame with canonical samples for filtering
        canonical_lf = pl.DataFrame({"sample": self._sample_ids}).lazy()

        # Filter metadata using built-in method
        if self.metadata is not None:
            self.metadata = self.metadata._filter_by_sample(canonical_lf)  # type: ignore

        # Filter profiles using built-in method
        if self.profiles is not None:
            self.profiles = self.profiles._filter_by_sample(canonical_lf)

        # Filter feature sets using built-in method
        for name, feature_set in list(self.feature_sets.items()):
            self.feature_sets[name] = feature_set.filter_samples(
                self._sample_ids  # type: ignore
            )

        # Align labels and groupings (left join to keep canonical samples)
        canonical_df = canonical_lf.collect()

        if self.labels is not None:
            self.labels = canonical_df.join(
                self.labels, on="sample", how="left", coalesce=True
            )

        if self.groupings is not None:
            self.groupings = canonical_df.join(
                self.groupings, on="sample", how="left", coalesce=True
            )

        # Filter splits (holdout and CV schemes in each SplitManager)
        if self.splits:
            for label, split_manager in self.splits.items():
                if split_manager.holdout is not None:
                    split_manager.holdout = canonical_df.join(
                        split_manager.holdout, on="sample", how="left", coalesce=True
                    )
                for scheme_name, cv_df in split_manager.cv_schemes.items():
                    split_manager.cv_schemes[scheme_name] = canonical_df.join(
                        cv_df, on="sample", how="left", coalesce=True
                    )

    def add_feature_set(
        self,
        features: Union[
            Dict[str, Union[str, Path, FeatureSet]], str, Path, FeatureSet
        ],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Add one or multiple feature sets.

        Args:
            features: Dict mapping names to sources, single file path, or FeatureSet instance
            name: Name for single feature set (required if features is not dict)
            **kwargs: Additional parameters passed to FeatureSet creation

        Returns:
            Self for chaining

        Examples:
            .add_feature_set({"kmers": "kmers.csv", "proteins": "proteins.csv"})
            .add_feature_set("features.csv", name="my_features")
            .add_feature_set(existing_featureset, name="existing")
        """
        if isinstance(features, dict):
            # Batch addition: dict mapping names to sources
            for fs_name, fs_source in features.items():
                if isinstance(fs_source, FeatureSet):
                    self.feature_sets[fs_name] = fs_source
                elif isinstance(fs_source, (str, Path)):
                    self.feature_sets[fs_name] = FeatureSet.load(fs_source)
                else:
                    raise ValueError(
                        f"Unsupported feature set source type for '{fs_name}': {type(fs_source)}"
                    )
        else:
            # Single addition
            if name is None:
                raise ValueError(
                    "name parameter required when adding single feature set"
                )

            if isinstance(features, FeatureSet):
                self.feature_sets[name] = features
            elif isinstance(features, (str, Path)):
                self.feature_sets[name] = FeatureSet.load(features)
            else:
                raise ValueError(
                    f"Unsupported feature set source type: {type(features)}"
                )

        self._sync_accessions()
        return self

    def add_labels(
        self,
        labels: Union[
            Dict[str, Union[str, Path, pl.DataFrame]], str, Path, pl.DataFrame
        ],
        name: Optional[str] = None,
    ) -> "Dataset":
        """Add labels to the dataset.

        Merges into self.labels DataFrame.

        Args:
            labels: Dict mapping names to sources, single file path, or DataFrame
            name: Name for single label set (required if labels is not dict and has single value column)

        Returns:
            Self for chaining

        Examples:
            .add_labels({"temp": "temp.csv", "ph": "ph.csv"})
            .add_labels("labels.csv", name="target")
            .add_labels(df, name="my_labels")
        """
        if isinstance(labels, dict):
            # Batch addition: dict mapping names to sources
            for label_name, label_source in labels.items():
                self.add_labels(label_source, name=label_name)
            return self

        # Load
        if isinstance(labels, (str, Path)):
            new_df = pl.read_csv(labels)
        elif isinstance(labels, pl.DataFrame):
            new_df = labels
        else:
            raise ValueError(f"Unsupported label source type: {type(labels)}")

        if "sample" not in new_df.columns:
            raise ValueError("Labels must contain 'sample' column")

        # Rename if name provided and single value column
        value_cols = [c for c in new_df.columns if c != "sample"]
        if name:
            if len(value_cols) == 1:
                new_df = new_df.rename({value_cols[0]: name})
            elif name not in value_cols:
                pass

        if self.labels is None:
            self.labels = new_df
        else:
            # Check collisions
            new_cols = set(new_df.columns) - {"sample"}
            existing_cols = set(self.labels.columns) - {"sample"}
            overlap = new_cols.intersection(existing_cols)
            if overlap:
                raise ValueError(f"Duplicate label columns: {overlap}")
            self.labels = self.labels.join(new_df, on="sample", how="outer", coalesce=True)

        self._sync_accessions()
        return self

    def add_groupings(
        self,
        groupings: Union[
            Dict[str, Union[str, Path, pl.DataFrame]], str, Path, pl.DataFrame
        ],
        name: Optional[str] = None,
    ) -> "Dataset":
        """Add groupings to the dataset.

        Merges into self.groupings DataFrame.

        Args:
            groupings: Dict mapping names to sources, single file path, or DataFrame
            name: Name for single grouping set (required if groupings is not dict and has single value column)

        Returns:
            Self for chaining
        """
        if isinstance(groupings, dict):
            # Batch addition: dict mapping names to sources
            for group_name, group_source in groupings.items():
                self.add_groupings(group_source, name=group_name)
            return self

        # Load
        if isinstance(groupings, (str, Path)):
            new_df = pl.read_csv(groupings)
        elif isinstance(groupings, pl.DataFrame):
            new_df = groupings
        else:
            raise ValueError(
                f"Unsupported grouping source type: {type(groupings)}"
            )

        if "sample" not in new_df.columns:
            raise ValueError("Groupings must contain 'sample' column")

        # Rename if name provided and single value column
        value_cols = [c for c in new_df.columns if c != "sample"]
        if name:
            if len(value_cols) == 1:
                new_df = new_df.rename({value_cols[0]: name})
            elif name not in value_cols:
                pass

        if self.groupings is None:
            self.groupings = new_df
        else:
            # Check collisions
            new_cols = set(new_df.columns) - {"sample"}
            existing_cols = set(self.groupings.columns) - {"sample"}
            overlap = new_cols.intersection(existing_cols)
            if overlap:
                raise ValueError(f"Duplicate grouping columns: {overlap}")

            self.groupings = self.groupings.join(
                new_df, on="sample", how="outer", coalesce=True
            )

        self._sync_accessions()
        return self

    def add_taxonomic_features(
        self,
        ranks: Optional[List[Union[str, TaxonomicRanks]]] = None,
        prefix: str = "tax",
    ) -> "Dataset":
        """Generate FeatureSets from TaxonomicProfiles at specified ranks.

        Args:
            ranks: List of taxonomic ranks (defaults to all standard ranks)
            prefix: Prefix for generated feature set names (default: "tax")

        Returns:
            Self for chaining

        Examples:
            .add_taxonomic_features()  # All ranks (phylum to species)
            .add_taxonomic_features(ranks=["genus", "family"])
            .add_taxonomic_features(ranks=[TaxonomicRanks.GENUS], prefix="rel")
        """
        if self.profiles is None:
            raise ValueError(
                "Profiles must be added before generating taxonomic features"
            )

        # Default to all standard ranks (e.g., phylum to species)
        if ranks is None:
            ranks_iter: Union[
                Iterator[TaxonomicRanks], List[Union[str, TaxonomicRanks]]
            ] = TaxonomicRanks.PHYLUM.iter_down()
        else:
            ranks_iter = ranks

        # Generate feature sets for each rank
        for rank in ranks_iter:
            # Convert string to enum if needed
            if isinstance(rank, str):
                rank = TaxonomicRanks.from_name(rank)

            feature_set = self.profiles.create_features(rank)
            feature_name = f"{prefix}_{rank.name.lower()}"
            feature_set.name = feature_name
            self.feature_sets[feature_name] = feature_set

        self._sync_accessions()
        return self

    def create_default_groupings(
        self,
        groupings: Optional[List[str]] = None,
        force: bool = False,
    ) -> "Dataset":
        """Create default grouping columns from metadata fields.

        Automatically extracts common grouping variables from metadata for use
        in train/test splitting and analysis. Fields are extracted as-is without
        transformation, and null values are preserved.

        Args:
            groupings: List of specific groupings to create. If None, creates all
                      available default groupings: ['bioproject', 'biome', 'domain',
                      'ecoregion', 'year', 'month', 'climate', 'season']
            force: If True, overwrite existing groupings. If False, merge with existing

        Returns:
            Self for chaining

        Examples:
            # Create all available default groupings
            dataset.create_default_groupings()

            # Create only specific groupings
            dataset.create_default_groupings(groupings=['bioproject', 'biome'])

            # Overwrite existing groupings
            dataset.create_default_groupings(force=True)

        Raises:
            ValueError: If metadata is not available
        """
        if self.metadata is None:
            raise ValueError(
                "Metadata must be added before creating default groupings"
            )

        # Define default groupings to attempt
        default_groupings = [
            "bioproject",
            "biome",
            "domain",
            "ecoregion",
            "year",
            "month",
            "climate",
            "season",
        ]

        # Use provided list or defaults
        fields_to_extract = (
            groupings if groupings is not None else default_groupings
        )

        # Try to extract each field, skip if not available
        extracted_fields = []
        for field in fields_to_extract:
            try:
                # Use the new metadata API to get the field
                field_data = getattr(self.metadata, field)
                # Verify it has data by checking if we can collect at least one row
                if field_data.head(1).collect().height > 0:
                    extracted_fields.append(field)
            except AttributeError:
                logger.warning(
                    f"Field '{field}' not found in metadata, skipping"
                )
            except Exception as e:
                logger.warning(
                    f"Could not extract field '{field}' from metadata: {e}"
                )

        if not extracted_fields:
            logger.warning(
                "No grouping fields could be extracted from metadata"
            )
            return self

        # Extract all available fields at once using the get() method
        groupings_data = self.metadata.get(*extracted_fields).collect()

        # Add or merge with existing groupings
        if self.groupings is None or force:
            self.groupings = groupings_data
        else:
            # Merge with existing groupings
            self.add_groupings(groupings_data)

        logger.info(f"Created default groupings: {extracted_fields}")

        return self

    def create_holdout_split(
        self,
        label: Optional[str] = None,
        test_size: float = 0.2,
        n_bins: int = 5,
        grouping: Optional[str] = None,
        random_state: int = 42,
        force: bool = False,
    ) -> "Dataset":
        """Create holdout train/test splits for one or all labels.

        Uses stratified sampling with optional group awareness.
        Handles both continuous (via binning) and categorical targets.

        Args:
            label: Specific label to split. If None, splits all labels.
            test_size: Fraction of samples for test set
            n_bins: Number of bins for continuous targets
            grouping: Optional grouping column to prevent leakage
            random_state: Random seed for reproducibility
            force: If True, overwrite existing splits

        Returns:
            Self for chaining
        """
        if self.labels is None:
            raise ValueError("Labels must be added before creating splits")

        # Get all label columns
        all_label_cols = [c for c in self.labels.columns if c != "sample"]
        if not all_label_cols:
            raise ValueError("No label columns found in labels DataFrame")

        # Determine which labels to process
        if label is not None:
            if label not in all_label_cols:
                raise ValueError(f"Label '{label}' not found in labels")
            label_list = [label]
        else:
            label_list = all_label_cols

        # Create splits for each label
        for lbl in label_list:
            # Check if split already exists
            if lbl in self.splits and not force:
                logger.warning(
                    f"Split for '{lbl}' already exists. Use force=True to overwrite."
                )
                continue

            logger.info(f"Creating holdout split for label: {lbl}")

            # Initialize SplitManager if needed
            if lbl not in self.splits:
                self.splits[lbl] = SplitManager(lbl)

            # Prepare data
            data = self.labels.select(["sample", lbl])

            if grouping is not None:
                if (
                    self.groupings is None
                    or grouping not in self.groupings.columns
                ):
                    raise ValueError(
                        f"Grouping '{grouping}' not found in groupings"
                    )
                groups = self.groupings.select(["sample", grouping])
                data = data.join(groups, on="sample", how="left", coalesce=True)

            # Filter nulls
            initial_count = data.height
            null_cols = [lbl] if grouping is None else [lbl, grouping]
            data = data.drop_nulls(subset=null_cols)
            if data.height < initial_count:
                logger.warning(
                    f"Dropped {initial_count - data.height} samples with null values for {lbl}"
                )

            # Create holdout split
            self.splits[lbl].create_holdout(
                data=data,
                grouping=grouping,
                test_size=test_size,
                n_bins=n_bins,
                random_state=random_state,
            )

        return self

    def create_cv_folds(
        self,
        label: Optional[str] = None,
        n_folds: int = 5,
        n_bins: int = 5,
        grouping: Optional[
            str
        ] = "all",  # New default: "all" creates all schemes
        random_state: int = 42,
        force: bool = False,
    ) -> "Dataset":
        """Create k-fold cross-validation splits for one or all labels.

        By default, creates CV folds for ALL available groupings plus a random baseline.
        Use grouping="random" to create only random splits, or specify a specific grouping.

        Args:
            label: Specific label to create folds for. If None, creates for all labels.
            n_folds: Number of folds
            n_bins: Number of bins for continuous targets
            grouping: Grouping strategy:
                - "all" (default): Create CV schemes for all groupings + random
                - "random": Create only random (no grouping) CV scheme
                - <column_name>: Create CV scheme for specific grouping column
                - None: Same as "random" (backward compatibility)
            random_state: Random seed for reproducibility
            force: If True, overwrite existing CV schemes

        Returns:
            Self for chaining

        Examples:
            # Create all CV schemes (random + all groupings) for all labels
            dataset.create_cv_folds()

            # Create all CV schemes for specific label
            dataset.create_cv_folds(label="pH")

            # Create only random CV for all labels
            dataset.create_cv_folds(grouping="random")

            # Create only specific grouping for all labels
            dataset.create_cv_folds(grouping="bioproject")
        """
        if self.labels is None:
            raise ValueError("Labels must be added before creating CV folds")

        # Get all label columns
        all_label_cols = [c for c in self.labels.columns if c != "sample"]
        if not all_label_cols:
            raise ValueError("No label columns found in labels DataFrame")

        # Determine which labels to process
        if label is not None:
            if label not in all_label_cols:
                raise ValueError(f"Label '{label}' not found in labels")
            label_list = [label]
        else:
            label_list = all_label_cols

        # Determine which groupings to use
        if grouping == "all":
            # Get all available grouping columns
            if self.groupings is not None:
                available_groupings = [
                    c for c in self.groupings.columns if c != "sample"
                ]
            else:
                available_groupings = []

            # Create list: random + all groupings
            groupings_to_use = [("random", None)] + [
                (g, g) for g in available_groupings
            ]

            if not available_groupings:
                logger.warning(
                    "No groupings available, creating only random CV"
                )

        elif grouping == "random" or grouping is None:
            # Only random
            groupings_to_use = [("random", None)]

        else:
            # Specific grouping column
            if (
                self.groupings is None
                or grouping not in self.groupings.columns
            ):
                raise ValueError(
                    f"Grouping '{grouping}' not found in groupings"
                )
            groupings_to_use = [(grouping, grouping)]

        # Create CV folds for each label × grouping combination
        for lbl in label_list:
            # Initialize SplitManager if needed
            if lbl not in self.splits:
                self.splits[lbl] = SplitManager(lbl)

            for scheme_name, grouping_col in groupings_to_use:
                # Check if scheme already exists
                if scheme_name in self.splits[lbl].cv_schemes and not force:
                    logger.warning(
                        f"CV scheme '{scheme_name}' for label '{lbl}' already exists. "
                        f"Use force=True to overwrite."
                    )
                    continue

                logger.info(
                    f"Creating {n_folds}-fold CV (scheme: {scheme_name}) for label: {lbl}"
                )

                # Prepare data
                data = self.labels.select(["sample", lbl])

                if grouping_col is not None and self.groupings is not None:
                    groups = self.groupings.select(["sample", grouping_col])
                    data = data.join(groups, on="sample", how="left", coalesce=True)

                # Filter nulls
                initial_count = data.height
                null_cols = (
                    [lbl] if grouping_col is None else [lbl, grouping_col]
                )
                data = data.drop_nulls(subset=null_cols)
                if data.height < initial_count:
                    logger.warning(
                        f"Dropped {initial_count - data.height} samples with null values for {lbl}"
                    )

                # Create CV folds
                self.splits[lbl].create_cv_folds(
                    data=data,
                    grouping=grouping_col,
                    n_folds=n_folds,
                    n_bins=n_bins,
                    random_state=random_state,
                    scheme_name=scheme_name,
                )

        return self

    def iter_cv_folds(
        self,
        label: Optional[str] = None,
        scheme_name: Optional[str] = None,
    ) -> Any:  # Returns generator
        """Iterate over CV folds across labels and schemes.

        Yields tuples of (label, scheme_name, cv_df) where cv_df has
        columns {sample, fold}.

        Args:
            label: Specific label to iterate over (all if None)
            scheme_name: Specific scheme to iterate over (all if None)

        Yields:
            Tuple of (label_name, scheme_name, cv_dataframe)
        """
        # Determine which labels to iterate
        if label is not None:
            if label not in self.splits:
                logger.warning(f"No splits found for label '{label}'")
                return
            label_list = [label]
        else:
            label_list = list(self.splits.keys())

        # Iterate over all label × scheme combinations
        for lbl in label_list:
            split_manager = self.splits[lbl]

            # Determine which schemes to iterate
            if scheme_name is not None:
                if scheme_name not in split_manager.cv_schemes:
                    logger.warning(
                        f"CV scheme '{scheme_name}' not found for label '{lbl}'"
                    )
                    continue
                scheme_list = [scheme_name]
            else:
                scheme_list = list(split_manager.cv_schemes.keys())

            # Yield each combination
            for scheme in scheme_list:
                cv_df = split_manager.cv_schemes[scheme]
                yield lbl, scheme, cv_df

    def apply_preprocessing(
        self,
        metadata_qc: bool = True,
        profiles_qc: bool = True,
        sync_after: bool = True,
        metadata_mbp_cutoff: int = 1000,
        profiles_cov_cutoff: float = 50.0,
        profiles_dominated_cutoff: float = 0.99,
        profiles_rank: Union[str, TaxonomicRanks] = TaxonomicRanks.ORDER,
    ) -> "Dataset":
        """Apply default QC methods from all internal classes.

        Steps:
        1. SampleMetadata QC (if metadata_qc=True):
           - Filters samples by sequencing depth (mbp_cutoff)

        2. TaxonomicProfiles QC (if profiles_qc=True):
           - Ensures filled format (fills if needed)
           - Converts to relative abundance (if needed)
           - Filters by coverage (cov_cutoff)
           - Filters dominated samples (dominated_cutoff at specified rank)

        3. Synchronize accessions across all components (if sync_after=True)

        Args:
            metadata_qc: Apply metadata quality control
            profiles_qc: Apply profiles quality control
            sync_after: Synchronize accessions after QC (recommended)
            metadata_mbp_cutoff: Minimum megabase pairs for metadata QC (default: 1000)
            profiles_cov_cutoff: Minimum coverage for profiles QC (default: 50.0)
            profiles_dominated_cutoff: Maximum single-group abundance (default: 0.99)
            profiles_rank: Taxonomic rank for domination check (default: ORDER)

        Returns:
            Self for chaining
        """
        # Apply metadata QC
        if (
            metadata_qc
            and self.metadata is not None
            and not self.metadata_qc_done
        ):
            self.metadata = self.metadata.default_qc(
                mbp_cutoff=metadata_mbp_cutoff
            )
            self.metadata_qc_done = True

        # Apply profiles QC
        if (
            profiles_qc
            and self.profiles is not None
            and not self.profiles_qc_done
        ):
            # Ensure profiles have root coverage data (do this BEFORE fill to avoid double counting)
            if self.profiles.root is None:
                self.profiles = self.profiles.create_root()

            # Ensure profiles are filled
            if not self.profiles.is_filled:
                self.profiles = self.profiles.fill_profiles()

            # Convert to relative abundance if needed
            profiles_lf = self.profiles.profiles
            if "coverage" in profiles_lf.collect_schema().names():
                # Need to convert from coverage to relabund
                self.profiles.profiles = self.profiles._to_relabund_lf()

            # Apply default QC (coverage and domination filters)
            self.profiles = self.profiles.default_qc(
                cov_cutoff=profiles_cov_cutoff,
                dominated_cutoff=profiles_dominated_cutoff,
                rank=profiles_rank,
            )
            self.profiles_qc_done = True

        # Synchronize accessions after QC
        if sync_after:
            self._sync_accessions()

        return self

    def save(self, path: Union[Path, str], compress: bool = False) -> None:
        """Save Dataset to directory structure with human-readable CSV files.

        Directory Structure:
            dataset/
                metadata/
                    metadata.csv
                    attributes.csv
                    study_titles.csv (if exists)
                profiles/
                    profiles.csv
                    root.csv (if exists)
                features/
                    {name}.csv (for each feature set)
                labels/
                    {name}.csv (for each label set)
                manifest.json (component metadata and sample tracking)

        Args:
            path: Directory or tar.gz path to save dataset
            compress: If True, package directory into tar.gz archive
        """
        path = Path(path)

        # Determine working directory
        if compress and str(path).endswith(".tar.gz"):
            # Remove .tar.gz extension for working directory
            work_dir = Path(str(path)[:-7])
        elif compress:
            # Add .tar.gz if compress=True but not in path
            work_dir = path
            path = Path(str(path) + ".tar.gz")
        else:
            work_dir = path

        # Create directory structure
        work_dir.mkdir(parents=True, exist_ok=True)

        # Save components
        manifest: Dict[str, Any] = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "components": {},
            "sample_ids": self._sample_ids if self._sample_ids else [],
        }

        # Save metadata
        if self.metadata is not None:
            metadata_dir = work_dir / "metadata"
            self.metadata.save(metadata_dir)
            metadata_lf = self.metadata.metadata
            n_samples = metadata_lf.select("sample").collect().height
            manifest["components"]["metadata"] = {
                "files": [
                    "metadata.csv",
                    "attributes.csv",
                    "study_titles.csv",
                ],
                "n_samples": n_samples,
            }

        # Save profiles
        if self.profiles is not None:
            profiles_dir = work_dir / "profiles"
            self.profiles.save(profiles_dir)
            manifest["components"]["profiles"] = {
                "files": (
                    ["profiles.csv", "root.csv"]
                    if self.profiles.root is not None
                    else ["profiles.csv"]
                ),
                "is_filled": self.profiles.is_filled,
            }

        # Save feature sets
        if self.feature_sets:
            features_dir = work_dir / "features"
            features_dir.mkdir(parents=True, exist_ok=True)
            manifest["components"]["features"] = {}

            for name, feature_set in self.feature_sets.items():
                fs_path = features_dir / f"{name}.csv"
                feature_set.save(fs_path)
                manifest["components"]["features"][name] = {
                    "file": f"{name}.csv",
                    "n_samples": len(feature_set.accessions),
                    "n_features": len(feature_set.feature_names),
                }

        # Save labels
        if self.labels is not None:
            labels_dir = work_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            labels_path = labels_dir / "labels.csv"
            self.labels.write_csv(labels_path)
            manifest["components"]["labels"] = {
                "file": "labels.csv",
                "n_samples": self.labels.height,
                "columns": [c for c in self.labels.columns if c != "sample"],
            }

        # Save groupings
        if self.groupings is not None:
            groupings_dir = work_dir / "groupings"
            groupings_dir.mkdir(parents=True, exist_ok=True)

            groupings_path = groupings_dir / "groupings.csv"
            self.groupings.write_csv(groupings_path)
            manifest["components"]["groupings"] = {
                "file": "groupings.csv",
                "n_samples": self.groupings.height,
                "columns": [
                    c for c in self.groupings.columns if c != "sample"
                ],
            }

        # Save splits (SplitManager structure)
        if self.splits:
            splits_dir = work_dir / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            manifest["components"]["splits"] = {}

            for label, split_manager in self.splits.items():
                label_dir = splits_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)

                label_info: Dict[str, Any] = {
                    "holdout": None,
                    "cv_schemes": [],
                }

                # Save holdout split
                if split_manager.holdout is not None:
                    holdout_path = label_dir / "holdout.csv"
                    split_manager.holdout.write_csv(holdout_path)
                    label_info["holdout"] = "holdout.csv"

                # Save CV schemes
                cv_schemes_list: List[str] = []
                for scheme_name, cv_df in split_manager.cv_schemes.items():
                    cv_path = label_dir / f"cv_{scheme_name}.csv"
                    cv_df.write_csv(cv_path)
                    cv_schemes_list.append(scheme_name)
                label_info["cv_schemes"] = cv_schemes_list

                manifest["components"]["splits"][label] = label_info

        # Save manifest
        manifest_path = work_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Compress if requested
        if compress:
            with tarfile.open(path, "w:gz") as tar:
                tar.add(work_dir, arcname=work_dir.name)

            # Clean up uncompressed directory
            import shutil

            shutil.rmtree(work_dir)

    @classmethod
    def load(cls, path: Union[Path, str], lazy: bool = True) -> "Dataset":
        """Load Dataset from directory or tar.gz archive.

        Args:
            path: Directory or tar.gz path containing saved dataset
            lazy: If True (default), scan components lazily; if False, read into memory

        Returns:
            Dataset instance with all components
        """
        path = Path(path)

        # Handle tar.gz extraction
        if str(path).endswith(".tar.gz") or (
            path.is_file() and tarfile.is_tarfile(path)
        ):
            import shutil
            import tempfile

            # Extract to temporary directory
            temp_dir = Path(tempfile.mkdtemp())
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(temp_dir)

            # Find the extracted directory
            extracted = list(temp_dir.iterdir())[0]

            # Load from extracted directory
            dataset = cls._load_from_directory(extracted, lazy=lazy)

            # Clean up
            shutil.rmtree(temp_dir)
            return dataset
        else:
            return cls._load_from_directory(path, lazy=lazy)

    @classmethod
    def _load_from_directory(cls, path: Path, lazy: bool = True) -> "Dataset":
        """Internal method to load from an uncompressed directory."""
        # Read manifest
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Create empty dataset
        dataset = cls()

        # Load metadata
        if "metadata" in manifest["components"]:
            metadata_dir = path / "metadata"
            if metadata_dir.exists():
                dataset.metadata = SampleMetadata.load(metadata_dir)

        # Load profiles
        if "profiles" in manifest["components"]:
            profiles_dir = path / "profiles"
            if profiles_dir.exists():
                dataset.profiles = TaxonomicProfiles.load(
                    profiles_dir, check_filled=False
                )

        # Load feature sets
        if "features" in manifest["components"]:
            features_dir = path / "features"
            for name, info in manifest["components"]["features"].items():
                fs_path = features_dir / info["file"]
                if fs_path.exists():
                    if lazy:
                        dataset.feature_sets[name] = FeatureSet.scan(
                            fs_path, name=name
                        )
                    else:
                        dataset.feature_sets[name] = FeatureSet.load(fs_path)
                        dataset.feature_sets[name].name = name

        # Load labels
        if "labels" in manifest["components"]:
            labels_info = manifest["components"]["labels"]
            labels_dir = path / "labels"

            # Check if new format (single file) or legacy (dict of files)
            if "file" in labels_info:
                # New format
                label_path = labels_dir / labels_info["file"]
                if label_path.exists():
                    if lazy:
                        dataset.labels = pl.scan_csv(label_path).collect()
                    else:
                        dataset.labels = pl.read_csv(label_path)
            else:
                # Legacy format
                for name, info in labels_info.items():
                    label_path = labels_dir / info["file"]
                    if label_path.exists():
                        # Use add_labels to merge legacy files
                        dataset.add_labels(label_path, name=name)

        # Load groupings
        if "groupings" in manifest["components"]:
            groupings_info = manifest["components"]["groupings"]
            groupings_dir = path / "groupings"

            if "file" in groupings_info:
                groupings_path = groupings_dir / groupings_info["file"]
                if groupings_path.exists():
                    if lazy:
                        dataset.groupings = pl.scan_csv(
                            groupings_path
                        ).collect()
                    else:
                        dataset.groupings = pl.read_csv(groupings_path)

        # Load splits (SplitManager structure)
        if "splits" in manifest["components"]:
            splits_info = manifest["components"]["splits"]
            splits_dir = path / "splits"

            # Check if new SplitManager format (dict of labels) or legacy format
            if isinstance(splits_info, dict) and "file" not in splits_info:
                # New format: Dict[label, {holdout, cv_schemes}]
                for label, label_info in splits_info.items():
                    label_dir = splits_dir / label
                    if not label_dir.exists():
                        continue

                    # Create SplitManager
                    split_manager = SplitManager(label)

                    # Load holdout
                    if label_info.get("holdout"):
                        holdout_path = label_dir / label_info["holdout"]
                        if holdout_path.exists():
                            if lazy:
                                split_manager.holdout = pl.scan_csv(
                                    holdout_path
                                ).collect()
                            else:
                                split_manager.holdout = pl.read_csv(
                                    holdout_path
                                )

                    # Load CV schemes
                    for scheme_name in label_info.get("cv_schemes", []):
                        cv_path = label_dir / f"cv_{scheme_name}.csv"
                        if cv_path.exists():
                            if lazy:
                                split_manager.cv_schemes[scheme_name] = (
                                    pl.scan_csv(cv_path).collect()
                                )
                            else:
                                split_manager.cv_schemes[scheme_name] = (
                                    pl.read_csv(cv_path)
                                )

                    dataset.splits[label] = split_manager

            elif "file" in splits_info:
                # Legacy format: single splits.csv file
                # Convert to new format by treating each column as a separate label's holdout
                splits_path = splits_dir / splits_info["file"]
                if splits_path.exists():
                    if lazy:
                        legacy_splits = pl.scan_csv(splits_path).collect()
                    else:
                        legacy_splits = pl.read_csv(splits_path)

                    # Each column (except 'sample') becomes a label's holdout split
                    for col in legacy_splits.columns:
                        if col == "sample":
                            continue
                        # Extract label name (remove '_split' suffix if present)
                        label = (
                            col.replace("_split", "")
                            if col.endswith("_split")
                            else col
                        )
                        split_manager = SplitManager(label)
                        split_manager.holdout = legacy_splits.select(
                            ["sample", col]
                        ).rename({col: "split"})
                        dataset.splits[label] = split_manager

        # Restore canonical sample IDs
        dataset._sample_ids = manifest.get("sample_ids", None)

        return dataset

    @classmethod
    def scan(cls, path: Union[Path, str]) -> "Dataset":
        """Lazy-load Dataset from directory or tar.gz without materializing
        data.

        Equivalent to load(path, lazy=True).

        Args:
            path: Directory or tar.gz path containing saved dataset

        Returns:
            Dataset instance with all components in lazy mode
        """
        return cls.load(path, lazy=True)
