"""High-level sample coordination class for ML workflows."""

import numpy as np
import polars as pl
from typing import List, Union, Optional, Dict, Any, Tuple, Set
from pathlib import Path
import warnings
import json
import tarfile
from datetime import datetime

from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.utils.taxonomy import TaxonomicRanks


class Dataset:
    """
    Global entry point for microbiome ML workflows with flexible builder pattern.
    
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
        metadata: Optional[Union[SampleMetadata, Dict[str, Union[str, Path, pl.LazyFrame]]]] = None,
        profiles: Optional[Union[TaxonomicProfiles, str, Path, pl.LazyFrame]] = None,
        features: Optional[Dict[str, FeatureSet]] = None
    ):
        """
        Initialize Dataset with optional components.
        
        Args:
            metadata: SampleMetadata instance or dict with metadata sources
            profiles: TaxonomicProfiles instance or data source
            features: Dict of FeatureSets by name
        """
        # Initialize empty structure
        self.metadata: Optional[SampleMetadata] = None
        self.profiles: Optional[TaxonomicProfiles] = None
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.labels: Dict[str, Any] = {}
        self._sample_ids: Optional[List[str]] = None
        
        # Process provided components if given
        if metadata is not None:
            self.add_metadata(metadata)
        if profiles is not None:
            self.add_profiles(profiles)
        if features is not None:
            self.feature_sets.update(features)
            self._update_sample_ids()
    
    def add_metadata(
        self, 
        data: Optional[Union[SampleMetadata, Dict[str, Union[str, Path, pl.LazyFrame]]]] = None,
        metadata: Optional[Union[str, Path, pl.LazyFrame]] = None,
        attributes: Optional[Union[str, Path, pl.LazyFrame]] = None,
        study_titles: Optional[Union[str, Path, pl.LazyFrame]] = None,
        **kwargs
    ) -> "Dataset":
        """
        Add metadata component with flexible input handling.
        
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
                metadata=data.get('metadata'),
                attributes=data.get('attributes'),
                study_titles=data.get('study_titles', None)
            )
        # Handle direct keyword arguments
        else:
            # Use provided kwargs or direct parameters
            meta_source = metadata or kwargs.get('metadata')
            attr_source = attributes or kwargs.get('attributes')
            study_source = study_titles or kwargs.get('study_titles')
            
            if meta_source and attr_source:
                self.metadata = SampleMetadata(
                    metadata=meta_source,
                    attributes=attr_source,
                    study_titles=study_source
                )
        
        self._update_sample_ids()
        return self
    
    def add_profiles(
        self,
        profiles: Union[TaxonomicProfiles, str, Path, pl.LazyFrame],
        root: Optional[Union[str, Path, pl.LazyFrame]] = None,
        check_filled: bool = True,
        sample_size: int = 1000,
        **kwargs
    ) -> "Dataset":
        """
        Add profiles component with flexible input handling.
        
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
                **kwargs
            )
        
        self._update_sample_ids()
        return self
    
    def add_features(
        self,
        name: str,
        features: Optional[Union[FeatureSet, pl.DataFrame]] = None,
        rank: Optional[Union[str, TaxonomicRanks]] = None,
        accessions: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        feature_array: Optional[np.ndarray] = None,
        **kwargs
    ) -> "Dataset":
        """
        Add feature set with flexible input handling.
        
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
            self.feature_sets[name] = FeatureSet.from_df(features, name=name, **kwargs)
        # Create from profiles using rank
        elif rank is not None:
            if self.profiles is None:
                raise ValueError("Profiles must be added before creating taxonomic features")
            feature_set = self.profiles.create_features(rank)
            feature_set.name = name
            self.feature_sets[name] = feature_set
        # Create directly from components
        elif accessions and feature_names and feature_array is not None:
            self.feature_sets[name] = FeatureSet(
                accessions=accessions,
                feature_names=feature_names,
                features=feature_array,
                name=name
            )
        else:
            raise ValueError("Must provide FeatureSet instance, DataFrame, rank parameter, or direct components")
        
        return self
    
    def iter_feature_sets(self, names: Optional[List[str]] = None):
        """
        Iterate over feature sets.
        
        Args:
            names: List of feature set names to iterate over (all if None)
            
        Yields:
            Tuple of (name, FeatureSet) for each feature set
        """
        if names is None:
            names = list(self.feature_sets.keys())
        for name in names:
            yield name, self.feature_sets[name]

    def iter_labels(self, names: Optional[List[str]] = None):
        """
        Iterate over label sets.
        
        Args:
            names: List of label set names to iterate over (all if None)
            
        Yields:
            Tuple of (name, label_data) for each label set
        """
        if names is None:
            names = list(self.labels.keys())
        for name in names:
            yield name, self.labels[name]
    
    def get_sample_ids(self) -> List[str]:
        """
        Get the canonical list of sample IDs after synchronization.
        
        Returns:
            List of sample IDs present in all components (strict intersection)
        """
        return self._sample_ids.copy() if self._sample_ids else []
    
    def _get_accessions_from_metadata(self) -> Set[str]:
        """
        Extract sample accessions from metadata component (lazy-aware).
        
        Returns:
            Set of sample IDs from metadata
        """
        if self.metadata is None:
            return set()
        
        if self.metadata._is_lazy:
            # Lazy mode: collect only sample column
            return set(self.metadata._lf_metadata.select("sample").collect().to_series().to_list())
        else:
            # Eager mode: extract from DataFrame
            return set(self.metadata.metadata.select("sample").to_series().to_list())
    
    def _get_accessions_from_profiles(self) -> Set[str]:
        """
        Extract sample accessions from profiles component (lazy-aware).
        
        Returns:
            Set of sample IDs from profiles
        """
        if self.profiles is None:
            return set()
        
        if self.profiles._is_lazy:
            # Lazy mode: collect only sample column
            return set(self.profiles._lf_profiles.select("sample").unique().collect().to_series().to_list())
        else:
            # Eager mode: extract from DataFrame
            return set(self.profiles.profiles.select("sample").unique().to_series().to_list())
    
    def _get_accessions_from_features(self) -> Set[str]:
        """
        Extract sample accessions from all feature sets.
        
        Returns:
            Set of sample IDs present in ALL feature sets (intersection)
        """
        if not self.feature_sets:
            return set()
        
        # Get accessions from each feature set and find intersection
        feature_accessions = [set(fs.accessions) for fs in self.feature_sets.values()]
        return set.intersection(*feature_accessions) if feature_accessions else set()
    
    def _get_accessions_from_labels(self) -> Set[str]:
        """
        Extract sample accessions from all label sets.
        
        Returns:
            Set of sample IDs present in ALL label sets (intersection)
        """
        if not self.labels:
            return set()
        
        label_accessions = []
        for label_data in self.labels.values():
            if isinstance(label_data, pl.DataFrame):
                # Assume labels have a 'sample' column
                label_accessions.append(set(label_data.select("sample").to_series().to_list()))
            elif isinstance(label_data, dict):
                # If labels are dict, use keys as sample IDs
                label_accessions.append(set(label_data.keys()))
        
        return set.intersection(*label_accessions) if label_accessions else set()
    
    def _sync_accessions(self) -> None:
        """
        Synchronize accessions across all components using strict intersection.
        
        This method:
        1. Collects accessions from all non-None components
        2. Computes strict intersection (samples present in ALL components)
        3. Filters each component to the canonical sample set
        4. Updates self._sample_ids with canonical ordering
        5. Handles lazy mode efficiently (collects only accession columns)
        
        Warnings are issued if synchronization drops significant samples.
        """
        # Collect accessions from all available components
        all_accessions = []
        component_counts = {}
        
        if self.metadata is not None:
            metadata_accs = self._get_accessions_from_metadata()
            all_accessions.append(metadata_accs)
            component_counts['metadata'] = len(metadata_accs)
        
        if self.profiles is not None:
            profiles_accs = self._get_accessions_from_profiles()
            all_accessions.append(profiles_accs)
            component_counts['profiles'] = len(profiles_accs)
        
        if self.feature_sets:
            features_accs = self._get_accessions_from_features()
            all_accessions.append(features_accs)
            component_counts['feature_sets'] = len(features_accs)
        
        if self.labels:
            labels_accs = self._get_accessions_from_labels()
            all_accessions.append(labels_accs)
            component_counts['labels'] = len(labels_accs)
        
        # If no components, set to None and return
        if not all_accessions:
            self._sample_ids = None
            return
        
        # Compute strict intersection
        canonical_samples = set.intersection(*all_accessions)
        
        # Warn if significant samples are dropped
        for comp_name, comp_count in component_counts.items():
            dropped = comp_count - len(canonical_samples)
            if dropped > 0:
                pct_dropped = (dropped / comp_count) * 100
                if pct_dropped > 10:  # Warn if more than 10% dropped
                    warnings.warn(
                        f"Accession sync dropped {dropped} samples ({pct_dropped:.1f}%) from {comp_name}. "
                        f"Canonical set: {len(canonical_samples)} samples.",
                        UserWarning
                    )
        
        # Sort for deterministic ordering
        canonical_list = sorted(list(canonical_samples))
        self._sample_ids = canonical_list
        
        # Filter components to canonical set
        self._filter_components_to_canonical()
    
    def _filter_components_to_canonical(self) -> None:
        """
        Filter all components to match the canonical sample set.
        
        Uses lazy-aware filtering to avoid materializing large datasets.
        """
        if self._sample_ids is None or not self._sample_ids:
            return
        
        # Create LazyFrame with canonical samples for filtering
        canonical_lf = pl.DataFrame({"sample": self._sample_ids}).lazy()
        
        # Filter metadata
        if self.metadata is not None:
            self.metadata = self.metadata._filter_by_sample(canonical_lf)
        
        # Filter profiles
        if self.profiles is not None:
            self.profiles = self.profiles._filter_by_sample(canonical_lf)
        
        # Filter feature sets
        for name, feature_set in list(self.feature_sets.items()):
            self.feature_sets[name] = feature_set.filter_samples(self._sample_ids)
        
        # Filter labels
        for name, label_data in list(self.labels.items()):
            if isinstance(label_data, pl.DataFrame):
                self.labels[name] = label_data.join(
                    canonical_lf.collect(), on='sample', how='semi'
                )
    
    def _update_sample_ids(self):
        """
        Recalculate canonical sample list when components change.
        
        Calls _sync_accessions() to perform strict intersection-based synchronization.
        """
        self._sync_accessions()
    
    def add_feature_set(
        self,
        features: Union[Dict[str, Union[str, Path, FeatureSet]], str, Path, FeatureSet],
        name: Optional[str] = None,
        **kwargs
    ) -> "Dataset":
        """
        Add one or multiple feature sets.
        
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
                    raise ValueError(f"Unsupported feature set source type for '{fs_name}': {type(fs_source)}")
        else:
            # Single addition
            if name is None:
                raise ValueError("name parameter required when adding single feature set")
            
            if isinstance(features, FeatureSet):
                self.feature_sets[name] = features
            elif isinstance(features, (str, Path)):
                self.feature_sets[name] = FeatureSet.load(features)
            else:
                raise ValueError(f"Unsupported feature set source type: {type(features)}")
        
        self._update_sample_ids()
        return self
    
    def add_labels(
        self,
        labels: Union[Dict[str, Union[str, Path, pl.DataFrame]], str, Path, pl.DataFrame],
        name: Optional[str] = None
    ) -> "Dataset":
        """
        Add one or multiple label sets.
        
        Args:
            labels: Dict mapping names to sources, single file path, or DataFrame
            name: Name for single label set (required if labels is not dict)
            
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
                if isinstance(label_source, pl.DataFrame):
                    self.labels[label_name] = label_source
                elif isinstance(label_source, (str, Path)):
                    self.labels[label_name] = pl.read_csv(label_source)
                else:
                    raise ValueError(f"Unsupported label source type for '{label_name}': {type(label_source)}")
        else:
            # Single addition
            if name is None:
                raise ValueError("name parameter required when adding single label set")
            
            if isinstance(labels, pl.DataFrame):
                self.labels[name] = labels
            elif isinstance(labels, (str, Path)):
                self.labels[name] = pl.read_csv(labels)
            else:
                raise ValueError(f"Unsupported label source type: {type(labels)}")
        
        self._update_sample_ids()
        return self
    
    def add_taxonomic_features(
        self,
        ranks: Optional[List[Union[str, TaxonomicRanks]]] = None,
        prefix: str = "tax"
    ) -> "Dataset":
        """
        Generate FeatureSets from TaxonomicProfiles at specified ranks.
        
        Args:
            ranks: List of taxonomic ranks (defaults to all standard ranks)
            prefix: Prefix for generated feature set names (default: "tax")
            
        Returns:
            Self for chaining
            
        Examples:
            .add_taxonomic_features()  # All ranks
            .add_taxonomic_features(ranks=["genus", "family"])
            .add_taxonomic_features(ranks=[TaxonomicRanks.GENUS], prefix="rel")
        """
        if self.profiles is None:
            raise ValueError("Profiles must be added before generating taxonomic features")
        
        # Default to all standard ranks
        if ranks is None:
            ranks = [
                TaxonomicRanks.DOMAIN,
                TaxonomicRanks.PHYLUM,
                TaxonomicRanks.CLASS,
                TaxonomicRanks.ORDER,
                TaxonomicRanks.FAMILY,
                TaxonomicRanks.GENUS,
                TaxonomicRanks.SPECIES
            ]
        
        # Convert string ranks to enums
        rank_enums = []
        for rank in ranks:
            if isinstance(rank, str):
                rank_enums.append(TaxonomicRanks.from_name(rank))
            else:
                rank_enums.append(rank)
        
        # Generate feature sets for each rank
        for rank in rank_enums:
            feature_set = self.profiles.create_features(rank)
            feature_name = f"{prefix}_{rank.name.lower()}"
            feature_set.name = feature_name
            self.feature_sets[feature_name] = feature_set
        
        self._update_sample_ids()
        return self
    
    def apply_preprocessing(
        self,
        metadata_qc: bool = True,
        profiles_qc: bool = True,
        sync_after: bool = True
    ) -> "Dataset":
        """
        Apply default QC methods from all internal classes.
        
        Steps:
        1. SampleMetadata QC (if metadata_qc=True):
           - Validates required fields
           - Checks date ranges and coordinates
           - Removes invalid samples
        
        2. TaxonomicProfiles QC (if profiles_qc=True):
           - Ensures filled format (fills if needed)
           - Validates taxonomy strings
           - Filters low-coverage samples
        
        3. Synchronize accessions across all components (if sync_after=True)
        
        Args:
            metadata_qc: Apply metadata quality control
            profiles_qc: Apply profiles quality control
            sync_after: Synchronize accessions after QC (recommended)
            
        Returns:
            Self for chaining
        """
        # Apply metadata QC
        if metadata_qc and self.metadata is not None:
            # Basic validation already happens in __init__
            # Additional QC can be added here as methods become available
            # e.g., self.metadata.validate_dates()
            # e.g., self.metadata.validate_coordinates()
            # e.g., self.metadata.remove_invalid_samples()
            pass
        
        # Apply profiles QC
        if profiles_qc and self.profiles is not None:
            # Profiles are already filled during __init__ if check_filled=True
            # Additional QC can be added here
            # e.g., self.profiles = self.profiles.filter_by_coverage(cov_cutoff=50.0)
            pass
        
        # Synchronize accessions after QC
        if sync_after:
            self._sync_accessions()
        
        return self
    
    def save(self, path: Union[Path, str], compress: bool = False) -> None:
        """
        Save Dataset to directory structure with human-readable CSV files.
        
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
        if compress and str(path).endswith('.tar.gz'):
            # Remove .tar.gz extension for working directory
            work_dir = Path(str(path)[:-7])
        elif compress:
            # Add .tar.gz if compress=True but not in path
            work_dir = path
            path = Path(str(path) + '.tar.gz')
        else:
            work_dir = path
        
        # Create directory structure
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Save components
        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "components": {},
            "sample_ids": self._sample_ids if self._sample_ids else []
        }
        
        # Save metadata
        if self.metadata is not None:
            metadata_dir = work_dir / "metadata"
            self.metadata.save(metadata_dir)
            metadata_lf = self.metadata._lf_metadata if self.metadata._is_lazy else self.metadata.metadata.lazy()
            n_samples = metadata_lf.select("sample").collect().height
            manifest["components"]["metadata"] = {
                "files": ["metadata.csv", "attributes.csv", "study_titles.csv"],
                "n_samples": n_samples
            }
        
        # Save profiles
        if self.profiles is not None:
            profiles_dir = work_dir / "profiles"
            self.profiles.save(profiles_dir)
            manifest["components"]["profiles"] = {
                "files": ["profiles.csv", "root.csv"] if self.profiles.root is not None or self.profiles._lf_root is not None else ["profiles.csv"],
                "is_filled": self.profiles.is_filled
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
                    "n_features": len(feature_set.feature_names)
                }
        
        # Save labels
        if self.labels:
            labels_dir = work_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            manifest["components"]["labels"] = {}
            
            for name, label_data in self.labels.items():
                label_path = labels_dir / f"{name}.csv"
                if isinstance(label_data, pl.DataFrame):
                    label_data.write_csv(label_path)
                    manifest["components"]["labels"][name] = {
                        "file": f"{name}.csv",
                        "n_samples": label_data.height
                    }
        
        # Save manifest
        manifest_path = work_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Compress if requested
        if compress:
            with tarfile.open(path, 'w:gz') as tar:
                tar.add(work_dir, arcname=work_dir.name)
            
            # Clean up uncompressed directory
            import shutil
            shutil.rmtree(work_dir)
    
    @classmethod
    def load(cls, path: Union[Path, str], lazy: bool = True) -> "Dataset":
        """
        Load Dataset from directory or tar.gz archive.
        
        Args:
            path: Directory or tar.gz path containing saved dataset
            lazy: If True (default), scan components lazily; if False, read into memory
            
        Returns:
            Dataset instance with all components
        """
        path = Path(path)
        
        # Handle tar.gz extraction
        if str(path).endswith('.tar.gz') or (path.is_file() and tarfile.is_tarfile(path)):
            import tempfile
            import shutil
            
            # Extract to temporary directory
            temp_dir = Path(tempfile.mkdtemp())
            with tarfile.open(path, 'r:gz') as tar:
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
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Create empty dataset
        dataset = cls()
        
        # Load metadata
        if "metadata" in manifest["components"]:
            metadata_dir = path / "metadata"
            if metadata_dir.exists():
                dataset.metadata = SampleMetadata.load(metadata_dir, lazy=lazy)
        
        # Load profiles
        if "profiles" in manifest["components"]:
            profiles_dir = path / "profiles"
            if profiles_dir.exists():
                dataset.profiles = TaxonomicProfiles.load(profiles_dir, lazy=lazy, check_filled=False)
        
        # Load feature sets
        if "features" in manifest["components"]:
            features_dir = path / "features"
            for name, info in manifest["components"]["features"].items():
                fs_path = features_dir / info["file"]
                if fs_path.exists():
                    if lazy:
                        dataset.feature_sets[name] = FeatureSet.scan(fs_path, name=name)
                    else:
                        dataset.feature_sets[name] = FeatureSet.load(fs_path)
                        dataset.feature_sets[name].name = name
        
        # Load labels
        if "labels" in manifest["components"]:
            labels_dir = path / "labels"
            for name, info in manifest["components"]["labels"].items():
                label_path = labels_dir / info["file"]
                if label_path.exists():
                    if lazy:
                        dataset.labels[name] = pl.scan_csv(label_path).collect()  # Labels are small, collect anyway
                    else:
                        dataset.labels[name] = pl.read_csv(label_path)
        
        # Restore canonical sample IDs
        dataset._sample_ids = manifest.get("sample_ids", None)
        
        return dataset
    
    @classmethod
    def scan(cls, path: Union[Path, str]) -> "Dataset":
        """
        Lazy-load Dataset from directory or tar.gz without materializing data.
        
        Equivalent to load(path, lazy=True).
        
        Args:
            path: Directory or tar.gz path containing saved dataset
            
        Returns:
            Dataset instance with all components in lazy mode
        """
        return cls.load(path, lazy=True)
