"""Core data models for microbiome data structures."""

import numpy as np
import polars as pl
from typing import List, Any, Optional, TYPE_CHECKING, Union
from pathlib import Path

if TYPE_CHECKING:
    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

from microbiome_ml.utils.taxonomy import TaxonomicRanks


# standardised error messages
ERR_FEATURESET_NAME_UNDEFINED = "FeatureSet name must be defined"


class FeatureSet:
    """
    ML-ready feature set with numpy arrays and metadata.
    
    Stores features as numpy arrays with associated metadata lists
    for efficient ML operations while maintaining data provenance.
    """
    
    def __init__(
        self, 
        accessions: List[str], 
        feature_names: List[str], 
        features: Any,
        name: str,
        _is_lazy: bool = False):
        """
        Initialize FeatureSet with validation. Supports both eager (numpy) and lazy (LazyFrame) modes.
        
        Args:
            accessions: Ordered sample/species IDs
            feature_names: Ordered feature names
            features: Either numpy array (eager) or LazyFrame (lazy)
            name: Name for the FeatureSet
            _is_lazy: Internal flag indicating lazy mode (LazyFrame)
            
        Raises:
            ValueError: If dimensions don't match or name is missing
        """
        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)
            
        self.accessions = accessions
        self.feature_names = feature_names
        self.name = name
        self._is_lazy = _is_lazy
        
        if _is_lazy:
            # Lazy mode: features is a LazyFrame
            self._lf = features
            self.features = None
        else:
            # Eager mode: features is numpy array
            if features.shape[0] != len(accessions):
                raise ValueError(f"Features array rows ({features.shape[0]}) must match accessions length ({len(accessions)})")
            if features.shape[1] != len(feature_names):
                raise ValueError(f"Features array cols ({features.shape[1]}) must match feature_names length ({len(feature_names)})")
            self.features = features
            self._lf = None
        
        # Cache indices for O(1) lookups (always built for fast queries)
        self._accession_idx = {acc: i for i, acc in enumerate(accessions)}
        self._feature_idx = {fname: i for i, fname in enumerate(feature_names)}
        
    @classmethod
    def scan(
        cls,
        path: Union[Path, str],
        name: str,
        acc_column: Optional[str] = None,
    ) -> "FeatureSet":
        """
        Lazily load features from a file without pulling into memory.
        
        Args:
            path: Path to CSV file
            name: Name for the FeatureSet
            acc_column: Column name for accessions (auto-detects 'acc' or 'sample' if None)
            
        Returns:
            LazyFeatureSet instance (lazy-loaded)
        """
        lf = pl.scan_csv(path)
        schema = lf.collect_schema()
        
        # Auto-detect accession column
        if acc_column is None:
            if 'acc' in schema.names():
                acc_column = 'acc'
            elif 'sample' in schema.names():
                acc_column = 'sample'
            else:
                raise ValueError("No accession column found. Expected 'acc' or 'sample' column, or specify acc_column")
        
        # Extract accessions (requires collecting just that column)
        accessions = lf.select(acc_column).collect().to_series().to_list()
        feature_names = [col for col in schema.names() if col != acc_column]
        
        return cls(
            accessions=accessions,
            feature_names=feature_names,
            features=lf,
            name=name,
            _is_lazy=True
        )
        
    @classmethod
    def from_df(
        cls, 
        df: pl.DataFrame, 
        name: str,
        acc_column: Optional[str] = None,
        ) -> "FeatureSet":
        """
        Create FeatureSet from wide-form DataFrame (features only, no labels).
        
        Args:
            df: Wide-form DataFrame with features as columns
            acc_column: Column name for accessions (defaults to auto-detect)
            name: Optional name for the FeatureSet
            
        Returns:
            FeatureSet instance
        """
        # If acc_column=None, test for 'acc' first, then 'sample'
        if acc_column is None:
            if 'acc' in df.columns:
                acc_column = 'acc'
            elif 'sample' in df.columns:
                acc_column = 'sample'
            else:
                raise ValueError("No accession column found. Expected 'acc' or 'sample' column, or specify acc_column")

        if acc_column not in df.columns:
            raise ValueError(f"Specified acc_column '{acc_column}' not found in DataFrame columns")

        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)
        
        # Extract components
        accessions = df.select(acc_column).to_series().to_list()
        
        # Feature columns are everything except acc_column
        feature_columns = [col for col in df.columns if col != acc_column]
        feature_names = feature_columns
        
        # Convert features to numpy array
        features = df.select(feature_columns).to_numpy()
        
        # All validation happens in __init__
        return cls(accessions=accessions, feature_names=feature_names,
                  features=features, name=name)

    
    @classmethod
    def from_profiles(
        cls, 
        profiles: "TaxonomicProfiles", 
        sample_ids: List[str], 
        name: str,
        rank: "TaxonomicRanks",
        ) -> "FeatureSet":
        """
        Create FeatureSet from TaxonomicProfiles at specified taxonomic rank.
        
        Args:
            profiles: TaxonomicProfiles instance
            sample_ids: List of sample IDs to include
            rank: Taxonomic rank to extract features from
            name: Optional name for the FeatureSet
            
        Returns:
            FeatureSet instance with features at the specified rank
        """

        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)
            
        # Convert rank to enum if string
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)
        
        # Filter profiles to specified samples
        sample_filter = pl.DataFrame({"sample": sample_ids}).lazy()
        filtered_profiles = profiles._filter_by_sample(sample_filter)
        
        # Use the TaxonomicProfiles create_features method
        feature_set = filtered_profiles.create_features(rank)
        
        # Set name if provided
        if name is not None:
            feature_set.name = name
        
        return feature_set

    def materialize(self) -> "FeatureSet":
        """
        Convert lazy FeatureSet to eager (numpy) mode.
        
        If already eager, returns self. If lazy, collects LazyFrame and converts to numpy.
        
        Returns:
            Eager FeatureSet with numpy array
        """
        if self._is_lazy:
            df = self._lf.collect()
            return FeatureSet.from_df(df, name=self.name)
        return self

    @classmethod
    def from_lf(cls, lf: pl.LazyFrame, name: str) -> "FeatureSet":
        """
        Create FeatureSet from a LazyFrame.
        
        Args:
            lf: Polars LazyFrame with features
            name: Optional name for the FeatureSet
        Returns:
            FeatureSet instance
        """
        # Collect LazyFrame to DataFrame
        df = lf.collect()
        
        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)
        
        # Create FeatureSet from DataFrame
        return cls.from_df(df, name=name)

    
    def to_df(
        self, 
        ) -> pl.DataFrame:
        """
        Convert FeatureSet back to wide-form DataFrame.
        
        For lazy FeatureSets, this will materialize the LazyFrame first.
            
        Returns:
            Wide-form DataFrame with accessions and features
        """
        if self._is_lazy:
            # Lazy mode: collect and return
            return self._lf.collect()
        else:
            # Eager mode: construct from numpy array
            df_dict = {"sample": self.accessions}
            for i, feature_name in enumerate(self.feature_names):
                df_dict[feature_name] = self.features[:, i]
            return pl.DataFrame(df_dict)

    
    def get_samples(self, sample_ids: List[str]) -> np.ndarray:
        """
        Get features for specific samples.
        
        Works in both lazy and eager modes. In lazy mode, collects only requested samples.
        
        Args:
            sample_ids: List of sample IDs to retrieve
            
        Returns:
            numpy array of shape (len(sample_ids), n_features)
            
        Raises:
            ValueError: If any sample ID not found
        """
        if self._is_lazy:
            # Lazy mode: collect only requested samples
            acc_col = 'acc' if 'acc' in self._lf.collect_schema().names() else 'sample'
            df = self._lf.filter(pl.col(acc_col).is_in(sample_ids)).collect()
            return df.drop(acc_col).to_numpy()
        else:
            # Eager mode: use cached index lookup (O(1))
            try:
                indices = [self._accession_idx[sample_id] for sample_id in sample_ids]
            except KeyError as e:
                raise ValueError(f"Sample ID {e} not found in FeatureSet")
            return self.features[indices, :]
    
    def get_features(self, feature_names: List[str]) -> np.ndarray:
        """
        Get features for specific feature names.
        
        Works in both lazy and eager modes. In lazy mode, collects only requested features.
        
        Args:
            feature_names: List of feature names to retrieve
            
        Returns:
            numpy array of shape (n_samples, len(feature_names))
            
        Raises:
            ValueError: If any feature name not found
        """
        if self._is_lazy:
            # Lazy mode: collect only requested features
            acc_col = 'acc' if 'acc' in self._lf.collect_schema().names() else 'sample'
            df = self._lf.select([acc_col] + feature_names).collect()
            return df.drop(acc_col).to_numpy()
        else:
            # Eager mode: use cached index lookup (O(1))
            try:
                indices = [self._feature_idx[fname] for fname in feature_names]
            except KeyError as e:
                raise ValueError(f"Feature name {e} not found in FeatureSet")
            return self.features[:, indices]
    
    def get_sample_accs(self) -> List[str]:
        """Get list of all sample/accession IDs."""
        return self.accessions.copy()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names.copy()
    
    def filter_samples(self, sample_ids: List[str]) -> "FeatureSet":
        """
        Filter FeatureSet to specific samples.
        
        Works in both lazy and eager modes. Preserves the mode of the original FeatureSet.
        
        Args:
            sample_ids: List of sample IDs to keep
            
        Returns:
            New FeatureSet with filtered samples (same mode as original)
        """
        # Find indices of samples to keep
        keep_indices = [i for i, acc in enumerate(self.accessions) if acc in sample_ids]
        
        if not keep_indices:
            raise ValueError("No matching samples found")
        
        # Filter accessions list
        filtered_accessions = [self.accessions[i] for i in keep_indices]
        
        if self._is_lazy:
            # Lazy mode: filter LazyFrame
            acc_col = 'acc' if 'acc' in self._lf.collect_schema().names() else 'sample'
            filtered_lf = self._lf.filter(pl.col(acc_col).is_in(sample_ids))
            return FeatureSet(
                accessions=filtered_accessions,
                feature_names=self.feature_names.copy(),
                features=filtered_lf,
                name=self.name,
                _is_lazy=True
            )
        else:
            # Eager mode: filter numpy array
            filtered_features = self.features[keep_indices, :]
            return FeatureSet(
                accessions=filtered_accessions,
                feature_names=self.feature_names.copy(),
                features=filtered_features,
                name=self.name,
                _is_lazy=False
            )

    def save(self, path: Union[Path, str]) -> None:
        """
        Save FeatureSet to disk as a .csv file.
        
        For lazy FeatureSets, this will materialize the data first.
        
        Args:
            path: Path to save the .csv file
        """
        path = Path(path)
        
        if self._is_lazy:
            # Materialize lazy data before saving
            self._lf.collect().write_csv(path)
        else:
            # Eager mode: use to_df()
            self.to_df().write_csv(path)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "FeatureSet":
        """
        Load FeatureSet from a .csv file.
        
        Args:
            path: Path to the .csv file to load from
        Returns:
            Loaded FeatureSet instance
        """
        path = Path(path)
        
        # Read CSV into DataFrame
        df = pl.read_csv(path)
        
        # Create FeatureSet from DataFrame
        return cls.from_df(df, name=path.stem)
        