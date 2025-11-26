"""Core data models for microbiome data structures."""

import numpy as np
import polars as pl
from typing import List, Any, Optional, TYPE_CHECKING, Union, Set
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
        name: str):
        """
        Initialize FeatureSet with validation. Always stores as LazyFrame.
        
        Args:
            accessions: Ordered sample/species IDs
            feature_names: Ordered feature names
            features: LazyFrame, DataFrame, or numpy array
            name: Name for the FeatureSet
            
        Raises:
            ValueError: If dimensions don't match or name is missing
        """
        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)
            
        self.accessions = accessions
        self.feature_names = feature_names
        self.name = name
        
        if isinstance(features, pl.LazyFrame):
            self.features = features
        elif isinstance(features, pl.DataFrame):
            self.features = features.lazy()
        elif isinstance(features, np.ndarray):
            # Convert numpy array to LazyFrame
            if features.shape[0] != len(accessions):
                raise ValueError(f"Features array rows ({features.shape[0]}) must match accessions length ({len(accessions)})")
            if features.shape[1] != len(feature_names):
                raise ValueError(f"Features array cols ({features.shape[1]}) must match feature_names length ({len(feature_names)})")
            
            # Construct dictionary for DataFrame creation
            # Note: This can be slow for very large arrays with many columns
            data = {"sample": accessions}
            # Use a more efficient way if possible, but for now:
            # Create DataFrame from numpy array directly with schema
            schema = ["sample"] + feature_names
            # Prepend accessions column to features
            # This is tricky with mixed types (str vs float). 
            # Better to create DF from features and add sample column
            df = pl.DataFrame(features, schema=feature_names)
            df = df.with_columns(pl.Series("sample", accessions)).select(["sample"] + feature_names)
            self.features = df.lazy()
        else:
            raise TypeError(f"Unsupported type for features: {type(features)}")
        
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
            FeatureSet instance (lazy-loaded)
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
            name=name
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
        
        # Pass DataFrame directly (will be converted to LazyFrame in __init__)
        return cls(accessions=accessions, feature_names=feature_names,
                  features=df, name=name)

    
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

    def _get_sample_list(self) -> Set[str]:
        """
        Extract sample IDs from this feature set.
        
        Returns:
            Set of sample IDs (accessions)
        """
        return set(self.accessions)
    
    def collect(self) -> pl.DataFrame:
        """
        Collect the internal LazyFrame to a DataFrame.
        
        Returns:
            DataFrame containing features and accessions
        """
        return self.features.collect()

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
        # We need accessions and feature names. 
        # This requires collecting schema and accessions column.
        schema = lf.collect_schema()
        
        acc_column = None
        if 'acc' in schema.names():
            acc_column = 'acc'
        elif 'sample' in schema.names():
            acc_column = 'sample'
        else:
            raise ValueError("No accession column found. Expected 'acc' or 'sample' column")
            
        accessions = lf.select(acc_column).collect().to_series().to_list()
        feature_names = [col for col in schema.names() if col != acc_column]
        
        if name is None:
            raise ValueError(ERR_FEATURESET_NAME_UNDEFINED)
        
        return cls(accessions=accessions, feature_names=feature_names, features=lf, name=name)

    
    def to_df(
        self, 
        ) -> pl.DataFrame:
        """
        Convert FeatureSet back to wide-form DataFrame.
        
        Returns:
            Wide-form DataFrame with accessions and features
        """
        return self.features.collect()

    
    def get_samples(self, sample_ids: List[str]) -> np.ndarray:
        """
        Get features for specific samples.
        
        Args:
            sample_ids: List of sample IDs to retrieve
            
        Returns:
            numpy array of shape (len(sample_ids), n_features)
            
        Raises:
            ValueError: If any sample ID not found
        """
        acc_col = 'acc' if 'acc' in self.features.collect_schema().names() else 'sample'
        
        # Filter and collect
        df = self.features.filter(pl.col(acc_col).is_in(sample_ids)).collect()
        
        # Ensure order matches sample_ids
        # This is important because filter doesn't guarantee order if not sorted
        # But we can reindex in numpy or polars
        
        # Check if all samples found
        found_samples = set(df[acc_col].to_list())
        missing = set(sample_ids) - found_samples
        if missing:
             raise ValueError(f"Sample IDs {missing} not found in FeatureSet")
             
        # Reorder to match input sample_ids
        # Create a mapping df
        order_df = pl.DataFrame({acc_col: sample_ids, "order": range(len(sample_ids))})
        df = df.join(order_df, on=acc_col).sort("order").drop("order")
        
        return df.drop(acc_col).to_numpy()
    
    def get_features(self, feature_names: List[str]) -> np.ndarray:
        """
        Get features for specific feature names.
        
        Args:
            feature_names: List of feature names to retrieve
            
        Returns:
            numpy array of shape (n_samples, len(feature_names))
            
        Raises:
            ValueError: If any feature name not found
        """
        acc_col = 'acc' if 'acc' in self.features.collect_schema().names() else 'sample'
        
        # Check if features exist
        schema_names = self.features.collect_schema().names()
        missing = set(feature_names) - set(schema_names)
        if missing:
            raise ValueError(f"Feature names {missing} not found in FeatureSet")
            
        df = self.features.select([acc_col] + feature_names).collect()
        return df.drop(acc_col).to_numpy()
    
    def get_sample_accs(self) -> List[str]:
        """Get list of all sample/accession IDs."""
        return self.accessions.copy()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names.copy()
    
    def filter_samples(self, sample_ids: List[str]) -> "FeatureSet":
        """
        Filter FeatureSet to specific samples.
        
        Args:
            sample_ids: List of sample IDs to keep
            
        Returns:
            New FeatureSet with filtered samples
        """
        # Find indices of samples to keep
        keep_indices = [i for i, acc in enumerate(self.accessions) if acc in sample_ids]
        
        if not keep_indices:
            raise ValueError("No matching samples found")
        
        # Filter accessions list
        filtered_accessions = [self.accessions[i] for i in keep_indices]
        
        acc_col = 'acc' if 'acc' in self.features.collect_schema().names() else 'sample'
        filtered_lf = self.features.filter(pl.col(acc_col).is_in(sample_ids))
        
        return FeatureSet(
            accessions=filtered_accessions,
            feature_names=self.feature_names.copy(),
            features=filtered_lf,
            name=self.name
        )

    def save(self, path: Union[Path, str]) -> None:
        """
        Save FeatureSet to disk as a .csv file.
        
        Args:
            path: Path to save the .csv file
        """
        path = Path(path)
        self.features.collect().write_csv(path)

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
        
        # Use scan instead of read_csv to be lazy
        return cls.scan(path, name=path.stem)
        