"""Core data models for microbiome data structures."""

import numpy as np
import polars as pl
from typing import List, Any, Optional, Set, Union
from pathlib import Path
from enum import Enum

from microbiome_ml.utils._cache import load_data


class MetadataFields(Enum):
    """Enumeration of metadata fields with validation properties."""

    def __init__(self, column_name: str, dtype, required: bool, description: str, alternatives: List[str] = None):
        self.column_name = column_name
        self.dtype = dtype
        self.required = required
        self.description = description
        self.alternatives = alternatives or []
        self.all_names = [column_name] + self.alternatives
    
    def find_column_name(self, lf: pl.LazyFrame) -> Optional[str]:
        """Find the actual column name in the LazyFrame from possible alternatives."""
        schema_names = lf.collect_schema().names()
        for name in self.all_names:
            if name in schema_names:
                return name
        return None


class CoreMetadataFields(MetadataFields):
    """Enumeration of core metadata fields with validation properties."""
    
    # Format: (column_name, polars_dtype, is_required, description, alternative_names)
    RUN_ACC         = ("sample", pl.Utf8, True, "Unique sample identifier", ["acc", "accession", "run_acc", "sample_id", "sample"])
    BIOSAMPLE_ACC   = ("biosample", pl.Utf8, True, "NCBI BioSample accession ID", ["biosample_acc", "bio_sample"])
    BIOPROJECT_ACC  = ("bioproject", pl.Utf8, True, "NCBI BioProject accession ID", ["bioproject_acc", "bio_project"])
    LATITUDE        = ("lat", pl.Float64, True, "Sample latitude coordinate in decimal degrees", ["latitude"])
    LONGITUDE       = ("lon", pl.Float64, True, "Sample longitude coordinate in decimal degrees", ["longitude"])
    DEPTH           = ("depth", pl.Float64, False, "Sample depth in meters", ["depth_m", "sample_depth"])
    DATE            = ("collection_date", pl.Date, True, "Sample collection date", ["date", "sample_date", "collection_timestamp"])
    BIOME           = ("biome", pl.Utf8, True, "Environmental biome classification (e.g., 'soil', 'marine')", [])
    SEQUENCE_DEPTH  = ("mbases", pl.UInt32, True, "Sequencing depth in megabases", ["mbp"])

class DerivedMetadataFields(MetadataFields):
    """Enumeration of derived metadata fields to be created from core fields with validation properties."""
    DOMAIN          = ("domain", pl.Utf8, False, "Terrestrial, Marine, or Both", [])
    ECOREGION       = ("ecoregion", pl.Utf8, False, "WWF ecoregion classification", [])
    YEAR            = ("year", pl.Int32, False, "Year of sample collection", [])
    MONTH           = ("month", pl.Int32, False, "Month of sample collection", [])
    DAY             = ("day", pl.Int32, False, "Day of sample collection", [])
    SEASON_CLASS    = ("season_class", pl.Utf8, False, "Season classification based on Koppen classification", [])
    SEASON          = ("season", pl.Utf8, False, "Season based on month and season class", [])    


class AttributesFields(MetadataFields):
    """Enumeration of attributes AutoFrame fields."""
    RUN_ACC         = ("sample", pl.Utf8, True, "Unique sample identifier", ["acc", "accession", "run_acc", "sample_id", "sample"])
    KEY             = ("key", pl.Utf8, True, "Attribute key name", [])
    VALUE           = ("value", pl.Utf8, True, "Attribute value", [])


class StudyMetadataFields(MetadataFields):
    """Enumeration of study metadata fields."""
    RUN_ACC         = ("sample", pl.Utf8, True, "Unique sample identifier", ["acc", "accession", "run_acc", "sample_id", "sample"])
    STUDY_TITLE     = ("study_title", pl.Utf8, True, "Title of the research study", ["title", "study_name"])
    ABSTRACT        = ("abstract", pl.Utf8, True, "Abstract or description of the study", ["study_abstract", "description"])


class SampleMetadata:
    """
    Triple-structure sample metadata container with dual-mode support (eager/lazy).
    
    Maintains three data structures:
    - metadata: Wide-form core fields (sample, lat, lon, date, etc.)
    - attributes: Long-form study-specific data (sample, key, value)  
    - study_titles: Study-level information (sample, study_title, abstract)
    
    Attributes:
        metadata: Polars DataFrame (eager) or None (lazy)
        attributes: Polars DataFrame (eager) or None (lazy)
        study_titles: Optional DataFrame (eager) or None (lazy)
        _is_lazy: Boolean flag indicating lazy mode
        _lf_metadata: LazyFrame (lazy) or None (eager)
        _lf_attributes: LazyFrame (lazy) or None (eager)
        _lf_study_titles: LazyFrame (lazy) or None (eager)
    """
    
    def __init__(
        self, 
        metadata: Union[Path, str, pl.LazyFrame, pl.DataFrame], 
        attributes: Union[Path, str, pl.LazyFrame, pl.DataFrame], 
        study_titles: Optional[Union[Path, str, pl.LazyFrame, pl.DataFrame]] = None,
        _is_lazy: bool = False):
        """
        Initialize SampleMetadata with validation and standardization.
        
        Args:
            metadata: Path to metadata file or LazyFrame/DataFrame with core fields
            attributes: Path to attributes file or LazyFrame/DataFrame with sample, key, value columns
            study_titles: Optional path to study titles file or LazyFrame/DataFrame with study information
            _is_lazy: Internal flag indicating lazy mode
            
        Raises:
            ValueError: If required columns are missing
        """
        self._is_lazy = _is_lazy
        
        # Load and standardize all data
        metadata_lf = self._load_and_standardize(metadata, CoreMetadataFields)
        attributes_lf = self._load_and_standardize(attributes, AttributesFields)
        study_titles_lf = self._load_and_standardize(study_titles, StudyMetadataFields) if study_titles is not None else None
        
        if _is_lazy:
            # Lazy mode: store LazyFrames
            self._lf_metadata = metadata_lf
            self._lf_attributes = attributes_lf
            self._lf_study_titles = study_titles_lf
            self.metadata = None
            self.attributes = None
            self.study_titles = None
        else:
            # Eager mode: collect to DataFrames
            self.metadata = metadata_lf.collect()
            self.attributes = attributes_lf.collect()
            self.study_titles = study_titles_lf.collect() if study_titles_lf is not None else None
            self._lf_metadata = None
            self._lf_attributes = None
            self._lf_study_titles = None



    def _load_data(self, data_source: Union[Path, str, pl.LazyFrame, pl.DataFrame]) -> pl.LazyFrame:
        """
        Load data from various sources.
        
        Args:
            data_source: Path to file, existing LazyFrame, or DataFrame
            
        Returns:
            LazyFrame
        """
        if isinstance(data_source, pl.LazyFrame):
            return data_source
        elif isinstance(data_source, pl.DataFrame):
            return data_source.lazy()
        elif isinstance(data_source, (str, Path)):
            return load_data(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

    def _validate_and_standardize(self, lf: pl.LazyFrame, field_enum) -> pl.LazyFrame:
        """
        Validate required fields and standardize column names.
        
        Args:
            lf: LazyFrame to validate and standardize
            field_enum: Enum class with field definitions
            
        Returns:
            LazyFrame with standardized column names
        """
        missing_required = []
        rename_mapping = {}
        
        # Find actual columns and build rename mapping
        for field in field_enum:
            actual_column = field.find_column_name(lf)
            if actual_column:
                # Only rename if the actual column name is different from standard
                if actual_column != field.column_name:
                    rename_mapping[actual_column] = field.column_name
            elif field.required:
                missing_required.append(f"{field.column_name} (tried: {field.all_names})")
        
        if missing_required:
            raise ValueError(f"Missing required fields: {missing_required}")
        
        # Rename columns to standard names
        if rename_mapping:
            lf = lf.rename(rename_mapping)
        
        return lf

    def _load_and_standardize(self, data_source: Union[Path, str, pl.LazyFrame, pl.DataFrame], field_enum) -> pl.LazyFrame:
        """
        Load data and standardize column names.
        
        Args:
            data_source: Data source to load
            field_enum: Field enum for validation
            
        Returns:
            Standardized LazyFrame
        """
        if data_source is None:
            return None
        lf = self._load_data(data_source)
        return self._validate_and_standardize(lf, field_enum)
    
    @classmethod
    def scan(
        cls,
        metadata: Union[Path, str],
        attributes: Union[Path, str],
        study_titles: Optional[Union[Path, str]] = None
    ) -> "SampleMetadata":
        """
        Lazily load SampleMetadata from files without pulling into memory.
        
        Args:
            metadata: Path to metadata CSV file
            attributes: Path to attributes CSV file
            study_titles: Optional path to study titles CSV file
            
        Returns:
            SampleMetadata instance in lazy mode
        """
        return cls(
            metadata=metadata,
            attributes=attributes,
            study_titles=study_titles,
            _is_lazy=True
        )
    
    def materialize(self) -> "SampleMetadata":
        """
        Convert lazy SampleMetadata to eager (DataFrame) mode in-place.
        
        If already eager, returns self. If lazy, collects LazyFrames and converts to DataFrames.
        
        Returns:
            Self after materialization
        """
        if self._is_lazy:
            self.metadata = self._lf_metadata.collect()
            self.attributes = self._lf_attributes.collect()
            self.study_titles = self._lf_study_titles.collect() if self._lf_study_titles is not None else None
            self._is_lazy = False
            self._lf_metadata = None
            self._lf_attributes = None
            self._lf_study_titles = None
        return self

    def _filter_by_sample(self, samples: pl.LazyFrame) -> "SampleMetadata":
        """
        Create new SampleMetadata filtered by sample IDs.
        
        Works in both lazy and eager modes. Preserves the mode of the original instance.
        
        Args:
            samples: LazyFrame containing sample IDs to keep
            
        Returns:
            New SampleMetadata instance with filtered data (same mode as original)
        """
        if self._is_lazy:
            # Lazy mode: filter LazyFrames
            filtered_metadata = self._lf_metadata.join(samples, on='sample', how='semi')
            filtered_attributes = self._lf_attributes.join(samples, on='sample', how='semi')
            filtered_study_titles = self._lf_study_titles.join(samples, on='sample', how='semi') if self._lf_study_titles is not None else None
            
            # Create new instance in lazy mode
            new_instance = SampleMetadata.__new__(SampleMetadata)
            new_instance._is_lazy = True
            new_instance._lf_metadata = filtered_metadata
            new_instance._lf_attributes = filtered_attributes
            new_instance._lf_study_titles = filtered_study_titles
            new_instance.metadata = None
            new_instance.attributes = None
            new_instance.study_titles = None
        else:
            # Eager mode: filter DataFrames
            filtered_metadata = self.metadata.join(samples.collect(), on='sample', how='semi')
            filtered_attributes = self.attributes.join(samples.collect(), on='sample', how='semi')
            filtered_study_titles = self.study_titles.join(samples.collect(), on='sample', how='semi') if self.study_titles is not None else None
            
            # Create new instance in eager mode
            new_instance = SampleMetadata.__new__(SampleMetadata)
            new_instance._is_lazy = False
            new_instance.metadata = filtered_metadata
            new_instance.attributes = filtered_attributes
            new_instance.study_titles = filtered_study_titles
            new_instance._lf_metadata = None
            new_instance._lf_attributes = None
            new_instance._lf_study_titles = None
        
        return new_instance

    def _filter_metadata_by_depth(self, mbp_cutoff: int) -> pl.LazyFrame:
        """Internal method: filter metadata by depth and return LazyFrame."""
        metadata_lf = self._lf_metadata if self._is_lazy else self.metadata.lazy()
        return metadata_lf.filter(pl.col("mbases").cast(pl.UInt32) > mbp_cutoff)

    def _add_domain_to_metadata(self, mapping_file: str) -> pl.LazyFrame:
        """Internal method: add domain column and return LazyFrame."""
        metadata_lf = self._lf_metadata if self._is_lazy else self.metadata.lazy()
        biome_mapping = pl.scan_csv(mapping_file)
        return metadata_lf.join(biome_mapping, left_on="biome", right_on="organism", how="left")

    def filter_by_sequence_depth(self, mbp_cutoff: int = 1000) -> "SampleMetadata":
        """Filter samples based on quality metrics.
        
        Preserves the mode (lazy/eager) of the original instance.
        
        Args:
            mbp_cutoff: Minimum megabase pairs cutoff (default: 1000)
        """
        filtered_metadata = self._filter_metadata_by_depth(mbp_cutoff)
        
        if self._is_lazy:
            # Create new instance in lazy mode
            new_instance = self.__class__.__new__(self.__class__)
            new_instance._is_lazy = True
            new_instance._lf_metadata = filtered_metadata
            new_instance._lf_attributes = self._lf_attributes
            new_instance._lf_study_titles = self._lf_study_titles
            new_instance.metadata = None
            new_instance.attributes = None
            new_instance.study_titles = None
        else:
            # Create new instance in eager mode
            new_instance = self.__class__.__new__(self.__class__)
            new_instance._is_lazy = False
            new_instance.metadata = filtered_metadata.collect()
            new_instance.attributes = self.attributes
            new_instance.study_titles = self.study_titles
            new_instance._lf_metadata = None
            new_instance._lf_attributes = None
            new_instance._lf_study_titles = None
        
        return new_instance

    def filter_by_biome(self, allowed_biomes: Optional[List[str]] = None,
                        excluded_biomes: Optional[List[str]] = None,
                        require_biome_annotation: bool = True) -> "SampleMetadata":
        """Filter samples based on biome information.
        
        Args:
            allowed_biomes: List of biomes to include (None = include all)
            excluded_biomes: List of biomes to exclude (None = exclude none)
            require_biome_annotation: Whether to require biome annotation (default: True)
            
        Returns:
            Filtered SampleMetadata instance
        """
        filtered_metadata = self.metadata
        
        # Apply biome filtering logic
        if require_biome_annotation:
            filtered_metadata = filtered_metadata.filter(pl.col("biome").is_not_null())
        
        # Apply allowed biomes filter
        if allowed_biomes:
            filtered_metadata = filtered_metadata.filter(
                pl.col("biome").is_in(allowed_biomes)
            )
        
        # Apply excluded biomes filter
        if excluded_biomes:
            filtered_metadata = filtered_metadata.filter(
                ~pl.col("biome").is_in(excluded_biomes)
            )
        
        passing_samples = filtered_metadata.select("sample")
        return self._filter_by_sample(passing_samples)

    def filter_by_study(self, excluded_study_patterns: Optional[List[str]] = None) -> "SampleMetadata":
        """Filter samples based on study information.
        
        Args:
            excluded_study_patterns: List of regex patterns to exclude from study titles
            
        Returns:
            Filtered SampleMetadata instance
        """
        if self.study_titles is None:
            raise ValueError("No study titles available for filtering")
        
        filtered_study_titles = self.study_titles
        
        # Apply exclusion patterns if specified
        if excluded_study_patterns:
            for pattern in excluded_study_patterns:
                filtered_study_titles = filtered_study_titles.filter(
                    ~pl.col("study_title").str.contains(f"(?i){pattern}")
                )
        
        passing_samples = filtered_study_titles.select("sample")
        return self._filter_by_sample(passing_samples)

    def add_domain_from_biome(self, mapping_file: str) -> "SampleMetadata":
        """
        Add domain information based on biome classification.
        
        Preserves the mode (lazy/eager) of the original instance.
        
        Args:
            mapping_file: Path to CSV mapping biomes to domains
            
        Returns:
            New SampleMetadata instance with domain column added
        """
        enhanced_metadata = self._add_domain_to_metadata(mapping_file)
        
        if self._is_lazy:
            # Create new instance in lazy mode
            new_instance = SampleMetadata.__new__(SampleMetadata)
            new_instance._is_lazy = True
            new_instance._lf_metadata = enhanced_metadata
            new_instance._lf_attributes = self._lf_attributes
            new_instance._lf_study_titles = self._lf_study_titles
            new_instance.metadata = None
            new_instance.attributes = None
            new_instance.study_titles = None
        else:
            # Create new instance in eager mode
            new_instance = SampleMetadata.__new__(SampleMetadata)
            new_instance._is_lazy = False
            new_instance.metadata = enhanced_metadata.collect()
            new_instance.attributes = self.attributes
            new_instance.study_titles = self.study_titles
            new_instance._lf_metadata = None
            new_instance._lf_attributes = None
            new_instance._lf_study_titles = None
        
        return new_instance

    def filter_by_domain(self, allowed_domains: List[str]) -> "SampleMetadata":
        """
        Filter samples by environmental domain.
        
        Args:
            allowed_domains: List of domains to include ('marine', 'terrestrial', 'both')
            
        Returns:
            Filtered SampleMetadata instance
        """
        metadata_lf = self._lf_metadata if self._is_lazy else self.metadata.lazy()
        schema_names = metadata_lf.collect_schema().names()
        if "domain" not in schema_names:
            raise ValueError("Domain column not found. Use add_domain_from_biome() first.")
        
        passing_samples = metadata_lf.filter(
            pl.col("domain").is_in(allowed_domains)
        ).select("sample")
        
        return self._filter_by_sample(passing_samples)
    
    def save(self, path: Union[Path, str]) -> None:
        """
        Save SampleMetadata to CSV files in a directory.
        
        For lazy instances, this will materialize the data first.
        
        Args:
            path: Directory path to save files (will be created if doesn't exist)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self._is_lazy:
            # Lazy mode: collect before saving
            self._lf_metadata.collect().write_csv(path / "metadata.csv")
            self._lf_attributes.collect().write_csv(path / "attributes.csv")
            if self._lf_study_titles is not None:
                self._lf_study_titles.collect().write_csv(path / "study_titles.csv")
        else:
            # Eager mode: save DataFrames directly
            self.metadata.write_csv(path / "metadata.csv")
            self.attributes.write_csv(path / "attributes.csv")
            if self.study_titles is not None:
                self.study_titles.write_csv(path / "study_titles.csv")
    
    @classmethod
    def load(cls, path: Union[Path, str], lazy: bool = True) -> "SampleMetadata":
        """
        Load SampleMetadata from directory containing CSV files.
        
        Args:
            path: Directory path containing metadata.csv, attributes.csv, and optionally study_titles.csv
            lazy: If True (default), scan files lazily; if False, read into memory
            
        Returns:
            SampleMetadata instance in lazy or eager mode
        """
        path = Path(path)
        
        metadata_path = path / "metadata.csv"
        attributes_path = path / "attributes.csv"
        study_titles_path = path / "study_titles.csv"
        
        if not metadata_path.exists() or not attributes_path.exists():
            raise FileNotFoundError(f"Required files not found in {path}")
        
        study_titles = study_titles_path if study_titles_path.exists() else None
        
        if lazy:
            return cls.scan(
                metadata=str(metadata_path),
                attributes=str(attributes_path),
                study_titles=str(study_titles) if study_titles else None
            )
        else:
            return cls(
                metadata=str(metadata_path),
                attributes=str(attributes_path),
                study_titles=str(study_titles) if study_titles else None,
                _is_lazy=False
            )