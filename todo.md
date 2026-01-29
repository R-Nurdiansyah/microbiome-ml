# MicrobiomeML TODO

## Core Infrastructure (wrangle/)
- [x] `Dataset` class with builder pattern
- [x] `SampleMetadata` class
- [x] `TaxonomicProfiles` class
- [x] `FeatureSet` class hierarchy
  - **Abstract base class**: `FeatureSet` with scan/load/from_df methods
  - **Species-level features**: `SpeciesFeatureSet` for taxonomy-based features
  - **Sample-level features**: `SampleFeatureSet` for aggregated sample data
  - **Type-safe polymorphism**: Proper method signatures across inheritance hierarchy
- [x] `FeatureAggregator` class for species-to-sample conversion
  - **8 aggregation methods**: arithmetic/geometric/harmonic mean, median, presence/absence, top-k abundant, min/max abundance
  - **3 weighting strategies**: none, abundance-weighted, sqrt abundance-weighted
  - **Configurable parameters**: k for top-k, pseudocount for geometric/harmonic, min_abundance threshold
  - **Memory optimization**: Polars LazyFrame processing for large datasets
  - **Feature quality control**: Remove features present in <10 samples (configurable)
- [x] Add/load labels from multiple sources
- [x] Add/load groupings from multiple sources
- [x] Synchronization across all components
- [x] Save/load with human-readable CSV structure
- [x] QC/preprocessing pipeline
- [x] Stratified train/test splitting (continuous + categorical)
- [x] `SplitManager` class for organizing splits per label
- [x] K-fold cross-validation with group awareness
- [x] Multiple CV schemes per label (random, grouped, etc.)
- [x] `iter_cv_folds()` for iterating over all CV combinations

## Dataset Enhancements
- [x] `create_default_groupings()` method
  - Auto-generate groupings from metadata (e.g., bioproject, biome)
- [x] `create_holdout_split()` for train/test splits
  - Supports single or all labels
  - Group-aware stratified sampling
- [x] `create_cv_folds()` for k-fold cross-validation
  - Multiple schemes per label
  - Auto-iteration over all labels when label=None
- [x] Iterator methods for CV folds
  - `iter_cv_folds()` to yield (label, scheme, cv_df) tuples
- [x] Compression support for `save()` method
  - tar.gz compression with compress=True parameter
- [x] Species-to-sample feature aggregation system
  - **Multiple aggregation methods**: arithmetic/geometric/harmonic mean, median, presence/absence, top-k abundant, min/max abundance
  - **Weighting strategies**: none, abundance-weighted, sqrt abundance-weighted
  - **Memory-efficient processing**: Uses Polars LazyFrame for large datasets
  - **Automatic relative abundance conversion**: TaxonomicProfiles converts coverage to relabund
  - **Feature filtering**: Remove uncommon features with configurable cutoff
  - **Unified API**: `add_taxonomic_features()` method with single or batch aggregation
- [ ] Helper methods for accessing split data
  - `get_train_samples(label, fold=None)`
  - `get_test_samples(label, fold=None)`

## Machine Learning (train/)
- [ ] `CrossValidator` class
  - Iterate over feature sets, models, and labels
  - Handle k-fold cross-validation using dataset splits
  - Support for scikit-learn and custom models
  - Metric calculation (RMSE, R², accuracy, etc.)
- [ ] `ModelTrainer` base class
  - Standardized interface for training
  - Support for regression and classification
- [ ] `Results` data structure
  - Store predictions, metrics, feature importances
  - Serialization/deserialization

## Visualization (visualise/)
- [ ] `Visualiser` class
  - `plot_performance_metrics()` - compare models/features
  - `plot_feature_importances()` - across models
  - `plot_predictions()` - actual vs predicted
  - `plot_splits_distribution()` - verify stratification
- [ ] Export plots to publication-ready formats

## Testing
- [x] Unit tests for `Dataset` splitting logic
- [x] Tests for eager/lazy modes
- [x] Test file organization and cleanup
- [x] Type-safe testing configuration (tests excluded from mypy)
- [ ] Integration tests for full workflows
- [ ] Tests for `CrossValidator`
- [ ] Tests for `Visualiser`

## Code Quality & Type Safety
- [x] Full mypy type checking with strict configuration
- [x] Method signature compatibility across base/subclasses
- [x] Abstract base class design (FeatureSet hierarchy)
- [x] Polars DataFrame/LazyFrame type annotations
- [x] Pre-commit hooks configuration aligned with type checking
- [x] Mypy configuration optimized for src/ only
- [x] Development workflow optimization

## Documentation
- [x] Updated README.md with current features and development workflow
- [x] Updated TODO list to reflect completed work
- [ ] API reference documentation
- [ ] Tutorial notebooks
  - Basic dataset construction
  - Custom feature sets
  - Cross-validation workflow
  - Visualization examples
- [ ] Examples gallery with real datasets

## Performance & Optimization
- [ ] Benchmark lazy vs eager loading
- [ ] Memory profiling for large datasets
- [ ] Parallel processing for cross-validation
- [ ] Caching for expensive operations

## Future Enhancements
- [ ] Support for other model types (PyTorch, XGBoost)
- [ ] Feature selection methods
- [ ] Hyperparameter tuning integration
- [ ] Pipeline serialization (dataset + model + results)
- [ ] Web interface for interactive exploration
