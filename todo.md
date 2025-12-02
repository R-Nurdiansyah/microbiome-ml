# MicrobiomeML TODO

## Core Infrastructure (wrangle/)
- [x] `Dataset` class with builder pattern
- [x] `SampleMetadata` class
- [x] `TaxonomicProfiles` class
- [x] `FeatureSet` class
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
- [ ] Integration tests for full workflows
- [ ] Tests for `CrossValidator`
- [ ] Tests for `Visualiser`

## Documentation
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