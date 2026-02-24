# microbiomeML

A Python package for machine learning on microbiome datasets with comprehensive feature engineering, cross-validation, and model evaluation capabilities.

## Features

- **Dataset Management**: Unified handling of taxonomic profiles, metadata, and labels with builder pattern API
- **Taxonomic Features**: Generate features from taxonomic profiles at different ranks (phylum to species)
- **Species-to-Sample Aggregation**: Convert species-level features to sample-level data
  - 8 aggregation methods: arithmetic/geometric/harmonic mean, median, presence/absence, top-k abundant, min/max
  - 3 weighting strategies: none, abundance-weighted, sqrt abundance-weighted
  - Memory-efficient processing with Polars LazyFrame for large datasets
- **FeatureSet Types**:
  - `SpeciesFeatureSet` for taxonomy-indexed features (genes, pathways, etc.)
  - `SampleFeatureSet` for sample-level aggregated data
- **Split Management**: Stratified train/test splits with group awareness to prevent data leakage
- **Cross-Validation**: K-fold CV with multiple schemes per label (random, grouped, stratified)
- **Type Safety**: Full mypy type checking with strict configuration for reliable development
- **Save/Load**: Human-readable CSV structure with optional compression for reproducibility
- **Development Workflow**: Pre-commit hooks with automated linting, formatting, and type checking

## Installation

This project uses [pixi](https://pixi.sh/) for environment management.

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone <repository-url>
cd microbiomeML

# Install dependencies and activate environment
pixi install
pixi shell

# Development setup with pre-commit hooks
pixi run pre-commit install
```

## Quick Start

```python
from microbiome_ml import Dataset
from microbiome_ml import CrossValidator, Visualiser

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Build dataset with flexible builder pattern
dataset = (
    Dataset()
    .add_metadata(
        metadata="path/to/metadata.csv",                                            # Required
        attributes="path/to/attributes.csv",                                        # Required
        study_titles="path/to/study_titles.csv"                                     # Optional
        )
    .add_profiles(
        profiles="path/to/profiles.csv",                                            # Required
        root="path/to/root.csv",)                                                   # Required for relative abundance data
    .add_species_features("gene_features", data="path/to/gene_features.csv")        # Species-level features (Optional)
    .add_species_features("pathway_features", data="path/to/pathway_features.csv")  # Optional
    .add_labels({
        "temperature": "path/to/temperature_labels.csv",                            # Required
        "ph": "path/to/ph_labels.csv",                                              # Required
        "oxygen": "path/to/oxygen_labels.csv",                                      # Required
        })
    .add_groupings(
        custom_groupings="path/to/custom_groupings.csv"                             # Optional
    )
    .apply_preprocessing()
    .add_taxonomic_features()  # Generate abundance features at different ranks
    .aggregate_species_to_samples(  # Convert species features to sample features
        species_feature_name="gene_features",
        method="arithmetic_mean",
        weighting="abundance"
    )
    .aggregate_species_to_samples(
        species_feature_name="pathway_features",
        method="top_k_abundant",
        k=20
    )
    .create_default_groupings()                                                     # Extract bioproject, biome, etc.
    )

# Create holdout train/test splits (supports multiple labels)
dataset.create_holdout_split(
    label="temperature",    # Or None to split all labels
    grouping="bioproject",  # Prevent group leakage
    test_size=0.2
)

# Create k-fold cross-validation folds (multiple schemes per label)
dataset.create_cv_folds(
    n_folds=5,
    use_holdout=True        # To use previous holdout test/train and create fold based on train data only
)

# Iterate over all CV folds
for label, scheme, cv_df in dataset.iter_cv_folds():
    print(f"Label: {label}, Scheme: {scheme}, Samples: {cv_df.height}")
```

## Feature Engineering Examples

```python
# Generate taxonomic features at specific ranks
dataset.add_taxonomic_features(
    ranks=["genus", "species"],  # Only genus and species
    prefix="tax"  # Creates tax_genus, tax_species feature sets
)

# Aggregate species-level features to sample-level
# Single aggregation with specific parameters
dataset.aggregate_species_to_samples(
    species_feature_name="gene_features",
    output_name="sample_genes",
    method="geometric_mean",
    weighting="sqrt_abundance",
    min_abundance=0.001
)

# Create all possible aggregation combinations
dataset.aggregate_species_to_samples(
    species_feature_name="pathway_features",
    create_all=True  # Creates all method × weighting combinations
)

# Access the resulting feature sets
for name, feature_set in dataset.feature_sets.items():
    print(f"{name}: {feature_set.df.shape}")
    # e.g., "tax_genus", "sample_genes", "pathway_features_arithmetic_mean_none"
```

## Cross-validation

```python
# Cross validation - build cv and iterates over all feature sets, models, and labels
# The function will try to look any scheme or fold definition based on the result of create_cv_fold and
cv = CrossValidator(
    dataset,
    models=[RandomForestRegressor(), GradientBoostingRegressor()]
    )

# Specify the label or scheme(s); This case will only work with pH label and bioproject or ecoregion
cv = CrossValidator(
    dataset,
    models=[RandomForestRegressor(), GradientBoostingRegressor()],
    label="ph",
    scheme=["bioproject","ecoregion"]
    )

# Cross validation run; parameters is required and n_jobs is number of CV done in parallel. Default will detect CPU core
results = cv.run(param_path="parameters.yaml", n_jobs = 8)

# Grid Cross validation run -> using GridCV from sklearn
# Add n_jobs to specify parallelization in GridCV (None will dynamically determine based on CPU and hyperparams combos)
results_grid = cv.run_grid(param_path="hyperparameters.yaml")

# Export CV result package (will save everything but not the best model)
CV_Result.export_result(results, "out/cv_results")

# OR just save the best model
if cv_temp.best_model_estimator is not None and cv_temp.best_result is not None:
    # full export package (model, ndjson, feature_importance)
    CV_Result.export_result({"best": cv_temp.best_result}, out_best)

```

## Final Holdout-Evaluation

After selecting the best CV configuration (`cv.best_result`), retrain it on the
holdout-train split and evaluate on holdout-test:

```python
from microbiome_ml.train.trainer import ModelTrainer
from microbiome_ml import Visualiser

if cv.best_result is None:
    raise RuntimeError("Run CV first so best_result is available")

# trainer will export whole data to the output_model_path (model.pkl, result, feature importance)
trainer = ModelTrainer(
    dataset=dataset,
    best_result=cv.best_result,
    output_model_path="out/holdout",
)
evaluation = trainer.train_and_evaluate()

print(evaluation.metrics)
# keys include: mae, mse, r2, q2, pcc, pval, feature_set, label, scheme, n_test

# Optional holdout diagnostics plot
vis = Visualiser(out="out/figures")
scheme = evaluation.metrics.get("scheme")
groups = [scheme] * len(evaluation.targets) if scheme else None
vis.visualise_model_performance(
    evaluation.predictions,
    evaluation.targets,
    groups=groups,
    title="Holdout diagnostics",
    file_name="holdout_diagnostics.png",
)
```

Outputs:
- Trained holdout model persisted to `output_model_path`
- `HoldoutEvaluation` object with predictions, targets, metrics, and feature names

## Feature Importance Workflow

Use this workflow to inspect feature importance at both stages:

1. **CV stage**: use feature importance from the best CV model for stability checks.
2. **Holdout stage**: retrain on holdout-train and use holdout feature importance for final reporting.
3. **Export stage**: save all CV outputs, including `feature_importances.csv`.

```python
from microbiome_ml import Visualiser
from microbiome_ml.train.trainer import ModelTrainer
from microbiome_ml.train.results import CV_Result

# 1) Run CV
results = cv.run(param_path="parameters.yaml", n_jobs=8)

# 2) Export CV package (includes feature_importances.csv)
CV_Result.export_result(results, "out/cv_results")

# 3) Plot feature importance directly from best CV result
vis = Visualiser(out="out/figures")
if cv.best_result is not None:
    vis.plot_feature_importances(
        cv.best_result,
        output="cv_feature_importance.png",
    )

# 4) Train final holdout model and plot final feature importance
trainer = ModelTrainer(
    dataset=dataset,
    best_result=cv.best_result,
    output_model_path="out/holdout",
)
evaluation = trainer.train_and_evaluate()
vis.plot_feature_importances(
    evaluation,
    output="holdout_feature_importance.png",
)
```

Notes:
- `plot_feature_importances(...)` accepts `CV_Result`, `HoldoutEvaluation`, estimator objects, dict payloads with `{"model": ...}`, or a pickle model path.
- If feature names are not passed explicitly, the function uses result metadata, then estimator metadata (`feature_names_in_`), then fallback names like `feature_0`.


## Save, Load, and Visualize

### Save and load dataset

```python
# Save as compressed archive (.tar.gz)
dataset.save("path/to/save/dataset", compress=True)

# Save as directory structure
dataset.save("path/to/save/dataset")

# Load from archive or directory
dataset = Dataset.load("path/to/save/dataset.tar.gz")
dataset = Dataset.load("path/to/save/dataset")
```

### Save model and CV results

```python
from microbiome_ml.train.results import CV_Result

# Export CV package: manifest + ndjson/csv tables + model pickles
CV_Result.export_result(results, "out/results")
CV_Result.export_result(results_grid, "out/grid-results")

# Optionally persist the single best estimator/result
if cv.best_model_estimator is not None and cv.best_result is not None:
    CV_Result.save_model(
        cv.best_model_estimator,
        "out/best_model.pkl.gz",
        compress=True,
    )
    CV_Result.save_cv_result(cv.best_result, "out/best_result.ndjson")
```

### Visualization

```python
from microbiome_ml import Visualiser

vis = Visualiser(out="out/figures")

# CV bar plots: one PNG per feature_set / label / scheme / model
vis.plot_cv_bars(results="out/results")

# Feature importance from best CV result
if cv.best_result is not None:
    vis.plot_feature_importances(
        cv.best_result,
        output="cv_feature_importance.png",
    )

# Feature importance from holdout evaluation
vis.plot_feature_importances(
    evaluation,
    output="holdout_feature_importance.png",
)

# Holdout diagnostics: actual vs predicted + residual histogram
scheme = evaluation.metrics.get("scheme")
groups = [scheme] * len(evaluation.targets) if scheme else None
vis.visualise_model_performance(
    evaluation.predictions,
    evaluation.targets,
    groups=groups,
    title="Holdout diagnostics",
    file_name="holdout_diagnostics.png",
)
```

`plot_cv_bars` consumes a results NDJSON file or a directory containing
`results.ndjson` / `best_result.ndjson` and writes one file per combination.

## Development

This project uses strict type checking and code quality tools for reliable development:

```bash
# Run type checking
pixi run type-check

# Run all pre-commit hooks
pixi run pre-commit run --all-files

# Run tests with coverage
pixi run test
```

### Type Safety
- Full mypy type checking with strict configuration
- Only source code (`src/`) is type-checked; tests are excluded for faster development
- Pre-commit hooks ensure consistent code quality

### Project Structure
- `src/microbiome_ml/`: Main package code
- `test/`: Test files (excluded from type checking)
- Configuration: `pyproject.toml`, `pixi.toml`
