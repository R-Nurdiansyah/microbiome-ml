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

## Terms

| Term | Description |
|------|-------------|
| **feature** | A single input variable (column) used for model training, e.g. the relative abundance of a genus. |
| **feature_set** | A named collection of features that can be used as model input, e.g. `tax_genus` (genus-level abundances) or `pathway_features_arithmetic_mean_none` (aggregated pathway features). Each feature set is a matrix of samples × features. |
| **label** | The continuous target variable to predict, e.g. `temperature`, `ph`, or `oxygen`. One model is trained per label. |
| **scheme** | The cross-validation partitioning strategy that defines how samples are assigned to folds, e.g. `random` (no grouping), `bioproject` (group by sequencing project), or `ecoregion` (group by biome). Grouped schemes prevent samples from the same group appearing in both train and validation splits. |
| **fold** | A single train/validation split within a CV scheme. The number of folds is set by `n_folds` in `create_cv_folds`. |
| **grouping** | A categorical sample attribute (e.g. `bioproject`, `biome`) used to define CV schemes or holdout splits, preventing data leakage across related samples. |
| **holdout** | A reserved test set that is withheld from all CV training and used only for the final evaluation. Created via `create_holdout_split`. |
| **CV_Result** | A container storing the outcome of one CV run for a specific (feature_set, label, scheme, model) combination, including per-fold metrics, the best estimator, and feature names. |
| **HoldoutEvaluation** | The result of retraining the best CV configuration on the holdout-train split and evaluating on the holdout-test split. Contains predictions, targets, metrics, and feature names. |


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
    .add_taxonomic_features()          # Defaults to class rank; use all=True for all ranks
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
    use_holdout=True,       # To use previous holdout test/train and create fold based on train data only
    strict=False,           # If True, skip schemes where data produces fewer than n_folds populated folds
)

# Iterate over all CV folds
for label, scheme, cv_df in dataset.iter_cv_folds():
    print(f"Label: {label}, Scheme: {scheme}, Samples: {cv_df.height}")
```

## Big Data Script (No Notebook)

For large datasets, run the end-to-end pipeline as a script:

```bash
# 1) Copy and edit the example config with your file paths
cp scripts/pipeline.example.yaml pipeline.yaml

# 2) Run dataset build -> CV -> best export -> holdout final evaluation
pixi run python scripts/run_cv_to_final_eval.py --config pipeline.yaml
```

Outputs are written under `outputs.base_dir` from the config:
- `cv_results/` (all CV artifacts)
- `best_models/` (best model package per label when multiple labels)
- `holdout/` (final holdout model(s) + `holdout_metrics.json`)

## Feature Engineering Examples

```python
# Default: generate taxonomic features at class rank only
dataset.add_taxonomic_features()  # Creates tax_class

# Generate features at all standard ranks (phylum → species)
dataset.add_taxonomic_features(all=True)  # Creates tax_phylum, tax_class, tax_order, …

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

# Best selections after run:
# - cv.best_result: global best across all labels/schemes/models
# - cv.best_result_by_label: best per label
print(cv.best_result)
print(cv.best_result_by_label)

# Grid Cross validation run -> using GridCV from sklearn
# Add n_jobs to specify parallelization in GridCV (None will dynamically determine based on CPU and hyperparams combos)
results_grid = cv.run_grid(param_path="hyperparameters.yaml")

# Export CV result package (will save everything but not the best model)
CV_Result.export_result(results, "out/cv_results")

# Export best result(s):
# - if a single label exists, export directly to out_best
# - if multiple labels exist, export one best model package per label directory
CV_Result.export_best_results(results, "out/best_models")

# Or explicitly export best-per-label map from CrossValidator
CV_Result.export_best_results(cv.best_result_by_label, "out/best_models_by_label")

```

## Final Holdout-Evaluation

After selecting the best CV configuration(s), retrain on the holdout-train split
and evaluate on holdout-test.

```python
from microbiome_ml.train.trainer import ModelTrainer
from microbiome_ml import Visualiser

if cv.best_result is None and not cv.best_result_by_label:
    raise RuntimeError("Run CV first so best result(s) are available")

# Use global best (single-result flow) or per-label best map (multi-label flow)
best_for_holdout = (
    cv.best_result_by_label if cv.best_result_by_label else cv.best_result
)

# trainer exports package(s) into output_model_path
trainer = ModelTrainer(
    dataset=dataset,
    best_result=best_for_holdout,
    output_model_path="out/holdout",
)
evaluation = trainer.train_and_evaluate()

if isinstance(evaluation, dict):
    for label, ev in evaluation.items():
        print(label, ev.metrics)
else:
    print(evaluation.metrics)
# metrics keys include: mae, mse, r2, q2, pcc, pval, feature_set, label, scheme, n_test

# Optional holdout diagnostics plot
vis = Visualiser(out="out/figures")
if isinstance(evaluation, dict):
    for label, ev in evaluation.items():
        scheme = ev.metrics.get("scheme")
        groups = [scheme] * len(ev.targets) if scheme else None
        vis.visualise_model_performance(
            ev.predictions,
            ev.targets,
            groups=groups,
            title=f"Holdout diagnostics ({label})",
            file_name=f"holdout_diagnostics_{label}",
        )
else:
    scheme = evaluation.metrics.get("scheme")
    groups = [scheme] * len(evaluation.targets) if scheme else None
    vis.visualise_model_performance(
        evaluation.predictions,
        evaluation.targets,
        groups=groups,
        title="Holdout diagnostics",
        file_name="holdout_diagnostics",
    )
```

Outputs:
- Trained holdout model package(s) persisted to `output_model_path`
- Single-label: `HoldoutEvaluation`
- Multi-label: dict of `label -> HoldoutEvaluation`

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
    best_result=cv.best_result_by_label if cv.best_result_by_label else cv.best_result,
    output_model_path="out/holdout",
)
evaluation = trainer.train_and_evaluate()
if isinstance(evaluation, dict):
    for label, ev in evaluation.items():
        vis.plot_feature_importances(
            ev,
            output=f"holdout_feature_importance_{label}",
        )
else:
    vis.plot_feature_importances(
        evaluation,
        output="holdout_feature_importance",
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

# Export best model package(s), one per label when multiple labels are present
CV_Result.export_best_results(results, "out/best-models")

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

# Default: saves as PNG only
vis = Visualiser(out="out/figures")

# Save in multiple formats simultaneously
vis = Visualiser(out="out/figures", formats=["png", "svg"])

# All supported formats: "png", "svg", "eps", "pdf"
vis = Visualiser(out="out/figures", formats=["png", "svg", "eps", "pdf"])

# CV bar plots: one file per feature_set / label / scheme / model, in every format
vis.plot_cv_bars(results="out/results")

# Feature importance from best CV result
if cv.best_result is not None:
    vis.plot_feature_importances(
        cv.best_result,
        output="cv_feature_importance",   # extension is ignored; formats control output
    )

# Feature importance from holdout evaluation
if isinstance(evaluation, dict):
    for label, ev in evaluation.items():
        vis.plot_feature_importances(
            ev,
            output=f"holdout_feature_importance_{label}",
        )
else:
    vis.plot_feature_importances(
        evaluation,
        output="holdout_feature_importance",
    )

# Holdout diagnostics: actual vs predicted + residual histogram
if isinstance(evaluation, dict):
    for label, ev in evaluation.items():
        scheme = ev.metrics.get("scheme")
        groups = [scheme] * len(ev.targets) if scheme else None
        vis.visualise_model_performance(
            ev.predictions,
            ev.targets,
            groups=groups,
            title=f"Holdout diagnostics ({label})",
            file_name=f"holdout_diagnostics_{label}",
        )
else:
    scheme = evaluation.metrics.get("scheme")
    groups = [scheme] * len(evaluation.targets) if scheme else None
    vis.visualise_model_performance(
        evaluation.predictions,
        evaluation.targets,
        groups=groups,
        title="Holdout diagnostics",
        file_name="holdout_diagnostics",      # extension is ignored; formats control output
    )
```

`plot_cv_bars` consumes a results NDJSON file or a directory containing
`results.ndjson` / `best_result.ndjson` and writes one file per combination.
All three plotting methods (`plot_cv_bars`, `plot_feature_importances`,
`visualise_model_performance`) respect the `formats` list set on `Visualiser`,
so each call produces one output file per format.

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
