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
        metadata="path/to/metadata.csv",
        attributes="path/to/attributes.csv",
        study_titles="path/to/study_titles.csv"
        )
    .add_profiles(
        profiles="path/to/profiles.csv",
        root="path/to/root.csv",)
    .add_species_features("gene_features", data="path/to/gene_features.csv")  # Species-level features
    .add_species_features("pathway_features", data="path/to/pathway_features.csv")
    .add_labels({
        "temperature": "path/to/temperature_labels.csv",
        "ph": "path/to/ph_labels.csv",
        "oxygen": "path/to/oxygen_labels.csv",
        })
    .add_groupings(
        custom_groupings="path/to/custom_groupings.csv"
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
    .create_default_groupings()  # Extract bioproject, biome, etc.
    )

# Create holdout train/test splits (supports multiple labels)
dataset.create_holdout_split(
    label="temperature",  # Or None to split all labels
    grouping="bioproject",  # Prevent group leakage
    test_size=0.2
)

# Create k-fold cross-validation folds (multiple schemes per label)
dataset.create_all_cv_schemes(
    n_folds=5
)

# Iterate over all CV folds
for label, scheme, cv_df in dataset.iter_cv_folds():
    print(f"Label: {label}, Scheme: {scheme}, Samples: {cv_df.height}")

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

# Save and load (human-readable directory structure) and Result visualization
```python
dataset.save("path/to/save/dataset", compress=True)  # .tar.gz
dataset.save("path/to/save/dataset") # saved in directory
dataset = Dataset.load("path/to/save/dataset.tar.gz")
dataset = Dataset.load("path/to/save/dataset") #load from directory

# Machine learning - iterates over all feature sets, models, and labels
cv = CrossValidator(
    dataset,
    models=[RandomForestRegressor(), GradientBoostingRegressor()]
    )

results = cv.run()

# Visualization
visualiser = Visualiser(results)
visualiser.plot_performance_metrics()
visualiser.plot_feature_importances()
```

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
