# microbiomeML

A Python module for standardizing machine learning applications within our research group. This package provides a unified API for feature manipulation, sample annotation with third-party data, and label parsing/cleaning.

## Features

- Opinionated dataset structure for consistent and robust microbiome machine learning tasks
- Easy to use user interface for building datasets from various data sources
- Integration with popular machine learning libraries

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
    .add_feature_set({
        "kmer_features": "path/to/kmer_features.csv",
        "protein_features": "path/to/protein_features.csv",
        })
    .add_labels({
        "temperature": "path/to/temperature_labels.csv",
        "ph": "path/to/ph_labels.csv",
        "oxygen": "path/to/oxygen_labels.csv",
        })
    .add_groupings(
        custom_groupings="path/to/custom_groupings.csv"
    )
    .apply_preprocessing()
    .add_taxonomic_features()  # Creates features from profiles
    .create_default_groupings()  # Extracts bioproject, biome, etc.
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

# Save and load (human-readable directory structure)
dataset.save("path/to/save/dataset", compress=True)  # .tar.gz
dataset = Dataset.load("path/to/save/dataset.tar.gz")

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

