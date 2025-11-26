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

# input
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
        ...
        })
    .add_labels({
        "temperature": "path/to/temperature_labels.csv",
        "ph": "path/to/ph_labels.csv",
        "oxygen": "path/to/oxygen_labels.csv",
        ...
        })
    .define_groups({
        "group1": "path/to/group1.csv",
        "group2": "path/to/group2.csv",
        ...
        })
    .apply_preprocessing()
    .create_taxonomic_features()
    )

# save and load
# saves to a human readable directory structure with .csv files
dataset.save("path/to/save/dataset")
dataset = Dataset.load("path/to/save/dataset", compression=False)

# machine learning
# by default they always iterate over all feature sets, models and labels
cv = CrossValidator(
    dataset, 
    model=[RandomForestRegressor(), GradientBoostingRegressor()]
    )

results = cv.run()

# visualisation
visualiser = Visualiser(results)
visualiser.plot_performance_metrics()
visualiser.plot_feature_importances()
```

