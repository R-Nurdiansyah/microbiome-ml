"""Every transformation the pipeline applies, in one place.

Two kinds, applied at different points for a reason:

FEATURES  (transforms/features.py)
    How the taxon table's values are expressed: relabund, clr, presence,
    log. Each works on one sample's row at a time and fits nothing, so it
    is applied once, when `prepare` builds the feature set, and cannot
    leak anything from a test sample into training. The choice is part of
    the feature set's name: tax_genus, tax_genus_clr, tax_genus_pa,
    tax_genus_log.

    Config:  features.transform: relabund | clr | presence | log

LABELS  (transforms/labels.py)
    What the model is trained to predict: none, zscore, log1p,
    yeo_johnson, quantile_normal, ... Several of these are *fitted* (a
    mean, a power, quantiles), so they cannot be applied once up front:
    each model is wrapped so the transform is fitted on its own training
    rows (every CV fold, the full-pool fit, the holdout's training split)
    and its predictions are converted back. Every score stays in label
    units (°C, pH, µmol/kg), whatever the transform.
    Also here: data.label_limits, which drops physically impossible
    label values before the split.

    Config:  cv.target_transform, data.label_limits

What is NOT here, because it selects rather than transforms:
    features.min_prevalence / prevalence_by   wrangle/profiles.py
    features.icc                              wrangle/icc.py
    features.shadow                           wrangle/shadow.py
and wrangle/recipe.py, which applies prevalence, ICC, a variance
threshold and then a feature transform to one prepared base table in
each CV job (features.base_set), so a sweep needs only one prepare.
and the storage of the matrix models train on (sparse or dense), which
changes no value: train/matrix.py.

Import from the submodule you need; this package imports nothing itself,
so loading the feature transforms never pulls in scikit-learn.
"""
