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
  - by metadata grouping: whole bioprojects / regions / years held out together
  - by **community membership**: the taxon at a chosen rank whose prevalence is nearest the target test size defines the split, so training never sees a community carrying it
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
| **holdout** | A reserved test set that is withheld from all CV training and used only for the final evaluation. Created via `create_holdout_split` (stratified random, or group-aware on a metadata column) or `create_taxon_holdout_split` (defined by one taxon's presence/absence). |
| **taxon holdout** | A holdout defined by community membership instead of metadata: the taxon at a chosen rank whose prevalence across the labelled samples is nearest `test_size` separates carriers from non-carriers. Recorded in `holdout.csv` as `holdout_taxon`, `holdout_rank` and `holdout_taxon_in_test`. |
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

# ...or hold out by community membership instead of metadata: the family
# whose prevalence is nearest test_size splits carriers from non-carriers,
# so training never sees a community containing that taxon.
dataset.taxon_prevalence("family").head()   # inspect the candidates first
dataset.create_taxon_holdout_split(
    rank="family",          # or "f__" / TaxonomicRanks.FAMILY
    label="temperature",
    test_size=0.2,
    min_test_samples=2,     # refuse a taxon leaving fewer on either side
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
- `cv_results/` (`results.ndjson`, `results_summary.csv`, `feature_importances.csv`, `manifest.json`; set `outputs.cv_save_models: true` / `outputs.cv_fold_table: true` to also keep per-combination model pickles and `results_folds.csv`)
- `best_models/<label>/<scheme>/` (best model package for every label × CV scheme, plus `best_models_summary.csv` sorted best → worst by CV R²) — every scheme is kept, not just the overall winner, because the ungrouped `random` scheme usually wins CV by leaking group structure and must be compared against grouped schemes on the holdout
- `splits/<label>/holdout.csv` (every sample tagged `train`/`test`; the `test` rows are the holdout set) and `splits/<label>/cv_<scheme>.csv` (fold membership per CV scheme, drawn from holdout-`train` only) — override the directory with `outputs.splits_dir`
- `holdout/<label>/<scheme>/` (final holdout model per label × scheme) + `holdout_metrics.json` and `holdout_summary.csv` (sorted best → worst by holdout R²)

### Overriding config values from the command line

Any key can be overridden with `--set dotted.key=value` (repeatable; values are
parsed as YAML, so `null`, `true`, `0.2` and `[a,b]` all work):

```bash
pixi run python scripts/run_cv_to_final_eval.py --config pipeline.yaml \
  --set split.holdout.grouping=bioproject \
  --set outputs.base_dir=out/holdout_bioproject \
  --set cv.scheme=[random,bioproject]
```

This is how to compare **holdout** groupings without editing the YAML — one run
per grouping, each into its own `base_dir`:

```bash
for g in null bioproject ecoregion month; do
  pixi run -e shap python scripts/run_cv_to_final_eval.py --config pipeline.yaml \
    --set split.holdout.grouping=$g --set outputs.base_dir=out/holdout_$g
done
```

On SLURM the same loop becomes an array job so the groupings run in parallel:

```bash
#SBATCH --array=0-3
GROUPINGS=(null bioproject ecoregion month)
g=${GROUPINGS[$SLURM_ARRAY_TASK_ID]}
pixi run -e shap python scripts/run_cv_to_final_eval.py --config pipeline.yaml \
  --set split.holdout.grouping=$g --set outputs.base_dir=out/holdout_$g
```

Every `holdout/holdout_summary.csv` carries a `holdout_grouping` column, so the
runs collate into one table:

```bash
pixi run python - <<'PY'
import glob, polars as pl
files = glob.glob("out/holdout_*/holdout/holdout_summary.csv")
pl.concat([pl.read_csv(f) for f in files]).sort("r2", descending=True) \
  .write_csv("out/holdout_sweep_summary.tsv", separator="\t")
PY
```

### Sweeping holdout groupings in one job

`scripts/run_holdout_sweep.py` runs the pipeline once per holdout grouping
**in a single process** and collates the results. The dataset (inputs,
groupings, QC, taxonomic features) is built once; only the split → CV →
(holdout) part is repeated per grouping.

```bash
pixi run python scripts/run_holdout_sweep.py --config pipeline.yaml \
  --groupings random bioproject ecoregion month
```

A sweep token is one of:

| token | holdout strategy |
|---|---|
| `random` (or `null` / `none`) | stratified random — the positive control |
| `<column>` | group-aware on that metadata column (added from metadata if needed) |
| `taxon:<rank>` | community membership at that rank (see *Holding out by community membership*); also spelled `taxon_<rank>` |

so a run comparing all three kinds is:

```bash
pixi run python scripts/run_holdout_sweep.py --config pipeline.yaml \
  --groupings random bioproject ecoregion taxon:family taxon:genus
```

and with Snakemake:

```bash
pixi run snakemake -s workflow/Snakefile --configfile pipeline.yaml \
  --config holdout_groupings=random,bioproject,taxon:family --cores 32
```

Each lands in `holdout_<token>/` (`holdout_taxon_family/`), is stamped in the
`holdout_grouping` column of every summary, and appears as its own row in
`cv_sweep_headline.tsv`. A `taxon:<rank>` token is rejected before any CV runs
if `features.ranks` did not build that rank's feature set.

`random` (or `null`/`none`) is the stratified-random baseline. A grouping that
is not yet in the groupings table is pulled from the metadata automatically
(same as listing it under `groupings.columns`); only a column that exists
nowhere is an error, raised before any CV starts. Every
`--set KEY=VALUE` override is applied to all runs. Under `outputs.base_dir`
(or `--output-dir`):

| File | Contents |
|---|---|
| `holdout_<grouping>/` | a full pipeline output per grouping (`splits/`, `cv_results/`, `best_models/`, `holdout/` …) |
| `cv_sweep_headline.tsv` | control vs best grouped scheme per holdout grouping × label, with the fold profile of that grouped scheme |
| `cv_sweep_fold_profile.tsv` | whether each scheme’s CV folds were usable at all |
| `cv_sweep_all.tsv` | every CV combination for every holdout grouping, same sort |
| `holdout_sweep_summary.tsv` | holdout metrics per label × scheme × grouping, sorted by `r2` — only when `holdout_evaluation.enabled: true` |
| `sweep_status.tsv` | `ok` / `failed` per grouping; a failed grouping does not stop the others |

With `holdout_evaluation.enabled: false` the sweep is CV-only: use
`cv_sweep_headline.tsv` to see, for each way of holding data out, which scheme /
model / feature set wins and by how much.

To send it to the queue once, use the template `scripts/holdout_sweep.sbatch`
(edit the `#SBATCH` lines for your cluster):

```bash
sbatch scripts/holdout_sweep.sbatch pipeline.yaml random bioproject ecoregion
```

It runs from the repo root, picks the `shap` pixi environment (override with
`PIXI_ENV=default`), and passes the allocated CPU count to `cv.n_jobs`.

### Parallel sweep with Snakemake (large data)

`run_holdout_sweep.py` runs the groupings **in series** in one process. For
large data that can exceed a queue's wall-clock limit, so the same sweep is
also available as a Snakemake workflow that runs **one job per holdout
grouping × label × CV scheme** in parallel:

```bash
pixi run -e shap snakemake -s workflow/Snakefile \
  --configfile pipeline.yaml \
  --config holdout_groupings=random,bioproject,ecoregion \
  --cores 32
```

(`pixi run sweep …` is a shorthand for the first line.) The YAML is the same
one the single-run script takes; the groupings come from `--config
holdout_groupings=…` or a `sweep: {holdout_groupings: [...]}` section in it.

DAG, with outputs under `outputs.base_dir`:

| Rule | Jobs | Output |
|---|---|---|
| `prepare` | 1 | `prepared/` — the dataset built once (inputs, groupings, QC, features) |
| `split` | 1 per grouping | `holdout_<g>/splits/` — `holdout.csv`, `cv_<scheme>.csv`, `schemes.json` |
| `cv` | 1 per grouping × label × scheme | `holdout_<g>/runs/<label>/<scheme>/` — `cv_results/`, `best_models/`, `holdout/` (if enabled), plots |
| `collate` | 1 | `cv_sweep_headline.tsv`, `cv_sweep_fold_profile.tsv`, `cv_sweep_all.tsv`, `holdout_sweep_summary.tsv`, `sweep_status.tsv` |

Each `cv` job gets `cv.n_jobs` threads (default 8) and passes exactly that to
the CV, so jobs never oversubscribe the allocation. Logs are in
`<base_dir>/logs/`. Snakemake resumes an interrupted sweep from whatever
finished (`--rerun-incomplete`), and `-n` previews the plan.

Every rule declares `threads`, `mem_mb` and `runtime` (minutes) in Snakemake's
standard resource names, so an executor profile can map them straight to the
queue. Defaults: `prepare` 4 threads / 32 GB / 3 h, `split` 2 / 16 GB / 1 h,
`cv` `cv.n_jobs` (8) / 64 GB / 12 h, `collate` 1 / 4 GB / 15 min. Size them for
a dataset in the YAML:

```yaml
sweep:
  holdout_groupings: [random, bioproject, ecoregion]
  resources:
    prepare: {mem_mb: 96000, runtime: 360}
    cv:      {threads: 16, mem_mb: 128000, runtime: 1440}
```

Anything not listed keeps its default. A profile's `set-resources:` still
overrides these, as usual in Snakemake.

**`cv_sweep_headline.tsv`** answers "what should I read first": one row per
holdout grouping × label, pairing the `random` scheme (the **positive
control** — it leaks group structure into the validation folds, so it is
expected to score well) against the best *grouped* scheme, with the feature
set and model each of them selected:

| column | meaning |
|---|---|
| `control_scheme/feature_set/model/r2/mse` | the `random` winner for that grouping × label |
| `best_grouped_scheme/feature_set/model/r2/mse` | the best non-random CV scheme |
| `r2_gap` | `control_r2 − best_grouped_r2`; large = the signal is mostly within-group |

It is written by both sweep paths. To rebuild it (and the fold profile) from a
sweep directory without recomputing anything:

```bash
pixi run python -c "
import sys; sys.path.insert(0, 'scripts')
import pipeline_steps as s
d = 'out/pipeline'
print(s.write_summaries(s.load_sweep_best(d), d))
"
```

`load_sweep_best` gathers the per-run `best_models_summary.csv` files directly,
so this also works while jobs are still finishing.

**`cv_sweep_fold_profile.tsv`** says whether each CV scheme's split was usable
at all, so a grouped R² can be judged rather than taken at face value. A group
cannot be split across folds, so one oversized group forces a lopsided split —
and the iteration that validates on the big fold trains on whatever is left:

| column | meaning |
|---|---|
| `n_folds`, `fold_sizes` | how many folds were populated, and their sizes |
| `min_train_n` | samples available to the *worst* CV iteration (`n_total − largest fold`) |
| `max_fold_pct` | share of samples in the largest fold |
| `verdict` | `ok`, or why the R² should not be trusted |

A scheme is flagged when a single fold holds more than 50 % of the samples, or
the worst iteration trains on fewer than 30 samples (`MAX_FOLD_PCT` /
`MIN_TRAIN_N` in `scripts/pipeline_steps.py`). The same columns are joined onto
`cv_sweep_headline.tsv` as `best_grouped_n_folds`, `best_grouped_fold_sizes`,
`best_grouped_min_train_n`, `best_grouped_max_fold_pct` and
`best_grouped_verdict`, and printed in the collate log.

For example, a `bioproject` scheme whose folds are `[568, 11]` reports
`min_train_n=11`, `max_fold_pct=98.1` and a verdict naming both problems: one
of its two CV iterations fitted a model on 11 samples, so its R² measures that,
not cross-group generalisation. The `random` control is unaffected and its rows
should show balanced folds.

**Retries on out-of-memory.** An OOM kill leaves no traceback — the step's log
just stops mid-way — so jobs are retried with more memory: attempt 2 gets
`mem_mb × mem_scale`, attempt 3 `× mem_scale²`, and each attempt also gets
proportionally more `runtime`. Defaults: `mem_scale: 2.0`, 2 retries for
`prepare`/`split`/`cv`. Logs are appended, not overwritten, and every attempt
starts with a header line naming the host, rule, `mem_mb` and threads it got,
so `<base_dir>/logs/` shows the whole history. Tune in the YAML:

```yaml
sweep:
  mem_scale: 2.0
  retries: {prepare: 3}
  resources:
    prepare: {mem_mb: 256000, runtime: 600}
```

If the first attempt is reliably too small for your data, raise the starting
`mem_mb` rather than relying on retries — a doubling retry wastes the time the
killed attempt already spent.

To fan the jobs out to a cluster queue instead of one node, use the generic
cluster executor with your submit command, e.g.

```bash
pixi run -e shap snakemake -s workflow/Snakefile --configfile pipeline.yaml \
  --config holdout_groupings=random,bioproject \
  --executor cluster-generic \
  --cluster-generic-submit-cmd "mqsub -t {threads} -m 64 --hours 12 --" \
  --jobs 20
```

(requires the `snakemake-executor-plugin-cluster-generic` package in the
environment; adapt the submit command to your scheduler.)

The steps themselves live in `scripts/pipeline_steps.py` (`prepare`, `split`,
`cv`, `collate`) and can be run by hand for debugging.

### Stopping after CV

Set `holdout_evaluation.enabled: false` to run everything up to and including
model selection and stop there: `cv_results/`, `best_models/<label>/<scheme>/`
and `best_models_summary.csv` are written, the log prints the best combination
per label/scheme, and nothing is fitted or scored on the holdout test set. Use
this to inspect the CV picture first (e.g. does a grouped scheme come close to
`random`?) before spending the one-shot holdout. The holdout split is still
created and saved under `splits/`, so re-running with `enabled: true` and the
same `split.*` settings evaluates on the identical test samples.

### Holding out by community membership

Instead of a metadata column, the holdout can be defined by the microbiome
itself: pick a taxonomic rank, and the taxon at that rank whose prevalence
across the labelled samples is closest to `test_size` becomes the split.

```yaml
split:
  holdout:
    taxon_rank: family        # or f__ ; overrides `grouping`
    test_size: 0.2
    presence_threshold: 0.0   # present = value above this
    min_test_samples: 2       # refuse a split leaving fewer on either side
```

How it works, in order:

1. samples with a null label are dropped (per label), and the profiles are
   restricted to the labelled samples that have a profile;
2. prevalence is computed for every taxon at the rank, from the `tax_<rank>`
   feature set (so `features.ranks` must include that rank);
3. the taxon nearest the target is chosen — either *presence* → test (a taxon
   in ~20 % of samples) or *absence* → test (a taxon in ~80 %), whichever
   lands closer; ties prefer presence;
4. carriers go to one side, non-carriers to the other, so **the training set
   never contains a community carrying that taxon**.

Worked example — 4 samples, families `a,b,c` / `c,d,e` / `x,b,c` / `b,c,d`
with `test_size: 0.2`: `f__a`, `f__e` and `f__x` each occur in 1 of 4 samples
(25 %, the nearest available to 20 %), so one of them is chosen and its single
carrier becomes the holdout; the other three samples go to CV.

The split is saved as usual to `splits/<label>/holdout.csv`, with three extra
columns recording the decision: `holdout_taxon`, `holdout_rank` and
`holdout_taxon_in_test` (`presence` or `absence`). Each label is handled
independently, so different labels may end up on different taxa. If no taxon
can leave `min_test_samples` on both sides the label is skipped with a
warning rather than given a degenerate split.

Related helper: `Dataset.taxon_prevalence(rank)` returns
`{taxon, n_present, n_samples, prevalence}` so you can see the candidates
before committing to a split.

### Choosing grouping columns

Groupings are the columns that `split.holdout.grouping`, `split.cv.grouping`
and `cv.scheme` can name. The `groupings:` section builds them in three layers,
none of which overwrites the others:

```yaml
groupings:
  defaults: true                      # bioproject, biome, domain, ecoregion,
                                      # year, month, climate, season
  columns: [depth_category, zone]     # any other metadata column; error if absent
  file: /path/to/my_groupings.csv     # `sample` + derived columns, e.g. lat bands
```

- `split.holdout.grouping` takes **one** column (or `null`): every value of that
  column is placed wholly in `train` or wholly in `test`. To compare holdouts by
  different groupings, run the pipeline once per grouping — or use the sweep,
  which does that for you. Setting `split.holdout.taxon_rank` instead replaces
  the metadata grouping with a community-membership split (see *Holding out by
  community membership*); it needs no grouping column.
- `split.cv.grouping: all` builds a fold scheme for every grouping column;
  `cv.scheme: [random, month, zone]` then limits which schemes are actually
  trained. `random` is the ungrouped baseline.
- Any name that is not a grouping column fails immediately after the groupings
  are built, before feature engineering, with the list of available columns.

Older configs using `features.create_default_groupings` and `data.groupings`
still work; they map onto `groupings.defaults` and `groupings.file`.

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


## SHAP Analysis (optional)

SHAP provides sample-level feature attribution beyond mean importance scores.
It is an **optional** dependency — the rest of the pipeline works without it.

### Installation

```bash
# Activate the shap pixi environment (includes shap + all dev deps)
pixi run -e shap python scripts/run_cv_to_final_eval.py --config pipeline.yaml

# Or install shap manually into your active environment
pip install shap
```

### Basic usage (after holdout evaluation)

```python
from microbiome_ml.train.shap_analysis import SHAPAnalyser
from microbiome_ml import Visualiser

# Assumes `evaluation` is a HoldoutEvaluation and X_test is the feature matrix
# used during trainer.train_and_evaluate()

analyser = SHAPAnalyser(
    model=evaluation.estimator,
    X=X_test,                           # numpy array, shape (n_samples, n_features)
    feature_names=evaluation.feature_names,
    max_background=100,                 # subsample cap — keeps compute lightweight
)
shap_result = analyser.compute()

# Export summaries
shap_result.save_summary("out/shap_summary.csv")  # mean |SHAP| per feature
shap_result.save("out/shap_values.csv")            # raw SHAP matrix

# Plot
vis = Visualiser(out="out/figures")
vis.plot_shap_summary(shap_result, style="bar",      output="shap_bar")
vis.plot_shap_summary(shap_result, style="beeswarm", output="shap_beeswarm",
                      X=X_test)                      # X needed for colour coding
```

### SHAP in the pipeline script

`scripts/run_cv_to_final_eval.py` runs SHAP on every holdout model when the
`shap` section of the config is enabled:

```yaml
shap:
  enabled: true
  max_background: 100
  top_n: 20
```

Run it in an environment that has `shap` installed:

```bash
pixi run -e shap python scripts/run_cv_to_final_eval.py --config pipeline.yaml
```

Per `holdout/<label>/<scheme>/` you get `shap_summary.csv` and `shap_values.csv`;
`holdout_metrics.json` gains `shap_top_features` and
`shap_top_features_direction` per model, and with `visualise.enabled: true`
two figures per model: `holdout_shap_bar_<label>__<scheme>` and
`holdout_shap_beeswarm_<label>__<scheme>`. If `shap` is not installed the
pipeline logs a warning and continues without it.

**Reading direction.** `mean_abs_shap` (the bar length) is magnitude only —
the absolute value is taken before averaging, so it cannot tell you whether a
taxon pushes the label up or down. `shap_summary.csv` therefore also carries:

- `mean_shap` — signed mean SHAP across samples;
- `direction` — Spearman correlation between the feature value (abundance)
  and its SHAP value, in [-1, 1]. Positive: more abundant → higher predicted
  label; negative: more abundant → lower predicted label. Empty when the
  feature is constant in the test split.

The bar plot colours bars blue (positive direction) / red (negative); the
beeswarm shows every sample so you can also see how consistent the effect is.
Rank-based `direction` is used rather than a linear slope because microbiome
abundances are zero-inflated and heavily skewed.

### Automatic SHAP during holdout training

Pass `compute_shap=True` to `train_and_evaluate` and the trainer handles
everything: SHAP is computed on the test split, `shap_summary.csv` and
`shap_values.csv` are written alongside the model package, and the result is
stored on `evaluation.shap_result`.

```python
trainer = ModelTrainer(
    dataset=dataset,
    best_result=cv.best_result_by_label or cv.best_result,
    output_model_path="out/holdout",
)
evaluation = trainer.train_and_evaluate(
    compute_shap=True,
    max_shap_background=100,   # background subsample size
)

# SHAP result is available directly
if evaluation.shap_result is not None:
    vis.plot_shap_summary(evaluation.shap_result, output="shap_bar")
    print(evaluation.shap_result.top_features(n=10))
```

For multi-label pipelines the trainer loops over labels; set
`compute_shap=True` once and SHAP is computed per label automatically.

### Parallel SHAP for multiple models

```python
from microbiome_ml.train.shap_analysis import SHAPAnalyser

# evaluations: dict[str, HoldoutEvaluation], X_by_label: dict[str, np.ndarray]
jobs = [
    (ev.estimator, X_by_label[label], ev.feature_names)
    for label, ev in evaluations.items()
]
shap_results = SHAPAnalyser.compute_parallel(jobs, max_background=100, n_jobs=-1)
```

### Outputs

| File | Description |
|------|-------------|
| `shap_summary.csv` | Ranked table: `rank, feature, mean_abs_shap` |
| `shap_values.csv`  | Full SHAP matrix: rows = samples, cols = features |

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
