# microbiomeML

A Python package for machine learning on microbiome datasets with comprehensive feature engineering, cross-validation, and model evaluation capabilities.

## TL;DR

Predict an environmental label (pH, temperature, oxygen …) from taxonomic
profiles, and find out whether the prediction survives being tested on
*unseen groups* rather than on a random split.

```bash
pixi install                                   # environment + this package
cp scripts/pipeline.example.yaml pipeline.yaml # then edit the paths at the top
```

Run it. One job, everything in series:

```bash
pixi run python scripts/run_cv_to_final_eval.py --config pipeline.yaml
```

Or in parallel on a cluster — same YAML, one job per holdout × label × CV
scheme, one set of summary tables at the end:

```bash
pixi run sweep --configfile pipeline.yaml --cores 32
```

**Read `cv_sweep_headline.tsv` first.** One row per holdout grouping × label:
the `random` scheme (the positive control) next to the best *grouped* scheme,
and the `r2_gap` between them. A large gap means the signal is mostly
within-group, not generalisable. Check `verdict` in the same row before
believing any R² — a grouped scheme whose folds collapsed is flagged there.

The four things worth knowing before you start:

| | |
|---|---|
| **Two levels of splitting** | A **holdout** is withheld from everything and scored once at the end. **CV folds** are drawn from the holdout-train samples only. Keep `split.cv.use_holdout: true`. |
| **Grouping is the whole point** | `grouping: bioproject` sends each bioproject wholly to train or wholly to test. Comparing that against `random` is how you tell real signal from leaked group structure. |
| **Work up in stages** | `cv.enabled: false` stops after the split and its distribution plots, so you can check the splits are sane before paying for models. Faster still: `scripts/check_holdout_metadata.py` answers "which grouping gives a representative holdout?" from the metadata alone, in seconds. |
| **Config changes do not re-trigger Snakemake** | Rules re-run on missing *files*, not changed settings. After editing `features`/`preprocessing`, force the stage: `-R prepare`. |
| **One transform per prepared dataset** | `features.transform` is a single value and the feature sets are listed in `manifest.json`, so a second `prepare` into the same `base_dir` orphans the first. Use [the transform sweep](#comparing-transforms-in-one-sweep) to compare them. |

Everything below is detail. The sections to jump to are
[Big Data Script](#big-data-script-no-notebook) for the config-driven
pipeline, [Parallel sweep with Snakemake](#parallel-sweep-with-snakemake-large-data)
for the cluster workflow, and
[Comparing transforms in one sweep](#comparing-transforms-in-one-sweep) when
you want to know whether CLR or presence/absence buys you anything.

## Features

- **Dataset Management**: Unified handling of taxonomic profiles, metadata, and labels with builder pattern API
- **Taxonomic Features**: Generate features from taxonomic profiles at different ranks (phylum to species)
  - three transforms: relative abundance, **CLR**, and **presence/absence**
  - prevalence filtering, counted over samples or over **distinct studies**
  - **per-taxon ICC**: rank every taxon by how much of its variance is its bioproject, to see the protocol artefacts before they reach a tree
  - **shadow features**: shuffled copies of real columns, giving the importance ranking a measured noise floor
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
- **Cross-Validation**: multiple schemes per label (random, grouped, stratified)
  - k-fold and/or **leave-one-group-out** (sklearn `LeaveOneGroupOut`), toggled per run
  - best model kept per label **and** per scheme, so grouped schemes can be compared against the `random` control rather than discarded
  - a fold profile per scheme (`min_train_n`, `max_fold_pct`, `verdict`) saying whether the split was usable at all
- **Full-fit diagnostics**: score the model fitted on the *whole* training pool, not k-1 folds — in-sample R2 as a ceiling, out-of-bag R2, and the overfit gap against the CV score
- **Final Holdout Evaluation**: refit the CV winners on holdout-train, score once on holdout-test
- **Feature report and recursive pruning**: how concentrated the importance is, which taxa carry it, and how far you can prune before the score moves
- **SHAP** (optional): per-sample attribution with **direction**, so a taxon that pushes the label *down* is visible, not just one with a large effect
- **Config-driven pipeline**: one YAML runs build → split → CV → holdout → SHAP, with `--set dotted.key=value` overrides
- **Transform sweep**: one full run per transform into its own directory, feature-set names derived rather than hand-written, with one summary and one figure per label across the runs
- **Sweeps**: compare every way of holding data out in one submission — serially in one process, or as a Snakemake workflow with one job per holdout × label × scheme, per-rule resources, and retries that escalate RAM
- **Large data**: feature tables are streamed to disk a chunk of samples at a time and read back as scans, so peak memory is `chunk_samples × n_taxa` instead of the whole matrix
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
| **fold** | A single train/validation split within a CV scheme. For k-fold the count is `n_folds`; for leave-one-group-out it is the number of groups. A fold of `-1` means "training only, never validated". |
| **LOGO** | Leave-one-group-out: each group takes a turn as the sole validation fold (sklearn's `LeaveOneGroupOut`). Scheme named `<grouping>_logo`. |
| **grouping** | A categorical sample attribute (e.g. `bioproject`, `biome`) used to define CV schemes or holdout splits, preventing data leakage across related samples. |
| **holdout** | A reserved test set that is withheld from all CV training and used only for the final evaluation. Created via `create_holdout_split` (stratified random, or group-aware on a metadata column) or `create_taxon_holdout_split` (defined by one taxon's presence/absence). |
| **taxon holdout** | A holdout defined by community membership instead of metadata: the taxon at a chosen rank whose prevalence across the labelled samples is nearest `test_size` separates carriers from non-carriers. Recorded in `holdout.csv` as `holdout_taxon`, `holdout_rank` and `holdout_taxon_presence` (per sample: does it carry the taxon). |
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

**Pairing a label with its own holdout.** The sweep's default is the
cross-product — every grouping against every label. When you already know
which holdout suits which label, `sweep.pairs` runs one job per pair instead.
A label may name a **list** of groupings, and each becomes its own run:

```yaml
sweep:
  pairs:
    geo_ph: [organism, bioproject]   # two runs
    geo_temperature: realm           # one run
    ph: "taxon:family"            # quote any taxon token in YAML
```

gives four runs:

```
holdout_organism__geo_ph/
holdout_bioproject__geo_ph/
holdout_realm__geo_temperature/
holdout_taxon_family__ph/
```

Because each combination is its own directory, adding a grouping to a label
later only adds the run that is missing — the ones already on disk are left
alone and still appear in the summaries. Several labels may share a grouping.

```bash
pixi run snakemake -s workflow/Snakefile --configfile pipeline.yaml --cores 32
pixi run python scripts/run_holdout_sweep.py --config pipeline.yaml   # serial
```

Each pair becomes `<base_dir>/holdout_<grouping>__<label>/` and is pinned to
its label at every stage (`split.holdout.label`, `split.cv.label`, `cv.label`),
so a label that cannot be split never trips up another one's CV. Cross-validation
still covers every scheme inside each run, and all the usual summaries
(`cv_sweep_*.tsv`, `holdout_distribution_summary.txt`) come out the same, with
`holdout_grouping` reading `organism__geo_ph`.

`pairs` wins if both it and `holdout_groupings` are present. On the command
line: `--config pairs=geo_ph:organism,geo_temperature:realm` (one grouping per
label in that form; use the YAML for lists).

**Summaries accumulate across re-runs.** `collate` rebuilds the sweep tables
from every `holdout_*/` directory under `base_dir`, not just the runs of the
current invocation, so adding a grouping and re-running *extends*
`cv_sweep_all.tsv`, `cv_sweep_headline.tsv`, `sweep_status.tsv` and
`holdout_sweep_summary.tsv` rather than replacing them
(`cv_sweep_fold_profile.tsv` and `holdout_distribution_summary.txt` already
behaved this way). Re-running the *same* grouping overwrites that run's own
files in place, so you get one row per run, never duplicates. The flip side:
**deleting a run directory is how you drop it from the summaries** — a
directory left over from an abandoned configuration keeps showing up until
it is removed.

The dataset is still built **once** for the whole sweep, which is the reason to
use this rather than a bash loop over `run_cv_to_final_eval.py` — a loop repeats
the feature-building stage per pair.

A sweep token is one of:

| token | holdout strategy |
|---|---|
| `random` (or `null` / `none`) | stratified random — the positive control |
| `<column>` | group-aware on that metadata column (added from metadata if needed) |
| `taxon:<rank>` | community membership at that rank (see *Holding out by community membership*) |
| `taxon_<rank>` | the same token, underscored |

**Quote a `taxon:` token inside a YAML flow sequence** — `[random, taxon:family]`
is a parse error, because YAML reads the colon as an indicator there:

```yaml
sweep:
  holdout_groupings: [random, bioproject, "taxon:family"]   # quoted
  # or sidestep it entirely with the underscored spelling:
  # holdout_groupings: [random, bioproject, taxon_family]
```

On the command line (`--groupings`, `--config holdout_groupings=…`) the
tokens are plain argv strings, so no quoting is needed.

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

To send it to the queue once, submit the command itself — it is a single
process, so any scheduler will do:

```bash
mqsub -t 16 -m 128 --hours 48 --   pixi run -e shap python scripts/run_holdout_sweep.py     --config pipeline.yaml --groupings random bioproject ecoregion     --set cv.n_jobs=16
```

For anything large prefer the Snakemake path below, which submits one job per
holdout × label × scheme instead of running them back to back in one
allocation.

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
| `split` | 1 per run | `<run>/splits/` — `holdout.csv`, `cv_<scheme>.csv`, `schemes.json` |
| `cv` | 1 per run × label × scheme | `<run>/runs/<label>/<scheme>/` — `cv_results/`, `best_models/`, `holdout/` (if enabled), plots |
| `split_report` | 1 | `holdout_distribution_summary.txt` — train/test agreement per label × grouping, ranked best KS first. Runs even with `cv.enabled: false`, which is the point of a splits-only sweep |
| `collate` | 1 | `cv_sweep_headline.tsv`, `cv_sweep_fold_profile.tsv`, `cv_sweep_all.tsv`, `holdout_sweep_summary.tsv`, `sweep_status.tsv` — built from **every** `holdout_*/` on disk |

`<run>` is `holdout_<grouping>` for a cross-product sweep and
`holdout_<grouping>__<label>` for a `pairs` sweep.

Each `cv` job gets `cv.n_jobs` threads (default 8) and passes exactly that to
the CV, so jobs never oversubscribe the allocation. Logs are in
`<base_dir>/logs/`. Snakemake resumes an interrupted sweep from whatever
finished (`--rerun-incomplete`), and `-n` previews the plan.

Every rule declares `threads`, `mem_mb` and `runtime` (minutes) in Snakemake's
standard resource names, so an executor profile can map them straight to the
queue. Defaults: `prepare` 8 threads / 64 GB / 6 h, `split` 2 / 16 GB / 3 h,
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
| `r2_gap` | control − best grouped, on pooled R² when available, else the mean; large = the signal is mostly within-group |
| `control_pooled_r2` / `best_grouped_pooled_r2` | the variance-weighted score each one earned |
| `best_grouped_median_r2`, `_iqr_r2`, `_frac_folds_negative_r2` | the spread behind that single number |
| `best_grouped_n_folds_evaluable` / `_n_folds` | how many of its folds could be scored |

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

**How a scheme's folds are summarised.** An unweighted mean of per-fold R² is
the wrong summary once folds are small, and leave-one-group-out makes them
small by construction. Every result therefore carries five more columns:

| column | meaning |
|---|---|
| `pooled_validation_r2` | `1 − ΣSSE/ΣSST` over evaluable folds — each fold's R² weighted by the variance it had to explain. **This is what the headline ranks on.** |
| `median_validation_r2` | median per-fold R², evaluable folds only |
| `iqr_validation_r2` | interquartile range of per-fold R² |
| `frac_folds_negative_r2` | share of evaluable folds scoring below zero |
| `n_folds_evaluable` / `n_folds` | how many folds could be scored at all |

`avg_validation_r2` is still written, so nothing that read it breaks.

**Why pooled.** A held-out group whose label barely varies has SST ≈ 0, so its
R² explodes while the model may have done nothing unusual. One real run: 26
leave-one-group-out folds, one with SST = 0.008 scoring −824,781, mean R²
−32,730 — set almost entirely by that one fold. Pooled R², weighting each fold
by its SST, gives −17.7 while still charging that fold's actual error.

**Evaluable folds.** When a held-out group's label is *constant*, SST is
exactly zero, R² is undefined, and scikit-learn returns `0.0` — which reads as
"mediocre but fine". Those folds are excluded from all five columns and
counted in `n_folds_evaluable`. In that same run six of 26 folds were like
this, and dropping them moved the share of folds below zero from 77 % to
100 %: the zeros had been flattering the result.

Per-fold `SSE` and `SST` are recorded by two extra scorers in
`cross_validate`, so they are exact rather than reconstructed. They need the
`cv` jobs re-run — `collate` only stacks what those jobs wrote.

**Selection uses the same number.** `CV_Result.rank_score` is pooled R²
when the run recorded it and the mean otherwise, and every "best result"
decision reads it: the global best, best-per-label,
best-per-label/scheme, the row order in `results_summary.csv` and
`best_models_summary.csv`, and the sweep headline. So the model exported
as best for a scheme is the one the headline then ranks. A result from an
older run has no SSE/SST and falls back to the mean, so a sweep mixing old
and new directories still collates.

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
`cv`, `collate`, `split_report`) and can be run by hand for debugging.

### Trying holdout groupings from metadata alone (fast)

A holdout split needs only labels and grouping columns — profiles are for
CV and for taxon holdouts. `scripts/check_holdout_metadata.py` exploits that:
it answers "which grouping gives a representative holdout for this label?"
from one metadata table in seconds, instead of waiting for the pipeline to
build features it then will not use.

```bash
pixi run python scripts/check_holdout_metadata.py \
  --metadata ML_inputs/ITS_marine/metadata.tsv \
  --labels geo_ph geo_temperature \
  --groupings random bioproject ecoregion realm organism \
  --out out/holdout_check
```

The table needs a `sample` column (`--sample-col` to rename), the label
column(s), and the grouping columns. It calls the same functions as the
pipeline — `Dataset.create_holdout_split`, `Visualiser.plot_holdout_distribution`,
`collate_split_report` — so the splits, figures and KS values match what a
full run would give for those groupings.

Output mirrors the sweep layout:

```
<out>/holdout_<grouping>/splits/<label>/holdout.csv
<out>/holdout_<grouping>/split_visualisations/holdout_distribution_<label>.png
<out>/holdout_distribution_summary.txt     ranked best KS -> worst, per label
```

Options worth knowing: `--test-size`, `--n-bins` (label strata for the split),
`--bins` (histogram resolution), `--stat count probability`, `--random-state`.
Use the same `--test-size`, `--n-bins` and `--random-state` as the pipeline
config and the split is identical, so the grouping you pick here is the one
you will get.

**Not covered:** `taxon:<rank>` holdouts, which need the community profiles —
run those through the pipeline.

### The three stages, and stopping between them

The pipeline is three stages, each with its own switch, so a sweep can be run
as far as it is worth running:

| stage | does | switch | outputs |
|---|---|---|---|
| 1 split | holdout split + CV folds, then the label distribution plot | always | `splits/`, `split_visualisations/` |
| 2 CV | cross-validation on holdout-train (k-fold and/or leave-one-group-out), best model per scheme, CV bars | `cv.enabled` | `cv_results/`, `best_models/`, `cv_visualisations/` |
| 3 holdout | refit the winners, score on test, feature importance, SHAP | `holdout_evaluation.enabled` | `holdout/` |

```yaml
cv:
  enabled: false              # stop after stage 1
holdout_evaluation:
  enabled: false              # run stage 2, stop before stage 3
visualise:
  enabled: true
  holdout_distribution:
    enabled: true
    grouping: auto            # stack by whatever defined the holdout
    bins: 30
    stat: probability
```

With `cv.enabled: false` a sweep becomes a fast way to see every holdout
option at once — splits and distribution plots for each grouping, nothing
trained:

```bash
pixi run python scripts/run_holdout_sweep.py --config pipeline.yaml \
  --groupings random bioproject ecoregion taxon:family taxon:phylum
```

Each run writes `holdout_<token>/split_visualisations/holdout_distribution_<label>.png`:
train and test side by side on shared bins, stacked by group, annotated with
n / median / sd and a KS test (high p = the two sides agree). Under Snakemake
the same flag stops the DAG after the `split` jobs, so only those run.

Whichever stage you stop at, the holdout split is created and saved, so
re-running later with the same `split.*` settings evaluates on the identical
test samples — looking first costs you nothing. `holdout_evaluation.enabled:
false` is the one to use when you want the CV picture (does any grouped scheme
come close to `random`?) before spending the one-shot holdout.

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
columns:

| column | |
|---|---|
| `holdout_taxon` | the taxon the split was built on, e.g. `f__Aspergillaceae` |
| `holdout_rank` | the rank it came from, e.g. `family` |
| `holdout_taxon_presence` | **per sample**: `true` when that sample carries the taxon |

`holdout_taxon_presence` is uniform within each side and opposite between
them — that contrast *is* the split. Which side is which depends on the
direction chosen: with a ~20 % taxon the carriers (`true`) are the test set;
with a ~80 % taxon the non-carriers (`false`) are, and the log line says
which (`presence -> test` / `absence -> test`).

Each label is handled independently, so different labels may end up on
different taxa. If no taxon can leave `min_test_samples` on both sides the
label is skipped with a warning rather than given a degenerate split.

Related helper: `Dataset.taxon_prevalence(rank)` returns
`{taxon, n_present, n_samples, prevalence}` so you can see the candidates
before committing to a split.

### Leave-one-group-out CV

A grouped k-fold scheme packs the groups into `n_folds` folds, so each
validation fold mixes several groups. **Leave-one-group-out** instead gives
every group a turn as the sole validation set — one CV iteration per group,
training on all the others. It answers a sharper question: *does the model
transfer to this particular bioproject?*, and it shows you the spread across
groups rather than an average over arbitrary bundles of them.

Toggle it per run; `methods` takes one or both:

```yaml
split:
  cv:
    grouping: all
    n_folds: 5
    methods: [kfold, logo]   # kfold (default), logo, or both
    logo_max_groups: 50
```

Both gives two schemes per grouping — `bioproject` (k-fold) and
`bioproject_logo` — which are trained and reported independently, so
`cv_sweep_headline.tsv` can pick whichever did better as the best grouped
scheme. From Python:

```python
dataset.create_cv_folds(methods=["kfold", "logo"])
dataset.create_cv_folds(grouping="bioproject", methods="logo")
```

The splitter is sklearn's `LeaveOneGroupOut`; the fold table it produces is
stored as `{sample, fold}` like every other scheme and replayed through
`PredefinedSplit`, so nothing downstream needs to know which kind it is.

Three things to know:

| | |
|---|---|
| **`random` never gets a LOGO variant** | Without a grouping column every sample would be its own group, which is leave-one-out, not LOGO. The ungrouped baseline stays k-fold. |
| **One-sample groups are training-only** | R² is undefined on a single point, so a singleton group cannot be a validation fold. Those samples get fold `-1` — `PredefinedSplit` keeps them in training for every iteration rather than discarding them. The log says how many. |
| **Cost scales with group count** | LOGO fits one model per group, per model type, per parameter set. `logo_max_groups` (default 50) refuses the scheme above that and tells you the count, rather than quietly queueing 500 fits. Singletons do not count toward it. |

LOGO suits a grouping with few, well-populated groups — `realm`, `biome`,
`organism`. For something like `bioproject` with hundreds of levels, k-fold
is the practical choice, and `cv_sweep_fold_profile.tsv` will tell you if its
folds came out lopsided.

### Keeping memory down on large data

The expensive stage is `prepare`, and the expensive part of it is the pivot
from long-format profiles to a wide sample × taxon matrix. Polars has no lazy
pivot, so the eager path holds the long frame and the full wide table at once.
At a few hundred thousand samples that is where a job dies.

Each rank is therefore **streamed**: two cheap passes collect the taxa (the
columns) and the samples (the rows), then one chunk of samples at a time is
pivoted, padded onto the full column list and appended to
`<dataset>/features/tax_<rank>.csv`. The feature set that comes back is a scan
of that file, not the table itself, so peak memory is roughly
`chunk_samples × n_taxa`:

```yaml
features:
  add_taxonomic_features: true
  all: false
  ranks: [phylum, class, order]
  prefix: tax
  # Samples per chunk when streaming. Lower it if prepare still runs out of
  # memory; the cost is more passes over the profiles.
  chunk_samples: 5000
  # Build features only for samples with at least one non-null label. CV, the
  # trainer and SHAP never touch an unlabelled sample, so this is free unless
  # you want the saved dataset to cover everything.
  labelled_only: false
```

`scripts/pipeline_steps.py prepare` always streams (into `<out>/features/`).
From Python, pass `stream_dir` yourself:

```python
dataset.add_taxonomic_features(
    ranks=["order", "genus"],
    stream_dir="out/prepared/features",   # None = build in memory
    chunk_samples=5000,
    labelled_only=True,
)
```

**Build only the ranks you will train on.** A rank costs its full table twice
over — once while building, once on disk. When `cv.feature_set` names the sets
to use, `prepare` builds only those ranks and skips the rest:

```yaml
cv:
  feature_set: [tax_order, tax_genus]   # null = every set on the dataset
```

A deep rank is much wider than a shallow one (genus is several times order
— 2029 vs 264 columns on one ITS marine set),
and ranks are **not** a parallel dimension — one `cv` job loops over every
feature set in the dataset, in one process. Adding a rank therefore makes each
`cv` job bigger rather than adding jobs, so size `sweep.resources.cv.mem_mb`
accordingly or pin `cv.feature_set`.

### Parquet feature tables

CSV is the default and is fine at order level. At genus it is the single
biggest cost, because **text has no dtype**: every `0` is one byte on disk and
eight bytes once polars reads it back as Float64, and every job re-parses the
whole file.

```yaml
features:
  format: parquet     # csv (default) | parquet
```

Measured on one 16S soil dataset, 245,464 samples x 59,443 genera:

| | CSV | parquet |
|---|---|---|
| on disk | 18.6 GB | a fraction of it — a 99%-sparse column compresses hard |
| dtype on read | Float64 (inferred) | Float32 (stored) |
| read into memory | ~117 GB | ~58 GB, and only the columns asked for |

Written as a **directory** of row-group parts, `tax_genus.parquet/part_00000.parquet`
— one per sample chunk, since the pivot only ever holds a chunk at a time and
parquet cannot be appended to in place without pyarrow. Everything downstream
reads it through `scan_parquet("<dir>/*.parquet")`; `manifest.json` records the
name, so nothing else needs to know.

### Prevalence filter

Drop taxa that are present in too few samples. Absent means absent from the
table, not zero-filled:

```yaml
features:
  min_prevalence: 0.01    # null or absent = keep every taxon
```

Measured on that same genus table:

| keep if present in | genera | % of columns | abundance kept |
|---|---|---|---|
| any sample | 59,443 | 100 % | 100 % |
| **≥ 1 %** | **8,563** | 14.4 % | **97.6 %** |
| ≥ 2 % | 5,177 | 8.7 % | 94.9 % |
| ≥ 5 % | 2,510 | 4.2 % | 88.7 % |
| ≥ 10 % | 1,340 | 2.3 % | 77.2 % |

1 % is the usual choice: an 85 % cut in columns for 2.4 % of the abundance.
Most of what goes is genera seen in a handful of samples out of a quarter of a
million — artifacts that can never support a tree split.

**It is not leakage.** The filter reads abundances only, never a label, so
unlike supervised feature selection it cannot inflate R². It is counted once,
on the long-format profiles before the pivot — a `group_by` over sparse rows,
not a pass over the wide matrix — and **not per CV fold**, which would give
each fold a different feature space and make the folds incomparable. One
caveat for your methods section: counting over all samples, holdout included,
is technically transductive.

### Transforms: relative abundance, CLR, presence/absence

How the abundances are expressed. One value per run — see the warning below.

```yaml
features:
  transform: relabund        # relabund (default) | clr | presence
  clr_pseudocount: 1.0e-6    # only read under clr
  presence_threshold: 0.0    # only read under presence
```

| value | the cell holds | names the set |
|---|---|---|
| `relabund` | relative abundance, a constant-sum composition | `tax_genus` |
| `clr` | `ln(x + p)` minus the mean of those logs, per sample across the taxa kept at that rank | `tax_genus_clr` |
| `presence` | `1` where the value exceeds `presence_threshold`, else `0` | `tax_genus_pa` |

`clr` moves the features onto an additive scale, which trees and correlations
handle more sensibly than proportions. The pseudocount exists because the table
is mostly zeros and `log(0)` is undefined; it sets how far below the smallest
real value a zero lands.

`presence` throws the abundance away. That is the point: sequencing depth and
library protocol shift *how much* of a taxon you see far more than they shift
*whether* you see it, so binarising sheds much of the between-study artefact
that a leave-one-bioproject-out split has to survive. Raise
`presence_threshold` (say `1.0e-4`) to ignore barcode bleed; `0.0` counts any
non-zero value, matching how `min_prevalence` counts.

Accepted spellings for `presence`: `pa`, `binary`, `presence_absence`.

> **One transform per prepared dataset.** `transform` is a single value, and
> `Dataset.load` reads its feature sets from `manifest.json` — so a second
> `prepare` into the same `base_dir` rewrites the manifest and orphans the
> first transform's table on disk. Comparing transforms means one run each into
> its own `base_dir`; [the transform sweep](#comparing-transforms-in-one-sweep)
> drives that loop.

Changing the transform needs the dataset rebuilt: `-R prepare`.

### Counting prevalence by group

`min_prevalence` counts samples by default. Point `prevalence_by` at a grouping
column and it counts **distinct groups** instead:

```yaml
features:
  min_prevalence: 0.01
  prevalence_by: bioproject    # must exist under `groupings`
```

The question changes from "in how many samples does this taxon appear" to "in
how many independent studies", so one large project can no longer make a taxon
look common on its own. Names the set `tax_genus_bio`. It has no effect without
`min_prevalence`, and the pipeline warns if you set it alone.

### Feature-set names

Every build option that changes the matrix leaves a suffix, and `cv.feature_set`
has to name the result exactly or CV silently finds nothing to train on:

| built with | set name |
|---|---|
| defaults | `tax_genus` |
| `transform: clr` | `tax_genus_clr` |
| `transform: presence` | `tax_genus_pa` |
| `transform: presence` + `prevalence_by` | `tax_genus_pa_bio` |
| any of the above + `shadow` | the same, plus `tax_genus_pa_bio_shad` |

> `presence` names its sets **`_pa`**, not `_presence`. A hand-written
> `tax_genus_presence_bio` does not exist. The transform sweep derives these
> names from the package itself for exactly this reason.

Two options deliberately leave *no* mark: `min_prevalence` and the ICC filter.
Runs at `min_prevalence: 0.01` and `0.05`, or at `max_icc: null` and `0.9`,
produce sets with identical names — keep them in separate `base_dir`s.

### Study markers: per-taxon ICC

Intraclass correlation asks, for one taxon, what fraction of its variance
across samples is explained by knowing which study the sample came from.

```yaml
features:
  icc:
    enabled: false
    group: bioproject   # any column under `groupings`
    report: true        # write the ranked table
    max_icc: null       # rank only; a number drops taxa above it
```

A taxon at **ICC ≈ 1** is flat within every study and differs between them:
knowing the bioproject tells you the abundance. That is the signature of a
protocol artefact — primer bias, extraction kit, depth, a lab's contaminant —
and a tree splitting on it is learning which study it is looking at, which
earns nothing under a grouped split. A taxon at **ICC ≈ 0** moves as much
inside a study as across studies.

Writes `<base_dir>/prepared/diagnostics/<set>_icc.tsv`, ranked ICC-descending
so the study markers are at the top. Two columns matter: `icc` is the
variance-components estimate and `icc_raw` the plain sum-of-squares ratio.
Prefer `icc` — a one-sample study can only push the raw ratio up, and with
every group a singleton the raw ratio reads 1.00 for every taxon while `icc`
correctly reports nothing.

Like the prevalence filter it is unsupervised: it reads abundances and the
grouping, never a label. It is always measured on **relative abundance**,
whatever `transform` is set to — it runs on the long table, before the pivot
and before `clr`/`presence` — and after the prevalence filter, so it describes
the taxa that will actually reach the model.

> **`max_icc` defaults to off, and should usually stay off.** Every label in
> this kind of data carries most of its variance between bioprojects, so a
> taxon that genuinely predicts the label *must* also vary between studies and
> will also score a high ICC. Variance structure alone cannot separate batch
> from signal under that confound. Reading the table and excluding named
> contaminants is a safer move than a blind threshold.

### Shadow features: a measured noise floor

A shadow is a real feature column with its values shuffled across samples: same
distribution, same sparsity, same scale, every relationship to the label
destroyed. Whatever importance it earns is what the model hands out for
nothing.

```yaml
features:
  shadow:
    enabled: false
    n: 200        # ~2% of the column count is usually enough
    seed: 1234    # fixes which columns are copied and how they shuffle

cv:
  feature_set: [tax_genus, tax_genus_shad]   # both, to compare them
```

This is the comparison the importance report cannot otherwise make: "the forest
used 1,155 of 4,260 columns" and "1,155 columns the tree had to split on
something" look identical without a null.

`<set>_shad` holds **every real column plus** the shadows — the comparison only
means anything inside one fit — and the clean set is left untouched. A mapping
of which real column each shadow was copied from goes to
`<base_dir>/prepared/diagnostics/<set>_shadow_map.tsv`. The feature report
gains a `shadow_floor.tsv` and a section reading it.

> Because `_shad` is a *superset* of its clean twin, the two are expected to
> score about the same, and which one edges ahead is fold noise. Matching
> scores are a sanity check passing, not a finding. The question "did the model
> lean on the shuffled columns" is answered by importances in
> `shadow_floor.tsv`, never by the scores.

This is **not** Boruta. Boruta shadows every feature and repeats ~100 forest
fits with a binomial test per feature; this adds a sample of shadows to one
ordinary run, which costs almost nothing and calibrates the ranking, but gives
no per-feature p-value. One caveat travels with any number read off it: with
`n` shadows against `m` real columns the largest shadow importance is the max
of `n` draws from the null, not `m`, so counting real features above it
over-counts when `n < m`. Prefer the reported high quantile, and raise `n` when
the answer is close.

### Converting an existing dataset

Switching `features.format` does **not** mean rebuilding. The expensive part of
`prepare` is the pivot, and that is already in the CSVs:

```bash
pixi run python scripts/convert_features.py <base_dir>/prepared --dry-run
pixi run python scripts/convert_features.py <base_dir>/prepared
```

It writes `features/<name>.parquet/` beside each CSV and rewrites
`manifest.json` to point at it. The manifest is written **last**, so a crash
part-way leaves a working CSV dataset rather than a half-converted one. The
CSVs are kept unless you pass `--delete-csv`.

Then re-run the sweep. Because `manifest.json` is now newer than the splits,
Snakemake redoes `split` (minutes, and deterministic at the same
`random_state`, so the folds are identical) and then `cv`:

```bash
pixi run sweep --configfile pipeline.yaml --cores 16 --jobs 10
```

To keep the existing splits and redo only the CV jobs:

```bash
pixi run sweep --configfile pipeline.yaml --touch
pixi run sweep --configfile pipeline.yaml -R cv --cores 16 --jobs 10
```

`min_prevalence` is the exception: prevalence is counted during the pivot, so
turning it on needs `-R prepare`. Convert to parquet first and measure — it may
be enough on its own.

Other levers, in the order worth trying:

| | |
|---|---|
| `features.labelled_only: true` | drops unlabelled samples before the pivot |
| `features.chunk_samples` | lower = flatter memory, more passes |
| `cv.feature_set` | fewer ranks built *and* fewer trained |
| `features.format: parquet` | typed and columnar; no Float64 widening on read |
| `features.min_prevalence` | drops the rare taxa that dominate the column count |
| `cv.n_jobs` | each worker holds one feature matrix; halving it roughly halves CV peak |
| `sweep.resources` / `mem_scale` | per-rule request, and what a retry escalates to |

Feature matrices are cast to float32 before training, which halves the largest
arrays sklearn sees.

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
  trained. `random` is the ungrouped baseline. With
  `split.cv.methods: [kfold, logo]` each grouping yields two schemes,
  `<grouping>` and `<grouping>_logo` (see *Leave-one-group-out CV*), both
  nameable in `cv.scheme`.
- Any name that is not a grouping column fails immediately after the groupings
  are built, before feature engineering, with the list of available columns.

From Python the same three layers are `create_default_groupings`, which merges
into the existing groupings table rather than replacing it:

```python
dataset.create_default_groupings()                       # the defaults
# strict=True raises if the metadata has no such column;
# the default, False, skips it with a warning
dataset.create_default_groupings(groupings=["depth_category"], strict=True)
```

Use `strict=True` for a column the user named — a silent skip there turns into
a confusing "not a grouping column" error later. `force=True` replaces the
table instead of merging.

Older configs using `features.create_default_groupings` and `data.groupings`
still work; they map onto `groupings.defaults` and `groupings.file`.

## Comparing transforms in one sweep

Because a prepared dataset holds one transform, comparing them means one full
run each into its own `base_dir`. `scripts/sweep_transforms.py` drives that
loop, derives the feature-set names, and writes one summary across the runs:

```bash
pixi run python scripts/sweep_transforms.py \
    --config pipeline.yaml --out out/transform_sweep --cores 24
```

What is an axis here and what is not:

| | |
|---|---|
| `transform` | a real axis — one run per value (default: all three) |
| `--min-prevalence 0.01 0.05` | a real axis — one run per value per transform |
| `--icc-max 0.9 0.75` | one *extra* run per transform per cutoff |
| `shadow` | **not** an axis — `<set>_shad` is built beside the clean set and both train in the same run, so it is one flag |

So the grid is
`len(transforms) × max(1, len(min-prevalence)) × (1 + len(icc-max))` runs.
With neither optional flag that is three. Note that
`icc.enabled: true, max_icc: null` returns the feature frame *untouched* —
report-only ICC is a diagnostic, not a variant, so it never forks the grid.

`min_prevalence` is worth sweeping precisely because it leaves **no mark on
the feature-set name**: `tax_genus_bio` at 1 % and at 5 % are the same string,
so the only safe way to compare them is one `base_dir` each, which is what the
driver gives them. Omit the flag and each run keeps whatever the template sets.

### From one YAML, top to bottom

Every setting can live in the config's `sweep:` section, so one file and one
command run the whole benchmark. `scripts/sweep_transforms.example.yaml` lists
every key.

```bash
pixi run python scripts/sweep_transforms.py --config sweep.yaml
```

| Stage | What runs | Written to |
|---|---|---|
| 1 | `sweep.subset`: up to `n_per_label` samples per label (optional) | `<out>/subset_n<N>_seed<S>/` |
| 2-3 | the grid, one `config.yaml` per combination, checked before anything starts | `<combo>/config.yaml` |
| 4 | prepare, split, CV, feature report | `<combo>/` |
| 5 | the summary, then one figure set per CV scheme (`sweep.figure_schemes`) | `summary/`, `summary/figures/` |
| 6 | ICC and shadow-floor figures | `summary/figures/` |
| 7 | `sweep.prune`: pruning on each combination × label, on `sweep.prune.scheme` (null = each row's best grouped scheme, never `random`) | `<combo>/prune/<settings>/` |
| 8 | CV, whole-pool fit and holdout, on all features and on those pruning kept | `<combo>/prune/<settings>/<target>/{unpruned,pruned}/` |
| 9 | scoreboard and runtime figures, the ranking table | `summary/benchmark_ranking.tsv` |
| 10 | a list of what was written | printed at the end |

A flag on the command line wins over the file, so `--models rf` still works for
a cheap first pass. Stages 7 to 9 only run when `sweep.prune.enabled` is true.

**`sweep.runner`** picks how stage 4 runs:

- `snakemake` (the default) is the workflow, which can submit to a queue.
- `local` runs the same steps in the same folder layout on the machine you are
  on, with no Snakemake and no queue. Use it inside an interactive job. With
  `cores: null` it shares `PBS_NCPUS` across `max_workers`.

Stages 7 and 8 always run on the machine running the script. With Snakemake
submitting to a queue from a login node, leave `sweep.prune` off.

Each combination runs as its own process, logging to
`<combo>/sweep_transforms.log` and `<combo>/sweep_finish.log`. One that fails
or runs out of memory is recorded and the rest carry on. Rerunning skips every
step with finished output, so changing only `sweep.prune` costs just the new
pruning, in a new folder.

`--rerun` is refused with the local runner, because skipped steps would leave
old results under new settings. Use a new `sweep.out`, or delete the
combination's folder. A subset folder also refuses to be rebuilt with different
settings.

### How far each combination runs

By default, exactly as far as your template does. A template carrying
`holdout_evaluation: {enabled: false}` stops at CV for every combination, and
the summary then ranks **validation folds** — which makes the winner a
candidate, not a result.

| flag | effect on every derived config |
|---|---|
| `--holdout` | `holdout_evaluation.enabled: true` — each combination runs through to the final test |
| `--final-fit` | `final_fit.enabled: true` — whole-training-pool diagnostics (see below) |

Both are forced on for *every* combination rather than merged per combination,
on purpose: a holdout that ran for two combinations out of six is not a
benchmark. With `--holdout`, `summary/SUMMARY.txt` gains a **HOLDOUT** block
ranking the combinations on samples no fold ever saw, and `summary/holdout.tsv`
collects the rows.

```bash
pixi run python scripts/sweep_transforms.py \
    --config test_transform.yaml --out out/transform_sweep \
    --min-prevalence 0.01 0.05 0.1 --holdout --final-fit --cores 24
```

That is 3 transforms × 3 thresholds = 9 runs, each ending in its own holdout.

Each combination gets its own directory holding a derived `config.yaml`, its
prepared dataset, splits and CV outputs, so nothing collides and any one can be
rerun alone. `cv.feature_set` is **derived** rather than copied from your
template, which is the point: the template's value was written for whatever
transform it had set.

```
out/transform_sweep/
  relabund/   config.yaml  prepared/  holdout_*/  cv_sweep_headline.tsv
  clr/        …
  presence/   …
  summary/    SUMMARY.txt  headline.tsv  all_models.tsv  combo_status.tsv
```

`summary/SUMMARY.txt` is the one file to read — one block per label, the
combinations ranked, the winner starred. A combination that fails does not stop
the sweep or poison the summary: it is absent from the tables, even if an
earlier run left its results on disk, and named with its exit code. A job
stopped by SIGTERM (`qdel`, walltime) or Ctrl-C ends its running combinations,
marks them `interrupted` and still writes the summary. Rerun a combination by
hand? Delete `sweep_transforms_status.json`, then use `--summary-only`.

Relative paths in `data`, `groupings.file` and `cv.param_path` are taken from
the directory you launch the sweep from, and written into each derived config
as absolute paths. `cv.feature_set` in your template picks the ranks to build,
as it does in a normal run; only the names are re-derived.

### Growing a sweep, and keeping it fast

The summary covers **every combination under `--out`**, not only the ones on
this command line, and a combination that already finished with the same config
is not launched again. So adding a threshold costs only the new runs:

```bash
pixi run python scripts/sweep_transforms.py --config pipeline.yaml --out out/transform_sweep --min-prevalence 0.01 0.05
pixi run python scripts/sweep_transforms.py --config pipeline.yaml --out out/transform_sweep --min-prevalence 0.1
```

If settings that affect results have changed, the sweep refuses to start.
Snakemake does not treat the config as an input, so finished steps would not
rerun and old results would sit under the new settings. Use a new `--out`, or
pass `--rerun`, adding `--extra -R prepare` or `--extra -R cv` to rebuild from
that step. Missing input files and a `--min-prevalence` outside 0 to 1 are also
refused before any job starts.

CV is nearly all of the runtime: in one 18S run `prepare` and each `split` took
under a minute, against 89 minutes of RF and 175 of XGBoost summed over its nine
CV jobs. Shadow sets add a second feature set for CV to train. So screen first,
into its own `--out`:

```bash
pixi run python scripts/sweep_transforms.py --config pipeline.yaml --out out/screen --min-prevalence 0.01 0.05 0.1 --models rf --no-shadow
```

Then run the full models and `--holdout` on the candidates. Fewer CV schemes
help too: every grouping in `split.cv.grouping` adds one CV job per label.

Useful flags: `--dry-run` writes every `config.yaml` and prints the commands
without running anything, leaving the status file and summary as they were;
`--summary-only` re-reads what is on disk (no axis flags needed) and
rewrites the summary; `--stop-on-error` abandons the sweep at the first
failure and marks the rest `not run`; `--extra` passes everything after it to
Snakemake (must be last).

### Running the combinations in parallel

`--max-workers N` makes the driver a manager: it writes every `config.yaml`
first, then keeps N combinations in flight at once, records each one into
`sweep_transforms_status.json` as it lands, and collates the summary when the
last one is done.

```bash
pixi run python scripts/sweep_transforms.py \
    --config pipeline.yaml --out out/transform_sweep \
    --min-prevalence 0.01 0.05 0.1 --holdout \
    --max-workers 3 --cores 8
```

Each combination is a separate Snakemake process given its own
`--directory`, so they keep separate `.snakemake` state and never contend
for one lock — which also means an interrupted combination is unlocked on
its own rather than blocking the whole sweep.

**The budgets multiply.** `--max-workers 3 --cores 8` asks for 24 cores, not
8. Size them against the node you are on. Snakemake still schedules rules
within each combination, so there are two levels of parallelism at once.

A worker that fails, or even raises, is recorded as `failed` and the other
combinations continue; `--stop-on-error` starts nothing further, lets the ones
already running finish, and records the rest as `not run`.

### Sending a transform sweep to a cluster

Two ways. Let Snakemake submit each rule, with the driver on the login node:

```bash
nohup pixi run python scripts/sweep_transforms.py --config pipeline.yaml --out out/transform_sweep --extra --executor cluster-generic --cluster-generic-submit-cmd "mqsub -t {threads} -m 64 --hours 12 --" --jobs 20 > sweep.log 2>&1 &
```

The driver drops its own `--cores` when `--extra` sets `--jobs`, and `--extra`
reaches `subprocess` as a list, so a quoted submit command stays one argument.

Or submit each combination as its own job, so that each gets its own
allocation:

```bash
pixi run python scripts/sweep_transforms.py --config pipeline.yaml --out out/transform_sweep --dry-run
for d in out/transform_sweep/*/; do mqsub -t 24 -m 64 --hours 24 -- pixi run snakemake -s workflow/Snakefile --configfile "$d/config.yaml" --cores 24; done
pixi run python scripts/sweep_transforms.py --config pipeline.yaml --out out/transform_sweep --summary-only
```

### Plotting a transform sweep

A sweep now draws these itself, one figure set per CV scheme, into
`summary/figures/`. To redraw them by hand, which draws the best-scoring scheme
only:

```bash
pixi run python scripts/plot_transform_sweep.py \
    --summary out/transform_sweep/summary --formats png svg
```

One figure per label — they do not share a useful y axis, and putting them in
one panel hides the comparison inside each. Three panels, left to right:

- **(a) which CV scheme** — pooled R² for every scheme at full range. Drawn
  first and unzoomed because it is usually the largest effect in the table by
  an order of magnitude. A scheme that leaves the same studies on both sides of
  a fold scores well; one that does not can go deeply negative. No transform
  rescues that, so reading (b) without (a) is how a sweep gets misread.
- **(b) which transform** — the same scores zoomed to the best-scoring scheme,
  each transform's clean set beside its `_shad` set.
- **(c) how stable** — per-fold R², one dot per fold, with the median and the
  pooled score marked. A pooled 0.74 built from folds spanning 0.07 to 0.92 is
  a different claim from twelve folds at 0.74.

When the sweep ran with `--holdout`, a second figure per label is written,
`transform_sweep_holdout_<label>.png`, with two panels:

- **(a) final test, ranked** — holdout R² per combination, best first, coloured
  by transform. This is the ranking to quote.
- **(b) CV vs final test** — the same combinations with their CV score beside
  their holdout score. A combination whose dark bar sits well below its light
  one won CV optimistically; the gap is what a grouped split was supposed to
  predict and sometimes does not.

It is drawn for the best-scoring scheme only, because two schemes mean two
different test sets and one bar chart cannot honestly hold both.

## Scoring the fit on the whole training pool

Every CV number comes from a model trained on k−1 folds. The model you would
actually ship is trained on *all* of the training pool — the CV stage already
fits it, and until now only read feature importances off it.

```yaml
final_fit:
  enabled: false
  oob: true      # out-of-bag R2; random-forest only
  top_n: 20
```

Writes `<base_dir>/final_fit/full_fit_metrics.csv`, one row per label × scheme,
plus `full_fit_importances.csv`. **No model is retrained** — this scores the
fit that already happens, and the out-of-bag score is requested *before* that
fit so it rides along instead of costing a refit. The only extra work is one
`predict` per combination.

| column | what it is |
|---|---|
| `cv_pooled_r2` | the held-out score you already have |
| `in_sample_r2` | the fit scored on its own training rows — a **ceiling**, not an estimate. A random forest lands near 1.0 on almost anything |
| `oob_r2` | out-of-bag: each row scored by the trees that did not draw it. Empty for XGBoost, and for a forest with `bootstrap=False` |
| `overfit_gap` | `in_sample_r2 − cv_pooled_r2` — how much of the fit is memorisation |

> **Read `oob_r2` like a random split.** Bootstrap samples rows, not groups, so
> a row's own study is almost always represented among the trees scoring it.
> Where study identity predicts the label, OOB will look like the `random`
> scheme and far better than `bioproject_logo`. It says the fit converged, not
> that it transfers to a new study.

This is independent of `holdout_evaluation`: it describes the training fit, not
the test set, so it still runs with the holdout stage switched off.

## Feature report and recursive pruning

### The report (cheap, on by default)

One extra Snakemake rule at the end of a sweep, reading CV exports that already
exist — no model is refitted or unpickled, so it costs seconds.

```yaml
feature_report:
  enabled: true
  coverage: [0.5, 0.9, 0.95]   # shares of total importance, one column each
  top: 25                      # features printed per combination
```

Writes `<base_dir>/feature_report/`:

| file | holds |
|---|---|
| `rank_comparison.tsv` | pooled R² per rank, coarse → fine, so you can see whether genus beats order |
| `importance_concentration.tsv` | how many features carry each share of the total importance, and how many the model never split on |
| `importance_kept_taxa.tsv` | those features named, with cumulative share, for the largest `coverage` |
| `shadow_floor.tsv` | the noise floor, when the run had shadow sets |
| `feature_report.txt` | all of it, readable |

Run it alone against any finished output directory:

```bash
pixi run python scripts/rank_feature_report.py --root out/pipeline
```

Two columns people ask about: **`n_zero`** is how many features came back with
importance exactly zero — the model never split on them. **`n_for_95pct`** is
the smallest number of features carrying 95 % of the total importance. Both are
about *split work*, not contribution to R²: tree importance is impurity
reduction, so a feature can be heavily used and still not help the score.
Shadow columns are excluded from these statistics, so the numbers stay
comparable with a shadow-free run.

### Recursive pruning (costs real compute, off by default)

Each round runs CV, drops features by the chosen criterion, and runs the same
CV again on what is left. It stops early when a round drops nothing.

```yaml
feature_pruning:
  enabled: false
  iterations: 1        # rounds after the baseline; 0 = baseline only
  prune: zero          # zero | coverage
  coverage: 0.95       # only read under `prune: coverage`
  min_features: 10     # a round never leaves fewer than this
```

`zero` keeps every feature with importance above zero — it drops only what the
model never split on, and settles after a round or two. `coverage` keeps the
smallest set reaching that share of total importance; it is far more aggressive,
it *does* drop features the model used, and because it takes a fraction of
whatever is left each round it keeps shrinking until `min_features` stops it.
Set that on purpose.

It runs against a prepared dataset and one label x scheme, not the whole
config, so `--dataset`, `--splits`, `--out`, `--label` and `--scheme` are all
required; `--config` only supplies the defaults above.

```bash
pixi run python scripts/prune_iterate_cv.py --dataset out/pipeline/prepared --splits out/pipeline/holdout_bioproject/splits --label ph --scheme bioproject_logo --out out/prune_ph --config pipeline.yaml --iterations 5
```

```bash
pixi run python scripts/plot_prune_history.py --history out/prune_ph/prune_history.tsv --out out/prune_ph/figures
```

`--iterations` on the command line overrides `enabled` either way. The plot
draws score against features retained, log-scaled and inverted so reading left
to right follows the pruning; pass several `prune_history.tsv` files and they
overlay, which is how you compare criteria.

> Scores along a pruning trajectory come from the folds the pruning was derived
> from, so the figure says how much is *droppable*, not that a pruned model is
> better. Confirm on the holdout.

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
    # .features is a LazyFrame; .to_df() collects it
    print(f"{name}: {len(feature_set.accessions)} x {len(feature_set.feature_names)}")
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

# Restrict which feature sets are used (also the main memory lever on
# large data -- see "Keeping memory down on large data")
cv = CrossValidator(dataset, models=["rf", "xgboost"], feature_set=["tax_order"])

# Short names: "rf", "gb", "xgb" (or "xgboost") and "svm" (or "svr"). "svm" is
# sklearn's SVR with a linear kernel behind a MaxAbsScaler, so its grid keys
# take the step prefix (`svr__C`, `svr__epsilon`, `svr__kernel`); see
# hyperparameters.yaml. Its fit time grows with the square of the sample
# count, so cap the 16S sets before adding it. It has no feature importances.

# Cross validation run; parameters is required and n_jobs is number of CV done in parallel. Default will detect CPU core
results = cv.run(param_path="parameters.yaml", n_jobs = 8)

# Grid Cross validation run -> using GridSearchCV from sklearn
# n_jobs=None uses every core this job can use. Every candidate is written to
# grid_results.csv by export_result; see "Grid search on all four datasets"
results_grid = cv.run_grid(param_path="hyperparameters.yaml", n_jobs=None)

# Export CV result package. save_models/fold_table default to True here;
# the pipeline turns them off via outputs.cv_save_models / cv_fold_table
CV_Result.export_result(
    results, "out/cv_results", save_models=False, fold_table=False
)
```

### Grid search on all four datasets

`cv.run_grid` is a thin layer over sklearn's `GridSearchCV`: one search per
feature set x label x scheme x model, on the predefined folds. Each estimator
trains on one thread while `GridSearchCV` spreads the fits across `n_jobs`
workers. The winner is the best pooled R2, the score the rest of the workflow
ranks on, and is refit once with all the threads.

```bash
pixi run python scripts/run_grid_search.py --dataset 18S_marine
```

`--dataset` also takes `ITS_soil`, `16S_marine`, `16S_soil` or `all`. They share
one config, `scripts/grid_search.yaml`, so no copy of it can keep the previous
dataset's paths. The grids are in `hyperparameters.yaml`. Rerunning the same
command resumes, skipping every finished label x scheme.

Every labelled sample is used unless `--n-per-label N` caps each label. The
holdout is split off and left unscored: the run stops at the final fit of each
winner on the CV pool (`final_fit/`), and saves that model under `best_models/`
so the holdout can be scored once a setting is chosen.

Under `ML_outputs/grid_search_full/`, or `grid_search_n<N>/` with a cap
(`--root` changes it):

| Path | Contents |
|---|---|
| `<dataset>/relabund_prev0/holdout_*/runs/<label>/<scheme>/cv_results/grid_results.csv` | Every candidate: every column of sklearn's `cv_results_` unchanged (per-fold scores, means, ranks, timings, `param_*`), led by `pooled_r2`, `rank_pooled_r2` and `selected` |
| `.../runs/<label>/<scheme>/final_fit/full_fit_metrics.csv` | Each winner refit on the whole CV pool: in-sample and out-of-bag R2 beside its CV score |
| `grid_report.csv`, `figures/grid_report__<dataset>.png` | Per label x scheme, the settings each model chose (searched ones only), its CV score, and the holdout score of the model sent to the holdout |
| `grid_winners.csv` | The model each dataset x label x scheme sent to the holdout, with both scores |
| `figures/grid_overview.png` | That table as a figure: CV beside holdout |
| `figures/grid_params__<dataset>__<label>__<model>.png` | Pooled R2 against each searched hyperparameter, per scheme |

`scripts/plot_grid_search.py --root <root>` redraws the figures from whatever
has finished so far.

### Picking the best: per label, per scheme

After a run the winners are tracked at three levels:

| attribute | best across |
|---|---|
| `cv.best_result` | everything — one global winner |
| `cv.best_result_by_label` | each label |
| `cv.best_result_by_label_scheme` | each **(label, scheme)** pair |

The last one is what the pipeline uses, and the reason is the `random`
control: it leaks group structure into the validation folds, so it usually
wins outright. Collapsing to one winner per label therefore throws away every
grouped scheme — exactly the results that say whether the model generalises.

```python
# One best-model package per label x scheme, under <out>/<label>/<scheme>/,
# plus best_models_summary.csv sorted best -> worst by avg validation R2
CV_Result.export_best_results_by_scheme(
    cv.best_result_by_label_scheme,
    "out/best_models",
    best_result_key_by_label_scheme=cv.best_result_key_by_label_scheme,
)

for (label, scheme), result in cv.best_result_by_label_scheme.items():
    print(label, scheme, result.avg_validation_r2)
```

`CV_Result.export_best_results(results, "out/best_models")` is the older
one-per-label export and still works; use it when you genuinely want a single
winner per label.


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
| `shap_summary.csv` | Ranked table: `rank, feature, mean_abs_shap, mean_shap, direction` — sorted by `mean_abs_shap`. See *Reading direction* above for the last two. |
| `shap_values.csv`  | Full SHAP matrix: rows = samples, cols = features |

`SHAPResult.top_features(n)` returns the top feature names by magnitude;
`top_features_signed(n)` returns `(feature, direction)` pairs instead, so a
negative association is visible without reading the CSV.

## Save, Load, and Visualize

### Save and load dataset

```python
# Save as compressed archive (.tar.gz)
dataset.save("path/to/save/dataset", compress=True)

# Save as directory structure
dataset.save("path/to/save/dataset")

# Load from archive or directory. lazy=True (the default) scans the CSVs
# instead of reading them in, so a wide feature table is never materialised
# until something asks for it.
dataset = Dataset.load("path/to/save/dataset.tar.gz")
dataset = Dataset.load("path/to/save/dataset")
dataset = Dataset.load("path/to/save/dataset", lazy=False)   # read eagerly
dataset = Dataset.scan("path/to/save/dataset")               # = load(lazy=True)
```

The pipeline writes the dataset itself only when asked:

```yaml
outputs:
  save_dataset_path: out/prepared     # omit or null = do not save
  save_dataset_compress: false        # true -> .tar.gz
```

That saved directory is what the Snakemake `prepare` rule produces and every
later job loads, which is why the build happens once per sweep.

### Save splits without saving the dataset

The split tables are usually the only part worth keeping:

```python
# <out>/<label>/holdout.csv  — every sample tagged train/test
dataset.save_holdout_splits("out/splits")
dataset.save_holdout_splits("out/splits", label="temperature")

# <out>/<label>/cv_<scheme>.csv — fold membership per scheme
dataset.save_cv_folds("out/splits")
```

Same layout `Dataset.save` writes under `splits/`, so these files are
interchangeable with a full save and are what `scripts/check_holdout_metadata.py`
and the sweep read back.

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
Every plotting method (`plot_cv_bars`, `plot_feature_importances`,
`visualise_model_performance`, `plot_holdout_distribution`, `plot_shap_summary`)
respects the `formats` list set on `Visualiser`, so each call produces one
output file per format.

### Is the holdout representative?

`plot_holdout_distribution` puts the label's train and test distributions side
by side on **shared bins**, stacked by whichever column defined the split, and
returns the numbers behind the figure:

```python
stats = vis.plot_holdout_distribution(
    "out/splits/temperature/holdout.csv",   # path or DataFrame
    label="temperature",        # default: first column that is not sample/split
    grouping="bioproject",      # stack by this; None = one colour
    bins=30,
    stat=("count", "probability"),   # one row of panels each
    max_categories=10,          # rarer groups fold into a grey "Other"
    output="holdout_distribution_temperature",
)
print(stats["ks"], stats["p"])
```

`stat="count"` keeps the 80/20 size difference visible; `"probability"` puts
both sides on one scale so the *shapes* can be compared. Pass both (the
default) and you get both rows.

The returned dict carries `label`, `n_train`, `n_test`, `median_train`,
`median_test`, `sd_train`, `sd_test`, `ks` and `p` — a two-sample
Kolmogorov–Smirnov test of train against test. **High p = the two sides
agree**, which is what you want; a grouped holdout legitimately scores p ≈ 0,
because holding whole bioprojects out is *meant* to shift the distribution.
Compare against the `random` control to see how much.

The sweep calls this per run and ranks every result into
`holdout_distribution_summary.txt`; `scripts/check_holdout_metadata.py` calls
the same function on metadata alone.

## Development

This project uses strict type checking and code quality tools for reliable development:

```bash
pixi run lint          # flake8 over src/ and test/tests/
pixi run type-check    # mypy, strict
pixi run test          # pytest test/tests/
pixi run local-check   # all three, in that order — run before committing

pixi run format        # autoflake + docformatter + isort + black
pixi run clean         # drop __pycache__ and *.pyc
pixi run sweep …       # shorthand for `snakemake -s workflow/Snakefile`

pixi run pre-commit run --all-files
```

### Type Safety
- Full mypy type checking with strict configuration
- Only source code (`src/`) is type-checked; tests are excluded for faster development
- Pre-commit hooks ensure consistent code quality

### Project Structure

| Path | |
|---|---|
| `src/microbiome_ml/wrangle/` | `Dataset`, metadata/profiles/features components, splitting |
| `src/microbiome_ml/wrangle/icc.py` | per-taxon intraclass correlation against a grouping |
| `src/microbiome_ml/wrangle/shadow.py` | shuffled shadow columns and the noise floor they measure |
| `src/microbiome_ml/train/` | `CrossValidator`, `ModelTrainer`, `CV_Result`, `SHAPAnalyser` |
| `src/microbiome_ml/train/full_fit.py` | diagnostics for the model fitted on the whole training pool |
| `src/microbiome_ml/visualise/` | `Visualiser` |
| `scripts/run_cv_to_final_eval.py` | the whole pipeline in one process, driven by the YAML |
| `scripts/run_holdout_sweep.py` | the same pipeline once per holdout grouping, serially |
| `scripts/pipeline_steps.py` | the pipeline split into `prepare` / `split` / `cv` / `collate` / `split_report` CLI steps |
| `scripts/sweep_tokens.py` | token parsing (`random`, `<column>`, `taxon:<rank>`), dependency-free so the Snakefile can import it without pulling in the package |
| `scripts/check_holdout_metadata.py` | holdout groupings from a metadata table alone |
| `scripts/sweep_transforms.py` | a whole transform benchmark from one YAML: optional subset, one run per combination, summary, figures, pruning and final fit |
| `scripts/sweep_transforms.example.yaml` | every key that script reads |
| `scripts/plot_transform_sweep.py` | one figure per label from a transform sweep |
| `scripts/rank_feature_report.py` | rank comparison, importance concentration and the shadow floor, from finished CV exports |
| `scripts/prune_iterate_cv.py` | the prune/re-run loop |
| `scripts/plot_prune_history.py` | score against features retained, one line per pruning criterion |
| `scripts/convert_features.py` | CSV feature tables to parquet, in place, without rebuilding |
| `scripts/pipeline.example.yaml` | annotated config — every key, with what it does |
| `workflow/Snakefile` | the parallel sweep |
| `test/tests/` | pytest suite (excluded from type checking) |
| `parameters.yaml`, `hyperparameters.yaml` | model params for `cv.run` and `cv.run_grid` |
| `pyproject.toml`, `pixi.toml` | configuration |
