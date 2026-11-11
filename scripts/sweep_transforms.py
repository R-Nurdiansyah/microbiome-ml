#!/usr/bin/env python3
"""Run the pipeline once per feature-transform combination, then summarise.

`features.transform` is a single value, and `Dataset.load` reads its
feature sets from `manifest.json`, so two transforms cannot share one
prepared dataset -- the second `prepare` rewrites the manifest and
orphans the first. Comparing transforms therefore means one full run per
transform, into its own `base_dir`. This drives that loop.

What is an axis here and what is not:

  transform   relabund / clr / presence. A real axis: one run each.
  prevalence  `features.min_prevalence`. A real axis: one run per value
              of `--min-prevalence`. It leaves NO suffix on the
              feature-set name, so two thresholds cannot share a
              `base_dir` without silently overwriting each other --
              which is exactly what a per-combo directory prevents.
  icc         `max_icc: null` returns the feature frame UNTOUCHED, so
              report-only ICC is a diagnostic, not a variant. Only a
              numeric `--icc-max` forks the grid, and each value given
              adds one run per transform.
  shadow      Not an axis. Shadows are added as a companion `<set>_shad`
              beside the untouched clean set, both trained in the same
              run, so `--shadow` is one flag applied to every combo.

So the grid is
`len(transforms) x max(1, len(min_prevalence)) x (1 + len(icc_max))`.

HOW FAR EACH COMBO RUNS is the template's business, not the grid's. A
template with `holdout_evaluation: {enabled: false}` stops at CV, and so
does every combo derived from it: the summary then ranks validation
folds, which makes a winner a candidate rather than a result.
`--holdout` turns the final test on for every combo, `--final-fit` adds
the whole-training-pool diagnostics, and only with `--holdout` does the
summary carry a HOLDOUT block.

Each combo gets its own directory with its own config, prepared dataset,
splits and CV outputs, so nothing collides and any combo can be rerun on
its own. `cv.feature_set` is derived with the package's own
`feature_set_name`, which is the point: a hand-written
`tax_genus_presence_bio` does not exist (the suffix is `_pa`) and CV would
find no feature set to train on.

    # the three transforms, shadows on, ICC reported but not filtering
    pixi run python scripts/sweep_transforms.py \
        --config test_transform.yaml --out ML_outputs/transform_sweep

    # also try dropping the study markers, at two thresholds
    ... --icc-max 0.9 0.75

    # transform x prevalence, all the way through the final test:
    # 3 transforms x 3 thresholds = 9 runs
    ... --min-prevalence 0.01 0.05 0.1 --holdout --final-fit

    # re-read what is on disk without running anything; no axis flags needed
    ... --summary-only

ADDING TO A SWEEP is cheap. The summary covers every combination under
--out, not only this command line's, and a combination already finished
with the same config is not launched again. So add a threshold or a
cutoff by rerunning into the same --out. If settings that affect results
have changed, the sweep refuses to start rather than leave old results
under new settings; use a new --out, or --rerun.

SCREENING: CV is nearly all of the runtime, so `--models rf --no-shadow`
into its own --out is the cheap first pass. Carry the candidates through
the full models and --holdout afterwards.

`--max-workers N` turns the driver into a manager: every config is
written up front, N combinations run at once (each a separate Snakemake
process with its own `--directory`, so they never share a lock), each is
recorded as it lands, and the summary is collated at the end. The
budgets multiply -- N workers at `--cores C` ask for N*C cores.

The summary is written even when combos fail or the job is stopped with
SIGTERM or Ctrl-C, so it still compares the ones that worked. SIGKILL
cannot be caught; rerun with --summary-only after one. A combo that did
not finish is left out, even if an earlier run left its tables on disk.

TOP TO BOTTOM FROM ONE YAML. Everything above can also be set in the
config's `sweep:` section (a flag still wins), so this is enough:

    pixi run python scripts/sweep_transforms.py --config sweep.yaml

scripts/sweep_transforms.example.yaml lists every key. In order, it runs:

  1. subset       `sweep.subset`: cut the inputs to a few samples per label
  2-3. the grid   one config.yaml per combination, checked before starting
  4. each combo   prepare, split, CV, feature report
  5. summary      SUMMARY.txt, the tables, one figure set per CV scheme
  6. diagnostics  ICC and shadow-floor figures
  7. pruning      `sweep.prune`: prune_iterate_cv on each combo x label
  8. final fit    CV, whole-pool fit and holdout, unpruned against pruned
  9. benchmark    scoreboard and runtime figures, the ranking table
  10. the list    what was written, and where

`sweep.runner` picks how stage 4 runs. `snakemake` (the default) is the
workflow above, which can submit to a queue. `local` runs the same steps
with the same folder layout on the machine you are on, with no Snakemake
and no queue, which is the way to go inside an interactive job. Stages 7
and 8 always run on the machine running this script, so with Snakemake
submitting to a cluster, leave `sweep.prune` off on a login node.

Each combination is a child process in either case, so one running out
of memory cannot sink the others, and rerunning skips finished work.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import copy
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
)

import polars as pl
import yaml
from sweep_tokens import run_specs

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.dataset import (
    feature_set_name,
    rank_from_feature_set,
)
from microbiome_ml.wrangle.profiles import canonical_transform
from microbiome_ml.wrangle.shadow import SHADOW_SUFFIX

LOGGER = logging.getLogger("sweep-transforms")

ROOT = Path(__file__).resolve().parents[1]

# Per-combo tables written by `pipeline_steps collate`, concatenated here.
HEADLINE = "cv_sweep_headline.tsv"
ALL_MODELS = "cv_sweep_all.tsv"
STATUS = "sweep_status.tsv"
# Also written by collate, but only when the holdout stage ran -- so
# its absence is the normal state for a CV-only sweep, not a fault.
HOLDOUT = "holdout_sweep_summary.tsv"

# Ranked on the first of these present. Pooled is preferred because
# unequal LOGO folds make the mean of per-fold R2s a weighted average of
# nothing in particular.
RANK_ON = ("best_grouped_pooled_r2", "best_grouped_r2")

# Identifies a run in every summary table.
COMBO_KEYS = [
    "combo",
    "transform",
    "min_prevalence",
    "max_icc",
    "shadow",
]

# Combos that did not end on their own. Anything of theirs on disk may be
# left over from an earlier run into the same --out, so it is not read.
# `not run` is written for every combo before it starts, so it also covers
# a driver killed outright, which has no chance to record anything.
UNFINISHED = ("failed", "interrupted", "not run")

# Differences here change no score, so they are not a conflict.
COSMETIC = ("sweep", "visualise")
# Scores from combos that disagree on these are not comparable at all.
SHARED = ("data", "groupings", "preprocessing", "split")

# Seconds from asking running combos to stop to killing them. PBS allows
# 10 s between SIGTERM and SIGKILL, and the summary has to fit in that.
KILL_AFTER = 5.0

# Set by the signal handler: nothing new starts, and everything already
# in flight winds down through the normal path to the summary.
STOPPING = threading.Event()
RUNNING: Set[subprocess.Popen] = set()

RUNNERS = ("snakemake", "local")
DEFAULT_TRANSFORMS = ["relabund", "clr", "presence"]
DEFAULT_CORES = 8

SCORE = "pooled_validation_r2"
# The fitted model's name in the tables -> the name CV and pruning take.
MODEL_ALIASES = {
    "RandomForestRegressor": "rf",
    "XGBRegressor": "xgboost",
    "GradientBoostingRegressor": "gb",
    "ScaledSVR": "svm",
}
PRUNE_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "mode": "zero",
    "coverage": 0.95,
    "iterations": 3,
    "min_features": 10,
    "scheme": None,
}
VERSIONS = ("unpruned", "pruned")
STAGES = ["prepare", "split", "cv", "reports", "prune", "final_fit"]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
class Combo:
    """One run: a transform, a prevalence, an ICC cutoff, its own dir.

    `min_prevalence=None` means "whatever the template says" -- the
    prevalence axis is opt-in, so a sweep that does not ask for it
    keeps the ids it had before (`clr`, not `clr_prevNone`).
    """

    def __init__(
        self,
        transform: str,
        max_icc: Optional[float],
        shadow: bool,
        out_root: Path,
        min_prevalence: Optional[float] = None,
    ) -> None:
        self.transform = canonical_transform(transform)
        self.max_icc = max_icc
        self.shadow = shadow
        self.min_prevalence = min_prevalence
        parts = [self.transform]
        if min_prevalence is not None:
            # 0.01 -> prev0p01, for the same reason as the ICC part.
            parts.append(f"prev{min_prevalence:g}".replace(".", "p"))
        if max_icc is not None:
            # 0.9 -> icc0p9: a dot in a directory name reads as an
            # extension to half the tools that will touch this tree.
            parts.append(f"icc{max_icc:g}".replace(".", "p"))
        self.id = "_".join(parts)
        self.dir = out_root / self.id
        self.config_path = self.dir / "config.yaml"

    def row(self) -> Dict[str, Any]:
        return {
            "combo": self.id,
            "transform": self.transform,
            "min_prevalence": self.min_prevalence,
            "max_icc": self.max_icc,
            "shadow": self.shadow,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"Combo({self.id})"


def build_grid(
    transforms: List[str],
    icc_max: List[float],
    shadow: bool,
    out_root: Path,
    min_prevalence: Optional[List[float]] = None,
) -> List[Combo]:
    """Transform x prevalence, no ICC filter then one per cutoff.

    An empty (or absent) `min_prevalence` is not a zero-length axis:
    it means one run per transform at whatever the template sets.
    """
    cutoffs: List[Optional[float]] = [None, *[float(v) for v in icc_max]]
    prevalences: List[Optional[float]] = (
        [float(v) for v in min_prevalence] if min_prevalence else [None]
    )
    combos = [
        Combo(transform, cutoff, shadow, out_root, prevalence)
        for transform in transforms
        for prevalence in prevalences
        for cutoff in cutoffs
    ]
    # `pa` and `presence`, or a value given twice, are one id and so one
    # directory. Two runs there would overwrite each other.
    return list({combo.id: combo for combo in combos}.values())


def discover_combos(out_root: Path) -> List[Combo]:
    """Every combination with a config under `out_root`.

    The summary covers what is on disk rather than what this command line
    asked for, so a later sweep into the same --out extends the comparison,
    and --summary-only needs no axis flags repeated.
    """
    combos: List[Combo] = []
    for path in sorted(out_root.glob("*/config.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        features = cfg.get("features") or {}
        icc = (features.get("icc") or {}).get("max_icc")
        shadow = bool((features.get("shadow") or {}).get("enabled"))
        transform = features.get("transform", "relabund")
        # The id only carries a prevalence when it was a sweep axis, so
        # the directory name says which of these two this one was.
        for prevalence in (features.get("min_prevalence"), None):
            combo = Combo(transform, icc, shadow, out_root, prevalence)
            if combo.id == path.parent.name:
                combos.append(combo)
                break
        else:
            LOGGER.warning("%s does not match its directory; skipped", path)
    return combos


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------
def ranks_of(features: Dict[str, Any]) -> List[str]:
    """The ranks this config builds.

    Mirrors `Dataset.add_taxonomic_features`: `all` wins over `ranks`,
    and a missing `ranks` means class only.
    """
    if features.get("all", True):
        return [r.name.lower() for r in TaxonomicRanks.PHYLUM.iter_down()]
    ranks = features.get("ranks")
    if ranks is None:
        return ["class"]
    if isinstance(ranks, (list, tuple)):
        return [str(r).lower() for r in ranks]
    return [str(ranks).lower()]


def named_ranks(wanted: Any, prefix: str) -> List[str]:
    """The ranks a template's `cv.feature_set` asks for.

    Only the first word after the prefix counts, so a name written for
    another transform -- or one that never existed, such as
    `tax_genus_presence_bio` -- still says genus. Anything that is not a
    rank is ignored.
    """
    if not wanted:
        return []
    names = wanted if isinstance(wanted, (list, tuple)) else [wanted]
    known = {rank.name.lower() for rank in TaxonomicRanks}
    found = (rank_from_feature_set(str(name), prefix) for name in names)
    words = [rank.split("_")[0] for rank in found if rank]
    return list(dict.fromkeys(word for word in words if word in known))


def derive_feature_sets(
    features: Dict[str, Any], combo: Combo, wanted: Any = None
) -> List[str]:
    """The set names this combo will actually produce.

    The names are derived rather than taken from the template, because
    the template's `cv.feature_set` was written for whatever transform it
    had set. Its ranks still count: `build_dataset` lets `cv.feature_set`
    narrow what gets built, so a sweep has to as well or `all: true` would
    train all six ranks. The clean set comes first so it leads the summary
    tables; the shadow set follows, and exists only to calibrate the clean
    one's importances.
    """
    prefix = str(features.get("prefix", "tax"))
    prevalence_by = features.get("prevalence_by") or None
    names: List[str] = []
    for rank in named_ranks(wanted, prefix) or ranks_of(features):
        clean = feature_set_name(prefix, rank, combo.transform, prevalence_by)
        names.append(clean)
        if combo.shadow:
            names.append(f"{clean}_{SHADOW_SUFFIX}")
    return names


def _force_enabled(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Set `cfg[name]["enabled"] = True`, repairing a non-mapping.

    `holdout_evaluation: false` is valid in a template and is not a
    mapping, so setdefault alone would return the bool and the
    assignment would raise.
    """
    block = cfg.get(name)
    if not isinstance(block, dict):
        block = {}
        cfg[name] = block
    block["enabled"] = True
    return block


def _absolute(value: Any, base: Path) -> Any:
    """`value` with relative paths anchored at `base`, nested dicts too.

    PurePosixPath as well as Path: on a Windows mount, Path does not count
    a cluster path such as /home/me/x as absolute.
    """
    if isinstance(value, dict):
        return {k: _absolute(v, base) for k, v in value.items()}
    if not value or not isinstance(value, str):
        return value
    if PurePosixPath(value).is_absolute() or Path(value).is_absolute():
        return value
    return (base / value).as_posix()


def derive_config(
    template: Dict[str, Any],
    combo: Combo,
    icc_group: str,
    holdout: bool = False,
    final_fit: bool = False,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The template with this combo's build options and its own paths."""
    cfg = copy.deepcopy(template)
    features = cfg.setdefault("features", {})
    features["transform"] = combo.transform
    # Only when this combo carries one: otherwise the template's own
    # min_prevalence is the setting under test, and overwriting it
    # with a default would quietly change what the sweep compares.
    if combo.min_prevalence is not None:
        features["min_prevalence"] = combo.min_prevalence

    template_shadow = (template.get("features") or {}).get("shadow") or {}
    # Report is always on: it costs one group_by over the long table and
    # is the half of ICC worth having. Filtering happens only when this
    # combo carries a cutoff.
    features["icc"] = {
        "enabled": True,
        "group": icc_group,
        "report": True,
        "max_icc": combo.max_icc,
    }
    features["shadow"] = {
        "enabled": bool(combo.shadow),
        "n": int(template_shadow.get("n", 200)),
        "seed": int(template_shadow.get("seed", 1234)),
    }

    # Forced on for every combo rather than merged per combo: a
    # holdout that ran for two combinations out of six is not a
    # benchmark, it is two numbers and four blanks. Left alone when
    # not asked for, so the sweep stops where the template stops.
    if holdout:
        _force_enabled(cfg, "holdout_evaluation")
    if final_fit:
        block = _force_enabled(cfg, "final_fit")
        block.setdefault("oob", True)
        block.setdefault("top_n", 20)

    base = combo.dir
    outputs = cfg.setdefault("outputs", {})
    # as_posix, not str: the driver may be run from a Windows mount of a
    # Linux filesystem, and a backslash in a config bound for the cluster
    # is not a separator there.
    outputs["base_dir"] = base.as_posix()
    # Repointed rather than inherited: the template's paths are absolute
    # and shared, so leaving them would have every combo overwrite the
    # same splits and the same saved dataset.
    outputs["splits_dir"] = (base / "holdout_splits").as_posix()
    if outputs.get("save_dataset_path"):
        outputs["save_dataset_path"] = (base / "dataset").as_posix()

    # Snakemake runs each combo inside its own --directory, so a path that
    # was relative to where the sweep was launched would no longer resolve.
    launch_dir = Path.cwd()
    if "data" in cfg:
        cfg["data"] = _absolute(cfg["data"], launch_dir)
    if isinstance(cfg.get("groupings"), dict):
        cfg["groupings"]["file"] = _absolute(
            cfg["groupings"].get("file"), launch_dir
        )
    cv = cfg.setdefault("cv", {})
    if models:
        cv["models"] = list(models)
    # Written out even when absent: the steps' default is relative as well.
    cv["param_path"] = _absolute(
        cv.get("param_path", "parameters.yaml"), launch_dir
    )
    cv["feature_set"] = derive_feature_sets(
        features, combo, cv.get("feature_set")
    )
    return cfg


def write_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _section(cfg: Dict[str, Any], name: str) -> Any:
    """One config section as far as results go; n_jobs changes speed only."""
    value = cfg.get(name)
    if name == "cv" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "n_jobs"}
    return value


def config_conflicts(
    configs: Dict[str, Dict[str, Any]], prepared: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Where new configs disagree with combos already built under --out.

    Snakemake does not treat the config as an input, so against its own
    earlier config a changed combo would skip `prepare` and report the old
    data under the new settings. Against any other combo, the risk is the
    summary ranking scores that were never comparable.
    """
    conflicts: List[str] = []
    for combo_id, cfg in configs.items():
        for other_id, old in prepared.items():
            names = (
                (set(cfg) | set(old)) - set(COSMETIC)
                if other_id == combo_id
                else SHARED
            )
            changed = sorted(
                name
                for name in names
                if _section(cfg, name) != _section(old, name)
            )
            if changed:
                conflicts.append(
                    "  {} differs from {} in: {}".format(
                        combo_id,
                        (
                            "its earlier run"
                            if other_id == combo_id
                            else other_id
                        ),
                        ", ".join(changed),
                    )
                )
    return conflicts


def _input_paths(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [p for v in value.values() for p in _input_paths(v)]
    return [value] if value and isinstance(value, str) else []


def missing_inputs(cfg: Dict[str, Any]) -> List[str]:
    """Input files that do not exist, found before any job is started."""
    groupings = cfg.get("groupings")
    paths = _input_paths(cfg.get("data")) + _input_paths(
        {
            "file": (
                groupings.get("file") if isinstance(groupings, dict) else None
            ),
            "param_path": (cfg.get("cv") or {}).get("param_path"),
        }
    )
    return [path for path in paths if not Path(path).exists()]


def override_warnings(template: Dict[str, Any], shadow: bool) -> List[str]:
    """Template settings this sweep is about to replace in every combo."""
    features = template.get("features") or {}
    warnings: List[str] = []
    max_icc = (features.get("icc") or {}).get("max_icc")
    if max_icc not in (None, "", False):
        warnings.append(
            "the template sets features.icc.max_icc={}, but combinations "
            "without --icc-max run with NO ICC filter. Pass --icc-max {} to "
            "test it.".format(max_icc, max_icc)
        )
    if shadow and not (features.get("shadow") or {}).get("enabled"):
        warnings.append(
            "the template leaves features.shadow off, but this sweep turns "
            "it on, which adds a feature set for CV to train. Pass "
            "--no-shadow to keep it off."
        )
    return warnings


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
# Snakemake's job-count flags. If `--extra` carries one of these the
# driver leaves its own `--cores` off, because on a cluster run the
# budget is `--jobs` and passing both invites the two to disagree.
JOB_FLAGS = ("--cores", "-c", "--jobs", "-j")


def command_for(
    combo: Combo, cores: int, pixi: bool, extra: List[str]
) -> List[str]:
    """The shell command that runs one combo.

    `extra` is appended verbatim and never shell-interpreted, so a submit
    command with spaces (`--cluster-generic-submit-cmd "mqsub ... --"`)
    survives as one argument instead of being re-split.
    """
    # --directory gives this combination its own `.snakemake`, so parallel
    # combinations do not fight over one lock and an interrupted combination
    # is unlocked on its own. That makes the Snakefile path absolute: it is
    # no longer relative to the directory Snakemake ends up working in.
    cmd = [
        "snakemake",
        "-s",
        (ROOT / "workflow" / "Snakefile").as_posix(),
        "--directory",
        combo.dir.as_posix(),
        "--configfile",
        combo.config_path.as_posix(),
    ]
    if any(arg.split("=", 1)[0] in JOB_FLAGS for arg in extra):
        LOGGER.info(
            "--extra sets the job count, so --cores %d is not passed", cores
        )
    else:
        cmd += ["--cores", str(cores)]
    if pixi:
        cmd = ["pixi", "run", "-e", "shap", *cmd]
    return [*cmd, *extra]


def _stop_running(signum: int, _frame: Any) -> None:
    """Signal handler: wind the sweep down so the summary still gets written.

    Raises nothing on purpose. The combos end, each is recorded as
    interrupted, and `main` carries on to the summary as usual.
    """
    LOGGER.error(
        "signal %s: stopping %d running combination(s)", signum, len(RUNNING)
    )
    STOPPING.set()
    # Only these: a process started after the signal was never asked to
    # stop. The timer is never cancelled, so reading RUNNING when it fires
    # killed whatever had started since -- in the tests, the next test's
    # child, seconds later and in the same process.
    targets = list(RUNNING)
    for proc in targets:
        proc.terminate()

    def kill_stragglers() -> None:
        for proc in targets:
            if proc.poll() is None:
                proc.kill()

    timer = threading.Timer(KILL_AFTER, kill_stragglers)
    timer.daemon = True
    timer.start()


def _run_to_end(cmd: List[str], handle: Any) -> int:
    """Run `cmd`, keeping it where `_stop_running` can reach it."""
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
    )
    RUNNING.add(proc)
    try:
        # The signal may have landed between the check in run_combo and
        # RUNNING.add, in which case nothing has told this one to stop.
        if STOPPING.is_set():
            proc.terminate()
        return proc.wait()
    finally:
        RUNNING.discard(proc)


def run_combo(
    combo: Combo,
    cores: int,
    pixi: bool,
    extra: List[str],
    dry_run: bool,
) -> Dict[str, Any]:
    """Run one combo under Snakemake; returns its status row."""
    return _launch(
        combo,
        command_for(combo, cores, pixi, extra),
        "sweep_transforms.log",
        dry_run,
    )


def child_command(
    combo: Combo, stage_flag: str, cores: int, log_level: str
) -> List[str]:
    """This script, rerun on one combination's config.

    A child process rather than a call: a combination that runs out of
    memory dies alone, its memory is returned before the next one starts,
    and `_stop_running` can reach it the same way it reaches Snakemake.
    """
    return [
        sys.executable,
        "-u",
        Path(__file__).resolve().as_posix(),
        stage_flag,
        combo.config_path.as_posix(),
        "--cores",
        str(cores),
        "--log-level",
        str(log_level),
    ]


def _launch(
    combo: Combo, cmd: List[str], log_name: str, dry_run: bool
) -> Dict[str, Any]:
    """Run `cmd` for `combo`; returns its status row. Never raises on failure."""
    LOGGER.info("[%s] %s", combo.id, " ".join(cmd))
    row = {**combo.row(), "command": " ".join(cmd)}
    if dry_run:
        return {**row, "status": "dry-run", "returncode": None, "seconds": 0.0}
    if STOPPING.is_set():
        return {
            **row,
            "status": "interrupted",
            "returncode": None,
            "seconds": 0.0,
        }

    log_path = combo.dir / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        try:
            code = _run_to_end(cmd, handle)
        except OSError as exc:
            handle.write("\ncould not launch: " + str(exc) + "\n")
            code = -1
    elapsed = time.time() - started
    if code == 0:
        status = "ok"
    else:
        status = "interrupted" if STOPPING.is_set() else "failed"
    LOGGER.log(
        logging.INFO if code == 0 else logging.ERROR,
        "[%s] %s in %.1f min (rc=%s); log: %s",
        combo.id,
        status,
        elapsed / 60.0,
        code,
        log_path,
    )
    return {
        **row,
        "status": status,
        "returncode": code,
        "seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------
def read_status(path: Path) -> List[Dict[str, Any]]:
    """The recorded status rows. Unreadable means none, not a lost summary."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return []


def _read_tsv(path: Path, separator: str = "\t") -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pl.read_csv(
            path, separator=separator, infer_schema_length=10000
        )
    except Exception as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return None
    return frame if frame.height else None


def gather(
    combos: List[Combo],
    filename: str,
    separator: str = "\t",
    optional: bool = False,
    skip: Collection[str] = (),
) -> Optional[pl.DataFrame]:
    """Concatenate one per-combo table across combos, stamped with the combo.

    `diagonal_relaxed`, because a combo that failed early or ran under an
    older build can be missing columns the others have; dropping to the
    intersection would quietly lose the metric being compared.

    `skip` names combos that did not finish. A table of theirs may be left
    over from an earlier run, and would then rank as if it were this one's.
    """
    frames = []
    for combo in combos:
        if combo.id in skip:
            LOGGER.info(
                "[%s] did not finish; not reading %s", combo.id, filename
            )
            continue
        frame = _read_tsv(combo.dir / filename, separator)
        if frame is None:
            # `optional` tables are missing by design whenever the
            # stage that writes them was switched off, so saying so
            # once per combo at WARNING would be noise, not news.
            log = LOGGER.debug if optional else LOGGER.warning
            log("[%s] no %s", combo.id, filename)
            continue
        # A table that already carries one of the stamp names would be
        # duplicated by the select below and take the whole summary down
        # with it. The driver knows which combo wrote the file and the
        # file does not, so the stamp wins.
        clash = [c for c in COMBO_KEYS if c in frame.columns]
        if clash:
            LOGGER.warning(
                "[%s] %s already has column(s) %s; overwriting with the "
                "combination's own values",
                combo.id,
                filename,
                clash,
            )
            frame = frame.drop(clash)
        stamp = combo.row()
        frames.append(
            frame.with_columns(
                [pl.lit(stamp[k]).alias(k) for k in COMBO_KEYS]
            ).select([*COMBO_KEYS, *frame.columns])
        )
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def _rank_column(frame: pl.DataFrame) -> Optional[str]:
    for name in RANK_ON:
        if name in frame.columns:
            return name
    return None


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return format(number, spec)


def report_text(
    headline: Optional[pl.DataFrame],
    status: pl.DataFrame,
    combos: List[Combo],
    holdout: Optional[pl.DataFrame] = None,
) -> str:
    """The one file to read: which transform won, per label."""
    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add("TRANSFORM SWEEP SUMMARY")
    add("=" * 78)
    add("")
    add(str(len(combos)) + " combination(s) in the grid:")
    for combo in combos:
        icc = (
            "none"
            if combo.max_icc is None
            else "max_icc=" + format(combo.max_icc, "g")
        )
        prev = (
            "template"
            if combo.min_prevalence is None
            else format(combo.min_prevalence, "g")
        )
        add(
            "  {:<28} transform={:<9} min_prev={:<9} "
            "icc={:<14} shadow={}".format(
                combo.id,
                combo.transform,
                prev,
                icc,
                "on" if combo.shadow else "off",
            )
        )
    add("")

    if "status" in status.columns:
        bad = status.filter(pl.col("status").is_in(list(UNFINISHED)))
        if bad.height:
            add("!! " + str(bad.height) + " combination(s) did not finish:")
            for row in bad.iter_rows(named=True):
                add(
                    "   {:<24} {} rc={}".format(
                        row.get("combo"),
                        row.get("status"),
                        row.get("returncode"),
                    )
                )
            add("   Their rows are absent below, not zero. Read their logs.")
            add(
                "   Rerun one by hand since? Delete "
                "sweep_transforms_status.json and use --summary-only."
            )
            add("")
        # "unknown" means only that this summary was rebuilt without the
        # sweep's status file (--summary-only on a tree built elsewhere),
        # not that anything failed. Said plainly rather than alarmingly.
        unknown = status.filter(pl.col("status") == "unknown")
        if unknown.height:
            add(
                "Note: no sweep status file, so exit codes are unknown; "
                "the tables below are whatever is on disk."
            )
            add("")

    if headline is None:
        add("No " + HEADLINE + " found in any combination.")
        add("Nothing to compare -- check the per-combo logs.")
        return "\n".join(lines) + "\n"

    rank_on = _rank_column(headline)
    if rank_on is None:
        add("Headline tables carry neither pooled nor mean validation R2.")
        return "\n".join(lines) + "\n"

    add("Ranked on " + rank_on + " (the honest, grouped CV scheme).")
    add("")

    group_keys = [
        c for c in ("holdout_grouping", "label") if c in headline.columns
    ]
    # fill_nan: polars sorts NaN above every number, so a combo with no
    # usable score would otherwise take the star.
    frame = headline.with_columns(
        pl.col(rank_on).cast(pl.Float64, strict=False).fill_nan(None)
    )
    # A combo from an older build has no pooled score at all, which says
    # nothing about how good it is. Named, so last place is not misread.
    unscored = frame.filter(pl.col(rank_on).is_null())["combo"].to_list()
    if unscored:
        add(
            "No " + rank_on + ", so ranked last whatever their other "
            "scores: " + ", ".join(dict.fromkeys(unscored))
        )
        add("")
    for keys, block in frame.group_by(group_keys, maintain_order=True):
        add("-" * 78)
        add("  ".join(k + "=" + str(v) for k, v in zip(group_keys, keys)))
        add("-" * 78)
        add(
            "  {:<28} {:>8} {:>8} {:>8}  {:<30} {}".format(
                "combo", "R2", "control", "gap", "feature set", "model"
            )
        )
        ranked = block.sort(rank_on, descending=True, nulls_last=True)
        for i, row in enumerate(ranked.iter_rows(named=True)):
            control = row.get("control_pooled_r2")
            if control is None:
                control = row.get("control_r2")
            add(
                "{} {:<28} {:>8} {:>8} {:>8}  {:<30} {}".format(
                    "*" if i == 0 else " ",
                    str(row.get("combo")),
                    _fmt(row.get(rank_on)),
                    _fmt(control),
                    _fmt(row.get("r2_gap")),
                    str(row.get("best_grouped_feature_set") or "-"),
                    str(row.get("best_grouped_model") or "-"),
                )
            )
        add("")

    # Noted, not alarmed at. A `_shad` set holds every real column PLUS
    # the shuffled ones, so it is a superset of its clean twin and the
    # two are expected to score about the same; which one edges ahead is
    # fold noise. An earlier version of this report called it a finding,
    # which was wrong -- the scores cannot answer that question.
    if "best_grouped_feature_set" in headline.columns:
        shadowed = headline.filter(
            pl.col("best_grouped_feature_set")
            .cast(pl.Utf8)
            .str.ends_with("_" + SHADOW_SUFFIX)
        )
        if shadowed.height:
            add(
                "The shadow-augmented set edged ahead in "
                + str(shadowed.height)
                + " case(s):"
            )
            for row in shadowed.iter_rows(named=True):
                add(
                    "   {} / {} -> {}".format(
                        row.get("combo"),
                        row.get("label"),
                        row.get("best_grouped_feature_set"),
                    )
                )
            add(
                "   Expected: a _shad set is its clean twin plus shuffled "
                "columns, so the two should score alike and either may "
                "come out on top."
            )
            add(
                "   Whether the model LEANED on the shuffled columns is "
                "an importance question, not a score one -- see "
                "<combo>/feature_report/shadow_floor.tsv."
            )
            add("")

    add("=" * 78)
    add("* = best combination for that grouping x label.")
    add("")
    add("Read with care:")
    add(
        "  - `control` is the random-split positive control; it leaks "
        "group structure, so it should look good."
    )
    add(
        "  - `gap` = control - grouped. Large means the signal was "
        "mostly within-study."
    )
    if holdout is None:
        add(
            "  - These are validation folds, not the holdout. A "
            "transform winning here is a candidate, not a result; "
            "rerun the sweep with --holdout to settle it."
        )
    else:
        add(
            "  - The blocks above are validation folds. The HOLDOUT "
            "block is the final test, on samples no fold ever saw; "
            "where the two disagree, that one is the result."
        )
    add("=" * 78)
    return _holdout_block(lines, holdout)


def _holdout_block(lines: List[str], holdout: Optional[pl.DataFrame]) -> str:
    """Append the final-test ranking, when the holdout ran.

    Printed after the CV blocks and the caveats, because it is the
    only table here that answers the question rather than nominating
    a candidate for it: these are samples no fold ever saw.
    """
    add = lines.append
    if holdout is None or "r2" not in holdout.columns:
        return "\n".join(lines) + "\n"
    add("")
    add("=" * 78)
    add("HOLDOUT -- THE FINAL TEST")
    add("=" * 78)
    frame = holdout.with_columns(
        pl.col("r2").cast(pl.Float64, strict=False).fill_nan(None)
    )
    keys = [
        c
        for c in ("holdout_grouping", "label", "scheme")
        if c in frame.columns
    ]
    if not keys:
        keys = ["combo"]
    for values, block in frame.group_by(keys, maintain_order=True):
        add("-" * 78)
        add("  ".join(k + "=" + str(v) for k, v in zip(keys, values)))
        add("-" * 78)
        add(
            "  {:<28} {:>8} {:>8} {:>8} {:>7}  {}".format(
                "combo", "R2", "MAE", "PCC", "n_test", "feature set"
            )
        )
        ranked = block.sort("r2", descending=True, nulls_last=True)
        for i, row in enumerate(ranked.iter_rows(named=True)):
            add(
                "{} {:<28} {:>8} {:>8} {:>8} {:>7}  {}".format(
                    "*" if i == 0 else " ",
                    str(row.get("combo")),
                    _fmt(row.get("r2")),
                    _fmt(row.get("mae")),
                    _fmt(row.get("pcc")),
                    str(row.get("n_test") or "-"),
                    str(row.get("feature_set") or "-"),
                )
            )
        add("")
    add(
        "A combination missing here ran CV but not the holdout, or "
        "its holdout produced no rows."
    )
    add("=" * 78)
    return "\n".join(lines) + "\n"


def summarise(combos: List[Combo], out_root: Path) -> Path:
    """Write the combined tables and the text report. Returns the report."""
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    status_path = out_root / "sweep_transforms_status.json"
    rows = read_status(status_path) or [
        {**c.row(), "status": "unknown", "returncode": None} for c in combos
    ]
    # Every row, not the first 100: only a late row may carry `error`.
    status = pl.DataFrame(rows, infer_schema_length=None)
    skip = {r["combo"] for r in rows if r.get("status") in UNFINISHED}

    headline = gather(combos, HEADLINE, skip=skip)
    all_models = gather(combos, ALL_MODELS, skip=skip)
    run_status = gather(combos, STATUS, skip=skip)
    holdout = gather(combos, HOLDOUT, optional=True, skip=skip)

    if headline is not None:
        headline.write_csv(summary_dir / "headline.tsv", separator="\t")
        LOGGER.info(
            "Wrote %s (%d rows)",
            summary_dir / "headline.tsv",
            headline.height,
        )
    if all_models is not None:
        all_models.write_csv(summary_dir / "all_models.tsv", separator="\t")
        LOGGER.info(
            "Wrote %s (%d rows)",
            summary_dir / "all_models.tsv",
            all_models.height,
        )
    if run_status is not None:
        run_status.write_csv(summary_dir / "run_status.tsv", separator="\t")
    status.write_csv(summary_dir / "combo_status.tsv", separator="\t")
    if holdout is not None:
        holdout.write_csv(summary_dir / "holdout.tsv", separator="\t")
        LOGGER.info(
            "Wrote %s (%d rows)",
            summary_dir / "holdout.tsv",
            holdout.height,
        )
    else:
        LOGGER.info(
            "No %s in any combination: the holdout stage did not run. "
            "Rerun with --holdout for a final-test comparison.",
            HOLDOUT,
        )

    report = summary_dir / "SUMMARY.txt"
    report.write_text(
        report_text(headline, status, combos, holdout), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", report)
    return report


def run_all(
    combos: List[Combo],
    template: Dict[str, Any],
    icc_group: str,
    args: argparse.Namespace,
    status_path: Path,
) -> List[Dict[str, Any]]:
    """Write every config, then run the combinations and record them.

    Up to `--max-workers` combinations run at once. Each one is a
    separate Snakemake process with its own working directory, so the
    only shared state is the status file, which is written under a
    lock as each combination lands -- an interrupted sweep still leaves
    a usable record. Rows for combinations outside this run are kept.

    A combination that already finished with this exact config is not
    launched again, unless --rerun asks for it, so extending a sweep only
    pays for what is new.

    Note that the budgets multiply: N workers each given `--cores C`
    ask for N*C cores in total.
    """
    # Round-tripped so it compares equal to what comes back from disk.
    configs = {
        combo.id: yaml.safe_load(
            yaml.safe_dump(
                derive_config(
                    template,
                    combo,
                    icc_group,
                    holdout=args.holdout,
                    final_fit=args.final_fit,
                    models=args.models,
                ),
                sort_keys=False,
            )
        )
        for combo in combos
    }
    earlier = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in combos[0].dir.parent.glob("*/config.yaml")
    }
    # Before anything is written, so a refused run leaves the tree as it was.
    if not args.rerun:
        conflicts = config_conflicts(
            configs,
            {
                name: cfg
                for name, cfg in earlier.items()
                if (
                    combos[0].dir.parent / name / "prepared" / "manifest.json"
                ).exists()
            },
        )
        if conflicts:
            sys.exit(
                "These settings differ from combinations already built "
                "under --out:\n"
                + "\n".join(conflicts)
                + "\nFinished steps would not rerun, so old results would "
                "sit under the new settings. Use a new --out, or pass "
                "--rerun (add `--extra -R prepare` or `--extra -R cv` to "
                "rebuild from that step)."
            )

    recorded = {row["combo"]: row for row in read_status(status_path)}
    to_run: List[Combo] = []
    for combo in combos:
        cfg = configs[combo.id]
        if (
            not args.rerun
            and not args.dry_run
            and earlier.get(combo.id) == cfg
            and recorded.get(combo.id, {}).get("status") == "ok"
            and (combo.dir / HEADLINE).exists()
        ):
            LOGGER.info("[%s] already finished; not rerun", combo.id)
            continue
        write_config(cfg, combo.config_path)
        LOGGER.info(
            "[%s] config -> %s (feature_set=%s)",
            combo.id,
            combo.config_path,
            cfg["cv"]["feature_set"],
        )
        to_run.append(combo)

    statuses: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def save() -> None:
        # A dry run ran nothing, so it has nothing to say about the record
        # of a sweep that did.
        if not args.dry_run:
            status_path.write_text(
                json.dumps(list(recorded.values()), indent=2),
                encoding="utf-8",
            )

    def not_run(combo: Combo) -> Dict[str, Any]:
        return {
            **combo.row(),
            "status": "not run",
            "returncode": None,
            "seconds": 0.0,
        }

    def record(row: Dict[str, Any]) -> None:
        with lock:
            statuses.append(row)
            recorded[row["combo"]] = row
            save()
        LOGGER.info(
            "progress: %d/%d combination(s) finished",
            len(statuses),
            len(to_run),
        )

    # Checked by each worker rather than by cancelling futures, because a
    # free worker would otherwise pick up the next combination, and start
    # hours of work, before the main thread had a chance to cancel it.
    abandon = threading.Event()

    def run_one(combo: Combo) -> None:
        if abandon.is_set():
            record(not_run(combo))
            return
        try:
            if args.runner == "local":
                row = _launch(
                    combo,
                    child_command(
                        combo, "--run-combo", args.cores, args.log_level
                    ),
                    "sweep_transforms.log",
                    args.dry_run,
                )
            else:
                row = run_combo(
                    combo,
                    args.cores,
                    not args.no_pixi,
                    list(args.extra),
                    args.dry_run,
                )
        except Exception as exc:  # a worker must not sink the sweep
            LOGGER.exception("[%s] worker raised", combo.id)
            row = {
                **combo.row(),
                "status": "failed",
                "returncode": None,
                "error": str(exc),
            }
        if row["status"] == "failed" and args.stop_on_error:
            LOGGER.error("stopping: %s failed", combo.id)
            abandon.set()
        record(row)

    recorded.update({combo.id: not_run(combo) for combo in to_run})
    save()
    workers = max(1, int(args.max_workers))
    LOGGER.info(
        "running %d combination(s), %d at a time (%d x --cores %d)",
        len(to_run),
        workers,
        workers,
        args.cores,
    )
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one, to_run))
    return statuses


# ---------------------------------------------------------------------------
# Stage 1: the subset
# ---------------------------------------------------------------------------
def subset_paths(data: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Where `subset_tables` writes each input.

    `study_titles` and `root` are optional, and left out of a subset.
    """
    out_dir = Path(out_dir)
    subset: Dict[str, Any] = {"study_titles": None, "root": None}
    for key in ("metadata", "attributes", "profiles"):
        if data.get(key):
            subset[key] = (out_dir / f"{key}.csv").as_posix()
    subset["labels"] = {
        name: (out_dir / f"label_{name}.csv").as_posix()
        for name in data["labels"]
    }
    return subset


def subset_tables(
    data: Dict[str, Any], out_dir: Path, n_per_label: int, seed: int
) -> Dict[str, Any]:
    """Copy the inputs, cut to the chosen samples. Values stay as text.

    Picks up to `n_per_label` samples with a value for EACH label, so rare
    labels do not disappear as they would from a plain random draw. A
    sample can serve several labels, so the total is usually under
    labels x n_per_label. The profiles can be many GB, so they are streamed.
    """
    out_dir = Path(out_dir)
    subset = subset_paths(data, out_dir)
    manifest = out_dir / "subset_manifest.json"
    wanted = {"n_per_label": n_per_label, "seed": seed, "data": data}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) == wanted:
            LOGGER.info("Reusing the subset in %s", out_dir)
            return subset
        # Combos built from these files would keep the same paths in their
        # config, so the check for changed settings could not see this.
        sys.exit(
            f"{out_dir} holds a subset made with different settings. "
            "Set sweep.subset.dir to a new folder."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        name: pl.read_csv(path, infer_schema_length=0)
        for name, path in data["labels"].items()
    }
    picked = []
    for name, table in tables.items():
        usable = table.filter(
            pl.col(table.columns[1]).cast(pl.Float64, strict=False).is_finite()
        )
        chosen = usable.sample(n=min(n_per_label, usable.height), seed=seed)
        LOGGER.info(
            "subset: %s has %d labelled samples, picked %d",
            name,
            usable.height,
            chosen.height,
        )
        picked.append(chosen.select("sample"))
    samples = pl.concat(picked).unique().sort("sample")
    LOGGER.info(
        "subset: %d samples (a sample can serve several labels)",
        samples.height,
    )

    for name, table in tables.items():
        table.join(samples, on="sample", how="semi").write_csv(
            subset["labels"][name]
        )
    for key in ("metadata", "attributes", "profiles"):
        if data.get(key):
            pl.scan_csv(data[key], infer_schema_length=0).join(
                samples.lazy(), on="sample", how="semi"
            ).sink_csv(subset[key])

    # Written last, so an interrupted subset is rebuilt rather than reused.
    manifest.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    return subset


# ---------------------------------------------------------------------------
# Stage 4: one combination, without Snakemake
# ---------------------------------------------------------------------------
class StageClock:
    """Seconds per stage, in <combo>/stage_times.json across resumes."""

    def __init__(self, base: Path) -> None:
        self.path = base / "stage_times.json"
        self.times = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {}
        )

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.time()
        yield
        self.times[name] = self.times.get(name, 0.0) + time.time() - started
        self.path.write_text(
            json.dumps(self.times, indent=2), encoding="utf-8"
        )


def _read_config(config_path: Path) -> Dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def run_stages(config_path: Path, cores: int) -> None:
    """The Snakefile's rules, in order and with its arguments, for one combo.

    A step whose output already exists is skipped, as Snakemake would.
    """
    import pipeline_steps as steps
    import rank_feature_report as report

    base = config_path.parent
    cfg = _read_config(config_path)
    clock = StageClock(base)
    sweep = cfg.get("sweep") or {}
    # The Snakefile hands the steps a copy without its own `sweep` section.
    resolved = base / "resolved_config.yaml"
    resolved.write_text(
        yaml.safe_dump(
            {k: v for k, v in cfg.items() if k != "sweep"}, sort_keys=False
        ),
        encoding="utf-8",
    )
    common = {"config": str(resolved)}
    prepared = base / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    if not (prepared / "manifest.json").exists():
        with clock.stage("prepare"):
            steps.step_prepare(
                argparse.Namespace(**common, set=[], out=str(prepared))
            )

    specs = run_specs(sweep.get("pairs"), sweep.get("holdout_groupings"))
    jobs = []
    with clock.stage("split"):
        for run, spec in specs.items():
            splits = base / run / "splits"
            splits.mkdir(parents=True, exist_ok=True)
            if not (splits / "schemes.json").exists():
                # Pins a pair run to its one label, as _label_sets does.
                pins = (
                    [
                        f"{key}={spec['label']}"
                        for key in (
                            "split.holdout.label",
                            "split.cv.label",
                            "cv.label",
                        )
                    ]
                    if spec["label"] is not None
                    else []
                )
                steps.step_split(
                    argparse.Namespace(
                        **common,
                        set=pins,
                        dataset=str(prepared),
                        grouping=spec["token"],
                        run_name=spec["stamp"],
                        out=str(splits),
                    )
                )
            schemes = json.loads(
                (splits / "schemes.json").read_text(encoding="utf-8")
            )
            for label, names in schemes.items():
                jobs += [(run, spec, label, scheme) for scheme in names]
        steps.step_split_report(argparse.Namespace(out=str(base)))

    # With CV off the workflow stops after the splits, as the Snakefile does.
    if not bool((cfg.get("cv") or {}).get("enabled", True)):
        return
    if not jobs:
        raise RuntimeError("No label/scheme to run; see the split warnings")

    with clock.stage("cv"):
        for run, spec, label, scheme in jobs:
            out = base / run / "runs" / label / scheme
            if (out / "status.json").exists():
                continue
            LOGGER.info("cv %s / %s", label, scheme)
            out.mkdir(parents=True, exist_ok=True)
            steps.step_cv(
                argparse.Namespace(
                    **common,
                    set=[f"cv.n_jobs={cores}"],
                    dataset=str(prepared),
                    splits=str(base / run / "splits"),
                    grouping=spec["token"],
                    run_name=spec["stamp"],
                    label=label,
                    scheme=scheme,
                    out=str(out),
                )
            )

    with clock.stage("reports"):
        steps.step_collate(argparse.Namespace(out=str(base), runs=[]))
        if not (base / HEADLINE).exists():
            raise RuntimeError("CV produced no results for any scheme")
        feature_report = cfg.get("feature_report") or {}
        if bool(feature_report.get("enabled", True)):
            report.run_report(
                base,
                base / "feature_report",
                [
                    float(c)
                    for c in feature_report.get("coverage") or (0.5, 0.9, 0.95)
                ],
                int(feature_report.get("top", 25)),
                str((cfg.get("features") or {}).get("prefix", "tax")),
            )


# ---------------------------------------------------------------------------
# Stages 7 and 8: pruning, then the final fit
# ---------------------------------------------------------------------------
def draws_transform_figures(template: Dict[str, Any]) -> bool:
    """False for a grid search, whose figures compare its settings instead.

    The transform figures rank transforms and read one model per cell; on a
    grid run they show a single transform under headings about choosing one.
    """
    mode = str((template.get("cv") or {}).get("mode", "run")).strip().lower()
    return mode != "run_grid"


def prune_settings(sweep: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """`sweep.prune` over the defaults."""
    return {**PRUNE_DEFAULTS, **((sweep or {}).get("prune") or {})}


def prune_tag(settings: Dict[str, Any]) -> str:
    """Folder name for these settings, so results of others never mix."""
    mode = str(settings["mode"])
    if mode == "coverage":
        mode += f"{float(settings['coverage']):g}"
    return f"{mode}_it{settings['iterations']}_min{settings['min_features']}"


def prune_targets(
    models: pl.DataFrame, scheme: Optional[str] = None
) -> pl.DataFrame:
    """What pruning works on: per combo x label, the best clean set.

    With `scheme`, every row uses that scheme, so their CV scores and
    pruning are comparable. Without it, each row takes its own best grouped
    scheme. `random` is left out then: it leaks group structure into its
    folds, so it would win nearly every time and say nothing about new
    studies.
    """
    on_scheme = (
        pl.col("scheme") == scheme if scheme else pl.col("scheme") != "random"
    )
    return (
        models.filter(
            ~pl.col("feature_set").str.ends_with("_" + SHADOW_SUFFIX)
            & on_scheme
        )
        .sort(SCORE, descending=True, nulls_last=True)
        .group_by(["combo", "label"], maintain_order=True)
        .first()
        .select(
            "combo",
            "transform",
            "min_prevalence",
            "max_icc",
            "holdout_grouping",
            "label",
            "scheme",
            "feature_set",
            "model",
            SCORE,
        )
        .sort("label", "combo")
    )


def warn_missing_targets(
    models: pl.DataFrame, targets: pl.DataFrame, scheme: Optional[str]
) -> None:
    """Say which labels a fixed scheme leaves out, and what they did run."""
    if not scheme:
        return
    lost = (
        models.select("combo", "label")
        .unique()
        .join(targets, on=["combo", "label"], how="anti")
        .get_column("label")
        .unique()
        .sort()
        .to_list()
    )
    for label in lost:
        ran = (
            models.filter(pl.col("label") == label)
            .get_column("scheme")
            .unique()
            .sort()
            .to_list()
        )
        LOGGER.warning(
            "label '%s' has combination(s) without scheme '%s', so those "
            "are not pruned or fitted. Its schemes: %s",
            label,
            scheme,
            ran,
        )


def prune_dir(base: Path, tag: str, row: Dict[str, Any]) -> Path:
    name = f"{row['label']}__{row['scheme']}__{row['model']}"
    return base / "prune" / tag / name


def kept_features(out: Path, round_row: Dict[str, Any]) -> List[str]:
    """The feature list for one round, named after the fitted model."""
    stem = (
        f"{round_row['label']}__{round_row['scheme']}__{round_row['model']}"
        f"__it{round_row['iteration']}.txt"
    )
    text = (out / "kept_features" / stem).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line]


def final_fit(
    base: Path,
    cfg: Dict[str, Any],
    row: Dict[str, Any],
    keep: List[str],
    out_dir: Path,
    cores: int,
) -> None:
    """CV, whole-pool fit and holdout on `keep`, the way step_cv runs them."""
    done = out_dir / "fit_summary.json"
    if done.exists():
        return

    import pipeline_steps as steps
    import prune_iterate_cv as prune

    from microbiome_ml.train.cv import CrossValidator
    from microbiome_ml.train.full_fit import (
        collect_rows,
        write_full_fit_report,
    )
    from microbiome_ml.train.trainer import ModelTrainer
    from microbiome_ml.visualise.visualisations import Visualiser
    from microbiome_ml.wrangle.dataset import Dataset

    label, scheme, name = row["label"], row["scheme"], row["feature_set"]
    visualise = cfg.get("visualise") or {}
    top_n = int(visualise.get("top_n", 20))
    dataset = steps._attach_splits(
        Dataset.load(base / "prepared"),
        base / f"holdout_{row['holdout_grouping']}" / "splits",
        label,
        scheme,
    )
    narrowed = prune.restrict_features(dataset, name, keep)
    cv = CrossValidator(
        dataset=narrowed,
        models=[MODEL_ALIASES[row["model"]]],
        label=label,
        scheme=scheme,
        feature_set=name,
        full_fit=True,
        full_fit_oob=bool((cfg.get("final_fit") or {}).get("oob", True)),
    )
    cv.run(param_path=str(cfg["cv"]["param_path"]), n_jobs=cores)
    best = cv.best_result_by_label_scheme
    if not best:
        raise RuntimeError(f"no CV result for {base.name} {label}/{scheme}")
    write_full_fit_report(best, out_dir / "final_fit", top_n=top_n)
    diagnostics = collect_rows(best)[0]

    evaluation = ModelTrainer(
        dataset=narrowed,
        best_result=best,
        output_model_path=out_dir / "holdout",
    ).train_and_evaluate()
    if isinstance(evaluation, dict):
        evaluation = next(iter(evaluation.values()))

    vis = Visualiser(
        out_dir / "holdout", formats=list(visualise.get("formats") or ["png"])
    )
    tag = f"{label}__{scheme}"
    vis.visualise_model_performance(
        evaluation.predictions,
        evaluation.targets,
        title=f"Holdout diagnostics ({base.name}, {out_dir.name}, {tag})",
        groups=[scheme] * len(evaluation.targets),
        file_name=f"holdout_diagnostics_{tag}",
    )
    vis.plot_feature_importances(
        evaluation, output=f"holdout_feature_importance_{tag}", top_n=top_n
    )

    metrics = evaluation.metrics
    summary = {
        "n_features": len(keep),
        "n_fit_samples": diagnostics.get("n_fit_samples"),
        "cv_pooled_r2": diagnostics.get("cv_pooled_r2"),
        "in_sample_r2": diagnostics.get("in_sample_r2"),
        "oob_r2": diagnostics.get("oob_r2"),
        "overfit_gap": diagnostics.get("overfit_gap"),
        "holdout_r2": metrics.get("r2"),
        "holdout_mae": metrics.get("mae"),
        "holdout_pcc": metrics.get("pcc"),
        "n_test": metrics.get("n_test"),
    }
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def finish_combo(config_path: Path, cores: int) -> int:
    """Prune each target of one combo, then fit both versions.

    Returns how many steps failed. One failure does not stop the rest:
    the other labels are worth having.
    """
    import plot_transform_sweep as pts
    import prune_iterate_cv as prune

    base = config_path.parent
    cfg = _read_config(config_path)
    settings = prune_settings(cfg.get("sweep"))
    tag = prune_tag(settings)
    models = pts.load_models(base.parent / "summary" / "all_models.tsv")
    targets = prune_targets(
        models.filter(pl.col("combo") == base.name), settings["scheme"]
    )
    clock = StageClock(base)
    failed = 0
    for row in targets.iter_rows(named=True):
        if row["model"] not in MODEL_ALIASES:
            LOGGER.warning("no alias for %s; not pruned", row["model"])
            continue
        out = prune_dir(base, tag, row)
        try:
            if not (out / "prune_history.tsv").exists():
                LOGGER.info("prune %s / %s", row["label"], row["scheme"])
                with clock.stage("prune"):
                    prune.run_loop(
                        dataset_dir=base / "prepared",
                        splits_dir=base
                        / f"holdout_{row['holdout_grouping']}"
                        / "splits",
                        out_dir=out,
                        label=row["label"],
                        scheme=row["scheme"],
                        feature_set=row["feature_set"],
                        models=[MODEL_ALIASES[row["model"]]],
                        iterations=int(settings["iterations"]),
                        min_features=int(settings["min_features"]),
                        mode=str(settings["mode"]),
                        coverage=float(settings["coverage"]),
                        param_path=str(cfg["cv"]["param_path"]),
                        n_jobs=cores,
                    )
            rounds = pl.read_csv(
                out / "prune_history.tsv",
                separator="\t",
                infer_schema_length=10000,
            )
            features = {
                "unpruned": kept_features(out, rounds.row(0, named=True)),
                "pruned": kept_features(out, rounds.row(-1, named=True)),
            }
        except Exception:
            LOGGER.exception("pruning failed: %s", row["label"])
            failed += 1
            continue
        for version in VERSIONS:
            LOGGER.info(
                "final fit %s %s: %d features",
                row["label"],
                version,
                len(features[version]),
            )
            try:
                with clock.stage("final_fit"):
                    final_fit(
                        base, cfg, row, features[version], out / version, cores
                    )
            except Exception:
                LOGGER.exception(
                    "final fit failed: %s %s", row["label"], version
                )
                failed += 1
    return failed


def _finish_done(combo: Combo, targets: pl.DataFrame, tag: str) -> bool:
    """Every target of `combo` already has both of its final fits."""
    return all(
        (
            prune_dir(combo.dir, tag, row) / version / "fit_summary.json"
        ).exists()
        for row in targets.filter(pl.col("combo") == combo.id).iter_rows(
            named=True
        )
        if row["model"] in MODEL_ALIASES
        for version in VERSIONS
    )


def run_finish(
    combos: List[Combo],
    targets: pl.DataFrame,
    tag: str,
    args: argparse.Namespace,
) -> List[str]:
    """Stages 7 and 8, up to --max-workers at once. Returns who failed."""

    def one(combo: Combo) -> Dict[str, Any]:
        cmd = child_command(
            combo, "--finish-combo", args.cores, args.log_level
        )
        return _launch(combo, cmd, "sweep_finish.log", False)

    pending = [c for c in combos if not _finish_done(c, targets, tag)]
    workers = max(1, int(args.max_workers))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, pending))
    return [row["combo"] for row in rows if row["status"] != "ok"]


# ---------------------------------------------------------------------------
# Stages 5, 6 and 9: the tables behind the figures
# ---------------------------------------------------------------------------
def icc_tables(combos: List[Combo]) -> List[Any]:
    """(combo, ICC values) for unfiltered runs: taxa before any cut."""
    tables = []
    for combo in combos:
        if combo.max_icc is not None:
            continue
        diagnostics = combo.dir / "prepared" / "diagnostics"
        for path in sorted(diagnostics.glob("*_icc.tsv")):
            frame = pl.read_csv(
                path, separator="\t", infer_schema_length=10000
            )
            values = (
                frame.get_column("icc")
                .cast(pl.Float64, strict=False)
                .drop_nulls()
                .drop_nans()
            )
            tables.append((combo.id, values.to_list()))
    return tables


def shadow_floor_table(
    combos: List[Combo], targets: pl.DataFrame
) -> Optional[pl.DataFrame]:
    """Each shadow floor, for the scheme and model pruning will use."""
    frames = [
        pl.read_csv(
            path, separator="\t", infer_schema_length=10000
        ).with_columns(pl.lit(combo.id).alias("combo"))
        for combo in combos
        for path in [combo.dir / "feature_report" / "shadow_floor.tsv"]
        if path.exists()
    ]
    if not frames:
        return None
    keys = ["combo", "label", "scheme", "model", "feature_set"]
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .join(
            targets.select(
                "combo",
                "label",
                "scheme",
                "model",
                (pl.col("feature_set") + "_" + SHADOW_SUFFIX).alias(
                    "feature_set"
                ),
            ),
            on=keys,
            how="semi",
        )
        .with_columns(
            (100 * pl.col("n_real_above_q95") / pl.col("n_real")).alias(
                "pct_above_noise"
            )
        )
    )


def benchmark_table(
    targets: pl.DataFrame, out_root: Path, tag: str
) -> Optional[pl.DataFrame]:
    """One row per target x version, from each finished final fit."""
    rows = []
    for row in targets.iter_rows(named=True):
        out = prune_dir(out_root / row["combo"], tag, row)
        for version in VERSIONS:
            path = out / version / "fit_summary.json"
            if path.exists():
                rows.append(
                    {
                        "combo": row["combo"],
                        "transform": row["transform"],
                        "min_prevalence": row["min_prevalence"],
                        "max_icc": row["max_icc"],
                        "label": row["label"],
                        "scheme": row["scheme"],
                        "feature_set": row["feature_set"],
                        "model": row["model"],
                        "version": version,
                        **json.loads(path.read_text(encoding="utf-8")),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else None


def rank_benchmark(benchmark: pl.DataFrame) -> pl.DataFrame:
    """Combos ranked by pruned holdout within each label."""
    pruned = benchmark.filter(pl.col("version") == "pruned")
    unpruned = benchmark.filter(pl.col("version") == "unpruned").select(
        "combo",
        "label",
        pl.col("holdout_r2").alias("holdout_r2_unpruned"),
        pl.col("n_features").alias("features_before"),
    )
    return (
        pruned.join(unpruned, on=["combo", "label"], how="left")
        .with_columns(
            (pl.col("holdout_r2") - pl.col("holdout_r2_unpruned")).alias(
                "pruning_change"
            ),
            pl.col("holdout_r2")
            .fill_nan(None)
            .rank(method="min", descending=True)
            .over("label")
            .alias("rank"),
        )
        .select(
            "label",
            "rank",
            "combo",
            "model",
            "scheme",
            "features_before",
            pl.col("n_features").alias("features_after"),
            "holdout_r2_unpruned",
            pl.col("holdout_r2").alias("holdout_r2_pruned"),
            "pruning_change",
            "cv_pooled_r2",
            "overfit_gap",
        )
        .sort(["label", "rank"], nulls_last=True)
    )


def stage_table(combos: List[Combo]) -> Optional[pl.DataFrame]:
    """Minutes per stage per combo, cheapest first."""
    rows = []
    for combo in combos:
        path = combo.dir / "stage_times.json"
        if path.exists():
            times = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "combo": combo.id,
                    **{s: round(times.get(s, 0.0) / 60, 2) for s in STAGES},
                }
            )
    if not rows:
        return None
    return (
        pl.DataFrame(rows)
        .with_columns(pl.sum_horizontal(STAGES).round(2).alias("total_min"))
        .sort("total_min")
    )


def prune_history_table(
    targets: pl.DataFrame, out_root: Path, tag: str
) -> Optional[pl.DataFrame]:
    """Every pruning round, set up so the combos overlay in one figure.

    plot_prune_history draws one line per model in one panel per feature
    set. Setting these draws one line per combo in one panel per label x
    scheme.
    """
    import plot_prune_history as pph

    frames = []
    for row in targets.iter_rows(named=True):
        out = prune_dir(out_root / row["combo"], tag, row)
        path = out / "prune_history.tsv"
        if path.exists():
            frames.append(
                pph.load_history([path]).with_columns(
                    pl.lit(row["combo"]).alias("combo"),
                    pl.col("model").alias("fitted_model"),
                    pl.lit(row["combo"]).alias("model"),
                    pl.lit("all combos").alias("feature_set"),
                )
            )
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed").sort(
        [*pph.PANEL_KEYS, *pph.SERIES_KEYS, "iteration"]
    )


def _best_effort(what: str, draw: Callable[[], Any]) -> None:
    """A table or figure that cannot be made must not cost the results."""
    try:
        draw()
    except Exception:
        LOGGER.exception("could not %s", what)


def _write_tsv(frame: Optional[pl.DataFrame], path: Path) -> None:
    if frame is not None:
        frame.write_csv(path, separator="\t")
        LOGGER.info("Wrote %s (%d rows)", path, frame.height)


def list_outputs(out_root: Path, combos: List[Combo]) -> str:
    """Stage 10: what was written."""
    summary = out_root / "summary"
    lines = ["", "Written under " + out_root.as_posix() + ":"]
    lines += [
        "  " + path.relative_to(out_root).as_posix()
        for path in sorted(summary.rglob("*"))
        if path.is_file()
    ]
    if combos:
        lines += ["", "One combination: " + combos[0].dir.as_posix()]
        lines += [
            "  " + path.name + ("/" if path.is_dir() else "")
            for path in sorted(combos[0].dir.iterdir())
        ]
    return "\n".join(lines)


def final_stages(
    combos: List[Combo],
    out_root: Path,
    template: Dict[str, Any],
    args: argparse.Namespace,
    finish: bool = True,
) -> List[str]:
    """Stages 5 to 10 over what is on disk. Returns combos whose 7-8 failed.

    `finish` False draws only from what exists, for --summary-only.
    """
    on_disk = discover_combos(out_root)
    report = summarise(on_disk, out_root)
    print("\nSummary: " + str(report))
    models_path = report.parent / "all_models.tsv"
    if STOPPING.is_set() or not models_path.exists():
        return []

    import plot_prune_history as pph
    import plot_transform_sweep as pts

    sweep = template.get("sweep") or {}
    formats = list((template.get("visualise") or {}).get("formats") or ["png"])
    figures = report.parent / "figures"
    headline = report.parent / "headline.tsv"
    prune_scheme = prune_settings(sweep)["scheme"]
    try:
        models = pts.load_models(models_path)
        targets = prune_targets(models, prune_scheme)
    except Exception:
        LOGGER.exception("could not read %s; no figures", models_path)
        return []
    warn_missing_targets(models, targets, prune_scheme)
    _write_tsv(targets, report.parent / "prune_targets.tsv")

    schemes = sweep.get("figure_schemes") or sorted(
        models.get_column("scheme").unique().to_list()
    )
    holdout_path = report.parent / "holdout.tsv"

    def draw_schemes() -> None:
        holdout = pts.load_holdout(holdout_path)
        for scheme in schemes:
            pts.plot_sweep(
                models, figures, formats, headline=headline, scheme=scheme
            )
            if holdout is not None:
                pts.plot_holdout(
                    holdout, models, figures, formats, scheme=scheme
                )

    if draws_transform_figures(template):
        _best_effort("draw the sweep figures", draw_schemes)
    else:
        LOGGER.info(
            "cv.mode is run_grid: transform figures skipped; "
            "plot_grid_search.py draws the grid report"
        )
    cutoffs = sorted({c.max_icc for c in on_disk if c.max_icc is not None})
    _best_effort(
        "draw the ICC figure",
        lambda: pts.plot_icc_distribution(
            icc_tables(on_disk), cutoffs, args.icc_group, figures, formats
        ),
    )

    def draw_floor() -> None:
        floor = shadow_floor_table(on_disk, targets)
        if floor is not None:
            pts.plot_shadow_floor(floor, figures, formats)

    _best_effort("draw the shadow figure", draw_floor)

    failed: List[str] = []
    settings = prune_settings(sweep)
    tag = prune_tag(settings)
    if finish and settings["enabled"] and not STOPPING.is_set():
        recorded = {
            row["combo"]: row
            for row in read_status(out_root / "sweep_transforms_status.json")
        }
        failed = run_finish(
            [
                c
                for c in combos
                if recorded.get(c.id, {}).get("status") == "ok"
            ],
            targets,
            tag,
            args,
        )

    def draw_pruning() -> None:
        history = prune_history_table(targets, out_root, tag)
        if history is not None:
            _write_tsv(history, report.parent / "prune_history_all.tsv")
            pph.plot_history(
                history, figures, formats, stem="prune_trajectory"
            )

    def draw_benchmark() -> None:
        benchmark = benchmark_table(targets, out_root, tag)
        if benchmark is not None:
            _write_tsv(benchmark, report.parent / "final_fit_benchmark.tsv")
            _write_tsv(
                rank_benchmark(benchmark),
                report.parent / "benchmark_ranking.tsv",
            )
            pts.plot_scoreboard(benchmark, figures, formats)

    def draw_runtime() -> None:
        table = stage_table(on_disk)
        if table is not None:
            _write_tsv(table, report.parent / "stage_minutes.tsv")
            pts.plot_runtime(table, STAGES, figures, formats)

    _best_effort("draw the pruning figure", draw_pruning)
    _best_effort("build the benchmark", draw_benchmark)
    _best_effort("draw the runtime figure", draw_runtime)
    print(list_outputs(out_root, combos or on_disk))
    return failed


# ---------------------------------------------------------------------------
# Settings: the flag, else the YAML's sweep section, else the default
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _first(*values: Any) -> Any:
    """The first value that was actually set; an empty list counts."""
    return next((v for v in values if v is not None), None)


def resolve_settings(
    args: argparse.Namespace, template: Dict[str, Any]
) -> None:
    """Fill `args` from the template's `sweep` section where no flag was given."""
    sweep = template.get("sweep") or {}
    out = _first(args.out, sweep.get("out"))
    args.out = None if out is None else Path(out)
    args.runner = _first(args.runner, sweep.get("runner"), "snakemake")
    args.transform = _as_list(
        _first(args.transform, sweep.get("transforms"), DEFAULT_TRANSFORMS)
    )
    args.icc_max = [
        float(v) for v in _as_list(_first(args.icc_max, sweep.get("icc_max")))
    ]
    args.min_prevalence = [
        float(v)
        for v in _as_list(
            _first(args.min_prevalence, sweep.get("min_prevalence"))
        )
    ]
    args.icc_group = (
        args.icc_group
        or sweep.get("icc_group")
        or ((template.get("features") or {}).get("icc") or {}).get("group")
        or "bioproject"
    )
    args.models = _first(args.models, sweep.get("models"))
    if args.models is not None:
        args.models = _as_list(args.models)
    args.holdout = args.holdout or bool(sweep.get("holdout"))
    args.final_fit = args.final_fit or bool(sweep.get("final_fit"))
    args.stop_on_error = args.stop_on_error or bool(sweep.get("stop_on_error"))
    args.no_shadow = args.no_shadow or not sweep.get("shadow", True)
    args.no_pixi = args.no_pixi or not sweep.get("pixi", True)
    args.extra = args.extra or _as_list(sweep.get("snakemake_args"))
    args.max_workers = int(
        _first(args.max_workers, sweep.get("max_workers"), 1)
    )
    args.log_level = _first(args.log_level, sweep.get("log_level"), "INFO")

    cores = _first(args.cores, sweep.get("cores"))
    if cores is None and args.runner == "local":
        # The whole allocation shared out, so workers cannot oversubscribe it.
        allocated = os.environ.get("PBS_NCPUS")
        if allocated:
            cores = max(1, int(allocated) // max(1, args.max_workers))
    args.cores = int(_first(cores, DEFAULT_CORES))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "pipeline YAML. Its `sweep:` section can set everything below; "
            "a flag given here wins over it"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "root for the per-combination directories and the summary "
            "(default: sweep.out)"
        ),
    )
    parser.add_argument(
        "--runner",
        choices=RUNNERS,
        default=None,
        help=(
            "how each combination runs: `snakemake`, or `local` for this "
            "machine with no Snakemake or queue (default: sweep.runner, "
            "else snakemake)"
        ),
    )
    parser.add_argument(
        "--transform",
        nargs="+",
        default=None,
        help="transforms to compare (default: sweep.transforms, else all three)",
    )
    parser.add_argument(
        "--icc-max",
        nargs="*",
        type=float,
        default=None,
        help=(
            "ICC cutoffs to also try, one extra run per transform each. "
            "Omit to report ICC without filtering (recommended first)."
        ),
    )
    parser.add_argument(
        "--icc-group",
        default=None,
        help="grouping for ICC (default: features.icc.group, else bioproject)",
    )
    parser.add_argument(
        "--min-prevalence",
        nargs="*",
        type=float,
        default=None,
        help=(
            "min_prevalence values to compare, one run per value "
            "per transform. Omit to keep whatever the template sets."
        ),
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help=(
            "enable holdout_evaluation in every derived config, so each "
            "combination runs through to the final test"
        ),
    )
    parser.add_argument(
        "--final-fit",
        action="store_true",
        help=("enable final_fit: score the fit on the whole training pool"),
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        help="skip the shadow companion sets",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "replace cv.models in every combination. `--models rf` is a "
            "cheap first pass; use its own --out"
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "launch every combination even if already finished, and allow "
            "its settings to differ from what is on disk"
        ),
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "cores per combination (default: sweep.cores, else 8; the local "
            "runner shares out PBS_NCPUS when that is set)"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "combinations to run at once (default: 1, serial). Each "
            "gets --cores, so the totals multiply."
        ),
    )
    parser.add_argument(
        "--no-pixi",
        action="store_true",
        help="do not prefix commands with `pixi run`",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the configs and print the commands, run nothing",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="re-read what is on disk and rewrite the summary",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abandon the sweep at the first failing combination",
    )
    parser.add_argument("--log-level", default=None)
    # Used by the sweep itself, one child per combination. They also rerun
    # one combination by hand, given the config.yaml in its folder.
    parser.add_argument(
        "--run-combo",
        type=Path,
        default=None,
        metavar="CONFIG",
        help="run stage 4 for one combination's config.yaml, in this process",
    )
    parser.add_argument(
        "--finish-combo",
        type=Path,
        default=None,
        metavar="CONFIG",
        help="run stages 7-8 for one combination's config.yaml, in this process",
    )
    # REMAINDER last, so everything after --extra goes to Snakemake.
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="everything after this is appended to the snakemake command",
    )
    args = parser.parse_args()

    if args.run_combo or args.finish_combo:
        sys.exit(_run_child(args))
    if args.config is None:
        parser.error("--config is required")

    template = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    sweep = template.get("sweep") or {}
    resolve_settings(args, template)
    if args.out is None:
        parser.error("give --out, or set sweep.out in the config")
    if args.runner not in RUNNERS:
        parser.error(
            f"sweep.runner must be one of {list(RUNNERS)}, got {args.runner!r}"
        )
    if args.runner == "local" and args.rerun:
        # A local run skips any step with finished output, so --rerun would
        # let changed settings through and leave the old results in place.
        parser.error(
            "--rerun needs the Snakemake runner. For a local run, use a new "
            "output folder, or delete the combination's folder"
        )
    # A percentage typed by mistake would otherwise run for a long time
    # before anything noticed that no taxon can pass it.
    for value in [*args.min_prevalence, *args.icc_max]:
        if not 0 <= value <= 1:
            parser.error(
                f"--min-prevalence and --icc-max are fractions from 0 to 1, "
                f"got {value:g}"
            )

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    icc_group = args.icc_group
    out_root = args.out.resolve()
    combos = build_grid(
        list(args.transform),
        list(args.icc_max),
        not args.no_shadow,
        out_root,
        list(args.min_prevalence),
    )
    LOGGER.info("%d combination(s): %s", len(combos), [c.id for c in combos])

    if args.summary_only:
        on_disk = discover_combos(out_root)
        if not on_disk:
            # Refused rather than overwriting a report with an empty one.
            sys.exit(f"No <combination>/config.yaml under {out_root}.")
        final_stages(on_disk, out_root, template, args, finish=False)
        return

    if args.runner == "local" and args.extra:
        LOGGER.warning("the local runner has no use for --extra; ignored")
    # Asking for shadows in the sweep section is not a surprise to warn about.
    surprise_shadow = not args.no_shadow and sweep.get("shadow") is None
    for warning in override_warnings(template, surprise_shadow):
        LOGGER.warning(warning)
    if not args.dry_run:
        missing = missing_inputs(derive_config(template, combos[0], icc_group))
        if missing:
            sys.exit("Input file(s) not found:\n  " + "\n  ".join(missing))

    subset = sweep.get("subset") or {}
    if subset.get("n_per_label") is not None:
        n_per_label = int(subset["n_per_label"])
        seed = int(subset.get("seed", 42))
        where = Path(
            subset.get("dir") or out_root / f"subset_n{n_per_label}_seed{seed}"
        ).resolve()
        template["data"] = (
            subset_paths(template["data"], where)
            if args.dry_run
            else subset_tables(template["data"], where, n_per_label, seed)
        )

    out_root.mkdir(parents=True, exist_ok=True)
    # PBS sends SIGTERM at walltime and on qdel.
    signal.signal(signal.SIGTERM, _stop_running)
    signal.signal(signal.SIGINT, _stop_running)
    statuses = run_all(
        combos,
        template,
        icc_group,
        args,
        out_root / "sweep_transforms_status.json",
    )
    if args.dry_run:
        print(
            "\nDry run: configs written; status and summary left as they were."
        )
        return

    finish_failed = final_stages(combos, out_root, template, args)
    failed = [s for s in statuses if s["status"] in UNFINISHED]
    if failed:
        print(
            str(len(failed))
            + " combination(s) did not finish: "
            + str([s["combo"] for s in failed])
        )
    if finish_failed:
        print(
            "Pruning or the final fit failed for "
            + str(finish_failed)
            + "; see sweep_finish.log in each folder."
        )
    if failed or finish_failed:
        sys.exit(1)


def _run_child(args: argparse.Namespace) -> int:
    """One combination's stages, in this process. Returns the exit code."""
    import run_cv_to_final_eval as pipeline

    level = str(args.log_level or "INFO")
    pipeline.setup_logging(level)
    numeric = getattr(logging, level.upper(), logging.INFO)
    for name in ("steps", "prune-iterate", LOGGER.name):
        logging.getLogger(name).setLevel(numeric)
    pipeline.LOGGER.setLevel(numeric)

    cores = args.cores or DEFAULT_CORES
    if args.run_combo:
        run_stages(args.run_combo.resolve(), cores)
        return 0
    return 1 if finish_combo(args.finish_combo.resolve(), cores) else 0


if __name__ == "__main__":
    main()
