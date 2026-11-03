#!/usr/bin/env python3
"""Run the pipeline once per feature-transform combination, then summarise.

`features.transform` is a single value, and `Dataset.load` reads its
feature sets from `manifest.json`, so two transforms cannot share one
prepared dataset -- the second `prepare` rewrites the manifest and
orphans the first. Comparing transforms therefore means one full run per
transform, into its own `base_dir`. This drives that loop.

What is an axis here and what is not:

  transform   relabund / clr / presence. A real axis: one run each.
  icc         `max_icc: null` returns the feature frame UNTOUCHED, so
              report-only ICC is a diagnostic, not a variant. Only a
              numeric `--icc-max` forks the grid, and each value given
              adds one run per transform.
  shadow      Not an axis. Shadows are added as a companion `<set>_shad`
              beside the untouched clean set, both trained in the same
              run, so `--shadow` is one flag applied to every combo.

So the grid is `len(transforms) x (1 + len(icc_max))` runs.

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

    # re-read what is on disk without running anything
    ... --summary-only

The summary is written whatever happened, so a sweep where one combo
failed still produces a comparison of the ones that worked.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import yaml

from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.dataset import feature_set_name
from microbiome_ml.wrangle.profiles import canonical_transform
from microbiome_ml.wrangle.shadow import SHADOW_SUFFIX

LOGGER = logging.getLogger("sweep-transforms")

ROOT = Path(__file__).resolve().parents[1]

# Per-combo tables written by `pipeline_steps collate`, concatenated here.
HEADLINE = "cv_sweep_headline.tsv"
ALL_MODELS = "cv_sweep_all.tsv"
STATUS = "sweep_status.tsv"

# Ranked on the first of these present. Pooled is preferred because
# unequal LOGO folds make the mean of per-fold R2s a weighted average of
# nothing in particular.
RANK_ON = ("best_grouped_pooled_r2", "best_grouped_r2")

# Identifies a run in every summary table.
COMBO_KEYS = ["combo", "transform", "max_icc", "shadow"]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
class Combo:
    """One run: a transform, optionally an ICC cutoff, its own directory."""

    def __init__(
        self,
        transform: str,
        max_icc: Optional[float],
        shadow: bool,
        out_root: Path,
    ) -> None:
        self.transform = canonical_transform(transform)
        self.max_icc = max_icc
        self.shadow = shadow
        parts = [self.transform]
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
) -> List[Combo]:
    """Every transform, with no ICC filter and then one per cutoff."""
    cutoffs: List[Optional[float]] = [None, *[float(v) for v in icc_max]]
    return [
        Combo(transform, cutoff, shadow, out_root)
        for transform in transforms
        for cutoff in cutoffs
    ]


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


def derive_feature_sets(features: Dict[str, Any], combo: Combo) -> List[str]:
    """The set names this combo will actually produce.

    Derived rather than taken from the template, because the template's
    `cv.feature_set` was written for whatever transform it had set. The
    clean set comes first so it leads the summary tables; the shadow set
    follows, and exists only to calibrate the clean one's importances.
    """
    prefix = str(features.get("prefix", "tax"))
    prevalence_by = features.get("prevalence_by") or None
    names: List[str] = []
    for rank in ranks_of(features):
        clean = feature_set_name(prefix, rank, combo.transform, prevalence_by)
        names.append(clean)
        if combo.shadow:
            names.append(f"{clean}_{SHADOW_SUFFIX}")
    return names


def derive_config(
    template: Dict[str, Any], combo: Combo, icc_group: str
) -> Dict[str, Any]:
    """The template with this combo's transform, ICC, shadow and paths."""
    cfg = copy.deepcopy(template)
    features = cfg.setdefault("features", {})
    features["transform"] = combo.transform

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

    cfg.setdefault("cv", {})["feature_set"] = derive_feature_sets(
        features, combo
    )
    return cfg


def write_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
# Snakemake's job-count flags. If `--extra` carries one of these the
# driver leaves its own `--cores` off, because on a cluster run the
# budget is `--jobs` and passing both invites the two to disagree.
JOB_FLAGS = ("--cores", "-c", "--jobs", "-j")


def command_for(
    combo: Combo, runner: str, cores: int, pixi: bool, extra: List[str]
) -> List[str]:
    """The shell command that runs one combo.

    `extra` is appended verbatim and never shell-interpreted, so a submit
    command with spaces (`--cluster-generic-submit-cmd "mqsub ... --"`)
    survives as one argument instead of being re-split.
    """
    if runner == "snakemake":
        cmd = [
            "snakemake",
            "-s",
            "workflow/Snakefile",
            "--configfile",
            combo.config_path.as_posix(),
        ]
        if any(arg.split("=", 1)[0] in JOB_FLAGS for arg in extra):
            LOGGER.info(
                "--extra sets the job count, so --cores %d is not passed",
                cores,
            )
        else:
            cmd += ["--cores", str(cores)]
    else:
        cmd = [
            "python",
            "scripts/run_cv_to_final_eval.py",
            "--config",
            combo.config_path.as_posix(),
        ]
    if pixi:
        cmd = ["pixi", "run", *cmd]
    return [*cmd, *extra]


def run_combo(
    combo: Combo,
    runner: str,
    cores: int,
    pixi: bool,
    extra: List[str],
    dry_run: bool,
) -> Dict[str, Any]:
    """Run one combo; returns its status row. Never raises on failure."""
    cmd = command_for(combo, runner, cores, pixi, extra)
    LOGGER.info("[%s] %s", combo.id, " ".join(cmd))
    row = {**combo.row(), "command": " ".join(cmd)}
    if dry_run:
        return {**row, "status": "dry-run", "returncode": None, "seconds": 0.0}

    log_path = combo.dir / "sweep_transforms.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
            )
            code = completed.returncode
        except OSError as exc:
            handle.write("\ncould not launch: " + str(exc) + "\n")
            code = -1
    elapsed = time.time() - started
    status = "ok" if code == 0 else "failed"
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
def _read_tsv(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pl.read_csv(path, separator="\t", infer_schema_length=10000)
    except Exception as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return None
    return frame if frame.height else None


def gather(combos: List[Combo], filename: str) -> Optional[pl.DataFrame]:
    """Concatenate one per-combo table across combos, stamped with the combo.

    `diagonal_relaxed`, because a combo that failed early or ran under an
    older build can be missing columns the others have; dropping to the
    intersection would quietly lose the metric being compared.
    """
    frames = []
    for combo in combos:
        frame = _read_tsv(combo.dir / filename)
        if frame is None:
            LOGGER.warning("[%s] no %s", combo.id, filename)
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
        add(
            "  {:<24} transform={:<9} icc filter={:<14} shadow={}".format(
                combo.id,
                combo.transform,
                icc,
                "on" if combo.shadow else "off",
            )
        )
    add("")

    if "status" in status.columns:
        bad = status.filter(pl.col("status") == "failed")
        if bad.height:
            add("!! " + str(bad.height) + " combination(s) did not finish:")
            for row in bad.iter_rows(named=True):
                add(
                    "   {:<24} rc={}".format(
                        row.get("combo"), row.get("returncode")
                    )
                )
            add("   Their rows are absent below, not zero. Read their logs.")
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
    frame = headline.with_columns(
        pl.col(rank_on).cast(pl.Float64, strict=False)
    )
    for keys, block in frame.group_by(group_keys, maintain_order=True):
        add("-" * 78)
        add("  ".join(k + "=" + str(v) for k, v in zip(group_keys, keys)))
        add("-" * 78)
        add(
            "  {:<24} {:>8} {:>8} {:>8}  {:<30} {}".format(
                "combo", "R2", "control", "gap", "feature set", "model"
            )
        )
        ranked = block.sort(rank_on, descending=True, nulls_last=True)
        for i, row in enumerate(ranked.iter_rows(named=True)):
            control = row.get("control_pooled_r2")
            if control is None:
                control = row.get("control_r2")
            add(
                "{} {:<24} {:>8} {:>8} {:>8}  {:<30} {}".format(
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
    add(
        "  - These are validation folds, not the holdout. A transform "
        "winning here is a candidate, not a result."
    )
    add("=" * 78)
    return "\n".join(lines) + "\n"


def summarise(combos: List[Combo], out_root: Path) -> Path:
    """Write the combined tables and the text report. Returns the report."""
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    status_path = out_root / "sweep_transforms_status.json"
    if status_path.exists():
        rows = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        rows = [
            {**c.row(), "status": "unknown", "returncode": None}
            for c in combos
        ]
    status = pl.DataFrame(rows)

    headline = gather(combos, HEADLINE)
    all_models = gather(combos, ALL_MODELS)
    run_status = gather(combos, STATUS)

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

    report = summary_dir / "SUMMARY.txt"
    report.write_text(report_text(headline, status, combos), encoding="utf-8")
    LOGGER.info("Wrote %s", report)
    return report


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="template pipeline YAML"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="root for the per-combination directories and the summary",
    )
    parser.add_argument(
        "--transform",
        nargs="+",
        default=["relabund", "clr", "presence"],
        help="transforms to compare (default: all three)",
    )
    parser.add_argument(
        "--icc-max",
        nargs="*",
        type=float,
        default=[],
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
        "--no-shadow",
        action="store_true",
        help="skip the shadow companion sets",
    )
    parser.add_argument(
        "--runner",
        choices=["snakemake", "script"],
        default="snakemake",
        help="snakemake (default) or run_cv_to_final_eval.py directly",
    )
    parser.add_argument("--cores", type=int, default=8)
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
    parser.add_argument("--log-level", default="INFO")
    # REMAINDER last, so everything after --extra goes to the runner.
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="everything after this is appended to the runner command",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    template = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    icc_group = (
        args.icc_group
        or ((template.get("features") or {}).get("icc") or {}).get("group")
        or "bioproject"
    )
    out_root = args.out.resolve()
    combos = build_grid(
        list(args.transform),
        list(args.icc_max),
        not args.no_shadow,
        out_root,
    )
    LOGGER.info("%d combination(s): %s", len(combos), [c.id for c in combos])

    if args.summary_only:
        print("\nSummary: " + str(summarise(combos, out_root)))
        return

    out_root.mkdir(parents=True, exist_ok=True)
    statuses: List[Dict[str, Any]] = []
    status_path = out_root / "sweep_transforms_status.json"

    for combo in combos:
        cfg = derive_config(template, combo, icc_group)
        write_config(cfg, combo.config_path)
        LOGGER.info(
            "[%s] config -> %s (feature_set=%s)",
            combo.id,
            combo.config_path,
            cfg["cv"]["feature_set"],
        )
        statuses.append(
            run_combo(
                combo,
                args.runner,
                args.cores,
                not args.no_pixi,
                list(args.extra),
                args.dry_run,
            )
        )
        # Written after every combo, so an interrupted sweep still has a
        # record of what ran and --summary-only can use it.
        status_path.write_text(
            json.dumps(statuses, indent=2), encoding="utf-8"
        )
        if args.stop_on_error and statuses[-1]["status"] == "failed":
            LOGGER.error("stopping: %s failed", combo.id)
            break

    report = summarise(combos, out_root)
    failed = [s for s in statuses if s["status"] == "failed"]
    print("\nSummary: " + str(report))
    if failed:
        print(
            str(len(failed))
            + " combination(s) failed: "
            + str([s["combo"] for s in failed])
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
