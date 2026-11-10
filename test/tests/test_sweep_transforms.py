"""The transform sweep: does each combo get the right config and name?

The thing most worth guarding is `cv.feature_set`. A transform's suffix
is not its config value -- `presence` builds `_pa` -- so a hand-written
name silently trains nothing. Deriving it from the package's own
`feature_set_name` is the whole point of the driver, and these pin it.

Second is isolation: every combo writes a prepared dataset, splits and a
saved dataset, and the template's paths are absolute and shared. A combo
that inherits them overwrites its neighbour.
"""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import sweep_tokens  # noqa: E402
import sweep_transforms as sw  # noqa: E402

TEMPLATE = {
    "data": {"metadata": "/abs/meta.csv"},
    "features": {
        "all": False,
        "ranks": ["genus"],
        "prefix": "tax",
        "min_prevalence": 0.01,
        "labelled_only": True,
        "transform": "presence",
        "prevalence_by": "bioproject",
    },
    "cv": {
        "models": ["rf", "xgboost"],
        # Deliberately wrong: `presence` names its sets `_pa`, so this
        # set does not exist. The driver must replace it.
        "feature_set": ["tax_genus_presence_bio"],
    },
    "outputs": {
        "base_dir": "/shared/run",
        "splits_dir": "/shared/run/holdout_splits",
        "save_dataset_path": "/shared/run/dataset",
    },
    "sweep": {"pairs": {"ph": ["bioproject"]}},
}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
def test_three_transforms_is_three_runs(tmp_path):
    grid = sw.build_grid(["relabund", "clr", "presence"], [], True, tmp_path)
    assert [c.id for c in grid] == ["relabund", "clr", "presence"]


def test_each_icc_cutoff_adds_one_run_per_transform(tmp_path):
    grid = sw.build_grid(
        ["relabund", "clr", "presence"], [0.9, 0.75], True, tmp_path
    )
    assert len(grid) == 9
    assert "clr_icc0p9" in [c.id for c in grid]


def test_shadow_is_not_a_grid_axis(tmp_path):
    """Shadows ride inside a run, so they must not multiply it."""
    on = sw.build_grid(["clr"], [], True, tmp_path)
    off = sw.build_grid(["clr"], [], False, tmp_path)
    assert len(on) == len(off) == 1
    assert on[0].id == off[0].id == "clr"


def test_cutoff_does_not_put_a_dot_in_a_directory_name(tmp_path):
    grid = sw.build_grid(["clr"], [0.9], True, tmp_path)
    assert grid[1].id == "clr_icc0p9"
    assert "." not in grid[1].dir.name


def test_transform_aliases_collapse_to_one_directory(tmp_path):
    for alias in ("pa", "binary", "presence_absence"):
        assert sw.Combo(alias, None, True, tmp_path).id == "presence"


def test_an_unknown_transform_is_refused(tmp_path):
    with pytest.raises(ValueError, match="transform must be one of"):
        sw.Combo("logratio", None, True, tmp_path)


# ---------------------------------------------------------------------------
# Feature set names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "transform,expected",
    [
        ("relabund", ["tax_genus_bio", "tax_genus_bio_shad"]),
        ("clr", ["tax_genus_clr_bio", "tax_genus_clr_bio_shad"]),
        ("presence", ["tax_genus_pa_bio", "tax_genus_pa_bio_shad"]),
    ],
)
def test_feature_sets_match_what_prepare_builds(transform, expected, tmp_path):
    combo = sw.Combo(transform, None, True, tmp_path)
    assert sw.derive_feature_sets(TEMPLATE["features"], combo) == expected


def test_no_shadow_means_the_clean_set_only(tmp_path):
    combo = sw.Combo("presence", None, False, tmp_path)
    assert sw.derive_feature_sets(TEMPLATE["features"], combo) == [
        "tax_genus_pa_bio"
    ]


def test_no_prevalence_by_means_no_bio_suffix(tmp_path):
    combo = sw.Combo("clr", None, False, tmp_path)
    features = {"all": False, "ranks": ["genus"]}
    assert sw.derive_feature_sets(features, combo) == ["tax_genus_clr"]


def test_ranks_mirror_add_taxonomic_features(tmp_path):
    """`all` wins over `ranks`, and a missing `ranks` means class only."""
    assert sw.ranks_of({"all": True}) == [
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    assert sw.ranks_of({"all": False}) == ["class"]
    assert sw.ranks_of({"all": False, "ranks": ["Genus"]}) == ["genus"]


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------
def test_the_templates_feature_set_is_replaced_not_trusted(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject")
    assert cfg["cv"]["feature_set"] == [
        "tax_genus_clr_bio",
        "tax_genus_clr_bio_shad",
    ]
    assert "tax_genus_presence_bio" not in cfg["cv"]["feature_set"]


def test_every_output_path_moves_inside_the_combo_directory(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject")
    own = combo.dir.as_posix()
    for key in ("base_dir", "splits_dir", "save_dataset_path"):
        assert cfg["outputs"][key].startswith(own), key


def test_a_template_without_a_saved_dataset_does_not_gain_one(tmp_path):
    template = {"features": {"all": False, "ranks": ["genus"]}, "outputs": {}}
    cfg = sw.derive_config(
        template, sw.Combo("clr", None, False, tmp_path), "bioproject"
    )
    assert "save_dataset_path" not in cfg["outputs"]


def test_the_template_is_not_mutated(tmp_path):
    """Combos are derived in a loop; a shallow copy would leak."""
    for combo in sw.build_grid(["relabund", "clr"], [0.9], True, tmp_path):
        sw.derive_config(TEMPLATE, combo, "bioproject")
    assert TEMPLATE["features"]["transform"] == "presence"
    assert TEMPLATE["cv"]["feature_set"] == ["tax_genus_presence_bio"]
    assert "icc" not in TEMPLATE["features"]
    assert TEMPLATE["outputs"]["base_dir"] == "/shared/run"


def test_icc_reports_always_and_filters_only_with_a_cutoff(tmp_path):
    plain = sw.derive_config(
        TEMPLATE, sw.Combo("clr", None, True, tmp_path), "bioproject"
    )
    assert plain["features"]["icc"] == {
        "enabled": True,
        "group": "bioproject",
        "report": True,
        "max_icc": None,
    }
    filtered = sw.derive_config(
        TEMPLATE, sw.Combo("clr", 0.9, True, tmp_path), "bioproject"
    )
    assert filtered["features"]["icc"]["max_icc"] == 0.9


def test_unrelated_config_survives(tmp_path):
    cfg = sw.derive_config(
        TEMPLATE, sw.Combo("clr", None, True, tmp_path), "bioproject"
    )
    assert cfg["features"]["min_prevalence"] == 0.01
    assert cfg["features"]["labelled_only"] is True
    assert cfg["sweep"]["pairs"] == {"ph": ["bioproject"]}
    assert cfg["data"]["metadata"] == "/abs/meta.csv"


def test_shadow_count_and_seed_come_from_the_template(tmp_path):
    template = dict(TEMPLATE)
    template["features"] = {
        **TEMPLATE["features"],
        "shadow": {"n": 500, "seed": 7},
    }
    cfg = sw.derive_config(
        template, sw.Combo("clr", None, True, tmp_path), "bioproject"
    )
    assert cfg["features"]["shadow"] == {
        "enabled": True,
        "n": 500,
        "seed": 7,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def test_snakemake_command_points_at_this_combos_config(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cmd = sw.command_for(combo, 16, True, [])
    assert cmd[:2] == ["pixi", "run"]
    assert combo.config_path.as_posix() in cmd
    assert "--cores" in cmd and "16" in cmd


def test_extra_is_passed_through_last(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cmd = sw.command_for(combo, 4, False, ["--rerun-incomplete", "-n"])
    assert cmd[0] == "snakemake"
    assert cmd[-2:] == ["--rerun-incomplete", "-n"]


def test_extra_setting_jobs_suppresses_cores(tmp_path):
    """On a cluster the budget is --jobs; two budgets would disagree."""
    combo = sw.Combo("clr", None, True, tmp_path)
    for extra in (["--jobs", "20"], ["--jobs=20"], ["-j", "20"]):
        cmd = sw.command_for(combo, 24, False, extra)
        assert "--cores" not in cmd, extra


def test_extra_that_is_not_a_job_flag_keeps_cores(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cmd = sw.command_for(combo, 24, False, ["--rerun-incomplete"])
    assert "--cores" in cmd and "24" in cmd


def test_a_submit_command_with_spaces_stays_one_argument(tmp_path):
    """subprocess gets a list, so the quoted submit command must not split."""
    combo = sw.Combo("clr", None, True, tmp_path)
    submit = "mqsub -t {threads} -m 64 --hours 12 --"
    cmd = sw.command_for(
        combo,
        24,
        False,
        [
            "--executor",
            "cluster-generic",
            "--cluster-generic-submit-cmd",
            submit,
            "--jobs",
            "20",
        ],
    )
    assert submit in cmd
    assert cmd[cmd.index("--cluster-generic-submit-cmd") + 1] == submit


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------
def _headline(label, grouped, control, feature_set):
    return pl.DataFrame(
        [
            {
                "holdout_grouping": "bioproject",
                "label": label,
                "control_scheme": "random",
                "control_feature_set": feature_set,
                "control_model": "rf",
                "control_r2": control,
                "control_pooled_r2": control,
                "best_grouped_scheme": "bioproject_logo",
                "best_grouped_feature_set": feature_set,
                "best_grouped_model": "rf",
                "best_grouped_r2": grouped,
                "best_grouped_pooled_r2": grouped,
                "r2_gap": round(control - grouped, 4),
            }
        ]
    )


def _sweep(tmp_path, scores, shadow_won=()):
    """Lay out a finished sweep on disk; returns its combos."""
    combos = sw.build_grid(list(scores), [], True, tmp_path)
    feat = {
        "relabund": "tax_genus_bio",
        "clr": "tax_genus_clr_bio",
        "presence": "tax_genus_pa_bio",
    }
    for combo in combos:
        combo.dir.mkdir(parents=True, exist_ok=True)
        if scores[combo.id] is None:  # the combo failed; wrote nothing
            continue
        grouped, control = scores[combo.id]
        name = feat[combo.id]
        if combo.id in shadow_won:
            name = f"{name}_{sw.SHADOW_SUFFIX}"
        _headline("ph", grouped, control, name).write_csv(
            combo.dir / sw.HEADLINE, separator="\t"
        )
    return combos


def test_summary_stamps_the_combination_onto_every_row(tmp_path):
    combos = _sweep(
        tmp_path,
        {"relabund": (0.12, 0.55), "clr": (0.21, 0.58)},
    )
    sw.summarise(combos, tmp_path)
    table = pl.read_csv(tmp_path / "summary" / "headline.tsv", separator="\t")
    assert table.columns[: len(sw.COMBO_KEYS)] == sw.COMBO_KEYS
    assert sorted(table["combo"].to_list()) == ["clr", "relabund"]
    assert table.height == 2


def test_a_failed_combination_is_absent_not_zero(tmp_path):
    combos = _sweep(
        tmp_path,
        {"relabund": (0.12, 0.55), "clr": None, "presence": (0.18, 0.52)},
    )
    (tmp_path / "sweep_transforms_status.json").write_text(
        json.dumps(
            [
                {
                    **c.row(),
                    "status": "ok" if c.id != "clr" else "failed",
                    "returncode": 0 if c.id != "clr" else 1,
                }
                for c in combos
            ]
        ),
        encoding="utf-8",
    )
    report = sw.summarise(combos, tmp_path)
    table = pl.read_csv(tmp_path / "summary" / "headline.tsv", separator="\t")
    assert "clr" not in table["combo"].to_list()
    text = report.read_text(encoding="utf-8")
    assert "did not finish" in text
    assert "absent below, not zero" in text


def _ranked_combos(text):
    """Combo names in the order one label block lists them.

    Parsed from the body rows only: the grid listing at the top of the
    report is indented the same way but carries `transform=`.
    """
    names = []
    for line in text.splitlines():
        if not line.startswith(("* ", "  ")) or "transform=" in line:
            continue
        fields = line[2:].split()
        if fields and fields[0] not in ("combo", "=", "-"):
            names.append(fields[0])
    return names


def test_the_winner_is_marked_and_ranked_first(tmp_path):
    combos = _sweep(
        tmp_path,
        {
            "relabund": (0.12, 0.55),
            "clr": (0.21, 0.58),
            "presence": (0.18, 0.52),
        },
    )
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert _ranked_combos(text) == ["clr", "presence", "relabund"]
    assert any(line.startswith("* clr") for line in text.splitlines())


def test_a_winning_shadow_set_is_noted_not_alarmed_at(tmp_path):
    """A _shad set is a SUPERSET of its clean twin, so this is expected.

    The report used to call it a finding ("shuffled columns outscored
    the real ones"), which is wrong: the shadow set still holds every
    real column, so the two scoring alike says nothing about whether the
    model used the noise. That question lives in the importances.
    """
    combos = _sweep(
        tmp_path,
        {"relabund": (0.02, 0.41), "clr": (0.01, 0.40)},
        shadow_won=("relabund",),
    )
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "edged ahead" in text
    assert "tax_genus_bio_shad" in text
    assert "Expected:" in text
    assert "shadow_floor.tsv" in text
    # The claim that was wrong must not come back.
    assert "outscored the real ones" not in text
    assert "A SHADOW SET WON" not in text


def test_a_stamp_column_already_in_the_table_does_not_break_it(tmp_path):
    """Upstream owning a `combo` column must not take the summary down."""
    combos = sw.build_grid(["clr"], [], True, tmp_path)
    combos[0].dir.mkdir(parents=True, exist_ok=True)
    _headline("ph", 0.2, 0.5, "tax_genus_clr_bio").with_columns(
        pl.lit("from-upstream").alias("combo")
    ).write_csv(combos[0].dir / sw.HEADLINE, separator="\t")
    sw.summarise(combos, tmp_path)
    table = pl.read_csv(tmp_path / "summary" / "headline.tsv", separator="\t")
    assert table["combo"].to_list() == ["clr"]


def test_an_empty_sweep_reports_rather_than_crashes(tmp_path):
    combos = sw.build_grid(["clr"], [], True, tmp_path)
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "Nothing to compare" in text


def test_missing_status_file_is_not_reported_as_failure(tmp_path):
    combos = _sweep(tmp_path, {"clr": (0.21, 0.58)})
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "did not finish" not in text
    assert "exit codes are unknown" in text


# ---------------------------------------------------------------------------
# the prevalence axis
# ---------------------------------------------------------------------------
def test_prevalence_multiplies_the_grid(tmp_path):
    grid = sw.build_grid(
        ["relabund", "clr", "presence"], [], True, tmp_path, [0.01, 0.05]
    )
    assert len(grid) == 6
    assert len({c.dir for c in grid}) == 6


def test_prevalence_and_icc_multiply_together(tmp_path):
    grid = sw.build_grid(
        ["relabund", "clr", "presence"],
        [0.9, 0.75],
        True,
        tmp_path,
        [0.01, 0.05, 0.1],
    )
    # 3 transforms x 3 thresholds x (no filter + 2 cutoffs)
    assert len(grid) == 27
    assert len({c.id for c in grid}) == 27


def test_no_prevalence_axis_keeps_the_old_ids(tmp_path):
    """An opt-in axis must not rename the directories of old sweeps."""
    assert [c.id for c in sw.build_grid(["clr"], [], True, tmp_path)] == [
        "clr"
    ]
    assert [c.id for c in sw.build_grid(["clr"], [], True, tmp_path, [])] == [
        "clr"
    ]


def test_a_dot_never_reaches_a_directory_name(tmp_path):
    grid = sw.build_grid(["clr"], [0.9], True, tmp_path, [0.01])
    assert [c.id for c in grid] == ["clr_prev0p01", "clr_prev0p01_icc0p9"]


def test_prevalence_reaches_the_derived_config(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path, 0.05)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject")
    assert cfg["features"]["min_prevalence"] == 0.05


def test_no_prevalence_leaves_the_template_value_alone(tmp_path):
    """The template's own threshold is the thing under test otherwise."""
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject")
    assert cfg["features"].get("min_prevalence") == TEMPLATE["features"].get(
        "min_prevalence"
    )


# ---------------------------------------------------------------------------
# how far each combination runs
# ---------------------------------------------------------------------------
def test_the_sweep_stops_where_the_template_stops(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject")
    assert cfg.get("holdout_evaluation", {}).get("enabled") is not True
    assert "final_fit" not in cfg or not cfg["final_fit"].get("enabled")


def test_holdout_flag_enables_the_final_test(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject", holdout=True)
    assert cfg["holdout_evaluation"]["enabled"] is True


def test_final_fit_flag_fills_in_its_defaults(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(TEMPLATE, combo, "bioproject", final_fit=True)
    assert cfg["final_fit"] == {"enabled": True, "oob": True, "top_n": 20}


def test_a_non_mapping_stage_block_is_repaired(tmp_path):
    """`holdout_evaluation: false` is valid YAML and not a mapping."""
    template = copy.deepcopy(TEMPLATE)
    template["holdout_evaluation"] = False
    combo = sw.Combo("clr", None, True, tmp_path)
    cfg = sw.derive_config(template, combo, "bioproject", holdout=True)
    assert cfg["holdout_evaluation"] == {"enabled": True}


def test_enabling_a_stage_does_not_mutate_the_template(tmp_path):
    template = copy.deepcopy(TEMPLATE)
    template["holdout_evaluation"] = {"enabled": False}
    for combo in sw.build_grid(["clr", "presence"], [], True, tmp_path):
        sw.derive_config(template, combo, "bioproject", holdout=True)
    assert template["holdout_evaluation"]["enabled"] is False


# ---------------------------------------------------------------------------
# the holdout block in the summary
# ---------------------------------------------------------------------------
def _holdout_table(label, rows):
    """`holdout_sweep_summary.tsv` as collate writes it."""
    return pl.DataFrame(
        {
            "holdout_grouping": ["bioproject"] * len(rows),
            "label": [label] * len(rows),
            "scheme": ["bioproject_logo"] * len(rows),
            "feature_set": ["tax_genus_bio"] * len(rows),
            "r2": list(rows),
            "mae": [0.4] * len(rows),
            "pcc": [0.6] * len(rows),
            "n_test": [120] * len(rows),
        }
    )


def test_no_holdout_file_is_not_an_error(tmp_path):
    """The normal state of a CV-only sweep."""
    combos = _sweep(tmp_path, {"clr": (0.21, 0.58)})
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "HOLDOUT" not in text
    assert "--holdout to settle it" in text
    assert not (tmp_path / "summary" / "holdout.tsv").exists()


def test_the_holdout_block_ranks_the_final_test(tmp_path):
    combos = _sweep(tmp_path, {"clr": (0.21, 0.58), "presence": (0.18, 0.52)})
    for combo, r2 in zip(combos, (0.11, 0.33)):
        _holdout_table("ph", [r2]).write_csv(
            combo.dir / sw.HOLDOUT, separator="\t"
        )
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "HOLDOUT -- THE FINAL TEST" in text
    # presence scored higher on the holdout even though clr won CV.
    assert "* presence" in text
    assert (tmp_path / "summary" / "holdout.tsv").exists()


def test_the_holdout_table_is_stamped_with_the_combination(tmp_path):
    combos = _sweep(tmp_path, {"clr": (0.21, 0.58)})
    _holdout_table("ph", [0.3]).write_csv(
        combos[0].dir / sw.HOLDOUT, separator="\t"
    )
    sw.summarise(combos, tmp_path)
    table = pl.read_csv(tmp_path / "summary" / "holdout.tsv", separator="\t")
    assert table.columns[: len(sw.COMBO_KEYS)] == sw.COMBO_KEYS
    assert table["combo"].to_list() == ["clr"]


def test_the_caveat_changes_once_the_holdout_exists(tmp_path):
    combos = _sweep(tmp_path, {"clr": (0.21, 0.58)})
    _holdout_table("ph", [0.3]).write_csv(
        combos[0].dir / sw.HOLDOUT, separator="\t"
    )
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "--holdout to settle it" not in text
    assert "that one is the result" in text


# ---------------------------------------------------------------------------
# the manager: running combinations in parallel
# ---------------------------------------------------------------------------
def _args(**kw):
    base = dict(
        cores=8,
        models=None,
        rerun=False,
        no_pixi=False,
        extra=[],
        dry_run=False,
        stop_on_error=False,
        max_workers=1,
        holdout=False,
        final_fit=False,
        runner="snakemake",
        log_level="INFO",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_worker(record, fail=(), raise_on=()):
    """A run_combo stand-in that records how many ran at once."""
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def worker(combo, cores, pixi, extra, dry_run):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        record.append(combo.id)
        if combo.id in raise_on:
            raise RuntimeError("boom")
        bad = combo.id in fail
        return {
            **combo.row(),
            "status": "failed" if bad else "ok",
            "returncode": 1 if bad else 0,
        }

    return worker, state


def test_combinations_run_concurrently(tmp_path, monkeypatch):
    ran = []
    worker, state = _fake_worker(ran)
    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(
        ["relabund", "clr", "presence"], [], True, tmp_path, [0.01, 0.05]
    )
    rows = sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(max_workers=3),
        tmp_path / "status.json",
    )
    assert len(rows) == 6
    assert state["max"] > 1, "nothing actually overlapped"
    assert state["max"] <= 3, "more ran at once than --max-workers allows"
    assert sorted(ran) == sorted(c.id for c in combos)


def test_one_worker_is_still_serial(tmp_path, monkeypatch):
    """The default must behave exactly as the old loop did."""
    worker, state = _fake_worker([])
    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(["relabund", "clr"], [], True, tmp_path)
    sw.run_all(
        combos, TEMPLATE, "bioproject", _args(), tmp_path / "status.json"
    )
    assert state["max"] == 1


def test_the_status_file_is_written_under_a_lock(tmp_path, monkeypatch):
    worker, _ = _fake_worker([])
    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(
        ["relabund", "clr", "presence"], [], True, tmp_path, [0.01, 0.05]
    )
    status = tmp_path / "status.json"
    sw.run_all(combos, TEMPLATE, "bioproject", _args(max_workers=4), status)
    on_disk = json.loads(status.read_text(encoding="utf-8"))
    assert {r["combo"] for r in on_disk} == {c.id for c in combos}


def test_every_combination_gets_its_config_before_anything_runs(
    tmp_path, monkeypatch
):
    seen = []

    def worker(combo, *a, **k):
        # Every config must already exist, or a parallel worker could
        # start before its own file was written.
        seen.append(all(c.config_path.exists() for c in combos))
        return {**combo.row(), "status": "ok", "returncode": 0}

    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(["relabund", "clr"], [], True, tmp_path, [0.01])
    sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(max_workers=2),
        tmp_path / "status.json",
    )
    assert all(seen)


def test_a_worker_that_raises_is_recorded_not_lost(tmp_path, monkeypatch):
    worker, _ = _fake_worker([], raise_on=("clr",))
    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(["relabund", "clr"], [], True, tmp_path)
    rows = sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(max_workers=2),
        tmp_path / "status.json",
    )
    assert len(rows) == 2
    assert [r["status"] for r in rows if r["combo"] == "clr"] == ["failed"]


def test_stop_on_error_stops_a_parallel_sweep(tmp_path, monkeypatch):
    ran = []
    worker, _ = _fake_worker(ran, fail=("relabund_prev0p01",))
    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(
        ["relabund", "clr", "presence"], [], True, tmp_path, [0.01, 0.05]
    )
    rows = sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(max_workers=2, stop_on_error=True),
        tmp_path / "status.json",
    )
    assert len(ran) < len(combos)
    assert any(r["status"] == "failed" for r in rows)
    # Every combination is accounted for: what ran, and what never did.
    assert {r["combo"] for r in rows} == {c.id for c in combos}
    assert sum(r["status"] == "not run" for r in rows) == len(combos) - len(
        ran
    )


def test_stop_on_error_starts_nothing_more_when_serial(tmp_path, monkeypatch):
    ran = []
    worker, _ = _fake_worker(ran, fail=("relabund",))
    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(["relabund", "clr", "presence"], [], True, tmp_path)
    sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(stop_on_error=True),
        tmp_path / "status.json",
    )
    assert ran == ["relabund"]


def test_a_combo_already_running_at_the_stop_is_still_recorded(
    tmp_path, monkeypatch
):
    """It runs to its end; a failure there must not vanish from the record."""
    release = threading.Event()

    def worker(combo, cores, pixi, extra, dry_run):
        if combo.id == "relabund":
            release.wait(10)
            return {**combo.row(), "status": "failed", "returncode": 1}
        release.set()
        return {**combo.row(), "status": "failed", "returncode": 1}

    monkeypatch.setattr(sw, "run_combo", worker)
    combos = sw.build_grid(["relabund", "clr", "presence"], [], True, tmp_path)
    sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(max_workers=2, stop_on_error=True),
        tmp_path / "status.json",
    )
    on_disk = {
        r["combo"]: r["status"]
        for r in json.loads((tmp_path / "status.json").read_text("utf-8"))
    }
    assert on_disk == {
        "relabund": "failed",
        "clr": "failed",
        "presence": "not run",
    }


def test_each_combination_gets_its_own_snakemake_workdir(tmp_path):
    """Without this, parallel combinations fight over one .snakemake lock."""
    combo = sw.Combo("clr", None, True, tmp_path, 0.05)
    cmd = sw.command_for(combo, 16, True, [])
    assert "--directory" in cmd
    assert cmd[cmd.index("--directory") + 1] == combo.dir.as_posix()
    # The Snakefile path can no longer be relative to the working dir.
    assert Path(cmd[cmd.index("-s") + 1]).is_absolute()


# ---------------------------------------------------------------------------
# paths survive --directory
# ---------------------------------------------------------------------------
def test_relative_input_paths_become_absolute(tmp_path, monkeypatch):
    """Snakemake runs inside the combo directory, so these would not resolve."""
    monkeypatch.chdir(tmp_path)
    template = copy.deepcopy(TEMPLATE)
    template["data"] = {
        "metadata": "in/meta.csv",
        "study_titles": None,
        "labels": {"ph": "in/ph.csv"},
    }
    template["groupings"] = {"file": "in/groups.csv"}
    template["cv"]["param_path"] = "parameters.yaml"
    cfg = sw.derive_config(
        template, sw.Combo("clr", None, True, tmp_path), "bioproject"
    )
    launch = Path.cwd().as_posix()
    assert cfg["data"]["metadata"] == f"{launch}/in/meta.csv"
    assert cfg["data"]["labels"]["ph"] == f"{launch}/in/ph.csv"
    assert cfg["data"]["study_titles"] is None
    assert cfg["groupings"]["file"] == f"{launch}/in/groups.csv"
    assert cfg["cv"]["param_path"] == f"{launch}/parameters.yaml"


def test_the_default_param_path_is_made_absolute_too(tmp_path, monkeypatch):
    """Absent from the template, the steps fall back to a relative name."""
    monkeypatch.chdir(tmp_path)
    cfg = sw.derive_config(
        {"features": {"all": False, "ranks": ["genus"]}},
        sw.Combo("clr", None, True, tmp_path),
        "bioproject",
    )
    assert (
        cfg["cv"]["param_path"] == f"{Path.cwd().as_posix()}/parameters.yaml"
    )


def test_absolute_paths_are_left_alone(tmp_path):
    template = copy.deepcopy(TEMPLATE)
    template["cv"]["param_path"] = "/home/me/parameters.yaml"
    cfg = sw.derive_config(
        template, sw.Combo("clr", None, True, tmp_path), "bioproject"
    )
    assert cfg["data"]["metadata"] == "/abs/meta.csv"
    assert cfg["cv"]["param_path"] == "/home/me/parameters.yaml"


# ---------------------------------------------------------------------------
# which ranks a combo trains
# ---------------------------------------------------------------------------
def test_template_feature_set_narrows_the_ranks(tmp_path):
    """`all: true` plus a named set builds one rank in a normal run."""
    template = {
        "features": {"all": True},
        "cv": {"feature_set": ["tax_genus_bio"]},
    }
    cfg = sw.derive_config(
        template, sw.Combo("clr", None, False, tmp_path), "bioproject"
    )
    assert cfg["cv"]["feature_set"] == ["tax_genus_clr"]


def test_a_name_that_never_existed_still_gives_its_rank(tmp_path):
    """`tax_genus_presence_bio` is not a real set, but it does say genus."""
    combo = sw.Combo("clr", None, False, tmp_path)
    features = {"all": True, "prevalence_by": "bioproject"}
    assert sw.derive_feature_sets(
        features, combo, ["tax_genus_presence_bio"]
    ) == ["tax_genus_clr_bio"]


def test_a_single_string_feature_set_is_accepted(tmp_path):
    combo = sw.Combo("clr", None, False, tmp_path)
    assert sw.derive_feature_sets({"all": True}, combo, "tax_family") == [
        "tax_family_clr"
    ]


def test_a_feature_set_naming_no_rank_falls_back_to_the_template(tmp_path):
    combo = sw.Combo("relabund", None, False, tmp_path)
    features = {"all": False, "ranks": ["order"]}
    assert sw.derive_feature_sets(features, combo, ["nonsense"]) == [
        "tax_order"
    ]


def test_no_feature_set_with_all_still_means_every_rank(tmp_path):
    combo = sw.Combo("relabund", None, False, tmp_path)
    assert len(sw.derive_feature_sets({"all": True}, combo)) == 6


# ---------------------------------------------------------------------------
# what the summary refuses to believe
# ---------------------------------------------------------------------------
def _write_status(tmp_path, combos, status_of):
    (tmp_path / "sweep_transforms_status.json").write_text(
        json.dumps(
            [
                {
                    **c.row(),
                    "status": status_of.get(c.id, "ok"),
                    "returncode": 0 if c.id not in status_of else 1,
                }
                for c in combos
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_tables_left_by_an_earlier_run_are_not_read(tmp_path, status):
    """The combo did not finish THIS time; its old file must not rank."""
    combos = _sweep(tmp_path, {"relabund": (0.12, 0.55), "clr": (0.99, 0.99)})
    clr = next(c for c in combos if c.id == "clr")
    _holdout_table("ph", [0.9]).write_csv(clr.dir / sw.HOLDOUT, separator="\t")
    _write_status(tmp_path, combos, {"clr": status})
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    table = pl.read_csv(tmp_path / "summary" / "headline.tsv", separator="\t")
    assert table["combo"].to_list() == ["relabund"]
    assert not (tmp_path / "summary" / "holdout.tsv").exists()
    assert "did not finish" in text
    assert any(
        line.split()[:2] == ["clr", status] for line in text.splitlines()
    )
    assert any(line.startswith("* relabund") for line in text.splitlines())


def test_a_nan_score_does_not_take_the_star(tmp_path):
    """Polars sorts NaN above every number."""
    combos = _sweep(
        tmp_path, {"relabund": (0.12, 0.55), "clr": (float("nan"), 0.5)}
    )
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert any(line.startswith("* relabund") for line in text.splitlines())
    assert not any(line.startswith("* clr") for line in text.splitlines())


def test_a_nan_holdout_score_does_not_take_the_star(tmp_path):
    combos = _sweep(tmp_path, {"clr": (0.21, 0.58), "presence": (0.18, 0.52)})
    for combo, r2 in zip(combos, (float("nan"), 0.33)):
        _holdout_table("ph", [r2]).write_csv(
            combo.dir / sw.HOLDOUT, separator="\t"
        )
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    block = text.split("HOLDOUT -- THE FINAL TEST")[1]
    assert any(line.startswith("* presence") for line in block.splitlines())
    assert not any(line.startswith("* clr") for line in block.splitlines())


# ---------------------------------------------------------------------------
# being stopped, and dry runs
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_stop(monkeypatch):
    """Reset the stop flag, shorten the grace period, restore the handlers."""
    term = signal.getsignal(signal.SIGTERM)
    interrupt = signal.getsignal(signal.SIGINT)
    sw.STOPPING.clear()
    monkeypatch.setattr(sw, "KILL_AFTER", 2.0)
    yield
    sw.STOPPING.clear()
    signal.signal(signal.SIGTERM, term)
    signal.signal(signal.SIGINT, interrupt)


SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_stopping_ends_the_running_combo_and_starts_no_more(
    tmp_path, monkeypatch, clean_stop
):
    monkeypatch.setattr(sw, "command_for", lambda *a, **k: list(SLEEP))
    combos = sw.build_grid(["relabund", "clr"], [], True, tmp_path)
    rows = []

    def drive():
        rows.extend(
            sw.run_all(
                combos,
                TEMPLATE,
                "bioproject",
                _args(),
                tmp_path / "status.json",
            )
        )

    thread = threading.Thread(target=drive)
    thread.start()
    deadline = time.time() + 30
    while not sw.RUNNING and time.time() < deadline:
        time.sleep(0.05)
    assert sw.RUNNING, "the combination never started"
    sw._stop_running(signal.SIGTERM, None)
    thread.join(timeout=30)
    assert not thread.is_alive(), "the sweep did not wind down"

    assert [r["status"] for r in rows] == ["interrupted", "interrupted"]
    assert rows[1]["returncode"] is None
    # The second combination never launched, so it never opened a log.
    assert (combos[0].dir / "sweep_transforms.log").exists()
    assert not (combos[1].dir / "sweep_transforms.log").exists()
    assert not sw.RUNNING


@pytest.mark.skipif(
    sys.platform == "win32", reason="terminate is already a kill on Windows"
)
def test_the_kill_timer_spares_a_process_started_after_the_signal(
    monkeypatch, clean_stop
):
    """The timer killed the next test's child (rc=-9) when it read RUNNING
    at fire time rather than at the signal."""
    monkeypatch.setattr(sw, "KILL_AFTER", 0.5)
    # Ignores SIGTERM, so only the timer's kill can end it.
    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    later = None
    try:
        assert stubborn.stdout.readline().strip() == "ready"
        sw.RUNNING.add(stubborn)
        sw._stop_running(signal.SIGTERM, None)
        sw.RUNNING.discard(stubborn)

        later = subprocess.Popen(SLEEP)
        sw.RUNNING.add(later)
        time.sleep(1.5)

        assert stubborn.poll() == -signal.SIGKILL
        assert later.poll() is None, "a process started later was killed"
    finally:
        for proc in (stubborn, later):
            if proc is not None:
                sw.RUNNING.discard(proc)
                proc.kill()
                proc.wait()
                if proc.stdout is not None:
                    proc.stdout.close()


def test_a_combo_that_finishes_as_the_signal_lands_stays_ok(
    tmp_path, monkeypatch, clean_stop
):
    monkeypatch.setattr(
        sw, "command_for", lambda *a, **k: [sys.executable, "-c", "pass"]
    )
    monkeypatch.setattr(
        sw.subprocess.Popen,
        "wait",
        _setting_stop_after(sw.subprocess.Popen.wait),
    )
    combo = sw.Combo("clr", None, True, tmp_path)
    row = sw.run_combo(combo, 1, False, [], False)
    assert row["status"] == "ok"


def _setting_stop_after(wait):
    def patched(self, *args, **kwargs):
        code = wait(self, *args, **kwargs)
        sw.STOPPING.set()
        return code

    return patched


def _main(monkeypatch, tmp_path, *flags):
    config = tmp_path / "template.yaml"
    metadata = tmp_path / "meta.csv"
    params = tmp_path / "parameters.yaml"
    metadata.touch()
    params.touch()
    template = copy.deepcopy(TEMPLATE)
    template["data"] = {"metadata": metadata.as_posix()}
    template["cv"]["param_path"] = params.as_posix()
    config.write_text(yaml.safe_dump(template), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_transforms.py",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "out"),
            "--transform",
            "relabund",
            "clr",
            "--no-pixi",
            *flags,
        ],
    )
    sw.main()


@pytest.mark.skipif(
    sys.platform == "win32", reason="no way to signal yourself on Windows"
)
def test_sigterm_still_writes_the_summary(tmp_path, monkeypatch, clean_stop):
    """Job 27567706 died with exit 143 and left no summary at all."""
    monkeypatch.setattr(sw, "command_for", lambda *a, **k: list(SLEEP))

    def send_when_running():
        deadline = time.time() + 30
        while not sw.RUNNING and time.time() < deadline:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=send_when_running, daemon=True).start()
    with pytest.raises(SystemExit) as stopped:
        _main(monkeypatch, tmp_path)
    assert stopped.value.code == 1
    text = (tmp_path / "out" / "summary" / "SUMMARY.txt").read_text(
        encoding="utf-8"
    )
    assert "did not finish" in text
    assert "interrupted" in text


def test_a_dry_run_leaves_the_record_of_a_real_sweep_alone(
    tmp_path, monkeypatch, clean_stop
):
    out = tmp_path / "out"
    out.mkdir()
    status = out / "sweep_transforms_status.json"
    status.write_text(
        '[{"combo": "clr", "status": "failed"}]', encoding="utf-8"
    )
    (out / "summary").mkdir()
    report = out / "summary" / "SUMMARY.txt"
    report.write_text("the real report", encoding="utf-8")

    _main(monkeypatch, tmp_path, "--dry-run")

    assert (
        json.loads(status.read_text(encoding="utf-8"))[0]["status"] == "failed"
    )
    assert report.read_text(encoding="utf-8") == "the real report"
    assert (out / "clr" / "config.yaml").exists()


def test_a_dry_run_on_a_fresh_tree_creates_no_status(
    tmp_path, monkeypatch, clean_stop
):
    """Otherwise a later --summary-only reads every combo as `dry-run`."""
    _main(monkeypatch, tmp_path, "--dry-run")
    assert not (tmp_path / "out" / "sweep_transforms_status.json").exists()
    assert not (tmp_path / "out" / "summary").exists()


# ---------------------------------------------------------------------------
# a sweep that grows: the grid comes from disk
# ---------------------------------------------------------------------------
def _scored(combo, grouped=0.2):
    _headline("ph", grouped, 0.5, "tax_genus_bio").write_csv(
        combo.dir / sw.HEADLINE, separator="\t"
    )


def test_summary_only_needs_no_axis_flags(tmp_path, monkeypatch, clean_stop):
    """The README's short form used to find nothing after a prevalence sweep."""
    _main(
        monkeypatch, tmp_path, "--dry-run", "--min-prevalence", "0.01", "0.05"
    )
    out = tmp_path / "out"
    for combo in sw.discover_combos(out):
        _scored(combo)
    _main(monkeypatch, tmp_path, "--summary-only")
    table = pl.read_csv(out / "summary" / "headline.tsv", separator="\t")
    assert sorted(table["combo"].to_list()) == [
        "clr_prev0p01",
        "clr_prev0p05",
        "relabund_prev0p01",
        "relabund_prev0p05",
    ]


def test_summary_only_with_nothing_on_disk_keeps_the_old_report(
    tmp_path, monkeypatch, clean_stop
):
    summary = tmp_path / "out" / "summary"
    summary.mkdir(parents=True)
    (summary / "SUMMARY.txt").write_text("the real report", encoding="utf-8")
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path, "--summary-only")
    assert (summary / "SUMMARY.txt").read_text("utf-8") == "the real report"


def test_discovery_rebuilds_every_kind_of_combination(tmp_path):
    """A template's prevalence must not put `prev` into the id."""
    combos = [
        sw.Combo("presence", None, True, tmp_path),
        sw.Combo("clr", 0.9, False, tmp_path, 0.05),
    ]
    for combo in combos:
        sw.write_config(
            sw.derive_config(TEMPLATE, combo, "bioproject"), combo.config_path
        )
    found = {c.id: c for c in sw.discover_combos(tmp_path)}
    assert set(found) == {"presence", "clr_prev0p05_icc0p9"}
    assert found["presence"].min_prevalence is None
    assert found["clr_prev0p05_icc0p9"].min_prevalence == 0.05
    assert found["clr_prev0p05_icc0p9"].max_icc == 0.9
    assert found["clr_prev0p05_icc0p9"].shadow is False


def test_a_duplicate_is_one_combination(tmp_path):
    grid = sw.build_grid(
        ["presence", "pa", "clr"], [0.9, 0.9], True, tmp_path, [0.01, 0.01]
    )
    assert [c.id for c in grid] == [
        "presence_prev0p01",
        "presence_prev0p01_icc0p9",
        "clr_prev0p01",
        "clr_prev0p01_icc0p9",
    ]


def test_a_later_sweep_keeps_the_status_of_earlier_combos(
    tmp_path, monkeypatch
):
    worker, _ = _fake_worker([], fail=("clr",))
    monkeypatch.setattr(sw, "run_combo", worker)
    status = tmp_path / "status.json"
    sw.run_all(
        sw.build_grid(["clr"], [], True, tmp_path),
        TEMPLATE,
        "bioproject",
        _args(),
        status,
    )
    rows = sw.run_all(
        sw.build_grid(["relabund"], [], True, tmp_path),
        TEMPLATE,
        "bioproject",
        _args(),
        status,
    )
    assert [r["combo"] for r in rows] == ["relabund"]
    on_disk = {
        r["combo"]: r["status"] for r in json.loads(status.read_text("utf-8"))
    }
    assert on_disk == {"clr": "failed", "relabund": "ok"}


def test_status_is_marked_before_a_combo_starts(tmp_path, monkeypatch):
    """A driver killed outright records nothing, so this must already be there."""
    status = tmp_path / "status.json"
    seen = []

    def worker(combo, cores, pixi, extra, dry_run):
        seen.append(json.loads(status.read_text("utf-8"))[0]["status"])
        return {**combo.row(), "status": "ok", "returncode": 0}

    monkeypatch.setattr(sw, "run_combo", worker)
    sw.run_all(
        sw.build_grid(["clr"], [], True, tmp_path),
        TEMPLATE,
        "bioproject",
        _args(),
        status,
    )
    assert seen == ["not run"]


def test_a_finished_combo_is_not_launched_again(tmp_path, monkeypatch):
    ran = []
    worker, _ = _fake_worker(ran)
    monkeypatch.setattr(sw, "run_combo", worker)
    status = tmp_path / "status.json"
    first = sw.build_grid(["clr"], [], True, tmp_path)
    sw.run_all(first, TEMPLATE, "bioproject", _args(), status)
    _scored(first[0])

    grown = sw.build_grid(["clr", "presence"], [], True, tmp_path)
    sw.run_all(grown, TEMPLATE, "bioproject", _args(), status)
    assert ran == ["clr", "presence"]

    sw.run_all(first, TEMPLATE, "bioproject", _args(rerun=True), status)
    assert ran == ["clr", "presence", "clr"]


def test_a_finished_combo_runs_again_if_it_failed_or_lost_its_output(
    tmp_path, monkeypatch
):
    ran = []
    worker, _ = _fake_worker(ran, fail=("clr",))
    monkeypatch.setattr(sw, "run_combo", worker)
    status = tmp_path / "status.json"
    grid = sw.build_grid(["clr", "presence"], [], True, tmp_path)
    sw.run_all(grid, TEMPLATE, "bioproject", _args(), status)
    # clr failed; presence is ok but has no headline.
    _scored(grid[0])
    sw.run_all(grid, TEMPLATE, "bioproject", _args(), status)
    assert ran == ["clr", "presence", "clr", "presence"]


# ---------------------------------------------------------------------------
# refusing to mix old results with new settings
# ---------------------------------------------------------------------------
def _built(combo):
    """As if `prepare` had finished for this combination."""
    (combo.dir / "prepared").mkdir(parents=True)
    (combo.dir / "prepared" / "manifest.json").touch()


def _changed(section, key, value):
    template = copy.deepcopy(TEMPLATE)
    template[section][key] = value
    return template


def _sweep_twice(tmp_path, monkeypatch, template, transforms=("clr",), **kw):
    worker, _ = _fake_worker([])
    monkeypatch.setattr(sw, "run_combo", worker)
    status = tmp_path / "status.json"
    first = sw.build_grid(["clr"], [], True, tmp_path)
    sw.run_all(first, TEMPLATE, "bioproject", _args(), status)
    _built(first[0])
    before = first[0].config_path.read_text("utf-8")
    grid = sw.build_grid(list(transforms), [], True, tmp_path)
    try:
        sw.run_all(grid, template, "bioproject", _args(**kw), status)
    finally:
        after = first[0].config_path.read_text("utf-8")
    return before, after


def test_a_changed_setting_on_a_built_combo_is_refused(tmp_path, monkeypatch):
    template = _changed("features", "presence_threshold", 0.5)
    with pytest.raises(SystemExit):
        _sweep_twice(tmp_path, monkeypatch, template)
    # Refused before writing, so the next run can still see the difference.
    config = sw.Combo("clr", None, True, tmp_path).config_path
    assert "presence_threshold" not in config.read_text("utf-8")


def test_rerun_allows_the_change(tmp_path, monkeypatch):
    template = _changed("features", "presence_threshold", 0.5)
    before, after = _sweep_twice(tmp_path, monkeypatch, template, rerun=True)
    assert before != after


def test_a_dry_run_cannot_slip_a_changed_config_past_the_check(
    tmp_path, monkeypatch
):
    template = _changed("features", "presence_threshold", 0.5)
    with pytest.raises(SystemExit):
        _sweep_twice(tmp_path, monkeypatch, template, dry_run=True)


def test_settings_that_change_no_score_are_allowed(tmp_path, monkeypatch):
    template = _changed("cv", "n_jobs", 32)
    template["sweep"] = {"pairs": {"oxy": ["bioproject"]}}
    template["visualise"] = {"enabled": False}
    _sweep_twice(tmp_path, monkeypatch, template)


def test_a_new_combo_cannot_join_under_different_data(tmp_path, monkeypatch):
    """It has no earlier self, but its scores would rank beside clr's."""
    template = _changed("data", "metadata", "/abs/other.csv")
    with pytest.raises(SystemExit):
        _sweep_twice(tmp_path, monkeypatch, template, transforms=("presence",))


def test_a_new_combo_may_differ_on_its_own_axis(tmp_path, monkeypatch):
    template = _changed("features", "min_prevalence", 0.5)
    _sweep_twice(tmp_path, monkeypatch, template, transforms=("presence",))


def test_an_unbuilt_combo_may_change_freely(tmp_path, monkeypatch):
    worker, _ = _fake_worker([])
    monkeypatch.setattr(sw, "run_combo", worker)
    grid = sw.build_grid(["clr"], [], True, tmp_path)
    status = tmp_path / "status.json"
    sw.run_all(grid, TEMPLATE, "bioproject", _args(), status)
    template = _changed("features", "presence_threshold", 0.5)
    sw.run_all(grid, template, "bioproject", _args(), status)
    assert "presence_threshold" in grid[0].config_path.read_text("utf-8")


# ---------------------------------------------------------------------------
# saying so, and failing early
# ---------------------------------------------------------------------------
def test_a_replaced_icc_cutoff_is_warned_about():
    features = {"icc": {"max_icc": 0.9}, "shadow": {"enabled": True}}
    warnings = sw.override_warnings({"features": features}, True)
    assert len(warnings) == 1
    assert "max_icc=0.9" in warnings[0]


def test_shadow_turned_on_by_the_sweep_is_warned_about():
    assert "--no-shadow" in sw.override_warnings({"features": {}}, True)[0]
    assert sw.override_warnings({"features": {}}, False) == []
    assert sw.override_warnings(TEMPLATE, False) == []


def test_a_combo_without_the_ranking_score_is_named(tmp_path):
    combos = _sweep(tmp_path, {"relabund": (0.12, 0.55), "clr": (0.21, 0.58)})
    old = next(c for c in combos if c.id == "clr")
    _headline("ph", 0.9, 0.9, "tax_genus_clr_bio").drop(
        "best_grouped_pooled_r2"
    ).write_csv(old.dir / sw.HEADLINE, separator="\t")
    text = sw.summarise(combos, tmp_path).read_text(encoding="utf-8")
    assert "ranked last whatever their other scores: clr" in text
    assert any(line.startswith("* relabund") for line in text.splitlines())


def test_missing_inputs_are_found_before_anything_runs(
    tmp_path, monkeypatch, clean_stop
):
    template = {
        "data": {
            "metadata": (tmp_path / "there.csv").as_posix(),
            "study_titles": None,
            "labels": {"ph": (tmp_path / "gone.csv").as_posix()},
        },
        "groupings": {"file": (tmp_path / "gone2.csv").as_posix()},
        "cv": {"param_path": (tmp_path / "gone3.yaml").as_posix()},
    }
    (tmp_path / "there.csv").touch()
    assert sorted(Path(p).name for p in sw.missing_inputs(template)) == [
        "gone.csv",
        "gone2.csv",
        "gone3.yaml",
    ]


def test_a_missing_input_stops_the_sweep_before_any_combo(
    tmp_path, monkeypatch, clean_stop
):
    ran = []
    worker, _ = _fake_worker(ran)
    monkeypatch.setattr(sw, "run_combo", worker)
    monkeypatch.setattr(sw, "missing_inputs", lambda cfg: ["/gone.csv"])
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path)
    assert ran == []


def test_a_percentage_is_refused_as_a_prevalence(
    tmp_path, monkeypatch, clean_stop
):
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path, "--min-prevalence", "5")
    assert not (tmp_path / "out").exists()


def test_models_flag_replaces_the_template_models(tmp_path):
    cfg = sw.derive_config(
        TEMPLATE,
        sw.Combo("clr", None, True, tmp_path),
        "bioproject",
        models=["rf"],
    )
    assert cfg["cv"]["models"] == ["rf"]
    assert TEMPLATE["cv"]["models"] == ["rf", "xgboost"]


# ---------------------------------------------------------------------------
# settings: the flag, else the sweep section, else the default
# ---------------------------------------------------------------------------
def _cli(**kw):
    base = dict(
        out=None,
        runner=None,
        transform=None,
        icc_max=None,
        min_prevalence=None,
        icc_group=None,
        models=None,
        holdout=False,
        final_fit=False,
        stop_on_error=False,
        no_shadow=False,
        no_pixi=False,
        extra=[],
        max_workers=None,
        log_level=None,
        cores=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _resolved(sweep=None, **cli):
    args = _cli(**cli)
    sw.resolve_settings(args, {"sweep": sweep} if sweep is not None else {})
    return args


def test_with_no_sweep_section_the_old_defaults_hold():
    args = _resolved()
    assert args.runner == "snakemake"
    assert args.transform == ["relabund", "clr", "presence"]
    assert args.icc_max == [] and args.min_prevalence == []
    assert args.cores == 8 and args.max_workers == 1
    assert args.no_shadow is False and args.holdout is False


def test_the_sweep_section_sets_every_input():
    args = _resolved(
        {
            "out": "somewhere",
            "runner": "local",
            "transforms": ["clr"],
            "icc_max": [0.9],
            "min_prevalence": 0.1,
            "icc_group": "study",
            "shadow": False,
            "models": "rf",
            "holdout": True,
            "final_fit": True,
            "cores": 3,
            "max_workers": 2,
            "stop_on_error": True,
            "pixi": False,
            "snakemake_args": ["--keep-going"],
            "log_level": "DEBUG",
        }
    )
    assert args.out == Path("somewhere")
    assert args.runner == "local"
    assert args.transform == ["clr"]
    assert args.icc_max == [0.9]
    assert args.min_prevalence == [0.1], "a single value is still an axis"
    assert args.icc_group == "study"
    assert args.no_shadow is True
    assert args.models == ["rf"]
    assert args.holdout and args.final_fit and args.stop_on_error
    assert (args.cores, args.max_workers) == (3, 2)
    assert args.no_pixi is True
    assert args.extra == ["--keep-going"]
    assert args.log_level == "DEBUG"


def test_a_flag_wins_over_the_sweep_section():
    args = _resolved(
        {"transforms": ["clr"], "cores": 3, "runner": "local"},
        transform=["presence"],
        cores=5,
        runner="snakemake",
    )
    assert args.transform == ["presence"]
    assert args.cores == 5
    assert args.runner == "snakemake"


def test_an_empty_flag_clears_an_axis_set_in_the_section():
    args = _resolved({"icc_max": [0.9]}, icc_max=[])
    assert args.icc_max == []


def test_null_in_the_section_means_the_default():
    args = _resolved({"transforms": None, "cores": None, "max_workers": None})
    assert args.transform == sw.DEFAULT_TRANSFORMS
    assert args.cores == sw.DEFAULT_CORES
    assert args.max_workers == 1


def test_the_local_runner_shares_out_the_allocation(monkeypatch):
    monkeypatch.setenv("PBS_NCPUS", "24")
    assert _resolved({"runner": "local", "max_workers": 3}).cores == 8
    assert _resolved({"runner": "local", "max_workers": 48}).cores == 1
    assert _resolved({"runner": "local", "cores": 5}).cores == 5
    # Snakemake's budget is its own business, so it is left alone.
    assert _resolved({"runner": "snakemake"}).cores == sw.DEFAULT_CORES


def _main_yaml(monkeypatch, tmp_path, sweep, *flags):
    metadata = tmp_path / "meta.csv"
    params = tmp_path / "parameters.yaml"
    metadata.touch()
    params.touch()
    template = copy.deepcopy(TEMPLATE)
    template["data"] = {"metadata": metadata.as_posix()}
    template["cv"]["param_path"] = params.as_posix()
    template["sweep"] = sweep
    config = tmp_path / "sweep.yaml"
    config.write_text(yaml.safe_dump(template), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep_transforms.py", "--config", str(config), *flags],
    )
    sw.main()


def test_the_output_folder_can_come_from_the_yaml_alone(tmp_path, monkeypatch):
    out = tmp_path / "from_yaml"
    _main_yaml(
        monkeypatch,
        tmp_path,
        {"out": out.as_posix(), "transforms": ["clr"], "pixi": False},
        "--dry-run",
    )
    assert (out / "clr" / "config.yaml").exists()


def test_no_output_folder_anywhere_is_refused(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main_yaml(monkeypatch, tmp_path, {})
    assert "sweep.out" in capsys.readouterr().err


def test_an_unknown_runner_is_refused(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main_yaml(
            monkeypatch,
            tmp_path,
            {"out": (tmp_path / "o").as_posix(), "runner": "slurm"},
        )
    assert "sweep.runner" in capsys.readouterr().err


def test_rerun_is_refused_with_the_local_runner(tmp_path, monkeypatch, capsys):
    """Steps with finished output are skipped, so old results would stay."""
    with pytest.raises(SystemExit):
        _main_yaml(
            monkeypatch,
            tmp_path,
            {"out": (tmp_path / "o").as_posix(), "runner": "local"},
            "--rerun",
        )
    assert "--rerun" in capsys.readouterr().err


def test_shadow_asked_for_in_the_section_is_not_warned_about(
    tmp_path, monkeypatch, caplog
):
    _main_yaml(
        monkeypatch,
        tmp_path,
        {
            "out": (tmp_path / "o").as_posix(),
            "transforms": ["clr"],
            "shadow": True,
            "pixi": False,
        },
        "--dry-run",
    )
    assert "features.shadow off" not in caplog.text


# ---------------------------------------------------------------------------
# the local runner
# ---------------------------------------------------------------------------
def test_a_local_combo_is_this_script_on_its_own_config(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cmd = sw.child_command(combo, "--run-combo", 4, "DEBUG")
    assert cmd[0] == sys.executable
    assert Path(cmd[2]).name == "sweep_transforms.py"
    assert cmd[3:5] == ["--run-combo", combo.config_path.as_posix()]
    assert cmd[5:] == ["--cores", "4", "--log-level", "DEBUG"]


def test_the_local_runner_does_not_touch_snakemake(tmp_path, monkeypatch):
    launched = []

    def fake_launch(combo, cmd, log_name, dry_run):
        launched.append((combo.id, cmd[3], log_name))
        return {**combo.row(), "status": "ok", "returncode": 0}

    def no_snakemake(*_args):
        raise AssertionError("Snakemake was called")

    monkeypatch.setattr(sw, "_launch", fake_launch)
    monkeypatch.setattr(sw, "run_combo", no_snakemake)
    combos = sw.build_grid(["clr", "presence"], [], True, tmp_path)
    sw.run_all(
        combos,
        TEMPLATE,
        "bioproject",
        _args(runner="local"),
        tmp_path / "status.json",
    )
    assert sorted(launched) == [
        ("clr", "--run-combo", "sweep_transforms.log"),
        ("presence", "--run-combo", "sweep_transforms.log"),
    ]


# ---------------------------------------------------------------------------
# run specs, shared with the Snakefile
# ---------------------------------------------------------------------------
def test_each_pair_is_one_run_pinned_to_its_label():
    assert sweep_tokens.run_specs({"ph": "bioproject"}) == {
        "holdout_bioproject__ph": {
            "token": "bioproject",
            "label": "ph",
            "stamp": "bioproject__ph",
        }
    }


def test_a_label_may_name_several_groupings():
    specs = sweep_tokens.run_specs({"ph": ["bioproject", None]})
    assert sorted(specs) == ["holdout_bioproject__ph", "holdout_random__ph"]


def test_pairs_win_over_holdout_groupings():
    specs = sweep_tokens.run_specs({"ph": "realm"}, ["random"])
    assert list(specs) == ["holdout_realm__ph"]


def test_holdout_groupings_are_not_pinned_to_a_label():
    specs = sweep_tokens.run_specs(None, ["random", "taxon:family"])
    assert specs["holdout_random"]["label"] is None
    assert specs["holdout_taxon_family"]["token"] == "taxon:family"


def test_pairs_can_be_written_as_a_command_line_string():
    specs = sweep_tokens.run_specs("ph:realm,oxy:bioproject")
    assert sorted(specs) == ["holdout_bioproject__oxy", "holdout_realm__ph"]


@pytest.mark.parametrize("pairs", [None, ["a"], {"ph": 3}])
def test_a_config_naming_nothing_to_run_is_refused(pairs):
    with pytest.raises(ValueError):
        sweep_tokens.run_specs(pairs, None)


# ---------------------------------------------------------------------------
# stage 1: the subset
# ---------------------------------------------------------------------------
def _subset_inputs(tmp_path):
    """40 samples: half have a temperature, and only six have a pH."""
    root = tmp_path / "in"
    root.mkdir()
    samples = [f"S{i:02d}" for i in range(40)]
    (root / "meta.csv").write_text(
        "sample,x\n" + "".join(f"{s},1\n" for s in samples)
    )
    (root / "attrs.csv").write_text(
        "sample,key,value\n" + "".join(f"{s},k,v\n" for s in samples)
    )
    (root / "prof.csv").write_text(
        "sample,coverage,taxonomy\n"
        + "".join(f'{s},1.0,"Root; d__B, odd; g__X"\n' for s in samples)
    )
    (root / "temp.csv").write_text(
        "sample,temp\n"
        + "".join(
            f"{s},{i if i % 2 == 0 else ''}\n" for i, s in enumerate(samples)
        )
    )
    (root / "ph.csv").write_text(
        "sample,ph\n"
        + "".join(
            f"{s},{7.1 if i < 6 else ''}\n" for i, s in enumerate(samples)
        )
    )
    return {
        "metadata": (root / "meta.csv").as_posix(),
        "attributes": (root / "attrs.csv").as_posix(),
        "study_titles": None,
        "profiles": (root / "prof.csv").as_posix(),
        "root": None,
        "labels": {
            "temperature": (root / "temp.csv").as_posix(),
            "ph": (root / "ph.csv").as_posix(),
        },
    }


def test_a_subset_picks_per_label_and_keeps_every_table_in_step(tmp_path):
    data = _subset_inputs(tmp_path)
    subset = sw.subset_tables(data, tmp_path / "sub", 5, 42)

    def labelled(path):
        frame = pl.read_csv(path, infer_schema_length=0)
        return frame.filter(pl.col(frame.columns[1]).is_not_null())

    # The rare label survives, and a blank never counts as labelled.
    assert labelled(subset["labels"]["ph"]).height >= 5
    assert labelled(subset["labels"]["temperature"]).height >= 5
    samples = set(
        pl.read_csv(subset["metadata"], infer_schema_length=0)["sample"]
    )
    assert 5 <= len(samples) <= 10, "at most labels x n_per_label"
    for key in ("attributes", "profiles"):
        frame = pl.read_csv(subset[key], infer_schema_length=0)
        assert set(frame["sample"]) == samples
    # A comma inside a quoted taxonomy must come through whole.
    taxonomy = pl.read_csv(subset["profiles"], infer_schema_length=0)
    assert taxonomy["taxonomy"][0] == "Root; d__B, odd; g__X"


def test_a_subset_asking_for_more_than_exists_takes_what_is_there(tmp_path):
    data = _subset_inputs(tmp_path)
    subset = sw.subset_tables(data, tmp_path / "sub", 1000, 42)
    assert pl.read_csv(subset["metadata"]).height == 23, "20 even + 3 odd ph"


def test_the_same_subset_is_reused_not_rebuilt(tmp_path):
    data = _subset_inputs(tmp_path)
    subset = sw.subset_tables(data, tmp_path / "sub", 5, 42)
    marker = Path(subset["metadata"])
    marker.write_text("sample,x\nkept,1\n")
    sw.subset_tables(data, tmp_path / "sub", 5, 42)
    assert marker.read_text() == "sample,x\nkept,1\n"


@pytest.mark.parametrize("n, seed", [(6, 42), (5, 1)])
def test_a_different_subset_cannot_overwrite_one_in_use(tmp_path, n, seed):
    """Built combos keep the same paths, so nothing else would notice."""
    data = _subset_inputs(tmp_path)
    sw.subset_tables(data, tmp_path / "sub", 5, 42)
    with pytest.raises(SystemExit, match="different settings"):
        sw.subset_tables(data, tmp_path / "sub", n, seed)


def test_an_interrupted_subset_is_rebuilt(tmp_path):
    data = _subset_inputs(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "metadata.csv").write_text("half a file")
    subset = sw.subset_tables(data, tmp_path / "sub", 5, 42)
    assert "sample" in pl.read_csv(subset["metadata"]).columns


def test_the_subset_reaches_the_combos_and_a_dry_run_writes_none(
    tmp_path, monkeypatch
):
    data = _subset_inputs(tmp_path)
    out = tmp_path / "o"
    template = copy.deepcopy(TEMPLATE)
    template["data"] = data
    template["sweep"] = {
        "out": out.as_posix(),
        "transforms": ["clr"],
        "pixi": False,
        "subset": {"n_per_label": 5},
    }
    config = tmp_path / "sweep.yaml"
    config.write_text(yaml.safe_dump(template), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep_transforms.py", "--config", str(config), "--dry-run"],
    )
    sw.main()
    derived = yaml.safe_load((out / "clr" / "config.yaml").read_text())
    assert "subset_n5_seed42" in derived["data"]["metadata"]
    assert not (out / "subset_n5_seed42").exists()


# ---------------------------------------------------------------------------
# stages 7-9: choosing, collecting and ranking
# ---------------------------------------------------------------------------
def _models(*rows):
    """Rows of (combo, scheme, feature_set, score)."""
    return pl.DataFrame(
        [
            {
                "combo": combo,
                "transform": "clr",
                "min_prevalence": 0.1,
                "max_icc": None,
                "holdout_grouping": "bioproject__ph",
                "label": "ph",
                "scheme": scheme,
                "feature_set": feature_set,
                "model": "RandomForestRegressor",
                sw.SCORE: score,
            }
            for combo, scheme, feature_set, score in rows
        ]
    )


def test_pruning_takes_the_best_clean_set_on_a_grouped_scheme():
    targets = sw.prune_targets(
        _models(
            # random scores best, and must not be chosen.
            ("a", "random", "s", 0.9),
            ("a", "month", "s", 0.4),
            ("a", "eco", "s", 0.2),
            # A shadow set is a noise floor, never a candidate.
            ("a", "eco", "s_shad", 0.8),
            ("b", "eco", "s", 0.3),
        )
    )
    assert targets.select("combo", "scheme").rows() == [
        ("a", "month"),
        ("b", "eco"),
    ]


def test_a_fixed_scheme_puts_every_row_on_that_scheme():
    models = _models(
        ("a", "month", "s", 0.4),
        ("a", "eco", "s", 0.6),
        ("b", "month", "s", 0.1),
        ("b", "eco", "s", 0.9),
    )
    targets = sw.prune_targets(models, "month")
    assert targets.select("combo", "scheme", sw.SCORE).rows() == [
        ("a", "month", 0.4),
        ("b", "month", 0.1),
    ]


def test_a_label_without_the_fixed_scheme_is_named_not_filled_in(caplog):
    """A row must never quietly fall back to a different scheme."""
    models = _models(("a", "month", "s", 0.4), ("a", "eco", "s", 0.6))
    targets = sw.prune_targets(models, "bioproject")
    assert targets.height == 0
    sw.warn_missing_targets(models, targets, "bioproject")
    assert "without scheme 'bioproject'" in caplog.text
    assert "'eco', 'month'" in caplog.text


def test_with_no_fixed_scheme_nothing_is_warned_about(caplog):
    models = _models(("a", "month", "s", 0.4))
    sw.warn_missing_targets(models, sw.prune_targets(models), None)
    assert caplog.text == ""


def test_a_sweep_with_only_random_has_nothing_to_prune():
    models = _models(("a", "random", "s", 0.9))
    assert sw.prune_targets(models).height == 0


def test_pruning_is_off_unless_asked_for():
    assert sw.prune_settings(None)["enabled"] is False
    settings = sw.prune_settings({"prune": {"mode": "coverage"}})
    assert settings["mode"] == "coverage" and settings["iterations"] == 3


@pytest.mark.parametrize(
    "settings, tag",
    [
        ({"mode": "zero"}, "zero_it3_min10"),
        ({"mode": "coverage"}, "coverage0.95_it3_min10"),
        ({"mode": "coverage", "coverage": 0.5}, "coverage0.5_it3_min10"),
        ({"mode": "zero", "iterations": 1}, "zero_it1_min10"),
    ],
)
def test_each_set_of_prune_settings_gets_its_own_folder(settings, tag):
    assert sw.prune_tag({**sw.PRUNE_DEFAULTS, **settings}) == tag


def test_kept_features_reads_the_file_of_that_round(tmp_path):
    kept = tmp_path / "kept_features"
    kept.mkdir()
    (kept / "ph__month__Model__it2.txt").write_text("a b\nc\n\n")
    row = dict(label="ph", scheme="month", model="Model", iteration=2)
    assert sw.kept_features(tmp_path, row) == ["a b", "c"]


def _finished_fit(root, row, tag, version, **summary):
    out = sw.prune_dir(root / row["combo"], tag, row) / version
    out.mkdir(parents=True)
    (out / "fit_summary.json").write_text(json.dumps(summary))


def test_the_benchmark_collects_what_finished(tmp_path):
    targets = sw.prune_targets(
        _models(("a", "m", "s", 0.4), ("b", "m", "s", 0.3))
    )
    row = next(targets.iter_rows(named=True))
    assert sw.benchmark_table(targets, tmp_path, "t") is None
    _finished_fit(tmp_path, row, "t", "unpruned", n_features=9, holdout_r2=0.5)
    _finished_fit(tmp_path, row, "t", "pruned", n_features=3, holdout_r2=0.6)
    table = sw.benchmark_table(targets, tmp_path, "t")
    assert table.select("combo", "version", "n_features").rows() == [
        ("a", "unpruned", 9),
        ("a", "pruned", 3),
    ]
    # Other settings live in another folder, so they are not mixed in.
    assert sw.benchmark_table(targets, tmp_path, "other") is None


def test_finished_means_both_versions_of_every_target(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    targets = sw.prune_targets(_models(("clr", "m", "s", 0.4)))
    row = next(targets.iter_rows(named=True))
    assert not sw._finish_done(combo, targets, "t")
    _finished_fit(tmp_path, row, "t", "unpruned", n_features=9)
    assert not sw._finish_done(combo, targets, "t")
    _finished_fit(tmp_path, row, "t", "pruned", n_features=3)
    assert sw._finish_done(combo, targets, "t")


def _fit(combo, label, version, holdout, n=10):
    return dict(
        combo=combo,
        label=label,
        version=version,
        holdout_r2=holdout,
        n_features=n,
        model="m",
        scheme="month",
        cv_pooled_r2=0.1,
        overfit_gap=0.2,
    )


def test_the_ranking_is_on_pruned_holdout_within_each_label():
    ranking = sw.rank_benchmark(
        pl.DataFrame(
            [
                _fit("a", "ph", "unpruned", 0.5, 100),
                _fit("a", "ph", "pruned", 0.4, 10),
                _fit("b", "ph", "unpruned", 0.3, 100),
                _fit("b", "ph", "pruned", 0.6, 20),
                _fit("a", "oxy", "unpruned", 0.1, 100),
                _fit("a", "oxy", "pruned", 0.2, 30),
            ]
        )
    )
    assert ranking.select("label", "rank", "combo").rows() == [
        ("oxy", 1, "a"),
        ("ph", 1, "b"),
        ("ph", 2, "a"),
    ]
    best = ranking.filter(pl.col("combo") == "b").row(0, named=True)
    assert (best["features_before"], best["features_after"]) == (100, 20)
    assert best["pruning_change"] == pytest.approx(0.3)


def test_a_nan_holdout_does_not_take_first_place():
    ranking = sw.rank_benchmark(
        pl.DataFrame(
            [
                _fit("a", "ph", "unpruned", 0.5),
                _fit("a", "ph", "pruned", float("nan")),
                _fit("b", "ph", "unpruned", 0.3),
                _fit("b", "ph", "pruned", 0.1),
            ]
        )
    )
    assert ranking["combo"].to_list() == ["b", "a"]
    assert ranking["rank"].to_list() == [1, None]


def test_stage_minutes_are_totalled_cheapest_first(tmp_path):
    slow = sw.Combo("clr", None, True, tmp_path)
    fast = sw.Combo("presence", None, True, tmp_path)
    missing = sw.Combo("relabund", None, True, tmp_path)
    for combo, cv in ((slow, 600), (fast, 60)):
        combo.dir.mkdir()
        (combo.dir / "stage_times.json").write_text(json.dumps({"cv": cv}))
    table = sw.stage_table([slow, fast, missing])
    assert table["combo"].to_list() == ["presence", "clr"]
    assert table["total_min"].to_list() == [1.0, 10.0]


def test_one_failed_pruning_is_counted_and_the_next_label_still_runs(
    tmp_path, monkeypatch
):
    import prune_iterate_cv as prune

    combo = sw.Combo("clr", None, True, tmp_path)
    combo.dir.mkdir()
    combo.config_path.write_text(
        yaml.safe_dump({"cv": {"param_path": "p.yaml"}}), encoding="utf-8"
    )
    (tmp_path / "summary").mkdir()
    rows = _models(("clr", "m", "s", 0.4)).with_columns(
        pl.lit("x").alias("transform")
    )
    pl.concat(
        [rows, rows.with_columns(pl.lit("oxy").alias("label"))]
    ).write_csv(tmp_path / "summary" / "all_models.tsv", separator="\t")
    tried = []

    def fail_on_ph(**kw):
        tried.append(kw["label"])
        raise RuntimeError("out of memory")

    monkeypatch.setattr(prune, "run_loop", fail_on_ph)
    assert sw.finish_combo(combo.config_path, 1) == 2
    assert sorted(tried) == ["oxy", "ph"]


def test_a_broken_figure_does_not_cost_the_results(caplog):
    def boom():
        raise RuntimeError("no such column")

    sw._best_effort("draw it", boom)
    assert "could not draw it" in caplog.text


# ---------------------------------------------------------------------------
# the whole thing, from one YAML
# ---------------------------------------------------------------------------
def _real_inputs(root):
    """A small dataset that survives grouped CV: 90 samples, 12 studies."""
    import numpy as np

    root.mkdir()
    rng = np.random.default_rng(3)
    n = 90
    samples = [f"S{i:03d}" for i in range(n)]
    temp = 5 + 25 * rng.random(n)
    pl.DataFrame(
        {
            "sample": samples,
            "bioproject": [f"P{i % 12}" for i in range(n)],
            "month": [i % 6 + 1 for i in range(n)],
            "biome": ["marine"] * n,
            "biosample": samples,
            "lat": [10.0] * n,
            "lon": [20.0] * n,
            "collection_date": ["2020-01-15"] * n,
            "mbp": [500.0] * n,
            "organism": ["marine metagenome"] * n,
        }
    ).write_csv(root / "metadata.csv")
    pl.DataFrame(
        {"sample": samples, "key": ["depth"] * n, "value": ["10"] * n}
    ).write_csv(root / "attributes.csv")
    rows = []
    for i, sample in enumerate(samples):
        for taxon in range(14):
            scale = np.exp(0.12 * temp[i]) if taxon < 5 else 3.0
            rows.append(
                {
                    "sample": sample,
                    "coverage": float(rng.gamma(2.0, scale / 2.0)),
                    "taxonomy": (
                        f"Root; d__B; p__P; c__C; o__O{taxon}; f__F{taxon}; "
                        f"g__G{taxon}"
                    ),
                }
            )
    pl.DataFrame(rows).write_csv(root / "profiles.csv")
    pl.DataFrame(
        {"sample": samples, "temp": [round(float(v), 2) for v in temp]}
    ).write_csv(root / "temp.csv")
    (root / "parameters.yaml").write_text("rf:\n  n_estimators:\n    - 5\n")


def _whole_sweep_config(inputs, out):
    return {
        "sweep": {
            "out": out.as_posix(),
            "runner": "local",
            "cores": 1,
            "subset": {"n_per_label": 80},
            "transforms": ["relabund"],
            "pairs": {"t": "bioproject"},
            "shadow": True,
            "final_fit": True,
            "prune": {
                "enabled": True,
                "mode": "coverage",
                "coverage": 0.9,
                "iterations": 1,
                "min_features": 3,
            },
        },
        "data": {
            "metadata": (inputs / "metadata.csv").as_posix(),
            "attributes": (inputs / "attributes.csv").as_posix(),
            "profiles": (inputs / "profiles.csv").as_posix(),
            "labels": {"t": (inputs / "temp.csv").as_posix()},
        },
        "groupings": {
            "defaults": False,
            "columns": ["month", "bioproject"],
            "file": None,
        },
        "features": {
            "add_taxonomic_features": True,
            "all": False,
            "ranks": ["order"],
            "prefix": "x",
            "format": "parquet",
            "labelled_only": True,
            "shadow": {"n": 4, "seed": 1},
        },
        "preprocessing": {"enabled": False},
        "split": {
            "holdout": {"test_size": 0.2, "n_bins": 3},
            "cv": {
                "n_folds": 3,
                "n_bins": 3,
                "methods": ["kfold"],
                "grouping": "all",
                "strict": False,
            },
        },
        "cv": {
            "models": ["rf"],
            "mode": "run",
            "param_path": (inputs / "parameters.yaml").as_posix(),
            "feature_set": ["x_order"],
        },
        "holdout_evaluation": {"enabled": False},
        "outputs": {"cv_save_models": False},
        "visualise": {"enabled": False, "formats": ["png"]},
        "shap": {"enabled": False},
    }


def test_one_yaml_runs_the_sweep_from_top_to_bottom(
    tmp_path, monkeypatch, capsys
):
    inputs = tmp_path / "in"
    _real_inputs(inputs)
    out = tmp_path / "o"
    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(_whole_sweep_config(inputs, out)), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys, "argv", ["sweep_transforms.py", "--config", str(config)]
    )
    # main installs handlers for these; leave the test runner's alone.
    monkeypatch.setattr(signal, "signal", lambda *_args: None)

    sw.main()

    summary = out / "summary"
    for table in (
        "SUMMARY.txt",
        "all_models.tsv",
        "prune_targets.tsv",
        "prune_history_all.tsv",
        "final_fit_benchmark.tsv",
        "benchmark_ranking.tsv",
        "stage_minutes.tsv",
    ):
        assert (summary / table).exists(), table
    for figure in (
        "transform_sweep_t__random",
        "transform_sweep_t__month",
        "icc_distribution",
        "shadow_floor",
        "prune_trajectory",
        "benchmark_scoreboard",
        "benchmark_runtime",
    ):
        assert (summary / "figures" / f"{figure}.png").exists(), figure
    benchmark = pl.read_csv(
        summary / "final_fit_benchmark.tsv", separator="\t"
    )
    assert sorted(benchmark["version"].to_list()) == ["pruned", "unpruned"]
    assert benchmark["holdout_r2"].is_not_null().all()
    assert "Written under" in capsys.readouterr().out

    # A rerun finds nothing to do and leaves the results as they were.
    before = (summary / "final_fit_benchmark.tsv").read_bytes()
    log = out / "relabund" / "sweep_transforms.log"
    finish_log = out / "relabund" / "sweep_finish.log"
    times = (log.stat().st_mtime, finish_log.stat().st_mtime)
    sw.main()
    assert (log.stat().st_mtime, finish_log.stat().st_mtime) == times
    assert (summary / "final_fit_benchmark.tsv").read_bytes() == before
