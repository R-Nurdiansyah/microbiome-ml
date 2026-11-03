"""The transform sweep: does each combo get the right config and name?

The thing most worth guarding is `cv.feature_set`. A transform's suffix
is not its config value -- `presence` builds `_pa` -- so a hand-written
name silently trains nothing. Deriving it from the package's own
`feature_set_name` is the whole point of the driver, and these pin it.

Second is isolation: every combo writes a prepared dataset, splits and a
saved dataset, and the template's paths are absolute and shared. A combo
that inherits them overwrites its neighbour.
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

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
    cmd = sw.command_for(combo, "snakemake", 16, True, [])
    assert cmd[:2] == ["pixi", "run"]
    assert combo.config_path.as_posix() in cmd
    assert "--cores" in cmd and "16" in cmd


def test_script_runner_and_extra_passthrough(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cmd = sw.command_for(combo, "script", 4, False, ["--log-level", "DEBUG"])
    assert cmd[0] == "python"
    assert cmd[-2:] == ["--log-level", "DEBUG"]


def test_extra_setting_jobs_suppresses_cores(tmp_path):
    """On a cluster the budget is --jobs; two budgets would disagree."""
    combo = sw.Combo("clr", None, True, tmp_path)
    for extra in (["--jobs", "20"], ["--jobs=20"], ["-j", "20"]):
        cmd = sw.command_for(combo, "snakemake", 24, False, extra)
        assert "--cores" not in cmd, extra


def test_extra_that_is_not_a_job_flag_keeps_cores(tmp_path):
    combo = sw.Combo("clr", None, True, tmp_path)
    cmd = sw.command_for(combo, "snakemake", 24, False, ["--rerun-incomplete"])
    assert "--cores" in cmd and "24" in cmd


def test_a_submit_command_with_spaces_stays_one_argument(tmp_path):
    """subprocess gets a list, so the quoted submit command must not split."""
    combo = sw.Combo("clr", None, True, tmp_path)
    submit = "mqsub -t {threads} -m 64 --hours 12 --"
    cmd = sw.command_for(
        combo,
        "snakemake",
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
    assert table.columns[:4] == sw.COMBO_KEYS
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
