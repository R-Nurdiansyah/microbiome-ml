"""Feature recipes (wrangle/recipe.py) and the sweep that drives them."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from microbiome_ml.wrangle import recipe as rc
from microbiome_ml.wrangle.features import SampleFeatureSet

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import pipeline_steps as steps  # noqa: E402
import sweep_transforms as sw  # noqa: E402


def _studies(seed=0):
    """30 samples in 3 studies of 2, 14 and 14, and 4 taxa.

    rare_marker: only in the 2 samples of study a -- rare, ICC 1.
    common_marker: everywhere, high in a and low in b, c -- ICC near 1.
    noise, noise2: everywhere, unrelated to the study -- ICC near 0.
    """
    rng = np.random.default_rng(seed)
    study = ["a"] * 2 + ["b"] * 14 + ["c"] * 14
    n = len(study)
    in_a = np.array([s == "a" for s in study])
    wide = pl.DataFrame(
        {
            "sample": [f"s{i}" for i in range(n)],
            "rare_marker": np.where(in_a, 0.3, 0.0),
            "common_marker": np.where(in_a, 0.5, 0.01)
            + rng.uniform(0, 1e-3, n),
            "noise": rng.uniform(0.01, 0.2, n),
            "noise2": rng.uniform(0.01, 0.2, n),
        }
    )
    groups = pl.DataFrame({"sample": wide["sample"], "bioproject": study})
    return wide, groups


def test_prevalence_counts_groups_not_samples():
    wide = pl.DataFrame(
        {
            "sample": list("abcdef"),
            # 3 samples, all in project p1
            "one_project": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            # 2 samples, in two projects
            "two_projects": [0.1, 0.0, 0.0, 0.2, 0.0, 0.0],
        }
    )
    groups = pl.DataFrame(
        {
            "sample": list("abcdef"),
            "bioproject": ["p1", "p1", "p1", "p2", "p2", "p3"],
        }
    )
    # two of three projects: need = int(0.67 * 3) = 2, as prepare counts
    recipe = rc.Recipe(min_prevalence=0.67, prevalence_by="bioproject")
    kept, report = rc.select(wide, recipe, groups)
    assert kept == ["two_projects"]
    row = report.filter(pl.col("taxonomy") == "one_project").row(
        0, named=True
    )
    assert row["dropped_by"] == "prevalence"
    assert row["n_units"] == 3
    assert row["prevalence"] == pytest.approx(1 / 3)


def test_icc_on_rare_taxa_spares_common_ones():
    wide, groups = _studies()
    every = rc.Recipe(icc_group="bioproject", max_icc=0.75)
    kept, _ = rc.select(wide, every, groups)
    assert "rare_marker" not in kept and "common_marker" not in kept
    assert {"noise", "noise2"} <= set(kept)

    rare_only = rc.Recipe(
        icc_group="bioproject", max_icc=0.75, icc_rare_below=0.1
    )
    kept, report = rc.select(wide, rare_only, groups)
    assert "rare_marker" not in kept
    assert "common_marker" in kept
    rare = report.filter(pl.col("taxonomy") == "rare_marker").row(
        0, named=True
    )
    assert rare["dropped_by"] == "icc"
    assert rare["sample_prevalence"] == pytest.approx(2 / 30)


def test_variance_threshold_matches_sklearn():
    sklearn_fs = pytest.importorskip("sklearn.feature_selection")
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (40, 6)) * np.array(
        [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 0.0]
    )
    taxa = [f"t{i}" for i in range(X.shape[1])]
    wide = pl.DataFrame(X, schema=taxa, orient="row").with_columns(
        pl.Series("sample", [f"s{i}" for i in range(40)])
    )
    threshold = 1e-5
    kept, _ = rc.select(wide, rc.Recipe(variance_threshold=threshold))
    expected = sklearn_fs.VarianceThreshold(threshold).fit(X).get_support()
    assert kept == [t for t, keep in zip(taxa, expected) if keep]
    assert "t5" not in kept  # constant


def test_selection_is_fitted_on_the_training_samples_only():
    wide = pl.DataFrame(
        {
            "sample": ["tr1", "tr2", "tr3", "te1"],
            "everywhere": [0.5, 0.4, 0.6, 0.5],
            "test_only": [0.0, 0.0, 0.0, 0.5],
        }
    )
    table, report = rc.build(
        wide, ["tr1", "tr2", "tr3"], rc.Recipe(min_prevalence=0.1)
    )
    assert table.columns == ["sample", "everywhere"]
    # the test sample keeps its row, with the training pool's columns
    assert table.get_column("sample").to_list() == wide["sample"].to_list()
    gone = report.filter(pl.col("taxonomy") == "test_only")
    assert gone["dropped_by"].item() == "prevalence"


def test_clr_is_taken_over_the_taxa_kept():
    wide, _ = _studies()
    table, _ = rc.build(
        wide,
        wide["sample"].to_list(),
        rc.Recipe(transform="clr", min_prevalence=0.5),
    )
    taxa = [c for c in table.columns if c != "sample"]
    assert taxa == ["common_marker", "noise", "noise2"]
    sums = table.select(pl.sum_horizontal(taxa)).to_series().to_numpy()
    assert np.allclose(sums, 0.0, atol=1e-4)


def test_a_recipe_that_drops_everything_says_so():
    wide, _ = _studies()
    with pytest.raises(ValueError, match="dropped every taxon"):
        rc.build(
            wide, wide["sample"].to_list(), rc.Recipe(variance_threshold=1.0)
        )


def test_recipe_reads_the_features_section():
    recipe = rc.Recipe.from_features(
        {
            "transform": "CLR",
            "min_prevalence": 0.2,
            "prevalence_by": "bioproject",
            "icc": {
                "enabled": True,
                "group": "bioproject",
                "max_icc": 0.75,
                "rare_below": 0.05,
            },
            "variance_threshold": 1e-6,
        }
    )
    assert recipe.transform == "clr"
    assert recipe.min_prevalence == 0.2
    assert (recipe.max_icc, recipe.icc_rare_below) == (0.75, 0.05)
    assert recipe.variance_threshold == 1e-6
    # ICC switched off means no ICC, whatever max_icc says
    off = rc.Recipe.from_features({"icc": {"enabled": False, "max_icc": 0.5}})
    assert off.max_icc is None and off.icc_group is None
    with pytest.raises(ValueError, match="fraction"):
        rc.Recipe.from_features({"min_prevalence": 20})


def test_apply_to_dataset_replaces_the_base_set(tmp_path):
    wide, groups = _studies()
    dataset = SimpleNamespace(
        feature_sets={"tax_order": SampleFeatureSet.from_df(wide, "tax_order")},
        groupings=groups,
    )
    features = {
        "base_set": "tax_order",
        "transform": "relabund",
        "min_prevalence": 0.67,
        "prevalence_by": "bioproject",
    }
    name = steps.recipe_set_name(features)
    assert name == "tax_order_bio"
    rc.apply_to_dataset(
        dataset,
        features,
        wide["sample"].to_list(),
        name,
        report_path=tmp_path / "feature_selection.tsv",
    )
    assert list(dataset.feature_sets) == ["tax_order_bio"]
    built = dataset.feature_sets["tax_order_bio"].features.collect()
    # rare_marker is in 1 of 3 projects, short of the 2 needed
    assert "rare_marker" not in built.columns
    assert (tmp_path / "feature_selection.tsv").exists()


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
TEMPLATE = {
    "data": {"metadata": "/abs/meta.csv"},
    "features": {
        "all": False,
        "ranks": ["order"],
        "prefix": "tax",
        "transform": "relabund",
        "prevalence_by": "bioproject",
        "shadow": {"enabled": False},
    },
    "cv": {"models": ["rf", "xgboost", "svm"], "feature_set": ["tax_order"]},
    "outputs": {"base_dir": "/shared/run"},
    "sweep": {"pairs": {"ph": ["bioproject"]}},
}


def test_variance_is_a_grid_axis(tmp_path):
    combos = sw.build_grid(
        ["relabund", "clr"],
        [0.75],
        False,
        tmp_path,
        [0.1, 0.2],
        [1e-6, 1e-5],
    )
    assert len(combos) == 2 * 2 * 2 * 3
    ids = {c.id for c in combos}
    assert "clr_prev0p2_icc0p75_var1e-06" in ids
    assert "relabund_prev0p1" in ids


def test_a_base_prepared_combo_runs_no_prepare(tmp_path):
    combo = sw.Combo("clr", 0.75, False, tmp_path, 0.2, 1e-6)
    cfg = sw.derive_config(
        TEMPLATE,
        combo,
        "bioproject",
        base_prepared="/base/prepared",
        icc_rare_below=0.05,
    )
    assert cfg["outputs"]["prepared_dir"] == "/base/prepared"
    features = cfg["features"]
    assert features["base_set"] == "tax_order"
    assert features["variance_threshold"] == 1e-6
    assert features["icc"]["max_icc"] == 0.75
    assert features["icc"]["rare_below"] == 0.05
    # the name the CV job gives the recipe's set
    assert cfg["cv"]["feature_set"] == [steps.recipe_set_name(features)]


def test_combos_with_variance_are_found_again(tmp_path):
    combos = sw.build_grid(["clr"], [0.75], False, tmp_path, [0.2], [1e-5])
    for combo in combos:
        sw.write_config(
            sw.derive_config(
                TEMPLATE, combo, "bioproject", base_prepared="/base"
            ),
            combo.config_path,
        )
    found = {c.id for c in sw.discover_combos(tmp_path)}
    assert found == {c.id for c in combos}
