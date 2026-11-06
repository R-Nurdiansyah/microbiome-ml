"""Transform-sweep figures: the data handling, not the pixels.

What is worth pinning is everything that happens before matplotlib: the
duplicate rows collate hands over, which model wins a cell, which
scheme leads, and that a label with no usable rows is skipped rather
than drawn empty. The drawing itself is exercised only to the extent
that a figure comes out and a file lands on disk.
"""

import json
import sys
from pathlib import Path

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import plot_transform_sweep as ptw  # noqa: E402

SCHEMES = ("month_logo", "ecoregion_logo", "bioproject_logo")


def _row(
    combo,
    feature_set,
    label,
    scheme,
    model,
    pooled,
    folds=None,
):
    return {
        "combo": combo,
        "transform": combo,
        "max_icc": None,
        "shadow": True,
        "holdout_grouping": f"bioproject__{label}",
        "feature_set": feature_set,
        "label": label,
        "scheme": scheme,
        "model": model,
        "avg_validation_r2": pooled,
        "pooled_validation_r2": pooled,
        "n_label": 500,
        "cross_val_scores": json.dumps(folds if folds else [pooled] * 4),
    }


def _table(tmp_path, rows):
    path = tmp_path / "all_models.tsv"
    pl.DataFrame(rows).write_csv(path, separator="\t")
    return path


def _sweep_rows(label="ph"):
    """One row per transform x scheme x {clean, shadow} x model."""
    sets = {
        "relabund": "tax_genus_bio",
        "clr": "tax_genus_clr_bio",
        "presence": "tax_genus_pa_bio",
    }
    # month_logo is positive, the other two collapse -- the real shape.
    base = {
        "month_logo": 0.30,
        "ecoregion_logo": -3.0,
        "bioproject_logo": -5.0,
    }
    rows = []
    for i, (combo, clean) in enumerate(sets.items()):
        for scheme, score in base.items():
            for j, name in enumerate((clean, f"{clean}_shad")):
                for k, model in enumerate(
                    ("RandomForestRegressor", "XGBRegressor")
                ):
                    rows.append(
                        _row(
                            combo,
                            name,
                            label,
                            scheme,
                            model,
                            score + 0.01 * i + 0.002 * j + 0.005 * k,
                        )
                    )
    return rows


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_duplicate_rows_from_collate_are_dropped(tmp_path):
    """collate globs run dirs at two depths, so rows arrive twice."""
    rows = _sweep_rows()
    path = _table(tmp_path, rows + rows[:5])
    frame = ptw.load_models(path)
    assert frame.height == len(rows)


def test_rows_without_a_usable_score_are_dropped(tmp_path):
    rows = _sweep_rows()
    rows.append(
        _row("clr", "tax_genus_clr_bio", "ph", "month_logo", "M", None)
    )
    frame = ptw.load_models(_table(tmp_path, rows))
    assert frame.filter(pl.col("model") == "M").height == 0


def test_a_table_missing_a_column_says_which(tmp_path):
    path = tmp_path / "bad.tsv"
    pl.DataFrame({"combo": ["clr"], "label": ["ph"]}).write_csv(
        path, separator="\t"
    )
    with pytest.raises(ValueError, match="missing column"):
        ptw.load_models(path)


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
def test_each_cell_keeps_its_better_model(tmp_path):
    rows = [
        _row("clr", "tax_genus_clr_bio", "ph", "month_logo", "rf", 0.20),
        _row("clr", "tax_genus_clr_bio", "ph", "month_logo", "xgb", 0.35),
    ]
    best = ptw.best_per_cell(ptw.load_models(_table(tmp_path, rows)))
    assert best.height == 1
    assert best.row(0, named=True)["model"] == "xgb"
    assert best.row(0, named=True)[ptw.SCORE] == pytest.approx(0.35)


def test_clean_and_shadow_sets_stay_separate_cells(tmp_path):
    """They are compared side by side, so they must not collapse."""
    rows = [
        _row("clr", "tax_genus_clr_bio", "ph", "month_logo", "rf", 0.20),
        _row("clr", "tax_genus_clr_bio_shad", "ph", "month_logo", "rf", 0.22),
    ]
    best = ptw.best_per_cell(ptw.load_models(_table(tmp_path, rows)))
    assert best.height == 2


def test_schemes_are_ordered_best_first(tmp_path):
    best = ptw.best_per_cell(ptw.load_models(_table(tmp_path, _sweep_rows())))
    assert ptw.scheme_order(best) == list(SCHEMES)


def test_scheme_order_ignores_the_shadow_sets(tmp_path):
    """A shadow set must not be able to reorder the schemes."""
    rows = [
        _row("clr", "tax_genus_clr_bio", "ph", "month_logo", "rf", 0.10),
        _row("clr", "tax_genus_clr_bio", "ph", "bioproject_logo", "rf", -1.0),
        # a wildly high shadow score on the losing scheme
        _row(
            "clr", "tax_genus_clr_bio_shad", "ph", "bioproject_logo", "rf", 9.0
        ),
    ]
    best = ptw.best_per_cell(ptw.load_models(_table(tmp_path, rows)))
    assert ptw.scheme_order(best)[0] == "month_logo"


def test_transforms_keep_a_fixed_order(tmp_path):
    best = ptw.best_per_cell(ptw.load_models(_table(tmp_path, _sweep_rows())))
    assert ptw.transform_order(best) == ["relabund", "clr", "presence"]


def test_an_unknown_transform_sorts_after_the_known_ones(tmp_path):
    rows = _sweep_rows() + [
        _row("zzz", "tax_genus_zzz", "ph", "month_logo", "rf", 0.1)
    ]
    best = ptw.best_per_cell(ptw.load_models(_table(tmp_path, rows)))
    assert ptw.transform_order(best) == [
        "relabund",
        "clr",
        "presence",
        "zzz",
    ]


def test_each_transform_keeps_one_colour(tmp_path):
    hues = ptw.hue_map(["relabund", "clr", "presence"])
    assert len(set(hues.values())) == 3
    assert hues == ptw.hue_map(["relabund", "clr", "presence"])


# ---------------------------------------------------------------------------
# Fold parsing
# ---------------------------------------------------------------------------
def test_folds_are_read_from_cross_val_scores():
    cell = {"cross_val_scores": "[0.1, 0.2, 0.3]"}
    assert ptw._folds(cell) == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "raw", [None, "", "not json", "{}", "[]", '["a", "b"]']
)
def test_unusable_fold_scores_give_an_empty_list(raw):
    assert ptw._folds({"cross_val_scores": raw}) == []


def test_non_finite_folds_are_dropped():
    assert ptw._folds({"cross_val_scores": "[0.5, null, 0.7]"}) == [0.5, 0.7]


def test_a_missing_cell_has_no_folds():
    assert ptw._folds(None) == []


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_one_figure_per_label(tmp_path):
    rows = _sweep_rows("ph") + _sweep_rows("temperature")
    out = tmp_path / "figs"
    written = ptw.plot_sweep(
        ptw.load_models(_table(tmp_path, rows)), out, ["png"]
    )
    assert len(written) == 2
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == [
        "transform_sweep_ph.png",
        "transform_sweep_temperature.png",
    ]


def test_only_the_requested_labels_are_drawn(tmp_path):
    rows = _sweep_rows("ph") + _sweep_rows("temperature")
    out = tmp_path / "figs"
    written = ptw.plot_sweep(
        ptw.load_models(_table(tmp_path, rows)), out, ["png"], labels=["ph"]
    )
    assert len(written) == 1
    assert [p.name for p in out.glob("*.png")] == ["transform_sweep_ph.png"]


def test_a_label_with_no_rows_is_skipped_not_drawn(tmp_path):
    out = tmp_path / "figs"
    written = ptw.plot_sweep(
        ptw.load_models(_table(tmp_path, _sweep_rows("ph"))),
        out,
        ["png"],
        labels=["ph", "absent"],
    )
    assert len(written) == 1


def test_a_sweep_without_fold_scores_still_draws(tmp_path):
    """Panel (c) is dropped; (a) and (b) must survive."""
    rows = [{**r, "cross_val_scores": None} for r in _sweep_rows("ph")]
    out = tmp_path / "figs"
    written = ptw.plot_sweep(
        ptw.load_models(_table(tmp_path, rows)), out, ["png"]
    )
    assert len(written) == 1
    assert (out / "transform_sweep_ph.png").stat().st_size > 0


def test_n_label_is_read_from_the_headline(tmp_path):
    headline = tmp_path / "headline.tsv"
    pl.DataFrame(
        {"label": ["ph", "temperature"], "n_label": [330, 1936]}
    ).write_csv(headline, separator="\t")
    assert ptw.label_sizes(headline) == {"ph": 330, "temperature": 1936}


def test_a_named_scheme_gets_its_own_figure_instead_of_the_best(tmp_path):
    out = tmp_path / "figs"
    models = ptw.load_models(_table(tmp_path, _sweep_rows("ph")))
    written = ptw.plot_sweep(models, out, ["png"], scheme="ecoregion_logo")
    assert [p.name for p in written] == ["transform_sweep_ph__ecoregion_logo"]
    assert (out / "transform_sweep_ph__ecoregion_logo.png").exists()

    fig = ptw.draw_label(
        "ph", ptw.best_per_cell(models), None, scheme="ecoregion_logo"
    )
    assert "ecoregion_logo" in fig.axes[1].get_title(loc="left")
    # month_logo scores best here, and panel (a) still has to show it.
    assert "month" in fig.axes[0].get_xticklabels()[0].get_text()
    ptw.plt.close(fig)


def test_a_label_that_never_ran_the_scheme_is_skipped(tmp_path):
    rows = _sweep_rows("ph") + [
        r
        for r in _sweep_rows("temperature")
        if r["scheme"] != "bioproject_logo"
    ]
    out = tmp_path / "figs"
    written = ptw.plot_sweep(
        ptw.load_models(_table(tmp_path, rows)),
        out,
        ["png"],
        scheme="bioproject_logo",
    )
    assert [p.name for p in written] == ["transform_sweep_ph__bioproject_logo"]


def test_holdout_figure_keeps_to_the_named_scheme(tmp_path):
    holdout = pl.DataFrame(
        {
            "combo": ["a", "a"],
            "transform": ["relabund", "relabund"],
            "label": ["ph", "ph"],
            "scheme": ["random", "month"],
            "r2": [0.9, 0.2],
        }
    )
    models = ptw.load_models(_table(tmp_path, _sweep_rows("ph")))
    out = tmp_path / "figs"
    written = ptw.plot_holdout(holdout, models, out, ["png"], scheme="month")
    assert [p.name for p in written] == ["transform_sweep_holdout_ph__month"]

    fig = ptw.draw_holdout("ph", holdout, models, scheme="month")
    assert fig.axes[0].patches[0].get_height() == 0.2
    ptw.plt.close(fig)
    assert ptw.draw_holdout("ph", holdout, models, scheme="absent") is None


def test_a_missing_headline_is_not_fatal(tmp_path):
    assert ptw.label_sizes(tmp_path / "nope.tsv") == {}
    assert ptw.label_sizes(None) == {}


# ---------------------------------------------------------------------------
# Figures across every combination
# ---------------------------------------------------------------------------
def test_the_icc_figure_is_written_and_skips_when_empty(tmp_path):
    tables = [("clr", [0.1, 0.95, 0.2]), ("presence", [0.5])]
    target = ptw.plot_icc_distribution(
        tables, [0.9], "bioproject", tmp_path, ["png"]
    )
    assert (tmp_path / f"{target.name}.png").exists()
    assert ptw.plot_icc_distribution([], [], "x", tmp_path, ["png"]) is None


def test_the_shadow_figure_is_written_and_skips_when_empty(tmp_path):
    floor = pl.DataFrame(
        {
            "combo": ["a", "b"],
            "label": ["ph", "ph"],
            "n_real": [100, 80],
            "n_real_above_q95": [10, 0],
            "pct_above_noise": [10.0, 0.0],
        }
    )
    target = ptw.plot_shadow_floor(floor, tmp_path, ["png"])
    assert (tmp_path / f"{target.name}.png").exists()
    assert ptw.plot_shadow_floor(floor.head(0), tmp_path, ["png"]) is None


def _scoreboard_rows():
    rows = []
    for combo, cv, unpruned, pruned in (
        ("a", 0.5, 0.4, 0.45),
        ("b", 0.3, float("nan"), None),
        ("c", 0.2, -3.5, 0.0),
    ):
        for version, holdout, n in (
            ("unpruned", unpruned, 100),
            ("pruned", pruned, 10),
        ):
            rows.append(
                {
                    "combo": combo,
                    "label": "ph",
                    "scheme": "month",
                    "version": version,
                    "cv_pooled_r2": cv,
                    "holdout_r2": holdout,
                    "n_features": n,
                }
            )
    return rows


def test_the_scoreboard_copes_with_blown_up_and_missing_scores(tmp_path):
    frame = pl.DataFrame(_scoreboard_rows(), infer_schema_length=None)
    target = ptw.plot_scoreboard(frame, tmp_path, ["png"])
    assert (tmp_path / f"{target.name}.png").exists()
    assert ptw.plot_scoreboard(frame.head(0), tmp_path, ["png"]) is None


def test_the_runtime_figure_is_written_even_when_nothing_took_time(tmp_path):
    table = pl.DataFrame({"combo": ["a"], "cv": [0.0], "prune": [0.0]})
    target = ptw.plot_runtime(table, ["cv", "prune"], tmp_path, ["png"])
    assert (tmp_path / f"{target.name}.png").exists()
    assert ptw.plot_runtime(table.head(0), ["cv"], tmp_path, ["png"]) is None
