"""Which tables the feature report reads, and what dtype it gets.

The same basename is written in three places per (label, scheme), and
reading all of them is not merely wasteful: the holdout copy's
`pooled_validation_r2` column is empty, an all-empty CSV column parses
as text, and one text column turns the whole concatenated score column
into strings. That made `max()` alphabetical -- naming the wrong best
rank -- before it crashed on `last - first`.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import rank_feature_report as rfr  # noqa: E402

CV_ROWS = {
    "label": ["ph", "ph"],
    "scheme": ["bioproject_logo", "bioproject_logo"],
    "model": ["RandomForestRegressor", "XGBRegressor"],
    "feature_set": ["tax_genus", "tax_genus"],
    "pooled_validation_r2": [-6.3, 0.2],
}


def _run_dir(root: Path, label: str, scheme: str) -> Path:
    return root / f"holdout_bioproject__{label}" / "runs" / label / scheme


def _tree(root: Path, label: str = "ph", scheme: str = "bioproject_logo"):
    """A run directory with all three results_summary.csv copies."""
    run = _run_dir(root, label, scheme)
    cv = run / "cv_results"
    best = run / "best_models" / label / scheme
    holdout = run / "holdout"
    for d in (cv, best, holdout):
        d.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(CV_ROWS).write_csv(cv / "results_summary.csv")
    # best_models repeats the winning row.
    pl.DataFrame({k: v[1:] for k, v in CV_ROWS.items()}).write_csv(
        best / "results_summary.csv"
    )
    # The holdout table has the column but never fills it in.
    pl.DataFrame(
        {
            "label": [label],
            "scheme": [scheme],
            "model": ["RandomForestRegressor"],
            "feature_set": ["tax_genus"],
            "pooled_validation_r2": [None],
            "r2": [0.11],
        }
    ).write_csv(holdout / "results_summary.csv")
    return run


def test_only_the_cv_export_is_read(tmp_path):
    _tree(tmp_path)
    paths = rfr._cv_exports(tmp_path, "results_summary.csv")
    assert [p.parent.name for p in paths] == ["cv_results"]


def test_the_holdout_table_is_never_read(tmp_path):
    _tree(tmp_path)
    paths = rfr._cv_exports(tmp_path, "results_summary.csv")
    assert not any(p.parent.name == "holdout" for p in paths)


def test_the_best_models_duplicate_is_not_counted_twice(tmp_path):
    _tree(tmp_path)
    frame = rfr._load(tmp_path, "results_summary.csv")
    assert frame.height == 2, "the winning row was counted twice"


def test_the_score_column_stays_numeric(tmp_path):
    """The regression: an empty column elsewhere must not make it text."""
    _tree(tmp_path)
    frame = rfr._load(tmp_path, "results_summary.csv")
    assert frame.schema["pooled_validation_r2"] == pl.Float64


def test_the_coarse_to_fine_difference_can_be_computed(tmp_path):
    """`last - first` raised TypeError when the column came back text."""
    _tree(tmp_path)
    frame = rfr._load(tmp_path, "results_summary.csv")
    values = frame["pooled_validation_r2"].to_list()
    assert isinstance(values[-1] - values[0], float)


def test_the_best_score_is_chosen_by_value_not_spelling(tmp_path):
    """Alphabetically "-6.3" > "0.2"; numerically it is far worse."""
    _tree(tmp_path)
    frame = rfr._load(tmp_path, "results_summary.csv")
    assert frame["pooled_validation_r2"].max() == 0.2


def test_a_tree_without_cv_results_still_reports(tmp_path):
    """Older output has no cv_results/; it should degrade, not vanish."""
    run = _run_dir(tmp_path, "ph", "bioproject_logo")
    (run / "holdout").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(CV_ROWS).write_csv(run / "results_summary.csv")
    pl.DataFrame({**CV_ROWS, "pooled_validation_r2": [None, None]}).write_csv(
        run / "holdout" / "results_summary.csv"
    )
    paths = rfr._cv_exports(tmp_path, "results_summary.csv")
    assert len(paths) == 1
    assert paths[0].parent.name != "holdout"
