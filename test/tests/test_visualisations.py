import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from microbiome_ml.train.results import CV_Result, HoldoutEvaluation
from microbiome_ml.visualise.visualisations import Visualiser

matplotlib.use("Agg")


def _write_records(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_plot_cv_bars_creates_one_file_per_model(tmp_path):
    records = [
        {
            "feature_set": "Feature Set",
            "label": "lbl",
            "scheme": "schemeA",
            "validation_r2_per_fold": [0.1, 0.2],
            "avg_validation_r2": 0.15,
            "avg_validation_mse": 0.05,
            "model": "RandomForestRegressor",
        },
        {
            "feature_set": "Feature Set",
            "label": "lbl",
            "scheme": "schemeA",
            "validation_r2_per_fold": [0.3, 0.4],
            "avg_validation_r2": 0.35,
            "avg_validation_mse": 0.02,
            "model": "XGBRegressor",
        },
    ]
    ndjson = tmp_path / "results.ndjson"
    _write_records(ndjson, records)
    plots_dir = tmp_path / "plots"
    vis = Visualiser(out=tmp_path / "figs")
    vis.plot_cv_bars(results=ndjson, out_dir=plots_dir)

    names = {p.name for p in plots_dir.glob("*.png")}
    assert names == {
        "Feature_Set__lbl__schemeA__RandomForestRegressor.png",
        "Feature_Set__lbl__schemeA__XGBRegressor.png",
    }


def test_visualise_model_performance_saves_holdout(tmp_path):
    vis = Visualiser(out=tmp_path / "figs")
    predictions = np.array([1.0, 2.0, 3.0])
    values = np.array([1.1, 1.9, 3.05])
    vis.visualise_model_performance(
        predictions,
        values,
        file_name="holdout.png",
    )

    assert (tmp_path / "figs" / "holdout.png").exists()


def test_plot_feature_importances_accepts_result_objects(tmp_path):
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = RandomForestRegressor(n_estimators=8, random_state=0)
    model.fit(X, y)

    cv_result = CV_Result(
        feature_set="fset",
        label="target",
        scheme="schemeA",
        trained_model=model,
        feature_names=["f1", "f2"],
    )
    holdout_eval = HoldoutEvaluation(
        metrics={
            "feature_set": "fset",
            "label": "target",
            "scheme": "holdout",
        },
        estimator=model,
        predictions=np.array([0.1, 0.8]),
        targets=np.array([0.0, 1.0]),
        feature_names=["f1", "f2"],
    )

    vis = Visualiser(out=tmp_path / "figs")
    vis.plot_feature_importances(cv_result, output="fi_cv.png")
    vis.plot_feature_importances(holdout_eval, output="fi_holdout.png")

    assert (tmp_path / "figs" / "fi_cv.png").exists()
    assert (tmp_path / "figs" / "fi_holdout.png").exists()


def test_plot_cv_bars_svg_eps_formats(tmp_path):
    records = [
        {
            "feature_set": "FS",
            "label": "lbl",
            "scheme": "schemeA",
            "validation_r2_per_fold": [0.1, 0.2],
            "avg_validation_r2": 0.15,
            "model": "RF",
        }
    ]
    ndjson = tmp_path / "results.ndjson"
    _write_records(ndjson, records)
    plots_dir = tmp_path / "plots"
    vis = Visualiser(out=tmp_path / "figs", formats=["svg", "eps"])
    vis.plot_cv_bars(results=ndjson, out_dir=plots_dir)

    assert (plots_dir / "FS__lbl__schemeA__RF.svg").exists()
    assert (plots_dir / "FS__lbl__schemeA__RF.eps").exists()
    assert not (plots_dir / "FS__lbl__schemeA__RF.png").exists()


def test_visualise_model_performance_svg_eps(tmp_path):
    vis = Visualiser(out=tmp_path / "figs", formats=["svg", "eps"])
    predictions = np.array([1.0, 2.0, 3.0])
    values = np.array([1.1, 1.9, 3.05])
    vis.visualise_model_performance(
        predictions,
        values,
        file_name="holdout.png",
    )

    assert (tmp_path / "figs" / "holdout.svg").exists()
    assert (tmp_path / "figs" / "holdout.eps").exists()
    assert not (tmp_path / "figs" / "holdout.png").exists()


def test_invalid_format_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Unsupported format"):
        Visualiser(out=tmp_path / "figs", formats=["tiff"])


def test_plot_holdout_distribution(tmp_path):
    """Train/test panels render from a holdout.csv, stacked by grouping."""
    import numpy as np
    import polars as pl

    from microbiome_ml.visualise.visualisations import Visualiser

    rng = np.random.default_rng(0)
    n = 120
    df = pl.DataFrame(
        {
            "sample": [f"s{i}" for i in range(n)],
            "split": ["train"] * 96 + ["test"] * 24,
            "ph": rng.normal(7, 0.5, n),
            "bioproject": [f"PRJ{i % 25}" for i in range(n)],
        }
    )
    path = tmp_path / "holdout.csv"
    df.write_csv(path)

    vis = Visualiser(tmp_path, formats=["png"])
    # from a path, with grouping, label inferred as the first non-key column
    vis.plot_holdout_distribution(
        path, grouping="bioproject", output="dist_grouped"
    )
    assert (tmp_path / "dist_grouped.png").exists()

    # from a DataFrame, no grouping, raw counts
    vis.plot_holdout_distribution(
        df, label="ph", stat="count", output="dist_plain"
    )
    assert (tmp_path / "dist_plain.png").exists()


def test_plot_holdout_distribution_validates_input(tmp_path):
    import polars as pl

    from microbiome_ml.visualise.visualisations import Visualiser

    vis = Visualiser(tmp_path, formats=["png"])
    with pytest.raises(ValueError, match="split"):
        vis.plot_holdout_distribution(pl.DataFrame({"sample": ["a"]}))
    with pytest.raises(ValueError, match="not in holdout columns"):
        vis.plot_holdout_distribution(
            pl.DataFrame({"sample": ["a"], "split": ["train"], "ph": [7.0]}),
            grouping="nope",
        )


def test_plot_holdout_distribution_empty_side(tmp_path, recwarn):
    """A label with no test samples must still render, and must not warn
    about an unused palette (hue would be an empty list)."""
    import polars as pl

    from microbiome_ml.visualise.visualisations import Visualiser

    df = pl.DataFrame(
        {
            "sample": ["a", "b", "c"],
            "split": ["train", "train", "train"],
            "ph": [7.0, 7.2, 7.1],
            "bioproject": ["P1", "P1", "P2"],
        }
    )
    vis = Visualiser(tmp_path, formats=["png"])
    stats = vis.plot_holdout_distribution(
        df, grouping="bioproject", output="empty_side"
    )
    assert (tmp_path / "empty_side.png").exists()
    assert stats["n_test"] == 0
    assert stats["ks"] is None
    assert not [w for w in recwarn if "no `hue` variable" in str(w.message)]


def test_collate_split_report_tolerates_string_columns(tmp_path):
    """A run whose stats are all-null reads back as String; the report
    must still sort and format rather than raising."""
    import sys

    import polars as pl

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import pipeline_steps as steps

    good = tmp_path / "holdout_random" / "split_visualisations"
    bad = tmp_path / "holdout_year" / "split_visualisations"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    pl.DataFrame(
        {
            "holdout_grouping": ["random"],
            "label": ["ph"],
            "n_train": [80],
            "n_test": [20],
            "median_train": [7.0],
            "median_test": [7.1],
            "sd_train": [0.2],
            "sd_test": [0.2],
            "ks": [0.05],
            "p": [0.9],
        }
    ).write_csv(good / "holdout_distribution_stats.tsv", separator="\t")
    # all-null numeric columns -> String on read-back
    pl.DataFrame(
        {
            "holdout_grouping": ["year"],
            "label": ["ph"],
            "n_train": [5],
            "n_test": [0],
            "median_train": [None],
            "median_test": [None],
            "sd_train": [None],
            "sd_test": [None],
            "ks": [None],
            "p": [None],
        }
    ).write_csv(bad / "holdout_distribution_stats.tsv", separator="\t")

    out = steps.collate_split_report(tmp_path)
    assert out is not None and out.exists()
    text = out.read_text()
    assert "random" in text and "year" in text
    assert "n/a" in text  # the all-null run renders, not crashes
    # smallest KS first
    assert text.index("random") < text.index("year")


def test_plot_holdout_distribution_multiple_stats(tmp_path):
    """stat can be a list: one row of train/test panels per statistic."""
    import numpy as np
    import polars as pl

    from microbiome_ml.visualise.visualisations import Visualiser

    rng = np.random.default_rng(1)
    n = 80
    df = pl.DataFrame(
        {
            "sample": [f"s{i}" for i in range(n)],
            "split": ["train"] * 64 + ["test"] * 16,
            "ph": rng.normal(7, 0.4, n),
            "bioproject": [f"PRJ{i % 4}" for i in range(n)],
        }
    )
    vis = Visualiser(tmp_path, formats=["png"])

    vis.plot_holdout_distribution(
        df,
        grouping="bioproject",
        stat=["count", "probability"],
        output="two_stats",
    )
    assert (tmp_path / "two_stats.png").exists()

    # a single string still works
    vis.plot_holdout_distribution(df, stat="count", output="one_stat")
    assert (tmp_path / "one_stat.png").exists()

    with pytest.raises(ValueError, match="at least one statistic"):
        vis.plot_holdout_distribution(df, stat=[], output="none")
