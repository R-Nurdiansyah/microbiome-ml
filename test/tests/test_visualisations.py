import json
from pathlib import Path

import matplotlib
import numpy as np
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
