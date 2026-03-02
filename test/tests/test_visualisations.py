import json
from pathlib import Path

import matplotlib
import numpy as np

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
