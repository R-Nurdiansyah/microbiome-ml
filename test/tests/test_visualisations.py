import json
from pathlib import Path

import matplotlib

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
    vis = Visualiser(results=ndjson, out=tmp_path / "dummy.png")
    vis.plot_cv_bars(out_dir=plots_dir)

    names = {p.name for p in plots_dir.glob("*.png")}
    assert names == {
        "Feature_Set__lbl__schemeA__RandomForestRegressor.png",
        "Feature_Set__lbl__schemeA__XGBRegressor.png",
    }
