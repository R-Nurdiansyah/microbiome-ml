"""Tests for SHAPResult statistics that do not require the shap package."""

import matplotlib
import numpy as np
import polars as pl
import pytest

from microbiome_ml.train.shap_analysis import SHAPResult

matplotlib.use("Agg")


def _make_result(with_x: bool = True) -> SHAPResult:
    rng = np.random.default_rng(0)
    n = 50
    x_pos = rng.uniform(0, 1, n)  # higher value -> higher SHAP
    x_neg = rng.uniform(0, 1, n)  # higher value -> lower SHAP
    x_const = np.zeros(n)  # constant feature -> direction undefined
    X = np.column_stack([x_pos, x_neg, x_const])
    shap_values = np.column_stack(
        [
            2.0 * (x_pos - 0.5),
            -1.0 * (x_neg - 0.5),
            rng.normal(0, 1e-3, n),
        ]
    )
    return SHAPResult(
        shap_values=shap_values,
        base_value=0.0,
        feature_names=["pos", "neg", "const"],
        X=X if with_x else None,
    )


def test_direction_sign_and_magnitude():
    result = _make_result()

    assert result.direction[0] > 0.9  # strongly positive
    assert result.direction[1] < -0.9  # strongly negative
    assert np.isnan(result.direction[2])  # constant feature

    # mean_abs_shap ranks the larger-effect feature first regardless of sign
    assert result.top_features(2) == ["pos", "neg"]
    signed = dict(result.top_features_signed(2))
    assert signed["pos"] > 0 and signed["neg"] < 0

    # mean_shap is signed; both centred features average near zero
    assert abs(result.mean_shap[0]) < 0.2
    assert abs(result.mean_shap[1]) < 0.2


def test_direction_nan_without_x():
    result = _make_result(with_x=False)
    assert np.all(np.isnan(result.direction))
    # magnitude stats still work
    assert result.top_features(1) == ["pos"]


def test_save_summary_has_direction_columns(tmp_path):
    result = _make_result()
    path = tmp_path / "shap_summary.csv"
    result.save_summary(path)

    df = pl.read_csv(path)
    assert df.columns == [
        "rank",
        "feature",
        "mean_abs_shap",
        "mean_shap",
        "direction",
    ]
    assert df["feature"].to_list() == ["pos", "neg", "const"]
    assert df["direction"][0] > 0
    assert df["direction"][1] < 0
    assert df["direction"][2] is None  # empty cell -> null


@pytest.mark.parametrize("style", ["bar", "beeswarm"])
def test_plot_shap_summary_uses_direction_and_x(tmp_path, style):
    from microbiome_ml.visualise.visualisations import Visualiser

    result = _make_result()
    vis = Visualiser(tmp_path, formats=["png"])
    vis.plot_shap_summary(result, style=style, top_n=3, output=f"shap_{style}")

    assert (tmp_path / f"shap_{style}.png").exists()
