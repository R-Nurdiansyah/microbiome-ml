"""Label transforms: fitted per fit, scored in label units, transparent."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from microbiome_ml.train.cv import CrossValidator
from microbiome_ml.train.full_fit import enable_oob, full_fit_diagnostics
from microbiome_ml.transforms.labels import (
    TRANSFORMS,
    TargetTransformed,
    apply_label_limits,
    canonical_target_transform,
    estimator_name,
    label_variants,
    split_label_variant,
    transforms_for,
    wrap_target,
)


def _xy(n=60, low=1.0, high=400.0, seed=0):
    # A fresh generator per call: tests that compare two runs must train
    # both on the same data (a shared generator gave each its own draw).
    X = np.random.default_rng(seed).uniform(0, 1, size=(n, 3))
    y = low + (high - low) * X[:, 0] ** 2
    return X, y


# --- names -----------------------------------------------------------------
@pytest.mark.parametrize(
    "given,expected",
    [
        (None, "none"),
        ("", "none"),
        ("Yeo-Johnson", "yeo_johnson"),
        ("yj", "yeo_johnson"),
        ("standardize", "zscore"),
        ("Box Cox", "box_cox"),
        ("quantile", "quantile_normal"),
        ("arrhenius", "inv_kelvin"),
    ],
)
def test_transform_names_and_aliases(given, expected):
    assert canonical_target_transform(given) == expected


def test_unknown_transform_is_an_error():
    with pytest.raises(ValueError, match="Unknown target transform"):
        canonical_target_transform("cube_root")


# --- the wrapper -----------------------------------------------------------
@pytest.mark.parametrize("name", [t for t in TRANSFORMS if t != "none"])
def test_predictions_come_back_in_label_units(name):
    """1-NN repeats a training label exactly on the transformed scale, so
    predicting the training rows must give the labels back, in °C, not
    in z-scores or log units."""
    X, y = _xy()
    model = TargetTransformed(
        KNeighborsRegressor(n_neighbors=1), transform=name
    )
    pred = model.fit(X, y).predict(X)
    tol = 2.0 if name.startswith("quantile") else 1e-6
    np.testing.assert_allclose(pred, y, atol=tol)


def test_the_transform_is_fitted_on_the_rows_it_is_given():
    """Fitted inside fit(): a CV fold's validation rows never shape it."""
    X, y = _xy()
    model = TargetTransformed(KNeighborsRegressor(), transform="zscore")
    model.fit(X[:30], y[:30])
    assert model.transformer_.mean_[0] == pytest.approx(y[:30].mean())


def test_none_leaves_the_estimator_alone():
    rf = RandomForestRegressor()
    assert wrap_target(rf, "none") is rf
    assert wrap_target(rf, None) is rf


def test_the_wrapper_is_transparent_to_the_pipeline():
    wrapped = wrap_target(RandomForestRegressor(random_state=0), "zscore")
    # reports name the regressor
    assert estimator_name(wrapped) == "RandomForestRegressor"
    # the regressor's own parameter names, without `regressor__`
    assert "n_jobs" in wrapped.get_params()
    wrapped.set_params(max_depth=3, n_estimators=7)
    assert wrapped.regressor.max_depth == 3
    # a clone keeps the transform and the settings
    twin = clone(wrapped)
    assert twin.transform == "zscore"
    assert twin.regressor.n_estimators == 7
    # fitted attributes are the regressor's
    X, y = _xy()
    wrapped.fit(X, y)
    assert len(wrapped.feature_importances_) == X.shape[1]


def test_grid_keys_need_no_prefix():
    X, y = _xy()
    wrapped = wrap_target(RandomForestRegressor(random_state=0), "log1p")
    gs = GridSearchCV(wrapped, {"max_depth": [2, 4]}, cv=3).fit(X, y)
    assert set(gs.best_params_) == {"max_depth"}


def test_out_of_bag_score_is_in_label_units():
    X, y = _xy(n=200)
    wrapped = wrap_target(
        RandomForestRegressor(n_estimators=50, random_state=0), "log1p"
    )
    assert enable_oob(wrapped)
    wrapped.fit(X, y)
    inner = wrapped.regressor_
    back = np.expm1(inner.oob_prediction_)
    expected = 1 - ((y - back) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert wrapped.oob_score_ == pytest.approx(expected)
    # and the full-fit report reads it, with the in-sample R2 on y itself
    diag = full_fit_diagnostics(wrapped, X, y)
    assert diag["oob_r2"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "name,bad", [("log1p", -2.0), ("sqrt", -1.0), ("box_cox", 0.0)]
)
def test_values_outside_a_transform_are_refused(name, bad):
    X, y = _xy()
    y[0] = bad
    with pytest.raises(ValueError, match=name):
        TargetTransformed(KNeighborsRegressor(), transform=name).fit(X, y)


# --- CV -----------------------------------------------------------------------
class _FeatureSet:
    def __init__(self, samples, X):
        self._samples, self._X = samples, X

    def to_df(self):
        cols = {f"f{i}": self._X[:, i] for i in range(self._X.shape[1])}
        return pl.DataFrame({"sample": self._samples, **cols})


class _Dataset:
    def __init__(self, samples, X, y):
        self.feature_sets = {"feat": _FeatureSet(samples, X)}
        self.labels = pl.DataFrame({"sample": samples, "target": list(y)})
        folds = {s: i % 3 for i, s in enumerate(samples)}
        self.splits = {"target": {"schemeA": folds}}

    def iter_labels(self):
        yield ("target", self.labels)

    def iter_cv_folds(self, label=None):
        yield ("target", "schemeA", None)


def _cv(tmp_path, transform, mode="run", model="rf"):
    samples = [f"s{i}" for i in range(60)]
    X, y = _xy()
    params = tmp_path / "params.json"
    if model == "rf":
        params.write_text(json.dumps({"rf": [{"n_estimators": [20]}]}))
        models = "rf"
    else:
        params.write_text(json.dumps({"linearregression": [{}]}))
        models = LinearRegression()
    cv = CrossValidator(
        _Dataset(samples, X, y),
        models=models,
        label="target",
        scheme="schemeA",
        target_transform=transform,
    )
    run = cv.run if mode == "run" else cv.run_grid
    return next(iter(run(param_path=str(params), n_jobs=1).values()))


@pytest.mark.parametrize("mode", ["run", "run_grid"])
def test_cv_scores_are_in_label_units(tmp_path, mode):
    """Least squares on a z-scored label gives the same predictions once
    converted back, exactly, so the MSE in label units must match. Scored
    on the z-scale it would be smaller by the label's variance (~14,000
    here)."""
    plain = _cv(tmp_path, None, mode, model="linear")
    zscored = _cv(tmp_path, "zscore", mode, model="linear")
    assert isinstance(zscored.model, TargetTransformed)
    assert zscored.avg_validation_mse == pytest.approx(
        plain.avg_validation_mse, rel=1e-6
    )


def test_forest_scores_are_in_label_units_too(tmp_path):
    """Not asserted equal: the forest's trees on a z-scored label came out
    different from the plain ones in practice (MSE 160 vs 149 here). In
    label units both are the same size; on the z-scale one would be
    ~14,000 times smaller."""
    plain = _cv(tmp_path, None).avg_validation_mse
    zscored = _cv(tmp_path, "zscore").avg_validation_mse
    assert 0.5 < zscored / plain < 2.0


def test_cv_reports_the_regressor_name(tmp_path):
    result = _cv(tmp_path, "yeo_johnson")
    assert estimator_name(result.model) == "RandomForestRegressor"


# --- config: variants and limits ---------------------------------------------
def test_transforms_for_a_list_applies_to_every_label():
    assert transforms_for("ph", ["none", "zscore"]) == ["none", "zscore"]
    assert transforms_for("ph", None) == ["none"]


def test_transforms_for_a_mapping_matches_geo_and_comb_versions():
    spec = {
        "default": ["none", "zscore"],
        "oxy": ["none", "log1p"],
        "comb_oxy": ["none"],
    }
    assert transforms_for("geo_oxy", spec) == ["none", "log1p"]
    assert transforms_for("comb_oxy", spec) == ["none"]  # exact beats suffix
    assert transforms_for("ph", spec) == ["none", "zscore"]


def test_label_variants_round_trip():
    names = label_variants("geo_ph", ["none", "Yeo-Johnson"])
    assert names == ["geo_ph", "geo_ph@yeo_johnson"]
    assert split_label_variant("geo_ph@yeo_johnson") == (
        "geo_ph",
        "yeo_johnson",
    )
    assert split_label_variant("geo_ph") == ("geo_ph", "none")


def test_label_limits_drop_impossible_values():
    labels = pl.DataFrame(
        {
            "sample": ["a", "b", "c", "d"],
            "temperature": [-5.0, 130.0, 20.0, None],
            "geo_oxy": [-1.0, 300.0, 700.0, 10.0],
            "ph": [7.0, 15.0, 8.0, 6.0],
        }
    )
    limits = {"temperature": [-30, 122], "oxy": [0, None], "ph": [0, 14]}
    out, removed = apply_label_limits(labels, limits)
    assert out["temperature"].to_list() == [-5.0, None, 20.0, None]
    # `oxy` covers geo_oxy; an open upper side keeps 700
    assert out["geo_oxy"].to_list() == [None, 300.0, 700.0, 10.0]
    assert out["ph"].to_list() == [7.0, None, 8.0, 6.0]
    assert removed == {"temperature": 1, "geo_oxy": 1, "ph": 1}


# --- the cv step's variant ---------------------------------------------------
def test_a_variant_trains_on_its_base_label_and_split():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import pipeline_steps as steps
    from microbiome_ml.wrangle.splits import SplitManager

    class _DS:
        pass

    ds = _DS()
    ds.labels = pl.DataFrame({"sample": ["a", "b"], "ph": [7.0, 8.0]})
    manager = SplitManager("ph")
    ds.splits = {"ph": manager}

    out = steps._as_variant(ds, "ph", "ph@zscore")
    assert out.labels["ph@zscore"].to_list() == [7.0, 8.0]
    assert out.splits == {"ph@zscore": manager}
    assert manager.label == "ph@zscore"
