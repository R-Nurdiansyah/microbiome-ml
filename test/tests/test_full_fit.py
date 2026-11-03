"""Full-pool fit diagnostics.

Everything here is duck-typed against the estimator API, so the metrics
are checked against hand-computed values with fake estimators rather
than by training anything. The cases that matter are the ones where a
number must be *absent*: a constant target has no R2, XGBoost has no
out-of-bag score, and a forest with `bootstrap=False` has nothing out of
bag to score. Reporting 0.0 for any of those would read as a finding.
"""

import csv

import numpy as np
import pytest

from microbiome_ml.train.full_fit import (
    FULL_FIT_COLUMNS,
    collect_rows,
    enable_oob,
    full_fit_diagnostics,
    r2,
    supports_oob,
    top_importances,
    write_full_fit_report,
)

Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# One unit out on the last point: SSE = 1, SST = 10, so R2 = 0.9.
PRED = np.array([1.0, 2.0, 3.0, 4.0, 4.0])
X = np.zeros((5, 7))


class FakeEstimator:
    """The slice of the sklearn API this module touches."""

    def __init__(self, pred=None, params=None, importances=None, oob=None):
        self._pred = None if pred is None else np.asarray(pred, dtype=float)
        self._params = dict(params or {})
        if importances is not None:
            self.feature_importances_ = np.asarray(importances, dtype=float)
        if oob is not None:
            self.oob_score_ = oob

    def predict(self, X):
        return self._pred

    def get_params(self, deep=True):
        return dict(self._params)

    def set_params(self, **kwargs):
        self._params.update(kwargs)
        return self


# ---------------------------------------------------------------------------
# r2
# ---------------------------------------------------------------------------
def test_r2_matches_the_hand_computed_value():
    assert r2(Y, PRED) == pytest.approx(0.9)


def test_a_perfect_fit_is_one_and_the_mean_is_zero():
    assert r2(Y, Y) == pytest.approx(1.0)
    assert r2(Y, np.full(5, Y.mean())) == pytest.approx(0.0)


def test_a_constant_target_has_no_r2():
    """SST is zero, so R2 is undefined -- not 0.0, which reads as 'mean'."""
    constant = np.array([3.0, 3.0, 3.0])
    assert r2(constant, constant) is None


def test_r2_refuses_mismatched_shapes_and_empties():
    assert r2(Y, Y[:3]) is None
    assert r2(np.array([]), np.array([])) is None


# ---------------------------------------------------------------------------
# out-of-bag support
# ---------------------------------------------------------------------------
def test_a_bootstrapping_forest_supports_oob():
    est = FakeEstimator(params={"oob_score": False, "bootstrap": True})
    assert supports_oob(est)
    assert enable_oob(est) is True
    assert est.get_params()["oob_score"] is True


def test_bootstrap_off_means_no_oob():
    """Every tree sees every row, so there is nothing out of bag."""
    est = FakeEstimator(params={"oob_score": False, "bootstrap": False})
    assert supports_oob(est) is False
    assert enable_oob(est) is False


def test_an_estimator_without_the_knob_is_refused():
    assert supports_oob(FakeEstimator(params={})) is False  # xgboost-like
    assert supports_oob(object()) is False  # no get_params at all
    assert enable_oob(FakeEstimator(params={"bootstrap": True})) is False


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def test_metrics_against_hand_computed_values():
    got = full_fit_diagnostics(FakeEstimator(PRED), X, Y, cv_score=0.30)
    assert got["in_sample_r2"] == pytest.approx(0.9)
    assert got["in_sample_mse"] == pytest.approx(0.2)
    assert got["in_sample_mae"] == pytest.approx(0.2)
    assert got["residual_max_abs"] == pytest.approx(1.0)
    assert got["n_fit_samples"] == 5
    assert got["n_fit_features"] == 7


def test_the_overfit_gap_is_in_sample_minus_the_cv_score():
    got = full_fit_diagnostics(FakeEstimator(PRED), X, Y, cv_score=0.30)
    assert got["overfit_gap"] == pytest.approx(0.6)


def test_no_cv_score_means_no_gap():
    got = full_fit_diagnostics(FakeEstimator(PRED), X, Y)
    assert got["overfit_gap"] is None
    assert got["cv_pooled_r2"] is None


def test_oob_is_read_when_the_fit_produced_one():
    got = full_fit_diagnostics(FakeEstimator(PRED, oob=0.42), X, Y)
    assert got["oob_r2"] == pytest.approx(0.42)


def test_a_model_without_oob_reports_none_not_zero():
    assert full_fit_diagnostics(FakeEstimator(PRED), X, Y)["oob_r2"] is None


def test_a_nan_oob_becomes_none():
    got = full_fit_diagnostics(FakeEstimator(PRED, oob=float("nan")), X, Y)
    assert got["oob_r2"] is None


def test_a_failing_predict_gives_all_none_not_an_exception():
    """A diagnostic must never be what fails a finished CV run."""

    class Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    got = full_fit_diagnostics(Broken(), X, Y, cv_score=0.3)
    assert got["in_sample_r2"] is None
    assert got["cv_pooled_r2"] == pytest.approx(0.3)


def test_a_wrong_length_prediction_is_refused():
    got = full_fit_diagnostics(FakeEstimator(np.array([1.0, 2.0])), X, Y)
    assert got["in_sample_r2"] is None


def test_every_metric_key_is_always_present():
    """A table built from several rows must keep one shape."""
    expected = set(FULL_FIT_COLUMNS) - {
        "label",
        "scheme",
        "feature_set",
        "model",
    }
    for estimator in (FakeEstimator(PRED), FakeEstimator(np.array([1.0]))):
        assert set(full_fit_diagnostics(estimator, X, Y)) == expected


# ---------------------------------------------------------------------------
# importances
# ---------------------------------------------------------------------------
def test_importances_come_back_ranked():
    est = FakeEstimator(importances=[0.1, 0.5, 0.2, 0.05, 0.0])
    names = [f"g__{i}" for i in range(5)]
    assert top_importances(est, names, top_n=3) == [
        ("g__1", pytest.approx(0.5)),
        ("g__2", pytest.approx(0.2)),
        ("g__0", pytest.approx(0.1)),
    ]


def test_a_model_without_importances_gives_an_empty_list():
    assert top_importances(FakeEstimator(), ["a"], 3) == []


def test_mismatched_importance_length_is_refused():
    """A mislabelled importance is worse than a missing one."""
    est = FakeEstimator(importances=[0.1, 0.2])
    assert top_importances(est, ["a", "b", "c"], 3) == []


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
class FakeResult:
    def __init__(self, label, scheme, model, full_fit, names=()):
        self.label = label
        self.scheme = scheme
        self.feature_set = "tax_genus_pa_bio"
        self.model = model
        self.full_fit = full_fit
        self.feature_names = list(names)


def _diag(cv, in_sample, oob=None):
    return {
        "n_fit_samples": 330,
        "n_fit_features": 5,
        "cv_pooled_r2": cv,
        "in_sample_r2": in_sample,
        "oob_r2": oob,
        "overfit_gap": in_sample - cv,
        "in_sample_mse": 0.01,
        "in_sample_mae": 0.08,
        "residual_mean": 0.0,
        "residual_std": 0.1,
        "residual_max_abs": 0.4,
    }


def _winners():
    names = [f"g__{i}" for i in range(5)]
    return {
        ("ph", "month_logo"): FakeResult(
            "ph",
            "month_logo",
            FakeEstimator(importances=[0.2] * 5),
            _diag(0.386, 0.99),
            names,
        ),
        ("ph", "bioproject_logo"): FakeResult(
            "ph",
            "bioproject_logo",
            FakeEstimator(importances=[0.4, 0.3, 0.1, 0.1, 0.1], oob=0.55),
            _diag(-6.34, 0.94, oob=0.55),
            names,
        ),
    }


def test_the_report_has_one_row_per_winner_in_column_order(tmp_path):
    write_full_fit_report(_winners(), tmp_path / "final_fit", top_n=3)
    with (tmp_path / "final_fit" / "full_fit_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert list(rows[0]) == list(FULL_FIT_COLUMNS)


def test_the_worst_overfit_gap_leads(tmp_path):
    write_full_fit_report(_winners(), tmp_path / "final_fit")
    with (tmp_path / "final_fit" / "full_fit_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scheme"] == "bioproject_logo"


def test_a_winner_without_diagnostics_is_skipped(tmp_path):
    """The normal state with the stage off; not an error."""
    winners = _winners()
    winners[("oxy", "month_logo")] = FakeResult(
        "oxy", "month_logo", FakeEstimator(), None
    )
    write_full_fit_report(winners, tmp_path / "final_fit")
    with (tmp_path / "final_fit" / "full_fit_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["label"] for r in rows] == ["ph", "ph"]


def test_nothing_to_report_writes_nothing(tmp_path):
    target = tmp_path / "final_fit"
    assert write_full_fit_report({}, target) is None
    assert not target.exists(), "an empty directory is a misleading artefact"


def test_importances_are_written_per_winner(tmp_path):
    write_full_fit_report(_winners(), tmp_path / "final_fit", top_n=3)
    with (tmp_path / "final_fit" / "full_fit_importances.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6  # two winners x top 3
    assert rows[0]["rank"] == "1"


def test_collect_rows_reads_the_label_off_the_result_not_the_key():
    rows = collect_rows(
        {
            "whatever": FakeResult(
                "ph", "month_logo", FakeEstimator(), _diag(0.3, 0.9)
            )
        }
    )
    assert rows[0]["label"] == "ph"
    assert rows[0]["scheme"] == "month_logo"
