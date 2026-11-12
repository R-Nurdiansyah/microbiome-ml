"""Sparse feature matrices, and threads for the models that share data."""

import json

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import microbiome_ml.train.cv as cv_module
from microbiome_ml.train.cv import CrossValidator
from microbiome_ml.train.matrix import feature_matrix, sparse_mode
from microbiome_ml.train.trainer import ModelTrainer


def _table(n=40, p=30, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.random((n, p)).astype(np.float32)
    values[rng.random((n, p)) > density] = 0.0
    samples = [f"s{i}" for i in range(n)]
    frame = pl.DataFrame(
        {"sample": samples, **{f"g{j}": values[:, j] for j in range(p)}}
    )
    return frame, values, samples


@pytest.mark.parametrize("batch_rows", [7, 1000])
def test_rows_come_back_in_the_requested_order(batch_rows):
    frame, values, samples = _table()
    wanted = ["s5", "s0", "s39", "s12", "s5"]  # out of order, repeated
    cols = [c for c in frame.columns if c != "sample"]
    X = feature_matrix(frame.lazy(), wanted, cols, batch_rows=batch_rows)
    assert sp.issparse(X)
    assert X.dtype == np.float32
    rows = [samples.index(s) for s in wanted]
    np.testing.assert_array_equal(X.toarray(), values[rows])


def test_a_dense_table_stays_dense():
    frame, values, samples = _table(density=0.9)
    cols = [c for c in frame.columns if c != "sample"]
    X = feature_matrix(frame.lazy(), samples, cols)
    assert isinstance(X, np.ndarray)
    np.testing.assert_array_equal(X, values)


def test_sparse_can_be_forced_either_way():
    frame, values, samples = _table(density=0.9)
    cols = [c for c in frame.columns if c != "sample"]
    assert sp.issparse(feature_matrix(frame.lazy(), samples, cols, sparse=True))
    frame, values, samples = _table(density=0.01)
    dense = feature_matrix(frame.lazy(), samples, cols, sparse=False)
    assert isinstance(dense, np.ndarray)


def test_a_sample_without_features_gets_the_fill_row():
    frame, values, samples = _table()
    cols = [c for c in frame.columns if c != "sample"]
    X = feature_matrix(frame.lazy(), ["s1", "nope"], cols)
    np.testing.assert_array_equal(X.toarray()[0], values[1])
    assert X[1].nnz == 0


def test_nulls_are_filled():
    frame = pl.DataFrame({"sample": ["a", "b"], "g0": [None, 0.5]})
    X = feature_matrix(frame.lazy(), ["a", "b"], ["g0"], sparse=False)
    np.testing.assert_array_equal(X, [[0.0], [0.5]])


@pytest.mark.parametrize(
    "given,mode",
    [(None, "auto"), ("auto", "auto"), (True, "sparse"), ("false", "dense")],
)
def test_sparse_setting(given, mode):
    assert sparse_mode(given) == mode


# --- CV ----------------------------------------------------------------------
class _FeatureSet:
    def __init__(self, frame):
        self.features = frame.lazy()

    def to_df(self):
        return self.features.collect()


class _Dataset:
    def __init__(self, frame, y):
        samples = frame["sample"].to_list()
        self.feature_sets = {"feat": _FeatureSet(frame)}
        self.labels = pl.DataFrame({"sample": samples, "target": list(y)})
        self.splits = {
            "target": {"schemeA": {s: i % 3 for i, s in enumerate(samples)}}
        }

    def iter_labels(self):
        yield ("target", self.labels)

    def iter_cv_folds(self, label=None):
        yield ("target", "schemeA", None)


def _sparse_dataset():
    frame, values, _ = _table(n=60, p=40, density=0.1)
    y = values[:, 0] * 10 + values[:, 1]
    return _Dataset(frame, y)


def test_cv_hands_the_models_a_sparse_matrix():
    ds = _sparse_dataset()
    cv = CrossValidator(ds, models="rf")
    (_, X, y, meta), *_ = list(
        cv._prepare_inputs(ds, label="target", scheme="schemeA")
    )
    assert sp.issparse(X)
    assert X.shape == (60, 40) and len(y) == 60 == len(meta.fold)


def test_sparse_and_dense_cv_agree_for_a_forest(tmp_path):
    """Same model, same folds: only the storage differs."""
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"rf": [{"n_estimators": [10]}]}))
    scores = {}
    for mode in (True, False):
        cv = CrossValidator(
            _sparse_dataset(),
            models="rf",
            label="target",
            scheme="schemeA",
            sparse=mode,
        )
        result = next(iter(cv.run(param_path=str(params), n_jobs=1).values()))
        scores[mode] = result.avg_validation_r2
    assert scores[True] == pytest.approx(scores[False], abs=0.05)


def _capture_cross_validate(monkeypatch):
    seen = []

    def fake(estimator, X, y, **kwargs):
        seen.append((estimator.get_params().get("n_jobs"), kwargs["n_jobs"]))
        n = kwargs["cv"].get_n_splits()
        return {
            "test_r2": np.ones(n),
            "test_mse": -np.ones(n),
            "test_sse": np.ones(n),
            "test_sst": np.full(n, 2.0),
        }

    monkeypatch.setattr(cv_module, "cross_validate", fake)
    return seen


def test_a_forest_takes_the_threads_and_folds_run_in_turn(
    monkeypatch, tmp_path
):
    """One combination: 4 cores go to the forest's own threads (one copy
    of the data), not to 4 fold processes (4 copies)."""
    seen = _capture_cross_validate(monkeypatch)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"rf": [{"n_estimators": [5]}]}))
    cv = CrossValidator(
        _sparse_dataset(), models="rf", label="target", scheme="schemeA"
    )
    cv.run(param_path=str(params), n_jobs=4)
    assert seen == [(4, 1)]


def test_a_model_without_threads_keeps_the_parallel_folds(
    monkeypatch, tmp_path
):
    seen = _capture_cross_validate(monkeypatch)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"linearregression": [{}]}))
    cv = CrossValidator(
        _sparse_dataset(),
        models=LinearRegression(),
        label="target",
        scheme="schemeA",
    )
    cv.run(param_path=str(params), n_jobs=4)
    assert [folds for _, folds in seen] == [4]


# --- holdout -----------------------------------------------------------------
def test_the_holdout_trainer_reads_a_sparse_matrix(tmp_path):
    frame, values, samples = _table(n=40, p=30, density=0.1)
    y = values[:, 0] * 10 + values[:, 1]
    labels = pl.DataFrame({"sample": samples, "target": list(y)})

    class _DS:
        feature_sets = {"feat": _FeatureSet(frame)}

        def get_train_samples(self, label, fold=None, metadata=True):
            return labels.head(30)

        def get_test_samples(self, label, fold=None, metadata=True):
            return labels.tail(10)

    from microbiome_ml.train.results import CV_Result

    result = CV_Result(
        feature_set="feat",
        label="target",
        scheme="random",
        trained_model=RandomForestRegressor(n_estimators=10, random_state=0),
    )
    trainer = ModelTrainer(_DS(), result, tmp_path / "m.pkl")
    X, _, cols = trainer._assemble_split(
        trainer._materialize_feature_set("feat"), labels.head(30), "target"
    )
    assert sp.issparse(X) and X.shape == (30, 30) and len(cols) == 30
    evaluation = trainer.train_and_evaluate()
    assert evaluation.metrics["n_test"] == 10
