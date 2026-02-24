import json
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

import microbiome_ml.train.cv as cv_module
from microbiome_ml.train.cv import CrossValidator
from microbiome_ml.train.results import CV_Result


class DummyFeatureSet:
    def __init__(self, samples, X):
        self._samples = samples
        self._X = X

    def to_df(self):
        df = {"sample": self._samples}
        for i in range(self._X.shape[1]):
            df[f"f{i}"] = list(self._X[:, i])
        return pl.DataFrame(df)


class DummyDataset:
    def __init__(self, samples, X, y):
        self.feature_sets = {"feat": DummyFeatureSet(samples, X)}
        self.labels = pl.DataFrame({"sample": samples, "target": list(y)})
        # splits: label -> scheme -> {sample: fold}
        fold_map = {s: int(i % 2) for i, s in enumerate(samples)}
        self.splits = {"target": {"schemeA": fold_map}}

    def iter_labels(self):
        yield ("target", self.labels)

    def iter_cv_folds(self, label=None):
        # yield (label, scheme_name, arbitrary)
        yield ("target", "schemeA", None)


def write_param_yaml(path: Path):
    # provide an entry for LinearRegression so GridSearchCV/run_grid finds a grid
    payload = {"linearregression": [{}]}
    path.write_text(json.dumps(payload))


def test_run_and_run_grid(tmp_path):
    # simple dataset: y = f0 + noise
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(0).randn(6) * 0.01

    ds = DummyDataset(samples, X, y)

    # param file
    p = tmp_path / "params.json"
    write_param_yaml(p)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
    )

    res_all = cv.run(param_path=str(p))
    assert isinstance(res_all, dict)
    assert len(res_all) >= 1
    # values should be CV_Result instances
    vals = list(res_all.values())
    assert isinstance(vals[0], CV_Result)
    assert vals[0].avg_validation_r2 is not None

    res_grid = cv.run_grid(param_path=str(p))
    assert isinstance(res_grid, dict)
    assert len(res_grid) >= 1
    vals2 = list(res_grid.values())
    assert isinstance(vals2[0], CV_Result)
    assert vals2[0].avg_validation_r2 is not None


def test_invalid_folds_are_skipped(tmp_path):
    # create dataset where one fold has fewer than 2 samples -> should be skipped
    samples = ["s0", "s1", "s2"]
    X = np.arange(6).reshape(3, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(1).randn(3) * 0.01

    class SmallFoldDataset(DummyDataset):
        def __init__(self, samples, X, y):
            self.feature_sets = {"feat": DummyFeatureSet(samples, X)}
            self.labels = pl.DataFrame({"sample": samples, "target": list(y)})
            # fold 0 has only one sample -> min count < 2
            fold_map = {"s0": 0, "s1": 1, "s2": 1}
            self.splits = {"target": {"schemeA": fold_map}}

        def iter_labels(self):
            yield ("target", self.labels)

        def iter_cv_folds(self, label=None):
            yield ("target", "schemeA", None)

    ds = SmallFoldDataset(samples, X, y)
    p = tmp_path / "params.json"
    write_param_yaml(p)
    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
    )
    res = cv.run(param_path=str(p))
    assert isinstance(res, dict)
    assert len(res) == 0


def test_params_and_model_saving(tmp_path):
    # verify run respects param grid (multiple param combos -> multiple results)
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(0).randn(6) * 0.01
    ds = DummyDataset(samples, X, y)

    # param file with two parameter combos for LinearRegression
    # use dict-of-lists format so ParameterGrid sees list values
    payload = {"linearregression": {"fit_intercept": [True, False]}}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(payload))

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
    )
    res_run = cv.run(param_path=str(p))
    # two parameter combos should produce two results
    assert isinstance(res_run, dict)
    assert len(res_run) == 2
    for entry in res_run.values():
        assert entry.model is not None

    # run_grid should produce a best estimator we can save/load
    res_grid = cv.run_grid(param_path=str(p))
    assert isinstance(res_grid, dict)
    assert len(res_grid) >= 1
    # best_result should exist (we can use it to save any model)
    assert cv.best_result is not None
    # fit a trivial estimator on the full data and save/load it to test persistence
    est = LinearRegression().fit(X, y)
    out_model = tmp_path / "best_model.pkl"
    cv.best_result.save_model(est, out_model)
    loaded = CV_Result.load_model(out_model)
    # loaded object should be an estimator with predict
    assert hasattr(loaded, "predict")


def test_string_alias_and_mixed_models(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(2).randn(6) * 0.01
    ds = DummyDataset(samples, X, y)

    p = tmp_path / "params.json"
    p.write_text(json.dumps({}))

    cv = CrossValidator(
        ds,
        models=["rf", LinearRegression()],
        cv_folds=2,
        label="target",
        scheme="schemeA",
    )

    res_run = cv.run(param_path=str(p))
    assert isinstance(res_run, dict)
    assert len(res_run) >= 1

    res_grid = cv.run_grid(param_path=str(p))
    assert isinstance(res_grid, dict)
    assert len(res_grid) >= 1


def test_feature_set_filtering(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(6).randn(6) * 0.01

    class MultiFeatDataset(DummyDataset):
        def __init__(self, samples, X, y):
            super().__init__(samples, X, y)
            self.feature_sets = {
                "featA": DummyFeatureSet(samples, X),
                "featB": DummyFeatureSet(samples, X + 1),
            }

    ds = MultiFeatDataset(samples, X, y)
    p = tmp_path / "params.json"
    write_param_yaml(p)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
        feature_set="featB",
    )
    res = cv.run(param_path=str(p))
    assert res  # ensure we have results
    assert all(k.startswith("featB::") for k in res.keys())

    cv_multi = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
        feature_set=["featA", "featB"],
    )
    res_multi = cv_multi.run(param_path=str(p))
    prefixes = {k.split("::")[0] for k in res_multi.keys()}
    assert prefixes == {"featA", "featB"}


def test_blank_param_grid_defaults(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(4).randn(6) * 0.01
    ds = DummyDataset(samples, X, y)

    # Provide explicit key with null value -> should fall back to [{}]
    payload = {"linearregression": None}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(payload))

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
    )
    res = cv.run(param_path=str(p))
    assert isinstance(res, dict)
    assert len(res) == 1


def test_run_parallel_uses_outer_jobs(monkeypatch, tmp_path):
    class TwoSchemeDataset(DummyDataset):
        def __init__(self, samples, X, y):
            super().__init__(samples, X, y)
            fold_a = {s: int(i % 2) for i, s in enumerate(samples)}
            fold_b = {s: int((i + 1) % 2) for i, s in enumerate(samples)}
            self.splits = {"target": {"schemeA": fold_a, "schemeB": fold_b}}

        def iter_cv_folds(self, label=None):
            yield ("target", "schemeA", None)
            yield ("target", "schemeB", None)

    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(5).randn(6) * 0.01
    ds = TwoSchemeDataset(samples, X, y)

    p = tmp_path / "params.json"
    write_param_yaml(p)

    parallel_calls = {"n_jobs": None, "invocations": 0}
    inner_jobs_seen: List[int] = []

    class FakeParallel:
        def __init__(self, n_jobs):
            parallel_calls["n_jobs"] = n_jobs
            parallel_calls["invocations"] += 1

        def __call__(self, iterator):
            tasks = list(iterator)
            return [task() for task in tasks]

    def fake_delayed(func):
        def _wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return _wrapper

    def fake_cross_validate(estimator, X, y, **kwargs):
        inner_jobs_seen.append(kwargs.get("n_jobs"))
        cv = kwargs["cv"]
        n_splits = cv.get_n_splits()
        return {
            "test_r2": np.ones(n_splits),
            "test_mse": -np.ones(n_splits),
        }

    monkeypatch.setattr(cv_module, "Parallel", FakeParallel)
    monkeypatch.setattr(cv_module, "delayed", fake_delayed)
    monkeypatch.setattr(cv_module, "cross_validate", fake_cross_validate)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme=["schemeA", "schemeB"],
    )

    res = cv.run(param_path=str(p), n_jobs=4)
    assert len(res) == 2  # two schemes -> two combos
    assert parallel_calls["invocations"] == 1
    assert parallel_calls["n_jobs"] == 2  # capped by combo count
    assert inner_jobs_seen == [1, 1]  # inner CV forced to sequential


def test_export_result_writes_manifest_and_models(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(3).randn(6) * 0.01
    ds = DummyDataset(samples, X, y)

    payload = {"linearregression": {"fit_intercept": [True]}}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(payload))

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        cv_folds=2,
        label="target",
        scheme="schemeA",
    )
    res = cv.run(param_path=str(p))
    out_dir = tmp_path / "cv_export"
    CV_Result.export_result(res, out_dir)

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "results.ndjson").exists()
    assert (out_dir / "results_summary.csv").exists()
    assert (out_dir / "feature_importances.csv").exists()
    models = list((out_dir / "models").rglob("*.pkl"))
    assert models

    first_result = next(iter(res.values()))
    single_dir = tmp_path / "single_result"
    single_dir.mkdir(parents=True, exist_ok=True)
    CV_Result.save_cv_result(first_result, single_dir)
    assert (single_dir / "results.ndjson").exists()
