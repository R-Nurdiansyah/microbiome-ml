import json
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import pytest
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


@pytest.mark.parametrize(
    "n_jobs,max_combos,expected",
    [
        # One combination: its folds get exactly the budget it was given.
        # This was -1, so a job told 6 cores could take every core it saw.
        (6, 1, (1, 6)),
        (1, 1, (1, 1)),
        # Nothing asked for: joblib counts the cores this job can use.
        (None, 1, (1, -1)),
        # Several combinations fan out instead, never both levels at once.
        (6, 2, (2, 1)),
        (2, 9, (2, 1)),
        (None, 2, (2, 1)),
    ],
)
def test_cv_stays_within_its_core_budget(n_jobs, max_combos, expected):
    assert cv_module._cv_parallelism(n_jobs, max_combos, 128) == expected


@pytest.mark.parametrize(
    "n_jobs,parallel_jobs,inner_cv_jobs,expected",
    [
        (8, 2, 1, 4),  # two combinations share 8 cores
        (16, 4, 1, 4),
        (3, 2, 1, 1),  # never below one thread
        (8, 1, 8, 1),  # the folds already use the budget
        (8, 8, 1, 1),
    ],
)
def test_models_get_their_share_of_the_threads(
    n_jobs, parallel_jobs, inner_cv_jobs, expected
):
    got = cv_module._model_threads(n_jobs, parallel_jobs, inner_cv_jobs)
    assert got == expected


def test_run_gives_each_model_its_threads(tmp_path):
    # XGBoost took every core of the node in each worker; a forest took 1.
    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(0).randn(8) * 0.01
    ds = DummyDataset(samples, X, y)
    ds.feature_sets["feat2"] = DummyFeatureSet(samples, X[:, ::-1])
    p = tmp_path / "params.json"
    p.write_text(json.dumps({"rf": [{"n_estimators": [5]}]}))

    cv = CrossValidator(ds, models="rf", label="target", scheme="schemeA")
    results = cv.run(param_path=str(p), n_jobs=4)

    # 2 combinations x 1 fold worker x 2 threads = the 4 cores given
    assert len(results) == 2
    assert {r.model.get_params()["n_jobs"] for r in results.values()} == {2}


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
        label="target",
        scheme="schemeA",
    )

    res_run = cv.run(param_path=str(p))
    assert isinstance(res_run, dict)
    assert len(res_run) >= 1

    # run_grid needs a key per named model; an empty one means defaults.
    p.write_text(json.dumps({"rf": None}))
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
            "test_sse": np.ones(n_splits),
            "test_sst": np.full(n_splits, 2.0),
        }

    monkeypatch.setattr(cv_module, "Parallel", FakeParallel)
    monkeypatch.setattr(cv_module, "delayed", fake_delayed)
    monkeypatch.setattr(cv_module, "cross_validate", fake_cross_validate)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        label="target",
        scheme=["schemeA", "schemeB"],
    )

    res = cv.run(param_path=str(p), n_jobs=4)
    assert len(res) == 2  # two schemes -> two combos
    assert parallel_calls["invocations"] == 1
    assert parallel_calls["n_jobs"] == 2  # capped by combo count
    assert inner_jobs_seen == [1, 1]  # inner CV forced to sequential


def test_best_result_is_tracked_per_label(tmp_path):
    class TwoLabelDataset(DummyDataset):
        def __init__(self, samples, X, y):
            self.feature_sets = {"feat": DummyFeatureSet(samples, X)}
            self.labels = pl.DataFrame(
                {
                    "sample": samples,
                    "target": list(y),
                    "target2": list(y * 2.0 + 1.0),
                }
            )
            fold_map = {s: int(i % 2) for i, s in enumerate(samples)}
            self.splits = {
                "target": {"schemeA": fold_map},
                "target2": {"schemeA": fold_map},
            }

        def iter_labels(self):
            yield ("target", self.labels.select(["sample", "target"]))
            yield ("target2", self.labels.select(["sample", "target2"]))

        def iter_cv_folds(self, label=None):
            if label in (None, "target"):
                yield ("target", "schemeA", None)
            if label in (None, "target2"):
                yield ("target2", "schemeA", None)

    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(7).randn(8) * 0.01
    ds = TwoLabelDataset(samples, X, y)

    p = tmp_path / "params.json"
    write_param_yaml(p)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        scheme="schemeA",
    )
    res = cv.run(param_path=str(p))

    assert isinstance(res, dict)
    assert len(res) >= 2
    assert cv.best_result is not None
    assert "target" in cv.best_result_by_label
    assert "target2" in cv.best_result_by_label
    assert cv.best_result_by_label["target"].label == "target"
    assert cv.best_result_by_label["target2"].label == "target2"


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


def test_export_best_models_multi_label_creates_label_directories(tmp_path):
    class TwoLabelDataset(DummyDataset):
        def __init__(self, samples, X, y):
            self.feature_sets = {"feat": DummyFeatureSet(samples, X)}
            self.labels = pl.DataFrame(
                {
                    "sample": samples,
                    "target": list(y),
                    "target2": list(y * 2.0 + 1.0),
                }
            )
            fold_map = {s: int(i % 2) for i, s in enumerate(samples)}
            self.splits = {
                "target": {"schemeA": fold_map},
                "target2": {"schemeA": fold_map},
            }

        def iter_labels(self):
            yield ("target", self.labels.select(["sample", "target"]))
            yield ("target2", self.labels.select(["sample", "target2"]))

        def iter_cv_folds(self, label=None):
            if label in (None, "target"):
                yield ("target", "schemeA", None)
            if label in (None, "target2"):
                yield ("target2", "schemeA", None)

    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(8).randn(8) * 0.01
    ds = TwoLabelDataset(samples, X, y)

    p = tmp_path / "params.json"
    write_param_yaml(p)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        scheme="schemeA",
    )
    cv.run(param_path=str(p))

    out_dir = tmp_path / "best_by_label"
    CV_Result.export_best_results(
        cv.best_result_by_label,
        out_dir,
        best_result_key_by_label=cv.best_result_key_by_label,
        fallback_best_result=cv.best_result,
        fallback_best_key=cv.best_result_key or "best_result",
    )

    assert (out_dir / "target" / "manifest.json").exists()
    assert (out_dir / "target2" / "manifest.json").exists()
    assert list((out_dir / "target" / "models").rglob("*.pkl"))
    assert list((out_dir / "target2" / "models").rglob("*.pkl"))


def test_export_best_models_single_label_exports_in_place(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(9).randn(6) * 0.01
    ds = DummyDataset(samples, X, y)

    p = tmp_path / "params.json"
    write_param_yaml(p)

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        label="target",
        scheme="schemeA",
    )
    cv.run(param_path=str(p))

    out_dir = tmp_path / "best_single"
    CV_Result.export_best_results(
        cv.best_result_by_label,
        out_dir,
        best_result_key_by_label=cv.best_result_key_by_label,
        fallback_best_result=cv.best_result,
        fallback_best_key=cv.best_result_key or "best_result",
    )

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "results.ndjson").exists()
    assert list((out_dir / "models").rglob("*.pkl"))


def test_export_best_models_from_full_results_keeps_one_per_label(tmp_path):
    class TwoLabelDataset(DummyDataset):
        def __init__(self, samples, X, y):
            self.feature_sets = {"feat": DummyFeatureSet(samples, X)}
            self.labels = pl.DataFrame(
                {
                    "sample": samples,
                    "target": list(y),
                    "target2": list(y * 2.0 + 1.0),
                }
            )
            fold_map = {s: int(i % 2) for i, s in enumerate(samples)}
            self.splits = {
                "target": {"schemeA": fold_map},
                "target2": {"schemeA": fold_map},
            }

        def iter_labels(self):
            yield ("target", self.labels.select(["sample", "target"]))
            yield ("target2", self.labels.select(["sample", "target2"]))

        def iter_cv_folds(self, label=None):
            if label in (None, "target"):
                yield ("target", "schemeA", None)
            if label in (None, "target2"):
                yield ("target2", "schemeA", None)

    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(10).randn(8) * 0.01
    ds = TwoLabelDataset(samples, X, y)

    p = tmp_path / "params.json"
    # two param combos per label to ensure full results contains >1 per label
    payload = {"linearregression": {"fit_intercept": [True, False]}}
    p.write_text(json.dumps(payload))

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        scheme="schemeA",
    )
    full_results = cv.run(param_path=str(p))

    out_dir = tmp_path / "best_from_full_results"
    CV_Result.export_best_results(full_results, out_dir)

    assert (out_dir / "target" / "manifest.json").exists()
    assert (out_dir / "target2" / "manifest.json").exists()

    target_models = list((out_dir / "target" / "models").rglob("*.pkl"))
    target2_models = list((out_dir / "target2" / "models").rglob("*.pkl"))
    assert len(target_models) == 1
    assert len(target2_models) == 1


def test_export_result_opt_out_models_and_fold_table(tmp_path):
    """save_models=False / fold_table=False skip those artefacts and the
    manifest only lists what was actually written."""
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
        label="target",
        scheme="schemeA",
    )
    res = cv.run(param_path=str(p))
    out_dir = tmp_path / "cv_export_trimmed"
    CV_Result.export_result(res, out_dir, save_models=False, fold_table=False)

    # Kept
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "results.ndjson").exists()
    assert (out_dir / "results_summary.csv").exists()
    assert (out_dir / "feature_importances.csv").exists()

    # Trimmed
    assert not (out_dir / "results_folds.csv").exists()
    assert not (out_dir / "models").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "models" not in manifest["components"]
    assert (
        "results_folds.csv" not in manifest["components"]["results"]["files"]
    )


class TwoSchemeDummyDataset(DummyDataset):
    """DummyDataset with a second CV scheme for the same label."""

    def __init__(self, samples, X, y):
        super().__init__(samples, X, y)
        fold_a = {s: int(i % 2) for i, s in enumerate(samples)}
        fold_b = {s: int(i < len(samples) // 2) for i, s in enumerate(samples)}
        self.splits = {"target": {"schemeA": fold_a, "schemeB": fold_b}}

    def iter_cv_folds(self, label=None):
        yield ("target", "schemeA", None)
        yield ("target", "schemeB", None)


def _two_scheme_cv(tmp_path):
    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] * 1.0 + np.random.RandomState(1).randn(8) * 0.01
    ds = TwoSchemeDummyDataset(samples, X, y)
    p = tmp_path / "params.json"
    p.write_text(json.dumps({"linearregression": {"fit_intercept": [True]}}))
    cv = CrossValidator(ds, models=LinearRegression())
    res = cv.run(param_path=str(p))
    return cv, res


def test_best_result_tracked_per_label_scheme(tmp_path):
    cv, _ = _two_scheme_cv(tmp_path)

    assert set(cv.best_result_by_label_scheme) == {
        ("target", "schemeA"),
        ("target", "schemeB"),
    }
    for (label, scheme), result in cv.best_result_by_label_scheme.items():
        assert result.label == label
        assert result.scheme == scheme
        assert (label, scheme) in cv.best_result_key_by_label_scheme

    # The per-label winner is one of the per-scheme winners.
    assert cv.best_result_by_label["target"] in list(
        cv.best_result_by_label_scheme.values()
    )


def test_export_best_results_by_scheme_layout(tmp_path):
    cv, _ = _two_scheme_cv(tmp_path)
    out_dir = tmp_path / "best_models"

    summary_path = CV_Result.export_best_results_by_scheme(
        cv.best_result_by_label_scheme,
        out_dir,
        best_result_key_by_label_scheme=cv.best_result_key_by_label_scheme,
    )

    for scheme in ("schemeA", "schemeB"):
        scheme_dir = out_dir / "target" / scheme
        assert (scheme_dir / "manifest.json").exists()
        assert (scheme_dir / "results.ndjson").exists()
        assert list((scheme_dir / "models").rglob("*.pkl"))

    assert summary_path == out_dir / "best_models_summary.csv"
    summary = pl.read_csv(summary_path)
    assert summary.height == 2
    assert set(summary["scheme"].to_list()) == {"schemeA", "schemeB"}
    r2 = summary["avg_validation_r2"].to_list()
    assert r2 == sorted(r2, reverse=True)


def test_results_summary_sorted_best_to_worst(tmp_path):
    def _result(r2: float) -> CV_Result:
        return CV_Result(
            feature_set="feat",
            label="target",
            scheme="s",
            validation_r2_per_fold=[r2],
            trained_model=LinearRegression(),
        )

    # Deliberately inserted worst -> best, with one unscored result.
    results = {
        "worst": _result(0.1),
        "mid": _result(0.5),
        "unscored": CV_Result(feature_set="feat", label="target", scheme="s"),
        "best": _result(0.9),
    }
    out_dir = tmp_path / "sorted"
    CV_Result.export_result(results, out_dir, save_models=False)

    summary = pl.read_csv(out_dir / "results_summary.csv")
    r2 = summary["avg_validation_r2"].to_list()
    scored = [v for v in r2 if v is not None]
    assert scored == [0.9, 0.5, 0.1]
    assert r2[-1] is None  # unscored last

    with (out_dir / "results.ndjson").open() as fh:
        ndjson_r2 = [json.loads(line)["avg_validation_r2"] for line in fh]
    assert ndjson_r2 == [0.9, 0.5, 0.1, None]


def test_prepare_inputs_streams_one_matrix_at_a_time(tmp_path):
    """_prepare_inputs is a generator: consuming it must not materialise
    every combination's feature matrix at once (that was ~1 TB on the
    large profiles), and X must come back float32."""
    import inspect

    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(0).randn(8) * 0.01
    ds = DummyDataset(samples, X, y)

    cv = CrossValidator(ds, models=LinearRegression())
    gen = cv._prepare_inputs(ds, label="target", scheme="schemeA")
    assert inspect.isgenerator(gen)

    alive = []
    for key, X_np, y_np, meta in gen:
        assert X_np.dtype == np.float32
        assert X_np.shape[0] == y_np.shape[0] == len(meta.fold)
        assert "sample" not in meta.feature_columns
        assert "fold" not in meta.feature_columns
        alive.append(key)
    assert alive, "generator yielded nothing"


def test_run_still_returns_results_after_streaming(tmp_path):
    """The streaming refactor must not change what run() produces."""
    samples = [f"s{i}" for i in range(8)]
    X = np.arange(16).reshape(8, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(1).randn(8) * 0.01
    ds = DummyDataset(samples, X, y)

    p = tmp_path / "params.json"
    p.write_text(json.dumps({"linearregression": {"fit_intercept": [True]}}))

    cv = CrossValidator(
        ds,
        models=LinearRegression(),
        label="target",
        scheme="schemeA",
    )
    res = cv.run(param_path=str(p), n_jobs=1)
    assert res and all(isinstance(v, CV_Result) for v in res.values())
    assert cv.best_result is not None
    assert cv.best_result_by_label_scheme


def test_fold_minus_one_trains_but_never_validates(tmp_path):
    """PredefinedSplit reads -1 as "always train, never validate".

    Leave-one-group-out parks single-sample groups there, because R2 is
    undefined on one point. Without the >= 0 filter in _prepare_inputs and
    in run_grid, the lone -1 looks like a one-sample fold and the whole
    scheme is silently dropped.
    """
    samples = [f"s{i}" for i in range(7)]
    X = np.arange(14).reshape(7, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(2).randn(7) * 0.01
    ds = DummyDataset(samples, X, y)
    # one training-only sample, then two groups of three
    folds = [-1, 0, 0, 0, 1, 1, 1]
    ds.splits["target"]["schemeA"] = dict(zip(samples, folds))

    p = tmp_path / "params.json"
    write_param_yaml(p)
    cv = CrossValidator(
        ds, models=LinearRegression(), label="target", scheme="schemeA"
    )

    assert cv.run(param_path=str(p)), "run() dropped the scheme"
    assert cv.run_grid(param_path=str(p)), "run_grid() dropped the scheme"


# ---------------------------------------------------------------------------
# Fold-distribution metrics: pooled R2, median, IQR, % negative
# ---------------------------------------------------------------------------
def test_pooled_r2_discounts_a_fold_with_no_variance_to_explain():
    """The failure this exists for, with numbers from a real LOGO run.

    One held-out bioproject had an almost constant geo_temperature:
    SST = 0.008, so its R2 came out at -824,781 and dragged the mean of
    26 folds to -32,730. Weighting each fold by the variance it actually
    had to explain puts the summary back near the median.
    """
    r2 = [-824780.5, -8.5, -4.3, -1.6]
    sse = [6724.6, 180.0, 90.0, 40.0]
    sst = [0.008, 18.9, 17.0, 15.4]

    res = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=r2,
        validation_sse_per_fold=sse,
        validation_sst_per_fold=sst,
    )

    assert res.avg_validation_r2 == pytest.approx(-206198.7, rel=1e-4)
    # 1 - sum(sse)/sum(sst); the catastrophic fold carries 0.02% of the
    # denominator, so it can no longer set the headline number
    assert res.pooled_validation_r2 == pytest.approx(
        1 - sum(sse) / sum(sst), rel=1e-9
    )
    assert res.pooled_validation_r2 > -250
    assert res.n_folds_evaluable == 4


def test_zero_variance_folds_are_excluded_not_scored_zero():
    """sklearn returns 0.0 when a fold's label is constant.

    Counting that as a real 0.0 flatters every summary: in the run above
    six of 26 folds were like this, and dropping them moved the share of
    folds below zero from 77% to 100%.
    """
    res = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=[0.0, -4.0, -2.0, 0.0, 0.5],
        validation_sse_per_fold=[5.0, 8.0, 6.0, 3.0, 2.0],
        validation_sst_per_fold=[0.0, 1.6, 2.0, 0.0, 4.0],
    )

    assert res.n_folds == 5
    assert res.n_folds_evaluable == 3
    # median/IQR/% negative all ignore the two SST=0 folds
    assert res.median_validation_r2 == pytest.approx(-2.0)
    assert res.frac_folds_negative_r2 == pytest.approx(2 / 3)
    # the degenerate folds contribute no SST, so pooled uses the other 3
    assert res.pooled_validation_r2 == pytest.approx(
        1 - (8.0 + 6.0 + 2.0) / (1.6 + 2.0 + 4.0)
    )


def test_fold_metrics_fall_back_without_sse_and_sst():
    """An older run has no SSE/SST; exact 0.0 means degenerate."""
    res = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=[0.0, -4.0, -2.0, 0.5],
    )
    assert res.pooled_validation_r2 is None
    assert res.n_folds_evaluable == 3
    assert res.median_validation_r2 == pytest.approx(-2.0)


def test_no_evaluable_folds_leaves_the_metrics_empty():
    res = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=[0.0, 0.0],
        validation_sse_per_fold=[1.0, 2.0],
        validation_sst_per_fold=[0.0, 0.0],
    )
    assert res.n_folds_evaluable == 0
    assert res.pooled_validation_r2 is None
    assert res.median_validation_r2 is None
    assert res.frac_folds_negative_r2 is None


def test_best_result_is_chosen_on_pooled_not_mean():
    """Selection and the sweep headline must rank on the same number.

    Model A wins on the unweighted mean only because one near-constant
    fold wrecks model B's mean; B explains far more of the variance that
    was actually there, so B is the one to export.
    """
    a = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=[-1.0, -1.0, -1.0],
        validation_sse_per_fold=[20.0, 20.0, 20.0],
        validation_sst_per_fold=[10.0, 10.0, 10.0],
    )
    b = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=[0.8, 0.8, -500.0],
        validation_sse_per_fold=[2.0, 2.0, 5.0],
        validation_sst_per_fold=[10.0, 10.0, 0.01],
    )

    assert a.avg_validation_r2 > b.avg_validation_r2  # mean prefers A
    assert b.pooled_validation_r2 > a.pooled_validation_r2  # pooled prefers B
    assert b.rank_score == b.pooled_validation_r2

    order = CV_Result._sorted_by_avg_r2({"a": a, "b": b})
    assert [k for k, _ in order] == ["b", "a"], "export must follow pooled"


def test_rank_score_falls_back_to_the_mean():
    """A result from an older run has no SSE/SST and must still rank."""
    old = CV_Result(
        feature_set="f",
        label="y",
        scheme="s",
        validation_r2_per_fold=[0.4, 0.6],
    )
    assert old.pooled_validation_r2 is None
    assert old.rank_score == old.avg_validation_r2


def test_n_label_records_the_samples_cv_ran_on(tmp_path):
    """Every score in a row needs its n beside it."""
    samples = [f"s{i}" for i in range(6)]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = X[:, 0] + np.random.RandomState(7).randn(6) * 0.01
    ds = DummyDataset(samples, X, y)

    p = tmp_path / "params.json"
    write_param_yaml(p)
    cv = CrossValidator(
        ds, models=LinearRegression(), label="target", scheme="schemeA"
    )
    res = cv.run(param_path=str(p))

    assert res
    assert all(r.n_label == 6 for r in res.values())
