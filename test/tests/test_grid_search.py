"""The grid search: what it selects, what it reports, and that it repeats."""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import plot_grid_search as pgs  # noqa: E402
import run_grid_search as rgs  # noqa: E402

import microbiome_ml.train.cv as cv_module  # noqa: E402
from microbiome_ml.train.cv import CrossValidator  # noqa: E402
from microbiome_ml.train.results import (  # noqa: E402
    CV_Result,
    best_candidate_index,
    pooled_r2_per_candidate,
)


class _Features:
    def __init__(self, samples, X):
        self.samples, self.X = samples, X

    def to_df(self):
        columns = {f"f{i}": self.X[:, i] for i in range(self.X.shape[1])}
        return pl.DataFrame({"sample": self.samples, **columns})


class _Dataset:
    """Two folds over one label: enough for a real GridSearchCV."""

    def __init__(self, n=30, seed=0):
        rng = np.random.RandomState(seed)
        samples = [f"s{i}" for i in range(n)]
        X = rng.rand(n, 4)
        y = 3 * X[:, 0] + rng.normal(0, 0.2, n)
        self.feature_sets = {"feat": _Features(samples, X)}
        self.labels = pl.DataFrame({"sample": samples, "target": y.tolist()})
        folds = {s: i % 3 for i, s in enumerate(samples)}
        self.splits = {"target": {"schemeA": folds}}

    def iter_labels(self):
        yield ("target", self.labels)

    def iter_cv_folds(self, label=None):
        yield ("target", "schemeA", None)


def _cv_results(sse, sst, mean_r2=None):
    sse, sst = np.asarray(sse, float), np.asarray(sst, float)
    out = {f"split{i}_test_sse": sse[:, i] for i in range(sse.shape[1])} | {
        f"split{i}_test_sst": sst[:, i] for i in range(sst.shape[1])
    }
    out["mean_test_r2"] = np.asarray(
        mean_r2 if mean_r2 is not None else [0.0] * len(sse), float
    )
    return out


def _params(tmp_path, payload):
    path = tmp_path / "grids.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_selection_is_by_pooled_r2_not_the_mean_of_folds():
    # Candidate 0 wins the mean only through a fold with almost no variance;
    # candidate 1 explains far more of what there is to explain.
    results = _cv_results(
        sse=[[900.0, 0.1], [300.0, 5.0]],
        sst=[[1000.0, 1.0], [1000.0, 1.0]],
        mean_r2=[0.5, -1.65],
    )
    assert np.argmax(results["mean_test_r2"]) == 0
    assert best_candidate_index(results) == 1


def test_a_fold_with_no_variance_does_not_count():
    results = _cv_results(sse=[[50.0, 7.0]], sst=[[100.0, 0.0]])
    assert pooled_r2_per_candidate(results)[0] == pytest.approx(0.5)


def test_a_failed_candidate_cannot_win():
    results = _cv_results(
        sse=[[np.nan, np.nan], [90.0, 90.0]],
        sst=[[np.nan, np.nan], [100.0, 100.0]],
    )
    assert np.isnan(pooled_r2_per_candidate(results)[0])
    assert best_candidate_index(results) == 1


def test_no_pooled_score_at_all_falls_back_to_the_mean():
    results = _cv_results(
        sse=[[1.0], [2.0]], sst=[[0.0], [0.0]], mean_r2=[0.0, 0.3]
    )
    assert best_candidate_index(results) == 1


def _search(tmp_path, n_jobs=1):
    cv = CrossValidator(
        _Dataset(),
        models=DecisionTreeRegressor(random_state=0),
        label="target",
        scheme="schemeA",
    )
    path = _params(
        tmp_path, {"decisiontreeregressor": {"max_depth": [1, 2, 3]}}
    )
    return cv.run_grid(param_path=path, n_jobs=n_jobs)


def test_every_candidate_reaches_the_report(tmp_path):
    results = _search(tmp_path)
    result = next(iter(results.values()))
    CV_Result.export_result(results, tmp_path / "out", save_models=False)

    table = pl.read_csv(tmp_path / "out" / "grid_results.csv")
    assert table.height == 3
    assert set(table.get_column("param_max_depth")) == {1, 2, 3}
    assert sorted(table.get_column("rank_pooled_r2")) == [1, 2, 3]

    selected = table.filter(pl.col("selected"))
    assert selected.height == 1
    assert selected["rank_pooled_r2"][0] == 1
    assert selected["param_max_depth"][0] == result.best_params["max_depth"]
    assert selected["pooled_r2"][0] == pytest.approx(
        result.pooled_validation_r2
    )


def test_a_tie_ranks_the_selected_candidate_first(tmp_path):
    cvres = _cv_results(sse=[[50.0], [50.0]], sst=[[100.0], [100.0]])
    cvres.update(
        params=[{"a": 1}, {"a": 2}],
        param_a=np.ma.masked_array([1, 2], mask=[False, False]),
        std_test_r2=[0.0, 0.0],
        mean_test_mse=[-1.0, -1.0],
        mean_fit_time=[0.1, 0.1],
        mean_score_time=[0.1, 0.1],
    )
    result = CV_Result(
        feature_set="f", label="l", scheme="s", grid_results=cvres
    )
    path = tmp_path / "grid.csv"
    assert CV_Result.export_grid_results({"k": result}, path) == 2

    table = pl.read_csv(path)
    top = table.filter(pl.col("rank_pooled_r2") == 1)
    assert top["selected"][0] is True
    assert top["param_a"][0] == 1
    # sklearn's sign, unchanged: MSE negated so that higher is better.
    assert table["mean_test_mse"][0] == -1.0


def test_every_column_sklearn_reports_is_kept(tmp_path):
    results = _search(tmp_path)
    cvres = next(iter(results.values())).grid_results
    CV_Result.export_result(results, tmp_path / "out", save_models=False)

    table = pl.read_csv(tmp_path / "out" / "grid_results.csv")
    assert set(cvres) <= set(table.columns)
    assert {"split0_test_r2", "rank_test_r2", "std_fit_time", "params"} <= set(
        table.columns
    )
    assert table.columns[:7] == [
        "feature_set",
        "label",
        "scheme",
        "model",
        "selected",
        "rank_pooled_r2",
        "pooled_r2",
    ]


def test_a_none_setting_counts_as_one_of_the_values_tried():
    grid = pl.DataFrame(
        {
            "model": ["RandomForestRegressor"] * 2 + ["XGBRegressor"] * 2,
            "param_max_samples": [0.5, None, None, None],
            "param_max_depth": [None, None, 4, 4],
        }
    )
    assert pgs._varying_params(grid) == {
        "RandomForestRegressor": {"max_samples"},
        "XGBRegressor": set(),
    }


def test_the_model_kept_is_the_candidate_selected(tmp_path):
    result = next(iter(_search(tmp_path).values()))
    assert result.model.get_params()["max_depth"] == (
        result.best_params["max_depth"]
    )
    assert result.model.tree_ is not None


def test_winning_params_are_real_json_in_the_tables(tmp_path):
    results = _search(tmp_path)
    expected = next(iter(results.values())).best_params
    CV_Result.export_result(results, tmp_path / "out", save_models=False)

    record = json.loads(
        (tmp_path / "out" / "results.ndjson")
        .read_text("utf-8")
        .splitlines()[0]
    )
    assert record["best_params"] == expected
    summary = pl.read_csv(tmp_path / "out" / "results_summary.csv")
    assert json.loads(summary["best_params"][0]) == expected


def test_a_run_without_a_grid_writes_no_grid_table(tmp_path):
    CV_Result.export_result(
        {"k": CV_Result(feature_set="f", label="l", scheme="s")},
        tmp_path,
        save_models=False,
    )
    assert not (tmp_path / "grid_results.csv").exists()


def test_a_model_missing_from_the_grids_is_an_error(tmp_path):
    cv = CrossValidator(
        _Dataset(), models="rf", label="target", scheme="schemeA"
    )
    with pytest.raises(ValueError, match="No grid for model 'rf'"):
        cv.run_grid(param_path=_params(tmp_path, {"xgb": {}}), n_jobs=1)


def test_an_empty_key_means_sklearn_defaults(tmp_path):
    cv = CrossValidator(
        _Dataset(), models="rf", label="target", scheme="schemeA"
    )
    results = cv.run_grid(param_path=_params(tmp_path, {"rf": None}), n_jobs=1)
    assert next(iter(results.values())).best_params == {}


def test_a_second_run_gives_the_same_scores(tmp_path):
    def scores():
        cv = CrossValidator(
            _Dataset(), models="rf", label="target", scheme="schemeA"
        )
        path = _params(tmp_path, {"rf": {"n_estimators": [5, 10]}})
        result = next(iter(cv.run_grid(param_path=path, n_jobs=1).values()))
        return result.validation_r2_per_fold

    assert scores() == scores()


# SVR (libsvm) is deterministic and has no random_state. Listed by name, so a
# new stochastic model cannot slip through unseeded.
_DETERMINISTIC = {"svm"}


@pytest.mark.parametrize(
    "model", sorted(set(cv_module._MODEL_CONSTRUCTORS) - _DETERMINISTIC)
)
def test_every_named_model_is_seeded(model):
    params = cv_module._MODEL_CONSTRUCTORS[model]().get_params()
    assert params["random_state"] == cv_module.RANDOM_STATE


def test_the_svm_has_nothing_to_seed():
    params = cv_module._MODEL_CONSTRUCTORS["svm"]().get_params()
    assert not any(k.endswith("random_state") for k in params)


def test_threads_go_to_the_refit_and_not_the_search(tmp_path, monkeypatch):
    seen = []
    real = cv_module.GridSearchCV

    def spy(estimator, **kwargs):
        seen.append((estimator.get_params()["n_jobs"], kwargs["n_jobs"]))
        return real(estimator, **kwargs)

    monkeypatch.setattr(cv_module, "GridSearchCV", spy)
    cv = CrossValidator(
        _Dataset(), models="rf", label="target", scheme="schemeA"
    )
    path = _params(tmp_path, {"rf": {"n_estimators": [5]}})
    result = next(iter(cv.run_grid(param_path=path, n_jobs=2).values()))

    assert seen == [(1, 2)]
    assert result.model.get_params()["n_jobs"] == 2


def test_an_estimator_passed_in_is_left_as_it_was(tmp_path):
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    cv = CrossValidator(
        _Dataset(),
        models=model,
        label="target",
        scheme="schemeA",
        full_fit=True,
    )
    cv.run_grid(
        param_path=_params(tmp_path, {"rf": {"max_depth": [2]}}), n_jobs=2
    )
    assert model.get_params()["n_jobs"] is None
    assert model.get_params()["oob_score"] is False


def test_oob_is_asked_of_the_winner_only(tmp_path):
    cv = CrossValidator(
        _Dataset(),
        models="rf",
        label="target",
        scheme="schemeA",
        full_fit=True,
    )
    path = _params(tmp_path, {"rf": {"n_estimators": [20]}})
    result = next(iter(cv.run_grid(param_path=path, n_jobs=1).values()))
    assert hasattr(result.model, "oob_score_")
    assert result.full_fit["oob_r2"] is not None


@pytest.mark.parametrize(
    "alias",
    [
        "svm",
        "SVR",
        "linear svr",
        "linear svm",
        "support vector machine",
        "Support-Vector Regression",
    ],
)
def test_svm_answers_to_its_usual_names(alias):
    assert cv_module._resolve_model_alias(alias) == "svm"


def test_svm_runs_the_grid_search_like_the_other_models(tmp_path):
    cv = CrossValidator(
        _Dataset(),
        models="svm",
        label="target",
        scheme="schemeA",
        full_fit=True,
    )
    path = _params(
        tmp_path, {"svm": {"svr__C": [1, 10], "svr__epsilon": [0.1]}}
    )
    result = next(iter(cv.run_grid(param_path=path, n_jobs=1).values()))

    # Named for itself in every report, not "Pipeline".
    assert type(result.model).__name__ == "ScaledSVR"
    # SVR with a linear kernel, not LinearSVR
    assert type(result.model.named_steps["svr"]).__name__ == "SVR"
    assert result.model.get_params()["svr__kernel"] == "linear"
    assert len(result.grid_results["params"]) == 2
    assert result.model.get_params()["svr__C"] in (1, 10)
    # No OOB and no importances to give, and neither stops the run.
    assert result.full_fit["oob_r2"] is None
    # The label is 3 x the first feature plus noise: a linear model must find it.
    assert result.full_fit["in_sample_r2"] > 0.5


def test_svm_settings_survive_a_clone():
    from sklearn.base import clone

    model = cv_module.ScaledSVR().set_params(svr__C=5, svr__epsilon=0.3)
    copy = clone(model)
    assert copy.get_params()["svr__C"] == 5
    assert copy.get_params()["svr__epsilon"] == 0.3
    assert copy.steps[1][1] is not model.steps[1][1]


def test_svm_scaling_keeps_sparse_input_sparse():
    from scipy import sparse

    rng = np.random.RandomState(0)
    X = rng.rand(40, 6) * (rng.rand(40, 6) < 0.2)
    y = X[:, 0] * 10
    model = cv_module.ScaledSVR().fit(sparse.csr_matrix(X), y)
    assert sparse.issparse(
        model.named_steps["scale"].transform(sparse.csr_matrix(X))
    )
    assert np.allclose(model.predict(sparse.csr_matrix(X)), model.predict(X))


def test_a_plain_pipeline_is_not_taken_for_the_svm(tmp_path):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(), DecisionTreeRegressor(random_state=0)
    )
    cv = CrossValidator(
        _Dataset(), models=model, label="target", scheme="schemeA"
    )
    # The svm grid would fail on this pipeline: it has no `svr` step.
    path = _params(tmp_path, {"svm": {"svr__C": [1, 10]}})
    result = next(iter(cv.run_grid(param_path=path, n_jobs=1).values()))
    assert len(result.grid_results["params"]) == 1


_SPLIT_SCRIPT = """
import polars as pl
from microbiome_ml.wrangle.splits import SplitManager

low = [(f"lo{i}", float(i % 3), f"g{i // 2}") for i in range(20)]
high = [(f"hi{i}", 100.0 + i % 3, "big") for i in range(20)]
rows = low + high
data = pl.DataFrame(
    {
        "sample": [r[0] for r in rows],
        "target": [r[1] for r in rows],
        "grp": [r[2] for r in rows],
    }
)
manager = SplitManager("target")
manager.create_holdout(data, grouping="grp", test_size=0.25, n_bins=2)
test = manager.holdout.filter(pl.col("split") == "test")["sample"].to_list()
print(",".join(sorted(test)))
"""


def test_a_seed_gives_one_holdout_whatever_the_hash_seed():
    """Equal-sized groups tie, and a set's order changes per process."""

    def holdout(hash_seed):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        return subprocess.run(
            [sys.executable, "-c", _SPLIT_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    first = holdout("1")
    assert len(first.split(",")) == 10
    assert all(holdout(seed) == first for seed in ("2", "3", "4", "5"))


@pytest.mark.parametrize("dataset", rgs.DATASETS)
def test_a_config_names_only_its_own_dataset(dataset, tmp_path):
    text = rgs.resolve_config(
        rgs.TEMPLATE.read_text(encoding="utf-8"), dataset, tmp_path
    )
    assert "{" + "dataset" + "}" not in text
    cfg = yaml.safe_load(text)

    paths = [
        cfg["data"]["metadata"],
        cfg["data"]["attributes"],
        cfg["data"]["profiles"],
        *cfg["data"]["labels"].values(),
    ]
    assert all(f"/{dataset}/" in path for path in paths)
    assert cfg["sweep"]["out"].endswith(f"/{dataset}")
    others = [d for d in rgs.DATASETS if d != dataset]
    assert not any(other in text for other in others)


def test_the_config_uses_every_sample_unless_capped(tmp_path):
    template = rgs.TEMPLATE.read_text(encoding="utf-8")

    full = rgs.build_config(template, "ITS_soil", tmp_path)
    assert "subset" not in full["sweep"]
    capped = rgs.build_config(template, "ITS_soil", tmp_path, 20000)
    assert capped["sweep"]["subset"] == {"n_per_label": 20000, "seed": 42}


def test_runs_of_different_sizes_never_share_a_folder():
    assert rgs.default_root(None).name == "grid_search_full"
    assert rgs.default_root(5000).name == "grid_search_n5000"


def test_the_holdout_is_left_alone_and_the_final_fit_runs(tmp_path):
    cfg = rgs.build_config(
        rgs.TEMPLATE.read_text(encoding="utf-8"), "ITS_soil", tmp_path
    )
    # sweep.holdout true would switch holdout_evaluation back on.
    assert cfg["sweep"]["holdout"] is False
    assert cfg["holdout_evaluation"]["enabled"] is False
    assert cfg["sweep"]["final_fit"] is True
    assert cfg["final_fit"]["enabled"] is True


def test_the_config_asks_for_one_plain_grid_search(tmp_path):
    text = rgs.resolve_config(
        rgs.TEMPLATE.read_text(encoding="utf-8"), "ITS_soil", tmp_path
    )
    cfg = yaml.safe_load(text)
    assert cfg["cv"]["mode"] == "run_grid"
    assert cfg["cv"]["param_path"] == "hyperparameters.yaml"
    assert cfg["sweep"]["transforms"] == ["relabund"]
    assert cfg["sweep"]["min_prevalence"] == [0]
    assert cfg["sweep"]["icc_max"] == []
    assert cfg["sweep"]["shadow"] is False
    assert cfg["sweep"]["prune"]["enabled"] is False


def test_the_shipped_grids_cover_the_configured_models():
    repo = Path(__file__).resolve().parents[2]
    grids = yaml.safe_load((repo / "hyperparameters.yaml").read_text("utf-8"))
    assert {"rf", "xgb", "svm"} <= set(grids)
    for grid in grids.values():
        assert all(isinstance(values, list) for values in grid.values())
    assert "ccp_alpha" not in grids["rf"]
    # Every svm key must name a real setting of the pipeline.
    first = {key: values[0] for key, values in grids["svm"].items()}
    cv_module.ScaledSVR().set_params(**first)


def _write_run(root, dataset, label, scheme, holdout_r2=0.2):
    run = (
        root
        / dataset
        / "relabund_prev0"
        / f"holdout_bioproject__{label}"
        / "runs"
        / label
        / scheme
    )
    (run / "cv_results").mkdir(parents=True)
    (run / "holdout").mkdir()
    rows = []
    for model, top, extra in (
        ("RandomForestRegressor", 0.3, "param_max_features"),
        ("XGBRegressor", 0.5, "param_max_depth"),
    ):
        for rank, value in enumerate(("1", "2"), start=1):
            rows.append(
                {
                    "feature_set": "tax_order_bio",
                    "label": label,
                    "scheme": scheme,
                    "model": model,
                    "selected": str(rank == 1),
                    "rank_pooled_r2": str(rank),
                    "pooled_r2": str(top - 0.1 * (rank - 1)),
                    "mean_test_r2": "0.1",
                    "std_test_r2": "0.1",
                    "mean_fit_time": "1.0",
                    extra: value,
                }
            )
    pl.concat([pl.DataFrame([row]) for row in rows], how="diagonal").write_csv(
        run / "cv_results" / "grid_results.csv"
    )
    pl.DataFrame(
        {
            "label": [label],
            "scheme": [scheme],
            "feature_set": ["tax_order_bio"],
            "r2": [holdout_r2],
        }
    ).write_csv(run / "holdout" / "holdout_summary.csv")


def test_winners_take_the_best_model_of_each_scheme(tmp_path):
    _write_run(tmp_path, "18S_marine", "temperature", "random")
    _write_run(tmp_path, "18S_marine", "temperature", "month")
    _write_run(tmp_path, "ITS_soil", "geo_temperature", "month")

    table = pgs.winners(pgs.load_grid(tmp_path), pgs.load_holdout(tmp_path))

    assert table.height == 3
    assert set(table.get_column("model")) == {"XGBRegressor"}
    assert table.get_column("pooled_r2").to_list() == [0.5, 0.5, 0.5]
    assert table.get_column("holdout_r2").to_list() == [0.2, 0.2, 0.2]


def test_a_dataset_still_running_does_not_stop_the_figures(tmp_path):
    _write_run(tmp_path, "18S_marine", "temperature", "random")
    (tmp_path / "16S_soil" / "relabund_prev0").mkdir(parents=True)

    written = pgs.plot_all(tmp_path, ["png"])

    figures = tmp_path / "figures"
    assert written
    assert (tmp_path / "grid_winners.csv").exists()
    assert list(figures.glob("grid_overview*"))
    assert list(figures.glob("grid_params__18S_marine__temperature__XGB*"))


def _write_all_models(root, dataset="18S_marine"):
    """all_models.tsv as the sweep writes it: winners up to three times."""
    xgb = json.dumps(
        {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 500}
    )
    rf = json.dumps({"max_features": 0.33, "n_estimators": 500})

    def row(scheme, model, params, n_folds, r2, pooled):
        return {
            "label": "geo_temperature",
            "scheme": scheme,
            "feature_set": "tax_order_bio",
            "model": model,
            "avg_validation_r2": r2,
            "pooled_validation_r2": pooled,
            "n_folds": n_folds,
            "best_params": params,
        }

    rows = [
        row("month", "XGBRegressor", xgb, "5", "0.68", "0.67"),
        row("month", "XGBRegressor", xgb, "5", "0.68", "0.67"),
        row("month", "RandomForestRegressor", rf, "5", "0.66", "0.66"),
        row("month", "XGBRegressor", xgb, "1", "0.63", ""),
        row("random", "XGBRegressor", xgb, "5", "0.82", "0.82"),
        row("random", "RandomForestRegressor", rf, "5", "0.81", "0.81"),
        row("random", "XGBRegressor", xgb, "1", "-2.2", ""),
    ]
    summary = root / dataset / "summary"
    summary.mkdir(parents=True)
    pl.DataFrame(rows).write_csv(summary / "all_models.tsv", separator="\t")


def test_the_report_keeps_one_row_per_model_and_scheme(tmp_path):
    _write_all_models(tmp_path)
    table = pgs.report_table(pgs.load_models(tmp_path), None)

    assert table.height == 4
    month = table.filter(pl.col("scheme") == "month")
    assert month.get_column("cv_pooled_r2").to_list() == [0.66, 0.67]


def test_the_holdout_score_goes_to_the_model_that_was_tested(tmp_path):
    _write_all_models(tmp_path)
    table = pgs.report_table(pgs.load_models(tmp_path), None)

    sent = table.filter(pl.col("sent_to_holdout"))
    assert sent.select("scheme", "model", "holdout_r2").rows() == [
        ("month", "XGBRegressor", 0.63),
        ("random", "XGBRegressor", -2.2),
    ]
    assert (
        table.filter(~pl.col("sent_to_holdout"))["holdout_r2"].is_null().all()
    )


def test_settings_show_only_what_the_grid_searched(tmp_path):
    _write_all_models(tmp_path)
    grid = pl.DataFrame(
        {
            "model": ["XGBRegressor"] * 2 + ["RandomForestRegressor"] * 2,
            "param_learning_rate": ["0.05", "0.1", None, None],
            "param_max_depth": ["6", "6", None, None],
            "param_n_estimators": ["500"] * 4,
            "param_max_features": [None, None, "0.33", "sqrt"],
        }
    )
    table = pgs.report_table(pgs.load_models(tmp_path), grid)
    settings = dict(table.select("model", "settings").unique().rows())

    assert settings["XGBRegressor"] == "learning_rate=0.05"
    assert settings["RandomForestRegressor"] == "max_features=0.33"


def test_without_a_grid_every_setting_is_shown():
    assert pgs._settings('{"b": 2, "a": 1}', None) == "a=1, b=2"
    assert pgs._settings("{}", None) == "defaults"
    assert pgs._settings("null", None) == "defaults"


def test_the_report_figure_is_drawn_even_past_the_floor(tmp_path):
    _write_all_models(tmp_path)

    written = pgs.plot_all(tmp_path, ["png"])

    assert (tmp_path / "grid_report.csv").exists()
    assert list((tmp_path / "figures").glob("grid_report__18S_marine*"))
    assert written


def test_a_grid_run_skips_the_transform_figures():
    import sweep_transforms as sw

    assert sw.draws_transform_figures({"cv": {"mode": "run"}})
    assert sw.draws_transform_figures({})
    assert not sw.draws_transform_figures({"cv": {"mode": "run_grid"}})


def test_nothing_finished_yet_draws_nothing(tmp_path):
    assert pgs.plot_all(tmp_path, ["png"]) == []


def test_numbers_sort_as_numbers_before_words():
    assert pgs._ordered(["sqrt", "10", "2", "None"]) == [
        "2",
        "10",
        "None",
        "sqrt",
    ]
