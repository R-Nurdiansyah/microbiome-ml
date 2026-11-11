"""Tests for the standalone metadata-only holdout checker."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_holdout_metadata as chm  # noqa: E402


@pytest.fixture
def metadata(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    return pl.DataFrame(
        {
            "sample": [f"s{i}" for i in range(n)],
            "geo_ph": rng.normal(8, 0.2, n),
            "bioproject": [f"PRJ{i % 6}" for i in range(n)],
            "ecoregion": [f"ER{i % 4}" for i in range(n)],
        }
    )


def test_run_check_outputs_match_sweep_layout(metadata, tmp_path):
    out = chm.run_check(
        metadata=metadata,
        labels=["geo_ph"],
        groupings=["random", "bioproject"],
        out_dir=tmp_path / "check",
        bins=10,
    )

    for token in ("random", "bioproject"):
        run = out / f"holdout_{token}"
        holdout = run / "splits" / "geo_ph" / "holdout.csv"
        assert holdout.exists()
        df = pl.read_csv(holdout)
        # grouping columns are carried so the figures can stack by them
        assert {"sample", "split", "geo_ph", "bioproject"} <= set(df.columns)
        assert set(df["split"].unique()) == {"train", "test"}
        figs = list((run / "split_visualisations").glob("*.png"))
        assert figs, f"no figure for {token}"

    # the collated report the sweep would also produce
    summary = out / "holdout_distribution_summary.txt"
    assert summary.exists()
    text = summary.read_text()
    assert "geo_ph" in text and "random" in text and "bioproject" in text
    stats = pl.read_csv(out / "holdout_distribution_stats.tsv", separator="\t")
    assert set(stats["holdout_grouping"]) == {"random", "bioproject"}


def test_grouped_holdout_keeps_groups_whole(metadata, tmp_path):
    out = chm.run_check(
        metadata=metadata,
        labels=["geo_ph"],
        groupings=["bioproject"],
        out_dir=tmp_path / "check",
        bins=10,
    )
    df = pl.read_csv(out / "holdout_bioproject/splits/geo_ph/holdout.csv")
    straddling = (
        df.group_by("bioproject")
        .agg(pl.col("split").n_unique().alias("sides"))
        .filter(pl.col("sides") > 1)
        .height
    )
    assert straddling == 0


def test_missing_columns_are_reported(metadata, tmp_path):
    with pytest.raises(ValueError, match="Label column"):
        chm.run_check(
            metadata=metadata,
            labels=["nope"],
            groupings=["random"],
            out_dir=tmp_path / "a",
        )
    with pytest.raises(ValueError, match="Grouping column"):
        chm.run_check(
            metadata=metadata,
            labels=["geo_ph"],
            groupings=["nope"],
            out_dir=tmp_path / "b",
        )


def test_read_table_infers_separator(tmp_path):
    df = pl.DataFrame({"sample": ["a"], "x": [1]})
    tsv, csv = tmp_path / "m.tsv", tmp_path / "m.csv"
    df.write_csv(tsv, separator="\t")
    df.write_csv(csv)
    assert chm.read_table(tsv).columns == ["sample", "x"]
    assert chm.read_table(csv).columns == ["sample", "x"]


def test_collate_accumulates_runs_already_on_disk(tmp_path):
    """A later sweep that adds a grouping must extend the summaries, not
    replace them: collation reads every run directory present."""
    import argparse
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import pipeline_steps as steps

    def make_run(name, label, scheme, r2):
        run = tmp_path / f"holdout_{name}" / "runs" / label / scheme
        (run / "best_models").mkdir(parents=True)
        (run / "cv_results").mkdir(parents=True)
        pl.DataFrame(
            {
                "label": [label],
                "scheme": [scheme],
                "feature_set": ["tax_order"],
                "model": ["RandomForestRegressor"],
                "avg_validation_r2": [r2],
            }
        ).write_csv(run / "best_models" / "best_models_summary.csv")
        pl.DataFrame(
            {
                "feature_set": ["tax_order"],
                "label": [label],
                "scheme": [scheme],
                "model": ["RandomForestRegressor"],
                "avg_validation_r2": [r2],
            }
        ).write_csv(run / "cv_results" / "results_summary.csv")
        (run / "status.json").write_text(
            json.dumps(
                {
                    "holdout_grouping": name,
                    "label": label,
                    "scheme": scheme,
                    "status": "ok",
                    "error": "",
                }
            )
        )
        return run / "status.json"

    first = make_run("random", "ph", "random", 0.7)
    # collate knowing only about the first run
    steps.step_collate(
        argparse.Namespace(out=str(tmp_path), runs=[str(first)])
    )
    one = pl.read_csv(tmp_path / "cv_sweep_all.tsv", separator="\t")
    assert set(one["holdout_grouping"]) == {"random"}

    # a second sweep adds a grouping; the first must survive
    second = make_run("bioproject", "ph", "bioproject", 0.3)
    steps.step_collate(
        argparse.Namespace(out=str(tmp_path), runs=[str(second)])
    )
    both = pl.read_csv(tmp_path / "cv_sweep_all.tsv", separator="\t")
    assert set(both["holdout_grouping"]) == {"random", "bioproject"}
    status = pl.read_csv(tmp_path / "sweep_status.tsv", separator="\t")
    assert status.height == 2


def test_collate_counts_each_model_once(tmp_path):
    """best_models/ and holdout/ repeat results_summary.csv; only the
    cv_results/ copy belongs in cv_sweep_all.tsv."""
    import argparse

    import pipeline_steps as steps

    run = tmp_path / "holdout_bioproject" / "runs" / "t" / "random"
    rows = pl.DataFrame(
        {
            "feature_set": ["tax_order", "tax_order"],
            "label": ["t", "t"],
            "scheme": ["random", "random"],
            "model": ["RandomForestRegressor", "XGBRegressor"],
            "avg_validation_r2": [0.7, 0.6],
        }
    )
    for where, frame in [
        (run / "cv_results", rows),
        (run / "best_models" / "t" / "random", rows.head(1)),
        (run / "holdout", rows.head(1)),
    ]:
        where.mkdir(parents=True)
        frame.write_csv(where / "results_summary.csv")

    steps.step_collate(argparse.Namespace(out=str(tmp_path), runs=[]))
    got = pl.read_csv(tmp_path / "cv_sweep_all.tsv", separator="\t")
    assert got.height == 2
    assert sorted(got["model"].to_list()) == [
        "RandomForestRegressor",
        "XGBRegressor",
    ]


def test_stack_by_is_a_column_not_the_run_stamp(tmp_path, monkeypatch):
    """The Snakemake run stamp is not a column, so it cannot stack bars.

    step_split names a run `bioproject__temperature` and passed that as
    the thing to stack by, so the pipeline's figures silently came out
    unstacked while the standalone script's were stacked. The two
    arguments do different jobs and are now separate.
    """
    import run_cv_to_final_eval as pipeline

    splits = tmp_path / "splits" / "temperature"
    splits.mkdir(parents=True)
    pl.DataFrame(
        {
            "sample": [f"s{i}" for i in range(8)],
            "split": ["train"] * 6 + ["test"] * 2,
            "temperature": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "bioproject": ["A"] * 4 + ["B"] * 4,
        }
    ).write_csv(splits / "holdout.csv")

    seen = {}

    class Spy:
        def __init__(self, *a, **k):
            pass

        def plot_holdout_distribution(self, path, **kw):
            seen["grouping"] = kw.get("grouping")
            return {
                "label": "temperature",
                "n_train": 6,
                "n_test": 2,
                "ks": 0.5,
                "p": 0.1,
            }

    monkeypatch.setattr(pipeline, "Visualiser", Spy)

    pipeline.plot_holdout_distributions(
        tmp_path / "splits",
        {"enabled": True, "holdout_distribution": {"enabled": True}},
        tmp_path / "figs",
        holdout_grouping="bioproject__temperature",
        stack_by="bioproject",
    )
    assert seen["grouping"] == "bioproject", "must stack by the column"


def test_stack_by_falls_back_when_the_column_is_absent(tmp_path, monkeypatch):
    """A taxon holdout stacks by carrier/non-carrier; random not at all."""
    import run_cv_to_final_eval as pipeline

    splits = tmp_path / "splits" / "ph"
    splits.mkdir(parents=True)
    pl.DataFrame(
        {
            "sample": [f"s{i}" for i in range(6)],
            "split": ["train"] * 4 + ["test"] * 2,
            "ph": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
            "holdout_taxon_presence": [False] * 4 + [True] * 2,
        }
    ).write_csv(splits / "holdout.csv")

    seen = {}

    class Spy:
        def __init__(self, *a, **k):
            pass

        def plot_holdout_distribution(self, path, **kw):
            seen["grouping"] = kw.get("grouping")
            return {
                "label": "ph",
                "n_train": 4,
                "n_test": 2,
                "ks": 0.5,
                "p": 0.1,
            }

    monkeypatch.setattr(pipeline, "Visualiser", Spy)

    pipeline.plot_holdout_distributions(
        tmp_path / "splits",
        {"enabled": True, "holdout_distribution": {"enabled": True}},
        tmp_path / "figs",
        holdout_grouping="taxon_family",
        stack_by=None,
    )
    assert seen["grouping"] == "holdout_taxon_presence"
