"""Streamed feature building must match the in-memory pivot exactly."""

from pathlib import Path

import polars as pl
import pytest

from microbiome_ml.wrangle.profiles import TaxonomicProfiles


def _profiles(n_samples=23, n_families=7):
    """Long-format filled profiles spanning several families."""
    rows = {"sample": [], "taxonomy": [], "relabund": []}
    for s in range(n_samples):
        # each sample carries a different subset, so some chunks are
        # missing taxa that others have -- the padding path
        for f in range(n_families):
            if (s + f) % 3 == 0:
                continue
            rows["sample"].append(f"s{s:02d}")
            rows["taxonomy"].append(
                f"d__Bacteria;p__P{f % 2};c__C{f % 3};o__O{f};f__F{f}"
            )
            rows["relabund"].append(round(0.01 * (s + 1) * (f + 1), 6))
    return pl.DataFrame(rows)


@pytest.fixture
def profiles():
    return TaxonomicProfiles(profiles=_profiles(), check_filled=False)


def _sorted(df):
    return df.sort("sample").select(sorted(df.columns))


@pytest.mark.parametrize("chunk", [1, 4, 10, 1000])
def test_streamed_matches_in_memory(profiles, tmp_path, chunk):
    eager = profiles.create_features("family").to_df()
    streamed = profiles.create_features(
        "family",
        stream_path=tmp_path / f"tax_family_{chunk}.csv",
        chunk_samples=chunk,
    ).to_df()

    assert _sorted(streamed).equals(_sorted(eager))
    assert streamed.height == eager.height
    assert set(streamed.columns) == set(eager.columns)


def test_streamed_feature_set_is_lazy_and_knows_its_file(profiles, tmp_path):
    path = tmp_path / "tax_family.csv"
    fs = profiles.create_features("family", stream_path=path, chunk_samples=5)

    assert path.exists()
    assert isinstance(fs.features, pl.LazyFrame)
    assert Path(fs.source_path) == path
    # a single header, i.e. chunks were appended not re-headered
    header = path.read_text().splitlines()[0]
    assert header.startswith("sample,")
    headers = [
        line
        for line in path.read_text().splitlines()
        if line.startswith("sample,")
    ]
    assert len(headers) == 1


def test_streaming_restricted_to_given_samples(profiles, tmp_path):
    keep = ["s00", "s01", "s02"]
    fs = profiles.create_features(
        "family",
        stream_path=tmp_path / "subset.csv",
        chunk_samples=2,
        samples=keep,
    )
    df = fs.to_df()
    assert sorted(df["sample"].to_list()) == keep


def test_streaming_handles_a_rank_with_no_hits(profiles, tmp_path):
    fs = profiles.create_features(
        "species", stream_path=tmp_path / "tax_species.csv"
    )
    assert fs.to_df().height == 0


# The chain that emptied every streamed feature table on the cluster:
# add_taxonomic_features streams a rank to <out>/features/tax_<rank>.csv and
# keeps a scan of it, _sync_accessions replaces the feature set with
# filter_samples(...) (dropping source_path), and Dataset.save then sinks
# that scan straight back onto the file it reads from -- truncating it and
# leaving nothing but the header.
def test_save_over_own_source_keeps_the_rows(profiles, tmp_path):
    path = tmp_path / "tax_family.csv"
    fs = profiles.create_features("family", stream_path=path, chunk_samples=4)
    expected = fs.to_df()
    assert expected.height > 0

    fs.save(path)

    reloaded = pl.read_csv(path)
    assert reloaded.height == expected.height
    assert reloaded.columns == expected.columns


def test_save_over_own_source_after_a_no_op_filter(profiles, tmp_path):
    """The real path: _sync_accessions filters before Dataset.save."""
    path = tmp_path / "tax_family.csv"
    fs = profiles.create_features("family", stream_path=path, chunk_samples=4)
    expected = fs.to_df()

    filtered = fs.filter_samples(fs.accessions)
    filtered.save(path)

    assert pl.read_csv(path).height == expected.height


def test_filter_samples_keeps_source_path_only_when_nothing_dropped(
    profiles, tmp_path
):
    path = tmp_path / "tax_family.csv"
    fs = profiles.create_features("family", stream_path=path, chunk_samples=4)

    kept = fs.filter_samples(fs.accessions)
    assert Path(kept.source_path) == path

    # After a real filter the file holds rows the frame does not, so
    # Dataset.save must not take the "already on disk" shortcut.
    subset = fs.filter_samples(fs.accessions[:2])
    assert subset.source_path is None


def test_save_leaves_no_temp_file_behind(profiles, tmp_path):
    path = tmp_path / "tax_family.csv"
    fs = profiles.create_features("family", stream_path=path, chunk_samples=4)
    fs.save(path)

    assert [p.name for p in tmp_path.iterdir()] == ["tax_family.csv"]


# ---------------------------------------------------------------------------
# features.format: parquet  /  features.min_prevalence
# ---------------------------------------------------------------------------
def test_parquet_matches_csv_exactly(profiles, tmp_path):
    csv = profiles.create_features(
        "family", stream_path=tmp_path / "f.csv", chunk_samples=4
    ).to_df()
    pq = profiles.create_features(
        "family", stream_path=tmp_path / "f.parquet", chunk_samples=4
    ).to_df()

    assert _sorted(pq).equals(_sorted(csv))


def test_parquet_is_a_directory_of_row_group_parts(profiles, tmp_path):
    path = tmp_path / "tax_family.parquet"
    fs = profiles.create_features("family", stream_path=path, chunk_samples=4)

    assert path.is_dir()
    parts = sorted(path.glob("*.parquet"))
    assert len(parts) > 1, "one part per chunk"
    assert Path(fs.source_path) == path
    assert isinstance(fs.features, pl.LazyFrame)


def test_features_are_float32_not_float64(profiles, tmp_path):
    """A text zero has no dtype; CSV inference used to widen to Float64."""
    for suffix in ("csv", "parquet"):
        fs = profiles.create_features(
            "family", stream_path=tmp_path / f"t.{suffix}", chunk_samples=4
        )
        dtypes = fs.features.collect_schema()
        taxa = [c for c in dtypes.names() if c != "sample"]
        assert all(dtypes[c] == pl.Float32 for c in taxa), suffix


def test_absent_taxa_are_zero_not_null(profiles, tmp_path):
    fs = profiles.create_features(
        "family", stream_path=tmp_path / "f.parquet", chunk_samples=2
    )
    df = fs.to_df()
    assert df.null_count().sum_horizontal().item() == 0


def _prevalence_profiles():
    """10 samples: one family in all, one in half, one in a single one."""
    rows = {"sample": [], "taxonomy": [], "relabund": []}
    for i in range(10):
        present = (
            ["common"]
            + (["mid"] if i < 5 else [])
            + (["rare"] if i == 0 else [])
        )
        for fam in present:
            rows["sample"].append(f"s{i:02d}")
            rows["taxonomy"].append(f"d__Bacteria;p__P;c__C;o__O;f__{fam}")
            rows["relabund"].append(0.1 * (i + 1))
    return TaxonomicProfiles(profiles=pl.DataFrame(rows), check_filled=False)


def test_min_prevalence_drops_rare_taxa():
    prof = _prevalence_profiles()
    assert set(prof.create_features("family").to_df().columns) == {
        "sample",
        "f__common",
        "f__mid",
        "f__rare",
    }

    # 0.5 -> present in >= 5 of 10 samples: common (10) and mid (5) stay,
    # rare (1) goes.
    kept = prof.create_features("family", min_prevalence=0.5).to_df()
    assert set(kept.columns) == {"sample", "f__common", "f__mid"}
    assert kept.height == 10, "dropping a taxon must not drop its sample"

    # 0.9 -> only the family in every sample survives
    strict = prof.create_features("family", min_prevalence=0.9).to_df()
    assert set(strict.columns) == {"sample", "f__common"}


def test_min_prevalence_keeps_samples_whose_taxa_all_went(tmp_path):
    """A sample left with nothing becomes a row of zeros, not a gone row.

    Seen on real 18S marine data: a 1% filter cut 68 phyla to 34 and the
    table lost exactly the 50 samples whose every phylum was rare. The
    filter is about columns; those samples still carry a label.
    """
    rows = {"sample": [], "taxonomy": [], "relabund": []}
    for i in range(10):
        # s09 carries only a one-off family, everyone else the common one
        fam = "odd" if i == 9 else "common"
        rows["sample"].append(f"s{i:02d}")
        rows["taxonomy"].append(f"d__Bacteria;p__P;c__C;o__O;f__{fam}")
        rows["relabund"].append(0.5)
    prof = TaxonomicProfiles(profiles=pl.DataFrame(rows), check_filled=False)

    for path in (None, "t.csv", "t.parquet"):
        kept = prof.create_features(
            "family",
            min_prevalence=0.5,
            **({} if path is None else {"stream_path": tmp_path / path}),
        ).to_df()
        assert set(kept.columns) == {"sample", "f__common"}, path
        assert kept.height == 10, f"{path}: s09 must survive as zeros"
        lonely = kept.filter(pl.col("sample") == "s09")
        assert lonely.height == 1
        assert lonely["f__common"].item() == 0.0, "zero, not null"


def test_min_prevalence_none_keeps_everything(profiles, tmp_path):
    a = profiles.create_features("family").to_df()
    b = profiles.create_features("family", min_prevalence=None).to_df()
    assert sorted(a.columns) == sorted(b.columns)


# ---------------------------------------------------------------------------
# features.transform: clr  /  features.prevalence_by
# ---------------------------------------------------------------------------
def _clr_profiles():
    """4 samples x 3 families, no zeros, so CLR is checkable by hand."""
    rows = {"sample": [], "taxonomy": [], "relabund": []}
    for i in range(4):
        for j, fam in enumerate(["a", "b", "c"]):
            rows["sample"].append(f"s{i}")
            rows["taxonomy"].append(f"d__D;p__P;c__C;o__O;f__{fam}")
            rows["relabund"].append(float(j + 1) * (i + 1))
    return TaxonomicProfiles(profiles=pl.DataFrame(rows), check_filled=False)


def test_clr_matches_the_definition():
    """clr(x)_j = ln(x_j + p) - mean_k ln(x_k + p)."""
    import math

    prof = _clr_profiles()
    p = 1e-6
    out = (
        prof.create_features("family", transform="clr", clr_pseudocount=p)
        .to_df()
        .sort("sample")
    )

    taxa = [c for c in out.columns if c != "sample"]
    row = out.filter(pl.col("sample") == "s0")
    # s0 carries 1, 2, 3
    raw = {"f__a": 1.0, "f__b": 2.0, "f__c": 3.0}
    logs = {k: math.log(v + p) for k, v in raw.items()}
    gm = sum(logs.values()) / 3
    for col in taxa:
        assert row[col].item() == pytest.approx(logs[col] - gm, abs=1e-5)

    # a CLR row sums to zero by construction
    assert sum(row[c].item() for c in taxa) == pytest.approx(0.0, abs=1e-4)


def test_clr_is_the_same_streamed_as_eager(tmp_path):
    prof = _clr_profiles()
    eager = prof.create_features("family", transform="clr").to_df()
    for suffix in ("csv", "parquet"):
        streamed = prof.create_features(
            "family",
            transform="clr",
            stream_path=tmp_path / f"c.{suffix}",
            chunk_samples=2,
        ).to_df()
        assert _sorted(streamed).equals(_sorted(eager)), suffix


def test_transform_defaults_to_relabund():
    prof = _clr_profiles()
    a = prof.create_features("family").to_df()
    b = prof.create_features("family", transform="relabund").to_df()
    assert _sorted(a).equals(_sorted(b))
    assert a["f__a"].max() > 0, "untransformed values stay positive"


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="relabund"):
        _clr_profiles().create_features("family", transform="log")


# ---------------------------------------------------------------------------
# features.transform: presence
# ---------------------------------------------------------------------------
def _presence_profiles():
    """3 samples x 3 families, mixing absent, near-zero and plain values.

    `f__b` is absent from s0 as a missing row and present-but-zero in
    s1, which are two different ways into the same cell; both have to
    read as absence. s2 carries a 1e-5 trace, which only a raised
    threshold should discard.
    """
    rows = [
        ("s0", "f__a", 0.5),
        ("s0", "f__c", 0.5),
        ("s1", "f__a", 0.9),
        ("s1", "f__b", 0.0),
        ("s1", "f__c", 0.1),
        ("s2", "f__a", 1e-5),
        ("s2", "f__b", 0.4),
        ("s2", "f__c", 0.6),
    ]
    return TaxonomicProfiles(
        profiles=pl.DataFrame(
            {
                "sample": [r[0] for r in rows],
                "taxonomy": [f"d__D;p__P;c__C;o__O;{r[1]}" for r in rows],
                "relabund": [r[2] for r in rows],
            }
        ),
        check_filled=False,
    )


def test_presence_is_binary_and_marks_the_right_cells():
    out = (
        _presence_profiles()
        .create_features("family", transform="presence")
        .to_df()
        .sort("sample")
    )
    taxa = sorted(c for c in out.columns if c != "sample")
    assert taxa == ["f__a", "f__b", "f__c"]
    assert set(out.select(taxa).to_numpy().ravel().tolist()) <= {0.0, 1.0}

    got = {
        row["sample"]: [row[c] for c in taxa]
        for row in out.iter_rows(named=True)
    }
    # s0 has no f__b row at all, s1 has one holding zero: both absent.
    assert got["s0"] == [1.0, 0.0, 1.0]
    assert got["s1"] == [1.0, 0.0, 1.0]
    assert got["s2"] == [1.0, 1.0, 1.0], "1e-5 is present at threshold 0"


def test_presence_threshold_drops_trace_detections():
    out = (
        _presence_profiles()
        .create_features(
            "family", transform="presence", presence_threshold=1e-4
        )
        .to_df()
        .sort("sample")
    )
    s2 = out.filter(pl.col("sample") == "s2")
    assert s2["f__a"].item() == 0.0, "1e-5 is below the threshold"
    assert s2["f__b"].item() == 1.0, "0.4 is well above it"


def test_presence_is_the_same_streamed_as_eager(tmp_path):
    prof = _presence_profiles()
    eager = prof.create_features("family", transform="presence").to_df()
    for suffix in ("csv", "parquet"):
        streamed = prof.create_features(
            "family",
            transform="presence",
            stream_path=tmp_path / f"p.{suffix}",
            chunk_samples=2,
        ).to_df()
        assert _sorted(streamed).equals(_sorted(eager)), suffix


def test_presence_keeps_every_sample(tmp_path):
    """Binarising changes values, never the row set."""
    prof = _presence_profiles()
    assert (
        prof.create_features("family", transform="presence").to_df().height
        == 3
    )


def test_presence_aliases_resolve_to_one_transform():
    from microbiome_ml.wrangle.profiles import canonical_transform

    for alias in ("presence", "pa", "PA", " Binary ", "presence_absence"):
        assert canonical_transform(alias) == "presence"


def test_prevalence_by_group_counts_projects_not_samples():
    """One big project must not make a taxon look common.

    `hog` is in 6 of 9 samples (67%) but only 1 of 4 projects (25%).
    `both` is in 4 of 9 samples (44%) but 4 of 4 projects (100%).
    At a 0.6 threshold the two counting modes therefore disagree on
    every taxon, which is the whole point of the setting.
    """
    rows = {"sample": [], "taxonomy": [], "relabund": []}

    def add(sample, fam):
        rows["sample"].append(sample)
        rows["taxonomy"].append(f"d__D;p__P;c__C;o__O;f__{fam}")
        rows["relabund"].append(0.5)

    for i in range(6):  # project A: six samples, all hog
        add(f"a{i}", "hog")
    add("a0", "both")
    for proj in ("b", "c", "d"):  # three one-sample projects
        add(f"{proj}0", "both")

    prof = TaxonomicProfiles(profiles=pl.DataFrame(rows), check_filled=False)
    groups = pl.DataFrame(
        {
            "sample": [f"a{i}" for i in range(6)] + ["b0", "c0", "d0"],
            "bioproject": ["A"] * 6 + ["B", "C", "D"],
        }
    )

    by_sample = prof.create_features("family", min_prevalence=0.6).to_df()
    assert "f__hog" in by_sample.columns, "6 of 9 samples"
    assert "f__both" not in by_sample.columns, "4 of 9 samples"

    by_project = prof.create_features(
        "family", min_prevalence=0.6, prevalence_groups=groups
    ).to_df()
    assert "f__both" in by_project.columns, "4 of 4 projects"
    assert "f__hog" not in by_project.columns, "1 of 4 projects"
    assert by_project.height == 9, "no sample lost to the filter"
