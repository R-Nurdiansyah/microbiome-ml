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
