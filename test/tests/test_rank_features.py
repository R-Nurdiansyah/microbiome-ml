"""A rank feature is the taxon's total, whichever profile layout it comes from.

Two layouts reach create_features:

- filled: a row at every rank, the parent holding the total (the unit-test
  data in conftest.py and test/data);
- leaf-only: one row per species, each a disjoint share (the PR2, UNITE and
  MFD profiles of the real runs).

Before the fix, pivot(aggregate_function="first") was right on filled data
only because parents came first, and on leaf-only data kept the most
abundant species of each taxon alone.
"""

import logging

import polars as pl
import pytest

from microbiome_ml.wrangle.profiles import TaxonomicProfiles

LOGGER = "microbiome_ml.wrangle.profiles"
BASE = "d__Bacteria;p__P1;c__C1"


def _profiles(rows):
    df = pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    )
    return TaxonomicProfiles(profiles=df, check_filled=False)


def _features(prof, rank, path):
    if path is None:
        return prof.create_features(rank).to_df()
    return prof.create_features(
        rank, stream_path=path, chunk_samples=1
    ).to_df()


def _cells(df):
    """{sample: {taxon: value}}, so column order cannot matter."""
    out = {}
    for row in df.iter_rows(named=True):
        sample = row.pop("sample")
        out[sample] = {k: pytest.approx(v, abs=1e-6) for k, v in row.items()}
    return out


@pytest.fixture(params=["in_memory", "streamed"])
def path(request, tmp_path):
    return None if request.param == "in_memory" else tmp_path / "tax.csv"


# Sorted by abundance within the sample, as the real profiles are: the old
# code took 0.5 for o__A, the largest of its three species.
LEAF = [
    ("S1", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.5),
    ("S1", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.2),
    ("S1", f"{BASE};o__A;f__F1;g__G1;s__a2", 0.2),
    ("S1", f"{BASE};o__A;f__F2;g__G2;s__a3", 0.1),
    ("S2", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.6),
    ("S2", f"{BASE};o__B;f__F3;g__G3;s__b2", 0.4),
]


def test_leaf_profile_sums_every_species_of_the_taxon(path):
    got = _cells(_features(_profiles(LEAF), "order", path))
    assert got == {
        "S1": {"o__A": 0.8, "o__B": 0.2},
        "S2": {"o__A": 0.0, "o__B": 1.0},
    }


def test_leaf_profile_rows_sum_to_one_at_every_rank(path):
    for rank in ("phylum", "order", "family", "genus", "species"):
        df = _features(_profiles(LEAF), rank, path)
        totals = df.select(
            pl.sum_horizontal(pl.exclude("sample"))
        ).to_series()
        assert totals.to_list() == pytest.approx([1.0, 1.0]), rank


# A row at every rank. Order's own row (0.9) is more than its family (0.8),
# because 0.1 of the reads were assigned no deeper than the order.
FILLED = [
    ("S1", "d__Bacteria", 1.0),
    ("S1", "d__Bacteria;p__P1", 1.0),
    ("S1", BASE, 1.0),
    ("S1", f"{BASE};o__A", 0.9),
    ("S1", f"{BASE};o__A;f__F1", 0.8),
    ("S1", f"{BASE};o__A;f__F1;g__G1", 0.5),
    ("S1", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.3),
    ("S1", f"{BASE};o__B", 0.1),
    ("S1", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.1),
]


@pytest.mark.parametrize("order", ["parents_first", "children_first"])
def test_filled_profile_takes_the_taxons_own_row(path, order):
    rows = FILLED if order == "parents_first" else FILLED[::-1]
    got = _cells(_features(_profiles(rows), "order", path))
    # not 0.9 + 0.8 + 0.5 + 0.3, and not whichever row happened to be first
    assert got == {"S1": {"o__A": 0.9, "o__B": 0.1}}


def test_filled_profile_matches_the_conftest_fixture(sample_profiles):
    # coverage / root: S1 order row 90 / 150, Lactobacillales species 25 / 150
    got = _cells(sample_profiles.create_features("order").to_df())
    assert got["S1"]["o__Enterobacterales"] == pytest.approx(90 / 150)
    assert got["S1"]["o__Lactobacillales"] == pytest.approx(25 / 150)


def test_a_lineage_stopping_early_is_still_counted(path):
    # Empty ranks keep the row at species depth, so it is a sibling of the
    # named species and adds to the order, not a parent that replaces it.
    rows = [
        ("S1", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.6),
        ("S1", f"{BASE};o__A;f__;g__;s__", 0.3),
        ("S1", f"{BASE};o__;f__;g__;s__", 0.1),
    ]
    got = _cells(_features(_profiles(rows), "order", path))
    # the row with an empty order belongs to no order, so S1 sums to 0.9
    assert got == {"S1": {"o__A": 0.9}}


def test_no_warning_when_proportions_add_up(path, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _features(_profiles(LEAF), "order", path)
    assert "summing above 1" not in caplog.text


def test_a_rank_summing_above_one_is_warned_once(path, caplog):
    doubled = [(s, t, 2 * v) for s, t, v in LEAF]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        df = _features(_profiles(doubled), "order", path)
    # a warning, not an error: the table is still built
    assert df.height == 2
    warnings = [
        r for r in caplog.records if "summing above 1" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "2 of 2 sample(s)" in warnings[0].getMessage()
