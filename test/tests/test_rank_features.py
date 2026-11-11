"""A rank feature is the taxon's total, whichever layout the profile has.

SingleM's glossary terms (https://wwood.github.io/singlem/Glossary):

- unfilled: a row holds the reads assigned to that taxon and no deeper, so
  a taxon's total is the sum of every row under it. SingleM's standard
  (condensed) output; one row per species, as in the PR2, UNITE and MFD
  profiles of the real runs, is the extreme case.
- filled: a row already includes all of its descendants, so a taxon's total
  is its own row (the unit-test data in conftest.py and test/data).

Before the fix, the layout check was inverted, the fill step it triggered
lost rows, and pivot(aggregate_function="first") kept one row per taxon.
"""

import json
import logging

import polars as pl
import pytest

from microbiome_ml.wrangle.dataset import Dataset
from microbiome_ml.wrangle.profiles import TaxonomicProfiles

LOGGER = "microbiome_ml.wrangle.profiles"
BASE = "d__Bacteria;p__P1;c__C1"


def _profiles(rows, value="relabund", **kwargs):
    df = pl.DataFrame(rows, schema=["sample", "taxonomy", value], orient="row")
    return TaxonomicProfiles(profiles=df, **kwargs)


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


# One row per species, sorted by abundance as the real profiles are: the
# old code took 0.5 for o__A, the largest of its three species.
LEAF = [
    ("S1", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.5),
    ("S1", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.2),
    ("S1", f"{BASE};o__A;f__F1;g__G1;s__a2", 0.2),
    ("S1", f"{BASE};o__A;f__F2;g__G2;s__a3", 0.1),
    ("S2", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.6),
    ("S2", f"{BASE};o__B;f__F3;g__G3;s__b2", 0.4),
]

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

# SingleM condensed output: reads assigned no deeper than the domain (0.4)
# or the phylum (0.2) sit on their own rows. Shares of the sample, sum 1.
SINGLEM = [
    ("S1", "Root;d__Bacteria", 0.4),
    ("S1", "Root;d__Bacteria;p__Ps", 0.2),
    ("S1", "Root;d__Bacteria;p__Ps;c__G;o__Pm;f__Pf;g__Pg;s__Pa", 0.3),
    ("S1", "Root;d__Bacteria;p__Ba;c__B;o__La;f__Lf;g__Lg;s__Lb", 0.1),
]


# ---------------------------------------------------------------------------
# Which layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rows, expected",
    [(LEAF, False), (FILLED, True), (FILLED[::-1], True), (SINGLEM, False)],
    ids=["one_row_per_species", "filled", "filled_reversed", "singlem"],
)
def test_the_layout_is_read_from_the_data(rows, expected):
    assert _profiles(rows).is_filled is expected


def test_the_conftest_and_real_test_data_read_as_filled(
    profiles_csv, root_csv
):
    assert TaxonomicProfiles(profiles_csv, root=root_csv).layout == "filled"
    real = TaxonomicProfiles("test/data/subset_real_filled_coverage.csv.gz")
    assert real.layout == "filled"


def test_a_given_layout_is_not_second_guessed():
    assert _profiles(LEAF, layout="filled").is_filled is True
    assert _profiles(FILLED, layout="unfilled").is_filled is False


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="layout must be one of"):
        _profiles(LEAF, layout="condensed")


def test_unchecked_input_is_still_taken_as_filled():
    assert _profiles(LEAF, check_filled=False).is_filled is True


# ---------------------------------------------------------------------------
# Rank features
# ---------------------------------------------------------------------------
def test_unfilled_sums_every_species_of_the_taxon(path):
    got = _cells(_features(_profiles(LEAF), "order", path))
    assert got == {
        "S1": {"o__A": 0.8, "o__B": 0.2},
        "S2": {"o__A": 0.0, "o__B": 1.0},
    }


def test_unfilled_rows_sum_to_one_at_every_rank(path):
    for rank in ("phylum", "order", "family", "genus", "species"):
        df = _features(_profiles(LEAF), rank, path)
        totals = df.select(pl.sum_horizontal(pl.exclude("sample"))).to_series()
        assert totals.to_list() == pytest.approx([1.0, 1.0]), rank


@pytest.mark.parametrize("order", ["parents_first", "children_first"])
def test_filled_takes_the_taxons_own_row(path, order):
    rows = FILLED if order == "parents_first" else FILLED[::-1]
    got = _cells(_features(_profiles(rows), "order", path))
    # not 0.9 + 0.8 + 0.5 + 0.3, and not whichever row happened to be first
    assert got == {"S1": {"o__A": 0.9, "o__B": 0.1}}


def test_filled_matches_the_conftest_fixture(sample_profiles):
    # coverage / root: S1 order row 90 / 150, Lactobacillales species 25 / 150
    got = _cells(sample_profiles.create_features("order").to_df())
    assert got["S1"]["o__Enterobacterales"] == pytest.approx(90 / 150)
    assert got["S1"]["o__Lactobacillales"] == pytest.approx(25 / 150)


def test_singlem_reads_above_species_are_counted(path):
    prof = _profiles(SINGLEM)
    # Ps holds 0.2 of its own plus its 0.3 species. Taking only the
    # shallowest row, as the previous fix did, would give 0.2.
    assert _cells(_features(prof, "phylum", path)) == {
        "S1": {"p__Ps": 0.5, "p__Ba": 0.1}
    }
    assert _cells(_features(prof, "domain", path)) == {
        "S1": {"d__Bacteria": 1.0}
    }


def test_a_lineage_stopping_early_is_still_counted(path):
    # `o__A;f__;g__;s__` is reads assigned no deeper than o__A: it adds to
    # the order. A row with an empty order belongs to no order at all.
    rows = [
        ("S1", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.6),
        ("S1", f"{BASE};o__A;f__;g__;s__", 0.3),
        ("S1", f"{BASE};o__;f__;g__;s__", 0.1),
    ]
    got = _cells(_features(_profiles(rows), "order", path))
    assert got == {"S1": {"o__A": 0.9}}


# ---------------------------------------------------------------------------
# Coverage -> relative abundance
# ---------------------------------------------------------------------------
def test_unfilled_coverage_is_divided_by_every_classified_row():
    rows = [(s, t, 10 * v) for s, t, v in SINGLEM] + [("S1", "Root", 5.0)]
    prof = _profiles(rows, value="coverage")
    # 4 + 2 + 3 + 1 = 10. The bare Root row is unclassified even at the
    # domain, so it is left out, as SingleM leaves it out.
    assert prof.root.collect()["root_coverage"].to_list() == [10.0]
    got = _cells(prof.create_features("order").to_df())
    assert got == {"S1": {"o__Pm": 0.3, "o__La": 0.1}}


def test_filled_coverage_is_divided_by_the_domain_row():
    rows = [(s, t, 10 * v) for s, t, v in FILLED]
    prof = _profiles(rows, value="coverage")
    # The domain row (10), not every row added up (57) as before.
    assert prof.root.collect()["root_coverage"].to_list() == [10.0]
    got = _cells(prof.create_features("order").to_df())
    assert got == {"S1": {"o__A": 0.9, "o__B": 0.1}}


# ---------------------------------------------------------------------------
# Dominated samples, filling, save and load
# ---------------------------------------------------------------------------
def test_dominated_filter_reads_unfilled_totals():
    rows = [
        ("S1", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.600),
        ("S1", f"{BASE};o__A;f__F1;g__G1;s__a2", 0.395),
        ("S1", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.005),
        ("S2", f"{BASE};o__A;f__F1;g__G1;s__a1", 0.5),
        ("S2", f"{BASE};o__B;f__F3;g__G3;s__b1", 0.5),
    ]
    kept = _profiles(rows).filter_dominated_samples(0.99, "order")
    # o__A is 0.995 of S1 although no single row is; reading only rows
    # that end at the order found none and dropped both samples.
    assert kept.get_samples() == ["S2"]


def test_fill_profiles_gives_the_same_features():
    unfilled = _profiles(SINGLEM)
    filled = unfilled.fill_profiles()
    assert filled.is_filled is True
    for rank in ("domain", "phylum", "order", "species"):
        a = _cells(unfilled.create_features(rank).to_df())
        b = _cells(filled.create_features(rank).to_df())
        assert a == b, rank


def test_the_layout_survives_save_and_load(tmp_path):
    Dataset(profiles=_profiles(LEAF)).save(tmp_path / "ds")
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest["components"]["profiles"]["layout"] == "unfilled"
    loaded = Dataset.load(tmp_path / "ds")
    assert loaded.profiles.is_filled is False


def test_an_old_manifest_is_detected_again(tmp_path):
    # Before `layout` was saved, the broken check stored is_filled=True for
    # one-row-per-species profiles; that flag must not be trusted.
    Dataset(profiles=_profiles(LEAF)).save(tmp_path / "ds")
    path = tmp_path / "ds" / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["components"]["profiles"]["layout"]
    manifest["components"]["profiles"]["is_filled"] = True
    path.write_text(json.dumps(manifest))
    assert Dataset.load(tmp_path / "ds").profiles.is_filled is False


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
def test_no_warning_when_proportions_add_up(path, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _features(_profiles(LEAF), "order", path)
    assert "summing above 1" not in caplog.text


def test_a_filled_profile_read_as_unfilled_is_warned_once(path, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        df = _features(_profiles(FILLED, layout="unfilled"), "order", path)
    # a warning, not an error: the table is still built
    assert df.height == 1
    warnings = [
        r for r in caplog.records if "summing above 1" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "1 of 1 sample(s)" in warnings[0].getMessage()
