"""Per-taxon ICC: the cases where the answer is known by hand.

The non-obvious part is that the long frame is sparse -- a taxon absent
from a sample has no row -- while ICC needs every zero. These check that
the implicit zeros are carried correctly, because a bug there would read
as a plausible-looking number rather than a crash.
"""

import math

import polars as pl
import pytest

from microbiome_ml.wrangle.icc import ICC_COLUMNS, taxon_icc


def _long(rows):
    """rows: [(sample, taxon, value), ...] -> the long frame."""
    return pl.DataFrame(
        {
            "sample": [r[0] for r in rows],
            "taxonomy": [r[1] for r in rows],
            "relabund": [float(r[2]) for r in rows],
        }
    ).lazy()


def _groups(mapping):
    """mapping: {sample: bioproject}."""
    return pl.DataFrame(
        {
            "sample": list(mapping),
            "bioproject": [mapping[s] for s in mapping],
        }
    )


GROUPS_2x3 = _groups(
    {
        "s0": "A",
        "s1": "A",
        "s2": "A",
        "s3": "B",
        "s4": "B",
        "s5": "B",
    }
)


def _icc_of(table, taxon):
    row = table.filter(pl.col("taxonomy") == taxon)
    assert row.height == 1, f"{taxon} appears {row.height} times"
    return row.row(0, named=True)


def test_flat_within_study_differing_between_is_icc_one():
    """Knowing the study tells you the abundance -> ICC ~ 1."""
    rows = [
        ("s0", "g__x", 0.10),
        ("s1", "g__x", 0.11),
        ("s2", "g__x", 0.09),
        ("s3", "g__x", 0.50),
        ("s4", "g__x", 0.51),
        ("s5", "g__x", 0.49),
    ]
    got = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "g__x")
    assert got["icc"] == pytest.approx(1.0, abs=0.01)
    assert got["icc_raw"] == pytest.approx(1.0, abs=0.01)
    assert got["prevalence"] == pytest.approx(1.0)


def test_varying_within_study_with_equal_means_is_icc_zero():
    """The study tells you nothing -> ICC ~ 0."""
    rows = [
        ("s0", "g__y", 0.10),
        ("s1", "g__y", 0.50),
        ("s2", "g__y", 0.30),
        ("s3", "g__y", 0.12),
        ("s4", "g__y", 0.48),
        ("s5", "g__y", 0.30),
    ]
    got = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "g__y")
    assert got["icc"] == pytest.approx(0.0, abs=0.01)
    assert got["icc_raw"] == pytest.approx(0.0, abs=0.01)


def test_absences_count_as_zeros_not_as_missing():
    """A taxon in one study only is a study marker, not a clean feature.

    `g__only_b` has no row at all for study A. If those absences were
    dropped instead of read as zeros it would look like a constant --
    zero variance, no ICC -- rather than the strongly study-associated
    taxon it is.
    """
    rows = [
        ("s3", "g__only_b", 0.40),
        ("s4", "g__only_b", 0.50),
        ("s5", "g__only_b", 0.45),
    ]
    got = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "g__only_b")
    assert got["icc"] > 0.9, "absent-from-A must read as study-associated"
    assert got["prevalence"] == pytest.approx(0.5), "3 of 6 samples"
    assert got["n_groups_present"] == 1
    assert got["mean"] == pytest.approx((0.40 + 0.50 + 0.45) / 6)


def test_several_rows_for_one_taxon_in_a_sample_are_one_value():
    """A taxon split over lineages gives the ICC of its per-sample totals.

    Squaring the rows one by one makes the within-group sum of squares go
    negative. It then clamps to zero, and the taxon reads as ICC 1.0 with
    a prevalence above 1, whatever the data says.
    """
    totals = [
        ("s0", 0.10),
        ("s1", 0.50),
        ("s2", 0.30),
        ("s3", 0.12),
        ("s4", 0.48),
        ("s5", 0.30),
    ]
    split = [
        (sample, "o__x", value * share)
        for sample, value in totals
        for share in (0.5, 0.3, 0.2)
    ]
    summed = [(sample, "o__x", value) for sample, value in totals]

    got = _icc_of(taxon_icc(_long(split), GROUPS_2x3), "o__x")
    want = _icc_of(taxon_icc(_long(summed), GROUPS_2x3), "o__x")

    assert got["icc"] == pytest.approx(0.0, abs=0.01)
    assert got["prevalence"] == pytest.approx(1.0)
    assert got["n_present"] == 6
    for column in ("mean", "var_between", "var_within", "icc", "icc_raw"):
        assert got[column] == pytest.approx(want[column]), column


def test_a_taxon_split_in_some_samples_only_counts_each_sample_once():
    rows = [
        ("s0", "o__x", 0.2),
        ("s0", "o__x", 0.3),
        ("s3", "o__x", 0.5),
    ]
    got = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "o__x")
    assert got["n_present"] == 2
    assert got["prevalence"] == pytest.approx(2 / 6)
    assert got["mean"] == pytest.approx(1.0 / 6)
    assert got["n_groups_present"] == 2


def test_prevalence_and_mean_span_every_sample_in_the_design():
    """Denominators are all samples, not the ones carrying a row."""
    rows = [("s0", "g__rare", 0.60)]
    got = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "g__rare")
    assert got["n_present"] == 1
    assert got["prevalence"] == pytest.approx(1 / 6)
    assert got["mean"] == pytest.approx(0.60 / 6)


def test_all_singleton_groups_give_no_estimate():
    """With one sample per study there is no within-study variance.

    The raw ratio reads 1.0 for every taxon here, which is exactly the
    trap; `icc` must be null rather than report that as a finding.
    """
    groups = _groups({"s0": "A", "s1": "B", "s2": "C", "s3": "D"})
    rows = [
        ("s0", "g__x", 0.1),
        ("s1", "g__x", 0.5),
        ("s2", "g__x", 0.3),
        ("s3", "g__x", 0.2),
    ]
    got = _icc_of(taxon_icc(_long(rows), groups), "g__x")
    assert got["icc"] is None
    assert got["icc_raw"] == pytest.approx(1.0)


def test_single_group_gives_no_estimate():
    groups = _groups({"s0": "A", "s1": "A", "s2": "A"})
    rows = [("s0", "g__x", 0.1), ("s1", "g__x", 0.5), ("s2", "g__x", 0.3)]
    got = _icc_of(taxon_icc(_long(rows), groups), "g__x")
    assert got["icc"] is None


def test_singletons_inflate_the_raw_ratio():
    """Why `icc` and not `icc_raw`: a singleton study can only push up.

    Three studies whose means are identical, plus two one-sample studies
    at the extremes. The variance-components estimate discounts the
    singletons; the raw sum-of-squares ratio does not.
    """
    groups = _groups(
        {
            "s0": "A",
            "s1": "A",
            "s2": "A",
            "s3": "B",
            "s4": "B",
            "s5": "B",
            "s6": "C",
            "s7": "C",
            "s8": "C",
            "s9": "D",
            "s10": "E",
        }
    )
    vals = [0.10, 0.50, 0.30, 0.12, 0.48, 0.30, 0.20, 0.40, 0.30, 0.80, 0.05]
    rows = [(f"s{i}", "g__x", v) for i, v in enumerate(vals)]
    got = _icc_of(taxon_icc(_long(rows), groups), "g__x")
    assert got["icc_raw"] == pytest.approx(0.65, abs=0.01)
    assert got["icc"] == pytest.approx(0.46, abs=0.01)
    assert got["icc"] < got["icc_raw"]


def test_samples_argument_restricts_the_design():
    """Dropping study B leaves one study, so no estimate is possible."""
    rows = [
        ("s0", "g__x", 0.10),
        ("s1", "g__x", 0.11),
        ("s2", "g__x", 0.09),
        ("s3", "g__x", 0.50),
        ("s4", "g__x", 0.51),
        ("s5", "g__x", 0.49),
    ]
    full = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "g__x")
    assert full["icc"] is not None
    only_a = taxon_icc(_long(rows), GROUPS_2x3, samples=["s0", "s1", "s2"])
    assert _icc_of(only_a, "g__x")["icc"] is None
    assert _icc_of(only_a, "g__x")["prevalence"] == pytest.approx(1.0)


def test_sorted_by_icc_descending_with_the_schema_promised():
    rows = [
        # study marker
        ("s0", "g__batch", 0.10),
        ("s1", "g__batch", 0.10),
        ("s2", "g__batch", 0.10),
        ("s3", "g__batch", 0.90),
        ("s4", "g__batch", 0.90),
        ("s5", "g__batch", 0.90),
        # no study signal
        ("s0", "g__clean", 0.10),
        ("s1", "g__clean", 0.50),
        ("s2", "g__clean", 0.30),
        ("s3", "g__clean", 0.10),
        ("s4", "g__clean", 0.50),
        ("s5", "g__clean", 0.30),
    ]
    table = taxon_icc(_long(rows), GROUPS_2x3)
    assert list(table.columns) == list(ICC_COLUMNS)
    assert table.get_column("taxonomy").to_list() == ["g__batch", "g__clean"]


def test_empty_grouping_returns_an_empty_table_not_a_crash():
    rows = [("s0", "g__x", 0.1)]
    table = taxon_icc(_long(rows), _groups({"other": "A"}))
    assert table.height == 0


def test_variance_components_are_non_negative():
    """(MSB - MSW) can go negative; a variance cannot."""
    rows = [
        ("s0", "g__x", 0.50),
        ("s1", "g__x", 0.10),
        ("s2", "g__x", 0.30),
        ("s3", "g__x", 0.30),
        ("s4", "g__x", 0.50),
        ("s5", "g__x", 0.10),
    ]
    got = _icc_of(taxon_icc(_long(rows), GROUPS_2x3), "g__x")
    assert got["var_between"] >= 0.0
    assert got["var_within"] >= 0.0
    assert 0.0 <= got["icc"] <= 1.0
    assert not math.isnan(got["icc"])
