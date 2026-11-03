"""Shadow features: same values, no relationship to anything.

A shadow is only a valid null if it keeps the column's distribution and
loses the sample it belonged to. Both halves are checked here -- a
shadow that quietly stayed aligned with its source would read as a
suspiciously strong "noise floor" and silently invalidate the report it
exists to calibrate.
"""

import polars as pl
import pytest

from microbiome_ml.wrangle.shadow import (
    SHADOW_PREFIX,
    SHADOW_SUFFIX,
    add_shadow_columns,
    is_shadow,
    shadow_floor,
    split_shadows,
)


def _frame(n_samples=40, n_taxa=6):
    """Sparse-ish wide features with distinct values per column."""
    data = {"sample": [f"s{i:02d}" for i in range(n_samples)]}
    for t in range(n_taxa):
        # Distinct values so a permutation is detectable, and some zeros
        # so the columns look like real taxa.
        data[f"g__t{t}"] = [
            0.0 if (i + t) % 4 == 0 else round(0.001 * (i + 1) * (t + 1), 6)
            for i in range(n_samples)
        ]
    return pl.DataFrame(data).lazy()


def _names(frame):
    return [c for c in frame.collect_schema().names() if c != "sample"]


def test_adds_the_requested_number_of_prefixed_columns():
    frame = _frame()
    out, mapping = add_shadow_columns(
        frame, "sample", _names(frame), 4, seed=1
    )
    cols = out.collect_schema().names()
    shade = [c for c in cols if is_shadow(c)]
    assert len(shade) == 4
    assert len(mapping) == 4
    assert all(c.startswith(SHADOW_PREFIX) for c in shade)


def test_real_columns_and_rows_are_untouched():
    frame = _frame()
    before = frame.collect()
    out, _ = add_shadow_columns(frame, "sample", _names(frame), 3, seed=1)
    after = out.collect()
    assert after.height == before.height
    assert sorted(after.get_column("sample").to_list()) == sorted(
        before.get_column("sample").to_list()
    )
    for col in _names(frame):
        assert (
            after.sort("sample").get_column(col).to_list()
            == before.sort("sample").get_column(col).to_list()
        ), col


def test_a_shadow_is_a_permutation_of_its_source():
    frame = _frame()
    out, mapping = add_shadow_columns(
        frame, "sample", _names(frame), 6, seed=3
    )
    df = out.collect()
    for shadow, source in mapping:
        assert sorted(df.get_column(shadow).to_list()) == sorted(
            df.get_column(source).to_list()
        ), f"{shadow} is not a reordering of {source}"


def test_a_shadow_is_not_still_aligned_with_its_source():
    """The whole point: the sample-to-value link must be broken."""
    frame = _frame(n_samples=60)
    out, mapping = add_shadow_columns(
        frame, "sample", _names(frame), 6, seed=5
    )
    df = out.collect()
    unchanged = [
        shadow
        for shadow, source in mapping
        if df.get_column(shadow).to_list() == df.get_column(source).to_list()
    ]
    assert not unchanged, f"shadow(s) identical to their source: {unchanged}"


def test_dtype_survives_the_shuffle():
    base = _frame()
    frame = base.with_columns(pl.col(_names(base)).cast(pl.Float32))
    out, mapping = add_shadow_columns(
        frame, "sample", _names(frame), 2, seed=1
    )
    schema = out.collect_schema()
    for shadow, source in mapping:
        assert schema[shadow] == schema[source] == pl.Float32


def test_same_seed_reproduces_the_same_null():
    frame = _frame()
    a, map_a = add_shadow_columns(frame, "sample", _names(frame), 4, seed=11)
    b, map_b = add_shadow_columns(frame, "sample", _names(frame), 4, seed=11)
    assert map_a == map_b
    assert a.collect().equals(b.collect())


def test_different_seeds_give_a_different_null():
    frame = _frame()
    a, _ = add_shadow_columns(frame, "sample", _names(frame), 4, seed=11)
    b, _ = add_shadow_columns(frame, "sample", _names(frame), 4, seed=12)
    assert not a.collect().equals(b.collect())


def test_asking_for_more_shadows_than_columns_is_clamped():
    frame = _frame(n_taxa=3)
    out, mapping = add_shadow_columns(
        frame, "sample", _names(frame), 99, seed=1
    )
    assert len(mapping) == 3
    assert len([c for c in out.collect_schema().names() if is_shadow(c)]) == 3


def test_zero_shadows_leaves_the_frame_alone():
    frame = _frame()
    out, mapping = add_shadow_columns(frame, "sample", _names(frame), 0)
    assert mapping == []
    assert out.collect_schema().names() == frame.collect_schema().names()


def test_existing_shadow_columns_are_refused():
    """Shadowing a shadow set would make the two indistinguishable."""
    frame = _frame().with_columns(pl.lit(0.0).alias(f"{SHADOW_PREFIX}0000"))
    with pytest.raises(ValueError, match="indistinguishable"):
        add_shadow_columns(frame, "sample", _names(frame), 2, seed=1)


def test_split_shadows_partitions_names():
    real, shade = split_shadows(
        ["g__a", f"{SHADOW_PREFIX}0001", "g__b", f"{SHADOW_PREFIX}0002"]
    )
    assert real == ["g__a", "g__b"]
    assert len(shade) == 2


# ---------------------------------------------------------------------------
# shadow_floor: the summary the report prints
# ---------------------------------------------------------------------------
def _importances(real, shadows):
    return pl.DataFrame(
        {
            "feature": (
                [f"g__r{i}" for i in range(len(real))]
                + [f"{SHADOW_PREFIX}{i:04d}" for i in range(len(shadows))]
            ),
            "importance": [float(v) for v in (*real, *shadows)],
        }
    )


def test_floor_is_none_without_shadows():
    assert shadow_floor(_importances([0.5, 0.3, 0.2], [])) is None


def test_floor_counts_real_features_above_the_best_shadow():
    # three real features clearly above, two buried in the noise
    frame = _importances([0.50, 0.30, 0.20, 0.004, 0.0], [0.01, 0.005, 0.0])
    row = shadow_floor(frame).row(0, named=True)
    assert row["n_real"] == 5
    assert row["n_shadow"] == 3
    assert row["shadow_max"] == pytest.approx(0.01)
    assert row["n_real_above_max"] == 3
    assert row["shadow_total_share"] == pytest.approx(0.015 / (1.004 + 0.015))


def test_floor_q95_is_not_above_the_max():
    frame = _importances([0.9, 0.05], [0.02, 0.01, 0.005, 0.0])
    row = shadow_floor(frame).row(0, named=True)
    assert row["shadow_q95"] <= row["shadow_max"]
    assert row["n_real_above_q95"] >= row["n_real_above_max"]


def test_floor_handles_an_all_zero_importance_table():
    """A model that split on nothing has no distribution to describe."""
    row = shadow_floor(_importances([0.0, 0.0], [0.0])).row(0, named=True)
    assert row["shadow_total_share"] is None
    assert row["n_real_above_max"] == 0


def test_suffix_is_what_the_dataset_appends():
    assert SHADOW_SUFFIX == "shad"
