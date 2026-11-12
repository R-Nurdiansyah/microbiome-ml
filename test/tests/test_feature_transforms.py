"""Feature transforms (transforms/features.py)."""

import math

import numpy as np
import polars as pl
import pytest

from microbiome_ml.transforms import features as ft
from microbiome_ml.wrangle.dataset import (
    feature_set_name,
    rank_from_feature_set,
)


def _wide():
    return pl.DataFrame(
        {
            "sample": ["a", "b"],
            "g1": [0.0, 0.5],
            "g2": [1.6e-5, 0.0],
            "g3": [0.9999840, 0.5],
        }
    ).with_columns(pl.col(["g1", "g2", "g3"]).cast(pl.Float32))


@pytest.mark.parametrize(
    "given,expected",
    [
        ("relabund", "relabund"),
        ("PA", "presence"),
        ("log1p", "log"),
        ("log_ppm", "log"),
        ("clr", "clr"),
    ],
)
def test_config_spellings(given, expected):
    assert ft.canonical_transform(given) == expected


def test_every_transform_has_its_own_name_suffix():
    assert set(ft.SUFFIX) == set(ft.TRANSFORMS)
    assert len(set(ft.SUFFIX.values())) == len(ft.SUFFIX)


def test_log_is_per_million_and_keeps_zeros_zero():
    out = ft.apply(_wide(), ["g1", "g2", "g3"], "log")
    assert out["g1"][0] == 0.0 and out["g2"][1] == 0.0
    assert out["g2"][0] == pytest.approx(math.log1p(16.0), rel=1e-4)
    assert out["g1"][1] == pytest.approx(math.log1p(5e5), rel=1e-4)
    assert out.schema["g1"] == pl.Float32


def test_log_spreads_values_log1p_alone_would_not():
    """Why the per-million scale: log1p of a raw fraction is the fraction."""
    small = np.array([1.6e-5, 6.6e-5, 3.9e-4, 5.2e-3])
    assert np.allclose(np.log1p(small), small, rtol=1e-2)
    spread = np.log1p(small * ft.LOG_SCALE)
    assert spread.max() - spread.min() > 5


def test_log_keeps_each_taxons_order_of_samples():
    """So tree models split it exactly as they split relabund."""
    wide = pl.DataFrame({"sample": list("abcd"), "g": [0.3, 0.0, 1e-5, 0.02]})
    out = ft.apply(wide, ["g"], "log")
    assert np.argsort(out["g"].to_numpy()).tolist() == np.argsort(
        wide["g"].to_numpy()
    ).tolist()


def test_relabund_is_returned_unchanged():
    wide = _wide()
    assert ft.apply(wide, ["g1", "g2", "g3"], "relabund").equals(wide)


def test_presence_and_clr_still_do_what_they_did():
    pa = ft.apply(_wide(), ["g1", "g2", "g3"], "presence", presence_threshold=0)
    assert pa.select(["g1", "g2", "g3"]).rows() == [(0, 1, 1), (1, 0, 1)]
    clr = ft.apply(_wide(), ["g1", "g2", "g3"], "clr", clr_pseudocount=1e-6)
    sums = clr.select(pl.sum_horizontal(["g1", "g2", "g3"]))
    assert np.allclose(sums.to_series().to_numpy(), 0.0, atol=1e-4)


def test_log_names_its_feature_set():
    assert feature_set_name("tax", "genus", "log") == "tax_genus_log"
    assert rank_from_feature_set("tax_genus_log") == "genus"
    assert rank_from_feature_set("tax_genus_log_bio") == "genus"
