"""Figures put labels side by side in one order: temperature, ph, oxy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from label_order import label_key, sort_labels  # noqa: E402


def test_temperature_then_ph_then_oxy_each_plain_geo_comb():
    labels = [
        "comb_oxy",
        "geo_ph",
        "oxy",
        "temperature",
        "comb_temperature",
        "ph",
        "geo_temperature",
        "comb_ph",
        "geo_oxy",
    ]
    assert sort_labels(labels) == [
        "temperature",
        "geo_temperature",
        "comb_temperature",
        "ph",
        "geo_ph",
        "comb_ph",
        "oxy",
        "geo_oxy",
        "comb_oxy",
    ]


def test_transforms_follow_their_label_and_others_go_last():
    assert sort_labels(["salinity", "ph@yeo_johnson", "geo_ph", "ph"]) == [
        "ph",
        "ph@yeo_johnson",
        "geo_ph",
        "salinity",
    ]
    # base labels too, as the CV-sweep grid sorts its rows by them
    assert sorted(["oxy", "ph", "temperature"], key=label_key) == [
        "temperature",
        "ph",
        "oxy",
    ]
