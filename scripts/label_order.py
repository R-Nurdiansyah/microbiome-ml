"""One label order for every figure that puts labels side by side.

Temperature, then pH, then oxygen; within each, the plain label, then
geo_, then comb_. A label transform (`ph@yeo_johnson`) follows its label,
and anything else comes last, alphabetically. So a CV-sweep, final-fit,
grid-search or transform-sweep figure always reads in the same order, and
two figures can be compared panel for panel.

Standard library only: the plotting scripts import it, and so can the
Snakefile side without pulling in the package.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

VARIABLES = ("temperature", "ph", "oxy")
ALIASES = {"temp": "temperature", "oxygen": "oxy", "o2": "oxy"}
VARIANTS = ("", "geo_", "comb_")


def split_label(label: str) -> Tuple[str, str, str]:
    """(variant, base, transform): `geo_ph@yeo_johnson` -> geo_, ph, @..."""
    text = str(label)
    name, at, transform = text.partition("@")
    variant = next((v for v in VARIANTS[1:] if name.startswith(v)), "")
    return variant, name[len(variant) :], at + transform


def base_key(base: str) -> Tuple[int, str]:
    """Sort key for a base label (temperature, ph, oxy, then the rest)."""
    name = str(base).partition("@")[0].lower()
    name = ALIASES.get(name, name)
    if name in VARIABLES:
        return VARIABLES.index(name), ""
    return len(VARIABLES), name


def label_key(label: str) -> Tuple[int, str, int, str, str]:
    """Sort key for a label: variable, then plain / geo_ / comb_."""
    variant, base, transform = split_label(label)
    rank, other = base_key(base)
    return rank, other, VARIANTS.index(variant), transform, str(label)


def sort_labels(labels: Iterable[str]) -> List[str]:
    """*labels* in figure order (duplicates kept once)."""
    return sorted(dict.fromkeys(str(x) for x in labels), key=label_key)
