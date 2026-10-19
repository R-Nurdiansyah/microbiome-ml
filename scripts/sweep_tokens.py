"""Sweep token parsing, shared by the Snakefile and the sweep scripts.

Deliberately dependency-free (stdlib only). The Snakefile imports this at
DAG-build time, and importing `microbiome_ml` there would run its
`setup_logging()` on the root logger, which makes Snakemake's own log lines
print twice -- besides pulling polars/sklearn into every `snakemake` call.
"""

from __future__ import annotations

from typing import Optional, Tuple

RANDOM = "random"  # directory/CLI name for "no holdout grouping"
TAXON_PREFIX = "taxon_"  # dir-safe form of the "taxon:<rank>" token


def holdout_spec(name: str) -> Tuple[str, Optional[str]]:
    """Interpret a sweep token as a holdout strategy.

    random | null | none  -> ("column", None)   stratified random
    taxon:family          -> ("taxon", "family") community membership
    taxon_family          -> ("taxon", "family") (dir-safe spelling)
    <anything else>       -> ("column", <name>) metadata grouping
    """
    text = name.strip()
    if text.lower() in {RANDOM, "null", "none", ""}:
        return ("column", None)
    if text.startswith("taxon:"):
        return ("taxon", text.split(":", 1)[1])
    if text.startswith(TAXON_PREFIX):
        return ("taxon", text[len(TAXON_PREFIX) :])
    return ("column", text)


def spec_dir_name(name: str) -> str:
    """Directory/wildcard-safe name for a sweep token."""
    kind, value = holdout_spec(name)
    if kind == "taxon":
        return f"{TAXON_PREFIX}{value}"
    return RANDOM if value is None else value
