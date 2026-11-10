"""Sweep token parsing, shared by the Snakefile and the sweep scripts.

Deliberately dependency-free (stdlib only). The Snakefile imports this at
DAG-build time, and importing `microbiome_ml` there would run its
`setup_logging()` on the root logger, which makes Snakemake's own log lines
print twice -- besides pulling polars/sklearn into every `snakemake` call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


def run_specs(
    pairs: Any = None, holdout_groupings: Any = None
) -> Dict[str, Dict[str, Optional[str]]]:
    """The runs a sweep performs, as {dir_name: spec}.

    Two modes:
      holdout_groupings: [random, bioproject, "taxon:family"]
          every grouping x every label -- the cross-product.
      pairs: {geo_ph: organism, geo_temperature: realm}
          one run per label, each with its own holdout grouping.
    `pairs` wins if both are given. A pair run restricts the label at
    every stage, so a label without a holdout cannot trip up CV.

    Raises ValueError on a config that names nothing to run.
    """
    specs: Dict[str, Dict[str, Optional[str]]] = {}
    if pairs:
        if isinstance(pairs, str):  # --config pairs=geo_ph:organism,...
            pairs = dict(
                item.split(":", 1) for item in pairs.split(",") if item
            )
        if not isinstance(pairs, dict):
            raise ValueError(
                "sweep.pairs must be a mapping of label -> holdout grouping"
            )
        for label, tokens in pairs.items():
            # A label may name several groupings; each becomes its own run,
            # so adding one later does not disturb the runs already done.
            if tokens is None or isinstance(tokens, str):
                tokens = [tokens]
            elif not isinstance(tokens, (list, tuple)):
                raise ValueError(
                    f"sweep.pairs[{label}] must be a grouping or a list of "
                    f"groupings, got {tokens!r}"
                )
            for token in tokens:
                token = RANDOM if token is None else str(token).strip()
                name = spec_dir_name(token)
                specs[f"holdout_{name}__{label}"] = {
                    "token": token,
                    "label": str(label),
                    "stamp": f"{name}__{label}",
                }
        return specs

    if holdout_groupings is None:
        raise ValueError(
            "Nothing to sweep. Give either sweep.holdout_groupings "
            "([random, bioproject, ...]) or sweep.pairs "
            "({label: grouping}); --config holdout_groupings=... and "
            "--config pairs=label:grouping,... also work."
        )
    if isinstance(holdout_groupings, str):
        holdout_groupings = holdout_groupings.split(",")
    for grouping in holdout_groupings:
        token = RANDOM if grouping is None else str(grouping).strip()
        name = spec_dir_name(token)
        specs.setdefault(
            f"holdout_{name}",
            {"token": token, "label": None, "stamp": name},
        )
    return specs
