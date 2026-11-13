#!/usr/bin/env python
"""Draw holdout maps for a run that has already split.

Train, test and buffer samples per label, the test cells shaded for a
spatial holdout -- the same figures the split step draws, without
rerunning the split (which would make Snakemake redo everything after it).

    pixi run python scripts/plot_holdout_map.py <base_dir>/holdout_spatial/splits

Figures go to <run>/split_visualisations/holdout_map_<label>.png, beside
the holdout distributions, unless --out says otherwise. Works for any
holdout (bioproject too); cell sizes come from spatial_report.tsv when the
run has one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_cv_to_final_eval as pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "splits", type=Path, help="a run's splits/ directory (holdout.csv)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write (default: <run>/split_visualisations)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["png"], help="png, svg, pdf, eps"
    )
    args = parser.parse_args()
    splits = args.splits.resolve()
    if not any(splits.glob("*/holdout.csv")):
        parser.error(f"no <label>/holdout.csv under {splits}")
    out = args.out or splits.parent / "split_visualisations"
    pipeline.setup_logging("INFO")
    pipeline.plot_holdout_maps(
        splits,
        {"enabled": True, "holdout_map": True, "formats": args.formats},
        out,
    )
    print(f"maps in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
