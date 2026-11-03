#!/usr/bin/env python3
"""Convert a prepared dataset's feature tables from CSV to parquet.

The expensive part of `prepare` is the pivot from long-format profiles to
a wide sample x taxon matrix -- hours on a large dataset. That work is
already in the CSVs, so switching `features.format` to parquet does not
need the dataset rebuilt: convert the tables in place and point the
manifest at them.

    pixi run python scripts/convert_features.py <prepared_dir>
    pixi run python scripts/convert_features.py <prepared_dir> --dry-run

Each `features/<name>.csv` becomes `features/<name>.parquet/`, a
directory of row-group parts, and `manifest.json` is rewritten to match.
The CSV is kept unless --delete-csv is given, so the conversion is
reversible by restoring the manifest.

Not covered: `features.min_prevalence`. Prevalence is counted on the
long-format profiles during the pivot, so turning it on needs
`-R prepare`. Convert first and measure -- parquet alone may be enough.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
LOGGER = logging.getLogger("convert_features")


def convert(csv_path: Path, row_group_size: int) -> Path:
    """Write <stem>.parquet/ beside *csv_path* and return it."""
    out = csv_path.with_suffix(".parquet")
    staging = csv_path.with_name(f".{out.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    lf = pl.scan_csv(csv_path)
    names = lf.collect_schema().names()
    acc = "sample" if "sample" in names else names[0]
    features = [c for c in names if c != acc]
    LOGGER.info(
        "%s: %d feature column(s) -> %s",
        csv_path.name,
        len(features),
        out.name,
    )

    # Declare the schema so the parser emits Float32 instead of widening
    # every text zero to Float64 and casting afterwards.
    lf = pl.scan_csv(
        csv_path, schema_overrides={c: pl.Float32 for c in features}
    )
    lf.sink_parquet(
        staging / "part_00000.parquet", row_group_size=row_group_size
    )

    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="prepared dataset directory")
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=10000,
        help="rows per parquet row group (default 10000); lower it if the "
        "conversion runs out of memory on a very wide table",
    )
    ap.add_argument(
        "--delete-csv",
        action="store_true",
        help="remove each CSV once its parquet is written",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.exists():
        LOGGER.error("no manifest.json in %s", args.dataset)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("components", {}).get("features") or {}
    if not entries:
        LOGGER.error("manifest lists no feature sets")
        return 1

    todo = {
        name: info
        for name, info in entries.items()
        if str(info.get("file", "")).endswith(".csv")
    }
    if not todo:
        LOGGER.info("every feature set is already parquet; nothing to do")
        return 0

    for name, info in todo.items():
        csv_path = args.dataset / "features" / info["file"]
        size = csv_path.stat().st_size / 1e9 if csv_path.exists() else 0.0
        LOGGER.info("%-16s %s (%.2f GB)", name, info["file"], size)
    if args.dry_run:
        LOGGER.info("dry run: nothing written")
        return 0

    for name, info in todo.items():
        csv_path = args.dataset / "features" / info["file"]
        if not csv_path.exists():
            LOGGER.warning("%s missing; skipping", csv_path)
            continue
        out = convert(csv_path, args.row_group_size)
        after = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e9
        LOGGER.info("%s -> %s (%.2f GB)", csv_path.name, out.name, after)
        info["file"] = out.name
        if args.delete_csv:
            csv_path.unlink()
            LOGGER.info("removed %s", csv_path.name)

    # Written last: until the manifest points at them the parquet
    # directories are ignored, so a crash mid-way leaves a working CSV
    # dataset rather than a half-converted one.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("updated %s", manifest_path)
    LOGGER.info(
        "Snakemake will now redo split (manifest.json is newer) and then "
        "cv. Use --touch first if you want to keep the existing splits."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
