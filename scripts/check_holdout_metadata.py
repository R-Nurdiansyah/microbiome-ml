#!/usr/bin/env python3
"""Try holdout groupings using only a metadata table -- no profiles.

The holdout split needs labels and grouping columns; taxonomic profiles
are only needed for CV and for taxon-based holdouts. So the question
"which grouping gives a representative holdout for this label?" can be
answered from one metadata table in seconds, instead of waiting for the
pipeline to build features it will not use.

It calls the same functions as the pipeline -- `Dataset.create_holdout_split`
for the split, `Visualiser.plot_holdout_distribution` for the figures,
`collate_split_report` for the ranked text summary -- so the numbers match
what a full run would produce for those groupings.

Example:
    python scripts/check_holdout_metadata.py \\
        --metadata ML_inputs/ITS_marine/metadata.tsv \\
        --labels geo_ph geo_temperature \\
        --groupings random bioproject ecoregion realm \\
        --out out/holdout_check

Outputs, mirroring the sweep layout so the collation works unchanged:
    <out>/holdout_<grouping>/splits/<label>/holdout.csv
    <out>/holdout_<grouping>/split_visualisations/holdout_distribution_*.png
    <out>/holdout_<grouping>/split_visualisations/*_stats.tsv
    <out>/holdout_distribution_stats.tsv      all runs, one table
    <out>/holdout_distribution_summary.txt    ranked best KS -> worst

Not covered: taxon:<rank> holdouts, which need the community profiles.
Use the pipeline for those.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import polars as pl

matplotlib.use("Agg")  # headless

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_steps as steps  # noqa: E402
import run_cv_to_final_eval as pipeline  # noqa: E402

from microbiome_ml.wrangle.dataset import Dataset  # noqa: E402

LOGGER = logging.getLogger("holdout-check")

RANDOM = "random"


def read_table(path: Path, sep: Optional[str] = None) -> pl.DataFrame:
    """Read a delimited table; separator inferred from the suffix."""
    if sep is None:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
    return pl.read_csv(path, separator=sep, infer_schema_length=10000)


def coerce_labels(df: pl.DataFrame, labels: List[str]) -> pl.DataFrame:
    """Make label columns numeric where they can be.

    The splitter bins continuous labels and treats anything else as
    categorical, so a numeric column read as text would be stratified as
    if each value were its own class.
    """
    casts = []
    for col in labels:
        if df.schema[col] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            casts.append(pl.col(col).cast(pl.Float64))
            continue
        as_num = df.select(
            pl.col(col).cast(pl.Float64, strict=False).alias("v")
        )["v"]
        if as_num.null_count() < df.height:  # at least some parsed
            LOGGER.info("Treating label '%s' as continuous", col)
            casts.append(pl.col(col).cast(pl.Float64, strict=False))
        else:
            LOGGER.info("Treating label '%s' as categorical", col)
    return df.with_columns(casts) if casts else df


def run_check(
    metadata: pl.DataFrame,
    labels: List[str],
    groupings: List[str],
    out_dir: Path,
    sample_col: str = "sample",
    test_size: float = 0.2,
    n_bins: int = 5,
    random_state: int = 42,
    bins: int = 30,
    stat: Any = ("count", "probability"),
    formats: Optional[List[str]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_col != "sample":
        metadata = metadata.rename({sample_col: "sample"})
    if "sample" not in metadata.columns:
        raise ValueError(
            f"No '{sample_col}' column in the metadata "
            f"(columns: {metadata.columns[:12]}...)"
        )

    missing = [c for c in labels if c not in metadata.columns]
    if missing:
        raise ValueError(f"Label column(s) not in the metadata: {missing}")
    group_cols = [g for g in groupings if g != RANDOM]
    missing = [c for c in group_cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"Grouping column(s) not in the metadata: {missing}")

    metadata = coerce_labels(metadata, labels)
    # Carried onto every holdout.csv so the figures can stack by group.
    context = metadata.select(["sample", *dict.fromkeys(group_cols)])

    vis_cfg: Dict[str, Any] = {
        "enabled": True,
        "formats": list(formats or ["png"]),
        "holdout_distribution": {
            "enabled": True,
            "grouping": "auto",
            "bins": bins,
            "stat": stat,
        },
    }

    for token in groupings:
        grouping = None if token == RANDOM else token
        run_dir = out_dir / f"holdout_{token}"
        splits_dir = run_dir / "splits"
        LOGGER.info("=" * 70)
        LOGGER.info("holdout grouping = %s", token)

        # Same splitter the pipeline uses; labels + groupings only.
        dataset = Dataset()
        dataset.add_labels(metadata.select(["sample", *labels]))
        if group_cols:
            dataset.add_groupings(context)
        dataset.create_holdout_split(
            label=None,
            test_size=test_size,
            n_bins=n_bins,
            grouping=grouping,
            random_state=random_state,
            force=True,
        )

        # Dataset.save_holdout_splits would write sample/split/label only
        # (there is no SampleMetadata to enrich from), and the figures
        # stack by a grouping column -- so attach the context here.
        for label, split_manager in dataset.splits.items():
            if split_manager.holdout is None:
                continue
            table = split_manager.holdout
            extra = [c for c in context.columns if c not in table.columns]
            if extra:
                table = table.join(
                    context.select(["sample", *extra]),
                    on="sample",
                    how="left",
                )
            label_dir = splits_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            table.write_csv(label_dir / "holdout.csv")

        pipeline.plot_holdout_distributions(
            splits_dir,
            vis_cfg,
            run_dir / "split_visualisations",
            holdout_grouping=token,
        )

    steps.collate_split_report(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="TSV/CSV with a sample column, the label(s) and the groupings",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Label column(s) to split on, e.g. geo_ph geo_temperature",
    )
    parser.add_argument(
        "--groupings",
        nargs="+",
        required=True,
        help="Grouping columns to try, plus 'random' for the stratified "
        "random control, e.g. random bioproject ecoregion",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-col", default="sample")
    parser.add_argument("--sep", default=None, help="override the delimiter")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="label bins used to stratify the split (not the histogram)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--bins", type=int, default=30, help="histogram bins in the figures"
    )
    parser.add_argument(
        "--stat",
        nargs="+",
        default=["count", "probability"],
        help="one row of panels per statistic",
    )
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    pipeline.setup_logging(args.log_level)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)
    pipeline.LOGGER.setLevel(level)

    out = run_check(
        metadata=read_table(args.metadata, args.sep),
        labels=list(args.labels),
        groupings=list(dict.fromkeys(args.groupings)),
        out_dir=args.out,
        sample_col=args.sample_col,
        test_size=args.test_size,
        n_bins=args.n_bins,
        random_state=args.random_state,
        bins=args.bins,
        stat=list(args.stat),
        formats=list(args.formats),
    )
    LOGGER.info("Done. Summary: %s", out / "holdout_distribution_summary.txt")


if __name__ == "__main__":
    main()
