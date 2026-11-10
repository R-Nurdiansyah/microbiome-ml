"""Run the sklearn grid search on one dataset, or on all four in turn.

Fills scripts/grid_search.yaml in for the dataset, hands it to
sweep_transforms.py, then redraws the figures from everything finished under
--root. Rerunning the same command resumes: finished label x scheme jobs are
skipped, so a job that hits its walltime loses only the one in progress.

    pixi run python scripts/run_grid_search.py --dataset 18S_marine
    pixi run python scripts/run_grid_search.py --dataset all
    pixi run python scripts/run_grid_search.py --dataset 16S_soil --n-per-label 20000

Without --n-per-label every labelled sample is used. Results go under
ML_outputs/grid_search_full, or grid_search_n<N> with a cap, so runs of
different sizes never share a folder. Each dataset takes the whole job's
PBS_NCPUS, so separate jobs are usually faster than `all`.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOGGER = logging.getLogger("run-grid-search")

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().with_name("grid_search.yaml")
DATASETS = ("18S_marine", "ITS_soil", "16S_marine", "16S_soil")
OUTPUTS = (
    "/home/n11702427/m/big_data_microbiome/rizky/machine_learning/all/"
    "all_amplicon/ML_outputs"
)
SUBSET_SEED = 42


def default_root(n_per_label: Optional[int]) -> Path:
    """Where a run of this size goes when --root is not given."""
    name = "grid_search_full" if n_per_label is None else (
        f"grid_search_n{n_per_label}"
    )
    return Path(OUTPUTS) / name


def resolve_config(template: str, dataset: str, root: Path) -> str:
    """The template for one dataset."""
    return template.replace("{root}", root.as_posix()).replace(
        "{dataset}", dataset
    )


def build_config(
    template: str,
    dataset: str,
    root: Path,
    n_per_label: Optional[int] = None,
) -> Dict[str, Any]:
    """This dataset's config: all labelled samples, or a capped subset."""
    cfg = yaml.safe_load(resolve_config(template, dataset, root))
    sweep = cfg.setdefault("sweep", {})
    if n_per_label is None:
        sweep.pop("subset", None)
    else:
        sweep["subset"] = {"n_per_label": int(n_per_label), "seed": SUBSET_SEED}
    return cfg


def run_dataset(
    dataset: str, root: Path, n_per_label: Optional[int] = None
) -> int:
    """Write this dataset's config and run it; returns the exit code."""
    config = root / "configs" / f"{dataset}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_config(
        TEMPLATE.read_text(encoding="utf-8"), dataset, root, n_per_label
    )
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    LOGGER.info("%s: config %s", dataset, config)
    # A child process, so one dataset running out of memory cannot sink the
    # next. cwd is the repo, where the template's param_path is relative to.
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sweep_transforms.py"),
            "--config",
            str(config),
        ],
        cwd=ROOT,
        check=False,
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=[*DATASETS, "all"]
    )
    parser.add_argument(
        "--n-per-label",
        type=int,
        default=None,
        help="cap each label at this many samples; default: all of them",
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    root = args.root or default_root(args.n_per_label)

    wanted = DATASETS if args.dataset == "all" else (args.dataset,)
    failed: List[str] = [
        d for d in wanted if run_dataset(d, root, args.n_per_label) != 0
    ]

    import plot_grid_search as pgs

    pgs.plot_all(root, ["png"])
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}. Rerun to resume.")


if __name__ == "__main__":
    main()
