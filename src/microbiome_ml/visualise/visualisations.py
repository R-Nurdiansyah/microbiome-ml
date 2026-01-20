"""Plot CV export history as histogram summaries.

Loads a results NDJSON file (or a directory containing `results.ndjson`) created
by `CV_Result.export_result`. It builds three charts that highlight per-fold
validation R² (flattened), average validation R² per run, and average validation
MSE per run so you can eyeball stability and spread across combinations.

Usage:
    from visualisations.visualisations import Visualiser
    vis = Visualiser("path/to/results.ndjson", out="figs.png")
    vis.plot_cv_bars()
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Visualiser:
    """Encapsulate loading, extracting and plotting CV results."""

    def __init__(
        self,
        results: Path,
        out: Path = Path("visualisations/cv_histograms.png"),
    ):
        self.results = Path(results)
        self.out = Path(out)

    def _load_ndjson(self, path: Path) -> List[Dict[str, Any]]:
        """Load NDJSON or JSON results.

        Accepts either a file path to an NDJSON/JSON file or a directory path
        containing `results.ndjson`.
        """
        if path.is_dir():
            ndjson = path / "results.ndjson"
        else:
            ndjson = path

        if not ndjson.exists():
            raise FileNotFoundError(ndjson)

        records: List[Dict[str, Any]] = []
        with ndjson.open("r", encoding="utf-8") as f:
            first = f.readline()
            if not first:
                return []
            first = first.strip()
            if first.startswith("["):
                f.seek(0)
                records = json.load(f)
                return list(records)
            else:
                try:
                    records.append(json.loads(first))
                except Exception:
                    f.seek(0)
                    records = json.load(f)
                    return list(records)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        return records

    def _extract_metrics(
        self, records: List[Dict[str, Any]]
    ) -> tuple[List[float], List[float], List[float]]:
        """Return (per_fold_r2, avg_r2_list, avg_mse_list).

        - per_fold_r2: flattened list of all validation_r2_per_fold values across
          results
        - avg_r2_list: list of avg_validation_r2 (skip None)
        - avg_mse_list: list of avg_validation_mse (skip None)
        """
        per_fold_r2: List[float] = []
        avg_r2_list: List[float] = []
        avg_mse_list: List[float] = []

        for rec in records:
            r2_fold = (
                rec.get("validation_r2_per_fold")
                or rec.get("validation_r2")
                or []
            )
            if isinstance(r2_fold, (list, tuple)):
                for v in r2_fold:
                    try:
                        per_fold_r2.append(float(v))
                    except Exception:
                        continue
            try:
                ar2 = rec.get("avg_validation_r2")
                if ar2 is not None:
                    avg_r2_list.append(float(ar2))
            except Exception:
                pass
            try:
                amse = rec.get("avg_validation_mse")
                if amse is not None:
                    avg_mse_list.append(float(amse))
            except Exception:
                pass

        return per_fold_r2, avg_r2_list, avg_mse_list

    def _resolve_results_path(self, results: Optional[Path]) -> Path:
        """Resolve a results path from argument, instance, env, or prompt.

        Order of resolution:
        1. explicit `results` argument
        2. `self.results` set at construction
        3. environment variables `MICROBIOME_RESULTS` or `RESULTS`
        4. interactive prompt asking the user for a path
        """
        # 1 & 2
        if results is not None:
            candidate = Path(results)
        else:
            candidate = self.results

        # 3: env
        if not candidate or not candidate.exists():
            for key in ("MICROBIOME_RESULTS", "RESULTS"):
                val = os.environ.get(key)
                if val:
                    candidate = Path(val)
                    break

        # 4: prompt
        if not candidate or not candidate.exists():
            try:
                user_in = input(
                    "Enter path to results.ndjson or directory: "
                ).strip()
            except Exception:
                user_in = ""
            if user_in:
                candidate = Path(user_in)

        if not candidate or not candidate.exists():
            raise FileNotFoundError("No valid results path provided")

        return candidate

    def plot_cv_bars(
        self,
        results: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        show_values: bool = True,
        bar_color: str = "#4c72b0",
        figsize_per_fold: float = 0.8,
    ) -> None:
        """Plot bar chart (one file per combo, including model information) showing validation_r2_per_fold and avg_validation_r2.

        - Groups records by (feature_set, label, scheme).
        - Each group's file is saved as <feature_set>__<label>__<scheme>.png in `out_dir`
          (defaults to `self.out.parent / "per_combo"`).
        """
        load_path = Path(results) if results is not None else self.results
        records = self._load_ndjson(self._resolve_results_path(load_path))

        # group by tuple key
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for rec in records:
            key = (
                rec.get("feature_set") or "unknown_feature_set",
                rec.get("label") or "unknown_label",
                rec.get("scheme") or "unknown_scheme",
                rec.get("model") or "unknown_model",
            )
            groups.setdefault(key, []).append(rec)

        # prepare output directory
        if out_dir is None:
            out_dir = self.out.parent / "cv_results"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        import re

        def _safe_name(s: str) -> str:
            s = re.sub(r"[^\w\-]+", "_", s)
            return s.strip("_") or "item"

        for (feature_set, label, scheme, model), recs in sorted(
            groups.items()
        ):
            # If multiple records per key, flatten their folds together in order
            folds: List[float] = []
            avg_r_vals: List[float] = []
            avg_mse_vals: List[float] = []
            for r in recs:
                vals = (
                    r.get("validation_r2_per_fold")
                    or r.get("validation_r2")
                    or []
                )
                if isinstance(vals, (list, tuple)):
                    for v in vals:
                        try:
                            folds.append(float(v))
                        except Exception:
                            continue
                ar2 = r.get("avg_validation_r2")
                if ar2 is not None:
                    try:
                        avg_r_vals.append(float(ar2))
                    except Exception:
                        pass
                # collect avg_validation_mse if present
                amse = r.get("avg_validation_mse")
                if amse is not None:
                    try:
                        # some records may store as list or scalar
                        if isinstance(amse, (list, tuple)):
                            for v in amse:
                                avg_mse_vals.append(float(v))
                        else:
                            avg_mse_vals.append(float(amse))
                    except Exception:
                        pass

            if not folds:
                continue

            # compute avg line value (prefer provided avg if single rec, else mean of avg_r_vals or mean of folds)
            if len(recs) == 1:
                avg_line = recs[0].get("avg_validation_r2")
                try:
                    avg_line = (
                        float(avg_line)
                        if avg_line is not None
                        else float(sum(folds) / len(folds))
                    )
                except Exception:
                    avg_line = float(sum(folds) / len(folds))
            else:
                if avg_r_vals:
                    avg_line = float(sum(avg_r_vals) / len(avg_r_vals))
                else:
                    avg_line = float(sum(folds) / len(folds))

            # compute avg_validation_mse to display under the plot (mean if multiple)
            if len(recs) == 1:
                mse_val = recs[0].get("avg_validation_mse")
                try:
                    mse_val = (
                        float(mse_val)
                        if mse_val is not None
                        else (float(sum(folds) / len(folds)))
                    )
                except Exception:
                    mse_val = float(sum(folds) / len(folds))
            else:
                if avg_mse_vals:
                    mse_val = float(sum(avg_mse_vals) / len(avg_mse_vals))
                else:
                    mse_val = None

            n = len(folds)
            width = max(4, n * figsize_per_fold)
            fig, ax = plt.subplots(figsize=(width, 4))
            x = list(range(1, n + 1))
            ax.bar(x, folds, color=bar_color, edgecolor="black")
            ax.plot(
                x,
                [avg_line] * n,
                color="k",
                linestyle="--",
                marker="o",
                label=f"average r2 {avg_line:.3f}",
            )
            ax.set_xlabel("Fold")
            ax.set_ylabel("Validation R²")
            title = f"{feature_set} — {label} — {scheme} — {model}"
            ax.set_title(title, wrap=True)
            ax.set_xticks(x)

            if show_values:
                for xi, v in zip(x, folds):
                    ax.text(
                        xi,
                        v,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )

            ax.legend(loc="best")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            # place average mse legend text below the axis so it clears the x-axis label
            if mse_val is not None:
                try:
                    fig.text(
                        0.5,
                        0.02,
                        f"average_mse = {mse_val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#555555",
                    )
                except Exception:
                    pass
            out_name = f"{_safe_name(feature_set)}__{_safe_name(label)}__{_safe_name(scheme)}__{_safe_name(model)}.png"
            out_path = out_dir / out_name
            fig.tight_layout(rect=(0, 0.05, 1, 0.95))
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logging.info("Saved CV results to %s", out_path)
