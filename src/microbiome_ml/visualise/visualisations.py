"""Class to do visualisations.

Visualiser now owns a user-configured output path and exposes helpers:
`plot_cv_bars(results=...)` for saving per-combination CV summaries and
`visualise_model_performance(...)` for the core hexbin/residual diagnostics.

Usage:
    from microbiome_ml.visualise.visualisations import Visualiser
    vis = Visualiser(out="figures")
    vis.plot_cv_bars(results="path/to/results.ndjson")

    evaluation = trainer.train_and_evaluate()
    scheme = evaluation.metrics.get("scheme")
    groups = [scheme] * len(evaluation.targets) if scheme else None
    vis.visualise_model_performance(
        evaluation.predictions,
        evaluation.targets,
        groups=groups,
        title="Holdout diagnostics",
        file_name="holdout.png",
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, List, Optional, Tuple, Union

import cmasher as cmr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
import seaborn as sns
from pandas import Categorical
from pandas.errors import PerformanceWarning
from scipy.stats import pearsonr, zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Visualiser:
    """Wrap CV exports and holdout diagnostics around a single output root."""

    def __init__(
        self,
        out: Path = Path("visualisations"),
    ):
        self.out = Path(out)
        if self.out.suffix:
            self.output_dir = self.out.parent
        else:
            self.output_dir = self.out
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_ndjson(self, path: Path) -> List[Dict[str, Any]]:
        """Load NDJSON or JSON results.

        Accepts either a file path to an NDJSON/JSON file or a directory path
        containing `results.ndjson` or `best_result.ndjson`.
        """
        if path.is_dir():
            ndjson = path / "results.ndjson"
            if not ndjson.exists():
                ndjson = path / "best_result.ndjson"
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
        2. environment variables `MICROBIOME_RESULTS` or `RESULTS`
        3. interactive prompt asking the user for a path
        """
        candidate: Optional[Path] = (
            Path(results) if results is not None else None
        )
        if candidate and not candidate.exists():
            candidate = None

        # 2: env
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
        """Plot bar chart (one file per combo, including model information)
        showing validation_r2_per_fold and avg_validation_r2.

                The NDJSON/directory path must be passed through `results` argument or resolved from environment variable or prompt. The expected format is either:
        - a single NDJSON file containing records with keys like `feature_set`, `label`, `scheme`, `model`, `validation_r2_per_fold`, and `avg_validation_r2`, or
        - a directory containing `results.ndjson` or `best_result.ndjson` with the same format.

                - Groups records by (feature_set, label, scheme).
                        - Each group's file is saved as <feature_set>__<label>__<scheme>__<model>.png in `out_dir`
                            (defaults to `self.output_dir / "cv_results"`).
        """
        load_path = Path(results) if results is not None else None
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
            out_dir = self.output_dir / "cv_results"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _custom_formatter(x: float, pos: int) -> str:
        if x >= 1000:
            return f"{float(x * 1e-3):.1f}k"
        return str(int(x))

    @staticmethod
    def _remove_outliers_zscore(
        data: Union[np.ndarray, pl.Series],
        colours: Optional[pl.Series],
        threshold: float,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        data_arr = np.asarray(data)
        z_scores = zscore(data_arr)
        mask = (z_scores < threshold) & (z_scores > -threshold)
        filtered_data = data_arr[mask]
        if colours is None:
            return filtered_data, None
        colours_list = colours.to_list()
        filtered_colours = [c for c, keep in zip(colours_list, mask) if keep]
        return filtered_data, filtered_colours

    @staticmethod
    def _calculate_metrics(
        predictions: np.ndarray, values: np.ndarray
    ) -> Tuple[float, float, float, float]:
        mse = mean_squared_error(values, predictions)
        r2 = r2_score(values, predictions)
        mae = mean_absolute_error(values, predictions)
        pcc, _ = pearsonr(values, predictions)
        return mse, r2, mae, pcc

    @staticmethod
    def _create_hexbin_plot(
        fig: plt.Figure,
        ax: plt.Axes,
        values: np.ndarray,
        predictions: np.ndarray,
        cmap: str,
        cax: plt.Axes,
        lower_bound: float,
        upper_bound: float,
    ) -> Tuple[float, float]:
        cmap = cmr.get_sub_cmap(cmap, 0.2, 0.8)
        hb = ax.hexbin(
            values,
            predictions,
            cmap=cmap,
            bins="log",
            zorder=3,
            gridsize=50,
            mincnt=1,
            linewidths=0.2,
            extent=(lower_bound, upper_bound, lower_bound, upper_bound),
        )
        ax.axline((0, 0), slope=1, color="#7272a1", linestyle="dashed")
        fig.colorbar(
            hb,
            cax=cax,
            label="Count (log_10)",
            drawedges=False,
            orientation="horizontal" if len(fig.axes) == 4 else "vertical",
        )
        cax.grid()
        m, b = np.polyfit(values, predictions, 1)
        ax.plot(
            values,
            m * np.array(values, dtype=np.float32) + b,
            color="red",
            alpha=0.75,
            zorder=2,
            linewidth=4,
        )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        return m, b

    @staticmethod
    def _set_axis_properties(
        ax: plt.Axes, predictions: np.ndarray, values: np.ndarray, units: str
    ) -> Tuple[float, float, float]:
        all_values = np.concatenate((predictions, values))
        buffer = np.ptp(all_values) * 0.1
        lower_bound = min(all_values) - buffer
        upper_bound = max(all_values) + buffer
        ax.set_xlim((lower_bound, upper_bound))
        ax.set_ylim((lower_bound, upper_bound))
        ax.set_xlabel(f"Actual Values ({units})")
        ax.set_ylabel(f"Predicted Values ({units})")
        return lower_bound, upper_bound, buffer

    @staticmethod
    def _generate_hist_cmap(
        c: Optional[pl.Series], c_list: List[Any], cmap: str
    ) -> Tuple[Union[dict, list, None], Optional[List[Any]], bool]:
        palette = cmr.get_sub_cmap(cmap, 0.2, 0.8)
        legend = False
        if c is None:
            return None, None, legend
        if c.dtype == pl.datatypes.Utf8:
            legend = True
            unique_entries = len(set(c_list))
            if unique_entries > 20:
                most_common = [
                    item for item, _ in Counter(c_list).most_common(20)
                ] + ["Other"]
                c_list = [
                    item if item in most_common and item != "NA" else "Other"
                    for item in c_list
                ]
                colors_dict = {
                    label: palette(idx)
                    for label, idx in zip(
                        most_common, np.linspace(0, 1, len(most_common))
                    )
                }
                colors_dict["Other"] = (0.6, 0.6, 0.6, 1.0)
                cat_list = Categorical(
                    c_list, categories=most_common, ordered=True
                )
                return colors_dict, cat_list.to_list(), legend
            if unique_entries > 10:
                palette_list = [
                    palette(i) for i in np.linspace(0, 1, unique_entries)
                ]
                return palette_list, c_list, legend
            unique = sorted(set(c_list))
            palette_list = [palette(i) for i in np.linspace(0, 1, len(unique))]
            return palette_list, c_list, legend
        c = c.cast(pl.Int64)
        return palette, c_list, legend

    def _create_histogram(
        self,
        fig: plt.Figure,
        hax: plt.Axes,
        predictions: np.ndarray,
        values: np.ndarray,
        units: str,
        hist_bins: int,
        hist_trim: float,
        cax: plt.Axes,
        c: Optional[pl.Series],
        cmap: str,
        cbar_label: Optional[str],
    ) -> None:
        diff = values - predictions
        c_list: List[Any] = c.to_list() if c is not None else []
        if hist_trim:
            diff, trimmed = self._remove_outliers_zscore(diff, c, hist_trim)
            if trimmed is not None:
                c_list = trimmed
        palette, hue, legend = self._generate_hist_cmap(c, c_list, cmap)
        sns.histplot(
            x=diff,
            bins=hist_bins,
            hue=hue,
            palette=palette,
            multiple="stack",
            zorder=2,
            ec=None,
            alpha=1,
            legend=legend,
            ax=hax,
        )
        hax.axvline(x=0, color="#7272a1", linestyle="dashed", zorder=1)
        hax.set_xlabel(f"Error [True - Pred] ({units})")
        hax.set_ylabel("")
        hax.yaxis.tick_right()
        hax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        hax.yaxis.set_major_formatter(
            ticker.FuncFormatter(self._custom_formatter)
        )
        if c is None:
            cax.axis("off")
        elif legend:
            legend_obj = hax.get_legend()
            if legend_obj is not None:
                handles = [
                    h for h in legend_obj.legend_handles if h is not None
                ]
                labels = [
                    "\n".join(wrap(text.get_text(), 20))
                    for text in legend_obj.texts
                ]
                if handles and labels:
                    hax.legend(handles[: len(labels)], labels[: len(handles)])
            sns.move_legend(
                hax,
                "upper left",
                bbox_to_anchor=(1.15, 1),
                title=None,
                frameon=False,
            )
            cax.axis("off")
        else:
            if not c_list:
                c_list = [0, 1]
            norm = plt.Normalize(float(min(c_list)), float(max(c_list)))
            cbar_cmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(
                cbar_cmap,
                cax=cax,
                label=cbar_label,
                drawedges=False,
                orientation="horizontal",
            )
            cax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x)}")
            )
            cax.grid()

    @staticmethod
    def _set_title(
        fig: plt.Figure,
        title: str,
        pcc: float,
        r2: float,
        m: float,
        b: float,
        mse: float,
        mae: float,
        n: int,
    ) -> None:
        diagnostics = (
            f"R2: {r2:.3f} | PCC: {pcc:.3f} | MSE: {mse:.3f} | MAE: {mae:.3f} | "
            f"m: {m:.3f} | b: {b:.3f} | n: {n}"
        )
        fig.suptitle(title, fontsize="large", weight="bold", y=0.95)
        fig.text(
            0.5,
            0.9,
            diagnostics,
            ha="center",
            va="center",
            fontsize="medium",
            weight="normal",
        )

    def visualise_model_performance(
        self,
        predictions: Union[np.ndarray, pl.Series, list],
        values: Union[np.ndarray, pl.Series, list],
        title: str = "Actual vs. Predicted Values",
        units: str = "",
        groups: Union[pl.Series, list, None] = None,
        hist_cmap: str = "viridis",
        density_cmap: str = "inferno_r",
        cbar_label: Optional[str] = None,
        hist_trim: int = 4,
        hist_bins: int = 20,
        file_name: Optional[str] = None,
    ) -> None:
        mse, r2, mae, pcc = self._calculate_metrics(
            np.asarray(predictions), np.asarray(values)
        )
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(values, list):
            values = np.array(values)
        if isinstance(groups, list):
            groups = pl.Series(groups)
        if groups is None:
            figsize = (10, 5.25)
        elif groups.unique().len() == 1:
            figsize = (10, 5.25)
            groups = None
        elif groups.dtype == pl.datatypes.Utf8:
            figsize = (12, 6.25)
            groups = groups.fill_null("NA").str.replace("N/A", "NA")
        else:
            figsize = (10, 5.25)
            groups = groups.fill_null(0)
        fig = plt.figure(figsize=figsize)
        axs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[5, 0.25])
        ax = plt.subplot(axs[0])
        hax = plt.subplot(axs[1])
        cax1 = plt.subplot(axs[2])
        cax2 = plt.subplot(axs[3])
        lower_bound, upper_bound, _ = self._set_axis_properties(
            ax, np.asarray(predictions), np.asarray(values), units
        )
        m, b = self._create_hexbin_plot(
            fig,
            ax,
            np.asarray(values),
            np.asarray(predictions),
            density_cmap,
            cax1,
            lower_bound,
            upper_bound,
        )
        self._create_histogram(
            fig,
            hax,
            np.asarray(predictions),
            np.asarray(values),
            units,
            hist_bins,
            hist_trim,
            cax2,
            groups,
            hist_cmap,
            cbar_label,
        )
        self._set_title(
            fig, title, pcc, r2, m, b, mse, mae, len(np.asarray(values))
        )
        if file_name:
            file_path = Path(file_name)
            if (
                file_path.parent in (Path(""), Path("."))
                and not file_path.is_absolute()
            ):
                file_path = self.output_dir / file_path.name
            else:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            plt.savefig(file_path, dpi=300)
        plt.show()
