"""Helpers for persisting CV_Result objects and their trained estimators.

Typical workflow:

    results = cv.run(param_path="hyperparameters.yaml")
    # flush out manifests, ndjson/csv and per-combo model pickles
    CV_Result.export_result(results, "out/cv_results")

Use `CV_Result.save_cv_result(result, path)` when you just need the
metadata for a single combo, and `CV_Result.save_model(estimator, path)` for
pickling models on demand.  `load_model` can restore those pickles later.
"""

import csv
import gzip
import hashlib
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


class CV_Result:
    """Container for CV outputs with simple serialization/export helpers.

    Public API and behavior are preserved from the previous implementation;
    this refactor only cleans internals for readability.
    """

    def __init__(
        self,
        feature_set: Optional[str] = None,
        label: Optional[str] = None,
        scheme: Optional[str] = None,
        cross_val_scores: Optional[List[float]] = None,
        validation_r2_per_fold: Optional[List[float]] = None,
        validation_mse_per_fold: Optional[List[float]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        trained_model: Optional[Any] = None,
    ) -> None:
        self.feature_set = feature_set
        self.label = label
        self.scheme = scheme

        # store lists defensively
        self.cross_val_scores = (
            list(cross_val_scores) if cross_val_scores else []
        )
        self.validation_r2_per_fold = (
            list(validation_r2_per_fold) if validation_r2_per_fold else []
        )
        self.validation_mse_per_fold = (
            list(validation_mse_per_fold) if validation_mse_per_fold else []
        )

        # optional: best hyperparameters (None when not provided)
        self.best_params: Optional[Dict[str, Any]] = (
            dict(best_params) if best_params is not None else None
        )

        # optional: trained estimator associated with this result
        self.model: Optional[Any] = trained_model

        # derived values
        self.avg_validation_r2: Optional[float] = None
        self.avg_validation_mse: Optional[float] = None
        self._compute_averages()

    def _compute_averages(self) -> None:
        """Compute and cache mean per-fold metrics (or None if unavailable)."""
        if self.validation_r2_per_fold:
            self.avg_validation_r2 = float(
                sum(self.validation_r2_per_fold)
                / len(self.validation_r2_per_fold)
            )
        else:
            self.avg_validation_r2 = None

        if self.validation_mse_per_fold:
            self.avg_validation_mse = float(
                sum(self.validation_mse_per_fold)
                / len(self.validation_mse_per_fold)
            )
        else:
            self.avg_validation_mse = None

    def _serialize_value(self, v: Any) -> Any:
        """Make values JSON/Polars-friendly.

        numpy scalars/arrays are converted, lists/tuples are recursively
        serialized, Paths become strings, and fall back to `str()` for unknown
        objects.
        """
        if isinstance(v, Path):
            return str(v)
        if v is None:
            return None
        if isinstance(v, (str, bool, int, float)):
            return v
        # numpy types
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        if isinstance(v, np.ndarray):
            return [self._serialize_value(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [self._serialize_value(x) for x in v]
        try:
            return str(v)
        except Exception:
            return None

    def _to_dict(self) -> dict:
        """Return a JSON-serializable representation of this result."""
        self._compute_averages()
        return {
            "feature_set": self._serialize_value(self.feature_set),
            "label": self._serialize_value(self.label),
            "scheme": self._serialize_value(self.scheme),
            "cross_val_scores": self._serialize_value(self.cross_val_scores),
            "validation_r2_per_fold": self._serialize_value(
                self.validation_r2_per_fold
            ),
            "avg_validation_r2": self._serialize_value(self.avg_validation_r2),
            "validation_mse_per_fold": self._serialize_value(
                self.validation_mse_per_fold
            ),
            "best_params": self._serialize_value(self.best_params),
            "avg_validation_mse": self._serialize_value(
                self.avg_validation_mse
            ),
        }

    # ----- Export helpers -----

    @staticmethod
    def _sanitize_segment(value: Optional[str], fallback: str) -> str:
        text = str(value) if value else fallback
        sanitized = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", text)
        return sanitized.strip("_") or fallback

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        return CV_Result._sanitize_segment(value, "model")

    @staticmethod
    def _model_file_name(key: Optional[str]) -> str:
        key_str = str(key) if key is not None else "model"
        short_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:8]
        label = CV_Result._sanitize_filename(key_str)[:120] or "model"
        return f"{label}_{short_hash}.pkl"

    @staticmethod
    def _normalize_results_input(
        values: Union[
            "CV_Result", Sequence["CV_Result"], Dict[str, "CV_Result"]
        ]
    ) -> Dict[str, "CV_Result"]:
        if isinstance(values, CV_Result):
            return {"result": values}
        if isinstance(values, dict):
            if not all(isinstance(v, CV_Result) for v in values.values()):
                raise TypeError("All values must be CV_Result instances")
            return values
        if isinstance(values, Sequence) and not isinstance(
            values, (str, bytes)
        ):
            normalized: Dict[str, "CV_Result"] = {}
            for idx, item in enumerate(values):
                if not isinstance(item, CV_Result):
                    raise TypeError(
                        "Sequence items must be CV_Result instances"
                    )
                normalized[f"result_{idx}"] = item
            return normalized
        raise TypeError("Unsupported results payload passed to export_result")

    @staticmethod
    def _extract_metadata(
        key: str, result: "CV_Result"
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        feature_set = result.feature_set
        label = result.label
        scheme = result.scheme
        model_name: Optional[str] = None
        if result.model is not None:
            model_name = result.model.__class__.__name__
        elif isinstance(key, str) and "::" in key:
            try:
                left, _ = key.rsplit("::", 1)
                if "::" in left:
                    _, model_name = left.rsplit("::", 1)
                else:
                    model_name = left
            except Exception:  # pragma: no cover - conservative fallback
                model_name = None
        return feature_set or "unknown_feature_set", label, scheme, model_name

    @staticmethod
    def _combo_dir_name(result: "CV_Result") -> str:
        feat = CV_Result._sanitize_segment(result.feature_set, "feature_set")
        label = CV_Result._sanitize_segment(result.label, "label")
        scheme = CV_Result._sanitize_segment(result.scheme, "scheme")
        return f"{feat};{label};{scheme}"

    @staticmethod
    def export_result(
        results: Union[
            "CV_Result", Sequence["CV_Result"], Dict[str, "CV_Result"]
        ],
        path: Union[str, Path],
        indent: int = 2,
    ) -> None:
        results_map = CV_Result._normalize_results_input(results)
        out_dir = Path(path)
        if out_dir.exists() and out_dir.is_file():
            out_dir = out_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        CV_Result.results_dict_to_streaming_files(results_map, out_dir)

        models_dir = out_dir / "models"
        model_files: List[str] = []
        for key, result in results_map.items():
            model_obj = getattr(result, "model", None)
            if model_obj is None:
                continue
            model_path = (
                models_dir
                / CV_Result._combo_dir_name(result)
                / CV_Result._model_file_name(key)
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            CV_Result.save_model(model_obj, model_path)
            model_files.append(str(model_path.relative_to(out_dir)))

        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "components": {
                "results": {
                    "files": [
                        "results.ndjson",
                        "results_summary.csv",
                        "results_folds.csv",
                    ],
                    "n_results": len(results_map),
                },
                "models": {
                    "path": "models",
                    "n_models": len(model_files),
                    "files": model_files,
                },
            },
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=indent)
        )

    @staticmethod
    def save_cv_result(
        result: "CV_Result",
        path: Union[str, Path],
    ) -> None:
        outp = Path(path)
        if outp.exists() and outp.is_dir():
            ndjson_path = outp / "results.ndjson"
        else:
            ndjson_path = outp
            ndjson_path.parent.mkdir(parents=True, exist_ok=True)
        with ndjson_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(result._to_dict(), default=str) + "\n")

    @staticmethod
    def results_dict_to_streaming_files(
        results_map: Dict[str, "CV_Result"], out_dir: Union[str, Path]
    ) -> None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        ndjson_path = outp / "results.ndjson"
        folds_path = outp / "results_folds.csv"
        summary_path = outp / "results_summary.csv"

        with (
            ndjson_path.open("w", encoding="utf-8") as ndj_f,
            folds_path.open("w", encoding="utf-8", newline="") as folds_f,
            summary_path.open("w", encoding="utf-8", newline="") as sum_f,
        ):
            folds_writer = csv.writer(folds_f)
            sum_writer = csv.writer(sum_f)

            folds_writer.writerow(
                [
                    "feature_set",
                    "label",
                    "scheme",
                    "model",
                    "fold_index",
                    "validation_r2",
                    "validation_mse",
                ]
            )
            sum_writer.writerow(
                [
                    "feature_set",
                    "label",
                    "scheme",
                    "model",
                    "avg_validation_r2",
                    "avg_validation_mse",
                    "cross_val_scores",
                ]
            )

            for key, res in results_map.items():
                (
                    feat_name,
                    label_name,
                    scheme_name,
                    model_name,
                ) = CV_Result._extract_metadata(key, res)

                out_record = res._to_dict()
                out_record["model"] = model_name
                ndj_f.write(json.dumps(out_record, default=str) + "\n")

                r2_list = out_record.get("validation_r2_per_fold", []) or []
                mse_list = out_record.get("validation_mse_per_fold", []) or []
                n = max(len(r2_list), len(mse_list))
                for i in range(n):
                    r2 = r2_list[i] if i < len(r2_list) else ""
                    mse = mse_list[i] if i < len(mse_list) else ""
                    folds_writer.writerow(
                        [
                            feat_name,
                            label_name,
                            scheme_name,
                            model_name,
                            i,
                            r2,
                            mse,
                        ]
                    )

                sum_writer.writerow(
                    [
                        feat_name,
                        label_name,
                        scheme_name,
                        model_name,
                        out_record.get("avg_validation_r2"),
                        out_record.get("avg_validation_mse"),
                        json.dumps(
                            out_record.get("cross_val_scores", []), default=str
                        ),
                    ]
                )

    @staticmethod
    def save_model(
        model: Any, path: Union[str, Path], compress: bool = False
    ) -> None:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with gzip.open(str(outp), "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with outp.open("wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(path: Union[str, Path]) -> Any:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if str(p).endswith(".gz"):
            with gzip.open(str(p), "rb") as f:
                return pickle.load(f)
        with p.open("rb") as f:
            return pickle.load(f)
