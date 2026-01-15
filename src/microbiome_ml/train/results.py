# microbiome_ml/src/microbiome_ml/train/results.py

import csv
import gzip
import json
import pickle
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

        Preserves previous behavior: numpy scalars/arrays are converted,
        lists/tuples are recursively serialized, Paths become strings, and
        fall back to `str()` for unknown objects.
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

    def to_json(
        self, path: Optional[Union[str, Path]] = None, indent: int = 2
    ) -> str:
        j = json.dumps(self._to_dict(), indent=indent)
        if path:
            Path(path).write_text(j)
        return j

    def summary(self, save_json: Optional[Union[str, Path]] = None) -> dict:
        """Print a short human summary and optionally persist JSON."""
        self._compute_averages()
        print(f"Feature set: {self.feature_set}")
        print(f"Label: {self.label}")
        print(f"Scheme: {self.scheme}")
        print(f"avg_validation_r2: {self.avg_validation_r2}")
        print(f"avg_validation_mse: {self.avg_validation_mse}")
        print(f"cross_val_scores: {self.cross_val_scores}")

        out = self._to_dict()
        if save_json:
            Path(save_json).write_text(json.dumps(out, indent=2))
        return out

    # ----- Export helpers -----
    @staticmethod
    def export_result(
        results_list: Union[List["CV_Result"], Dict[str, "CV_Result"]],
        path: Union[str, Path],
        indent: int = 2,
    ) -> None:
        outp = Path(path)
        if isinstance(results_list, dict):
            if outp.exists() and outp.is_file():
                outp = outp.parent
            CV_Result.results_dict_to_streaming_files(results_list, outp)
            return

        rows = [r._to_dict() for r in results_list]
        if outp.exists() and outp.is_dir():
            ndjson_path = outp / "results.ndjson"
        else:
            ndjson_path = outp
            ndjson_path.parent.mkdir(parents=True, exist_ok=True)

        with ndjson_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

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
                model_name = None
                feat_name = res.feature_set
                label_name = res.label
                scheme_name = res.scheme
                if isinstance(key, str) and "::" in key:
                    try:
                        left, model_name = key.rsplit("::", 1)
                        parts = left.split("::")
                        if len(parts) >= 1:
                            feat_name = parts[0]
                        if len(parts) >= 2:
                            label_name = parts[1]
                        if len(parts) >= 3:
                            scheme_name = parts[2]
                    except Exception:
                        model_name = None

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

    def save(self, path: Union[str, Path], compress: bool = False) -> None:
        outp = Path(path)
        if compress and str(outp).endswith(".tar.gz"):
            work_dir = Path(str(outp)[:-7])
        elif compress:
            work_dir = outp
            outp = Path(str(outp) + ".tar.gz")
        else:
            work_dir = outp

        work_dir.mkdir(parents=True, exist_ok=True)
        self.results_dict_to_streaming_files({"result": self}, work_dir)

        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "components": {
                "results": {
                    "files": [
                        "results.ndjson",
                        "results_folds.csv",
                        "results_summary.csv",
                    ],
                    "n_results": 1,
                }
            },
        }
        (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        if compress:
            with tarfile.open(outp, "w:gz") as tar:
                tar.add(work_dir, arcname=work_dir.name)
            shutil.rmtree(work_dir)

    def save_model(
        self, model: Any, path: Union[str, Path], compress: bool = False
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

    @staticmethod
    def save_mapping(
        results_map: Dict[str, "CV_Result"],
        path: Union[str, Path],
        compress: bool = False,
    ) -> None:
        outp = Path(path)
        if compress and str(outp).endswith(".tar.gz"):
            work_dir = Path(str(outp)[:-7])
        elif compress:
            work_dir = outp
            outp = Path(str(outp) + ".tar.gz")
        else:
            work_dir = outp

        work_dir.mkdir(parents=True, exist_ok=True)
        CV_Result.results_dict_to_streaming_files(results_map, work_dir)

        n_results = len(results_map)
        models = []
        for k in results_map.keys():
            if isinstance(k, str) and "::" in k:
                try:
                    _, model = k.rsplit("::", 1)
                    models.append(model)
                except Exception:
                    pass

        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "components": {
                "results": {
                    "files": [
                        "results.ndjson",
                        "results_folds.csv",
                        "results_summary.csv",
                    ],
                    "n_results": n_results,
                    "models": list(sorted(set(models))) if models else None,
                }
            },
        }
        (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        if compress:
            with tarfile.open(outp, "w:gz") as tar:
                tar.add(work_dir, arcname=work_dir.name)
            shutil.rmtree(work_dir)
