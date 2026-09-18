"""SHAP-based feature attribution for trained microbiome-ML models.

``shap`` is an **optional** dependency. Importing this module always succeeds;
calling :meth:`SHAPAnalyser.compute` raises ``ImportError`` with an install
hint when shap is absent.

Typical usage after holdout training::

    from microbiome_ml.train.shap_analysis import SHAPAnalyser

    analyser = SHAPAnalyser(
        model=evaluation.estimator,
        X=X_test,
        feature_names=evaluation.feature_names,
        max_background=100,
    )
    result = analyser.compute()
    result.save_summary("out/shap_summary.csv")

For multi-label parallel runs use the static helper::

    jobs = [(ev.estimator, X, ev.feature_names) for ev in evaluations]
    shap_results = SHAPAnalyser.compute_parallel(jobs, n_jobs=-1)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_SHAP_INSTALL_MSG = (
    "shap is not installed. "
    "Install it with: pip install shap  "
    "or add it to your pixi environment with: pixi add shap"
)

# Classes for which shap.TreeExplainer is applicable.
_TREE_MODEL_CLASSES = frozenset(
    {
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "LGBMRegressor",
        "LGBMClassifier",
    }
)


@dataclass
class SHAPResult:
    """SHAP attribution output for one model/dataset pair.

    Attributes:
        shap_values: Array of shape ``(n_samples, n_features)``.
        base_value: Model expected output (SHAP intercept).
        feature_names: Ordered feature column names.
        X: Feature matrix the SHAP values were computed on (same shape as
            ``shap_values``). Optional; needed for ``direction`` and for
            colour-coded beeswarm plots.
        mean_abs_shap: Per-feature ``mean(|SHAP|)`` — magnitude only.
        mean_shap: Per-feature signed ``mean(SHAP)``. Positive means the
            feature pushes predictions up on average across samples.
        direction: Per-feature Spearman correlation between feature value
            and SHAP value, in [-1, 1]. Positive: higher abundance -> higher
            prediction; negative: higher abundance -> lower prediction.
            ``nan`` when ``X`` is missing or the feature is constant.
    """

    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    X: Optional[np.ndarray] = None
    mean_abs_shap: np.ndarray = field(init=False)
    mean_shap: np.ndarray = field(init=False)
    direction: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.shap_values = np.asarray(self.shap_values, dtype=float)
        self.mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        self.mean_shap = self.shap_values.mean(axis=0)
        self.direction = self._compute_direction()

    def _compute_direction(self) -> np.ndarray:
        """Spearman correlation of feature value vs SHAP value, per feature.

        Rank-based so it is robust to the skewed, zero-inflated abundance
        distributions typical of microbiome features.
        """
        n_features = self.shap_values.shape[1]
        direction: np.ndarray = np.full(n_features, np.nan, dtype=float)
        if self.X is None:
            return direction
        X = np.asarray(self.X, dtype=float)
        if X.shape != self.shap_values.shape:
            logger.warning(
                "X shape %s does not match shap_values shape %s; "
                "direction not computed",
                X.shape,
                self.shap_values.shape,
            )
            return direction

        from scipy.stats import spearmanr

        for j in range(n_features):
            x = X[:, j]
            s = self.shap_values[:, j]
            if np.ptp(x) == 0 or np.ptp(s) == 0:
                continue  # constant column: correlation undefined
            rho = spearmanr(x, s).correlation
            direction[j] = float(rho) if np.isfinite(rho) else np.nan
        return direction

    def top_features(self, n: int = 20) -> List[str]:
        """Return the top-n feature names sorted by mean |SHAP|."""
        order = np.argsort(self.mean_abs_shap)[::-1]
        return [self.feature_names[i] for i in order[:n]]

    def top_features_signed(self, n: int = 20) -> List[Tuple[str, float]]:
        """Top-n ``(feature, direction)`` pairs sorted by mean |SHAP|.

        ``direction`` is the Spearman correlation described on the class;
        ``nan`` when it could not be computed.
        """
        order = np.argsort(self.mean_abs_shap)[::-1]
        return [
            (self.feature_names[i], float(self.direction[i]))
            for i in order[:n]
        ]

    def save(self, path: Union[str, Path]) -> None:
        """Write raw SHAP values to CSV (rows=samples, cols=features)."""
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.feature_names)
            for row in self.shap_values:
                writer.writerow([float(v) for v in row])
        logger.info(
            "Saved SHAP values (%d rows) to %s", len(self.shap_values), outp
        )

    def save_summary(self, path: Union[str, Path]) -> None:
        """Write per-feature summary ranked by mean |SHAP|.

        Columns: ``rank, feature, mean_abs_shap, mean_shap, direction``.
        ``direction`` is empty when it could not be computed.
        """
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        order = np.argsort(self.mean_abs_shap)[::-1]
        with outp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["rank", "feature", "mean_abs_shap", "mean_shap", "direction"]
            )
            for rank, idx in enumerate(order, start=1):
                i = int(idx)
                dir_val = self.direction[i]
                writer.writerow(
                    [
                        rank,
                        self.feature_names[i],
                        float(self.mean_abs_shap[i]),
                        float(self.mean_shap[i]),
                        float(dir_val) if np.isfinite(dir_val) else "",
                    ]
                )
        logger.info(
            "Saved SHAP summary (%d features) to %s",
            len(self.feature_names),
            outp,
        )


class SHAPAnalyser:
    """Compute SHAP attributions for a fitted sklearn-compatible estimator.

    Uses ``shap.TreeExplainer`` for tree-based models (RF, GB, XGB, LGBM)
    and falls back to ``shap.Explainer`` for everything else.

    Args:
        model: Fitted estimator.
        X: Feature matrix to explain, shape ``(n_samples, n_features)``.
        feature_names: Column names for X. Falls back to
            ``model.feature_names_in_`` then ``feature_0…feature_n``.
        max_background: Maximum background samples for the explainer.
            Keeps memory and compute cost bounded on large datasets.
        n_jobs: Used by :meth:`compute_parallel` when distributing
            independent SHAP jobs across CPUs (e.g. multi-label runs).
    """

    def __init__(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_background: int = 100,
        n_jobs: int = 1,
    ) -> None:
        self.model = model
        self.X: np.ndarray = np.asarray(X, dtype=float)
        self.max_background = max_background
        self.n_jobs = n_jobs
        self.feature_names: List[str] = self._resolve_feature_names(
            feature_names, model, self.X.shape[1]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_feature_names(
        names: Optional[List[str]], model: Any, n: int
    ) -> List[str]:
        if names and len(names) == n:
            return [str(s) for s in names]
        inferred = getattr(model, "feature_names_in_", None)
        if inferred is not None:
            candidate = [str(s) for s in inferred]
            if len(candidate) == n:
                return candidate
        return [f"feature_{i}" for i in range(n)]

    @staticmethod
    def _is_tree_model(model: Any) -> bool:
        return type(model).__name__ in _TREE_MODEL_CLASSES

    def _sample_background(self) -> np.ndarray:
        """Return a random subsample of X for use as background data."""
        n = self.X.shape[0]
        if n <= self.max_background:
            return self.X
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n, size=self.max_background, replace=False)
        background: np.ndarray = self.X[idx]
        return background

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self) -> SHAPResult:
        """Compute SHAP values and return a :class:`SHAPResult`.

        Raises:
            ImportError: When ``shap`` is not installed.
        """
        try:
            import shap  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(_SHAP_INSTALL_MSG) from exc

        background = self._sample_background()
        model_name = type(self.model).__name__

        if self._is_tree_model(self.model):
            logger.info(
                "TreeExplainer for %s (background=%d)",
                model_name,
                len(background),
            )
            explainer = shap.TreeExplainer(
                self.model,
                data=background,
                feature_perturbation="interventional",
            )
            raw = explainer.shap_values(self.X, check_additivity=False)
        else:
            logger.info(
                "Generic shap.Explainer for %s (background=%d)",
                model_name,
                len(background),
            )
            explainer = shap.Explainer(self.model, background)
            explanation = explainer(self.X)
            raw = explanation.values

        shap_values = np.asarray(raw, dtype=float)
        # Multi-output models return shape (n_samples, n_features, n_outputs);
        # take the first output.
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        base = explainer.expected_value
        base_value = float(base[0] if hasattr(base, "__len__") else base)

        return SHAPResult(
            shap_values=shap_values,
            base_value=base_value,
            feature_names=self.feature_names,
            X=self.X,
        )

    # ------------------------------------------------------------------
    # Parallel helper for multi-label runs
    # ------------------------------------------------------------------

    @staticmethod
    def compute_parallel(
        jobs: List[tuple],  # (model, X, feature_names | None)
        max_background: int = 100,
        n_jobs: int = -1,
    ) -> List[SHAPResult]:
        """Compute SHAP for multiple (model, X, feature_names) in parallel.

        Each tuple in ``jobs`` is ``(model, X, feature_names)`` where
        ``feature_names`` may be ``None``.

        Args:
            jobs: List of ``(model, X, feature_names)`` tuples.
            max_background: Background subsample size applied per job.
            n_jobs: Joblib parallel workers (``-1`` = all CPUs).

        Returns:
            List of :class:`SHAPResult` in the same order as ``jobs``.

        Raises:
            ImportError: When ``shap`` is not installed.
        """
        # Trigger the import check early with a clear message.
        try:
            import shap as _shap  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(_SHAP_INSTALL_MSG) from exc

        from joblib import Parallel, delayed  # noqa: PLC0415

        def _run(
            model: Any,
            X: np.ndarray,
            feature_names: Optional[List[str]],
        ) -> SHAPResult:
            return SHAPAnalyser(
                model=model,
                X=X,
                feature_names=feature_names,
                max_background=max_background,
            ).compute()

        results: List[SHAPResult] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
            delayed(_run)(m, x, fn) for m, x, fn in jobs
        )
        return results
