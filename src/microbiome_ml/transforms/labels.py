"""Label (target) transforms, fitted per fold and scored in label units.

A model can be trained on a transformed label -- z-scored, Yeo-Johnson,
quantile-normalised, log1p -- while every score the pipeline reports
stays in the label's own units (°C, pH, µmol/kg). `wrap_target` puts
the regressor inside a `TargetTransformed`, sklearn's
TransformedTargetRegressor with three differences:

* The transform is fitted inside every `fit`: on each CV fold's training
  rows, then on the whole training pool, then on the holdout's training
  split. The validation and test labels never shape it.
* `predict` converts back, so CV R², holdout R² and the final fit's
  in-sample R² are all on the original scale and comparable across
  transforms. (A back-converted log is the median, not the mean, which
  costs a little R²; that cost is part of what the sweep measures.)
* It is transparent to the rest of the pipeline. Hyperparameters are
  the regressor's own (`max_depth`, not `regressor__max_depth`), reports
  name the regressor (`RandomForestRegressor`), and fitted attributes
  such as `feature_importances_` are the regressor's. The out-of-bag
  score is recomputed in label units.

A label transform changes a tree model: splits and leaf means are
squared error on the transformed scale. Even z-score and min-max, which
are affine and should in exact arithmetic leave a forest's trees as
they were, do not in practice: sklearn's random forest grew different
trees on a z-scored label (predictions up to 2.8 apart on a 1-400
label, CV MSE 160 vs 149 on 60 samples), because near-tied splits round
differently on the two scales. Read a z-score/min-max difference for a
forest as noise of that size, not as an effect. Where these two do
matter is the SVR, whose epsilon is in label units.

`label_variants` names the runs a sweep makes: `ph` for no transform,
`ph@yeo_johnson` for one, so every transform of a label is trained and
tested on the same split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)

NONE = "none"
VARIANT_SEP = "@"
# Boltzmann constant in eV/K, for the Arrhenius scale 1/kT.
K_B = 8.617333262e-5
KELVIN = 273.15

# name -> what it is and which labels it is for (for the docs and errors)
TRANSFORMS: Dict[str, str] = {
    NONE: "the label as it is",
    "zscore": "(y - mean) / sd; trees: noise only; matters for SVR",
    "minmax": "scaled to [0, 1]; trees: noise only; matters for SVR",
    "log1p": "log(1 + y), y > -1; right-skewed labels such as oxygen",
    "sqrt": "sqrt(y), y >= 0; a milder log",
    "yeo_johnson": "power transform fitted to the data, any sign",
    "box_cox": "power transform fitted to the data, y > 0 only",
    "quantile_normal": "ranks mapped onto a normal distribution",
    "quantile_uniform": "ranks mapped onto [0, 1]",
    "inv_kelvin": "1/kT in 1/eV (Arrhenius scale), temperature in °C",
}

_ALIASES = {
    "": NONE,
    "null": NONE,
    "raw": NONE,
    "identity": NONE,
    "z": "zscore",
    "standard": "zscore",
    "standardise": "zscore",
    "standardize": "zscore",
    "min_max": "minmax",
    "log": "log1p",
    "yeojohnson": "yeo_johnson",
    "yj": "yeo_johnson",
    "boxcox": "box_cox",
    "quantile": "quantile_normal",
    "arrhenius": "inv_kelvin",
    "inverse_temperature": "inv_kelvin",
}

# Smallest value each function transform accepts (exclusive unless noted).
_DOMAIN: Dict[str, Tuple[float, bool, str]] = {
    "log1p": (-1.0, False, "log1p needs every value > -1"),
    "sqrt": (0.0, True, "sqrt needs every value >= 0"),
    "box_cox": (0.0, False, "box_cox needs every value > 0; use yeo_johnson"),
    "inv_kelvin": (-KELVIN, False, "inv_kelvin needs °C above -273.15"),
}


def canonical_target_transform(name: Optional[str]) -> str:
    """The registered name for *name*; ValueError for an unknown one."""
    text = "" if name is None else str(name).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = _ALIASES.get(text, text)
    if text not in TRANSFORMS:
        raise ValueError(
            f"Unknown target transform {name!r}; known: {', '.join(TRANSFORMS)}"
        )
    return text


# Module-level so a fitted model pickles.
def _to_inv_kelvin(t: np.ndarray) -> np.ndarray:
    # Annotated, not returned directly: numpy's stubs type the
    # arithmetic as Any, which mypy's no-any-return rejects.
    out: np.ndarray = 1.0 / (K_B * (np.asarray(t, dtype=float) + KELVIN))
    return out


def _from_inv_kelvin(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (K_B * np.asarray(x, dtype=float)) - KELVIN
    return out


_FUNCTIONS = {
    "log1p": (np.log1p, np.expm1),
    "sqrt": (np.sqrt, np.square),
    "inv_kelvin": (_to_inv_kelvin, _from_inv_kelvin),
}


def _make_transformer(name: str) -> Any:
    if name == "zscore":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "yeo_johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if name == "box_cox":
        return PowerTransformer(method="box-cox", standardize=True)
    if name in ("quantile_normal", "quantile_uniform"):
        return QuantileTransformer(
            output_distribution=name.split("_")[1], random_state=42
        )
    return None


def check_domain(name: str, y: Any) -> None:
    """Raise if *y* has values the transform cannot take."""
    if name not in _DOMAIN:
        return
    floor, inclusive, message = _DOMAIN[name]
    values = np.asarray(y, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low = float(values.min())
    if low < floor or (low == floor and not inclusive):
        raise ValueError(f"{message} (smallest value {low:g})")


class TargetTransformed(TransformedTargetRegressor):
    """A regressor fitted on a transformed label, predicting in label units.

    Its parameters are `regressor` and `transform` (a name from
    `TRANSFORMS`); the transformer itself is derived from the name, so a
    clone or a pickle carries it. See the module docstring for how it
    stays transparent to the pipeline.
    """

    _parameter_constraints: dict = {
        **TransformedTargetRegressor._parameter_constraints,
        "transform": [str],
    }

    def __init__(
        self, regressor: Any = None, *, transform: str = NONE
    ) -> None:
        # Deliberately not super().__init__: transformer/func/inverse_func
        # are derived from `transform` (properties below), not parameters.
        self.regressor = regressor
        self.transform = transform

    # --- what TransformedTargetRegressor.fit reads ----------------------
    @property
    def transformer(self) -> Any:
        return _make_transformer(canonical_target_transform(self.transform))

    @property
    def func(self) -> Any:
        pair = _FUNCTIONS.get(canonical_target_transform(self.transform))
        return None if pair is None else pair[0]

    @property
    def inverse_func(self) -> Any:
        pair = _FUNCTIONS.get(canonical_target_transform(self.transform))
        return None if pair is None else pair[1]

    @property
    def check_inverse(self) -> bool:
        # Quantile maps are not exactly invertible at ties and bounds, and
        # sklearn would warn on every fold; the round trip is tested instead.
        return False

    # --- transparent to the rest of the pipeline -------------------------
    @property
    def model_name(self) -> str:
        """The regressor's class name, which is what reports should say."""
        # No regressor means sklearn's default, a LinearRegression.
        return estimator_name(self.regressor) or "LinearRegression"

    @property
    def fitted_model(self) -> Any:
        """The fitted regressor (on the transformed scale), e.g. for SHAP."""
        return self.__dict__.get("regressor_")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(super().get_params(deep=deep))
        if deep and self.regressor is not None:
            # The regressor's own names too, so `n_jobs`, `oob_score`
            # and grid keys are found without the `regressor__` prefix.
            for key, value in self.regressor.get_params(deep=True).items():
                params.setdefault(key, value)
        return params

    def set_params(self, **params: Any) -> "TargetTransformed":
        own = set(super().get_params(deep=False))
        mine = {k: v for k, v in params.items() if k.split("__")[0] in own}
        theirs = {k: v for k, v in params.items() if k not in mine}
        if mine:
            super().set_params(**mine)
        if theirs:
            if self.regressor is None:
                raise ValueError(
                    f"No regressor to take parameters {sorted(theirs)}"
                )
            self.regressor.set_params(**theirs)
        return self

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "TargetTransformed":
        name = canonical_target_transform(self.transform)
        check_domain(name, y)
        super().fit(X, y, **fit_params)
        # A forest's own oob_score_ is R² on the transformed scale; redo
        # it on the original one so it compares with every other score.
        self.__dict__.pop("oob_score_", None)
        oob = getattr(self.regressor_, "oob_prediction_", None)
        if oob is not None:
            pred = np.asarray(oob, dtype=float).reshape(-1, 1)
            back = self.transformer_.inverse_transform(pred).ravel()
            truth = np.asarray(y, dtype=float).ravel()
            ok = np.isfinite(back) & np.isfinite(truth)
            sst = float(((truth[ok] - truth[ok].mean()) ** 2).sum())
            if ok.any() and sst > 0:
                sse = float(((truth[ok] - back[ok]) ** 2).sum())
                self.oob_score_ = 1.0 - sse / sst
        return self

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails. Fitted attributes
        # (`feature_importances_`, `coef_`) come from the regressor; the
        # out-of-bag ones do not, as theirs are on the transformed scale.
        if (
            name.endswith("_")
            and not name.startswith("_")
            and name not in ("regressor_", "transformer_")
            and not name.startswith("oob_")
        ):
            inner = self.__dict__.get("regressor_")
            if inner is not None:
                return getattr(inner, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


def wrap_target(estimator: Any, transform: Optional[str]) -> Any:
    """*estimator* trained on the transformed label; as is for `none`."""
    name = canonical_target_transform(transform)
    if name == NONE:
        return estimator
    return TargetTransformed(regressor=estimator, transform=name)


def estimator_name(estimator: Any) -> Optional[str]:
    """The name reports use: the regressor's, through any wrapper."""
    if estimator is None:
        return None
    named = getattr(estimator, "model_name", None)
    return named if isinstance(named, str) else type(estimator).__name__


# ---------------------------------------------------------------------------
# Label variants and limits (config handling)
# ---------------------------------------------------------------------------
def _applies(label: str, key: str) -> bool:
    """`oxy` covers oxy, geo_oxy and comb_oxy; an exact name only itself."""
    return label == key or label.endswith(f"_{key}")


def transforms_for(label: str, spec: Any) -> List[str]:
    """The transforms *label* is trained with, from `cv.target_transform`.

    A string or a list applies to every label. A mapping gives lists per
    label, `default` for the rest; a key also covers its `geo_` and
    `comb_` versions (`oxy` -> geo_oxy, comb_oxy), and an exact name
    beats such a match.
    """
    if spec is None:
        return [NONE]
    if isinstance(spec, Mapping):
        if label in spec:
            chosen = spec[label]
        else:
            matched = [k for k in spec if k != "default" and _applies(label, k)]
            chosen = spec[matched[0]] if matched else spec.get("default", NONE)
    else:
        chosen = spec
    names = [chosen] if isinstance(chosen, str) or chosen is None else chosen
    return list(dict.fromkeys(canonical_target_transform(n) for n in names))


def variant_name(label: str, transform: str) -> str:
    name = canonical_target_transform(transform)
    return label if name == NONE else f"{label}{VARIANT_SEP}{name}"


def label_variants(label: str, spec: Any) -> List[str]:
    """`ph`, `ph@yeo_johnson`, ... -- one CV job each, on ph's split."""
    return [variant_name(label, t) for t in transforms_for(label, spec)]


def split_label_variant(name: str) -> Tuple[str, str]:
    """`ph@yeo_johnson` -> ('ph', 'yeo_johnson'); `ph` -> ('ph', 'none')."""
    base, sep, transform = str(name).partition(VARIANT_SEP)
    return base, canonical_target_transform(transform if sep else NONE)


def apply_label_limits(
    labels: pl.DataFrame,
    limits: Optional[Mapping[str, Sequence[Optional[float]]]],
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """Null every label value outside its [low, high] (inclusive).

    For values that cannot be real -- above 122 °C, a pH of 15, negative
    oxygen -- so they are dropped like any missing label rather than
    trained on. A key covers its `geo_`/`comb_` versions as in
    `transforms_for`; `null` leaves that side open. Returns the table
    and how many values each column lost.
    """
    if not limits or labels is None:
        return labels, {}
    removed: Dict[str, int] = {}
    exprs: List[pl.Expr] = []
    for column in labels.columns:
        if column == "sample":
            continue
        bounds = [b for k, b in limits.items() if _applies(column, k)]
        if not bounds:
            continue
        low = max((float(b[0]) for b in bounds if b[0] is not None), default=None)
        high = min((float(b[1]) for b in bounds if b[1] is not None), default=None)
        value = pl.col(column).cast(pl.Float64, strict=False)
        bad = pl.lit(False)
        if low is not None:
            bad = bad | (value < low)
        if high is not None:
            bad = bad | (value > high)
        removed[column] = int(labels.select(bad.fill_null(False).sum()).item())
        exprs.append(
            pl.when(bad.fill_null(False))
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )
    return (labels.with_columns(exprs) if exprs else labels), removed


def describe() -> str:
    """One line per transform, for logs and --help."""
    return "\n".join(f"  {k:<16} {v}" for k, v in TRANSFORMS.items())


__all__ = [
    "TRANSFORMS",
    "TargetTransformed",
    "apply_label_limits",
    "canonical_target_transform",
    "check_domain",
    "estimator_name",
    "label_variants",
    "split_label_variant",
    "transforms_for",
    "variant_name",
    "wrap_target",
]
