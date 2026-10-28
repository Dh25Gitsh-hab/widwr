"""
The pre-registered candidate families and settings grids (existing_data/protocol.json).

Regression configurations predict error_mw (corrected forecast = forecast_mw - prediction);
classification configurations predict P(error_mw >= 1000 MW). All preprocessing sits inside
each estimator, so it is fitted on training rows only.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression, QuantileRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from app.services.feature_builder import FEATURE_COLUMNS
from ml.v2.features import WIND_FEATURES

SEED = 42
LEVEL = ("forecast_mw",)
EXISTING9 = tuple(FEATURE_COLUMNS)
CURVE = ("forecast_mw", "hour_sin", "hour_cos")
WIND_ALL = tuple(WIND_FEATURES)
INTERACTION_COLUMNS = ["forecast_mw", "hour_sin", "hour_cos", "err_last3h_mw", "err_7d_mw", "forecast_spread_mw", "forecast_revision_mw"]


@dataclass(frozen=True)
class Config:
    family: str
    task: str  # "regression" | "classification"
    approach: str
    features: tuple[str, ...]
    build: Callable[[], object] | None = None
    kind: str = "model"  # "model" | "official" | "rolling_bias" | "rolling_offset"
    rolling_days: int | None = None
    params: tuple = ()

    @property
    def name(self) -> str:
        return self.family + (f"[{','.join(f'{k}={v}' for k, v in self.params)}]" if self.params else "")


# ---- preprocessing (fitted inside each training set) -----------------------------------------


def _spline(knots: int = 5) -> SplineTransformer:
    return SplineTransformer(n_knots=knots, degree=3, knots="quantile", extrapolation="linear")


def _level() -> ColumnTransformer:
    return ColumnTransformer([("level", "passthrough", ["forecast_mw"])])


def _curve(knots: int = 5) -> ColumnTransformer:
    return ColumnTransformer([("level", _spline(knots), ["forecast_mw"]), ("hour", "passthrough", ["hour_sin", "hour_cos"])])


def _wind_all() -> ColumnTransformer:
    rest = [f for f in WIND_FEATURES if f != "forecast_mw"]
    return ColumnTransformer([("level", _spline(), ["forecast_mw"]),
                              ("rest", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), rest)])  # fmt: skip


def _interactions() -> ColumnTransformer:
    pairs = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PolynomialFeatures(2, interaction_only=True, include_bias=False))
    return ColumnTransformer([("level", _spline(), ["forecast_mw"]), ("pairs", pairs, INTERACTION_COLUMNS)])


def _median(alpha: float = 0.0) -> QuantileRegressor:
    return QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")


def _grid(**values) -> list[dict]:
    keys = list(values)
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values.values())]


def _family(family, task, approach, features, builder, grid=None, **extra) -> list[Config]:
    grid = grid or [{}]
    return [Config(family, task, approach, features, (lambda p=p: builder(**p)) if builder else None,
                   params=tuple(p.items()), **extra) for p in grid]  # fmt: skip


def _rolling(family, task, approach, features, builder, windows, kind="model") -> list[Config]:
    return [Config(family, task, approach, features, builder, kind=kind, rolling_days=w, params=(("W", w),)) for w in windows]


RF_GRID = _grid(min_samples_leaf=[20, 50, 100], max_features=[0.3, 0.5, 1.0])
GBM_GRID = _grid(learning_rate=[0.03, 0.1], max_leaf_nodes=[7, 15], min_samples_leaf=[40, 100])


def _gbm(**p) -> dict:
    return dict(max_iter=300, l2_regularization=1.0, monotonic_cst={"forecast_mw": 1}, random_state=SEED, **p)


REGRESSION: list[Config] = [
    Config("R00_official", "regression", "baseline", LEVEL, kind="official"),
    *_family("R01_linear_level", "regression", "existing model", LEVEL, lambda: make_pipeline(_level(), LinearRegression())),
    *_family("R02_random_forest", "regression", "tune existing", EXISTING9,
             lambda **p: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **p), RF_GRID),
    *_family("R10_spline_level", "regression", "nonlinear", LEVEL,
             lambda n_knots: make_pipeline(ColumnTransformer([("level", _spline(n_knots), ["forecast_mw"])]), LinearRegression()),
             _grid(n_knots=[3, 5, 7])),
    *_family("R11_isotonic_level", "regression", "nonlinear", LEVEL,
             lambda: make_pipeline(_level(), IsotonicRegression(increasing=True, out_of_bounds="clip"))),
    *_family("R12_spline_level_hour", "regression", "nonlinear", CURVE,
             lambda n_knots: make_pipeline(_curve(n_knots), LinearRegression()), _grid(n_knots=[3, 5, 7])),
    *_family("R13_ridge_wind_all", "regression", "nonlinear", WIND_ALL,
             lambda alpha: make_pipeline(_wind_all(), Ridge(alpha=alpha)), _grid(alpha=[1, 10, 100, 1000])),
    *_family("R14_huber_level", "regression", "nonlinear", LEVEL,
             lambda epsilon: make_pipeline(_level(), StandardScaler(), HuberRegressor(epsilon=epsilon, max_iter=2000)),
             _grid(epsilon=[1.35, 2.0])),
    *_family("R20_median_level", "regression", "MAE objective", LEVEL, lambda: make_pipeline(_level(), _median())),
    *_family("R21_median_curve", "regression", "MAE objective", CURVE,
             lambda n_knots: make_pipeline(_curve(n_knots), _median()), _grid(n_knots=[3, 5, 7])),
    *_family("R22_median_wind_all", "regression", "MAE objective", WIND_ALL,
             lambda alpha: make_pipeline(_wind_all(), _median(alpha)), _grid(alpha=[0, 1, 10])),
    *_family("R23_gbm_absolute", "regression", "MAE objective", WIND_ALL,
             lambda **p: HistGradientBoostingRegressor(loss="absolute_error", **_gbm(**p)), GBM_GRID),
    *_rolling("R30_rolling_mean_bias", "regression", "drift", LEVEL, None, (7, 14, 28), kind="rolling_bias"),
    *_rolling("R31_linear_plus_rolling_offset", "regression", "drift", LEVEL,
              lambda: make_pipeline(_level(), LinearRegression()), (7, 14, 28), kind="rolling_offset"),
    *_rolling("R32_linear_rolling_refit", "regression", "drift", LEVEL, lambda: make_pipeline(_level(), LinearRegression()), (14, 28)),
    *_rolling("R33_median_curve_rolling_refit", "regression", "drift", CURVE, lambda: make_pipeline(_curve(5), _median()), (14, 28)),
]  # fmt: skip

CLASSIFICATION: list[Config] = [
    *_family("C00_base_rate", "classification", "baseline", LEVEL, lambda: DummyClassifier(strategy="prior")),
    *_family("C01_logistic_level", "classification", "existing model", LEVEL,
             lambda: make_pipeline(_level(), StandardScaler(), LogisticRegression())),
    *_family("C02_random_forest", "classification", "tune existing", EXISTING9,
             lambda **p: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **p), RF_GRID),
    *_family("C10_logistic_curve", "classification", "nonlinear", CURVE,
             lambda n_knots: make_pipeline(_curve(n_knots), LogisticRegression(max_iter=5000)), _grid(n_knots=[3, 5, 7])),
    *_family("C11_logistic_wind_all", "classification", "features + weighting", WIND_ALL,
             lambda C, class_weight: make_pipeline(_wind_all(), LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
             _grid(C=[0.01, 0.1, 1, 10], class_weight=[None, "balanced"])),
    *_family("C12_logistic_interactions", "classification", "interactions", tuple(dict.fromkeys(("forecast_mw", *INTERACTION_COLUMNS))),
             lambda C: make_pipeline(_interactions(), LogisticRegression(C=C, max_iter=5000)), _grid(C=[0.1, 1])),
    *_family("C13_gbm_classifier", "classification", "gradient boosting", WIND_ALL,
             lambda **p: HistGradientBoostingClassifier(**_gbm(**p)), GBM_GRID),
    *_rolling("C14_logistic_curve_rolling_refit", "classification", "drift", CURVE,
              lambda: make_pipeline(_curve(5), LogisticRegression(max_iter=5000)), (14, 28)),
]  # fmt: skip

CONFIGS = {c.name: c for c in REGRESSION + CLASSIFICATION}
PRODUCTION_SPEC = {"regression": "R01_linear_level", "classification": "C01_logistic_level"}
STATIC_COUNTERPART = {"R30_rolling_mean_bias": "R01_linear_level", "R31_linear_plus_rolling_offset": "R01_linear_level",
                      "R32_linear_rolling_refit": "R01_linear_level", "R33_median_curve_rolling_refit": "R21_median_curve",
                      "C14_logistic_curve_rolling_refit": "C10_logistic_curve"}  # fmt: skip


def fit(config: Config, train: pd.DataFrame):
    if config.kind in ("official", "rolling_bias"):
        return None
    target = train["error_mw"] if config.task == "regression" else train["shortfall_1gw"].astype(int)
    if config.task == "classification" and target.nunique() < 2:  # a short rolling window can hold one class
        return DummyClassifier(strategy="prior").fit(train[list(config.features)], target)
    return config.build().fit(train[list(config.features)], target)


def predict(config: Config, fitted, frame: pd.DataFrame) -> np.ndarray:
    if config.kind == "official":
        return np.zeros(len(frame))
    if config.task == "regression":
        return np.asarray(fitted.predict(frame[list(config.features)]), dtype=float)
    proba = fitted.predict_proba(frame[list(config.features)])
    return np.full(len(frame), float(fitted.classes_[0])) if proba.shape[1] == 1 else proba[:, 1]
