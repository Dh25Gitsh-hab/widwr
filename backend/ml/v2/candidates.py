"""
The pre-registered v2 candidates (ml/v2/protocol.json), exactly as written there.

Regression candidates predict error_mw (corrected forecast = forecast_mw - prediction);
classification candidates predict P(error_mw >= 1000 MW). Builders return unfitted
estimators; all preprocessing sits inside the estimator, so it is fitted on training rows only.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

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
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from app.services.feature_builder import FEATURE_COLUMNS
from ml.v2.common import SHORTFALL_THRESHOLD_MW
from ml.v2.features import ALL_FEATURES

LEVEL = ["forecast_mw"]
EXISTING9 = list(FEATURE_COLUMNS)
CURVE = ["forecast_mw", "hour_sin", "hour_cos"]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEED = 42


def _spline() -> SplineTransformer:
    return SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear")


def _curve_inputs() -> ColumnTransformer:
    return ColumnTransformer([("level", _spline(), ["forecast_mw"]), ("hour", "passthrough", ["hour_sin", "hour_cos"])])


def _all_inputs() -> ColumnTransformer:
    rest = [f for f in ALL_FEATURES if f != "forecast_mw"]
    return ColumnTransformer([
        ("level", _spline(), ["forecast_mw"]),
        ("rest", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), rest),
    ])  # fmt: skip


def _level_only() -> ColumnTransformer:
    return ColumnTransformer([("level", "passthrough", ["forecast_mw"])])


def _median() -> QuantileRegressor:
    return QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")


def _gbm(**kwargs) -> dict:
    return dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                random_state=SEED, **kwargs)  # fmt: skip


@dataclass(frozen=True)
class Candidate:
    name: str
    task: str  # "regression" | "classification"
    features: list[str]
    build: Callable[[], object] | None = None
    kind: str = "model"  # "model" | "official" | "ensemble" | "quantile_distribution"
    rolling_days: int | None = None  # refit for every decision day on the last N days known at its cutoff
    members: tuple[str, ...] = ()


REGRESSION = [
    Candidate("R0_official", "regression", LEVEL, kind="official"),
    Candidate("R1_linear_by_level", "regression", LEVEL, lambda: make_pipeline(_level_only(), LinearRegression())),
    Candidate("R2_random_forest", "regression", EXISTING9,
              lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, random_state=SEED, n_jobs=-1)),
    Candidate("R3_median_by_level", "regression", LEVEL, lambda: make_pipeline(_level_only(), _median())),
    Candidate("R4_median_curve", "regression", CURVE, lambda: make_pipeline(_curve_inputs(), _median())),
    Candidate("R5_median_all", "regression", ALL_FEATURES, lambda: make_pipeline(_all_inputs(), _median())),
    Candidate("R6_gbm_absolute", "regression", ALL_FEATURES,
              lambda: HistGradientBoostingRegressor(loss="absolute_error", monotonic_cst={"forecast_mw": 1}, **_gbm())),
    *[Candidate(f"R7_median_curve_rolling_w{w}", "regression", CURVE, lambda: make_pipeline(_curve_inputs(), _median()),
                rolling_days=w) for w in (30, 60, 90)],
    Candidate("R8_ensemble_r5_r6", "regression", ALL_FEATURES, kind="ensemble", members=("R5_median_all", "R6_gbm_absolute")),
]  # fmt: skip

CLASSIFICATION = [
    Candidate("C0_base_rate", "classification", LEVEL, lambda: DummyClassifier(strategy="prior")),
    Candidate("C1_logistic_by_level", "classification", LEVEL,
              lambda: make_pipeline(_level_only(), StandardScaler(), LogisticRegression())),
    Candidate("C2_random_forest", "classification", EXISTING9,
              lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.5, random_state=SEED, n_jobs=-1)),
    Candidate("C3_logistic_all", "classification", ALL_FEATURES,
              lambda: make_pipeline(_all_inputs(), LogisticRegression(C=1.0, max_iter=5000))),
    Candidate("C4_gbm_classifier", "classification", ALL_FEATURES,
              lambda: HistGradientBoostingClassifier(monotonic_cst={"forecast_mw": 1}, **_gbm())),
    Candidate("C5_quantile_distribution", "classification", ALL_FEATURES, kind="quantile_distribution"),
    *[Candidate(f"C6_logistic_curve_rolling_w{w}", "classification", CURVE,
                lambda: make_pipeline(_curve_inputs(), LogisticRegression(max_iter=5000)), rolling_days=w) for w in (30, 60, 90)],
]  # fmt: skip

CANDIDATES = {c.name: c for c in REGRESSION + CLASSIFICATION}


def fit(candidate: Candidate, train: pd.DataFrame):
    """Fit one candidate on training rows only. Returns whatever predict() needs."""
    if candidate.kind == "official":
        return None
    if candidate.kind == "ensemble":
        return {name: fit(CANDIDATES[name], train) for name in candidate.members}
    if candidate.kind == "quantile_distribution":
        return [HistGradientBoostingRegressor(loss="quantile", quantile=q, **_gbm()).fit(train[candidate.features], train["error_mw"])
                for q in QUANTILES]  # fmt: skip
    target = train["error_mw"] if candidate.task == "regression" else train["shortfall_1gw"].astype(int)
    if candidate.task == "classification" and target.nunique() < 2:  # a short rolling window can hold one class
        return DummyClassifier(strategy="prior").fit(train[candidate.features], target)
    return candidate.build().fit(train[candidate.features], target)


def predict(candidate: Candidate, fitted, frame: pd.DataFrame) -> np.ndarray:
    """Regression: predicted error (MW). Classification: probability of a >= 1 GW shortfall."""
    if candidate.kind == "official":
        return np.zeros(len(frame))
    if candidate.kind == "ensemble":
        return np.mean([predict(CANDIDATES[name], fitted[name], frame) for name in candidate.members], axis=0)
    if candidate.kind == "quantile_distribution":
        quantiles = np.sort(np.column_stack([m.predict(frame[candidate.features]) for m in fitted]), axis=1)
        return np.array([1.0 - np.interp(SHORTFALL_THRESHOLD_MW, row, QUANTILES) for row in quantiles])
    if candidate.task == "regression":
        return fitted.predict(frame[candidate.features])
    proba = fitted.predict_proba(frame[candidate.features])
    if proba.shape[1] == 1:  # only one class was seen in training
        return np.full(len(frame), float(fitted.classes_[0]))
    return proba[:, 1]


@dataclass
class V2Model:
    """A frozen v2 model (offline artifact; never loaded by the January application)."""

    name: str
    task: str
    fitted: object  # None for rolling candidates: they refit on data known at each cutoff
    trained_from: date
    trained_to: date
    protocol_sha256: str
    extras: dict = field(default_factory=dict)  # Platt calibration and thresholds for the classifier

    @property
    def candidate(self) -> Candidate:
        return CANDIDATES[self.name]
