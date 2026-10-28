"""
Candidate models, built from the locked protocol (protocol.json lists every family and grid).

Production specifications (linear_by_level, logistic_by_level, both random forests and the
references) are imported from app.models.ml_model, never re-typed, so the baseline here is the
model the application serves.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.models.ml_model import CLASSIFICATION_CANDIDATES, RANDOM_STATE, REGRESSION_CANDIDATES
from ml.v2.ridge_gbm.config import ALL, LEVEL

TARGET = {"regression": "error_mw", "classification": "shortfall"}
BASELINE = {"regression": "linear_by_level", "classification": "logistic_by_level"}
REFERENCE = {"regression": "official_forecast", "classification": "base_rate"}
PRODUCTION = {("regression", c.name): c.build for c in REGRESSION_CANDIDATES} | {
    ("classification", c.name): c.build for c in CLASSIFICATION_CANDIDATES
}
GBM_FIXED = {"max_iter": 200, "l2_regularization": 1.0, "early_stopping": False, "random_state": RANDOM_STATE}


def hour_circle(hours) -> np.ndarray:
    """UK hour on a 24 h circle, so 23:00 and 00:00 are neighbours for a linear model."""
    radians = 2 * np.pi * np.asarray(hours, dtype=float) / 24
    return np.column_stack([np.sin(radians), np.cos(radians)])


def linear_inputs() -> ColumnTransformer:
    """All 9 features for a linear model: training-median imputation, standardised, hour on a circle."""
    numeric = [column for column in ALL if column != "hour_uk"]
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("hour", FunctionTransformer(hour_circle), ["hour_uk"]),
    ])  # fmt: skip


BUILDERS = {
    ("regression", "ridge_level"): lambda alpha: make_pipeline(ColumnTransformer([("level", StandardScaler(), LEVEL)]), Ridge(alpha=alpha)),
    ("regression", "ridge_all"): lambda alpha: make_pipeline(linear_inputs(), Ridge(alpha=alpha)),
    ("regression", "gbm_all"): lambda **params: HistGradientBoostingRegressor(**params, **GBM_FIXED),
    ("classification", "logistic_all"): lambda C: make_pipeline(linear_inputs(), LogisticRegression(C=C, max_iter=2000)),
    ("classification", "gbm_all"): lambda **params: HistGradientBoostingClassifier(**params, **GBM_FIXED),
}  # fmt: skip


@dataclass(frozen=True)
class Config:
    task: str
    family: str
    params: tuple = ()  # (name, value) pairs in protocol order

    @property
    def name(self) -> str:
        return self.family + (f"[{', '.join(f'{k}={v}' for k, v in self.params)}]" if self.params else "")

    def build(self):
        if (self.task, self.family) in PRODUCTION:
            return PRODUCTION[(self.task, self.family)]()
        return BUILDERS[(self.task, self.family)](**dict(self.params))

    def predict(self, model, frame) -> np.ndarray:
        if self.task == "regression":
            return np.asarray(model.predict(frame[ALL]), dtype=float)
        return np.asarray(model.predict_proba(frame[ALL])[:, 1], dtype=float)


def configs(protocol: dict, task: str) -> list[Config]:
    """Every configuration of every family, in protocol order (simplest first within a grid)."""
    out = []
    for family, spec in protocol["candidates"][task].items():
        grid = {key: values for key, values in spec.items() if key != "role"}
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)) if keys else [()]:
            out.append(Config(task, family, tuple(zip(keys, values, strict=True))))
    return out
