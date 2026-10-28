"""
ML model interface - what the models predict, which candidates are compared,
and how they are scored, saved and loaded.

TWO QUESTIONS, TWO MODELS (the same leakage-free features feed both)

  1. Regression - "how wrong will the official forecast be?"
       target:  error_mw = forecast_mw - actual_mw
       output:  corrected forecast = forecast_mw - predicted error
       scored:  MAE / RMSE / bias of the corrected forecast against actual
                generation, side by side with the official forecast itself.
       Predicting the *error* (not the generation) keeps the official
       forecast as the anchor - the model only learns the correction.

  2. Classification - "how likely is a shortfall of at least 1 GW?"
       target:  shortfall_1gw = 1 if error_mw >= 1,000 MW else 0
       output:  a probability per hour
       scored:  Brier score (probability quality), ROC-AUC, PR-AUC,
                precision, recall, F1 and accuracy at a 0.5 threshold.

CANDIDATES - simplest first; a complex model has to beat the baselines

  regression       official_forecast   no correction (the published baseline)
                   mean_bias           subtract the average training error
                   linear_by_level     error as a straight line of forecast size
                   random_forest       RandomForestRegressor on all features
                   gradient_boosting   HistGradientBoostingRegressor on all features

  classification   base_rate           always the training shortfall frequency
                   logistic_by_level   logistic regression on forecast size
                   random_forest       RandomForestClassifier on all features
                   gradient_boosting   HistGradientBoostingClassifier on all features

Hyper-parameters are deliberately conservative (large leaves, shallow
boosting): ~90 training days is a small sample and over-fitting is the main risk.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PROBABILITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    build: Callable[[], object]  # returns an unfitted scikit-learn estimator


def _forecast_level_only() -> ColumnTransformer:
    """Use just the forecast size - the simplest explanation of the error."""
    return ColumnTransformer([("level", "passthrough", ["forecast_mw"])])


REGRESSION_CANDIDATES = [
    Candidate("official_forecast", "Published forecast, no correction (baseline)",
              lambda: DummyRegressor(strategy="constant", constant=0.0)),
    Candidate("mean_bias", "Subtract the average training error",
              lambda: DummyRegressor(strategy="mean")),
    Candidate("linear_by_level", "Linear correction by forecast size",
              lambda: make_pipeline(_forecast_level_only(), LinearRegression())),
    Candidate("random_forest", "Random Forest regressor on all features",
              lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                            random_state=RANDOM_STATE, n_jobs=-1)),
    Candidate("gradient_boosting", "Histogram gradient boosting on all features",
              lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                                    min_samples_leaf=40, l2_regularization=1.0,
                                                    random_state=RANDOM_STATE)),
]  # fmt: skip

CLASSIFICATION_CANDIDATES = [
    Candidate("base_rate", "Always the training shortfall frequency (baseline)",
              lambda: DummyClassifier(strategy="prior")),
    Candidate("logistic_by_level", "Logistic regression on forecast size",
              lambda: make_pipeline(_forecast_level_only(), StandardScaler(), LogisticRegression())),
    Candidate("random_forest", "Random Forest classifier on all features",
              lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                             random_state=RANDOM_STATE, n_jobs=-1)),
    Candidate("gradient_boosting", "Histogram gradient boosting on all features",
              lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                                     min_samples_leaf=40, l2_regularization=1.0,
                                                     random_state=RANDOM_STATE)),
]  # fmt: skip


# ---- Scoring ------------------------------------------------------------------


def regression_scores(actual: pd.Series, predicted_generation: pd.Series) -> dict:
    """MAE / RMSE / bias (prediction - actual) of a generation forecast, in MW."""
    errors = np.asarray(predicted_generation, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae": round(float(np.mean(np.abs(errors))), 1),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 1),
        "bias": round(float(np.mean(errors)), 1),
    }


def classification_scores(y_true: pd.Series, probability: np.ndarray, threshold: float = PROBABILITY_THRESHOLD) -> dict:
    """Probability quality (Brier, AUCs) and threshold metrics for the shortfall event."""
    y_true = np.asarray(y_true, dtype=int)
    predicted = (probability >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "brier": round(float(brier_score_loss(y_true, probability)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4) if both_classes else None,
        "pr_auc": round(float(average_precision_score(y_true, probability)), 4) if both_classes else None,
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "observed_rate": round(float(y_true.mean()), 4),
        "predicted_rate": round(float(predicted.mean()), 4),
    }


def day_block_bootstrap(
    days: pd.Series, baseline_abs_error: pd.Series, model_abs_error: pd.Series, n_resamples: int = 2000
) -> dict:
    """
    95% interval for the MAE reduction (baseline - model), resampling whole days.

    Hours within a day share the same weather and forecast run, so they are not
    independent; resampling days (not hours) gives an honest interval.
    """
    frame = pd.DataFrame({"day": days.to_numpy(), "diff": (baseline_abs_error - model_abs_error).to_numpy()})
    per_day = frame.groupby("day")["diff"].agg(["sum", "count"])
    rng = np.random.default_rng(RANDOM_STATE)
    picks = rng.integers(0, len(per_day), size=(n_resamples, len(per_day)))
    sums, counts = per_day["sum"].to_numpy(), per_day["count"].to_numpy()
    reductions = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mae_reduction_mw": round(float(frame["diff"].mean()), 1),
        "ci_low_mw": round(float(np.percentile(reductions, 2.5)), 1),
        "ci_high_mw": round(float(np.percentile(reductions, 97.5)), 1),
        "days": len(per_day),
        "method": f"day-block bootstrap, {n_resamples} resamples",
    }


def calibration_table(y_true: pd.Series, probability: np.ndarray, bins: int = 5) -> list[dict]:
    """Does a predicted 70% happen ~70% of the time? Grouped by probability band."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": probability})
    edges = np.linspace(0, 1, bins + 1)
    labels = [f"{low:.1f}-{high:.1f}" for low, high in zip(edges[:-1], edges[1:], strict=True)]
    frame["band"] = pd.cut(frame["p"], edges, labels=labels, include_lowest=True)
    rows = []
    for band, group in frame.groupby("band", observed=True):
        rows.append({
            "band": str(band),
            "n": len(group),
            "mean_predicted": round(float(group["p"].mean()), 3),
            "observed_rate": round(float(group["y"].mean()), 3),
        })  # fmt: skip
    return rows


# ---- Persistence ----------------------------------------------------------------


@dataclass
class ModelBundle:
    """A fitted model plus everything needed to use it correctly later."""

    task: str  # "regression" | "classification"
    name: str
    description: str
    estimator: object
    features: list[str]
    trained_from: date
    trained_to: date
    sklearn_version: str = sklearn.__version__

    @property
    def inputs(self) -> list[str]:
        """The columns the estimator really uses (a column selector may narrow ``features``)."""
        first_step = self.estimator.steps[0][1] if isinstance(self.estimator, Pipeline) else None
        if isinstance(first_step, ColumnTransformer):
            return [col for _, transformer, cols in first_step.transformers_ if transformer != "drop" for col in cols]
        return list(self.features)

    def predict_error(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(features[self.features])

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(features[self.features])[:, 1]


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> ModelBundle:
    try:
        bundle = joblib.load(path)
    except OSError:
        raise  # missing or unreadable file - already a clear message
    except Exception as exc:  # empty, truncated or not a joblib file: EOFError, UnpicklingError, ...
        raise ValueError(f"{path} is not a readable model file ({type(exc).__name__}: {exc})") from exc
    if not isinstance(bundle, ModelBundle):
        raise ValueError(f"{path} does not contain a ModelBundle")
    return bundle
