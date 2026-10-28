from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from app.models.ml_model import (
    CLASSIFICATION_CANDIDATES,
    REGRESSION_CANDIDATES,
    ModelBundle,
    calibration_table,
    classification_scores,
    day_block_bootstrap,
    load_bundle,
    regression_scores,
    save_bundle,
)
from app.services.feature_builder import FEATURE_COLUMNS


def test_regression_scores():
    scores = regression_scores(pd.Series([1_000.0, 1_000.0, 1_000.0]), pd.Series([1_100.0, 900.0, 1_300.0]))
    assert scores == {"mae": 166.7, "rmse": 191.5, "bias": 100.0}


def test_classification_scores_threshold_metrics():
    scores = classification_scores(pd.Series([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1]))
    assert scores["accuracy"] == 0.5
    assert scores["precision"] == 0.5 and scores["recall"] == 0.5
    assert scores["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4, abs=1e-4)
    assert scores["roc_auc"] == 0.75


def test_day_block_bootstrap_interval_contains_estimate():
    days = pd.Series([d for d in range(10) for _ in range(24)])
    baseline = pd.Series(np.full(240, 2_000.0))
    model = pd.Series(np.linspace(1_000, 1_800, 240))
    result = day_block_bootstrap(days, baseline, model, n_resamples=500)
    assert result["ci_low_mw"] <= result["mae_reduction_mw"] <= result["ci_high_mw"]
    assert result["days"] == 10


def test_calibration_table_bands():
    rows = calibration_table(pd.Series([0, 1, 1, 1]), np.array([0.05, 0.15, 0.95, 0.99]))
    assert [(r["band"], r["n"]) for r in rows] == [("0.0-0.2", 2), ("0.8-1.0", 2)]


def test_official_forecast_baseline_predicts_zero_correction():
    official = next(c for c in REGRESSION_CANDIDATES if c.name == "official_forecast")
    X = pd.DataFrame(np.ones((5, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    model = official.build().fit(X, pd.Series([500.0] * 5))
    assert (model.predict(X) == 0).all()


@pytest.mark.parametrize("candidate", REGRESSION_CANDIDATES + CLASSIFICATION_CANDIDATES, ids=lambda c: c.name)
def test_every_candidate_handles_missing_feature_values(candidate):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_COLUMNS))) * 1_000, columns=FEATURE_COLUMNS)
    X.loc[:20, "recent_error_mw"] = np.nan  # like the first day of the data
    y = (X["forecast_mw"] > 0).astype(int) if candidate in CLASSIFICATION_CANDIDATES else X["forecast_mw"] * 0.1
    model = candidate.build().fit(X, y)
    assert len(model.predict(X)) == 200


def test_bundle_round_trip(tmp_path):
    X = pd.DataFrame(np.zeros((3, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    bundle = ModelBundle(
        task="regression", name="mean_bias", description="test",
        estimator=DummyRegressor().fit(X, [1.0, 2.0, 3.0]), features=FEATURE_COLUMNS,
        trained_from=date(2025, 10, 2), trained_to=date(2025, 12, 31),
    )  # fmt: skip
    save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_bundle(tmp_path / "model.joblib")
    assert loaded.name == "mean_bias"
    assert loaded.predict_error(X).tolist() == [2.0, 2.0, 2.0]
