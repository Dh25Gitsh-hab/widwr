"""Shared paths, window and helpers for the February 2026 holdout."""

import hashlib
import json
from datetime import date, time
from pathlib import Path

import pandas as pd

from ml.build_dataset import ML_DIR, PROJECT_DIR

GATE_CLOSURE = time(9, 50)  # UK local time on D-1, same as training

# Holdout = purchase decisions made in February 2026. 1 Feb delivery is NOT included:
# it was decided on 31 Jan and is already part of the January test set.
FIRST_DECISION = date(2026, 2, 1)
LAST_DECISION = date(2026, 2, 28)

# Inputs. January files are needed only for history features of the first
# February days (forecasts and actuals published BEFORE those cutoffs).
ACTUAL_FILES = [PROJECT_DIR / "DATASETS" / "Jan26_Actual.json", PROJECT_DIR / "DATASETS" / "Feb26_Actual.json"]
FORECAST_FILES = [PROJECT_DIR / "DATASETS1" / "Jan26_Forecast.json", PROJECT_DIR / "DATASETS1" / "Feb26_Forecast.json"]
LABEL_FILE = PROJECT_DIR / "DATASETS" / "Feb26_Actual.json"

OUT_DIR = ML_DIR / "holdout_feb2026"
MODELS_DIR = OUT_DIR / "models"
REPORTS_DIR = OUT_DIR / "reports"
FEATURES_FILE = OUT_DIR / "features.csv"
PREDICTIONS_FILE = OUT_DIR / "predictions_sealed.csv"
MANIFEST_FILE = OUT_DIR / "sealed_manifest.json"

# Existing results that must not change (verified before and after).
FROZEN_FILES = [
    ML_DIR / "models" / "error_regressor.joblib",
    ML_DIR / "models" / "shortfall_classifier.joblib",
    ML_DIR / "processed" / "ml_dataset.csv",
    ML_DIR / "reports" / "metrics.json",
    ML_DIR / "predictions" / "test_predictions.csv",
]

REGRESSION_MODELS = ["official_forecast", "random_forest", "linear_by_level"]
CLASSIFICATION_MODELS = ["random_forest", "logistic_by_level", "base_rate"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_DIR).as_posix()


def load_records(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))) for p in paths], ignore_index=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
