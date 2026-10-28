"""
Paths, data, features, folds and segments for the Ridge / gradient-boosting experiment (offline).

Input: the production feature table ml/processed/ml_dataset.csv, read only. It was built by
ml/build_dataset.py from the Elexon files already in the project (DATASETS/, DATASETS1/); every
feature uses only data published by 09:50 UK time on D-1 (app/services/feature_builder.py and its
tests). Nothing here is loaded by the application.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from app.services.feature_builder import FEATURE_COLUMNS
from ml.v2.common import LEVEL_BANDS_MW, ML_DIR
from ml.v2.existing_data.config import FOLDS, GAP, fold_label, fold_masks  # same folds as the existing-data run

EXP_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXP_DIR / "reports"
MODELS_DIR = EXP_DIR / "models"
PROTOCOL_FILE = EXP_DIR / "protocol.json"
PROTOCOL_LOCK = REPORTS_DIR / "protocol_lock.json"
SELECTION_FILE = REPORTS_DIR / "selection.json"
DATASET_FILE = ML_DIR / "processed" / "ml_dataset.csv"

PRODUCTION_LINEAR = ML_DIR / "holdout_feb2026" / "models" / "regression_linear_by_level.joblib"
PRODUCTION_LOGISTIC = ML_DIR / "holdout_feb2026" / "models" / "classification_logistic_by_level.joblib"
PRODUCTION_RF_REGRESSOR = ML_DIR / "models" / "error_regressor.joblib"
PRODUCTION_RF_CLASSIFIER = ML_DIR / "models" / "shortfall_classifier.joblib"

DEV_SPLITS = ("train", "validation")  # deliveries 2025-10-02 .. 2025-12-31 (decisions before January)
TEST_SPLIT = "test"  # January 2026 decisions: deliveries 2026-01-02 .. 2026-02-01
DEV_LAST_DELIVERY = date(2025, 12, 31)

LEVEL = ["forecast_mw"]
ALL = list(FEATURE_COLUMNS)  # the 9 production features
HIGH_FORECAST_MW = 16_000.0  # fixed before any candidate was fitted (the 3 Jan investigation threshold)
BANDS_MW = LEVEL_BANDS_MW  # 0-5, 5-10, 10-15, 15-20, 20+ GW, fixed in ml/v2/common.py
BAND_LABELS = ["0-5", "5-10", "10-15", "15-20", "20+"]

__all__ = ["FOLDS", "GAP", "fold_label", "fold_masks"]


def _read(splits: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(DATASET_FILE)
    frame = frame.loc[frame["split"].isin(splits)].copy()
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True)
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"]).dt.date
    frame["shortfall"] = frame["shortfall_1gw"].astype(int)
    return frame.sort_values("startTime").reset_index(drop=True)


def load_development() -> pd.DataFrame:
    """Oct-Dec 2025 deliveries: the only rows used to fit, tune and choose."""
    frame = _read(DEV_SPLITS)
    if frame["delivery_date"].max() > DEV_LAST_DELIVERY:
        raise ValueError("development data reaches into January")
    return frame


def load_january() -> pd.DataFrame:
    """The designated test period. Read only by run.evaluate_january, after selection.json is written."""
    return _read((TEST_SPLIT,))


def bands(forecast_mw: pd.Series) -> pd.Series:
    return pd.cut(forecast_mw, BANDS_MW, right=False, labels=BAND_LABELS).astype(str)
