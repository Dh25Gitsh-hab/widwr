"""
Paths, data files, periods and folds for the existing-data experiment (offline).

The only inputs are the Elexon files already in the project (DATASETS/, DATASETS1/).
Nothing in ml/v2/data/raw (downloaded before the scope correction) is read.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ml.v2.common import PROJECT_DIR

EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "processed"
REPORTS_DIR = EXP_DIR / "reports"
MODELS_DIR = EXP_DIR / "models"
PROTOCOL_FILE = EXP_DIR / "protocol.json"
PROTOCOL_LOCK = REPORTS_DIR / "protocol_lock.json"

MONTHS = ("Oct25", "Nov25", "Dec25", "Jan26", "Feb26")
ACTUAL_FILES = [PROJECT_DIR / "DATASETS" / f"{m}_Actual.json" for m in MONTHS]
FORECAST_FILES = [PROJECT_DIR / "DATASETS1" / f"{m}_Forecast.json" for m in MONTHS]

# Delivery dates (decision = delivery - 1 day)
DEV = (date(2025, 10, 2), date(2025, 12, 31))  # development: decisions 1 Oct - 30 Dec 2025 (before January)
JANUARY = (date(2026, 1, 2), date(2026, 2, 1))  # the application period: decisions 1-31 Jan 2026 (seen before)
FEBRUARY = (date(2026, 2, 2), date(2026, 3, 1))  # previously examined: decisions 1-28 Feb 2026

# Deliveries up to d-2 have every label published before d's cutoff (09:50 on d-1).
GAP = timedelta(days=2)
# Eight weekly validation blocks, 6 Nov - 31 Dec 2025; each trains on everything up to 2 days before it.
FOLDS = [(date(2025, 11, 6) + timedelta(days=7 * i), date(2025, 11, 12) + timedelta(days=7 * i)) for i in range(8)]
EARLY_FOLDS, LATE_FOLDS = FOLDS[:4], FOLDS[4:]  # nested check: choose on the early four, confirm on the late four


def fold_masks(dates: pd.Series, fold: tuple[date, date]) -> tuple[pd.Series, pd.Series]:
    start, end = fold
    return dates <= start - GAP, (dates >= start) & (dates <= end)


def rolling_window(dates: pd.Series, day: date, days: int) -> pd.Series:
    """The last `days` delivery days whose labels are all known at day's cutoff."""
    return (dates >= day - timedelta(days=days + 1)) & (dates <= day - GAP)


def fold_label(fold: tuple[date, date]) -> str:
    return f"{fold[0]:%m-%d}..{fold[1]:%m-%d}"
