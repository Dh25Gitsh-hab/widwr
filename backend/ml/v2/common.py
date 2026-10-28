"""
Shared paths, data windows, fixed constants and the protocol lock for the v2 experiments.

Everything in ml/v2 is OFFLINE: nothing here is loaded by the January application,
and no production artifact is read for writing.
"""

import hashlib
import json
import math
from datetime import date, datetime, time, timezone
from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
ML_DIR = V2_DIR.parent
BACKEND_DIR = ML_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

RAW_DIR = V2_DIR / "data" / "raw"
PROCESSED_DIR = V2_DIR / "processed"
REPORTS_DIR = V2_DIR / "reports"
MODELS_DIR = V2_DIR / "models"
PROTOCOL_FILE = V2_DIR / "protocol.json"
PROTOCOL_LOCK = REPORTS_DIR / "protocol_lock.json"

GATE_CLOSURE = time(9, 50)  # UK local time on D-1, as in production
SHORTFALL_THRESHOLD_MW = 1_000.0

# Development: every decision before the January application period.
DEV_FIRST_DECISION, DEV_LAST_DECISION = date(2024, 10, 1), date(2025, 12, 31)
# The January application period: context only - it has been evaluated before.
JAN_FIRST_DECISION, JAN_LAST_DECISION = date(2026, 1, 1), date(2026, 1, 31)
# Supplementary evaluation: its data may only be downloaded after the protocol is locked.
SUPP_FIRST_DECISION, SUPP_LAST_DECISION = date(2026, 3, 1), date(2026, 5, 31)

# History downloaded by fetch_history.py (the project's DATASETS folders start in Oct 2025).
HISTORY_FORECASTS_PUBLISHED = (datetime(2024, 9, 24, tzinfo=timezone.utc), datetime(2025, 10, 1, tzinfo=timezone.utc))
HISTORY_ACTUALS_SETTLEMENT = (date(2024, 9, 24), date(2025, 9, 30))
NDFD_PUBLISHED = (datetime(2024, 9, 24, tzinfo=timezone.utc), datetime(2026, 1, 31, tzinfo=timezone.utc))
HISTORY_FORECAST_FILE = RAW_DIR / "windfor_published_2024-09-24_2025-09-30.csv"
HISTORY_ACTUAL_FILE = RAW_DIR / "fuelhh_wind_settlement_2024-09-24_2025-09-30.csv"
NDFD_FILE = RAW_DIR / "ndfd_published_2024-09-24_2026-01-30.csv"

# The user's existing raw files (read only, never copied).
EXISTING_MONTHS = ("Oct25", "Nov25", "Dec25", "Jan26")
EXISTING_ACTUAL_FILES = [PROJECT_DIR / "DATASETS" / f"{m}_Actual.json" for m in EXISTING_MONTHS]
EXISTING_FORECAST_FILES = [PROJECT_DIR / "DATASETS1" / f"{m}_Forecast.json" for m in EXISTING_MONTHS]

# Forecast-size bands, fixed in advance (not fitted to any data).
LEVEL_BANDS_MW = [0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0, math.inf]

# England & Wales bank holidays, published by the government years ahead (known at any cutoff).
BANK_HOLIDAYS = {
    date(2024, 12, 25), date(2024, 12, 26), date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 5), date(2025, 5, 26), date(2025, 8, 25), date(2025, 12, 25), date(2025, 12, 26),
    date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 4), date(2026, 5, 25),
}  # fmt: skip


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def lock_protocol(force: bool = False, protocol_file: Path | None = None, lock_file: Path | None = None) -> dict:
    """Record the protocol's fingerprint. Refuses to re-lock unless forced (a re-lock is visible)."""
    protocol_file, lock_file = protocol_file or PROTOCOL_FILE, lock_file or PROTOCOL_LOCK
    if lock_file.is_file() and not force:
        raise SystemExit(f"Protocol already locked ({lock_file}); use --force to re-lock deliberately")
    lock = {"protocol_file": protocol_file.name, "sha256": sha256(protocol_file),
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}  # fmt: skip
    write_json(lock_file, lock)
    return lock


def verify_protocol(protocol_file: Path | None = None, lock_file: Path | None = None) -> dict:
    """Abort unless the protocol is byte-identical to what was locked."""
    protocol_file, lock_file = protocol_file or PROTOCOL_FILE, lock_file or PROTOCOL_LOCK
    if not lock_file.is_file():
        raise SystemExit(f"Protocol not locked - no {lock_file}")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    if sha256(protocol_file) != lock["sha256"]:
        raise SystemExit(f"{protocol_file.name} changed after it was locked - results would not be pre-registered")
    return lock


def load_protocol(protocol_file: Path | None = None) -> dict:
    return json.loads((protocol_file or PROTOCOL_FILE).read_text(encoding="utf-8"))
