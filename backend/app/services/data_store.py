"""
Data store - where the cleaned datasets live while the API runs.

WHY
    The API must answer in milliseconds and keep working when Elexon is slow
    or unavailable (for example during a live demo). The data in scope is
    small (~20k forecast rows, ~1.6k actuals), so we:
      1. download it from Elexon once;
      2. save the raw records as CSV snapshots in ``backend/data/``. They are
         committed to the repository, so the project runs offline on any
         machine and the numbers are reproducible even if Elexon later
         revises its data;
      3. clean it once at start-up and keep it in memory as DataFrames.
    A database would add set-up work for every reviewer with no benefit at
    this size. Swapping to one later only means re-implementing this module.

HOW
    ``load_store(settings)`` is called once by the FastAPI start-up hook.
    ``refresh_snapshot(settings)`` (used by ``scripts/fetch_data.py``)
    re-downloads the data.

WHICH DATA IS FETCHED (derived from the business scope in config.py)
    actuals:   settlement dates (first delivery day - 1) .. last delivery day
    forecasts: runs published from 3 days before the first delivery day
               (WINDFOR looks up to ~66 h ahead, so earlier runs cannot
               cover the window) until the end of the last delivery day
               (so later runs can show how the forecast evolved).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.services.data_processor import CleaningReport, clean_actuals, clean_forecasts
from app.services.elexon_client import ElexonClient

log = logging.getLogger(__name__)

ACTUALS_FILE = "fuelhh_wind_raw.csv"
FORECASTS_FILE = "windfor_raw.csv"
METADATA_FILE = "snapshot_meta.json"


@dataclass(frozen=True)
class FetchWindow:
    actuals_first_day: date
    actuals_last_day: date
    forecasts_published_from: datetime
    forecasts_published_to: datetime

    @classmethod
    def from_settings(cls, settings: Settings) -> "FetchWindow":
        return cls(
            actuals_first_day=settings.first_delivery_date - timedelta(days=1),
            actuals_last_day=settings.last_delivery_date,
            forecasts_published_from=datetime.combine(
                settings.first_delivery_date - timedelta(days=3), time(0), tzinfo=timezone.utc
            ),
            forecasts_published_to=datetime.combine(
                settings.last_delivery_date + timedelta(days=1), time(0), tzinfo=timezone.utc
            ),
        )


@dataclass
class DataStore:
    """Clean, in-memory datasets plus the provenance needed to trust them."""

    actuals: pd.DataFrame  # startTime, publishTime, generation (half-hourly)
    forecasts: pd.DataFrame  # publishTime, startTime, generation, horizon_hours (hourly)
    first_delivery_date: date
    last_delivery_date: date
    source: str  # "snapshot" or "elexon-live"
    cleaning: list[CleaningReport] = field(default_factory=list)
    snapshot_fetched_at: datetime | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def forecast_runs(self) -> int:
        return int(self.forecasts["publishTime"].nunique())

    def contains_delivery_range(self, first: date, last: date) -> bool:
        return self.first_delivery_date <= first <= last <= self.last_delivery_date


def load_store(settings: Settings) -> DataStore:
    """Load (or download) the raw data, clean it and return the in-memory store."""
    data_dir = Path(settings.data_dir)
    if settings.data_source == "snapshot" and _snapshot_exists(data_dir):
        log.info("Loading Elexon snapshot from %s", data_dir)
        raw_actuals = pd.read_csv(data_dir / ACTUALS_FILE)
        raw_forecasts = pd.read_csv(data_dir / FORECASTS_FILE)
        source = "snapshot"
    else:
        log.info("No snapshot (or live mode): downloading from Elexon")
        raw_actuals, raw_forecasts = refresh_snapshot(settings)
        source = "elexon-live"

    actuals, actuals_report = clean_actuals(raw_actuals)
    forecasts, forecasts_report = clean_forecasts(raw_forecasts)
    log.info("Clean data: %d actual half-hours, %d forecast values", len(actuals), len(forecasts))

    return DataStore(
        actuals=actuals,
        forecasts=forecasts,
        first_delivery_date=settings.first_delivery_date,
        last_delivery_date=settings.last_delivery_date,
        source=source,
        cleaning=[actuals_report, forecasts_report],
        snapshot_fetched_at=_read_fetched_at(data_dir),
    )


def refresh_snapshot(settings: Settings, client: ElexonClient | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the in-scope data from Elexon and overwrite the CSV snapshot."""
    window = FetchWindow.from_settings(settings)
    owns_client = client is None
    client = client or ElexonClient(
        base_url=settings.elexon_base_url,
        timeout_seconds=settings.elexon_timeout_seconds,
        max_retries=settings.elexon_max_retries,
        chunk_days=settings.elexon_chunk_days,
    )
    try:
        raw_actuals = pd.DataFrame(client.fetch_wind_actuals(window.actuals_first_day, window.actuals_last_day))
        raw_forecasts = pd.DataFrame(
            client.fetch_wind_forecasts(window.forecasts_published_from, window.forecasts_published_to)
        )
    finally:
        if owns_client:
            client.close()

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_actuals.to_csv(data_dir / ACTUALS_FILE, index=False)
    raw_forecasts.to_csv(data_dir / FORECASTS_FILE, index=False)
    metadata = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": settings.elexon_base_url,
        "actuals_settlement_dates": [window.actuals_first_day.isoformat(), window.actuals_last_day.isoformat()],
        "forecasts_published": [
            window.forecasts_published_from.isoformat(),
            window.forecasts_published_to.isoformat(),
        ],
        "rows": {"actuals": len(raw_actuals), "forecasts": len(raw_forecasts)},
    }
    (data_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Snapshot saved to %s", data_dir)
    return raw_actuals, raw_forecasts


def _snapshot_exists(data_dir: Path) -> bool:
    return (data_dir / ACTUALS_FILE).is_file() and (data_dir / FORECASTS_FILE).is_file()


def _read_fetched_at(data_dir: Path) -> datetime | None:
    try:
        metadata = json.loads((data_dir / METADATA_FILE).read_text(encoding="utf-8"))
        return datetime.fromisoformat(metadata["fetched_at"])
    except (OSError, KeyError, ValueError):
        return None
