"""
Shared test fixtures: a small synthetic market where every expected number
can be worked out by hand.

Scenario (all times UTC; January, so UK time = UTC)
    Delivery days 3 and 4 Jan 2026  (decisions on 2 and 3 Jan).
    Actual wind is a flat 10,000 MW, except one missing half-hour
    (4 Jan 12:00) to exercise missing-data handling.
    Three forecast runs, each covering 2 Jan 21:00 .. 4 Jan 21:00:
        published 2 Jan 07:30 -> 11,000 MW
        published 2 Jan 09:30 -> 12,000 MW
        published 2 Jan 11:30 -> 13,000 MW
    So, with the default 09:50 gate closure:
        3 Jan: decided 2 Jan 09:50 -> the 09:30 run (12,000) -> error +2,000
        4 Jan: decided 3 Jan 09:50 -> the 11:30 run (13,000) -> error +3,000,
               and 22:00/23:00 have no forecast at all (runs end at 21:00).
"""

from datetime import date

import pandas as pd
import pytest

from app.config import Settings
from app.services.data_processor import clean_actuals, clean_forecasts
from app.services.data_store import DataStore

ISO = "%Y-%m-%dT%H:%M:%SZ"
FIRST_DELIVERY = date(2026, 1, 3)
LAST_DELIVERY = date(2026, 1, 4)
RUNS = {"2026-01-02T07:30:00Z": 11_000.0, "2026-01-02T09:30:00Z": 12_000.0, "2026-01-02T11:30:00Z": 13_000.0}
MISSING_ACTUAL = "2026-01-04T12:00:00Z"


def make_raw_actuals(value: float = 10_000.0, first: str = "2026-01-02", end: str = "2026-01-05") -> pd.DataFrame:
    times = pd.date_range(first, end, freq="30min", inclusive="left", tz="UTC")
    rows = [
        {
            "startTime": t.strftime(ISO),
            "publishTime": (t + pd.Timedelta(minutes=30)).strftime(ISO),
            "fuelType": "WIND",
            "generation": value,
        }
        for t in times
        if t.strftime(ISO) != MISSING_ACTUAL
    ]
    return pd.DataFrame(rows)


def make_run(publish_time: str, value: float, first="2026-01-01T21:00:00Z", last="2026-01-04T21:00:00Z") -> list[dict]:
    """One WINDFOR run - like the real ones, it starts before its own publish time."""
    targets = pd.date_range(first, last, freq="1h")
    return [{"publishTime": publish_time, "startTime": t.strftime(ISO), "generation": value} for t in targets]


def make_raw_forecasts() -> pd.DataFrame:
    return pd.DataFrame([row for publish, value in RUNS.items() for row in make_run(publish, value)])


@pytest.fixture
def actuals() -> pd.DataFrame:
    return clean_actuals(make_raw_actuals())[0]


@pytest.fixture
def forecasts() -> pd.DataFrame:
    return clean_forecasts(make_raw_forecasts())[0]


@pytest.fixture
def store(actuals: pd.DataFrame, forecasts: pd.DataFrame) -> DataStore:
    return DataStore(
        actuals=actuals,
        forecasts=forecasts,
        first_delivery_date=FIRST_DELIVERY,
        last_delivery_date=LAST_DELIVERY,
        source="synthetic",
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        first_decision_date=date(2026, 1, 2),
        last_decision_date=date(2026, 1, 3),
        data_dir=tmp_path,
        ml_dir=tmp_path / "ml",  # no trained models here unless a test provides them
    )
