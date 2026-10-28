"""
Regression tests on the committed January 2026 snapshot.

They pin down facts about the real data that the product relies on, so a
code change (or a refreshed snapshot) that breaks them is noticed. Skipped
if the snapshot files are not present.
"""

from datetime import time

import pandas as pd
import pytest

from app.config import Settings
from app.services.data_store import ACTUALS_FILE, FORECASTS_FILE, load_store
from app.services.forecast_evaluator import SelectionPolicy, compute_metrics, evaluate

settings = Settings(data_source="snapshot")
pytestmark = pytest.mark.skipif(
    not ((settings.data_dir / ACTUALS_FILE).is_file() and (settings.data_dir / FORECASTS_FILE).is_file()),
    reason="no data snapshot in backend/data",
)


@pytest.fixture(scope="module")
def store():
    return load_store(settings)


@pytest.fixture(scope="module")
def frame(store):
    return evaluate(
        store.forecasts, store.actuals, store.first_delivery_date, store.last_delivery_date, SelectionPolicy()
    )


def test_every_delivery_hour_of_january_is_scored(frame):
    metrics = compute_metrics(frame)
    assert metrics["intervals_total"] == 31 * 24
    assert metrics["coverage_pct"] == 100.0


def test_no_forecast_published_after_the_cutoff_is_used(frame):
    assert (frame["publishTime"] <= frame["cutoff"]).all()


def test_decision_uses_the_0930_run_on_the_previous_day(frame):
    expected = frame["cutoff"].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    assert (frame["publishTime"] == expected).all()


def test_earlier_gate_closure_uses_an_earlier_run(store):
    early = evaluate(
        store.forecasts, store.actuals, store.first_delivery_date, store.last_delivery_date,
        SelectionPolicy(gate_closure=time(9, 20)),
    )  # fmt: skip
    assert (early["publishTime"].dt.strftime("%H:%M") == "07:30").all()
