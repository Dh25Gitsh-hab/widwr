"""
Assignment scope: "consider data relevant only for purchase decisions to be made in January 2026".

Every view the application serves must be for a purchase decision made 1-31 January 2026.
The decision is made on D-1 for delivery day D, so delivery days run 2 January - 1 February.
February 2026 results exist only as offline ML evaluation evidence and are never served.
"""

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.data_store import ACTUALS_FILE, FORECASTS_FILE
from app.services.prediction_service import CLASSIFIER_FILES, DATASET_FILE, METRICS_FILE, REGRESSOR_FILES

FIRST_DECISION, LAST_DECISION = date(2026, 1, 1), date(2026, 1, 31)
SETTINGS = Settings()
HAS_DATA = all((SETTINGS.data_dir / f).is_file() for f in (ACTUALS_FILE, FORECASTS_FILE))
HAS_MODELS = all(
    (SETTINGS.ml_dir / f).is_file() for f in [DATASET_FILE, METRICS_FILE, *REGRESSOR_FILES.values(), *CLASSIFIER_FILES.values()]
)
PREDICTION_REQUESTS = [
    ("predictions", {}),
    ("predictions", {"regressor": "random_forest", "classifier": "random_forest"}),
    ("corrected-forecast", {}),
    ("corrected-forecast", {"model": "random_forest"}),
    ("shortfall-risk", {}),
    ("shortfall-risk", {"model": "random_forest"}),
]


def uk_dates(timestamps: list[str]) -> pd.Series:
    return pd.to_datetime(pd.Series(timestamps), utc=True).dt.tz_convert("Europe/London").dt.date


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_default_scope_is_january_2026_purchase_decisions():
    settings = Settings(_env_file=None)
    assert (settings.first_decision_date, settings.last_decision_date) == (FIRST_DECISION, LAST_DECISION)
    assert (settings.first_delivery_date, settings.last_delivery_date) == (date(2026, 1, 2), date(2026, 2, 1))


@pytest.mark.skipif(not HAS_MODELS, reason="model artifacts missing")
@pytest.mark.parametrize(("path", "params"), PREDICTION_REQUESTS)
def test_every_prediction_is_for_a_january_2026_purchase_decision(client, path, params):
    points = client.get(f"/api/ml/{path}", params=params).json()["points"]
    decision_dates = uk_dates([p["decision_cutoff"] for p in points])
    delivery_dates = uk_dates([p["start_time"] for p in points])

    assert len(points) == 31 * 24
    assert decision_dates.between(FIRST_DECISION, LAST_DECISION).all()
    assert decision_dates.nunique() == 31  # every January decision day, and no other
    assert (pd.to_datetime(delivery_dates) - pd.to_datetime(decision_dates) == pd.Timedelta(days=1)).all()


@pytest.mark.skipif(not HAS_MODELS, reason="model artifacts missing")
def test_models_endpoint_states_the_january_decision_window(client):
    assert client.get("/api/ml/models").json()["decision_dates"] == {"first": "2026-01-01", "last": "2026-01-31"}


@pytest.mark.skipif(not HAS_DATA, reason="no data snapshot in backend/data")
def test_forecast_views_cover_january_decisions_only(client):
    first = client.get("/api/forecasts/day-ahead", params={"delivery_date": "2026-01-02"}).json()
    last = client.get("/api/forecasts/day-ahead", params={"delivery_date": "2026-02-01"}).json()
    assert (first["decision_date"], last["decision_date"]) == ("2026-01-01", "2026-01-31")

    deliveries = uk_dates([p["start_time"] for p in client.get("/api/analytics/timeseries").json()["points"]])
    assert (deliveries.min(), deliveries.max()) == (date(2026, 1, 2), date(2026, 2, 1))
    for delivery in ("2026-01-01", "2026-02-02"):  # decided on 31 Dec and 1 Feb: out of scope
        assert client.get("/api/forecasts/day-ahead", params={"delivery_date": delivery}).status_code == 422
