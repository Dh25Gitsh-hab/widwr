"""End-to-end tests of the HTTP API, running on the synthetic data store from conftest.py."""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import create_app


@pytest.fixture
def client(settings, store):
    with TestClient(create_app(settings=settings, store=store)) as test_client:
        yield test_client


def test_health_reports_data_and_cleaning(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["delivery_dates"] == {"first": "2026-01-03", "last": "2026-01-04"}
    assert body["decision_dates"] == {"first": "2026-01-02", "last": "2026-01-03"}
    assert body["forecast_runs"] == 3


def test_summary_default_policy(client):
    body = client.get("/api/analytics/summary").json()
    assert body["policy"]["kind"] == "day_ahead"
    assert body["policy"]["gate_closure_uk"] == "09:50"
    assert body["metrics"]["intervals_scored"] == 45
    assert body["metrics"]["missing_forecast"] == 2


def test_summary_custom_gate_closure_changes_result(client):
    body = client.get("/api/analytics/summary", params={"start": "2026-01-03", "end": "2026-01-03", "gate_closure": "09:20"}).json()
    assert body["metrics"]["bias"] == 1_000.0


def test_timeseries_is_half_hourly_with_hourly_forecasts(client):
    points = client.get("/api/analytics/timeseries", params={"start": "2026-01-03", "end": "2026-01-03"}).json()["points"]
    assert len(points) == 48
    assert points[0]["start_time"] == "2026-01-03T00:00:00Z"
    assert points[0]["forecast_mw"] == 12_000.0
    assert points[1]["forecast_mw"] is None and points[1]["actual_mw"] == 10_000.0


@pytest.mark.parametrize("by", ["hour_of_day", "delivery_date", "forecast_level"])
def test_breakdown_dimensions(client, by):
    response = client.get("/api/analytics/breakdown", params={"by": by})
    assert response.status_code == 200
    assert response.json()["rows"]


def test_horizon_endpoint(client):
    rows = client.get("/api/analytics/horizon", params={"max_hours": 24, "step_hours": 12}).json()["rows"]
    assert [row["horizon_hours"] for row in rows] == [0, 12, 24]


def test_day_ahead_endpoint(client):
    body = client.get("/api/forecasts/day-ahead", params={"delivery_date": "2026-01-03"}).json()
    assert body["decision_date"] == "2026-01-02"
    assert body["decision_cutoff"] == "2026-01-02T09:50:00Z"
    assert body["forecast_run_used"] == "2026-01-02T09:30:00Z"
    assert body["minutes_before_cutoff"] == 20.0
    assert [run["used_for_decision"] for run in body["runs"]] == [False, True, False]


def test_forecasts_published_before_filter(client):
    params = {"start": "2026-01-03", "end": "2026-01-03"}
    everything = client.get("/api/forecasts", params=params).json()
    known_then = client.get("/api/forecasts", params={**params, "published_before": "2026-01-02T09:50:00Z"}).json()
    assert everything["count"] == 72  # 3 runs x 24 h
    assert known_then["count"] == 48  # the 11:30 run was not yet published


@pytest.mark.parametrize(
    "url",
    [
        "/api/analytics/summary?gate_closure=9.50",
        "/api/analytics/summary?start=2025-12-01",
        "/api/analytics/summary?start=2026-01-04&end=2026-01-03",
        "/api/analytics/summary?policy=fixed_horizon&horizon_hours=99",
        "/api/analytics/breakdown?by=weekday",
        "/api/forecasts/day-ahead?delivery_date=2026-03-01",
        "/api/forecasts/day-ahead?delivery_date=2026-02-30",
        "/api/forecasts/day-ahead",
        "/api/forecasts?start=2026-01-03&end=2026-01-03&published_before=yesterday",
    ],
)
def test_invalid_input_returns_422(client, url):
    assert client.get(url).status_code == 422


def test_openapi_schema_documents_every_endpoint(client):
    response = client.get("/openapi.json")  # the source of the interactive /docs page
    assert response.status_code == 200
    assert set(response.json()["paths"]) == {
        "/api/health",
        "/api/forecasts", "/api/forecasts/day-ahead",
        "/api/analytics/summary", "/api/analytics/timeseries", "/api/analytics/breakdown", "/api/analytics/horizon",
        "/api/ml/evaluate", "/api/ml/models", "/api/ml/predictions", "/api/ml/corrected-forecast", "/api/ml/shortfall-risk",
    }  # fmt: skip


FRONTEND_ORIGIN = "https://gridbricks-wind-front.vercel.app"


def settings_from_env(settings, monkeypatch, cors_origins: str) -> Settings:
    """The test settings, but with WFM_CORS_ORIGINS read from the environment - as on the host (Vercel)."""
    monkeypatch.setenv("WFM_CORS_ORIGINS", cors_origins)
    return Settings(**settings.model_dump(exclude={"cors_origins"}))


# An address copied from the browser bar often ends in "/", but a browser's Origin header never does.
@pytest.mark.parametrize("configured", [f'["{FRONTEND_ORIGIN}"]', f'["{FRONTEND_ORIGIN}/"]', f'[" {FRONTEND_ORIGIN}/ "]'])
def test_cors_allows_the_configured_frontend_origin(settings, store, monkeypatch, configured):
    with TestClient(create_app(settings=settings_from_env(settings, monkeypatch, configured), store=store)) as client:
        simple = client.get("/api/health", headers={"Origin": FRONTEND_ORIGIN})
        preflight = client.options(
            "/api/health", headers={"Origin": FRONTEND_ORIGIN, "Access-Control-Request-Method": "GET"}
        )
    assert simple.status_code == 200
    assert simple.headers.get("access-control-allow-origin") == FRONTEND_ORIGIN
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == FRONTEND_ORIGIN


@pytest.mark.parametrize("origin", ["https://evil.example", "http://gridbricks-wind-front.vercel.app", "http://localhost:5173"])
def test_cors_refuses_origins_that_are_not_configured(settings, store, monkeypatch, origin):
    with TestClient(create_app(settings=settings_from_env(settings, monkeypatch, f'["{FRONTEND_ORIGIN}/"]'), store=store)) as client:
        simple = client.get("/api/health", headers={"Origin": origin})
        preflight = client.options("/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in simple.headers
    assert preflight.status_code == 400


def test_service_starts_and_reports_when_data_cannot_load(settings, monkeypatch):
    def broken_loader(_settings):
        raise ValueError("Elexon unreachable and no snapshot")

    monkeypatch.setattr(main_module, "load_store", broken_loader)
    with TestClient(create_app(settings=settings)) as client:
        health = client.get("/api/health")
        assert health.status_code == 503
        assert "Elexon unreachable" in health.json()["detail"]
        assert client.get("/api/analytics/summary").status_code == 503
