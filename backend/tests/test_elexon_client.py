"""The Elexon client is tested against a fake HTTP transport - no network needed."""

from datetime import date, datetime, timezone

import httpx
import pytest

from app.services.elexon_client import ElexonAPIError, ElexonClient


def client_with(handler) -> ElexonClient:
    return ElexonClient(
        base_url="https://example.test/api",
        transport=httpx.MockTransport(handler),
        max_retries=3,
        chunk_days=7,
        backoff_seconds=0,
    )


def test_retries_transient_errors_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[{"startTime": "2026-01-01T00:00:00Z"}])

    with client_with(handler) as client:
        records = client.fetch_wind_actuals(date(2026, 1, 1), date(2026, 1, 1))
    assert len(records) == 1
    assert len(calls) == 2


def test_client_errors_are_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, text="bad request")

    with client_with(handler) as client, pytest.raises(ElexonAPIError, match="HTTP 400"):
        client.fetch_wind_actuals(date(2026, 1, 1), date(2026, 1, 1))
    assert len(calls) == 1


def test_gives_up_after_max_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with client_with(handler) as client, pytest.raises(ElexonAPIError, match="after 3 attempts"):
        client.fetch_wind_actuals(date(2026, 1, 1), date(2026, 1, 1))


def test_long_ranges_are_split_into_chunks():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    with client_with(handler) as client:
        client.fetch_wind_actuals(date(2026, 1, 1), date(2026, 1, 10))
        client.fetch_wind_forecasts(
            datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 10, tzinfo=timezone.utc)
        )
    assert [(p["settlementDateFrom"], p["settlementDateTo"]) for p in seen[:2]] == [
        ("2026-01-01", "2026-01-07"),
        ("2026-01-08", "2026-01-10"),
    ]
    assert seen[0]["fuelType"] == "WIND"
    assert [(p["publishDateTimeFrom"], p["publishDateTimeTo"]) for p in seen[2:]] == [
        ("2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z"),
        ("2026-01-08T00:00:00Z", "2026-01-10T00:00:00Z"),
    ]


def test_accepts_wrapped_data_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"a": 1}, {"a": 2}]})

    with client_with(handler) as client:
        assert len(client.fetch_wind_actuals(date(2026, 1, 1), date(2026, 1, 1))) == 2
