"""
Elexon Insights (BMRS) API client.

WHY a separate module
    This is the only code that talks to the outside world. Isolating it means:
      * the rest of the backend never deals with HTTP, URLs or retries;
      * tests run without a network (we inject a fake transport);
      * if Elexon changes its API, only this file changes.

HOW
    A thin wrapper around ``httpx`` with three robustness features:
      * chunking - long windows are split into N-day requests. The "stream"
        endpoints are not paginated, so one huge request means one huge,
        fragile response;
      * retries with exponential back-off on timeouts, HTTP 429 and 5xx;
      * one domain error type (``ElexonAPIError``) so callers never have to
        know about httpx exceptions.

    It returns the raw JSON records untouched. Validation and cleaning are
    the data processor's job (single responsibility).

DATASETS
    FUELHH  - half-hourly outturn generation by fuel type (we request WIND).
    WINDFOR - wind generation forecasts; one "run" per publishTime, hourly values.
"""

import logging
import time as clock
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

FUELHH_STREAM = "/datasets/FUELHH/stream"
WINDFOR_STREAM = "/datasets/WINDFOR/stream"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ElexonAPIError(RuntimeError):
    """Elexon could not be reached or returned an unusable response."""


class ElexonClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        chunk_days: int = 7,
        transport: httpx.BaseTransport | None = None,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers={"Accept": "application/json"},
        )
        self._max_retries = max(1, max_retries)
        self._chunk_days = max(1, chunk_days)
        self._backoff_seconds = backoff_seconds

    # Context-manager support so the connection pool is always closed.
    def __enter__(self) -> "ElexonClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ---- Public API --------------------------------------------------------

    def fetch_wind_actuals(self, first_day: date, last_day: date) -> list[dict[str, Any]]:
        """FUELHH records with fuelType=WIND for settlement dates ``first_day..last_day`` (inclusive)."""
        records: list[dict[str, Any]] = []
        for chunk_start, chunk_end in _date_chunks(first_day, last_day, self._chunk_days):
            records += self._get(
                FUELHH_STREAM,
                {
                    "settlementDateFrom": chunk_start.isoformat(),
                    "settlementDateTo": chunk_end.isoformat(),
                    "fuelType": "WIND",
                },
            )
        log.info("FUELHH: %d wind records for %s..%s", len(records), first_day, last_day)
        return records

    def fetch_wind_forecasts(self, publish_from: datetime, publish_to: datetime) -> list[dict[str, Any]]:
        """WINDFOR records whose publishTime falls in ``[publish_from, publish_to)``."""
        records: list[dict[str, Any]] = []
        for chunk_start, chunk_end in _datetime_chunks(publish_from, publish_to, self._chunk_days):
            records += self._get(
                WINDFOR_STREAM,
                {
                    "publishDateTimeFrom": _iso_utc(chunk_start),
                    "publishDateTimeTo": _iso_utc(chunk_end),
                },
            )
        log.info("WINDFOR: %d forecast records published %s..%s", len(records), publish_from, publish_to)
        return records

    # ---- Internals ---------------------------------------------------------

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """GET one chunk, retrying transient failures with exponential back-off."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:  # timeouts, DNS, refused/reset connections
                last_error = exc
            else:
                if response.status_code == 200:
                    return _parse_records(response, path)
                if response.status_code not in _RETRYABLE_STATUS:
                    # 4xx (other than 429) means our request is wrong - retrying will not help.
                    raise ElexonAPIError(f"{path} returned HTTP {response.status_code}: {response.text[:200]}")
                last_error = ElexonAPIError(f"HTTP {response.status_code}")

            if attempt < self._max_retries:
                wait = self._backoff_seconds * 2 ** (attempt - 1)
                log.warning("Elexon %s failed (%s); retry %d in %.0fs", path, last_error, attempt, wait)
                clock.sleep(wait)

        raise ElexonAPIError(f"{path} failed after {self._max_retries} attempts: {last_error}") from last_error


def _parse_records(response: httpx.Response, path: str) -> list[dict[str, Any]]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ElexonAPIError(f"{path} returned invalid JSON") from exc
    # Stream endpoints return a bare JSON list; other Elexon endpoints wrap it in {"data": [...]}.
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ElexonAPIError(f"{path} returned an unexpected payload type: {type(payload).__name__}")
    return payload


def _date_chunks(first: date, last: date, size_days: int) -> Iterator[tuple[date, date]]:
    """Split an inclusive date range into consecutive inclusive chunks."""
    start = first
    while start <= last:
        end = min(start + timedelta(days=size_days - 1), last)
        yield start, end
        start = end + timedelta(days=1)


def _datetime_chunks(start: datetime, end: datetime, size_days: int) -> Iterator[tuple[datetime, datetime]]:
    """Split a half-open datetime range into consecutive half-open chunks."""
    while start < end:
        stop = min(start + timedelta(days=size_days), end)
        yield start, stop
        start = stop


def _iso_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
