"""
Download the pre-January history the v2 experiments need (offline; the app never reads it).

    WINDFOR  wind forecasts published 24 Sep 2024 .. 30 Sep 2025   (the project's files start in Oct 2025)
    FUELHH   wind actuals, settlement dates 24 Sep 2024 .. 30 Sep 2025
    NDFD     daily national demand forecasts published 24 Sep 2024 .. 30 Jan 2026

Every record keeps its publishTime, so each feature can prove it was available at the cutoff.
Oct 2025 - Jan 2026 wind data is read from the existing DATASETS folders, never re-downloaded.

Usage (from the backend folder):  python -m ml.v2.fetch_history
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from app.config import get_settings
from app.services.elexon_client import ElexonClient, _datetime_chunks, _iso_utc
from ml.v2.common import (
    HISTORY_ACTUAL_FILE,
    HISTORY_ACTUALS_SETTLEMENT,
    HISTORY_FORECAST_FILE,
    HISTORY_FORECASTS_PUBLISHED,
    NDFD_FILE,
    NDFD_PUBLISHED,
    RAW_DIR,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
NDFD_STREAM = "/datasets/NDFD/stream"


class HistoryClient(ElexonClient):
    """The production client (retries, chunking) plus the one extra dataset v2 needs."""

    def fetch_ndfd(self, publish_from: datetime, publish_to: datetime) -> list[dict]:
        records: list[dict] = []
        for start, stop in _datetime_chunks(publish_from, publish_to, 30):
            records += self._get(NDFD_STREAM, {"publishDateTimeFrom": _iso_utc(start), "publishDateTimeTo": _iso_utc(stop)})
        return records


def main() -> None:
    settings = get_settings()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with HistoryClient(settings.elexon_base_url, settings.elexon_timeout_seconds, settings.elexon_max_retries,
                       settings.elexon_chunk_days) as client:  # fmt: skip
        forecasts = pd.DataFrame(client.fetch_wind_forecasts(*HISTORY_FORECASTS_PUBLISHED))
        actuals = pd.DataFrame(client.fetch_wind_actuals(*HISTORY_ACTUALS_SETTLEMENT))
        ndfd = pd.DataFrame(client.fetch_ndfd(*NDFD_PUBLISHED))

    outputs = {HISTORY_FORECAST_FILE: forecasts, HISTORY_ACTUAL_FILE: actuals, NDFD_FILE: ndfd}
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
        print(f"{path.name}: {len(frame):,} rows")
    write_json(RAW_DIR / "fetch_meta.json", {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": settings.elexon_base_url,
        "windows": {"WINDFOR_published": [str(t) for t in HISTORY_FORECASTS_PUBLISHED],
                    "FUELHH_settlement_dates": [str(d) for d in HISTORY_ACTUALS_SETTLEMENT],
                    "NDFD_published": [str(t) for t in NDFD_PUBLISHED]},
        "rows": {path.name: len(frame) for path, frame in outputs.items()},
    })  # fmt: skip


if __name__ == "__main__":
    main()
