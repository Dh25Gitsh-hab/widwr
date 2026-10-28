"""
Refresh the Elexon data snapshot in backend/data/.

The API normally serves the committed CSV snapshot, so this is only needed
to re-download the data (e.g. after Elexon revisions, or after changing the
evaluation window in config / environment variables).

Usage, from the backend folder:
    python -m scripts.fetch_data
"""

import logging

from app.config import get_settings
from app.services.data_processor import clean_actuals, clean_forecasts
from app.services.data_store import refresh_snapshot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    print(
        f"Decisions {settings.first_decision_date} .. {settings.last_decision_date} "
        f"-> delivery days {settings.first_delivery_date} .. {settings.last_delivery_date}"
    )
    raw_actuals, raw_forecasts = refresh_snapshot(settings)

    # Clean once here too, so problems show up now rather than at API start-up.
    for _, report in (clean_actuals(raw_actuals), clean_forecasts(raw_forecasts)):
        print(report)
    print(f"Snapshot written to {settings.data_dir}")


if __name__ == "__main__":
    main()
