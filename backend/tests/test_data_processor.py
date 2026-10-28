import pandas as pd
import pytest

from app.services.data_processor import clean_actuals, clean_forecasts


def test_actuals_keep_wind_only_and_drop_invalid_rows():
    raw = pd.DataFrame(
        [
            {"startTime": "2026-01-10T00:00:00Z", "publishTime": "2026-01-10T00:30:00Z", "fuelType": "WIND", "generation": 9000},
            {"startTime": "2026-01-10T00:00:00Z", "publishTime": "2026-01-10T00:30:00Z", "fuelType": "CCGT", "generation": 5000},
            {"startTime": "2026-01-10T00:30:00Z", "publishTime": "2026-01-10T01:00:00Z", "fuelType": "WIND", "generation": "n/a"},
            {"startTime": "garbage", "publishTime": "2026-01-10T01:00:00Z", "fuelType": "WIND", "generation": 1},
        ]
    )  # fmt: skip
    clean, report = clean_actuals(raw)
    assert clean["generation"].tolist() == [9000.0]
    assert (report.raw_rows, report.other_fuel_rows, report.invalid_rows, report.clean_rows) == (4, 1, 2, 1)


def test_actuals_keep_latest_publication_of_a_half_hour():
    raw = pd.DataFrame(
        [
            {"startTime": "2026-01-10T00:00:00Z", "publishTime": "2026-01-10T00:30:00Z", "fuelType": "WIND", "generation": 9000},
            {"startTime": "2026-01-10T00:00:00Z", "publishTime": "2026-01-10T05:00:00Z", "fuelType": "WIND", "generation": 9100},
        ]
    )  # fmt: skip
    clean, report = clean_actuals(raw)
    assert clean["generation"].tolist() == [9100.0]
    assert report.duplicate_rows == 1


def test_forecasts_drop_targets_that_already_started():
    raw = pd.DataFrame(
        [
            {"publishTime": "2026-01-10T09:30:00Z", "startTime": "2026-01-10T08:00:00Z", "generation": 1},  # past
            {"publishTime": "2026-01-10T09:30:00Z", "startTime": "2026-01-10T10:00:00Z", "generation": 2},
            {"publishTime": "2026-01-10T09:30:00Z", "startTime": "2026-01-11T09:30:00Z", "generation": 3},
        ]
    )  # fmt: skip
    clean, report = clean_forecasts(raw)
    assert clean["generation"].tolist() == [2.0, 3.0]
    assert clean["horizon_hours"].tolist() == [0.5, 24.0]
    assert report.past_target_rows == 1


def test_forecasts_drop_duplicates_and_invalid_values():
    raw = pd.DataFrame(
        [
            {"publishTime": "2026-01-10T09:30:00Z", "startTime": "2026-01-10T10:00:00Z", "generation": 2},
            {"publishTime": "2026-01-10T09:30:00Z", "startTime": "2026-01-10T10:00:00Z", "generation": 2},
            {"publishTime": "2026-01-10T09:30:00Z", "startTime": "2026-01-10T11:00:00Z", "generation": None},
        ]
    )  # fmt: skip
    clean, report = clean_forecasts(raw)
    assert len(clean) == 1
    assert (report.duplicate_rows, report.invalid_rows) == (1, 1)


@pytest.mark.parametrize("cleaner", [clean_actuals, clean_forecasts])
def test_missing_columns_fail_loudly(cleaner):
    with pytest.raises(ValueError, match="missing required fields"):
        cleaner(pd.DataFrame())
