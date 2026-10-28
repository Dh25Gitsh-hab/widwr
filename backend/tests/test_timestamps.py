from datetime import date, time

import pandas as pd
import pytest

from app.utils.timestamps import (
    gate_closure_utc,
    hourly_targets_utc,
    parse_hhmm,
    to_utc,
    uk_day_bounds_utc,
    uk_local_date,
)


def test_gate_closure_in_winter_equals_utc():
    # GMT = UTC, decision on the previous day
    assert gate_closure_utc(date(2026, 1, 15), time(9, 50)) == pd.Timestamp("2026-01-14T09:50Z")


def test_gate_closure_in_summer_accounts_for_bst():
    # 09:50 BST is 08:50 UTC
    assert gate_closure_utc(date(2026, 7, 15), time(9, 50)) == pd.Timestamp("2026-07-14T08:50Z")


@pytest.mark.parametrize(
    ("day", "hours"),
    [(date(2026, 1, 15), 24), (date(2026, 3, 29), 23), (date(2026, 10, 25), 25)],  # normal, spring, autumn
)
def test_uk_delivery_day_length_handles_clock_changes(day, hours):
    start, end = uk_day_bounds_utc(day)
    assert (end - start) == pd.Timedelta(hours=hours)
    assert len(hourly_targets_utc(day, day)) == hours


def test_summer_delivery_day_starts_at_2300_utc_previous_day():
    start, _ = uk_day_bounds_utc(date(2026, 7, 15))
    assert start == pd.Timestamp("2026-07-14T23:00Z")


def test_uk_local_date_of_utc_timestamps():
    stamps = pd.Series(pd.to_datetime(["2026-07-14T23:30Z", "2026-01-14T23:30Z"]))
    assert list(uk_local_date(stamps)) == [date(2026, 7, 15), date(2026, 1, 14)]


def test_to_utc_turns_garbage_into_nat():
    parsed = to_utc(pd.Series(["2026-01-01T00:00:00Z", "not a time"]))
    assert parsed.iloc[0] == pd.Timestamp("2026-01-01T00:00Z")
    assert pd.isna(parsed.iloc[1])
    assert str(parsed.dtype) == "datetime64[ns, UTC]"


def test_parse_hhmm():
    assert parse_hhmm("09:50") == time(9, 50)
    with pytest.raises(ValueError):
        parse_hhmm("9.50")
