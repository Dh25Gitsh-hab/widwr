"""
Time handling - the only module that knows about time zones.

WHY
    The whole product is about *when* a forecast was known. An off-by-one-hour
    bug would silently pick the wrong forecast run and make the accuracy
    numbers look better (or worse) than what the trader really had. So every
    time-zone conversion lives here, once, and is unit-tested.

RULES
    1. Internally every timestamp is a timezone-aware pandas Timestamp in UTC
       (Elexon publishes UTC with a trailing "Z").
    2. Business concepts - the delivery day and the gate closure - are defined
       in UK local time (Europe/London) and converted to UTC here.
    3. January is GMT (= UTC), but nothing relies on that: in summer
       (BST = UTC+1) the same functions still give the right answer, and a
       delivery day correctly has 23 or 25 hours on clock-change days.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

UK_TZ_NAME = "Europe/London"
UK_TZ = ZoneInfo(UK_TZ_NAME)  # needs the "tzdata" package on Windows


def to_utc(values: pd.Series) -> pd.Series:
    """
    Parse ISO-8601 strings into timezone-aware UTC timestamps.

    Unparsable values become NaT (not an exception) so the data processor can
    count and drop them. The unit is fixed to nanoseconds so every timestamp
    column in the app has the same dtype - pandas joins require that.
    """
    return pd.to_datetime(values, utc=True, errors="coerce").dt.as_unit("ns")


def uk_day_bounds_utc(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``[start, end)`` of a UK-local calendar day, expressed in UTC."""
    start = datetime.combine(day, time(0), tzinfo=UK_TZ)
    end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=UK_TZ)
    return (
        pd.Timestamp(start).tz_convert("UTC").as_unit("ns"),
        pd.Timestamp(end).tz_convert("UTC").as_unit("ns"),
    )


def gate_closure_utc(delivery_day: date, gate_closure: time) -> pd.Timestamp:
    """
    The moment the purchase decision for ``delivery_day`` must be made.

    Day-ahead auctions close on the day *before* delivery (D-1) at a fixed UK
    local clock time, e.g. 09:50 for Nord Pool's GB auction.
    """
    decision_day = delivery_day - timedelta(days=1)
    moment = datetime.combine(decision_day, gate_closure, tzinfo=UK_TZ)
    return pd.Timestamp(moment).tz_convert("UTC").as_unit("ns")


def hourly_targets_utc(first_day: date, last_day: date) -> pd.Series:
    """
    Every hourly target time (UTC) covering UK delivery days ``first_day..last_day``.

    WINDFOR forecasts are hourly, so the hour is the unit we score at.
    """
    start, _ = uk_day_bounds_utc(first_day)
    _, end = uk_day_bounds_utc(last_day)
    index = pd.date_range(start, end, freq="1h", inclusive="left")
    return pd.Series(index.as_unit("ns"), name="startTime")


def uk_local_date(timestamps: pd.Series) -> pd.Series:
    """UK delivery date of each UTC timestamp."""
    return timestamps.dt.tz_convert(UK_TZ_NAME).dt.date


def uk_local_hour(timestamps: pd.Series) -> pd.Series:
    """UK clock hour (0-23) of each UTC timestamp - what a trader sees on the wall clock."""
    return timestamps.dt.tz_convert(UK_TZ_NAME).dt.hour


def parse_hhmm(value: str) -> time:
    """Parse a ``"HH:MM"`` string (e.g. a gate-closure query parameter) into a time."""
    try:
        hours, minutes = value.split(":")
        return time(int(hours), int(minutes))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"expected a time as HH:MM, got {value!r}") from exc
