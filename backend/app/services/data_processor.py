"""
Data validation and cleaning.

WHY
    Raw API data is untrusted. Everything downstream (forecast selection,
    metrics, ML) assumes clean, typed, de-duplicated data, so all cleaning
    happens here, once. Every removed row is *counted* - never silently
    dropped - and the counts are exposed through ``/api/health`` so a
    reviewer can see exactly what was discarded and why.

    Missing stays missing: we never fill a gap with zero. A zero would look
    like "no wind" and distort every metric.

HOW - actuals (FUELHH)
    1. keep fuelType == "WIND" only
    2. parse timestamps to UTC, generation to a number; drop unparsable rows
    3. if a half-hour was published more than once, keep the latest publication
    -> columns: startTime, publishTime, generation
       (publishTime is kept so the ML features can check *when* an actual
       value became known - see feature_builder.py)

HOW - forecasts (WINDFOR)
    1. parse timestamps to UTC, generation to a number; drop unparsable rows
    2. drop "past targets": every WINDFOR run covers a fixed window that
       starts at 21:00 the previous day, so ~20% of each run's values are for
       hours that had already started when it was published. Those values
       were never a forecast, so they must not be scored as one.
    3. drop exact duplicates of (publishTime, startTime)
    4. add horizon_hours = startTime - publishTime (how far ahead it was made)
    -> columns: publishTime, startTime, generation, horizon_hours
"""

from dataclasses import asdict, dataclass

import pandas as pd

from app.utils.timestamps import to_utc


@dataclass(frozen=True)
class CleaningReport:
    """How many rows each cleaning rule removed - an audit trail for the data."""

    dataset: str
    raw_rows: int
    invalid_rows: int = 0  # unparsable timestamp or generation value
    other_fuel_rows: int = 0  # actuals only: fuel types other than WIND
    duplicate_rows: int = 0
    past_target_rows: int = 0  # forecasts only: target hour already started at publish time
    clean_rows: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def clean_actuals(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate FUELHH records and return half-hourly wind generation."""
    _require_columns(raw, ("startTime", "fuelType", "generation"), "FUELHH")
    raw_rows = len(raw)

    is_wind = raw["fuelType"].astype(str).str.upper() == "WIND"
    df = raw.loc[is_wind].copy()
    other_fuel_rows = int((~is_wind).sum())

    df["startTime"] = to_utc(df["startTime"])
    if "publishTime" in df.columns:
        df["publishTime"] = to_utc(df["publishTime"])
    else:
        df["publishTime"] = pd.NaT
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

    valid = df["startTime"].notna() & df["generation"].notna()
    invalid_rows = int((~valid).sum())
    df = df.loc[valid]

    # A corrected value may be re-published for the same half-hour: the latest one wins.
    df = df.sort_values(["startTime", "publishTime"])
    duplicated = df.duplicated("startTime", keep="last")
    df = df.loc[~duplicated]

    clean = df[["startTime", "publishTime", "generation"]].astype({"generation": "float64"}).reset_index(drop=True)
    report = CleaningReport(
        dataset="FUELHH (actual wind generation)",
        raw_rows=raw_rows,
        invalid_rows=invalid_rows,
        other_fuel_rows=other_fuel_rows,
        duplicate_rows=int(duplicated.sum()),
        clean_rows=len(clean),
    )
    return clean, report


def clean_forecasts(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate WINDFOR records and return genuine forward-looking forecasts."""
    _require_columns(raw, ("publishTime", "startTime", "generation"), "WINDFOR")
    raw_rows = len(raw)

    df = raw[["publishTime", "startTime", "generation"]].copy()
    df["publishTime"] = to_utc(df["publishTime"])
    df["startTime"] = to_utc(df["startTime"])
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

    valid = df["publishTime"].notna() & df["startTime"].notna() & df["generation"].notna()
    invalid_rows = int((~valid).sum())
    df = df.loc[valid]

    # A value for an hour that had already started is not a forecast.
    past_target = df["startTime"] < df["publishTime"]
    df = df.loc[~past_target]

    duplicated = df.duplicated(["publishTime", "startTime"], keep="last")
    df = df.loc[~duplicated].copy()

    df["generation"] = df["generation"].astype("float64")
    df["horizon_hours"] = (df["startTime"] - df["publishTime"]).dt.total_seconds() / 3600
    clean = df.sort_values(["startTime", "publishTime"]).reset_index(drop=True)

    report = CleaningReport(
        dataset="WINDFOR (wind generation forecast)",
        raw_rows=raw_rows,
        invalid_rows=invalid_rows,
        duplicate_rows=int(duplicated.sum()),
        past_target_rows=int(past_target.sum()),
        clean_rows=len(clean),
    )
    return clean, report


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], dataset: str) -> None:
    """Fail loudly if the API contract changed (or no data came back at all)."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset} data is missing required fields {missing} ({len(df)} rows received)")
