"""
Build the v2 datasets (offline): validate the raw data, then build one leakage-free row per
delivery hour with the production features plus the new v2 features.

    processed/dev_dataset.csv      decisions 1 Oct 2024 .. 31 Dec 2025 (development: CV and final fit)
    processed/january_dataset.csv  decisions 1 .. 31 Jan 2026 (context only - seen before)
    reports/data_validation.json   coverage, gaps, publish lags, demand-forecast availability

Sources: ml/v2/data/raw (history downloaded by fetch_history.py) + the project's existing
DATASETS / DATASETS1 files for Oct 2025 - Jan 2026 (read only). No February or later data is read.

Usage (from the backend folder):  python -m ml.v2.build_dataset
"""

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from app.services.data_processor import clean_actuals, clean_forecasts
from ml.v2.common import (
    DEV_FIRST_DECISION,
    DEV_LAST_DECISION,
    EXISTING_ACTUAL_FILES,
    EXISTING_FORECAST_FILES,
    HISTORY_ACTUAL_FILE,
    HISTORY_FORECAST_FILE,
    JAN_FIRST_DECISION,
    JAN_LAST_DECISION,
    NDFD_FILE,
    PROCESSED_DIR,
    REPORTS_DIR,
    sha256,
    verify_protocol,
    write_json,
)
from ml.v2.features import ALL_FEATURES, build_v2_table, clean_ndfd

DEV_FILE = PROCESSED_DIR / "dev_dataset.csv"
JANUARY_FILE = PROCESSED_DIR / "january_dataset.csv"
ID_COLUMNS = ["delivery_date", "startTime", "cutoff", "forecast_published"]
LABEL_COLUMNS = ["actual_mw", "error_mw", "shortfall_1gw"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def from_json(paths):
        return pd.DataFrame([row for path in paths for row in json.loads(path.read_text(encoding="utf-8"))])

    forecasts = pd.concat([pd.read_csv(HISTORY_FORECAST_FILE), from_json(EXISTING_FORECAST_FILES)], ignore_index=True)
    actuals = pd.concat([pd.read_csv(HISTORY_ACTUAL_FILE), from_json(EXISTING_ACTUAL_FILES)], ignore_index=True)
    return forecasts, actuals, pd.read_csv(NDFD_FILE)


def validate(forecasts: pd.DataFrame, actuals: pd.DataFrame, ndfd: pd.DataFrame, reports: dict) -> dict:
    expected = pd.date_range(actuals["startTime"].min(), actuals["startTime"].max(), freq="30min")
    missing = expected.difference(actuals["startTime"])
    runs = forecasts["publishTime"].drop_duplicates()
    runs_per_day = runs.dt.date.value_counts()
    lag_minutes = (actuals["publishTime"] - actuals["startTime"]).dt.total_seconds() / 60
    return {
        "cleaning": {name: report.as_dict() for name, report in reports.items()},
        "actuals": {
            "first_half_hour": str(actuals["startTime"].min()), "last_half_hour": str(actuals["startTime"].max()),
            "missing_half_hours": len(missing),
            "missing_days": sorted({str(t.date()) for t in missing}),
            "publish_lag_minutes": {"min": float(lag_minutes.min()), "median": float(lag_minutes.median()), "max": float(lag_minutes.max())},
        },
        "forecasts": {
            "first_run": str(runs.min()), "last_run": str(runs.max()), "runs": int(len(runs)),
            "runs_per_day": {"min": int(runs_per_day.min()), "median": float(runs_per_day.median()), "max": int(runs_per_day.max())},
            "days_with_fewer_than_6_runs": sorted(str(d) for d in runs_per_day[runs_per_day < 6].index),
        },
        "ndfd": {
            "runs": int(ndfd["publishTime"].nunique()),
            "first_run": str(ndfd["publishTime"].min()), "last_run": str(ndfd["publishTime"].max()),
            "demand_mw": {"min": float(ndfd["demand"].min()), "max": float(ndfd["demand"].max())},
        },
    }  # fmt: skip


def finish(table: pd.DataFrame, period: str) -> tuple[pd.DataFrame, dict]:
    """Keep rows with a decision-time forecast and a label; count what is excluded and why."""
    no_forecast = table["forecast_mw"].isna()
    no_label = table["error_mw"].isna() & ~no_forecast
    kept = table.loc[~no_forecast & ~no_label, ID_COLUMNS + ALL_FEATURES + ["lead_hours"] + LABEL_COLUMNS].copy()
    kept["shortfall_1gw"] = kept["shortfall_1gw"].astype(int)
    summary = {
        "period": period,
        "delivery_dates": [str(kept["delivery_date"].min()), str(kept["delivery_date"].max())],
        "rows": len(kept), "days": int(kept["delivery_date"].nunique()),
        "excluded_no_forecast": int(no_forecast.sum()),
        "excluded_no_actual": int(no_label.sum()),
        "excluded_days_without_actual": sorted({str(d) for d in table.loc[no_label, "delivery_date"]}),
        "missing_feature_values": {c: int(n) for c, n in kept[ALL_FEATURES].isna().sum().items() if n},
        "shortfall_rate": round(float(kept["shortfall_1gw"].mean()), 4),
    }  # fmt: skip
    return kept, summary


def main() -> None:
    lock = verify_protocol()
    raw_forecasts, raw_actuals, raw_ndfd = load_raw()
    forecasts, forecast_report = clean_forecasts(raw_forecasts)
    actuals, actual_report = clean_actuals(raw_actuals)
    ndfd = clean_ndfd(raw_ndfd)
    # The January files end with 1 Feb (delivery for the 31 Jan decision); nothing later may be read.
    latest_input = max(forecasts["publishTime"].max(), actuals["startTime"].max())
    if latest_input >= pd.Timestamp(JAN_LAST_DECISION + timedelta(days=2), tz="UTC"):
        raise SystemExit(f"Input data reaches {latest_input}: data after the January application window must not be read here")

    dev, dev_summary = finish(
        build_v2_table(forecasts, actuals, ndfd, DEV_FIRST_DECISION + timedelta(days=1), DEV_LAST_DECISION + timedelta(days=1)),
        "development: decisions 2024-10-01 .. 2025-12-31",
    )
    january, january_summary = finish(
        build_v2_table(forecasts, actuals, ndfd, JAN_FIRST_DECISION + timedelta(days=1), JAN_LAST_DECISION + timedelta(days=1)),
        "January 2026 application period: context only",
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dev.to_csv(DEV_FILE, index=False, float_format="%.4f")
    january.to_csv(JANUARY_FILE, index=False, float_format="%.4f")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": lock["sha256"],
        "inputs": {p.name: sha256(p) for p in [HISTORY_FORECAST_FILE, HISTORY_ACTUAL_FILE, NDFD_FILE,
                                              *EXISTING_FORECAST_FILES, *EXISTING_ACTUAL_FILES]},
        "validation": validate(forecasts, actuals, ndfd, {"forecasts": forecast_report, "actuals": actual_report}),
        "datasets": {DEV_FILE.name: {**dev_summary, "sha256": sha256(DEV_FILE)},
                     JANUARY_FILE.name: {**january_summary, "sha256": sha256(JANUARY_FILE)}},
    }  # fmt: skip
    write_json(REPORTS_DIR / "data_validation.json", report)
    for name, summary in report["datasets"].items():
        print(f"{name}: {summary['rows']:,} rows, {summary['days']} days {summary['delivery_dates']}, "
              f"shortfall rate {summary['shortfall_rate']}, excluded no-forecast {summary['excluded_no_forecast']} / "
              f"no-actual {summary['excluded_no_actual']} {summary['excluded_days_without_actual']}")  # fmt: skip
        print(f"   missing feature values: {summary['missing_feature_values'] or 'none'}")
    v = report["validation"]
    print(f"actuals: missing half-hours {v['actuals']['missing_half_hours']} on {v['actuals']['missing_days']}")
    print(f"forecasts: {v['forecasts']['runs']} runs, per day {v['forecasts']['runs_per_day']}, thin days {v['forecasts']['days_with_fewer_than_6_runs']}")
    print(f"ndfd: {v['ndfd']}")


if __name__ == "__main__":
    main()
