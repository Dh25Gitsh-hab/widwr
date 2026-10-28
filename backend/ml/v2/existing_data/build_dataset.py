"""
Build the existing-data datasets: one leakage-free row per delivery hour with the 9 production
features plus the wind-only v2 features (ml/v2/features.py, no demand forecast).

    processed/dev.csv        deliveries 2 Oct - 31 Dec 2025 (development: CV, tuning, final fit)
    processed/january.csv    deliveries 2 Jan - 1 Feb 2026 (application period; seen before)
    processed/february.csv   deliveries 2 Feb - 28 Feb 2026 (previously examined; 1 Mar has no actuals)
    reports/data_validation.json

Inputs: only the Elexon files already in DATASETS/ and DATASETS1/ (Oct 2025 - Feb 2026).

Usage (from the backend folder):  python -m ml.v2.existing_data.build_dataset
"""

import json
from datetime import datetime, timezone

import pandas as pd

from app.services.data_processor import clean_actuals, clean_forecasts
from ml.v2.common import sha256, write_json
from ml.v2.existing_data.config import ACTUAL_FILES, DEV, FEBRUARY, FORECAST_FILES, JANUARY, PROCESSED_DIR, REPORTS_DIR
from ml.v2.features import WIND_FEATURES, build_v2_table

ID_COLUMNS = ["delivery_date", "startTime", "cutoff", "forecast_published"]
LABEL_COLUMNS = ["actual_mw", "error_mw", "shortfall_1gw"]
PERIODS = {"dev": DEV, "january": JANUARY, "february": FEBRUARY}


def load_raw(paths) -> pd.DataFrame:
    return pd.DataFrame([row for path in paths for row in json.loads(path.read_text(encoding="utf-8"))])


def finish(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep rows with a decision-time forecast and a label; count every exclusion."""
    no_forecast = table["forecast_mw"].isna()
    no_label = table["error_mw"].isna() & ~no_forecast
    kept = table.loc[~no_forecast & ~no_label, ID_COLUMNS + WIND_FEATURES + ["lead_hours"] + LABEL_COLUMNS].copy()
    kept["shortfall_1gw"] = kept["shortfall_1gw"].astype(int)
    return kept, {
        "delivery_dates": [str(kept["delivery_date"].min()), str(kept["delivery_date"].max())],
        "rows": len(kept), "days": int(kept["delivery_date"].nunique()),
        "excluded_no_forecast": int(no_forecast.sum()), "excluded_no_actual": int(no_label.sum()),
        "excluded_days_without_actual": sorted({str(d) for d in table.loc[no_label, "delivery_date"]}),
        "missing_feature_values": {c: int(n) for c, n in kept[WIND_FEATURES].isna().sum().items() if n},
        "shortfall_rate": round(float(kept["shortfall_1gw"].mean()), 4),
    }  # fmt: skip


def main() -> None:
    forecasts, forecast_report = clean_forecasts(load_raw(FORECAST_FILES))
    actuals, actual_report = clean_actuals(load_raw(ACTUAL_FILES))
    expected = pd.date_range(actuals["startTime"].min(), actuals["startTime"].max(), freq="30min")
    missing = expected.difference(actuals["startTime"])
    runs = forecasts["publishTime"].drop_duplicates()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, (first, last) in PERIODS.items():
        kept, summary = finish(build_v2_table(forecasts, actuals, None, first, last))
        path = PROCESSED_DIR / f"{name}.csv"
        kept.to_csv(path, index=False, float_format="%.4f")
        datasets[path.name] = {**summary, "sha256": sha256(path)}

    write_json(REPORTS_DIR / "data_validation.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {p.relative_to(p.parents[1]).as_posix(): sha256(p) for p in [*FORECAST_FILES, *ACTUAL_FILES]},
        "cleaning": {"forecasts": forecast_report.as_dict(), "actuals": actual_report.as_dict()},
        "actuals": {"first": str(actuals["startTime"].min()), "last": str(actuals["startTime"].max()),
                    "missing_half_hours": len(missing), "missing_days": sorted({str(t.date()) for t in missing})},
        "forecasts": {"runs": int(len(runs)), "first_run": str(runs.min()), "last_run": str(runs.max()),
                      "runs_per_day_min_max": [int(runs.dt.date.value_counts().min()), int(runs.dt.date.value_counts().max())]},
        "datasets": datasets,
    })  # fmt: skip
    for name, s in datasets.items():
        print(f"{name}: {s['rows']} rows, {s['days']} days {s['delivery_dates']}, shortfall rate {s['shortfall_rate']}, "
              f"excluded no-forecast {s['excluded_no_forecast']} / no-actual {s['excluded_no_actual']} {s['excluded_days_without_actual']}; "
              f"missing features {s['missing_feature_values'] or 'none'}")  # fmt: skip
    print(f"actuals missing half-hours: {len(missing)} on {sorted({str(t.date()) for t in missing})}")


if __name__ == "__main__":
    main()
