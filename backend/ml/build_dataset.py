"""
ML step 1 - validate the raw files, then build ONE clean, leakage-free dataset.

Inputs (read-only - the original files are never modified)
    <project>/DATASETS/*.json    FUELHH actual wind generation, one file per month
    <project>/DATASETS1/*.json   WINDFOR wind forecasts, one file per month

Outputs
    ml/reports/data_validation.json   per-file and combined checks
    ml/processed/ml_dataset.csv       one row per delivery hour: ids, features, labels, split
    ml/reports/dataset_summary.json   rows kept / excluded and why, split sizes

Usage (from the backend folder)
    python -m ml.build_dataset [--actuals-dir PATH] [--forecasts-dir PATH]

If validation finds an *error* (wrong schema, wrong dataset, unparsable values)
the script stops before writing anything - no model is ever trained on
unvalidated data. *Warnings* (e.g. a missing day) are recorded and handled
explicitly: affected rows are excluded, never filled in.

HOW THE FILES ARE COMBINED
    1. read every monthly file; validate each one on its own
    2. concatenate; check for overlaps / conflicts between files and gaps
    3. clean with the same code the API uses (data_processor)
    4. build features with the 09:50 UK decision cutoff (feature_builder)
    5. drop hours with no decision-time forecast or no actual (counted)
    6. label each row train / validation / test by its DECISION month
"""

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.services.data_processor import clean_actuals, clean_forecasts
from app.services.feature_builder import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, build_feature_table
from app.utils.timestamps import uk_local_date

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent.parent
PROCESSED_FILE = ML_DIR / "processed" / "ml_dataset.csv"
REPORTS_DIR = ML_DIR / "reports"

GATE_CLOSURE = time(9, 50)  # UK local time on D-1 (Nord Pool GB day-ahead)

# Chronological split by the month the purchase DECISION is made (D-1).
# January 2026 decisions are the brief's scope, so they are the untouched test set.
SPLITS = [
    ("train", date(2025, 10, 1), date(2025, 11, 30)),
    ("validation", date(2025, 12, 1), date(2025, 12, 31)),
    ("test", date(2026, 1, 1), date(2026, 1, 31)),
]

# The training pipeline reads exactly these monthly files. Later months (e.g. the
# February 2026 holdout, see ml/holdout/) sit in the same folders but are
# deliberately excluded here, so they can never leak into training, validation
# or model selection.
TRAINING_MONTHS = ("Oct25", "Nov25", "Dec25", "Jan26")

ACTUAL_KEYS = {"dataset", "publishTime", "startTime", "settlementDate", "settlementPeriod", "fuelType", "generation"}
FORECAST_KEYS = {"dataset", "publishTime", "startTime", "generation"}


# ---- 1. Validation ---------------------------------------------------------------


def validate_actual_file(name: str, records: list[dict], issues: dict) -> dict:
    df = pd.DataFrame(records)
    _check_schema(name, records, ACTUAL_KEYS, issues)
    if not ACTUAL_KEYS <= set(df.columns):
        return {"file": name, "records": len(records)}
    start = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
    generation = pd.to_numeric(df["generation"], errors="coerce")
    _check_values(name, df, start, generation, "FUELHH", issues)
    if set(df["fuelType"]) != {"WIND"}:
        issues["errors"].append(f"{name}: unexpected fuel types {sorted(set(df['fuelType']))}")

    uk_date = start.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    periods_per_day = df.groupby("settlementDate")["settlementPeriod"].count()
    gaps = _missing_ranges(start.dropna(), "30min")
    if gaps:
        issues["warnings"].append(f"{name}: missing half-hours {gaps}")
    return {
        "file": name,
        "records": len(df),
        "start_time_range": [str(start.min()), str(start.max())],
        "settlement_date_range": [df["settlementDate"].min(), df["settlementDate"].max()],
        "null_values": int(df.isna().sum().sum()),
        "non_numeric_generation": int(generation.isna().sum()),
        "negative_generation": int((generation < 0).sum()),
        "duplicate_start_times": int(start.duplicated().sum()),
        "missing_half_hours": gaps,
        "settlement_date_matches_uk_date": bool((uk_date == df["settlementDate"]).all()),
        "days_without_48_periods": {day: int(n) for day, n in periods_per_day[periods_per_day != 48].items()},
        "generation_mw": _describe(generation),
    }


def validate_forecast_file(name: str, records: list[dict], issues: dict) -> dict:
    df = pd.DataFrame(records)
    _check_schema(name, records, FORECAST_KEYS, issues)
    if not FORECAST_KEYS <= set(df.columns):
        return {"file": name, "records": len(records)}
    publish = pd.to_datetime(df["publishTime"], utc=True, errors="coerce")
    start = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
    generation = pd.to_numeric(df["generation"], errors="coerce")
    _check_values(name, df, start, generation, "WINDFOR", issues)
    if publish.isna().any():
        issues["errors"].append(f"{name}: {int(publish.isna().sum())} unparsable publishTime values")

    run_sizes = publish.value_counts()
    runs_per_day = publish.drop_duplicates().dt.tz_convert("Europe/London").dt.date.value_counts()
    irregular_days = {str(day): int(n) for day, n in runs_per_day.items() if n != 8}
    if irregular_days:
        issues["warnings"].append(f"{name}: days with a run count other than 8: {irregular_days}")
    return {
        "file": name,
        "records": len(df),
        "publish_time_range": [str(publish.min()), str(publish.max())],
        "start_time_range": [str(start.min()), str(start.max())],
        "runs": int(len(run_sizes)),
        "values_per_run": {int(k): int(v) for k, v in run_sizes.value_counts().items()},
        "days_with_irregular_run_count": irregular_days,
        "null_values": int(df.isna().sum().sum()),
        "non_numeric_generation": int(generation.isna().sum()),
        "negative_generation": int((generation < 0).sum()),
        "duplicate_publish_start_pairs": int(pd.DataFrame({"p": publish, "s": start}).duplicated().sum()),
        "targets_not_on_the_hour": int((start.dt.minute != 0).sum()),
        "targets_before_publish_time": int((start < publish).sum()),
        "generation_mw": _describe(generation),
    }


def validate_combined(actuals: pd.DataFrame, forecasts: pd.DataFrame, issues: dict) -> dict:
    a_start = pd.to_datetime(actuals["startTime"], utc=True)
    f_publish = pd.to_datetime(forecasts["publishTime"], utc=True)
    f_start = pd.to_datetime(forecasts["startTime"], utc=True)

    conflicts = actuals.assign(s=a_start).groupby("s")["generation"].nunique().gt(1).sum()
    if conflicts:
        issues["errors"].append(f"{conflicts} half-hours have conflicting actual values across files")
    gaps = _missing_ranges(a_start.drop_duplicates(), "30min")
    if gaps:
        issues["warnings"].append(f"combined actuals are missing {gaps} - affected hours are excluded, not filled")
    run_times = f_publish.drop_duplicates().sort_values()
    return {
        "actuals": {
            "records": len(actuals),
            "start_time_range": [str(a_start.min()), str(a_start.max())],
            "duplicates_across_files": int(a_start.duplicated().sum()),
            "conflicting_duplicates": int(conflicts),
            "missing_half_hours": gaps,
        },
        "forecasts": {
            "records": len(forecasts),
            "runs": int(run_times.size),
            "publish_time_range": [str(run_times.min()), str(run_times.max())],
            "longest_gap_between_runs_hours": float(run_times.diff().max().total_seconds() / 3600),
            "duplicates_across_files": int(pd.DataFrame({"p": f_publish, "s": f_start}).duplicated().sum()),
        },
        "units": "Both datasets report generation in MW (Elexon schema). "
        f"Mean actual {actuals['generation'].mean():,.0f} MW vs mean forecast {forecasts['generation'].mean():,.0f} MW "
        "- same order of magnitude, so no unit conversion is needed.",
    }


def _check_schema(name: str, records: list[dict], required: set, issues: dict) -> None:
    missing = {key for record in records for key in required - record.keys()}
    if missing:
        issues["errors"].append(f"{name}: records missing required fields {sorted(missing)}")


def _check_values(name, df, start, generation, dataset, issues) -> None:
    if set(df["dataset"]) != {dataset}:
        issues["errors"].append(f"{name}: expected dataset {dataset}, found {sorted(set(df['dataset']))}")
    if start.isna().any():
        issues["errors"].append(f"{name}: {int(start.isna().sum())} unparsable startTime values")
    if generation.isna().any():
        issues["errors"].append(f"{name}: {int(generation.isna().sum())} non-numeric generation values")


def _missing_ranges(times: pd.Series, freq: str) -> list[str]:
    """Collapse missing timestamps into readable 'first -> last (n)' ranges."""
    if times.empty:
        return []
    expected = pd.date_range(times.min(), times.max(), freq=freq)
    missing = expected.difference(pd.DatetimeIndex(times))
    if missing.empty:
        return []
    step = pd.Timedelta(freq)
    groups = (pd.Series(missing).diff() != step).cumsum()
    return [
        f"{chunk.iloc[0]:%Y-%m-%d %H:%M} -> {chunk.iloc[-1]:%Y-%m-%d %H:%M} UTC ({len(chunk)})"
        for _, chunk in pd.Series(missing).groupby(groups)
    ]


def _describe(values: pd.Series) -> dict:
    return {"min": float(values.min()), "mean": round(float(values.mean()), 1), "max": float(values.max())}


# ---- 2. Build -------------------------------------------------------------------


def assign_split(decision_dates: pd.Series) -> pd.Series:
    split = pd.Series("unused", index=decision_dates.index)
    for name, first, last in SPLITS:
        split[(decision_dates >= first) & (decision_dates <= last)] = name
    return split


def training_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.json") if path.name.startswith(TRAINING_MONTHS))


def build(actuals_dir: Path, forecasts_dir: Path, output_dir: Path = ML_DIR) -> None:
    issues: dict = {"errors": [], "warnings": []}
    processed_file = output_dir / PROCESSED_FILE.relative_to(ML_DIR)
    reports_dir = output_dir / REPORTS_DIR.relative_to(ML_DIR)
    actual_files = training_files(actuals_dir)
    forecast_files = training_files(forecasts_dir)
    if not actual_files or not forecast_files:
        sys.exit(f"No JSON files found in {actuals_dir} and/or {forecasts_dir}")

    print(f"Reading {len(actual_files)} actual and {len(forecast_files)} forecast files ...")
    raw_actuals, raw_forecasts, per_file = [], [], []
    for path in actual_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        per_file.append(validate_actual_file(path.name, records, issues))
        raw_actuals.append(pd.DataFrame(records))
    for path in forecast_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        per_file.append(validate_forecast_file(path.name, records, issues))
        raw_forecasts.append(pd.DataFrame(records))
    actuals_raw = pd.concat(raw_actuals, ignore_index=True)
    forecasts_raw = pd.concat(raw_forecasts, ignore_index=True)
    combined = validate_combined(actuals_raw, forecasts_raw, issues)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"actuals_dir": str(actuals_dir), "forecasts_dir": str(forecasts_dir)},
        "files": per_file,
        "combined": combined,
        "errors": issues["errors"],
        "warnings": issues["warnings"],
        "passed": not issues["errors"],
    }
    _write_json(reports_dir / "data_validation.json", report)
    for warning in issues["warnings"]:
        print(f"  WARNING  {warning}")
    if issues["errors"]:
        for error in issues["errors"]:
            print(f"  ERROR    {error}")
        sys.exit("Validation failed - dataset NOT built. See ml/reports/data_validation.json")
    print("Validation passed.")

    # Clean with the API's own rules, then build leakage-free features.
    actuals, actuals_report = clean_actuals(actuals_raw)
    forecasts, forecasts_report = clean_forecasts(forecasts_raw)
    first_day, last_day = uk_local_date(actuals["startTime"]).min(), uk_local_date(actuals["startTime"]).max()
    table = build_feature_table(forecasts, actuals, first_day, last_day, GATE_CLOSURE)

    no_forecast = table["forecast_mw"].isna()
    no_actual = table["actual_mw"].isna() & ~no_forecast
    excluded_days = {
        "no_decision_time_forecast": sorted({str(d) for d in table.loc[no_forecast, "delivery_date"]}),
        "no_actual_generation": sorted({str(d) for d in table.loc[no_actual, "delivery_date"]}),
    }
    dataset = table.loc[~no_forecast & ~no_actual].copy()
    dataset.insert(1, "decision_date", dataset["delivery_date"] - timedelta(days=1))
    dataset.insert(0, "split", assign_split(dataset["decision_date"]))
    unused = int((dataset["split"] == "unused").sum())
    dataset = dataset.loc[dataset["split"] != "unused"]

    processed_file.parent.mkdir(parents=True, exist_ok=True)
    out = dataset.copy()
    for column in ("startTime", "cutoff", "forecast_published"):
        out[column] = out[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(processed_file, index=False, float_format="%.3f")

    summary = {
        "generated_at": report["generated_at"],
        "decision_rule": f"Latest forecast published by {GATE_CLOSURE:%H:%M} UK local time on D-1 (DST-aware)",
        "cleaning": [actuals_report.as_dict(), forecasts_report.as_dict()],
        "delivery_hours_built": len(table),
        "excluded": {
            "hours_without_decision_time_forecast": int(no_forecast.sum()),
            "hours_without_actual": int(no_actual.sum()),
            "hours_outside_split_months": unused,
            "days": excluded_days,
        },
        "rows_in_dataset": len(dataset),
        "splits": [
            {
                "name": name,
                "decision_dates": [str(first), str(last)],
                "delivery_dates": [str(part["delivery_date"].min()), str(part["delivery_date"].max())],
                "days": int(part["delivery_date"].nunique()),
                "hours": len(part),
                "shortfall_rate": round(float(part["shortfall_1gw"].mean()), 3),
                "mean_error_mw": round(float(part["error_mw"].mean()), 1),
            }
            for name, first, last in SPLITS
            for part in [dataset.loc[dataset["split"] == name]]
        ],
        "features": FEATURE_DESCRIPTIONS,
        "missing_feature_values": {c: int(n) for c, n in dataset[FEATURE_COLUMNS].isna().sum().items() if n},
    }
    _write_json(reports_dir / "dataset_summary.json", summary)

    print(f"Dataset: {len(dataset)} hourly rows -> {processed_file}")
    for part in summary["splits"]:
        print(
            f"  {part['name']:<10} decisions {part['decision_dates'][0]}..{part['decision_dates'][1]}  "
            f"{part['days']:>3} days {part['hours']:>5} hours  shortfall rate {part['shortfall_rate']:.0%}"
        )
    print(f"  excluded: {summary['excluded']}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--actuals-dir", type=Path, default=PROJECT_DIR / "DATASETS")
    parser.add_argument("--forecasts-dir", type=Path, default=PROJECT_DIR / "DATASETS1")
    parser.add_argument("--output-dir", type=Path, default=ML_DIR, help="where processed/ and reports/ are written")
    args = parser.parse_args()
    build(args.actuals_dir, args.forecasts_dir, args.output_dir)


if __name__ == "__main__":
    main()
