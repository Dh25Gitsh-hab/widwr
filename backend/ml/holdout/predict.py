"""
February 2026 holdout - STAGE 1: produce SEALED predictions.

Nothing is fitted, tuned or selected using February labels. This stage never
evaluates: it writes predictions WITHOUT any actual-generation column and
records their SHA-256, so stage 2 can prove they were not changed after the
labels were read.

Steps
  1. Freeze    record SHA-256 of the existing models and results (must stay unchanged)
  2. Models    load the saved Random Forests as they are; rebuild the simple models
               that were never saved (linear_by_level, logistic_by_level, base_rate)
               from the unchanged Oct-Dec training rows and VERIFY that every model
               reproduces its saved January predictions exactly - so these are
               provably the same models that were evaluated in January
  3. Data      validate the January + February files (structure only, no errors computed)
  4. Features  delivery days 2 Feb - 1 Mar, 09:50 UK cutoff, same feature_builder
  5. Audit     for EVERY delivery day: the forecast used was published by the cutoff,
               and overwriting every forecast and actual published after the cutoff
               leaves that day's features unchanged (so actuals can only be targets)
  6. Seal      predict; write predictions_sealed.csv (no labels) and a manifest with
               all hashes and the analysis plan, fixed before any label is read

Usage (from the backend folder)
    python -m ml.holdout.predict            # refuses to overwrite an existing seal
    python -m ml.holdout.predict --force    # deliberately re-seal
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.models.ml_model import (
    CLASSIFICATION_CANDIDATES,
    REGRESSION_CANDIDATES,
    ModelBundle,
    load_bundle,
    save_bundle,
)
from app.services.data_processor import clean_actuals, clean_forecasts
from app.services.feature_builder import FEATURE_COLUMNS, SHORTFALL_THRESHOLD_MW, build_feature_table
from ml.build_dataset import ML_DIR, PROCESSED_FILE, validate_actual_file, validate_combined, validate_forecast_file
from ml.holdout.common import (
    ACTUAL_FILES,
    FEATURES_FILE,
    FIRST_DECISION,
    FORECAST_FILES,
    FROZEN_FILES,
    GATE_CLOSURE,
    LAST_DECISION,
    MANIFEST_FILE,
    MODELS_DIR,
    PREDICTIONS_FILE,
    REPORTS_DIR,
    load_records,
    relative,
    sha256,
    write_json,
)

ANALYSIS_PLAN = {
    "eligible_observations": "delivery hours with a sealed prediction from every model AND an actual value - "
    "the identical set for every model",
    "regression": "MAE, RMSE, bias and sample count per model; daily MAE per model",
    "primary_comparison": "random_forest vs linear_by_level: paired difference in MAE",
    "secondary_comparisons": [
        "official_forecast vs random_forest (MAE, RMSE)",
        "official_forecast vs linear_by_level (MAE, RMSE)",
        "random_forest vs linear_by_level (RMSE)",
    ],
    "uncertainty": "95% circular moving-block bootstrap over whole days (3-day blocks primary, 1-day blocks as "
    "sensitivity), 5,000 resamples; Diebold-Mariano test with Newey-West variance, max lag 48 h",
    "interpretation_rule": "a difference is called significant only if the primary 95% interval excludes zero; "
    "otherwise the comparison is reported as inconclusive",
    "classification": "evaluated separately from MAE: ROC-AUC, PR-AUC (average precision), Brier score with "
    "3-day-block intervals; precision / recall / F1 at probability 0.5; calibration in five probability bands",
    "shortfall_label": f"official forecast - actual >= {SHORTFALL_THRESHOLD_MW:,.0f} MW",
}


# ---- 1-2. Freeze and load the models ------------------------------------------------


def frozen_hashes() -> dict[str, str]:
    return {relative(path): sha256(path) for path in FROZEN_FILES}


def load_models() -> tuple[dict, dict, dict]:
    """Return (regressors, classifiers, provenance). Every model is checked against its January predictions."""
    dataset = pd.read_csv(PROCESSED_FILE)
    train_val = dataset.loc[dataset["split"].isin(["train", "validation"])]
    test = dataset.loc[dataset["split"] == "test"].reset_index(drop=True)
    saved = pd.read_csv(ML_DIR / "predictions" / "test_predictions.csv")
    if not (test["startTime"].to_numpy() == saved["startTime"].to_numpy()).all():
        sys.exit("January rows of ml_dataset.csv and test_predictions.csv are not aligned - aborting")

    rf_regressor = load_bundle(ML_DIR / "models" / "error_regressor.joblib")
    rf_classifier = load_bundle(ML_DIR / "models" / "shortfall_classifier.joblib")

    def rebuild(candidates, name, target) -> ModelBundle:
        candidate = next(c for c in candidates if c.name == name)
        estimator = candidate.build().fit(train_val[FEATURE_COLUMNS], target)
        task = "regression" if candidates is REGRESSION_CANDIDATES else "classification"
        return ModelBundle(task, name, candidate.description, estimator, FEATURE_COLUMNS,
                           rf_regressor.trained_from, rf_regressor.trained_to)  # fmt: skip

    linear = rebuild(REGRESSION_CANDIDATES, "linear_by_level", train_val["error_mw"])
    logistic = rebuild(CLASSIFICATION_CANDIDATES, "logistic_by_level", train_val["shortfall_1gw"].astype(int))
    base_rate = rebuild(CLASSIFICATION_CANDIDATES, "base_rate", train_val["shortfall_1gw"].astype(int))

    regressors = {"random_forest": rf_regressor, "linear_by_level": linear}
    classifiers = {"random_forest": rf_classifier, "logistic_by_level": logistic, "base_rate": base_rate}

    provenance = {}
    for name, bundle in regressors.items():
        rebuilt = (test["forecast_mw"] - bundle.predict_error(test)).round(1)
        provenance[f"regression/{name}"] = _reproduction(rebuilt, saved[f"corrected_{name}"], 0.05)
    for name, bundle in classifiers.items():
        rebuilt = np.round(bundle.predict_probability(test), 4)
        provenance[f"classification/{name}"] = _reproduction(rebuilt, saved[f"probability_{name}"], 5e-5)

    for key, check in provenance.items():
        if not check["reproduces_saved_january_predictions"]:
            sys.exit(f"{key} does not reproduce its saved January predictions ({check}) - aborting")

    for name, bundle in [("linear_by_level", linear), ("logistic_by_level", logistic), ("base_rate", base_rate)]:
        path = MODELS_DIR / f"{bundle.task}_{name}.joblib"
        save_bundle(bundle, path)
        provenance[f"{bundle.task}/{name}"].update(
            source="rebuilt from the unchanged Oct-Dec training rows of ml_dataset.csv (never saved originally)",
            artifact=relative(path),
        )  # fmt: skip
    provenance["regression/random_forest"]["source"] = "saved artifact, loaded unchanged"
    provenance["regression/random_forest"]["artifact"] = "backend/ml/models/error_regressor.joblib"
    provenance["classification/random_forest"]["source"] = "saved artifact, loaded unchanged"
    provenance["classification/random_forest"]["artifact"] = "backend/ml/models/shortfall_classifier.joblib"
    for check in provenance.values():
        check["sha256"] = sha256(ML_DIR.parent.parent / check["artifact"])
    return regressors, classifiers, provenance


def _reproduction(rebuilt, saved, tolerance: float) -> dict:
    difference = float(np.max(np.abs(np.asarray(rebuilt, dtype=float) - saved.to_numpy(dtype=float))))
    return {"max_abs_difference_vs_saved_january": difference, "reproduces_saved_january_predictions": difference <= tolerance}


# ---- 3. Data validation --------------------------------------------------------------


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    issues: dict = {"errors": [], "warnings": []}
    files = []
    for path in ACTUAL_FILES:
        files.append(validate_actual_file(path.name, json.loads(path.read_text(encoding="utf-8")), issues))
    for path in FORECAST_FILES:
        files.append(validate_forecast_file(path.name, json.loads(path.read_text(encoding="utf-8")), issues))
    raw_actuals, raw_forecasts = load_records(ACTUAL_FILES), load_records(FORECAST_FILES)
    report = {
        "files": files,
        "input_sha256": {relative(p): sha256(p) for p in ACTUAL_FILES + FORECAST_FILES},
        "combined": validate_combined(raw_actuals, raw_forecasts, issues),
        "errors": issues["errors"],
        "warnings": issues["warnings"],
        "note": "Structural checks only - no forecast error is computed at this stage.",
    }
    write_json(REPORTS_DIR / "data_validation.json", report)
    if issues["errors"]:
        sys.exit(f"Validation failed: {issues['errors']}")
    return raw_actuals, raw_forecasts, report


# ---- 5. Cutoff audit -------------------------------------------------------------------


def audit_cutoffs(forecasts: pd.DataFrame, actuals: pd.DataFrame, table: pd.DataFrame) -> dict:
    late = table["forecast_published"] > table["cutoff"]
    days = []
    for day, rows in table.groupby("delivery_date", sort=True):
        cutoff = rows["cutoff"].iloc[0]
        tampered_forecasts = forecasts.copy()
        tampered_forecasts.loc[tampered_forecasts["publishTime"] > cutoff, "generation"] = 99_999.0
        tampered_actuals = actuals.copy()
        tampered_actuals.loc[tampered_actuals["publishTime"] > cutoff, "generation"] = -99_999.0
        rebuilt = build_feature_table(tampered_forecasts, tampered_actuals, day, day, GATE_CLOSURE).set_index("startTime")
        original = rows.set_index("startTime")[FEATURE_COLUMNS].to_numpy(dtype=float)
        unchanged = np.allclose(original, rebuilt.loc[rows["startTime"], FEATURE_COLUMNS].to_numpy(dtype=float),
                                rtol=0, atol=1e-6, equal_nan=True)  # fmt: skip
        runs = sorted({str(t) for t in rows["forecast_published"].dropna()})
        days.append({
            "delivery_date": str(day),
            "cutoff_uk": str(cutoff.tz_convert("Europe/London")),
            "cutoff_utc": str(cutoff),
            "forecast_runs_used": runs,
            "hours": len(rows),
            "forecast_published_by_cutoff": bool((rows["forecast_published"] <= rows["cutoff"]).all()),
            "features_unchanged_when_post_cutoff_data_overwritten": bool(unchanged),
        })  # fmt: skip
    return {
        "rule": f"forecast used = latest published at or before {GATE_CLOSURE:%H:%M} Europe/London on D-1",
        "forecasts_published_after_cutoff_used": int(late.sum()),
        "all_days_pass": all(d["forecast_published_by_cutoff"] and d["features_unchanged_when_post_cutoff_data_overwritten"] for d in days),
        "days": days,
    }


# ---- Main --------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal February 2026 holdout predictions")
    parser.add_argument("--force", action="store_true", help="overwrite an existing seal")
    args = parser.parse_args()
    if PREDICTIONS_FILE.exists() and not args.force:
        sys.exit(f"{relative(PREDICTIONS_FILE)} already exists - predictions are sealed. Use --force to re-seal.")

    frozen = frozen_hashes()
    print("1. Existing results fingerprinted:", len(frozen), "files")

    regressors, classifiers, provenance = load_models()
    print("2. Models verified - each reproduces its saved January predictions exactly:")
    for key, check in provenance.items():
        print(f"     {key:<32} max diff {check['max_abs_difference_vs_saved_january']:.4g}  ({check['source'].split(' (')[0]})")

    raw_actuals, raw_forecasts, validation = validate_inputs()
    for warning in validation["warnings"]:
        print("   WARNING", warning)
    print("3. Input files validated (structure only)")

    actuals, _ = clean_actuals(raw_actuals)
    forecasts, _ = clean_forecasts(raw_forecasts)
    first_day, last_day = FIRST_DECISION + timedelta(days=1), LAST_DECISION + timedelta(days=1)
    table = build_feature_table(forecasts, actuals, first_day, last_day, GATE_CLOSURE)
    table.insert(1, "decision_date", table["delivery_date"] - timedelta(days=1))
    without_forecast = table["forecast_mw"].isna()
    table = table.loc[~without_forecast].reset_index(drop=True)
    print(f"4. Features built: {len(table)} delivery hours {first_day}..{last_day} "
          f"({int(without_forecast.sum())} hours without a decision-time forecast)")  # fmt: skip

    audit = audit_cutoffs(forecasts, actuals, table)
    write_json(REPORTS_DIR / "cutoff_audit.json", audit)
    if not audit["all_days_pass"]:
        sys.exit("Cutoff audit FAILED - see reports/cutoff_audit.json")
    print(f"5. Cutoff audit passed for all {len(audit['days'])} delivery days")

    # 6. Predict and seal - label columns are dropped before anything is written.
    ids = table[["delivery_date", "decision_date", "startTime", "cutoff", "forecast_published"]]
    predictions = ids.assign(forecast_mw=table["forecast_mw"], corrected_official_forecast=table["forecast_mw"])
    for name, bundle in regressors.items():
        predictions[f"corrected_{name}"] = (table["forecast_mw"] - bundle.predict_error(table)).round(3)
    for name, bundle in classifiers.items():
        predictions[f"probability_{name}"] = np.round(bundle.predict_probability(table), 6)

    _write_csv(predictions, PREDICTIONS_FILE)
    _write_csv(table[list(ids.columns) + FEATURE_COLUMNS], FEATURES_FILE)
    manifest = {
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": "Predictions were produced from frozen models and cutoff-audited features. No February "
        "label was used to fit, tune or select any model, and no label column is present in the sealed file.",
        "holdout": {"decision_dates": [str(FIRST_DECISION), str(LAST_DECISION)],
                    "delivery_dates": [str(first_day), str(last_day)],
                    "excluded_by_design": "delivery 1 Feb 2026 (decided 31 Jan, part of the January test set)"},  # fmt: skip
        "rows": len(predictions),
        "predictions_file": relative(PREDICTIONS_FILE),
        "predictions_sha256": sha256(PREDICTIONS_FILE),
        "features_sha256": sha256(FEATURES_FILE),
        "models": provenance,
        "frozen_existing_results_sha256": frozen,
        "input_files_sha256": validation["input_sha256"],
        "analysis_plan": ANALYSIS_PLAN,
    }
    write_json(MANIFEST_FILE, manifest)
    if frozen_hashes() != frozen:
        sys.exit("An existing result file changed during this run - aborting")
    print(f"6. Sealed {len(predictions)} predictions -> {relative(PREDICTIONS_FILE)}")
    print(f"   sha256 {manifest['predictions_sha256'][:16]}...  (manifest: {relative(MANIFEST_FILE)})")
    print("   Next: python -m ml.holdout.evaluate")


def _write_csv(frame: pd.DataFrame, path) -> None:
    out = frame.copy()
    for column in ("startTime", "cutoff", "forecast_published"):
        out[column] = out[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
