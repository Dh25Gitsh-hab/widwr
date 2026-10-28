"""
February 2026 holdout - STAGE 2: evaluate the sealed predictions.

Steps
  1. Verify  the sealed predictions, the holdout models and the existing results
             are byte-identical to what the manifest recorded (else abort)
  2. Labels  read February actual generation - for the first time - and join it to
             the sealed predictions on the delivery hour
  3. Sample  eligible = hours with an actual AND a prediction from every model:
             the identical set for every model; exclusions are listed
  4. Score   corrected forecast: MAE / RMSE / bias / n per model, and daily MAE
  5. Compare paired loss differences with day-block bootstrap intervals
             (3-day blocks primary, 1-day sensitivity) and Diebold-Mariano (HAC) tests
  6. Risk    shortfall probability, separately from MAE: ROC-AUC, PR-AUC, Brier
             (with intervals), precision / recall / F1, calibration
  7. Write   reports/holdout_metrics.json, reports/daily_results.csv,
             evaluated_predictions.csv (the sealed file itself is never modified)

Usage (from the backend folder):  python -m ml.holdout.evaluate
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from app.models.ml_model import calibration_table, classification_scores, regression_scores
from app.services.data_processor import clean_actuals
from app.services.feature_builder import SHORTFALL_THRESHOLD_MW
from ml.holdout.common import (
    CLASSIFICATION_MODELS,
    LABEL_FILE,
    MANIFEST_FILE,
    OUT_DIR,
    PREDICTIONS_FILE,
    REGRESSION_MODELS,
    REPORTS_DIR,
    load_records,
    relative,
    sha256,
    write_json,
)
from ml.holdout.stats import block_bootstrap, diebold_mariano, interval

PROJECT_ROOT = OUT_DIR.parent.parent.parent
PAIRS = [("random_forest", "linear_by_level"), ("official_forecast", "random_forest"), ("official_forecast", "linear_by_level")]
BLOCKS = {"primary_3_day_blocks": 3, "sensitivity_1_day_blocks": 1}


# ---- 1. Verify the seal -----------------------------------------------------------------


def verify_seal() -> dict:
    if not MANIFEST_FILE.is_file():
        sys.exit("No sealed predictions - run `python -m ml.holdout.predict` first")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    checks = {manifest["predictions_file"]: manifest["predictions_sha256"], **manifest["frozen_existing_results_sha256"]}
    checks.update({m["artifact"]: m["sha256"] for m in manifest["models"].values()})
    changed = [path for path, digest in checks.items() if sha256(PROJECT_ROOT / path) != digest]
    if changed:
        sys.exit(f"Seal broken - these files changed after sealing: {changed}")
    return manifest


# ---- 2-3. Labels and the eligible sample ------------------------------------------------


def load_eligible() -> tuple[pd.DataFrame, dict]:
    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions["startTime"] = pd.to_datetime(predictions["startTime"], utc=True)
    actuals, cleaning = clean_actuals(load_records([LABEL_FILE]))  # labels are read here, after sealing
    predictions["actual_mw"] = predictions["startTime"].map(actuals.set_index("startTime")["generation"])

    model_columns = [f"corrected_{m}" for m in REGRESSION_MODELS] + [f"probability_{m}" for m in CLASSIFICATION_MODELS]
    has_actual = predictions["actual_mw"].notna()
    has_all_predictions = predictions[model_columns].notna().all(axis=1)
    eligible = predictions.loc[has_actual & has_all_predictions].sort_values("startTime").reset_index(drop=True)
    eligible["shortfall"] = ((eligible["forecast_mw"] - eligible["actual_mw"]) >= SHORTFALL_THRESHOLD_MW).astype(int)
    for model in REGRESSION_MODELS:
        eligible[f"error_{model}"] = eligible[f"corrected_{model}"] - eligible["actual_mw"]

    excluded = predictions.loc[~(has_actual & has_all_predictions)]
    sample = {
        "sealed_predictions": len(predictions),
        "eligible_hours": len(eligible),
        "eligible_days": int(eligible["delivery_date"].nunique()),
        "delivery_dates": [eligible["delivery_date"].min(), eligible["delivery_date"].max()],
        "excluded_hours": len(excluded),
        "excluded_reasons": {
            "no_actual_generation": sorted(excluded.loc[excluded["actual_mw"].isna(), "delivery_date"].unique().tolist()),
            "missing_a_model_prediction": int((~has_all_predictions).sum()),
        },
        "label_cleaning": cleaning.as_dict(),
        "shortfall_rate": round(float(eligible["shortfall"].mean()), 4),
        "note": "Accuracy is scored per hour: WINDFOR forecasts are hourly and are compared with the actual "
        "for the same startTime (as in the January evaluation).",
    }
    return eligible, sample


# ---- 4-5. Corrected forecast ---------------------------------------------------------------


def regression_report(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    models = {}
    for model in REGRESSION_MODELS:
        models[model] = {"n": len(df), **regression_scores(df["actual_mw"], df[f"corrected_{model}"])}
        models[model]["mae_change_vs_official_pct"] = round(
            100 * (models[model]["mae"] / models["official_forecast"]["mae"] - 1), 1
        )

    comparisons = []
    for a, b in PAIRS:
        abs_a, abs_b = df[f"error_{a}"].abs().to_numpy(), df[f"error_{b}"].abs().to_numpy()
        sq_a, sq_b = df[f"error_{a}"].to_numpy() ** 2, df[f"error_{b}"].to_numpy() ** 2
        entry = {
            "a": a,
            "b": b,
            "reading": f"positive difference = {b} more accurate than {a}",
            "mae_difference_a_minus_b": round(float(abs_a.mean() - abs_b.mean()), 1),
            "rmse_difference_a_minus_b": round(float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), 1),
            "hours_b_better": int((abs_b < abs_a).sum()),
            "days_b_better": int((df.assign(d=abs_a - abs_b).groupby("delivery_date")["d"].mean() > 0).sum()),
            "intervals_95": {},
            "diebold_mariano": {
                "absolute_error": diebold_mariano(abs_a, abs_b),
                "squared_error": diebold_mariano(sq_a, sq_b),
            },
        }
        for label, block_days in BLOCKS.items():
            mae = block_bootstrap(df["delivery_date"], lambda r: abs_a[r].mean() - abs_b[r].mean(), block_days)
            rmse = block_bootstrap(df["delivery_date"], lambda r: np.sqrt(sq_a[r].mean()) - np.sqrt(sq_b[r].mean()), block_days)
            entry["intervals_95"][label] = {"mae_difference": interval(mae), "rmse_difference": interval(rmse)}
        low, high = entry["intervals_95"]["primary_3_day_blocks"]["mae_difference"]
        entry["mae_verdict"] = (
            f"{b} significantly more accurate" if low > 0
            else f"{a} significantly more accurate" if high < 0
            else "inconclusive - interval includes zero"
        )  # fmt: skip
        comparisons.append(entry)

    daily = df.groupby("delivery_date").agg(
        hours=("actual_mw", "size"),
        mean_actual_mw=("actual_mw", "mean"),
        mean_forecast_mw=("forecast_mw", "mean"),
        **{f"mae_{m}": (f"error_{m}", lambda e: e.abs().mean()) for m in REGRESSION_MODELS},
        shortfall_rate=("shortfall", "mean"),
        **{f"mean_probability_{m}": (f"probability_{m}", "mean") for m in CLASSIFICATION_MODELS},
    )
    daily["best_model"] = daily[[f"mae_{m}" for m in REGRESSION_MODELS]].idxmin(axis=1).str.removeprefix("mae_")
    daily = daily.round(3).reset_index()
    return {"models": models, "paired_comparisons": comparisons,
            "days_best": daily["best_model"].value_counts().to_dict()}, daily  # fmt: skip


# ---- 6. Shortfall risk ------------------------------------------------------------------------


def classification_report(df: pd.DataFrame) -> dict:
    y = df["shortfall"].to_numpy()
    probabilities = {m: df[f"probability_{m}"].to_numpy() for m in CLASSIFICATION_MODELS}

    def safe_auc(rows, p):
        return roc_auc_score(y[rows], p[rows]) if len(np.unique(y[rows])) == 2 else np.nan

    models = {}
    for model, p in probabilities.items():
        threshold = classification_scores(y, p)
        auc = block_bootstrap(df["delivery_date"], lambda r: safe_auc(r, p), 3, n_resamples=2000)
        brier = block_bootstrap(df["delivery_date"], lambda r: brier_score_loss(y[r], p[r]), 3, n_resamples=2000)
        calibration = calibration_table(y, p)
        models[model] = {
            "roc_auc": threshold["roc_auc"],
            "roc_auc_95": interval(auc[~np.isnan(auc)]),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "pr_auc_no_skill_baseline": round(float(y.mean()), 4),
            "brier": threshold["brier"],
            "brier_95": interval(brier),
            "brier_skill_vs_base_rate": None,
            "at_threshold_0_5": {k: threshold[k] for k in ("accuracy", "precision", "recall", "f1", "predicted_rate")},
            "calibration": calibration,
            "expected_calibration_error": round(
                sum(row["n"] * abs(row["mean_predicted"] - row["observed_rate"]) for row in calibration) / len(y), 4
            ),
        }
    for model in models:
        models[model]["brier_skill_vs_base_rate"] = round(1 - models[model]["brier"] / models["base_rate"]["brier"], 4)

    p_rf, p_log = probabilities["random_forest"], probabilities["logistic_by_level"]
    brier_diff = block_bootstrap(
        df["delivery_date"], lambda r: brier_score_loss(y[r], p_rf[r]) - brier_score_loss(y[r], p_log[r]), 3, n_resamples=2000
    )
    always = classification_scores(y, np.ones(len(y)))
    return {
        "label": f"official forecast - actual >= {SHORTFALL_THRESHOLD_MW:,.0f} MW",
        "observed_shortfall_rate": round(float(y.mean()), 4),
        "models": models,
        "random_forest_minus_logistic_brier": {
            "difference": round(float(brier_score_loss(y, p_rf) - brier_score_loss(y, p_log)), 4),
            "interval_95_3_day_blocks": interval(brier_diff),
            "reading": "negative = random forest has better (lower) Brier score",
        },
        "reference_always_predict_shortfall": {k: always[k] for k in ("accuracy", "precision", "recall", "f1")},
    }


# ---- Main -------------------------------------------------------------------------------------


def main() -> None:
    manifest = verify_seal()
    print("1. Seal verified: sealed predictions, holdout models and existing results unchanged")
    eligible, sample = load_eligible()
    print(f"2-3. Labels joined: {sample['eligible_hours']} eligible hours over {sample['eligible_days']} days "
          f"({sample['excluded_hours']} excluded: {sample['excluded_reasons']})")  # fmt: skip

    regression, daily = regression_report(eligible)
    classification = classification_report(eligible)

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sealed_at": manifest["sealed_at"],
        "predictions_sha256": manifest["predictions_sha256"],
        "analysis_plan": manifest["analysis_plan"],
        "sample": sample,
        "corrected_forecast": regression,
        "shortfall_risk": classification,
        "limitations": [
            f"{sample['eligible_days']} days is a short holdout; intervals account for day-to-day dependence but are wide.",
            "February is a winter month like the training data; it says nothing about spring or summer.",
            "Forecast accuracy is not trading profit. No market prices, imbalance costs, trading costs or volumes "
            "were used, so no profitability claim is made; that requires a separately validated trading backtest.",
        ],
    }
    write_json(REPORTS_DIR / "holdout_metrics.json", metrics)
    daily.to_csv(REPORTS_DIR / "daily_results.csv", index=False)
    evaluated = eligible.copy()
    evaluated["startTime"] = evaluated["startTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    evaluated.to_csv(OUT_DIR / "evaluated_predictions.csv", index=False)
    if sha256(PREDICTIONS_FILE) != manifest["predictions_sha256"]:
        sys.exit("Sealed predictions changed during evaluation - results invalid")
    print(f"7. Reports written to {relative(REPORTS_DIR)}")
    _print_summary(regression, classification, sample)


def _print_summary(regression: dict, classification: dict, sample: dict) -> None:
    print(f"\nCORRECTED FORECAST - {sample['eligible_hours']} hours, {sample['eligible_days']} days (MW)")
    print(f"  {'model':<20}{'n':>6}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'vs official':>13}")
    for name, m in regression["models"].items():
        print(f"  {name:<20}{m['n']:>6}{m['mae']:>9,.0f}{m['rmse']:>9,.0f}{m['bias']:>+9,.0f}{m['mae_change_vs_official_pct']:>+12.1f}%")
    print("\n  Paired MAE differences (a - b; positive = b better), 95% intervals:")
    for c in regression["paired_comparisons"]:
        ci3 = c["intervals_95"]["primary_3_day_blocks"]["mae_difference"]
        ci1 = c["intervals_95"]["sensitivity_1_day_blocks"]["mae_difference"]
        dm = c["diebold_mariano"]["absolute_error"]
        print(f"  {c['a']:>17} - {c['b']:<16}{c['mae_difference_a_minus_b']:>+8,.0f}  3-day [{ci3[0]:+,.0f}, {ci3[1]:+,.0f}]"
              f"  1-day [{ci1[0]:+,.0f}, {ci1[1]:+,.0f}]  DM p={dm['p_value']:.3f}  days b better {c['days_b_better']}/{sample['eligible_days']}"
              f"  -> {c['mae_verdict']}")  # fmt: skip
    print(f"  days best: {regression['days_best']}")
    print(f"\nSHORTFALL RISK - observed rate {classification['observed_shortfall_rate']:.0%}")
    print(f"  {'model':<20}{'ROC-AUC':>9}{'95% CI':>18}{'PR-AUC':>8}{'Brier':>8}{'95% CI':>18}{'ECE':>7}{'prec':>7}{'recall':>7}{'F1':>6}")
    for name, m in classification["models"].items():
        t = m["at_threshold_0_5"]
        auc = m["roc_auc"] if m["roc_auc"] is not None else float("nan")
        print(f"  {name:<20}{auc:>9.3f}  [{m['roc_auc_95'][0]:.3f}, {m['roc_auc_95'][1]:.3f}]{m['pr_auc']:>8.3f}"
              f"{m['brier']:>8.3f}  [{m['brier_95'][0]:.3f}, {m['brier_95'][1]:.3f}]{m['expected_calibration_error']:>7.3f}"
              f"{t['precision']:>7.3f}{t['recall']:>7.3f}{t['f1']:>6.3f}")  # fmt: skip
    d = classification["random_forest_minus_logistic_brier"]
    print(f"  Brier RF - logistic: {d['difference']:+.4f}, 95% CI {d['interval_95_3_day_blocks']}")
    print(f"  always-predict-shortfall: {classification['reference_always_predict_shortfall']}")


if __name__ == "__main__":
    main()
