"""
ML step 2 - train, compare and evaluate the models, then save everything.

Input
    ml/processed/ml_dataset.csv            (from: python -m ml.build_dataset)

Outputs
    ml/models/error_regressor.joblib       selected regression model, refit on train+validation
    ml/models/shortfall_classifier.joblib  selected classification model, refit on train+validation
    ml/reports/metrics.json                every candidate's validation and test scores,
                                           confidence interval, calibration, feature importance
    ml/predictions/test_predictions.csv    hour-by-hour January predictions of every candidate

Usage (from the backend folder)
    python -m ml.train

PROTOCOL - chronological, never shuffled
    1. fit every candidate on TRAIN       (decisions Oct-Nov 2025)
    2. score on VALIDATION                (decisions Dec 2025) and select the best:
           regression     -> lowest MAE of the corrected forecast
           classification -> lowest Brier score
       A baseline is allowed to win - then the baseline is what we ship.
    3. refit every candidate on TRAIN + VALIDATION (more, and more recent, data)
    4. score ONCE on TEST                 (decisions Jan 2026 - the brief's scope)
       All candidates are reported on test for transparency; the choice was
       already made in step 2, so the test set never influences selection.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from app.models.ml_model import (
    CLASSIFICATION_CANDIDATES,
    PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    REGRESSION_CANDIDATES,
    ModelBundle,
    calibration_table,
    classification_scores,
    day_block_bootstrap,
    regression_scores,
    save_bundle,
)
from app.services.feature_builder import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, SHORTFALL_THRESHOLD_MW
from ml.build_dataset import ML_DIR, PROCESSED_FILE, REPORTS_DIR, SPLITS

MODELS_DIR = ML_DIR / "models"
PREDICTIONS_FILE = ML_DIR / "predictions" / "test_predictions.csv"
REGRESSOR_FILE = MODELS_DIR / "error_regressor.joblib"
CLASSIFIER_FILE = MODELS_DIR / "shortfall_classifier.joblib"


def load_dataset() -> pd.DataFrame:
    if not PROCESSED_FILE.is_file():
        raise SystemExit("ml_dataset.csv not found - run `python -m ml.build_dataset` first")
    df = pd.read_csv(PROCESSED_FILE)
    for column in ("startTime", "cutoff", "forecast_published"):
        df[column] = pd.to_datetime(df[column], utc=True)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"]).dt.date
    return df


# ---- Regression: correct the official forecast ------------------------------------


def run_regression(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict, ModelBundle]:
    train_val = pd.concat([train, validation])
    results, test_predictions, fitted = [], {}, {}

    for candidate in REGRESSION_CANDIDATES:
        # Step 1-2: fit on train, score on validation.
        model = candidate.build().fit(train[FEATURE_COLUMNS], train["error_mw"])
        val_corrected = validation["forecast_mw"] - model.predict(validation[FEATURE_COLUMNS])
        # Step 3-4: refit on train+validation, score on test.
        final = candidate.build().fit(train_val[FEATURE_COLUMNS], train_val["error_mw"])
        test_corrected = test["forecast_mw"] - final.predict(test[FEATURE_COLUMNS])

        fitted[candidate.name] = final
        test_predictions[f"corrected_{candidate.name}"] = test_corrected.round(1)
        results.append({
            "name": candidate.name,
            "description": candidate.description,
            "validation": regression_scores(validation["actual_mw"], val_corrected),
            "test": regression_scores(test["actual_mw"], test_corrected),
        })  # fmt: skip

    official_test_mae = next(r for r in results if r["name"] == "official_forecast")["test"]["mae"]
    official_val_mae = next(r for r in results if r["name"] == "official_forecast")["validation"]["mae"]
    for result in results:
        result["validation_mae_change_pct"] = _pct_change(result["validation"]["mae"], official_val_mae)
        result["test_mae_change_pct"] = _pct_change(result["test"]["mae"], official_test_mae)

    selected = min(results, key=lambda r: r["validation"]["mae"])
    candidate = next(c for c in REGRESSION_CANDIDATES if c.name == selected["name"])
    selected_abs_error = (test_predictions[f"corrected_{candidate.name}"] - test["actual_mw"]).abs()
    report = {
        "target": "error_mw = forecast_mw - actual_mw; corrected forecast = forecast_mw - predicted error",
        "selection_rule": "lowest validation MAE of the corrected forecast",
        "selected": selected["name"],
        "candidates": results,
        "test_improvement_vs_official": day_block_bootstrap(
            test["delivery_date"], test["error_mw"].abs(), selected_abs_error
        ),
        "feature_importance": _importance(fitted[candidate.name], test, test["error_mw"], "neg_mean_absolute_error"),
    }
    bundle = ModelBundle(
        task="regression",
        name=candidate.name,
        description=candidate.description,
        estimator=fitted[candidate.name],
        features=FEATURE_COLUMNS,
        trained_from=train_val["delivery_date"].min(),
        trained_to=train_val["delivery_date"].max(),
    )
    return report, test_predictions, bundle


# ---- Classification: probability of a >= 1 GW shortfall ---------------------------


def run_classification(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict, ModelBundle]:
    train_val = pd.concat([train, validation])
    results, test_predictions, fitted = [], {}, {}

    for candidate in CLASSIFICATION_CANDIDATES:
        model = candidate.build().fit(train[FEATURE_COLUMNS], train["shortfall_1gw"].astype(int))
        val_probability = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        final = candidate.build().fit(train_val[FEATURE_COLUMNS], train_val["shortfall_1gw"].astype(int))
        test_probability = final.predict_proba(test[FEATURE_COLUMNS])[:, 1]

        fitted[candidate.name] = (final, test_probability)
        test_predictions[f"probability_{candidate.name}"] = np.round(test_probability, 4)
        results.append({
            "name": candidate.name,
            "description": candidate.description,
            "validation": classification_scores(validation["shortfall_1gw"], val_probability),
            "test": classification_scores(test["shortfall_1gw"], test_probability),
        })  # fmt: skip

    selected = min(results, key=lambda r: r["validation"]["brier"])
    candidate = next(c for c in CLASSIFICATION_CANDIDATES if c.name == selected["name"])
    final, test_probability = fitted[candidate.name]
    always_yes = classification_scores(test["shortfall_1gw"], np.ones(len(test)))
    report = {
        "target": f"shortfall_1gw = 1 if forecast_mw - actual_mw >= {SHORTFALL_THRESHOLD_MW:,.0f} MW",
        "selection_rule": "lowest validation Brier score",
        "probability_threshold": PROBABILITY_THRESHOLD,
        "selected": selected["name"],
        "candidates": results,
        "reference_always_predict_shortfall_test": {
            key: always_yes[key] for key in ("accuracy", "precision", "recall", "f1")
        },
        "test_calibration": calibration_table(test["shortfall_1gw"], test_probability),
        "feature_importance": _importance(final, test, test["shortfall_1gw"].astype(int), "neg_brier_score"),
    }
    bundle = ModelBundle(
        task="classification",
        name=candidate.name,
        description=candidate.description,
        estimator=final,
        features=FEATURE_COLUMNS,
        trained_from=train_val["delivery_date"].min(),
        trained_to=train_val["delivery_date"].max(),
    )
    return report, test_predictions, bundle


# ---- Helpers ------------------------------------------------------------------------


def _importance(model, test: pd.DataFrame, target: pd.Series, scoring: str) -> list[dict]:
    """
    Permutation importance on the test set: shuffle one feature at a time and
    measure how much the score gets worse. Model-agnostic and honest - it shows
    what the model actually relies on, on data it has not seen.
    """
    result = permutation_importance(
        model, test[FEATURE_COLUMNS], target, scoring=scoring, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    rows = [
        {"feature": name, "importance": round(float(mean), 4), "std": round(float(std), 4)}
        for name, mean, std in zip(FEATURE_COLUMNS, result.importances_mean, result.importances_std, strict=True)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def _pct_change(value: float, reference: float) -> float:
    return round(100 * (value - reference) / reference, 1)


def _limitations(dataset: pd.DataFrame) -> list[str]:
    rates = dataset.groupby("split")["shortfall_1gw"].mean()
    return [
        "Small sample: 61 training days for model selection, 91 days (Oct-Dec 2025) for the final fit, "
        "31 test days. Hours within a day share weather and forecast run, so the effective sample is "
        "closer to the number of days - see the day-block bootstrap interval.",
        "One season only (autumn/winter). Nothing here supports using the model in spring or summer "
        "without retraining on those months.",
        f"Distribution shift: the shortfall rate was {rates.get('train', float('nan')):.0%} in training, "
        f"{rates.get('validation', float('nan')):.0%} in validation and {rates.get('test', float('nan')):.0%} in the January test set.",
        "1 Jan 2026 actuals are missing from the provided files: that delivery day is excluded, and for the "
        "2 Jan decision the 'latest actual' and 'recent error' features come from 31 Dec (older than usual).",
        "1 Oct 2025 is excluded: no forecast published before its cutoff (September runs not provided).",
        "The likely driver of the bias (wind curtailment) is not in the data. If curtailment patterns "
        "change, the correction must be retrained.",
        "Forecast accuracy is not trading profit. No prices, imbalance costs or volumes are in scope, so "
        "no claim about trading performance or profitability is made.",
    ]


def main() -> None:
    dataset = load_dataset()
    train, validation, test = (dataset.loc[dataset["split"] == name] for name, _, _ in SPLITS)
    print(f"Training on {len(train)} h, validating on {len(validation)} h, testing on {len(test)} h")

    regression, reg_predictions, regressor = run_regression(train, validation, test)
    classification, clf_predictions, classifier = run_classification(train, validation, test)

    save_bundle(regressor, REGRESSOR_FILE)
    save_bundle(classifier, CLASSIFIER_FILE)

    predictions = test[["delivery_date", "startTime", "forecast_mw", "actual_mw", "error_mw", "shortfall_1gw"]].copy()
    predictions["startTime"] = predictions["startTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    predictions = predictions.assign(**reg_predictions, **clf_predictions)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "decision_rule": "Latest forecast published by 09:50 UK local time on D-1 (DST-aware); "
        "every feature uses only data published by that cutoff",
        "shortfall_threshold_mw": SHORTFALL_THRESHOLD_MW,
        "splits": [
            {
                "name": name,
                "decision_dates": [str(first), str(last)],
                "delivery_dates": [str(part["delivery_date"].min()), str(part["delivery_date"].max())],
                "days": int(part["delivery_date"].nunique()),
                "hours": len(part),
            }
            for name, first, last in SPLITS
            for part in [dataset.loc[dataset["split"] == name]]
        ],
        "features": [{"name": name, "description": text} for name, text in FEATURE_DESCRIPTIONS.items()],
        "regression": regression,
        "classification": classification,
        "limitations": _limitations(dataset),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    _print_summary(regression, classification)


def _print_summary(regression: dict, classification: dict) -> None:
    print("\nREGRESSION - corrected forecast vs actual (MW)        validation MAE   test MAE  test RMSE  test bias  vs official")
    for r in regression["candidates"]:
        mark = "*" if r["name"] == regression["selected"] else " "
        print(f" {mark} {r['name']:<20}{r['description'][:30]:<32}{r['validation']['mae']:>10,.0f}{r['test']['mae']:>11,.0f}"
              f"{r['test']['rmse']:>11,.0f}{r['test']['bias']:>11,.0f}{r['test_mae_change_pct']:>+11.1f}%")  # fmt: skip
    ci = regression["test_improvement_vs_official"]
    print(f"   selected model cuts test MAE by {ci['mae_reduction_mw']:,.0f} MW "
          f"(95% CI {ci['ci_low_mw']:,.0f} to {ci['ci_high_mw']:,.0f} MW, {ci['method']})")  # fmt: skip

    print("\nCLASSIFICATION - P(shortfall >= 1 GW)      val Brier  test Brier  ROC-AUC  PR-AUC  accuracy  precision  recall    F1")
    for r in classification["candidates"]:
        mark = "*" if r["name"] == classification["selected"] else " "
        t = r["test"]
        print(f" {mark} {r['name']:<20}{'':<18}{r['validation']['brier']:>9.3f}{t['brier']:>11.3f}"
              f"{(t['roc_auc'] or 0):>9.3f}{(t['pr_auc'] or 0):>8.3f}{t['accuracy']:>10.3f}{t['precision']:>11.3f}"
              f"{t['recall']:>8.3f}{t['f1']:>6.3f}")  # fmt: skip
    ref = classification["reference_always_predict_shortfall_test"]
    print(f"   reference 'always predict shortfall' on test: accuracy {ref['accuracy']:.3f}, F1 {ref['f1']:.3f}")
    print("\n* = selected on validation. Saved models, metrics and predictions under backend/ml/.")


if __name__ == "__main__":
    main()
