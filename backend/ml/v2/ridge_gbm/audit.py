"""
Phase 1 (development data only): why can a forecast-size-only linear correction over-correct?

Reads only deliveries 2 Oct - 31 Dec 2025; no January row is read. Writes reports/audit.json.

  1. the production formula, and what least squares targets (the conditional MEAN error)
  2. mean vs median error by forecast size: with right-skewed errors, a mean-targeting line
     subtracts more than the typical hour's error
  3. the straight line vs the binned in-sample error (is a line the right shape?)
  4. stability: the same line refitted on each development month
  5. the production specification out-of-sample (8 weekly folds): over-correction by size
  6. how much of the remaining error is a whole-day offset that the forecast size cannot see

Usage (from backend/):  python -m ml.v2.ridge_gbm.audit
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LinearRegression

from app.models.ml_model import REGRESSION_CANDIDATES, load_bundle
from ml.v2.common import write_json
from ml.v2.ridge_gbm.config import (
    ALL,
    BAND_LABELS,
    FOLDS,
    HIGH_FORECAST_MW,
    PRODUCTION_LINEAR,
    REPORTS_DIR,
    bands,
    fold_masks,
    load_development,
)

LINEAR_SPEC = next(c for c in REGRESSION_CANDIDATES if c.name == "linear_by_level")


def band_table(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    """Corrected error = corrected - actual = error_mw - predicted error; negative = over-corrected."""
    predicted = pd.Series(predicted, index=frame.index)
    corrected_error = frame["error_mw"] - predicted
    groups = {label: bands(frame["forecast_mw"]) == label for label in BAND_LABELS}
    groups[f">={HIGH_FORECAST_MW / 1000:.0f} GW (high)"] = frame["forecast_mw"] >= HIGH_FORECAST_MW
    table = {}
    for label, mask in groups.items():
        if not mask.any():
            continue
        error, corrected = frame.loc[mask, "error_mw"], corrected_error[mask]
        table[label] = {
            "hours": int(mask.sum()),
            "days": int(frame.loc[mask, "delivery_date"].nunique()),
            "official_mean_error": round(float(error.mean()), 1),
            "official_median_error": round(float(error.median()), 1),
            "official_error_skew": round(float(skew(error)), 2),
            "mean_predicted_error": round(float(predicted[mask].mean()), 1),
            "corrected_mean_error": round(float(corrected.mean()), 1),
            "corrected_median_error": round(float(corrected.median()), 1),
            "over_corrected_hours_pct": round(float((corrected < 0).mean() * 100), 1),
            "official_mae": round(float(error.abs().mean()), 1),
            "corrected_mae": round(float(corrected.abs().mean()), 1),
        }
    return table


def day_structure(frame: pd.DataFrame, corrected_error: pd.Series) -> dict:
    """Share of the error variance that is a whole-day offset, and how persistent the daily offset is."""
    daily_mean = corrected_error.groupby(frame["delivery_date"]).transform("mean")
    per_day = corrected_error.groupby(frame["delivery_date"]).mean().sort_index()
    consecutive = pd.Series(per_day.to_numpy(), index=pd.to_datetime(per_day.index)).asfreq("D")
    return {
        "day_offset_share_of_variance": round(float(daily_mean.var() / corrected_error.var()), 3),
        "daily_mean_autocorrelation_lag1": round(float(consecutive.autocorr(1)), 3),
        "daily_mean_autocorrelation_lag2": round(float(consecutive.autocorr(2)), 3),
        "note": "lag 2 is the most recent fully known day at a decision cutoff (labels of D-1 are incomplete at 09:50 on D-1)",
    }


def main() -> None:
    dev = load_development()
    production = load_bundle(PRODUCTION_LINEAR)
    lr = production.estimator.steps[-1][1]
    slope, intercept = float(lr.coef_[0]), float(lr.intercept_)
    in_sample = production.predict_error(dev)

    monthly = {}
    for month, rows in dev.groupby(dev["delivery_date"].map(lambda d: f"{d:%Y-%m}")):
        fit = LinearRegression().fit(rows[["forecast_mw"]], rows["error_mw"])
        monthly[month] = {
            "hours": len(rows),
            "slope": round(float(fit.coef_[0]), 4),
            "intercept": round(float(fit.intercept_), 1),
            "implied_correction_mw": {f"{gw} GW": round(float(fit.intercept_ + fit.coef_[0] * gw * 1000), 0) for gw in (5, 10, 15, 17.4, 20)},
        }

    # The production specification out-of-sample: refit on each fold's training days, predict its week.
    oof = np.full(len(dev), np.nan)
    for fold in FOLDS:
        train, valid = fold_masks(dev["delivery_date"], fold)
        model = LINEAR_SPEC.build().fit(dev.loc[train, ALL], dev.loc[train, "error_mw"])
        oof[valid.to_numpy()] = model.predict(dev.loc[valid, ALL])
    in_cv = ~np.isnan(oof)
    cv = dev.loc[in_cv]
    cv_corrected_error = cv["error_mw"] - oof[in_cv]
    slope_of_residual = LinearRegression().fit(cv[["forecast_mw"]], cv_corrected_error)

    report = {
        "scope": "development deliveries 2025-10-02 .. 2025-12-31 only; no January row read",
        "production_formula": {
            "predicted_error": f"{slope:.6f} * forecast_mw + ({intercept:.3f})",
            "corrected": f"{1 - slope:.6f} * forecast_mw + {-intercept:.3f}",
            "inputs": production.inputs,
            "fitted_by": "ordinary least squares, which estimates the conditional MEAN error for each forecast size",
            "trained_on_deliveries": [str(production.trained_from), str(production.trained_to)],
        },
        "in_sample_by_forecast_band": band_table(dev, in_sample),
        "monthly_refits": monthly,
        "out_of_sample_production_spec": {
            "folds": len(FOLDS),
            "hours": int(in_cv.sum()),
            "mae": round(float(np.abs(cv_corrected_error).mean()), 1),
            "official_mae": round(float(cv["error_mw"].abs().mean()), 1),
            "bias": round(float(cv_corrected_error.mean()), 1),
            "by_forecast_band": band_table(cv, oof[in_cv]),
            "corrected_error_vs_forecast_slope": round(float(slope_of_residual.coef_[0]), 4),
            "day_structure": day_structure(cv, cv_corrected_error),
        },
        "in_sample_day_structure": day_structure(dev, dev["error_mw"] - in_sample),
    }
    write_json(REPORTS_DIR / "audit.json", report)

    print(f"Production: predicted error = {report['production_formula']['predicted_error']}")
    print("\nIn-sample by forecast band (GW): mean vs median official error, and the line's effect")
    for label, row in report["in_sample_by_forecast_band"].items():
        print(f"  {label:<14} n={row['hours']:>4}  mean err {row['official_mean_error']:>7.0f}  median {row['official_median_error']:>7.0f}  "
              f"skew {row['official_error_skew']:>5.2f}  predicted {row['mean_predicted_error']:>7.0f}  "
              f"corrected median {row['corrected_median_error']:>7.0f}  over-corrected {row['over_corrected_hours_pct']:>5.1f}% of hours")
    print("\nMonthly refits (implied correction at 17.4 GW):")
    for month, row in monthly.items():
        print(f"  {month}: slope {row['slope']:.3f}, intercept {row['intercept']:.0f} -> {row['implied_correction_mw']['17.4 GW']:.0f} MW")
    out = report["out_of_sample_production_spec"]
    print(f"\nOut-of-sample (8 weekly folds, {out['hours']} h): MAE {out['mae']} (official {out['official_mae']}), bias {out['bias']}, "
          f"residual slope vs forecast {out['corrected_error_vs_forecast_slope']}")
    for label, row in out["by_forecast_band"].items():
        print(f"  {label:<14} n={row['hours']:>4}  corrected mean err {row['corrected_mean_error']:>7.0f}  median {row['corrected_median_error']:>7.0f}  "
              f"over-corrected {row['over_corrected_hours_pct']:>5.1f}%  MAE {row['corrected_mae']:>6.0f} (official {row['official_mae']:.0f})")
    print(f"  day structure: {out['day_structure']}")
    print(f"In-sample day structure: {report['in_sample_day_structure']}")
    print(f"\nWrote {REPORTS_DIR / 'audit.json'}")


if __name__ == "__main__":
    main()
