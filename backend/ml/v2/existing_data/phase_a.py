"""
Phase A: verify the reported baselines, then analyse where the existing models go wrong.

    Development (deliveries Oct-Dec 2025): out-of-fold residuals of the PRODUCTION SPECIFICATIONS
        (linear correction, logistic regression, random forests), refitted for each of the eight
        weekly chronological folds the experiments use. These guide the experiments.
    January and February 2026: residuals of the production models' SAVED predictions
        (previously examined - descriptive context only, never used to choose).

Usage (from the backend folder):  python -m ml.v2.existing_data.phase_a
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from ml.v2.candidates import CANDIDATES as V1
from ml.v2.common import ML_DIR, write_json
from ml.v2.existing_data.config import FOLDS, PROCESSED_DIR, REPORTS_DIR, fold_label, fold_masks
from ml.v2.existing_data.metrics import classification_metrics, error_distribution, regression_metrics, segments
from ml.v2.features import WIND_FEATURES

PRODUCTION_SPECS = {"linear_by_level": "R1_linear_by_level", "random_forest_regressor": "R2_random_forest",
                    "logistic_by_level": "C1_logistic_by_level", "random_forest_classifier": "C2_random_forest"}  # fmt: skip


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True)
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"]).dt.date
    return frame.sort_values("startTime").reset_index(drop=True)


def dev_oof(dev: pd.DataFrame) -> pd.DataFrame:
    from ml.v2.candidates import fit, predict

    parts = []
    for fold in FOLDS:
        train, valid = fold_masks(dev["delivery_date"], fold)
        part = dev.loc[valid].copy()
        for label, name in PRODUCTION_SPECS.items():
            part[label] = predict(V1[name], fit(V1[name], dev.loc[train]), part)
        parts.append(part.assign(fold=fold_label(fold)))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    metrics = json.loads((ML_DIR / "reports" / "metrics.json").read_text(encoding="utf-8"))
    holdout = json.loads((ML_DIR / "holdout_feb2026" / "reports" / "holdout_metrics.json").read_text(encoding="utf-8"))
    baselines = {
        "february_reported_by_user": {"official_mae": 1693.6, "linear_mae": 1082.1, "reduction_pct": 36.1, "logistic_auc": 0.6932},
        "february_in_artifacts": {"official_mae": holdout["corrected_forecast"]["models"]["official_forecast"]["mae"],
                                  "linear_mae": holdout["corrected_forecast"]["models"]["linear_by_level"]["mae"],
                                  "reduction_pct": -holdout["corrected_forecast"]["models"]["linear_by_level"]["mae_change_vs_official_pct"],
                                  "logistic_auc": holdout["shortfall_risk"]["models"]["logistic_by_level"]["roc_auc"]},
        "january_in_artifacts": {c["name"]: c["test"] for c in metrics["regression"]["candidates"] + metrics["classification"]["candidates"]
                                 if c["name"] in ("official_forecast", "linear_by_level", "random_forest", "logistic_by_level")},
    }  # fmt: skip

    dev = load("dev")
    oof = dev_oof(dev)
    linear = oof["linear_by_level"].to_numpy()
    residual = oof["error_mw"] - linear  # corrected-forecast error of the production specification
    feature_links = {f: round(float(oof[f].corr(residual, method="spearman")), 3) for f in WIND_FEATURES if oof[f].notna().sum() > 50}
    weekly = oof.assign(residual=residual).groupby("fold")["residual"].agg(["mean", lambda s: s.abs().mean()]).round(0)
    worst = oof.assign(residual=residual, abs_residual=residual.abs()).nlargest(10, "abs_residual")

    jan = load("january").merge(pd.read_csv(ML_DIR / "predictions" / "test_predictions.csv", parse_dates=["startTime"])
                                [["startTime", "corrected_linear_by_level", "probability_logistic_by_level"]], on="startTime")  # fmt: skip
    feb_saved = pd.read_csv(ML_DIR / "holdout_feb2026" / "evaluated_predictions.csv", parse_dates=["startTime"])
    feb = load("february").merge(feb_saved[["startTime", "corrected_linear_by_level", "probability_logistic_by_level"]], on="startTime")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baselines": baselines,
        "dev_oof_production_specs": {
            "folds": [fold_label(f) for f in FOLDS],
            "regression": {k: {**regression_metrics(oof, oof[k] if k != "official" else np.zeros(len(oof))),
                               "distribution": error_distribution(oof, oof[k] if k != "official" else np.zeros(len(oof)))}
                           for k in ("official", "linear_by_level", "random_forest_regressor")},
            "classification": {k: classification_metrics(oof, oof[k]) for k in ("logistic_by_level", "random_forest_classifier")},
            "regression_segments": segments(oof, {"official": np.zeros(len(oof)), "linear_by_level": linear,
                                                  "random_forest": oof["random_forest_regressor"].to_numpy()}, "regression"),
            "classification_segments": segments(oof, {"logistic_by_level": oof["logistic_by_level"].to_numpy(),
                                                      "random_forest": oof["random_forest_classifier"].to_numpy()}, "classification"),
            "residual_feature_spearman": feature_links,
            "weekly_residual_mean_and_mae": {f: {"mean": float(r["mean"]), "mae": float(r["<lambda_0>"])} for f, r in weekly.iterrows()},
            "largest_residuals_kept_not_removed": worst[["startTime", "forecast_mw", "actual_mw", "error_mw", "residual", "forecast_revision_mw",
                                                         "err_last3h_mw"]].astype({"startTime": str}).round(0).to_dict(orient="records"),
        },
        "previously_examined_production_predictions": {
            name: {"regression": regression_metrics(frame, frame["forecast_mw"] - frame["corrected_linear_by_level"]),
                   "segments": segments(frame, {"linear_by_level": (frame["forecast_mw"] - frame["corrected_linear_by_level"]).to_numpy()}, "regression"),
                   "classification": classification_metrics(frame, frame["probability_logistic_by_level"])}
            for name, frame in (("january", jan), ("february", feb))
        },
    }  # fmt: skip
    write_json(REPORTS_DIR / "phase_a.json", report)

    d = report["dev_oof_production_specs"]
    print("Baselines in artifacts (Feb):", baselines["february_in_artifacts"])
    print("Dev OOF (8 weekly folds, 6 Nov - 31 Dec 2025):")
    for k, v in d["regression"].items():
        print(f"  {k:<24} MAE {v['mae']:7.1f} RMSE {v['rmse']:7.1f} bias {v['bias']:7.1f} reduction {v['mae_reduction_vs_official_pct']}%  dist {v['distribution']}")
    for k, v in d["classification"].items():
        print(f"  {k:<24} AUC {v['roc_auc']} PR-AUC {v['pr_auc']} Brier {v['brier']} ECE {v['ece_5_bands']}")
    for dim in ("forecast_band_gw", "hour_group_uk", "forecast_revision", "month"):
        rows = {k: (v["linear_by_level"]["n"], v["linear_by_level"]["bias"], v["linear_by_level"]["mae"], v["official"]["mae"]) for k, v in d["regression_segments"][dim].items()}
        print(f"  linear residual by {dim} (n, bias, MAE, official MAE): {rows}")
    rows = {k: (v["logistic_by_level"]["n"], v["logistic_by_level"]["observed_rate"], v["logistic_by_level"]["roc_auc"]) for k, v in d["classification_segments"]["forecast_band_gw"].items()}
    print(f"  logistic by forecast band (n, shortfall rate, AUC): {rows}")
    print("  residual vs feature (Spearman):", dict(sorted(feature_links.items(), key=lambda kv: -abs(kv[1]))))
    print("  weekly residual (mean, MAE):", {k: (v["mean"], v["mae"]) for k, v in d["weekly_residual_mean_and_mae"].items()})
    print("  largest residuals:", [(r["startTime"][:13], r["forecast_mw"], r["actual_mw"], r["residual"]) for r in d["largest_residuals_kept_not_removed"][:5]])
    for name, v in report["previously_examined_production_predictions"].items():
        bands = {k: (s["linear_by_level"]["bias"], s["linear_by_level"]["mae"]) for k, s in v["segments"]["forecast_band_gw"].items()}
        print(f"  {name} production linear: {v['regression']}; by band (bias, MAE): {bands}")


if __name__ == "__main__":
    main()
