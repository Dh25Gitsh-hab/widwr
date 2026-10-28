"""
Stage D: rolling-origin cross-validation of the pre-registered candidates, selection by the
locked rules, final refit on all development data, and a January context report.

    1. Verify   protocol.json is unchanged since it was locked
    2. CV       9 monthly folds (Apr-Dec 2025). Static candidates: fitted once per fold on deliveries
                up to 2 days before the fold. Rolling candidates: refitted for every delivery day d on
                deliveries in [d-1-W, d-2]
    3. Select   regression: lowest mean fold MAE; classification: highest mean fold ROC-AUC
                (ties within 0.005 -> lowest mean fold Brier)
    4. Extras   Platt calibration and two thresholds from the selected classifier's OOF predictions
    5. Freeze   refit the selected candidates on all development data -> ml/v2/models/
    6. Context  January 2026 predictions next to the production defaults (seen before: context only)

Usage (from the backend folder):  python -m ml.v2.experiment
"""

import json
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.v2.candidates import CANDIDATES, CLASSIFICATION, REGRESSION, Candidate, V2Model, fit, predict
from ml.v2.common import LEVEL_BANDS_MW, ML_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, sha256, verify_protocol, write_json

FOLD_MONTHS = [f"2025-{m:02d}" for m in range(4, 13)]
GAP = timedelta(days=2)  # deliveries up to d-2 have every label published before d's cutoff (09:50 on d-1)
TIE_AUC = 0.005
LABELS = ["delivery_date", "startTime", "forecast_mw", "actual_mw", "error_mw", "shortfall_1gw", "hour_uk"]


def load(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True)
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"]).dt.date
    return frame.sort_values("startTime").reset_index(drop=True)


def fold_split(dev: pd.DataFrame, month: str) -> tuple[pd.Series, pd.Series]:
    first = date.fromisoformat(f"{month}-01")
    valid = dev["delivery_date"].map(lambda d: d.strftime("%Y-%m")) == month
    train = dev["delivery_date"] <= first - GAP
    return train, valid


def rolling_window(dates: pd.Series, day: date, days: int) -> pd.Series:
    return (dates >= day - timedelta(days=days + 1)) & (dates <= day - GAP)


def predictions_for(candidate: Candidate, history: pd.DataFrame, train_mask: pd.Series, target: pd.DataFrame) -> np.ndarray:
    """Static: one fit on history[train_mask]. Rolling: one fit per target day on its own window."""
    if candidate.rolling_days is None:
        return predict(candidate, fit(candidate, history.loc[train_mask]), target)
    out = np.empty(len(target))
    for day, rows in target.groupby("delivery_date"):
        window = history.loc[rolling_window(history["delivery_date"], day, candidate.rolling_days)]
        out[target.index.get_indexer(rows.index)] = predict(candidate, fit(candidate, window), rows)
    return out


# ---- Metrics --------------------------------------------------------------------------------


def regression_metrics(frame: pd.DataFrame, predicted_error: np.ndarray) -> dict:
    corrected_error = frame["error_mw"].to_numpy() - predicted_error  # (forecast - prediction) - actual
    official_mae = np.abs(frame["error_mw"]).mean()
    mae = np.abs(corrected_error).mean()
    return {"n": len(frame), "mae": round(float(mae), 1), "rmse": round(float(np.sqrt((corrected_error ** 2).mean())), 1),
            "bias": round(float(corrected_error.mean()), 1),
            "mae_reduction_vs_official_pct": round(float(100 * (official_mae - mae) / official_mae), 1)}  # fmt: skip


def ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    band = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((band == b).mean() * abs(p[band == b].mean() - y[band == b].mean()) for b in range(bins) if (band == b).any()))


def classification_metrics(frame: pd.DataFrame, p: np.ndarray, thresholds: dict | None = None) -> dict:
    y = frame["shortfall_1gw"].to_numpy().astype(int)
    both = len(np.unique(y)) == 2 and np.ptp(p) > 0
    out = {"n": len(frame), "observed_rate": round(float(y.mean()), 4),
           "roc_auc": round(float(roc_auc_score(y, p)), 4) if both else (0.5 if len(np.unique(y)) == 2 else None),
           "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) == 2 else None,
           "brier": round(float(brier_score_loss(y, p)), 4), "ece_5_bands": round(ece(y, p), 4)}  # fmt: skip
    for label, t in {"at_0_5": 0.5, **(thresholds or {})}.items():
        pred = p >= t
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[label] = {"threshold": round(float(t), 4), "precision": round(precision, 4), "recall": round(recall, 4),
                      "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}  # fmt: skip
    return out


def segments(frame: pd.DataFrame, columns: dict[str, np.ndarray], task: str) -> dict:
    groups = {
        "forecast_band_gw": pd.cut(frame["forecast_mw"], LEVEL_BANDS_MW, right=False,
                                   labels=["0-5", "5-10", "10-15", "15-20", "20+"]).astype(str),
        "hour_group_uk": pd.cut(frame["hour_uk"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str),
        "month": frame["delivery_date"].map(lambda d: d.strftime("%Y-%m")),
    }  # fmt: skip
    out = {}
    for dimension, keys in groups.items():
        out[dimension] = {}
        for key in sorted(keys.unique()):
            mask = (keys == key).to_numpy()
            rows = frame.loc[mask]
            out[dimension][key] = {name: (regression_metrics(rows, pred[mask]) if task == "regression"
                                          else classification_metrics(rows, pred[mask])) for name, pred in columns.items()}  # fmt: skip
    return out


def thresholds_from(y: np.ndarray, p: np.ndarray) -> dict:
    """(a) highest threshold whose recall >= 0.80; (b) the F1-maximising threshold (OOF data only)."""
    best_recall, best_f1, best_f1_t = None, -1.0, 0.5
    for t in np.unique(p)[::-1]:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        recall = tp / (y == 1).sum()
        precision = tp / pred.sum()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if best_recall is None and recall >= 0.80:
            best_recall = t
        if f1 > best_f1:
            best_f1, best_f1_t = f1, t
    return {"recall_80": float(best_recall), "max_f1": float(best_f1_t)}


def platt(y: np.ndarray, p: np.ndarray) -> dict:
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    model = LogisticRegression(C=1e6).fit(logit, y)
    return {"slope": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


def apply_platt(p: np.ndarray, params: dict) -> np.ndarray:
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(params["slope"] * logit + params["intercept"])))


# ---- Main -----------------------------------------------------------------------------------


def main() -> None:
    lock = verify_protocol()
    dev = load(PROCESSED_DIR / "dev_dataset.csv")
    january = load(PROCESSED_DIR / "january_dataset.csv")

    # 2. Cross-validation (out-of-fold predictions for every candidate)
    oof_parts, fold_rows = [], []
    for month in FOLD_MONTHS:
        train, valid = fold_split(dev, month)
        target = dev.loc[valid]
        part = target[LABELS].copy()
        for candidate in REGRESSION + CLASSIFICATION:
            if candidate.kind != "ensemble":
                part[candidate.name] = predictions_for(candidate, dev, train, target)
        for candidate in REGRESSION + CLASSIFICATION:
            if candidate.kind == "ensemble":
                part[candidate.name] = part[list(candidate.members)].mean(axis=1)
        for candidate in REGRESSION:
            fold_rows.append({"fold": month, "candidate": candidate.name, **regression_metrics(part, part[candidate.name].to_numpy())})
        for candidate in CLASSIFICATION:
            fold_rows.append({"fold": month, "candidate": candidate.name, **classification_metrics(part, part[candidate.name].to_numpy())})
        train_days = sorted(dev.loc[train, "delivery_date"].unique())
        print(f"fold {month}: train {train_days[0]}..{train_days[-1]} ({len(train_days)} days), valid {len(target)} hours", flush=True)
        oof_parts.append(part.assign(fold=month))
    oof = pd.concat(oof_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    # 3. Selection by the locked rules
    reg_summary = folds[folds.candidate.isin([c.name for c in REGRESSION])].groupby("candidate")[["mae", "rmse", "bias", "mae_reduction_vs_official_pct"]].mean()
    cls_summary = folds[folds.candidate.isin([c.name for c in CLASSIFICATION])].groupby("candidate")[["roc_auc", "pr_auc", "brier", "ece_5_bands"]].mean()
    selected_regressor = reg_summary["mae"].idxmin()
    contenders = cls_summary[cls_summary["roc_auc"] >= cls_summary["roc_auc"].max() - TIE_AUC]
    selected_classifier = contenders["brier"].idxmin()

    # 4. Classifier extras from out-of-fold predictions only
    y_oof, p_oof = oof["shortfall_1gw"].to_numpy().astype(int), oof[selected_classifier].to_numpy()
    thresholds = thresholds_from(y_oof, p_oof)
    calibration = platt(y_oof, p_oof)

    # 5. Freeze: refit the selected candidates on all development data
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained = (dev["delivery_date"].min(), dev["delivery_date"].max())
    frozen = {}
    for name in (selected_regressor, selected_classifier):
        candidate = CANDIDATES[name]
        fitted = None if candidate.rolling_days else fit(candidate, dev)
        extras = {"thresholds": thresholds, "platt": calibration} if candidate.task == "classification" else {}
        frozen[name] = V2Model(name, candidate.task, fitted, *trained, lock["sha256"], extras)
        joblib.dump(frozen[name], MODELS_DIR / f"{candidate.task}_{name}.joblib", compress=3)

    # 6. January context (seen before) - v2 selected vs production, same hours
    history = pd.concat([dev, january], ignore_index=True)
    jan = january[LABELS].copy()
    for name in (selected_regressor, selected_classifier):
        candidate = CANDIDATES[name]
        jan[name] = (predictions_for(candidate, history, None, january) if candidate.rolling_days
                     else predict(candidate, frozen[name].fitted, january))  # fmt: skip
    production = pd.read_csv(ML_DIR / "predictions" / "test_predictions.csv")
    production["startTime"] = pd.to_datetime(production["startTime"], utc=True)
    jan = jan.merge(production[["startTime", "corrected_linear_by_level", "corrected_random_forest",
                                "probability_logistic_by_level", "probability_random_forest"]], on="startTime")  # fmt: skip
    jan_reg = {"R0_official (production baseline)": np.zeros(len(jan)),
               "production linear_by_level (default)": jan["forecast_mw"] - jan["corrected_linear_by_level"],
               "production random_forest": jan["forecast_mw"] - jan["corrected_random_forest"],
               f"v2 {selected_regressor}": jan[selected_regressor]}  # fmt: skip
    jan_cls = {"production logistic_by_level (default)": jan["probability_logistic_by_level"].to_numpy(),
               "production random_forest": jan["probability_random_forest"].to_numpy(),
               f"v2 {selected_classifier}": jan[selected_classifier].to_numpy()}  # fmt: skip

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False, float_format="%.4f")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": lock["sha256"],
        "dev_dataset_sha256": sha256(PROCESSED_DIR / "dev_dataset.csv"),
        "folds": FOLD_MONTHS,
        "cv_mean": {"regression": reg_summary.round(4).to_dict(orient="index"), "classification": cls_summary.round(4).to_dict(orient="index")},
        "cv_by_fold": fold_rows,
        "selected": {"regression": selected_regressor, "classification": selected_classifier,
                     "classification_tie_contenders": list(contenders.index)},
        "classifier_thresholds_from_oof": thresholds,
        "platt_from_oof": calibration,
        "oof_selected_classifier": {"raw": classification_metrics(oof, p_oof, thresholds),
                                    "platt_in_sample": classification_metrics(oof, apply_platt(p_oof, calibration), thresholds)},
        "oof_segments": {
            "regression": segments(oof, {n: oof[n].to_numpy() for n in ("R0_official", "R1_linear_by_level", "R2_random_forest", selected_regressor)}, "regression"),
            "classification": segments(oof, {n: oof[n].to_numpy() for n in ("C1_logistic_by_level", "C2_random_forest", selected_classifier)}, "classification"),
        },
        "frozen_models": {name: {"file": f"models/{m.task}_{name}.joblib", "trained_deliveries": [str(d) for d in trained],
                                 "rolling_days": CANDIDATES[name].rolling_days} for name, m in frozen.items()},
        "january_context_seen_before": {
            "note": "January 2026 was evaluated before and informed the production defaults; context only, not used to choose.",
            "regression": {k: regression_metrics(jan, np.asarray(v)) for k, v in jan_reg.items()},
            "classification": {k: classification_metrics(jan, v, thresholds if k.startswith("v2") else None) for k, v in jan_cls.items()},
        },
    }  # fmt: skip
    write_json(REPORTS_DIR / "cv_results.json", report)

    print("\nCV mean over 9 folds - regression (corrected forecast):")
    print(reg_summary.sort_values("mae").round(1).to_string())
    print("\nCV mean over 9 folds - classification:")
    print(cls_summary.sort_values("roc_auc", ascending=False).round(4).to_string())
    print(f"\nSelected: regression {selected_regressor}; classification {selected_classifier} (contenders {list(contenders.index)})")
    print(f"Thresholds (OOF): {thresholds}; Platt: {calibration}")


if __name__ == "__main__":
    main()
