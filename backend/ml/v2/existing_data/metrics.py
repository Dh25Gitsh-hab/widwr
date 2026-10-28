"""Scoring and segment helpers shared by phase_a.py and run.py (one definition of every metric)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.v2.common import LEVEL_BANDS_MW

BAND_LABELS = ["0-5", "5-10", "10-15", "15-20", "20+"]


def regression_metrics(frame: pd.DataFrame, predicted_error) -> dict:
    """Corrected forecast = forecast - predicted error; its error = error_mw - predicted error."""
    corrected = frame["error_mw"].to_numpy() - np.asarray(predicted_error, dtype=float)
    official_mae = np.abs(frame["error_mw"]).mean()
    mae = np.abs(corrected).mean()
    return {"n": len(frame), "mae": round(float(mae), 1), "rmse": round(float(np.sqrt((corrected ** 2).mean())), 1),
            "bias": round(float(corrected.mean()), 1),
            "mae_reduction_vs_official_pct": round(float(100 * (official_mae - mae) / official_mae), 1)}  # fmt: skip


def error_distribution(frame: pd.DataFrame, predicted_error) -> dict:
    corrected = frame["error_mw"].to_numpy() - np.asarray(predicted_error, dtype=float)
    p = np.percentile(corrected, [5, 25, 50, 75, 95])
    return {"p5": round(float(p[0])), "p25": round(float(p[1])), "median": round(float(p[2])), "p75": round(float(p[3])),
            "p95": round(float(p[4])), "within_500_mw_pct": round(float((np.abs(corrected) <= 500).mean() * 100), 1),
            "within_1000_mw_pct": round(float((np.abs(corrected) <= 1000).mean() * 100), 1)}  # fmt: skip


def ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    band = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((band == b).mean() * abs(p[band == b].mean() - y[band == b].mean()) for b in range(bins) if (band == b).any()))


def classification_metrics(frame: pd.DataFrame, probability, thresholds: dict | None = None) -> dict:
    y = frame["shortfall_1gw"].to_numpy().astype(int)
    p = np.asarray(probability, dtype=float)
    two_classes = len(np.unique(y)) == 2
    out = {"n": len(frame), "observed_rate": round(float(y.mean()), 4),
           "roc_auc": (round(float(roc_auc_score(y, p)), 4) if np.ptp(p) > 0 else 0.5) if two_classes else None,
           "pr_auc": round(float(average_precision_score(y, p)), 4) if two_classes else None,
           "brier": round(float(brier_score_loss(y, p)), 4), "ece_5_bands": round(ece(y, p), 4)}  # fmt: skip
    for label, t in {"at_0_5": 0.5, **(thresholds or {})}.items():
        pred = p >= t
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[label] = {"threshold": round(float(t), 4), "precision": round(precision, 4), "recall": round(recall, 4),
                      "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}  # fmt: skip
    return out


def segment_keys(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "forecast_band_gw": pd.cut(frame["forecast_mw"], LEVEL_BANDS_MW, right=False, labels=BAND_LABELS).astype(str),
        "hour_group_uk": pd.cut(frame["hour_uk"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str),
        "forecast_revision": pd.cut(frame["forecast_revision_mw"], [-np.inf, -1000, 0, 1000, np.inf],
                                    labels=["<=-1GW", "-1GW..0", "0..1GW", ">=1GW"], right=False).astype(str),
        "month": frame["delivery_date"].map(lambda d: str(d)[:7]),
    }  # fmt: skip


def segments(frame: pd.DataFrame, predictions: dict, task: str) -> dict:
    """Metrics of every model in every segment (same rows for every model)."""
    out = {}
    for dimension, keys in segment_keys(frame).items():
        out[dimension] = {}
        for key in sorted(keys.unique()):
            mask = (keys == key).to_numpy()
            rows = frame.loc[mask]
            out[dimension][key] = {name: (regression_metrics(rows, np.asarray(p)[mask]) if task == "regression"
                                          else classification_metrics(rows, np.asarray(p)[mask]))
                                   for name, p in predictions.items()}  # fmt: skip
    return out
