"""
Ridge / gradient-boosting experiment, phases 2-3 (protocol: ridge_gbm/protocol.json, locked).

  1. verify   protocol.json is byte-identical to the locked version
  2. CV       every configuration on the 8 weekly folds of Oct-Dec 2025 (development rows only)
  3. select   per family: lowest pooled out-of-fold MAE (regression) / Brier (classification)
  4. freeze   refit the selected configurations on all development rows, save them and write
              reports/selection.json - before any January row is read
  5. January  read the January rows once; score every frozen model; segments, residuals,
              bootstrap intervals and the pre-registered decision

Usage (from backend/):  python -m ml.v2.ridge_gbm.run [--dev-only]
"""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from app.models.ml_model import load_bundle
from ml.holdout.stats import block_bootstrap, interval
from ml.v2.common import BACKEND_DIR, load_protocol, sha256, verify_protocol, write_json
from ml.v2.ridge_gbm.candidates import BASELINE, REFERENCE, TARGET, Config, configs
from ml.v2.ridge_gbm.config import (
    ALL,
    DATASET_FILE,
    FOLDS,
    GAP,
    HIGH_FORECAST_MW,
    MODELS_DIR,
    PRODUCTION_LINEAR,
    PRODUCTION_LOGISTIC,
    PRODUCTION_RF_CLASSIFIER,
    PRODUCTION_RF_REGRESSOR,
    PROTOCOL_FILE,
    PROTOCOL_LOCK,
    REPORTS_DIR,
    SELECTION_FILE,
    bands,
    fold_label,
    fold_masks,
    load_development,
    load_january,
)

ARTIFACTS = {
    ("regression", "linear_by_level"): PRODUCTION_LINEAR,
    ("regression", "random_forest"): PRODUCTION_RF_REGRESSOR,
    ("classification", "logistic_by_level"): PRODUCTION_LOGISTIC,
    ("classification", "random_forest"): PRODUCTION_RF_CLASSIFIER,
}
SELECT_BY = {"regression": "mae", "classification": "brier"}
TASKS = ("regression", "classification")


# ---- Scores ------------------------------------------------------------------------


def regression_scores(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    """Corrected error = corrected - actual = error_mw - predicted error; negative = over-corrected."""
    corrected_error = frame["error_mw"].to_numpy() - predicted
    return {
        "hours": len(frame),
        "mae": float(np.abs(corrected_error).mean()),
        "rmse": float(np.sqrt((corrected_error**2).mean())),
        "bias": float(corrected_error.mean()),
        "over_corrected_pct": float((corrected_error < 0).mean() * 100),
    }


def classification_scores(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    y = frame["shortfall"].to_numpy()
    both = len(np.unique(y)) == 2
    return {
        "hours": len(frame),
        "observed_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if both else None,
        "pr_auc": float(average_precision_score(y, probability)) if both else None,
        "brier": float(brier_score_loss(y, probability)),
    }


def scores(task: str, frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return regression_scores(frame, prediction) if task == "regression" else classification_scores(frame, prediction)


# ---- Phases 2-4: development only -----------------------------------------------------


def cross_validate(dev: pd.DataFrame, task: str, candidates: list[Config]) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Expanding-window CV: each weekly fold trains on deliveries up to 2 days before it."""
    target, metric = TARGET[task], SELECT_BY[task]
    oof = {c.name: np.full(len(dev), np.nan) for c in candidates}
    per_fold = []
    for fold in FOLDS:
        train, valid = fold_masks(dev["delivery_date"], fold)
        if dev.loc[train, "delivery_date"].max() > fold[0] - GAP:
            raise AssertionError("training rows reach past the label gap")
        for c in candidates:
            model = c.build().fit(dev.loc[train, ALL], dev.loc[train, target])
            prediction = c.predict(model, dev.loc[valid])
            oof[c.name][valid.to_numpy()] = prediction
            per_fold.append({"config": c.name, "fold": fold_label(fold), metric: scores(task, dev.loc[valid], prediction)[metric]})
    in_cv = ~np.isnan(oof[candidates[0].name])
    pooled = pd.DataFrame([{"family": c.family, "config": c.name, **scores(task, dev.loc[in_cv], oof[c.name][in_cv])} for c in candidates])
    folds = pd.DataFrame(per_fold).pivot(index="config", columns="fold", values=metric).add_prefix(f"{metric}_fold_")
    return pooled.merge(folds, left_on="config", right_index=True), oof, in_cv


def select(table: pd.DataFrame, task: str, candidates: list[Config]) -> dict[str, Config]:
    """Per family, the lowest pooled CV score; ties go to the earlier (simpler) configuration."""
    by_name = {c.name: c for c in candidates}
    return {family: by_name[rows.loc[rows[SELECT_BY[task]].idxmin(), "config"]] for family, rows in table.groupby("family", sort=False)}


def freeze(dev: pd.DataFrame, task: str, chosen: dict[str, Config]) -> tuple[dict, dict]:
    """Final models: production artifacts where they exist, otherwise one refit on all development rows."""
    models, record = {}, {}
    for family, config in chosen.items():
        artifact = ARTIFACTS.get((task, family))
        refit = config.build().fit(dev[ALL], dev[TARGET[task]])
        if artifact:
            model = load_bundle(artifact).estimator
            gap = float(np.max(np.abs(config.predict(model, dev) - config.predict(refit, dev))))
            record[family] = {"config": config.name, "source": "production artifact (unchanged)", "path": artifact.relative_to(BACKEND_DIR).as_posix(),
                              "sha256": sha256(artifact), "max_abs_difference_vs_refit_on_development": gap}  # fmt: skip
        else:
            model = refit
            entry = {"config": config.name, "source": "fitted once on all development deliveries"}
            if family not in REFERENCE.values():
                path = MODELS_DIR / f"{task}_{family}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path, compress=3)
                entry |= {"path": path.relative_to(BACKEND_DIR).as_posix(), "sha256": sha256(path)}
            record[family] = entry
        models[family] = model
    return models, record


# ---- Phase 5: January, read once -------------------------------------------------------


def segment_table(frame: pd.DataFrame, task: str, predictions: dict, keys: pd.Series) -> list[dict]:
    rows = []
    for key in sorted(keys.unique()):
        mask = (keys == key).to_numpy()
        row = {"segment": key, "hours": int(mask.sum())}
        for family, prediction in predictions.items():
            for name, value in scores(task, frame.loc[mask], prediction[mask]).items():
                if name not in ("hours", "observed_rate"):
                    row[f"{family}_{name}"] = value
        rows.append(row)
    return rows


def high_forecast_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    day_mean = frame.groupby("delivery_date")["forecast_mw"].transform("mean")
    return {"high_forecast_hours": (frame["forecast_mw"] >= HIGH_FORECAST_MW).to_numpy(), "high_forecast_days": (day_mean >= HIGH_FORECAST_MW).to_numpy()}


def regression_segments(frame: pd.DataFrame, predictions: dict) -> dict:
    out = {
        "by_hour_uk": segment_table(frame, "regression", predictions, frame["hour_uk"].map(lambda h: f"{h:02d}:00")),
        "by_forecast_band_gw": segment_table(frame, "regression", predictions, bands(frame["forecast_mw"])),
    }
    for name, mask in high_forecast_masks(frame).items():
        out[name] = {"hours": int(mask.sum()), "days": int(frame.loc[mask, "delivery_date"].nunique()),
                     **{family: regression_scores(frame.loc[mask], p[mask]) for family, p in predictions.items()}}  # fmt: skip
    return out


def residuals(frame: pd.DataFrame, predictions: dict) -> dict:
    out = {}
    for family, prediction in predictions.items():
        corrected_error = pd.Series(frame["error_mw"].to_numpy() - prediction, index=frame.index)
        slope = LinearRegression().fit(frame[["forecast_mw"]], corrected_error).coef_[0]
        day_mean = corrected_error.groupby(frame["delivery_date"]).transform("mean")
        out[family] = {
            "corrected_error_vs_forecast_slope_mw_per_gw": float(slope * 1000),
            "day_offset_share_of_variance": float(day_mean.var() / corrected_error.var()),
            "mean_corrected_error_by_band": {band: float(corrected_error[bands(frame["forecast_mw"]) == band].mean()) for band in sorted(bands(frame["forecast_mw"]).unique())},
        }
    return out


def high_forecast_days(frame: pd.DataFrame, predictions: dict) -> list[dict]:
    rows = []
    for day, part in frame.groupby("delivery_date"):
        if part["forecast_mw"].mean() < HIGH_FORECAST_MW:
            continue
        index = frame.index.get_indexer(part.index)
        row = {"delivery_date": str(day), "mean_forecast_mw": float(part["forecast_mw"].mean()), "mean_actual_mw": float(part["actual_mw"].mean())}
        for family, prediction in predictions.items():
            corrected_error = part["error_mw"].to_numpy() - prediction[index]
            row[f"{family}_bias"] = float(corrected_error.mean())
            row[f"{family}_mae"] = float(np.abs(corrected_error).mean())
        rows.append(row)
    return rows


def regression_intervals(frame: pd.DataFrame, predictions: dict) -> dict:
    """Baseline minus candidate (positive = candidate better), 3-day blocks; high-forecast days with 1-day blocks."""
    days = frame["delivery_date"]
    base = frame["error_mw"].to_numpy() - predictions[BASELINE["regression"]]
    high = high_forecast_masks(frame)["high_forecast_days"]
    out = {}
    for family, prediction in predictions.items():
        if family in (BASELINE["regression"], REFERENCE["regression"]):
            continue
        cand = frame["error_mw"].to_numpy() - prediction
        mae = block_bootstrap(days, lambda r: np.abs(base[r]).mean() - np.abs(cand[r]).mean(), 3)
        rmse = block_bootstrap(days, lambda r: np.sqrt((base[r] ** 2).mean()) - np.sqrt((cand[r] ** 2).mean()), 3)
        high_days = days[high].reset_index(drop=True)
        b_high, c_high = base[high], cand[high]
        high_mae = block_bootstrap(high_days, lambda r: np.abs(b_high[r]).mean() - np.abs(c_high[r]).mean(), 1)
        high_abs_bias = block_bootstrap(high_days, lambda r: abs(b_high[r].mean()) - abs(c_high[r].mean()), 1)
        out[family] = {
            "mae_gain": {"point": float(np.abs(base).mean() - np.abs(cand).mean()), "ci95": interval(mae)},
            "rmse_gain": {"point": float(np.sqrt((base**2).mean()) - np.sqrt((cand**2).mean())), "ci95": interval(rmse)},
            "high_forecast_days_mae_gain": {"point": float(np.abs(b_high).mean() - np.abs(c_high).mean()), "ci95": interval(high_mae)},
            "high_forecast_days_abs_bias_reduction": {"point": float(abs(b_high.mean()) - abs(c_high.mean())), "ci95": interval(high_abs_bias)},
        }
    return out


def classification_intervals(frame: pd.DataFrame, predictions: dict) -> dict:
    days, y = frame["delivery_date"], frame["shortfall"].to_numpy()
    base = predictions[BASELINE["classification"]]
    out = {}
    for family, p in predictions.items():
        if family in (BASELINE["classification"], REFERENCE["classification"]):
            continue
        brier = block_bootstrap(days, lambda r: ((base[r] - y[r]) ** 2).mean() - ((p[r] - y[r]) ** 2).mean(), 3)
        auc = block_bootstrap(days, lambda r: roc_auc_score(y[r], p[r]) - roc_auc_score(y[r], base[r]), 3, n_resamples=2000)
        out[family] = {
            "brier_gain": {"point": float(((base - y) ** 2).mean() - ((p - y) ** 2).mean()), "ci95": interval(brier)},
            "roc_auc_gain": {"point": float(roc_auc_score(y, p) - roc_auc_score(y, base)), "ci95": interval(auc)},
        }
    return out


def decisions(task: str, cv: pd.DataFrame, chosen: dict, intervals: dict) -> dict:
    metric = SELECT_BY[task]
    cv_score = {family: float(cv.loc[cv["config"] == config.name, metric].iloc[0]) for family, config in chosen.items()}
    gain_key = "mae_gain" if task == "regression" else "brier_gain"
    out = {}
    for family, result in intervals.items():
        better_in_cv = cv_score[family] < cv_score[BASELINE[task]]
        low = result[gain_key]["ci95"][0]
        out[family] = {
            f"cv_{metric}": cv_score[family],
            f"baseline_cv_{metric}": cv_score[BASELINE[task]],
            "better_in_cv": better_in_cv,
            f"january_{gain_key}_ci95_low": low,
            "supported": bool(better_in_cv and low > 0),
        }
    return out


def _tidy(value):
    """Round floats for the reports (3 decimals) without touching the numbers used for decisions."""
    if isinstance(value, float):
        return round(value, 4) if abs(value) < 10 else round(value, 1)
    if isinstance(value, dict):
        return {k: _tidy(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_tidy(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(dev_only: bool = False) -> None:
    lock = verify_protocol(PROTOCOL_FILE, PROTOCOL_LOCK)
    protocol = load_protocol(PROTOCOL_FILE)
    dev = load_development()
    print(f"Protocol {lock['sha256'][:12]} verified. Development: {len(dev)} h, {dev['delivery_date'].min()} .. {dev['delivery_date'].max()}")

    chosen, models, cv, oof, record = {}, {}, {}, {}, {}
    for task in TASKS:
        candidates = configs(protocol, task)
        print(f"\n[{task}] cross-validating {len(candidates)} configurations on {len(FOLDS)} folds ...")
        cv[task], oof_all, in_cv = cross_validate(dev, task, candidates)
        cv[task].to_csv(REPORTS_DIR / f"cv_{task}.csv", index=False, float_format="%.4f")
        chosen[task] = select(cv[task], task, candidates)
        oof[task] = (in_cv, {family: oof_all[config.name] for family, config in chosen[task].items()})
        models[task], record[task] = freeze(dev, task, chosen[task])
        metric = SELECT_BY[task]
        for family, config in chosen[task].items():
            row = cv[task].loc[cv[task]["config"] == config.name].iloc[0]
            print(f"  {family:<18} {config.name:<85} CV {metric} {row[metric]:.4f}")

    cv_segments = {}
    in_cv, oof_regression = oof["regression"]
    cv_frame = dev.loc[in_cv].reset_index(drop=True)
    cv_segments["regression"] = regression_segments(cv_frame, {f: p[in_cv] for f, p in oof_regression.items()})
    cv_segments["regression"]["residuals"] = residuals(cv_frame, {f: p[in_cv] for f, p in oof_regression.items()})

    selection = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": lock["sha256"],
        "dataset_file": DATASET_FILE.relative_to(BACKEND_DIR).as_posix(),
        "dataset_sha256": sha256(DATASET_FILE),
        "development_hours": len(dev),
        "january_rows_read": False,
        "selected": {task: {family: config.name for family, config in chosen[task].items()} for task in TASKS},
        "models": record,
    }
    write_json(SELECTION_FILE, _tidy(selection))
    if dev_only:
        print(f"\nSelection written ({SELECTION_FILE.name}); --dev-only: January not read.")
        return
    print(f"\nSelection frozen: {SELECTION_FILE.name} (sha256 {sha256(SELECTION_FILE)[:12]}). Reading January now.")

    jan = load_january()
    results = {"protocol_sha256": lock["sha256"], "selection_sha256": sha256(SELECTION_FILE), "january_hours": len(jan),
               "cv_pooled": {}, "cv_segments": cv_segments, "january": {}}  # fmt: skip
    for task in TASKS:
        predictions = {family: config.predict(models[task][family], jan) for family, config in chosen[task].items()}
        selected_rows = cv[task].loc[cv[task]["config"].isin([c.name for c in chosen[task].values()])]
        results["cv_pooled"][task] = {row["family"]: {k: row[k] for k in row.index if not k.startswith(SELECT_BY[task] + "_fold_") and k != "family"}
                                      for _, row in selected_rows.iterrows()}  # fmt: skip
        overall = {family: scores(task, jan, p) for family, p in predictions.items()}
        if task == "regression":
            segments = regression_segments(jan, predictions)
            intervals = regression_intervals(jan, predictions)
            pd.DataFrame(segments["by_hour_uk"]).to_csv(REPORTS_DIR / "january_regression_by_hour.csv", index=False, float_format="%.1f")
            pd.DataFrame(segments["by_forecast_band_gw"]).to_csv(REPORTS_DIR / "january_regression_by_band.csv", index=False, float_format="%.1f")
            pd.DataFrame(high_forecast_days(jan, predictions)).to_csv(REPORTS_DIR / "january_high_forecast_days.csv", index=False, float_format="%.1f")
            extra = {"segments": segments, "residuals": residuals(jan, predictions), "high_forecast_days": high_forecast_days(jan, predictions)}
        else:
            intervals = classification_intervals(jan, predictions)
            by_band = segment_table(jan, task, predictions, bands(jan["forecast_mw"]))
            pd.DataFrame(by_band).to_csv(REPORTS_DIR / "january_classification_by_band.csv", index=False, float_format="%.4f")
            extra = {"by_forecast_band_gw": by_band}
        results["january"][task] = {"overall": overall, "intervals": intervals, "decision": decisions(task, cv[task], chosen[task], intervals), **extra}
    write_json(REPORTS_DIR / "results.json", _tidy(results))

    for task in TASKS:
        print(f"\n[{task}] January (744 h), each model scored once")
        for family, row in results["january"][task]["overall"].items():
            text = (f"MAE {row['mae']:7.1f}  RMSE {row['rmse']:7.1f}  bias {row['bias']:+7.1f}  over-corrected {row['over_corrected_pct']:4.1f}%"
                    if task == "regression" else f"ROC-AUC {row['roc_auc']:.4f}  PR-AUC {row['pr_auc']:.4f}  Brier {row['brier']:.4f}")
            print(f"  {family:<18} {text}")
        for family, verdict in results["january"][task]["decision"].items():
            print(f"  decision {family:<18} {verdict}")
    print(f"\nWrote {REPORTS_DIR / 'results.json'} and CSV tables in {REPORTS_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ridge / gradient-boosting experiment")
    parser.add_argument("--dev-only", action="store_true", help="stop after selection; do not read January")
    main(dev_only=parser.parse_args().dev_only)
