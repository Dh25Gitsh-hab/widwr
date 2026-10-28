"""
Phases B-D of the existing-data experiment (protocol: existing_data/protocol.json, locked).

    1. Verify     the protocol is unchanged since it was locked
    2. CV         every configuration on 8 weekly folds (6 Nov - 31 Dec 2025); static models fitted on
                  deliveries up to 2 days before each fold, rolling ones refitted daily on [d-1-W, d-2]
    3. Select     tune within each family (mean over 8 folds), then pick the best family winner
    4. Checks     nested check (choose on folds 1-4, score on 5-8); rolling vs static counterpart
    5. Ensembles  members and weights from folds 1-4, scored on folds 5-8 against the best single model
    6. Classifier calibration (none / Platt / isotonic, nested) and thresholds from development OOF only
    7. Freeze     the selected models, refitted on all development data -> models/
    8. Evaluate   ONCE on January and February 2026 (both previously examined), next to production

Usage (from the backend folder):  python -m ml.v2.existing_data.run
"""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ml.holdout.stats import block_bootstrap, interval
from ml.v2.common import ML_DIR, sha256, verify_protocol, write_json
from ml.v2.existing_data.candidates import (
    CLASSIFICATION,
    CONFIGS,
    PRODUCTION_SPEC,
    REGRESSION,
    STATIC_COUNTERPART,
    Config,
    fit,
    predict,
)
from ml.v2.existing_data.config import (
    EARLY_FOLDS,
    FOLDS,
    LATE_FOLDS,
    MODELS_DIR,
    PROCESSED_DIR,
    PROTOCOL_FILE,
    PROTOCOL_LOCK,
    REPORTS_DIR,
    fold_label,
    fold_masks,
    rolling_window,
)
from ml.v2.existing_data.metrics import classification_metrics, error_distribution, regression_metrics, segments
from ml.v2.existing_data.phase_a import load

LABELS = ["delivery_date", "startTime", "forecast_mw", "actual_mw", "error_mw", "shortfall_1gw", "hour_uk", "forecast_revision_mw"]
EARLY, LATE = [fold_label(f) for f in EARLY_FOLDS], [fold_label(f) for f in LATE_FOLDS]
RF_PRODUCTION = "[min_samples_leaf=20,max_features=0.5]"
TIE_AUC = 0.005


# ---- predictions --------------------------------------------------------------------------


def predictions_for(config: Config, history: pd.DataFrame, target: pd.DataFrame, train_mask=None, fitted=None) -> np.ndarray:
    """Static: one fit (or a frozen model). Rolling kinds: every target day uses only its own known window."""
    if config.kind in ("model", "official") and config.rolling_days is None:
        model = fitted if fitted is not None else fit(config, history.loc[train_mask])
        return predict(config, model, target)
    base = None
    if config.kind == "rolling_offset":
        base = fitted if fitted is not None else fit(config, history.loc[train_mask])
    out = np.empty(len(target))
    for day, rows in target.groupby("delivery_date"):
        window = history.loc[rolling_window(history["delivery_date"], day, config.rolling_days)]
        positions = target.index.get_indexer(rows.index)
        if config.kind == "rolling_bias":
            out[positions] = window["error_mw"].mean()
        elif config.kind == "rolling_offset":
            offset = (window["error_mw"] - predict(config, base, window)).mean()
            out[positions] = predict(config, base, rows) + offset
        else:
            out[positions] = predict(config, fit(config, window), rows)
    return out


def score(config: Config, frame: pd.DataFrame, prediction) -> dict:
    return regression_metrics(frame, prediction) if config.task == "regression" else classification_metrics(frame, prediction)


# ---- selection helpers ----------------------------------------------------------------------


def family_winners(folds: pd.DataFrame, task: str, fold_names: list[str]) -> pd.DataFrame:
    metric, ascending = ("mae", True) if task == "regression" else ("roc_auc", False)
    rows = folds[(folds.task == task) & folds.fold.isin(fold_names)]
    means = rows.groupby(["family", "config"])[["mae", "rmse", "bias", "mae_reduction_vs_official_pct"] if task == "regression"
                                               else ["roc_auc", "pr_auc", "brier", "ece_5_bands"]].mean().reset_index()  # fmt: skip
    best = means.sort_values(metric, ascending=ascending).groupby("family").head(1)
    return best.sort_values(metric, ascending=ascending).reset_index(drop=True)


def choose(winners: pd.DataFrame, task: str) -> str:
    if task == "regression":
        return winners.loc[winners["mae"].idxmin(), "config"]
    contenders = winners[winners["roc_auc"] >= winners["roc_auc"].max() - TIE_AUC]
    return contenders.loc[contenders["brier"].idxmin(), "config"]


def per_fold(folds: pd.DataFrame, config: str, metric: str, fold_names: list[str]) -> pd.Series:
    rows = folds[(folds.config == config) & folds.fold.isin(fold_names)]
    return rows.set_index("fold")[metric].reindex(fold_names)


def beats(folds, candidate, reference, task, fold_names) -> dict:
    metric = "mae" if task == "regression" else "roc_auc"
    a, b = per_fold(folds, candidate, metric, fold_names), per_fold(folds, reference, metric, fold_names)
    better = (a < b) if task == "regression" else (a > b)
    return {"candidate_mean": round(float(a.mean()), 4), "reference_mean": round(float(b.mean()), 4),
            "folds_better": int(better.sum()), "folds": len(fold_names),
            "better_on_mean": bool(a.mean() < b.mean()) if task == "regression" else bool(a.mean() > b.mean())}  # fmt: skip


def ensemble_check(oof: pd.DataFrame, folds: pd.DataFrame, task: str) -> dict:
    """Members and weights from folds 1-4; scored on folds 5-8 against the best single (also chosen on 1-4)."""
    winners = family_winners(folds, task, EARLY)
    winners = winners[~winners.family.isin(["R00_official", "C00_base_rate"])]
    best_single = choose(winners, task)
    late = oof.fold.isin(LATE)
    results = {"best_single_chosen_on_folds_1_4": best_single}
    metric = "mae" if task == "regression" else "roc_auc"
    single_late = {f: score(CONFIGS[best_single], oof[oof.fold == f], oof.loc[oof.fold == f, best_single])[metric] for f in LATE}
    for size in (2, 3):
        members = winners.head(size)
        for weighting in ("equal", "validation"):
            if weighting == "equal":
                weights = np.full(size, 1 / size)
            else:
                raw = (1 / members["mae"]) if task == "regression" else members["roc_auc"]
                weights = (raw / raw.sum()).to_numpy()
            blended = sum(w * oof[c] for w, c in zip(weights, members["config"], strict=True))
            late_by_fold = {f: score(CONFIGS[best_single], oof[oof.fold == f], blended[oof.fold == f])[metric] for f in LATE}
            overall = score(CONFIGS[best_single], oof[late], blended[late])
            single_overall = score(CONFIGS[best_single], oof[late], oof.loc[late, best_single])
            wins = sum((late_by_fold[f] < single_late[f]) if task == "regression" else (late_by_fold[f] > single_late[f]) for f in LATE)
            better = overall[metric] < single_overall[metric] if task == "regression" else overall[metric] > single_overall[metric]
            results[f"top{size}_{weighting}"] = {
                "members": list(members["config"]), "weights": [round(float(w), 3) for w in weights],
                "late_folds_ensemble": overall, "late_folds_best_single": single_overall,
                "late_folds_won": f"{wins}/4", "accepted": bool(better and wins >= 3),
            }
    return results


def calibration_check(oof: pd.DataFrame, config: str) -> dict:
    y, p = oof["shortfall_1gw"].to_numpy().astype(int), oof[config].to_numpy()
    early, late = oof.fold.isin(EARLY).to_numpy(), oof.fold.isin(LATE).to_numpy()
    fitted = {"platt": _fit_platt(y[early], p[early]), "isotonic": _fit_isotonic(y[early], p[early])}
    late_brier = {"none": brier_score_loss(y[late], p[late])}
    late_brier.update({name: brier_score_loss(y[late], _apply(name, model, p[late])) for name, model in fitted.items()})
    chosen = min(late_brier, key=late_brier.get)
    final = None if chosen == "none" else (_fit_platt(y, p) if chosen == "platt" else _fit_isotonic(y, p))
    return {"late_fold_brier": {k: round(float(v), 4) for k, v in late_brier.items()}, "chosen": chosen, "model": final}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _fit_platt(y, p):
    return LogisticRegression(C=1e6).fit(_logit(p), y)


def _fit_isotonic(y, p):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)


def _apply(method: str, model, p: np.ndarray) -> np.ndarray:
    if method == "none" or model is None:
        return p
    return model.predict_proba(_logit(p))[:, 1] if method == "platt" else model.predict(p)


def thresholds_from(y: np.ndarray, p: np.ndarray) -> dict:
    recall_80, best_f1, best_t = None, -1.0, 0.5
    for t in np.unique(p)[::-1]:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        recall, precision = tp / (y == 1).sum(), tp / pred.sum()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if recall_80 is None and recall >= 0.80:
            recall_80 = t
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return {"recall_80": float(recall_80), "max_f1": float(best_t)}


# ---- evaluation on previously examined months ---------------------------------------------------


def evaluate_period(name, frame, history, chosen, frozen, calibration, thresholds, production_file, production_names) -> dict:
    saved = pd.read_csv(production_file, parse_dates=["startTime"])
    frame = frame.merge(saved[["startTime", *production_names.values()]], on="startTime").reset_index(drop=True)
    reg, cls = chosen["regression"], chosen["classification"]
    reg_pred = predictions_for(CONFIGS[reg], history, frame, fitted=frozen[reg])
    cls_raw = predictions_for(CONFIGS[cls], history, frame, fitted=frozen[cls])
    cls_cal = _apply(calibration["chosen"], calibration["model"], cls_raw)
    prod_linear = (frame["forecast_mw"] - frame[production_names["linear"]]).to_numpy()
    prod_rf = (frame["forecast_mw"] - frame[production_names["rf_reg"]]).to_numpy()
    prod_logistic, prod_rf_cls = frame[production_names["logistic"]].to_numpy(), frame[production_names["rf_cls"]].to_numpy()

    days, err, y = frame["delivery_date"], frame["error_mw"].to_numpy(), frame["shortfall_1gw"].to_numpy().astype(int)
    abs_prod, abs_new = np.abs(err - prod_linear), np.abs(err - reg_pred)
    mae_gain = block_bootstrap(days, lambda r: abs_prod[r].mean() - abs_new[r].mean(), 3)
    auc_gain = block_bootstrap(days, lambda r: roc_auc_score(y[r], cls_raw[r]) - roc_auc_score(y[r], prod_logistic[r]), 3, n_resamples=2000)
    brier_gain = block_bootstrap(days, lambda r: ((prod_logistic[r] - y[r]) ** 2).mean() - ((cls_cal[r] - y[r]) ** 2).mean(), 3)
    return {
        "period": name, "hours": len(frame), "days": int(days.nunique()),
        "regression": {"official": regression_metrics(frame, np.zeros(len(frame))),
                       "production linear_by_level (default)": regression_metrics(frame, prod_linear),
                       "production random_forest": regression_metrics(frame, prod_rf),
                       f"candidate {reg}": regression_metrics(frame, reg_pred)},
        "classification": {"production logistic_by_level (default)": classification_metrics(frame, prod_logistic),
                           "production random_forest": classification_metrics(frame, prod_rf_cls),
                           f"candidate {cls} (raw)": classification_metrics(frame, cls_raw),  # thresholds live in calibrated space
                           f"candidate {cls} ({calibration['chosen']} calibration)": classification_metrics(frame, cls_cal, thresholds)},
        "distribution": {"production linear_by_level": error_distribution(frame, prod_linear), f"candidate {reg}": error_distribution(frame, reg_pred)},
        "segments": segments(frame, {"production linear_by_level": prod_linear, f"candidate {reg}": reg_pred}, "regression"),
        "differences_vs_production_default": {
            "mae_gain_mw": {"estimate": round(float(abs_prod.mean() - abs_new.mean()), 1), "ci95": interval(mae_gain)},
            "roc_auc_gain": {"estimate": round(float(roc_auc_score(y, cls_raw) - roc_auc_score(y, prod_logistic)), 4), "ci95": interval(auc_gain)},
            "brier_gain": {"estimate": round(float(((prod_logistic - y) ** 2).mean() - ((cls_cal - y) ** 2).mean()), 4), "ci95": interval(brier_gain)},
        },
    }  # fmt: skip


# ---- main -------------------------------------------------------------------------------------


def main() -> None:
    lock = verify_protocol(PROTOCOL_FILE, PROTOCOL_LOCK)
    dev, january, february = load("dev"), load("january"), load("february")
    configs = REGRESSION + CLASSIFICATION

    # 2. Cross-validation
    parts, fold_rows = [], []
    for fold in FOLDS:
        train, valid = fold_masks(dev["delivery_date"], fold)
        target = dev.loc[valid]
        part = target[LABELS].copy().assign(fold=fold_label(fold))
        for config in configs:
            prediction = predictions_for(config, dev, target, train)
            part[config.name] = prediction
            fold_rows.append({"fold": fold_label(fold), "task": config.task, "family": config.family, "config": config.name,
                              "approach": config.approach, **score(config, target, prediction)})  # fmt: skip
        parts.append(part)
        print(f"fold {fold_label(fold)}: trained on {dev.loc[train, 'delivery_date'].min()}..{dev.loc[train, 'delivery_date'].max()}", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    all_folds = [fold_label(f) for f in FOLDS]

    # 3. Tuning and selection
    winners = {task: family_winners(folds, task, all_folds) for task in ("regression", "classification")}
    chosen = {task: choose(winners[task], task) for task in winners}

    # 4. Checks
    vs_production = {task: beats(folds, chosen[task], PRODUCTION_SPEC[task], task, all_folds) for task in chosen}
    nested = {}
    for task in chosen:
        early_choice = choose(family_winners(folds, task, EARLY), task)
        nested[task] = {"chosen_on_folds_1_4": early_choice, "same_as_full_cv_choice": early_choice == chosen[task],
                        "late_folds_vs_production_spec": beats(folds, early_choice, PRODUCTION_SPEC[task], task, LATE)}  # fmt: skip
    rolling = {}
    for task in winners:
        table = winners[task]
        for family, counterpart in STATIC_COUNTERPART.items():
            if family in set(table.family) and counterpart in set(table.family):
                rolling[family] = {"rolling_winner": table.set_index("family").loc[family, "config"],
                                   "static_counterpart": table.set_index("family").loc[counterpart, "config"],
                                   **beats(folds, table.set_index("family").loc[family, "config"],
                                           table.set_index("family").loc[counterpart, "config"], task, all_folds)}  # fmt: skip
                rolling[family]["retained"] = rolling[family]["better_on_mean"] and rolling[family]["folds_better"] >= 5
    criteria = {task: vs_production[task]["better_on_mean"] and vs_production[task]["folds_better"] >= 5
                and nested[task]["late_folds_vs_production_spec"]["better_on_mean"] for task in chosen}  # fmt: skip

    # 5. Ensembles
    ensembles = {task: ensemble_check(oof, folds, task) for task in chosen}

    # 6. Calibration and thresholds (development OOF only)
    calibration = calibration_check(oof, chosen["classification"])
    calibrated_oof = _apply(calibration["chosen"], calibration["model"], oof[chosen["classification"]].to_numpy())
    thresholds = thresholds_from(oof["shortfall_1gw"].to_numpy().astype(int), calibrated_oof)

    # 7. Freeze the selected models (refit on all development data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for task, name in chosen.items():
        config = CONFIGS[name]
        frozen[name] = None if config.kind == "rolling_bias" or (config.rolling_days and config.kind == "model") else fit(config, dev)
        joblib.dump({"config": name, "task": task, "fitted": frozen[name], "trained_deliveries": [str(dev.delivery_date.min()), str(dev.delivery_date.max())],
                     "rolling_days": config.rolling_days, "protocol_sha256": lock["sha256"],
                     **({"calibration": calibration["chosen"], "calibrator": calibration["model"], "thresholds": thresholds} if task == "classification" else {})},
                    MODELS_DIR / f"{task}_{config.family}.joblib", compress=3)  # fmt: skip

    # 8. One evaluation on the previously examined months
    names = {"linear": "corrected_linear_by_level", "rf_reg": "corrected_random_forest",
             "logistic": "probability_logistic_by_level", "rf_cls": "probability_random_forest"}  # fmt: skip
    evaluation = {
        "january": evaluate_period("January 2026 (application period; previously examined)", january,
                                   pd.concat([dev, january], ignore_index=True), chosen, frozen, calibration, thresholds,
                                   ML_DIR / "predictions" / "test_predictions.csv", names),
        "february": evaluate_period("February 2026 (previously examined; not an untouched holdout)", february,
                                    pd.concat([dev, january, february], ignore_index=True), chosen, frozen, calibration, thresholds,
                                    ML_DIR / "holdout_feb2026" / "evaluated_predictions.csv", names),
    }  # fmt: skip
    recommendation = {}
    for task, metric in (("regression", "mae_gain_mw"), ("classification", "roc_auc_gain")):
        gains = [evaluation[p]["differences_vs_production_default"][metric] for p in ("january", "february")]
        positive = all(g["estimate"] > 0 for g in gains)
        significant = all(g["ci95"][0] > 0 for g in gains)
        recommendation[task] = ("strong" if criteria[task] and positive and significant
                                else "suggestive" if criteria[task] and positive else "not_supported")  # fmt: skip

    # Reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(REPORTS_DIR / "cv_folds.csv", index=False)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False, float_format="%.4f")
    reference = {"regression": ["R00_official", "R01_linear_level", f"R02_random_forest{RF_PRODUCTION}", chosen["regression"]],
                 "classification": ["C01_logistic_level", f"C02_random_forest{RF_PRODUCTION}", chosen["classification"]]}  # fmt: skip
    write_json(REPORTS_DIR / "results.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": lock["sha256"], "dev_sha256": sha256(PROCESSED_DIR / "dev.csv"),
        "folds": all_folds, "configurations": {"regression": len(REGRESSION), "classification": len(CLASSIFICATION)},
        "family_winners_8_folds": {t: w.round(4).to_dict(orient="records") for t, w in winners.items()},
        "all_configurations_8_folds": folds.groupby(["task", "family", "config"]).mean(numeric_only=True).round(4).reset_index().to_dict(orient="records"),
        "selected": chosen, "vs_production_spec_8_folds": vs_production, "nested_check": nested,
        "rolling_vs_static": rolling, "cv_criteria_met": criteria, "ensembles": ensembles,
        "calibration": {k: v for k, v in calibration.items() if k != "model"}, "thresholds_from_dev_oof": thresholds,
        "oof_segments": {"regression": segments(oof, {c: oof[c].to_numpy() for c in reference["regression"]}, "regression"),
                         "classification": segments(oof, {c: oof[c].to_numpy() for c in reference["classification"]}, "classification")},
        "oof_distributions": {c: error_distribution(oof, oof[c]) for c in reference["regression"]},
        "evaluation_once": evaluation, "recommendation": recommendation,
    })  # fmt: skip

    print("\nRegression family winners (mean over 8 folds):")
    print(winners["regression"][["family", "config", "mae", "rmse", "bias", "mae_reduction_vs_official_pct"]].round(1).to_string(index=False))
    print("\nClassification family winners (mean over 8 folds):")
    print(winners["classification"][["family", "config", "roc_auc", "pr_auc", "brier", "ece_5_bands"]].round(4).to_string(index=False))
    print(f"\nSelected: {chosen}\nvs production spec: {vs_production}\nnested: {nested}\nrolling: {rolling}\ncriteria: {criteria}")
    print(f"ensembles: { {t: {k: (v['accepted'], v['late_folds_won']) for k, v in e.items() if isinstance(v, dict)} for t, e in ensembles.items()} }")
    print(f"calibration: {calibration['late_fold_brier']} -> {calibration['chosen']}; thresholds {thresholds}")
    for p, e in evaluation.items():
        print(f"\n{e['period']}:")
        for k, v in e["regression"].items():
            print(f"  {k:<48} MAE {v['mae']:7.1f} RMSE {v['rmse']:7.1f} bias {v['bias']:7.1f} reduction {v['mae_reduction_vs_official_pct']}%")
        for k, v in e["classification"].items():
            print(f"  {k:<48} AUC {v['roc_auc']} PR-AUC {v['pr_auc']} Brier {v['brier']} ECE {v['ece_5_bands']}")
        print(f"  differences vs production default: {e['differences_vs_production_default']}")
    print(f"\nRecommendation (protocol rule): {recommendation}")


if __name__ == "__main__":
    main()
