"""
Prediction service - serves the trained ML models through the API.

WHY
    Training is an offline, reproducible pipeline (backend/ml). The API only
    LOADS what the pipeline saved, so:
      * requests are fast (no training in the request path);
      * the model served is exactly the model that was evaluated;
      * a missing or broken model never takes the rest of the API down.

TWO SEPARATE PREDICTION COMPONENTS - one task each; no model does both
    corrected forecast  regression: the official forecast minus its predicted error (MW)
        default     linear_by_level    holdout_feb2026/models/regression_linear_by_level.joblib
        comparison  random_forest      models/error_regressor.joblib
    shortfall risk      classification: probability that actual <= forecast - 1 GW
        default     logistic_by_level  holdout_feb2026/models/classification_logistic_by_level.joblib
        comparison  random_forest      models/shortfall_classifier.joblib
    SELECTION HISTORY (disclosed by GET /api/ml/models)
      1. The rule fixed before January (lowest December validation error) chose
         the Random Forests. All models were trained on data up to 31 Dec 2025.
      2. The defaults were chosen later, using the January test results and an
         offline February 2026 check, and approved on 2026-10-26. So the CHOICE
         reflects evidence from after the January decisions; the prediction
         inputs and model weights do not. The Random Forests stay loaded for
         comparison. The default per task is a setting (WFM_ML_DEFAULT_*).

SCOPE - the application covers purchase decisions made 1-31 January 2026 only.
    February 2026 data and results are offline evaluation evidence: this service
    never loads them (it reads only the sealed manifest's artifact fingerprints).

HOW
    ml/processed/ml_dataset.csv   leakage-free features of the January test days
                                  (an "offline feature store": features are computed
                                  once by the pipeline, with the 09:50 cutoff rule)
    ml/reports/metrics.json       January evaluation             -> GET /api/ml/evaluate
    ml/holdout_feb2026/sealed_manifest.json
                                  artifact fingerprints          -> GET /api/ml/models

GUARDS - checked whenever a service is built; a failure disables only /api/ml
    * every served row was built for a 09:50 UK cutoff on D-1 (the cutoff the
      models were trained and evaluated with), from a forecast published by then;
    * no model reads a label column (actual_mw, error_mw, shortfall_1gw);
    * the feature table has every column the models need and a forecast for
      every served hour - so a bad table fails at start-up, not on a request.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path

import pandas as pd

from app.config import ClassifierName, RegressorName
from app.models.ml_model import ModelBundle, load_bundle
from app.services.feature_builder import ID_COLUMNS, LABEL_COLUMNS
from app.utils.timestamps import gate_closure_utc

REGRESSOR_FILES = {
    "linear_by_level": Path("holdout_feb2026") / "models" / "regression_linear_by_level.joblib",
    "random_forest": Path("models") / "error_regressor.joblib",
}
CLASSIFIER_FILES = {
    "logistic_by_level": Path("holdout_feb2026") / "models" / "classification_logistic_by_level.joblib",
    "random_forest": Path("models") / "shortfall_classifier.joblib",
}
METRICS_FILE = Path("reports") / "metrics.json"
HOLDOUT_MANIFEST_FILE = Path("holdout_feb2026") / "sealed_manifest.json"
DATASET_FILE = Path("processed") / "ml_dataset.csv"
SERVED_SPLIT = "test"  # January 2026 decisions - the product's scope, unseen during training
MODEL_CUTOFF_UK = time(9, 50)  # the gate closure every model was trained and evaluated with
SELECTION_STATUS = (
    "Scope: purchase decisions made 1-31 January 2026. Defaults approved on 2026-10-26 for the current "
    "application. Selection history: the rule fixed before January (lowest December validation error) "
    "chose the Random Forests; the defaults were chosen later using the January test results and an "
    "offline February 2026 check, so the choice reflects evidence from after the January decisions. "
    "Prediction inputs and model weights use only data available at each 09:50 UK cutoff. The scores "
    "measure forecast accuracy only: they are not proof of long-term superiority or evidence of "
    "trading profit."
)


@dataclass
class PredictionService:
    regressors: dict[str, ModelBundle]  # corrected-forecast component, by model name
    classifiers: dict[str, ModelBundle]  # shortfall-risk component, by model name
    default_regressor: str
    default_classifier: str
    metrics: dict  # January evaluation (metrics.json)
    features: pd.DataFrame  # one row per delivery hour, feature + label columns
    artifacts: dict[str, dict] = field(default_factory=dict)  # "task/name" -> path, sha256, holdout match

    def __post_init__(self) -> None:
        for task, models, default in [("regression", self.regressors, self.default_regressor),
                                      ("classification", self.classifiers, self.default_classifier)]:  # fmt: skip
            if default not in models:
                raise ValueError(f"default {task} model '{default}' is not loaded")
            for name, bundle in models.items():
                if bundle.task != task:
                    raise ValueError(f"'{name}' is a {bundle.task} model, not a {task} model")
                leaked = set(bundle.features) & set(LABEL_COLUMNS)
                if leaked:
                    raise ValueError(f"{task} model '{name}' reads label columns {sorted(leaked)}")
        needed = {*ID_COLUMNS, *LABEL_COLUMNS}.union(*(b.features for b in [*self.regressors.values(), *self.classifiers.values()]))
        missing = needed - set(self.features.columns)
        if missing:
            raise ValueError(f"ML feature table is missing columns {sorted(missing)}")
        if self.features["forecast_mw"].isna().any():
            raise ValueError("ML feature table has hours with no forecast available at the cutoff")
        _check_cutoff(self.features)

    @classmethod
    def load(
        cls,
        ml_dir: Path,
        default_regressor: RegressorName = "linear_by_level",
        default_classifier: ClassifierName = "logistic_by_level",
    ) -> "PredictionService":
        ml_dir = Path(ml_dir)
        table = pd.read_csv(ml_dir / DATASET_FILE)
        table = table.loc[table["split"] == SERVED_SPLIT].copy()
        if table.empty:
            raise ValueError(f"no '{SERVED_SPLIT}' rows in {ml_dir / DATASET_FILE}")
        for column in ("startTime", "cutoff", "forecast_published"):
            table[column] = pd.to_datetime(table[column], utc=True)
        table["delivery_date"] = pd.to_datetime(table["delivery_date"]).dt.date

        manifest = _read_json(ml_dir / HOLDOUT_MANIFEST_FILE) or {}
        sealed = {key: entry["sha256"] for key, entry in manifest.get("models", {}).items()}
        regressors, regressor_artifacts = _load_models(ml_dir, REGRESSOR_FILES, "regression", sealed)
        classifiers, classifier_artifacts = _load_models(ml_dir, CLASSIFIER_FILES, "classification", sealed)
        return cls(
            regressors=regressors,
            classifiers=classifiers,
            default_regressor=default_regressor,
            default_classifier=default_classifier,
            metrics=json.loads((ml_dir / METRICS_FILE).read_text(encoding="utf-8")),
            features=table.sort_values("startTime").reset_index(drop=True),
            artifacts=regressor_artifacts | classifier_artifacts,
        )

    @property
    def first_day(self) -> date:
        return self.features["delivery_date"].min()

    @property
    def last_day(self) -> date:
        return self.features["delivery_date"].max()

    def regressor(self, name: str | None = None) -> ModelBundle:
        return self.regressors[name or self.default_regressor]

    def classifier(self, name: str | None = None) -> ModelBundle:
        return self.classifiers[name or self.default_classifier]

    def corrected_forecast(self, first_day: date, last_day: date, model: str | None = None) -> pd.DataFrame:
        """Component 1: the corrected forecast (MW) for every hour of the delivery days."""
        rows = self._rows(first_day, last_day)
        result = rows[["startTime", "forecast_published", "cutoff", "forecast_mw", "actual_mw", "error_mw"]].copy()
        result["corrected_forecast_mw"] = rows["forecast_mw"] - self.regressor(model).predict_error(rows)
        result["corrected_error_mw"] = result["corrected_forecast_mw"] - rows["actual_mw"]
        return result.reset_index(drop=True)

    def shortfall_risk(self, first_day: date, last_day: date, model: str | None = None) -> pd.DataFrame:
        """Component 2: the probability of a >= 1 GW shortfall for every hour of the delivery days."""
        rows = self._rows(first_day, last_day)
        result = rows[["startTime", "forecast_published", "cutoff", "forecast_mw", "shortfall_1gw"]].copy()
        result["shortfall_probability"] = self.classifier(model).predict_probability(rows)
        return result.reset_index(drop=True)

    def predict(
        self, first_day: date, last_day: date, regressor: str | None = None, classifier: str | None = None
    ) -> pd.DataFrame:
        """Both components side by side, hour by hour (each still computed by its own model)."""
        risk = self.shortfall_risk(first_day, last_day, classifier)
        return self.corrected_forecast(first_day, last_day, regressor).assign(
            shortfall_1gw=risk["shortfall_1gw"], shortfall_probability=risk["shortfall_probability"]
        )

    def january_test_scores(self, task: str, name: str) -> dict | None:
        """One model's January 2026 test scores, as recorded by ml/train.py."""
        candidates = self.metrics.get(task, {}).get("candidates", [])
        return next((c["test"] for c in candidates if c["name"] == name), None)

    def selected_before_january(self, task: str, name: str) -> bool:
        """Did the selection rule fixed before January (December validation) choose this model?"""
        return self.metrics.get(task, {}).get("selected") == name

    def _rows(self, first_day: date, last_day: date) -> pd.DataFrame:
        return self.features.loc[self.features["delivery_date"].between(first_day, last_day)]


def _load_models(ml_dir: Path, files: dict[str, Path], task: str, sealed: dict) -> tuple[dict, dict]:
    """Load one component's models, fingerprinting each artifact against the holdout seal."""
    bundles, artifacts = {}, {}
    for name, relative in files.items():
        path = ml_dir / relative
        bundles[name] = load_bundle(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"{task}/{name}"
        artifacts[key] = {
            "path": relative.as_posix(),
            "sha256": digest,
            "matches_holdout_artifact": digest == sealed[key] if key in sealed else None,
        }
    return bundles, artifacts


def _check_cutoff(features: pd.DataFrame) -> None:
    """Refuse features that were not built for the 09:50 UK cutoff, or that use a later forecast."""
    expected = pd.to_datetime(features["delivery_date"].map(lambda day: gate_closure_utc(day, MODEL_CUTOFF_UK)), utc=True)
    cutoff = pd.to_datetime(features["cutoff"], utc=True)
    if (cutoff != expected).any():
        raise ValueError("ML features were not built with the 09:50 UK decision cutoff")
    if (pd.to_datetime(features["forecast_published"], utc=True) > cutoff).any():
        raise ValueError("ML features use a forecast published after the decision cutoff")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
