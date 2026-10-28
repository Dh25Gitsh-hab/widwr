"""
ML API tests.

* Wiring tests use the served model types fitted on the synthetic scenario (no files).
* Guard tests prove the service refuses features that break the 09:50 cutoff,
  models that read labels, a model placed in the wrong component, and an
  incomplete feature table.
* Artifact tests check the real models in backend/ml: the API serves exactly what
  ml/train.py evaluated on January, and the artifacts sealed for the February
  holdout; a missing or broken artifact disables only /api/ml; requests never
  train or reload a model. Skipped if the artifacts are not present.
"""

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.main import create_app
from app.models.ml_model import CLASSIFICATION_CANDIDATES, REGRESSION_CANDIDATES, ModelBundle
from app.services.feature_builder import FEATURE_COLUMNS, build_feature_table
from app.services.prediction_service import (
    CLASSIFIER_FILES,
    DATASET_FILE,
    HOLDOUT_MANIFEST_FILE,
    METRICS_FILE,
    REGRESSOR_FILES,
    PredictionService,
)

ML_DIR = Settings().ml_dir
FEBRUARY_RESULTS = ML_DIR / "holdout_feb2026" / "reports" / "holdout_metrics.json"  # offline evidence, not served
TRAINED = all((ML_DIR / f).is_file() for f in [*REGRESSOR_FILES.values(), *CLASSIFIER_FILES.values(), METRICS_FILE])
ONE_DAY = {"start": "2026-01-03", "end": "2026-01-03"}


def _fit(candidates, name: str, task: str, rows: pd.DataFrame, target) -> ModelBundle:
    candidate = next(c for c in candidates if c.name == name)
    estimator = candidate.build().fit(rows[FEATURE_COLUMNS], target)
    return ModelBundle(task, name, candidate.description, estimator, FEATURE_COLUMNS, date(2026, 1, 3), date(2026, 1, 4))


@pytest.fixture
def feature_table(forecasts, actuals) -> pd.DataFrame:
    table = build_feature_table(forecasts, actuals, date(2026, 1, 3), date(2026, 1, 4))
    return table.dropna(subset=["forecast_mw"]).reset_index(drop=True)


@pytest.fixture
def synthetic_predictor(feature_table) -> PredictionService:
    """The served model types, fitted on the scenario, where every error is exactly forecast - 10,000 MW."""
    labelled = feature_table.dropna(subset=["error_mw"])
    # The real label (error >= 1 GW) is 1 for every scenario hour. A 2.5 GW cut gives the
    # classifiers two classes to learn: 3 Jan (error 2 GW) -> 0, 4 Jan (error 3 GW) -> 1.
    stand_in_label = (labelled["error_mw"] >= 2_500).astype(int)
    return PredictionService(
        regressors={name: _fit(REGRESSION_CANDIDATES, name, "regression", labelled, labelled["error_mw"])
                    for name in ("linear_by_level", "random_forest")},
        classifiers={name: _fit(CLASSIFICATION_CANDIDATES, name, "classification", labelled, stand_in_label)
                     for name in ("logistic_by_level", "random_forest")},
        default_regressor="linear_by_level",
        default_classifier="logistic_by_level",
        metrics={},
        features=feature_table,
    )  # fmt: skip


@pytest.fixture
def client(settings, store, synthetic_predictor):
    with TestClient(create_app(settings=settings, store=store, predictor=synthetic_predictor)) as test_client:
        yield test_client


# ---- Wiring (synthetic) ----------------------------------------------------------------


def test_predictions_serve_the_default_of_each_component(client):
    body = client.get("/api/ml/predictions", params=ONE_DAY).json()
    assert (body["regressor"]["name"], body["regressor"]["role"]) == ("linear_by_level", "default")
    assert (body["classifier"]["name"], body["classifier"]["role"]) == ("logistic_by_level", "default")
    assert len(body["points"]) == 24
    # The linear correction learns error = forecast - 10,000 exactly, so it recovers the flat actual.
    assert all(p["corrected_forecast_mw"] == pytest.approx(10_000, abs=0.1) for p in body["points"])
    assert body["scores"]["official"]["mae"] == 2_000.0
    assert body["scores"]["corrected"]["mae"] == 0.0


def test_comparison_models_can_be_requested(client):
    body = client.get("/api/ml/predictions", params={"regressor": "random_forest", "classifier": "random_forest"}).json()
    assert body["regressor"]["name"] == body["classifier"]["name"] == "random_forest"
    assert body["regressor"]["role"] == body["classifier"]["role"] == "comparison"


def test_corrected_forecast_component(client):
    body = client.get("/api/ml/corrected-forecast", params=ONE_DAY).json()
    assert (body["model"]["task"], body["model"]["inputs"]) == ("regression", ["forecast_mw"])
    assert "shortfall_probability" not in body["points"][0]  # outputs stay separate
    assert body["points"][0]["corrected_forecast_mw"] == pytest.approx(10_000, abs=0.1)
    assert body["scores"]["corrected"]["mae"] == 0.0
    # Every point says which forecast run it used and the cutoff it respects.
    assert body["points"][0]["publish_time"] == "2026-01-02T09:30:00Z"
    assert body["points"][0]["decision_cutoff"] == "2026-01-02T09:50:00Z"


def test_shortfall_risk_component(client):
    body = client.get("/api/ml/shortfall-risk").json()
    assert (body["model"]["task"], body["model"]["name"]) == ("classification", "logistic_by_level")
    assert "corrected_forecast_mw" not in body["points"][0]  # outputs stay separate
    points = pd.DataFrame(body["points"])
    by_day = points.groupby(points["start_time"].str[:10])["shortfall_probability"].mean()
    assert by_day["2026-01-03"] < 0.5 < by_day["2026-01-04"]
    # Every scenario hour is a real (>= 1 GW) shortfall: one class, so ranking scores are undefined.
    assert body["scores"]["observed_rate"] == 1.0
    assert body["scores"]["roc_auc"] is None


CLASSIFIER_SELECTION = [
    pytest.param({}, "logistic_by_level", "default", id="default-logistic"),
    pytest.param({"model": "random_forest"}, "random_forest", "comparison", id="explicit-random-forest"),
]


@pytest.mark.parametrize(("params", "name", "role"), CLASSIFIER_SELECTION)
def test_each_classifier_returns_valid_probabilities(client, params, name, role):
    body = client.get("/api/ml/shortfall-risk", params=params).json()
    assert (body["model"]["name"], body["model"]["role"]) == (name, role)
    assert body["model"]["inputs"] == (["forecast_mw"] if name == "logistic_by_level" else FEATURE_COLUMNS)
    probabilities = pd.Series([p["shortfall_probability"] for p in body["points"]])
    assert len(probabilities) == 46  # every scenario hour with a forecast
    assert probabilities.between(0, 1).all()


def test_models_endpoint_lists_each_component(client):
    body = client.get("/api/ml/models").json()
    assert body["decision_cutoff_uk"] == "09:50"
    assert body["decision_dates"] == {"first": "2026-01-02", "last": "2026-01-03"}  # the scenario's decisions
    # The approval is stated with its scope, selection history and qualifications.
    assert "approved" in body["status"]
    assert "1-31 January 2026" in body["status"]
    assert "chosen later using the January test results and an offline February 2026 check" in body["status"]
    assert "not proof of long-term superiority or evidence of trading profit" in body["status"]
    roles = {component: {m["name"]: m["role"] for m in body[component]} for component in ("corrected_forecast", "shortfall_risk")}
    assert roles == {
        "corrected_forecast": {"linear_by_level": "default", "random_forest": "comparison"},
        "shortfall_risk": {"logistic_by_level": "default", "random_forest": "comparison"},
    }
    assert not any("february_holdout" in m for component in ("corrected_forecast", "shortfall_risk") for m in body[component])


def test_a_model_cannot_be_used_for_the_other_task(client):
    assert client.get("/api/ml/shortfall-risk", params={"model": "linear_by_level"}).status_code == 422
    assert client.get("/api/ml/predictions", params={"regressor": "logistic_by_level"}).status_code == 422
    assert client.get("/api/ml/corrected-forecast", params={"model": "gradient_boosting"}).status_code == 422


def test_invalid_ml_dates_are_rejected(client):
    for path in ("/api/ml/predictions", "/api/ml/corrected-forecast", "/api/ml/shortfall-risk"):
        assert client.get(path, params={"start": "2026-02-10"}).status_code == 422  # outside the window
        assert client.get(path, params={"start": "2026-01-04", "end": "2026-01-03"}).status_code == 422  # reversed
        assert client.get(path, params={"start": "2026-13-01"}).status_code == 422  # not a date


def test_a_missing_actual_is_null_and_left_out_of_the_scores(client):
    day = {"start": "2026-01-04", "end": "2026-01-04"}  # the scenario has no actual for 4 Jan 12:00
    corrected = client.get("/api/ml/corrected-forecast", params=day).json()
    hour = next(p for p in corrected["points"] if p["start_time"] == "2026-01-04T12:00:00Z")
    assert hour["actual_mw"] is None and hour["error_mw"] is None and hour["corrected_error_mw"] is None
    assert hour["corrected_forecast_mw"] is not None  # the prediction itself needs no actual

    risk = client.get("/api/ml/shortfall-risk", params=day).json()
    assert next(p for p in risk["points"] if p["start_time"] == "2026-01-04T12:00:00Z")["shortfall_occurred"] is None
    assert len(risk["points"]) == 22  # 00:00-21:00: no run covers 22:00 and 23:00
    assert risk["scores"]["hours"] == 21  # the hour without an actual is not scored


def test_ml_endpoints_return_503_without_models(settings, store):
    with TestClient(create_app(settings=settings, store=store)) as client:
        for path in ("evaluate", "models", "predictions", "corrected-forecast", "shortfall-risk"):
            assert client.get(f"/api/ml/{path}").status_code == 503
        assert client.get("/api/health").json()["ml_models"].startswith("unavailable")
        assert client.get("/api/analytics/summary").status_code == 200  # the rest of the API is unaffected


def test_default_model_setting_is_validated():
    assert Settings(ml_default_classifier="random_forest").ml_default_classifier == "random_forest"
    with pytest.raises(ValidationError):
        Settings(ml_default_classifier="linear_by_level")  # a regressor cannot be the shortfall model


# ---- Leakage and separation guards -----------------------------------------------------


def test_service_refuses_a_forecast_published_after_the_cutoff(feature_table, synthetic_predictor):
    late = feature_table.copy()
    late.loc[0, "forecast_published"] = late.loc[0, "cutoff"] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="after the decision cutoff"):
        replace(synthetic_predictor, features=late)


def test_service_refuses_features_built_for_another_cutoff(forecasts, actuals, synthetic_predictor):
    table = build_feature_table(forecasts, actuals, date(2026, 1, 3), date(2026, 1, 4), gate_closure=time(11, 0))
    with pytest.raises(ValueError, match="09:50"):
        replace(synthetic_predictor, features=table.dropna(subset=["forecast_mw"]))


def test_service_refuses_a_model_that_reads_labels(synthetic_predictor):
    peeking = replace(synthetic_predictor.regressor(), features=[*FEATURE_COLUMNS, "actual_mw"])
    with pytest.raises(ValueError, match="label columns"):
        replace(synthetic_predictor, regressors={"linear_by_level": peeking})


def test_service_refuses_a_model_in_the_wrong_component(synthetic_predictor):
    with pytest.raises(ValueError, match="not a regression model"):
        replace(synthetic_predictor, regressors={"linear_by_level": synthetic_predictor.classifier()})


def test_service_refuses_a_feature_table_missing_a_model_input(feature_table, synthetic_predictor):
    with pytest.raises(ValueError, match="missing columns"):
        replace(synthetic_predictor, features=feature_table.drop(columns="forecast_spread_mw"))


def test_service_refuses_hours_without_a_forecast(feature_table, synthetic_predictor):
    gap = feature_table.copy()
    gap.loc[0, "forecast_mw"] = float("nan")
    with pytest.raises(ValueError, match="no forecast"):
        replace(synthetic_predictor, features=gap)


# ---- Real artifacts ---------------------------------------------------------------------

requires_trained_models = pytest.mark.skipif(
    not TRAINED, reason="model artifacts missing - see backend/ml/README.md to rebuild them"
)


@pytest.fixture(scope="module")
def trained_client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def saved_january() -> pd.DataFrame:
    return pd.read_csv(ML_DIR / "predictions" / "test_predictions.csv")


@requires_trained_models
def test_evaluate_reports_selected_models(trained_client):
    body = trained_client.get("/api/ml/evaluate").json()
    assert body["regression"]["selected"] in {c["name"] for c in body["regression"]["candidates"]}
    assert [s["name"] for s in body["splits"]] == ["train", "validation", "test"]


@requires_trained_models
def test_served_defaults_match_evaluated_predictions(trained_client, saved_january):
    metrics = json.loads((ML_DIR / METRICS_FILE).read_text(encoding="utf-8"))
    body = trained_client.get("/api/ml/predictions").json()
    assert (body["regressor"]["name"], body["classifier"]["name"]) == ("linear_by_level", "logistic_by_level")

    points = pd.DataFrame(body["points"])
    assert len(points) == len(saved_january) == 744
    assert (pd.to_datetime(points["publish_time"]) <= pd.to_datetime(points["decision_cutoff"])).all()
    assert (points["corrected_forecast_mw"] - saved_january["corrected_linear_by_level"]).abs().max() < 0.2
    assert (points["shortfall_probability"] - saved_january["probability_logistic_by_level"]).abs().max() < 2e-4
    linear_test = next(c for c in metrics["regression"]["candidates"] if c["name"] == "linear_by_level")["test"]
    assert body["scores"]["corrected"]["mae"] == pytest.approx(linear_test["mae"], abs=0.2)


@requires_trained_models
def test_comparison_regressor_matches_evaluated_predictions(trained_client, saved_january):
    corrected = trained_client.get("/api/ml/corrected-forecast", params={"model": "random_forest"}).json()["points"]
    served_corrected = pd.Series([p["corrected_forecast_mw"] for p in corrected])
    assert (served_corrected - saved_january["corrected_random_forest"]).abs().max() < 0.2


@requires_trained_models
@pytest.mark.parametrize(("params", "name", "role"), CLASSIFIER_SELECTION)
def test_shortfall_classifier_selection_on_the_real_models(trained_client, saved_january, params, name, role):
    metrics = json.loads((ML_DIR / METRICS_FILE).read_text(encoding="utf-8"))
    body = trained_client.get("/api/ml/shortfall-risk", params=params).json()
    assert (body["model"]["name"], body["model"]["role"]) == (name, role)

    probabilities = pd.Series([p["shortfall_probability"] for p in body["points"]])
    assert len(probabilities) == 744
    assert probabilities.between(0, 1).all()
    # The probabilities come from the requested model: they equal its evaluated January predictions.
    assert (probabilities - saved_january[f"probability_{name}"]).abs().max() < 2e-4
    january = next(c for c in metrics["classification"]["candidates"] if c["name"] == name)["test"]
    assert body["scores"]["brier"] == pytest.approx(january["brier"], abs=1e-4)


@requires_trained_models
def test_selecting_the_random_forest_classifier_leaves_the_regressor_unchanged(trained_client, saved_january):
    body = trained_client.get("/api/ml/predictions", params={"classifier": "random_forest"}).json()
    assert (body["classifier"]["name"], body["classifier"]["role"]) == ("random_forest", "comparison")
    assert (body["regressor"]["name"], body["regressor"]["role"]) == ("linear_by_level", "default")
    probabilities = pd.Series([p["shortfall_probability"] for p in body["points"]])
    assert (probabilities - saved_january["probability_random_forest"]).abs().max() < 2e-4


@requires_trained_models
@pytest.mark.skipif(not FEBRUARY_RESULTS.is_file(), reason="February holdout not evaluated")
def test_random_forest_classifier_artifact_and_results_are_kept(trained_client):
    metrics = json.loads((ML_DIR / METRICS_FILE).read_text(encoding="utf-8"))
    served = next(m for m in trained_client.get("/api/ml/models").json()["shortfall_risk"] if m["name"] == "random_forest")

    assert (served["role"], served["artifact"]) == ("comparison", CLASSIFIER_FILES["random_forest"].as_posix())
    assert served["matches_holdout_artifact"] is True  # the same file that was evaluated, untouched
    january = next(c for c in metrics["classification"]["candidates"] if c["name"] == "random_forest")["test"]
    assert served["january_test"]["brier"] == january["brier"]
    # Its February results are kept as offline evidence (not served by the January application).
    february = json.loads(FEBRUARY_RESULTS.read_text(encoding="utf-8"))
    assert "random_forest" in february["shortfall_risk"]["models"]


@requires_trained_models
@pytest.mark.skipif(not (ML_DIR / HOLDOUT_MANIFEST_FILE).is_file(), reason="model fingerprints not sealed")
def test_models_endpoint_reports_artifacts_and_selection_history(trained_client):
    metrics = json.loads((ML_DIR / METRICS_FILE).read_text(encoding="utf-8"))
    body = trained_client.get("/api/ml/models").json()
    served = {(component, m["name"]): m for component in ("corrected_forecast", "shortfall_risk") for m in body[component]}

    assert all(m["matches_holdout_artifact"] for m in served.values())  # serving the models that were evaluated
    linear = served["corrected_forecast", "linear_by_level"]
    logistic = served["shortfall_risk", "logistic_by_level"]
    assert linear["inputs"] == logistic["inputs"] == ["forecast_mw"]
    linear_january = next(c for c in metrics["regression"]["candidates"] if c["name"] == "linear_by_level")["test"]
    assert linear["january_test"] == linear_january
    # Selection history: the rule fixed before January chose the Random Forests, not the approved defaults.
    assert {key: m["selected_by_pre_january_rule"] for key, m in served.items()} == {
        ("corrected_forecast", "linear_by_level"): False, ("corrected_forecast", "random_forest"): True,
        ("shortfall_risk", "logistic_by_level"): False, ("shortfall_risk", "random_forest"): True,
    }  # fmt: skip
    assert not any("february_holdout" in m for m in served.values())


@requires_trained_models
def test_prediction_requests_never_train_or_reload_a_model(trained_client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a prediction request tried to train or reload a model")

    for estimator in (LinearRegression, LogisticRegression, RandomForestRegressor, RandomForestClassifier, Pipeline):
        monkeypatch.setattr(estimator, "fit", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    artifacts = [ML_DIR / f for f in [*REGRESSOR_FILES.values(), *CLASSIFIER_FILES.values()]]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts]

    requests = [
        ("predictions", {}), ("predictions", {"regressor": "random_forest", "classifier": "random_forest"}),
        ("corrected-forecast", {}), ("corrected-forecast", {"model": "random_forest"}),
        ("shortfall-risk", {}), ("shortfall-risk", {"model": "random_forest"}), ("models", {}), ("evaluate", {}),
    ]  # fmt: skip
    for path, params in requests:
        assert trained_client.get(f"/api/ml/{path}", params=params).status_code == 200
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts] == before


@pytest.fixture
def ml_copy(settings) -> Path:
    """The served ML files, copied into this test's own ml folder so one can be broken."""
    for relative in [DATASET_FILE, METRICS_FILE, *REGRESSOR_FILES.values(), *CLASSIFIER_FILES.values()]:
        target = settings.ml_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ML_DIR / relative, target)
    return settings.ml_dir


@requires_trained_models
def test_an_intact_copy_of_the_ml_folder_loads(settings, store, ml_copy):  # control for the test below
    with TestClient(create_app(settings=settings, store=store)) as client:
        assert client.get("/api/ml/shortfall-risk").status_code == 200


@requires_trained_models
@pytest.mark.parametrize("damage", ["missing", "empty", "not_a_model"])
def test_a_broken_model_artifact_disables_only_the_ml_endpoints(settings, store, ml_copy, damage):
    artifact = ml_copy / CLASSIFIER_FILES["logistic_by_level"]
    if damage == "missing":
        artifact.unlink()
    elif damage == "empty":
        artifact.write_bytes(b"")
    else:
        joblib.dump({"not": "a model"}, artifact)

    with TestClient(create_app(settings=settings, store=store)) as client:
        assert client.get("/api/ml/shortfall-risk").status_code == 503
        assert client.get("/api/ml/corrected-forecast").status_code == 503
        assert artifact.name in client.get("/api/health").json()["ml_models"]  # the reason names the file
        assert client.get("/api/analytics/summary").status_code == 200  # the rest of the API is unaffected


@requires_trained_models
def test_default_model_is_a_setting(store):
    with TestClient(create_app(settings=Settings(ml_default_classifier="random_forest"), store=store)) as client:
        body = client.get("/api/ml/shortfall-risk", params={"start": "2026-01-15", "end": "2026-01-15"}).json()
    assert (body["model"]["name"], body["model"]["role"]) == ("random_forest", "default")
