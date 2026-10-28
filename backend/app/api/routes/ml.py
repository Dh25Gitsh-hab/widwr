"""
ML endpoints - the optional extension on top of the forecast evaluation.

Two separate prediction components, each with its own models, inputs, outputs and scores:

GET /api/ml/corrected-forecast  component 1 (regression): corrected forecast in MW, hour by hour
GET /api/ml/shortfall-risk      component 2 (classification): probability of a >= 1 GW shortfall
GET /api/ml/predictions         both components side by side (one call for the dashboard)
GET /api/ml/models              every served model: role (default / comparison), inputs,
                                artifact fingerprint, January test scores, and the
                                selection history (which model the pre-January rule chose)
GET /api/ml/evaluate            the January model comparison exactly as ml/train.py wrote it

Each prediction endpoint takes an optional model name; the default is a setting.
The models are trained offline (backend/ml); these routes only serve them.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import MlPeriodDep, PredictorDep
from app.config import ClassifierName, RegressorName
from app.models.ml_model import PROBABILITY_THRESHOLD, classification_scores, regression_scores
from app.models.schemas import (
    DateRange,
    MlCorrectedForecastPoint,
    MlCorrectedForecastResponse,
    MlEvaluationResponse,
    MlModelInfo,
    MlModelsResponse,
    MlPeriodScores,
    MlPredictionPoint,
    MlPredictionsResponse,
    MlRiskScores,
    MlServedModel,
    MlShortfallRiskPoint,
    MlShortfallRiskResponse,
)
from app.services.prediction_service import MODEL_CUTOFF_UK, SELECTION_STATUS, PredictionService

router = APIRouter(prefix="/ml", tags=["ml"])

RegressorQuery = Annotated[
    RegressorName | None, Query(description="Corrected-forecast model. Default: the configured default (see /api/ml/models)")
]
ClassifierQuery = Annotated[
    ClassifierName | None, Query(description="Shortfall-risk model. Default: the configured default (see /api/ml/models)")
]


@router.get("/evaluate", response_model=MlEvaluationResponse, summary="January model comparison and test results")
def evaluate_models(predictor: PredictorDep) -> MlEvaluationResponse:
    return MlEvaluationResponse(**predictor.metrics)


@router.get("/models", response_model=MlModelsResponse, summary="Served models, their roles, inputs and scores")
def served_models(predictor: PredictorDep) -> MlModelsResponse:
    return MlModelsResponse(
        status=SELECTION_STATUS,
        decision_cutoff_uk=MODEL_CUTOFF_UK.strftime("%H:%M"),
        decision_dates=DateRange(first=predictor.first_day - timedelta(days=1), last=predictor.last_day - timedelta(days=1)),
        served_period=DateRange(first=predictor.first_day, last=predictor.last_day),
        corrected_forecast=[_served_model(predictor, "regression", name) for name in predictor.regressors],
        shortfall_risk=[_served_model(predictor, "classification", name) for name in predictor.classifiers],
    )


@router.get("/corrected-forecast", response_model=MlCorrectedForecastResponse, summary="Corrected forecast (MW)")
def corrected_forecast(
    predictor: PredictorDep, period: MlPeriodDep, model: RegressorQuery = None
) -> MlCorrectedForecastResponse:
    frame = predictor.corrected_forecast(period.first, period.last, model)
    return MlCorrectedForecastResponse(
        period=DateRange(first=period.first, last=period.last),
        model=_model_info(predictor, "regression", model),
        scores=_regression_scores(frame),
        points=MlCorrectedForecastPoint.from_frame(frame),
    )


@router.get("/shortfall-risk", response_model=MlShortfallRiskResponse, summary="Probability of a >= 1 GW shortfall")
def shortfall_risk(
    predictor: PredictorDep, period: MlPeriodDep, model: ClassifierQuery = None
) -> MlShortfallRiskResponse:
    frame = predictor.shortfall_risk(period.first, period.last, model)
    scored = frame.dropna(subset=["shortfall_1gw"])
    scores = None
    if not scored.empty:
        scores = MlRiskScores(
            hours=len(scored),
            probability_threshold=PROBABILITY_THRESHOLD,
            **classification_scores(scored["shortfall_1gw"], scored["shortfall_probability"].to_numpy()),
        )
    return MlShortfallRiskResponse(
        period=DateRange(first=period.first, last=period.last),
        model=_model_info(predictor, "classification", model),
        scores=scores,
        points=MlShortfallRiskPoint.from_frame(frame),
    )


@router.get("/predictions", response_model=MlPredictionsResponse, summary="Both components, hour by hour")
def predictions(
    predictor: PredictorDep,
    period: MlPeriodDep,
    regressor: RegressorQuery = None,
    classifier: ClassifierQuery = None,
) -> MlPredictionsResponse:
    frame = predictor.predict(period.first, period.last, regressor, classifier)
    return MlPredictionsResponse(
        period=DateRange(first=period.first, last=period.last),
        regressor=_model_info(predictor, "regression", regressor),
        classifier=_model_info(predictor, "classification", classifier),
        scores=_regression_scores(frame),
        points=MlPredictionPoint.from_frame(frame),
    )


def _regression_scores(frame) -> MlPeriodScores | None:
    scored = frame.dropna(subset=["actual_mw"])
    if scored.empty:
        return None
    return MlPeriodScores(
        official=regression_scores(scored["actual_mw"], scored["forecast_mw"]),
        corrected=regression_scores(scored["actual_mw"], scored["corrected_forecast_mw"]),
    )


def _model_info(predictor: PredictionService, task: str, name: str | None) -> MlModelInfo:
    default = predictor.default_regressor if task == "regression" else predictor.default_classifier
    name = name or default
    bundle = predictor.regressor(name) if task == "regression" else predictor.classifier(name)
    return MlModelInfo(
        task=task,
        name=name,
        role="default" if name == default else "comparison",
        description=bundle.description,
        inputs=bundle.inputs,
        trained_from=bundle.trained_from,
        trained_to=bundle.trained_to,
    )


def _served_model(predictor: PredictionService, task: str, name: str) -> MlServedModel:
    artifact = predictor.artifacts.get(f"{task}/{name}", {})
    return MlServedModel(
        **_model_info(predictor, task, name).model_dump(),
        artifact=artifact.get("path"),
        sha256=artifact.get("sha256"),
        matches_holdout_artifact=artifact.get("matches_holdout_artifact"),
        january_test=predictor.january_test_scores(task, name),
        selected_by_pre_january_rule=predictor.selected_before_january(task, name),
    )
