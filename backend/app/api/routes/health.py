"""
GET /api/health

WHY: the first thing to call when something looks wrong (locally, in Docker
or after deployment). Besides "is the server up?" it answers "which data is
it serving, and how much was cleaned away?" - the data audit trail.
"""

from fastapi import APIRouter, Request, Response, status

from app.api.dependencies import SettingsDep
from app.models.schemas import CleaningReportOut, DateRange, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and data status")
def health(request: Request, response: Response, settings: SettingsDep) -> HealthResponse:
    store = getattr(request.app.state, "store", None)
    if store is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", detail=getattr(request.app.state, "load_error", None))

    return HealthResponse(
        status="ok",
        data_source=store.source,
        snapshot_fetched_at=store.snapshot_fetched_at,
        loaded_at=store.loaded_at,
        decision_dates=DateRange(first=settings.first_decision_date, last=settings.last_decision_date),
        delivery_dates=DateRange(first=store.first_delivery_date, last=store.last_delivery_date),
        actual_intervals=len(store.actuals),
        forecast_values=len(store.forecasts),
        forecast_runs=store.forecast_runs,
        cleaning=[CleaningReportOut(**report.as_dict()) for report in store.cleaning],
        ml_models=_ml_status(request),
    )


def _ml_status(request: Request) -> str:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return f"unavailable: {getattr(request.app.state, 'ml_error', None) or 'not loaded'}"
    def listing(models: dict, default: str) -> str:
        return " + ".join(f"{name} (default)" if name == default else name for name in models)

    return (
        f"loaded: corrected_forecast={listing(predictor.regressors, predictor.default_regressor)}, "
        f"shortfall_risk={listing(predictor.classifiers, predictor.default_classifier)}"
    )
