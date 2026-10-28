"""
Analytics endpoints - client question 2: "how accurate was it?"

All endpoints share the same query parameters (see dependencies.py):
    start, end        UK delivery dates (default: the whole evaluation window)
    policy            day_ahead (default) | fixed_horizon
    gate_closure      HH:MM UK time on D-1 (day_ahead)
    horizon_hours     0-48 (fixed_horizon)

GET /api/analytics/summary     headline metrics + data coverage
GET /api/analytics/timeseries  half-hourly actual, forecast and error for charts
GET /api/analytics/breakdown   metrics per hour of day / delivery day / forecast size
GET /api/analytics/horizon     metrics vs. forecast lead time (0-48 h)
"""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import PeriodDep, PolicyDep, StoreDep
from app.models.schemas import (
    AccuracyMetrics,
    BreakdownResponse,
    BreakdownRow,
    DateRange,
    HorizonResponse,
    HorizonRow,
    IntervalPoint,
    PolicyInfo,
    SummaryResponse,
    TimeseriesResponse,
)
from app.services.forecast_evaluator import (
    BreakdownKey,
    breakdown,
    compute_metrics,
    evaluate,
    horizon_profile,
    interval_table,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=SummaryResponse, summary="Headline accuracy metrics")
def summary(store: StoreDep, period: PeriodDep, policy: PolicyDep) -> SummaryResponse:
    frame = evaluate(store.forecasts, store.actuals, period.first, period.last, policy)
    return SummaryResponse(
        period=DateRange(first=period.first, last=period.last),
        policy=PolicyInfo.from_policy(policy),
        metrics=AccuracyMetrics.from_dict(compute_metrics(frame)),
    )


@router.get("/timeseries", response_model=TimeseriesResponse, summary="Half-hourly forecast vs actual")
def timeseries(store: StoreDep, period: PeriodDep, policy: PolicyDep) -> TimeseriesResponse:
    frame = evaluate(store.forecasts, store.actuals, period.first, period.last, policy)
    table = interval_table(frame, store.actuals, period.first, period.last)
    return TimeseriesResponse(
        period=DateRange(first=period.first, last=period.last),
        policy=PolicyInfo.from_policy(policy),
        points=IntervalPoint.from_frame(table),
    )


@router.get("/breakdown", response_model=BreakdownResponse, summary="Accuracy by hour, day or forecast size")
def accuracy_breakdown(
    store: StoreDep,
    period: PeriodDep,
    policy: PolicyDep,
    by: Annotated[BreakdownKey, Query(description="Grouping dimension")] = "hour_of_day",
) -> BreakdownResponse:
    frame = evaluate(store.forecasts, store.actuals, period.first, period.last, policy)
    return BreakdownResponse(
        period=DateRange(first=period.first, last=period.last),
        policy=PolicyInfo.from_policy(policy),
        by=by,
        rows=[BreakdownRow(**row) for row in breakdown(frame, by)],
    )


@router.get("/horizon", response_model=HorizonResponse, summary="Accuracy vs forecast lead time")
def horizon(
    store: StoreDep,
    period: PeriodDep,
    max_hours: Annotated[int, Query(ge=1, le=48, description="Longest lead time to evaluate")] = 48,
    step_hours: Annotated[int, Query(ge=1, le=12, description="Spacing between lead times")] = 1,
) -> HorizonResponse:
    rows = horizon_profile(store.forecasts, store.actuals, period.first, period.last, range(0, max_hours + 1, step_hours))
    return HorizonResponse(
        period=DateRange(first=period.first, last=period.last),
        rows=[HorizonRow(**row) for row in rows],
    )
