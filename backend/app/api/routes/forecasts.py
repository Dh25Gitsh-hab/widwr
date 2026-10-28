"""
Forecast endpoints - client question 1: "what forecast did we have?"

GET /api/forecasts            raw forecast records (every run) for a period,
                              optionally only those published before a moment
GET /api/forecasts/day-ahead  the decision view for one delivery day

Routes stay thin: validate input (via dependencies), call a service, convert
the result into a response schema. No business logic lives here.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import GateClosureQuery, PeriodDep, SettingsDep, StoreDep
from app.models.schemas import (
    AccuracyMetrics,
    DateRange,
    DayAheadResponse,
    ExpectedProfile,
    ForecastRecord,
    ForecastRun,
    ForecastsResponse,
    IntervalPoint,
    PolicyInfo,
    RunPoint,
)
from app.services.forecast_evaluator import day_ahead_view
from app.utils.timestamps import parse_hhmm, uk_day_bounds_utc

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("", response_model=ForecastsResponse, summary="Forecast records for a delivery period")
def list_forecasts(
    store: StoreDep,
    period: PeriodDep,
    published_before: Annotated[
        datetime | None,
        Query(description="Only runs published at or before this moment (ISO-8601, UTC if no offset) - "
              "i.e. 'what was known at that time'."),
    ] = None,
) -> ForecastsResponse:
    start, _ = uk_day_bounds_utc(period.first)
    _, end = uk_day_bounds_utc(period.last)
    forecasts = store.forecasts
    selected = forecasts.loc[(forecasts["startTime"] >= start) & (forecasts["startTime"] < end)]
    if published_before is not None:
        moment = published_before if published_before.tzinfo else published_before.replace(tzinfo=timezone.utc)
        selected = selected.loc[selected["publishTime"] <= pd.Timestamp(moment)]

    records = [
        ForecastRecord(
            start_time=row.startTime.to_pydatetime(),
            publish_time=row.publishTime.to_pydatetime(),
            generation_mw=row.generation,
            horizon_hours=row.horizon_hours,
        )
        for row in selected.itertuples(index=False)
    ]
    return ForecastsResponse(
        period=DateRange(first=period.first, last=period.last),
        published_before=published_before,
        count=len(records),
        records=records,
    )


@router.get("/day-ahead", response_model=DayAheadResponse, summary="Decision view for one delivery day")
def day_ahead(
    store: StoreDep,
    settings: SettingsDep,
    delivery_date: Annotated[date, Query(description="UK delivery day (D). The decision is made on D-1.")],
    gate_closure: GateClosureQuery = None,
) -> DayAheadResponse:
    if not store.contains_delivery_range(delivery_date, delivery_date):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"delivery_date must be within {store.first_delivery_date} .. {store.last_delivery_date}",
        )
    closure = parse_hhmm(gate_closure) if gate_closure else settings.default_gate_closure
    view = day_ahead_view(store.forecasts, store.actuals, delivery_date, closure)

    run_used = view["forecast_run_used"]
    return DayAheadResponse(
        delivery_date=delivery_date,
        decision_date=delivery_date - timedelta(days=1),
        decision_cutoff=view["cutoff"].to_pydatetime(),
        forecast_run_used=run_used.to_pydatetime() if run_used is not None else None,
        minutes_before_cutoff=(view["cutoff"] - run_used).total_seconds() / 60 if run_used is not None else None,
        policy=PolicyInfo.from_policy(view["policy"]),
        expected=ExpectedProfile(**view["expected"]) if view["expected"] else None,
        hindsight=AccuracyMetrics.from_dict(view["hindsight"]) if view["hindsight"] else None,
        intervals=IntervalPoint.from_frame(view["intervals"]),
        runs=[
            ForecastRun(
                publish_time=run["publish_time"].to_pydatetime(),
                available_at_decision=run["available_at_decision"],
                used_for_decision=run["used_for_decision"],
                points=[
                    RunPoint(start_time=point.startTime.to_pydatetime(), generation_mw=point.generation)
                    for point in run["points"].itertuples(index=False)
                ],
            )
            for run in view["runs"]
        ],
    )
