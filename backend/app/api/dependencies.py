"""
Shared request dependencies (FastAPI ``Depends``).

WHY
    Every analytics endpoint needs the same three inputs: the loaded data, a
    validated delivery-date period and a forecast-selection policy. Declaring
    them once here means:
      * route functions stay short and read like the business question;
      * validation rules and error messages are identical on every endpoint;
      * tests can swap in a small fake data store.

HOW
    FastAPI calls these functions for each request, validates the query
    parameters they declare (and shows them in /docs), and passes the results
    to the route. Invalid input becomes HTTP 422 with a readable message.
"""

from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Query, Request, status

from app.config import Settings, get_settings
from app.services.data_store import DataStore
from app.services.forecast_evaluator import SelectionPolicy
from app.services.prediction_service import PredictionService
from app.utils.timestamps import parse_hhmm

HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"

GateClosureQuery = Annotated[
    str | None,
    Query(
        pattern=HHMM_PATTERN,
        description="Decision cutoff as HH:MM UK local time on the day before delivery. "
        "Default 09:50 (Nord Pool GB day-ahead gate closure); EPEX's GB auction is ~09:20.",
        examples=["09:50"],
    ),
]


def get_store(request: Request) -> DataStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        reason = getattr(request.app.state, "load_error", None) or "data is still loading"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Data unavailable: {reason}")
    return store


def get_predictor(request: Request) -> PredictionService:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        reason = getattr(request.app.state, "ml_error", None) or "models are not loaded"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"ML unavailable: {reason}")
    return predictor


StoreDep = Annotated[DataStore, Depends(get_store)]
PredictorDep = Annotated[PredictionService, Depends(get_predictor)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@dataclass(frozen=True)
class Period:
    """An inclusive range of UK delivery dates."""

    first: date
    last: date


def get_period(
    store: StoreDep,
    start: Annotated[
        date | None, Query(description="First UK delivery date (YYYY-MM-DD). Default: start of the window.")
    ] = None,
    end: Annotated[date | None, Query(description="Last UK delivery date, inclusive. Default: end of the window.")] = None,
) -> Period:
    first = start or store.first_delivery_date
    last = end or store.last_delivery_date
    if first > last:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="start must be on or before end")
    if not store.contains_delivery_range(first, last):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Delivery dates must be within {store.first_delivery_date} .. {store.last_delivery_date}",
        )
    return Period(first, last)


def get_policy(
    settings: SettingsDep,
    gate_closure: GateClosureQuery = None,
    policy: Annotated[
        Literal["day_ahead", "fixed_horizon"],
        Query(description="day_ahead: latest forecast before gate closure on D-1. "
              "fixed_horizon: latest forecast published at least `horizon_hours` before each hour."),
    ] = "day_ahead",
    horizon_hours: Annotated[float, Query(ge=0, le=48, description="Lead time for policy=fixed_horizon")] = 24,
) -> SelectionPolicy:
    closure = parse_hhmm(gate_closure) if gate_closure else settings.default_gate_closure
    return SelectionPolicy(kind=policy, gate_closure=closure, horizon_hours=horizon_hours)


def get_ml_period(
    predictor: PredictorDep,
    start: Annotated[date | None, Query(description="First UK delivery date (default: first served day)")] = None,
    end: Annotated[date | None, Query(description="Last UK delivery date, inclusive")] = None,
) -> Period:
    first, last = start or predictor.first_day, end or predictor.last_day
    if not predictor.first_day <= first <= last <= predictor.last_day:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Delivery dates must be within {predictor.first_day} .. {predictor.last_day} with start <= end",
        )
    return Period(first, last)


PeriodDep = Annotated[Period, Depends(get_period)]
PolicyDep = Annotated[SelectionPolicy, Depends(get_policy)]
MlPeriodDep = Annotated[Period, Depends(get_ml_period)]
