"""
Forecast evaluation - the business logic of the product.

It answers the client's two questions:
    Q1  "Which forecast did we actually have when we had to buy?"
        -> select_forecasts(), day_ahead_view()
    Q2  "How wrong was it, and how wrong is it typically?"
        -> evaluate(), compute_metrics(), breakdown(), horizon_profile()

THE KEY IDEA - no look-ahead
    WINDFOR holds many forecasts ("runs") for the same target hour, one per
    publishTime. For every target hour t we first compute a decision cutoff
    c(t). Only runs with publishTime <= c(t) are eligible, and we take the
    most recent one. Choosing a later run would score the forecast with
    information the trader did not have - the numbers would look better than
    reality ("look-ahead bias").

    pandas.merge_asof(by="startTime", direction="backward") does exactly this
    in one vectorised step: for each (startTime, cutoff) row it finds the last
    forecast row with the same startTime whose publishTime <= cutoff.

TWO CUTOFF POLICIES
    day_ahead (default)  c(t) = gate closure (UK time) on the day before t's
                         delivery day - mirrors the real purchase process.
    fixed_horizon(h)     c(t) = t - h hours - answers "how accurate are
                         forecasts made h hours ahead?"

RESOLUTION
    WINDFOR is hourly and FUELHH half-hourly. We score each hourly forecast
    against the actual with the same startTime (the brief names startTime as
    the shared key). Scoring against the mean of the two half-hours instead
    changes MAE by under 1% on the January data, so the choice is immaterial.

SIGN CONVENTION (used everywhere)
    error = forecast - actual   [MW]
      > 0  over-forecast  -> less wind than planned -> the buyer ends up SHORT
      < 0  under-forecast -> more wind than planned -> the buyer ends up LONG
"""

from dataclasses import dataclass
from datetime import date, time
from typing import Literal

import numpy as np
import pandas as pd

from app.utils.timestamps import (
    gate_closure_utc,
    hourly_targets_utc,
    uk_day_bounds_utc,
    uk_local_date,
    uk_local_hour,
)

PolicyKind = Literal["day_ahead", "fixed_horizon"]
BreakdownKey = Literal["hour_of_day", "delivery_date", "forecast_level"]

# Forecast-size bands: wind forecast error is known to depend on how much wind is forecast.
FORECAST_LEVEL_BINS = [0, 5_000, 10_000, 15_000, float("inf")]
FORECAST_LEVEL_LABELS = ["0-5 GW", "5-10 GW", "10-15 GW", "15+ GW"]

_ERROR_METRIC_KEYS = (
    "mae", "rmse", "bias", "median_abs_error", "p90_abs_error",
    "error_p10", "error_p50", "error_p90",
    "max_abs_error", "max_error", "max_error_time",
    "mean_actual", "mean_forecast", "nmae_pct", "over_forecast_pct",
)  # fmt: skip


# ---------------------------------------------------------------------------
# Q1 - which forecast was available at decision time?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionPolicy:
    """Rule that decides, for each target hour, the moment after which forecasts no longer count."""

    kind: PolicyKind = "day_ahead"
    gate_closure: time = time(9, 50)  # used by day_ahead (UK local time on D-1)
    horizon_hours: float = 24.0  # used by fixed_horizon

    def cutoffs(self, targets: pd.Series) -> pd.Series:
        if self.kind == "day_ahead":
            delivery_days = uk_local_date(targets)
            cutoff_by_day = {day: gate_closure_utc(day, self.gate_closure) for day in delivery_days.unique()}
            return pd.to_datetime(delivery_days.map(cutoff_by_day), utc=True).dt.as_unit("ns")
        return targets - pd.Timedelta(hours=self.horizon_hours)

    def describe(self) -> str:
        if self.kind == "day_ahead":
            return f"Latest forecast published by {self.gate_closure:%H:%M} UK time on the day before delivery"
        return f"Latest forecast published at least {self.horizon_hours:g} h before each target hour"


def select_forecasts(forecasts: pd.DataFrame, targets: pd.Series, policy: SelectionPolicy) -> pd.DataFrame:
    """
    For each target hour, return the forecast that was available at its cutoff.

    Returns columns: startTime, cutoff, publishTime, forecast, horizon_hours.
    publishTime/forecast are empty (NaT/NaN) when no run had been published
    before the cutoff - reported later as "missing forecast", never as zero.
    """
    frame = pd.DataFrame({"startTime": targets.reset_index(drop=True)})
    frame["cutoff"] = policy.cutoffs(frame["startTime"])
    picked = latest_forecast_as_of(forecasts, frame).rename(columns={"generation": "forecast"})
    picked["horizon_hours"] = (picked["startTime"] - picked["publishTime"]).dt.total_seconds() / 3600
    return picked


def latest_forecast_as_of(forecasts: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    """
    The rule itself: for each (startTime, cutoff) request row, the forecast for
    that startTime with the latest publishTime <= cutoff.

    Returns the request rows (sorted by startTime) plus publishTime and
    generation - empty when nothing had been published by the cutoff.
    Also used by the ML feature builder, so both apply the identical rule.
    """
    candidates = forecasts[["startTime", "publishTime", "generation"]].sort_values("publishTime")
    picked = pd.merge_asof(
        requests.sort_values("cutoff"),
        candidates,
        left_on="cutoff",
        right_on="publishTime",
        by="startTime",  # only forecasts for the same target hour
        direction="backward",  # publishTime <= cutoff, take the latest
        allow_exact_matches=True,  # published exactly at the cutoff still counts
    )
    return picked.sort_values("startTime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Q2 - how accurate was it?
# ---------------------------------------------------------------------------


def evaluate(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    first_day: date,
    last_day: date,
    policy: SelectionPolicy,
) -> pd.DataFrame:
    """One row per target hour of the delivery days: decision-time forecast, actual and error."""
    frame = select_forecasts(forecasts, hourly_targets_utc(first_day, last_day), policy)
    frame["actual"] = frame["startTime"].map(actuals.set_index("startTime")["generation"])
    frame["error"] = frame["forecast"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    return frame


def compute_metrics(frame: pd.DataFrame) -> dict:
    """
    Headline accuracy metrics for an evaluation frame.

    MAE    typical size of the miss (MW) - the number a trader remembers
    RMSE   like MAE but punishes big misses more (big misses cost the most)
    bias   mean signed error - a systematic over/under-forecast
    P10/P90 of the signed error - in 80% of hours the error fell inside this
           range, which is the uncertainty band a buyer can plan around
    nMAE   MAE as % of mean actual - comparable across periods
    MAPE is deliberately not used: it explodes when actual wind is near zero.
    """
    total = len(frame)
    has_forecast = frame["forecast"].notna()
    has_actual = frame["actual"].notna()
    scored = frame.loc[has_forecast & has_actual]

    metrics: dict = {
        "intervals_total": total,
        "intervals_scored": len(scored),
        "coverage_pct": _round(100 * len(scored) / total) if total else 0.0,
        "missing_forecast": int((~has_forecast).sum()),
        "missing_actual": int((~has_actual).sum()),
    }
    if scored.empty:
        return metrics | dict.fromkeys(_ERROR_METRIC_KEYS)

    errors = scored["error"]
    mae = errors.abs().mean()
    mean_actual = scored["actual"].mean()
    worst = scored.loc[scored["abs_error"].idxmax()]
    return metrics | {
        "mae": _round(mae),
        "rmse": _round(np.sqrt(np.mean(errors**2))),
        "bias": _round(errors.mean()),
        "median_abs_error": _round(errors.abs().median()),
        "p90_abs_error": _round(errors.abs().quantile(0.9)),
        "error_p10": _round(errors.quantile(0.1)),
        "error_p50": _round(errors.quantile(0.5)),
        "error_p90": _round(errors.quantile(0.9)),
        "max_abs_error": _round(worst["abs_error"]),
        "max_error": _round(worst["error"]),
        "max_error_time": worst["startTime"],
        "mean_actual": _round(mean_actual),
        "mean_forecast": _round(scored["forecast"].mean()),
        "nmae_pct": _round(100 * mae / mean_actual) if mean_actual else None,
        "over_forecast_pct": _round(100 * (errors > 0).mean()),
    }


def breakdown(frame: pd.DataFrame, by: BreakdownKey) -> list[dict]:
    """
    Accuracy per group, to show *when* the forecast is weak:
      hour_of_day     UK clock hour - is the morning ramp worse than the night?
      delivery_date   one row per day - which days hurt the most?
      forecast_level  size of the forecast - are windy days less reliable?
    """
    scored = frame.dropna(subset=["error"])
    if by == "hour_of_day":
        keys = uk_local_hour(scored["startTime"]).map(lambda hour: f"{hour:02d}:00")
    elif by == "delivery_date":
        keys = uk_local_date(scored["startTime"]).map(date.isoformat)
    elif by == "forecast_level":
        keys = pd.cut(scored["forecast"], bins=FORECAST_LEVEL_BINS, labels=FORECAST_LEVEL_LABELS, right=False)
    else:
        raise ValueError(f"unknown breakdown: {by}")

    return [
        {"group": str(key), **_error_stats(group)}
        for key, group in scored.groupby(keys, observed=True, sort=True)
    ]


def horizon_profile(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    first_day: date,
    last_day: date,
    horizons: range = range(0, 49),
) -> list[dict]:
    """
    Accuracy as a function of how far ahead the forecast was made.

    For each h we re-run the selection with the fixed_horizon policy, i.e.
    "the latest forecast published at least h hours before the target".
    Runs are published every 2-4 hours, so the real lead time is often a
    little longer than h - reported as mean_lead_hours.
    """
    rows = []
    for hours in horizons:
        frame = evaluate(forecasts, actuals, first_day, last_day, SelectionPolicy("fixed_horizon", horizon_hours=hours))
        scored = frame.dropna(subset=["error"])
        rows.append(
            {
                "horizon_hours": hours,
                "mean_lead_hours": _round(scored["horizon_hours"].mean()) if len(scored) else None,
                **_error_stats(scored),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Views assembled for the API
# ---------------------------------------------------------------------------


def interval_table(frame: pd.DataFrame, actuals: pd.DataFrame, first_day: date, last_day: date) -> pd.DataFrame:
    """
    Half-hourly rows for charting: every actual value, plus the decision-time
    forecast and error on the hours where a forecast exists.
    """
    start, _ = uk_day_bounds_utc(first_day)
    _, end = uk_day_bounds_utc(last_day)
    grid = pd.DataFrame({"startTime": pd.date_range(start, end, freq="30min", inclusive="left").as_unit("ns")})
    grid["actual"] = grid["startTime"].map(actuals.set_index("startTime")["generation"])
    columns = ["startTime", "forecast", "error", "publishTime", "horizon_hours"]
    return grid.merge(frame[columns], on="startTime", how="left")


def day_ahead_view(forecasts: pd.DataFrame, actuals: pd.DataFrame, delivery_day: date, gate_closure: time) -> dict:
    """
    Everything the trader needs for one delivery day:
      * the decision cutoff and which forecast run was the latest before it;
      * that forecast, hour by hour (what they could plan on);
      * the actual outturn and error (hindsight, once the day is over);
      * every other run for the day, flagged available/not available at the
        cutoff, to show how the forecast moved after the decision.
    """
    policy = SelectionPolicy("day_ahead", gate_closure=gate_closure)
    cutoff = gate_closure_utc(delivery_day, gate_closure)
    decision = evaluate(forecasts, actuals, delivery_day, delivery_day, policy)
    used_runs = set(decision["publishTime"].dropna())

    start, end = uk_day_bounds_utc(delivery_day)
    day_forecasts = forecasts.loc[(forecasts["startTime"] >= start) & (forecasts["startTime"] < end)]
    runs = [
        {
            "publish_time": publish_time,
            "available_at_decision": bool(publish_time <= cutoff),
            "used_for_decision": publish_time in used_runs,
            "points": run[["startTime", "generation"]],
        }
        for publish_time, run in day_forecasts.groupby("publishTime", sort=True)
    ]

    planned = decision.dropna(subset=["forecast"])
    expected = None
    if not planned.empty:
        low, high = planned.loc[planned["forecast"].idxmin()], planned.loc[planned["forecast"].idxmax()]
        expected = {
            "mean_mw": _round(planned["forecast"].mean()),
            "min_mw": _round(low["forecast"]),
            "min_time": low["startTime"],
            "max_mw": _round(high["forecast"]),
            "max_time": high["startTime"],
        }

    has_actuals = decision["actual"].notna().any()
    return {
        "delivery_date": delivery_day,
        "cutoff": cutoff,
        "policy": policy,
        "forecast_run_used": max(used_runs) if used_runs else None,
        "intervals": interval_table(decision, actuals, delivery_day, delivery_day),
        "expected": expected,
        "hindsight": compute_metrics(decision) if has_actuals else None,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_stats(scored: pd.DataFrame) -> dict:
    """Compact metrics for one group of scored rows."""
    if scored.empty:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "error_p10": None, "error_p90": None,
                "mean_actual": None, "mean_forecast": None}  # fmt: skip
    errors = scored["error"]
    return {
        "n": len(scored),
        "mae": _round(errors.abs().mean()),
        "rmse": _round(np.sqrt(np.mean(errors**2))),
        "bias": _round(errors.mean()),
        "error_p10": _round(errors.quantile(0.1)),
        "error_p90": _round(errors.quantile(0.9)),
        "mean_actual": _round(scored["actual"].mean()),
        "mean_forecast": _round(scored["forecast"].mean()),
    }


def _round(value: float, digits: int = 1) -> float | None:
    """Round for display; NaN becomes None (JSON null) instead of an invalid JSON number."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)
