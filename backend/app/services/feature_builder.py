"""
Feature builder - turns clean forecasts and actuals into a leakage-free ML table.

WHY a separate module
    A model is only honest if every input was genuinely known when the
    purchase decision had to be made. All feature logic lives here, built on
    the same selection rule the API uses (``latest_forecast_as_of``), so that
    guarantee can be reviewed in one place and is covered by tests that
    change "future" data and check that no feature moves.

ONE ROW PER DELIVERY HOUR t.  Cutoff c = gate closure on D-1 (09:50 UK time).

FEATURES - each uses only information published at or before c
    forecast_mw           latest forecast for t published <= c (the official number)
    lead_hours            t minus that forecast's publish time
    hour_uk               UK clock hour of t
    forecast_revision_mw  that forecast minus the previous run's forecast for t
                          (is the forecast still moving?)
    forecast_spread_mw    std-dev of every forecast for t published in the 24 h
                          before c (disagreement between runs = uncertainty)
    forecast_ramp_mw      forecast(t) - forecast(t - 1 h) in the same run
                          (steep ramps are harder to predict)
    forecast_day_mean_mw  mean forecast over the whole delivery day (windy day?)
    latest_actual_mw      the most recent actual generation already published at c
    recent_error_mw       mean day-ahead error over the last 24 h whose actuals
                          were published by c ("how wrong have we been lately?")

LABELS - training targets only, never inputs
    actual_mw, error_mw = forecast_mw - actual_mw, shortfall_1gw = error_mw >= 1,000 MW
"""

from datetime import date, time, timedelta

import pandas as pd

from app.services.forecast_evaluator import SelectionPolicy, evaluate, latest_forecast_as_of
from app.utils.timestamps import uk_local_date, uk_local_hour

SHORTFALL_THRESHOLD_MW = 1_000.0
RECENT_WINDOW = pd.Timedelta(hours=24)

FEATURE_DESCRIPTIONS = {
    "forecast_mw": "Official forecast for the hour, latest published by the cutoff (MW)",
    "lead_hours": "Hours between that forecast's publication and the target hour",
    "hour_uk": "UK clock hour of the target hour (0-23)",
    "forecast_revision_mw": "Change versus the previous forecast run for the same hour (MW)",
    "forecast_spread_mw": "Std-dev of all forecasts for the hour published in the 24 h before the cutoff (MW)",
    "forecast_ramp_mw": "Forecast change from the previous hour in the same run (MW)",
    "forecast_day_mean_mw": "Mean forecast over the delivery day (MW)",
    "latest_actual_mw": "Most recent actual generation already published at the cutoff (MW)",
    "recent_error_mw": "Mean day-ahead forecast error over the last 24 h known at the cutoff (MW)",
}
FEATURE_COLUMNS = list(FEATURE_DESCRIPTIONS)
ID_COLUMNS = ["delivery_date", "startTime", "cutoff", "forecast_published"]
LABEL_COLUMNS = ["actual_mw", "error_mw", "shortfall_1gw"]


def build_feature_table(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    first_day: date,
    last_day: date,
    gate_closure: time = time(9, 50),
) -> pd.DataFrame:
    """
    Build the feature table for UK delivery days ``first_day..last_day``.

    ``forecasts`` / ``actuals`` are the cleaned frames from data_processor
    (actuals must include publishTime). Data from before ``first_day`` is
    used for history features when it exists.
    """
    policy = SelectionPolicy("day_ahead", gate_closure=gate_closure)
    frame = _decision_frame(forecasts, actuals, first_day, last_day, policy)

    frame["hour_uk"] = uk_local_hour(frame["startTime"])
    frame["forecast_revision_mw"] = frame["forecast_mw"] - _previous_run_forecast(forecasts, frame)
    frame["forecast_spread_mw"] = _recent_run_spread(forecasts, frame)
    frame["forecast_ramp_mw"] = frame["forecast_mw"] - _same_run_previous_hour(forecasts, frame)
    frame["forecast_day_mean_mw"] = frame.groupby("delivery_date")["forecast_mw"].transform("mean")

    # History features need the day-ahead errors of the days *before* each delivery
    # day, so evaluate two extra days back (the 24 h window never reaches further).
    history = _decision_frame(forecasts, actuals, first_day - timedelta(days=2), last_day, policy)
    latest_actual, recent_error = _actuals_known_at_cutoff(frame["cutoff"].unique(), actuals, history)
    frame["latest_actual_mw"] = frame["cutoff"].map(latest_actual)
    frame["recent_error_mw"] = frame["cutoff"].map(recent_error)

    frame["shortfall_1gw"] = (frame["error_mw"] >= SHORTFALL_THRESHOLD_MW).astype("float64")
    frame.loc[frame["error_mw"].isna(), "shortfall_1gw"] = float("nan")
    return frame[ID_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS]


def _decision_frame(forecasts, actuals, first_day, last_day, policy) -> pd.DataFrame:
    """The API's evaluation frame, renamed to ML vocabulary."""
    frame = evaluate(forecasts, actuals, first_day, last_day, policy).rename(
        columns={
            "publishTime": "forecast_published",
            "forecast": "forecast_mw",
            "horizon_hours": "lead_hours",
            "actual": "actual_mw",
            "error": "error_mw",
        }
    )
    frame.insert(0, "delivery_date", uk_local_date(frame["startTime"]))
    return frame.drop(columns="abs_error")


def _previous_run_forecast(forecasts: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Forecast for the same hour from the run published just before the one used."""
    used = frame.dropna(subset=["forecast_published"])
    requests = pd.DataFrame(
        {"startTime": used["startTime"], "cutoff": used["forecast_published"] - pd.Timedelta(seconds=1)}
    )
    previous = latest_forecast_as_of(forecasts, requests).set_index("startTime")["generation"]
    return frame["startTime"].map(previous)


def _recent_run_spread(forecasts: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Std-dev of all runs' forecasts for each hour, published in the 24 h up to the cutoff."""
    joined = frame[["startTime", "cutoff"]].merge(forecasts[["startTime", "publishTime", "generation"]], on="startTime")
    in_window = (joined["publishTime"] <= joined["cutoff"]) & (joined["publishTime"] > joined["cutoff"] - RECENT_WINDOW)
    spread = joined.loc[in_window].groupby("startTime")["generation"].std()
    return frame["startTime"].map(spread)


def _same_run_previous_hour(forecasts: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Forecast for t - 1 h taken from the same run as the forecast for t."""
    wanted = pd.DataFrame(
        {"publishTime": frame["forecast_published"], "startTime": frame["startTime"] - pd.Timedelta(hours=1)}
    )
    found = wanted.merge(forecasts[["publishTime", "startTime", "generation"]], on=["publishTime", "startTime"], how="left")
    return pd.Series(found["generation"].to_numpy(), index=frame.index)


def _actuals_known_at_cutoff(cutoffs, actuals: pd.DataFrame, history: pd.DataFrame) -> tuple[dict, dict]:
    """
    For each cutoff: the latest actual value already published, and the mean
    day-ahead error over the previous 24 h of hours whose actual was published.

    An hour's error becomes known when its actual is published (its forecast
    was published the day before, so it is always known earlier). Actuals
    with no publishTime are treated as never published - the safe choice.
    """
    published_actuals = actuals.dropna(subset=["publishTime"])
    known_at = published_actuals.set_index("startTime")["publishTime"]
    errors = history.dropna(subset=["error_mw"])[["startTime", "error_mw"]].copy()
    errors["known_at"] = errors["startTime"].map(known_at)

    latest_actual: dict = {}
    recent_error: dict = {}
    for cutoff in cutoffs:
        visible = published_actuals.loc[published_actuals["publishTime"] <= cutoff]
        if not visible.empty:
            latest_actual[cutoff] = visible.loc[visible["startTime"].idxmax(), "generation"]
        window = errors.loc[(errors["known_at"] <= cutoff) & (errors["startTime"] >= cutoff - RECENT_WINDOW)]
        if not window.empty:
            recent_error[cutoff] = window["error_mw"].mean()
    return latest_actual, recent_error
