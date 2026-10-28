"""
v2 features (offline).

The 9 production features come from app.services.feature_builder, imported unchanged.
The new features below are added on top. Every value uses only rows published at or
before the cutoff c = 09:50 UK time on D-1 (the same rule as production), so the
leakage tests in ml/v2/tests can change anything published later and see no feature move.

NEW FEATURES
    hour_sin, hour_cos         UK clock hour on a circle (23:00 is next to 00:00)
    err_last3h_mw              mean day-ahead error of the 3 latest hours whose actual was published by c
    err_prev_day_mw            mean day-ahead error over UK day D-2 (needs >= 20 known hours)
    err_7d_mw                  mean day-ahead error of hours in the 7 days before c, known by c
    err_band_14d_mw            same over 14 days, only hours in the same fixed forecast-size band
    forecast_revision_24h_mw   forecast_mw minus the latest forecast for the hour published by c - 24 h
    forecast_day_max/min/std   shape of the decision-time forecast over delivery day D
    day_of_week, bank_holiday  calendar of D
    ndfd_demand_mw             NDFD national demand forecast for D, latest run published by c
    wind_share_ndfd            forecast_day_mean_mw / ndfd_demand_mw
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.services.feature_builder import FEATURE_COLUMNS, build_feature_table
from app.services.forecast_evaluator import SelectionPolicy, evaluate, latest_forecast_as_of
from app.utils.timestamps import to_utc, uk_local_date
from ml.v2.common import BANK_HOLIDAYS, GATE_CLOSURE, LEVEL_BANDS_MW

NEW_FEATURES = [
    "hour_sin", "hour_cos", "err_last3h_mw", "err_prev_day_mw", "err_7d_mw", "err_band_14d_mw",
    "forecast_revision_24h_mw", "forecast_day_max_mw", "forecast_day_min_mw", "forecast_day_std_mw",
    "day_of_week", "bank_holiday", "ndfd_demand_mw", "wind_share_ndfd",
]  # fmt: skip
ALL_FEATURES = [f for f in FEATURE_COLUMNS if f != "lead_hours"] + NEW_FEATURES
NDFD_FEATURES = ["ndfd_demand_mw", "wind_share_ndfd"]
WIND_FEATURES = [f for f in ALL_FEATURES if f not in NDFD_FEATURES]  # existing Elexon wind data only
HISTORY_DAYS = 15  # the longest look-back (14 days) plus one day of margin


def clean_ndfd(raw: pd.DataFrame) -> pd.DataFrame:
    """NDFD rows -> publishTime (UTC), forecastDate (date), demand (MW); latest duplicate wins."""
    df = raw[["publishTime", "forecastDate", "demand"]].copy()
    df["publishTime"] = to_utc(df["publishTime"])
    df["forecastDate"] = pd.to_datetime(df["forecastDate"], errors="coerce").dt.date
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna().drop_duplicates(["publishTime", "forecastDate"], keep="last")
    return df.sort_values("publishTime").reset_index(drop=True)


def build_v2_table(forecasts: pd.DataFrame, actuals: pd.DataFrame, ndfd: pd.DataFrame | None, first_day: date, last_day: date) -> pd.DataFrame:
    """One row per delivery hour of first_day..last_day: production features + new features + labels.

    ndfd=None builds the wind-only features (WIND_FEATURES) from WINDFOR and FUELHH alone.
    """
    table = build_feature_table(forecasts, actuals, first_day, last_day, GATE_CLOSURE)
    table["hour_sin"] = np.sin(2 * np.pi * table["hour_uk"] / 24)
    table["hour_cos"] = np.cos(2 * np.pi * table["hour_uk"] / 24)
    table = table.join(_error_history_features(forecasts, actuals, table, first_day, last_day))
    table["forecast_revision_24h_mw"] = table["forecast_mw"] - _forecast_known_24h_earlier(forecasts, table)

    by_day = table.groupby("delivery_date")["forecast_mw"]
    table["forecast_day_max_mw"] = by_day.transform("max")
    table["forecast_day_min_mw"] = by_day.transform("min")
    table["forecast_day_std_mw"] = by_day.transform("std")
    table["day_of_week"] = pd.to_datetime(table["delivery_date"]).dt.day_of_week
    table["bank_holiday"] = table["delivery_date"].isin(sorted(BANK_HOLIDAYS)).astype(int)

    if ndfd is not None:
        table["ndfd_demand_mw"] = _ndfd_known_at_cutoff(ndfd, table)
        table["wind_share_ndfd"] = table["forecast_day_mean_mw"] / table["ndfd_demand_mw"]
    return table


def _error_history_features(forecasts, actuals, table, first_day, last_day) -> pd.DataFrame:
    """Day-ahead errors of earlier hours, each counted only once its actual was published (<= c)."""
    policy = SelectionPolicy("day_ahead", gate_closure=GATE_CLOSURE)
    history = evaluate(forecasts, actuals, first_day - timedelta(days=HISTORY_DAYS), last_day, policy)
    history = history.dropna(subset=["error"])[["startTime", "forecast", "error"]].copy()
    known_at = actuals.dropna(subset=["publishTime"]).set_index("startTime")["publishTime"]
    history["known_at"] = history["startTime"].map(known_at)  # an actual with no publishTime never counts
    history = history.dropna(subset=["known_at"])
    history["uk_date"] = uk_local_date(history["startTime"])
    history["band"] = pd.cut(history["forecast"], LEVEL_BANDS_MW, right=False, labels=False)
    row_band = pd.cut(table["forecast_mw"], LEVEL_BANDS_MW, right=False, labels=False)

    out = pd.DataFrame(index=table.index, columns=["err_last3h_mw", "err_prev_day_mw", "err_7d_mw", "err_band_14d_mw"], dtype=float)
    for (cutoff, delivery_day), rows in table.groupby(["cutoff", "delivery_date"]):
        known = history.loc[history["known_at"] <= cutoff]
        if known.empty:
            continue
        latest3 = known.nlargest(3, "startTime")["error"]
        out.loc[rows.index, "err_last3h_mw"] = latest3.mean() if len(latest3) == 3 else np.nan
        prev_day = known.loc[known["uk_date"] == delivery_day - timedelta(days=2), "error"]
        out.loc[rows.index, "err_prev_day_mw"] = prev_day.mean() if len(prev_day) >= 20 else np.nan
        week = known.loc[known["startTime"] >= cutoff - pd.Timedelta(days=7), "error"]
        out.loc[rows.index, "err_7d_mw"] = week.mean() if len(week) else np.nan
        fortnight = known.loc[known["startTime"] >= cutoff - pd.Timedelta(days=14)]
        out.loc[rows.index, "err_band_14d_mw"] = row_band[rows.index].map(fortnight.groupby("band")["error"].mean())
    return out


def _forecast_known_24h_earlier(forecasts: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """The latest forecast for each hour that was published at least 24 h before the cutoff."""
    requests = pd.DataFrame({"startTime": table["startTime"], "cutoff": table["cutoff"] - pd.Timedelta(hours=24)})
    earlier = latest_forecast_as_of(forecasts, requests).set_index("startTime")["generation"]
    return table["startTime"].map(earlier)


def _ndfd_known_at_cutoff(ndfd: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """Demand forecast for delivery day D from the latest NDFD run published by D's cutoff."""
    requests = table[["delivery_date", "cutoff"]].drop_duplicates().sort_values("cutoff")
    picked = pd.merge_asof(requests, ndfd, left_on="cutoff", right_on="publishTime", left_by="delivery_date",
                           right_by="forecastDate", direction="backward", allow_exact_matches=True)  # fmt: skip
    demand = picked.set_index(["delivery_date", "cutoff"])["demand"]
    return pd.Series(demand.reindex(pd.MultiIndex.from_frame(table[["delivery_date", "cutoff"]])).to_numpy(), index=table.index)
