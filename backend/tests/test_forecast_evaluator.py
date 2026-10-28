"""
Tests for the core business rule: only forecasts published at or before the
decision cutoff may be used, and the latest of those is chosen.
The synthetic scenario is described in conftest.py.
"""

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from app.services.forecast_evaluator import (
    SelectionPolicy,
    breakdown,
    compute_metrics,
    day_ahead_view,
    evaluate,
    horizon_profile,
)

DAY = date(2026, 1, 3)


def forecast_values(frame: pd.DataFrame) -> set:
    return set(frame["forecast"].dropna())


def test_day_ahead_uses_latest_run_before_gate_closure(forecasts, actuals):
    frame = evaluate(forecasts, actuals, DAY, DAY, SelectionPolicy("day_ahead", gate_closure=time(9, 50)))
    assert forecast_values(frame) == {12_000.0}  # the 09:30 run, never the 11:30 one
    assert set(frame["publishTime"]) == {pd.Timestamp("2026-01-02T09:30Z")}
    assert (frame["publishTime"] <= frame["cutoff"]).all()


def test_earlier_gate_closure_falls_back_to_earlier_run(forecasts, actuals):
    frame = evaluate(forecasts, actuals, DAY, DAY, SelectionPolicy("day_ahead", gate_closure=time(9, 20)))
    assert forecast_values(frame) == {11_000.0}


def test_run_published_exactly_at_cutoff_is_eligible(forecasts, actuals):
    frame = evaluate(forecasts, actuals, DAY, DAY, SelectionPolicy("day_ahead", gate_closure=time(9, 30)))
    assert forecast_values(frame) == {12_000.0}


def test_no_eligible_run_stays_missing_not_zero(forecasts, actuals):
    frame = evaluate(forecasts, actuals, DAY, DAY, SelectionPolicy("day_ahead", gate_closure=time(7, 0)))
    assert frame["forecast"].isna().all()
    metrics = compute_metrics(frame)
    assert metrics["missing_forecast"] == 24
    assert metrics["intervals_scored"] == 0
    assert metrics["mae"] is None


def test_fixed_horizon_policy(forecasts, actuals):
    # Target 3 Jan 05:00 with h=20 -> cutoff 2 Jan 09:00 -> only the 07:30 run qualifies.
    frame = evaluate(forecasts, actuals, DAY, DAY, SelectionPolicy("fixed_horizon", horizon_hours=20))
    row = frame.set_index("startTime").loc[pd.Timestamp("2026-01-03T05:00Z")]
    assert row["forecast"] == 11_000.0
    assert row["horizon_hours"] == pytest.approx(21.5)


def test_metrics_on_whole_window(forecasts, actuals):
    frame = evaluate(forecasts, actuals, date(2026, 1, 3), date(2026, 1, 4), SelectionPolicy())
    m = compute_metrics(frame)
    # 3 Jan: 24 h at +2,000. 4 Jan: 22 forecast hours at +3,000, minus the hour with no actual.
    assert m["intervals_total"] == 48
    assert m["missing_forecast"] == 2  # 4 Jan 22:00 and 23:00
    assert m["missing_actual"] == 1  # 4 Jan 12:00
    assert m["intervals_scored"] == 45
    errors = np.array([2_000.0] * 24 + [3_000.0] * 21)
    assert m["mae"] == pytest.approx(errors.mean(), abs=0.1)
    assert m["rmse"] == pytest.approx(np.sqrt((errors**2).mean()), abs=0.1)
    assert m["bias"] == m["mae"]  # every error is positive
    assert m["over_forecast_pct"] == 100.0


def test_metrics_formulas_on_hand_made_errors():
    frame = pd.DataFrame(
        {
            "startTime": pd.date_range("2026-01-10", periods=3, freq="1h", tz="UTC"),
            "forecast": [1_100.0, 900.0, 1_300.0],
            "actual": [1_000.0, 1_000.0, 1_000.0],
        }
    )
    frame["error"] = frame["forecast"] - frame["actual"]  # +100, -100, +300
    frame["abs_error"] = frame["error"].abs()
    m = compute_metrics(frame)
    assert m["mae"] == pytest.approx(166.7)
    assert m["rmse"] == pytest.approx(191.5)
    assert m["bias"] == pytest.approx(100.0)
    assert m["max_abs_error"] == 300.0
    assert m["max_error_time"] == pd.Timestamp("2026-01-10T02:00Z")
    assert m["nmae_pct"] == pytest.approx(16.7)


def test_breakdown_by_forecast_level(forecasts, actuals):
    frame = evaluate(forecasts, actuals, date(2026, 1, 3), date(2026, 1, 4), SelectionPolicy())
    rows = breakdown(frame, "forecast_level")
    assert [(r["group"], r["n"]) for r in rows] == [("10-15 GW", 45)]


def test_breakdown_by_delivery_date(forecasts, actuals):
    frame = evaluate(forecasts, actuals, date(2026, 1, 3), date(2026, 1, 4), SelectionPolicy())
    rows = {r["group"]: r for r in breakdown(frame, "delivery_date")}
    assert rows["2026-01-03"]["bias"] == 2_000.0
    assert rows["2026-01-04"]["bias"] == 3_000.0


def test_horizon_profile_rows(forecasts, actuals):
    rows = horizon_profile(forecasts, actuals, DAY, DAY, range(0, 25, 12))
    assert [r["horizon_hours"] for r in rows] == [0, 12, 24]
    assert all(r["mean_lead_hours"] >= r["horizon_hours"] for r in rows)


def test_day_ahead_view_flags_runs(forecasts, actuals):
    view = day_ahead_view(forecasts, actuals, DAY, time(9, 50))
    assert view["cutoff"] == pd.Timestamp("2026-01-02T09:50Z")
    assert view["forecast_run_used"] == pd.Timestamp("2026-01-02T09:30Z")
    flags = {run["publish_time"].strftime("%H:%M"): (run["available_at_decision"], run["used_for_decision"]) for run in view["runs"]}
    assert flags == {"07:30": (True, False), "09:30": (True, True), "11:30": (False, False)}
    assert view["expected"]["mean_mw"] == 12_000.0
    assert view["hindsight"]["bias"] == 2_000.0
    assert len(view["intervals"]) == 48  # half-hourly
