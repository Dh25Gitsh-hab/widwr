"""
Leakage tests for the ML features.

The strongest test of "no look-ahead" is a perturbation test: change every
value that was published AFTER a delivery day's cutoff and check that none of
that day's features move. Only the labels (actuals) may change.
Synthetic scenario: see conftest.py (cutoff for 3 Jan = 2 Jan 09:50 UTC).
The same tests are repeated in UK summer time, on the day the clocks change.
"""

from datetime import date, time

import pandas as pd
import pytest

from app.config import Settings
from app.services.data_processor import clean_actuals, clean_forecasts
from app.services.feature_builder import FEATURE_COLUMNS, build_feature_table
from app.utils.timestamps import gate_closure_utc
from tests.conftest import make_raw_actuals, make_raw_forecasts, make_run

DAY = date(2026, 1, 3)
CUTOFF = pd.Timestamp("2026-01-02T09:50Z")

# Delivery 26 Oct 2025 is the 25-hour day when the clocks go back. It is decided on
# 25 Oct, still in BST, so the cutoff is 09:50 BST = 08:50 UTC. The run published at
# 09:30 UTC (10:30 BST) came AFTER the cutoff, although 09:30 < 09:50: a rule that
# ignored UK summer time would wrongly use it.
CLOCK_CHANGE_DAY = date(2025, 10, 26)
BST_CUTOFF = pd.Timestamp("2025-10-25T08:50Z")
BST_RUNS = {"2025-10-25T07:30:00Z": 11_000.0, "2025-10-25T08:30:00Z": 12_000.0, "2025-10-25T09:30:00Z": 13_000.0}
ML_DATASET = Settings().ml_dir / "processed" / "ml_dataset.csv"


def features_for_day(raw_actuals: pd.DataFrame, raw_forecasts: pd.DataFrame, day: date = DAY) -> pd.DataFrame:
    actuals, _ = clean_actuals(raw_actuals)
    forecasts, _ = clean_forecasts(raw_forecasts)
    table = build_feature_table(forecasts, actuals, day, day)
    return table.set_index("startTime")


def bst_raw_forecasts() -> pd.DataFrame:
    window = {"first": "2025-10-24T21:00:00Z", "last": "2025-10-27T21:00:00Z"}
    return pd.DataFrame([row for publish, value in BST_RUNS.items() for row in make_run(publish, value, **window)])


def bst_raw_actuals() -> pd.DataFrame:
    return make_raw_actuals(first="2025-10-23", end="2025-10-28")


def test_forecasts_published_after_cutoff_do_not_change_features():
    raw_forecasts = make_raw_forecasts()
    baseline = features_for_day(make_raw_actuals(), raw_forecasts)

    late = pd.to_datetime(raw_forecasts["publishTime"], utc=True) > CUTOFF
    assert late.any()
    tampered = raw_forecasts.copy()
    tampered.loc[late, "generation"] = 99_999.0
    after = features_for_day(make_raw_actuals(), tampered)

    pd.testing.assert_frame_equal(baseline[FEATURE_COLUMNS], after[FEATURE_COLUMNS])


def test_actuals_published_after_cutoff_do_not_change_features():
    raw_actuals = make_raw_actuals()
    baseline = features_for_day(raw_actuals, make_raw_forecasts())

    late = pd.to_datetime(raw_actuals["publishTime"], utc=True) > CUTOFF
    tampered = raw_actuals.copy()
    tampered.loc[late, "generation"] = 1.0
    after = features_for_day(tampered, make_raw_forecasts())

    pd.testing.assert_frame_equal(baseline[FEATURE_COLUMNS], after[FEATURE_COLUMNS])
    assert not baseline["actual_mw"].equals(after["actual_mw"])  # labels DO change - as they should


def test_feature_values_on_known_scenario(forecasts, actuals):
    table = build_feature_table(forecasts, actuals, DAY, date(2026, 1, 4)).set_index("startTime")
    hour = table.loc[pd.Timestamp("2026-01-03T05:00Z")]
    assert hour["forecast_mw"] == 12_000.0  # the 09:30 run
    assert hour["forecast_revision_mw"] == 1_000.0  # 12,000 - 11,000 (07:30 run)
    assert hour["forecast_spread_mw"] == pytest.approx(707.1, abs=0.1)  # std of 11,000 and 12,000
    assert hour["forecast_ramp_mw"] == 0.0  # flat run
    assert hour["latest_actual_mw"] == 10_000.0  # 2 Jan 09:00, published 09:30 <= cutoff
    assert pd.isna(hour["recent_error_mw"])  # no earlier decision exists in the scenario
    assert hour["error_mw"] == 2_000.0 and hour["shortfall_1gw"] == 1.0

    # 4 Jan is decided on 3 Jan 09:50: by then the 3 Jan errors (+2,000) are known.
    next_day = table.loc[pd.Timestamp("2026-01-04T05:00Z")]
    assert next_day["recent_error_mw"] == 2_000.0
    assert next_day["forecast_mw"] == 13_000.0


def test_missing_values_stay_missing(forecasts, actuals):
    table = build_feature_table(forecasts, actuals, date(2026, 1, 4), date(2026, 1, 4)).set_index("startTime")
    assert pd.isna(table.loc[pd.Timestamp("2026-01-04T23:00Z"), "forecast_mw"])  # no run covers it
    assert pd.isna(table.loc[pd.Timestamp("2026-01-04T12:00Z"), "shortfall_1gw"])  # no actual


# ---- UK summer time and the clock change ------------------------------------------------


def test_bst_cutoff_uses_the_run_published_by_0950_uk_time():
    table = features_for_day(bst_raw_actuals(), bst_raw_forecasts(), CLOCK_CHANGE_DAY)
    assert len(table) == 25  # the clocks go back: a 25-hour delivery day
    assert (table["cutoff"] == BST_CUTOFF).all()
    assert (table["forecast_published"] == pd.Timestamp("2025-10-25T08:30Z")).all()  # 09:30 BST, not 09:30 UTC
    assert (table["forecast_mw"] == 12_000.0).all()
    assert (table["latest_actual_mw"] == 10_000.0).all()  # 08:00 UTC half-hour, published 08:30 UTC


def test_bst_data_published_after_the_cutoff_does_not_change_features():
    raw_actuals, raw_forecasts = bst_raw_actuals(), bst_raw_forecasts()
    baseline = features_for_day(raw_actuals, raw_forecasts, CLOCK_CHANGE_DAY)

    late_forecasts = pd.to_datetime(raw_forecasts["publishTime"], utc=True) > BST_CUTOFF
    late_actuals = pd.to_datetime(raw_actuals["publishTime"], utc=True) > BST_CUTOFF
    assert late_forecasts.any() and late_actuals.any()
    tampered_forecasts, tampered_actuals = raw_forecasts.copy(), raw_actuals.copy()
    tampered_forecasts.loc[late_forecasts, "generation"] = 99_999.0
    tampered_actuals.loc[late_actuals, "generation"] = 1.0
    after = features_for_day(tampered_actuals, tampered_forecasts, CLOCK_CHANGE_DAY)

    pd.testing.assert_frame_equal(baseline[FEATURE_COLUMNS], after[FEATURE_COLUMNS])


@pytest.mark.skipif(not ML_DATASET.is_file(), reason="ML dataset not built - run python -m ml.build_dataset")
def test_real_ml_dataset_respects_the_uk_cutoff_across_the_clock_change():
    data = pd.read_csv(ML_DATASET)
    cutoff = pd.to_datetime(data["cutoff"], utc=True)
    expected = pd.to_datetime(data["delivery_date"].map(lambda d: gate_closure_utc(date.fromisoformat(d), time(9, 50))))
    assert (cutoff == expected).all()  # every row, October 2025 (BST) to January 2026 (GMT)
    assert (pd.to_datetime(data["forecast_published"], utc=True) <= cutoff).all()
    clock_change = data["delivery_date"] == CLOCK_CHANGE_DAY.isoformat()
    assert clock_change.sum() == 25
    assert (cutoff[clock_change] == BST_CUTOFF).all()
