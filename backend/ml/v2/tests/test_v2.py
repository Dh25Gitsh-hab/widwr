"""
Tests for the offline v2 experiments. Run with:  pytest ml/v2/tests
(They live outside tests/ so the production suite is unchanged.)

* Leakage: changing any forecast, actual or demand forecast published after the cutoff
  leaves every v2 feature unchanged - in GMT and on the BST clock-change day.
* Known values on the hand-made scenario of tests/conftest.py.
* Protocol lock, fold timing and data separation.
"""

import json
import shutil
from datetime import date, timedelta

import pandas as pd
import pytest

import ml.v2.common as common
from app.services.data_processor import clean_actuals, clean_forecasts
from ml.v2.candidates import CANDIDATES
from ml.v2.experiment import FOLD_MONTHS, fold_split, rolling_window
from ml.v2.features import ALL_FEATURES, build_v2_table, clean_ndfd
from tests.conftest import make_raw_actuals, make_raw_forecasts, make_run
from tests.test_feature_builder import BST_CUTOFF, CLOCK_CHANGE_DAY, bst_raw_actuals, bst_raw_forecasts

GMT_DAY, GMT_CUTOFF = date(2026, 1, 4), pd.Timestamp("2026-01-03T09:50Z")


def make_raw_ndfd(run_dates: list[str], publish_time: str) -> pd.DataFrame:
    """One NDFD run per day at `publish_time` (UTC), each forecasting the next 14 days; later runs differ."""
    rows = []
    for i, run_date in enumerate(run_dates):
        for ahead in range(1, 15):
            forecast_date = (pd.Timestamp(run_date) + pd.Timedelta(days=ahead)).date()
            rows.append({"publishTime": f"{run_date}T{publish_time}Z", "forecastDate": str(forecast_date), "demand": 30_000 + 100 * i})
    return pd.DataFrame(rows)


def v2_table(raw_actuals, raw_forecasts, raw_ndfd, day) -> pd.DataFrame:
    actuals, _ = clean_actuals(raw_actuals)
    forecasts, _ = clean_forecasts(raw_forecasts)
    return build_v2_table(forecasts, actuals, clean_ndfd(raw_ndfd), day, day).set_index("startTime")


def tamper(frame: pd.DataFrame, cutoff: pd.Timestamp, column: str, value: float) -> pd.DataFrame:
    late = pd.to_datetime(frame["publishTime"], utc=True) > cutoff
    assert late.any(), "the test needs some data published after the cutoff"
    changed = frame.copy()
    changed.loc[late, column] = value
    return changed


# ---- Leakage -------------------------------------------------------------------------------


def gmt_inputs():
    late_run = pd.DataFrame(make_run("2026-01-03T11:30:00Z", 14_000.0))  # published after the 4 Jan cutoff
    forecasts = pd.concat([make_raw_forecasts(), late_run], ignore_index=True)
    ndfd = make_raw_ndfd(["2025-12-31", "2026-01-01", "2026-01-02", "2026-01-03"], "14:45:00")
    return make_raw_actuals(), forecasts, ndfd


def test_post_cutoff_data_does_not_change_v2_features_in_gmt():
    actuals, forecasts, ndfd = gmt_inputs()
    baseline = v2_table(actuals, forecasts, ndfd, GMT_DAY)
    after = v2_table(tamper(actuals, GMT_CUTOFF, "generation", 1.0), tamper(forecasts, GMT_CUTOFF, "generation", 99_999.0),
                     tamper(ndfd, GMT_CUTOFF, "demand", 1.0), GMT_DAY)  # fmt: skip
    pd.testing.assert_frame_equal(baseline[ALL_FEATURES], after[ALL_FEATURES])


def test_post_cutoff_data_does_not_change_v2_features_on_the_bst_clock_change_day():
    # NDFD runs at 14:45 BST = 13:45 UTC; the 25 Oct run comes after the 08:50 UTC cutoff.
    actuals, forecasts = bst_raw_actuals(), bst_raw_forecasts()
    ndfd = make_raw_ndfd(["2025-10-22", "2025-10-23", "2025-10-24", "2025-10-25"], "13:45:00")
    baseline = v2_table(actuals, forecasts, ndfd, CLOCK_CHANGE_DAY)
    after = v2_table(tamper(actuals, BST_CUTOFF, "generation", 1.0), tamper(forecasts, BST_CUTOFF, "generation", 99_999.0),
                     tamper(ndfd, BST_CUTOFF, "demand", 1.0), CLOCK_CHANGE_DAY)  # fmt: skip
    pd.testing.assert_frame_equal(baseline[ALL_FEATURES], after[ALL_FEATURES])
    assert len(baseline) == 25  # the clocks go back
    assert (baseline["ndfd_demand_mw"] == 30_200).all()  # the 24 Oct run, not the later 25 Oct run


# ---- Known values ---------------------------------------------------------------------------


def test_v2_feature_values_on_the_scenario():
    hour = v2_table(*gmt_inputs(), GMT_DAY).loc[pd.Timestamp("2026-01-04T05:00Z")]
    # By the 3 Jan 09:50 cutoff, 3 Jan hours up to 09:00 are known, each with error 12,000 - 10,000.
    assert hour["err_last3h_mw"] == 2_000.0
    assert hour["err_7d_mw"] == 2_000.0
    assert hour["err_band_14d_mw"] == 2_000.0  # same 10-15 GW band as the 13,000 MW forecast
    assert pd.isna(hour["err_prev_day_mw"])  # D-2 = 2 Jan has no day-ahead forecast in the scenario
    assert hour["forecast_revision_24h_mw"] == 1_000.0  # 13,000 now vs the 12,000 run known by 2 Jan 09:50
    assert (hour["forecast_day_max_mw"], hour["forecast_day_min_mw"], hour["forecast_day_std_mw"]) == (13_000.0, 13_000.0, 0.0)
    assert (hour["day_of_week"], hour["bank_holiday"]) == (6, 0)  # 4 Jan 2026 is a Sunday
    assert hour["ndfd_demand_mw"] == 30_200  # the 2 Jan 14:45 run (the 3 Jan run comes after the cutoff)
    assert hour["wind_share_ndfd"] == pytest.approx(13_000 / 30_200)
    assert hour["hour_sin"] == pytest.approx(0.9659, abs=1e-4)  # 05:00


# ---- Protocol and data separation -----------------------------------------------------------


def test_protocol_is_locked_and_unchanged():
    lock = common.verify_protocol()
    assert lock["sha256"] == common.sha256(common.PROTOCOL_FILE)


def test_a_changed_protocol_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "protocol.json"
    shutil.copy(common.PROTOCOL_FILE, copy)
    copy.write_text(copy.read_text(encoding="utf-8").replace("lowest mean fold MAE", "lowest mean fold RMSE"), encoding="utf-8")
    monkeypatch.setattr(common, "PROTOCOL_FILE", copy)
    with pytest.raises(SystemExit, match="changed after it was locked"):
        common.verify_protocol()


def test_candidates_are_exactly_the_preregistered_ones():
    registered = common.load_protocol()["candidates"]
    keys = [key.split(" /")[0].removesuffix("_w30") for task in registered.values() for key in task]
    coded = {name.removesuffix("_w30").removesuffix("_w60").removesuffix("_w90") for name in CANDIDATES}
    assert coded == set(keys)


def test_every_fold_trains_only_on_labels_known_before_its_first_cutoff():
    dev = pd.DataFrame({"delivery_date": pd.date_range("2024-10-02", "2025-12-31").date})
    assert FOLD_MONTHS == [f"2025-{m:02d}" for m in range(4, 13)]
    for month in FOLD_MONTHS:
        train, valid = fold_split(dev, month)
        first_valid = dev.loc[valid, "delivery_date"].min()
        assert dev.loc[train, "delivery_date"].max() == first_valid - timedelta(days=2)
        assert not (train & valid).any()


def test_rolling_window_ends_two_days_before_the_delivery_day():
    dates = pd.Series(pd.date_range("2025-01-01", "2025-06-30").date)
    window = dates[rolling_window(dates, date(2025, 5, 10), 30)]
    assert (window.min(), window.max(), len(window)) == (date(2025, 4, 9), date(2025, 5, 8), 30)


@pytest.mark.skipif(not (common.PROCESSED_DIR / "dev_dataset.csv").is_file(), reason="v2 datasets not built")
def test_datasets_stay_inside_their_periods():
    dev = pd.read_csv(common.PROCESSED_DIR / "dev_dataset.csv")
    january = pd.read_csv(common.PROCESSED_DIR / "january_dataset.csv")
    dev_decisions = pd.to_datetime(dev["delivery_date"]) - pd.Timedelta(days=1)
    jan_decisions = pd.to_datetime(january["delivery_date"]) - pd.Timedelta(days=1)
    assert dev_decisions.max() <= pd.Timestamp(common.DEV_LAST_DECISION)
    assert jan_decisions.between(pd.Timestamp(common.JAN_FIRST_DECISION), pd.Timestamp(common.JAN_LAST_DECISION)).all()
    report = json.loads((common.REPORTS_DIR / "data_validation.json").read_text(encoding="utf-8"))
    assert not any(name.startswith(("Feb", "Mar", "Apr", "May")) for name in report["inputs"])  # no Feb+ files read
