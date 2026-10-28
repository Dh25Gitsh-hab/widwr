"""
Tests for the existing-data experiment. Run with:  pytest ml/v2/existing_data/tests
(The wind-feature leakage tests live in ml/v2/tests and cover the same feature code.)

* Scope: only the Elexon files already in the project are read.
* Timing: every fold trains only on deliveries up to 2 days before it; rolling corrections
  for day d never see an outcome from d-1 or later.
* Pre-registration: the protocol is locked and the coded families are exactly the registered ones.
"""

import json
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.v2.common import sha256, verify_protocol
from ml.v2.existing_data.candidates import CONFIGS
from ml.v2.existing_data.config import (
    ACTUAL_FILES,
    FOLDS,
    FORECAST_FILES,
    PROTOCOL_FILE,
    PROTOCOL_LOCK,
    REPORTS_DIR,
    fold_masks,
)
from ml.v2.existing_data.run import predictions_for


def test_inputs_are_only_the_existing_project_files():
    for path in [*ACTUAL_FILES, *FORECAST_FILES]:
        assert path.parent.name in ("DATASETS", "DATASETS1") and path.is_file()
    report = REPORTS_DIR / "data_validation.json"
    if report.is_file():
        inputs = json.loads(report.read_text(encoding="utf-8"))["inputs"]
        assert set(inputs) == {f"{p.parent.name}/{p.name}" for p in [*FORECAST_FILES, *ACTUAL_FILES]}


def test_protocol_is_locked_and_unchanged():
    assert verify_protocol(PROTOCOL_FILE, PROTOCOL_LOCK)["sha256"] == sha256(PROTOCOL_FILE)


def test_coded_families_are_exactly_the_registered_ones():
    registered = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))["candidates"]
    assert {c.family for c in CONFIGS.values()} == {name for task in registered.values() for name in task}


def test_every_fold_trains_only_up_to_two_days_before_it():
    dates = pd.Series(pd.date_range("2025-10-02", "2025-12-31").date)
    assert (FOLDS[0][0], FOLDS[-1][1]) == (date(2025, 11, 6), date(2025, 12, 31))
    for fold in FOLDS:
        train, valid = fold_masks(dates, fold)
        assert dates[train].max() == fold[0] - timedelta(days=2)
        assert dates[valid].min() == fold[0] and not (train & valid).any()


def _history() -> pd.DataFrame:
    # Every day has its own error level, so a window that moves by even one day changes any fit (median included).
    days = pd.date_range("2025-10-01", "2025-10-31").date
    rows = [{"delivery_date": d, "forecast_mw": 10_000.0 + 100 * h, "hour_sin": np.sin(h), "hour_cos": np.cos(h),
             "error_mw": 1_000.0 + 50 * i + 10 * h, "shortfall_1gw": int((h + i) % 2)}
            for i, d in enumerate(days) for h in range(24)]  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.parametrize(("name", "probe_days"), [
    ("R30_rolling_mean_bias[W=7]", None), ("R31_linear_plus_rolling_offset[W=7]", None),
    ("R32_linear_rolling_refit[W=14]", None), ("C14_logistic_curve_rolling_refit[W=14]", None),
    # A median over an even number of days is not unique, so a leaked day could hide; an odd window cannot.
    ("R33_median_curve_rolling_refit[W=14]", 13),
])  # fmt: skip
def test_rolling_corrections_never_see_outcomes_from_d_minus_1_onwards(name, probe_days):
    config = CONFIGS[name] if probe_days is None else replace(CONFIGS[name], rolling_days=probe_days)
    history, day = _history(), date(2025, 10, 20)
    target = history[history.delivery_date == day]
    train = history.delivery_date <= day - timedelta(days=2)
    baseline = predictions_for(config, history, target, train)

    tampered = history.copy()
    late = tampered.delivery_date >= day - timedelta(days=1)  # outcomes not yet published at d's cutoff
    tampered.loc[late, "error_mw"] = -1e6  # d-1 had the largest errors; making it the smallest moves any median too
    tampered.loc[late, "shortfall_1gw"] = 1 - tampered.loc[late, "shortfall_1gw"]
    after = predictions_for(config, tampered, tampered[tampered.delivery_date == day], train)
    np.testing.assert_allclose(baseline, after)
