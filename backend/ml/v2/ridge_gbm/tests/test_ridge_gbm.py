"""Scope, timing, leakage and protocol tests for the Ridge / gradient-boosting experiment (offline)."""

import json
from datetime import timedelta

import numpy as np
import pytest

from app.services.feature_builder import FEATURE_COLUMNS, LABEL_COLUMNS
from ml.v2.common import load_protocol, sha256, verify_protocol
from ml.v2.ridge_gbm.candidates import configs, hour_circle
from ml.v2.ridge_gbm.config import (
    ALL,
    DEV_LAST_DELIVERY,
    FOLDS,
    PRODUCTION_LINEAR,
    PROTOCOL_FILE,
    PROTOCOL_LOCK,
    REPORTS_DIR,
    SELECTION_FILE,
    fold_masks,
    load_development,
)


@pytest.fixture(scope="module")
def dev():
    return load_development()


def test_development_rows_stop_before_january(dev):
    assert set(dev["split"]) == {"train", "validation"}
    assert dev["delivery_date"].max() == DEV_LAST_DELIVERY
    assert len(dev) == 2185


def test_folds_train_only_on_labels_known_before_the_fold(dev):
    for start, end in FOLDS:
        train, valid = fold_masks(dev["delivery_date"], (start, end))
        assert dev.loc[train, "delivery_date"].max() <= start - timedelta(days=2)
        assert dev.loc[valid, "delivery_date"].between(start, end).all()
        assert not (train & valid).any()
    assert FOLDS[-1][1] == DEV_LAST_DELIVERY


def test_features_are_the_production_cutoff_features():
    assert ALL == FEATURE_COLUMNS
    assert not set(ALL) & {*LABEL_COLUMNS, "shortfall"}


def test_no_candidate_reads_a_label(dev):
    """Fit one configuration of every family, then poison every label: predictions must not move."""
    sample = dev.loc[dev["delivery_date"] <= dev["delivery_date"].min() + timedelta(days=20)]
    protocol = load_protocol(PROTOCOL_FILE)
    for task in ("regression", "classification"):
        target = "error_mw" if task == "regression" else "shortfall"
        first_per_family = {c.family: c for c in reversed(configs(protocol, task))}
        for config in first_per_family.values():
            model = config.build()
            if "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)  # parallel tree averaging varies in the last bits (~1e-12 MW) between calls
            model.fit(sample[ALL], sample[target])
            poisoned = sample.copy()
            for column in [*LABEL_COLUMNS, "shortfall"]:
                poisoned[column] = 99_999
            assert np.array_equal(config.predict(model, sample), config.predict(model, poisoned)), config.name


def test_hour_is_on_a_circle():
    encoded = hour_circle([0, 23, 12])
    assert np.linalg.norm(encoded[0] - encoded[1]) < np.linalg.norm(encoded[0] - encoded[2])


def test_protocol_is_locked_and_unchanged():
    lock = verify_protocol(PROTOCOL_FILE, PROTOCOL_LOCK)
    assert lock["sha256"] == sha256(PROTOCOL_FILE)


def test_selection_was_frozen_from_development_before_january():
    selection = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    assert selection["protocol_sha256"] == sha256(PROTOCOL_FILE)
    assert selection["january_rows_read"] is False
    assert selection["development_hours"] == 2185
    baseline = selection["models"]["regression"]["linear_by_level"]
    assert baseline["sha256"] == sha256(PRODUCTION_LINEAR)  # the served production model, unchanged
    assert baseline["max_abs_difference_vs_refit_on_development"] == 0.0


def test_results_were_computed_from_the_frozen_selection():
    results_file = REPORTS_DIR / "results.json"
    if not results_file.is_file():
        pytest.skip("results.json not written yet (run python -m ml.v2.ridge_gbm.run)")
    results = json.loads(results_file.read_text(encoding="utf-8"))
    assert results["selection_sha256"] == sha256(SELECTION_FILE)
    assert results["protocol_sha256"] == sha256(PROTOCOL_FILE)
    assert results["january_hours"] == 744
