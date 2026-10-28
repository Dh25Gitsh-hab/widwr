"""
Tests for the February holdout: the statistics, the seal, and (when present)
the real sealed outputs.
"""

import json

import numpy as np
import pandas as pd
import pytest

import ml.holdout.evaluate as evaluate_module
from ml.holdout.common import MANIFEST_FILE, PREDICTIONS_FILE, REPORTS_DIR, sha256
from ml.holdout.stats import block_bootstrap, diebold_mariano, interval


def test_block_bootstrap_of_a_constant_difference_is_that_constant():
    days = pd.Series(np.repeat(np.arange(10), 24))
    values = np.full(240, 150.0)
    samples = block_bootstrap(days, lambda rows: values[rows].mean(), block_days=3, n_resamples=200)
    assert interval(samples) == (150.0, 150.0)


def test_block_bootstrap_resamples_whole_consecutive_days():
    days = pd.Series(np.repeat(np.arange(9), 2))
    day_values = days.to_numpy()
    samples = block_bootstrap(days, lambda rows: day_values[rows], block_days=3, n_resamples=50)
    for picked in samples:
        assert len(picked) == 18  # 9 days x 2 hours, whole days only
        firsts = picked[::6]  # each block of 3 days starts every 6 rows
        assert all((picked[i * 6 : i * 6 + 6] == np.repeat((first + np.arange(3)) % 9, 2)).all() for i, first in enumerate(firsts))


def test_diebold_mariano_detects_a_clear_difference_and_not_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(1_000, 200, 600)
    clear = diebold_mariano(base + 300, base)
    noise = diebold_mariano(base + rng.normal(0, 50, 600), base)
    assert clear["mean_loss_difference"] == pytest.approx(300)
    assert noise["p_value"] > 0.05


def test_evaluation_refuses_a_broken_seal(tmp_path, monkeypatch):
    sealed = tmp_path / "predictions.csv"
    sealed.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "predictions_file": "predictions.csv",
        "predictions_sha256": sha256(sealed),
        "frozen_existing_results_sha256": {},
        "models": {},
    }), encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(evaluate_module, "MANIFEST_FILE", manifest)
    monkeypatch.setattr(evaluate_module, "PROJECT_ROOT", tmp_path)

    evaluate_module.verify_seal()  # intact -> passes
    sealed.write_text("a,b\n1,3\n", encoding="utf-8")  # tamper after sealing
    with pytest.raises(SystemExit, match="Seal broken"):
        evaluate_module.verify_seal()


requires_seal = pytest.mark.skipif(not MANIFEST_FILE.is_file(), reason="February holdout not sealed yet")


@requires_seal
def test_sealed_predictions_contain_no_labels_and_respect_the_cutoff():
    sealed = pd.read_csv(PREDICTIONS_FILE)
    assert not any(column.startswith(("actual", "error", "shortfall")) for column in sealed.columns)
    assert (pd.to_datetime(sealed["forecast_published"]) <= pd.to_datetime(sealed["cutoff"])).all()
    assert pd.to_datetime(sealed["delivery_date"]).min() == pd.Timestamp("2026-02-02")  # 1 Feb belongs to January's test


@requires_seal
def test_seal_and_cutoff_audit_are_intact():
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    assert sha256(PREDICTIONS_FILE) == manifest["predictions_sha256"]
    assert all(check["reproduces_saved_january_predictions"] for check in manifest["models"].values())
    audit = json.loads((REPORTS_DIR / "cutoff_audit.json").read_text(encoding="utf-8"))
    assert audit["all_days_pass"] and audit["forecasts_published_after_cutoff_used"] == 0
