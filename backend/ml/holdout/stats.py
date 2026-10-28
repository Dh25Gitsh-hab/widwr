"""
Statistics for comparing forecasts on a short holdout with time-dependent errors.

WHY special care
    Hourly errors are not independent: the hours of one day share the same
    forecast run and weather, and windy spells last several days. Treating
    648 hours as 648 independent observations would make every difference
    look "significant". Two complementary tools are used:

    block_bootstrap   resamples blocks of consecutive DAYS (circular moving
                      blocks of 1 and 3 days) -> 95% interval for any statistic
                      (MAE difference, RMSE difference, ROC-AUC, Brier ...)
    diebold_mariano   the standard test for "are two forecasts equally
                      accurate?" on the hourly loss differential, with a
                      Newey-West (HAC) variance that allows autocorrelation
                      up to `max_lag` hours
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

SEED = 42


def block_bootstrap(
    days: pd.Series,
    statistic: Callable[[np.ndarray], float | np.ndarray],
    block_days: int,
    n_resamples: int = 5000,
) -> np.ndarray:
    """
    Distribution of ``statistic(row_indices)`` over circular moving-block
    resamples of whole days. Returns an array of shape (n_resamples, ...).
    """
    day_values = days.to_numpy()
    unique_days = np.sort(pd.unique(day_values))
    rows_by_day = [np.flatnonzero(day_values == day) for day in unique_days]
    n_days = len(unique_days)
    n_blocks = math.ceil(n_days / block_days)

    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks))
    day_picks = ((starts[:, :, None] + np.arange(block_days)) % n_days).reshape(n_resamples, -1)[:, :n_days]
    return np.array([statistic(np.concatenate([rows_by_day[d] for d in picks])) for picks in day_picks])


def interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 48) -> dict:
    """
    H0: forecasts A and B are equally accurate. d_t = loss_a - loss_b in time order.
    A positive mean means B had the smaller loss. Two-sided normal p-value.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    centred = d - d.mean()
    variance = centred @ centred / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        variance += 2 * weight * (centred[lag:] @ centred[:-lag]) / n
    statistic = d.mean() / math.sqrt(variance / n) if variance > 0 else float("nan")
    return {
        "mean_loss_difference": round(float(d.mean()), 4),
        "dm_statistic": round(float(statistic), 3),
        "p_value": round(float(math.erfc(abs(statistic) / math.sqrt(2))), 4),
        "hac_max_lag_hours": max_lag,
    }
