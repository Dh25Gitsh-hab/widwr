/**
 * Summaries computed in the browser from the API's hourly points. They are simple
 * averages and counts of returned values - no model output is created here.
 */

import { ukDateOf } from "./dates.js";
import { isNumber } from "./format.js";

export function mean(values) {
  const numbers = values.filter(isNumber);
  return numbers.length ? numbers.reduce((sum, v) => sum + v, 0) / numbers.length : null;
}

/** Headline figures for one delivery day of /api/ml/predictions points. */
export function summarizeDay(points) {
  if (!points?.length) return null;
  const probabilities = points.map((p) => p.shortfall_probability);
  let peak = 0;
  probabilities.forEach((p, i) => {
    if (p > probabilities[peak]) peak = i;
  });
  const withOutcome = points.filter((p) => p.shortfall_occurred !== null && p.shortfall_occurred !== undefined);
  return {
    hours: points.length,
    meanOfficial: mean(points.map((p) => p.forecast_mw)),
    meanCorrected: mean(points.map((p) => p.corrected_forecast_mw)),
    meanRisk: mean(probabilities),
    peakRisk: probabilities[peak],
    peakRiskTime: points[peak].start_time,
    highRiskHours: probabilities.filter((p) => p >= 0.5).length,
    hoursWithOutcome: withOutcome.length,
    observedShortfalls: withOutcome.filter((p) => p.shortfall_occurred).length,
  };
}

/** Round to 0.1 MW, the precision of the API's own MAE scores. */
function toTenth(value) {
  return isNumber(value) ? Math.round(value * 10) / 10 : value;
}

/**
 * Mean absolute error per UK delivery day (hours with an actual only), rounded to 0.1 MW like
 * the API's scores so the month view and the selected day's tile show the same whole-MW value.
 */
export function dailyErrors(points) {
  const byDay = new Map();
  for (const point of points ?? []) {
    if (!isNumber(point.actual_mw)) continue;
    const day = ukDateOf(point.start_time);
    if (!byDay.has(day)) byDay.set(day, { official: [], corrected: [] });
    byDay.get(day).official.push(Math.abs(point.error_mw));
    byDay.get(day).corrected.push(Math.abs(point.corrected_error_mw));
  }
  return [...byDay.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, errors]) => ({ date, hours: errors.official.length, official: toTenth(mean(errors.official)), corrected: toTenth(mean(errors.corrected)) }));
}

/** Percentage reduction of `value` versus `baseline` (positive = lower than baseline). */
export function reductionPct(baseline, value) {
  return isNumber(baseline) && isNumber(value) && baseline !== 0 ? (100 * (baseline - value)) / baseline : null;
}
