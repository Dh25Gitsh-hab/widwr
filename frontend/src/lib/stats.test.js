import { describe, expect, it } from "vitest";
import day from "../test/fixtures/predictions_2026-01-15.json";
import { mw } from "./format.js";
import { dailyErrors, mean, reductionPct, summarizeDay } from "./stats.js";

const POINTS = [
  { start_time: "2026-01-15T00:00:00Z", forecast_mw: 10000, corrected_forecast_mw: 9000, shortfall_probability: 0.2, actual_mw: 9500, error_mw: 500, corrected_error_mw: -500, shortfall_occurred: false },
  { start_time: "2026-01-15T01:00:00Z", forecast_mw: 12000, corrected_forecast_mw: 10000, shortfall_probability: 0.8, actual_mw: 10500, error_mw: 1500, corrected_error_mw: -500, shortfall_occurred: true },
  { start_time: "2026-01-15T02:00:00Z", forecast_mw: 14000, corrected_forecast_mw: 12000, shortfall_probability: 0.6, actual_mw: null, error_mw: null, corrected_error_mw: null, shortfall_occurred: null },
];

describe("summaries of API points", () => {
  it("averages, finds the peak risk and counts outcomes", () => {
    const s = summarizeDay(POINTS);
    expect(s.meanOfficial).toBe(12000);
    expect(s.meanCorrected).toBeCloseTo(10333.33, 1);
    expect(s.peakRisk).toBe(0.8);
    expect(s.peakRiskTime).toBe("2026-01-15T01:00:00Z");
    expect(s.highRiskHours).toBe(2);
    expect(s.hoursWithOutcome).toBe(2); // the hour without an actual is not counted
    expect(s.observedShortfalls).toBe(1);
  });

  it("returns null for an empty day", () => {
    expect(summarizeDay([])).toBeNull();
  });

  it("computes daily MAE from hours with an actual only", () => {
    expect(dailyErrors(POINTS)).toEqual([{ date: "2026-01-15", hours: 2, official: 1000, corrected: 500 }]);
  });

  it("matches the backend's own MAE for a real day", () => {
    const [only] = dailyErrors(day.points);
    expect(only.official).toBeCloseTo(day.scores.official.mae, 0);
    expect(only.corrected).toBeCloseTo(day.scores.corrected.mae, 0);
    expect(mw(only.official)).toBe(mw(day.scores.official.mae));
    expect(mw(only.corrected)).toBe(mw(day.scores.corrected.mae));
  });

  it("rounds daily MAE to 0.1 MW like the API, so the month view and the day tile show the same whole MW", () => {
    // 11 Jan pattern: the hourly errors average 4,948.458 MW; the API reports 4,948.5, shown as 4,949 MW.
    const points = Array.from({ length: 24 }, (_, h) => ({
      start_time: `2026-01-11T${String(h).padStart(2, "0")}:00:00Z`,
      actual_mw: 10000,
      error_mw: h === 0 ? 4959 : 4948,
      corrected_error_mw: -100,
    }));
    const [only] = dailyErrors(points);
    expect(only.official).toBe(4948.5);
    expect(mw(only.official)).toBe("4,949 MW");
    expect(only.corrected).toBe(100);
  });

  it("handles missing values safely", () => {
    expect(mean([null, undefined])).toBeNull();
    expect(reductionPct(1000, 500)).toBe(50);
    expect(reductionPct(null, 500)).toBeNull();
  });
});
