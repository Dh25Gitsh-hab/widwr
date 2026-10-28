// @vitest-environment node
/**
 * Opt-in checks against the REAL backend (skipped unless LIVE_API_URL is set):
 *   LIVE_API_URL=http://127.0.0.1:8000 npm run test:live
 * They confirm that the routes and fields the dashboard relies on exist exactly as used.
 */
import { describe, expect, it } from "vitest";
import { apiGet } from "./client.js";

const BASE = process.env.LIVE_API_URL;
const get = (path, params) => apiGet(path, params, { baseUrl: BASE });

describe.skipIf(!BASE)("live backend", () => {
  it("reports January 2026 decisions and the approved default models", async () => {
    const health = await get("/api/health");
    expect(health.status).toBe("ok");
    expect(health.decision_dates).toEqual({ first: "2026-01-01", last: "2026-01-31" });
    expect(health.ml_models).toContain("linear_by_level (default)");
    expect(health.ml_models).toContain("logistic_by_level (default)");
  });

  it("serves hourly predictions with every field the dashboard reads", async () => {
    const month = await get("/api/ml/predictions");
    expect(month.points).toHaveLength(744);
    expect(Object.keys(month.points[0])).toEqual(
      expect.arrayContaining(["start_time", "publish_time", "decision_cutoff", "forecast_mw", "corrected_forecast_mw", "shortfall_probability", "actual_mw", "error_mw", "corrected_error_mw", "shortfall_occurred"]),
    );
    expect([month.regressor.role, month.classifier.role]).toEqual(["default", "default"]);
    expect(Object.keys(month.scores)).toEqual(["official", "corrected"]);
  });

  it("switches to the comparison models when asked", async () => {
    const day = await get("/api/ml/predictions", { start: "2026-01-15", end: "2026-01-15", regressor: "random_forest", classifier: "random_forest" });
    expect(day.points).toHaveLength(24);
    expect([day.regressor.role, day.classifier.role]).toEqual(["comparison", "comparison"]);
  });

  it("returns the decision view and model listing the dashboard uses", async () => {
    const ahead = await get("/api/forecasts/day-ahead", { delivery_date: "2026-01-15" });
    expect(ahead).toMatchObject({ decision_date: "2026-01-14", decision_cutoff: "2026-01-14T09:50:00Z" });
    expect(ahead.runs.some((run) => run.used_for_decision)).toBe(true);
    const models = await get("/api/ml/models");
    expect(models.decision_dates).toEqual({ first: "2026-01-01", last: "2026-01-31" });
    expect(models.corrected_forecast.map((m) => m.role)).toEqual(["default", "comparison"]);
  });

  it("rejects a date outside the January window with 422", async () => {
    await expect(get("/api/ml/predictions", { start: "2026-02-10" })).rejects.toMatchObject({ kind: "validation", status: 422 });
  });
});
