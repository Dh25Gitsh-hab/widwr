import { describe, expect, it, vi } from "vitest";
import { ApiError, apiGet } from "./client.js";

const BASE = "http://backend.test";

function fakeFetch(status, body, { json = true } = {}) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: json ? () => Promise.resolve(body) : () => Promise.reject(new SyntaxError("Unexpected token")),
  });
}

describe("apiGet", () => {
  it("builds the URL from the exact route and skips empty parameters", async () => {
    const fetchImpl = fakeFetch(200, { ok: true });
    await apiGet("/api/ml/predictions", { start: "2026-01-15", end: "2026-01-15", regressor: undefined, classifier: null }, { baseUrl: BASE, fetchImpl });
    expect(fetchImpl.mock.calls[0][0]).toBe(`${BASE}/api/ml/predictions?start=2026-01-15&end=2026-01-15`);
  });

  it("returns the JSON body unchanged", async () => {
    const body = { status: "ok", decision_dates: { first: "2026-01-01", last: "2026-01-31" } };
    await expect(apiGet("/api/health", {}, { baseUrl: BASE, fetchImpl: fakeFetch(200, body) })).resolves.toEqual(body);
  });

  it("explains a 422 validation error from FastAPI's list format", async () => {
    const body = { detail: [{ loc: ["query", "start"], msg: "Input should be a valid date" }] };
    const error = await apiGet("/api/ml/predictions", {}, { baseUrl: BASE, fetchImpl: fakeFetch(422, body) }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ kind: "validation", status: 422, message: "start: Input should be a valid date" });
  });

  it("passes on the backend's reason for a 503", async () => {
    const error = await apiGet("/api/ml/models", {}, { baseUrl: BASE, fetchImpl: fakeFetch(503, { detail: "ML unavailable: model file missing" }) }).catch((e) => e);
    expect(error).toMatchObject({ kind: "unavailable", status: 503, message: "ML unavailable: model file missing" });
  });

  it("turns a network or CORS failure into a clear message naming the backend URL", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await apiGet("/api/health", {}, { baseUrl: BASE, fetchImpl }).catch((e) => e);
    expect(error.kind).toBe("network");
    expect(error.message).toContain(`Cannot reach the backend at ${BASE}`);
    expect(error.message).toContain("CORS");
  });

  it("rejects a non-JSON success response", async () => {
    const error = await apiGet("/api/health", {}, { baseUrl: BASE, fetchImpl: fakeFetch(200, null, { json: false }) }).catch((e) => e);
    expect(error.kind).toBe("format");
  });

  it("lets an aborted request stay an AbortError (not shown as a failure)", async () => {
    const abort = Object.assign(new Error("aborted"), { name: "AbortError" });
    await expect(apiGet("/api/health", {}, { baseUrl: BASE, fetchImpl: vi.fn().mockRejectedValue(abort) })).rejects.toBe(abort);
  });
});
