/**
 * Thin client for the FastAPI backend.
 *
 * Every function maps to one existing GET endpoint. Responses are returned exactly as the
 * backend sends them; nothing is invented or filled in here. Failures become an ApiError
 * with a `kind` the UI can explain:
 *   network      the backend could not be reached (not running, wrong URL, or CORS blocked)
 *   validation   HTTP 422 - a request parameter was rejected
 *   unavailable  HTTP 503 - data or ML models could not be loaded on the server
 *   http         any other non-2xx status
 *   format       the response was not JSON
 */

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(message, { kind, status = null, url = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.url = url;
  }
}

function describeDetail(detail) {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => [item.loc?.slice(1).join("."), item.msg].filter(Boolean).join(": ")).join("; ");
  }
  return JSON.stringify(detail);
}

export async function apiGet(path, params = {}, { signal, baseUrl = API_BASE_URL, fetchImpl = globalThis.fetch } = {}) {
  const url = new URL(baseUrl + path);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  }

  let response;
  try {
    response = await fetchImpl(url.toString(), { signal, headers: { Accept: "application/json" } });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    const origin = globalThis.location?.origin ?? "this page";
    throw new ApiError(
      `Cannot reach the backend at ${baseUrl}. Check that it is running and that its CORS settings ` +
        `(WFM_CORS_ORIGINS) allow ${origin}.`,
      { kind: "network", url: url.toString() },
    );
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const kind = response.status === 422 ? "validation" : response.status === 503 ? "unavailable" : "http";
    throw new ApiError(describeDetail(body?.detail) || `The backend returned HTTP ${response.status}.`, {
      kind,
      status: response.status,
      url: url.toString(),
    });
  }
  if (body === null) {
    throw new ApiError("The backend returned a response that is not JSON.", { kind: "format", status: response.status, url: url.toString() });
  }
  return body;
}

export const api = {
  /** Service status, decision/delivery windows and loaded models. */
  health: (options) => apiGet("/api/health", {}, options),
  /** Served models: roles, inputs, January test scores, selection history. */
  models: (options) => apiGet("/api/ml/models", {}, options),
  /** Hourly corrected forecast + shortfall probability. Omitted model names use the approved defaults. */
  predictions: ({ start, end, regressor, classifier } = {}, options) =>
    apiGet("/api/ml/predictions", { start, end, regressor, classifier }, options),
  /** The decision view for one delivery day: cutoff, forecast run used, every run. */
  dayAhead: (deliveryDate, options) => apiGet("/api/forecasts/day-ahead", { delivery_date: deliveryDate }, options),
  /** Accuracy of the official decision-time forecast over the whole window. */
  summary: (options) => apiGet("/api/analytics/summary", {}, options),
};
