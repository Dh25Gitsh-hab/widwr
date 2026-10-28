import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import dayAhead from "./test/fixtures/day_ahead_2026-01-15.json";
import health from "./test/fixtures/health.json";
import models from "./test/fixtures/models.json";
import predictionsRf from "./test/fixtures/predictions_2026-01-15_rf.json";
import predictions from "./test/fixtures/predictions_2026-01-15.json";
import summary from "./test/fixtures/summary.json";

function respond(body, status = 200) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) });
}

/** A fake network that answers the dashboard's routes with real captured backend responses. */
function backend(overrides = {}) {
  return vi.fn((url) => {
    const { pathname, searchParams } = new URL(url);
    if (overrides[pathname]) return overrides[pathname](url);
    const routes = {
      "/api/health": health,
      "/api/ml/models": models,
      "/api/analytics/summary": summary,
      "/api/forecasts/day-ahead": dayAhead,
      "/api/ml/predictions": searchParams.get("regressor") === "random_forest" ? predictionsRf : predictions,
    };
    return routes[pathname] ? respond(routes[pathname]) : respond({ detail: "Not Found" }, 404);
  });
}

const requested = (fragment) => globalThis.fetch.mock.calls.some(([url]) => url.includes(fragment));

/** A response the test releases by hand, to hold one request while others complete. */
function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, release: () => act(async () => resolve()) };
}

// A second delivery day (2 Jan) with distinct values, so a mix-up between days is visible.
const predictionsJan2 = {
  ...predictions,
  period: { first: "2026-01-02", last: "2026-01-02" },
  scores: { official: { ...predictions.scores.official, mae: 2000 }, corrected: { ...predictions.scores.corrected, mae: 999 } },
};
const dayAheadJan2 = {
  ...dayAhead,
  delivery_date: "2026-01-02",
  decision_date: "2026-01-01",
  expected: { ...dayAhead.expected, min_mw: 1111, max_mw: 2222 },
};
const dayPredictions = (url) => respond(url.includes("start=2026-01-02") ? predictionsJan2 : predictions);
const dayAheadFor = (url) => respond(url.includes("delivery_date=2026-01-02") ? dayAheadJan2 : dayAhead);
const JAN2_LOW_HIGH = /Low 1,111 MW at 16:00 · high 2,222 MW at 00:00/;
const JAN15_LOW_HIGH = /Low 8,957 MW at 16:00 · high 15,828 MW at 00:00/;
const JAN2_MAE = /Official forecast 2,000 MW · corrected 999 MW/;
const JAN15_MAE = /Official forecast 1,094 MW · corrected 864 MW/;
const chooseDecision = (value) => fireEvent.change(screen.getByLabelText("Decision date (D-1)"), { target: { value } });

beforeEach(() => {
  globalThis.fetch = backend();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App with the backend available", () => {
  it("renders API data: header window, summary figures and model roles", async () => {
    render(<App />);
    expect(await screen.findByText("1 Jan – 31 Jan 2026")).toBeInTheDocument();
    expect((await screen.findAllByText("864 MW")).length).toBeGreaterThan(0);
    expect((await screen.findAllByText("Approved default")).length).toBe(2);
    expect(screen.getByText("Backend connected")).toBeInTheDocument();
  });

  it("asks for the delivery day after the chosen decision date", async () => {
    render(<App />);
    await waitFor(() => expect(requested("start=2026-01-02&end=2026-01-02")).toBe(true)); // default: first decision day
    fireEvent.change(screen.getByLabelText("Decision date (D-1)"), { target: { value: "2026-01-14" } });
    await waitFor(() => expect(requested("start=2026-01-15&end=2026-01-15")).toBe(true));
    expect(requested("delivery_date=2026-01-15")).toBe(true);
  });

  it("rejects a decision date outside January 2026 without calling the API", async () => {
    render(<App />);
    await screen.findByText("1 Jan – 31 Jan 2026");
    fireEvent.change(screen.getByLabelText("Decision date (D-1)"), { target: { value: "2026-02-10" } });
    expect(await screen.findByText(/Choose a decision date between/)).toBeInTheDocument();
    expect(requested("2026-02-11")).toBe(false);
  });

  it("switches to a comparison model on request and says so", async () => {
    render(<App />);
    await screen.findAllByText("Approved default");
    fireEvent.change(screen.getByLabelText("Corrected-forecast model"), { target: { value: "random_forest" } });
    await waitFor(() => expect(requested("regressor=random_forest")).toBe(true));
    expect(screen.getByText(/Showing a comparison model/)).toBeInTheDocument();
  });
});

describe("App never mixes days or presents old values as current", () => {
  it("does not show the previous day's low/high next to the new day's forecast while the day-ahead view loads", async () => {
    const later = deferred();
    globalThis.fetch = backend({
      "/api/ml/predictions": dayPredictions,
      "/api/forecasts/day-ahead": (url) => (url.includes("delivery_date=2026-01-15") ? later.promise.then(() => dayAheadFor(url)) : dayAheadFor(url)),
    });
    render(<App />);
    expect(await screen.findByText(JAN2_LOW_HIGH)).toBeInTheDocument();
    chooseDecision("2026-01-14");
    expect(await screen.findByText(JAN15_MAE)).toBeInTheDocument(); // 15 Jan predictions are in, its day-ahead is not
    expect(screen.queryByText(JAN2_LOW_HIGH)).toBeNull();
    await later.release();
    expect(await screen.findByText(JAN15_LOW_HIGH)).toBeInTheDocument();
  });

  it("does not show the new day's low/high next to the previous day's forecast while the predictions load", async () => {
    const later = deferred();
    globalThis.fetch = backend({
      "/api/ml/predictions": (url) => (url.includes("start=2026-01-15") ? later.promise.then(() => dayPredictions(url)) : dayPredictions(url)),
      "/api/forecasts/day-ahead": dayAheadFor,
    });
    render(<App />);
    expect(await screen.findByText(JAN2_LOW_HIGH)).toBeInTheDocument();
    chooseDecision("2026-01-14");
    expect(await screen.findByText(/14 Jan 2026, 09:50 GMT/)).toBeInTheDocument(); // 15 Jan day-ahead is in
    expect(screen.getByText(JAN2_MAE)).toBeInTheDocument(); // 2 Jan tiles stay (dimmed) until 15 Jan arrives
    expect(screen.queryByText(JAN15_LOW_HIGH)).toBeNull();
    expect(screen.queryByText(JAN2_LOW_HIGH)).toBeNull();
    await later.release();
    expect(await screen.findByText(JAN15_MAE)).toBeInTheDocument();
    expect(await screen.findByText(JAN15_LOW_HIGH)).toBeInTheDocument();
  });

  it("dims a day table and marks it busy while the next delivery day loads", async () => {
    const later = deferred();
    globalThis.fetch = backend({
      "/api/ml/predictions": (url) => (url.includes("start=2026-01-15") ? later.promise.then(() => dayPredictions(url)) : dayPredictions(url)),
    });
    render(<App />);
    const card = await screen.findByRole("region", { name: "Official vs corrected forecast" });
    fireEvent.click(await within(card).findByRole("button", { name: "Table" }));
    expect(within(card).getByRole("table").closest("[aria-busy]")).toBeNull();
    chooseDecision("2026-01-14");
    expect(within(card).getByRole("table").closest("[aria-busy='true']")).toHaveClass("is-refreshing");
    await later.release();
    await waitFor(() => expect(within(card).getByRole("table").closest("[aria-busy]")).toBeNull());
  });

  it("dims the daily-error table and marks it busy while another model's month loads", async () => {
    const later = deferred();
    const month = (url) => respond(url.includes("regressor=random_forest") ? predictionsRf : predictions);
    globalThis.fetch = backend({
      "/api/ml/predictions": (url) => (url.includes("regressor=random_forest") && !url.includes("start=") ? later.promise.then(() => month(url)) : month(url)),
    });
    render(<App />);
    const card = await screen.findByRole("region", { name: /^Daily error across the decision period/ });
    fireEvent.click(await within(card).findByRole("button", { name: "Table" }));
    expect(within(card).getByRole("table").closest("[aria-busy]")).toBeNull();
    fireEvent.change(screen.getByLabelText("Corrected-forecast model"), { target: { value: "random_forest" } });
    expect(within(card).getByRole("table").closest("[aria-busy='true']")).toHaveClass("is-refreshing");
    await later.release();
    await waitFor(() => expect(within(card).getByRole("table").closest("[aria-busy]")).toBeNull());
  });
});

describe("App model selectors", () => {
  const regressorSelect = () => screen.getByLabelText("Corrected-forecast model");
  const classifierSelect = () => screen.getByLabelText("Shortfall-risk model");
  const filters = () => screen.getByRole("region", { name: "Filters" });
  const optionTexts = (select) => within(select).getAllByRole("option").map((option) => option.textContent);
  const modelParamSent = () => globalThis.fetch.mock.calls.some(([url]) => /[?&](regressor|classifier)=/.test(url));

  it("lists the served models and keeps the approved defaults selected", async () => {
    render(<App />);
    await waitFor(() => expect(regressorSelect()).toBeEnabled());
    expect(regressorSelect()).toHaveValue("linear_by_level");
    expect(classifierSelect()).toHaveValue("logistic_by_level");
    expect(optionTexts(regressorSelect())).toEqual(["Linear correction · Approved default", "Random Forest · Comparison"]);
    expect(optionTexts(classifierSelect())).toEqual(["Logistic regression · Approved default", "Random Forest · Comparison"]);
    await waitFor(() => expect(requested("start=2026-01-02&end=2026-01-02")).toBe(true));
    expect(modelParamSent()).toBe(false); // no explicit choice: the backend applies its approved defaults
    expect(screen.queryByText(/Showing a comparison model/)).toBeNull();
  });

  it("says the models are loading only while the list is on its way", async () => {
    const later = deferred();
    globalThis.fetch = backend({ "/api/ml/models": () => later.promise.then(() => respond(models)) });
    render(<App />);
    await waitFor(() => expect(requested("/api/ml/models")).toBe(true));
    for (const select of [regressorSelect(), classifierSelect()]) {
      expect(select).toBeDisabled();
      expect(optionTexts(select)).toEqual(["Loading models…"]);
    }
    await later.release();
    await waitFor(() => expect(regressorSelect()).toHaveValue("linear_by_level"));
    expect(classifierSelect()).toHaveValue("logistic_by_level");
  });

  it("shows the list as unavailable when /api/ml/models fails, offers no invented model, and recovers on retry", async () => {
    let fail = true;
    globalThis.fetch = backend({
      "/api/ml/models": () => (fail ? respond({ detail: "ML unavailable: model file missing" }, 503) : respond(models)),
    });
    render(<App />);
    const alert = await within(filters()).findByRole("alert");
    expect(alert).toHaveTextContent("Service unavailable");
    expect(alert).toHaveTextContent("ML unavailable: model file missing");
    for (const select of [regressorSelect(), classifierSelect()]) {
      expect(select).toBeDisabled();
      expect(optionTexts(select)).toEqual(["Models unavailable"]);
    }
    expect(screen.queryByText("Loading models…")).toBeNull();
    await waitFor(() => expect(requested("start=2026-01-02&end=2026-01-02")).toBe(true));
    expect(modelParamSent()).toBe(false); // predictions still name no model, so the backend's defaults apply

    fail = false;
    fireEvent.click(within(filters()).getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(regressorSelect()).toHaveValue("linear_by_level"));
    expect(classifierSelect()).toHaveValue("logistic_by_level");
    expect(regressorSelect()).toBeEnabled();
    expect(within(filters()).queryByRole("alert")).toBeNull();
  });

  it("shows the list as unavailable, not loading, when the health check fails", async () => {
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<App />);
    await screen.findByText("Backend unreachable");
    for (const select of [regressorSelect(), classifierSelect()]) {
      expect(optionTexts(select)).toEqual(["Models unavailable"]);
    }
    expect(requested("/api/ml/models")).toBe(false);
  });
});

describe("App health check", () => {
  it("says it is checking the backend while the health check is pending, then connected", async () => {
    const later = deferred();
    globalThis.fetch = backend({ "/api/health": () => later.promise.then(() => respond(health)) });
    render(<App />);
    expect(within(screen.getByRole("complementary")).getByText("Checking backend…")).toBeInTheDocument();
    expect(screen.getAllByText("Checking backend…").length).toBeGreaterThan(1); // the data blocks say so too
    expect(screen.queryByText("No data for this selection.")).toBeNull();
    expect(screen.queryByText("Backend connected")).toBeNull();
    expect(screen.getByText("Loading the decision window…")).toBeInTheDocument();
    await later.release();
    expect(await screen.findByText("Backend connected")).toBeInTheDocument();
    expect(screen.queryByText("Checking backend…")).toBeNull();
  });

  it("reports a backend that answers but is unavailable (503) without calling it connected or unreachable", async () => {
    globalThis.fetch = backend({ "/api/health": () => respond({ status: "unavailable", detail: "Data unavailable: snapshot missing" }, 503) });
    render(<App />);
    expect(await screen.findByText("The backend reported a problem.")).toBeInTheDocument();
    expect(screen.getByText("Data unavailable: snapshot missing")).toBeInTheDocument();
    expect(screen.getByText("Backend unavailable")).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable: the backend reported a problem.").length).toBeGreaterThan(0);
    expect(screen.getByText("Decision window unavailable until the backend health check succeeds.")).toBeInTheDocument();
    for (const text of ["Backend connected", "Backend unreachable", "Checking backend…", "No data for this selection."]) {
      expect(screen.queryByText(text)).toBeNull();
    }
  });
});

describe("App keeps developer details out of the normal view", () => {
  it("shows the backend status without the API address, which is under Technical details", async () => {
    render(<App />);
    expect(await within(screen.getByRole("complementary")).findByText("Backend connected")).toBeInTheDocument();
    expect(within(screen.getByRole("complementary")).queryByText("http://127.0.0.1:8000")).toBeNull();
    expect((await screen.findByText("http://127.0.0.1:8000")).closest("details")).not.toBeNull();
  });

  it("names the correction model once in the forecast subtitle and marks the daily error as hindsight", async () => {
    render(<App />);
    expect(await screen.findByText("Linear correction · MW per hour, UK time")).toBeInTheDocument();
    expect(screen.queryByText(/correction correction/)).toBeNull();
    expect(screen.getByText("Logistic regression · hourly probability from the shortfall model")).toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Daily error across the decision period Hindsight" })).toBeInTheDocument();
  });

  it("gives no developer commands when the backend is unreachable; the reason is under Technical details", async () => {
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<App />);
    expect(await screen.findByText("The dashboard cannot reach the backend.")).toBeInTheDocument();
    expect(screen.getByText("Forecasts and results will appear as soon as the backend is available.")).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the backend at/).closest("details")).not.toBeNull();
    expect(document.body.textContent).not.toMatch(/python|uvicorn|\.venv|npm run/);
  });
});

describe("App error states", () => {
  it("shows one clear banner when the backend cannot be reached (down or CORS)", async () => {
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<App />);
    expect(await screen.findByText("The dashboard cannot reach the backend.")).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the backend at http:\/\/127\.0\.0\.1:8000/)).toBeInTheDocument();
    expect(screen.getByText("Backend unreachable")).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable until the backend is reachable.").length).toBeGreaterThan(0);
  });

  it("keeps the forecast views working when only the ML service is unavailable", async () => {
    const unavailable = () => respond({ detail: "ML unavailable: model file missing" }, 503);
    globalThis.fetch = backend({ "/api/ml/models": unavailable, "/api/ml/predictions": unavailable });
    render(<App />);
    expect((await screen.findAllByText("ML unavailable: model file missing")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Service unavailable").length).toBeGreaterThan(0);
    expect(await screen.findByText(/14 Jan 2026, 09:50 GMT/)).toBeInTheDocument(); // day-ahead timeline still works
  });
});
