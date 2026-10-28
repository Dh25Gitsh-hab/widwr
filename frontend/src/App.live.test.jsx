/**
 * Opt-in end-to-end render against the REAL backend (skipped unless LIVE_API_URL is set).
 * The dashboard fetches from VITE_API_BASE_URL (default http://127.0.0.1:8000), so run the
 * backend there first:  LIVE_API_URL=http://127.0.0.1:8000 npm run test:live
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App.jsx";

describe.skipIf(!process.env.LIVE_API_URL)("dashboard against the live backend", () => {
  it("loads real data into every section", async () => {
    render(<App />);
    expect(await screen.findByText("Backend connected", {}, { timeout: 10000 })).toBeInTheDocument();
    expect(await screen.findByText("1 Jan – 31 Jan 2026", {}, { timeout: 10000 })).toBeInTheDocument();
    // Overview tiles and the decision timeline for the first decision day (delivery 2 Jan 2026)
    expect(await screen.findByText("Official forecast · daily mean", {}, { timeout: 10000 })).toBeInTheDocument();
    expect(await screen.findByText(/1 Jan 2026, 09:50 GMT/, {}, { timeout: 10000 })).toBeInTheDocument();
    // Model information from /api/ml/models
    expect((await screen.findAllByText("Approved default", {}, { timeout: 10000 })).length).toBe(2);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument(); // no error state anywhere
  }, 20000);
});
