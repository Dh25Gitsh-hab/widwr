import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import dayAhead from "../test/fixtures/day_ahead_2026-01-15.json";
import health from "../test/fixtures/health.json";
import models from "../test/fixtures/models.json";
import day from "../test/fixtures/predictions_2026-01-15.json";
import { ModelEvaluation, PeriodScores } from "./AccuracyPanel.jsx";
import { Card } from "./Card.jsx";
import { DateSelector, ModelSelect } from "./FilterBar.jsx";
import { ForecastChart, ForecastTable } from "./ForecastChart.jsx";
import { MethodNotes } from "./MethodNotes.jsx";
import { ModelInfo } from "./ModelInfo.jsx";
import { DecisionTimeline, SummaryCards } from "./Overview.jsx";
import { RiskChart, RiskTable } from "./RiskChart.jsx";
import { ErrorBox } from "./StateViews.jsx";

const RANGE = { first: "2026-01-01", last: "2026-01-31" };

describe("SummaryCards (real API response for 15 Jan 2026)", () => {
  it("shows the backend's day scores and labels hindsight figures", () => {
    render(<SummaryCards day={day} dayAhead={dayAhead} />);
    expect(screen.getByText("864 MW")).toBeInTheDocument(); // corrected MAE from the API (864.0)
    expect(screen.getByText(/Official forecast 1,094 MW · corrected 864 MW/)).toBeInTheDocument();
    expect(screen.getByText(/Forecast error · MAE \(hindsight\)/)).toBeInTheDocument();
    expect(screen.getByText(/21.0% lower error/)).toBeInTheDocument(); // (1093.6 - 864.0) / 1093.6
    expect(screen.getByText(/Linear correction · Approved default/)).toBeInTheDocument();
    expect(screen.getByText(/Logistic regression · Approved default/)).toBeInTheDocument();
  });

  it("shows the peak probability returned by the API", () => {
    render(<SummaryCards day={day} dayAhead={dayAhead} />);
    const peak = Math.max(...day.points.map((p) => p.shortfall_probability));
    expect(screen.getByText(`${Math.round(peak * 100)}%`)).toBeInTheDocument();
  });
});

describe("DecisionTimeline", () => {
  it("shows the 09:50 cutoff, the run used and the later runs", () => {
    render(<DecisionTimeline dayAhead={dayAhead} />);
    expect(screen.getByText(/14 Jan 2026, 09:50 GMT/)).toBeInTheDocument();
    expect(screen.getByText(/Published .*09:30 GMT/)).toBeInTheDocument();
    expect(screen.getByText(/20 minutes before the cutoff/)).toBeInTheDocument();
    const later = dayAhead.runs.filter((run) => !run.available_at_decision).length;
    expect(screen.getByText(`${later} runs published after the cutoff`)).toBeInTheDocument();
  });
});

describe("DateSelector", () => {
  it("is limited to the backend's decision window and shows the delivery day", () => {
    render(<DateSelector value="2026-01-14" range={RANGE} error={null} onChange={() => {}} />);
    const input = screen.getByLabelText("Decision date (D-1)");
    expect(input).toHaveAttribute("min", "2026-01-01");
    expect(input).toHaveAttribute("max", "2026-01-31");
    expect(screen.getByText(/Delivery day Thu,? 15 Jan 2026/)).toBeInTheDocument();
  });

  it("disables stepping outside the window and steps by one day", () => {
    const onChange = vi.fn();
    render(<DateSelector value="2026-01-01" range={RANGE} error={null} onChange={onChange} />);
    expect(screen.getByLabelText("Previous decision day")).toBeDisabled();
    fireEvent.click(screen.getByLabelText("Next decision day"));
    expect(onChange).toHaveBeenCalledWith("2026-01-02");
  });

  it("shows a validation message", () => {
    render(<DateSelector value="2026-02-10" range={RANGE} error="Choose a decision date between Thu 1 Jan 2026 and Sat 31 Jan 2026." onChange={() => {}} />);
    expect(screen.getByLabelText("Decision date (D-1)")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText(/Choose a decision date between/)).toBeInTheDocument();
  });
});

describe("ModelSelect", () => {
  const optionTexts = () => screen.getAllByRole("option").map((option) => option.textContent);

  it("offers exactly the models from the API and shows the selected one", () => {
    render(<ModelSelect id="m" label="Model" models={models.corrected_forecast} value="linear_by_level" onChange={() => {}} />);
    expect(screen.getByLabelText("Model")).toHaveValue("linear_by_level");
    expect(screen.getByLabelText("Model")).toBeEnabled();
    expect(optionTexts()).toEqual(["Linear correction · Approved default", "Random Forest · Comparison"]);
  });

  it("says loading while there is no list yet", () => {
    render(<ModelSelect id="m" label="Model" value={null} onChange={() => {}} />);
    expect(screen.getByLabelText("Model")).toBeDisabled();
    expect(optionTexts()).toEqual(["Loading models…"]);
  });

  it("says unavailable, with no options, when the list failed to load", () => {
    render(<ModelSelect id="m" label="Model" value={null} failed onChange={() => {}} />);
    expect(screen.getByLabelText("Model")).toBeDisabled();
    expect(optionTexts()).toEqual(["Models unavailable"]);
  });
});

describe("tables (the non-hover way to read every chart value)", () => {
  it("list every hour of the day", () => {
    const { unmount } = render(<ForecastTable points={day.points} />);
    expect(screen.getAllByRole("row")).toHaveLength(25); // header + 24 hours
    expect(screen.getByRole("row", { name: /00:00 15,828 MW 13,467 MW 14,602 MW/ })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Actual (hindsight)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Corrected error (hindsight)" })).toBeInTheDocument();
    unmount();
    render(<RiskTable points={day.points} />);
    expect(screen.getByRole("row", { name: /00:00 78% 15,828 MW Yes/ })).toBeInTheDocument();
  });
});

describe("charts", () => {
  it("draw one line per series and a legend that names each one", () => {
    const { container } = render(<ForecastChart points={day.points} />);
    expect(container.querySelectorAll(".recharts-line")).toHaveLength(3);
    for (const label of ["Official forecast (decision time)", "Corrected forecast", "Actual generation (hindsight)"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("draw one probability bar per hour and mark the hours with a shortfall", () => {
    const { container } = render(<RiskChart points={day.points} />);
    expect(container.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(24);
    const occurred = day.points.filter((p) => p.shortfall_occurred).length;
    expect(container.querySelectorAll(".recharts-scatter circle")).toHaveLength(occurred); // none for hours without a shortfall
  });
});

describe("Model information and evaluation", () => {
  it("marks the approved defaults and comparison models from the API", () => {
    render(<ModelInfo models={models} />);
    expect(screen.getAllByText("Approved default")).toHaveLength(2);
    expect(screen.getAllByText("Comparison")).toHaveLength(2);
    expect(screen.getAllByText(/matches its sealed fingerprint/)).toHaveLength(4);
    expect(screen.getByText(models.status)).toBeInTheDocument();
  });

  it("shows the January test scores, e.g. logistic ROC-AUC", () => {
    render(<ModelEvaluation models={models} />);
    const shortfall = screen.getByRole("table", { name: "Shortfall risk" });
    const logistic = within(shortfall).getByRole("row", { name: /Logistic regression/ });
    expect(within(logistic).getByText("0.755")).toBeInTheDocument();
  });

  it("explains how to read the scores, using the API's own shortfall rate", () => {
    render(<ModelEvaluation models={models} />);
    expect(screen.getByText(/0\.5 is no better than chance/)).toHaveTextContent("66% of these January hours");
  });

  it("describes each model in plain language and keeps implementation detail under Technical details", () => {
    render(<ModelInfo models={models} />);
    expect(screen.getByText(/straight-line correction learned from past forecast errors/)).toBeInTheDocument();
    expect(screen.getAllByText("Official forecast size")).toHaveLength(2);
    expect(screen.getAllByText(/^9 inputs known at the 09:50 UK cutoff: official forecast size, forecast lead time/)).toHaveLength(2);
    expect(screen.getAllByText(/^Approved default, chosen after the January test/)).toHaveLength(2);
    expect(screen.getAllByText(/^Chosen by the rule fixed before January/)).toHaveLength(2);
    expect(screen.getByText("How the defaults were chosen and what the scores mean")).toBeInTheDocument();
    for (const raw of ["forecast_revision_mw", "holdout_feb2026/models/regression_linear_by_level.joblib", "linear_by_level"]) {
      for (const element of screen.getAllByText(raw)) expect(element.closest("details")).not.toBeNull();
    }
  });
});

describe("MethodNotes", () => {
  it("states the source, metrics and limitations, with diagnostics under Technical details", () => {
    const { container } = render(<MethodNotes health={health} />);
    expect(screen.getByText(/^Saved Elexon snapshot, downloaded Sun,? 25 Oct 2026, 19:31 UK time$/)).toBeInTheDocument();
    for (const term of ["MAE (mean absolute error)", "RMSE (root mean squared error)", "Mean bias", "ROC-AUC", "PR-AUC", "Brier score"]) {
      expect(screen.getByText(term)).toBeInTheDocument();
    }
    expect(screen.getByText(/one month of decisions .*31 days/)).toBeInTheDocument();
    expect(screen.getByText(/not proof of long-term superiority/)).toBeInTheDocument();
    const details = container.querySelector("details");
    expect(within(details).getByText("http://127.0.0.1:8000")).toBeInTheDocument();
    expect(within(details).getByText("4,081")).toBeInTheDocument(); // forecast values published after the hour they forecast
  });

  it("still lists the API address under Technical details when the health check failed", () => {
    render(<MethodNotes health={null} />);
    expect(screen.getByText("http://127.0.0.1:8000").closest("details")).not.toBeNull();
  });
});

describe("Card table view", () => {
  it("dims the table and marks it busy only while its data reloads", () => {
    const table = (
      <table>
        <tbody>
          <tr>
            <td>1,000 MW</td>
          </tr>
        </tbody>
      </table>
    );
    const { rerender } = render(
      <Card title="Panel" table={table}>
        chart
      </Card>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Table" }));
    expect(screen.getByRole("table").closest("[aria-busy]")).toBeNull();
    rerender(
      <Card title="Panel" table={table} busy>
        chart
      </Card>,
    );
    expect(screen.getByRole("table").closest("[aria-busy='true']")).toHaveClass("is-refreshing");
  });

  it("keeps a badge a separate word in the card's accessible name", () => {
    render(
      <Card title="Whole decision period" badge={{ text: "Hindsight" }}>
        body
      </Card>,
    );
    expect(screen.getByRole("region", { name: "Whole decision period Hindsight" })).toBeInTheDocument();
  });
});

describe("PeriodScores", () => {
  const noActual = (p) => ({ ...p, actual_mw: null, error_mw: null, corrected_error_mw: null, shortfall_occurred: null });

  it("counts every hour when all have an actual", () => {
    render(<PeriodScores month={day} />);
    expect(screen.getByText(/lower than the official forecast over 24 hours$/)).toBeInTheDocument();
    expect(screen.queryByText(/excluded/)).toBeNull();
  });

  it("counts only hours with an actual as scored and names the excluded ones", () => {
    render(<PeriodScores month={{ ...day, points: day.points.map((p, i) => (i < 2 ? noActual(p) : p)) }} />);
    expect(screen.getByText(/over 22 hours \(2 hours without actual generation excluded\)/)).toBeInTheDocument();
  });

  it("uses the singular for one excluded hour", () => {
    render(<PeriodScores month={{ ...day, points: day.points.map((p, i) => (i === 0 ? noActual(p) : p)) }} />);
    expect(screen.getByText(/over 23 hours \(1 hour without actual generation excluded\)/)).toBeInTheDocument();
  });
});

describe("ErrorBox", () => {
  it("names the problem and offers a retry", () => {
    const onRetry = vi.fn();
    render(<ErrorBox error={{ kind: "unavailable", message: "ML unavailable: model file missing" }} onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Service unavailable");
    expect(screen.getByRole("alert")).toHaveTextContent("ML unavailable: model file missing");
    fireEvent.click(screen.getByText("Try again"));
    expect(onRetry).toHaveBeenCalled();
  });

  it("shows a plain message; the reason, without paths or commands, is under Technical details", () => {
    const error = {
      kind: "unavailable",
      status: 503,
      url: "http://127.0.0.1:8000/api/ml/models",
      message:
        "ML unavailable: [Errno 2] No such file or directory: 'E:\\\\GridBricks Job\\\\backend\\\\ml\\\\models\\\\error_regressor.joblib' - the artifacts are built by `python -m ml.train`",
    };
    const { container } = render(<ErrorBox error={error} />);
    expect(screen.getByText("This part of the service is unavailable right now. Other sections may still work.")).toBeInTheDocument();
    expect(screen.getByText(/'error_regressor\.joblib'\. Setup steps are in the project README\./).closest("details")).not.toBeNull();
    expect(container.textContent).not.toMatch(/GridBricks Job|E:\\|python -m/);
  });
});
