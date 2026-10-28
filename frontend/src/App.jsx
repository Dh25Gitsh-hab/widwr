import { useState } from "react";
import { api } from "./api/client.js";
import { ModelEvaluation, OfficialForecastScores, PeriodScores } from "./components/AccuracyPanel.jsx";
import { Card, Section } from "./components/Card.jsx";
import { DateSelector, FilterBar, ModelSelect } from "./components/FilterBar.jsx";
import { ForecastChart, ForecastTable } from "./components/ForecastChart.jsx";
import { Header, Sidebar, StatusBanner } from "./components/Layout.jsx";
import { MethodNotes } from "./components/MethodNotes.jsx";
import { ModelInfo } from "./components/ModelInfo.jsx";
import { MonthAccuracyChart, MonthAccuracyTable } from "./components/MonthAccuracyChart.jsx";
import { DecisionTimeline, SummaryCards } from "./components/Overview.jsx";
import { RiskChart, RiskTable } from "./components/RiskChart.jsx";
import { AsyncBlock, ErrorBox } from "./components/StateViews.jsx";
import { useApi } from "./hooks/useApi.js";
import { addDays, inRange, isIsoDate, longDate } from "./lib/dates.js";
import { defaultModel, modelLabel } from "./lib/models.js";

function validateDecision(value, range) {
  if (!range || value === null) return null;
  if (!isIsoDate(value) || !inRange(value, range.first, range.last)) {
    return `Choose a decision date between ${longDate(range.first)} and ${longDate(range.last)}.`;
  }
  return null;
}

/** What an API-backed block says when it has nothing to show. Until /api/health answers, nothing else is requested. */
function idleMessage(health, dateError) {
  if (health.error) return health.error.kind === "network" ? "Unavailable until the backend is reachable." : "Unavailable: the backend reported a problem.";
  if (!health.data) return "Checking backend…";
  if (dateError) return "Choose a valid decision date.";
  return "No data for this selection.";
}

export default function App() {
  // Health first: it supplies the decision window, and if the backend is down we show one clear banner.
  const health = useApi("health", (signal) => api.health({ signal }));
  const ready = Boolean(health.data);
  const models = useApi(ready ? "models" : null, (signal) => api.models({ signal }));
  const summary = useApi(ready ? "summary" : null, (signal) => api.summary({ signal }));

  const range = health.data?.decision_dates ?? null;
  const [decisionInput, setDecisionInput] = useState(null);
  const decisionDate = decisionInput ?? range?.first ?? null;
  const dateError = validateDecision(decisionDate, range);
  const deliveryDate = range && decisionDate && !dateError ? addDays(decisionDate, 1) : null;

  // null = the backend's approved default; a name = an explicit choice (default or comparison).
  const [regressor, setRegressor] = useState(null);
  const [classifier, setClassifier] = useState(null);
  const regressorName = regressor ?? defaultModel(models.data?.corrected_forecast);
  const classifierName = classifier ?? defaultModel(models.data?.shortfall_risk);
  const modelParams = { regressor: regressor ?? undefined, classifier: classifier ?? undefined };
  // The model list is requested once health has answered; if either fails, there is no list to offer.
  const modelsFailed = Boolean(health.error || models.error);
  const comparison =
    models.data &&
    (regressorName !== defaultModel(models.data.corrected_forecast) || classifierName !== defaultModel(models.data.shortfall_risk));

  const day = useApi(deliveryDate ? `day|${deliveryDate}|${regressor}|${classifier}` : null, (signal) =>
    api.predictions({ start: deliveryDate, end: deliveryDate, ...modelParams }, { signal }),
  );
  const dayAhead = useApi(deliveryDate ? `ahead|${deliveryDate}` : null, (signal) => api.dayAhead(deliveryDate, { signal }));
  const month = useApi(ready ? `month|${regressor}|${classifier}` : null, (signal) => api.predictions(modelParams, { signal }));

  const idleText = idleMessage(health, dateError);
  const noPoints = (data) => !data.points?.length;
  const retryAll = () => [health, models, summary, day, dayAhead, month].forEach((state) => state.retry());

  return (
    <div className="app">
      <Sidebar health={health} />
      <div className="main">
        <Header health={health} />
        <StatusBanner error={health.error} onRetry={retryAll} />

        <FilterBar>
          <DateSelector value={decisionDate} range={range} error={dateError} unavailable={Boolean(health.error)} onChange={setDecisionInput} />
          <ModelSelect id="regressor" label="Corrected-forecast model" models={models.data?.corrected_forecast} value={regressorName} failed={modelsFailed} onChange={setRegressor} />
          <ModelSelect id="classifier" label="Shortfall-risk model" models={models.data?.shortfall_risk} value={classifierName} failed={modelsFailed} onChange={setClassifier} />
          {models.error && <ErrorBox error={models.error} onRetry={models.retry} compact />}
          {comparison && (
            <p className="filter-note" role="status">
              Showing a comparison model. The approved defaults are {modelLabel(defaultModel(models.data.corrected_forecast))} and{" "}
              {modelLabel(defaultModel(models.data.shortfall_risk)).toLowerCase()}.
            </p>
          )}
        </FilterBar>

        <main className="content">
          <Section
            id="overview"
            title="Decision overview"
            description={deliveryDate ? `Delivery ${longDate(deliveryDate)}, bought on ${longDate(decisionDate)} by 09:50 UK time.` : "Select a decision date to begin."}
          >
            {/* The tiles join two responses: use the day-ahead view only when it is for the same delivery day. */}
            <AsyncBlock state={day} height={150} isEmpty={noPoints} emptyText={idleText}>
              {(data) => <SummaryCards day={data} dayAhead={dayAhead.data?.delivery_date === data.period.first ? dayAhead.data : null} />}
            </AsyncBlock>
            <Card title="What was known at the decision" subtitle="The forecast run used, the cutoff, and the runs that came too late">
              <AsyncBlock state={dayAhead} height={110} emptyText={idleText}>
                {(data) => <DecisionTimeline dayAhead={data} />}
              </AsyncBlock>
            </Card>
          </Section>

          <Section id="forecast" title="Wind generation forecast" description="Hourly official forecast available at the cutoff, the model's corrected forecast, and the actual outcome.">
            <Card
              title="Official vs corrected forecast"
              subtitle={day.data ? `${modelLabel(day.data.regressor.name)} · MW per hour, UK time` : "MW per hour, UK time"}
              table={day.data?.points?.length ? <ForecastTable points={day.data.points} /> : null}
              busy={day.loading}
              footnote="Actual generation is known only after delivery and is shown for evaluation, not as decision-time information."
            >
              <AsyncBlock state={day} height={340} isEmpty={noPoints} emptyText={idleText}>
                {(data) => <ForecastChart points={data.points} />}
              </AsyncBlock>
            </Card>
          </Section>

          <Section id="risk" title="Shortfall risk" description="Probability that actual wind generation comes in at least 1 GW below the official forecast, from the shortfall model.">
            <Card
              title="Hourly shortfall probability"
              subtitle={day.data ? `${modelLabel(day.data.classifier.name)} · hourly probability from the shortfall model` : "Hourly probability from the shortfall model"}
              table={day.data?.points?.length ? <RiskTable points={day.data.points} /> : null}
              busy={day.loading}
              footnote="Dots mark hours where a shortfall actually occurred (hindsight). The 50% line is a reference, not a trading threshold."
            >
              <AsyncBlock state={day} height={340} isEmpty={noPoints} emptyText={idleText}>
                {(data) => <RiskChart points={data.points} />}
              </AsyncBlock>
            </Card>
          </Section>

          <Section id="accuracy" title="Forecast accuracy" description="Evaluation against actual generation. These are measured results, not live predictions.">
            <Card
              title="Daily error across the decision period"
              subtitle="Mean absolute error per delivery day (hindsight). Click a day to select it."
              badge={{ text: "Hindsight" }}
              table={month.data?.points?.length ? <MonthAccuracyTable points={month.data.points} /> : null}
              busy={month.loading}
            >
              <AsyncBlock state={month} height={340} isEmpty={noPoints} emptyText={idleText}>
                {(data) => <MonthAccuracyChart points={data.points} selectedDelivery={deliveryDate} onSelectDelivery={(date) => setDecisionInput(addDays(date, -1))} />}
              </AsyncBlock>
            </Card>
            <div className="card-grid">
              <Card title="Whole decision period" subtitle="Selected correction model vs official forecast" badge={{ text: "Hindsight" }}>
                <AsyncBlock state={month} emptyText={idleText}>
                  {(data) => <PeriodScores month={data} />}
                </AsyncBlock>
              </Card>
              <Card title="Official forecast" subtitle="Decision-time WINDFOR forecast, all January decisions" badge={{ text: "Hindsight" }}>
                <AsyncBlock state={summary} emptyText={idleText}>
                  {(data) => <OfficialForecastScores summary={data} />}
                </AsyncBlock>
              </Card>
              <Card title="Model evaluation" subtitle="Offline scores on the January 2026 test set, recorded when the models were evaluated" badge={{ text: "Evaluation", tone: "accent" }} className="card--wide">
                <AsyncBlock state={models} emptyText={idleText}>
                  {(data) => <ModelEvaluation models={data} />}
                </AsyncBlock>
              </Card>
            </div>
          </Section>

          <Section id="models" title="Models" description="The approved default models and the comparison models, with how each was chosen.">
            <AsyncBlock state={models} height={220} emptyText={idleText}>
              {(data) => <ModelInfo models={data} />}
            </AsyncBlock>
          </Section>

          <Section id="method" title="Data & method">
            <Card title="Data and definitions">
              <MethodNotes health={health.data} />
            </Card>
          </Section>

          <footer className="footer">Wind Forecast Intelligence · Elexon BMRS data · purchase decisions 1–31 January 2026 · forecast accuracy only, not trading advice</footer>
        </main>
      </div>
    </div>
  );
}
