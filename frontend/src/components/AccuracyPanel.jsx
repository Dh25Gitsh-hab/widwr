import { isNumber, mw, percent, score, signedMw } from "../lib/format.js";
import { modelLabel, roleLabel } from "../lib/models.js";
import { reductionPct } from "../lib/stats.js";

function MetricRows({ rows }) {
  return (
    <dl className="metric-list">
      {rows.map(([label, value, hint]) => (
        <div key={label}>
          <dt>
            {label}
            {hint && <span className="metric-list__hint">{hint}</span>}
          </dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Backend-computed scores for the whole decision period, for the models currently selected. */
export function PeriodScores({ month }) {
  const { official, corrected } = month.scores ?? {};
  if (!official || !corrected) return <p className="empty-state">No actual generation in this period.</p>;
  // The backend scores only hours with an actual; count the same hours and name any that were left out.
  const scoredHours = month.points.filter((p) => isNumber(p.actual_mw)).length;
  const unscoredHours = month.points.length - scoredHours;
  return (
    <div className="compare">
      <div className="compare__head">
        <span />
        <span>Official</span>
        <span>{modelLabel(month.regressor.name)}</span>
      </div>
      {[
        ["MAE", official.mae, corrected.mae],
        ["RMSE", official.rmse, corrected.rmse],
        ["Mean bias", official.bias, corrected.bias],
      ].map(([label, a, b]) => (
        <div key={label} className="compare__row">
          <span>{label}</span>
          <span>{label === "Mean bias" ? signedMw(a) : mw(a)}</span>
          <span>{label === "Mean bias" ? signedMw(b) : mw(b)}</span>
        </div>
      ))}
      <p className="compare__summary">
        MAE {percent(reductionPct(official.mae, corrected.mae))} lower than the official forecast over {scoredHours} hours
        {unscoredHours > 0 && ` (${unscoredHours} ${unscoredHours === 1 ? "hour" : "hours"} without actual generation excluded)`}
      </p>
    </div>
  );
}

/** /api/analytics/summary: the official decision-time forecast over the window. */
export function OfficialForecastScores({ summary }) {
  const m = summary.metrics;
  return (
    <MetricRows
      rows={[
        ["MAE", mw(m.mae)],
        ["RMSE", mw(m.rmse)],
        ["Mean bias", signedMw(m.bias), "positive = forecast too high"],
        ["MAE as share of mean output", percent(m.nmae_pct)],
        ["Hours over-forecast", percent(m.over_forecast_pct)],
        ["Hours scored", `${m.intervals_scored} of ${m.intervals_total}`],
      ]}
    />
  );
}

/** How to read the evaluation scores. The shortfall rate is the API's own January figure. */
function MetricKey({ observedRate }) {
  return (
    <p className="note metric-key">
      How to read: for MAE, RMSE and the size of the bias, lower is better. ROC-AUC: 0.5 is no better than chance, 1 is a perfect ranking of shortfall hours.
      PR-AUC: higher is better; its chance level is the share of shortfall hours
      {isNumber(observedRate) ? ` (${percent(observedRate * 100, 0)} of these January hours)` : ""}. Brier score: the average squared gap between the
      probability and what happened; lower is better. Precision, recall and F1 count hours at a 50% threshold.
    </p>
  );
}

/** /api/ml/models january_test: offline evaluation, not live predictions. */
export function ModelEvaluation({ models }) {
  const observedRate = models.shortfall_risk.find((m) => m.role === "default")?.january_test?.observed_rate;
  return (
    <>
      <div className="table-wrap">
        <table className="data-table data-table--compact">
          <caption>Corrected forecast</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">MAE</th>
              <th scope="col">RMSE</th>
              <th scope="col">Bias</th>
            </tr>
          </thead>
          <tbody>
            {models.corrected_forecast.map((m) => (
              <tr key={m.name}>
                <th scope="row">
                  {modelLabel(m.name)} <span className="muted">· {roleLabel(m.role)}</span>
                </th>
                <td>{mw(m.january_test?.mae)}</td>
                <td>{mw(m.january_test?.rmse)}</td>
                <td>{signedMw(m.january_test?.bias)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <table className="data-table data-table--compact">
          <caption>Shortfall risk</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">ROC-AUC</th>
              <th scope="col">PR-AUC</th>
              <th scope="col">Brier</th>
              <th scope="col">Precision</th>
              <th scope="col">Recall</th>
              <th scope="col">F1</th>
            </tr>
          </thead>
          <tbody>
            {models.shortfall_risk.map((m) => (
              <tr key={m.name}>
                <th scope="row">
                  {modelLabel(m.name)} <span className="muted">· {roleLabel(m.role)}</span>
                </th>
                <td>{score(m.january_test?.roc_auc)}</td>
                <td>{score(m.january_test?.pr_auc)}</td>
                <td>{score(m.january_test?.brier)}</td>
                <td>{score(m.january_test?.precision)}</td>
                <td>{score(m.january_test?.recall)}</td>
                <td>{score(m.january_test?.f1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <MetricKey observedRate={observedRate} />
    </>
  );
}
