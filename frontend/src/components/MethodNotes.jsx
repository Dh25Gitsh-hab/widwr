import { API_BASE_URL } from "../api/client.js";
import { addDays, longDate, ukFullDateTime } from "../lib/dates.js";
import { safeDetail } from "../lib/diagnostics.js";
import { TechnicalDetails } from "./TechnicalDetails.jsx";

const ENDPOINTS = ["GET /api/health", "GET /api/ml/models", "GET /api/ml/predictions", "GET /api/forecasts/day-ahead", "GET /api/analytics/summary"];

const METRICS = [
  ["MAE (mean absolute error)", "The average size of the forecast error in MW, whatever its direction. Lower is better."],
  ["RMSE (root mean squared error)", "Like MAE, but large errors count more. Lower is better."],
  ["Mean bias", "The average signed error. Positive means the forecast was too high on average, so less wind arrived than planned."],
  ["ROC-AUC", "How well the shortfall probabilities rank shortfall hours above other hours. 0.5 is no better than chance; 1 is perfect."],
  ["PR-AUC", "How reliable the shortfall warnings are across all probability thresholds. Higher is better; chance level is the share of shortfall hours."],
  ["Brier score", "The average squared gap between the shortfall probability and what happened (0 or 1). Lower is better."],
];

function snapshotText(health) {
  if (health.data_source === "snapshot") {
    return health.snapshot_fetched_at
      ? `Saved Elexon snapshot, downloaded ${ukFullDateTime(health.snapshot_fetched_at)} UK time`
      : "Saved Elexon snapshot";
  }
  if (health.data_source === "elexon-live") return `Downloaded from Elexon when the service started, ${ukFullDateTime(health.loaded_at)} UK time`;
  return health.data_source;
}

function dayCount(range) {
  let days = 0;
  for (let day = range.first; day <= range.last; day = addDays(day, 1)) days += 1;
  return days;
}

function CleaningTable({ cleaning }) {
  return (
    <div className="table-wrap">
      <table className="data-table data-table--compact">
        <caption>Data cleaning (rows received from Elexon and why any were removed)</caption>
        <thead>
          <tr>
            <th scope="col">Dataset</th>
            <th scope="col">Received</th>
            <th scope="col">Invalid</th>
            <th scope="col">Other fuel</th>
            <th scope="col">Duplicate</th>
            <th scope="col">Published after the hour it forecasts</th>
            <th scope="col">Kept</th>
          </tr>
        </thead>
        <tbody>
          {cleaning.map((report) => (
            <tr key={report.dataset}>
              <th scope="row">{report.dataset}</th>
              {[report.raw_rows, report.invalid_rows, report.other_fuel_rows, report.duplicate_rows, report.past_target_rows, report.clean_rows].map((value, i) => (
                <td key={i}>{value?.toLocaleString("en-GB") ?? "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Data provenance, definitions, metrics and limitations; the provenance figures come from GET /api/health. */
export function MethodNotes({ health }) {
  const decisions = health?.decision_dates;
  return (
    <div className="method">
      <dl className="metric-list">
        <div>
          <dt>Source</dt>
          <dd>Elexon BMRS: WINDFOR (wind forecasts) and FUELHH (actual wind generation), national level</dd>
        </div>
        {health && (
          <>
            <div>
              <dt>Data snapshot</dt>
              <dd>{snapshotText(health)}</dd>
            </div>
            <div>
              <dt>Data loaded</dt>
              <dd>
                {health.forecast_runs?.toLocaleString("en-GB")} forecast runs · {health.actual_intervals?.toLocaleString("en-GB")} actual half-hours
              </dd>
            </div>
          </>
        )}
      </dl>

      <div className="method__notes">
        <ul className="method__list">
          <li>
            <strong>Decision rule.</strong> For each delivery hour, the forecast is the latest WINDFOR value published at or before 09:50 UK time on the day before
            delivery. Every model input was available by then.
          </li>
          <li>
            <strong>Corrected forecast.</strong> The official forecast minus the model&apos;s predicted error. Error is forecast minus actual, in MW; positive means the forecast was too high.
          </li>
          <li>
            <strong>Shortfall risk.</strong> The model&apos;s probability that actual generation comes in at least 1 GW below the official forecast.
          </li>
          <li>
            <strong>Hindsight.</strong> Actual generation, errors and &ldquo;shortfall occurred&rdquo; are known only after delivery. They are shown to evaluate the forecasts, not as information available at the decision.
          </li>
          <li>
            <strong>Scope.</strong> Purchase decisions made 1–31 January 2026. The figures describe forecast accuracy only; they are not trading advice or evidence of profit.
          </li>
        </ul>

        <h4 className="group-title">Evaluation metrics</h4>
        <dl className="definitions">
          {METRICS.map(([term, meaning]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{meaning}</dd>
            </div>
          ))}
        </dl>

        <h4 className="group-title">Limitations</h4>
        <ul className="method__list">
          <li>
            {decisions
              ? `The results come from one month of decisions (${longDate(decisions.first)} – ${longDate(decisions.last)}, ${dayCount(decisions)} days). Accuracy can differ in other months and seasons.`
              : "The results come from one month of decisions (January 2026). Accuracy can differ in other months and seasons."}
          </li>
          <li>
            The approved default models were chosen partly using the January test results and an offline February check, so the January scores favour them. They are
            not proof of long-term superiority.
          </li>
          <li>Forecast errors vary strongly from day to day. On some days the corrected forecast moves too far and ends up below actual generation.</li>
          <li>The shortfall model ranks risky hours better than chance but is far from perfect: see its ROC-AUC under Forecast accuracy.</li>
        </ul>
      </div>

      <div className="method__technical">
        <TechnicalDetails>
          <dl className="tech-list">
            <div>
              <dt>API address</dt>
              <dd>
                <code>{API_BASE_URL}</code>
              </dd>
            </div>
            {health && (
              <>
                <div>
                  <dt>Data mode</dt>
                  <dd>
                    <code>{health.data_source}</code>
                  </dd>
                </div>
                <div>
                  <dt>Service started</dt>
                  <dd>{ukFullDateTime(health.loaded_at)} UK time</dd>
                </div>
                <div>
                  <dt>Delivery days served</dt>
                  <dd>
                    {health.delivery_dates.first} to {health.delivery_dates.last}
                  </dd>
                </div>
                <div>
                  <dt>Forecast values</dt>
                  <dd>{health.forecast_values?.toLocaleString("en-GB")}</dd>
                </div>
                <div>
                  <dt>Model service</dt>
                  <dd>{safeDetail(health.ml_models)}</dd>
                </div>
              </>
            )}
            <div>
              <dt>API endpoints used</dt>
              <dd>
                {ENDPOINTS.map((endpoint) => (
                  <code key={endpoint}>{endpoint}</code>
                ))}
              </dd>
            </div>
          </dl>
          {health?.cleaning?.length > 0 && <CleaningTable cleaning={health.cleaning} />}
        </TechnicalDetails>
      </div>
    </div>
  );
}
