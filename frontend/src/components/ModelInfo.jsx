import { longDate } from "../lib/dates.js";
import { inputSummary, modelDescription, modelLabel, roleLabel, selectionText } from "../lib/models.js";
import { TechnicalDetails } from "./TechnicalDetails.jsx";

const TASKS = {
  regression: "Regression: predicts the official forecast's error in MW",
  classification: "Classification: predicts the probability of a shortfall",
};

function Fingerprint({ model }) {
  if (!model.sha256) return "—";
  return (
    <>
      <code title={model.sha256}>{model.sha256.slice(0, 12)}…</code>
      {model.matches_holdout_artifact === true && <span className="check">✓ matches its sealed fingerprint</span>}
      {model.matches_holdout_artifact === false && <span className="check check--failed">✗ does not match its sealed fingerprint</span>}
    </>
  );
}

function ModelCard({ model, cutoff }) {
  return (
    <article className={`model-card${model.role === "default" ? " model-card--default" : ""}`}>
      <header>
        <h4>{modelLabel(model.name)}</h4>
        <span className={`badge badge--${model.role === "default" ? "accent" : "neutral"}`}>{roleLabel(model.role)}</span>
      </header>
      <p>{modelDescription(model)}</p>
      <dl className="model-card__facts">
        <div>
          <dt>Uses</dt>
          <dd>{inputSummary(model.inputs, cutoff)}</dd>
        </div>
        <div>
          <dt>Trained on deliveries</dt>
          <dd>
            {longDate(model.trained_from)} – {longDate(model.trained_to)}
          </dd>
        </div>
        <div>
          <dt>How it was chosen</dt>
          <dd>{selectionText(model)}</dd>
        </div>
      </dl>
      <TechnicalDetails>
        <dl className="tech-list">
          <div>
            <dt>Model ID</dt>
            <dd>
              <code>{model.name}</code>
            </dd>
          </div>
          <div>
            <dt>Task</dt>
            <dd>{TASKS[model.task] ?? model.task}</dd>
          </div>
          <div>
            <dt>Specification</dt>
            <dd>{model.description}</dd>
          </div>
          <div>
            <dt>Input columns</dt>
            <dd>
              {model.inputs.map((input) => (
                <code key={input}>{input}</code>
              ))}
            </dd>
          </div>
          <div>
            <dt>Saved model file</dt>
            <dd>
              <code>{model.artifact ?? "—"}</code>
            </dd>
          </div>
          <div>
            <dt>SHA-256</dt>
            <dd>
              <Fingerprint model={model} />
            </dd>
          </div>
        </dl>
      </TechnicalDetails>
    </article>
  );
}

/** The served models (GET /api/ml/models) in plain language; implementation detail under Technical details. */
export function ModelInfo({ models }) {
  return (
    <>
      <div className="model-grid">
        <div>
          <h3 className="group-title">Corrected forecast</h3>
          {models.corrected_forecast.map((model) => (
            <ModelCard key={model.name} model={model} cutoff={models.decision_cutoff_uk} />
          ))}
        </div>
        <div>
          <h3 className="group-title">Shortfall risk</h3>
          {models.shortfall_risk.map((model) => (
            <ModelCard key={model.name} model={model} cutoff={models.decision_cutoff_uk} />
          ))}
        </div>
      </div>
      <div className="model-note">
        <h3 className="group-title">How the defaults were chosen and what the scores mean</h3>
        <p className="note">{models.status}</p>
      </div>
    </>
  );
}
