import { addDays, longDate } from "../lib/dates.js";
import { modelLabel, roleLabel } from "../lib/models.js";

/**
 * The one filter row: everything below it re-renders for the chosen decision date and models.
 * The date range comes from /api/health (decision_dates), not from a hard-coded constant.
 */
export function DateSelector({ value, range, error, unavailable = false, onChange }) {
  const disabled = !range;
  const canStep = (days) => Boolean(range && value && !error && addDays(value, days) >= range.first && addDays(value, days) <= range.last);
  return (
    <div className="field">
      <label htmlFor="decision-date">Decision date (D-1)</label>
      <div className="date-stepper">
        <button type="button" className="icon-button" aria-label="Previous decision day" disabled={!canStep(-1)} onClick={() => onChange(addDays(value, -1))}>
          ‹
        </button>
        <input
          id="decision-date"
          type="date"
          value={value ?? ""}
          min={range?.first}
          max={range?.last}
          disabled={disabled}
          aria-invalid={Boolean(error)}
          aria-describedby="decision-date-help"
          onChange={(event) => onChange(event.target.value)}
        />
        <button type="button" className="icon-button" aria-label="Next decision day" disabled={!canStep(1)} onClick={() => onChange(addDays(value, 1))}>
          ›
        </button>
      </div>
      <p id="decision-date-help" className={error ? "field__error" : "field__help"}>
        {error ??
          (value && range
            ? `Delivery day ${longDate(addDays(value, 1))}`
            : unavailable
              ? "Decision window unavailable until the backend health check succeeds."
              : "Loading the decision window…")}
      </p>
    </div>
  );
}

/** Options come only from /api/ml/models. `failed`: the list could not be loaded, so none is offered. */
export function ModelSelect({ id, label, models, value, failed = false, onChange }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value ?? ""} disabled={!models?.length} onChange={(event) => onChange(event.target.value)}>
        {!models?.length && <option value="">{failed ? "Models unavailable" : "Loading models…"}</option>}
        {models?.map((model) => (
          <option key={model.name} value={model.name}>
            {modelLabel(model.name)} · {roleLabel(model.role)}
          </option>
        ))}
      </select>
    </div>
  );
}

export function FilterBar({ children }) {
  return (
    <div className="filter-bar" role="region" aria-label="Filters">
      {children}
    </div>
  );
}
