import { longDate, ukDateTime, ukTime, ukZoneName } from "../lib/dates.js";
import { mw, percent, probability, signedMw } from "../lib/format.js";
import { modelLabel, roleLabel } from "../lib/models.js";
import { reductionPct, summarizeDay } from "../lib/stats.js";
import { COLORS } from "../lib/palette.js";

function StatTile({ label, color, value, children, note }) {
  return (
    <article className="stat">
      <p className="stat__label">
        {color && <span className="key key--line" style={{ background: color }} aria-hidden="true" />}
        {label}
      </p>
      <p className="stat__value">{value}</p>
      <div className="stat__detail">{children}</div>
      {note && <p className="stat__note">{note}</p>}
    </article>
  );
}

function Change({ value }) {
  if (value === null) return null;
  const better = value > 0;
  return (
    <span className={`change ${better ? "change--better" : "change--worse"}`}>
      <span aria-hidden="true">{better ? "▼" : "▲"}</span> {percent(Math.abs(value))} {better ? "lower error" : "higher error"}
    </span>
  );
}

/** The four headline tiles for the selected delivery day. */
export function SummaryCards({ day, dayAhead }) {
  const summary = summarizeDay(day.points);
  if (!summary) return null;
  const { official, corrected } = day.scores ?? {};
  const expected = dayAhead?.expected;
  return (
    <div className="stat-grid">
      <StatTile label="Official forecast · daily mean" color={COLORS.official} value={mw(summary.meanOfficial)} note="Elexon WINDFOR run used at the decision cutoff">
        {expected && (
          <p>
            Low {mw(expected.min_mw)} at {ukTime(expected.min_time)} · high {mw(expected.max_mw)} at {ukTime(expected.max_time)}
          </p>
        )}
      </StatTile>

      <StatTile label="Corrected forecast · daily mean" color={COLORS.corrected} value={mw(summary.meanCorrected)} note={`${modelLabel(day.regressor.name)} · ${roleLabel(day.regressor.role)}`}>
        <p>{signedMw(summary.meanCorrected - summary.meanOfficial)} versus the official forecast</p>
      </StatTile>

      <StatTile
        label="Forecast error · MAE (hindsight)"
        value={corrected ? mw(corrected.mae) : "Not available"}
        note="Measured against actual generation, known only after delivery"
      >
        {official && corrected ? (
          <>
            <p>Official forecast {mw(official.mae)} · corrected {mw(corrected.mae)}</p>
            <Change value={reductionPct(official.mae, corrected.mae)} />
          </>
        ) : (
          <p>No actual generation for this day.</p>
        )}
      </StatTile>

      <StatTile label="Shortfall risk · peak hour" color={COLORS.risk} value={probability(summary.peakRisk)} note={`${modelLabel(day.classifier.name)} · ${roleLabel(day.classifier.role)} · actual ≥ 1 GW below forecast`}>
        <p>
          At {ukTime(summary.peakRiskTime)} · {summary.highRiskHours} of {summary.hours} hours at 50% or more · mean {probability(summary.meanRisk)}
        </p>
        {summary.hoursWithOutcome > 0 && (
          <p className="muted">
            Hindsight: a shortfall occurred in {summary.observedShortfalls} of {summary.hoursWithOutcome} hours
          </p>
        )}
      </StatTile>
    </div>
  );
}

/** What was known at the decision: the cutoff, the run used, and the runs that came too late. */
export function DecisionTimeline({ dayAhead }) {
  const before = dayAhead.runs.filter((run) => run.available_at_decision).length;
  const after = dayAhead.runs.length - before;
  const zone = ukZoneName(dayAhead.decision_cutoff);
  const steps = [
    { title: "Forecast used", text: dayAhead.forecast_run_used ? `Published ${ukDateTime(dayAhead.forecast_run_used)} ${zone}` : "No forecast published before the cutoff", detail: dayAhead.minutes_before_cutoff !== null ? `${Math.round(dayAhead.minutes_before_cutoff)} minutes before the cutoff · ${before} runs available by then` : null },
    { title: "Decision cutoff", text: `${longDate(dayAhead.decision_date)}, ${ukTime(dayAhead.decision_cutoff)} ${zone}`, detail: "Day-ahead gate closure (Nord Pool GB)" },
    { title: "Later runs", text: `${after} runs published after the cutoff`, detail: "Not available for the purchase; shown for context only" },
    { title: "Delivery day", text: longDate(dayAhead.delivery_date), detail: `${dayAhead.intervals.length / 2} hours, scored against actual generation afterwards` },
  ];
  return (
    <ol className="timeline">
      {steps.map((step, index) => (
        <li key={step.title} className={index === 1 ? "timeline__step timeline__step--cutoff" : "timeline__step"}>
          <span className="timeline__marker" aria-hidden="true" />
          <p className="timeline__title">{step.title}</p>
          <p className="timeline__text">{step.text}</p>
          {step.detail && <p className="timeline__detail">{step.detail}</p>}
        </li>
      ))}
    </ol>
  );
}
