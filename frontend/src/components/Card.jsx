import { useId, useState } from "react";

/**
 * A titled panel. With `table`, it offers a Chart / Table toggle so no value is hover-only.
 * `busy` dims the table and marks it aria-busy while its data reloads, as AsyncBlock does for charts.
 */
export function Card({ title, subtitle, badge, table, busy = false, footnote, children, className = "" }) {
  const [view, setView] = useState("chart");
  const titleId = useId();
  return (
    <section className={`card ${className}`} aria-labelledby={titleId}>
      <header className="card__header">
        <div>
          <h3 id={titleId} className="card__title">
            {title}
            {badge && (
              <>
                {/* The space keeps title and badge separate words for screen readers; the flex gap sets the visual spacing. */}{" "}
                <span className={`badge badge--${badge.tone ?? "neutral"}`}>{badge.text}</span>
              </>
            )}
          </h3>
          {subtitle && <p className="card__subtitle">{subtitle}</p>}
        </div>
        {table && (
          <div className="segmented" role="group" aria-label={`${title}: view`}>
            {["chart", "table"].map((option) => (
              <button key={option} type="button" aria-pressed={view === option} onClick={() => setView(option)}>
                {option === "chart" ? "Chart" : "Table"}
              </button>
            ))}
          </div>
        )}
      </header>
      <div className="card__body">
        {table && view === "table" ? (
          <div className={busy ? "is-refreshing" : undefined} aria-busy={busy || undefined}>
            {table}
          </div>
        ) : (
          children
        )}
      </div>
      {footnote && <p className="card__footnote">{footnote}</p>}
    </section>
  );
}

export function Section({ id, title, description, children }) {
  return (
    <section id={id} className="section" aria-labelledby={`${id}-title`}>
      <div className="section__heading">
        <h2 id={`${id}-title`}>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {children}
    </section>
  );
}
