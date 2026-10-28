import { shortDate } from "../lib/dates.js";
import { ErrorFacts } from "./StateViews.jsx";
import { TechnicalDetails } from "./TechnicalDetails.jsx";

const NAV_ITEMS = [
  { id: "overview", label: "Overview" },
  { id: "forecast", label: "Forecast" },
  { id: "risk", label: "Shortfall risk" },
  { id: "accuracy", label: "Accuracy" },
  { id: "models", label: "Models" },
  { id: "method", label: "Data & method" },
];

function BrandMark() {
  return (
    <svg className="brand__mark" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" />
      <path d="M7 12h12a3 3 0 1 0-3-3M7 17h16a3 3 0 1 1-3 3M7 22h8" />
    </svg>
  );
}

function BackendStatus({ health }) {
  const state = health.error ? "down" : health.data ? "up" : "pending";
  // A backend that answers with an error (e.g. 503 while its data cannot load) is not "unreachable".
  const down = { network: "Backend unreachable", unavailable: "Backend unavailable" }[health.error?.kind] ?? "Backend error";
  const text = { up: "Backend connected", down, pending: "Checking backend…" }[state];
  return (
    <div className="backend-status">
      <span className={`status-dot status-dot--${state}`} aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}

export function Sidebar({ health }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <BrandMark />
        <div>
          <strong>Wind Forecast Intelligence</strong>
          <span>Day-ahead purchasing desk</span>
        </div>
      </div>
      <nav aria-label="Sections">
        <ul className="nav">
          {NAV_ITEMS.map((item) => (
            <li key={item.id}>
              <a href={`#${item.id}`}>{item.label}</a>
            </li>
          ))}
        </ul>
      </nav>
      <BackendStatus health={health} />
    </aside>
  );
}

export function Header({ health }) {
  const range = health.data?.decision_dates;
  const period = range ? `${shortDate(range.first)} – ${shortDate(range.last)} ${range.last.slice(0, 4)}` : "January 2026";
  return (
    <header className="topbar">
      <div className="topbar__title">
        <h1>UK wind generation · day-ahead view</h1>
        <p>Corrected wind forecasts and shortfall risk for purchase decisions made in January 2026.</p>
      </div>
      <dl className="facts">
        <div>
          <dt>Decision period</dt>
          <dd>{period}</dd>
        </div>
        <div>
          <dt>Decision cutoff</dt>
          <dd>09:50 UK time, day before delivery</dd>
        </div>
        <div>
          <dt>Times shown in</dt>
          <dd>UK local time (GMT in January)</dd>
        </div>
      </dl>
      <nav aria-label="Sections" className="topbar__nav">
        {NAV_ITEMS.map((item) => (
          <a key={item.id} href={`#${item.id}`}>
            {item.label}
          </a>
        ))}
      </nav>
    </header>
  );
}

export function StatusBanner({ error, onRetry }) {
  if (!error) return null;
  const network = error.kind === "network";
  return (
    <div className="banner" role="alert">
      <div>
        <strong>{network ? "The dashboard cannot reach the backend." : "The backend reported a problem."}</strong>
        <p>{network ? "Forecasts and results will appear as soon as the backend is available." : "The data could not be loaded. Try again in a moment."}</p>
        <TechnicalDetails>
          <ErrorFacts error={error} />
          {network && (
            <p className="banner__hint">
              Start the backend as described in the project README (Quick start). If it runs at another address, set <code>VITE_API_BASE_URL</code> and restart the
              dashboard; the backend&apos;s <code>WFM_CORS_ORIGINS</code> setting must allow this page&apos;s address.
            </p>
          )}
        </TechnicalDetails>
      </div>
      <button type="button" className="button" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}
