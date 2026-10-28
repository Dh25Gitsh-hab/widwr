/** Loading, error and empty states shared by every section. */

import { safeDetail } from "../lib/diagnostics.js";
import { TechnicalDetails } from "./TechnicalDetails.jsx";

export function Skeleton({ height = 120, label = "Loading" }) {
  return (
    <div className="skeleton" style={{ height }} role="status" aria-label={label}>
      <span className="visually-hidden">{label}…</span>
    </div>
  );
}

const ERROR_TITLES = {
  network: "Backend not reachable",
  unavailable: "Service unavailable",
  validation: "Request rejected",
  format: "Unexpected response",
};

// What a business reader needs; the backend's own reason is kept under "Technical details".
const PLAIN_MESSAGES = {
  network: "The dashboard could not reach the backend. Try again in a moment.",
  unavailable: "This part of the service is unavailable right now. Other sections may still work.",
  validation: "The request was not accepted.",
  format: "The backend sent a response the dashboard could not read.",
};

/** Diagnostics of a failed request: the reason (paths shortened, commands removed), status and URL. */
export function ErrorFacts({ error }) {
  const reason = safeDetail(error.message);
  return (
    <dl className="tech-list">
      {reason && (
        <div>
          <dt>Reason</dt>
          <dd>{reason}</dd>
        </div>
      )}
      {error.status && (
        <div>
          <dt>HTTP status</dt>
          <dd>{error.status}</dd>
        </div>
      )}
      {error.url && (
        <div>
          <dt>Request</dt>
          <dd>
            <code>{error.url}</code>
          </dd>
        </div>
      )}
    </dl>
  );
}

export function ErrorBox({ error, onRetry, compact = false }) {
  if (!error) return null;
  return (
    <div className={`error-box${compact ? " error-box--compact" : ""}`} role="alert">
      <div>
        <strong>{ERROR_TITLES[error.kind] ?? "Request failed"}</strong>
        <p>{PLAIN_MESSAGES[error.kind] ?? "The request failed. Try again in a moment."}</p>
        <TechnicalDetails>
          <ErrorFacts error={error} />
        </TechnicalDetails>
      </div>
      {onRetry && (
        <button type="button" className="button button--secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ children }) {
  return <p className="empty-state">{children}</p>;
}

/**
 * Standard wrapper for an API-backed block: skeleton on first load, error with retry,
 * empty message, otherwise the content (dimmed while a new request is in flight).
 */
export function AsyncBlock({ state, height = 120, isEmpty = () => false, emptyText = "No data for this selection.", children }) {
  if (state.error) return <ErrorBox error={state.error} onRetry={state.retry} />;
  if (!state.data) return state.loading ? <Skeleton height={height} /> : <EmptyState>{emptyText}</EmptyState>;
  if (isEmpty(state.data)) return <EmptyState>{emptyText}</EmptyState>;
  return (
    <div className={state.loading ? "is-refreshing" : undefined} aria-busy={state.loading || undefined}>
      {children(state.data)}
    </div>
  );
}
