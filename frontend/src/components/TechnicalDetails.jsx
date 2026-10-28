/** A collapsed "Technical details" block for implementation detail that business readers do not need. */
export function TechnicalDetails({ children, label = "Technical details" }) {
  return (
    <details className="tech-details">
      <summary>{label}</summary>
      <div className="tech-details__body">{children}</div>
    </details>
  );
}
