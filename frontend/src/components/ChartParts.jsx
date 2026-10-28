/**
 * Chart chrome shared by every chart: a legend whose keys mirror the marks, and a tooltip
 * that leads with the value. Labels use text colours; only the small key beside them carries
 * the series colour (series definitions live in src/lib/chartConfig.js).
 */

function Key({ shape, color }) {
  if (shape === "bar") return <span className="key key--bar" style={{ background: color }} aria-hidden="true" />;
  if (shape === "dot") return <span className="key key--dot" style={{ background: color }} aria-hidden="true" />;
  return <span className="key key--line" style={{ background: color }} aria-hidden="true" />;
}

export function ChartLegend({ items }) {
  return (
    <ul className="chart-legend">
      {items.map((item) => (
        <li key={item.label}>
          <Key shape={item.shape} color={item.color} />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

/** Recharts custom tooltip: heading, then one row per series with the value first. */
export function ChartTooltip({ active, payload, label, heading, rows }) {
  if (!active || !payload?.length) return null;
  const datum = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__heading">{heading ? heading(datum, label) : label}</div>
      {rows(datum).map((row) => (
        <div key={row.label} className="chart-tooltip__row">
          <Key shape={row.shape} color={row.color} />
          <strong>{row.value}</strong>
          <span>{row.label}</span>
        </div>
      ))}
    </div>
  );
}
