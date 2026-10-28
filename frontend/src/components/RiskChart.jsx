import { Bar, CartesianGrid, ComposedChart, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { ukTime } from "../lib/dates.js";
import { mw, probability } from "../lib/format.js";
import { AXIS_PROPS, SERIES } from "../lib/chartConfig.js";
import { COLORS } from "../lib/palette.js";
import { ChartLegend, ChartTooltip } from "./ChartParts.jsx";

const OCCURRED_Y = 104; // markers sit just above the 100% line

function toRows(points) {
  return points.map((p) => ({
    time: ukTime(p.start_time),
    probability: p.shortfall_probability * 100,
    raw: p.shortfall_probability,
    occurred: p.shortfall_occurred,
    marker: p.shortfall_occurred ? OCCURRED_Y : null,
    forecast: p.forecast_mw,
  }));
}

function OccurredMarker({ cx, cy, payload }) {
  // Recharts positions empty values at the baseline; draw only hours where a shortfall occurred.
  if (payload?.occurred !== true || !Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  return <circle cx={cx} cy={cy} r={4} fill={COLORS.actual} stroke={COLORS.surface} strokeWidth={2} />;
}

function outcomeText(occurred) {
  return occurred === null || occurred === undefined ? "No actual" : occurred ? "Yes" : "No";
}

export function RiskTable({ points }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <caption className="visually-hidden">Hourly shortfall probability and outcome, UK time</caption>
        <thead>
          <tr>
            <th scope="col">Hour (UK)</th>
            <th scope="col">Shortfall probability</th>
            <th scope="col">Official forecast</th>
            <th scope="col">Shortfall occurred (hindsight)</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.start_time}>
              <th scope="row">{ukTime(p.start_time)}</th>
              <td>{probability(p.shortfall_probability)}</td>
              <td>{mw(p.forecast_mw)}</td>
              <td>{outcomeText(p.shortfall_occurred)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Hourly probability that actual wind comes in at least 1 GW below the official forecast. */
export function RiskChart({ points }) {
  const rows = toRows(points);
  return (
    <figure className="chart" aria-label="Bar chart of the hourly shortfall probability returned by the model">
      <ChartLegend
        items={[
          { ...SERIES.risk, label: "Shortfall probability (model)", shape: "bar" },
          { label: "Shortfall occurred (hindsight)", color: COLORS.actual, shape: "dot" },
        ]}
      />
      <div className="chart__plot">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 12, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={COLORS.grid} vertical={false} />
            <XAxis dataKey="time" {...AXIS_PROPS} interval={2} minTickGap={8} />
            <YAxis {...AXIS_PROPS} domain={[0, 108]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(v) => `${v}%`} width={44} />
            <ReferenceLine y={50} stroke={COLORS.muted} strokeDasharray="4 4" label={{ value: "50%", position: "insideTopRight", fill: COLORS.muted, fontSize: 11 }} />
            <Tooltip
              cursor={{ fill: "rgba(58, 67, 80, 0.06)" }}
              content={
                <ChartTooltip
                  heading={(datum) => `${datum.time} UK`}
                  rows={(datum) => [
                    { label: "Shortfall probability", color: COLORS.risk, shape: "bar", value: probability(datum.raw) },
                    { label: "Shortfall occurred (hindsight)", color: COLORS.actual, shape: "dot", value: outcomeText(datum.occurred) },
                  ]}
                />
              }
            />
            <Bar dataKey="probability" fill={COLORS.risk} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            <Scatter dataKey="marker" shape={<OccurredMarker />} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
