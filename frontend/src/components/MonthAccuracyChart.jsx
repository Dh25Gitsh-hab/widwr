import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { shortDate } from "../lib/dates.js";
import { compactMw, mw, percent } from "../lib/format.js";
import { AXIS_PROPS, SERIES } from "../lib/chartConfig.js";
import { COLORS } from "../lib/palette.js";
import { dailyErrors, reductionPct } from "../lib/stats.js";
import { ChartLegend, ChartTooltip } from "./ChartParts.jsx";

const LINES = [
  { key: "official", label: "Official forecast MAE", color: SERIES.official.color },
  { key: "corrected", label: "Corrected forecast MAE", color: SERIES.corrected.color },
];

function toRows(points) {
  return dailyErrors(points).map((day) => ({ ...day, label: shortDate(day.date) }));
}

export function MonthAccuracyTable({ points }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <caption className="visually-hidden">Daily mean absolute error by delivery day</caption>
        <thead>
          <tr>
            <th scope="col">Delivery day</th>
            <th scope="col">Official MAE</th>
            <th scope="col">Corrected MAE</th>
            <th scope="col">Change</th>
          </tr>
        </thead>
        <tbody>
          {toRows(points).map((day) => {
            const change = reductionPct(day.official, day.corrected);
            return (
              <tr key={day.date}>
                <th scope="row">{day.label}</th>
                <td>{mw(day.official)}</td>
                <td>{mw(day.corrected)}</td>
                <td>{change === null ? "—" : `${change > 0 ? "−" : "+"}${percent(Math.abs(change))}`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Daily MAE across the decision period (hindsight). Clicking a day selects it. */
export function MonthAccuracyChart({ points, selectedDelivery, onSelectDelivery }) {
  const rows = toRows(points);
  const selectedLabel = selectedDelivery ? shortDate(selectedDelivery) : null;
  return (
    <figure className="chart" aria-label="Line chart of daily mean absolute error for the official and corrected forecasts">
      <ChartLegend items={LINES.map((line) => ({ ...line, shape: "line" }))} />
      <div className="chart__plot">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={rows}
            margin={{ top: 8, right: 12, bottom: 4, left: 4 }}
            onClick={(state) => {
              const date = state?.activePayload?.[0]?.payload?.date;
              if (date) onSelectDelivery(date);
            }}
          >
            <CartesianGrid stroke={COLORS.grid} vertical={false} />
            <XAxis dataKey="label" {...AXIS_PROPS} interval={4} minTickGap={8} />
            <YAxis {...AXIS_PROPS} tickFormatter={compactMw} width={44} label={{ value: "MW", angle: -90, position: "insideLeft", fill: COLORS.muted, fontSize: 12 }} />
            {selectedLabel && <ReferenceLine x={selectedLabel} stroke={COLORS.actual} strokeWidth={1} label={{ value: "Selected", position: "insideTopLeft", fill: COLORS.muted, fontSize: 11 }} />}
            <Tooltip
              cursor={{ stroke: COLORS.axis, strokeWidth: 1 }}
              content={
                <ChartTooltip
                  heading={(datum) => `Delivery ${datum.label} · ${datum.hours} h`}
                  rows={(datum) => LINES.map((line) => ({ label: line.label, color: line.color, shape: "line", value: mw(datum[line.key]) }))}
                />
              }
            />
            {LINES.map((line) => (
              <Line key={line.key} type="linear" dataKey={line.key} stroke={line.color} strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: COLORS.surface, strokeWidth: 2 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
