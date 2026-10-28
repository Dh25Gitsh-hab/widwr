import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ukTime } from "../lib/dates.js";
import { compactMw, mw } from "../lib/format.js";
import { AXIS_PROPS, SERIES } from "../lib/chartConfig.js";
import { COLORS } from "../lib/palette.js";
import { ChartLegend, ChartTooltip } from "./ChartParts.jsx";

const LINES = ["official", "corrected", "actual"];

function toRows(points) {
  return points.map((p) => ({ time: ukTime(p.start_time), official: p.forecast_mw, corrected: p.corrected_forecast_mw, actual: p.actual_mw }));
}

export function ForecastTable({ points }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <caption className="visually-hidden">Hourly forecasts and actual generation, UK time</caption>
        <thead>
          <tr>
            <th scope="col">Hour (UK)</th>
            <th scope="col">Official forecast</th>
            <th scope="col">Corrected forecast</th>
            <th scope="col">Actual (hindsight)</th>
            <th scope="col">Corrected error (hindsight)</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.start_time}>
              <th scope="row">{ukTime(p.start_time)}</th>
              <td>{mw(p.forecast_mw)}</td>
              <td>{mw(p.corrected_forecast_mw)}</td>
              <td>{mw(p.actual_mw)}</td>
              <td>{mw(p.corrected_error_mw)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Official vs corrected forecast vs actual generation for one delivery day (one MW axis). */
export function ForecastChart({ points }) {
  const rows = toRows(points);
  return (
    <figure className="chart" aria-label="Line chart of official forecast, corrected forecast and actual wind generation by hour">
      <ChartLegend items={LINES.map((key) => ({ ...SERIES[key], shape: "line" }))} />
      <div className="chart__plot">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={COLORS.grid} vertical={false} />
            <XAxis dataKey="time" {...AXIS_PROPS} interval={2} minTickGap={8} />
            <YAxis {...AXIS_PROPS} tickFormatter={compactMw} width={44} domain={["auto", "auto"]} label={{ value: "MW", angle: -90, position: "insideLeft", fill: COLORS.muted, fontSize: 12 }} />
            <Tooltip
              cursor={{ stroke: COLORS.axis, strokeWidth: 1 }}
              content={
                <ChartTooltip
                  heading={(datum) => `${datum.time} UK`}
                  rows={(datum) => LINES.map((key) => ({ label: SERIES[key].label, color: SERIES[key].color, shape: "line", value: mw(datum[key]) }))}
                />
              }
            />
            {LINES.map((key) => (
              <Line
                key={key}
                type="linear"
                dataKey={key}
                name={SERIES[key].label}
                stroke={SERIES[key].color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: COLORS.surface, strokeWidth: 2 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
