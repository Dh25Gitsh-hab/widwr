import { COLORS } from "./palette.js";

/** Series identity: the same entity keeps the same colour and name in every chart. */
export const SERIES = {
  official: { label: "Official forecast (decision time)", color: COLORS.official },
  corrected: { label: "Corrected forecast", color: COLORS.corrected },
  actual: { label: "Actual generation (hindsight)", color: COLORS.actual },
  risk: { label: "Shortfall probability", color: COLORS.risk },
};

/** Recessive axes: hairline rule, muted tick text, no tick marks. */
export const AXIS_PROPS = {
  stroke: COLORS.axis,
  tick: { fill: COLORS.muted, fontSize: 12 },
  tickLine: false,
};
