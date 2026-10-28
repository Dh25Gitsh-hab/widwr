const DASH = "—";

const integer = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

export function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** "11,678 MW" */
export function mw(value) {
  return isNumber(value) ? `${integer.format(value)} MW` : DASH;
}

/** "+1,094 MW" / "−288 MW" (true minus sign) */
export function signedMw(value) {
  if (!isNumber(value)) return DASH;
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${integer.format(Math.abs(value))} MW`;
}

/** Probability 0-1 as "78%". */
export function probability(value) {
  return isNumber(value) ? `${Math.round(value * 100)}%` : DASH;
}

/** A percentage already in 0-100 units, e.g. 40.9 -> "40.9%". */
export function percent(value, digits = 1) {
  return isNumber(value) ? `${value.toFixed(digits)}%` : DASH;
}

/** A plain score such as ROC-AUC 0.755 -> "0.755". */
export function score(value, digits = 3) {
  return isNumber(value) ? value.toFixed(digits) : DASH;
}

/** Axis ticks: 12000 -> "12k". */
export function compactMw(value) {
  if (!isNumber(value)) return "";
  return Math.abs(value) >= 1000 ? `${integer.format(value / 1000)}k` : integer.format(value);
}
