/**
 * Chart colours - one job each, the same entity always the same colour.
 * Validated with the dataviz palette checker against the white card surface (#ffffff):
 * lightness band, chroma floor, colour-blind separation (worst adjacent ΔE 9.4),
 * normal-vision separation (ΔE 20.1) and >= 3:1 contrast all pass.
 * Actual generation is the neutral reference line (ink), not a categorical hue.
 * src/styles/app.css repeats these values as --series-* for HTML legend keys.
 */
export const COLORS = {
  official: "#35639f",
  corrected: "#1f9e7c",
  risk: "#c97b2f",
  actual: "#3a4350",
  grid: "#e8eaee",
  axis: "#c9ced6",
  muted: "#7b8594",
  surface: "#ffffff",
};
