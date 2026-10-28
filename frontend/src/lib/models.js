/** Display names for the model identifiers the backend returns (unknown names are shown as-is). */
const LABELS = {
  linear_by_level: "Linear correction",
  logistic_by_level: "Logistic regression",
  random_forest: "Random Forest",
};

export function modelLabel(name) {
  return LABELS[name] ?? name;
}

export function roleLabel(role) {
  return role === "default" ? "Approved default" : role === "comparison" ? "Comparison" : role;
}

/** Name of the model with role "default" in a /api/ml/models component list. */
export function defaultModel(list) {
  return list?.find((model) => model.role === "default")?.name ?? null;
}

// Plain-language descriptions for a business reader; the API's own description stays under Technical details.
const DESCRIPTIONS = {
  "regression/linear_by_level":
    "Adjusts the official forecast by an amount that depends only on how large the forecast is: a straight-line correction learned from past forecast errors.",
  "regression/random_forest": "Adjusts the official forecast using many decision trees that combine all the decision-time inputs below.",
  "classification/logistic_by_level": "Estimates the chance of a shortfall from the size of the official forecast alone.",
  "classification/random_forest": "Estimates the chance of a shortfall using many decision trees that combine all the decision-time inputs below.",
};

export function modelDescription(model) {
  return DESCRIPTIONS[`${model.task}/${model.name}`] ?? model.description;
}

// Meaning of each model input (backend/app/services/feature_builder.py); all are known at the decision cutoff.
const INPUT_LABELS = {
  forecast_mw: "official forecast size",
  lead_hours: "forecast lead time",
  hour_uk: "hour of day (UK)",
  forecast_revision_mw: "change since the previous forecast run",
  forecast_spread_mw: "spread between recent forecast runs",
  forecast_ramp_mw: "hour-to-hour change in the forecast",
  forecast_day_mean_mw: "average forecast for the day",
  latest_actual_mw: "latest actual generation known at the cutoff",
  recent_error_mw: "recent forecast error known at the cutoff",
};

/** "Official forecast size", or "9 inputs known at the 09:50 UK cutoff: …". */
export function inputSummary(inputs, cutoff) {
  const labels = inputs.map((name) => INPUT_LABELS[name] ?? name);
  if (labels.length === 1) return labels[0].charAt(0).toUpperCase() + labels[0].slice(1);
  return `${labels.length} inputs known at the ${cutoff ?? "decision"} UK cutoff: ${labels.join(", ")}`;
}

/** How a served model came to be a default or a comparison model, from the API's role and selection flag. */
export function selectionText(model) {
  const byRule = model.selected_by_pre_january_rule;
  if (model.role === "default") {
    return byRule
      ? "Approved default; also the choice of the rule fixed before January."
      : "Approved default, chosen after the January test and an offline February check (not by the rule fixed before January).";
  }
  return byRule ? "Chosen by the rule fixed before January (lowest December validation error); kept for comparison." : "Kept for comparison.";
}
