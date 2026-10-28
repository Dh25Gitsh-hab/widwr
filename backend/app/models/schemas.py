"""
API response schemas - the contract between the backend and the React app.

WHY
    * The frontend can rely on exact field names and types; a backend change
      that breaks the contract fails here, not silently in the browser.
    * FastAPI turns these models into the interactive OpenAPI docs at /docs,
      so the API documents itself.
    * Field names are explicit about units (``_mw``) and time zones (all
      datetimes are UTC and serialised with a trailing "Z"; the UI converts
      to UK local time for display).

HOW
    The services work with pandas DataFrames. The small ``from_frame`` /
    ``from_dict`` builders below convert them into these models, turning
    pandas' NaN/NaT into JSON null on the way.
"""

from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.services.forecast_evaluator import SelectionPolicy


def _datetime_or_none(value) -> datetime | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).to_pydatetime()


def _float_or_none(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


# ---- Shared building blocks -------------------------------------------------


class DateRange(BaseModel):
    first: date
    last: date


class PolicyInfo(BaseModel):
    kind: Literal["day_ahead", "fixed_horizon"]
    gate_closure_uk: str | None = Field(None, description="Gate closure, UK local time on D-1 (day_ahead only)")
    horizon_hours: float | None = Field(None, description="Minimum lead time in hours (fixed_horizon only)")
    description: str

    @classmethod
    def from_policy(cls, policy: "SelectionPolicy") -> "PolicyInfo":
        day_ahead = policy.kind == "day_ahead"
        return cls(
            kind=policy.kind,
            gate_closure_uk=f"{policy.gate_closure:%H:%M}" if day_ahead else None,
            horizon_hours=None if day_ahead else policy.horizon_hours,
            description=policy.describe(),
        )


class AccuracyMetrics(BaseModel):
    """Headline accuracy of the decision-time forecast. Error = forecast - actual (MW)."""

    intervals_total: int = Field(description="Target hours in the period")
    intervals_scored: int = Field(description="Hours with both a forecast and an actual")
    coverage_pct: float
    missing_forecast: int = Field(description="Hours with no forecast published before the cutoff")
    missing_actual: int = Field(description="Hours with no actual generation record")
    mae: float | None = Field(description="Mean absolute error, MW")
    rmse: float | None = Field(description="Root mean squared error, MW")
    bias: float | None = Field(description="Mean error, MW. Positive = over-forecast on average")
    median_abs_error: float | None
    p90_abs_error: float | None = Field(description="90% of hours had an absolute error below this, MW")
    error_p10: float | None = Field(description="10th percentile of the signed error, MW")
    error_p50: float | None
    error_p90: float | None = Field(description="90th percentile of the signed error, MW")
    max_abs_error: float | None
    max_error: float | None = Field(description="Signed error of the worst hour, MW")
    max_error_time: datetime | None
    mean_actual: float | None
    mean_forecast: float | None
    nmae_pct: float | None = Field(description="MAE as a percentage of mean actual generation")
    over_forecast_pct: float | None = Field(description="Share of hours where the forecast was too high")

    @classmethod
    def from_dict(cls, metrics: dict) -> "AccuracyMetrics":
        return cls(**{**metrics, "max_error_time": _datetime_or_none(metrics.get("max_error_time"))})


class IntervalPoint(BaseModel):
    """One half-hour. Forecast fields are filled on the hour only (WINDFOR is hourly)."""

    start_time: datetime
    actual_mw: float | None
    forecast_mw: float | None = Field(description="Forecast available at the decision cutoff")
    error_mw: float | None
    publish_time: datetime | None = Field(description="When the forecast used was published")
    horizon_hours: float | None = Field(description="How far ahead of the target the forecast was published")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> list["IntervalPoint"]:
        return [
            cls(
                start_time=_datetime_or_none(row.startTime),
                actual_mw=_float_or_none(row.actual),
                forecast_mw=_float_or_none(row.forecast),
                error_mw=_float_or_none(row.error),
                publish_time=_datetime_or_none(row.publishTime),
                horizon_hours=_float_or_none(row.horizon_hours),
            )
            for row in frame.itertuples(index=False)
        ]


class GroupStats(BaseModel):
    n: int
    mae: float | None
    rmse: float | None
    bias: float | None
    error_p10: float | None
    error_p90: float | None
    mean_actual: float | None
    mean_forecast: float | None


# ---- /api/health ------------------------------------------------------------


class CleaningReportOut(BaseModel):
    dataset: str
    raw_rows: int
    invalid_rows: int
    other_fuel_rows: int
    duplicate_rows: int
    past_target_rows: int
    clean_rows: int


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    detail: str | None = None
    data_source: str | None = Field(None, description="'snapshot' (CSV in backend/data) or 'elexon-live'")
    snapshot_fetched_at: datetime | None = None
    loaded_at: datetime | None = None
    decision_dates: DateRange | None = None
    delivery_dates: DateRange | None = None
    actual_intervals: int | None = None
    forecast_values: int | None = None
    forecast_runs: int | None = None
    cleaning: list[CleaningReportOut] = []
    ml_models: str | None = Field(None, description="Loaded ML models, or why they are unavailable")


# ---- /api/forecasts -----------------------------------------------------------


class ForecastRecord(BaseModel):
    start_time: datetime
    publish_time: datetime
    generation_mw: float
    horizon_hours: float


class ForecastsResponse(BaseModel):
    period: DateRange
    published_before: datetime | None
    count: int
    records: list[ForecastRecord]


class RunPoint(BaseModel):
    start_time: datetime
    generation_mw: float


class ForecastRun(BaseModel):
    publish_time: datetime
    available_at_decision: bool = Field(description="Published at or before the decision cutoff")
    used_for_decision: bool = Field(description="The latest run before the cutoff - the one the desk would use")
    points: list[RunPoint]


class ExpectedProfile(BaseModel):
    mean_mw: float
    min_mw: float
    min_time: datetime
    max_mw: float
    max_time: datetime


class DayAheadResponse(BaseModel):
    delivery_date: date
    decision_date: date
    decision_cutoff: datetime = Field(description="Gate closure on D-1, in UTC")
    forecast_run_used: datetime | None = Field(description="publishTime of the forecast used for the decision")
    minutes_before_cutoff: float | None = Field(description="How fresh that forecast was at the cutoff")
    policy: PolicyInfo
    expected: ExpectedProfile | None = Field(description="What the decision-time forecast said about the day")
    hindsight: AccuracyMetrics | None = Field(description="How accurate it turned out (null until actuals exist)")
    intervals: list[IntervalPoint]
    runs: list[ForecastRun] = Field(description="Every forecast run covering the day, before and after the cutoff")


# ---- /api/analytics -----------------------------------------------------------


class SummaryResponse(BaseModel):
    period: DateRange
    policy: PolicyInfo
    metrics: AccuracyMetrics


class TimeseriesResponse(BaseModel):
    period: DateRange
    policy: PolicyInfo
    points: list[IntervalPoint]


class BreakdownRow(GroupStats):
    group: str


class BreakdownResponse(BaseModel):
    period: DateRange
    policy: PolicyInfo
    by: Literal["hour_of_day", "delivery_date", "forecast_level"]
    rows: list[BreakdownRow]


class HorizonRow(GroupStats):
    horizon_hours: int = Field(description="Forecast published at least this many hours before the target")
    mean_lead_hours: float | None = Field(description="Actual average lead time of the forecasts used")


class HorizonResponse(BaseModel):
    period: DateRange
    rows: list[HorizonRow]


# ---- /api/ml --------------------------------------------------------------------
# Mirrors ml/reports/metrics.json, written by `python -m ml.train`.


class MlSplit(BaseModel):
    name: str
    decision_dates: list[date]
    delivery_dates: list[date]
    days: int
    hours: int


class MlFeature(BaseModel):
    name: str
    description: str


class MlImportance(BaseModel):
    feature: str
    importance: float = Field(description="Score loss when the feature is shuffled (MW of MAE, or Brier)")
    std: float


class MlRegressionCandidate(BaseModel):
    name: str
    description: str
    validation: dict[str, float]
    test: dict[str, float]
    validation_mae_change_pct: float = Field(description="MAE change vs the official forecast, %")
    test_mae_change_pct: float


class MlBootstrapInterval(BaseModel):
    mae_reduction_mw: float
    ci_low_mw: float
    ci_high_mw: float
    days: int
    method: str


class MlRegressionReport(BaseModel):
    target: str
    selection_rule: str
    selected: str
    candidates: list[MlRegressionCandidate]
    test_improvement_vs_official: MlBootstrapInterval
    feature_importance: list[MlImportance]


class MlClassificationCandidate(BaseModel):
    name: str
    description: str
    validation: dict[str, float | None]
    test: dict[str, float | None]


class MlCalibrationRow(BaseModel):
    band: str
    n: int
    mean_predicted: float
    observed_rate: float


class MlClassificationReport(BaseModel):
    target: str
    selection_rule: str
    probability_threshold: float
    selected: str
    candidates: list[MlClassificationCandidate]
    reference_always_predict_shortfall_test: dict[str, float]
    test_calibration: list[MlCalibrationRow]
    feature_importance: list[MlImportance]


class MlEvaluationResponse(BaseModel):
    generated_at: datetime
    sklearn_version: str
    decision_rule: str
    shortfall_threshold_mw: float
    splits: list[MlSplit]
    features: list[MlFeature]
    regression: MlRegressionReport
    classification: MlClassificationReport
    limitations: list[str]


class MlModelInfo(BaseModel):
    task: str = Field(description="'regression' (corrected forecast) or 'classification' (shortfall risk)")
    name: str
    role: Literal["default", "comparison"] = Field(description="Served by default, or kept for comparison")
    description: str
    inputs: list[str] = Field(description="Feature columns the model uses - all known at the 09:50 cutoff")
    trained_from: date
    trained_to: date


class MlServedModel(MlModelInfo):
    artifact: str | None = Field(description="Saved model file, relative to backend/ml")
    sha256: str | None
    matches_holdout_artifact: bool | None = Field(
        description="Byte-identical to the fingerprint sealed when the models were frozen (null if not sealed)"
    )
    january_test: dict[str, float | None] | None = Field(description="Scores on the January 2026 test set")
    selected_by_pre_january_rule: bool = Field(
        description="Chosen by the selection rule fixed before January (lowest December validation error)"
    )


class MlModelsResponse(BaseModel):
    status: str = Field(description="Scope, selection history of the defaults, and what the scores do and do not show")
    decision_cutoff_uk: str = Field(description="Every model input was available by this time on D-1")
    decision_dates: DateRange = Field(description="Purchase-decision dates the predictions cover (D-1)")
    served_period: DateRange = Field(description="Delivery dates the prediction endpoints cover")
    corrected_forecast: list[MlServedModel] = Field(description="Component 1 - regression models")
    shortfall_risk: list[MlServedModel] = Field(description="Component 2 - classification models")


class MlPredictionPoint(BaseModel):
    start_time: datetime
    publish_time: datetime = Field(description="When the forecast used was published - always <= decision_cutoff")
    decision_cutoff: datetime = Field(description="Gate closure the inputs respect: 09:50 UK time on D-1, in UTC")
    forecast_mw: float = Field(description="Official forecast available at the cutoff")
    corrected_forecast_mw: float = Field(description="Official forecast minus the model's predicted error")
    shortfall_probability: float = Field(description="Model probability that actual < forecast - 1 GW")
    actual_mw: float | None
    error_mw: float | None = Field(description="Official forecast error (forecast - actual)")
    corrected_error_mw: float | None
    shortfall_occurred: bool | None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> list["MlPredictionPoint"]:
        return [
            cls(
                start_time=_datetime_or_none(row.startTime),
                publish_time=_datetime_or_none(row.forecast_published),
                decision_cutoff=_datetime_or_none(row.cutoff),
                forecast_mw=round(float(row.forecast_mw), 1),
                corrected_forecast_mw=round(float(row.corrected_forecast_mw), 1),
                shortfall_probability=round(float(row.shortfall_probability), 4),
                actual_mw=_float_or_none(row.actual_mw),
                error_mw=_float_or_none(row.error_mw),
                corrected_error_mw=None if pd.isna(row.corrected_error_mw) else round(float(row.corrected_error_mw), 1),
                shortfall_occurred=None if pd.isna(row.shortfall_1gw) else bool(row.shortfall_1gw),
            )
            for row in frame.itertuples(index=False)
        ]


class MlPeriodScores(BaseModel):
    official: dict[str, float] = Field(description="MAE / RMSE / bias of the official forecast, MW")
    corrected: dict[str, float] = Field(description="MAE / RMSE / bias of the corrected forecast, MW")


class MlPredictionsResponse(BaseModel):
    period: DateRange
    regressor: MlModelInfo
    classifier: MlModelInfo
    scores: MlPeriodScores | None
    points: list[MlPredictionPoint]


class MlCorrectedForecastPoint(BaseModel):
    start_time: datetime
    publish_time: datetime = Field(description="When the forecast used was published - always <= decision_cutoff")
    decision_cutoff: datetime = Field(description="Gate closure the inputs respect: 09:50 UK time on D-1, in UTC")
    forecast_mw: float = Field(description="Official forecast available at the cutoff")
    corrected_forecast_mw: float = Field(description="Official forecast minus the model's predicted error")
    actual_mw: float | None
    error_mw: float | None = Field(description="Official forecast error (forecast - actual)")
    corrected_error_mw: float | None = Field(description="Corrected forecast error (corrected - actual)")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> list["MlCorrectedForecastPoint"]:
        return [
            cls(
                start_time=_datetime_or_none(row.startTime),
                publish_time=_datetime_or_none(row.forecast_published),
                decision_cutoff=_datetime_or_none(row.cutoff),
                forecast_mw=round(float(row.forecast_mw), 1),
                corrected_forecast_mw=round(float(row.corrected_forecast_mw), 1),
                actual_mw=_float_or_none(row.actual_mw),
                error_mw=_float_or_none(row.error_mw),
                corrected_error_mw=None if pd.isna(row.corrected_error_mw) else round(float(row.corrected_error_mw), 1),
            )
            for row in frame.itertuples(index=False)
        ]


class MlCorrectedForecastResponse(BaseModel):
    period: DateRange
    model: MlModelInfo
    scores: MlPeriodScores | None
    points: list[MlCorrectedForecastPoint]


class MlShortfallRiskPoint(BaseModel):
    start_time: datetime
    publish_time: datetime = Field(description="When the forecast used was published - always <= decision_cutoff")
    decision_cutoff: datetime = Field(description="Gate closure the inputs respect: 09:50 UK time on D-1, in UTC")
    forecast_mw: float = Field(description="Official forecast available at the cutoff")
    shortfall_probability: float = Field(description="Model probability that actual < forecast - 1 GW")
    shortfall_occurred: bool | None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> list["MlShortfallRiskPoint"]:
        return [
            cls(
                start_time=_datetime_or_none(row.startTime),
                publish_time=_datetime_or_none(row.forecast_published),
                decision_cutoff=_datetime_or_none(row.cutoff),
                forecast_mw=round(float(row.forecast_mw), 1),
                shortfall_probability=round(float(row.shortfall_probability), 4),
                shortfall_occurred=None if pd.isna(row.shortfall_1gw) else bool(row.shortfall_1gw),
            )
            for row in frame.itertuples(index=False)
        ]


class MlRiskScores(BaseModel):
    hours: int = Field(description="Hours with a known outcome in the period")
    observed_rate: float = Field(description="Share of those hours with a >= 1 GW shortfall")
    brier: float = Field(description="Mean squared probability error (lower is better)")
    roc_auc: float | None = Field(description="Ranking quality; null if only one outcome occurred")
    pr_auc: float | None = Field(description="Average precision; compare with observed_rate (no-skill level)")
    probability_threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float = Field(description="Misleading on its own: shortfalls are the majority class")
    predicted_rate: float


class MlShortfallRiskResponse(BaseModel):
    period: DateRange
    model: MlModelInfo
    scores: MlRiskScores | None
    points: list[MlShortfallRiskPoint]
