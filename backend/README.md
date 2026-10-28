# Backend: Wind Forecast Intelligence API

A Python / FastAPI service that answers two questions for a UK day-ahead power-buying desk:

1. **Which wind forecast did we have when we had to buy?** The latest forecast published before the day-ahead gate closure on D-1.
2. **How accurate was it?** MAE, RMSE, bias, error percentiles and breakdowns, measured against actual generation.

The data is Elexon BMRS **WINDFOR** (forecasts) and **FUELHH** (actual wind generation), national level, for purchase decisions made in **January 2026**.

The design reasoning is in [../docs/BACKEND_DESIGN.md](../docs/BACKEND_DESIGN.md) and the data evidence in [../docs/DATA_FINDINGS.md](../docs/DATA_FINDINGS.md).

---

## Quick start

### Environment requirements

| Requirement | Detail |
| --- | --- |
| Python | **3.12 or newer** (`python --version`). The pinned NumPy and pandas versions need it |
| Operating system | Windows, macOS or Linux |
| Packages | Exact versions in `requirements.txt` (FastAPI, uvicorn, pandas, NumPy, scikit-learn, joblib, pydantic-settings, httpx, tzdata). `requirements-dev.txt` adds pytest |
| Database, API key, network | None needed. The January 2026 data snapshot (`data/`) and the trained models (`ml/`) ship with the repository, so the API runs offline |

### Windows (PowerShell)

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

### macOS / Linux

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Start-up is ready when the log shows `ML models loaded` and `Application startup complete`. Then open these local URLs:

| What | URL |
| --- | --- |
| Swagger UI (try every endpoint) | <http://127.0.0.1:8000/docs> |
| Health and loaded models | <http://127.0.0.1:8000/api/health> |
| Served models (roles, inputs, scores) | <http://127.0.0.1:8000/api/ml/models> |
| Corrected forecast, one day | <http://127.0.0.1:8000/api/ml/corrected-forecast?start=2026-01-15&end=2026-01-15> |
| Shortfall risk, one day | <http://127.0.0.1:8000/api/ml/shortfall-risk?start=2026-01-15&end=2026-01-15> |

These are local addresses only; there is no public deployment yet.

> On Windows, if `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload` without activating. If port 8000 is busy, add `--port 8001` and use that port in the URLs.

### What the server loads at start-up

The application needs only the files below. Everything else in the repository is for rebuilding, evaluating or experimenting. If an ML file is missing or unreadable, only `/api/ml/*` returns 503, and `/api/health` names the file.

| Files | Used by |
| --- | --- |
| `data/windfor_raw.csv`, `data/fuelhh_wind_raw.csv`, `data/snapshot_meta.json` | Forecast and accuracy endpoints (January 2026 snapshot) |
| `ml/processed/ml_dataset.csv` | ML endpoints: the January features, built with the 09:50 cutoff |
| `ml/holdout_feb2026/models/regression_linear_by_level.joblib`, `ml/holdout_feb2026/models/classification_logistic_by_level.joblib` | The approved default models |
| `ml/models/error_regressor.joblib`, `ml/models/shortfall_classifier.joblib` | The Random Forest comparison models |
| `ml/reports/metrics.json` | `/api/ml/evaluate` and the January scores in `/api/ml/models` |
| `ml/holdout_feb2026/sealed_manifest.json` | Artifact fingerprints reported by `/api/ml/models` |

Models are loaded once at start-up and never trained or reloaded by a request.

### Calling the API

Every endpoint is a `GET` that returns JSON. With the server running:

```bash
# macOS / Linux / Git Bash  (on Windows PowerShell, type curl.exe instead of curl)
curl "http://127.0.0.1:8000/api/health"
curl "http://127.0.0.1:8000/api/forecasts/day-ahead?delivery_date=2026-01-15"
curl "http://127.0.0.1:8000/api/analytics/summary?gate_closure=09:20"
curl "http://127.0.0.1:8000/api/ml/corrected-forecast?start=2026-01-15&end=2026-01-15"
curl "http://127.0.0.1:8000/api/ml/shortfall-risk?start=2026-01-15&end=2026-01-15"
curl "http://127.0.0.1:8000/api/ml/shortfall-risk?start=2026-01-15&end=2026-01-15&model=random_forest"
curl "http://127.0.0.1:8000/api/ml/models"
```

```powershell
# Windows PowerShell: parsed into objects
Invoke-RestMethod "http://127.0.0.1:8000/api/ml/shortfall-risk?start=2026-01-15&end=2026-01-15" | ConvertTo-Json -Depth 5
```

The [API reference](#api-reference) below shows the response of each endpoint.

### Run the tests

```bash
pytest            # 125 tests, about 30-60 seconds, no network needed; ends with "125 passed"
```

The production suite covers:

- every endpoint, and the 422 response for each kind of invalid input
- the 09:50 UK cutoff, including clock-change days
- leakage
- the January-only scope
- missing or broken model files
- the fact that requests never train or reload a model

The offline experiments in `ml/v2/` have their own tests. These are not part of the application:

```bash
pytest ml/v2/tests ml/v2/existing_data/tests ml/v2/ridge_gbm/tests
```

### ML workflow (optional extension)

Trained models are included, so the `/api/ml/*` endpoints work immediately. To rebuild them from the raw monthly files in `../DATASETS` (actuals) and `../DATASETS1` (forecasts):

```bash
python -m ml.build_dataset   # validate the raw files, then build the leakage-free dataset
python -m ml.train           # train, compare and evaluate the models (about 40 s), save everything
```

The method, results and limitations are documented in [ml/README.md](ml/README.md).

The **February 2026 fresh holdout** re-tests the frozen models on data that played no part in any choice. Its predictions are sealed before the labels are read:

```bash
python -m ml.holdout.predict    # verify frozen models, audit the 09:50 cutoff, seal predictions
python -m ml.holdout.evaluate   # verify the seal, then evaluate against February actuals
```

See [ml/holdout_feb2026/README.md](ml/holdout_feb2026/README.md).

### Refresh the data from Elexon (optional)

```bash
python -m scripts.fetch_data
```

This re-downloads the data for the configured window into `data/` (about 10 requests and a few seconds). The API reads the snapshot at start-up, so restart the server afterwards.

---

## Configuration

Every setting has a default. To change one, copy `.env.example` to `.env` or set environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `WFM_FIRST_DECISION_DATE` | `2026-01-01` | First purchase-decision day (D-1) |
| `WFM_LAST_DECISION_DATE` | `2026-01-31` | Last purchase-decision day. Delivery days run one day later |
| `WFM_DEFAULT_GATE_CLOSURE` | `09:50` | Decision cutoff, UK local time on D-1 (Nord Pool GB day-ahead gate closure) |
| `WFM_DATA_SOURCE` | `snapshot` | `snapshot`: use `data/*.csv`, downloading once if missing. `live`: always download at start-up |
| `WFM_CORS_ORIGINS` | `["http://localhost:5173", "http://127.0.0.1:5173"]` | Front-end origins allowed to call the API (JSON list) |
| `WFM_ELEXON_BASE_URL` | `https://data.elexon.co.uk/bmrs/api/v1` | Elexon Insights API |
| `WFM_ELEXON_TIMEOUT_SECONDS` / `_MAX_RETRIES` / `_CHUNK_DAYS` | `60` / `3` / `7` | HTTP robustness settings |
| `WFM_ML_DEFAULT_REGRESSOR` | `linear_by_level` (approved) | Default corrected-forecast model: `linear_by_level` or `random_forest` |
| `WFM_ML_DEFAULT_CLASSIFIER` | `logistic_by_level` (approved) | Default shortfall-risk model: `logistic_by_level` or `random_forest` |

If you change the decision window, run `python -m scripts.fetch_data` to download matching data.

---

## Conventions used by every endpoint

- **Times** are UTC ISO-8601 with a trailing `Z`. The UI converts them to UK time.
- **Delivery date** means the UK calendar day of delivery (D). The purchase decision is made on D-1.
- **Error = forecast − actual**, in MW. A positive error is an over-forecast: less wind than planned, so the buyer is **short**.
- **Resolution:** forecasts are hourly and actuals half-hourly. Accuracy is scored per hour. Time series return every half-hour, with forecast fields filled on the hour.
- **Missing data is `null`**, never 0.

### Shared query parameters (all `/api/analytics/*` endpoints)

| Parameter | Default | Description |
| --- | --- | --- |
| `start`, `end` | whole window | Delivery dates `YYYY-MM-DD`, inclusive, within 2026-01-02 … 2026-02-01 |
| `policy` | `day_ahead` | `day_ahead`: the latest forecast before gate closure on D-1. `fixed_horizon`: the latest forecast published at least `horizon_hours` before each hour |
| `gate_closure` | `09:50` | `HH:MM` UK local time on D-1 (EPEX's GB auction is about `09:20`) |
| `horizon_hours` | `24` | 0–48, used with `policy=fixed_horizon` |

Invalid values return **HTTP 422** with a readable message.

---

## API reference

### `GET /api/health`

Service status, which data is loaded, and how many rows each cleaning rule removed.

```json
{
  "status": "ok",
  "data_source": "snapshot",
  "decision_dates": { "first": "2026-01-01", "last": "2026-01-31" },
  "delivery_dates": { "first": "2026-01-02", "last": "2026-02-01" },
  "actual_intervals": 1536,
  "forecast_values": 15775,
  "forecast_runs": 272,
  "ml_models": "loaded: corrected_forecast=linear_by_level (default) + random_forest, shortfall_risk=logistic_by_level (default) + random_forest",
  "cleaning": [
    { "dataset": "WINDFOR (wind generation forecast)", "raw_rows": 19856, "past_target_rows": 4081, "clean_rows": 15775, "...": "..." }
  ]
}
```

It returns **503** with a `detail` if the data could not be loaded.

### `GET /api/forecasts/day-ahead?delivery_date=2026-01-15[&gate_closure=09:50]`

The decision view for one delivery day.

```json
{
  "delivery_date": "2026-01-15",
  "decision_date": "2026-01-14",
  "decision_cutoff": "2026-01-14T09:50:00Z",
  "forecast_run_used": "2026-01-14T09:30:00Z",
  "minutes_before_cutoff": 20.0,
  "expected": { "mean_mw": 11678.0, "min_mw": 8957.0, "min_time": "2026-01-15T16:00:00Z", "max_mw": 15828.0, "max_time": "2026-01-15T00:00:00Z" },
  "hindsight": { "mae": "...", "bias": "...", "...": "same fields as summary metrics" },
  "intervals": [
    { "start_time": "2026-01-15T00:00:00Z", "actual_mw": 14602.0, "forecast_mw": 15828.0, "error_mw": 1226.0,
      "publish_time": "2026-01-14T09:30:00Z", "horizon_hours": 14.5 },
    { "start_time": "2026-01-15T00:30:00Z", "actual_mw": 14311.0, "forecast_mw": null, "error_mw": null, "...": "..." }
  ],
  "runs": [
    { "publish_time": "2026-01-14T09:30:00Z", "available_at_decision": true, "used_for_decision": true, "points": [ { "start_time": "...", "generation_mw": 15828.0 } ] },
    { "publish_time": "2026-01-14T11:30:00Z", "available_at_decision": false, "used_for_decision": false, "points": [ "..." ] }
  ]
}
```

`runs` lists every forecast run that covers the day, both before and after the cutoff, to show how the forecast moved after the decision.

### `GET /api/forecasts?start=…&end=…[&published_before=2026-01-14T09:50:00Z]`

Raw forecast records (every run) for target hours in the period. `published_before` limits the result to what was known at that moment. It takes ISO-8601, and a value without an offset is read as UTC.

```json
{
  "period": { "first": "2026-01-15", "last": "2026-01-15" },
  "published_before": "2026-01-14T09:50:00Z",
  "count": 272,
  "records": [
    { "start_time": "2026-01-15T00:00:00Z", "publish_time": "2026-01-13T02:30:00Z", "generation_mw": 13924.0, "horizon_hours": 45.5 },
    "..."
  ]
}
```

### `GET /api/analytics/summary`

Headline accuracy and coverage. These are the January 2026 values with the defaults:

```json
{
  "policy": { "kind": "day_ahead", "gate_closure_uk": "09:50", "description": "Latest forecast published by 09:50 UK time on the day before delivery" },
  "metrics": {
    "intervals_total": 744, "intervals_scored": 744, "coverage_pct": 100.0,
    "missing_forecast": 0, "missing_actual": 0,
    "mae": 1995.6, "rmse": 2498.8, "bias": 1728.0,
    "median_abs_error": 1625.5, "p90_abs_error": 4183.9,
    "error_p10": -502.7, "error_p50": 1506.5, "error_p90": 4183.9,
    "max_abs_error": 6857.0, "max_error": 6857.0, "max_error_time": "2026-01-12T01:00:00Z",
    "mean_actual": 11285.5, "mean_forecast": 13013.5,
    "nmae_pct": 17.7, "over_forecast_pct": 84.3
  }
}
```

### `GET /api/analytics/timeseries`

Half-hourly points for charting. The whole month is 1,488 points. Forecasts are hourly, so the forecast fields are `null` on the half-hours.

```json
{
  "period": { "first": "2026-01-15", "last": "2026-01-15" },
  "policy": { "kind": "day_ahead", "gate_closure_uk": "09:50", "horizon_hours": null, "description": "..." },
  "points": [
    { "start_time": "2026-01-15T00:00:00Z", "actual_mw": 14602.0, "forecast_mw": 15828.0, "error_mw": 1226.0, "publish_time": "2026-01-14T09:30:00Z", "horizon_hours": 14.5 },
    { "start_time": "2026-01-15T00:30:00Z", "actual_mw": 14311.0, "forecast_mw": null, "error_mw": null, "publish_time": null, "horizon_hours": null }
  ]
}
```

### `GET /api/analytics/breakdown?by=hour_of_day|delivery_date|forecast_level`

Metrics per group.

```json
{
  "by": "hour_of_day",
  "rows": [
    { "group": "00:00", "n": 31, "mae": 1970.5, "rmse": 2503.3, "bias": 1906.9, "error_p10": 105.0, "error_p90": 4151.0, "mean_actual": 11662.1, "mean_forecast": 13569.0 },
    "..."
  ]
}
```

### `GET /api/analytics/horizon[?max_hours=48&step_hours=1]`

For each lead time h, the accuracy of "the latest forecast published at least h hours before the target". `mean_lead_hours` gives the real average lead time. `step_hours` is at most 12.

```json
{
  "rows": [
    { "horizon_hours": 0, "mean_lead_hours": 1.6, "n": 744, "mae": 1679.9, "rmse": 2141.1, "bias": 1471.1, "...": "..." },
    { "horizon_hours": 12, "mean_lead_hours": 13.6, "n": 744, "mae": 1791.2, "rmse": 2263.6, "bias": 1595.1, "...": "..." },
    { "horizon_hours": 24, "mean_lead_hours": 25.6, "n": 744, "mae": 1932.2, "rmse": 2399.4, "bias": 1736.2, "...": "..." }
  ]
}
```

### ML endpoints: two separate prediction components

The ML extension serves two components. Each has its own models, inputs, outputs and scores, and no model does both jobs.

| Component | Question | Default (approved) | Comparison | Inputs of the default |
| --- | --- | --- | --- | --- |
| Corrected forecast (regression) | How wrong will the official forecast be, in MW? | `linear_by_level` | `random_forest` | `forecast_mw` |
| Shortfall risk (classification) | How likely is actual wind to come in at least 1 GW below the forecast? | `logistic_by_level` | `random_forest` | `forecast_mw` |

- **Scope:** every prediction is for a purchase decision made 1–31 January 2026, and `tests/test_scope.py` enforces this. February 2026 results are offline evidence ([holdout_feb2026](ml/holdout_feb2026/README.md)); the API never loads them.
- **Defaults:** approved for the current application on 2026-10-26. They remain subject to the safeguards below.
  - **Selection history:** the rule fixed before January (lowest December error) chose the Random Forests. The defaults were chosen later, using the January test and an offline February check. So the choice reflects post-January evidence; the inputs and model weights do not. Details are in the [root README](../README.md#model-selection-history).
  - This is a selection on January–February 2026 data: not proof of long-term superiority, and not evidence of trading profit.
  - The decision record is in [ml/README.md](ml/README.md).
  - `WFM_ML_DEFAULT_REGRESSOR` and `WFM_ML_DEFAULT_CLASSIFIER` can switch a default, but only as part of a new selection decision.
- **Cutoff:** every input is known by **09:50 UK time on D-1**, the cutoff the models were trained with. So the ML endpoints always use 09:50, whatever `WFM_DEFAULT_GATE_CLOSURE` says.
- **Start-up checks:** the service refuses:
  - features built for another cutoff or from a later forecast
  - any model that reads a label column
  - any model placed in the wrong component
  - a feature table that is missing a model input, or an hour without a forecast
- **Coverage:** the prediction endpoints cover the January delivery days (2026-01-02 … 2026-02-01) and accept `start` and `end`.
- **Saved models only:** models are loaded once at start-up. A request never trains or reloads a model; a test enforces this.
- **Traceable cutoff:** every prediction point carries `publish_time` (the forecast run it used) and `decision_cutoff`, so anyone can check that `publish_time <= decision_cutoff`.
- **Errors:** see [Errors](#errors).

### `GET /api/ml/corrected-forecast[?start=…&end=…&model=random_forest]`

Component 1, hour by hour: the official forecast, the corrected forecast, the actual and both errors (MW). `scores` compares the official and corrected forecasts (MAE, RMSE and bias, in MW) over the requested period.

```text
GET /api/ml/corrected-forecast?start=2026-01-15&end=2026-01-15
```

```json
{
  "period": { "first": "2026-01-15", "last": "2026-01-15" },
  "model": {
    "task": "regression", "name": "linear_by_level", "role": "default",
    "description": "Linear correction by forecast size", "inputs": ["forecast_mw"],
    "trained_from": "2025-10-02", "trained_to": "2025-12-31"
  },
  "scores": {
    "official": { "mae": 1093.6, "rmse": 1229.4, "bias": 1093.6 },
    "corrected": { "mae": 864.0, "rmse": 984.7, "bias": -288.1 }
  },
  "points": [
    {
      "start_time": "2026-01-15T00:00:00Z", "publish_time": "2026-01-14T09:30:00Z", "decision_cutoff": "2026-01-14T09:50:00Z",
      "forecast_mw": 15828.0, "corrected_forecast_mw": 13467.1,
      "actual_mw": 14602.0, "error_mw": 1226.0, "corrected_error_mw": -1134.9
    },
    "... 23 more hours"
  ]
}
```

### `GET /api/ml/shortfall-risk[?start=…&end=…&model=random_forest]`

Component 2, hour by hour: the probability (0–1) that actual wind comes in at least 1 GW below the official forecast, and whether it did. `scores` gives:

- the Brier score
- ROC-AUC and PR-AUC
- precision, recall and F1 at a 0.5 threshold
- the observed shortfall rate

Accuracy is included, but it misleads on its own because shortfalls are the majority class. `roc_auc` and `pr_auc` are `null` when only one outcome occurred in the period.

```text
GET /api/ml/shortfall-risk?start=2026-01-15&end=2026-01-15
```

```json
{
  "period": { "first": "2026-01-15", "last": "2026-01-15" },
  "model": {
    "task": "classification", "name": "logistic_by_level", "role": "default",
    "description": "Logistic regression on forecast size", "inputs": ["forecast_mw"],
    "trained_from": "2025-10-02", "trained_to": "2025-12-31"
  },
  "scores": {
    "hours": 24, "observed_rate": 0.5417, "brier": 0.3372, "roc_auc": 0.3497, "pr_auc": 0.548,
    "probability_threshold": 0.5, "precision": 0.3333, "recall": 0.3077, "f1": 0.32, "accuracy": 0.2917, "predicted_rate": 0.5
  },
  "points": [
    {
      "start_time": "2026-01-15T00:00:00Z", "publish_time": "2026-01-14T09:30:00Z", "decision_cutoff": "2026-01-14T09:50:00Z",
      "forecast_mw": 15828.0, "shortfall_probability": 0.7797, "shortfall_occurred": true
    },
    "... 23 more hours"
  ]
}
```

One day is a small sample, so its scores are noisy. On 15 Jan the logistic model ranked hours worse than chance. Judge model quality over the whole month, or from the holdout results in `/api/ml/models`. Add `&model=random_forest` for the comparison classifier; `model.role` is then `"comparison"`.

### `GET /api/ml/predictions[?start=…&end=…&regressor=…&classifier=…]`

Both components side by side in one call (for the dashboard), each computed by its own model. `scores` compares the official and corrected forecasts over the period.

```text
GET /api/ml/predictions?start=2026-01-15&end=2026-01-15&classifier=random_forest
```

```json
{
  "period": { "first": "2026-01-15", "last": "2026-01-15" },
  "regressor": { "task": "regression", "name": "linear_by_level", "role": "default", "...": "..." },
  "classifier": { "task": "classification", "name": "random_forest", "role": "comparison", "inputs": ["forecast_mw", "lead_hours", "...9 features"], "...": "..." },
  "scores": { "official": { "mae": 1093.6, "...": "..." }, "corrected": { "mae": 864.0, "...": "..." } },
  "points": [
    {
      "start_time": "2026-01-15T00:00:00Z", "publish_time": "2026-01-14T09:30:00Z", "decision_cutoff": "2026-01-14T09:50:00Z",
      "forecast_mw": 15828.0, "corrected_forecast_mw": 13467.1, "shortfall_probability": 0.6135,
      "actual_mw": 14602.0, "error_mw": 1226.0, "corrected_error_mw": -1134.9, "shortfall_occurred": true
    }
  ]
}
```

### `GET /api/ml/models`

Every served model, with:

- its role (`default` or `comparison`) and the inputs it actually uses
- its training window
- its artifact path and SHA-256, and whether it is byte-identical to the fingerprint sealed when the models were frozen
- its January test scores
- `selected_by_pre_january_rule`: whether the rule fixed before January (lowest December error) chose it

`status` states the January scope, the selection history and the limits of the scores.

```json
{
  "status": "Scope: purchase decisions made 1-31 January 2026. Defaults approved on 2026-10-26 for the current application. Selection history: the rule fixed before January (lowest December validation error) chose the Random Forests; the defaults were chosen later ... they are not proof of long-term superiority or evidence of trading profit.",
  "decision_cutoff_uk": "09:50",
  "decision_dates": { "first": "2026-01-01", "last": "2026-01-31" },
  "served_period": { "first": "2026-01-02", "last": "2026-02-01" },
  "corrected_forecast": [
    {
      "task": "regression", "name": "linear_by_level", "role": "default", "inputs": ["forecast_mw"],
      "trained_from": "2025-10-02", "trained_to": "2025-12-31",
      "artifact": "holdout_feb2026/models/regression_linear_by_level.joblib", "sha256": "e7704b7b...", "matches_holdout_artifact": true,
      "january_test": { "mae": 1178.6, "rmse": 1535.1, "bias": 31.2 },
      "selected_by_pre_january_rule": false
    },
    { "name": "random_forest", "role": "comparison", "january_test": { "mae": 1381.3, "...": "..." }, "selected_by_pre_january_rule": true, "...": "..." }
  ],
  "shortfall_risk": [
    { "name": "logistic_by_level", "role": "default", "january_test": { "brier": 0.2015, "roc_auc": 0.755, "...": "..." }, "selected_by_pre_january_rule": false, "...": "..." },
    { "name": "random_forest", "role": "comparison", "january_test": { "brier": 0.2209, "roc_auc": 0.7232, "...": "..." }, "selected_by_pre_january_rule": true, "...": "..." }
  ]
}
```

The linear and logistic files sit in `holdout_feb2026/models/` because they were first saved during that check. They contain October–December fits only: they reproduce their January predictions exactly.

### `GET /api/ml/evaluate`

The January model comparison exactly as produced by `python -m ml.train`. It includes:

- the splits and the feature list
- every regression and classification candidate, with validation and January test scores
- which model the January selection rule picked, and why. This is a historical record. The models served now are listed by `/api/ml/models`
- a bootstrap confidence interval, calibration results and feature importance
- the limitations

### Errors

| Status | When | Body |
| --- | --- | --- |
| **422** | A parameter has the wrong type or format, such as a bad date, an unknown model name or a model of the other task | FastAPI's validation list, naming the parameter |
| **422** | A valid value is outside the data window, or `start` is after `end` | `{"detail": "<readable message>"}` |
| **503** | The data, or (for `/api/ml/*` only) the models, could not be loaded at start-up | `{"detail": "<reason>"}`. `/api/health` shows the same reason |

```text
GET /api/ml/shortfall-risk?model=linear_by_level        (a regressor is not a shortfall model)
```

```json
{ "detail": [ { "type": "literal_error", "loc": ["query", "model"], "msg": "Input should be 'logistic_by_level' or 'random_forest'", "input": "linear_by_level" } ] }
```

```text
GET /api/ml/predictions?start=2026-03-01
```

```json
{ "detail": "Delivery dates must be within 2026-01-02 .. 2026-02-01 with start <= end" }
```

```text
GET /api/ml/predictions          (model artifacts missing)
```

```json
{ "detail": "ML unavailable: [Errno 2] No such file or directory: '.../ml/processed/ml_dataset.csv' - the artifacts are built by `python -m ml.build_dataset`, `python -m ml.train` and `python -m ml.holdout.predict`" }
```

---

## Project layout

```text
backend/
├── app/
│   ├── main.py                  # app factory: start-up, CORS, routers
│   ├── config.py                # settings (env vars / .env)
│   ├── api/
│   │   ├── dependencies.py      # shared validation: data store, period, policy, ML models
│   │   └── routes/              # health.py, forecasts.py, analytics.py, ml.py (thin)
│   ├── services/
│   │   ├── elexon_client.py     # HTTP to Elexon: chunking, retries
│   │   ├── data_processor.py    # validation + cleaning with an audit trail
│   │   ├── data_store.py        # snapshot cache + in-memory data
│   │   ├── forecast_evaluator.py# business logic: selection, metrics, breakdowns
│   │   ├── feature_builder.py   # leakage-free ML features (09:50 cutoff)
│   │   └── prediction_service.py# loads the trained models and serves predictions
│   ├── models/
│   │   ├── schemas.py           # API response contract (pydantic)
│   │   └── ml_model.py          # ML candidates, scoring, model save/load
│   └── utils/timestamps.py      # all UTC / UK-time logic
├── data/                        # January 2026 Elexon snapshot (CSV + metadata)
├── ml/                          # offline ML workflow: build_dataset.py, train.py, outputs, README
│   ├── holdout/, holdout_feb2026/  # February 2026 sealed evaluation (and the default model files)
│   └── v2/                      # offline experiments - not used by the application (see ml/v2/existing_data/README.md, ml/v2/ridge_gbm/README.md)
├── scripts/fetch_data.py        # refresh the snapshot
├── tests/                       # pytest suite (synthetic, leakage and real-data regression tests)
├── requirements.txt             # runtime dependencies (pinned)
├── requirements-dev.txt         # + pytest
└── .env.example                 # configuration template
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'app'` | Run commands from inside the `backend/` folder |
| `ZoneInfoNotFoundError: Europe/London` | `pip install -r requirements.txt` (installs `tzdata`, which Windows needs) |
| `/api/health` returns 503 | The snapshot is missing and Elexon could not be reached. Check your connection and run `python -m scripts.fetch_data` |
| `/api/ml/*` returns 503 | A model artifact is missing, empty or unreadable, or the feature table failed a start-up check. `/api/health` (`ml_models`) shows the reason and names the file |
| Browser shows a CORS error | Add the front-end URL to `WFM_CORS_ORIGINS` |
| Port 8000 already in use | `uvicorn app.main:app --reload --port 8001` |
