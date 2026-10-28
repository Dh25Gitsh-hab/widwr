# Backend design: why it is built this way, and how it works

This document explains the reasoning behind the Python backend, step by step. It is written so the design can be explained to a technical reviewer, from the business problem down to individual functions.

Related documents: [DATA_FINDINGS.md](DATA_FINDINGS.md) (the data evidence) and [backend/README.md](../backend/README.md) (how to run it, plus the API reference).

---

## 0. In one paragraph

The backend downloads two Elexon datasets once: wind forecasts (WINDFOR) and actual wind generation (FUELHH). It cleans them, keeps them in memory and serves a REST API with two jobs:

1. **Rebuild exactly which forecast the trading desk had at the day-ahead decision cutoff.** For each hour, that is the latest forecast published before gate closure on the previous day.
2. **Measure how accurate those forecasts were**: MAE, RMSE, bias, error percentiles, and breakdowns by hour, day, forecast size and lead time.

The design principle is **temporal correctness first**. A forecast published after the decision can never be used to judge that decision.

---

## 1. From the business problem to engineering requirements

The buyer must purchase on day D-1 the power needed for day D. Roughly, purchase ≈ demand − expected wind. Wind is uncertain, so the buyer relies on a forecast. The brief asks the product to:

| Business need | Engineering requirement |
| --- | --- |
| Q1: "Understand the latest forecasts available when we need to make decisions" | **R1** For any delivery day, return the forecast as it stood at the decision cutoff, and say which run it was and when it was published |
| Q2: "Evaluate accuracy quantitatively" | **R2** Compare those decision-time forecasts with actuals and compute standard error metrics |
| "...so they can decide considering the quantified error" | **R3** Report the *distribution* of the error (bias, P10/P90), not only an average, and show *when* the error is large |
| Scope: Elexon national wind only, January 2026 decisions | **R4** Fetch exactly that data and make the window configurable |
| A product for a client, reviewed by others | **R5** Reproducible, runs offline on any machine, fails gracefully, is tested and documented |

What we are **not** building: our own weather or wind model, a trading engine, or a demand forecast. The scope gives us no demand data, so the product supports the purchase decision without computing it.

---

## 2. The thought process, in the order it happened

### Step 1: Understand the domain before the data

- Day-ahead auctions close on D-1. Nord Pool's GB auction closes at 09:50 UK time and EPEX's at about 09:20. This defines the **cutoff**.
- A forecast is useless to a decision if it was published after the decision. Using it anyway is **look-ahead bias**: the accuracy numbers would look better than reality.

### Step 2: Profile the real data before writing pipeline code

This step is in [DATA_FINDINGS.md](DATA_FINDINGS.md). Four facts shaped the design:

- WINDFOR is **hourly**, while actuals are **half-hourly**. That leads to scoring per hour.
- There are **8 runs a day**, so the same target hour has many competing forecasts. That leads to a selection rule.
- Each run repeats **hours that had already started**, which are not real forecasts. Those are dropped.
- The forecast is **biased by about +1.7 GW**. That becomes the headline insight for the client and the starting point for the ML discussion.

### Step 3: Define the core rule precisely, before the code

For every target hour *t*:

```text
cutoff(t)   = gate closure (UK local time) on the day before t's UK delivery day
eligible(t) = { forecast runs for hour t  with  publishTime <= cutoff(t) }
chosen(t)   = the eligible run with the latest publishTime   (none -> "missing")
error(t)    = chosen(t).generation - actual(t)               (positive = over-forecast)
```

A second policy, `fixed_horizon`, uses `cutoff(t) = t - h`. It answers a different question: how accuracy changes with lead time.

### Step 4: Choose an architecture that keeps that rule isolated and testable

The backend is a **modular monolith**: one deployable FastAPI app, split into layers that each have one responsibility. Microservices would add deployment and network complexity with no benefit at this size.

### Step 5: Build bottom-up, testing each layer

The order was time handling, then the API client, cleaning, storage, evaluation, response schemas and finally the HTTP routes. Each layer only depends on the ones below it.

### Step 6: Prove the rule with hand-checkable tests, then with real-data regression tests

### Step 7: Make failures visible and graceful

Every removed row is counted. Bad input gets a readable HTTP 422. If the data cannot be loaded, the server still starts and reports the reason.

### Step 8: Document the reasoning (this file)

---

## 3. Architecture

```text
                     ┌───────────────────────── FastAPI app (app/main.py) ─────────────────────────┐
  Browser / React    │                                                                              │
  ───── HTTP ──────► │  api/routes/        health.py   forecasts.py   analytics.py                  │
                     │        │ (thin: validate → call service → build schema)                     │
                     │  api/dependencies.py   store · period · policy  (shared validation)          │
                     │        │                                                                     │
                     │  services/forecast_evaluator.py   ◄── the business logic (Q1 + Q2)          │
                     │        │                                                                     │
                     │  services/data_store.py   in-memory clean data + provenance                 │
                     │        │            ▲                                                        │
                     │        │            │ clean once at start-up                                 │
                     │        │   services/data_processor.py   validation + cleaning               │
                     │        │            ▲                                                        │
                     │        ▼            │                                                        │
                     │  backend/data/*.csv snapshot  ◄── services/elexon_client.py ◄── Elexon API   │
                     │                                                                              │
                     │  utils/timestamps.py (all time-zone logic)   models/schemas.py (API contract) │
                     └──────────────────────────────────────────────────────────────────────────────┘
```

**Start-up flow:** `config` → `data_store.load_store()` → read the CSV snapshot (or download it via `elexon_client`) → `data_processor` cleans it → DataFrames are held in `app.state`.

**Request flow** (e.g. `GET /api/analytics/summary?gate_closure=09:20`):

1. The dependencies validate the query and build a `Period` and a `SelectionPolicy`.
2. The route calls `forecast_evaluator.evaluate()` and then `compute_metrics()`.
3. The result is wrapped in the `SummaryResponse` schema and returned as JSON.

---

## 4. Folder structure: blueprint vs. what was built

```text
backend/
├── app/
│   ├── main.py                    # app factory, start-up, CORS, router registration
│   ├── config.py                  # typed settings from environment variables
│   ├── api/
│   │   ├── dependencies.py        # + shared request validation (store, period, policy)
│   │   └── routes/
│   │       ├── health.py
│   │       ├── forecasts.py
│   │       ├── analytics.py
│   │       └── ml.py              # /api/ml/corrected-forecast, /shortfall-risk, /predictions, /models, /evaluate
│   ├── services/
│   │   ├── elexon_client.py
│   │   ├── data_processor.py
│   │   ├── data_store.py          # + snapshot caching and in-memory store
│   │   ├── forecast_evaluator.py
│   │   ├── feature_builder.py     # + leakage-free ML features (same cutoff rule)
│   │   └── prediction_service.py  # two prediction components (default + comparison model each)
│   ├── models/
│   │   ├── schemas.py
│   │   └── ml_model.py            # ML candidates, scoring, save/load
│   └── utils/
│       └── timestamps.py
├── data/                          # + committed January 2026 snapshot (1.2 MB)
├── ml/                            # + offline ML workflow and its outputs (see ml/README.md)
├── scripts/fetch_data.py          # + command to refresh the snapshot
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

The blueprint's structure is followed as given. There are six small additions, each with a reason:

| Addition | Why it was needed |
| --- | --- |
| `services/data_store.py` | Caching and holding data is a different job from fetching it (`elexon_client`) or cleaning it (`data_processor`). Without this module that logic would leak into `main.py` or the routes. |
| `api/dependencies.py` | Six endpoints share the same validation (dates in range, gate-closure format, policy). Writing it once keeps the rules and messages identical, and lets tests inject fake data. |
| `data/` snapshot | The app runs offline and gives the same numbers on every machine, even if Elexon is down or later revises its data. |
| `scripts/fetch_data.py` | One documented command to refresh the snapshot. |
| `services/feature_builder.py` | The ML features must obey the same no-look-ahead rule as the API. Keeping them in one module, built on the API's own selection function, makes that reviewable and testable. |
| `ml/` | The blueprint's ML pipeline (build data → features → chronological split → baselines → evaluate → save) as two reproducible commands, with their outputs kept next to them. |

---

## 5. Module by module

### `utils/timestamps.py`: all time-zone logic in one place

- **Why:** the product is entirely about *when*. One time-zone bug selects the wrong forecast run without any visible error.
- **How:**
  - Everything is stored as timezone-aware UTC.
  - UK concepts (the delivery day and gate closure) are converted with `zoneinfo("Europe/London")`.
  - January is GMT = UTC, but the code does not rely on that. Tests prove that a summer gate closure is 08:50 UTC and that clock-change days have 23 or 25 hours.
- **Detail:** timestamps are fixed to nanosecond resolution, because pandas refuses to join datetime columns with different units.
- **Detail:** Windows has no built-in time-zone database, hence the `tzdata` dependency.

### `services/elexon_client.py`: the only code that touches the network

- **Why:** isolating external input/output keeps the rest of the code pure and testable, and makes an Elexon API change a one-file change.
- **How:**
  - `httpx` with a timeout.
  - Requests are split into **7-day chunks**, because the stream endpoints are not paginated.
  - **Retries with exponential back-off** on timeouts, HTTP 429 and 5xx.
  - 4xx errors are not retried, because retrying a bad request cannot help.
  - Everything surfaces as a single `ElexonAPIError` type.
  - The client returns raw records only and never cleans them.
- **Tested with:** `httpx.MockTransport`, so the tests run without a network.

### `services/data_processor.py`: validation and cleaning, with an audit trail

- **Why:** downstream code must be able to trust its input. Silent data loss is worse than visible data loss.
- **How:** every rule returns a count in a `CleaningReport`, which is exposed at `/api/health`.

| Rule | Actuals | Forecasts | Reason |
| --- | --- | --- | --- |
| Keep `fuelType == WIND` | ✓ | | Scope |
| Unparsable timestamp or value → drop and count | ✓ | ✓ | Never guess |
| Duplicate → keep the latest publication | ✓ | ✓ | Corrections supersede earlier values |
| Target already started at publish time → drop | | ✓ | That value was not a forecast (20.6% of rows) |
| Add `horizon_hours = startTime − publishTime` | | ✓ | Needed for the lead-time analysis |

- **Missing values stay missing.** Filling a gap with 0 would look like "no wind" and distort every metric.

### `services/data_store.py`: fetch once, serve from memory

- **Why:**
  - Speed: responses take milliseconds instead of seconds.
  - Resilience: a live demo does not depend on Elexon being up.
  - Reproducibility: the snapshot is committed, so every reviewer gets the same numbers.
- **Why not a database:** the data is about 17k clean rows (1.2 MB). Postgres or SQLite would add set-up steps for every reviewer and gain nothing. The store's interface (`actuals`, `forecasts`) would stay the same if one were added later.
- **How:** `load_store()` reads the CSVs in `data/`, or downloads and saves them if they are missing or `WFM_DATA_SOURCE=live`. It then cleans them and records provenance: the source, when the data was fetched and the cleaning reports.

### `services/forecast_evaluator.py`: the business logic

**`SelectionPolicy`** captures the decision rule as data: `day_ahead` + `gate_closure`, or `fixed_horizon` + `horizon_hours`. Because the rule is an object passed into functions, the same code serves both policies, and adding a new policy changes only this class.

**`select_forecasts()`** implements requirement R1 in a single vectorised operation:

```python
pd.merge_asof(
    targets.sort_values("cutoff"),         # one row per target hour, with its cutoff
    forecasts.sort_values("publishTime"),  # every forecast value of every run
    left_on="cutoff", right_on="publishTime",
    by="startTime",            # only forecasts for the same target hour
    direction="backward",      # publishTime <= cutoff ... and the latest such one
)
```

Worked example from the real data: target hour **15 Jan 10:00 UTC**, 09:50 gate closure, so the cutoff is 14 Jan 09:50. Twenty runs forecast this hour. Some of them:

| Run published | Forecast (MW) | Eligible? (≤ 14 Jan 09:50) |
| --- | --- | --- |
| 13 Jan 02:30 (earliest) | 12,672 | yes, but older |
| 14 Jan 07:30 | 12,081 | yes, but older |
| **14 Jan 09:30** | **12,193** | **yes, the latest → chosen (24.5 h ahead)** |
| 14 Jan 11:30 | 12,362 | no, published after the decision |
| 14 Jan 15:30 | 13,340 | no |
| 15 Jan 09:30 (last) | 12,176 | no |

The actual output was 9,807 MW, so the decision-time error is **+2,386 MW**: the desk would have planned on about 2.4 GW of wind that did not arrive.

**Why `merge_asof`** rather than a Python loop or a "latest run before cutoff" lookup:

- It works per target hour. If a run were partial or missing, the rule still picks the best available forecast for each hour.
- It is vectorised: 744 hours are processed in milliseconds.
- It states the business rule exactly as an "as-of" join.

**`evaluate()`** builds a target grid of every hour of the UK delivery days, selects the forecasts, attaches the actuals and computes `error = forecast − actual`. The grid is generated from the calendar, not from the data, so a missing forecast or actual shows up as a counted gap instead of disappearing.

**`compute_metrics()`** produces the headline numbers:

| Metric | Formula | Why it is there |
| --- | --- | --- |
| MAE | mean(\|e\|) | The typical miss in MW, easy to explain to a trader |
| RMSE | √mean(e²) | Penalises big misses, which are the costly ones |
| Bias | mean(e) | Shows whether there is a *systematic* over- or under-forecast (there is: +1.7 GW) |
| P10 / P50 / P90 of e | Percentiles | "In 80% of hours the error was between P10 and P90": the uncertainty band to plan around |
| nMAE | MAE / mean actual | Makes periods comparable |
| Over-forecast % | share of e > 0 | How often the buyer would be short |
| Coverage, missing counts | | Shows how much of the period the metrics are actually based on |
| Worst hour | max \|e\| and its time | Shows the tail risk |

MAPE is **deliberately excluded**: it explodes when actual wind is near zero.

**`breakdown()`** groups the errors by UK clock hour, delivery date or forecast size. **`horizon_profile()`** re-runs the selection with `fixed_horizon` for h = 0…48. **`day_ahead_view()`** assembles the decision screen: the cutoff, the run used and how fresh it was, the expected day profile, hindsight accuracy, and every run for the day, flagged as available or not available at the cutoff.

### `models/schemas.py`: the API contract

- **Why:**
  - The React app can rely on exact field names and types.
  - FastAPI generates interactive docs at `/docs` from these models.
  - Units and conventions are explicit in the names (`_mw`, UTC datetimes ending in `Z`).
- **How:** small `from_frame` / `from_dict` builders convert pandas NaN/NaT to JSON `null`. A raw NaN would produce invalid JSON.

### `api/dependencies.py` and `api/routes/*`: thin HTTP layer

- **Why thin:** the business logic must be testable without HTTP and reusable from scripts or notebooks. Routes only validate, call a service and shape the response.
- **Validation examples:**
  - `gate_closure` must match `HH:MM`.
  - Dates must lie inside the loaded window, and `start ≤ end`.
  - `horizon_hours` must be between 0 and 48.
  - Any violation returns HTTP 422 with a readable message.

### `main.py`: app factory

- **Why a factory:** production calls `create_app()` with no arguments. Tests call `create_app(settings, store)` with synthetic data, so they run in milliseconds with no files or network.
- **Graceful start-up:** if the data cannot be loaded (no snapshot and Elexon unreachable), the server still starts. `/api/health` returns 503 with the reason, and the data endpoints return 503 instead of crashing.

---

## 6. API overview

| Endpoint | Answers |
| --- | --- |
| `GET /api/health` | Is it up? Which data is it serving? What was cleaned away? |
| `GET /api/forecasts` | Every forecast record for a period, optionally only those published before a moment ("what was known then") |
| `GET /api/forecasts/day-ahead?delivery_date=` | **Q1**: the decision view for one delivery day |
| `GET /api/analytics/summary` | **Q2**: headline metrics and coverage |
| `GET /api/analytics/timeseries` | Half-hourly actual, forecast and error, for charts |
| `GET /api/analytics/breakdown?by=` | Error by `hour_of_day`, `delivery_date` or `forecast_level` |
| `GET /api/analytics/horizon` | Error as a function of lead time, 0 to 48 h |

All analytics endpoints share `start`, `end`, `policy`, `gate_closure` and `horizon_hours`. The full reference is in [backend/README.md](../backend/README.md) and the live docs at `/docs`.

---

## 7. Reliability: what happens when things go wrong

| Situation | Behaviour |
| --- | --- |
| Elexon is slow or returns 5xx/429 | Up to 3 attempts with back-off (1 s, 2 s), then a clear `ElexonAPIError` |
| Elexon is unreachable at start-up and there is no snapshot | The server still starts; `/api/health` returns 503 with the reason |
| The API contract changes (missing fields) | Loading fails loudly, naming the missing fields |
| Bad or duplicate rows | Dropped, counted and shown in `/api/health` |
| No forecast before the cutoff, or no actual for an hour | Counted as `missing_forecast` / `missing_actual`, excluded from the metrics, never treated as zero |
| Invalid query (date outside the window, bad time format) | HTTP 422 with a readable message |
| Browser on another origin | CORS allow-list from configuration |

---

## 8. Testing strategy (125 tests, about 30 s)

| File | What it proves |
| --- | --- |
| `test_timestamps.py` | GMT vs BST cutoffs, 23/25-hour clock-change days, garbage timestamps become NaT |
| `test_data_processor.py` | Each cleaning rule, the latest-publication-wins rule, loud failure on missing columns |
| `test_forecast_evaluator.py` | **The no-look-ahead rule.** A 09:50 cutoff picks the 09:30 run and never the 11:30 one; 09:20 falls back to 07:30; a run published exactly at the cutoff counts; no eligible run means missing, not zero; metric formulas are checked against hand calculations |
| `test_elexon_client.py` | Retries on 503, no retry on 400, giving up after N attempts, chunk boundaries, response formats |
| `test_api.py` | Every endpoint end-to-end, every validation error returns 422 (including impossible dates and malformed timestamps), a failed data load returns 503, and the OpenAPI schema behind `/docs` lists every endpoint |
| `test_real_snapshot.py` | Regression checks on the real January data: 744/744 hours scored; **no forecast used was published after its cutoff**; the 09:30 D-1 run is always the one chosen |
| `test_feature_builder.py` | **ML leakage tests.** Overwriting every forecast or actual published after the cutoff leaves every feature unchanged, in GMT and on the BST clock-change day (25 hours). On that day the 09:30 UTC run, published at 10:30 BST, is correctly *not* used. Every row of the real ML dataset carries the 09:50 UK cutoff, including across the clock change. Feature values are checked on the hand-made scenario |
| `test_ml_model.py` | Scoring formulas, the day-block bootstrap, calibration bands, NaN handling for every candidate, model save/load |
| `test_ml_api.py` | Both prediction components and their default and comparison models. Logistic is the default shortfall classifier, and the Random Forest is returned only when explicitly requested; both give valid probabilities equal to their evaluated predictions, and the Random Forest's artifact and results are kept. A model of one task is rejected for the other; the 503 when no models are present. **Start-up guards:** the service refuses features built for another cutoff or from a later forecast, any model that reads a label, and an incomplete feature table. A missing, empty or unreadable model file returns 503 on `/api/ml` only, naming the file. **Requests never train or reload a model:** `fit` and `joblib.load` are trapped during every ML request. **The API serves exactly the predictions that were evaluated**, from the artifacts sealed for the February holdout |
| `test_scope.py` | **Assignment scope.** The default window is January 2026 decisions. Every served prediction, for each model on each ML endpoint, is for a decision made 1–31 January 2026, with delivery the next day. The forecast views reject deliveries decided outside January |
| `test_holdout.py` | February holdout: the day-block bootstrap and Diebold-Mariano test, tamper detection on the sealed predictions, no label columns in the sealed file, the cutoff audit |

The synthetic scenario in `tests/conftest.py` uses three runs with values of 11, 12 and 13 GW against a flat actual of 10 GW. Every expected number can be checked by hand.

---

## 9. Trade-offs and alternatives considered

| Decision | Alternative | Why this choice |
| --- | --- | --- |
| Per-target-hour selection (`merge_asof`) | Pick one whole run per day | The same result when runs are complete, and more robust when they are not |
| Score hourly (exact `startTime` join) | Interpolate forecasts to half-hours, or average the actuals to hours | Follows the brief's definition of the shared key; the alternative changes MAE by under 1% |
| Committed CSV snapshot + memory | Database; fetching live on every request | Small data, reproducible, offline, zero set-up |
| Cutoff as a parameter (default 09:50) | Hard-coded | Two GB auctions close at different times; the client may use either |
| Synchronous pandas in FastAPI | Async everything | Requests are CPU-bound and take a few milliseconds; async would add complexity with no gain |
| Keep missing data as `null` | Fill with 0 or interpolate | Filling would invent data and distort the metrics |
| Exact version pins | Version ranges | The same behaviour on every machine; Docker will pin the Python version too |

**Performance:** start-up takes about 50 ms from the snapshot. The full-month summary takes about 30 ms, the 0 to 48 h horizon profile about 80 ms, and the day-ahead view about 40 ms.

---

## 10. The ML extension (summary)

The full write-up is in [backend/ml/README.md](../backend/ml/README.md). In short:

- **Data:** four months of raw files (Oct 2025 – Jan 2026) are validated, combined and turned into one leakage-free table, with one row per delivery hour. The table is built by `feature_builder.py` on top of the same `latest_forecast_as_of` rule the API uses.
- **Split and selection:** the split is chronological by decision month (train Oct–Nov, validation Dec, test Jan). Simple baselines compete with Random Forest and gradient boosting, and selection happens on validation only.
- **Result:** the corrected forecast cuts January MAE by 31% (Random Forest, selected on validation). The simplest linear correction by forecast size did even better (−41%). The shortfall classifier ranks risk reasonably but is only modestly better than the base rate.
- **February 2026 holdout:** a fresh re-test with predictions sealed before any label was read. The linear correction was the only model clearly better than the official forecast (−36% MAE). The logistic model gave the better-calibrated shortfall probabilities.
- **Serving:** `prediction_service.py` loads two separate components at start-up, and each has its own models, inputs, outputs and scores:
  - **Corrected forecast:** the linear correction by default, the Random Forest for comparison.
  - **Shortfall risk:** the logistic model by default, the Random Forest for comparison.
- **Why two components:** one model never does both jobs.
- **Defaults:** the defaults are a setting. They were approved for the current application on 2026-10-26 and stay subject to the start-up safeguards. The approval is a selection on two winter months, not proof of long-term superiority or of trading profit.
- **Selection history:** the rule fixed before January chose the Random Forests. The defaults were chosen later, using the January test and the offline February check, so the choice reflects post-January evidence; the inputs and weights do not. `/api/ml/models` discloses this (`selected_by_pre_january_rule`).
- **Scope:** the application serves January 2026 decisions only and never loads February data or results. `tests/test_scope.py` enforces this.
- **Start-up checks:** the service refuses features not built for the 09:50 cutoff and any model that reads a label. Tests prove the served predictions equal the evaluated ones.

## 11. What comes next

1. **React frontend**, consuming the endpoints above.
2. **Docker** and deployment.

---

## 12. Questions a reviewer is likely to ask

**Why not just take the newest forecast for each hour?**
Because the newest forecast didn't exist when the purchase was made. Using it is look-ahead bias, and it would flatter the forecast's accuracy.

**How do you guarantee there is no look-ahead?**
The rule is a single `merge_asof(direction="backward")` on `publishTime ≤ cutoff`. It is tested on synthetic data, and a real-data regression test asserts `publishTime ≤ cutoff` for all 744 hours.

**Why is the default cutoff 09:50?**
It is the gate closure of Nord Pool's GB day-ahead auction. EPEX's GB auction closes around 09:20, and `gate_closure` lets the client pick. In January 2026 this switches the run used from 09:30 to 07:30 and changes MAE by only about 26 MW.

**WINDFOR is hourly but actuals are half-hourly. What did you do?**
We score per hour using the shared `startTime`, as the brief defines it. Averaging the two half-hours instead changes MAE by under 1%, which was checked.

**Why remove 20% of the forecast rows?**
Every WINDFOR run starts at 21:00 the previous day, so it repeats hours that had already started when it was published. Those values were never forecasts. Each removal is counted and visible in `/api/health`.

**What did the data tell you?**
The decision-time forecast over-predicted by +1.7 GW on average (MAE 2.0 GW, 84% of hours too high). The over-prediction grows to +2.75 GW when more than 15 GW is forecast, and it persists even 1–2 hours ahead. That points to a systematic cause, likely curtailment of wind on high-output days, rather than weather uncertainty. For the buyer it means that trusting the raw forecast leads to systematic under-buying.

**Why no database?**
The data is 1.2 MB. A database would add set-up for every reviewer and no capability. The data store is the single place that would change if one were added.

**What if Elexon is down during the demo?**
The app serves the committed snapshot. If neither is available, it still starts and explains the problem at `/api/health`.

**How would this scale to years of data or live operation?**
Swap the CSV snapshot for a database (e.g. PostgreSQL/TimescaleDB) behind `data_store`, and schedule `scripts/fetch_data.py` to append new runs every 2–4 hours. The evaluator and API would not change.
