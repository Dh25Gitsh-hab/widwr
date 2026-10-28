# Wind Forecast Intelligence

**Decision support for a UK utility's day-ahead power-buying desk.**

The desk buys power on day D-1 for delivery on day D. What it needs to buy depends on how much wind the grid will produce, which is uncertain. This product answers two questions:

1. **What wind forecast did we actually have when we had to buy?** It rebuilds the latest forecast published before the day-ahead auction gate closure, with no hindsight.
2. **How far can we trust it?** It measures that forecast against actual generation: average error, systematic bias, the uncertainty band, and when the forecast is weakest.

The data is Elexon BMRS national wind **forecasts (WINDFOR)** and **actual generation (FUELHH)** for purchase decisions made in **January 2026**.

> **Status:** the backend (Python / FastAPI), the ML extension and the React dashboard are built and tested locally. Deployment has not started.

---

## Scope

The brief asks us to "consider data relevant only for purchase decisions to be made in January 2026". A decision on day D-1 buys power for delivery day D, so January decisions (1–31 Jan) cover delivery days 2 Jan – 1 Feb 2026.

| Part | Data it uses | January-only? |
| --- | --- | --- |
| **Core application**: the forecast available at decision time, and its accuracy | January purchase decisions only. The snapshot also holds the forecasts published on 30–31 Dec, which the lead-time analysis needs | **Yes.** Every endpoint rejects other dates, and tests enforce it |
| **ML predictions** served by the API | Features for the January decisions, built only from data published by each 09:50 cutoff. Models trained on Oct–Dec 2025, all published before the first January decision | **Yes,** for inputs and model weights |
| **ML model choice** (which model is the default) | January test results and an offline February check | **Disclosed deviation**: see below |
| **February 2026 data and results** | An offline evaluation in `backend/ml/holdout_feb2026/`. The application never loads February data or results | Outside scope; kept as supplementary evidence |

### Model-selection history

1. **Before January:** the models were trained on Oct–Nov 2025 and compared on December. The rule fixed in advance (lowest December error) chose the **Random Forest** for both tasks.
2. **January test:** the simpler linear correction scored better on January (MAE 1.18 vs 1.38 GW). From then on, January results had shaped the preference.
3. **February 2026 offline check:** to test that preference on unseen data, February predictions were sealed before any February actual was read. The linear correction was the only model clearly better than the official forecast, and the logistic model was the better-calibrated shortfall model.
4. **Approval (2026-10-26):** the linear correction and the logistic model became the defaults. Both Random Forests remain as comparison models.

So the **choice** of defaults used evidence from after the January decisions; the prediction inputs and model weights did not. `GET /api/ml/models` says so and marks the models the pre-January rule chose (`selected_by_pre_january_rule`). The defaults are a selection for the current application, not proof of long-term superiority or of trading profit.

---

## Headline finding (January 2026, 09:50 gate closure)

| | |
| --- | --- |
| Typical error (MAE) | **2.0 GW**, 17.7% of average wind output |
| Systematic bias | **+1.7 GW**: the forecast was too high in **84% of hours** |
| On windy days (forecast > 15 GW) | Bias rises to **+2.75 GW** |
| 80% uncertainty band | Actual wind was between 4.2 GW below and 0.5 GW above the forecast |

Taking the forecast at face value would have meant systematically **buying too little** power. Details are in [docs/DATA_FINDINGS.md](docs/DATA_FINDINGS.md).

### ML extension (trained on Oct–Dec 2025, tested on January 2026)

- **A corrected forecast** cuts the January error substantially. The model chosen by the pre-set rule (Random Forest) reduces MAE from 2.0 to 1.4 GW (−31%, 95% CI 0.2–1.1 GW).
- **The simplest correction did best on January.** A linear correction by forecast size reached 1.2 GW (−41%).
- **Re-tested offline on February 2026** (outside the January scope; supplementary). Predictions were sealed before any February actual was read:
  - The **linear correction cut MAE by 36%**, a statistically clear gain (95% CI 290–955 MW).
  - The Random Forest's −23% gain was borderline.
  - Random Forest vs linear remains inconclusive.
- **The shortfall-risk model** ranks risky hours better than chance (ROC-AUC about 0.69–0.76), but it is only modestly better than "a shortfall is likely". The logistic version is the better-calibrated one.
- **What the API serves:** two separate components, each with its own model.
  - **Corrected forecast:** the linear correction.
  - **Shortfall risk:** the logistic model.
  - These defaults were approved for the current application on 2026-10-26. The Random Forests stay available as comparison models.
  - The choice used January and February evidence (see [Model-selection history](#model-selection-history)). It is not proof of long-term superiority.

The method, honest results and limitations are in [backend/ml/README.md](backend/ml/README.md) and [backend/ml/holdout_feb2026/README.md](backend/ml/holdout_feb2026/README.md). No trading-profit claims are made.

---

## Quick start

**Requirements:** Python 3.12+. No database or API key is needed. The data snapshot is included, so the app works offline.

```bash
cd backend
python -m venv .venv
# Windows:      .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open **<http://127.0.0.1:8000/docs>** to explore the API, or run `pytest` to run the 125 tests. Every local URL is listed in [backend/README.md](backend/README.md#quick-start).

Then, in a second terminal, start the dashboard (Node.js 20.19+):

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**. Configuration, tests and details are in [frontend/README.md](frontend/README.md).

Full instructions, configuration and the API reference are in [backend/README.md](backend/README.md).

---

## Repository layout

```text
.
├── backend/          Python FastAPI service: data pipeline, evaluation, REST API, tests
│   └── ml/           offline ML workflow: validate → build dataset → train → evaluate (+ outputs)
│       └── v2/       offline improvement experiments; not used by the application
│                     (results: backend/ml/v2/existing_data/README.md, backend/ml/v2/ridge_gbm/README.md)
├── DATASETS/         raw Elexon FUELHH wind actuals, Oct 2025 – Feb 2026 (read-only; Feb = holdout only)
├── DATASETS1/        raw Elexon WINDFOR wind forecasts, Oct 2025 – Feb 2026 (read-only; Feb = holdout only)
├── frontend/         React + Vite dashboard: forecasts, shortfall risk, accuracy, models (reads the API only)
├── docs/
│   ├── BACKEND_DESIGN.md   why and how the backend is built (architecture, decisions, trade-offs)
│   └── DATA_FINDINGS.md    data profiling, processing rules and January 2026 results
└── README.md
```

## How it works (short version)

```text
Elexon API ──► elexon_client ──► CSV snapshot ──► data_processor ──► in-memory store
                                                    (clean, count)        │
                                    forecast_evaluator ◄──────────────────┘
                     (latest forecast published ≤ cutoff, per hour → error metrics)
                                          │
                               FastAPI REST API ──► React dashboard
```

The key rule: for each delivery hour, only forecasts **published at or before the decision cutoff** are eligible, and the latest of those is used. Using a later forecast would be look-ahead bias. The cutoff defaults to **09:50 UK time on D-1** (Nord Pool GB day-ahead gate closure) and can be changed per request.

## Use of AI tools

As the brief allows, AI tools were used to build this project. This section states that explicitly.

- **Claude (Anthropic), used through Claude Code**, throughout the build and under my direction and review:
  - exploring and validating the Elexon data files;
  - the FastAPI backend: data pipeline, forecast evaluation and REST API;
  - the ML pipeline, the February 2026 holdout check and the offline experiments in `backend/ml/v2/`;
  - the React dashboard;
  - the backend and frontend tests, the integration audits and the documentation.
- **ChatGPT (OpenAI)**: early domain research, and drafting the project blueprint and structure.

All design decisions, data findings and code were reviewed, run and checked by me. Every number in the documentation was produced by the code in this repository.
