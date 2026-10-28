# Frontend: Wind Forecast Intelligence dashboard

A React dashboard for a UK utility's day-ahead purchasing team. It covers purchase decisions made **1–31 January 2026**, with the **09:50 UK decision cutoff** on the day before delivery.

The dashboard shows only what the FastAPI backend returns. Nothing is hard-coded, simulated or invented: no predictions, actual generation, accuracy figures or probabilities.

## What it shows

| Section | Content | Source |
| --- | --- | --- |
| Header | Decision period, cutoff, time zone | `GET /api/health` |
| Filter row | Decision-date selector, limited to the backend's decision window; model selectors (approved default or comparison) | `GET /api/health`, `GET /api/ml/models` |
| Overview | Official forecast, corrected forecast, forecast error (hindsight) and peak shortfall risk for the delivery day. A decision timeline shows the run used, the 09:50 cutoff and the runs that came too late | `GET /api/ml/predictions`, `GET /api/forecasts/day-ahead` |
| Forecast | Hourly official vs corrected forecast vs actual generation, with a chart/table toggle | `GET /api/ml/predictions` |
| Shortfall risk | The model's hourly probability that actual wind comes in ≥ 1 GW below the forecast; hours where a shortfall occurred are marked (hindsight) | `GET /api/ml/predictions` |
| Accuracy | Daily error across January (click a day to select it), whole-period scores, official forecast accuracy, and the offline model evaluation (MAE, RMSE, bias, ROC-AUC, PR-AUC, Brier, precision, recall, F1) with a plain-English key | `GET /api/ml/predictions`, `GET /api/analytics/summary`, `GET /api/ml/models` |
| Models | Approved defaults and comparison models in plain language: what each does, what it uses, its training window and how it was chosen, plus the backend's selection-history note | `GET /api/ml/models` |
| Data & method | Provenance, the decision rule, definitions, the evaluation metrics explained, and the limitations | `GET /api/health` |

**Labelling rules:**

- Actual generation, errors and "shortfall occurred" are marked **hindsight**: they were known only after delivery.
- Model evaluation metrics are labelled as **offline evaluation**, separate from the selected day's predictions.
- No trading recommendations or financial claims are made.

**Technical details:** implementation detail is kept out of the business view, in collapsed **Technical details** sections. Nothing is removed:

- **Each model card:** model ID, task, the backend's specification, raw input column names, saved model file and SHA-256 fingerprint check.
- **Data & method:** API address, data mode, service start time, delivery window, row counts, the data-cleaning report, the model-service status and the endpoints used.
- **Each error message:** the backend's reason, HTTP status and request URL.
  - Absolute file paths in a reason are cut to the file name, and developer commands are replaced by a pointer to the README.
  - No credentials, secrets or stack traces are ever shown.

## Requirements

- **Node.js 20.19 or newer** (tested with 20.19.0) and npm 10.
- **The backend running locally.** See [../backend/README.md](../backend/README.md). By default the dashboard calls `http://127.0.0.1:8000`.

## Install and run

```powershell
# Terminal 1 - backend (from the project root)
cd "E:\GridBricks Job\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# Terminal 2 - frontend
cd "E:\GridBricks Job\frontend"
npm install
npm run dev
```

Then open **<http://localhost:5173>**.

On macOS or Linux, use the same commands with `source .venv/bin/activate` and `uvicorn app.main:app --reload` for the backend.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend base URL. Copy `.env.example` to `.env.local` to change it, then restart `npm run dev` |

- Every `VITE_` variable is compiled into the public bundle, so never put secrets here. None are needed.
- **CORS:** the backend accepts browser requests from `http://localhost:5173` and `http://127.0.0.1:5173` (its `WFM_CORS_ORIGINS` setting). The dev server is pinned to port 5173 (`strictPort`). If that port is busy, it stops with an error instead of moving to a port the backend would reject.

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Development server on <http://localhost:5173> |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the production build locally (on port 4173; add that origin to the backend's `WFM_CORS_ORIGINS` first) |
| `npm run lint` | ESLint |
| `npm test` | Unit, component and app tests with a mocked network that replays captured real backend responses (70 tests) |
| `npm run test:live` | The same contract checks against the **running** backend. PowerShell: `$env:LIVE_API_URL="http://127.0.0.1:8000"; npm run test:live`. Bash: `LIVE_API_URL=http://127.0.0.1:8000 npm run test:live` |

**What the tests cover:**

- dates, including UK summer time and the cutoff;
- summaries, which match the backend's own day MAE (including its 0.1 MW rounding);
- API error handling for 422, 503, network/CORS failures and non-JSON responses;
- each component, including the scored-hour count when some hours have no actual;
- the full app: decision-to-delivery mapping, out-of-window dates, comparison-model switching, backend unreachable, backend unavailable (503), the pending health check, the model selectors (loading, loaded with the approved defaults, failed with retry), and ML endpoints unavailable while the forecast views keep working;
- no mixing of days or stale values: the overview tiles never combine two delivery days while one response is still loading, and tables are dimmed and marked busy while a new date or model loads;
- no developer detail in the business view: the API address, raw input names and model file paths sit under Technical details, and error reasons are shown without file paths or commands.

## States you may see

| State | Meaning |
| --- | --- |
| "Checking backend…" | The health check (`GET /api/health`) has not answered yet. Nothing else is requested until it does, because it supplies the decision window |
| Grey placeholder blocks | Loading (first load only; later loads dim the previous chart or table and mark it busy instead) |
| Red banner "cannot reach the backend", sidebar "Backend unreachable" | Backend not running, wrong `VITE_API_BASE_URL`, or CORS blocked. The banner's Technical details show the address tried and the settings to check; start-up steps are in the project README |
| Banner "The backend reported a problem.", sidebar "Backend unavailable" | The backend answered the health check with an error, for example 503 when its data could not load. Its reason is under Technical details |
| "Service unavailable" in a section | The backend answered 503, for example a missing model file. Its reason is under Technical details. Other sections keep working |
| Model selectors show "Loading models…" | The model list (`GET /api/ml/models`) is still loading |
| Model selectors show "Models unavailable" | The model list could not be loaded, or the health check failed. No model is offered or guessed. Predictions name no model, so the backend applies its approved defaults. "Try again" beside the selectors reloads the list |
| Date message in the filter row | The date is outside 1–31 January 2026; nothing is requested |

**Rounding:** the backend computes every score from unrounded values and returns it rounded to 0.1 MW. Its hourly values are also rounded to 0.1 MW. The daily-error chart and table compute each day's MAE in the browser from those hourly values and round it to 0.1 MW as well. All figures are then shown in whole MW. Because the hourly inputs are already rounded, a browser day value can differ from the backend's own day score by up to 0.1 MW. For all 31 served days and both correction models, checked against the live backend, the whole-MW figures in the month view match the day tiles exactly. The whole-period scores and the day tiles always come from the backend.

## Structure

```text
frontend/
├── index.html, vite.config.js, eslint.config.js, package.json, .env.example
└── src/
    ├── main.jsx, App.jsx          # page layout and data loading
    ├── api/client.js              # the five backend routes; errors -> ApiError(kind)
    ├── hooks/useApi.js            # load / abort / retry; keeps old data while refreshing
    ├── lib/                       # dates (UK time), formatting, summaries, model names, chart palette
    ├── components/                # layout, filters, overview, charts, accuracy, models, method, states
    ├── styles/app.css             # light theme, responsive layout
    └── test/                      # setup + fixtures captured from the real backend
```

**Design:** a light, restrained theme. Chart colours have one meaning each, and the palette was checked for colour-blind separation and contrast. Every chart has a legend, a crosshair tooltip and a table view. The layout adapts to desktop (sidebar), tablet (top navigation) and phone (single column), in portrait or landscape.

## Limitations

- The charts were checked by automated tests and a live-backend render in jsdom. They still need a visual review in a real browser.
- Recharts 2.x and ESLint 9.25 are pinned, known-good versions, but npm flags them as no longer actively maintained. Upgrading means moving to Recharts 3.
- Light theme only, as requested.
- Deployment is not set up yet.
