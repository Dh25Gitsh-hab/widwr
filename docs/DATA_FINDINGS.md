# Data findings: what the Elexon data actually looks like

This is the result of step 02 of the blueprint ("API exploration and data profiling"), done before writing any pipeline code.
Every rule in the backend traces back to a fact on this page.

All figures come from the committed snapshot in `backend/data/`, downloaded from the Elexon Insights API on 25 Oct 2026.
Times are UTC. In January, UK time equals UTC.

---

## 1. Scope

| Item | Value | Why |
| --- | --- | --- |
| Purchase decisions | 1 to 31 Jan 2026 | The brief: "data relevant only for purchase decisions to be made in January 2026" |
| Delivery days scored | 2 Jan to 1 Feb 2026 (31 days, 744 hours) | A day-ahead decision on D-1 buys for day D |
| Decision cutoff (default) | **09:50 UK time on D-1** | Gate closure of Nord Pool's GB day-ahead auction. EPEX's GB auction closes around 09:20 UK time; both are selectable in the API |
| Actuals fetched | Settlement dates 1 Jan to 1 Feb 2026 | Delivery window plus one day of context |
| Forecasts fetched | Runs published 30 Dec 2025 to 1 Feb 2026 | Earliest run that can reach 2 Jan, through the last delivery day |

## 2. Data dictionary

### FUELHH: actual generation (`/datasets/FUELHH/stream?fuelType=WIND`)

| Field | Meaning | Used for |
| --- | --- | --- |
| `startTime` | Start of the 30-minute settlement period (UTC) | Join key |
| `generation` | Average wind output in the period, MW | The "actual" |
| `publishTime` | When Elexon published the value (30 to 145 min after `startTime`) | Choosing between duplicates |
| `settlementDate`, `settlementPeriod` | Market calendar (period 1 to 48) | Not needed: `startTime` is unambiguous |
| `fuelType` | Generation category | Filter to `WIND` |

### WINDFOR: forecast generation (`/datasets/WINDFOR/stream`)

| Field | Meaning | Used for |
| --- | --- | --- |
| `publishTime` | When this forecast run was published (UTC) | **Deciding what was known at decision time** |
| `startTime` | Target hour being forecast (UTC) | Join key |
| `generation` | Forecast wind output, MW | The "forecast" |

## 3. Profiling facts, and the rule each one led to

| # | Observation | Consequence in the backend |
| --- | --- | --- |
| F1 | **WINDFOR is hourly**: every `startTime` is on the hour. FUELHH is half-hourly. | Accuracy is scored per hour. Each hourly forecast is compared with the actual for the same `startTime`. Comparing with the mean of the two half-hours instead changes MAE by under 1% (2,013 vs 1,996 MW), so the choice does not change any conclusion. Charts still show every half-hour of actuals. |
| F2 | **There are 8 runs per day**, published at 02:30, 04:30, 07:30, 09:30, 11:30, 15:30, 18:30 and 22:30. One run in the window was published at 12:25 instead of 11:30 (15 Jan). | No hard-coded schedule. The selection rule uses whatever `publishTime` values exist. |
| F3 | **Each run has 73 hourly values on a fixed window**, from 21:00 the day before publication to 21:00 two days after, whatever the publish time. So every run contains 6 to 26 hourly values (6 for the 02:30 run, 26 for the 22:30 run) whose hour had **already started** when it was published. | Those values are not forecasts and are removed: 4,081 of 19,856 rows (20.6%). Leaving them in would make accuracy look better than it was. |
| F4 | The furthest horizon is about 66 h. The 09:30 run on D-1 covers all of day D, at 14.5 to 37.5 h ahead. | Forecasts are fetched from 3 days before the first delivery day. |
| F5 | **With a 09:50 cutoff, the 09:30 run on D-1 is used every day. With 09:20, it is the 07:30 run.** The 09:30 run lands 20 minutes before Nord Pool's gate closure. | The cutoff is a parameter (`gate_closure`), not a constant. The UI should show which run was used and how many minutes before the cutoff it was published. |
| F6 | FUELHH has no gaps, duplicates or nulls in the window: 1,536 of 1,536 half-hours. WINDFOR has no nulls or duplicates. | The cleaning code still handles these cases (and is tested on them), because a refreshed snapshot or a different month may not be as clean. |
| F7 | There are no negative or non-numeric generation values. | Values are kept as-is. Any unparsable value is counted and dropped, never replaced with zero. |

## 4. Headline result for January 2026 (day-ahead, 09:50 cutoff)

| Metric | Value |
| --- | --- |
| Hours scored | 744 / 744 (100% coverage) |
| **MAE** | **1,996 MW** (17.7% of mean actual output) |
| RMSE | 2,499 MW |
| **Bias (mean error)** | **+1,728 MW**: the forecast was too high on average |
| Hours over-forecast | 84.3% |
| Error P10 / P50 / P90 | -503 / +1,507 / +4,184 MW |
| Worst hour | 12 Jan 01:00, forecast 6,857 MW too high |
| Mean forecast vs mean actual | 13,014 vs 11,286 MW |

### The forecast systematically over-predicts, and more so on windy days

| Forecast size | Hours | MAE (MW) | Bias (MW) |
| --- | --- | --- | --- |
| 0-5 GW | 31 | 1,257 | +201 |
| 5-10 GW | 210 | 1,427 | +790 |
| 10-15 GW | 206 | 1,573 | +1,439 |
| 15+ GW | 297 | 2,768 | **+2,751** |

### Accuracy degrades only slowly with lead time; the bias does not go away

| Published at least | Mean real lead time | MAE (MW) | Bias (MW) |
| --- | --- | --- | --- |
| 0 h before | 1.6 h | 1,680 | +1,471 |
| 12 h before | 13.6 h | 1,791 | +1,595 |
| 24 h before | 25.6 h | 1,932 | +1,736 |
| 36 h before | 37.6 h | 2,143 | +1,834 |
| 48 h before | 49.4 h | 2,301 | +1,801 |

**What this means**

- Even a forecast made about 1.6 hours ahead is biased by about +1.5 GW. The bias therefore isn't a weather-forecast problem.
- The bias is largest when a lot of wind is forecast. A likely explanation is that the forecast predicts wind farms' available output, while the metered FUELHH figure is lower after **curtailment**: in high-wind periods the system operator pays wind farms to reduce output because of grid constraints.
- This is a hypothesis consistent with the data, not something we verified. Proving it would need curtailment data, which is outside the brief's scope.
- For the buyer the practical message is the same either way: **taking the forecast at face value would systematically under-buy**, and the error is predictable rather than random.

### Sensitivity to the cutoff

| Gate closure | Run used | MAE (MW) | Bias (MW) |
| --- | --- | --- | --- |
| 09:20 (EPEX) | 07:30 D-1 | 2,022 | +1,746 |
| 09:50 (Nord Pool) | 09:30 D-1 | 1,996 | +1,728 |

Waiting for the 09:30 run gains only about 26 MW of MAE. The choice of auction matters much less than the bias does.

## 5. Open points (for the next phase)

- **ML / uncertainty.** The bias depends strongly on forecast size, which suggests a simple, explainable correction should beat the raw forecast. This is to be discussed before the ML phase starts.
- **Curtailment hypothesis.** Worth stating in the pitch as a likely cause, clearly labelled as a hypothesis.
