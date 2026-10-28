# ML v2: offline experiments

This folder holds **offline** experiments that try to improve the corrected forecast (MAE) and the shortfall-risk model (ROC-AUC). Nothing here is loaded by the January application:

- its API, endpoints, schemas and date limits are unchanged
- its approved defaults (linear correction, logistic regression) and their saved files are unchanged
- the sealed February files are unchanged

A v2 model can only be proposed for the application after a significant result on the supplementary evaluation, and only with the owner's explicit approval.

## Scope and data

| Period | Purchase decisions | Use |
| --- | --- | --- |
| Development | 1 Oct 2024 – 31 Dec 2025 | Cross-validation, model choice, final fit. All of it was published before the first January decision |
| January 2026 | 1–31 Jan 2026 | **Context only.** It was evaluated before and informed the production defaults, so it is never used to choose |
| February 2026 | – | **Not used.** It already influenced the production defaults |
| Supplementary | 1 Mar – 31 May 2026 | One-time evaluation under the locked protocol. **Not run yet;** its data has not been downloaded |

The sources are Elexon WINDFOR (wind forecasts), FUELHH (wind actuals) and NDFD (daily national demand forecast). Every record keeps its publishTime. Oct 2025 – Jan 2026 wind data is read from the project's `DATASETS*` folders; only the older history and NDFD are downloaded into `data/raw/`.

These sources were checked and rejected:

- **DGWS (day-ahead wind):** published at 16:45 on D-1, after the 09:50 cutoff, and produced by the same operator (NESO) as WINDFOR.
- **NDF (half-hourly demand):** at the cutoff it covers only 00:00–04:30 of day D.

## Pre-registration

[`protocol.json`](protocol.json) fixes the following before any v2 data was downloaded or any model fitted:

- the candidates and features
- the folds and metrics
- the selection rules
- the supplementary procedure

`lock_protocol.py` recorded its SHA-256 in `reports/protocol_lock.json`. Every later step refuses to run if the file has changed.

## Run order (from `backend/`)

```bash
python -m ml.v2.lock_protocol     # done once, before anything else
python -m ml.v2.fetch_history     # pre-January history + NDFD  -> data/raw/
python -m ml.v2.build_dataset     # validate, build features    -> processed/, reports/data_validation.json
python -m ml.v2.experiment        # CV, selection, freeze       -> reports/cv_results.json, models/
pytest ml/v2/tests                # leakage, protocol and fold tests (separate from the production suite)
```

## Files

| File | Content |
| --- | --- |
| `common.py` | Paths, data windows, fixed constants, protocol lock |
| `features.py` | The 9 production features (imported unchanged) plus 14 new ones, each from data published by 09:50 UK time on D-1 |
| `candidates.py` | The pre-registered candidates: 11 regression, 9 classification |
| `experiment.py` | Rolling-origin CV (9 monthly folds, Apr–Dec 2025), selection, final refit, January context |
| `tests/test_v2.py` | Leakage perturbation tests (GMT and the BST clock-change day), known values, protocol lock, fold timing, data separation |
