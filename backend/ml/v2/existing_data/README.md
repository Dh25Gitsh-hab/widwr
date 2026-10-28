# Existing-data experiment (offline)

Can the corrected forecast (MAE) and the shortfall-risk model (ROC-AUC) be improved using **only the Elexon files already in the project**? Those are `DATASETS/` and `DATASETS1/`, Oct 2025 – Feb 2026. No downloads were made.

Nothing here is loaded by the January application. Its API, defaults, saved models and sealed February files are unchanged.

## Method

- **Data:**
  - **Development:** deliveries 2 Oct – 31 Dec 2025 (2,185 hours), the only data used to fit, tune and choose.
  - **January 2026** (744 h) and **February 2026** (648 h) were both examined before. Each was scored **once**, after selection. No untouched test period exists in the existing data.
- **Features:** the 9 production features, plus wind-only features built from data published by 09:50 UK time on D-1. These are recent errors, the 24-hour revision, the day's forecast shape, the calendar and the hour on a circle.
- **Pre-registration:** [`protocol.json`](protocol.json) was locked (`reports/protocol_lock.json`) after the Phase A residual analysis and before any candidate was fitted.
- **Validation:** 8 weekly chronological folds, 6 Nov – 31 Dec 2025. Static models train on deliveries up to 2 days before each fold; rolling ones refit daily on days d−1−W to d−2.
- **Candidates:** 49 regression and 34 classification configurations, across baselines, tuned existing models, nonlinear corrections, objectives that fit MAE directly, drift and rolling corrections, interactions, gradient boosting, ensembles and calibration.

## Results

| | Development CV (pooled, 1,344 h) | January (seen) | February (seen) |
| --- | --- | --- | --- |
| Official forecast MAE | 1,817.5 | 1,995.6 | 1,692.7 |
| Production linear correction MAE | 1,477.8 (spec refitted per fold) | **1,178.6** | **1,082.1** |
| CV-selected Random Forest regressor MAE | **1,157.0** | 1,457.7 | 1,397.6 |
| Production logistic ROC-AUC | 0.781 (spec) | **0.755** | **0.693** |
| CV-selected Random Forest classifier ROC-AUC | **0.836** | 0.723 | 0.690 |

**The protocol's rule rates both candidates "not supported".** They win in development cross-validation but lose to the production defaults on both January and February. The January MAE loss is significant (95% interval −408 to −139 MW). In the nested check, the classifier also fails: it is better than production logistic in only 1 of the 4 late folds.

**Conclusion:** with about 90 days of development data, the ranking of the models flips between months. Tree models win in Nov–Dec 2025, while the simple linear correction wins in Jan–Feb 2026. The existing data cannot support a validated improvement over the approved defaults, so they remain the best-supported choice.

Full results are in `reports/results.json` (every configuration, segments, ensembles, calibration, evaluation), `reports/cv_folds.csv` and `reports/phase_a.json`.

## Run order (from `backend/`)

```bash
python -m ml.v2.existing_data.build_dataset   # datasets from the existing files only
python -m ml.v2.existing_data.phase_a         # baselines and residual analysis
python -m ml.v2.existing_data.run             # CV, selection, checks, freeze, one evaluation
pytest ml/v2/existing_data/tests              # scope, timing, protocol, rolling-leak tests
```
