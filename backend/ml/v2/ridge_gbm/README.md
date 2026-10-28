# Ridge / gradient-boosting experiment (offline)

**Question.** Why can the production linear correction over-correct? And do Ridge regression or gradient boosting, using only features known at the 09:50 UK decision cutoff, improve on it out of sample?

**Scope.** Nothing here is loaded by the application. Its API, frontend, approved defaults (`linear_by_level`, `logistic_by_level`), model files and data are unchanged. The only input is the production feature table `ml/processed/ml_dataset.csv`, read only. It was built from the Elexon files already in the project; no downloads and no other sources were used.

**Answer.**
- Least squares on the forecast size alone gives every day the same correction for the same forecast. That correction is pulled off course by a curved, drifting relationship and by whole-day errors the forecast size cannot see.
- Ridge, gradient boosting and the Random Forest all win in Oct–Dec cross-validation but **lose to the production linear correction in January**. On January's high-forecast days they over-correct **more**, not less.
- **No candidate meets the pre-registered rule. Keep the approved defaults.**

## 1. Why a forecast-only line can over-correct (development data only)

Source: `reports/audit.json` (`python -m ml.v2.ridge_gbm.audit`). Only deliveries from 2 Oct to 31 Dec 2025 were used.

Production: **predicted error = 0.235936 × forecast − 1,373.5 MW**, fitted by ordinary least squares on Oct–Dec 2025. It estimates the *mean* error for each forecast size and applies it to every hour with that forecast.

| Cause | Evidence |
| --- | --- |
| **The relation is curved, not straight** | The mean official error by forecast band is 117 / 420 / 957 / 2,907 / 4,732 MW for 0–5 / 5–10 / 10–15 / 15–20 / 20+ GW. A line through it over-corrects the middle (in-sample, 53–64% of 5–15 GW hours over-corrected) and under-corrects the ends. |
| **The relation drifts month to month** | Refitted per month, the slope is 0.208 / 0.248 / 0.317 and the correction at 17.4 GW is 3,178 / 2,338 / 2,958 MW (Oct / Nov / Dec). Out of sample (8 weekly folds, Nov–Dec), the production specification's corrected bias is **−635 MW**, worse than official in the 5–15 GW band. |
| **Most remaining error is a whole-day offset** | 66% of the out-of-sample corrected-error variance is the day's mean (59% in January). Forecast size cannot see it. It barely persists to the most recent fully known day: the daily autocorrelation at lag 2 is 0.16. |
| Not the cause: skewed errors | The mean and median error are close in every band (\|skew\| ≤ 0.2), so targeting the mean instead of the median is not what drives it. |

Where the over-correction lands moves between months. In the Nov–Dec CV it was 5–15 GW. In January it is 15–20 GW, with a bias of −392 MW (3 Jan is one of these days). So it is not a fixed offset that could be calibrated away.

## 2. Method

[`protocol.json`](protocol.json) was written after the audit and **locked** (SHA-256 in `reports/protocol_lock.json`) before any candidate was fitted.

- **Data:**
  - **Development:** deliveries 2 Oct – 31 Dec 2025 (2,185 h), used to fit, tune and choose.
  - **Test:** January 2026 decisions (deliveries 2 Jan – 1 Feb, 744 h), read **once** after `reports/selection.json` was written.
  - **February:** not used.
- **January is not untouched.** The defaults were chosen partly on January evidence, earlier experiments scored their winners on it, and its diagnostics had been viewed. That favours the production baseline in any January comparison.
- **Features:** the 9 production features, each built from data published by 09:50 UK on D-1. Linear models get the hour on a 24 h circle, training-median imputation and standardisation. Tree models get the raw features.
- **Validation:** the existing-data experiment's 8 weekly chronological folds (6 Nov – 31 Dec 2025). Each trains on every delivery up to 2 days before the fold, so all its labels were published before the fold's first cutoff. 1,344 pooled out-of-fold hours.
- **Candidates (45 configurations):**
  - **Regression:** official forecast (reference), production linear (baseline), Ridge on forecast size (α 1–10,000), Ridge on all 9 features (α 0.1–10,000), and gradient boosting (16 settings, squared or absolute loss). The Random Forest is the production artifact, not tuned.
  - **Classification:** base rate, production logistic, L2 logistic on all 9 features (C 0.01–10), gradient boosting (8 settings), and the production Random Forest.
- **Selection:** within each family, the lowest pooled CV MAE (regression) or Brier score (classification). January was never used to select.
- **Final models:** one refit on all development data. The four production models are the unchanged artifacts; refits reproduce each exactly (difference 0.0).
- **Decision rule:** a candidate is *supported* only if (a) its CV score beats the baseline and (b) the 95% interval of the January gain is above zero. The interval uses a 3-day block bootstrap over days, 5,000 resamples.

## 3. Results

### Corrected forecast (MW; bias = corrected − actual, negative = over-corrected)

| Model (selected setting) | CV MAE | CV bias | January MAE | January RMSE | January bias | January MAE gain vs production (95% CI) | Supported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Official forecast | 1,817.5 | +1,046.2 | 1,995.6 | 2,498.8 | +1,728.0 | — | — |
| **Production linear** | 1,477.8 | −634.6 | **1,178.6** | **1,535.1** | +31.2 | baseline | — |
| Ridge, forecast only (α = 1) | 1,477.8 | −634.5 | 1,178.5 | 1,535.0 | +31.3 | +0.1 (−0.1, +0.2) | no (not better in CV) |
| Ridge, all 9 (α = 10) | 1,314.4 | −332.9 | 1,380.3 | 1,701.8 | −35.2 | −201.7 (−327.7, −90.0) | no |
| Gradient boosting | **1,120.3** | −108.0 | 1,538.3 | 1,896.0 | −79.3 | −359.7 (−528.4, −171.3) | no |
| Random Forest (existing) | 1,158.5 | −314.6 | 1,381.3 | 1,739.6 | −22.1 | −202.7 (−319.9, −67.5) | no |

- **Regularising the forecast-only line does not help.** Stronger shrinkage only worsened CV MAE (α = 10,000 gives 1,738.9), and the bias stayed between −574 and −635 MW.
- **More features and flexibility help in Nov–Dec but hurt in January.** All three losses are significant (the intervals exclude zero).
- **By UK hour:** the production line, or its identical Ridge twin, has the lowest January MAE in **all 24 hours** (`reports/january_regression_by_hour.csv`).

### High-forecast days (daily mean forecast ≥ 16 GW): does flexibility reduce over-correction?

| | Nov–Dec CV: 9 days, bias / MAE | January: 10 days, bias / MAE |
| --- | --- | --- |
| Production linear | +791 / 1,281 | **+23 / 1,089** |
| Ridge, all 9 | +765 / 1,198 | −174 / 1,294 |
| Gradient boosting | +380 / 1,033 | −650 / 1,400 |
| Random Forest | +302 / 1,039 | −502 / 1,308 |

- **Nov–Dec:** the production line *under*-corrected high-forecast days, and the flexible models fixed most of that.
- **January:** the same models over-correct those days (gradient boosting is below actual in 72% of high-forecast hours). The production line is almost unbiased.
- **Day by day** (`reports/january_high_forecast_days.csv`):
  - On 3 Jan, gradient boosting over-corrects by 1,688 MW and the production line by 622 MW.
  - All high-day MAE gains are negative: −205, −311 and −219 MW. Their intervals just include zero (10 days only).

**More flexibility did not reduce over-correction out of sample; it made it worse.**

### Shortfall risk (classification, evaluated separately)

| Model | CV ROC-AUC | CV PR-AUC | CV Brier | January ROC-AUC | January PR-AUC | January Brier | January Brier gain (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base rate | — | — | 0.2617 | 0.500 | 0.660 | 0.2434 | — |
| **Production logistic** | 0.781 | 0.805 | 0.1987 | **0.755** | **0.874** | **0.2015** | baseline |
| L2 logistic, all 9 (C = 0.1) | 0.813 | 0.829 | 0.1777 | 0.752 | 0.863 | 0.2101 | −0.0086 (−0.0231, +0.0088) |
| Gradient boosting | 0.838 | 0.846 | 0.1683 | 0.742 | 0.873 | 0.2162 | −0.0147 (−0.0435, +0.0155) |
| Random Forest (existing) | 0.836 | 0.844 | 0.1651 | 0.723 | 0.864 | 0.2209 | −0.0194 (−0.0478, +0.0087) |

- **None is supported.** Every candidate's January Brier and ROC-AUC are worse than production logistic, though the intervals include zero.
- **The base rate's pooled CV ROC-AUC (0.368) is an artefact, not a skill measure.** Its constant differs per fold, so pooling ranks folds against each other; its Brier score is the meaningful reference.

## 4. Conclusion and recommended next step

- **Keep the approved defaults.** No result here supports a production change. The models that win in Nov–Dec CV lose in January, as in the earlier existing-data experiment (`../existing_data/README.md`).
- **Why:** with about 90 days of development data, the relationship between forecast and error changes from month to month. A flexible model follows the training months' pattern more closely and transfers worse. The simple line happens to transfer best to January.
- **Next step:** stop searching models on the existing data. Two pre-registered experiments now agree. More searching against January would tune to the test period.
- **What could change the answer:** more history covering more weather regimes, and a genuinely untouched test period. Both need data outside the current approved scope, so they are the user's decision.

## 5. Limitations

- **January was seen before** (see Method), and three candidates per task are compared without a multiplicity correction. Neither would turn a loss into a win, but small differences should not be read as evidence.
- **Small samples:** the CV covers 8 weeks and the high-forecast-day analysis 9–10 days, so intervals are wide.
- **Unexplored territory:** only the 9 production features and modest grids were tried. Richer features, rolling refits, ensembles and calibration were covered by the earlier existing-data experiment, with the same conclusion.
- **Time alignment:** the forecast-to-actual pairing follows the brief's start-time rule. Forecast and actual changes correlate best at a 1–2 hour offset across the data, a question noted in the 3 Jan 2026 investigation; it is not resolved here.

## Reproduce (from `backend/`)

```bash
python -m ml.v2.ridge_gbm.audit            # phase 1: development-only audit -> reports/audit.json
python -m ml.v2.ridge_gbm.lock             # done once on 2026-10-27; refuses to re-lock unless --force
python -m ml.v2.ridge_gbm.run --dev-only   # CV + selection only (does not read January)
python -m ml.v2.ridge_gbm.run              # CV + selection + freeze, then the single January evaluation
pytest ml/v2/ridge_gbm/tests               # scope, fold timing, label-poisoning, protocol and selection checks
```

All randomness is seeded (42), and gradient boosting does not use early stopping. One caveat: the Random Forest averages trees in parallel, so its predictions can vary in the last bits (~1e-12 MW) between calls.

| File | Content |
| --- | --- |
| `protocol.json`, `reports/protocol_lock.json` | Pre-registered plan and its fingerprint |
| `reports/audit.json` | Phase 1 audit |
| `reports/cv_regression.csv`, `reports/cv_classification.csv` | Every configuration: pooled and per-fold CV scores |
| `reports/selection.json` | Selected configurations and model fingerprints, written before January was read |
| `reports/results.json` | January scores, segments, residuals, bootstrap intervals, decisions, and the CV segments |
| `reports/january_regression_by_hour.csv`, `january_regression_by_band.csv`, `january_high_forecast_days.csv`, `january_classification_by_band.csv` | Tables |
| `reports/run_log.txt` | Console output of the evaluation run |
| `models/*.joblib` | The selected Ridge and gradient-boosting models (experimental; not served) |
