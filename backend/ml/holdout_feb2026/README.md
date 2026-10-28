# February 2026 fresh holdout: Random Forest vs linear correction vs official forecast

**Why this exists.**

- In January, the simple linear correction had a lower MAE than the Random Forest, which had been selected on December data.
- Because January had already been looked at, it could no longer fairly decide between the two.
- February 2026 is a **fresh holdout**. The models were frozen and their predictions sealed **before any February actual was read**.

## Reproduce (from `backend/`)

```bash
python -m ml.holdout.predict    # stage 1: verify models, build features, audit cutoff, SEAL predictions (~20 s)
python -m ml.holdout.evaluate   # stage 2: verify seal, read February actuals, evaluate (~25 s)
```

`predict` refuses to overwrite an existing seal unless `--force` is given. `evaluate` aborts if the sealed predictions, the holdout models or any earlier result file changed after sealing (SHA-256 check).

## 1. What already existed and was kept unchanged

| Artifact | Status |
| --- | --- |
| `ml/models/error_regressor.joblib` | Random Forest regressor (300 trees, `min_samples_leaf=20`, `max_features=0.5`, seed 42), trained on delivery days 2 Oct – 31 Dec 2025. **Loaded unchanged** |
| `ml/models/shortfall_classifier.joblib` | Random Forest classifier (same settings and training window). **Loaded unchanged** |
| `ml/processed/ml_dataset.csv`, `ml/reports/metrics.json`, `ml/predictions/test_predictions.csv` | January-stage results. **Not modified** (SHA-256 recorded in `sealed_manifest.json` and re-checked) |
| Linear correction (`linear_by_level`) and logistic shortfall model | **Never saved as files.** Only their January predictions existed |

The linear and logistic models (and the base-rate reference) were therefore **rebuilt** with the same code from the same unchanged Oct–Dec training rows. No February data was involved. They were then accepted only after reproducing their saved January predictions **exactly** (maximum difference 0.0 MW, and 0 for probabilities).

The rebuilt linear correction is: predicted error = 0.236 × forecast − 1,374 MW, so corrected = forecast − predicted error. The rebuilt models are saved in `holdout_feb2026/models/`.

## 2. Data and processing steps

**Inputs (read-only):**

| File | Role |
| --- | --- |
| `DATASETS/Feb26_Actual.json` | 1,344 half-hours, 1–28 Feb: **labels**, read only in stage 2 (and for history features, see below) |
| `DATASETS1/Feb26_Forecast.json` | 224 runs (8 a day), published 1–28 Feb |
| `DATASETS/Jan26_Actual.json`, `DATASETS1/Jan26_Forecast.json` | Only for history features of the first February days: data published before those cutoffs |

**Validation** (`reports/data_validation.json`, structural checks only, no errors computed):

- No nulls, non-numeric values, negatives, gaps or duplicates.
- 1 Feb appears in both actual files with **identical** values (48 half-hours, no conflicts); the duplicates are removed.
- One run was published at 13:58 instead of 11:30. This makes no difference: it is after the cutoff.
- Units are MW in both datasets.
- **DST:** February is GMT throughout (UK clocks change on 29 Mar 2026), so 09:50 UK time = 09:50 UTC. The DST logic itself is unchanged and was exercised on the 26 Oct 2025 clock change in the training data.

**Steps:**

1. Clean both datasets with the backend's own `data_processor`. This includes dropping forecast values for hours that had already started when published.
2. Build features for delivery days 2 Feb – 1 Mar with the **same** `feature_builder`.
3. The cutoff is 09:50 Europe/London on D-1. Each hour uses the latest forecast published at or before it.
4. **Cutoff audit** (`reports/cutoff_audit.json`), run for **every one of the 28 delivery days**:
   - The forecast used was published by the cutoff. It was the 09:30 run on D-1 every day, and 0 later forecasts were used.
   - Overwriting every forecast published after the cutoff with 99,999 MW, and every actual published after it with −99,999 MW, leaves all of that day's features unchanged.
   - So **actual generation of the delivery day can only act as the evaluation target**, never as an input.
5. Predict with the frozen models, write `predictions_sealed.csv` with **no actual or error columns**, and record its SHA-256 and the **analysis plan** in `sealed_manifest.json`, all before the labels were read.
6. Stage 2 joins the February actuals by delivery hour and evaluates.

**Sample and exclusions:**

| | Hours | Why |
| --- | --- | --- |
| Sealed predictions | 672 | Delivery 2 Feb – 1 Mar (decisions 1–28 Feb) |
| Excluded: delivery 1 Mar | 24 | No actual generation in the provided files |
| Excluded by design: delivery 1 Feb | – | Decided on 31 Jan, so already part of the January test set |
| **Eligible, identical for every model** | **648 (27 days)** | Every model has a prediction and an actual exists |

Accuracy is scored per hour, because WINDFOR is hourly, exactly as in January.

## 3. Results: corrected forecast (648 hours, 27 days)

| Model | n | MAE (MW) | RMSE (MW) | Bias (MW) | MAE vs official |
| --- | --- | --- | --- | --- | --- |
| Official forecast | 648 | 1,693 | 2,015 | +1,326 | – |
| Random Forest | 648 | 1,298 | 1,630 | −78 | −23.3% |
| **Linear correction** | 648 | **1,082** | **1,442** | −343 | **−36.1%** |

**Paired comparisons.** These use the same hours for both models. A positive difference means B was more accurate.

| A − B | MAE diff | 95% CI, 3-day blocks (primary) | 95% CI, 1-day blocks | DM p (abs) | Days B better | Verdict (pre-registered rule) |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest − linear** (primary) | **+216** | **[−64, +491]** | [−46, +466] | 0.084 | 16 / 27 | **Inconclusive** |
| Official − Random Forest | +395 | [−9, +771] | [+43, +741] | 0.025 | 17 / 27 | Inconclusive (borderline) |
| Official − linear | +610 | [+290, +955] | [+263, +938] | < 0.001 | 21 / 27 | **Linear significantly better** |

- **RMSE differences:** Random Forest − linear +188 [−157, +497]; official − Random Forest +385 [−7, +750]; official − linear +572 [+191, +948].
- **Best model by day:** linear 13, Random Forest 10, official 4. The daily table is in `reports/daily_results.csv`.
- **Uncertainty methods:** the intervals come from a circular moving-block bootstrap over whole days (5,000 resamples). DM is the Diebold–Mariano test with a Newey–West variance (48 h max lag). Both allow for hourly errors that are correlated within and across days.

**What this shows:**

1. **The linear correction significantly beat the official forecast on fresh data:** −610 MW MAE, 95% CI 290 to 955 MW, and better on 21 of 27 days. This confirms the January finding on data that played no part in any choice.
2. **The Random Forest was also better than the official forecast** (−395 MW), but its primary interval just includes zero, so on this rule the improvement is not established.
3. **Random Forest vs linear is not statistically settled.** The point estimate favours linear by 216 MW (and January agreed), but the interval includes zero. February does not prove that linear is better than the Random Forest. It shows that the Random Forest offers **no demonstrated advantage** over a far simpler model.
4. **The linear correction can backfire.** In February it slightly over-corrected (bias −343 MW), because the forecast's bias was smaller than in training. On days when the forecast was already *too low* (16 Feb: 2,923 vs 1,555 MW), the correction made things worse. Any fixed correction needs monitoring and periodic refitting.

## 4. Results: shortfall risk (P[actual ≤ forecast − 1 GW]), evaluated separately from MAE

The observed shortfall rate was 62.2%.

| Model | ROC-AUC [95% CI] | PR-AUC (no-skill = 0.622) | Brier [95% CI] | Skill vs base rate | Calibration error (ECE) | Precision / recall / F1 at 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.690 [0.515, 0.807] | 0.808 | 0.244 [0.175, 0.332] | 0.6% | 0.151 | 0.693 / 0.695 / 0.694 |
| **Logistic by forecast size** | 0.693 [0.598, 0.770] | 0.808 | **0.215** [0.183, 0.246] | **12.2%** | **0.073** | 0.693 / 0.690 / 0.692 |
| Base rate (reference) | 0.500 | 0.622 | 0.245 | – | 0.100 | 0.622 / 1.000 / 0.767 |

- **Ranking:** both models rank risky hours better than chance, with similar ROC-AUC (about 0.69) and PR-AUC (0.81 vs 0.62 no-skill).
- **Probabilities:** the Random Forest's probabilities had **almost no skill** over always quoting the base rate (Brier 0.244 vs 0.245), and they are poorly calibrated. Hours it rated 0–20% (mean 11%) had shortfalls 41% of the time.
- **Logistic model:** better calibrated, with 12% Brier skill. The Random Forest − logistic Brier difference is +0.029, 95% CI [−0.023, +0.096], so it is not significant.
- **Hard yes/no warnings:** at a 0.5 threshold, neither model beats "always predict shortfall" on F1 (0.767).

## 5. Conclusion and recommendation

- **Corrected forecast.** On fresh data, the simple linear correction is the only model with a statistically clear improvement over the official forecast. The Random Forest showed no advantage over it. Recommend shipping the **linear correction** as the product default on the grounds of evidence, simplicity and explainability ("the official forecast over-predicts by roughly a quarter of its own size, less 1.4 GW"). Keep the Random Forest as a monitored challenger. Do not claim that linear is proven better than the Random Forest.
- **Shortfall risk.** Present it as a ranking or risk band rather than precise probabilities. If probabilities are shown, use the logistic model, which is better calibrated. Recalibrate on recent data before relying on them.
- **No trading claims.** Nothing here measures profit. A trading claim needs a separately validated backtest with day-ahead and imbalance prices, trading costs, volumes and the real decision timing.

## 6. Limitations

- **Short holdout.** 27 days, all winter. The intervals are wide, and nothing here covers spring or summer.
- **Same model family as January.** One holdout month cannot rule out that results vary month to month. Monthly re-evaluation is recommended.
- **Unobserved cause.** The likely cause of the bias (curtailment) is not observed. If it changes, both corrections must be refitted.

## Files

| File | Content |
| --- | --- |
| `sealed_manifest.json` | Seal time, SHA-256 of predictions, models, inputs and earlier results, model provenance, the **pre-registered analysis plan** |
| `predictions_sealed.csv` | 672 predictions, with no labels (never modified after sealing) |
| `features.csv` | The cutoff-audited features used for those predictions |
| `models/*.joblib` | Rebuilt linear, logistic and base-rate models (verified against January) |
| `reports/data_validation.json` | Structural checks of the input files |
| `reports/cutoff_audit.json` | Per-day cutoff and leakage audit |
| `reports/holdout_metrics.json` | All metrics, intervals, tests and calibration tables |
| `reports/daily_results.csv` | Daily MAE per model, shortfall rate and mean probabilities |
| `evaluated_predictions.csv` | Sealed predictions joined with actuals and errors (written by stage 2) |
