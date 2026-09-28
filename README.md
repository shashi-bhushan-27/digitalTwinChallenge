# T2D Digital Twin: a virtual Type 2 diabetes patient for 2-hour glucose-spike prediction and what-if simulation

**Digital Twin Challenge 2026 (Happiest Health)** · Team **AMRIT_VIT**

> Synthetic patients only. Decision support prototype, not a medical device.

| | |
|---|---|
| Team | AMRIT_VIT: _member names to be added_ |
| College / incubator | _VIT (full name and campus to be confirmed)_ |
| Condition | Type 2 diabetes (India) |
| Status | Phase 1 (synthetic data generation and validation) complete; Phase 2 (models) next |
| Video | _link to be added (20+ minute recording)_ |
| Architecture diagram | _docs/architecture.pdf (Phase 5)_ |
| Presentation | _docs/presentation.pdf (Phase 5)_ |
| Licence | _to be confirmed by the team (default MIT)_ |

## Problem and healthcare use case

India has one of the world's largest populations with Type 2 diabetes, and many patients are diagnosed younger, at lower BMI, and with poorer glucose control than Western cohorts. A doctor in a busy clinic sees each patient for a few minutes, while continuous glucose monitors (CGM) and wrist wearables now produce thousands of readings a week that nobody has time to read.

This project builds a **digital twin** of a person with T2D: a virtual patient that fuses the **static record** (EHR: demographics, labs, medications, allergies, diet) with **dynamic streams** (CGM every 5 minutes, heart rate, HRV, sleep stages, steps) in order to:

1. **Predict a glucose spike about 2 hours ahead** (event detection with a calibrated probability, plus a forecast with an uncertainty band);
2. **Explain why** in clinical language (grouped SHAP drivers such as "large evening carbohydrate load" or "poor sleep last night");
3. **Let the doctor ask "what if?"** (swap dinner, add a post-meal walk, change a drug dose or timing) on a physiological model of that patient, with constraint checks (allergies, diet, kidney function) and a hypoglycaemia safety flag.

No real patient data is used anywhere (DPDP Act and HIPAA): the patients are simulated, and real open datasets are used only to calibrate and validate the simulator.

## How it works

```
 EHR cohort (copula) ──► hidden physiology ──► behaviour (Indian meals, walks, sleep, stress)
                              │                         │
                              ▼                         ▼
                    glucose-insulin ODE  ◄── meals, doses taken, exercise
                              │
                              ▼
                 sensor layer (CGM lag/noise/gaps, wearable artefacts)
                              │
        observed tables ──────┴────► models (Phase 2) ──► dashboard (Phase 4)
        truth tables (hidden) ─────► validation of data and explanations only
```

### Synthetic data generator (Phase 1, done)
- **Cohort layer** (`data_gen/cohort.py`): a Gaussian copula over age, BMI, waist, HbA1c, fasting glucose, diabetes duration, eGFR, ALT, lipids and blood pressure with Indian-clinic-style marginals; then fatty liver, comorbidities, allergies and intolerances, region and diet type (vegetarian, non-vegetarian, Jain), habits, a 12-month HbA1c history and medications (metformin, sulfonylureas, DPP-4 and SGLT2 inhibitors, basal insulin) assigned by eGFR-aware rules. No genetic fields.
- **Latent physiology / twin core** (`data_gen/physiology.py`): hidden insulin sensitivity, beta-cell function and hepatic glucose output derived from the record plus variation the record cannot explain; a hybrid Bergman / Dalla Man glucose-insulin ODE with meal absorption, insulin secretion (with incretin effect), hepatic output, renal glucose loss, counter-regulation, drug effects (metformin, sulfonylurea, DPP-4, SGLT2, glargine) and exercise. Each patient is anchored so their typical day matches their recorded HbA1c (with realistic scatter).
- **Behaviour layer** (`data_gen/behaviour.py`): 47 Indian meal "plates" (idli-sambar, rajma chawal, curd rice, biryani, poha...) filtered by diet, allergies and intolerances; meal timing with late dinners, weekend shifts and eating out; morning and post-meal walks, commuting; sleep architecture; stress and illness days; heart rate and HRV; imperfect medication taking; a partial, biased meal log.
- **Sensor layer** (`data_gen/sensors.py`): CGM interstitial lag (5-15 min), calibration error and drift, autocorrelated noise, 2-hour warm-up, dropouts, compression lows; wearable charging and off-wrist gaps, phantom steps, sleep-stage misclassification, noisy HRV.

The dataset (400 patients x 14 days) regenerates identically from one seed and is split 70/15/15 **by patient**.

### Validation against real open data (Phase 1, done)
Real data is used **only** to calibrate four population-level settings and to validate. It is never used to train the product models.
- Shanghai T2DM (Zhao et al., Scientific Data 2023; CC BY 4.0): split by patient into a calibration half and a held-out half.
- CGMacros T2D participants (PhysioNet; CC BY-NC-SA 4.0): external check, never used for tuning.

Results are in [`docs/validation_report.md`](docs/validation_report.md) (summary below).

## Results so far

All numbers come from `uv run python -m validation.report` and `uv run python -m data_gen.describe` (dataset v0.1, seed 20261020). Details: [`docs/validation_report.md`](docs/validation_report.md), [`docs/data_card.md`](docs/data_card.md).

**The synthetic dataset.** 400 patients x 14 days (5,600 patient-days, 1.6 M CGM readings, 2.8 % missing). The simulated CGM has a MARD of 7.1 % against the hidden true glucose. Median time in range is 46 % at a median HbA1c of 8.3 %, and simulated mean glucose follows the HbA1c-based estimate (ADAG).

![three synthetic patients](results/data/example_patients.png)

**Glucose dynamics vs real T2D CGM (level-matched).** Real CGM in the open datasets sits well below the HbA1c-implied level (monitored, treated periods), so each real patient is compared with the synthetic patient of the closest mean glucose. On the held-out Shanghai half (49 patients), all 12 compared metrics fall within half an interquartile range of the real median: CV 0.26 vs 0.27, MAGE 80 vs 92 mg/dL, night/mean ratio 0.84 vs 0.82, time to post-meal peak 94 vs 90 min, spikes above 180 per day 1.60 vs 1.63.

![matched dynamics](results/validation/matched_distributions.png)

**Train on synthetic, test on real (TSTR).** A CGM-only forecaster trained only on synthetic patients predicts real Shanghai glucose 60 minutes ahead with a MAE of 19.2 mg/dL. That beats persistence (19.9) and sits close to a model trained on real patients (17.5). For spike detection, PR-AUC is 0.33, against 0.28 for persistence and 0.37 for the real-trained model.

**Where the simulator still falls short.** A real-vs-synthetic discriminator still separates 12-hour windows easily: AUC 0.95 level-matched. The main cue is that real glucose falls faster after peaks (median steepest 15-minute fall −19.8 vs −14 mg/dL). Time-of-day patterns also don't transfer between a Shanghai hospital meal schedule and an Indian home schedule. Both are listed as next steps in the validation report.

## Repository layout

```
configs/        YAML configs; literature values carry {source, verified} tags
common/         seeding, config loading, units (ADAG, GMI, CKD-EPI 2021), deterministic I/O
data_gen/       cohort, physiology (ODE twin core), behaviour, sensors, generate, describe
validation/     real-data download and loaders, metrics, level-matched comparison,
                calibration, discriminator, TSTR, report
models/         Phase 2: features, baselines, LightGBM fusion, calibration, SHAP
twin/           Phase 3: what-if simulator (constraints, safety checks)
dashboard/      Phase 4: Streamlit + Plotly
docs/           decisions, data dictionary, data card, validation report, review lists, tasks
results/        metrics and figures produced by the pipeline (only numbers from actual runs)
tests/          one test module per layer + determinism and leakage tests
```

## Setup and run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                          # create the environment from uv.lock
uv run python -m data_gen.generate               # build the synthetic dataset (about 1 minute)
uv run pytest                                    # run the tests
uv run python -m validation.real_data.download   # fetch the real validation data (about 47 MB)
uv run python -m validation.calibrate            # optional: re-run the calibration search
uv run python -m validation.report               # validation report and figures
uv run python -m data_gen.describe               # data dictionary and data card
uv run python -m common.provenance               # list of literature values still to verify
```

## Responsible-data notes (DPDP-aware design)
- No personal data is collected, stored or processed: every patient is simulated.
- Real open datasets stay outside the repository (`data/real/` is gitignored); only download scripts are shared, and licences are respected (CGMacros is non-commercial, share-alike).
- Hidden simulator state is stored separately from what a clinic would observe, and tests check that no hidden field leaks into model inputs.
- Every assumption taken from literature is tagged and listed for verification; nothing is presented as a clinical fact until checked.
- The dashboard will always show "Synthetic patient. Decision support only."

## Limitations
- Simulated physiology encodes our assumptions; models trained on it can partly rediscover them (circularity). Hidden variation and validation on real CGM reduce but do not remove this. Performance on synthetic data is an upper bound.
- The real validation datasets are not Indian and are small (tens of patients); their CGM reflects monitored, treated periods.
- Indian population values, meal macros and drug rules are unverified assumptions awaiting checks (`docs/parameters_to_verify.md`, `docs/clinical_rules_to_review.md`).

## Future work
- Genetic risk and pharmacogenomic markers (deliberately out of scope for this prototype).
- Calibrate against an Indian CGM cohort if an open one becomes available; prandial and premixed insulin; GLP-1 receptor agonists.

## Data sources and licences
- Shanghai T2DM / T1DM datasets: Zhao Q. et al., "Chinese diabetes datasets for data-driven machine learning", Scientific Data (2023); figshare, CC BY 4.0.
- CGMacros v1.0.0: PhysioNet, CC BY-NC-SA 4.0.
