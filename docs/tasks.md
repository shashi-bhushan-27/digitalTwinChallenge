# Task list

Status: `[x]` done, `[~]` in progress, `[ ]` to do. Updated at the end of each work session.

## Phase 1: Data generation (29 Sep - 8 Oct)
- [x] 1. Scaffold: git (local only), uv project, config loader with provenance tags, seeding, units, deterministic I/O
- [x] 2. Real-data download scripts (Shanghai T2DM, CGMacros CSVs via range requests) and loaders to a common format
- [x] 3. Cohort generator (Gaussian copula, comorbidities, allergies, diet, habits, medications, HbA1c history)
- [x] 4. Physiology: hidden parameters, hybrid glucose-insulin ODE, drug and exercise effects, HbA1c anchoring
- [x] 5. Behaviour: Indian meal library (47 plates), meal timing/portions/eating out, steps, sleep stages, stress/illness, HR, HRV, medication taking, meal log
- [x] 6. Sensors: CGM lag/noise/drift/warm-up/dropouts/compression lows; wearable gaps, phantom steps, stage confusion, HRV noise
- [x] 7. `generate` command, observed/truth split, manifest with content hashes, determinism tests
- [x] 8. Validation metrics, level-matched comparison, calibration on the Shanghai calibration half (loss 41.2 -> 0.54)
- [x] 9. Discriminator and TSTR (with TRTR reference)
- [x] 10. Validation report, data card, data dictionary, parameters-to-verify list, README, phase review and commit
- [x] 11. Generator follow-up (D-018): fall-rate targets, p2 in the search, local refinement (partial improvement, gap documented)

## Waiting on the team
- [x] Confirm spike definition (D-004) and whether CLAUDE.md / BUILD_PROMPT.md stay out of the public repo (D-003)
- [x] Team details (solo: Shashi Bhushan, VIT Vellore); licence MIT
- [ ] Clinical review of `docs/clinical_rules_to_review.md`
- [ ] Check literature values in `docs/parameters_to_verify.md` (ICMR-INDIAB, IFCT 2017, drug labels)

## Phase 2: Models (done 30 Sep)
- [x] Feature builder on observed tables only, with a no-future-leakage test
- [x] Baselines: persistence, linear extrapolation, LightGBM CGM-only
- [x] LightGBM fusion: forecasts 15-120 min with conformal 80 % band, peak model, calibrated spike classifier
- [x] Ablation ladder (+ wearables, + EHR, + meal log) with patient-bootstrap intervals and strata
- [x] Metrics: MAE/RMSE per horizon, Clarke grid, PR-AUC, precision/recall, lead time, false alerts per patient-day, calibration, Brier
- [x] SHAP grouped into clinical concepts vs a similar-patient reference, templated sentences, checks against generator truth
- [x] `docs/model_report.md`, `docs/example_explanations.md`, inference API for the dashboard
- [ ] Clinician read of `docs/example_explanations.md`

## Phase 3: Twin core and what-if
- [x] Twin state estimation from observed CGM + meal log + EHR (record prior + grid fit; D-027)
- [x] `twin/` what-if API on the ODE core (next hours and typical day); Indian meal library for swaps
- [x] Constraint checks (diet, allergies, intolerances, eGFR drug limits, max doses) with reasons; hypoglycaemia flag and risk factors
- [x] Exact truth replay for ground-truth counterfactuals; validation on test patients (recovery, fidelity, counterfactuals, hypo check)
- [x] Twin state card; worked examples with plots; `docs/clinical_rules_to_review.md` section D
- [ ] Clinical review of constraint rules (D1-D9)
## Phase 4: Dashboard (next)
## Phase 5: Submission package
