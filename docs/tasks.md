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
- [ ] 11. Generator follow-up (D-018): faster post-peak fall; add fall rate to calibration targets

## Waiting on the team
- [ ] Confirm spike definition (D-004) and whether CLAUDE.md / BUILD_PROMPT.md stay out of the public repo (D-003)
- [ ] Team member names, college name for the README; licence (default MIT)
- [ ] Clinical review of `docs/clinical_rules_to_review.md`
- [ ] Check literature values in `docs/parameters_to_verify.md` (ICMR-INDIAB, IFCT 2017, drug labels)

## Phase 2: Models (next)
- [ ] Feature builder (CGM lags/slopes/rolling stats, wearable summaries, EHR) on observed tables only
- [ ] Baselines: persistence, linear extrapolation, LightGBM CGM-only
- [ ] LightGBM fusion: multi-horizon quantile forecast + calibrated spike classifier
- [ ] Ablation ladder and with/without meal log
- [ ] Metrics: MAE/RMSE per horizon, Clarke grid, PR-AUC, precision/recall, lead time, false alerts per patient-day, calibration
- [ ] SHAP grouped into clinical concepts, templated sentences, validation against generator truth

## Phase 3: Twin core and what-if
## Phase 4: Dashboard
## Phase 5: Submission package
