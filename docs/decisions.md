# Decision log

Choices made while building the prototype, with the reason. Items marked **(pending confirmation)** are defaults chosen to keep work moving; change them if the team decides otherwise.

## D-001 Tooling and repository
- Python 3.12 with `uv` (pinned `uv.lock`), YAML configs in `configs/`, parquet tables, pytest. Top-level packages `common/`, `data_gen/`, `validation/`, `models/`, `twin/`, `dashboard/` as proposed in CLAUDE.md.
- One command per stage (`uv run python -m data_gen.generate`, `uv run python -m validation.calibrate`, `uv run python -m validation.report`).
- Git is local only. Nothing is pushed to GitHub until the project runs end to end (team instruction, 29 Sep 2026).

## D-002 Dataset shape
- 400 synthetic patients x 14 observed days (plus one discarded warm-up day so the ODE starts settled). CGM every 5 min (Dexcom-G6-like), wearables at 5 min.
- Split 70 / 15 / 15 by patient (`observed/splits.parquet`), never by time window.
- Tables are separated into `observed/` (what a clinic system would hold) and `truth/` (hidden simulator state). Models may only read `observed/`; `truth/` is for validation and for checking explanations. A test enforces that no hidden column leaks into `observed/`.

## D-003 Working notes kept out of the public repo (confirmed 29 Sep 2026)
`CLAUDE.md` and `BUILD_PROMPT.md` are gitignored so the judged repository contains only project material. 

## D-004 Spike event definition (confirmed 29 Sep 2026)
Default: from a moment when CGM is at or below 180 mg/dL, a spike is CGM above 180 mg/dL for at least 15 minutes within the next 120 minutes (onset prediction). Reason: in a cohort with median HbA1c above 8 %, many moments are already above 180, which would make "above 180 within 2 h" trivially true. Threshold and duration are configurable. Used so far only in `validation/tstr.py` and `validation/metrics.py`.

## D-005 Real data handling
- Shanghai T2DM (figshare 21600933, CC BY 4.0) and CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0) are downloaded by `validation/real_data/download.py` into `data/real/` (gitignored). Only the CGMacros CSV members are fetched (HTTP range requests, ~43 MB instead of the 657 MB archive with photos).
- They are used only to calibrate and validate the generator (and TSTR). They are never redistributed and never used to train the product models.
- CGMacros is non-commercial (NC) and share-alike (SA). Nothing derived from it beyond summary statistics is stored in the repo.

## D-006 ODE model structure
A hybrid rather than plain Bergman: the Bergman minimal model has no meal input and no realistic insulin secretion, which matters for T2D patients who mostly rely on their own insulin. The twin uses a Bergman-style glucose / remote insulin action core, Dalla Man-style gut absorption (two gamma kernels per meal, slowed by fat, protein and fibre) and beta-cell secretion (static saturating term, first-phase term, incretin boost), saturable hepatic glucose output suppressed by insulin (weaker with fatty liver), renal glucose loss above a threshold, counter-regulation below 70 mg/dL, and drug and exercise hooks. Explicit 1-minute Euler steps, vectorised over patients (400 patients x 15 days in about 20 s).

## D-007 Hidden parameters from the record
- Insulin sensitivity and beta-cell function follow log-linear links to EHR fields (BMI, waist, fatty liver, HbA1c, age, activity; duration, HbA1c, BMI) plus log-normal noise the record cannot explain.
- The fasting set point blends the single lab FPG with the fasting level implied by HbA1c (FPG is one noisy measurement).
- Beta-cell function is then scaled per patient so that a fixed standard Indian day gives a mean glucose equal to the ADAG estimate from HbA1c times 7 % unexplained scatter (glycation gap, diet). Beta is bounded so basal insulin stays between 3 and 45 uU/mL; patients who hit a bound keep a glucose-HbA1c discordance, as real patients sometimes do. Without this anchoring, some patients with HbA1c 6.4 % spiked to 370 mg/dL.
- Hepatic output is solved last so the fasting steady state is exact.

## D-008 Randomness
Every random stream is derived from the master seed plus string keys (layer, patient, purpose), so adding a patient or a draw in one layer does not shift another layer. Tests check that the dataset regenerates identically and that a different seed changes it.

## D-009 Real-data split
Shanghai patients with at least 3 days of CGM are split once by patient (seed 7): 48 for calibration, 49 held out. CGMacros T2D participants (HbA1c >= 6.5 %, 14 people) are never used for tuning. CGMacros stores Libre readings interpolated to 1 minute; only the original 15-minute points are used.

## D-010 Level-matched validation
Finding: in both real datasets the CGM recorded during the study is far below the HbA1c-implied level (Shanghai median mean CGM ~140 mg/dL at median HbA1c ~8.5 %, where ADAG predicts ~197). The recordings capture a monitored, treated period. The synthetic cohort (Indian outpatients, HbA1c current) keeps the ADAG level on purpose, and dynamics are compared between real and synthetic patients matched on mean CGM, using a 300-patient comparison pool generated with the same configuration but a wider HbA1c range. Unmatched numbers are reported too.

## D-011 Calibration scope
Seven population-level scalars are tuned on the calibration half only (random search of 32 candidates plus 12 local refinements, fixed seed): meal absorption speed, fasting-to-mean ratio, insulin-action speed (p2), CGM noise size (absolute and relative), CGM noise autocorrelation, and day-to-day insulin-sensitivity variability. Result: loss 47.1 with hand-set defaults, 0.88 after tuning (`configs/calibrated.yaml`; the loss includes the fall/rise-rate targets added in D-018). The tuned fasting-to-mean ratio (about 0.63) is a model-internal set point: overnight glucose sits above it (waning drugs, dawn effect, late dinners), and the simulated 06:00-07:00 glucose matches the recorded FPG (median ratio 0.98, IQR 0.87-1.08, correlation 0.70). Loss = weighted squared median gaps (in units of the real IQR) for CV, MAGE, night/mean ratio, SD of 15-minute changes, roughness (SD of second differences), time to post-meal peak, day-to-day CV, and post-meal rise at low weight. Roughness was added after a first discriminator run separated real from synthetic mainly on it: the simulated physiology already had real-like roughness, the excess came from white sensor noise, while real Libre readings are smoothed by the device. Meal size is deliberately not tuned: Shanghai meals are small hospital portions and half the patients use mealtime insulin, so matching their rise would impose that diet on the Indian cohort.

## D-012 TRTR reference
Approved by the team (29 Sep 2026): the TSTR comparison includes a small CGM-only model trained on the real calibration half, inside `validation/` only.

## D-013 Meal library
47 Indian "plates" (complete meals at a standard portion) with diet type, Jain suitability, allergens and a GI class. Macros are rough estimates for the prototype and must be checked against the Indian Food Composition Tables 2017 (listed in `docs/parameters_to_verify.md`).

## D-014 Sensor model
CGM: first-order interstitial lag drawn per patient from 5-15 min, per-session calibration gain/offset with drift, one AR(1) error process whose size grows with glucose (absolute + 3 % relative part, higher on day 1; autocorrelation tuned by calibration), 2-h warm-up at each 10-day sensor, dropouts and rare long gaps, compression lows at night, clipping to 40-400. Wearables: 5-minute HR with timestamp jitter; charging, removal and not-worn nights (HR missing, steps recorded as zero); phantom steps; sleep stages through a confusion matrix; nightly HRV with noise.

## D-015 No genetic data
No genotype, variant or polygenic fields anywhere; a test fails if such a column appears. Genetics appears only as future work in the README.

## D-016 Prescribing rules need clinical review
Medication assignment in the cohort and drug effect sizes in the ODE are recalled from labels and general practice, not checked. They are listed in `docs/clinical_rules_to_review.md`.

## D-017 TSTR without clock features
Time-of-day features encode the local meal schedule (Shanghai hospital timetable vs Indian home meals), so they do not transfer between the two. The fair TSTR result uses CGM history only; the with-clock variant is reported to show the effect. In the Indian product, time-of-day features will be learned from Indian patients.

## D-018 Post-peak fall rate (generator follow-up)
The first discriminator runs separated real and synthetic 12-hour windows mainly on the steepest 15-minute fall (real -19.8 vs synthetic -14 mg/dL, in Shanghai patients with and without insulin alike). Follow-up (30 Sep 2026): the 5th and 95th percentiles of 15-minute changes were added to the calibration targets, insulin-action speed (p2) was added to the search (tuned value 0.036/min vs 0.025 before), and a local refinement stage was added to the random search. Result: 5th-percentile fall -13.0 vs -14.4 mg/dL real (calibration half), steepest fall per window -15.0 vs -18 to -20 real, synthetic-trained 60-min MAE on real held-out data 18.7 mg/dL (was 19.2; persistence 19.9). The discriminator AUC stayed at 0.95: the remaining cues (sharp single-step drops, 2-hour autocorrelation) are left as known limitations rather than tuned further.

## D-019 Team details and licence
Solo team AMRIT_VIT: Shashi Bhushan, VIT Vellore. The README shows name, college and email; the phone number is only for the submission platform and is not published in the repo. Licence: MIT (default; no other licence was requested).

## D-020 Phase 2 prediction set-up
Prediction times sit on each patient's 5-minute CGM grid wherever a current reading exists and 6 hours of history are available: every 15 minutes for train/validation patients, every 5 minutes for test patients (so alerts are evaluated as a continuous stream). Forecasts predict the change from the current reading; a separate model predicts the peak over the next 2 hours (the quantity explained to the doctor). Features use data at or before the prediction time only (tested by changing the future and checking nothing earlier moves). Validation patients drive early stopping, isotonic calibration, the conformal band width and the alert threshold (maximum F1); every reported number comes from the 60 test patients.

## D-021 Product model = full fusion with meal log
The dashboard model uses CGM + wearables + EHR + meal log. Meal-log features are simply missing for patients who do not log, which LightGBM handles; the ladder reports the model without meal information (CGM + wearables + EHR) as the "no meal log" setting required by the brief.

## D-022 Honest ablation result
Wearables and the EHR do not improve the average forecast once hours of CGM history are available (wearables +0.11 mg/dL MAE, 95 % CI 0.04-0.19; EHR no effect), while the meal log clearly helps (-0.78 mg/dL). Wearables help where the physiology says they should (right after a walk, -0.90 mg/dL) and the EHR adds a small spike-detection gain in a patient's first 24 hours. The generator was not changed to make fusion look better.

## D-023 Explanation design
SHAP explains the peak model in mg/dL against a reference cohort of similar training patients (age +-10 y, diabetes duration +-5 y), interventional TreeSHAP with 100 background rows. Glucose level and short-term trend are one concept (their features are collinear, so separate concepts got arbitrary opposite-signed credit). Time since waking counts as time of day, not sleep. Contributions of concepts whose data are missing (no sleep data, no heart rate) are shown under "Data quality". Sentences compare with similar patients' typical values. Medicines that push the prediction up are described as a marker of harder-to-control diabetes, not as raising glucose.

## D-024 Explanation checks use contrasts, not correlations
Raw correlations between SHAP and hidden truth were confounded by time of day (steps mostly mean "daytime, meals coming"). The checks compare like with like (same post-meal window with and without a walk; short vs long hidden sleep; lowest vs highest third of hidden insulin sensitivity at similar glucose). Two contrasts go against the naive expectation and are reported with their explanation: the post-walk rebound (the model is right, the expectation was naive) and insulin resistance (the model hardly uses the record; effect below 1 mg/dL).

## D-025 Dashboard alert threshold 0.5
Team decision (30 Sep 2026): the dashboard defaults to a calibrated spike probability of 0.5 instead of the validation max-F1 point (0.36). On test patients this catches 92.2 % of spikes (vs 95.9 %) with 0.69 instead of 1.01 false alert episodes per patient-day, and a median lead time of 80 min (vs 95). The doctor can still move the sensitivity slider.

## D-026 The twin sees only observed data
The personalised twin (`twin/`) uses the same ODE structure as the simulator but reads only what a clinic holds: the record, prescriptions with reported adherence, CGM, wearable steps and the meal log. Hidden simulator values are used only by `twin/truth.py` and `twin/validate.py` to score it. Because the model structure is identical, the validation shows best-case (no structural error) behaviour; this is stated wherever the results appear.

## D-027 Personalisation method
Record-based prior (the simulator's population links with all hidden noise switched off, reported adherence), then a coarse-to-fine grid search over insulin sensitivity, beta-cell function and the fasting set point, fitted to the first 7 days. Post-meal CGM is fitted as the rise from the reading at meal time (robust to invisible day-level shifts); night-time CGM (2-6 h) is fitted as absolute level. Logged carbohydrates are divided by an assumed logging bias of 0.9 (to verify); CGM rises of 35 mg/dL or more in 45 min with no logged meal in the previous 150 min are filled with a typical meal for that time of day. A per-patient meal-size factor was tried and dropped: it absorbed input errors and shrank what-if effects (validated on 8 patients: half-portion MAE 15.2 -> 10.5 mg/dL, low-GI swap 18.4 -> 14.5 after the change).

## D-028 What-if modes
Meals, walks and one-off dose changes: simulate the next 5 hours from the twin's current state (the record replayed to "now", glucose nudged to the latest CGM corrected for sensor lag) with and without the intervention; planned doses at full adherence. Drug-regimen changes: compare day 3 of a steady-state simulation of the patient's typical day (from their own meal log) under the current vs changed regimen, reported as changes, not absolute HbA1c predictions. Every result shows "patient factors considered", the constraint verdict, a hypoglycaemia check, and "Simulation for discussion with the patient, not a prescription."

## D-029 Constraints block before simulating
Blocked options (diet, allergy, intolerance, eGFR limits, maximum doses) are never simulated and always carry their reason; warnings (age, eGFR and sulfonylurea hypoglycaemia risk, sulfonamide allergy, neuropathy before walks, large insulin steps) are simulated and shown with the result. All rules are listed in `docs/clinical_rules_to_review.md`.
