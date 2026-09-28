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

## D-003 Working notes kept out of the public repo (pending confirmation)
`CLAUDE.md` and `BUILD_PROMPT.md` are gitignored so the judged repository contains only project material. Remove them from `.gitignore` if the team wants them public.

## D-004 Spike event definition (pending confirmation)
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
Six population-level scalars are tuned on the calibration half only (random search, 32 candidates, fixed seed): meal absorption speed, fasting-to-mean ratio, CGM noise size (absolute and relative), CGM noise autocorrelation, and day-to-day insulin-sensitivity variability. Result: loss 41.2 with hand-set defaults, 0.54 after tuning (`configs/calibrated.yaml`). The tuned fasting-to-mean ratio (0.65) is a model-internal set point: overnight glucose sits above it (waning drugs, dawn effect, late dinners), and the simulated 06:00-07:00 glucose matches the recorded FPG (median ratio 1.00). Loss = weighted squared median gaps (in units of the real IQR) for CV, MAGE, night/mean ratio, SD of 15-minute changes, roughness (SD of second differences), time to post-meal peak, day-to-day CV, and post-meal rise at low weight. Roughness was added after a first discriminator run separated real from synthetic mainly on it: the simulated physiology already had real-like roughness, the excess came from white sensor noise, while real Libre readings are smoothed by the device. Meal size is deliberately not tuned: Shanghai meals are small hospital portions and half the patients use mealtime insulin, so matching their rise would impose that diet on the Indian cohort.

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

## D-018 Known generator gap carried forward
The discriminator still separates real and synthetic 12-hour windows (level-matched AUC 0.95). Its main cue: real glucose falls faster after peaks (median steepest 15-minute fall -19.8 vs -14 mg/dL), in insulin users and non-users alike. Candidate fix for the next generator iteration: faster insulin action (p2) or a sharper meal-absorption tail, with post-peak fall rate added to the calibration targets.
