# Clinical rules to review

Rules the simulator and (later) the what-if simulator rely on. They were written from general knowledge and drug labels as remembered, **not checked against sources**. Please have someone with clinical knowledge (a doctor, pharmacologist or senior medical student) review each line and mark it OK, change it, or remove it.

## A. Prescribing rules used to build the synthetic cohort (`configs/cohort.yaml`, `data_gen/cohort.py`)

| # | Rule as implemented | Where | Review |
|---|---|---|---|
| A1 | Metformin not prescribed if eGFR < 30 | `medications.metformin.min_egfr` | |
| A2 | Metformin total daily dose capped at 1000 mg if eGFR 30-44 | `reduced_max_below_egfr`, `reduced_max_mg` | |
| A3 | Metformin not prescribed with metformin GI intolerance | cohort.py | |
| A4 | Sulfonylurea (glimepiride 1-4 mg, gliclazide MR 30-120 mg) more likely with higher HbA1c and longer duration | `sulfonylurea.*_slope` | |
| A5 | Sulfonylurea prescribed 5x less often if the patient has a sulfonamide-antibiotic allergy (cross-reactivity is debated) | `sulfa_allergy_factor` | |
| A6 | DPP-4 inhibitors: teneligliptin 20 mg, sitagliptin 100 mg (50 mg if eGFR 30-44, 25 mg if < 30), vildagliptin 50 mg twice daily | `dpp4_inhibitor` | |
| A7 | SGLT2 inhibitors (dapagliflozin 10 mg, empagliflozin 10/25 mg) mainly if eGFR >= 45 because glucose-lowering falls below that | `sglt2_inhibitor` | |
| A8 | Basal insulin glargine 0.15-0.35 U/kg at bedtime, more likely with HbA1c > 9 % and long duration | `basal_insulin` | |
| A9 | Every patient with HbA1c >= 7 % or duration >= 2 years gets at least one glucose-lowering drug | cohort.py | |
| A10 | Context drugs: statin more likely with dyslipidaemia; telmisartan / amlodipine with hypertension; aspirin 75 mg with CAD; levothyroxine with hypothyroidism | `medications.other` | |

## B. Drug and physiology effects in the twin (`configs/physiology.yaml`, `data_gen/physiology.py`)

| # | Effect as implemented | Review |
|---|---|---|
| B1 | Metformin lowers hepatic glucose output by up to 30 % (dose-dependent, Emax model) and raises insulin sensitivity slightly | |
| B2 | Sulfonylureas raise insulin secretion both with and without glucose, so they can cause lows when a meal is skipped | |
| B3 | Sulfonylurea clearance falls with eGFR (down to 50 % of normal), so levels accumulate in kidney disease and raise hypoglycaemia risk | |
| B4 | Insulin clearance falls with eGFR (down to 75 % of normal) | |
| B5 | DPP-4 inhibitors amplify the meal-driven (incretin) part of insulin secretion only, so they carry little hypoglycaemia risk | |
| B6 | SGLT2 inhibitors add urinary glucose loss above ~110 mg/dL, scaled by eGFR | |
| B7 | Counter-regulation (glucagon) response below 70 mg/dL weakens with age | |
| B8 | Dawn phenomenon: hepatic output up to 15 % higher around 06:00 | |
| B9 | Insulin sensitivity ~15 % lower in the late evening | |
| B10 | Walking raises insulin-independent glucose uptake immediately and insulin sensitivity for a few hours afterwards | |
| B11 | Short sleep (below 7 h) lowers next-day insulin sensitivity by ~8 % per missing hour; stress and illness lower it further and raise hepatic output | |
| B12 | Fatty liver makes hepatic glucose output harder for insulin to suppress | |

## C. Hypoglycaemia flag (to be used by the what-if simulator, Phase 3)
| # | Rule | Review |
|---|---|---|
| C1 | Flag any simulated trough below 70 mg/dL | |
| C2 | Risk is higher with lower eGFR, older age and sulfonylurea use (these act through B3, B4 and B7 above, not as separate multipliers) | |

## D. What-if constraints (to be completed in Phase 3)
Allergies and intolerances block meals containing the allergen; diet type (vegetarian, Jain) filters the meal library; eGFR limits block metformin and SGLT2 changes as in A1, A2 and A7.
