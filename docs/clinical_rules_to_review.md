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

## C. Hypoglycaemia flag (what-if simulator, `twin/whatif.py`)
| # | Rule | Review |
|---|---|---|
| C1 | Flag any simulated trough below 70 mg/dL as "predicted low"; below 80 mg/dL as "caution" | |
| C2 | Show risk factors with every result: age 65 or older, eGFR below 45, sulfonylurea or basal insulin use (they act through B3, B4 and B7, not as separate multipliers) | |

## D. What-if constraints (`configs/twin.yaml`, `twin/constraints.py`)
| # | Rule as implemented | Review |
|---|---|---|
| D1 | Meal swaps blocked if the dish conflicts with diet type (vegetarian: no meat, fish or egg; Jain: also no onion, garlic or root vegetables) | |
| D2 | Meal swaps blocked if the dish contains a recorded food allergen (peanut, tree nut, milk, egg, fish, shellfish) or intolerance (lactose, gluten) | |
| D3 | Metformin: blocked if eGFR < 30; limited to 1000 mg/day if eGFR 30-44; maximum 2000 mg/day in this twin; warning if metformin GI intolerance | |
| D4 | SGLT2 inhibitors: do not start if eGFR < 25; warning "little glucose-lowering effect" if eGFR < 45 | |
| D5 | Sulfonylureas: maximum glimepiride 8 mg/day, gliclazide MR 120 mg/day; warning if age >= 65 or eGFR < 60 (hypoglycaemia); warning if sulfonamide-antibiotic allergy | |
| D6 | Sitagliptin: maximum 50 mg/day if eGFR 30-44, 25 mg/day if eGFR < 30 | |
| D7 | Basal insulin: warning if one change exceeds 20 % of the current dose | |
| D8 | Walks: warning to check feet and footwear if peripheral neuropathy is recorded | |
| D9 | Stopping a glucose-lowering drug: warning only (simulated, flagged for review) | |
