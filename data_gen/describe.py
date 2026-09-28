"""Write docs/data_dictionary.md and docs/data_card.md from the frozen dataset.

    uv run python -m data_gen.describe

Field meanings and roles are declared here; distributions and counts are computed from the
dataset on disk, so the documents always match the data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import config
from data_gen.generate import load_configs


def plot_examples(root, pids: list[str], out_path) -> None:
    """Three days of one synthetic patient per row: true plasma glucose, CGM, meals, steps, sleep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cgm = pd.read_parquet(root / "observed/cgm.parquet")
    gt = pd.read_parquet(root / "truth/glucose_true.parquet")
    meals = pd.read_parquet(root / "truth/meals_true.parquet")
    steps = pd.read_parquet(root / "observed/wearable_steps.parquet")
    sleep = pd.read_parquet(root / "observed/wearable_sleep_stages.parquet")
    pat = pd.read_parquet(root / "observed/patients.parquet").set_index("patient_id")
    t0 = cgm["timestamp"].min().normalize() + pd.Timedelta(days=2)
    t1 = t0 + pd.Timedelta(days=3)
    fig, axes = plt.subplots(len(pids), 1, figsize=(13, 3.0 * len(pids)), squeeze=False)
    for ax, pid in zip(axes[:, 0], pids, strict=True):
        def win(d, pid=pid):
            return d[(d["patient_id"] == pid) & (d["timestamp"] >= t0) & (d["timestamp"] < t1)]
        g, c, m, s, z = win(gt), win(cgm), win(meals), win(steps), win(sleep)
        ax.axhspan(70, 180, color="#e9f3ea", zorder=0)
        ax.axhline(180, color="#c0392b", lw=0.8, ls="--")
        ax.plot(g["timestamp"], g["glucose_mgdl"], color="#9a9a9a", lw=1, label="true plasma glucose (hidden)")
        ax.plot(c["timestamp"], c["glucose_mgdl"], ".", ms=2.2, color="#1f5fa8", label="CGM (observed)")
        for r in m.itertuples():
            ax.annotate(f"{r.carbs_g:.0f} g", (r.timestamp, 48), fontsize=6.5, rotation=90, color="#a0522d")
        ax2 = ax.twinx()
        ax2.bar(s["timestamp"], s["steps"], width=0.0035, color="#5a9e5a", alpha=0.45)
        ax2.set_ylim(0, 3000)
        ax2.set_yticks([])
        ax.scatter(z["timestamp"], np.full(len(z), 392), c=z["stage"], cmap="Purples", vmin=-1, vmax=3, s=4, marker="s")
        p = pat.loc[pid]
        ax.set_title(f"{pid}: {p.age_yr} y {p.sex}, HbA1c {p.hba1c_pct} %, BMI {p.bmi}, eGFR {p.egfr}, {p.diet_type}, {p.region} "
                     f"| green bars: steps, purple strip: sleep stages, brown: carbs eaten", fontsize=8)
        ax.set_ylim(40, 400)
        ax.set_ylabel("mg/dL")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

# role: M = modifier (changes simulated physiology or drug response), C = constraint (limits
# what-if options), X = context (explains, no effect on the maths), O = observed signal,
# H = hidden truth (never a model feature), ID = identifier/time.
PATIENT_FIELDS = {
    "patient_id": ("Synthetic patient identifier (SP-0001...)", "", "sequential", "ID"),
    "sex": ("Sex", "M/F", "Bernoulli", "M: height, waist and HDL offsets; CKD-EPI"),
    "age_yr": ("Age", "years", "copula, truncated normal", "M: insulin sensitivity, counter-regulation, deep sleep, HRV"),
    "height_cm": ("Height", "cm", "normal by sex", "X (BMI to weight)"),
    "weight_kg": ("Body weight", "kg", "BMI x height^2", "M: glucose and insulin distribution volumes"),
    "bmi": ("Body-mass index", "kg/m2", "copula, truncated log-normal", "M: insulin sensitivity (down), beta-cell output (up), resting HR"),
    "waist_cm": ("Waist circumference", "cm", "copula, truncated normal, female offset", "M: insulin sensitivity; fatty-liver score"),
    "hba1c_pct": ("Current HbA1c", "%", "copula, shifted log-normal", "M: anchors mean glucose (ADAG) and beta-cell function"),
    "hba1c_12m_ago_pct": ("HbA1c 12 months ago", "%", "current minus 12-month trend + visit noise", "X (trend line)"),
    "hba1c_trend_12m": ("Change in HbA1c over 12 months", "% points", "normal, drifts up with duration", "X"),
    "fpg_mgdl": ("Fasting plasma glucose (single lab value)", "mg/dL", "copula, shifted log-normal", "M: blended into the fasting set point"),
    "duration_yr": ("Years since diagnosis", "years", "copula, log-normal, onset >= 22 y", "M: beta-cell function (down), HRV (down)"),
    "egfr": ("eGFR (CKD-EPI 2021)", "mL/min/1.73m2", "copula, truncated normal", "M: insulin and sulfonylurea clearance, renal glucose loss; C: metformin/SGLT2 limits"),
    "creatinine_mgdl": ("Serum creatinine", "mg/dL", "inverse CKD-EPI 2021 from eGFR, age, sex", "X"),
    "alt_ul": ("ALT", "U/L", "copula, log-normal", "X; feeds fatty-liver score"),
    "fatty_liver": ("Fatty liver flag", "bool", "top-scoring 55 % on waist, ALT, TG, HbA1c + noise", "M: weaker insulin suppression of hepatic output, lower SI"),
    "ldl_mgdl": ("LDL cholesterol", "mg/dL", "copula, normal", "X"),
    "hdl_mgdl": ("HDL cholesterol", "mg/dL", "copula, normal, female offset", "X"),
    "tg_mgdl": ("Triglycerides", "mg/dL", "copula, log-normal", "X"),
    "sbp_mmhg": ("Systolic BP", "mmHg", "copula, normal", "X"),
    "dbp_mmhg": ("Diastolic BP", "mmHg", "copula, normal", "X"),
    "hypertension": ("Hypertension", "bool", "BP >= 140/90 or treated-controlled (25 %)", "X"),
    "dyslipidaemia": ("Dyslipidaemia", "bool", "LDL/TG/HDL thresholds or treated (30 %)", "X"),
    "coronary_artery_disease": ("Coronary artery disease", "bool", "top 8 % on age, sex, duration, SBP + noise", "X"),
    "ckd": ("Chronic kidney disease (eGFR < 60)", "bool", "derived", "X (C through eGFR)"),
    "hypothyroidism": ("Hypothyroidism", "bool", "Bernoulli by sex", "X"),
    "peripheral_neuropathy": ("Peripheral neuropathy", "bool", "top 20 % on duration, HbA1c, age + noise", "M: slower gastric emptying, lower HRV"),
    "family_history_diabetes": ("First-degree family history of diabetes", "bool", "Bernoulli", "X"),
    "family_history_premature_cvd": ("Family history of premature CVD", "bool", "Bernoulli", "X"),
    "food_allergies": ("Food allergies", "';'-joined", "independent Bernoulli per allergen", "C: meal library filter"),
    "drug_allergies": ("Drug allergies", "';'-joined", "independent Bernoulli", "C: prescribing (sulfonylurea caution)"),
    "intolerances": ("Intolerances (lactose, gluten, metformin GI)", "';'-joined", "independent Bernoulli", "C: meal filter, metformin"),
    "region": ("Home region (cuisine)", "north/south/west/east", "categorical", "X; shapes favourite dishes"),
    "diet_type": ("Diet type", "vegetarian/non_vegetarian/jain", "by region", "C: meal library filter"),
    "typical_meals": ("Typical staples (text)", "", "from region and diet", "X"),
    "usual_dinner_time": ("Reported usual dinner time", "HH:MM", "regional mean + noise", "X; actual time differs (hidden)"),
    "activity_level": ("Reported activity level", "3 levels", "latent from BMI, age + noise", "M: SI; steps target; resting HR"),
    "typical_daily_steps": ("Typical steps for that level", "steps/day", "lookup", "X"),
    "morning_walk_habit": ("Walks in the morning", "bool", "Bernoulli by activity level", "Behaviour"),
    "post_meal_walk_habit": ("Walks after meals", "bool", "Bernoulli", "Behaviour"),
    "skips_breakfast": ("Usually skips breakfast", "bool", "Bernoulli", "Behaviour"),
    "n_glucose_lowering_drugs": ("Number of glucose-lowering drugs", "count", "derived", "X"),
    "adherence_reported": ("Self-reported adherence", "0-1", "true adherence + over-reporting", "X; true adherence is hidden"),
}

OTHER_TABLES = {
    "observed/hba1c_history": "Quarterly HbA1c (months_ago 12, 9, 6, 3, 0) per patient. Context for the trend sparkline.",
    "observed/medications": "One row per prescription: drug, class, total daily dose and unit, dose times (';'-joined), reported adherence. Modifier (drives simulated drug effect) and constraint (what-if).",
    "observed/cgm": "CGM every 5 min: timestamp, glucose_mgdl (NaN when missing), flag (ok, gap, warmup, low, high), sensor_session. Observed signal.",
    "observed/wearable_heart_rate": "5-minute mean heart rate with +-60 s timestamp jitter; rows absent when the device was off the wrist. Observed signal.",
    "observed/wearable_steps": "Steps per 5 minutes; zero (not missing) while the device was off, plus occasional phantom steps. Observed signal.",
    "observed/wearable_sleep_stages": "Device sleep stage per 5-minute epoch in bed (0 awake, 1 light, 2 deep, 3 REM), after a stage-confusion matrix. Observed signal.",
    "observed/wearable_hrv": "RMSSD (ms) per 5-minute epoch during sleep, with measurement noise. Observed signal.",
    "observed/meal_log": "Meals the patient chose to log: time (+-10 min error), meal type, dish name, estimated carbs (biased low, noisy, rounded to 5 g). About 20 % of patients never log.",
    "observed/splits": "train / val / test assignment by patient.",
    "truth/latent": "Hidden per-patient physiology (insulin sensitivity, beta-cell function, hepatic output, drug responsiveness, true adherence, CGM lag) and hidden habits. Never a model feature; used to validate explanations and the twin.",
    "truth/glucose_true": "True plasma glucose every 5 min (before interstitial lag and sensor error).",
    "truth/meals_true": "Everything actually eaten: dish, portion, carbs, protein, fat, fibre, GI class, eating out.",
    "truth/med_doses": "Doses actually taken (adherence and timing jitter applied).",
    "truth/daily_state": "Per patient-day: sleep hours before the day, stress z-score, illness flag, true steps.",
    "truth/nights": "True sleep architecture per night (total, deep, REM, light, awake minutes, efficiency).",
}


def _describe(s: pd.Series) -> str:
    if s.dtype == bool:
        return f"{100 * s.mean():.1f} % true"
    if pd.api.types.is_numeric_dtype(s):
        q1, med, q3 = np.percentile(s.dropna(), [25, 50, 75])
        return f"median {med:.3g} (IQR {q1:.3g}-{q3:.3g}), range {s.min():.3g}-{s.max():.3g}"
    vc = s.replace("", "(none)").value_counts(normalize=True)
    top = ", ".join(f"{k} {100 * v:.0f}%" for k, v in vc.head(4).items())
    return top + (", ..." if len(vc) > 4 else "")


def main() -> None:
    cfgs = load_configs()
    root = config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"] / cfgs["pipeline"]["dataset_version"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pts = pd.read_parquet(root / "observed/patients.parquet")
    meds = pd.read_parquet(root / "observed/medications.parquet")

    lines = [
        "# Data dictionary",
        "",
        f"_Generated by `uv run python -m data_gen.describe` from dataset {manifest['dataset_version']} (seed {manifest['seed']}, "
        f"{manifest['n_patients']} patients x {manifest['n_days']} days). Distributions are computed from the data on disk._",
        "",
        "Roles: **M** modifier (changes simulated physiology or drug response), **C** constraint (limits what-if options), "
        "**X** context (explains, no effect on the maths). Literature-based values behind each distribution are listed in "
        "`docs/parameters_to_verify.md`.",
        "",
        "## observed/patients (static EHR snapshot)",
        "",
        "| field | meaning | unit | generation | role | distribution in this dataset |",
        "|---|---|---|---|---|---|",
    ]
    for col in pts.columns:
        desc, unit, gen, role = PATIENT_FIELDS.get(col, ("", "", "", ""))
        dist = "" if col == "patient_id" else _describe(pts[col])
        lines.append(f"| `{col}` | {desc} | {unit} | {gen} | {role} | {dist} |")
    lines += ["", "## Medications in this dataset", "", "| drug | class | patients | share | daily dose (median) |", "|---|---|---|---|---|"]
    for (drug, cls), g in meds.groupby(["drug", "drug_class"]):
        lines.append(f"| {drug} | {cls} | {g['patient_id'].nunique()} | {100 * g['patient_id'].nunique() / len(pts):.0f} % | {g['daily_dose'].median():g} {g['dose_unit'].iat[0]} |")
    lines += ["", "## Other tables", "", "| table | rows | columns | contents and role |", "|---|---|---|---|"]
    for name, desc in OTHER_TABLES.items():
        t = manifest["tables"][name]
        lines.append(f"| `{name}` | {t['rows']:,} | {', '.join(t['columns'])} | {desc} |")
    (config.REPO_ROOT / "docs" / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- data card ----------------
    cgm = pd.read_parquet(root / "observed/cgm.parquet")
    true = pd.read_parquet(root / "truth/glucose_true.parquet")
    lat = pd.read_parquet(root / "truth/latent.parquet")
    meals = pd.read_parquet(root / "truth/meals_true.parquet")
    log = pd.read_parquet(root / "observed/meal_log.parquet")
    hr = pd.read_parquet(root / "observed/wearable_heart_rate.parquet")
    steps = pd.read_parquet(root / "observed/wearable_steps.parquet")
    daily = pd.read_parquet(root / "truth/daily_state.parquet")
    g = cgm["glucose_mgdl"]
    ref = true.set_index(["patient_id", "timestamp"])["glucose_mgdl"]
    c = cgm[cgm["flag"] == "ok"].copy()
    c["t5"] = c["timestamp"].dt.floor("5min")
    rv = ref.reindex(pd.MultiIndex.from_arrays([c["patient_id"], c["t5"]])).to_numpy()
    ok = ~np.isnan(rv)
    mard = float(np.mean(np.abs(c["glucose_mgdl"].to_numpy()[ok] - rv[ok]) / rv[ok]))
    n_days = manifest["n_days"] * manifest["n_patients"]
    val_path = config.REPO_ROOT / "results" / "validation" / "summary.json"
    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.exists() else None

    card = [
        "# Data card: synthetic Indian Type 2 diabetes twin dataset",
        "",
        f"_Numbers generated by `uv run python -m data_gen.describe` from dataset {manifest['dataset_version']}._",
        "",
        "## Summary",
        f"- **What:** {manifest['n_patients']} synthetic adults with Type 2 diabetes, each with a static EHR record and "
        f"{manifest['n_days']} days of CGM (5-min), wrist-wearable heart rate, steps, sleep stages and HRV, and a partial meal log.",
        "- **Why:** to build and evaluate a digital-twin prototype (2-hour glucose-spike prediction and what-if simulation) "
        "without any real patient data (DPDP Act / HIPAA).",
        "- **Contains no real patient data.** Every record is simulated. Real open datasets were used only to calibrate a few "
        "population-level settings and to validate realism (see `docs/validation_report.md`).",
        "- **No genetic data** of any kind.",
        "",
        "## How it was made",
        "1. Cohort layer: correlated sampling (Gaussian copula) of labs and anthropometrics with Indian-clinic-style marginals, "
        "then comorbidities, allergies, diet, habits and medications by rule (`data_gen/cohort.py`).",
        "2. Latent physiology: hidden insulin sensitivity, beta-cell function and hepatic output from EHR fields plus unexplained "
        "variation; beta anchored so mean glucose matches the HbA1c (`data_gen/physiology.py`, decisions D-006, D-007).",
        "3. Behaviour: Indian meals (47 plates), timing, portions, eating out, walking, sleep architecture, stress and illness "
        "days, heart rate and HRV (`data_gen/behaviour.py`).",
        "4. Glucose-insulin ODE simulation at 1-minute steps with meals, drugs (as actually taken) and exercise.",
        "5. Sensor layer: CGM lag, calibration error and drift, noise, warm-up, dropouts, compression lows; wearable gaps and "
        "artefacts (`data_gen/sensors.py`).",
        "",
        "## Reproducibility",
        f"- Master seed **{manifest['seed']}**; regenerate with `uv run python -m data_gen.generate` "
        f"(Python {manifest['python']}, numpy {manifest['libraries']['numpy']}, pandas {manifest['libraries']['pandas']}, "
        f"scipy {manifest['libraries']['scipy']}).",
        "- `manifest.json` stores a SHA-256 content hash for every table; `tests/test_generate.py` checks that generation is deterministic.",
        "- Config hashes: " + ", ".join(f"{k} `{v}`" for k, v in manifest["config_hashes"].items()),
        "",
        "## Key statistics (this dataset)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| patient-days | {n_days:,} |",
        f"| CGM readings | {len(cgm):,} ({100 * g.isna().mean():.1f} % missing) |",
        f"| CGM accuracy vs true plasma glucose (MARD) | {100 * mard:.1f} % |",
        f"| mean CGM glucose, median across patients | {cgm.groupby('patient_id')['glucose_mgdl'].mean().median():.0f} mg/dL |",
        f"| time in range 70-180, median | {100 * cgm.groupby('patient_id')['glucose_mgdl'].apply(lambda x: ((x >= 70) & (x <= 180)).mean()).median():.0f} % |",
        f"| readings < 70 mg/dL | {100 * (g < 70).mean():.2f} % |",
        f"| meals eaten per day | {len(meals) / n_days:.2f} |",
        f"| share of meals logged | {100 * len(log) / len(meals):.0f} % |",
        f"| heart-rate coverage (5-min slots with a value) | {100 * len(hr) / len(steps):.0f} % |",
        f"| true steps per day, median | {daily['steps_true'].median():.0f} |",
        f"| sleep before each day, median | {daily['sleep_hours'].median():.1f} h |",
        f"| patients whose SI was adjusted for plausibility | {100 * lat['si_adjusted_for_plausibility'].mean():.0f} % |",
        f"| patients whose beta anchor hit a bound | {100 * ((lat['anchor_beta_multiplier'] <= 0.201) | (lat['anchor_beta_multiplier'] >= 4.99)).mean():.0f} % |",
        "",
        "## Validation (summary)",
        "",
    ]
    if val:
        d = val["discriminator"]
        card += [
            f"- Real-vs-synthetic discriminator (12-h windows, level-matched): accuracy {d['level_matched']['accuracy']:.2f}, AUC {d['level_matched']['auc']:.2f}. "
            f"Unmatched main cohort: accuracy {d['unmatched_main_cohort']['accuracy']:.2f}.",
            "- Full tables (level-matched dynamics, post-meal curves, TSTR) in `docs/validation_report.md`.",
        ]
        tstr_path = config.REPO_ROOT / "results" / "validation" / "tstr.csv"
        if tstr_path.exists():
            t = pd.read_csv(tstr_path)
            t = t[t["test"] == "Shanghai held-out"].set_index(["train", "features"])
            syn = t.loc[("synthetic main cohort (train split)", "CGM history only")]
            per = t.loc[("persistence (no training)", "-")]
            real = t.loc[("real Shanghai calibration half (TRTR)", "CGM history only")]
            card.append(
                f"- TSTR on the Shanghai held-out half (CGM history only): 60-min MAE {syn['mae_60']:.1f} mg/dL trained on synthetic vs "
                f"{per['mae_60']:.1f} persistence and {real['mae_60']:.1f} trained on real; spike PR-AUC {syn['spike_pr_auc']:.2f} vs "
                f"{per['spike_pr_auc']:.2f} and {real['spike_pr_auc']:.2f}."
            )
    else:
        card.append("- Not yet run: `uv run python -m validation.report`.")
    card += [
        "",
        "## Intended use",
        "- Developing and demonstrating the twin prototype, teaching, method comparison.",
        "- **Not** for clinical decisions, prevalence estimates, or claims about real Indian patients.",
        "",
        "## Known limitations",
        "- **Circularity.** The simulator encodes our assumptions about how EHR fields, behaviour and physiology relate. A model "
        "trained on this data can partly rediscover those rules. Mitigations: hidden per-patient variation the record does not "
        "explain (insulin sensitivity and beta-cell noise, drug response, gastric emptying, true vs reported adherence and dinner "
        "time, meal-level absorption jitter, day-level stress, sleep and illness), sensor noise, and validation against real CGM. "
        "Performance on this data is an upper bound for performance on real patients, and explanation checks validate against the "
        "simulated world only.",
        "- **Glucose level.** Anchored to HbA1c via the ADAG relation (outpatients); the two real datasets sit lower because they "
        "record monitored, treated periods (decision D-010).",
        "- **Population values are assumptions.** Indian-specific marginals, prevalences, meal macros and drug effects are informed "
        "guesses awaiting checks (`docs/parameters_to_verify.md`, `docs/clinical_rules_to_review.md`).",
        "- **Treatment scope.** Oral agents and basal insulin only; no prandial/premixed insulin, GLP-1 agonists, pioglitazone or acarbose.",
        "- **Wearables** are plausible by construction but not validated against real wearable data (the real CGM datasets used have no sleep/HRV).",
        "- **Meals** are complete 'plates' at standard portions with rough macros; no mixed-meal glycaemic-index interactions beyond fat, protein and fibre slowing.",
        "",
        "## Licence and provenance",
        "- The synthetic data and code are released under the repository licence.",
        "- Real data used for calibration/validation only, not redistributed: Shanghai T2DM (Zhao et al., Scientific Data 2023; figshare, CC BY 4.0) "
        "and CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0).",
    ]
    (config.REPO_ROOT / "docs" / "data_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    # example patients: low, middle and high HbA1c in the test split
    test_ids = pd.read_parquet(root / "observed/splits.parquet").query("split == 'test'")["patient_id"]
    tp = pts[pts["patient_id"].isin(test_ids)].sort_values("hba1c_pct")
    plot_examples(root, [tp["patient_id"].iat[1], tp["patient_id"].iat[len(tp) // 2], tp["patient_id"].iat[-2]],
                  config.REPO_ROOT / "results" / "data" / "example_patients.png")
    print("wrote docs/data_dictionary.md, docs/data_card.md, results/data/example_patients.png")


if __name__ == "__main__":
    main()
