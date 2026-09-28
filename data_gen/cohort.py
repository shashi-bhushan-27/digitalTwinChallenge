"""Cohort layer: synthetic EHR records for an Indian Type 2 diabetes clinic population.

Continuous labs and anthropometrics are drawn jointly with a Gaussian copula (correlated latent
normals pushed through truncated marginals). Categorical fields (comorbidities, medications,
allergies, diet, habits) are then sampled conditionally on them. No genetic fields exist.

Outputs (all keyed by patient_id):
    patients        one row per patient, the static EHR snapshot the models and dashboard see
    hba1c_history   quarterly HbA1c over the last 12 months
    medications     one row per prescribed drug (drug, class, dose, times, reported adherence)
    truth           hidden per-patient values (true adherence); never used as model features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from common import seeding
from common.units import creatinine_from_egfr

GLUCOSE_LOWERING_CLASSES = ("biguanide", "sulfonylurea", "dpp4_inhibitor", "sglt2_inhibitor", "basal_insulin")

STAPLES = {
    ("north", False): "roti/paratha, dal, sabzi, rice",
    ("north", True): "roti/paratha, dal, sabzi, rice, chicken/egg",
    ("south", False): "rice, sambar, idli/dosa, curd",
    ("south", True): "rice, sambar, idli/dosa, fish/chicken curry",
    ("west", False): "roti/bhakri, dal, rice, poha",
    ("west", True): "roti/bhakri, dal, rice, poha, chicken/fish",
    ("east", False): "rice, dal, sabzi, roti",
    ("east", True): "rice, dal, fish curry, roti",
}


@dataclass
class Cohort:
    patients: pd.DataFrame
    hba1c_history: pd.DataFrame
    medications: pd.DataFrame
    truth: pd.DataFrame


# ----------------------------------------------------------------------------------------------
# Gaussian copula
# ----------------------------------------------------------------------------------------------


def correlation_matrix(names: list[str], pairs: list[list]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    r = np.eye(len(names))
    for a, b, rho in pairs:
        r[idx[a], idx[b]] = r[idx[b], idx[a]] = float(rho)
    return nearest_correlation(r)


def nearest_correlation(r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues and rescale to a unit diagonal (no-op for a valid matrix)."""
    w, v = np.linalg.eigh((r + r.T) / 2)
    if w.min() >= eps:
        return r
    w = np.clip(w, eps, None)
    fixed = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def _base_quantile(spec: dict, z: np.ndarray) -> np.ndarray:
    """Monotone map from a standard normal draw to the untruncated marginal."""
    dist = spec["dist"]
    if dist == "normal":
        return spec["mean"] + spec["sd"] * z
    if dist == "lognormal":
        return spec["median"] * np.exp(spec["sigma"] * z)
    if dist == "shifted_lognormal":
        return spec["shift"] + spec["median"] * np.exp(spec["sigma"] * z)
    raise ValueError(f"unknown dist {dist}")


def _inverse_base(spec: dict, x: float) -> float:
    dist = spec["dist"]
    if dist == "normal":
        return (x - spec["mean"]) / spec["sd"]
    if dist == "lognormal":
        return float(np.log(x / spec["median"]) / spec["sigma"])
    if dist == "shifted_lognormal":
        return float(np.log((x - spec["shift"]) / spec["median"]) / spec["sigma"])
    raise ValueError(f"unknown dist {dist}")


def truncated_marginal(spec: dict, z: np.ndarray) -> np.ndarray:
    """Truncated marginal via the probability integral transform (no point masses at the bounds)."""
    lo = stats.norm.cdf(_inverse_base(spec, spec["min"])) if "min" in spec else 0.0
    hi = stats.norm.cdf(_inverse_base(spec, spec["max"])) if "max" in spec else 1.0
    u = lo + stats.norm.cdf(z) * (hi - lo)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return _base_quantile(spec, stats.norm.ppf(u))


def sample_continuous(cfg: dict, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    specs = cfg["continuous"]
    names = list(specs)
    r = correlation_matrix(names, cfg["correlations"]["pairs"])
    chol = np.linalg.cholesky(r)
    z = rng.standard_normal((n, len(names))) @ chol.T
    out = {name: truncated_marginal(specs[name], z[:, i]) for i, name in enumerate(names)}
    return pd.DataFrame(out), z


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def _std(x: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the highest-scoring `fraction` of rows (prevalence hits the target)."""
    k = int(round(fraction * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    if k > 0:
        mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _clock_add(clock: str, minutes: float) -> str:
    h, m = map(int, clock.split(":"))
    total = int(round((h * 60 + m + minutes) / 15.0)) * 15 % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


# ----------------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------------


def generate_cohort(cfg: dict, n: int, seed: int, id_prefix: str = "SP") -> Cohort:
    r_cont = seeding.rng(seed, "cohort", "continuous")
    r_cat = seeding.rng(seed, "cohort", "categorical")
    r_med = seeding.rng(seed, "cohort", "medications")
    r_hist = seeding.rng(seed, "cohort", "hba1c_history")

    ids = [f"{id_prefix}-{i + 1:04d}" for i in range(n)]
    df, _ = sample_continuous(cfg, n, r_cont)
    df.insert(0, "patient_id", ids)

    # Sex and sex-specific shifts
    female = r_cat.random(n) >= cfg["sex"]["p_male"]
    df.insert(1, "sex", np.where(female, "F", "M"))
    for name, spec in cfg["continuous"].items():
        if "sex_offset_female" in spec:
            df.loc[female, name] = np.clip(
                df.loc[female, name] + spec["sex_offset_female"], spec["min"], spec["max"]
            )

    # Height, weight, BMI (recomputed from rounded values so the three agree)
    h = cfg["height_cm"]
    height = np.where(
        female,
        r_cat.normal(h["female"]["mean"], h["female"]["sd"], n),
        r_cat.normal(h["male"]["mean"], h["male"]["sd"], n),
    ).round(0)
    weight = (df["bmi"] * (height / 100) ** 2).round(1)
    df["height_cm"] = height
    df["weight_kg"] = weight
    df["bmi"] = (weight / (height / 100) ** 2).round(1)

    # Age, duration (onset no earlier than min_onset_age)
    df["age_yr"] = df["age_yr"].round(0).astype(int)
    max_dur = np.maximum(df["age_yr"] - cfg["min_onset_age_yr"], 0.3)
    df["duration_yr"] = np.minimum(df["duration_yr"], max_dur).round(1)

    # Renal: eGFR reported as an integer, creatinine by inverting CKD-EPI 2021
    df["egfr"] = df["egfr"].round(0).astype(int)
    df["creatinine_mgdl"] = creatinine_from_egfr(df["egfr"], df["age_yr"], female).round(2)

    # Rounding to lab precision
    df["waist_cm"] = df["waist_cm"].round(0)
    df["hba1c_pct"] = df["hba1c_pct"].round(1)
    df["fpg_mgdl"] = df["fpg_mgdl"].round(0)
    for col in ["alt_ul", "tg_mgdl", "ldl_mgdl", "hdl_mgdl", "sbp_mmhg", "dbp_mmhg"]:
        df[col] = df[col].round(0)

    # Fatty liver: highest latent score, prevalence fixed by config
    fl = cfg["fatty_liver"]
    score = sum(w * _std(df[col]) for col, w in fl["weights"].items()) + r_cat.logistic(0, 1, n)
    df["fatty_liver"] = _top_fraction(score, fl["prevalence"])

    # Activity level (lifestyle context) from BMI, age and noise
    act = cfg["activity"]
    latent = sum(w * _std(df[col]) for col, w in act["latent_weights"].items()) + r_cat.normal(0, 1, n)
    cuts = np.quantile(latent, np.cumsum(act["probs"])[:-1])
    level_idx = np.searchsorted(cuts, latent)
    # higher latent = more active; levels are listed least to most active
    df["activity_level"] = np.array(act["levels"])[level_idx]
    df["typical_daily_steps"] = df["activity_level"].map(act["typical_steps"]).astype(int)

    # Comorbidities
    cm = cfg["comorbidities"]
    ht = cm["hypertension"]
    df["hypertension"] = (
        (df["sbp_mmhg"] >= ht["sbp_threshold"])
        | (df["dbp_mmhg"] >= ht["dbp_threshold"])
        | (r_cat.random(n) < ht["p_controlled_on_treatment"])
    )
    dl = cm["dyslipidaemia"]
    hdl_low = np.where(female, df["hdl_mgdl"] < dl["hdl_low_female"], df["hdl_mgdl"] < dl["hdl_low_male"])
    df["dyslipidaemia"] = (
        (df["ldl_mgdl"] >= dl["ldl_threshold"])
        | (df["tg_mgdl"] >= dl["tg_threshold"])
        | hdl_low
        | (r_cat.random(n) < dl["p_treated_normal_lipids"])
    )
    cad_score = (
        0.06 * df["age_yr"] + 0.5 * (~female) + 0.04 * df["duration_yr"] + 0.01 * df["sbp_mmhg"]
        + r_cat.logistic(0, 1, n)
    )
    df["coronary_artery_disease"] = _top_fraction(np.asarray(cad_score), cm["coronary_artery_disease"]["base_prevalence"])
    p_thy = np.where(female, cm["hypothyroidism"]["p_female"], cm["hypothyroidism"]["p_male"])
    df["hypothyroidism"] = r_cat.random(n) < p_thy
    neuro_score = _std(df["duration_yr"]) + 0.6 * _std(df["hba1c_pct"]) + 0.4 * _std(df["age_yr"]) + r_cat.logistic(0, 1, n)
    df["peripheral_neuropathy"] = _top_fraction(neuro_score, cm["peripheral_neuropathy"]["base_prevalence"])
    df["ckd"] = df["egfr"] < cm["ckd_egfr_threshold"]

    fh = cfg["family_history"]
    df["family_history_diabetes"] = r_cat.random(n) < fh["diabetes"]
    df["family_history_premature_cvd"] = r_cat.random(n) < fh["premature_cvd"]

    # Allergies and intolerances (independent, low prevalence)
    al = cfg["allergies"]

    def _draw(group: dict) -> list[str]:
        cols = {k: r_cat.random(n) < p for k, p in group.items()}
        return [";".join(k for k in group if cols[k][i]) for i in range(n)]

    df["food_allergies"] = _draw(al["food"])
    df["drug_allergies"] = _draw(al["drug"])
    df["intolerances"] = _draw(al["intolerance"])

    # Region, diet type and habits (reported lifestyle context)
    reg = cfg["region"]["probs"]
    df["region"] = r_cat.choice(list(reg), size=n, p=list(reg.values()))
    p_veg = df["region"].map(cfg["diet"]["p_vegetarian_by_region"]).to_numpy()
    veg = r_cat.random(n) < p_veg
    jain = veg & (r_cat.random(n) < cfg["diet"]["p_jain_given_vegetarian"])
    df["diet_type"] = np.where(jain, "jain", np.where(veg, "vegetarian", "non_vegetarian"))
    df["typical_meals"] = [
        STAPLES[(reg_, dt == "non_vegetarian")] for reg_, dt in zip(df["region"], df["diet_type"], strict=True)
    ]
    hb = cfg["habits"]
    df["usual_dinner_time"] = [
        _clock_add(hb["dinner_time_by_region"][reg_], r_cat.normal(0, hb["dinner_time_sd_min"]))
        for reg_ in df["region"]
    ]
    df["morning_walk_habit"] = r_cat.random(n) < df["activity_level"].map(hb["p_regular_morning_walk_by_activity"]).to_numpy()
    df["post_meal_walk_habit"] = r_cat.random(n) < hb["p_post_meal_walk_habit"]
    df["skips_breakfast"] = r_cat.random(n) < hb["p_breakfast_skipper"]

    # HbA1c history (quarterly) and 12-month trend
    hist_rows = []
    hc = cfg["hba1c_history"]
    trend = r_hist.normal(hc["trend_mean"], hc["trend_sd"], n) + 0.01 * (df["duration_yr"].to_numpy() - 6)
    months = np.array(hc["months"], dtype=float)
    for i, pid in enumerate(ids):
        cur = df["hba1c_pct"].iat[i]
        vals = cur + trend[i] * months / 12.0 + r_hist.normal(0, hc["visit_noise_sd"], len(months))
        vals[months == 0] = cur
        vals = np.maximum(vals, hc["min_value"]).round(1)
        for m, v in zip(months, vals, strict=True):
            hist_rows.append((pid, int(m), float(v)))
    hist = pd.DataFrame(hist_rows, columns=["patient_id", "months_ago", "hba1c_pct"])
    hist["months_ago"] = -hist["months_ago"]
    first = hist[hist["months_ago"] == 12].set_index("patient_id")["hba1c_pct"]
    df["hba1c_12m_ago_pct"] = df["patient_id"].map(first).to_numpy()
    df["hba1c_trend_12m"] = (df["hba1c_pct"] - df["hba1c_12m_ago_pct"]).round(1)

    meds, truth = _sample_medications(cfg["medications"], df, r_med)
    df["n_glucose_lowering_drugs"] = (
        df["patient_id"].map(meds[meds["drug_class"].isin(GLUCOSE_LOWERING_CLASSES)].groupby("patient_id").size())
        .fillna(0).astype(int).to_numpy()
    )
    df["adherence_reported"] = df["patient_id"].map(truth.set_index("patient_id")["adherence_reported"]).to_numpy()

    col_order = [
        "patient_id", "sex", "age_yr", "height_cm", "weight_kg", "bmi", "waist_cm",
        "hba1c_pct", "hba1c_12m_ago_pct", "hba1c_trend_12m", "fpg_mgdl", "duration_yr",
        "egfr", "creatinine_mgdl", "alt_ul", "fatty_liver",
        "ldl_mgdl", "hdl_mgdl", "tg_mgdl", "sbp_mmhg", "dbp_mmhg",
        "hypertension", "dyslipidaemia", "coronary_artery_disease", "ckd", "hypothyroidism",
        "peripheral_neuropathy", "family_history_diabetes", "family_history_premature_cvd",
        "food_allergies", "drug_allergies", "intolerances",
        "region", "diet_type", "typical_meals", "usual_dinner_time", "activity_level",
        "typical_daily_steps", "morning_walk_habit", "post_meal_walk_habit", "skips_breakfast",
        "n_glucose_lowering_drugs", "adherence_reported",
    ]
    return Cohort(patients=df[col_order], hba1c_history=hist, medications=meds, truth=truth)


def _sample_medications(mc: dict, df: pd.DataFrame, r: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    truth_rows: list[dict] = []
    a, b = mc["adherence"]["true_beta"]

    def pick(drugs: dict) -> tuple[str, dict]:
        names = list(drugs)
        p = np.array([drugs[k]["p"] for k in names], dtype=float)
        k = names[r.choice(len(names), p=p / p.sum())]
        return k, drugs[k]

    def dose_of(spec: dict) -> float:
        doses = spec["doses_mg"]
        probs = spec.get("dose_probs", [1.0 / len(doses)] * len(doses))
        return float(doses[r.choice(len(doses), p=np.asarray(probs) / np.sum(probs))])

    for i in range(len(df)):
        p = df.iloc[i]
        pid = p["patient_id"]
        egfr = p["egfr"]
        hba1c = p["hba1c_pct"]
        dur = p["duration_yr"]
        drug_allergy = p["drug_allergies"].split(";") if p["drug_allergies"] else []
        intol = p["intolerances"].split(";") if p["intolerances"] else []
        adh_true = float(r.beta(a, b))
        adh_rep = float(np.clip(adh_true + r.normal(mc["adherence"]["overreport_mean"], mc["adherence"]["overreport_sd"]), 0.3, 1.0))
        truth_rows.append({"patient_id": pid, "adherence_true": round(adh_true, 3), "adherence_reported": round(adh_rep, 2)})
        pat_meds: list[dict] = []

        def add(drug, cls, dose, unit, times, _pid=pid, _meds=pat_meds):
            _meds.append({"patient_id": _pid, "drug": drug, "drug_class": cls, "daily_dose": dose,
                          "dose_unit": unit, "times": ";".join(times)})

        # Metformin
        m = mc["metformin"]
        if egfr >= m["min_egfr"] and "metformin_gi" not in intol and r.random() < m["p_base"]:
            daily = float(m["daily_mg"][r.choice(len(m["daily_mg"]), p=m["daily_mg_probs"])])
            if egfr < m["reduced_max_below_egfr"]:
                daily = min(daily, float(m["reduced_max_mg"]))
            times = ["08:30"] if daily <= 500 else ["08:30", "20:30"]
            add("metformin", "biguanide", daily, "mg", times)

        # Sulfonylurea
        s = mc["sulfonylurea"]
        logit = _logit(s["p_base"]) + s["hba1c_slope"] * (hba1c - 8) + s["duration_slope"] * (dur - 6)
        p_su = _sigmoid(logit) * (s["sulfa_allergy_factor"] if "sulfonamide_antibiotics" in drug_allergy else 1.0)
        if r.random() < p_su:
            name, spec = pick(s["drugs"])
            add(name, "sulfonylurea", dose_of(spec), "mg", spec["times"])

        # DPP-4 inhibitor
        d = mc["dpp4_inhibitor"]
        if r.random() < d["p_base"]:
            name, spec = pick(d["drugs"])
            dose = dose_of(spec)
            rd = spec.get("renal_dose_mg")
            if rd:
                if egfr < 30:
                    dose = float(rd["egfr_below_30"])
                elif egfr < 45:
                    dose = float(rd["egfr_below_45"])
            add(name, "dpp4_inhibitor", dose, "mg", spec["times"])

        # SGLT2 inhibitor
        g = mc["sglt2_inhibitor"]
        p_sg = g["p_base"] if egfr >= g["min_egfr_glycaemic"] else g["p_below_min"]
        if r.random() < p_sg:
            name, spec = pick(g["drugs"])
            add(name, "sglt2_inhibitor", dose_of(spec), "mg", spec["times"])

        # Basal insulin
        bi = mc["basal_insulin"]
        logit = _logit(bi["p_base"]) + bi["hba1c_slope"] * (hba1c - 8) + bi["duration_slope"] * (dur - 6)
        if r.random() < _sigmoid(logit):
            lo, hi = bi["units_per_kg"]
            units = float(np.round(r.uniform(lo, hi) * p["weight_kg"]))
            add("insulin_glargine", "basal_insulin", units, "U", [bi["time"]])

        # Nobody with established T2D and HbA1c >= 7 goes untreated in this cohort
        if not pat_meds and not (hba1c < 7.0 and dur < 2.0):
            if egfr >= m["min_egfr"] and "metformin_gi" not in intol:
                add("metformin", "biguanide", 1000.0 if egfr >= m["reduced_max_below_egfr"] else 500.0, "mg", ["08:30", "20:30"])
            else:
                add("teneligliptin", "dpp4_inhibitor", 20.0, "mg", ["08:00"])

        # Context-only drugs
        o = mc["other"]
        if r.random() < (o["statin_p_if_dyslipidaemia"] if p["dyslipidaemia"] else o["statin_p_otherwise"]):
            add("atorvastatin", "statin", float(r.choice([10, 20, 40])), "mg", ["21:00"])
        if p["hypertension"]:
            if r.random() < o["arb_p_if_hypertension"]:
                add("telmisartan", "arb", float(r.choice([40, 80])), "mg", ["08:00"])
            if r.random() < o["ccb_p_if_hypertension"]:
                add("amlodipine", "calcium_channel_blocker", 5.0, "mg", ["08:00"])
        if p["coronary_artery_disease"] and r.random() < o["aspirin_p_if_cad"]:
            add("aspirin", "antiplatelet", 75.0, "mg", ["13:00"])
        if p["hypothyroidism"]:
            add("levothyroxine", "thyroid_hormone", float(r.choice([25, 50, 75, 100])), "mcg", ["06:30"])

        for med in pat_meds:
            med["adherence_reported"] = round(adh_rep, 2)
        rows.extend(pat_meds)

    meds = pd.DataFrame(rows, columns=["patient_id", "drug", "drug_class", "daily_dose", "dose_unit", "times", "adherence_reported"])
    return meds, pd.DataFrame(truth_rows)
