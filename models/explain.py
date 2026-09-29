"""Explanations for the peak-glucose model: grouped SHAP, clinical sentences, and checks.

    uv run python -m models.explain

* The explained quantity is the predicted peak glucose over the next 2 hours (mg/dL), so every
  contribution reads in mg/dL. The model predicts the rise from the current reading; the current
  reading itself is added back as a linear term and credited to "current glucose".
* Contributions are measured against a reference cohort of similar patients (training patients
  within +-10 years of age and +-5 years of diabetes duration), using interventional TreeSHAP.
  They sum exactly: predicted peak = reference peak + sum of group contributions.
* Features are summed into clinical concepts and turned into templated sentences with the real
  values (top 3 pushing up, top 2 pushing down). No raw feature names reach the doctor.
* SHAP says what the model relied on, not what would happen if something changed. What-if
  answers come from the ODE twin only (Phase 3).

The checks at the end compare explanations with the simulator's hidden truth (validation
against the simulated world only).
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import config
from models.inference import TwinPredictor
from models.train import load_features

RES = config.REPO_ROOT / "results" / "explain"

# group key -> (label shown to the doctor, feature-name test)
GROUPS: dict[str, tuple[str, callable]] = {
    # level and short-term trend are one concept: the features are strongly collinear, so SHAP
    # would otherwise split their credit arbitrarily (large opposite-signed parts)
    "glucose_now": ("Current glucose and its trend", lambda f: f == "cgm_now" or f.startswith(("cgm_lag_", "cgm_delta_", "cgm_accel", "cgm_rise_from", "cgm_min_since"))
                    or f in ("cgm_mean_60", "cgm_sd_60", "cgm_min_60", "cgm_max_60")),
    "glucose_hours": ("Glucose over the last few hours", lambda f: f.endswith(("_180", "_360")) and f.startswith("cgm_") or f == "cgm_minutes_above_180_2h"),
    "glucose_day": ("Glucose control over the last day", lambda f: f.endswith("_24h") and f.startswith("cgm_")),
    "data_quality": ("Data quality", lambda f: f in ("cgm_missing_2h", "wear_hr_missing_2h")),
    "time_of_day": ("Time of day", lambda f: f.startswith("tod_") or f in ("wear_min_since_wake", "wear_in_bed_now")),
    "recent_meal": ("Logged meals", lambda f: f.startswith("meal_")),
    "activity": ("Physical activity", lambda f: f.startswith("wear_steps")),
    "heart_rate": ("Heart rate now", lambda f: f in ("wear_hr_30", "wear_hr_30_minus_sleep_hr")),
    "sleep": ("Last night's sleep", lambda f: f in ("wear_sleep_total_min", "wear_sleep_deep_min", "wear_sleep_rem_min", "wear_sleep_wake_min")),
    "recovery": ("Overnight recovery (HRV)", lambda f: f in ("wear_sleep_hrv", "wear_sleep_hrv_rel", "wear_sleep_hr")),
    "baseline_control": ("Baseline glycaemic control", lambda f: f in ("ehr_hba1c_pct", "ehr_hba1c_trend_12m", "ehr_fpg_mgdl", "ehr_duration_yr")),
    "body_liver": ("Body composition and liver", lambda f: f in ("ehr_bmi", "ehr_waist_cm", "ehr_fatty_liver", "ehr_alt_ul", "ehr_tg_mgdl",
                                                                 "ehr_hdl_mgdl", "ehr_ldl_mgdl")),
    "kidney": ("Kidney function", lambda f: f in ("ehr_egfr", "ehr_ckd")),
    "medications": ("Medications", lambda f: f in ("ehr_met_g_day", "ehr_su_eq_day", "ehr_dpp4", "ehr_sglt2", "ehr_basal_u_per_kg",
                                                   "ehr_n_glucose_lowering_drugs", "ehr_adherence_reported") or f.startswith("ehr_min_since")),
    "profile": ("Other patient factors", lambda f: f.startswith("ehr_")),
}


def group_of(features: list[str]) -> dict[str, str]:
    out = {}
    for f in features:
        for key, (_, test) in GROUPS.items():
            if test(f):
                out[f] = key
                break
        else:
            out[f] = "profile"
    return out


def group_matrix(phi: np.ndarray, features: list[str]) -> pd.DataFrame:
    gm = group_of(features)
    df = pd.DataFrame(phi, columns=features)
    return df.T.groupby(pd.Series(gm)).sum().T.reindex(columns=list(GROUPS), fill_value=0.0)


def _fmt_h(minutes: float) -> str:
    return f"{minutes / 60:.1f} h" if minutes >= 90 else f"{minutes:.0f} min"


TIME_WINDOWS = [(0, 5, "overnight"), (5, 7, "early morning, before breakfast"), (7, 11, "breakfast hours"), (11, 12, "late morning"),
                (12, 16, "lunch hours"), (16, 19, "evening snack hours"), (19, 23, "dinner hours"), (23, 24, "late night")]


def sentence(key: str, up: bool, x: pd.Series, ref: pd.Series | None = None) -> str:
    """Templated clinical sentence for one concept, using the actual values (x = feature row) and,
    where it helps, the typical value in the reference cohort of similar patients (ref = means)."""
    v = x.get

    def r(col: str, fmt: str = ".0f", scale: float = 1.0) -> str:
        if ref is None or col not in ref or pd.isna(ref[col]):
            return ""
        return f"; similar patients {ref[col] * scale:{fmt}}"

    if key == "glucose_now":
        d = v("cgm_delta_30")
        trend = "trend unclear (CGM gap)" if pd.isna(d) else f"{'rising' if d > 2 else 'falling' if d < -2 else 'flat'} ({d:+.0f} mg/dL in 30 min)"
        return f"Glucose {v('cgm_now'):.0f} mg/dL, {trend}"
    if key == "glucose_hours":
        return f"Last 3 h averaged {v('cgm_mean_180'):.0f} mg/dL (range {v('cgm_min_180'):.0f}-{v('cgm_max_180'):.0f}{r('cgm_mean_180')})"
    if key == "glucose_day":
        if pd.isna(v("cgm_tir_24h")):
            return "Less than a day of CGM history"
        return f"Last 24 h: {100 * v('cgm_tir_24h'):.0f} % time in range, mean {v('cgm_mean_24h'):.0f} mg/dL{r('cgm_mean_24h')}"
    if key == "data_quality":
        gaps = []
        if v("cgm_missing_2h", 0) > 0.05:
            gaps.append(f"CGM missing {100 * v('cgm_missing_2h'):.0f} % of the last 2 h")
        if pd.isna(v("wear_sleep_total_min")):
            gaps.append("no sleep data last night")
        if pd.isna(v("wear_hr_30")):
            gaps.append("no recent heart rate")
        return "Data gaps: " + ("; ".join(gaps) if gaps else "minor") + " (confidence reduced)"
    if key == "time_of_day":
        ts = pd.Timestamp(x["timestamp"]) if "timestamp" in x else None
        if ts is None:
            return "Time of day (usual meal pattern for this hour)"
        h = ts.hour + ts.minute / 60
        label = next(lab for lo, hi, lab in TIME_WINDOWS if lo <= h < hi)
        woke = v("wear_min_since_wake")
        awake = f", awake {_fmt_h(woke)}" if pd.notna(woke) and woke < 20 * 60 else ""
        return f"Time of day {ts:%H:%M}, {label}{awake}"
    if key == "recent_meal":
        m, c = v("meal_min_since_last"), v("meal_carbs_last")
        if pd.notna(m) and m <= 240:
            return f"Logged meal of about {c:.0f} g carbohydrate {_fmt_h(m)} ago"
        return "No meal logged in the last 4 h" if v("meal_n_logs_24h", 0) > 0 else "No meals logged (meal log not used)"
    if key == "activity":
        s60, s120 = v("wear_steps_60"), v("wear_steps_120")
        return f"{s60:.0f} steps in the last hour, {s120:.0f} in 2 h{r('wear_steps_60')}"
    if key == "heart_rate":
        return f"Heart rate {v('wear_hr_30'):.0f} bpm over the last 30 min{r('wear_hr_30')}" if pd.notna(v("wear_hr_30")) else "No recent heart-rate data"
    if key == "sleep":
        t = v("wear_sleep_total_min")
        if pd.isna(t):
            return "No sleep recorded last night (device not worn)"
        return f"Sleep last night {t / 60:.1f} h, {v('wear_sleep_deep_min'):.0f} min deep{r('wear_sleep_total_min', '.1f', 1 / 60)} h"
    if key == "recovery":
        hv = v("wear_sleep_hrv_rel")
        if pd.isna(hv):
            return "Overnight HRV not available"
        return f"Overnight HRV {v('wear_sleep_hrv'):.0f} ms, {100 * hv:.0f} % of this patient's usual"
    if key == "baseline_control":
        return f"HbA1c {v('ehr_hba1c_pct'):.1f} %{r('ehr_hba1c_pct', '.1f')} %; fasting glucose {v('ehr_fpg_mgdl'):.0f} mg/dL; {v('ehr_duration_yr'):.0f} y since diagnosis"
    if key == "body_liver":
        return f"BMI {v('ehr_bmi'):.1f}{r('ehr_bmi', '.1f')}; waist {v('ehr_waist_cm'):.0f} cm{'; fatty liver' if v('ehr_fatty_liver') == 1 else ''}"
    if key == "kidney":
        return f"eGFR {v('ehr_egfr'):.0f}{r('ehr_egfr')}"
    if key == "medications":
        meds = [n for n, f in (("metformin", "ehr_met_g_day"), ("sulfonylurea", "ehr_su_eq_day"), ("DPP-4 inhibitor", "ehr_dpp4"),
                               ("SGLT2 inhibitor", "ehr_sglt2"), ("basal insulin", "ehr_basal_u_per_kg")) if v(f, 0) > 0]
        if up and meds:
            return f"On {len(meds)} glucose-lowering medicine{'s' if len(meds) > 1 else ''} ({', '.join(meds)}), a marker of harder-to-control diabetes"
        return f"Medicines on board: {', '.join(meds) if meds else 'none recorded'}"
    return "Other profile factors (age, sex, activity level, diet)"


class Explainer:
    """Cohort-referenced, grouped SHAP explanations of the predicted 2-hour peak."""

    def __init__(self, predictor: TwinPredictor, reference_pool: pd.DataFrame, ecfg: dict, seed: int = 0):
        self.p = predictor
        self.model: lgb.Booster = predictor.peak["point"]
        self.pool = reference_pool
        self.ecfg = ecfg
        self.rng = np.random.default_rng(seed)
        self.features = predictor.features

    def reference(self, row: pd.Series) -> pd.DataFrame:
        a, dur = row["ehr_age_yr"], row["ehr_duration_yr"]
        m = (self.pool["ehr_age_yr"].sub(a).abs() <= self.ecfg["age_band_yr"]) & (self.pool["ehr_duration_yr"].sub(dur).abs() <= self.ecfg["duration_band_yr"])
        cand = self.pool[m] if m.sum() >= 50 else self.pool
        k = min(self.ecfg["background_size"], len(cand))
        return cand.iloc[self.rng.choice(len(cand), size=k, replace=False)]

    def explain(self, row: pd.Series) -> dict:
        ref = self.reference(row)
        ex = shap.TreeExplainer(self.model, data=ref[self.features], feature_perturbation="interventional")
        phi = ex.shap_values(row[self.features].to_frame().T.astype(float))[0]
        grouped = group_matrix(phi[None, :], self.features).iloc[0]
        # the model predicts the rise; peak = current + rise, so the current reading's linear part
        # (current minus the reference mean) is credited to "current glucose"
        grouped["glucose_now"] += row["cgm_now"] - ref["cgm_now"].mean()
        grouped = reroute_missing(grouped, row)
        base = float(ex.expected_value) + float(ref["cgm_now"].mean())
        pred = float(row["cgm_now"] + self.model.predict(row[self.features].to_frame().T.astype(float))[0])
        order = grouped.sort_values()
        up = [(k, float(c)) for k, c in order[::-1].items() if c > 0.5][:3]
        down = [(k, float(c)) for k, c in order.items() if c < -0.5][:2]
        ref_means = ref[self.features].mean()
        return {
            "predicted_peak": pred, "reference_peak": base, "sum_check": base + float(grouped.sum()),
            "reference_patients": int(ref["patient_id"].nunique()) if "patient_id" in ref else None,
            "contributions": grouped.to_dict(),
            "up": [{"concept": GROUPS[k][0], "mg_dl": c, "text": sentence(k, True, row, ref_means)} for k, c in up],
            "down": [{"concept": GROUPS[k][0], "mg_dl": c, "text": sentence(k, False, row, ref_means)} for k, c in down],
        }


def reroute_missing(grouped: pd.Series, row: pd.Series) -> pd.Series:
    """Contributions of concepts whose data are missing are shown under 'Data quality', so a
    missing night of sleep data never reads as a clinical driver."""
    g = grouped.copy()
    if pd.isna(row.get("wear_sleep_total_min")):
        for k in ("sleep", "recovery"):
            g["data_quality"] += g[k]
            g[k] = 0.0
    if pd.isna(row.get("wear_hr_30")):
        g["data_quality"] += g["heart_rate"]
        g["heart_rate"] = 0.0
    return g


def global_contributions(model: lgb.Booster, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Path-dependent TreeSHAP (exact, fast) grouped into concepts, for global plots and checks."""
    contrib = model.predict(X[features], pred_contrib=True)
    return group_matrix(contrib[:, :-1], features)


def validate(pred: TwinPredictor, df_test: pd.DataFrame, pool: pd.DataFrame, ecfg: dict, truth_root) -> dict:
    """Explanation checks against the simulator's hidden truth (simulated world only)."""
    rng = np.random.default_rng(3)
    model = pred.peak["point"]
    feats = pred.features
    s = df_test.iloc[rng.choice(len(df_test), size=min(ecfg["n_eval_samples"], len(df_test)), replace=False)].reset_index(drop=True)
    G = global_contributions(model, s, feats)

    daily = pd.read_parquet(truth_root / "truth" / "daily_state.parquet")
    lat = pd.read_parquet(truth_root / "truth" / "latent.parquet").set_index("patient_id")
    date = pd.DatetimeIndex(s["timestamp"]).normalize()
    dmap = daily.assign(date=pd.to_datetime(daily["date"])).set_index(["patient_id", "date"])["sleep_hours"]
    sleep_true = dmap.reindex(pd.MultiIndex.from_arrays([s["patient_id"], date])).to_numpy()
    si = s["patient_id"].map(lat["si"]).to_numpy()

    hour = pd.DatetimeIndex(s["timestamp"]).hour
    day = (hour >= 7) & (hour < 22)

    def contrast(name, concept, a_mask, b_mask, expect):
        c = G[list(concept)].sum(1) if isinstance(concept, tuple) else G[concept]
        a, b = c[a_mask].to_numpy(), c[b_mask].to_numpy()
        if len(a) < 10 or len(b) < 10:
            return {"n_a": int(len(a)), "n_b": int(len(b)), "note": "too few rows"}
        from scipy.stats import mannwhitneyu

        diff = float(a.mean() - b.mean())
        return {"group_a": name[0], "group_b": name[1], "mean_a_mg_dl": float(a.mean()), "mean_b_mg_dl": float(b.mean()),
                "difference": diff, "expected": expect, "direction_ok": (diff > 0) == (expect == "a > b"),
                "mann_whitney_p": float(mannwhitneyu(a, b).pvalue), "n_a": int(len(a)), "n_b": int(len(b))}

    post_meal = (s["meal_min_since_last"] >= 30) & (s["meal_min_since_last"] <= 120)
    tert = np.nanpercentile(si, [33.3, 66.7])
    mid_glucose = s["cgm_now"].between(120, 180) & day
    checks = {
        "logged meal": contrast(("large meal (>= 60 g) logged within 45 min", "no meal logged in the last 4 h (daytime)"), "recent_meal",
                                ((s["meal_min_since_last"] <= 45) & (s["meal_carbs_last"] >= 60)).to_numpy(),
                                (((s["meal_min_since_last"] > 240) | s["meal_min_since_last"].isna()) & day).to_numpy(), "a > b"),
        "activity": contrast((">= 1500 steps in the last hour, 30-120 min after a logged meal", "<= 300 steps, same post-meal window"), "activity",
                             (post_meal & (s["wear_steps_60"] >= 1500)).to_numpy(), (post_meal & (s["wear_steps_60"] <= 300)).to_numpy(), "b > a"),
        "sleep": contrast(("hidden sleep < 5.5 h last night", "hidden sleep >= 7.5 h"), ("sleep", "recovery"),
                          (sleep_true < 5.5) & s["wear_sleep_total_min"].notna().to_numpy(), (sleep_true >= 7.5) & s["wear_sleep_total_min"].notna().to_numpy(), "a > b"),
        "insulin resistance": contrast(("hidden insulin sensitivity, lowest third", "highest third (daytime, glucose 120-180)"), ("baseline_control", "body_liver"),
                                       (si <= tert[0]) & mid_glucose.to_numpy(), (si >= tert[1]) & mid_glucose.to_numpy(), "a > b"),
    }
    # top-driver agreement: a large logged meal just eaten should rank among the top-3 upward concepts
    big = (s["meal_min_since_last"] <= 45) & (s["meal_carbs_last"] >= 60)
    top3 = G.apply(lambda r: set(r.sort_values(ascending=False).index[:3]), axis=1)
    agree = float(np.mean([("recent_meal" in top3[i]) for i in np.flatnonzero(big)])) if big.any() else float("nan")

    # deletion test: neutralise the top-3 concepts (set to reference means) vs 3 random concepts
    means = pool[feats].mean()
    gm = group_of(feats)
    base_pred = model.predict(s[feats])
    top_drop, rand_drop, pred_sum, act = [], [], [], []
    for i in range(min(600, len(s))):
        order = G.iloc[i].abs().sort_values(ascending=False)
        top = list(order.index[:3])
        rnd = list(rng.choice([g for g in GROUPS if g != "glucose_now"], size=3, replace=False))
        for groups, sink in ((top, top_drop), (rnd, rand_drop)):
            x = s.iloc[[i]][feats].copy()
            cols = [f for f in feats if gm[f] in groups]
            x[cols] = means[cols].to_numpy()
            d = float(model.predict(x)[0] - base_pred[i])
            sink.append(abs(d))
            if sink is top_drop:
                pred_sum.append(-float(G.iloc[i][top].sum()))
                act.append(d)
    deletion = {"mean_abs_change_top3": float(np.mean(top_drop)), "mean_abs_change_random3": float(np.mean(rand_drop)),
                "corr_expected_vs_actual_change": float(np.corrcoef(pred_sum, act)[0, 1])}

    # consistency: nearest-neighbour pairs from different patients vs random pairs
    Z = s[feats].astype(float)
    Z = ((Z - Z.mean()) / (Z.std() + 1e-9)).fillna(0).to_numpy()
    Gv = G.to_numpy()
    sims_nn, sims_rand = [], []
    for i in range(min(400, len(s))):
        d = np.sum((Z - Z[i]) ** 2, 1)
        d[s["patient_id"].to_numpy() == s["patient_id"].iat[i]] = np.inf
        j = int(np.argmin(d))
        k = int(rng.integers(0, len(s)))

        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        sims_nn.append(cos(Gv[i], Gv[j]))
        sims_rand.append(cos(Gv[i], Gv[k]))
    consistency = {"cosine_similar_moments": float(np.mean(sims_nn)), "cosine_random_pairs": float(np.mean(sims_rand))}
    # context for the activity contrast: what actually happens after a post-meal walk (all test rows)
    t = df_test
    pm = t["meal_min_since_last"].between(30, 120)
    ctx = {}
    for label, m in (("walked", pm & (t["wear_steps_60"] >= 1500)), ("did not walk", pm & (t["wear_steps_60"] <= 300))):
        x = t[m & t["peak_120"].notna()]
        ctx[label] = {"n": int(len(x)), "median_current_glucose": float(x["g_now"].median()),
                      "median_actual_rise_to_2h_peak": float((x["peak_120"] - x["g_now"]).median())}
    ehr_share = float(G[["baseline_control", "body_liver", "kidney", "medications", "profile"]].abs().sum(1).mean())
    # fusion interaction: for similar logged meals (40-100 g, within the hour), does the meal's
    # contribution grow with HbA1c? (least-squares slope, mg/dL per % HbA1c, adjusting for carbs)
    mm = (s["meal_min_since_last"] <= 60) & s["meal_carbs_last"].between(40, 100)
    A = np.column_stack([np.ones(mm.sum()), s.loc[mm, "meal_carbs_last"], s.loc[mm, "ehr_hba1c_pct"]])
    coef = np.linalg.lstsq(A, G.loc[mm, "recent_meal"].to_numpy(), rcond=None)[0]
    interaction = {"n": int(mm.sum()), "slope_per_pct_hba1c": float(coef[2]), "slope_per_10g_carbs": float(10 * coef[1])}
    return {"truth_contrasts": checks, "activity_context": ctx, "mean_abs_ehr_concepts_mg_dl": ehr_share, "meal_hba1c_interaction": interaction,
            "top3_includes_meal_after_big_logged_meal": agree, "n_big_meal_rows": int(big.sum()),
            "deletion": deletion, "consistency": consistency, "global": G.abs().mean().sort_values(ascending=False).to_dict(), "sample": s, "G": G}


def main() -> None:
    from data_gen.generate import load_configs

    mcfg = config.load("models")
    ecfg = mcfg["explain"]
    cfgs = load_configs()
    truth_root = config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"] / cfgs["pipeline"]["dataset_version"]
    pred = TwinPredictor.load()
    df = load_features()
    pool = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    RES.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    v = validate(pred, te, pool, ecfg, truth_root)
    G, s = v.pop("G"), v.pop("sample")
    (RES / "explanation_checks.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(json.dumps({k: v[k] for k in ("truth_contrasts", "top3_includes_meal_after_big_logged_meal", "deletion", "consistency")}, indent=1))

    # global importance and a fusion dependence plot
    imp = pd.Series(v["global"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh([GROUPS[k][0] for k in imp.index], imp.to_numpy(), color="#3b6ea8")
    ax.set_xlabel("mean |contribution| to predicted 2-h peak (mg/dL), test sample")
    ax.set_title("What the peak model relies on (global)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "global_importance.png", dpi=130)
    plt.close(fig)
    m = s["meal_min_since_last"] <= 60
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sc = ax.scatter(s.loc[m, "meal_carbs_last"], G.loc[m, "recent_meal"], c=s.loc[m, "ehr_hba1c_pct"], cmap="viridis", s=10, alpha=0.8)
    fig.colorbar(sc, label="HbA1c (%)")
    ax.set_xlabel("carbohydrate in last logged meal (g), logged within the past hour")
    ax.set_ylabel("contribution of logged meals to peak (mg/dL)")
    ax.set_title("Fusion: same meal, larger predicted peak at higher HbA1c?", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "dependence_meal_hba1c.png", dpi=130)
    plt.close(fig)

    # example sentences for clinician review
    ex = Explainer(pred, pool, ecfg, seed=1)
    outs = pred.predict(te)
    rng = np.random.default_rng(5)
    picks = list(rng.choice(np.flatnonzero(outs["alert"].to_numpy()), 5, replace=False)) + \
        list(rng.choice(np.flatnonzero(~outs["alert"].to_numpy() & (outs["spike_prob"] < 0.2).to_numpy()), 3, replace=False))
    lines = ["# Example explanations for clinical review", "",
             "_Generated by `uv run python -m models.explain` from held-out synthetic test patients. Contributions are in mg/dL of "
             "predicted 2-hour peak glucose, relative to similar patients (age +-10 y, diabetes duration +-5 y). They describe what the "
             "model relied on, not the effect of changing anything._", ""]
    for i in picks:
        row = te.iloc[i]
        e = ex.explain(row)
        o = outs.iloc[i]
        lines += [f"## {row['patient_id']} at {pd.Timestamp(row['timestamp']):%a %d %b %H:%M}",
                  f"Now {row['cgm_now']:.0f} mg/dL. Spike probability {100 * o['spike_prob']:.0f} % ({'ALERT' if o['alert'] else 'no alert'}). "
                  f"Predicted 2-h peak {e['predicted_peak']:.0f} mg/dL (80 % range {o['peak_lo']:.0f}-{o['peak_hi']:.0f}); "
                  f"similar patients' typical peak {e['reference_peak']:.0f} mg/dL. Check: reference + contributions = {e['sum_check']:.1f}.", "",
                  "Pushing the peak up:"]
        lines += [f"- {u['text']} (**{u['mg_dl']:+.0f} mg/dL**)" for u in e["up"]] or ["- none"]
        lines += ["", "Pulling it down:"]
        lines += [f"- {d['text']} (**{d['mg_dl']:+.0f} mg/dL**)" for d in e["down"]] or ["- none"]
        lines.append("")
    (config.REPO_ROOT / "docs" / "example_explanations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[explain] done in {time.time() - tic:.0f}s")


if __name__ == "__main__":
    main()
