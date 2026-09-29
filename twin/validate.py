"""Validate the personalised twin against the simulator's hidden truth (simulated world only).

    uv run python -m twin.validate            # writes results/twin/ and docs/twin_report.md

For each test patient: fit the twin on the first 7 observed days, then on held-out days check
1. parameter recovery: estimated vs true insulin sensitivity and beta-cell function;
2. fidelity: CGM error after logged meals and at night (record-only prior, fitted twin, and an
   oracle with the true parameters but the same observed inputs);
3. counterfactuals: for a logged dinner, the twin's predicted change in the 5-hour peak for four
   interventions vs the TRUE change from replaying the patient with the intervention applied to
   the true inputs; compared with using the cohort-average effect;
4. hypoglycaemia flag: for sulfonylurea users, "maximum sulfonylurea dose and breakfast skipped":
   does the twin flag the lows the truth shows?
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config
from data_gen import physiology as ph
from data_gen.generate import load_configs
from twin import constraints as C
from twin import estimate as E
from twin import whatif as W
from twin.model import Observed, Window, build_inputs, lagged, twin_physiology_config
from twin.truth import regenerate

RES = config.REPO_ROOT / "results" / "twin"
LOW_GI = ["millet_bhakri_meal", "brown_rice_dal_salad"]
INTERVENTIONS = {
    "half portion": W.Intervention("scale_meal", factor=0.5),
    "swap to a low-GI plate": None,     # plate chosen per patient (first allowed of LOW_GI)
    "20-min walk after the meal": W.Intervention("add_walk", start_after_min=20, duration_min=20, cadence_spm=100),
    "skip the meal": W.Intervention("skip_meal"),
}


def load_tables(cfgs: dict) -> dict:
    root = config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"] / cfgs["pipeline"]["dataset_version"]
    names = ["patients", "medications", "cgm", "meal_log", "wearable_steps", "splits"]
    return {n: pd.read_parquet(root / "observed" / f"{n}.parquet") for n in names}, root


def fidelity(twin: E.TwinState, obs: Observed, true_lat: pd.DataFrame, pcfg: dict, tcfg: dict, day0: int) -> dict:
    """CGM RMSE on held-out days (after logged meals and at night) for prior, fitted and oracle twins."""
    t0 = pd.Timestamp(obs.cgm["timestamp"].min()).floor("min")
    n = int((pd.Timestamp(obs.cgm["timestamp"].max()) - t0) / pd.Timedelta(minutes=1)) + 1
    win = Window(t0, n)
    fm = E._fit_mask(obs, twin.meals, win, tcfg["estimation"])
    mins, y = fm["mins"], fm["y"]
    keep = mins >= day0 * 1440
    mins, y = mins[keep], y[keep]
    out = {}
    tl = true_lat.copy()
    tl["adherence_true"] = float(obs.patient["adherence_reported"])   # the oracle sees the same doses as the twin
    for name, lat, cs in (("prior", twin.prior, 1.0), ("fitted", twin.latent, twin.carb_scale), ("oracle", tl, 1.0)):
        inp = build_inputs(lat, twin.meals, obs, win, pcfg, carb_scale=cs)
        g = ph.simulate(lat, inp, twin_physiology_config(pcfg), record_every=1440).glucose.astype(float)
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((y - lagged(g, tcfg["estimation"]["cgm_lag_min"])[0, mins]) ** 2)))
    out["n_points"] = int(len(y))
    return out


def pick_meal(obs: Observed, tp, day0: int, slot: str) -> tuple[pd.Series, pd.Series] | None:
    """A logged meal of `slot` on held-out days and the matching true meal (+-30 min)."""
    start = pd.Timestamp(obs.cgm["timestamp"].min()).normalize() + pd.Timedelta(days=day0)
    lg = obs.meal_log[(obs.meal_log["meal_type"] == slot) & (obs.meal_log["timestamp"] >= start)]
    tm = tp.meals.copy()
    tm["timestamp"] = tp.t_start + pd.to_timedelta(tm["minute"], unit="m")
    for _, row in lg.iterrows():
        d = (tm["timestamp"] - row["timestamp"]).abs()
        cand = tm[(d <= pd.Timedelta(minutes=30)) & (tm["meal_type"] == slot)]
        if len(cand):
            return row, cand.iloc[0]
    return None


def true_counterfactual(tp, true_meal: pd.Series, kind: str, plate: dict | None, t_now: pd.Timestamp, hours: float) -> tuple[float, float, float, float]:
    """(base peak, alt peak, base trough, alt trough) of true plasma glucose over [t_now, t_now + hours]."""
    a = tp.minute(t_now)
    b = a + int(hours * 60)
    base = tp.simulate()
    meals = tp.meals.copy()
    idx = meals.index[(meals["minute"] == true_meal["minute"]) & (meals["plate_id"] == true_meal["plate_id"])][0]
    extra = None
    if kind == "half portion":
        for col in ("carbs_g", "fat_g", "protein_g", "fibre_g"):
            meals.loc[idx, col] *= 0.5
    elif kind == "swap to a low-GI plate":
        for col, key in (("carbs_g", "carbs"), ("fat_g", "fat"), ("protein_g", "protein"), ("fibre_g", "fibre")):
            meals.loc[idx, col] = float(plate[key])
        meals.loc[idx, "gi"] = plate["gi"]
    elif kind == "20-min walk after the meal":
        extra = np.zeros(tp.n_min, np.float32)
        s0 = int(true_meal["minute"]) + 20
        extra[s0 : s0 + 20] = 100
    elif kind == "skip the meal":
        # keep the row count (and so every later random draw) identical: zero the meal instead of deleting it
        for col in ("carbs_g", "fat_g", "protein_g", "fibre_g"):
            meals.loc[idx, col] = 0.0
    alt = tp.simulate(meals=meals, extra_steps=extra)
    return float(base[a:b].max()), float(alt[a:b].max()), float(base[a:b].min()), float(alt[a:b].min())


def hypo_scenario(twin: E.TwinState, obs: Observed, tp, pcfg: dict, tcfg: dict, day0: int, lib: dict) -> dict | None:
    """Sulfonylurea users: SU at maximum dose and breakfast skipped, 8 hours from 07:30."""
    su = obs.meds[obs.meds["drug_class"] == "sulfonylurea"]
    if su.empty:
        return None
    drug = su["drug"].iat[0]
    mx = float(tcfg["constraints"]["sulfonylurea"]["max_daily_mg"][drug])
    found = pick_meal(obs, tp, day0, "breakfast")
    if found is None:
        return None
    logged, true_meal = found
    t_now = pd.Timestamp(logged["timestamp"]).normalize() + pd.Timedelta(hours=7, minutes=30)
    hours = 8
    # truth: the morning SU dose (if taken) becomes the maximum dose; breakfast skipped
    doses = tp.med_doses.copy()
    day_start, day_end = tp.minute(t_now), tp.minute(t_now) + hours * 60
    m = (doses["drug"] == drug) & doses["minute"].between(day_start, day_end)
    doses.loc[m, "daily_dose"] = mx
    if not m.any():                                  # patient missed the dose that day: add it as prescribed
        doses = pd.concat([doses, pd.DataFrame([{"patient_id": tp.pid, "minute": tp.minute(t_now) + 30, "drug": drug, "drug_class": "sulfonylurea",
                                                 "daily_dose": mx, "dose_unit": "mg", "n_times": 1}])], ignore_index=True)
    meals = tp.meals.copy()
    idx = meals.index[(meals["minute"] == true_meal["minute"]) & (meals["plate_id"] == true_meal["plate_id"])][0]
    for col in ("carbs_g", "fat_g", "protein_g", "fibre_g"):
        meals.loc[idx, col] = 0.0
    g = tp.simulate(meals=meals, med_doses=doses)
    true_trough = float(g[day_start:day_end].min())
    state = W.state_at(twin, obs, t_now, pcfg, tcfg)
    plan = W.Plan(meals=[], dose_changes={drug: mx})
    o = W.simulate_plan(twin, obs, state, t_now, plan, pcfg, tcfg, lib, hours=hours)
    return {"true_trough": true_trough, "twin_trough": o.trough, "true_low": true_trough < 70, "twin_flag": o.trough < 70,
            "twin_caution": o.trough < tcfg["whatif"]["caution_threshold_mgdl"], "drug": drug}


def run_patient(pid: str, tabs: dict, cfgs: dict, tcfg: dict, library: list[dict], day0: int) -> dict:
    pc = cfgs["physiology"]
    lib = {p["id"]: p for p in library}
    obs = Observed.from_tables(tabs, pid)
    twin = E.fit(obs, pc, tcfg, library)
    tp = regenerate(pid)
    rec = {"patient_id": pid, "si_true": float(tp.latent["si"].iat[0]), "beta_true": float(tp.latent["beta"].iat[0]),
           "si_prior": float(twin.prior["si"].iat[0]), "beta_prior": float(twin.prior["beta"].iat[0]),
           "si_fit": float(twin.latent["si"].iat[0]), "beta_fit": float(twin.latent["beta"].iat[0]),
           "si_log_sd": twin.si_log_sd, "beta_log_sd": twin.beta_log_sd, "carb_scale": twin.carb_scale, **twin.fit}
    rec.update(fidelity(twin, obs, tp.latent, pc, tcfg, day0))
    cf = []
    found = pick_meal(obs, tp, day0, "dinner")
    if found is not None:
        logged, true_meal = found
        plate_by_name = {p["name"]: p for p in library}
        p0 = plate_by_name.get(logged["plate_name"])
        if p0 is not None:
            t_now = pd.Timestamp(logged["timestamp"]) - pd.Timedelta(minutes=15)
            state = W.state_at(twin, obs, t_now, pc, tcfg)
            prior_twin = E.TwinState(pid, twin.prior, twin.prior, 1.0, 0, 0, 0, 0, 0, {}, twin.meals)
            prior_state = W.state_at(prior_twin, obs, t_now, pc, tcfg)
            # the twin's best guess of the portion: the patient's logged carbs corrected for typical under-reporting
            portion = float(logged["carbs_g_est"]) / float(tcfg["estimation"]["logging_bias"]) / p0["carbs"]
            plan = W.Plan(meals=[W.PlannedMeal(15, p0["id"], portion)])
            low_gi = next((pid_ for pid_ in LOW_GI if C.check_plate(lib[pid_], obs.patient).allowed), None)
            for name, iv in INTERVENTIONS.items():
                plate = None
                if iv is None:
                    if low_gi is None:
                        continue
                    iv, plate = W.Intervention("swap_meal", plate_id=low_gi), lib[low_gi]
                tb, ta, _, _ = true_counterfactual(tp, true_meal, name, plate, t_now, tcfg["whatif"]["horizon_h"])
                r_fit = W.whatif(twin, obs, t_now, plan, iv, pc, tcfg, library, state=state)
                r_pri = W.whatif(prior_twin, obs, t_now, plan, iv, pc, tcfg, library, state=prior_state)
                cf.append({"patient_id": pid, "intervention": name, "true_delta": ta - tb, "fitted_delta": r_fit.delta_peak,
                           "prior_delta": r_pri.delta_peak, "true_base_peak": tb, "fitted_base_peak": r_fit.base.peak})
    hypo = hypo_scenario(twin, obs, tp, pc, tcfg, day0, lib)
    return {"record": rec, "counterfactuals": cf, "hypo": ({"patient_id": pid, **hypo} if hypo else None)}


def spearman(a, b) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def summarise(recs: pd.DataFrame, cf: pd.DataFrame, hy: pd.DataFrame) -> dict:
    out = {"n_patients": int(len(recs)),
           "recovery": {"si_prior_vs_true": spearman(recs["si_prior"], recs["si_true"]), "si_fit_vs_true": spearman(recs["si_fit"], recs["si_true"]),
                        "beta_prior_vs_true": spearman(recs["beta_prior"], recs["beta_true"]), "beta_fit_vs_true": spearman(recs["beta_fit"], recs["beta_true"]),
                        "si_fit_x_beta_fit_vs_true_product": spearman(recs["si_fit"] * recs["beta_fit"], recs["si_true"] * recs["beta_true"]),
                        "si_prior_x_beta_prior_vs_true_product": spearman(recs["si_prior"] * recs["beta_prior"], recs["si_true"] * recs["beta_true"])},
           "fidelity_heldout_rmse_median": {k: float(recs[f"rmse_{k}"].median()) for k in ("prior", "fitted", "oracle")},
           "counterfactuals": {}}
    for name, d in cf.groupby("intervention"):
        loo = np.array([d["true_delta"].drop(i).mean() for i in d.index])
        out["counterfactuals"][name] = {
            "n": int(len(d)), "true_delta_median": float(d["true_delta"].median()),
            "mae_fitted": float((d["fitted_delta"] - d["true_delta"]).abs().mean()), "mae_prior": float((d["prior_delta"] - d["true_delta"]).abs().mean()),
            "bias_fitted": float((d["fitted_delta"] - d["true_delta"]).mean()),
            "base_peak_bias_fitted": float((d["fitted_base_peak"] - d["true_base_peak"]).mean()),
            "mae_cohort_average": float(np.abs(loo - d["true_delta"]).mean()),
            "spearman_fitted": spearman(d["fitted_delta"], d["true_delta"]), "spearman_prior": spearman(d["prior_delta"], d["true_delta"]),
            "sign_agreement_fitted": float(np.mean(np.sign(d["fitted_delta"]) == np.sign(d["true_delta"]))),
        }
    if len(hy):
        tp_ = int((hy["true_low"] & hy["twin_flag"]).sum())
        out["hypo"] = {"n_su_users": int(len(hy)), "true_lows": int(hy["true_low"].sum()), "twin_flags": int(hy["twin_flag"].sum()),
                       "flagged_true_lows": tp_, "missed_true_lows": int((hy["true_low"] & ~hy["twin_flag"]).sum()),
                       "false_flags": int((~hy["true_low"] & hy["twin_flag"]).sum()),
                       "true_lows_with_caution_or_flag": int((hy["true_low"] & hy["twin_caution"]).sum()),
                       "trough_spearman": spearman(hy["twin_trough"], hy["true_trough"]),
                       "trough_mae": float((hy["twin_trough"] - hy["true_trough"]).abs().mean())}
    return out


def plots(cf: pd.DataFrame, hy: pd.DataFrame) -> None:
    names = list(cf["intervention"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        d = cf[cf["intervention"] == name]
        lo, hi = min(d["true_delta"].min(), d["fitted_delta"].min()) - 5, max(d["true_delta"].max(), d["fitted_delta"].max()) + 5
        ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1)
        ax.scatter(d["true_delta"], d["prior_delta"], s=14, color="#bbb", label="record-only twin")
        ax.scatter(d["true_delta"], d["fitted_delta"], s=16, color="#3b6ea8", label="personalised twin")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("true change in 5-h peak (mg/dL)")
        ax.set_ylabel("twin's predicted change (mg/dL)")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("What-if answers vs the simulator's true counterfactuals (held-out dinners, test patients)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "counterfactuals.png", dpi=130)
    plt.close(fig)
    if len(hy):
        fig, ax = plt.subplots(figsize=(5, 4.3))
        ax.axhline(70, color="#c0392b", lw=0.8, ls=":")
        ax.axvline(70, color="#c0392b", lw=0.8, ls=":")
        ax.scatter(hy["true_trough"], hy["twin_trough"], s=18, color="#d9822b")
        lo, hi = min(hy["true_trough"].min(), hy["twin_trough"].min()) - 5, max(hy["true_trough"].max(), hy["twin_trough"].max()) + 5
        ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1)
        ax.set_xlabel("true trough (mg/dL)")
        ax.set_ylabel("twin's predicted trough (mg/dL)")
        ax.set_title("Max sulfonylurea dose + breakfast skipped (8 h)", fontsize=10)
        fig.tight_layout()
        fig.savefig(RES / "hypo_scenario.png", dpi=130)
        plt.close(fig)


def main(limit: int | None = None) -> None:
    cfgs = load_configs()
    tcfg = config.load("twin")
    library = config.load("meal_library")["plates"]
    tabs, _ = load_tables(cfgs)
    ids = list(tabs["splits"].query("split == 'test'")["patient_id"])[: limit or None]
    day0 = int(tcfg["estimation"]["fit_days"])
    RES.mkdir(parents=True, exist_ok=True)
    recs, cfs, hys = [], [], []
    tic = time.time()
    for i, pid in enumerate(ids):
        r = run_patient(pid, tabs, cfgs, tcfg, library, day0)
        recs.append(r["record"])
        cfs.extend(r["counterfactuals"])
        if r["hypo"]:
            hys.append(r["hypo"])
        print(f"[twin] {i + 1}/{len(ids)} {pid} rmse prior/fit/oracle {r['record']['rmse_prior']:.1f}/{r['record']['rmse_fitted']:.1f}/"
              f"{r['record']['rmse_oracle']:.1f} ({time.time() - tic:.0f}s)", flush=True)
    recs, cf, hy = pd.DataFrame(recs), pd.DataFrame(cfs), pd.DataFrame(hys)
    recs.to_csv(RES / "twin_fits.csv", index=False)
    cf.to_csv(RES / "counterfactuals.csv", index=False)
    hy.to_csv(RES / "hypo_scenario.csv", index=False)
    summary = summarise(recs, cf, hy)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plots(cf, hy)
    print(json.dumps(summary, indent=1))


def resummarise() -> None:
    """Recompute summary.json and the plots from the saved per-patient CSVs (no re-simulation)."""
    recs, cf = pd.read_csv(RES / "twin_fits.csv"), pd.read_csv(RES / "counterfactuals.csv")
    hp = RES / "hypo_scenario.csv"
    hy = pd.read_csv(hp) if hp.exists() and hp.stat().st_size > 2 else pd.DataFrame()
    summary = summarise(recs, cf, hy)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plots(cf, hy)
    print(json.dumps(summary["counterfactuals"], indent=1))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--summarise-only":
        resummarise()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
