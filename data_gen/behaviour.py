"""Behaviour layer: what each synthetic patient eats, how they move and sleep, and how their
heart rate and HRV respond, minute by minute.

Patient habits start from the EHR lifestyle fields (region, diet type, activity level, usual
dinner time, walking habits) and add hidden variation (appetite, favourite dishes, chronotype,
employment, logging behaviour). Days then vary around the habits (weekday/weekend, eating out,
short nights, stress, illness). The hidden daily state (sleep, stress, illness) also changes
insulin sensitivity in the physiology layer, which is what gives wearables real predictive value
without making them a fixed function of the EHR.

Time base: minute 0 is 00:00 on simulation day 0 (the discarded warm-up day).
Sleep stage codes: -1 out of bed, 0 awake in bed, 1 light, 2 deep, 3 REM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from common import seeding

STAGE_OUT, STAGE_WAKE, STAGE_LIGHT, STAGE_DEEP, STAGE_REM = -1, 0, 1, 2, 3
ALLERGY_TO_TAG = {"peanut": "peanut", "tree_nut": "tree_nut", "shellfish": "shellfish", "fish": "fish",
                  "egg": "egg", "milk_protein": "dairy"}
INTOLERANCE_TO_TAG = {"lactose": "dairy", "gluten": "gluten"}
ACTIVITY_LEVEL = {"sedentary": 0, "lightly_active": 1, "moderately_active": 2}


@dataclass
class BehaviourData:
    meals: pd.DataFrame          # true meals (composition hidden from models)
    meal_log: pd.DataFrame       # what the patient logged (observed, incomplete, biased)
    med_doses: pd.DataFrame      # doses actually taken (hidden)
    daily: pd.DataFrame          # hidden daily state per patient-day
    nights: pd.DataFrame         # true sleep summary per night
    habits: pd.DataFrame         # hidden habit parameters per patient
    hrv: pd.DataFrame            # true 5-min RMSSD during sleep
    steps: np.ndarray            # (N, T) int16 steps per minute
    sleep_stage: np.ndarray      # (N, T) int8
    heart_rate: np.ndarray       # (N, T) float32 bpm


def _clock_min(clock: str) -> int:
    h, m = map(int, clock.split(":"))
    return h * 60 + m


def eligible_plates(library: list[dict], patient: pd.Series | dict) -> list[dict]:
    """Plates this patient can eat given diet type, allergies and intolerances."""
    diet = patient["diet_type"]
    banned: set[str] = set()
    for a in filter(None, str(patient["food_allergies"]).split(";")):
        if a in ALLERGY_TO_TAG:
            banned.add(ALLERGY_TO_TAG[a])
    for t in filter(None, str(patient["intolerances"]).split(";")):
        if t in INTOLERANCE_TO_TAG:
            banned.add(INTOLERANCE_TO_TAG[t])
    out = []
    for p in library:
        if diet in ("vegetarian", "jain") and p["diet"] != "veg":
            continue
        if diet == "jain" and not p["jain_ok"]:
            continue
        if banned.intersection(p["allergens"]):
            continue
        out.append(p)
    return out


def _favourites(plates: list[dict], meal: str, region: str, n: int, weight: float, r: np.random.Generator) -> tuple[list[dict], np.ndarray]:
    cand = [p for p in plates if meal in p["meal"]]
    if not cand:
        return [], np.array([])
    w = np.array([weight if ("all" in p["regions"] or region in p["regions"]) else 1.0 for p in cand])
    k = min(n, len(cand))
    idx = r.choice(len(cand), size=k, replace=False, p=w / w.sum())
    fav = [cand[i] for i in idx]
    pref = r.dirichlet(np.full(k, 2.0))
    return fav, pref


def _hypnogram(n_min: int, age: float, cfg: dict, r: np.random.Generator) -> np.ndarray:
    """Sleep stages for one night of `n_min` minutes after sleep onset."""
    s = cfg["sleep"]
    stages = np.full(n_min, STAGE_LIGHT, dtype=np.int8)
    age_f = max(0.3, 1 - s["deep_age_decline_per_10y"] * (age - 30) / 10)
    t, k = 0, 0
    while t < n_min:
        length = int(r.uniform(*s["cycle_min"]))
        deep = s["deep_fraction_first_cycle"] * s["deep_fraction_decay"] ** k * age_f
        rem = min(0.35, s["rem_fraction_first_cycle"] + s["rem_fraction_growth"] * k)
        n_deep, n_rem = int(deep * length), int(rem * length)
        n_light = length - n_deep - n_rem
        seq = np.concatenate([
            np.full(n_light // 2, STAGE_LIGHT), np.full(n_deep, STAGE_DEEP),
            np.full(n_light - n_light // 2, STAGE_LIGHT), np.full(n_rem, STAGE_REM),
        ]).astype(np.int8)
        end = min(t + length, n_min)
        stages[t:end] = seq[: end - t]
        t, k = end, k + 1
    n_wake = r.poisson(s["awakenings_per_hour"] * n_min / 60)
    for _ in range(n_wake):
        start = int(r.integers(0, max(1, n_min - 1)))
        stages[start : start + int(r.integers(s["awakening_min"][0], s["awakening_min"][1] + 1))] = STAGE_WAKE
    return stages


def _add_bout(steps: np.ndarray, start: int, minutes: int, cadence: tuple[float, float], r: np.random.Generator) -> int:
    start = max(0, start)
    end = min(len(steps), start + max(1, minutes))
    if end <= start:
        return 0
    val = r.uniform(*cadence) * np.exp(r.normal(0, 0.05, end - start))
    steps[start:end] = np.maximum(steps[start:end], val.round().astype(np.int16))
    return int(val.sum())


def generate_patient(patient: pd.Series, meds: pd.DataFrame, adherence_true: float, library: list[dict],
                     heavy_ids: set[str], cfg: dict, n_days: int, start: pd.Timestamp, seed: int) -> dict:
    """All behaviour for one patient over `n_days` days starting at `start` (00:00 of day 0)."""
    pid = patient["patient_id"]
    r = seeding.rng(seed, "behaviour", pid)
    T = n_days * 1440
    mc, ac, sc = cfg["meals"], cfg["activity"], cfg["sleep"]
    age = float(patient["age_yr"])
    dates = [start + pd.Timedelta(days=d) for d in range(n_days)]
    weekend = np.array([dt.weekday() >= 5 for dt in dates])

    # ---------------- hidden habits ----------------
    employed = age < ac["retirement_age"] and r.random() < ac["p_employed_by_sex"][patient["sex"]]
    appetite = float(np.exp(r.normal(0, mc["portion_patient_sd"])))
    activity_factor = float(np.exp(r.normal(0, 0.2)))
    habit_times = {m: _clock_min(mc[m]["mean"]) + r.normal(0, mc[m]["between_patient_sd_min"]) for m in ("breakfast", "lunch", "snack")}
    habit_times["dinner"] = _clock_min(patient["usual_dinner_time"]) + r.normal(0, mc["dinner_actual_vs_reported_sd_min"])
    p_snack = float(r.beta(*mc["p_snack_beta"]))
    bedtime_mean = _clock_min(sc["bedtime_mean"]) + r.normal(0, sc["bedtime_between_sd_min"])
    sleep_need = float(np.clip(r.normal(sc["duration_mean_h"], sc["duration_between_sd_h"]), 4.5, 8.8))
    logs_at_all = r.random() >= cfg["meal_log"]["p_never_logs"]
    log_rate = float(r.beta(*cfg["meal_log"]["logging_rate_beta"])) if logs_at_all else 0.0

    plates = eligible_plates(library, patient)
    region = patient["region"]
    fav = {m: _favourites(plates, m, region, mc["n_favourites"][m], mc["region_match_weight"], r)
           for m in ("breakfast", "lunch", "dinner", "snack")}
    heavy = [p for p in plates if p["id"] in heavy_ids and ("lunch" in p["meal"] or "dinner" in p["meal"])]
    dessert = [p for p in plates if "dessert" in p["meal"]]
    buttermilk = [p for p in plates if p["id"] == "buttermilk"]

    # ---------------- daily hidden state: stress and illness ----------------
    st = cfg["stress"]
    stress = np.zeros(n_days)
    for d in range(n_days):
        prev = stress[d - 1] if d else r.normal()
        stress[d] = st["ar_rho"] * prev + np.sqrt(1 - st["ar_rho"] ** 2) * r.normal()
    stress += np.where(weekend, 0.0, st["weekday_offset"])
    illness = np.zeros(n_days, dtype=bool)
    d = 0
    while d < n_days:
        if r.random() < st["p_illness_start"]:
            k = int(r.integers(st["illness_days"][0], st["illness_days"][1] + 1))
            illness[d : d + k] = True
            d += k
        d += 1

    # ---------------- sleep: nights -1 .. n_days-1 ----------------
    stage = np.full(T, STAGE_OUT, dtype=np.int8)
    wake_min = np.zeros(n_days, dtype=int)       # minute of day each day starts (wake time)
    sleep_hours = np.zeros(n_days)
    night_rows = []
    for k in range(-1, n_days):
        free_morning = k + 1 < n_days and bool(weekend[k + 1])   # Friday and Saturday nights
        bed = bedtime_mean + r.normal(0, sc["bedtime_day_sd_min"]) + (sc["weekend_bedtime_shift_min"] if free_morning else 0)
        dur_h = sleep_need + r.normal(0, sc["duration_day_sd_h"]) + (sc["weekend_extra_h"] if free_morning else 0)
        if r.random() < sc["p_short_night"] or (k >= 0 and stress[k] > 1.5 and r.random() < 0.3):
            dur_h -= sc["short_night_loss_h"]
        dur_h = float(np.clip(dur_h, 3.0, 10.0))
        latency = int(max(2, r.normal(sc["latency_mean_min"], 8)))
        bed_abs = int(k * 1440 + bed)
        onset = bed_abs + latency
        n_sleep = int(dur_h * 60 * 1.07)          # in-bed span after onset, incl. brief awakenings
        hyp = _hypnogram(n_sleep, age, cfg, r)
        wake_abs = onset + n_sleep
        lo, hi = max(bed_abs, 0), min(wake_abs, T)
        if hi > lo:
            if onset > lo:
                stage[lo:min(onset, T)] = STAGE_WAKE
            s0 = max(onset, 0)
            if hi > s0:
                stage[s0:hi] = hyp[s0 - onset : hi - onset]
        asleep = hyp != STAGE_WAKE
        if k + 1 < n_days:
            wake_min[k + 1] = wake_abs - (k + 1) * 1440
            sleep_hours[k + 1] = asleep.sum() / 60.0
        if k >= 0:
            night_rows.append({
                "patient_id": pid, "night_of": dates[k].date(), "bedtime_min": bed_abs, "wake_min": wake_abs,
                "total_sleep_min": int(asleep.sum()), "deep_min": int((hyp == STAGE_DEEP).sum()),
                "rem_min": int((hyp == STAGE_REM).sum()), "light_min": int((hyp == STAGE_LIGHT).sum()),
                "awake_min": int((hyp == STAGE_WAKE).sum()) + latency,
                "efficiency": round(float(asleep.sum() / (n_sleep + latency)), 3),
            })

    # ---------------- meals ----------------
    meal_rows = []

    def eat(day: int, minute_of_day: float, meal: str, plate: dict, portion: float, out: bool):
        minute = int(day * 1440 + minute_of_day)
        if not (0 <= minute < T):
            return
        meal_rows.append({
            "patient_id": pid, "day": day, "minute": minute, "meal_type": meal, "plate_id": plate["id"],
            "plate_name": plate["name"], "portion": round(portion, 2),
            "carbs_g": round(plate["carbs"] * portion, 1), "protein_g": round(plate["protein"] * portion, 1),
            "fat_g": round(plate["fat"] * portion, 1), "fibre_g": round(plate["fibre"] * portion, 1),
            "gi": plate["gi"], "eating_out": out,
        })

    def choose(meal: str) -> dict | None:
        plist, pref = fav[meal]
        if not plist:
            return None
        return plist[r.choice(len(plist), p=pref)]

    skip_p = mc["p_skip_breakfast"]["skipper" if patient["skips_breakfast"] else "other"]
    lunch_times = np.zeros(n_days)
    dinner_times = np.zeros(n_days)
    for d in range(n_days):
        wk = weekend[d]
        portion_day = mc["weekend_portion_factor"] if wk else 1.0
        wake = wake_min[d]
        out_today = r.random() < mc["p_eating_out"]["weekend" if wk else "weekday"]
        # breakfast (when ill, eaten half the time)
        if (not illness[d] and r.random() >= skip_p) or (illness[d] and r.random() < 0.5):
            t = habit_times["breakfast"] + (mc["breakfast"]["weekend_shift_min"] if wk else 0) + r.normal(0, mc["breakfast"]["day_sd_min"])
            t = max(t, wake + 20)
            p = choose("breakfast")
            if p:
                eat(d, t, "breakfast", p, appetite * portion_day * np.exp(r.normal(0, mc["portion_meal_sd"])), False)
        # lunch
        t = habit_times["lunch"] + (mc["lunch"]["weekend_shift_min"] if wk else 0) + r.normal(0, mc["lunch"]["day_sd_min"])
        lunch_out = out_today and wk and r.random() < 0.5
        p = r.choice(heavy) if (lunch_out and heavy) else choose("lunch")
        if p:
            f = mc["eating_out_portion_factor"] if lunch_out else 1.0
            eat(d, t, "lunch", p, appetite * portion_day * f * np.exp(r.normal(0, mc["portion_meal_sd"])), bool(lunch_out))
            if buttermilk and r.random() < mc["p_lunch_buttermilk"]:
                eat(d, t + 5, "snack", buttermilk[0], 1.0, False)
        lunch_times[d] = t
        # evening snack
        if r.random() < p_snack:
            t = habit_times["snack"] + r.normal(0, mc["snack"]["day_sd_min"])
            p = choose("snack")
            if p:
                eat(d, t, "snack", p, np.exp(r.normal(0, mc["portion_meal_sd"])), False)
        # dinner
        t = habit_times["dinner"] + (mc["dinner_weekend_shift_min"] if wk else 0) + r.normal(0, mc["dinner_day_sd_min"])
        dinner_out = out_today and not lunch_out
        p = r.choice(heavy) if (dinner_out and heavy) else choose("dinner")
        if p:
            f = mc["eating_out_portion_factor"] if dinner_out else 1.0
            eat(d, t, "dinner", p, appetite * portion_day * f * np.exp(r.normal(0, mc["portion_meal_sd"])), bool(dinner_out))
            if dinner_out and dessert and r.random() < mc["p_dessert_when_out"]:
                eat(d, t + 30, "dessert", dessert[0], 1.0, True)
        dinner_times[d] = t
        if r.random() < mc["p_late_snack"]:
            p = choose("snack")
            if p:
                eat(d, t + r.uniform(90, 150), "snack", p, np.exp(r.normal(0, mc["portion_meal_sd"])), False)

    meals = pd.DataFrame(meal_rows)

    # ---------------- steps ----------------
    steps = np.zeros(T, dtype=np.int16)
    target_base = patient["typical_daily_steps"] * activity_factor
    walk_habit = bool(patient["morning_walk_habit"])
    pm_habit = bool(patient["post_meal_walk_habit"])
    daily_steps_target = np.zeros(n_days)
    for d in range(n_days):
        wk = weekend[d]
        target = target_base * np.exp(r.normal(0, ac["day_total_sd_log"]))
        target *= ac["weekend_factor"]["employed" if employed else "not_employed"] if wk else 1.0
        if illness[d]:
            target *= 0.5
        daily_steps_target[d] = target
        base = d * 1440
        structured = 0
        mw = ac["morning_walk"]
        if r.random() < (mw["p_day_if_habit"] if walk_habit else mw["p_day_otherwise"]) and not illness[d]:
            t = max(_clock_min(mw["time"]) + r.normal(0, mw["time_sd_min"]), wake_min[d] + 5)
            structured += _add_bout(steps, int(base + t), int(r.uniform(*mw["duration_min"])), tuple(mw["cadence"]), r)
        if employed and not wk:
            cm = ac["commute"]
            for ct in cm["times"]:
                structured += _add_bout(steps, int(base + _clock_min(ct) + r.normal(0, 15)), int(r.uniform(*cm["duration_min"])), tuple(cm["cadence"]), r)
        pw = ac["post_meal_walk"]
        for meal_time in (lunch_times[d], dinner_times[d]):
            if r.random() < (pw["p_if_habit"] if pm_habit else pw["p_otherwise"]):
                t = meal_time + r.uniform(*pw["delay_min"])
                structured += _add_bout(steps, int(base + t), int(r.uniform(*pw["duration_min"])), tuple(pw["cadence"]), r)
        # incidental bouts across waking minutes, weighted towards late morning and evening
        remaining = max(target - structured, 600)
        mean_bout = ac["incidental_bout_mean_min"] * np.mean(ac["incidental_cadence"])
        n_bouts = r.poisson(remaining / mean_bout)
        day_slice = slice(base, base + 1440)
        awake_mask = stage[day_slice] == STAGE_OUT
        mod = np.arange(1440) / 60.0
        prof = 0.6 + np.exp(-0.5 * ((mod - 10.5) / 2.0) ** 2) + np.exp(-0.5 * ((mod - 18.5) / 2.0) ** 2)
        w = prof * awake_mask
        if w.sum() > 0 and n_bouts > 0:
            starts = r.choice(1440, size=n_bouts, p=w / w.sum())
            for s0 in starts:
                _add_bout(steps, int(base + s0), int(r.geometric(1 / ac["incidental_bout_mean_min"])), tuple(ac["incidental_cadence"]), r)
    steps[stage >= STAGE_WAKE] = 0  # no steps while in bed

    # ---------------- heart rate ----------------
    hc = cfg["heart"]
    level = ACTIVITY_LEVEL[patient["activity_level"]]
    rhr = hc["rhr_mean"] + r.normal(0, hc["rhr_between_sd"]) + hc["rhr_bmi_per_unit"] * (patient["bmi"] - 25) + hc["rhr_activity_per_level"] * level
    minute_day = np.arange(T) // 1440
    tod = (np.arange(T) % 1440) / 60.0
    hr = np.full(T, rhr, dtype=float)
    in_bed = stage >= STAGE_WAKE
    hr += np.where(in_bed, 0.0, hc["awake_offset"])
    for code, key in ((STAGE_LIGHT, "light"), (STAGE_DEEP, "deep"), (STAGE_REM, "rem"), (STAGE_WAKE, "wake")):
        hr[stage == code] += hc["sleep_offsets"][key]
    # exertion responds with a short lag
    hr += hc["steps_gain_per_spm"] * lfilter([0.3], [1, -0.7], steps.astype(float))
    hr += hc["stress_bpm_per_sd"] * stress[minute_day] + hc["illness_bpm"] * illness[minute_day]
    hr += hc["circadian_amp"] * np.sin(2 * np.pi * (tod - 10) / 24)
    noise = lfilter([np.sqrt(1 - hc["noise_rho"] ** 2)], [1, -hc["noise_rho"]], r.normal(0, hc["noise_sd"], T))
    hr = np.clip(hr + noise, 38, 190).astype(np.float32)

    # ---------------- HRV (RMSSD) during sleep, 5-min epochs ----------------
    hv = cfg["hrv"]
    base_log = (np.log(hv["rmssd_ref_ms"]) + hv["age_per_10y_log"] * (age - 50) / 10
                + hv["duration_per_yr_log"] * (patient["duration_yr"] - 6)
                + hv["neuropathy_log"] * bool(patient["peripheral_neuropathy"]) + r.normal(0, hv["between_sd_log"]))
    # A night belongs to the day that follows it (the day whose insulin sensitivity it affects).
    night_level = (base_log + hv["stress_per_sd_log"] * stress + hv["illness_log"] * illness
                   + hv["sleep_hours_log"] * (sleep_hours - 7) + r.normal(0, hv["night_sd_log"], n_days))
    epochs = stage[: T - T % 5].reshape(-1, 5)
    counts = np.stack([(epochs == c).sum(1) for c in (STAGE_WAKE, STAGE_LIGHT, STAGE_DEEP, STAGE_REM)], axis=1)
    keep = counts.sum(1) >= 4
    dom = counts.argmax(1)  # 0 wake, 1 light, 2 deep, 3 rem
    factor = np.log(np.array([hv["stage_factor"][k] for k in ("wake", "light", "deep", "rem")]))
    t0 = np.arange(len(epochs)) * 5
    day = np.minimum((t0 + 12 * 60) // 1440, n_days - 1)
    rmssd = np.exp(night_level[day] + factor[dom] + r.normal(0, hv["epoch_sd_log"], len(epochs)))
    hrv = pd.DataFrame({"patient_id": pid, "minute": t0[keep], "rmssd_ms": np.round(rmssd[keep], 1)})

    # ---------------- medication taking ----------------
    mt = cfg["medication_taking"]
    dose_rows = []
    for m in meds.itertuples(index=False):
        times = [_clock_min(x) for x in m.times.split(";")]
        for d in range(n_days):
            for tm in times:
                p_take = adherence_true - (mt["weekend_adherence_drop"] if weekend[d] else 0) - (mt["evening_adherence_drop"] if tm >= 18 * 60 else 0)
                if r.random() < p_take:
                    minute = int(d * 1440 + tm + r.normal(0, mt["time_sd_min"]))
                    if 0 <= minute < T:
                        dose_rows.append({"patient_id": pid, "minute": minute, "drug": m.drug, "drug_class": m.drug_class,
                                          "daily_dose": m.daily_dose, "dose_unit": m.dose_unit, "n_times": len(times)})
    med_doses = pd.DataFrame(dose_rows, columns=["patient_id", "minute", "drug", "drug_class", "daily_dose", "dose_unit", "n_times"])

    # ---------------- meal log (observed) ----------------
    ml = cfg["meal_log"]
    log_rows = []
    for m in meals.itertuples(index=False):
        if r.random() < log_rate:
            est = m.carbs_g * np.exp(ml["carb_estimate_bias_log"] + r.normal(0, ml["carb_estimate_sd_log"]))
            log_rows.append({"patient_id": pid, "minute": int(m.minute + round(r.normal(0, ml["time_error_sd_min"]))),
                             "meal_type": m.meal_type, "plate_name": m.plate_name, "carbs_g_est": float(5 * round(est / 5))})
    meal_log = pd.DataFrame(log_rows, columns=["patient_id", "minute", "meal_type", "plate_name", "carbs_g_est"])

    daily = pd.DataFrame({
        "patient_id": pid, "day": np.arange(n_days), "date": [dt.date() for dt in dates], "weekend": weekend,
        "sleep_hours": np.round(sleep_hours, 2), "stress": np.round(stress, 3), "illness": illness,
        "steps_true": [int(steps[d * 1440 : (d + 1) * 1440].sum()) for d in range(n_days)],
    })
    habits = pd.DataFrame([{
        "patient_id": pid, "employed": employed, "appetite": round(appetite, 3), "activity_factor": round(activity_factor, 3),
        "p_snack": round(p_snack, 3), "sleep_need_h": round(sleep_need, 2), "bedtime_mean_min": round(bedtime_mean, 1),
        "dinner_time_actual_min": round(habit_times["dinner"], 1), "meal_log_rate": round(log_rate, 3),
        "n_eligible_plates": len(plates),
    }])
    return {"meals": meals, "meal_log": meal_log, "med_doses": med_doses, "daily": daily, "nights": pd.DataFrame(night_rows),
            "habits": habits, "hrv": hrv, "steps": steps, "sleep_stage": stage, "heart_rate": hr}


def generate_behaviour(patients: pd.DataFrame, medications: pd.DataFrame, truth: pd.DataFrame, library_cfg: dict,
                       cfg: dict, n_days: int, start: pd.Timestamp, seed: int) -> BehaviourData:
    library = library_cfg["plates"]
    heavy_ids = set(library_cfg.get("heavy_meal_ids", []))
    adh = truth.set_index("patient_id")["adherence_true"]
    parts = []
    for _, p in patients.iterrows():
        meds = medications[medications["patient_id"] == p["patient_id"]]
        parts.append(generate_patient(p, meds, float(adh[p["patient_id"]]), library, heavy_ids, cfg, n_days, start, seed))

    def cat(key: str) -> pd.DataFrame:
        frames = [x[key] for x in parts if len(x[key])]
        return pd.concat(frames, ignore_index=True) if frames else parts[0][key]

    return BehaviourData(
        meals=cat("meals"), meal_log=cat("meal_log"), med_doses=cat("med_doses"), daily=cat("daily"),
        nights=cat("nights"), habits=cat("habits"), hrv=cat("hrv"),
        steps=np.stack([x["steps"] for x in parts]), sleep_stage=np.stack([x["sleep_stage"] for x in parts]),
        heart_rate=np.stack([x["heart_rate"] for x in parts]),
    )
