"""T2D digital twin: doctor-facing dashboard.

    uv run streamlit run dashboard/app.py

Tabs: Patient (header, alert, 24 h + 2 h forecast, wearable strip, why, record, twin state),
What-if (personalised ODE twin with constraint checks and a hypoglycaemia check), Triage
(patients ranked by predicted risk), About this twin (evaluation, provenance, limitations).
"""

from __future__ import annotations

import json
import sys
import time as _time
from datetime import datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import charts, service  # noqa: E402

st.set_page_config(page_title="T2D Digital Twin", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 1rem; max-width: 1500px;}
.banner {background:#fff4e0; border:1px solid #f0c98a; color:#6b4500; padding:6px 14px; border-radius:6px; font-weight:600; margin-bottom:8px; font-size:15px;}
.chip {display:inline-block; padding:3px 11px; border-radius:14px; margin:2px 4px 2px 0; background:#eef2f7; color:#1f2937; font-size:14px;}
.chip-red {background:#c0392b; color:#fff; font-weight:600;}
.chip-amber {background:#c77c0e; color:#fff; font-weight:600;}
.chip-green {background:#1e8449; color:#fff; font-weight:600;}
.chip-allergy {background:#fde8e6; color:#9b1c1c; border:1px solid #f5b5ae; font-weight:600;}
.chip-warn {background:#fff4e0; color:#6b4500; border:1px solid #f0c98a;}
.header {font-size:22px; font-weight:700; margin-bottom:2px;}
.annot {font-size:18px; padding:8px 12px; border-left:5px solid #2f6fb0; background:#f5f8fc; border-radius:4px; margin:6px 0 4px 0;}
.card {border:1px solid #e5e7eb; border-radius:8px; padding:10px 12px; background:#fff; margin-bottom:10px;}
.card h4 {margin:0 0 6px 0; font-size:15px; color:#374151;}
.kv {font-size:14px; line-height:1.6;}
.muted {color:#6b7280; font-size:13px;}
.driver {display:flex; align-items:center; gap:8px; margin:4px 0; font-size:15px;}
.bar {height:10px; border-radius:3px;}
.pct {height:8px; background:#e5e7eb; border-radius:4px; position:relative; margin:3px 0 8px 0;}
.pct span {position:absolute; top:-3px; width:4px; height:14px; background:#2f6fb0; border-radius:2px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="banner">Synthetic patient. Decision support only.</div>', unsafe_allow_html=True)

missing = service.missing_cache()
if missing:
    st.error("The demo cache is missing (" + ", ".join(missing) + "). Build it with:")
    st.code("uv run python -m data_gen.generate\nuv run python -m models.features\nuv run python -m models.train\nuv run python -m dashboard.prepare", language="bash")
    st.stop()


@st.cache_resource(show_spinner="Loading the synthetic patients and models...")
def data() -> service.DemoData:
    return service.load()


dd = data()


@st.cache_data(show_spinner=False, max_entries=200)
def explain(pid: str, t: str) -> dict:
    f = service.at(dd.features, pid, pd.Timestamp(t))
    return service.explanation(dd, f)


@st.cache_data(show_spinner=False, max_entries=20)
def triage(t: str, thr: float) -> pd.DataFrame:
    return service.triage(dd, pd.Timestamp(t), thr)


@st.cache_resource(show_spinner=False, max_entries=20)
def observed(pid: str):
    return service.observed(dd, pid)


@st.cache_resource(show_spinner=False, max_entries=20)
def twin(pid: str):
    return service.twin_for(dd, pid)


@st.cache_data(show_spinner=False, max_entries=50)
def twin_state(pid: str, t: str) -> dict:
    from twin.whatif import state_at

    return state_at(twin(pid), observed(pid), pd.Timestamp(t), dd.cfgs["physiology"], dd.tcfg)


@st.cache_data(show_spinner=False, max_entries=20)
def card(pid: str) -> dict:
    from twin.card import twin_card

    return twin_card(twin(pid), observed(pid), dd.cfgs["physiology"])


# ------------------------------------------------------------------------------------------------
# navigation state
# ------------------------------------------------------------------------------------------------

scenes = {s["title"]: s for s in dd.scenes}
FREE = "Free exploration"


def set_time(t: pd.Timestamp) -> None:
    st.session_state["day"] = t.date()
    st.session_state["clock"] = t.time()


def on_scene() -> None:
    s = scenes.get(st.session_state["scene"])
    if s:
        st.session_state["pid"] = s["patient_id"]
        set_time(pd.Timestamp(s["timestamp"]))


def shift(minutes: int) -> None:
    t = datetime.combine(st.session_state["day"], st.session_state["clock"]) + timedelta(minutes=minutes)
    set_time(pd.Timestamp(t))


def open_patient(pid: str) -> None:
    st.session_state["pid"] = pid
    st.session_state["scene"] = FREE


if "pid" not in st.session_state:
    st.session_state["scene"] = next(iter(scenes), FREE)
    first = dd.scenes[0] if dd.scenes else {"patient_id": dd.patient_ids[0], "timestamp": str(service.times(dd, dd.patient_ids[0])[300])}
    st.session_state["pid"] = first["patient_id"]
    set_time(pd.Timestamp(first["timestamp"]))

if "_goto" in st.session_state:          # a time jump requested after the widgets were drawn (replay)
    set_time(pd.Timestamp(st.session_state.pop("_goto")))

with st.sidebar:
    st.markdown("### T2D digital twin")
    st.selectbox("Demo scene", list(scenes) + [FREE], key="scene", on_change=on_scene)
    sc = scenes.get(st.session_state["scene"])
    if sc:
        st.caption(sc["story"])
    else:
        st.selectbox("Patient", dd.patient_ids, key="pid")
    pid = st.session_state["pid"]
    ts = service.times(dd, pid)
    days = sorted({t.date() for t in ts})
    if st.session_state["day"] not in days:
        set_time(ts[len(ts) // 2])
    st.selectbox("Day", days, key="day", format_func=lambda d: d.strftime("%a %d %b"))
    st.slider("Time", min_value=time(0, 0), max_value=time(23, 55), step=timedelta(minutes=5), key="clock", format="HH:mm")
    c1, c2, c3 = st.columns(3)
    c1.button("−1 h", on_click=shift, args=(-60,), width="stretch")
    c2.button("+15 min", on_click=shift, args=(15,), width="stretch")
    c3.button("+1 h", on_click=shift, args=(60,), width="stretch")
    play = st.button("▶ Replay the next 2 hours", width="stretch")
    st.toggle("Show what actually happened (replay)", key="replay", value=False)
    st.divider()
    thr = st.slider("Alert sensitivity (probability threshold)", 0.2, 0.7, float(dd.predictor.threshold), 0.05,
                    help="Lower = catch more spikes, more false alerts. Higher = fewer, more certain alerts.")
    op = service.operating_point(dd, thr)
    if op:
        st.caption(f"On held-out test patients at {op['threshold']:.2f}: {100 * op['spikes_caught']:.0f} % of spikes caught, "
                   f"median warning {op['lead']:.0f} min ahead, {op['false_per_day']:.2f} false alerts per patient-day.")

now = pd.Timestamp(datetime.combine(st.session_state["day"], st.session_state["clock"]))
patient = service.patient_row(dd, pid)
meds = dd.tables["medications"][dd.tables["medications"]["patient_id"] == pid]


# ------------------------------------------------------------------------------------------------
# pieces
# ------------------------------------------------------------------------------------------------

def header(snap: dict, t: pd.Timestamp | None = None) -> str:
    p = patient
    glucose_meds = [m.drug.replace("_", " ") for m in meds.itertuples() if m.drug_class in
                    ("biguanide", "sulfonylurea", "dpp4_inhibitor", "sglt2_inhibitor", "basal_insulin")]
    trend = "↑" if p["hba1c_trend_12m"] > 0.2 else "↓" if p["hba1c_trend_12m"] < -0.2 else "→"
    chips = [f'<span class="chip-{snap["colour"]} chip">{snap["label"]}</span>',
             f'<span class="chip">{int(p["age_yr"])} y {"male" if p["sex"] == "M" else "female"}</span>',
             f'<span class="chip">Type 2 diabetes, {p["duration_yr"]:.0f} y</span>',
             f'<span class="chip">HbA1c {p["hba1c_pct"]:.1f} % {trend}</span>',
             f'<span class="chip">eGFR {int(p["egfr"])}</span>',
             f'<span class="chip">{", ".join(glucose_meds) or "no glucose-lowering drugs"}</span>']
    return f'<div class="header">{pid} · synthetic patient · {(t or now):%a %d %b %H:%M}</div>' + "".join(chips)


def allergy_chips(p) -> str:
    items = [f"{a.replace('_', ' ')} allergy" for a in str(p["food_allergies"] or "").split(";") if a]
    items += [f"{a.replace('_', ' ')} allergy" for a in str(p["drug_allergies"] or "").split(";") if a]
    items += [f"{a.replace('_', ' ')} intolerance" for a in str(p["intolerances"] or "").split(";") if a]
    return "".join(f'<span class="chip chip-allergy">⚠ {i}</span>' for i in items) or '<span class="muted">No allergies or intolerances recorded</span>'


def record_card() -> None:
    p = patient
    other = [m.drug.replace("_", " ") + f" {m.daily_dose:g} {m.dose_unit}" for m in meds.itertuples()]
    comorb = [n for n, c in (("hypertension", "hypertension"), ("dyslipidaemia", "dyslipidaemia"), ("CAD", "coronary_artery_disease"),
                             ("CKD", "ckd"), ("hypothyroidism", "hypothyroidism"), ("neuropathy", "peripheral_neuropathy"), ("fatty liver", "fatty_liver")) if p[c]]
    st.markdown(
        f'<div class="card"><h4>Record</h4>{allergy_chips(p)}<div class="kv" style="margin-top:6px">'
        f'<b>{p["diet_type"].replace("_", "-").capitalize()}</b> diet · {p["region"]} India · {p["activity_level"].replace("_", " ")}<br>'
        f'BMI {p["bmi"]:.1f} · waist {p["waist_cm"]:.0f} cm<br>'
        f'HbA1c {p["hba1c_pct"]:.1f} % (12 months ago {p["hba1c_12m_ago_pct"]:.1f} %) · fasting glucose {p["fpg_mgdl"]:.0f}<br>'
        f'eGFR {int(p["egfr"])} · creatinine {p["creatinine_mgdl"]:.2f} · ALT {p["alt_ul"]:.0f}<br>'
        f'BP {p["sbp_mmhg"]:.0f}/{p["dbp_mmhg"]:.0f} · LDL {p["ldl_mgdl"]:.0f} · HDL {p["hdl_mgdl"]:.0f} · TG {p["tg_mgdl"]:.0f}<br>'
        f'<b>Medicines:</b> {"; ".join(other)}<br><span class="muted">reported adherence {100 * p["adherence_reported"]:.0f} %</span><br>'
        f'<b>Conditions:</b> {", ".join(comorb) or "none recorded"}</div></div>', unsafe_allow_html=True)
    st.caption("HbA1c over the last 12 months")
    st.plotly_chart(charts.sparkline(service.hba1c_history(dd, pid)), width="stretch", config={"displayModeBar": False}, key="spark")


def ordinal(x: float) -> str:
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def twin_card_view() -> None:
    if not service.pickle_path(pid).exists():
        st.markdown('<div class="card"><h4>Twin state</h4><span class="muted">This patient\'s twin is not personalised yet.</span></div>', unsafe_allow_html=True)
        if st.button("Personalise the twin (about 20 s)", key="fit_left"):
            with st.spinner("Fitting the ODE twin to the first 7 days of this patient's data..."):
                twin(pid)
            st.rerun()
        return
    c = card(pid)
    rows = ""
    for r in c["rows"]:
        pct = r["percentile"] if r["percentile"] is not None else 50
        rows += (f'<div class="kv">{r["label"]}: <b>{r["band"]}</b> vs similar patients '
                 f'<span class="muted">({ordinal(pct)} percentile, confidence {r["confidence"]})</span></div>'
                 f'<div class="pct"><span style="left:{min(max(pct, 1), 99)}%"></span></div>')
    sm = c["standard_meal"]
    st.markdown(f'<div class="card"><h4>Twin state (personalised ODE model)</h4>{rows}'
                f'<div class="kv">A standard 60 g carbohydrate meal: <b>+{sm["rise"]:.0f} mg/dL</b>, peaking after {sm["time_to_peak_min"]} min</div>'
                f'<div class="kv">Fasting set point {c["fasting_set_point_mgdl"]:.0f} mg/dL · liver glucose output {c["hepatic_output_mg_kg_min"]:.1f} mg/kg/min</div>'
                f'<div class="muted" style="margin-top:4px">{c["fit_note"]}.</div></div>', unsafe_allow_html=True)


def why_panel(e: dict, snap: dict) -> None:
    mx = max([abs(d["mg_dl"]) for d in e["up"] + e["down"]] + [1])

    def row(d, up):
        col = "#c0392b" if up else "#1e8449"
        w = int(160 * abs(d["mg_dl"]) / mx)
        return (f'<div class="driver"><span style="width:70px;text-align:right;color:{col};font-weight:700">{"▲" if up else "▼"} {d["mg_dl"]:+.0f}</span>'
                f'<span class="bar" style="width:{w}px;background:{col}"></span><span>{d["text"]}</span></div>')

    ups = "".join(row(d, True) for d in e["up"]) or '<div class="muted">nothing notable</div>'
    downs = "".join(row(d, False) for d in e["down"]) or '<div class="muted">nothing notable</div>'
    st.markdown(f'<div class="card"><h4>Why this prediction</h4>'
                f'<div class="kv">Predicted 2-hour peak <b>{e["predicted_peak"]:.0f} mg/dL</b> vs {e["reference_peak"]:.0f} typical for similar patients '
                f'(same age band and diabetes duration). Contributions in mg/dL:</div>'
                f'<div style="margin-top:6px"><b>Pushing it up</b>{ups}</div><div style="margin-top:6px"><b>Pulling it down</b>{downs}</div>'
                f'<div class="muted" style="margin-top:6px">What the model relied on, not the effect of changing it. Use the What-if tab to test a change '
                f'on this patient\'s physiological twin.</div></div>', unsafe_allow_html=True)


def hero_figure(t: pd.Timestamp, snap: dict):
    t0 = t - pd.Timedelta(hours=24)
    cgm = service.window(dd, "cgm", pid, t0, t)
    meals = service.window(dd, "meal_log", pid, t0, t)
    steps = service.window(dd, "wearable_steps", pid, t0, t)
    hr = service.window(dd, "wearable_heart_rate", pid, t0, t)
    sl = service.window(dd, "wearable_sleep_stages", pid, t0, t)
    hv = service.window(dd, "wearable_hrv", pid, t0, t)
    fut = service.window(dd, "cgm", pid, t, t + pd.Timedelta(hours=2)) if st.session_state.get("replay") else None
    return charts.hero(cgm, snap["forecast"], meals, steps, hr, sl, hv, snap["time"], fut, snap["colour"])


# ------------------------------------------------------------------------------------------------
# tabs
# ------------------------------------------------------------------------------------------------

tab_patient, tab_whatif, tab_triage, tab_about = st.tabs(["Patient", "What-if", "Triage", "About this twin"])

with tab_patient:
    snap = service.snapshot(dd, pid, now, thr)
    head = st.empty()
    head.markdown(header(snap), unsafe_allow_html=True)
    annot = st.empty()
    dq = snap["quality"]
    annot.markdown(f'<div class="annot">{snap["annotation"]}</div>', unsafe_allow_html=True)
    if dq["messages"]:
        st.markdown(f'<span class="chip chip-warn">Data quality: {dq["level"]} · {"; ".join(dq["messages"])}</span>', unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="medium")
    with left:                            # drawn first so it stays visible during a replay
        record_card()
        twin_card_view()
    with right:
        chart = st.empty()
        chart.plotly_chart(hero_figure(now, snap), width="stretch", config={"displayModeBar": False}, key="hero")
        if play:
            for k in range(1, 9):
                t = now + pd.Timedelta(minutes=15 * k)
                s2 = service.snapshot(dd, pid, t, thr)
                head.markdown(header(s2, t), unsafe_allow_html=True)
                annot.markdown(f'<div class="annot">{s2["annotation"]}</div>', unsafe_allow_html=True)
                chart.plotly_chart(hero_figure(t, s2), width="stretch", config={"displayModeBar": False}, key=f"hero_play_{k}")
                _time.sleep(0.5)
            st.session_state["_goto"] = str(now + pd.Timedelta(hours=2))   # applied before the widgets on the next run
            st.rerun()
        why_panel(explain(pid, str(snap["time"])), snap)

with tab_whatif:
    from twin import constraints as C
    from twin import whatif as W

    st.markdown(f'<div class="header">What-if for {pid} at {now:%H:%M}</div>', unsafe_allow_html=True)
    if not service.pickle_path(pid).exists():
        st.info("The what-if simulator runs on this patient's personalised ODE twin, which has not been fitted yet.")
        if st.button("Personalise the twin (about 20 s)", key="fit_whatif"):
            with st.spinner("Fitting the ODE twin to the first 7 days of this patient's data..."):
                twin(pid)
            st.rerun()
    else:
        tw, obs = twin(pid), observed(pid)
        factors = W.patient_factors(patient, meds)
        st.markdown("Patient factors considered: " + "".join(f'<span class="chip">{f}</span>' for f in factors), unsafe_allow_html=True)
        offset, plate_id, portion, slot = service.next_meal(dd, pid, now)
        recent = service.recent_meal(dd, pid, now)
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            st.markdown("**Current plan**")
            if recent is not None:
                st.markdown(f'<span class="chip">Just eaten: {recent["plate_name"]}, about {recent["carbs_g_est"]:.0f} g carbohydrate logged at '
                            f'{pd.Timestamp(recent["timestamp"]):%H:%M} (included from the record)</span>', unsafe_allow_html=True)
            with_meal = st.toggle("Plan a next meal", value=offset is not None and recent is None, key=f"withmeal_{pid}_{now}")
            dish, meal_offset = None, 0
            slots = ["breakfast", "lunch", "snack", "dinner"]
            if with_meal:
                slot = st.selectbox("Meal", slots, index=slots.index(slot))
            options = C.meal_options(dd.library, patient, slot)
            if with_meal:
                ids = [p["id"] for p, v in options if v.allowed]
                dish = st.selectbox("Dish", ids, index=ids.index(plate_id) if plate_id in ids else 0, format_func=lambda i: dd.plates[i]["name"])
                portion = st.slider("Portion", 0.25, 2.0, float(portion), 0.05)
                meal_offset = st.slider("Starts in (minutes)", 0, 240, int(offset or 30), 5)
        with c2:
            st.markdown("**Change to test**")
            if with_meal:
                choices = ["Swap the dish", "Change the portion", "Walk after the meal", "Skip the meal", "Change a medicine"]
            else:
                choices = ["Walk now", "Change a medicine"]
            kind = st.segmented_control("Intervention", choices, default=choices[0], key=f"kind_{pid}_{now}_{with_meal}")
            iv, typical = None, False
            if kind == "Swap the dish":
                all_ids = [p["id"] for p, _ in options if p["id"] != dish]
                verdicts = {p["id"]: v for p, v in options}
                new = st.selectbox("Swap to", all_ids, format_func=lambda i: dd.plates[i]["name"] + ("   (blocked)" if not verdicts[i].allowed else "")
                                   + f"  · {dd.plates[i]['carbs']} g carbs, {dd.plates[i]['gi']} GI")
                iv = W.Intervention("swap_meal", plate_id=new, portion=1.0)
                blocked = [(p, v) for p, v in options if not v.allowed]
                if blocked:
                    with st.expander(f"{len(blocked)} dishes blocked for this patient"):
                        for p, v in blocked:
                            st.markdown(f'<span class="muted">⛔ {p["name"]}: {"; ".join(v.reasons)}</span>', unsafe_allow_html=True)
            elif kind == "Change the portion":
                iv = W.Intervention("scale_meal", factor=st.slider("New portion (x current)", 0.25, 1.5, 0.5, 0.05))
            elif kind in ("Walk after the meal", "Walk now"):
                iv = W.Intervention("add_walk", start_after_min=st.slider("Start after the meal (min)" if with_meal else "Start in (min)", 0, 60,
                                                                          20 if with_meal else 0, 5),
                                    duration_min=st.slider("Duration (min)", 10, 45, 20, 5), cadence_spm=100)
            elif kind == "Skip the meal":
                iv = W.Intervention("skip_meal")
            elif kind == "Change a medicine":
                drugs = sorted(C.DRUG_CLASS)
                cur = {m.drug: m.daily_dose for m in meds.itertuples()}
                drug = st.selectbox("Medicine", drugs, format_func=lambda d: d.replace("_", " ") + (f" (now {cur[d]:g}/day)" if d in cur else " (not taking)"))
                default = float(cur.get(drug, {"metformin": 1000, "glimepiride": 1, "gliclazide_mr": 60, "teneligliptin": 20, "sitagliptin": 100,
                                                  "vildagliptin": 100, "dapagliflozin": 10, "empagliflozin": 10, "insulin_glargine": 10}[drug]))
                dose = st.number_input("New total daily dose (mg, or units for insulin)", 0.0, 3000.0, default, step=1.0 if default < 50 else 250.0)
                typical = st.toggle("Compare a typical day at steady state (recommended for drug changes)", value=True)
                v = C.check_drug_change(patient, drug, dose, dd.tcfg["constraints"], meds)
                st.markdown({"allowed": "✅ Allowed", "warn": "⚠️ Allowed with caution", "blocked": "⛔ Blocked"}[v.status]
                            + (": " + "; ".join(v.reasons) if v.reasons else ""))
                iv = W.Intervention("change_dose", drug=drug, new_daily_dose=dose)
            run = st.button("Simulate on the twin", type="primary")
        if run and iv is not None:
            plan = W.Plan(meals=[W.PlannedMeal(meal_offset, dish, portion)] if with_meal else [])
            with st.spinner("Simulating..."):
                if typical:
                    res = W.typical_day(tw, obs, {iv.drug: iv.new_daily_dose}, dd.cfgs["physiology"], dd.tcfg)
                else:
                    state = twin_state(pid, str(now))
                    res = W.whatif(tw, obs, now, plan, iv, dd.cfgs["physiology"], dd.tcfg, dd.library, state=state)
            st.divider()
            if typical:
                v = res["verdicts"][iv.drug]
                cur, chg = res["current"], res.get("changed")
                if chg is None:
                    st.error("⛔ Not simulated: " + "; ".join(v.reasons))
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Mean glucose", f"{chg['outcome'].mean:.0f} mg/dL", f"{chg['outcome'].mean - cur['outcome'].mean:+.0f}", delta_color="inverse")
                    m2.metric("Time in range 70-180", f"{100 * chg['tir']:.0f} %", f"{100 * (chg['tir'] - cur['tir']):+.0f} points")
                    m3.metric("Lowest glucose", f"{chg['outcome'].trough:.0f} mg/dL", f"{chg['outcome'].trough - cur['outcome'].trough:+.0f}")
                    m4.metric("HbA1c change (rough)", f"{chg['est_hba1c'] - cur['est_hba1c']:+.1f} points")
                    st.plotly_chart(charts.whatif(cur["outcome"], chg["outcome"], cur["outcome"].trajectory["timestamp"].iat[0],
                                                  iv.describe(dd.plates), "current regimen"), width="stretch", config={"displayModeBar": False})
                    st.caption("Typical day from this patient's own meal log, day 3 at steady state; x-axis in hours from midnight. "
                               "Changes are more reliable than absolute levels.")
                safety = res["safety"]
            else:
                if res.alt is None:
                    st.error(f"⛔ {res.description}: not simulated. " + "; ".join(res.verdict.reasons))
                else:
                    if res.verdict.reasons:
                        st.warning("Caution: " + "; ".join(res.verdict.reasons))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("5-hour peak", f"{res.alt.peak:.0f} mg/dL", f"{res.delta_peak:+.0f}", delta_color="inverse")
                    m2.metric("Minutes above 180", f"{res.alt.minutes_above_180:.0f}", f"{res.alt.minutes_above_180 - res.base.minutes_above_180:+.0f}",
                              delta_color="inverse")
                    m3.metric("Lowest glucose", f"{res.alt.trough:.0f} mg/dL", f"{res.alt.trough - res.base.trough:+.0f}")
                    m4.metric("Mean glucose", f"{res.alt.mean:.0f} mg/dL", f"{res.alt.mean - res.base.mean:+.0f}", delta_color="inverse")
                st.plotly_chart(charts.whatif(res.base, res.alt, now, res.description), width="stretch", config={"displayModeBar": False})
                safety = res.safety
            msg = safety["message"] + (f" · risk factors: {', '.join(safety['risk_factors'])}" if safety["risk_factors"] else "")
            {"hypo": st.error, "caution": st.warning, "ok": st.success}[safety["level"]](msg)
            st.caption(dd.tcfg["whatif"]["label"] + " Answers come from the patient's personalised ODE twin, not from the prediction model.")

with tab_triage:
    st.markdown(f'<div class="header">Triage at {now:%a %d %b %H:%M}</div>', unsafe_allow_html=True)
    st.caption("All synthetic test patients at this clinic time. Predicted lows first, then the highest predicted 2-hour peak.")
    with st.spinner("Scoring patients..."):
        tri = triage(str(now), thr)
    colours = {"red": "#fde8e6", "amber": "#fff4e0", "green": "#e8f5ec"}

    def style(row):
        return [f"background-color: {colours[row['colour']]}" if c == "status" else "" for c in row.index]

    shown = tri.drop(columns=["colour"])
    st.dataframe(tri.style.apply(style, axis=1), width="stretch", hide_index=True, height=520,
                 column_order=list(shown.columns),
                 column_config={"spike probability": st.column_config.ProgressColumn("spike probability", format="percent", min_value=0, max_value=1)})
    c1, c2 = st.columns([2, 5])
    choice = c1.selectbox("Open a patient", tri["patient"], key="triage_pick")
    c2.button("Open in the Patient tab", on_click=open_patient, args=(choice,))

with tab_about:
    res = service.config.REPO_ROOT / "results"

    def load_json(p):
        p = res / p
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    ms, vs, tw_s, ex = load_json("models/summary.json"), load_json("validation/summary.json"), load_json("twin/summary.json"), load_json("explain/explanation_checks.json")
    st.markdown('<div class="header">About this twin</div>', unsafe_allow_html=True)
    st.markdown("A proof-of-concept digital twin of people with Type 2 diabetes, built entirely on **synthetic patients**. It fuses a static record "
                "(demographics, labs, medicines, allergies, diet) with CGM, wearable and meal-log streams to predict glucose spikes about 2 hours ahead, "
                "explains the prediction, and lets a doctor test changes on a personalised physiological model. All numbers below come from held-out "
                "synthetic test patients (or held-out real CGM where stated). Full reports are in `docs/`.")
    a, b, c, d = st.columns(4)
    op = service.operating_point(dd, dd.predictor.threshold)
    if op:
        a.metric("Spikes caught (default threshold)", f"{100 * op['spikes_caught']:.0f} %", help="Held-out synthetic test patients")
        b.metric("Median warning time", f"{op['lead']:.0f} min")
        c.metric("False alerts per patient-day", f"{op['false_per_day']:.2f}")
    if ms:
        d.metric("Spike PR-AUC / Brier", f"{ms['classification_at_threshold']['pr_auc']:.3f} / {ms['classification_at_threshold']['brier']:.3f}")
    img = service.config.REPO_ROOT / "results"
    t1, t2, t3, t4 = st.tabs(["Fusion vs baselines", "Real-data validation", "What-if validation", "Limitations and provenance"])
    with t1:
        st.image(str(img / "models" / "ablation.png"), width="stretch")
        st.markdown("With hours of CGM history, the meal log is the stream that clearly improves prediction; wearables help right after a walk and "
                    "the record helps in a patient's first day. See `docs/model_report.md` for bootstrap intervals.")
        cc1, cc2 = st.columns(2)
        cc1.image(str(img / "models" / "calibration.png"), width="stretch")
        cc2.image(str(img / "models" / "alert_tradeoff.png"), width="stretch")
    with t2:
        st.markdown("The synthetic data were checked against real open T2D CGM (Shanghai T2DM, CGMacros). Real data were used only to calibrate a few "
                    "population settings and to validate, never to train the models.")
        tstr = pd.read_csv(img / "validation" / "tstr.csv") if (img / "validation" / "tstr.csv").exists() else pd.DataFrame()
        if len(tstr):
            t = tstr[(tstr["test"] == "Shanghai held-out") & tstr["features"].isin(["-", "CGM history only"])]
            st.dataframe(t[["train", "mae_30", "mae_60", "spike_pr_auc"]].rename(columns={"train": "trained on", "mae_30": "MAE 30 min", "mae_60": "MAE 60 min",
                                                                                   "spike_pr_auc": "spike PR-AUC"}).round(3), hide_index=True, width="stretch")
            st.caption("Train on synthetic, test on real (TSTR), CGM-only, real Shanghai patients held out from calibration.")
        st.image(str(img / "validation" / "matched_distributions.png"), width="stretch")
    with t3:
        st.image(str(img / "twin" / "counterfactuals.png"), width="stretch")
        if tw_s:
            rows = [{"intervention": k, "true change (median)": f"{v['true_delta_median']:+.0f}", "error: personalised twin": f"{v['mae_fitted']:.1f}",
                     "error: record-only twin": f"{v['mae_prior']:.1f}", "error: cohort average": f"{v['mae_cohort_average']:.1f}"}
                    for k, v in tw_s["counterfactuals"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("What-if answers compared with the true counterfactual from replaying each synthetic patient (mg/dL, change in the 5-hour peak).")
    with t4:
        st.markdown("""
**Limitations**
- Every patient is simulated. The simulator encodes our assumptions, and a model trained on it can partly rediscover them; results here are an upper bound for real patients.
- The twin's physiology model is the model that generated the data, so what-if validation shows best-case behaviour.
- The real CGM datasets used for validation are not Indian and are small; their glucose sits below the HbA1c-implied level (monitored, treated periods).
- Indian population values, meal macros and all drug and constraint rules are assumptions awaiting verification and clinical review.
- Decision support only: nothing here is a diagnosis or a prescription.

**Data provenance**
- Synthetic cohort: 400 patients x 14 days, generated from a fixed seed (`data_gen/`), with a data card and data dictionary in `docs/`.
- Real validation data (not redistributed): Shanghai T2DM (Zhao et al., Scientific Data 2023, CC BY 4.0) and CGMacros (PhysioNet, CC BY-NC-SA 4.0).
- No genetic data. No real patient data.
""")
