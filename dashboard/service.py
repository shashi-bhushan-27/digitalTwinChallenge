"""Everything the dashboard needs, behind plain functions (no Streamlit here).

The UI never touches model internals: it asks this module for a patient snapshot at a time,
the triage list, the twin and what-if results. Data come from the demo cache built by
`uv run python -m dashboard.prepare` (predictions for test patients, their features, a
reference pool for explanations, scripted demo scenes, fitted twins).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import config

DEMO = config.REPO_ROOT / "data" / "demo"
HORIZONS = [15, 30, 60, 90, 120]
STAGE_NAMES = {0: "awake", 1: "light", 2: "deep", 3: "REM"}


@dataclass
class DemoData:
    tables: dict            # observed tables for the demo patients
    features: pd.DataFrame  # 5-minute features of the demo patients
    preds: pd.DataFrame     # model outputs at every 5-minute prediction time
    scenes: list[dict]
    pool: pd.DataFrame      # reference pool for explanations (training patients)
    predictor: object
    cfgs: dict
    mcfg: dict
    tcfg: dict
    library: list[dict]
    sensitivity: pd.DataFrame

    @property
    def patient_ids(self) -> list[str]:
        return sorted(self.preds["patient_id"].unique())

    @property
    def plates(self) -> dict:
        return {p["id"]: p for p in self.library}


def missing_cache() -> list[str]:
    need = ["predictions.parquet", "features.parquet", "reference_pool.parquet", "scenes.json", "observed"]
    return [n for n in need if not (DEMO / n).exists()]


def load() -> DemoData:
    from data_gen.generate import load_configs
    from models.inference import TwinPredictor

    cfgs = load_configs()
    obs = {p.stem: pd.read_parquet(p) for p in (DEMO / "observed").glob("*.parquet")}
    sens_path = config.REPO_ROOT / "results" / "models" / "fusion_alert_sensitivity.csv"
    return DemoData(
        tables=obs, features=pd.read_parquet(DEMO / "features.parquet"), preds=pd.read_parquet(DEMO / "predictions.parquet"),
        scenes=json.loads((DEMO / "scenes.json").read_text(encoding="utf-8")), pool=pd.read_parquet(DEMO / "reference_pool.parquet"),
        predictor=TwinPredictor.load(), cfgs=cfgs, mcfg=config.load("models"), tcfg=config.load("twin"),
        library=config.load("meal_library")["plates"], sensitivity=pd.read_csv(sens_path) if sens_path.exists() else pd.DataFrame(),
    )


# ----------------------------------------------------------------------------------------------
# status, annotation, data quality
# ----------------------------------------------------------------------------------------------


def status(p: pd.Series, threshold: float) -> tuple[str, str]:
    """(colour, label) of the alert chip: red / amber / green only."""
    low = min(p[f"f{h}_lo"] for h in HORIZONS)
    if low < 70:
        return "red", "Low glucose risk"
    if bool(p["already_high"]):
        return "amber", "High now"
    if p["spike_prob"] >= threshold:
        return "red", "Spike alert"
    if p["spike_prob"] >= 0.3:
        return "amber", "Watch"
    return "green", "Stable"


def data_quality(f: pd.Series) -> dict:
    msgs = []
    miss = float(f.get("cgm_missing_2h", 0) or 0)
    if miss > 0.08:
        msgs.append(f"CGM gap: {miss * 120:.0f} min missing in the last 2 h")
    if pd.isna(f.get("wear_sleep_total_min")):
        msgs.append("no sleep data from last night")
    if float(f.get("wear_hr_missing_2h", 0) or 0) > 0.5:
        msgs.append("wearable off for most of the last 2 h")
    level = "reduced" if (miss > 0.08 or len(msgs) >= 2) else "minor" if msgs else "good"
    return {"level": level, "messages": msgs}


def uncertain(p: pd.Series, dq: dict) -> list[str]:
    """Reasons the model is unsure (shown to the doctor rather than hidden)."""
    why = []
    if 0.35 <= p["spike_prob"] <= 0.65 and not p["already_high"]:
        why.append(f"spike probability {100 * p['spike_prob']:.0f} % is close to a coin toss")
    width = p["f60_hi"] - p["f60_lo"]
    if width > 70:
        why.append(f"the 80 % range one hour ahead is wide ({p['f60_lo']:.0f}-{p['f60_hi']:.0f} mg/dL)")
    if dq["level"] == "reduced":
        why.append("input data are incomplete")
    return why


def crossing_minutes(p: pd.Series, level: float = 180.0) -> float | None:
    """First forecast time above `level` (linear between horizons), None if the point forecast stays below."""
    prev_t, prev_g = 0.0, float(p["cgm_now"])
    for h in HORIZONS:
        g = float(p[f"f{h}"])
        if g > level >= prev_g:
            return prev_t + (level - prev_g) / max(g - prev_g, 1e-6) * (h - prev_t)
        prev_t, prev_g = float(h), g
    return None


def annotation(p: pd.Series, threshold: float, dq: dict) -> str:
    rng = f"peak about {p['peak']:.0f} mg/dL (80 % range {p['peak_lo']:.0f}-{p['peak_hi']:.0f})"
    prob = f"{100 * p['spike_prob']:.0f} %"
    if p["already_high"]:
        txt = f"Glucose is above 180 now ({p['cgm_now']:.0f} mg/dL); {rng} over the next 2 h."
    elif p["spike_prob"] >= threshold:
        c = crossing_minutes(p)
        when = f"in about {5 * round(c / 5):.0f} min" if c is not None else "within 2 h"
        txt = f"Spike likely ({prob}): above 180 {when}, {rng}."
    elif p["spike_prob"] >= 0.3:
        txt = f"Possible spike ({prob}); {rng}."
    else:
        txt = f"Stable: no spike expected in the next 2 h ({prob}); {rng}."
    low = min(p[f"f{h}_lo"] for h in HORIZONS)
    if low < 70:
        txt += f" Low risk: the 80 % range dips to {low:.0f} mg/dL."
    why = uncertain(p, dq)
    if why:
        txt += " The model is uncertain here: " + "; ".join(why) + "."
    return txt


# ----------------------------------------------------------------------------------------------
# patient views
# ----------------------------------------------------------------------------------------------


def patient_row(dd: DemoData, pid: str) -> pd.Series:
    p = dd.tables["patients"]
    return p[p["patient_id"] == pid].iloc[0]


def times(dd: DemoData, pid: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dd.preds.loc[dd.preds["patient_id"] == pid, "timestamp"])


def at(df: pd.DataFrame, pid: str, t: pd.Timestamp) -> pd.Series:
    d = df[(df["patient_id"] == pid) & (df["timestamp"] <= t)]
    return d.iloc[-1] if len(d) else df[df["patient_id"] == pid].iloc[0]


def window(dd: DemoData, name: str, pid: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    d = dd.tables[name]
    d = d[d["patient_id"] == pid]
    return d[(d["timestamp"] >= t0) & (d["timestamp"] <= t1)]


def snapshot(dd: DemoData, pid: str, t: pd.Timestamp, threshold: float) -> dict:
    p = at(dd.preds, pid, t)
    f = at(dd.features, pid, t)
    dq = data_quality(f)
    colour, label = status(p, threshold)
    tp = pd.Timestamp(p["timestamp"])
    fc = pd.DataFrame({"timestamp": [tp] + [tp + pd.Timedelta(minutes=h) for h in HORIZONS],
                       "point": [p["cgm_now"]] + [p[f"f{h}"] for h in HORIZONS],
                       "lo": [p["cgm_now"]] + [p[f"f{h}_lo"] for h in HORIZONS],
                       "hi": [p["cgm_now"]] + [p[f"f{h}_hi"] for h in HORIZONS]})
    return {"pred": p, "features": f, "time": tp, "colour": colour, "label": label, "quality": dq,
            "annotation": annotation(p, threshold, dq), "forecast": fc, "uncertain": uncertain(p, dq)}


def explanation(dd: DemoData, feature_row: pd.Series) -> dict:
    from models.explain import Explainer

    return Explainer(dd.predictor, dd.pool, dd.mcfg["explain"], seed=0).explain(feature_row)


def hba1c_history(dd: DemoData, pid: str) -> pd.DataFrame:
    h = dd.tables["hba1c_history"]
    return h[h["patient_id"] == pid].sort_values("months_ago", ascending=False)


def triage(dd: DemoData, t: pd.Timestamp, threshold: float) -> pd.DataFrame:
    """All demo patients at clinic time t, ranked by severity (predicted lows first, then the highest
    predicted 2-hour peak), with a one-line reason."""
    from models.explain import global_contributions, reroute_missing, sentence

    rows, feats = [], []
    for pid in dd.patient_ids:
        p = at(dd.preds, pid, t)
        f = at(dd.features, pid, t)
        rows.append(p)
        feats.append(f)
    P, Fe = pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(feats).reset_index(drop=True)
    G = global_contributions(dd.predictor.peak["point"], Fe, dd.predictor.features)
    out = []
    for i in range(len(P)):
        p, f = P.iloc[i], Fe.iloc[i]
        g = reroute_missing(G.iloc[i].copy(), f)
        # time of day is shared by every patient at one clinic time, so it cannot say why this patient ranks high
        g = g.drop(labels=["data_quality", "time_of_day"], errors="ignore")
        top = g.idxmax() if g.max() > 1 else None
        reason = sentence(top, True, f) if top else "no strong driver"
        colour, label = status(p, threshold)
        # severity: predicted lows first, then the highest predicted 2-hour peak
        rank = 0 if label == "Low glucose risk" else 1
        trend = f.get("cgm_delta_30")
        arrow = "" if pd.isna(trend) else "↑↑" if trend > 30 else "↑" if trend > 10 else "↓↓" if trend < -30 else "↓" if trend < -10 else "→"
        age = (t - pd.Timestamp(p["timestamp"])).total_seconds() / 60
        out.append({"rank": rank, "peak_value": float(p["peak"]), "status": label, "colour": colour, "patient": p["patient_id"], "glucose": f"{p['cgm_now']:.0f} {arrow}",
                    "spike probability": float(p["spike_prob"]), "2-h peak": f"{p['peak']:.0f} ({p['peak_lo']:.0f}-{p['peak_hi']:.0f})",
                    "reason": reason, "data": data_quality(f)["level"], "last reading": f"{age:.0f} min ago"})
    return (pd.DataFrame(out).sort_values(["rank", "peak_value"], ascending=[True, False]).drop(columns=["rank", "peak_value"])
            .reset_index(drop=True))


def operating_point(dd: DemoData, threshold: float) -> dict | None:
    s = dd.sensitivity
    if s.empty:
        return None
    r = s.iloc[int((s["threshold"] - threshold).abs().idxmin())]
    return {"threshold": float(r["threshold"]), "spikes_caught": float(r["event_recall"]), "precision": float(r["episode_precision"]),
            "false_per_day": float(r["false_alerts_per_patient_day"]), "lead": float(r["median_lead_time_min"])}


# ----------------------------------------------------------------------------------------------
# twin and what-if
# ----------------------------------------------------------------------------------------------


def observed(dd: DemoData, pid: str):
    from twin.model import Observed

    return Observed.from_tables({**dd.tables}, pid)


def twin_for(dd: DemoData, pid: str):
    """Personalised twin (cached on disk; fitted on the first 7 days if not cached, ~20 s)."""
    from twin.estimate import fit

    path = DEMO / "twins" / f"{pid}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    tw = fit(observed(dd, pid), dd.cfgs["physiology"], dd.tcfg, dd.library)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tw, fh)
    return tw


def recent_meal(dd: DemoData, pid: str, t: pd.Timestamp, minutes: int = 45) -> pd.Series | None:
    """The last logged meal within `minutes` before t (already eaten; still being absorbed)."""
    ml = dd.tables["meal_log"]
    ml = ml[(ml["patient_id"] == pid) & (ml["timestamp"] <= t) & (ml["timestamp"] > t - pd.Timedelta(minutes=minutes))]
    return ml.iloc[-1] if len(ml) else None


SLOTS = {"breakfast": (7 * 60, 10 * 60 + 30), "lunch": (12 * 60, 15 * 60), "snack": (16 * 60, 18 * 60 + 30), "dinner": (19 * 60, 22 * 60 + 30)}
DEFAULT_PLATE = {"breakfast": ("idli_sambar", 8 * 60 + 30), "lunch": ("dal_chawal", 13 * 60 + 30), "snack": ("tea_biscuits", 17 * 60 + 30),
                 "dinner": ("roti_dal_sabzi", 21 * 60)}


def next_meal(dd: DemoData, pid: str, t: pd.Timestamp) -> tuple[int | None, str, float, str]:
    """(minutes until, plate id, portion, slot) of the patient's next usual meal from their own log.
    Minutes is None when no usual meal is due in the next 4 hours or the current slot's meal is
    already logged today."""
    from twin import constraints as C

    ml = dd.tables["meal_log"]
    ml = ml[(ml["patient_id"] == pid) & (ml["timestamp"] < t)]
    patient = patient_row(dd, pid)
    now = t.hour * 60 + t.minute
    by_name = {p["name"]: p for p in dd.library}
    today = ml[ml["timestamp"].dt.date == t.date()]
    chosen = None
    for slot, (a, b) in SLOTS.items():
        if now > b or ((today["meal_type"] == slot).any() and now >= a):
            continue
        chosen = slot
        break
    slot = chosen or "breakfast"
    d = ml[ml["meal_type"] == slot]
    plate_id, mod = DEFAULT_PLATE[slot]
    portion = 1.0
    if len(d):
        name = d["plate_name"].mode().iat[0]
        if name in by_name:
            plate_id = by_name[name]["id"]
            portion = float(np.clip(d["carbs_g_est"].median() / dd.tcfg["estimation"]["logging_bias"] / by_name[name]["carbs"], 0.5, 2.0))
        mod = int((d["timestamp"].dt.hour * 60 + d["timestamp"].dt.minute).median())
    if not C.check_plate(dd.plates[plate_id], patient).allowed:
        plate_id = next(p["id"] for p in dd.library if slot in p["meal"] and C.check_plate(p, patient).allowed)
    offset = max(mod - now, 10) if chosen else None
    if offset is not None and offset > 240:
        offset = None
    return offset, plate_id, round(portion, 2), slot


def pickle_path(pid: str) -> Path:
    return DEMO / "twins" / f"{pid}.pkl"
