"""Build the dashboard's demo cache in data/demo/.

    uv run python -m dashboard.prepare               # scenes' twins only (~2 min)
    uv run python -m dashboard.prepare --all-twins   # also fit a twin for every test patient (~20 min)

Contents: observed tables and 5-minute features of the held-out test patients, model outputs at
every prediction time, a reference pool of training rows for explanations, three scripted demo
scenes (a correct alert, a stable patient, an uncertain moment), and fitted twins.
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from common import config
from dashboard.service import DEMO, crossing_minutes
from data_gen.generate import load_configs
from models.inference import TwinPredictor

OBS_TABLES = ["patients", "hba1c_history", "medications", "cgm", "wearable_heart_rate", "wearable_steps", "wearable_sleep_stages",
              "wearable_hrv", "meal_log"]


def pick_scenes(P: pd.DataFrame, threshold: float, fit_days: int) -> list[dict]:
    start = P.groupby("patient_id")["timestamp"].transform("min").dt.normalize()
    day = (P["timestamp"] - start).dt.days
    h = P["timestamp"].dt.hour
    base = (day >= fit_days + 1) & h.between(7, 21)
    scenes = []

    alert = P[base & P["spike_eligible"] & P["spike_label"] & (P["spike_prob"].between(0.75, 0.95)) & (P["meal_min_since_last"] <= 30)
              & (P["cgm_missing_2h"] == 0) & P["wear_sleep_total_min"].notna() & (P["peak_120"] >= 210)].copy()
    alert["cross"] = [crossing_minutes(r) for _, r in alert.iterrows()]
    alert = alert[alert["cross"].between(30, 100)]
    if len(alert):
        r = alert.iloc[len(alert) // 2]
        scenes.append({"id": "alert", "title": "1. Correct alert after a meal", "patient_id": r["patient_id"], "timestamp": str(r["timestamp"]),
                       "story": (f"A meal was logged {r['meal_min_since_last']:.0f} min ago. The model raises a spike alert "
                                 f"({100 * r['spike_prob']:.0f} %) about {r['cross']:.0f} min before glucose is forecast to cross 180. "
                                 f"What actually happened: CGM peaked at {r['peak_120']:.0f} mg/dL within 2 h.")})

    stable = P[base & ~P["spike_label"] & (P["spike_prob"] < 0.05) & P["cgm_now"].between(100, 150) & (P["ehr_hba1c_pct"] <= 7.5)
               & h.between(15, 18) & (P["cgm_missing_2h"] == 0) & P["wear_sleep_total_min"].notna()]
    if len(stable):
        r = stable.iloc[len(stable) // 2]
        scenes.append({"id": "stable", "title": "2. Stable, well-controlled patient", "patient_id": r["patient_id"], "timestamp": str(r["timestamp"]),
                       "story": (f"HbA1c {r['ehr_hba1c_pct']:.1f} %, glucose {r['cgm_now']:.0f} mg/dL, no alert ({100 * r['spike_prob']:.0f} %). "
                                 f"What actually happened: the 2-hour peak was {r['peak_120']:.0f} mg/dL.")})

    unc = P[base & P["spike_eligible"] & P["spike_prob"].between(0.4, 0.6) & ((P["f60_hi"] - P["f60_lo"]) > 60)]
    gap = unc[(unc["cgm_missing_2h"] >= 0.1) | unc["wear_sleep_total_min"].isna()]
    pool = gap if len(gap) else unc
    if len(pool):
        r = pool.iloc[len(pool) // 2]
        scenes.append({"id": "uncertain", "title": "3. The model is unsure and says so", "patient_id": r["patient_id"], "timestamp": str(r["timestamp"]),
                       "story": (f"Spike probability {100 * r['spike_prob']:.0f} % with a wide range "
                                 f"({r['f60_lo']:.0f}-{r['f60_hi']:.0f} mg/dL in 1 h)"
                                 + (" and missing input data" if (r["cgm_missing_2h"] >= 0.1 or pd.isna(r["wear_sleep_total_min"])) else "")
                                 + f". What actually happened: {'a spike' if r['spike_label'] else 'no spike'} (2-hour peak {r['peak_120']:.0f} mg/dL).")})
    return scenes


def main() -> None:
    all_twins = "--all-twins" in sys.argv
    tic = time.time()
    cfgs = load_configs()
    root = config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"] / cfgs["pipeline"]["dataset_version"]
    feat_path = config.REPO_ROOT / "data" / "features" / cfgs["pipeline"]["dataset_version"] / "features.parquet"
    if not feat_path.exists():
        raise SystemExit("features missing: run `uv run python -m models.features` and `uv run python -m models.train` first")
    splits = pd.read_parquet(root / "observed" / "splits.parquet")
    test = set(splits.loc[splits["split"] == "test", "patient_id"])
    (DEMO / "observed").mkdir(parents=True, exist_ok=True)
    for name in OBS_TABLES:
        d = pd.read_parquet(root / "observed" / f"{name}.parquet")
        d[d["patient_id"].isin(test)].reset_index(drop=True).to_parquet(DEMO / "observed" / f"{name}.parquet", index=False)

    feats = pd.read_parquet(feat_path)
    te = feats[feats["patient_id"].isin(test)].reset_index(drop=True)
    te.to_parquet(DEMO / "features.parquet", index=False)
    pred = TwinPredictor.load()
    out = pred.predict(te)
    keep = ["patient_id", "timestamp", "cgm_now", "spike_label", "spike_eligible", "peak_120", "meal_min_since_last", "cgm_missing_2h",
            "wear_sleep_total_min", "ehr_hba1c_pct"]
    P = pd.concat([te[keep].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    slim = P[[c for c in P.columns if c not in ("meal_min_since_last", "cgm_missing_2h", "wear_sleep_total_min", "ehr_hba1c_pct")]].copy()
    num = slim.select_dtypes("float64").columns
    slim[num] = slim[num].astype("float32")
    slim.to_parquet(DEMO / "predictions.parquet", index=False)
    tr = feats[feats["split"] == "train"]
    tr.iloc[np.random.default_rng(0).choice(len(tr), size=min(20000, len(tr)), replace=False)].reset_index(drop=True).to_parquet(
        DEMO / "reference_pool.parquet", index=False)
    print(f"[prepare] observed, features, predictions, reference pool ({time.time() - tic:.0f}s)")

    tcfg = config.load("twin")
    scenes = pick_scenes(P, pred.threshold, int(tcfg["estimation"]["fit_days"]))
    (DEMO / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    for s in scenes:
        print(f"[prepare] scene {s['id']}: {s['patient_id']} at {s['timestamp']}")

    from twin.estimate import fit
    from twin.model import Observed

    obs_tabs = {n: pd.read_parquet(DEMO / "observed" / f"{n}.parquet") for n in ("patients", "medications", "cgm", "meal_log", "wearable_steps")}
    lib = config.load("meal_library")["plates"]
    ids = sorted(test) if all_twins else sorted({s["patient_id"] for s in scenes})
    (DEMO / "twins").mkdir(exist_ok=True)
    for i, pid in enumerate(ids):
        path = DEMO / "twins" / f"{pid}.pkl"
        if path.exists():
            continue
        tw = fit(Observed.from_tables(obs_tabs, pid), cfgs["physiology"], tcfg, lib)
        with open(path, "wb") as fh:
            pickle.dump(tw, fh)
        print(f"[prepare] twin {i + 1}/{len(ids)} {pid} ({time.time() - tic:.0f}s)", flush=True)
    print(f"[prepare] done in {time.time() - tic:.0f}s -> {DEMO}")


if __name__ == "__main__":
    main()
