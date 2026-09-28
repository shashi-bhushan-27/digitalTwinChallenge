"""Level-matched comparison of synthetic and real CGM (shared by calibration and the report).

Real CGM in both open datasets sits well below the level implied by the recorded HbA1c (the
recordings capture a treated, monitored period), while the synthetic cohort is anchored to
HbA1c. Comparing raw distributions would mostly measure that level gap, so each real patient is
matched to the synthetic patient with the closest mean CGM (without replacement) and the
*dynamics* are compared: variability, excursions, post-meal response, night-time level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import config, seeding
from validation import metrics as M
from validation.real_data import loaders as L


def real_sets(vcfg: dict) -> dict[str, dict]:
    """Real T2D data split into 'calibration' (Shanghai half), 'validation' (other Shanghai half)
    and 'external' (CGMacros T2D)."""
    data = L.real_t2d()
    cgm = data["cgm"]
    span = cgm.groupby("patient_id")["timestamp"].agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)
    ok = set(span[span >= vcfg["real_split"]["min_days"]].index)
    pts = data["patients"][data["patients"]["patient_id"].isin(ok)]
    sh = sorted(pts.loc[pts["dataset"] == "shanghai_t2dm", "patient_id"])
    r = seeding.rng(vcfg["seed"], "real_split")
    perm = [sh[i] for i in r.permutation(len(sh))]
    k = int(round(vcfg["real_split"]["calibration_fraction"] * len(sh)))
    groups = {"calibration": set(perm[:k]), "validation": set(perm[k:]),
              "external": set(pts.loc[pts["dataset"] == "cgmacros", "patient_id"])}
    out = {}
    for name, ids in groups.items():
        out[name] = {
            "patients": pts[pts["patient_id"].isin(ids)].reset_index(drop=True),
            "cgm": cgm[cgm["patient_id"].isin(ids)].reset_index(drop=True),
            "meals": data["meals"][data["meals"]["patient_id"].isin(ids)].reset_index(drop=True),
        }
    return out


def summarise(cgm: pd.DataFrame, meals: pd.DataFrame, resample_min: int | None = None) -> pd.DataFrame:
    """Per-patient metrics including post-meal medians and spike rate."""
    if resample_min:
        cgm = M.resample(cgm, resample_min)
    s = M.patient_summaries(cgm)
    pp = M.postprandial(cgm, meals)
    if len(pp):
        agg = pp.groupby("patient_id").agg(pp_rise=("rise", "median"), pp_time_to_peak=("time_to_peak_min", "median"), pp_n=("rise", "size"))
        s = s.join(agg)
    else:
        s["pp_rise"] = s["pp_time_to_peak"] = np.nan
        s["pp_n"] = 0
    s = s.join(M.spike_onsets_per_day(cgm))
    return s


def synthetic_observed(tables: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cgm, meal_log) from generate.generate() output, in the common column format."""
    cgm = tables["observed/cgm"][["patient_id", "timestamp", "glucose_mgdl"]]
    meals = tables["observed/meal_log"][["patient_id", "timestamp"]]
    return cgm, meals


def match_by_mean(real: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    """Synthetic rows matched 1:1 to real rows on mean CGM (greedy, closest first, no reuse)."""
    avail = syn.dropna(subset=["mean"]).copy()
    chosen = []
    for pid, m in real["mean"].sort_values().items():
        if avail.empty:
            break
        j = (avail["mean"] - m).abs().idxmin()
        chosen.append((pid, j))
        avail = avail.drop(index=j)
    out = syn.loc[[j for _, j in chosen]].copy()
    out["matched_real_id"] = [pid for pid, _ in chosen]
    return out


def compare_table(real: pd.DataFrame, syn: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        a = real[m].dropna().to_numpy(float)
        b = syn[m].dropna().to_numpy(float)
        if len(a) < 3 or len(b) < 3:
            continue
        q1, q3 = np.percentile(a, [25, 75])
        iqr = max(q3 - q1, 1e-9)
        ks = stats.ks_2samp(a, b)
        rows.append({
            "metric": m, "real_median": np.median(a), "real_q1": q1, "real_q3": q3,
            "synthetic_median": np.median(b), "synthetic_q1": np.percentile(b, 25), "synthetic_q3": np.percentile(b, 75),
            "median_gap_in_real_iqr": (np.median(b) - np.median(a)) / iqr, "ks_stat": ks.statistic, "ks_p": ks.pvalue,
            "n_real": len(a), "n_synthetic": len(b),
        })
    return pd.DataFrame(rows)


def loss(table: pd.DataFrame, weights: dict) -> float:
    t = table.set_index("metric")
    return float(sum(w * t.loc[m, "median_gap_in_real_iqr"] ** 2 for m, w in weights.items() if m in t.index))


def pool_configs(base_cfgs: dict, vcfg: dict) -> dict:
    """Configs for the level-matched comparison pool (broader, lower HbA1c range)."""
    cfgs = dict(base_cfgs)
    cfgs["cohort"] = config.deep_merge(base_cfgs["cohort"], vcfg["pool"]["cohort_overlay"])
    return cfgs
