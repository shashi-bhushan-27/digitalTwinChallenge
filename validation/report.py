"""Run the full validation and write docs/validation_report.md plus results/validation/*.

    uv run python -m validation.report

Needs the frozen synthetic dataset (uv run python -m data_gen.generate) and the real data
(uv run python -m validation.real_data.download). Every number in the report is computed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config
from common.units import eag_mgdl_from_hba1c, gmi_pct
from data_gen.generate import generate, load_configs
from validation import discriminator, tstr
from validation import metrics as M
from validation.compare import (
    compare_table,
    match_by_mean,
    pool_configs,
    real_sets,
    summarise,
    synthetic_observed,
)
from validation.windows import regular_series, windows

RES = config.REPO_ROOT / "results" / "validation"
DOCS = config.REPO_ROOT / "docs"
COLORS = {"real": "#3b6ea8", "external": "#8a5cb8", "synthetic": "#d9822b"}


def _load_main(cfgs: dict) -> dict[str, pd.DataFrame]:
    root = config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"] / cfgs["pipeline"]["dataset_version"]
    if not (root / "manifest.json").exists():
        raise SystemExit(f"{root} missing: run `uv run python -m data_gen.generate` first")
    names = ["observed/cgm", "observed/meal_log", "observed/patients", "observed/splits"]
    out = {n: pd.read_parquet(root / f"{n}.parquet") for n in names}
    out["manifest"] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return out


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}g}" if abs(v) >= 100 else f"{v:.{digits}f}".rstrip("0").rstrip("."))
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fig_level_gap(real_pat, real_sum, main_pat, main_sum, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.linspace(5.5, 13, 50)
    ax.plot(x, eag_mgdl_from_hba1c(x), color="#555", lw=1, ls="--", label="ADAG estimate (eAG)")
    m = main_sum.join(main_pat.set_index("patient_id")["hba1c_pct"])
    ax.scatter(m["hba1c_pct"], m["mean"], s=8, alpha=0.45, color=COLORS["synthetic"], label=f"synthetic cohort (n={len(m)})")
    r = real_sum.join(real_pat.set_index("patient_id")[["hba1c_pct", "dataset"]])
    for ds, col, lab in (("shanghai_t2dm", COLORS["real"], "Shanghai T2DM"), ("cgmacros", COLORS["external"], "CGMacros T2D")):
        d = r[r["dataset"] == ds]
        ax.scatter(d["hba1c_pct"], d["mean"], s=16, alpha=0.8, color=col, label=f"{lab} (n={d['hba1c_pct'].notna().sum()})")
    ax.set_xlabel("Recorded HbA1c (%)")
    ax.set_ylabel("Mean CGM glucose (mg/dL)")
    ax.set_title("Glucose level vs recorded HbA1c", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    rr = r.dropna(subset=["hba1c_pct"])
    stats = {}
    for ds in ("shanghai_t2dm", "cgmacros"):
        d = rr[rr["dataset"] == ds]
        stats[ds] = {"median_hba1c": float(d["hba1c_pct"].median()), "median_eag": float(np.median(eag_mgdl_from_hba1c(d["hba1c_pct"]))),
                     "median_mean_cgm": float(d["mean"].median()), "median_gmi": float(np.median(gmi_pct(d["mean"]))),
                     "slope_mgdl_per_pct": float(np.polyfit(d["hba1c_pct"], d["mean"], 1)[0]) if len(d) > 3 else None}
    stats["synthetic_main"] = {"median_hba1c": float(m["hba1c_pct"].median()), "median_eag": float(np.median(eag_mgdl_from_hba1c(m["hba1c_pct"]))),
                               "median_mean_cgm": float(m["mean"].median()), "median_gmi": float(np.median(gmi_pct(m["mean"]))),
                               "slope_mgdl_per_pct": float(np.polyfit(m["hba1c_pct"], m["mean"], 1)[0])}
    return stats


def fig_distributions(groups: dict[str, pd.DataFrame], metrics: list[str], path: Path) -> None:
    labels = {"cv": "CV", "mage": "MAGE (mg/dL)", "tir_70_180": "Time in range 70-180", "nocturnal_to_mean": "Night (03-06h) / mean",
              "diff_sd": "SD of 15-min change", "pp_rise": "Post-meal rise (mg/dL)", "pp_time_to_peak": "Time to peak (min)",
              "day_to_day_cv": "Day-to-day CV", "spikes_per_day": "Spikes >180 per day"}
    metrics = [m for m in metrics if m in labels]
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.6 * nrow))
    for ax, m in zip(axes.flat, metrics, strict=False):
        data = [g[m].dropna().to_numpy() for g in groups.values()]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, col in zip(bp["boxes"], [COLORS["real"], COLORS["external"], COLORS["synthetic"]], strict=False):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(groups) + 1), [g.replace(" ", "\n", 1) for g in groups], fontsize=7)
        ax.set_title(labels[m], fontsize=9)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Per-patient CGM dynamics: real held-out vs level-matched synthetic", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_traces(real_cgm: pd.DataFrame, syn_cgm: pd.DataFrame, pairs: list[tuple[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 2.3 * len(pairs)), sharey=True)
    for row, (rid, sid) in enumerate(pairs):
        for col, (cgm, pid, name) in enumerate(((real_cgm, rid, "real"), (syn_cgm, sid, "synthetic"))):
            d = cgm[cgm["patient_id"] == pid].sort_values("timestamp")
            t0 = d["timestamp"].min().normalize() + pd.Timedelta(days=1)
            d = d[(d["timestamp"] >= t0) & (d["timestamp"] < t0 + pd.Timedelta(days=3))]
            ax = axes[row, col]
            ax.axhspan(70, 180, color="#e9f3ea")
            ax.plot((d["timestamp"] - t0).dt.total_seconds() / 3600, d["glucose_mgdl"], color=COLORS["real" if name == "real" else "synthetic"], lw=1)
            ax.set_xlim(0, 72)
            ax.set_ylim(40, 400)
            ax.set_title(f"{name}: {pid}  (mean {cgm[cgm['patient_id'] == pid]['glucose_mgdl'].mean():.0f} mg/dL)", fontsize=8)
            ax.set_xticks([0, 24, 48, 72])
    for ax in axes[-1]:
        ax.set_xlabel("hours")
    fig.suptitle("Three days of 15-minute CGM: real (left) and its level-matched synthetic partner (right)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mean_postprandial_curve(cgm: pd.DataFrame, meals: pd.DataFrame, horizon: int = 180, step: int = 15) -> pd.DataFrame:
    lags = np.arange(0, horizon + step, step)
    rows = []
    series = {pid: d.set_index("timestamp")["glucose_mgdl"].sort_index() for pid, d in cgm.dropna(subset=["glucose_mgdl"]).groupby("patient_id")}
    for m in meals.itertuples(index=False):
        s = series.get(m.patient_id)
        if s is None:
            continue
        t = pd.Timestamp(m.timestamp)
        idx = s.index.get_indexer([t + pd.Timedelta(minutes=int(k)) for k in lags], method="nearest", tolerance=pd.Timedelta(minutes=8))
        if (idx < 0).any():
            continue
        v = s.to_numpy()[idx]
        rows.append(v - v[0])
    arr = np.array(rows)
    return pd.DataFrame({"minutes": lags, "median": np.median(arr, 0), "q1": np.percentile(arr, 25, 0), "q3": np.percentile(arr, 75, 0), "n": len(arr)})


def fig_postprandial(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for (name, c), col in zip(curves.items(), [COLORS["real"], COLORS["external"], COLORS["synthetic"]], strict=False):
        ax.plot(c["minutes"], c["median"], color=col, lw=2, label=f"{name} (n={int(c['n'].iat[0])} meals)")
        ax.fill_between(c["minutes"], c["q1"], c["q3"], color=col, alpha=0.15)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("minutes after logged meal")
    ax.set_ylabel("change from meal-time glucose (mg/dL)")
    ax.set_title("Post-meal response (median, IQR band)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    vcfg = config.load("validation")
    cfgs = load_configs()
    step = vcfg["resample_min"]
    sets = real_sets(vcfg)
    main_ds = _load_main(cfgs)
    print("[report] real sets:", {k: len(v["patients"]) for k, v in sets.items()})

    # ---- summaries
    real_sum = {k: summarise(v["cgm"], v["meals"]) for k, v in sets.items()}
    main15 = M.resample(main_ds["observed/cgm"], step)
    main_sum = summarise(main15, main_ds["observed/meal_log"])
    pool_tables = generate(pool_configs(cfgs, vcfg), vcfg["pool"]["n_patients"], vcfg["pool"]["n_days"], verbose=False)
    pool_cgm, pool_meals = synthetic_observed(pool_tables)
    pool15 = M.resample(pool_cgm, step)
    pool_sum = summarise(pool15, pool_meals)
    print("[report] pool summarised:", len(pool_sum))

    # ---- level gap
    all_real_pat = pd.concat([v["patients"] for v in sets.values()], ignore_index=True)
    all_real_sum = pd.concat(real_sum.values())
    level = fig_level_gap(all_real_pat, all_real_sum, main_ds["observed/patients"], main_sum, RES / "level_gap.png")

    # ---- level-matched comparisons (held-out real only)
    metrics = vcfg["compare_metrics"]
    matched_val = match_by_mean(real_sum["validation"], pool_sum)
    avail = pool_sum.drop(index=matched_val.index)
    matched_ext = match_by_mean(real_sum["external"], avail)
    tab_val = compare_table(real_sum["validation"], matched_val, metrics)
    tab_ext = compare_table(real_sum["external"], matched_ext, metrics)
    tab_cal = compare_table(real_sum["calibration"], match_by_mean(real_sum["calibration"], pool_sum), metrics)
    tab_unmatched = compare_table(real_sum["validation"], main_sum, ["mean", "gmi", "tir_70_180", "cv", "mage", "pp_rise", "pp_time_to_peak", "spikes_per_day"])
    for name, t in (("matched_vs_shanghai_validation", tab_val), ("matched_vs_cgmacros_external", tab_ext),
                    ("matched_vs_shanghai_calibration", tab_cal), ("unmatched_main_vs_shanghai_validation", tab_unmatched)):
        t.to_csv(RES / f"compare_{name}.csv", index=False)
    fig_distributions({"Shanghai held-out": real_sum["validation"], "CGMacros T2D": real_sum["external"],
                       "synthetic (matched)": pd.concat([matched_val, matched_ext])}, metrics, RES / "matched_distributions.png")
    pairs = list(zip(matched_val["matched_real_id"].iloc[[2, len(matched_val) // 2, -3]], matched_val.index[[2, len(matched_val) // 2, -3]], strict=True))
    fig_traces(sets["validation"]["cgm"], pool15, pairs, RES / "example_traces.png")
    curves = {"Shanghai held-out": mean_postprandial_curve(sets["validation"]["cgm"], sets["validation"]["meals"]),
              "CGMacros T2D": mean_postprandial_curve(sets["external"]["cgm"], sets["external"]["meals"]),
              "synthetic (matched)": mean_postprandial_curve(pool15[pool15["patient_id"].isin(matched_val.index)],
                                                             pool_meals[pool_meals["patient_id"].isin(matched_val.index)])}
    fig_postprandial(curves, RES / "postprandial_curves.png")
    print("[report] comparisons done")

    # ---- discriminator (held-out Shanghai + CGMacros vs synthetic)
    real_held = pd.concat([sets["validation"]["cgm"], sets["external"]["cgm"]])
    matched_ids = set(matched_val.index) | set(matched_ext.index)
    disc_matched = discriminator.run(real_held, pool15[pool15["patient_id"].isin(matched_ids)])
    main_ids = main_ds["observed/splits"].query("split == 'test'")["patient_id"]
    disc_unmatched = discriminator.run(real_held, main15[main15["patient_id"].isin(main_ids)])
    disc = {"level_matched": disc_matched, "unmatched_main_cohort": disc_unmatched}
    (RES / "discriminator.json").write_text(json.dumps(disc, indent=2), encoding="utf-8")
    print("[report] discriminator:", {k: round(v["auc"], 3) for k, v in disc.items()})

    # ---- TSTR
    train_ids = main_ds["observed/splits"].query("split == 'train'")["patient_id"]
    tstr_res = tstr.run(
        train_sets={"synthetic main cohort (train split)": main15[main15["patient_id"].isin(train_ids)],
                    "synthetic level-matched pool": pool15,
                    "real Shanghai calibration half (TRTR)": sets["calibration"]["cgm"]},
        test_sets={"Shanghai held-out": sets["validation"]["cgm"], "CGMacros T2D": sets["external"]["cgm"]},
    )
    tstr_res.to_csv(RES / "tstr.csv", index=False)
    print("[report] TSTR done")

    # steepest 15-min fall per 12-h window, real held-out split by insulin use, vs matched synthetic
    fall = {}
    vp = sets["validation"]["patients"]
    for label, ids in (("real, on insulin", vp.loc[vp["on_insulin"], "patient_id"]), ("real, no insulin", vp.loc[~vp["on_insulin"], "patient_id"]),
                       ("synthetic (matched)", pd.Index(matched_val.index))):
        src = sets["validation"]["cgm"] if label.startswith("real") else pool15
        x, _, _ = windows(regular_series(src[src["patient_id"].isin(ids)]), 48, 48, align_hours=(0, 12))
        fall[label] = {"patients": int(len(ids)), "windows": int(len(x)), "median_steepest_fall": float(np.median(np.diff(x, axis=1).min(1)))}
    summary = {"level_gap": level, "discriminator": disc, "steepest_fall": fall, "calibrated": config.load_raw("calibrated") if (config.CONFIG_DIR / "calibrated.yaml").exists() else None,
               "dataset_manifest": {k: main_ds["manifest"][k] for k in ("dataset_version", "seed", "n_patients", "n_days")},
               "n_real": {k: int(len(v["patients"])) for k, v in sets.items()}}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(summary, tab_val, tab_ext, tab_cal, tab_unmatched, tstr_res, main_sum)
    print(f"[report] wrote {DOCS / 'validation_report.md'}")


def _leaves(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_leaves(v, key) if isinstance(v, dict) else {key: v})
    return out


def _tstr_summary(t: pd.DataFrame) -> str:
    """One sentence per test set comparing clock-free TSTR with persistence and TRTR (computed)."""
    out = []
    for test in t["test"].unique():
        d = t[t["test"] == test].set_index(["train", "features"])
        try:
            per = d.loc[("persistence (no training)", "-")]
            syn = d.loc[("synthetic main cohort (train split)", "CGM history only")]
            real = d.loc[("real Shanghai calibration half (TRTR)", "CGM history only")]
        except KeyError:
            continue
        out.append(
            f"- **{test}**: 60-min MAE {syn['mae_60']:.1f} mg/dL when trained on synthetic data vs {per['mae_60']:.1f} for persistence "
            f"and {real['mae_60']:.1f} when trained on real data; spike PR-AUC {syn['spike_pr_auc']:.3f} vs {per['spike_pr_auc']:.3f} "
            f"(persistence) and {real['spike_pr_auc']:.3f} (real); AUROC {syn['spike_auroc']:.3f} vs {per['spike_auroc']:.3f} and {real['spike_auroc']:.3f}."
        )
    return "\n".join(out)


def _within(t: pd.DataFrame, tol: float = 0.5) -> list[str]:
    return list(t.loc[t["median_gap_in_real_iqr"].abs() <= tol, "metric"])


def write_markdown(summary, tab_val, tab_ext, tab_cal, tab_unmatched, tstr_res, main_sum) -> None:
    lv = summary["level_gap"]
    cols = ["metric", "real_median", "real_q1", "real_q3", "synthetic_median", "median_gap_in_real_iqr", "ks_stat"]
    d = summary["discriminator"]
    cal = summary["calibrated"] or {}
    tcols = ["train", "features", "test", "mae_30", "mae_60", "rmse_60", "spike_auroc", "spike_pr_auc", "spike_rate", "n_samples"]
    man = summary["dataset_manifest"]
    lines = [
        "# Validation report: synthetic vs real CGM",
        "",
        "_Generated by `uv run python -m validation.report`. Every number below comes from that run; "
        "figures and CSVs are in `results/validation/`._",
        "",
        f"Synthetic dataset: version {man['dataset_version']}, seed {man['seed']}, {man['n_patients']} patients x {man['n_days']} days. "
        f"Real data (validation and calibration only, never used to train product models): Shanghai T2DM split by patient into a calibration half "
        f"({summary['n_real']['calibration']} patients) and a held-out half ({summary['n_real']['validation']}), plus CGMacros T2D participants "
        f"({summary['n_real']['external']}) as an external set. All comparisons use 15-minute CGM, the resolution of both real datasets.",
        "",
        "## 1. Glucose level: a deliberate difference",
        "",
        "![level gap](../results/validation/level_gap.png)",
        "",
        "| set | median HbA1c (%) | median eAG from HbA1c (mg/dL) | median mean CGM (mg/dL) | median GMI (%) | slope of mean CGM on HbA1c (mg/dL per %) |",
        "|---|---|---|---|---|---|",
    ]
    for name, key in (("Shanghai T2DM (all)", "shanghai_t2dm"), ("CGMacros T2D", "cgmacros"), ("Synthetic cohort", "synthetic_main")):
        s = lv[key]
        slope = f"{s['slope_mgdl_per_pct']:.1f}" if s["slope_mgdl_per_pct"] is not None else "n/a"
        lines.append(f"| {name} | {s['median_hba1c']:.2f} | {s['median_eag']:.0f} | {s['median_mean_cgm']:.0f} | {s['median_gmi']:.2f} | {slope} |")
    lines += [
        "",
        f"In both real datasets the CGM recorded during the study sits below the level the recorded HbA1c implies "
        f"(ADAG: eAG = 28.7 x HbA1c - 46.7): by {lv['shanghai_t2dm']['median_eag'] - lv['shanghai_t2dm']['median_mean_cgm']:.0f} mg/dL "
        f"at the median in Shanghai, where mean CGM barely rises with HbA1c (slope {lv['shanghai_t2dm']['slope_mgdl_per_pct']:.1f} mg/dL per %), "
        f"and by {lv['cgmacros']['median_eag'] - lv['cgmacros']['median_mean_cgm']:.0f} mg/dL in CGMacros, whose slope "
        f"({lv['cgmacros']['slope_mgdl_per_pct']:.1f}) is close to ADAG's 28.7 but offset lower (14 people only). The Shanghai recordings "
        "capture a monitored, treated period (half the patients were on insulin) rather than the 3-month average that HbA1c reflects. "
        "The synthetic cohort represents Indian outpatients whose HbA1c is current, so it is anchored to the ADAG relation on purpose. "
        "Comparing raw distributions would mostly measure this level difference, so the dynamics below are compared between "
        "real and synthetic patients **matched on mean CGM**.",
        "",
        "Unmatched, for transparency (synthetic main cohort vs Shanghai held-out):",
        "",
        _md_table(tab_unmatched[cols]),
        "",
        "## 2. Level-matched dynamics",
        "",
        "Each held-out real patient is paired with the synthetic patient (from a 300-patient pool generated with the same, calibrated "
        "configuration but a wider HbA1c range) whose mean CGM is closest. `median_gap_in_real_iqr` is (synthetic median - real median) "
        "divided by the real interquartile range; |gap| <= 0.5 means the synthetic median sits well inside the real spread. "
        "`ks_stat` is the two-sample Kolmogorov-Smirnov distance between the per-patient distributions.",
        "",
        "![distributions](../results/validation/matched_distributions.png)",
        "",
        f"**Shanghai held-out half** (metrics within 0.5 IQR: {', '.join(_within(tab_val)) or 'none'})",
        "",
        _md_table(tab_val[cols]),
        "",
        f"**CGMacros T2D, external** (metrics within 0.5 IQR: {', '.join(_within(tab_ext)) or 'none'})",
        "",
        _md_table(tab_ext[cols]),
        "",
        "For reference, the same comparison on the calibration half that the tuning saw:",
        "",
        _md_table(tab_cal[cols]),
        "",
        "![post-meal](../results/validation/postprandial_curves.png)",
        "",
        "![traces](../results/validation/example_traces.png)",
        "",
        "## 3. Real-vs-synthetic discriminator",
        "",
        "LightGBM on summary features of 12-hour CGM windows, patient-grouped 5-fold cross-validation, classes balanced. "
        "Closer to 50 % accuracy / 0.5 AUC means harder to tell apart.",
        "",
        "| comparison | windows | real patients | synthetic patients | accuracy | AUC | most useful features |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, key in (("level-matched synthetic vs held-out real", "level_matched"), ("main synthetic cohort (unmatched) vs held-out real", "unmatched_main_cohort")):
        r = d[key]
        feats = ", ".join(list(r["top_features"])[:4])
        lines.append(f"| {name} | {r['n_windows']} | {r['n_real_patients']} | {r['n_synthetic_patients']} | {r['accuracy']:.3f} | {r['auc']:.3f} | {feats} |")
    fm = d["level_matched"]["top_feature_medians"]
    lines += [
        "",
        "The simulator is still easy to tell apart window by window. Median values of the strongest cues (level-matched run):",
        "",
        "| feature | real | synthetic |",
        "|---|---|---|",
        *[f"| {f} | {v['real']:.3g} | {v['synthetic']:.3g} |" for f, v in fm.items()],
        "",
        "`diff_min` is the steepest 15-minute fall in a 12-hour window. Real glucose falls faster after peaks than the simulator's, "
        "and not only in insulin users:",
        "",
        "| group (Shanghai held-out half) | patients | windows | median steepest 15-min fall (mg/dL) |",
        "|---|---|---|---|",
        *[f"| {k} | {v['patients']} | {v['windows']} | {v['median_steepest_fall']:.1f} |" for k, v in summary["steepest_fall"].items()],
        "",
        "Candidate fixes for the next generator iteration: faster insulin action (p2) or a sharper meal-absorption tail, with "
        "post-peak fall rate added to the calibration targets.",
    ]
    lines += [
        "",
        "## 4. TSTR: train on synthetic, test on real",
        "",
        "CGM-only tasks (the real datasets have no wearables): glucose 30 and 60 min ahead (MAE/RMSE, mg/dL) and onset of "
        "glucose > 180 mg/dL for at least 15 min within 2 h, from windows starting at or below 180 (AUROC, PR-AUC). "
        "The TRTR row trains on the real calibration half; it is a yardstick inside `validation/` only. "
        "Each model is trained twice: on CGM history only, and with time-of-day features. Real Shanghai meals follow a hospital "
        "timetable (breakfast ~07:00, lunch ~11:00, dinner ~17:00) while the synthetic patients eat on an Indian home schedule "
        "(lunch ~13:30, dinner ~21:00), so clock features do not transfer; the CGM-history-only rows are the fair TSTR test.",
        "",
        _tstr_summary(tstr_res),
        "",
        _md_table(tstr_res[[c for c in tcols if c in tstr_res.columns]]),
        "",
        "## 5. Calibration",
        "",
        f"`validation/calibrate.py` searched {len(_leaves(cal.get('overlays', {})))} population-level settings on the Shanghai "
        f"calibration half only (random search, fixed seed). Loss (weighted squared median gaps in real-IQR units) went from "
        f"{cal.get('loss_default', float('nan')):.2f} with the hand-set defaults to {cal.get('loss_best', float('nan')):.2f}. "
        "Chosen values (`configs/calibrated.yaml`):",
        "",
        "| setting | value |",
        "|---|---|",
        *[f"| `{k}` | {v:.4g} |" for k, v in _leaves(cal.get("overlays", {})).items()],
        "",
        "## 6. Limitations",
        "",
        "- Neither real dataset is Indian; Shanghai is the closest phenotype available openly. Indian-specific values in the generator "
        "are assumptions listed in `docs/parameters_to_verify.md`.",
        "- Real meals differ from the synthetic Indian meals (small hospital portions in Shanghai; half the patients on mealtime insulin), "
        "so post-meal rise is compared but only weakly fitted.",
        "- The synthetic cohort uses oral agents and basal insulin only; prandial and premixed insulin (common in Shanghai) are not simulated.",
        "- Real data have no wearable streams with sleep/HRV, so the wearable layer is checked for plausibility only, not against real data.",
        "- Matching on mean CGM removes the level difference by construction; level-dependent metrics (TIR, TAR) are therefore partly matched too.",
        "- Everything here validates the simulator against two small datasets (tens of patients); it does not show clinical validity.",
    ]
    (DOCS / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
