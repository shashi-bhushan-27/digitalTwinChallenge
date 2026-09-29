"""Build docs/architecture.pdf and docs/architecture.png (vector SVG rendered by headless Chrome).

    uv run python scripts/build_architecture.py

Numbers in the footer are read from results/ so the diagram never goes stale.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
W, H = 1600, 900

INK, MUTED, ACCENT = "#1f2937", "#6b7280", "#2f6fb0"
FILL = {"data": "#eef4fb", "store": "#f4f6f8", "model": "#eef7f0", "twin": "#fdf3e7", "ui": "#f3effa", "real": "#fdecea"}
EDGE = {"data": "#2f6fb0", "store": "#6b7280", "model": "#1e8449", "twin": "#c77c0e", "ui": "#6d5ba6", "real": "#c0392b"}


def box(x, y, w, h, title, lines, kind, title_size=16):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{FILL[kind]}" stroke="{EDGE[kind]}" stroke-width="1.6"/>',
           f'<text x="{x + 12}" y="{y + 24}" font-size="{title_size}" font-weight="700" fill="{INK}">{title}</text>']
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + 12}" y="{y + 46 + 18 * i}" font-size="13" fill="{INK}">{ln}</text>')
    return "\n".join(out)


def lane(x, y, w, h, title, kind):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{EDGE[kind]}" stroke-width="1.2" stroke-dasharray="2 4"/>'
            f'<text x="{x + 14}" y="{y + 26}" font-size="18" font-weight="700" fill="{EDGE[kind]}">{title}</text>')


def arrow(x1, y1, x2, y2, dashed=False, color=INK, label=None, lx=None, ly=None):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8"{dash} marker-end="url(#arr)"/>'
    if label:
        s += f'<text x="{lx if lx is not None else (x1 + x2) / 2}" y="{ly if ly is not None else (y1 + y2) / 2 - 6}" font-size="12" fill="{MUTED}" text-anchor="middle">{label}</text>'
    return s


def results_footer() -> str:
    def j(p):
        f = ROOT / "results" / p
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    parts = []
    sens = ROOT / "results" / "models" / "fusion_alert_sensitivity.csv"
    if sens.exists():
        import pandas as pd

        s = pd.read_csv(sens)
        r = s.iloc[(s["threshold"] - 0.5).abs().idxmin()]
        parts.append(f"{100 * r['event_recall']:.0f} % of spikes caught, median {r['median_lead_time_min']:.0f} min ahead, "
                     f"{r['false_alerts_per_patient_day']:.2f} false alerts/patient-day (threshold 0.5)")
    tw = j("twin/summary.json").get("counterfactuals", {}).get("swap to a low-GI plate")
    if tw:
        parts.append(f"what-if dish swap error {tw['mae_fitted']:.0f} mg/dL vs {tw['mae_cohort_average']:.0f} for the cohort average")
    tstr = ROOT / "results" / "validation" / "tstr.csv"
    if tstr.exists():
        import pandas as pd

        t = pd.read_csv(tstr)
        t = t[(t["test"] == "Shanghai held-out")]
        syn = t[(t["train"] == "synthetic main cohort (train split)") & (t["features"] == "CGM history only")]["mae_60"].iat[0]
        per = t[t["train"] == "persistence (no training)"]["mae_60"].iat[0]
        parts.append(f"train-on-synthetic, test-on-real 60-min error {syn:.1f} vs {per:.1f} mg/dL (no-change baseline)")
    return " · ".join(parts)


def svg() -> str:
    e = []
    e.append(f'<text x="40" y="48" font-size="28" font-weight="800" fill="{INK}">T2D Digital Twin: architecture</text>')
    e.append(f'<text x="40" y="74" font-size="15" fill="{MUTED}">Team AMRIT_VIT · Digital Twin Challenge 2026 · synthetic patients only · decision support only</text>')

    # lane 1: synthetic data generator
    e.append(lane(30, 95, 390, 700, "1  Synthetic data (data_gen/)", "data"))
    y = 135
    specs = [
        ("Cohort: the medical record", ["400 Indian T2D patients, Gaussian copula", "labs, eGFR, liver, meds, allergies, diet", "no genetic data"]),
        ("Hidden physiology", ["insulin sensitivity, beta cells, liver", "from the record + unexplained variation", "anchored to each patient's HbA1c"]),
        ("Behaviour (14 days)", ["47 Indian dishes, late dinners, eating out", "walks, sleep stages, stress, illness", "missed doses, partial meal log"]),
        ("Glucose-insulin ODE", ["Bergman / Dalla Man hybrid, 1-min steps", "meals, 5 drug classes, exercise"]),
        ("Sensor realism", ["CGM lag, drift, noise, gaps (MARD 7 %)", "wearable gaps, phantom steps, stage errors"]),
    ]
    heights = [90, 90, 90, 72, 72]
    ys = []
    for (t, lines), h in zip(specs, heights, strict=True):
        e.append(box(45, y, 360, h, t, lines, "data"))
        ys.append((y, h))
        y += h + 22
    for (y0, h0), (y1, _h1) in zip(ys[:-1], ys[1:], strict=True):
        e.append(arrow(225, y0 + h0, 225, y1 - 2))

    # real data
    e.append(box(45, 700, 360, 82, "Real open CGM (validation only)", ["Shanghai T2DM (CC BY 4.0), CGMacros (CC BY-NC-SA)", "tune 7 settings on half, test on the rest"], "real", 15))
    e.append(arrow(405, 741, 470, 741, dashed=True, color=EDGE["real"]))
    e.append(box(470, 700, 250, 82, "validation/", ["level-matched dynamics, TSTR,", "real-vs-synthetic discriminator"], "real", 15))

    # lane 2: stores
    e.append(lane(450, 95, 290, 590, "Data stores", "store"))
    e.append(box(465, 150, 260, 200, "observed/  (clinic view)", ["EHR record + HbA1c history", "prescriptions + reported adherence", "CGM every 5 min", "heart rate, steps", "sleep stages, overnight HRV", "patient's meal log", "split by patient 70/15/15"], "store"))
    e.append(box(465, 400, 260, 140, "truth/  (hidden)", ["true meals, doses taken", "true physiology parameters", "plasma glucose", "used only to score models,", "explanations and the twin"], "store"))
    e.append(arrow(405, 600, 465, 300))                               # sensors -> observed
    e.append(arrow(405, 507, 465, 470, dashed=True, color=MUTED))       # simulator state -> truth

    # lane 3: models + twin
    e.append(lane(770, 95, 400, 330, "2  Prediction (models/)", "model"))
    e.append(box(785, 135, 370, 72, "Feature builder (past data only)", ["CGM history, wearables, meal log, record", "a test proves no future data leaks in"], "model"))
    e.append(box(785, 225, 370, 90, "LightGBM fusion", ["glucose 15-120 min + 80 % band (conformal)", "2-hour peak; calibrated spike probability", "ablation: CGM > + wearables > + EHR > + meal log"], "model"))
    e.append(box(785, 333, 370, 76, "Explanations (grouped SHAP)", ["clinical sentences in mg/dL vs similar patients", "checked against the hidden truth"], "model"))
    e.append(arrow(970, 207, 970, 223))
    e.append(arrow(970, 315, 970, 331))

    e.append(lane(770, 445, 400, 350, "3  Personalised twin (twin/)", "twin"))
    e.append(box(785, 485, 370, 72, "Personalise the ODE", ["record-based prior + fit to 7 days of CGM", "twin card: sensitivity, secretion, meal response"], "twin"))
    e.append(box(785, 575, 370, 58, "Constraint checks (before)", ["diet, allergies, intolerances, eGFR drug limits"], "twin"))
    e.append(box(785, 645, 370, 72, "What-if simulator", ["swap / resize / skip a meal, walk, dose change", "next 5 h, or a typical day at steady state"], "twin"))
    e.append(box(785, 729, 370, 52, "Safety check (after)", ["predicted low < 70 mg/dL + risk factors"], "twin"))
    for a, b in ((557, 573), (633, 643), (717, 727)):
        e.append(arrow(970, a, 970, b))

    # data flows into models and twin
    e.append(arrow(725, 200, 783, 170))
    e.append(arrow(725, 320, 783, 515))
    e.append(arrow(725, 450, 783, 375, dashed=True, color=MUTED))
    e.append(arrow(725, 520, 783, 680, dashed=True, color=MUTED))

    # lane 4: dashboard
    e.append(lane(1200, 95, 370, 700, "4  Dashboard (dashboard/)", "ui"))
    e.append(box(1215, 135, 340, 76, "Service layer", ["the only code the UI calls", "Streamlit + Plotly front end"], "ui"))
    tabs = [("Patient", ["alert chip, forecast sentence", "24 h CGM + 2 h forecast band", "steps, HR, sleep, HRV strip", "why panel, record, twin card"]),
            ("What-if", ["constraint verdicts, blocked reasons", "simulated curves, hypo zone"]),
            ("Triage", ["patients ranked by severity", "one-line reason each"]),
            ("About this twin", ["evaluation, validation,", "limitations, data provenance"])]
    y = 235
    for t, lines in tabs:
        h = 40 + 18 * len(lines)
        e.append(box(1215, y, 340, h, t, lines, "ui", 15))
        y += h + 12
    e.append(box(1215, y + 2, 340, 78, "Always on screen", ["\"Synthetic patient. Decision support only.\"", "sensitivity slider, replay, what happened"], "ui", 15))
    e.append(arrow(1155, 270, 1213, 165))
    e.append(arrow(1155, 680, 1213, 195))

    # legend
    lx, ly = 45, 815
    e.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 40}" y2="{ly}" stroke="{INK}" stroke-width="1.8" marker-end="url(#arr)"/>'
             f'<text x="{lx + 50}" y="{ly + 5}" font-size="13" fill="{INK}">data flow</text>')
    e.append(f'<line x1="{lx + 150}" y1="{ly}" x2="{lx + 190}" y2="{ly}" stroke="{MUTED}" stroke-width="1.8" stroke-dasharray="7 5" marker-end="url(#arr)"/>'
             f'<text x="{lx + 200}" y="{ly + 5}" font-size="13" fill="{INK}">hidden truth: used only to check</text>')
    parts = results_footer().split(" · ")
    e.append(f'<text x="40" y="848" font-size="14" fill="{INK}"><tspan font-weight="700">Held-out results: </tspan>{" · ".join(parts[:2])}</text>')
    if len(parts) > 2:
        e.append(f'<text x="40" y="870" font-size="14" fill="{INK}">{" · ".join(parts[2:])}</text>')
    defs = (f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="Segoe UI, Arial, sans-serif">{defs}<rect width="{W}" height="{H}" fill="white"/>' + "\n".join(e) + "</svg>")


def chrome() -> str:
    for c in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              shutil.which("google-chrome") or "", shutil.which("chromium") or ""):
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome or Edge is needed to render the PDF")


def main() -> None:
    src = DOCS / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "architecture.svg").write_text(svg(), encoding="utf-8")
    html = src / "architecture.html"
    html.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Architecture</title>
<style>@page {{ size: {W}px {H}px; margin: 0 }} html,body {{ margin:0; padding:0; background:white }}</style></head>
<body>{svg()}</body></html>""", encoding="utf-8")
    exe = chrome()
    uri = html.resolve().as_uri()
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={DOCS / 'architecture.pdf'}", uri],
                   check=True, capture_output=True, timeout=120)
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}", f"--screenshot={DOCS / 'architecture.png'}", uri],
                   check=True, capture_output=True, timeout=120)
    print(f"wrote {DOCS / 'architecture.pdf'} and {DOCS / 'architecture.png'}")


if __name__ == "__main__":
    main()
