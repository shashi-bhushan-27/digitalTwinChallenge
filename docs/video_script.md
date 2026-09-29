# Video script (about 20 minutes)

The brief asks for a video of at least 20 minutes, and the submission page says 15-20 minutes, so aim for **20:00 to 20:30**. Upload it to YouTube as **Unlisted** and paste the link in the README's first table.

**What you need open:** `docs/presentation.pptx` (slide show; the speaker notes hold the full narration for every slide), and the dashboard (`uv run streamlit run dashboard/app.py`) in a browser at 100 % zoom on a screen at least 1440 pixels wide. Click through the three demo scenes once before recording so the caches are warm.

**Recording:** 1080p, screen plus voice (PowerPoint's "Record" tab, OBS, or Xbox Game Bar on Windows). Keep the mouse still while talking and pause a second after each click. Say "on synthetic test patients" whenever you quote a number.

## Timeline

| Time | Slide / screen | Length | Key points (full wording in the slide's speaker notes) |
|---|---|---|---|
| 0:00 | 1. Title | 0:45 | Who you are, team AMRIT_VIT, VIT Vellore; one sentence on what the twin does; synthetic patients only. |
| 0:45 | 2. Problem | 1:45 | 101 M adults with diabetes in India; HbA1c hides spikes; CGM + wearables = too much to read; DPDP Act; the use case. |
| 2:30 | 3. What we built | 1:15 | Two streams in (record + CGM/wearables), three things out: predict, explain, simulate. No genetic markers, by design. |
| 3:45 | 4. Architecture | 1:15 | Walk left to right: generator, observed vs hidden stores, models and twin, dashboard; real data only for calibration and validation. |
| 5:00 | 5. Synthetic data | 1:30 | Four layers; hidden variation the record cannot explain (avoids circularity); 400 × 14 days, one seed. |
| 6:30 | 6. Example patients | 0:45 | Grey truth vs blue CGM; three very different responders. |
| 7:15 | 7. Real-data validation | 1:30 | Level gap and level matching; 12 of 14 metrics; TSTR 18.7 vs 19.9 vs 17.5; the honest gap (AUC 0.95, faster real falls). |
| 8:45 | 8. Prediction | 1:00 | 60-min MAE 14.1 vs 24.6; Clarke A+B 99.6 %; PR-AUC 0.848; calibrated probabilities. |
| 9:45 | 9. Ablation | 1:00 | Meal log helps most; wearables help after walks; EHR helps new patients; without a meal log it still works. |
| 10:45 | 10. Alerts | 0:45 | Default 0.5: 92 % caught, 80 min warning, 0.69 false alerts a day; the slider. |
| 11:30 | 11. Explanations | 0:45 | Sentences in mg/dL vs similar patients; three checks; SHAP is never a what-if. |
| 12:15 | 12. Twin validation | 0:45 | Replay against the truth; beats the cohort average; direction right for 94 %; limits. |
| 13:00 | 13. Safety | 0:45 | Constraints first, simulate, flag lows; sulfonylurea stress test. |
| 13:45 | **Live demo** (dashboard) | 5:15 | See below. |
| 19:00 | 16. Limitations | 0:45 | Responsible data; circularity; small non-Indian validation sets; rules await clinical review. |
| 19:45 | 17. Impact and next steps | 0:45 | What it enables; next steps; repository link; thank you. |
| **20:30** | end | | |

Slides 14 and 15 (dashboard screenshots) are for the PDF deck and the jury; in the video, show the live dashboard instead. If you run short, show slide 14 for 30 seconds before switching to the browser.

## Live demo (5:15)

Follow `docs/project_guide.md` section 7.2 for the exact clicks. Suggested timing:

| Time | Scene | What to show |
|---|---|---|
| 13:45 | Tour (0:45) | Banner "Synthetic patient. Decision support only."; the four tabs; the sidebar (scenes, time, replay, sensitivity slider). |
| 14:30 | Scene 1: correct alert (1:45) | Red chip and sentence; the meal on the chart; "Why this prediction"; "Show what actually happened" (peak 218, inside the range); What-if: walk now (small, honest effect); metformin 2000 blocked by eGFR 40. |
| 16:15 | Scene 2: stable patient (1:00) | Green chip; What-if: vegetarian dinner swap; the list of blocked dishes and why; the lower peak. |
| 17:15 | Scene 3: model unsure (0:30) | Amber chip; "the model is uncertain here"; the CGM-gap data-quality chip. |
| 17:45 | Scene 4: hypoglycaemia check (0:45) | SP-0218 at 07:30, breakfast skipped, glimepiride 8 mg: predicted low about 67 mg/dL, risk factors shown. |
| 18:30 | Triage and About (0:30) | Triage at 10:30: who to open first, with reasons; About this twin: evidence and limitations. |

## Checklist before uploading

- [ ] Length between 20:00 and 20:59.
- [ ] Voice is clear; the dashboard text is readable at 1080p.
- [ ] No personal data on screen (browser bookmarks, notifications, other tabs).
- [ ] YouTube visibility set to **Unlisted** and the link opens in a private window.
- [ ] Link added to the README (first table, "Video" row) and pushed.
