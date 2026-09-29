// Builds docs/presentation.pptx (then export docs/presentation.pdf, see README).
//   cd scripts/presentation && npm install && node build_presentation.cjs
// Every number on the slides comes from results/ or docs/*_report.md (dataset v0.1, seed 20261020).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const R = (...p) => path.join(ROOT, ...p);
const OUT = R("docs", "presentation.pptx");

// palette: deep ink (clinical, calm), saffron accent (India), data blue and green, risk red
const INK = "12344D", SAFFRON = "E07A2F", BLUE = "2F6FB0", GREEN = "1E8449", RED = "C0392B";
const TEXT = "1F2933", MUTED = "5B6B7B", TINT = "F2F5F9", LINE = "D5DDE6", ICE = "CADCFC", GREY = "9AA5B1";
const HEAD = "Cambria", BODY = "Calibri";
const W = 13.333, M = 0.6;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Shashi Bhushan (Team AMRIT_VIT)";
pres.title = "T2D Digital Twin: Digital Twin Challenge 2026";

// no line break between a number and its unit ("94 %", "5 min")
const nb = (t) => t.replace(/(\d) (%|min|h|mg)/g, "$1 $2");
const _addSlide = pres.addSlide.bind(pres);
pres.addSlide = (...a) => {
  const s = _addSlide(...a), addText = s.addText.bind(s);
  s.addText = (t, o) => addText(typeof t === "string" ? nb(t) : t.map((r) => ({ ...r, text: nb(r.text) })), o);
  return s;
};

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function fitImage(slide, file, x, y, w, h, alt, align = "center") {
  const s = pngSize(file), r = s.w / s.h;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  const ix = align === "left" ? x : x + (w - iw) / 2;
  const iy = align === "top" || align === "left" ? y : y + (h - ih) / 2;
  slide.addImage({ path: file, x: ix, y: iy, w: iw, h: ih, altText: alt });
  return { x: ix, y: iy, w: iw, h: ih };
}

function title(slide, text) {
  slide.addText(text, { x: M, y: 0.38, w: W - 2 * M, h: 0.85, fontFace: HEAD, fontSize: 30, bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true });
}

let page = 0;
function footer(slide, dark = false) {
  page += 1;
  const c = dark ? "8FA6BD" : GREY;
  slide.addText("Team AMRIT_VIT · T2D digital twin · synthetic patients only", { x: M, y: 7.02, w: 8, h: 0.3, fontFace: BODY, fontSize: 10, color: c, margin: 0, isTextBox: true });
  slide.addText(String(page), { x: W - M - 0.8, y: 7.02, w: 0.8, h: 0.3, fontFace: BODY, fontSize: 10, color: c, align: "right", margin: 0, isTextBox: true });
}

function circle(slide, n, x, y, d = 0.46, fill = SAFFRON) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  slide.addText(String(n), { x, y, w: d, h: d, fontFace: BODY, fontSize: d > 0.4 ? 16 : 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
}

function card(slide, x, y, w, h, fill = TINT) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}

function stat(slide, value, label, x, y, w, color = INK, size = 34) {
  slide.addText(value, { x, y, w, h: 0.62, fontFace: HEAD, fontSize: size, bold: true, color, margin: 0, valign: "bottom", isTextBox: true });
  slide.addText(label, { x, y: y + 0.66, w, h: 0.55, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
}

function bullets(slide, items, x, y, w, h, size = 15, color = TEXT, gap = 8) {
  const arr = items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({
      text: r.text,
      options: { bold: !!r.bold, color: r.color || color, italic: !!r.italic,
                 ...(j === 0 ? { bullet: { indent: 16 }, paraSpaceAfter: gap } : {}),
                 ...(j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) },
    }));
  }).flat();
  slide.addText(arr, { x, y, w, h, fontFace: BODY, fontSize: size, color, valign: "top", margin: 0, isTextBox: true });
}

const chartFrame = (extra) => ({
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: "E3E8EE", size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, titleFontFace: BODY, titleFontSize: 13, titleColor: TEXT, dataLabelFontFace: BODY, dataLabelFontSize: 10,
  dataLabelColor: TEXT, legendFontFace: BODY, legendFontSize: 11, legendColor: TEXT, ...extra,
});

// ---------------------------------------------------------------- 1. title
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("DIGITAL TWIN CHALLENGE 2026  ·  HAPPIEST HEALTH", { x: 0.7, y: 0.75, w: 7, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: SAFFRON, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("A digital twin for Type 2 diabetes", { x: 0.7, y: 1.25, w: 6.6, h: 1.7, fontFace: HEAD, fontSize: 42, bold: true, color: "FFFFFF", margin: 0, valign: "top", isTextBox: true });
  s.addText("Predicts a glucose spike about 2 hours ahead, explains why in clinical language, and lets a doctor test what-ifs on a personalised virtual patient.",
    { x: 0.7, y: 3.1, w: 6.2, h: 1.3, fontFace: BODY, fontSize: 18, color: ICE, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Team AMRIT_VIT", options: { bold: true, color: "FFFFFF", fontSize: 18, breakLine: true } },
    { text: "Shashi Bhushan (team leader)", options: { color: ICE, fontSize: 15, breakLine: true } },
    { text: "Vellore Institute of Technology (VIT), Vellore", options: { color: ICE, fontSize: 15 } },
  ], { x: 0.7, y: 4.75, w: 6.2, h: 1.2, fontFace: BODY, margin: 0, valign: "top", isTextBox: true });
  s.addText("Synthetic patients only. Decision support prototype, not a medical device.", { x: 0.7, y: 6.55, w: 7, h: 0.35, fontFace: BODY, fontSize: 11, italic: true, color: "8FA6BD", margin: 0, isTextBox: true });
  card(s, 7.45, 1.0, 5.3, 5.3, "FFFFFF");
  fitImage(s, R("docs", "img", "panel_hero.png"), 7.6, 1.15, 5.0, 5.0, "Dashboard chart: 24 hours of CGM with the 2-hour forecast band and wearable strips");
  s.addNotes("Hello, I am Shashi Bhushan from VIT Vellore, presenting team AMRIT_VIT's entry. This is a proof-of-concept digital twin of a person with Type 2 diabetes. It fuses the static clinical record with continuous glucose monitor and wearable streams, predicts a glucose spike about two hours ahead, explains the prediction in clinical language, and lets a doctor ask what-if questions on a personalised physiological model of that patient. Every patient you will see is synthetic; real open datasets were used only to calibrate and validate the simulator.");
  page += 1;
}

// ---------------------------------------------------------------- 2. problem
{
  const s = pres.addSlide();
  title(s, "The problem: plenty of glucose data, a few minutes per patient");
  bullets(s, [
    "India has an estimated 101 million adults with diabetes (ICMR-INDIAB, 2023). Type 2 diabetes often starts younger and at a lower BMI than in Western cohorts.",
    "HbA1c is a three-month average: it cannot show when, or why, glucose spikes after today's meals.",
    "A CGM gives 288 readings a day, and a wrist wearable adds heart rate, HRV, sleep and steps. Nobody can read all of it in a short consultation.",
    "Real patient data cannot be freely shared to build AI (DPDP Act 2023), so development needs realistic synthetic patients.",
  ], M, 1.45, 7.2, 3.6, 16);
  card(s, M, 5.2, 7.2, 1.55);
  s.addText([
    { text: "Healthcare use case", options: { bold: true, color: INK, fontSize: 15, breakLine: true } },
    { text: "A diabetologist following CGM-wearing Type 2 patients, in clinic or remotely: who needs attention in the next 2 hours, why, and which change to discuss (a meal, a walk, or a medicine).", options: { color: TEXT, fontSize: 14 } },
  ], { x: M + 0.25, y: 5.3, w: 6.7, h: 1.35, fontFace: BODY, margin: 0, valign: "middle", isTextBox: true });
  const sx = 8.45;
  [["101 M", "adults with diabetes in India (ICMR-INDIAB, Lancet Diabetes Endocrinol 2023)", SAFFRON],
   ["4,032", "CGM readings per patient in two weeks (one every 5 minutes)", BLUE],
   ["0", "real patient records used to build or train anything here", GREEN]].forEach(([v, l, c], i) => {
    card(s, sx, 1.45 + i * 1.8, 4.3, 1.6);
    stat(s, v, l, sx + 0.3, 1.55 + i * 1.8, 3.8, c, 36);
  });
  s.addNotes("The condition is Type 2 diabetes, chosen because India carries one of the largest burdens in the world: the ICMR-INDIAB study estimated about 101 million adults with diabetes in 2023. HbA1c, the routine control marker, is a three-month average and hides when and why glucose spikes. Continuous glucose monitors and wearables fill that gap, but they generate thousands of readings that a doctor cannot read in a few minutes. And because real patient data is protected under the DPDP Act, we built everything on realistic synthetic patients. The use case is a diabetologist following CGM-wearing patients who needs to know who needs attention now, why, and what change to discuss.");
  footer(s);
}

// ---------------------------------------------------------------- 3. solution
{
  const s = pres.addSlide();
  title(s, "What we built: a twin that predicts, explains and simulates");
  const inputs = [
    ["Static record (EHR)", "Age, sex, BMI, waist · HbA1c and its 12-month trend · fasting glucose · eGFR, ALT, fatty liver · lipids, BP · medicines with dose and adherence · allergies and intolerances · diet (vegetarian, non-vegetarian, Jain)"],
    ["Dynamic streams", "CGM every 5 minutes · heart rate and HRV · sleep stages · steps · the patient's own meal log (optional, often incomplete)"],
  ];
  inputs.forEach(([h, b], i) => {
    const y = 1.5 + i * 2.65;
    card(s, M, y, 3.9, 2.4);
    s.addText(h, { x: M + 0.25, y: y + 0.15, w: 3.4, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: M + 0.25, y: y + 0.65, w: 3.45, h: 1.65, fontFace: BODY, fontSize: 13, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.75, y: 3.55, w: 0.7, h: 0.55, fill: { color: SAFFRON }, line: { color: SAFFRON } });
  const pillars = [
    ["Predict", "Calibrated probability that glucose goes above 180 mg/dL within 2 hours, plus a 15-120 minute forecast with an 80 % band. Model: LightGBM fusion of all streams."],
    ["Explain", "SHAP contributions to the predicted peak, grouped into clinical concepts and written as sentences in mg/dL, compared with similar patients."],
    ["Simulate", "A personalised glucose-insulin ODE twin answers \"what if?\" for meals, walks and medicines. Constraints are checked first; every result gets a hypoglycaemia check."],
  ];
  pillars.forEach(([h, b], i) => {
    const y = 1.5 + i * 1.72;
    circle(s, i + 1, 5.85, y + 0.05, 0.55);
    s.addText(h, { x: 6.6, y, w: 6.1, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: 6.6, y: y + 0.48, w: 6.1, h: 1.05, fontFace: BODY, fontSize: 14, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("All three are shown to the doctor in one dashboard: Patient · What-if · Triage · About this twin.", { x: 5.85, y: 6.45, w: 6.9, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("The twin fuses two kinds of input, exactly as the brief asks. On the left, the static record: demographics, HbA1c and its trend, kidney and liver markers, medicines with adherence, allergies and diet type. And the dynamic streams: CGM every five minutes, heart rate, HRV, sleep stages, steps, and an optional meal log. From these it does three things. It predicts, with a calibrated probability and a forecast band. It explains, in clinical sentences. And it simulates, using a physiological model personalised to the patient, so the doctor can test a change before suggesting it. Deliberately, no genetic markers are used; they are future work.");
  footer(s);
}

// ---------------------------------------------------------------- 4. architecture
{
  const s = pres.addSlide();
  title(s, "Architecture");
  fitImage(s, R("docs", "architecture.png"), M, 1.3, W - 2 * M, 5.6, "Architecture diagram: synthetic data layers, data stores, prediction and twin, dashboard");
  s.addNotes("Here is the whole system. Left to right: the synthetic data generator in four layers, the data stores, where what a clinic would observe is kept strictly apart from the simulator's hidden truth, then the two engines: the LightGBM prediction and explanation models, and the personalised ODE twin. Both feed the Streamlit dashboard through a small service layer, so dashboard code stays separate from model code. The real open datasets, Shanghai T2DM and CGMacros, only touch calibration and validation; they are never used for training. Everything is config-driven and reproducible from one seed. The same diagram is in the repository as docs/architecture.pdf.");
  footer(s);
}

// ---------------------------------------------------------------- 5. synthetic data
{
  const s = pres.addSlide();
  title(s, "Realistic synthetic patients, built in four layers");
  const layers = [
    ["Cohort (the record)", "Gaussian copula over age, BMI, waist, HbA1c, fasting glucose, eGFR, lipids and BP with Indian-clinic distributions; allergies, diet type, eGFR-aware prescribing. No genetic fields."],
    ["Hidden physiology", "Insulin sensitivity, beta-cell function and liver glucose output derived from the record plus variation the record cannot explain; a Bergman / Dalla Man glucose-insulin ODE with drugs and exercise."],
    ["Behaviour", "47 Indian plates (idli-sambar, rajma chawal, poha...), late dinners, weekend shifts, walks, sleep, stress and illness days, missed doses, and a partial, biased meal log."],
    ["Sensors", "CGM lag of 5-15 min, noise, drift, warm-up, dropouts and compression lows; wearable gaps, phantom steps and noisy HRV."],
  ];
  layers.forEach(([h, b], i) => {
    const y = 1.45 + i * 1.33;
    circle(s, i + 1, M, y + 0.05);
    s.addText(h, { x: M + 0.7, y, w: 6.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: M + 0.7, y: y + 0.42, w: 6.8, h: 0.85, fontFace: BODY, fontSize: 13, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  });
  const gx = 8.65, gw = 2.0;
  [["400 × 14", "patients × days, split 70/15/15 by patient", INK],
   ["1.6 M", "CGM readings, 2.8 % missing", BLUE],
   ["7.3 %", "CGM error (MARD) vs the hidden true glucose", SAFFRON],
   ["8.3 %", "median HbA1c, 45 % time in range", GREEN]].forEach(([v, l, c], i) => {
    const x = gx + (i % 2) * (gw + 0.2), y = 1.5 + Math.floor(i / 2) * 1.9;
    card(s, x, y, gw, 1.7);
    stat(s, v, l, x + 0.2, y + 0.12, gw - 0.35, c, 28);
  });
  card(s, gx, 5.4, 2 * gw + 0.2, 1.35);
  s.addText("Every patient regenerates identically from one seed. The hidden truth (true glucose, meals eaten, doses taken) is stored apart from what a clinic would see, and tests check nothing hidden leaks into the models.",
    { x: gx + 0.2, y: 5.45, w: 2 * gw - 0.2, h: 1.25, fontFace: BODY, fontSize: 12, color: TEXT, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("The synthetic data is built in four layers. First the cohort: a Gaussian copula samples correlated clinical values with Indian-clinic distributions, then allergies, diet type and medicines assigned by kidney-aware rules. Second, hidden physiology: each patient gets insulin sensitivity, beta-cell function and liver glucose output, derived from the record but with variation the record cannot explain, so models cannot simply rediscover a rule. A glucose-insulin differential equation model then simulates every minute. Third, behaviour: 47 Indian meals, realistic timing, walks, sleep, stress and illness days, missed doses, and a meal log that is incomplete, like real life. Fourth, sensor realism: CGM lag, noise, drift and gaps, and wearable artefacts. The result is 400 patients over 14 days, 1.6 million CGM readings.");
  footer(s);
}

// ---------------------------------------------------------------- 6. example patients
{
  const s = pres.addSlide();
  title(s, "What the generator produces: three patients, three days");
  fitImage(s, R("results", "data", "example_patients.png"), M, 1.3, 8.6, 5.6, "CGM and true glucose for three synthetic patients over three days", "left");
  bullets(s, [
    [{ text: "Grey: ", bold: true }, { text: "true plasma glucose, hidden from every model." }],
    [{ text: "Blue: ", bold: true }, { text: "what the CGM reports, with lag, noise and gaps." }],
    [{ text: "Brown: ", bold: true }, { text: "carbohydrate eaten. Green: steps. Purple: sleep." }],
    "The same meal gives very different curves at HbA1c 7.0, 8.4 and 13.0 %.",
    "Because we know the truth, we can test the models and the twin in ways real data never allows.",
  ], 9.55, 1.45, 3.2, 5.3, 14, TEXT, 10);
  s.addNotes("This is what three synthetic patients look like over three days. The grey line is the true plasma glucose, which no model ever sees. The blue dots are what the CGM reports, including lag, noise and dropouts. Carbohydrate labels, steps and sleep strips come from the behaviour layer. Notice how differently the three people respond: the top patient with HbA1c 7 still spikes after large meals, while the bottom patient with HbA1c 13 never gets back into range. Keeping the true values lets us validate the models and the what-if twin against ground truth, which is impossible with real data.");
  footer(s);
}

// ---------------------------------------------------------------- 7. validation
{
  const s = pres.addSlide();
  title(s, "Checked against real Type 2 CGM data");
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: INK } } });
  const rows = [
    [hdr("metric (held-out Shanghai, 49 patients)"), hdr("real"), hdr("synthetic")],
    ["coefficient of variation", "0.27", "0.26"],
    ["night / mean glucose", "0.82", "0.82"],
    ["time to post-meal peak", "90 min", "93 min"],
    ["spikes above 180 per day", "1.63", "1.60"],
    ["MAGE (outside the band)", "75", "92"],
    ["fastest 15-min fall, p5 (outside)", "−11", "−14"],
  ];
  s.addTable(rows, { x: M, y: 1.45, w: 6.3, colW: [3.9, 1.2, 1.2], fontFace: BODY, fontSize: 13, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, rowH: 0.42, valign: "middle", margin: 0.06 });
  s.addText("12 of 14 metrics fall within half an interquartile range of the real median. Real CGM runs lower than HbA1c implies (Shanghai: mean 142 vs HbA1c-estimated 194 mg/dL), so each real patient is compared with the synthetic patient of closest mean glucose.",
    { x: M, y: 4.6, w: 6.3, h: 1.0, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  card(s, M, 5.7, 6.3, 1.1);
  s.addText([
    { text: "Where it still differs: ", options: { bold: true, color: RED } },
    { text: "a classifier can tell real from synthetic 12-hour windows (AUC 0.95), and real glucose falls faster after peaks (−18 to −20 vs −15 mg/dL per 15 min).", options: { color: TEXT } },
  ], { x: M + 0.2, y: 5.75, w: 5.9, h: 1.0, fontFace: BODY, fontSize: 13, margin: 0, valign: "middle", isTextBox: true });
  s.addChart(pres.charts.BAR, [{ name: "60-min MAE on real patients (mg/dL)", labels: ["Persistence (no model)", "Trained on SYNTHETIC", "Trained on real (reference)"], values: [19.9, 18.7, 17.5] }],
    { x: 7.3, y: 1.4, w: 5.45, h: 4.3, barDir: "bar", chartColors: [BLUE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 22,
      showLegend: false, title: "Train on synthetic, test on real (TSTR): 60-min forecast error", catAxisOrientation: "maxMin", ...chartFrame({ valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12 }) });
  s.addText("A CGM-only model trained only on synthetic patients beats persistence on real Shanghai patients and comes close to a model trained on real data. Real data calibrates 7 settings and validates; it never trains the product models.",
    { x: 7.3, y: 5.8, w: 5.45, h: 1.0, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Is the synthetic data realistic? We checked against two open real datasets: Shanghai T2DM and CGMacros. Real data was used only to tune seven population-level settings on half the Shanghai patients and to validate on the rest; it never trains the product models. One finding: real CGM runs well below the level HbA1c implies, because recordings capture monitored, treated periods, so comparisons are level-matched. On the held-out half, 12 of 14 dynamics metrics fall within half an interquartile range of the real data. The strongest test is train-on-synthetic, test-on-real: a forecaster trained only on synthetic patients beats persistence on real patients and comes close to one trained on real data. And we are honest about the gap: a classifier still separates real from synthetic windows with AUC 0.95, mostly because real glucose falls faster after peaks.");
  footer(s);
}

// ---------------------------------------------------------------- 8. prediction
{
  const s = pres.addSlide();
  title(s, "Forecast and spike detection on held-out patients");
  s.addChart(pres.charts.BAR, [
    { name: "Persistence (glucose stays the same)", labels: ["30 min", "60 min", "120 min"], values: [13.69, 24.64, 41.19] },
    { name: "Fusion model (CGM + wearables + EHR + meal log)", labels: ["30 min", "60 min", "120 min"], values: [7.91, 14.14, 22.73] },
  ], { x: M, y: 1.4, w: 7.0, h: 4.6, barGrouping: "clustered", chartColors: [GREY, BLUE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
       showLegend: true, legendPos: "b", title: "Forecast error (MAE, mg/dL) by horizon: lower is better", valAxisMinVal: 0, ...chartFrame({ valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 13 }) });
  s.addText("60 test patients never seen in training (230,609 prediction times). Spike = CGM above 180 mg/dL for at least 15 minutes within the next 2 hours.",
    { x: M, y: 6.1, w: 7.0, h: 0.7, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  const gx = 8.0, gw = 2.25;
  [["14.1", "mg/dL error at 60 min (persistence 24.6)", BLUE],
   ["99.6 %", "of 60-min forecasts in Clarke zones A+B", GREEN],
   ["0.848", "spike PR-AUC (AUROC 0.915; spike rate 35 %)", SAFFRON],
   ["0.115", "Brier score after isotonic calibration", INK],
   ["79 %", "of outcomes inside the 80 % band", BLUE],
   ["80 min", "median warning before a spike (threshold 0.5)", RED]].forEach(([v, l, c], i) => {
    const x = gx + (i % 2) * (gw + 0.25), y = 1.45 + Math.floor(i / 2) * 1.8;
    card(s, x, y, gw, 1.6);
    stat(s, v, l, x + 0.18, y + 0.08, gw - 0.3, c, 28);
  });
  s.addNotes("Now the prediction model. It is a LightGBM model that fuses CGM features, wearable summaries, the record and the meal log, built strictly from data up to the prediction moment, with a test that proves no future data leaks in. On 60 held-out patients, the 60-minute forecast error is 14.1 mg/dL against 24.6 for persistence, and 99.6 % of forecasts fall in Clarke error-grid zones A and B. The 80 % band is conformal-calibrated and covers 79 % of outcomes. For spike detection, PR-AUC is 0.848 and the probabilities are isotonic-calibrated, with a Brier score of 0.115, so 70 % really means about 70 %. At the default threshold the median warning is 80 minutes before the spike.");
  footer(s);
}

// ---------------------------------------------------------------- 9. ablation
{
  const s = pres.addSlide();
  title(s, "Does fusing the streams help? An honest ablation");
  s.addChart(pres.charts.BAR, [{ name: "MAE at 60 min", labels: ["Persistence", "Linear extrapolation", "CGM only", "+ wearables", "+ EHR record", "+ meal log (fusion)"], values: [24.64, 28.93, 14.82, 14.93, 14.92, 14.14] }],
    { x: M, y: 1.4, w: 6.4, h: 5.3, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [BLUE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
      showLegend: false, title: "60-minute forecast error (mg/dL), test patients", valAxisMinVal: 0, valAxisMaxVal: 33, ...chartFrame({ valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12 }) });
  const rows = [
    ["Meal log", "Clear gain: 60-min error −0.78 mg/dL [−0.90, −0.64], PR-AUC +0.010.", GREEN],
    ["Wearables", "Help where physiology says they should: right after a walk, −0.90 mg/dL [−1.64, −0.24]. Slightly worse overall (+0.11).", BLUE],
    ["EHR record", "Adds PR-AUC +0.010 in a patient's first 24 h, when CGM history is short. No overall effect.", SAFFRON],
    ["No meal log", "Robust to messy inputs: CGM + wearables alone still reach 14.9 mg/dL (persistence 24.6).", INK],
  ];
  rows.forEach(([h, b, c], i) => {
    const y = 1.5 + i * 1.18;
    circle(s, i + 1, 7.4, y + 0.02, 0.42, c);
    s.addText(h, { x: 8.0, y, w: 4.75, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: 8.0, y: y + 0.38, w: 4.75, h: 0.75, fontFace: BODY, fontSize: 13, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Paired differences with patient-level bootstrap 95 % intervals. With hours of CGM history, the trace already carries most of today's physiology.",
    { x: 7.4, y: 6.25, w: 5.35, h: 0.6, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Does fusion actually help? We ran an ablation ladder: persistence, CGM only, then adding wearables, the record and the meal log, and compared them with paired patient-level bootstrap intervals. The honest answer is nuanced. The meal log is the clear winner. Wearables slightly hurt overall but help exactly where physiology says they should, right after a walk. The record adds a small gain in a patient's first day, when CGM history is short. And following the brief's point about messy inputs, we report the model without any meal log: it still beats persistence by a wide margin. With hours of CGM history, the glucose trace itself carries most of what the record tells you about today; the record matters most for new patients and for the twin.");
  footer(s);
}

// ---------------------------------------------------------------- 10. alerts
{
  const s = pres.addSlide();
  title(s, "Alerts a clinic can live with");
  const th = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7"];
  s.addChart(pres.charts.LINE, [
    { name: "Spikes caught (%)", labels: th, values: [97.9, 96.8, 93.8, 92.2, 89.6, 84.9] },
    { name: "Alerts followed by a spike (%)", labels: th, values: [44.4, 50.5, 61.8, 68.2, 76.5, 85.9] },
  ], { x: M, y: 1.4, w: 7.1, h: 4.4, chartColors: [BLUE, SAFFRON], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 8, showLegend: true, legendPos: "b",
       title: "Alert threshold (calibrated probability) vs catch rate and precision", valAxisMinVal: 40, valAxisMaxVal: 100, showCatAxisTitle: true, catAxisTitle: "alert threshold",
       catAxisTitleFontSize: 11, catAxisTitleColor: MUTED, ...chartFrame({}) });
  const fr = [["threshold", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"], ["false alerts / patient-day", "1.32", "1.15", "0.85", "0.69", "0.49", "0.27"]];
  s.addTable(fr.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: j === 4 ? RED : TEXT, fill: { color: i === 0 ? TINT : "FFFFFF" } } }))),
    { x: M, y: 5.95, w: 7.1, colW: [2.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], fontFace: BODY, fontSize: 12, border: { type: "solid", color: LINE, pt: 0.75 }, rowH: 0.36, align: "center", valign: "middle", margin: 0.04 });
  const gx = 8.2;
  s.addText("Dashboard default: threshold 0.5", { x: gx, y: 1.45, w: 4.5, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0, isTextBox: true });
  [["92 %", "of spikes caught", BLUE], ["80 min", "median warning time", SAFFRON], ["0.69", "false alert episodes per patient-day", RED], ["68 %", "of alerts followed by a spike", GREEN]].forEach(([v, l, c], i) => {
    const x = gx + (i % 2) * 2.3, y = 2.0 + Math.floor(i / 2) * 1.65;
    card(s, x, y, 2.1, 1.45);
    stat(s, v, l, x + 0.17, y + 0.05, 1.8, c, 28);
  });
  s.addText("The doctor's sensitivity slider moves along this curve: at 0.36, 96 % of spikes are caught at 1.0 false alert a day; at 0.7, 85 % at 0.27 a day. Fewer false alarms means less alert fatigue.",
    { x: gx, y: 5.45, w: 4.5, h: 1.3, fontFace: BODY, fontSize: 13, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("A good model is not enough; alerts must be usable. Alert fatigue is a real problem in clinics, so the dashboard has a sensitivity slider mapped to the model's operating point. This chart shows the trade-off on test patients. We chose 0.5 as the default: it catches 92 % of spikes with a median warning of 80 minutes, and produces 0.69 false alert episodes per patient per day. A doctor who wants to miss fewer spikes can move the slider down, accepting more alarms, and the screen always shows the resulting catch rate and false alerts per day.");
  footer(s);
}

// ---------------------------------------------------------------- 11. explanations
{
  const s = pres.addSlide();
  title(s, "Why this alert: explanations a doctor can read");
  const ps = pngSize(R("docs", "img", "panel_why.png")), pw = 7.5, ph = pw * ps.h / ps.w;
  s.addImage({ path: R("docs", "img", "panel_why.png"), x: M, y: 1.45 + (5.35 - ph) / 2, w: pw, h: ph, altText: "The dashboard's 'Why this prediction' panel with drivers in mg/dL" });
  s.addText("Scene 1 (SP-0218, 13:05): a meal logged a minute ago pushes the predicted peak up; her current glucose of 116 pulls it down.",
    { x: M, y: 1.45 + (5.35 + ph) / 2 + 0.1, w: pw, h: 0.6, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  const gx = 8.55;
  [["99.8 %", "after a logged meal of 60 g or more, \"logged meals\" is a top-3 driver", GREEN],
   ["19.5 vs 5.6", "mg/dL moved by removing the top-3 concepts vs 3 random ones (deletion test)", BLUE],
   ["0.75 vs 0.04", "explanation similarity for similar moments vs random pairs", SAFFRON]].forEach(([v, l, c], i) => {
    const y = 1.45 + i * 1.45;
    stat(s, v, l, gx, y, 4.2, c, 28);
  });
  card(s, gx, 5.85, 4.2, 0.95);
  s.addText("SHAP shows what the model relied on, never the effect of a change. What-if answers come only from the ODE twin.",
    { x: gx + 0.18, y: 5.9, w: 3.9, h: 0.85, fontFace: BODY, fontSize: 12, color: TEXT, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Every alert comes with a reason. We run SHAP on the predicted two-hour peak, so contributions are in mg/dL, not log-odds. Features are grouped into clinical concepts, such as recent glucose trend, logged meals, baseline control or last night's sleep, and compared with similar patients of the same age band and diabetes duration. The top drivers become sentences like 'Logged meal of about 80 g carbohydrate 1 minute ago, plus 37 mg/dL'. Contributions sum exactly to the prediction. Because we own the hidden truth, we validated the explanations: after a large logged meal, logged meals is a top-3 driver 99.8 % of the time; removing the top drivers moves the prediction far more than removing random ones; and similar moments get similar explanations. One important safety rule: SHAP is never presented as an intervention.");
  footer(s);
}

// ---------------------------------------------------------------- 12. twin validation
{
  const s = pres.addSlide();
  title(s, "What-if answers, checked against the truth");
  const labels = ["Swap to a low-GI plate", "Half portion", "Skip the meal", "20-min walk after"];
  s.addChart(pres.charts.BAR, [
    { name: "Personalised twin", labels, values: [13.1, 13.3, 27.1, 3.3] },
    { name: "Record-only twin", labels, values: [16.2, 13.0, 25.8, 3.3] },
    { name: "Cohort average", labels, values: [31.0, 15.5, 34.7, 2.7] },
  ], { x: M, y: 1.4, w: 7.4, h: 5.35, barGrouping: "clustered", chartColors: [BLUE, "7FA7CF", GREY], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
       showLegend: true, legendPos: "b", title: "Error in the predicted change of the 5-hour peak (mg/dL, lower is better)", valAxisMinVal: 0,
       ...chartFrame({ valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12 }) });
  bullets(s, [
    "Only synthetic data allows this test: each of 50 test patients is replayed with the change applied to their true meals, doses and activity.",
    "True median effects: swap −34, half portion −45, skip −84, walk −5 mg/dL.",
    "The predicted change has the right direction for 94 % of patients (84 % for the swap). Personalisation helps most for dish swaps (Spearman 0.91).",
    "Both twins beat the cohort average for every meal change. Large cuts are under-estimated, and a walk's small effect is not predictable.",
  ], 8.35, 1.5, 4.4, 5.3, 14, TEXT, 12);
  s.addNotes("The twin is a glucose-insulin model personalised to each patient: it starts from the physiology implied by the record, then fits insulin sensitivity, beta-cell function and the fasting set point to a week of the patient's own CGM and meal log. It never reads the simulator's hidden values. Can we trust its what-if answers? With synthetic data we can check: we replay each test patient's day with the change applied to their true meals and doses, and compare with what the twin predicted. The personalised twin beats the cohort average for every meal change, gets the direction right for 94 % of patients, and personalisation clearly helps for dish swaps. We also report where it is weak: large cuts are under-estimated, and a short walk's effect is too small to predict usefully.");
  footer(s);
}

// ---------------------------------------------------------------- 13. safety
{
  const s = pres.addSlide();
  title(s, "Constraints first, a safety check always");
  const steps = [
    ["Check", "Diet type, allergies, intolerances and eGFR drug limits are checked before anything is simulated. Blocked options show why."],
    ["Simulate", "Allowed options run on the patient's personalised twin for the next 5 hours, or as a typical day for a medicine change."],
    ["Flag", "A predicted trough below 70 mg/dL is flagged as hypoglycaemia (below 80: caution), with the patient's risk factors."],
  ];
  steps.forEach(([h, b], i) => {
    const y = 1.45 + i * 1.3;
    circle(s, i + 1, M, y + 0.03);
    s.addText(h, { x: M + 0.7, y, w: 6.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: M + 0.7, y: y + 0.42, w: 6.3, h: 0.8, fontFace: BODY, fontSize: 13, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, M, 5.4, 7.0, 0.62, "FDECEA");
  s.addText("⛔  Metformin 1000 → 2000 mg/day: BLOCKED. eGFR 40 (30-44): metformin limited to 1000 mg/day.", { x: M + 0.2, y: 5.4, w: 6.7, h: 0.62, fontFace: BODY, fontSize: 13, bold: true, color: RED, margin: 0, valign: "middle", isTextBox: true });
  s.addText("Every result is labelled \"Simulation for discussion with the patient, not a prescription\". The rules are listed for clinical review (docs/clinical_rules_to_review.md).",
    { x: M, y: 6.15, w: 7.0, h: 0.7, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  const im = fitImage(s, R("results", "twin", "hypo_scenario.png"), 8.05, 1.35, 4.7, 3.9, "Twin's predicted trough vs true trough in a sulfonylurea stress test");
  s.addText("Stress test on 24 sulfonylurea users (maximum dose, breakfast skipped): all 4 true lows were flagged at least \"caution\", 1 as a predicted low, with no false alarms.",
    { x: 8.05, y: im.y + im.h + 0.15, w: 4.7, h: 1.1, fontFace: BODY, fontSize: 12, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Safety is built into the flow. First, constraints: before anything is simulated, options that conflict with the patient's diet, allergies, intolerances or kidney function are blocked and the reason is shown, for example metformin limited to 1000 mg a day at an eGFR of 40. Second, allowed options are simulated on the personalised twin. Third, every result carries a hypoglycaemia check with the patient's risk factors, such as sulfonylurea use, lower eGFR or older age. We stress-tested this on sulfonylurea users given the maximum dose with breakfast skipped: all true lows were at least flagged as caution, with no false alarms. Every result says it is a simulation for discussion, not a prescription, and the rules are listed for review by a clinician.");
  footer(s);
}

// ---------------------------------------------------------------- 14. dashboard main
{
  const s = pres.addSlide();
  title(s, "The doctor's screen: one patient at a glance");
  const im = fitImage(s, R("docs", "img", "dashboard_patient.png"), M, 1.3, 8.3, 5.6, "Patient tab of the dashboard", "left");
  const notes = [
    ["Alert chip and one sentence", "\"Spike likely (90 %): above 180 in about 50 min, peak about 214 mg/dL (80 % range 193-263).\""],
    ["24 h CGM + 2 h forecast", "Target range, spike line, logged meals and the 80 % band."],
    ["Wearables on the same time axis", "Steps, heart rate, sleep stages and HRV."],
    ["Record card", "HbA1c trend, eGFR, medicines, allergy badges; the twin's state below."],
    ["Sensitivity slider and replay", "Replay steps the twin forward; \"show what actually happened\" for the demo."],
  ];
  notes.forEach(([h, b], i) => {
    const y = 1.35 + i * 1.1;
    circle(s, i + 1, im.x + im.w + 0.3, y + 0.02, 0.4);
    s.addText(h, { x: im.x + im.w + 0.85, y, w: W - M - (im.x + im.w + 0.85), h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: im.x + im.w + 0.85, y: y + 0.35, w: W - M - (im.x + im.w + 0.85), h: 0.7, fontFace: BODY, fontSize: 12, color: TEXT, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("This is the main screen a doctor sees, built in Streamlit and Plotly. A persistent banner says synthetic patient, decision support only. The header shows the patient, the alert chip and one plain-language sentence with the timing, the peak and its range. The hero chart shows the last 24 hours of CGM and the two-hour forecast with its band, with meals marked, and the wearable strip on the same time axis. On the left is the record with the HbA1c trend, kidney function, medicines and allergy badges, and below it the twin's state: insulin sensitivity and secretion compared with similar patients. The sidebar has three scripted demo scenes, a replay button, and the alert-sensitivity slider.");
  footer(s);
}

// ---------------------------------------------------------------- 15. what-if + triage
{
  const s = pres.addSlide();
  title(s, "What-if on the twin, and triage across patients");
  const a = fitImage(s, R("docs", "img", "panel_whatif.png"), M, 1.3, 4.6, 5.0, "What-if tab: a dish swap simulated on the twin", "left");
  const bx = a.x + a.w + 0.45;
  const b = fitImage(s, R("docs", "img", "panel_triage.png"), bx, 1.3, W - M - bx, 5.0, "Triage tab: patients ranked by predicted risk", "left");
  const cy = Math.max(a.y + a.h, b.y + b.h) + 0.15;
  s.addText([{ text: "What-if: ", options: { bold: true, color: INK } }, { text: "patient factors as chips, the current plan vs the change, and the hypoglycaemia check.", options: { color: TEXT } }],
    { x: a.x, y: cy, w: a.w, h: 0.75, fontFace: BODY, fontSize: 12, margin: 0, valign: "top", isTextBox: true });
  s.addText([{ text: "Triage: ", options: { bold: true, color: INK } }, { text: "every patient at one clinic time, predicted lows first, then the highest 2-hour peak, with a one-line reason and data quality.", options: { color: TEXT } }],
    { x: b.x, y: cy, w: b.w, h: 0.75, fontFace: BODY, fontSize: 12, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("The What-if tab lets the doctor pick a change: swap the next meal for another Indian dish, change the portion, add a walk, skip a meal, or change a medicine. Patient factors that were considered appear as chips at the top. Blocked options are listed with their reasons. The result shows the current plan and the change side by side, with the peak, minutes above 180, the lowest glucose, and the hypoglycaemia check. The Triage tab ranks every patient at a given clinic time, predicted lows first and then the highest predicted two-hour peak, each with a one-line reason and a data-quality flag, so the doctor knows whom to open first. A fourth tab, About this twin, shows the evaluation, real-data validation and limitations.");
  footer(s);
}

// ---------------------------------------------------------------- 16. limitations
{
  const s = pres.addSlide();
  title(s, "Responsible data use and honest limitations");
  const colW = (W - 2 * M - 0.5) / 2;
  card(s, M, 1.4, colW, 4.75);
  s.addText("Responsible data (DPDP-aware)", { x: M + 0.3, y: 1.55, w: colW - 0.6, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: GREEN, margin: 0, isTextBox: true });
  bullets(s, [
    "No personal data anywhere: every patient is simulated.",
    "Real open datasets are downloaded by script, never committed, and used only to calibrate and validate (licences respected; CGMacros is non-commercial).",
    "Hidden simulator truth is stored apart; tests check nothing hidden leaks into model inputs.",
    "Every literature value carries a source and a \"verified\" flag (43 still to verify).",
    "\"Synthetic patient. Decision support only.\" on every screen. No genetic markers, by design.",
  ], M + 0.3, 2.15, colW - 0.6, 3.9, 14, TEXT, 9);
  const x2 = M + colW + 0.5;
  card(s, x2, 1.4, colW, 4.75);
  s.addText("Limitations", { x: x2 + 0.3, y: 1.55, w: colW - 0.6, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: RED, margin: 0, isTextBox: true });
  bullets(s, [
    "Models trained and scored on simulated patients can partly rediscover the simulator's assumptions; scores are an upper bound for real patients.",
    "The real validation sets are small and not Indian, and a classifier still separates real from synthetic windows (AUC 0.95).",
    "A week of CGM does not pin down the twin's parameters exactly, so the twin card shows its uncertainty.",
    "Indian population values, meal macros and drug rules await expert review before any clinical use.",
  ], x2 + 0.3, 2.15, colW - 0.6, 3.9, 14, TEXT, 9);
  s.addText("Details: docs/data_card.md (circularity), docs/validation_report.md, docs/parameters_to_verify.md, docs/clinical_rules_to_review.md.",
    { x: M, y: 6.35, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("We designed for responsible data use from the start. There is no personal data anywhere. Real open datasets are downloaded by script, never committed, used only for calibration and validation, and their licences are respected. The simulator's hidden truth is kept apart, and tests check that nothing hidden leaks into the models. Every value taken from literature is tagged with its source and whether it has been verified. And we are open about the limitations: models trained on simulated patients can partly rediscover the simulator's assumptions, so the scores are an upper bound; the real validation sets are small and not Indian; the twin's parameters are uncertain after one week; and the Indian parameters and drug rules need clinical review before any real use.");
  footer(s);
}

// ---------------------------------------------------------------- 17. impact
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("Impact and next steps", { x: 0.7, y: 0.5, w: 11.9, h: 0.9, fontFace: HEAD, fontSize: 36, bold: true, color: "FFFFFF", margin: 0, valign: "middle", isTextBox: true });
  const colW = 5.7;
  s.addText("What it enables", { x: 0.7, y: 1.65, w: colW, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: SAFFRON, margin: 0, isTextBox: true });
  bullets(s, [
    "Triage of CGM-monitored patients by 2-hour risk, with a reason for each.",
    "Alerts about 80 minutes ahead, explained in clinical language.",
    "Meal, walk and medicine changes discussed on the patient's own twin, with Indian foods and safety checks.",
    "Privacy-safe development: realistic synthetic patients instead of real records.",
  ], 0.7, 2.2, colW, 3.6, 15, "FFFFFF", 10);
  s.addText("Next steps", { x: 6.95, y: 1.65, w: colW, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: SAFFRON, margin: 0, isTextBox: true });
  bullets(s, [
    "Validate on an Indian CGM cohort; clinician review of rules and sentences.",
    "Prandial and premixed insulin, GLP-1 receptor agonists.",
    "Patient-side nudges (WhatsApp / SMS) and regional-language summaries.",
    "A silent-mode prospective study before any clinical use.",
  ], 6.95, 2.2, colW, 3.6, 15, "FFFFFF", 10);
  s.addText([
    { text: "github.com/shashi-bhushan-27/digitalTwinChallenge", options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: "Team AMRIT_VIT · Shashi Bhushan · VIT Vellore · Thank you", options: { color: ICE } },
  ], { x: 0.7, y: 6.0, w: 11.9, h: 0.8, fontFace: BODY, fontSize: 15, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("To sum up: this twin lets a doctor triage CGM-monitored patients by two-hour risk, see alerts about 80 minutes ahead with reasons in clinical language, and try meal, walk and medicine changes on the patient's own physiological model, with Indian foods and safety checks built in, all developed without touching a real patient record. The next steps are validation on an Indian CGM cohort, clinical review of the rules, more drug classes, patient-side nudges in regional languages, and a silent-mode prospective study. The code, data card, reports and this deck are in the public repository. Thank you.");
  footer(s, true);
}

pres.writeFile({ fileName: OUT }).then((f) => console.log(`wrote ${path.relative(ROOT, f)}`));
