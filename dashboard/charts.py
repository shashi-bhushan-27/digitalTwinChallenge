"""Plotly figures for the dashboard (restrained palette: one accent, red/amber/green only for risk)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACCENT = "#2f6fb0"
BAND = "rgba(47,111,176,0.18)"
RANGE = "rgba(46,139,87,0.09)"
RISK = {"red": "#c0392b", "amber": "#c77c0e", "green": "#1e8449"}
GREY = "#6b7280"
STAGE_Y = {0: 3, 3: 2, 1: 1, 2: 0}          # awake, REM, light, deep (top to bottom)
STAGE_LABEL = {3: "awake", 2: "REM", 1: "light", 0: "deep"}


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(size=13, color="#1f2937"), hovermode="x unified", showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=12)))
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f3", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f3", zeroline=False)
    return fig


def hero(cgm: pd.DataFrame, forecast: pd.DataFrame, meals: pd.DataFrame, steps: pd.DataFrame, hr: pd.DataFrame, sleep: pd.DataFrame,
         hrv: pd.DataFrame, now: pd.Timestamp, future: pd.DataFrame | None = None, colour: str = "green") -> go.Figure:
    """Last 24 h of CGM + 2-h forecast with band, and the wearable strip on the same time axis."""
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.56, 0.11, 0.11, 0.11, 0.11])
    t0, t1 = now - pd.Timedelta(hours=24), now + pd.Timedelta(hours=2, minutes=10)
    fig.add_hrect(y0=70, y1=180, fillcolor=RANGE, line_width=0, row=1, col=1)
    fig.add_hline(y=180, line=dict(color=RISK["red"], width=1, dash="dash"), row=1, col=1)
    fig.add_hline(y=70, line=dict(color=RISK["red"], width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cgm["timestamp"], y=cgm["glucose_mgdl"], mode="lines", name="CGM", line=dict(color=ACCENT, width=2),
                             hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(forecast["timestamp"]) + list(forecast["timestamp"][::-1]), y=list(forecast["hi"]) + list(forecast["lo"][::-1]),
                             fill="toself", fillcolor=BAND, mode="lines", line=dict(width=0), name="80 % range", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["point"], mode="lines+markers", name="forecast",
                             line=dict(color=ACCENT, width=2, dash="dash"), marker=dict(size=5),
                             hovertemplate="%{y:.0f} mg/dL<extra>forecast</extra>"), row=1, col=1)
    if future is not None and len(future):
        fig.add_trace(go.Scatter(x=future["timestamp"], y=future["glucose_mgdl"], mode="lines", name="what actually happened (replay)",
                                 line=dict(color=GREY, width=1.5, dash="dot"), hovertemplate="%{y:.0f} mg/dL<extra>actual</extra>"), row=1, col=1)
    if len(meals):
        fig.add_trace(go.Scatter(x=meals["timestamp"], y=[52] * len(meals), mode="markers+text", name="logged meal",
                                 marker=dict(symbol="triangle-up", size=12, color="#8a5a2b"),
                                 text=[f"{c:.0f} g" for c in meals["carbs_g_est"]], textposition="top center", textfont=dict(size=10, color="#8a5a2b"),
                                 customdata=meals["plate_name"], hovertemplate="%{customdata}<br>~%{text} carbs<extra>logged meal</extra>"), row=1, col=1)
    fig.add_vrect(x0=now, x1=now + pd.Timedelta(hours=2), fillcolor="rgba(47,111,176,0.05)", line_width=0, row="all", col=1)
    fig.add_annotation(x=now + pd.Timedelta(hours=1), y=0.02, yref="y domain", text="next 2 h", showarrow=False,
                       font=dict(size=11, color=ACCENT), row=1, col=1)
    fig.add_vline(x=now, line=dict(color="#111827", width=1.5))
    fig.add_annotation(x=now, y=1, yref="y domain", text="now", showarrow=False, xanchor="left", yanchor="top", font=dict(size=12), row=1, col=1)
    fig.add_trace(go.Bar(x=steps["timestamp"], y=steps["steps"], name="steps", marker_color="#5b8c5a", showlegend=False,
                         hovertemplate="%{y} steps / 5 min<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hr["timestamp"], y=hr["heart_rate_bpm"], mode="lines", name="heart rate", line=dict(color="#9b4d6b", width=1),
                             showlegend=False, hovertemplate="%{y:.0f} bpm<extra></extra>"), row=3, col=1)
    if len(sleep):
        y = sleep["stage"].map(STAGE_Y)
        fig.add_trace(go.Scatter(x=sleep["timestamp"], y=y, mode="markers", marker=dict(symbol="square", size=6, color=y, colorscale="Purples",
                                 cmin=-1, cmax=3), name="sleep", showlegend=False, customdata=y.map(STAGE_LABEL),
                                 hovertemplate="%{customdata}<extra>sleep</extra>"), row=4, col=1)
    if len(hrv):
        fig.add_trace(go.Scatter(x=hrv["timestamp"], y=hrv["rmssd_ms"], mode="markers", marker=dict(size=4, color="#6d5ba6"), name="HRV",
                                 showlegend=False, hovertemplate="RMSSD %{y:.0f} ms<extra></extra>"), row=5, col=1)
    _base_layout(fig, 640)
    fig.update_xaxes(range=[t0, t1])
    fig.update_yaxes(title_text="glucose mg/dL", range=[40, max(320, float(max(cgm["glucose_mgdl"].max() if len(cgm) else 0, forecast["hi"].max())) + 20)], row=1, col=1)
    fig.update_yaxes(title_text="steps", row=2, col=1, title_font=dict(size=11))
    fig.update_yaxes(title_text="HR", row=3, col=1, title_font=dict(size=11))
    fig.update_yaxes(title_text="sleep", row=4, col=1, tickvals=[0, 1, 2, 3], ticktext=["deep", "light", "REM", "awake"], range=[-0.5, 3.5],
                     tickfont=dict(size=9), title_font=dict(size=11))
    fig.update_yaxes(title_text="HRV", row=5, col=1, title_font=dict(size=11))
    return fig


def whatif(base, alt, now: pd.Timestamp, alt_label: str, base_label: str = "current plan") -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor=RANGE, line_width=0)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(192,57,43,0.08)", line_width=0)
    fig.add_hline(y=180, line=dict(color=RISK["red"], width=1, dash="dash"))
    fig.add_hline(y=70, line=dict(color=RISK["red"], width=1, dash="dot"), annotation_text="hypoglycaemia zone", annotation_position="bottom left")
    for o, lab, col, dash in ((base, base_label, GREY, "solid"), (alt, alt_label, ACCENT, "solid")):
        if o is None:
            continue
        h = (o.trajectory["timestamp"] - now).dt.total_seconds() / 3600
        fig.add_trace(go.Scatter(x=h, y=o.trajectory["glucose_mgdl"], mode="lines", name=f"{lab} (peak {o.peak:.0f}, low {o.trough:.0f})",
                                 line=dict(color=col, width=3, dash=dash), hovertemplate="%{y:.0f} mg/dL<extra></extra>"))
    _base_layout(fig, 380)
    fig.update_xaxes(title_text="hours from now")
    ymax = max([o.peak for o in (base, alt) if o is not None] + [200]) + 20
    fig.update_yaxes(title_text="glucose mg/dL (twin)", range=[40, ymax])
    return fig


def sparkline(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=-hist["months_ago"], y=hist["hba1c_pct"], mode="lines+markers", line=dict(color=ACCENT, width=2),
                               marker=dict(size=5), hovertemplate="%{y:.1f} %<extra></extra>"))
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=True, tickfont=dict(size=9), gridcolor="#eef0f3")
    return fig
