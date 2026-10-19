"""
ptf_engine.plots -- Plotly figures (interactive, responsive on phones/tablets).

Conventions kept from the original app:
  SWRC  : x = theta (0-0.7), y = pF (-2..7); band drawn horizontally in theta
  K     : x = theta (0-0.7), y = log10 K (-10..7)
  FC green marker, PWP red marker, PAW shaded light blue.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

THETA_LIMITS = (0.0, 0.7)
PF_LIMITS = (-2.0, 7.0)
LOGK_LIMITS = (-10.0, 7.0)

INK = "#334155"
GRID = "rgba(100,116,139,0.18)"
BLUE = "#1f6fb2"
BLUE_BAND = "rgba(31,111,178,0.22)"
PAW_FILL = "rgba(125,196,240,0.55)"
RED = "#b4432f"
RED_BAND = "rgba(180,67,47,0.20)"
FC_GREEN = "#1e8a4c"
PWP_RED = "#c62828"

# single-hue sequential ramps (light -> dark): magnitude only
SEQ_WATER = [[0, "#eef5fb"], [0.25, "#bcd7ee"], [0.5, "#79acd9"],
             [0.75, "#3b7cbf"], [1, "#0d3f7a"]]
SEQ_KSAT = [[0, "#fdf1e6"], [0.25, "#f8cfa7"], [0.5, "#ee9a5c"],
            [0.75, "#cf6326"], [1, "#7f3005"]]
SEQ_SD = [[0, "#f3f0f9"], [0.5, "#a99bd0"], [1, "#4a3593"]]

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
          "toImageButtonOptions": {"format": "png", "scale": 3}}


def _base(fig, title, height=430):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=14, color=INK)),
        height=height, margin=dict(l=55, r=15, t=45, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=INK), hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, font=dict(size=11)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, showline=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, showline=True)
    return fig


def _hband(fig, y, lo, hi, color, name):
    """Horizontal (theta-direction) band as one closed polygon."""
    lo = np.clip(lo, 0, 1)
    hi = np.clip(hi, 0, 1)
    fig.add_trace(go.Scatter(x=np.concatenate([lo, hi[::-1]]),
                             y=np.concatenate([y, y[::-1]]),
                             fill="toself", fillcolor=color, line=dict(width=0),
                             hoverinfo="skip", name=name, showlegend=True))


def swrc_figure(pf, mean, lo, hi, band_label, title, fc=None, pwp=None):
    """fc / pwp: (theta, pF) tuples or None."""
    fig = go.Figure()
    if fc and pwp and np.isfinite([*fc, *pwp]).all() and pwp[1] > fc[1]:
        seg = (pf >= fc[1]) & (pf <= pwp[1])
        y = np.concatenate([[fc[1]], pf[seg], [pwp[1]]])
        x = np.concatenate([[fc[0]], mean[seg], [pwp[0]]])
        x = np.maximum(x, pwp[0])
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, np.full_like(x, pwp[0])[::-1]]),
            y=np.concatenate([y, y[::-1]]), fill="toself", fillcolor=PAW_FILL,
            line=dict(width=0), hoverinfo="skip",
            name=f"PAW = {fc[0] - pwp[0]:.3f}"))
    _hband(fig, pf, lo, hi, BLUE_BAND, band_label)
    fig.add_trace(go.Scatter(x=mean, y=pf, mode="lines", line=dict(color=BLUE, width=2.2),
                             name="Predicted SWRC",
                             hovertemplate="θ = %{x:.3f}<br>pF = %{y:.2f}<extra></extra>"))
    for pt, color, lab in ((fc, FC_GREEN, "FC"), (pwp, PWP_RED, "PWP")):
        if pt:
            fig.add_trace(go.Scatter(
                x=[pt[0]], y=[pt[1]], mode="markers",
                marker=dict(size=11, color=color, line=dict(color="white", width=2)),
                name=f"{lab} = {pt[0]:.3f}",
                hovertemplate=f"{lab}<br>θ = %{{x:.3f}}<br>pF = %{{y:.2f}}<extra></extra>"))
    _base(fig, title)
    fig.update_xaxes(title="θ (cm³ cm⁻³)", range=THETA_LIMITS, dtick=0.1)
    fig.update_yaxes(title="pF = log₁₀(|h| cm)", range=PF_LIMITS, dtick=1)
    return fig


def k_figure(th_mean, th_lo, th_hi, lk_mean, band_label, title):
    fig = go.Figure()
    _hband(fig, lk_mean, th_lo, th_hi, RED_BAND, band_label)
    fig.add_trace(go.Scatter(x=th_mean, y=lk_mean, mode="lines",
                             line=dict(color=RED, width=2.2), name="Predicted log₁₀K",
                             hovertemplate="θ = %{x:.3f}<br>log₁₀K = %{y:.2f}<extra></extra>"))
    _base(fig, title)
    fig.update_xaxes(title="θ (cm³ cm⁻³)", range=THETA_LIMITS, dtick=0.1)
    fig.update_yaxes(title="log₁₀K (cm day⁻¹)", range=LOGK_LIMITS, dtick=2)
    return fig


def _display_grid(arr, max_px):
    """Block-average a map down to <= max_px on its long side for the browser.
    Exports always use the full-resolution array."""
    h, w = arr.shape
    f = int(np.ceil(max(h, w) / max_px))
    if f <= 1:
        return arr, 1
    H, W = (h // f) * f, (w // f) * f
    a = arr[:H, :W].reshape(H // f, f, W // f, f)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(a).sum(axis=(1, 3))
        s = np.nansum(a, axis=(1, 3))
    out = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32), f


def map_figure(arr, title, unit, colorscale, zmin=None, zmax=None,
               pixel_m=100.0, max_px=420, fmt=".3f"):
    disp, f = _display_grid(arr, max_px)
    h, w = disp.shape
    km = pixel_m * f / 1000.0
    x = np.arange(w) * km
    y = np.arange(h) * km
    fig = go.Figure(go.Heatmap(
        z=disp, x=x, y=y, colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(title=dict(text=unit, side="right"), thickness=12, len=0.9),
        hovertemplate=f"%{{z:{fmt}}} {unit}<extra></extra>", hoverongaps=False))
    _base(fig, title, height=400)
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x",
                     scaleratio=1, constrain="domain")
    if f > 1:
        fig.add_annotation(text=f"display {int(pixel_m*f)} m · download is 100 m",
                           xref="paper", yref="paper", x=0, y=-0.02, showarrow=False,
                           font=dict(size=10, color="#64748b"), xanchor="left")
    return fig
