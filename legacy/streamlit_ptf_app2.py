"""
Physics-Constrained Inverse-PTF Explorer  (v4 - 12-model edition)
=================================================================

Replaces streamlit_ptf_app1.py, which was wired to a single trained model.

WHAT IS NEW IN v4
-----------------
  * 12 models instead of 1.  The user picks
        INPUT SET   : SSC | SSCBD | SSCOC | SSCBDOC
        MODEL       : Classic VG | Enhanced VG (dry-corrected) | PDI
    and the matching trained deep ensemble is loaded on demand and cached.
  * Tab 1 (manual soil):  SWRC over pF -2 .. 7 (step 0.05),
                          K(h)  over pF -10 .. 7 (step 0.05),
                          Ksat printed as a single number.
  * Shaded uncertainty band over the predicted SWRC and K(theta):
    10-member deep-ensemble mean +/- 1 SD, used consistently for the
    curves, the FC/PWP/PAW metrics and the fitted-parameter table.
  * Sand/silt/clay are renormalised to sum to 100 before prediction.
  * Tab 2 (SOLUS soil data): the input set decides which SOLUS rasters are
    read, the model choice decides the physics, and the outputs are
    SWRC, K(h), FC, PWP, PAW and Ksat maps for a county / drawn AOI / point.
  * Only the input variables the chosen model was trained on are requested.
    Picking SSC never asks for bulk density or organic carbon.

RUN
---
    streamlit run streamlit_ptf_app2.py

Requires ptf_core.py next to this file, plus the folder produced by
build_model_bundle.py (or the raw Final_PTF folder).

Author: Nikhil Raj Deep
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import MultipleLocator

from ptf_core import (INPUT_SETS, MODEL_FAMILIES, FEATURE_LABELS,
                      bundle_key, ensemble_band, load_model, normalize_texture,
                      pf_grid, predict_mean_curves, predict_members, texture_class)

# ---- geospatial stack (Tab 2 only) ---------------------------------------
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.io import MemoryFile
    from rasterio import features as rio_features
    from rasterio.transform import rowcol
    from rasterio.warp import transform_bounds, transform_geom
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except Exception:
    HAS_GEOPANDAS = False

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
    from shapely.geometry import (shape as shp_shape, box as shp_box,
                                  mapping as shp_mapping, Point as ShpPoint)
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False


# ===========================================================================
# CONFIGURATION -- edit these paths for your machine / server
# ===========================================================================

MODEL_ROOT = Path(r"C:\Users\Nikhil\Downloads\ptf_model_bundle")
# Fall back to the raw training folder if the bundle has not been built yet.
FALLBACK_ROOT = Path(r"C:\Users\Nikhil\Downloads\Final_PTF")

RASTER_DIR = Path(r"C:\Users\Nikhil\Downloads\Final_PTF\Soil Properties Maps")
COUNTY_DIR = Path(r"C:\Users\Nikhil\Downloads\PTF_SWRC"
                  r"\Florida_County_Boundaries_with_FDOT_Districts")

SOLUS_DEPTHS = [5, 15, 30, 60, 100]

# SOLUS rasters are stored as scaled integers.  These are the EXPECTED
# divisors, but the actual one is inferred from each raster's own values
# (see infer_scale) so a differently-scaled layer cannot silently feed the
# model a number 100x too large.
RASTER_SCALE = {"Sand": 1.0, "Silt": 1.0, "Clay": 1.0,
                "BD": 100.0,        # BD raster = BD x 100  -> g cm-3
                "SOC": 1000.0}      # SOC raster = SOC x 1000 -> %

# Physically plausible ranges in the units the models were trained on.
# Used to pick the divisor and to warn when an input layer looks wrong.
# Bounds are on the MEDIAN of the layer, so they exclude absurd scalings
# without excluding a genuinely low-carbon or low-clay landscape.
PLAUSIBLE = {"Sand": (0.5, 100.0), "Silt": (0.05, 100.0), "Clay": (0.05, 100.0),
             "BD": (0.2, 2.4), "SOC": (0.05, 60.0)}


def infer_scale(var: str, arr: np.ndarray) -> float:
    """Choose the divisor that puts this raster inside its plausible range.

    SOLUS stores BD as g cm-3 x 100 and SOC as % x 1000, but a re-exported
    layer may already be in real units.  Feeding BD = 135 instead of 1.35 to
    a MinMaxScaler fitted on 0.37-1.9 pushes every sigmoid head into
    saturation and produces a flat, identical curve for every pixel -- a
    silent wrong answer.  Picking the divisor from the data prevents that.
    """
    lo, hi = PLAUSIBLE[var]
    med = float(np.nanmedian(arr))
    if not np.isfinite(med):
        return RASTER_SCALE[var]
    expected = RASTER_SCALE[var]
    # Trust the documented SOLUS divisor whenever it gives a sensible answer;
    # only go looking if it does not.
    if lo <= med / expected <= hi:
        return expected
    for div in (1.0, 10.0, 100.0, 1000.0):
        if lo <= med / div <= hi:
            return div
    return expected
RASTER_ALIASES = {"Sand": ["Sand"], "Silt": ["Silt"], "Clay": ["Clay"],
                  "BD": ["BD", "BulkDensity"], "SOC": ["SOC", "OC", "Corg"]}

# Which SOLUS variable supplies which model feature.
FEATURE_TO_RASTER = {"Sand_pct": "Sand", "Silt_pct": "Silt", "Clay_pct": "Clay",
                     "BD_g_cm3": "BD", "OC_pct": "SOC"}

# Prediction grid.  Both the SWRC and the K(h) curve are traced along the same
# pF grid; pF is the curve parameter, theta and log10K are what get plotted.
SWRC_PF = (-2.0, 7.0, 0.05)
KH_PF = (-2.0, 7.0, 0.05)

# Fixed axis limits (theta on x for both panels).
THETA_LIMITS = (0.0, 0.7)
PF_LIMITS = (-2.0, 7.0)
LOGK_LIMITS = (-10.0, 7.0)

st.set_page_config(page_title="Inverse-PTF Explorer (12 models)",
                   page_icon="water_drop", layout="wide")


# ===========================================================================
# Loading (cached)
# ===========================================================================

def active_root() -> Path:
    return MODEL_ROOT if (MODEL_ROOT / "manifest.json").exists() \
        or any(MODEL_ROOT.glob("*/ensemble_models")) else FALLBACK_ROOT


@st.cache_resource(show_spinner="Loading trained ensemble ...")
def get_model(family: str, input_set: str):
    return load_model(active_root(), family, input_set)


@st.cache_data
def get_manifest():
    p = active_root() / "manifest.json"
    return json.load(open(p)) if p.exists() else None


# ===========================================================================
# Uncertainty helper
# ===========================================================================

Z_SCORE = 1.0
BAND_LABEL = "Deep ensemble mean +/- 1 SD (10 members)"


def band(stack):
    """stack: (M, ...).  Deep-ensemble mean +/- 1 SD across members.

    The SAME interval is used everywhere in the app -- curves, FC/PWP/PAW
    metrics and the fitted-parameter table -- so every number on screen is on
    one scale.
    """
    m, lo, hi, sd = ensemble_band(stack, z=Z_SCORE)
    return m, lo, hi, sd


# ===========================================================================
# Plots
# ===========================================================================

def _theta_axis(ax):
    ax.set_xlabel(r"$\theta$ (cm$^3$ cm$^{-3}$)")
    ax.set_xlim(*THETA_LIMITS)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))


def plot_swrc(pf, th_mean, th_lo, th_hi, label, title, marks=None, paw=None):
    """SWRC with theta on x and pF on y.

    The uncertainty band is horizontal: at every pF level the curve may sit
    anywhere between th_lo and th_hi, so fill_betweenx is the honest
    orientation.

    paw = (pf_fc, pf_pwp, theta_fc, theta_pwp) shades the plant-available
    water: the water held between field capacity and the wilting point.  With
    theta on x it is the region between theta_PWP and the curve, over the pF
    band from FC down to PWP -- widest at FC, closing to zero at PWP.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    # PAW first so the uncertainty band and the curve stay readable on top.
    if paw is not None:
        pf_fc, pf_pwp, th_fc, th_pwp = paw
        if np.isfinite([pf_fc, pf_pwp, th_fc, th_pwp]).all() and pf_pwp > pf_fc:
            seg = (pf >= pf_fc) & (pf <= pf_pwp)
            pf_seg = np.concatenate([[pf_fc], pf[seg], [pf_pwp]])
            th_seg = np.concatenate([[th_fc], th_mean[seg], [th_pwp]])
            ax.fill_betweenx(pf_seg, th_pwp, np.maximum(th_seg, th_pwp),
                             color="skyblue", alpha=0.55, zorder=1,
                             label=f"PAW = {th_fc - th_pwp:.3f}")

    ax.fill_betweenx(pf, np.clip(th_lo, 0, 1), np.clip(th_hi, 0, 1),
                     color="tab:blue", alpha=0.25, zorder=2, label=label)
    ax.plot(th_mean, pf, color="tab:blue", lw=2, zorder=3,
            label="Predicted SWRC")
    if marks:
        for theta_v, pf_v, c, txt in marks:
            ax.scatter([theta_v], [pf_v], s=80, color=c, zorder=5, label=txt)
    _theta_axis(ax)
    ax.set_ylabel("pF = log10(|h| in cm)")
    ax.set_ylim(*PF_LIMITS)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_kh(th_mean, th_lo, th_hi, lk_mean, label, title):
    """K(theta) with theta on x and log10 K on y.

    Both axes are traced along the same pF grid, so the curve is
    (theta(pF), log10K(pF)).  The band is drawn horizontally in theta at each
    log10K level, matching the SWRC panel.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.fill_betweenx(lk_mean, np.clip(th_lo, 0, 1), np.clip(th_hi, 0, 1),
                     color="tab:red", alpha=0.22, label=label)
    ax.plot(th_mean, lk_mean, color="tab:red", lw=2,
            label=r"Predicted $\log_{10}K$")
    _theta_axis(ax)
    ax.set_ylabel(r"$\log_{10} K$ (cm day$^{-1}$)")
    ax.set_ylim(*LOGK_LIMITS)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ===========================================================================
# Shared model/input-set selector
# ===========================================================================

def model_selector(key_prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        iset = st.selectbox(
            "1. Input set", list(INPUT_SETS),
            format_func=lambda k: f"{k}  -  {INPUT_SETS[k]}",
            key=f"{key_prefix}_iset")
    with c2:
        fam = st.selectbox(
            "2. Model", list(MODEL_FAMILIES),
            format_func=lambda k: MODEL_FAMILIES[k]["label"],
            index=2, key=f"{key_prefix}_fam")
    return fam, iset


# ===========================================================================
# TAB 1 -- manual soil properties
# ===========================================================================

def manual_tab():
    st.subheader("Own soil data - enter measured properties")
    fam, iset = model_selector("m")

    try:
        lm = get_model(fam, iset)
    except Exception as e:
        st.error(f"Could not load {bundle_key(fam, iset)}:\n\n{e}")
        return

    st.caption(f"Active model: **{bundle_key(fam, iset)}** | "
               f"{len(lm.models)}-member ensemble | "
               f"features = {', '.join(lm.features)} | "
               f"retention = {lm.config['physics']['retention']} | "
               f"K(h) = {lm.config['physics']['conductivity']}")

    left, right = st.columns([1, 1.6])
    values = {}
    with left:
        st.markdown("#### Soil inputs")
        for feat in lm.features:
            label, lo, hi, default, step = FEATURE_LABELS[feat]
            if feat == "Depth_harm_cm":
                values[feat] = float(st.selectbox(
                    "Depth (cm)", SOLUS_DEPTHS, index=0, key="m_depth"))
            else:
                values[feat] = st.number_input(label, lo, hi, default, step,
                                               key=f"m_{feat}")
        # The models were trained on fractions summing to 100, so renormalise
        # rather than extrapolating off the training manifold.
        total = sum(values[k] for k in ("Sand_pct", "Silt_pct", "Clay_pct"))
        if total <= 0:
            st.error("Sand + Silt + Clay must be greater than zero.")
            return
        if abs(total - 100) > 0.01:
            sa, si, cl = normalize_texture(values["Sand_pct"], values["Silt_pct"],
                                           values["Clay_pct"])
            values["Sand_pct"] = float(sa)
            values["Silt_pct"] = float(si)
            values["Clay_pct"] = float(cl)
            st.info(
                f"Sand + Silt + Clay = {total:.1f}%, renormalised to 100%: "
                f"sand {values['Sand_pct']:.1f}, silt {values['Silt_pct']:.1f}, "
                f"clay {values['Clay_pct']:.1f}%. Ratios are preserved.")

        st.markdown("#### Retention points")
        fc_choice = st.radio("Field-capacity convention",
                             ["pF 2.0 (sandy soils)", "pF 2.54 (classic)"],
                             index=0, key="m_fc")
        pf_fc = 2.0 if fc_choice.startswith("pF 2.0") else 2.54
        pf_pwp = st.number_input("Wilting point pF", 3.0, 5.0, 4.2, 0.01,
                                 key="m_pwp")

    # ---- prediction -------------------------------------------------------
    pf_s = pf_grid(*SWRC_PF)
    pf_k = pf_grid(*KH_PF)
    extra = np.array([pf_fc, pf_pwp])
    # theta is needed on the SWRC grid, at FC/PWP, and on the K(h) grid
    pf_s_all = np.concatenate([pf_s, extra, pf_k])

    x = np.array([[values[c] for c in lm.features]], dtype=np.float64)
    theta_m, logk_m, par_m = predict_members(lm, x, pf_s_all, pf_k)

    th = theta_m[:, 0, :len(pf_s)]                  # (M, n_pf)   SWRC grid
    th_pts = theta_m[:, 0, len(pf_s):len(pf_s) + 2]  # (M, 2)     FC / PWP
    thk = theta_m[:, 0, len(pf_s) + 2:]             # (M, n_pf_k) K(h) grid
    lk = logk_m[:, 0, :]                            # (M, n_pf_k)

    th_mean, th_lo, th_hi, _ = band(th)
    thk_mean, thk_lo, thk_hi, _ = band(thk)
    lk_mean, lk_lo, lk_hi, _ = band(lk)

    fc_mean, fc_lo, fc_hi, _ = band(th_pts[:, 0:1])
    pwp_mean, pwp_lo, pwp_hi, _ = band(th_pts[:, 1:2])
    paw_stack = th_pts[:, 0:1] - th_pts[:, 1:2]
    paw_mean, paw_lo, paw_hi, _ = band(paw_stack)

    logks = par_m["log10_ksat"][:, 0]
    ks_mean, ks_lo, ks_hi, _ = band(logks.reshape(-1, 1))

    with right:
        st.markdown("#### Saturated hydraulic conductivity")
        k1, k2 = st.columns([1, 1.4])
        k1.metric("Ksat (cm day-1)", f"{10 ** float(ks_mean[0]):,.2f}")
        k2.caption(f"95% band: {10 ** float(ks_lo[0]):,.2f} to "
                   f"{10 ** float(ks_hi[0]):,.2f} cm day-1  "
                   f"({10 ** float(ks_mean[0]) / 24:,.3f} cm h-1)")

        st.markdown("#### Water-retention points")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"FC (pF {pf_fc})", f"{float(fc_mean[0]):.3f}",
                  help=f"{float(fc_lo[0]):.3f} - {float(fc_hi[0]):.3f}")
        c2.metric(f"PWP (pF {pf_pwp})", f"{float(pwp_mean[0]):.3f}",
                  help=f"{float(pwp_lo[0]):.3f} - {float(pwp_hi[0]):.3f}")
        c3.metric("PAW = FC - PWP", f"{float(paw_mean[0]):.3f}",
                  help=f"{float(paw_lo[0]):.3f} - {float(paw_hi[0]):.3f}")
        st.caption(f"Texture class: **{texture_class(values['Sand_pct'], values['Silt_pct'], values['Clay_pct'])}**"
                   f"  |  uncertainty = {BAND_LABEL}")

        st.markdown("#### Fitted hydraulic parameters")
        st.caption("Same interval as the curves: ensemble mean +/- 1 SD "
                   "across the 10 members.")
        rows = []
        for name in lm.param_names:
            key = "log10_ksat" if name == "ksat" else name
            v = par_m[key][:, 0].astype(float)
            m, lo, hi, sd = band(v.reshape(-1, 1))
            m, lo, hi = float(m[0]), float(lo[0]), float(hi[0])
            if name == "ksat":
                # Ksat is predicted in log10 space, so the interval is
                # back-transformed: the mean is a geometric mean.
                rows.append(["Ksat (cm d-1)", 10 ** m,
                             f"{10 ** lo:.4g} - {10 ** hi:.4g}"])
            else:
                rows.append([name, m, f"{lo:.4g} - {hi:.4g}"])
        st.table(pd.DataFrame(
            rows, columns=["Parameter", "Ensemble mean", "+/- 1 SD"])
            .style.format({"Ensemble mean": "{:.4g}"}))

    # ---- figures ----------------------------------------------------------
    g1, g2 = st.columns(2)
    with g1:
        fig = plot_swrc(
            pf_s, th_mean, th_lo, th_hi, BAND_LABEL,
            f"SWRC - {MODEL_FAMILIES[fam]['label']} / {iset}",
            marks=[(float(fc_mean[0]), pf_fc, "green", f"FC = {float(fc_mean[0]):.3f}"),
                   (float(pwp_mean[0]), pf_pwp, "red", f"PWP = {float(pwp_mean[0]):.3f}")],
            paw=(pf_fc, pf_pwp, float(fc_mean[0]), float(pwp_mean[0])))
        st.pyplot(fig)
    with g2:
        st.pyplot(plot_kh(thk_mean, thk_lo, thk_hi, lk_mean, BAND_LABEL,
                          f"K(theta) - {MODEL_FAMILIES[fam]['label']} / {iset}"))

    # ---- downloads --------------------------------------------------------
    swrc_df = pd.DataFrame({"pF": pf_s, "h_cm": 10.0 ** pf_s,
                            "theta_mean": th_mean, "theta_lo": th_lo,
                            "theta_hi": th_hi})
    kh_df = pd.DataFrame({"pF": pf_k, "h_cm": 10.0 ** pf_k,
                          "theta_mean": thk_mean, "theta_lo": thk_lo,
                          "theta_hi": thk_hi,
                          "log10K_mean": lk_mean, "log10K_lo": lk_lo,
                          "log10K_hi": lk_hi,
                          "K_cm_d_mean": 10.0 ** lk_mean})
    d1, d2 = st.columns(2)
    d1.download_button("Download SWRC (CSV)", swrc_df.to_csv(index=False).encode(),
                       file_name=f"SWRC_{bundle_key(fam, iset)}.csv", mime="text/csv")
    d2.download_button("Download K(h) (CSV)", kh_df.to_csv(index=False).encode(),
                       file_name=f"Kh_{bundle_key(fam, iset)}.csv", mime="text/csv")


# ===========================================================================
# SOLUS raster handling
# ===========================================================================

def find_raster(variable: str, depth: int):
    for alias in RASTER_ALIASES[variable]:
        for pat in (f"{alias}{depth}", f"{alias}0{depth}", f"{alias}_{depth}"):
            for ext in (".tif", ".tiff"):
                hits = [p for p in RASTER_DIR.glob(f"*{ext}")
                        if p.stem.lower() == pat.lower()]
                if hits:
                    return hits[0]
    return None


@st.cache_data(show_spinner="Reading SOLUS rasters for this depth ...")
def load_depth_stack(depth: int, variables: tuple):
    """Read + co-register only the SOLUS layers the chosen input set needs.

    Returns (stack, ref, scales); stack values are already in the real units
    the models were trained on."""
    paths = {v: find_raster(v, depth) for v in variables}
    missing = [v for v, p in paths.items() if p is None]
    if missing:
        raise FileNotFoundError(
            f"Missing SOLUS rasters for depth {depth} cm in {RASTER_DIR}: "
            f"{missing}. Expected e.g. Sand{depth}.tif, BD{depth}.tif.")

    ref_var = "Clay" if "Clay" in variables else variables[0]
    with rasterio.open(paths[ref_var]) as ref_src:
        ref = {"crs": ref_src.crs.to_string(),
               "transform": tuple(ref_src.transform)[:6],
               "width": ref_src.width, "height": ref_src.height,
               "bounds": tuple(ref_src.bounds),
               "profile": ref_src.profile.copy()}

    def read_on_ref(path):
        with rasterio.open(path) as src:
            with WarpedVRT(src, crs=ref["crs"],
                           transform=rasterio.Affine(*ref["transform"]),
                           width=ref["width"], height=ref["height"],
                           resampling=Resampling.bilinear) as vrt:
                arr = np.asarray(vrt.read(1), dtype=np.float32)
                nod = vrt.nodata
        if nod is not None:
            arr = np.where(arr == nod, np.nan, arr)
        return arr

    stack, scales = {}, {}
    for v in variables:
        raw = read_on_ref(paths[v])
        div = infer_scale(v, raw)
        scales[v] = {"file": paths[v].name, "divisor": div,
                     "expected_divisor": RASTER_SCALE[v]}
        stack[v] = raw / div

    # SOLUS sand/silt/clay are not a closed composition (they sum to ~81% over
    # north Florida and sand alone can exceed 100), but the models were
    # trained on fractions summing to 100.  Renormalise each pixel.
    if {"Sand", "Silt", "Clay"} <= set(stack):
        before = stack["Sand"] + stack["Silt"] + stack["Clay"]
        sa, si, cl = normalize_texture(stack["Sand"], stack["Silt"], stack["Clay"])
        stack["Sand"], stack["Silt"], stack["Clay"] = (
            sa.astype(np.float32), si.astype(np.float32), cl.astype(np.float32))
        fin = before[np.isfinite(before)]
        scales["_texture_norm"] = {
            "sum_before_min": float(fin.min()) if fin.size else float("nan"),
            "sum_before_median": float(np.median(fin)) if fin.size else float("nan"),
            "sum_before_max": float(fin.max()) if fin.size else float("nan")}
    return stack, ref, scales


@st.cache_data
def load_counties():
    if not (HAS_GEOPANDAS and COUNTY_DIR.exists()):
        return None, None
    shp = list(COUNTY_DIR.glob("*.shp"))
    if not shp:
        return None, None
    counties = gpd.read_file(shp[0])
    cand = [c for c in counties.columns
            if "name" in c.lower() or "county" in c.lower()]
    return counties.to_json(), (cand[0] if cand else counties.columns[0])


def stack_valid_mask(lm, stack):
    """Flat boolean mask of pixels where every required layer is finite.

    Computed layer by layer so the full (pixels x features) matrix is never
    built for the whole grid -- that is 1.3 GB on a SOLUS grid.
    """
    mask = None
    for feat in lm.features:
        if feat == "Depth_harm_cm":
            continue
        ok = np.isfinite(stack[FEATURE_TO_RASTER[feat]]).ravel()
        mask = ok if mask is None else (mask & ok)
    return mask


def features_from_stack(lm, stack, depth, index):
    """Assemble the (N, n_features) matrix for the selected pixels only,
    in the exact column order the model was trained with."""
    n = len(index)
    out = np.empty((n, len(lm.features)), np.float32)
    for j, feat in enumerate(lm.features):
        if feat == "Depth_harm_cm":
            out[:, j] = float(depth)
        else:
            out[:, j] = stack[FEATURE_TO_RASTER[feat]].ravel()[index]
    return out


def show_input_diagnostics(lm, stack, scales, depth):
    """Show exactly what reaches the network, and shout if it is out of range.

    The models are MinMaxScaler-fronted: an input well outside the training
    range saturates every sigmoid head and returns the same flat curve for
    every pixel.  That failure is invisible on a map (it just looks uniform),
    so the numbers that produce it are put on screen.
    """
    sc = lm.scaler
    lo_tr = np.asarray(getattr(sc, "data_min_", []), float)
    hi_tr = np.asarray(getattr(sc, "data_max_", []), float)

    rows, bad = [], []
    for i, feat in enumerate(lm.features):
        t_lo = float(lo_tr[i]) if i < len(lo_tr) else np.nan
        t_hi = float(hi_tr[i]) if i < len(hi_tr) else np.nan

        if feat == "Depth_harm_cm":
            vmin = vmax = float(depth)
            frac_out = 0.0 if (t_lo <= depth <= t_hi) else 1.0
            src = "selected depth"
        else:
            var = FEATURE_TO_RASTER[feat]
            a = stack[var]
            fin = a[np.isfinite(a)]
            vmin, vmax = float(fin.min()), float(fin.max())
            frac_out = float(((fin < t_lo) | (fin > t_hi)).mean()) \
                if np.isfinite(t_lo) else 0.0
            info = scales[var]
            src = f"{info['file']} / {info['divisor']:g}"
            if info["divisor"] != info["expected_divisor"]:
                src += f" (expected /{info['expected_divisor']:g})"

        # A few edge pixels outside the training hull are normal; a layer that
        # is mostly outside it means something is wrong with the units.
        out = frac_out > 0.01
        if out:
            bad.append(feat)
        rows.append({"Feature": feat, "Source": src,
                     "Raster min": round(vmin, 4), "Raster max": round(vmax, 4),
                     "Training min": round(t_lo, 4),
                     "Training max": round(t_hi, 4),
                     "% outside training": round(100 * frac_out, 2),
                     "OK": "no" if out else "yes"})

    norm = scales.get("_texture_norm")

    with st.expander("Input layers reaching the model"
                     + ("  --  WARNING: out of training range" if bad else ""),
                     expanded=bool(bad)):
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        if norm is not None:
            st.caption(
                "Sand/silt/clay renormalised to sum to 100 per pixel. "
                f"Before normalising the sum ranged "
                f"{norm['sum_before_min']:.1f} to {norm['sum_before_max']:.1f}% "
                f"(median {norm['sum_before_median']:.1f}%). "
                "The min/max above are AFTER normalising, i.e. what the "
                "network actually receives.")
        if bad:
            st.error(
                "These layers fall outside the range the model was trained on: "
                + ", ".join(bad)
                + ". Predictions will be flat and identical across pixels. "
                  "Check the raster scaling above (BD should be ~0.2-2.4 "
                  "g cm-3 and OC ~0-60%, not 100x larger).")


# ===========================================================================
# TAB 2 -- SOLUS spatial explorer
# ===========================================================================

def spatial_tab():
    st.subheader("SOLUS soil data - spatial prediction")

    if not HAS_RASTERIO:
        st.error("This tab needs rasterio  (conda install -c conda-forge rasterio).")
        return
    if not RASTER_DIR.exists():
        st.error(f"SOLUS raster folder not found: {RASTER_DIR}")
        return

    fam, iset = model_selector("s")
    try:
        lm = get_model(fam, iset)
    except Exception as e:
        st.error(f"Could not load {bundle_key(fam, iset)}:\n\n{e}")
        return

    needed = tuple(FEATURE_TO_RASTER[f] for f in lm.features
                   if f != "Depth_harm_cm")
    st.caption(f"Active model: **{bundle_key(fam, iset)}** | "
               f"SOLUS layers required: {', '.join(needed)}")

    c0, c1, c2 = st.columns(3)
    with c0:
        depth = st.selectbox("SOLUS depth (cm)", SOLUS_DEPTHS, index=0, key="s_depth")
    with c1:
        fc_choice = st.radio("Field-capacity convention",
                             ["pF 2.0 (sandy soils)", "pF 2.54 (classic)"],
                             index=0, key="s_fc")
        pf_fc = 2.0 if fc_choice.startswith("pF 2.0") else 2.54
    with c2:
        pf_pwp = st.number_input("Wilting point pF", 3.0, 5.0, 4.2, 0.01, key="s_pwp")

    mode = st.radio("Where?",
                    ["County boundary", "Draw AOI (polygon/rectangle)",
                     "Draw a point on the map", "Enter latitude / longitude"],
                    index=0, horizontal=True, key="s_mode")

    try:
        stack, ref, scales = load_depth_stack(depth, needed)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    H, W = stack[needed[0]].shape
    st.caption(f"Grid: {W} x {H} pixels, CRS {ref['crs']}.")
    show_input_diagnostics(lm, stack, scales, depth)

    minlon, minlat, maxlon, maxlat = transform_bounds(ref["crs"], "EPSG:4326",
                                                      *ref["bounds"])
    footprint = shp_box(*ref["bounds"]) if HAS_FOLIUM else None

    # ------------------------------------------------------------------ county
    if mode == "County boundary":
        cjson, name_col = load_counties()
        if cjson is None:
            st.error(f"County shapefile not found in {COUNTY_DIR} "
                     "(or geopandas missing). Use AOI / lat-lon instead.")
            return
        counties = gpd.read_file(cjson)
        names = sorted(counties[name_col].dropna().astype(str).str.strip().unique())
        choice = st.selectbox("County", names, key="s_county")
        if not st.button("Run prediction for this county", type="primary"):
            st.info("Pick a county and press the button.")
            return
        cg = counties[counties[name_col].astype(str).str.strip() == choice].to_crs(ref["crs"])
        mask = rio_features.geometry_mask(
            [g.__geo_interface__ for g in cg.geometry], out_shape=(H, W),
            transform=rasterio.Affine(*ref["transform"]), invert=True)
        area_prediction(lm, stack, ref, depth, mask, pf_fc, pf_pwp, choice)
        return

    # ------------------------------------------------------------------ lat/lon
    if mode == "Enter latitude / longitude":
        cl, cr = st.columns(2)
        lat = cl.number_input("Latitude", value=float((minlat + maxlat) / 2),
                              format="%.5f", key="s_lat")
        lon = cr.number_input("Longitude", value=float((minlon + maxlon) / 2),
                              format="%.5f", key="s_lon")
        st.caption(f"Raster extent: lat {minlat:.3f}..{maxlat:.3f}, "
                   f"lon {minlon:.3f}..{maxlon:.3f}")
        if not st.button("Predict here", type="primary"):
            return
        if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
            st.error("That coordinate is outside the raster extent.")
            return
        p = shp_shape(transform_geom("EPSG:4326", ref["crs"],
                                     shp_mapping(ShpPoint(lon, lat))))
        point_prediction(lm, stack, ref, depth, p.x, p.y, pf_fc, pf_pwp,
                         f"{lat:.4f}N {abs(lon):.4f}W")
        return

    # ------------------------------------------------------------------ map
    if not HAS_FOLIUM:
        st.error("Interactive map needs folium + streamlit-folium + shapely.")
        return

    is_point = mode == "Draw a point on the map"
    fmap = folium.Map(location=[(minlat + maxlat) / 2, (minlon + maxlon) / 2],
                      zoom_start=7, tiles="OpenStreetMap")
    folium.Rectangle([[minlat, minlon], [maxlat, maxlon]], color="blue",
                     weight=2, fill=False, tooltip="Raster extent").add_to(fmap)
    fmap.fit_bounds([[minlat, minlon], [maxlat, maxlon]])
    Draw(export=False, draw_options={"polyline": False, "circle": False,
                                     "circlemarker": False, "marker": is_point,
                                     "rectangle": not is_point,
                                     "polygon": not is_point}).add_to(fmap)
    out = st_folium(fmap, width=820, height=520,
                    key="s_pointmap" if is_point else "s_drawmap")

    if is_point:
        ll = None
        for sh in reversed((out or {}).get("all_drawings") or []):
            if sh["geometry"]["type"] == "Point":
                ll = tuple(sh["geometry"]["coordinates"])
                break
        if ll is None and (out or {}).get("last_clicked"):
            ll = (out["last_clicked"]["lng"], out["last_clicked"]["lat"])
        if ll is None:
            st.info("Place a marker or click the map.")
            return
        p = shp_shape(transform_geom("EPSG:4326", ref["crs"],
                                     shp_mapping(ShpPoint(*ll))))
        if not footprint.contains(p):
            st.error("That point is outside the raster extent.")
            return
        point_prediction(lm, stack, ref, depth, p.x, p.y, pf_fc, pf_pwp,
                         f"{ll[1]:.4f}N {abs(ll[0]):.4f}W")
        return

    polys = [s for s in ((out or {}).get("all_drawings") or [])
             if s["geometry"]["type"] in ("Polygon", "MultiPolygon")]
    if not polys:
        st.info("Draw a polygon or rectangle to run the prediction.")
        return
    geom = shp_shape(transform_geom("EPSG:4326", ref["crs"],
                                    shp_mapping(shp_shape(polys[-1]["geometry"]))))
    if not geom.intersects(footprint):
        st.error("The AOI lies outside the raster extent.")
        return
    geom = geom.intersection(footprint)
    mask = rio_features.geometry_mask([shp_mapping(geom)], out_shape=(H, W),
                                      transform=rasterio.Affine(*ref["transform"]),
                                      invert=True)
    area_prediction(lm, stack, ref, depth, mask, pf_fc, pf_pwp, "drawn AOI")


# ---------------------------------------------------------------- one pixel
def point_prediction(lm, stack, ref, depth, x_native, y_native,
                     pf_fc, pf_pwp, label):
    row, col = rowcol(rasterio.Affine(*ref["transform"]), x_native, y_native)
    H, W = stack[list(stack)[0]].shape
    if not (0 <= row < H and 0 <= col < W):
        st.error("Point outside the raster grid.")
        return
    vals = {v: float(stack[v][row, col]) for v in stack}
    if not np.all(np.isfinite(list(vals.values()))):
        st.error("This pixel is NoData. Pick another location.")
        return

    feats = []
    for f in lm.features:
        feats.append(float(depth) if f == "Depth_harm_cm"
                     else vals[FEATURE_TO_RASTER[f]])
    x = np.array([feats], dtype=np.float64)

    pf_s = pf_grid(*SWRC_PF)
    pf_k = pf_grid(*KH_PF)
    pf_s_all = np.concatenate([pf_s, [pf_fc, pf_pwp], pf_k])
    th_m, lk_m, par_m = predict_members(lm, x, pf_s_all, pf_k)

    th = th_m[:, 0, :len(pf_s)]
    pts = th_m[:, 0, len(pf_s):len(pf_s) + 2]
    thk = th_m[:, 0, len(pf_s) + 2:]
    lk = lk_m[:, 0, :]

    th_mean, th_lo, th_hi, _ = band(th)
    thk_mean, thk_lo, thk_hi, _ = band(thk)
    lk_mean, lk_lo, lk_hi, _ = band(lk)
    fc = float(pts[:, 0].mean()); pwp = float(pts[:, 1].mean())
    ks = 10 ** float(par_m["log10_ksat"][:, 0].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"FC (pF {pf_fc})", f"{fc:.3f}")
    c2.metric(f"PWP (pF {pf_pwp})", f"{pwp:.3f}")
    c3.metric("PAW", f"{fc - pwp:.3f}")
    c4.metric("Ksat (cm d-1)", f"{ks:,.2f}")

    st.caption("SOLUS pixel: " + " | ".join(f"{k} = {v:.3g}" for k, v in vals.items())
               + f" | depth {depth} cm"
               + (f" | texture {texture_class(vals['Sand'], vals['Silt'], vals['Clay'])}"
                  if {"Sand", "Silt", "Clay"} <= set(vals) else ""))

    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(plot_swrc(pf_s, th_mean, th_lo, th_hi, BAND_LABEL,
                            f"SWRC - {label}",
                            marks=[(fc, pf_fc, "green", f"FC = {fc:.3f}"),
                                   (pwp, pf_pwp, "red", f"PWP = {pwp:.3f}")],
                            paw=(pf_fc, pf_pwp, fc, pwp)))
    with g2:
        st.pyplot(plot_kh(thk_mean, thk_lo, thk_hi, lk_mean, BAND_LABEL,
                          f"K(theta) - {label}"))

    st.download_button(
        "Download curves (CSV)",
        pd.DataFrame({"pF": pf_s, "theta_mean": th_mean,
                      "theta_lo": th_lo, "theta_hi": th_hi})
        .to_csv(index=False).encode(),
        file_name=f"SWRC_{bundle_key(lm.family, lm.input_set)}_{depth}cm.csv")


# ---------------------------------------------------------------- an area
def area_prediction(lm, stack, ref, depth, sel_mask, pf_fc, pf_pwp, area_label):
    H, W = stack[list(stack)[0]].shape
    valid = stack_valid_mask(lm, stack) & sel_mask.ravel()
    idx = np.flatnonzero(valid)
    n_valid = int(idx.size)
    if n_valid == 0:
        st.error("No valid pixels in the selection (all NoData or no overlap).")
        return
    feat_sel = features_from_stack(lm, stack, depth, idx)

    pf_s = pf_grid(*SWRC_PF)
    pf_k = pf_grid(*KH_PF)

    # ---- maps: only the two retention points + Ksat, over all valid pixels --
    with st.spinner(f"Predicting {n_valid:,} pixels with "
                    f"{len(lm.models)} ensemble members ..."):
        th_pts, _, par = predict_members(
            lm, feat_sel, np.array([pf_fc, pf_pwp]), np.array([0.0]))
        fc_v = th_pts[:, :, 0].mean(axis=0)
        pwp_v = th_pts[:, :, 1].mean(axis=0)
        paw_v = fc_v - pwp_v
        ks_v = 10 ** par["log10_ksat"].mean(axis=0)

    def rebuild(v):
        full = np.full(H * W, np.nan, np.float32)
        full[valid] = v
        return full.reshape(H, W)

    fc_map, pwp_map, paw_map = rebuild(fc_v), rebuild(pwp_v), rebuild(paw_v)
    ks_map = rebuild(ks_v)
    st.success(f"Predicted {n_valid:,} pixels in {area_label}.")

    rows_any = np.where(sel_mask.any(axis=1))[0]
    cols_any = np.where(sel_mask.any(axis=0))[0]
    r0, r1 = rows_any.min(), rows_any.max() + 1
    c0, c1 = cols_any.min(), cols_any.max() + 1
    pr, pc = max(1, (r1 - r0) // 20), max(1, (c1 - c0) // 20)
    r0, r1 = max(0, r0 - pr), min(H, r1 + pr)
    c0, c1 = max(0, c0 - pc), min(W, c1 + pc)
    cut = lambda m: m[r0:r1, c0:c1]

    # All four maps go in ONE 2x2 figure.  Two separate figures each get
    # stretched to the container width by Streamlit, so a 1x3 row and a lone
    # 1x1 map end up at wildly different panel sizes -- the Ksat map three
    # times the size of the others.  One figure fixes the relative sizes.
    theta_maps = [cut(fc_map), cut(pwp_map), cut(paw_map)]
    allv = np.concatenate([m[np.isfinite(m)] for m in theta_maps])
    vmin, vmax = np.percentile(allv, [2, 98])
    cmap = plt.cm.viridis.copy(); cmap.set_bad("white")
    cmap_k = plt.cm.magma.copy(); cmap_k.set_bad("white")

    # Every panel shows the same window, so they share one pixel aspect;
    # sizing the figure from it keeps the four boxes square-on to the data
    # instead of letterboxing the maps inside wide axes.
    nrow, ncol = theta_maps[0].shape
    panel_w = 4.6
    panel_h = float(np.clip(panel_w * nrow / max(ncol, 1), 2.8, 6.5))

    fig, axes = plt.subplots(2, 2, figsize=(2 * panel_w + 1.6, 2 * panel_h + 1.0),
                             constrained_layout=True)
    panels = [
        (axes[0, 0], theta_maps[0], f"Field capacity (pF {pf_fc})", cmap, vmin, vmax,
         r"$\theta$ (cm$^3$ cm$^{-3}$)"),
        (axes[0, 1], theta_maps[1], f"Wilting point (pF {pf_pwp})", cmap, vmin, vmax,
         r"$\theta$ (cm$^3$ cm$^{-3}$)"),
        (axes[1, 0], theta_maps[2], "Plant-available water", cmap, vmin, vmax,
         r"$\theta$ (cm$^3$ cm$^{-3}$)"),
        (axes[1, 1], np.log10(np.clip(cut(ks_map), 1e-6, None)),
         "log10 Ksat", cmap_k, None, None, r"$\log_{10}K_{sat}$ (cm day$^{-1}$)"),
    ]
    for ax, m, title, cm_, lo_, hi_, cb_label in panels:
        im = ax.imshow(m, cmap=cm_, vmin=lo_, vmax=hi_, aspect="equal")
        ax.set_title(f"{title} - {depth} cm", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        # fraction=0.046/pad=0.04 makes each colorbar track its own image
        # height, so all four panels stay the same size.
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(cb_label, fontsize=9)
    fig.suptitle(f"{MODEL_FAMILIES[lm.family]['label']} / {lm.input_set}"
                 f"  --  {area_label}", fontsize=12)
    st.pyplot(fig)
    st.caption("Field capacity, wilting point and plant-available water share "
               "one colour scale so they can be compared directly; Ksat has "
               "its own.")

    # ---- area-averaged curves, over EVERY valid pixel ----------------------
    # Accumulated in chunks rather than held in one array: the full
    # (members x pixels x pF) block for a county would be several GB.
    bar = st.progress(0.0, text=f"Area-averaged curves over all "
                               f"{n_valid:,} pixels ...")
    th_area, thk_area, lk_area = predict_mean_curves(
        lm, feat_sel, pf_s, pf_k,
        progress=lambda f: bar.progress(min(f, 1.0),
                                        text=f"Area-averaged curves over all "
                                             f"{n_valid:,} pixels ... {100*f:.0f}%"))
    bar.empty()

    th_m, th_lo, th_hi, _ = band(th_area)
    thk_m, thk_lo, thk_hi, _ = band(thk_area)
    lk_m, _, _, _ = band(lk_area)

    # FC/PWP read off the plotted area-mean curve, so the markers sit exactly
    # on the line rather than a hair off it.
    fc_a = float(np.interp(pf_fc, pf_s, th_m))
    pwp_a = float(np.interp(pf_pwp, pf_s, th_m))

    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(plot_swrc(pf_s, th_m, th_lo, th_hi, BAND_LABEL,
                            f"Area-averaged SWRC - {area_label}",
                            marks=[(fc_a, pf_fc, "green", f"FC = {fc_a:.3f}"),
                                   (pwp_a, pf_pwp, "red", f"PWP = {pwp_a:.3f}")],
                            paw=(pf_fc, pf_pwp, fc_a, pwp_a)))
    with g2:
        st.pyplot(plot_kh(thk_m, thk_lo, thk_hi, lk_m, BAND_LABEL,
                          f"Area-averaged K(theta) - {area_label}"))

    st.markdown("#### AOI summary")
    st.table(pd.DataFrame({
        "Variable": ["Field capacity", "Wilting point", "Plant-available water",
                     "Ksat (cm d-1)"],
        "Mean": [np.nanmean(fc_map), np.nanmean(pwp_map), np.nanmean(paw_map),
                 np.nanmean(ks_map)],
        "SD": [np.nanstd(fc_map), np.nanstd(pwp_map), np.nanstd(paw_map),
               np.nanstd(ks_map)],
    }).round(4))

    # ---- GeoTIFF exports ---------------------------------------------------
    def to_tif(arr):
        prof = ref["profile"].copy()
        prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        with MemoryFile() as mf:
            with mf.open(**prof) as dst:
                dst.write(arr.astype(np.float32), 1)
            return mf.read()

    tag = f"{bundle_key(lm.family, lm.input_set)}_{depth}cm"
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("FC GeoTIFF", to_tif(fc_map), file_name=f"FC_{tag}.tif")
    d2.download_button("PWP GeoTIFF", to_tif(pwp_map), file_name=f"PWP_{tag}.tif")
    d3.download_button("PAW GeoTIFF", to_tif(paw_map), file_name=f"PAW_{tag}.tif")
    d4.download_button("Ksat GeoTIFF", to_tif(ks_map), file_name=f"Ksat_{tag}.tif")


# ===========================================================================
# Sidebar + main
# ===========================================================================

def sidebar():
    st.sidebar.header("Model library")
    root = active_root()
    st.sidebar.caption(f"Reading from: `{root}`")
    man = get_manifest()
    if man:
        rows = [{"Run": k, "Model": v.get("family_label", ""),
                 "Inputs": v.get("input_set", ""), "Members": v.get("n_members", 0)}
                for k, v in man["runs"].items() if v.get("status") == "ok"]
        st.sidebar.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)
        st.sidebar.caption(f"Bundle built {man['created']}")
    else:
        st.sidebar.info("No manifest.json found - running straight off the "
                        "training folders. Run build_model_bundle.py to make a "
                        "portable bundle.")
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Grids and axes**\n\n"
        f"- Curve parameter: pF {SWRC_PF[0]} to {SWRC_PF[1]}, step {SWRC_PF[2]}\n"
        f"- SWRC panel: x = theta {THETA_LIMITS[0]}-{THETA_LIMITS[1]}, "
        f"y = pF {PF_LIMITS[0]} to {PF_LIMITS[1]}\n"
        f"- K panel: x = theta {THETA_LIMITS[0]}-{THETA_LIMITS[1]}, "
        f"y = log10 K {LOGK_LIMITS[0]} to {LOGK_LIMITS[1]}\n\n"
        "**Inputs**\n\n"
        "- Sand/silt/clay are renormalised to sum to 100% before prediction, "
        "in both tabs, because the models were trained on closed "
        "compositions.\n"
        "- Area-averaged curves use every valid pixel in the selection.\n\n"
        "**Uncertainty**\n\n"
        "- 10-member deep-ensemble mean +/- 1 SD, everywhere: the shaded "
        "curve bands, the FC/PWP/PAW metrics and the parameter table all use "
        "this one interval.\n"
        "- It is model (epistemic) spread across the ensemble, not a "
        "calibrated predictive interval against measurements.")
    st.sidebar.divider()
    st.sidebar.caption(
        f"{'rasterio OK' if HAS_RASTERIO else 'rasterio MISSING'} | "
        f"{'geopandas OK' if HAS_GEOPANDAS else 'geopandas MISSING'} | "
        f"{'folium OK' if HAS_FOLIUM else 'folium MISSING'}")


def main():
    st.title("Physics-Constrained Inverse-PTF Explorer")
    st.caption("Deep-ensemble inverse neural-network pedotransfer functions. "
               "Three retention/conductivity models x four input sets = 12 "
               "trained ensembles. Outputs: SWRC, K(h), Ksat, FC, PWP and PAW.")
    sidebar()
    t1, t2 = st.tabs(["Own soil data (manual input)",
                      "SOLUS soil data (spatial)"])
    with t1:
        manual_tab()
    with t2:
        spatial_tab()


if __name__ == "__main__":
    main()
