"""
Inverse-PTF Explorer -- physics-constrained deep-ensemble pedotransfer functions
================================================================================

Streamlit app (works in any modern browser: phone, tablet, laptop).

  Tab 1  Single soil      type sand/silt/clay (+BD, OC) -> SWRC, K(theta), Ksat,
                          FC, PWP, PAW and the fitted hydraulic parameters
  Tab 2  SOLUS maps       county / drawn area / point anywhere in Florida ->
                          FC, PWP, PAW, Ksat maps + area-averaged curves
  Tab 3  Models & method  accuracy of the 12 ensembles and how the app works

Run locally:   streamlit run app.py
Configuration (environment variables, all optional):
  PTF_MODEL_DIR       folder with the 12 exported ensembles  (models/ptf_model_bundle)
  PTF_DATA            folder OR https URL of the SOLUS COGs   (data/solus_cog)
  PTF_CURVE_PIXELS    max pixels for exact area curves      (20000)

Author: Nikhil Raj Deep  |  License: MIT (code), CC BY 4.0 (models & derived data)
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ptf_engine import core, plots, raster
from ptf_engine.core import FEATURE_LABELS, INPUT_SETS, MODEL_FAMILIES, run_key

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("PTF_MODEL_DIR", ROOT / "models" / "ptf_model_bundle"))


def _default_data() -> str:
    """PTF_DATA if set; else data/solus_cog in the repo; else a sibling folder
    named Soil_Properties_Maps_COG (the layout used next to the original data)."""
    if os.environ.get("PTF_DATA"):
        return os.environ["PTF_DATA"]
    for cand in (ROOT / "data" / "solus_cog", ROOT.parent / "Soil_Properties_Maps_COG"):
        if (cand / "catalog.json").exists():
            return str(cand)
    return str(ROOT / "data" / "solus_cog")


DATA_LOC = _default_data()
COUNTIES_5070 = ROOT / "data" / "counties" / "fl_counties_epsg5070.geojson"
COUNTIES_WGS84 = ROOT / "data" / "counties" / "fl_counties_wgs84.geojson"
CURVE_PIXELS = int(os.environ.get("PTF_CURVE_PIXELS", 20_000))
MAX_AOI_PIXELS = 4_000_000          # ~ the four largest counties together

SWRC_PF = (-2.0, 7.0, 0.05)
Z_SCORE = 1.0
BAND_LABEL = "Ensemble mean ± 1 SD (10 members)"

st.set_page_config(page_title="Inverse-PTF Explorer", page_icon="💧",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
  @media (max-width: 640px) {
    .block-container {padding-left: .8rem; padding-right: .8rem;}
    h1 {font-size: 1.55rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.25rem;}
  }
  [data-testid="stMetricLabel"] p {font-size: .85rem;}
</style>""", unsafe_allow_html=True)


# ===========================================================================
# Cached resources
# ===========================================================================
@st.cache_resource(show_spinner="Loading ensemble ...")
def get_model(family: str, input_set: str) -> core.Ensemble:
    return core.Ensemble.load(MODEL_ROOT, family, input_set)


@st.cache_resource(show_spinner=False)
def get_store():
    try:
        return raster.SolusStore(DATA_LOC), None
    except Exception as e:                       # missing data -> tab shows why
        return None, str(e)


@st.cache_resource(show_spinner="Preparing the fast raster engine (first run on this server only) ...")
def warm_engine():
    """Compile the Numba kernels once per server process (cached on disk)."""
    pf = np.array([0.0, 1.0])
    for fam in MODEL_FAMILIES:
        en = get_model(fam, "SSC")
        en.predict_area(np.array([[90.0, 5.0, 5.0, 5.0]]), np.array([1.0]), pf, [2.0, 4.2])
    return True


@st.cache_data(show_spinner=False)
def load_counties():
    g5070 = json.loads(COUNTIES_5070.read_text())
    gwgs = json.loads(COUNTIES_WGS84.read_text())
    geoms = {f["properties"]["NAME"]: f["geometry"] for f in g5070["features"]}
    return sorted(geoms), geoms, gwgs


@st.cache_data(show_spinner=False)
def model_metrics():
    rows = []
    for fam in MODEL_FAMILIES:
        for iset in INPUT_SETS:
            p = MODEL_ROOT / run_key(fam, iset) / "test_bootstrap_metric_CIs.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            for task in ("SWRC_theta", "K(h)_log10", "Ksat_log10"):
                for met in ("RMSE", "R2", "KGE"):
                    r = d[(d.Task == task) & (d.Metric == met)]
                    if len(r):
                        r = r.iloc[0]
                        rows.append(dict(Model=MODEL_FAMILIES[fam]["short"], Inputs=iset,
                                         Task=task, Metric=met, Value=r.PointEstimate,
                                         CI=f"{r['CI_2.5%']:.3f} – {r['CI_97.5%']:.3f}"))
    return pd.DataFrame(rows)


def band(stack):
    m, lo, hi, sd = core.ensemble_band(stack, z=Z_SCORE)
    return m, lo, hi, sd


# ===========================================================================
# Shared widgets
# ===========================================================================
def model_selector(prefix: str):
    c1, c2 = st.columns(2)
    iset = c1.selectbox("Input set", list(INPUT_SETS),
                        format_func=lambda k: f"{k} — {INPUT_SETS[k]}",
                        key=f"{prefix}_iset")
    fam = c2.selectbox("Model", list(MODEL_FAMILIES), index=2,
                       format_func=lambda k: MODEL_FAMILIES[k]["label"],
                       key=f"{prefix}_fam")
    return fam, iset


def retention_points(prefix: str):
    c1, c2 = st.columns(2)
    fc = c1.radio("Field capacity", ["pF 2.0 (sandy soils)", "pF 2.54 (classic)"],
                  key=f"{prefix}_fc", horizontal=True)
    pwp = c2.number_input("Wilting point (pF)", 3.0, 5.0, 4.2, 0.01, key=f"{prefix}_pwp")
    return (2.0 if fc.startswith("pF 2.0") else 2.54), float(pwp)


def curve_panels(pf, th, lk, fc_pt, pwp_pt, title_suffix, k_theta=None):
    """th / lk: (M, P) member stacks.  Draws SWRC + K(theta) side by side
    (stacked on phones)."""
    th_m, th_lo, th_hi, _ = band(th)
    lk_m, _, _, _ = band(lk)
    g1, g2 = st.columns(2)
    g1.plotly_chart(plots.swrc_figure(pf, th_m, th_lo, th_hi, BAND_LABEL,
                                      f"SWRC — {title_suffix}", fc=fc_pt, pwp=pwp_pt),
                    use_container_width=True, config=plots.CONFIG)
    g2.plotly_chart(plots.k_figure(th_m, th_lo, th_hi, lk_m, BAND_LABEL,
                                   f"K(θ) — {title_suffix}"),
                    use_container_width=True, config=plots.CONFIG)
    return th_m, th_lo, th_hi, lk_m


# ===========================================================================
# TAB 1 -- one soil
# ===========================================================================
def manual_tab():
    st.markdown("##### Enter measured soil properties")
    fam, iset = model_selector("m")
    en = get_model(fam, iset)

    left, right = st.columns([1, 1.7], gap="large")
    vals = {}
    with left:
        for f in en.features:
            label, lo, hi, default, step = FEATURE_LABELS[f]
            if f == "Depth_harm_cm":
                vals[f] = float(st.selectbox("Depth (cm)", raster.DEPTHS, key="m_depth"))
            else:
                vals[f] = st.number_input(label, lo, hi, default, step, key=f"m_{f}")
        total = vals["Sand_pct"] + vals["Silt_pct"] + vals["Clay_pct"]
        if total <= 0:
            st.error("Sand + silt + clay must be greater than zero.")
            return
        if abs(total - 100) > 0.01:
            sa, si, cl = core.normalize_texture(vals["Sand_pct"], vals["Silt_pct"],
                                                vals["Clay_pct"])
            vals["Sand_pct"], vals["Silt_pct"], vals["Clay_pct"] = float(sa), float(si), float(cl)
            st.info(f"Sand + silt + clay = {total:.1f} %, rescaled to 100 % "
                    f"({vals['Sand_pct']:.1f} / {vals['Silt_pct']:.1f} / "
                    f"{vals['Clay_pct']:.1f}). Ratios are kept.")
        pf_fc, pf_pwp = retention_points("m")

    pf = core.pf_grid(*SWRC_PF)
    x = np.array([[vals[f] for f in en.features]])
    th_all, lk, par = en.predict_members(x, np.concatenate([pf, [pf_fc, pf_pwp]]), pf)
    th = th_all[:, 0, :len(pf)]
    pts = th_all[:, 0, len(pf):]
    lk = lk[:, 0, :]
    fc_m, fc_lo, fc_hi, _ = band(pts[:, :1])
    pwp_m, pwp_lo, pwp_hi, _ = band(pts[:, 1:])
    paw_m, paw_lo, paw_hi, _ = band(pts[:, :1] - pts[:, 1:])
    ks_m, ks_lo, ks_hi, _ = band(par["log10_ksat"][:, :1])

    with right:
        c = st.columns(4)
        c[0].metric(f"FC (pF {pf_fc})", f"{fc_m[0]:.3f}", help=f"± 1 SD: {fc_lo[0]:.3f} – {fc_hi[0]:.3f}")
        c[1].metric(f"PWP (pF {pf_pwp})", f"{pwp_m[0]:.3f}", help=f"{pwp_lo[0]:.3f} – {pwp_hi[0]:.3f}")
        c[2].metric("PAW", f"{paw_m[0]:.3f}", help=f"{paw_lo[0]:.3f} – {paw_hi[0]:.3f}")
        c[3].metric("Ksat (cm d⁻¹)", f"{10 ** ks_m[0]:,.1f}",
                    help=f"{10 ** ks_lo[0]:,.2f} – {10 ** ks_hi[0]:,.2f} cm d⁻¹ "
                         f"({10 ** ks_m[0] / 24:,.3f} cm h⁻¹)")
        st.caption(f"Texture class **{core.texture_class(vals['Sand_pct'], vals['Silt_pct'], vals['Clay_pct'])}** · "
                   f"θ in cm³ cm⁻³ · {BAND_LABEL}")
        rows = []
        for name in en.param_names:
            key = "log10_ksat" if name == "ksat" else name
            m, lo, hi, _ = band(par[key][:, :1].astype(float))
            if name == "ksat":
                rows.append(["Ksat (cm d⁻¹)", f"{10 ** m[0]:.4g}", f"{10 ** lo[0]:.4g} – {10 ** hi[0]:.4g}"])
            else:
                rows.append([name, f"{m[0]:.4g}", f"{lo[0]:.4g} – {hi[0]:.4g}"])
        st.dataframe(pd.DataFrame(rows, columns=["Parameter", "Ensemble mean", "± 1 SD"]),
                     hide_index=True, use_container_width=True)

    th_m, th_lo, th_hi, lk_m = curve_panels(
        pf, th, lk, (float(fc_m[0]), pf_fc), (float(pwp_m[0]), pf_pwp),
        f"{MODEL_FAMILIES[fam]['short']} / {iset}")

    df = pd.DataFrame({"pF": pf, "h_cm": 10.0 ** pf, "theta_mean": th_m, "theta_lo": th_lo,
                       "theta_hi": th_hi, "log10K_mean": lk_m, "K_cm_d": 10.0 ** lk_m})
    st.download_button("⬇ Curves (CSV)", df.to_csv(index=False).encode(),
                       file_name=f"curves_{run_key(fam, iset)}.csv", mime="text/csv")


# ===========================================================================
# TAB 2 -- SOLUS maps
# ===========================================================================
@st.cache_data(show_spinner=False, max_entries=12)
def run_area(fam, iset, depth, geom_json, label, pf_fc, pf_pwp, exact_curves):
    """Everything for one AOI.  Cached: re-rendering (map pans, widget clicks)
    or asking again for the same county is instant."""
    t0 = time.perf_counter()
    en = get_model(fam, iset)
    store, _ = get_store()
    aoi = store.aoi_from_geometry(json.loads(geom_json), label)
    if aoi is None:
        return {"error": "The area does not overlap the SOLUS grid."}
    if aoi.mask.sum() > MAX_AOI_PIXELS:
        return {"error": f"The area covers {aoi.mask.sum():,} pixels; the limit is "
                         f"{MAX_AOI_PIXELS:,} (about four large counties). Draw a smaller area."}
    inp = raster.gather_inputs(store, en.features, depth, aoi, en.train_min, en.train_max)
    if inp is None:
        return {"error": "No valid SOLUS pixels in this area (water or nodata)."}
    t_read = time.perf_counter() - t0

    t1 = time.perf_counter()
    maps = en.predict_area(inp.X_unique, inp.counts, [], [pf_fc, pf_pwp])
    t_maps = time.perf_counter() - t1

    t2 = time.perf_counter()
    pf = core.pf_grid(*SWRC_PF)
    Xc, wc, n_used, exact = raster.curve_rows(
        inp, max_pixels=10 ** 12 if exact_curves else CURVE_PIXELS)
    cur = en.predict_area(Xc, wc, pf, [])
    t_curves = time.perf_counter() - t2
    # sampling standard error of the ensemble-mean area curve (0 if exact)
    if exact:
        se = 0.0
    else:
        fpc = 1 - n_used / inp.n_valid
        se = float(np.sqrt(cur["curve_theta_var"].mean(0) / n_used * fpc).max())

    return dict(
        fc=raster.scatter(maps["points_mean"][:, 0], inp),
        pwp=raster.scatter(maps["points_mean"][:, 1], inp),
        paw=raster.scatter(maps["paw_mean"], inp),
        paw_sd=raster.scatter(maps["paw_sd"], inp),
        ksat=raster.scatter(10 ** maps["log10ks_mean"], inp),
        curve_theta=cur["curve_theta"], curve_log10k=cur["curve_log10k"],
        n_valid=inp.n_valid, n_unique=len(inp.counts), curve_pixels=n_used,
        curve_exact=exact, curve_se=se, diagnostics=inp.diagnostics,
        texture_sum=inp.texture_sum, transform=tuple(aoi.transform)[:6],
        crs=store.crs, timings=dict(read=t_read, maps=t_maps, curves=t_curves,
                                    total=time.perf_counter() - t0))


def geotiff_zip(res, tag):
    import rasterio
    aoi_like = raster.AOI(window=None, mask=None,
                          transform=rasterio.Affine(*res["transform"]), label=tag)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key, desc in (("fc", "Field capacity (cm3 cm-3)"),
                          ("pwp", "Wilting point (cm3 cm-3)"),
                          ("paw", "Plant-available water (cm3 cm-3)"),
                          ("paw_sd", "PAW ensemble SD (cm3 cm-3)"),
                          ("ksat", "Saturated hydraulic conductivity (cm d-1)")):
            z.writestr(f"{key.upper()}_{tag}.tif",
                       raster.to_geotiff_bytes(res[key], aoi_like, res["crs"], desc))
    return buf.getvalue()


def show_area_result(res, fam, iset, depth, label, pf_fc, pf_pwp):
    if "error" in res:
        st.error(res["error"])
        return
    tm = res["timings"]
    c = st.columns(4)
    c[0].metric("Pixels (100 m)", f"{res['n_valid']:,}")
    c[1].metric("Unique soils", f"{res['n_unique']:,}",
                help="Pixels with identical SOLUS inputs are computed once.")
    c[2].metric("Mean PAW", f"{np.nanmean(res['paw']):.3f}")
    c[3].metric("Compute time", f"{tm['total']:.1f} s",
                help=f"read {tm['read']:.2f} s · maps {tm['maps']:.2f} s · "
                     f"curves {tm['curves']:.2f} s")

    theta_maps = [res["fc"], res["pwp"], res["paw"]]
    allv = np.concatenate([m[np.isfinite(m)] for m in theta_maps])
    vmin, vmax = (float(v) for v in np.percentile(allv, [2, 98]))
    title = f"{MODEL_FAMILIES[fam]['short']} / {iset} · {depth} cm"
    r1 = st.columns(2)
    r1[0].plotly_chart(plots.map_figure(res["fc"], f"Field capacity (pF {pf_fc})", "θ",
                                        plots.SEQ_WATER, vmin, vmax),
                       use_container_width=True, config=plots.CONFIG)
    r1[1].plotly_chart(plots.map_figure(res["pwp"], f"Wilting point (pF {pf_pwp})", "θ",
                                        plots.SEQ_WATER, vmin, vmax),
                       use_container_width=True, config=plots.CONFIG)
    r2 = st.columns(2)
    r2[0].plotly_chart(plots.map_figure(res["paw"], "Plant-available water", "θ",
                                        plots.SEQ_WATER, vmin, vmax),
                       use_container_width=True, config=plots.CONFIG)
    r2[1].plotly_chart(plots.map_figure(np.log10(np.clip(res["ksat"], 1e-6, None)),
                                        "log₁₀ Ksat", "log₁₀ cm d⁻¹", plots.SEQ_KSAT,
                                        fmt=".2f"),
                       use_container_width=True, config=plots.CONFIG)
    st.caption(f"{title} — {label}. FC, PWP and PAW share one colour scale "
               "(2nd–98th percentile); Ksat has its own.")

    with st.expander("PAW uncertainty map (ensemble SD)"):
        st.plotly_chart(plots.map_figure(res["paw_sd"], "PAW ensemble SD", "θ",
                                         plots.SEQ_SD),
                        use_container_width=True, config=plots.CONFIG)

    st.markdown("##### Area-averaged curves")
    pf = core.pf_grid(*SWRC_PF)
    th_m = res["curve_theta"].mean(0)
    fc_a = float(np.interp(pf_fc, pf, th_m))
    pwp_a = float(np.interp(pf_pwp, pf, th_m))
    th_m, th_lo, th_hi, lk_m = curve_panels(pf, res["curve_theta"], res["curve_log10k"],
                                            (fc_a, pf_fc), (pwp_a, pf_pwp), label)
    if res["curve_exact"]:
        st.caption(f"Averaged over every one of the {res['n_valid']:,} pixels.")
    else:
        st.caption(f"Averaged over a random sample of {res['curve_pixels']:,} of "
                   f"{res['n_valid']:,} pixels; sampling standard error ≤ "
                   f"{res['curve_se']:.4f} cm³ cm⁻³ (largest over all pF). Maps always "
                   "use every pixel. Tick *Exact area curves* for the full average.")

    summ = pd.DataFrame({
        "Variable": ["Field capacity", "Wilting point", "Plant-available water",
                     "PAW ensemble SD", "Ksat (cm d⁻¹)"],
        "Mean": [np.nanmean(res[k]) for k in ("fc", "pwp", "paw", "paw_sd", "ksat")],
        "SD across pixels": [np.nanstd(res[k]) for k in ("fc", "pwp", "paw", "paw_sd", "ksat")],
    }).round(4)
    st.dataframe(summ, hide_index=True, use_container_width=True)

    diag = pd.DataFrame(res["diagnostics"])
    bad = diag[diag.OK == "no"].Feature.tolist()
    with st.expander("Input layers reaching the model" + ("  ⚠ out of training range" if bad else ""),
                     expanded=bool(bad)):
        st.dataframe(diag, hide_index=True, use_container_width=True)
        if res["texture_sum"]:
            lo, med, hi = res["texture_sum"]
            st.caption(f"SOLUS sand + silt + clay ranged {lo:.0f}–{hi:.0f} % (median {med:.0f} %) "
                       "before each pixel was rescaled to 100 %. Values above are after rescaling.")
        if bad:
            st.warning("More than 1 % of pixels fall outside the training range for: "
                       + ", ".join(bad) + ". Treat those pixels with caution.")

    tag = f"{run_key(fam, iset)}_{depth}cm_{label.replace(' ', '_')}"
    curves = pd.DataFrame({"pF": pf, "theta_mean": th_m, "theta_lo": th_lo,
                           "theta_hi": th_hi, "log10K_mean": lk_m})
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Maps — GeoTIFF (zip)", geotiff_zip(res, tag),
                       file_name=f"maps_{tag}.zip", mime="application/zip",
                       use_container_width=True)
    d2.download_button("⬇ Area curves (CSV)", curves.to_csv(index=False).encode(),
                       file_name=f"area_curves_{tag}.csv", mime="text/csv",
                       use_container_width=True)


def point_result(en, store, depth, x, y, label, pf_fc, pf_pwp):
    vars_ = [raster.FEATURE_TO_VAR[f] for f in en.features if f != "Depth_harm_cm"]
    vals = store.pixel(x, y, list(dict.fromkeys(vars_ + ["Sand", "Silt", "Clay"])), depth)
    if vals is None:
        st.error("No SOLUS data at this location (outside Florida, water or nodata).")
        return
    sa, si, cl = core.normalize_texture(vals["Sand"], vals["Silt"], vals["Clay"])
    phys = dict(vals, Sand=float(sa), Silt=float(si), Clay=float(cl))
    xrow = np.array([[float(depth) if f == "Depth_harm_cm" else phys[raster.FEATURE_TO_VAR[f]]
                      for f in en.features]])
    pf = core.pf_grid(*SWRC_PF)
    th_all, lk, par = en.predict_members(xrow, np.concatenate([pf, [pf_fc, pf_pwp]]), pf)
    th, pts, lk = th_all[:, 0, :len(pf)], th_all[:, 0, len(pf):], lk[:, 0]
    fc, pwp = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    c = st.columns(4)
    c[0].metric(f"FC (pF {pf_fc})", f"{fc:.3f}")
    c[1].metric(f"PWP (pF {pf_pwp})", f"{pwp:.3f}")
    c[2].metric("PAW", f"{fc - pwp:.3f}")
    c[3].metric("Ksat (cm d⁻¹)", f"{10 ** float(par['log10_ksat'][:, 0].mean()):,.1f}")
    st.caption("SOLUS pixel (after texture rescaling): "
               + " · ".join(f"{k} {v:.3g}" for k, v in phys.items())
               + f" · {depth} cm · {core.texture_class(phys['Sand'], phys['Silt'], phys['Clay'])}")
    th_m, th_lo, th_hi, lk_m = curve_panels(pf, th, lk, (fc, pf_fc), (pwp, pf_pwp), label)
    st.download_button("⬇ Curves (CSV)",
                       pd.DataFrame({"pF": pf, "theta_mean": th_m, "theta_lo": th_lo,
                                     "theta_hi": th_hi, "log10K_mean": lk_m})
                       .to_csv(index=False).encode(),
                       file_name=f"curves_{run_key(en.family, en.input_set)}_{depth}cm_point.csv")


def spatial_tab():
    store, err = get_store()
    if store is None:
        st.error(f"SOLUS data not found at `{DATA_LOC}`.\n\n{err}\n\n"
                 "Run `python tools/build_cogs.py` or set PTF_DATA.")
        return
    st.markdown("##### Predict from SOLUS soil maps (100 m, Florida)")
    fam, iset = model_selector("s")
    en = get_model(fam, iset)
    c0, c1 = st.columns([1, 2])
    depth = c0.selectbox("Depth (cm)", raster.DEPTHS, key="s_depth")
    with c1:
        pf_fc, pf_pwp = retention_points("s")

    mode = st.radio("Location", ["County", "Draw an area", "Pick a point", "Latitude / longitude"],
                    horizontal=True, key="s_mode")
    exact = st.checkbox("Exact area curves (use every pixel — slower for SOC models)",
                        key="s_exact")

    from rasterio.warp import transform_bounds, transform_geom
    from shapely.geometry import Point, box, mapping, shape
    minlon, minlat, maxlon, maxlat = transform_bounds(store.crs, "EPSG:4326", *store.bounds)

    request = None
    if mode == "County":
        names, geoms, _ = load_counties()
        cc1, cc2 = st.columns([3, 1])
        county = cc1.selectbox("County", names, index=names.index("Alachua"),
                               key="s_county", label_visibility="collapsed")
        if cc2.button("Run", type="primary", use_container_width=True):
            request = (json.dumps(geoms[county]), f"{county} County")

    elif mode == "Latitude / longitude":
        cl, cr, cb = st.columns([2, 2, 1])
        lat = cl.number_input("Latitude", value=29.6516, format="%.5f", key="s_lat")
        lon = cr.number_input("Longitude", value=-82.3248, format="%.5f", key="s_lon")
        cb.write("")
        if cb.button("Predict", type="primary", use_container_width=True):
            st.session_state["s_point"] = (lon, lat)
        if st.session_state.get("s_point") and mode == "Latitude / longitude":
            lon, lat = st.session_state["s_point"]
            p = shape(transform_geom("EPSG:4326", store.crs, mapping(Point(lon, lat))))
            point_result(en, store, depth, p.x, p.y, f"{lat:.4f}, {lon:.4f}", pf_fc, pf_pwp)
        return

    else:
        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
        is_point = mode == "Pick a point"
        fmap = folium.Map(location=[(minlat + maxlat) / 2, (minlon + maxlon) / 2],
                          zoom_start=6, tiles="CartoDB positron", control_scale=True)
        _, _, gwgs = load_counties()
        folium.GeoJson(gwgs, name="Counties",
                       style_function=lambda f: dict(color="#64748b", weight=0.7, fillOpacity=0),
                       tooltip=folium.GeoJsonTooltip(fields=["NAME"], labels=False)).add_to(fmap)
        fmap.fit_bounds([[minlat, minlon], [maxlat, maxlon]])
        Draw(export=False, draw_options={"polyline": False, "circle": False,
                                         "circlemarker": False, "marker": is_point,
                                         "rectangle": not is_point,
                                         "polygon": not is_point}).add_to(fmap)
        st.caption("Tap/click the map to place a point." if is_point
                   else "Use the ▭ or ⬠ tool (top-left) to draw your area, then press Run.")
        out = st_folium(fmap, height=430, use_container_width=True,
                        returned_objects=["all_drawings", "last_clicked"],
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
                return
            p = shape(transform_geom("EPSG:4326", store.crs, mapping(Point(*ll))))
            point_result(en, store, depth, p.x, p.y, f"{ll[1]:.4f}, {ll[0]:.4f}", pf_fc, pf_pwp)
            return
        polys = [s for s in ((out or {}).get("all_drawings") or [])
                 if s["geometry"]["type"] in ("Polygon", "MultiPolygon")]
        if polys and st.button("Run for drawn area", type="primary"):
            g = shape(transform_geom("EPSG:4326", store.crs, polys[-1]["geometry"]))
            g = g.intersection(box(*store.bounds))
            if g.is_empty:
                st.error("The area lies outside the SOLUS grid.")
                return
            request = (json.dumps(mapping(g)), "drawn area")

    if request is not None:
        st.session_state["s_request"] = (fam, iset, depth, request[0], request[1],
                                         pf_fc, pf_pwp, exact)
    req = st.session_state.get("s_request")
    if req is None:
        st.info("Choose a location and press **Run**.")
        return
    if req[:3] != (fam, iset, depth) or req[5:] != (pf_fc, pf_pwp, exact):
        # settings changed since the last run: recompute for the same place
        req = (fam, iset, depth, req[3], req[4], pf_fc, pf_pwp, exact)
        st.session_state["s_request"] = req
    if core.HAS_NUMBA:
        warm_engine()
    with st.spinner(f"Predicting {req[4]} ..."):
        res = run_area(*req)
    show_area_result(res, req[0], req[1], req[2], req[4], req[5], req[6])


# ===========================================================================
# TAB 3 -- about
# ===========================================================================
def about_tab():
    st.markdown("""
##### What the app does
A physics-constrained **inverse pedotransfer function (PTF)**: a neural network
predicts the *parameters* of a closed-form soil hydraulic model from basic soil
properties, and the physics model turns those parameters into the full
**soil-water retention curve (SWRC)** and **unsaturated hydraulic conductivity
K(h)**. Because the curve always comes from a physical equation, it is smooth,
monotonic and bounded for every soil.

| Model family | Retention θ(h) | Conductivity K(h) | Parameters |
|---|---|---|---|
| Classic VG | van Genuchten (1980), m = 1 − 1/n | VG–Mualem | θr, θs, α, n, Ks, τ |
| Enhanced VG | VG with dry-end correction (θr → 0 at pF 6.8) | VG–Mualem | θr, θs, α, n, Ks, τ |
| PDI | Peters–Durner–Iden: capillary + adsorbed water | capillary + film flow | θr, θs, α, n, Ks, ω, λ |

Four input sets (SSC, SSC+BD, SSC+OC, SSC+BD+OC) × three families = **12
ensembles of 10 networks** each. The shaded band is the ensemble mean ± 1 SD
(epistemic spread across members, not a calibrated predictive interval).

**Inputs** — sand, silt and clay are rescaled to sum to 100 % (the models were
trained on closed compositions; SOLUS layers are not). Depth is a model input.
**Spatial data** — SOLUS 100 m soil-property maps for Florida (5–100 cm),
stored as Cloud-Optimized GeoTIFFs on one grid.

##### Why the maps are fast
1. Only the tiles that touch the chosen area are read (COG windowed reads).
2. Pixels with identical inputs are computed once (exact de-duplication).
3. The physics runs in compiled, multithreaded kernels; maps need only the
   two retention points, so they are ready first.
4. Area-averaged curves use every pixel up to 20 000 unique soils, otherwise a
   random pixel sample with its standard error shown (tick *Exact* for all).
5. Results are cached — repeating a county or changing tabs is instant.
""")
    df = model_metrics()
    if len(df):
        st.markdown("##### Independent test-set accuracy (cluster bootstrap, 95 % CI)")
        task = st.radio("Target", ["SWRC_theta", "K(h)_log10", "Ksat_log10"], horizontal=True,
                        format_func={"SWRC_theta": "θ (SWRC)", "K(h)_log10": "log₁₀K(h)",
                                     "Ksat_log10": "log₁₀Ksat"}.get, key="a_task")
        t = df[df.Task == task].pivot_table(index=["Model", "Inputs"], columns="Metric",
                                            values="Value").reset_index()
        st.dataframe(t.round(3), hide_index=True, use_container_width=True)
    st.caption(f"Engine: NumPy{' + Numba' if core.HAS_NUMBA else ''} · models from `{MODEL_ROOT.name}` · "
               f"data from `{DATA_LOC}`")


# ===========================================================================
def main():
    st.title("💧 Inverse-PTF Explorer")
    st.caption("Soil-water retention, hydraulic conductivity and plant-available water from "
               "physics-constrained deep-ensemble pedotransfer functions — for one soil or any area in Florida.")
    t1, t2, t3 = st.tabs(["🧪 Single soil", "🗺️ SOLUS maps", "📘 Models & method"])
    with t1:
        manual_tab()
    with t2:
        spatial_tab()
    with t3:
        about_tab()


main()
