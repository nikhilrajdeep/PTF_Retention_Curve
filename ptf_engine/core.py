"""
ptf_engine.core
===============
Torch-free inference for the 12 physics-constrained inverse-PTF ensembles
(3 model families x 4 input sets).

The networks are small multilayer perceptrons, so the forward pass is written
directly in NumPy from the exported weights (tools/export_models.py).  The
physics (retention + conductivity) is a line-by-line port of ptf_core.py, which
was itself copied from the training notebooks.  tests/test_engine.py checks
this module against the original PyTorch code.

Two code paths share the same equations:

* vectorised NumPy  -- used for single soils (manual tab, a clicked pixel)
* fused Numba kernels -- used for rasters.  One pass per ensemble member gives
  every pixel's FC/PWP and the area-averaged curves together, multithreaded,
  without materialising (pixels x pF) arrays.  If Numba is not installed the
  NumPy path is used in chunks instead (same numbers, slower).

Author: Nikhil Raj Deep
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:                                    # optional accelerator
    import numba as nb
    HAS_NUMBA = True
except Exception:                       # pragma: no cover
    nb = None
    HAS_NUMBA = False

EPS = 1.0e-12
LOG_TINY = -60.0
LN10 = math.log(10.0)

# ===========================================================================
# Registry
# ===========================================================================
MODEL_FAMILIES = {
    "VGM":   dict(label="Classic VG (van Genuchten-Mualem)", short="Classic VG"),
    "VGDRY": dict(label="Enhanced VG (dry-end corrected)",   short="Enhanced VG"),
    "PDI":   dict(label="PDI (Peters-Durner-Iden)",          short="PDI"),
}
INPUT_SETS = {
    "SSC":     "Sand, Silt, Clay",
    "SSCBD":   "Sand, Silt, Clay, Bulk density",
    "SSCOC":   "Sand, Silt, Clay, Organic carbon",
    "SSCBDOC": "Sand, Silt, Clay, Bulk density, Organic carbon",
}
# feature -> (label, min, max, default, step)
FEATURE_LABELS = {
    "Sand_pct":      ("Sand (%)",              0.0, 100.0, 92.0, 0.1),
    "Silt_pct":      ("Silt (%)",              0.0, 100.0,  5.0, 0.1),
    "Clay_pct":      ("Clay (%)",              0.0, 100.0,  3.0, 0.1),
    "BD_g_cm3":      ("Bulk density (g cm-3)", 0.5,   2.2,  1.50, 0.01),
    "OC_pct":        ("Organic carbon (%)",    0.0,  30.0,  1.0, 0.1),
    "Depth_harm_cm": ("Depth (cm)",            0.0, 250.0,  5.0, 1.0),
}


def run_key(family: str, input_set: str) -> str:
    return f"{family}_{input_set}"


def pf_grid(lo: float, hi: float, step: float) -> np.ndarray:
    return np.round(np.arange(lo, hi + 0.5 * step, step), 5)


# ===========================================================================
# Elementwise helpers (NumPy, float32 like the PyTorch originals)
# ===========================================================================
def _softplus(x):
    # torch.nn.functional.softplus(beta=1, threshold=20)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def _logsigmoid(x):
    return np.minimum(x, 0.0) - np.log1p(np.exp(-np.abs(x)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_ACT = {
    "relu": lambda x: np.maximum(x, 0.0, out=x),
    "tanh": lambda x: np.tanh(x, out=x),
    "sigmoid": lambda x: _sigmoid(x),
    "silu": lambda x: x * _sigmoid(x),
}


# ===========================================================================
# Physics -- NumPy.  p[...] are (N,1) arrays, pf is (1,P) or (N,P).
# ===========================================================================
def _vg_log_components(log_alpha, n, pf):
    log_alpha_h = log_alpha + pf * LN10
    x = n * log_alpha_h
    log_u = _logsigmoid(-x)
    log_one_minus_u = _logsigmoid(x)
    m = np.maximum(1.0 - 1.0 / n, 1.0e-6)
    return m * log_u, log_one_minus_u, m


def vg_theta(p, pf):
    log_se, _, _ = _vg_log_components(p["log_alpha"], p["n"], pf)
    se = np.clip(np.exp(log_se), EPS, 1.0)
    return np.clip(p["theta_r"] + (p["theta_s"] - p["theta_r"]) * se, 0.0, 1.0)


def vgm_log10k(p, pf):
    log_se, log_1mu, m = _vg_log_components(p["log_alpha"], p["n"], pf)
    z = np.minimum(m * log_1mu, -1.0e-12)
    log_bracket = np.log(np.maximum(-np.expm1(z), 1.0e-30))
    ln_krel = p["tau"] * log_se + 2.0 * log_bracket
    return p["log10_ksat"] + ln_krel / LN10


def vgdry_theta(p, pf, pf_dry_ref=6.8):
    log_se, _, _ = _vg_log_components(p["log_alpha"], p["n"], pf)
    se = np.exp(log_se)
    la = p["log_alpha"]
    c = 1.0 + 0.1 / p["n"]
    num = _softplus(c * (la + pf * LN10))
    den = np.maximum(_softplus(c * (la + pf_dry_ref * LN10)), EPS)
    f_dry = np.clip(1.0 - num / den, 0.0, 1.0)
    theta = (p["theta_s"] - p["theta_r"]) * se + p["theta_r"] * f_dry
    return np.clip(theta, 0.0, 1.0)


def _pdi_sat(p, pf, h0_pf, ts_gap):
    la, n = p["log_alpha"], p["n"]
    tr, ts = p["theta_r"], p["theta_s"]
    log_se, _, _ = _vg_log_components(la, n, pf)
    log_se0, _, _ = _vg_log_components(la, n, h0_pf)                 # (N,1)
    delta = np.minimum(log_se0 - log_se, -1.0e-7)
    log_1m_ratio = np.log(np.maximum(-np.expm1(delta), 1.0e-30))
    log_denom = np.log(np.maximum(-np.expm1(np.minimum(log_se0, -1.0e-7)), 1.0e-30))
    log_scap = np.clip(log_se + log_1m_ratio - log_denom, LOG_TINY, 0.0)
    scap = np.clip(np.exp(log_scap), 0.0, 1.0)

    xa = -p["log10_alpha"]
    gap = np.maximum(ts - tr, ts_gap)
    ratio = tr / gap
    b = np.maximum(0.1 + (0.2 / n ** 2) * (1.0 - np.exp(-ratio ** 2)), 1e-6)
    sad = 1.0 + (pf - xa + b * _softplus((xa - pf) / b)) / np.minimum(xa - h0_pf, -1.0e-6)
    sad = np.clip(sad, 0.0, 1.0)

    dry = pf >= h0_pf
    scap = np.where(dry, 0.0, scap)
    log_scap = np.where(dry, LOG_TINY, log_scap)
    sad = np.where(dry, 0.0, sad)
    return log_scap, scap, sad


def pdi_theta(p, pf, h0_pf=6.8, ts_gap=0.01):
    _, scap, sad = _pdi_sat(p, pf, h0_pf, ts_gap)
    theta = (p["theta_s"] - p["theta_r"]) * scap + p["theta_r"] * sad
    return np.clip(theta, 0.0, 1.0)


def pdi_log10k(p, pf, h0_pf=6.8, h0_cm=None, film_a=-1.5, ts_gap=0.01):
    if h0_cm is None:
        h0_cm = 10.0 ** h0_pf
    la, n = p["log_alpha"], p["n"]
    omega = np.clip(p["omega"], 1.0e-7, 1.0 - 1.0e-7)
    log_scap, _, sad = _pdi_sat(p, pf, h0_pf, ts_gap)
    _, log_1mu, m = _vg_log_components(la, n, pf)
    _, log_1mu0, _ = _vg_log_components(la, n, h0_pf)
    log_ratio = np.minimum(log_1mu - log_1mu0, 0.0)
    z = np.minimum(m * log_ratio, -1.0e-7)
    log_bracket = np.maximum(np.log(np.maximum(-np.expm1(z), 1.0e-30)), LOG_TINY)
    ln_kcap = np.minimum(p["lambda"] * log_scap + 2.0 * log_bracket, 0.0)
    ln_kcap = np.where(pf >= h0_pf, LOG_TINY, ln_kcap)
    ln_kfilm = np.minimum(film_a * (1.0 - sad) * (math.log(h0_cm) + la), 0.0)
    ln_rel = np.logaddexp(np.log1p(-omega) + ln_kcap, np.log(omega) + ln_kfilm)
    ln_rel = np.minimum(ln_rel, 0.0)
    return p["log10_ksat"] + ln_rel / LN10


# ===========================================================================
# Physics -- fused Numba kernels for rasters
# ===========================================================================
if HAS_NUMBA:
    _jit = dict(fastmath=False, cache=True, nogil=True)

    @nb.njit(**_jit)
    def _nb_softplus(x):
        if x > 20.0:
            return x
        return math.log1p(math.exp(x))

    @nb.njit(**_jit)
    def _nb_logsig(x):
        return min(x, 0.0) - math.log1p(math.exp(-abs(x)))

    @nb.njit(parallel=True, **_jit)
    def _nb_vg_family(tr, ts, la, n, l10ks, tau, w, pf, n_curve, dry, pf_dry_ref,
                      block):
        """Classic / dry-corrected VG.  Returns
        th_sum (P_curve,), lk_sum (P_curve,) -- weighted sums over rows
        pts    (N, P - P_curve)               -- theta at the extra pF points"""
        N = tr.shape[0]
        P = pf.shape[0]
        n_pts = P - n_curve
        nblk = (N + block - 1) // block
        th_acc = np.zeros((nblk, n_curve))
        t2_acc = np.zeros((nblk, n_curve))
        lk_acc = np.zeros((nblk, n_curve))
        pts = np.empty((N, n_pts), dtype=np.float32)
        for bi in nb.prange(nblk):
            i0 = bi * block
            i1 = min(N, i0 + block)
            for i in range(i0, i1):
                ni = n[i]
                m = max(1.0 - 1.0 / ni, 1.0e-6)
                lai = la[i]
                c = 1.0 + 0.1 / ni
                den = max(_nb_softplus(c * (lai + pf_dry_ref * LN10)), EPS)
                wi = w[i]
                for j in range(P):
                    lah = lai + pf[j] * LN10
                    x = ni * lah
                    L = math.log1p(math.exp(-abs(x)))      # shared by both log-sigmoids
                    log_se = m * (min(-x, 0.0) - L)
                    if dry:
                        se = math.exp(log_se)
                        fd = 1.0 - _nb_softplus(c * lah) / den
                        fd = min(max(fd, 0.0), 1.0)
                        th = (ts[i] - tr[i]) * se + tr[i] * fd
                    else:
                        se = min(max(math.exp(log_se), EPS), 1.0)
                        th = tr[i] + (ts[i] - tr[i]) * se
                    th = min(max(th, 0.0), 1.0)
                    if j < n_curve:
                        z = min(m * (min(x, 0.0) - L), -1.0e-12)
                        lb = math.log(max(-math.expm1(z), 1.0e-30))
                        lk = l10ks[i] + (tau[i] * log_se + 2.0 * lb) / LN10
                        th_acc[bi, j] += wi * th
                        t2_acc[bi, j] += wi * th * th
                        lk_acc[bi, j] += wi * lk
                    else:
                        pts[i, j - n_curve] = th
        return th_acc.sum(axis=0), t2_acc.sum(axis=0), lk_acc.sum(axis=0), pts

    @nb.njit(parallel=True, **_jit)
    def _nb_pdi(tr, ts, la, l10a, n, l10ks, omega, lam, w, pf, n_curve,
                h0_pf, h0_cm, film_a, ts_gap, block):
        N = tr.shape[0]
        P = pf.shape[0]
        n_pts = P - n_curve
        nblk = (N + block - 1) // block
        th_acc = np.zeros((nblk, n_curve))
        t2_acc = np.zeros((nblk, n_curve))
        lk_acc = np.zeros((nblk, n_curve))
        pts = np.empty((N, n_pts), dtype=np.float32)
        ln_h0 = math.log(h0_cm)
        for bi in nb.prange(nblk):
            i0 = bi * block
            i1 = min(N, i0 + block)
            for i in range(i0, i1):
                ni = n[i]
                lai = la[i]
                m = max(1.0 - 1.0 / ni, 1.0e-6)
                x0 = ni * (lai + h0_pf * LN10)
                log_se0 = m * _nb_logsig(-x0)
                log_1mu0 = _nb_logsig(x0)
                log_denom = math.log(max(-math.expm1(min(log_se0, -1.0e-7)), 1.0e-30))
                xa = -l10a[i]
                gap = max(ts[i] - tr[i], ts_gap)
                ratio = tr[i] / gap
                b = max(0.1 + (0.2 / (ni * ni)) * (1.0 - math.exp(-ratio * ratio)), 1e-6)
                sad_den = min(xa - h0_pf, -1.0e-6)
                om = min(max(omega[i], 1.0e-7), 1.0 - 1.0e-7)
                l1m_om = math.log1p(-om)
                l_om = math.log(om)
                ln_h0_ha = ln_h0 + lai
                wi = w[i]
                for j in range(P):
                    pfj = pf[j]
                    x = ni * (lai + pfj * LN10)
                    L = math.log1p(math.exp(-abs(x)))      # shared by both log-sigmoids
                    log_se = m * (min(-x, 0.0) - L)
                    if pfj >= h0_pf:
                        scap = 0.0
                        log_scap = LOG_TINY
                        sad = 0.0
                    else:
                        delta = min(log_se0 - log_se, -1.0e-7)
                        l1r = math.log(max(-math.expm1(delta), 1.0e-30))
                        log_scap = min(max(log_se + l1r - log_denom, LOG_TINY), 0.0)
                        scap = min(max(math.exp(log_scap), 0.0), 1.0)
                        sad = 1.0 + (pfj - xa + b * _nb_softplus((xa - pfj) / b)) / sad_den
                        sad = min(max(sad, 0.0), 1.0)
                    th = (ts[i] - tr[i]) * scap + tr[i] * sad
                    th = min(max(th, 0.0), 1.0)
                    if j < n_curve:
                        log_ratio = min(min(x, 0.0) - L - log_1mu0, 0.0)
                        z = min(m * log_ratio, -1.0e-7)
                        lb = max(math.log(max(-math.expm1(z), 1.0e-30)), LOG_TINY)
                        if pfj >= h0_pf:
                            lkc = LOG_TINY
                        else:
                            lkc = min(lam[i] * log_scap + 2.0 * lb, 0.0)
                        lkf = min(film_a * (1.0 - sad) * ln_h0_ha, 0.0)
                        a1 = l1m_om + lkc
                        a2 = l_om + lkf
                        # logaddexp(a1, a2) with a single exp
                        ln_rel = max(a1, a2) + math.log1p(math.exp(-abs(a1 - a2)))
                        ln_rel = min(ln_rel, 0.0)
                        lk = l10ks[i] + ln_rel / LN10
                        th_acc[bi, j] += wi * th
                        t2_acc[bi, j] += wi * th * th
                        lk_acc[bi, j] += wi * lk
                    else:
                        pts[i, j - n_curve] = th
        return th_acc.sum(axis=0), t2_acc.sum(axis=0), lk_acc.sum(axis=0), pts


# ===========================================================================
# Ensemble
# ===========================================================================
@dataclass
class Ensemble:
    family: str
    input_set: str
    features: List[str]
    activation: str
    layers: List[List[tuple]]          # per member: [(W, b), ...]
    scaler_min: np.ndarray
    scaler_scale: np.ndarray
    train_min: np.ndarray
    train_max: np.ndarray
    bounds: Dict
    physics: Dict
    meta: Dict = field(default_factory=dict)

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, root: Path, family: str, input_set: str) -> "Ensemble":
        run_dir = Path(root) / run_key(family, input_set)
        meta = json.loads((run_dir / "engine.json").read_text())
        z = np.load(run_dir / "weights.npz")
        layers = []
        for k in range(meta["n_members"]):
            layers.append([(z[f"m{k}_W{i}"].T.copy(), z[f"m{k}_b{i}"])
                           for i in range(meta["n_linear"])])
        sc = meta["scaler"]
        return cls(family=family, input_set=input_set,
                   features=list(meta["features"]), activation=meta["activation"],
                   layers=layers,
                   scaler_min=np.asarray(sc["min_"], np.float64),
                   scaler_scale=np.asarray(sc["scale_"], np.float64),
                   train_min=np.asarray(sc["data_min_"], np.float64),
                   train_max=np.asarray(sc["data_max_"], np.float64),
                   bounds=meta["bounds"], physics=meta["physics"], meta=meta)

    @property
    def n_members(self) -> int:
        return len(self.layers)

    @property
    def param_names(self) -> List[str]:
        if self.family == "PDI":
            return ["theta_r", "theta_s", "alpha", "n", "ksat", "omega", "lambda"]
        return ["theta_r", "theta_s", "alpha", "n", "ksat", "tau"]

    # ---------------------------------------------------------------- network
    def scale(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return (X * self.scaler_scale + self.scaler_min).astype(np.float32)

    def params(self, Xs: np.ndarray, k: int) -> Dict[str, np.ndarray]:
        """Network k on already-scaled inputs -> bounded hydraulic parameters,
        each (N,) float32.  Mirrors InverseVGMPTF / InversePDIPTF.forward."""
        h = Xs
        layers = self.layers[k]
        act = _ACT[self.activation]
        for W, b in layers[:-1]:
            h = act(h @ W + b)
        W, b = layers[-1]
        raw = (h @ W + b).astype(np.float32)
        s = _sigmoid(raw)
        B = self.bounds
        f32 = np.float32
        tr = f32(B["tr_lo"]) + f32(B["tr_hi"] - B["tr_lo"]) * s[:, 0]
        ts_lower = np.maximum(f32(B["ts_lo"]), tr + f32(B["ts_gap"]))
        ts = ts_lower + (f32(B["ts_hi"]) - ts_lower) * s[:, 1]
        l_alo, l_ahi = math.log10(B["a_lo"]), math.log10(B["a_hi"])
        l10a = f32(l_alo) + f32(l_ahi - l_alo) * s[:, 2]
        alpha = np.exp(l10a * f32(LN10))
        n = f32(B["n_lo"]) + f32(B["n_hi"] - B["n_lo"]) * s[:, 3]
        l_klo, l_khi = math.log10(B["k_lo"]), math.log10(B["k_hi"])
        l10k = f32(l_klo) + f32(l_khi - l_klo) * s[:, 4]
        out = dict(theta_r=tr, theta_s=ts, alpha=alpha, log10_alpha=l10a,
                   log_alpha=np.log(np.maximum(alpha, f32(EPS))),
                   n=n, m=1.0 - 1.0 / n, log10_ksat=l10k,
                   ksat=np.exp(l10k * f32(LN10)))
        if self.family == "PDI":
            out["omega"] = f32(B["om_lo"]) + f32(B["om_hi"] - B["om_lo"]) * s[:, 5]
            out["lambda"] = f32(B["lam_lo"]) + f32(B["lam_hi"] - B["lam_lo"]) * s[:, 6]
        else:
            out["tau"] = f32(B["tau_lo"]) + f32(B["tau_hi"] - B["tau_lo"]) * s[:, 5]
        return out

    # ---------------------------------------------------------------- physics
    def theta(self, p, pf):
        B = self.bounds
        q = {k: v[:, None] for k, v in p.items()}
        pf = np.asarray(pf, np.float32)[None, :]
        if self.family == "PDI":
            return pdi_theta(q, pf, B["h0_pf"], B["ts_gap"])
        if self.family == "VGDRY":
            return vgdry_theta(q, pf, B["pf_dry_ref"])
        return vg_theta(q, pf)

    def log10k(self, p, pf):
        B = self.bounds
        q = {k: v[:, None] for k, v in p.items()}
        pf = np.asarray(pf, np.float32)[None, :]
        if self.family == "PDI":
            return pdi_log10k(q, pf, B["h0_pf"], B["h0_cm"], B["film_a"], B["ts_gap"])
        return vgm_log10k(q, pf)

    # ---------------------------------------------------------------- small N
    def predict_members(self, X, pf_swrc, pf_kh):
        """Every member for a handful of soils (manual entry, one pixel).
        Returns theta (M,N,len(pf_swrc)), log10k (M,N,len(pf_kh)),
        params {name: (M,N)} -- same layout as ptf_core.predict_members."""
        Xs = self.scale(X)
        M = self.n_members
        th, lk, par = [], [], None
        for k in range(M):
            p = self.params(Xs, k)
            th.append(self.theta(p, pf_swrc))
            lk.append(self.log10k(p, pf_kh))
            if par is None:
                par = {n_: np.empty((M, len(Xs)), np.float32) for n_ in p}
            for n_, v in p.items():
                par[n_][k] = v
        return np.stack(th).astype(np.float32), np.stack(lk).astype(np.float32), par

    # ---------------------------------------------------------------- rasters
    def predict_area(self, X, weights, pf_curve, pf_points, progress=None,
                     use_numba: Optional[bool] = None, block: int = 256):
        """One fused pass over (unique) soils for a whole AOI.

        X         (U, F)  unique feature rows, in physical units
        weights   (U,)    pixels each unique row stands for
        pf_curve  pF values for the area-averaged curves (may be empty)
        pf_points pF values evaluated per row for maps (e.g. [FC, PWP])

        Returns dict:
          points_mean, points_sd (U, n_points)  ensemble mean / SD of theta
          paw_mean, paw_sd       (U,)           if two points (FC, PWP) given
          log10ks_mean           (U,)
          curve_theta, curve_log10k (M, n_curve)  weighted area mean per member
          curve_theta_var        (M, n_curve)   weighted variance across pixels
        """
        if use_numba is None:
            use_numba = HAS_NUMBA
        Xs = self.scale(X)
        U = len(Xs)
        M = self.n_members
        w = np.ascontiguousarray(weights, np.float64)
        wsum = float(w.sum())
        pf_curve = np.asarray(pf_curve, np.float64).ravel()
        pf_points = np.asarray(pf_points, np.float64).ravel()
        pf_all = np.concatenate([pf_curve, pf_points])
        nC, nP = len(pf_curve), len(pf_points)

        pts_sum = np.zeros((U, nP), np.float64)
        pts_sq = np.zeros((U, nP), np.float64)
        paw_sq = np.zeros(U, np.float64)
        lks_sum = np.zeros(U, np.float64)
        curve_th = np.zeros((M, nC))
        curve_t2 = np.zeros((M, nC))
        curve_lk = np.zeros((M, nC))
        B = self.bounds
        f64 = lambda a: np.ascontiguousarray(a, np.float64)

        for k in range(M):
            p = self.params(Xs, k)
            if use_numba:
                if self.family == "PDI":
                    th_s, t2_s, lk_s, pts = _nb_pdi(
                        f64(p["theta_r"]), f64(p["theta_s"]), f64(p["log_alpha"]),
                        f64(p["log10_alpha"]), f64(p["n"]), f64(p["log10_ksat"]),
                        f64(p["omega"]), f64(p["lambda"]), w, pf_all, nC,
                        float(B["h0_pf"]), float(B["h0_cm"]), float(B["film_a"]),
                        float(B["ts_gap"]), block)
                else:
                    th_s, t2_s, lk_s, pts = _nb_vg_family(
                        f64(p["theta_r"]), f64(p["theta_s"]), f64(p["log_alpha"]),
                        f64(p["n"]), f64(p["log10_ksat"]), f64(p["tau"]), w, pf_all,
                        nC, self.family == "VGDRY",
                        float(B.get("pf_dry_ref", 6.8)), block)
            else:
                th_s, t2_s, lk_s = np.zeros(nC), np.zeros(nC), np.zeros(nC)
                pts = np.empty((U, nP), np.float32)
                step = 8192
                for s0 in range(0, U, step):
                    sl = slice(s0, s0 + step)
                    q = {kk: v[sl] for kk, v in p.items()}
                    th = self.theta(q, pf_all).astype(np.float64)
                    pts[sl] = th[:, nC:]
                    if nC:
                        lk = self.log10k(q, pf_curve)
                        ws = w[sl, None]
                        th_s += (ws * th[:, :nC]).sum(0)
                        t2_s += (ws * th[:, :nC] ** 2).sum(0)
                        lk_s += (ws * lk).sum(0)
            if nC:
                curve_th[k] = th_s / wsum
                curve_t2[k] = t2_s / wsum
                curve_lk[k] = lk_s / wsum
            pts = pts.astype(np.float64)
            pts_sum += pts
            pts_sq += pts * pts
            if nP >= 2:
                d = pts[:, 0] - pts[:, 1]
                paw_sq += d * d
            lks_sum += p["log10_ksat"]
            if progress is not None:
                progress((k + 1) / M)

        corr = M / max(M - 1, 1)            # sample SD across members (ddof=1)
        mean_pts = pts_sum / M
        out = dict(points_mean=mean_pts.astype(np.float32),
                   points_sd=np.sqrt(np.maximum(pts_sq / M - mean_pts ** 2, 0) * corr
                                     ).astype(np.float32),
                   log10ks_mean=(lks_sum / M).astype(np.float32),
                   curve_theta=curve_th.astype(np.float32),
                   curve_theta_var=np.maximum(curve_t2 - curve_th ** 2, 0),
                   curve_log10k=curve_lk.astype(np.float32))
        if nP >= 2:
            paw_mean = mean_pts[:, 0] - mean_pts[:, 1]
            out["paw_mean"] = paw_mean.astype(np.float32)
            out["paw_sd"] = np.sqrt(np.maximum(paw_sq / M - paw_mean ** 2, 0) * corr
                                    ).astype(np.float32)
        return out


# ===========================================================================
# Helpers shared by the app
# ===========================================================================
def normalize_texture(sand, silt, clay, total: float = 100.0):
    """Rescale sand/silt/clay to sum to `total` (ratios preserved).  Rows that
    sum to zero or are non-finite come back NaN.  Same as ptf_core."""
    sand, silt, clay = (np.asarray(a) for a in (sand, silt, clay))
    dt = np.result_type(sand.dtype, silt.dtype, clay.dtype, np.float32)
    ssum = sand.astype(dt) + silt + clay
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(np.isfinite(ssum) & (ssum > 0), dt.type(total) / ssum, dt.type(np.nan))
    return sand * f, silt * f, clay * f


def ensemble_band(stack: np.ndarray, z: float = 1.0, ddof: int = 1):
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=ddof if stack.shape[0] > 1 else 0)
    return mean, mean - z * sd, mean + z * sd, sd


def texture_class(sand: float, silt: float, clay: float) -> str:
    s, si, c = sand, silt, clay
    if c >= 40 and si < 40 and s <= 45:                        return "Clay"
    if c >= 35 and s >= 45:                                    return "Sandy clay"
    if c >= 40 and si >= 40:                                   return "Silty clay"
    if 27 <= c < 40 and 20 < s <= 45:                          return "Clay loam"
    if 27 <= c < 40 and s <= 20:                               return "Silty clay loam"
    if 20 <= c < 35 and si < 28 and s > 45:                    return "Sandy clay loam"
    if si >= 80 and c < 12:                                    return "Silt"
    if si >= 50 and 12 <= c < 27:                              return "Silt loam"
    if 50 <= si < 80 and c < 12:                               return "Silt loam"
    if 7 <= c < 27 and 28 <= si < 50 and s <= 52:              return "Loam"
    if s > 85 and (si + 1.5 * c) < 15:                         return "Sand"
    if s > 70 and (si + 1.5 * c) >= 15 and (si + 2 * c) < 30:  return "Loamy sand"
    return "Sandy loam"
