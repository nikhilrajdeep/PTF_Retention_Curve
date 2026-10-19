"""
ptf_core.py
===========
Shared inference core for the 12 physics-constrained inverse-PTF models
(3 model families x 4 input sets).

This module contains ONLY what is needed for inference:

  * the three network architectures, byte-identical to the training notebooks
    (PDI_*_Final.ipynb, VGM_*_Final_new.ipynb,
     VG_DryCorrected_*_Final_CurveUIDPrior_Tau.ipynb),
  * the three physics forward models (retention + conductivity),
  * a registry that resolves (model family, input set) -> bundle folder,
  * the ensemble loader and the deep-ensemble / bootstrap predictor.

Both build_model_bundle.py and streamlit_ptf_app2.py import from here, so the
physics exists in exactly one place.

Author: Nikhil Raj Deep
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-12
LOG_TINY = -60.0
DEVICE = torch.device("cpu")          # inference on CPU is plenty fast


# ===========================================================================
# 1. REGISTRY -- the 12 models
# ===========================================================================

MODEL_FAMILIES = {
    # key          label shown in the app                 checkpoint prefix
    "VGM":   dict(label="Classic VG (van Genuchten-Mualem)", prefix="vgm"),
    "VGDRY": dict(label="Enhanced VG (dry-end corrected)",   prefix="vgdry"),
    "PDI":   dict(label="PDI (Peters-Durner-Iden)",          prefix="pdi"),
}

INPUT_SETS = {
    "SSC":     "Sand, Silt, Clay",
    "SSCBD":   "Sand, Silt, Clay, Bulk density",
    "SSCOC":   "Sand, Silt, Clay, Organic carbon",
    "SSCBDOC": "Sand, Silt, Clay, Bulk density, Organic carbon",
}

# Folder names as produced by the training notebooks, inside Final_PTF.
SOURCE_FOLDER = {
    ("PDI",   "SSC"):     "PDI_DNN_SSC",
    ("PDI",   "SSCBD"):   "PDI_DNN_SSCBD",
    ("PDI",   "SSCOC"):   "PDI_DNN_SSCOC",
    ("PDI",   "SSCBDOC"): "PDI_DNN_SSCBDOC",
    ("VGM",   "SSC"):     "VGM_DNN_SSC_new",
    ("VGM",   "SSCBD"):   "VGM_DNN_SSCBD_new",
    ("VGM",   "SSCOC"):   "VGM_DNN_SSCOC_new",
    ("VGM",   "SSCBDOC"): "VGM_DNN_SSCBDOC_new",
    ("VGDRY", "SSC"):     "VG_DryCorrected_DNN_SSC",
    ("VGDRY", "SSCBD"):   "VG_DryCorrected_DNN_SSCBD",
    ("VGDRY", "SSCOC"):   "VG_DryCorrected_DNN_SSCOC",
    ("VGDRY", "SSCBDOC"): "VG_DryCorrected_DNN_SSCBDOC",
}

# Folder name used inside the self-contained bundle.
def bundle_key(family: str, input_set: str) -> str:
    return f"{family}_{input_set}"


# Canonical feature names used by the notebooks.  The exact order is always
# read back from model_config.json, never hard-coded here.
FEATURE_LABELS = {
    "Sand_pct":      ("Sand (%)",                 0.0, 100.0, 92.0, 0.1),
    "Silt_pct":      ("Silt (%)",                 0.0, 100.0,  5.0, 0.1),
    "Clay_pct":      ("Clay (%)",                 0.0, 100.0,  3.0, 0.1),
    "BD_g_cm3":      ("Bulk density (g cm-3)",    0.5,   2.2,  1.50, 0.01),
    "OC_pct":        ("Organic carbon (%)",       0.0,  30.0,  1.0, 0.1),
    "Depth_harm_cm": ("Depth (cm)",               0.0, 250.0,  5.0, 1.0),
}


# ===========================================================================
# 2. NETWORKS -- identical to the training notebooks
# ===========================================================================

_ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "sigmoid": nn.Sigmoid}


class _Encoder(nn.Module):
    """Shared trunk.  Layer ordering (Linear, act, [Dropout]) must match the
    notebooks exactly, because Dropout shifts nn.Sequential indices and hence
    the state_dict keys."""

    def __init__(self, n_features: int, hidden: int, layers: int,
                 activation: str, dropout: float):
        super().__init__()
        act = _ACTS[activation]
        mods, previous = [], n_features
        for _ in range(layers):
            mods += [nn.Linear(previous, hidden), act()]
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            previous = hidden
        self.encoder = nn.Sequential(*mods)
        self.out_width = previous


class InverseVGMPTF(_Encoder):
    """Classic constrained VG (1980) retention + VG-Mualem K(h).
    Head outputs: theta_r, theta_s, log10(alpha), n, log10(Ksat), tau."""

    N_OUT = 6

    def __init__(self, n_features, hidden, layers, activation, dropout, b):
        super().__init__(n_features, hidden, layers, activation, dropout)
        self.b = b
        self.head = nn.Linear(self.out_width, self.N_OUT)

    def forward(self, x):
        b = self.b
        raw = self.head(self.encoder(x))

        theta_r = b["tr_lo"] + (b["tr_hi"] - b["tr_lo"]) * torch.sigmoid(raw[:, 0:1])
        ts_lower = torch.maximum(torch.full_like(theta_r, b["ts_lo"]),
                                 theta_r + b["ts_gap"])
        theta_s = ts_lower + (b["ts_hi"] - ts_lower) * torch.sigmoid(raw[:, 1:2])

        log10_alpha = (math.log10(b["a_lo"])
                       + (math.log10(b["a_hi"]) - math.log10(b["a_lo"]))
                       * torch.sigmoid(raw[:, 2:3]))
        alpha = torch.exp(log10_alpha * math.log(10.0))

        n = b["n_lo"] + (b["n_hi"] - b["n_lo"]) * torch.sigmoid(raw[:, 3:4])
        m = 1.0 - 1.0 / n

        log10_ksat = (math.log10(b["k_lo"])
                      + (math.log10(b["k_hi"]) - math.log10(b["k_lo"]))
                      * torch.sigmoid(raw[:, 4:5]))
        ksat = torch.exp(log10_ksat * math.log(10.0))

        tau = b["tau_lo"] + (b["tau_hi"] - b["tau_lo"]) * torch.sigmoid(raw[:, 5:6])

        return dict(theta_r=theta_r, theta_s=theta_s, alpha=alpha,
                    log10_alpha=log10_alpha, n=n, m=m,
                    log10_ksat=log10_ksat, ksat=ksat, tau=tau)


class InverseVGDryPTF(InverseVGMPTF):
    """Same head/parameterisation as VGM; only the retention equation differs
    (dry-end correction).  Architecture is identical, so the class is reused."""
    pass


class InversePDIPTF(_Encoder):
    """Complete VG-PDI retention + complete PDI (capillary + film) K(h).
    Head outputs: theta_r, theta_s, log10(alpha), n, log10(Ksat), omega, lambda."""

    N_OUT = 7

    def __init__(self, n_features, hidden, layers, activation, dropout, b):
        super().__init__(n_features, hidden, layers, activation, dropout)
        self.b = b
        self.head = nn.Linear(self.out_width, self.N_OUT)

    def forward(self, x):
        b = self.b
        raw = self.head(self.encoder(x))

        theta_r = b["tr_lo"] + (b["tr_hi"] - b["tr_lo"]) * torch.sigmoid(raw[:, 0:1])
        ts_lower = torch.maximum(torch.full_like(theta_r, b["ts_lo"]),
                                 theta_r + b["ts_gap"])
        theta_s = ts_lower + (b["ts_hi"] - ts_lower) * torch.sigmoid(raw[:, 1:2])

        log10_alpha = (math.log10(b["a_lo"])
                       + (math.log10(b["a_hi"]) - math.log10(b["a_lo"]))
                       * torch.sigmoid(raw[:, 2:3]))
        alpha = torch.exp(log10_alpha * math.log(10.0))

        n = b["n_lo"] + (b["n_hi"] - b["n_lo"]) * torch.sigmoid(raw[:, 3:4])
        m = 1.0 - 1.0 / n

        log10_ksat = (math.log10(b["k_lo"])
                      + (math.log10(b["k_hi"]) - math.log10(b["k_lo"]))
                      * torch.sigmoid(raw[:, 4:5]))
        ksat = torch.exp(log10_ksat * math.log(10.0))

        omega = b["om_lo"] + (b["om_hi"] - b["om_lo"]) * torch.sigmoid(raw[:, 5:6])
        lam = b["lam_lo"] + (b["lam_hi"] - b["lam_lo"]) * torch.sigmoid(raw[:, 6:7])

        return dict(theta_r=theta_r, theta_s=theta_s, alpha=alpha,
                    log10_alpha=log10_alpha, n=n, m=m,
                    log10_ksat=log10_ksat, ksat=ksat,
                    omega=omega, **{"lambda": lam})


# ===========================================================================
# 3. PHYSICS -- log-space, copied verbatim from the notebooks
# ===========================================================================

def _vg_log_components(alpha, n, pf):
    """Return log(Se), log(1 - Se^(1/m)), m  for h = 10^pF cm, m = 1 - 1/n."""
    log_alpha_h = torch.log(alpha.clamp_min(EPS)) + pf * math.log(10.0)
    x = n * log_alpha_h
    log_u = F.logsigmoid(-x)                 # log Se^(1/m)
    log_one_minus_u = F.logsigmoid(x)
    m = (1.0 - 1.0 / n).clamp_min(1.0e-6)
    return m * log_u, log_one_minus_u, m


# ---------------------------- classic VG -----------------------------------

def vg_theta(p, pf):
    log_se, _, _ = _vg_log_components(p["alpha"], p["n"], pf)
    se = torch.exp(log_se).clamp(EPS, 1.0)
    return (p["theta_r"] + (p["theta_s"] - p["theta_r"]) * se).clamp(0.0, 1.0)


def vgm_log10k(p, pf):
    """K = Ks * Se^tau * [1 - (1-Se^(1/m))^m]^2"""
    log_se, log_one_minus_u, m = _vg_log_components(p["alpha"], p["n"], pf)
    z = (m * log_one_minus_u).clamp_max(-1.0e-12)
    log_bracket = torch.log((-torch.expm1(z)).clamp_min(1.0e-30))
    ln_krel = p["tau"] * log_se + 2.0 * log_bracket
    return p["log10_ksat"] + ln_krel / math.log(10.0)


# ------------------------ dry-end corrected VG -----------------------------

def vgdry_theta(p, pf, pf_dry_ref=6.8):
    """theta = (theta_s-theta_r)*Se + theta_r*Fdry,
       Fdry = 1 - ln(1+(alpha h)^c) / ln(1+(alpha 10^pf_dry_ref)^c),  c = 1+0.1/n"""
    log_se, _, _ = _vg_log_components(p["alpha"], p["n"], pf)
    se = torch.exp(log_se)
    alpha, n = p["alpha"], p["n"]
    log_alpha_h = torch.log(alpha.clamp_min(EPS)) + pf * math.log(10.0)
    c = 1.0 + 0.1 / n
    numerator = F.softplus(c * log_alpha_h)
    log_alpha_h_dry = torch.log(alpha.clamp_min(EPS)) + pf_dry_ref * math.log(10.0)
    denominator = F.softplus(c * log_alpha_h_dry).clamp_min(EPS)
    f_dry = (1.0 - numerator / denominator).clamp(0.0, 1.0)
    theta = (p["theta_s"] - p["theta_r"]) * se + p["theta_r"] * f_dry
    return theta.clamp(0.0, 1.0)


# ------------------------------- PDI ---------------------------------------

def _pdi_saturation_components(p, pf, h0_pf, ts_gap):
    alpha, n = p["alpha"], p["n"]
    theta_r, theta_s = p["theta_r"], p["theta_s"]

    log_se, _, _ = _vg_log_components(alpha, n, pf)
    pf0 = torch.full_like(pf, h0_pf)
    log_se0, _, _ = _vg_log_components(alpha, n, pf0)

    delta = (log_se0 - log_se).clamp_max(-1.0e-7)
    log_one_minus_ratio = torch.log((-torch.expm1(delta)).clamp_min(1.0e-30))
    log_denom = torch.log((-torch.expm1(log_se0.clamp_max(-1.0e-7))).clamp_min(1.0e-30))
    log_scap = (log_se + log_one_minus_ratio - log_denom).clamp(min=LOG_TINY, max=0.0)
    scap = torch.exp(log_scap).clamp(0.0, 1.0)

    xa = -p["log10_alpha"]                       # ha = 1/alpha  ->  xa = -log10(alpha)
    x0 = torch.full_like(pf, h0_pf)
    capillary_gap = (theta_s - theta_r).clamp_min(ts_gap)
    ratio_theta = theta_r / capillary_gap
    b = (0.1 + (0.2 / n.pow(2)) * (1.0 - torch.exp(-ratio_theta.pow(2)))).clamp_min(1e-6)
    sad = 1.0 + (pf - xa + b * F.softplus((xa - pf) / b)) / (xa - x0).clamp_max(-1.0e-6)
    sad = sad.clamp(0.0, 1.0)

    dry = pf >= h0_pf
    scap = torch.where(dry, torch.zeros_like(scap), scap)
    log_scap = torch.where(dry, torch.full_like(log_scap, LOG_TINY), log_scap)
    sad = torch.where(dry, torch.zeros_like(sad), sad)
    return log_se, scap, log_scap, sad, b


def pdi_theta(p, pf, h0_pf=6.8, ts_gap=0.01):
    _, scap, _, sad, _ = _pdi_saturation_components(p, pf, h0_pf, ts_gap)
    theta = (p["theta_s"] - p["theta_r"]) * scap + p["theta_r"] * sad
    return theta.clamp(0.0, 1.0)


def pdi_log10k(p, pf, h0_pf=6.8, h0_cm=None, film_a=-1.5, ts_gap=0.01):
    """K = Ks[(1-omega) Krel_cap + omega Krel_film], entirely in log space."""
    if h0_cm is None:
        h0_cm = 10.0 ** h0_pf
    alpha, n = p["alpha"], p["n"]
    lam = p["lambda"]
    omega = p["omega"].clamp(1.0e-7, 1.0 - 1.0e-7)

    _, _, log_scap, sad, _ = _pdi_saturation_components(p, pf, h0_pf, ts_gap)
    _, log_one_minus_u, m = _vg_log_components(alpha, n, pf)
    pf0 = torch.full_like(pf, h0_pf)
    _, log_one_minus_u0, _ = _vg_log_components(alpha, n, pf0)

    log_ratio = (log_one_minus_u - log_one_minus_u0).clamp_max(0.0)
    z = (m * log_ratio).clamp_max(-1.0e-7)
    log_bracket = torch.log((-torch.expm1(z)).clamp_min(1.0e-30)).clamp_min(LOG_TINY)

    ln_kcap = lam * log_scap + 2.0 * log_bracket
    ln_kcap = torch.minimum(ln_kcap, torch.zeros_like(ln_kcap))
    ln_kcap = torch.where(pf >= h0_pf, torch.full_like(ln_kcap, LOG_TINY), ln_kcap)

    ln_h0_over_ha = math.log(h0_cm) + torch.log(alpha.clamp_min(EPS))
    ln_kfilm = film_a * (1.0 - sad) * ln_h0_over_ha
    ln_kfilm = torch.minimum(ln_kfilm, torch.zeros_like(ln_kfilm))

    ln_rel = torch.logaddexp(torch.log1p(-omega) + ln_kcap,
                             torch.log(omega) + ln_kfilm)
    ln_rel = torch.minimum(ln_rel, torch.zeros_like(ln_rel))
    return p["log10_ksat"] + ln_rel / math.log(10.0)


# ===========================================================================
# 4. BUNDLE LOADING
# ===========================================================================

@dataclass
class LoadedModel:
    family: str
    input_set: str
    features: List[str]
    models: List[nn.Module]
    scaler: object
    cfg: Dict
    config: Dict            # full model_config.json
    bounds: Dict

    # ---- physics dispatch -------------------------------------------------
    def theta(self, params, pf):
        if self.family == "PDI":
            return pdi_theta(params, pf, self.bounds["h0_pf"], self.bounds["ts_gap"])
        if self.family == "VGDRY":
            return vgdry_theta(params, pf, self.bounds["pf_dry_ref"])
        return vg_theta(params, pf)

    def log10k(self, params, pf):
        if self.family == "PDI":
            return pdi_log10k(params, pf, self.bounds["h0_pf"],
                              self.bounds["h0_cm"], self.bounds["film_a"],
                              self.bounds["ts_gap"])
        return vgm_log10k(params, pf)

    @property
    def param_names(self) -> List[str]:
        if self.family == "PDI":
            return ["theta_r", "theta_s", "alpha", "n", "ksat", "omega", "lambda"]
        return ["theta_r", "theta_s", "alpha", "n", "ksat", "tau"]


def _bounds_from_config(family: str, config: Dict) -> Dict:
    ph = config["physics"]
    b = dict(
        tr_lo=ph["theta_r_bounds"][0], tr_hi=ph["theta_r_bounds"][1],
        ts_lo=ph["theta_s_bounds"][0], ts_hi=ph["theta_s_bounds"][1],
        ts_gap=ph["theta_s_minus_theta_r_min"],
        a_lo=ph["alpha_bounds_1_per_cm"][0], a_hi=ph["alpha_bounds_1_per_cm"][1],
        n_lo=ph["n_bounds"][0], n_hi=ph["n_bounds"][1],
        k_lo=ph["Ksat_bounds_cm_d"][0], k_hi=ph["Ksat_bounds_cm_d"][1],
    )
    if family == "PDI":
        b.update(om_lo=ph["omega_bounds"][0], om_hi=ph["omega_bounds"][1],
                 lam_lo=ph["lambda_bounds"][0], lam_hi=ph["lambda_bounds"][1],
                 h0_pf=ph.get("h0_pf", 6.8),
                 h0_cm=ph.get("h0_cm", 10.0 ** ph.get("h0_pf", 6.8)),
                 film_a=ph.get("film_a", -1.5))
    else:
        b.update(tau_lo=ph["tau_bounds"][0], tau_hi=ph["tau_bounds"][1])
    if family == "VGDRY":
        b["pf_dry_ref"] = ph.get("dry_correction", {}).get("dry_reference_pF", 6.8)
    return b


def resolve_run_dir(root: Path, family: str, input_set: str) -> Path:
    """Works both for the self-contained bundle (PDI_SSC/) and for a raw
    Final_PTF folder (PDI_DNN_SSC/)."""
    root = Path(root)
    p = root / bundle_key(family, input_set)
    if (p / "ensemble_models").exists():
        return p
    p = root / SOURCE_FOLDER[(family, input_set)]
    if (p / "ensemble_models").exists():
        return p
    raise FileNotFoundError(
        f"No trained artifacts for {family}/{input_set} under {root}. "
        f"Looked for {bundle_key(family, input_set)}/ and "
        f"{SOURCE_FOLDER[(family, input_set)]}/.")


def load_model(root: Path, family: str, input_set: str) -> LoadedModel:
    """Read-only loader.  Never trains."""
    run_dir = resolve_run_dir(root, family, input_set)
    prefix = MODEL_FAMILIES[family]["prefix"]

    files = sorted((run_dir / "ensemble_models").glob(f"{prefix}_member_*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No {prefix}_member_*.pt checkpoints in {run_dir/'ensemble_models'}.")

    with open(run_dir / "model_config.json") as f:
        config = json.load(f)
    features = list(config["features"])

    hp = run_dir / "best_hyperparameters.json"
    cfg = json.load(open(hp))["config"] if hp.exists() else config["best_config"]

    scaler = joblib.load(run_dir / "input_scaler.joblib")
    bounds = _bounds_from_config(family, config)

    cls = {"PDI": InversePDIPTF, "VGM": InverseVGMPTF,
           "VGDRY": InverseVGDryPTF}[family]

    models = []
    for file in files:
        net = cls(len(features), int(cfg["hidden"]), int(cfg["layers"]),
                  str(cfg["activation"]), float(cfg["dropout"]), bounds).to(DEVICE)
        ckpt = torch.load(file, map_location=DEVICE)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        net.load_state_dict(state)
        net.eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        models.append(net)

    return LoadedModel(family=family, input_set=input_set, features=features,
                       models=models, scaler=scaler, cfg=cfg,
                       config=config, bounds=bounds)


# ===========================================================================
# 5. PREDICTION
# ===========================================================================

def pf_grid(lo: float, hi: float, step: float) -> np.ndarray:
    return np.round(np.arange(lo, hi + 0.5 * step, step), 5)


@torch.no_grad()
def predict_members(lm: LoadedModel, features_2d: np.ndarray,
                    pf_swrc: np.ndarray, pf_kh: np.ndarray,
                    batch: int = 100_000):
    """
    Run every ensemble member.

    Returns
    -------
    theta     : (M, N, len(pf_swrc))          theta on the SWRC grid
    log10k    : (M, N, len(pf_kh))            log10 K on the K(h) grid
    theta_kh  : (M, N, len(pf_kh))            theta on the K(h) grid, so K can
                                              be plotted against theta
    params    : dict name -> (M, N) array     (log10_ksat included)
    """
    x_np = lm.scaler.transform(np.asarray(features_2d, dtype=np.float64)).astype(np.float32)
    n_rows = len(x_np)
    M = len(lm.models)

    theta = np.empty((M, n_rows, len(pf_swrc)), np.float32)
    logk = np.empty((M, n_rows, len(pf_kh)), np.float32)
    theta_kh = np.empty((M, n_rows, len(pf_kh)), np.float32)
    keys = None
    params = None

    t_sw = torch.tensor(pf_swrc, dtype=torch.float32, device=DEVICE)
    t_kh = torch.tensor(pf_kh, dtype=torch.float32, device=DEVICE)

    for mi, net in enumerate(lm.models):
        for s in range(0, n_rows, batch):
            xb = torch.from_numpy(x_np[s:s + batch]).to(DEVICE)
            p = net(xb)
            nb = xb.shape[0]
            theta[mi, s:s + nb] = lm.theta(p, t_sw.view(1, -1).expand(nb, -1)).cpu().numpy()
            logk[mi, s:s + nb] = lm.log10k(p, t_kh.view(1, -1).expand(nb, -1)).cpu().numpy()
            if keys is None:
                keys = list(p.keys())
                params = {k: np.empty((M, n_rows), np.float32) for k in keys}
            for k in keys:
                params[k][mi, s:s + nb] = p[k].cpu().numpy().ravel()
    return theta, logk, params


@torch.no_grad()
def predict_mean_curves(lm: "LoadedModel", features_2d: np.ndarray,
                        pf_swrc: np.ndarray, pf_kh: np.ndarray,
                        chunk: int = 20_000, progress=None):
    """Area-mean curves over EVERY row, without ever holding all of them.

    predict_members returns (M, N, n_pF), which for a whole county is several
    GB.  Here the per-pixel curves are accumulated into a running sum in
    chunks, so memory stays at one chunk and every valid pixel still counts
    towards the mean.

    When pf_swrc and pf_kh are the same grid the retention curve is evaluated
    once and reused, halving the work.

    Returns (theta_swrc, theta_kh, logk) each of shape (M, n_pF): the mean
    over rows, still resolved by ensemble member.
    """
    x_np = lm.scaler.transform(np.asarray(features_2d, dtype=np.float64)).astype(np.float32)
    n_rows = len(x_np)
    M = len(lm.models)
    same_grid = (len(pf_swrc) == len(pf_kh)) and bool(np.allclose(pf_swrc, pf_kh))

    t_sw = torch.tensor(pf_swrc, dtype=torch.float32, device=DEVICE)
    t_kh = torch.tensor(pf_kh, dtype=torch.float32, device=DEVICE)

    sum_sw = np.zeros((M, len(pf_swrc)), np.float64)
    sum_kh = np.zeros((M, len(pf_kh)), np.float64)
    sum_lk = np.zeros((M, len(pf_kh)), np.float64)

    n_chunks = max(1, (n_rows + chunk - 1) // chunk)
    for ci, s0 in enumerate(range(0, n_rows, chunk)):
        xb = torch.from_numpy(x_np[s0:s0 + chunk]).to(DEVICE)
        nb = xb.shape[0]
        for mi, net in enumerate(lm.models):
            p = net(xb)
            th_sw = lm.theta(p, t_sw.view(1, -1).expand(nb, -1))
            sum_sw[mi] += th_sw.sum(dim=0).double().cpu().numpy()
            th_kh = th_sw if same_grid else lm.theta(p, t_kh.view(1, -1).expand(nb, -1))
            sum_kh[mi] += th_kh.sum(dim=0).double().cpu().numpy()
            sum_lk[mi] += lm.log10k(p, t_kh.view(1, -1).expand(nb, -1)
                                    ).sum(dim=0).double().cpu().numpy()
        if progress is not None:
            progress((ci + 1) / n_chunks)

    return (sum_sw / n_rows).astype(np.float32), \
           (sum_kh / n_rows).astype(np.float32), \
           (sum_lk / n_rows).astype(np.float32)


def normalize_texture(sand, silt, clay, total: float = 100.0):
    """Rescale sand/silt/clay so they sum to `total`, elementwise.

    SOLUS fractions do not form a closed composition (they sum to ~81% over
    north Florida, and sand alone can exceed 100), while the models were
    trained on fractions that do sum to 100.  Renormalising puts the inputs
    back on the training manifold and preserves the ratios between the three.

    Rows whose sum is zero or non-finite come back as NaN so they are dropped
    as invalid rather than silently becoming something else.
    """
    sand = np.asarray(sand)
    silt = np.asarray(silt)
    clay = np.asarray(clay)
    # Stay in the input dtype: on a full SOLUS grid (67M pixels) an
    # upcast to float64 costs ~0.5 GB per temporary.
    dt = np.result_type(sand.dtype, silt.dtype, clay.dtype, np.float32)
    ssum = sand.astype(dt, copy=True)
    ssum += silt
    ssum += clay
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(np.isfinite(ssum) & (ssum > 0), dt.type(total) / ssum,
                     dt.type(np.nan))
    del ssum
    return sand * f, silt * f, clay * f


def ensemble_band(stack: np.ndarray, z: float = 1.0, ddof: int = 1):
    """Deep-ensemble mean +/- z*SD across members.  stack: (M, ...)."""
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=ddof if stack.shape[0] > 1 else 0)
    return mean, mean - z * sd, mean + z * sd, sd


def bootstrap_band(stack: np.ndarray, n_boot: int = 1000,
                   lo_q: float = 2.5, hi_q: float = 97.5, seed: int = 0):
    """
    Nonparametric bootstrap over ensemble members.

    The 10 trained members are resampled WITH replacement n_boot times; each
    replicate's mean curve is recorded and the lo/hi percentiles of those
    replicate means form the shaded band.  This is the curve-level analogue of
    the cluster bootstrap used for the test metrics in the notebooks.

    stack : (M, ...)  ->  returns (mean, lo, hi)
    """
    rng = np.random.default_rng(seed)
    M = stack.shape[0]
    idx = rng.integers(0, M, size=(n_boot, M))
    reps = stack[idx].mean(axis=1)          # (n_boot, ...)
    mean = stack.mean(axis=0)
    lo = np.percentile(reps, lo_q, axis=0)
    hi = np.percentile(reps, hi_q, axis=0)
    return mean, lo, hi


def texture_class(sand: float, silt: float, clay: float) -> str:
    s, si, c = sand, silt, clay
    if c >= 40 and si < 40 and s <= 45:                      return "Clay"
    if c >= 35 and s >= 45:                                  return "Sandy clay"
    if c >= 40 and si >= 40:                                 return "Silty clay"
    if 27 <= c < 40 and 20 < s <= 45:                        return "Clay loam"
    if 27 <= c < 40 and s <= 20:                             return "Silty clay loam"
    if 20 <= c < 35 and si < 28 and s > 45:                  return "Sandy clay loam"
    if si >= 80 and c < 12:                                  return "Silt"
    if si >= 50 and 12 <= c < 27:                            return "Silt loam"
    if 50 <= si < 80 and c < 12:                             return "Silt loam"
    if 7 <= c < 27 and 28 <= si < 50 and s <= 52:            return "Loam"
    if s > 85 and (si + 1.5 * c) < 15:                       return "Sand"
    if s > 70 and (si + 1.5 * c) >= 15 and (si + 2 * c) < 30: return "Loamy sand"
    return "Sandy loam"
