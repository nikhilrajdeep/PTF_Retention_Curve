"""
Equivalence tests: the torch-free engine must reproduce the original
PyTorch inference code (legacy/ptf_core.py) for all 12 ensembles.

    pytest -q tests/test_engine.py            (needs torch + scikit-learn)

Skipped automatically when torch is not installed, so the deployed app does
not need it.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))
BUNDLE = ROOT / "models" / "ptf_model_bundle"

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
warnings.filterwarnings("ignore")

import ptf_core as legacy                       # noqa: E402
from ptf_engine import core                     # noqa: E402

RUNS = [(f, s) for f in ("VGM", "VGDRY", "PDI") for s in ("SSC", "SSCBD", "SSCOC", "SSCBDOC")]
PF = core.pf_grid(-2.0, 7.0, 0.05)


def random_soils(features, n=400, seed=0):
    rng = np.random.default_rng(seed)
    tex = rng.dirichlet([2.0, 1.0, 1.0], size=n) * 100
    cols = {"Sand_pct": tex[:, 0], "Silt_pct": tex[:, 1], "Clay_pct": tex[:, 2],
            "BD_g_cm3": rng.uniform(0.8, 1.8, n), "OC_pct": rng.uniform(0.05, 8, n),
            "Depth_harm_cm": rng.choice([5, 15, 30, 60, 100], n).astype(float)}
    return np.column_stack([cols[f] for f in features])


@pytest.mark.parametrize("family,input_set", RUNS)
def test_members_match_torch(family, input_set):
    lm = legacy.load_model(BUNDLE, family, input_set)
    en = core.Ensemble.load(BUNDLE, family, input_set)
    assert en.features == lm.features
    X = random_soils(en.features)
    th_t, lk_t, par_t = legacy.predict_members(lm, X, PF, PF)
    th_n, lk_n, par_n = en.predict_members(X, PF, PF)
    assert np.abs(th_t - th_n).max() < 2e-4
    for name in ("theta_r", "theta_s", "n", "log10_ksat"):
        assert np.abs(par_t[name] - par_n[name]).max() < 1e-4
    # log10 K: compare where it is plotted (>= -10) -- deep-dry tails are
    # float32 clamps (1e-30) in both codes and carry no information.
    vis = lk_t > -10
    assert np.abs(lk_t - lk_n)[vis].max() < 5e-3


@pytest.mark.parametrize("family,input_set", RUNS)
@pytest.mark.parametrize("use_numba", [True, False])
def test_area_path_matches_torch(family, input_set, use_numba):
    if use_numba and not core.HAS_NUMBA:
        pytest.skip("numba not installed")
    lm = legacy.load_model(BUNDLE, family, input_set)
    en = core.Ensemble.load(BUNDLE, family, input_set)
    X = random_soils(en.features, n=300, seed=1)
    # duplicate rows with counts: the weighted unique-row path must equal
    # the plain every-row mean.
    counts = np.random.default_rng(2).integers(1, 6, len(X))
    X_full = np.repeat(X, counts, axis=0)
    th_t, _, lk_t = legacy.predict_mean_curves(lm, X_full, PF, PF)
    pts_t, _, par_t = legacy.predict_members(lm, X, np.array([2.0, 4.2]), np.array([0.0]))

    out = en.predict_area(X, counts, PF, [2.0, 4.2], use_numba=use_numba)
    assert np.abs(out["curve_theta"] - th_t).max() < 2e-4
    assert np.abs(out["curve_log10k"] - lk_t).max() < 2e-2
    assert np.abs(out["points_mean"] - pts_t.mean(0)).max() < 2e-4
    assert np.abs(out["log10ks_mean"] - par_t["log10_ksat"].mean(0)).max() < 1e-4
