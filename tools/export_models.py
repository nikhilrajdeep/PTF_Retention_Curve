"""
export_models.py -- turn the 12 trained PyTorch ensembles into a torch-free bundle.

For every run in ptf_model_bundle/<RUN>/ this writes, next to the originals:

    weights.npz    W0,b0, W1,b1, ..., Whead,bhead  for each member (float32)
                   stored as  m{k}_W{i} / m{k}_b{i}
    engine.json    features, activation, physics bounds, scaler (min_/scale_),
                   number of members and layers -- everything inference needs

The checkpoints, scaler and configs are only READ.  Torch and scikit-learn are
needed to run this script once; the app itself then needs neither.

Usage
-----
    python tools/export_models.py --bundle models/ptf_model_bundle
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch

PREFIX = {"VGM": "vgm", "VGDRY": "vgdry", "PDI": "pdi"}


def linear_layers(state):
    """Return [(W, b), ...] in forward order: encoder Linear layers, then head.

    Dropout shifts nn.Sequential indices (encoder.0, encoder.3, ...), so the
    order is recovered from the integer index rather than assumed."""
    enc = sorted({int(m.group(1)) for k in state
                  if (m := re.match(r"encoder\.(\d+)\.weight$", k))})
    layers = [(state[f"encoder.{i}.weight"], state[f"encoder.{i}.bias"]) for i in enc]
    layers.append((state["head.weight"], state["head.bias"]))
    return [(w.detach().cpu().numpy().astype(np.float32),
             b.detach().cpu().numpy().astype(np.float32)) for w, b in layers]


def bounds(family, ph):
    b = dict(tr_lo=ph["theta_r_bounds"][0], tr_hi=ph["theta_r_bounds"][1],
             ts_lo=ph["theta_s_bounds"][0], ts_hi=ph["theta_s_bounds"][1],
             ts_gap=ph["theta_s_minus_theta_r_min"],
             a_lo=ph["alpha_bounds_1_per_cm"][0], a_hi=ph["alpha_bounds_1_per_cm"][1],
             n_lo=ph["n_bounds"][0], n_hi=ph["n_bounds"][1],
             k_lo=ph["Ksat_bounds_cm_d"][0], k_hi=ph["Ksat_bounds_cm_d"][1])
    if family == "PDI":
        h0_pf = ph.get("h0_pf", 6.8)
        b.update(om_lo=ph["omega_bounds"][0], om_hi=ph["omega_bounds"][1],
                 lam_lo=ph["lambda_bounds"][0], lam_hi=ph["lambda_bounds"][1],
                 h0_pf=h0_pf, h0_cm=ph.get("h0_cm", 10.0 ** h0_pf),
                 film_a=ph.get("film_a", -1.5))
    else:
        b.update(tau_lo=ph["tau_bounds"][0], tau_hi=ph["tau_bounds"][1])
    if family == "VGDRY":
        b["pf_dry_ref"] = ph.get("dry_correction", {}).get("dry_reference_pF", 6.8)
    return b


def export_run(run_dir: Path):
    family, input_set = run_dir.name.split("_", 1)
    config = json.loads((run_dir / "model_config.json").read_text())
    hp = run_dir / "best_hyperparameters.json"
    cfg = json.loads(hp.read_text())["config"] if hp.exists() else config["best_config"]
    scaler = joblib.load(run_dir / "input_scaler.joblib")

    files = sorted((run_dir / "ensemble_models").glob(f"{PREFIX[family]}_member_*.pt"))
    arrays = {}
    for k, f in enumerate(files):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        state = ck["state_dict"] if "state_dict" in ck else ck
        for i, (w, b) in enumerate(linear_layers(state)):
            arrays[f"m{k}_W{i}"] = w
            arrays[f"m{k}_b{i}"] = b
    np.savez_compressed(run_dir / "weights.npz", **arrays)

    meta = dict(
        run=run_dir.name, family=family, input_set=input_set,
        features=list(config["features"]),
        n_members=len(files), n_linear=len(arrays) // (2 * len(files)),
        activation=str(cfg["activation"]), hidden=int(cfg["hidden"]),
        layers=int(cfg["layers"]),
        scaler=dict(min_=np.asarray(scaler.min_, float).tolist(),
                    scale_=np.asarray(scaler.scale_, float).tolist(),
                    data_min_=np.asarray(scaler.data_min_, float).tolist(),
                    data_max_=np.asarray(scaler.data_max_, float).tolist(),
                    feature_range=list(scaler.feature_range)),
        bounds=bounds(family, config["physics"]),
        physics=config["physics"],
        uncertainty=config.get("uncertainty", {}),
    )
    (run_dir / "engine.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    args = ap.parse_args()
    for run_dir in sorted(p for p in args.bundle.iterdir() if (p / "ensemble_models").exists()):
        m = export_run(run_dir)
        print(f"{m['run']:15s} {m['n_members']} members, {m['n_linear']} linear layers, "
              f"{m['activation']}, features={m['features']}")


if __name__ == "__main__":
    main()
