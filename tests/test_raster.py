"""The de-duplicated raster path must give the same per-pixel maps as running
every pixel through the model one by one.  Needs data/solus_cog."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptf_engine import core, raster          # noqa: E402

DATA = ROOT / "data" / "solus_cog"
pytestmark = pytest.mark.skipif(not (DATA / "catalog.json").exists(),
                                reason="SOLUS COGs not present")


@pytest.mark.parametrize("family,input_set", [("VGM", "SSC"), ("PDI", "SSCBDOC"),
                                              ("VGDRY", "SSCOC")])
def test_dedup_is_exact(family, input_set):
    store = raster.SolusStore(DATA)
    en = core.Ensemble.load(ROOT / "models" / "ptf_model_bundle", family, input_set)
    # ~6 x 6 km box near Gainesville
    x0, y0 = 1_380_000.0, 600_000.0
    geom = {"type": "Polygon", "coordinates": [[(x0, y0), (x0 + 6000, y0),
            (x0 + 6000, y0 + 6000), (x0, y0 + 6000), (x0, y0)]]}
    aoi = store.aoi_from_geometry(geom, "box")
    inp = raster.gather_inputs(store, en.features, 15, aoi, en.train_min, en.train_max)
    assert inp is not None and inp.n_valid > 1000
    out = en.predict_area(inp.X_unique, inp.counts, [], [2.0, 4.2])
    fc_map = raster.scatter(out["points_mean"][:, 0], inp)

    # naive: every valid pixel, no de-duplication
    X_px = inp.X_unique[inp.inverse]
    th, _, _ = en.predict_members(X_px, np.array([2.0, 4.2]), np.array([0.0]))
    naive = th[:, :, 0].mean(0)
    rr, cc = np.divmod(inp.valid_flat, inp.shape[1])
    assert np.abs(fc_map[rr, cc] - naive).max() < 1e-5
    assert np.isnan(fc_map).sum() == fc_map.size - inp.n_valid
