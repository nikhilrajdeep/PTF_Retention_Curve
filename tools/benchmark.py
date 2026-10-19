"""
benchmark.py -- time the new raster pipeline for one county and, if a result
from the original app (bench_original.py) is available, compare pixel by pixel.

    python tools/benchmark.py --data data/solus_cog --family PDI --inputs SSCBDOC --county Alachua
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptf_engine import core, raster          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--family", default="PDI")
ap.add_argument("--inputs", default="SSCBDOC")
ap.add_argument("--county", default="Alachua")
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--counties", default=str(ROOT / "data/counties/fl_counties_epsg5070.geojson"))
ap.add_argument("--compare", default=None)
ap.add_argument("--no-numba", action="store_true")
a = ap.parse_args()

T = {}
t = time.time(); en = core.Ensemble.load(ROOT / "models/ptf_model_bundle", a.family, a.inputs); T["load model"] = time.time() - t
t = time.time(); store = raster.SolusStore(a.data); T["open store"] = time.time() - t
gj = json.loads(Path(a.counties).read_text())
geom = next(f["geometry"] for f in gj["features"] if f["properties"]["NAME"] == a.county)
t = time.time(); aoi = store.aoi_from_geometry(geom, a.county); T["AOI window + mask"] = time.time() - t
t = time.time(); inp = raster.gather_inputs(store, en.features, a.depth, aoi, en.train_min, en.train_max); T["read COG window + dedup"] = time.time() - t
pf = core.pf_grid(-2, 7, 0.05)
if not a.no_numba and core.HAS_NUMBA:      # first call compiles; time the warm call
    t = time.time(); en.predict_area(inp.X_unique[:50], inp.counts[:50], pf, [2.0, 4.2]); T["(numba JIT warm-up, once per server)"] = time.time() - t
t = time.time()
out = en.predict_area(inp.X_unique, inp.counts, pf, [2.0, 4.2], use_numba=not a.no_numba)
T["predict maps + area curves"] = time.time() - t
t = time.time()
fc = raster.scatter(out["points_mean"][:, 0], inp); pwp = raster.scatter(out["points_mean"][:, 1], inp)
ks = raster.scatter(10 ** out["log10ks_mean"], inp)
T["scatter to maps"] = time.time() - t
total = sum(v for k, v in T.items() if not k.startswith("("))
print(f"{a.family}/{a.inputs} {a.county} {a.depth} cm: {inp.n_valid:,} px -> {len(inp.counts):,} unique rows "
      f"({inp.n_valid/len(inp.counts):.1f}x fewer)  numba={'off' if a.no_numba else core.HAS_NUMBA}")
for k, v in T.items(): print(f"  {k:40s} {v:7.3f} s")
print(f"  {'TOTAL (excluding one-off JIT)':40s} {total:7.3f} s")

if a.compare:
    o = np.load(a.compare)
    W = int(o["W"])
    r0, c0 = int(aoi.window.row_off), int(aoi.window.col_off)
    h, w = inp.shape
    rr, cc = np.divmod(inp.valid_flat, w)
    new_full = (rr + r0) * W + (cc + c0)
    common, i_new, i_old = np.intersect1d(new_full, o["idx"], return_indices=True)
    print(f"  pixels: new {len(new_full):,}  original {len(o['idx']):,}  common {len(common):,}")
    for name, new, old in [("FC", out["points_mean"][inp.inverse, 0], o["fc"]),
                           ("PWP", out["points_mean"][inp.inverse, 1], o["pwp"]),
                           ("log10Ksat", out["log10ks_mean"][inp.inverse], np.log10(o["ks"]))]:
        d = np.abs(new[i_new] - old[i_old])
        print(f"  max |new - original| {name:10s} {d.max():.2e}   mean {d.mean():.2e}")
    print(f"  area SWRC curve max diff {np.abs(out['curve_theta'] - o['th_a']).max():.2e}  "
          f"log10K curve (>-10) max diff {np.abs(out['curve_log10k'] - o['lk_a'])[o['lk_a'] > -10].max():.2e}")
