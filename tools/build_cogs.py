"""
build_cogs.py -- convert the SOLUS GeoTIFFs into analysis-ready Cloud-Optimized GeoTIFFs.

The original rasters are left untouched.  For every variable and depth the script

  1. snaps the layer onto ONE common 100 m grid (the Clay grid, EPSG:5070).
     The SOLUS layers already share pixel size and their origins differ by whole
     pixels, so this is an exact integer-offset window read -- no resampling.
  2. keeps the native integer storage (uint8 / uint16) and a single nodata value,
  3. writes a COG: 512x512 internal tiles, DEFLATE + horizontal predictor,
     internal overviews, and the physical scale factor in the band metadata
     (BD: 0.01 -> g cm-3, SOC: 0.001 -> %), so readers never have to guess it.

Because every layer now sits on the same grid, the app reads an AOI with plain
windowed reads (a few tiles), instead of warping the whole state on every run.

Usage
-----
    python tools/build_cogs.py --src "C:/Users/Nikhil/Downloads/Final_PTF/Soil Properties Maps" \
                               --out "C:/Users/Nikhil/Downloads/Final_PTF/Soil_Properties_Maps_COG"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window

VARIABLES = ["Sand", "Silt", "Clay", "BD", "SOC"]
DEPTHS = [5, 15, 30, 60, 100]
REFERENCE = "Clay"
# Physical value = stored integer * SCALE.  Documented SOLUS encodings; checked
# against each layer's median below so a re-exported layer cannot slip through.
SCALE = {"Sand": 1.0, "Silt": 1.0, "Clay": 1.0, "BD": 0.01, "SOC": 0.001}
# Texture can legitimately have a median of 0 (e.g. silt in deep Florida sands),
# so only BD and SOC -- the layers whose scale factor matters -- get a floor.
PLAUSIBLE_MEDIAN = {"Sand": (0.0, 110), "Silt": (0.0, 110), "Clay": (0.0, 110),
                    "BD": (0.2, 2.4), "SOC": (0.05, 60)}
UNITS = {"Sand": "%", "Silt": "%", "Clay": "%", "BD": "g cm-3", "SOC": "%"}
NODATA = {"uint8": 255, "uint16": 65535}


def aligned_read(path: Path, ref) -> tuple[np.ndarray, float]:
    """Read `path` on the reference grid by integer pixel offset (exact)."""
    with rasterio.open(path) as src:
        if src.crs != ref.crs or src.res != ref.res:
            raise ValueError(f"{path.name}: CRS/resolution differ from reference")
        dx = (ref.transform.c - src.transform.c) / src.res[0]
        dy = (src.transform.f - ref.transform.f) / src.res[1]
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
            raise ValueError(f"{path.name}: grid not pixel-aligned with reference")
        win = Window(int(round(dx)), int(round(dy)), ref.width, ref.height)
        arr = src.read(1, window=win, boundless=True, fill_value=src.nodata)
        return arr, src.nodata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--blocksize", type=int, default=512)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalog = {"crs": None, "transform": None, "width": None, "height": None,
               "layers": {}}
    for depth in DEPTHS:
        with rasterio.open(args.src / f"{REFERENCE}{depth}.tif") as ref:
            for var in VARIABLES:
                arr, nod = aligned_read(args.src / f"{var}{depth}.tif", ref)
                dtype = "uint8" if arr.dtype == np.uint8 else "uint16"
                out_nod = NODATA[dtype]
                arr = arr.astype(dtype)
                if nod is not None and nod != out_nod:
                    # the old nodata value must not collide with a real value
                    arr = np.where(arr == np.asarray(nod).astype(dtype), out_nod, arr)
                valid = arr != out_nod
                med = float(np.median(arr[valid])) * SCALE[var]
                lo, hi = PLAUSIBLE_MEDIAN[var]
                if not (lo <= med <= hi):
                    raise ValueError(f"{var}{depth}: median {med} outside {lo}-{hi} "
                                     f"after scale {SCALE[var]} -- check encoding")

                prof = dict(driver="GTiff", width=ref.width, height=ref.height, count=1,
                            dtype=dtype, crs=ref.crs, transform=ref.transform,
                            nodata=out_nod, tiled=True, blockxsize=args.blocksize,
                            blockysize=args.blocksize, compress="deflate", predictor=2)
                tmp = args.out / f"_{var}{depth}.tmp.tif"
                with rasterio.open(tmp, "w", **prof) as dst:
                    dst.write(arr, 1)
                    dst.scales = (SCALE[var],)
                    dst.offsets = (0.0,)
                    dst.set_band_description(1, f"{var} {depth} cm ({UNITS[var]})")
                    dst.update_tags(1, variable=var, depth_cm=str(depth),
                                    units=UNITS[var], scale=str(SCALE[var]),
                                    source="SOLUS (Nauman et al.), Florida subset")
                out = args.out / f"{var}{depth}_cog.tif"
                rio_copy(tmp, out, driver="COG", blocksize=args.blocksize,
                         compress="DEFLATE", predictor="YES", level=9,
                         overview_resampling="nearest", bigtiff="IF_SAFER")
                tmp.unlink()
                catalog["layers"][f"{var}{depth}"] = {
                    "file": out.name, "variable": var, "depth_cm": depth,
                    "dtype": dtype, "nodata": out_nod, "scale": SCALE[var],
                    "units": UNITS[var], "median_physical": round(med, 4),
                    "valid_pixels": int(valid.sum()),
                    "size_mb": round(out.stat().st_size / 1e6, 2)}
                print(f"{out.name:22s} {catalog['layers'][f'{var}{depth}']['size_mb']:6.1f} MB"
                      f"  median {med:.3f} {UNITS[var]}")
            catalog.update(crs=ref.crs.to_string(), transform=list(ref.transform)[:6],
                           width=ref.width, height=ref.height,
                           bounds=list(ref.bounds))
    (args.out / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print("wrote", args.out / "catalog.json")


if __name__ == "__main__":
    main()
