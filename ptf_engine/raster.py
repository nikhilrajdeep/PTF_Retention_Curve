"""
ptf_engine.raster
=================
Fast access to the SOLUS soil-property COGs and the AOI -> model-input step.

Why this is fast
----------------
* All layers sit on ONE 100 m grid (tools/build_cogs.py), so an AOI is a plain
  pixel window: only the 512x512 tiles that touch it are decompressed, instead
  of warping the whole of Florida (67 M pixels x 5 layers) on every run.
* Values stay as their stored integers until the very end.  SOLUS rasters are
  quantised (sand 96 %, BD 144 -> 1.44 g cm-3 ...), so thousands of pixels in a
  county share an identical input vector.  The pixels are collapsed to unique
  input rows (with a count each); the network and physics run once per unique
  row, and results are scattered back.  This is exact -- the area mean is a
  count-weighted mean -- and typically 3-250x fewer evaluations.
* Data can live next to the app or behind any HTTP server that supports range
  requests (GitHub releases, S3, Zenodo); GDAL reads COG tiles over HTTP
  directly, so nothing has to be downloaded in full.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

from .core import normalize_texture

# model feature -> SOLUS variable
FEATURE_TO_VAR = {"Sand_pct": "Sand", "Silt_pct": "Silt", "Clay_pct": "Clay",
                  "BD_g_cm3": "BD", "OC_pct": "SOC"}
DEPTHS = [5, 15, 30, 60, 100]
# bits used to pack each variable's stored integer into one int64 key
_KEY_BITS = {"Sand": 8, "Silt": 8, "Clay": 8, "BD": 16, "SOC": 16}

# GDAL settings for cloud-native reads.  Set as environment variables so they
# apply in every thread Streamlit uses (a rasterio.Env is thread-local).
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.json",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_MULTIPLEX="YES", VSI_CACHE="TRUE",
                GDAL_CACHEMAX="256")
for _k, _v in GDAL_ENV.items():
    os.environ.setdefault(_k, _v)


@dataclass
class AOI:
    window: Window
    mask: np.ndarray            # (h, w) bool, True inside the AOI
    transform: rasterio.Affine  # transform of the window
    label: str


class SolusStore:
    """Thin read-only wrapper around the folder (or URL) of SOLUS COGs."""

    def __init__(self, location: str | Path):
        self.location = str(location).rstrip("/")
        self.remote = self.location.startswith(("http://", "https://"))
        cat_path = self._path("catalog.json")
        if self.remote:
            import urllib.request
            with urllib.request.urlopen(cat_path, timeout=30) as r:
                self.catalog = json.loads(r.read().decode())
        else:
            self.catalog = json.loads(Path(cat_path).read_text())
        self.crs = self.catalog["crs"]
        self.transform = rasterio.Affine(*self.catalog["transform"])
        self.width = self.catalog["width"]
        self.height = self.catalog["height"]
        self.bounds = tuple(self.catalog["bounds"])
        self._handles: Dict[str, rasterio.DatasetReader] = {}

    # ------------------------------------------------------------ files
    def _path(self, name: str) -> str:
        if self.remote:
            return f"{self.location}/{name}"
        return str(Path(self.location) / name)

    def _open(self, var: str, depth: int):
        key = f"{var}{depth}"
        if key not in self._handles:
            name = self.catalog["layers"][key]["file"]
            p = self._path(name)
            self._handles[key] = rasterio.open(f"/vsicurl/{p}" if self.remote else p)
        return self._handles[key]

    def layer_info(self, var: str, depth: int) -> Dict:
        return self.catalog["layers"][f"{var}{depth}"]

    # ------------------------------------------------------------ AOIs
    def aoi_from_geometry(self, geom, label: str, pad: int = 2) -> Optional[AOI]:
        """geom: a GeoJSON-like mapping in the store CRS (EPSG:5070)."""
        from shapely.geometry import shape
        g = shape(geom)
        if g.is_empty:
            return None
        win = from_bounds(*g.bounds, transform=self.transform)
        c0 = max(0, int(np.floor(win.col_off)) - pad)
        r0 = max(0, int(np.floor(win.row_off)) - pad)
        c1 = min(self.width, int(np.ceil(win.col_off + win.width)) + pad)
        r1 = min(self.height, int(np.ceil(win.row_off + win.height)) + pad)
        if c1 <= c0 or r1 <= r0:
            return None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        wt = window_transform(win, self.transform)
        mask = rio_features.geometry_mask([geom], out_shape=(win.height, win.width),
                                          transform=wt, invert=True,
                                          all_touched=False)
        return AOI(window=win, mask=mask, transform=wt, label=label)

    def read(self, var: str, depth: int, window: Window) -> Tuple[np.ndarray, int]:
        ds = self._open(var, depth)
        return ds.read(1, window=window), ds.nodata

    def pixel(self, x: float, y: float, variables: Sequence[str], depth: int):
        """Physical values of one pixel (x, y in EPSG:5070), or None for nodata."""
        r, c = rowcol(self.transform, x, y)
        if not (0 <= r < self.height and 0 <= c < self.width):
            return None
        win = Window(c, r, 1, 1)
        out = {}
        for v in variables:
            a, nod = self.read(v, depth, win)
            if a[0, 0] == nod:
                return None
            out[v] = float(a[0, 0]) * self.layer_info(v, depth)["scale"]
        return out


# ===========================================================================
# AOI -> unique model inputs
# ===========================================================================
@dataclass
class AOIInputs:
    X_unique: np.ndarray        # (U, F) physical units, model feature order
    counts: np.ndarray          # (U,)  pixels per unique row
    inverse: np.ndarray         # (n_valid,) unique-row index for each valid pixel
    valid_flat: np.ndarray      # (n_valid,) flat index into the AOI window
    shape: Tuple[int, int]
    n_valid: int
    diagnostics: List[Dict]
    texture_sum: Optional[Tuple[float, float, float]]


def gather_inputs(store: SolusStore, features: List[str], depth: int, aoi: AOI,
                  train_min: np.ndarray, train_max: np.ndarray) -> Optional[AOIInputs]:
    """Read the layers the model needs inside the AOI and collapse the valid
    pixels to unique input rows."""
    vars_ = [FEATURE_TO_VAR[f] for f in features if f != "Depth_harm_cm"]
    h, w = aoi.mask.shape
    valid = aoi.mask.copy()
    raw = {}
    for v in vars_:
        a, nod = store.read(v, depth, aoi.window)
        valid &= a != nod
        raw[v] = a
    flat = np.flatnonzero(valid.ravel())
    if flat.size == 0:
        return None

    # pack stored integers into one int64 key per pixel -> np.unique on 1-D
    key = np.zeros(flat.size, np.int64)
    shift = 0
    for v in vars_:
        key |= raw[v].ravel()[flat].astype(np.int64) << shift
        shift += _KEY_BITS[v]
    ukey, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)

    phys = {}
    shift = 0
    for v in vars_:
        bits = _KEY_BITS[v]
        stored = (ukey >> shift) & ((1 << bits) - 1)
        phys[v] = stored.astype(np.float64) * store.layer_info(v, depth)["scale"]
        shift += bits

    tex_sum = None
    if {"Sand", "Silt", "Clay"} <= set(phys):
        s = phys["Sand"] + phys["Silt"] + phys["Clay"]
        sp = np.repeat(s, counts)
        tex_sum = (float(sp.min()), float(np.median(sp)), float(sp.max()))
        phys["Sand"], phys["Silt"], phys["Clay"] = normalize_texture(
            phys["Sand"], phys["Silt"], phys["Clay"])

    X = np.empty((len(ukey), len(features)), np.float64)
    for j, f in enumerate(features):
        X[:, j] = float(depth) if f == "Depth_harm_cm" else phys[FEATURE_TO_VAR[f]]

    # rows whose texture summed to 0 are NaN -> drop (pixels become nodata)
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        keep_px = ok[inverse]
        remap = np.cumsum(ok) - 1
        flat, inverse = flat[keep_px], remap[inverse[keep_px]]
        X, counts = X[ok], counts[ok]
        if flat.size == 0:
            return None

    # diagnostics: what actually reaches the network, vs. the training range
    diags = []
    for j, f in enumerate(features):
        col = X[:, j]
        lo, hi = float(train_min[j]), float(train_max[j])
        out_frac = float(counts[(col < lo) | (col > hi)].sum() / counts.sum())
        info = (store.layer_info(FEATURE_TO_VAR[f], depth)
                if f != "Depth_harm_cm" else None)
        diags.append({"Feature": f,
                      "Source": info["file"] if info else "selected depth",
                      "Scale": info["scale"] if info else 1.0,
                      "AOI min": round(float(col.min()), 4),
                      "AOI max": round(float(col.max()), 4),
                      "Training min": round(lo, 4), "Training max": round(hi, 4),
                      "% outside training": round(100 * out_frac, 2),
                      "OK": "yes" if out_frac <= 0.01 else "no"})

    return AOIInputs(X_unique=X, counts=counts, inverse=inverse, valid_flat=flat,
                     shape=(h, w), n_valid=int(flat.size), diagnostics=diags,
                     texture_sum=tex_sum)


def scatter(values: np.ndarray, inp: AOIInputs) -> np.ndarray:
    """Unique-row values -> (h, w) float32 map with NaN outside the AOI."""
    out = np.full(inp.shape[0] * inp.shape[1], np.nan, np.float32)
    out[inp.valid_flat] = values[inp.inverse]
    return out.reshape(inp.shape)


def to_geotiff_bytes(arr: np.ndarray, aoi: AOI, crs: str, description: str) -> bytes:
    """Write one AOI map as a small Cloud-Optimized GeoTIFF in memory."""
    from rasterio.io import MemoryFile
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs=crs, transform=aoi.transform, nodata=np.nan,
                compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256)
    with MemoryFile() as mf:
        with mf.open(**prof) as dst:
            dst.write(arr.astype(np.float32), 1)
            dst.set_band_description(1, description)
        return mf.read()


def curve_rows(inp: AOIInputs, max_pixels: int, seed: int = 0):
    """Rows + weights for the area-averaged curves.

    If the AOI has at most `max_pixels` unique input rows the curves use every
    pixel exactly (unique rows weighted by their pixel counts).  Otherwise a
    simple random sample of `max_pixels` pixels is drawn without replacement
    and collapsed to its own unique rows -- an unbiased estimate of the
    every-pixel mean whose standard error is reported alongside.

    Returns (X, weights, n_used_pixels, exact)
    """
    if len(inp.counts) <= max_pixels:
        return inp.X_unique, inp.counts, inp.n_valid, True
    rng = np.random.default_rng(seed)
    pick = rng.choice(inp.n_valid, size=max_pixels, replace=False)
    c = np.bincount(inp.inverse[pick], minlength=len(inp.counts))
    keep = np.flatnonzero(c)
    return inp.X_unique[keep], c[keep], max_pixels, False
