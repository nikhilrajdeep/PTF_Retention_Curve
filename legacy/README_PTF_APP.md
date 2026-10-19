# Inverse-PTF Explorer v4 — 12-model edition

Three files replace the single-model `streamlit_ptf_app1.py`:

| File | Role |
|---|---|
| `ptf_core.py` | Inference core: the 3 network architectures + 3 physics models + registry + deep-ensemble predictor. The only place the physics lives. |
| `build_model_bundle.py` | One-off script that collects the 12 trained runs into a small portable folder. |
| `streamlit_ptf_app2.py` | The web application. |

## 1. Build the model bundle (run once)

```bash
python build_model_bundle.py ^
  --source "C:\Users\Nikhil\Downloads\Final_PTF" ^
  --out    "C:\Users\Nikhil\Downloads\ptf_model_bundle"
```

It copies, for each of the 12 runs, only what inference needs:

```
ptf_model_bundle/
  manifest.json                      <- what is inside, used by the app sidebar
  VGM_SSC/        VGM_SSCBD/      VGM_SSCOC/      VGM_SSCBDOC/
  VGDRY_SSC/      VGDRY_SSCBD/    VGDRY_SSCOC/    VGDRY_SSCBDOC/
  PDI_SSC/        PDI_SSCBD/      PDI_SSCOC/      PDI_SSCBDOC/
      ensemble_models/<prefix>_member_01..10.pt
      input_scaler.joblib
      model_config.json
      best_hyperparameters.json
      test_metrics.csv (if present)
```

Left behind on purpose: `main_curve_arrays_cache_*.joblib` (~27 MB each),
`optuna_study.joblib`, every CSV/XLSX/PNG result table, profile exports.
Bundle size ends up around 15–25 MB — small enough to zip, version-control
or push to a server.

The script finishes by loading every bundled run and predicting a test soil,
so a broken copy is caught immediately.

## 2. Run the app

```bash
streamlit run streamlit_ptf_app2.py
```

Edit the paths at the top of `streamlit_ptf_app2.py` first:

```python
MODEL_ROOT    = Path(r"C:\Users\Nikhil\Downloads\ptf_model_bundle")
FALLBACK_ROOT = Path(r"C:\Users\Nikhil\Downloads\Final_PTF")
RASTER_DIR    = Path(r"C:\Users\Nikhil\Downloads\Final_PTF\Soil Properties Maps")
COUNTY_DIR    = Path(r"...\Florida_County_Boundaries_with_FDOT_Districts")
```

If the bundle is missing the app falls back to reading `Final_PTF` in place.

## 3. How the 12 models are addressed

`ptf_core.SOURCE_FOLDER` maps `(family, input_set)` to the training folder name,
and `MODEL_FAMILIES[fam]["prefix"]` gives the checkpoint prefix:

| Family | Label in app | Folder pattern | Checkpoints | Head outputs |
|---|---|---|---|---|
| `VGM` | Classic VG (van Genuchten–Mualem) | `VGM_DNN_<SET>_new` | `vgm_member_*.pt` | 6 |
| `VGDRY` | Enhanced VG (dry-end corrected) | `VG_DryCorrected_DNN_<SET>` | `vgdry_member_*.pt` | 6 |
| `PDI` | PDI (Peters–Durner–Iden) | `PDI_DNN_<SET>` | `pdi_member_*.pt` | 7 |

Nothing about the architecture is hard-coded in the app: `model_config.json`
supplies the feature list and order, the parameter bounds and the physics
constants; `best_hyperparameters.json` supplies hidden width, depth, activation
and dropout. That is why selecting SSC never asks for bulk density, and why
picking SSCBDOC automatically requests five predictors in the right order.

## 4. What the app does

**Tab 1 — own soil data.** Pick input set → pick model → type the predictors.
Returns:
* SWRC: x = θ (0 – 0.7), y = pF (−2 … 7)
* K(θ): x = θ (0 – 0.7), y = log₁₀K (−10 … 7)
* Ksat as a single number (with its ± 1 SD band)
* FC (green marker), PWP (red marker) and PAW shaded in blue on the curve
* the full fitted parameter table (θr, θs, α, n, Ksat, and τ or ω/λ)

**Tab 2 — SOLUS soil data.** Pick input set → model → depth (5/15/30/60/100 cm)
→ location (county, drawn AOI, map point, or lat/lon). Only the SOLUS layers the
chosen input set needs are read from disk. Returns FC, PWP, PAW and log₁₀Ksat
as one 2×2 map figure, area-averaged SWRC and K(θ), an AOI summary table, and
GeoTIFF downloads.

### Map layout

All four maps live in a **single 2×2 figure**. They were previously two
figures — a 1×3 row and a lone Ksat map — and Streamlit stretches each figure
to the container width independently, so the single map rendered about three
times the size of the other three. One figure fixes the relative sizes.

Each panel is the same box, the figure height is derived from the AOI window's
own pixel aspect (clipped so a long thin county cannot produce an absurdly
tall figure), and `imshow(aspect="equal")` keeps every map at its true shape.
The colorbars use `fraction=0.046, pad=0.04` so each tracks its own image
height rather than the axes box — without that, the panel with a differently
shaped colorbar ends up a different width. FC, PWP and PAW share one colour
scale (2nd–98th percentile of the three pooled) so they are directly
comparable; Ksat gets its own. That means the θ colorbar is drawn three times,
which is the price of keeping all four panels identical in size.

## 5. The shaded band

One interval, used everywhere: the **10-member deep-ensemble mean ± 1 SD**
(≈ 68 %). The shaded curve bands, the FC/PWP/PAW metrics and the
fitted-parameter table are all on this scale, so nothing on screen needs
mental conversion. It is model (epistemic) spread across the ensemble — not a
calibrated predictive interval against measurements. Change `Z_SCORE` at the
top of the app to widen it (1.96 for ≈ 95 %); the label follows automatically
via `BAND_LABEL`.

With θ on the x-axis the band is drawn horizontally (`fill_betweenx`): at each
pF level — and at each log₁₀K level in the conductivity panel — the curve may
sit anywhere between the low and high θ.

### PAW shading

Every SWRC panel — manual entry, single raster pixel, and area-averaged —
marks field capacity in **green**, the wilting point in **red**, and shades
plant-available water in **blue**. With θ on the x-axis the PAW region is
bounded on the left by θ at the wilting point and on the right by the curve,
spanning the pF band from FC down to PWP: widest at FC, closing to zero at
PWP. Its area is the water the soil can actually give up to a crop. The legend
carries the number. On the area plot FC and PWP are read off the plotted
area-mean curve so the markers sit exactly on the line.

The PAW wedge is drawn underneath the uncertainty band (a lighter `skyblue`
against the band's `tab:blue`) so both stay readable; the shading is skipped
if FC and PWP are reversed or non-finite.

## 5b. Plot axes

Both panels put water content on the x-axis:

| Panel | x | y |
|---|---|---|
| SWRC | θ, 0 – 0.7 cm³ cm⁻³ | pF, −2 … 7 |
| Conductivity | θ, 0 – 0.7 cm³ cm⁻³ | log₁₀K, −10 … 7 (cm d⁻¹) |

pF remains the *parameter* along which both curves are traced (−2 … 7, step
0.05, set by `SWRC_PF` / `KH_PF`); it is simply no longer an x-axis. Fixed
limits live in `THETA_LIMITS`, `PF_LIMITS` and `LOGK_LIMITS` at the top of the
app. For coarse sands the VG and VG-Dry conductivity curves run below
log₁₀K = −10 at the dry end and are clipped by the fixed axis — widen
`LOGK_LIMITS` if you want the whole tail.

## 5c. Raster scaling and the input-layer check

SOLUS stores BD as g cm⁻³ × 100 and SOC as % × 1000. Rather than trusting a
fixed divisor, `infer_scale()` picks the one that puts each layer's **median**
inside a physically plausible range (the documented divisor is tried first).
A layer that arrives already in real units is therefore handled correctly, and
one that is 100× too large cannot slip through.

This matters because the models sit behind a `MinMaxScaler`. Feeding BD = 135
instead of 1.35 to a scaler fitted on 0.37–1.9 drives every sigmoid output head
into saturation: θs collapses to its lower bound, FC ≈ PWP, PAW ≈ 0, and every
pixel returns the *same* curve. On a map that just looks like a flat colour —
a silent wrong answer.

So the spatial tab now prints an **"Input layers reaching the model"** panel
showing, per feature, the source file, the divisor applied, the raster min/max,
the training min/max, and the percentage of pixels outside the training hull.
More than 1 % outside opens the panel with a red warning. A few edge pixels are
normal and do not trigger it.

## 6. Texture renormalisation

SOLUS sand/silt/clay are not a closed composition: statewide at 5 cm the sum
runs from 0 to 157 % (median 101 %), and `Sand5.tif` alone reaches 104 %. The
models were trained on fractions summing to 100, so both tabs renormalise
before prediction — each pixel's three fractions are rescaled to sum to 100,
preserving their ratios (`ptf_core.normalize_texture`). Pixels whose sum is
zero or non-finite become NaN and drop out as invalid.

This matters: before renormalising, 21 % of sand pixels sat above the 100 %
training maximum; after, essentially none do. The input-layer panel reports the
pre-normalisation sum range and shows the post-normalisation values, i.e. what
the network actually receives.

Note the renormalisation on the manual tab too — if you type values that do not
sum to 100 the app tells you what it used instead.

## 7. Area-averaged curves use every pixel

There is no pixel cap. The curves are averaged over every valid pixel in the
county or AOI, matching what app1 did for its maps. Holding all of them at once
would need several GB (members × pixels × pF points), so
`ptf_core.predict_mean_curves` accumulates the mean in chunks and never
materialises the full block; a progress bar shows how far it has got. A
~140 k-pixel county takes roughly 30 s with a 10-member ensemble.
