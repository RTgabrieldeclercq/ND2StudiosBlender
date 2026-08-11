# `piv_field` — PIV (OpenPIV) integration contract

## 1. Purpose + where the real math lives

Runs **2D multipass window-deformation Particle Image Velocimetry** on already-prepared
`(H, W)` frame pairs (or an ordered series) and returns the displacement field on the
interrogation-window grid as a :class:`PIVResult` — the FLOW sibling of `dic_correlate`
(independent windows + per-vector signal-to-noise, versus DIC's globally-regularized
FE solve).

**The real correlation math is EXTERNAL.** FFT cross-correlation, subpixel peak fits,
signal-to-noise ratios, validation tests, vector replacement and the smoothn smoother all
live in the third-party **`openpiv`** package (**GPLv3** — the reason this kernel is a
lazy adapter and none of that code is vendored into this repo), imported **lazily**
inside `_require_openpiv`.

The **in-repo glue** is:
- a **multipass driver** (`_piv_one_pair` + `_deform_pass`) that mirrors
  `openpiv.windef.piv` / `simple_multipass` pass-for-pass but (a) RETURNS the final
  pass's signal-to-noise field, which `windef` computes and discards, (b) stays in image
  coordinates (no `transform_coordinates` y-flip), and (c) survives the all-vectors-invalid
  case that makes `windef.multipass_img_deform` raise `ValueError`;
- a **pass-ladder builder** (`_ladder`) with honest degradation (§6);
- ROI handling on the vector grid (§6).

> **Validated 2026-08-10** against openpiv's own synthetic-shift suite replicated in
> `scripts/piv_synthetic_bench.py`, plus analytic rotation/shear/sinusoid fields and a
> **bit-identical parity assertion** against `windef.simple_multipass`. See §12.

## 2. Entry points

```python
run_piv_pair(
    frame_a: np.ndarray,               # (H, W) reference
    frame_b: np.ndarray,               # (H, W) current
    voxel_size_um: Tuple[float, float],# (dy, dx) µm/px — echoed, NOT applied
    params: Dict[str, Any],
    *,
    roi_mask: Optional[np.ndarray] = None,   # (H, W), truthy = inside ROI
) -> PIVResult

run_piv_series(
    images: Sequence[np.ndarray],      # lazily indexed; each (H, W)
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, float],
    *,
    pairing: str = "previous",         # "previous" | "fixed_head"
    roi_mask: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,   # 0-100 per pair
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> List[PIVResult]                    # len == len(images) - 1

openpiv_available() -> bool             # find_spec probe, no import side effects
```

`pairing="previous"` correlates `(images[i-1], images[i])` — one velocity field per step.
`pairing="fixed_head"` correlates `(images[0], images[i])` — displacement from a fixed
reference the CALLER prepends (prepending the reference plane itself yields a ~0
self-pair, matching the DIC node's behaviour). `images` is **indexed lazily**
(`__len__`/`__getitem__` suffice); each frame is pulled exactly once and peak memory is
~two planes. A truthy `cancelled_cb` stops between pairs and returns the partial list.
Unlike al-dic, openpiv holds **no cross-pair caches**, so there is no speed penalty
hidden in pair-at-a-time semantics — the series call is a convenience, not a contract.

## 3. Inputs

| name | type | shape | axis order | units | required? | meaning & constraints |
|------|------|-------|------------|-------|-----------|-----------------------|
| `frame_a` / `frame_b` | `np.ndarray` | `(H, W)` | `[y, x]` | intensity | required | Grayscale pair, same shape, any numeric dtype (cast to float64). `ndim != 2` raises. |
| `images` | `Sequence[np.ndarray]` | each `(H, W)` | `[y, x]` | intensity | required | ≥ 2 planes; `images[0]` is the head (reference for `fixed_head`, first frame for `previous`). |
| `voxel_size_um` | `(float, float)` | len 2 | `(dy, dx)` | µm/px | required | Echoed onto the result; **never applied** — displacements stay in px (README convention 2). |
| `roi_mask` | `np.ndarray` or `None` | `(H, W)` | `[y, x]` | — | optional | Truthy = INSIDE the ROI (opposite of numpy.ma convention). One static mask per call. Wrong shape raises. |

## 4. Parameters (`params` dict — kernel defaults in parentheses)

| name | type | default | semantics |
|------|------|---------|-----------|
| `windowsizes` | tuple[int] | `(64, 32, 16)` | Coarse→fine interrogation-window ladder, px. Subject to §6 trimming. |
| `overlap` | float | 0.5 | Window overlap as a FRACTION of each pass's window (`round(w*f)`, clamped `[0, w-1]`). |
| `correlation_method` | str | `"circular"` | `circular` (FFT wraparound) or `linear` (zero-padded; **forces** `normalized_correlation=True` per openpiv's contract). |
| `normalized_correlation` | bool | False | Mean-subtract + variance-scale windows before correlating. Implied by `linear`. |
| `subpixel_method` | str | `"gaussian"` | `gaussian` \| `parabolic` \| `centroid` (openpiv falls back to centroid on negative peaks). |
| `deformation_method` | str | `"symmetric"` | `symmetric` (both frames deformed half-way — central-difference, Wereley & Gui 2003) or `"second image"`. |
| `interpolation_order` | int | 3 | B-spline order for image deformation AND the deformation-field spline. |
| `sig2noise_method` | str | `"peak2mean"` | `peak2mean` \| `peak2peak` — what the `qfactor` output means. |
| `sig2noise_mask` | int | 2 | Half-width masked around the first peak for `peak2peak`. |
| `sig2noise_validate` / `sig2noise_threshold` | bool / float | True / 1.0 | S2N validation; 1.0 is nearly inert (openpiv default). |
| `max_disp_px` | float | 0 | Global validation limit ±. **0 = auto: half the coarsest EFFECTIVE window.** |
| `std_threshold` | float | 10.0 | Global std test. |
| `median_test` | str | `"universal"` | `universal` (Westerweel–Scarano 2005, ε=0.2 hardcoded upstream, threshold ~2) \| `classic` (px units) \| `off` (threshold forced to 1e12). |
| `median_threshold` / `median_size` | float / int | 2.0 / 1 | Threshold (meaning depends on `median_test`); kernel = `(2·size+1)²` windows. |
| `replace_vectors` | bool | True | Governs the FINAL pass only — between passes replacement is unconditional (holes poison the predictor, mirroring `windef`). |
| `filter_method` / `max_filter_iteration` / `filter_kernel_size` | str/int/int | localmean / 4 / 2 | Replacement inpainting (openpiv `replace_outliers`). |
| `smoothn` / `smoothn_p` | bool / float | False / 0.05 | Garcia-2010 DCT-PLS smoothing of every NON-final pass (and of the output in a single-pass run — mirrors `windef.piv`). Re-masked after each call (smoothn strips maskedness). |

## 5. Output — `PIVResult` (dataclass)

Field names/layouts match `DVCResult` where they overlap, so the catalog's shared
`_dvc_rows` flattener consumes either. `G = (Gy, Gx)` is the final pass's vector grid.

| field | shape | dtype | axis order | units | meaning |
|-------|-------|-------|------------|-------|---------|
| `dim` | scalar | int | — | — | always `2`. |
| `grid_coords` | `(*G, 2)` | float64 | `[y, x]` | px | Window-centre coordinates (image rows/cols, y DOWN). |
| `displacement_field` | `(*G, 2)` | float64 | `[dy, dx]` | px | `+dy` = down rows, `+dx` = right cols. **NaN = no honest vector** (outside ROI, or failed validation with replacement off/impossible) — DROP these, don't propagate. |
| `strain_field` | — | — | — | — | always `None` (PIV measures displacement; differentiate downstream). |
| `qfactor` | `(*G,)` | float64 | — | — | Final-pass correlation signal-to-noise (`sig2noise_method`). The per-vector confidence DIC cannot provide. |
| `flags` | `(*G,)` | bool | — | — | True = failed final validation. When replacement ran, the value at a flagged point is INPAINTED (the flag means "filled from neighbours"); when it did not, the point is NaN. |
| `excluded` | `(*G,)` | bool | — | — | True = window centre outside `roi_mask`. |
| `voxel_size_um` | len-2 tuple | float | `(dy, dx)` | µm/px | echoed. |
| `method` | str | — | — | — | `"OpenPIV"`. |
| `diagnostics` | dict | — | — | — | `windowsizes`, `overlaps`, `n_passes`, `ladder_trimmed`, `correlation`, `normalized_correlation`, `subpixel`, `deformation`, `n_flagged`, `n_excluded`. **Read `windowsizes` back** — §6 trimming may have changed what you asked for. |

## 6. Conventions & GOTCHAS (real integration risk)

- **Sign convention was established EMPIRICALLY, not from docs.** Raw openpiv
  (`extended_search_area_piv`, `first_pass`) returns `u` = +x (columns right) and `v` =
  **+y (ROWS DOWN)** — exactly this repo's image convention, so this kernel does **no
  axis swap**. It is `windef.simple_multipass` that flips at the very end
  (`tools.transform_coordinates`: `y → y[::-1]`, `v → -v`) to a y-UP physical frame.
  **If you compare against `simple_multipass` output, negate its `v`** (the bench does).
- **The pass ladder degrades honestly** (`_ladder`): windows larger than `min(H, W)` are
  dropped; in a MULTIPASS run every pass additionally needs **≥ 4 vector rows and
  columns** (both the predictor interpolation and the deformation field use cubic
  `RectBivariateSpline` over the vector grid — fewer nodes would crash scipy mid-pass).
  If no multipass ladder survives, the kernel runs a SINGLE pass on the smallest
  requested window; if even that window does not fit, `ValueError`. A small crop can
  therefore silently run single-pass — check `diagnostics["ladder_trimmed"]`.
- **A fresh `PIVSettings` is built per pair** because `windef` MUTATES the settings
  object it is handed (`multipass_img_deform` nulls `sig2noise_method` when validation
  is off). Never share a settings object across calls if you bypass this kernel.
- **`replace_vectors` governs only the final pass.** Between passes, failed vectors are
  ALWAYS inpainted (mirroring `windef.piv` — a hole in the predictor poisons the next
  pass's deformation). Consequence of off: NaN at flagged points, not raw values.
- **`roi_mask` truthy = INSIDE** (the analysis region) — the opposite of numpy.ma's
  mask convention, which `excluded` follows. Sampling onto the grid is nearest-neighbour
  (`order=0`) — a deliberate deviation from `windef`, which cubic-spline-interpolates
  its boolean static mask. Masking is at the VECTOR-GRID level: image pixels outside the
  ROI still contribute texture to windows that straddle the boundary.
- **`linear` correlation forces normalized correlation** (openpiv raises garbage
  otherwise; the docstring states the requirement). `circular` is the openpiv default
  and ~2× faster.
- **The universal median test can flag REAL steep gradients.** On a clean λ=64 px
  sinusoid it flags ~26% of vectors and replacement flattens them (bench case 7b:
  RMSE 1.23 px with default validation vs 0.64 px with `median_test="off"`, where 0.64
  is the pure sinc window-attenuation floor). Fields with genuine sharp structure need
  `median_test="off"` or a higher threshold.
- **Window averaging attenuates structure below ~4 windows/wavelength** — physics, not
  a defect: a top-hat window of width `w` passes `sinc(π·w/λ)` of the amplitude
  (`w=λ/2` → 0.64, `w=λ` → **zero**, so a coarse pass at the structure's wavelength
  contributes nothing to the predictor). Choose the final window ≤ λ/4.
- **Self-pair (A vs A) is well-defined** and returns ~0 displacement with high s2n —
  callers may prepend the reference for DIC-parity `fixed_head` semantics.
- **Blank/flat frames do not crash** — every vector fails validation, nothing can be
  inpainted, the whole field comes back NaN with `flags` all-True (this is the case
  where upstream's `multipass_img_deform` raises `ValueError("Something happened in the
  validation")`).
- **Displacement stays in px** (README convention 2): multiply by `voxel_size_um`
  per-component downstream; `dt` is pinned to 1.0 — velocity is the caller's divide.
- **CALLER OWNS ALL PREP** — channel selection, m/t/z looping, crop, downsample,
  registration, background subtraction. Feed final `(H, W)` arrays.

## 7. Dependencies

| package | import-time or lazy | why |
|---------|---------------------|-----|
| `numpy` | import-time | arrays throughout. |
| `scipy` | **lazy** (inside `_deform_pass` / `_roi_on_grid`) | `RectBivariateSpline` predictor interp, `map_coordinates` deformation + ROI sampling. |
| `openpiv` | **lazy** (inside `_require_openpiv`) | ALL the correlation math. **GPLv3.** Pulls `scikit-image`, `imageio`, `matplotlib`, `natsort`, `tqdm`. Absent → friendly `ImportError` only when `run_piv_*` is called; module import still succeeds. |

Install the solver: `pip install openpiv` (verified against 0.25.4).

## 8. Failure modes / edge cases

- **`openpiv` not installed** → `ImportError` with install hint, raised only on call.
- **< 2 images** → `ValueError("PIV needs at least 2 images …")`.
- **Non-2D frame / shape mismatch / bad `roi_mask` shape / unknown `pairing` or
  `deformation_method`** → `ValueError` naming the problem.
- **Final window > image** → `ValueError` (reduce window or crop less).
- **All vectors invalid** (blank frames, wrong channel) → all-NaN field, `flags` all
  True, no exception (§6).
- **Cancel** (`cancelled_cb()` truthy) → partial list, no exception.

## 9. Minimal runnable example

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from nodegraph.kernels.piv_field import run_piv_pair

rng = np.random.default_rng(0)
A = gaussian_filter(rng.random((256, 256)), 1.5) * 250
B = np.roll(A, (3, 5), axis=(0, 1))            # planted +3 rows, +5 cols

res = run_piv_pair(A, B, voxel_size_um=(0.5, 0.5),
                   params={"windowsizes": (64, 32), "overlap": 0.5})
print(res.grid_coords.shape)                   # (Gy, Gx, 2)  [y, x] px
print(np.nanmedian(res.displacement_field[..., 0]))   # ~3.0  (dy, +down)
print(np.nanmedian(res.displacement_field[..., 1]))   # ~5.0  (dx, +right)
print(np.nanmedian(res.qfactor))               # correlation S/N per vector
```

## 10. Pipeline wiring (nodegraph)

- **Feeds in:** `analysis.piv` supplies final `(H, W)` planes after channel selection
  and the m/z/T loop, an optional ROI plane from a Voxel mask layer, and
  `pixel_size_um` as `voxel_size_um`. An external rest-state acquisition (TFM relaxed
  gel) arrives as the prepended head of a `fixed_head` series.
- **Feeds out:** the node flattens each `PIVResult` through the shared `_dvc_rows`
  into Point rows (`disp_y`/`disp_x`/`disp_mag_um` in µm, `qfactor`, `replaced`),
  drops NaN vectors, and feeds `transform.rasterize_field` for dense maps.

## 11. Provenance

New in-repo code (2026-08-10, V3 W5-P2 "Correlation alternatives") — **not** vendored
from ND2Studios v1 (v1 had no PIV) and **not** copied from openpiv (GPLv3 stays in the
external package). The driver structure deliberately parallels
`openpiv.windef.piv`/`simple_multipass` so that behaviour is upstream-predictable; the
bench pins that parity bit-for-bit where the paths coincide.

## 12. Validation (2026-08-10, openpiv 0.25.4 — `scripts/piv_synthetic_bench.py`)

256² Gaussian-smoothed random texture (σ=1.5, seed 42), interior windows only (32 px
margin), default `(64, 32)` ladder at 50% overlap unless noted, worst of RMSE_y/RMSE_x:

| case | measured | tol | note |
|------|----------|-----|------|
| zero / self-pair | 0.0000 | 0.01 | noise floor |
| integer translation (3, 5) | 0.0291 | 0.05 | |
| sub-pixel translation (2.5, −1.25) | 0.0310 | 0.06 | no peak-locking S-curve at this scale |
| large translation (0, 11), 3-pass | 0.0291 | 0.06 | ladder reach |
| rotation 1° | 0.0417 | 0.08 | window deformation tracks the linear field |
| shear ∂u/∂y = 0.02 | 0.0291 | 0.08 | |
| sinusoid λ=64, amp 2 px, median OFF | 0.6434 | 0.75 | = the sinc(π/2)≈0.64 window-attenuation floor |
| sinusoid λ=64, amp 2 px, default validation | 1.2271 | 1.35 | universal median test flags real gradients (§6) |
| sub-pixel + 5% gaussian noise | 0.0520 | 0.15 | |
| upstream `create_pair` replica | 0.1463 | 0.25 | upstream's own THRESHOLD, their dense-noise fixture |
| `windef.simple_multipass` parity | **0.00e+00** | 1e-9 | bit-identical after undoing their y-flip/v-negation |

Upstream's shipped tolerances are loose (0.25 px, best-of-N-trials logic in
`test_process.py`); the bounds above were set from the measured floor instead, per this
repo's validation policy.
