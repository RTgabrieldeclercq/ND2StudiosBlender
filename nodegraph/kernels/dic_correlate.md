# `dic_correlate` — DIC (pyALDIC) integration contract

## 1. Purpose + where the real math lives

Runs **2D Augmented-Lagrangian Digital Image Correlation** on an already-prepared
reference/deformed image pair (or ordered series) and returns a dense
displacement field resampled onto a regular grid as a `DVCResult`.

**The real correlation math is EXTERNAL.** Local IC-GN subset matching + the
global ADMM solve over an *adaptive quadtree finite-element mesh* live entirely
inside the third-party **`al-dic` (pyALDIC)** package
(`al_dic.core.pipeline.run_aldic`), which this module imports **lazily**.

The **in-repo math** vendored here is only the thin shell around it:
- **input adapter** — a lazy `FrameProvider` that hands al_dic one frame at a time and
  lets it z-score them itself; map param dict → an `al_dic` `DICPara` with pow2/even
  snapping and (with a mask) an ROI-bbox correlation range; rasterize an ROI mask.
- **output adapter** — de-interleave pyALDIC's `U=[u,v,...]` onto either the FE mesh
  nodes or a regular grid, applying the load-bearing `[x,y]→[y,x]` / `[u,v]→[dy,dx]`
  axis swap **and the cross-term sign fix on the strain tensor** (§6).

> **Validated 2026-07-31** against pyALDIC's own synthetic suite, replicated in
> `scripts/dic_synthetic_bench.py`: 11 of 12 cases land inside upstream's own tolerances,
> matching or beating a direct `run_aldic()` call. See §12.

## 2. Entry points

```python
run_pyaldic_pair(
    ref_img: np.ndarray,              # (H, W)
    def_img: np.ndarray,              # (H, W)
    voxel_size_um: Tuple[float, ...], # (y, x)
    params: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    *,
    roi_mask: Optional[np.ndarray] = None,      # (H, W)
    refinement: Optional[Dict[str, Any]] = None,
    resample: str = "grid",                     # "grid" | "nodes"  (§5)
) -> DVCResult                                   # cumulative field for def vs ref

run_pyaldic_series(
    images: Sequence[np.ndarray],               # [ref, def1, def2, ...] each (H,W)
    masks: Optional[Sequence[np.ndarray]],      # per-image (H,W) or None
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, ...],           # (y, x)
    *,
    reference_mode: str = "accumulative",       # "accumulative" | "incremental"
    refinement: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    resample: str = "grid",                     # "grid" | "nodes"  (§5)
) -> List[Dict[str, DVCResult]]                  # len == len(images) - 1

# Helper (optional): build a boolean ROI mask from serializable vector shapes.
build_roi_mask(shapes: Optional[Sequence[Dict]], H: int, W: int) -> np.ndarray  # (H,W) bool

# Availability probe (no import side effects):
al_dic_available() -> bool
```

**Call `run_pyaldic_series` ONCE for a series — never loop `run_pyaldic_pair`.** al_dic
caches its reference bundle, subpb1 precompute and 6-DOF IC-GN context keyed on the
reference frame index, and remembers the FFT search radius that worked. A pair-per-frame
loop discards all of it: measured **1.8× slower at both 256²/T=9 and 1024²/T=5**, for
**bit-identical** fields. End to end through `analysis.dic_correlate` the same change is
2.3–3.2× (it also drops per-plane copies and the resample).

**`init_guess_mode` matters here.** `"fft"` (what the node passes) gives every frame its
own coarse integer search. `"auto"` is mapped by al_dic to `"previous"`, warm-starting each
frame from the last solution — marginally faster on smoothly-growing motion, but it cannot
find a large FIRST step, and in `accumulative` mode a leading self-pair solves to ~0 and
hands that forward. Measured: an 11 px shift came back as a 6.3 px error under `"auto"` and
0.003 px under `"fft"`.

`images` is **indexed lazily** — anything with `__len__`/`__getitem__` works, so a caller
with a plane reader can stream a long series at ~two planes of memory instead of the whole
`T×H×W` float64 stack.

## 3. Inputs

| name | Python type | shape | dtype | axis order | units | required? | meaning & constraints |
|------|-------------|-------|-------|------------|-------|-----------|-----------------------|
| `ref_img` / `def_img` | `np.ndarray` | `(H, W)` | any numeric (uint16/float…) | `[y, x]` | intensity | required | Reference & deformed grayscale frames, same shape. A `(C,H,W)` array is auto-collapsed (max if ≤4 chans, else mid slice). Normalized internally to float64 `[0,1]`. |
| `images` | `Sequence[np.ndarray]` | each `(H, W)` | any numeric | `[y, x]` | intensity | required | `images[0]` is the reference; ≥2 required. |
| `masks` | `Sequence[np.ndarray]` or `None` | each `(H, W)` | numeric/bool | `[y, x]` | — | optional | Per-image ROI. `None` → all-ones (no masking). Cast to **float64** at the al_dic boundary. |
| `roi_mask` (pair) | `np.ndarray` or `None` | `(H, W)` | numeric/bool | `[y, x]` | — | optional | Single mask applied to BOTH ref & def. |
| `voxel_size_um` | `Tuple[float, ...]` | len 2 | float | `(y, x)` | µm/px | required | Physical pixel size; stored on the result, applied by `DVCResult.displacement_um()`. **Not** applied to the raw field. |
| `refinement` | `Dict` or `None` | — | — | — | optional | Adaptive-mesh refinement spec (see §4); best-effort, silently ignored if al_dic lacks the module. |

## 4. Parameters (`params` dict)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `winsize` | int | 40 | ≥2, forced **even** | IC-GN subset size (px). |
| `winstepsize` | int | 16 | snapped to nearest **power of 2**, ≥2 | Grid pitch between subset centers (px). Becomes the output grid `step`. |
| `winsize_min` | int | 8 | snapped **pow2**, clamped ≤ `winstepsize` | Min adaptive element size. |
| `init_guess_mode` | str | `"auto"` | al_dic-defined (e.g. `auto`) | Initial-guess strategy for IC-GN. |
| `mu` | float | 1e-3 | >0 | ADMM penalty weight. |
| `tol` | float | 1e-2 | >0 | Convergence tolerance. |
| `admm_max_iter` | int | 3 | ≥1 | ADMM outer iterations. |
| `icgn_max_iter` | int | 100 | ≥1 | IC-GN inner iterations. |
| `disp_smoothness` | float | 5e-4 | ≥0 | Displacement regularization — **AL-DIC only** (read solely inside al_dic's `if para.use_global_step:` block). Response is sharp and the default is below its onset: 0 and 5e-4 are indistinguishable; 5e-3 cuts noise-case error 3× and blurs a 2% stretch field 50× worse. |
| `strain_smoothness` | float | 1e-5 | ≥0 | Strain regularization. Read by the global step **and** by the strain path, so it is live whenever either is on. |
| `size_of_fft_search_region` | int | 20 | ≥1 | **Starting** ± radius (px) of the FFT integer search. al_dic auto-grows it on boundary clipping (≤6 retries, `max(needed, 2×current)`, capped at `max(32, min(H,W)//2)`) and remembers the learned radius per reference — so a wrong value costs search passes, not correctness. Also silently clamped when `2·search + winsize > min(H,W)/4`. |
| `use_global_step` | bool | True | — | `True` = AL-DIC (local IC-GN + global ADMM/FEM). `False` = plain **Local DIC**: ~2× faster and no less accurate on smooth fields (upstream case9). |
| `compute_strain` | bool | True | — | Passed to `run_aldic`; when on, `strain_field` is filled from al_dic's FEM `StrainResult` (§5, §6). Costs ~20%. |
| `reference_mode` (series arg) | str | `"accumulative"` | `accumulative` \| `incremental` | Cumulative-vs-per-step referencing (native to pyALDIC). |
| `resample` (kwarg) | str | `"grid"` | `grid` \| `nodes` | Output geometry — see §5. |

`refinement` dict shape: `{"criteria": {"mask_boundary": bool, "roi_edge": bool, "brush": bool}, "brush": <(H,W) array>, "min_element_size": int}`.

## 5. Output — `DVCResult` (dataclass)

For a pair: one `DVCResult`. For a series: `List[Dict[str, DVCResult]]`, one dict
per deformed frame, keys `"primary"` (cumulative — from pyALDIC `U_accum`, falling
back to `U`) and `"increment"` (per-step `U`).

`resample` chooses the output geometry. `G` below is `(Gy, Gx)` for `"grid"` (a regular
lattice of pitch `winstepsize`, the historical contract) and `(N,)` for `"nodes"` (the FE
mesh nodes themselves).

| field | shape | dtype | axis order | units | meaning |
|-------|-------|-------|------------|-------|---------|
| `dim` | scalar | int | — | — | always `2`. |
| `grid_coords` | `(*G, 2)` | float64 | `[y, x]` | pixels (downsampled) | Node coordinates; `[...,0]=y`, `[...,1]=x`. |
| `displacement_field` | `(*G, 2)` | float64 | `[y, x]` | pixels | `[...,0]=dy`, `[...,1]=dx`. Convert to µm via `.displacement_um()`. **Non-finite at nodes the solver could not resolve** (masked-out subsets) — that is the honest answer, not a defect. |
| `strain_field` | `(*G, 2, 2)` or `None` | float64 | `[y, x]` | — | `[i, j] = ∂disp_i/∂axis_j`, from al_dic's FEM nodal gradients. `None` unless `compute_strain=True`. **Cross terms are sign-corrected — see §6.** |
| `voxel_size_um` | len-2 tuple | float | `(y, x)` | µm/px | as supplied. |
| `qfactor` | — | — | — | — | `None`. al-dic 0.7.2 keeps its per-point NCC quality factors inside `solver/integer_search.py` (`fft_info["cc_max"]`) and never places them on `PipelineResult`, so there is no honest per-node confidence to report without patching the solver. |
| `converged` | scalar | bool | — | — | `False` for frames at/after an early stop (`PipelineResult.stopped_at_frame`), else `True`. |
| `iterations` | scalar | int | — | — | ADMM steps actually configured for the solver used — `1` in Local-DIC mode. |
| `mu` | scalar | float | — | — | echoed from params. |
| `method` | str | — | — | — | `"pyALDIC"`. |
| `notes` | str | — | — | — | `"cumulative"` or `"increment"`. |
| `diagnostics` | dict | — | — | — | `engine`, `n_nodes`, `grid_shape`, `winsize`, `winstepsize`, `reference_mode`, `resample`, `solver`, `has_strain`. |

Convenience: `.magnitude` `(Gy,Gx)` in px; `.displacement_um()` scales by
`voxel_size_um`; `.magnitude_um()` in µm.

## 6. Conventions & GOTCHAS (real integration risk)

- **Axis swap is LOAD-BEARING.** pyALDIC uses node coords `[x, y]` and
  interleaved `U=[u0,v0,u1,v1,...]` with `u = x-displacement`, `v = y-displacement`.
  `_frame_to_dvcresult` de-interleaves and **swaps to `DVCResult` `[y,x]`/`[dy,dx]`**.
  If you bypass this adapter and read `U` directly, remember it is x-major.
- **THE STRAIN CROSS TERMS ARE SIGN-FLIPPED UPSTREAM, and this adapter negates them
  back.** al_dic reports its y-derivative and v-component under a y-**up** convention
  while its *displacements* are plain image-row (y-down), so `dudy` and `dvdx` — the terms
  with an odd number of y/v factors — come back with the opposite sign to `dudx`/`dvdy`.
  Measured on three analytic fields: for `u = 0.015·(y−cy)` simple shear al_dic returns
  `dudy = −0.01494` where the truth is **+0.015**; for a 2° rotation it returns
  `dudy = +0.03486` / `dvdx = −0.03486` for truths **−0.03490** / **+0.03490**; both
  diagonals are correct. Upstream's own tests cannot detect this — `test_synthetic.py`
  checks case5_shear with `strain_tol=0.04` and case10_rotation with `strain_tol=0.08`, and
  in both the tolerance exceeds *twice* the signal, so a fully inverted cross term passes.
  If you bypass `_strain_components` and read `StrainResult` directly, negate them yourself.
- **Masks are float64 at the al_dic boundary, NOT bool.** They are cast via
  `.astype(np.float64)`.
- **`DICPara.use_masks` is INERT in al-dic 0.7.2.** The field is declared at
  `core/data_structures.py:288` and referenced *nowhere else in the package*; real masking
  flows through the `masks` list, which `run_aldic` installs as `para.img_ref_mask` per
  frame (`core/pipeline.py:1004`). This adapter still computes
  `use_masks = any(mask.min() < 1.0)` because it is the honest value **and** because it is
  the "is there a real mask?" predicate that gates the ROI-bbox tightening below. Do not
  read it as a solver switch.
- **A mask TIGHTENS the correlation range to the mask bbox**, matching upstream's own
  `roi_range_from` (`examples/batch_process.py:131/221`). Consequence: the grid **origin
  moves** when a mask is present, so node positions differ from an unmasked run. They are
  subset centres, not landmarks — this changes sampling, not measurements. Untightened, the
  solver grids the whole frame and NaNs everything outside the ROI: on a disc ROI over ~28%
  of a 512² frame that was 68% of the emitted points.
- **`um2px` is forced to `1.0`.** Displacement stays in (downsampled) **pixels**;
  the µm conversion is deferred to `DVCResult.displacement_um()` via
  `voxel_size_um`. Do not double-apply pixel size.
- **Grid ≠ image.** The output grid spans only the FE-mesh node bounding box
  (ROI-limited), with pitch = the snapped `winstepsize`. `grid_coords` are in
  the (possibly downsampled) pixel space the caller fed in.
- **Snapping.** `winstepsize`/`winsize_min` are snapped to powers of two;
  `winsize` is forced even. Your requested values may be silently adjusted —
  read them back from `diagnostics`.
- **The `"grid"` resample is an exact index scatter, not interpolation, for a uniform
  mesh.** al_dic's default `mesh_type="uniform"` mesh IS the regular lattice
  `_regular_grid` reconstructs, so `_lattice_index` detects the coincidence and permutes.
  `griddata` (linear + nearest hole-fill) survives only as the fallback for a genuinely
  irregular quadtree-refined mesh — where note that resampling onto the coarse lattice
  **discards exactly the refined nodes the refinement produced**. Use `resample="nodes"`
  whenever you care about adaptive refinement, or whenever your consumer is a point table
  rather than an image.
- **`run_pyaldic_series` returns `len(images) - 1` entries** (no self-pair for
  the reference).
- **CALLER OWNS ALL PREP** — no file I/O, no multipoint/timepoint looping, no
  crop/downsample/registration/exclusion here. Feed final `(H,W)` arrays.

## 7. Dependencies

| package | import-time or lazy | why |
|---------|---------------------|-----|
| `numpy` | import-time | arrays throughout. |
| `scipy` | **lazy** (inside `_interp_to_grid`) | `scipy.interpolate.griddata` scattered→regular resample. Needed whenever a result is adapted. |
| `scikit-image` | **lazy** (inside `_rasterize_region`) | `skimage.draw` disk/ellipse/polygon — only if you use `build_roi_mask`. |
| `al-dic` (pyALDIC) | **lazy** (inside `_require_al_dic` / `_refinement_policy`) | the actual DIC solver. **Transitively pulls `numba` and `PySide6>=6.6`.** Absent → friendly `ImportError` only when you call `run_pyaldic_*`; module import still succeeds. |

Install the solver: `pip install al-dic`.

## 8. Failure modes / edge cases

- **`al-dic` not installed** → `ImportError` with install hint, raised only on
  `run_pyaldic_pair`/`run_pyaldic_series` call (import & adapters still work).
- **< 2 images** → `ValueError("AL-DIC needs at least 2 images …")`.
- **User cancel** (`cancelled_cb()` truthy) → surfaces as `RuntimeError` /
  empty series; pair convenience raises `RuntimeError` if no result produced.
- **Flat image** (`std < 1e-12` over the ROI) → al_dic's `normalize_one` substitutes
  `std = 1.0`, so the frame stays constant and the solver has nothing to track.
- **Degenerate `U`** (`u.size < 2*n`) → per-node displacement defaults to 0.
- **Empty/missing ROI shapes** → `build_roi_mask` returns all-False; caller
  decides whether "no ROI" means full frame. An all-empty mask list leaves the range
  untightened (`_mask_bbox` → `None`) rather than producing a zero-size ROI.
- **Bad `refinement` policy** → swallowed, treated as no refinement.
- **Frames of differing shape** → `_LazyFrameProvider.get_normalized` raises `ValueError`
  naming the offending frame index, rather than letting al_dic fail obscurely later.
- **Unsolved nodes** → non-finite entries in `displacement_field`. Callers building a
  point table should DROP them; `analysis.dic_correlate` does, and reports the count as
  `dic_unsolved_points`.

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np, dic_correlate as d

H = W = 256
ref = (np.random.rand(H, W) * 4095).astype(np.uint16)
# a synthetic 3-px x-shift as the "deformed" frame:
defo = np.roll(ref, 3, axis=1)

# requires al-dic installed; otherwise raises a friendly ImportError:
res = d.run_pyaldic_pair(ref, defo, voxel_size_um=(0.5, 0.5),
                         params={"winsize": 40, "winstepsize": 16})
print(res.grid_coords.shape)          # (Gy, Gx, 2)
print(res.displacement_field.shape)   # (Gy, Gx, 2)  -> [dy, dx] in px
print(res.displacement_um().shape)    # same, scaled by voxel_size_um
```

Adapter-only check (no al-dic needed) — verifies the axis swap:

```python
class M: coordinates_fem = np.array([[0,0],[10,0],[0,10],[10,10],[5,5]], float)
U = np.zeros(10); U[0::2] = 1.0; U[1::2] = 2.0   # u(x)=1, v(y)=2
r = d._frame_to_dvcresult(M(), U, step=5, voxel_size_um=(1,1),
                          method="pyALDIC", params={}, reference_mode="accumulative")
# r.displacement_field[...,0] ~= 2 (dy),  [...,1] ~= 1 (dx)
```

## 10. Pipeline wiring (original ND2Studios)

- **Feeds in:** the caller (page/plane-runner) supplies final `(H,W)` frames
  after channel selection, optional downsample, crop, and registration; an
  optional ROI mask (built from drawn vector shapes via `build_roi_mask`); and
  the ND2 `pixel_size_um` as `voxel_size_um`. A "refine" node may supply the
  `refinement` dict.
- **Feeds out:** the `DVCResult` (`(Gy,Gx,2)` displacement in px + `voxel_size_um`)
  goes to the DVC panel / downstream strain & visualization, which derive strain
  from the displacement field and convert to µm on demand.

## 11. Provenance (branch `Version-1.45`)

- `nd2studios/backend/dic/engine.py` — `run_pyaldic_pair`, `run_pyaldic_series`,
  `_frame_to_dvcresult`, `_interp_to_grid`, `_regular_grid`, `_mesh_coords`,
  `_build_dicpara`, `_to_float01`, `_snap_pow2`, `_refinement_policy`,
  `_require_al_dic`, `al_dic_available`, `_INSTALL_HINT` (verbatim).
- `nd2studios/backend/dic/roi.py` — `build_roi_mask`, `_rasterize_region`,
  `has_region`, `OP_ADD`, `OP_CUT` (verbatim).
- `nd2studios/core/dvc_registry.py` — `DVCResult` dataclass only (verbatim).

**Dropped (UI/registry-only, not on the compute path):** `DVCMethod` ABC,
`DVCParams`, the `@DVCMethod.register` registry, and the `ParamSpec` import from
`dvc_registry`. **No helper renames** were needed (no name collisions).

## 12. Validation against pyALDIC's own synthetic suite (2026-07-31)

`scripts/dic_synthetic_bench.py` replicates upstream
`tests/test_integration/test_synthetic.py` + `tests/conftest.py` case-for-case — 256²
speckle (Gaussian-filtered noise, σ=3, seed 42, values in [20, 235]) deformed by the
Lagrangian fixed-point warp with order-5 quintic B-splines — and adds sub-pixel, large-
displacement, noise and spatial-resolution probes upstream does not carry.

Interior nodes only (32 px edge margin, upstream's `compute_disp_rmse_interior`), our
node's default solver settings, worst of RMSE_u/RMSE_v in px:

| case | ours | upstream tol | note |
|------|------|--------------|------|
| zero | 0.0037 | 0.01 | the noise floor |
| translation (2.5, −1.8) | 0.0046 | 0.03 | |
| affine 2% biaxial | 0.0045 | 0.05 | |
| shear 1.5% | 0.0024 | 0.05 | |
| rotation 2° | 0.0039 | 0.05 | |
| large_deform 10%+5% | 0.813 | 1.0 | upstream expects a large residual here |
| local_only (ADMM off) | 0.0041 | 0.05 | 2.4× faster than AL-DIC, same accuracy |
| sub-pixel 0.50 / 0.25 px | 0.0039 / 0.0041 | 0.03 | no S-curve bias |
| 11 px translation | 0.0034 | 0.05 | FFT search range |
| translation + 2% noise | 0.0386 | 0.10 | |
| sinusoid, λ=64 px | 0.395 | — | *our* probe; see below |

**11 of 12 inside upstream's own tolerances**, and our adapter matches or beats a direct
`run_aldic()` call on nearly every case (`--upstream` reproduces that comparison).

Two caveats worth knowing:

1. **Upstream's quoted ~0.005 px RMSEs are a converged-from-truth floor.** Every case in
   `test_synthetic.py` passes `U0=` the *exact* ground-truth displacement at every mesh
   node plus a pre-built `mesh=`. Real data has no such oracle, so the table above is the
   honest FFT-seeded number. `--u0` reproduces the oracle path.
2. **The sinusoid case is physics, not a defect** — a subset of 40 px spans 62% of a 64 px
   wavelength, so subset averaging necessarily attenuates it (a direct upstream call with
   `winsize=32` gets 0.295 the same way). It is in the bench as the **spatial-resolution**
   probe: error falls monotonically with subset size (0.097 at 16 px → 0.516 at 48 px)
   while noise robustness improves in the opposite direction (0.100 → 0.029), and subset
   size is very nearly free in wall-clock. That trade-off is the reason `winsize` is the
   node's most consequential knob.

**DEVIATION from verbatim (2026-07-27, al-dic ≥0.7 compat):** `_build_dicpara` now sets
`gridxy_roi_range` to the full image extent `(gridx=(0, W), gridy=(0, H))`. al-dic 0.7.x
requires this ROI range EXPLICITLY when `run_aldic()` is called directly — it defaults to a
zero-size box and otherwise raises `ValueError("No grid points generated … ROI is empty")`.
The solver insets the range by `winsize//2` (`integer_search`: `min_x = max(gridx[0], w//2)`,
`max_x = min(gridx[1], W-1-w//2)`) and applies `use_masks` itself, so the full-image default is
correct and the mask still restricts correlation. A future ROI-bbox tightening (grid limited to
the mask's bounding box) is an optional optimization. Verified: recovers a planted (1,3)px shift
on a 128² speckle pair (`nodegraph.selftest:test_catalog_dic`).
