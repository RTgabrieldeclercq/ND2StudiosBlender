# `aldvc_field` — DVC (pyALDVC) integration contract

> **3D ONLY.** pyALDVC cannot correlate a single plane. 2D image correlation is
> [`dic_correlate`](dic_correlate.md) (the pyALDIC sibling package), exposed as
> `analysis.dic_correlate`.

## 1. Purpose + where the real math lives

Runs the **official pyALDVC** Augmented-Lagrangian Digital Volume Correlation solver
over an ordered stack of 3D volumes and packages each frame pair as a `DVCResult` on
a regular subset grid: dense displacement (voxels) + a strain tensor + a per-subset
ZNCC.

**All of the correlation math is EXTERNAL**, in the `al-dvc` package
(`al_dvc.core.pipeline.run_aldvc`), imported **lazily**:

| | |
|---|---|
| package | `al-dvc` on PyPI, import name `al_dvc` |
| source | <https://github.com/zachtong/pyALDVC> |
| docs | <https://zachtong.github.io/pyALDVC/> |
| DOI | 10.5281/zenodo.22883767 |
| verified against | al-dvc **1.2.0**, numpy 2.4.6, scipy 1.18.0, numba 0.66.0 |

It implements MATLAB `main_ALDVC.m` Sections 2–8: node grid → pyramid-NCC initial
guess → 12-DOF local IC-GN (subproblem 1) → global compatibility solve (subproblem 2)
with L-curve β auto-tuning → ADMM outer loop → strain. Same author as pyALDIC.

**The in-repo math is ONLY the adapter:**

* *input* — node params → an `al_dvc` `DVCPara` with the `(z,y,x)`→`(x,y,z)` reversal
  and even/≥4 subset snapping (`build_dvcpara`); a lazy `VolumeProvider` over a
  caller-supplied frame getter (`LazyVolumeProvider`).
* *output* — the **load-bearing `[x,y,z]`→`[z,y,x]` reversal** of node coordinates,
  displacement and the strain tensor, reshaped from pyALDVC's flat `(N, …)` node
  arrays onto the `(Gz,Gy,Gx, …)` grid this repo's Point flattener expects
  (`frame_to_dvcresult`, `_strain_tensor_zyx`).

### What this replaced (2026-09-25)

This module used to be a **2998-line clean-room numpy/scipy port** of FranckLab's
MATLAB ALDVC. It was replaced wholesale: pyALDVC is the maintained implementation,
it is numba/CUDA accelerated, and it carries its own validation against the same
MATLAB reference. The port's dimension-agnostic leftovers that other nodes depend on
(`DVCResult`, the subset `Grid`, the strain-measure stack, Lagrangian accumulation)
moved to [`field_math`](field_math.md); nothing else survived.

## 2. Entry points

```python
run_aldvc_series(get_volume, n_frames, shape, *, voxel_size_um, params=None,
                 ref_indices=None, compute_strain=True, cumulative=False,
                 progress_cb=None, stop_cb=None) -> list[DVCResult]
```

The one you want. Correlates the whole ordered stack in **one** `al_dvc.run_aldvc`
call and returns `n_frames - 1` results, for frames `1 … n_frames-1`.

One call rather than N pair calls matters: pyALDVC builds the node grid once, caches
each reference frame's bundle (normalized volume + its three gradient volumes) across
every frame that references it, and auto-tunes β once per reference rather than once
per pair. On an accumulative schedule that is a single reference bundle for the
entire series.

```python
run_aldvc(ref_vol, def_vol, *, voxel_size_um=(1,1,1), params=None,
          compute_strain=True, progress_cb=None) -> DVCResult
```

Convenience for a single pair. Raises if either volume is not 3-D, naming
`dic_correlate` as the 2D route.

```python
al_dvc_available() -> bool        # is the optional dep importable?
build_dvcpara(params, voxel_size_um, *, n_frames=2, ref_indices=None) -> DVCPara
frame_to_dvcresult(mesh, frame, strain, voxel_size_um, *, ...) -> DVCResult
normalize_strain_type(name) -> str        # 'green-lagrange' -> 'green_lagrange'
LazyVolumeProvider(get_volume, n_frames, shape, *, voi=None, cache_size=3)
```

## 3. Inputs

| arg | shape / type | notes |
|---|---|---|
| `get_volume(i)` | `(nz, ny, nx)` array | any dtype; **every frame must have the same shape**. Frame 0 is the first reference. |
| `n_frames` | int ≥ 2 | |
| `shape` | `(nz, ny, nx)` | asserted against each fetched volume |
| `voxel_size_um` | `(z, y, x)` floats | **slowest-first**, this repo's order. Positive and finite. |
| `ref_indices` | `(n_frames-1,)` ints or None | `ref_indices[i]` is the reference for deformed frame `i+1`; must satisfy `0 <= ref_indices[i] <= i`. `None` falls back to `params["reference_mode"]`. |

The caller owns all preparation: no file I/O, no m/t looping beyond the stack you
hand it, no crop/downsample/registration/exclusion. Normalization IS done for you —
pyALDVC's own `normalize_volume` (z-score over the VOI), so the numbers the solver
sees match what it would compute from a plain list of volumes.

**Masks are not wired.** `LazyVolumeProvider.get_mask` returns `None`. pyALDVC's mask
path also drives `subset_split` and the VOI, which the node does not expose; crop
before you call.

## 4. Parameters (`params` dict — pyALDVC defaults in parentheses)

Every key is optional; an empty dict reproduces `dvcpara_default()`. Names are this
repo's; the `DVCPara` field each maps to is given.

| key | → `DVCPara` | default | meaning |
|---|---|---|---|
| `subset_size` | `winsize[x,y]` | 32 | subset edge, **voxels**. Snapped UP to even ≥4 — `DVCPara` refuses anything else. Upstream's rule: ≥ 4–5× the speckle/pore diameter. |
| `subset_size_z` | `winsize[z]` | 0 = same as lateral | axial subset edge; set it for anisotropic stacks (upstream's own advice, e.g. 32/32/16) |
| `subset_spacing` | `winstepsize[x,y]` | 16 | node spacing, voxels. Half the subset for smooth fields, a quarter when chasing gradients. |
| `subset_spacing_z` | `winstepsize[z]` | 0 = same as lateral | |
| `search_radius` | `search_radius` | 0 = **leave to solver** (8) | NCC half-width at the coarsest pyramid level; auto-expands on clipped peaks |
| `init_guess` | `init_guess_method` | `pyramid` | `pyramid` / `ncc` / `zero` / `previous` |
| `global_shift` | `global_shift` | True | rigid whole-volume pre-shift by phase correlation |
| `init_coarse_factor` | `init_coarse_factor` | 1 | >1: seed on every k-th node, interpolate U **and** F to the rest |
| `prefilter_sigma` | `prefilter_sigma` | 0.0 | Gaussian pre-smoothing, voxels (0.6–1.0 for SNR < 5) |
| `interp_method` | `interp_method` | `cubic` | `cubic` (Keys = MATLAB `ba_interp3`) / `bspline` / `linear` |
| `icgn_max_iter` | `icgn_max_iter` | 100 | per-subset iteration cap |
| `subset_stride` | `subset_stride` | 1 | sample every k-th subset voxel; **clamped** so ≥5 samples per axis remain |
| `use_global_step` | `use_global_step` | True | False = plain local subset DVC |
| `admm_iterations` | `admm_max_iter` | 4 | ≥1. Published guidance: 3–5 suffice. |
| `mu` | `mu` | 1e-3 | ADMM penalty; upstream says it rarely needs changing |
| `beta` | `beta` | 0.0 = **auto** | >0 pins it; 0 → `None` → L-curve sweep per reference frame |
| `disp_smoothing` | `disp_smoothing` | 0.0 | Gaussian σ in **node** units |
| `strain_smooth` | `strain_smoothing` | 0.0 | Gaussian σ in **node** units |
| `strain_method` | `strain_method` | `plane_fit` | `plane_fit` / `fem` / `fd` / `direct` |
| `strain_type` | `strain_type` | `infinitesimal` | `infinitesimal` / `green_lagrange` / `euler_almansi` / `hencky`; hyphenated and legacy spellings accepted via `normalize_strain_type` |
| `strain_halfwidth` | `strain_plane_fit_halfwidth` | 1 | plane-fit window half-width in nodes (1 = 3×3×3) |
| `backend` | `backend` | `auto` | `auto` / `numba` / `numpy` / `cuda` |
| `n_threads` | `n_threads` | 0 = all cores | numba thread count (in-process; **no** spawn hazard) |
| `tile_local` | `tile_local` | 0 = off | solve in boxes of this edge, in voxels, to bound memory |
| `reference_mode` | `reference_mode` | `accumulative` | only read when `ref_indices` is None |

`voxel_size_um` is passed through to `DVCPara.voxel_size` (reversed to `(x,y,z)`) and
`units` is set to `"um"` — upstream warns if a scaled voxel size is still labelled
`"voxel"`.

## 5. Output — `DVCResult` (dataclass)

`d = 3` always. `(Gz, Gy, Gx)` is the solver's node grid, **not** the image grid.

| field | shape | meaning |
|---|---|---|
| `dim` | — | always 3 |
| `grid_coords` | `(Gz,Gy,Gx,3)` | subset-centre coordinates in **voxels**, `[..., 0] = z` |
| `displacement_field` | `(Gz,Gy,Gx,3)` | displacement in **voxels**, `[..., 0] = dz` |
| `strain_field` | `(Gz,Gy,Gx,3,3)` | **symmetric** strain tensor, `[i,j]` over axes `(z,y,x)`, in **physical** units |
| `strain_type` | str | the resolved `DVCPara.strain_type` |
| `qfactor` | `(Gz,Gy,Gx)` | per-subset ZNCC |
| `converged` | bool | True only if EVERY node's final status is `converged` |
| `iterations` | int | ADMM steps actually run |
| `mu`, `beta` | float | the values used (β is the auto-tuned one when auto) |
| `method` | str | `"pyALDVC (al-dvc)"` |
| `diagnostics` | dict | `engine`, `ref_frame`, `n_nodes`, `grid_shape`, `n_converged`, `n_nodes_bad`, `n_outlier`, `median_zncc` |

`DVCResult.displacement_um()` applies `voxel_size_um`; the node's Point flattener
(`catalog/_shared/dvc.py::_dvc_rows`) does the same multiply.

## 6. Conventions & GOTCHAS (the real integration risk)

### 6a. THE AXIS REVERSAL — the single load-bearing adapter

pyALDVC and this repo order axes **oppositely**. Everything in `frame_to_dvcresult`
exists to bridge it.

| quantity | pyALDVC native | what this kernel returns |
|---|---|---|
| volume array | `(nz, ny, nx)` | same — no change |
| `DVCPara` triples | `(x, y, z)` | caller passes `(z, y, x)` |
| node coordinates | `(N,3)` `[x, y, z]` | `(Gz,Gy,Gx,3)` `[z, y, x]` |
| displacement `U` | `(N,3)` `[u, v, w]` | `(Gz,Gy,Gx,3)` `[dz, dy, dx]` |
| gradient `F[i,j]` | `du_i/dx_j`, `i,j ∈ xyz` | strain `[i,j]`, `i,j ∈ zyx` |

The reversal is a full `[::-1]` on the component axis — and on **both** tensor axes
for strain. **Reversing one and not the other transposes the strain tensor, which is
invisible on any symmetric fixture.** That is why `scripts/_aldvc_validate.py`
group B uses an *asymmetric* gradient (`du_x/dy = 0.030`, `du_y/dx = 0.010`).

`mesh.grid_shape` is already `(nz, ny, nx)` node counts with node
`n = iz*ny*nx + iy*nx + ix`, so a plain C-order reshape lands on the right grid — the
**component** axis is the only thing that needs reversing, never a transpose of the
grid itself.

### 6b. pyALDVC is NOT pyALDIC on signs — do not "fix" it

`dic_correlate` has to **negate** pyALDIC's two 2D strain cross-terms. **pyALDVC does
not need this**, and adding it by analogy with the sibling package would introduce a
bug. Measured 2026-09-25 against analytic truth on a 72³ bead volume:

| fixture | truth | pyALDVC returns |
|---|---|---|
| translation `t=(+2,+1,-1.5)` in `(x,y,z)` | — | `U` median `[+1.9999, +1.0000, -1.5001]` |
| simple shear `du/dy=+0.02` | `F[0,1]=+0.02`, `F[1,0]=0` | `+0.0202`, `+1e-5` |
| **antisymmetric** `du/dy=+0.02, dv/dx=-0.02` | `+0.02`, `-0.02` | `+0.0202`, `-0.0202` |

The antisymmetric fixture is the one that would expose a negation or a transpose, and
it exposes neither. Its rigid-rotation reading is the standing regression: a rotation
must give **zero** infinitesimal shear strain, so a single negated cross-term would
read `±g` instead of `~0`.

### 6c. Tensor shear, not engineering shear

`StrainResult.exy/exz/eyz` are **tensor** shear (½ the engineering shear): the
`du/dy = +0.02` fixture reports `exy = +0.0101`. That matches
`field_math.strain_from_gradient`'s `½(G+Gᵀ)`, so the two are interchangeable and the
Point columns mean what they always did.

### 6d. Displacement is voxels, strain is physical — deliberately

`displacement_field` comes from `FrameResult.U`, which stays in **voxels** whatever
`voxel_size` is. `strain_field` comes from `StrainResult`, which pyALDVC computes in
**physical** units (`scale_to_physical(U, F, para.voxel_size)`).

This split is load-bearing for anisotropic voxels. A confocal z-step of 0.5 µm against
a 0.1 µm xy pixel makes every off-diagonal strain term wrong by the 5× anisotropy
ratio unless the solver is told the voxel size — so `voxel_size_um` **is** passed
through, and strain comes back dimensionless-correct. Validated: with
`voxel_size_um=(0.5,0.2,0.2)`, `strain[x,z]` scales by exactly `v_x/v_z = 0.400`
while `strain[x,y]` (both axes lateral) is unchanged, and displacement does not move
at all.

### 6e. Geometry constraints that reject a volume outright

Three separate rules, each raising from a different place upstream:

1. `al_dvc.utils.validation` — **every axis must be ≥ 16 voxels**.
2. same — **`winsize[axis] + 11 ≤ extent[axis]`** (subset half-width plus a 5-voxel
   margin each side).
3. `al_dvc.mesh.grid_mesh.build_grid_axes` — **≥ 2 nodes per axis**, i.e.
   `extent - winsize - 10 ≥ winstepsize`, where the border is
   `GRADIENT_BORDER + INTERP_MARGIN = 5`.

A shallow confocal stack hits (1) and (3) first. The fix is `subset_size_z` /
`subset_spacing_z`, which is why those sockets exist. The smallest fixture that
satisfies all three and still yields a 3×3×3 grid is `(28, 44, 44)` with subset
12 / axial 8 and step 10 / axial 4 — that is exactly the selftest's fixture.

### 6f. `beta = 0` means auto, and `search_radius = 0` means "solver's default"

`DVCPara` refuses a non-positive `beta` and treats `None` as "auto-tune by L-curve",
so this repo spells auto as `0`. Likewise `search_radius = 0` would make the NCC seed
a single-point lookup, so `0` is mapped to "omit the key" and upstream's 8 applies.
Neither can be passed through literally.

### 6g. Threads, not processes

The old port forced `n_workers=1` because it used a process pool and embedding it
risked a spawn/import hazard. pyALDVC uses **numba's in-process thread pool**, so
`n_threads=0` (all cores) is safe inside the engine and is the default. There is no
spawn hazard to avoid any more.

### 6h. First call in a process is slow

Numba compiles on first use. Measured on the selftest fixture: **2.30 s** cold,
**0.02 s** warm — the JIT cache is on disk, so it is a per-machine one-off, not a
per-process one. `al_dvc.warmup()` exists if you want to pay it at start-up.

## 7. Dependencies

| pip name | import | required? |
|---|---|---|
| `al-dvc` | `al_dvc` | **optional, lazily imported** |
| `numpy` | `numpy` | yes (top-level) |

Importing this module never imports `al_dvc`: the catalog loads on a machine without
it and only a *pull* fails, with `INSTALL_HINT`. Install with `pip install al-dvc`,
or `pip install "al-dvc[gpu]"` for the CUDA local solver. `al-dvc` itself pulls
numba, scipy, h5py, tifffile, pyyaml, matplotlib and — for its own desktop app, not
for this kernel — pyvista/pyvistaqt/vtk.

## 8. Failure modes / edge cases

| situation | behaviour |
|---|---|
| `al_dvc` not installed | `RuntimeError(INSTALL_HINT)` naming the package and the install line |
| 2-D input to `run_aldvc` | `ValueError` pointing at `dic_correlate` |
| ref/def shapes differ | `ValueError` naming both shapes |
| any axis < 16 voxels | upstream `ValueError` (see 6e) |
| subset too large for the axis | upstream `ValueError` naming the needed extent |
| fewer than 2 nodes on an axis | upstream `ValueError` suggesting smaller winsize/step |
| `voxel_size_um` not length 3, or ≤ 0 | `ValueError` from `build_dvcpara` |
| `ref_indices[i] > i` | upstream `ValueError` — no frame may reference a future one |
| unknown `strain_type` | `ValueError` listing the four valid measures |
| a frame's shape ≠ `shape` | `ValueError` from `LazyVolumeProvider` naming the frame index |
| no usable CUDA device with `backend="auto"` | logs one line and falls back to CPU; `backend="cuda"` raises instead |

## 9. Minimal runnable example

```python
import numpy as np
from al_dvc import synthetic as syn
from nodegraph.kernels.aldvc_field import run_aldvc

ref = syn.generate_bead_volume((72, 72, 72), n_beads=9000, radius=1.8, seed=0)
dfm = syn.warp_volume_lagrangian(ref, syn.affine_displacement(t=(2.5, 0.0, 0.0)))

r = run_aldvc(ref, dfm, voxel_size_um=(0.5, 0.2, 0.2),
              params=dict(subset_size=24, subset_spacing=12, admm_iterations=3))

print(r.displacement_field.shape)          # (Gz, Gy, Gx, 3)
print(np.median(r.displacement_field[..., 2]))   # ~2.5 voxels along x
print(np.nanmedian(r.qfactor))             # ~1.0 on synthetic data
print(r.strain_field.shape, r.strain_type) # (Gz, Gy, Gx, 3, 3) infinitesimal
```

## 10. Pipeline wiring (nodegraph)

`analysis.dvc_field` (label **DVC (pyALDVC)**) calls `run_aldvc_series` **once per
m-position** over the whole T stack, then flattens each frame with
`catalog/_shared/dvc.py::_dvc_rows` into Point rows carrying `disp_*` in µm, nine
`strain_*` columns and `qfactor`.

* Footprint `Granularity.WHOLE_SERIES`, `kernel_axes={t,z,y,x}` — it crosses T.
* `reference_mode` `fixed_frame` → the reference volume is **prepended** as frame 0
  and `ref_indices` is all-zero. That keeps pyALDVC's DAG rule (`ref[i] <= i`)
  satisfied for **any** `reference_frame`, including one later in the series than the
  frame being correlated. `previous_frame` → `ref_indices = (0,1,…,T-2)` and t=0
  yields no row.
* The optional `reference` Dataset input takes the same prepended-frame-0 path, so a
  separate undeformed stack becomes the reference.
* Downstream: `analysis.accumulate_field` composes a `previous_frame` increment
  series into a cumulative field (via [`field_math`](field_math.md)), and
  `transform.rasterize_field` interpolates the sparse Point grid onto voxels.

## 11. Provenance

* **Solver** — external, `al-dvc` 1.2.0 (pyALDVC), Zach Tong.
  <https://github.com/zachtong/pyALDVC>, DOI 10.5281/zenodo.22883767.
* **Method** — Yang, Hazlett, Landauer & Franck, *Augmented Lagrangian Digital Volume
  Correlation*, Exp. Mech. 60 (2020), DOI 10.1007/s11340-020-00607-3;
  <https://github.com/FranckLab/ALDVC>.
* **Adapter** — in-repo, written 2026-09-25, replacing the in-repo clean-room port
  that had held this filename since the v1 vendoring pass.

## 12. Validation (2026-09-25, al-dvc 1.2.0 — `scripts/_aldvc_validate.py`)

Upstream validates the **solver** against the MATLAB reference; re-running the
paper's benchmark here would measure upstream's code. This repo validates the
**adapter**, whose failure modes all produce a correctly-shaped field of finite
numbers that is transposed, negated or mis-scaled. Five groups, 18 checks, all
passing:

| group | what it pins | representative result |
|---|---|---|
| A axis isolation | the `[x,y,z]`→`[z,y,x]` reversal | `+2.5` vox on x → `[0.0001, -0.0001, 2.5001]` |
| B strain relabelling | `_strain_tensor_zyx`, via an **asymmetric** gradient | `strain[x,y] = 0.02024` for mean(0.030, 0.010) = 0.020; `[x,y] == [y,x]`; a +4% x-stretch lands on `[x,x]`, not `[z,z]` |
| C sign | pyALDVC does **not** negate (pyALDIC does) | rigid rotation → `strain[x,y] = −1e-5`, not `+0.020`; `u_y = −g·(x−cx)` to 0.0044 vox |
| D anisotropic voxels | `voxel_size` pass-through and its `(x,y,z)` order | `strain[x,z]` ratio **0.400** = `v_x/v_z` exactly; `strain[x,y]` unchanged; displacement unchanged |
| E frame schedule | `ref_indices` pairing and grid reuse | accumulative `[0.894, 1.792, 2.693]` vs truth `[0.9, 1.8, 2.7]`; incremental `[0.894]×3`; one shared grid |

Run it with `PYTHONUTF8=1 python scripts/_aldvc_validate.py` (add `--quick` for
groups A and C only). It is **not** part of `nodegraph.selftest` — it runs a real
solver a dozen times. The selftest's own DVC group covers the wiring, the schema, the
calibration fence and the retired-param refusal.
