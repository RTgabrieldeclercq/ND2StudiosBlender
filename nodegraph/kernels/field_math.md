# `field_math` — shared correlation-field maths integration contract

## 1. Purpose + where the real math lives

**In-repo**, pure numpy/scipy. This is the dimension-agnostic maths every correlation
node needs *after* a solver has produced a displacement field on a regular grid: the
`DVCResult` output contract, the subset `Grid`, harmonic NaN inpainting, the
displacement-gradient → strain-measure stack, and Lagrangian accumulation of an
increment series. **No correlation happens here** — the solvers are
[`aldvc_field`](aldvc_field.md) (3D, pyALDVC), [`dic_correlate`](dic_correlate.md)
(2D, pyALDIC) and [`piv_field`](piv_field.md) (2D, OpenPIV).

### Why it is its own module (2026-09-25)

`aldvc_field` used to be a 2998-line clean-room port of FranckLab's MATLAB ALDVC, and
this maths was buried inside it. When that port was replaced by the official pyALDVC
package, two nodes that had **never used the solver** would have broken with it:

* `analysis.accumulate_field` → `Grid`, `accumulate_incremental`, `compute_strain`
* `analysis.track_field` → `strain_from_gradient`

So the maths moved here rather than dying with the port. Every symbol is
**byte-verbatim** from the module it came out of; only the imports were consolidated.

## 2. Entry points

```python
# the output contract shared by every correlation node
DVCResult(dim, grid_coords, displacement_field, voxel_size_um=(), strain_field=None,
          strain_type="", qfactor=None, converged=False, iterations=0, mu=0.0,
          beta=0.0, method="", notes="", diagnostics={})

# the regular subset grid
build_grid(shape, subset_size, subset_spacing, border_margin=0) -> Grid
Grid(axes, coords, grid_shape, step, ndim)   # .n_nodes, .coords_flat()

# gap filling
inpaint_nans(field, *, iterations=60) -> ndarray          # (*grid,) harmonic fill
inpaint_vector(u_grid) -> ndarray                          # (ndim, *grid)

# strain
displacement_gradient(u_grid, grid_step, voxel_size=None, smooth_sigma=0.0) -> G
strain_from_gradient(G, strain_type="infinitesimal") -> strain
compute_strain(u_grid, grid_step, voxel_size=None, *, strain_type="infinitesimal",
               smooth_sigma=0.0) -> (F_def, strain)

# accumulation
accumulate_incremental(grid, increments) -> [(t, u_accum_grid), ...]
build_accumulated_results(grid, increment_results, voxel_size_um, *,
                          strain_type="infinitesimal", strain_smooth=0.0) -> {t: DVCResult}
```

## 3. Inputs

| arg | shape | notes |
|---|---|---|
| `u_grid` | `(ndim, *grid)` | displacement, component axis **first**, in voxels |
| `grid_step` | `(ndim,)` | node spacing per axis, in voxels — `Grid.step` |
| `voxel_size` | `(ndim,)` or None | physical size per axis, **slowest-first**. None = leave in voxel units. |
| `G` | `(ndim, ndim, *grid)` | `G[i,j] = ∂u_i/∂x_j` |
| `increments` | `[(t, u_grid), …]` | **ascending** frame order, each an increment `t-1 → t` |

## 4. Parameters

| param | default | meaning |
|---|---|---|
| `strain_type` | `infinitesimal` | `infinitesimal` / `green-lagrange` / `almansi` / `hencky` (aliases: `small`, `engineering`, `green`, `lagrange`, `euler-almansi`, `log`, `logarithmic`) |
| `smooth_sigma` | 0.0 | Gaussian σ on the displacement **before** differentiation, in node units |
| `subset_spacing` | — | node pitch in voxels |
| `border_margin` | 0 | extra inset beyond `subset_size // 2`, so a deformed search window still fits |
| `iterations` | 60 | Jacobi sweeps toward the harmonic solution in `inpaint_nans` |

## 5. Output

`DVCResult` — see [`aldvc_field.md` §5](aldvc_field.md). `strain_from_gradient`
returns `(ndim, ndim, *grid)`; `compute_strain` returns `(F_def, strain)` where
`F_def = I + G` is the deformation gradient.

`accumulate_incremental` returns `[(t, u_accum_grid), …]`, each `(ndim, *grid)`, the
cumulative displacement **from the reference frame** obtained by tracking the
reference grid points through the increments (Lagrangian, not a running sum).

## 6. Conventions & GOTCHAS

### 6a. Slowest-axis-first, everywhere

Every field is `(z, y, x)` in 3D and `(y, x)` in 2D. `grid_coords[..., 0]` and
`displacement_field[..., 0]` are the **slowest** axis, not x. A solver whose native
order is `[x, y, z]` (pyALDVC, pyALDIC) must reverse **before** it builds a
`DVCResult`; that reversal lives in the solver kernel, never here.

### 6b. `strain_from_gradient` returns the SYMMETRIC tensor

`strain[i,j] == strain[j,i]`, and the shear terms are **tensor** shear — half the
engineering shear. A pure `∂u_x/∂y = 0.02` gives `strain[x,y] = 0.01`. This matches
pyALDVC's `StrainResult.exy` convention exactly, so the two are interchangeable.

The four measures are *not* interchangeable once strain leaves the small-deformation
regime: `infinitesimal` is `½(G+Gᵀ)` and is **not** invariant to rigid rotation, so a
rotating sample picks up spurious strain; `green-lagrange` (`½(FᵀF−I)`) is exactly
zero under rotation.

### 6c. `voxel_size` applies the cross-axis rescale — skipping it is a silent error

`displacement_gradient` multiplies `G[i,j]` by `voxel_i / voxel_j`. Converting **both**
the displacement components and the spatial axes to micrometres is what makes strain
dimensionless and correct on non-cubic voxels. With a confocal 0.5 µm z-step against a
0.1 µm xy pixel, omitting it leaves every off-diagonal term wrong by the 5× anisotropy
ratio — and the diagonal terms right, which is what makes it hard to notice.

### 6d. A singleton axis contributes zero gradient, it does not raise

`np.gradient` needs ≥2 samples along an axis. `displacement_gradient` **skips** any
axis with `grid_shape[j] < 2` and leaves `G[:, j]` at zero, so a 1-node-deep Z grid
produces a well-formed tensor with no through-plane terms rather than an exception.

### 6e. `inpaint_nans` is harmonic, not nearest-neighbour

It solves the discrete Laplace equation over the NaN set (nearest-value seed, then
Jacobi sweeps restricted to the NaN set with the finite nodes as Dirichlet data). That
is **exact for any locally linear field**, which is the regime a displacement field is
in over one grid step. The nearest-finite fill it replaced is exact only for a
*constant* field: it produces piecewise-constant blocks whose internal gradient is zero
and whose boundary gradient is a step — and those blocks feed straight into the
strain gradient.

### 6f. `build_grid`'s inset, and why `border_margin` exists

Centres are inset by `subset_size // 2 + border_margin`. The margin is not decoration:
with a bare half-subset inset the outermost ring of centres sits exactly on the edge
of the legal region, so its search window is clipped on one side and the seed there
cannot represent displacement of the clipped sign (measured at 18–40 % of nodes on a
stretch field). The inset is never allowed to collapse past the point where the subset
window still fits; an axis shorter than `subset_size + 1` gets a *reduced* window
instead of centres whose window hangs off the end.

`build_grid` also forces **≥2 centres** on any axis whose extent allows it, because
the finite-difference gradient operator and `np.gradient` are both undefined on a
size-1 axis.

### 6g. Accumulation is Lagrangian

`accumulate_incremental` tracks the reference grid points *through* the increments,
interpolating each increment at the points' **current** positions
(`RegularGridInterpolator`, linear, extrapolating at borders). It is not a sum of the
increment fields — those are sampled at fixed grid nodes, which is a different
(Eulerian) quantity and drifts from the true cumulative displacement as soon as the
points move off their nodes.

## 7. Dependencies

| pip name | import | required? |
|---|---|---|
| `numpy` | `numpy` | yes |
| `scipy` | `scipy.ndimage`, `scipy.interpolate` | yes |

All top-level; nothing optional, nothing lazy.

## 8. Failure modes / edge cases

| situation | behaviour |
|---|---|
| all-NaN field into `inpaint_nans` | returns zeros (no finite data to propagate) |
| no NaNs | returns the input unchanged, no sweeps |
| `grid_shape[j] < 2` | that axis contributes zero gradient (§6d) |
| axis shorter than `subset_size + 1` | reduced window on that axis, not an error |
| unknown `strain_type` | `ValueError` naming the value |
| `voxel_size` omitted on anisotropic data | silently returns **voxel-unit** strain (§6c) |
| empty `increments` | returns `[]` |

## 9. Minimal runnable example

```python
import numpy as np
from nodegraph.kernels.field_math import build_grid, compute_strain

g = build_grid((40, 40, 40), subset_size=16, subset_spacing=8)
u = np.zeros((3,) + g.grid_shape)
u[2] = 0.02 * g.coords[..., 1]              # u_x = 0.02 * y  (a simple shear)

F, strain = compute_strain(u, g.step, strain_type="infinitesimal")
print(g.grid_shape, g.step)                 # (3, 3, 3) [8. 8. 8.]
print(np.median(strain[2, 1]))              # 0.01  — tensor shear, half of 0.02
print(np.median(strain[1, 2]))              # 0.01  — symmetric
```

## 10. Pipeline wiring (nodegraph)

| consumer | uses |
|---|---|
| `analysis.accumulate_field` | `Grid`, `accumulate_incremental`, `compute_strain` |
| `analysis.track_field` | `strain_from_gradient` (shared so a tracked field and a correlated one report the same measure) |
| `aldvc_field`, `dic_correlate`, `piv_field` | the `DVCResult` **shape** — each carries its own copy so it stays byte-portable (`README.md`'s doctrine); this module holds the canonical one |

## 11. Provenance

Byte-verbatim out of the v1 ND2Studios app (branch `Version-1.45`), via the
`aldvc_field` vendoring pass, then lifted here on 2026-09-25:

| symbols | came from |
|---|---|
| `DVCResult` | `nd2studios/core/dvc_registry.py` |
| `Grid`, `build_grid` | `nd2studios/backend/dvc/mesh.py` |
| `inpaint_nans`, `inpaint_vector` | `nd2studios/backend/dvc/outliers.py` |
| `displacement_gradient` … `compute_strain` | `nd2studios/backend/dvc/strain.py` |
| `accumulate_incremental`, `build_accumulated_results` | `nd2studios/backend/dvc/tracking.py` |
