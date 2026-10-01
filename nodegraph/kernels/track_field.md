# `track_field` — tracked-field kinematics integration contract

## 1. Purpose + where the real math lives

Turns a **tracked** point field — particle positions plus the displacement that a tracker's
correspondence implies — into the quantities a deformation measurement is actually for: the
**displacement gradient**, the **strain** (via the shared measure code), and, with a material
model stated, the **Cauchy stress** and its invariants.

This is the post-processing half of SerialTrack, lifted out of the tracker.
[`track_objects`](track_objects.md) ends at the correspondence problem: which detection in
frame *t* is which detection in frame *t-1*. Everything downstream of that answer needs
nothing from the tracker but the ids, which is why it lives here and works on **any** tracked
field — SerialTrack's, `track.link`'s nearest-neighbour linker's, or a hand-built one.

**The math is in-repo, and split by parentage:**

- `mls_displacement_gradient` is SerialTrack's own scattered strain gauge —
  `funCompDefGrad3.m`, already ported as the per-particle loop
  `track_objects.compute_strain_mls` and re-derived here in vectorised form. The two are
  pinned together on a fixture by `nodegraph.selftest::test_track_field`, so this port
  cannot drift from the code it was derived from.
- `linear_elastic_stress` / `stress_invariants` are **NOT from SerialTrack**. SerialTrack
  computes displacement and strain and stops there, because stress is a claim about the
  *specimen* rather than about the images. The isotropic linear-elastic law here is the
  textbook one and is exposed as an explicit, named choice with its own moduli.
- `strain_invariants` is plain tensor algebra, here so the strain and stress halves report
  their scalars through one code path.

The four strain **measures** are not in this file at all: they come from
[`field_math`](field_math.md)`.strain_from_gradient`, shared with `analysis.dvc_field` so
the two nodes cannot disagree about what "Green-Lagrange" means. (That symbol lived in
`aldvc_field` until 2026-09-25, when the official pyALDVC package replaced the in-repo
ALDVC port and the dimension-agnostic maths moved to its own module.)

## 2. Entry points

```python
mls_displacement_gradient(
    coords: np.ndarray,          # (N, D) positions, slowest-first, physical length unit
    disp: np.ndarray,            # (N, D) displacement, SAME unit as coords
    *,
    radius: float,               # neighbourhood radius in that unit
    n_neighbors: int,            # KNN ceiling before the radius cut
    rcond: float = 1e-8,         # relative singular-value floor (conditioning guard)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]   # (u_fit (N,D), G (N,D,D), valid (N,))

lame_parameters(youngs_modulus: float, poisson_ratio: float) -> Tuple[float, float]

linear_elastic_stress(
    strain: np.ndarray,          # (D, D, N), D in (2, 3)
    *, youngs_modulus: float, poisson_ratio: float,
    plane: str = "strain",       # "strain" | "stress" — read only when D == 2
) -> np.ndarray                  # sigma (3, 3, N) — ALWAYS the full tensor

strain_invariants(strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    # (D, D, N) -> (volumetric (N,), max_shear (N,))

stress_invariants(sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    # (3, 3, N) -> (pressure (N,), von_mises (N,), max_shear (N,))
```

## 3. Inputs

| arg | shape | meaning |
|---|---|---|
| `coords` | `(N, D)` float | particle positions, `(z, y, x)` in 3-D or `(y, x)` in 2-D, in a physical length unit |
| `disp` | `(N, D)` float | displacement at each particle, in the **same** unit; a non-finite row is an UNMEASURED particle |
| `strain` | `(D, D, N)` float | the strain tensor per particle; `D` = 2 or 3 |
| `sigma` | `(3, 3, N)` float | stress per particle, always the full 3×3 |

`N` may be 0; every function returns correctly-shaped empties.

## 4. Parameters

| param | default | effect |
|---|---|---|
| `radius` | — | neighbourhood radius, the **gauge length** of the strain measurement. Small resolves sharp concentrations but is noisy and leaves sparse particles invalid; large is smooth and stable and systematically flattens peak strain. |
| `n_neighbors` | — | KNN ceiling applied **before** the radius cut — a cost bound, not a second radius. Clamped to `N - 1`. |
| `rcond` | `1e-8` | relative singular-value floor on the normal-equation matrix. A degenerate neighbourhood (collinear in 2-D, coplanar in 3-D) is marked invalid instead of taking `pinv`'s least-norm answer. |
| `youngs_modulus` | — | sets the **unit** of every stress output and scales them linearly. Must be positive and finite. |
| `poisson_ratio` | — | must lie in `(-1, 0.5)`, exclusive at both ends. |
| `plane` | `"strain"` | the 2-D out-of-plane assumption; inert for `D == 3`. |

## 5. Output

`mls_displacement_gradient` → `(u_fit, G, valid)`:

- `u_fit` `(N, D)` — the smoothed (fitted) displacement; NaN where the fit failed.
- `G` `(N, D, D)` — `G[p, i, j] = ∂u_i / ∂x_j`; NaN where the fit failed.
- `valid` `(N,)` bool — a well-conditioned fit from the particle's **own measured**
  displacement plus at least `D` measured neighbours.

`linear_elastic_stress` → `sigma (3, 3, N)`, always the full tensor in `(z, y, x)` even for a
2-D field, because both plane assumptions produce an out-of-plane component (one of them as
exactly zero) and von Mises needs it.

`strain_invariants` → `(volumetric, max_shear)`; `stress_invariants` → `(pressure,
von_mises, max_shear)`. Signs and definitions are in §6.

## 6. Conventions & GOTCHAS (the real integration risk)

1. **`G` is the displacement gradient, not the deformation gradient.** `F = I + G`.
   `field_math.strain_from_gradient` takes `G` and adds the identity itself — pass it `F`
   and every measure is wrong.

2. **Fit in the REFERENCE configuration.** `F = I + G` is the deformation gradient only for
   the *material* gradient `∂u/∂X`. If you fit at the deformed positions you get `∂u/∂x`,
   and `I + ∂u/∂x` is the **inverse** deformation gradient — so `infinitesimal` stays right
   to first order (the two differ by `1/(1+ε)`, 6% at 6% strain) while Green-Lagrange,
   Almansi and Hencky all come back quietly wrong, Green-Lagrange returning Almansi's
   answer under Green-Lagrange's name. Callers pass `coords - disp`, and
   `analysis.track_field` does. Only the finite measures can catch this, which is why the
   selftest checks all four.

3. **Everything physical on the way in.** `coords` and `disp` must share one length unit, so
   the gradient is dimensionless with no anisotropy correction left to apply. This
   deliberately differs from `track_objects.compute_strain_mls`, which takes voxel
   coordinates and rescales afterwards through `pixel_steps`; that path exists for the
   tracker's internal grid and is the one place the two functions may disagree.

4. **Axis order is slowest-first** — `(z, y, x)` in 3-D, `(y, x)` in 2-D. For a 2-D stress
   tensor the in-plane block occupies slots 1–2 of the returned 3×3 and the out-of-plane
   component is slot 0.

5. **A non-finite `disp` row is an unmeasured particle**, not a zero. It is dropped from
   every neighbourhood it appears in AND gets no gradient of its own — even where its
   neighbours would support a well-conditioned fit. That fit would be a legitimate
   *interpolation* of the field at that location, and reporting it in the same column as
   measured particles would be fabrication wearing a measurement's name.

6. **Plane strain vs plane stress.** `plane="strain"` sets `ε_zz = 0` (a slice of a THICK
   specimen — a TFM gel) and yields `σ_zz = ν(σ_yy + σ_xx)`. `plane="stress"` sets
   `σ_zz = 0` (a THIN free-standing film) and replaces λ by `2λμ/(λ+2μ) = Eν/(1-ν²)`.
   Plane strain reproduces the 3-D law fed `ε_zz = 0` exactly.

7. **Invariant sign conventions.** `pressure = -tr(σ)/3`, so COMPRESSION is positive — the
   opposite of the raw trace, and the convention of the gel-mechanics literature.
   `von_mises = sqrt(1.5 · s:s)` over the deviator. `max_shear` is Tresca,
   `(σ₁ - σ₃)/2`, on both the stress and the strain side.

8. **`ν = 0.5` is refused, not clamped.** There the bulk modulus is infinite and the
   pressure is a Lagrange multiplier this local law cannot supply. Use 0.49 / 0.499 for a
   nearly-incompressible gel, which is what the TFM literature does.

9. **`eigvalsh` is NaN-intolerant** — it raises for the whole batch. `_principal` excludes
   the non-finite particles and fills NaN, and symmetrises first so that symmetry is an
   assertion rather than an assumption.

## 7. Dependencies

| dep | when | why |
|---|---|---|
| numpy | import time | everything |
| scipy | **lazy** | `scipy.spatial.cKDTree`, imported inside `mls_displacement_gradient` so the core stays scipy-free |

Nothing else. No numba, no pandas — unlike `track_objects`, this module imports cheaply.

## 8. Failure modes / edge cases

| situation | behaviour |
|---|---|
| `N < D + 1` | all-NaN output, `valid` all False — no exception |
| lone particle (radius below the spacing) | invalid: the particle is always in its own fit, so it cannot be fitted to nothing |
| collinear (2-D) / coplanar (3-D) neighbourhood | invalid, via the `rcond` guard |
| duplicate coordinates | fine — zero distances, the extra rows just add weight |
| `coords`/`disp` shape mismatch | `ValueError` |
| `strain` not `(D, D, N)` with `D ∈ {2,3}` | `ValueError` |
| `E <= 0` or non-finite | `ValueError` naming the stiffness |
| `ν` outside `(-1, 0.5)` | `ValueError` naming the incompressible limit |
| non-finite entries anywhere in an invariant's input | that particle is NaN; the batch survives |

## 9. Minimal runnable example

```python
import numpy as np
from nodegraph.kernels.track_field import (
    mls_displacement_gradient, linear_elastic_stress, stress_invariants)
from nodegraph.kernels.field_math import strain_from_gradient

rng = np.random.default_rng(0)
X = rng.uniform(0, 100, size=(400, 3))          # (z, y, x) in um
G_true = np.diag([0.0, 0.0, 0.03])              # 3% stretch along x
U = X @ G_true.T                                 # displacement in um

# fit in the REFERENCE configuration (see gotcha 2)
_, G, valid = mls_displacement_gradient(X, U, radius=25.0, n_neighbors=25)
print(valid.mean(), np.nanmax(np.abs(G[valid] - G_true)))     # 1.0  ~1e-15

eps = strain_from_gradient(np.moveaxis(G[valid], 0, -1), "green-lagrange")
sig = linear_elastic_stress(eps, youngs_modulus=3000.0, poisson_ratio=0.45)
p, vm, tau = stress_invariants(sig)
print(eps[2, 2, 0], sig[2, 2, 0], vm[0])
```

## 10. Pipeline wiring (nodegraph)

```
detect.particles (3D)  ->  track.objects (serialtrack)  ->  analysis.track_field
      or                                                            |
analysis.label  ->  track.objects (any linker)           transform.rasterize_field
                                                                    |
                                                              view.overlay
```

`analysis.track_field` owns everything above this kernel: resolving the member table,
converting index coordinates to µm through `pixel_size_um` / `z_step_um`, building the
per-track displacement, grouping the fit per `(m,t,c)` volume or `(m,t,c,z)` plane, and
emitting the Point table. It emits the same schema as `analysis.dvc_field`
(`_shared.dvc._dvc_rows`), which is what lets `transform.rasterize_field` interpolate every
column into full-resolution Voxel maps with no adapter.

## 11. Provenance

New in-repo code (2026-09-17), not a vendor pass. `mls_displacement_gradient` is derived
from `track_objects.compute_strain_mls` — itself the vendored port of FranckLab SerialTrack's
`funCompDefGrad3.m` — and is pinned to it by selftest. The elasticity half has no SerialTrack
ancestor; it is the standard isotropic linear-elastic law, added because the node it serves
is used for traction-force microscopy, where strain alone is not the answer anyone wants.

Surface **traction** is deliberately absent: it is an inverse problem over a half-space
(a Boussinesq / FTTC inversion with its own regularisation), not a local constitutive law,
and belongs in its own kernel.

## 12. Validation (2026-09-17)

Closed-form, not eyeballed. `nodegraph.selftest::test_track_field` drives an exact affine
deformation through the node and asserts:

| check | result |
|---|---|
| gradient of an affine field, 2-D and 3-D | exact to `~1e-15` |
| parity with the `compute_strain_mls` loop | `max|ΔG| < 1e-8` over the shared valid set |
| collinear / sub-spacing neighbourhood | refused, never fabricated |
| 3-D uniaxial, shear and hydrostatic stress states | match `E·ε`, `2μ·ε`, `-K·tr(ε)` exactly |
| von Mises of pure shear | `√3·τ` |
| 2-D plane stress `σ_xx` | `E/(1-ν²)(ε_xx + ν ε_yy)` |
| 2-D plane strain | identical to the 3-D law with `ε_zz = 0` |
| all four strain measures at `F_xx = 1.06` | `0.06`, `0.0618`, `ln(1.06)`, `0.055` — to `1e-9` |
| speed vs the loop port | 6× on 8000 points (0.11 s vs 0.7 s) |
