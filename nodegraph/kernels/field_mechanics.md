# `field_mechanics` — field kinematics and constitutive laws integration contract

## 1. Purpose + where the real math lives

Turns a sampled **displacement** (or **velocity**) field — one row per field node, the shape
every correlation node in this repo writes — into what a deformation measurement is for: the
**gradient**, the **strain** under a chosen measure with its principal values and invariants,
the **curl** and finite **rotation**, the **volume change** (Jacobian, dilatation, divergence)
and, through a stated material law, the **Cauchy stress**. The five `field.*` nodes are thin
wrappers around it.

**In-repo math, reusing what exists rather than copying it:**

- the four strain measures are [`field_math`](field_math.md)`.strain_from_gradient`, shared
  with `analysis.dvc_field` / `dvc.strain` and `analysis.track_field`, so the three cannot
  disagree about what "Green-Lagrange" means;
- the scattered-point gradient is [`track_field`](track_field.md)`.mls_displacement_gradient`
  (SerialTrack's strain gauge), and Lamé parameters, Hooke's law under plane strain / plane
  stress, `_principal` and `stress_invariants` are `track_field`'s too;
- new here: regular-lattice detection and a NaN-aware finite-difference gradient (correlation
  grids are lattices with holes), the rotation / curl / volume-change quantities, and three
  more laws — Saint Venant-Kirchhoff, a compressible neo-Hookean solid and a Newtonian fluid.

## 2. Entry points

```python
lattice_of(coords, *, tol=1e-3) -> Optional[Lattice]          # (N, D) → shape/step/index or None
node_spacing(coords) -> float                                  # median nearest-neighbour distance
shell_distance(coords) -> (float, int)                         # distance to the 3^D-1-th neighbour
gradient(coords, scale, vec, *, method="auto", radius_um=0.0,
         neighbors=0, smooth=0.0) -> GradientResult            # G (N,D,D), valid, method, radius, spacing
strain(G, measure="infinitesimal") -> (N, D, D)                # symmetric, tensor shear
principal(T) -> (values (N,D) largest first, angle_deg (N,) | None)
equivalent_strain(T) -> (N,)      max_shear(T) -> (N,)
curl(G) -> {"curl"} | {"curl_z","curl_y","curl_x","curl_mag","curl_view"}
rotation_deg(G) -> (N,)           volume_change(G) -> {"jacobian","dilatation","divergence"}
stress(G, model, *, youngs_modulus, poisson_ratio, plane="strain",
       viscosity=1.0) -> (sigma (N,3,3) in (z,y,x), energy (N,))
stress_summary(sigma) -> {"stress_p1..p3","pressure","von_mises","max_shear"}
```

## 3. Inputs

| name | shape | dtype | axis order | units | meaning |
|------|-------|-------|------------|-------|---------|
| `coords` | `(N, D)` | float | `(y, x)` / `(z, y, x)` | index units (px, plane steps) | node positions; lattice detection runs on these clean values |
| `scale` | `(D,)` | float | same | µm per index unit | `pixel_size_um` (and `z_step_um` first in 3-D) |
| `vec` | `(N, D)` | float | same | µm (displacement) or µm/s (velocity) | NaN = an unmeasured node (a hole) |
| `G` | `(N, D, D)` | float | same | 1 or 1/s | `G[p, i, j] = ∂u_i/∂X_j` |

## 4. Parameters

| name | default | meaning |
|------|---------|---------|
| `method` | `auto` | `lattice` (finite differences; refused off a lattice), `local_fit` (MLS plane fit), `auto` (lattice when there is one) |
| `radius_um` | 0 | local-fit radius; 0 = 1.5 × `shell_distance` (≈ 2 node spacings, reaches the planes of an anisotropic stack) |
| `neighbors` | 0 | KNN ceiling before the radius cut; 0 = from the radius and the shell density (≤ 400) |
| `smooth` | 0 | Gaussian pre-smoothing on the lattice in NODE units, hole-aware (normalized convolution) |
| `measure` | `infinitesimal` | `infinitesimal`, `green_lagrange`, `euler_almansi`, `hencky` |
| `model` | — | `linear_elastic`, `saint_venant_kirchhoff`, `neo_hookean`, `newtonian_fluid` |
| `plane` | `strain` | 2-D only: `strain` (F_zz = 1) or `stress` (σ_zz = 0) |

## 5. Output

Every array is per node, NaN where undefined (a non-finite gradient, a degenerate fit, a
non-positive J). Stress is always the full `(3, 3)` in `(z, y, x)`; for a 2-D field the z
slot is the plane assumption's (`F_zz = 1` → σ_zz ≠ 0, or σ_zz = 0).

## 6. Conventions & GOTCHAS

- **`G` is the MATERIAL gradient** (reference configuration): `F = I + G`. For tracked particles
  the caller must fit at `x − u` (the `field.gradient` node does, from `track_id`); fitting the
  spatial gradient makes Green-Lagrange report Almansi.
- **Image axes, y DOWN.** 2-D `curl = ∂u_x/∂y − ∂u_y/∂x` is **positive counter-clockwise on
  screen** (twice the small rotation in radians); `rotation_deg` (exact, `atan2(F_xy − F_yx,
  F_xx + F_yy)`) has the same sign; the 2-D principal angle is counter-clockwise from +x on
  screen. 3-D curl components are in (x right, y down, z = plane index), right-handed when z
  runs away from the viewer; `curl_view = −curl_z` carries the 2-D sign.
- **Lattice gradient:** central where both neighbours exist, one-sided beside a hole or at an
  edge, NaN when both neighbours along an axis are missing. A lattice must fill ≥ 50 % of its
  bounding grid (`MIN_FILL`) — scattered detections on whole pixels are NOT a lattice.
- **2-D equivalent strain** takes `e_zz = −(e_yy + e_xx)` (incompressibility, the commercial-DIC
  convention).
- **Units:** stress is in the unit of `youngs_modulus` (Pa by the node's socket); energy in
  J/m³ = Pa; a velocity gradient is 1/s and only `linear` quantities apply to it.

## 7. Dependencies

numpy at import; scipy lazily (`cKDTree` for spacing and the local fit, `gaussian_filter` for
smoothing). Imports `field_math` and `track_field` from this package.

## 8. Failure modes / edge cases

- `method="lattice"` off a lattice → `ValueError` naming `local fit`.
- E ≤ 0, ν ∉ (−1, 0.5), viscosity ≤ 0 → `ValueError` (from `lame_parameters` / `stress`).
- An unknown measure / model / plane → `ValueError`.
- Fewer than D + 1 finite nodes, or a degenerate neighbourhood → NaN rows, never an exception.
- Neo-Hookean plane stress: Newton on `μ(s² − 1) + λ ln(J₂ s) = 0` (monotone in s > 0, step
  halving keeps s > 0); a node with J₂ ≤ 0 is NaN.

## 9. Minimal runnable example

```python
import numpy as np
import nodegraph.kernels.field_mechanics as FM
yy, xx = np.mgrid[0:200:16, 0:240:16]
co = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)
A = np.array([[0.02, -0.01], [0.03, 0.015]])
u = (co * 0.5) @ A.T
r = FM.gradient(co, np.array([0.5, 0.5]), u)           # lattice path, G == A
E = FM.strain(r.G, "green_lagrange")
sigma, W = FM.stress(r.G, "neo_hookean", youngs_modulus=5e3, poisson_ratio=0.45, plane="strain")
print(FM.volume_change(r.G)["jacobian"][0], FM.stress_summary(sigma)["von_mises"][0])
```

## 10. Pipeline wiring (nodegraph)

`field.gradient` → `field.strain` / `field.rotation` / `field.volume_change` / `field.stress`
(`nodegraph/catalog/field/`), sharing `catalog/_shared/field_nodes.py`: the field layer is read
by its `disp_*` / `v*` columns, 2-D vs 3-D from its `z_kind`, coordinates scaled with
`ctx.calib`, and every card writes its columns back onto the same table with units recorded and
its settings under `__field_mechanics__[layer]`. `bundle:mechanics` places gradient → strain →
stress wired.

## 11. Provenance

In-repo, 2026-10-09. The finite-strain laws are textbook (Bonet & Wood, *Nonlinear Continuum
Mechanics for Finite Element Analysis*: compressible neo-Hookean `σ = μ/J (B − I) + λ ln J / J
I`, Saint Venant-Kirchhoff `S = λ tr(E) I + 2μE`); the polar-decomposition rotation and the
2-D `atan2` form are standard. Reused symbols keep their own provenance (`field_math`,
`track_field`).

## 12. Validation (2026-10-09)

`nodegraph.selftest::test_field_mechanics`: an affine field's gradient exact on the lattice
(≤1e-10, holes included) and by local fit (≤1e-8) in 2-D and on an anisotropic 3-D stack; a
10° on-screen counter-clockwise turn reads +10° with zero Green-Lagrange / Almansi / Hencky
strain; J 1.21 / 1.331 for a 1.1 stretch; simple-shear curl +γ and principal angle −45°;
uniaxial plane-stress Hooke; plane-stress σ_zz = 0 for Saint Venant-Kirchhoff and neo-Hookean,
both within 1e-3 of Hooke at 1e-4 strain; Newtonian shear stress ηγ̇; the refusals.
