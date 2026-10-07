# Bead Slabs — integration contract

## 1. Purpose + where the real math lives

Find the sub-voxel centroids of **fluorescent beads** in one single-channel 3-D volume by
**slab projection**: flatten overlapping Z sub-stacks, detect in 2-D on each, refine every
hit on the raw stack (a skewed axial fit, a 2-D Gaussian lateral fit), keep only what is
bead-shaped, merge duplicates across slabs. Written for the Bead Finder node
(`detect.beads`, 2026-10-07); the design brief is quoted in
`CodeLog/Updates/worklog/2026-10-07_bead-finder-node.md`.

**The real math is IN-REPO** (this module), numpy + scipy only — `scipy.ndimage`
(LoG, Gaussian derivatives, max/median filters), `scipy.optimize.curve_fit`
(Levenberg–Marquardt with analytic Jacobians), `scipy.spatial.cKDTree`. No numba.
It is **not** the vendored `bead_detect` kernel (the v1 `ParticleDetector`); the two
answer different questions — this one assumes the objects are beads of a known size and
uses that to refuse everything else.

## 2. Entry point

```python
def find_beads(volume_zyx, voxel_size_um, params) -> tuple[np.ndarray, dict[str, np.ndarray], dict]
```

Also public, for callers that want the same expectations the node uses:
`expected_sigmas_px(diameter_um, *, pixel_size_um, z_step_um, psf_sigma_xy_um, psf_sigma_z_um)`
and `auto_slab_thickness(n_beads, shape_zyx, sigma_xy_px, sigma_z_px, *, target_overlap)`.
Everything else (`_despike`, `_detect_2d`, `_z_profile`, `_fit_z`, `_fit_xy`, `_dedup`,
`_one_pass`) is support code. Call **only** the three above.

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------------|-------|-----------|-------|---------------------|-----------------------|
| `volume_zyx` | `np.ndarray` | `(Z, Y, X)` | any real; cast to float64 | `(z, y, x)` | raw intensity | required | One channel's raw stack. `Z < 3` → `ValueError` (a slab needs planes); `ndim != 3` → `ValueError`. |
| `voxel_size_um` | sequence of 3 floats | `(3,)` | float | `(dz, dy, dx)` | micrometers | required | **Reported only** (copied into `info`). Every size in `params` is already in voxels — the caller converts. |
| `params` | `dict` | — | — | — | — | optional (may be `{}`) | Read with `.get` over `DEFAULTS` — see Parameters. |

## 4. Parameters (keys inside `params`, defaults in `DEFAULTS`)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `sigma_xy_px` | float | `1.5` | `>= 0.5` | Expected apparent **lateral Gaussian sigma** of one bead, in pixels (what a least-squares Gaussian fit returns on the imaged sphere — `0.272 d` for a solid sphere of diameter `d`, PSF added in quadrature; `expected_sigmas_px` computes it). Sets the LoG scale, the local-maximum footprint, the crop size, the size band. |
| `sigma_z_px` | float | `1.5` | `>= 0.5` | Expected apparent **axial sigma**, in planes (`0.28 d` + PSF in quadrature). Sets the z-fit window, the core-projection depth, the minimum slab, the auto overlap, the axial size band. |
| `slab_px` | int | `0` | `0` = auto, else `1..Z` | Slab thickness in planes. `0` → derived from the detected density (§6). |
| `overlap_px` | int | `-1` | `<0` = auto, else `0..slab-1` | Planes shared by consecutive slabs. Auto = `round(2 sigma_z)`, clamped. |
| `projection` | str | `"max"` | `"max"` \| `"mean"` \| `"min"` | How a slab is flattened. `"min"` inverts the whole volume (`max − v`) and proceeds as `"max"`: dark beads on a bright background. |
| `min_snr` | float | `5.0` | `> 0` | Minimum LoG peak response, in robust (MAD) sigmas of **that slab's** response, for a pixel to become a candidate — AND the floor the fitted 2-D amplitude must clear over the raw volume's robust noise (`dim` otherwise). |
| `size_tolerance` | float | `0.5` | `0..1` | Accept a fitted lateral sigma within `±tol` of `sigma_xy_px` (geometric mean of σy, σx). The axial band is `±min(0.95, 2 tol)`. |
| `max_aspect` | float | `1.5` | `>= 1` | Reject a fitted bead whose `max(σy,σx)/min(σy,σx)` exceeds this. |
| `target_overlap` | float | `0.10` | `0..1` | Auto-slab only: the tolerated expected fraction of beads with another bead inside their projected footprint (§6). |

## 5. Output

Returns a 3-tuple `(points_zyx, columns, info)`.

**`points_zyx`** — `np.ndarray`, shape `(N, 3)` float64 C-contiguous, **`(z, y, x)` in
voxel units** (sub-voxel), `(0, 3)` when nothing survives.

**`columns`** — `dict[str, np.ndarray]`, every array length `N`, index-aligned with the
points:

| key | dtype | units | meaning |
|-----|-------|-------|---------|
| `amplitude` | float | raw intensity | Fitted 2-D Gaussian peak above its local background (after `min` inversion: the depth of the dip). |
| `snr` | float | — | The LoG peak response over the slab's robust noise sigma (the detection statistic). |
| `sigma_xy_px` | float | px | Fitted lateral sigma, `sqrt(σy · σx)`. |
| `sigma_z_px` | float | planes | Fitted axial sigma, `(σ_lo + σ_hi) / 2`; **NaN** when the split-Gaussian fit fell back to a parabola (short window, non-convergence). |
| `skew_z` | float | — | `(σ_hi − σ_lo) / (σ_hi + σ_lo)` ∈ (−1, 1): positive = wider above the bead (larger z) than below. NaN with `sigma_z_px`. |
| `slab` | int64 | — | Index of the slab whose projection found it (after merging, the brightest duplicate's). |

**`info`** — `dict`: `slab_px`, `overlap_px`, `n_slabs` (the final pass), `n_candidates`,
`rejected` (`{edge_z, other_slab, fit_failed, dim, size_xy, aspect, size_z, duplicate}`
counts),
`dropped` (list of `(z, y, x, reason, detail_dict)` per refused candidate — for a
validation asking what happened to a planted bead), `passes` (one summary per pass:
`slab_px`, `overlap_px`, `n_slabs`, `n`), `despiked_voxels`, `voxel_size_um`,
`projection`, `sigma_xy_px`, `sigma_z_px`.

## 6. Conventions & GOTCHAS (the real integration risk)

- **Everything is voxels.** `sigma_*_px`, `slab_px`, `overlap_px`, the points, the sigma
  columns. µm ↔ px happens in the caller, once each way (`detect.beads` multiplies
  `sigma_xy_px` by `pixel_size_um` and `sigma_z_px` by `z_step_um` for its µm columns).
- **"Size" means the fitted Gaussian sigma, not the FWHM and not the bead radius.** A
  1 µm solid sphere at 0.325 µm/px fits with σ ≈ 0.87 px, not 1.5. Feed FWHM-matched
  sigmas (`0.368 d`) and every real bead lands near the bottom of the size band.
- **Hot voxels are clamped first** (`_despike`): a voxel > 8 noise sigmas above the
  volume median whose 26-neighbour MEDIAN holds < 15 % of its excess is replaced by that
  median. Skipped when `sigma_xy_px < 0.75`. Without it a spike beside a bead swallows the
  bead's LoG peak and the bead is lost.
- **The auto slab is density-driven and capped at half the stack.** Pass 0 projects the
  whole stack; with `n` beads found, `rho = n / (Z·Y·X)`, `a = π (2.355 σxy)²`,
  `S = target_overlap / (rho · a)` planes, clamped to `[round(2 σz), ceil(Z/2)]`;
  re-run while the choice moves by more than 25 % (at most two re-runs, three passes
  total). The cap is deliberate: a whole-stack projection lets one fibre or aggregate
  hide every bead beneath it.
- **A candidate's z is searched near its slab**: `argmax` of the matched-filter profile
  over `[a − ceil(1.5 σz), b + ceil(1.5 σz))`. A peak on plane `0` or `Z−1` is refused
  (`edge_z`) — a bead whose brightest plane was never acquired cannot be localised. A
  peak on the **window's** edge with the profile still rising beyond it is refused as
  `other_slab`: this slab only saw the bead's axial tail, and fitting the tail would plant
  a ghost two or three planes from the real bead that the neighbouring slab finds.
- **Neighbour masking + the unmasked re-fit rule.** The 2-D fit leaves out pixels nearer
  to another LoG candidate of the same slab (Voronoi), so touching beads do not widen each
  other. A fibre's own chain of candidates would cut its END into a bead-sized cell too —
  so a candidate none of whose masking neighbours was itself accepted is re-fitted
  **without** the mask and must pass the width tests on its own.
- **The ridge test is fixed at 2.5**, independent of `max_aspect`: Hessian anisotropy of
  the Gaussian-smoothed projection at the peak (`sqrt(|λ_strong|/|λ_weak|)`, infinite on
  a ridge). 99 % of true beads read below 2.25 even when a fifth of them touch; fibres and
  scan lines read ≥ 5. Both this and the fitted-aspect refusal count under `aspect`.
- **Duplicates across slabs** are merged greedily, brightest first, inside a radius of
  2 in sigma-normalised `(z/σz, y/σxy, x/σxy)` distance — anisotropy-aware.
- **Determinism:** no random numbers anywhere; identical input ⇒ identical output.
- **The z bias is the PSF's, not the fitter's.** On the bead phantom (a skewed `sinc⁴`
  axial PSF whose peak is at the bead centre) the fitted z sits `+0.06 µm` (0.15 plane)
  above the planted centre at every density: the split Gaussian's mode lands slightly on
  the wide side of a skewed peak. A constant offset for all beads of one stack, so it
  cancels in displacements; it is NOT corrected here.

## 7. Dependencies

| pip package | why | import time |
|-------------|-----|-------------|
| `numpy` | everything | **import-time** |
| `scipy` | `ndimage` (gaussian_laplace, gaussian_filter derivatives, maximum_filter, median_filter), `optimize.curve_fit`, `spatial.cKDTree` | **lazy** — imported inside the functions |

## 8. Failure modes / edge cases

- `volume_zyx.ndim != 3` or `Z < 3` → `ValueError`.
- `projection` not in `max|mean|min` → `ValueError`.
- Flat / all-zero volume: LoG noise is zero → threshold falls back to the response std;
  typically no candidates → `(0, 3)`, empty columns, `rejected` all zero.
- Every candidate refused → `(0, 3)` and the reasons in `info["rejected"]`.
- A fit that does not converge degrades, never raises: the xy fit to moments (then
  `fit_failed`); the z fit to a 3-point parabola — which is accepted only on a stack of
  fewer than 5 planes (`sigma_z_px`/`skew_z` NaN, no axial size test). On a deeper stack a
  candidate whose axial profile the split Gaussian cannot fit is refused (`fit_failed`): a
  bead has an axial profile, a noise blip or the cap of an aggregate seen through a thin
  slab does not.
- Beads at the lateral border: the crop is clipped, the fit is one-sided; they are kept
  if they pass the tests, but their `sigma_xy_px` is less reliable.

## 9. Minimal runnable example

```python
import numpy as np
from nodegraph.kernels.bead_slabs import find_beads, expected_sigmas_px

Z, Y, X = 20, 64, 64
vol = np.full((Z, Y, X), 200.0)
zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(float)
for (cz, cy, cx) in [(6.3, 20.2, 20.7), (10.8, 40.5, 44.1), (14.1, 50.0, 15.6)]:
    vol += 2000 * np.exp(-((zz - cz) ** 2 / (2 * 1.0 ** 2)
                           + (yy - cy) ** 2 / (2 * 0.9 ** 2) + (xx - cx) ** 2 / (2 * 0.9 ** 2)))
vol += np.random.default_rng(0).normal(0, 10, vol.shape)

s_xy, s_z = expected_sigmas_px(1.0, pixel_size_um=0.325, z_step_um=0.4,
                               psf_sigma_xy_um=0.078, psf_sigma_z_um=0.49 / 2.355)
pts, cols, info = find_beads(vol, (0.4, 0.325, 0.325),
                             {"sigma_xy_px": s_xy, "sigma_z_px": s_z})
# pts.shape == (3, 3), (z, y, x) voxels, each within ~0.1 voxel of the planted centre
# cols["sigma_xy_px"] ≈ 0.9, cols["skew_z"] ≈ 0 (a symmetric synthetic profile)
# info["passes"][0]["slab_px"] == 20 (pass 0, whole stack), then the auto choice
```

## 10. Pipeline wiring

- **Upstream (caller-owned prep):** one channel of one (m, t) as a raw `(Z, Y, X)`
  volume; the voxel size; `expected_sigmas_px` from the bead diameter and the optics
  (`detect.beads` uses `0.21 λ/NA` for the lateral PSF sigma and the derived confocal
  axial FWHM `/ 2.355`); slab sizes converted from `um_axial` to planes.
- **Downstream:** `points_zyx` → `structure.point_table(..., z_kind="subpixel")` with the
  `columns` attached; the node converts the sigma columns to µm. Track Linking / Track
  Objects / Cluster Points / Measure read the Point table by its layer name.

## 11. Validation

`scripts/_bead_finder_validate.py` runs the finder on the `beads3d` phantom
(`nodegraph/phantom.py`: a confocal z-stack of 1 µm spheres through a skewed `sinc⁴`
axial PSF, with aggregates, fibres, a scan line, haze, hot voxels and out-of-stack beads
planted) across a density sweep and a left-to-right density gradient, scoring recall,
precision and localisation error against the planted truth, and lists what happened to
every missed bead. `selftest::test_detect_beads` pins the headline numbers.
