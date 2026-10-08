# `flow_viewer` — 3-D flow reconstruction + self-contained WebGL viewer

Kernel module: `nodegraph/kernels/flow_viewer.py` (+ `flow_viewer_template.html`, the page).
Node: `io.write_flow_viewer` (`nodegraph/catalog/io/write_flow_viewer.py`).
Origin: a port of the lab's granular-flow viewer pipeline
(`microfluidic-LLS-Paper/DataAnalysis/granular_flow_viewer/analysis/{reconstruct_3d,
precompute_geometry, build_webgl_viewer}.py`, 2026-07-11) whose output is deployed at
<https://mcgheelab.com/DataIllustrations/viewer3d.html>. The template is that deployed file
(pipeline template + website patch: embed mode, saved-view config API, render-loop pause,
mobile layout) with its dataset strings replaced by placeholders.

## 1. Entry points

```python
U, V, n = assemble_grid(iz, iy, ix, u, v, shape=(nz, ny, nx))
geometry, report = build_geometry(U, V, s=..., px_per_cell=..., plane_names=[...], **tunables)
html = render_html(geometry, report, title=..., subtitle=..., units=..., cell_um=...)
```

Stage functions are public too: `prepare_fields`, `w_trapezoid`, `w_regularized`,
`residual_divergence_3d`, `channel_connectivity`, `interslice_correlation`, `upsample_z`,
`streamlines`, `isosurface`, `detect_plane_grains`, `detect_grains_3d`, `bundles`,
`vessels`, `nice_ticks`.

## 2. Inputs

| name | shape / type | meaning |
|---|---|---|
| `U`, `V` | `(nz, ny, nx)` float | in-plane velocity per plane on a REGULAR grid of cells (one PIV window pitch per cell); `U` is +x (columns), `V` is +y (rows, image convention, +down). NaN = no measurement (inside a grain, outside the mosaic, a dropped vector). Any units; the page labels them. |
| `s` | float | `dz / dx`: the plane spacing in cells (`z_step_um / cell_um`). `w = s · W`. One plane → 1. |
| `px_per_cell` | float | pixels per cell, used to scale grain cross-sections from the downsampled image to cells. |
| `grain_planes` | list of `(img, mask)` per plane, each `(h, w)` arrays downsampled by `grain_ds` from the full canvas, either may be None | where the grain pack is detected (Otsu on `img`, or `mask` directly; `img` then only colours the grains). None → no grains. |

## 3. Tunables (`build_geometry`)

| name | default | effect |
|---|---|---|
| `w_mode` | `regularized` | `regularized` (sparse least squares, CG), `trapezoid` (per-column integration), `off` (`W = 0`) |
| `smooth_sigma` | 1.5 cells | Gaussian σ for the NaN-aware smoothing / inpainting before differentiating |
| `lam_z`, `lam_xy` | 0.2, 0.1 | regularisation weights (vertical curvature, in-plane gradient) |
| `z_upsample` | 8 | PCHIP planes inserted between measured planes |
| `iso_pctl` | 90 | speed percentile of the channel iso-surface |
| `vessel_pctl` | 70 | speed percentile defining vessel lumen |
| `seed_stride`, `max_lines`, `seed_min_fraction` | 11, 1400, 0.1 | streamline seeding: cell spacing, cap, skip seeds slower than this fraction of the 95th-percentile speed |
| `grain_ds`, `grain_min_area`, `grain_min_dist` | 4, 200, 16 | grain detector: downsampling of the planes handed in, minimum region area (downsampled px²) and watershed peak separation (downsampled px) |

## 4. Outputs

`geometry` is the page's `GEOM` object: `meta` (grid sizes, `s`, `zspan`, speed range,
grain/vessel/bundle ranges and counts, `names` = one label per measured plane) plus base64
buffers `line_pos` (Float32 ×3), `line_spd` (Float32), `line_ranges` (Uint32 ×2 start/count),
`iso_pos`/`iso_norm` (Float32 ×3), `iso_idx` (Uint32 ×3), `grains` (Float32 ×9:
`x, y, z, a, b, c, phi, inten, n_planes`), `vessels` (Float32 ×5: `x, y, z, radius, speed`),
`bundles` (Float32 ×7: `x, y, z, track_count, dirx, diry, dirz`).

`report` holds the diagnostics: `s_used`, `w_mode`, `cg_info` (0 = converged),
`channel_frac` (volume fraction above the 80th speed percentile), cluster counts,
`w_over_u`, `mean_inplane_speed`, `residual_div_2d` / `residual_div_3d` (mean |div| before
and after adding `W`), `interslice_speed_correlation`, the geometry counts and
`support_fraction` (cells with a measurement nearby).

## 5. Coordinate and unit conventions (load-bearing)

* **Display frame** of every buffer: `X = x cells`, `Y = (ny - 1) - y cells` (y flipped so the
  frame is right-handed with z up), `Z = plane index · s`. The page scales Z by its own
  `zs/s` slider; buffers carry `s` already applied.
* **`W` is solved at unit plane spacing**; streamlines integrate `(W, V, U)` in
  `(zeta, y, x)` space and report speed as `sqrt(u² + v² + (s·W)²)`.
* **Gauge**: `W` is zero-mean per plane (the continuity solve fixes `W` only up to a
  per-column constant).
* **Cells with no support** (Gaussian coverage of finite neighbours < 0.15) are set to zero
  velocity and zero confidence — a solid — not inpainted. Smaller holes are inpainted.
* Velocities keep the caller's units; `units` is a label only.

## 6. Failure modes

* `assemble_grid` never raises on bad rows (non-finite or out-of-range → skipped).
* `w_regularized` with `nz < 2` returns zeros; `isosurface` returns empty arrays when the
  volume has fewer than 2 samples on any axis or the level is outside the data range;
  `vessels` falls back to a 2-D skeleton for a single plane; every geometry function returns
  a `(0, k)` array rather than raising on an empty input.
* `render_html` raises `FileNotFoundError` if the template is missing beside the module.

## 7. Example

```python
import numpy as np
from nodegraph.kernels import flow_viewer as FV
nz, ny, nx = 3, 40, 60
yy, xx = np.mgrid[0:ny, 0:nx]
U = np.stack([2 + k + 0 * xx for k in range(nz)], 0).astype(float)
V = np.stack([0.5 * np.sin(xx / 6.0) for _ in range(nz)], 0)
geom, rep = FV.build_geometry(U, V, s=2.0, px_per_cell=16, plane_names=["z0", "z1", "z2"],
                              max_lines=200, z_upsample=4)
html = FV.render_html(geom, rep, title="demo", subtitle="synthetic", units="flow speed (a.u.)",
                      cell_um=8.0)
open("demo_viewer.html", "w", encoding="utf-8").write(html)
```

## 8. Performance

A 36-position × 11-plane chip (378 × 378 cells) takes about a minute: the regularised CG
solve (1.6 M unknowns) and the 3-D skeletonisation of the 8×-upsampled volume dominate;
streamlines are vectorised over seeds. Memory peaks at the fine volume
(`(nz-1)·z_upsample+1` planes × ny × nx × 4 float32 fields).

## 9. Determinism

Fixed RNG seeds (seed shuffling), no wall-clock input: the same grids give the same page
bytes.

## 10. Dependencies

numpy, scipy (`ndimage`, `interpolate`, `sparse`, `sparse.linalg`), scikit-image
(`measure.marching_cubes`, `measure.regionprops`, `filters`, `feature.peak_local_max`,
`segmentation.watershed`, `morphology.skeletonize`), lazily imported.

## 11. Pipeline wiring

`analysis.piv` (Velocity on, ROI = pore mask) → `io.write_flow_viewer` (`source = piv`,
`grains = mask`, `grain_mask = <inverted pore mask>`), several positions placed by the
payload's `stage_xy_um`. The node derives `s` from `z_step_um` and the window pitch; the
kernel never sees calibration.
