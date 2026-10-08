"""flow_viewer — 3-D flow reconstruction + geometry for a self-contained WebGL viewer.

A port of the lab's granular-flow viewer pipeline (``microfluidic-LLS-Paper/DataAnalysis/
granular_flow_viewer/analysis``: ``reconstruct_3d.py`` → ``precompute_geometry.py`` →
``build_webgl_viewer.py``, built 2026-07-11 and deployed at mcgheelab.com) into one Qt-free,
Dataset-free kernel. The caller hands in **per-plane in-plane velocity grids** (what PIV
measures) and gets back the geometry blob + the HTML page the deployed viewer draws.

Stages (each a public function, so a caller can stop anywhere):

1. :func:`assemble_grid` — scatter per-vector samples onto a regular ``(nz, ny, nx)`` grid
   (duplicates average; empty cells are NaN).
2. :func:`prepare_fields` — NaN-aware smoothing / inpainting of small holes, zero velocity
   where there is no support (solid grains), the per-plane 2-D divergence and a soft
   confidence weight.
3. :func:`w_regularized` / :func:`w_trapezoid` — the out-of-plane velocity from mass
   continuity ``dw/dz = -(du/dx + dv/dy)``, solved at UNIT plane spacing (``W``) and scaled
   by ``s = dz / dx`` (``w = s * W``). Read ``w`` as a continuity-derived STRUCTURAL estimate,
   never as a calibrated velocity: every bit of measured 2-D divergence is attributed to
   out-of-plane flow, and a granular pack also compacts and dilates.
4. :func:`upsample_z` — PCHIP interpolation along z for a continuous volume.
5. :func:`streamlines` / :func:`isosurface` / :func:`detect_grains_3d` / :func:`bundles` /
   :func:`vessels` — the five geometry layers of the viewer, in one DISPLAY frame
   ``X = x cells, Y = (ny-1) - y cells, Z = plane index * s``.
6. :func:`build_geometry` runs 2-5 and packs base64 Float32/Uint32 buffers;
   :func:`render_html` injects them into ``flow_viewer_template.html`` (the deployed page,
   with its dataset strings turned into placeholders) and returns the HTML text.

Units: velocities are whatever the caller passes (the viewer labels them with ``units``);
coordinates are grid CELLS (one PIV window pitch) so the page's geometry stays compact and
the z stretch ``s`` is meaningful. Deterministic: fixed RNG seeds, no wall-clock input.

Deps: numpy, scipy (ndimage, interpolate, sparse); scikit-image for the isosurface, grain
watershed and vessel skeleton (lazily imported inside the functions that need it).
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "flow_viewer_template.html")

#: Columns of the ``grains`` buffer (one row per 3-D grain).
GRAIN_COLUMNS = ("x", "y", "z", "a", "b", "c", "phi", "inten", "n_planes")
#: Columns of the ``vessels`` buffer.
VESSEL_COLUMNS = ("x", "y", "z", "radius", "speed")
#: Columns of the ``bundles`` buffer.
BUNDLE_COLUMNS = ("x", "y", "z", "track_count", "dirx", "diry", "dirz")


def b64(arr: np.ndarray, dtype: Any) -> str:
    """Base64 of ``arr`` as little-endian ``dtype`` bytes (what the page's ``dec()`` reads)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


# ---------------------------------------------------------------------------
# 1. grid assembly
# ---------------------------------------------------------------------------
def assemble_grid(iz: np.ndarray, iy: np.ndarray, ix: np.ndarray, u: np.ndarray,
                  v: np.ndarray, shape: Tuple[int, int, int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter samples onto a ``(nz, ny, nx)`` grid → ``(U, V, count)``.

    Rows that land on the same cell (several timepoints of one window) are AVERAGED; a
    cell nobody hit is NaN in ``U``/``V`` and 0 in ``count``. Rows with a non-finite value or
    an index outside ``shape`` are skipped, not an error — a PIV table legitimately carries
    dropped vectors."""
    nz, ny, nx = shape
    iz = np.asarray(iz, int); iy = np.asarray(iy, int); ix = np.asarray(ix, int)
    u = np.asarray(u, float); v = np.asarray(v, float)
    ok = (np.isfinite(u) & np.isfinite(v) & (iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny)
          & (ix >= 0) & (ix < nx))
    U = np.zeros(shape); V = np.zeros(shape); n = np.zeros(shape)
    np.add.at(U, (iz[ok], iy[ok], ix[ok]), u[ok])
    np.add.at(V, (iz[ok], iy[ok], ix[ok]), v[ok])
    np.add.at(n, (iz[ok], iy[ok], ix[ok]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(n > 0, U / n, np.nan)
        V = np.where(n > 0, V / n, np.nan)
    return U, V, n


# ---------------------------------------------------------------------------
# 2. field preparation
# ---------------------------------------------------------------------------
def nan_gaussian(a: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized-convolution Gaussian smoothing of a 2-D field with NaN holes →
    ``(smoothed, coverage)``. ``coverage`` is the Gaussian-weighted fraction of finite
    neighbours (1 inside solid data, →0 far from any sample)."""
    from scipy.ndimage import gaussian_filter
    w = np.isfinite(a).astype(float)
    a0 = np.where(w > 0, a, 0.0)
    if sigma <= 0:
        return a0, w
    num = gaussian_filter(a0, sigma, mode="nearest")
    cov = gaussian_filter(w, sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(cov > 1e-6, num / np.maximum(cov, 1e-6), 0.0)
    return sm, cov


def prepare_fields(U: np.ndarray, V: np.ndarray, *, smooth_sigma: float = 1.5,
                   conf_soft: float = 0.3, min_support: float = 0.15) -> Dict[str, Any]:
    """Smooth + inpaint ``(nz, ny, nx)`` velocity planes and derive divergence/confidence.

    * NaN holes smaller than the kernel (a dropped vector, a tile seam) are filled by the
      normalized convolution; cells whose Gaussian coverage is below ``min_support`` have no
      measurement anywhere near them — inside a grain, outside the mosaic — and are set to
      ZERO velocity, which is what a solid is. ``support`` reports that coverage.
    * ``div`` is ``du/dx + dv/dy`` per plane at unit cell spacing.
    * ``conf`` is the soft confidence ``speed / (speed + conf_soft)`` (``conf_soft`` in the
      field's own velocity units, scaled to its median speed by the caller) times the
      support, so the ``w`` solve trusts measured, moving fluid and ignores the rest."""
    nz = U.shape[0]
    Us = np.empty_like(U, dtype=float); Vs = np.empty_like(V, dtype=float)
    sup = np.empty_like(U, dtype=float)
    for k in range(nz):
        su, cu = nan_gaussian(U[k], smooth_sigma)
        sv, _ = nan_gaussian(V[k], smooth_sigma)
        keep = cu >= min_support
        Us[k] = np.where(keep, su, 0.0)
        Vs[k] = np.where(keep, sv, 0.0)
        sup[k] = cu
    div = np.stack([np.gradient(Us[k], axis=1) + np.gradient(Vs[k], axis=0)
                    for k in range(nz)])
    speed = np.hypot(Us, Vs)
    conf = (speed / (speed + conf_soft)) * np.clip(sup / max(min_support, 1e-6), 0.0, 1.0)
    return {"U": Us, "V": Vs, "div": div, "speed": speed, "conf": conf, "support": sup}


# ---------------------------------------------------------------------------
# 3. the out-of-plane velocity from continuity
# ---------------------------------------------------------------------------
def _zero_mean_per_plane(W: np.ndarray) -> np.ndarray:
    return W - W.reshape(W.shape[0], -1).mean(axis=1)[:, None, None]


def w_trapezoid(div: np.ndarray) -> np.ndarray:
    """``W`` at unit plane spacing by per-column trapezoidal integration of ``-div`` from
    plane 0 upward; zero-mean gauge per plane."""
    nz = div.shape[0]
    W = np.zeros_like(div, dtype=float)
    for k in range(1, nz):
        W[k] = W[k - 1] - 0.5 * (div[k] + div[k - 1])
    return _zero_mean_per_plane(W)


def w_regularized(div: np.ndarray, conf: np.ndarray, *, lam_z: float = 0.2,
                  lam_xy: float = 0.1, eps: float = 1e-3, maxiter: int = 2000
                  ) -> Tuple[np.ndarray, int]:
    """``W`` at unit plane spacing by regularised 3-D least squares → ``(W, cg_info)``.

    Minimises ``sum_k conf (W[k+1]-W[k] + div_mid)^2 + lam_z |d2W/dk2|^2 + lam_xy |grad_xy W|^2
    + eps |W|^2`` through the normal equations and conjugate gradient. ``cg_info`` is scipy's
    convergence flag (0 = converged). Zero-mean gauge per plane. One plane → zeros."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    nz, ny, nx = div.shape
    if nz < 2:
        return np.zeros_like(div, dtype=float), 0
    N = nz * ny * nx
    idx_all = np.arange(N).reshape(nz, ny, nx)
    rows: List[np.ndarray] = []; cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []; rhs: List[np.ndarray] = []
    r = 0
    for k in range(nz - 1):                                   # data term
        wt = np.sqrt(np.minimum(conf[k], conf[k + 1]) + 1e-6).ravel()
        dmid = (0.5 * (div[k] + div[k + 1])).ravel()
        a = idx_all[k + 1].ravel(); b = idx_all[k].ravel()
        rr = r + np.arange(a.size)
        rows += [rr, rr]; cols += [a, b]; vals += [wt, -wt]
        rhs.append(-wt * dmid); r += a.size
    if lam_z > 0:                                             # vertical curvature
        s = np.sqrt(lam_z)
        for k in range(1, nz - 1):
            n = ny * nx; rr = r + np.arange(n)
            rows += [rr, rr, rr]
            cols += [idx_all[k - 1].ravel(), idx_all[k].ravel(), idx_all[k + 1].ravel()]
            vals += [s * np.ones(n), -2 * s * np.ones(n), s * np.ones(n)]
            rhs.append(np.zeros(n)); r += n
    if lam_xy > 0:                                            # in-plane gradient
        s = np.sqrt(lam_xy)
        for k in range(nz):
            rr = r + np.arange(ny * (nx - 1))
            rows += [rr, rr]
            cols += [idx_all[k, :, 1:].ravel(), idx_all[k, :, :-1].ravel()]
            vals += [s * np.ones(rr.size), -s * np.ones(rr.size)]
            rhs.append(np.zeros(rr.size)); r += rr.size
            rr = r + np.arange((ny - 1) * nx)
            rows += [rr, rr]
            cols += [idx_all[k, 1:, :].ravel(), idx_all[k, :-1, :].ravel()]
            vals += [s * np.ones(rr.size), -s * np.ones(rr.size)]
            rhs.append(np.zeros(rr.size)); r += rr.size
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(r, N))
    bvec = np.concatenate(rhs)
    AtA = (A.T @ A + eps * sp.identity(N)).tocsr()
    Atb = A.T @ bvec
    W, info = spla.cg(AtA, Atb, rtol=1e-6, maxiter=maxiter)
    return _zero_mean_per_plane(np.asarray(W).reshape(nz, ny, nx)), int(info)


def residual_divergence_3d(U: np.ndarray, V: np.ndarray, W_unit: np.ndarray) -> np.ndarray:
    """``du/dx + dv/dy + dW/dk`` with the reconstructed ``W`` (unit plane spacing)."""
    nz = U.shape[0]
    div2d = np.stack([np.gradient(U[k], axis=1) + np.gradient(V[k], axis=0)
                      for k in range(nz)])
    if nz < 2:
        return div2d
    return div2d + np.gradient(W_unit, axis=0)


def channel_connectivity(speed: np.ndarray, pctl: float) -> Dict[str, Any]:
    """Threshold the speed volume at ``pctl`` and label 26-connected clusters: how many,
    whether one spans every plane, and the high-speed volume fraction."""
    from scipy.ndimage import label
    thr = float(np.percentile(speed, pctl))
    mask = speed >= thr
    lab, n = label(mask, structure=np.ones((3, 3, 3), int))
    nz = speed.shape[0]
    clusters = []
    for cid in range(1, n + 1):
        zs = np.where(lab == cid)[0]
        clusters.append({"id": int(cid), "z_extent": int(zs.max() - zs.min() + 1),
                         "voxels": int((lab == cid).sum())})
    clusters.sort(key=lambda d: d["voxels"], reverse=True)
    return {"threshold": thr, "n_clusters": int(n),
            "n_spanning_all_planes": int(sum(1 for c in clusters if c["z_extent"] == nz)),
            "largest": clusters[:5], "highspeed_volume_fraction": float(mask.mean())}


def interslice_correlation(speed: np.ndarray) -> List[float]:
    out = []
    for k in range(speed.shape[0] - 1):
        a, bb = speed[k].ravel(), speed[k + 1].ravel()
        out.append(float(np.corrcoef(a, bb)[0, 1]) if a.std() > 0 and bb.std() > 0 else 0.0)
    return out


# ---------------------------------------------------------------------------
# 4. z upsampling
# ---------------------------------------------------------------------------
def upsample_z(field: np.ndarray, n_between: int) -> Tuple[np.ndarray, np.ndarray]:
    """PCHIP-interpolate ``(nz, ny, nx)`` along z → ``(field_fine, zeta_fine)``; ``zeta`` is in
    measured-plane-index units. One plane (or ``n_between <= 0``) returns the input."""
    from scipy.interpolate import PchipInterpolator
    nz = field.shape[0]
    if nz < 2 or n_between <= 0:
        return np.asarray(field, dtype=np.float32), np.arange(nz, dtype=float)
    zeta = np.arange(nz, dtype=float)
    zf = np.linspace(0.0, nz - 1.0, (nz - 1) * n_between + 1)
    out = PchipInterpolator(zeta, field, axis=0)(zf)
    return np.asarray(out, dtype=np.float32), zf


# ---------------------------------------------------------------------------
# 5. geometry
# ---------------------------------------------------------------------------
def streamlines(Uf: np.ndarray, Vf: np.ndarray, Wf: np.ndarray, zf: np.ndarray, s: float, *,
                seed_stride: int = 11, z_seeds: Optional[Sequence[float]] = None,
                n_steps: int = 70, dt: float = 0.5, min_speed: float = 0.0,
                max_lines: int = 1400, rng_seed: int = 0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bidirectional RK4 streamlines through the ``(zeta, y, x)`` volume (vectorised over
    seeds) → ``(pos (N,3) float32 display coords, speed (N,) float32, ranges (L,2) uint32)``.

    Seeds every ``seed_stride`` cells at the ``z_seeds`` depths (zeta units; default three
    depths across the stack), shuffled with a fixed seed, slow seeds (in-plane speed below
    ``min_speed``) skipped, at most ``max_lines`` kept. A line stops at the volume edge or
    where the flow is stuck. Lines shorter than 6 vertices are dropped."""
    from scipy.interpolate import RegularGridInterpolator
    nzf, NY, NX = Uf.shape
    gz = (zf, np.arange(NY, dtype=float), np.arange(NX, dtype=float))
    kw = dict(bounds_error=False, fill_value=0.0)
    fu = RegularGridInterpolator(gz, Uf, **kw)
    fv = RegularGridInterpolator(gz, Vf, **kw)
    fw = RegularGridInterpolator(gz, Wf, **kw)

    def vel(P: np.ndarray) -> np.ndarray:                     # d(zeta, y, x)/dt
        return np.column_stack([fw(P), fv(P), fu(P)])

    if z_seeds is None:
        zmax = float(zf[-1])
        z_seeds = (0.5 * zmax / 4, zmax / 2, zmax - 0.5 * zmax / 4) if zmax > 0 else (0.0,)
    my = min(6, max(0, (NY - 1) // 4)); mx = min(6, max(0, (NX - 1) // 4))   # edge margin
    seeds = [[float(zc), float(yy), float(xx)] for zc in z_seeds
             for yy in range(my, max(NY - my, my + 1), max(1, seed_stride))
             for xx in range(mx, max(NX - mx, mx + 1), max(1, seed_stride))]
    if not seeds:
        return (np.zeros((0, 3), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 2), np.uint32))
    seeds_a = np.array(seeds, float)
    np.random.default_rng(rng_seed).shuffle(seeds_a, axis=0)
    sp0 = np.hypot(fu(seeds_a), fv(seeds_a))
    seeds_a = seeds_a[sp0 >= min_speed][: max(1, 2 * max_lines)]
    N = len(seeds_a)
    if N == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 2), np.uint32))
    lo = np.array([zf[0], 0.0, 0.0]); hi = np.array([zf[-1], NY - 1e-6, NX - 1e-6])
    trajs = {}
    for sign in (+1.0, -1.0):
        P = seeds_a.copy(); alive = np.ones(N, bool)
        traj = np.full((n_steps, N, 3), np.nan)
        for i in range(n_steps):
            k1 = vel(P); k2 = vel(P + sign * dt / 2 * k1)
            k3 = vel(P + sign * dt / 2 * k2); k4 = vel(P + sign * dt * k3)
            step = sign * dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Pn = P + step
            inside = np.all((Pn >= lo) & (Pn <= hi), axis=1)
            moving = np.linalg.norm(step, axis=1) > 1e-6
            alive &= inside & moving
            if not alive.any():
                break
            P = np.where(alive[:, None], Pn, P)
            traj[i, alive] = Pn[alive]
        trajs[sign] = traj
    pos: List[np.ndarray] = []; ranges: List[Tuple[int, int]] = []
    cur = 0
    for n in range(N):
        if len(ranges) >= max_lines:
            break
        fwd = trajs[1.0][:, n]; fwd = fwd[np.isfinite(fwd[:, 0])]
        bwd = trajs[-1.0][:, n]; bwd = bwd[np.isfinite(bwd[:, 0])]
        line = np.concatenate([fwd[::-1], bwd], axis=0) if len(fwd) + len(bwd) else fwd
        if len(line) < 6:
            continue
        pos.append(line); ranges.append((cur, len(line))); cur += len(line)
    if not pos:
        return (np.zeros((0, 3), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 2), np.uint32))
    P_all = np.concatenate(pos, axis=0)
    u = fu(P_all); v = fv(P_all); w = s * fw(P_all)
    spd = np.sqrt(u * u + v * v + w * w).astype(np.float32)
    disp = np.column_stack([P_all[:, 2], (NY - 1) - P_all[:, 1], P_all[:, 0] * s]
                           ).astype(np.float32)
    return disp, spd, np.array(ranges, np.uint32)


def isosurface(speed3d_f: np.ndarray, zf: np.ndarray, s: float, *, pctl: float = 90.0,
               ds: int = 0, voxel_budget: int = 2_500_000
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marching cubes on the fine speed volume at its ``pctl`` percentile →
    ``(verts (N,3) display, normals (N,3), faces (F,3) uint32)``; empty when the volume is
    too thin (fewer than 2 samples on any axis) or flat. ``ds = 0`` picks the in-plane
    subsampling that keeps the marched volume under ``voxel_budget`` voxels (never below
    2), so a chip-sized grid does not put tens of MB of triangles in the page."""
    from skimage.measure import marching_cubes
    NY = speed3d_f.shape[1]
    if ds <= 0:
        ds = max(2, int(np.ceil(np.sqrt(speed3d_f.size / float(voxel_budget)))))
    sub = speed3d_f[:, ::ds, ::ds]
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
             np.zeros((0, 3), np.uint32))
    if min(sub.shape) < 2:
        return empty
    lev = float(np.percentile(speed3d_f, pctl))
    if not (sub.min() < lev < sub.max()):
        return empty
    dz = (float(zf[1] - zf[0]) if len(zf) > 1 else 1.0) * s
    verts, faces, normals, _ = marching_cubes(np.ascontiguousarray(sub, dtype=np.float32),
                                              level=lev, spacing=(dz, ds, ds))
    disp = np.column_stack([verts[:, 2], (NY - 1) - verts[:, 1], verts[:, 0]]).astype(np.float32)
    nrm = np.column_stack([normals[:, 2], -normals[:, 1], normals[:, 0]]).astype(np.float32)
    return disp, nrm, faces.astype(np.uint32)


def detect_plane_grains(img: Optional[np.ndarray], mask: Optional[np.ndarray], *,
                        ds: int, px_per_cell: float, min_area: int = 200,
                        min_dist: int = 16, smooth: float = 1.6) -> List[Dict[str, float]]:
    """Grain cross-sections on ONE plane, from an intensity image (Otsu) and/or a binary
    mask, both already downsampled by ``ds``. Distance-transform watershed splits touching
    grains; each region → centroid / equivalent radius / ellipse axes / orientation /
    intensity, in grid CELLS (``ds`` px per sample, ``px_per_cell`` px per cell)."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, threshold_otsu
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    if mask is None and img is None:
        return []
    inten = None
    if img is not None:
        inten = gaussian(np.asarray(img, float), smooth, preserve_range=True)
    if mask is None:
        finite = np.isfinite(inten)
        if not finite.any() or inten[finite].max() <= inten[finite].min():
            return []
        try:
            thr = threshold_otsu(inten[finite])
        except ValueError:
            return []
        m = inten > thr
    else:
        m = np.asarray(mask).astype(bool)
    m = ndi.binary_opening(m, iterations=1)
    if not m.any():
        return []
    dist = ndi.distance_transform_edt(m)
    coords = peak_local_max(dist, min_distance=max(1, int(min_dist)), labels=m.astype(int),
                            exclude_border=False)
    markers = np.zeros(dist.shape, int)
    for i, (rr, cc) in enumerate(coords, 1):
        markers[rr, cc] = i
    if markers.max() == 0:
        markers, _ = ndi.label(m)
    lab = watershed(-dist, markers, mask=m)
    scale = ds / float(px_per_cell)
    out = []
    for rp in regionprops(lab, intensity_image=inten if inten is not None else dist):
        if rp.area < min_area:
            continue
        y0, x0 = rp.centroid
        r_eq = float(np.sqrt(rp.area / np.pi))
        major = float(rp.axis_major_length or 2 * r_eq)
        minor = float(rp.axis_minor_length or 2 * r_eq)
        if minor <= 1e-6:
            minor = major
        out.append(dict(x=x0 * scale, y=y0 * scale, r=r_eq * scale, major=major * scale,
                        minor=minor * scale, angle=float(rp.orientation),
                        inten=float(rp.intensity_mean), area=float(rp.area) * scale * scale))
    return out


def detect_grains_3d(planes: Sequence[Tuple[Optional[np.ndarray], Optional[np.ndarray]]], *,
                     ds: int, px_per_cell: float, ny_cells: int, s: float,
                     link_frac: float = 0.55, min_area: int = 200, min_dist: int = 16
                     ) -> np.ndarray:
    """Per-plane cross-sections (:func:`detect_plane_grains`) linked across adjacent planes
    by centroid proximity (union-find) → ``(N, 9)`` float32 rows in :data:`GRAIN_COLUMNS`
    order: display centre, super-ellipsoid semi-axes ``a >= b``, ``c = b`` (the z extent is
    not measured), in-plane orientation ``phi`` in display space, mean intensity and the
    number of planes the grain was seen in (its confidence). A coarse structural
    reconstruction — grains are placed at their brightest / largest cross-section."""
    per_plane: List[List[Dict[str, float]]] = []
    for k, (img, mask) in enumerate(planes):
        g = detect_plane_grains(img, mask, ds=ds, px_per_cell=px_per_cell,
                                min_area=min_area, min_dist=min_dist)
        for d in g:
            d["k"] = float(k)
        per_plane.append(g)
    flat = [d for g in per_plane for d in g]
    if not flat:
        return np.zeros((0, 9), np.float32)
    parent = list(range(len(flat)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    offs = np.cumsum([0] + [len(g) for g in per_plane])
    for k in range(len(per_plane) - 1):
        A, B = per_plane[k], per_plane[k + 1]
        if not A or not B:
            continue
        ax_ = np.array([a["x"] for a in A]); ay_ = np.array([a["y"] for a in A])
        ar_ = np.array([a["r"] for a in A])
        for j, bgr in enumerate(B):
            dr = np.hypot(ax_ - bgr["x"], ay_ - bgr["y"])
            lim = link_frac * np.maximum(ar_, bgr["r"])
            for i in np.nonzero(dr < lim)[0]:
                parent[find(int(offs[k] + i))] = find(int(offs[k + 1] + j))
    clusters: Dict[int, List[Dict[str, float]]] = {}
    for idx, d in enumerate(flat):
        clusters.setdefault(find(idx), []).append(d)
    grains = []
    for members in clusters.values():
        w = np.array([m["inten"] * m["area"] for m in members], float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(members), 1.0 / len(members))
        xs = np.array([m["x"] for m in members]); ys = np.array([m["y"] for m in members])
        ks = np.array([m["k"] for m in members])
        ref = max(members, key=lambda m: m["inten"] * m["area"])
        xc = float((w * xs).sum()); yc = float((w * ys).sum()); zc = float((w * ks).sum())
        a = 0.5 * ref["major"]
        b = 0.5 * max(ref["minor"], 0.35 * ref["major"])
        th = ref["angle"]
        phi = float(np.arctan2(np.cos(th), -np.sin(th)))
        npl = len(set(ks.tolist()))
        inten = float(np.mean([m["inten"] for m in members]))
        grains.append((xc, (ny_cells - 1) - yc, zc * s, a, b, b, phi, inten, float(npl)))
    return np.array(grains, np.float32).reshape(-1, 9)


def bundles(pos: np.ndarray, ranges: np.ndarray, *, K: int = 20, thresh: float = 22.0,
            min_size: int = 4, dense: int = 34) -> np.ndarray:
    """QuickBundles-style clustering of the streamlines (mean direct/flipped distance below
    ``thresh`` display units joins a tract to a bundle); each bundle with at least
    ``min_size`` tracts → ``dense`` tube nodes ``(N, 7)`` in :data:`BUNDLE_COLUMNS` order."""
    tracts = []
    for st, ct in ranges:
        Pp = pos[st:st + ct]
        seg = np.linalg.norm(np.diff(Pp, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] < 1e-3:
            continue
        u = np.linspace(0, cum[-1], K)
        tracts.append(np.stack([np.interp(u, cum, Pp[:, d]) for d in range(3)], axis=1))
    if not tracts:
        return np.zeros((0, 7), np.float32)
    cents: List[np.ndarray] = []; counts: List[int] = []
    for R in tracts:
        best, bd, bflip = -1, 1e18, False
        for ci, Cc in enumerate(cents):
            dd = np.linalg.norm(R - Cc, axis=1).mean()
            df = np.linalg.norm(R[::-1] - Cc, axis=1).mean()
            d = min(dd, df)
            if d < bd:
                bd, best, bflip = d, ci, df < dd
        if bd < thresh:
            Rr = R[::-1] if bflip else R
            n = counts[best]
            cents[best] = (cents[best] * n + Rr) / (n + 1); counts[best] += 1
        else:
            cents.append(R.copy()); counts.append(1)
    nodes = []
    for Cc, cnt in zip(cents, counts):
        if cnt < min_size:
            continue
        seg = np.linalg.norm(np.diff(Cc, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        u = np.linspace(0, cum[-1], dense)
        Cd = np.stack([np.interp(u, cum, Cc[:, d]) for d in range(3)], axis=1)
        tang = np.gradient(Cd, axis=0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        for p, t in zip(Cd, tang):
            nodes.append((p[0], p[1], p[2], float(cnt), t[0], t[1], t[2]))
    return np.array(nodes, np.float32).reshape(-1, 7)


def vessels(speed3d_f: np.ndarray, zf: np.ndarray, s: float, *, pctl: float = 70.0,
            ds: int = 2, max_nodes: int = 9000) -> np.ndarray:
    """The flow→vasculature analogy: threshold the fine speed volume into "lumen" at
    ``pctl``, keep sizeable components, skeletonise, radius = local half-width →
    ``(N, 5)`` nodes in :data:`VESSEL_COLUMNS` order. A visual device, not a model."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    NY = speed3d_f.shape[1]
    sub = speed3d_f[:, ::ds, ::ds]
    thr = float(np.percentile(speed3d_f, pctl))
    mask = sub > thr
    if not mask.any():
        return np.zeros((0, 5), np.float32)
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), int))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = sizes >= max(40, int(0.0006 * mask.size))
    mask = keep[lab]
    if not mask.any():
        return np.zeros((0, 5), np.float32)
    dz_disp = (float(zf[1] - zf[0]) if len(zf) > 1 else 1.0) * s
    dt = ndi.distance_transform_edt(mask, sampling=(dz_disp, ds, ds))
    if mask.shape[0] >= 2:
        skel = skeletonize(mask)
    else:
        skel = skeletonize(mask[0])[None]
    pts = np.argwhere(skel)
    if len(pts) > max_nodes:
        pts = pts[np.linspace(0, len(pts) - 1, max_nodes).astype(int)]
    out = []
    for (k, yy, xx) in pts:
        r = float(dt[k, yy, xx])
        if r < 0.4:
            continue
        out.append((xx * ds, (NY - 1) - yy * ds, zf[k] * s, r, float(sub[k, yy, xx])))
    return np.array(out, np.float32).reshape(-1, 5)


# ---------------------------------------------------------------------------
# 6. build + render
# ---------------------------------------------------------------------------
def build_geometry(U: np.ndarray, V: np.ndarray, *, s: float, px_per_cell: float,
                   plane_names: Sequence[str], w_mode: str = "regularized",
                   smooth_sigma: float = 1.5, lam_z: float = 0.2, lam_xy: float = 0.1,
                   z_upsample: int = 8, iso_pctl: float = 90.0, vessel_pctl: float = 70.0,
                   seed_stride: int = 11, max_lines: int = 1400,
                   seed_min_fraction: float = 0.1,
                   grain_planes: Optional[Sequence[Tuple[Optional[np.ndarray],
                                                         Optional[np.ndarray]]]] = None,
                   grain_ds: int = 4, grain_min_area: int = 200, grain_min_dist: int = 16,
                   progress: Optional[Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the whole pipeline on ``(nz, ny, nx)`` in-plane velocity grids (NaN = no
    measurement) → ``(geometry, report)``. ``geometry`` is the page's ``GEOM`` object (a
    ``meta`` dict + base64 buffers); ``report`` carries the reconstruction diagnostics
    (``s_used``, the high-speed volume fraction, ``|w|/|u|``, residual divergences, the
    inter-slice correlation, counts). ``progress(k, n, text)`` is called between stages."""
    def _p(k: int, text: str) -> None:
        if progress is not None:
            progress(k, 7, text)
    nz, ny, nx = U.shape
    finite = np.isfinite(U) & np.isfinite(V)
    med_speed = float(np.median(np.hypot(U[finite], V[finite]))) if finite.any() else 1.0
    conf_soft = 0.3 * med_speed if med_speed > 0 else 0.3
    _p(0, "smoothing the velocity planes")
    fp = prepare_fields(U, V, smooth_sigma=smooth_sigma, conf_soft=conf_soft)
    Us, Vs, div, conf, speed = fp["U"], fp["V"], fp["div"], fp["conf"], fp["speed"]
    _p(1, f"out-of-plane velocity ({w_mode})")
    cg_info = 0
    if w_mode == "trapezoid" and nz > 1:
        W = w_trapezoid(div)
    elif w_mode == "regularized" and nz > 1:
        W, cg_info = w_regularized(div, conf, lam_z=lam_z, lam_xy=lam_xy)
    else:
        W = np.zeros_like(Us)
    w_phys = s * W
    res_in = residual_divergence_3d(Us, Vs, np.zeros_like(W))
    res_w = residual_divergence_3d(Us, Vs, W)
    _p(2, "interpolating the volume")
    Uf, zf = upsample_z(Us, z_upsample)
    Vf, _ = upsample_z(Vs, z_upsample)
    Wf, _ = upsample_z(W, z_upsample)
    speed3d_f = np.sqrt(Uf ** 2 + Vf ** 2 + (s * Wf) ** 2).astype(np.float32)
    conn = channel_connectivity(speed3d_f, 80.0)
    _p(3, "tracing streamlines")
    sp_ref = float(np.percentile(speed[speed > 0], 95)) if (speed > 0).any() else 0.0
    pos, spd, ranges = streamlines(Uf, Vf, Wf, zf, s, seed_stride=seed_stride,
                                   min_speed=seed_min_fraction * sp_ref, max_lines=max_lines)
    _p(4, "channel iso-surface")
    ivp, ivn, ifc = isosurface(speed3d_f, zf, s, pctl=iso_pctl)
    _p(5, "grains")
    if grain_planes is not None:
        grains = detect_grains_3d(grain_planes, ds=grain_ds, px_per_cell=px_per_cell,
                                  ny_cells=ny, s=s, min_area=grain_min_area,
                                  min_dist=grain_min_dist)
    else:
        grains = np.zeros((0, 9), np.float32)
    _p(6, "bundles + vessels")
    bnodes = bundles(pos, ranges) if len(ranges) else np.zeros((0, 7), np.float32)
    ves = vessels(speed3d_f, zf, s, pctl=vessel_pctl)
    inplane = float(speed.mean()) if speed.size else 0.0
    w_over_u = float(np.abs(w_phys).mean() / (inplane + 1e-9)) if nz > 1 else 0.0
    meta = dict(
        nx=int(nx), ny=int(ny), nz=int(nz), s=float(s), px_per_cell=float(px_per_cell),
        zspan=float((nz - 1) * s),
        speed_min=float(spd.min()) if spd.size else 0.0,
        speed_max=float(np.percentile(spd, 99)) if spd.size else 1.0,
        grain_inten_min=float(grains[:, 7].min()) if len(grains) else 0.0,
        grain_inten_max=float(grains[:, 7].max()) if len(grains) else 1.0,
        grain_a_max=float(grains[:, 3].max()) if len(grains) else 1.0,
        n_lines=int(len(ranges)), n_grains=int(len(grains)),
        n_grains_multi=int((grains[:, 8] >= 2).sum()) if len(grains) else 0,
        n_vessels=int(len(ves)), vessel_r_max=float(ves[:, 3].max()) if len(ves) else 1.0,
        vessel_spd_max=float(np.percentile(ves[:, 4], 99)) if len(ves) else 1.0,
        n_bundle_nodes=int(len(bnodes)),
        bundle_count_max=float(bnodes[:, 3].max()) if len(bnodes) else 1.0,
        names=[str(n) for n in plane_names])
    geometry = dict(
        meta=meta,
        line_pos=b64(pos, np.float32), line_spd=b64(spd, np.float32),
        line_ranges=b64(ranges, np.uint32),
        iso_pos=b64(ivp, np.float32), iso_norm=b64(ivn, np.float32), iso_idx=b64(ifc, np.uint32),
        grains=b64(grains, np.float32), vessels=b64(ves, np.float32),
        bundles=b64(bnodes, np.float32))
    report = dict(
        s_used=float(s), w_mode=w_mode, cg_info=int(cg_info),
        channel_frac=float(conn["highspeed_volume_fraction"]),
        n_channel_clusters=int(conn["n_clusters"]),
        n_spanning_all_planes=int(conn["n_spanning_all_planes"]),
        w_over_u=round(w_over_u, 4),
        mean_inplane_speed=round(inplane, 6),
        residual_div_2d=round(float(np.abs(res_in).mean()), 6),
        residual_div_3d=round(float(np.abs(res_w).mean()), 6),
        interslice_speed_correlation=[round(x, 4) for x in interslice_correlation(speed)],
        n_lines=meta["n_lines"], n_grains=meta["n_grains"], n_vessels=meta["n_vessels"],
        n_bundle_nodes=meta["n_bundle_nodes"],
        support_fraction=round(float((fp["support"] >= 0.15).mean()), 4))
    return geometry, report


def nice_ticks(extent_um: float, cell_um: float, *, target: int = 5) -> Tuple[List[float], List[str]]:
    """Axis ticks for a ``extent_um``-long axis → ``(positions in cells, labels in µm)``."""
    if not (extent_um > 0) or not (cell_um > 0):
        return [0.0], ["0"]
    raw = extent_um / max(target, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = float(min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw),
                     default=10 * mag))
    vals = np.arange(0.0, extent_um - 1e-9, step)
    pos = [float(v / cell_um) for v in vals]
    lab = [("%g" % v) for v in vals]
    return pos, lab


def render_html(geometry: Dict[str, Any], report: Dict[str, Any], *, title: str,
                subtitle: str, units: str, cell_um: float, x_label: str = "x (µm)",
                y_label: str = "y (µm)", snapshot_name: str = "flow_view.png",
                template_path: str = TEMPLATE_PATH) -> str:
    """Fill the viewer template → the complete self-contained HTML text."""
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    meta = geometry["meta"]
    xt, xtl = nice_ticks(meta["nx"] * cell_um, cell_um)
    yt, ytl = nice_ticks(meta["ny"] * cell_um, cell_um)
    extra = {"s_used": report["s_used"], "channel_frac": report["channel_frac"],
             "w_over_u": report["w_over_u"]}
    rep = {k: v for k, v in (("__DATA__", json.dumps(geometry)),
                             ("__META__", json.dumps(extra)),
                             ("__SUSED__", "%g" % float(report["s_used"])),
                             ("__TITLE__", _esc(title)), ("__SUBTITLE__", _esc(subtitle)),
                             ("__UNITS__", _esc(units)),
                             ("__XT__", json.dumps(xt)), ("__XTL__", json.dumps(xtl)),
                             ("__YT__", json.dumps(yt)), ("__YTL__", json.dumps(ytl)),
                             ("__XLABEL__", json.dumps(x_label)),
                             ("__YLABEL__", json.dumps(y_label)),
                             ("__SNAPNAME__", json.dumps(snapshot_name)))}
    for k, v in rep.items():
        html = html.replace(k, v)
    return html


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
