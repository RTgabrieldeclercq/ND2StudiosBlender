"""bead_detect — Bead / particle detection kernel (self-contained, portable).

WHAT THIS IS
------------
The first granule-separation stage: detect bead centroids in one already-prepared
single-channel ``(Z, H, W)`` volume and return them as an ``(N, 3)`` point cloud in
``(z, y, x)`` VOXEL order plus the ``List[Dict]`` DATA rows.

WHERE THE REAL MATH LIVES
-------------------------
IN-REPO. The blob detector is a hand-written LoG / connected-component particle
detector (``ParticleDetector``) with numba-accelerated sub-voxel refinement kernels
(3-point parabola for the LoG path; radial-symmetry, Liu et al. 2013, for the TPT
path). It uses ``scipy.ndimage`` for LoG / labeling / max-filter and ``numba`` for the
sub-voxel kernels. No external tracking package is required. Richardson-Lucy PSF
deconvolution (``skimage.restoration``) is only touched when ``cfg.psf`` is set, which
this wrapper never sets, so skimage is NOT a hard dependency of the compute path.

PROVENANCE (branch: Version-1.45)
---------------------------------
Vendored verbatim by byte-copy from:
  * nd2studios/backend/analysis/bead_detect.py      (wrapper: detect_beads + helpers)
  * nd2studios/backend/serialtrack/detection.py     (ParticleDetector + numba kernels)
  * nd2studios/backend/serialtrack/config.py         (DetectionConfig, DetectionMethod)
  * nd2studios/backend/analysis/granule_types.py     (make_point_rows, NOISE_LABEL)

Vendored verbatim; imports nothing from nd2studios; caller owns all prep (no file I/O,
no per-multipoint/per-timepoint looping, no crop/downsample/registration/exclusion).

EDITS MADE (per extraction rules)
---------------------------------
* (rule a) Removed ``from .config import ...`` (detection.py) and
  ``from nd2studios.backend.analysis.granule_types import make_point_rows`` (bead_detect.py);
  those symbols are now defined in this file.
* (rule a) ``_require_backends`` no longer imports the serialtrack modules from
  nd2studios; it keeps its friendly find_spec dependency gate and returns the
  in-file ``DetectionConfig / DetectionMethod / ParticleDetector``.

DROPPED members (rule c — NOT on the compute path):
  * config.py: enums ``GlobalSolver``, ``LocalSolver``, ``TrackingMode`` and dataclasses
    ``TrajectoryConfig``, ``TrackingConfig`` (tracking-solver config, unused by detection).
  * granule_types.py: everything except ``make_point_rows`` + ``NOISE_LABEL`` — i.e. the
    ``GranuleBoundary`` / ``GranuleTessellation`` dataclasses, the ``*_ATTR`` record-storage
    string constants, ``points_from_rows``, and the ``GRANULE_PALETTE`` / ``granule_color``
    viewer-coloring helpers (all downstream / storage / UI, not on this kernel's path).
  Nothing on the detection compute path was removed or altered.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
import numba as nb


# ══════════════════════════════════════════════════════════════════════════════
#  Vendored from nd2studios/backend/serialtrack/config.py
# ══════════════════════════════════════════════════════════════════════════════
class DetectionMethod(IntEnum):
    """Particle detection strategy."""
    TPT = 1       # Blob → centroid → radial symmetry sub-pixel
    TRACTRAC = 2  # LoG blob → local max → 2nd-order poly sub-pixel


@dataclass
class DetectionConfig:
    """Particle detection / localization parameters.

    Replaces MATLAB ``BeadPara`` struct.  All lengths are in pixels.
    Works for both 2-D and 3-D images; dimension is inferred at runtime.
    """
    method: DetectionMethod = DetectionMethod.TRACTRAC
    threshold: float = 0.4
    bead_radius: float = 3.0   # 0 → use regionprops centroid only
    min_size: int = 2          # min blob volume (3D) or area (2D) [px^d]
    max_size: int = 1000
    color: str = "white"       # foreground colour: "white" | "black"
    # Optional PSF deconvolution (Richardson-Lucy)
    psf: Optional[np.ndarray] = None
    deconv_iters: int = 6
    # TPT / radial-symmetry params (3-D only)
    win_size: Tuple[int, ...] = (5, 5, 5)
    dccd: Tuple[float, ...] = (1.0, 1.0, 1.0)
    abc: Tuple[float, ...] = (1.0, 1.0, 1.0)
    rand_noise: float = 1e-7


# ============================================================================
#  Vendored from nd2studios/backend/serialtrack/detection.py (numba kernels + ParticleDetector)
# ============================================================================
# ─────────────────────────────────────────────────────────────
#  Numba-accelerated sub-pixel localization kernels
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _fspecial_log(sigma: float, ndim: int) -> np.ndarray:
    """MATLAB ``fspecial('log', ceil(sigma)*2+1, sigma)``, generalised to n-D.

    ``f_detect_particles.m`` / ``f_detect_particles3.m`` build the
    Laplacian-of-Gaussian at a *hard* half-width of ``ceil(sigma)``, i.e. only
    7 taps for the σ=3 the examples use.  That truncation is not incidental:
    it is what makes the filter respond at bead scale rather than at 4σ, and
    reproducing it is worth ~1.8× in localisation accuracy and ~25% more
    particles found versus ``scipy.ndimage.gaussian_laplace`` (which truncates
    at 4σ).

    Returned kernel sums to zero, as MATLAB's does.
    """
    half = int(np.ceil(sigma))
    n = 2 * half + 1
    ax = np.arange(-half, half + 1, dtype=np.float64)
    grids = np.meshgrid(*([ax] * ndim), indexing="ij")
    r2 = sum(g * g for g in grids)
    std2 = float(sigma) ** 2

    h = np.exp(-r2 / (2.0 * std2))
    h[h < np.finfo(np.float64).eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h = h / s
    # ∇²G = (r² - d·σ²)/σ⁴ · G   (d = ndim; MATLAB's 2-D form uses 2·σ²)
    h1 = h * (r2 - ndim * std2) / (std2 ** 2)
    return h1 - h1.sum() / h1.size


@nb.njit(cache=True)
def _subpixel_poly_2d(log_img, xs, ys):
    """3-point parabola sub-pixel refinement for 2-D peaks.

    For each peak at integer coords (xs[i], ys[i]) in *log_img*,
    fit y = a + bx + cx² along each axis and return the shift to
    the parabola vertex.

    Returns (dx, dy) arrays — shifts to add to integer coords.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    h, w = log_img.shape  # note: (x-dim, y-dim) in our convention

    for i in range(n):
        xi, yi = xs[i], ys[i]
        # x-direction
        if 0 < xi < h - 1:
            a = log_img[xi - 1, yi]
            b = log_img[xi, yi]
            c = log_img[xi + 1, yi]
            d = 2.0 * (a - 2.0 * b + c)
            dx[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y-direction
        if 0 < yi < w - 1:
            a = log_img[xi, yi - 1]
            b = log_img[xi, yi]
            c = log_img[xi, yi + 1]
            d = 2.0 * (a - 2.0 * b + c)
            dy[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
    return dx, dy


@nb.njit(cache=True)
def _subpixel_poly_3d(log_img, xs, ys, zs):
    """3-point parabola sub-pixel refinement for 3-D peaks.

    Returns (dx, dy, dz) arrays.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)
    sx, sy, sz = log_img.shape

    for i in range(n):
        xi, yi, zi = xs[i], ys[i], zs[i]
        # x
        if 0 < xi < sx - 1:
            a, b, c = log_img[xi-1,yi,zi], log_img[xi,yi,zi], log_img[xi+1,yi,zi]
            d = 2.0*(a - 2.0*b + c)
            dx[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y
        if 0 < yi < sy - 1:
            a, b, c = log_img[xi,yi-1,zi], log_img[xi,yi,zi], log_img[xi,yi+1,zi]
            d = 2.0*(a - 2.0*b + c)
            dy[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
        # z
        if 0 < zi < sz - 1:
            a, b, c = log_img[xi,yi,zi-1], log_img[xi,yi,zi], log_img[xi,yi,zi+1]
            d = 2.0*(a - 2.0*b + c)
            dz[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dz[i] = 0.0
    return dx, dy, dz


@nb.njit(parallel=True, cache=True)
def _radial_symmetry_3d(patches, half_win, dccd, abc):
    """Radial-symmetry sub-voxel localization (Liu et al. 2013).

    Port of ``radialcenter3dvec.m``.  Finds the point that minimises the
    intensity-gradient-weighted sum of squared perpendicular distances to the
    gradient lines::

        [Σ qᵢ(I - nᵢnᵢᵀ)] c = Σ qᵢ(I - nᵢnᵢᵀ) pᵢ ,   qᵢ = |∇I|² / |pᵢ - p̄|

    Two origins are in play and they are **not** the same one:

    * ``pᵢ`` — the voxel position, measured from the **patch centre**.  This is
      the origin the returned offset is expressed in, so the caller must add it
      to the *integer* voxel the patch was cut around.
    * ``p̄`` — the intensity-weighted centroid, which appears **only** inside the
      weight denominator ``qᵢ`` (MATLAB ``d``, `radialcenter3dvec.m:78`).

    Measuring ``pᵢ`` from the centroid instead of the patch centre — which an
    earlier revision did — silently drops the centroid offset from the answer
    and inflates the localisation error by ~5× (0.064 px → 0.34 px on the
    SerialTrack3D synthetic beads).  See ``scripts/_serialtrack_validate.py``.

    Parameters
    ----------
    patches : (N, w0, w1, w2)  float64 array of image patches
    half_win : (3,) int array   half window sizes
    dccd : (3,) float array     voxel spacing along each patch axis
    abc  : (3,) float array     anisotropy factor per patch axis

    Returns
    -------
    d0, d1, d2 : (N,) sub-voxel shifts along patch axes 0/1/2, relative to the
        patch centre.

    Notes
    -----
    ``dccd``/``abc`` are indexed *by patch axis* here.  MATLAB pairs its
    ``dccd(1)``/``abc(1)`` with array dimension **2** (because ``meshgrid``
    makes ``px`` vary along dim 2), so the two agree exactly for the isotropic
    ``[1,1,1]`` values every shipped SerialTrack example uses, and differ only
    for anisotropic settings — where MATLAB's own indexing is inconsistent.
    """
    N = patches.shape[0]
    w0, w1, w2 = patches.shape[1], patches.shape[2], patches.shape[3]
    d0 = np.zeros(N, dtype=np.float64)
    d1 = np.zeros(N, dtype=np.float64)
    d2 = np.zeros(N, dtype=np.float64)
    a, b, c = abc[0], abc[1], abc[2]
    s0, s1, s2 = dccd[0], dccd[1], dccd[2]

    for pi in nb.prange(N):
        # --- intensity-weighted centroid (patch-centre origin, unscaled by abc,
        #     exactly as MATLAB's xm/ym/zm) ---
        m0 = 0.0; m1 = 0.0; m2 = 0.0; tot = 0.0
        for i0 in range(w0):
            p0 = (i0 - (w0 - 1) * 0.5) * s0
            for i1 in range(w1):
                p1 = (i1 - (w1 - 1) * 0.5) * s1
                for i2 in range(w2):
                    v = patches[pi, i0, i1, i2]
                    m0 += v * p0
                    m1 += v * p1
                    m2 += v * (i2 - (w2 - 1) * 0.5) * s2
                    tot += v
        if tot < 1e-30:
            continue
        xm = m0 / tot;  ym = m1 / tot;  zm = m2 / tot

        # --- build 3×3 normal system from gradient votes ---
        A00=0.;A01=0.;A02=0.;A11=0.;A12=0.;A22=0.
        B0=0.;B1=0.;B2=0.

        for i0 in range(1, w0-1):
            xp = (i0 - (w0 - 1) * 0.5) * s0 / a       # patch-centre-relative
            for i1 in range(1, w1-1):
                yp = (i1 - (w1 - 1) * 0.5) * s1 / b
                for i2 in range(1, w2-1):
                    zp = (i2 - (w2 - 1) * 0.5) * s2 / c
                    # central differences over the abc-scaled spacing, matching
                    # MATLAB's dx = dxccd/a
                    gu = (patches[pi,i0+1,i1,i2] - patches[pi,i0-1,i1,i2])/(s0/a)
                    gv = (patches[pi,i0,i1+1,i2] - patches[pi,i0,i1-1,i2])/(s1/b)
                    gw = (patches[pi,i0,i1,i2+1] - patches[pi,i0,i1,i2-1])/(s2/c)
                    gm = np.sqrt(gu*gu + gv*gv + gw*gw)
                    if gm < 1e-30:
                        continue
                    gu /= gm; gv /= gm; gw /= gm

                    # weight denominator: distance to the intensity centroid
                    ex = xp - xm; ey = yp - ym; ez = zp - zm
                    dd = np.sqrt(ex*ex + ey*ey + ez*ez)
                    if dd < 1e-30:
                        continue
                    q = gm*gm / dd

                    A00 += q*(1-gu*gu); A01 += q*(-gu*gv); A02 += q*(-gu*gw)
                    A11 += q*(1-gv*gv); A12 += q*(-gv*gw); A22 += q*(1-gw*gw)
                    dot = gu*xp + gv*yp + gw*zp
                    B0 += q*(xp - gu*dot)
                    B1 += q*(yp - gv*dot)
                    B2 += q*(zp - gw*dot)

        # --- solve 3×3 symmetric system via Cramer ---
        det = (A00*(A11*A22 - A12*A12)
             - A01*(A01*A22 - A02*A12)
             + A02*(A01*A12 - A02*A11))
        if abs(det) < 1e-30:
            continue
        inv = 1.0 / det
        d0[pi] = ((A11*A22-A12*A12)*B0 + (A02*A12-A01*A22)*B1 + (A01*A12-A02*A11)*B2)*inv*a
        d1[pi] = ((A02*A12-A01*A22)*B0 + (A00*A22-A02*A02)*B1 + (A01*A02-A00*A12)*B2)*inv*b
        d2[pi] = ((A01*A12-A02*A11)*B0 + (A01*A02-A00*A12)*B1 + (A00*A11-A01*A01)*B2)*inv*c

    return d0, d1, d2


# ─────────────────────────────────────────────────────────────
#  Main detector class
# ─────────────────────────────────────────────────────────────

class ParticleDetector:
    """Detect and localise particles in 2-D or 3-D images.

    Methods
    -------
    detect(img)
        Full pipeline: threshold → filter → detect → sub-pixel.
        Returns ``(N, ndim)`` coordinate array.

    Examples
    --------
    >>> cfg = DetectionConfig(threshold=0.3, bead_radius=4)
    >>> det = ParticleDetector(cfg)
    >>> coords = det.detect(image_3d)       # shape (N, 3)
    >>> coords = det.detect(image_2d)       # shape (N, 2)
    """

    def __init__(self, config: DetectionConfig):
        self.cfg = config

    # ── public API ──────────────────────────────────────────

    def detect(
        self,
        img: np.ndarray,
        roi_slices: Optional[Tuple[slice, ...]] = None,
    ) -> np.ndarray:
        """Run full detection pipeline. Returns (N, ndim) coords."""
        ndim = img.ndim

        # ROI crop
        offset = np.zeros(ndim, dtype=np.float64)
        if roi_slices is not None:
            offset = np.array([s.start or 0 for s in roi_slices], dtype=np.float64)
            img = img[roi_slices].copy()

        # Deconvolution
        if self.cfg.psf is not None:
            from skimage.restoration import richardson_lucy
            img = richardson_lucy(
                img.astype(np.float64), self.cfg.psf,
                num_iter=self.cfg.deconv_iters, clip=False,
            )

        # Invert for dark particles
        if self.cfg.color == "black":
            img = img.max() - img

        # Normalise to [0, 1]
        img = img.astype(np.float64)
        vmax = img.max()
        if vmax > 0:
            img_n = img / vmax
        else:
            return np.empty((0, ndim))

        # Dispatch
        if self.cfg.method == DetectionMethod.TRACTRAC:
            coords = self._detect_tractrac(img_n)
        else:
            coords = self._detect_tpt(img_n, img)

        # Offset back to full-image coords & clip
        if coords.size:
            coords += offset
        return coords

    # ── TracTrac method ─────────────────────────────────────

    def _detect_tractrac(self, img_n: np.ndarray) -> np.ndarray:
        """LoG → local-maximum → sub-pixel polynomial fit."""
        ndim = img_n.ndim
        bw = self._size_filtered_mask(img_n)
        img_m = img_n * bw  # masked image

        if self.cfg.bead_radius > 0:
            return self._log_detect(img_m, img_n, ndim)
        else:
            return self._centroid_detect(img_n, ndim)

    def _log_detect(self, img_m, img_n, ndim):
        sigma = self.cfg.bead_radius

        # Laplacian of Gaussian.  MATLAB's fspecial('log') is truncated to
        # ceil(sigma)*2+1 taps; scipy's gaussian_laplace truncates at 4σ
        # (25 taps for σ=3), which is a far coarser filter and both loses
        # particles and degrades the sub-pixel fit.
        log_img = ndimage.correlate(img_m, -_fspecial_log(sigma, ndim),
                                    mode="nearest")

        # Local-maximum filter.  MATLAB uses strel('square', 2σ+1) in 2-D — a
        # (2σ+1)-wide box, i.e. 7×7 for σ=3, roughly the bead spacing.  A box
        # twice that wide suppresses every second particle in a dense field.
        fp_size = int(2 * sigma) + 1
        rng = np.random.default_rng(42)
        noise = rng.random(log_img.shape) * 1e-5
        dilated = ndimage.maximum_filter(log_img + noise, size=fp_size,
                                         mode="nearest")
        # MATLAB thresholds the *masked* image (`im = im.*BW3` above), so the
        # size gate applies to the peaks too.
        peaks = ((log_img + noise) == dilated) & (img_m > self.cfg.threshold)

        coords_int = np.asarray(np.nonzero(peaks), dtype=np.int64).T  # (N, ndim)
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Trim border: MATLAB keeps `y >= nb+1 & y < h-nb` in 1-based indexing.
        nb_ = int((sigma + 2) // 2)
        mask = np.ones(len(coords_int), dtype=np.bool_)
        for d in range(ndim):
            mask &= ((coords_int[:, d] >= nb_)
                     & (coords_int[:, d] <= img_n.shape[d] - nb_ - 2))
        coords_int = coords_int[mask]
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Sub-pixel via parabola on log(LoG)
        log_safe = np.log(np.clip(log_img - log_img.min() + 1e-8, 1e-12, None))

        if ndim == 2:
            dx, dy = _subpixel_poly_2d(log_safe, coords_int[:, 0], coords_int[:, 1])
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
        else:
            dx, dy, dz = _subpixel_poly_3d(
                log_safe, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
            )
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5) & (np.abs(dz) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
            out[:, 2] += dz[valid]
        return out

    # ── TPT method ──────────────────────────────────────────

    def _detect_tpt(self, img_n: np.ndarray, img_raw: np.ndarray) -> np.ndarray:
        """Blob centroid → radial-symmetry sub-voxel refinement.

        Mirrors ``locateParticles.m`` → ``radialcenter3dvec.m``: the blob stage
        produces an **integer voxel** seed (MATLAB rounds the *binary* blob
        centroid), the patch is cut around that integer, and the radial-symmetry
        offset — which is measured from the patch centre — is added to it.

        Adding the offset to an *unrounded* sub-voxel centroid instead (as an
        earlier revision did) double-counts the sub-voxel part.
        """
        ndim = img_n.ndim
        seed = self._blob_seed(img_n, ndim)
        if len(seed) == 0 or ndim != 3:
            # 2-D upstream never reaches radialcenter: every shipped
            # `fun_SerialTrack_2D_*` calls `f_detect_particles` (the LoG path)
            # and leaves locateBeads/radial2center commented out.  Fall back to
            # the intensity-weighted blob centroid, which is strictly better
            # than the integer seed.
            return self._centroid_detect(img_n, ndim) if ndim != 3 else seed

        ws = np.array(self.cfg.win_size[:3], dtype=np.int64)
        half = ws // 2
        ci = seed.astype(np.int64)

        # radialcenter3dvec works on the RAW (un-normalised) image plus a
        # dither, and pads with near-zero noise (padNoise.m) — NOT by
        # reflection, which would mirror real bead intensity into the border and
        # bias particles within half a window of the edge.
        img_f = img_raw.astype(np.float64)
        img_f = img_f + self.cfg.rand_noise * np.random.default_rng(0).random(img_f.shape)
        img_p = np.pad(img_f, [(h, h) for h in half], mode="constant",
                       constant_values=0.0)

        patches = np.empty((len(ci), ws[0], ws[1], ws[2]), dtype=np.float64)
        for i, c in enumerate(ci):
            cp = c + half  # padded coords
            patches[i] = img_p[
                cp[0]-half[0]:cp[0]+half[0]+1,
                cp[1]-half[1]:cp[1]+half[1]+1,
                cp[2]-half[2]:cp[2]+half[2]+1,
            ]

        dccd = np.array(self.cfg.dccd[:3], dtype=np.float64)
        abc = np.array(self.cfg.abc[:3], dtype=np.float64)

        off = np.column_stack(_radial_symmetry_3d(patches, half, dccd, abc))

        out = ci.astype(np.float64)
        # A solve that lands outside the window is not a refinement — drop the
        # offset for those (MATLAB drops the particle on NaN; keeping the seed
        # preserves the detection, which the linker can still use).
        ok = np.all(np.isfinite(off), axis=1) & np.all(np.abs(off) < half, axis=1)
        out[ok] += off[ok]
        return out

    # ── shared helpers ──────────────────────────────────────

    def _label_blobs(self, img_n: np.ndarray):
        """Threshold → connected components → (labels, n, sizes)."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return labeled, 0, np.empty(0)
        sizes = ndimage.sum_labels(bw, labeled, np.arange(1, n + 1))
        return labeled, n, sizes

    def _size_filtered_mask(self, img_n: np.ndarray) -> np.ndarray:
        """Threshold → label → keep blobs of at least ``min_size`` voxels.

        ``f_detect_particles.m`` computes ``BW3 = bwareaopen(BW,minSize) -
        bwareaopen(BW,maxSize)`` and then unconditionally discards it::

            if sum(mean(BW3(:))) < 1, BW3 = BW1; end

        ``mean`` of a 0/1 array is the *fraction* of set voxels, so that test is
        true unless the whole image is foreground — the LoG path therefore
        applies the **minimum** size filter only.  Reinstating the maximum here
        would silently delete merged bead pairs that upstream keeps.
        """
        labeled, n, sizes = self._label_blobs(img_n)
        if n == 0:
            return labeled.astype(bool)
        return np.concatenate([[False], sizes >= self.cfg.min_size])[labeled]

    def _blob_seed(self, img_n: np.ndarray, ndim: int) -> np.ndarray:
        """Integer-voxel blob seeds, exactly as ``locateParticles.m``.

        The centroid is taken over the **binary** mask (MATLAB's
        ``regionprops(CC,'Centroid')`` is unweighted) and rounded, and the size
        gate uses MATLAB's *strict* ``> minSize & < maxSize``.
        """
        labeled, n, sizes = self._label_blobs(img_n)
        if n == 0:
            return np.empty((0, ndim))
        keep = (sizes > self.cfg.min_size) & (sizes < self.cfg.max_size)
        if not keep.any():
            return np.empty((0, ndim))
        idx = np.flatnonzero(keep) + 1
        cen = np.asarray(ndimage.center_of_mass(labeled > 0, labeled, idx),
                         dtype=np.float64).reshape(-1, ndim)
        return np.round(cen)

    def _centroid_detect(self, img_n: np.ndarray, ndim: int) -> np.ndarray:
        """Intensity-weighted connected-component centroids, filtered by size."""
        labeled, n, sizes = self._label_blobs(img_n)
        if n == 0:
            return np.empty((0, ndim))
        keep = (sizes >= self.cfg.min_size) & (sizes <= self.cfg.max_size)
        if not keep.any():
            return np.empty((0, ndim))
        idx = np.flatnonzero(keep) + 1
        return np.asarray(ndimage.center_of_mass(img_n, labeled, idx),
                          dtype=np.float64).reshape(-1, ndim)

    @staticmethod
    def clip_to_bounds(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Remove coords outside [0, shape) for each dimension."""
        if coords.size == 0:
            return coords
        mask = np.ones(len(coords), dtype=bool)
        for d in range(coords.shape[1]):
            mask &= (coords[:, d] >= 0) & (coords[:, d] < shape[d])
        return coords[mask]

# ============================================================================
#  Vendored from nd2studios/backend/analysis/granule_types.py (make_point_rows, NOISE_LABEL)
# ============================================================================
# Sentinel granule id for an unclustered / noise point.
NOISE_LABEL = -1


def make_point_rows(points_zyx: np.ndarray,
                    m: int, t: int,
                    voxel_size_um: Tuple[float, float, float],
                    labels: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Build DATA rows (P0 §3 schema) from an ``(N, 3)`` ``(z, y, x)`` voxel cloud.

    ``voxel_size_um`` is ``(dz, dy, dx)``. ``labels`` (optional) fills ``granule_id``;
    ``None`` leaves it ``None`` (pre-clustering).
    """
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    rows: List[Dict[str, Any]] = []
    for i in range(pts.shape[0]):
        z, y, x = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
        gid = None
        if labels is not None:
            gid = int(labels[i])
        rows.append({
            "bead_id": int(i),
            "m_position": int(m),
            "frame": int(t),
            "centroid_z_px": z, "centroid_y_px": y, "centroid_x_px": x,
            "centroid_z_um": z * dz, "centroid_y_um": y * dy, "centroid_x_um": x * dx,
            "granule_id": gid,
        })
    return rows


# ============================================================================
#  Vendored from nd2studios/backend/analysis/bead_detect.py (detect_beads + helpers)
# ============================================================================

# ── parameter defaults (mirrored by ``param_specs_for`` when wired in P6) ─────
_DEFAULT_DETECT_MODE = "log"        # "log" | "components"
_DEFAULT_MIN_DISTANCE_PX = 5        # peak separation / NMS radius (voxels)
_DEFAULT_THRESHOLD = 0.0            # normalized [0, 1]; 0 -> auto (Otsu)
_DEFAULT_MIN_INTENSITY = 0.0        # absolute raw-intensity floor at the peak
_DEFAULT_SUBPIXEL = True
_DEFAULT_MIN_SIZE = 1               # min blob volume/area [voxels]
_DEFAULT_MAX_SIZE = 2 ** 31 - 1     # effectively unbounded


def _require_backends() -> tuple:
    """Import the SerialTrack detector, gating its optional deps.

    Returns ``(DetectionConfig, DetectionMethod, ParticleDetector)``. Raises a
    friendly :class:`ImportError` naming the missing package.
    """
    for mod in ("numpy", "scipy", "numba"):
        if importlib.util.find_spec(mod) is None:
            raise ImportError(
                f"Bead detection requires the optional dependency '{mod}'. "
                f"Install it with `pip install {mod}`."
            )
    # Vendored in-file (see module header); no nd2studios import needed.
    return DetectionConfig, DetectionMethod, ParticleDetector


def _otsu_threshold_norm(vol: np.ndarray) -> float:
    """Otsu threshold expressed in the detector's normalized ``[0, 1]`` space.

    The detector compares ``img / img.max()`` against ``cfg.threshold``, so we
    compute Otsu on the max-normalized histogram and return the threshold in the
    same space (numpy-only, no scikit-image import).
    """
    a = np.asarray(vol, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    vmax = float(a.max())
    if vmax <= 0.0:
        return 0.0
    an = a / vmax
    hist, edges = np.histogram(an, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) * 0.5
    wb = np.cumsum(hist)
    wf = total - wb
    cum = np.cumsum(hist * centers)
    mb = np.divide(cum, wb, out=np.zeros_like(cum), where=wb > 0)
    mf = np.divide(cum[-1] - cum, wf, out=np.zeros_like(cum), where=wf > 0)
    between = wb * wf * (mb - mf) ** 2
    idx = int(np.argmax(between))
    return float(centers[idx])


def _sample_intensity(vol_zhw: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    """Raw intensity at each (rounded, clipped) ``(z, y, x)`` voxel."""
    if coords_zyx.size == 0:
        return np.zeros((0,), dtype=float)
    z, h, w = vol_zhw.shape
    zi = np.clip(np.rint(coords_zyx[:, 0]).astype(np.int64), 0, z - 1)
    yi = np.clip(np.rint(coords_zyx[:, 1]).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(coords_zyx[:, 2]).astype(np.int64), 0, w - 1)
    return np.asarray(vol_zhw[zi, yi, xi], dtype=float)


@nb.njit(cache=True)
def _nms_kernel(coords: np.ndarray, order: np.ndarray, md2: float) -> np.ndarray:
    """Greedy-NMS hot loop: walk ``order`` (point indices, brightest first) and keep
    a point iff it is >= ``sqrt(md2)`` from every already-kept point. Returns a
    boolean keep-mask indexed by original point index.

    The accept/reject is sequential (a point's fate depends on the running kept-set)
    so it does not vectorize; the ndim-agnostic distance is an explicit component
    loop so numba emits a tight scalar kernel with no temporaries (~500x over the
    pure-Python original — see ``scripts/_bench_nms_numba.py``).
    """
    n = coords.shape[0]
    ndim = coords.shape[1]
    keep = np.zeros(n, dtype=np.bool_)
    kept_idx = np.empty(n, dtype=np.int64)
    n_kept = 0
    for oi in range(order.shape[0]):
        idx = order[oi]
        ok = True
        for kk in range(n_kept):
            kp = kept_idx[kk]
            d2 = 0.0
            for c in range(ndim):
                diff = coords[idx, c] - coords[kp, c]
                d2 += diff * diff
            if d2 < md2:
                ok = False
                break
        if ok:
            keep[idx] = True
            kept_idx[n_kept] = idx
            n_kept += 1
    return keep


def _suppress_close(coords_zyx: np.ndarray, intensities: np.ndarray,
                    min_distance: float) -> np.ndarray:
    """Greedy non-max suppression: keep the brightest point in each
    ``min_distance`` (voxel Euclidean) neighborhood.

    Returns the row indices (in original order) to keep. The O(n²) greedy walk runs
    in the numba kernel :func:`_nms_kernel`; the intensity sort (already fast C, and
    its tie-breaking fixes which of two equal-intensity points survives) stays here.
    """
    n = coords_zyx.shape[0]
    if min_distance <= 0.0 or n <= 1:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(intensities)[::-1].astype(np.int64)
    coords = np.ascontiguousarray(coords_zyx, dtype=np.float64)
    keep = _nms_kernel(coords, order, float(min_distance) ** 2)
    return np.nonzero(keep)[0].astype(np.int64)   # np.nonzero is already ascending


def _detector_coords(detector: Any, vol_zhw: np.ndarray, is_3d: bool) -> np.ndarray:
    """Run the detector and return centroids in ``(z, y, x)`` voxel order.

    The detector's native axis order is ``(x, y, z)`` (2-D: ``(x, y)``), so we feed
    it the volume transposed into that order and reverse the output columns. THIS
    is the single point where the ``(x,y,z) -> (z,y,x)`` flip happens (P0 §5).
    """
    if is_3d:
        # (Z, H, W) -> (W, H, Z) == (x, y, z) for the detector.
        img_xyz = np.ascontiguousarray(np.transpose(vol_zhw, (2, 1, 0)))
        coords_xyz = np.asarray(detector.detect(img_xyz), dtype=float)
        if coords_xyz.size == 0:
            return np.zeros((0, 3), dtype=float)
        return coords_xyz[:, ::-1]              # (x, y, z) -> (z, y, x)
    # 2-D fallback: (H, W) -> (W, H) == (x, y); detect -> (x, y) -> (y, x); z = 0.
    img2d = vol_zhw[0] if vol_zhw.ndim == 3 else vol_zhw
    img_xy = np.ascontiguousarray(np.transpose(img2d, (1, 0)))
    coords_xy = np.asarray(detector.detect(img_xy), dtype=float)
    if coords_xy.size == 0:
        return np.zeros((0, 3), dtype=float)
    yx = coords_xy[:, ::-1]                      # (x, y) -> (y, x)
    zeros = np.zeros((yx.shape[0], 1), dtype=float)
    return np.hstack([zeros, yx])               # (0, y, x)


def detect_beads(volume_zhw, voxel_size_um, params) -> tuple[np.ndarray, list[dict]]:
    """Detect bead centroids in one raw ``(Z, H, W)`` volume.

    Parameters
    ----------
    volume_zhw : np.ndarray
        Raw single-channel volume, ``(Z, H, W)`` (float or integer). ``Z == 1``
        (or a plain ``(H, W)`` array) triggers the 2-D fallback: every
        ``centroid_z_px`` is ``0.0`` and the cloud keeps its ``(N, 3)`` shape with a
        zero z-column, so the downstream chain stays uniform (clustering degrades to
        2-D).
    voxel_size_um : tuple[float, float, float]
        Physical voxel spacing ``(dz, dy, dx)`` µm (``viz3d.prep.Spacing`` order).
    params : dict
        Plain dict read with ``.get`` defaults:

        * ``detect_mode`` (``"log"`` | ``"components"``) — LoG local-maxima vs.
          connected-component centroids.
        * ``min_distance_px`` (int) — minimum peak separation; also LoG scale.
        * ``threshold`` (float) — normalized ``[0, 1]``; ``0`` -> auto (Otsu).
        * ``min_intensity`` (float) — drop peaks below this raw intensity.
        * ``subpixel`` (bool, default ``True``) — keep sub-voxel refinement; when
          ``False`` centroids are rounded to integer voxels.
        * ``all_multipoints`` (bool) — consumed by the P6 page handler, not here.
        * ``m_position`` / ``frame`` (int) — the current ``(m, t)`` written onto the
          DATA rows (default ``0``).

    Returns
    -------
    (np.ndarray, list[dict])
        ``points_zyx`` — ``(N, 3)`` float array of centroids in ``(z, y, x)`` voxel
        order (the fast path for P2/P3), and ``rows`` — the ``List[Dict]`` DATA rows
        built by :func:`granule_types.make_point_rows` (P0 §3 schema, ``granule_id``
        left ``None`` pre-clustering).
    """
    DetectionConfig, DetectionMethod, ParticleDetector = _require_backends()

    vol = np.asarray(volume_zhw)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(
            f"detect_beads expects a (Z, H, W) or (H, W) array, got shape {vol.shape}"
        )
    is_3d = vol.shape[0] > 1

    dz = float(voxel_size_um[0])
    dy = float(voxel_size_um[1])
    dx = float(voxel_size_um[2])

    # ── read params ──────────────────────────────────────────────────────────
    detect_mode = str(params.get("detect_mode", _DEFAULT_DETECT_MODE)).lower()
    min_distance_px = float(params.get("min_distance_px", _DEFAULT_MIN_DISTANCE_PX))
    threshold = float(params.get("threshold", _DEFAULT_THRESHOLD))
    min_intensity = float(params.get("min_intensity", _DEFAULT_MIN_INTENSITY))
    subpixel = bool(params.get("subpixel", _DEFAULT_SUBPIXEL))
    min_size = int(params.get("min_size", _DEFAULT_MIN_SIZE))
    max_size = int(params.get("max_size", _DEFAULT_MAX_SIZE))
    m = int(params.get("m_position", params.get("m", 0)))
    t = int(params.get("frame", params.get("t", 0)))

    # ── map params -> DetectionConfig ─────────────────────────────────────────
    if detect_mode == "components":
        # Connected-component centroids. Sub-voxel via radial symmetry (TPT, 3-D
        # only) when requested; otherwise a pure centroid (bead_radius == 0).
        method = DetectionMethod.TPT if subpixel else DetectionMethod.TRACTRAC
        bead_radius = 0.0
    else:                                       # "log" (default)
        method = DetectionMethod.TRACTRAC
        # LoG scale: sigma ~ half the min separation so the max-filter footprint
        # (~4*sigma+1) keeps detections at least ~min_distance_px apart.
        bead_radius = max(1.0, min_distance_px / 2.0)

    thr = threshold if threshold > 0.0 else _otsu_threshold_norm(vol)

    # Anisotropy factors in the detector's (x, y, z) order; dccd stays 1 so the
    # sub-voxel shifts come out in voxel units (the cloud is voxel coords).
    vals = [v for v in (dx, dy, dz) if v > 0.0]
    aniso_min = min(vals) if vals else 1.0
    abc = (dx / aniso_min, dy / aniso_min, dz / aniso_min)

    cfg = DetectionConfig(
        method=method,
        threshold=float(thr),
        bead_radius=float(bead_radius),
        min_size=min_size,
        max_size=max_size,
        color="white",
        win_size=(5, 5, 5),
        dccd=(1.0, 1.0, 1.0),
        abc=abc,
    )
    detector = ParticleDetector(cfg)

    # ── detect + flip to (z, y, x) ────────────────────────────────────────────
    coords_zyx = _detector_coords(detector, vol, is_3d)
    if not is_3d:
        coords_zyx[:, 0] = 0.0                  # enforce centroid_z_px == 0

    # ── post-filters: min-intensity floor, then min-distance NMS ──────────────
    if coords_zyx.shape[0]:
        intens = _sample_intensity(vol, coords_zyx)
        if min_intensity > 0.0:
            keep = intens >= min_intensity
            coords_zyx = coords_zyx[keep]
            intens = intens[keep]
        if coords_zyx.shape[0] and min_distance_px > 0.0:
            keep_idx = _suppress_close(coords_zyx, intens, min_distance_px)
            coords_zyx = coords_zyx[keep_idx]

    if not subpixel and coords_zyx.shape[0]:
        coords_zyx = np.rint(coords_zyx)

    coords_zyx = np.ascontiguousarray(coords_zyx.reshape(-1, 3).astype(float))

    rows = make_point_rows(coords_zyx, m=m, t=t,
                           voxel_size_um=(dz, dy, dx), labels=None)
    return coords_zyx, rows
