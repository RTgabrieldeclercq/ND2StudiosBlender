"""bead_slabs — slab-projected 3-D bead finder (in-repo maths, 2026-10-07).

The integration contract is ``bead_slabs.md`` next to this file; read it before calling.
Call **only** :func:`find_beads`. Everything else is support code.

The algorithm, in the order the data flows (the design brief is quoted in
``CodeLog/Updates/worklog/2026-10-07_bead-finder-node.md``):

0. **Despike.** A single-voxel outlier (hot pixel, PMT event) is clamped to its
   neighbours; a bead is extended and is left alone.
1. **Slabs.** Cut the Z axis into sub-stacks of ``S`` planes, equally spaced with ``G``
   planes of overlap, and flatten each by max / mean projection (``min`` projection for
   dark beads is the max projection of the inverted volume). ``S`` can be chosen
   automatically from the bead density: a first pass on the whole-volume projection
   counts the beads, and ``S`` is then set so that at most ~10 % of beads have another
   bead inside their own footprint in a slab projection — thinner slabs for a denser
   field, never thicker than half the stack.
2. **2-D detection per slab.** A scale-normalised Laplacian-of-Gaussian at the bead's
   expected apparent lateral size, local maxima over a bead-sized footprint, kept when the
   response exceeds ``min_snr`` robust noise sigmas of that slab's response.
3. **Axial localisation per candidate.** The raw column under the (y, x) hit is reduced
   to a background-subtracted, matched-filter z-profile and EVERY prominent peak of it
   near the slab becomes a bead to fit — two beads stacked in z under one (y, x) are two
   peaks. Each is fitted with a **split Gaussian** (one sigma below the peak, another
   above — the confocal axial profile is skewed by the PSF and the index mismatch) over a
   window bounded by the valleys to its neighbours, or jointly with a neighbour closer
   than four sigmas.
4. **Lateral sub-pixel + size.** The planes around each z peak are averaged (that peak's
   plane alone when the column holds several) and a 2-D Gaussian (axis-aligned, 6
   parameters) fitted for the sub-pixel (y, x) and the apparent sigmas; pixels nearer to
   ANOTHER candidate of the same slab are left out of the fit, so two touching beads do
   not inflate each other's width. A compact blob too wide or too long for one bead is
   offered the **two-bead model** (two round Gaussians, one shared sigma), accepted only
   when both halves are bead-sized and bead-bright, at least three quarters of a diameter
   apart (beads are rigid spheres), and together explain the pixels markedly better than
   ONE object — one bead, or one elongated Gaussian at any angle, which is what a fibre
   segment is.
5. **Filters** — only things that look like a bead survive: a Hessian ridge test on the
   projection (a fibre or a scan line has no curvature along itself), lateral size inside
   a tolerance band around the expected apparent size, fitted aspect ratio below a bound,
   axial size inside a (wider) band, a peak that is not on the first/last plane.
6. **Merge** the same bead found in two overlapping slabs: greedy, brightest first, within
   half a bead diameter (two beads cannot be closer than one), per axis in voxels.

Everything here is voxel units; the caller converts microns to voxels and back.
numpy + scipy only; no numba — the hot path is scipy's LoG and per-bead
Levenberg–Marquardt fits with analytic Jacobians (~1 ms each).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["find_beads", "DEFAULTS", "auto_slab_thickness", "expected_sigmas_px"]

#: The parameter defaults :func:`find_beads` reads with ``.get`` — keep in step with the
#: node's SocketSpec defaults (``nodegraph/catalog/detect/beads.py``).
DEFAULTS: Dict[str, Any] = {
    "sigma_xy_px": 1.5,       # expected apparent lateral Gaussian sigma of ONE bead, px
    "sigma_z_px": 1.5,        # expected apparent axial sigma, planes
    "slab_px": 0,             # slab thickness in planes; 0 → automatic from density
    "overlap_px": -1,         # slab overlap in planes; <0 → automatic (~2 sigma_z)
    "projection": "max",      # "max" | "mean" | "min"
    "min_snr": 5.0,           # LoG peak over robust noise sigma of the slab response
    "size_tolerance": 0.5,    # accept sigma_xy within ±tol of expected (axial: ±2·tol)
    "max_aspect": 1.5,        # sigma_major / sigma_minor bound (rods, stripes, fibres)
    "target_overlap": 0.10,   # auto-slab: tolerated fraction of beads with a projected neighbour
    "deblend": True,          # split a too-wide compact blob into two beads when the data say so
    "multi_peak": True,       # one (y, x) hit may hold several beads stacked in z
    "diameter_px": 0.0,       # the bead diameter in pixels (0 = unknown): beads are rigid spheres,
    "diameter_zpx": 0.0,      # so two centres are never closer than this — and anything closer
                              # than half of it is one bead seen twice; in planes along z
}

_REJECT_KEYS = ("edge_z", "other_slab", "fit_failed", "dim", "size_xy", "aspect", "size_z",
                "duplicate")

#: The Hessian ridge test's bound on ``sigma_major / sigma_minor`` measured from the smoothed
#: projection's principal curvatures. Fixed, not the user's ``max_aspect``: that one judges
#: the fitted bead; this one only has to tell a BLOB from a RIDGE. Measured on the bead
#: phantom (2026-10-07): 99 % of true beads read below 2.25 even at a density where a fifth
#: of them touch another (a touching neighbour flattens the curvature along the pair axis),
#: while a fibre or a scan line reads 5 or infinite. 2.5 loses 0.2 % of beads.
_RIDGE_ASPECT = 2.5

#: A peak in the z-profile counts as a bead when its prominence is at least this fraction of
#: its own height and it is at least ``_PEAK_REL_HEIGHT`` of the window's tallest peak (both
#: on top of a 3-sigma noise floor). The dimmer of two beads stacked in z sits on the brighter
#: one's axial tail, so both are relative to what the profile shows, not absolute.
_PEAK_REL_PROM = 0.10
_PEAK_REL_HEIGHT = 0.15
#: When a column holds several z peaks, fit each bead's (y, x) on its OWN peak plane only
#: instead of the usual ±sigma_z core: the neighbour's tail reaches the adjacent planes.
_STACKED_SINGLE_PLANE = True


# ── expected sizes ──────────────────────────────────────────────────────────────

def expected_sigmas_px(diameter_um: float, *, pixel_size_um: float, z_step_um: float,
                       psf_sigma_xy_um: float, psf_sigma_z_um: float) -> Tuple[float, float]:
    """The apparent Gaussian sigmas, in voxels, of a solid fluorescent sphere of
    ``diameter_um`` imaged through a PSF with the given Gaussian sigmas.

    The finder fits Gaussians, so "size" means the sigma a **least-squares Gaussian fit**
    returns on a solid sphere — not the sphere's FWHM. Computed numerically (2026-10-07):
    the projected sphere ``2·sqrt(R² − rho²)`` fits with ``sigma = 0.272 d`` laterally; the
    axial profile a laterally-integrating matched filter sees lies between the cross-section
    area (``0.258 d``) and the centre chord (``0.297 d``), taken as ``0.28 d``. (FWHM matching
    would say ``0.368 d`` and second moments ``0.224 d`` — both are the wrong estimator for
    a fitted width.) The imaged bead is the body convolved with the PSF, so the sigmas add
    in quadrature. The floor of 0.5 voxel keeps a sub-resolution bead on a coarse grid from
    asking for a sub-voxel filter."""
    d = max(0.0, float(diameter_um))
    sxy_um = math.sqrt((0.272 * d) ** 2 + max(0.0, float(psf_sigma_xy_um)) ** 2)
    sz_um = math.sqrt((0.28 * d) ** 2 + max(0.0, float(psf_sigma_z_um)) ** 2)
    sxy = sxy_um / max(1e-9, float(pixel_size_um))
    sz = sz_um / max(1e-9, float(z_step_um))
    return max(0.5, sxy), max(0.5, sz)


def auto_slab_thickness(n_beads: int, shape_zyx: Sequence[int], sigma_xy_px: float,
                        sigma_z_px: float, *, target_overlap: float = 0.10) -> int:
    """Slab thickness ``S`` (planes) from an estimated bead count.

    Two beads closer than one lateral FWHM merge in a projection, so each bead owns an
    exclusion disc of area ``a = pi · (2.355 sigma_xy)²``. With ``rho`` beads per voxel, a
    slab of ``S`` planes puts on average ``lambda = rho · S · a`` other beads inside that
    disc; ``S`` is set so ``lambda <= target_overlap``, clamped between about two axial
    sigmas (thinner than a bead buys nothing) and HALF the stack: a projection of the
    whole stack is the one case where a fibre, an aggregate or a hot voxel anywhere in Z
    hides every bead beneath it, and two overlapping slabs already give each bead one
    projection free of most of them."""
    nz, ny, nx = (int(v) for v in shape_zyx)
    s_min = max(1, int(round(2.0 * sigma_z_px)))
    s_max = max(s_min, int(math.ceil(nz / 2.0)))
    if n_beads <= 0 or nz <= s_min:
        return int(min(nz, s_max))
    rho = float(n_beads) / float(max(1, nz * ny * nx))
    area = math.pi * (2.355 * sigma_xy_px) ** 2
    s = float(target_overlap) / max(1e-12, rho * area)
    return int(np.clip(int(round(s)), s_min, s_max))


def _auto_overlap(slab: int, sigma_z_px: float) -> int:
    return int(np.clip(int(round(2.0 * sigma_z_px)), 0, max(0, slab - 1)))


def _slab_ranges(nz: int, slab: int, overlap: int) -> List[Tuple[int, int]]:
    """``[(a, b), …]`` half-open plane ranges covering ``[0, nz)``; the last one is pulled
    back so it ends exactly at ``nz``."""
    slab = int(np.clip(slab, 1, nz))
    step = max(1, slab - int(np.clip(overlap, 0, slab - 1)))
    out: List[Tuple[int, int]] = []
    a = 0
    while True:
        b = min(nz, a + slab)
        if out and b == out[-1][1]:
            break
        out.append((a, b))
        if b >= nz:
            break
        a += step
        if a + slab > nz:
            a = nz - slab
    return out


# ── step 0: hot voxels ──────────────────────────────────────────────────────────

def _despike(vol: np.ndarray, sigma_xy: float) -> Tuple[np.ndarray, int]:
    """Clamp single-voxel spikes (hot pixels, PMT events) to the median of their 26
    neighbours, and say how many were clamped.

    A bead is EXTENDED: at the expected lateral sigma its peak voxel's in-plane neighbours
    carry ``exp(-1 / (2 sigma²))`` of its excess over background (52 % at 0.87 px), the
    diagonals less, the next planes depend on the axial sigma — the MEDIAN of all 26 is
    roughly a quarter of the excess. A hot voxel's neighbours carry ~nothing, and that
    stays true when it happens to sit beside a bright bead, because the bead occupies a
    minority of the 26 (the brightest neighbour would not tell them apart — that is why
    the test is a median, not a max). So a voxel well above the noise whose median
    neighbour holds under 15 % of its own excess is a spike, not an object. Why this
    matters in a projection: a spike one or two pixels from a bead produces a LoG response
    several times the bead's, and the local-maximum footprint then swallows the bead's own
    peak — the bead becomes a candidate that is (correctly) rejected for being a quarter-
    pixel wide, and is lost. Skipped when the expected sigma is under 0.75 px: there a
    genuine bead is itself barely more than one voxel wide and the test cannot tell them
    apart."""
    if sigma_xy < 0.75:
        return vol, 0
    from scipy.ndimage import median_filter
    fp = np.ones((3, 3, 3), dtype=bool)
    fp[1, 1, 1] = False
    nbr = median_filter(vol, footprint=fp, mode="nearest")
    bg = float(np.median(vol))
    noise = 1.4826 * float(np.median(np.abs(vol - bg))) or 1.0
    excess = vol - bg
    spike = (excess > 8.0 * noise) & ((nbr - bg) < 0.15 * excess)
    n = int(spike.sum())
    if n == 0:
        return vol, 0
    out = vol.copy()
    out[spike] = nbr[spike]
    return out, n


# ── step 2: 2-D detection on one projection ─────────────────────────────────────

def _project(vol: np.ndarray, a: int, b: int, how: str) -> np.ndarray:
    sub = vol[a:b]
    if how == "mean":
        return sub.mean(axis=0)
    return sub.max(axis=0)


def _detect_2d(proj: np.ndarray, sigma_xy: float, min_snr: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale-normalised LoG peaks above ``min_snr`` robust sigmas →
    ``(ys, xs, snr, ridge_aspect)``.

    ``ridge_aspect`` is the Hessian anisotropy of the Gaussian-smoothed projection at each
    peak: ``sqrt(|lambda_max| / |lambda_min|)`` of the two principal curvatures, which for
    an elliptical Gaussian equals ``sigma_major / sigma_minor``, and is infinite on a RIDGE
    (a fibre, a scan line), where the curvature along the structure is ~0 or positive. It
    is measured on the whole projection, before any neighbour masking, so a fibre cut into
    bead-sized Voronoi cells by its own chain of LoG peaks still reads as a fibre."""
    from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter
    img = proj.astype(float)
    resp = -(sigma_xy ** 2) * gaussian_laplace(img, sigma_xy, mode="reflect")
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= 0:
        noise = float(resp.std()) or 1.0
    fp = max(3, 2 * int(round(1.2 * sigma_xy)) + 1)
    peaks = (resp == maximum_filter(resp, size=fp, mode="reflect")) & (resp > med + min_snr * noise)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return ys, xs, np.zeros(0), np.zeros(0)
    hyy = gaussian_filter(img, sigma_xy, order=(2, 0), mode="reflect")[ys, xs]
    hxx = gaussian_filter(img, sigma_xy, order=(0, 2), mode="reflect")[ys, xs]
    hxy = gaussian_filter(img, sigma_xy, order=(1, 1), mode="reflect")[ys, xs]
    half_tr = 0.5 * (hyy + hxx)
    disc = np.sqrt((0.5 * (hyy - hxx)) ** 2 + hxy ** 2)
    lam_strong, lam_weak = half_tr - disc, half_tr + disc        # both < 0 on a blob
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.where(lam_weak < 0, np.sqrt(np.abs(lam_strong) / np.abs(lam_weak)), np.inf)
    return ys, xs, (resp[ys, xs] - med) / noise, aspect


# ── step 3/4: per-candidate refinement ──────────────────────────────────────────

def _z_profile(vol: np.ndarray, y: float, x: float, sigma_xy: float
               ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """Matched-filter z-profile under ``(y, x)``: per plane, the Gaussian-weighted mean of
    a ``±half`` crop minus the median of the crop's outer ring. Returns the profile, the
    crop bounds ``(y0, y1, x0, x1)`` and the weights' L2 norm — the factor that turns the
    raw per-voxel noise into this profile's noise."""
    nz, ny, nx = vol.shape
    half = int(math.ceil(3.0 * sigma_xy)) + 1
    yi, xi = int(round(y)), int(round(x))
    y0, y1 = max(0, yi - half), min(ny, yi + half + 1)
    x0, x1 = max(0, xi - half), min(nx, xi + half + 1)
    crop = vol[:, y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    r2 = (yy - y) ** 2 + (xx - x) ** 2
    w = np.exp(-r2 / (2.0 * sigma_xy ** 2))
    ring = r2 > (2.5 * sigma_xy) ** 2
    if ring.sum() < 4:
        bg = crop.reshape(nz, -1).min(axis=1)
    else:
        bg = np.median(crop[:, ring], axis=1)
    w = w / max(1e-12, w.sum())
    prof = np.tensordot(crop - bg[:, None, None], w, axes=([1, 2], [0, 1]))
    return prof, (y0, y1, x0, x1), float(math.sqrt((w ** 2).sum()))


def _z_peaks(prof: np.ndarray, lo: int, hi: int, noise_prof: float) -> List[int]:
    """Every bead-like peak of the profile inside the window ``[lo, hi)``: interior local
    maxima whose prominence over the deeper of the two flanking valleys is at least 3
    profile-noise sigmas and 15 % of their own height, at least a quarter as high as the
    window's maximum, and at least 2 planes apart. Two beads stacked in z under one (y, x)
    are two peaks here, where a single ``argmax`` saw only the brighter one."""
    seg = prof[lo:hi]
    if len(seg) < 3:
        return []
    from scipy.signal import find_peaks
    idx, props = find_peaks(seg, prominence=max(3.0 * noise_prof, 1e-9), distance=2)
    if len(idx) == 0:
        return []
    top = float(seg.max())
    keep = [int(i) + lo for i, pr in zip(idx, props["prominences"])
            if seg[i] >= _PEAK_REL_HEIGHT * top and pr >= _PEAK_REL_PROM * seg[i]]
    return keep


def _split_gauss(z: np.ndarray, amp: float, z0: float, s_lo: float, s_hi: float,
                 base: float) -> np.ndarray:
    s = np.where(z < z0, abs(s_lo), abs(s_hi))
    return base + amp * np.exp(-((z - z0) ** 2) / (2.0 * s ** 2))


def _split_gauss_jac(z: np.ndarray, amp: float, z0: float, s_lo: float, s_hi: float,
                     base: float) -> np.ndarray:
    lo = z < z0
    s = np.where(lo, abs(s_lo), abs(s_hi))
    d = z - z0
    g = np.exp(-(d ** 2) / (2.0 * s ** 2))
    ds = amp * g * d ** 2 / s ** 3
    return np.stack([g, amp * g * d / s ** 2,
                     np.where(lo, ds, 0.0) * (1.0 if s_lo >= 0 else -1.0),
                     np.where(lo, 0.0, ds) * (1.0 if s_hi >= 0 else -1.0),
                     np.ones_like(z)], axis=1)


def _gauss1d(z: np.ndarray, amp: float, z0: float, s: float, base: float) -> np.ndarray:
    return base + amp * np.exp(-((z - z0) ** 2) / (2.0 * s ** 2))


def _gauss1d_jac(z: np.ndarray, amp: float, z0: float, s: float, base: float) -> np.ndarray:
    d = z - z0
    g = np.exp(-(d ** 2) / (2.0 * s ** 2))
    return np.stack([g, amp * g * d / s ** 2, amp * g * d ** 2 / s ** 3, np.ones_like(z)],
                    axis=1)


def _two_gauss1d(z: np.ndarray, a1: float, z1: float, a2: float, z2: float, s: float,
                 base: float) -> np.ndarray:
    return (base + a1 * np.exp(-((z - z1) ** 2) / (2.0 * s ** 2))
            + a2 * np.exp(-((z - z2) ** 2) / (2.0 * s ** 2)))


def _curve_fit(fn, x, y, p0, jac=None, maxfev=300):
    """``scipy.optimize.curve_fit`` with LM and the covariance warning silenced (pcov is
    never used); ``None`` instead of raising when the solver gives up."""
    from scipy.optimize import curve_fit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fn, x, y, p0=p0, jac=jac, method="lm", maxfev=maxfev)
    except Exception:  # noqa: BLE001 — a failed fit is a result, not an error
        return None
    return [float(v) for v in popt]


def _fit_z(prof: np.ndarray, z_peak: int, sigma_z: float, lo_bound: int = 0,
           hi_bound: Optional[int] = None
           ) -> Tuple[float, float, float, float, bool, bool]:
    """Axial fit around ``z_peak`` → ``(z0, amp, sigma_lo, sigma_hi, fitted, split)``.

    The window is ``±ceil(3 sigma_z)`` planes, clipped to the volume and to
    ``[lo_bound, hi_bound)`` — the valleys toward neighbouring peaks, so a bead stacked
    above or below this one does not enter its fit. With 6+ planes the model is the split
    Gaussian (one sigma each side of the peak); with 4–5 a symmetric Gaussian (``split``
    False, the two sigmas equal); fewer falls back to a 3-point parabola (``fitted``
    False, sigmas NaN). A fit that does not converge, or that wanders more than 1.5 planes
    from the peak it started on, also falls back."""
    nz = len(prof)
    hi_bound = nz if hi_bound is None else hi_bound
    w = max(2, int(math.ceil(3.0 * sigma_z)))
    a, b = max(0, z_peak - w, lo_bound), min(nz, z_peak + w + 1, hi_bound)
    zz = np.arange(a, b, dtype=float)
    pp = prof[a:b]
    amp0 = float(prof[z_peak] - pp.min()) if len(pp) else 0.0

    def parabola() -> Tuple[float, float, float, float, bool, bool]:
        if 0 < z_peak < nz - 1:
            l, c, r = prof[z_peak - 1], prof[z_peak], prof[z_peak + 1]
            den = l - 2.0 * c + r
            dz = 0.5 * (l - r) / den if den < 0 else 0.0
            dz = float(np.clip(dz, -0.5, 0.5))
        else:
            dz = 0.0
        return float(z_peak + dz), amp0, float("nan"), float("nan"), False, False

    s_max = 6.0 * sigma_z + 2
    if amp0 <= 0 or len(zz) < 4:
        return parabola()
    if len(zz) >= 6:
        popt = _curve_fit(_split_gauss, zz, pp, [amp0, float(z_peak), sigma_z, sigma_z,
                                                 float(pp.min())], jac=_split_gauss_jac,
                          maxfev=200)
        if popt is not None:
            amp, z0, s_lo, s_hi, _base = popt
            s_lo, s_hi = abs(s_lo), abs(s_hi)
            if (np.isfinite(z0) and amp > 0 and abs(z0 - z_peak) <= 1.5
                    and 0.2 <= s_lo <= s_max and 0.2 <= s_hi <= s_max):
                return z0, amp, s_lo, s_hi, True, True
    popt = _curve_fit(_gauss1d, zz, pp, [amp0, float(z_peak), sigma_z, float(pp.min())],
                      jac=_gauss1d_jac, maxfev=200)
    if popt is not None:
        amp, z0, s, _base = popt
        s = abs(s)
        if np.isfinite(z0) and amp > 0 and abs(z0 - z_peak) <= 1.5 and 0.2 <= s <= s_max:
            return z0, amp, s, s, True, False
    return parabola()


def _fit_two_z(prof: np.ndarray, k1: int, k2: int, sigma_z: float
               ) -> Tuple[float, float, float, bool]:
    """Two beads stacked in z, too close for separate windows: fit the sum of two
    Gaussians with one shared sigma over the union window and return the component that
    started at ``k1`` → ``(z0, amp, sigma, fitted)``."""
    nz = len(prof)
    w = max(2, int(math.ceil(3.0 * sigma_z)))
    a, b = max(0, min(k1, k2) - w), min(nz, max(k1, k2) + w + 1)
    zz = np.arange(a, b, dtype=float)
    pp = prof[a:b]
    base0 = float(pp.min())
    p0 = [float(prof[k1] - base0), float(k1), float(prof[k2] - base0), float(k2), sigma_z, base0]
    popt = _curve_fit(_two_gauss1d, zz, pp, p0, maxfev=400)
    if popt is None:
        return float(k1), 0.0, float("nan"), False
    a1, z1, a2, z2, s, _base = popt
    s = abs(s)
    if not (np.isfinite(z1) and np.isfinite(z2) and a1 > 0 and a2 > 0
            and abs(z1 - k1) <= 1.5 and abs(z2 - k2) <= 1.5 and 0.2 <= s <= 6.0 * sigma_z + 2
            and abs(z1 - z2) >= 1.0):
        return float(k1), 0.0, float("nan"), False
    return z1, a1, s, True


def _gauss2d(coords: Tuple[np.ndarray, np.ndarray], amp: float, y0: float, x0: float,
             sy: float, sx: float, base: float) -> np.ndarray:
    yy, xx = coords
    return base + amp * np.exp(-((yy - y0) ** 2) / (2.0 * sy ** 2)
                               - ((xx - x0) ** 2) / (2.0 * sx ** 2))


def _gauss2d_jac(coords: Tuple[np.ndarray, np.ndarray], amp: float, y0: float, x0: float,
                 sy: float, sx: float, base: float) -> np.ndarray:
    yy, xx = coords
    dy, dx = yy - y0, xx - x0
    g = np.exp(-(dy ** 2) / (2.0 * sy ** 2) - (dx ** 2) / (2.0 * sx ** 2))
    ag = amp * g
    return np.stack([g, ag * dy / sy ** 2, ag * dx / sx ** 2,
                     ag * dy ** 2 / sy ** 3, ag * dx ** 2 / sx ** 3,
                     np.ones_like(yy)], axis=1)


def _rot_gauss2d(coords: Tuple[np.ndarray, np.ndarray], amp: float, y0: float, x0: float,
                 s_major: float, s_minor: float, theta: float, base: float) -> np.ndarray:
    """One elongated Gaussian at any angle — what a fibre segment, a scan line or a motion
    smear looks like inside a crop. The two-bead model has to beat THIS, not only the
    axis-aligned single Gaussian, before a wide blob is called two beads."""
    yy, xx = coords
    ct, st = math.cos(theta), math.sin(theta)
    u = (yy - y0) * ct + (xx - x0) * st
    v = -(yy - y0) * st + (xx - x0) * ct
    return base + amp * np.exp(-(u ** 2) / (2.0 * s_major ** 2) - (v ** 2) / (2.0 * s_minor ** 2))


def _rot_gauss2d_jac(coords: Tuple[np.ndarray, np.ndarray], amp: float, y0: float, x0: float,
                     s_major: float, s_minor: float, theta: float, base: float) -> np.ndarray:
    yy, xx = coords
    ct, st = math.cos(theta), math.sin(theta)
    dy, dx = yy - y0, xx - x0
    u = dy * ct + dx * st
    v = -dy * st + dx * ct
    g = np.exp(-(u ** 2) / (2.0 * s_major ** 2) - (v ** 2) / (2.0 * s_minor ** 2))
    ag = amp * g
    # d/dy0: u_y0 = -ct, v_y0 = st ; d/dx0: u_x0 = -st, v_x0 = -ct ; d/dtheta: u_t = v, v_t = -u
    return np.stack([g,
                     ag * (u * ct / s_major ** 2 - v * st / s_minor ** 2),
                     ag * (u * st / s_major ** 2 + v * ct / s_minor ** 2),
                     ag * u ** 2 / s_major ** 3,
                     ag * v ** 2 / s_minor ** 3,
                     ag * (-u * v / s_major ** 2 + u * v / s_minor ** 2),
                     np.ones_like(yy)], axis=1)


def _fit_rot_xy(ys: np.ndarray, xs: np.ndarray, vs: np.ndarray, y_c: float, x_c: float,
                amp0: float, sy: float, sx: float, base0: float, theta0: float) -> float:
    """SSE of the best rotated elongated Gaussian on these pixels (``inf`` when the solver
    gives up), seeded at the blob's principal-axis angle and at 90° to it — LM will not
    cross a saddle in theta on its own, and the two seeds cover both elongation senses."""
    best = float("inf")
    s_major, s_minor = max(sy, sx), min(sy, sx)
    for th in (theta0, theta0 + 0.5 * math.pi):
        popt = _curve_fit(_rot_gauss2d, (ys, xs), vs,
                          [amp0, y_c, x_c, s_major, s_minor, th, base0],
                          jac=_rot_gauss2d_jac, maxfev=150)
        if popt is None:
            continue
        sse = float(((_rot_gauss2d((ys, xs), *popt) - vs) ** 2).sum())
        best = min(best, sse)
    return best


def _two_gauss2d(coords: Tuple[np.ndarray, np.ndarray], a1: float, y1: float, x1: float,
                 a2: float, y2: float, x2: float, s: float, base: float) -> np.ndarray:
    yy, xx = coords
    g1 = np.exp(-((yy - y1) ** 2 + (xx - x1) ** 2) / (2.0 * s ** 2))
    g2 = np.exp(-((yy - y2) ** 2 + (xx - x2) ** 2) / (2.0 * s ** 2))
    return base + a1 * g1 + a2 * g2


def _two_gauss2d_jac(coords: Tuple[np.ndarray, np.ndarray], a1: float, y1: float, x1: float,
                     a2: float, y2: float, x2: float, s: float, base: float) -> np.ndarray:
    yy, xx = coords
    d1y, d1x, d2y, d2x = yy - y1, xx - x1, yy - y2, xx - x2
    g1 = np.exp(-(d1y ** 2 + d1x ** 2) / (2.0 * s ** 2))
    g2 = np.exp(-(d2y ** 2 + d2x ** 2) / (2.0 * s ** 2))
    return np.stack([g1, a1 * g1 * d1y / s ** 2, a1 * g1 * d1x / s ** 2,
                     g2, a2 * g2 * d2y / s ** 2, a2 * g2 * d2x / s ** 2,
                     (a1 * g1 * (d1y ** 2 + d1x ** 2) + a2 * g2 * (d2y ** 2 + d2x ** 2)) / s ** 3,
                     np.ones_like(yy)], axis=1)


def _crop_pixels(crop: np.ndarray, y_off: int, x_off: int, y_guess: float, x_guess: float,
                 others: Optional[np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The pixels a fit may use — the crop minus the Voronoi cells of other candidates —
    as flat ``(ys, xs, values)`` plus the border-median background estimate."""
    h, w = crop.shape
    yy, xx = np.mgrid[y_off:y_off + h, x_off:x_off + w].astype(float)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    base0 = float(np.median(border))
    use = np.ones((h, w), dtype=bool)
    if others is not None and len(others):
        d_self = (yy - y_guess) ** 2 + (xx - x_guess) ** 2
        d_other = np.min((yy[..., None] - others[:, 0]) ** 2
                         + (xx[..., None] - others[:, 1]) ** 2, axis=-1)
        use &= d_self <= d_other
    return yy[use], xx[use], crop[use].astype(float), base0


def _fit_xy(crop: np.ndarray, y_off: int, x_off: int, y_guess: float, x_guess: float,
            sigma_xy: float, others: Optional[np.ndarray] = None
            ) -> Tuple[float, float, float, float, float, bool, float]:
    """Axis-aligned 2-D Gaussian fit → ``(y0, x0, amp, sigma_y, sigma_x, fitted, sse)``.

    ``others`` is an ``(K, 2)`` array of OTHER candidates' ``(y, x)`` in the same
    projection: a pixel nearer to one of them than to ``(y_guess, x_guess)`` is left out of
    the fit, so a touching neighbour does not widen this bead (the Voronoi cell of the
    candidate). Moments fallback (fitted=False, sse NaN) when the solver fails or the fitted
    centre leaves the crop."""
    h, w = crop.shape
    ys, xs, vs, base0 = _crop_pixels(crop, y_off, x_off, y_guess, x_guess, others)
    sig = np.clip(vs - base0, 0.0, None)
    tot = float(sig.sum())
    nan = float("nan")
    if tot <= 0 or len(vs) < 7:
        return y_guess, x_guess, 0.0, nan, nan, False, nan
    cy = float((sig * ys).sum() / tot)
    cx = float((sig * xs).sum() / tot)
    vy = float((sig * (ys - cy) ** 2).sum() / tot)
    vx = float((sig * (xs - cx) ** 2).sum() / tot)
    sy0, sx0 = math.sqrt(max(vy, 0.09)), math.sqrt(max(vx, 0.09))
    amp0 = float(vs.max() - base0)
    if amp0 <= 0:
        return cy, cx, amp0, sy0, sx0, False, nan
    popt = _curve_fit(_gauss2d, (ys, xs), vs, [amp0, y_guess, x_guess, sigma_xy, sigma_xy, base0],
                      jac=_gauss2d_jac, maxfev=300)
    if popt is None:
        return cy, cx, amp0, sy0, sx0, False, nan
    amp, y0, x0, sy, sx, base = popt
    sy, sx = abs(sy), abs(sx)
    inside = (y_off - 0.5 <= y0 <= y_off + h - 0.5) and (x_off - 0.5 <= x0 <= x_off + w - 0.5)
    if not (np.isfinite(y0) and np.isfinite(x0) and amp > 0 and inside
            and np.isfinite(sy) and np.isfinite(sx)):
        return cy, cx, amp0, sy0, sx0, False, nan
    sse = float(((_gauss2d((ys, xs), amp, y0, x0, sy, sx, base) - vs) ** 2).sum())
    return y0, x0, amp, sy, sx, True, sse


def _fit_two_xy(crop: np.ndarray, y_off: int, x_off: int, y_guess: float, x_guess: float,
                sigma_xy: float, others: Optional[np.ndarray], y_c: float, x_c: float,
                sy1: float, sx1: float
                ) -> Optional[Tuple[List[Tuple[float, float, float]], float, float, float]]:
    """Two touching beads that the LoG saw as one blob: fit the sum of two round Gaussians
    with ONE shared sigma. Seeded along the blob's principal axis at the separation the
    single fit's excess width implies (``sigma_major² ≈ sigma² + d²/4``). Returns
    ``([(y, x, amp), (y, x, amp)], sigma, sse, sse_ridge)`` — the last being the best a
    single rotated elongated Gaussian does on the same pixels — or ``None`` when the solver
    gives up or the two centres leave the crop."""
    h, w = crop.shape
    ys, xs, vs, base0 = _crop_pixels(crop, y_off, x_off, y_guess, x_guess, others)
    if len(vs) < 10:
        return None
    sig = np.clip(vs - base0, 0.0, None)
    tot = float(sig.sum())
    if tot <= 0:
        return None
    cy, cx = float((sig * ys).sum() / tot), float((sig * xs).sum() / tot)
    cov = np.cov(np.stack([ys - cy, xs - cx]), aweights=sig + 1e-12, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, int(np.argmax(vals))]
    lam = float(max(vals.max(), max(sy1, sx1) ** 2))
    d0 = 2.0 * math.sqrt(max(lam - sigma_xy ** 2, 0.25 * sigma_xy ** 2))
    amp0 = 0.5 * float(vs.max() - base0)
    p0 = [amp0, y_c - 0.5 * d0 * u[0], x_c - 0.5 * d0 * u[1],
          amp0, y_c + 0.5 * d0 * u[0], x_c + 0.5 * d0 * u[1], sigma_xy, base0]
    popt = _curve_fit(_two_gauss2d, (ys, xs), vs, p0, jac=_two_gauss2d_jac, maxfev=200)
    if popt is None:
        return None
    a1, y1, x1, a2, y2, x2, s, base = popt
    s = abs(s)

    def inside(y, x):
        return (y_off - 0.5 <= y <= y_off + h - 0.5) and (x_off - 0.5 <= x <= x_off + w - 0.5)

    if not (np.isfinite(s) and inside(y1, x1) and inside(y2, x2) and a1 > 0 and a2 > 0):
        return None
    sse = float(((_two_gauss2d((ys, xs), a1, y1, x1, a2, y2, x2, s, base) - vs) ** 2).sum())
    sse_ridge = _fit_rot_xy(ys, xs, vs, y_c, x_c, 2.0 * amp0, sy1, sx1, base0,
                            math.atan2(u[1], u[0]))
    return [(y1, x1, a1), (y2, x2, a2)], s, sse, sse_ridge


# ── step 6: merge duplicates across slabs ───────────────────────────────────────

def _dedup(pts: np.ndarray, score: np.ndarray, scale: Tuple[float, float, float],
           radius: float = 1.0) -> np.ndarray:
    """Indices to keep: greedy, best score first, suppressing anything within ``radius``
    in sigma-normalised ``(z, y, x)`` distance. The same bead found from two slabs lands
    within a fraction of a voxel of itself (the fits read the same raw column), so one sigma
    is ample — and small enough that two deblended beads a sigma apart both survive."""
    n = len(pts)
    if n <= 1:
        return np.arange(n)
    from scipy.spatial import cKDTree
    q = pts / np.asarray(scale, dtype=float)[None, :]
    tree = cKDTree(q)
    order = np.argsort(-score, kind="stable")
    alive = np.ones(n, dtype=bool)
    keep: List[int] = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(int(i))
        for j in tree.query_ball_point(q[i], radius):
            alive[j] = False
    return np.array(sorted(keep), dtype=int)


# ── one pass ────────────────────────────────────────────────────────────────────

def _one_pass(vol: np.ndarray, slab: int, overlap: int, p: Dict[str, Any]
              ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    nz, ny, nx = vol.shape
    s_xy, s_z = float(p["sigma_xy_px"]), float(p["sigma_z_px"])
    tol = float(p["size_tolerance"])
    how = str(p["projection"])
    min_snr = float(p["min_snr"])
    ranges = _slab_ranges(nz, slab, overlap)
    med_raw = float(np.median(vol))
    noise_raw = 1.4826 * float(np.median(np.abs(vol - med_raw))) or 1.0
    rej = {k: 0 for k in _REJECT_KEYS}

    rows: List[Tuple[float, ...]] = []        # z, y, x, amp, snr, sxy, sz, skew, slab, flags
    dropped: List[Tuple[float, float, float, str, Dict[str, float]]] = []   # diagnostics
    half_z_core = max(1, int(round(s_z)))
    z_margin = max(1, int(math.ceil(1.5 * s_z)))
    half = int(math.ceil(3.0 * s_xy)) + 1
    tol_z = min(0.95, 2.0 * tol)
    n_deblended = 0
    d_px, d_zpx = float(p.get("diameter_px") or 0.0), float(p.get("diameter_zpx") or 0.0)
    # the closest two deblended beads may be: three quarters of a diameter (two touching
    # spheres whose centres differ by up to the core depth in z project a little closer than
    # a diameter), never less than a sigma
    min_sep = max(1.0 * s_xy, 0.75 * d_px)
    # the merge radius: half a diameter when the diameter is known — closer than that is the
    # same bead found twice, because two beads cannot be — else one sigma
    merge_scale = ((max(0.5 * d_zpx, s_z), max(0.5 * d_px, s_xy), max(0.5 * d_px, s_xy))
                   if d_px > 0 and d_zpx > 0 else (s_z, s_xy, s_xy))

    def drop(reason: str, z: float, y: float, x: float, **detail: float) -> None:
        rej[reason] += 1
        dropped.append((float(z), float(y), float(x), reason, dict(detail)))

    from scipy.spatial import cKDTree

    def size_ok(sy: float, sx: float) -> bool:
        s_fit = math.sqrt(sy * sx)
        return (1.0 - tol) * s_xy <= s_fit <= (1.0 + tol) * s_xy

    def aspect_ok(sy: float, sx: float) -> bool:
        return max(sy, sx) / max(1e-9, min(sy, sx)) <= float(p["max_aspect"])

    def window_verdict(prof: np.ndarray, lo: int, hi: int) -> str:
        """Why a window holds no bead-like peak: its argmax is on the volume edge, on the
        window edge with the profile still rising beyond (the neighbouring slab's bead), or
        there simply is nothing prominent (dim)."""
        zp = int(lo + np.argmax(prof[lo:hi]))
        if zp <= 0 or zp >= nz - 1:
            return "edge_z"
        if (zp == lo and prof[lo - 1] > prof[lo]) or (zp == hi - 1 and prof[hi] > prof[hi - 1]):
            return "other_slab"
        return "dim"

    def axial(y0: float, x0: float, z_peak: int, lo: int, hi: int
              ) -> Tuple[str, float, float, float, bool]:
        """The axial stage for one lateral position → ``(verdict, z0, sigma_z, skew, ok)``,
        verdict ``""`` when accepted. Re-profiles at the refined (y, x), finds the peak
        nearest the one this component came from, and fits it bounded by the valleys to
        its neighbouring peaks — or jointly with a neighbour closer than 4 sigma."""
        prof, _b, wn = _z_profile(vol, y0, x0, s_xy)
        peaks = _z_peaks(prof, lo, hi, noise_raw * wn)
        if not peaks:
            return window_verdict(prof, lo, hi), float(z_peak), float("nan"), float("nan"), False
        zp = min(peaks, key=lambda k: abs(k - z_peak))
        if abs(zp - z_peak) > 2:
            return "fit_failed", float(zp), float("nan"), float("nan"), False
        if zp <= 0 or zp >= nz - 1:
            return "edge_z", float(zp), float("nan"), float("nan"), False
        left = [k for k in peaks if k < zp]
        right = [k for k in peaks if k > zp]
        near_nb = [k for k in left + right if abs(k - zp) <= 4.0 * s_z]
        if near_nb:
            nb = min(near_nb, key=lambda k: abs(k - zp))
            z0, amp_z, s_fit_z, ok = _fit_two_z(prof, zp, nb, s_z)
            s_lo = s_hi = s_fit_z
            split = False
        else:
            lo_b = (max(left) + int(np.argmin(prof[max(left):zp + 1]))) if left else 0
            hi_b = (zp + int(np.argmin(prof[zp:min(right) + 1])) + 1) if right else nz
            z0, amp_z, s_lo, s_hi, ok, split = _fit_z(prof, zp, s_z, lo_b, hi_b)
        if not (0.0 <= z0 <= nz - 1.0):
            return "edge_z", z0, float("nan"), float("nan"), False
        if not ok:
            if nz >= 5:
                # a bead HAS an axial profile a Gaussian fits; what has none — a noise blip,
                # the cap of a large aggregate seen through a thin slab, a fibre crossing —
                # is exactly what the axial size test exists to refuse, and a fallback
                # parabola would wave it through unmeasured
                return "fit_failed", z0, float("nan"), float("nan"), False
            return "", z0, float("nan"), float("nan"), True
        s_zfit = 0.5 * (s_lo + s_hi)
        if not ((1.0 - tol_z) * s_z <= s_zfit <= (1.0 + tol_z) * s_z):
            return "size_z", z0, s_zfit, float("nan"), False
        skew = (s_hi - s_lo) / max(1e-9, s_hi + s_lo) if split else float("nan")
        return "", z0, s_zfit, skew, True

    for si, (a, b) in enumerate(ranges):
        proj = _project(vol, a, b, how)
        ys, xs, snr, ridge = _detect_2d(proj, s_xy, min_snr)
        cand = np.stack([ys, xs], axis=1).astype(float) if len(ys) else np.zeros((0, 2))
        ctree = cKDTree(cand) if len(cand) > 1 else None
        lo_z, hi_z = max(0, a - z_margin), min(nz, b + z_margin)
        # accepted components of THIS slab, held back until the slab is done: a component
        # whose fit was masked by neighbours is only trusted if one of those neighbours
        # turned out to be a bead too (see the second loop)
        held: List[Tuple[Tuple[float, ...], int, List[int], bool,
                         np.ndarray, int, int, int, int]] = []
        for k, (yi, xi, q) in enumerate(zip(ys.tolist(), xs.tolist(), snr.tolist())):
            if not (ridge[k] <= _RIDGE_ASPECT):
                drop("aspect", a, yi, xi, ridge=float(min(ridge[k], 99.0)), snr=q)
                continue
            prof, _bounds, wn = _z_profile(vol, float(yi), float(xi), s_xy)
            peaks = _z_peaks(prof, lo_z, hi_z, noise_raw * wn)
            if peaks and not p.get("multi_peak", True):
                peaks = [max(peaks, key=lambda k: prof[k])]
            if not peaks:
                drop(window_verdict(prof, lo_z, hi_z), lo_z + int(np.argmax(prof[lo_z:hi_z])),
                     yi, xi, snr=q)
                continue
            others = None
            near: List[int] = []
            if ctree is not None:
                near = [j for j in ctree.query_ball_point(cand[k], 2.0 * half + 1.0) if j != k]
                if near:
                    others = cand[near]
            y0c, y1c = max(0, yi - half), min(ny, yi + half + 1)
            x0c, x1c = max(0, xi - half), min(nx, xi + half + 1)
            for z_peak in peaks:
                if z_peak <= 0 or z_peak >= nz - 1:
                    drop("edge_z", z_peak, yi, xi)
                    continue
                # lateral sub-pixel + size on the planes around THIS peak, with the pixels
                # that belong to a neighbouring candidate left out
                hz = 0 if (len(peaks) > 1 and _STACKED_SINGLE_PLANE) else half_z_core
                za, zb = max(0, z_peak - hz), min(nz, z_peak + hz + 1)
                core = vol[za:zb].mean(axis=0)
                crop = core[y0c:y1c, x0c:x1c].astype(float)
                if crop.shape[0] < 3 or crop.shape[1] < 3:
                    drop("fit_failed", z_peak, yi, xi)
                    continue
                y0, x0, amp, sy, sx, ok_xy, sse1 = _fit_xy(crop, y0c, x0c, float(yi), float(xi),
                                                           s_xy, others)
                if not ok_xy or not (np.isfinite(sy) and np.isfinite(sx)):
                    drop("fit_failed", z_peak, yi, xi)
                    continue
                # the fitted peak must clear the SAME floor over the raw noise that the LoG
                # peak cleared over the response noise: a max projection of several planes
                # has a heavier tail than one plane, and with many thin slabs a noise blip a
                # few counts high slips past the response test and then fits as a sub-pixel
                # "bead"
                if amp < min_snr * noise_raw:
                    drop("dim", z_peak, y0, x0, amp=amp, noise=noise_raw, snr=q)
                    continue
                comps: List[Tuple[float, float, float, float, float, bool]] = []
                if size_ok(sy, sx) and aspect_ok(sy, sx):
                    comps = [(y0, x0, amp, sy, sx, False)]
                elif p.get("deblend", True) and math.sqrt(sy * sx) <= 2.5 * s_xy:
                    # a compact blob that is too wide or too long for one bead may be TWO
                    # touching beads: try the two-bead model and accept it only when both
                    # halves are bead-sized and bead-bright, clearly apart, and together
                    # explain the pixels markedly better than ONE object did — one bead,
                    # and one elongated object at any angle (a fibre segment), which is the
                    # thing a pair of round blobs can impersonate inside a masked crop
                    two = _fit_two_xy(crop, y0c, x0c, float(yi), float(xi), s_xy, others,
                                      y0, x0, sy, sx)
                    if two is not None:
                        (c1, c2), s2, sse2, sse_ridge = two
                        sep = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                        one = min(sse1, sse_ridge) if np.isfinite(sse1) else sse_ridge
                        # the pair's shared sigma must sit inside HALF the size tolerance:
                        # with two centres free the model can absorb a wrong width into a
                        # wrong separation, so a pair that is not plainly bead-sized is not
                        # a pair
                        pair_size_ok = (1.0 - 0.5 * tol) * s_xy <= s2 <= (1.0 + 0.5 * tol) * s_xy
                        if (pair_size_ok and sep >= min_sep and sep <= 2.0 * half
                                and min(c1[2], c2[2]) >= min_snr * noise_raw
                                and np.isfinite(one) and sse2 <= 0.6 * one):
                            comps = [(c1[0], c1[1], c1[2], s2, s2, True),
                                     (c2[0], c2[1], c2[2], s2, s2, True)]
                            n_deblended += 1
                if not comps:
                    reason = "size_xy" if not size_ok(sy, sx) else "aspect"
                    drop(reason, z_peak, y0, x0, sy=sy, sx=sx, amp=amp, snr=q)
                    continue
                for (cy, cx, camp, csy, csx, deb) in comps:
                    verdict, z0, s_zfit, skew, ok_z = axial(cy, cx, z_peak, lo_z, hi_z)
                    if verdict:
                        drop(verdict, z0, cy, cx, amp=camp, snr=q, deblended=float(deb))
                        continue
                    flags = (1.0 if deb else 0.0) + (2.0 if len(peaks) > 1 else 0.0)
                    held.append(((z0, cy, cx, camp, q, math.sqrt(csy * csx), s_zfit, skew,
                                  float(si), flags), k, near, deb, crop, y0c, x0c, yi, xi))

        # A masked fit is only as trustworthy as the neighbours it was masked against. Two
        # touching beads mask each other and BOTH pass — fine. The END of a fibre also
        # passes its masked fit, because the chain of LoG peaks along the fibre cuts the
        # crop into a bead-sized Voronoi cell — but every one of those neighbours was itself
        # refused (ridge, width). So: a single bead none of whose masking neighbours became
        # a bead is re-fitted WITHOUT the mask and has to pass the width tests on its own;
        # a deblended pair in the same position is not trusted at all (an unmasked single
        # fit cannot vouch for a pair, and a pair the mask conjured out of a ridge is the
        # one false positive this whole stage exists to prevent).
        acc = {h[1] for h in held}
        for row, k, near, deb, crop, y0c, x0c, yi, xi in held:
            if near and not (set(near) & acc):
                if deb:
                    drop("aspect", row[0], row[1], row[2], deblend_unverified=1.0, snr=row[4])
                    continue
                y1, x1, amp1, sy1, sx1, ok1, _sse = _fit_xy(crop, y0c, x0c, float(yi),
                                                            float(xi), s_xy)
                if not ok1:
                    drop("fit_failed", row[0], row[1], row[2], unmasked=1.0)
                    continue
                if not size_ok(sy1, sx1):
                    drop("size_xy", row[0], row[1], row[2], sy=sy1, sx=sx1, amp=amp1,
                         snr=row[4], unmasked=1.0)
                    continue
                if not aspect_ok(sy1, sx1):
                    drop("aspect", row[0], row[1], row[2], sy=sy1, sx=sx1, amp=amp1,
                         snr=row[4], unmasked=1.0)
                    continue
            rows.append(row)

    if not rows:
        pts = np.zeros((0, 3), dtype=float)
        cols = {k: np.zeros(0, dtype=float) for k in
                ("amplitude", "snr", "sigma_xy_px", "sigma_z_px", "skew_z")}
        cols["slab"] = np.zeros(0, dtype=np.int64)
        cols["flags"] = np.zeros(0, dtype=np.int64)
    else:
        arr = np.asarray(rows, dtype=float)
        keep = _dedup(arr[:, :3], arr[:, 3], merge_scale)
        rej["duplicate"] = int(len(arr) - len(keep))
        arr = arr[keep]
        pts = np.ascontiguousarray(arr[:, :3])
        cols = {
            "amplitude": arr[:, 3].copy(), "snr": arr[:, 4].copy(),
            "sigma_xy_px": arr[:, 5].copy(), "sigma_z_px": arr[:, 6].copy(),
            "skew_z": arr[:, 7].copy(), "slab": arr[:, 8].astype(np.int64),
            # 1 = one of a deblended pair, 2 = its candidate had several z peaks
            "flags": arr[:, 9].astype(np.int64),
        }
    info = {"slab_px": int(slab), "overlap_px": int(overlap), "n_slabs": len(ranges),
            "n_candidates": int(sum(rej.values()) + len(pts)), "rejected": rej,
            "deblended_pairs": int(n_deblended),
            # every candidate that did not become a bead, with where it was and why —
            # for a validation to ask "what happened to the bead I planted HERE"
            "dropped": dropped}
    return pts, cols, info


# ── entry point ─────────────────────────────────────────────────────────────────

def find_beads(volume_zyx: np.ndarray, voxel_size_um: Sequence[float],
               params: Optional[Dict[str, Any]] = None
               ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Find fluorescent beads in one raw ``(Z, Y, X)`` volume.

    ``voxel_size_um`` is ``(dz, dy, dx)`` and is used **only** for the ``info`` report; all
    sizes in ``params`` are already voxels (see :data:`DEFAULTS`). Returns
    ``(points_zyx, columns, info)``: sub-voxel ``(z, y, x)`` positions ``(N, 3)`` float64 in
    voxel units, index-aligned per-bead columns (``amplitude``, ``snr``, ``sigma_xy_px``,
    ``sigma_z_px``, ``skew_z``, ``slab``), and a dict describing the run (slab thickness
    and overlap used, slab count, candidates, rejections by reason, the dropped
    candidates, passes run, despiked voxels)."""
    vol = np.asarray(volume_zyx)
    if vol.ndim != 3:
        raise ValueError(f"find_beads: expected a (Z, Y, X) volume, got shape {vol.shape}")
    nz = vol.shape[0]
    if nz < 3:
        raise ValueError(f"find_beads: a slab needs at least 3 planes, got Z={nz}")
    p = dict(DEFAULTS)
    p.update(params or {})
    how = str(p.get("projection", "max")).lower()
    if how not in ("max", "mean", "min"):
        raise ValueError(f"find_beads: projection must be max|mean|min, got {how!r}")
    v = vol.astype(float, copy=False)
    if how == "min":                       # dark beads: min-projection == max of the inverse
        v = float(v.max()) - v
        p["projection"] = "max"
    else:
        p["projection"] = how
    s_xy, s_z = float(p["sigma_xy_px"]), float(p["sigma_z_px"])
    v, n_spikes = _despike(v, s_xy)
    slab_req = int(p.get("slab_px") or 0)
    ov_req = int(p.get("overlap_px") if p.get("overlap_px") is not None else -1)
    target = float(p.get("target_overlap", 0.10))

    def summary(info: Dict[str, Any], n: int) -> Dict[str, Any]:
        return {k: info[k] for k in ("slab_px", "overlap_px", "n_slabs")} | {"n": n}

    passes: List[Dict[str, Any]] = []
    if slab_req > 0:
        slab = int(np.clip(slab_req, 1, nz))
        ov = _auto_overlap(slab, s_z) if ov_req < 0 else int(np.clip(ov_req, 0, slab - 1))
        pts, cols, info = _one_pass(v, slab, ov, p)
        passes.append(summary(info, len(pts)))
    else:
        # pass 0: the whole stack as one projection — the density estimate the slab
        # thickness is derived from (and the sparse-field detector, were it not capped)
        slab = nz
        pts, cols, info = _one_pass(v, slab, 0, p)
        passes.append(summary(info, len(pts)))
        for _ in range(2):
            want = auto_slab_thickness(len(pts), v.shape, s_xy, s_z, target_overlap=target)
            if want >= slab or abs(want - slab) <= max(1, int(0.25 * slab)):
                break
            slab = want
            ov = _auto_overlap(slab, s_z)
            pts, cols, info = _one_pass(v, slab, ov, p)
            passes.append(summary(info, len(pts)))
    info["passes"] = passes
    info["despiked_voxels"] = int(n_spikes)
    info["voxel_size_um"] = tuple(float(x) for x in voxel_size_um)
    info["projection"] = how
    info["sigma_xy_px"], info["sigma_z_px"] = s_xy, s_z
    return pts, cols, info
