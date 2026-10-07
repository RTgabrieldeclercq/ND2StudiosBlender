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
   to a background-subtracted, matched-filter z-profile; the peak is sought near the slab
   that found it, then a **split Gaussian** (one sigma below the peak, another above — the
   confocal axial profile is skewed by the PSF and the index mismatch) is fitted to it.
4. **Lateral sub-pixel + size.** The planes around the fitted z are averaged and a 2-D
   Gaussian (axis-aligned, 6 parameters) fitted for the sub-pixel (y, x) and the apparent
   sigmas; pixels nearer to ANOTHER candidate of the same slab are left out of the fit, so
   two touching beads do not inflate each other's width.
5. **Filters** — only things that look like a bead survive: a Hessian ridge test on the
   projection (a fibre or a scan line has no curvature along itself), lateral size inside
   a tolerance band around the expected apparent size, fitted aspect ratio below a bound,
   axial size inside a (wider) band, a peak that is not on the first/last plane.
6. **Merge** the same bead found in two overlapping slabs: greedy, brightest first, in
   sigma-normalised anisotropic distance.

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
               ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Matched-filter z-profile under ``(y, x)``: per plane, the Gaussian-weighted mean of
    a ``±half`` crop minus the median of the crop's outer ring. Returns the profile and the
    crop bounds ``(y0, y1, x0, x1)``."""
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
    return prof, (y0, y1, x0, x1)


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


def _fit_z(prof: np.ndarray, z_peak: int, sigma_z: float
           ) -> Tuple[float, float, float, float, bool]:
    """Split-Gaussian fit around ``z_peak`` → ``(z0, amp, sigma_lo, sigma_hi, fitted)``.
    Falls back to a 3-point parabola (sigmas NaN) when the window is too short, the fit
    does not converge, or it wanders more than 1.5 planes from the peak it started on."""
    nz = len(prof)
    w = max(2, int(math.ceil(3.0 * sigma_z)))
    a, b = max(0, z_peak - w), min(nz, z_peak + w + 1)
    zz = np.arange(a, b, dtype=float)
    pp = prof[a:b]
    amp0 = float(prof[z_peak] - pp.min())

    def parabola() -> Tuple[float, float, float, float, bool]:
        if 0 < z_peak < nz - 1:
            l, c, r = prof[z_peak - 1], prof[z_peak], prof[z_peak + 1]
            den = l - 2.0 * c + r
            dz = 0.5 * (l - r) / den if den < 0 else 0.0
            dz = float(np.clip(dz, -0.5, 0.5))
        else:
            dz = 0.0
        return float(z_peak + dz), amp0, float("nan"), float("nan"), False

    if len(zz) < 5 or amp0 <= 0:
        return parabola()
    from scipy.optimize import curve_fit
    p0 = [amp0, float(z_peak), sigma_z, sigma_z, float(pp.min())]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # pcov is never used
            popt, _ = curve_fit(_split_gauss, zz, pp, p0=p0, jac=_split_gauss_jac,
                                method="lm", maxfev=200)
    except Exception:  # noqa: BLE001 — a failed fit degrades to the parabola
        return parabola()
    amp, z0, s_lo, s_hi, _base = (float(v) for v in popt)
    s_lo, s_hi = abs(s_lo), abs(s_hi)
    if not (np.isfinite(z0) and amp > 0 and abs(z0 - z_peak) <= 1.5
            and 0.2 <= s_lo <= 6.0 * sigma_z + 2 and 0.2 <= s_hi <= 6.0 * sigma_z + 2):
        return parabola()
    return z0, amp, s_lo, s_hi, True


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


def _fit_xy(crop: np.ndarray, y_off: int, x_off: int, y_guess: float, x_guess: float,
            sigma_xy: float, others: Optional[np.ndarray] = None
            ) -> Tuple[float, float, float, float, float, bool]:
    """Axis-aligned 2-D Gaussian fit → ``(y0, x0, amp, sigma_y, sigma_x, fitted)``.

    ``others`` is an ``(K, 2)`` array of OTHER candidates' ``(y, x)`` in the same
    projection: a pixel nearer to one of them than to ``(y_guess, x_guess)`` is left out of
    the fit, so a touching neighbour does not widen this bead (the Voronoi cell of the
    candidate). Moments fallback (fitted=False) when the solver fails or the fitted centre
    leaves the crop."""
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
    ys, xs, vs = yy[use], xx[use], crop[use].astype(float)
    sig = np.clip(vs - base0, 0.0, None)
    tot = float(sig.sum())
    if tot <= 0 or use.sum() < 7:
        return y_guess, x_guess, 0.0, float("nan"), float("nan"), False
    cy = float((sig * ys).sum() / tot)
    cx = float((sig * xs).sum() / tot)
    vy = float((sig * (ys - cy) ** 2).sum() / tot)
    vx = float((sig * (xs - cx) ** 2).sum() / tot)
    sy0, sx0 = math.sqrt(max(vy, 0.09)), math.sqrt(max(vx, 0.09))
    amp0 = float(vs.max() - base0)
    if amp0 <= 0:
        return cy, cx, amp0, sy0, sx0, False
    from scipy.optimize import curve_fit
    p0 = [amp0, y_guess, x_guess, sigma_xy, sigma_xy, base0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # pcov is never used
            popt, _ = curve_fit(_gauss2d, (ys, xs), vs, p0=p0, jac=_gauss2d_jac,
                                method="lm", maxfev=300)
    except Exception:  # noqa: BLE001 — a failed fit degrades to moments
        return cy, cx, amp0, sy0, sx0, False
    amp, y0, x0, sy, sx, _base = (float(v) for v in popt)
    sy, sx = abs(sy), abs(sx)
    inside = (y_off - 0.5 <= y0 <= y_off + h - 0.5) and (x_off - 0.5 <= x0 <= x_off + w - 0.5)
    if not (np.isfinite(y0) and np.isfinite(x0) and amp > 0 and inside
            and np.isfinite(sy) and np.isfinite(sx)):
        return cy, cx, amp0, sy0, sx0, False
    return y0, x0, amp, sy, sx, True


# ── step 6: merge duplicates across slabs ───────────────────────────────────────

def _dedup(pts: np.ndarray, score: np.ndarray, scale: Tuple[float, float, float],
           radius: float = 2.0) -> np.ndarray:
    """Indices to keep: greedy, best score first, suppressing anything within ``radius``
    in sigma-normalised ``(z, y, x)`` distance."""
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
    ranges = _slab_ranges(nz, slab, overlap)
    med_raw = float(np.median(vol))
    noise_raw = 1.4826 * float(np.median(np.abs(vol - med_raw))) or 1.0
    rej = {k: 0 for k in _REJECT_KEYS}

    rows: List[Tuple[float, ...]] = []        # z, y, x, amp, snr, sxy, sz, skew, slab
    dropped: List[Tuple[float, float, float, str, Dict[str, float]]] = []   # diagnostics
    half_z_core = max(1, int(round(s_z)))
    z_margin = max(1, int(math.ceil(1.5 * s_z)))
    half = int(math.ceil(3.0 * s_xy)) + 1

    def drop(reason: str, z: float, y: float, x: float, **detail: float) -> None:
        rej[reason] += 1
        dropped.append((float(z), float(y), float(x), reason, dict(detail)))

    from scipy.spatial import cKDTree

    def size_ok(sy: float, sx: float) -> bool:
        s_fit = math.sqrt(sy * sx)
        return (1.0 - tol) * s_xy <= s_fit <= (1.0 + tol) * s_xy

    def aspect_ok(sy: float, sx: float) -> bool:
        return max(sy, sx) / max(1e-9, min(sy, sx)) <= float(p["max_aspect"])

    for si, (a, b) in enumerate(ranges):
        proj = _project(vol, a, b, how)
        ys, xs, snr, ridge = _detect_2d(proj, s_xy, float(p["min_snr"]))
        cand = np.stack([ys, xs], axis=1).astype(float) if len(ys) else np.zeros((0, 2))
        ctree = cKDTree(cand) if len(cand) > 1 else None
        # accepted candidates of THIS slab, held back until the slab is done: a candidate
        # whose fit was masked by neighbours is only trusted if one of those neighbours
        # turned out to be a bead too (see the second loop)
        held: List[Tuple[Tuple[float, ...], int, List[int], np.ndarray, int, int, int, int]] = []
        for k, (yi, xi, q) in enumerate(zip(ys.tolist(), xs.tolist(), snr.tolist())):
            if not (ridge[k] <= _RIDGE_ASPECT):
                drop("aspect", a, yi, xi, ridge=float(min(ridge[k], 99.0)), snr=q)
                continue
            prof, _bounds = _z_profile(vol, float(yi), float(xi), s_xy)
            lo_z, hi_z = max(0, a - z_margin), min(nz, b + z_margin)
            z_peak = int(lo_z + np.argmax(prof[lo_z:hi_z]))
            if z_peak <= 0 or z_peak >= nz - 1:
                drop("edge_z", z_peak, yi, xi)
                continue
            # The peak must belong to THIS slab: a slab that only sees a bead's axial tail
            # puts the window's argmax on the window edge with the profile still rising
            # beyond it. That bead is the neighbouring slab's to find — fitting it here
            # would plant a ghost two or three planes from the real one.
            if ((z_peak == lo_z and prof[lo_z - 1] > prof[lo_z])
                    or (z_peak == hi_z - 1 and prof[hi_z] > prof[hi_z - 1])):
                drop("other_slab", z_peak, yi, xi)
                continue
            # lateral sub-pixel + size on the planes around the peak, with the pixels that
            # belong to a neighbouring candidate left out
            za, zb = max(0, z_peak - half_z_core), min(nz, z_peak + half_z_core + 1)
            core = vol[za:zb].mean(axis=0)
            y0c, y1c = max(0, yi - half), min(ny, yi + half + 1)
            x0c, x1c = max(0, xi - half), min(nx, xi + half + 1)
            crop = core[y0c:y1c, x0c:x1c].astype(float)
            if crop.shape[0] < 3 or crop.shape[1] < 3:
                drop("fit_failed", z_peak, yi, xi)
                continue
            others = None
            near: List[int] = []
            if ctree is not None:
                near = [j for j in ctree.query_ball_point(cand[k], 2.0 * half + 1.0) if j != k]
                if near:
                    others = cand[near]
            y0, x0, amp, sy, sx, ok_xy = _fit_xy(crop, y0c, x0c, float(yi), float(xi), s_xy,
                                                 others)
            if not ok_xy or not (np.isfinite(sy) and np.isfinite(sx)):
                drop("fit_failed", z_peak, yi, xi)
                continue
            # the fitted peak must clear the SAME floor over the raw noise that the LoG peak
            # cleared over the response noise: a max projection of several planes has a
            # heavier tail than one plane, and with many thin slabs a noise blip a few
            # counts high slips past the response test and then fits as a sub-pixel "bead"
            if amp < float(p["min_snr"]) * noise_raw:
                drop("dim", z_peak, y0, x0, amp=amp, noise=noise_raw, snr=q)
                continue
            s_fit = math.sqrt(sy * sx)
            if not size_ok(sy, sx):
                drop("size_xy", z_peak, y0, x0, sy=sy, sx=sx, amp=amp, snr=q)
                continue
            if not aspect_ok(sy, sx):
                drop("aspect", z_peak, y0, x0, sy=sy, sx=sx, amp=amp, snr=q)
                continue
            # axial: re-profile at the refined (y, x), fit the split Gaussian
            prof, _bounds = _z_profile(vol, y0, x0, s_xy)
            z_peak = int(lo_z + np.argmax(prof[lo_z:hi_z]))
            if z_peak <= 0 or z_peak >= nz - 1:
                drop("edge_z", z_peak, y0, x0)
                continue
            if ((z_peak == lo_z and prof[lo_z - 1] > prof[lo_z])
                    or (z_peak == hi_z - 1 and prof[hi_z] > prof[hi_z - 1])):
                drop("other_slab", z_peak, y0, x0)
                continue
            z0, amp_z, s_lo, s_hi, ok_z = _fit_z(prof, z_peak, s_z)
            if not (0.0 <= z0 <= nz - 1.0):
                drop("edge_z", z0, y0, x0)
                continue
            if not ok_z and nz >= 5:
                # a bead HAS an axial profile a split Gaussian fits; what has none — a noise
                # blip, the cap of a large aggregate seen through a thin slab, a fibre
                # crossing — is exactly what the axial size test exists to refuse, and a
                # fallback parabola would wave it through unmeasured
                drop("fit_failed", z0, y0, x0, axial=1.0, amp=amp, snr=q)
                continue
            if ok_z:
                s_zfit = 0.5 * (s_lo + s_hi)
                tol_z = min(0.95, 2.0 * tol)
                if not ((1.0 - tol_z) * s_z <= s_zfit <= (1.0 + tol_z) * s_z):
                    drop("size_z", z0, y0, x0, s_lo=s_lo, s_hi=s_hi, amp=amp, snr=q)
                    continue
                skew = (s_hi - s_lo) / max(1e-9, s_hi + s_lo)
            else:
                s_zfit, skew = float("nan"), float("nan")
            held.append(((z0, y0, x0, amp, q, s_fit, s_zfit, skew, float(si)),
                         k, near, crop, y0c, x0c, yi, xi))

        # A masked fit is only as trustworthy as the neighbours it was masked against. Two
        # touching beads mask each other and BOTH pass — fine. The END of a fibre also
        # passes its masked fit, because the chain of LoG peaks along the fibre cuts the
        # crop into a bead-sized Voronoi cell — but every one of those neighbours was itself
        # refused (ridge, width). So: a candidate none of whose masking neighbours became a
        # bead is re-fitted WITHOUT the mask and has to pass the width tests on its own.
        acc = {h[1] for h in held}
        for row, k, near, crop, y0c, x0c, yi, xi in held:
            if near and not (set(near) & acc):
                y1, x1, amp1, sy1, sx1, ok1 = _fit_xy(crop, y0c, x0c, float(yi), float(xi), s_xy)
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
    else:
        arr = np.asarray(rows, dtype=float)
        keep = _dedup(arr[:, :3], arr[:, 3], (s_z, s_xy, s_xy))
        rej["duplicate"] = int(len(arr) - len(keep))
        arr = arr[keep]
        pts = np.ascontiguousarray(arr[:, :3])
        cols = {
            "amplitude": arr[:, 3].copy(), "snr": arr[:, 4].copy(),
            "sigma_xy_px": arr[:, 5].copy(), "sigma_z_px": arr[:, 6].copy(),
            "skew_z": arr[:, 7].copy(), "slab": arr[:, 8].astype(np.int64),
        }
    info = {"slab_px": int(slab), "overlap_px": int(overlap), "n_slabs": len(ranges),
            "n_candidates": int(sum(rej.values()) + len(pts)), "rejected": rej,
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
