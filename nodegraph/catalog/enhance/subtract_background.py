"""Subtract Background (``enhance.subtract_background``) — Estimate a background surface
with any of nine popular estimators (Sternberg rolling ball, tunable ellipsoid, morphological
opening, minimum / median / percentile box, large-sigma Gaussian, least-squares polynomial
surface, whole-unit percentile) and remove it: subtract / signed subtract / divide, for a dark
or a light background, or output the estimate itself."""

from __future__ import annotations

import itertools

import numpy as np

from typing import Callable, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import subtract_background as _meta_subtract_background
from nodegraph.registry import (DimMode, Granularity, InBool, InDataset, InFloat, InInt,
                                InString, Mode, OutDataset)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.kernel_radius import (
    _AXIAL_TWIN_DOC,
    _radius_px,
    _radius_z_px,
    _require_window,
    _win,
)
from nodegraph.catalog._shared.map_image import _map_image
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Subtract Background — the estimator is a Mode, the arithmetic is a Mode ─────
#
# ImageJ's `Process > Subtract Background` under the name people look for, generalized to
# this engine's 6-D float Datasets and to the estimators the other popular packages reach
# for. It is ONE node because every one of them is the same data contract — estimate a
# smooth surface from the image's own pixels, then remove it — which is the
# `analysis.segment` charter: one node per contract, the algorithm as a Mode.
#
# **What it does NOT duplicate.** `enhance.tophat` is the *morphological* background
# subtractor (image minus its opening) and stays the fast default for small puncta;
# `enhance.flatten_field` is the Cell-Tracker port, fixed to a large-sigma Gaussian estimate
# but with the two things this node deliberately leaves out — a mean-restoring arithmetic
# (`subtract_mean` / `divide_mean`) and a background AVERAGED OVER T. That second one is why
# it is a separate node rather than a tenth method here: a time-averaged estimate reads the
# whole T series, so it needs a WHOLE_SERIES footprint, and `granularity` is keyed by exactly
# one Mode (`NodeSpec.footprint_mode`) — which this node spends on its 2D/3D lever. Declaring
# WHOLE_SERIES for every method to buy one of them would force whole-series reads on a plain
# rolling ball, and declaring a per-plane footprint while reading across T is the
# misdeclaration `analysis.threshold`'s comment records. So the temporal estimate stays where
# it already works, and this node stays honestly per-unit.

#: ``method`` -> the surface estimator. Ordered cheapest-to-explain first, with the name
#: users arrive searching for at the top.
_METHODS: Tuple[str, ...] = (
    "rolling_ball", "ellipsoid", "opening", "minimum", "median", "percentile",
    "gaussian", "polynomial", "global_percentile",
)
#: The estimators driven by a spatial RADIUS (the `radius` / `radius_z` sockets). `gaussian`
#: has its own sigma, and the last two have no kernel at all.
_RADIUS_METHODS = frozenset({"rolling_ball", "ellipsoid", "opening", "minimum", "median",
                             "percentile"})
#: The estimators whose cost grows with the radius, i.e. the ones `shrink` exists for — and
#: the `available_in` gate on that socket, so it is not even offered for the others.
#: MEASURED on this machine, not assumed (2026-08-05, 1024^2 float64 plane, r = 29 px /
#: 59-px box): rolling_ball 2.1 s, median_filter 57 s, percentile_filter 29 s — against
#: minimum_filter 0.02 s and grey_opening 0.04 s, which scipy runs as separable running
#: min/max and so are already O(1) per pixel in the radius. Those two therefore stay EXACT
#: and this socket stays hidden for them: there is no speed to buy, so paying accuracy for it
#: would be a pure loss, and offering a control that bought nothing would be worse.
_SHRINK_METHODS = frozenset({"rolling_ball", "ellipsoid", "median", "percentile"})
#: When coarsening, which reduction preserves what the estimator means. The three
#: "go under the surface" estimators want the block MINIMUM (ImageJ's own choice, and it
#: keeps the coarse surface below the fine one); a median/percentile background is a LEVEL,
#: and a min-reduce would quietly turn it into a minimum background.
_MEAN_REDUCED = frozenset({"median", "percentile"})
#: The estimators that read the `percentile` socket.
_PCT_METHODS = frozenset({"percentile", "global_percentile"})
#: ``combine`` -> the arithmetic that removes the estimate.
_COMBINES: Tuple[str, ...] = ("subtract", "subtract_signed", "divide")
#: Highest polynomial total degree accepted. Past this the fit is ill-conditioned even on
#: normalized coordinates, and a "background" with that many inflections is fitting objects.
_MAX_DEGREE = 6
#: Most pixels the polynomial fit samples. A smooth surface does not need four million
#: points, and a full 2048^2 design matrix at degree 4 is 15 columns x 4 M rows = 500 MB.
_POLY_SAMPLES = 200_000
#: Default background scale, in microns. Comfortably above one adherent cell, which is the
#: condition every estimator here needs to not subtract the objects themselves.
_R_DEFAULT = 25.0

#: ``approach`` -> the two things this node can mean by "subtract background" (2026-10-02).
#: ``estimate_surface`` is everything above: fit a smooth surface, remove it arithmetically.
#: ``zero_regions`` is the other request users make: DECIDE which pixels are background and
#: set them to nothing, leaving every object pixel untouched.
_APPROACHES: Tuple[str, ...] = ("estimate_surface", "zero_regions")
_SURFACE = frozenset({"estimate_surface"})
_ZERO = frozenset({"zero_regions"})
#: ``detector`` -> how the ``zero_regions`` approach decides what is background.
_DETECTORS: Tuple[str, ...] = ("sampled_region", "adaptive")
_SAMPLED = frozenset({"sampled_region"})
_ADAPTIVE = frozenset({"adaptive"})
#: Fewest region pixels the sampled detector accepts: a standard deviation needs two, and a
#: drawn region with one pixel in it is a mis-click, not a background sample.
_MIN_SAMPLE = 2


# ── the zero-regions detectors ─────────────────────────────────────────────────

def _region_shapes(raw):
    """The ``shapes`` param -> a list of shape dicts, or ``None`` when nothing is drawn.

    The same two forms ``analysis.roi_mask`` accepts, for the same reason: the GUI's draw
    tool writes JSON text into a STRING socket (``pick_kind="shapes"``), a headless caller
    hands in the list. Malformed JSON is refused with the parse error. Kept here rather than
    imported from the ROI node because a node module may not import another node's module."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if not isinstance(raw, str):
        return raw
    import json as _json
    try:
        val = _json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"subtract background (sampled region): `shapes` is not valid JSON ({exc}). Draw "
            "the background region with the Pick tool, or pass a list of shape objects.") from exc
    if val in (None, [], {}):
        return None
    if not isinstance(val, list):
        raise ValueError("subtract background (sampled region): `shapes` must be a JSON "
                         f"list, got {type(val).__name__}")
    return val


def _sampled_background(s: np.ndarray, region2d: np.ndarray, tolerance: float) -> np.ndarray:
    """Background = every voxel no brighter than the sampled region's own distribution.

    ``s`` is the unit in SIGNAL space (positive-going, so the polarity algebra has already
    run); ``region2d`` is the drawn ``(Y, X)`` mask, applied to every plane of a volume. The
    sample's mean and standard deviation set one cut, ``mean + tolerance * sd``, and
    everything at or below it is background — "similar to what you showed me, or darker".
    Darker-than-sample is included deliberately: a camera-offset corner is still not an
    object. Bright pixels inside the drawn region (a stray object the user swept over) widen
    the sd and so loosen the cut a little; they do not break it."""
    sample = s[..., region2d] if s.ndim == 3 else s[region2d]
    mu, sd = float(sample.mean()), float(sample.std())
    return s <= mu + tolerance * sd


def _adaptive_background(s: np.ndarray, sigmas: Sequence[float], tolerance: float,
                         low: float, high: float) -> np.ndarray:
    """Background = every voxel at or below a LOCAL cut, clamped into ``[low, high]``.

    ``cut = local_mean + tolerance * noise_sd``: the local mean under a Gaussian of the block
    scale rides the shading the way the surface estimators do, and is then used as a
    DECISION, not subtracted. The margin is ONE robust noise width for the whole unit —
    ``1.4826 * median(|s - local_mean|)``, the MAD of the high-pass residual — rather than
    Niblack's local standard deviation, on purpose: a local sd balloons next to any bright
    object (measured here: a compact object whose core sat 3x above its local mean was still
    zeroed at tolerance 3), so it ate exactly the pixels the approach exists to protect. The
    median is immune to the objects' tails as long as they cover under half the field.

    The clamp is the "within a user-defined range" half of the request, and it is what keeps
    adaptive thresholding from its classic failure: inside an object wider than the block
    the local mean rises to the object's own brightness, so its interior sits below its
    local cut and would be zeroed. An upper limit set below object brightness makes that
    impossible; a lower limit keeps the cut from collapsing onto a flat, noise-free field
    and zeroing nothing. ``high <= 0`` means no upper limit, mirroring this node's
    ``0 = auto`` convention elsewhere."""
    from scipy import ndimage as ndi
    mean = ndi.gaussian_filter(s, sigma=sigmas, mode="nearest")
    resid = s - mean
    noise = 1.4826 * float(np.median(np.abs(resid)))
    cut = mean + tolerance * noise
    cut = np.clip(cut, low, high if high > 0.0 else None)
    return s <= cut


# ── the estimators ─────────────────────────────────────────────────────────────

def _box(radii: Sequence[float]) -> Tuple[int, ...]:
    """Per-axis radii (px) -> an odd box size per axis. Box rather than ball footprints,
    like ``enhance.tophat`` and ``enhance.median``: scipy runs a rectangular min/max as a
    separable running extremum (0.02 s where a ball footprint takes seconds), and a
    background estimate is smooth enough that the corner voxels change nothing."""
    return tuple(_win(r) for r in radii)


def _ball_background(a: np.ndarray, radii: Sequence[float],
                     intensity: float) -> np.ndarray:
    """The Sternberg rolling ball, generalized to one radius per axis.

    Always goes through an explicit ``ellipsoid_kernel`` rather than
    ``rolling_ball(radius=)``, because microscope voxels are anisotropic and a single scalar
    radius would roll a ball that reaches through many more planes than it does pixels. This
    is not a departure from skimage: ``restoration.ball_kernel(r, n)`` IS
    ``ellipsoid_kernel([2r+1]*n, r)`` — same ``sqrt(r**2 - sum(d**2))`` surface — so an
    isotropic call here reproduces ``rolling_ball(radius=r)`` exactly, and `rolling_ball`'s
    ``intensity = radius`` is what the caller passes when the user has not set a height.
    """
    from skimage.restoration import ellipsoid_kernel, rolling_ball
    shape = tuple(2 * max(1, int(round(r))) + 1 for r in radii)
    return rolling_ball(a, kernel=ellipsoid_kernel(shape, float(intensity)))


def _monomials(ndim: int, degree: int) -> Tuple[Tuple[int, ...], ...]:
    """Every exponent tuple of TOTAL degree <= ``degree`` — the polynomial basis.

    Total degree, not per-axis: a tensor-product basis at degree 3 in 3D would carry an
    ``x**3 y**3 z**3`` term (degree 9) and fit structure no shading has."""
    return tuple(pw for pw in itertools.product(range(degree + 1), repeat=ndim)
                 if sum(pw) <= degree)


def _poly_background(a: np.ndarray, degree: int) -> np.ndarray:
    """A least-squares polynomial surface over the unit's own axes.

    Coordinates are normalized to ``[-1, 1]`` per axis before the fit: a raw pixel-index
    basis is catastrophically ill-conditioned past degree 2 on a 2048-px axis (the Vandermonde
    columns differ by 2048**degree), so the same fit that is exact here would return noise
    there. The fit is taken on a strided subsample of at most :data:`_POLY_SAMPLES` points and
    then EVALUATED on the full grid — a smooth surface is over-determined by 200 000 points,
    and the full design matrix is what would run the node out of memory. A singleton axis
    contributes only its constant term (its normalized coordinate is 0), so this is the same
    function in 2D and 3D."""
    powers = _monomials(a.ndim, degree)
    coords = [np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1) for n in a.shape]
    stride = max(1, int(np.ceil((a.size / float(_POLY_SAMPLES)) ** (1.0 / a.ndim))))
    picks = tuple(slice(None, None, stride) for _ in a.shape)
    sub = np.meshgrid(*[c[s] for c, s in zip(coords, picks)], indexing="ij")
    design = np.stack([np.prod([g ** p for g, p in zip(sub, pw)], axis=0).ravel()
                       for pw in powers], axis=1)
    if design.shape[0] < design.shape[1]:
        raise ValueError(
            f"subtract background (polynomial): a degree-{degree} surface has "
            f"{design.shape[1]} coefficients but this unit only offers "
            f"{design.shape[0]} samples — lower `degree`.")
    coef, *_ = np.linalg.lstsq(design, np.asarray(a, dtype=float)[picks].ravel(),
                               rcond=None)
    out = np.zeros((1,) * a.ndim, dtype=float)
    for cf, pw in zip(coef, powers):
        term = np.full((1,) * a.ndim, float(cf))
        for axis, p in enumerate(pw):
            if not p:
                continue
            shape = [1] * a.ndim
            shape[axis] = -1
            term = term * np.power(coords[axis], p).reshape(shape)
        out = out + term
    return np.broadcast_to(out, a.shape).copy()


def _imagej_shrink(r_px: float) -> int:
    """ImageJ ``BackgroundSubtracter``'s shrink-factor table, keyed on the ball radius in
    pixels. Reproduced rather than invented because it is the table two decades of published
    "rolling ball, radius 50" methods sections were actually run with."""
    if r_px <= 10.0:
        return 1
    if r_px <= 30.0:
        return 2
    if r_px <= 100.0:
        return 4
    return 8


def _shrink_factors(radii: Sequence[float], requested: int,
                    method: str) -> Tuple[int, ...]:
    """The per-axis coarsening factors, or all-1 for "estimate at full resolution".

    Clamped per axis so no axis is left with a sub-2-px radius: a 3-plane stack with a 2-px
    axial radius must not be shrunk by 8 in z, which would leave nothing to roll a ball
    under. That clamp is what makes one lateral factor safe to apply to an anisotropic
    volume."""
    if method not in _SHRINK_METHODS or requested == 1 or not radii:
        return tuple(1 for _ in radii)
    base = requested if requested > 1 else _imagej_shrink(max(radii))
    return tuple(max(1, min(base, int(r // 2))) for r in radii)


def _coarse(est: Callable[[np.ndarray, Sequence[float]], np.ndarray], a: np.ndarray,
            radii: Sequence[float], factors: Sequence[int],
            reduce_fn) -> np.ndarray:
    """Estimate the background on a block-reduced copy, then interpolate it back up.

    skimage's own ``rolling_ball`` docstring recommends exactly this ("downscaling-then-
    upscaling to reduce the size of the input processed") because the algorithm is polynomial
    in the radius with degree equal to the dimensionality — so a 3D ball is N**3 and a 2D one
    at radius 120 px took 105 s per 2048^2 plane when measured here. Verified on a synthetic
    plane with a known smooth background (2026-08-05): at radius 30 px, shrink 2 is 9.6x
    faster for a mean error of 0.7% of the shading range; at radius 60 px, shrink 4 is 86x
    faster for 2.5% — smaller, in both cases, than the exact ball's own bias against the true
    surface. Padding uses the reduction's own neutral value so a ragged edge block cannot drag
    the estimate down (a zero-padded min-reduce would put a false trough along two borders)."""
    from scipy import ndimage as ndi
    from skimage.measure import block_reduce
    neutral = float(a.max()) if reduce_fn is np.min else float(a.mean())
    small = block_reduce(a, tuple(factors), reduce_fn, cval=neutral)
    bg = est(small, tuple(max(1.0, r / f) for r, f in zip(radii, factors)))
    zoom = tuple(n / s for n, s in zip(a.shape, small.shape))
    up = ndi.zoom(bg, zoom, order=1, grid_mode=True, mode="nearest")
    if up.shape != a.shape:                    # a rounding sliver, either direction
        up = np.pad(up, [(0, max(0, n - s)) for n, s in zip(a.shape, up.shape)],
                    mode="edge")
    return np.asarray(up[tuple(slice(0, n) for n in a.shape)], dtype=float)


def _background_fn(method: str, *, radii: Tuple[float, ...],
                   sigmas: Tuple[float, ...], pct: float, degree: int, height: float,
                   shrink: int) -> Callable[[np.ndarray], np.ndarray]:
    """Bind one estimator for one unit shape: ``est(a) -> background``, ``a.ndim ==
    len(radii)`` (or the unit's ndim for the kernel-free methods)."""

    def base(x: np.ndarray, rr: Sequence[float]) -> np.ndarray:
        from scipy import ndimage as ndi
        if method in ("rolling_ball", "ellipsoid"):
            return _ball_background(x, rr, height if height > 0.0 else rr[-1])
        if method == "opening":
            return ndi.grey_opening(x, size=_box(rr))
        if method == "minimum":
            return ndi.minimum_filter(x, size=_box(rr))
        if method == "median":
            return ndi.median_filter(x, size=_box(rr))
        if method == "percentile":
            return ndi.percentile_filter(x, pct, size=_box(rr))
        if method == "gaussian":
            return ndi.gaussian_filter(x, sigma=sigmas)
        if method == "polynomial":
            return _poly_background(x, degree)
        return np.full(x.shape, float(np.percentile(x, pct)))     # global_percentile

    factors = _shrink_factors(radii, shrink, method)

    def est(a: np.ndarray) -> np.ndarray:
        if max(factors) > 1:
            return _coarse(base, a, radii, factors,
                           np.mean if method in _MEAN_REDUCED else np.min)
        return base(a, radii)

    return est


# ── the compute ────────────────────────────────────────────────────────────────

def _compute_subtract_background(ctx: EvalContext) -> Dataset:
    """Estimate a background surface from the image's own pixels and remove it.

    Resolved spec (build-node-v2 §0):

    * **Contract** — enhancement, Image -> Image. No axis change, no structure, no new layer.
      The only thing that can change meaning is the intensity scale, and only under
      ``combine="divide"`` (§7c, below).
    * **`method`** picks the ESTIMATOR; ``polarity`` says which way the objects point;
      ``output`` chooses the corrected image or the estimate itself; ``combine`` picks the
      arithmetic. Every socket is gated to the methods that read it, so at most five fields
      are ever live at once.
    * **2D/3D** — a real lever, not a stack-of-2D fallback: every estimator has a genuine
      volumetric form (the ball becomes an anisotropic ellipsoid sized by ``radius`` and
      ``radius_z``; the box filters take a 3-axis box; the Gaussian takes ``sigma_z``; the
      polynomial fits a 3-D total-degree basis; the whole-unit percentile spans the volume).
    * **Footprint** — ``{2D: WHOLE_PLANE, 3D: WHOLE_VOLUME}``, deliberately NOT ``TILEABLE``
      in 2D even though most of the estimators are local. Two of them (``polynomial``,
      ``global_percentile``) are unit-global by construction, and the ``shrink`` path
      block-reduces the whole unit, so a tile's coarse grid would not line up with its
      neighbour's and the seams would show. A background radius is also comparable to the
      field by definition, so a tile plus a radius-sized halo would re-read most of the plane
      anyway — the same reasoning ``analysis.threshold`` records for dropping its shipped
      ``TILEABLE``.

    **The polarity algebra**, which is where the two ImageJ checkboxes live. Everything is
    done in "signal" space, ``s``, which is positive-going in both polarities::

        s      = a               (dark_background)   |   white - a   (light_background)
        bg_s   = estimate(s)
        bg     = bg_s                                |   white - bg_s      <- `output=background`
        subtract        : out_s = clip(s - bg_s, 0)  ; out = out_s | white - out_s
        subtract_signed : out_s = s - bg_s           ; out = out_s | white - out_s
        divide          : out   = a / max(bg, floor)                 (both polarities)

    ``divide`` is deliberately the one that does NOT go through signal space: shading enters
    a transmitted-light image multiplicatively on the RECORDED values, so ``a / bg`` reads as
    "times the local background" either way — above 1 on a fluorescent object, below 1 on a
    cell in brightfield. Inverting first would have flipped the image for one polarity only.

    ``white`` is the light-background inversion point, and it is read from the declared
    intensity scale (``2**bit_depth - 1``) rather than a dtype maximum, because most ND2s here
    are 12-bit and a 16-bit ceiling would put the inversion 16x too high. With no declared
    depth — the state after a percentile ``enhance.normalize`` — it falls back to the unit's
    own maximum, which is ~1.0 on that data and correct (§2, "raw-count consumer reads the
    depth"). Both that read and ``bg_floor``'s happen ONLY on the paths that use them, so a
    plain dark-background subtract is not memo-fenced on ``bit_depth`` it never looked at
    (R1).

    ``presmooth`` reproduces ImageJ's default 3x3 mean: the estimate is made on the smoothed
    copy, the arithmetic is applied to the ORIGINAL pixels. skimage's docstring gives the
    reason directly — the rolling ball is sensitive to salt-and-pepper noise, because one dead
    pixel lets the ball reach the floor under it.

    **The second approach — ``approach="zero_regions"`` (2026-10-02).** Instead of estimating
    a surface and subtracting it from every pixel, DECIDE which pixels are background and set
    them to nothing, leaving object pixels bit-for-bit untouched. ``detector`` picks the
    decision rule:

    * ``sampled_region`` — the user draws one or more regions of pure background (the
      ``shapes`` socket, the same draw tool as ``analysis.roi_mask``); every voxel no brighter
      than ``mean + tolerance * sd`` of the sampled pixels is background. The region is drawn
      in (Y, X) and applies to every plane of a volume; the statistics pool over the whole
      unit's region voxels. An empty region is REFUSED — "similar to nothing" is not a rule.
    * ``adaptive`` — a local cut, ``local_mean + tolerance * noise_sd``: the mean under a
      Gaussian of ``block_size`` (and ``block_size_z`` in 3D), plus a margin of one robust
      noise width for the unit (MAD of the high-pass residual), clamped into
      ``[range_low, range_high]``. The clamp is what stops the interior of an object wider
      than the block from falling below its own local mean and being zeroed.

    Both run in the same signal space as the estimators, so ``polarity`` means the same
    thing: on a light background "nothing" is the white level, and background pixels are set
    to white, not black. ``presmooth`` governs the DECISION only (the mask is computed on the
    smoothed copy, applied to the original pixels), exactly as it governs the estimate above.
    ``output="corrected"`` zeroes the background; ``output="background"`` zeroes everything
    else, so the diagnostic shows precisely the pixels about to be removed, in their own
    units. ``combine`` is not read on this path and is hidden. The intensity scale never
    changes (a pixel is either itself or nothing), so ``bit_depth`` survives on every branch
    of this approach, and the meta_transform says so. Footprint unchanged: both detectors
    need the whole unit (the sample's statistics, the clamp over the unit).
    """
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
    approach = str(modes.get("approach") or "estimate_surface")
    if approach not in _APPROACHES:
        raise ValueError(f"subtract background: unknown approach {approach!r} — one of "
                         f"{list(_APPROACHES)}")
    if approach == "zero_regions":
        return _compute_zero_regions(ctx, ds, modes)
    method = str(modes.get("method") or "rolling_ball")
    if method not in _METHODS:
        raise ValueError(f"subtract background: unknown method {method!r} — one of "
                         f"{list(_METHODS)}")
    output = str(modes.get("output") or "corrected")
    combine = str(modes.get("combine") or "subtract")
    if output == "corrected" and combine not in _COMBINES:
        raise ValueError(f"subtract background: unknown combine {combine!r} — one of "
                         f"{list(_COMBINES)}")
    light = str(modes.get("polarity") or "dark_background") == "light_background"
    vol = ctx.is_volume

    # ── the estimator's own params, resolved ONLY where the method reads them ──
    radii: Tuple[float, ...] = ()
    sigmas: Tuple[float, ...] = ()
    pct = float(ctx.params.get("percentile", 10.0))
    degree = int(ctx.params.get("degree", 2))
    height = float(ctx.params.get("height", 0.0))
    shrink = int(ctx.params.get("shrink", 0))
    if method in _RADIUS_METHODS:
        node = f"subtract background ({method})"
        # a 1-voxel window makes the estimate the image itself — the `tophat` failure mode,
        # and just as silent: `subtract` then returns an empty frame and the user finds out
        # at the next threshold, as "no objects".
        degenerate = ("estimates the image itself as its own background, so the corrected "
                      "output comes out uniformly empty")
        ry = _radius_px(ctx, "radius", _R_DEFAULT)
        _require_window(ctx, _win(ry), ry, node=node, zero_is_identity=False,
                        degenerate=degenerate)
        if vol:
            rz = _radius_z_px(ctx, "radius", _R_DEFAULT)
            _require_window(ctx, _win(rz), rz, node=node, param="radius_z", axis="axial",
                            zero_is_identity=False, degenerate=degenerate)
            radii = (rz, ry, ry)
        else:
            radii = (ry, ry)
    elif method == "gaussian":
        sigma_um = float(ctx.params.get("sigma", 20.0))
        px = ctx.calib("pixel_size_um") or 0.1
        sxy = to_pixels_v2(sigma_um, "um", pixel_size_um=px)
        if sxy <= 0.0:
            # eager, not inside the lazy kernel: under C1 the kernel runs at READ time, so a
            # backend error there is memoized into a poisoned payload (flatten_field's
            # precedent) and surfaces far from this node.
            raise ValueError(
                f"subtract background (gaussian): sigma must be > 0, but {sigma_um:g} um is "
                f"{sxy:.3f} px at pixel_size_um={px:g}. A zero-width background estimate IS "
                f"the image, so the corrected output comes out uniformly empty.")
        sz = to_pixels_v2(float(ctx.params.get("sigma_z", sigma_um)), "um_axial",
                          z_step_um=ctx.calib("z_step_um") or 0.5)
        sigmas = (max(1e-6, sz), sxy, sxy) if vol else (sxy, sxy)
    elif method == "polynomial":
        if not 0 <= degree <= _MAX_DEGREE:
            raise ValueError(
                f"subtract background (polynomial): degree must be 0..{_MAX_DEGREE}, got "
                f"{degree}. Past {_MAX_DEGREE} the fit is ill-conditioned and a surface with "
                f"that many inflections is fitting your objects, not the shading.")
    if method in _PCT_METHODS and not 0.0 <= pct <= 100.0:
        raise ValueError(f"subtract background ({method}): percentile must be 0..100, got "
                         f"{pct:g}.")
    if height < 0.0:
        raise ValueError(f"subtract background (ellipsoid): height must be >= 0 (0 = auto), "
                         f"got {height:g}.")
    if shrink < 0:
        raise ValueError(f"subtract background: shrink must be >= 0 (0 = auto, 1 = exact), "
                         f"got {shrink}.")

    # White level + divide floor: read only where the selected path uses them (R1).
    white = None
    if light:
        bd = ctx.calib("bit_depth")
        white = float(2 ** int(bd) - 1) if bd else None
    floor = (float(ctx.channel(0).param("bg_floor", 1.0))
             if output == "corrected" and combine == "divide" else 0.0)
    presmooth = bool(ctx.params.get("presmooth", True))

    def unit_fn(a: np.ndarray, est: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        # The inversion point is never allowed BELOW a real value: a pixel above the declared
        # white level (a gained or summed image whose depth was not restamped) would otherwise
        # invert to a negative signal, and every estimator here assumes a positive-going one.
        # Costs the ordinary case nothing — data inside its declared depth keeps the declared
        # level, so two units stay comparable.
        mx = max(white, float(a.max())) if white is not None else float(a.max())
        s = (mx - a) if light else a
        bg_s = est(ndi.uniform_filter(s, size=3, mode="nearest") if presmooth else s)
        if output == "background":
            return (mx - bg_s) if light else bg_s
        if combine == "divide":
            bg = (mx - bg_s) if light else bg_s
            return a / np.maximum(bg, floor)
        d = s - bg_s
        if combine == "subtract":
            d = np.clip(d, 0.0, None)
        return (mx - d) if light else d

    def bound(ndim: int) -> Callable[[np.ndarray], np.ndarray]:
        rr = radii if radii else tuple(1.0 for _ in range(ndim))
        est = _background_fn(method, radii=rr, sigmas=sigmas, pct=pct, degree=degree,
                             height=height, shrink=shrink)
        return lambda a: unit_fn(a, est)

    def emit(result: Dataset) -> Dataset:
        # `divide` is the one path whose output is no longer counts — drop bit_depth on the
        # PAYLOAD in lockstep with the meta_transform, which has already dropped it in the
        # envelope (§7c). Synced from the mode state, never re-derived from the depth.
        dimensionless = output == "corrected" and combine == "divide"
        return result.with_metadata(bit_depth=None) if dimensionless else result

    # ONE closure, bound to the resolved unit's dimensionality — `_map_image` branches on the
    # same `ctx.is_volume` on both its lazy and its eager path, so exactly one of the two
    # slots is ever called and binding a 3-axis estimator to a plane is impossible.
    #
    # halo defaults to 0 on purpose: the declared unit is a whole plane / whole volume, so
    # there is nothing lateral outside it to fetch, and this node's true influence radius IS
    # the unit (`polynomial` and `global_percentile` are unit-global) — a finite halo would
    # understate it for the next node's tile sizing.
    unit = bound(3 if vol else 2)
    return emit(_map_image(ctx, ds, plane_fn=unit, volume_fn=unit if vol else None))


def _compute_zero_regions(ctx: EvalContext, ds: Dataset, modes) -> Dataset:
    """The ``approach="zero_regions"`` half of :func:`_compute_subtract_background`: decide
    which voxels are background, then set them to nothing. See that docstring for the spec;
    this function is the branch, split out so each approach reads its own params only (R1:
    a plain surface subtract must not be memo-fenced on a ``shapes`` string it never read)."""
    from scipy import ndimage as ndi
    detector = str(modes.get("detector") or "sampled_region")
    if detector not in _DETECTORS:
        raise ValueError(f"subtract background: unknown detector {detector!r} — one of "
                         f"{list(_DETECTORS)}")
    output = str(modes.get("output") or "corrected")
    light = str(modes.get("polarity") or "dark_background") == "light_background"
    vol = ctx.is_volume
    tolerance = float(ctx.params.get("tolerance", 3.0))
    if tolerance < 0.0:
        raise ValueError(f"subtract background ({detector}): tolerance must be >= 0 standard "
                         f"deviations, got {tolerance:g}.")
    presmooth = bool(ctx.params.get("presmooth", True))
    white = None
    if light:
        bd = ctx.calib("bit_depth")
        white = float(2 ** int(bd) - 1) if bd else None

    region2d = None
    sigmas: Tuple[float, ...] = ()
    low = high = 0.0
    if detector == "sampled_region":
        from nodegraph.kernels.dic_mesh_region import build_roi_mask, has_region
        shapes = _region_shapes(ctx.params.get("shapes"))
        if not has_region(shapes):
            raise ValueError(
                "subtract background (sampled region): no background region is drawn. Use the "
                "Pick tool on the `Background sample` socket to outline one or more patches of "
                "pure background — the node zeroes everything that looks like them. With "
                "nothing drawn there is nothing to compare against, so refusing is the only "
                "honest answer (the ROI Mask node's whole-frame default would mean 'everything "
                "is background').")
        ax = ds.image.axes
        region2d = np.asarray(build_roi_mask(shapes, ax.y, ax.x), dtype=bool)
        if int(region2d.sum()) < _MIN_SAMPLE:
            raise ValueError(
                f"subtract background (sampled region): the drawn region covers "
                f"{int(region2d.sum())} pixel(s); at least {_MIN_SAMPLE} are needed to "
                "estimate a spread. Draw a larger patch.")
    else:
        node = "subtract background (adaptive)"
        degenerate = ("compares every pixel to itself, so the local cut equals the pixel and "
                      "every pixel is zeroed")
        ry = _radius_px(ctx, "block_size", _R_DEFAULT)
        _require_window(ctx, _win(ry), ry, node=node, zero_is_identity=False,
                        degenerate=degenerate, param="block_size")
        if vol:
            rz = _radius_z_px(ctx, "block_size", _R_DEFAULT)
            _require_window(ctx, _win(rz), rz, node=node, param="block_size_z", axis="axial",
                            zero_is_identity=False, degenerate=degenerate)
            sigmas = (max(1e-6, rz / 2.0), ry / 2.0, ry / 2.0)
        else:
            sigmas = (ry / 2.0, ry / 2.0)
        low = float(ctx.params.get("range_low", 0.0))
        high = float(ctx.params.get("range_high", 0.0))
        if low < 0.0 or (high > 0.0 and high < low):
            raise ValueError(
                f"subtract background (adaptive): the clamp range must satisfy 0 <= low <= "
                f"high (or high = 0 for no upper limit), got low={low:g}, high={high:g}.")

    def unit_fn(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        mx = max(white, float(a.max())) if white is not None else float(a.max())
        s = (mx - a) if light else a
        s_est = ndi.uniform_filter(s, size=3, mode="nearest") if presmooth else s
        if detector == "sampled_region":
            bg = _sampled_background(s_est, region2d, tolerance)
        else:
            # the clamp is typed in RECORDED units; in signal space a light-background
            # range flips and reflects about the white level
            if light:
                s_low = (mx - high) if high > 0.0 else 0.0
                s_high = mx - low
            else:
                s_low, s_high = low, high
            bg = _adaptive_background(s_est, sigmas, tolerance, s_low, s_high)
        keep = ~bg if output == "corrected" else bg
        out_s = np.where(keep, s, 0.0)
        return (mx - out_s) if light else out_s

    return _map_image(ctx, ds, plane_fn=unit_fn, volume_fn=unit_fn if vol else None)


register_node(
    _compute_subtract_background, op_key="enhance.subtract_background",
    label="Subtract Background", category="enhancement",
    inputs=[
        InDataset(),
        InFloat("radius", "Background radius", unit="um", field=False,
                default=_R_DEFAULT, pick_kind="radius",
                available_in={"approach": _SURFACE, "method": _RADIUS_METHODS},
                description=
                "The background scale in MICRONS — the one knob that decides what counts as "
                "background. Structure LARGER than this survives in the estimate and is "
                "removed; anything smaller is kept. It must therefore be comfortably bigger "
                "than the objects you are measuring: set it near cell size and each cell "
                "becomes its own background and subtracts itself, leaving hollow rings and "
                "reported intensities far below the truth. A few times the largest object "
                "diameter is the working range. Cost grows steeply with it for the ball, "
                "median and percentile estimators, which is what `Shrink factor` is for. "
                "Not offered for the Gaussian, polynomial and whole-unit-percentile "
                "methods, which carry their own scale control or none at all."),
        InFloat("radius_z", "Background radius Z", unit="um_axial", field=False,
                default=_R_DEFAULT,
                available_in={"approach": _SURFACE, "dim": frozenset({"3D"}),
                              "method": _RADIUS_METHODS},
                description=_AXIAL_TWIN_DOC.format(lateral="Background radius")),
        InFloat("sigma", "Background sigma", unit="um", field=False, default=20.0,
                pick_kind="radius",
                available_in={"approach": _SURFACE, "method": frozenset({"gaussian"})},
                description=
                "Width of the Gaussian background estimate, in microns. Plays the same role "
                "as `Background radius` but for a smooth blur rather than a kernel, so the "
                "same rule applies: keep it well above object size or the blur follows the "
                "objects and subtracts them. Roughly comparable to a ball radius of 1.5-2x "
                "its value. Matches `enhance.flatten_field`'s default of 20 um, so the two "
                "nodes agree when set the same. Only read by the `gaussian` method."),
        InFloat("sigma_z", "Background sigma Z", unit="um_axial", field=False, default=20.0,
                available_in={"approach": _SURFACE, "dim": frozenset({"3D"}),
                              "method": frozenset({"gaussian"})},
                description=_AXIAL_TWIN_DOC.format(lateral="Background sigma")),
        InFloat("percentile", "Percentile", unit="", field=False, default=10.0,
                available_in={"approach": _SURFACE, "method": _PCT_METHODS},
                description=
                "Which point in the local intensity distribution is taken as the background, "
                "0-100. LOWER is a darker, more conservative estimate that removes less and "
                "leaves more of the object intensity intact; HIGHER removes more and starts "
                "eating into the objects, so it lowers every intensity a downstream table "
                "reports. 0 is the minimum (identical to the `minimum` method, and as "
                "noise-sensitive), 50 is the median. Around 10 is a good compromise on data "
                "where objects cover a modest fraction of the field; raise it toward 25-50 "
                "when the field is crowded. Read by both `percentile` (in a box around each "
                "voxel) and `global_percentile` (once over the whole plane or volume) — for "
                "the latter, a low value here is the usual way to strip a flat camera "
                "offset."),
        InInt("degree", "Polynomial degree", unit="", field=False, default=2,
                available_in={"approach": _SURFACE, "method": frozenset({"polynomial"})},
                description=
                "Total degree of the fitted surface. 0 subtracts a single mean level, 1 a "
                "tilted plane, 2 the saddle/dome that models ordinary vignetting — 2 is the "
                "usual answer and the default. HIGHER follows the shading more closely but "
                "starts bending to fit the objects, which removes real signal and biases "
                "reported intensities down; because the fit is global, that damage is spread "
                "over the whole field rather than being visible as a halo. Capped at 6, "
                "beyond which the fit is ill-conditioned. Unlike every other method this one "
                "has no length scale at all, which is exactly why it cannot subtract a cell: "
                "a low-degree surface has nowhere to put one. Only read by `polynomial`."),
        InFloat("height", "Ball height", unit="", field=False, default=0.0,
                available_in={"approach": _SURFACE, "method": frozenset({"ellipsoid"})},
                description=
                "How far the ball may curve in INTENSITY, in the image's own units — the "
                "control `rolling_ball` does not expose. 0 means auto, which reproduces "
                "skimage's and ImageJ's coupling of intensity extent to the pixel radius; "
                "that coupling is arbitrary, since counts and pixels are unrelated "
                "quantities, and it is why the same ball radius behaves differently on a "
                "12-bit and a normalized image. Set it explicitly and the two are "
                "independent: SMALLER makes a stiffer ball that rides higher and removes "
                "more (a flatter background), LARGER lets it sag into intensity valleys and "
                "removes less. Start near the peak-to-trough range of the shading you can "
                "see. Only read by `ellipsoid`."),
        InInt("shrink", "Shrink factor", unit="", field=False, default=0,
                available_in={"approach": _SURFACE, "method": _SHRINK_METHODS},
                description=
                "Estimate the background on a coarsened copy, then interpolate it back up — "
                "the speed control. 0 is auto (ImageJ's own table: 1 up to a 10 px radius, "
                "then 2, 4 and 8 past 30 and 100 px), 1 forces the exact full-resolution "
                "estimate, and a larger number coarsens harder. It matters more than it "
                "sounds: the rolling ball is polynomial in the radius with degree equal to "
                "the dimensionality, so measured here one 2048-square plane took 8.7 s at a "
                "29 px radius and 105 s at 120 px, and a 3D volume is a further power worse. "
                "Auto is 10-86x faster for a mean error under 3% of the shading range — "
                "smaller than the exact ball's own bias — so raise it if a preview is slow "
                "and drop it to 1 only when you need bit-exact agreement with ImageJ or "
                "skimage. Shown only for the four estimators whose cost grows with the "
                "radius; the box minimum and opening are already radius-independent in scipy "
                "and are always exact."),
        InBool("presmooth", "Pre-smooth", field=False, default=True,
                description=
                "Estimate the background from a 3x3 mean-smoothed copy instead of the raw "
                "pixels — ImageJ's default, which its dialog exposes as the inverse checkbox "
                "'Disable smoothing'. It changes only the ESTIMATE: the arithmetic is always "
                "applied to your original pixels, so no smoothing reaches the output. Leave "
                "it on. The reason is in skimage's own docstring — the rolling ball is "
                "sensitive to salt-and-pepper noise, because a single dead pixel lets the "
                "ball reach the floor underneath it and punches a pit in the background that "
                "then gets added back to a whole ball-sized neighbourhood. Turn it off only "
                "to match an ImageJ run that had smoothing disabled, or when the data is "
                "already denoised and you want the estimate to follow a genuinely sharp "
                "shading edge. Barely matters for the polynomial and whole-unit-percentile "
                "methods, which average over far more pixels than a 3x3 window."),
        InFloat("bg_floor", "Background floor", unit="", field=False, default=1.0,
                derive="1.0 if bit_depth else 1e-06",
                available_in={"approach": _SURFACE, "combine": frozenset({"divide"}),
                              "output": frozenset({"corrected"})},
                description=
                "Smallest value the background may take before it is used as a divisor — it "
                "stops a genuinely black region from producing a division by ~0 and a wall of "
                "huge values. Auto resolves it from the declared intensity scale: one raw "
                "count on integer data, or 1e-6 when no bit depth is declared, which is the "
                "state after a percentile Normalize. That second case is why it is not simply "
                "fixed at 1.0 — on [0,1] data a floor of 1.0 exceeds every value in the "
                "image, so the division would quietly return the input unchanged. Raise it to "
                "clamp harder in dark regions; it has no effect where the background is well "
                "above it. Only read by the `divide` arithmetic."),
        # ── the zero-regions approach's own sockets ──
        InString("shapes", "Background sample", field=False, default="",
                 pick_kind="shapes",
                 available_in={"approach": _ZERO, "detector": _SAMPLED},
                 description=
                 "One or more patches of PURE background, drawn on the viewer with the Pick "
                 "button (rectangle / ellipse / polygon / freehand, with Add and Cut). The "
                 "node measures the intensity distribution under what you draw and zeroes "
                 "every pixel in the frame that is no brighter than it — so the patch should "
                 "be representative, not tiny: a few hundred pixels spanning the dim and the "
                 "brighter parts of the empty field. Drawing over an object widens the "
                 "spread and makes the cut more aggressive, eating dim objects. The region is "
                 "drawn in (Y, X) and is applied to every Z plane of a volume, and the "
                 "statistics are pooled over the whole plane or volume. EMPTY is REFUSED "
                 "rather than defaulting to the whole frame: with no sample there is nothing "
                 "to be similar to. Only read by the `sampled_region` detector."),
        InFloat("tolerance", "Tolerance", unit="", field=False, default=3.0,
                available_in={"approach": _ZERO},
                description=
                "How many noise widths above the background's mean still count as "
                "background. For `sampled_region` the mean and spread are those of the pixels "
                "you drew; for `adaptive` the mean is each pixel's local mean under the block "
                "and the spread is one robust noise estimate for the whole plane or volume. "
                "HIGHER zeroes more — it reaches into the dim tail of real objects and "
                "lowers their reported areas — and LOWER leaves speckles of unzeroed noise in "
                "the field. 3 is the natural starting point: on Gaussian noise it keeps "
                "99.7 percent of genuine background pixels on the background side, so the "
                "field comes out clean while anything a few noise widths above it survives. "
                "Drop toward 2 when objects are faint; raise toward 4 to 5 on noisy cameras."),
        InFloat("block_size", "Block size", unit="um", field=False, default=_R_DEFAULT,
                pick_kind="radius",
                available_in={"approach": _ZERO, "detector": _ADAPTIVE},
                description=
                "Radius in MICRONS of the neighbourhood whose local mean sets the cut at each "
                "pixel (the Gaussian's sigma is half of it, so the window spans "
                "about this far). The same rule as `Background radius`: keep it well above "
                "object size, or a cell becomes its own neighbourhood, its local mean rises "
                "to its own brightness, and its interior is zeroed from the inside out. The "
                "`Upper limit` clamp exists for exactly that case, but the right block size "
                "makes it unnecessary. Only read by the `adaptive` detector."),
        InFloat("block_size_z", "Block size Z", unit="um_axial", field=False,
                default=_R_DEFAULT,
                available_in={"approach": _ZERO, "detector": _ADAPTIVE,
                              "dim": frozenset({"3D"})},
                description=_AXIAL_TWIN_DOC.format(lateral="Block size")),
        InFloat("range_low", "Lower limit", unit="", field=False, default=0.0,
                available_in={"approach": _ZERO, "detector": _ADAPTIVE},
                description=
                "Floor for the local cut, in the image's own intensity units — the local "
                "threshold is never allowed below this. 0 (the default) imposes nothing. "
                "Raise it to guarantee that everything under a known intensity is zeroed even "
                "where the field is so flat and clean that the local spread collapses and the "
                "adaptive cut would otherwise leave a faint pedestal in place. Typed in "
                "recorded units in both polarities (for a light background, 'below the cut' "
                "means closer to white). Only read by the `adaptive` detector."),
        InFloat("range_high", "Upper limit", unit="", field=False, default=0.0,
                available_in={"approach": _ZERO, "detector": _ADAPTIVE},
                description=
                "Ceiling for the local cut, in the image's own intensity units — the local "
                "threshold is never allowed above this, so no pixel brighter than it can ever "
                "be zeroed. 0 (the default) means no ceiling. This is the control that makes "
                "adaptive thresholding safe on large bright objects: without it, a cell wider "
                "than the block has a local mean near its own brightness and its interior falls "
                "below the cut. Set it a little under the dimmest object you want to keep and "
                "the object is protected whatever the block size does. Must be at or above "
                "the lower limit when both are set. Only read by the `adaptive` detector."),
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("approach", list(_APPROACHES), default="estimate_surface", label="Approach",
             description=
             "What 'subtract background' means for this node. The two options are different "
             "operations, not two settings of one: the first changes every pixel by a smooth "
             "amount, the second changes a chosen set of pixels completely and leaves the rest "
             "exactly as they were. Each shows only its own controls.",
             choice_docs={
                 "estimate_surface":
                     "Fit a smooth background surface from the image's own pixels with the "
                     "chosen `Estimator`, then remove it with the chosen `Arithmetic` — "
                     "ImageJ's `Subtract Background`, and the default. Every pixel moves, by "
                     "the local background amount; objects keep their shape and lose their "
                     "pedestal. The right tool for photometry and for uneven illumination.",
                 "zero_regions":
                     "Decide which pixels ARE background and set them to nothing (0 on a dark "
                     "background, the white level on a light one), touching no other pixel. "
                     "Objects keep their exact recorded intensities, so a downstream mean over "
                     "an object is unchanged, while the field becomes exactly empty. The right "
                     "tool for cleaning a field before display, for a hard mask of 'not "
                     "background', or when you can show the node what background looks like.",
             }),
        Mode("detector", list(_DETECTORS), default="sampled_region", label="Detector",
             available_in={"approach": _ZERO},
             description=
             "How the `zero_regions` approach decides what is background. One rule is taught "
             "by example, the other is computed from each pixel's own neighbourhood; both end "
             "in a cut that `Tolerance` moves.",
             choice_docs={
                 "sampled_region":
                     "You draw one or more patches of pure background (`Background sample`); "
                     "every pixel in the frame no brighter than that sample's mean plus "
                     "`Tolerance` spreads is zeroed. One global cut per plane or volume, so it "
                     "assumes the background is roughly uniform across the field — under "
                     "strong shading, flatten the image first or use `adaptive`. Needs the "
                     "drawn region; nothing drawn is refused.",
                 "adaptive":
                     "A local cut at every pixel: its neighbourhood's mean under a Gaussian of "
                     "`Block size`, plus `Tolerance` noise widths (one robust estimate per "
                     "plane or volume), clamped between `Lower limit` and `Upper limit`. "
                     "Follows uneven illumination without any "
                     "drawing, at the cost of the classic adaptive failure inside objects "
                     "wider than the block — which the upper limit is there to prevent. Needs "
                     "no input from you beyond a block size above object size.",
             }),
        Mode("method", list(_METHODS), default="rolling_ball", label="Estimator",
             available_in={"approach": _SURFACE},
             description=
             "How the background surface is estimated. Every option answers the same "
             "question — what does this image look like with the objects taken out — and they "
             "differ in what they assume about the shading and in what they cost. The first "
             "six read `Background radius`; `gaussian` has its own sigma; the last two have "
             "no length scale at all.",
             choice_docs={
                 "rolling_ball":
                     "Sternberg's rolling ball, ImageJ's `Subtract Background` and the name "
                     "most protocols cite. A ball of the given radius is rolled underneath "
                     "the intensity landscape and its apex traces the background, which "
                     "follows curved shading far better than a flat morphological estimate. "
                     "The most expensive option by a wide margin — cost is polynomial in the "
                     "radius, degree equal to the dimensionality — so it leans on `Shrink "
                     "factor`. In 3D the ball becomes an ellipsoid sized by both radii, "
                     "since voxels are anisotropic.",
                 "ellipsoid":
                     "The same rolling ball with its INTENSITY extent decoupled from its "
                     "pixel radius, via `Ball height`. Use it when the ball behaves "
                     "differently than expected after a rescale or on a different bit depth: "
                     "`rolling_ball` inherits skimage's and ImageJ's convention of tying the "
                     "intensity semi-axis to the radius in pixels, and those two quantities "
                     "have nothing to do with each other. Identical to `rolling_ball` when "
                     "the height is left at auto.",
                 "opening":
                     "A morphological grey opening (erode then dilate) in a box of the given "
                     "radius — the estimate `enhance.tophat` subtracts, exposed so you can "
                     "look at it and pick the arithmetic. Fast and radius-independent in "
                     "scipy, but it produces a piecewise-flat surface rather than a smooth "
                     "one, so under strong curved shading it leaves visible terracing where "
                     "the ball would not.",
                 "minimum":
                     "The local minimum in a box — the crudest and cheapest estimate, and the "
                     "most aggressive: it sits at the darkest pixel of every neighbourhood, "
                     "so it removes the noise floor along with the background and biases "
                     "reported intensities down. Also the most noise-sensitive, since one "
                     "dead pixel drags a whole box down. Reach for it when speed matters more "
                     "than photometry, or as the seed for a chained smooth (`minimum` then "
                     "`enhance.gaussian`) which is how CellProfiler builds its background.",
                 "median":
                     "The local median in a box: robust to bright objects in a way that the "
                     "mean and the Gaussian are not, because an object has to fill more than "
                     "half the box before it can move the estimate at all. The best choice "
                     "when objects are large and sparse and a Gaussian visibly bulges under "
                     "them. Slow — measured at 57 s per 1024-square plane on a 59 px box — so "
                     "leave `Shrink factor` on auto.",
                 "percentile":
                     "The local N-th percentile in a box, the knob that spans `minimum` (0) "
                     "and `median` (50). This is the one to use when the field is crowded "
                     "enough that even the median is pulled up by objects: drop the "
                     "percentile until the estimate sits in the gaps between them. Same cost "
                     "class as `median`.",
                 "gaussian":
                     "A large-sigma Gaussian blur — the smoothest estimate and the cheapest "
                     "of the ones that follow curvature, since scipy runs it separably. It "
                     "is a weighted MEAN, though, so bright objects pull it up and it "
                     "subtracts a shadow of them (a halo, and dimmer objects); the "
                     "`median`/`percentile` options exist for exactly that failure. Matches "
                     "`enhance.flatten_field`, which is the node to use instead if you want "
                     "the estimate averaged over time or the frame's mean brightness "
                     "restored.",
                 "polynomial":
                     "A least-squares polynomial surface of the chosen degree, fitted over "
                     "the whole plane or volume. The only option with no length scale, which "
                     "makes it the only one that CANNOT subtract an object: a degree-2 "
                     "surface has nowhere to put a cell. That is what makes it the safest "
                     "choice for photometry on a crowded field, and useless for anything but "
                     "smooth, global illumination or a tilted stage. Very fast — the fit "
                     "samples a strided subset and is evaluated in closed form.",
                 "global_percentile":
                     "One scalar for the entire plane or volume: its N-th percentile, "
                     "subtracted everywhere. No spatial model at all, so it corrects a flat "
                     "pedestal — a camera offset, a stray-light floor, a fluorescent medium — "
                     "and nothing else. The right answer when the shading is genuinely "
                     "uniform, and the honest one when you do not want an estimator inventing "
                     "structure it cannot verify. Costs one sort.",
             }),
        Mode("polarity", ["dark_background", "light_background"],
             default="dark_background", label="Polarity",
             description=
             "Which way the objects point, i.e. ImageJ's 'Light background' checkbox. Every "
             "estimator here goes UNDER a positive-going signal, so on the wrong setting it "
             "estimates the objects instead of the background and returns a nearly empty "
             "frame rather than an inverted one. This is not a display preference — it is a "
             "property of your contrast mode.",
             choice_docs={
                 "dark_background":
                     "Bright objects on a dark field: fluorescence, and the default. The "
                     "background is estimated from the pixels as recorded and subtracted "
                     "directly.",
                 "light_background":
                     "Dark objects on a bright field: brightfield, phase, DIC, stained "
                     "histology, a scanned gel. The image is inverted about its white level, "
                     "the background estimated there, and the result put back — so the output "
                     "keeps a light background with dark objects, exactly as ImageJ does it. "
                     "The white level comes from the declared bit depth (most ND2s here are "
                     "12-bit, not 16), falling back to each unit's own maximum when no depth "
                     "is declared.",
             }),
        Mode("output", ["corrected", "background"], default="corrected", label="Output",
             description=
             "Whether the node emits the corrected image or the background it estimated — "
             "ImageJ's 'Create background (don't subtract)'. The estimate is worth looking at "
             "before you trust any of this: it is the one image that shows whether the radius "
             "is small enough to be tracing your objects.",
             choice_docs={
                 "corrected":
                     "The image with the background removed — by the `Arithmetic` below under "
                     "`estimate_surface`, or with the detected background pixels set to nothing "
                     "under `zero_regions`. What you want downstream, and the default.",
                 "background":
                     "The background itself, in the input's own intensity units — the "
                     "diagnostic. Under `estimate_surface` it is the estimated surface: objects "
                     "visible in it are objects about to be subtracted from themselves, which "
                     "means the radius is too small. Under `zero_regions` it is the complement "
                     "of the corrected image — only the pixels about to be zeroed, everything "
                     "else blank — so you can see exactly what the detector decided. Either way "
                     "it is the one check worth making before trusting any number downstream. "
                     "Wire it to a Viewer, or to `io.write_tiff` to keep it beside the run. "
                     "The `Arithmetic` control is hidden on this output because nothing reads "
                     "it, and the intensity scale is unchanged whatever it was set to.",
             }),
        Mode("combine", list(_COMBINES), default="subtract", label="Arithmetic",
             available_in={"approach": _SURFACE, "output": frozenset({"corrected"})},
             description=
             "How the estimate is removed. The real question is whether your shading is an "
             "additive offset (stray light, camera bias, autofluorescent medium) or a "
             "multiplicative gain (uneven excitation, vignetting, an uneven condenser) — a "
             "property of the optics, not a preference. Only `divide` leaves the input's "
             "count scale, and it drops the declared bit depth accordingly. Hidden when the "
             "output is the background itself, which reads none of these.",
             choice_docs={
                 "subtract":
                     "`clip(signal - background, 0)` — ImageJ's behaviour and the default. "
                     "Additive model. Anything dimmer than its local background becomes "
                     "exactly 0, so the dark end is clipped and unrecoverable; that is fine "
                     "for segmentation and it biases a mean-intensity measurement upward, "
                     "because the noise that would have gone negative is folded back to zero.",
                 "subtract_signed":
                     "The same subtraction with the clip REMOVED, so background noise stays "
                     "symmetric about 0 and negative values survive. The one to use when you "
                     "are going to measure intensities: an unbiased mean needs the negative "
                     "half of the noise. Some downstream nodes assume non-negative counts, "
                     "and a viewer will show the negatives as black.",
                 "divide":
                     "`image / background` — the multiplicative model, and the correct one "
                     "for vignetting, uneven excitation and brightfield transmission. Output "
                     "is dimensionless: about 1 in empty background, above 1 on a fluorescent "
                     "object, below 1 on an absorbing one, and comparable across frames and "
                     "fields. No longer counts, so the declared bit depth is dropped and a "
                     "downstream raw-count consumer will refuse it. `Background floor` guards "
                     "the divisor.",
             }),
    ],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    meta_transform=_meta_subtract_background,
    description="Estimate a background surface and remove it — rolling ball (Sternberg / "
                "ImageJ), tunable ellipsoid, morphological opening, box minimum / median / "
                "percentile, large-sigma Gaussian, least-squares polynomial surface, or one "
                "whole-unit percentile. Dark or light background, subtract / signed subtract "
                "/ divide, or output the estimate itself. Auto-shrinks the expensive "
                "estimators (ImageJ's factor table); `divide` drops bit_depth. Or, as a "
                "second approach, DECIDE which pixels are background — by similarity to a "
                "drawn sample, or by an adaptive local cut clamped to a range — and set "
                "exactly those to nothing, leaving every object pixel untouched.")
