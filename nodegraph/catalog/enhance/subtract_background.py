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
                                Mode, OutDataset)

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
    """
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
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


register_node(
    _compute_subtract_background, op_key="enhance.subtract_background",
    label="Subtract Background", category="enhancement",
    inputs=[
        InDataset(),
        InFloat("radius", "Background radius", unit="um", field=False,
                default=_R_DEFAULT, pick_kind="radius",
                available_in={"method": _RADIUS_METHODS},
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
                available_in={"dim": frozenset({"3D"}), "method": _RADIUS_METHODS},
                description=_AXIAL_TWIN_DOC.format(lateral="Background radius")),
        InFloat("sigma", "Background sigma", unit="um", field=False, default=20.0,
                pick_kind="radius", available_in={"method": frozenset({"gaussian"})},
                description=
                "Width of the Gaussian background estimate, in microns. Plays the same role "
                "as `Background radius` but for a smooth blur rather than a kernel, so the "
                "same rule applies: keep it well above object size or the blur follows the "
                "objects and subtracts them. Roughly comparable to a ball radius of 1.5-2x "
                "its value. Matches `enhance.flatten_field`'s default of 20 um, so the two "
                "nodes agree when set the same. Only read by the `gaussian` method."),
        InFloat("sigma_z", "Background sigma Z", unit="um_axial", field=False, default=20.0,
                available_in={"dim": frozenset({"3D"}),
                              "method": frozenset({"gaussian"})},
                description=_AXIAL_TWIN_DOC.format(lateral="Background sigma")),
        InFloat("percentile", "Percentile", unit="", field=False, default=10.0,
                available_in={"method": _PCT_METHODS},
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
                available_in={"method": frozenset({"polynomial"})},
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
                available_in={"method": frozenset({"ellipsoid"})},
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
                available_in={"method": _SHRINK_METHODS},
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
                available_in={"combine": frozenset({"divide"}),
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
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("method", list(_METHODS), default="rolling_ball", label="Estimator",
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
                     "The image with the background removed, by the `Arithmetic` below. What "
                     "you want downstream, and the default.",
                 "background":
                     "The estimated background surface itself, in the input's own intensity "
                     "units — the diagnostic. Objects visible in it are objects about to be "
                     "subtracted from themselves, which means the radius is too small; that "
                     "is the one check worth making before trusting any number downstream, "
                     "and it is far easier to see here than in the corrected image. Wire it "
                     "to a Viewer, or to `io.write_tiff` to keep it beside the run. The "
                     "`Arithmetic` control is hidden on this output because nothing reads it, "
                     "and the intensity scale is unchanged whatever it was set to.",
             }),
        Mode("combine", list(_COMBINES), default="subtract", label="Arithmetic",
             available_in={"output": frozenset({"corrected"})},
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
                "estimators (ImageJ's factor table); `divide` drops bit_depth.")
