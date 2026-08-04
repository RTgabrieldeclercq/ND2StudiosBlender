"""Temporal Gain (``enhance.temporal_gain``) — Correct intensity drift along T by a per-frame gain: reference series_mean (Cell-Tracker Bleach Correction) / rolling_mean (its Temporal Fold Correction) / exponential_fit…"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Temporal Gain (CT Bleach Correction + Temporal Fold Correction) ─────────────

#: ``reference`` → what each frame's brightness is compared against.
_TEMPORAL_REFERENCES = ("series_mean", "rolling_mean", "exponential_fit")
#: ``extent`` → the spatial granularity the comparison is made at.
_TEMPORAL_EXTENTS = ("global", "tile", "gaussian")
def _rolling_mean_t(signal: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean along axis 0 with edge padding.

    Cell-Tracker spells this as ``np.pad(mode="edge")`` → ``np.convolve(ones/w, "same")``
    → slice back, which for an **odd** window is exactly
    ``uniform_filter1d(..., mode="nearest")`` — verified index by index, including both
    ends, where the padding is what keeps the first and last frames from being pulled
    toward zero. Using the scipy call directly keeps it one pass and vectorized over
    whatever trailing axes ``signal`` carries (a scalar per frame, a tile grid, or a full
    pixel grid). The caller forces the window odd, since a centred window is only
    symmetric for an odd size.

    A float32 input stays float32 (see :func:`_exp_fit_t` for why that matters here)."""
    from scipy.ndimage import uniform_filter1d
    sig = np.asarray(signal)
    if sig.dtype.kind != "f":
        sig = sig.astype(float)
    return uniform_filter1d(sig, size=int(window), axis=0, mode="nearest")
def _exp_fit_t(signal: np.ndarray, eps: float) -> np.ndarray:
    """Least-squares exponential ``A·e^{k t}`` fitted along axis 0, evaluated at every t.

    Fitted in log space (``log y = log A + k·t``), which makes it the closed-form linear
    least squares — one pass of sums over T, no iteration, and vectorized across any
    trailing axes. Non-positive samples have no logarithm, so they are floored at ``eps``;
    a flat signal fits ``k=0``, which yields a constant curve and therefore a gain of
    exactly 1 rather than a division by zero.

    A float32 input stays float32 (rather than being widened to float64), because the
    ``gaussian`` extent's signal is a whole ``(T, Y, X)`` series and the working copies
    dominate that node's memory. A multiplicative gain near 1 needs nothing like float64."""
    y = np.asarray(signal)
    if y.dtype.kind != "f":
        y = y.astype(float)
    n = y.shape[0]
    if n < 2:
        return y.copy()
    # the frame axis takes the SIGNAL's float width, not float64 — otherwise every
    # downstream product is silently widened and the dtype economy above is undone
    t = np.arange(n, dtype=y.dtype)
    ly = np.log(np.maximum(y, eps))
    tm = float(t.mean())
    col = (n,) + (1,) * (y.ndim - 1)                 # broadcast t along the trailing axes
    dt = (t - tm).reshape(col)
    k = (dt * (ly - ly.mean(axis=0, keepdims=True))).sum(axis=0) / float(((t - tm) ** 2).sum())
    log_a = ly.mean(axis=0) - k * tm
    return np.exp(log_a[None, ...] + k[None, ...] * t.reshape(col))
def _temporal_gain_field(signal: np.ndarray, reference: str, window: int,
                         threshold: float) -> np.ndarray:
    """The per-frame multiplicative **gain** for one ``(T, ...)`` brightness signal.

    ``signal`` is the observed brightness per frame — a scalar under ``global``, a coarse
    grid under ``tile``, a full plane under ``gaussian`` — and the result has the same
    shape. The gain is ``target / observed``:

    * ``series_mean`` — target is the whole series' own mean. Cell-Tracker's **Bleach
      Correction** (``normalize_timeseries``: ``f_t · grand_mean / mean_t``): every frame
      is pulled onto one common level, so a monotone bleaching decay is flattened out.
    * ``rolling_mean`` — target is a centred ``window``-frame rolling mean. Cell-Tracker's
      **Temporal Fold Correction**: a frame is compared with its NEIGHBOURS, so a slow
      trend survives untouched and only a locally aberrant frame is pulled back.
    * ``exponential_fit`` — target is the mean of a least-squares exponential through the
      signal, and the *observed* value is taken from the fitted curve rather than the noisy
      sample, so per-frame shot noise is not injected into the correction. This is the
      decay model Cell-Tracker's Bleach Correction offered in its UI and never implemented
      — its ``method`` choice was inert, both values ran the ratio path — so it is
      implemented here rather than ported as a dead control.

    ``threshold`` is a dead band: a gain within ``|g - 1| <= threshold`` is set to exactly
    1, leaving that sample untouched. Cell-Tracker applies it to the fold correction only;
    here it is uniform and the caller passes 0 for the two whole-series references, which
    reproduces both behaviours.

    A float32 input stays float32 throughout (see :func:`_exp_fit_t`) — the ``gaussian``
    extent's signal is a whole ``(T, Y, X)`` series, so widening it here would triple this
    node's peak memory for no gain the arithmetic can use."""
    obs = np.asarray(signal)
    if obs.dtype.kind != "f":
        obs = obs.astype(float)
    mx = float(np.max(obs)) if obs.size else 0.0
    # relative, not Cell-Tracker's absolute 1e-10: this must also work on [0,1] floats,
    # where an absolute floor is either irrelevant or the whole signal.
    eps = max(1e-12, 1e-9 * mx)
    obs = np.maximum(obs, eps)
    if reference == "series_mean":
        target, observed = obs.mean(axis=0, keepdims=True), obs
    elif reference == "rolling_mean":
        target, observed = _rolling_mean_t(obs, window), obs
    else:
        fit = np.maximum(_exp_fit_t(obs, eps), eps)
        target, observed = fit.mean(axis=0, keepdims=True), fit
    gain = np.broadcast_to(target, obs.shape) / observed
    if threshold > 0.0:
        gain = np.where(np.abs(gain - 1.0) > threshold, gain, 1.0)
    return np.asarray(gain)
def _tile_means(plane: np.ndarray, tile: int) -> np.ndarray:
    """Block means of a plane on a ``tile``×``tile`` grid — Cell-Tracker's ``local-tile``
    extent. One ``np.add.reduceat`` pass regardless of block count, and a ragged trailing
    block is averaged over the pixels it actually has rather than being zero-padded, which
    would drag the frame border's mean down and over-correct it."""
    f = np.asarray(plane, dtype=float)
    h, w = f.shape
    ys = np.arange(0, h, tile)
    xs = np.arange(0, w, tile)
    sums = np.add.reduceat(np.add.reduceat(f, ys, axis=0), xs, axis=1)
    counts = np.outer(np.diff(np.append(ys, h)), np.diff(np.append(xs, w))).astype(float)
    return sums / np.maximum(counts, 1.0)
def _tile_expand(coarse: np.ndarray, tile: int, shape: Tuple[int, int]) -> np.ndarray:
    """Block-constant expansion of a tile grid back to the full plane — every pixel of a
    block gets that block's number, exactly how Cell-Tracker applies its per-tile ratio.
    Blocky by construction, seams included; the ``gaussian`` extent is the smooth
    alternative."""
    h, w = shape
    out = np.repeat(np.repeat(np.asarray(coarse, dtype=float), tile, axis=0), tile, axis=1)
    return out[:h, :w]
def _compute_temporal_gain(ctx: EvalContext) -> Dataset:
    """Correct intensity drift **along time** by multiplying each frame by a gain that
    brings its brightness onto a reference level. ``reference`` picks the level,
    ``extent`` picks how locally the comparison is made. Ports two Cell-Tracker plugins:

    * ``reference=series_mean``, ``extent=global`` = **Bleach Correction**;
    * ``reference=rolling_mean`` × all three extents = **Temporal Fold Correction**
      (``global`` / ``local-tile`` / ``local-gaussian``).

    The two are orthogonal — the reference is a *temporal* choice and the extent a
    *spatial* one — so all nine combinations are live and meaningful, and neither Mode is
    gated. ``window`` and ``threshold`` belong to the rolling reference alone and ARE
    gated to it; the two whole-series references correct every frame, i.e. threshold 0.

    ``extent``
        * ``global`` — one gain per frame from the whole-plane mean. Cheapest, and the
          right choice when the illumination changes uniformly.
        * ``tile`` — a gain per ``tile_size`` block, expanded block-constant. Corrects
          shading that drifts differently across the field; the block edges are visible in
          the result, which is the honest signature of the method.
        * ``gaussian`` — a gain per PIXEL, computed from ``local_sigma``-blurred frames.
          Smooth, and the most faithful when illumination drifts gradually in space.
          Because a rolling mean and a blur are both linear, blurring first and then
          taking the temporal reference is identical to the reverse, which is what makes
          this a single pass.

    Parity with Cell-Tracker is pinned END TO END, not just at the kernel level:
    ``scripts/_temporal_gain_parity.py`` runs CT's own ``_correct_global`` /
    ``_correct_local`` / ``_correct_local_gaussian`` against this node driven through the
    ``Engine``, and ``selftest::test_celltracker_parity`` asserts the same three at
    ``rtol=1e-6``. All three agree to ~1e-7 relative, which is CT's own
    ``astype(np.float32)`` in the multiply and nothing else — ragged tile blocks included.

    ONE deliberate difference remains, in the ``gaussian`` extent: the blur is scipy's
    ``gaussian_filter`` at its default ``mode="reflect"``, where Cell-Tracker used
    ``cv2.GaussianBlur``, i.e. ``BORDER_REFLECT_101``. Both truncate the kernel at 4σ, so
    the frame INTERIOR agrees to 1.5e-7 (float32 noise) at every σ tested; the two differ
    only within ~4σ of the frame edge, by up to 2% of the peak, because reflect-101 does
    not duplicate the edge sample and scipy's reflect does. scipy-at-default is the
    convention across this whole catalog — including this node's spatial sibling
    ``enhance.flatten_field``, which estimates the same kind of illumination field — so
    matching the catalog is worth more here than matching CT's border pixels.

    Grouped per ``(m, z, c)`` — one T series per plane position. Z is not pooled: two z
    planes are separate optical sections and bleach at their own rate. There is no 2D/3D
    lever for the same reason, and because the correction runs along T rather than through
    space; the declared footprint is ``WHOLE_SERIES`` over ``(t, y, x)``.

    T=1 is **refused**: with a single frame every gain is exactly 1, so the node cannot do
    anything, and silently passing the image through would hide a mis-wired graph.

    Memory: ``global`` and ``tile`` reduce each frame as they read it and never hold the
    series. ``gaussian`` must keep one blurred ``(T, Y, X)`` float32 array per group (as
    Cell-Tracker does) because the gain is per pixel.

    Intensity scale unchanged — a gain near 1 leaves the data on its count scale — so no
    ``meta_transform`` is owed (§7c)."""
    from scipy.ndimage import gaussian_filter
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("temporal gain needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    reference = str(modes.get("reference") or "series_mean")
    if reference not in _TEMPORAL_REFERENCES:
        raise ValueError(f"temporal gain: unknown reference {reference!r} — one of "
                         f"{list(_TEMPORAL_REFERENCES)}")
    extent = str(modes.get("extent") or "global")
    if extent not in _TEMPORAL_EXTENTS:
        raise ValueError(f"temporal gain: unknown extent {extent!r} — one of "
                         f"{list(_TEMPORAL_EXTENTS)}")
    if ax.t < 2:
        raise ValueError(
            "temporal gain corrects brightness ALONG TIME, and this Dataset has a single "
            "timepoint (T=1) — there is no series to compare against, so every gain would "
            "be exactly 1. Wire it downstream of a multi-frame source, or use "
            "enhance.flatten_field for a spatial (within-plane) correction.")
    window, threshold = 11, 0.0
    if reference == "rolling_mean":
        window = max(3, int(ctx.params.get("window", 11)))
        if window % 2 == 0:
            window += 1            # a centred window is only symmetric when odd
        threshold = max(0.0, float(ctx.params.get("threshold", 0.1)))
    # pixel size is read ONLY by the two spatial extents, so a `global` pull is not
    # memo-fenced on a calibration key it cannot use (the R1 rule)
    tile_px, sigma_px, px = 0, 0.0, 0.1
    if extent == "tile":
        px = ctx.calib("pixel_size_um") or 0.1
        tile_px = max(1, int(round(to_pixels_v2(
            float(ctx.params.get("tile_size", 50.0)), "um", pixel_size_um=px))))
    elif extent == "gaussian":
        px = ctx.calib("pixel_size_um") or 0.1
        sigma_px = to_pixels_v2(float(ctx.params.get("local_sigma", 10.0)), "um",
                                pixel_size_um=px)
        if sigma_px <= 0.0:
            raise ValueError(
                f"temporal gain: local σ must be > 0, but "
                f"{float(ctx.params.get('local_sigma', 10.0)):g} µm is {sigma_px:.3f} px "
                f"at pixel_size_um={px:g} — a zero-width blur makes the `gaussian` extent "
                f"a per-pixel correction with no spatial support, which just re-imprints "
                f"each pixel's own noise. Raise it, or use the `global` extent.")

    # ── the eager pass: one gain field per (m, z, c) T-series ─────────────────
    gains: Dict[Tuple[int, int, int], np.ndarray] = {}
    # The unit of WORK here is one plane read, not one (m,z,c) group: a group IS the whole
    # T-series, so counting groups leaves the bar at zero for the entire read on the common
    # single-group dataset. Counting plane reads instead gives the sub bar something to
    # move through while the frame bar tracks the T-axis (V2.17).
    n_units = max(1, ax.m * ax.z * ax.c * ax.t)
    done = 0

    def _read_tick(m: int, t: int, z: int, c: int) -> None:
        nonlocal done
        done += 1
        ctx.progress(done, n_units, f"m={m} t={t} z={z} c={c}", frames=ax.t)

    ctx.progress(0, n_units, "measuring intensity drift", frames=ax.t)
    for m in range(ax.m):
        for z in range(ax.z):
            for c in range(ax.c):
                if extent == "global":
                    signal = np.empty(ax.t, dtype=float)
                    for t in range(ax.t):
                        signal[t] = float(np.mean(
                            prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)))
                        _read_tick(m, t, z, c)
                elif extent == "tile":
                    first = _tile_means(
                        prov.get_region(0, m, 0, z, c, 0, ax.y, 0, ax.x), tile_px)
                    signal = np.empty((ax.t,) + first.shape, dtype=float)
                    signal[0] = first
                    _read_tick(m, 0, z, c)
                    for t in range(1, ax.t):
                        signal[t] = _tile_means(
                            prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), tile_px)
                        _read_tick(m, t, z, c)
                else:
                    signal = np.empty((ax.t, ax.y, ax.x), dtype=np.float32)
                    for t in range(ax.t):
                        signal[t] = gaussian_filter(
                            prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x
                                            ).astype(np.float32), sigma=sigma_px)
                        _read_tick(m, t, z, c)
                gains[(m, z, c)] = _temporal_gain_field(signal, reference, window,
                                                        threshold)
                # (the gaussian extent's gain is a whole (T,Y,X) array; it inherits the
                # signal's float32 so the eager pass stays half the size it would be)

    def gain_of(m: int, t: int, z: int, c: int, shape) -> Any:
        g = gains[(m, z, c)]
        if extent == "global":
            return float(g[t])
        if extent == "tile":
            return _tile_expand(g[t], tile_px, shape)
        return g[t]

    def apply_gain(a, m, t, z, c, *_):
        f = np.asarray(a, dtype=float)
        return f * gain_of(m, t, z, c, f.shape)

    cache = ctx.tiles
    if cache is None:                       # pre-C1 eager fallback (a bare EvalContext)
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m, t, z, c in _each_plane(ax):
            out[m, t, z, c] = apply_gain(
                prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), m, t, z, c)
        return ds.with_image(ArrayProvider(out))
    fp = stream_fp("temporal_gain", ctx.op_key, ctx.params,
                   ctx.reads.declared_reads(), (), prov)
    return ds.with_image(MapComputeProvider(prov, apply_gain, unit="plane",
                                            fp=fp, cache=cache))
register_node(
    _compute_temporal_gain, op_key="enhance.temporal_gain", label="Temporal Gain",
    category="enhancement",
    inputs=[
        InDataset(),
        InInt("window", "Rolling window", unit="", field=False, default=11,
              available_in={"reference": frozenset({"rolling_mean"})},
              description=
              "How many FRAMES the rolling reference averages over, centred on the frame "
              "being corrected. This sets what counts as 'normal' brightness locally: "
              "SHORT windows follow the data closely, so only a single-frame flicker is "
              "corrected and any drift lasting longer than the window is preserved; LONG "
              "windows approach the whole-series mean and start removing genuine slow "
              "trends. Forced ODD (an even window cannot be centred symmetrically) and "
              "floored at 3. Frames, not seconds, because the correction is defined on the "
              "sampling grid rather than on elapsed time. Only read by the rolling "
              "reference."),
        InFloat("threshold", "Correction threshold", unit="", field=False, default=0.1,
                available_in={"reference": frozenset({"rolling_mean"})},
                description=
                "Dead band on the fold deviation: a sample whose gain is within this much "
                "of 1.0 is left EXACTLY untouched. 0.1 means 'correct only what is more "
                "than 10% off its neighbours', which is what makes the rolling reference a "
                "repair for aberrant frames rather than a continuous re-levelling — set it "
                "to 0 and every frame is adjusted. Raise it to touch only gross outliers. "
                "Only read by the rolling reference; the two whole-series references "
                "always correct every frame."),
        InFloat("tile_size", "Tile size", unit="um", field=False, default=50.0,
                pick_kind="grid",
                available_in={"extent": frozenset({"tile"})},
                description=
                "Edge of the square block each gain is computed over, in microns. SMALLER "
                "blocks track shading that varies quickly across the field but average over "
                "fewer pixels, so the gain itself gets noisy and the block seams become "
                "obvious; LARGER blocks are steadier and more like a global correction. It "
                "should be several times your object size, or a single bright cell drags "
                "its own block's gain. Cell-Tracker used 128x128 PIXELS with independently "
                "settable width and height; one physical size is used here because a "
                "correction block has no reason to be anisotropic. Only read by the `tile` "
                "extent."),
        InFloat("local_sigma", "Local σ", unit="um", field=False, default=10.0,
                pick_kind="radius",
                available_in={"extent": frozenset({"gaussian"})},
                description=
                "Blur scale, in microns, used to build the smooth per-pixel gain field. It "
                "plays the same role as the tile size — the spatial scale over which "
                "brightness is pooled before being compared across time — but produces a "
                "continuous field with no block seams. SMALL values approach a per-pixel "
                "correction that re-imprints each pixel's own noise; LARGE values approach "
                "the global correction. Cell-Tracker's default was 100 px. Only read by the "
                "`gaussian` extent, which is also the only one that holds the blurred "
                "series in memory."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("reference", list(_TEMPORAL_REFERENCES), default="series_mean",
                label="Reference",
                description=
                "What brightness each frame is corrected TOWARD — a temporal choice, "
                "independent of Extent. This is what decides whether a slow trend is REMOVED "
                "or PRESERVED, which is the difference between correcting bleaching and "
                "correcting one bad frame. Getting it backwards silently deletes the signal "
                "you were measuring.",
                choice_docs={
                    "series_mean":
                        "Target is the whole series' own mean, so every frame is pulled onto one "
                        "common level. Flattens a monotone bleaching decay — and equally "
                        "flattens a real, gradual intensity change, so do not use it when the "
                        "trend IS the measurement. Cell-Tracker's Bleach Correction.",
                    "rolling_mean":
                        "Target is a centred rolling mean over Window frames, so each frame is "
                        "compared with its NEIGHBOURS. A slow trend survives untouched and only "
                        "a locally aberrant frame — a flicker, a lamp glitch — is pulled back. "
                        "Cell-Tracker's Temporal Fold Correction; the dead-band threshold "
                        "applies here.",
                    "exponential_fit":
                        "Fit a least-squares exponential decay through the signal and correct "
                        "against the FITTED curve rather than the noisy samples, so per-frame "
                        "shot noise is not injected into the correction. The most appropriate "
                        "reference for genuine photobleaching, which is exponential; it assumes "
                        "the decay really is, and a non-monotone series is fitted poorly.",
                }),
           Mode("extent", list(_TEMPORAL_EXTENTS), default="global", label="Extent",
                description=
                "How LOCALLY the gain is computed — a spatial choice, independent of Reference, "
                "so all nine combinations are meaningful. It also drives the cost: the first "
                "two reduce each frame as they read it, while `gaussian` holds a blurred copy of "
                "the whole series per group.",
                choice_docs={
                    "global":
                        "One gain per frame, from the whole plane's mean. Cheapest, and exactly "
                        "right when the illumination changes uniformly — it cannot introduce any "
                        "spatial artefact, because it multiplies the whole frame by one number.",
                    "tile":
                        "A gain per tile-sized block, applied block-constant. Corrects shading "
                        "that drifts differently across the field, which one global number "
                        "cannot — and the block edges are visible in the result, which is the "
                        "honest signature of the method rather than a bug.",
                    "gaussian":
                        "A gain per PIXEL, computed from blurred copies of the frames. The "
                        "smoothest and most faithful option when illumination drifts gradually "
                        "in space, with no block seams; it is also the memory-hungry one, since "
                        "a blurred (T,Y,X) copy is held per position/z/channel group.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="Correct intensity drift along T by a per-frame gain: reference "
                "series_mean (Cell-Tracker Bleach Correction) / rolling_mean (its Temporal "
                "Fold Correction) / exponential_fit (the decay model its UI promised but "
                "never ran), at global / tile / gaussian spatial extent. Per (m,z,c); "
                "refuses T=1.")
