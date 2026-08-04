"""Flatten Illumination (``enhance.flatten_field``) — Flatten uneven illumination by removing a large-σ Gaussian background: subtract / subtract+mean / divide+mean / ratio, estimated per plane or averaged over T."""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import flatten_field as _meta_flatten_field
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, InFloat, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_plane, _each_plane_p
from nodegraph.catalog._shared.units import to_pixels_v2

# ══════════════════════════════════════════════════════════════════════════════
#  Cell-Tracker parity ports (V2.13)
# ══════════════════════════════════════════════════════════════════════════════
#
# The five nodes below port the processing that existed only in the standalone
# **Cell-Tracker** app (McGheeLab/Cell-Tracker): everything in its
# ``plugins/enhancement/builtin.py``, ``backend/normalization.py``,
# ``backend/measurement.py`` and ``backend/fields.py`` this catalog did not already
# cover. Its other two halves were vendored earlier and are NOT repeated here — StarDist
# nuclear segmentation is ``analysis.segment`` method=``stardist``
# (:mod:`nodegraph.kernels.stardist_segment`), and its three cell linkers are
# ``track.objects`` methods ``topology`` / ``fingerprint`` / ``overlap``
# (:mod:`nodegraph.kernels.track_objects`).
#
# **Why five nodes and not ten.** Four of Cell-Tracker's plugins (Background Subtract,
# Spatial Flatness ×2 methods, Local Contrast) are the same large-σ Gaussian background
# estimate with a different closing arithmetic line, and two (Bleach Correction, Temporal
# Fold Correction) are the same per-frame multiplicative gain with a different reference
# signal. They land as ``method`` / ``reference`` Modes on two nodes, which is the
# ``analysis.segment`` charter: one node per data contract, the algorithm as a Mode.
#
# **Four deliberate departures from the source**, each forced by the fact that
# Cell-Tracker assumes a single uint16 ``(T,H,W)`` stack while this engine is float and 6-D:
#
#   1. Every radius / σ / tile extent is authored in **µm** and converted through
#      :func:`to_pixels_v2`, so one recipe means the same physical scale at any
#      magnification. Cell-Tracker's numbers are pixels and do not transfer.
#   2. The ``np.clip(..., 0, 65535)`` that ends most Cell-Tracker plugins is **dropped**.
#      A hard 16-bit ceiling would silently saturate the 12-bit data most ND2s carry once
#      it is scaled up, and is meaningless on the [0,1] floats a percentile Normalize
#      produces. Values stay float and unclamped; only ``subtract`` keeps its floor at 0,
#      because a negative "amount of signal above background" is not a measurement.
#   3. A divide-by-background floors its denominator at a **derived** value rather than
#      the literal ``1.0``. One raw count is right for integers and a total no-op on
#      normalized floats, where it would silently turn ``f/bg`` into ``f`` (§7c).
#   4. Per-object analysis reports **physical** units — µm, µm/s, 1/s — not px/frame.
#
# Both analysis nodes are **2-D**, like their sources and like ``track.objects``: they
# refuse a 3D (``z_kind="subpixel"``) structure table rather than pretend a 2-D
# neighbourhood estimator or a 2-D curl means something in a volume.


# ── Flatten Illumination (CT Background Subtract + Spatial Flatness + Local Contrast) ──

#: ``method`` → the arithmetic that removes the estimated background.
_FLATTEN_METHODS = ("subtract", "subtract_mean", "divide_mean", "ratio")
#: the three methods that DIVIDE by, or add the mean of, the background — they need the
#: denominator floor; plain ``subtract`` does not (Cell-Tracker floors the same three).
_FLATTEN_FLOORED = frozenset({"subtract_mean", "divide_mean", "ratio"})
def _compute_flatten_field(ctx: EvalContext) -> Dataset:
    """Flatten the illumination field: estimate a smooth background with a large-σ Gaussian
    blur, then remove it. ``method`` picks the arithmetic, ``reference`` picks what the
    background is estimated from. Ports three Cell-Tracker plugins at once:

    ==================  ====================================  ==========================
    ``method``          output                                Cell-Tracker plugin
    ==================  ====================================  ==========================
    ``subtract``        ``clip(f - bg, 0, ∞)``                **Background Subtract**
    ``subtract_mean``   ``f - bg + mean(bg)``                 **Spatial Flatness** subtract
    ``divide_mean``     ``(f / bg) · mean(bg)``               **Spatial Flatness** divide
    ``ratio``           ``f / bg``                            **Local Contrast**
    ==================  ====================================  ==========================

    ``subtract`` removes the background outright, so absolute intensities afterwards are
    background-subtracted rather than raw and anything dimmer than its local background is
    now exactly 0. The two ``*_mean`` methods add the background's own mean back, which
    keeps the frame at roughly its original brightness — the difference between them is
    whether shading is treated as an additive offset (stray light, camera bias) or a
    multiplicative gain (uneven excitation, vignetting), and that is a property of your
    optics, not a preference. ``ratio`` is the dimensionless version: ≈1 in flat
    background, >1 on an object. Cell-Tracker's Local Contrast additionally squeezed the
    ratio through a fixed ``(x-0.5)/1.5`` display window and clipped it to the dtype
    range; that is a *display* transform, so it is not reproduced here — chain
    ``enhance.normalize`` if you want it.

    ``reference``
        * ``per_plane`` — the background is estimated from the plane being corrected.
          Self-contained, so it streams lazily one plane at a time.
        * ``time_averaged`` — the background is the **mean over T** of the per-frame
          estimates, computed once per ``(m, z, c)`` and applied to every frame. Better
          when the illumination is steady and the cells are not: a per-plane estimate
          partly follows the cells around and subtracts them, an averaged one cannot.
          Z is deliberately NOT pooled — two z planes are different optical sections with
          their own shading, and averaging them would flatten each by the other's.

    **No 2D/3D lever.** Illumination flatness is a property of the lateral light path, and
    each z plane is its own optical section, so the background is always estimated in-plane
    — which is also exactly what the 2-D source did. The footprint is therefore a flat
    ``WHOLE_SERIES``: ``time_averaged``'s statistic spans T, and (like
    ``enhance.normalize``, which declares the same for the same reason) the compute then
    streams the *apply* step per plane so only touched planes materialize.

    Intensity scale: three methods stay on the input's count scale, ``ratio`` does not, so
    the ``flatten_field`` meta_transform drops ``bit_depth`` for that one method and the
    compute drops it on the payload in lockstep (§7c)."""
    from scipy.ndimage import gaussian_filter
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("flatten field needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    method = str(modes.get("method") or "subtract")
    if method not in _FLATTEN_METHODS:
        raise ValueError(f"flatten field: unknown method {method!r} — one of "
                         f"{list(_FLATTEN_METHODS)}")
    averaged = str(modes.get("reference") or "per_plane") == "time_averaged"
    px = ctx.calib("pixel_size_um") or 0.1
    sigma_um = float(ctx.params.get("sigma", 20.0))
    sigma_px = to_pixels_v2(sigma_um, "um", pixel_size_um=px)
    if sigma_px <= 0.0:
        # eager: under C1 the kernel runs at READ time, so a backend error here would be
        # memoized into a poisoned lazy payload and surface far from this node.
        raise ValueError(
            f"flatten field: σ must be > 0, but {sigma_um:g} µm is {sigma_px:.3f} px at "
            f"pixel_size_um={px:g}. A zero-width background estimate IS the image, which "
            f"makes every method either a no-op or a divide by itself.")
    # Floored eagerly, through ctx.channel(0) so the socket's `derive` resolves headless
    # too (the `analysis.segment` threshold pattern): one raw count on integer data, 1e-6
    # on the [0,1] floats a percentile Normalize leaves behind. Read ONLY on the methods
    # that use it, so a plain `subtract` pull is not memo-fenced on bit_depth (R1).
    floor = (float(ctx.channel(0).param("bg_floor", 1.0))
             if method in _FLATTEN_FLOORED else 0.0)

    def background(plane: np.ndarray) -> np.ndarray:
        return gaussian_filter(np.asarray(plane, dtype=float), sigma=sigma_px)

    def apply(plane: np.ndarray, bg: np.ndarray) -> np.ndarray:
        f = np.asarray(plane, dtype=float)
        if method == "subtract":
            return np.clip(f - bg, 0.0, None)          # CT leaves this one unfloored
        b = np.maximum(bg, floor)
        if method == "subtract_mean":
            return f - b + float(b.mean())
        if method == "divide_mean":
            return (f / b) * float(b.mean())
        return f / b                                    # ratio

    def emit(result: Dataset) -> Dataset:
        # `ratio` is the one method whose output is no longer a count scale — drop
        # bit_depth on the PAYLOAD in lockstep with the meta_transform, which already
        # dropped it in the envelope (§7c).
        return result.with_metadata(bit_depth=None) if method == "ratio" else result

    bgs: Dict[Tuple[int, int, int], np.ndarray] = {}
    if averaged:
        # Eager whole-population statistic (a mean over T), hence the WHOLE_SERIES
        # footprint. Accumulated in ONE pass over the planes, keyed (m, z, c).
        for m, t, z, c in _each_plane_p(ctx, ax, "averaging background"):
            key = (m, z, c)
            b = background(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x))
            bgs[key] = b if key not in bgs else bgs[key] + b
        for key in list(bgs):
            bgs[key] = bgs[key] / max(1, ax.t)

    cache = ctx.tiles
    if cache is None:                       # pre-C1 eager fallback (a bare EvalContext)
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m, t, z, c in _each_plane(ax):
            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
            out[m, t, z, c] = apply(plane, bgs[(m, z, c)] if averaged
                                    else background(plane))
        return emit(ds.with_image(ArrayProvider(out)))
    fp = stream_fp("flatten", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)

    def apply_plane(a, m, t, z, c, *_):
        return apply(a, bgs[(m, z, c)] if averaged else background(a))

    return emit(ds.with_image(MapComputeProvider(
        prov, apply_plane, unit="plane", fp=fp, cache=cache)))
register_node(
    _compute_flatten_field, op_key="enhance.flatten_field",
    label="Flatten Illumination", category="enhancement",
    inputs=[
        InDataset(),
        InFloat("sigma", "Background σ", unit="um", field=False, default=20.0,
                pick_kind="radius",
                description=
                "Spatial scale of the background estimate, in microns — the ONE knob that "
                "decides what counts as background. It must be comfortably LARGER than the "
                "objects you want to keep: at or below cell size the blur follows the cells "
                "themselves, so each cell becomes its own background and subtracts itself "
                "away, leaving hollow rings. Too large and genuine shading survives "
                "uncorrected. A few times the largest object diameter is the working range; "
                "20 µm suits typical adherent cells. Cell-Tracker's equivalents default to "
                "100 px (Spatial Flatness / Background Subtract) and 50 px (Local Contrast), "
                "which are only comparable once multiplied by that dataset's pixel size."),
        InFloat("bg_floor", "Background floor", unit="", field=False, default=1.0,
                derive="1.0 if bit_depth else 1e-06",
                available_in={"method": frozenset(_FLATTEN_FLOORED)},
                description=
                "Smallest value the background may take before it is used as a divisor (or "
                "before its mean is added back) — it stops a genuinely black region from "
                "producing a division by ~0 and a wall of huge values. Auto resolves it from "
                "the declared intensity scale: ONE RAW COUNT on integer data (what "
                "Cell-Tracker hardcodes), or 1e-6 when no bit depth is declared, which is "
                "the state after a percentile Normalize. That second case is why it is not "
                "simply fixed at 1.0 — on [0,1] data a floor of 1.0 exceeds every value in "
                "the image, so the division would quietly return the input unchanged. Raise "
                "it to clamp harder in dark regions; it has no effect where the background "
                "is well above it. Not read by the `subtract` method."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("method", list(_FLATTEN_METHODS), default="subtract", label="Method",
                description=
                "The arithmetic that removes the estimated background — all four share the same "
                "large-σ Gaussian estimate and differ only in how it is applied. The real "
                "question they answer is whether your shading is an additive offset (stray "
                "light, camera bias) or a multiplicative gain (uneven excitation, vignetting), "
                "which is a property of the optics rather than a preference. Only `ratio` "
                "leaves the input's intensity scale, and it drops the declared bit depth "
                "accordingly.",
                choice_docs={
                    "subtract":
                        "`clip(f - bg, 0, ∞)` — remove the background outright. Intensities "
                        "afterwards are background-SUBTRACTED, not raw, and anything dimmer than "
                        "its local background becomes exactly 0, so the dark end is clipped and "
                        "unrecoverable. Cell-Tracker's Background Subtract; the default.",
                    "subtract_mean":
                        "`f - bg + mean(bg)` — subtract the shading but add its average back, so "
                        "the frame keeps roughly its original brightness and nothing clips at 0. "
                        "Treats shading as an ADDITIVE offset. Cell-Tracker's Spatial Flatness "
                        "(subtract).",
                    "divide_mean":
                        "`(f / bg) · mean(bg)` — divide the shading out and rescale to the "
                        "background's mean, keeping the original brightness. Treats shading as a "
                        "multiplicative GAIN, which is the correct model for uneven excitation "
                        "and vignetting. Cell-Tracker's Spatial Flatness (divide).",
                    "ratio":
                        "`f / bg` — the dimensionless version: about 1 in flat background and "
                        "above 1 on an object, so values read directly as \"times the local "
                        "background\". Comparable across frames and fields, but no longer in "
                        "counts, so the declared bit depth is dropped. Cell-Tracker's Local "
                        "Contrast (without its display window).",
                }),
           Mode("reference", ["per_plane", "time_averaged"], default="per_plane",
                label="Reference",
                description=
                "What the background is estimated FROM. The hazard both options negotiate is "
                "that a background estimate made from an image containing cells partly follows "
                "the cells — so subtracting it removes some of the signal you are measuring.",
                choice_docs={
                    "per_plane":
                        "Estimate the background from the plane being corrected. Self-contained "
                        "and streams one plane at a time, and it adapts if the illumination "
                        "genuinely changes — but with sparse bright objects it partly subtracts "
                        "them, dimming exactly what you care about.",
                    "time_averaged":
                        "Average the per-frame estimates over T, once per (position, z, channel), "
                        "and apply that to every frame. Cells move, illumination does not, so "
                        "averaging suppresses the cells' contribution and leaves the real "
                        "shading — the better choice for a timelapse of a steady field. Z is "
                        "deliberately NOT pooled: two z planes are different optical sections "
                        "with their own shading.",
                })],
    # Flat WHOLE_SERIES + no dim lever: `time_averaged`'s statistic spans T, and the
    # background is always a LATERAL estimate (see the compute docstring). Same shape as
    # `enhance.normalize`, which declares WHOLE_SERIES for its `series` scope and then
    # streams the apply step per plane.
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"y", "x"}),
    meta_transform=_meta_flatten_field,
    description="Flatten uneven illumination by removing a large-σ Gaussian background: "
                "subtract / subtract+mean / divide+mean / ratio, estimated per plane or "
                "averaged over T. Ports Cell-Tracker's Background Subtract, Spatial "
                "Flatness and Local Contrast. `ratio` drops bit_depth (dimensionless).")
