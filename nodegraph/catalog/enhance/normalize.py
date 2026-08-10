"""Normalize (``enhance.normalize``) — Rescale to [0,1] from percentile bounds (scope plane/volume/series) or from an ABSOLUTE intensity window; the statistics scope is a Mode, not the 2D/3D lever (H24)."""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import value_rescaled as _meta_value_rescaled
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, InFloat, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, VolumeComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.map_image import _map_image
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.scope import ScopeMode

# ── Normalize (percentile — scope Mode, NOT the dim lever, H24) ──────────────────

#: ``bounds`` → where the two endpoints of the linear rescale COME FROM. Both land in
#: [0,1] and both drop ``bit_depth``; they differ in whether the endpoints are measured
#: from the data or typed by the user (V2.23).
_NORMALIZE_BOUNDS = ("percentile", "absolute")
#: How much of the series each ``bounds`` setting must read — the honest footprint,
#: resolved through ``NodeSpec.footprint_mode="bounds"``. ``percentile`` keeps the flat
#: WHOLE_SERIES it has always declared (its widest scope, ``series``, pools over T);
#: ``absolute`` reads nothing but the voxel it is rescaling, so it is genuinely TILEABLE
#: and the viewer computes only the tiles it is showing.
_NORMALIZE_GRAN: Dict[str, Granularity] = {
    "percentile": Granularity.WHOLE_SERIES,
    "absolute": Granularity.TILEABLE,
}
def _compute_normalize(ctx: EvalContext) -> Dataset:
    """Linearly rescale to [0,1], clipping outside the window. ``bounds`` (a Mode) picks
    where the window's two endpoints come from — the two are genuinely different
    operations, not a preference:

    * ``percentile`` (default) — the endpoints are MEASURED from the data, as the
      ``low_pct``/``high_pct`` percentiles of the population named by ``scope``. Adaptive:
      the same input value maps to a different output on a dim plane than on a bright one,
      unless ``scope`` is wide enough to pool them.
    * ``absolute`` (V2.23) — the endpoints are the ``low_value``/``high_value`` sockets
      VERBATIM, in the image's own intensity units. Nothing is measured, so one recipe is
      one fixed transfer curve for every plane of every file — the same distinction
      ``enhance.gamma``'s ``full_range`` draws against its ``plane_max``. This is what
      makes Cell-Tracker's Local Contrast display window expressible: chain it after
      ``enhance.flatten_field`` ``method=ratio`` with ``low_value=0.5``, ``high_value=2.0``
      to get that plugin's ``clip((f/bg - 0.5) / 1.5, 0, 1)`` exactly. A percentile
      Normalize cannot express it at any setting, because 0.5 and 2.0 are absolute
      positions on the ratio scale and a percentile is a position in the *distribution*.

    The statistics ``scope`` is a **Mode** (plane/volume/series), never the 2D/3D lever
    (H24): a lever picks the compute *footprint*, whereas normalization scope picks the
    *statistical population*. It is Mode-gated to ``bounds=percentile`` (§5c) — an
    absolute window has no population to pick, so the dropdown would be a dead control.

    **Footprint per ``bounds``** (``footprint_mode="bounds"``): ``percentile`` keeps
    WHOLE_SERIES, ``absolute`` is TILEABLE, since a pointwise clip-and-scale reads only
    the voxel it writes. ``kernel_axes`` stays empty for both — pointwise either way.

    Both settings return [0,1] floats, so ``value_rescaled`` drops ``bit_depth`` for both
    and no branch owes a different meta_transform (§7c)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("normalize needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    bounds = str(modes.get("bounds") or "percentile")
    if bounds not in _NORMALIZE_BOUNDS:
        raise ValueError(f"normalize: unknown bounds {bounds!r} — one of "
                         f"{list(_NORMALIZE_BOUNDS)}")
    scope = modes.get("scope", "plane")
    lo_p = float(ctx.params.get("low_pct", 1.0))
    hi_p = float(ctx.params.get("high_pct", 99.0))

    def rescale(block: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(block, lo_p), np.percentile(block, hi_p)
        if hi <= lo:
            return np.zeros_like(block)
        return np.clip((block - lo) / (hi - lo), 0.0, 1.0)

    def apply_lohi(block: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi <= lo:
            return np.zeros_like(block)
        return np.clip((block - lo) / (hi - lo), 0.0, 1.0)

    if bounds == "absolute":
        # `high_value`'s derive is read through ctx.channel(0) so it resolves headless too
        # (the `analysis.threshold` fixed-level pattern): the declared full scale on integer
        # data, 1.0 when no bit depth is declared. Read ONLY on this branch, so a percentile
        # pull is not memo-fenced on a key it ignores (R1).
        lo = float(ctx.params.get("low_value", 0.0))
        hi = float(ctx.channel(0).param("high_value", 1.0))
        if hi <= lo:
            # Eager, unlike the percentile path's silent zeros: an absolute window is known
            # WITHOUT reading a pixel, so this is a user error the node can name. Under C1 a
            # backend error raised inside the lazy apply would be memoized into a poisoned
            # payload and surface far from this node (the `flatten_field` σ≤0 precedent).
            raise ValueError(
                f"normalize: the absolute window is empty — high_value ({hi:g}) must be "
                f"strictly greater than low_value ({lo:g}). Both are in the image's own "
                f"intensity units; auto sets high to the declared full scale, and after a "
                f"node that dropped the bit depth (a percentile Normalize, or "
                f"enhance.flatten_field method=ratio) that is 1.0 rather than a count.")
        # Pointwise ⇒ TILEABLE, so `_map_image` streams per canonical tile with no halo.
        return _map_image(
            ctx, ds, plane_fn=lambda a: apply_lohi(a, lo, hi)
        ).with_metadata(bit_depth=None)

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        img = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m, t, z, c in _each_plane(ax):
            img[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        out = np.empty_like(img)
        if scope == "plane":
            for m, t, z, c in _each_plane(ax):
                out[m, t, z, c] = rescale(img[m, t, z, c])
        elif scope == "volume":
            for m in range(ax.m):
                for t in range(ax.t):
                    for c in range(ax.c):
                        out[m, t, :, c] = rescale(img[m, t, :, c])
        else:  # series — the whole T stack per (m,c)
            for m in range(ax.m):
                for c in range(ax.c):
                    out[m, :, :, c] = rescale(img[m, :, :, c])
        return ds.with_image(ArrayProvider(out)).with_metadata(bit_depth=None)
    # C1 (V2.04 §6b sliver): eager-stat / lazy-apply, unit = the statistics scope.
    # `plane` scope is SELF-CONTAINED per plane (percentiles of the plane it normalizes),
    # so it needs no pre-stat → a per-plane MapComputeProvider. `volume` scope per (m,t,c)
    # → VolumeComputeProvider. `series` scope's stat spans T, so the (lo,hi) per (m,c) is
    # computed EAGERLY once (a percentile needs the whole population regardless) and baked
    # into a lazy per-plane apply. Only touched units compute; the eager fallback above
    # uses the SAME rescale so the bytes are identical.
    fp = stream_fp("normalize", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    if scope == "plane":
        return ds.with_image(MapComputeProvider(
            prov, lambda a, m, t, z, c, *_: rescale(a), unit="plane", fp=fp, cache=cache)
            ).with_metadata(bit_depth=None)
    if scope == "volume":
        return ds.with_image(VolumeComputeProvider(
            prov, lambda v, m, t, c: rescale(v), fp=fp, cache=cache)
            ).with_metadata(bit_depth=None)
    lohi: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for m in range(ax.m):
        for c in range(ax.c):
            block = np.stack([prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                              for t in range(ax.t) for z in range(ax.z)]).astype(float)
            lohi[(m, c)] = (float(np.percentile(block, lo_p)),
                            float(np.percentile(block, hi_p)))
    return ds.with_image(MapComputeProvider(
        prov, lambda a, m, t, z, c, *_: apply_lohi(a, *lohi[(m, c)]),
        unit="plane", fp=fp, cache=cache)).with_metadata(bit_depth=None)
register_node(
    _compute_normalize, op_key="enhance.normalize", label="Normalize",
    category="enhancement",
    inputs=[InDataset(),
            # The two percentile sockets are gated to `bounds=percentile`: an absolute
            # window measures nothing, so the compute never reads them there and a visible
            # percentile field would be a live-looking control the branch ignores (socket
            # contract clause 2). Names stay DISJOINT from the absolute pair — the node card
            # relayouts on the socket-name list, so one socket meaning two things would not
            # redraw (the `analysis.segment` lesson).
            InFloat("low_pct", "Low %", unit="", field=True, default=1.0,
                    pick_kind="percentile", pick_peer="high_pct",
                    available_in={"bounds": frozenset({"percentile"})},
                    description=
                    "The percentile mapped to 0 — everything at or below it CLIPS to 0 and "
                    "is lost. 1 discards the darkest 1% of samples, which removes the "
                    "read-noise floor; 0 uses the true minimum and lets a single dead pixel "
                    "set the black point. Raise it to crush more background, but note dim "
                    "real signal clips away with it. The population it is taken over is set "
                    "by Scope, not by this socket."),
            InFloat("high_pct", "High %", unit="", field=True, default=99.0,
                    pick_kind="percentile", pick_peer="low_pct",
                    available_in={"bounds": frozenset({"percentile"})},
                    description=
                    "The percentile mapped to 1 — everything at or above it CLIPS to 1 and "
                    "becomes indistinguishable. 99 ignores the brightest 1% so a few hot "
                    "pixels cannot compress the whole range; 100 uses the true maximum. "
                    "LOWER brightens the image and saturates more of it. Because clipping is "
                    "irreversible, a too-low value silently flattens the brightest objects "
                    "into one value — the usual cause of peak intensities that all read "
                    "exactly 1. If the window collapses (high ≤ low) the output is all "
                    "zeros rather than an error."),
            # ── the absolute window (V2.23), gated to `bounds=absolute` ──
            # An eyedropper (`pick_kind="level"`) is the right gesture for both: these are
            # intensities in the image's own units, so clicking the dimmest background pixel
            # and the brightest object pixel IS the window. Peered, so one gesture aims the
            # interval in two phases.
            # field=False: this compute resolves both endpoints to SCALARS and never
            # evaluates a wired Field, so advertising one would be the lie the catalog
            # forbids (selftest asserts exactly this on track.link and analysis.segment).
            # A per-voxel window is also not the operation — the whole point of `absolute`
            # is ONE fixed transfer curve; vary it spatially and it is neither fixed nor
            # comparable, which is what `percentile` already offers.
            InFloat("low_value", "Low value", unit="", field=False, default=0.0,
                    pick_kind="level", pick_peer="high_value",
                    available_in={"bounds": frozenset({"absolute"})},
                    description=
                    "The intensity mapped to 0 — everything at or below it CLIPS to 0 and is "
                    "lost. In the image's OWN units, taken verbatim, so unlike Low % it means "
                    "the same brightness on every plane and in every file: that is the whole "
                    "point of an absolute window, and also why it does not adapt if your "
                    "exposure changes. 0 keeps the full dark end. RAISING it crushes more "
                    "background to black, and any dim real signal below it goes with it — "
                    "irreversibly, so an intensity you measure downstream is a clipped one. "
                    "On the dimensionless output of `enhance.flatten_field` method=ratio, 1.0 "
                    "is the local background level, so a value near 0.5 sets black at half "
                    "the background. Only read when Bounds is `absolute`."),
            InFloat("high_value", "High value", unit="", field=False, default=1.0,
                    pick_kind="level", pick_peer="low_value",
                    derive="(2**bit_depth - 1) if bit_depth else 1.0",
                    available_in={"bounds": frozenset({"absolute"})},
                    description=
                    "The intensity mapped to 1 — everything at or above it CLIPS to 1 and "
                    "becomes indistinguishable. In the image's OWN units, taken verbatim, so "
                    "one setting is one fixed transfer curve across every plane and every "
                    "file. Auto follows the declared intensity scale: the FULL declared range "
                    "(4095 on 12-bit data), which clips nothing, falling back to 1.0 when no "
                    "bit depth is declared — i.e. after a percentile Normalize or after "
                    "`enhance.flatten_field` method=ratio. LOWERING it brightens the image and "
                    "saturates more of it, flattening the brightest objects into one value. On "
                    "a ratio image 2.0 means \"twice the local background is full white\", "
                    "which with Low value 0.5 reproduces Cell-Tracker's Local Contrast display "
                    "window. Must be strictly greater than Low value — an empty window is "
                    "refused with an error rather than returning a black image, because "
                    "absolute bounds are wrong before any pixel is read. Only read when Bounds "
                    "is `absolute`.")],
    outputs=[OutDataset()],
    modes=[Mode("bounds", list(_NORMALIZE_BOUNDS), default="percentile", label="Bounds",
                description=
                "Where the two endpoints of the rescale COME FROM — measured from your data, "
                "or typed by you. This is the setting that decides whether the node is an "
                "ADAPTIVE contrast stretch or a FIXED transfer curve, which is the same "
                "distinction Gamma's Scale mode draws. Both options clip outside the window "
                "and both return [0,1] floats, so both drop the declared bit depth.",
                choice_docs={
                    "percentile":
                        "Endpoints measured from the data, as the Low %/High % percentiles of "
                        "the population named by Scope. Adapts to whatever it is given, so it "
                        "needs no knowledge of your camera and always produces a full-contrast "
                        "image — but the same input value maps to a different output on a dim "
                        "plane than on a bright one unless Scope is wide enough to pool them, "
                        "and an empty plane comes out as noise stretched to full scale. The "
                        "default, and the node's only behaviour before V2.23.",
                    "absolute":
                        "Endpoints taken verbatim from the Low value/High value sockets, in "
                        "the image's own intensity units. Nothing is measured, so one recipe "
                        "is one fixed curve for every plane of every file and frames stay "
                        "comparable — at the cost of meaning nothing until you know your "
                        "data's scale, and of clipping everything if you guess it wrong. It is "
                        "also the only way to express a window at ABSOLUTE positions, such as "
                        "Cell-Tracker's Local Contrast display window (0.5 … 2.0 on the output "
                        "of Flatten Illumination method=ratio) — a percentile cannot, because "
                        "it names a position in the distribution rather than on the scale. "
                        "Pointwise, so it is also the cheaper of the two: TILEABLE, and it "
                        "never reads a second plane.",
                }),
           # Gated to `percentile` (wire-node-v2 §5c): an absolute window has no statistical
           # population, so a population picker under it would be a dead dropdown — exactly
           # what `analysis.threshold` does with this same Scope vocabulary under `fixed`.
           #
           # Declared through the shared factory (V2.27) so it carries `role="scope"`: this node
           # is the standing PROOF that a population is not a footprint — its `footprint_mode`
           # is `bounds`, and binding the card's footprint band to that field would have offered
           # percentile/absolute (which decides *which sockets are live*) instead of the three
           # populations below. The prose stays local: these percentiles are a black/white point,
           # not a histogram cut.
           ScopeMode(("plane", "volume", "series"), default="plane",
                available_in={"bounds": frozenset({"percentile"})},
                description=
                "The population the two percentiles are measured over — how much data has to "
                "agree on one black point and one white point. Widening it makes frames "
                "comparable to each other at the cost of adapting to any of them, and it is "
                "the setting that decides whether a brightening time course survives "
                "normalization or is flattened away.",
                choice_docs={
                    "plane":
                        "Percentiles from each (Y,X) plane on its own. Every plane is "
                        "stretched to fill [0,1], so contrast is best per plane and the "
                        "cheapest to compute — but a real intensity change between frames or "
                        "Z planes is normalized away, and an empty plane comes out as noise "
                        "amplified to full scale.",
                    "volume":
                        "One black/white point per (Z,Y,X) volume, i.e. per timepoint, "
                        "channel and position. Z planes stay comparable to each other — "
                        "attenuation with depth is preserved as a real signal — while each "
                        "timepoint still adapts on its own.",
                    "series":
                        "One black/white point per (position, channel), measured over the "
                        "whole T×Z stack. The only scope under which intensities are "
                        "comparable ACROSS TIME, so it is the right one before measuring a "
                        "time course; it also has to read the whole series before the first "
                        "plane comes out, which is the slowest start.",
                })],
    # Keyed by `bounds`, not by a dim lever this node does not have (NodeSpec.footprint_mode).
    # `percentile` keeps the WHOLE_SERIES it always declared — its widest scope pools over T,
    # and the narrower scopes over-declare harmlessly. `absolute` reads only the voxel it
    # writes, so claiming WHOLE_SERIES there would forfeit tiling for nothing.
    granularity=_NORMALIZE_GRAN, footprint_mode="bounds", kernel_axes=frozenset(),
    # axis-preserving, but it changes what the NUMBERS mean: [0,1] floats are not raw
    # counts, so `bit_depth` is dropped (edit-time + payload in lockstep) and every
    # downstream raw-count consumer sees "no declared integer scale" (wire-node-v2 §7c).
    # True for BOTH bounds settings — each returns [0,1] — so the flat transform stands.
    meta_transform=_meta_value_rescaled,
    description="Rescale to [0,1] from percentile bounds — the statistics scope "
                "(plane/volume/series) is a Mode, not the 2D/3D lever (H24) — or from an "
                "ABSOLUTE intensity window typed in the image's own units, which is one "
                "fixed transfer curve for the whole series and the only way to express a "
                "window at absolute positions (Cell-Tracker's Local Contrast display "
                "window after Flatten Illumination method=ratio). Scope and the percentile "
                "sockets show only under `percentile`, the value sockets only under "
                "`absolute`. Drops bit_depth either way — the output is no longer raw "
                "integer counts.")
