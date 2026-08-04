"""Normalize (``enhance.normalize``) — Percentile normalize to [0,1]; the statistics scope (plane/volume/series) is a Mode, not the 2D/3D lever (H24)."""

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
from nodegraph.catalog._shared.planes import _each_plane

# ── Normalize (percentile — scope Mode, NOT the dim lever, H24) ──────────────────

def _compute_normalize(ctx: EvalContext) -> Dataset:
    """Percentile-rescale to [0,1]. The statistics ``scope`` is a **Mode**
    (plane/volume/series), never the 2D/3D lever (H24): a lever picks the compute
    *footprint*, whereas normalization scope picks the *statistical population*."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("normalize needs an image provider on its input Dataset")
    ax = prov.axes
    scope = ctx.params.get("__modes__", {}).get("scope", "plane")
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
            InFloat("low_pct", "Low %", unit="", field=True, default=1.0,
                    pick_kind="percentile", pick_peer="high_pct",
                    description=
                    "The percentile mapped to 0 — everything at or below it CLIPS to 0 and "
                    "is lost. 1 discards the darkest 1% of samples, which removes the "
                    "read-noise floor; 0 uses the true minimum and lets a single dead pixel "
                    "set the black point. Raise it to crush more background, but note dim "
                    "real signal clips away with it. The population it is taken over is set "
                    "by Scope, not by this socket."),
            InFloat("high_pct", "High %", unit="", field=True, default=99.0,
                    pick_kind="percentile", pick_peer="low_pct",
                    description=
                    "The percentile mapped to 1 — everything at or above it CLIPS to 1 and "
                    "becomes indistinguishable. 99 ignores the brightest 1% so a few hot "
                    "pixels cannot compress the whole range; 100 uses the true maximum. "
                    "LOWER brightens the image and saturates more of it. Because clipping is "
                    "irreversible, a too-low value silently flattens the brightest objects "
                    "into one value — the usual cause of peak intensities that all read "
                    "exactly 1. If the window collapses (high ≤ low) the output is all "
                    "zeros rather than an error.")],
    outputs=[OutDataset()],
    modes=[Mode("scope", ["plane", "volume", "series"], default="plane",
                label="Scope",
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
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    # axis-preserving, but it changes what the NUMBERS mean: [0,1] floats are not raw
    # counts, so `bit_depth` is dropped (edit-time + payload in lockstep) and every
    # downstream raw-count consumer sees "no declared integer scale" (wire-node-v2 §7c).
    meta_transform=_meta_value_rescaled,
    description="Percentile normalize to [0,1]; the statistics scope "
                "(plane/volume/series) is a Mode, not the 2D/3D lever (H24). Drops "
                "bit_depth — the output is no longer raw integer counts.")
