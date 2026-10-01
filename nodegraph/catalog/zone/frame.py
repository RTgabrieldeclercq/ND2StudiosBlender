"""Frame (per-t) (``zone.frame``) — Per-frame-T slice: inside a zone body, iteration t yields frame t of its (T-stacked) input (t==1)."""

from __future__ import annotations

import numpy as np

from dataclasses import replace

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import AXIS_ORDER, axes_of, is_lattice
from nodegraph.engine import EvalContext
from nodegraph.memo import digest as _digest
from nodegraph.metadata import frame_slice as _meta_frame_slice
from nodegraph.provider import TileProvider
from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.sampling import _sampled

class _FrameView(TileProvider):
    """A lazy single-timepoint (t==1) view of one frame of another provider — the
    per-frame-T slice (zone.frame). Reads always resolve to the fixed source frame."""

    def __init__(self, base: TileProvider, frame: int) -> None:
        self._base = base
        self._t = int(frame)
        self.tile = base.tile
        self.levels = base.levels
        self.axes = replace(base.axes, t=1)
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)
        self._fp = _digest("frameview", base.fingerprint(), self._t)   # flat (C1)

    def level_axes(self, level: int) -> AxisSizes:
        return replace(self._base.level_axes(level), t=1)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        return self._base.read_region(
            level, m, self._t, z, c, y0, y1, x0, x1, b=b)

    def fingerprint(self) -> tuple:
        return ("frameview", self._fp)
# ── Frame slice (per-frame-T Simulation specialization) ────────────────────────

def _compute_zone_frame(ctx: EvalContext) -> Dataset:
    """Slice frame ``__frame__`` (a single timepoint) from the input — the per-frame-T
    slice a zone stamps per iteration (``nodegraph.zones.FRAME_OP``): inside a Simulation
    zone body, iteration *t* reads frame *t* of the (T-stacked) input while the Sim
    In/Out feedback carries state. Yields a t==1 Dataset — the image via a lazy
    :class:`_FrameView`; any t-bearing lattice attribute layer is sliced (keepdim) so it
    is not dropped; layers without a ``t`` axis (and structure layers) pass through."""
    ds: Dataset = ctx.inputs[0]
    t = int(ctx.params.get("__frame__", 0))
    ax = ds.axes
    if not (0 <= t < ax.t):
        raise ValueError(f"zone.frame: frame {t} out of range [0, {ax.t}) — set the "
                         f"zone's iterations to the input's T")
    out = Dataset(axes=replace(ax, t=1), metadata=dict(ds.metadata))
    out = _sampled(out, f"zone.frame[t={t}]")   # t collapses to a single chosen frame
    if ds.image is not None:
        out = out.with_image(_FrameView(ds.image, t))
    for attr in ds.attributes.values():
        axset = axes_of(attr.domain) or frozenset()
        if is_lattice(attr.domain) and "t" in axset:
            order = [a for a in AXIS_ORDER if a in axset]
            sliced = np.take(attr.values, [t], axis=order.index("t"))   # keepdim → t==1
            out = out.with_layer(attr.domain, attr.name, sliced, attr.layer)
        else:
            out = out.with_layer(attr.domain, attr.name, attr.values, attr.layer)
    return out
register_node(
    _compute_zone_frame, op_key="zone.frame", label="Frame (per-t)", category="zone",
    inputs=[InDataset()], outputs=[OutDataset()],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_frame_slice,
    description="Per-frame-T slice: inside a zone body, iteration t yields frame t of "
                "its (T-stacked) input (t==1). Drives the Simulation per-frame pattern.")
