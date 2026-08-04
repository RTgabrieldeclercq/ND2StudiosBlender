"""Select Channel (``channel.select``) — Subset/reorder the channel axis (metadata follows in lockstep)."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from typing import Sequence

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.engine import EvalContext
from nodegraph.memo import digest as _digest
from nodegraph.metadata import (channel_select as _meta_channel_select,
                                channel_subset as _channel_subset, parse_channels)
from nodegraph.provider import TileProvider
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.sampling import CHANNEL_STAMP, _sampled

# ── Select Channel (H12 — the deconvolve prerequisite) ────────────────────────

class _ChannelView(TileProvider):
    """A lazy channel-subset view over another provider (Select Channel output)."""

    def __init__(self, base: TileProvider, channels: Sequence[int]) -> None:
        self._base = base
        self._ch = [int(c) for c in channels]
        self.tile = base.tile
        self.levels = base.levels
        self.axes = replace(base.axes, c=len(self._ch))
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)
        # flat digest, computed once (a nested-tuple fp recurses in _canon and goes
        # quadratic/overflows on deep unrolled chains — C1 / V2.04 §6b)
        self._fp = _digest("channelview", base.fingerprint(), tuple(self._ch))

    def level_axes(self, level: int) -> AxisSizes:
        return replace(self._base.level_axes(level), c=len(self._ch))

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        return self._base.read_region(level, m, t, z, self._ch[c], y0, y1, x0, x1)

    def fingerprint(self) -> tuple:
        return ("channelview", self._fp)
def _compute_select_channel(ctx: EvalContext) -> Dataset:
    ds: Dataset = ctx.inputs[0]
    # parse_channels is SHARED with the `channel_select` meta_transform so the predicted
    # envelope and the produced payload cannot drift (build-node-v2 §2).
    requested = parse_channels(ctx.params.get("channels"))
    channels = list(requested or range(ds.axes.c))
    channels = [c for c in channels if 0 <= c < ds.axes.c]
    if requested and not channels:
        # A non-empty request that survives NO index is a user error, and it used to produce
        # a c=0 Dataset silently. Nothing refuses an empty channel axis downstream, so the
        # degenerate payload travelled until something indexed into it — on a 1-channel ND2,
        # `channels="1"` (or Python's "-1" for last, which is not supported here) surfaced
        # as `IndexError: index 0 is out of bounds for axis 0 with size 0` from inside
        # skimage, naming neither this node nor the parameter. Refuse at the source, the way
        # util.crop already refuses an empty region.
        raise ValueError(
            f"channel.select: {sorted(requested)} selects no channel — this Dataset has "
            f"{ds.axes.c} channel(s), so the only valid indices are "
            f"{'0' if ds.axes.c == 1 else f'0..{ds.axes.c - 1}'}. Negative indices are not "
            f"'from the end' here, they are simply out of range. Clear the socket to keep "
            f"every channel.")
    new_axes = replace(ds.axes, c=len(channels))
    out = replace(ds, axes=new_axes)
    if ds.image is not None:
        out = out.with_image(_ChannelView(ds.image, channels))
    # EVERY per-channel list follows the selection in lockstep, via the SAME shared
    # `channel_subset` the `channel_select` meta_transform uses, so the payload and the
    # predicted envelope cannot drift (build-node-v2 §2). Doing this for the calibration key
    # alone is what made a tap on channel 1 report channel 0's NAME: the lists are read
    # POSITIONALLY, so a full-length survivor on a c=1 Dataset does not look stale — it is
    # the wrong channel's label in the Viewer's channel strip, the card's sockets and the
    # hover readout. Reading them off `ds.metadata` is strict-read safe for the display keys
    # (they are non-calibration, so `_StrictCalibMetadata` passes them through) and is what
    # `view.overlay` already does for `channel_names`.
    changes = _channel_subset(ds.metadata, channels)
    # ...with the one calibration key preferring ctx.calib: the meta_transform has already
    # subset it in the envelope, so mirroring THAT keeps the envelope the single source of
    # truth and keeps the read memo-fenced. Both routes narrow the same list by the same
    # validated indices, so they agree; the fallback matters when the envelope carries no
    # emission list at all (an unseeded source), where the payload's own list still has to
    # be narrowed rather than left at full length.
    emis = ctx.calib("channel_emission_nm")
    if isinstance(emis, (list, tuple)):
        changes["channel_emission_nm"] = list(emis)
    if changes:
        out = out.with_metadata(**changes)
    # A channel-only stamp (`c:`): the selection reindexes the c axis and leaves every
    # (z,y,x) address pointing at the same physical location, so a consumer comparing two
    # single-channel branches drops it. See nodegraph.catalog._shared.sampling._sampling_of.
    return _sampled(out.reshaped_axes(new_axes),
                    f"{CHANNEL_STAMP}channel.select{tuple(channels)}")
register_node(
    _compute_select_channel,
    op_key="channel.select", label="Select Channel", category="channel",
    # The node is palette-visible (it is NOT in nodelab_v2.scene.HIDDEN_OP_PREFIXES), so
    # a user can drag it onto the canvas — but `channels` had no socket, leaving them a
    # card with no controls that silently passes every channel through. The GUI's
    # per-channel tap materializer writes a LIST here; a user types "0,2".
    inputs=[InDataset(),
            InString("channels", "Channels", field=False, default="",
                     pick_kind="channels",
                     description=
                     "Which channels to keep, as 0-based indices — \"0,2\" keeps the first "
                     "and third and DROPS the rest, narrowing the channel axis to 2. Order "
                     "is honoured, so \"2,0\" also reorders. Empty = keep everything "
                     "unchanged. Out-of-range indices are ignored rather than raising. "
                     "Per-channel calibration (emission wavelengths) is subset in lockstep, "
                     "so downstream optics-derived defaults stay correct — and because the "
                     "channel axis shrinks, layers whose shape no longer fits are dropped "
                     "from the bundle.")],
    outputs=[OutDataset()],
    granularity=Granularity.TILEABLE, meta_transform=_meta_channel_select,
    description="Subset/reorder the channel axis (metadata follows in lockstep).",
)
