"""Split Channels (``channel.split``) — Fan a multi-channel Dataset out into per-channel outputs (each a single-channel Dataset);."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node

def _compute_split_channels(ctx: EvalContext) -> Dataset:
    """Split Channels — a domain-transparent pass-through of the full multi-channel
    bundle. Its single real ``out`` socket carries the input unchanged; the GUI adds
    one synthetic per-channel output socket (``ch0…chN-1``), and each wired per-channel
    tap is materialized into a ``channel.select`` at graph-build time (the engine is
    one-payload-per-node, so distinct per-channel payloads come from real select taps,
    not from this node). No calibration/axes change here (TILEABLE, no meta_transform)."""
    return ctx.inputs[0]
register_node(
    _compute_split_channels,
    op_key="channel.split", label="Split Channels", category="channel",
    inputs=[InDataset(),
            InString("groups", "Groups", field=False, default="", presentation=True,
                     description=
                     "Split into RANGES instead of one output per channel: type the groups as "
                     "0-based channel indices, inclusive ranges, `;` between groups — `0-3; "
                     "4-7; 8-11` — with an optional name in front of a group (`top: 0-3; mid: "
                     "4-7`). The card then grows one output PER GROUP, each carrying exactly "
                     "that subset of the channel axis, and the per-channel outputs are put "
                     "away. Blank = one output per channel, as before. A wired group "
                     "materializes into a Select Channel with that list at run time, shared by "
                     "every branch that reads it, so this never changes `out` or anything this "
                     "node itself computes; a group that reaches past the end is labelled so "
                     "on the card and refused with the real channel length when pulled.")],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE,
    description="Fan a multi-channel Dataset out into per-channel outputs (each a "
                "single-channel Dataset) — or, with Groups set, into one output per "
                "RANGE of channels; the full bundle also passes through 'out'.",
)
