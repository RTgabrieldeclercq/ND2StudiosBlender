"""Split Channels (``channel.split``) — Fan a multi-channel Dataset out into per-channel outputs (each a single-channel Dataset);."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset

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
    inputs=[InDataset()], outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE,
    description="Fan a multi-channel Dataset out into per-channel outputs (each a "
                "single-channel Dataset); the full bundle also passes through 'out'.",
)
