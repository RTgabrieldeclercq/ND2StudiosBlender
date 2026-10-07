"""Split Z (``util.split_z``) — Fan a z-stack out into one output per plane (each a
single-plane Dataset); the whole stack also passes through ``out``."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.split_grouping import grouping_mode, grouping_sockets


def _compute_split_z(ctx: EvalContext) -> Dataset:
    """Split Z — a domain-transparent pass-through of the full stack, the Z-axis twin of
    ``channel.split`` and ``util.split_positions`` (2026-10-07).

    Its single real ``out`` socket carries the input unchanged. The GUI adds one synthetic
    per-plane output socket (``z0…zZ-1``, labelled with the plane's index and, when the stack
    carries a z step, its height above plane 0) once the wire carries two or more planes,
    and each wired per-plane tap is materialized into a real ``util.select_plane`` at
    graph-build time — the engine is one-payload-per-node, so distinct per-plane payloads
    come from real select taps, not from this node. No calibration or axes change here
    (``TILEABLE``, no meta_transform). Drop it after a Load to run a different branch per
    plane, or to hand ONE plane to Registration while the full stack goes on to Shift.
    """
    return ctx.inputs[0]


register_node(
    _compute_split_z,
    op_key="util.split_z", label="Split Z", category="utility",
    inputs=[InDataset(description=
                      "The z-stack to fan out. Its planes appear as one output socket each "
                      "on the card (`z0…`, with the plane's height when the stack has a z "
                      "step); `out` still carries the whole stack."),
            *grouping_sockets('plane', 'Z',
                              'a Crop in frames mode keeping those planes', 4)],
    outputs=[OutDataset("out")],
    modes=[grouping_mode('plane', 'Z')],
    granularity=Granularity.TILEABLE,
    description="Fan a z-stack out into one output per plane (each a single-plane Dataset, "
                "zK = plane K) — or, with Groups set, one output per RANGE of planes (a "
                "sub-stack each); the whole stack also passes through 'out'. The Z "
                "axis's twin of Split Channels and Split Positions.",
)
