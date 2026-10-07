"""Split Positions (``util.split_positions``) — Fan a multipoint Dataset out into one output
per stage position (each a single-position Dataset); the whole set also passes through
``out``."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.split_grouping import grouping_mode, grouping_sockets


def _compute_split_positions(ctx: EvalContext) -> Dataset:
    """Split Positions — a domain-transparent pass-through of the full multipoint bundle,
    the M-axis twin of ``channel.split`` (2026-10-02).

    Its single real ``out`` socket carries the input unchanged. The GUI adds one synthetic
    per-position output socket (``pos0…posM-1``, labelled with the acquisition's point name
    where the file carries one, else ``m0…``) once the wire carries two or more positions,
    and each wired per-position tap is materialized into a real ``util.select_position`` at
    graph-build time — the engine is one-payload-per-node, so distinct per-position payloads
    come from real select taps, not from this node. No calibration or axes change here
    (``TILEABLE``, no meta_transform). Drop it after a multipoint Load (or after a Timeseries
    Builder laid files onto M) to run a different branch per well, dish or field.
    """
    return ctx.inputs[0]


register_node(
    _compute_split_positions,
    op_key="util.split_positions", label="Split Positions", category="utility",
    inputs=[InDataset(description=
                      "The multipoint Dataset to fan out. Its positions appear as one "
                      "output socket each on the card, labelled with the file's point names "
                      "when it has them; `out` still carries the whole set."),
            *grouping_sockets('position', 'M (position)',
                              'a Crop in frames mode keeping those positions', 2)],
    outputs=[OutDataset("out")],
    modes=[grouping_mode('position', 'M (position)')],
    granularity=Granularity.TILEABLE,
    description="Fan a multipoint Dataset out into one output per stage position (each a "
                "single-position Dataset) — or, with Groups set, one output per RANGE of "
                "positions; the whole set also passes through 'out'. The positions "
                "axis's twin of Split Channels.",
)
