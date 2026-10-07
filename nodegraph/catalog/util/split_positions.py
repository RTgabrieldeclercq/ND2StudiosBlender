"""Split Positions (``util.split_positions``) — Fan a multipoint Dataset out into one output
per stage position (each a single-position Dataset); the whole set also passes through
``out``."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node


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
            InString("groups", "Groups", field=False, default="", presentation=True,
                     description=
                     "Split into RANGES instead of one output per position: type the groups as "
                     "0-based position indices, inclusive ranges, `;` between groups — `0-3; "
                     "4-7; 8-11` — with an optional name in front of a group (`top: 0-3; mid: "
                     "4-7`), or `every 10` for consecutive chunks of ten. The card then grows "
                     "one output PER GROUP, each carrying exactly that subset of the M "
                     "(position) axis, and the per-position outputs are put away. Blank = one "
                     "output per position, up to 24 of them; past that the card offers only "
                     "`out`, and this is how to split. A wired group materializes into a Crop "
                     "in frames mode keeping those positions at run time, shared by every "
                     "branch that reads it, so this never changes `out` or anything this node "
                     "itself computes. A group reaching past the end is labelled so on the "
                     "card; what lies past the end is dropped at the pull, and a group "
                     "entirely past it is refused with the real M (position) length.")],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE,
    description="Fan a multipoint Dataset out into one output per stage position (each a "
                "single-position Dataset) — or, with Groups set, one output per RANGE of "
                "positions; the whole set also passes through 'out'. The positions "
                "axis's twin of Split Channels.",
)
