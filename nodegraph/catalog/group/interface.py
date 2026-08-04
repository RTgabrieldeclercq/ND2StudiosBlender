"""group.interface (``group.interface``) — Group interface marker (pass-through);."""

from __future__ import annotations


from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.passthrough import _compute_zone_passthrough

# ── group boundary nodes (Group Input / Output — identity pass-through) ─────────
#
# The paired interface markers of a node group (V2.00; Phase 4b). Pure pass-throughs:
# the compute returns the input Dataset unchanged. The group's inline-expand semantics
# (replace an instance node with a fresh unique-id copy of the body + stitch the single
# DATASET interface) live in :mod:`nodegraph.groups`; these nodes only mark the
# interface input/output that ``expand()`` wires through.

for _op, _label in (("group.input", "Group Input"), ("group.output", "Group Output")):
    register_node(
        _compute_zone_passthrough, op_key=_op, label=_label, category="group",
        inputs=[InDataset()], outputs=[OutDataset()],
        granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
        description="Group interface marker (pass-through); paired Input/Output bound a "
                    "reusable subgraph — see nodegraph.groups.")
