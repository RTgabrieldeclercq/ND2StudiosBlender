"""zone.boundary (``zone.boundary``) — Zone boundary marker (pass-through);."""

from __future__ import annotations


from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.passthrough import _compute_zone_passthrough

for _op, _label in (("zone.repeat_in", "Repeat In"), ("zone.repeat_out", "Repeat Out"),
                    ("zone.sim_in", "Sim In"), ("zone.sim_out", "Sim Out")):
    register_node(
        _compute_zone_passthrough, op_key=_op, label=_label, category="zone",
        inputs=[InDataset()], outputs=[OutDataset()],
        granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
        description="Zone boundary marker (pass-through); paired In/Out bound the "
                    "iterated body — see nodegraph.zones.")
