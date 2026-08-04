"""Reroute (``rr.reroute``) — an identity pass-through used to route wires on the canvas."""
from __future__ import annotations

from nodegraph.catalog._base import register_node
from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset


def _compute_reroute(ctx: EvalContext) -> Dataset:
    """Reroute — an identity pass-through of the Dataset on the wire (Blender's reroute
    node). Purely a wire-routing convenience for the GUI: it returns its input unchanged,
    changes no axes/calibration (TILEABLE, no meta_transform, identity envelope), so the
    engine and memo treat it as a transparent hop. Hidden from the palette (``rr.``
    prefix); created by double-clicking a wire in the canvas."""
    return ctx.inputs[0]


register_node(
    _compute_reroute,
    op_key="rr.reroute", label="Reroute", category="general",
    inputs=[InDataset()], outputs=[OutDataset()],
    granularity=Granularity.TILEABLE,
    description="Identity pass-through used to route wires cleanly on the canvas.",
)
