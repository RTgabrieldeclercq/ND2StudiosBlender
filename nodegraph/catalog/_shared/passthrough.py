"""passthrough — shared catalog helpers."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext

def _compute_zone_passthrough(ctx: EvalContext) -> Dataset:
    return ctx.inputs[0]
