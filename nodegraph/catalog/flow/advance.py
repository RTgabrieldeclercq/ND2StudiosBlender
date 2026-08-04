"""Search Step (``flow.advance``) — One step of an Iterate node's feedback search…"""

from __future__ import annotations


from typing import List

from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutValue
from nodegraph.sockets import SocketType

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.global_scalar import _global_scalar

# ── flow.advance — the hidden per-probe search step (feedback mode only) ────────
#
# Minted by `nodegraph.iterate.unroll`, never placed by a user, hidden from the palette.
# It exists because a feedback value is a RESULT and so can only reach its clone as a
# payload; this is the node that computes it. Every param is machine-set and dunder-named
# for the same reason `zone.frame`'s `__frame__` is — bookkeeping, not a control, so the
# param/socket contract exempts it and no socket offers it for editing.


def _compute_advance(ctx: EvalContext) -> float:
    """The next parameter value to probe, from every earlier probe's metric.

    **Stateless by necessity.** The engine is one-payload-per-node, so this cannot emit
    both a value and a carried bracket — it re-derives the whole search by replaying
    :func:`nodegraph.iterate.advance_value` forward over the metrics it can see. Iteration
    *i* reads probes 0..i-1, so the graph is O(N²) edges of pure arithmetic: free beside one
    segmentation, and it makes each step a pure function of its inputs, which is what the
    memo requires."""
    from nodegraph.iterate import advance_value
    probes = ctx.input("probes") or ()
    if not isinstance(probes, tuple):
        probes = (probes,)
    name = str(ctx.params.get("__metric__", "") or "")
    metrics: List[float] = []
    for ds in probes:
        value = _global_scalar(ds, name)
        if value is None:
            raise ValueError(
                f"Iterate (feedback): an iteration produced no Global scalar {name!r} to "
                f"steer by — the search has nothing to follow. Put a 'Reduce → Scalar' "
                f"node inside the iterated chain and name it in the Iterate node's Metric.")
        metrics.append(value)
    search = str(ctx.params.get("__search__", "golden"))
    lo = float(ctx.params.get("__lo__", 0.0) or 0.0)
    hi = float(ctx.params.get("__hi__", 1.0) or 1.0)
    direction = str(ctx.params.get("__direction__", "max"))
    target = float(ctx.params.get("__target__", 0.0) or 0.0)
    xs: List[float] = []
    for k in range(len(metrics)):
        xs.append(advance_value(search, lo, hi, direction=direction, target=target,
                                probes=tuple(zip(xs[:k], metrics[:k]))))
    return advance_value(search, lo, hi, direction=direction, target=target,
                         probes=tuple(zip(xs, metrics)))
register_node(
    _compute_advance, op_key="flow.advance", label="Search Step", category="flow",
    inputs=[InDataset("probes", multi=True, label="Probes")],
    outputs=[OutValue("value", SocketType.FLOAT, "Value")],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="One step of an Iterate node's feedback search (hidden; minted by the "
                "iterate rewrite, never placed by hand).")
