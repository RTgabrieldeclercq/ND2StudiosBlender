"""planes — shared catalog helpers."""

from __future__ import annotations

import itertools

from nodegraph.dataset import AxisSizes
from nodegraph.engine import EvalContext

def _each_plane(ax: AxisSizes):
    """Iterate (m,t,z,c) over a dataset's acquisition axes."""
    return itertools.product(range(ax.m), range(ax.t), range(ax.z), range(ax.c))
def _each_plane_p(ctx: EvalContext, ax: AxisSizes, note: str = ""):
    """:func:`_each_plane` that reports **per-node progress** as it goes (``done`` counted
    on completion of each plane, so the bar reflects finished work, not started work).

    Reports **two levels** — ``ax.t`` is handed over as the frame count, so the UI draws
    the frame/total-frames bar and the within-this-frame bar off one per-plane tick. That
    is why every eager node gets both bars without touching its loop.

    Use in an **eager** compute — one that realizes planes inside the call. A compute
    returning a lazy provider must not report a fraction it isn't paying: an unreported
    node is how the UI knows the cost is deferred to read time (see
    :data:`nodegraph.engine.Observer`)."""
    n = ax.m * ax.t * ax.z * ax.c
    ctx.progress(0, n, note, frames=ax.t)
    for i, unit in enumerate(_each_plane(ax)):
        yield unit
        ctx.progress(i + 1, n, note, frames=ax.t)
def _each_volume_p(ctx: EvalContext, ax: AxisSizes, note: str = ""):
    """As :func:`_each_plane_p`, but over ``(m, t, c)`` volumes — the unit of a
    WHOLE_VOLUME compute."""
    n = ax.m * ax.t * ax.c
    ctx.progress(0, n, note, frames=ax.t)
    for i, unit in enumerate(itertools.product(range(ax.m), range(ax.t), range(ax.c))):
        yield unit
        ctx.progress(i + 1, n, note, frames=ax.t)
