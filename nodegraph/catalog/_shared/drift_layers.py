"""drift layers — shared catalog helpers."""

from __future__ import annotations


from nodegraph.domains import Domain

def _layers_drift(params, modes):
    """`align.drift` / `registration.stabilize` store the estimated per-frame shift as
    two LITERAL Frame layers — no socket names them, so nothing else can see them."""
    return ((Domain.FRAME, "drift_y"), (Domain.FRAME, "drift_x"))
