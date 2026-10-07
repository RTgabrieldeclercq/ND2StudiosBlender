"""drift layers — shared catalog helpers."""

from __future__ import annotations


from nodegraph.domains import Domain

def _layers_drift(params, modes):
    """`align.drift` / `registration.stabilize` store the estimated per-frame shift as
    two LITERAL Frame layers — no socket names them, so nothing else can see them."""
    return ((Domain.FRAME, "drift_y"), (Domain.FRAME, "drift_x"))


def _layers_stabilize(params, modes):
    """``registration.stabilize`` stores the per-frame shift (``drift_y``/``drift_x``, plus
    ``drift_z`` under the 3D lever) and the per-frame ``drift_confidence`` — the NCC / ECC
    coefficient / RANSAC inlier fraction the estimate scored, so a frame whose correction
    is not believed can be found in a table rather than by eye (2026-10-07)."""
    layers = [(Domain.FRAME, "drift_y"), (Domain.FRAME, "drift_x")]
    if (modes or {}).get("dim") == "3D":
        layers.insert(0, (Domain.FRAME, "drift_z"))
    layers.append((Domain.FRAME, "drift_confidence"))
    return tuple(layers)


def _layers_shift(params, modes):
    """``align.shift`` stores the per-frame shift it APPLIED as ``shift_y`` / ``shift_x``
    (plus ``shift_z`` under the 3D lever when a z layer is named) — the literal names the
    demo window and a table read back, so the apply half of registration is as inspectable
    as the estimate half (2026-10-07). Total: a blank or absent ``shift_z`` is simply no
    axial layer."""
    layers = [(Domain.FRAME, "shift_y"), (Domain.FRAME, "shift_x")]
    if (modes or {}).get("dim") == "3D" and str((params or {}).get("shift_z") or "").strip():
        layers.insert(0, (Domain.FRAME, "shift_z"))
    return tuple(layers)
