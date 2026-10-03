"""Drawn-region helpers shared by the nodes that rasterize a ``shapes`` socket.

A drawn shape is stored in **full-frame source pixels** — the viewer shifts a shape drawn
inside a troubleshooting window back into the frame when the pick is committed — so the
list means the same thing whether or not a window was active when it was drawn. The
compute that rasterizes it, however, may be running ON a window: under the Viewer's
troubleshooting region the runner serves every node a :class:`~nodegraph.streaming.WindowView`
of the source and stamps where that window sits (:data:`WINDOW_ORIGIN_KEY`, ``[y0, x0]``
in source pixels). Rasterizing full-frame coordinates into a window-sized frame puts the
shape ``(y0, x0)`` too far down and right, or off the frame altogether — the bug reported
on 2026-10-02 — so every shapes consumer shifts the list into the window first, through
:func:`shapes_in_frame`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Metadata key the runner stamps on a windowed seed: ``[y0, x0]``, the window's top-left
#: corner in SOURCE pixels. Absent (or ``[0, 0]``) means the frame is the whole source.
#: Non-calibration provenance like the placement keys; it rides ``ds.metadata`` unchanged
#: through computes that keep the geometry.
WINDOW_ORIGIN_KEY = "window_origin_px"


def window_origin_px(metadata: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    """``(y0, x0)`` the current frame is offset from the full source, or ``(0, 0)``."""
    v = (metadata or {}).get(WINDOW_ORIGIN_KEY)
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return 0.0, 0.0
    try:
        return float(v[0]), float(v[1])
    except (TypeError, ValueError):
        return 0.0, 0.0


def shapes_in_frame(shapes: Optional[Sequence[Dict[str, Any]]],
                    metadata: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The shape list moved into the frame the compute is running on: every vertex and
    centre has the window origin subtracted. Shapes outside the window simply rasterize to
    nothing, which is what a region elsewhere in the frame should do here. Returns new
    dicts; the stored list (the node's own param) is never mutated."""
    oy, ox = window_origin_px(metadata)
    out: List[Dict[str, Any]] = []
    for sh in shapes or ():
        if not isinstance(sh, dict):
            continue
        if not (oy or ox):
            out.append(sh)
            continue
        moved = dict(sh)
        verts = sh.get("vertices")
        if isinstance(verts, (list, tuple)):
            moved["vertices"] = [[float(v[0]) - oy, float(v[1]) - ox] for v in verts
                                 if isinstance(v, (list, tuple)) and len(v) >= 2]
        centre = sh.get("center")
        if isinstance(centre, (list, tuple)) and len(centre) >= 2:
            moved["center"] = [float(centre[0]) - oy, float(centre[1]) - ox]
        out.append(moved)
    return out
