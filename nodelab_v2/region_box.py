"""The troubleshooting REGION box — the maths behind the draggable amber rectangle the
Viewer shows while the solo-frame scope (F9) is armed (2026-10-02).

Qt-free on purpose, like :mod:`nodelab_v2.picker`: everything here is plain geometry on a
``(y0, y1, x0, x1)`` half-open window in SOURCE pixel coordinates, so ``nodegraph.selftest``
can test it headless and the Viewer only has to translate widget points in and out.

A region is ``None`` when it covers the whole source frame. That is not a convenience: the
runner's pin carries the region, the pin keys every memo entry, and a full-frame region
must be the SAME identity as "no region" or arming the box would re-key every result the
user already has.

Handles are named by compass point (``n``, ``ne``, ``e`` …) plus ``move`` for the interior.
"""
from __future__ import annotations

from typing import Optional, Tuple

#: ``(y0, y1, x0, x1)``, half-open, source pixels.
Region = Tuple[int, int, int, int]
#: ``(Y, X)`` of the source frame.
Extent = Tuple[int, int]

#: Smallest side a dragged region may have, in source pixels. Below this a kernel's halo
#: is the whole window and the result says nothing about the parameters.
MIN_SIDE = 8

HANDLES: Tuple[str, ...] = ("nw", "n", "ne", "w", "e", "sw", "s", "se")


def full(extent: Extent) -> Region:
    return (0, int(extent[0]), 0, int(extent[1]))


def clamp(region: Optional[Region], extent: Extent,
          min_side: int = MIN_SIDE) -> Optional[Region]:
    """``region`` clipped into ``extent`` and floored at ``min_side``; ``None`` when it
    covers the whole frame (or when there is nothing to clamp to).

    Clamping rather than refusing: the source can shrink under a remembered region (a
    graph edit, a different node viewed), and the nearest real window is more useful than
    an error on the next pull."""
    if region is None or not extent:
        return None
    Y, X = max(1, int(extent[0])), max(1, int(extent[1]))
    y0, y1, x0, x1 = (int(round(v)) for v in region)
    y0, y1 = sorted((max(0, min(y0, Y)), max(0, min(y1, Y))))
    x0, x1 = sorted((max(0, min(x0, X)), max(0, min(x1, X))))
    ms_y, ms_x = min(min_side, Y), min(min_side, X)
    if y1 - y0 < ms_y:
        y1 = min(Y, y0 + ms_y)
        y0 = max(0, y1 - ms_y)
    if x1 - x0 < ms_x:
        x1 = min(X, x0 + ms_x)
        x0 = max(0, x1 - ms_x)
    if (y0, y1, x0, x1) == (0, Y, 0, X):
        return None
    return (y0, y1, x0, x1)


def resolve(region: Optional[Region], extent: Extent) -> Region:
    """The window a pull evaluates: ``region`` or, for ``None``, the whole frame."""
    return clamp(region, extent) or full(extent)


def hit(region: Region, x: float, y: float, tol: float) -> Optional[str]:
    """Which handle a point ``(x, y)`` (same coordinate space as ``region``) is on, within
    ``tol``: a corner first, then an edge, then ``move`` for the interior, else ``None``.
    ``tol`` is the grab radius already converted to this space by the caller — the Viewer
    passes a few screen pixels' worth, so the grab zone stays constant under zoom."""
    y0, y1, x0, x1 = region
    near_w, near_e = abs(x - x0) <= tol, abs(x - x1) <= tol
    near_n, near_s = abs(y - y0) <= tol, abs(y - y1) <= tol
    in_x, in_y = x0 - tol <= x <= x1 + tol, y0 - tol <= y <= y1 + tol
    if not (in_x and in_y):
        return None
    if near_n and near_w:
        return "nw"
    if near_n and near_e:
        return "ne"
    if near_s and near_w:
        return "sw"
    if near_s and near_e:
        return "se"
    if near_n:
        return "n"
    if near_s:
        return "s"
    if near_w:
        return "w"
    if near_e:
        return "e"
    if x0 < x < x1 and y0 < y < y1:
        return "move"
    return None


def drag(start: Region, handle: str, dx: float, dy: float, extent: Extent,
         min_side: int = MIN_SIDE) -> Region:
    """``start`` after dragging ``handle`` by ``(dx, dy)`` source pixels, kept inside
    ``extent``. A ``move`` keeps the size and slides (stopping at the frame edge, so a
    region can never leave the source); an edge or corner moves only its own sides and
    never crosses the opposite one — the window stays at least ``min_side`` wide, so a
    drag past the far edge pins rather than flips."""
    Y, X = max(1, int(extent[0])), max(1, int(extent[1]))
    y0, y1, x0, x1 = start
    ms_y, ms_x = min(min_side, Y), min(min_side, X)
    if handle == "move":
        h, w = y1 - y0, x1 - x0
        ny0 = int(round(max(0, min(y0 + dy, Y - h))))
        nx0 = int(round(max(0, min(x0 + dx, X - w))))
        return (ny0, ny0 + h, nx0, nx0 + w)
    if "n" in handle:
        y0 = int(round(max(0, min(y0 + dy, y1 - ms_y))))
    if "s" in handle:
        y1 = int(round(max(y0 + ms_y, min(y1 + dy, Y))))
    if "w" in handle:
        x0 = int(round(max(0, min(x0 + dx, x1 - ms_x))))
    if "e" in handle:
        x1 = int(round(max(x0 + ms_x, min(x1 + dx, X))))
    return (y0, y1, x0, x1)


def label(region: Optional[Region], extent: Optional[Extent] = None) -> str:
    """The region in a few characters for the status chip — ``64×48@y32,x16`` — or an
    empty string for the whole frame."""
    if region is None:
        return ""
    y0, y1, x0, x1 = region
    if extent and clamp(region, extent) is None:
        return ""
    return f"{x1 - x0}×{y1 - y0}@y{y0},x{x0}"
