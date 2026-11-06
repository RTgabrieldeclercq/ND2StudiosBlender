"""Planes for the HTML viewer: read at a pixel budget, quantised to 16 bits, packed.

:func:`read_plane` is the Qt-free twin of the figure editor's
:func:`nodelab_v2.figure_editor.render.read_plane` without its window: the coarsest pyramid
level that still has a side at or above the budget, strided down (never interpolated) when
that level is still above the budget — so a label raster's ids survive and a 500-frame
export reads 500 small planes.

:func:`quantize` maps any plane onto the uint16 the PNG packing carries and records the
affine ``value = code * scale + offset`` that the JavaScript undoes, so the hover readout
and the LUT handles work in the data's own units: an integer plane up to 16 bits is exact
(scale 1, offset 0), a wider integer or a float plane is spread over the 16-bit range.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nodegraph.htmlview.png import data_uri, plane_png

#: the longest side of an exported plane, by default — the Viewer's display budget
DISPLAY_MAX_PX = 1024


def pick_level(prov: Any, max_px: int) -> Tuple[int, Any]:
    """The coarsest pyramid level whose plane still has a side ≥ ``max_px`` (or level 0)."""
    levels = int(getattr(prov, "levels", 1) or 1)
    best = 0
    for lv in range(levels):
        ax = prov.level_axes(lv)
        if max(int(ax.y), int(ax.x)) >= int(max_px):
            best = lv
        else:
            break
    return best, prov.level_axes(best)


def read_plane(ds: Any, m: int, t: int, z: Any, c: int, *, max_px: int = DISPLAY_MAX_PX
               ) -> Tuple[np.ndarray, int]:
    """``(plane, stride)`` for channel ``c`` at ``(m, t, z)``, ``z`` an index or ``max`` /
    ``mean`` (a projection over every plane), at the level the budget picks; ``stride``
    is the whole-pixel step taken on top of that level (1 = none)."""
    prov = getattr(ds, "image", None)
    if prov is None or not hasattr(prov, "get_region"):
        raise ValueError("the source carries no image")
    level, lax = pick_level(prov, max_px)
    H, W = int(lax.y), int(lax.x)
    nz = int(getattr(lax, "z", 1) or 1)
    if z in ("max", "mean"):
        acc = None
        for zi in range(nz):
            a = np.asarray(prov.get_region(level, m, t, zi, c, 0, H, 0, W))
            a = a.astype(np.float64) if z == "mean" else a
            if acc is None:
                acc = np.array(a, copy=True)
            elif z == "max":
                np.maximum(acc, a, out=acc)
            else:
                acc += a
        if acc is None:
            acc = np.zeros((H, W))
        elif z == "mean" and nz > 0:
            acc /= nz
        plane = acc
    else:
        zi = min(max(int(z), 0), max(0, nz - 1))
        plane = np.asarray(prov.get_region(level, m, t, zi, c, 0, H, 0, W))
    # the budget is a CEILING here (the figure render tolerates twice it before striding):
    # the user chose max_px for the file's size, so a 1024 px plane at 512 is strided by 2
    longest = max(plane.shape[0], plane.shape[1]) if plane.size else 0
    k = 1
    if longest > int(max_px):
        k = int(math.ceil(longest / float(max_px)))
        plane = np.ascontiguousarray(plane[::k, ::k])
    return plane, k


def quantize(plane: np.ndarray, *, nan0: bool = False) -> Tuple[np.ndarray, float, float, bool]:
    """``(codes uint16, scale, offset, nan0)``: ``value = code * scale + offset``.

    With ``nan0`` a NaN (a field raster's hole) is coded 0 and the data starts at code 1,
    so the viewer can tell a hole from the lowest value; without it NaN becomes the
    minimum. A boolean plane is 0 / 1."""
    a = np.asarray(plane)
    if a.dtype == np.bool_:
        return a.astype(np.uint16), 1.0, 0.0, False
    if np.issubdtype(a.dtype, np.integer):
        lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
        if 0 <= lo and hi <= 65535 and not nan0:
            return a.astype(np.uint16), 1.0, 0.0, False
        f = a.astype(np.float64)
    else:
        f = a.astype(np.float64)
    finite = np.isfinite(f)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint16), 1.0, 0.0, bool(nan0)
    lo, hi = float(f[finite].min()), float(f[finite].max())
    span = hi - lo
    top = 65534.0 if nan0 else 65535.0
    scale = (span / top) if span > 0 else 1.0
    codes = np.zeros(a.shape, dtype=np.float64)
    codes[finite] = np.clip(np.round((f[finite] - lo) / scale), 0, top)
    if nan0:
        codes[finite] += 1.0
        offset = lo - scale
    else:
        offset = lo
    return codes.astype(np.uint16), scale, offset, bool(nan0)


def plane_record(plane: np.ndarray, *, nan0: bool = False) -> Dict[str, Any]:
    """One embedded plane: its data URI, encoding and the affine back to data units."""
    codes, scale, offset, is_nan0 = quantize(plane, nan0=nan0)
    png, enc = plane_png(codes)
    rec: Dict[str, Any] = {"png": data_uri(png), "enc": enc, "w": int(codes.shape[1]),
                           "h": int(codes.shape[0])}
    if scale != 1.0 or offset != 0.0:
        rec["scale"] = float(scale)
        rec["offset"] = float(offset)
    if is_nan0:
        rec["nan0"] = True
    return rec


def record_bytes(rec: Dict[str, Any]) -> int:
    return len(rec.get("png", ""))


def dtype_range(ds: Any, plane: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """The full data range a LUT's handles clamp to: the bit depth the metadata declares,
    else the plane's own dtype, else its values."""
    md = getattr(ds, "metadata", None) or {}
    bd = md.get("bit_depth")
    try:
        bd = int(bd) if bd else 0
    except (TypeError, ValueError):
        bd = 0
    if 1 <= bd <= 32:
        return 0.0, float(2 ** bd - 1)
    if plane is not None:
        a = np.asarray(plane)
        if np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_:
            info = np.iinfo(a.dtype)
            return float(max(0, info.min)), float(min(info.max, 65535))
        if a.size:
            finite = a[np.isfinite(a)]
            if finite.size:
                return float(finite.min()), float(finite.max())
    return 0.0, 1.0


__all__ = ["DISPLAY_MAX_PX", "pick_level", "read_plane", "quantize", "plane_record",
           "record_bytes", "dtype_range"]
