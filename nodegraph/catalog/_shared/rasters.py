"""rasters — shared catalog helpers for a Voxel raster read off ANOTHER wire (V4.00 step 12).

A mask made on one branch and used on another (Mask Math's ``other``, Crop by Region's
``regions``) rarely has exactly the primary's shape: it was made on one channel of a
three-channel image, or on one timepoint. The rule both nodes apply is here, once: a size-1
``m``/``t``/``z``/``c`` axis stretches over the primary's, ``y``/``x`` must match exactly, and
an axis LONGER than the primary's is refused with the axes named — a 3-channel mask cannot
say which channel it means on a 1-channel image.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["broadcast_raster"]

_AXES = "mtzcyx"


def broadcast_raster(values: Any, shape: Sequence[int], *, node: str, what: str) -> np.ndarray:
    """``values`` (a ``(m,t,z,c,y,x)`` raster) as a BOOLEAN array of ``shape`` — nonzero is
    inside. Read-only broadcast view; copy before writing."""
    a = np.asarray(values)
    shape = tuple(int(s) for s in shape)
    if a.ndim != len(shape):
        raise ValueError(f"{node}: {what} is {a.ndim}-D, expected a (m,t,z,c,y,x) raster")
    bad = [name for name, have, want in zip(_AXES, a.shape, shape)
           if have != want and not (have == 1 and name in "mtzc")]
    if bad:
        raise ValueError(
            f"{node}: {what} has shape {tuple(a.shape)} but the data is {shape} — the axes "
            f"{bad} disagree. A raster from another wire may be narrower than the data only "
            f"on m/t/z/c (size 1 applies to every index); y and x must match, so crop or "
            f"resample the two branches the same way, or make the mask on this branch.")
    return np.broadcast_to(a != 0, shape)
