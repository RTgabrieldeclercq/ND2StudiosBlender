"""ndimage — shared catalog helpers."""

from __future__ import annotations

import nodegraph.gpu as _gpu
import numpy as np

from typing import Any

def _ndi(name: str, arr: np.ndarray, **kw: Any) -> np.ndarray:
    """``scipy.ndimage.<name>(arr, **kw)`` with an optional CUDA fast path (V2.14).

    :func:`nodegraph.gpu.ndimage` returns ``None`` whenever the GPU is not the right
    answer — no CuPy, no device, the op has not reproduced scipy on this machine, the array
    is too small to pay for the PCIe round trip, or device memory is short — so this
    degrades to the identical scipy call rather than to an error. Use it for the
    ``ndimage`` kernels inside a ``_map_image`` plane/volume function; keep calling scipy
    directly where the op has no ``cupyx`` counterpart."""
    out = _gpu.ndimage(name, np.asarray(arr), **kw)
    if out is not None:
        return out
    from scipy import ndimage as _sndi
    return getattr(_sndi, name)(arr, **kw)
