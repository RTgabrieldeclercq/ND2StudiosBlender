"""global scalar — shared catalog helpers."""

from __future__ import annotations

import numpy as np

from typing import Optional

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain

def _global_scalar(ds: Dataset, name: str) -> Optional[float]:
    layer = ds.get(Domain.GLOBAL, name) if name else None
    if layer is None:
        return None
    flat = np.asarray(layer.values, dtype=float).reshape(-1)
    return float(flat[0]) if flat.size else None
