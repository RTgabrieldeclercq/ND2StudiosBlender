"""Renderer maths for the scalar, vector and comparison overlays — moved to
:mod:`nodegraph.catalog._shared.overlay_math` (2026-11-03, V5.00 step 1).

The figure core (``nodegraph/figure``) colour-maps fields and decimates quivers headless,
and a node cannot import the GUI package, so the maths went down to the engine — the same
move :mod:`nodegraph.tables` made for the spreadsheet. Every name is re-exported here
unchanged, so :mod:`nodelab_v2.overlays` and the tests keep their imports and there is
still exactly one implementation.
"""
from __future__ import annotations

from nodegraph.catalog._shared.overlay_math import (  # noqa: F401 — re-exported
    COLORMAPS, DIFF_COLORS, _DIVERGING, colormap_lut, match_labels, match_points,
    normalize_for_map, quiver_arrows, scalar_rgba)

__all__ = [
    "COLORMAPS", "colormap_lut", "normalize_for_map", "scalar_rgba",
    "quiver_arrows", "match_points", "match_labels", "DIFF_COLORS",
]
