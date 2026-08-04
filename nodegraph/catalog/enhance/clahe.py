"""CLAHE (``enhance.clahe``) — Contrast-limited adaptive histogram equalization; 2D per-plane vs 3D."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InDataset, InFloat, InInt, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.map_image import _map_image

# ── CLAHE (local contrast) ──────────────────────────────────────────────────────

def _compute_clahe(ctx: EvalContext) -> Dataset:
    """Contrast-limited adaptive histogram equalization.

    ``tile_grid`` is the number of contextual regions **per axis** — CLAHE equalizes each
    region's own histogram, so this is the knob that decides how local "local contrast" is.
    It was pinned to 4 before V2.13 (a bare ``s // 4``), which made it an invisible
    constant no user could reach; the default is now 8, matching the ``tileGridSize=(8,8)``
    of Cell-Tracker's OpenCV CLAHE plugin.

    Note the ``clip_limit`` scales differ and cannot be carried across: scikit-image takes
    a **fraction of the region's pixel count** (0–1, default 0.01), OpenCV a per-bin count
    multiplier (Cell-Tracker's default 2.0). A recipe's number is not portable."""
    from skimage.exposure import equalize_adapthist as clahe
    ds = ctx.inputs[0]
    clip = float(ctx.params.get("clip_limit", 0.01))
    grid = max(1, int(ctx.params.get("tile_grid", 8)))

    def apply(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return a.copy()
        # per-axis kernel = extent / regions-per-axis, floored at 2 (3D-safe: a z of 3
        # over an 8-region grid would otherwise ask for a 0-thick kernel)
        ks = tuple(max(2, s // grid) for s in a.shape)
        eq = clahe((a - mn) / (mx - mn), kernel_size=ks, clip_limit=clip)
        return eq * (mx - mn) + mn

    return _map_image(ctx, ds, plane_fn=apply, volume_fn=apply)
register_node(
    _compute_clahe, op_key="enhance.clahe", label="CLAHE", category="enhancement",
    inputs=[InDataset(), InFloat("clip_limit", "Clip limit", unit="", field=True,
                                 default=0.01,
                                 description=
                                 "Ceiling on how much local contrast may be amplified, in "
                                 "[0,1]. HIGHER reveals more faint structure and amplifies "
                                 "noise with it, and in near-empty regions it stretches pure "
                                 "background into visible texture; LOWER stays closer to the "
                                 "original. 0.01 is conservative, 0.03–0.05 is assertive. "
                                 "Note this is scikit-image's scale — a FRACTION of each "
                                 "region's pixel count — not OpenCV's per-bin multiplier, so "
                                 "a Cell-Tracker recipe's 2.0 does not transfer. This is a "
                                 "strongly NON-LINEAR, spatially-varying intensity change: "
                                 "two pixels of equal brightness in different regions come "
                                 "out different, so never quantify intensity downstream."),
            InInt("tile_grid", "Tile grid", unit="", field=False, default=8,
                  description=
                  "How many contextual regions to divide each axis into — this is what makes "
                  "the equalization LOCAL. Each region's own histogram is equalized and the "
                  "results interpolated between region centres, so MORE regions means a "
                  "smaller neighbourhood, stronger local contrast and more amplified noise; "
                  "FEWER approaches a plain global equalization. 8 matches the 8x8 grid of "
                  "Cell-Tracker's OpenCV CLAHE (before V2.13 this was pinned to 4 and "
                  "unreachable from the GUI). The kernel is the axis extent divided by this, "
                  "floored at 2 px, and the same count applies to Z in 3D — on a thin stack "
                  "that floor is what stops it asking for a zero-thick kernel.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,   # shape-derived grid + min/max
    description="Contrast-limited adaptive histogram equalization; 2D per-plane vs 3D. "
                "`tile_grid` sets the regions per axis (8 = Cell-Tracker's OpenCV default).")
