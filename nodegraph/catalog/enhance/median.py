"""Median (``enhance.median``) — Edge-preserving median filter;."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN, _DIM_KAX
from nodegraph.catalog._shared.kernel_radius import (
    _InRadius,
    _radius_px,
    _radius_z_px,
    _require_window,
    _win,
)
from nodegraph.catalog._shared.map_image import _map_image
from nodegraph.catalog._shared.ndimage import _ndi

# ── Median ─────────────────────────────────────────────────────────────────────

def _compute_median(ctx: EvalContext) -> Dataset:
    from scipy.ndimage import median_filter
    ds = ctx.inputs[0]
    ry = _radius_px(ctx, "radius", 0.3)
    wy = _require_window(ctx, _win(ry), ry, node="median")
    if ctx.is_volume:
        rz = _radius_z_px(ctx, "radius", 0.3)
        wz = _require_window(ctx, _win(rz), rz, node="median", param="radius_z",
                             axis="axial")
    else:
        wz = wy
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: _ndi("median_filter", a, size=(wy, wy)),
        volume_fn=lambda v: _ndi("median_filter", v, size=(wz, wy, wy)),
        halo=wy // 2)
register_node(
    _compute_median, op_key="enhance.median", label="Median", category="enhancement",
    inputs=[InDataset(), *_InRadius(description=
            "Half-width of the sliding window, in microns — each pixel becomes the MEDIAN "
            "of its neighbourhood. LARGER removes bigger speckles and, unlike a Gaussian, "
            "keeps edges crisp while doing it; but any object smaller than the window is "
            "erased outright, so keep this below the size of what you intend to measure. "
            "Cost grows with the window, and a median is much slower than a blur of the "
            "same reach. Best tool for salt-and-pepper or hot-pixel noise; poor for "
            "Gaussian read noise, where a blur does better.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Edge-preserving median filter; 2D per-plane vs anisotropic 3D window.")
