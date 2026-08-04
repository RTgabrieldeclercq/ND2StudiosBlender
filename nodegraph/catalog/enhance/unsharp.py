"""Unsharp Mask (``enhance.unsharp``) — Unsharp-mask sharpening (blur radius in µm);."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN, _DIM_KAX
from nodegraph.catalog._shared.kernel_radius import _InRadius, _radius_px, _radius_z_px
from nodegraph.catalog._shared.map_image import _map_image

# ── Unsharp Mask (edge sharpening) ──────────────────────────────────────────────

def _compute_unsharp(ctx: EvalContext) -> Dataset:
    from skimage.filters import unsharp_mask
    ds = ctx.inputs[0]
    amount = float(ctx.params.get("amount", 1.0))
    rxy = _radius_px(ctx, "radius", 0.3)
    halo = int(4.0 * rxy + 0.5)               # gaussian blur influence (truncate=4)
    if ctx.is_volume:
        rz = _radius_z_px(ctx, "radius", 0.3)
        return _map_image(
            ctx, ds,
            plane_fn=lambda a: unsharp_mask(a, radius=rxy, amount=amount, preserve_range=True),
            volume_fn=lambda v: unsharp_mask(v, radius=(rz, rxy, rxy), amount=amount,
                                             preserve_range=True),
            halo=halo)
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: unsharp_mask(a, radius=rxy, amount=amount, preserve_range=True),
        halo=halo)
register_node(
    _compute_unsharp, op_key="enhance.unsharp", label="Unsharp Mask",
    category="enhancement",
    inputs=[InDataset(), *_InRadius(description=
            "Blur radius of the mask, in microns — it picks WHICH SCALE gets sharpened. "
            "Sharpening boosts detail around this size, so set it near the finest structure "
            "you care about; too large and it exaggerates broad shading instead of detail, "
            "producing halos around big objects."),
            InFloat("amount", "Amount", unit="", field=True, default=1.0,
                    description=
                    "How much of the blurred-difference mask to add back. 0 is a no-op; 1 is "
                    "a normal sharpen; above about 2 the edges gain visible bright/dark "
                    "halos and noise is amplified along with the detail. Sharpening ADDS "
                    "intensity at edges and removes it beside them, so it changes measured "
                    "peak intensities — do not sharpen a branch you intend to quantify.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Unsharp-mask sharpening (blur radius in µm); 2D vs anisotropic 3D.")
