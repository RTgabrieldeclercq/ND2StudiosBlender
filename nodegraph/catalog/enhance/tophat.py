"""Top-Hat (``enhance.tophat``) — White/black top-hat (flatten background / pop dark features);."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InDataset, Mode, OutDataset

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

# ── Top-Hat (white / black — background subtraction / dark-spot pop) ────────────

def _compute_tophat(ctx: EvalContext) -> Dataset:
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    variant = ctx.params.get("__modes__", {}).get("variant", "white")
    fn = ndi.white_tophat if variant == "white" else ndi.black_tophat
    ry = _radius_px(ctx, "radius", 0.5)
    # a 1-voxel top-hat is `a - opening(a, 1) == 0` everywhere — an all-black output
    wy = _require_window(ctx, _win(ry), ry, node=f"top-hat ({variant})",
                         zero_is_identity=False,
                         degenerate="subtracts the image from itself and returns an "
                                    "entirely BLACK image")
    if ctx.is_volume:
        rz = _radius_z_px(ctx, "radius", 0.5)
        wz = _require_window(ctx, _win(rz), rz, node=f"top-hat ({variant})",
                             param="radius_z", axis="axial", zero_is_identity=False,
                             degenerate="subtracts the image from itself and returns an "
                                        "entirely BLACK image")
    else:
        wz = wy
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: fn(a, size=(wy, wy)),
        volume_fn=lambda v: fn(v, size=(wz, wy, wy)),
        halo=2 * (wy // 2))          # tophat wraps an opening/closing: 2-pass reach
register_node(
    _compute_tophat, op_key="enhance.tophat", label="Top-Hat", category="enhancement",
    inputs=[InDataset(), *_InRadius(default=0.5, description=
            "The BACKGROUND scale, in microns: anything larger than this is treated as "
            "background and subtracted away, anything smaller survives. So it must be set "
            "LARGER than the features you want to keep — set it near object size and the "
            "objects subtract themselves, leaving hollow rings and a near-empty image. A "
            "few times the object diameter is the usual starting point. `white` keeps "
            "bright features on a flattened background; `black` keeps dark ones. Note that "
            "background is genuinely removed, so absolute intensities afterwards are "
            "background-subtracted, not raw.")],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("variant", ["white", "black"], default="white",
                description=
                "Which polarity of feature survives the background subtraction. Both "
                "variants remove everything LARGER than Radius; they differ in whether what "
                "is kept is brighter or darker than its surroundings, so picking the wrong "
                "one returns a nearly empty image rather than an inverted one.",
                choice_docs={
                    "white":
                        "Image minus its opening: keeps features BRIGHTER than the local "
                        "background and flattens uneven illumination under them. The one to "
                        "use for fluorescent puncta, beads and nuclei on a drifting "
                        "background.",
                    "black":
                        "Closing minus the image: keeps features DARKER than the local "
                        "background, and outputs them as positive values. The one to use for "
                        "brightfield/phase cells, dark pores or shadows — on that data "
                        "`white` finds only the bright halo.",
                })],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="White/black top-hat (flatten background / pop dark features); "
                "2D vs anisotropic 3D.")
