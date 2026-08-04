"""Morphology (``enhance.morphology``) — Grayscale morphology (erode/dilate/open/close);."""

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

# ── Morphology (erode / dilate / open / close) ──────────────────────────────────

def _compute_morphology(ctx: EvalContext) -> Dataset:
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    op = ctx.params.get("__modes__", {}).get("op", "open")
    fn = {"erode": ndi.grey_erosion, "dilate": ndi.grey_dilation,
          "open": ndi.grey_opening, "close": ndi.grey_closing}[op]
    ry = _radius_px(ctx, "radius", 0.3)
    wy = _require_window(ctx, _win(ry), ry, node=f"morphology ({op})")
    if ctx.is_volume:
        rz = _radius_z_px(ctx, "radius", 0.3)
        wz = _require_window(ctx, _win(rz), rz, node=f"morphology ({op})",
                             param="radius_z", axis="axial")
    else:
        wz = wy
    # open/close = erosion∘dilation: TWO passes each reaching w//2 → influence 2·(w//2)
    # (halo=w//2 leaves real border errors — C1 audit / V2.04 §2)
    reach = (wy // 2) if op in ("erode", "dilate") else 2 * (wy // 2)
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: fn(a, size=(wy, wy)),
        volume_fn=lambda v: fn(v, size=(wz, wy, wy)),
        halo=reach)
register_node(
    _compute_morphology, op_key="enhance.morphology", label="Morphology",
    category="enhancement",
    inputs=[InDataset(), *_InRadius(description=
            "Size of the structuring element, in microns — the scale of feature the chosen "
            "operation acts on. With `erode` bright regions shrink by roughly this much and "
            "with `dilate` they grow by it, so both MOVE EVERY BOUNDARY and change any area "
            "measured downstream. `open` (erode then dilate) deletes bright specks smaller "
            "than this while leaving larger objects near their original size; `close` fills "
            "dark gaps narrower than this. Pick it from the size of the junk you want gone, "
            "not from the size of your objects.")],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("op", ["erode", "dilate", "open", "close"], default="open",
                description=
                "Which of the four grayscale morphology operations to run with the "
                "structuring element set by Radius. The first two move every boundary in "
                "one direction; the last two are those two composed, which is what makes "
                "them size filters that leave surviving objects roughly where they were.",
                choice_docs={
                    "erode":
                        "Replace each pixel with the MINIMUM under the structuring element: "
                        "bright regions shrink by about the radius, dark ones grow, and "
                        "bright specks smaller than the element disappear. Every object gets "
                        "smaller, so areas measured downstream shrink with it.",
                    "dilate":
                        "Replace each pixel with the MAXIMUM under the element: bright "
                        "regions grow by about the radius, gaps and dark cracks narrower "
                        "than it fill in, and near neighbours can merge into one blob. The "
                        "mirror image of erode.",
                    "open":
                        "Erode then dilate. Deletes bright specks smaller than the element "
                        "while leaving larger objects near their original size — the size "
                        "filter to reach for when the problem is salt noise or debris rather "
                        "than object shape.",
                    "close":
                        "Dilate then erode. Fills dark gaps, holes and cracks narrower than "
                        "the element while leaving object outlines near where they were — "
                        "the fix for a nucleus stain that is speckled inside, or for a "
                        "boundary broken by one dim pixel.",
                })],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Grayscale morphology (erode/dilate/open/close); 2D vs anisotropic 3D.")
