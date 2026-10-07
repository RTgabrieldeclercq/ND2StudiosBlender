"""Mask Math (``math.mask``) — set algebra on masks: A minus B, A or B, A and B, A xor B, or
the frame minus A. An EMPTY A is the whole frame — its full area or volume — so `subtract`
with B = your masks is the background mask in one card."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.units import with_unit

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.labels import _resolve_layer, _voxel_layers
from nodegraph.catalog._shared.rasters import broadcast_raster

# ── Mask Math (V4.00 step 12) ────────────────────────────────────────────────────
#
# The gap this fills: masks were made (Threshold, ROI Mask, Draw Regions, Connected
# Components' raster) and consumed, but never COMBINED. "The background is everything that
# is not a cell" had no card: the user drew the inverse by hand or thresholded the other way.
# Set algebra over Voxel rasters is four lines of numpy; what the node adds is the frame as
# an operand (A empty = every voxel, which is how "the area of the image minus the masks"
# reads), a second wire so the two masks may come from different branches (one per channel,
# the user's case), and the broadcast rule that lets a one-channel mask cut a three-channel
# one.

#: The operations, in menu order. ``invert`` is unary (B and `other` are hidden under it).
_MASK_OPS: Tuple[str, ...] = ("subtract", "union", "intersect", "xor", "invert")
_MASK_BINARY = frozenset({"subtract", "union", "intersect", "xor"})


def _compute_mask_math(ctx: EvalContext) -> Dataset:
    """Combine two masks into one Voxel mask layer.

    Resolved spec (build-node-v2 §0, V4.00 step 12)
    -----------------------------------------------
    * **Kind** math → ``op_key="math.mask"``, category ``"math"``. Dataset in, the same
      Dataset out plus one Voxel layer; the image is untouched. Any Voxel raster is a mask
      here (nonzero = inside), so a label raster or a distance field qualifies as well.
    * **Operands.** ``A`` is a Voxel layer on ``data``, or — EMPTY — the whole frame, every
      voxel. ``B`` is a Voxel layer on ``other`` when that is wired, else on ``data``; empty
      B is the only raster on its wire (other than A), refused when there are several.
    * **Broadcast.** B may be narrower than the data on m/t/z/c (size 1 stretches); y/x
      must match (:func:`~nodegraph.catalog._shared.rasters.broadcast_raster`).
    * **2D/3D** none: voxel-wise logic. **Footprint** ``WHOLE_PLANE`` over y,x like ROI
      Mask: the layers are whole arrays, and nothing is read from the image.
    * **Output** ``uint8`` 0/1, recorded dimensionless (:func:`nodegraph.units.with_unit`).
    """
    ds = ctx.inputs[0]
    shape = ds.axes.shape_for(Domain.VOXEL)
    modes = ctx.params.get("__modes__", {}) or {}
    op = str(modes.get("op") or "subtract")
    if op not in _MASK_OPS:
        raise ValueError(f"mask math: unknown operation {op!r} — one of {list(_MASK_OPS)}")
    a_name = ctx.layer("a")
    if a_name:
        a_attr = ds.get(Domain.VOXEL, a_name)
        if a_attr is None:
            raise ValueError(
                f"mask math: no Voxel layer {a_name!r} on the data (it carries "
                f"{_voxel_layers(ds)}) — pick A from its dropdown, or leave it EMPTY for "
                f"the whole frame")
        a = broadcast_raster(a_attr.values, shape, node="mask math",
                             what=f"A ({a_name!r})")
    else:
        a = np.ones(shape, dtype=bool)                 # the frame: every voxel
    if op == "invert":
        out = ~a
    else:
        other = ctx.input("other")
        src = other if other is not None else ds
        where = "the `other` input" if other is not None else "the `data` input"
        cands = _voxel_layers(src)
        if other is None and a_name:
            cands = [c for c in cands if c != a_name]
        b_name, _note = _resolve_layer(
            cands, ctx.layer("b"), node="mask math", socket="b", what="Voxel layer",
            where=where,
            remedy="wire a mask into `data` or `other` (Threshold, ROI Mask, Draw Regions, "
                   "Connected Components all make one)", ctx=ctx)
        b = broadcast_raster(src.get(Domain.VOXEL, b_name).values, shape,
                             node="mask math", what=f"B ({b_name!r})")
        if op == "subtract":
            out = a & ~b
        elif op == "union":
            out = a | b
        elif op == "intersect":
            out = a & b
        else:
            out = a ^ b
    name = ctx.layer("name")
    inside = int(np.count_nonzero(out))
    ctx.progress(1, 1, f"{name}: {inside} of {out.size} voxels inside")
    result = ds.with_layer(Domain.VOXEL, name, np.ascontiguousarray(out, dtype=np.uint8))
    return with_unit(result, Domain.VOXEL, name, "")


register_node(
    batch_aware(_compute_mask_math), op_key="math.mask", label="Mask Math", category="math",
    adds_domains=frozenset({Domain.VOXEL}),
    inputs=[
        InDataset(description=
                  "The data the result is added to, and the wire mask A (and B, when "
                  "`Other` is unwired) are read from. Its image passes through untouched; "
                  "only the new mask layer is added."),
        InDataset("other", label="Other", passes_domains=False,
                  description=
                  "Optional: a second branch carrying mask B — a mask made on another "
                  "channel, or on a page of its own. Unwired, B is read off `data`. Only "
                  "B's layer is read from it: its image and other layers do not pass "
                  "downstream, so its geometry must match the data on y/x and may be "
                  "narrower (size 1) on m/t/z/c."),
        InString("a", "A", field=False, default="", layer_in=Domain.VOXEL,
                 description=
                 "The first mask, a Voxel layer on `data` (any raster counts: a mask, a "
                 "label raster, a distance field — nonzero is inside). LEAVE IT EMPTY FOR "
                 "THE WHOLE FRAME — every voxel of the image — which with `subtract` and "
                 "B = your masks gives the background mask; with `intersect` it is just B."),
        InString("b", "B", field=False, default="", layer_in=Domain.VOXEL, layer_from="other",
                 available_in={"op": _MASK_BINARY},
                 description=
                 "The second mask: a Voxel layer on `Other` when that is wired, else on "
                 "`data`. Empty takes the only raster on that wire (other than A) and "
                 "refuses when there are several, listing them. Hidden under `invert`, "
                 "which has no B."),
        InString("name", "Output layer", field=False, default="mask_math",
                 layer_out=(Domain.VOXEL,),
                 description=
                 "Name of the Voxel mask this node writes (1 inside, 0 outside). Anything "
                 "that takes a mask accepts it: Connected Components, Crop by Region, "
                 "Threshold's `per_roi` scope, the mask overlay. Call it what it is — "
                 "`background`, `cells_not_nuclei` — so later pickers read clearly."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("op", list(_MASK_OPS), default="subtract", label="Operation",
             description=
             "How the two masks combine, voxel by voxel. The choice folds into the memo "
             "key; the image is never changed, only the new layer.",
             choice_docs={
                 "subtract": "A minus B: inside A and NOT inside B. With A empty (the whole "
                             "frame) this is everything that is not in B — the BACKGROUND of "
                             "a segmentation, the usual reason to reach for this node.",
                 "union": "A or B: inside either. Joins masks made per channel or per "
                          "threshold into one; where the two overlap the result is simply "
                          "inside once.",
                 "intersect": "A and B: inside both. The overlap of two masks — a nucleus "
                              "mask restricted to the cell mask, or a drawn ROI applied to "
                              "a thresholded mask.",
                 "xor": "A xor B: inside exactly one of them. The disagreement between two "
                        "masks — what one threshold found and the other did not — useful "
                        "for comparing methods.",
                 "invert": "The frame minus A: outside A becomes inside and inside becomes "
                           "outside; B and `Other` are not read. With A empty the result is "
                           "empty (nothing is outside the whole frame).",
             })],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Set algebra on masks — subtract, union, intersect, xor, invert — between a "
                "Voxel layer and another (from this wire or a second branch); an EMPTY A is "
                "the whole frame, so frame minus masks is the background. Writes one new "
                "mask layer; the image is untouched.")
