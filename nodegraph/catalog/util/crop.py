"""Crop (``util.crop``) — Crop Y,X (and Z in 3D mode);."""

from __future__ import annotations


from dataclasses import replace
from typing import Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import crop as _meta_crop
from nodegraph.registry import DimMode, InDataset, InInt, OutDataset
from nodegraph.streaming import WindowView

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN
from nodegraph.catalog._shared.sampling import _sampled

# ── Crop (axis-changing: shrink Y,X and, in 3D, Z) ──────────────────────────────

def _compute_crop(ctx: EvalContext) -> Dataset:
    """Crop the spatial extent (and, in 3D mode, the Z range). Its ``crop``
    meta_transform tracks the new extent at edit time; pixel size is preserved
    (origin is deferred, V2.00 §16)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("crop needs an image provider on its input Dataset")
    ax = prov.axes

    def bound(v, default, hi):
        return max(0, min(int(v) if v is not None else default, hi))

    y0, y1 = bound(ctx.params.get("y0"), 0, ax.y), bound(ctx.params.get("y1"), ax.y, ax.y)
    x0, x1 = bound(ctx.params.get("x0"), 0, ax.x), bound(ctx.params.get("x1"), ax.x, ax.x)
    if ctx.is_volume:
        z0 = bound(ctx.params.get("z0"), 0, ax.z)
        z1 = bound(ctx.params.get("z1"), ax.z, ax.z)
    else:
        z0, z1 = 0, ax.z
    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        raise ValueError(f"crop produced an empty region "
                         f"(y[{y0}:{y1}] x[{x0}:{x1}] z[{z0}:{z1}])")
    ny, nx, nz = y1 - y0, x1 - x0, z1 - z0
    new_axes = replace(ax, z=nz, y=ny, x=nx)
    # C1: a pure lazy offset view (the _ChannelView pattern) — no pixels move, the
    # source dtype is preserved, and a kernel op downstream clips its halo at THIS
    # view's extents (= the eager reflect-at-crop-edge behavior, V2.04 §6b).
    view = WindowView(prov, z0=z0, y0=y0, x0=x0, axes=new_axes)
    out = _sampled(ds.with_image(view).reshaped_axes(new_axes),
                   f"crop[z{z0}:{z1},y{y0}:{y1},x{x0}:{x1}]")
    # A crop MOVES the field's corner, so the payload must carry the moved origin or it
    # would disagree with the header the meta_transform already predicted (§8). SYNCED
    # from `ctx.calib`, never re-derived here: the env has already had `_meta_crop` applied,
    # so recomputing the shift would apply the cut twice.
    origin = ctx.calib("origin_um")
    return out.with_metadata(origin_um=origin) if origin is not None else out
#: Shared hover text for the six crop bounds. One template rather than six near-identical
#: paragraphs: what differs between them is the axis and the inclusive/exclusive end, and
#: everything else — why the unit is px, what shrinking an axis does to the layers, that the
#: view is lazy — is the same sentence six times over.
#: The two bound GROUPS a crop pick writes (V2.16). Declared once and shared by all six
#: sockets so arming from any row produces the same gesture — the registry requires every
#: member to name the identical group, which a per-socket literal would eventually violate.
#: Split lateral from axial deliberately: a rectangle drawn on a plane says nothing about Z,
#: and the two halves are also gated apart (z0/z1 are 3D-only), which the registry's
#: gated-together rule would otherwise refuse.
_CROP_RECT: Tuple[str, ...] = ("y0", "y1", "x0", "x1")
_CROP_ZRANGE: Tuple[str, ...] = ("z0", "z1")
_CROP_BOUND_DOC = (
    "{axis} {end} of the window that is KEPT, in PIXELS — not µm, because a crop is an index "
    "range into the array and rounding a physical extent would make the output size depend on "
    "calibration. {extra} Cropping shrinks the {axis} axis, so any attribute layer whose shape "
    "no longer fits is dropped from the bundle, and a window that collapses to nothing is "
    "refused rather than silently produced. No pixels are copied — the result is a lazy view, "
    "so a filter downstream still reads real data for its halo right up to the crop edge.")
register_node(
    _compute_crop, op_key="util.crop", label="Crop", category="utility",
    inputs=[
        InDataset(),
        InInt("y0", "Y start", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              description=_CROP_BOUND_DOC.format(
                  axis="Y", end="start",
                  extra="INCLUSIVE — this row is kept. 0 or unset starts at the top edge.")),
        InInt("y1", "Y end", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              description=_CROP_BOUND_DOC.format(
                  axis="Y", end="end",
                  extra="EXCLUSIVE, like a Python slice — this row is the first one dropped. "
                        "Unset keeps everything to the bottom edge; a value past the edge is "
                        "clamped rather than an error.")),
        InInt("x0", "X start", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              description=_CROP_BOUND_DOC.format(
                  axis="X", end="start",
                  extra="INCLUSIVE — this column is kept. 0 or unset starts at the left "
                        "edge.")),
        InInt("x1", "X end", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              description=_CROP_BOUND_DOC.format(
                  axis="X", end="end",
                  extra="EXCLUSIVE, like a Python slice — this column is the first one "
                        "dropped. Unset keeps everything to the right edge; a value past the "
                        "edge is clamped.")),
        InInt("z0", "Z start", unit="px", field=False, pick_bounds=_CROP_ZRANGE, pick_kind="zrange",
              available_in={"dim": frozenset({"3D"})},
              description=_CROP_BOUND_DOC.format(
                  axis="Z", end="start",
                  extra="INCLUSIVE, and a PLANE INDEX rather than a depth in microns. 3D "
                        "only — with the lever on 2D every plane is kept and this is "
                        "ignored.")),
        InInt("z1", "Z end", unit="px", field=False, pick_bounds=_CROP_ZRANGE, pick_kind="zrange",
              available_in={"dim": frozenset({"3D"})},
              description=_CROP_BOUND_DOC.format(
                  axis="Z", end="end",
                  extra="EXCLUSIVE plane index — the first plane dropped. Unset keeps "
                        "everything to the last plane. 3D only.")),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=frozenset(),
    meta_transform=_meta_crop,
    description="Crop Y,X (and Z in 3D mode); pixel size preserved (origin deferred).")
