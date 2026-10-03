"""ROI Mask (``analysis.roi_mask``) — Rasterize a serializable shape list ([y,x] verts, in-order add/cut/invert/clear) → a boolean Voxel ROI mask;."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware

# ── ROI mask (serializable shape list → boolean Voxel mask, 2D) ────────────────

def _roi_shapes(raw):
    """The ``shapes`` param → a list of shape dicts, or ``None`` for a whole-frame ROI.

    Two producers, so two accepted forms: a **list** handed in programmatically by a
    headless caller, and the socket's **JSON text**, which is what the GUI's ROI draw tool
    writes (V2.16 — the socket declares ``pick_kind="shapes"``) and the only form a shape
    list can take, since it has no matching :class:`SocketType`. Malformed JSON raises with
    the parse error rather than silently degrading to a whole-frame ROI, which would look
    like the node ran fine and quietly analyse the entire image."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if not isinstance(raw, str):
        return raw
    import json as _json
    try:
        val = _json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"ROI mask: `shapes` is not valid JSON ({exc}). Expected a list of shape "
            'objects, e.g. [{"type": "rect", "op": "add", "vertices": [[5,5],[15,15]]}]'
        ) from exc
    if val in (None, [], {}):
        return None
    if not isinstance(val, list):
        raise ValueError(f"ROI mask: `shapes` must be a JSON list, got {type(val).__name__}")
    return val
def _compute_roi_mask(ctx: EvalContext) -> Dataset:
    """Rasterize an ordered list of vector shapes (rect/ellipse/circle/polygon/brush +
    invert/clear) into a boolean Voxel ROI mask — a general drawn-region mask (crop/
    exclude/analysis ROI; ported v1 ``dic_mesh_region`` kernel). Pure pixel geometry (no
    calibration): the ``shapes`` param carries the drawn ROI ([y,x] vertices, in-order
    stateful replay); an empty/absent list → a whole-frame ROI. Frame-independent (broadcast
    to all m/t/z/c). Kernel: :func:`nodegraph.kernels.dic_mesh_region.build_roi_mask`."""
    from nodegraph.kernels.dic_mesh_region import build_roi_mask, has_region
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("ROI mask needs an image provider on its input Dataset")
    ax = prov.axes
    from nodegraph.catalog._shared.regions import shapes_in_frame
    shapes = _roi_shapes(ctx.params.get("shapes"))    # list of shape dicts, or None
    if shapes is not None:
        # full-frame pixels → the window this compute may be running on (2026-10-02)
        shapes = shapes_in_frame(shapes, ds.metadata)
    if has_region(shapes):
        m2d = np.asarray(build_roi_mask(shapes, ax.y, ax.x), dtype=np.int64)
    else:
        m2d = np.ones((ax.y, ax.x), dtype=np.int64)   # no shapes → whole-frame ROI
    mask6 = np.broadcast_to(m2d, (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)).copy()
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), mask6)
register_node(
    batch_aware(_compute_roi_mask), op_key="analysis.roi_mask", label="ROI Mask",
    category="analysis",
    # reads only the image EXTENT (pure pixel geometry), so nothing is required of the
    # input beyond a provider; it adds the ROI raster.
    adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            # The shape list is structured data with no matching SocketType, so it rides in
            # a STRING socket as JSON. `pick_kind="shapes"` is what makes that liveable:
            # the GUI's draw tool writes this socket (V2.16). Before it existed the node was
            # configurable only by hand-typing vertex arrays, i.e. in practice inert.
            InString("shapes", "Region", field=False, default="",
                     pick_kind="shapes",
                     description=
                     "The regions of interest. DRAW them on the viewer — the Pick button (or the "
                     "ring on the node card) opens rectangle / ellipse / circle / polygon / "
                     "freehand tools with Add, Cut, Invert and Clear, which is how this parameter "
                     "is meant to be set. Underneath it is a JSON list of shape objects in PIXEL "
                     "coordinates, replayed in order, so it stays scriptable and diffable. EMPTY "
                     "— the default — selects the whole frame, so an unconfigured node passes "
                     "everything and masks nothing. Malformed JSON is REFUSED with the parse "
                     "error rather than falling back to the whole frame, which would look like "
                     "a clean run that had quietly analysed the entire image."),
            InString("name", "Output layer", field=False, default="roi_mask",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the ROI mask layer this node writes (1 inside the shapes, 0 "
                     "outside). It is an ordinary Voxel mask, so anything that takes a mask "
                     "layer accepts it — Segmentation's watershed foreground, or a Transfer "
                     "Domain reduction restricted to the ROI.")],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Rasterize a serializable shape list ([y,x] verts, in-order add/cut/"
                "invert/clear) → a boolean Voxel ROI mask; empty → whole frame "
                "(ported v1 kernel).")
