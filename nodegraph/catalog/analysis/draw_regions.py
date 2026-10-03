"""Draw Regions (``analysis.draw_regions``) — Hand-drawn labelled regions, each pinned to the frame (m, t, z) it was drawn on: rectangles, circles, ellipses and closed polygons rasterized to a Label layer plus a Label table, so a drawn patch can be measured, overlaid, or handed to Subtract Background as the sample of what counts as background."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import LABEL_INVARIANT, on_layer

#: how a shape's ``frame`` stamp is read. ``drawn_frame`` honours it; ``all_frames`` ignores it.
_SCOPES = ("drawn_frame", "all_frames")

#: shape types that produce a region of their own (one id each); ``cut`` shapes subtract from
#: every region drawn before them, ``clear`` discards everything before it.
_REGION_TYPES = ("rect", "ellipse", "circle", "polygon", "brush")


def _parse_shapes(raw: Any) -> List[Dict[str, Any]]:
    """The ``shapes`` param -> a list of shape dicts (``[]`` when nothing is drawn).

    The same two forms ROI Mask accepts: a list from a headless caller, or the JSON text
    the viewer's draw tool writes into the STRING socket. Each shape may carry a
    ``frame`` stamp — ``[m, t, z]`` — which the viewer adds at the moment the shape is
    finished (the frame it was drawn on). Malformed JSON is refused with the parse error:
    a node that silently drew nothing would look like a clean run."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    if not isinstance(raw, str):
        val = raw
    else:
        try:
            val = json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                f"draw regions: `shapes` is not valid JSON ({exc}). Draw the regions on the "
                "viewer with the Pick tool; by hand, a list of shape objects, e.g. "
                '[{"type": "rect", "op": "add", "vertices": [[5,5],[15,15]], '
                '"frame": [0, 3, 0]}]') from exc
    if val in (None, [], {}):
        return []
    if not isinstance(val, list):
        raise ValueError(f"draw regions: `shapes` must be a JSON list, got "
                         f"{type(val).__name__}")
    out = []
    for i, s in enumerate(val):
        if not isinstance(s, dict):
            raise ValueError(f"draw regions: shape {i} is not an object: {s!r}")
        out.append(s)
    return out


def _frame_of(shape: Dict[str, Any]) -> Optional[tuple]:
    """``(m, t, z)`` the shape is pinned to, or ``None`` for every frame."""
    fr = shape.get("frame")
    if fr is None:
        return None
    if isinstance(fr, dict):
        fr = (fr.get("m", 0), fr.get("t", 0), fr.get("z", 0))
    try:
        m, t, z = (int(v) for v in tuple(fr)[:3])
    except (TypeError, ValueError):
        raise ValueError(f"draw regions: a shape's `frame` must be [m, t, z], got {fr!r}")
    return (m, t, z)


def _applies(shape: Dict[str, Any], m: int, t: int, z: int, per_frame: bool) -> bool:
    if not per_frame:
        return True
    fr = _frame_of(shape)
    return fr is None or fr == (m, t, z)


def rasterize_regions(shapes: Sequence[Dict[str, Any]], H: int, W: int, *,
                      m: int = 0, t: int = 0, z: int = 0, per_frame: bool = True
                      ) -> np.ndarray:
    """One frame's label raster: region shapes get ids 1..K in draw order (a later shape
    overwrites an earlier one where they overlap), ``cut`` shapes carve out of everything
    drawn before them, ``clear`` starts over. Ids are the shape's POSITION among the region
    shapes that apply to this frame, so the same drawn patch keeps its id on every frame
    it appears on. ``invert`` is ignored: the complement of a set of labelled regions is
    not a region."""
    from nodegraph.kernels.dic_mesh_region import _rasterize_region
    lab = np.zeros((H, W), dtype=np.int64)
    k = 0
    for shape in shapes:
        typ = str(shape.get("type", ""))
        if typ == "clear":
            lab[...] = 0
            k = 0
            continue
        if typ not in _REGION_TYPES or not _applies(shape, m, t, z, per_frame):
            continue
        mask = np.asarray(_rasterize_region(shape, H, W), dtype=bool)
        if str(shape.get("op", "add")) == "cut":
            lab[mask] = 0
            continue
        k += 1
        lab[mask] = k
    return lab


def _compute_draw_regions(ctx: EvalContext) -> Dataset:
    """Rasterize the drawn shapes into a Label raster per frame and emit the matching
    Label table (``id, m, t, c, area, z, y, x`` — the invariant schema, centroid in
    pixels), so the regions overlay draws them with ids, Measure can report on them, and
    Subtract Background can sample them. Under ``scope = drawn_frame`` a shape lands only
    on the frame it was drawn on (its ``frame`` stamp) and an unstamped shape on every
    frame; under ``all_frames`` every shape lands everywhere. Nothing drawn is not an
    error: the node emits an empty layer, which is what an unconfigured annotation is."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("draw regions needs an image provider on its input Dataset")
    ax = prov.axes
    from nodegraph.catalog._shared.regions import shapes_in_frame
    # the shapes are stored in FULL-FRAME pixels; under a troubleshooting window this
    # compute runs on the window, so move them into it first (regions.py says why)
    shapes = shapes_in_frame(_parse_shapes(ctx.params.get("shapes")), ds.metadata)
    modes = ctx.params.get("__modes__", {}) or {}
    scope = str(modes.get("scope") or "drawn_frame")
    if scope not in _SCOPES:
        raise ValueError(f"draw regions: unknown scope {scope!r} — one of {list(_SCOPES)}")
    per_frame = scope != "all_frames"
    layer = ctx.layer("name")
    raster = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    cols: Dict[str, list] = {k: [] for k in LABEL_INVARIANT}
    offset = 0
    for m in range(ax.m):
        for t in range(ax.t):
            for z in range(ax.z):
                lab = rasterize_regions(shapes, ax.y, ax.x, m=m, t=t, z=z,
                                        per_frame=per_frame)
                top = int(lab.max()) if lab.size else 0
                if top <= 0:
                    continue
                for c in range(ax.c):
                    raster[m, t, z, c] = np.where(lab > 0, lab + offset, 0)
                ys, xs = np.nonzero(lab)
                ids = lab[ys, xs]
                for k in range(1, top + 1):
                    sel = ids == k
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    for c in range(ax.c):
                        cols["id"].append(offset + k)
                        cols["m"].append(m)
                        cols["t"].append(t)
                        cols["c"].append(c)
                        cols["area"].append(n)
                        cols["z"].append(z)
                        cols["y"].append(float(ys[sel].mean()))
                        cols["x"].append(float(xs[sel].mean()))
                offset += top
    out = ds.with_layer(Domain.VOXEL, layer, raster)
    if cols["id"]:
        out = out.with_structure(StructureTable(
            Domain.LABEL, {k: np.asarray(v) for k, v in cols.items()},
            layer=layer, z_kind="plane_index"))
    return out


def _columns_draw_regions(params, modes, incoming):
    return on_layer(Domain.LABEL, str((params or {}).get("name") or "regions"),
                    LABEL_INVARIANT)


register_node(
    batch_aware(_compute_draw_regions), op_key="analysis.draw_regions",
    label="Draw Regions", category="analysis",
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_columns=_columns_draw_regions,
    inputs=[
        InDataset(description=
                  "The image the regions are drawn on. Only its extent and frame count are "
                  "read; the pixels pass through untouched, with the regions added as a "
                  "Label layer beside them."),
        InString("shapes", "Regions", field=False, default="",
                 pick_kind="shapes",
                 description=
                 "The drawn regions. DRAW them on the viewer with the Pick button: "
                 "rectangle, ellipse, circle, closed polygon (double-click to close) or "
                 "freehand, with Add and Cut; each finished shape is stamped with the frame "
                 "(m, t, z) you were looking at, so under `scope = drawn_frame` it appears on "
                 "that frame only — step to another frame and draw its regions there. The "
                 "shapes are stored as a JSON list in full-frame pixel coordinates (a region "
                 "drawn inside a troubleshooting window is shifted back into the full frame), "
                 "so the list is scriptable and diffable. Each region shape becomes one "
                 "labelled region, numbered in draw order; a Cut shape carves out of the "
                 "regions before it. Empty draws nothing and the output layer is all zero."),
        # ── the drawing tools, as node settings (presentation: outside the recipe hash,
        # never read by the compute — the GUI reads them live and drives the gesture) ──
        InString("tool", "Tool", field=False, default="rect", presentation=True,
                 choices=["rect", "ellipse", "circle", "polygon", "brush"],
                 choice_docs={
                     "rect": "Rectangle — press one corner on the image and drag to the "
                             "opposite corner; the region is the axis-aligned box.",
                     "ellipse": "Ellipse — drag its bounding box; the region is the ellipse "
                                "inscribed in that box (a circle if the box is square).",
                     "circle": "Circle — press at the centre and drag outward; the radius is "
                               "the drag distance, so the centre stays where you pressed.",
                     "polygon": "Closed polygon — click each corner in turn and double-click "
                                "(or press Enter) to close it; at least three corners.",
                     "brush": "Freehand band — drag a path; the region is the path thickened "
                              "to `Brush size` pixels, for an irregular patch of background.",
                 },
                 description=
                 "Which shape the next drag on the image makes. A node setting rather than a "
                 "toolbar on the viewer, so the whole drawing is configured here; it does "
                 "not change the result (the shapes do) and is remembered with the node."),
        InString("op", "Operation", field=False, default="add", presentation=True,
                 choices=["add", "cut"],
                 choice_docs={
                     "add": "The next shape is a region of its own: it gets the next id and "
                            "overwrites any earlier region where they overlap.",
                     "cut": "The next shape is carved OUT of every region drawn before it — "
                            "for excluding a cell that sits inside a background patch.",
                 },
                 description=
                 "Whether the next shape adds a region or cuts a hole in the regions drawn so "
                 "far. Presentation only: it configures the gesture, the shapes carry the "
                 "result."),
        InFloat("brush_px", "Brush size", unit="px", field=False, default=8.0,
                presentation=True,
                description=
                "Width in pixels of the freehand band (the `brush` tool only). Read live while "
                "drawing; changing it does not re-run anything."),
        InString("name", "Output layer", field=False, default="regions",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "Name of the Label layer this node writes: an integer raster with one id "
                 "per drawn region (0 outside), plus the matching Label table with each "
                 "region's frame, area and centroid. Anything that takes a Label layer "
                 "accepts it — Measure, Filter Labels, the Labels overlay — and Subtract "
                 "Background's `Background regions` input samples it.")],
    outputs=[OutDataset()],
    modes=[
        Mode("scope", list(_SCOPES), default="drawn_frame", label="Scope",
             choice_docs={
                 "drawn_frame":
                     "Each shape appears only on the frame (m, t, z) it was drawn on, "
                     "which the viewer stamps onto it when the shape is finished. A frame "
                     "you drew nothing on gets no regions. A shape with no stamp (typed in "
                     "by hand) appears on every frame.",
                 "all_frames":
                     "Every shape appears on every frame, whatever it was drawn on — the "
                     "right choice when the background patch or the region of interest "
                     "does not move and you want to draw it once.",
             },
             description=
             "Whether a drawn shape belongs to the one frame it was drawn on or to the "
             "whole series. Folds into the memo key, so switching re-rasterizes.")],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Hand-drawn labelled regions (rect / circle / ellipse / polygon / freehand), "
                "each pinned to the frame (m, t, z) it was drawn on → a Label raster + Label "
                "table; Subtract Background's `Background regions` input samples them.")
