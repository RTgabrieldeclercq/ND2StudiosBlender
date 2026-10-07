"""Crop by Region (``util.crop_region``) — crop an image to a MASK, to drawn regions or to a
label raster rather than to a rectangle: keep the frame's extent and blank everything
outside, shrink to the bounding box of everything inside, or cut every object out into a
position of its own — each keeping its place in the field. The card's second socket,
`outside`, carries the inverse."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AttributeLayer, Dataset
from nodegraph.domains import Domain, is_lattice
from nodegraph.engine import EvalContext
from nodegraph.memo import digest
from nodegraph.metadata import crop_region as _meta_crop_region, position_subset
from nodegraph.placement import field_box, sub_field_box
from nodegraph.provider import ArrayProvider
from nodegraph.registry import (DimMode, Granularity, InDataset, InInt, InString, Mode,
                                OutDataset)
from nodegraph.spill import dense_output
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers, _voxel_layers
from nodegraph.catalog._shared.rasters import broadcast_raster
from nodegraph.catalog._shared.sampling import _sampled

# ── Crop by Region (V4.00 step 12) ───────────────────────────────────────────────
#
# `util.crop` cuts a rectangle; this cuts a SHAPE. The region is any Voxel raster on the
# wire or on a second branch — a Threshold mask, an ROI Mask, Draw Regions' labelled
# patches, a segmentation's label raster — and "crop" means three different things the
# user asked for by name (2026-10-07):
#
# * **frame** — the image keeps its extent and every pixel outside the region is filled.
#   Nothing moves; a measurement downstream reads the same coordinates. This is masking.
# * **fit** — the image shrinks to the bounding box of everything inside (plus a margin),
#   outside pixels within the box still filled. The field's corner moves, and the payload
#   says where (`origin_um`), so the crop can be placed back.
# * **each** — every object becomes a POSITION of its own: the output's M axis is the
#   objects, each in a window of the common size anchored at its own corner, with
#   `origin_um` and a `position_name` (`obj3`, `B03_obj3`) — so Split Positions fans them
#   out, Stitch or Canvas puts them back where they were, and every per-position card
#   downstream reads which object it is on. "Objects always keep their relative position."
#
# `keep = outside` is the inverse — everything NOT in the region, at the frame's extent —
# and is what the card's synthetic `outside` socket carries: the GUI materializes a wire
# from it into a sibling of this node with `keep` flipped (nodelab_v2.ops), because the
# engine is one-payload-per-node.

CROP_REGION_OP = "util.crop_region"
#: The synthetic output socket the GUI adds to the card: the inverse of `out`.
OUTSIDE_SOCKET = "outside"
KEEP_MODE = "keep"

_EXTENTS: Tuple[str, ...] = ("frame", "fit", "each")
_OBJECTS: Tuple[str, ...] = ("components", "labels")
_KEEP: Tuple[str, ...] = ("inside", "outside")
_FILL: Tuple[str, ...] = ("zero", "nan")
#: Per-M keys that describe WHERE a field is; a window moves the corner, so they are
#: retired and `origin_um` restated from the window (the maintained key, `field_box`).
_GEOMETRY_POSITION_KEYS = ("stage_xy_um", "stage_z_um", "__align_um__", "align_to_ncc")


class _Win(NamedTuple):
    m: int
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int
    obj: Optional[np.ndarray]      # (D,H,W) bool: this object's own voxels, or None = all


def _bbox(foot: np.ndarray, *, margin: int, fit_z: bool) -> Optional[Tuple[int, ...]]:
    """``(z0,z1,y0,y1,x0,x1)`` of a ``(Z,Y,X)`` footprint, ``margin`` px added on y/x and
    clamped to the frame; Z whole unless ``fit_z``. ``None`` for an empty footprint."""
    Z, Y, X = foot.shape
    if not foot.any():
        return None
    ys = np.flatnonzero(foot.any(axis=(0, 2)))
    xs = np.flatnonzero(foot.any(axis=(0, 1)))
    y0, y1 = max(0, int(ys[0]) - margin), min(Y, int(ys[-1]) + 1 + margin)
    x0, x1 = max(0, int(xs[0]) - margin), min(X, int(xs[-1]) + 1 + margin)
    if fit_z:
        zs = np.flatnonzero(foot.any(axis=(1, 2)))
        z0, z1 = int(zs[0]), int(zs[-1]) + 1
    else:
        z0, z1 = 0, Z
    return z0, z1, y0, y1, x0, x1


def _components(foot: np.ndarray, is_3d: bool) -> Tuple[np.ndarray, int]:
    """Connected components of a ``(Z,Y,X)`` footprint: 26-connected in 3D, else 8-connected
    on the Z-collapsed plane and broadcast over Z (an object is one column of the stack).
    Returns ``(label raster (Z,Y,X), count)``."""
    from scipy import ndimage as ndi
    if is_3d:
        lab, n = ndi.label(foot, structure=ndi.generate_binary_structure(3, 3))
        return lab, int(n)
    lab2, n = ndi.label(foot.any(axis=0), structure=ndi.generate_binary_structure(2, 2))
    return np.broadcast_to(lab2[None], foot.shape), int(n)


def _windows(inside6: np.ndarray, raster6: np.ndarray, ax: Any, *, extent: str,
             objects: str, is_3d: bool, margin: int) -> List[_Win]:
    """The output windows, one per output position, in order: the whole frame per position
    (``frame``), the box of everything inside per position (``fit``), or one per object
    (``each`` — a connected region of the footprint, or one label id)."""
    wins: List[_Win] = []
    for m in range(ax.m):
        if extent == "frame":
            wins.append(_Win(m, 0, ax.z, 0, ax.y, 0, ax.x, None))
            continue
        foot = inside6[m].any(axis=(0, 2))                       # (Z,Y,X) over t and c
        if extent == "fit":
            bb = _bbox(foot, margin=margin, fit_z=is_3d)
            if bb is not None:
                wins.append(_Win(m, *bb, None))
            continue
        if objects == "labels":
            from scipy import ndimage as ndi
            rast = np.asarray(raster6[m])                        # (T,Z,C,Y,X) ids
            ids = np.unique(rast)
            ids = ids[ids != 0]
            boxes: Dict[int, List[int]] = {}
            for t in range(rast.shape[0]):
                for c in range(rast.shape[2]):
                    vol = np.ascontiguousarray(rast[t, :, c]).astype(np.int64, copy=False)
                    for k, sl in enumerate(ndi.find_objects(vol), start=1):
                        if sl is None:
                            continue
                        b = boxes.setdefault(k, [sl[0].start, sl[0].stop, sl[1].start,
                                                 sl[1].stop, sl[2].start, sl[2].stop])
                        b[0], b[1] = min(b[0], sl[0].start), max(b[1], sl[0].stop)
                        b[2], b[3] = min(b[2], sl[1].start), max(b[3], sl[1].stop)
                        b[4], b[5] = min(b[4], sl[2].start), max(b[5], sl[2].stop)
            for k in ids.tolist():
                b = boxes.get(int(k))
                if b is None:
                    continue
                tz0, tz1, ty0, ty1, tx0, tx1 = b
                y0, y1 = max(0, ty0 - margin), min(ax.y, ty1 + margin)
                x0, x1 = max(0, tx0 - margin), min(ax.x, tx1 + margin)
                z0, z1 = (tz0, tz1) if is_3d else (0, ax.z)
                obj = (rast[:, z0:z1, :, y0:y1, x0:x1] == k).any(axis=(0, 2))
                wins.append(_Win(m, z0, z1, y0, y1, x0, x1, obj))
            continue
        lab, n = _components(foot, is_3d)
        from scipy import ndimage as ndi
        slices = ndi.find_objects(np.ascontiguousarray(lab)) if n else []
        for k, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            y0, y1 = max(0, sl[1].start - margin), min(ax.y, sl[1].stop + margin)
            x0, x1 = max(0, sl[2].start - margin), min(ax.x, sl[2].stop + margin)
            z0, z1 = (sl[0].start, sl[0].stop) if is_3d else (0, ax.z)
            obj = np.asarray(lab[z0:z1, y0:y1, x0:x1] == k)
            wins.append(_Win(m, z0, z1, y0, y1, x0, x1, obj))
    return wins


def _source_position_names(md: Any, m: int) -> List[str]:
    """The input's position names (``position_name``, else ``m{index in file}``), the
    prefix an object's name takes on a multipoint (``B03_obj1``)."""
    names = md.get("position_name")
    if isinstance(names, (list, tuple)) and len(names) == m:
        return [str(v) for v in names]
    idx = md.get("position_index")
    if isinstance(idx, (list, tuple)) and len(idx) == m:
        try:
            return [f"m{int(v)}" for v in idx]
        except (TypeError, ValueError):
            pass
    return [f"m{i}" for i in range(m)]


def _compute_crop_region(ctx: EvalContext) -> Dataset:
    """Crop to a region: keep the frame, fit the box, or cut each object out.

    Resolved spec (build-node-v2 §0, V4.00 step 12)
    -----------------------------------------------
    * **Kind** utility, axis-changing → ``op_key="util.crop_region"``, category
      ``"utility"``, ``meta_transform=crop_region`` (frame/outside: identity; fit: y/x
      unknown, z too in 3D; each: m unknown as well; the field's corner and stage keys are
      dropped at edit time and RESTATED by the payload, the ``util.crop_to`` rule).
    * **Region** any Voxel raster (nonzero = inside) on the ``regions`` wire if wired, else
      on ``data``; the only one is inferred, several are refused by name. From another wire
      it may be narrower on m/t/z/c (:func:`broadcast_raster`).
    * **Modes** ``extent`` frame/fit/each; ``objects`` components/labels (each only);
      ``keep`` inside/outside (outside is always at the frame's extent); ``fill`` zero/nan
      (nan forces float32); the dim lever: in 3D the Z range shrinks too and objects are 3D
      components, in 2D every object is a column of the stack and Z is kept whole.
    * **Footprint** ``WHOLE_SERIES``, no kernel axes: a window depends on every frame of the
      region, so nothing here is tileable; eager, realized through ``dense_output``.
    * **What follows the pixels.** Voxel layers are windowed and blanked like the image;
      per-position layers are reindexed to the output positions; Label and Point rows
      outside the kept pixels are dropped and the rest shifted into their window (and, under
      ``each``, moved to their object's position); per-M metadata follows the source position
      of each window, ``origin_um`` is restated per window from the field box, and under
      ``each`` the positions are named ``obj1…`` (``B03_obj1`` on a multipoint).
    """
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("crop by region needs an image provider on its `data` input")
    ax = ds.axes
    modes = ctx.params.get("__modes__", {}) or {}
    extent = str(modes.get("extent") or "frame")
    objects = str(modes.get("objects") or "components")
    keep_mode = str(modes.get(KEEP_MODE) or "inside")
    fill_mode = str(modes.get("fill") or "zero")
    is_3d = str(modes.get("dim") or "2D") == "3D"
    for val, allowed, what in ((extent, _EXTENTS, "extent"), (objects, _OBJECTS, "objects"),
                               (keep_mode, _KEEP, "keep"), (fill_mode, _FILL, "fill")):
        if val not in allowed:
            raise ValueError(f"crop by region: unknown {what} {val!r} — one of {list(allowed)}")
    raw_margin = ctx.params.get("margin")
    margin = max(0, int(raw_margin)) if raw_margin is not None else 0

    # ── the region: a raster on `regions` (wired) or on `data` ──────────────────
    regions = ctx.input("regions")
    src = regions if regions is not None else ds
    where = "the `regions` input" if regions is not None else "the `data` input"
    layer, _note = _resolve_layer(
        _voxel_layers(src), ctx.layer("region"), node="crop by region", socket="region",
        what="Voxel layer", where=where,
        remedy="make one upstream — Threshold, ROI Mask, Draw Regions or Connected "
               "Components — on this branch or on a branch wired into `regions`", ctx=ctx)
    shape = ax.shape_for(Domain.VOXEL)
    raw_raster = src.get(Domain.VOXEL, layer).values
    inside6 = broadcast_raster(raw_raster, shape, node="crop by region",
                               what=f"the region {layer!r}")
    raster6 = np.broadcast_to(np.asarray(raw_raster), shape) if objects == "labels" \
        else inside6
    if keep_mode == "outside":
        keep6, extent = ~inside6, "frame"
    else:
        keep6 = inside6
    if not keep6.any():
        raise ValueError(
            f"crop by region: nothing is kept — the region {layer!r} is "
            f"{'everywhere' if keep_mode == 'outside' else 'empty'} on every frame, so the "
            f"result would have no pixels. Check the mask upstream (or flip `keep`).")

    # ── the windows ────────────────────────────────────────────────────────────
    wins = _windows(inside6, raster6, ax, extent=extent, objects=objects, is_3d=is_3d,
                    margin=margin)
    if not wins:
        raise ValueError(
            f"crop by region: the region {layer!r} marks nothing on any position, so there "
            f"is nothing to crop to")
    D = max(w.z1 - w.z0 for w in wins)
    H = max(w.y1 - w.y0 for w in wins)
    W = max(w.x1 - w.x0 for w in wins)
    z_cropped = any(w.z0 != 0 or w.z1 != ax.z for w in wins) or D != ax.z
    n_out = len(wins)

    # which window each kept voxel belongs to (0 = none): the pixel mask of every window at
    # once, and the lookup a structure row uses to find its new position and offset
    widx6 = np.zeros(shape, dtype=np.int32)
    for i, w in enumerate(wins, start=1):
        sub = widx6[w.m, :, w.z0:w.z1, :, w.y0:w.y1, w.x0:w.x1]
        k = keep6[w.m, :, w.z0:w.z1, :, w.y0:w.y1, w.x0:w.x1]
        if w.obj is not None:
            k = k & w.obj[None, :, None, :, :]
        sub[k] = i

    # ── the pixels ─────────────────────────────────────────────────────────────
    first = np.asarray(prov.get_region(0, wins[0].m, 0, wins[0].z0, 0, wins[0].y0,
                                       wins[0].y0 + 1, wins[0].x0, wins[0].x0 + 1))
    fill_nan = fill_mode == "nan"
    dtype = np.float32 if fill_nan else first.dtype
    fill = np.nan if fill_nan else 0
    out_arr = dense_output((n_out, ax.t, D, ax.c, H, W), dtype, tag=f"crop_region_{ctx.node_id}")
    arr = out_arr.array
    if fill_nan:
        arr[...] = np.nan
    total = sum((w.z1 - w.z0) for w in wins) * ax.t * ax.c
    done = 0
    ctx.progress(0, total, f"cropping to {layer!r} ({extent}, {n_out} window(s))", frames=ax.t)
    for i, w in enumerate(wins):
        h, wd = w.y1 - w.y0, w.x1 - w.x0
        for t in range(ax.t):
            for z in range(w.z0, w.z1):
                for c in range(ax.c):
                    plane = np.asarray(prov.get_region(0, w.m, t, z, c, w.y0, w.y1, w.x0, w.x1))
                    k = widx6[w.m, t, z, c, w.y0:w.y1, w.x0:w.x1] == (i + 1)
                    arr[i, t, z - w.z0, c, :h, :wd] = np.where(k, plane, fill)
                    done += 1
            ctx.progress(done, total, f"cropping to {layer!r}", frames=ax.t)

    # ── the Dataset: axes, layers, rows, metadata ──────────────────────────────
    new_axes = replace(ax, m=n_out, z=D, y=H, x=W)
    out = ds.with_image(ArrayProvider(out_arr.seal())).reshaped_axes(new_axes)
    for attr in list(ds.attributes.values()):
        if not is_lattice(attr.domain):
            continue
        arr0 = np.asarray(attr.values)
        if tuple(arr0.shape) != ax.shape_for(attr.domain):
            continue                                     # stale before this ran
        axes_in = ax.axis_list(attr.domain)
        if attr.domain is Domain.VOXEL:
            new = np.zeros((n_out, ax.t, D, ax.c, H, W), dtype=arr0.dtype)
            for i, w in enumerate(wins):
                sub = arr0[w.m, :, w.z0:w.z1, :, w.y0:w.y1, w.x0:w.x1]
                k = widx6[w.m, :, w.z0:w.z1, :, w.y0:w.y1, w.x0:w.x1] == (i + 1)
                new[i, :, :w.z1 - w.z0, :, :w.y1 - w.y0, :w.x1 - w.x0] = np.where(k, sub, 0)
        elif "z" in axes_in and z_cropped:
            continue                                     # a per-plane value of cut planes
        elif "m" in axes_in:
            new = np.take(arr0, [w.m for w in wins], axis=axes_in.index("m"))
        else:
            new = arr0
        out = out.with_attribute(AttributeLayer(attr.domain, attr.name, new, attr.layer))
    out = _crop_rows(out, ds, wins, widx6, z_cropped=z_cropped, identity=(extent == "frame"))

    if extent == "frame":
        return out                                       # nothing moved: no stamp, same keys
    md = ds.metadata
    src_ms = [w.m for w in wins]
    changes: Dict[str, Any] = dict(position_subset(md, src_ms))
    origins: Optional[List[List[float]]] = []
    old_origin = md.get("origin_um")
    z_step = md.get("z_step_um")
    for w in wins:
        box = field_box(md, ax, w.m)
        if box is None:
            origins = None
            break
        sub = sub_field_box(box, (w.y0 / ax.y, w.y1 / ax.y, w.x0 / ax.x, w.x1 / ax.x))
        z_um = 0.0
        if isinstance(old_origin, (list, tuple)) and w.m < len(old_origin):
            try:
                z_um = float(old_origin[w.m][0])
            except (TypeError, ValueError, IndexError):
                z_um = float(box.z0 or 0.0)
        elif box.z0 is not None:
            z_um = float(box.z0)
        if z_cropped and z_step:
            try:
                z_um += float(z_step) * w.z0
            except (TypeError, ValueError):
                pass
        origins.append([z_um, float(sub.y0), float(sub.x0)])
    if origins is not None:
        changes["origin_um"] = origins
    elif old_origin is not None:
        changes["origin_um"] = None
    for key in _GEOMETRY_POSITION_KEYS:
        if md.get(key) is not None:
            changes[key] = None
    if extent == "each":
        pnames = _source_position_names(md, ax.m)
        counters: Dict[int, int] = {}
        names = []
        for w in wins:
            counters[w.m] = counters.get(w.m, 0) + 1
            names.append(f"obj{counters[w.m]}" if ax.m == 1
                         else f"{pnames[w.m]}_obj{counters[w.m]}")
        changes["position_name"] = names
        changes["position_index"] = None
    out = out.with_metadata(**changes)
    spec = ";".join(f"m{w.m}z{w.z0}:{w.z1}y{w.y0}:{w.y1}x{w.x0}:{w.x1}" for w in wins)
    if len(spec) > 160:
        spec = digest("crop_region", spec)[:16]
    return _sampled(out, f"crop_region[{extent}:{spec}]")


def _crop_rows(out: Dataset, ds: Dataset, wins: Sequence[_Win], widx6: np.ndarray, *,
               z_cropped: bool, identity: bool) -> Dataset:
    """Label and Point rows: keep the ones whose centroid voxel is kept, move each into its
    window (y/x — and z when planes were cut — minus the window's corner) and onto the
    output position its window became. A table whose rows all survive an identity window is
    left untouched (same layers, same revisions)."""
    ax = ds.axes
    y0s = np.array([w.y0 for w in wins], dtype=float)
    x0s = np.array([w.x0 for w in wins], dtype=float)
    z0s = np.array([w.z0 for w in wins], dtype=float)
    for domain in (Domain.LABEL, Domain.POINT):
        for table in _structure_layers(ds, domain):
            cols = {k[2]: np.asarray(ds.attributes[k].values) for k in ds.attributes
                    if k[0] is domain and k[1] == table}
            if not {"m", "t", "y", "x"} <= set(cols) or not len(cols["m"]):
                continue
            clamp = lambda v, n: np.clip(np.rint(np.asarray(v, dtype=float)).astype(int), 0, n - 1)
            mi, ti = clamp(cols["m"], ax.m), clamp(cols["t"], ax.t)
            ci = clamp(cols["c"], ax.c) if "c" in cols else np.zeros(len(mi), int)
            zi = clamp(cols["z"], ax.z) if "z" in cols else np.zeros(len(mi), int)
            yi, xi = clamp(cols["y"], ax.y), clamp(cols["x"], ax.x)
            idx = widx6[mi, ti, zi, ci, yi, xi] - 1
            keep = idx >= 0
            if identity and keep.all():
                continue
            for k in list(out.attributes):
                if k[0] is domain and k[1] == table:
                    out = out.without(domain, k[2], layer=table)
            if not keep.any():
                continue
            new = {k: v[keep] for k, v in cols.items()}
            sel = idx[keep]
            new["m"] = sel.astype(np.int64)
            new["y"] = np.asarray(new["y"], dtype=float) - y0s[sel]
            new["x"] = np.asarray(new["x"], dtype=float) - x0s[sel]
            if "z" in new and z_cropped:
                new["z"] = np.asarray(new["z"], dtype=float) - z0s[sel]
            zk = ds.structure_zkind(domain, table)
            out = out.with_structure(StructureTable(domain, new, layer=table,
                                                    z_kind=zk or "subpixel"))
    return out


register_node(
    batch_aware(_compute_crop_region), op_key=CROP_REGION_OP, label="Crop by Region",
    category="utility",
    inputs=[
        InDataset(description=
                  "The image to crop. Its masks are windowed and blanked with it, its Label "
                  "and Point rows follow (rows outside are dropped, the rest shifted into "
                  "the window), its per-position metadata follows each window's source "
                  "position, and the field's corner is restated so the crop can be placed "
                  "back."),
        InDataset("regions", label="Regions", passes_domains=False,
                  description=
                  "Optional: a second branch carrying the region — a mask thresholded on "
                  "another channel, Draw Regions' patches, a segmentation. Unwired, the "
                  "region is read off `data`. Only the raster is read from it; its image "
                  "and other layers do not pass downstream. It may be narrower than the "
                  "data on m/t/z/c (size 1 applies to all) but must match on y/x."),
        InString("region", "Region layer", field=False, default="", layer_in=Domain.VOXEL,
                 layer_from="regions",
                 description=
                 "Which Voxel raster is the region: nonzero is inside. A Threshold mask, an "
                 "ROI Mask, Draw Regions' label raster, Connected Components' ids — any of "
                 "them. Empty takes the only raster on the wire it reads (`Regions` if "
                 "wired, else `data`) and refuses when there are several, listing them. "
                 "Regions need not be rectangular or connected: `fit` boxes them all, "
                 "`each` cuts every one out."),
        InInt("margin", "Margin", unit="px", field=False, default=0,
              available_in={"extent": frozenset({"fit", "each"})},
              description=
              "Pixels of context added on every side of a window (y and x; Z is never "
              "padded) under `fit` and `each`, clamped at the frame edge. The pixels in the "
              "margin that are outside the region are still filled — the margin changes the "
              "window, not what is kept — so a filter downstream has room for its halo "
              "without reading a neighbour's pixels. 0 is the tight box."),
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("extent", list(_EXTENTS), default="frame", label="Extent",
             description=
             "How big the result is. The region decides WHICH pixels survive; this decides "
             "the frame they survive in — the whole image, the box around them, or one "
             "window per object. The card's `outside` socket is always at the frame's "
             "extent, whatever is chosen here.",
             choice_docs={
                 "frame": "Keep the image's full extent and fill every pixel outside the "
                          "region. Nothing moves: coordinates, positions and calibration are "
                          "unchanged, so a measurement downstream lands where it would have. "
                          "This is masking — the choice when the crop must stay comparable "
                          "to the uncropped image.",
                 "fit": "Shrink to the bounding box of everything inside the region, plus "
                        "`Margin`; pixels inside the box but outside the region are still "
                        "filled. The field's corner moves and the payload says where "
                        "(`origin_um`), so Stitch, Overlay and Canvas place it back. Several "
                        "positions are each boxed and padded to a common size. A position "
                        "with nothing inside is dropped.",
                 "each": "Cut every object out into a POSITION of its own: the output's M "
                         "axis is the objects (`Objects` says what an object is), each in a "
                         "window of the common size anchored at its own corner, with its "
                         "`origin_um` and a `position_name` (`obj1`, `obj2`, … or "
                         "`B03_obj1` on a multipoint). Split Positions then fans them out, "
                         "and every card downstream reads which object it is on. Pixels of a "
                         "neighbouring object that fall inside a window are filled, so each "
                         "window holds exactly one object.",
             }),
        Mode("objects", list(_OBJECTS), default="components", label="Objects",
             available_in={"extent": frozenset({"each"})},
             description=
             "What counts as one object under `each`. Folds into the memo key.",
             choice_docs={
                 "components": "A connected region of the region's FOOTPRINT over every "
                               "frame — in 2D, 8-connected on the Z-collapsed plane so an "
                               "object is one column of the stack across all timepoints; in "
                               "3D, 26-connected in the volume. A cell that moves over time "
                               "is one object whose window covers everywhere it went; two "
                               "cells that touch are one object.",
                 "labels": "One object per distinct nonzero VALUE of the raster — a "
                           "segmentation's ids, Draw Regions' numbered patches. Touching "
                           "objects stay separate. Note that most producers number per "
                           "frame, so on a series a cell gets a different id on every frame "
                           "and each becomes its own object; use `components` for a series, "
                           "`labels` for a single frame or a tracked raster.",
             }),
        Mode(KEEP_MODE, list(_KEEP), default="inside", label="Keep",
             description=
             "Which side of the region this node's `out` carries. The card's `outside` "
             "socket is the other side regardless, so one card gives both.",
             choice_docs={
                 "inside": "Keep the pixels inside the region and fill the rest, at the "
                           "extent chosen above. The crop.",
                 "outside": "Keep the pixels OUTSIDE the region and fill the inside, always "
                            "at the frame's full extent (the complement of a box is not a "
                            "box). The inverse — the background around the objects, the "
                            "field with the cells removed. What the `outside` socket "
                            "carries.",
             }),
        Mode("fill", list(_FILL), default="zero", label="Fill",
             description=
             "The value written where a pixel is not kept. It decides what a measurement "
             "downstream sees in the blanked area, so choose by what reads it next.",
             choice_docs={
                 "zero": "Write 0. The output keeps the image's own dtype (a uint16 stays "
                         "uint16), and every downstream node accepts it; but a mean over a "
                         "window that includes blanked pixels is pulled toward 0, and a "
                         "threshold sees them as background.",
                 "nan": "Write NaN, which no statistic can mistake for a measurement — a "
                        "NaN-aware mean ignores the blanked pixels. Forces a float32 output, "
                        "so a node that needs integer counts downstream (Histogram "
                        "Threshold) will refuse it.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    meta_transform=_meta_crop_region,
    description="Crop an image to a MASK, drawn regions or a label raster instead of a "
                "rectangle: keep the frame and blank outside, shrink to the box around "
                "everything inside, or cut every object out into a position of its own that "
                "keeps its place in the field; the card's `outside` socket is the inverse. "
                "Masks, rows and per-position metadata follow the pixels.")
