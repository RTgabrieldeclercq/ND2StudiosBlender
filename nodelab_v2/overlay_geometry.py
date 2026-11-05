"""Overlay GEOMETRY from a pulled Dataset — the marks the overlay renderer paints, as
plain functions (V5.00 step 3; moved out of :class:`nodelab_v2.viewer.ViewerPanel`).

The Viewer and the Figure Editor's image item draw the same overlays on the same kind of
payload, so what a frame's overlays ARE — which Point rows belong on it, every track's
positions ordered by t, the label raster's plane and its identity palette, a scalar wash, a
field raster, the vectors, the correlation grid, the mesh cross-sections — is computed
here, once, from ``(dataset, coords, settings)``, and both callers hand the result to
:class:`nodelab_v2.overlays.OverlayRenderer`. Each function is the Viewer's former method
with its state made explicit: the payload, the coordinate INTO the payload, the overlay
settings group it reads, the identity palette, and (for the label source) the viewed
node's own layer names. The docstrings carry the reasoning the methods carried.

Everything returns the renderer's mark records (:class:`~nodelab_v2.overlays.PointMark`,
``TrackPath``, ``MeshSection``) or numpy arrays; nothing here paints.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph import fieldmap as FV
from nodegraph.domains import Domain
from nodegraph.mesh import mesh_part as MESH_PART
from nodelab_v2 import overlays as OV

#: Past this many physical objects the neighbour de-confliction is skipped (each object
#: still keeps one colour across T) — the palette would otherwise dominate a repaint.
MAX_PALETTE_OBJECTS = 10_000
#: How many ``(m, t)`` frames the neighbour graph samples — the same cells are neighbours
#: in every frame, so sampling costs nothing in quality.
MAX_NEIGHBOR_FRAMES = 64

#: one-shot stderr warnings, process-wide (the Viewer's were per panel)
_warned: set = set()


def _warn_once(key: str, text: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(text, file=sys.stderr, flush=True)


@dataclass
class Palette:
    """A dataset's **identity palette**: per layer, an ``id → palette slot`` LUT.

    Built once per dataset by :func:`build_palette` and read by every overlay that colours
    per item. Dense int arrays rather than dicts because the label fill indexes one
    straight into its per-pixel colour table.
    """
    labels: Dict[Any, np.ndarray] = dc_field(default_factory=dict)   # (LABEL, layer) → LUT
    points: Dict[Any, np.ndarray] = dc_field(default_factory=dict)   # (POINT, layer) → LUT
    tracks: Dict[Any, np.ndarray] = dc_field(default_factory=dict)   # track layer → LUT


# ── tables of a payload ───────────────────────────────────────────────────────────────
def _attrs(ds: Any) -> Dict[tuple, Any]:
    return ds.attributes if ds is not None and hasattr(ds, "attributes") else {}


def drop_synth(ds: Any, by_layer: Dict[Any, Dict[str, np.ndarray]],
               domain: Domain) -> Dict[Any, Dict[str, np.ndarray]]:
    """``by_layer`` without the tables a table node wrote (V4.00): a Table Concat holds
    other inputs' rows at positions on THEIR images, a Table Join repeats its left table's
    — drawn over this image they would be detections it does not have. Nor a PIXEL table
    (Measure, Members = pixel): its rows are the image's own voxels, up to millions of
    them, and one marker each would bury the picture. A picker naming one still shows it."""
    from nodegraph.catalog._shared.table_ops import synthesized
    from nodegraph.tables import pixel_layers
    md = getattr(ds, "metadata", None) or {}
    skip = synthesized(md, domain)
    if domain is Domain.POINT:
        skip = skip | pixel_layers(md)
    if not skip:
        return by_layer
    return {k: v for k, v in by_layer.items() if str(k) not in skip}


def member_layers(ds: Any, *, on_ragged: Optional[Any] = None
                  ) -> Dict[tuple, Dict[str, np.ndarray]]:
    """``{(domain, layer): {col: values}}`` for every Point/Label layer that carries a
    joinable ``id,y,x`` triple — the member position source for Track trajectories.

    **Every layer is clipped to its shortest column** (V2.27): a producer that emits a
    column with a different row count leaves the instance RAGGED, and an out-of-range
    index in a paint call is a crash. The rows every column agrees on survive; the tail
    is dropped and said once on stderr."""
    out: Dict[tuple, Dict[str, np.ndarray]] = {}
    attrs = _attrs(ds)
    if not attrs:
        return out
    from nodegraph.tables import pixel_layers
    pixels = pixel_layers(ds.metadata)
    for (dom, layer, name), attr in attrs.items():
        if dom in (Domain.POINT, Domain.LABEL) and not (
                dom is Domain.POINT and str(layer) in pixels):
            out.setdefault((dom, layer), {})[name] = attr.values
    keep = {k: v for k, v in out.items() if {"id", "y", "x"} <= set(v)}
    for key, cols in keep.items():
        try:
            lens = {name: len(val) for name, val in cols.items()}
        except TypeError:                       # a scalar attribute — not a column
            continue
        if len(set(lens.values())) <= 1:
            continue
        n = min(lens.values())
        worst = sorted(lens.items(), key=lambda kv: kv[1])
        _warn_once(f"ragged:{key}",
                   f"[overlays] {key[0].value} layer {key[1]!r} is ragged — its columns "
                   f"disagree in length ({worst[0][0]}={worst[0][1]}, "
                   f"{worst[-1][0]}={worst[-1][1]}); drawing the first {n} rows")
        if on_ragged is not None:
            on_ragged(key, lens)
        keep[key] = {name: np.asarray(val)[:n] for name, val in cols.items()}
    return keep


def track_layers(ds: Any) -> Dict[Any, Dict[str, np.ndarray]]:
    """``{layer: {col: values}}`` for every Track-domain table in the dataset."""
    out: Dict[Any, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in _attrs(ds).items():
        if dom is Domain.TRACK:
            out.setdefault(layer, {})[name] = attr.values
    return drop_synth(ds, out, Domain.TRACK)


def best_member_layer(members: Dict[tuple, Dict[str, np.ndarray]],
                      member_ids: np.ndarray) -> Optional[tuple]:
    """Which member layer a Track table is talking about: Point and Label ids share an id
    space (both start at 1), so the layer is resolved by *coverage* — the one holding the
    most of this table's member ids; ties go to the lexicographically first key."""
    want = set(np.asarray(member_ids).astype(np.int64).tolist())
    best, best_cov = None, 0
    for key in sorted(members, key=lambda k: (str(k[0]), str(k[1]))):
        ids = np.asarray(members[key]["id"]).astype(np.int64).tolist()
        cov = len(want & set(ids))
        if cov > best_cov:
            best, best_cov = key, cov
    return best


# ── the identity palette (V2.17) ──────────────────────────────────────────────────────
def build_palette(ds: Any, *, on_ragged: Optional[Any] = None) -> Palette:
    """One palette slot per **physical object**, indexed by member id. ``on_ragged``
    is told ``(layer key, column lengths)`` when a member layer had to be clipped.

    1. **Objects.** Every Point/Label row belongs to a track (its object is that track) or
       to nothing (its object is itself). A track *prefers* the slot of its first member,
       so at the frame where a cell first appears it keeps the colour it would have had
       untracked — and holds it for the rest of the series.
    2. **Neighbours.** Per ``(m, t)``, each member's nearest few members; mapped to object
       pairs and deduplicated across frames.
    3. **Slots.** :func:`~nodelab_v2.overlays.deconflict_slots` hands out the slots, moving
       an object off its preference only when a neighbour is too close on the colour wheel.

    Anything unexpected in the tables is skipped rather than raised on: this feeds a paint
    call, and a colour is never worth a crash."""
    members = member_layers(ds, on_ragged=on_ragged)
    if not members:
        return Palette()
    tracks = track_layers(ds)

    obj_row: Dict[tuple, np.ndarray] = {}
    ids_of: Dict[tuple, np.ndarray] = {}
    for key, cols in members.items():
        ids = np.asarray(cols["id"]).astype(np.int64)
        ids_of[key] = ids
        obj_row[key] = np.full(len(ids), -1, dtype=np.int64)
    prefer: Dict[int, int] = {}
    # each layer gets its own disjoint band of preferences — two layers numbering their
    # ids from 0 would otherwise prefer, and keep, the same slots
    layer_base: Dict[Any, int] = {}
    _base = 0
    for key in sorted(ids_of, key=str):
        layer_base[key] = _base
        ids = ids_of[key]
        _base += int(max(0, int(ids.max()) + 1)) if ids.size else 0
    track_slots: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
    next_obj = 0
    for tlayer in sorted(tracks, key=str):
        cols = tracks[tlayer]
        if not {"track_id", "member_id"} <= set(cols):
            continue
        tid = np.asarray(cols["track_id"]).astype(np.int64)
        mem = np.asarray(cols["member_id"]).astype(np.int64)
        mkey = best_member_layer(members, mem)
        if mkey is None:
            continue
        ids = ids_of[mkey]
        if ids.size == 0:
            continue
        order = np.argsort(ids, kind="stable")
        pos = np.searchsorted(ids[order], mem)
        hit = (pos < len(ids)) & (ids[order][np.clip(pos, 0, len(ids) - 1)] == mem)
        rows = order[np.clip(pos, 0, max(0, len(ids) - 1))][hit]
        tid_hit = tid[hit]
        if rows.size == 0:
            continue
        uniq, inv = np.unique(tid_hit, return_inverse=True)
        objs = np.arange(next_obj, next_obj + len(uniq), dtype=np.int64)
        next_obj += len(uniq)
        obj_row[mkey][rows] = objs[inv]
        first = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, inv, ids[rows])
        prefer.update({int(o): int(f) + layer_base.get(mkey, 0)
                       for o, f in zip(objs.tolist(), first.tolist())})
        track_slots[tlayer] = (uniq, objs)
    for key, ids in ids_of.items():             # untracked rows: each is its own object
        loose = np.nonzero(obj_row[key] < 0)[0]
        if loose.size:
            objs = np.arange(next_obj, next_obj + loose.size, dtype=np.int64)
            next_obj += loose.size
            obj_row[key][loose] = objs
            prefer.update({int(o): int(v) + layer_base.get(key, 0)
                           for o, v in zip(objs.tolist(), ids[loose].tolist())})

    too_many = next_obj > MAX_PALETTE_OBJECTS
    if too_many:
        _warn_once("palette", f"[overlays] {next_obj} objects (>{MAX_PALETTE_OBJECTS}) - "
                              f"neighbouring colours not de-conflicted; each object still "
                              f"keeps one colour across T")
    chunks: List[np.ndarray] = []
    for key, cols in members.items():
        ids = ids_of[key]
        if len(ids) < 2 or too_many:
            continue
        ys = np.asarray(cols["y"], dtype=float)
        xs = np.asarray(cols["x"], dtype=float)
        zeros = np.zeros(len(ids), dtype=np.int64)
        ms = np.asarray(cols["m"]).astype(np.int64) if "m" in cols else zeros
        ts = np.asarray(cols["t"]).astype(np.int64) if "t" in cols else zeros
        frame = ms * (int(ts.max()) + 1) + ts if ts.size else zeros
        frames = np.unique(frame)
        if len(frames) > MAX_NEIGHBOR_FRAMES:
            frames = frames[np.linspace(0, len(frames) - 1,
                                        MAX_NEIGHBOR_FRAMES).astype(int)]
        for f in frames.tolist():
            sel = np.nonzero(frame == f)[0]
            if sel.size < 2:
                continue
            pairs = OV.neighbor_pairs(np.stack([ys[sel], xs[sel]], axis=1))
            if pairs.size:
                chunks.append(obj_row[key][sel][pairs])
    pairs_all = (np.unique(np.concatenate(chunks), axis=0) if chunks
                 else np.zeros((0, 2), dtype=np.int64))
    pairs_all = pairs_all[pairs_all[:, 0] != pairs_all[:, 1]]

    slots = (dict(prefer) if too_many else
             OV.deconflict_slots(prefer, [(int(a), int(b)) for a, b in pairs_all.tolist()]))
    lookup = np.arange(max(1, next_obj), dtype=np.int64)
    for o, sl in slots.items():
        lookup[o] = sl
    pal = Palette()
    for key, ids in ids_of.items():
        lut = OV.slot_lut(ids, lookup[obj_row[key]])
        (pal.labels if key[0] is Domain.LABEL else pal.points)[key] = lut
    for tlayer, (tids, objs) in track_slots.items():
        pal.tracks[tlayer] = OV.slot_lut(tids, lookup[objs])
    return pal


# ── points ────────────────────────────────────────────────────────────────────────────
def point_marks(ds: Any, coords: Tuple[int, int, int, int], s: Any, palette: Palette
                ) -> List[OV.PointMark]:
    """Every Point-domain row that belongs on the frame, as draw-ready marks.

    Positions stay in *structure* coordinates (full-resolution image pixels); the renderer
    scales them into the displayed plane. The colour key is the point's **palette slot**;
    the layer index is the per-layer key. ``s.z_project`` is authoritative for every layer,
    3-D ones included: off, only the detections whose ``z`` lands on the viewed plane are
    drawn; on, off-plane detections come along flagged ``on_plane=False``."""
    attrs = _attrs(ds)
    if not attrs:
        return []
    m, t, z, _c = coords
    project = bool(getattr(s, "z_project", False))
    by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in attrs.items():
        if dom is Domain.POINT:
            by_layer.setdefault(layer, {})[name] = attr.values
    want = str(getattr(s, "layer", "") or "").strip()
    if want and want in {str(k) for k in by_layer}:
        by_layer = {k: v for k, v in by_layer.items() if str(k) == want}
    else:
        by_layer = drop_synth(ds, by_layer, Domain.POINT)
        _grids = (getattr(ds, "metadata", None) or {}).get("__analysis_mesh__") or {}
        by_layer = {k: v for k, v in by_layer.items() if str(k) not in _grids}
    out: List[OV.PointMark] = []
    for li, layer in enumerate(sorted(by_layer, key=lambda v: str(v))):
        cols = by_layer[layer]
        if "y" not in cols or "x" not in cols:
            continue
        ys, xs = cols["y"], cols["x"]
        zs = cols.get("z"); ms = cols.get("m"); ts = cols.get("t")
        ids = cols.get("id")
        slots = palette.points.get((Domain.POINT, layer))
        for i in range(len(ys)):
            if ms is not None and int(ms[i]) != m:
                continue
            if ts is not None and int(ts[i]) != t:
                continue
            zpl = z if zs is None else int(round(float(zs[i])))
            on_plane = zpl == z
            if not on_plane and not project:
                continue
            key = OV.slot_of(slots, int(ids[i])) if ids is not None else i + 1
            out.append(OV.PointMark(float(ys[i]), float(xs[i]), key, li, on_plane, zpl))
    return out


# ── tracks ────────────────────────────────────────────────────────────────────────────
def track_paths(ds: Any, coords: Tuple[int, int, int, int], palette: Palette
                ) -> List[OV.TrackPath]:
    """For the frame's M, one :class:`~nodelab_v2.overlays.TrackPath` per track — its
    members' positions ordered by ``t``, the index of the vertex at the frame's T, each
    vertex's timepoint, and its palette slot (the same colour as the regions it threads
    through). A Track layer stores only ``track_id, t, member_id``; the geometry lives on
    the member layer that best covers the member ids."""
    attrs = _attrs(ds)
    if not attrs:
        return []
    m, t, _z, _c = coords
    tl = track_layers(ds)
    if not tl:
        return []
    members = member_layers(ds)
    trajectories: List[OV.TrackPath] = []
    for tlayer, cols in tl.items():
        if not {"track_id", "t", "member_id"} <= set(cols):
            continue
        tid = np.asarray(cols["track_id"]).astype(np.int64)
        tt = np.asarray(cols["t"]).astype(np.int64)
        mem = np.asarray(cols["member_id"]).astype(np.int64)
        mkey = best_member_layer(members, mem)
        if mkey is None:
            continue
        best_cols = members[mkey]
        slots = palette.tracks.get(tlayer)
        ids = np.asarray(best_cols["id"]).astype(np.int64)
        ys = np.asarray(best_cols["y"], dtype=float)
        xs = np.asarray(best_cols["x"], dtype=float)
        ms = np.asarray(best_cols["m"]).astype(np.int64) if "m" in best_cols else None
        id_to_i = {int(v): i for i, v in enumerate(ids.tolist())}
        for track in np.unique(tid).tolist():
            sel = tid == track
            order = np.argsort(tt[sel], kind="stable")
            ts_ord = tt[sel][order].tolist()
            mem_ord = mem[sel][order].tolist()
            path, cur, ts_kept = [], None, []
            for tv, mv in zip(ts_ord, mem_ord):
                i = id_to_i.get(int(mv))
                if i is None or (ms is not None and int(ms[i]) != m):
                    continue
                path.append((float(ys[i]), float(xs[i])))
                ts_kept.append(int(tv))
                if int(tv) == t:
                    cur = len(path) - 1
            if path:
                trajectories.append(OV.TrackPath(int(track), path, cur, ts_kept,
                                                 OV.slot_of(slots, int(track))))
    return trajectories


# ── label rasters ─────────────────────────────────────────────────────────────────────
def label_rasters(ds: Any) -> Dict[str, np.ndarray]:
    """Every full-resolution 6-D integer Voxel layer of ``ds``, by name."""
    rasters: Dict[str, np.ndarray] = {}
    for (dom, _layer, name), attr in _attrs(ds).items():
        if dom is not Domain.VOXEL or not name:
            continue
        vals = attr.values
        if np.issubdtype(vals.dtype, np.integer) and vals.ndim == 6:
            rasters[str(name)] = vals
    return rasters


def label_source(ds: Any, coords: Tuple[int, int, int, int], want: str = "",
                 own_layers: Optional[Sequence[str]] = None) -> Optional[np.ndarray]:
    """The full-resolution 6-D integer Voxel layer the Labels overlay draws.

    An explicit ``want`` wins when present (empty plane or not); a stale name falls
    through to the guess. Auto prefers the viewed node's own output layers in declaration
    order, then the raster with the most regions in the viewed plane — counted, not the
    largest id, which a raster carrying another node's numbering wins on while showing
    almost nothing."""
    rasters = label_rasters(ds)
    if not rasters:
        return None
    want = str(want or "").strip()
    if want and want in rasters:
        return rasters[want]
    m, t, z, c = coords
    for nm in own_layers or ():
        if nm in rasters:
            return rasters[nm]
    best, best_regions = None, 0
    for vals in rasters.values():
        try:
            plane = vals[m, t, z, c]
        except IndexError:
            continue
        regions = int(np.count_nonzero(np.unique(plane)))
        if regions >= 1 and regions >= best_regions:
            best, best_regions = vals, regions
    return best


def decimate_to(plane: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """``plane`` thinned by whole steps to about ``ref_shape`` (the displayed plane)."""
    ay, ax_ = plane.shape[:2]
    ty = max(1, int(np.ceil(ay / max(1, ref_shape[0]))))
    tx = max(1, int(np.ceil(ax_ / max(1, ref_shape[1]))))
    return plane[::ty, ::tx]


def label_plane(source: Optional[np.ndarray], coords: Tuple[int, int, int, int],
                ref_shape: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    """The viewed plane of a label raster, decimated to the reference plane."""
    if source is None or ref_shape is None:
        return None
    m, t, z, c = coords
    try:
        best = source[m, t, z, c]
    except IndexError:
        return None
    return decimate_to(best, ref_shape)


def label_keys(lab: Optional[np.ndarray], palette: Palette) -> Optional[np.ndarray]:
    """The ``label id → palette slot`` LUT for the plane: with one Label layer there is
    nothing to resolve; with several, the plane's top id says which id space it came from."""
    if lab is None or lab.size == 0 or not palette.labels:
        return None
    luts = [palette.labels[k] for k in sorted(palette.labels, key=lambda k: str(k[1]))]
    if len(luts) == 1:
        return luts[0]
    top = int(lab.max())
    covering = [lut for lut in luts if top < len(lut)]
    return covering[0] if covering else max(luts, key=len)


def scalar_plane(ds: Any, coords: Tuple[int, int, int, int],
                 ref_shape: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    """The Voxel-domain SCALAR layer for the viewed plane, or ``None``: anything
    non-integer is a measurement, an integer raster is a label map (the Labels overlay's);
    a boolean mask is admitted. Picks the layer with the largest finite spread."""
    attrs = _attrs(ds)
    if not attrs or ref_shape is None:
        return None
    m, t, z, c = coords
    best, best_spread = None, -1.0
    for (dom, _layer, _name), attr in attrs.items():
        if dom is not Domain.VOXEL:
            continue
        vals = attr.values
        if vals.ndim != 6:
            continue
        if np.issubdtype(vals.dtype, np.integer) and vals.dtype != np.bool_:
            continue
        try:
            plane = np.asarray(vals[m, t, z, c], dtype=float)
        except IndexError:
            continue
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            continue
        spread = float(finite.max() - finite.min())
        if spread >= best_spread:
            best, best_spread = plane, spread
    if best is None:
        return None
    return decimate_to(best, ref_shape)


# ── mesh ──────────────────────────────────────────────────────────────────────────────
def mesh_sections(ds: Any, coords: Tuple[int, int, int, int], s: Any
                  ) -> List[OV.MeshSection]:
    """Every Mesh element on the frame, cut by the viewed Z plane (or its near-plane
    vertices under the ``points`` style). Walks the three CSR strata directly so a
    partially-written or foreign mesh is skipped, never raised on."""
    attrs = _attrs(ds)
    out: List[OV.MeshSection] = []
    if not attrs:
        return out
    m, t, z, c = coords
    near = max(0.0, float(getattr(s, "near_z", 1.0)))
    vertex_style = getattr(s, "style", "wireframe") == "points"
    by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in attrs.items():
        if dom is Domain.MESH and layer is not None:
            by_layer.setdefault(layer, {})[name] = attr.values
    for layer in sorted(by_layer, key=lambda v: str(v)):
        base, part = MESH_PART(str(layer))
        if part is not None:
            continue
        el = by_layer[layer]
        vt = by_layer.get(f"{base}/vert")
        fc = by_layer.get(f"{base}/face")
        if vt is None or fc is None:
            continue
        if not ({"id", "m", "t", "c", "vert_start", "vert_count", "face_start",
                 "face_count"} <= set(el) and {"z", "y", "x"} <= set(vt)
                and {"v0", "v1", "v2"} <= set(fc)):
            continue
        vz, vy, vx = (np.asarray(vt[k], dtype=float) for k in ("z", "y", "x"))
        tri = np.column_stack([np.asarray(fc[f"v{i}"], dtype=np.int64) for i in range(3)])
        verts = np.column_stack([vz, vy, vx])
        for row in range(len(np.asarray(el["id"]))):
            if (int(np.asarray(el["m"])[row]) != m or int(np.asarray(el["t"])[row]) != t
                    or int(np.asarray(el["c"])[row]) != c):
                continue
            vs = int(np.asarray(el["vert_start"])[row])
            vc = int(np.asarray(el["vert_count"])[row])
            fs = int(np.asarray(el["face_start"])[row])
            nf = int(np.asarray(el["face_count"])[row])
            oid = int(np.asarray(el["id"])[row])
            sub_v = verts[vs:vs + vc]
            sub_f = tri[fs:fs + nf] - vs
            sec = OV.MeshSection(oid)
            if vertex_style:
                keep = np.abs(sub_v[:, 0] - float(z)) <= near
                sec.verts = [(float(a), float(b)) for a, b in zip(sub_v[keep, 1], sub_v[keep, 2])]
            else:
                sec.loops, sec.closed = OV.mesh_section(sub_v, sub_f, float(z))
            if sec.loops or sec.verts:
                out.append(sec)
    return out


# ── fields, vectors, grids ────────────────────────────────────────────────────────────
def vector_info(ds: Any, coords: Tuple[int, int, int, int], layer: str = ""):
    """The Vectors tab's :class:`nodegraph.fieldmap.VectorField` for the plane."""
    if not _attrs(ds):
        return None
    m, t, z, c = coords
    return FV.vectors(ds, m, t, z, c, layer=layer)


def field_info(ds: Any, coords: Tuple[int, int, int, int], s: Any,
               cache: Optional[Dict[Any, Any]] = None):
    """``(raster, limits, colour map, colour-bar label)`` of the Field tab for the plane, or
    ``None``: the picked (or Auto) table and column, rasterised by
    :func:`nodegraph.fieldmap.raster`; limits by ``s.limits`` (``series`` percentiles are
    cached in ``cache`` per payload and column)."""
    if not _attrs(ds):
        return None
    layer, col = FV.auto_pick(ds, getattr(s, "layer", ""), getattr(s, "column", ""))
    if not col:
        return None
    m, t, z, c = coords
    fr = FV.raster(ds, layer, col, m, t, z, c)
    if fr is None:
        return None
    limits = getattr(s, "limits", "series")
    if limits == "manual":
        lims = (float(s.clim_lo), float(s.clim_hi))
    elif limits == "frame":
        g = fr.grid[np.isfinite(fr.grid)]
        lims = (tuple(float(v) for v in np.percentile(g, [1.0, 99.0])) if g.size
                else (0.0, 1.0))
    else:
        key = (id(ds), layer, col)
        if cache is not None and key in cache:
            lims = cache[key]
        else:
            lims = FV.series_limits(ds, layer, col)
            if cache is not None:
                if len(cache) > 32:
                    cache.clear()
                cache[key] = lims
    cmap_setting = getattr(s, "colormap", "auto")
    cmap = cmap_setting if cmap_setting != "auto" else (
        "coolwarm" if lims[0] < 0.0 < lims[1] else "turbo")
    unit = FV.column_unit(ds, layer, col)
    return fr, lims, cmap, (f"{col} ({unit})" if unit else col)


def glyph_info(ds: Any, coords: Tuple[int, int, int, int], s: Any):
    """The Field tab's principal-axis glyphs for the plane, or ``None``."""
    if not _attrs(ds):
        return None
    layer, _col = FV.auto_pick(ds, getattr(s, "layer", ""), "")
    if not layer:
        return None
    m, t, z, c = coords
    return FV.tensor_glyphs(ds, layer, getattr(s, "glyphs", "none"), m, t, z, c,
                            every=int(getattr(s, "glyph_every", 1)))


def deformed_lines(ds: Any, coords: Tuple[int, int, int, int], vectors_layer: str,
                   exaggerate: float):
    """The Grid tab's deformed lattice: the field's nodes joined through their displaced
    positions, ``exaggerate`` times the displacement."""
    if not _attrs(ds):
        return None
    m, t, z, c = coords
    return FV.deformed_lines(ds, vectors_layer, m, t, z, c, exaggerate=exaggerate) or None


def grid_rects(ds: Any, coords: Tuple[int, int, int, int], g: Any) -> Optional[np.ndarray]:
    """The correlation grid on the plane → ``(N, 5)`` ``y_lo, y_hi, x_lo, x_hi, level``
    boxes in full-resolution image pixels, from every ``<field>_mesh`` table the
    ``__analysis_mesh__`` metadata names; ``subsets`` draws the window around every Nth
    node of the field table instead."""
    attrs = _attrs(ds)
    if not attrs:
        return None
    marks = (getattr(ds, "metadata", None) or {}).get("__analysis_mesh__") or {}
    if not marks:
        return None
    m, t, z, _c = coords
    style = getattr(g, "style", "elements")
    every = max(1, int(getattr(g, "every", 1)))
    by_layer: Dict[str, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in attrs.items():
        if dom is Domain.POINT and layer:
            by_layer.setdefault(str(layer), {})[str(name)] = attr.values
    out: List[np.ndarray] = []
    for mesh_layer, info in marks.items():
        info = info if isinstance(info, dict) else {}
        kind = str(info.get("kind", ""))
        if style == "subsets":
            cols = by_layer.get(str(info.get("field", "")), {})
            win = info.get("window")
            if not cols or "y" not in cols or not win:
                continue
            keep = np.ones(len(cols["y"]), dtype=bool)
            for key, want in (("m", m), ("t", t)):
                if key in cols:
                    keep &= np.asarray(cols[key]) == want
            if "z" in cols:
                zz = np.asarray(cols["z"], dtype=float)
                if kind == "hex":
                    half_z = float((list(win) + [0, 0, 0])[2] or win[0]) / 2.0
                    keep &= np.abs(zz - z) <= half_z
                else:
                    keep &= np.rint(zz) == z
            idx = np.flatnonzero(keep)[::every]
            if idx.size == 0:
                continue
            wy, wx = float(win[0]) / 2.0, float(win[1] if len(win) > 1 else win[0]) / 2.0
            yy = np.asarray(cols["y"], dtype=float)[idx]
            xx = np.asarray(cols["x"], dtype=float)[idx]
            out.append(np.stack([yy - wy, yy + wy, xx - wx, xx + wx, np.zeros(idx.size)],
                                axis=1))
            continue
        cols = by_layer.get(str(mesh_layer), {})
        if "y_lo" not in cols:
            continue
        keep = np.ones(len(cols["y_lo"]), dtype=bool)
        if "m" in cols:
            keep &= np.asarray(cols["m"]) == m
        if "t" in cols:
            tt = np.asarray(cols["t"])
            keep &= (tt == t) | (tt < 0)
        if kind == "hex":
            keep &= ((np.asarray(cols["z_lo"], float) <= z + 0.5)
                     & (np.asarray(cols["z_hi"], float) >= z - 0.5))
        elif "z" in cols:
            keep &= np.rint(np.asarray(cols["z"], float)) == z
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            continue
        lv = (np.asarray(cols["level"], float)[idx] if "level" in cols
              else np.zeros(idx.size))
        out.append(np.stack([np.asarray(cols[k], dtype=float)[idx]
                             for k in ("y_lo", "y_hi", "x_lo", "x_hi")] + [lv], axis=1))
    return np.concatenate(out) if out else None


def diff_sets(ds: Any, coords: Tuple[int, int, int, int]):
    """The two Point sets the comparison overlay judges → ``(a, b)`` of ``(N,2)`` yx, from
    the two Point layers on the frame in first-appearance order; ``(None, None)`` unless
    there are exactly two."""
    attrs = _attrs(ds)
    if not attrs:
        return (None, None)
    m, t, z, c = coords
    from nodegraph.tables import pixel_layers
    pixels = pixel_layers(ds.metadata)
    by_layer: Dict[str, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in attrs.items():
        if dom is Domain.POINT and layer and str(layer) not in pixels:
            by_layer.setdefault(str(layer), {})[str(name)] = attr.values
    usable = [(lyr, cols) for lyr, cols in by_layer.items() if "y" in cols and "x" in cols]
    if len(usable) != 2:
        return (None, None)
    out = []
    for lyr, cols in usable:
        keep = np.ones(len(cols["y"]), dtype=bool)
        for key, want in (("m", m), ("t", t), ("c", c)):
            if key in cols:
                keep &= np.asarray(cols[key]) == want
        try:
            z_kind = ds.structure_zkind(Domain.POINT, lyr)
        except Exception:      # noqa: BLE001 — provenance is advisory
            z_kind = None
        if z_kind == "plane_index" and "z" in cols:
            keep &= np.asarray(cols["z"]) == z
        idx = np.flatnonzero(keep)
        out.append(np.stack([np.asarray(cols["y"], dtype=float)[idx],
                             np.asarray(cols["x"], dtype=float)[idx]], axis=1))
    return (out[0], out[1])


# ── everything a frame needs, in one call ─────────────────────────────────────────────
@dataclass
class FrameGeometry:
    """What :class:`~nodelab_v2.overlays.OverlayFrame` is built from, for one frame."""
    points: List[OV.PointMark] = dc_field(default_factory=list)
    tracks: List[OV.TrackPath] = dc_field(default_factory=list)
    mesh: List[OV.MeshSection] = dc_field(default_factory=list)
    label_plane: Optional[np.ndarray] = None
    label_keys: Optional[np.ndarray] = None
    scalar: Optional[np.ndarray] = None
    vectors: Any = None                       # the VectorField, or None
    field: Any = None                         # (raster, lims, cmap, label), or None
    glyphs: Any = None
    grid_lines: Any = None
    grid: Optional[np.ndarray] = None
    diff: Tuple[Any, Any] = (None, None)

    def frame(self, map_pt, plane_wh: Tuple[int, int], *, sy: float = 1.0, sx: float = 1.0,
              current_t: int = 0, stroke_scale: float = 0.0) -> OV.OverlayFrame:
        """The renderer's frame for this geometry under a plane→surface mapping."""
        vf = self.vectors
        return OV.OverlayFrame(
            map_pt=map_pt, plane_wh=plane_wh, sy=sy, sx=sx,
            points=self.points, tracks=self.tracks, mesh=self.mesh,
            label_plane=self.label_plane, label_keys=self.label_keys,
            scalar=self.scalar, vectors=(None if vf is None else vf.rows),
            vec_hi=(vf.mag_hi if vf is not None else 0.0),
            vec_auto=(FV.auto_scale(vf) if vf is not None else 0.0),
            field=(self.field[0] if self.field is not None else None),
            field_lims=(self.field[1] if self.field is not None else None),
            field_cmap=(self.field[2] if self.field is not None else ""),
            field_label=(self.field[3] if self.field is not None else ""),
            glyphs=self.glyphs, grid_lines=self.grid_lines, grid=self.grid,
            diff_a=self.diff[0], diff_b=self.diff[1], current_t=current_t,
            stroke_scale=stroke_scale)


def frame_geometry(ds: Any, coords: Tuple[int, int, int, int],
                   ref_shape: Optional[Tuple[int, int]], s: OV.OverlaySettings,
                   palette: Palette, *, own_layers: Optional[Sequence[str]] = None,
                   field_cache: Optional[Dict[Any, Any]] = None) -> FrameGeometry:
    """Every enabled overlay's geometry for one frame — what the Viewer's
    ``_ensure_geometry`` computes and what the figure's image item asks for."""
    g = FrameGeometry()
    g.points = point_marks(ds, coords, s.points, palette) if s.points.enabled else []
    g.tracks = track_paths(ds, coords, palette) if s.tracks.enabled else []
    if s.labels.enabled:
        src = label_source(ds, coords, s.labels.layer, own_layers)
        g.label_plane = label_plane(src, coords, ref_shape)
        g.label_keys = label_keys(g.label_plane, palette)
    g.mesh = mesh_sections(ds, coords, s.mesh) if s.mesh.enabled else []
    g.scalar = scalar_plane(ds, coords, ref_shape) if s.voxels.enabled else None
    g.vectors = vector_info(ds, coords, s.vectors.layer) if s.vectors.enabled else None
    g.field = field_info(ds, coords, s.field, field_cache) if s.field.enabled else None
    g.glyphs = (glyph_info(ds, coords, s.field)
                if s.field.enabled and s.field.glyphs in ("strain", "stress") else None)
    deformed = s.grid.enabled and s.grid.style == "deformed"
    g.grid_lines = (deformed_lines(ds, coords, s.vectors.layer, s.grid.exaggerate)
                    if deformed else None)
    g.grid = grid_rects(ds, coords, s.grid) if s.grid.enabled and not deformed else None
    g.diff = diff_sets(ds, coords) if s.diff.enabled else (None, None)
    return g


__all__ = ["Palette", "FrameGeometry", "frame_geometry", "build_palette", "member_layers",
           "track_layers", "best_member_layer", "drop_synth", "point_marks", "track_paths",
           "label_rasters", "label_source", "label_plane", "label_keys", "decimate_to",
           "scalar_plane", "mesh_sections", "vector_info", "field_info", "glyph_info",
           "deformed_lines", "grid_rects", "diff_sets", "MAX_PALETTE_OBJECTS",
           "MAX_NEIGHBOR_FRAMES"]
