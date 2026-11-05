"""The identity palette and the label / scalar planes of a payload — Qt-free (2026-11-05).

What a frame's overlays colour BY: one palette slot per physical object (a track, or an
untracked region), the label raster a frame draws, the scalar layer a wash shows. It lived in
:mod:`nodelab_v2.overlay_geometry` until the standalone HTML viewer (:mod:`nodegraph.htmlview`)
needed the same colours and the same planes in the engine, which may not import Qt — the
same reason :mod:`nodegraph.catalog._shared.label_paint` exists. ``overlay_geometry``
re-exports every name under its old spelling, so the Viewer and the figure are unchanged.

Only numpy, the Dataset's tables and :mod:`label_paint`; nothing here paints.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.catalog._shared.label_paint import deconflict_slots, neighbor_pairs, slot_lut
from nodegraph.domains import Domain

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
    3. **Slots.** :func:`~nodegraph.catalog._shared.label_paint.deconflict_slots` hands out
       the slots, moving an object off its preference only when a neighbour is too close on
       the colour wheel.

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
            pairs = neighbor_pairs(np.stack([ys[sel], xs[sel]], axis=1))
            if pairs.size:
                chunks.append(obj_row[key][sel][pairs])
    pairs_all = (np.unique(np.concatenate(chunks), axis=0) if chunks
                 else np.zeros((0, 2), dtype=np.int64))
    pairs_all = pairs_all[pairs_all[:, 0] != pairs_all[:, 1]]

    slots = (dict(prefer) if too_many else
             deconflict_slots(prefer, [(int(a), int(b)) for a, b in pairs_all.tolist()]))
    lookup = np.arange(max(1, next_obj), dtype=np.int64)
    for o, sl in slots.items():
        lookup[o] = sl
    pal = Palette()
    for key, ids in ids_of.items():
        lut = slot_lut(ids, lookup[obj_row[key]])
        (pal.labels if key[0] is Domain.LABEL else pal.points)[key] = lut
    for tlayer, (tids, objs) in track_slots.items():
        pal.tracks[tlayer] = slot_lut(tids, lookup[objs])
    return pal


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


def label_source_name(ds: Any, source: Optional[np.ndarray]) -> str:
    """The layer name of ``source`` (an array :func:`label_source` returned), or ``""``."""
    if source is None:
        return ""
    for name, vals in label_rasters(ds).items():
        if vals is source:
            return name
    return ""


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


__all__ = ["MAX_PALETTE_OBJECTS", "MAX_NEIGHBOR_FRAMES", "Palette", "drop_synth",
           "member_layers", "track_layers", "best_member_layer", "build_palette",
           "label_rasters", "label_source", "label_source_name", "decimate_to",
           "label_plane", "label_keys", "scalar_plane"]
