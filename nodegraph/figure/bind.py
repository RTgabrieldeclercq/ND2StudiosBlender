"""What each item of a figure document must READ (V5.00 step 1).

An image item names a source, a frame and a region; a zoom link says one item shows a part
of another. This module turns those into **read plans** — which multipoint, frame and
plane to pull, which fractional window of the plane, which channels — and resolves a
popout's box on its parent from the two fields' stage metadata, through the same maths
``view.overlay`` and the Viewer's compositor use (:mod:`nodegraph.placement`:
:func:`~nodegraph.placement.field_box`, :func:`~nodegraph.placement.source_window`,
:func:`~nodegraph.placement.sub_field_box`). Nothing here reads pixels; the GUI host and
the offscreen renderer execute the plans.

An **envelope** here is anything that carries ``axes`` and ``metadata``: a
:class:`~nodegraph.metadata.MetaEnvelope`, a Dataset, or an ``(axes, metadata)`` pair.

**Handedness.** ``source_window`` maps a µm box onto a tile's pixel grid under the tile's
flips; the Viewer derives those from the Overlay node's ``flip_x`` / ``flip_y`` and the
canvas orientation (:func:`~nodegraph.placement.frame_mapping`). The plans take the flips
as arguments and default to :func:`~nodegraph.placement.canvas_flip` of the metadata — an
unflipped camera field — so the GUI passes the Viewer's handedness when it has one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph import placement as P
from nodegraph.figure import schema as S

Box = Tuple[float, float, float, float]          # y0, y1, x0, x1 in stage µm
Window = Tuple[float, float, float, float]       # fy0, fy1, fx0, fx1 fractions of a plane


@dataclass(frozen=True)
class RegionPlan:
    """Where an item's picture comes from within its source's field."""
    kind: str                       # full | rect_um | placement | object
    m: int
    box_um: Optional[Box]           # the region's field in stage µm (None: unplaceable)
    window: Optional[Window]        # the fraction of the source plane to read (None: all)
    refusal: str = ""

    @property
    def ok(self) -> bool:
        return not self.refusal


@dataclass(frozen=True)
class ReadPlan:
    """One read of one source for one item."""
    item: str
    source: str
    m: int
    t: int
    z: Any                          # an index, or "max" / "mean"
    channels: Tuple[int, ...]
    window: Optional[Window]
    box_um: Optional[Box]
    resolution: str                 # display | native
    note: str = ""


def envelope_parts(env: Any) -> Tuple[Any, Mapping[str, Any]]:
    """``(axes, metadata)`` of an envelope in any of the accepted forms."""
    if isinstance(env, (tuple, list)) and len(env) == 2:
        return env[0], (env[1] or {})
    if isinstance(env, Mapping) and "axes" in env and "metadata" in env:
        return env["axes"], (env["metadata"] or {})
    return getattr(env, "axes", None), (getattr(env, "metadata", None) or {})


def _size(axes: Any, name: str) -> int:
    try:
        return max(1, int(getattr(axes, name, 1) or 1))
    except (TypeError, ValueError):
        return 1


def _clamp(v: Any, n: int) -> int:
    try:
        i = int(v)
    except (TypeError, ValueError):
        i = 0
    return min(max(i, 0), max(0, n - 1))


def resolve_frame(frame: Mapping[str, Any], axes: Any,
                  cursor: Optional[Mapping[str, Any]] = None) -> Tuple[int, int, Any]:
    """``(m, t, z)`` of a frame spec: ``cursor`` entries read the page cursor, indices are
    clamped to the axes, and ``z`` keeps ``max`` / ``mean`` as a projection request."""
    cur = cursor or {}
    m = frame.get("m", 0)
    t = frame.get("t", 0)
    z = frame.get("z", "max")
    m = _clamp(cur.get("m", 0) if m == "cursor" else m, _size(axes, "m"))
    t = _clamp(cur.get("t", 0) if t == "cursor" else t, _size(axes, "t"))
    if z == "cursor":
        z = _clamp(cur.get("z", 0), _size(axes, "z"))
    elif z not in ("max", "mean"):
        z = _clamp(z, _size(axes, "z"))
    return m, t, z


def flips_of(env: Any, *, flip_x: Optional[bool] = None, flip_y: Optional[bool] = None
             ) -> Tuple[bool, bool]:
    """``(flip_x, flip_y)`` for ``source_window``: the given values, else the canvas
    orientation the metadata records (an unflipped field reads ``(False, False)``)."""
    _axes, md = envelope_parts(env)
    cx, cy = P.canvas_flip(md)
    return (bool(cx) if flip_x is None else bool(flip_x),
            bool(cy) if flip_y is None else bool(flip_y))


def source_box(env: Any, m: int) -> Optional[P.FieldBox]:
    """Multipoint ``m``'s field of ``env`` in stage µm, or ``None`` when unplaceable."""
    axes, md = envelope_parts(env)
    return P.field_box(md, axes, int(m))


def _as_tuple(box: Optional[P.FieldBox]) -> Optional[Box]:
    return None if box is None else (float(box.y0), float(box.y1), float(box.x0),
                                     float(box.x1))


def _overlap_of(a: Box, b: Box) -> float:
    """Fraction of ``a``'s area covered by ``b``."""
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    area = (ay1 - ay0) * (ax1 - ax0)
    if area <= 0:
        return 0.0
    dy = min(ay1, by1) - max(ay0, by0)
    dx = min(ax1, bx1) - max(ax0, bx0)
    if dy <= 0 or dx <= 0:
        return 0.0
    return float(dy * dx / area)


def _object_window(region: Mapping[str, Any], tables: Optional[Mapping[Any, Mapping[str, Any]]],
                   axes: Any, md: Mapping[str, Any], m: int, t: int
                   ) -> Tuple[Optional[Window], str]:
    """The window round one object (``region.object``'s id) in the tabulated tables:
    its ``(y, x)`` row for this ``(m, t)`` ± a radius from its area, plus ``margin_um``."""
    if not tables:
        return None, "no tables to find the object in"
    oid = int(region.get("object"))
    want_domain = region.get("domain")
    want_layer = region.get("layer")
    px = float(md.get("pixel_size_um") or 0.0)
    margin_px = (float(region.get("margin_um", 5.0)) / px) if px > 0 else 0.0
    ny, nx = _size(axes, "y"), _size(axes, "x")
    for (domain, layer), cols in tables.items():
        if want_domain and domain != want_domain:
            continue
        if want_layer and layer != want_layer:
            continue
        if "id" not in cols or "y" not in cols or "x" not in cols:
            continue
        ids = np.asarray(cols["id"])
        sel = ids == oid
        if "m" in cols:
            sel &= np.asarray(cols["m"]) == m
        if "t" in cols:
            sel &= np.asarray(cols["t"]) == t
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        i = int(idx[0])
        cy, cx = float(np.asarray(cols["y"])[i]), float(np.asarray(cols["x"])[i])
        half = margin_px
        if "area" in cols:
            try:
                half += math.sqrt(max(0.0, float(np.asarray(cols["area"])[i])) / math.pi)
            except (TypeError, ValueError):
                pass
        if half <= 0:
            half = 8.0
        fy0, fy1 = (cy - half) / ny, (cy + half) / ny
        fx0, fx1 = (cx - half) / nx, (cx + half) / nx
        win = (max(0.0, fy0), min(1.0, fy1), max(0.0, fx0), min(1.0, fx1))
        if not (win[1] > win[0] and win[3] > win[2]):
            return None, f"object {oid} lies outside the field"
        return win, ""
    return None, f"object {oid} is not in the tables at m={m}, t={t}"


def region_plan(doc: Mapping[str, Any], item_id: str, envelopes: Mapping[str, Any], *,
                cursor: Optional[Mapping[str, Any]] = None,
                tables: Optional[Mapping[Any, Mapping[str, Any]]] = None,
                flip_x: Optional[bool] = None, flip_y: Optional[bool] = None
                ) -> RegionPlan:
    """The region of ``item_id``'s source its picture shows — resolved from the item's
    (twin-inherited) image block and the source's envelope. ``placement`` is the item's
    whole field: the *box on the parent* is :func:`zoom_box`'s job."""
    img = S.effective_image(doc, item_id)
    sid = img.get("source")
    if sid is None or sid not in envelopes:
        return RegionPlan("full", 0, None, None, f"{item_id}: source {sid!r} has no envelope")
    env = envelopes[sid]
    axes, md = envelope_parts(env)
    m, t, _z = resolve_frame(img.get("frame") or {}, axes, cursor)
    region = img.get("region") or {"from": "full"}
    kind = str(region.get("from", "full"))
    box = source_box(env, m)
    fx, fy = flips_of(env, flip_x=flip_x, flip_y=flip_y)
    if kind in ("full", "placement"):
        return RegionPlan(kind, m, _as_tuple(box), None)
    if kind == "rect_um":
        y0, y1, x0, x1 = (float(v) for v in region["rect_um"])
        want = P.FieldBox(y0=y0, y1=y1, x0=x0, x1=x1)
        if box is None:
            return RegionPlan(kind, m, (y0, y1, x0, x1), None,
                              f"{item_id}: the source has no stage placement, so a µm rect "
                              f"cannot be located in it")
        win = P.source_window(want, box, flip_y=fy, flip_x=fx)
        if win is None:
            return RegionPlan(kind, m, (y0, y1, x0, x1), None,
                              f"{item_id}: the rect lies outside the source's field")
        return RegionPlan(kind, m, (y0, y1, x0, x1), tuple(float(v) for v in win))
    if kind == "rect_frac":
        fy0, fy1, fx0, fx1 = (float(v) for v in region["rect_frac"])
        sub = _as_tuple(P.sub_field_box(box, (fy0, fy1, fx0, fx1))) if box is not None else None
        return RegionPlan(kind, m, sub, (fy0, fy1, fx0, fx1))
    if kind == "object":
        win, why = _object_window(region, tables, axes, md, m, t)
        if win is None:
            return RegionPlan(kind, m, None, None, f"{item_id}: {why}")
        sub = _as_tuple(P.sub_field_box(box, win)) if box is not None else None
        return RegionPlan(kind, m, sub, win)
    return RegionPlan(kind, m, None, None, f"{item_id}: unknown region kind {kind!r}")


def zoom_box(doc: Mapping[str, Any], link: Mapping[str, Any], envelopes: Mapping[str, Any],
             *, cursor: Optional[Mapping[str, Any]] = None,
             tables: Optional[Mapping[Any, Mapping[str, Any]]] = None,
             flip_x: Optional[bool] = None, flip_y: Optional[bool] = None) -> Dict[str, Any]:
    """Where the child of a ``zoom`` link sits on its parent: ``{"parent_m", "window"
    (fractions of the parent's plane), "rect_px" (y0, y1, x0, x1 on the parent's plane),
    "box_um", "coverage" (how much of the child's field the parent tile holds),
    "refusal"}``. A montage parent tiled by position is searched across every position;
    any other parent is read at its own frame's position."""
    parent_id, child_id = str(link["parent"]), str(link["child"])
    child = region_plan(doc, child_id, envelopes, cursor=cursor, tables=tables,
                        flip_x=flip_x, flip_y=flip_y)
    out: Dict[str, Any] = {"parent": parent_id, "child": child_id, "parent_m": None,
                           "window": None, "rect_px": None, "box_um": child.box_um,
                           "coverage": 0.0, "refusal": ""}
    if not child.ok:
        out["refusal"] = child.refusal
        return out
    cimg = S.effective_image(doc, child_id)
    pimg_src = S.effective_image(doc, parent_id).get("source")
    if child.window is not None and cimg.get("source") == pimg_src             and child.kind in ("rect_frac", "object", "rect_um"):
        # the same source: the child's window IS its box on the parent's plane
        penv = envelopes.get(pimg_src)
        paxes, _pmd = envelope_parts(penv) if penv is not None else (None, {})
        ny, nx = _size(paxes, "y"), _size(paxes, "x")
        fy0, fy1, fx0, fx1 = child.window
        out.update(parent_m=child.m, window=child.window, coverage=1.0,
                   rect_px=(fy0 * ny, fy1 * ny, fx0 * nx, fx1 * nx))
        return out
    if child.box_um is None:
        out["refusal"] = f"{child_id}: its source has no stage placement"
        return out
    pimg = S.effective_image(doc, parent_id)
    psid = pimg.get("source")
    if psid is None or psid not in envelopes:
        out["refusal"] = f"{parent_id}: source {psid!r} has no envelope"
        return out
    penv = envelopes[psid]
    paxes, _pmd = envelope_parts(penv)
    pm, _t, _z = resolve_frame(pimg.get("frame") or {}, paxes, cursor)
    pitem = S.item(doc, parent_id)
    candidates = (range(_size(paxes, "m"))
                  if pitem.get("kind") == "montage"
                  and str((pitem.get("montage") or {}).get("tile_by")) == "m" else [pm])
    best: Optional[Tuple[float, int, P.FieldBox]] = None
    for m in candidates:
        pbox = source_box(penv, m)
        if pbox is None:
            continue
        cov = _overlap_of(child.box_um, _as_tuple(pbox))   # type: ignore[arg-type]
        if cov > 0 and (best is None or cov > best[0]):
            best = (cov, int(m), pbox)
    if best is None:
        out["refusal"] = (f"{child_id}'s field does not overlap {parent_id}'s — the two "
                          f"outputs were not taken over the same stage area")
        return out
    cov, m, pbox = best
    fx, fy = flips_of(penv, flip_x=flip_x, flip_y=flip_y)
    cy0, cy1, cx0, cx1 = child.box_um
    win = P.source_window(P.FieldBox(y0=cy0, y1=cy1, x0=cx0, x1=cx1), pbox,
                          flip_y=fy, flip_x=fx)
    if win is None:
        out["refusal"] = f"{child_id}'s field misses {parent_id}'s tile {m}"
        return out
    ny, nx = _size(paxes, "y"), _size(paxes, "x")
    fy0, fy1, fx0, fx1 = (float(v) for v in win)
    out.update(parent_m=m, window=(fy0, fy1, fx0, fx1), coverage=float(cov),
               rect_px=(fy0 * ny, fy1 * ny, fx0 * nx, fx1 * nx))
    return out


def read_plan(doc: Mapping[str, Any], item_id: str, envelopes: Mapping[str, Any], *,
              cursor: Optional[Mapping[str, Any]] = None,
              tables: Optional[Mapping[Any, Mapping[str, Any]]] = None,
              resolution: str = "display", flip_x: Optional[bool] = None,
              flip_y: Optional[bool] = None) -> ReadPlan:
    """The read an image or montage item needs for its base frame."""
    img = S.effective_image(doc, item_id)
    sid = img.get("source")
    if sid is None or sid not in envelopes:
        return ReadPlan(item_id, str(sid), 0, 0, "max", (), None, None, resolution,
                        note=f"{item_id}: source {sid!r} has no envelope")
    axes, _md = envelope_parts(envelopes[sid])
    m, t, z = resolve_frame(img.get("frame") or {}, axes, cursor)
    region = region_plan(doc, item_id, envelopes, cursor=cursor, tables=tables,
                         flip_x=flip_x, flip_y=flip_y)
    chans = tuple(sorted({int(c.get("c", 0)) for c in img.get("channels") or []})) or (0,)
    nc = _size(axes, "c")
    chans = tuple(c for c in chans if 0 <= c < nc) or (0,)
    return ReadPlan(item_id, str(sid), m, t, z, chans, region.window, region.box_um,
                    resolution, note=region.refusal)


def read_plans(doc: Mapping[str, Any], envelopes: Mapping[str, Any], *,
               cursor: Optional[Mapping[str, Any]] = None,
               tables: Optional[Mapping[Any, Mapping[str, Any]]] = None,
               resolution: str = "display") -> Dict[str, ReadPlan]:
    """A read plan per image / montage item of ``doc``, keyed by item id."""
    return {it["id"]: read_plan(doc, it["id"], envelopes, cursor=cursor, tables=tables,
                                resolution=resolution)
            for it in doc.get("items") or [] if it.get("kind") in ("image", "montage")}


def window_px(window: Window, ny: int, nx: int) -> Tuple[int, int, int, int]:
    """A fractional window as whole pixel bounds ``(y0, y1, x0, x1)``, snapped outward."""
    return P.pixel_window(window, int(ny), int(nx))
