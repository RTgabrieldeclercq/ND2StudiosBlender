"""Link behaviours on a figure document (V5.00 step 3) — Qt-free helpers the panel calls.

* **zoom** — where each popout's box sits on its parent, from
  :func:`nodegraph.figure.bind.zoom_box` (stage placement, a µm rect, an object, or a
  fraction of the same plane);
* **twin** — resolved by :func:`nodegraph.figure.schema.effective_image`; nothing to do
  here beyond naming the halves;
* **lut_share** — which item leads a share group, so the others take its windows;
* **cursor** — which items follow the page cursor on which axes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.figure import bind as B
from nodegraph.figure import schema as S


def zoom_links(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [ln for ln in doc.get("links") or [] if ln.get("kind") == "zoom"]


def zoom_boxes(doc: Mapping[str, Any], envelopes: Mapping[str, Any],
               cursor: Optional[Mapping[str, Any]] = None,
               tables: Optional[Mapping[Any, Mapping[str, Any]]] = None
               ) -> Dict[str, List[Tuple[str, Tuple[float, float, float, float], Dict[str, Any]]]]:
    """``{parent id: [(child id, window on the parent's plane, box style + "_tile"), …]}``
    for every zoom link that resolves; an unresolved one is left out (its refusal is in
    :func:`zoom_refusals`)."""
    out: Dict[str, List[Tuple[str, Tuple[float, float, float, float], Dict[str, Any]]]] = {}
    for ln in zoom_links(doc):
        zb = B.zoom_box(doc, ln, envelopes, cursor=cursor, tables=tables)
        if zb.get("refusal") or zb.get("window") is None:
            continue
        box = dict(ln.get("box") or {})
        box["_tile"] = int(zb.get("parent_m") or 0)
        out.setdefault(str(ln["parent"]), []).append((str(ln["child"]), tuple(zb["window"]), box))
    return out


def zoom_refusals(doc: Mapping[str, Any], envelopes: Mapping[str, Any],
                  cursor: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """``{child id: why its box cannot be drawn}``."""
    out: Dict[str, str] = {}
    for ln in zoom_links(doc):
        zb = B.zoom_box(doc, ln, envelopes, cursor=cursor)
        if zb.get("refusal"):
            out[str(ln["child"])] = str(zb["refusal"])
    return out


def share_groups(doc: Mapping[str, Any]) -> List[List[str]]:
    """The LUT-share groups, each in link order (the first member leads)."""
    return [list(ln.get("items") or []) for ln in doc.get("links") or []
            if ln.get("kind") == "lut_share"]


def share_leader(doc: Mapping[str, Any], item_id: str) -> Optional[str]:
    """The item whose windows ``item_id`` takes, or ``None`` when it shares with nobody
    (or leads its group)."""
    for grp in share_groups(doc):
        if item_id in grp and grp[0] != item_id:
            return grp[0]
    return None


def twin_of(doc: Mapping[str, Any], item_id: str) -> Optional[str]:
    tp = S.twin_parent(doc, item_id)
    return tp[0] if tp else None


def follows_cursor(image: Mapping[str, Any]) -> List[str]:
    """The axes an image's frame reads from the page cursor."""
    fr = image.get("frame") or {}
    return [ax for ax in ("m", "t", "z") if fr.get(ax) == "cursor"]


__all__ = ["zoom_links", "zoom_boxes", "zoom_refusals", "share_groups", "share_leader",
           "twin_of", "follows_cursor"]
