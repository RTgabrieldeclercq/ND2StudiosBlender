"""The figure half of an Analysis page (V5.00 step 2) — Qt-free.

Since V5.00 an **Analysis** page is a figure page: beside its :class:`GraphDocument` it
carries a :class:`FigureDocument` — the figure document of
:mod:`nodegraph.figure.schema` (sources, items, links), with a revision, listeners and the
one spelling the render cache keys on. The page's graph document is the figure's **shadow
graph**: one hidden ``page.input`` per figure source (ids ``__fig_src__<source id>``),
so :meth:`Workspace.compose`, the memo, the cross-page staleness signature, the Node
queue and the LabLink publisher all work on a figure page unchanged — the editor fetches a
source's Dataset exactly as the Movie Editor computes its A/B/C.

The sync is one-directional and idempotent: :meth:`FigureDocument.sync_shadow` adds the
inputs the figure names, re-binds the ones whose source changed and removes the ones whose
source is gone, and never touches a node it did not make — the table-preparation nodes a
later step compiles, and whatever a user placed by hand on an older Analysis page, stay.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from nodegraph.figure import schema as S
from nodelab_v2.ops import PAGE_INPUT_OP, PAGE_ITEM_KEY, PAGE_SOURCE_KEY

#: the prefix of a shadow input's node id; the rest is the figure source's id
SHADOW_PREFIX = "__fig_src__"


class FigureDocument:
    """A figure document with a revision and listeners, and its shadow-graph sync."""

    def __init__(self, doc: Optional[Mapping[str, Any]] = None) -> None:
        self._doc: Dict[str, Any] = S.normalize(doc) if doc is not None else S.new_document()
        #: bumped on every accepted change
        self.revision = 0
        self._listeners: List[Callable[[], None]] = []

    # ── listeners ──────────────────────────────────────────────────────────────
    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def off_change(self, fn: Callable[[], None]) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def _fire(self) -> None:
        for fn in list(self._listeners):
            fn()

    # ── the document ───────────────────────────────────────────────────────────
    @property
    def document(self) -> Dict[str, Any]:
        """The live normalized document — read it, never mutate it (use :meth:`edit`)."""
        return self._doc

    @property
    def goal(self) -> Optional[str]:
        return self._doc.get("goal")

    @property
    def sources(self) -> List[Dict[str, Any]]:
        return list(self._doc.get("sources") or [])

    @property
    def items(self) -> List[Dict[str, Any]]:
        return list(self._doc.get("items") or [])

    def to_dict(self) -> Dict[str, Any]:
        """A deep copy of the normalized document — what the workspace file carries."""
        return copy.deepcopy(self._doc)

    def canonical(self) -> str:
        return S.canonical_json(self._doc)

    def digest(self) -> str:
        return S.digest(self._doc)

    def replace(self, doc: Mapping[str, Any]) -> bool:
        """Make ``doc`` the document. Validated first (:class:`nodegraph.figure.schema.
        FigureError` names the bad value), accepted only when its canonical spelling
        differs — a no-op edit bumps nothing. Returns whether anything changed."""
        nd = S.normalize(doc)
        if S.canonical_json(nd) == self.canonical():
            return False
        self._doc = nd
        self.revision += 1
        self._fire()
        return True

    def edit(self, mutate: Callable[[Dict[str, Any]], None]) -> bool:
        """``mutate`` a copy of the document, then :meth:`replace` with it."""
        d = self.to_dict()
        mutate(d)
        return self.replace(d)

    def set_goal(self, goal: Optional[str]) -> bool:
        return self.edit(lambda d: d.__setitem__("goal", goal))

    # ── sources ↔ the shadow graph ─────────────────────────────────────────────
    @staticmethod
    def shadow_id(source_id: str) -> str:
        return f"{SHADOW_PREFIX}{source_id}"

    @staticmethod
    def is_shadow(node_id: str) -> bool:
        return str(node_id).startswith(SHADOW_PREFIX)

    def source_ids(self) -> List[str]:
        return [str(s["id"]) for s in self.sources]

    def sync_shadow(self, graph: Any) -> Tuple[List[str], List[str]]:
        """Bring ``graph`` (the page's :class:`GraphDocument`) in step with the figure's
        sources: one ``page.input`` per source, bound to it (and to its item, for a
        several-item Output). Returns ``(added or re-bound ids, removed ids)``."""
        want: Dict[str, Dict[str, Any]] = {self.shadow_id(s["id"]): s for s in self.sources}
        removed: List[str] = []
        for nid in list(graph.nodes):
            if self.is_shadow(nid) and nid not in want:
                graph.remove_node(nid)
                removed.append(nid)
        changed: List[str] = []
        for k, (nid, s) in enumerate(want.items()):
            params = {PAGE_SOURCE_KEY: str(s["source"])}
            if s.get("item"):
                params[PAGE_ITEM_KEY] = str(s["item"])
            rec = graph.nodes.get(nid)
            if rec is None:
                graph.add_node(PAGE_INPUT_OP, node_id=nid, params=params, x=40.0,
                               y=120.0 + 90.0 * k)
                changed.append(nid)
                continue
            if rec.params.get(PAGE_SOURCE_KEY) != params[PAGE_SOURCE_KEY] or \
                    rec.params.get(PAGE_ITEM_KEY) != params.get(PAGE_ITEM_KEY):
                rec.params[PAGE_SOURCE_KEY] = params[PAGE_SOURCE_KEY]
                if PAGE_ITEM_KEY in params:
                    rec.params[PAGE_ITEM_KEY] = params[PAGE_ITEM_KEY]
                else:
                    rec.params.pop(PAGE_ITEM_KEY, None)
                graph.touch(nid)
                changed.append(nid)
        return changed, removed

    def envelopes(self, ws: Any, page_id: str) -> Dict[str, Any]:
        """``{source id: MetaEnvelope}`` — each source's upstream Output envelope as the
        workspace resolves it now (its item's, for an item source); a source that resolves
        to nothing is left out."""
        out: Dict[str, Any] = {}
        for s in self.sources:
            value = s["source"]
            res = (ws.resolve_item(page_id, value, str(s["item"])) if s.get("item")
                   else ws.resolve_source(page_id, value))
            if res is None:
                continue
            up, nid = res
            page = ws.pages.get(up)
            env = page.doc.envs.get(nid) if page is not None else None
            if env is not None:
                out[str(s["id"])] = env
        return out

    def run_ids(self, composed: Any, page_id: str) -> Dict[str, str]:
        """``{source id: run id}`` in a :class:`ComposedGraph` of ``page_id`` — the upstream
        Output's run id for a resolved source, the shadow input's own for an unbound one."""
        out: Dict[str, str] = {}
        for s in self.sources:
            nid = self.shadow_id(s["id"])
            out[str(s["id"])] = composed.id_map.get((page_id, nid), f"{page_id}/{nid}")
        return out

    def source_labels(self, ws: Any, page_id: str) -> Dict[str, str]:
        """``{source id: the label the Source menus show}`` (``"Image Processing · cells"``),
        the source value itself when it resolves to nothing."""
        labels = dict(ws.available_sources(page_id))
        return {str(s["id"]): labels.get(s["source"], f"{s['source']} (unbound)")
                for s in self.sources}


__all__ = ["FigureDocument", "SHADOW_PREFIX"]
