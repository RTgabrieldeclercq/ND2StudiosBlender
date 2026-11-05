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
#: the prefix of a data entry's stage nodes (``__fig_data__<id>__concat`` …)
DATA_PREFIX = "__fig_data__"


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

    @property
    def data(self) -> List[Dict[str, Any]]:
        return list(self._doc.get("data") or [])

    @staticmethod
    def data_id(data_id: str, stage: str = "") -> str:
        """The shadow node of a data entry's ``stage`` (``concat`` | ``where`` | ``agg``)."""
        return f"{DATA_PREFIX}{data_id}" + (f"__{stage}" if stage else "")

    @staticmethod
    def data_stages(entry: Mapping[str, Any]) -> List[str]:
        """The stages a data entry compiles to, in wiring order."""
        out = []
        if len(entry.get("from") or []) > 1 and entry.get("pool", True):
            out.append("concat")
        if entry.get("where"):
            out.append("where")
        if entry.get("aggregate"):
            out.append("agg")
        return out

    def data_terminal(self, data_id: str) -> Optional[str]:
        """The shadow node whose output IS the data entry: its last stage, or the source's
        own input when it has no stage (a plain read of one source)."""
        entry = next((d for d in self.data if d["id"] == data_id), None)
        if entry is None:
            return None
        stages = self.data_stages(entry)
        if stages:
            return self.data_id(data_id, stages[-1])
        frm = entry.get("from") or []
        return self.shadow_id(str(frm[0])) if frm else None

    def data_table_name(self, data_id: str) -> str:
        """The table name a chart reads from the data entry."""
        entry = next((d for d in self.data if d["id"] == data_id), None)
        if entry is None:
            return ""
        stages = self.data_stages(entry)
        if not stages:
            return str(entry.get("table") or "")
        last = stages[-1]
        if last == "agg":
            return str((entry.get("aggregate") or {}).get("name") or "summary")
        if last == "where":
            return f"{data_id}_pass"
        return str(entry.get("name") or "combined")

    def sync_shadow(self, graph: Any) -> Tuple[List[str], List[str]]:
        """Bring ``graph`` (the page's :class:`GraphDocument`) in step with the figure: one
        ``page.input`` per source, bound to it (and to its item, for a several-item
        Output), and the stages of every data entry (step 4: ``table.concat`` to pool
        several sources, ``analysis.if_else`` for a filter, ``table.aggregate`` for a
        summary), wired in order. Returns ``(added or re-bound ids, removed ids)``."""
        want: Dict[str, Dict[str, Any]] = {self.shadow_id(s["id"]): s for s in self.sources}
        # the data stages and their wires
        stage_nodes: Dict[str, Tuple[str, Dict[str, Any], Dict[str, str], List[Tuple[str, str]]]] = {}
        for entry in self.data:
            did = str(entry["id"])
            frm = [self.shadow_id(str(s)) for s in (entry.get("from") or [])]
            domain = str(entry.get("domain") or "label")
            table = str(entry.get("table") or "")
            prev: List[str] = frm
            prev_table = table
            for stage in self.data_stages(entry):
                nid = self.data_id(did, stage)
                if stage == "concat":
                    op = "table.concat"
                    params = {"table": table, "name": str(entry.get("name") or "combined")}
                    modes = {"domain": domain}
                    wires = [(p, "out") for p in prev]
                    prev_table = params["name"]
                elif stage == "where":
                    op = "analysis.if_else"
                    conds = list(entry.get("where") or [])[:4]
                    params = {"pass_name": f"{did}_pass", "fail_name": f"{did}_fail",
                              "labels" if domain == "label" else "points": prev_table}
                    for k, (col, oper, val) in enumerate(conds, start=1):
                        params[f"column{k}"] = str(col)
                        params[f"op{k}"] = str(oper)
                        try:
                            params[f"value{k}"] = float(val)
                        except (TypeError, ValueError):
                            params[f"value{k}"] = 0.0
                    modes = {"target": "point" if domain == "point" else "label",
                             "terms": str(max(1, len(conds))),
                             "match": str(entry.get("match") or "all")}
                    wires = [(prev[0], "out")] if prev else []
                    prev_table = f"{did}_pass"
                else:
                    op = "table.aggregate"
                    agg = dict(entry.get("aggregate") or {})
                    gb = agg.get("group_by") or ["t"]
                    reducers = agg.get("reducers") or ["mean", "sem", "count"]
                    params = {"table": prev_table,
                              "group_by": ",".join(str(g) for g in gb),
                              "value": str(agg.get("value") or "area"),
                              "reducers": ",".join(str(r) for r in reducers),
                              "name": str(agg.get("name") or "summary")}
                    modes = {"domain": domain}
                    wires = [(prev[0], "out")] if prev else []
                    prev_table = params["name"]
                stage_nodes[nid] = (op, params, modes, wires)
                prev = [nid]
        removed: List[str] = []
        for nid in list(graph.nodes):
            if (self.is_shadow(nid) and nid not in want) or \
                    (nid.startswith(DATA_PREFIX) and nid not in stage_nodes):
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
        for k, (nid, (op, params, modes, wires)) in enumerate(stage_nodes.items()):
            rec = graph.nodes.get(nid)
            if rec is not None and rec.op_key != op:
                graph.remove_node(nid)
                rec = None
            if rec is None:
                graph.add_node(op, node_id=nid, params=dict(params), modes=dict(modes),
                               x=340.0 + 300.0 * k, y=120.0)
                changed.append(nid)
            else:
                dirty = False
                for key, val in params.items():
                    if rec.params.get(key) != val:
                        rec.params[key] = val
                        dirty = True
                for key, val in modes.items():
                    if rec.modes.get(key) != val:
                        rec.modes[key] = val
                        dirty = True
                if dirty:
                    graph.touch(nid)
                    changed.append(nid)
            have = {(e[0], e[1]) for e in graph.edges if e[2] == nid and e[3] == "data"}
            for src, sock in wires:
                if (src, sock) not in have and src in graph.nodes:
                    graph.connect(src, sock, nid, "data")
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
        """``{source id | data id: run id}`` in a :class:`ComposedGraph` of ``page_id`` —
        the upstream Output's run id for a resolved source (the shadow input's own for an
        unbound one), a data entry's terminal stage for a data entry."""
        out: Dict[str, str] = {}
        for s in self.sources:
            nid = self.shadow_id(s["id"])
            out[str(s["id"])] = composed.id_map.get((page_id, nid), f"{page_id}/{nid}")
        for d in self.data:
            nid = self.data_terminal(str(d["id"]))
            if nid:
                out[str(d["id"])] = composed.id_map.get((page_id, nid), f"{page_id}/{nid}")
        return out

    def data_envelopes(self, page: Any) -> Dict[str, Any]:
        """``{data id: MetaEnvelope}`` of each data entry's terminal node on the figure
        page's own graph (its columns are what a chart's pickers offer)."""
        out: Dict[str, Any] = {}
        for d in self.data:
            nid = self.data_terminal(str(d["id"]))
            env = page.doc.envs.get(nid) if nid else None
            if env is not None:
                out[str(d["id"])] = env
        return out

    def source_labels(self, ws: Any, page_id: str) -> Dict[str, str]:
        """``{source id: the label the Source menus show}`` (``"Image Processing · cells"``),
        the source value itself when it resolves to nothing."""
        labels = dict(ws.available_sources(page_id))
        return {str(s["id"]): labels.get(s["source"], f"{s['source']} (unbound)")
                for s in self.sources}


__all__ = ["FigureDocument", "SHADOW_PREFIX", "DATA_PREFIX"]
