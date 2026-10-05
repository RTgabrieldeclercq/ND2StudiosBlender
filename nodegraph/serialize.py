"""Save / load a nodegraph v2 model to a JSON-native dict — the headless core of the
``*.nd2graph.json`` file format (V2.00 §12; the multi-page WORKSPACE shape since V4.00, see
:data:`WORKSPACE_FORMAT_VERSION`). Qt-free; standard library only.

This module round-trips the **structural** model only — exactly what lives in the
headless dataclasses:

* :class:`~nodegraph.graph.Graph` / :class:`~nodegraph.graph.NodeInstance` /
  :class:`~nodegraph.graph.Edge` — node ids, op_keys, params, modes, and every edge
  (INCLUDING ``kind="back"`` zone-feedback edges, which are load-bearing).
* :class:`~nodegraph.zones.Zone` — the Repeat/Sim unroll markers.
* :class:`~nodegraph.groups.Group` — a reusable subgraph whose ``body`` is itself a
  nested :class:`Graph`, serialized recursively via the same node/edge helpers.

Design notes
------------
* **JSON-native values only.** ``params`` / ``modes`` are assumed to already hold plain
  Python scalars / containers (numbers, strings, bools, lists — incl. the optional
  ``__locked__`` sticky-override list). No numpy: the saved model is pure structure, so
  this module never imports numpy, Qt, or ``nodegraph.nodes`` (no compute is needed).
* **Deterministic output** so a saved file diffs cleanly: node/edge/zone/group lists are
  emitted in a stable sorted order, and :func:`to_json` passes ``sort_keys=True`` so
  every nested mapping (params/modes included) is key-sorted too.
* **Validated on load.** An unknown/absent ``format_version`` or malformed structure
  raises a clear :class:`ValueError`.
* **Extension point.** GUI-only per-node state (canvas position, mute/collapse, frame
  membership) is *not* in the headless :class:`NodeInstance` yet, so it is intentionally
  NOT serialized here. When those fields land, extend :func:`_node_to_dict` /
  :func:`_node_from_dict` (and bump :data:`FORMAT_VERSION`); unknown extra keys in a node
  object are tolerated on load so a forward-written file still reads back.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.zones import Zone
from nodegraph.groups import Group

#: The on-disk format version of a SINGLE-GRAPH ``*.nd2graph.json`` document — the shape
#: :func:`to_dict` writes (``{format_version, graph, zones, groups}``) and every pre-V4 file
#: carries. A file whose ``format_version`` is not in :data:`SUPPORTED_VERSIONS` is rejected
#: by :func:`from_dict`.
FORMAT_VERSION: str = "2.0"

#: The format version of a WORKSPACE document (ND2Studios V4.00, 2026-10-05):
#: ``{format_version, app_version, workspace: {active, next_page_seq, pages: [...]}}`` where
#: each page record is a single-graph body (``graph``/``zones``/``groups``/``ui``) plus
#: ``id``/``name``/``kind`` — or, for a page linked to a master, ``master`` + ``overrides``
#: and no graph. Written by :func:`to_workspace_dict`. :func:`from_dict` reads one and
#: returns its ACTIVE page, so a reader that only knows single graphs still gets a graph;
#: the page-aware readers are :func:`workspace_pages` and :func:`from_dict_page`, and the
#: cross-page COMPOSITION (an Input node replaced by the upstream page's Output) lives in
#: :mod:`nodelab_v2.workspace`, not here — this module never resolves anything.
WORKSPACE_FORMAT_VERSION: str = "3.0"

#: Versions :func:`from_dict` is willing to read.
SUPPORTED_VERSIONS = frozenset({FORMAT_VERSION, WORKSPACE_FORMAT_VERSION})


# ── low-level element (de)serialization ──────────────────────────────────────

def _node_to_dict(node: NodeInstance) -> Dict[str, Any]:
    """One :class:`NodeInstance` → a JSON-native dict (headless fields only)."""
    return {
        "id": node.id,
        "op_key": node.op_key,
        "params": dict(node.params),
        "modes": dict(node.modes),
    }


def _node_from_dict(d: Mapping[str, Any]) -> NodeInstance:
    _require(isinstance(d, Mapping), "node must be an object")
    nid = d.get("id")
    op_key = d.get("op_key")
    _require(isinstance(nid, str), "node.id must be a string")
    _require(isinstance(op_key, str), f"node {nid!r}.op_key must be a string")
    params = d.get("params", {})
    modes = d.get("modes", {})
    _require(isinstance(params, Mapping), f"node {nid!r}.params must be an object")
    _require(isinstance(modes, Mapping), f"node {nid!r}.modes must be an object")
    return NodeInstance(nid, op_key, params=dict(params), modes=dict(modes))


def _edge_to_dict(e: Edge) -> Dict[str, Any]:
    return {
        "src": e.src,
        "dst": e.dst,
        "src_socket": e.src_socket,
        "dst_socket": e.dst_socket,
        "kind": e.kind,
    }


def _edge_from_dict(d: Mapping[str, Any]) -> Edge:
    _require(isinstance(d, Mapping), "edge must be an object")
    src, dst = d.get("src"), d.get("dst")
    _require(isinstance(src, str), "edge.src must be a string")
    _require(isinstance(dst, str), "edge.dst must be a string")
    src_socket = d.get("src_socket", "out")
    dst_socket = d.get("dst_socket", "data")
    kind = d.get("kind", "forward")
    _require(isinstance(src_socket, str), f"edge {src}->{dst}.src_socket must be a string")
    _require(isinstance(dst_socket, str), f"edge {src}->{dst}.dst_socket must be a string")
    _require(isinstance(kind, str), f"edge {src}->{dst}.kind must be a string")
    return Edge(src, dst, src_socket, dst_socket, kind)


def _edge_sort_key(d: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return (d["src"], d["dst"], d["src_socket"], d["dst_socket"], d["kind"])


def _graph_to_dict(graph: Graph) -> Dict[str, Any]:
    """A :class:`Graph` → ``{"nodes": [...], "edges": [...]}`` with stable ordering."""
    nodes = sorted((_node_to_dict(n) for n in graph.nodes.values()),
                   key=lambda n: n["id"])
    edges = sorted((_edge_to_dict(e) for e in graph.edges), key=_edge_sort_key)
    return {"nodes": nodes, "edges": edges}


def _graph_from_dict(d: Mapping[str, Any]) -> Graph:
    _require(isinstance(d, Mapping), "graph must be an object")
    raw_nodes = d.get("nodes", [])
    raw_edges = d.get("edges", [])
    _require(isinstance(raw_nodes, list), "graph.nodes must be a list")
    _require(isinstance(raw_edges, list), "graph.edges must be a list")
    nodes: Dict[str, NodeInstance] = {}
    for nd in raw_nodes:
        node = _node_from_dict(nd)
        if node.id in nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        nodes[node.id] = node
    edges: List[Edge] = [_edge_from_dict(ed) for ed in raw_edges]
    return Graph(nodes, edges)


def _zone_to_dict(z: Zone) -> Dict[str, Any]:
    return {
        "id": z.id,
        "kind": z.kind,
        "in_id": z.in_id,
        "out_id": z.out_id,
        "body": sorted(z.body),          # frozenset → stable list
        "iterations": z.iterations,
        "impure": z.impure,
    }


def _zone_from_dict(d: Mapping[str, Any]) -> Zone:
    _require(isinstance(d, Mapping), "zone must be an object")
    zid = d.get("id")
    _require(isinstance(zid, str), "zone.id must be a string")
    for k in ("kind", "in_id", "out_id"):
        _require(isinstance(d.get(k), str), f"zone {zid!r}.{k} must be a string")
    body = d.get("body", [])
    _require(isinstance(body, list) and all(isinstance(b, str) for b in body),
             f"zone {zid!r}.body must be a list of strings")
    iterations = d.get("iterations", 1)
    impure = d.get("impure", False)
    _require(isinstance(iterations, int) and not isinstance(iterations, bool),
             f"zone {zid!r}.iterations must be an int")
    _require(isinstance(impure, bool), f"zone {zid!r}.impure must be a bool")
    return Zone(zid, d["kind"], d["in_id"], d["out_id"],
                body=frozenset(body), iterations=iterations, impure=impure)


def _group_to_dict(g: Group) -> Dict[str, Any]:
    return {
        "name": g.name,
        "input_id": g.input_id,
        "output_id": g.output_id,
        "body": _graph_to_dict(g.body),   # nested Graph → recurse
    }


def _group_from_dict(d: Mapping[str, Any]) -> Group:
    _require(isinstance(d, Mapping), "group must be an object")
    name = d.get("name")
    _require(isinstance(name, str), "group.name must be a string")
    for k in ("input_id", "output_id"):
        _require(isinstance(d.get(k), str), f"group {name!r}.{k} must be a string")
    _require("body" in d, f"group {name!r} is missing its body graph")
    body = _graph_from_dict(d["body"])
    return Group(name, body, d["input_id"], d["output_id"])


# ── public API ────────────────────────────────────────────────────────────────

def to_dict(graph: Graph, *, zones: Iterable[Zone] = (),
            groups: Iterable[Group] = ()) -> Dict[str, Any]:
    """Serialize a graph and its zones/groups to a JSON-native dict.

    Lists are emitted in a stable order (nodes by id, edges by their 5-tuple, zones by
    id, groups by name) so the result diffs cleanly.
    """
    return {
        "format_version": FORMAT_VERSION,
        "graph": _graph_to_dict(graph),
        "zones": sorted((_zone_to_dict(z) for z in zones), key=lambda z: z["id"]),
        "groups": sorted((_group_to_dict(g) for g in groups), key=lambda g: g["name"]),
    }


def from_dict(d: Mapping[str, Any]) -> Tuple[Graph, List[Zone], List[Group]]:
    """Rebuild ``(graph, zones, groups)`` from a dict produced by :func:`to_dict` — or, for a
    WORKSPACE document (:data:`WORKSPACE_FORMAT_VERSION`), from its active page.

    Raises :class:`ValueError` on an unknown/absent ``format_version`` or malformed
    structure.
    """
    _check_version(d)
    if is_workspace_dict(d):
        return page_from_dict(active_page_record(d))
    return page_from_dict(d)


def page_from_dict(d: Mapping[str, Any]) -> Tuple[Graph, List[Zone], List[Group]]:
    """``(graph, zones, groups)`` from a single-graph BODY — a 2.0 document or one workspace
    page record — without the version check (the caller did it, or the body has none)."""
    _require(isinstance(d, Mapping), "a graph body must be an object")
    _require("graph" in d, "document is missing its 'graph'")
    graph = _graph_from_dict(d["graph"])

    raw_zones = d.get("zones", [])
    raw_groups = d.get("groups", [])
    _require(isinstance(raw_zones, list), "'zones' must be a list")
    _require(isinstance(raw_groups, list), "'groups' must be a list")
    zones = [_zone_from_dict(z) for z in raw_zones]
    groups = [_group_from_dict(g) for g in raw_groups]
    return graph, zones, groups


def to_page_dict(graph: Graph, *, zones: Iterable[Zone] = (),
                 groups: Iterable[Group] = ()) -> Dict[str, Any]:
    """:func:`to_dict` without the ``format_version`` — the body one workspace page carries."""
    d = to_dict(graph, zones=zones, groups=groups)
    d.pop("format_version", None)
    return d


def is_workspace_dict(d: Any) -> bool:
    """Is this a WORKSPACE document (several pages) rather than a single graph?"""
    return isinstance(d, Mapping) and d.get("format_version") == WORKSPACE_FORMAT_VERSION


def to_workspace_dict(pages: Iterable[Mapping[str, Any]], *, active: str,
                      next_page_seq: int, app_version: str = "") -> Dict[str, Any]:
    """Serialize page records into a :data:`WORKSPACE_FORMAT_VERSION` document.

    Each record carries ``id``/``name``/``kind`` and EITHER a graph body (``graph`` and
    optionally ``zones``/``groups``/``ui``) OR a ``master`` page id plus ``overrides`` (a
    linked page: same nodes as its master, its own parameter values). The records are
    emitted in the order given — that order is the page order the editor shows.
    """
    recs: List[Dict[str, Any]] = []
    for p in pages:
        _require(isinstance(p, Mapping), "a workspace page must be an object")
        pid = p.get("id")
        _require(isinstance(pid, str) and bool(pid), "a workspace page needs a string 'id'")
        _require(isinstance(p.get("name", ""), str), f"page {pid!r} needs a string 'name'")
        _require(isinstance(p.get("kind", ""), str), f"page {pid!r} needs a string 'kind'")
        has_master = bool(p.get("master"))
        _require(("graph" in p) != has_master,
                 f"page {pid!r} must carry either a graph or a master, not both or neither")
        recs.append(dict(p))
    ids = [r["id"] for r in recs]
    _require(len(set(ids)) == len(ids), "duplicate page id in the workspace")
    _require(not recs or active in ids,
             f"active page {active!r} is not one of the pages {ids}")
    return {
        "format_version": WORKSPACE_FORMAT_VERSION,
        "app_version": str(app_version),
        "workspace": {"active": active, "next_page_seq": int(next_page_seq),
                      "pages": recs},
    }


def workspace_pages(d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The raw page records of a document, in file order, validated for id/name/kind/master.

    A SINGLE-GRAPH (2.0) document is reported as one record ``{"id": "pg1", "name": "",
    "kind": "free", "master": None, graph…}``, so a caller can treat both shapes alike —
    that is how a pre-V4 file opens as one Free page."""
    version = _check_version(d)
    if version != WORKSPACE_FORMAT_VERSION:
        rec: Dict[str, Any] = {"id": "pg1", "name": "", "kind": "free", "master": None}
        rec.update({k: d[k] for k in ("graph", "zones", "groups", "ui") if k in d})
        return [rec]
    ws = d.get("workspace")
    _require(isinstance(ws, Mapping), "workspace document is missing its 'workspace'")
    pages = ws.get("pages", [])
    _require(isinstance(pages, list) and len(pages) > 0,
             "'workspace.pages' must be a non-empty list")
    all_ids = {p.get("id") for p in pages if isinstance(p, Mapping)}
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for p in pages:
        _require(isinstance(p, Mapping), "a workspace page must be an object")
        pid = p.get("id")
        _require(isinstance(pid, str) and bool(pid), "page.id must be a non-empty string")
        _require(pid not in seen, f"duplicate page id {pid!r}")
        seen.add(pid)
        _require(isinstance(p.get("name", ""), str), f"page {pid!r}.name must be a string")
        _require(isinstance(p.get("kind", ""), str), f"page {pid!r}.kind must be a string")
        master = p.get("master")
        _require(master is None or isinstance(master, str),
                 f"page {pid!r}.master must be a string or null")
        if master:
            _require(master in all_ids, f"page {pid!r} links to unknown master {master!r}")
            _require(master != pid, f"page {pid!r} links to itself")
        else:
            _require("graph" in p, f"page {pid!r} is missing its 'graph'")
        out.append(dict(p))
    return out


def active_page_record(d: Mapping[str, Any]) -> Dict[str, Any]:
    """The page record :func:`from_dict` reads from a workspace document: the one named by
    ``workspace.active``; a linked active page stands in for its MASTER (the graph it
    shares), since the overrides need the Workspace to apply. A 2.0 document is its own
    single record."""
    pages = workspace_pages(d)
    by_id = {p["id"]: p for p in pages}
    rec = pages[0]
    if is_workspace_dict(d):
        active = d["workspace"].get("active")
        rec = by_id.get(active, pages[0]) if isinstance(active, str) else pages[0]
    hops = 0
    while rec.get("master") and hops < len(pages):
        rec = by_id[rec["master"]]
        hops += 1
    _require(not rec.get("master"), "workspace has no plain page to read")
    return rec


def from_dict_page(d: Mapping[str, Any], page: Optional[str] = None
                   ) -> Tuple[Graph, List[Zone], List[Group]]:
    """``(graph, zones, groups)`` of ONE page, chosen by id or by name; ``None``/``""`` =
    the active page. A linked page carries no graph of its own here, so selecting one is
    refused — :class:`nodelab_v2.workspace.Workspace` is what materializes it."""
    if not page:
        rec = active_page_record(d)
    else:
        recs = workspace_pages(d)
        match = ([p for p in recs if p["id"] == page]
                 or [p for p in recs if p.get("name") == page])
        if not match:
            names = ", ".join(p["id"] + (f" ({p['name']!r})" if p.get("name") else "")
                              for p in recs)
            raise ValueError(f"no page {page!r} in this document (pages: {names})")
        rec = match[0]
        if rec.get("master"):
            raise ValueError(
                f"page {rec['id']!r} is linked to {rec['master']!r} and carries no graph of "
                f"its own; open it through nodelab_v2.workspace.Workspace")
    return page_from_dict(rec)


def _check_version(d: Mapping[str, Any]) -> str:
    _require(isinstance(d, Mapping), "top-level document must be an object")
    version = d.get("format_version")
    if version is None:
        raise ValueError("missing 'format_version' (not a nd2graph document)")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported format_version {version!r} "
            f"(this build reads {sorted(SUPPORTED_VERSIONS)})")
    return version


def to_json(graph: Graph, *, zones: Iterable[Zone] = (),
            groups: Iterable[Group] = (), indent: int = 2) -> str:
    """:func:`to_dict` → a deterministic JSON string (``sort_keys=True`` so nested
    param/mode maps are key-sorted too)."""
    return json.dumps(to_dict(graph, zones=zones, groups=groups),
                      indent=indent, sort_keys=True)


def from_json(s: str) -> Tuple[Graph, List[Zone], List[Group]]:
    """Parse a JSON string and rebuild ``(graph, zones, groups)`` via :func:`from_dict`."""
    try:
        d = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    return from_dict(d)


# ── helpers ───────────────────────────────────────────────────────────────────

def _require(cond: bool, msg: str) -> None:
    """Raise a clear :class:`ValueError` when a load-time structural invariant fails."""
    if not cond:
        raise ValueError(f"malformed nd2graph document: {msg}")


__all__ = [
    "FORMAT_VERSION", "WORKSPACE_FORMAT_VERSION", "SUPPORTED_VERSIONS",
    "to_dict", "from_dict", "to_json", "from_json",
    "page_from_dict", "to_page_dict", "is_workspace_dict", "to_workspace_dict",
    "workspace_pages", "active_page_record", "from_dict_page",
]
