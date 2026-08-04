"""Minimal node-graph model (nodegraph v2, Phase-2a-prime).

Just enough graph to run the edit-time :mod:`nodegraph.metadata` MetaEnvelope pass
(V2.03 §2 A3): node **instances** (op + params + chosen mode state) and directed
**edges** between named sockets, with a topological order and cycle rejection
(V2.00 §7 rejects cycles outside zones — zones arrive with the Phase-4 zone model).

This is intentionally small; the Phase-2 eval engine extends it with multi-input
order, node groups, and Repeat/Simulation zones. Qt-free; pure standard library.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from nodegraph.registry import NODES, NodeSpec


@dataclass(frozen=True)
class NodeInstance:
    """One placed node: its type (``op_key``), its value-socket param overrides, and
    its chosen in-body mode values (including the 2D/3D ``dim`` lever)."""

    id: str
    op_key: str
    params: Dict[str, Any] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)

    def spec(self) -> Optional[NodeSpec]:
        return NODES.get(self.op_key)

    def state(self, spec: Optional[NodeSpec] = None) -> Dict[str, str]:
        """Resolved mode state: the spec's defaults overridden by this instance's
        chosen modes. Unknown ops (spec=None) fall back to the raw ``modes``."""
        spec = spec or self.spec()
        base = spec.default_state() if spec is not None else {}
        base.update(self.modes)
        return base


#: Edge kinds that are **invisible to the DAG** — skipped by ``preds`` /
#: ``dataset_preds`` / ``roots`` / ``topo_order`` so the graph stays acyclic for ordering
#: and cycle rejection (V2.00 §7/§8). Each is consumed by a rewrite that runs *before* the
#: engine ever sees the graph, and each one exists because the wire the user draws is a
#: genuine cycle on the canvas while the computation it denotes is not:
#:
#: * ``"back"``   — a zone feedback wire (``Repeat/Sim Out`` → ``In``), consumed by
#:   :mod:`nodegraph.zones` when it unrolls a zone into a per-iteration chain.
#: * ``"driver"`` — a ``flow.iterate`` wire from a swept variable's output into the
#:   PARAMETER of a node inside its cone, consumed by :mod:`nodegraph.iterate`. The
#:   Iterate card drives a node whose result flows back into its own ``collect`` input,
#:   so driver + collect close a loop on screen; the unrolled form is N independent
#:   (or, in feedback mode, chained) copies with no loop at all.
#:
#: This is a **set**, not a literal comparison against ``"back"``, precisely so a second
#: non-DAG kind cannot be added by declaring it and silently be treated as forward — the
#: failure mode would be a cycle rejection at edit time and, if that were bypassed, an
#: infinite recursion in ``Engine._entry``.
NON_DAG_KINDS: FrozenSet[str] = frozenset({"back", "driver"})


def is_dag_edge(edge: "Edge") -> bool:
    """True for an edge the topological order and the engine walk actually follow."""
    return edge.kind not in NON_DAG_KINDS


@dataclass(frozen=True)
class Edge:
    """A directed wire ``src.src_socket → dst.dst_socket``.

    ``kind`` is ``"forward"`` (default) or one of :data:`NON_DAG_KINDS` — ``"back"`` (a
    zone feedback edge) or ``"driver"`` (a ``flow.iterate`` parameter wire). A non-DAG
    edge is skipped by ``preds``/``dataset_preds``/``roots``/``topo_order``, so a zoned or
    swept graph stays acyclic for ordering + cycle rejection; the corresponding rewrite
    (:mod:`nodegraph.zones` / :mod:`nodegraph.iterate`) consumes it and emits a flat DAG.
    """

    src: str
    dst: str
    src_socket: str = "out"
    dst_socket: str = "data"
    kind: str = "forward"


class Graph:
    """A bag of node instances + edges with topological ordering."""

    def __init__(self, nodes: Optional[Mapping[str, NodeInstance]] = None,
                 edges: Optional[List[Edge]] = None) -> None:
        self.nodes: Dict[str, NodeInstance] = dict(nodes or {})
        self.edges: List[Edge] = list(edges or [])

    # ── build ────────────────────────────────────────────────────────────────
    def add(self, node: NodeInstance) -> NodeInstance:
        self.nodes[node.id] = node
        return node

    def connect(self, src: str, dst: str, *, src_socket: str = "out",
                dst_socket: str = "data", kind: str = "forward") -> Edge:
        e = Edge(src, dst, src_socket, dst_socket, kind)
        self.edges.append(e)
        return e

    # ── query ────────────────────────────────────────────────────────────────
    def preds(self, node_id: str) -> List[Edge]:
        """Forward edges INTO ``node_id`` (its incoming wires). :data:`NON_DAG_KINDS`
        edges are excluded — a zone back-edge and an Iterate driver wire are both consumed
        by their rewrite before the engine runs, and are invisible to the DAG (V2.00 §8)."""
        return [e for e in self.edges if e.dst == node_id and is_dag_edge(e)]

    def dataset_preds(self, node_id: str) -> List[Edge]:
        """Incoming edges landing on a DATASET input socket of ``node_id`` — the
        metadata-carrying wires. Falls back to all preds if the op is unknown.

        Ordered by the node's **declared dataset-socket position** (stable within a
        socket, so a ``multi`` socket keeps insertion order) — NOT edge-insertion order.
        This makes ``dpreds[0]`` the FIRST-declared dataset input (the *primary*, e.g.
        ``data`` before an optional ``reference``) regardless of the order the edges were
        wired, so :func:`nodegraph.metadata.propagate_meta` seeds a node's calibration
        env from its primary input — matching the engine's own socket-ordered ``preds``.
        """
        node = self.nodes.get(node_id)
        spec = node.spec() if node else None
        if spec is None:
            return self.preds(node_id)
        ds_inputs = [s.name for s in spec.inputs if s.type.value == "dataset"]
        pos = {name: i for i, name in enumerate(ds_inputs)}
        ds_set = set(ds_inputs)
        edges = [e for e in self.preds(node_id) if e.dst_socket in ds_set]
        edges.sort(key=lambda e: pos.get(e.dst_socket, len(pos)))   # stable → primary first
        return edges

    def roots(self) -> List[str]:
        """Nodes with no forward incoming edge (sources — they need a seed envelope).
        A :data:`NON_DAG_KINDS` edge does not make its target a non-root."""
        has_in = {e.dst for e in self.edges if is_dag_edge(e)}
        return [nid for nid in self.nodes if nid not in has_in]

    def topo_order(self) -> List[str]:
        """Kahn topological order over the FORWARD edges; raises ``ValueError`` on a
        cycle. :data:`NON_DAG_KINDS` edges are excluded, so a well-formed zoned or swept
        graph orders fine and only a genuine cycle is rejected (V2.00 §7/§8)."""
        fwd = [e for e in self.edges if is_dag_edge(e)]
        indeg: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for e in fwd:
            if e.dst in indeg:
                indeg[e.dst] += 1
        q: deque = deque(sorted(nid for nid, d in indeg.items() if d == 0))
        order: List[str] = []
        while q:
            nid = q.popleft()
            order.append(nid)
            for e in sorted((e for e in fwd if e.src == nid), key=lambda e: e.dst):
                if e.dst in indeg:
                    indeg[e.dst] -= 1
                    if indeg[e.dst] == 0:
                        q.append(e.dst)
        if len(order) != len(self.nodes):
            raise ValueError("graph has a cycle (rejected outside zones, V2.00 §7)")
        return order


__all__ = ["NodeInstance", "Edge", "Graph", "NON_DAG_KINDS", "is_dag_edge"]
