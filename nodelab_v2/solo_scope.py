"""The troubleshooting scope's PLAN (2026-10-12): where in a graph the Viewer's frame picks
are applied, so that a pick always means the frame the user picked on the VIEWED node's own
M/T/Z strips.

Troubleshooting (``F9``) runs a pull over only the frames picked on the Viewer's strips, or
the one the cursor is on. It cuts the SOURCE seeds, which is right only while every node
between a source and the viewed node keeps the source's indexing. Some nodes do not: they
**re-address** an axis (:data:`READDRESSING`). A Timeseries Builder onto T makes frame 7 of
its output some file's frame 1 — and lays a bundle card's files, which arrive as positions,
onto T, so it re-addresses M too. A Time Shift plays frame 5 at 7; a Select Frame's one
frame is the source's frame k; a concatenating Merge appends one file after another. Below
such a node the cursor indexes ITS output, so cutting a source to that index builds the
shown frame out of the wrong source frames — or, with a bundle cut to one position, out of
one file: the builder then passes that file through and the built axis collapses.

So the plan cuts AFTER the last re-addressing node instead (:func:`scope_plan`): the sources
behind it stay whole on that axis, and :func:`scoped_graph` puts a frames-mode
``util.crop`` with the picks on its output in the RUN graph — the canvas and the saved file
never see it. The builder is a lazy index remap, so a cut on its output reads only the
source frames the picked output frames come from: picking 3 frames of a rebuilt 150-frame
series computes 3 frames, as picking them on a plain source does.

The plan also says how far the Viewer's strips span under the scope
(:meth:`nodelab_v2.document.NodeDocument.source_scope_totals`): an axis cut after a
re-addressing node spans that node's OUTPUT — the rebuilt series — not one file's frames.

Qt-free: the document (for the strips) and the runner (for the run) both read it.
"""
from __future__ import annotations

from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

#: The frame axes a scope can cut, in the order a pin carries them.
SCOPE_AXES: Tuple[str, ...] = ("m", "t", "z")

#: The nodes whose OUTPUT addresses an axis differently from their input.
#: ``op_key -> (mode, {mode value: axes})`` or ``op_key -> axes`` for a node that always
#: re-addresses the same ones. The builder re-addresses M on every axis it builds onto:
#: its input M is every file's positions end to end (a bundle card's files ARE positions),
#: its output M is one file's.
READDRESSING: Dict[str, Any] = {
    "util.timeseries": ("chain_axis", {"T": ("t", "m"), "M": ("m",), "Z": ("z", "m"),
                                       "C": ("m",)}),
    "util.merge": ("merge_axis", {"T": ("t",), "M": ("m",), "Z": ("z",)}),
    "util.time_shift": ("t",),
    "util.select_frame": ("t",),
}

#: The suffix of the run-graph node that carries a cut. ``#`` keeps it on its card in
#: :func:`nodelab_v2.workspace.doc_id_of` terms: its progress is the re-addressing node's.
CUT_SUFFIX = "#solo"


def readdressed_by(node: Any) -> FrozenSet[str]:
    """The axes ``node`` (anything with ``op_key`` and a ``state()`` resolving its modes
    over the spec's defaults) re-addresses — none for a muted node, which the run bypasses.

    The RESOLVED state, not the explicit modes: a builder left at its default (onto T)
    carries no ``chain_axis`` of its own, and reading only what was set missed exactly the
    most common builder there is."""
    if node is None or getattr(node, "muted", False):
        return frozenset()
    rule = READDRESSING.get(getattr(node, "op_key", ""))
    if rule is None:
        return frozenset()
    if isinstance(rule, tuple) and len(rule) == 2 and isinstance(rule[1], dict):
        mode, table = rule
        try:
            state = node.state()
        except Exception:  # noqa: BLE001 — an op that is gone: its explicit modes
            state = dict(getattr(node, "modes", None) or {})
        return frozenset(table.get(str(state.get(mode, "")).upper(), ()))
    return frozenset(rule)


class ScopePlan(NamedTuple):
    """Where a scoped pull of one node applies the picks.

    ``cuts`` — node → the axes cut on its OUTPUT (the last re-addressing node on every
    path to the target). ``whole`` — source → the axes its seed keeps whole (it is behind a
    cut). ``unscoped`` — axes no cut can serve: some node reaches the target both through a
    re-addressing node and around it, so there is no single place where the picks mean
    the target's frames. Those run whole everywhere — slower, never wrong."""

    cuts: Mapping[str, FrozenSet[str]]
    whole: Mapping[str, FrozenSet[str]]
    unscoped: FrozenSet[str]

    @property
    def readdressed(self) -> FrozenSet[str]:
        """Every axis some node upstream of the target re-addresses."""
        out = set(self.unscoped)
        for axes in self.cuts.values():
            out |= axes
        return frozenset(out)


#: The plan of a target nothing upstream re-addresses: cut at the sources, as always.
NO_PLAN = ScopePlan({}, {}, frozenset())


def scope_plan(target: str, preds: Callable[[str], Iterable[str]],
               rule: Callable[[str], FrozenSet[str]]) -> ScopePlan:
    """The :class:`ScopePlan` for a scoped pull of ``target``. ``preds(n)`` lists the nodes
    wired INTO ``n``; ``rule(n)`` the axes ``n`` re-addresses (:func:`readdressed_by`).

    Per axis: a node is *dirty* if some path from it to the target passes a re-addressing
    node after it, and has a *clean* path if some path passes none. A re-addressing node
    with no re-addressing node after it is a cut; a dirty source stays whole. A node that
    is BOTH feeds the target that axis in two index spaces at once, and the axis is
    unscoped."""
    cone: List[str] = []
    seen: set = set()
    stack = [target]
    up: Dict[str, Tuple[str, ...]] = {}
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        cone.append(n)
        up[n] = tuple(dict.fromkeys(preds(n)))
        stack.extend(up[n])
    down: Dict[str, List[str]] = {n: [] for n in cone}
    for n in cone:
        for p in up[n]:
            down.setdefault(p, []).append(n)
    rules = {n: rule(n) for n in cone}
    # consumers before producers (Kahn over the reversed cone): an Iterate unrolls into
    # chains hundreds of nodes deep, too deep for a recursive walk
    left = {n: len(down.get(n, ())) for n in cone}
    order: List[str] = [n for n in cone if not left[n]]
    for n in order:
        for p in up[n]:
            left[p] -= 1
            if not left[p]:
                order.append(p)
    cuts: Dict[str, set] = {}
    whole: Dict[str, set] = {}
    unscoped: set = set()
    for a in SCOPE_AXES:
        hits = {n for n in cone if a in rules[n]}
        if not hits:
            continue
        dirty: Dict[str, bool] = {}
        clean: Dict[str, bool] = {}
        for n in order:
            dirty[n], clean[n] = False, n == target
            for c in down.get(n, ()):
                dirty[n] = dirty[n] or c in hits or dirty.get(c, False)
                clean[n] = clean[n] or (c not in hits and clean.get(c, False))
        if len(order) < len(cone) or any(dirty[n] and clean[n] for n in order):
            unscoped.add(a)
            for n in cone:
                if not up[n]:
                    whole.setdefault(n, set()).add(a)
            continue
        for n in hits:
            if not dirty[n]:
                cuts.setdefault(n, set()).add(a)
        for n in cone:
            if not up[n] and dirty[n]:
                whole.setdefault(n, set()).add(a)
    return ScopePlan({n: frozenset(v) for n, v in cuts.items()},
                     {n: frozenset(v) for n, v in whole.items()},
                     frozenset(unscoped))


def graph_scope_plan(graph: Any, target: str) -> ScopePlan:
    """:func:`scope_plan` over a run :class:`~nodegraph.graph.Graph`."""
    nodes = getattr(graph, "nodes", {}) or {}
    return scope_plan(target, lambda n: [e.src for e in graph.preds(n)],
                      lambda n: readdressed_by(nodes.get(n)))


def source_pin(pin: Optional[Sequence[Any]], whole: Iterable[str]) -> Optional[Tuple]:
    """``pin`` as one SOURCE seed takes it: the axes the plan keeps whole there are
    ``None`` (every index of that source)."""
    if pin is None:
        return None
    keep = set(whole)
    if not keep:
        return tuple(pin)
    out = list(pin)
    for i, a in enumerate(SCOPE_AXES):
        if a in keep and i < len(out):
            out[i] = None
    return tuple(out)


def _clamped(picks: Sequence[int], n: Optional[int]) -> Tuple[int, ...]:
    """The picks clamped into ``0..n-1`` — a cursor left past the end of an axis an edit
    shrank names its last frame, as a source pin's clamp does, rather than failing the pull."""
    if not n or n < 1:
        return tuple(sorted({int(i) for i in picks}))
    return tuple(sorted({min(max(0, int(i)), int(n) - 1) for i in picks}))


def scoped_graph(graph: Any, target: str, plan: ScopePlan, pin: Optional[Sequence[Any]],
                 extent: Optional[Callable[[str], Any]] = None) -> Tuple[Any, str, Tuple]:
    """``graph`` with a frames-mode ``util.crop`` after every cut of ``plan``, carrying
    ``pin``'s picks on the axes cut there — and the id to pull (the target's cut when the
    target itself re-addresses, so the delivered payload is the picked frames), and the
    injected ``(node id, frames spec)`` pairs (the run's identity: two scopes of one
    revision are two graphs).

    ``extent(node)`` gives a cut node's output axes (an ``AxisSizes``-like), used to clamp
    the picks; without it they pass as they are. The graph handed in is not modified."""
    if pin is None or not plan.cuts:
        return graph, target, ()
    from nodegraph.graph import Edge, Graph, NodeInstance
    from nodegraph.metadata import format_frame_spec
    nodes = dict(graph.nodes)
    edges = list(graph.edges)
    injected: List[Tuple[str, str]] = []
    pull = target
    for node, axes in sorted(plan.cuts.items()):
        ax = extent(node) if extent is not None else None
        picks: Dict[str, Tuple[int, ...]] = {}
        for i, a in enumerate(SCOPE_AXES):
            want = pin[i] if i < len(pin) else None
            if a in axes and want is not None:
                picks[a] = _clamped(want, getattr(ax, a, None) if ax is not None else None)
        if not picks:
            continue
        spec = format_frame_spec(picks)
        inst = nodes.get(node)
        nspec = inst.spec() if inst is not None else None
        out_sock = nspec.outputs[0].name if nspec is not None and nspec.outputs else "out"
        cid = f"{node}{CUT_SUFFIX}"
        nodes[cid] = NodeInstance(cid, "util.crop", params={"frames": spec},
                                  modes={"region": "frames"})
        edges = [Edge(cid, e.dst, "out", e.dst_socket, e.kind)
                 if e.src == node and e.src_socket == out_sock else e for e in edges]
        edges.append(Edge(node, cid, out_sock, "data"))
        injected.append((cid, spec))
        if node == target:
            pull = cid
    if not injected:
        return graph, target, ()
    return Graph(nodes=nodes, edges=edges), pull, tuple(injected)


__all__ = ["CUT_SUFFIX", "NO_PLAN", "READDRESSING", "SCOPE_AXES", "ScopePlan",
           "graph_scope_plan", "readdressed_by", "scope_plan", "scoped_graph", "source_pin"]
