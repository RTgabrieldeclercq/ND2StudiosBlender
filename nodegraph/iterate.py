"""Parameter iteration — the ``flow.iterate`` cone rewrite (nodegraph v2, V2.19).

A **Repeat zone** iterates *data* with state carried across iterations. This module
iterates **parameters**: a ``flow.iterate`` card drives one or more params of nodes
downstream of it and collects their results back, so the same chain is evaluated once per
parameter value and exactly one of those results is preserved.

**The wire is a loop; the computation is not.** The Iterate card's variable output feeds a
param of some node, that node's result flows on and eventually returns to the card's
``collect`` input — a cycle on the canvas. The driver wires therefore carry
``kind="driver"`` and sit in :data:`~nodegraph.graph.NON_DAG_KINDS`, invisible to
``preds``/``topo_order``, and :func:`unroll` expands the whole thing into a flat DAG the
stock :class:`~nodegraph.engine.Engine` runs unchanged. Same shape as
:func:`nodegraph.zones.unroll`, same reason.

**The cone.** What gets cloned is the set of nodes on some path from a driven node to a
collect source — everything whose result the swept param can change, and nothing else. A
node feeding the cone from outside is loop-invariant and feeds every iteration.

**Two rewrites, one node.**

* ``mode="sweep"`` — every value is known before anything runs, so the rewrite **bakes**
  it into the clone's ``params`` (or ``modes``, for a swept Mode). Literal params are what
  keep :func:`nodegraph.metadata.propagate_meta` exact at edit time, which is why an
  axis-changing param (crop bounds, resample scale) and a Mode may be swept but may NOT be
  driven in feedback.
* ``mode="feedback"`` — iteration *i*'s value is a *result* of the earlier ones, so it can
  only arrive as a payload. The rewrite chains the clones through hidden
  :data:`ADVANCE_OP` nodes and the engine's wired-scalar rule
  (:func:`nodegraph.engine._with_driven_params`) folds it into the compute's params.

**The advance nodes are stateless.** Iteration *i*'s advance reads *every* earlier probe
(a ``multi`` Dataset input) and replays the whole search recurrence from scratch, rather
than carrying a bracket forward. That is deliberate: the engine is one-payload-per-node,
so a stateful advance would need a second output socket it cannot have. Replaying is
O(N²) edges for N ≤ :data:`MAX_ITERATIONS` iterations of pure arithmetic — free next to
one segmentation.

**Honest cost note.** In *sweep* mode two iterations that resolve to the same value share a
``recipe_hash`` and the second is a memo hit. In *feedback* mode they do not: clone *i*'s
value arrives from advance *i*, whose own predecessors differ, so a converged search still
recomputes its remaining iterations. Convergence saves nothing there; ``tol`` is a
reporting threshold, not an exit.

Qt-free; standard library only (no numpy — this module touches no pixels).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from nodegraph.graph import Edge, Graph, NodeInstance, is_dag_edge
from nodegraph.registry import DIM_MODE, NODES, NodeSpec
from nodegraph.sockets import SocketType

#: The iterating node's frozen op_key.
ITERATE_OP = "flow.iterate"

#: The hidden per-iteration search node minted by the FEEDBACK rewrite. Never placed by a
#: user and hidden from the palette: it exists only between two clones.
ADVANCE_OP = "flow.advance"

#: A driver edge whose destination is a **Mode** rather than a value socket uses this
#: reserved ``dst_socket`` prefix (``"__mode__:method"``). Modes have no port, so the GUI
#: card grows a synthetic one; the name never reaches the engine because :func:`unroll`
#: consumes every driver edge and bakes the value into the clone's ``modes`` dict.
MODE_TARGET_PREFIX = "__mode__:"

#: How many variables one Iterate card can sweep. Fixed because the driver outputs and
#: their value fields are DECLARED sockets gated by the ``variables`` Mode — see the
#: ``flow.iterate`` registration in :mod:`nodegraph.nodes`.
MAX_VARIABLES = 4

#: Refusal threshold on the resolved iteration count. A grid explodes multiplicatively and
#: every iteration is a full clone of the cone, so an unnoticed 8×8×4 is not a slow run,
#: it is an unusable session.
MAX_ITERATIONS = 64

#: Machine-set params. ``ITERS_KEY`` is written by :func:`unroll` onto the Iterate node so
#: its compute knows which iteration each collected payload is; ``SWEEP_KEY`` is written by
#: the GUI's "Run sweep" action and holds the recorded table. Both are dunder-prefixed for
#: the reason ``io.dock``'s ``__bake__`` is: engine/GUI bookkeeping, not a user control, so
#: the param↔socket contract exempts them and no socket may offer them for editing.
ITERS_KEY = "__iters__"
SWEEP_KEY = "__sweep__"

#: Namespaced NON-calibration metadata keys the Iterate compute stamps on its output: the
#: per-iteration results table and the variable labels that head its columns. Metadata
#: rather than params because the compute is the only place that holds every iteration's
#: result at once, and the GUI is already handed the payload — so the table needs no second
#: pull and no extra plumbing. Non-calibration, so ``_StrictCalibMetadata`` passes it
#: through and it can never shadow a real calibration name (`wire-node-v2` §7b).
SWEEP_ROWS_KEY = "__sweep_rows__"
SWEEP_LABELS_KEY = "__sweep_labels__"

#: ``mode`` values.
MODE_SWEEP, MODE_FEEDBACK = "sweep", "feedback"
#: ``preserve`` values.
PRESERVE_FIRST, PRESERVE_LAST, PRESERVE_BEST, PRESERVE_PICKED = (
    "first", "last", "best", "picked")
#: ``combine`` values.
COMBINE_ZIP, COMBINE_GRID = "zip", "grid"
#: per-variable ``v{k}_source`` values.
SRC_LIST, SRC_LINEAR, SRC_LOG, SRC_AROUND = "list", "linear", "log", "around"
#: per-variable ``v{k}_type`` values.
TYPE_NUMBER, TYPE_TEXT = "number", "text"
#: ``search`` values.
SEARCH_GOLDEN, SEARCH_SECANT = "golden", "secant"

#: The sources that define a numeric BRACKET (and so can drive a feedback search).
_BRACKETED = (SRC_LINEAR, SRC_LOG, SRC_AROUND)

#: golden-section ratio complement, 1 - 1/φ.
_INV_PHI2 = (3.0 - math.sqrt(5.0)) / 2.0
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


# ── socket / mode names (declared ONCE, here) ────────────────────────────────

def var_mode_names(k: int) -> Tuple[str, str]:
    """``(type_mode, source_mode)`` for variable slot ``k``."""
    return (f"v{k}_type", f"v{k}_source")


def var_out_names(k: int) -> Tuple[str, str]:
    """``(numeric_output, text_output)`` for slot ``k`` — exactly one is active, gated by
    that slot's ``v{k}_type`` Mode. Two sockets rather than one because ``STRING`` has no
    implicit conversion to anything (``sockets._CONVERSIONS``), so a single port could not
    serve both a σ and a method name."""
    return (f"var{k}", f"var{k}_text")


def var_field_names(k: int) -> Dict[str, str]:
    """The value-source input sockets for slot ``k``, keyed by role."""
    return {
        "list": f"v{k}_list", "start": f"v{k}_start", "stop": f"v{k}_stop",
        "center": f"v{k}_center", "percent": f"v{k}_percent", "steps": f"v{k}_steps",
    }


# ── resolved model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Target:
    """One thing a variable drives: a value socket, or a Mode when ``is_mode``."""

    node_id: str
    name: str                       # socket name, or the Mode name when is_mode
    is_mode: bool = False

    @property
    def socket(self) -> str:
        return (MODE_TARGET_PREFIX + self.name) if self.is_mode else self.name


@dataclass(frozen=True)
class Variable:
    slot: int
    kind: str                                   # TYPE_NUMBER | TYPE_TEXT
    source: str                                 # SRC_*
    values: Tuple[Any, ...]
    targets: Tuple[Target, ...]
    bracket: Optional[Tuple[float, float]] = None   # feedback search range

    @property
    def label(self) -> str:
        """The name this variable reports under — the first target's param, which is what
        the user recognizes (``sigma``), falling back to the slot."""
        return self.targets[0].name if self.targets else f"var{self.slot}"


@dataclass(frozen=True)
class Iteration:
    index: int
    values: Tuple[Any, ...]         # one per variable, aligned with plan.variables


@dataclass(frozen=True)
class IteratePlan:
    node_id: str
    mode: str
    preserve: str
    combine: str
    search: str
    direction: str
    metric: str
    tol: float
    target_value: float
    variables: Tuple[Variable, ...]
    iterations: Tuple[Iteration, ...]
    cone: FrozenSet[str]
    collect_src: Tuple[str, ...]
    minted: Tuple[int, ...]         # the iteration indices unroll will actually mint

    @property
    def n(self) -> int:
        return len(self.iterations)

    def rows(self) -> List[Dict[str, Any]]:
        """The results-table skeleton — one row per iteration, values only. The GUI's
        "Run sweep" action fills in the metric column and writes it to ``SWEEP_KEY``."""
        return [{"iter": it.index,
                 **{v.label: it.values[j] for j, v in enumerate(self.variables)}}
                for it in self.iterations]


def iter_id(node_id: str, iterate_id: str, i: int) -> str:
    """The cloned id of ``node_id`` in iteration ``i`` of ``iterate_id`` — the same
    ``base#owner@index`` shape :func:`nodegraph.zones.iter_id` uses, so one convention
    covers every unroll in the engine."""
    return f"{node_id}#{iterate_id}@{i}"


def advance_id(iterate_id: str, i: int) -> str:
    return f"{iterate_id}#adv@{i}"


# ── value generation ──────────────────────────────────────────────────────────

def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_list(text: Any, numeric: bool) -> Tuple[Any, ...]:
    """The ``list`` source: comma (or semicolon) separated. Numeric slots parse to float
    and refuse a non-number rather than silently dropping it — a typo in a sweep list would
    otherwise shorten the sweep by one and look like nothing happened."""
    toks = [t.strip() for t in str(text or "").replace(";", ",").split(",")]
    toks = [t for t in toks if t]
    if not numeric:
        return tuple(toks)
    out: List[float] = []
    for t in toks:
        try:
            out.append(float(t))
        except ValueError:
            raise ValueError(
                f"{t!r} is not a number — a numeric sweep list must be values like "
                f"'0.3, 0.5, 0.8'. (Set this variable's Type to 'text' to sweep names.)")
    return tuple(out)


def span(start: float, stop: float, steps: int, *, log: bool = False) -> Tuple[float, ...]:
    """``steps`` values from ``start`` to ``stop`` inclusive, linearly or logarithmically.
    One step yields ``start`` alone (not the midpoint) — the least surprising reading of
    "one value from this range"."""
    n = max(1, int(steps))
    if n == 1:
        return (float(start),)
    if log:
        if start <= 0 or stop <= 0:
            raise ValueError(
                f"a logarithmic sweep needs positive endpoints (got {start} → {stop}); "
                f"use linear spacing, or move the range above zero")
        la, lb = math.log(float(start)), math.log(float(stop))
        return tuple(math.exp(la + (lb - la) * i / (n - 1)) for i in range(n))
    a, b = float(start), float(stop)
    return tuple(a + (b - a) * i / (n - 1) for i in range(n))


def around(center: float, percent: float, steps: int) -> Tuple[float, ...]:
    """``steps`` values spanning ``center`` ± ``percent``%. The metadata-anchored source:
    with an empty center the caller resolves the TARGET socket's own value — its user
    override, else its ``derive`` against the live envelope — so the sweep follows the
    file's optics instead of pinning numbers authored against some other file."""
    c, p = float(center), max(0.0, float(percent)) / 100.0
    return span(c * (1.0 - p), c * (1.0 + p), steps)


# ── search (feedback mode) ────────────────────────────────────────────────────

def advance_value(search: str, lo: float, hi: float, *, direction: str = "max",
                  target: float = 0.0,
                  probes: Sequence[Tuple[float, float]] = ()) -> float:
    """The next x to probe, given every ``(x, metric)`` already measured, in order.

    **Pure and replayable** — it holds no state, which is what lets each advance node
    reconstruct the search from the probes alone (see the module docstring). ``golden``
    runs a golden-section extremum search over ``[lo, hi]``; ``secant`` drives the metric
    to ``target`` by the secant step, clamped into the bracket and falling back to
    bisection whenever the step is degenerate or leaves it."""
    lo, hi = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
    n = len(probes)
    if search == SEARCH_SECANT:
        if n == 0:
            return lo
        if n == 1:
            return hi
        (x0, f0), (x1, f1) = probes[-2], probes[-1]
        g0, g1 = f0 - target, f1 - target
        if g1 != g0:
            x = x1 - g1 * (x1 - x0) / (g1 - g0)
            if lo <= x <= hi and x not in (x0, x1):
                return x
        return 0.5 * (x0 + x1) if x0 != x1 else 0.5 * (lo + hi)
    # golden-section. The first two probes are the canonical interior pair; from there each
    # measured pair shrinks the bracket, replayed from the front.
    if n == 0:
        return lo + _INV_PHI2 * (hi - lo)
    if n == 1:
        return lo + _INV_PHI * (hi - lo)
    better = (lambda a, b: a > b) if direction == "max" else (lambda a, b: a < b)
    a, b = lo, hi
    c, fc = probes[0]
    d, fd = probes[1]
    for k in range(2, n + 1):
        if better(fc, fd):
            b, d, fd = d, c, fc
            c = a + _INV_PHI2 * (b - a)
            if k < n:
                c, fc = probes[k]
            else:
                return c
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            if k < n:
                d, fd = probes[k]
            else:
                return d
    return 0.5 * (a + b)


# ── plan resolution ───────────────────────────────────────────────────────────

def _param(spec: Optional[NodeSpec], params: Mapping[str, Any], name: str,
           fallback: Any = None) -> Any:
    """A param resolved override → ``SocketSpec.default`` → ``fallback``. The engine hands
    computes raw overrides and never default-fills, so the plan must do it here — and must
    do it the same way ``registry.layer_value`` does for layer names: one declared default,
    in the SocketSpec."""
    v = params.get(name)
    if v is None or v == "":
        sock = spec.input(name) if spec is not None else None
        v = getattr(sock, "default", None) if sock is not None else None
    return fallback if v is None or v == "" else v


def _target_default(graph: Graph, envs: Optional[Mapping[str, Any]],
                    target: Target) -> Optional[float]:
    """The value a driven socket would have if nobody swept it: the user's override, else
    its ``derive`` against that node's live envelope, else its static default. This is what
    an empty ``around`` center resolves to — the same override → derive → default order
    :class:`~nodegraph.engine.ChannelContext` uses, so an anchored sweep centres on exactly
    the number the inspector is showing."""
    node = graph.nodes.get(target.node_id)
    if node is None or target.is_mode:
        return None
    spec = node.spec()
    if spec is None:
        return None
    if target.name in node.params:
        return _as_float(node.params[target.name], 0.0)
    sock = spec.input(target.name)
    if sock is None:
        return None
    if getattr(sock, "derive", "") and envs is not None and target.node_id in envs:
        from nodegraph.metadata import envelope_symbols, eval_derive
        try:
            return float(eval_derive(sock.derive, envelope_symbols(envs[target.node_id])))
        except Exception:                       # noqa: BLE001 — fall through to the default
            pass
    d = getattr(sock, "default", None)
    return None if d is None else _as_float(d, 0.0)


def _driver_targets(graph: Graph, node_id: str) -> Dict[int, List[Target]]:
    """Slot index → the targets its driver wires land on, in stable wire order."""
    spec = NODES.get(ITERATE_OP)
    out_slot: Dict[str, int] = {}
    for k in range(MAX_VARIABLES):
        num, txt = var_out_names(k)
        out_slot[num] = k
        out_slot[txt] = k
    by_slot: Dict[int, List[Target]] = {}
    for e in graph.edges:
        if e.kind != "driver" or e.src != node_id:
            continue
        slot = out_slot.get(e.src_socket)
        if slot is None:
            raise ValueError(
                f"{node_id}: a driver wire leaves socket {e.src_socket!r}, which is not one "
                f"of this node's variable outputs")
        if e.dst_socket.startswith(MODE_TARGET_PREFIX):
            tgt = Target(e.dst, e.dst_socket[len(MODE_TARGET_PREFIX):], is_mode=True)
        else:
            tgt = Target(e.dst, e.dst_socket, is_mode=False)
        by_slot.setdefault(slot, []).append(tgt)
    _ = spec
    return by_slot


def _check_target(graph: Graph, plan_mode: str, target: Target) -> None:
    """Refuse a driver wire that cannot mean what it appears to mean."""
    node = graph.nodes.get(target.node_id)
    if node is None:
        raise ValueError(f"a driver wire points at unknown node {target.node_id!r}")
    spec = node.spec()
    label = (getattr(spec, "label", "") or node.op_key) if spec else node.op_key
    if target.is_mode:
        mode = next((m for m in (spec.modes if spec else ()) if m.name == target.name), None)
        if mode is None:
            raise ValueError(
                f"{label} has no mode {target.name!r} to drive")
        if mode.is_dim_lever or target.name == DIM_MODE:
            raise ValueError(
                f"the 2D/3D lever on {label} cannot be swept: it selects the node's "
                f"data-access footprint and can be outright invalid (3D on a z==1 series, "
                f"or a method that refuses 3D), so iterations would raise rather than "
                f"compare. Sweep a parameter instead, or place two nodes.")
        if plan_mode == MODE_FEEDBACK:
            raise ValueError(
                f"a Mode ({label}.{target.name}) can only be swept, not driven in feedback "
                f"mode — its value has to be literal on the node before the run so the "
                f"footprint and the socket set resolve. Set Mode to 'sweep'.")
        return
    sock = spec.input(target.name) if spec else None
    if sock is None:
        raise ValueError(f"{label} has no parameter {target.name!r} to drive")
    if sock.type is SocketType.DATASET:
        raise ValueError(
            f"{label}.{target.name} is a Dataset input, not a parameter — wire the image "
            f"there directly")
    if plan_mode == MODE_FEEDBACK and spec is not None and spec.meta_transform is not None:
        raise ValueError(
            f"{label}.{target.name} belongs to a node that changes the axes or calibration "
            f"of its output, so its value must be known before the run — the edit-time "
            f"metadata pass cannot see a value that arrives on a wire. Sweep it instead of "
            f"driving it in feedback mode.")


def _resolve_variables(graph: Graph, node: NodeInstance, spec: Optional[NodeSpec],
                       state: Mapping[str, str], envs: Optional[Mapping[str, Any]],
                       plan_mode: str) -> Tuple[Variable, ...]:
    n_vars = max(1, min(MAX_VARIABLES, int(_as_float(state.get("variables", "1"), 1))))
    targets = _driver_targets(graph, node.id)
    out: List[Variable] = []
    for k in range(n_vars):
        tgts = tuple(targets.get(k, ()))
        if not tgts:
            continue                     # an armed-but-unwired slot contributes nothing
        for t in tgts:
            _check_target(graph, plan_mode, t)
        type_mode, source_mode = var_mode_names(k)
        kind = str(state.get(type_mode) or TYPE_NUMBER)
        source = str(state.get(source_mode) or SRC_LIST)
        f = var_field_names(k)
        steps = int(_as_float(_param(spec, node.params, f["steps"], 5), 5))
        bracket: Optional[Tuple[float, float]] = None
        if source == SRC_LIST:
            values = parse_list(_param(spec, node.params, f["list"], ""),
                                kind == TYPE_NUMBER)
        elif source in (SRC_LINEAR, SRC_LOG):
            lo = _as_float(_param(spec, node.params, f["start"], 0.0), 0.0)
            hi = _as_float(_param(spec, node.params, f["stop"], 1.0), 1.0)
            values = span(lo, hi, steps, log=(source == SRC_LOG))
            bracket = (lo, hi)
        elif source == SRC_AROUND:
            centre = _param(spec, node.params, f["center"], None)
            if centre is None:
                centre = _target_default(graph, envs, tgts[0])
            if centre is None:
                raise ValueError(
                    f"variable {k} is anchored 'around' its target's own value, but "
                    f"{tgts[0].node_id}.{tgts[0].name} has no resolvable value to centre "
                    f"on — type a centre, or use an explicit range")
            pct = _as_float(_param(spec, node.params, f["percent"], 50.0), 50.0)
            values = around(_as_float(centre, 0.0), pct, steps)
            bracket = (values[0], values[-1]) if values else None
        else:
            raise ValueError(f"unknown value source {source!r} on variable {k}")
        if not values:
            raise ValueError(
                f"variable {k} (driving {tgts[0].node_id}.{tgts[0].name}) has no values — "
                f"fill in its list or range")
        out.append(Variable(k, kind, source, tuple(values), tgts, bracket))
    if not out:
        raise ValueError(
            f"{node.id}: no variable is wired to anything — drag a wire from a variable "
            f"output onto the parameter you want to iterate")
    return tuple(out)


def _combos(variables: Sequence[Variable], combine: str) -> Tuple[Iteration, ...]:
    if len(variables) == 1:
        return tuple(Iteration(i, (v,)) for i, v in enumerate(variables[0].values))
    if combine == COMBINE_ZIP:
        lens = {len(v.values) for v in variables}
        if len(lens) != 1:
            raise ValueError(
                "zip combines the i-th value of every variable, so they must be the same "
                "length (got %s). Give them equal counts, or switch Combine to 'grid'."
                % ", ".join(f"{v.label}:{len(v.values)}" for v in variables))
        n = lens.pop()
        return tuple(Iteration(i, tuple(v.values[i] for v in variables))
                     for i in range(n))
    rows: List[Tuple[Any, ...]] = [()]
    for v in variables:                          # last variable varies fastest
        rows = [row + (val,) for row in rows for val in v.values]
    return tuple(Iteration(i, row) for i, row in enumerate(rows))


def iterate_nodes(graph: Graph) -> Tuple[str, ...]:
    return tuple(sorted(nid for nid, n in graph.nodes.items() if n.op_key == ITERATE_OP))


def is_driving(graph: Graph, node_id: str) -> bool:
    """True when this Iterate node has at least one driver wire attached.

    :func:`unroll` skips the ones that do not, rather than refusing. An Iterate that drives
    nothing turns up constantly in ordinary editing — the moment it is placed, the moment a
    target is muted or deleted — and failing the whole run-graph build over a half-finished
    edit would be hostile. Nothing is lost by passing it over: with no rewrite the node keeps
    no ``__iters__``, so its compute takes the not-rewritten branch and refuses at pull time
    if the sockets really do describe a sweep. The dangerous case is still caught; the
    mid-edit case is not punished for it."""
    return any(e.kind == "driver" and e.src == node_id for e in graph.edges)


def _forward_maps(graph: Graph) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    succ: Dict[str, List[str]] = {}
    pred: Dict[str, List[str]] = {}
    for e in graph.edges:
        if not is_dag_edge(e):
            continue
        succ.setdefault(e.src, []).append(e.dst)
        pred.setdefault(e.dst, []).append(e.src)
    return succ, pred


def _closure(adj: Mapping[str, List[str]], seeds: Iterable[str]) -> set:
    seen, stack = set(), list(seeds)
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(adj.get(nid, ()))
    return seen


def _is_docked(node: NodeInstance) -> bool:
    """Duck-typed dock check. Deliberately NOT ``nodelab_v2.ops.is_docked``: that module is
    Qt-free but sits *above* nodegraph, and importing it here would invert the dependency
    for the sake of two string constants."""
    return (node.op_key == "io.dock"
            and str((node.modes or {}).get("state") or "live") == "docked")


def plan(graph: Graph, node_id: str, *, envs: Optional[Mapping[str, Any]] = None,
         sweep_all: bool = False) -> IteratePlan:
    """Resolve one Iterate node into the iterations, the cone and the clones to mint.

    Every refusal below is a configuration that is *expressible* but produces a
    wrong-looking-right RESULT rather than an error if it is allowed — a sweep whose rows
    are all identical, or a chain whose iterations differ upstream of a wall. That is the
    class of defect this codebase spends the most effort designing out, so they are hard
    errors naming the fix."""
    node = graph.nodes[node_id]
    spec = node.spec()
    state = node.state(spec)
    mode = str(state.get("mode") or MODE_SWEEP)
    preserve = str(state.get("preserve") or PRESERVE_PICKED)
    combine = str(state.get("combine") or COMBINE_GRID)
    search = str(state.get("search") or SEARCH_GOLDEN)
    direction = str(state.get("direction") or "max")

    collect_src = tuple(e.src for e in graph.preds(node_id) if e.dst_socket == "collect")
    if not collect_src:
        raise ValueError(
            f"{node_id}: nothing is wired into 'collect' — connect the END of the chain "
            f"you are iterating back into it, so the sweep knows what to compare")

    variables = _resolve_variables(graph, node, spec, state, envs, mode)

    # ── the cone: driven ∪ descendants(driven), intersected with ancestors(collect) ──
    succ, pred = _forward_maps(graph)
    driven = {t.node_id for v in variables for t in v.targets}
    downstream = _closure(succ, driven)
    upstream = _closure(pred, collect_src)
    cone = frozenset(downstream & upstream)

    stranded = sorted(n for n in driven if n not in cone)
    if stranded:
        raise ValueError(
            f"{node_id}: driven node(s) {stranded} are not upstream of the 'collect' input, "
            f"so cloning them would change nothing and the sweep would silently do nothing. "
            f"Wire 'collect' to a node downstream of them.")

    docked = sorted(n for n in cone if _is_docked(graph.nodes[n]))
    if docked:
        raise ValueError(
            f"{node_id}: the iterated chain contains docked Dock node(s) {docked}. A docked "
            f"dock serves its checkpoint and its upstream is cut, so every iteration would "
            f"read the same baked pixels and produce identical results. Set it to 'live', "
            f"or move the Iterate node downstream of it.")

    nested = sorted(n for n in cone if graph.nodes[n].op_key == ITERATE_OP)
    if nested:
        raise ValueError(
            f"{node_id}: Iterate node(s) {nested} sit inside this one's iterated chain. "
            f"Nested iteration is not supported — sweep several parameters from ONE Iterate "
            f"card instead (raise 'Variables' and set Combine to 'grid').")

    # An edge into ANY Iterate node's `collect` is a collection point, not an escaping
    # branch — exempting it is what lets the two genuinely-different multi-Iterate mistakes
    # (overlapping cones, one param driven twice) reach their own specific messages instead
    # of all being reported as "this branch escapes".
    def _is_collect(e: Edge) -> bool:
        dst = graph.nodes.get(e.dst)
        return (dst is not None and dst.op_key == ITERATE_OP and e.dst_socket == "collect")

    escapes = sorted({(e.src, e.dst) for e in graph.edges
                      if is_dag_edge(e) and e.src in cone and e.dst not in cone
                      and e.dst != node_id and not _is_collect(e)})
    if escapes:
        src, dst = escapes[0]
        raise ValueError(
            f"{node_id}: {src!r} is inside the iterated chain but also feeds {dst!r} outside "
            f"it, which has no single iteration to read. Route that branch out of the "
            f"Iterate node's output instead, or move it upstream of the driven node."
            + (f" ({len(escapes) - 1} more like it.)" if len(escapes) > 1 else ""))

    # ── iterations ──────────────────────────────────────────────────────────────
    if mode == MODE_FEEDBACK:
        if len(variables) != 1:
            raise ValueError(
                f"{node_id}: feedback mode searches ONE variable (golden-section and secant "
                f"are both 1-D); {len(variables)} are wired. Set Variables to 1, or switch "
                f"Mode to 'sweep' to explore a grid.")
        var = variables[0]
        if var.kind != TYPE_NUMBER or var.bracket is None:
            raise ValueError(
                f"{node_id}: feedback mode needs a numeric SEARCH RANGE to work inside — set "
                f"this variable's Source to linear, log or around (a 'list' has no bracket, "
                f"and text values cannot be searched).")
        n = max(2, len(var.values))
        iterations = tuple(Iteration(i, (None,)) for i in range(n))
        preserve = PRESERVE_BEST         # forced; see the registration's refusal
    else:
        iterations = _combos(variables, combine)

    if len(iterations) > MAX_ITERATIONS:
        raise ValueError(
            f"{node_id}: this resolves to {len(iterations)} iterations, past the limit of "
            f"{MAX_ITERATIONS}. Every iteration is a full copy of the chain, so shorten a "
            f"variable, or switch Combine from 'grid' to 'zip'.")

    metric = str(_param(spec, node.params, "metric", "") or "")
    if preserve == PRESERVE_BEST and not metric:
        raise ValueError(
            f"{node_id}: Preserve is 'best', which needs a score to compare — name a Global "
            f"scalar in 'Metric'. Add a 'Reduce → Scalar' node inside the iterated chain to "
            f"produce one.")

    # ── which clones to mint ────────────────────────────────────────────────────
    if sweep_all or mode == MODE_FEEDBACK or preserve == PRESERVE_BEST:
        minted = tuple(range(len(iterations)))
    elif preserve == PRESERVE_FIRST:
        minted = (0,)
    elif preserve == PRESERVE_LAST:
        minted = (len(iterations) - 1,)
    else:
        idx = int(_as_float(_param(spec, node.params, "index", 0), 0))
        minted = (max(0, min(len(iterations) - 1, idx)),)

    return IteratePlan(
        node_id=node_id, mode=mode, preserve=preserve, combine=combine, search=search,
        direction=direction, metric=metric,
        tol=_as_float(_param(spec, node.params, "tol", 0.0), 0.0),
        target_value=_as_float(_param(spec, node.params, "target", 0.0), 0.0),
        variables=variables, iterations=iterations, cone=cone,
        collect_src=collect_src, minted=minted)


# ── the rewrite ───────────────────────────────────────────────────────────────

def _clone_params(p: IteratePlan, base: NodeInstance, i: int) -> Tuple[dict, dict]:
    """``(params, modes)`` for one cone node in iteration ``i``, with every value this
    iteration bakes already applied. Feedback bakes only iteration 0 (its first probe is
    deterministic); later iterations receive their value on a wire instead."""
    params, modes = dict(base.params), dict(base.modes)
    it = p.iterations[i]
    for j, var in enumerate(p.variables):
        if p.mode == MODE_FEEDBACK:
            if i != 0:
                # The value arrives on a wire from this iteration's advance node, so the
                # user's own literal must GO. Leaving it would put a second, stale value in
                # `params` that the engine's wired-scalar rule silently overrides — two
                # sources of truth for one number, and the losing one is the one the
                # inspector shows.
                for t in var.targets:
                    if t.node_id == base.id and not t.is_mode:
                        params.pop(t.name, None)
                continue
            lo, hi = var.bracket or (0.0, 1.0)
            value: Any = advance_value(p.search, lo, hi, direction=p.direction,
                                       target=p.target_value, probes=())
        else:
            value = it.values[j]
        for t in var.targets:
            if t.node_id != base.id:
                continue
            if t.is_mode:
                modes[t.name] = str(value)
            else:
                params[t.name] = value
    return params, modes


def _apply(p: IteratePlan, nodes: Dict[str, NodeInstance],
           edges: List[Edge]) -> Tuple[Dict[str, NodeInstance], List[Edge]]:
    cone, nid = p.cone, p.node_id
    # snapshot the originals BEFORE the cone leaves the working dict — the clones are built
    # from them, and with two Iterate plans in one graph the second pass would otherwise
    # look for nodes the first has already replaced.
    originals = {c: nodes[c] for c in cone}
    kept: List[Edge] = []
    for e in edges:
        if e.kind == "driver" and e.src == nid:
            continue                                   # consumed: baked or re-wired below
        if e.src in cone or e.dst in cone:
            continue                                   # re-emitted per iteration
        kept.append(e)
    for c in cone:
        nodes.pop(c, None)

    inner = [e for e in edges if is_dag_edge(e) and e.src in cone and e.dst in cone]
    incoming = [e for e in edges if is_dag_edge(e) and e.src not in cone and e.dst in cone]
    collect = [e for e in edges
               if is_dag_edge(e) and e.dst == nid and e.dst_socket == "collect"]

    for i in p.minted:
        for c in sorted(cone):
            base = originals[c]
            cp, cm = _clone_params(p, base, i)
            cid = iter_id(c, nid, i)
            nodes[cid] = NodeInstance(cid, base.op_key, params=cp, modes=cm)
        for e in inner:
            kept.append(Edge(iter_id(e.src, nid, i), iter_id(e.dst, nid, i),
                             e.src_socket, e.dst_socket))
        for e in incoming:                             # loop-invariant: feeds every iteration
            kept.append(Edge(e.src, iter_id(e.dst, nid, i), e.src_socket, e.dst_socket))
        for e in collect:
            kept.append(Edge(iter_id(e.src, nid, i), nid, e.src_socket, "collect"))

    if p.mode == MODE_FEEDBACK:
        var = p.variables[0]
        lo, hi = var.bracket or (0.0, 1.0)
        for i in p.minted:
            if i == 0:
                continue                               # its value is baked (see _clone_params)
            aid = advance_id(nid, i)
            nodes[aid] = NodeInstance(aid, ADVANCE_OP, params={
                "__search__": p.search, "__direction__": p.direction,
                "__metric__": p.metric, "__lo__": lo, "__hi__": hi,
                "__target__": p.target_value, "__index__": i})
            for j in range(i):                         # every earlier probe, in order
                for e in collect:
                    kept.append(Edge(iter_id(e.src, nid, j), aid, e.src_socket, "probes"))
            for t in var.targets:
                kept.append(Edge(aid, iter_id(t.node_id, nid, i), "value", t.name))

    node = nodes[nid]
    nodes[nid] = NodeInstance(
        nid, node.op_key,
        params={**node.params, ITERS_KEY: [
            {"index": i,
             "values": [None if p.mode == MODE_FEEDBACK else p.iterations[i].values[j]
                        for j in range(len(p.variables))]}
            for i in p.minted],
            "__labels__": [v.label for v in p.variables]},
        modes=dict(node.modes))
    return nodes, kept


def unroll(graph: Graph, *, envs: Optional[Mapping[str, Any]] = None,
           sweep_all: Iterable[str] = ()) -> Graph:
    """Expand every ``flow.iterate`` node into a flat DAG the stock engine can run.

    Returns the SAME graph object when there is nothing to do, so the common no-Iterate
    case costs one dict scan. ``sweep_all`` names the Iterate nodes that must mint every
    iteration regardless of their preserve mode — the GUI's "Run sweep" action, which needs
    all N results to fill the table.
    """
    ids = tuple(nid for nid in iterate_nodes(graph) if is_driving(graph, nid))
    if not ids:
        return graph
    all_sweep = set(sweep_all)
    plans = [plan(graph, nid, envs=envs, sweep_all=(nid in all_sweep)) for nid in ids]
    # Checked in this order deliberately: two cards fighting over ONE parameter is the more
    # specific diagnosis, and their cones necessarily overlap too — reporting the overlap
    # first would describe the symptom instead of the cause.
    seen: Dict[Tuple[str, str], str] = {}
    for p in plans:
        for var in p.variables:
            for t in var.targets:
                key = (t.node_id, t.socket)
                if key in seen:
                    raise ValueError(
                        f"{t.node_id}.{t.name} is driven by two Iterate nodes "
                        f"({seen[key]} and {p.node_id}) — one parameter can only follow one "
                        f"of them. Remove a driver wire.")
                seen[key] = p.node_id
    for a in range(len(plans)):
        for b in range(a + 1, len(plans)):
            shared = sorted(plans[a].cone & plans[b].cone)
            if shared:
                raise ValueError(
                    f"the chains iterated by {plans[a].node_id} and {plans[b].node_id} "
                    f"overlap at {shared} — a node cannot be cloned by two sweeps at once "
                    f"(which iteration of the first would each iteration of the second "
                    f"read?). Sweep both parameters from ONE Iterate card instead: raise "
                    f"'Variables' and set Combine to 'grid'.")
    # Two Iterate nodes cannot share a cone node (an escape refusal and the nesting refusal
    # between them see to that), so the plans compose by simple sequential application.
    nodes: Dict[str, NodeInstance] = dict(graph.nodes)
    edges: List[Edge] = list(graph.edges)
    for p in plans:
        nodes, edges = _apply(p, nodes, edges)
    return Graph(nodes, edges)


__all__ = [
    "ITERATE_OP", "ADVANCE_OP", "MODE_TARGET_PREFIX", "MAX_VARIABLES", "MAX_ITERATIONS",
    "ITERS_KEY", "SWEEP_KEY", "SWEEP_ROWS_KEY", "SWEEP_LABELS_KEY",
    "MODE_SWEEP", "MODE_FEEDBACK", "is_driving",
    "PRESERVE_FIRST", "PRESERVE_LAST", "PRESERVE_BEST", "PRESERVE_PICKED",
    "COMBINE_ZIP", "COMBINE_GRID", "SRC_LIST", "SRC_LINEAR", "SRC_LOG", "SRC_AROUND",
    "TYPE_NUMBER", "TYPE_TEXT", "SEARCH_GOLDEN", "SEARCH_SECANT",
    "Target", "Variable", "Iteration", "IteratePlan",
    "var_mode_names", "var_out_names", "var_field_names",
    "iter_id", "advance_id", "iterate_nodes", "parse_list", "span", "around",
    "advance_value", "plan", "unroll",
]
