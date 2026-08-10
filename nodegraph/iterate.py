"""Parameter iteration — the ``flow.iterate`` **segment** rewrite (nodegraph v2, V2.22).

A **Repeat zone** iterates *data* with state carried across iterations. This module
iterates **parameters**: a ``flow.iterate`` card drives one or more params of the nodes in a
declared stretch of the graph, so that stretch is evaluated once per parameter value and
exactly one of those results is preserved.

**The segment IS the wiring.** The card has no data route through it. You wire two things:
the variable outputs onto the params you want iterated, and the **segment** — ``from``, the
first node of the series to re-run, and ``to``, the last. Everything between them is cloned
per iteration; everything above ``from`` is loop-invariant and computed once.

**The result leaves through ``to``, not through the card.** The rewrite clones the segment
and then mints a *selector* — a ``flow.iterate`` node carrying :data:`ITERS_KEY` — **under
the end node's own id**. So every consumer of that node, wired before the Iterate card
existed, keeps reading it and now gets the preserved iteration. Nothing downstream is
re-routed, and iteration happens *automatically* the moment anything below the segment is
pulled, which is the whole point of the design: the card is a control, like Blender's
Random Value, not a stage in the pipeline. The card's own ``out`` serves the same payload
(it reads the selector through its ``to`` wire) for anyone who prefers an explicit route.

**The wire is a loop; the computation is not.** The card's variable output feeds a param of
a node inside the segment, whose result flows on and returns to the card's ``to`` input — a
cycle on the canvas. The driver wires therefore carry ``kind="driver"`` and sit in
:data:`~nodegraph.graph.NON_DAG_KINDS`, invisible to ``preds``/``topo_order``, and
:func:`unroll` expands the whole thing into a flat DAG the stock
:class:`~nodegraph.engine.Engine` runs unchanged. Same shape as
:func:`nodegraph.zones.unroll`, same reason.

**The cone.** What gets cloned is the set of nodes on some path from a driven node to
``to`` — everything whose result the swept param can change, and nothing else — intersected
with the descendants of ``from`` when one is wired. An unwired ``from`` therefore means
"start wherever the driven params are", which is the one-wire case; wiring it PINS the
start so the stretch cannot silently grow when a param further up is driven later.

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

**Choosing the target.** :func:`candidate_targets` scrapes the segment and returns every
param and Mode this card could legally drive, so the GUI's "iterate on" dropdown is a view
of the graph rather than a list anyone maintains. It filters through
:func:`_check_target` — the same predicate the rewrite refuses on — which is the whole point:
a menu built from a different rule than the refusals would become a way to construct exactly
the configurations those refusals exist to prevent.

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
from nodegraph.sockets import SocketType, can_convert

#: The iterating node's frozen op_key.
ITERATE_OP = "flow.iterate"

#: The hidden per-iteration search node minted by the FEEDBACK rewrite. Never placed by a
#: user and hidden from the palette: it exists only between two clones.
ADVANCE_OP = "flow.advance"

#: The two segment input sockets. ``SEG_TO`` is where the series ENDS — the node whose id
#: the rewrite's selector takes over, and therefore the one place a result can leave. It is
#: ``multi`` for one reason: the selector minted under that node's id reuses this very
#: socket to receive one payload per iteration, so the card and the selector are the same
#: node type reading the same port in two roles.
#:
#: ``SEG_FROM`` is optional and takes no payload at all — the rewrite reads it as a
#: STRUCTURAL reference (which node the stretch starts at) and consumes it, exactly like a
#: driver wire. It is a Dataset socket because the thing it points at is a node, and a wire
#: from that node's output is how you point at one.
SEG_FROM, SEG_TO = "from", "to"

#: Written by :func:`unroll` onto the CARD when it has minted a selector elsewhere: the card
#: is then a plain pass-through of what its ``to`` wire hands it. Without it the card would
#: run its own preserve logic over the single already-selected payload and stamp iteration
#: 0's values onto whatever the selector actually chose — a caption that disagrees with the
#: pixels under it, which is the failure mode this whole module is written against.
PASSTHROUGH_KEY = "__passthrough__"

#: Written by :func:`unroll` onto the SELECTOR: which card's sweep it is running. The
#: selector wears another node's id, so without this the GUI — handed a payload from
#: ``measure`` — would have no way back to the Iterate card whose results table and
#: iteration strip that payload belongs to.
OWNER_KEY = "__owner__"

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
#: …and which CARD the table belongs to. The selector wears the end node's id, so a GUI
#: handed this payload knows it is looking at an iterated result but not whose — and the
#: results table, the iteration strip and the "keep this one" edit all live on the card.
SWEEP_OWNER_KEY = "__sweep_owner__"

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
    #: the segment's END — the node whose id the selector takes over, and the only place a
    #: result leaves the iteration.
    end: str
    #: the segment's pinned START(s), empty when ``from`` is unwired (the cone then begins
    #: at the driven nodes themselves).
    start: Tuple[str, ...]
    minted: Tuple[int, ...]         # the iteration indices unroll will actually mint
    picked: int = 0                 # the card's `index` param, clamped to the iterations

    @property
    def n(self) -> int:
        return len(self.iterations)

    @property
    def view_index(self) -> int:
        """Which iteration a node INSIDE the segment resolves to when it is viewed on its
        own — the picked ``index`` if that iteration is minted, else the first that is.

        A cone node has one result per clone and no card of its own, so "view the node I am
        tuning" has to mean *some* iteration. It means the one the iteration strip is on,
        whatever ``preserve`` says: under ``best`` the winner is not known until the run has
        happened, and freezing the view on a number nobody chose would make stepping through
        the strip do nothing to the node actually being tuned."""
        if self.picked in self.minted:
            return self.picked
        return self.minted[0] if self.minted else 0

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
        # The lever is not the only footprint-selecting Mode (V2.27). `NodeSpec.footprint_mode`
        # names whichever one keys a Mapping granularity, and the refusal above applies verbatim
        # to those: `util.zproject`'s `method` has a `none` branch that changes the OUTPUT AXES,
        # so the graph downstream is structurally different per iteration and
        # `_require_same_grid` starts refusing mid-sweep; `view.overlay`'s `output` changes the
        # data rather than a parameter.
        #
        # The `role="scope"` mode is the deliberate exception, and the only one worth sweeping:
        # comparing per-plane against per-label IS the comparison a user wants, it changes no
        # axis, and every iteration remains self-consistent (the footprint is re-resolved per
        # pull from the clone's own baked mode). Its cost is not uniform, though — a wide scope
        # can move the level derivation onto the streaming surrogate, which is an APPROXIMATION
        # for li/triangle — so the sweep is permitted and the asymmetry is named here rather
        # than discovered as two iterations computed by different algorithms.
        if (not mode.is_scope
                and isinstance(getattr(spec, "granularity", None), Mapping)
                and target.name == getattr(spec, "footprint_mode", "")):
            raise ValueError(
                f"{label}.{target.name} cannot be swept: it is this node's `footprint_mode`, so "
                f"each value selects a different data-access footprint — and unlike a "
                f"statistics population (role='scope', which IS sweepable) such a mode can "
                f"change the node's output axes or its output kind, which makes the graph "
                f"downstream structurally different per iteration rather than comparable. "
                f"Sweep a parameter, or place two nodes and compare their results.")
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
    if getattr(sock, "layer_out", ()):
        raise ValueError(
            f"{label}.{target.name} NAMES the layer this node writes — sweeping it computes "
            f"the identical result N times under N different names, and only one of them "
            f"leaves the card anyway. Iterate a parameter that changes the pixels instead.")
    wired = next((e for e in graph.preds(target.node_id)
                  if e.dst_socket == target.name), None)
    if wired is not None:
        # The engine's wired-scalar rule is explicit that a wire BEATS a param
        # (:func:`nodegraph.engine._with_driven_params`), and the sweep's value arrives as a
        # param — baked in sweep mode, folded from the advance node in feedback. So every
        # iteration would run at the wired number and the table would come out with N
        # identical rows: the exact wrong-looking-right result every refusal here exists for.
        raise ValueError(
            f"{label}.{target.name} is already fed by a wire from {wired.src!r}, and a wired "
            f"value beats a swept one — every iteration would run at that same number and "
            f"the rows would come out identical. Unplug that wire, or iterate a different "
            f"parameter.")
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
            f"{node.id}: no variable points at anything yet — choose a parameter in the "
            f"card's V0 dropdown (it lists every one inside the segment), or drag a "
            f"variable output onto the control itself")
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


def segment_ends(graph: Graph, node_id: str) -> Tuple[str, ...]:
    """The node(s) wired into this card's ``to`` — the segment's END. More than one is a
    refusal in :func:`plan`, not here: the GUI needs to describe the mistake."""
    return tuple(e.src for e in graph.preds(node_id) if e.dst_socket == SEG_TO)


def segment_starts(graph: Graph, node_id: str) -> Tuple[str, ...]:
    """The node(s) wired into this card's ``from`` — the segment's pinned START. Empty is
    the ordinary case: the stretch then begins at whatever params are driven."""
    return tuple(e.src for e in graph.preds(node_id) if e.dst_socket == SEG_FROM)


def is_driving(graph: Graph, node_id: str) -> bool:
    """True when this Iterate node has both halves of a rewrite: at least one driver wire
    AND a segment end to collect at.

    :func:`unroll` skips the ones that do not, rather than refusing. A half-wired Iterate
    turns up constantly in ordinary editing — the moment it is placed, the moment a target
    is muted or deleted, every moment between wiring the driver and wiring the segment —
    and failing the whole run-graph build over a half-finished edit would be hostile.
    Nothing is lost by passing it over: with no rewrite the node keeps no ``__iters__``, so
    its compute takes the not-rewritten branch and refuses at pull time if the sockets
    really do describe a sweep. The dangerous case is still caught; the mid-edit case is not
    punished for it."""
    return (any(e.kind == "driver" and e.src == node_id for e in graph.edges)
            and bool(segment_ends(graph, node_id)))


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


#: the dock states in which the upstream edge is cut — the duplicate of
#: ``nodelab_v2.ops.DOCK_FROZEN`` this module deliberately keeps (see :func:`_is_frozen_dock`).
#: ``held`` is here for the same reason ``docked`` is: it cuts the edge, so an iterated chain
#: containing one would run N identical iterations.
_FROZEN_DOCK_STATES = ("held", "docked")


def _is_frozen_dock(node: NodeInstance) -> bool:
    """Duck-typed check for a dock whose upstream edge is CUT (``held`` or ``docked``).

    Deliberately NOT ``nodelab_v2.ops.is_frozen``: that module is Qt-free but sits *above*
    nodegraph, and importing it here would invert the dependency for the sake of a few string
    constants. The cost of the duplication is that a new frozen state has to be added in both
    places — which is why :data:`_FROZEN_DOCK_STATES` is named rather than inlined, and why
    ``nodelab_v2.ops.DOCK_FROZEN`` names its own copy."""
    return (node.op_key == "io.dock"
            and str((node.modes or {}).get("state") or "live") in _FROZEN_DOCK_STATES)


# ── the target picker (V2.22) ────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetOption:
    """One thing a variable slot may be pointed AT — the offer behind the card's
    "Iterate on" dropdown.

    Scraped, never authored: the active sockets and Modes of the nodes upstream of
    ``collect`` *are* the list, so a node type added to the catalog becomes sweepable the
    day it is registered and a param gated away by a Mode disappears from the menu the
    moment it stops existing. Every offer is put through :func:`_check_target` first, so the
    menu and the refusals cannot drift apart: if it is in the list, wiring it is legal.

    ``driver_id`` is set when something ALREADY drives this target — the slot that owns it
    (so the dropdown can show its current selection) or another card. Reported rather than
    filtered out, because "why is `threshold` not in the list" is a question the list itself
    should answer.
    """

    target: Target
    #: :data:`TYPE_NUMBER` or :data:`TYPE_TEXT` — which of the slot's two driver outputs
    #: this target must be wired from, and therefore what its ``v{k}_type`` has to be set to.
    kind: str
    node_label: str
    param_label: str
    driver_id: str = ""
    driver_slot: int = -1

    @property
    def key(self) -> Tuple[str, str]:
        """``(node_id, socket)`` — the identity a GUI stores on its combo item."""
        return (self.target.node_id, self.target.socket)

    @property
    def text(self) -> str:
        return f"{self.node_label} · {self.param_label}"

    @property
    def detail(self) -> str:
        return f"{self.target.node_id}.{self.target.name}"


def _driver_owners(graph: Graph) -> Dict[Tuple[str, str], Tuple[str, int]]:
    """``(node_id, socket) → (iterate_id, slot)`` for every driver wire in the graph."""
    slot_of = {name: k for k in range(MAX_VARIABLES) for name in var_out_names(k)}
    return {(e.dst, e.dst_socket): (e.src, slot_of.get(e.src_socket, -1))
            for e in graph.edges if e.kind == "driver"}


def candidate_targets(graph: Graph, node_id: str, *,
                      mode: Optional[str] = None) -> Tuple[TargetOption, ...]:
    """Every parameter and Mode this Iterate node could legally drive, in chain order.

    The candidate SET is the SEGMENT: everything upstream of ``to``, narrowed to the
    descendants of ``from`` when one is pinned. That is exactly the set :func:`plan`
    requires a driven node to be in, so the menu cannot offer a target whose own refusal
    ("outside the segment, so the sweep would silently do nothing") is the next thing the
    user would see. With no segment end wired there is no series to scrape and the result is
    empty: wire the segment first, then choose.

    Ordered by :meth:`~nodegraph.graph.Graph.topo_order`, so the menu reads down the series
    from its start to its end rather than in dictionary order.
    """
    node = graph.nodes.get(node_id)
    if node is None or node.op_key != ITERATE_OP:
        return ()
    plan_mode = str(mode or node.state(node.spec()).get("mode") or MODE_SWEEP)
    ends = segment_ends(graph, node_id)
    if not ends:
        return ()
    succ, pred = _forward_maps(graph)
    reach = _closure(pred, ends)
    starts = segment_starts(graph, node_id)
    if starts:
        reach &= _closure(succ, starts)
    reach.discard(node_id)
    try:
        order = [nid for nid in graph.topo_order() if nid in reach]
    except ValueError:                       # a cyclic mid-edit graph still gets a menu
        order = sorted(reach)
    owners = _driver_owners(graph)

    def spec_label(nid: str) -> str:
        n = graph.nodes.get(nid)
        s = n.spec() if n is not None else None
        return (getattr(s, "label", "") or (n.op_key if n is not None else nid))

    # Two nodes of the same type read as one entry twice ("Threshold · threshold" listed
    # under two different cards), so the id disambiguates — but ONLY where it has to, or
    # every line in the menu would carry an id nobody needs to see.
    seen_labels: Dict[str, int] = {}
    for nid in order:
        seen_labels[spec_label(nid)] = seen_labels.get(spec_label(nid), 0) + 1

    out: List[TargetOption] = []
    for nid in order:
        n2 = graph.nodes[nid]
        s2 = n2.spec()
        if s2 is None or n2.op_key in (ITERATE_OP, ADVANCE_OP):
            continue
        st = n2.state(s2)
        label = spec_label(nid)
        nlabel = f"{label} [{nid}]" if seen_labels.get(label, 0) > 1 else label
        offers: List[Tuple[Target, str, str]] = []
        for s in s2.active_inputs(st):
            if can_convert(SocketType.FLOAT, s.type):
                kind = TYPE_NUMBER
            elif can_convert(SocketType.STRING, s.type):
                kind = TYPE_TEXT
            else:                            # DATASET, MENU — nothing a variable can carry
                continue
            offers.append((Target(nid, s.name), kind, s.label or s.name))
        for m in s2.active_modes(st):
            offers.append((Target(nid, m.name, is_mode=True), TYPE_TEXT,
                           m.label or m.name))
        for target, kind, plabel in offers:
            try:
                _check_target(graph, plan_mode, target)
            except ValueError:               # the refusal IS the filter (see the docstring)
                continue
            owner, slot = owners.get((nid, target.socket), ("", -1))
            out.append(TargetOption(target, kind, nlabel, plabel, owner, slot))
    return tuple(out)


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

    ends = segment_ends(graph, node_id)
    starts = segment_starts(graph, node_id)
    if not ends:
        raise ValueError(
            f"{node_id}: the segment has no END — wire the LAST node of the series you want "
            f"iterated into this card's 'to' input. That node is where the iterated result "
            f"comes out, so everything already reading it keeps working.")
    if len(ends) > 1:
        raise ValueError(
            f"{node_id}: the segment's 'to' has {len(ends)} wires ({', '.join(ends)}), and a "
            f"series has one end. Keep the LAST node; a branch that must also see the "
            f"iteration should be moved below it.")
    end = ends[0]

    variables = _resolve_variables(graph, node, spec, state, envs, mode)

    # ── the cone: driven ∪ descendants(driven), ∩ ancestors(to), ∩ descendants(from) ──
    succ, pred = _forward_maps(graph)
    driven = {t.node_id for v in variables for t in v.targets}
    downstream = _closure(succ, driven)
    upstream = _closure(pred, [end])
    cone = downstream & upstream
    if starts:
        # A pinned start does two things at once: it bounds the clone set, and it declares
        # everything ABOVE it loop-invariant. Both fall out of the intersection — a node
        # upstream of `from` simply is not in the cone, so it is computed once and feeds
        # every iteration.
        cone &= _closure(succ, starts)
    cone = frozenset(cone)

    stranded = sorted(n for n in driven if n not in cone)
    if stranded:
        above = sorted(n for n in stranded if starts and n not in _closure(succ, starts))
        raise ValueError(
            f"{node_id}: driven node(s) {stranded} are outside the segment, so cloning them "
            f"would change nothing and the sweep would silently do nothing. "
            + (f"They sit ABOVE 'from' ({', '.join(starts)}) — move the segment start up to "
               f"cover them, or drive a parameter inside it."
               if above else
               f"Wire 'to' to a node BELOW them (the segment's end is {end!r})."))
    if end not in cone:
        raise ValueError(
            f"{node_id}: nothing inside the segment is driven — {end!r} is not downstream of "
            f"any parameter this card iterates, so every iteration would produce the same "
            f"result. Point a variable at a parameter inside the segment.")

    docked = sorted(n for n in cone if _is_frozen_dock(graph.nodes[n]))
    if docked:
        raise ValueError(
            f"{node_id}: the iterated chain contains frozen Dock node(s) {docked}. A held or "
            f"docked dock serves a frozen result and its upstream is cut, so every iteration "
            f"would read the same pixels and produce identical results. Set it to 'live', "
            f"or move the Iterate node downstream of it.")

    nested = sorted(n for n in cone if graph.nodes[n].op_key == ITERATE_OP)
    if nested:
        raise ValueError(
            f"{node_id}: Iterate node(s) {nested} sit inside this one's iterated chain. "
            f"Nested iteration is not supported — sweep several parameters from ONE Iterate "
            f"card instead (raise 'Variables' and set Combine to 'grid').")

    # A branch leaving the segment's END is the intended exit — the selector minted under
    # that node's id serves it the preserved iteration, so it needs no re-routing at all.
    # A branch leaving any OTHER node in the segment is the real hazard: that node exists
    # only as clones afterwards, so its outside consumer would have no single iteration to
    # read. An edge into ANY Iterate card's segment sockets is a reference, not a branch.
    def _is_segment_ref(e: Edge) -> bool:
        dst = graph.nodes.get(e.dst)
        return (dst is not None and dst.op_key == ITERATE_OP
                and e.dst_socket in (SEG_FROM, SEG_TO))

    escapes = sorted({(e.src, e.dst) for e in graph.edges
                      if is_dag_edge(e) and e.src in cone and e.src != end
                      and e.dst not in cone and not _is_segment_ref(e)})
    if escapes:
        src, dst = escapes[0]
        raise ValueError(
            f"{node_id}: {src!r} is inside the segment but also feeds {dst!r} outside it, "
            f"which has no single iteration to read. Move the segment's end ('to') down to "
            f"{src!r} or below, so that branch leaves from the end instead."
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
    picked = max(0, min(len(iterations) - 1,
                        int(_as_float(_param(spec, node.params, "index", 0), 0))))
    if sweep_all or mode == MODE_FEEDBACK or preserve == PRESERVE_BEST:
        minted = tuple(range(len(iterations)))
    elif preserve == PRESERVE_FIRST:
        minted = (0,)
    elif preserve == PRESERVE_LAST:
        minted = (len(iterations) - 1,)
    else:
        minted = (picked,)

    return IteratePlan(
        node_id=node_id, mode=mode, preserve=preserve, combine=combine, search=search,
        direction=direction, metric=metric,
        tol=_as_float(_param(spec, node.params, "tol", 0.0), 0.0),
        target_value=_as_float(_param(spec, node.params, "target", 0.0), 0.0),
        variables=variables, iterations=iterations, cone=cone,
        end=end, start=tuple(starts), minted=minted, picked=picked)


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
    """Clone the segment, then mint the selector **under the end node's own id**.

    That last step is the whole exit strategy. Everything wired to the end node — a Viewer,
    an Export, three branches drawn months before the Iterate card existed — keeps its edge
    and starts reading the preserved iteration, because the id it points at is still there
    and is now the thing that chooses. Nothing downstream is re-routed and nothing downstream
    knows an iteration happened."""
    cone, nid, end = p.cone, p.node_id, p.end
    # snapshot the originals BEFORE the cone leaves the working dict — the clones are built
    # from them, and with two Iterate plans in one graph the second pass would otherwise
    # look for nodes the first has already replaced.
    originals = {c: nodes[c] for c in cone}
    kept: List[Edge] = []
    for e in edges:
        if e.kind == "driver" and e.src == nid:
            continue                                   # consumed: baked or re-wired below
        if e.src == end:
            # Every edge leaving the end node is kept VERBATIM: its id now belongs to the
            # selector, so each of these — including the card's own `to` wire, which is how
            # the card gets the selected payload to pass through — reads the preserved
            # iteration without being touched.
            kept.append(e)
            continue
        if e.src in cone or e.dst in cone:
            continue                                   # re-emitted per iteration
        kept.append(e)
    for c in cone:
        nodes.pop(c, None)

    inner = [e for e in edges if is_dag_edge(e) and e.src in cone and e.dst in cone]
    incoming = [e for e in edges if is_dag_edge(e) and e.src not in cone and e.dst in cone]

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
        # …and every iteration's END feeds the selector, in iteration order.
        kept.append(Edge(iter_id(end, nid, i), end, "out", SEG_TO))

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
                kept.append(Edge(iter_id(end, nid, j), aid, "out", "probes"))
            for t in var.targets:
                kept.append(Edge(aid, iter_id(t.node_id, nid, i), "value", t.name))

    card = nodes[nid]
    # The SELECTOR: the card's own settings (preserve / metric / index / modes) under the end
    # node's id, plus the per-iteration table only the rewrite knows.
    nodes[end] = NodeInstance(
        end, ITERATE_OP,
        params={**card.params, ITERS_KEY: [
            {"index": i,
             "values": [None if p.mode == MODE_FEEDBACK else p.iterations[i].values[j]
                        for j in range(len(p.variables))]}
            for i in p.minted],
            "__labels__": [v.label for v in p.variables],
            OWNER_KEY: nid},
        modes=dict(card.modes))
    # …and the card itself becomes a pass-through of what the selector hands back, so the
    # two never disagree about which iteration won or what its values were.
    nodes[nid] = NodeInstance(nid, card.op_key,
                              params={**card.params, PASSTHROUGH_KEY: True},
                              modes=dict(card.modes))
    return nodes, kept


def aliases(graph: Graph, *, envs: Optional[Mapping[str, Any]] = None,
            sweep_all: Iterable[str] = ()) -> Dict[str, str]:
    """``original id → the clone a viewer should show`` for every node INSIDE a segment.

    A cone node is replaced by its clones, so its own id is gone from the run graph and a
    GUI that pulls "the node I clicked" would get a KeyError. This says which clone to pull
    instead — :attr:`IteratePlan.view_index`'s, i.e. the one the iteration strip is on — so
    a node being tuned can be viewed at the iteration under discussion. The segment's END is
    deliberately NOT aliased: its id still exists and belongs to the selector.

    Never raises: an unplannable card contributes nothing, exactly as it does to
    :func:`unroll`, because this runs on every selection change in the GUI."""
    out: Dict[str, str] = {}
    all_sweep = set(sweep_all)
    for nid in iterate_nodes(graph):
        if not is_driving(graph, nid):
            continue
        try:
            p = plan(graph, nid, envs=envs, sweep_all=(nid in all_sweep))
        except (ValueError, KeyError):
            continue
        for c in p.cone:
            if c != p.end:
                out[c] = iter_id(c, nid, p.view_index)
    return out


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
                    f"the segments iterated by {plans[a].node_id} and {plans[b].node_id} "
                    f"overlap at {shared} — a node cannot be cloned by two sweeps at once "
                    f"(which iteration of the first would each iteration of the second "
                    f"read?). Sweep both parameters from ONE Iterate card instead: raise "
                    f"'Variables' and set Combine to 'grid'.")
    # Two Iterate nodes cannot share a cone node (the overlap refusal above sees to that),
    # so the plans compose by simple sequential application.
    nodes: Dict[str, NodeInstance] = dict(graph.nodes)
    edges: List[Edge] = list(graph.edges)
    for p in plans:
        nodes, edges = _apply(p, nodes, edges)
    return Graph(nodes, edges)


__all__ = [
    "ITERATE_OP", "ADVANCE_OP", "MODE_TARGET_PREFIX", "MAX_VARIABLES", "MAX_ITERATIONS",
    "ITERS_KEY", "SWEEP_KEY", "SWEEP_ROWS_KEY", "SWEEP_LABELS_KEY", "SWEEP_OWNER_KEY",
    "SEG_FROM", "SEG_TO", "PASSTHROUGH_KEY", "OWNER_KEY",
    "segment_ends", "segment_starts", "aliases",
    "MODE_SWEEP", "MODE_FEEDBACK", "is_driving",
    "PRESERVE_FIRST", "PRESERVE_LAST", "PRESERVE_BEST", "PRESERVE_PICKED",
    "COMBINE_ZIP", "COMBINE_GRID", "SRC_LIST", "SRC_LINEAR", "SRC_LOG", "SRC_AROUND",
    "TYPE_NUMBER", "TYPE_TEXT", "SEARCH_GOLDEN", "SEARCH_SECANT",
    "Target", "TargetOption", "Variable", "Iteration", "IteratePlan",
    "var_mode_names", "var_out_names", "var_field_names",
    "iter_id", "advance_id", "iterate_nodes", "parse_list", "span", "around",
    "advance_value", "candidate_targets", "plan", "unroll",
]
