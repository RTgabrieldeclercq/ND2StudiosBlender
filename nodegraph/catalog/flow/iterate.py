"""Iterate (``flow.iterate``) — Run the chain in front of it once per parameter value and keep one result."""

from __future__ import annotations

import math
import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.iterate import MAX_VARIABLES as _ITERATE_MAX_VARIABLES
from nodegraph.registry import (
    Granularity,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
    OutValue,
    SocketSpec,
)
from nodegraph.sockets import SocketType

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.global_scalar import _global_scalar

# ── flow.iterate — sweep / search a parameter (V2.19) ───────────────────────────
#
# The node is deliberately thin, exactly like `io.dock`: all its compute does is choose
# which collected result to hand on and stamp what won. Everything that makes iteration
# happen is the graph rewrite in `nodegraph.iterate` — the cone is cloned there, the values
# are baked there, and the driver wires never reach the engine. The node is the control
# panel; the rewrite is the mechanism.


def _iter_slot_values(ctx: EvalContext, s_list: str, s_start: str, s_stop: str,
                      s_center: str, s_percent: str, s_steps: str) -> Dict[str, Any]:
    """One variable slot's value-source fields, read by literal socket name.

    Spelled out per slot (rather than an f-string loop) on purpose: the socket-contract
    guard resolves param reads through helper forwarding but only from *literal* keys, so a
    computed name would make all 24 of these read as dead controls."""
    return {
        "list": ctx.params.get(s_list, ""),
        "start": ctx.params.get(s_start, 0.0),
        "stop": ctx.params.get(s_stop, 1.0),
        "center": ctx.params.get(s_center, None),
        "percent": ctx.params.get(s_percent, 50.0),
        "steps": ctx.params.get(s_steps, 5),
    }
# One reader per slot. Four near-identical `def`s rather than a lambda table or an
# f-string loop, for one reason: the socket-contract guard resolves param reads by walking
# the AST of `nodegraph.nodes` for FunctionDefs and following LITERAL keys through helper
# calls. A lambda is not a FunctionDef and a computed name is not a literal, so either
# shortcut would make all 24 of these sockets read as dead controls the kernel ignores —
# and the guard would be telling the truth about the evidence it can see.

def _iter_slot0(ctx: EvalContext) -> Dict[str, Any]:
    return _iter_slot_values(ctx, "v0_list", "v0_start", "v0_stop",
                             "v0_center", "v0_percent", "v0_steps")
def _iter_slot1(ctx: EvalContext) -> Dict[str, Any]:
    return _iter_slot_values(ctx, "v1_list", "v1_start", "v1_stop",
                             "v1_center", "v1_percent", "v1_steps")
def _iter_slot2(ctx: EvalContext) -> Dict[str, Any]:
    return _iter_slot_values(ctx, "v2_list", "v2_start", "v2_stop",
                             "v2_center", "v2_percent", "v2_steps")
def _iter_slot3(ctx: EvalContext) -> Dict[str, Any]:
    return _iter_slot_values(ctx, "v3_list", "v3_start", "v3_stop",
                             "v3_center", "v3_percent", "v3_steps")
def _iter_slot_fields(ctx: EvalContext, slot: int) -> Dict[str, Any]:
    if slot == 0:
        return _iter_slot0(ctx)
    if slot == 1:
        return _iter_slot1(ctx)
    if slot == 2:
        return _iter_slot2(ctx)
    return _iter_slot3(ctx)
def _iterate_values(ctx: EvalContext, slot: int, kind: str, source: str) -> Tuple[Any, ...]:
    """This slot's value list resolved from the node's own sockets — the fallback for a
    graph that was never rewritten (a hand-built headless graph wiring one chain straight
    into ``collect``). The rewritten path reads ``__iters__`` instead, which is the only
    form that can carry an ``around`` centre resolved against the target's derive."""
    from nodegraph.iterate import (
        SRC_LINEAR, SRC_LIST, SRC_LOG, TYPE_NUMBER, around, parse_list, span)
    fields = _iter_slot_fields(ctx, slot)
    if source == SRC_LIST:
        return parse_list(fields["list"], kind == TYPE_NUMBER)
    if source in (SRC_LINEAR, SRC_LOG):
        return span(float(fields["start"] or 0.0), float(fields["stop"] or 1.0),
                    int(fields["steps"] or 1), log=(source == SRC_LOG))
    centre = fields["center"]
    if centre is None:
        return ()
    return around(float(centre), float(fields["percent"] or 0.0), int(fields["steps"] or 1))
def _compute_iterate(ctx: EvalContext) -> Dataset:
    """Choose which iteration's result to preserve, and stamp what won.

    **Resolved spec.** Category ``flow``; ``collect`` is a ``multi`` Dataset input carrying
    one payload per minted iteration (the rewrite appends them in iteration order, and the
    engine walks preds in canonical socket order, so the order is stable). Output is one of
    those payloads plus ``Global`` ``sweep_<var>`` scalars for the values that produced it.
    No 2D/3D lever and no axis change of its own: it selects among results, it does not
    touch pixels — ``TILEABLE``, no kernel axes.

    **It degrades to identity.** Pull a ``flow.iterate`` node in a graph nobody rewrote and
    it returns its single collected input, resolving what it can from its own sockets. That
    matters because the rewrite lives in the *materialize* path: a hand-built headless graph
    that skipped it must behave sanely rather than fail obscurely.
    """
    from nodegraph.iterate import (
        ITERS_KEY, MAX_VARIABLES, MODE_FEEDBACK, PRESERVE_BEST, PRESERVE_FIRST,
        PRESERVE_LAST, SEARCH_SECANT, SRC_LIST, SWEEP_LABELS_KEY, SWEEP_ROWS_KEY,
        TYPE_NUMBER, var_mode_names,
    )
    collected = ctx.input("collect")
    if not collected:
        raise ValueError(
            "Iterate: nothing is wired into 'collect' — connect the END of the chain you "
            "are iterating back into it, so the sweep has results to compare")
    if not isinstance(collected, tuple):
        collected = (collected,)
    modes = ctx.params.get("__modes__", {})
    mode = str(modes.get("mode", "sweep"))
    # Feedback IS an optimization, so its answer is the best probe. Forcing it here (rather
    # than offering a preserve the search would ignore) is what lets `metric`/`direction`
    # gate cleanly on preserve==best: `available_in` is a conjunction and cannot express
    # "live under best OR under feedback".
    preserve = PRESERVE_BEST if mode == MODE_FEEDBACK else str(
        modes.get("preserve", "picked"))
    metric_name = ctx.layer("metric")
    minted = [int(r.get("index", i)) for i, r in
              enumerate(ctx.params.get(ITERS_KEY) or [])]
    if len(minted) != len(collected):
        minted = list(range(len(collected)))

    scores: List[Optional[float]] = [_global_scalar(ds, metric_name) for ds in collected]
    if preserve == PRESERVE_BEST:
        # NaN is excluded, not compared: an averaging reducer over an iteration that
        # detected nothing returns NaN, and NaN silently wins or loses every comparison
        # depending on argument order. An all-NaN sweep is the same "nothing to compare"
        # case as a missing metric and gets the same message.
        usable = [(s, k) for k, s in enumerate(scores)
                  if s is not None and not math.isnan(s)]
        if not usable:
            raise ValueError(
                f"Iterate: preserve is 'best' but no iteration carries a comparable Global "
                f"scalar {metric_name!r} — put a 'Reduce → Scalar' node inside the iterated "
                f"chain that writes it, and name it here. (An all-NaN metric means every "
                f"iteration found nothing to measure.)")
        if mode == MODE_FEEDBACK and str(modes.get("search", "golden")) == SEARCH_SECANT:
            # secant drives the metric TO a target, so "best" is CLOSEST — extremizing here
            # would hand back the probe furthest from what the user asked for, having
            # searched correctly the whole way. `direction` is Mode-gated away in this
            # state for the same reason: it would be a live control nothing reads.
            aim = float(ctx.params.get("target", 0.0) or 0.0)
            pick = min(usable, key=lambda sk: abs(sk[0] - aim))[1]
        else:
            pick = (max(usable)[1] if str(modes.get("direction", "max")) == "max"
                    else min(usable)[1])
    elif preserve == PRESERVE_FIRST:
        pick = minted.index(min(minted))
    elif preserve == PRESERVE_LAST:
        pick = minted.index(max(minted))
    else:
        want = int(ctx.params.get("index", 0) or 0)
        pick = minted.index(want) if want in minted else 0

    out = collected[pick]
    # ── stamp what won ─────────────────────────────────────────────────────────
    labels = list(ctx.params.get("__labels__") or [])
    rows = ctx.params.get(ITERS_KEY) or []
    values: Sequence[Any] = (rows[pick].get("values", ()) if pick < len(rows) else ())
    if not rows:
        # NOT REWRITTEN. The node is being pulled on a graph the iterate rewrite never
        # touched, so nothing was iterated — the chain ran once, at whatever the target
        # node's own parameter says. Refuse when the sockets describe a real sweep, because
        # a silent single run is exactly the "looks like it worked" failure every refusal in
        # `nodegraph.iterate` exists to prevent. And stamp NOTHING: with no rewrite there is
        # no iteration whose value this payload can honestly be labelled with.
        n_vars = max(1, min(MAX_VARIABLES, int(modes.get("variables", "1") or 1)))
        planned = 1
        for k in range(n_vars):
            type_mode, source_mode = var_mode_names(k)
            planned *= max(1, len(_iterate_values(
                ctx, k, str(modes.get(type_mode) or TYPE_NUMBER),
                str(modes.get(source_mode) or SRC_LIST))))
        if planned > 1 or len(collected) > 1:
            raise ValueError(
                f"Iterate: this node describes {planned} iterations but nothing was "
                f"iterated. Either no variable output is wired to a parameter (drag one "
                f"onto the control you want to sweep — the target may also have been muted "
                f"or deleted), or the graph reached the engine without the iterate rewrite "
                f"(headlessly, build the run graph through nodegraph.iterate.unroll, or "
                f"nodelab_v2.ops.headless_engine which does it for you).")
        return collected[0]
    for label, value in zip(labels, values):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out = out.with_layer(Domain.GLOBAL, f"sweep_{label}",
                                 np.asarray(float(value)))
    # The results TABLE rides the payload's metadata (a namespaced non-calibration key, so
    # `_StrictCalibMetadata` passes it through and it never collides with a calibration
    # name). This compute is the only place that holds every iteration's result at once, so
    # it is the only place that can build the table without a second pull — and the GUI then
    # gets it from the payload it is already handed, with no extra plumbing.
    table = [{"iter": minted[k] if k < len(minted) else k,
              "metric": scores[k],
              "values": list(rows[k].get("values", ())) if k < len(rows) else [],
              "won": k == pick}
             for k in range(len(collected))]
    out = out.with_metadata(**{SWEEP_ROWS_KEY: table, SWEEP_LABELS_KEY: list(labels)})
    if scores[pick] is not None:
        out = out.with_layer(Domain.GLOBAL, "sweep_metric",
                             np.asarray(float(scores[pick])))
        # secant drives the metric TO a value, so the useful report is the residual and
        # whether it landed inside the tolerance the user asked for.
        if mode == MODE_FEEDBACK and str(modes.get("search", "golden")) == SEARCH_SECANT:
            residual = scores[pick] - float(ctx.params.get("target", 0.0) or 0.0)
            out = out.with_layer(Domain.GLOBAL, "sweep_residual",
                                 np.asarray(float(residual)))
            out = out.with_layer(
                Domain.GLOBAL, "sweep_converged",
                np.asarray(float(abs(residual) <= float(ctx.params.get("tol", 0.0) or 0.0))))
    return out
#: Mirrors :data:`nodegraph.iterate.MAX_VARIABLES` — the two must agree or the rewrite
#: would look for driver outputs this module never declared. Checked at import.
MAX_VARS = 4
def _iterate_slot_sockets(k: int) -> Tuple[List[SocketSpec], List[SocketSpec], List[Any]]:
    """``(inputs, outputs, modes)`` for one variable slot — declared, gated, documented.

    Slot ``k`` is live when the ``variables`` Mode is at least ``k+1``, so the gate is the
    set of counts that include it. Its driver port is a FLOAT/STRING **pair** gated by the
    slot's own ``v{k}_type``, because ``STRING`` has no implicit conversion (``sockets.
    _CONVERSIONS``) and one port therefore cannot serve both a σ and a method name."""
    from nodegraph.iterate import (
        SRC_AROUND, SRC_LINEAR, SRC_LIST, SRC_LOG, TYPE_NUMBER, TYPE_TEXT,
        var_field_names, var_mode_names, var_out_names,
    )
    live = frozenset(str(v) for v in range(k + 1, MAX_VARS + 1))
    on = {"variables": live}
    type_mode, source_mode = var_mode_names(k)
    num_out, txt_out = var_out_names(k)
    f = var_field_names(k)
    ranged = frozenset({SRC_LINEAR, SRC_LOG})
    nth = ("first", "second", "third", "fourth")[k]
    inputs = [
        InString(f["list"], f"V{k} values", field=False, default="",
                 available_in={**on, source_mode: frozenset({SRC_LIST})},
                 description=
                 f"The {nth} variable's values, comma separated — '0.3, 0.5, 0.8' for a "
                 f"number, 'otsu, li, yen' for a method name. This is the only source that "
                 f"can sweep text, and the only one that lets the values be uneven. A token "
                 f"that is not a number is refused rather than skipped, so a typo shortens "
                 f"nothing silently."),
        InFloat(f["start"], f"V{k} from", unit="", field=False, default=0.0,
                available_in={**on, source_mode: ranged},
                description=
                f"Start of the {nth} variable's range, in the TARGET parameter's own units "
                f"— whatever the socket you wired it into uses. Under 'log' spacing both "
                f"ends must be above zero."),
        InFloat(f["stop"], f"V{k} to", unit="", field=False, default=1.0,
                available_in={**on, source_mode: ranged},
                description=
                f"End of the {nth} variable's range, inclusive: the last iteration sits "
                f"exactly here, not one step short of it."),
        InFloat(f["center"], f"V{k} centre", unit="", field=False, default=None,
                available_in={**on, source_mode: frozenset({SRC_AROUND})},
                description=
                f"Centre of the {nth} variable's ± band. LEAVE IT EMPTY to anchor on the "
                f"target parameter's own current value — its override if you set one, "
                f"otherwise the value its metadata rule derives from this file's optics. "
                f"That is what keeps the sweep meaningful when you open a different file; a "
                f"typed centre pins it to the numbers you authored against."),
        InFloat(f["percent"], f"V{k} ±%", unit="", field=False, default=50.0,
                available_in={**on, source_mode: frozenset({SRC_AROUND})},
                description=
                f"How far either side of the centre the {nth} variable spans, as a "
                f"percentage. 50 means half to one-and-a-half times the centre. Widen it "
                f"when the sweep's best value sits at an end of the band — that is the sign "
                f"the answer is outside it."),
        InInt(f["steps"], f"V{k} steps", unit="", field=False, default=5,
              available_in={**on, source_mode: frozenset({SRC_LINEAR, SRC_LOG, SRC_AROUND})},
              description=
              f"How many values the {nth} variable's range produces — and, in feedback "
              f"mode, how many probes the search is allowed. Under 'grid' these MULTIPLY "
              f"across variables, so 5 and 5 is 25 runs of the whole chain, not 10."),
    ]
    outputs = [
        OutValue(num_out, SocketType.FLOAT, f"V{k}",
                 available_in={**on, type_mode: frozenset({TYPE_NUMBER})}),
        OutValue(txt_out, SocketType.STRING, f"V{k}",
                 available_in={**on, type_mode: frozenset({TYPE_TEXT})}),
    ]
    modes = [
        Mode(type_mode, [TYPE_NUMBER, TYPE_TEXT], default=TYPE_NUMBER,
             label=f"V{k} type", available_in=dict(on),
             description=
             f"What KIND of value the {nth} variable produces, which selects which of its two "
             f"driver outputs exists — a wire carries one type and STRING converts to nothing, "
             f"so one port cannot serve both. Set it before dragging the output onto a "
             f"parameter.",
             choice_docs={
                 TYPE_NUMBER:
                     "A float driver output, for any numeric parameter — a threshold, a radius, "
                     "a σ. The only type that can be swept as a range or searched in feedback "
                     "mode, since both need arithmetic on the values.",
                 TYPE_TEXT:
                     "A string driver output, for a name or a Mode value — 'otsu, li, yen', a "
                     "layer name, a model checkpoint. Only the comma-separated list source can "
                     "produce it, and feedback search cannot use it because there is nothing "
                     "to interpolate between two words.",
             }),
        Mode(source_mode, [SRC_LIST, SRC_LINEAR, SRC_LOG, SRC_AROUND], default=SRC_LIST,
             label=f"V{k} from", available_in=dict(on),
             description=
             f"Where the {nth} variable's values come FROM, which selects the fields below it. "
             f"The three range sources also define the BRACKET a feedback search hunts inside; "
             f"a plain list has no bracket and cannot be searched.",
             choice_docs={
                 SRC_LIST:
                     "Values typed out, comma separated. The only source that can sweep TEXT, "
                     "and the only one that allows uneven spacing — so it is the one to use for "
                     "a handful of specific values you care about. A token that is not a number "
                     "is refused rather than skipped.",
                 SRC_LINEAR:
                     "Evenly spaced values from `from` to `to` inclusive. The default shape of a "
                     "parameter scan, and the right one when you have no reason to think the "
                     "parameter acts multiplicatively.",
                 SRC_LOG:
                     "Values spaced evenly in the LOGARITHM, so each step multiplies rather than "
                     "adds. Use it when the interesting range spans orders of magnitude (a σ "
                     "from 0.1 to 100) — a linear scan there spends nearly every step at the "
                     "large end. Both endpoints must be above zero.",
                 SRC_AROUND:
                     "A ± band around a centre, as a percentage. Leave the centre empty and it "
                     "anchors on the TARGET parameter's own value — its override, else whatever "
                     "its metadata rule derives from this file's optics — which is what keeps a "
                     "saved sweep meaningful when you open a different file.",
             }),
    ]
    return inputs, outputs, modes
assert MAX_VARS == _ITERATE_MAX_VARIABLES, (
    "nodes.MAX_VARS and iterate.MAX_VARIABLES disagree — the rewrite resolves driver "
    "outputs by name, so a slot declared in one and not the other is a silently dead wire")
_ITER_IN: List[SocketSpec] = []
_ITER_OUT: List[SocketSpec] = []
_ITER_MODES: List[Any] = []
for _k in range(MAX_VARS):
    _i, _o, _m = _iterate_slot_sockets(_k)
    _ITER_IN += _i
    _ITER_OUT += _o
    _ITER_MODES += _m
register_node(
    _compute_iterate, op_key="flow.iterate", label="Iterate", category="flow",
    adds_domains=frozenset({Domain.GLOBAL}),
    # The metric is a Global scalar, and it is needed under `preserve=best` OR under
    # `mode=feedback` — feedback IS an optimization, so the compute forces preserve to
    # `best` there whatever the dropdown says. That disjunction is exactly why the field is
    # a UNION over modes: two entries, either one sufficient. It is also why the `metric`
    # socket's own `available_in` is approximate (a conjunction cannot say "best OR
    # feedback"), so the rail is stricter than the gate here on purpose.
    #
    # Read off the COLLECT wire, not off a primary input — this node has no other Dataset
    # input, so `input_domains` unions exactly the right edge.
    reads_domains_by_mode={
        "mode": {"feedback": frozenset({Domain.GLOBAL}), "sweep": frozenset()},
        "preserve": {"best": frozenset({Domain.GLOBAL})},
    },
    inputs=[
        InDataset("collect", multi=True, label="Collect"),
        InString("metric", "Metric", field=False, default="",
                 layer_in=Domain.GLOBAL,
                 available_in={"preserve": frozenset({"best"})},
                 description=
                 "The Global scalar each iteration is judged by — the name a "
                 "'Reduce → Scalar' node inside the iterated chain wrote. Only read when "
                 "Preserve is 'best' (and in feedback mode, which is always 'best'); the "
                 "picker offers exactly the Global scalars present on the Collect wire."),
        InInt("index", "Keep iteration", unit="", field=False, default=0,
              available_in={"preserve": frozenset({"picked"})},
              description=
              "Which iteration's result flows downstream, counting from 0. Ctrl-click the "
              "Viewer's iteration strip to set it by eye instead of typing. This is the "
              "ONLY preserve mode that costs one run instead of N: the rewrite mints just "
              "this iteration, so a finished graph stops paying for the sweep."),
        InFloat("target", "Target value", unit="", field=False, default=0.0,
                available_in={"mode": frozenset({"feedback"}),
                              "search": frozenset({"secant"})},
                description=
                "The metric value the secant search is driving TO — 500 to find the "
                "threshold that yields 500 objects. The search moves the parameter until "
                "the metric reaches this, rather than maximizing anything."),
        InFloat("tol", "Tolerance", unit="", field=False, default=0.0,
                available_in={"mode": frozenset({"feedback"}),
                              "search": frozenset({"secant"})},
                description=
                "How close to Target counts as a hit. It does NOT stop the search early — "
                "every probe always runs — it decides the sweep_converged flag the node "
                "stamps, so you can tell a search that landed from one that ran out of "
                "probes still moving."),
        *_ITER_IN,
    ],
    outputs=[OutDataset(), *_ITER_OUT],
    modes=[
        Mode("mode", ["sweep", "feedback"], default="sweep", label="Mode",
             description=
             "Whether every value is run, or the next value is CHOSEN from the last result's "
             "score. This is the node's biggest fork: it decides how many times the chain runs, "
             "and whether you end up with a set of results to compare or one converged answer.",
             choice_docs={
                 "sweep":
                     "Run the chain once per value and keep them all, so you can flip through "
                     "them on the card and compare. Exhaustive and predictable — the run count "
                     "is known before you start — and the only mode that can drive a Mode "
                     "dropdown or a text value.",
                 "feedback":
                     "Search for the value that optimizes the metric: each probe's Global scalar "
                     "decides where the next probe goes, and the search stops when the bracket "
                     "closes or the step budget runs out. Far fewer runs than a fine sweep, one "
                     "numeric variable only, and it needs a range source to give it a bracket.",
             }),
        Mode("variables", [str(i) for i in range(1, MAX_VARS + 1)], default="1",
             label="Variables",
             description=
             "How many parameters are iterated at once. Each one you enable adds a driver output "
             "to drag onto a parameter plus its own value-source fields; the slots above the "
             "count are hidden. Under `grid` the run count MULTIPLIES across variables, so this "
             "is also the fastest way to make a sweep enormous.",
             choice_docs={
                 "1": "One driver output. The normal case, and the only count feedback search "
                      "accepts — golden-section and secant both work on a single bracket.",
                 "2": "Two driver outputs, for a pair of parameters that interact — a threshold "
                      "and a minimum size, a radius and a sensitivity. 5 × 5 under `grid` is 25 "
                      "runs of the whole chain.",
                 "3": "Three driver outputs. Only practical with few values each or with `zip`: "
                      "a grid of 5 × 5 × 5 is 125 full runs of everything upstream of Collect.",
                 "4": "Four driver outputs — the maximum. Realistically for `zip`, where four "
                      "variables still mean one run per index; a four-way grid is a combinatorial "
                      "trap unless each list is tiny.",
             }),
        Mode("combine", ["grid", "zip"], default="grid", label="Combine",
             available_in={"mode": frozenset({"sweep"})},
             description=
             "How several variables' values are paired up into iterations. With one variable it "
             "makes no difference; with two or more it is the difference between exploring a "
             "space and walking a path through it — and between a run count that multiplies and "
             "one that does not.",
             choice_docs={
                 "grid":
                     "Every combination of every variable's values — the full Cartesian product. "
                     "The thorough option and the default; the run count is the PRODUCT of the "
                     "value counts, so a pair of 5-value ranges is 25 runs of the whole chain.",
                 "zip":
                     "The i-th value of each variable together, so N variables still give N "
                     "values worth of runs. For parameters that must move TOGETHER — a radius "
                     "and its matching σ, a low/high threshold pair — and it requires every "
                     "variable to have the same number of values.",
             }),
        Mode("preserve", ["picked", "first", "last", "best"], default="picked",
             label="Preserve", available_in={"mode": frozenset({"sweep"})},
             description=
             "Which single iteration's result leaves this node's output. Every iteration is still "
             "computed and still browsable on the card — this only chooses what flows DOWNSTREAM, "
             "so it is what the rest of the graph and any export sees.",
             choice_docs={
                 "picked":
                     "Whichever iteration you are currently looking at on the card. The default, "
                     "and the one to use while exploring: stepping through the results changes "
                     "what the downstream nodes show, so the whole chain follows your eye.",
                 "first":
                     "Always the first iteration, whatever you are viewing. A stable, "
                     "browsing-independent output — useful when the sweep exists to document "
                     "alternatives while the pipeline itself must keep using one fixed setting.",
                 "last":
                     "Always the final iteration — the natural choice when the values are ordered "
                     "so that the last one is the intended configuration, e.g. a coarse-to-fine "
                     "progression.",
                 "best":
                     "The iteration whose Global metric scalar is highest or lowest (see Best "
                     "is). This is what turns a sweep into an optimization; it needs a Reduce → "
                     "Scalar node inside the iterated chain producing the named metric, and "
                     "iterations whose metric is NaN are excluded rather than compared.",
             }),
        Mode("search", ["golden", "secant"], default="golden", label="Search",
             available_in={"mode": frozenset({"feedback"})},
             description=
             "Which search rule picks the next probe in feedback mode, from every (value, metric) "
             "pair measured so far. The two answer different questions — find an extremum, or hit "
             "a number — and that decides whether Best is / Target is the live control.",
             choice_docs={
                 "golden":
                     "Golden-section search for an EXTREMUM inside the bracket: each probe shrinks "
                     "the interval by a fixed ratio, needing no derivative and no assumption "
                     "beyond a single peak. Robust and steady — about 4 probes to halve the "
                     "bracket — and it will happily converge on a LOCAL optimum if the metric has "
                     "several.",
                 "secant":
                     "Drive the metric to a TARGET value by the secant step, i.e. straight-line "
                     "extrapolation through the last two probes, clamped into the bracket and "
                     "falling back to bisection when the step misbehaves. Much faster than "
                     "bisection on a smooth monotone metric, and it aims at a level rather than "
                     "an extremum, so it reads no direction at all.",
             }),
        # Shown for sweep+best (where `search` is hidden but still resolves to its default
        # 'golden') and for feedback+golden. Hidden under feedback+secant, which aims at a
        # target rather than an extremum and reads no direction at all.
        Mode("direction", ["max", "min"], default="max", label="Best is",
             available_in={"preserve": frozenset({"best"}),
                           "search": frozenset({"golden"})},
             description=
             "Which end of the metric counts as best — the sign convention for the comparison, "
             "and nothing else. Read only where an extremum is being chosen (preserve `best`, or "
             "a golden-section feedback search); a secant search aims at a target level and needs "
             "no direction.",
             choice_docs={
                 "max":
                     "Highest metric wins. Right for metrics that count or score something "
                     "desirable — objects found, mean area, a correlation quality — and the "
                     "default.",
                 "min":
                     "Lowest metric wins. Right for anything that measures ERROR or excess — a "
                     "residual, a false-positive count, a spread — and note that any metric can "
                     "be flipped into the other convention by negating it upstream.",
             }),
        *_ITER_MODES,
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Run the chain in front of it once per parameter value and keep one "
                "result. Wire the end of the chain into Collect, then pick what to iterate "
                "from the panel's dropdown (it lists every parameter in that chain) or drag "
                "a variable output onto the control itself, and flip through the results on "
                "the Viewer's iteration strip. 'best' picks by a Global scalar; 'feedback' "
                "searches for the value that maximizes it or hits a target.")
