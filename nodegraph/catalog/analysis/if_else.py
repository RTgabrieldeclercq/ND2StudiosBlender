"""If / Else (``analysis.if_else``) — Split a structure table's objects into a PASS set and
a FAIL set by up to four conditions on its own columns, with the condition palette drawn
from what the graph upstream actually measured."""

from __future__ import annotations

import numpy as np

from typing import Dict, List, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InDataset,
    InFloat,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.spill import dense_output
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import carried_over, on_layer
from nodegraph.catalog._shared.labels import (
    _label_raster, _label_tables, _point_layers, _resolve_layer,
)
from nodegraph.catalog._shared.planes import _each_plane

# ── If / Else (a PREDICATE over the table, evaluated per object) ────────────────
#
# The gap this fills: `analysis.filter_labels` cuts on ONE column, with the level derived
# from the population, and keeps one side. Three things were missing and each is an ordinary
# request — combining conditions ("big AND bright"), stating a level absolutely rather than
# asking the histogram for one, and keeping BOTH sides so the rejects can be looked at
# instead of vanishing.
#
# Like `filter_labels` it reads the TABLE and never the pixels. The columns it can test are
# therefore whatever the graph measured, and the edit-time column catalog (V2.28
# `adds_columns`) is what turns that from a fact the user discovers at pull time into a menu.

#: The comparison each condition applies. The six relational operators take ``value``; the
#: two predicates ignore it and ask only whether the column was measured at all.
_IF_OPS: Tuple[str, ...] = (">", ">=", "<", "<=", "==", "!=", "is_finite", "is_nan")

#: Operators that do NOT read their ``value`` socket — the socket stays live (it is one row
#: of a group the user is editing) but the compute must not pretend the number mattered.
_IF_PREDICATES = frozenset({"is_finite", "is_nan"})

#: How many condition slots the node offers. Four because that is where the inspector stops
#: being readable, and because a fifth term is expressible as a second If/Else in series —
#: which is also how a mixed AND/OR expression is built, since `match` is one word for the
#: whole group rather than a per-row connective.
_IF_TERMS: int = 4


def _condition_mask(values: np.ndarray, op: str, level: float) -> np.ndarray:
    """One condition → a boolean row mask.

    **NaN is never true.** A non-finite entry means the column was never measured for that
    object (a region the regionprops walk never saw, a velocity with no previous frame), and
    numpy's comparison semantics already answer ``False`` for every relational operator —
    which is the right answer, because the alternative is inventing a measurement. It is
    stated here rather than left implicit because it decides which SIDE an unmeasured object
    lands on, and the caller reports the count for exactly that reason.

    ``==``/``!=`` on floats are the operators a user is most likely to be surprised by, and
    they are passed through verbatim rather than given a tolerance: a silent epsilon would
    make ``==`` and ``!=`` non-complementary, and the column is in whatever unit it was
    measured in, so there is no scale this node could pick an epsilon from. ``!=`` excludes
    non-finite rows explicitly, since ``nan != x`` is True in numpy and would otherwise make
    the one relational operator that passes unmeasured objects."""
    if op == "is_finite":
        return np.isfinite(values)
    if op == "is_nan":
        return ~np.isfinite(values)
    if op not in _IF_OPS:
        raise ValueError(f"if/else: unknown operator {op!r} — expected one of "
                         f"{', '.join(_IF_OPS)}.")
    with np.errstate(invalid="ignore"):
        if op == ">":
            return values > level
        if op == ">=":
            return values >= level
        if op == "<":
            return values < level
        if op == "<=":
            return values <= level
        if op == "==":
            return values == level
        return np.isfinite(values) & (values != level)


def _track_joined(ds: Dataset, cols: Dict[str, np.ndarray],
                  ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Track-table columns projected onto the MEMBER rows, by the ``member_id`` join.

    ``track.objects`` writes ``track_length`` — how many frames an object was followed for —
    onto the **Track** table, and writes only ``track_id`` back onto the members. So a
    condition like "tracked for at least 10 frames" is a question about the label in front of
    the user whose answer lives on a different table, and without this join it would be
    unaskable at the point it is asked.

    Only columns the member table does NOT already carry are joined, so a member-side
    ``track_id`` always wins over the Track table's — they agree, but the member column is
    the one every other node in the catalog reads, and shadowing it here would make this
    node's ``track_id`` mean something subtly different from everyone else's.

    A member with no track (id 0, or absent from the membership) gets NaN, which
    :func:`_condition_mask` then routes to the fail side. That is the honest answer: an
    untracked object has no track length, and reporting 0 would read as "tracked for zero
    frames" and quietly satisfy ``< 5``."""
    out: Dict[str, np.ndarray] = {}
    for lyr in sorted({k[1] for k in ds.attributes if k[0] is Domain.TRACK and k[1]}):
        tcols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.TRACK)
                 if a.layer == lyr}
        member = tcols.get("member_id")
        if member is None:
            continue
        member = np.asarray(member, dtype=np.int64)
        for name, vals in tcols.items():
            # `t` is excluded with the join key: the Track table has one row per (track,
            # timepoint), so its `t` is the member's own frame and joining it would shadow
            # the member column of the same name with a value that is equal at best.
            if name in ("member_id", "t") or name in cols or name in out:
                continue
            vals = np.asarray(vals)
            if vals.shape != member.shape:
                continue
            lookup = dict(zip(member.tolist(), np.asarray(vals, dtype=float).tolist()))
            out[name] = np.array([lookup.get(int(i), np.nan) for i in ids.tolist()],
                                 dtype=float)
    return out


def _resolve_members(ctx: EvalContext, ds: Dataset, target: str):
    """``(domain, layer, columns, z_kind)`` for the member table being split.

    Deliberately NOT :func:`nodegraph.catalog._shared.objects._object_table`, which is
    otherwise the same resolution: that helper ends in a **2D-only refusal**, because the two
    nodes it serves fit in-plane neighbourhood gradients. This node reads a column and
    compares it to a number, an operation with no geometry at all — so refusing a volumetric
    table here would block a legitimate graph for a reason that does not apply to it."""
    domain = Domain.POINT if target == "point" else Domain.LABEL
    socket = "points" if target == "point" else "labels"
    # Both names spelled as LITERALS rather than resolved through `socket`: the socket
    # contract's clause (2) is checked by an AST scan for `ctx.layer("<name>")`, so a
    # computed key reads as a socket nothing consumes and the gate reports a dead control.
    want = ctx.layer("points") if target == "point" else ctx.layer("labels")
    layer, _note = _resolve_layer(
        _point_layers(ds) if target == "point" else _label_tables(ds),
        want, node="if/else", socket=socket,
        what=f"{domain.value} table", where="the `data` input",
        remedy="run analysis.segment / analysis.label (Label members) or detect.spots / "
               "detect.particles (Point members) upstream, then measure the column you want "
               "to test with analysis.measure or analysis.object_metrics",
        ctx=ctx)
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(domain) if a.layer == layer}
    if "id" not in cols:
        raise ValueError(
            f"if/else: no {domain.value} structure {layer!r} on the input Dataset, so there "
            f"is nothing to split. Run analysis.segment / analysis.label (Label members) or "
            f"detect.spots / detect.particles (Point members) upstream.")
    n = len(cols["id"])
    ragged = sorted(k for k, v in cols.items() if len(v) != n)
    if ragged:
        raise ValueError(
            f"if/else: column(s) {ragged} on layer {layer!r} disagree in length with 'id' "
            f"({n}) — every object would be judged against another object's value.")
    zk = ds.structure_zkind(domain, layer) or "plane_index"
    return domain, layer, cols, zk


def _term_spec(ctx: EvalContext, column_socket: str, op_socket: str,
               value_socket: str) -> Tuple[str, str, float]:
    """One condition row's ``(column, operator, value)``, read by LITERAL socket name.

    A helper rather than an ``f"column{i}"`` loop because the socket contract's clause (2)
    is enforced by an AST scan for ``ctx.params.get("<name>")``: a computed key is invisible
    to it, so every one of these twelve sockets would be reported as a control no compute
    reads. Forwarding literals through a helper argument is the pattern that scan already
    resolves (``_radius_px(ctx, "radius", 0.3)``), and it has the better property anyway —
    the socket names stay greppable in the source."""
    column = str(ctx.params.get(column_socket, "") or "").strip()
    op = str(ctx.params.get(op_socket, ">") or ">").strip()
    level = float(ctx.params.get(value_socket, 0.0) or 0.0)
    return column, op, level


def _active_terms(ctx: EvalContext, cols: Dict[str, np.ndarray],
                  joined: Dict[str, np.ndarray], layer: str
                  ) -> List[Tuple[str, str, float, np.ndarray]]:
    """The conditions the user actually filled in, resolved against the real table.

    A slot whose column is BLANK is skipped rather than refused: `terms` says how many rows
    the inspector shows, and a user who sets it to 3 and fills two is mid-edit, not in error.
    A slot whose column is FILLED but absent from the table is refused with the list of what
    is there, because that one is a typo or a missing upstream node, and silently dropping it
    would quietly widen the pass set — the failure mode this node exists to avoid."""
    terms: List[Tuple[str, str, float, np.ndarray]] = []
    try:
        n_terms = int(ctx.params.get("__modes__", {}).get("terms", "1") or 1)
    except (TypeError, ValueError):
        n_terms = 1
    available = dict(cols)
    available.update(joined)
    # Unrolled, and every row read whether or not it is active: reading a param is free, and
    # the four literal call sites are what make the twelve sockets visible to clause (2).
    rows = (_term_spec(ctx, "column1", "op1", "value1"),
            _term_spec(ctx, "column2", "op2", "value2"),
            _term_spec(ctx, "column3", "op3", "value3"),
            _term_spec(ctx, "column4", "op4", "value4"))
    for i, (column, op, level) in enumerate(rows[:min(n_terms, _IF_TERMS)], start=1):
        if not column:
            continue
        if op not in _IF_OPS:
            raise ValueError(f"if/else: condition {i} has unknown operator {op!r} — "
                             f"expected one of {', '.join(_IF_OPS)}.")
        if column not in available:
            joinable = sorted(joined)
            raise ValueError(
                f"if/else: condition {i} tests column {column!r}, which the {layer!r} table "
                f"does not carry. It has {sorted(k for k in cols if k != 'id')}"
                + (f", plus {joinable} joined from the Track table" if joinable else "")
                + ". Measure it first: analysis.measure `stats` gives the intensity "
                  "statistics and `shape` the regionprops geometry (eccentricity, "
                  "solidity, …), analysis.object_metrics gives speed / velocity / neighbour "
                  "distances, and track.objects gives track_length.")
        values = np.asarray(available[column], dtype=float)
        terms.append((column, op, level, _condition_mask(values, op, level)))
    if not terms:
        raise ValueError(
            "if/else: no condition is filled in, so every object would pass and the node "
            "would be a no-op wearing a filter's name. Set condition 1's column and "
            "operator — the dropdown lists the columns this graph has measured.")
    return terms


def _compute_if_else(ctx: EvalContext) -> Dataset:
    """Split a structure table's objects into a **pass** set and a **fail** set by up to four
    conditions on its own columns.

    Resolved spec (grilled 2026-09-08): category analysis; op ``analysis.if_else``; reads
    ``{VOXEL, LABEL}`` on the label branch and ``{POINT}`` on the point branch
    (``reads_domains_by_mode``), adds ``{VOXEL, LABEL, POINT}``; ``WHOLE_VOLUME``; **no 2D/3D
    lever** — it compares numbers in a table and rewrites a raster it does not interpret
    geometrically, which is ``analysis.filter_labels``' reasoning exactly. Reads **no**
    calibration key, so its memo fences on nothing: the column is already in whatever unit it
    was measured in, and this node does not convert it.

    **Two results, one payload.** The engine is one-payload-per-node, so "if" and "else"
    cannot be two output SOCKETS — they are two named layer instances on the single output,
    ``pass_name`` and ``fail_name``. Wiring the else branch onward is therefore an ordinary
    wire plus a layer pick downstream, not a second port. The alternative (the
    ``channel.split`` arrangement: GUI-synthetic ports materialized into real nodes at
    graph-build time) buys the second port at the cost of a rewrite pass, and would still be
    two instances underneath.

    **Ids are PRESERVED on both sides**, never renumbered, for ``filter_labels``' reason:
    label 7 stays label 7, so every upstream measurement and every track that references it
    still joins. The two sides are disjoint and their union is the input, so an object is
    never counted twice and never lost.

    **The conditions are COLUMNS, not measurements this node makes.** It reads the table and
    never the pixels, so it tests anything upstream produced — ``analysis.measure``'s
    intensity statistics and regionprops geometry, ``analysis.object_metrics``' speed and
    neighbour distances, ``track.objects``' ``track_length`` (via the ``member_id`` join in
    :func:`_track_joined`), a column this node has never heard of. Re-measuring here would
    have bought one fewer node in the graph and permanently limited the conditions to the
    handful it hardcoded — and would have made the same statistic mean two different things
    depending on whether it was measured upstream or inside the filter.

    **An unmeasured object fails.** A non-finite column value is not a small one; it means
    the measurement never happened, so it satisfies no relational operator and lands on the
    fail side. The count is reported on the progress rail, because an object silently treated
    as "not bright enough" when it was in fact "never measured" is the failure this node
    would otherwise hide. ``is_finite`` / ``is_nan`` make that split selectable directly.

    **An empty side is a result, not an error.** ``filter_labels`` refuses an empty output
    because there it means the node did nothing useful; here "nothing failed" is a legitimate
    and informative answer, and refusing it would break a graph whose data simply got better.
    Both counts go on the rail instead."""
    ds = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    target = modes.get("target", "label")
    match = modes.get("match", "all")
    if match not in ("all", "any"):
        raise ValueError(f"if/else: unknown match {match!r} — expected 'all' or 'any'.")

    domain, layer, cols, zk = _resolve_members(ctx, ds, target)
    ids = np.asarray(cols["id"], dtype=np.int64)
    if ids.size == 0:
        raise ValueError(
            f"if/else: the {domain.value} table {layer!r} has no rows, so there is nothing "
            f"to split. Whatever produced it found no objects — fix that upstream rather "
            f"than conditioning an empty result.")

    joined = _track_joined(ds, cols, ids)
    terms = _active_terms(ctx, cols, joined, layer)
    keep = terms[0][3].copy()
    for _c, _o, _l, mask in terms[1:]:
        keep = (keep & mask) if match == "all" else (keep | mask)

    pass_name = ctx.layer("pass_name")
    fail_name = ctx.layer("fail_name")
    if pass_name == fail_name:
        raise ValueError(
            f"if/else: both sides are named {pass_name!r}, so one would overwrite the other "
            f"and the split would be invisible. Give them different names.")
    for nm, side in ((pass_name, "pass"), (fail_name, "fail")):
        if nm == layer:
            raise ValueError(
                f"if/else: the {side} layer is also named {layer!r}, which would overwrite "
                f"the objects being split and leave no way to see what moved. Give it its "
                f"own name (defaults: `objects_pass` / `objects_fail`).")

    # Reported, not silently absorbed: an object that fails because it was never measured is
    # a different finding from one that fails on its value, and only the rail can say so.
    both = {**cols, **joined}
    unmeasured = sum(
        int(np.count_nonzero(~np.isfinite(np.asarray(both[c], dtype=float))))
        for c, o, _l, _m in terms if o not in _IF_PREDICATES)
    shown = " ".join(f"{c}{o}{l:g}" if o not in _IF_PREDICATES else f"{c} {o}"
                     for c, o, l, _m in terms)
    n_pass, n_fail = int(np.count_nonzero(keep)), int(np.count_nonzero(~keep))
    tail = (f" → {n_pass} pass / {n_fail} fail"
            + (f", {unmeasured} unmeasured value(s) failed" if unmeasured else ""))

    if domain is Domain.POINT:
        out = ds
        for name, sel in ((pass_name, keep), (fail_name, ~keep)):
            out = out.with_structure(StructureTable(
                Domain.POINT, {k: np.asarray(v)[sel] for k, v in cols.items()},
                layer=name, z_kind=zk))
        ctx.progress(1, 1, f"if/else {shown} ({match}){tail}")
        return out

    raster6, rzk = _label_raster(ds, layer, node="if/else")
    # One pass over the raster per side, plane by plane, so the progress bar is real and no
    # frame-sized boolean outlives the plane it belongs to. The parent's dtype is preserved:
    # an int32 raster stays int32 rather than being widened for nothing.
    out = ds
    n_units = ax.m * ax.t * ax.z * ax.c
    note = f"if/else {shown} ({match})"
    ctx.progress(0, 2 * n_units, note, frames=ax.t)
    for side, (name, sel) in enumerate(((pass_name, keep), (fail_name, ~keep))):
        keep_ids = np.sort(ids[sel])
        buf = dense_output(tuple(raster6.shape), raster6.dtype,
                           tag=f"if_else_{side}_{ctx.node_id}")
        arr = buf.array
        for i, (m, t, z, c) in enumerate(_each_plane(ax)):
            plane = np.asarray(raster6[m, t, z, c])
            arr[m, t, z, c] = np.where(np.isin(plane, keep_ids), plane, 0)
            ctx.progress(side * n_units + i + 1, 2 * n_units, note, frames=ax.t)
        out = (out.with_layer(Domain.VOXEL, name, buf.seal())
                  .with_structure(StructureTable(
                      Domain.LABEL, {k: np.asarray(v)[sel] for k, v in cols.items()},
                      layer=name, z_kind=rzk)))
    ctx.progress(2 * n_units, 2 * n_units, note + tail, frames=ax.t)
    return out


def _layers_if_else(params, modes):
    """The two instances this node writes, per branch (V2.11 ``extra_layers``).

    ``layer_out`` could not express it. It is a static tuple of domains, and this node's
    output domains follow the ``target`` lever — a label split writes a Voxel raster AND a
    Label table under each name, a point split writes only a Point table. Declaring the union
    would announce a Point layer that the label branch never creates, and the picker
    downstream would offer a layer no node writes, which is indistinguishable from a typo.

    Total by contract: runs inside ``propagate_meta`` on every keystroke."""
    try:
        names = [str((params or {}).get("pass_name") or "objects_pass"),
                 str((params or {}).get("fail_name") or "objects_fail")]
        if (modes or {}).get("target") == "point":
            return tuple((Domain.POINT, n) for n in names)
        return tuple((d, n) for n in names for d in (Domain.VOXEL, Domain.LABEL))
    except Exception:                        # pragma: no cover - defensive
        return ()


def _columns_if_else(params, modes, incoming):
    """Every column of the split table, re-offered under BOTH output names (V2.28).

    This node measures nothing and drops no column — it partitions rows — so each side
    carries exactly the member table's columns. Declaring them is what lets a second If/Else
    (or a Measure, or a viewer) chain onto either branch with a populated picker instead of a
    blank one, and chaining conditions is the whole reason the node has two outputs.

    Total by contract."""
    try:
        src = (Domain.POINT if (modes or {}).get("target") == "point" else Domain.LABEL)
        socket = "points" if src is Domain.POINT else "labels"
        base = str((params or {}).get(socket) or ("spots" if src is Domain.POINT
                                                  else "labels"))
        out: Tuple = ()
        for key, fallback in (("pass_name", "objects_pass"), ("fail_name", "objects_fail")):
            out += carried_over(incoming, src, base,
                                str((params or {}).get(key) or fallback))
        return out
    except Exception:                        # pragma: no cover - defensive
        return ()


#: The three sockets of one condition row. A helper rather than twelve hand-written
#: declarations so the four rows cannot drift apart in wording or in gating — the defect
#: `_InRadius` avoids for the lateral/axial pair.
def _InCondition(i: int) -> Tuple:
    gate = (None if i == 1 else
            {"terms": frozenset(str(k) for k in range(i, _IF_TERMS + 1))})
    nth = ("first", "second", "third", "fourth")[i - 1]
    return (
        InString(f"column{i}", f"Condition {i} column", field=False,
                 default="area" if i == 1 else "", available_in=gate,
                 column_in_mode="target", column_join=(Domain.TRACK,),
                 description=
                 f"Which per-object column the {nth} condition judges. The dropdown lists "
                 "the columns THIS graph has measured, so it grows as you add nodes: "
                 "segmentation gives `area` and the centroid `y`/`x`; Measure adds the "
                 "intensity statistics and, via its `shape` selector, `eccentricity` / "
                 "`solidity` / `perimeter`; Object Metrics adds `speed` and the neighbour "
                 "distances; Track Objects adds `track_length`, joined from the Track table "
                 "onto these rows. A CLOSED list, not free text: every node that writes columns "
                 "declares them, so anything missing here is a node you have not added "
                 "rather than a name you have to remember. LEAVE IT BLANK to skip this "
                 "row."
                 + ("" if i == 1 else " Only read while `terms` is at least "
                                      f"{i}.")),
        InString(f"op{i}", f"Condition {i} test", field=False, default=">",
                 available_in=gate, choices=_IF_OPS,
                 description=
                 f"How the {nth} condition's column is compared with its value. The six "
                 "relational tests read the value below; the two `is_` tests ignore it and "
                 "ask only whether the column was ever measured for that object. An object "
                 "whose value is missing (NaN) never satisfies a relational test, so it "
                 "lands on the FAIL side — the count is reported on the progress rail so "
                 "'never measured' does not silently read as 'too small'.",
                 choice_docs={
                     ">": "Keeps objects strictly ABOVE the value. The usual brightness or "
                          "size gate; excludes an object sitting exactly on the number.",
                     ">=": "As `>`, but an object exactly ON the value passes. Use it when "
                           "the value is a documented minimum ('at least 10 frames') rather "
                           "than a cut you tuned.",
                     "<": "Keeps objects strictly BELOW the value — debris gates, slow-moving "
                          "cells, anything where the value is a ceiling.",
                     "<=": "As `<`, but an object exactly ON the value passes. The "
                           "complement of `>`, so the two split the population with no gap.",
                     "==": "Keeps objects whose value is EXACTLY the number. Meant for "
                           "integer-valued columns — `track_id`, `m`, `c`, `t`. On a "
                           "measured float it will usually match nothing: no tolerance is "
                           "applied, deliberately, because the column's unit is whatever "
                           "produced it and this node has no scale to pick an epsilon from.",
                     "!=": "Keeps objects whose value is anything OTHER than the number, "
                           "unmeasured objects excluded. The complement of `==` over the "
                           "measured rows — pick one channel out, drop one track.",
                     "is_finite": "Keeps every object the column was actually MEASURED for, "
                                  "ignoring the value entirely. The way to drop objects a "
                                  "regionprops walk missed or that a velocity could not be "
                                  "computed for, before conditioning on the number itself.",
                     "is_nan": "Keeps only the objects the column was NOT measured for — the "
                               "inverse of `is_finite`. Diagnostic: wire the pass side to a "
                               "viewer to see WHICH objects your measurement is missing, "
                               "rather than inferring it from a count.",
                 }),
        InFloat(f"value{i}", f"Condition {i} value", field=False, default=0.0,
                available_in=gate, unit="",
                description=
                f"The number the {nth} condition compares against. **Unitless on purpose**: "
                "the column is already in whatever unit the node that measured it emitted — "
                "`area` in voxels from segmentation but µm² from Object Metrics, "
                "`mean_intensity` in raw counts, `speed` in µm/s, `eccentricity` a 0–1 ratio "
                "— and this node converts nothing, so the number you type is compared "
                "exactly as written. Check the column's unit in the Spreadsheet before "
                "trusting a threshold. Ignored when the test above is `is_finite` or "
                "`is_nan`."),
    )


register_node(
    _compute_if_else, op_key="analysis.if_else", label="If / Else", category="analysis",
    # The label branch reads the raster AND the table it rewrites; the point branch has no
    # raster at all. Stated per branch (V2.22) rather than as a union, so the domain rail
    # does not demand a Voxel layer of a graph that only ever made dots.
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.VOXEL, Domain.LABEL}),
        "point": frozenset({Domain.POINT}),
    }},
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL, Domain.POINT}),
    extra_layers=_layers_if_else,
    adds_columns=_columns_if_else,
    inputs=[
        InDataset(description=
                  "The Dataset carrying the objects to split — the structure table whose "
                  "columns the conditions test, plus (on Label members) the raster those ids "
                  "index. No pixels are read: whatever the columns were measured from "
                  "happened upstream."),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL, available_in={"target": frozenset({"label"})},
                 description=
                 "Which Label instance to split — a raster plus its table, from Segmentation, "
                 "Connected Components or Histogram Threshold. This layer is left UNCHANGED "
                 "on the wire; the two sides go to the output layers named below, so all "
                 "three are available downstream and the viewer can show what moved."),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT, available_in={"target": frozenset({"point"})},
                 description=
                 "Which Point structure to split — from Spot Detection, Particle Detection, "
                 "or Label → Points. Left UNCHANGED on the wire; the two sides go to the "
                 "output layers named below."),
        *_InCondition(1), *_InCondition(2), *_InCondition(3), *_InCondition(4),
        InString("pass_name", "Pass layer", field=False, default="objects_pass",
                 description=
                 "Name for the instance holding the objects that SATISFY the conditions — "
                 "the 'if' branch. A new layer, so the input is untouched and both sides can "
                 "be viewed at once. Ids are preserved, so this table still joins against "
                 "anything measured upstream."),
        InString("fail_name", "Fail layer", field=False, default="objects_fail",
                 description=
                 "Name for the instance holding the objects that do NOT satisfy the "
                 "conditions — the 'else' branch, including every object whose column was "
                 "never measured. This is what makes the node an if/else rather than a "
                 "filter: wire this node's output onward and point the downstream layer "
                 "picker at THIS name to build a second pipeline on the rejects."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("target", ["label", "point"], default="label", label="Members",
             description=
             "Which kind of object is being split. It selects the layer socket above and the "
             "domain the conditions read; a Label split also rewrites the id raster, a Point "
             "split only the table.",
             choice_docs={
                 "label": "Split REGIONS — a segmentation's labelled areas. Both sides get a "
                          "raster and a table, so either can be viewed as an overlay and fed "
                          "to anything that consumes labels. The conditions can test region "
                          "geometry (area, eccentricity, solidity) as well as intensity.",
                 "point": "Split DETECTIONS — dimensionless dots from a spot/particle "
                          "detector or Label → Points. No raster is read or written, so this "
                          "branch is cheap; the conditions can test position and any column "
                          "measured onto the points, but not region shape, which a point "
                          "does not have.",
             }),
        Mode("match", ["all", "any"], default="all", label="Match",
             description=
             "How the filled-in conditions combine into one decision. One word for the whole "
             "group rather than a connective per row — for a mixed expression, chain a second "
             "If / Else onto this one's pass or fail layer.",
             choice_docs={
                 "all": "An object passes only if EVERY filled-in condition holds (AND). "
                        "Narrows the pass set with each condition you add — the usual "
                        "'big and bright and round' gate.",
                 "any": "An object passes if AT LEAST ONE filled-in condition holds (OR). "
                        "Widens the pass set with each condition added — use it to collect "
                        "several distinct populations, or to catch objects failing any one "
                        "of several quality checks.",
             }),
        Mode("terms", ["1", "2", "3", "4"], default="1", label="Conditions",
             description=
             "How many condition rows are active. Rows beyond this are hidden and never "
             "read, so raising it later restores what you typed. A row left blank is skipped "
             "rather than refused, which is what makes editing the middle of a group safe.",
             choice_docs={
                 "1": "One condition — the plain case, and equivalent to Filter Labels with "
                      "an explicit level instead of a histogram-derived cut.",
                 "2": "Two conditions, combined by `match` — the common 'size AND intensity' "
                      "gate that needed two chained nodes before.",
                 "3": "Three conditions. Worth checking the pass count on the rail here: with "
                      "`match=all` each added condition can only shrink it, and three "
                      "conditions is where an empty pass set stops being obvious.",
                 "4": "Four conditions, the maximum. A fifth is a second If / Else chained "
                      "onto this one's pass layer — which is also how you build a mixed "
                      "AND/OR expression.",
             }),
    ],
    granularity=Granularity.WHOLE_VOLUME,
    description="Split objects into a PASS set and a FAIL set by up to four conditions on "
                "the structure table's own columns, combined with all/any. The condition "
                "dropdown offers the columns the graph has actually measured — `area` and "
                "the centroid from segmentation, Measure's intensity statistics and "
                "regionprops shape metrics, Object Metrics' speed and neighbour distances, "
                "and Track Objects' `track_length`, joined onto the member rows by "
                "`member_id`. Both sides are written as new instances (ids UNCHANGED, so "
                "upstream measurements and tracks still join), so the else branch can be "
                "wired onward by pointing a downstream layer picker at the fail layer. An "
                "object whose column was never measured fails, and is counted on the rail.",
)
