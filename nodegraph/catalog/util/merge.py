"""Merge (``util.merge``) — Merge two or more Datasets by growing ONE chosen axis
(Channel / Time / Multipoint / Z). Absorbs and retires ``channel.merge``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, FrozenSet, List, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (
    PER_POSITION_KEYS, PER_TIME_KEYS, SOURCE_FILE_KEY, merge_grow as _meta_merge_grow)
from nodegraph.provider import AxisConcatProvider, ChannelMergeProvider
from nodegraph.registry import Granularity, InBool, InDataset, InFloat, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.placement_entry import handedness_for, overlay_entry
from nodegraph.catalog._shared.sampling import _sampling_of

# ── Merge (§0 grill, 2026-09-15 — retires channel.merge) ─────────────────────────
#
# channel.merge put exactly two acquisitions on ONE channel axis, placed by absolute stage
# position and focus. Asked to generalise it to "merge different files and chain them onto
# T, M, C or Z" for a before/after 3D-indentation SerialTrack workflow, where two separately
# ingested volumes need to become two TIMEPOINTS of one series — a concatenation channel.merge
# never did (it only ever grew C).
#
# The grill settled two axis FAMILIES with genuinely different math, dispatched by the
# `merge_axis` Mode rather than split into separate nodes (the user's call, not the
# recommendation — the tradeoff was surfaced and this is the chosen shape):
#
# * `merge_axis="C"` — placement-based compositing, unchanged from channel.merge: two files
#   rarely share a pixel size or a focus range, so the secondary is RESAMPLED (nearest-
#   neighbour) and Z-reconciled (`nodegraph.placement.plan_placement`/`merge_z_grid`) onto
#   the running result. Generalised to N-ary by CHAINING: input 1 is placed against input 0,
#   input 2 against THAT result, and so on — reusing `ChannelMergeProvider` unmodified at
#   each step (nesting composes a correct fingerprint/version and a correct lazy read; this
#   is exactly what two chained Merge(C) nodes would do by hand).
# * `merge_axis="T"/"M"/"Z"` — literal index concatenation via `AxisConcatProvider`: input 0
#   is frame/position/plane 0..n0-1, input 1 is n0..n0+n1-1, and so on. Nothing is resampled
#   or placed. Mirrors `nodegraph.provider.MultiSourceProvider`'s existing, tested policy for
#   the ingest-time M-bundle: every OTHER axis must already match exactly (refuse otherwise,
#   naming the axis and the values) rather than silently reconciling a difference this branch
#   cannot see the physical meaning of.
#
# **Input arity and order.** One `InDataset(multi=True)` socket takes any number of upstream
# Datasets. Order is the engine's existing canonical multi-socket order (edge-creation order,
# `engine.py` `_entry`) — invisible in the GUI beyond a "· multi" hover tag, which matters a
# lot for T/M (frame 0 vs frame 1 IS the result) and not at all for C. Rather than add an
# editable order control (an "ask", against `wire-node-v2`'s "derive, don't ask"), the node
# stamps a resolved-order NOTE — the same idiom as channel.merge's `__merge__`/
# `PlacementPlan.describe()` — so a wrong order is something you SEE on the node card, not
# something you have to audit.
#
# **Backward compatibility.** `channel.merge` is DEREGISTERED, not aliased: a saved graph
# that used it (e.g. the repo's own `TFM_docked.nd2graph.json`) will silently stop merging on
# its next pull rather than erroring (`engine.py`'s unknown-op path treats it as an inert
# source) — a known, accepted cost of a clean break rather than a compatibility shim. Fix any
# such graph by hand: replace the node with this one, `merge_axis="C"`, same params.


#: Which axis the node grows — the header Mode this whole node dispatches on.
_MERGE_AXES: Tuple[str, ...] = ("C", "T", "M", "Z")

#: Footprint per `merge_axis` (`footprint_mode="merge_axis"`, the `util.zproject`/
#: `analysis.threshold` precedent for a non-dim Mode-keyed footprint). `C` inherits
#: channel.merge's MULTI_VIEW: one output channel-plane may read several of a secondary's
#: tiles. `T`/`M`/`Z` are a pure index remap — one output unit reads EXACTLY the
#: corresponding input's own unit, nothing more — so they are as cheap as a pass-through
#: (the same reasoning `util.zproject`'s `method="none"` reset uses for its own TILEABLE).
_MERGE_GRAN: Dict[str, Granularity] = {
    "C": Granularity.MULTI_VIEW,
    "T": Granularity.TILEABLE, "M": Granularity.TILEABLE, "Z": Granularity.TILEABLE,
}
_MERGE_KAX: Dict[str, FrozenSet[str]] = {
    "C": frozenset({"m", "y", "x"}),
    "T": frozenset(), "M": frozenset(), "Z": frozenset(),
}

#: Namespaced, non-calibration record of what the merge did — the note the node card reads
#: (`wire-node-v2` §7b stamp-and-inherit; the same idiom channel.merge used as `__merge__`).
MERGE_KEY = "__merge__"

#: Scalar calibration that must already agree across every input, regardless of which axis
#: is being grown — including when growing Z itself: this schema has only ONE z_step_um, so
#: two stacks at different spacings cannot be concatenated without lying about where a plane
#: sits. A caller with a real mismatch resamples upstream (`util.resample`) first.
_CALIBRATION_MUST_MATCH: Tuple[str, ...] = ("pixel_size_um", "z_step_um", "bit_depth")


def _compute_merge(ctx: EvalContext) -> Dataset:
    """Grow one axis — C, T, M or Z — across every Dataset wired into ``data``.

    Resolved spec (§0 grill, 2026-09-15)
    -------------------------------------
    * **Kind** utility, axis-changing → ``op_key="util.merge"``, category ``"utility"``.
      Retires ``channel.merge`` (absorbed whole as the ``merge_axis="C"`` branch).
    * **Data contract** grows exactly the axis ``merge_axis`` names; every other axis is
      required to already match across every input (refused otherwise) EXCEPT under
      ``merge_axis="C"``, which keeps channel.merge's placement/resampling reconciliation
      for a mismatched pixel size or focus range — see module docstring for why the two
      branches are not the same math.
    * **2D/3D** no lever, exactly as channel.merge: the node runs no kernel over voxels,
      only placement arithmetic (``C``) or index arithmetic (``T``/``M``/``Z``), and
      dimensionality is a property of the DATA, not something this node could add an
      opinion about (`wire-node-v2` §7b — derive, don't ask).
    * **Footprint** keyed by ``merge_axis`` (``footprint_mode="merge_axis"``): ``C`` is
      ``MULTI_VIEW`` (a channel-plane may fan out to several of a secondary's tiles,
      unchanged from channel.merge); ``T``/``M``/``Z`` are ``TILEABLE`` with no kernel
      axis (a pure index remap costs exactly what the un-merged read would).
    * **Backend** none for ``T``/``M``/``Z`` (:class:`~nodegraph.provider.AxisConcatProvider`
      is a lazy index remap, no different in kind from
      :class:`~nodegraph.provider.MultiSourceProvider`); :mod:`nodegraph.placement` +
      :class:`~nodegraph.provider.ChannelMergeProvider`, chained, for ``C``.

    **Input order** is the engine's canonical multi-socket order (edge-creation order) —
    input 0 is the reference every other input is checked against (and, under ``C``, the
    first primary in the placement chain). The result's :data:`MERGE_KEY` metadata carries
    a resolved-order note naming which input landed where, since the wiring order itself is
    not visible anywhere else in the GUI.
    """
    files: Tuple[Dataset, ...] = tuple(ctx.input("data") or ())
    if len(files) < 2:
        raise ValueError(
            "Merge needs at least two Datasets wired into its input — connect a second "
            "file (or chain) before this node; one input alone has nothing to merge with.")
    for i, ds in enumerate(files):
        if ds.image is None:
            raise ValueError(f"Merge: input {i} has no image to merge.")

    axis = str(ctx.params.get("__modes__", {}).get("merge_axis", "C"))
    if axis == "C":
        return _merge_channel_axis(ctx, files)
    if axis in ("T", "M", "Z"):
        return _merge_index_axis(files, axis)
    raise ValueError(f"Merge: unknown merge_axis {axis!r} (expected one of {_MERGE_AXES})")


# ── merge_axis="C" — chained placement-based compositing (channel.merge, generalised) ────

def _merge_channel_axis(ctx: EvalContext, files: Tuple[Dataset, ...]) -> Dataset:
    """Chain the placement-based channel merge pairwise: input 1 is placed against input 0,
    input 2 against THAT running result, and so on. Each step is byte-for-byte
    channel.merge's own logic (unchanged), just reading from the shared node's params
    instead of a fixed ``secondary`` socket, and folding into a Dataset instead of
    returning one — so two files merge exactly as they did before, and a third (or a
    twentieth) is the same operation applied again.
    """
    from nodegraph.placement import merge_z_grid, plan_placement

    modes = ctx.params.get("__modes__", {})
    offset = (float(ctx.params.get("offset_z", 0.0)),
              float(ctx.params.get("offset_y", 0.0)),
              float(ctx.params.get("offset_x", 0.0)))
    t_shift = int(ctx.params.get("t_shift", 0))
    min_coverage = float(ctx.params.get("min_coverage", 0.0))
    on_unplaceable = modes.get("unplaceable", "refuse")
    z_grid_mode = modes.get("z_grid", "union")

    result = files[0]
    notes: List[str] = []
    all_warnings: List[str] = []
    placed_by = "stage"
    added_channels = 0
    z_planes = int(files[0].axes.z)
    z_step = files[0].metadata.get("z_step_um")
    coverage: Dict[int, float] = {}
    for i, sec in enumerate(files[1:], start=1):
        plan = plan_placement(
            result.metadata, result.axes, sec.metadata, sec.axes,
            t_shift=t_shift, offset_um=offset,
            # Channel-blind, exactly as channel.merge/view.overlay: a channel axis is not a
            # spatial one, so a channel tap upstream leaves every (z,y,x) address in place.
            dst_sampling=_sampling_of(result, channel_axis=False),
            src_sampling=_sampling_of(sec, channel_axis=False),
            min_coverage=min_coverage, on_unplaceable=on_unplaceable)
        if plan.refusals:
            raise ValueError(f"Merge (C): cannot place input {i} onto the running result:\n"
                             + "\n".join(f"  - {r}" for r in plan.refusals))

        grid, z_warn, z_refuse = merge_z_grid(
            result.metadata, result.axes, sec.metadata, sec.axes, plan.tiles,
            mode=z_grid_mode, dz=offset[0])
        if z_refuse:
            raise ValueError(f"Merge (C): cannot build a Z grid for input {i}:\n"
                             + "\n".join(f"  - {r}" for r in z_refuse))

        ax, sax = result.axes, sec.axes
        # Handedness is DERIVED, not defaulted: an already-stitched secondary is in stage
        # coordinates and flipping it again mirrors the mosaic — see `handedness_for`.
        flip_x, flip_y, hand_warn = handedness_for(
            sec, ctx.params.get("flip_x", True), ctx.params.get("flip_y", False))
        entry = overlay_entry(
            ctx.node_id, plan,
            {"blend": "add",                 # unread here: a merge composites nothing
             "flip_x": flip_x, "flip_y": flip_y, "t_shift": t_shift, "offset_um": offset})
        new_axes = replace(ax, c=ax.c + sax.c, z=grid.n)
        prov = ChannelMergeProvider(
            result.image, sec.image, axes=new_axes, entry=entry, grid=grid,
            pri_md=dict(result.metadata), pri_axes=ax,
            sec_md=dict(sec.metadata), sec_axes=sax)

        names = list(result.metadata.get("channel_names") or [])
        while len(names) < ax.c:
            names.append(f"Ch{len(names) + 1}")
        snames = list(sec.metadata.get("channel_names") or [])
        for k in range(sax.c):
            label = str(snames[k]) if k < len(snames) and snames[k] else f"Ch{k + 1}"
            names.append(label if label not in names else f"{label} (2)")

        # bit_depth: THIS layer has both files (the meta_transform, handed only input 0,
        # cannot check), so restate it when they agree and drop it — the honest signal —
        # when they disagree or either is silent. See `_merge_channels_grow`.
        pri_bd, sec_bd = result.metadata.get("bit_depth"), sec.metadata.get("bit_depth")
        merged_bd = int(pri_bd) if (pri_bd and sec_bd and int(pri_bd) == int(sec_bd)) else None

        notes.append(f"input {i}: " + plan.describe(0)
                     + f"  ·  +{sax.c} channel(s)  ·  {grid.n} z @ {grid.step_um:g} um")
        step_warnings = list(plan.warnings) + list(z_warn) + list(hand_warn)
        notes.extend(f"input {i} ! {w}" for w in step_warnings)
        all_warnings.extend(step_warnings)
        if plan.placed_by != "stage":
            placed_by = plan.placed_by
        added_channels += int(sax.c)
        z_planes = int(grid.n)
        z_step = float(grid.step_um)
        coverage = dict(plan.coverage)

        result = (result.with_image(prov).reshaped_axes(new_axes)
                        .with_metadata(bit_depth=merged_bd, channel_names=names,
                                       z_step_um=(float(grid.step_um) if grid.n > 1
                                                 else None),
                                       origin_um=_merged_origins(result, grid)))

    note = {
        "placed_by": placed_by, "z_planes": z_planes, "z_step_um": z_step,
        "added_channels": added_channels,
        "coverage": [[m, round(c, 6)] for m, c in sorted(coverage.items())],
        "warnings": all_warnings,
        "note": "  ||  ".join(notes),
    }
    return result.with_metadata(**{MERGE_KEY: note})


def _merged_origins(ds: Dataset, grid) -> Any:
    """``origin_um`` with each field's Z corner moved to the merged grid's first plane —
    verbatim from channel.merge; see its original docstring for the reasoning."""
    origins = ds.metadata.get("origin_um")
    if not isinstance(origins, (list, tuple)) or not origins:
        return origins
    out = []
    for m, o in enumerate(origins):
        try:
            oz, oy, ox = (float(v) for v in o)
        except (TypeError, ValueError):
            return origins                       # malformed: leave it exactly as found
        z0 = grid.z0_um[m] if m < len(grid.z0_um) else None
        out.append([oz if z0 is None else float(z0), oy, ox])
    return out


# ── merge_axis in {"T","M","Z"} — literal index concatenation ────────────────────────────

def _merge_index_axis(files: Tuple[Dataset, ...], axis: str) -> Dataset:
    """Lay every input end to end on ``axis`` via :class:`AxisConcatProvider`. Refuses
    unless every OTHER axis and every scalar in :data:`_CALIBRATION_MUST_MATCH` already
    agrees across every input — see the provider's own docstring and the module docstring
    for why this branch does not resample or place anything the way ``C`` does."""
    ax_name = axis.lower()
    other = [n for n in ("m", "t", "z", "c", "y", "x") if n != ax_name]
    ref = files[0]
    ref_names = list(ref.metadata.get("channel_names") or [])
    for i, ds in enumerate(files[1:], start=1):
        bad = [n for n in other if int(getattr(ds.axes, n)) != int(getattr(ref.axes, n))]
        if bad:
            raise ValueError(
                f"Merge ({axis}): input {i} does not match input 0 on {', '.join(bad)} — "
                f"({', '.join(other)}) is "
                f"{tuple(int(getattr(ds.axes, n)) for n in other)} against "
                f"{tuple(int(getattr(ref.axes, n)) for n in other)}. Growing {axis} "
                f"requires every OTHER axis to already agree — resample or crop upstream "
                f"so they match, or pick a different merge axis.")
        names = list(ds.metadata.get("channel_names") or [])
        if names and ref_names and names != ref_names:
            raise ValueError(
                f"Merge ({axis}): input {i} has channels {names} but input 0 has "
                f"{ref_names} — growing {axis} assumes the channels already correspond "
                f"1:1; reorder or select channels upstream so they match.")
        for key in _CALIBRATION_MUST_MATCH:
            a, b = ref.metadata.get(key), ds.metadata.get(key)
            if a is not None and b is not None and abs(float(a) - float(b)) > 1e-9:
                raise ValueError(
                    f"Merge ({axis}): input {i} disagrees with input 0 on {key} "
                    f"({b!r} vs {a!r}) — growing {axis} assumes every input already "
                    f"shares the same physical scale; resample upstream so they match.")

    prov = AxisConcatProvider([ds.image for ds in files], axis=ax_name)
    out = ref.with_image(prov).reshaped_axes(prov.axes)
    out = _concat_axis_metadata(out, files, ax_name)
    if axis == "T":
        dts = {float(v) for v in (ds.metadata.get("dt_s") for ds in files) if v is not None}
        if len(dts) > 1:
            # Disagreement, not absence: before/after frames are not evenly spaced from
            # each other, so a single dt_s surviving the merge would misdescribe the seam.
            out = out.with_metadata(dt_s=None)

    note = "  ·  ".join(
        f"input {i}: {ax_name} {start}" if count <= 1
        else f"input {i}: {ax_name} {start}-{start + count - 1}"
        for i, (_label, start, count) in enumerate(prov.spans))
    return out.with_metadata(**{MERGE_KEY: {"note": note}})


def _concat_axis_metadata(out: Dataset, files: Tuple[Dataset, ...], axis: str) -> Dataset:
    """Concatenate the merged axis's own per-index calibration list(s) — ``PER_TIME_KEYS``
    for ``t``, ``PER_POSITION_KEYS`` for ``m`` — across every input, in input order. A key
    missing on any input, or whose list is the wrong length for THAT input's own axis
    extent, is dropped from the result whole rather than padded: a partial list would
    report some other input's timepoint/position instead of admitting it does not know
    (mirrors ``nodegraph.metadata.position_subset``'s own rule). ``z`` carries no per-index
    calibration list in this schema, so it is a no-op."""
    if axis == "t":
        keys: Tuple[str, ...] = PER_TIME_KEYS
    elif axis == "m":
        keys = PER_POSITION_KEYS
    else:
        return out
    changes: Dict[str, Any] = {}
    for key in keys:
        combined: List[Any] = []
        ok = True
        for i, ds in enumerate(files):
            n = int(getattr(ds.axes, axis))
            vals = ds.metadata.get(key)
            if key == SOURCE_FILE_KEY and vals is None:
                # Not itself a nested bundle/merge: every one of this input's own
                # positions came from this one (labelled by its input slot).
                vals = [f"input{i}"] * n
            if not isinstance(vals, (list, tuple)) or len(vals) != n:
                ok = False
                break
            combined.extend(vals)
        if ok and combined:
            changes[key] = combined
    return out.with_metadata(**changes) if changes else out


register_node(
    _compute_merge,
    op_key="util.merge", label="Merge", category="utility",
    inputs=[
        InDataset(multi=True, label="Inputs",
                 description=
                 "Two or more Datasets to merge. Order is the order you wired them in "
                 "(first-connected = input 0) — invisible elsewhere in the GUI, so check "
                 "the resolved-order note on this node's card after wiring, especially "
                 "under Time/Multipoint where which input is 'first' IS the result."),
        InFloat("offset_y", "Nudge Y", unit="um", field=False, default=0.0,
                available_in={"merge_axis": frozenset({"C"})},
                description=
                "Channel merge only. Manual correction to each later input's placement "
                "along Y, in microns of stage travel. The stage log is good to about 10 "
                "µm on an encoded stage, which at a fine pixel size is tens of pixels of "
                "visible misregistration — and here that error lands in a MEASUREMENT, "
                "because the merged channels are what a downstream node measures against "
                "each other. Applied before tile selection, so a large nudge correctly "
                "pulls in a neighbouring montage tile."),
        InFloat("offset_x", "Nudge X", unit="um", field=False, default=0.0,
                available_in={"merge_axis": frozenset({"C"})},
                description=
                "Channel merge only. Manual correction along X. See Nudge Y — same rule, "
                "same reason, and it also runs before tile selection."),
        InFloat("offset_z", "Nudge Z", unit="um", field=False, default=0.0,
                available_in={"merge_axis": frozenset({"C"})},
                description=
                "Channel merge only. Manual correction to each later input's focus, in "
                "microns. Two files focused independently can disagree in Z even when "
                "their stage XY agrees, and unlike XY there is often nothing in the image "
                "to line up against. This moves where the input's planes sit on the "
                "merged Z grid — so it changes which of its planes each output plane "
                "shows, and it can change how many planes the union grid has. POSITIVE "
                "raises that input toward higher stage Z. If the result never moves as "
                "you scroll Z, it is almost always this sign — the input has been pushed "
                "further away and is clamped to its first or last plane."),
        InFloat("t_shift", "Time shift", unit="", field=False, default=0.0,
                available_in={"merge_axis": frozenset({"C"})},
                description=
                "Channel merge only. Which timepoint of each later input pairs with each "
                "primary one: that input's frame t+shift goes with the running result's "
                "frame t. 0 pairs them in order, which is what a microscope that ran both "
                "sequences in one cycle actually did. An unpaired frame leaves that "
                "input's channels EMPTY at that timepoint rather than repeating a "
                "neighbour — a measurement must not be handed the wrong frame's pixels.\n\n"
                "INERT when an input has a single timepoint: a still is HELD across every "
                "primary frame instead."),
        InFloat("min_coverage", "Coverage floor", unit="", field=False, default=0.0,
                available_in={"merge_axis": frozenset({"C"})},
                description=
                "Channel merge only. Warn when a later input supplies less than this "
                "fraction of a field, 0 to 1. Raise it toward 1 when you are quantifying: "
                "a partly-covered field reads as zeros in the merged channel, and zeros "
                "entering a ratio or a colocalization statistic bias it without any "
                "warning of their own."),
        InBool("flip_x", "Flip X", field=False, default=True,
               available_in={"merge_axis": frozenset({"C"})},
               description=
               "Channel merge only. Whether a later input's image +x runs along stage "
               "−x. The file does not record how the camera is mounted, so for a RAW "
               "field this cannot be derived — the default matches Stitch's and "
               "Overlay's so the three can never disagree about which way is right.\n\n"
               "INERT when that input is an already-stitched canvas: the stitch answered "
               "this question to place its tiles, so flipping it again mirrors the "
               "mosaic inside its own footprint."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               available_in={"merge_axis": frozenset({"C"})},
               description=
               "Channel merge only. Whether a later input's image +y runs along stage "
               "−y. As Flip X — same default, same reason, and equally inert for an "
               "already-stitched input."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("merge_axis", list(_MERGE_AXES), default="C", label="Axis",
             description=
             "Which axis grows. This is the whole shape of the node: C reconciles files "
             "that disagree on pixel size or focus by PLACING them (resampled, never "
             "blended); T/M/Z lay inputs end to end with no resampling at all, refusing "
             "outright unless every other axis and physical scale already match.",
             choice_docs={
                 "C":
                     "Channel merge: put every input's channels on ONE channel axis, "
                     "each placed by absolute stage position and focus "
                     "(nearest-neighbour resampled if pixel sizes differ) and Z-"
                     "reconciled onto a shared grid. The only axis choice that tolerates "
                     "inputs with different pixel sizes or focus ranges — the others "
                     "refuse a mismatch instead of guessing how to reconcile it. Slowest "
                     "of the four: a channel-plane read may fan out to several of a "
                     "later input's tiles.",
                 "T":
                     "Time concatenation: input 0's timepoints, then input 1's, and so "
                     "on — frame counts simply add. The exact shape a 'before' and "
                     "'after' pair need to become one 2-frame series for a "
                     "frame-to-frame tracker (e.g. Track Objects' serialtrack method). "
                     "Refuses unless every input already shares m/z/c/y/x and physical "
                     "scale.",
                 "M":
                     "Multipoint concatenation: input 0's positions, then input 1's, "
                     "and so on — the node-graph equivalent of the ingest-time "
                     "'load as one bundle' picker, usable mid-graph after each input "
                     "has already been processed separately. Refuses unless every "
                     "input already shares t/z/c/y/x and physical scale.",
                 "Z":
                     "Z concatenation: input 0's planes, then input 1's, and so on. "
                     "Requires every input to already share the same z_step_um (there "
                     "is only one spacing on the merged stack) as well as m/t/c/y/x — "
                     "this is index concatenation, NOT the focus-based Z reconciliation "
                     "Channel merge does; two stacks focused at different absolute "
                     "heights are simply stacked in the order wired, not aligned by "
                     "focus.",
             }),
        Mode("z_grid", ["union", "primary"], default="union", label="Z grid",
             available_in={"merge_axis": frozenset({"C"})},
             description=
             "Channel merge only. Which Z planes the merged image has. This is the one "
             "place a channel merge cannot just inherit input 0's grid, because doing so "
             "would make which input happens to be first decide whether the others' "
             "stacks are reachable at all.",
             choice_docs={
                 "union":
                     "Span every input's focus range, at the finest of their Z steps — "
                     "so every plane any of them acquired is addressable. Each channel "
                     "still shows its own nearest real plane, so a coarser input repeats "
                     "a plane between the finer one's steps. The default, and it "
                     "collapses to the deepest input's own grid whenever the ranges "
                     "nest. Refused when the ranges are so far apart that the union "
                     "would be several times any input's own plane count — that is "
                     "several specimens, not one merge.",
                 "primary":
                     "Keep input 0's Z grid exactly: same plane count, spacing and "
                     "origin. What you want when a downstream measurement has to be "
                     "expressed on input 0's own planes. The cost is that any later "
                     "input's plane outside this range is unreachable — reported as a "
                     "warning, never silent.",
             }),
        Mode("unplaceable", ["refuse", "align_by_index"], default="refuse",
             label="If unplaceable", available_in={"merge_axis": frozenset({"C"})},
             description=
             "Channel merge only. What to do when a later input's metadata cannot PROVE "
             "where it belongs — no stage log (a bare TIFF), a cropped source that threw "
             "its origin away, incompatible geometry. It matters more here than in an "
             "Overlay: this output is measured, not just looked at.",
             choice_docs={
                 "refuse":
                     "Refuse, and say which piece of metadata is missing. The default, "
                     "because the alternative is a merged image that looks co-"
                     "registered, is not, and gets measured anyway.",
                 "align_by_index":
                     "Pair that input's fields BY INDEX instead — its field 0 onto the "
                     "running result's field 0 — and nudge by hand from there. Every "
                     "refusal it walks past is re-emitted as a warning marked "
                     "OVERRIDDEN so the reason survives onto the node card.",
             }),
    ],
    granularity=_MERGE_GRAN, footprint_mode="merge_axis", kernel_axes=_MERGE_KAX,
    meta_transform=_meta_merge_grow,
    description="Merge two or more Datasets by growing ONE chosen axis. 'C' places and "
                "resamples channels the way the retired Merge Channels did (handles "
                "mismatched pixel size/focus); 'T'/'M'/'Z' lay inputs end to end with no "
                "resampling, refusing unless every other axis already matches — the shape "
                "a 'before'/'after' pair needs to become one series for tracking.",
)
