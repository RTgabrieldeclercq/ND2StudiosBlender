"""Overlay (``view.overlay``) — Draw a second Dataset inside this one's field, placed by absolute stage position, pixel size and focus — so two files that ran through different graphs line up on the microscope's own coordinates."""

from __future__ import annotations


from typing import Any

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
import numpy as np

from nodegraph.metadata import overlay as _meta_overlay
from nodegraph.provider import ArrayProvider
from nodegraph.registry import (Granularity, InBool, InDataset, InFloat, InInt, InString, Mode,
                                OutDataset)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.placement_entry import handedness_for as _handedness_for
from nodegraph.catalog._shared.placement_entry import overlay_entry as _overlay_entry
from nodegraph.catalog._shared.placement_entry import overlay_settings
from nodegraph.catalog._shared.placement_entry import plan_kwargs as _plan_kwargs
from nodegraph.catalog._shared.placement_entry import z_pick as _z_pick
from nodegraph.catalog._shared.sampling import _sampling_of

# ── Overlay (view — physical co-display of two Datasets) ───────────────────────
#
# The first DISPLAY node in the engine: it changes nothing about the data and everything
# about how the Viewer draws it. Its whole job is to answer "where does the secondary go
# inside the primary's field, in microns of the real microscope" and to record that answer
# where the Viewer can read it.
#
# Why this is a graph node rather than a Viewer setting: the pairing, the nudge and the
# time shift are decisions about the EXPERIMENT, not about the window. They belong in the
# saved graph, they should diff, and they should survive being reopened on another machine.

#: Namespaced, non-calibration metadata key holding the overlay **recipe** — the ordered
#: list of sources the Viewer should composite, primary first (`wire-node-v2` §7b
#: stamp-and-inherit). A list, not a single entry, because Overlay nodes CHAIN: wiring an
#: Overlay's output into another Overlay's primary appends a third source, so N-way
#: overlay needs no N-ary socket and no bespoke layer-stack UI.
OVERLAY_KEY = "__overlay__"

#: The recipe-entry builder moved to ``_shared`` when ``channel.merge`` became a second caller
#: (2026-08-05): the catalog forbids a node module importing another, and two copies of a
#: placement record is precisely the drift the single-builder rule exists to prevent. Re-exported
#: here because the Viewer's runner and the selftest both reach for it by this name.
overlay_entry = _overlay_entry


def _overlay_of(ds: Any) -> tuple:
    """A Dataset's overlay recipe as a tuple (``()`` = not an overlay)."""
    return tuple(getattr(ds, "metadata", {}).get(OVERLAY_KEY, ()))
def _compute_overlay(ctx: EvalContext) -> Dataset:
    """Place the ``secondary`` Dataset inside the ``primary``'s field and record it.

    Resolved spec (§0 grill, 2026-07-31)
    ------------------------------------
    * **Kind** view / display → ``op_key="view.overlay"``, category ``"view"``.
    * **Data contract** the PRIMARY passes through untouched — same provider, same axes,
      same calibration, same layers — plus one namespaced metadata key. So it is NOT
      axis-changing, needs no ``meta_transform``, and every downstream node sees exactly
      what it would have seen without the overlay. That is deliberate: an overlay that
      perturbed the data would make "look at it" and "measure it" different pipelines.
    * **2D/3D** no lever. The node reads no pixels, and the secondary's dimensionality is
      inherited from ITS OWN metadata (`wire-node-v2` §7b: derive, don't ask) — a lever
      here could disagree with the data and silently mis-place a stack.
    * **Footprint** ``TILEABLE`` / ``kernel_axes=frozenset()`` — honest, because the compute
      touches no voxels at all. Placement is arithmetic over metadata.
    * **Backend** none. :mod:`nodegraph.placement`, pure numpy-free arithmetic.

    What it records, per source: which of the secondary's multipoints cover each primary
    field and by how much, the pixel-size ratio, the timepoint pairing with the absolute
    seconds between paired frames, and the axial offset. The Viewer composites from that;
    nothing here rasterizes anything.

    The refusals are the point of the node as much as the placement is. Placement is
    computed from the stage log, which describes where the CAMERA was — so it stops
    describing the DATA the moment an upstream node resamples it, and no pixel value
    reveals that. See :func:`nodegraph.placement.plan_placement` for the tiering.

    Experiment overlay (§0 grill, 2026-09-30)
    -----------------------------------------
    Grown from "one secondary file" into "every file of one experiment", still by CHAINING
    (one Overlay per added source — each keeps its own inspector, pins and label):

    * **Time rates.** ``t_pairing=rate`` pairs a source recorded at n× the primary's frame
      rate (``dt_s`` / ``frame_time_jd``, snapped to a whole ratio) — the recipe carries a
      ``t_map`` and a ``sub_ticks`` count, and the Viewer's Play all holds each primary frame
      for that many ticks so the fast source shows every one of its frames.
    * **Pins.** ``t_pins`` / ``z_pins`` are JSON ``[[primary, secondary], …]`` rows the Viewer
      writes when the user says "these frames go together"; the map runs straight through
      them (:func:`nodegraph.placement.plan_time`, :func:`~nodegraph.placement.secondary_z_weights`).
      Malformed JSON is refused here, never read as "no pins".
    * **Z interpolation.** ``z_sampling=linear`` blends the two bracketing secondary planes
      in the Viewer AND in the ``resample`` bake (one sampler, :func:`z_pick`), so looking and
      measuring cannot disagree. ``nearest`` is the old behaviour, bit for bit.
    * **Non-overlapping files.** ``unplaceable=align_centres`` places centre-on-centre at
      true pixel size — for files the stage cannot relate AND for files whose fields simply
      do not overlap. Stage placement still wins wherever any field overlaps.
    * **Label.** ``label`` names this source on the channel strip and its LUTs. Presentation:
      renaming repaints, it never re-keys the memo.
    """
    from nodegraph.placement import plan_placement

    ds = ctx.inputs[0]
    sec = ctx.input("secondary")
    if sec is None:
        raise ValueError(
            "Overlay needs a second Dataset: wire the node you want drawn over the "
            "primary into the `secondary` input. The primary stays the reference frame "
            "— it defines the canvas, the grid and the calibration.")

    modes = ctx.params.get("__modes__", {})
    # ONE reader of the node's settings, shared with the Viewer's display-time re-plan, so
    # the picture can never honour a setting the record ignored (or the other way round).
    s = overlay_settings(ctx, modes)
    offset = s["offset_um"]
    plan = plan_placement(
        ds.metadata, ds.axes, sec.metadata, sec.axes,
        # channel-blind provenance (2026-08-03): an overlay places by PHYSICAL µm, and the
        # channel axis is not a spatial axis — a tap leaves every (z,y,x) address pointing at
        # the same place and leaves the stage log describing exactly these pixels. Comparing
        # the full provenance made "overlay channel 0 on channel 1", one file, two taps, both
        # warn about geometry that never diverged AND hit the stale-stage-log refusal
        # ("its geometry has changed since the source"), which for a channel tap is false.
        dst_sampling=_sampling_of(ds, channel_axis=False),
        src_sampling=_sampling_of(sec, channel_axis=False),
        **_plan_kwargs(s))

    if plan.refusals:
        raise ValueError(
            "Overlay cannot place the secondary on the primary:\n  - "
            + "\n  - ".join(plan.refusals))

    # The recipe entry for THIS pairing. Plain JSON-shaped values only: the recipe rides in
    # `metadata`, which folds into `output_fingerprint`, so anything exotic in here would
    # make the fingerprint depend on object identity.
    # Handedness is DERIVED for an already-stitched secondary: its canvas is in stage
    # coordinates, so flipping it again mirrors the mosaic (`handedness_for`).
    flip_x, flip_y, hand_warn = _handedness_for(sec, s["flip_x"], s["flip_y"])
    entry = overlay_entry(
        ctx.node_id, plan, {**s, "flip_x": flip_x, "flip_y": flip_y},
        context=_context_boxes(ctx, ds, sec, plan, offset))
    if hand_warn:
        entry["warnings"] = list(entry.get("warnings") or ()) + list(hand_warn)
        entry["note"] = str(entry.get("note") or "") + "  ·  Flip inert (stitched secondary)"
    recipe = _overlay_of(ds) or ({"node": "primary", "role": "base"},)
    out = ds.with_metadata(**{OVERLAY_KEY: list(recipe) + [entry]})
    if modes.get("output") == "resample":
        if modes.get("canvas") == "context":
            raise ValueError(
                "Overlay cannot bake a `context` canvas: `resample` writes a real Dataset, "
                "and widening its extent past the primary's would change the grid every "
                "downstream measurement is expressed on. Use canvas=primary to bake, or "
                "output=display to look at the context.")
        out = _bake_resample(ctx, out, sec, entry)
    return out


def _context_boxes(ctx, ds, sec, plan, offset):
    """``[[m, [uy0,uy1,ux0,ux1], [py0,py1,px0,px1]], …]`` — the union and primary boxes per
    field, or ``[]`` when the canvas Mode is not asking for context (the usual case, and
    the recipe stays small)."""
    if ctx.params.get("__modes__", {}).get("canvas") != "context":
        return []
    from nodegraph.placement import context_extent
    out = []
    for m in sorted(plan.tiles):
        got = context_extent(ds.metadata, ds.axes, m, sec.metadata, sec.axes,
                             plan.tiles[m], offset_um=offset)
        if got is not None:
            u, pb = got
            out.append([m, [u.y0, u.y1, u.x0, u.x1], [pb.y0, pb.y1, pb.x0, pb.x1]])
    return out


def _bake_resample(ctx: EvalContext, ds: Dataset, sec: Dataset, entry: dict) -> Dataset:
    """Bake the placed secondary into a REAL extra channel — the ``resample`` output mode.

    In ``display`` mode the Viewer composites the secondary at draw time and nothing
    downstream ever sees it, which is right for looking and useless for measuring. This
    mode resamples the same placement into the Dataset itself, so ``analysis.measure`` can
    report the secondary's intensity inside objects the primary's chain segmented — the
    "quantify GFP per 640-detected cell" case the whole node exists to make possible.

    The SAME :func:`nodegraph.placement.compose_secondary_plane` the Viewer draws with, so
    the pixels you measure cannot drift from the picture you looked at.

    Cost is real and paid once: this reads every overlapping tile of the secondary for every
    (m, t, z) of the primary, which is why the footprint resolves to ``MULTI_VIEW`` in this
    mode and stays ``TILEABLE`` in the other.
    """
    from nodegraph.placement import compose_secondary_plane, paired_t

    prov = ds.image
    sprov = sec.image
    if prov is None or sprov is None:
        raise ValueError("Overlay's `resample` output needs an image on both inputs.")
    ax, sax = prov.axes, sprov.axes
    sec_c = int(ctx.params.get("secondary_channel", 0))
    if not (0 <= sec_c < sax.c):
        raise ValueError(
            f"secondary channel {sec_c} does not exist — the secondary has {sax.c} "
            f"channel(s), so pick 0..{sax.c - 1}.")
    tiles = dict((int(a), b) for a, b in (entry.get("tiles") or ()))

    new_axes = _replace_c(ax, ax.c + 1)
    out = np.zeros((ax.m, ax.t, ax.z, ax.c + 1, ax.y, ax.x), dtype=np.float32)
    units = ax.m * ax.t * ax.z
    done = 0
    for m in range(ax.m):
        hits = tiles.get(m) or ()
        for t in range(ax.t):
            t_sec = paired_t(entry, t)
            for z in range(ax.z):
                for c in range(ax.c):
                    out[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                if hits and t_sec is not None:
                    z_um = _z_um(ds.metadata, ax, m, z)
                    # The SAME slice choice and weights the Viewer composes with (`z_pick`):
                    # one plane for `nearest`, the two bracketing planes blended by distance
                    # for `linear`, and the user's Z pins either way.
                    blended = None
                    for z_sec, w in _z_pick(entry, sec.metadata, sax, int(hits[0][0]),
                                            ds.metadata, ax, z, z_um):

                        # `_want` (the fractional window the compositor needs) is ignored:
                        # this already reads level 0, so the whole tile IS the raw data and
                        # there is nothing a window would buy but a second code path.
                        def read_tile(j, _want=None, _t=t_sec, _z=z_sec):
                            return sprov.get_region(0, int(j), int(_t), int(_z), sec_c,
                                                    0, sax.y, 0, sax.x)

                        composed = compose_secondary_plane(
                            entry, (ax.y, ax.x), ds.metadata, ax, m,
                            sec.metadata, sax, read_tile)
                        if composed is not None:
                            part = composed * np.float32(w)
                            blended = part if blended is None else blended + part
                    if blended is not None:
                        out[m, t, z, ax.c] = blended
                done += 1
                ctx.progress(done, units, "resampling the overlay")
    # Payload and header in lockstep (§8): the same channel growth, the same dropped depth
    # and the same name list `_meta_overlay` predicted — synced from `ctx.calib`, never
    # re-derived, so a re-pull cannot grow the stack twice.
    sec_names = list(sec.metadata.get("channel_names") or [])
    label = str(sec_names[sec_c]) if sec_c < len(sec_names) and sec_names[sec_c] else "overlay"
    pri_names = list(ds.metadata.get("channel_names") or [])
    while len(pri_names) < ax.c:
        pri_names.append(f"Ch{len(pri_names) + 1}")
    return (ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes)
              .with_metadata(bit_depth=None, channel_names=pri_names + [f"ovl:{label}"]))


def _replace_c(axes, c: int):
    from dataclasses import replace as _dc_replace
    return _dc_replace(axes, c=int(c))


def _z_um(md, axes, m: int, z: int):
    """Absolute µm focus of the primary plane, for choosing the secondary's slice."""
    from nodegraph.placement import z_um_of_slice
    return z_um_of_slice(md, axes, int(m), int(z))
register_node(
    _compute_overlay, op_key="view.overlay", label="Overlay", category="view",
    inputs=[
        InDataset(),
        InDataset("secondary", label="Secondary"),
        InFloat("opacity", "Opacity", unit="", field=False, default=0.5,
                presentation=True,
                description=
                "How strongly the secondary is mixed into the picture, 0 (invisible) to 1 "
                "(full strength). Affects the DISPLAY only — no measurement anywhere "
                "downstream moves, because this node hands the primary through untouched. "
                "0.5 is a even mix; drop toward 0.2 when the secondary is a coarse "
                "context view you want to sit behind the detail rather than compete "
                "with it."),
        InString("label", "Label", field=False, default="", presentation=True,
                 description=
                 "The name this source goes by on the Viewer's channel strip, its LUTs and "
                 "the Play all readout — 'GFP 20x', 'brightfield', 'day 2' — so five "
                 "overlaid files are tellable apart without opening the graph. Blank uses "
                 "the secondary node's own name. Display only: renaming repaints, it never "
                 "re-runs anything, and the baked channel of Output 'resample' keeps the "
                 "secondary's channel name."),
        InInt("secondary_channel", "Secondary channel", unit="", field=False, default=0,
              available_in={"output": frozenset({"resample"})}, pick_kind="channel",
              description=
              "Which of the secondary's channels is baked into the output, 0-based. Only "
              "ONE is co-registered, because that is the channel you are going to measure "
              "against — and because the edit-time metadata pass is only shown the primary "
              "input, so a variable channel count could not be predicted and the axis would "
              "have to be marked unknown. Only read when Output is 'resample'."),
        InFloat("wipe_pos", "Wipe position", unit="", field=False, default=0.5,
                available_in={"blend": frozenset({"wipe"})}, presentation=True,
                description=
                "Where the wipe divider sits across the image, 0 (left edge) to 1 (right). "
                "The secondary is drawn to the LEFT of it and the primary to the right, so "
                "sliding this sweeps one image over the other with a hard seam — the "
                "quickest way to see whether two acquisitions line up, because a "
                "misalignment breaks structure exactly at the seam. Positioned in image "
                "space, so it stays on the same feature while you pan. Only read when "
                "Blend is 'wipe'."),
        InFloat("flicker_hz", "Flicker rate", unit="", field=False, default=2.0,
                available_in={"blend": frozenset({"flicker"})}, presentation=True,
                description=
                "How many times a second the overlay is blinked on and off, the classic "
                "blink comparator. Nothing is composited: the two images alternate at full "
                "strength, and your eye picks up the thing that MOVES between them — which "
                "is far more sensitive to a small shift than any static blend. 2 Hz is "
                "comfortable; raise it to make small displacements jump out, lower it to "
                "study each frame. Only read when Blend is 'flicker'."),
        InInt("t_shift", "Time shift", unit="", field=False, default=0,
              description=
              "Which secondary timepoint pairs with each primary one: secondary frame "
              "t+shift is drawn under primary frame t. 0 pairs them in order, which is "
              "what a microscope that ran both sequences in one cycle actually did. The "
              "node card reports the ABSOLUTE seconds between whichever frames you pair, "
              "read from the files' shared clock, so a wrong shift is visible rather than "
              "inferred — change this until that number is smallest if the two sequences "
              "were started at different times.\n\n"
              "INERT when the secondary has a single timepoint. A still has one answer for "
              "every primary frame and is HELD across all of them, so there is nothing to "
              "shift; the card says so rather than letting the spinner look broken. Also "
              "inert once any T pin is set — the pins decide the pairing."),
        InFloat("rate", "Rate", unit="", field=False, default=0.0,
                available_in={"t_pairing": frozenset({"rate"})},
                description=
                "How many secondary frames go by per primary frame. 0 (the default) MEASURES "
                "it from the two files' frame intervals (dt_s, or the frame clock) and snaps "
                "it to a whole ratio, so 1421.9 s against 1417.8 s is taken as the same rate "
                "rather than a 0.3% drift. Type a value to state it exactly: 4 for a 5 s "
                "series against a 20 s one (the secondary advances 4 frames per primary "
                "frame, and Play all shows all four), 0.25 the other way round (each "
                "secondary frame is held for 4 primary ones). Only read when T pairing is "
                "'rate'."),
        InString("t_pins", "T pins", field=False, default="",
                 description=
                 "Frames you have said go together, as JSON [[primary t, secondary t], …] — "
                 "normally written by the Viewer's Pin T button rather than typed. The "
                 "pairing runs straight through every pin: one pin re-anchors (it replaces "
                 "Time shift), two or more also fix the rate between them, so a series that "
                 "stalled or dropped a cycle can still be followed. A pin made in the Viewer "
                 "also records both frames' clock time, so it keeps pointing at the same "
                 "moments after an upstream crop re-numbers the frames. Pins that run "
                 "backwards are refused. Changes which secondary frame is drawn — and, under "
                 "Output 'resample', which one is MEASURED."),
        InFloat("offset_y", "Nudge Y", unit="um", field=False, default=0.0,
                pick_kind="nudge_xy", pick_bounds=("offset_y", "offset_x"),
                description=
                "Manual correction to the secondary's placement along Y, in microns of "
                "stage travel. The stage log is good to about 10 µm on an encoded stage, "
                "which at a fine pixel size is tens of pixels of visible misregistration; "
                "this is how you take it out by eye. Applied BEFORE tile selection, so a "
                "large nudge correctly pulls in a neighbouring montage tile instead of "
                "sliding the old one."),
        InFloat("offset_x", "Nudge X", unit="um", field=False, default=0.0,
                pick_kind="nudge_xy", pick_bounds=("offset_y", "offset_x"),
                description=
                "Manual correction to the secondary's placement along X, in microns of "
                "stage travel. See Nudge Y — same rule, same reason, and it also runs "
                "before tile selection. The Pick button sets both from two clicks: a "
                "feature in the primary, then the same feature in the overlaid source — "
                "the quickest way to anchor two files the stage cannot relate "
                "(If unplaceable = align_centres)."),
        InFloat("offset_z", "Nudge Z", unit="um", field=False, default=0.0,
                description=
                "Manual correction to the secondary's focus, in microns. Two files "
                "focused independently can disagree in Z even when their stage XY agrees, "
                "and unlike XY there is often nothing in the image to line up against. "
                "Moves only where the secondary is reported to sit relative to the "
                "primary's stack; it never restacks the primary. POSITIVE raises the SECONDARY toward higher stage Z, so the sign depends on which input you wired where: bring a deeper secondary UP with a positive value, a shallower one DOWN with a negative. Wiring the same two files the other way round FLIPS the sign you need. If the overlay never moves as you scroll Z, it is almost always this sign - the secondary has been pushed further away and is clamped to its first or last plane. INERT once any Z pin is set."),
        InString("z_pins", "Z pins", field=False, default="",
                 description=
                 "Planes you have said go together, as JSON [[primary z, secondary z], …] — "
                 "normally written by the Viewer's Pin Z button. The answer to two files "
                 "focused independently, where no stage Z relates them: one pin says 'this "
                 "plane is that plane' (it replaces Nudge Z), two or more also fix the axial "
                 "scale between them — which is what a refractive-index mismatch between two "
                 "objectives looks like. A pin made in the Viewer records both planes' "
                 "absolute focus, so it survives an upstream Z crop. Changes which secondary "
                 "plane is drawn — and, under Output 'resample', which one is MEASURED."),
        InBool("flip_x", "Flip X", field=False, default=True,
               description=
               "Whether image +x runs along stage −x. The file does NOT record how the "
               "camera is mounted, so this cannot be derived — it is a property of the "
               "scope. The default matches Stitch's, so a mosaic and an overlay of the "
               "same file can never disagree about which way is right. If a stitched "
               "montage from this scope comes out mirrored, flip it here too."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               description=
               "Whether image +y runs along stage −y. As Flip X, and defaulted to match "
               "Stitch for the same reason. Handedness affects only which way the "
               "secondary's pixels are sampled — never which tiles are chosen, since a "
               "mirrored field still occupies the same patch of stage."),
        InFloat("min_coverage", "Coverage floor", unit="", field=False, default=0.0,
                description=
                "Warn when the secondary supplies less than this fraction of a primary "
                "field, 0 to 1. 0 (the default) warns only about fields with a genuine "
                "hole. Raise it toward 1 when you are quantifying and a partly-covered "
                "field would bias whatever you compare — the warning names the fields, so "
                "you can drop them rather than discover the gap in the numbers."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("blend",
                ["add", "over", "difference", "checkerboard", "wipe", "flicker"],
                default="add", label="Blend",
                description=
                "How the two images are composited where they overlap. Three are continuous "
                "mixes and three are COMPARATORS that separate the sources in space or in time "
                "— the latter are what you want when the question is \"do these line up\", "
                "because a blend hides exactly the small misregistration you are hunting.",
                choice_docs={
                    "add":
                        "Sum the two, scaled by Opacity. Nothing is hidden — both images "
                        "contribute everywhere — which makes it the best general view of "
                        "co-localization; bright regions can saturate to white and lose "
                        "structure. The default.",
                    "over":
                        "Alpha-composite the secondary ON TOP, at Opacity. What you want when "
                        "the secondary is the subject and the primary is background context; the "
                        "primary is progressively hidden as opacity rises, rather than being "
                        "added to.",
                    "difference":
                        "Absolute difference of the two. Aligned structure cancels to BLACK and "
                        "anything that disagrees glows — the most sensitive registration check "
                        "here, and also the most sensitive to a brightness mismatch between the "
                        "two sources, which lights up everything.",
                    "checkerboard":
                        "Alternate square tiles from each source. A classic registration check: "
                        "a feature crossing a tile boundary either continues or steps, so a "
                        "sub-pixel shift becomes visible as a break — and neither image is "
                        "altered inside its own tiles.",
                    "wipe":
                        "Show the primary on one side of a movable divider and the secondary on "
                        "the other (Wipe position). Best for comparing a large-scale change "
                        "across the same field; drag the divider through a feature to see it "
                        "match or jump.",
                    "flicker":
                        "Alternate the two whole images in time, at Flicker rate. The eye is "
                        "extremely good at spotting what MOVES between two flashing frames, so "
                        "this catches shifts the static blends hide — at the cost of never "
                        "showing both at once, and of being unusable for a screenshot.",
                }),
           Mode("output", ["display", "resample"], default="display", label="Output",
                description=
                "Whether this node only RECORDS the placement for the Viewer, or actually bakes "
                "the secondary's pixels into the output Dataset. It is also this node's cost "
                "switch: display touches no voxel at all, resample reads every overlapping tile "
                "of the secondary for each primary frame.",
                choice_docs={
                    "display":
                        "Record the placement as metadata and pass the primary through "
                        "untouched. The Viewer composites from that recipe, so looking and "
                        "measuring stay the same pipeline and nothing downstream sees a modified "
                        "image. Free, and the default; Overlay nodes also chain, so a second one "
                        "adds a third source.",
                    "resample":
                        "Resample the chosen secondary channel onto the primary's grid and write "
                        "it as a real extra channel. The option to pick when the co-registered "
                        "pixels must be MEASURED — a ratio, a colocalization statistic, an "
                        "export — and the expensive one, since it genuinely reads and "
                        "interpolates the secondary.",
                }),
           Mode("unplaceable", ["refuse", "align_by_index", "align_centres"], default="refuse",
                label="If unplaceable",
                description=
                "What to do when the files cannot PROVE where the secondary belongs — no stage "
                "log (a bare TIFF), a cropped source that threw its origin away, incompatible "
                "geometry — or, for align_centres, when they can but do not overlap at all. "
                "The refusals are about what the metadata can establish, not about what is "
                "true, so there has to be a way to override them; it must not be the quiet "
                "default.",
                choice_docs={
                    "refuse":
                        "Refuse to place it, and say which piece of metadata is missing. The "
                        "default, because the alternative to an error here is an image that "
                        "looks plausibly aligned and is not — and a wrong overlay is read as "
                        "evidence.",
                    "align_by_index":
                        "Pair the fields BY INDEX instead — secondary field 0 onto primary field "
                        "0 — and nudge by hand from there. Honest about what it is doing: it "
                        "never fabricates a stage position, and every refusal it walks past is "
                        "re-emitted as a warning marked OVERRIDDEN so the reason survives onto "
                        "the node card. The secondary is STRETCHED onto the primary's field, "
                        "whatever the two pixel sizes.",
                    "align_centres":
                        "Place the secondary CENTRE ON CENTRE at each file's true pixel size — "
                        "a 60x field sits at its real size inside a 20x one instead of being "
                        "stretched over it. Used both when the files cannot prove a placement "
                        "and when they can but no field overlaps (two wells, two dishes), "
                        "which would otherwise draw nothing; files that overlap anywhere stay "
                        "stage-placed. Anchor the result with the µm nudge or the two-click "
                        "Nudge pick. Needs a pixel size on both, else falls back to index.",
                }),
           Mode("t_pairing", ["index", "rate"], default="index", label="T pairing",
                description=
                "How the secondary's timepoints follow the primary's. Either way the choice is "
                "explicit and reported — never a nearest-in-time search, which re-pairs frames "
                "whenever two loops drift a second apart. T pins, when set, override both.",
                choice_docs={
                    "index":
                        "Frame t+Time shift of the secondary under frame t of the primary — "
                        "what a microscope that ran both sequences in one cycle actually did. "
                        "Assumes the two run at the same rate; a 4x-faster source would be "
                        "shown at a quarter of its frames and run out early. The default.",
                    "rate":
                        "The secondary advances Rate frames per primary frame, measured from "
                        "the files' frame intervals unless you state it — for a fast series "
                        "against a slow one. Play all then plays the faster source n times "
                        "per primary frame so every one of its frames is shown. Refused when "
                        "neither file carries a clock and Rate is 0.",
                }),
           Mode("z_sampling", ["nearest", "linear"], default="nearest", label="Z sampling",
                description=
                "Which secondary plane(s) are drawn — and baked, under Output 'resample' — for "
                "each primary plane when the two stacks were taken at different Z steps.",
                choice_docs={
                    "nearest":
                        "The single secondary plane nearest in absolute focus. Every pixel is "
                        "one the microscope actually acquired; a coarser stack steps a whole "
                        "plane at a time as you scroll the finer one. The default.",
                    "linear":
                        "Blend the two bracketing secondary planes by distance, so a coarse "
                        "stack scrolls smoothly under a fine one. The blended planes were "
                        "never imaged as such — under 'resample' the MEASURED intensities are "
                        "interpolated, which blurs axially thin structure. Clamps to the end "
                        "plane beyond the stack rather than extrapolating.",
                }),
           # DISPLAY-only, and gated away from `resample` rather than merely ignored there:
           # a baked Dataset whose extent is the union rather than the primary's would
           # silently change the grid every downstream measurement is expressed on, which
           # is exactly what the primary-defines-the-canvas rule exists to prevent.
           Mode("canvas", ["primary", "context"], default="primary", label="Canvas",
                available_in={"output": frozenset({"display"})},
                description=
                "How much of the world the Viewer shows: only the primary's own field, or enough "
                "to include the secondary's extent around it. Display-only — a baked Dataset "
                "always uses the primary's grid, since changing the extent would change the "
                "coordinates every downstream measurement is expressed on.",
                choice_docs={
                    "primary":
                        "The canvas is exactly the primary's field. What the graph measures and "
                        "what you see cover the same area, so nothing on screen is outside the "
                        "data being analysed. The default.",
                    "context":
                        "Widen the view to include the secondary's extent, drawn as context "
                        "around the primary's field. For seeing where a high-magnification field "
                        "sits inside a whole-well overview — the primary's pixels and every "
                        "measurement are unchanged, only the visible frame grows. Refused under "
                        "`resample`, which writes a real Dataset.",
                })],
    # Footprint keyed on `output`, not on a dim lever: `display` touches no voxel at all,
    # while `resample` reads every overlapping tile of the SECONDARY for each primary
    # frame. Declaring one number for both would either over-promise the cheap path or
    # under-declare the expensive one, and the scheduler routes on it.
    footprint_mode="output",
    granularity={"display": Granularity.TILEABLE,
                 "resample": Granularity.MULTI_VIEW},
    kernel_axes={"display": frozenset(),
                 "resample": frozenset({"m", "y", "x"})},
    meta_transform=_meta_overlay,
    description="Draw a second Dataset inside this one's field, placed by absolute stage "
                "position, pixel size and focus — so two files that ran through different "
                "graphs line up on the microscope's own coordinates. The primary defines "
                "the canvas and passes through untouched; the secondary is matched tile by "
                "tile, and the pairing, coverage, time and focus offsets are reported "
                "rather than assumed. Chain Overlays for a third source.")
