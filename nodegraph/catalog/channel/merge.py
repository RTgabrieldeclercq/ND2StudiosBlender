"""Merge Channels (``channel.merge``) — Put two acquisitions on ONE channel axis, placed by absolute stage position and focus, so two files become one multi-channel image."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import merge_channels as _meta_merge_channels
from nodegraph.provider import ChannelMergeProvider
from nodegraph.registry import (Granularity, InBool, InDataset, InFloat, Mode, OutDataset)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.placement_entry import handedness_for, overlay_entry
from nodegraph.catalog._shared.sampling import _sampling_of

# ── Merge Channels (two files → one multi-channel Dataset, placed by metadata) ────
#
# The difference between this and `view.overlay` is what the RESULT is, not how it looks.
# An Overlay records where the secondary goes and lets the Viewer composite it: perfect for
# looking, and nothing downstream ever sees the second file. This node returns one Dataset
# carrying both, so every node after it — measure, segment, export, a Dock bake — treats the
# two acquisitions as channels of one image, which is what they physically are.
#
# Asked for 2026-08-05: "no need of an overlay, but instead a true metadata/image overlay",
# with each channel walking its own Z planes because the two are registered by metadata
# rather than by index.


def _compute_merge_channels(ctx: EvalContext) -> Dataset:
    """Place ``secondary``'s channels beside this Dataset's on one channel axis.

    Resolved spec (§0 grill, 2026-08-05)
    ------------------------------------
    * **Kind** utility / channel → ``op_key="channel.merge"``, category ``"channel"``,
      beside ``channel.split`` and ``channel.select``.
    * **Data contract** axis-changing in TWO axes: ``c`` grows by the secondary's channel
      count and ``z`` becomes the merged grid. Both are UNKNOWN at edit time because
      ``propagate_meta`` shows a transform only the primary's envelope — see
      :func:`nodegraph.metadata.merge_channels`.
    * **2D/3D** no lever. The node runs no kernel: it is placement arithmetic plus
      nearest-neighbour sampling, and the dimensionality of each side is a property of ITS
      metadata (`wire-node-v2` §7b — derive, don't ask).
    * **Footprint** ``MULTI_VIEW`` / ``kernel_axes={"m","y","x"}``: one output plane of a
      secondary channel reads every one of the secondary's fields that covers this field.
      The primary's own channels are a pass-through, but a footprint declares the worst case
      the scheduler must route for.
    * **Backend** none. :mod:`nodegraph.placement` for the geometry,
      :class:`~nodegraph.provider.ChannelMergeProvider` for the lazy reads.

    **Z is per channel, and nothing is interpolated.** Each output plane has an absolute µm
    focus; each side answers with its OWN plane nearest that focus. So scrolling Z walks real
    acquired planes on both channels rather than one file's planes and the other's guesses —
    which is the whole point of merging by metadata instead of by index.

    **Lateral placement IS a resampling**, because two files rarely share a pixel size (the
    WellA3 pair are 0.287 and 1.718 µm/px). It is nearest-neighbour, so every value in the
    output is a real sample of its source file, never a blend.

    The refusals are inherited wholesale from :func:`nodegraph.placement.plan_placement`, and
    they are the point as much as the placement is: a merged image that looks co-registered and
    is not gets measured, and a measurement is believed.
    """
    from nodegraph.placement import merge_z_grid, plan_placement

    ds = ctx.inputs[0]
    sec = ctx.input("secondary")
    if sec is None:
        raise ValueError(
            "Merge Channels needs a second Dataset: wire the file whose channels you want "
            "added into the `secondary` input. The primary defines the lateral grid — its "
            "pixel size, its extent and its calibration are what the result is expressed on.")
    if ds.image is None or sec.image is None:
        raise ValueError("Merge Channels needs an image on both inputs.")

    modes = ctx.params.get("__modes__", {})
    offset = (float(ctx.params.get("offset_z", 0.0)),
              float(ctx.params.get("offset_y", 0.0)),
              float(ctx.params.get("offset_x", 0.0)))
    plan = plan_placement(
        ds.metadata, ds.axes, sec.metadata, sec.axes,
        t_shift=int(ctx.params.get("t_shift", 0)), offset_um=offset,
        # Channel-blind, exactly as `view.overlay` is: a channel axis is not a spatial one, so a
        # channel tap upstream leaves every (z,y,x) address pointing at the same place.
        dst_sampling=_sampling_of(ds, channel_axis=False),
        src_sampling=_sampling_of(sec, channel_axis=False),
        min_coverage=float(ctx.params.get("min_coverage", 0.0)),
        on_unplaceable=modes.get("unplaceable", "refuse"))
    if plan.refusals:
        raise ValueError("Merge Channels cannot place the secondary on the primary:\n  - "
                         + "\n  - ".join(plan.refusals))

    grid, z_warn, z_refuse = merge_z_grid(
        ds.metadata, ds.axes, sec.metadata, sec.axes, plan.tiles,
        mode=modes.get("z_grid", "union"), dz=offset[0])
    if z_refuse:
        raise ValueError("Merge Channels cannot build a Z grid for these two:\n  - "
                         + "\n  - ".join(z_refuse))

    ax, sax = ds.axes, sec.axes
    # Handedness is DERIVED, not defaulted: an already-stitched secondary is in stage
    # coordinates and flipping it again mirrors the mosaic — see `handedness_for`.
    flip_x, flip_y, hand_warn = handedness_for(
        sec, ctx.params.get("flip_x", True), ctx.params.get("flip_y", False))
    entry = _merge_entry(ctx, plan, offset, flip_x, flip_y)
    new_axes = replace(ax, c=ax.c + sax.c, z=grid.n)
    prov = ChannelMergeProvider(
        ds.image, sec.image, axes=new_axes, entry=entry, grid=grid,
        pri_md=dict(ds.metadata), pri_axes=ax,
        sec_md=dict(sec.metadata), sec_axes=sax)

    names = list(ds.metadata.get("channel_names") or [])
    while len(names) < ax.c:
        names.append(f"Ch{len(names) + 1}")
    snames = list(sec.metadata.get("channel_names") or [])
    for k in range(sax.c):
        label = str(snames[k]) if k < len(snames) and snames[k] else f"Ch{k + 1}"
        names.append(label if label not in names else f"{label} (2)")
    # `bit_depth` gets the same treatment as `z_step_um` below, and for the same reason:
    # `metadata.merge_channels` drops it because that pass is handed only the PRIMARY
    # envelope and "asserting the primary's depth over the pair would be a claim about data
    # it never read" — but THIS layer has both files, so it can check instead of claim. When
    # the two agree the pair really is on one integer scale and the depth survives; when they
    # disagree (or either is silent) absent stays the honest signal.
    #
    # It matters because absent is not neutral downstream. `Viewer._display_range` falls back
    # to the dtype's full range for an integer plane and to the observed range for a float
    # one — and a merged payload is BOTH (`ChannelMergeProvider.read_region` passes the
    # primary through in its own dtype and composes the secondary in float32). So a merged
    # 12-bit pair rendered with channel 0 on a 0-65535 slider, i.e. almost black, beside
    # channel 1 auto-stretched to fit. With the depth restated both land on 0-4095.
    pri_bd, sec_bd = ds.metadata.get("bit_depth"), sec.metadata.get("bit_depth")
    merged_bd = int(pri_bd) if (pri_bd and sec_bd and int(pri_bd) == int(sec_bd)) else None
    # Payload and header in lockstep (§8): the same two grown axes the meta_transform
    # predicted as unknown, plus the Z spacing it deliberately dropped — re-stated here
    # because THIS is the layer that could compute it.
    return (ds.with_image(prov).reshaped_axes(new_axes)
              .with_metadata(bit_depth=merged_bd, channel_names=names,
                             z_step_um=(float(grid.step_um) if grid.n > 1 else None),
                             origin_um=_merged_origins(ds, grid),
                             **{MERGE_KEY: _merge_note(plan, grid, sax,
                                                       list(z_warn) + list(hand_warn))}))


#: Namespaced, non-calibration record of what the merge did — the note the node card and the
#: Viewer's status line read (`wire-node-v2` §7b stamp-and-inherit).
MERGE_KEY = "__merge__"


def _merge_note(plan, grid, sax, z_warn) -> Dict[str, Any]:
    return {"placed_by": plan.placed_by,
            "z_planes": int(grid.n), "z_step_um": float(grid.step_um),
            "added_channels": int(sax.c),
            "coverage": [[m, round(plan.coverage[m], 6)] for m in sorted(plan.coverage)],
            "warnings": list(plan.warnings) + list(z_warn),
            "note": plan.describe(0) + f"  ·  +{sax.c} channel(s)  ·  {grid.n} z "
                                       f"@ {grid.step_um:g} um"}


def _merged_origins(ds: Dataset, grid) -> Any:
    """``origin_um`` with each field's Z corner moved to the merged grid's first plane.

    The lateral corner is untouched — the primary defines the grid — but the axial one moves
    whenever the union reaches above the primary's own stack, and ``origin_um`` is the
    transform-maintained corner every downstream µm conversion reads. Leaving it stale is how a
    merged stack reports the right pixels at the wrong depth."""
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


def _merge_entry(ctx: EvalContext, plan, offset, flip_x: bool, flip_y: bool) -> Dict[str, Any]:
    """The placement record the provider samples through — the same JSON-shaped entry
    ``view.overlay`` stamps, from the one shared builder so the two cannot drift."""
    return overlay_entry(
        ctx.node_id, plan,
        {"blend": "add",                     # unread here: a merge composites nothing
         "flip_x": flip_x, "flip_y": flip_y,
         "t_shift": int(ctx.params.get("t_shift", 0)), "offset_um": offset})


register_node(
    _compute_merge_channels,
    op_key="channel.merge", label="Merge Channels", category="channel",
    inputs=[
        InDataset(),
        InDataset("secondary", label="Secondary"),
        InFloat("offset_y", "Nudge Y", unit="um", field=False, default=0.0,
                description=
                "Manual correction to the secondary's placement along Y, in microns of stage "
                "travel. The stage log is good to about 10 µm on an encoded stage, which at a "
                "fine pixel size is tens of pixels of visible misregistration — and here that "
                "error lands in a MEASUREMENT, because the merged channels are what a "
                "downstream node measures against each other. Applied before tile selection, so "
                "a large nudge correctly pulls in a neighbouring montage tile."),
        InFloat("offset_x", "Nudge X", unit="um", field=False, default=0.0,
                description=
                "Manual correction to the secondary's placement along X, in microns of stage "
                "travel. See Nudge Y — same rule, same reason, and it also runs before tile "
                "selection."),
        InFloat("offset_z", "Nudge Z", unit="um", field=False, default=0.0,
                description=
                "Manual correction to the secondary's focus, in microns. Two files focused "
                "independently can disagree in Z even when their stage XY agrees, and unlike XY "
                "there is often nothing in the image to line up against. This moves where the "
                "secondary's planes sit on the merged Z grid — so it changes which of its planes "
                "each output plane shows, and it can change how many planes the union grid has."),
        InFloat("t_shift", "Time shift", unit="", field=False, default=0.0,
                description=
                "Which secondary timepoint pairs with each primary one: secondary frame t+shift "
                "goes with primary frame t. 0 pairs them in order, which is what a microscope "
                "that ran both sequences in one cycle actually did. An unpaired frame leaves the "
                "secondary's channels EMPTY at that timepoint rather than repeating a "
                "neighbour — a measurement must not be handed the wrong frame's pixels."),
        InFloat("min_coverage", "Coverage floor", unit="", field=False, default=0.0,
                description=
                "Warn when the secondary supplies less than this fraction of a primary field, 0 "
                "to 1. Raise it toward 1 when you are quantifying: a partly-covered field reads "
                "as zeros in the merged channel, and zeros entering a ratio or a colocalization "
                "statistic bias it without any warning of their own. The message names the "
                "fields, so you can drop them rather than find the gap in the numbers."),
        InBool("flip_x", "Flip X", field=False, default=True,
               description=
               "Whether the secondary's image +x runs along stage −x. The file does not record "
               "how the camera is mounted, so for a RAW field this cannot be derived — it is a "
               "property of the scope, and the default matches Stitch's and Overlay's so the "
               "three can never disagree about which way is right.\n\n"
               "INERT when the secondary is an already-stitched canvas: the stitch answered this "
               "question to place its tiles, so its canvas is in stage coordinates and flipping "
               "it again mirrors the mosaic inside its own footprint (measured 784 µm — 456 px — "
               "out on the WellA3 pair). The node detects that and says so on its card."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               description=
               "Whether the secondary's image +y runs along stage −y. As Flip X — same default, "
               "same reason, and equally inert for an already-stitched secondary. Handedness "
               "affects only which way the secondary's pixels are sampled, never which tiles are "
               "chosen, since a mirrored field still occupies the same patch of stage."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("z_grid", ["union", "primary"], default="union", label="Z grid",
                description=
                "Which Z planes the merged image has. This is the one place a merge cannot just "
                "inherit the primary's grid, because doing so would make which file you happened "
                "to wire as the primary decide whether the other's stack is reachable at all.",
                choice_docs={
                    "union":
                        "Span both files' focus ranges, at the finer of their two Z steps — so "
                        "every plane either microscope acquired is addressable, whichever input "
                        "is the primary. Each channel still shows its own nearest real plane, so "
                        "the coarser file repeats a plane between the finer one's steps (it has "
                        "none of its own to show there). The default, and it collapses to the "
                        "deeper file's own grid whenever one range sits inside the other. "
                        "Refused when the two ranges are so far apart that the union would be "
                        "several times either input's plane count — that is two specimens, not "
                        "one merge.",
                    "primary":
                        "Keep this input's Z grid exactly: same plane count, same spacing, same "
                        "origin. What you want when a downstream measurement has to be expressed "
                        "on the primary's own planes, or when a saved graph must keep addressing "
                        "them. The cost is that any secondary plane outside this range is "
                        "unreachable — reported as a warning naming the fields, never silent.",
                }),
           Mode("unplaceable", ["refuse", "align_by_index"], default="refuse",
                label="If unplaceable",
                description=
                "What to do when the files cannot PROVE where the secondary belongs — no stage "
                "log (a bare TIFF), a cropped source that threw its origin away, incompatible "
                "geometry. The refusals are about what the metadata can establish, not about "
                "what is true, so there has to be a way past them; it must not be the quiet "
                "default. It matters more here than in an Overlay: this output is measured, not "
                "just looked at.",
                choice_docs={
                    "refuse":
                        "Refuse, and say which piece of metadata is missing. The default, "
                        "because the alternative is a merged image that looks co-registered, is "
                        "not, and gets measured anyway — and a number is believed in a way a "
                        "picture is not.",
                    "align_by_index":
                        "Pair the fields BY INDEX instead — secondary field 0 onto primary field "
                        "0 — and nudge by hand from there. Honest about what it is doing: it "
                        "never fabricates a stage position, and every refusal it walks past is "
                        "re-emitted as a warning marked OVERRIDDEN so the reason survives onto "
                        "the node card.",
                })],
    granularity=Granularity.MULTI_VIEW,
    kernel_axes=frozenset({"m", "y", "x"}),
    meta_transform=_meta_merge_channels,
    description="Put two acquisitions on ONE channel axis, placed by absolute stage position "
                "and focus — so two files that ran through different graphs become a single "
                "multi-channel image every downstream node can measure across. Unlike Overlay "
                "(which records a placement for the Viewer and changes nothing a node reads), "
                "this returns real merged data, and it is LAZY: a plane costs a plane. Each "
                "channel shows its OWN plane nearest the viewed focus, so scrolling Z walks "
                "acquired planes on both sides rather than interpolating one to fit the other.",
)
