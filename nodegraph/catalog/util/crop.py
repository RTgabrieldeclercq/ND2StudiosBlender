"""Crop (``util.crop``) — Crop Y,X (and Z in 3D mode), or keep only chosen M/T/Z frames."""

from __future__ import annotations


from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (crop as _meta_crop, format_indices, frame_spec_picks,
                                position_subset, respaced, shift_origin_um, time_subset,
                                z_home_after)
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import (DimMode, InDataset, InInt, InString, Mode, OutDataset)
from nodegraph.streaming import WindowView

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN
from nodegraph.catalog._shared.frame_subset import (AXIS_NOUN, FRAME_AXES,
                                                    subset_lattice_layers,
                                                    subset_structure_rows)
from nodegraph.catalog._shared.sampling import Z_STAMP, _sampled


# ── Crop (axis-changing: shrink Y,X and, in 3D, Z — or narrow M/T/Z by index) ────

def _compute_crop(ctx: EvalContext) -> Dataset:
    """Crop the spatial extent (and, in 3D mode, the Z range), or — in ``frames`` mode —
    keep only the chosen M / T / Z indices. Its ``crop`` meta_transform tracks the new
    extent at edit time; pixel size is preserved (origin is deferred, V2.00 §16)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("crop needs an image provider on its input Dataset")
    ax = prov.axes

    # ── frames mode: a SUBSET of m/t/z, not a window (V2.27) ──────────────────
    # The param is read here rather than inside the helper so the socket-contract gate sees
    # the literal key against this compute (`selftest::_param_key_index` resolves forwarded
    # KEYS, not reads that happen a call deeper).
    if ctx.params.get("__modes__", {}).get("region") == "frames":
        return _crop_frames(ctx, ds, prov, ctx.params.get("frames"))

    def bound(v, default, hi):
        return max(0, min(int(v) if v is not None else default, hi))

    y0, y1 = bound(ctx.params.get("y0"), 0, ax.y), bound(ctx.params.get("y1"), ax.y, ax.y)
    x0, x1 = bound(ctx.params.get("x0"), 0, ax.x), bound(ctx.params.get("x1"), ax.x, ax.x)
    if ctx.is_volume:
        z0 = bound(ctx.params.get("z0"), 0, ax.z)
        z1 = bound(ctx.params.get("z1"), ax.z, ax.z)
    else:
        z0, z1 = 0, ax.z
    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        raise ValueError(f"crop produced an empty region "
                         f"(y[{y0}:{y1}] x[{x0}:{x1}] z[{z0}:{z1}])")
    ny, nx, nz = y1 - y0, x1 - x0, z1 - z0
    new_axes = replace(ax, z=nz, y=ny, x=nx)
    # C1: a pure lazy offset view (the _ChannelView pattern) — no pixels move, the
    # source dtype is preserved, and a kernel op downstream clips its halo at THIS
    # view's extents (= the eager reflect-at-crop-edge behavior, V2.04 §6b).
    view = WindowView(prov, z0=z0, y0=y0, x0=x0, axes=new_axes)
    # A crop that keeps the FULL lateral extent only cuts along z, so every (y,x) address
    # still points at the same physical location — the `z:` marker says so, and a consumer
    # comparing two z==1 branches then drops it (`_sampling_of`). This is what lets a
    # single-plane z-crop of one channel be read against a Z-PROJECTION of another: both
    # collapse z, neither moves anything laterally. The moment y or x is windowed the corner
    # MOVES, and the stamp must stay comparable or a voxel-for-voxel consumer would read
    # every object at an offset.
    lateral_identity = (y0 == 0 and y1 == ax.y and x0 == 0 and x1 == ax.x)
    out = _sampled(ds.with_image(view).reshaped_axes(new_axes),
                   f"{Z_STAMP if lateral_identity else ''}"
                   f"crop[z{z0}:{z1},y{y0}:{y1},x{x0}:{x1}]")
    # A crop MOVES the field's corner, so the payload must carry the moved origin or it
    # would disagree with the header the meta_transform already predicted (§8). SYNCED
    # from `ctx.calib`, never re-derived here: the env has already had `_meta_crop` applied,
    # so recomputing the shift would apply the cut twice.
    origin = ctx.calib("origin_um")
    return out.with_metadata(origin_um=origin) if origin is not None else out




def _check_frame_picks(picks: Dict[str, Optional[Tuple[int, ...]]], raw: Any,
                       sizes: Dict[str, int]) -> None:
    """Refuse a frame spec that names an axis and keeps NOTHING on it.

    The reason ``channel.select`` refuses an empty channel list: a zero-length axis is a
    degenerate payload nothing downstream checks, so it travels until something indexes into
    it and reports the failure from inside a library, naming neither this node nor the param.
    The advisory ``meta_transform`` deliberately does NOT refuse — it re-runs on every
    keystroke and would otherwise error while somebody types the second digit of "t12"."""
    for axis, kept in picks.items():
        if kept != ():
            continue
        n, noun = sizes[axis], AXIS_NOUN[axis]
        raise ValueError(
            f"util.crop: `frames` = {raw!r} keeps no {noun} — this Dataset has {n}, so the "
            f"only valid {axis} indices are {'0' if n == 1 else f'0..{n - 1}'}. Ranges are "
            f"inclusive (\"{axis}0-{max(0, n - 1)}\" is all of them) and negative values are "
            f"simply out of range, not 'from the end'. Drop the `{axis}` section to keep "
            f"every {noun}, or clear the socket to keep every frame.")


def _crop_frames(ctx: EvalContext, ds: Dataset, prov: Any, raw: Any) -> Dataset:
    """The ``frames`` mode payload: narrow M/T/Z to the selected indices.

    Lazy, like the spatial half — :class:`~nodegraph.provider.FrameSubsetProvider` is a pure
    index remap, so no pixel is read or copied and the source dtype survives. It is the same
    view the GUI's troubleshooting scope seeds a graph with; using it here rather than a
    second implementation is what keeps "run on the frames I picked" and "crop to the frames
    I picked" the same operation with the same addressing.

    Everything that rides ALONGSIDE the image is carried too, and each half needs a
    different rule:

    * **lattice layers** (a mask, a per-plane statistic) are indexed on the same axes as the
      image, so they are subset with it (:func:`_subset_lattice_layers`) instead of being
      dropped by ``reshaped_axes`` for no longer fitting;
    * **structure rows** (Point / Label / Track) carry ``m``/``t``/``z`` ADDRESSES, so rows
      on a dropped frame are dropped and the survivors are renumbered
      (:func:`_subset_structure_rows`) — a row still claiming ``t=9`` in a 3-frame output is
      an out-of-range read waiting to happen;
    * **calibration** is taken from the envelope, which the ``crop`` meta_transform has
      already transformed; the non-calibration per-axis lists are narrowed here with the
      SAME shared helpers that transform used, so the two cannot drift.

    Nothing asked for is a genuine no-op: the input is returned untouched rather than
    wrapped in an identity view, so an unconfigured node in a graph costs nothing and stamps
    nothing (the same choice ``util.zproject``'s ``none`` method makes)."""
    ax = prov.axes
    keep = frame_spec_picks(raw, ax.m, ax.t, ax.z)
    _check_frame_picks(keep, raw, {"m": ax.m, "t": ax.t, "z": ax.z})
    ms, ts, zs = keep["m"], keep["t"], keep["z"]
    if ms is None and ts is None and zs is None:
        return ds
    keep_m = ms if ms is not None else tuple(range(ax.m))
    keep_t = ts if ts is not None else tuple(range(ax.t))
    keep_z = zs if zs is not None else tuple(range(ax.z))
    new_axes = replace(ax, m=len(keep_m), t=len(keep_t), z=len(keep_z))
    # `zs=None` (not the full range) when z is untouched: the provider then leaves z alone
    # entirely and keeps the fingerprint it would have had without a z pick.
    view = FrameSubsetProvider(prov, keep_m, keep_t, zs if zs is not None else None)

    out = subset_lattice_layers(ds.with_image(view), new_axes, keep)
    out = subset_structure_rows(out, keep)

    # ── the metadata that is indexed BY one of these axes ─────────────────────
    md = ds.metadata
    changes: Dict[str, Any] = {}
    if ms is not None:
        changes.update(position_subset(md, keep_m))
    if ts is not None:
        changes.update(time_subset(md, keep_t))
        changes["dt_s"] = respaced(md.get("dt_s"), keep_t)
    if zs is not None:
        changes["z_step_um"] = respaced(md.get("z_step_um"), keep_z)
        changes.update(z_home_after(md, keep_z))
    out = out.with_metadata(**changes)
    z_step = md.get("z_step_um")
    if zs is not None and z_step and keep_z[0]:
        # Cutting planes off the BOTTOM moves the field's corner, exactly as a y/x window
        # does. Applied to `out`, whose axes and origin list are already narrowed —
        # `read_origin_um` validates the list against `axes.m` and would reject it on the
        # pre-subset axes. The SOURCE step is the right multiplier: the corner moved by the
        # real planes that were dropped, not by the re-spaced ones that remain.
        try:
            dz = float(z_step) * int(keep_z[0])
        except (TypeError, ValueError):
            dz = 0.0
        if dz:
            out = out.with_metadata(**shift_origin_um(out, dz, 0.0, 0.0))
    # ...and the envelope wins for the three CALIBRATION keys, so it stays the single source
    # of truth and the reads stay memo-fenced (`channel.select` does the same for its
    # emission list). The local computation above is the fallback for an UNSEEDED source,
    # whose envelope carries nothing to prefer; it cannot disagree with the transform,
    # because it is the same `respaced` / `position_subset` / `shift_origin_um` applied to
    # the same input values.
    for key in ("dt_s", "z_step_um", "origin_um"):
        val = ctx.calib(key)
        if val is not None:
            out = out.with_metadata(**{key: val})

    # Which axes the reindex touched, as a `_sampling_of` axis declaration. A subset of m/t
    # alone leaves every (z,y,x) address pointing at the same physical location, and a pure
    # z subset is the `z:` case the spatial z-crop already stamps — but neither is dropped
    # by a consumer unless that axis is singleton on BOTH branches, and `m`/`t` never are
    # (see `_require_same_grid`: "frame 3" and "frame 7" are not one grid).
    touched = "".join(a for a in FRAME_AXES if keep[a] is not None)
    return _sampled(out, f"{touched}:crop.frames"
                         f"[m{_picks_text(ms)},t{_picks_text(ts)},z{_picks_text(zs)}]")


def _picks_text(picks: Optional[Tuple[int, ...]]) -> str:
    """A pick tuple for the sampling stamp; ``*`` for an axis that was left alone.

    Range-collapsed through the CANONICAL formatter, not spelled out: the stamp is provenance
    that gets COMPARED (``_sampling_of``), so it must have exactly one spelling per selection
    — which ``format_indices`` over an already sorted, de-duplicated tuple guarantees — while
    still being short enough to read in an error message about a grid mismatch."""
    return "*" if picks is None else format_indices(picks)


#: Shared hover text for the six crop bounds. One template rather than six near-identical
#: paragraphs: what differs between them is the axis and the inclusive/exclusive end, and
#: everything else — why the unit is px, what shrinking an axis does to the layers, that the
#: view is lazy — is the same sentence six times over.
#: The two bound GROUPS a crop pick writes (V2.16). Declared once and shared by all six
#: sockets so arming from any row produces the same gesture — the registry requires every
#: member to name the identical group, which a per-socket literal would eventually violate.
#: Split lateral from axial deliberately: a rectangle drawn on a plane says nothing about Z,
#: and the two halves are also gated apart (z0/z1 are 3D-only), which the registry's
#: gated-together rule would otherwise refuse.
_CROP_RECT: Tuple[str, ...] = ("y0", "y1", "x0", "x1")
_CROP_ZRANGE: Tuple[str, ...] = ("z0", "z1")
_CROP_BOUND_DOC = (
    "{axis} {end} of the window that is KEPT, in PIXELS — not µm, because a crop is an index "
    "range into the array and rounding a physical extent would make the output size depend on "
    "calibration. {extra} Cropping shrinks the {axis} axis, so any attribute layer whose shape "
    "no longer fits is dropped from the bundle, and a window that collapses to nothing is "
    "refused rather than silently produced. No pixels are copied — the result is a lazy view, "
    "so a filter downstream still reads real data for its halo right up to the crop edge.")
register_node(
    _compute_crop, op_key="util.crop", label="Crop", category="utility",
    inputs=[
        InDataset(),
        InInt("y0", "Y start", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              available_in={"region": frozenset({"spatial"})},
              description=_CROP_BOUND_DOC.format(
                  axis="Y", end="start",
                  extra="INCLUSIVE — this row is kept. 0 or unset starts at the top edge.")),
        InInt("y1", "Y end", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              available_in={"region": frozenset({"spatial"})},
              description=_CROP_BOUND_DOC.format(
                  axis="Y", end="end",
                  extra="EXCLUSIVE, like a Python slice — this row is the first one dropped. "
                        "Unset keeps everything to the bottom edge; a value past the edge is "
                        "clamped rather than an error.")),
        InInt("x0", "X start", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              available_in={"region": frozenset({"spatial"})},
              description=_CROP_BOUND_DOC.format(
                  axis="X", end="start",
                  extra="INCLUSIVE — this column is kept. 0 or unset starts at the left "
                        "edge.")),
        InInt("x1", "X end", unit="px", field=False, pick_bounds=_CROP_RECT, pick_kind="rect",
              available_in={"region": frozenset({"spatial"})},
              description=_CROP_BOUND_DOC.format(
                  axis="X", end="end",
                  extra="EXCLUSIVE, like a Python slice — this column is the first one "
                        "dropped. Unset keeps everything to the right edge; a value past the "
                        "edge is clamped.")),
        InInt("z0", "Z start", unit="px", field=False, pick_bounds=_CROP_ZRANGE, pick_kind="zrange",
              available_in={"dim": frozenset({"3D"}), "region": frozenset({"spatial"})},
              description=_CROP_BOUND_DOC.format(
                  axis="Z", end="start",
                  extra="INCLUSIVE, and a PLANE INDEX rather than a depth in microns. 3D "
                        "only — with the lever on 2D every plane is kept and this is "
                        "ignored.")),
        InInt("z1", "Z end", unit="px", field=False, pick_bounds=_CROP_ZRANGE, pick_kind="zrange",
              available_in={"dim": frozenset({"3D"}), "region": frozenset({"spatial"})},
              description=_CROP_BOUND_DOC.format(
                  axis="Z", end="end",
                  extra="EXCLUSIVE plane index — the first plane dropped. Unset keeps "
                        "everything to the last plane. 3D only.")),
        InString("frames", "Frames", field=False, default="", pick_kind="frames",
                 available_in={"region": frozenset({"frames"})},
                 description=
                 "WHICH FRAMES to keep, as one selection across the three frame axes: "
                 "\"m0-2,t3,z1-4\" keeps positions 0-2, timepoint 3 and planes 1-4. An axis "
                 "you do not name is kept WHOLE, so \"t3\" is timepoint 3 of every position "
                 "and \"m0,t3\" is a single frame; empty keeps everything. Indices are "
                 "0-based, ranges are inclusive at BOTH ends (unlike the exclusive Y/X bounds "
                 "on the other mode), and each axis may be SPARSE — \"t0,3,7\" keeps three "
                 "timepoints and drops what is between them, which a start/end pair could not "
                 "say. A bare list with no letter is read as timepoints. Press Pick to take "
                 "the boxes ticked on the viewer's M/T/Z strips, or — for an axis with nothing "
                 "ticked — the frame you are looking at, which is how you crop to the single "
                 "frame on screen; delete the \"z\" section afterwards to keep its whole "
                 "volume. No pixels are copied (this is a lazy index view) but the axes really "
                 "shrink, and everything indexed by them follows: stage coordinates and field "
                 "origins per position, acquisition times per timepoint, masks and per-plane "
                 "layers, and structure rows — a Point/Label/Track row on a dropped frame is "
                 "removed and the survivors renumbered, so the row COUNT a downstream table "
                 "reports changes. Keeping every Nth timepoint or plane MULTIPLIES the frame "
                 "interval or Z spacing by the stride (a velocity or a µm³ volume measured "
                 "downstream reads those), an unevenly spaced selection drops the spacing "
                 "rather than reporting one true of no pair, and dropping planes off the "
                 "bottom moves the volume's origin. Naming an axis but keeping nothing on it "
                 "is refused rather than producing an empty axis."),
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("region", ["spatial", "frames"], default="spatial", label="What to crop",
             description=
             "WHICH AXES this node cuts: a window out of each image, or whole frames out of "
             "the series. They are separate modes rather than one node with nine bounds "
             "because they are separate questions — one is about where in the field of view "
             "you are looking, the other about which acquisitions you want at all — and the "
             "params of the mode you are not in would sit there looking live. Each mode's "
             "sockets are remembered while the other is selected, and the choice folds into "
             "the memo key, so switching back and forth is free.",
             choice_docs={
                 "spatial": "Cut a rectangle out of every plane (Y/X), and in 3D a range of "
                            "planes as well. The output has fewer pixels per image but the "
                            "same number of images, and the field's origin moves to the cut "
                            "corner so placement and stitching stay correct. This is the "
                            "mode to trim a vignetted edge or to cut a region of interest "
                            "down to something a heavy filter can afford.",
                 "frames": "Keep only the multipoints, timepoints and Z planes you name, by "
                           "index. Every image keeps its full extent; there are simply fewer "
                           "of them. The selection may be sparse, and the Pick button fills "
                           "it in from the boxes ticked on the viewer's strips — which is "
                           "how you turn 'these three positions look right' into a graph "
                           "that only processes those three. Per-position and per-timepoint "
                           "metadata, the axis spacings and any structure rows follow the "
                           "selection.",
             }),
    ],
    granularity=_DIM_GRAN, kernel_axes=frozenset(),
    meta_transform=_meta_crop,
    description="Crop Y,X (and Z in 3D mode), or keep only chosen M/T/Z frames; "
                "pixel size preserved (origin deferred).")
