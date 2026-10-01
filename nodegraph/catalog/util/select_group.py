"""Select Group (``util.select_group``) — keep only the positions of one specimen group."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (format_indices, group_picks, position_group_plan,
                                position_subset, select_group as _meta_select_group)
from nodegraph.placement import GROUP_GAP_FACTOR
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import Granularity, InDataset, InFloat, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (subset_lattice_layers,
                                                    subset_structure_rows)
from nodegraph.catalog._shared.sampling import _sampled

# ── Select Group (axis-changing: M narrows to one specimen's fields) ────────────
#
# The multipoint axis of a real acquisition is frequently not one flat list. The lab's
# `9.1.26_CRC_Gradient/Channel640_Seq0001.nd2` records 54 positions that are six separate
# 3x3 mosaics — nine tiles at a 147 um pitch across a 293 um field, then a millimetre of
# empty stage to the next specimen. NIS writes that list flat, so every node downstream
# reads six experiments as one, and the damage is quiet: Stitch fuses all 54 into a single
# canvas with four enormous holes in it, a `scope="dataset"` threshold pools six unrelated
# samples into one histogram, and a per-position table reports 54 rows for a six-sample
# experiment.
#
# This node is the missing "which specimen?" selector. It changes no pixels — it is a lazy
# index remap, exactly like `util.crop`'s frames mode, and shares that node's machinery for
# everything that rides alongside the image (`_shared/frame_subset`). What it adds is the
# resolution step: turning "G3" into the multipoint indices that mean it, from the stage
# geometry (`nodegraph.placement.position_groups`) or from the grouping the GUI stored.


def _compute_select_group(ctx: EvalContext) -> Dataset:
    """Keep only the multipoints belonging to the selected position group(s).

    Resolved spec (build-node-v2 §0, 2026-09-15)
    --------------------------------------------
    * **Kind** utility, axis-changing -> ``op_key="util.select_group"``, category
      ``"utility"``, ``meta_transform=select_group``.
    * **Data contract** ``Dataset -> the same Dataset with a shorter M``. No calibration
      changes at all: nothing is cut out of a plane, no axis is re-spaced, and each
      surviving position keeps the ``origin_um`` it already had. Everything indexed BY m
      follows the selection — the per-M metadata family (:data:`PER_POSITION_KEYS`), lattice
      layers, and structure rows, which are filtered and renumbered.
    * **2D/3D** no lever. The node runs no kernel over voxels; it performs index arithmetic,
      so dimensionality is a property of the data and an opinion here could only disagree
      with it (`wire-node-v2` §7b — derive, don't ask). This is the same call ``util.merge``
      makes for its ``T``/``M``/``Z`` branches.
    * **Footprint** ``TILEABLE``, no kernel axes. A pure index remap costs exactly what the
      un-selected read would; it emphatically does NOT read across m, which is what
      separates it from the ``MULTI_VIEW`` three (Stitch and friends).
    * **Sockets** ``group`` (which specimen), ``gap_factor`` (the detector's threshold, read
      only when the Dataset carries no stored grouping). No modes.
    * **Backend** none. :class:`~nodegraph.provider.FrameSubsetProvider` is the whole image
      path; the detector is numpy-only arithmetic over metadata.
    * **Performance** no numba. There is no per-voxel work to fuse — the clustering is over
      tens of stage coordinates and runs once per pull, and once per keystroke at edit time.

    **Why the group is resolved from the INPUT, on both sides.** The edit-time transform
    (:func:`nodegraph.metadata.select_group`) and this compute must predict the same ``m``
    or the gate fails, and they cannot share a value — each is handed its own view of the
    world. So they share the *resolution*: both call
    :func:`~nodegraph.metadata.group_picks` on the pre-transform metadata and axes, which
    routes through :func:`~nodegraph.placement.field_box` and therefore agrees with where
    Stitch will actually place the tiles. Reading the stage geometry off the input payload
    is the sanctioned route for non-calibration provenance and is what
    ``registration.align_to`` already does.

    **A refusal, not a fallback, when the fields cannot be placed.** A Dataset with no
    position log and no stored grouping cannot be grouped, and the plausible guess — treat
    them as one group, or split into equal blocks — produces a result that looks exactly
    like a correct one. It refuses and names what is missing, the same rule ``util.stitch``
    applies to the same absence.
    """
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("select_group needs an image provider on its input Dataset")
    ax = prov.axes
    raw = ctx.params.get("group")
    gap = ctx.params.get("gap_factor")
    keep = group_picks(ds.metadata, ax, raw, gap)

    # Nothing asked for is a genuine no-op: the input is returned untouched rather than
    # wrapped in an identity view, so an unconfigured node costs nothing and stamps nothing
    # (the choice `util.crop`'s frames mode and `util.zproject`'s `none` method both make).
    if keep is None:
        return ds
    if not keep:
        raise ValueError(_no_match(ds, ax, raw, gap))
    if len(keep) == ax.m:
        return ds

    new_axes = replace(ax, m=len(keep))
    view = FrameSubsetProvider(prov, keep, tuple(range(ax.t)), None)
    picks = {"m": keep, "t": None, "z": None}

    out = subset_lattice_layers(ds.with_image(view), new_axes, picks)
    out = subset_structure_rows(out, picks)
    out = out.with_metadata(**position_subset(ds.metadata, keep))
    # ...and the envelope wins for the one CALIBRATION key that is indexed by m, so it stays
    # the single source of truth and the read stays memo-fenced (`util.crop` does the same).
    # It cannot disagree with the local computation above: that is the same `position_subset`
    # applied to the same list, and the transform has already applied it to the envelope.
    origin = ctx.calib("origin_um")
    if origin is not None:
        out = out.with_metadata(origin_um=origin)

    # Only m is touched, so every (z, y, x) address still points at the same physical
    # location. The stamp says so through the CANONICAL formatter — it is provenance that
    # gets compared (`_sampling_of`), so one selection must have exactly one spelling, and
    # the INDICES rather than the group key are what a comparison means: two keys that
    # resolve to the same positions are the same selection.
    return _sampled(out, f"m:select_group[{format_indices(keep)}]")


def _no_match(ds: Dataset, ax: Any, raw: Any, gap: Optional[float]) -> str:
    """The refusal message for a selection that kept nothing.

    Two very different causes reach here and the fix differs, so the message has to tell
    them apart: either the fields could not be PLACED (no stage log — nothing can be
    grouped, and no amount of retyping will help), or they were placed and the name simply
    does not match one of the groups that are there (in which case listing them is the
    whole answer)."""
    plan = position_group_plan(ds.metadata, ax, gap)
    if not plan.placed:
        return (
            f"util.select_group: cannot group {ax.m} positions — this Dataset carries no "
            f"usable field geometry. Grouping needs `pixel_size_um` plus a per-position "
            f"`origin_um` or `stage_xy_um` covering every one of the {ax.m}, and a partial "
            f"log is refused rather than used for the positions it happens to have. An ND2 "
            f"from a multipoint acquisition carries this; a TIFF and a hand-built Dataset "
            f"do not. Clear `group` to pass every position through unchanged.")
    listed = "; ".join(g.brief() for g in plan.groups)
    return (
        f"util.select_group: `group` = {raw!r} matches no position group. This Dataset has "
        f"{len(plan.groups)}: {listed}. Name one by its key (\"{plan.groups[0].key}\"), by "
        f"its 1-based ordinal (\"1\"), or several with a comma (\"G1,G3\"); clear the socket "
        f"to keep every position.")


register_node(
    _compute_select_group, op_key="util.select_group", label="Select Group",
    category="utility",
    inputs=[
        InDataset(),
        InString("group", "Group", field=False, default="",
                 description=
                 "WHICH SPECIMEN to keep. A multipoint file often holds several separate "
                 "samples — six 3x3 mosaics a millimetre apart, one plate well per site — "
                 "and this keeps the positions of the one you name and drops the rest. "
                 "Name it by key (\"G2\"), by 1-based ordinal (\"2\"), or by whatever you "
                 "renamed the group to; a comma list (\"G1,G3\") keeps several, and EMPTY "
                 "keeps every position, which makes an unconfigured node a true no-op. The "
                 "groups themselves are read from the acquisition if it stored them, and "
                 "otherwise recovered from the stage coordinates — so the keys you can type "
                 "here depend on the file. Type `?` (or any name that does not exist) to see "
                 "them: the refusal lists every group with its size and grid shape, which "
                 "is how you find out what this file contains. "
                 "No pixels are copied (this is a lazy index view) but the M axis really "
                 "shrinks, and everything indexed by it follows: stage coordinates and "
                 "field origins per position, per-position masks and layers, and structure "
                 "rows — a Point/Label/Track row on a dropped position is removed and the "
                 "survivors renumbered, so the row COUNT a downstream table reports "
                 "changes. Nothing is re-spaced and no measurement moves: a kept position "
                 "is bit-for-bit what it was. Selecting a group that does not exist is "
                 "refused, with the ones that do listed."),
        InFloat("gap_factor", "Group gap", unit="", field=False,
                default=GROUP_GAP_FACTOR,
                description=
                "How far apart two fields must be, IN FIELD WIDTHS, before they count as "
                "different specimens. The default 1.0 is not a tuned number: tiles of one "
                "mosaic have to overlap (or at worst touch) to be stitchable at all, so "
                "their centres are always closer than one field, and anything further is "
                "not a neighbouring tile. RAISE it to merge clusters that should be one "
                "specimen — sparse sampling across a large sample, where real gaps exceed a "
                "field; LOWER it to split a cluster that is really two. It changes only "
                "WHICH positions are grouped, never the pixels or any measurement. INERT "
                "when the Dataset already carries a stored grouping (from the file, or one "
                "you edited), because that answer is used as-is rather than re-derived."),
    ],
    outputs=[OutDataset()],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_select_group,
    description="Keep only the multipoints of one specimen group — the 3x3 mosaic, the "
                "plate well — recovered from the stage coordinates or read from the "
                "acquisition. Lazy: no pixels move, and per-position metadata, layers and "
                "structure rows follow the selection.")
