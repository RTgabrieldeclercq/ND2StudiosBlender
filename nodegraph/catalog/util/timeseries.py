"""Timeseries Builder (``util.timeseries``) — collect source FILES, one per slot, and lay
them onto one axis (Time by default, or Channel, Z or Multipoint) **in the order their
clocks say**, so a folder of exports becomes one correctly ordered series.

The problem. A microscope that exports one file per frame leaves a folder of
``WellA3_t001.nd2 .. WellA3_t120.nd2``. However those files reach the canvas, nothing in the
loader knows they are *timepoints*: a multi-file source card (a **file bundle**,
:class:`~nodegraph.provider.MultiSourceProvider`) lays them end to end on ``m``, because
``m`` is the only axis it can grow without being told what the files MEAN, and separate
cards are simply separate Datasets. Either way the series is 120 positions of a single
timepoint where the acquisition was one position at 120 timepoints, so every temporal node
downstream reads a series of length 1: ``util.stack`` fuses nothing, ``track.link`` has no
frames to link, ``analysis.optical_flow`` has no second frame to flow to. Nothing errors.
The graph just quietly answers a different question.

This node says which axis the files vary along, after the fact::

    in :  m = start_i + j       t = a        (120 files' positions, 1 timepoint)
    out:  m = j                 t = i*a0 + a (1 position, 120 timepoints)

**Slots that grow as you fill them** (2026-10-02, the same shape as the Viewer's sources):
``File 1`` plus one empty slot after the last wired file, so you keep collecting files
without ever seeing a column of unused inputs. A slot takes one bundle card, or one single
loader; each input is split into its own files by the per-position ``source_file`` labels
(:func:`~nodegraph.metadata.source_file_runs`), so a bundle of 60 in one slot plus two
single cards in two more is 62 files and the node never asks which shape it was handed.

**Ordered by the clock, not by how you wired them.** Every ND2 carries a per-frame absolute
wall clock (``frame_time_jd``, with its readable twin ``frame_datetime``). With
``File order = time`` (the default) the files are sorted by their first frame's clock, so
the series is right whatever order you dropped the files in and whatever their names say;
where any file carries no clock the node falls back to the counting field in the filenames
(``t9`` before ``t10``), and where names cannot order them either, to wiring order — and
the card's note SAYS which rule was used, because the file order is the one thing a wrong
result here depends on and nothing else in the GUI shows it. The output carries the
concatenated per-frame clock and timestamps, so a Viewer's timestamp overlay downstream
reads the real acquisition time of every frame of the built series.

**Why not ``util.merge``.** Merge does the same index arithmetic, and for two or three files
it is the right node. Its member order is the engine's edge-creation order — invisible in
the GUI, and load-bearing for T and Z (its own docstring says so) — and it cannot see inside
a bundle at all, so a 120-file series means 120 cards.

**Nothing is resampled, blended or copied.** One output voxel is one input voxel, so a
chained pull reads exactly the bytes an un-chained one would, from each file's own store —
which is why this is ``TILEABLE`` despite changing two axes.

**Files may differ freely on the axis being built.** A series exported in unequal chunks
— 5 frames, then 3 — is still one series, and the result's chained axis is the SUM of what
each file brings (``t = 8``), not a multiple of the first file's.

**What it will not do.** It refuses a mismatch on any axis that is NOT being chained (the
result is rectangular in those), files holding different numbers of POSITIONS when chaining
onto anything but M (the result has one position axis, so a file with more has nowhere to
put them — and the message says to chain onto M instead), inputs that disagree on
``pixel_size_um``/``z_step_um``/``bit_depth`` (chaining reconciles nothing, so the result
would carry one file's scale over another's pixels), and any input that already carries
masks, labels or measurement rows — each of those is indexed by the position axis this node
re-addresses, so re-addressing around them would leave a mask describing a frame it was
never computed on. Put the builder directly after the loaders, before any analysis.

Until 2026-10-02 this node was ``util.chain`` ("Chain Files") with one multi-input socket
and name order by default; saved graphs that name ``util.chain`` are loaded as this node.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.file_sequence import describe as _seq_describe
from nodegraph.metadata import (
    CHAIN_ORDERS, chain_grow as _meta_chain_grow, chain_member_clock, chain_members,
    chain_order_used, chain_position_spread, chain_position_tolerance, chained_metadata)
from nodegraph.placement import jd_to_datetime_text
from nodegraph.provider import AxisRespreadProvider
from nodegraph.registry import Granularity, InDataset, Mode, OutDataset

from nodegraph.catalog._base import register_node

#: Which axis the files are laid onto. ``M`` is included because laying files onto
#: positions is an ordinary chain: it is what JOINS separately wired cards, and for one
#: already-ordered bundle it is the identity. To switch the builder off without unwiring
#: it, MUTE the node — that is the general mechanism and it does not depend on this list.
_CHAIN_AXES: Tuple[str, ...] = ("T", "M", "C", "Z")

#: How many file slots the card can show. Slots reveal themselves one at a time
#: (``grow_group``), so this is a ceiling, not a column of empty inputs. A bundle card
#: counts as one slot however many files it holds, so the ceiling is on CARDS.
MAX_FILES = 12

#: Namespaced, non-calibration record of what the build actually did — the resolved file
#: ORDER above all, since that is the one thing a wrong result here depends on and the one
#: thing nothing else in the GUI shows (`wire-node-v2` Section 7b stamp-and-inherit; the
#: same idiom as ``util.merge``'s ``__merge__`` and ``util.crop_to``'s ``__crop_to__``).
#: Plain JSON-shaped values only — this folds into ``output_fingerprint``. The key keeps
#: its historical name so ``util.stitch``'s diagnostics (which read it) need not change.
CHAIN_KEY = "__chain__"

#: Scalar calibration every input must already agree on. Chaining resamples nothing, so a
#: disagreement cannot be reconciled — it can only be carried forward over pixels it does
#: not describe. The same tuple, for the same reason, as ``util.merge``'s concatenation
#: branch checks.
_CALIBRATION_MUST_MATCH: Tuple[str, ...] = ("pixel_size_um", "z_step_um", "bit_depth")

_SLOT_DOC = (
    "Another source file (or multi-file card) for the series. Slots appear one at a time "
    "as you fill them, so there is always exactly one empty slot to drop the next file "
    "into. The order you wire them in does not matter under `File order = time`: the "
    "files are sorted by their first frame's acquisition clock.")


def _compute_timeseries(ctx: EvalContext) -> Dataset:
    """Lay every wired slot's FILES onto ``chain_axis``, ordered by ``chain_order``.

    Resolved spec (Section 0 grill, 2026-09-29; multi-input 2026-09-29; slots + clock
    order 2026-10-02)
    ------------------------------------------------------------------
    * **Kind** utility, axis-changing -> ``op_key="util.timeseries"``, category
      ``"utility"``. Sits beside ``util.merge`` and deliberately does not absorb it: Merge
      reconciles differently-placed acquisitions onto C (resampling, focus-matching), which
      this node never does, and this node sees INSIDE a bundle, which Merge never does.
    * **Data contract** image -> image, lazily. Onto T/C/Z, ``m`` shrinks to the per-file
      position count and the chained axis grows by the file count; onto M the files line up
      as positions and ``m`` becomes ``k*n``. Every other axis and every scalar calibration
      key is untouched — nothing is resampled, so no value is rewritten. The per-M/per-T/
      per-C metadata families move in lockstep (:func:`~nodegraph.metadata.chained_metadata`,
      shared verbatim with the ``meta_transform`` so the two cannot drift).
    * **2D/3D** no lever. The node runs no kernel over voxels — only index re-addressing —
      so dimensionality is a property of the data, not an opinion this node could add
      (`wire-node-v2` Section 7b: derive, don't ask).
    * **Footprint** ``TILEABLE``, no kernel axes, on every branch: one output unit reads
      EXACTLY the corresponding input unit, so a chained read costs what the un-chained read
      costs.
    * **Backend** none — :class:`~nodegraph.provider.AxisRespreadProvider` is a lazy index
      remap, the same kind of object as :class:`~nodegraph.provider.AxisConcatProvider`.

    **The file order is derived, not wired.** ``chain_order="time"`` sorts by each file's
    first-frame clock (:func:`~nodegraph.metadata.chain_member_clock`) when every file has
    one; ``"sequence"`` by the numeric field that VARIES across the names; either falls
    back toward wiring order where its information is missing, and the stamped
    :data:`CHAIN_KEY` note says which rule actually applied (``"time"`` / ``"sequence"`` /
    ``"wired"``) and lists the resolved order with each file's first timestamp.

    **A no-op rather than a refusal** wherever there is nothing to do: fewer than two files
    across every slot, or an axis it does not recognise. The input comes back untouched, so
    an unconfigured node on a canvas costs and stamps nothing — the choice
    ``util.select_batch`` and ``util.crop`` both make.
    """
    inputs: List[Dataset] = [d for d in ctx.inputs if isinstance(d, Dataset)]
    if not inputs:
        raise ValueError("Timeseries Builder has no file wired to it.")
    ds = inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    axis = str(modes.get("chain_axis", "T")).lower()
    if axis not in ("t", "z", "c", "m"):
        return ds
    order = str(modes.get("chain_order", "time"))
    if order not in CHAIN_ORDERS:
        raise ValueError(f"Timeseries Builder: unknown file order {order!r} — one of "
                         f"{list(CHAIN_ORDERS)}")
    members, n, problem = chain_members(
        [(d.metadata, d.axes) for d in inputs], axis, order=order)
    if problem is not None:
        raise ValueError(problem)
    k = len(members)
    if k < 2:
        # One file and nothing to lay onto anything. A sequence of one IS a correct answer,
        # so this is a pass-through rather than an error -- the user who wired one card gets
        # their file back, not a traceback, and an unconfigured node on a canvas costs and
        # stamps nothing (the `util.select_batch` / `util.crop` choice).
        return ds

    for i, d in enumerate(inputs):
        if d.attributes:
            raise ValueError(
                f"Timeseries Builder re-addresses the position axis, which every mask, "
                f"label, track and measurement row is indexed by — re-addressing around "
                f"them would leave each one describing a frame it was never computed on. "
                f"File slot {i + 1} already carries them: put the builder directly after "
                f"the loaders, before any analysis node.")
        if d.image is None:
            raise ValueError(
                f"Timeseries Builder has no image to lay out for file slot {i + 1} — it "
                f"carries axes and metadata but no pixel source. Wire it to a loader (or to "
                f"a docked checkpoint), not to a node that produces measurements only.")

    # Files laid end to end keep their own numbers, so a scale that differs between them is
    # never reconciled -- it is silently averaged by whatever reads the result. Refused by
    # name here, exactly as `util.merge`'s concatenation branch refuses it, rather than
    # after a measurement has already mixed two pixel sizes.
    for key in _CALIBRATION_MUST_MATCH:
        seen = {i: d.metadata.get(key) for i, d in enumerate(inputs)
                if d.metadata.get(key) is not None}
        vals = set(seen.values())
        if len(vals) > 1:
            named = ", ".join(f"file slot {i + 1} has {v}" for i, v in seen.items())
            raise ValueError(
                f"Timeseries Builder cannot lay these files onto {axis.upper()}: they "
                f"disagree on {key} ({named}). Chaining resamples nothing, so the result "
                f"would carry one of these numbers over pixels acquired at the other. "
                f"Resample upstream (util.resample) so they match, or set the value on the "
                f"source cards.")

    prov = AxisRespreadProvider(
        [(inputs[mm.source].image, mm.start, mm.count) for mm in members], axis,
        labels=[mm.name or f"file slot {mm.source + 1}" for mm in members])
    # The chained axis is the SUM of what each file contributes, not a multiple of the
    # first file's: files are allowed to differ on the axis being chained (5 frames then
    # 3 is still one series), and only the axes NOT being chained have to agree.
    ax = ds.axes
    total = sum(mm.extent(axis) for mm in members)
    new_axes = replace(ax, m=total) if axis == "m" else replace(ax, m=n, **{axis: total})
    out = ds.with_image(prov).reshaped_axes(new_axes)

    changes: Dict[str, Any] = dict(chained_metadata(members, axis, n))
    names = [mm.name for mm in members]
    used = chain_order_used(members, order)
    clocks = [chain_member_clock(mm) for mm in members]
    changes[CHAIN_KEY] = {
        "axis": axis.upper(), "files": [nm or "?" for nm in names],
        "per_file": [mm.extent(axis) for mm in members], "positions": n,
        "inputs": len(inputs),
        # WHICH rule ordered them, not just which order resulted: the one thing a wrong
        # result here depends on, so it is stated rather than inferable.
        "order": used,
        "first_frame": [jd_to_datetime_text(c) if c is not None else "" for c in clocks],
        "note": (f"{k} files in acquisition-clock order"
                 + (f", {jd_to_datetime_text(clocks[0])} → {jd_to_datetime_text(clocks[-1])}"
                    if clocks and clocks[0] is not None and clocks[-1] is not None else "")
                 if used == "time" else
                 _seq_describe(names) if used == "sequence" else
                 f"{k} files in wiring order — no clock and no filenames to order by"),
    }
    # How far apart the files put each position, against how far they may be and still be
    # one field. Recorded because it DECIDES whether the stage log survives (see
    # `chained_metadata`), and a downstream Stitch that finds no log should be able to say
    # it was this node, by how much, rather than blaming the file format.
    spread = chain_position_spread(members, "stage_xy_um") if axis != "m" else None
    if spread is not None:
        changes[CHAIN_KEY]["stage_spread_um"] = round(spread[0], 3)
        changes[CHAIN_KEY]["stage_tolerance_um"] = round(chain_position_tolerance(members)[0], 3)
    return out.with_metadata(**changes)


register_node(
    _compute_timeseries,
    op_key="util.timeseries", label="Timeseries Builder", category="utility",
    inputs=[InDataset("data", label="File 1",
                      description=
                      "The first source file (or multi-file card) of the series. Its "
                      "calibration is the one every other file must match; its place in the "
                      "built series is decided by its clock like every other file's."),
            *[InDataset(f"file_{i}", label=f"File {i}", grow_group="files",
                        description=_SLOT_DOC) for i in range(2, MAX_FILES + 1)]],
    outputs=[OutDataset()],
    modes=[
        Mode("chain_axis", _CHAIN_AXES, default="T", label="Onto axis",
             description=(
                 "Which axis the wired files actually vary along. A loader can only put "
                 "files on POSITIONS, because that is all it can know about them; this "
                 "says what they really were, and re-addresses them with no resampling "
                 "and no copy."),
             choice_docs={
                 "T": "Timepoints — the usual case: one file per frame (or per chunk) of a "
                      "timelapse. 120 files become t=0..119, and util.stack, track.link and "
                      "analysis.optical_flow start seeing a series instead of 120 "
                      "unrelated fields. Files need NOT hold the same number of frames: a "
                      "series exported in unequal chunks (5 then 3) becomes t=0..7. "
                      "Separately loaded files each contribute their own clock, which is "
                      "concatenated so every frame of the result keeps its real acquisition "
                      "time; within one multi-file card the clock is DROPPED: the bundle "
                      "only ever carried the first file's. dt_s is left alone and still "
                      "describes spacing WITHIN a file, not the gap between two. The stage "
                      "log is kept (first file's reading) wherever every file puts a "
                      "position within a tenth of a field of the same place, so Stitch by "
                      "stage still works on the result.",
                 "M": "Positions — the files stay separate fields of one acquisition, "
                      "laid end to end on M. For a single multi-file card already in "
                      "filename order this is the identity (it is what the card did); for "
                      "several wired cards it is what joins them; for a card in the wrong "
                      "order it re-sorts the positions. Use it when the files are wells or "
                      "dishes rather than frames — but prefer Batch if you want per-file "
                      "statistics, since M pools a dataset-scoped threshold across every "
                      "file.",
                 "C": "Channels — one file per stain, acquired separately. Block i of the "
                      "output is file i's channels, and the channel NAMES are tiled, which "
                      "is exact rather than invented because the loader refuses to group "
                      "files whose channel names differ. That also means the result has "
                      "each name repeated once per file: use channel.select by index, not "
                      "by name, downstream. Unlike util.merge's C branch nothing is placed "
                      "or resampled here — grouped files already share a grid.",
                 "Z": "Focal planes — one file per plane of a stack. z_step_um is left "
                      "alone and still describes spacing within a file; the gap between "
                      "the last plane of one file and the first of the next is not "
                      "recorded anywhere, so a stack chained from irregularly spaced files "
                      "will measure axial distances wrongly. Check the spacing yourself, "
                      "or set it on the source card.",
             }),
        Mode("chain_order", CHAIN_ORDERS, default="time", label="File order",
             description=(
                 "Which file becomes frame 0, and the rest after it. This is the one "
                 "setting a wrong result here actually depends on, and nothing else in the "
                 "GUI shows the answer — the node card's note reports the rule that was "
                 "actually used and the resolved order, with each file's first timestamp."),
             choice_docs={
                 "time": "By each file's acquisition clock — its first frame's absolute "
                         "time as the microscope recorded it (frame_time_jd), so the series "
                         "is correct whatever order the files were wired in and whatever "
                         "their names say. Needs a clock on EVERY file; where one is "
                         "missing the node falls back to the filename sequence and the "
                         "card's note says so. The default.",
                 "sequence": "By the counting number in the filenames: the numeric field "
                             "that VARIES across them, so t9 sorts before t10 and a "
                             "constant _z002 elsewhere in the name is not mistaken for the "
                             "counter. Falls back to a plain natural sort when no single "
                             "field varies, and to wiring order when any file has no name. "
                             "Use it for files whose clocks are wrong or missing.",
                 "loaded": "Wiring order: the slots in the order you wired them, and each "
                           "card's own files in the order it holds them. Use it when both "
                           "the clocks and the names are unusable and you have assembled "
                           "the set by hand in the order you want. The card's note reads "
                           "'wired' whenever this order is in force, chosen or fallen back "
                           "to.",
             }),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_chain_grow,
    description=(
        "Build one series from many source FILES: wire each file (or multi-file card) into "
        "its own slot — a new empty slot appears as you fill the last — and the files are "
        "laid onto ONE axis (Time by default; or Channels, Z, positions) in the order their "
        "acquisition clocks say, so the series is right whatever order you dropped them in. "
        "Falls back to the counting number in the filenames, then to wiring order, and says "
        "which on the card. A pure index remap: nothing is resampled or copied. Put it "
        "directly after the loaders — it refuses an input that already carries masks, "
        "labels or measurements, since those are indexed by the axis it re-addresses. "
        "(Formerly Chain Files, util.chain.)"))
