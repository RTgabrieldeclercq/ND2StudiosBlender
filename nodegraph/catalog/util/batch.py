"""Batch (``util.batch``) — stack two or more files on the BATCH axis so one drawn
pipeline runs over all of them, independently."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import BATCH_FILE_KEY, batch_grow as _batch_grow
from nodegraph.provider import BatchProvider
from nodegraph.registry import Granularity, InDataset, Mode, OutDataset

from nodegraph.catalog._base import register_node

# ── Batch (V3.01) ────────────────────────────────────────────────────────────────
#
# The engine half of the canvas' "golden point": wire K files in, draw ONE pipeline, get K
# results out (`util.unbatch` splits them again).
#
# **Why this is not `util.merge(merge_axis="M")`.** Merging on M makes the files positions
# of one acquisition, and from that moment nothing can tell "position 3 of file 1" from
# "position 3 of the run". That is not cosmetic: a `scope="dataset"` statistic then pools
# its level ACROSS FILES — the caveat `MultiSourceProvider`'s own docstring names and
# `scripts/_probe_m_batch_independence.py` pins. Every file would get a threshold computed
# partly from the others, which is the precise failure a batch feature exists to avoid. On
# `b` the population stops at the file boundary, so a per-file answer is per-file.
#
# **Nothing here is resampled, blended or copied.** `BatchProvider` re-addresses: a read at
# member `b` is that member's own file, byte for byte, so a batched pull reads exactly what
# K separate graphs would have read.
#
# **Ragged files, and why `align` is a choice rather than a default.** An array axis is
# rectangular, so members must share (m,t,z,c,y,x) — but real files routinely do not: two
# scans of one plate came back with 748 and 754 positions (2026-09-29). `align="trim"` crops
# to the common extent and STAMPS what it dropped; `refuse` (the default) stops and names
# the axes. Trimming is not the default because it is a real loss: a per-file object COUNT
# computed over 748 fields against one over 754 is not comparable, and nothing downstream
# would ever say so. Only `m` and `t` are trimmable — see `_TRIMMABLE`.
#
# **Parallelism comes from the existing scheduler, not from anything here.** K members are
# K times the units for `nodegraph.parallel.map_units` to spread over the cores it already
# uses, and the streaming chain is demand-driven per tile, so member 1's segmentation runs
# while member 0's denoise is still going. Adding a batch-level thread pool would have hit
# `map_units`' own re-entrancy guard and serialized instead.

#: Cheapest possible footprint, and honestly so: this node computes nothing. It re-addresses
#: K providers behind one, so an output unit reads EXACTLY the corresponding member's own
#: unit — the same argument `util.merge`'s T/M/Z branch makes for its own TILEABLE.
_GRAN = Granularity.TILEABLE


#: Axes a batch may be TRIMMED on, and the only two that can honestly differ between files
#: of one experiment. ``m`` is how many fields the plate was scanned at and ``t`` how long
#: the run went — two acquisitions of the same plate routinely differ by a few of either.
#:
#: ``z``/``c``/``y``/``x`` are deliberately absent. A different stack depth, channel count or
#: frame size is not a longer run of the same thing, it is a different acquisition —
#: trimming those would be reconciling data that should not be compared, and the refusal is
#: the right answer.
_TRIMMABLE: Tuple[str, ...] = ("m", "t")

#: Namespaced, non-calibration record of what the batch did — the note the card reads,
#: the same idiom ``util.merge``'s ``__merge__`` uses.
BATCH_KEY = "__batch__"


def _aligned(members: Tuple[Dataset, ...]) -> Tuple[Tuple[Dataset, ...], str]:
    """Crop every member to the batch's common extent on :data:`_TRIMMABLE`.

    Returns the cropped members and a note naming exactly what was dropped, because a
    batch that silently discarded six of one file's positions would make every per-file
    count quietly incomparable — and the count is usually the answer.
    """
    from nodegraph.provider import FrameSubsetProvider
    from nodegraph.metadata import position_subset, time_subset
    from nodegraph.catalog._shared.frame_subset import (
        subset_lattice_layers, subset_structure_rows)

    sizes = {a: [int(getattr(ds.axes, a)) for ds in members] for a in _TRIMMABLE}
    keep = {a: min(v) for a, v in sizes.items()}
    dropped = {a: [n - keep[a] for n in v] for a, v in sizes.items()}
    if not any(any(d) for d in dropped.values()):
        return members, ""

    out: List[Dataset] = []
    for ds in members:
        ax = ds.axes
        ms = tuple(range(keep["m"]))
        ts = tuple(range(keep["t"]))
        if int(ax.m) == keep["m"] and int(ax.t) == keep["t"]:
            out.append(ds)
            continue
        new_axes = replace(ax, m=keep["m"], t=keep["t"])
        view = FrameSubsetProvider(ds.image, ms, ts, None)
        cut = subset_lattice_layers(ds.with_image(view), new_axes,
                                    {"m": ms, "t": ts, "z": None})
        cut = subset_structure_rows(cut, {"m": ms, "t": ts, "z": None})
        cut = cut.with_metadata(**position_subset(ds.metadata, list(ms)))
        cut = cut.with_metadata(**time_subset(ds.metadata, list(ts)))
        out.append(cut)

    parts = [f"{a}: kept {keep[a]}, dropped {d}" for a, d in dropped.items() if any(d)]
    return tuple(out), "trimmed to the common extent — " + "  ·  ".join(parts)


def _ragged(members: Tuple[Dataset, ...]) -> bool:
    """True when the members disagree on any axis a batch has to share."""
    return any(len({int(getattr(ds.axes, a)) for ds in members}) > 1
               for a in ("m", "t", "z", "c", "y", "x"))


def _ragged_refusal(members: Tuple[Dataset, ...]) -> str:
    """The message for a batch whose members do not share a grid, naming the way out."""
    bad = []
    for a in ("m", "t", "z", "c", "y", "x"):
        vals = [int(getattr(ds.axes, a)) for ds in members]
        if len(set(vals)) > 1:
            bad.append((a, vals))
    lines = "\n".join(f"  {a}: {vals}" for a, vals in bad)
    trimmable = [a for a, _ in bad if a in _TRIMMABLE]
    fix = (f"\n\nSet Align to 'trim' to crop every file to the common extent on "
           f"{', '.join(trimmable)} — the batch then runs, and the node's card says "
           f"exactly how many were dropped from which file."
           if trimmable and all(a in _TRIMMABLE for a, _ in bad) else
           "\n\nThese axes cannot be trimmed: a different stack depth, channel count or "
           "frame size is a different acquisition rather than a longer run of the same "
           "one, so batching them would compare data that should not be compared.")
    return (f"Batch: the files do not share a grid, and a batch runs ONE pipeline over "
            f"every member, so they have to.\n{lines}{fix}")


def _compute_batch(ctx: EvalContext) -> Dataset:
    """Stack every Dataset wired into ``data`` on the batch axis, in wiring order.

    Resolved spec
    -------------
    * **Kind** utility, axis-changing → ``op_key="util.batch"``, category ``"utility"``.
    * **Data contract** grows ``b`` and nothing else; every other axis must already match
      across every member (refused by name otherwise). Member ``b`` is input ``b``.
    * **2D/3D** no lever. The node runs no kernel over voxels — only index re-addressing —
      so dimensionality is a property of the data, not an opinion this node could add
      (`wire-node-v2` §7b: derive, don't ask).
    * **Footprint** ``TILEABLE``, no kernel axes: a pure re-address costs what the
      un-batched read costs.
    * **Backend** none — :class:`~nodegraph.provider.BatchProvider` is a lazy re-address.

    **Member order** is the engine's canonical multi-socket order (edge-creation order),
    the same rule ``util.merge`` documents. Unlike a merge on T or M, order here is not
    load-bearing for the arithmetic — no member is a reference and nothing is placed
    against anything — but it IS the order ``util.unbatch`` hands results back in, so the
    stamped ``batch_file`` list is what makes it readable rather than something to audit.
    """
    members: Tuple[Dataset, ...] = tuple(ctx.input("data") or ())
    if len(members) < 2:
        raise ValueError(
            "Batch needs at least two files wired into its input — connect a second "
            "source before this node. One file alone is not a batch; run it directly.")
    for i, ds in enumerate(members):
        if ds.image is None:
            raise ValueError(f"Batch: member {i} has no image to batch.")

    labels = _member_labels(members)
    # Align BEFORE the provider sees them: BatchProvider's grid check is a hard refusal by
    # design (a ragged array axis is not representable), so the policy has to be applied
    # here, where the user's choice is visible and what it cost can be stamped.
    align = str(ctx.params.get("__modes__", {}).get("align", "refuse"))
    note = ""
    if align == "trim":
        members, note = _aligned(members)
    elif _ragged(members):
        raise ValueError(_ragged_refusal(members))

    prov = BatchProvider([ds.image for ds in members], labels=labels)
    ref = members[0]
    out = ref.with_image(prov).reshaped_axes(prov.axes)
    # Provenance is stamped here and nowhere else: `util.unbatch` reads this list to know
    # how many members it has and what to call them, so it never needs the count as a
    # param that could disagree with the wiring.
    stamp = {BATCH_FILE_KEY: list(labels)}
    if note:
        stamp[BATCH_KEY] = {"note": note}
    return out.with_metadata(**stamp)


def _member_labels(members: Tuple[Dataset, ...]) -> List[str]:
    """A display name per member, preferring the file each one came from.

    A member that is itself a one-file load carries its name in ``source_file``; anything
    else (a synthesized stack, a chain that dropped the key) falls back to its slot. Names
    are made UNIQUE by suffixing a repeat, because two members sharing a label would make
    the unbatch's outputs indistinguishable on the canvas — the one thing this list exists
    to prevent.
    """
    from nodegraph.metadata import SOURCE_FILE_KEY
    out: List[str] = []
    seen: Dict[str, int] = {}
    for i, ds in enumerate(members):
        got = ds.metadata.get(SOURCE_FILE_KEY)
        name = ""
        if isinstance(got, (list, tuple)) and got:
            # every position of this member came from one file iff the list is constant;
            # a member that is itself a bundle has no single name, so it keeps its slot
            uniq = {str(v) for v in got}
            name = str(got[0]) if len(uniq) == 1 else ""
        name = name or f"file{i}"
        n = seen.get(name, 0)
        seen[name] = n + 1
        out.append(name if n == 0 else f"{name} ({n + 1})")
    return out


register_node(
    _compute_batch,
    op_key="util.batch", label="Batch", category="utility",
    inputs=[
        InDataset(multi=True, label="Files",
                  description=
                  "Two or more files to run ONE pipeline over. They are stacked on the "
                  "batch axis, not merged: each stays a separate acquisition, so a "
                  "threshold or any other data-derived level is computed per file rather "
                  "than pooled across them. That is the whole difference from Merge, "
                  "whose Multipoint axis makes the files positions of one run.\n\n"
                  "Every file must already share (m,t,z,c,y,x); a mismatch is refused "
                  "naming the axis, because a batch axis is rectangular and padding a "
                  "short file would silently invent frames. Order is the order you wired "
                  "them, and it is the order Unbatch hands the results back in."),
    ],
    outputs=[OutDataset("out")],
    modes=[Mode("align", ["refuse", "trim"], default="refuse", presentation="body",
                label="Align",
                description=
                "What to do when the files do not share a grid.\n\n"
                "REFUSE (default) stops and names the axes that differ. TRIM crops every "
                "file to the common extent on the position and time axes — two scans of "
                "one plate routinely differ by a few fields or a few frames, and trimming "
                "is usually what you meant. It is a real loss of data, so it is opt-in "
                "and the card states exactly how many were dropped from which file; a "
                "per-file COUNT computed over different numbers of fields is not "
                "comparable, and nothing else would say so.\n\n"
                "Only position and time can be trimmed. A different stack depth, channel "
                "count or frame size is a different acquisition rather than a longer run "
                "of the same one, and is refused under either setting.",
                choice_docs={
                    "refuse": "Stop, naming every axis the files disagree on and by how "
                              "much. The safe default: a batch that quietly reconciled a "
                              "mismatch would make its per-file results incomparable "
                              "without saying so.",
                    "trim": "Crop every file to the smallest common position and time "
                            "extent, then batch. Lossy by construction — the card says "
                            "what went — and the right answer when the files are the same "
                            "experiment scanned at slightly different lengths.",
                })],
    granularity=_GRAN, kernel_axes=frozenset(),
    meta_transform=_batch_grow,
    description="Stack two or more files on the batch axis so one pipeline runs over all "
                "of them independently — per-file statistics, not pooled. Unbatch splits "
                "the results back apart.",
)
