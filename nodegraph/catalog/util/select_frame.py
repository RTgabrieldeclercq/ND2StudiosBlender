"""Select Frame (``util.select_frame``) — keep ONE timepoint of a series, by its 0-based index
(picked off the viewer); the tap Split T's per-frame outputs materialize into, and usable on
its own."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (frame_pick, respaced, select_frame as _meta_select_frame,
                                time_subset)
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import Granularity, InDataset, InInt, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (subset_lattice_layers,
                                                    subset_structure_rows)
from nodegraph.catalog._shared.sampling import _sampled

# ── Select Frame (2026-10-07) ───────────────────────────────────────────────────
#
# The T-axis member of the tap family: ``channel.split`` → ``channel.select``,
# ``util.split_positions`` → ``util.select_position``, ``util.split_z`` → ``util.select_plane``,
# and now ``util.split_t`` → this node. The engine is one-payload-per-node, so K distinct
# per-frame payloads come from K real tap nodes; the split card is a pass-through.
#
# It keeps Crop's frames-mode rules for the metadata indexed by T — the per-T lists through
# ``time_subset`` and ``dt_s`` through ``respaced`` (a single index keeps the source interval)
# — rather than ``util.stack``'s "drop dt_s": Stack MERGES frames, so there is no interval
# left to describe; this keeps one real frame, and a Split T RANGE group materializes into
# Crop (``t0-9``), so the two ways of picking frame 3 must agree on what they stamp.


def _compute_select_frame(ctx: EvalContext) -> Dataset:
    """Keep exactly one timepoint of a series.

    Resolved spec (build-node-v2 §0, 2026-10-07)
    --------------------------------------------
    * **Kind** utility, axis-changing → ``op_key="util.select_frame"``, category
      ``"utility"``, ``meta_transform=select_frame``.
    * **Data contract** ``Dataset → the same Dataset with t = 1``. The T half of
      ``util.crop``'s frames mode exactly: nothing spatial changes, the per-T lists
      (``frame_time_jd``, …) are subset with the axis (:func:`~nodegraph.metadata.time_subset`),
      ``dt_s`` survives (one frame keeps the source interval — :func:`respaced`), lattice
      layers follow and structure rows on other frames are dropped, the survivors re-addressed
      to ``t = 0`` — the shared helpers every frame subset uses.
    * **2D/3D** no lever: index arithmetic only.
    * **Footprint** ``TILEABLE``, no kernel axes — one output unit reads exactly the
      corresponding input unit of the kept frame; the rest of the series is never read.
    * **Sockets** ``frame`` (0-based; default 0, the acquisition's first frame and the anchor
      Registration's ``first`` mode uses; ``pick_kind="frame"`` adopts the timepoint the
      viewer is showing). No modes.
    * **Backend** none — :class:`~nodegraph.provider.FrameSubsetProvider` with one timepoint.

    Resolution is shared with the edit-time transform (:func:`~nodegraph.metadata.frame_pick`)
    so the card and the pull agree. A value past the end is REFUSED with the series length —
    handing back some other frame's pixels under the index asked for is the failure a selector
    exists to prevent. A single-frame input is returned as is, whatever is asked.
    """
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Select Frame needs an image on its input Dataset.")
    ax = prov.axes
    raw = ctx.params.get("frame")
    if int(ax.t) <= 1:
        return ds                    # the only frame there is: the identity, whatever is asked
    k = frame_pick(int(ax.t), raw)
    if k is None:
        raise ValueError(_no_such_frame(raw, int(ax.t)))
    new_axes = replace(ax, t=1)
    view = FrameSubsetProvider(prov, tuple(range(ax.m)), (k,), None)
    picks = {"m": None, "t": (k,), "z": None}
    out = subset_lattice_layers(ds.with_image(view), new_axes, picks)
    out = subset_structure_rows(out, picks)
    md = ds.metadata
    changes: Dict[str, Any] = dict(time_subset(md, (k,)))
    changes["dt_s"] = respaced(md.get("dt_s"), (k,))
    out = out.with_metadata(**changes)
    # the envelope wins for the calibration key, so it stays the single source of truth and
    # the read stays memo-fenced (the same arithmetic on the same value, so no disagreement)
    val = ctx.calib("dt_s")
    if val is not None:
        out = out.with_metadata(dt_s=val)
    # A t-only stamp: the selection reindexes T and leaves every (z, y, x) address pointing
    # at the same location, so a consumer comparing this frame with the series may drop it.
    return _sampled(out, f"t:select_frame[{k}]")


def _no_such_frame(raw: Any, t: int) -> str:
    valid = "0" if t <= 1 else f"0..{t - 1}"
    return (f"Select Frame: this Dataset has no timepoint {raw!r} — it has {t} frame(s), so "
            f"the only valid indices are {valid} (0 is the first frame; blank is 0). Negative "
            f"values are not 'from the end' here, they are simply out of range. (If you "
            f"rewired the Split T node onto a shorter series, the slot you picked may be past "
            f"its end.)")


register_node(
    _compute_select_frame,
    op_key="util.select_frame", label="Select Frame", category="utility",
    inputs=[
        InDataset(),
        InInt("frame", "Frame", unit="", field=False, default=0, pick_kind="frame",
              description=
              "WHICH TIMEPOINT to keep, 0-based. Scrub Frame t in the Viewer to the frame you "
              "want and press Pick to take the one on screen; 0 (the default) is the first "
              "frame, the anchor Registration's `first` mode uses. The output has one frame, "
              "so whatever follows works on exactly this timepoint — a segmentation of the "
              "starting state, a reference for Align To — and the rest of the series is never "
              "read. A value past the end is refused with the series length rather than "
              "falling back to another frame; on a single-frame input every value is the "
              "identity."),
    ],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_select_frame,
    description="Keep ONE timepoint of a series, chosen on the Viewer (default: the first "
                "frame): T narrows to 1 and everything indexed by t follows — the per-frame "
                "clock, per-frame masks, the detections on that frame. The tap Split T's "
                "per-frame outputs materialize into, and usable on its own.",
)
