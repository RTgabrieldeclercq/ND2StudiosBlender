"""Time Shift (``util.time_shift``) — move a data stream along its own time axis by a whole
number of frames, later or earlier; the T-axis twin of ``align.shift`` (2026-10-07)."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.provider import FrameRemapProvider
from nodegraph.registry import Granularity, InDataset, InInt, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (remap_lattice_layers,
                                                    shift_structure_rows)
from nodegraph.catalog._shared.sampling import _sampled

# ── Time Shift (2026-10-07) ─────────────────────────────────────────────────────
#
# ``align.shift`` moves the CONTENT of every frame by its (z, y, x) shift under a fixed pixel
# grid. This node does the same thing one axis over: it moves the content along T under a
# fixed TIME grid — frame k of the output is frame k − Δ of the input. T, ``dt_s`` and the
# per-frame clock are untouched, because the grid is the same and only what sits on it moved;
# that is what lets two streams of one acquisition be lined up for a Merge or a comparison
# when one lags the other by a few frames. Nothing is interpolated: Δ is whole frames.


def time_shift_sources(t: int, delta: int, edges: str) -> List[Optional[int]]:
    """Which input frame each of the ``t`` output frames reads after a shift of ``delta``
    (positive = later): output ``k`` reads ``k − delta``. Past either end, ``hold`` reads the
    nearest edge frame and ``blank`` reads nothing (``None`` → zeros). Shared by the compute
    and its test so the two cannot disagree about the direction."""
    out: List[Optional[int]] = []
    for k in range(int(t)):
        j = k - int(delta)
        if 0 <= j < t:
            out.append(j)
        elif edges == "blank":
            out.append(None)
        else:
            out.append(0 if j < 0 else int(t) - 1)
    return out


def signed_delta(raw: Any, modes: Mapping[str, Any]) -> int:
    """The ``delta`` value ``raw`` (a magnitude in frames) with the ``direction`` Mode's sign: ``later`` is
    positive (the content appears later), ``earlier`` negative. A value that is not a whole
    number is refused — a fractional frame would mean interpolating in time, which this node
    does not do."""
    try:
        value = float(1 if raw is None else raw)
    except (TypeError, ValueError):
        raise ValueError(f"Time Shift: Delta must be a whole number of frames, got {raw!r}.")
    if value != int(value) or value < 0:
        raise ValueError(f"Time Shift: Delta must be a whole number of frames ≥ 0, got {raw!r}; "
                         f"the Direction mode carries the sign.")
    d = int(value)
    return -d if str(modes.get("direction", "later")) == "earlier" else d


def _compute_time_shift(ctx: EvalContext) -> Dataset:
    """Move the stream along T by a whole number of frames.

    Resolved spec (build-node-v2 §0, 2026-10-07)
    --------------------------------------------
    * **Kind** utility, axis-PRESERVING → ``op_key="util.time_shift"``, category
      ``"utility"``; no ``meta_transform`` — the axes and every metadata key are unchanged,
      because the time grid (T, ``dt_s``, the per-frame clock) is the same and only the
      content moved on it.
    * **Data contract** ``Dataset → the same Dataset re-indexed on T``: output frame ``k`` is
      input frame ``k − Δ`` (``direction`` = ``later`` ⇒ Δ > 0, the content appears Δ frames
      later; ``earlier`` ⇒ Δ < 0). Frames that fall off the end are gone; frames with no
      source repeat the nearest edge frame (``edges`` = ``hold``) or are zero (``blank``).
      Lattice layers (a mask, a label raster) move with the image frame for frame; Point /
      Label / Track rows move their ``t`` and rows shifted off either end are dropped (a
      mesh that would lose rows is dropped whole, as every frame subset does).
    * **2D/3D** no lever: index arithmetic only.
    * **Footprint** ``TILEABLE``, no kernel axes — one output unit reads exactly one input
      unit (of another frame); nothing else of the series is read.
    * **Sockets** ``delta`` (whole frames, ≥ 0; the sign is the ``direction`` Mode, because
      the card's numeric controls are non-negative). Modes ``direction`` (later / earlier),
      ``edges`` (hold / blank).
    * **Backend** none — :class:`~nodegraph.provider.FrameRemapProvider`, a lazy index remap.

    Δ = 0 and a single-frame input are the identity. A Δ beyond the series length is allowed:
    every frame then reads an edge frame (``hold``) or is blank.
    """
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Time Shift needs an image on its input Dataset.")
    modes = ctx.params.get("__modes__", {}) or {}
    d = signed_delta(ctx.params.get("delta", 1), modes)
    edges = str(modes.get("edges", "hold"))
    t = int(prov.axes.t)
    if t <= 1 or d == 0:
        return ds
    sources = time_shift_sources(t, d, edges)
    out = ds.with_image(FrameRemapProvider(prov, sources))
    out = remap_lattice_layers(out, "t", sources)
    out = shift_structure_rows(out, "t", d, t)
    # A t-only stamp: the content moved under a fixed (z, y, x) grid, exactly as Shift
    # stamps a drift apply — a consumer comparing this stream with the series may drop it.
    return _sampled(out, f"t:time_shift[{d:+d}]")


register_node(
    _compute_time_shift,
    op_key="util.time_shift", label="Time Shift", category="utility",
    inputs=[
        InDataset(description=
                  "The stream to move in time. Every position, plane and channel of it is "
                  "re-indexed along T the same way; its masks, label rasters and detections "
                  "move with it. T and the frame interval do not change."),
        InInt("delta", "Delta", unit="", field=False, default=1,
              description=
              "HOW MANY FRAMES the content moves, as a count ≥ 0 — the Direction mode says "
              "which way. 1 frame is one `dt_s` of the series (the frame labels on a Split T "
              "card show the time each index stands for). LARGER moves the stream further: "
              "with `later`, frame k of the output shows what frame k − Delta of the input "
              "showed, so events appear Delta frames later; frames shifted off the far end are "
              "gone, and the frames at the near end with nothing to show follow the Edges "
              "mode. 0 is the identity. A count beyond the series length leaves every frame "
              "reading an edge frame (hold) or blank. A fraction is refused: this node moves "
              "whole frames and never interpolates in time."),
    ],
    outputs=[OutDataset("out")],
    modes=[
        Mode("direction", ["later", "earlier"], default="later", label="Direction",
             description=
             "Which way the content moves along time. The card's numbers are magnitudes, so "
             "the sign of the shift lives here: `later` delays the stream (what was at frame "
             "k is now at frame k + Delta), `earlier` advances it (what was at frame k is now "
             "at frame k − Delta). Pick by what you are lining up with: a stream that LAGS a "
             "reference needs `earlier`; one that LEADS it needs `later`.",
             choice_docs={
                 "later":
                     "Delay: output frame k reads input frame k − Delta, so every event "
                     "appears Delta frames later than it did. The first Delta frames have "
                     "no source and follow the Edges mode; the last Delta input frames fall "
                     "off the end and are gone. Use it on a stream that runs AHEAD of the "
                     "one you are comparing it with.",
                 "earlier":
                     "Advance: output frame k reads input frame k + Delta, so every event "
                     "appears Delta frames sooner. The first Delta input frames are gone; "
                     "the last Delta output frames have no source and follow the Edges mode. "
                     "Use it on a stream that LAGS the one you are comparing it with — the "
                     "common case when a second camera or channel starts late.",
             }),
        Mode("edges", ["hold", "blank"], default="hold", label="Edges",
             description=
             "What the frames with NO SOURCE show — the Delta frames at the near end (later) "
             "or far end (earlier) that the shift opened up. T is kept either way, so the "
             "stream still lines up frame for frame with whatever it is combined with; this "
             "only decides what those frames contain.",
             choice_docs={
                 "hold":
                     "Repeat the nearest real frame: the opened frames at the start show the "
                     "first input frame, those at the end the last. Masks and rasters repeat "
                     "with it, so a segmentation stays visible there; detections are NOT "
                     "duplicated (a row exists once). Reads naturally when scrubbing and is "
                     "the right choice when the frames are padding rather than data.",
                 "blank":
                     "Leave the opened frames empty: zero pixels, zero masks, no detections. "
                     "Honest about the missing data — a measurement on those frames reads "
                     "nothing rather than a repeated frame's objects — at the cost of dark "
                     "frames when scrubbing. Choose it when downstream counts or intensities "
                     "on every frame matter more than a continuous-looking series.",
             }),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Move a stream along its own time axis by a whole number of frames, later or "
                "earlier: frame k of the output is frame k − Delta of the input. T, the frame "
                "interval and the per-frame clock stay — the grid is the same, the content "
                "moved on it — so two streams of one acquisition can be lined up for a Merge "
                "or a comparison when one lags the other. Masks and detections move with the "
                "image. The T twin of Shift.",
)
