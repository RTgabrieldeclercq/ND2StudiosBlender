"""Trim (``util.trim``) — scale an axis DOWN by keeping evenly spaced frames: 12 timepoints
trimmed to a third are frames 0, 3, 7 and 11. Works on T (the usual case), Z or M, and
before or after a Timeseries Builder."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (MetaEnvelope, format_indices, position_subset, respaced,
                                retime, scope_stamp_subset, z_home_after)
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (FRAME_AXES, subset_lattice_layers,
                                                    subset_structure_rows)
from nodegraph.catalog._shared.sampling import _sampled

# ── Trim (2026-10-16) ───────────────────────────────────────────────────────────
#
# Crop's frames mode keeps the frames you NAME; this keeps the frames a RULE names, so the
# same card thins a 12-frame test series and a 1,200-frame production run to the same
# share without retyping an index list. Everything downstream of the pick is Crop's
# frames-mode machinery, shared rather than copied: the lazy FrameSubsetProvider view,
# `subset_lattice_layers` / `subset_structure_rows` for what rides alongside the image, and
# the per-axis metadata families (`retime` for T, `position_subset`, `respaced`,
# `z_home_after`, `scope_stamp_subset`). That is what makes it safe on either side of a
# Timeseries Builder: `retime` re-indexes a file bundle's per-file clocks
# (`source_frame_time_jd`) one level down, so a builder downstream still interleaves the
# surviving frames by their real acquisition times.

#: The axes Trim can thin, as the Mode spells them; lower-cased they are FRAME_AXES.
_TRIM_AXES: Tuple[str, ...] = ("T", "Z", "M")

#: The scalar SPACING each axis carries. A uniform pick re-spaces it, an irregular one drops
#: it (:func:`~nodegraph.metadata.respaced`); M has none.
_SPACING_KEY: Dict[str, str] = {"t": "dt_s", "z": "z_step_um"}


def _axis_of(modes: Mapping[str, Any]) -> str:
    """The trimmed axis as a FRAME_AXES letter (``"t"`` when unset or unrecognised)."""
    a = str((modes or {}).get("axis", "T")).strip().lower()
    return a if a in FRAME_AXES else "t"


_DEFAULT_FRACTION = 0.5
_DEFAULT_COUNT = 10


def trim_picks(n: int, modes: Mapping[str, Any], fraction: Any = None,
               count: Any = None) -> Tuple[Optional[Tuple[int, ...]], str]:
    """The indices Trim keeps out of ``n``, or ``(None, why)`` when the request is invalid.

    Shared by the compute and :func:`_meta_trim`, so the card's predicted length and the
    pulled length cannot disagree. ``fraction`` / ``count`` are the raw param values; the
    engine hands over overrides only, so ``None`` or blank means the socket default.

    * **how many** ``k``: ``fraction`` keeps ``round(n × fraction)`` (half up), ``count``
      keeps ``count``; either is clamped to ``[1, n]``, so a count past the end keeps every
      frame. A fraction above 1 or at/below 0, or a count below 1, is refused — Trim only
      scales down, and "3" typed into Fraction most likely meant "a third".
    * **which** ``k``: ``span`` spreads them from the first frame to the last,
      ``i·(n−1)//(k−1)`` — integer arithmetic, so 12 → 4 is exactly 0, 3, 7, 11 with no
      float edge; ``stride`` takes ONE step ``(n−1)//(k−1)`` from frame 0, so the spacing
      is uniform and the axis interval survives (multiplied), at the cost of the tail.

    ``k == n`` (or ``n <= 1``) is the full range: the caller returns its input untouched.
    """
    n = int(n)
    if n <= 1:
        return tuple(range(max(n, 0))), ""
    size_by = str((modes or {}).get("size_by", "fraction"))
    spacing = str((modes or {}).get("spacing", "span"))
    if size_by == "count":
        raw = _DEFAULT_COUNT if count is None or count == "" else count
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"Trim: Count must be a whole number of frames, got {raw!r}."
        if not math.isfinite(value) or value != int(value) or value < 1:
            return None, (f"Trim: Count = {raw!r} keeps no frames — it is how many frames to "
                          f"KEEP, a whole number of at least 1 (this axis has {n}).")
        k = min(int(value), n)
    else:
        raw = _DEFAULT_FRACTION if fraction is None or fraction == "" else fraction
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"Trim: Fraction must be a number between 0 and 1, got {raw!r}."
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            return None, (f"Trim: Fraction = {raw!r} is outside (0, 1]. It is the share of "
                          f"frames to KEEP — 0.33 keeps a third, 0.5 every other — so it "
                          f"cannot grow the axis or keep nothing. To keep a set number of "
                          f"frames, switch Size by to `count`.")
        k = max(1, min(n, int(math.floor(n * value + 0.5))))
    if k >= n:
        return tuple(range(n)), ""
    if k == 1:
        return (0,), ""
    if spacing == "stride":
        step = (n - 1) // (k - 1)
        return tuple(i * step for i in range(k)), ""
    return tuple(i * (n - 1) // (k - 1) for i in range(k)), ""


def trim_changes(metadata: Mapping[str, Any], axes: Any, axis: str,
                 keep: Tuple[int, ...]) -> Dict[str, Any]:
    """The metadata changes a trim of ``axis`` to ``keep`` makes — shared by the compute and
    :func:`_meta_trim` so the envelope and the payload describe the same frames.

    Crop's frames-mode rules, one axis at a time: per-M lists follow a position trim, per-T
    lists (and a bundle's per-file clocks) follow a time trim, the axis spacing is re-spaced
    or dropped, and the troubleshooting scope's stamps are remapped. Frame 0 is always kept
    (both spacings start there), so ``origin_um`` never moves and a Z trim needs no shift."""
    changes: Dict[str, Any] = {}
    if axis == "m":
        changes.update(position_subset(metadata, keep))
    elif axis == "t":
        changes.update(retime(metadata, keep))
    else:
        changes["z_step_um"] = respaced(metadata.get("z_step_um"), keep)
        changes.update(z_home_after(metadata, keep))
    picks = {a: (keep if a == axis else None) for a in FRAME_AXES}
    changes.update(scope_stamp_subset(metadata, axes, picks))
    return changes


def _meta_trim(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.trim``: narrow one of M / T / Z to the evenly spaced picks of :func:`trim_picks`.

    Total, like every transform: an invalid request HOLDS the envelope and lets the payload
    raise the real message, and the full range is the identity."""
    ax = env.axes
    axis = _axis_of(modes)
    n = int(getattr(ax, axis))
    params = params or {}
    keep, _problem = trim_picks(n, modes or {}, params.get("fraction"), params.get("count"))
    if keep is None or len(keep) >= n:
        return env
    return (env.with_axes(replace(ax, **{axis: len(keep)}))
            .with_metadata(**trim_changes(env.metadata, ax, axis, keep)))


def _compute_trim(ctx: EvalContext) -> Dataset:
    """Keep evenly spaced frames of one axis.

    Resolved spec (build-node-v2 §0, 2026-10-16)
    --------------------------------------------
    * **Kind** utility, axis-changing → ``op_key="util.trim"``, category ``"utility"``,
      ``meta_transform=_meta_trim`` (local, its maths shared with this compute through
      :func:`trim_picks` / :func:`trim_changes`).
    * **Data contract** ``Dataset → the same Dataset with one of m / t / z narrowed``.
      Nothing spatial changes and no pixel is copied. Lattice layers are subset with the
      image; Point / Label / Track rows on a dropped frame are dropped and the survivors
      renumbered. Per-axis metadata follows by Crop's frames-mode rules (:func:`trim_changes`).
      T is retimed by the rule every T subset shares (:func:`~nodegraph.metadata.retime`):
      the per-frame clock follows exactly, and ``dt_s`` is MULTIPLIED by a uniform step and
      DROPPED by an uneven one — ``span`` is uneven whenever ``(n−1)/(k−1)`` is not whole
      (12 → 4 steps 3, 4, 4). A rate downstream then divides each step by that step's own
      gap, from the clock (or, on a clockless series, from the per-frame ``frame_time_s``
      the rule records). ``z_step_um`` has no per-plane twin, so an uneven Z trim leaves
      3D measurements without an axial scale — ``stride`` keeps it.
    * **Before or after Timeseries Builder.** Before it, each file's own frames are trimmed
      (a bundle's per-file clocks follow, so the builder still interleaves by real time);
      after it, the BUILT series is trimmed, evenly in acquisition order. M before a builder
      thins the files themselves (a per-frame export is one file per position).
    * **2D/3D** no lever: index arithmetic only.
    * **Footprint** ``TILEABLE``, no kernel axes — one output unit reads exactly the input
      unit of the kept frame; dropped frames are never read.
    * **Sockets** ``fraction`` (``size_by=fraction``) and ``count`` (``size_by=count``), each
      gated to its branch. Modes ``axis`` (T/Z/M), ``size_by`` (fraction/count), ``spacing``
      (span/stride).
    * **Backend** none — :class:`~nodegraph.provider.FrameSubsetProvider`.

    Keeping every frame (``k == n``, a 1-frame axis) returns the input untouched, so an
    unconfigured card costs and stamps nothing — the choice ``util.crop`` makes.
    """
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Trim needs an image on its input Dataset — it thins the frames of "
                         "one, and a measurements-only input has none.")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {}) or {}
    axis = _axis_of(modes)
    n = int(getattr(ax, axis))
    keep, problem = trim_picks(n, modes, ctx.params.get("fraction"), ctx.params.get("count"))
    if keep is None:
        raise ValueError(problem)
    if len(keep) >= n:
        return ds
    new_axes = replace(ax, **{axis: len(keep)})
    picks = {a: (keep if a == axis else None) for a in FRAME_AXES}
    # `zs=None` (not the full range) when z is untouched: the provider then leaves z alone
    # entirely and keeps the fingerprint it would have had without a z pick (Crop's rule).
    view = FrameSubsetProvider(prov,
                               keep if axis == "m" else tuple(range(ax.m)),
                               keep if axis == "t" else tuple(range(ax.t)),
                               keep if axis == "z" else None)
    out = subset_lattice_layers(ds.with_image(view), new_axes, picks)
    out = subset_structure_rows(out, picks)
    out = out.with_metadata(**trim_changes(ds.metadata, ax, axis, keep))
    # the envelope wins for the spacing key, so it stays the single source of truth and the
    # read stays memo-fenced; `_meta_trim` ran the same `respaced` on the same value
    key = _SPACING_KEY.get(axis)
    if key is not None:
        val = ctx.calib(key)
        if val is not None:
            out = out.with_metadata(**{key: val})
    # An axis-only stamp: m / t / z reindex but every (y, x) address still points at the
    # same place. Canonical index spelling, because sampling provenance is COMPARED.
    return _sampled(out, f"{axis}:trim[{format_indices(keep)}]")


register_node(
    _compute_trim,
    op_key="util.trim", label="Trim", category="utility",
    inputs=[
        InDataset(),
        InFloat("fraction", "Fraction", unit="", field=False, default=_DEFAULT_FRACTION,
                available_in={"size_by": frozenset({"fraction"})},
                description=
                "The SHARE of the axis to keep: 0.5 keeps every other frame, 0.33 a third "
                "(12 frames → 4), 0.1 one in ten. LOWER keeps fewer frames, so everything "
                "downstream runs proportionally faster and sees a coarser series — a track "
                "or a flow field then spans a bigger step per frame. Rounded half up and "
                "never below 1 frame; 1 keeps everything. Values above 1 or at/below 0 are "
                "refused, since Trim only scales down. Only read when Size by is `fraction`."),
        InInt("count", "Frames to keep", unit="", field=False, default=_DEFAULT_COUNT,
              available_in={"size_by": frozenset({"count"})},
              description=
              "HOW MANY frames to keep, whatever the axis length: the same card then thins a "
              "short test series and the full run to the same number of frames. A count at "
              "or past the axis length keeps every frame; below 1 is refused. Only read when "
              "Size by is `count`."),
    ],
    outputs=[OutDataset("out")],
    modes=[
        Mode("axis", _TRIM_AXES, default="T", label="Axis",
             description=(
                 "Which axis to thin. Only that axis shrinks; every image, mask and table "
                 "on the others is untouched, and nothing in a plane is resampled."),
             choice_docs={
                 "T": "Timepoints — the usual case. The time axis is rebuilt for the kept "
                      "frames: the per-frame clock (frame_time_jd, and each file's own clock "
                      "inside a multi-file card) follows them, so Viewer timestamps and a "
                      "Timeseries Builder downstream stay right; dt_s is multiplied by a "
                      "uniform step and dropped by an uneven one, and speeds and velocities "
                      "then use each step's own gap. Masks, detections and tracks on dropped "
                      "frames are removed.",
                 "Z": "Focal planes — a coarser stack, e.g. to test a 3D pipeline fast. "
                      "z_step_um is multiplied by a uniform step (so 3D volumes and "
                      "distances stay in true µm) and DROPPED by an uneven one, which 3D "
                      "measurements then report as unknown — use `stride` for a stack you "
                      "will measure. The bottom plane is always kept, so the stack does not "
                      "move.",
                 "M": "Positions (multipoints / tiles). Before a Timeseries Builder that is "
                      "the files themselves when each file is one frame. Per-position stage "
                      "coordinates and file names follow the kept positions; picks are "
                      "spread over positions in order, so on a multi-file card they need "
                      "not keep the same number of positions per file — and the builder "
                      "refuses files that differ.",
             }),
        Mode("size_by", ("fraction", "count"), default="fraction", label="Size by",
             description=(
                 "How the number of kept frames is given: as a share of the axis, or as a "
                 "fixed count."),
             choice_docs={
                 "fraction": "A share of the axis (Fraction): 0.33 of 12 frames keeps 4, of "
                             "120 keeps 40 — the trim scales with the series, so a graph "
                             "built on a short test file thins the full run the same way.",
                 "count": "A fixed number of frames (Frames to keep), whatever the axis "
                          "length — for a preview or a batch that must come out the same "
                          "size from series of different lengths.",
             }),
        Mode("spacing", ("span", "stride"), default="span", label="Spacing",
             description=(
                 "Where the kept frames fall. Both always keep frame 0, so a Registration "
                 "anchored on the first frame sees the same reference."),
             choice_docs={
                 "span": "From the FIRST frame to the LAST, as evenly as whole frames "
                         "allow: 12 → 4 keeps 0, 3, 7, 11. The whole series is covered, but "
                         "unless (n−1)/(k−1) is whole the steps differ by one frame (3, 4, "
                         "4), so dt_s is dropped rather than averaged: every kept frame "
                         "keeps its own time (the clock, or frame_time_s on a series with "
                         "none), and speeds, velocities and time axes downstream read each "
                         "step's real gap. On Z there is no per-plane twin, so z_step_um is "
                         "simply dropped and 3D measurements lose their axial scale.",
                 "stride": "ONE fixed step from frame 0, the largest that still fits the "
                           "count: 12 → 4 keeps 0, 3, 6, 9. The spacing is exactly uniform, "
                           "so dt_s / z_step_um is kept (multiplied by the step) — what a "
                           "Z stack you will measure, or anything that needs one interval "
                           "(a PIV ensemble, a time typed in seconds), wants — at the cost "
                           "of the tail (here frames 10 and 11).",
             }),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_trim,
    description="Scale one axis down by keeping evenly spaced frames — 12 timepoints trimmed "
                "to a third are frames 0, 3, 7 and 11 (or a fixed number of frames). Time by "
                "default, or Z planes or positions. The per-frame clock, masks and "
                "detections follow, and it works before a Timeseries Builder (each file's "
                "frames are trimmed and still interleave by their real times) or after it "
                "(the built series is trimmed). A lazy view: dropped frames are never read.",
)
