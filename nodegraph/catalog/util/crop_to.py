"""Crop To File (``util.crop_to``) — crop a Dataset to the field ANOTHER file occupies in
absolute stage coordinates (its stage XY plus its field-of-view size).

The question this answers is the one ``util.crop`` cannot: *where in my overview was the
other acquisition taken?* ``util.crop`` takes a pixel window, and a pixel window is only
meaningful inside one file's own grid — two files at 0.287 and 1.718 µm/px are both 1024²
and describe a 294 µm and a 1760 µm field, a 36× area difference index space cannot see
(:func:`nodegraph.placement.lateral_extent_um`). So the window is DERIVED here, from the
microscope frame the two files share: the reference's ``stage_xy_um`` and pixel extent give
a µm box, and :func:`nodegraph.placement.source_window` inverts that box onto this file's
pixel grid.

No new maths. This is the `util.merge` / `view.overlay` placement family read backwards —
those two RESAMPLE a secondary into a primary's field, and this one throws the primary's
field away down to where the secondary was. Same ``FieldBox``, same handedness rule, same
stage-versus-origin precedence.

Nothing is copied: the result is a :class:`~nodegraph.streaming.WindowView` over a
(optionally) :class:`~nodegraph.provider.FrameSubsetProvider`-narrowed source, exactly as
``util.crop`` is, so a filter downstream still reads real pixels for its halo right up to
the crop edge.

The reference input supplies **placement only, never pixels** — which is why it is not a
``view_source`` (§7e): drawing it would put the other file's picture on top of the crop it
was only ever consulted to locate.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import crop_to_field as _meta_crop_to, drop_position_keys
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import (DimMode, InBool, InDataset, InFloat, InInt, Mode, OutDataset)
from nodegraph.streaming import WindowView

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN
from nodegraph.catalog._shared.frame_subset import subset_lattice_layers, subset_structure_rows
from nodegraph.catalog._shared.placement_entry import handedness_for
from nodegraph.catalog._shared.sampling import _sampled


#: Namespaced, non-calibration record of what this crop actually did — which position it
#: took, how much of the region that position held, which positions it therefore LEFT OUT,
#: and how much of the requested µm box survived the edges (`wire-node-v2` §7b
#: stamp-and-inherit; the same idiom as ``util.merge``'s :data:`~nodegraph.catalog.util.merge.MERGE_KEY`).
#:
#: It rides the payload rather than being raised, because none of it is an error: a crop
#: that took the best of two straddled tiles is a correct answer to an ambiguous question,
#: and the user asked to be TOLD, not stopped. Plain JSON-shaped values only — this folds
#: into ``output_fingerprint``.
CROP_TO_KEY = "__crop_to__"

#: A region smaller than this fraction of ONE pixel of the cropped file is treated as a
#: degenerate request rather than rounded up to a 1-px window in silence. Set at a tenth of
#: a pixel: a legitimate crop of a much finer reference file out of a much coarser overview
#: really can land inside a single pixel (the WellA3 pair's 294 µm field is 171 px of the
#: 1.718 µm/px scan, so this only fires for something ~1700× finer), and a 1-px output that
#: nobody asked for is harder to diagnose than a refusal that says the ratio.
_MIN_WINDOW_FRAC = 0.1


def _compute_crop_to(ctx: EvalContext) -> Dataset:
    """Crop ``data`` to the field ``region`` occupies in absolute stage µm.

    Resolved spec (§0 grill, 2026-09-24)
    ------------------------------------
    * **Kind** utility, axis-changing (Y/X shrink, M collapses to 1, Z shrinks in 3D) →
      ``op_key="util.crop_to"``, category ``"utility"``. Sits beside ``util.crop``, which
      it deliberately does not absorb: that node takes a window you name in pixels, this
      one derives a window from a second file's placement, and folding a Dataset input
      into ``util.crop`` would put a live-looking socket on every graph that only wants
      ``y0:y1``.
    * **Data contract** image → image, lazily. Pixel size, ``dt_s``, ``z_step_um`` and
      ``bit_depth`` are untouched — nothing is resampled and no value is rewritten;
      ``origin_um`` moves to the cut corner and the per-position stage logs retire
      (:func:`~nodegraph.metadata.drop_position_keys`, ``util.stitch``'s rule, for
      ``util.stitch``'s reason: after this there is ONE field and its stage CENTRE is no
      longer the field centre).
    * **2D/3D** a real lever. 2D cuts Y/X only. 3D additionally keeps the planes whose
      absolute focus falls inside the reference's axial span, using the ZStackLoop
      anchoring (:func:`~nodegraph.placement.z_um_of_slice`) — and degrades to
      lateral-only, with a note, when either file carried no focus log. That is not an
      edge case: the WellA3 GFP tiles are exactly that state.
    * **Footprint** ``_DIM_GRAN`` / ``kernel_axes=frozenset()`` — byte-for-byte
      ``util.crop``'s declaration, and for the same reason: the compute reads no pixel at
      all, it builds a view, so the footprint it inherits is its base's.
    * **Backend** none. :mod:`nodegraph.placement` for the geometry
      (:func:`~nodegraph.placement.field_box`,
      :func:`~nodegraph.placement.overlap_fraction`,
      :func:`~nodegraph.placement.source_window`), then
      :class:`~nodegraph.provider.FrameSubsetProvider` + :class:`~nodegraph.streaming.WindowView`.
    * **Performance** no numba, no hot path: the whole compute is O(M) box arithmetic over
      at most a few dozen multipoints.

    Four questions the grill settled, all of which have a wrong answer that looks right
    ---------------------------------------------------------------------------------
    **The region may straddle several positions of ``data``.** A crop is ONE pixel window,
    so it can only come from one of them. The node takes the position with the largest
    overlap and writes a :data:`CROP_TO_KEY` note naming the coverage and the positions it
    therefore did not include, rather than refusing: a 68%/32% straddle is a correct answer
    to an ambiguous question, and the remedy (``util.stitch`` upstream) is a graph edit the
    user may not want. The note is how a partial crop stops being silent.

    **The reference may itself be multi-position.** ``extent="union"`` takes the bounding
    box of every field it could place — which is also the right answer for the ordinary
    single-position file, so the default needs no thought — and ``extent="position"``
    takes one named multipoint of it.

    **The box will often hang off the edge.** It is clamped to the intersection and the
    shortfall is recorded; only a MISS (zero overlap) is refused, because a crop of nothing
    has no honest output. :func:`~nodegraph.placement.source_window` does the clamping, so
    this node cannot drift from what the overlay already does with a partly-covered field.

    **Handedness is derived, not defaulted.** ``flip_x``/``flip_y`` describe how the camera
    is mounted relative to the stage, and they are *inapplicable* to an already-stitched
    input whose pixels are already laid out in stage coordinates — applying them again
    mirrors the mosaic inside its own footprint (measured 784 µm out on the WellA3 pair).
    :func:`~nodegraph.catalog._shared.placement_entry.handedness_for` is the one place that
    decides, shared with ``util.merge``.

    Two deliberate asymmetries with the ``meta_transform``
    -----------------------------------------------------
    :func:`nodegraph.metadata.crop_to_field` is handed only input 0's envelope
    (``propagate_meta`` reads ``dataset_preds[0]``), so it can see neither the reference's
    placement nor which position won. It therefore marks Y/X — and Z in 3D — **UNKNOWN**
    and DROPS ``origin_um``, exactly as ``util.stitch`` marks its mosaic extent unknown
    rather than guessing. The payload carries the truth. This is the one place in this node
    where the envelope is a strictly weaker claim than the payload, and it is deliberate:
    "unknown" is checkable, a guess is not.

    Consequence worth knowing: a ``util.crop`` placed directly downstream resolves
    ``ctx.calib("origin_um")`` to ``None`` and so leaves the inherited origin unshifted.
    Its own window is still correct; only the origin bookkeeping degrades, and it degrades
    the way it already does for any Dataset with no origin.

    Structure rows are narrowed on **M** (a Point/Label/Track row on a position this crop
    dropped is removed and the survivors renumbered) but their Y/X are left alone, which is
    exactly ``util.crop``'s spatial behaviour — the ``__sampling__`` stamp is what stops a
    downstream voxel-for-voxel consumer reading them against the cropped grid.
    """
    from nodegraph.placement import source_window, translate, z_um_of_slice

    ds = ctx.inputs[0] if ctx.inputs else None
    if not isinstance(ds, Dataset):
        raise ValueError(
            "Crop To File: nothing is wired into `data` — that is the Dataset being "
            "cropped. The second socket, `region`, is the file whose stage position says "
            "WHERE to crop it.")
    ref = ctx.input("region")
    if not isinstance(ref, Dataset):
        raise ValueError(
            "Crop To File: nothing is wired into `region`. This node takes its crop "
            "rectangle from a second file's absolute stage position and field size, so "
            "there is nothing to crop TO until that file is connected. For a rectangle "
            "you name in pixels, use Crop (`util.crop`) instead.")
    prov = ds.image
    if prov is None:
        raise ValueError("Crop To File: the `data` input carries no image to crop.")
    ax = prov.axes

    modes = ctx.params.get("__modes__", {})
    warnings: List[str] = []

    # ── 1. the reference's box, in absolute microscope µm ─────────────────────
    box, ref_used = _reference_box(ctx, ref, str(modes.get("extent", "union")))
    margin = float(ctx.params.get("margin", 0.0) or 0.0)
    margin_z = float(ctx.params.get("margin_z", 0.0) or 0.0) if ctx.is_volume else 0.0
    box = translate(box,
                    float(ctx.params.get("offset_z", 0.0) or 0.0) if ctx.is_volume else 0.0,
                    float(ctx.params.get("offset_y", 0.0) or 0.0),
                    float(ctx.params.get("offset_x", 0.0) or 0.0))
    unpadded = box
    box = _padded(box, margin, margin_z)
    requested = box
    # Caught HERE rather than left to the overlap test below, which would be true but
    # useless: a zero-area box misses every field, so the honest-looking answer would be
    # "the region does not touch the data at all" and its three suggested causes — wrong
    # well, wrong flip, stale nudge — are all wrong. The cause is on this node's own
    # `margin` socket, so the message names it and the number it may not go past.
    if box.y1 <= box.y0 or box.x1 <= box.x0:
        side = min(unpadded.y1 - unpadded.y0, unpadded.x1 - unpadded.x0)
        raise ValueError(
            f"Crop To File: `margin` = {margin:g} µm has shrunk the region to nothing. The "
            f"reference's field is {unpadded.y1 - unpadded.y0:.4g} x "
            f"{unpadded.x1 - unpadded.x0:.4g} µm, so a negative margin has to stay above "
            f"-{0.5 * side:.4g} µm to leave anything behind. To crop to a BORDER of the "
            f"reference rather than its middle, crop to the whole field and take the "
            f"window you want with Crop (`util.crop`) downstream.")

    # ── 2. which position of `data` holds it ──────────────────────────────────
    hits = _overlapping(ds, ax, box)
    if not hits:
        raise ValueError(_miss_message(ds, ax, ref, box, ref_used))
    best_m, coverage, best_box = hits[0]
    others = [(m, f) for m, f, _b in hits[1:]]
    if others:
        warnings.append(
            f"the region straddles {len(hits)} positions of `data` — this crop is position "
            f"{best_m} ({coverage:.0%} of the region); "
            + ", ".join(f"m{m} ({f:.0%})" for m, f in others)
            + " also overlap it and are NOT in the output. Put Stitch (`util.stitch`) "
              "upstream to keep all of it in one field.")
    if coverage < 1.0 - 1e-9:
        warnings.append(
            f"the region reaches outside position {best_m}'s field — the crop is clamped "
            f"to the {coverage:.0%} that exists.")

    # ── 3. µm box → this file's pixel grid ────────────────────────────────────
    flip_x, flip_y, hand_warn = handedness_for(
        ds, bool(ctx.params.get("flip_x", True)), bool(ctx.params.get("flip_y", False)))
    warnings.extend(hand_warn)
    frac = source_window(box, best_box, flip_y=flip_y, flip_x=flip_x)
    if frac is None:                                     # unreachable after `_overlapping`
        raise ValueError(_miss_message(ds, ax, ref, box, ref_used))
    y0, y1 = _window_px(frac[0], frac[1], ax.y, "Y", coverage)
    x0, x1 = _window_px(frac[2], frac[3], ax.x, "X", coverage)

    # ── 4. Z, in 3D only ──────────────────────────────────────────────────────
    z0, z1, zs_um = 0, ax.z, None
    if ctx.is_volume:
        z0, z1, zs_um, z_warn = _z_window(ctx, ds, ax, best_m, box, z_um_of_slice)
        warnings.extend(z_warn)

    # ── 5. the lazy view ──────────────────────────────────────────────────────
    base = prov
    picks: Optional[Dict[str, Any]] = None
    if ax.m > 1:
        picks = {"m": (best_m,), "t": None, "z": None}
        base = FrameSubsetProvider(prov, ms=(best_m,), ts=tuple(range(ax.t)), zs=None)
    new_axes = replace(base.axes, z=z1 - z0, y=y1 - y0, x=x1 - x0)
    view = WindowView(base, z0=z0, y0=y0, x0=x0, axes=new_axes)

    out = ds
    if picks is not None:
        # M first, on the PRE-window axes: a Voxel mask still fits its (m,t,z,c,y,x) shape
        # here, so it can be reindexed instead of being thrown away by the reshape a line
        # later. The lateral window then drops whatever no longer fits, which is
        # `util.crop`'s own rule (`reshaped_axes(drop_stale=True)`) and reaches non-spatial
        # lattice domains — a Plane statistic, a Frame series — that survive both steps.
        out = subset_lattice_layers(out, replace(ax, m=1), picks)
        out = subset_structure_rows(out, picks)
    out = out.with_image(view).reshaped_axes(new_axes)

    # ── 6. metadata: the corner moves, the stage logs retire ──────────────────
    delivered = _delivered_box(best_box, ax, (y0, y1), (x0, x1), flip_y, flip_x,
                               (z0, z1), zs_um)
    changes: Dict[str, Any] = dict(drop_position_keys(ds.metadata))
    changes["origin_um"] = [[delivered.z0 if delivered.z0 is not None else 0.0,
                             delivered.y0, delivered.x0]]
    out = out.with_metadata(**changes)
    # The corner MOVED and M was re-picked, so this is not confined to any one axis and the
    # stamp carries no `<axes>:` prefix — an unmarked stamp is never dropped by
    # `_sampling_of`, which is the safe default and the correct one here.
    out = _sampled(out, f"crop_to[m{best_m},z{z0}:{z1},y{y0}:{y1},x{x0}:{x1}]")
    return out.with_metadata(**{CROP_TO_KEY: _note(
        ds, best_m, coverage, others, ref_used, requested, delivered,
        (z0, z1, y0, y1, x0, x1), warnings)})


# ── the reference's box ──────────────────────────────────────────────────────────

def _axially_placed(md: Any) -> bool:
    """Whether this file's metadata says anything real about WHERE IN FOCUS it was.

    The question exists because ``origin_um``'s Z slot lies convincingly when it does not
    know. ``nodelab_v2.ingest._origin_um_from_stage`` fills it with ``0.0`` for a file that
    has stage XY but no focus log — deliberately, because a lateral-only placement is what
    tile selection needs and demanding Z would cost the whole origin — and
    :func:`~nodegraph.placement.field_box` hands that ``0.0`` back as a ``FieldBox`` whose
    ``z0 == z1 == 0.0``. Read literally, that is "this file was acquired at absolute focus
    zero", which in 3D would cut every plane of a stack sitting at 100 µm and then explain
    the result as a focus offset the user would go looking for and never find.

    The discriminator is the presence of either focus key, not the value of the Z slot.
    ``stage_z_um`` is the raw record; ``z_step_um`` survives on its own past a
    ``util.stitch`` (which retires the per-position stage logs but keeps the spacing and a
    real union corner), so testing both keeps an already-stitched reference axially placed
    instead of demoting it.
    """
    return md.get("stage_z_um") is not None or md.get("z_step_um") is not None


def _unfocused(box):
    """``box`` with its axial span blanked — the ``FieldBox`` state that means "placeable
    laterally, not axially", which is a normal and useful one."""
    from nodegraph.placement import FieldBox
    return FieldBox(y0=box.y0, y1=box.y1, x0=box.x0, x1=box.x1, z0=None, z1=None)


def _reference_box(ctx: EvalContext, ref: Dataset, extent: str):
    """``(FieldBox, positions_used)`` for the reference file, in absolute microscope µm."""
    from nodegraph.placement import FieldBox, field_box

    md, ax = ref.metadata, ref.axes
    focused = _axially_placed(md)
    if extent == "position":
        m = int(ctx.params.get("ref_position", 0) or 0)
        if not (0 <= m < ax.m):
            raise ValueError(
                f"Crop To File: `ref_position` = {m} is out of range — the file on "
                f"`region` has {ax.m} position(s), so the only valid indices are "
                f"{'0' if ax.m == 1 else f'0..{ax.m - 1}'}. Switch `What defines the "
                f"region` to `union` to use all of them at once.")
        one = field_box(md, ax, m)
        if one is None:
            raise ValueError(_unplaceable_message(ref, f"position {m}"))
        return (one if focused else _unfocused(one)), (m,)

    boxes = [(m, field_box(md, ax, m)) for m in range(max(1, ax.m))]
    good = [(m, b) for m, b in boxes if b is not None]
    if not good:
        raise ValueError(_unplaceable_message(ref, "any position"))
    zs0 = [b.z0 for _m, b in good if b.z0 is not None] if focused else []
    zs1 = [b.z1 for _m, b in good if b.z1 is not None] if focused else []
    union = FieldBox(
        y0=min(b.y0 for _m, b in good), y1=max(b.y1 for _m, b in good),
        x0=min(b.x0 for _m, b in good), x1=max(b.x1 for _m, b in good),
        z0=min(zs0) if zs0 else None, z1=max(zs1) if zs1 else None)
    return union, tuple(m for m, _b in good)


def _padded(box, margin: float, margin_z: float):
    """``box`` grown by ``margin`` µm laterally and ``margin_z`` µm axially.

    A negative margin shrinks, which is the honest reading of the same number and is how
    you ask for the *inside* of the other file's field — so it is allowed rather than
    clamped at zero. It cannot produce an empty box unnoticed: `_window_px` refuses a
    window that collapses, naming the margin as a likely cause.
    """
    from nodegraph.placement import FieldBox
    return FieldBox(
        y0=box.y0 - margin, y1=box.y1 + margin,
        x0=box.x0 - margin, x1=box.x1 + margin,
        z0=None if box.z0 is None else box.z0 - margin_z,
        z1=None if box.z1 is None else box.z1 + margin_z)


# ── which position of `data` holds the region ────────────────────────────────────

def _overlapping(ds: Dataset, ax: Any, box) -> List[Tuple[int, float, Any]]:
    """``[(m, fraction_of_the_REGION_this_position_holds, its_box)]``, largest first.

    ``overlap_fraction(a, b)`` is deliberately asymmetric — it answers "how much of ``a``
    does ``b`` supply" — and the argument order here is the whole question this node asks:
    how much of the REGION does each of the cropped file's fields contain. The symmetric
    answer would be meaningless at the 36× area ratio the WellA3 pair have.
    """
    from nodegraph.placement import field_box, overlap_fraction

    hits: List[Tuple[int, float, Any]] = []
    for m in range(max(1, ax.m)):
        b = field_box(ds.metadata, ax, m)
        if b is None:
            continue
        f = overlap_fraction(box, b)
        if f > 0:
            hits.append((m, f, b))
    hits.sort(key=lambda h: -h[1])
    return hits


def _window_px(f0: float, f1: float, n: int, axis: str, coverage: float) -> Tuple[int, int]:
    """A fractional span of a pixel grid → a half-open index window that keeps every pixel
    the span touches.

    ``floor`` on the low end and ``ceil`` on the high end, so the window is the smallest
    one that CONTAINS the requested µm box rather than the largest one contained by it: a
    crop that silently lost the outermost row of the region would be a sub-pixel error
    nobody could see, and one extra row is a cost nobody can either.

    The snap before the rounding is not tidiness. An edge that lands EXACTLY on a pixel
    boundary is the ordinary case — two files on the same stage grid, a margin that is a
    whole number of pixels — and ``f * n`` reaches it as ``23.999999999999996`` or
    ``24.000000000000004`` depending on nothing the user can see, which ``ceil`` turns
    into a window that is one column wider on some inputs than others. Snapping inside a
    nanopixel makes the boundary case deterministic; it cannot affect a genuinely
    fractional edge, which is many orders of magnitude away from the tolerance.
    """
    def snap(v: float) -> float:
        r = round(v)
        return float(r) if abs(v - r) < 1e-9 else v

    lo = max(0, min(n - 1, int(math.floor(snap(f0 * n)))))
    hi = max(lo + 1, min(n, int(math.ceil(snap(f1 * n)))))
    if (f1 - f0) * n < _MIN_WINDOW_FRAC:
        raise ValueError(
            f"Crop To File: the region is {(f1 - f0) * n:.3g} pixels across in {axis} — "
            f"less than a tenth of one pixel of the file being cropped, so there is no "
            f"window to take. Either the two files are at wildly different scales (crop "
            f"the FINER one to the coarser one's field, not the other way round), or a "
            f"negative `margin` has collapsed the box. Coverage of the region by this "
            f"position was {coverage:.0%}.")
    return lo, hi


def _z_window(ctx: EvalContext, ds: Dataset, ax: Any, m: int, box,
              z_um_of_slice) -> Tuple[int, int, Optional[List[float]], List[str]]:
    """``(z0, z1, per-plane absolute focus, warnings)`` — the planes of position ``m``
    whose focus falls inside the reference's axial span.

    Degrades to the whole stack rather than guessing, three ways, because each of them is
    an ordinary state rather than a fault: a single-plane file has no range to cut, a file
    with no ``ZStackLoop`` anchoring cannot say where plane *k* is (the plausible guess,
    ``home = 0``, is wrong by up to the full stack depth — 60 µm on the WellA3 640 file),
    and a reference with no focus log places laterally only, which is what a widefield
    montage always does.

    The window is CONTIGUOUS, because :class:`~nodegraph.streaming.WindowView` translates
    an offset rather than remapping an index list. That costs nothing here: focus rises or
    falls monotonically down a stack, so the planes inside an interval are contiguous by
    construction.
    """
    warn: List[str] = []
    if ax.z <= 1:
        return 0, ax.z, None, warn
    if box.z0 is None or box.z1 is None:
        return 0, ax.z, None, [
            "the file on `region` carries no focus log, so its axial span is unknown and "
            "every Z plane was kept. The lateral crop is unaffected."]
    zs_um = [z_um_of_slice(ds.metadata, ax, m, k) for k in range(ax.z)]
    if any(v is None for v in zs_um):
        return 0, ax.z, None, [
            "the file on `data` has a Z stack with no ZStackLoop anchoring, so which "
            "absolute focus each plane sits at is unknown and every plane was kept. The "
            "lateral crop is unaffected."]
    # Half a Z step of slack on each side: a plane is kept when the interval reaches its
    # SLAB, not only its centre. Without it a reference span that happens to fall between
    # two plane centres keeps nothing at all, which is the `_COVERAGE_TOL` lesson
    # (`nodegraph/placement.py`) in the axial direction.
    try:
        slack = 0.5 * abs(float(ctx.calib("z_step_um") or 0.0))
    except (TypeError, ValueError):                       # pragma: no cover - defensive
        slack = 0.0
    lo, hi = min(box.z0, box.z1) - slack, max(box.z0, box.z1) + slack
    keep = [k for k, v in enumerate(zs_um) if lo <= v <= hi]
    if not keep:
        return 0, ax.z, zs_um, [
            f"no plane of position {m} lies inside the reference's focus range "
            f"({min(box.z0, box.z1):.1f} to {max(box.z0, box.z1):.1f} µm) — the two "
            f"acquisitions do not overlap in Z, so every plane was kept rather than "
            f"producing nothing. Widen `margin_z`, or check `offset_z`."]
    z0, z1 = keep[0], keep[-1] + 1
    if z1 - z0 != len(keep):                              # pragma: no cover - defensive
        warn.append(f"the planes inside the reference's focus range were not contiguous; "
                    f"kept the span {z0}:{z1} that covers them all.")
    if (z0, z1) != (0, ax.z):
        warn.append(f"kept Z planes {z0}:{z1} of {ax.z} — the range that overlaps the "
                    f"reference's focus span.")
    return z0, z1, zs_um, warn


# ── what actually came out ───────────────────────────────────────────────────────

def _delivered_box(src, ax: Any, ys: Tuple[int, int], xs: Tuple[int, int],
                   flip_y: bool, flip_x: bool, zs: Tuple[int, int],
                   zs_um: Optional[Sequence[float]]):
    """The µm box the returned pixels actually cover — the payload's new ``origin_um``.

    Mapped back through ``src`` (the chosen position's field) rather than taken from the
    requested box, so it is true after the clamp AND after the floor/ceil rounding. The
    flip is undone here for the same reason it was applied: it says which SOURCE column an
    output column reads, never where that column is in µm, so the delivered span is
    ordered low-to-high either way — which is what ``origin_um`` means and what
    :func:`~nodegraph.placement.field_box` reads back out of it.
    """
    from nodegraph.placement import FieldBox

    ylo, yhi = _um_span(ys[0], ys[1], ax.y, src.y0, src.y1, flip_y)
    xlo, xhi = _um_span(xs[0], xs[1], ax.x, src.x0, src.x1, flip_x)
    if zs_um is not None and zs[1] > zs[0]:
        zlo, zhi = min(zs_um[zs[0]], zs_um[zs[1] - 1]), max(zs_um[zs[0]], zs_um[zs[1] - 1])
    else:
        zlo, zhi = src.z0, src.z1
    return FieldBox(y0=ylo, y1=yhi, x0=xlo, x1=xhi, z0=zlo, z1=zhi)


def _um_span(lo_px: int, hi_px: int, n: int, b_lo: float, b_hi: float,
             flip: bool) -> Tuple[float, float]:
    """The ``(low, high)`` µm covered by pixel rows/columns ``[lo_px, hi_px)`` of a field
    spanning ``[b_lo, b_hi]``."""
    if n <= 0:
        return b_lo, b_hi
    step = (b_hi - b_lo) / float(n)
    if flip:
        return b_hi - hi_px * step, b_hi - lo_px * step
    return b_lo + lo_px * step, b_lo + hi_px * step


def _note(ds: Dataset, m: int, coverage: float, others: Sequence[Tuple[int, float]],
          ref_used: Sequence[int], requested, delivered,
          window: Tuple[int, int, int, int, int, int],
          warnings: Sequence[str]) -> Dict[str, Any]:
    """The :data:`CROP_TO_KEY` record: what this crop took, and what it therefore left.

    It carries the per-position identity (``source_file`` / ``position_name``) of the
    position it kept, because :func:`~nodegraph.metadata.drop_position_keys` has just
    retired those LISTS off the payload — one field's worth of provenance belongs in a
    note, not in a length-1 positional list a downstream reader would index into as
    calibration.
    """
    md = ds.metadata

    def per_m(key: str):
        vals = md.get(key)
        if isinstance(vals, (list, tuple)) and 0 <= m < len(vals):
            return vals[m]
        return None

    head = f"took position {m}"
    if int(getattr(ds.axes, "m", 1) or 1) > 1:
        head += f" of {ds.axes.m}"
    head += f" ({coverage:.0%} of the region)"
    if others:
        head += ("; " + ", ".join(f"m{o} ({f:.0%})" for o, f in others)
                 + " also overlap and are NOT in this crop")
    note = head + "."
    if warnings:
        note += "  ·  " + warnings[0]
    return {
        "note": note,
        "position": int(m),
        "coverage": round(float(coverage), 6),
        "other_positions": [[int(o), round(float(f), 6)] for o, f in others],
        "source_file": per_m("source_file"),
        "position_name": per_m("position_name"),
        "ref_positions": [int(v) for v in ref_used],
        "requested_um": _box_list(requested),
        "delivered_um": _box_list(delivered),
        "window_px": [int(v) for v in window],
        "warnings": list(warnings),
    }


def _box_list(box) -> List[Optional[float]]:
    """A ``FieldBox`` as ``[y0, y1, x0, x1, z0, z1]`` — JSON-shaped, so the note survives a
    checkpoint round-trip and cannot make ``output_fingerprint`` depend on object identity."""
    return [round(float(box.y0), 4), round(float(box.y1), 4),
            round(float(box.x0), 4), round(float(box.x1), 4),
            None if box.z0 is None else round(float(box.z0), 4),
            None if box.z1 is None else round(float(box.z1), 4)]


# ── refusals ─────────────────────────────────────────────────────────────────────

def _unplaceable_message(ref: Dataset, which: str) -> str:
    md = ref.metadata
    missing = []
    if not md.get("pixel_size_um"):
        missing.append("`pixel_size_um` (without it a pixel count is not a field SIZE)")
    if not md.get("origin_um") and not md.get("stage_xy_um"):
        missing.append("`origin_um` / `stage_xy_um` (without one of them the field has no "
                       "position on the stage)")
    return (
        f"Crop To File: cannot place {which} of the file on `region` — it is missing "
        + (", and ".join(missing) if missing else "a usable stage record")
        + ". A TIFF, a synthetic Dataset, or an ND2 whose stage log does not cover every "
          "multipoint has nothing to crop TO. Load the reference from a file with stage "
          "metadata, or use Crop (`util.crop`) with a pixel window instead.")


def _miss_message(ds: Dataset, ax: Any, ref: Dataset, box, ref_used: Sequence[int]) -> str:
    from nodegraph.placement import field_box

    own = [field_box(ds.metadata, ax, m) for m in range(max(1, ax.m))]
    placed = [b for b in own if b is not None]
    if not placed:
        return _unplaceable_message(ds, "any position").replace(
            "the file on `region`", "the file on `data`")
    y0 = min(b.y0 for b in placed)
    y1 = max(b.y1 for b in placed)
    x0 = min(b.x0 for b in placed)
    x1 = max(b.x1 for b in placed)
    return (
        f"Crop To File: the region does not touch the data at all. The reference "
        f"(position{'s' if len(ref_used) != 1 else ''} "
        f"{', '.join(str(v) for v in ref_used)}) covers y {box.y0:.0f}..{box.y1:.0f} µm, "
        f"x {box.x0:.0f}..{box.x1:.0f} µm; the {len(placed)} placeable position(s) of "
        f"`data` together cover y {y0:.0f}..{y1:.0f} µm, x {x0:.0f}..{x1:.0f} µm. Those "
        f"are disjoint. Usual causes, in order: the two files are from different wells or "
        f"different sessions; `Flip X`/`Flip Y` is wrong for this camera mount (try "
        f"toggling Flip X); or `Nudge X`/`Nudge Y` is carrying a value from another pair.")


# ── registration ─────────────────────────────────────────────────────────────────

#: Shared tail for the two lateral nudges — what differs between them is the axis, and
#: everything else (that it is µm and not pixels, that it moves the CROP not the data, when
#: you would reach for it) is the same sentence twice.
_NUDGE_DOC = (
    "Shifts the crop rectangle along {axis} by this many microns BEFORE it is cut — it "
    "moves the window, never the pixels, so nothing is resampled and the output size is "
    "unchanged unless the shift pushes the box off an edge. Leave it at 0 unless you have "
    "measured a real offset between the two files: a stage that was re-homed between "
    "acquisitions, or a reference whose recorded position you already know is out. "
    "Positive is toward increasing stage {axis}. It is recorded in the node's "
    "`__crop_to__` note as part of the requested box, so a nudge left over from another "
    "pair of files is visible rather than baked in invisibly.")

register_node(
    _compute_crop_to, op_key="util.crop_to", label="Crop To File", category="utility",
    inputs=[
        InDataset(description=
                  "THE DATASET BEING CROPPED — the overview, the montage, the stitched "
                  "canvas. Its calibration, its layers and its domains are what the output "
                  "inherits, and its pixel grid is the one the crop window is expressed "
                  "in. It needs a stage record of its own (`origin_um`, or `stage_xy_um` "
                  "plus `pixel_size_um`): without one there is no way to say where the "
                  "other file falls inside it."),
        InDataset("region", label="Region", description=
                  "THE FILE THAT SAYS WHERE TO CROP — its absolute stage position and its "
                  "field-of-view size define the rectangle, and nothing else about it is "
                  "used. NO PIXELS ARE READ FROM IT: it is consulted for placement only, "
                  "so it can be the raw acquisition rather than any processed branch of "
                  "it, and it is not drawn in the viewer. It may be at a completely "
                  "different magnification — that is the point, since a 0.287 µm/px field "
                  "and a 1.718 µm/px field are both 1024 pixels wide and 6x different in "
                  "size. It needs `pixel_size_um` and a stage position; a TIFF usually has "
                  "neither."),
        InInt("ref_position", "Reference position", unit="", default=0, field=False,
              available_in={"extent": frozenset({"position"})},
              description=
              "WHICH multipoint of the file on `region` defines the rectangle, 0-based. "
              "Only read when `What defines the region` is `position` — under `union` it "
              "is inert and hidden. Use it when the reference visited several sites and "
              "you want the crop to be one of them rather than the whole tour: with a "
              "9-site reference, `union` gives you a rectangle big enough to hold all "
              "nine (including the empty stage between them), and this gives you site N. "
              "An index past the end is refused rather than clamped, because clamping "
              "would quietly crop somewhere you did not ask for."),
        InFloat("margin", "Margin", unit="um", default=0.0, field=False,
                description=
                "EXTRA CONTEXT around the region, in microns, added on all four lateral "
                "sides before the cut. 0 gives exactly the other file's footprint; 50 "
                "gives that plus a 50 µm border, which is what you want when the crop is "
                "going to be looked at rather than measured — a field with no surroundings "
                "is hard to locate by eye. It makes the output BIGGER, so it raises the "
                "cost of everything downstream, and it is clamped at the data's edge like "
                "any other overhang. NEGATIVE shrinks instead, which is how you ask for "
                "the inside of the reference's field and trim its vignetted border; shrink "
                "past the middle and the crop is refused rather than inverted. Lateral "
                "only — in 3D the axial direction has its own `Margin Z`, because a "
                "micron across the field and a micron of focus are not interchangeable."),
        InFloat("margin_z", "Margin Z", unit="um_axial", default=0.0, field=False,
                available_in={"dim": frozenset({"3D"})},
                description=
                "EXTRA PLANES above and below the reference's focus range, in microns of "
                "depth. 3D only; in 2D no plane is ever cut and this is inert. Its reason "
                "to exist is that two files' focus logs rarely agree to the micron — a "
                "reference stack that nominally spans 12 µm can easily sit 2 µm off the "
                "overview's idea of the same plane — so a range taken literally can keep "
                "fewer planes than the object needs, or none at all. Widen this until the "
                "node's note stops warning that the two acquisitions miss each other in Z. "
                "It adds whole planes, so it moves the output's Z size in steps of "
                "`z_step_um`, not continuously."),
        InFloat("offset_y", "Nudge Y", unit="um", default=0.0, field=False,
                description=_NUDGE_DOC.format(axis="Y")),
        InFloat("offset_x", "Nudge X", unit="um", default=0.0, field=False,
                description=_NUDGE_DOC.format(axis="X")),
        InFloat("offset_z", "Nudge Z", unit="um_axial", default=0.0, field=False,
                available_in={"dim": frozenset({"3D"})},
                description=
                "Shifts the reference's FOCUS RANGE by this many microns before the Z "
                "planes are chosen. 3D only; inert in 2D, where no plane is cut. Use it "
                "when the two acquisitions used a different focus reference — a different "
                "objective, a re-set Z home, a coverslip correction — so that their "
                "recorded depths are offset by a constant. It does not move the lateral "
                "window at all."),
        InBool("flip_x", "Flip X", default=True, field=False,
               description=
               "Whether image +X runs OPPOSITE stage +X on this microscope — the camera's "
               "mounting handedness, which no file records, so it has to be a setting. It "
               "decides which end of the data's own pixel grid the region's µm box lands "
               "on, so getting it wrong does not shrink or grow the crop, it MIRRORS where "
               "the crop is taken from: the output looks like plausible data from the "
               "wrong place, which is the worst failure mode this node has. The tell is a "
               "crop that misses by roughly the width of one field, or a 'does not touch "
               "the data at all' refusal on two files you know overlap. Default on, "
               "matching Merge and Overlay. IGNORED, with a note, when the data is an "
               "already-stitched canvas: a stitch has already resolved handedness and its "
               "output is laid out in stage coordinates, so flipping again would mirror "
               "the mosaic inside its own footprint (measured 784 µm out on the WellA3 "
               "pair)."),
        InBool("flip_y", "Flip Y", default=False, field=False,
               description=
               "The Y twin of Flip X — whether image +Y runs opposite stage +Y. Default "
               "OFF, because the common mounting inverts X only; turn it on if the crop "
               "lands mirrored top-to-bottom rather than left-to-right. Same failure mode "
               "and the same tell, and it is ignored under the same condition (an "
               "already-stitched input)."),
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("extent", ["union", "position"], default="union",
             label="What defines the region",
             description=
             "WHICH PART of the file on `region` supplies the rectangle. It only matters "
             "for a reference that visited more than one site — for an ordinary "
             "single-position acquisition the two answers are identical, which is why the "
             "default needs no thought. The choice folds into the memo key, so switching "
             "between them and back is free.",
             choice_docs={
                 "union": "The bounding box of EVERY position the reference file can be "
                          "placed at. For a single-position file that is simply its field. "
                          "For a multi-site file it is one rectangle big enough to hold "
                          "all of them — including whatever empty stage lies between the "
                          "sites, so a 9-well tour gives you a crop spanning the whole "
                          "tour rather than nine crops. Positions the reference cannot "
                          "place (a missing stage entry) are skipped rather than failing "
                          "the union.",
                 "position": "One named multipoint of the reference, chosen with the "
                             "`Reference position` socket this option reveals. Use it to "
                             "crop to a single site of a multi-site reference. An index "
                             "past the end of the file is refused, naming the valid "
                             "range, rather than being clamped to the last position.",
             }),
    ],
    granularity=_DIM_GRAN, kernel_axes=frozenset(),
    meta_transform=_meta_crop_to,
    description="Crop to the field another file occupies in absolute stage coordinates; "
                "M collapses to the covering position, pixel size preserved.")
