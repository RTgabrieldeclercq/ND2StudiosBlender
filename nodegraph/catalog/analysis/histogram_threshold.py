"""Histogram Threshold (``analysis.histogram_threshold``) — Histogram threshold segmentation (2D per-plane) → Voxel mask + Label raster + region table; 4 methods × 4 directions, morphology cleanup (ported v1 kernel)."""

from __future__ import annotations

import numpy as np
import warnings

from dataclasses import replace
from typing import Any, Dict, Optional

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.labels import (
    _label_raster,
    _resolve_label_instance,
    _resolve_layer,
    _voxel_layers,
)
from nodegraph.catalog._shared.planes import _each_plane_p
from nodegraph.catalog._shared.raw_measure import _InRaw, _intensity_provider
from nodegraph.catalog._shared.scope import (
    STRUCTURE_SCOPES,
    ScopeMode,
    roi_populations,
    scope_declarations,
    scope_is_3d,
    unit_populations,
)
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Histogram Threshold Segmenter (2D → mask + labels + region table) ──────────

#: the bit depths the vendored kernel's LUT tables support (``BIT_DEPTH_MAX``).
_KERNEL_BIT_DEPTHS = (8, 10, 12, 14, 16)
#: This node's slice of the shared population vocabulary (V2.27). `volume`/`series`/`dataset`
#: are absent because the vendored engine derives its level from ONE plane by construction —
#: offering them would be a control it could not honour. `per_plane=True` keeps the footprint
#: honest for the two structure scopes: this node reads a plane at a time and REFUSES a
#: volumetric Label instance rather than cutting one object at several levels.
_HT_SCOPES = ("plane",) + STRUCTURE_SCOPES
_HT_GRAN, _HT_READS = scope_declarations(_HT_SCOPES, per_plane=True)
def _snap_bit_depth(bits: Optional[float]) -> int:
    """The smallest kernel-supported depth that still holds ``bits`` (11 → 12, 12 → 12),
    or 16 when unknown. The kernel indexes ``BIT_DEPTH_MAX`` by exact key, so an odd
    sensor depth must round UP — never down, which would truncate the histogram LUT
    below the data and silently drop the bright tail out of every percentile."""
    if not bits:
        return 16
    b = int(bits)
    return next((d for d in _KERNEL_BIT_DEPTHS if d >= b), 16)
#: the (low-side, high-side) threshold param of each histogram-threshold method — what
#: ``direction`` selects, and what the inspector shows (mirrored by ``available_in``).
_THRESH_PARAMS = {"single": ("low", "high"),
                  "hysteresis": ("permissive", "strict"),
                  "percentile": ("percentile_low", "percentile_high"),
                  "relative": ("fraction_low", "fraction_high")}
#: image-relative defaults (percentile of the plane histogram / × its median) — **the v1
#: segmenter's own numbers** (`HistogramThresholdPipeline.get_params`: percentile 5/95,
#: fraction 0.30/1.50), not re-invented here. The absolute raw-count params
#: (low/high/strict/permissive) are deliberately ABSENT — v1 defaulted them to 30/80/50/500
#: for its 12-bit data, which is meaningless on another camera, so leaving one unset must
#: ask rather than guess. Keep in lockstep with the socket ``default=``s below (the engine
#: passes only stored params).
_THRESH_DEFAULTS = {"percentile_low": 5.0, "percentile_high": 95.0,
                    "fraction_low": 0.30, "fraction_high": 1.50}
def _resolve_populations(ctx: EvalContext, ds: Dataset, ax, scope: str):
    """``(populations6, layer)`` for a structure scope — the raster whose ids are the arenas.

    ``per_label`` takes a whole Label INSTANCE (a raster plus the table that proves its ids
    divide the foreground) and **refuses a volumetric one**: this node wraps a segmenter
    hardcoded to ``is_3d=False`` (``kernels/histogram_threshold.py``), so it can only ever
    derive a level from one PLANE of a region. Running it over a 3D segmentation would cut one
    object at a different level on every plane it spans — the precise defect the inherited-
    dimensionality rule exists to prevent — so it says so and names the node that does handle it.

    ``per_roi`` takes any Voxel mask and makes its own ids per plane, so a drawn ROI works with
    no segmentation upstream."""
    want = ctx.layer("regions")
    if scope == "per_label":
        layer, _note = _resolve_label_instance(
            ds, want, node="histogram threshold", socket="regions",
            remedy="scope=per_label derives one level per REGION, so it needs a label raster "
                   "AND its table — run analysis.segment / analysis.label upstream, or set "
                   "Scope back to plane",
            ctx=ctx)
        pops6, zk = _label_raster(ds, layer, node="histogram threshold")
        if scope_is_3d(zk, ax.z):
            raise ValueError(
                f"histogram threshold: the Label instance {layer!r} is VOLUMETRIC "
                f"(z_kind='subpixel'), but this node's segmenter is 2D by construction — it "
                f"would derive a different level for every plane one object spans, and report "
                f"each plane's slice as its own sub-object. Use analysis.threshold with "
                f"scope=per_label (it inherits the 3D population correctly, and emits a mask), "
                f"or segment with the 2D/3D lever on 2D so each plane's regions are their own "
                f"objects.")
        return pops6, layer
    layer, _note = _resolve_layer(
        _voxel_layers(ds), want, node="histogram threshold", socket="regions",
        what="Voxel mask", where="the `data` input",
        remedy="scope=per_roi derives one level per connected region of a MASK, so it needs "
               "one — draw it with analysis.roi_mask, or set Scope back to plane",
        ctx=ctx)
    attr = ds.get(Domain.VOXEL, layer)
    if attr is None:                                    # pragma: no cover - _resolve_layer
        raise ValueError(f"histogram threshold: no Voxel layer {layer!r}")
    return np.asarray(attr.values), layer


def _level_used(used: dict) -> float:
    """The single level the kernel actually applied, out of its ``threshold_used`` record.

    Reported so the per-parent number is auditable in the Spreadsheet — the whole point of
    deriving it per region is that the regions disagree, and a column of NaN-or-number is how
    the user sees which ones had a histogram at all. Prefers the HIGH side (the usual
    fluorescence direction); falls back to the low one for a `below`/`between` cut."""
    for key in ("high_resolved", "low_resolved"):
        v = used.get(key)
        if v is not None and np.isfinite(float(v)):
            return float(v)
    return float("nan")


def _segment_populations(seg, plane_int: np.ndarray, pops: np.ndarray, vox, *,
                         min_pixels: int, quiet: bool,
                         report: Optional[np.ndarray] = None, unit_scale: float = 1.0):
    """Run the vendored segmenter ONCE PER POPULATION over one plane.

    Returns ``(mask, labels, regions, pstats)`` — the plane's union mask, its densely numbered
    sub-object labels, the per-sub-object region dicts (each carrying ``parent_id`` and the
    ``level`` its parent used), and ``{parent_id: (level, n_above, n_total, n_sub)}``.

    **Why per parent rather than one call with a mask.** Three of the kernel's stages are
    whole-frame: ``apply_spatial_constraints`` (the µm morphology and the area window) and
    ``label(mask, connectivity=2)``. A single call would therefore (a) let two parents' selected
    pixels merge into one component across their shared border — so a sub-object would belong to
    two cells and ``parent_id`` would silently pick a winner — and (b) apply the opening/closing
    across parent boundaries. Cropping to each parent's bounding box makes both impossible by
    construction, and costs work proportional to the parent's own extent rather than the frame's.

    The mask is intersected with the parent before its components are counted, because the
    kernel legitimately thresholds every pixel of the box it was handed, including the parts
    belonging to neighbours."""
    from skimage.measure import label as _cc
    from nodegraph.kernels.histogram_threshold import _measure_regions
    mask = np.zeros(plane_int.shape, dtype=bool)
    labels = np.zeros(plane_int.shape, dtype=np.int32)
    regions: list = []
    pstats: Dict[int, tuple] = {}
    k = 0
    for pid, idx, vals in unit_populations(pops, plane_int):
        n_total = int(idx.size)
        if n_total < min_pixels or float(np.ptp(vals)) <= 0.0:
            pstats[pid] = (float("nan"), 0, n_total, 0)
            continue
        co = np.unravel_index(idx, plane_int.shape)
        y0, y1 = int(co[0].min()), int(co[0].max()) + 1
        x0, x1 = int(co[1].min()), int(co[1].max()) + 1
        box = np.ascontiguousarray(plane_int[y0:y1, x0:x1])
        pm = np.zeros(box.shape, dtype=bool)
        pm[co[0] - y0, co[1] - x0] = True
        with warnings.catch_warnings():
            if quiet:
                warnings.filterwarnings("ignore", message=r".*is <5% of.*",
                                        category=UserWarning)
            try:
                res = seg.run(box, reference_mask=pm, voxel_size=vox)
            except ValueError:            # the method declined this population
                pstats[pid] = (float("nan"), 0, n_total, 0)
                continue
        bm = np.asarray(res.mask, dtype=bool) & pm
        # back out of the kernel's fixed point, so `level` is comparable to the pixel values
        # the table reports and to the number the user typed
        lvl = _level_used(res.threshold_used or {}) / (unit_scale or 1.0)
        if not bm.any():
            pstats[pid] = (lvl, 0, n_total, 0)
            continue
        # re-CCL the parent-confined mask: the kernel's own labelling was over the whole box,
        # so clipping it to the parent could otherwise leave one id on two disjoint pieces
        blab = _cc(bm, connectivity=2).astype(np.int32)
        n_sub = int(blab.max())
        # measured on the image's OWN values when a scale was applied, so the table never
        # reports this node's private fixed-point integers (V2.28)
        rbox = box if report is None else report[y0:y1, x0:x1]
        sub = _measure_regions(blab, rbox, vox)
        for r in sub:
            r = dict(r)
            r["label_id"] = int(r["label_id"]) + k
            r["centroid_y"] = float(r["centroid_y"]) + y0
            r["centroid_x"] = float(r["centroid_x"]) + x0
            r["parent_id"] = int(pid)
            r["level"] = lvl
            regions.append(r)
        mask[y0:y1, x0:x1] |= bm
        labels[y0:y1, x0:x1] = np.where(blab > 0, blab + k, labels[y0:y1, x0:x1])
        k += n_sub
        pstats[pid] = (lvl, int(bm.sum()), n_total, n_sub)
    return mask, labels, regions, pstats


def _compute_histogram_threshold(ctx: EvalContext) -> Dataset:
    """Histogram-driven threshold segmentation (ported v1 kernel) → a Voxel ``mask`` +
    a Label raster + a per-region **Label** table, all in one node (the v1 segmenter's
    bundle). **2D-only** — each ``(m,t,z,c)`` plane is segmented independently: one of
    4 methods (single / hysteresis / percentile / relative) × 4 directions, fixed-order
    morphology cleanup, 8-connected CCL, and region props.

    Thresholds (``low``/``high``/``strict``/``permissive``) are in the image's **raw
    integer counts** — the plane is cast to ``uint16`` (rint+clip to 0..65535) before
    the kernel, whose bit-depth check is set non-strict (``make_config``). The declared
    depth is **read from the file** (``bit_depth`` calibration ← ND2
    ``bitsPerComponentSignificant``, usually **12**), snapped up to a kernel-supported
    LUT (8/10/12/14/16) and overridable per node; only a silent metadata falls back to 16.
    That sizes the percentile LUT to the real sensor range and retires the bogus "max 192
    is <5% of 16-bit range" warning on dim 12-bit data (2026-07-28). Ids are
    global-unique across planes, and the two Voxel rasters are stored at the kernel's own
    dtypes — ``mask`` ``uint8`` (0/1) and the label raster ``int32`` — never upcast to
    int64, which cost 12 B/voxel of nothing (5 GiB per raster on a 16-position 2048²×10
    series). Morphology radii convert µm→px; area filters µm²→px².
    ``min_area``/``max_area`` bound the region size window (``max_area`` 0 = no upper
    limit; a sub-pixel value is refused, since rounding it to 0 px² would read as that
    off-sentinel and invert the request — restored socket 2026-07-28, the compute read
    the param all along but nothing could set it). Because that window is authored in
    **µm²**, the Label table reports **``area_um2``** beside the raw ``area`` (px²), so
    the domain data can be read in the same unit the filter is typed in; and a window
    that discards *every* region raises with the areas actually measured on one plane
    instead of returning a blank mask (2026-07-28).

    The optional **``raw``** Dataset input re-measures the region table's intensity columns
    on those pixels while the mask/labels stay on the main input's — segment on the chain
    you tuned, report intensities from the unenhanced source (:func:`_intensity_provider`).

    **Threshold params are method-specific** (``available_in`` gates each one to its
    method/direction, so the inspector shows only the 1–2 that the current mode pair
    actually uses) and **0 = unset** — the GUI's spin boxes cannot express ``None``, and
    a threshold of exactly 0 counts / the 0th percentile / 0× the median is meaningless,
    so zero reads as "not set". An unset param falls back to the image-relative default
    where one exists (``percentile``/``relative``); the absolute raw-count methods
    (``single``/``hysteresis``) have no image-independent default, so an unset one raises
    an actionable error naming the fields to fill in (fix 2026-07-28 — the vendored
    ``ThresholdConfig`` raised a bare "requires `strict` and `permissive`" traceback).

    Resolved spec: category analysis; op ``analysis.histogram_threshold``; no lever
    (inherently per-plane), ``WHOLE_PLANE`` ``{y,x}``. Kernel:
    :mod:`nodegraph.kernels.histogram_threshold` (numpy/scipy/skimage)."""
    from nodegraph.kernels.histogram_threshold import (
        HistogramThresholdSegmenter, _measure_regions, make_config,
    )
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("histogram threshold needs an image provider on its input Dataset")
    # optional `raw`: the mask/labels stay on the main input's pixels (thresholds are the
    # user's, on the chain they tuned), but the region table's intensity columns are
    # re-measured on the raw pixels — the same vendored `_measure_regions`, unquantized
    # (a measurement has no reason to round through uint16 the way a raw-count cut does).
    raw_prov, on_raw = _intensity_provider(ctx, ds)
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    # NO fabricated calibration. v1 passed ``voxel_size=None`` when pixel_size_um was
    # absent (areas then reported in px only); the old ``or 0.1`` here invented a scale,
    # which silently rescaled every µm/µm² filter by (real/0.1)² on an uncalibrated TIFF.
    px = ctx.calib("pixel_size_um")
    pxv = float(px) if px else 0.0
    # The SIGNIFICANT sensor depth, read from the file's metadata (most ND2s are 12-bit)
    # — not assumed to be 16. It sets the kernel's percentile LUT range and its bit-depth
    # validation, so declaring 16 on 12-bit data both mis-sized the LUT and produced the
    # bogus "max 192 is <5% of 16-bit range" warning. `bit_depth` is a v2 calibration key,
    # so this read is memo-fenced: re-ingesting the same file at another depth re-keys.
    bd_override = ctx.params.get("bit_depth")
    bd_from_meta = ctx.calib("bit_depth")
    bit_depth = (_snap_bit_depth(bd_override) if bd_override
                 else _snap_bit_depth(bd_from_meta))
    #: authoritative = the depth came from the file (or the user), so the kernel's
    #: "may be lower-bit than declared" heuristic has nothing left to warn about — a dim
    #: 12-bit frame legitimately maxes in the low hundreds. Kept when we defaulted to 16.
    bd_authoritative = bool(bd_override or bd_from_meta)

    def _uncalibrated(name: str, v: float, unit: str) -> str:
        return (f"histogram threshold: {name}={v:g} {unit} cannot be converted to pixels "
                "— this image carries no `pixel_size_um` calibration. Supply it upstream "
                "(an ND2 carries it; a bare TIFF usually does not), or set the param to 0 "
                f"to switch that filter off. (v1 authored these filters in PIXELS; v2 "
                f"authors them in {unit} so they follow the objective.)")

    def _cleanup_um(name: str) -> float:
        """A cleanup param in µm/µm², resolved override → ``derive`` → default. The
        derives encode **the v1 segmenter's own pixel defaults** (1 px opening, 2 px
        closing, 50 px² holes, 100 px² min area) expressed in µm, so the shipped
        behaviour matches v1 while still following the objective. Uncalibrated ⇒ the
        derives yield 0 ⇒ cleanup is simply off, v1's ``voxel_size=None`` path.
        (``channel(0)`` because none of these vary per channel — pixel size only.)"""
        return float(ctx.channel(0).param(name, 0.0) or 0.0)

    def _rad(name: str) -> int:                               # µm radius → px
        v = _cleanup_um(name)
        if v == 0.0:
            return 0
        if pxv <= 0.0:
            raise ValueError(_uncalibrated(name, v, "µm"))
        return int(round(to_pixels_v2(v, "um", pixel_size_um=pxv)))

    def _area(name: str) -> int:                              # µm² area → px²
        v = _cleanup_um(name)
        if v == 0.0:
            return 0
        if pxv <= 0.0:
            raise ValueError(_uncalibrated(name, v, "µm²"))
        return int(round(v / (pxv * pxv)))

    def _opt(name: str):
        """A method-specific threshold param, or ``None`` when unset. **0 = unset** (see
        the docstring); an unset param falls back to its image-relative default when the
        method has one — the engine does NOT inject socket defaults into ``ctx.params``,
        so `_THRESH_DEFAULTS` (mirrored by the socket ``default=``s) is the real source."""
        v = ctx.params.get(name)
        if v is None or v == "" or float(v) == 0.0:
            return _THRESH_DEFAULTS.get(name)
        return v

    # area filters (µm² → px²). ``min_area``/``min_hole_size`` quantizing to 0 px² is
    # harmless — "drop objects below half a pixel" IS a no-op. ``max_area`` is the
    # opposite: 0 is its off-sentinel, so a sub-pixel value would silently INVERT the
    # request (keep everything instead of dropping all but sub-pixel specks) → refuse.
    min_area_px, max_area_px = _area("min_area"), _area("max_area")
    max_area_um2 = _cleanup_um("max_area")
    if max_area_um2 > 0.0 and max_area_px == 0:
        raise ValueError(
            f"histogram threshold: max_area={max_area_um2:g} µm² is under one pixel "
            f"({pxv * pxv:g} µm²) — it quantizes to 0 px², which is this filter's "
            "OFF value, so it would keep every blob instead of dropping the large "
            "ones. Raise it, or set exactly 0 to disable the filter deliberately.")
    if max_area_px > 0 and min_area_px > max_area_px:
        raise ValueError(
            f"histogram threshold: min_area ({min_area_px} px²) exceeds max_area "
            f"({max_area_px} px²) — the size window is empty, so every region would be "
            "discarded. Widen the window (max_area 0 = no upper limit).")

    method = modes.get("method", "percentile")
    direction = modes.get("direction", "above")
    # Pre-flight the method/direction contract HERE, so a missing threshold reads as an
    # instruction instead of a dataclass traceback out of the vendored kernel.
    if method not in _THRESH_PARAMS:
        raise ValueError(f"histogram threshold: unknown method {method!r} — expected one "
                         f"of {', '.join(_THRESH_PARAMS)}.")
    if method == "hysteresis" and direction not in ("below", "above"):
        raise ValueError(
            "histogram threshold: method=hysteresis supports direction=below/above only "
            f"(got {direction!r}) — hysteresis grows a permissive region out of a strict "
            "seed, which has no two-sided form. Use method=single or percentile for "
            f"direction={direction}.")
    lo, hi = _THRESH_PARAMS[method]
    if method == "hysteresis":
        need = [lo, hi]                     # both seeds always, whichever direction
    elif direction == "below":
        need = [lo]
    elif direction == "above":
        need = [hi]
    else:
        need = [lo, hi]                     # between / outside are two-sided
    missing = [n for n in need if _opt(n) is None]
    if missing:
        raise ValueError(
            f"histogram threshold: method={method}, direction={direction} needs "
            + " and ".join(f"`{n}`" for n in missing)
            + f" — set {'it' if len(missing) == 1 else 'them'} in the node's Parameters "
            "(0 = unset). low/high/strict/permissive are absolute RAW INTEGER COUNTS, so "
            "they have no image-independent default; the percentile (0–100) and relative "
            "(× the plane median) methods self-scale and work out of the box.")
    if method == "hysteresis":
        # v1 (`histothresh.thresholds.threshold_hysteresis`) requires strict ≤ permissive
        # for `below` and strict ≥ permissive for `above` — the CORE cut is the more
        # extreme one in the direction being selected. It raises a bare comparison error;
        # say what the two seeds mean instead, because the ordering reads as inverted.
        s_v, p_v = float(_opt(hi)), float(_opt(lo))            # hi=strict, lo=permissive
        if (s_v > p_v) if direction == "below" else (s_v < p_v):
            raise ValueError(
                f"histogram threshold: hysteresis {direction} needs strict "
                f"{'≤' if direction == 'below' else '≥'} permissive, but strict={s_v:g} "
                f"and permissive={p_v:g}. `strict` is the CORE cut (pixels definitely "
                "inside the mask) and `permissive` the FRINGE, kept only where it touches "
                f"a core region — so selecting {direction} makes the core the "
                f"{'darker' if direction == 'below' else 'brighter'} of the two. Swap them.")

    # ── the intensity REGIME: what an absolute threshold means (V2.28) ─────────
    #
    # Reported 2026-08-06: "the hysteresis only allows for numbers above 1, but the data has been
    # transformed to below 1". Three things blocked a sub-1 cut, and all three are lifted here —
    # the sockets are FLOAT now, this resolves what full scale means, and the plane is mapped to
    # the kernel's integer domain by ONE constant instead of being rounded (which sent every
    # value of a [0,1] image to 0 or 1) or refused outright.
    #
    # Two regimes, and the trigger for the new one is EXACTLY the condition that used to raise,
    # so every input that works today still takes the identical path:
    #
    # * **counts** — the file declares a bit depth, or the pixels are already integers. Scale
    #   1.0: the plane and the thresholds reach the kernel untouched, as they always have.
    # * **unit** — no declared depth AND fractional values inside [0,1], i.e. the output of a
    #   percentile Normalize or CLAHE (`metadata.value_rescaled` drops `bit_depth` precisely to
    #   say "no integer scale"). Thresholds are then in the DATA'S OWN UNITS: 0.35 means 0.35.
    #
    # The mapping is fixed-point rather than a float path through the kernel, because the kernel
    # is vendored (INV-11) and its `validate_bit_depth` requires an integer dtype while
    # `compute_histogram` bins with `bincount`. Scaling lets `run()` be used VERBATIM — same
    # morphology, same area filters, same CCL — and costs one part in 65535 of full scale, which
    # is ~1.5e-5 and nothing next to the noise on any cut worth making.
    #
    # `percentile` and `relative` are invariant to the factor (a percentile of scaled values is
    # the scaled percentile; a multiple of the median likewise), so only the four ABSOLUTE params
    # are converted — and `full_scale` is gated to the two methods that read one.
    fs_pin = float(ctx.params.get("full_scale", 0.0) or 0.0)
    unit_scale = 1.0
    full_scale = float((1 << bit_depth) - 1)
    if fs_pin > 0.0:
        # An explicit declaration wins over the metadata: the user is saying what "1.0" means
        # in this image, which is the only answer available for float data outside [0,1].
        full_scale, unit_scale = fs_pin, 65535.0 / fs_pin
    elif not bd_authoritative:
        first = prov.get_region(0, 0, 0, 0, 0, 0, ax.y, 0, ax.x)
        if (not np.array_equal(first, np.rint(first))
                and float(np.max(first)) <= 1.0):
            # Decided ONCE, from the first plane, so the whole series shares one scale — a
            # per-plane decision would make the same typed number mean different brightnesses
            # on different planes, which is the defect the scope work exists to prevent.
            full_scale, unit_scale = 1.0, 65535.0
    scaled = unit_scale != 1.0
    if scaled:
        bit_depth, bd_authoritative = 16, True    # [0,1] → the full 16-bit fixed-point range
        over = {n: float(_opt(n)) for n in ("low", "high", "strict", "permissive")
                if _opt(n) is not None and float(_opt(n)) > full_scale}
        if over:
            raise ValueError(
                "histogram threshold: "
                + ", ".join(f"{n}={v:g}" for n, v in sorted(over.items()))
                + f" exceeds this image's full scale ({full_scale:g}). The pixels here are "
                f"fractional with no declared bit depth — the output of a Normalize or CLAHE — "
                f"so an absolute cut is in the DATA'S OWN units: 0.35, not 350. Type a value in "
                f"0..{full_scale:g}, or set `full_scale` if this image's full scale really is "
                f"something else (a ratio image, a deconvolved float).")

    def _abs_opt(name: str):
        """An absolute threshold in the KERNEL's integer domain — the user's number × the
        regime's scale. Identity in the counts regime, so nothing existing moves."""
        v = _opt(name)
        return None if v is None else float(v) * unit_scale

    cfg = make_config(
        method=method,
        direction=direction,
        bit_depth=bit_depth,
        low=_abs_opt("low"), high=_abs_opt("high"),
        strict=_abs_opt("strict"), permissive=_abs_opt("permissive"),
        percentile_low=_opt("percentile_low"),
        percentile_high=_opt("percentile_high"),
        fraction_low=_opt("fraction_low"), fraction_high=_opt("fraction_high"),
        min_area=min_area_px, max_area=max_area_px,
        opening_radius=_rad("opening_radius"),
        closing_radius=_rad("closing_radius"),
        min_hole_size=_area("min_hole_size"))
    seg = HistogramThresholdSegmenter(cfg)
    # v1 parity: no pixel size ⇒ no µm² measurement (the kernel then omits area_um2 and
    # the Label column is NaN) rather than a fabricated area.
    vox = (pxv, pxv) if pxv > 0.0 else None
    layer = ctx.layer("name")

    def _alloc(dtype, what: str) -> np.ndarray:
        """A full-series Voxel raster at the kernel's OWN dtype (uint8 mask / int32 CCL
        raster — what ``SegmentationResult`` already carries). Upcasting both to int64
        spent 12 B/voxel on zero information: a (16,1,10,1,2048,2048) series is 5.0 GiB
        **per raster**, so a big multi-position stack died in ``np.zeros`` before the
        first plane was read (fix 2026-07-28). Numpy's own MemoryError names the shape
        but not a way out, so re-raise with the levers."""
        try:
            return np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=dtype)
        except MemoryError as ex:                             # incl. _ArrayMemoryError
            gib = (ax.m * ax.t * ax.z * ax.c * ax.y * ax.x
                   * np.dtype(dtype).itemsize) / float(1 << 30)
            raise MemoryError(
                f"histogram threshold: the {what} raster needs {gib:.2f} GiB "
                f"({ax.m}×{ax.t}×{ax.z}×{ax.c}×{ax.y}×{ax.x} voxels of "
                f"{np.dtype(dtype).name}) and this machine cannot allocate it. This node "
                "segments the WHOLE series eagerly, so narrow the input first: util.crop "
                "(a region and/or a z range), channel.select or a chK tap (one channel), "
                "or a per-position graph. A previously viewed result is also still held "
                "in the cache — pulling a smaller graph releases it.") from ex

    mask6 = _alloc(np.uint8, "mask")                          # binary: 0/1
    labels6 = _alloc(np.int32, "label")                       # global-unique CCL ids
    tables = []
    offset = 0
    last: Any = None                                          # (coords, plane) for diagnosis
    # ── the statistics population (V2.27) ──────────────────────────────────────
    #
    # `plane` is the historical behaviour and stays the default. The two structure scopes run
    # the SAME kernel once per parent region instead of once per plane, so the level is derived
    # from that region's own pixels — which is what `analysis.threshold_per_label` was a whole
    # node for, and is now this node's `scope`.
    scope = modes.get("scope", "plane")
    if scope not in ("plane",) + tuple(STRUCTURE_SCOPES):
        raise ValueError(f"histogram threshold: unknown scope {scope!r}.")
    if scope != "plane" and method in ("single", "hysteresis"):
        # A REFUSAL, not an `available_in` gate. Gating hides the control while the compute
        # still resolves its value (a hidden Mode keeps it), so the footprint chip would go on
        # claiming a per-region read for an absolute-count per-plane run. And the user has
        # asked for something incoherent, which is worth saying out loud.
        raise ValueError(
            f"histogram threshold: method={method} has no statistics population to scope — its "
            f"cut is an ABSOLUTE raw count "
            f"({'low/high' if method == 'single' else 'strict/permissive'}), the same number in "
            f"every region, so deriving it 'per {scope.replace('per_', '')}' would compute the "
            f"identical mask at N times the cost. Use method=percentile (a rank within each "
            f"region's own histogram) or method=relative (a multiple of each region's own "
            f"median) — those are the two that HAVE a population — or set Scope back to plane.")
    pops6 = pop_layer = None
    min_pixels = max(1, int(ctx.params.get("min_pixels", 50) or 50))
    parent_stats: Dict[int, tuple] = {}
    n_pops = 0
    if scope != "plane":
        pops6, pop_layer = _resolve_populations(ctx, ds, ax, scope)
        # Checked BEFORE the series is segmented, not at write time: a per-object run writes
        # sub-objects under `name` AND per-parent columns onto the population's own table, and
        # if those are one layer the two row sets land in one instance — a 3-row `id` column
        # beside a 2-row `area`, ragged and silently misaligned. It is the DEFAULT collision
        # (Segmentation emits `labels`, and so does this node), so it has to be said clearly and
        # said early rather than after a long run.
        for sock, other in (("name", layer), ("mask_name", ctx.layer("mask_name"))):
            if other == pop_layer:
                raise ValueError(
                    f"histogram threshold: scope={scope} thresholds INSIDE the regions of "
                    f"{pop_layer!r} and writes its own sub-objects, but the `{sock}` socket "
                    f"also says {other!r} — the parents and their children would share one "
                    f"table, with mismatched row counts under the same column names. Give the "
                    f"output its own name (`{sock}` = 'puncta', say): the parents keep "
                    f"{pop_layer!r} and gain `level`/`n_above`/`frac_above`/`n_sub`, and the "
                    f"children arrive as a separate Label instance carrying `parent_id`.")
    for m, t, z, c in _each_plane_p(ctx, ax, "segmenting"):
        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        # Into the kernel's integer domain. In the COUNTS regime this is the rounding it always
        # was: a raw-count image stored as float (ND2, deconvolved) is integral and rounds
        # losslessly. In the UNIT regime `unit_scale` maps [0, full_scale] onto the 16-bit range
        # first, so a [0,1] image keeps 65536 distinguishable levels instead of collapsing to
        # {0,1} — which is what used to make this input unusable and is why it was refused.
        # A value outside the declared full scale is clipped, exactly as an over-range count is.
        plane_int = np.rint(np.clip(plane * unit_scale, 0, 65535)).astype(np.uint16)
        if not scaled and not np.array_equal(plane, np.rint(plane)) \
                and float(np.max(plane)) <= 1.0:
            # Fractional, inside [0,1], and yet NOT in the unit regime — which now means only
            # one thing: a bit depth IS declared (or pinned), so the file claims integer counts
            # while carrying normalized values. Rounding would send every pixel to 0 or 1, so
            # this stays a refusal — but it names the two ways out rather than just the cause.
            raise ValueError(
                f"histogram threshold: this plane is fractional with max "
                f"{float(np.max(plane)):.3g} ≤ 1.0 — normalized data — but the image still "
                f"declares a {bit_depth}-bit integer scale, so an absolute threshold would be "
                f"read as raw counts and every pixel would round to 0 or 1. Either drop the "
                f"declared depth (set `bit_depth` to 0 so the data's own scale is used and a "
                f"cut of 0.35 means 0.35), or set `full_scale` to 1.0 to say so explicitly.")
        last = ((m, t, z, c), plane_int)
        if scope == "plane":
            with warnings.catch_warnings():
                if bd_authoritative:
                    # The kernel warns "max N is <5% of B-bit range … may be lower-bit than
                    # declared" — a guess-checking heuristic. With the depth read from the
                    # file that guess is settled, and a genuinely dim frame would fire it on
                    # every plane. The OVER-range warning is NOT filtered: values above the
                    # declared max mean the data really was rescaled, which does invalidate
                    # raw-count thresholds. (v1 blanket-suppressed every warning here.)
                    warnings.filterwarnings("ignore", message=r".*is <5% of.*",
                                            category=UserWarning)
                res = seg.run(plane_int, voxel_size=vox)
            res_mask, res_labels, regions = res.mask, res.labels, res.regions
        else:
            # `per_roi` arrives as a MASK, whose non-zero voxels all share one id — so its
            # arenas have to be minted per plane, or two ROIs drawn far apart would be pooled
            # into a single population and cut at one shared level (which is precisely what
            # `per_roi` exists not to do). A Label instance already carries its own ids.
            pops = (roi_populations(pops6[m, t, z, c], 8) if scope == "per_roi"
                    else pops6[m, t, z, c])
            res_mask, res_labels, regions, pstats = _segment_populations(
                seg, plane_int, pops, vox,
                min_pixels=min_pixels, quiet=bd_authoritative,
                # the UNSCALED plane, so a per-parent region's reported intensity is in the
                # image's own units and its `level` is the number the user typed
                report=(plane if scaled else None), unit_scale=unit_scale)
            n_pops += len(pstats)
            for pid, st in pstats.items():
                parent_stats[int(pid)] = st
        if on_raw:
            regions = _measure_regions(
                res_labels, raw_prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), vox)
        elif scaled:
            # Re-measure on the UNSCALED plane so the table reports the image's own numbers.
            # Without this a pixel the viewer calls 0.42 arrives as `mean_intensity` 27524, and
            # a Filter Labels cut on that column would have to be typed in the node's private
            # fixed-point units — numbers that disagree with the image are how a wrong threshold
            # gets typed with confidence. Same mechanism the `raw` socket above already uses.
            regions = _measure_regions(res_labels, plane, vox)
        mask6[m, t, z, c] = res_mask                          # bool → uint8 0/1
        lab = np.asarray(res_labels).astype(np.int32, copy=True)   # kernel already int32
        lab[lab > 0] += offset
        labels6[m, t, z, c] = lab
        k = int(np.asarray(res_labels).max())                 # labels are 1..k
        if regions:
            n = len(regions)
            tables.append(StructureTable(Domain.LABEL, {
                "id": np.array([r["label_id"] + offset for r in regions], dtype=np.int64),
                "m": np.full(n, m, dtype=np.int64),
                "t": np.full(n, t, dtype=np.int64),
                "c": np.full(n, c, dtype=np.int64),
                "z": np.full(n, z, dtype=float),
                "y": np.array([r["centroid_y"] for r in regions], dtype=float),
                "x": np.array([r["centroid_x"] for r in regions], dtype=float),
                "area": np.array([r["area_px"] for r in regions], dtype=np.int64),
                # the CALIBRATED area next to the raw pixel count: min_area/max_area are
                # authored in µm², so the domain data the user filters on must be readable
                # in µm² too (the kernel measures it; the node used to drop it). This is
                # what makes "pick a threshold from the spreadsheet" a real workflow.
                "area_um2": np.array([r.get("area_um2", float("nan"))
                                      for r in regions], dtype=float),
                "mean_intensity": np.array([r["mean_intensity"] for r in regions],
                                           dtype=float),
                # Under a structure scope every region belongs to a parent and was cut at that
                # parent's own level, so both travel WITH the region — `parent_id` is the join
                # that makes "puncta per cell" a groupby, and `level` is what makes the
                # per-region derivation auditable. Absent under `plane`, where there is no
                # parent and one level served the whole frame.
                **({"parent_id": np.array([r["parent_id"] for r in regions], dtype=np.int64),
                    "level": np.array([r["level"] for r in regions], dtype=float)}
                   if scope != "plane" else {}),
            }, layer=layer, z_kind="plane_index"))
        offset += k
    # Under a structure scope, a blank result has one more possible cause than the area window
    # below: every population may have been skipped for want of a histogram. Say which, with the
    # sizes actually measured, rather than letting the area-window diagnosis take the blame.
    if scope != "plane" and n_pops and not any(
            np.isfinite(st[0]) for st in parent_stats.values()):
        sizes = sorted(st[2] for st in parent_stats.values())
        raise ValueError(
            f"histogram threshold: scope={scope} found {n_pops} population(s) but not one of "
            f"them yielded a level, so the output would be empty. They measure "
            f"{sizes[0]}–{sizes[-1]} voxels (median {int(np.median(sizes))}) and `min_pixels` "
            f"is {min_pixels}: a region below that is skipped, because a percentile or a median "
            f"over so few samples describes the sampling noise rather than the signal. Lower "
            f"`min_pixels` if those sizes are what you mean to threshold, or segment larger "
            f"parents. (A region of any size is also skipped when every voxel in it has the "
            f"SAME value — there is no rank to take in a flat histogram.)")
    if scope != "plane" and not n_pops:
        raise ValueError(
            f"histogram threshold: scope={scope} found no region at all in {pop_layer!r}, so "
            f"there is nothing to derive a level inside. Check the segmentation (or the mask) "
            f"upstream, or set Scope back to plane.")
    # An area window that discards EVERY region leaves a blank viewer with no hint of
    # which knob did it. Re-segment the last plane with the area filters off (one plane,
    # only on this dead-end path) and report the areas actually present, in the same µm²
    # the user typed — the threshold, not the area, is the culprit when that is empty too.
    if not tables and (min_area_px > 0 or max_area_px > 0) and last is not None:
        coords, plane_int = last
        raw = HistogramThresholdSegmenter(replace(cfg, min_area=0, max_area=0)).run(
            plane_int, voxel_size=vox)
        window = (f"min_area={_cleanup_um('min_area'):g} µm² ({min_area_px} px²)"
                  + (f", max_area={max_area_um2:g} µm² ({max_area_px} px²)"
                     if max_area_px > 0 else ""))
        if raw.regions:
            areas = sorted(float(r.get("area_um2", float("nan"))) for r in raw.regions)
            raise ValueError(
                f"histogram threshold: the area window ({window}) discarded every region "
                f"in the whole series. At (m,t,z,c)={coords} the threshold DOES find "
                f"{len(areas)} region(s), but they measure {areas[0]:.3g}–{areas[-1]:.3g} "
                f"µm² (median {float(np.median(areas)):.3g}) — all outside your window. "
                "min_area defaults to the v1 cleanup (100 px² worth of µm²), so pin it to "
                "0 in the inspector to see every region's `area_um2` on the Label domain "
                "in the Spreadsheet, then pick the cut from those numbers.")
        raise ValueError(
            f"histogram threshold: nothing was segmented anywhere, and the area window "
            f"({window}) is not the cause — with the area filters off, "
            f"(m,t,z,c)={coords} still yields no region, so the THRESHOLD is what "
            "rejected everything. Check method/direction and the threshold value against "
            "the image's raw counts (percentile self-scales; low/high/strict/permissive "
            "are absolute counts).")
    mask_layer = ctx.layer("mask_name")
    if mask_layer == layer:
        raise ValueError(
            f"histogram threshold: the mask and label layers are both named "
            f"{mask_layer!r} — the second write would overwrite the first. Give them "
            "different names (defaults: `mask` / `labels`).")
    out = (ds.with_layer(Domain.VOXEL, mask_layer, mask6)
             .with_layer(Domain.VOXEL, layer, labels6))
    if tables:
        cols = {kk: np.concatenate([tb.columns[kk] for tb in tables])
                for kk in tables[0].columns}
        out = out.with_structure(StructureTable(Domain.LABEL, cols, layer=layer,
                                                z_kind="plane_index"))
    if parent_stats:
        # Per-PARENT columns, written back onto the population's own Label table (V2.27): the
        # level each parent used, how much of it was selected, and how many sub-objects came out
        # of it. `frac_above` answers "how much of each cell is bright" without the sub-objects.
        #
        # **Aligned to the table's EXISTING `id` column, and `id` is NOT re-emitted.** A Label
        # instance is stored as one array per column under one layer name, so a column written
        # with a different row count leaves the instance RAGGED — and nothing downstream
        # revalidates that. The parents visited here are the ids present in the RASTER, which is
        # not necessarily the table's row set (a region the segmenter's area filter dropped from
        # the raster can still have a row, and a table measured on a wider chain can list more).
        # Emitting `id` from the visited set therefore produced a 208-row `id` beside a 209-row
        # `m`, and the viewer's palette builder — which sizes its row map from `id` but derives
        # its frame selection from `m`/`t` — indexed past the end and crashed the run
        # (reported 2026-08-06: "IndexError: index 208 is out of bounds for axis 0 with size
        # 208"). `analysis.measure` warns about exactly this: re-emitting `id` in a different
        # order silently misaligns every sibling column.
        pcols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.LABEL)
                 if a.layer == pop_layer}
        base = pcols.get("id")
        emit_id = base is None or len(base) == 0
        order = (np.asarray(sorted(parent_stats), dtype=np.int64) if emit_id
                 else np.asarray(base, dtype=np.int64))
        #: a parent the table lists but the raster does not hold was never VISITED, so every
        #: number here is unknown rather than zero — hence NaN, and hence float even for the two
        #: counts. A parent that WAS visited and skipped keeps a real 0 for its counts and NaN
        #: for its level, so "no sub-objects found" stays distinguishable from "never looked".
        miss = (float("nan"),) * 4
        st = [parent_stats.get(int(p), miss) for p in order.tolist()]
        level = np.array([s[0] for s in st], dtype=float)
        above = np.array([s[1] for s in st], dtype=float)
        tot = np.array([s[2] for s in st], dtype=float)
        cols_out = {
            "level": level,
            "n_above": above,
            # NaN, not 0.0, for a region that was SKIPPED: "nothing was above its level" and
            # "it never got a level" are different facts, and 0.0 states the first while meaning
            # the second. The same NaN-is-missing rule `bridges._group_reduce` and
            # `analysis.measure` follow — 0 is a legal fraction, so it must not double as absent.
            "frac_above": np.where(np.isfinite(level) & (tot > 0),
                                   above / np.maximum(tot, 1.0), np.nan),
            "n_sub": np.array([s[3] for s in st], dtype=float),
        }
        if emit_id:
            cols_out["id"] = order
        out = out.with_structure(StructureTable(Domain.LABEL, cols_out, layer=pop_layer,
                                                z_kind="plane_index"))
    return out

def _columns_histogram_threshold(params, modes, incoming):
    """The sub-object Label table this node writes, plus the per-PARENT columns it writes
    back onto the population it thresholded within (V2.28).

    ``parent_id`` is what makes a sub-object joinable to the parent it was found in, and
    ``n_sub`` / ``frac_above`` are the parent-side summary of that same pass — two different
    tables, so both are declared or half the result would be unconditionable.

    The parent half is declared only when the ``regions`` socket names the layer explicitly:
    the compute resolves it through ``_resolve_populations`` against the wire, which an
    edit-time pass cannot do, and guessing would offer columns on a layer that may not be the
    one written to."""
    try:
        cols = ("id", "m", "t", "c", "area", "z", "y", "x", "mean_intensity", "level",
                "parent_id")
        out = on_layer(Domain.LABEL, str((params or {}).get("name") or "labels"), cols)
        parent = str((params or {}).get("regions") or "").strip()
        if parent:
            out += on_layer(Domain.LABEL, parent, ("n_sub", "frac_above"))
        return out
    except Exception:                        # pragma: no cover - defensive
        return ()

register_node(
    batch_aware(_compute_histogram_threshold), op_key="analysis.histogram_threshold",
    adds_columns=_columns_histogram_threshold,
    label="Histogram Threshold", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[
        InDataset(),
        # measurement-only pixel override: the mask/labels still come from `data`
        _InRaw(),
        # ONE socket for both structure scopes, gated to them (V2.27). Ships EMPTY so the layer
        # is inferred from the wire when there is only one candidate (§4g).
        InString("regions", "Regions", field=False, default="",
                 layer_in=Domain.VOXEL,
                 available_in={"scope": frozenset(STRUCTURE_SCOPES)},
                 description=
                 "Which objects define the populations, when Scope is per label or per ROI. "
                 "Under `per_label` this must be a real Label instance — a raster plus the "
                 "table that proves its ids divide the foreground (Segmentation, Connected "
                 "Components, or another Histogram Threshold) — and it is refused if that "
                 "segmentation was VOLUMETRIC, because this node's engine is 2D and would cut "
                 "one object at a different level on every plane it spans. Under `per_roi` any "
                 "binary mask works, including a drawn ROI, and its connected components become "
                 "the arenas. Leave it EMPTY to use the only candidate on the wire. Inert under "
                 "Scope = plane."),
        InInt("min_pixels", "Min pixels", unit="", field=False, default=50,
              available_in={"scope": frozenset(STRUCTURE_SCOPES)},
              description=
              "The smallest region that gets a level of its own, as a voxel COUNT. Below it the "
              "region is SKIPPED — no sub-objects, and `level` reported as NaN on its row — "
              "because a percentile or a median over a few dozen samples describes the sampling "
              "noise and returns a confident-looking number with half the region either side of "
              "it. Deliberately a sample count rather than a µm² area: what makes a histogram "
              "estimable is how many samples are in it, so a physical floor would demand MORE "
              "samples at fine resolution and fewer at coarse, which is backwards. Lower it and "
              "small regions start contributing noisy sub-objects; raise it and they drop out "
              "entirely, which the NaN is there to make visible. Only read under a per-object "
              "Scope."),
        # The significant sensor depth, read from the file (ND2 bitsPerComponentSignificant
        # → usually 12; TIFF bits-per-sample). 0 = use the metadata; a non-zero value
        # overrides it, for a file whose header lies or a rescaled TIFF. v1 exposed this as
        # a plain int param defaulting to 12 — v2 derives it instead, and only asks when
        # the metadata is silent (then 16).
        InInt("bit_depth", "Bit depth", unit="", field=False, default=0,
              derive="bit_depth or 16",
              description=
              "Significant sensor depth, which fixes the histogram size (2^depth bins) and the "
              "largest intensity the threshold engine will accept. Auto reads it from the file "
              "(usually 12 for ND2) and falls back to 16 when the metadata is silent; override "
              "only for a file whose header lies or one that has been rescaled. Getting it "
              "WRONG matters: too low and bright pixels exceed the ceiling (they are clamped "
              "with a warning, and any absolute Low/High you set means a different brightness "
              "than you intended); too high and the histogram is mostly empty bins, which "
              "makes percentile and Otsu-style levels less stable. Snapped up to the nearest "
              "supported depth of 8, 10, 12, 14 or 16."),
        # Output layer names. Both Voxel rasters were hard-named by the compute with no
        # socket to set them, so a graph could not carry two thresholded results (they
        # collided on `mask`/`labels`). Defaults are exactly the old fallbacks.
        InString("mask_name", "Mask layer", field=False, default="mask",
                 layer_out=(Domain.VOXEL,),
                 description=
                 "Name of the binary mask layer this node writes — the foreground AFTER the "
                 "opening/closing/hole-filling and area filters below, not the raw threshold "
                 "result. Rename it when a graph carries two thresholded results, or the "
                 "second node silently overwrites the first."),
        InString("name", "Label layer", field=False, default="labels",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "Name of the labelled output — one name into two domains: a Voxel raster where "
                 "each surviving object carries its own id (8-connected) and a Label table of "
                 "per-region measurements. Distinct from the Mask layer above, which is the "
                 "same foreground without ids. Downstream nodes select this by name."),
        # Threshold params are method-specific: each is gated to the method that reads it
        # and to the directions that need that side, so the inspector shows the 1–2 live
        # fields instead of all 8 (0 = unset; defaults mirror `_THRESH_DEFAULTS`).
        # What "1.0" means in this image, for the two methods that take an ABSOLUTE number
        # (V2.28). Auto: the declared bit depth's maximum when the file states one, else 1.0 for
        # fractional data — so a normalized image is thresholded in its own units and an ND2 is
        # thresholded in counts, both without being told. Pinning it is the only answer available
        # for float data outside [0,1], which carries no scale anywhere in the Dataset.
        InFloat("full_scale", "Full scale", unit="", field=False, default=0.0,
                derive="(2**bit_depth - 1) if bit_depth else 1.0",
                available_in={"method": frozenset({"single", "hysteresis"})},
                description=
                "The pixel value that counts as FULL SCALE, which is what makes an absolute "
                "threshold mean something. Auto reads it from the file: 4095 on a 12-bit ND2, so "
                "Low/High/Strict/Permissive are raw counts exactly as before — and 1.0 once a "
                "Normalize or CLAHE has dropped the declared depth, so on [0,1] data a cut of "
                "0.35 means 0.35. Set it only when neither applies: a ratio image, or a "
                "deconvolved float whose range is 0..3.7 — then type 3.7 here and the "
                "thresholds are read against that. It does NOT rescale the image or the reported "
                "intensities; it only says what your typed numbers are measured against. Ignored "
                "by the percentile and relative methods, which derive their own level and are "
                "therefore indifferent to the scale."),
        InFloat("low", "Low", unit="", field=False, default=0,
              pick_kind="level",
              available_in={"method": frozenset({"single"}),
                            "direction": frozenset({"below", "between", "outside"})},
              description=
              "The LOWER absolute intensity cut, in the image's OWN units — raw counts (0 … 2^bit_depth−1) when the file declares a bit depth, and the data's own scale when it does "
              "not (after a Normalize, 0.35 means 0.35; fractional values are exactly what this socket is for). With "
              "direction `below` the foreground is everything at or under this, so RAISING it "
              "grows the mask; with `between` it is the bottom of the kept band and with "
              "`outside` the bottom of the REJECTED band. Absolute, so it is only comparable "
              "across frames whose exposure matches — use the `percentile` or `relative` method "
              "when brightness drifts. Only shown for the `single` method in the directions "
              "that use a lower bound."),
        InFloat("high", "High", unit="", field=False, default=0,
              pick_kind="level",
              available_in={"method": frozenset({"single"}),
                            "direction": frozenset({"above", "between", "outside"})},
              description=
              "The UPPER absolute intensity cut, in the image's own units (raw counts, or the data's own scale on normalized input — see Low). With direction `above` the "
              "foreground is everything at or over this, so LOWERING it grows the mask; with "
              "`between` it is the top of the kept band and with `outside` the top of the "
              "rejected band. Only shown for the `single` method in the directions that use an "
              "upper bound."),
        InFloat("strict", "Strict", unit="", field=False, default=0,
              available_in={"method": frozenset({"hysteresis"})},
              description=
              "The SEED level for hysteresis, in the image's own units — raw counts, or the data's own scale on normalized input, where a value BELOW 1 is normal and expected. Only regions containing at least "
              "one pixel this extreme are kept at all. This is the confidence knob — make it "
              "demanding and marginal objects disappear entirely, however large they are. "
              "Hysteresis then grows each surviving seed out to the Permissive level, which is "
              "what lets it capture a faint halo without also lighting up every faint patch "
              "that has no bright core. With direction `below` it must be ≤ Permissive; with "
              "`above`, ≥. Hysteresis method only."),
        InFloat("permissive", "Permissive", unit="", field=False, default=0,
              available_in={"method": frozenset({"hysteresis"})},
              description=
              "The GROW level for hysteresis, in the image's own units (see Strict — below 1 on normalized data): starting from each seed, "
              "neighbouring pixels are absorbed as long as they reach this weaker level. It "
              "therefore sets each object's final EXTENT — and so its measured area — while "
              "Strict decides which objects exist. Move it toward Strict for tight masks, away "
              "for generous ones; too far and separate objects bleed together through the "
              "faint background that connects them. Hysteresis method only."),
        InFloat("percentile_low", "Percentile low", unit="", field=False, default=5.0,
                pick_kind="percentile",
                available_in={"method": frozenset({"percentile"}),
                              "direction": frozenset({"below", "between", "outside"})},
                description=
                "The lower cut expressed as a PERCENTILE of the frame's own histogram (0–100) "
                "rather than an absolute intensity — so it adapts automatically to exposure "
                "drift and photobleaching, which is the whole reason to choose this method. 5 "
                "means \"the dimmest 5% of pixels\". The trade-off: it assumes the foreground "
                "FRACTION is roughly constant, so a frame that genuinely contains more or "
                "fewer objects gets a different effective threshold. Only shown for the "
                "`percentile` method in directions using a lower bound."),
        InFloat("percentile_high", "Percentile high", unit="", field=False, default=95.0,
                pick_kind="percentile",
                available_in={"method": frozenset({"percentile"}),
                              "direction": frozenset({"above", "between", "outside"})},
                description=
                "The upper cut as a percentile of the frame's own histogram (0–100). 95 means "
                "\"the brightest 5% of pixels\", so LOWERING it admits more of the frame. Same "
                "adaptive benefit and same constant-fraction assumption as Percentile low. Only "
                "shown for the `percentile` method in directions using an upper bound."),
        InFloat("fraction_low", "Fraction low", unit="", field=False, default=0.30,
                available_in={"method": frozenset({"relative"}),
                              "direction": frozenset({"below", "between", "outside"})},
                description=
                "The lower cut as a MULTIPLE of the frame's median intensity — 0.30 means "
                "\"30% of median\". Like the percentile method it tracks overall brightness "
                "changes, but it keys on the median (a background estimate in a mostly-empty "
                "frame) instead of on a fixed pixel fraction, so it does NOT assume a constant "
                "foreground fraction and copes better with a varying number of objects. It does "
                "assume the median stays representative of background, which fails once "
                "objects cover much of the field. Only shown for the `relative` method in "
                "directions using a lower bound."),
        InFloat("fraction_high", "Fraction high", unit="", field=False, default=1.50,
                available_in={"method": frozenset({"relative"}),
                              "direction": frozenset({"above", "between", "outside"})},
                description=
                "The upper cut as a multiple of the frame's median intensity — 1.50 means "
                "\"anything at least 1.5× the median\", the natural way to say \"clearly "
                "brighter than background\". LOWERING it admits dimmer objects. Tracks overall "
                "brightness changes automatically, and assumes the median remains a background "
                "estimate. Only shown for the `relative` method in directions using an upper "
                "bound."),
        # Spatial cleanup. v1 shipped this ON (min_area 100 px², opening 1 px, closing
        # 2 px, min_hole 50 px²) and v2 shipped it all at 0 = off, which is why the same
        # image segmented to raw speckle here. The derives restore v1's numbers while
        # keeping them µm-authored, so they follow the objective instead of the sensor;
        # `or 0` means an UNCALIBRATED image resolves them to 0 = off (v1's None path).
        InFloat("min_area", "Min area", unit="um2", field=True, default=0.0,
                pick_kind="area", pick_peer="max_area",
                derive="100*(pixel_size_um or 0)**2",         # v1: 100 px²
                description=
                "Drop objects smaller than this, in µm² — the main speckle filter, and the "
                "first thing to raise when a result comes back as scattered dots instead of "
                "objects. 0 disables it, which is what an UNCALIBRATED image resolves to, so "
                "check this if a run is unexpectedly noisy. Auto reproduces the original "
                "pipeline's 100 px² using the file's own pixel size, so the physical size "
                "follows the objective rather than the sensor. Applied after the morphology "
                "below, so opening may already have removed the smallest specks."),
        # the v1 upper-area filter, previously read by the compute with no socket to set
        # it (so always off): drops blobs LARGER than this (0 = no upper limit, v1 default)
        InFloat("max_area", "Max area", unit="um2", field=True, default=0.0,
                pick_kind="area", pick_peer="min_area",
                description=
                "Drop objects LARGER than this, in µm². 0 (the default) means no upper limit. "
                "Its real use is discarding the merged blob you get when many objects touch, "
                "or a saturated artefact spanning much of the frame — both of which would "
                "otherwise dominate any per-object statistic. Because it removes whole objects "
                "it changes object COUNT as well as the area distribution."),
        InFloat("opening_radius", "Opening radius", unit="um", field=True, default=0.0,
                derive="1*(pixel_size_um or 0)",              # v1: 1 px
                description=
                "Morphological opening radius, in microns: erode then dilate, which deletes "
                "protrusions and specks thinner than this while leaving larger objects at "
                "roughly their original size. Runs FIRST in the cleanup order, so it is the "
                "cheap way to remove single-pixel noise before anything else looks at the "
                "mask. LARGER also rounds off genuine fine structure and can sever thin "
                "connections between parts of one object. 0 skips it; auto reproduces the "
                "original pipeline's 1 px at the file's own pixel size."),
        InFloat("closing_radius", "Closing radius", unit="um", field=True, default=0.0,
                derive="2*(pixel_size_um or 0)",              # v1: 2 px
                description=
                "Morphological closing radius, in microns: dilate then erode, which bridges "
                "gaps and notches narrower than this. Runs AFTER opening, so it repairs the "
                "object outlines that thresholding left ragged. LARGER also fuses genuinely "
                "separate objects that pass within this distance of each other — the usual "
                "cause of two cells being reported as one — so keep it below the smallest gap "
                "you need preserved. 0 skips it; auto reproduces the original 2 px."),
        InFloat("min_hole_size", "Min hole", unit="um2", field=True, default=0.0,
                pick_kind="area",
                derive="50*(pixel_size_um or 0)**2",          # v1: 50 px²
                description=
                "Fill interior holes smaller than this, in µm² — for objects that came out "
                "hollow because their centres are dimmer than their edges. It only ever "
                "INCREASES measured area, and a value large enough to swallow a genuine "
                "lumen (a vacuole, an unstained nucleus) will silently fill that too. 0 skips "
                "it; auto reproduces the original 50 px²."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("method", ["single", "hysteresis", "percentile", "relative"],
                default="percentile",
                description=
                "How the cut values are ARRIVED AT — and therefore which threshold sockets "
                "appear, since each method reads its own pair and the others are hidden. Two "
                "are absolute (you type raw camera counts, so they mean the same thing on "
                "every frame but nothing on a different camera) and two are image-relative "
                "(they re-derive per plane, so they follow a dimming series instead of "
                "drifting out of range).",
                choice_docs={
                    "single":
                        "One (or two) ABSOLUTE cuts in the image's own units, applied per pixel "
                        "— raw counts on a file that declares a bit depth, and fractional values "
                        "on normalized data, where a cut of 0.35 is normal. The most predictable "
                        "and the only one with no hidden statistics — and the one that needs "
                        "numbers you can only get by looking at your own histogram, which is why "
                        "it has no default and refuses to guess.",
                    "hysteresis":
                        "Two absolute seeds: pixels past `permissive` are kept only where they "
                        "are CONNECTED to a core of pixels past `strict`. Recovers a dim halo "
                        "attached to a bright core while rejecting the same dim intensity "
                        "elsewhere, which is what a single cut cannot do. Supports `below` and "
                        "`above` only — a two-sided region has no core to grow from.",
                    "percentile":
                        "Cuts placed at PERCENTILES of the plane's own histogram (5/95 by "
                        "default, the v1 segmenter's numbers). Self-scaling: the same settings "
                        "work across exposures and on a bleaching series, at the cost of "
                        "assuming a roughly constant fraction of the frame is foreground — an "
                        "empty plane still yields \"5% dark, 5% bright\".",
                    "relative":
                        "Cuts as MULTIPLES of the plane's median intensity (0.30× / 1.50× by "
                        "default) — i.e. relative to the background level rather than to a "
                        "rank. Robust to overall brightness like `percentile`, but it tracks "
                        "the background rather than the object fraction, which is the better "
                        "assumption when object coverage varies a lot between planes.",
                }),
           Mode("direction", ["below", "above", "between", "outside"], default="above",
                description=
                "Which side of the cut(s) becomes foreground. It decides how many threshold "
                "values are needed — one for the one-sided directions, both for the two-sided "
                "ones — and the unused socket is hidden. Comparisons are inclusive at the "
                "boundary, so a value exactly AT the cut is selected.",
                choice_docs={
                    "above":
                        "Foreground is everything at or ABOVE the high cut — the normal "
                        "fluorescence case (bright objects on dark background), and the "
                        "default. Only the high-side value is read.",
                    "below":
                        "Foreground is everything at or BELOW the low cut — for dark objects "
                        "on a bright field, i.e. brightfield or phase-contrast data. Only the "
                        "low-side value is read.",
                    "between":
                        "Foreground is the BAND between the two cuts, inclusive. Selects a "
                        "mid-grey population while rejecting both background and saturated "
                        "structure — useful for isolating one stain level, or for excluding "
                        "bright debris from an otherwise good mask. Not available for "
                        "hysteresis.",
                    "outside":
                        "Foreground is everything OUTSIDE the two cuts — the complement of "
                        "`between`. Selects both extremes at once, which is mostly a "
                        "quality-control tool: it is how you mask saturated and dead pixels "
                        "together. Not available for hysteresis.",
                }),
           # The population (V2.27) — the node's whole behaviour under one control, edited from
           # the card's footprint band. Deliberately NOT `available_in`-gated on `method`: the
           # two absolute methods have no population, and a HIDDEN mode keeps its value, so the
           # footprint chip would go on claiming a per-region read for a per-plane run. The
           # compute refuses that pairing in words instead, which is also the more useful answer.
           ScopeMode(_HT_SCOPES, default="plane",
                     description=
                     "Which pixels are pooled into the histogram each cut is derived from. "
                     "`plane` is the classic behaviour and the default — one cut for the whole "
                     "frame. The two per-object choices run this same engine, with all its "
                     "morphology and area filters, once per REGION instead: each cell is cut "
                     "against its own brightness, and the sub-objects that come out carry a "
                     "`parent_id` so \"how many puncta in this cell\" is a groupby rather than a "
                     "geometric re-derivation. Only the `percentile` and `relative` methods have "
                     "a population to scope — `single` and `hysteresis` are absolute raw counts, "
                     "and pairing them with a per-object Scope is refused rather than costing N "
                     "times as much for the identical mask."),
           ],
    granularity=_HT_GRAN, footprint_mode="scope", kernel_axes=frozenset({"y", "x"}),
    # per_label needs a whole Label INSTANCE and per_roi only a mask, stated per BRANCH so no
    # plane-scoped graph gets a red LABEL chip it does not need.
    reads_domains_by_mode=_HT_READS,
    description="Histogram threshold segmentation (2D per-plane) → Voxel mask + Label "
                "raster + region table; 4 methods × 4 directions, morphology cleanup "
                "(ported v1 kernel). low/high/strict/permissive are FLOAT and read in the "
                "image's OWN units — raw counts while a bit depth is declared, the data's "
                "own scale once one is not (after a Normalize, 0.35 means 0.35), or "
                "whatever `full_scale` pins for a float image outside [0,1]. 0 = unset, "
                "with no default — the percentile/relative methods self-scale instead; "
                "only the current method's params are shown. Optional `raw` input "
                "re-measures the region intensities on those pixels.")
