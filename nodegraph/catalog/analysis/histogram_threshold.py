"""Histogram Threshold (``analysis.histogram_threshold``) — Histogram threshold segmentation (2D per-plane) → Voxel mask + Label raster + region table; 4 methods × 4 directions, morphology cleanup (ported v1 kernel)."""

from __future__ import annotations

import numpy as np
import warnings

from dataclasses import replace
from typing import Any, Optional

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
from nodegraph.catalog._shared.planes import _each_plane_p
from nodegraph.catalog._shared.raw_measure import _InRaw, _intensity_provider
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Histogram Threshold Segmenter (2D → mask + labels + region table) ──────────

#: the bit depths the vendored kernel's LUT tables support (``BIT_DEPTH_MAX``).
_KERNEL_BIT_DEPTHS = (8, 10, 12, 14, 16)
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

    cfg = make_config(
        method=method,
        direction=direction,
        bit_depth=bit_depth,
        low=_opt("low"), high=_opt("high"),
        strict=_opt("strict"), permissive=_opt("permissive"),
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
    for m, t, z, c in _each_plane_p(ctx, ax, "segmenting"):
        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        # thresholds are in RAW INTEGER counts. A raw-count image stored as float
        # (ND2/deconvolved) rounds losslessly; a NORMALIZED [0,1] float would quantize
        # to {0,1} and make raw-count thresholds meaningless — surface that loudly
        # instead of silently producing garbage (review 2026-07-23).
        if not np.array_equal(plane, np.rint(plane)) and float(np.max(plane)) <= 1.0:
            raise ValueError(
                "histogram threshold needs raw integer counts, but the input looks "
                f"normalized (fractional, max={float(np.max(plane)):.3g} ≤ 1.0). "
                "Run it before a normalize/deconvolve step, or on the raw image; its "
                "thresholds (low/high/strict/permissive) are in raw counts.")
        plane_int = np.rint(np.clip(plane, 0, 65535)).astype(np.uint16)
        last = ((m, t, z, c), plane_int)
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
        regions = res.regions
        if on_raw:
            regions = _measure_regions(
                res.labels, raw_prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), vox)
        mask6[m, t, z, c] = res.mask                          # bool → uint8 0/1
        lab = res.labels.astype(np.int32, copy=True)           # kernel already int32
        lab[lab > 0] += offset
        labels6[m, t, z, c] = lab
        k = int(res.labels.max())                             # labels are 1..k
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
            }, layer=layer, z_kind="plane_index"))
        offset += k
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
    return out
register_node(
    _compute_histogram_threshold, op_key="analysis.histogram_threshold",
    label="Histogram Threshold", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[
        InDataset(),
        # measurement-only pixel override: the mask/labels still come from `data`
        _InRaw(),
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
        InInt("low", "Low", unit="", field=False, default=0,
              pick_kind="level",
              available_in={"method": frozenset({"single"}),
                            "direction": frozenset({"below", "between", "outside"})},
              description=
              "The LOWER absolute intensity cut, in raw counts (0 … 2^bit_depth−1). With "
              "direction `below` the foreground is everything at or under this, so RAISING it "
              "grows the mask; with `between` it is the bottom of the kept band and with "
              "`outside` the bottom of the REJECTED band. Absolute, so it is only comparable "
              "across frames whose exposure matches — use the `percentile` or `relative` method "
              "when brightness drifts. Only shown for the `single` method in the directions "
              "that use a lower bound."),
        InInt("high", "High", unit="", field=False, default=0,
              pick_kind="level",
              available_in={"method": frozenset({"single"}),
                            "direction": frozenset({"above", "between", "outside"})},
              description=
              "The UPPER absolute intensity cut, in raw counts. With direction `above` the "
              "foreground is everything at or over this, so LOWERING it grows the mask; with "
              "`between` it is the top of the kept band and with `outside` the top of the "
              "rejected band. Only shown for the `single` method in the directions that use an "
              "upper bound."),
        InInt("strict", "Strict", unit="", field=False, default=0,
              available_in={"method": frozenset({"hysteresis"})},
              description=
              "The SEED level for hysteresis, in raw counts: only regions containing at least "
              "one pixel this extreme are kept at all. This is the confidence knob — make it "
              "demanding and marginal objects disappear entirely, however large they are. "
              "Hysteresis then grows each surviving seed out to the Permissive level, which is "
              "what lets it capture a faint halo without also lighting up every faint patch "
              "that has no bright core. With direction `below` it must be ≤ Permissive; with "
              "`above`, ≥. Hysteresis method only."),
        InInt("permissive", "Permissive", unit="", field=False, default=0,
              available_in={"method": frozenset({"hysteresis"})},
              description=
              "The GROW level for hysteresis, in raw counts: starting from each seed, "
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
                        "One (or two) ABSOLUTE cuts in raw integer counts, applied per pixel. "
                        "The most predictable and the only one with no hidden statistics — and "
                        "the one that needs numbers you can only get by looking at your own "
                        "histogram, which is why it has no default and refuses to guess.",
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
                })],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Histogram threshold segmentation (2D per-plane) → Voxel mask + Label "
                "raster + region table; 4 methods × 4 directions, morphology cleanup "
                "(ported v1 kernel). low/high/strict/permissive are in raw integer "
                "counts (0 = unset, no default — the percentile/relative methods "
                "self-scale to the image); only the current method's params are shown. "
                "Optional `raw` input re-measures the region intensities on those pixels.")
