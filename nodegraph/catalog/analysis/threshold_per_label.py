"""Threshold Per Label (``analysis.threshold_per_label``) — Threshold INSIDE each label separately: every parent region derives its own level from its own pixel histogram (otsu/li/yen/triangle/mean/percentile/relative) → a sub-mask, re-CCL'd sub-objects carrying `parent_id`, and per-parent columns on the parent table."""

from __future__ import annotations

import numpy as np

from typing import Dict, List, Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InBool,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.spill import dense_output
from nodegraph.structure import StructureTable, _connectivity_rank

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _label_raster, _resolve_label_instance
from nodegraph.catalog._shared.raw_measure import _InRaw, _intensity_provider

# ── Threshold Per Label (each parent region derives its OWN level) ──────────────
#
# The gap this fills: every other threshold in the catalog derives its level over a
# POPULATION that is chosen geometrically — `analysis.threshold`'s scope is plane / volume /
# series / dataset, `analysis.threshold_local`'s is a µm block, and
# `analysis.histogram_threshold` is per plane by construction. None of them can be told
# "the population is this cell", which is the only population that answers "which pixels
# inside THIS cell are bright" on data where cells differ in brightness by more than the
# bright/dim contrast within any one of them. A global cut on that data reports every pixel
# of the bright cells and none of the dim ones, whatever method picks it.

#: level methods that read the region's histogram through skimage. Verified in-env against
#: skimage 0.26.0: each takes a flat 1-D vector, which is exactly a label's pixel list.
_SK_METHODS: Tuple[str, ...] = ("otsu", "li", "yen", "triangle", "mean")
#: every ``method`` choice — the five above plus the two SELF-SCALING ones
#: ``analysis.histogram_threshold`` already ships, computed here over the region's own
#: pixels rather than the plane's.
_PL_METHODS: Tuple[str, ...] = _SK_METHODS + ("percentile", "relative")


def _sk_threshold(name: str, px: np.ndarray) -> Optional[float]:
    """One skimage histogram level over ``px`` (a flat vector), or ``None`` when the method
    itself declines the region.

    Every one of the five raises or returns a non-finite level on a degenerate input — a
    single-valued vector has no inter-class variance to maximize — so the ``None`` return is
    the node's "this region has no level" signal rather than an exception the caller has to
    tell apart from a real bug. The caller has already excluded the two degeneracies that are
    *predictable* (too few pixels, zero spread); this catches whatever else a method decides
    it cannot do, and a non-finite result is refused here rather than being written into a
    comparison that would then select every voxel or none."""
    from skimage import filters
    fn = getattr(filters, f"threshold_{name}")
    try:
        lvl = float(fn(px))
    except (ValueError, RuntimeError):
        return None
    return lvl if np.isfinite(lvl) else None


def _region_level(px: np.ndarray, *, method: str, percentile: float,
                  fraction: float, min_pixels: int) -> Optional[float]:
    """The level for ONE parent region, or ``None`` when the region has no usable histogram.

    Two degeneracies are excluded before any method runs, because both produce a *number*
    rather than an error and the number is meaningless:

    * **too few pixels.** Otsu on 40 samples is noise — it maximizes inter-class variance on
      a histogram that is mostly empty bins, so it splits the sampling noise and reports a
      confident level either side of which lies half the region. ``min_pixels`` is the floor.
    * **zero spread.** A flat region (every voxel the same value) has no cut at all; whatever
      a method returned, ``>=`` would select all of it and ``<=`` likewise.

    ``percentile``/``relative`` are handled here rather than through skimage because neither
    is a histogram method: they are the self-scaling forms from
    ``analysis.histogram_threshold``, evaluated on this region's own pixels."""
    if px.size < min_pixels:
        return None
    if float(np.ptp(px)) <= 0.0:
        return None
    if method == "percentile":
        return float(np.percentile(px, percentile))
    if method == "relative":
        return float(fraction * np.median(px))
    return _sk_threshold(method, px)


def _sub_objects(coords: Tuple[np.ndarray, ...], rank: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Connected components of ONE parent's selected voxels → ``(per_voxel, areas, centroids,
    k)``.

    ``coords`` is the unravelled voxel-index coordinate arrays of the voxels this parent
    selected. Returns the component id of every entry of ``coords`` (1..k, aligned), the
    per-component areas, their centroids in voxel-index coordinates of the FULL unit
    (``(K, ndim)``), and the component count.

    **Why this runs per parent instead of once per unit.** Labelling the whole sub-mask in
    one pass is cheaper and wrong: two parents whose selected voxels touch across their
    shared border — adjacent cells, which is the normal case — merge into a single component,
    so one sub-object would be reported as belonging to two cells and the ``parent_id`` join
    would silently pick whichever parent won. Restricting the CCL to one parent's voxels
    makes that impossible by construction. It is affordable because the work is proportional
    to the parent's BOUNDING BOX, not the frame: the box is cut from the parent's own extent,
    so a 2048² plane holding 300 cells labels 300 small boxes rather than one big frame."""
    from scipy import ndimage as ndi
    ndim = len(coords)
    mins = tuple(int(c.min()) for c in coords)
    box_shape = tuple(int(c.max()) - lo + 1 for c, lo in zip(coords, mins))
    local = tuple(c - lo for c, lo in zip(coords, mins))
    box = np.zeros(box_shape, dtype=bool)
    box[local] = True
    lab_box, k = ndi.label(box, structure=ndi.generate_binary_structure(ndim, rank))
    per_voxel = lab_box[local].astype(np.int64)            # 1..k, aligned with `coords`
    areas = np.bincount(per_voxel, minlength=k + 1)[1:k + 1].astype(np.int64)
    if k:
        com = np.atleast_2d(np.asarray(
            ndi.center_of_mass(box, lab_box, np.arange(1, k + 1)), dtype=float))
        com = com + np.asarray(mins, dtype=float)          # box → full-unit coordinates
    else:
        com = np.zeros((0, ndim), dtype=float)
    return per_voxel, areas, com, int(k)


def _compute_threshold_per_label(ctx: EvalContext) -> Dataset:
    """Threshold **inside each label separately** — every parent region derives its own
    level from its own pixel histogram, and the comparison is made only against that.

    Resolved spec (grilled 2026-08-05): category analysis; op
    ``analysis.threshold_per_label``; reads ``{VOXEL, LABEL}``, adds ``{VOXEL, LABEL}``;
    ``WHOLE_VOLUME``; **no 2D/3D lever**; backends ``skimage.filters.threshold_*`` +
    ``scipy.ndimage.label`` (both lazily imported).

    **Three outputs, one pass.**

    * a binary Voxel **sub-mask** (``name``, 0/1 ``uint8``) — the selected voxels;
    * an optional re-CCL'd **sub-label instance** (``sub_name``): a Voxel raster of
      global-unique ids plus a Label table carrying ``parent_id`` and the ``level`` its
      parent used. That parent link is the point of doing this per label at all — without it
      "how many puncta in cell 7" is not a groupby but a geometric re-derivation;
    * per-parent columns on the **parent** Label table: ``level`` (NaN where the region was
      skipped, see below), ``n_above`` (selected voxel count), ``frac_above`` and — when
      sub-objects are on — ``n_sub``.

    **Dimensionality is INHERITED, not levered** (`wire-node-v2` §7b, the rule
    ``analysis.measure``'s shape walk and ``transform.label_to_points`` follow). A 2D-per-plane
    segmentation's regions ARE planes; a lever set to 3D over them would pool a cell with
    whatever sits above it in the next plane, and a lever set to 2D over a volumetric
    segmentation would cut one object at several different levels. The Label instance's own
    ``z_kind`` provenance already records which it is, so this node reads that instead of
    offering a control that can only disagree with the segmentation upstream. A single-plane
    Dataset is 2D whatever the table says (``ax.z > 1`` guard) — the same belt-and-braces
    ``analysis.measure`` applies for the same reason.

    **A skipped region is reported, not fabricated.** A parent below ``min_pixels`` or with
    zero spread has no histogram, so it yields NO sub-objects and ``level = NaN`` — never the
    frame's global level, which would put two incomparable numbers in one column and hide the
    small-sample problem behind a plausible value. If EVERY parent was skipped the node
    RAISES with the region sizes it actually measured, so the blank result names its own cause
    (the ``analysis.histogram_threshold`` precedent).

    **The channel sockets.** ``label_channel`` takes the parent geometry from one channel of
    the raster while the pixels come from another — "cells segmented on ch0, puncta in ch1",
    which no existing socket expresses: ``raw`` redirects the *chain* but reads the SAME
    channel index. ``signal_channel`` picks which channel's pixels are thresholded. The
    combination that would visit one parent id more than once (a fixed ``label_channel`` with
    every channel's pixels) is refused rather than silently writing two rows per parent under
    one id.

    ``min_pixels`` is a raw SAMPLE COUNT with no µm form on purpose (§2's scale-invariance
    clause): what makes a histogram estimable is the number of samples in it, so a µm² floor
    would be the wrong invariant — it would demand more samples at fine resolution and fewer
    at coarse, which is backwards. Nothing else here is a physical extent, so this node reads
    no calibration key at all and its memo fences on none."""
    ds = ctx.inputs[0]
    prov, on_raw = _intensity_provider(ctx, ds)
    if prov is None:
        raise ValueError(
            "threshold per label needs an image provider (on its `data` input, or via "
            "`raw`) — it thresholds PIXELS inside each region, so a Dataset carrying only "
            "structure has nothing to cut.")
    ax = ds.axes
    layer, lnote = _resolve_label_instance(
        ds, ctx.layer("labels"), node="threshold per label", socket="labels",
        remedy="this derives one level per REGION, so it needs a label raster AND its table "
               "— run analysis.segment / analysis.label / analysis.histogram_threshold "
               "upstream")
    raster6, zk = _label_raster(ds, layer, node="threshold per label")
    # 3D iff the segmentation itself was volumetric AND there is a volume to be had.
    is_3d = (zk == "subpixel") and ax.z > 1

    modes = ctx.params.get("__modes__", {})
    method = modes.get("method", "otsu")
    direction = modes.get("direction", "above")
    if method not in _PL_METHODS:
        raise ValueError(f"threshold per label: unknown method {method!r} — expected one of "
                         f"{', '.join(_PL_METHODS)}.")
    percentile = float(ctx.params.get("percentile", 95.0) or 95.0)
    fraction = float(ctx.params.get("fraction", 1.5) or 1.5)
    if method == "percentile" and not 0.0 <= percentile <= 100.0:
        raise ValueError(f"threshold per label: percentile={percentile:g} is outside 0–100 "
                         "— it is a rank within each region's own histogram.")
    min_pixels = max(1, int(ctx.params.get("min_pixels", 50) or 50))
    want_sub = bool(ctx.params.get("sub_labels", True))
    conn = int(ctx.params.get("connectivity", 0) or 0) or (26 if is_3d else 8)
    rank = _connectivity_rank(3 if is_3d else 2, conn)      # validates conn for the dim

    # ── channel routing ────────────────────────────────────────────────────────
    lc = int(ctx.params.get("label_channel", -1))
    sc = int(ctx.params.get("signal_channel", -1))
    for nm, v in (("label_channel", lc), ("signal_channel", sc)):
        if v >= ax.c:
            raise ValueError(
                f"threshold per label: {nm}={v} but this Dataset has {ax.c} channel(s) "
                f"(0..{ax.c - 1}) — set it to -1 to use the matching channel index.")
    signal_channels = list(range(ax.c)) if sc < 0 else [sc]
    if lc >= 0 and len(signal_channels) > 1:
        raise ValueError(
            f"threshold per label: label_channel={lc} pins every parent's geometry to one "
            f"channel, but signal_channel=-1 thresholds all {ax.c} of them — each parent id "
            f"would then be visited {ax.c} times and write that many rows under one id into "
            f"the parent table. Set `signal_channel` to the channel the signal is in (the "
            f"cross-channel case this pairing exists for), or set label_channel=-1 to "
            f"threshold each channel against its own labels.")

    # output names, all distinct from each other and from the parent layer
    sub_mask_layer = ctx.layer("name")
    sub_layer = ctx.layer("sub_name")
    taken = {"parent label layer": layer, "sub-mask layer": sub_mask_layer}
    if want_sub:
        taken["sub-label layer"] = sub_layer
    if len(set(taken.values())) != len(taken):
        raise ValueError(
            "threshold per label: the output layer names collide — "
            + ", ".join(f"{k} = {v!r}" for k, v in taken.items())
            + ". Each write would overwrite the previous one (defaults: the parent stays "
              "`labels`, the sub-mask is `submask`, the sub-objects are `subobjects`).")

    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in signal_channels] if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in signal_channels])
    # Both rasters are allocated at the kernel's own dtype and through `dense_output`, so a
    # series too big for RAM spills to a memmap instead of dying in np.zeros — the ceiling
    # `analysis.label` and `analysis.segment` already reach through their own masks.
    mask_out = dense_output(tuple(raster6.shape), np.uint8, tag=f"submask_{ctx.node_id}")
    sub_out = (dense_output(tuple(raster6.shape), np.int32, tag=f"sublabels_{ctx.node_id}")
               if want_sub else None)
    submask6 = mask_out.array
    sublab6 = sub_out.array if sub_out is not None else None

    # per-parent accumulators (parent id → value); ids are global-unique across units, so a
    # plain dict merges every unit's contribution with no collision to resolve.
    p_level: Dict[int, float] = {}
    p_above: Dict[int, int] = {}
    p_total: Dict[int, int] = {}
    p_nsub: Dict[int, int] = {}
    sub_cols: Dict[str, List] = {k: [] for k in ("id", "m", "t", "c", "z", "y", "x",
                                                 "area", "parent_id", "level")}
    sub_offset = 0
    sizes_seen: List[int] = []                  # for the all-skipped diagnosis
    if lnote:
        ctx.progress(0, len(units), "using " + lnote, frames=ax.t)
    note = ("thresholding raw pixels per label" if on_raw
            else "thresholding per label")
    ctx.progress(0, len(units), note, frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        rc = lc if lc >= 0 else c               # which channel's raster gives the geometry
        if is_3d:
            lab = np.asarray(raster6[m, t, :, rc])
            img = np.asarray(prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x),
                             dtype=float)
        else:
            lab = np.asarray(raster6[m, t, z, rc])
            img = np.asarray(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), dtype=float)
        shape = lab.shape
        flat_lab = lab.reshape(-1)
        flat_img = img.reshape(-1)
        # group the unit's voxels by parent id in ONE sort — the alternative (`lab == pid`
        # per id) is a full-frame pass per region, i.e. O(voxels × regions) on a frame that
        # routinely holds hundreds of them.
        nz = np.flatnonzero(flat_lab > 0)
        if nz.size:
            keys = flat_lab[nz]
            order = np.argsort(keys, kind="stable")
            sorted_idx = nz[order]
            sorted_key = keys[order]
            ids, starts = np.unique(sorted_key, return_index=True)
            counts = np.diff(np.append(starts, sorted_key.size))
            vals = flat_img[sorted_idx]
            for pid, s, cnt in zip(ids.tolist(), starts.tolist(), counts.tolist()):
                idx = sorted_idx[s:s + cnt]
                px = vals[s:s + cnt]
                sizes_seen.append(int(cnt))
                p_total[pid] = p_total.get(pid, 0) + int(cnt)
                lvl = _region_level(px, method=method, percentile=percentile,
                                    fraction=fraction, min_pixels=min_pixels)
                if lvl is None:
                    p_level.setdefault(pid, float("nan"))
                    p_above.setdefault(pid, 0)
                    continue
                p_level[pid] = lvl
                sel = (px >= lvl) if direction == "above" else (px <= lvl)
                sel_flat = idx[sel]
                p_above[pid] = p_above.get(pid, 0) + int(sel_flat.size)
                if sel_flat.size == 0:
                    continue
                # Unravel and index the 6-D array DIRECTLY. `submask6[m, t, :, c]` is a
                # non-contiguous view (c sits between z and y), so `.reshape(-1)` on it
                # returns a COPY and the assignment would be silently discarded — the whole
                # 3D branch would come back empty with no error anywhere.
                coords = np.unravel_index(sel_flat, shape)
                if is_3d:
                    at = (m, t, coords[0], c, coords[1], coords[2])
                else:
                    at = (m, t, z, c, coords[0], coords[1])
                submask6[at] = 1
                if not want_sub:
                    continue
                per_voxel, areas, com, k = _sub_objects(coords, rank)
                p_nsub[pid] = p_nsub.get(pid, 0) + k
                sublab6[at] = per_voxel + sub_offset
                for j in range(k):
                    sub_cols["id"].append(sub_offset + j + 1)
                    sub_cols["m"].append(m)
                    sub_cols["t"].append(t)
                    sub_cols["c"].append(c)
                    # 2D: z IS the plane index (z_kind="plane_index"), matching every other
                    # per-plane producer; 3D: the component's own subpixel centroid.
                    sub_cols["z"].append(float(com[j, 0]) if is_3d else float(z))
                    sub_cols["y"].append(float(com[j, 1] if is_3d else com[j, 0]))
                    sub_cols["x"].append(float(com[j, 2] if is_3d else com[j, 1]))
                    sub_cols["area"].append(int(areas[j]))
                    sub_cols["parent_id"].append(int(pid))
                    sub_cols["level"].append(float(lvl))
                sub_offset += k
        ctx.progress(i + 1, len(units), note, frames=ax.t)

    if not p_total:
        raise ValueError(
            f"threshold per label: the label raster {layer!r} holds no region at all in the "
            f"channel(s) being thresholded, so there is nothing to derive a level inside. "
            f"Check the segmentation upstream, and — if the parents live in a different "
            f"channel from the signal — `label_channel`.")
    if not any(np.isfinite(v) for v in p_level.values()):
        srt = sorted(sizes_seen)
        raise ValueError(
            f"threshold per label: not one of the {len(p_total)} region(s) yielded a level, "
            f"so the output would be empty. They measure {srt[0]}–{srt[-1]} voxels (median "
            f"{int(np.median(srt))}) and `min_pixels` is {min_pixels}: a region below that is "
            f"skipped because a histogram method on so few samples splits the sampling noise "
            f"rather than the signal. Lower `min_pixels` if those sizes are genuinely what "
            f"you want thresholded, or segment larger parents. (A region of any size is also "
            f"skipped when every voxel in it has the SAME value — there is no cut in a flat "
            f"histogram.)")

    out = ds.with_layer(Domain.VOXEL, sub_mask_layer, mask_out.seal())
    if want_sub:
        out = out.with_layer(Domain.VOXEL, sub_layer, sub_out.seal())
        if sub_cols["id"]:
            out = out.with_structure(StructureTable(
                Domain.LABEL,
                {"id": np.asarray(sub_cols["id"], dtype=np.int64),
                 "m": np.asarray(sub_cols["m"], dtype=np.int64),
                 "t": np.asarray(sub_cols["t"], dtype=np.int64),
                 "c": np.asarray(sub_cols["c"], dtype=np.int64),
                 "z": np.asarray(sub_cols["z"], dtype=float),
                 "y": np.asarray(sub_cols["y"], dtype=float),
                 "x": np.asarray(sub_cols["x"], dtype=float),
                 "area": np.asarray(sub_cols["area"], dtype=np.int64),
                 "parent_id": np.asarray(sub_cols["parent_id"], dtype=np.int64),
                 "level": np.asarray(sub_cols["level"], dtype=float)},
                layer=sub_layer, z_kind=("subpixel" if is_3d else "plane_index")))
    # per-parent columns onto the PARENT table (one row per parent id, sorted — the same
    # order `analysis.measure` writes, so the two join positionally as well as on `id`).
    pids = sorted(p_total)
    tot = np.asarray([p_total[p] for p in pids], dtype=float)
    above = np.asarray([p_above.get(p, 0) for p in pids], dtype=np.int64)
    pcols: Dict[str, np.ndarray] = {
        "id": np.asarray(pids, dtype=np.int64),
        "level": np.asarray([p_level.get(p, float("nan")) for p in pids], dtype=float),
        "n_above": above,
        "frac_above": np.where(tot > 0, above / np.maximum(tot, 1.0), np.nan),
    }
    if want_sub:
        pcols["n_sub"] = np.asarray([p_nsub.get(p, 0) for p in pids], dtype=np.int64)
    return out.with_structure(StructureTable(Domain.LABEL, pcols, layer=layer, z_kind=zk))


register_node(
    _compute_threshold_per_label, op_key="analysis.threshold_per_label",
    label="Threshold Per Label", category="analysis",
    reads_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[
        InDataset(description=
                  "The image whose pixels are thresholded, and the Dataset carrying the "
                  "parent Label instance. Both come from here unless `raw` is wired, which "
                  "redirects the pixels only."),
        # the catalog-wide "segment on enhanced, measure on raw" seam: the parents, the
        # geometry and the calibration env stay on the main chain.
        _InRaw(),
        InString("labels", "Parent labels", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 description=
                 "Which label raster supplies the PARENT regions — one level is derived per id "
                 "in it, from only the pixels that id owns. It must be a real Label instance "
                 "(a raster plus its table, i.e. the output of Segmentation, Connected "
                 "Components or Histogram Threshold): a bare binary mask is refused, because "
                 "its whole foreground is one undivided region and thresholding 'inside' it "
                 "would just be a global threshold wearing this node's name. Always read from "
                 "the MAIN input, even when `raw` is wired."),
        InInt("label_channel", "Parent channel", unit="", field=False, default=-1,
              description=
              "Which channel of the label raster the parent geometry comes from. -1 (the "
              "default) uses the MATCHING channel — each channel thresholded against its own "
              "labels. Set it to a channel index for the cross-channel case: cells segmented "
              "on ch0, the signal you want cut living in ch1. Pair it with `signal_channel`, "
              "since pinning the geometry to one channel while thresholding all of them would "
              "write one parent id into the results table once per channel; that combination "
              "is refused rather than producing duplicate rows."),
        InInt("signal_channel", "Signal channel", unit="", field=False, default=-1,
              description=
              "Which channel's PIXELS are thresholded. -1 (the default) does every channel, "
              "each against its own labels. Set it to one index to threshold only that "
              "channel — the other half of the cross-channel pairing above, and also the way "
              "to skip the cost of channels you do not care about. It does not change the "
              "levels derived for the channels that do run: each region's level comes from "
              "its own pixels either way."),
        InString("name", "Sub-mask layer", field=False, default="submask",
                 layer_out=(Domain.VOXEL,),
                 description=
                 "Name of the binary 0/1 Voxel layer holding the selected voxels — the union "
                 "over every parent, with no ids. This is the layer to feed anything that "
                 "takes a mask; it is written whether or not sub-objects are labelled. Must "
                 "differ from the parent and sub-label layers, or one write would overwrite "
                 "another."),
        InBool("sub_labels", "Label sub-objects", field=False, default=True,
               description=
               "Whether to run connected components on the selected voxels and emit them as "
               "their own Label instance, with a `parent_id` column joining each sub-object "
               "back to the region it came from. ON is what makes 'count the puncta in each "
               "cell' a groupby on that column. Turn it OFF for 'how much of each cell is "
               "bright' — the sub-mask and the per-parent `frac_above` answer that without "
               "paying for a second labelling pass or the `n_sub` column."),
        InString("sub_name", "Sub-label layer", field=False, default="subobjects",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "Name of the sub-object output — one name into two domains, exactly like "
                 "Connected Components: a Voxel raster of global-unique ids and a Label table "
                 "with one row per sub-object (`area`, centroid, `parent_id`, and the `level` "
                 "its parent used). Inert when Label sub-objects is off. Ids are minted fresh "
                 "rather than inherited from the parent, since one parent usually contains "
                 "several sub-objects — `parent_id` is the join, not `id`."),
        InInt("min_pixels", "Min pixels", unit="", field=False, default=50,
              description=
              "The smallest region that gets a level of its own, as a voxel COUNT. Below it "
              "the region is SKIPPED — no sub-objects, `level` reported as NaN — because a "
              "histogram method on a few dozen samples splits the sampling noise and returns "
              "a confident-looking level with half the region either side of it. Deliberately "
              "a sample count rather than a µm² area: what makes a histogram estimable is how "
              "many samples are in it, so a physical floor would demand MORE samples at fine "
              "resolution and fewer at coarse, which is backwards. Lower it and small regions "
              "start contributing noisy sub-objects; raise it and they drop out of the result "
              "entirely, which the NaN in the parent table is there to make visible."),
        InInt("connectivity", "Connectivity", unit="", field=False, default=0,
              description=
              "Which neighbours make selected voxels ONE sub-object. In 2D: 4 = edge-sharing "
              "only, 8 = edges + corners. In 3D: 6 faces, 18 faces + edges, 26 everything. "
              "Higher merges more, so it lowers the sub-object COUNT and raises individual "
              "areas — the usual cause of two adjacent puncta being reported as one. 0 or "
              "unset takes 8 in 2D and 26 in 3D. Only read when Label sub-objects is on; the "
              "sub-mask itself has no connectivity. Components are computed one parent at a "
              "time, so this can never join sub-objects across two different cells however "
              "high it is set."),
        InFloat("percentile", "Percentile", unit="", field=False, default=95.0,
                available_in={"method": frozenset({"percentile"})},
                description=
                "The cut as a percentile (0–100) of THIS region's own pixel histogram — 95 "
                "means 'the brightest 5% of this cell'. The most predictable choice on small "
                "regions, where the variance-maximizing methods get unstable, and the one to "
                "reach for when every cell should contribute a comparable FRACTION of itself. "
                "Its assumption is exactly that: a cell with no real signal still yields its "
                "top 5%, so it never reports emptiness. Only shown for the `percentile` "
                "method."),
        InFloat("fraction", "Fraction", unit="", field=False, default=1.5,
                available_in={"method": frozenset({"relative"})},
                description=
                "The cut as a multiple of THIS region's median intensity — 1.5 means 'at "
                "least 1.5× the middle of this cell'. Keys on the region's own background "
                "level instead of a fixed fraction of its pixels, so unlike `percentile` a "
                "genuinely uniform cell yields nothing at all, and a cell packed with signal "
                "yields most of itself. LOWER admits dimmer structure. Only shown for the "
                "`relative` method."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("method", list(_PL_METHODS), default="otsu",
             description=
             "How each region's own level is arrived at. Every choice sees only the pixels of "
             "the region it is cutting, which is what separates this node from every other "
             "threshold in the catalog — so a bright cell and a dim one get different "
             "absolute levels by design. The five histogram methods take no parameter; the "
             "last two self-scale through one, and their socket appears when you pick them. "
             "None of them can rescue a region with too few pixels or no spread: those are "
             "skipped and reported as NaN whatever is selected here.",
             choice_docs={
                 "otsu":
                     "Maximizes between-class variance — the standard two-population split, "
                     "and the right default when each cell really does contain a bright "
                     "structure on a dimmer body. It ASSUMES that bimodality: given a "
                     "uniformly bright cell it still returns a level and splits it in two, so "
                     "read `frac_above` before trusting a cell you expected to be empty.",
                 "li":
                     "Minimizes cross-entropy between the two classes. Tends to sit LOWER "
                     "than Otsu on a skewed histogram — the usual shape for sparse puncta in "
                     "a large cell body, where Otsu is pulled toward the dominant dim mode — "
                     "so it recovers faint structure Otsu misses, at more false area. The "
                     "only iterative method here; on a flat region it is skipped like the "
                     "rest rather than failing.",
                 "yen":
                     "Maximizes a Shannon-entropy criterion on the two classes. Usually "
                     "HIGHER than Otsu, i.e. more conservative — it keeps only clearly "
                     "separated bright structure, which makes it the choice when a false "
                     "positive costs more than a miss.",
                 "triangle":
                     "Geometric: the level furthest from the line joining the histogram's "
                     "peak to its far tail. Needs no bimodality at all, so it is the most "
                     "robust option on a cell whose signal is a long bright tail rather than "
                     "a second mode — exactly where Otsu and Yen misplace the cut.",
                 "mean":
                     "The region's own mean intensity. Crude, parameter-free and perfectly "
                     "predictable; on a cell that is mostly dim it lands just above the "
                     "background and selects a large, generous area. Useful as a baseline to "
                     "compare the others against.",
                 "percentile":
                     "A fixed rank of the region's histogram (the `percentile` socket). Not a "
                     "histogram method — it makes no assumption about the shape, so it stays "
                     "stable on the small regions where the four above wobble, in exchange "
                     "for always selecting that same fraction of every cell whether or not "
                     "there is anything there.",
                 "relative":
                     "A multiple of the region's median (the `fraction` socket) — relative to "
                     "the cell's own background rather than to a rank. The one option that "
                     "can legitimately return an EMPTY selection for a cell with no signal, "
                     "which is what makes it the honest choice when some cells really are "
                     "negative.",
             }),
        Mode("direction", ["above", "below"], default="above",
             description=
             "Which side of each region's level is selected. Inclusive at the boundary, so a "
             "voxel exactly AT the level is always selected. There is no two-sided form here: "
             "a band would need two levels per region and every method above produces one.",
             choice_docs={
                 "above":
                     "Select the voxels at or ABOVE the level — bright structure inside each "
                     "region, the fluorescence case and the default (puncta, foci, a bright "
                     "rim).",
                 "below":
                     "Select the voxels at or BELOW the level — DARK structure inside each "
                     "region: a vacuole or an unstained nucleus inside a stained cell, or "
                     "bright-field data where objects are darker than their surroundings.",
             }),
    ],
    granularity=Granularity.WHOLE_VOLUME,
    description="Threshold INSIDE each label separately — every parent region derives its "
                "own level from its own pixel histogram (otsu/li/yen/triangle/mean, or the "
                "self-scaling percentile/relative), so a bright cell and a dim one are cut "
                "at different absolute levels. Writes a 0/1 sub-mask, optional re-CCL'd "
                "sub-objects carrying `parent_id` (labelled per parent, so touching cells "
                "cannot merge), and per-parent `level`/`n_above`/`frac_above`/`n_sub` "
                "columns. 2D vs 3D is inherited from the Label instance's own provenance, "
                "not levered. A region below `min_pixels` or with no spread is skipped and "
                "reported as NaN, never given the frame's global level; `label_channel` / "
                "`signal_channel` express 'cells on ch0, signal in ch1'.",
)
