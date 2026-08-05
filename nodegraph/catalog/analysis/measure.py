"""Measure (``analysis.measure``) — Per-label statistics of the image (mean/max/min/area) via the Voxel→Label bridge, plus optional µm-aware `shape` geometry columns (eccentricity, perimeter, solidity, …) from regionprops; or, on Point members, each detection's µm position and the intensity of the voxel it sits on."""

from __future__ import annotations

import numpy as np

from typing import Dict, Sequence, Tuple

from nodegraph.bridges import voxel_to_label
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import map_units
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.structure import COORD_COLUMNS, StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import (
    _label_raster,
    _point_layers,
    _resolve_label_instance,
    _resolve_layer,
)
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.progress import _parallel_progress
from nodegraph.catalog._shared.raw_measure import _InRaw, _intensity_provider

#: measure stat → its Label column name (Voxel→Label bridge reducers).
_MEASURE_COLUMNS = {
    "mean": "mean_intensity", "max": "max_intensity", "min": "min_intensity",
    "sum": "total_intensity", "median": "median_intensity", "count": "area",
}
def _layers_measure(params, modes):
    """`analysis.measure` has no output-name socket: it attaches its statistic columns to
    the Label table named by its READ socket, creating that (LABEL, name) pair when the
    upstream produced only a Voxel raster.

    The **point** branch announces nothing, and that is not an omission: it writes into a
    Point table whose invariant coordinate columns it just read, so the ``(POINT, name)``
    pair is already in the catalog from whichever detector produced it. (Its `points` socket
    also ships EMPTY — the layer is inferred, §4g — so there would be no name to predict on
    a fresh node anyway, and inventing one here would offer a layer downstream that no node
    writes.)"""
    if (modes or {}).get("target") == "point":
        return ()
    return ((Domain.LABEL, params.get("labels") or "labels"),)
def _measure_stats(raw) -> list:
    """The ``stats`` selector → an ordered, de-duplicated list of reducer names.

    Accepts the socket's **comma-separated string** (``"mean,max,median"``) or an
    already-split sequence from a programmatic caller, because :class:`SocketType` has no
    LIST member. Unknown names are refused HERE, with the whole menu in the message,
    rather than deep inside ``bridges._group_reduce``; a blank selector falls back to the
    historical default set so an emptied text box is never a silent no-op."""
    names = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in (raw or ())])
    names = [s for s in names if s] or ["mean", "max", "min", "count"]
    bad = [s for s in names if s not in _MEASURE_COLUMNS]
    if bad:
        raise ValueError(f"unknown measure stat(s) {bad} — choose from "
                         f"{list(_MEASURE_COLUMNS)} (comma-separated)")
    return list(dict.fromkeys(names))          # de-dup, preserve the user's order
#: shape metric → (``skimage.measure.regionprops`` property, works in 3D). The 2D-only
#: three raise ``NotImplementedError`` inside skimage on a volume, so the compute refuses
#: them up front against the label table's own ``z_kind`` rather than letting the backend
#: fail mid-walk. Cell-Tracker's ``measure_frame`` emits ``eccentricity``; the rest are the
#: companions regionprops computes from the same cached region and are free once asked.
_MEASURE_SHAPE: Dict[str, Tuple[str, bool]] = {
    "eccentricity": ("eccentricity", False),
    "perimeter": ("perimeter", False),
    "orientation": ("orientation", False),
    "solidity": ("solidity", True),
    "extent": ("extent", True),
    "axis_major": ("axis_major_length", True),
    "axis_minor": ("axis_minor_length", True),
}
def _measure_shape(raw) -> list:
    """The ``shape`` selector → an ordered, de-duplicated list of shape-metric names.

    Unlike ``stats``, an empty selector means **none**: the shape pass is a regionprops
    walk over the label raster (a second traversal, and the only one that touches region
    geometry), so it is opt-in rather than defaulted-on. Same two accepted forms as
    :func:`_measure_stats`, and unknown names are refused here with the whole menu."""
    names = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in (raw or ())])
    names = [s for s in names if s]
    bad = [s for s in names if s not in _MEASURE_SHAPE]
    if bad:
        raise ValueError(f"unknown shape metric(s) {bad} — choose from "
                         f"{list(_MEASURE_SHAPE)} (comma-separated)")
    return list(dict.fromkeys(names))
def _measure_shape_columns(raster6: np.ndarray, names: Sequence[str], *,
                           is_3d: bool, spacing: tuple) -> Dict[str, Dict[int, float]]:
    """``{regionprops property: {label id: value}}`` for the requested shape metrics.

    Walks the label raster one **unit** at a time — a ``(Y,X)`` plane when the Label table
    is 2D-per-plane, the ``(Z,Y,X)`` volume when it is 3D — because that is the geometry
    each region was segmented in, and it is the only shape a 2D-only metric like
    eccentricity can be computed on at all. The choice is *inherited* from the table's
    ``z_kind`` provenance (§7b), never guessed from the image.

    ``spacing`` hands the physical voxel size to ``regionprops_table``, so lengths come
    back in **µm** and an anisotropic stack is not measured as though it were cubic.

    Region ids are globally unique across units (every structure producer in the catalog
    offsets them), so the per-unit tables merge into one id→value map with no collisions —
    and a label absent from the walk simply never gets a key, which the caller turns into
    NaN rather than a silent 0."""
    from skimage.measure import regionprops_table
    props = tuple(dict.fromkeys(_MEASURE_SHAPE[n][0] for n in names))
    out: Dict[str, Dict[int, float]] = {p: {} for p in props}
    n_m, n_t, n_z, n_c = raster6.shape[:4]
    for m in range(n_m):
        for t in range(n_t):
            for c in range(n_c):
                units = ([raster6[m, t, :, c]] if is_3d
                         else [raster6[m, t, z, c] for z in range(n_z)])
                for unit in units:
                    lab = np.asarray(unit).astype(np.int32, copy=False)
                    if not lab.any():
                        continue
                    tbl = regionprops_table(lab, properties=("label",) + props,
                                            spacing=spacing)
                    for p in props:
                        for lid, val in zip(tbl["label"], tbl[p]):
                            out[p][int(lid)] = float(val)
    return out
def _compute_measure(ctx: EvalContext) -> Dataset:
    """Measure the image over the members of a structure table → columns on that table.

    Two branches, selected by the **``target``** Mode, because the two member kinds are
    measurable in genuinely different ways rather than by the same code with a switch:

    * **``label``** (:func:`_measure_labels`) — a region has extent, so it has statistics:
      per-label ``mean``/``max``/``min``/``sum``/``median`` intensity and a voxel ``count``
      via the Voxel→Label bridge, plus optional µm-aware ``regionprops`` geometry.
    * **``point``** (:func:`_measure_points`) — a detection is dimensionless, so it has a
      *position* and one *sample*: ``x_um``/``y_um``/``z_um`` and the intensity of the voxel
      it sits on, via the Voxel→Point bridge's nearest-voxel rule.

    The optional **``raw``** Dataset input serves both: it redirects *which pixels are
    measured* while the structure keeps coming from the main input — the "segment on
    enhanced, measure on raw" workflow (:func:`_intensity_provider`). Unwired, each branch
    measures its own image.

    The image provider is resolved (and refused) HERE, before the branch, because neither
    branch has anything to measure without one and the `raw` guard is the same question for
    both."""
    ds = ctx.inputs[0]
    prov, on_raw = _intensity_provider(ctx, ds)
    if prov is None:
        raise ValueError("measure needs an image provider (on its input, or via `raw`)")
    if ctx.params.get("__modes__", {}).get("target", "label") == "point":
        return _measure_points(ctx, ds, prov, on_raw=on_raw)
    return _measure_labels(ctx, ds, prov, on_raw=on_raw)
def _measure_points(ctx: EvalContext, ds: Dataset, prov, *, on_raw: bool) -> Dataset:
    """``target=point`` — each detection's **physical position** and the intensity of the
    **voxel it sits on**, as columns on its own Point table.

    A Point row is dimensionless: there is no region to reduce over, so `stats` and `shape`
    are `available_in`-gated away and this branch writes a fixed set of four columns:

    ==================  ==========================================================
    ``x_um``/``y_um``   the row's own ``x``/``y`` scaled by ``pixel_size_um``
    ``z_um``            ``z`` scaled by ``z_step_um`` (0 on a single-plane Dataset)
    ``mean_intensity``  the value of the nearest voxel to ``(z, y, x)``
    ==================  ==========================================================

    **Why µm columns rather than just re-emitting z/y/x.** The invariant schema already
    carries ``z``,``y``,``x`` — in **voxel index** units — so those are not something this
    node can add. The physical position is, and it is the form a result actually gets
    reported in. They are *new* columns, not a rewrite: every downstream consumer
    (``transform.rasterize_field``, ``analysis.tessellate``, ``track.link``) reads ``y``/``x``
    as pixel indices, so those stay exactly as the detector wrote them.

    **``mean_intensity``, though one voxel is not a mean.** It is the stable downstream key
    — the label branch guarantees it (``mean`` is force-added there for this reason) and
    ``analysis.object_field``'s `intensity` socket defaults to it, so a Point table measured
    here can feed the same graph a Label table can. Naming it ``intensity`` instead would
    make Point members the one member kind that node cannot grid.

    **Nearest voxel, not interpolated**, matching ``bridges.voxel_to_point``'s ``nearest``:
    "the pixel it is on" is the question, and a LoG-fitted sub-pixel peak still sits on
    exactly one voxel. That also makes the two ``z_kind`` provenances one code path — a
    ``plane_index`` table's ``z`` IS the plane, and a ``subpixel`` table's rounds to it.

    **What is clipped and what is refused.** ``z``/``y``/``x`` are continuous coordinates, so
    a centroid at 511.6 on a 512-wide image legitimately lands on the last voxel and is
    clipped. ``m``/``t``/``c`` are discrete addresses naming which position, frame and
    channel, so a row outside them is not an edge case — the table came from a different
    chain, and clipping would report a neighbouring frame's intensity. Those rows get **NaN**
    (never 0, which is a legal intensity), and a table with no addressable row at all is
    refused rather than returning an all-NaN column.

    Footprint: only the planes that actually hold a detection are read, so a sparse cloud on
    a 49-position file does not pay for the whole volume. The reads are independent and each
    writes a disjoint set of rows, so they fan out (V2.14) exactly as the label branch's
    gather does."""
    ax = ds.axes
    # the ONE Point table on the wire, whatever it is called (§4g) — no literal default
    # could be right for detect.spots (`spots`), detect.particles (`particles`) AND
    # transform.label_to_points (`<labels>_points`) at once.
    layer, note = _resolve_layer(
        _point_layers(ds), ctx.layer("points"), node="measure", socket="points",
        what="Point table", where="the `data` input",
        remedy="this measures one row per DETECTION, so it needs a point cloud — run "
               "detect.spots / detect.particles / transform.label_to_points upstream (or "
               "switch `target` to Label to measure regions)")
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
            if a.layer == layer}
    missing = sorted(k for k in COORD_COLUMNS if k not in cols)
    if missing:
        raise ValueError(
            f"measure: the Point layer {layer!r} is missing the invariant column(s) "
            f"{missing}, so its detections have no position to report or to sample at.")
    n = int(len(cols["id"]))
    ragged = sorted(k for k, v in cols.items() if len(v) != n)
    if ragged:
        raise ValueError(
            f"measure: column(s) {ragged} on Point layer {layer!r} disagree in length with "
            f"'id' ({n}) — every sample would be taken from a misaligned row.")
    mm = np.asarray(cols["m"], dtype=np.int64)
    tt = np.asarray(cols["t"], dtype=np.int64)
    cc = np.asarray(cols["c"], dtype=np.int64)
    zf = np.asarray(cols["z"], dtype=float)
    yf = np.asarray(cols["y"], dtype=float)
    xf = np.asarray(cols["x"], dtype=float)
    # a non-finite coordinate has no voxel; substitute 0 for the cast (np.rint(nan) cast to
    # int64 is undefined, not an error) and drop the row via `inside` below.
    finite = np.isfinite(zf) & np.isfinite(yf) & np.isfinite(xf)

    def _voxel(v: np.ndarray, hi: int) -> np.ndarray:
        return np.clip(np.rint(np.where(finite, v, 0.0)), 0, hi - 1).astype(np.int64)

    zi, yi, xi = _voxel(zf, ax.z), _voxel(yf, ax.y), _voxel(xf, ax.x)
    inside = (finite & (mm >= 0) & (mm < ax.m) & (tt >= 0) & (tt < ax.t)
              & (cc >= 0) & (cc < ax.c))
    vals = np.full(n, np.nan, dtype=float)
    live = np.flatnonzero(inside)
    if live.size == 0:
        raise ValueError(
            f"measure: not one of the {n} rows on Point layer {layer!r} addresses a "
            f"(position, frame, channel) this Dataset has — its axes are m={ax.m}, "
            f"t={ax.t}, c={ax.c}. The points and the image are not from the same chain: a "
            f"frame slice, a channel tap or a temporal stack between the detector and here "
            f"would do it. Measure where the geometry still matches, or wire the detector's "
            f"own branch into `raw`.")
    # group the rows by the plane they sample from, so each plane is read ONCE — one
    # np.unique instead of a Python pass over what can be 10^6 detections.
    key = ((mm * ax.t + tt) * ax.z + zi) * ax.c + cc
    uk, inv = np.unique(key[live], return_inverse=True)
    counts = np.bincount(inv, minlength=uk.size)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    if note:                       # say which layer was inferred, before the reads
        ctx.progress(0, int(uk.size), "using " + note, frames=ax.t)
    tick = _parallel_progress(ctx, int(uk.size),
                              "sampling raw pixels" if on_raw else "sampling pixels",
                              frames=ax.t)

    def _sample(j: int) -> None:
        k = int(uk[j])
        c, k = k % ax.c, k // ax.c
        z, k = k % ax.z, k // ax.z
        t, m = k % ax.t, k // ax.t
        rows = live[order[starts[j]:starts[j] + counts[j]]]
        plane = np.asarray(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), dtype=float)
        vals[rows] = plane[yi[rows], xi[rows]]
        tick()

    map_units(_sample, list(range(int(uk.size))))
    px = ctx.calib("pixel_size_um") or 0.1
    # A single-plane Dataset puts every detection on plane 0, so the axial step cannot move
    # z_um — and reading a calibration key the node could not have used would fence the memo
    # on it (the R1 rule `_frame_interval_s` follows for dt_s).
    zs = (ctx.calib("z_step_um") or 0.5) if ax.z > 1 else 0.0
    out = {
        "id": np.asarray(cols["id"]),                # unchanged: the row-alignment anchor
        "z_um": zf * zs, "y_um": yf * px, "x_um": xf * px,
        "mean_intensity": vals,
    }
    # Re-emits the SAME Point layer with the new columns, so its z_kind provenance (§7b)
    # must be carried forward rather than clobbered with the StructureTable default
    # ("subpixel"), which would tell every downstream node this 2D cloud was volumetric.
    zk = ds.structure_zkind(Domain.POINT, layer) or "plane_index"
    return ds.with_structure(StructureTable(Domain.POINT, out, layer=layer, z_kind=zk))
def _measure_labels(ctx: EvalContext, ds: Dataset, prov, *, on_raw: bool) -> Dataset:
    """``target=label`` — per-region statistics of the image over a label raster →
    Label-domain attributes (Voxel→Label bridge, V2.00 §6). Emits one column per requested
    stat (``mean_intensity``/``max_intensity``/``min_intensity``/``total_intensity``/
    ``area`` = voxel count); ``mean`` is always present so ``mean_intensity`` stays a
    stable downstream key. Every stat shares the sorted-id order of the same raster.

    ``shape`` (V2.13) adds per-region **geometry** columns from ``regionprops`` —
    Cell-Tracker's ``eccentricity`` and its companions. They are a different kind of
    measurement from the intensity stats above: they read only the label raster, never the
    image, so ``raw`` does not affect them, and lengths are reported in **µm** via the
    voxel ``spacing``. Off by default (they cost a second walk), and the 2D-only ones are
    refused on a 3D Label table rather than failing inside skimage."""
    ax = ds.axes
    # the ONE Label instance on the wire, whatever it is called (`_resolve_layer`) — the
    # literal default agreed only with `analysis.segment`'s own default name.
    layer, _lnote = _resolve_label_instance(
        ds, ctx.layer("labels"), node="measure", socket="labels",
        remedy="this measures one row per REGION, so it needs a label raster AND its table "
               "— run analysis.segment / analysis.label upstream")
    raster6, _zk_declared = _label_raster(ds, layer, node="measure")
    img = np.zeros_like(raster6, dtype=float)
    # this gather REALIZES the whole lazy chain plane by plane — for a deep enhancement
    # chain it is the run's real cost, so it is worth a determinate bar. And because it IS
    # the real cost, it is also the thing worth fanning out (V2.14): the reads are
    # independent, each writes a disjoint slice, and the upstream tile computes they trigger
    # are pure. Purely a map — no ordering constraint, no fold state.
    _note = "reading raw planes" if on_raw else "reading planes"
    if _lnote:                     # say which layer was inferred, before the long gather
        ctx.progress(0, ax.m * ax.t * ax.z * ax.c, "using " + _lnote, frames=ax.t)
    tick = _parallel_progress(ctx, ax.m * ax.t * ax.z * ax.c, _note, frames=ax.t)

    def _gather(unit):
        m, t, z, c = unit
        img[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        tick()

    map_units(_gather, list(_each_plane(ax)))
    stats = _measure_stats(ctx.params.get("stats", ("mean", "max", "min", "count")))
    if "mean" not in stats:
        stats = ["mean", *stats]
    cols: Dict[str, np.ndarray] = {}
    for st in stats:
        ids, values = voxel_to_label(img, raster6, st)
        cols.setdefault("id", ids)                       # same sorted ids for every stat
        cols[_MEASURE_COLUMNS.get(st, st)] = values
    # Re-emits the SAME label layer with measurement columns — carry the label's z_kind
    # provenance forward (§7b) so it is not clobbered with the StructureTable default.
    #
    # `_label_raster` above already refused a layer with NO Label table, so this is the
    # declared provenance, never a guess. (Before 2026-07-30 the fallback here was
    # "subpixel" — assume 3D — which on a file with no Z axis ran the 3D regionprops walk
    # over a single (1,Y,X) plane; qhull cannot hull a coplanar point set and
    # `area / area_convex` divided by zero, so `solidity` came back **inf**.)
    zk = _zk_declared
    # ── shape metrics (V2.13): geometry, not intensity — raster only, `raw` irrelevant ──
    shape_names = _measure_shape(ctx.params.get("shape", ""))
    if shape_names:
        # Belt and braces: even CORRECT `subpixel` provenance is unmeasurable in 3D when
        # the volume is one plane deep — analysis.label with the lever on 3D stamps
        # `subpixel` truthfully on this z==1 ND2, and every region would still hit qhull.
        # A single plane IS a 2-D region whatever the table says, so walk it as one.
        shape_3d = (zk == "subpixel") and ax.z > 1
        flat = [n for n in shape_names if shape_3d and not _MEASURE_SHAPE[n][1]]
        if flat:
            raise ValueError(
                f"measure: shape metric(s) {flat} are defined for 2-D regions only and the "
                f"Label table {layer!r} is 3D (z_kind='subpixel'), so skimage cannot "
                f"compute them on a volume. Drop them (solidity / extent / axis_major / "
                f"axis_minor do work in 3D), or segment with the 2D/3D lever on 2D so each "
                f"plane's regions are their own objects.")
        px = ctx.calib("pixel_size_um") or 0.1
        spacing = (((ctx.calib("z_step_um") or 0.5), px, px) if shape_3d else (px, px))
        found = _measure_shape_columns(raster6, shape_names,
                                       is_3d=shape_3d, spacing=spacing)
        row_ids = cols["id"].tolist()                     # `mean` is forced ⇒ always set
        for name in shape_names:
            lookup = found[_MEASURE_SHAPE[name][0]]
            # NaN, not 0, for an id the walk never saw: 0 is a legal eccentricity (a
            # perfect circle), so a missing region must not read as a round one.
            cols[name] = np.array([lookup.get(int(i), np.nan) for i in row_ids],
                                  dtype=float)
    return ds.with_structure(StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk))
register_node(
    _compute_measure, op_key="analysis.measure", label="Measure", category="analysis",
    extra_layers=_layers_measure,
    # VOXEL is unconditional because it is the IMAGE domain — both branches measure pixels,
    # and §4f is explicit that the image is not a per-branch layer requirement. What DOES
    # vary is the structure: the label branch needs a Label table (its raster's ids must
    # divide the foreground into objects), the point branch a Point table. Declaring the
    # static union `{VOXEL, LABEL, POINT}` would paint a red PT chip on every Label graph
    # that works — the exact over-claim `analysis.object_field` was fixed for in V2.22.
    reads_domains=frozenset({Domain.VOXEL}),
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.VOXEL, Domain.LABEL}),
        "point": frozenset({Domain.POINT}),
    }},
    adds_domains=frozenset({Domain.LABEL, Domain.POINT}),
    modes=[Mode("target", ["label", "point"], default="label", label="Members",
                description=
                "Which members are measured — the REGIONS of a Label table or the "
                "DETECTIONS of a Point table. It selects the live source socket and, because "
                "the two member kinds are measurable in different ways, which columns come "
                "out: a region has extent, so it has statistics and geometry; a point is "
                "dimensionless, so it has a position and one sample. Both write onto the "
                "table they read, and both honour the `raw` input.",
                choice_docs={
                    "label":
                        "Measure the regions of a Label table (Segmentation / Connected "
                        "Components): per-region intensity statistics over every voxel the "
                        "region owns, plus optional µm regionprops geometry. This is the "
                        "branch `stats` and `shape` belong to; it needs a raster whose ids "
                        "divide the foreground into objects, not a bare mask.",
                    "point":
                        "Measure the detections of a Point table (Spot / Particle Detection, "
                        "Label to Points): writes `x_um`/`y_um`/`z_um` — the physical "
                        "position, which the invariant pixel `x`/`y`/`z` columns are not — "
                        "and `mean_intensity`, the value of the single voxel each detection "
                        "sits on. `stats` and `shape` are hidden here: there is no region to "
                        "reduce over or fit an ellipse to.",
                })],
    inputs=[InDataset(), _InRaw(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     available_in={"target": frozenset({"label"})},
                     description=
                     "Which label raster defines the regions to measure — the output of "
                     "Connected Components or Segmentation. One row is produced per label "
                     "id, so this choice fixes both WHAT gets measured and how many rows "
                     "come out. It always comes from the MAIN input, even when a `raw` "
                     "Dataset is wired: `raw` redirects only the pixels being measured."),
            # Ships EMPTY, unlike `labels` above: no literal default can be right for
            # detect.spots (`spots`), detect.particles (`particles`) and
            # transform.label_to_points (`<labels>_points`) at once, so the layer is
            # inferred from the wire when there is only one candidate (§4g).
            InString("points", "Point layer", field=False, default="",
                     layer_in=Domain.POINT,
                     available_in={"target": frozenset({"point"})},
                     description=
                     "Which point cloud is measured — the output of Spot Detection, Particle "
                     "Detection or Label to Points. One row comes out per DETECTION, and the "
                     "new columns are written back onto this same table. Leave it EMPTY and "
                     "the only Point table on `data` is used, which is what you want on a "
                     "single-branch graph; name one explicitly when the wire carries two. "
                     "Like `labels`, it always comes from the MAIN input even when `raw` is "
                     "wired — `raw` redirects only the pixels being sampled."),
            # comma-separated: SocketType has no LIST member. `mean` is force-added by
            # the compute, so `mean_intensity` stays a stable downstream key.
            InString("stats", "Statistics", field=False,
                     vocab=tuple(_MEASURE_COLUMNS),
                     default="mean,max,min,count",
                     available_in={"target": frozenset({"label"})},
                     description=
                     "Which statistics to compute, comma-separated, one Label column each: "
                     "`mean`→mean_intensity, `max`→max_intensity, `min`→min_intensity, "
                     "`sum`→total_intensity, `median`→median_intensity, `count`→area (the "
                     "voxel count, NOT a physical area). `mean` is always computed even if "
                     "omitted, so `mean_intensity` stays available to downstream nodes. "
                     "Asking for fewer stats writes fewer columns; it does not change any "
                     "value, since each is computed independently over the same regions.",
                     choice_docs={
                         "mean":
                             "→ `mean_intensity`: the region's average voxel value. The "
                             "size-independent measure of how bright a cell is, and the "
                             "column downstream nodes look for by default; always computed "
                             "even when unticked.",
                         "max":
                             "→ `max_intensity`: the region's brightest single voxel. "
                             "Sensitive to one hot pixel by construction, but the right "
                             "column for \"did this cell contain a punctum\" rather than "
                             "\"how bright is it overall\".",
                         "min":
                             "→ `min_intensity`: the region's dimmest single voxel. Mostly a "
                             "diagnostic — on a mask that leaked into background it reads "
                             "near the noise floor, which is how you catch an over-generous "
                             "threshold.",
                         "sum":
                             "→ `total_intensity`: every voxel value added up, so it scales "
                             "with BOTH brightness and size. The column to use for total "
                             "protein per cell; useless for comparing cells of different "
                             "sizes, where `mean` is what you want.",
                         "median":
                             "→ `median_intensity`: the region's middle value. Like `mean` "
                             "but unmoved by a few saturated voxels or a punctum, so it "
                             "reports the cell's bulk level rather than its brightest "
                             "structure.",
                         "count":
                             "→ `area`: the number of voxels in the region — a raw COUNT, "
                             "not a physical area, so it changes with binning and resolution. "
                             "For µm² / µm³ use Object Metrics, or scale by pixel_size_um "
                             "yourself.",
                     }),
            InString("shape", "Shape metrics", field=False, default="",
                     vocab=tuple(_MEASURE_SHAPE),
                     available_in={"target": frozenset({"label"})},
                     description=
                     "Per-region GEOMETRY columns, comma-separated — empty (the default) "
                     "computes none. `eccentricity` (0 = circle, →1 = elongated; this is the "
                     "one Cell-Tracker reports), `perimeter` and `axis_major`/`axis_minor` "
                     "in µm, `orientation` in radians, `solidity` (region area / its convex "
                     "hull's — below 1 means concave or clumped) and `extent` (region area / "
                     "bounding box). These read the LABEL RASTER only, never the pixels, so "
                     "a wired `raw` input does not affect them. Lengths use pixel_size_um "
                     "(and z_step_um in 3D) so they are physical, not pixel counts. "
                     "eccentricity, perimeter and orientation are 2-D-only in skimage and "
                     "are refused on a 3D Label table; the other four work in both. Costs a "
                     "second walk over the raster, which is why it is opt-in.",
                     choice_docs={
                         "eccentricity":
                             "Elongation of the best-fit ellipse: 0 is a perfect circle, "
                             "approaching 1 is a needle. The shape column Cell-Tracker "
                             "reports, and the usual discriminator between round and spread "
                             "cells. 2-D only — refused on a 3D Label table.",
                         "perimeter":
                             "Boundary length in µm (from pixel_size_um, not a pixel count). "
                             "Rises steeply with a ragged or noisy outline, so compare it "
                             "only between masks made the same way; pair it with area for a "
                             "circularity of your own. 2-D only.",
                         "orientation":
                             "Angle of the best-fit ellipse's major axis, in RADIANS, in "
                             "[-π/2, π/2] — the region's alignment, e.g. for cells oriented "
                             "along a fibre. Meaningless for a near-circular region, where "
                             "the axis is arbitrary. 2-D only.",
                         "solidity":
                             "Region area divided by its convex hull's. 1 means convex; "
                             "clearly below 1 means concave, branched, or two touching cells "
                             "merged into one label — which makes this the cheapest detector "
                             "of under-segmentation. Works in 2D and 3D.",
                         "extent":
                             "Region area divided by its bounding box's. High for a compact "
                             "axis-aligned blob, low for anything diagonal, curved or "
                             "sprawling. A rough, very cheap shape summary; unlike "
                             "`solidity` it also drops just for being tilted. 2D and 3D.",
                         "axis_major":
                             "Length of the best-fit ellipse/ellipsoid's LONGEST axis, in µm "
                             "(z_step_um included in 3D). The size measure to use when "
                             "\"how long is this cell\" matters more than its area.",
                         "axis_minor":
                             "Length of the SHORTEST axis of the same fit, in µm — the "
                             "region's width. Its ratio to `axis_major` is the aspect ratio, "
                             "and unlike `eccentricity` this pair is available in 3D.",
                     })],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_VOLUME,
    description="Measures the image over a structure table's members. On LABEL members: "
                "per-region statistics (mean/max/min/area) via the Voxel→Label bridge, plus "
                "optional µm-aware `shape` geometry columns (eccentricity, perimeter, "
                "solidity, …) from regionprops. On POINT members: each detection's physical "
                "position (`x_um`/`y_um`/`z_um`) and `mean_intensity`, the value of the "
                "voxel it sits on. Optional `raw` input measures THOSE pixels instead "
                "(segment on enhanced, measure on raw); the structure, the labels and the "
                "shape metrics always come from the main input.",
)
