"""Measure (``analysis.measure``) — Per-label statistics of the image (mean/max/min/area) via the Voxel→Label bridge, plus optional µm-aware `shape` geometry columns (eccentricity, perimeter, solidity, …) from regionprops; or, on Point members, each detection's µm position and the intensity of the voxel it sits on; or, on Pixel members, every voxel of chosen regions (or of the volume) as a row of a new Point table. Every branch also reports each row's absolute stage position (`stage_x_um`/`stage_y_um`/`stage_z_um`), for comparing objects across a multi-position acquisition."""

from __future__ import annotations

import re

import numpy as np

from typing import Dict, List, Optional, Sequence, Tuple

from nodegraph.bridges import voxel_to_label
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import map_units
from nodegraph.registry import (
    InBool, Granularity, InDataset, InInt, InString, Mode, OutDataset,
)
from nodegraph.structure import COORD_COLUMNS, StructureTable
from nodegraph.tables import mark_pixel_table, pixel_layers

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import on_layer, resolved_layer
from nodegraph.catalog._shared.labels import (
    _label_instances,
    _label_raster,
    _point_layers,
    _resolve_label_instance,
    _resolve_layer,
    _voxel_layers,
)
from nodegraph.catalog._shared.table_ops import envelope_has, frame_seconds
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.progress import _parallel_progress
from nodegraph.catalog._shared.raw_measure import _InRaw, _intensity_provider

#: measure stat → its Label column name (Voxel→Label bridge reducers).
_MEASURE_COLUMNS = {
    "mean": "mean_intensity", "max": "max_intensity", "min": "min_intensity",
    "sum": "total_intensity", "median": "median_intensity", "count": "area",
    "std": "std_intensity",
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
    writes.)

    The **pixel** branch writes a NEW Point table. Its `pixels` socket is a ``layer_out``
    and announces it — except when the box is blank, where the compute falls back to
    ``pixels`` and this announces that same fallback."""
    target = (modes or {}).get("target")
    if target == "point":
        return ()
    if target == "pixel":
        return ((Domain.POINT, str((params or {}).get("pixels") or _PIXEL_LAYER)),)
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
def _stage_position_columns(ds: Dataset, m: np.ndarray, x_um: np.ndarray, y_um: np.ndarray,
                            z_index: np.ndarray, flip: Tuple[bool, bool] = (True, False)
                            ) -> Dict[str, np.ndarray]:
    """``stage_x_um``/``stage_y_um``/``stage_z_um`` — each row's LOCAL ``x_um``/``y_um``
    (this field's own physical position) plus its multipoint's absolute corner
    (:func:`nodegraph.placement.field_box`), and its absolute focus
    (:func:`nodegraph.placement.z_um_of_slice`) at its own z-index ``z_index`` (a Point or
    Label row's ``z`` column — an integer plane index or a subpixel centroid, whichever
    ``z_um_of_slice`` was written to take; the two ``z_kind`` provenances are one code path
    there exactly as they are for the local ``z_um`` above).

    One :func:`field_box` call and two :func:`z_um_of_slice` samples per DISTINCT
    multipoint, never per row — ``z_um_of_slice`` is affine in ``k``
    (``z_nom + dir*(k-home)*step``), so sampling ``k=0`` and ``k=1`` recovers the intercept
    and slope without a second copy of the ``z_home_index``/``z_bottom_to_top`` anchor math,
    which must never drift from the one copy :mod:`nodegraph.placement` already owns.

    NaN wherever the field cannot be placed (no ``pixel_size_um``, no stage log at all, or —
    for Z only — no focus-log anchoring on a multi-slice stack): never a guessed 0, which is
    a legal position, matching the rule :mod:`nodegraph.placement` itself follows.

    **Handedness** (``flip``, 2026-10-13): whether the image's +x runs along stage −x and its
    +y along stage −y — :data:`nodegraph.placement.DEFAULT_FLIP`, the Stitch / Overlay /
    Merge default. Until this the row was always placed as ``corner + x_um``, i.e. unflipped,
    so on this scope every stage x was MIRRORED inside its field (up to ± half a field): one
    bead seen from two overlapping fields got two stage positions. Measured on the IL-8 bead
    data: with +x → −x, 1189 cross-field bead pairs coincide within 3 µm (median 0.74 µm);
    unflipped, 58.

    **Z after a Stitch** (2026-10-14). A mosaic is ONE field made of many, so Stitch drops
    the per-position focus log (``stage_z_um``) and :func:`z_um_of_slice` has nothing to
    anchor — every stitched row's ``stage_z_um`` came out NaN although ``origin_um``, the
    transform-maintained corner :func:`field_box` already places X/Y from, still carries the
    stack's lowest focus (``box.z0``; ``box.z1`` its highest). So when the log cannot place
    the field and the box can, slice ``k`` sits ``k`` steps above ``z0`` — below ``z1`` on a
    top-to-bottom stack — and a one-plane field sits at ``z0``. Seen on a 2×2 GFP spheroid
    mosaic whose 3-D centroid had X/Y and no Z."""
    from nodegraph.placement import field_box, z_um_of_slice
    md, axes = ds.metadata, ds.axes
    mi = np.asarray(m, dtype=np.int64)
    zk = np.asarray(z_index, dtype=float)
    n = mi.shape[0]
    sx = np.full(n, np.nan)
    sy = np.full(n, np.nan)
    sz = np.full(n, np.nan)
    for mv in np.unique(mi):
        mv = int(mv)
        if not (0 <= mv < axes.m):
            continue
        mask = mi == mv
        box = field_box(md, axes, mv)
        if box is not None:
            sx[mask] = (box.x1 - x_um[mask]) if flip[0] else (box.x0 + x_um[mask])
            sy[mask] = (box.y1 - y_um[mask]) if flip[1] else (box.y0 + y_um[mask])
        z0 = z_um_of_slice(md, axes, mv, 0.0)
        if z0 is None:
            if box is None or box.z0 is None:
                continue
            if axes.z <= 1:
                sz[mask] = box.z0
                continue
            step = (box.z1 - box.z0) / (axes.z - 1) if box.z1 is not None else 0.0
            if not step > 0:
                continue
            up = bool(md.get("z_bottom_to_top", True))
            sz[mask] = (box.z0 + step * zk[mask]) if up else (box.z1 - step * zk[mask])
            continue
        if axes.z <= 1:
            sz[mask] = z0
            continue
        z1 = z_um_of_slice(md, axes, mv, 1.0)
        if z1 is not None:
            sz[mask] = z0 + (z1 - z0) * zk[mask]
    return {"stage_x_um": sx, "stage_y_um": sy, "stage_z_um": sz}
def _label_position_columns(ds: Dataset, layer: str, row_ids: np.ndarray, *,
                            px: float, zs: float, flip: Tuple[bool, bool] = (True, False)
                            ) -> Dict[str, np.ndarray]:
    """Local ``x_um``/``y_um``/``z_um`` and absolute ``stage_x_um``/``stage_y_um``/
    ``stage_z_um`` for ``row_ids`` (measure's own ``id`` order), from the region centroid
    columns the label/segment node already wrote (invariant ``m``/``z``/``y``/``x``,
    :mod:`nodegraph.structure`). Measure never recomputes a centroid — it looks the
    original one up **by id**, the same way the ``shape`` columns above look up regionprops
    output — because :meth:`Dataset.with_structure` only replaces the columns THIS node's
    own table carries, so the producer's ``m``/``z``/``y``/``x`` survive untouched on the
    same ``(LABEL, layer)`` attributes for measure to read back."""
    ids_attr = ds.get(Domain.LABEL, "id", layer=layer)
    orig_ids = ids_attr.values

    def _by_id(name: str) -> Dict[int, float]:
        attr = ds.get(Domain.LABEL, name, layer=layer)
        return {int(i): float(v) for i, v in zip(orig_ids, attr.values)}

    m_by_id, z_by_id = _by_id("m"), _by_id("z")
    y_by_id, x_by_id = _by_id("y"), _by_id("x")
    m = np.array([m_by_id.get(int(i), -1) for i in row_ids], dtype=np.int64)
    z_px = np.array([z_by_id.get(int(i), np.nan) for i in row_ids], dtype=float)
    y_px = np.array([y_by_id.get(int(i), np.nan) for i in row_ids], dtype=float)
    x_px = np.array([x_by_id.get(int(i), np.nan) for i in row_ids], dtype=float)
    out = {"z_um": z_px * zs, "y_um": y_px * px, "x_um": x_px * px}
    out.update(_stage_position_columns(ds, m, out["x_um"], out["y_um"], z_px, flip=flip))
    return out
def _flip(ctx: EvalContext, ds: Optional[Dataset] = None) -> Tuple[bool, bool]:
    """The ``flip_x`` / ``flip_y`` handedness for the stage columns (defaults
    :data:`nodegraph.placement.DEFAULT_FLIP`, the same as Stitch's and Overlay's) — applied to
    a stage-Stitched mosaic exactly as to a camera field. ``ds`` is accepted and not read.

    **A mosaic is camera-handed** (2026-10-16): with Flip X on, ``util.stitch`` puts the
    highest-x tile on the LEFT (``u = xs.max() - xs``) and keeps every tile's pixels as the
    camera took them, so the mosaic's +x runs along stage −x like each field's. Verified on
    the IL-8 well: the beads of a 12 × 12 whole-well Cy5 mosaic land on the 20 beads the
    bead runs placed (median 5 µm) only when flipped; unflipped, none is within 30 µm. A
    15 Oct change that switched the flip off on a mosaic (following Overlay's "already
    stage-placed" rule) mirrored every mosaic row about the mosaic's centre — a spheroid
    centroid 46 µm off — and was reverted here. Overlay's rule went the same day
    (:func:`nodegraph.placement.frame_mapping`)."""
    from nodegraph.placement import DEFAULT_FLIP
    return (bool(ctx.params.get("flip_x", DEFAULT_FLIP[0])),
            bool(ctx.params.get("flip_y", DEFAULT_FLIP[1])))


# ── Members = pixel (2026-11-02): one row per voxel of chosen regions, or of the volume ──

#: The Point layer a pixel table is written under when the `pixels` socket is blank.
_PIXEL_LAYER = "pixels"
#: The `max_rows` default — a few million rows is ~0.5 GB of table at this node's column
#: dtypes, which a workstation holds comfortably and the strip view pages through.
_PIXEL_MAX_ROWS = 5_000_000
#: geometry option → its column (both in µm, NaN on a background voxel)
_PIXEL_GEOMETRY: Dict[str, str] = {"edge_dist": "edge_dist_um",
                                   "centroid_dist": "centroid_dist_um"}
#: the longest `ids` range written as `a-b` that is expanded (a typo is not 10^9 ids)
_PIXEL_MAX_ID_RANGE = 10_000_000


def _pixel_geometry(raw) -> list:
    """The ``geometry`` selector → an ordered, de-duplicated list of options — empty means
    none (each one is an extra pass). Unknown names are refused with the whole menu."""
    names = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in (raw or ())])
    names = [s for s in names if s]
    bad = [s for s in names if s not in _PIXEL_GEOMETRY]
    if bad:
        raise ValueError(f"unknown pixel geometry option(s) {bad} — choose from "
                         f"{list(_PIXEL_GEOMETRY)} (comma-separated)")
    return list(dict.fromkeys(names))


def _parse_ids(raw) -> Optional[np.ndarray]:
    """The ``ids`` selector → sorted unique region ids, or ``None`` (blank = every region).

    Accepts ``12, 40 100-120`` — commas, spaces or semicolons between ids, ``a-b`` for an
    inclusive range. Anything else is refused by name rather than skipped, because a
    silently dropped id is a region missing from the table with nothing saying so."""
    text = re.sub(r"\s*-\s*", "-", str(raw or "").strip())
    if not text:
        return None
    out: List[np.ndarray] = []
    for tok in re.split(r"[,;\s]+", text):
        if not tok:
            continue
        one = re.fullmatch(r"\d+", tok)
        span = re.fullmatch(r"(\d+)-(\d+)", tok)
        if one:
            out.append(np.array([int(tok)], dtype=np.int64))
        elif span:
            a, b = sorted((int(span.group(1)), int(span.group(2))))
            if b - a + 1 > _PIXEL_MAX_ID_RANGE:
                raise ValueError(f"measure: the id range {tok!r} spans {b - a + 1:,} ids — "
                                 f"more than {_PIXEL_MAX_ID_RANGE:,}; narrow it")
            out.append(np.arange(a, b + 1, dtype=np.int64))
        else:
            raise ValueError(f"measure: {tok!r} in `ids` is not a region id — write ids as "
                             f"whole numbers, comma-separated, with a-b for a range "
                             f"(e.g. `12, 40, 100-120`)")
    return np.unique(np.concatenate(out)) if out else None


def _channel_columns(names: Sequence[str], n_c: int) -> List[str]:
    """One intensity column per image channel, named for the channel (``GFP_intensity``) —
    ``c<i>_intensity`` when it has no usable name or two channels share one. None on a
    one-channel image: the row's own ``intensity`` already is that channel.

    The SAME function names them at edit time (from the envelope's ``channel_names``) and at
    pull time (from the payload's), so the column catalog predicts what the pull writes."""
    if int(n_c) <= 1:
        return []
    names = list(names or ())
    out: List[str] = []
    for i in range(int(n_c)):
        slug = re.sub(r"[^0-9A-Za-z]+", "_", str(names[i] if i < len(names) else "")).strip("_")
        col = f"{slug}_intensity" if slug else f"c{i}_intensity"
        if col in out:
            col = f"c{i}_intensity"
        out.append(col)
    return out


def _pixel_columns(channel_cols: Sequence[str], geometry: Sequence[str],
                   timed: bool) -> List[str]:
    """Every column a pixel table carries, in the order it is written — shared by the
    compute and its edit-time declaration."""
    cols = [*COORD_COLUMNS, "label_id", "x_um", "y_um", "z_um",
            "stage_x_um", "stage_y_um", "stage_z_um", "intensity", *channel_cols]
    if timed:
        cols.append("time_s")
    cols += [_PIXEL_GEOMETRY[g] for g in geometry]
    return cols


def _pixel_raster(ctx: EvalContext, ds: Dataset, *, volume: bool
                  ) -> Tuple[Optional[str], Optional[np.ndarray], str]:
    """The raster whose voxels are gathered → ``(layer, raster6, note)``.

    Unlike the label branch this accepts ANY Voxel raster the `labels` socket names — a
    Label instance, a threshold mask, a drawn region. The label branch refuses a bare mask
    because it would report one row for the whole foreground under per-object column names;
    here every row is one voxel, so a mask is honestly what it is, a region whose voxels
    all carry the mask's value in ``label_id``.

    ``volume``: a raster is optional. It fills ``label_id`` when the socket names one (or
    exactly one Label instance is on the wire); otherwise every voxel has ``label_id`` 0."""
    want = ctx.layer("labels")
    present = set(_voxel_layers(ds))
    if want and want in present:
        layer, note = want, ""
    elif volume:
        inst = _label_instances(ds)
        layer, note = ((inst[0], f"Label instance {inst[0]!r} (the only one on the `data` "
                                 f"input) for label_id") if len(inst) == 1 else (None, ""))
    else:
        layer, note = _resolve_label_instance(
            ds, want, node="measure", socket="labels",
            remedy="Members = pixel gathers the voxels of the regions of a raster — run "
                   "analysis.segment / analysis.label upstream, name a mask or a drawn "
                   "region in `Label layer`, or set Pixels of to volume for every voxel")
    if layer is None:
        return None, None, note
    raster = np.asarray(ds.get(Domain.VOXEL, layer).values)
    if raster.ndim != 6:
        raise ValueError(f"measure: the Voxel layer {layer!r} has {raster.ndim} axes, not the "
                         f"(m,t,z,c,y,x) a raster carries")
    if raster.dtype.kind == "f":
        finite = raster[np.isfinite(raster)]
        if finite.size and not np.all(finite == np.round(finite)):
            raise ValueError(
                f"measure: the Voxel layer {layer!r} holds fractional values, so it is a "
                f"field (a distance map, an image) rather than regions — its voxels have no "
                f"region id to report. Name a label raster or a mask in `Label layer`.")
    return layer, raster, note


def _edge_distances(raster: np.ndarray, rows: Dict[str, np.ndarray], label_id: np.ndarray,
                    *, is_3d: bool, spacing: tuple, ctx: EvalContext) -> np.ndarray:
    """Each row's distance (µm) to the nearest voxel OUTSIDE its own region — the region's
    Euclidean distance transform, one region at a time inside its bounding box padded by a
    voxel, so two touching regions each see the other as outside. A rim voxel reads one
    voxel spacing; the image border counts as outside (nothing beyond it was imaged). NaN
    for a background row (``label_id`` 0).

    The unit is the geometry the regions were segmented in: a plane for a per-plane Label
    table or a mask, the ``(Z,Y,X)`` volume for a 3D one (``spacing`` carries z_step_um)."""
    from scipy.ndimage import distance_transform_edt, find_objects
    n = int(label_id.shape[0])
    out = np.full(n, np.nan, dtype=np.float32)
    m, t, z, c = (rows[k].astype(np.int64) for k in ("m", "t", "z", "c"))
    y, x = rows["y"].astype(np.int64), rows["x"].astype(np.int64)
    cl = raster.shape[3]
    unit = (m * raster.shape[1] + t) * cl + c
    if not is_3d:
        unit = unit * raster.shape[2] + z
    live = np.flatnonzero(label_id != 0)
    if live.size == 0:
        return out
    order = live[np.argsort(unit[live], kind="stable")]
    bounds = np.flatnonzero(np.diff(unit[order])) + 1
    blocks = np.split(order, bounds)
    for bi, idx in enumerate(blocks):
        r0 = int(idx[0])
        sub = (raster[m[r0], t[r0], :, c[r0]] if is_3d
               else raster[m[r0], t[r0], z[r0], c[r0]])
        sub = np.asarray(sub).astype(np.int64, copy=False)
        want = np.unique(label_id[idx])
        pos = np.clip(np.searchsorted(want, sub), 0, want.size - 1)
        hit = (sub != 0) & (want[pos] == sub)
        compact = np.zeros(sub.shape, dtype=np.int32)
        compact[hit] = pos[hit] + 1
        dist = np.zeros(sub.shape, dtype=np.float32)
        for k, sl in enumerate(find_objects(compact), 1):
            if sl is None:
                continue
            box = tuple(slice(max(0, s.start - 1), min(dim, s.stop + 1))
                        for s, dim in zip(sl, sub.shape))
            region = compact[box] == k
            d = distance_transform_edt(np.pad(region, 1), sampling=spacing)
            d = d[tuple(slice(1, -1) for _ in range(region.ndim))]
            view = dist[box]
            view[region] = d[region]
        out[idx] = (dist[z[idx], y[idx], x[idx]] if is_3d else dist[y[idx], x[idx]])
        ctx.progress(bi + 1, len(blocks), "edge distance")
    return out


def _centroid_distances(rows: Dict[str, np.ndarray], label_id: np.ndarray,
                        phys: Tuple[np.ndarray, np.ndarray, np.ndarray], *,
                        is_3d: bool) -> np.ndarray:
    """Each row's distance (µm) from its region's centroid — the mean physical position of
    the region's voxels, which are all rows here (a region is gathered whole or not at all).
    Grouped per plane for a per-plane Label table or a mask, per ``(Z,Y,X)`` volume for a 3D
    one. NaN for a background row."""
    n = int(label_id.shape[0])
    out = np.full(n, np.nan, dtype=np.float32)
    live = np.flatnonzero(label_id != 0)
    if live.size == 0:
        return out
    keys = [rows[k][live].astype(np.int64) for k in (("m", "t", "c") if is_3d
                                                    else ("m", "t", "z", "c"))]
    keys.append(label_id[live].astype(np.int64))
    # one packed int64 key when it fits (a 1-D unique is ~10x the speed of a row-wise one
    # at millions of rows); the row-wise form only when the ids are too large to pack
    radix = [int(k.max()) + 1 for k in keys]
    if keys[-1].min() >= 0 and float(np.prod([float(r) for r in radix])) < 2.0 ** 62:
        packed = np.zeros(live.size, dtype=np.int64)
        for k, r in zip(keys, radix):
            packed = packed * r + k
        _u, inv = np.unique(packed, return_inverse=True)
    else:
        _u, inv = np.unique(np.stack(keys, axis=1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv).astype(float)
    sq = np.zeros(live.size, dtype=float)
    for p in phys:
        v = np.asarray(p, dtype=float)[live]
        sq += (v - (np.bincount(inv, weights=v) / cnt)[inv]) ** 2
    out[live] = np.sqrt(sq)
    return out


def _measure_pixels(ctx: EvalContext, ds: Dataset, prov, *, on_raw: bool) -> Dataset:
    """``target=pixel`` — one row per VOXEL, as a Point table of its own.

    Resolved spec (2026-11-02, `build-node-v2`):

    * **Which voxels** — ``Pixels of``: ``regions`` gathers the voxels inside the regions of
      the raster `labels` names (a Label instance, a mask, a drawn region; ``ids`` narrows it
      to chosen region ids), ``volume`` every voxel of the volume (``label_id`` from the
      raster when there is one, else 0).
    * **Where the rows go** — a Point table named by `pixels`, ``z_kind="plane_index"``: a
      voxel's address IS the invariant ``id,m,t,c,z,y,x`` schema, with no sub-pixel part. Its
      ``id`` is a running row number (1…n) rather than the region's id, because every
      consumer of a Point table reads an id as ONE object; the region is ``label_id``. The
      layer is marked in :data:`nodegraph.tables.PIXEL_TABLES_KEY` so the Viewer does not
      draw a marker per pixel, the spreadsheet shows a strip, and the export gives it a file.
    * **Columns** — ``label_id``; ``x_um``/``y_um``/``z_um`` and absolute
      ``stage_x_um``/``stage_y_um``/``stage_z_um`` (:func:`_stage_position_columns`, the same
      handedness as the object rows); ``intensity`` (the row's own channel) plus one
      ``<channel>_intensity`` per channel on a multi-channel image; ``time_s`` when the file
      has a per-frame clock; opt-in ``edge_dist_um`` / ``centroid_dist_um``.
    * **Size** — counted BEFORE a pixel is read and refused past ``max_rows`` (0 = no limit)
      with the count and the ways to narrow it. Coordinates are int32, intensities and local
      µm float32 (exact for 16-bit counts), stage positions float64.
    * **Footprint** — WHOLE_VOLUME like the label branch, but only the planes holding a
      gathered voxel are read, each once per channel, fanned out over planes.

    Verified by: aggregating ``intensity`` by ``label_id`` reproduces the label branch's
    ``mean_intensity`` and ``area`` exactly (``selftest.test_measure_pixels``)."""
    ax = ds.axes
    modes = ctx.params.get("__modes__", {}) or {}
    volume = str(modes.get("pixels_of", "regions")) == "volume"
    out_layer = ctx.layer("pixels") or _PIXEL_LAYER
    existing = {k[1] for k in ds.attributes if k[0] is Domain.POINT and k[1]}
    if out_layer in existing and out_layer not in pixel_layers(ds.metadata):
        raise ValueError(
            f"measure: {out_layer!r} is already a Point table on the wire (detections, not "
            f"pixels) — writing the pixel table under that name would replace it. Choose "
            f"another name in `Pixel table`.")
    geometry = _pixel_geometry(ctx.params.get("geometry", ""))
    limit = int(ctx.params.get("max_rows", _PIXEL_MAX_ROWS) or 0)
    layer, raster, note = _pixel_raster(ctx, ds, volume=volume)
    if raster is not None:
        want_shape = (ax.m, ax.t, ax.z, ax.y, ax.x)
        got = raster.shape[:3] + raster.shape[4:]
        if got != want_shape or raster.shape[3] not in (1, ax.c):
            raise ValueError(
                f"measure: the raster {layer!r} is shaped {raster.shape} but the image is "
                f"(m,t,z,c,y,x) = {(ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)} — a raster from "
                f"another branch (cropped, projected, a different file) has no voxel-for-"
                f"voxel match here.")
        shape = raster.shape
    else:
        shape = (ax.m, ax.t, ax.z, 1, ax.y, ax.x)

    # ── which voxels: counted BEFORE anything is read ─────────────────────────────────
    ids = None if volume else _parse_ids(ctx.params.get("ids", ""))
    sel = None
    if volume:
        n = int(np.prod(shape))
    else:
        sel = raster != 0
        if ids is not None:
            sel &= np.isin(raster, ids)
        n = int(np.count_nonzero(sel))
        if n == 0 and ids is not None:
            have = np.unique(raster[raster != 0])
            span = (f"its {have.size} region(s) run {int(have.min())}…{int(have.max())}"
                    if have.size else "it holds no regions at all")
            raise ValueError(f"measure: none of the ids in `ids` is a region of {layer!r} — "
                             f"{span}. Hover a region in the Viewer to read its id.")
    if limit > 0 and n > limit:
        raise ValueError(
            f"measure: Members = pixel would write {n:,} rows, more than `max_rows` "
            f"({limit:,}). Narrow it — name a few regions in `ids`, crop or pick frames "
            f"upstream (Crop, Select Frame), use one channel — or raise `max_rows` "
            f"(0 = no limit; a row costs ~60 bytes in memory plus ~8 per channel).")
    if note:
        ctx.progress(0, 1, "using " + note)

    # ── coordinates, in raster C order — so each (m,t,z) plane is one contiguous block ──
    rows = {k: np.empty(n, dtype=np.int32) for k in ("m", "t", "z", "c", "y", "x")}
    label_id = np.zeros(n, dtype=np.int64)
    flat_lab = None if raster is None else raster.reshape(-1)
    step = 1 << 22
    if volume:
        for s in range(0, n, step):
            e = min(n, s + step)
            coords = np.unravel_index(np.arange(s, e, dtype=np.int64), shape)
            for k, v in zip(("m", "t", "z", "c", "y", "x"), coords):
                rows[k][s:e] = v
            if flat_lab is not None:
                label_id[s:e] = flat_lab[s:e]
    else:
        flat = np.flatnonzero(sel.reshape(-1))
        del sel
        for s in range(0, n, step):
            e = min(n, s + step)
            coords = np.unravel_index(flat[s:e], shape)
            for k, v in zip(("m", "t", "z", "c", "y", "x"), coords):
                rows[k][s:e] = v
            label_id[s:e] = flat_lab[flat[s:e]]
        del flat

    # ── intensities: each plane holding a gathered voxel read once per channel ──────────
    n_c = int(ax.c)
    chan = np.empty((n_c, n), dtype=np.float32)
    if n:
        plane = (rows["m"].astype(np.int64) * ax.t + rows["t"]) * ax.z + rows["z"]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(plane)) + 1))
        stops = np.concatenate((starts[1:], [n]))
    else:
        starts = stops = np.zeros(0, dtype=np.int64)
    tick = _parallel_progress(ctx, int(starts.size),
                              "sampling raw pixels" if on_raw else "sampling pixels",
                              frames=ax.t)

    def _read(j: int) -> None:
        s, e = int(starts[j]), int(stops[j])
        m, t, z = int(rows["m"][s]), int(rows["t"][s]), int(rows["z"][s])
        yy, xx = rows["y"][s:e], rows["x"][s:e]
        for ch in range(n_c):
            img = np.asarray(prov.get_region(0, m, t, z, ch, 0, ax.y, 0, ax.x))
            chan[ch, s:e] = img[yy, xx]
        tick()

    map_units(_read, list(range(int(starts.size))))
    own_c = np.minimum(rows["c"].astype(np.int64), n_c - 1)
    intensity = (chan[0].copy() if n_c == 1
                 else np.take_along_axis(chan, own_c[None, :], axis=0)[0])

    # ── positions ─────────────────────────────────────────────────────────────────────
    px = ctx.calib("pixel_size_um") or 0.1
    zs = (ctx.calib("z_step_um") or 0.5) if ax.z > 1 else 0.0
    x_um = rows["x"] * float(px)
    y_um = rows["y"] * float(px)
    z_um = rows["z"] * float(zs)
    cols: Dict[str, np.ndarray] = {"id": np.arange(1, n + 1, dtype=np.int64)}
    cols.update(rows)
    cols["label_id"] = label_id
    cols["x_um"] = x_um.astype(np.float32)
    cols["y_um"] = y_um.astype(np.float32)
    cols["z_um"] = z_um.astype(np.float32)
    cols.update(_stage_position_columns(ds, rows["m"], x_um, y_um,
                                        rows["z"].astype(float), flip=_flip(ctx, ds)))
    cols["intensity"] = intensity
    src_md = (ctx.input("raw").metadata if on_raw else ds.metadata) or {}
    for ch, name in enumerate(_channel_columns(src_md.get("channel_names") or (), n_c)):
        cols[name] = chan[ch]
    secs = frame_seconds(ds.metadata, np.arange(ax.t))
    if secs is not None:
        cols["time_s"] = secs[rows["t"]]

    # ── geometry (opt-in) ─────────────────────────────────────────────────────────────
    if geometry:
        zk = ds.structure_zkind(Domain.LABEL, layer) if layer else None
        is_3d = zk == "subpixel" and ax.z > 1
        if "edge_dist" in geometry:
            if raster is None:
                cols["edge_dist_um"] = np.full(n, np.nan, dtype=np.float32)
            else:
                spacing = ((float(zs) or 1.0, float(px), float(px)) if is_3d
                           else (float(px), float(px)))
                cols["edge_dist_um"] = _edge_distances(raster, rows, label_id, is_3d=is_3d,
                                                       spacing=spacing, ctx=ctx)
        if "centroid_dist" in geometry:
            cols["centroid_dist_um"] = _centroid_distances(rows, label_id,
                                                           (z_um, y_um, x_um), is_3d=is_3d)

    # ── write: replace any earlier pixel table of this name whole ─────────────────────
    out = ds
    for key in [k for k in ds.attributes if k[0] is Domain.POINT and k[1] == out_layer]:
        out = out.without(key[0], key[2], key[1])
    out = out.with_structure(StructureTable(Domain.POINT, cols, layer=out_layer,
                                            z_kind="plane_index"))
    ctx.progress(1, 1, f"{n:,} pixels → {out_layer!r}"
                       + (f" from {layer!r}" if layer else " (every voxel)"))
    return mark_pixel_table(out, out_layer)


def _compute_measure(ctx: EvalContext) -> Dataset:
    """Measure the image over the members of a structure table → columns on that table.

    Three branches, selected by the **``target``** Mode, because the member kinds are
    measurable in genuinely different ways rather than by the same code with a switch. The
    third, **``pixel``** (:func:`_measure_pixels`, 2026-11-02), is the one that does not
    write onto the table it read: a voxel is a member of a region, and the voxels of chosen
    regions (or of the whole volume) become a Point table of their own, one row each.

    The first two:

    * **``label``** (:func:`_measure_labels`) — a region has extent, so it has statistics:
      per-label ``mean``/``max``/``min``/``sum``/``median`` intensity and a voxel ``count``
      via the Voxel→Label bridge, plus optional µm-aware ``regionprops`` geometry.
    * **``point``** (:func:`_measure_points`) — a detection is dimensionless, so it has a
      *position* and one *sample*: ``x_um``/``y_um``/``z_um`` and the intensity of the voxel
      it sits on, via the Voxel→Point bridge's nearest-voxel rule.

    Both branches also report each row's ABSOLUTE stage position — ``stage_x_um``/
    ``stage_y_um``/``stage_z_um``, via :func:`_stage_position_columns` — so objects from
    different multipoints of the same acquisition land in one comparable physical frame in
    the spreadsheet, not just their own field's local coordinates.

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
    target = ctx.params.get("__modes__", {}).get("target", "label")
    if target == "point":
        return _measure_points(ctx, ds, prov, on_raw=on_raw)
    if target == "pixel":
        return _measure_pixels(ctx, ds, prov, on_raw=on_raw)
    return _measure_labels(ctx, ds, prov, on_raw=on_raw)
def _measure_points(ctx: EvalContext, ds: Dataset, prov, *, on_raw: bool) -> Dataset:
    """``target=point`` — each detection's **physical position** and the intensity of the
    **voxel it sits on**, as columns on its own Point table.

    A Point row is dimensionless: there is no region to reduce over, so `stats` and `shape`
    are `available_in`-gated away and this branch writes a fixed set of columns:

    ========================================  ==================================================
    ``x_um``/``y_um``                         the row's own ``x``/``y`` scaled by ``pixel_size_um``
    ``z_um``                                  ``z`` scaled by ``z_step_um`` (0 on a single-plane Dataset)
    ``mean_intensity``                        the value of the nearest voxel to ``(z, y, x)``
    ``stage_x_um``/``stage_y_um``/``stage_z_um``  the ABSOLUTE microscope position — ``x_um``/
                                               ``y_um``/``z_um`` plus this row's multipoint's
                                               stage position (:func:`_stage_position_columns`);
                                               NaN when the file carries no placement metadata
    ========================================  ==================================================

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
    out.update(_stage_position_columns(ds, mm, out["x_um"], out["y_um"], zf,
                                       flip=_flip(ctx, ds)))
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
    refused on a 3D Label table rather than failing inside skimage.

    **Position (unconditional, like ``mean``).** ``x_um``/``y_um``/``z_um`` — the region's
    own centroid (already computed by the label producer, looked up by id) scaled to
    physical units — and ``stage_x_um``/``stage_y_um``/``stage_z_um``, its ABSOLUTE
    microscope position (:func:`_label_position_columns`, :func:`_stage_position_columns`).
    NaN wherever the file carries no placement metadata, never a guessed 0."""
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
    # ── position (V2.29): local centroid + absolute stage position ───────────────────
    # Uses the SAME px/z_step_um the shape pass above would (a fresh ctx.calib read here
    # rather than sharing that block's locals, since `shape_names` may be empty and this
    # runs unconditionally).
    px = ctx.calib("pixel_size_um") or 0.1
    zs = (ctx.calib("z_step_um") or 0.5) if ax.z > 1 else 0.0
    cols.update(_label_position_columns(ds, layer, cols["id"], px=px, zs=zs,
                                        flip=_flip(ctx, ds)))
    return ds.with_structure(StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk))

def _columns_measure(params, modes, incoming, inputs=()):
    """The columns each branch writes, resolved through the SAME selectors the compute
    parses (V2.28) — so a stat the user has not ticked is not offered as a condition.

    ``_measure_stats``/``_measure_shape`` REFUSE an unknown name, which is right at pull
    time and wrong here: a half-typed selector is the normal state of a text box mid-edit,
    and raising would blank every node's envelope graph-wide. So both are called
    defensively and a selector that does not parse yet contributes nothing.

    ``mean`` is forced by the compute, so ``mean_intensity`` is declared unconditionally on
    the label branch — it stays a stable downstream key however the selector is edited.

    The **pixel** branch reads the input envelopes (``wants_inputs``): its per-channel
    columns are named from the measured image's ``channel_names`` — the `raw` input's when
    one is wired — and ``time_s`` exists only when the file has a per-frame clock."""
    try:
        if (modes or {}).get("target") == "pixel":
            envs = {sock: env for sock, env in (inputs or ()) if env is not None}
            src = envs.get("raw") or envs.get("data")
            data = envs.get("data")
            chans = ([] if src is None else _channel_columns(
                (getattr(src, "metadata", None) or {}).get("channel_names") or (),
                int(getattr(getattr(src, "axes", None), "c", 1) or 1)))
            try:
                geometry = _pixel_geometry((params or {}).get("geometry", "") or "")
            except Exception:
                geometry = []                # a selector mid-edit offers none
            timed = data is not None and envelope_has(data, "frame_time_jd")
            return on_layer(Domain.POINT, str((params or {}).get("pixels") or _PIXEL_LAYER),
                            _pixel_columns(chans, geometry, timed))
        if (modes or {}).get("target") == "point":
            layer = resolved_layer(params, "points", incoming, Domain.POINT)
            return on_layer(Domain.POINT, layer,
                            ("z_um", "y_um", "x_um", "mean_intensity",
                             "stage_x_um", "stage_y_um", "stage_z_um"))
        layer = str((params or {}).get("labels") or "labels")
        names = ["mean_intensity", "x_um", "y_um", "z_um",
                 "stage_x_um", "stage_y_um", "stage_z_um"]
        try:
            names += [_MEASURE_COLUMNS[s] for s in _measure_stats(params.get("stats", ""))]
        except Exception:
            pass                             # a selector mid-edit offers only the forced one
        try:
            names += list(_measure_shape(params.get("shape", "") or ""))
        except Exception:
            pass
        return on_layer(Domain.LABEL, layer, dict.fromkeys(names))
    except Exception:                        # pragma: no cover - defensive
        return ()


_columns_measure.wants_inputs = True

register_node(
    batch_aware(_compute_measure), op_key="analysis.measure", label="Measure", category="analysis",
    adds_columns=_columns_measure,
    extra_layers=_layers_measure,
    # VOXEL is unconditional because it is the IMAGE domain — both branches measure pixels,
    # and §4f is explicit that the image is not a per-branch layer requirement. What DOES
    # vary is the structure: the label branch needs a Label table (its raster's ids must
    # divide the foreground into objects), the point branch a Point table. Declaring the
    # static union `{VOXEL, LABEL, POINT}` would paint a red PT chip on every Label graph
    # that works — the exact over-claim `analysis.object_field` was fixed for in V2.22.
    reads_domains=frozenset({Domain.VOXEL}),
    # `pixel` reads only VOXEL: any raster can name its regions (a mask, a drawn region),
    # and Pixels of = volume needs no raster at all — a Label table is welcome, not required.
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.VOXEL, Domain.LABEL}),
        "point": frozenset({Domain.POINT}),
        "pixel": frozenset({Domain.VOXEL}),
    }},
    adds_domains=frozenset({Domain.LABEL, Domain.POINT}),
    modes=[Mode("target", ["label", "point", "pixel"], default="label", label="Members",
                description=
                "Which members are measured — the REGIONS of a Label table, the "
                "DETECTIONS of a Point table, or the single PIXELS (voxels) of chosen "
                "regions. It selects the live source socket and, because the member kinds "
                "are measurable in different ways, which columns come out: a region has "
                "extent, so it has statistics and geometry; a point is dimensionless, so it "
                "has a position and one sample; a pixel has its own address and value. "
                "Regions and points write onto the table they read; pixels make a new "
                "table. All three honour the `raw` input.",
                choice_docs={
                    "label":
                        "Measure the regions of a Label table (Segmentation / Connected "
                        "Components): per-region intensity statistics over every voxel the "
                        "region owns, plus optional µm regionprops geometry, plus each "
                        "region's `x_um`/`y_um`/`z_um` centroid and absolute "
                        "`stage_x_um`/`stage_y_um`/`stage_z_um` position. This is the "
                        "branch `stats` and `shape` belong to; it needs a raster whose ids "
                        "divide the foreground into objects, not a bare mask.",
                    "point":
                        "Measure the detections of a Point table (Spot / Particle Detection, "
                        "Label to Points): writes `x_um`/`y_um`/`z_um` — the physical "
                        "position, which the invariant pixel `x`/`y`/`z` columns are not — "
                        "`mean_intensity`, the value of the single voxel each detection "
                        "sits on, and absolute `stage_x_um`/`stage_y_um`/`stage_z_um`. "
                        "`stats` and `shape` are hidden here: there is no region to "
                        "reduce over or fit an ellipse to.",
                    "pixel":
                        "Write every PIXEL (voxel) of chosen regions — or of the whole "
                        "volume — as a row of a new Point table (`Pixel table`): its "
                        "`m t c z y x` address, `label_id`, µm and absolute stage position, "
                        "`intensity` plus one column per channel, `time_s` when the file "
                        "has a clock, and optional edge / centroid distances. For the "
                        "distribution INSIDE an object (rim vs core, colocalization, a "
                        "histogram), which the per-region stats summarise away. The "
                        "spreadsheet shows it as a strip of pixels; it can be millions of "
                        "rows, so `max_rows` caps it.",
                }),
           Mode("pixels_of", ["regions", "volume"], default="regions", label="Pixels of",
                available_in={"target": frozenset({"pixel"})},
                description="Which voxels Members = pixel writes: those inside regions, or "
                            "every voxel of the volume.",
                choice_docs={
                    "regions":
                        "The voxels inside the regions of the raster `Label layer` names — "
                        "a segmentation's labels, a threshold mask, a drawn region — all of "
                        "them, or only the region ids typed in `ids`. Background voxels are "
                        "left out, so the row count is the regions' total area.",
                    "volume":
                        "Every voxel of the volume, background included — m × t × z × y × x "
                        "rows (times the raster's channels). `label_id` still names the "
                        "region a voxel is in when a raster is named (0 = background); "
                        "with none, it is 0 everywhere. Use it on a cropped or one-frame "
                        "stream: a whole series is easily past `max_rows`.",
                })],
    inputs=[InDataset(), _InRaw(),
            InBool("flip_x", "Flip X", field=False, default=True,
                   description="Whether the image's +x runs along stage −x when the "
                               "`stage_x_um` column places each row on the microscope. ON "
                               "(the default) is this scope's handedness and Stitch's, "
                               "Overlay's and Merge's default — so one object seen from two "
                               "overlapping fields gets ONE stage position. It applies to a "
                               "stage-Stitched mosaic too, which keeps the camera's "
                               "handedness. Changes only `stage_x_um`; every local and "
                               "intensity column is unaffected."),
            InBool("flip_y", "Flip Y", field=False, default=False,
                   description="Whether the image's +y runs along stage −y for `stage_y_um`. "
                               "OFF (the default) matches Stitch and Overlay. Changes only "
                               "`stage_y_um`."),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     available_in={"target": frozenset({"label", "pixel"})},
                     description=
                     "Which label raster defines the regions to measure — the output of "
                     "Connected Components or Segmentation. One row is produced per label "
                     "id, so this choice fixes both WHAT gets measured and how many rows "
                     "come out. With Members = pixel it may also name a plain mask or a "
                     "drawn region (its voxels' value becomes `label_id`); with Pixels of = "
                     "volume it only fills `label_id` and may name nothing. It always comes "
                     "from the MAIN input, even when a `raw` Dataset is wired: `raw` "
                     "redirects only the pixels being measured."),
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
                     "voxel count, NOT a physical area), `std`→std_intensity. `mean` is "
                     "always computed even if "
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
                         "std":
                             "→ `std_intensity`: the sample standard deviation (n − 1) of "
                             "the region's voxel values, NaN for a one-voxel region — the "
                             "spread WITHIN an object. With `count` it lets Plot XY pool "
                             "every voxel of every object (Error = pooled SD) without a "
                             "row per voxel.",
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
                     }),
            InString("pixels", "Pixel table", field=False, default=_PIXEL_LAYER,
                     layer_out=(Domain.POINT,),
                     available_in={"target": frozenset({"pixel"})},
                     description=
                     "Name of the Point table the pixels are written to — a new table beside "
                     "the regions', which are left as they were. Re-running replaces an "
                     "earlier pixel table of this name whole; a name that is already a "
                     "detection table (spots, particles) is refused rather than overwritten."),
            InString("ids", "ids", field=False, default="",
                     available_in={"target": frozenset({"pixel"}),
                                   "pixels_of": frozenset({"regions"})},
                     description=
                     "Which regions' pixels to write, by label id — `12, 40, 100-120`. Blank "
                     "writes every region's. Fewer ids is the main way to keep a pixel table "
                     "small; an id that is not a region is skipped, and a list matching none "
                     "is refused with the ids the raster does hold. Hover a region in the "
                     "Viewer to read its id."),
            InString("geometry", "Pixel geometry", field=False, default="",
                     vocab=tuple(_PIXEL_GEOMETRY),
                     available_in={"target": frozenset({"pixel"})},
                     description=
                     "Per-pixel distance columns, comma-separated — empty (the default) adds "
                     "none. Both are in µm (pixel_size_um, z_step_um for a 3D Label table), "
                     "measured within the region's own geometry (per plane, or the volume "
                     "for a 3D table), and NaN on a background pixel. Each is an extra pass "
                     "over the gathered pixels.",
                     choice_docs={
                         "edge_dist":
                             "→ `edge_dist_um`: distance to the nearest pixel OUTSIDE the "
                             "region — one pixel on the rim, largest at the core. Touching "
                             "regions count as outside each other, and so does the image "
                             "edge. The axis for rim-versus-core profiles (membrane vs "
                             "cytoplasm). Needs a raster; NaN with Pixels of = volume and "
                             "none named.",
                         "centroid_dist":
                             "→ `centroid_dist_um`: distance from the region's centroid "
                             "(the mean position of its pixels). The axis for radial "
                             "profiles; unlike `edge_dist` it keeps rising past the core of "
                             "an elongated cell. On a plain mask every pixel shares one id, "
                             "so the centroid is the whole foreground's in that plane.",
                     }),
            InInt("max_rows", "Max rows", unit="", field=False, default=_PIXEL_MAX_ROWS,
                  available_in={"target": frozenset({"pixel"})},
                  description=
                  "The most rows the pixel table may have. Counted before a pixel is read: "
                  "past it the node stops and says how many rows it would have written. "
                  "0 = no limit. A row costs ~60 bytes in memory plus ~8 per channel, so "
                  "the default (5 million) is about 0.3-0.5 GB. It never thins the table — "
                  "every chosen pixel is written, or none."),
    ],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_VOLUME,
    description="Measures the image over a structure table's members. On LABEL members: "
                "per-region statistics (mean/max/min/area) via the Voxel→Label bridge, plus "
                "optional µm-aware `shape` geometry columns (eccentricity, perimeter, "
                "solidity, …) from regionprops. On POINT members: each detection's physical "
                "position (`x_um`/`y_um`/`z_um`) and `mean_intensity`, the value of the "
                "voxel it sits on. On PIXEL members: every voxel of chosen regions (or of "
                "the volume) as a row of a new Point table — address, `label_id`, µm and "
                "stage position, per-channel intensity, optional edge/centroid distance. "
                "Every branch also reports each row's absolute "
                "`stage_x_um`/`stage_y_um`/`stage_z_um` microscope position, for comparing "
                "objects across a multi-position acquisition (NaN when the file carries no "
                "placement metadata). Optional `raw` input measures THOSE pixels instead "
                "(segment on enhanced, measure on raw); the structure, the labels and the "
                "shape metrics always come from the main input.",
)
