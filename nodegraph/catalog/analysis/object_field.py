"""Object Field (``analysis.object_field``) — Gridded Eulerian fields from per-object measurements — density (objects/µm²), mean area, intensity, fold change, velocity/speed (µm/s), divergence and curl (1/s) — as a Point layer on a grid_step…"""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.objects import (
    _InFrameInterval,
    _frame_interval_s,
    _object_table,
    _object_velocity,
)
from nodegraph.catalog._shared.units import to_pixels_v2

# ── gridded Eulerian object fields (CT compute_spatial_fields) ──────────────────

#: ``fields`` → the grid column(s) each entry writes, and what it needs.
_OBJECT_FIELDS: Dict[str, Tuple[str, ...]] = {
    "density": ("density",),
    "mean_area": ("mean_area",),
    "intensity": ("intensity",),
    "fold_change": ("fold_change",),
    "velocity": ("velocity_y", "velocity_x"),
    "speed": ("speed",),
    "divergence": ("divergence",),
    "curl": ("curl",),
}
_OBJECT_FIELDS_TRACKED = frozenset({"velocity", "speed", "divergence", "curl"})
_OBJECT_FIELDS_INTENSITY = frozenset({"intensity", "fold_change"})
def _object_field_names(raw) -> list:
    """The ``fields`` selector → an ordered, de-duplicated list of field names."""
    names = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in (raw or ())])
    names = [s for s in names if s] or ["density", "speed", "divergence"]
    bad = [s for s in names if s not in _OBJECT_FIELDS]
    if bad:
        raise ValueError(f"unknown object field(s) {bad} — choose from "
                         f"{list(_OBJECT_FIELDS)} (comma-separated)")
    return list(dict.fromkeys(names))
def _scatter_to_grid(points: np.ndarray, values: np.ndarray, gy: np.ndarray,
                     gx: np.ndarray, sigma_grid: float) -> np.ndarray:
    """Interpolate scattered per-object ``values`` onto the grid, then smooth — Cell-
    Tracker's ``_interp``: ``griddata`` linear, out-of-hull filled with the sample mean,
    then a Gaussian blur in GRID units.

    Fewer than 4 samples cannot define a 2-D linear interpolant, and a degenerate hull
    (collinear centroids) makes Qhull raise, so both fall back to a constant field at the
    sample mean rather than failing the pull — an all-NaN or an exception here would take
    down a frame that simply had too few objects."""
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    shape = (len(gy), len(gx))
    if finite.sum() < 4:
        fill = float(np.nanmean(vals)) if finite.any() else np.nan
        return np.full(shape, fill, dtype=float)
    mean = float(np.nanmean(vals[finite]))
    try:
        field = griddata(points[finite], vals[finite], (gy[:, None], gx[None, :]),
                         method="linear")
        field = np.nan_to_num(np.asarray(field, dtype=float), nan=mean)
    except Exception:                    # noqa: BLE001 — degenerate hull → constant fill
        return np.full(shape, mean, dtype=float)
    return gaussian_filter(field, sigma=sigma_grid)
def _compute_object_field(ctx: EvalContext) -> Dataset:
    """Gridded **Eulerian** fields from per-object measurements — Cell-Tracker's
    ``backend/fields.py::compute_spatial_fields``.

    Where ``analysis.object_metrics`` answers "what is this cell doing", this answers "what
    is happening HERE": every requested quantity is interpolated from the object centroids
    onto a regular ``grid_step`` lattice and smoothed, one lattice per ``(m, t, z, c)``.

    The output is a **Point layer on the grid nodes**, not a Voxel raster, which is the
    same shape ``analysis.dvc_field`` emits — so ``transform.rasterize_field`` renders any
    column of it to full resolution, and it inherits that node's ``z_kind`` routing for
    free (``plane_index`` ⇒ per-plane 2-D interpolation, §7b). Building the coarse field
    and the rendering as two nodes rather than one keeps the grid inspectable, and means a
    field can be exported as numbers without ever being rasterized.

    ==================  ==================================  ==============
    ``fields`` entry    column(s)                           unit
    ==================  ==================================  ==============
    ``density``         ``density``                         objects/µm²
    ``mean_area``       ``mean_area``                       µm²
    ``intensity``       ``intensity``                       image units
    ``fold_change``     ``fold_change``                     —
    ``velocity``        ``velocity_y``, ``velocity_x``      µm/s
    ``speed``           ``speed``                           µm/s
    ``divergence``      ``divergence``                      1/s
    ``curl``            ``curl``                            1/s
    ==================  ==================================  ==============

    ``density`` counts objects per grid cell, smooths, and divides by the cell's physical
    area, so it is objects/µm² rather than Cell-Tracker's raw smoothed count — a count per
    cell is not comparable between two runs with different ``grid_step``. ``mean_area``
    converts the Label table's voxel ``area`` with ``pixel_size_um²``. ``divergence`` and
    ``curl`` are finite differences of the gridded velocity taken with the **physical**
    grid spacing, hence 1/s: divergence positive where objects spread apart, curl the
    local rotation of the flow.

    Velocity comes from :func:`_object_velocity` — the same gap-aware definition
    ``analysis.object_metrics`` uses, so the two nodes cannot disagree. (Cell-Tracker's own
    field code differenced strictly adjacent frames via a merge on ``track_id``, which
    drops every gap-filled step.)

    Prerequisites are refused, not skipped: the four velocity fields need a ``track_id``
    column, ``mean_area`` needs ``area``, and ``intensity``/``fold_change`` need the named
    intensity column. 2D only, for the reason in :func:`_object_table`."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("object field needs an image provider to define the grid extent")
    ax = prov.axes
    domain, layer, cols, zk = _object_table(ctx, ds, node="object field")
    names = _object_field_names(ctx.params.get("fields", ""))
    px = ctx.calib("pixel_size_um") or 0.1
    need_track = bool(set(names) & _OBJECT_FIELDS_TRACKED)
    dt_s = _frame_interval_s(ctx, needed=need_track,
                             wanted=set(names) & _OBJECT_FIELDS_TRACKED,
                             node="object field")
    step_um = float(ctx.params.get("grid_step", 10.0))
    step_px = max(1, int(round(to_pixels_v2(step_um, "um", pixel_size_um=px))))
    smooth_um = max(0.0, float(ctx.params.get("smooth", 20.0)))
    # σ is expressed in GRID units, as Cell-Tracker's is — one grid cell spans step_px
    # pixels, i.e. step_px·px microns.
    sigma_grid = smooth_um / max(1e-9, step_px * px)
    out_layer = ctx.layer("name")

    track = cols.get("track_id")
    if need_track and track is None:
        raise ValueError(
            f"object field: {sorted(set(names) & _OBJECT_FIELDS_TRACKED)} are built from "
            f"object motion and layer {layer!r} carries no 'track_id' column. Run "
            f"track.objects (or track.link) upstream, or ask only for density / mean_area "
            f"/ intensity / fold_change.")
    if "mean_area" in names and "area" not in cols:
        raise ValueError(
            f"object field: 'mean_area' needs the 'area' column and layer {layer!r} has "
            f"none (Point tables never carry one). Track a Label layer, or drop "
            f"'mean_area'.")
    inten_name = ctx.params.get("intensity", "mean_intensity")
    if set(names) & _OBJECT_FIELDS_INTENSITY and inten_name not in cols:
        raise ValueError(
            f"object field: {sorted(set(names) & _OBJECT_FIELDS_INTENSITY)} need an "
            f"intensity column, but layer {layer!r} has no {inten_name!r} (it carries "
            f"{sorted(cols)}). Run analysis.measure upstream, or point `intensity` at an "
            f"existing column.")

    tt = np.rint(np.asarray(cols["t"], dtype=float)).astype(np.int64)
    zz = np.rint(np.asarray(cols["z"], dtype=float)).astype(np.int64)
    mm = np.asarray(cols["m"], dtype=np.int64)
    cc = np.asarray(cols["c"], dtype=np.int64)
    y_px = np.asarray(cols["y"], dtype=float)
    x_px = np.asarray(cols["x"], dtype=float)
    vy = vx = None
    if need_track:
        vy, vx = _object_velocity(np.asarray(track, dtype=np.int64), tt,
                                  y_px * px, x_px * px, dt_s)

    gy_px = np.arange(0, ax.y, step_px, dtype=float)
    gx_px = np.arange(0, ax.x, step_px, dtype=float)
    n_nodes = len(gy_px) * len(gx_px)
    cell_um2 = (step_px * px) ** 2
    wanted = [n for f in names for n in _OBJECT_FIELDS[f]]
    acc: Dict[str, list] = {k: [] for k in ("m", "t", "c", "z", "y", "x")}
    acc.update({name: [] for name in wanted})

    units = [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
             for z in range(ax.z) for c in range(ax.c)]
    ctx.progress(0, max(1, len(units)), "gridding fields", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        sel = np.flatnonzero((mm == m) & (tt == t) & (zz == z) & (cc == c))
        if sel.size == 0:
            ctx.progress(i + 1, len(units), "gridding fields", frames=ax.t)
            continue
        pts = np.column_stack([y_px[sel], x_px[sel]])
        grids: Dict[str, np.ndarray] = {}
        if "density" in names:
            # bin counts, smoothed in grid units, then per physical cell area so the
            # number is comparable across grid_step choices
            from scipy.ndimage import gaussian_filter
            counts = np.zeros((len(gy_px), len(gx_px)), dtype=float)
            iy = np.clip((y_px[sel] / step_px).astype(int), 0, len(gy_px) - 1)
            ix = np.clip((x_px[sel] / step_px).astype(int), 0, len(gx_px) - 1)
            np.add.at(counts, (iy, ix), 1.0)
            grids["density"] = gaussian_filter(counts, sigma=sigma_grid) / cell_um2
        if "mean_area" in names:
            grids["mean_area"] = _scatter_to_grid(
                pts, np.asarray(cols["area"], dtype=float)[sel] * px * px,
                gy_px, gx_px, sigma_grid)
        if set(names) & _OBJECT_FIELDS_INTENSITY:
            vals = np.asarray(cols[inten_name], dtype=float)[sel]
            if "intensity" in names:
                grids["intensity"] = _scatter_to_grid(pts, vals, gy_px, gx_px, sigma_grid)
            if "fold_change" in names:
                mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                grids["fold_change"] = _scatter_to_grid(
                    pts, vals / mean if mean and np.isfinite(mean) and mean > 0
                    else np.full(len(vals), np.nan), gy_px, gx_px, sigma_grid)
        if need_track:
            # NaN velocities (a track's first detection) are dropped by _scatter_to_grid's
            # finite mask rather than counted as zero motion.
            gvy = _scatter_to_grid(pts, vy[sel], gy_px, gx_px, sigma_grid)
            gvx = _scatter_to_grid(pts, vx[sel], gy_px, gx_px, sigma_grid)
            if "velocity" in names:
                grids["velocity_y"], grids["velocity_x"] = gvy, gvx
            if "speed" in names:
                grids["speed"] = np.hypot(gvy, gvx)
            if {"divergence", "curl"} & set(names):
                spacing = step_px * px                    # µm between grid nodes
                if "divergence" in names:
                    grids["divergence"] = (np.gradient(gvx, spacing, axis=1)
                                           + np.gradient(gvy, spacing, axis=0))
                if "curl" in names:
                    grids["curl"] = (np.gradient(gvx, spacing, axis=0)
                                     - np.gradient(gvy, spacing, axis=1))
        node_y = np.repeat(gy_px, len(gx_px))
        node_x = np.tile(gx_px, len(gy_px))
        acc["m"].append(np.full(n_nodes, m, dtype=np.int64))
        acc["t"].append(np.full(n_nodes, t, dtype=np.int64))
        acc["c"].append(np.full(n_nodes, c, dtype=np.int64))
        acc["z"].append(np.full(n_nodes, float(z)))
        acc["y"].append(node_y)
        acc["x"].append(node_x)
        for name in wanted:
            acc[name].append(np.asarray(grids[name], dtype=float).ravel())
        ctx.progress(i + 1, len(units), "gridding fields", frames=ax.t)

    if not acc["y"]:
        raise ValueError(
            f"object field: layer {layer!r} has no rows inside this Dataset's axes, so "
            f"there is nothing to grid. Check that the structure and the image come from "
            f"the same chain (a crop or channel tap between them would desynchronize "
            f"them).")
    columns = {k: np.concatenate(v) for k, v in acc.items()}
    columns["id"] = np.arange(len(columns["y"]), dtype=np.int64)   # global-unique
    table = StructureTable(Domain.POINT, columns, layer=out_layer,
                           z_kind="plane_index")
    # Provenance (§7b), the `analysis.segment` shape: namespaced non-calibration keys. The
    # step is not recoverable from the grid alone once a reader has only one row, and
    # divergence/curl cannot be interpreted without it, so it is recorded rather than
    # left to be re-derived by differencing coordinates.
    return ds.with_structure(table).with_metadata(
        object_field_source=layer, object_field_step_um=float(step_px * px))
register_node(
    _compute_object_field, op_key="analysis.object_field", label="Object Field",
    category="analysis",
    # ONE of the two, never both — `_object_table` reads the Label table under
    # `target=label` and the Point table under `target=point`. The shipped static union
    # was the opposite failure to the empty declaration elsewhere: it demanded Points of a
    # pure-Label graph and painted a red chip on a pipeline that was fine (V2.22). VOXEL
    # rides along on the label branch because the `labels` socket is `layer_in=VOXEL` —
    # what it names is a Label INSTANCE, and the picker has no names to offer without it.
    reads_domains=frozenset(),
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.LABEL, Domain.VOXEL}),
        "point": frozenset({Domain.POINT}),
    }},
    adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 available_in={"target": frozenset({"label"})},
                 description=
                 "Which Label table supplies the objects the field is built from — after "
                 "Measure and Tracking, so their area, intensity and track_id columns are "
                 "available. The objects' centroids are the interpolation samples; the "
                 "image on the main input only supplies the extent the grid covers."),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})},
                 description=
                 "Which Point table supplies the objects — Spot or Particle Detection. "
                 "Point tables carry no `area`, so `mean_area` is refused for them; the "
                 "other fields work the same as for labels."),
        InString("fields", "Fields", field=False, default="density,speed,divergence",
                 vocab=tuple(_OBJECT_FIELDS),
                 description=
                 "Which fields to grid, comma-separated, each one column on the output "
                 "grid: `density` (objects/um2), `mean_area` (um2), `intensity` (image "
                 "units), `fold_change` (intensity over the frame mean), `velocity` "
                 "(velocity_y + velocity_x, um/s), `speed` (um/s), `divergence` (1/s, "
                 "positive where objects spread apart) and `curl` (1/s, local rotation). "
                 "The four motion fields need a track_id column, `mean_area` needs area, "
                 "and the two intensity fields need the column named below. Divergence and "
                 "curl are differences of the gridded velocity, so they inherit its "
                 "smoothing: they are noisy at small Smoothing and over-flattened at large.",
                 choice_docs={
                     "density":
                         "→ `density` in objects/µm²: how many objects fall near each grid "
                         "node. The only field that needs nothing but positions, and the one "
                         "to check first — a density map shows immediately whether the grid "
                         "step is finer than your object spacing.",
                     "mean_area":
                         "→ `mean_area` in µm²: average size of the objects near each node. "
                         "Maps where cells are large or small (spread versus rounded, swollen "
                         "versus compact). Needs an `area` column, so it is refused for Point "
                         "members, which have no extent.",
                     "intensity":
                         "→ `intensity` in image units: the objects' measured intensity, "
                         "averaged per node. Absolute, so it reflects illumination and "
                         "bleaching as well as biology — use `fold_change` when frames must be "
                         "comparable. Needs the intensity column named below.",
                     "fold_change":
                         "→ `fold_change`: the same intensity divided by the frame's mean, so "
                         "each frame is self-normalized and a bleaching series stays "
                         "comparable. The field to use for \"where is expression elevated\" "
                         "rather than \"where is it bright\". Needs the intensity column.",
                     "velocity":
                         "→ `velocity_y`,`velocity_x` in µm/s: the signed flow at each node — "
                         "the vector field, which the Overlay can draw as arrows. Direction "
                         "survives averaging here, so opposing motion cancels to zero flow. "
                         "Needs a track_id column.",
                     "speed":
                         "→ `speed` in µm/s: magnitude of that flow, always positive. Shows "
                         "where motion is happening regardless of direction, and unlike "
                         "`velocity` it does NOT cancel where objects move against each other. "
                         "Needs a track_id column.",
                     "divergence":
                         "→ `divergence` in 1/s: spatial derivative of the gridded velocity — "
                         "positive where the tissue is spreading apart, negative where it is "
                         "converging (a closing wound, a packing front). A difference of "
                         "smoothed values, so it is noisy at small Smoothing and flattened at "
                         "large. Needs tracking.",
                     "curl":
                         "→ `curl` in 1/s: the rotational part of the same gridded velocity, "
                         "signed by handedness — vortices and swirling collective migration. "
                         "Inherits the same smoothing sensitivity as divergence, and needs "
                         "tracking.",
                 }),
        InFloat("grid_step", "Grid step", unit="um", field=False, default=10.0,
                pick_kind="grid",
                description=
                "Spacing of the output lattice, in microns — it decides the RESOLUTION of "
                "the field and how many objects fall in each cell. It should be at least a "
                "cell diameter: finer than the objects themselves and most cells are empty, "
                "so the interpolation is guessing between isolated samples; much coarser "
                "and genuine spatial structure is averaged away. Output row count grows as "
                "the inverse square of this, so halving it quadruples the table. "
                "Cell-Tracker's default was 20 PIXELS, which is only comparable after "
                "multiplying by that dataset's pixel size."),
        InFloat("smooth", "Smoothing", unit="um", field=False, default=20.0,
                pick_kind="radius",
                description=
                "Gaussian smoothing applied to every gridded field, in microns — converted "
                "internally to the grid-cell units the filter works in, so changing Grid "
                "step does not change how much real-world smoothing you get (in "
                "Cell-Tracker the equivalent was in grid cells and did). It is doing real "
                "work, not cosmetics: interpolating a handful of scattered cells leaves a "
                "spiky field, and divergence and curl differentiate it, which amplifies "
                "every spike. Roughly a couple of grid cells is the usual choice. 0 "
                "disables it."),
        InString("intensity", "Intensity column", field=False, default="mean_intensity",
                 description=
                 "Which measured column feeds `intensity` and `fold_change` — "
                 "`mean_intensity` is what analysis.measure writes for its `mean` stat. "
                 "Point it at another measured column to grid that instead. Not read unless "
                 "one of those two fields is requested."),
        InString("name", "Output layer", field=False, default="object_field",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point layer this node writes: one row per grid node per "
                 "(multipoint, frame, z, channel), with the requested fields as columns. "
                 "Wire it into transform.rasterize_field (as `source`) to render any column "
                 "as a full-resolution image layer, or read it straight out of the "
                 "spreadsheet as numbers."),
        *_InFrameInterval(),
    ],
    outputs=[OutDataset()],
    modes=[Mode("target", ["label", "point"], default="label", label="Members",
                description=
                "Which objects the fields are gridded FROM — the regions of a Label table or "
                "the detections of a Point table. It selects the live source socket and, "
                "because the two carry different columns, which fields can be computed. The "
                "output is a Point layer on the grid either way.",
                choice_docs={
                    "label":
                        "Grid from the regions of a Label table (Segmentation / Connected "
                        "Components), using their centroids as positions. Its `area` and "
                        "measured intensity columns make `mean_area` and the two intensity "
                        "fields available.",
                    "point":
                        "Grid from the detections of a Point table (Spot / Particle "
                        "Detection). Density and the motion fields behave identically, but "
                        "Point rows have no `area`, so `mean_area` is refused rather than "
                        "silently returning zeros.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Gridded Eulerian fields from per-object measurements — density (objects/"
                "µm²), mean area, intensity, fold change, velocity/speed (µm/s), "
                "divergence and curl (1/s) — as a Point layer on a grid_step lattice, "
                "ready for transform.rasterize_field. Ports Cell-Tracker's "
                "compute_spatial_fields; 2D, per (m,t,z,c).")
