"""analysis.gradient_map — a field interpolated from measured objects around a point source: the
exponential decay their values follow with distance, carried between and beyond them, drawn on
an image's own grid (2026-10-15, IL-8 capture beads around a secreting spheroid)."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import nodegraph.catalog._shared.figure as FIG
import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import BATCH_COLUMN, Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.metadata import PER_TIME_KEYS, SOURCE_CLOCK_KEY
from nodegraph.registry import (Granularity, InBool, InDataset, InFloat, InString, Mode,
                                OutDataset)

_GM_DOMAINS = ("label", "point")
_GM_METHODS = ("trend_idw", "trend", "idw", "point_source_idw", "point_source")
_TREND_METHODS = frozenset({"trend_idw", "trend", "point_source_idw", "point_source"})
#: the trend methods that also interpolate the objects' residuals
_IDW_TREND = frozenset({"trend_idw", "point_source_idw"})
#: the trend methods whose shape is a Point-Source Fit's (2026-10-16)
_PS_METHODS = frozenset({"point_source_idw", "point_source"})
_DEFAULT_NAME = "gradient"
#: the second table: the measured objects placed on this image (Plot Field's marks)
OBJECTS_SUFFIX = "_objects"
#: per-frame clock keys a map of GROUPS no longer has (its frames are not instants): every
#: per-T time list the metadata module knows (the clock, its text, ``frame_time_s``), a
#: bundle's per-file clocks, the old timestamps list and ``dt_s``
_CLOCK_KEYS = (*PER_TIME_KEYS, SOURCE_CLOCK_KEY, "frame_timestamps_s", "dt_s")
_COORDS = ("id", "m", "t", "c", "z", "y", "x")
_LATTICE_COLUMNS = ("stage_x_um", "stage_y_um", "distance_um", "trend", "nearest_um")
_OBJECT_COLUMNS = ("stage_x_um", "stage_y_um", "stage_z_um", "distance_um", "trend",
                   "log_residual", "trend_a", "trend_lambda_um", "trend_r2")


# ── geometry: stage µm ↔ this image's pixels ───────────────────────────────────────────────

def image_geometry(md: Mapping[str, Any], axes: Any, flip: Tuple[bool, bool]):
    """``(box, px, reverse_x, reverse_y)`` of a single-field image: its stage box
    (:func:`nodegraph.placement.field_box`), its pixel size, and which way its columns / rows
    run over the stage — an Experiment Canvas's own orientation (:func:`canvas_mapping`), else
    the camera handedness ``flip``, which a stage-Stitched mosaic keeps too (Stitch puts the
    highest-x tile on the left; Measure reads a mosaic the same way, 2026-10-16). ``None``
    when the image cannot be placed."""
    from nodegraph.catalog._shared.canvas import is_canvas
    from nodegraph.placement import canvas_mapping, field_box
    box = field_box(md, axes, 0)
    try:
        px = float(md.get("pixel_size_um") or 0.0)
    except (TypeError, ValueError):
        px = 0.0
    if box is None or not px > 0:
        return None
    if is_canvas(md):
        rx, ry = canvas_mapping(md)[:2]
    else:
        rx, ry = bool(flip[0]), bool(flip[1])
    return box, px, rx, ry


def stage_to_image(sx, sy, geo) -> Tuple[np.ndarray, np.ndarray]:
    """Stage µm → this image's continuous pixel coordinates ``(u, v)`` (column, row): pixel
    ``i`` spans ``[i, i + 1]``, the convention the overlay compositor samples with
    (:func:`nodegraph.placement.axis_map`) and a figure draws the image in."""
    box, px, rx, ry = geo
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    u = (box.x1 - sx) / px if rx else (sx - box.x0) / px
    v = (box.y1 - sy) / px if ry else (sy - box.y0) / px
    return u, v


def image_to_stage(u, v, geo) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of :func:`stage_to_image`."""
    box, px, rx, ry = geo
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    sx = box.x1 - u * px if rx else box.x0 + u * px
    sy = box.y1 - v * px if ry else box.y0 + v * px
    return sx, sy


# ── the maths: trend × inverse-distance-weighted residuals ────────────────────────────────

def idw(nodes: np.ndarray, pts: np.ndarray, vals: np.ndarray, *, power: float,
        prior_dist: float = 0.0, chunk: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weighting of ``vals`` at ``pts`` onto ``nodes`` (both ``(n, 2)`` or
    ``(n, 3)`` µm — the same dimensionality):
    ``Σ w·v / (Σ w + w0)`` with ``w = 1/d^power`` — exact at a point (a node ON a point takes
    its value) — and, when ``prior_dist`` > 0, a phantom point of value 0 at that distance
    from every node (``w0 = 1/prior_dist^power``), so the result fades toward 0 where every
    point is farther than about ``prior_dist``. Returns ``(values, nearest distance)``."""
    out = np.empty(len(nodes))
    near = np.empty(len(nodes))
    w0 = (1.0 / prior_dist ** power) if prior_dist > 0 else 0.0
    for a in range(0, len(nodes), chunk):
        nb = nodes[a:a + chunk]
        d = np.sqrt(((nb[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        near[a:a + chunk] = d.min(axis=1)
        hit = d < 1e-9
        with np.errstate(divide="ignore"):
            w = np.where(hit, 0.0, 1.0 / np.maximum(d, 1e-300) ** power)
        res = (w @ vals) / (w.sum(axis=1) + w0)
        on = hit.any(axis=1)
        if on.any():
            res[on] = vals[np.argmax(hit[on], axis=1)]
        out[a:a + chunk] = res
    return out, near


def point_source_trend(obj_d: np.ndarray, vals: np.ndarray, use: np.ndarray, profile,
                       curve: Optional[Mapping[str, Any]], weight: str
                       ) -> Optional[Dict[str, Any]]:
    """A Point-Source Fit's profile as a frame's trend (2026-10-16): ``profile(d)`` the
    fitted concentration at distance ``d`` and this frame's time, ``curve`` the standard
    curve back to the signal (``None`` when the values ARE the concentration). Only its
    AMPLITUDE is refitted, to the objects in ``use`` — so the map's beads (a height-limited
    subset, say) set the level and the fit over every bead sets the shape. Returns ``{"fn":
    d → trend, "scale", "r2", "n"}``, or ``None`` with no usable object."""
    import nodegraph.catalog._shared.point_source as PS
    base = np.asarray(profile(obj_d), dtype=float)
    ok = use & np.isfinite(base) & (base > 0) & np.isfinite(vals)
    if weight == "relative":
        ok &= vals > 0
    if not ok.any():
        return None
    v, b = vals[ok], base[ok]

    def model(s, bb):
        c = s * bb
        return PS.to_signal(c, curve) if curve else c

    if curve is None and weight == "relative":
        scale = float(np.exp(np.mean(np.log(v) - np.log(b))))
    elif curve is None:
        scale = float((b * v).sum() / (b * b).sum())
    else:
        from scipy.optimize import minimize_scalar

        def cost(ls):
            m = model(math.exp(ls), b)
            e = (np.log(np.maximum(m, 1e-300)) - np.log(v)) if weight == "relative" else m - v
            return float((e * e).sum())
        sol = minimize_scalar(cost, bounds=(-30.0, 30.0), method="bounded",
                              options={"xatol": 1e-6})
        scale = float(math.exp(sol.x))
    pred = model(scale, b)
    if weight == "relative":
        yy, mm = np.log(v), np.log(np.maximum(pred, 1e-300))
    else:
        yy, mm = v, pred
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float(((yy - mm) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"fn": lambda d: model(scale, np.asarray(profile(d), dtype=float)),
            "scale": scale, "r2": r2, "n": int(ok.sum())}


def frame_map(obj_pos: np.ndarray, obj_d: np.ndarray, vals: np.ndarray,
              nodes_pos: np.ndarray, node_d: np.ndarray, *, method: str, floor: float,
              weight: str, power: float, reach: float, unit: str = "",
              ps: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One frame's map on ``nodes_pos`` from the objects at ``obj_pos`` (µm, both XY or both
    XYZ — the space the residuals are interpolated in) with values ``vals``, ``obj_d`` /
    ``node_d`` their distances from the source (NaN without one).

    * ``trend``: the exponential ``A·e^(-d/λ)`` of :func:`figure.xy_exponential` — the very fit
      Plot XY draws — on the objects at or above ``floor``;
    * ``trend_idw``: that trend times ``exp`` of the inverse-distance-weighted LOG residuals
      ``ln(v / trend)``, so the map passes through every object's value and, between them,
      changes the way the decay does; residuals fade to 1 (the trend) beyond ``reach``;
    * ``idw``: ``exp`` of the inverse-distance-weighted ``ln v`` — no model, no source;
    * ``point_source`` / ``point_source_idw`` (2026-10-16): as ``trend`` / ``trend_idw``, the
      trend a Point-Source Fit's profile (``ps``: ``{"profile", "curve"}``) with its
      amplitude refitted to these objects (:func:`point_source_trend`).

    Returns the node values, the node trend, each node's nearest object distance, the
    objects' trend and log residual, and the fit (``None`` when no trend could be fitted, in
    which case every value is NaN)."""
    n_nodes = len(nodes_pos)
    nan_nodes = np.full(n_nodes, np.nan)
    use = np.isfinite(vals) & (vals > 0)
    if floor and math.isfinite(floor):
        use &= vals >= floor
    fit = None
    obj_trend = np.full(len(vals), np.nan)
    node_trend = nan_nodes.copy()
    if method in _PS_METHODS:
        pst = None if ps is None or ps.get("profile") is None else point_source_trend(
            obj_d, vals, use & np.isfinite(obj_d), ps["profile"], ps.get("curve"), weight)
        if pst is None:
            return {"value": nan_nodes, "trend": nan_nodes, "nearest": nan_nodes,
                    "obj_trend": obj_trend, "obj_res": np.full(len(vals), np.nan), "fit": None}
        fit = {"a": pst["scale"], "lam": math.nan, "r2": pst["r2"], "n": pst["n"],
               "model": "point_source"}
        with np.errstate(invalid="ignore", divide="ignore"):
            obj_trend = np.asarray(pst["fn"](obj_d), dtype=float)
            node_trend = np.asarray(pst["fn"](node_d), dtype=float)
    elif method in _TREND_METHODS:
        ok = np.isfinite(obj_d) & np.isfinite(vals)
        fit = FIG.xy_exponential(obj_d[ok], vals[ok], min_y=floor, weight=weight, unit=unit)
        if fit is None:
            return {"value": nan_nodes, "trend": nan_nodes, "nearest": nan_nodes,
                    "obj_trend": obj_trend, "obj_res": np.full(len(vals), np.nan), "fit": None}
        ln_a, k = math.log(fit["a"]), fit["k"]
        obj_trend = np.exp(ln_a + k * obj_d)
        node_trend = np.exp(ln_a + k * node_d)
    obj_res = np.full(len(vals), np.nan)
    if method != "idw":
        with np.errstate(divide="ignore", invalid="ignore"):
            obj_res[use] = np.log(vals[use]) - np.log(obj_trend[use])
        use &= np.isfinite(obj_res)
    if not use.any():
        return {"value": nan_nodes if method in _IDW_TREND or method == "idw" else node_trend,
                "trend": node_trend, "nearest": nan_nodes, "obj_trend": obj_trend,
                "obj_res": obj_res, "fit": fit}
    if method == "idw":
        lv, near = idw(nodes_pos, obj_pos[use], np.log(vals[use]), power=power)
        value = np.exp(lv)
    else:
        r, near = idw(nodes_pos, obj_pos[use], obj_res[use], power=power,
                      prior_dist=reach if method in _IDW_TREND else 0.0)
        value = node_trend * np.exp(r) if method in _IDW_TREND else node_trend
    return {"value": value, "trend": node_trend, "nearest": near, "obj_trend": obj_trend,
            "obj_res": obj_res, "fit": fit}


# ── the node ──────────────────────────────────────────────────────────────────────────────

def _point_source_record(pts: Dataset, vname: str):
    """``(record, curve)`` of the Point-Source Fit upstream for the mapped value: fitted on
    the value itself (``curve`` None), or on the concentration a standard curve made from it
    (``curve`` that curve, the profile mapped back through it). Refused without one."""
    fits = (pts.metadata or {}).get("point_source_fit") or {}
    if isinstance(fits, Mapping) and isinstance(fits.get(vname), Mapping):
        return dict(fits[vname]), None
    for rec in (fits.values() if isinstance(fits, Mapping) else ()):
        if isinstance(rec, Mapping) and rec.get("signal") == vname and rec.get("curve"):
            return dict(rec), dict(rec["curve"])
    raise ValueError(
        f"Gradient Map: method point_source takes its trend's shape from a Point-Source Fit "
        f"(table.point_source_fit) upstream of the Points table, fitted on {vname!r} or on "
        f"the concentration a Standard Curve made from it — none is on the wire "
        f"(fitted: {sorted(fits) if isinstance(fits, Mapping) else []}). Wire one, or use "
        f"trend_idw.")


def _first_seen(values: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """``(label, row indices)`` per distinct value in first-seen order; text kept as text, a
    number printed with ``%g``, a missing value grouped as ``(missing)``, last."""
    raw = np.asarray(values)
    text = raw.dtype.kind in "USO"
    keys: List[str] = []
    for v in raw.tolist():
        if text:
            k = "" if v is None else (v.decode("utf-8", "replace") if isinstance(v, bytes)
                                      else str(v))
        else:
            f = float(v)
            k = "" if not math.isfinite(f) else f"{f:g}"
        keys.append(k if k.strip() else "(missing)")
    order = list(dict.fromkeys(k for k in keys if k != "(missing)"))
    if "(missing)" in keys:
        order.append("(missing)")
    arr = np.asarray(keys, dtype=object)
    return [(k, np.flatnonzero(arr == k)) for k in order]


def _exclusion(ctx: EvalContext, cols: Mapping[str, np.ndarray]) -> float:
    """Exclude: a number (µm) or a column whose median is the radius; 0 for blank."""
    raw = str(ctx.params.get("exclude", "") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    if raw not in cols or TB.is_text(cols[raw]):
        raise ValueError(f"Gradient Map: Exclude {raw!r} is neither a number of µm nor a numeric "
                         f"column of the points table (it has {sorted(cols)}).")
    v = np.asarray(cols[raw], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        raise ValueError(f"Gradient Map: the Exclude column {raw!r} holds no number.")
    return max(0.0, float(np.median(v)))


def _source_position(ctx: EvalContext, names: Tuple[str, str, str]
                     ) -> Optional[Tuple[float, float, float]]:
    """The source's stage ``(x, y, z)``: the mean of the Source table's rows (``z`` NaN
    without a Z column); ``None`` when the Source input is unwired."""
    src = ctx.input("source")
    if src is None:
        return None
    modes = ctx.params.get("__modes__", {}) or {}
    sdom = TB.table_domain(modes, "source_domain")
    layer = _resolve_layer(
        _structure_layers(src, sdom), ctx.layer("source_table"), node="gradient map",
        socket="source_table", what=f"{sdom.value} table", where="the `source` input",
        remedy="wire the Measure that located the source (a spheroid's whole-mass label, "
               "say), or switch Source domain", ctx=ctx)[0]
    cols = TB.read_table(src, sdom, layer)
    out = []
    for nm in names:
        if not nm:
            out.append(float("nan"))
            continue
        if nm not in cols or TB.is_text(cols[nm]):
            raise ValueError(f"Gradient Map: the source table {layer!r} has no numeric column "
                             f"{nm!r}; it carries {sorted(cols)}. The stage columns come from "
                             f"Measure.")
        v = np.asarray(cols[nm], dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            raise ValueError(f"Gradient Map: the source table {layer!r} has no finite {nm!r}.")
        out.append(float(v.mean()))
    return out[0], out[1], out[2]


def _compute_gradient_map(ctx: EvalContext) -> Dataset:
    """A value measured at scattered objects (capture beads' MFI) as a FIELD over an image,
    following the exponential decay those values show with distance from a point source.

    Resolved spec (2026-10-15, `build-node-v2`):

    * **Kind** analysis → ``analysis.gradient_map``: tables in, two Point tables out, on the
      image of ``data`` (an Experiment Canvas with a baked background, a stitched mosaic):
      ONE field (M = 1) and one plane (Z = 1), placed by its stage box — refused otherwise
      with the node that makes one.
    * **Inputs** the objects' table (``points``, or ``data``'s own when unwired): ``x`` / ``y``
      / ``z`` stage columns (z blank = distances in the XY plane) and ``value``; the
      ``source`` table, whose rows' mean x / y / z is the source (required by the trend
      methods, optional for ``idw``).
    * **Frames** blank ``per_by`` = one map per frame of ``data`` from the rows with that ``t``;
      a column = one frame per distinct value in first-seen order (three timepoints from
      different files), T becomes their count — data-dependent, so the envelope marks T
      unknown — the image repeats its first frame (or frame k when the counts agree), the
      clock keys are dropped and ``frame_names`` carries the values (titles say them).
    * **Method** ``trend_idw`` (default): the exponential of :func:`frame_map` (the same
      ``xy_exponential`` Plot XY draws, ``weight`` relative = ln y / absolute = data units,
      rows under ``fit_floor`` left out) times ``exp`` of the inverse-distance-weighted
      (``power``) log residuals — exact at every object, the decay's shape between them, the
      trend again past ``reach`` (0 = never); ``trend`` the exponential alone; ``idw`` plain
      IDW of ln value. With a Z column the map is the horizontal plane ``height`` above the
      source, and every distance is 3-D: the trend's (node → source) and the interpolation's
      (node → object), so an object far above the plane weighs as far away — a dim bead
      665 µm over a spheroid does not punch a hole beside it at the spheroid's own height
      (2026-10-15, seen at T1). Without Z, both are XY. ``exclude`` (µm, or a column's
      median) blanks the nodes within that XY distance of the source — the source itself.
    * **Lattice** nodes ``spacing`` apart (rounded so they tile the image exactly), centred in
      their cells, in the compositor's continuous pixel convention (pixel i spans [i, i+1];
      Measure's centroids sit half a pixel off it). Columns: ``id m t c z y x`` (z = 0),
      ``stage_x_um`` / ``stage_y_um``, ``distance_um`` (to the source), ``<value>``,
      ``trend``, ``nearest_um``. The objects table copies each used row with its position on
      this image and ``distance_um``, ``trend``, ``log_residual`` and the frame's
      ``trend_a`` / ``trend_lambda_um`` / ``trend_r2``. Both carry the value column's unit.
    * **Footprint** WHOLE_SERIES, empty kernel_axes: no pixel is read; the image is passed on
      (re-indexed along T when frames are groups). Numpy only (scipy via the exponential's
      absolute fit). A frame with fewer than two usable objects is all-NaN, said on the rail.
    """
    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    method = str(modes.get("method", "trend_idw"))
    weight = str(modes.get("weight", "relative"))
    ax = ds.axes
    if int(ax.m) != 1 or int(ax.z) != 1:
        raise ValueError(
            f"Gradient Map draws on ONE field and ONE plane; `data` has M = {ax.m}, Z = {ax.z}. "
            f"Wire an Experiment Canvas (spanning every file) or a Stitch, then Z Project or "
            f"Select Plane.")
    flip = (bool(ctx.params.get("flip_x", True)), bool(ctx.params.get("flip_y", False)))
    geo = image_geometry(ds.metadata or {}, ax, flip)
    if geo is None:
        raise ValueError("Gradient Map: `data` has no stage placement (a pixel size and an "
                         "origin or stage log) — wire an Experiment Canvas or a stitched "
                         "acquisition.")
    pts_in = ctx.input("points")
    pts = ds if pts_in is None else pts_in
    layer = _resolve_layer(
        _structure_layers(pts, dom), ctx.layer("table"), node="gradient map", socket="table",
        what=f"{dom.value} table",
        where="the `data` input" if pts_in is None else "the `points` input",
        remedy=f"wire the Dataset whose {dom.value} table holds the measured objects (Measure "
               f"writes their stage columns), or switch Domain", ctx=ctx)[0]
    name = ctx.layer("name") or _DEFAULT_NAME
    obj_name = name + OBJECTS_SUFFIX
    for nm in (name, obj_name):
        TB.refuse_voxel_name(ds, nm, "Gradient Map")
    cols = TB.read_table(pts, dom, layer)
    xname = str(ctx.params.get("x", "stage_x_um") or "stage_x_um")
    yname = str(ctx.params.get("y", "stage_y_um") or "stage_y_um")
    zname = str(ctx.params.get("z", "stage_z_um") or "")
    vname = str(ctx.params.get("value", "") or "")
    if not vname:
        raise ValueError("Gradient Map: name the Value column to map — a bead's background-"
                         "subtracted MFI, say.")
    for nm in (xname, yname, zname, vname):
        if nm and (nm not in cols or TB.is_text(cols[nm])):
            raise ValueError(f"Gradient Map: the {layer!r} table has no numeric column "
                             f"{nm!r}; it carries {sorted(cols)}.")
    per_by = str(ctx.params.get("per_by", "") or "")
    if per_by and per_by not in cols:
        raise ValueError(f"Gradient Map: Per column {per_by!r} is not a column of the "
                         f"{layer!r} table (it has {sorted(cols)}).")
    floor = float(ctx.params.get("fit_floor", 0.0) or 0.0)
    power = float(ctx.params.get("power", 2.0))
    reach = float(ctx.params.get("reach", 0.0) or 0.0)
    height = float(ctx.params.get("height", 0.0) or 0.0)
    spacing = float(ctx.params.get("spacing", 25.0))
    for lab, v, ok in (("Power", power, power > 0), ("Reach", reach, reach >= 0),
                       ("Spacing", spacing, spacing > 0)):
        if not (math.isfinite(v) and ok):
            raise ValueError(f"Gradient Map: {lab} must be a finite number "
                             f"{'> 0' if lab != 'Reach' else '>= 0'}, got {v}.")
    src = _source_position(ctx, (xname, yname, zname))
    if src is None and method in _TREND_METHODS:
        raise ValueError(f"Gradient Map: method {method!r} fits a decay from the SOURCE — wire "
                         f"the source's table (the Measure that found the spheroid) into "
                         f"Source, or use method idw.")
    excl = _exclusion(ctx, cols)
    if excl > 0 and src is None:
        raise ValueError("Gradient Map: Exclude is a distance from the source — wire Source.")
    ps_rec, ps_curve = _point_source_record(pts, vname) if method in _PS_METHODS else \
        (None, None)

    # the lattice over the whole image
    box, px, _rx, _ry = geo
    H, W = int(ax.y), int(ax.x)
    sp = spacing / px
    nx, ny = max(1, int(round(W / sp))), max(1, int(round(H / sp)))
    u = (np.arange(nx) + 0.5) * (W / nx)
    v = (np.arange(ny) + 0.5) * (H / ny)
    V, U = np.meshgrid(v, u, indexing="ij")
    U, V = U.ravel(), V.ravel()
    nsx, nsy = image_to_stage(U, V, geo)
    three_d = bool(zname) and src is not None and math.isfinite(src[2])
    if three_d:
        nodes_pos = np.stack([nsx, nsy, np.full(len(nsx), src[2] + height)], axis=1)
    else:
        nodes_pos = np.stack([nsx, nsy], axis=1)
    if src is not None:
        dxy = np.hypot(nsx - src[0], nsy - src[1])
        node_d = np.hypot(dxy, height) if zname else dxy
    else:
        dxy = node_d = np.full(len(U), np.nan)
    unit = ""
    from nodegraph.units import format_unit, unit_of, with_unit
    vunit = unit_of(pts, dom, vname, layer)
    if src is not None:
        unit = "µm"

    px_all = np.asarray(cols[xname], dtype=float)
    py_all = np.asarray(cols[yname], dtype=float)
    pz_all = np.asarray(cols[zname], dtype=float) if zname else np.zeros(len(px_all))
    pv_all = np.asarray(cols[vname], dtype=float)
    if per_by:
        frames = _first_seen(cols[per_by])
    else:
        t_all = np.asarray(cols.get("t", np.zeros(len(px_all))), dtype=float)
        frames = [(str(k), np.flatnonzero(t_all == k)) for k in range(int(ax.t))]

    lat: Dict[str, List[np.ndarray]] = {k: [] for k in ("t", "y", "x") + _LATTICE_COLUMNS +
                                        (vname,)}
    obj_rows: List[np.ndarray] = []
    obj_extra: Dict[str, List[np.ndarray]] = {k: [] for k in ("t", "y", "x", vname) +
                                              _OBJECT_COLUMNS}
    notes: List[str] = []
    for k, (label, rows) in enumerate(frames):
        fin = rows[np.isfinite(px_all[rows]) & np.isfinite(py_all[rows]) &
                   np.isfinite(pz_all[rows])]
        ox, oy, oz, ov = px_all[fin], py_all[fin], pz_all[fin], pv_all[fin]
        if src is not None:
            od = np.hypot(np.hypot(ox - src[0], oy - src[1]), (oz - src[2]) if zname else 0.0)
        else:
            od = np.full(len(fin), np.nan)
        opos = np.stack([ox, oy, oz], axis=1) if three_d else np.stack([ox, oy], axis=1)
        ps = None
        if ps_rec is not None:
            import nodegraph.catalog._shared.point_source as PS
            t_k = PS.group_time(ps_rec, label)
            ps = {"curve": ps_curve, "profile": None if t_k is None else (
                lambda d, _t=t_k, _lab=label: PS.predict(
                    np.asarray(d, dtype=float), np.full(np.shape(d), _t), ps_rec, _lab))}
        fm = frame_map(opos, od, ov, nodes_pos, node_d, method=method,
                       floor=floor, weight=weight, power=power, reach=reach, unit=unit, ps=ps)
        val = fm["value"].copy()
        if excl > 0:
            val[dxy < excl] = np.nan
        lat["t"].append(np.full(len(U), k))
        lat["y"].append(V)
        lat["x"].append(U)
        lat["stage_x_um"].append(nsx)
        lat["stage_y_um"].append(nsy)
        lat["distance_um"].append(node_d)
        lat[vname].append(val)
        lat["trend"].append(fm["trend"])
        lat["nearest_um"].append(fm["nearest"])
        ou, ovv = stage_to_image(ox, oy, geo)
        f = fm["fit"]
        obj_rows.append(fin)
        obj_extra["t"].append(np.full(len(fin), k))
        obj_extra["y"].append(ovv)
        obj_extra["x"].append(ou)
        obj_extra[vname].append(ov)
        obj_extra["stage_x_um"].append(ox)
        obj_extra["stage_y_um"].append(oy)
        obj_extra["stage_z_um"].append(oz if zname else np.full(len(fin), np.nan))
        obj_extra["distance_um"].append(od)
        obj_extra["trend"].append(fm["obj_trend"])
        obj_extra["log_residual"].append(fm["obj_res"])
        for key, fk in (("trend_a", "a"), ("trend_lambda_um", "lam"), ("trend_r2", "r2")):
            obj_extra[key].append(np.full(len(fin), f[fk] if f else np.nan))
        if method in _PS_METHODS and f is None:
            notes.append(f"{label}: no trend (no usable object, or a timepoint the "
                         f"Point-Source Fit never saw)")
        elif method in _TREND_METHODS and f is None:
            notes.append(f"{label}: no trend (under two usable objects)")
        elif f is not None and f.get("model") == "point_source":
            notes.append(f"{label}: the fitted profile × {f['a']:.3g}, R² {f['r2']:.2f}, "
                         f"n {f['n']}")
        elif f is not None:
            notes.append(f"{label}: λ {f['lam']:.0f} µm, R² {f['r2']:.2f}, n {f['n']}")
        else:
            notes.append(f"{label}: {int(np.isfinite(ov).sum())} objects")
        ctx.progress(k + 1, len(frames), f"mapping frame {k + 1}/{len(frames)}")

    n_lat = len(U) * len(frames)
    lat_cols: Dict[str, np.ndarray] = {
        "id": np.arange(1, n_lat + 1), "m": np.zeros(n_lat, np.int64),
        "t": np.concatenate(lat["t"]).astype(np.int64) if frames else np.zeros(0, np.int64),
        "c": np.zeros(n_lat, np.int64), "z": np.zeros(n_lat, np.int64)}
    for key in ("y", "x") + _LATTICE_COLUMNS + (vname,):
        lat_cols[key] = np.concatenate(lat[key]) if frames else np.zeros(0)
    sel = np.concatenate(obj_rows) if obj_rows else np.zeros(0, np.int64)
    obj_cols: Dict[str, np.ndarray] = {
        c: np.asarray(vv)[sel] for c, vv in cols.items()
        if c not in _COORDS and c != BATCH_COLUMN}
    n_obj = len(sel)
    obj_cols.update({"id": np.arange(1, n_obj + 1), "m": np.zeros(n_obj, np.int64),
                     "c": np.zeros(n_obj, np.int64), "z": np.zeros(n_obj, np.int64)})
    for key, parts in obj_extra.items():
        obj_cols[key] = np.concatenate(parts) if parts else np.zeros(0)
    obj_cols["t"] = obj_cols["t"].astype(np.int64)

    out = ds
    n_t = len(frames)
    if per_by:
        t_in = int(ax.t)
        sources = [k if t_in == n_t else 0 for k in range(max(1, n_t))]
        if out.image is not None:
            from nodegraph.provider import FrameRemapProvider
            out = out.with_image(FrameRemapProvider(out.image, sources))
        from nodegraph.catalog._shared.frame_subset import remap_lattice_layers
        out = remap_lattice_layers(out, "t", sources)
        out = out.reshaped_axes(replace(ax, t=max(1, n_t)))
        out = out.with_metadata(**{k: None for k in _CLOCK_KEYS},
                                frame_names=[lab for lab, _r in frames])
    out = TB.write_table(out, Domain.POINT, name, lat_cols, "plane_index")
    out = TB.write_table(out, Domain.POINT, obj_name, obj_cols, "plane_index")
    if vunit is not None:
        for tb in (name, obj_name):
            for cname in (vname, "trend"):
                out = with_unit(out, Domain.POINT, cname, vunit, layer=tb)
    ctx.progress(1, 1, f"{method}: " + "; ".join(notes)
                 + (f"; {excl:.0f} µm round the source left blank" if excl > 0 else ""))
    return out


def _gm_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    name = TB.param_text(params, "name", _DEFAULT_NAME)
    return ((Domain.POINT, name), (Domain.POINT, name + OBJECTS_SUFFIX))


def _gm_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                incoming: Sequence = (), inputs: Sequence = ()):
    """The lattice table's columns and the objects table's (the points table's own plus the
    computed ones), on their two names (total)."""
    try:
        dom = TB.table_domain(modes)
        name = TB.param_text(params, "name", _DEFAULT_NAME)
        value = TB.param_text(params, "value")
        envs = {sock: env for sock, env in (inputs or ())}
        env = envs.get("points", envs.get("data"))
        if env is not None:
            src_cols = TB.envelope_columns(
                env, dom, TB.envelope_layer(env, dom, TB.param_text(params, "table")))
        else:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            want = TB.param_text(params, "table") or (known[0] if len(known) == 1 else "")
            src_cols = [c for d, lyr, c in incoming or () if d is dom and lyr == want]
        lat = list(_COORDS) + list(_LATTICE_COLUMNS) + ([value] if value else [])
        obj = list(_COORDS) + [c for c in src_cols if c not in _COORDS and c != BATCH_COLUMN] \
            + list(_OBJECT_COLUMNS) + ([value] if value else [])
        out = [(Domain.POINT, name, c) for c in dict.fromkeys(lat)]
        out += [(Domain.POINT, name + OBJECTS_SUFFIX, c) for c in dict.fromkeys(obj)]
        return tuple(out)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


_gm_columns.wants_inputs = True


def _meta_gradient(env, params: Mapping[str, Any], modes: Mapping[str, Any]):
    """With a Per column the frames are GROUPS — how many is in the data, so T is unknown at
    edit time — and the per-frame clock no longer describes them (dropped, as the compute
    drops it). Without one, the envelope passes through. Total."""
    try:
        if not TB.param_text(params, "per_by"):
            return env
        md = {k: v for k, v in (env.metadata or {}).items() if k not in _CLOCK_KEYS}
        return replace(env, metadata=md,
                       unknown_axes=frozenset(env.unknown_axes) | frozenset({"t"}))
    except Exception:                                   # noqa: BLE001 — total by contract
        return env


_TREND = {"method": frozenset({"trend_idw", "trend", "point_source_idw", "point_source"})}

register_node(
    _compute_gradient_map, op_key="analysis.gradient_map", label="Gradient Map",
    category="analysis",
    inputs=[
        InDataset("data", description="The image the map is drawn on — ONE field and ONE plane "
                                       "with a stage placement: an Experiment Canvas spanning "
                                       "every file (Overlay output = resample bakes a background "
                                       "onto it) or a stitched, projected mosaic. Passed on with "
                                       "the map's two Point tables added."),
        InDataset("points", label="Points", passes_domains=False,
                  description="The Dataset whose table holds the measured objects — beads with "
                              "their stage position and value. Unwired: `data`'s own table."),
        InDataset("source", label="Source", passes_domains=False,
                  description="The Dataset whose table locates the point source (the Measure "
                              "of a spheroid's whole mass): the mean of its rows' X / Y / Z. "
                              "Required by the trend methods; optional for idw."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 layer_from="points",
                 description="The objects' table. Blank uses the only one of its kind on the "
                             "Points wire (or on `data` when Points is unwired)."),
        InString("source_table", "Source table", field=False, default="",
                 layer_in_mode="source_domain", layer_from="source",
                 description="The table on the Source wire whose rows locate the source. "
                             "Blank uses the only one of its kind there."),
        InString("value", "Value", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="The column mapped — each object's measurement (a bead's "
                             "background-subtracted MFI, or its concentration after a "
                             "Standard Curve). Required. The map is in its units."),
        InString("x", "X", field=False, default="stage_x_um", column_in_mode="domain",
                 column_from="table",
                 description="The objects' stage X in µm, in the SAME stage frame as `data` "
                             "and the source (register later runs first)."),
        InString("y", "Y", field=False, default="stage_y_um", column_in_mode="domain",
                 column_from="table",
                 description="The objects' stage Y in µm, as X."),
        InString("z", "Z", field=False, default="stage_z_um", column_in_mode="domain",
                 column_from="table",
                 description="The objects' focus in µm: distances become 3-D — to the source "
                             "for the trend, and from each map node (in the plane Plane height "
                             "above the source) to each object for the interpolation — so a "
                             "bead high above a spheroid counts as far from it. Blank uses XY "
                             "distances only. Changes the fitted λ and every map value."),
        InString("per_by", "Per column", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="One map per distinct value of this column — a Table Concat's "
                             "`condition` (timepoints from different files) — in first-seen "
                             "order, as the frames of the output; titles name them. Blank: "
                             "one map per frame of `data`, from the rows of that frame."),
        InFloat("spacing", "Spacing", unit="um", field=False, default=25.0,
                description="Distance between the map's nodes. SMALLER is smoother and "
                            "slower (rows grow with the square: a 9 mm canvas at 25 µm is "
                            "~130,000 nodes a frame). Rounded so the nodes tile the image "
                            "exactly."),
        InFloat("fit_floor", "Fit floor", unit="", field=False, default=0.0,
                description="Objects whose value is below this leave the fit and the "
                            "interpolation (they stay in the objects table) — a background-"
                            "subtracted reading near zero is noise, and on a log scale a 0.01 "
                            "weighs like a real hundred-fold drop. ~1 count for a bead MFI. 0 "
                            "uses every value > 0. Moves every map value."),
        InFloat("power", "Power", unit="", field=False, default=2.0,
                available_in={"method": frozenset({"trend_idw", "idw", "point_source_idw"})},
                description="How fast an object's influence falls with distance: weight = "
                            "1/d^power. HIGHER keeps each object's value close to it and lets "
                            "the trend (or the next object) take over sooner; LOWER smooths "
                            "the map toward the average. 2 is the common choice."),
        InFloat("reach", "Reach", unit="um", field=False, default=0.0,
                available_in={"method": frozenset({"trend_idw", "point_source_idw"})},
                description="Beyond about this distance from every object the map relaxes to "
                            "the trend alone — a residual is only trusted near the bead that "
                            "measured it. 0 never relaxes (far away the map is the trend times "
                            "the objects' typical deviation). Only read by trend_idw."),
        InFloat("height", "Plane height", unit="um", field=False, default=0.0,
                description="The mapped plane's height above the source, for the distances "
                            "the trend is evaluated at: 0 is the plane through the source's "
                            "centre. Raise it to map the field at the beads' typical height. "
                            "Only read with a Z column."),
        InString("exclude", "Exclude", field=False, default="",
                 description="Leave the nodes within this XY distance of the source blank: a "
                             "number of µm, or a column whose median is the radius (a "
                             "spheroid's `spheroid_edge_um`) — the source's own body, where "
                             "the decay was never measured. Blank excludes nothing."),
        InBool("flip_x", "Flip X", field=False, default=True,
               description="The camera handedness of a `data` image that is not an "
                           "Experiment Canvas — a camera field or a stage-Stitched mosaic, "
                           "which keeps the camera's: on = the image's +x runs along stage "
                           "−x, this scope's default, as Measure uses it. Ignored on a "
                           "canvas, which carries its own orientation."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               description="As Flip X, for the image's +y. Off by default."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="Name of the map's lattice table (a Point table the Viewer's "
                             "Field tab and Plot Field colour); the objects placed on this "
                             "image are written beside it as `<name>_objects`."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_GM_DOMAINS), default="label", label="Domain",
             description="Which kind of table holds the measured objects.",
             choice_docs=domain_docs(_GM_DOMAINS)),
        Mode("source_domain", list(_GM_DOMAINS), default="label", label="Source domain",
             description="Which kind of table on the Source wire locates the source.",
             choice_docs=domain_docs(_GM_DOMAINS)),
        Mode("method", list(_GM_METHODS), default="trend_idw", label="Method",
             description="How the space between the objects is filled.",
             choice_docs={
                 "trend_idw": "The exponential decay fitted to the objects against their "
                              "distance from the source, times their inverse-distance-"
                              "weighted log deviations from it: the map passes through every "
                              "object's value and between them changes the way the decay "
                              "does; beyond Reach it is the decay alone. Needs Source.",
                 "trend": "The fitted exponential decay alone — a smooth radial field round "
                          "the source that ignores how each object departs from it (an "
                          "outlier bead leaves no mark). Needs Source.",
                 "idw": "Inverse-distance weighting of the log values, no model and no "
                        "source: the map is a weighted geometric mean of nearby objects, flat "
                        "between them and the average far from them.",
                 "point_source_idw": "As trend_idw, but the trend is the Crank point-source "
                                     "profile a Point-Source Fit upstream fitted (its D, and "
                                     "this timepoint's time), its AMPLITUDE refitted to these "
                                     "objects — so a subset (beads near the floor) sets the "
                                     "level, every bead the shape. Through the fit's "
                                     "standard curve when Value is the signal. Needs Source "
                                     "and the fit; Per column must be the fit's Group.",
                 "point_source": "The refitted point-source profile alone (see "
                                 "point_source_idw) — a smooth radial field that ignores how "
                                 "each object departs from it. Needs Source and the fit.",
             }),
        Mode("weight", list(FIG.XY_EXP_WEIGHTS), default="relative", label="Trend weighting",
             available_in=_TREND,
             description="How the decay is fitted to the objects — the same choice as Plot "
                         "XY's exponential trendline. Moves λ and every map value.",
             choice_docs={
                 "relative": "Fit ln value: each object weighs by its ratio to the curve, "
                             "so dim far beads count as much as bright near ones and λ is "
                             "stable between timepoints.",
                 "absolute": "Least squares in the value's own units: the brightest few "
                             "objects decide the curve, as the eye does on a linear plot; λ "
                             "can swing with a few bright beads.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT})}},
    adds_domains=frozenset({Domain.POINT}),
    extra_layers=_gm_layers, adds_columns=_gm_columns, meta_transform=_meta_gradient,
    description="A value measured at scattered objects (capture beads' MFI) as a map over an "
                "image: the exponential decay fitted against distance from a point source, "
                "times the objects' own deviations interpolated between them — one map per "
                "frame or per timepoint, as a lattice Point table that Plot Field and the "
                "Viewer's Field tab colour.")
