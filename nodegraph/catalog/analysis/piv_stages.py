"""Staged PIV (OpenPIV) — one node per pass of the multipass window-deformation
pipeline, placed together as the ``bundle:piv`` node bundle (2026-10-08).

``analysis.piv`` runs OpenPIV's ``windef.piv`` ladder (first pass → validate → replace →
deform pass → validate → …) as one card. These nodes are the same pieces, one each, the way
``nodegraph/kernels/piv_field.py::_piv_one_pair`` is written:

    piv.first_pass    the coarse FFT cross-correlation on the raw frames          (a grid + a field)
    piv.validate      signal-to-noise / global / median tests → flags, then inpainting
    piv.deform_pass   the next, finer window: images deformed by the previous field, re-correlated
    piv.smooth        Garcia-2010 smoothn on the field (optional, between passes)

A pass CHANGES THE GRID (a 32-px window has four times the vectors of a 64-px one), so a
deform pass writes a NEW table on the layer (the previous pass's rows are replaced), while
Validate and Smooth write columns onto the rows they were given. ``disp_*`` is in µm on the
wire, as ``analysis.piv`` writes it, so the Viewer's arrows, Rasterize Field and every
downstream consumer read any stage's output unchanged; ``qfactor`` is the correlation
signal-to-noise, ``excluded`` marks grid points outside the ROI, ``flags`` what validation
refused, ``replaced`` whether the refused vectors were filled from their neighbours.

What is decided once, on the first pass, and inherited: the channel, the reference policy,
the correlation method, the sub-pixel estimator, the ROI layer and the overlap — stamped
as ``piv_*`` provenance that every later stage reads. The window of a deform pass defaults
to HALF the previous one (the ladder convention), so the bundle's three passes at 64 / 32 /
16 px come from one number on the first card. Not here (the single node keeps them): the
ensemble average, velocity columns, the disparity uncertainty and an external reference
Dataset — chain a separate reference in front with Timeseries Builder instead.
"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from nodegraph.bundles import Bundle, BundleEdge, BundleNode, register_bundle
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
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import POINT_INVARIANT, on_layer
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer
from nodegraph.catalog._shared.progress import _UnitBar
from nodegraph.catalog._shared.analysis_units import scoped_frame
from nodegraph.catalog._shared.analysis_mesh import (concat_rows, lattice_quads,
                                                     mesh_columns, mesh_layer_name,
                                                     with_mesh)

_DEFAULT_LAYER = "piv"
FIELD_COLUMNS = ("disp_y", "disp_x", "disp_mag_um", "qfactor", "excluded")
VALIDATE_COLUMNS = ("disp_y", "disp_x", "disp_mag_um", "flags", "replaced")
SMOOTH_COLUMNS = ("disp_y", "disp_x", "disp_mag_um")

#: What the first pass stamps and every later stage reads.
_PASS_PROV = ("piv_window", "piv_overlap", "piv_overlap_px", "piv_grid_shape", "piv_pass",
              "piv_correlation", "piv_subpixel", "piv_reference_mode", "piv_reference_frame",
              "piv_channel", "piv_roi")


# ── shared plumbing ───────────────────────────────────────────────────────────────

def _piv_px(ctx: EvalContext) -> float:
    return float(ctx.calib("pixel_size_um") or 0.1)


def _piv_plan(ax, ref_mode: str, ref_frame: int) -> List[Tuple[int, int]]:
    """``[(t_out, t_ref), …]``: previous_frame pairs ``(t-1, t)`` for ``t ≥ 1``; fixed_frame
    pairs every t (the reference itself included — a self-pair is well-defined and ~0,
    as the single node does it) against ``reference_frame``."""
    if ref_mode == "previous_frame":
        return [(t, t - 1) for t in range(1, ax.t)]
    r = min(max(0, int(ref_frame)), max(0, ax.t - 1))
    return [(t, r) for t in range(ax.t)]


def _piv_blocks(ax, plan) -> List[Tuple[int, int, int, int]]:
    """``(m, z, t_out, t_ref)`` in the table's row-block order."""
    return [(m, z, t_out, t_ref) for m in range(ax.m) for z in range(ax.z) for (t_out, t_ref) in plan]


def _piv_plane(prov, m: int, t: int, z: int, c: int) -> np.ndarray:
    pax = prov.axes
    return np.asarray(prov.get_region(0, m, t, z, c, 0, pax.y, 0, pax.x), dtype=np.float64)


def _piv_roi_plane(ds: Dataset, roi_name: str, m: int, z: int, c: int) -> Optional[np.ndarray]:
    """The ROI mask plane for an ``(m, z)`` line, STATIC in t (the t=0 plane), truthy =
    inside — ``None`` when no mask layer of that name is on the Dataset."""
    if not roi_name:
        return None
    attr = ds.get(Domain.VOXEL, roi_name)
    if attr is None:
        return None
    v = attr.values
    s = v.shape
    return np.asarray(v[min(m, s[0] - 1), 0, min(z, s[2] - 1), min(c, s[3] - 1)]) > 0


def _piv_table(ds: Dataset, layer: str, node: str) -> Dict[str, np.ndarray]:
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT) if a.layer == layer}
    for req in ("id", "m", "t", "c", "z", "y", "x", "disp_y", "disp_x"):
        if req not in cols:
            raise ValueError(f"{node}: Point layer {layer!r} lacks the column {req!r} — run PIV "
                             "First Pass upstream (it writes the grid every later stage reads)")
    return cols


def _piv_prov(md, node: str, keys=_PASS_PROV) -> Dict[str, Any]:
    out = {}
    for k in keys:
        if k not in md:
            raise ValueError(
                f"{node}: the input carries no {k!r} provenance — this node reads its grid, "
                "correlation settings and reference policy from PIV First Pass's output; "
                "wire the PIV stages in order (the bundle places them so)")
        out[k] = md[k]
    return out


def _piv_rows(cols, m: int, t: int, z: int, c: int, n: int, node: str) -> np.ndarray:
    idx = np.flatnonzero((cols["m"] == m) & (cols["t"] == t) & (cols["z"] == z) & (cols["c"] == c))
    if len(idx) != n:
        raise ValueError(
            f"{node}: the layer holds {len(idx)} rows for (m={m}, t={t}, z={z}, c={c}) but the "
            f"pass grid has {n} vectors — the table was filtered or re-ordered between stages; "
            "wire the stages directly after PIV First Pass")
    return idx


def _piv_write(ds: Dataset, layer: str, cols: Dict[str, np.ndarray]) -> Dataset:
    for name, vals in cols.items():
        ds = ds.with_layer(Domain.POINT, name, np.asarray(vals), layer=layer)
    return ds


def _piv_cells(r: Dict[str, Any], m: int, z: int, c: int) -> Dict[str, np.ndarray]:
    """A pass's window-centre lattice as grid cells (t = -1: every frame of the line), the
    cells touching an excluded window dropped — what the Viewer draws as the PIV grid."""
    x, y = np.asarray(r["x"], float), np.asarray(r["y"], float)
    if x.ndim != 2 or min(x.shape) < 2:
        return {}
    return lattice_quads(y[:, 0], x[0, :], m=m, t=-1, c=c, z_plane=z,
                         keep=~np.asarray(r["excluded"], bool))


def _piv_rewrite(ds: Dataset, layer: str, cols: Dict[str, np.ndarray]) -> Dataset:
    """Replace the WHOLE table on ``layer`` (a pass changes the grid): every existing
    column of that layer is dropped first, so a stale column from a coarser pass cannot
    survive beside rows it no longer matches."""
    for a in list(ds.layers_on(Domain.POINT)):
        if a.layer == layer:
            ds = ds.without(Domain.POINT, a.name, layer)
    return ds.with_structure(StructureTable(Domain.POINT, cols, layer=layer, z_kind="plane_index"))


def _piv_source(ctx: EvalContext, ds: Dataset, node: str, want: str) -> str:
    layer, _note = _resolve_layer(
        _point_layers(ds), want, node=node, socket="source",
        what="PIV grid table", where="the `data` input",
        remedy="run PIV First Pass upstream — it writes the Point table the stages fill", ctx=ctx)
    return layer


def _piv_settings_params(prov: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """The kernel's params dict from the stamped correlation settings plus a stage's own."""
    p = {"correlation_method": str(prov["piv_correlation"]),
         "subpixel_method": str(prov["piv_subpixel"]),
         "overlap": float(prov["piv_overlap"])}
    p.update(extra)
    return p


def _piv_field_rows(x: np.ndarray, y: np.ndarray, u, v, s2n: np.ndarray, excluded: np.ndarray,
                px: float, m: int, t: int, z: int, c: int) -> Dict[str, np.ndarray]:
    """One pass's grid + field → the row block (C order over the vector grid)."""
    n = int(x.size)
    dy = np.ma.filled(np.ma.masked_array(v, mask=excluded), np.nan).astype(float).reshape(-1) * px
    dx = np.ma.filled(np.ma.masked_array(u, mask=excluded), np.nan).astype(float).reshape(-1) * px
    return {
        "m": np.full(n, m, dtype=np.int64), "t": np.full(n, t, dtype=np.int64),
        "c": np.full(n, c, dtype=np.int64), "z": np.full(n, float(z)),
        "y": np.asarray(y, float).reshape(-1), "x": np.asarray(x, float).reshape(-1),
        "disp_y": dy, "disp_x": dx, "disp_mag_um": np.sqrt(dy ** 2 + dx ** 2),
        "qfactor": np.asarray(s2n, float).reshape(-1),
        "excluded": np.asarray(excluded, bool).reshape(-1).astype(np.int64),
    }


def _piv_layers_on_source(params, modes):
    return ((Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER)),)


# ── PIV First Pass ────────────────────────────────────────────────────────────────

def _compute_piv_first_pass(ctx: EvalContext) -> Dataset:
    """The coarse pass of OpenPIV's multipass ladder as a node: ``windef.first_pass`` —
    FFT cross-correlation of every interrogation window of ``window_size`` px, no
    deformation — on each frame pair, written as a Point table (one row per window
    centre, in OpenPIV's grid order) with ``disp_*`` (µm), ``qfactor`` (signal-to-noise)
    and ``excluded`` (outside the ROI). Nothing is validated here: pull it to see the raw
    correlation field, then PIV Validate to see what the tests refuse.

    Resolved spec: category analysis; op ``piv.first_pass``; reads VOXEL, adds POINT; 2-D
    per plane (``z_kind`` plane_index); footprint WHOLE_SERIES / {t,y,x}. Stamps the
    ``piv_*`` provenance every later stage inherits: window, overlap, grid shape, pass
    number, correlation method, sub-pixel estimator, reference policy, channel, ROI layer.
    Kernel: :func:`nodegraph.kernels.piv_field.piv_stage_first_pass`."""
    from nodegraph.kernels.piv_field import piv_stage_first_pass, piv_stage_settings
    node = "PIV First Pass"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    ref_mode = str(modes.get("reference_mode", "previous_frame"))
    corr = str(modes.get("correlation", "circular"))
    subpix = str(modes.get("subpixel", "gaussian"))
    # under the quick-analysis scope a source frame index is not a payload index; the
    # stamped reference is what the later stages index the payload with
    ref_frame = scoped_frame(ds.metadata, int(ctx.params.get("reference_frame", 0)), ax.t,
                             mode=ref_mode, node="PIV First Pass")
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    window = max(8, int(ctx.params.get("window_size", 64)))
    overlap = min(0.75, max(0.0, float(ctx.params.get("overlap", 0.5))))
    px = _piv_px(ctx)
    layer = ctx.layer("name")
    roi_name = ctx.layer("roi") or ""
    params = {"correlation_method": corr, "subpixel_method": subpix, "overlap": overlap}
    s, ov_px, grid = piv_stage_settings(params, (ax.y, ax.x), window, overlap)
    plan = _piv_plan(ax, ref_mode, ref_frame)
    blocks = _piv_blocks(ax, plan)
    rows: List[Dict[str, np.ndarray]] = []
    cells: Dict[Any, Dict[str, np.ndarray]] = {}
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)),
                   note="first pass")
    for (m, z, t_out, t_ref) in blocks:
        roi = _piv_roi_plane(ds, roi_name, m, z, c)
        r = piv_stage_first_pass(_piv_plane(prov, m, t_ref, z, c), _piv_plane(prov, m, t_out, z, c), s, roi)
        rows.append(_piv_field_rows(r["x"], r["y"], r["u"], r["v"], r["s2n"], r["excluded"],
                                px, m, t_out, z, c))
        if (m, z) not in cells:                      # the window lattice, once per line
            cells[(m, z)] = _piv_cells(r, m, z, c)
        bar.finish_unit(note=f"m={m} z={z} t={t_out}: {grid[0]}×{grid[1]} vectors at {window} px")
    if rows:
        cols = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    else:
        cols = {k: np.zeros(0) for k in ("m", "t", "c", "z", "y", "x", "disp_y", "disp_x",
                                          "disp_mag_um", "qfactor", "excluded")}
    cols = {"id": np.arange(len(cols["m"]), dtype=np.int64), **cols}
    md = {"piv_window": window, "piv_overlap": overlap, "piv_overlap_px": int(ov_px),
          "piv_grid_shape": (int(grid[0]), int(grid[1])), "piv_pass": 1,
          "piv_correlation": corr, "piv_subpixel": subpix,
          "piv_reference_mode": ref_mode, "piv_reference_frame": ref_frame,
          "piv_channel": c, "piv_roi": roi_name, "piv_stage": "first_pass"}
    out = ds.with_structure(StructureTable(Domain.POINT, cols, layer=layer, z_kind="plane_index"))
    out = with_mesh(out, layer, concat_rows(list(cells.values())), kind="quad",
                    z_kind="plane_index", window=(window, window))
    return out.with_metadata(**md)


def _columns_piv_first_pass(params, modes, incoming):
    name = str((params or {}).get("name") or _DEFAULT_LAYER)
    return (tuple(on_layer(Domain.POINT, name, POINT_INVARIANT + FIELD_COLUMNS))
            + tuple(on_layer(Domain.POINT, mesh_layer_name(name), mesh_columns("quad"))))


def _layers_piv_first_pass(params, modes):
    """The window lattice's cell table, named after the field layer — no socket names it."""
    return ((Domain.POINT, mesh_layer_name(str((params or {}).get("name") or _DEFAULT_LAYER))),)


# ── PIV Validate ──────────────────────────────────────────────────────────────────

def _compute_piv_validate(ctx: EvalContext) -> Dataset:
    """OpenPIV's ``typical_validation`` as a node, on the pass the table holds: the
    signal-to-noise test, the global displacement limit, the standard-deviation test and
    the median test (universal, local, or off) flag vectors; with ``replace`` on, the
    flagged ones are filled from their valid neighbours (iterative local mean) — which
    the multipass driver ALWAYS does before a deform pass, since a hole poisons the
    predictor spline. Writes ``flags`` (refused by a test) and ``replaced`` (filled), and
    overwrites ``disp_*`` with the filled values, or with NaN at refused vectors when
    ``replace`` is off.

    Resolved spec: category analysis; op ``piv.validate``; reads POINT (no voxel is read),
    adds POINT; footprint WHOLE_SERIES with empty kernel_axes. Inherits the pass's window,
    overlap and grid from provenance; stamps its own thresholds."""
    from nodegraph.kernels.piv_field import piv_stage_settings, piv_stage_validate
    node = "PIV Validate"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset (for its axes)")
    ax = prov.axes
    layer = _piv_source(ctx, ds, node, ctx.layer("source"))
    cols = _piv_table(ds, layer, node)
    pv = _piv_prov(ds.metadata, node)
    modes = ctx.params.get("__modes__", {})
    px = _piv_px(ctx)
    extra = {
        "sig2noise_threshold": float(ctx.params.get("s2n_threshold", 1.0)),
        "max_disp_px": max(0.0, float(ctx.params.get("max_disp", 0.0))),
        "std_threshold": max(0.0, float(ctx.params.get("std_threshold", 10.0))),
        "median_test": str(modes.get("median_test", "universal")),
        "median_threshold": max(0.0, float(ctx.params.get("median_threshold", 2.0))),
    }
    replace = bool(ctx.params.get("replace", True))
    s, _ov, grid = piv_stage_settings(_piv_settings_params(pv, extra), (ax.y, ax.x),
                                      int(pv["piv_window"]), float(pv["piv_overlap"]))
    n = int(grid[0]) * int(grid[1])
    c = int(pv["piv_channel"])
    plan = _piv_plan(ax, str(pv["piv_reference_mode"]), int(pv["piv_reference_frame"]))
    n_rows = len(cols["id"])
    dy = np.asarray(cols["disp_y"], float).copy()
    dx = np.asarray(cols["disp_x"], float).copy()
    q = np.asarray(cols["qfactor"], float) if "qfactor" in cols else np.ones(n_rows)
    exc = (np.asarray(cols["excluded"]).astype(bool) if "excluded" in cols
           else np.zeros(n_rows, bool))
    flags = np.zeros(n_rows, dtype=np.int64)
    replaced = np.zeros(n_rows, dtype=np.int64)
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)),
                   note="validating")
    for (m, z, t_out, t_ref) in _piv_blocks(ax, plan):
        idx = _piv_rows(cols, m, t_out, z, c, n, node)
        shp = (int(grid[0]), int(grid[1]))
        r = piv_stage_validate((dx[idx] / px).reshape(shp), (dy[idx] / px).reshape(shp),
                               q[idx].reshape(shp), s, exc[idx].reshape(shp), replace=replace)
        dy[idx] = r["v"].reshape(-1) * px
        dx[idx] = r["u"].reshape(-1) * px
        flags[idx] = r["flags"].reshape(-1).astype(np.int64)
        replaced[idx] = int(r["replaced"]) * r["flags"].reshape(-1).astype(np.int64)
        bar.finish_unit(note=f"m={m} z={z} t={t_out}: {int(r['flags'].sum())}/{n} flagged"
                             + (", filled" if r["replaced"] else ""))
    out = {"disp_y": dy, "disp_x": dx, "disp_mag_um": np.sqrt(dy ** 2 + dx ** 2),
           "flags": flags, "replaced": replaced}
    md = {"piv_stage": "validate", "piv_s2n_threshold": extra["sig2noise_threshold"],
          "piv_max_disp": extra["max_disp_px"], "piv_std_threshold": extra["std_threshold"],
          "piv_median_test": extra["median_test"], "piv_median_threshold": extra["median_threshold"],
          "piv_replace": replace}
    return _piv_write(ds, layer, out).with_metadata(**md)


def _columns_piv_validate(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER),
                    VALIDATE_COLUMNS)


# ── PIV Smooth ────────────────────────────────────────────────────────────────────

def _compute_piv_smooth(ctx: EvalContext) -> Dataset:
    """Garcia-2010 ``smoothn`` (DCT penalised least squares) on the pass's field, as a
    node — what the multipass driver does between passes when smoothing is on, to
    stabilise the predictor before the next deformation. Overwrites ``disp_*``; a NaN
    (a refused, unfilled vector) is treated as missing and filled by the smoother.

    Resolved spec: category analysis; op ``piv.smooth``; reads POINT, adds POINT; footprint
    WHOLE_SERIES with empty kernel_axes."""
    from nodegraph.kernels.piv_field import piv_stage_settings, piv_stage_smooth
    node = "PIV Smooth"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset (for its axes)")
    ax = prov.axes
    layer = _piv_source(ctx, ds, node, ctx.layer("source"))
    cols = _piv_table(ds, layer, node)
    pv = _piv_prov(ds.metadata, node)
    px = _piv_px(ctx)
    strength = max(0.0, float(ctx.params.get("smooth_strength", 0.05)))
    s, _ov, grid = piv_stage_settings(_piv_settings_params(pv, {"smoothn": True, "smoothn_p": strength}),
                                      (ax.y, ax.x), int(pv["piv_window"]), float(pv["piv_overlap"]))
    n = int(grid[0]) * int(grid[1])
    c = int(pv["piv_channel"])
    plan = _piv_plan(ax, str(pv["piv_reference_mode"]), int(pv["piv_reference_frame"]))
    n_rows = len(cols["id"])
    dy = np.asarray(cols["disp_y"], float).copy()
    dx = np.asarray(cols["disp_x"], float).copy()
    exc = (np.asarray(cols["excluded"]).astype(bool) if "excluded" in cols
           else np.zeros(n_rows, bool))
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)),
                   note="smoothing")
    for (m, z, t_out, t_ref) in _piv_blocks(ax, plan):
        idx = _piv_rows(cols, m, t_out, z, c, n, node)
        shp = (int(grid[0]), int(grid[1]))
        us, vs = piv_stage_smooth((dx[idx] / px).reshape(shp), (dy[idx] / px).reshape(shp), s,
                                  exc[idx].reshape(shp))
        dy[idx] = vs.reshape(-1) * px
        dx[idx] = us.reshape(-1) * px
        bar.finish_unit(note=f"m={m} z={z} t={t_out}")
    out = {"disp_y": dy, "disp_x": dx, "disp_mag_um": np.sqrt(dy ** 2 + dx ** 2)}
    return _piv_write(ds, layer, out).with_metadata(piv_stage="smooth", piv_smooth_strength=strength)


def _columns_piv_smooth(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER),
                    SMOOTH_COLUMNS)


# ── PIV Deform Pass ───────────────────────────────────────────────────────────────

def _compute_piv_deform_pass(ctx: EvalContext) -> Dataset:
    """One window-deformation pass as a node: the previous pass's field is interpolated
    onto the new, finer grid as a predictor, both frames are deformed by half of it
    (``symmetric``) or the second frame by all of it (``second image``), the deformed
    pair is correlated at ``window_size`` px and the predictor added back — OpenPIV's
    ``multipass_img_deform``, pass by pass. ``window_size`` 0 means HALF the previous
    window (the ladder convention; never under 8 px). The grid changes, so this node
    writes a NEW table on the layer: the previous pass's rows and columns are replaced.
    Holes in the previous field (refused, unfilled vectors) are filled from their
    neighbours first — the predictor spline tolerates none.

    Resolved spec: category analysis; op ``piv.deform_pass``; reads VOXEL + POINT, adds
    POINT; footprint WHOLE_SERIES / {t,y,x}. Inherits correlation, sub-pixel, overlap,
    reference policy, channel and ROI from provenance; restamps window, grid and pass
    number. Kernel: :func:`nodegraph.kernels.piv_field.piv_stage_deform_pass`."""
    from nodegraph.kernels.piv_field import piv_stage_deform_pass, piv_stage_settings
    node = "PIV Deform Pass"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    layer = _piv_source(ctx, ds, node, ctx.layer("source"))
    cols = _piv_table(ds, layer, node)
    pv = _piv_prov(ds.metadata, node)
    modes = ctx.params.get("__modes__", {})
    px = _piv_px(ctx)
    prev_window = int(pv["piv_window"])
    window = int(ctx.params.get("window_size", 0))
    if window <= 0:
        window = max(8, prev_window // 2)
    overlap = float(ctx.params.get("overlap", 0.0))
    if overlap <= 0.0:
        overlap = float(pv["piv_overlap"])
    overlap = min(0.75, overlap)
    deform = str(modes.get("deformation", "symmetric"))
    s, ov_px, grid = piv_stage_settings(
        _piv_settings_params(pv, {"overlap": overlap, "deformation_method": deform}),
        (ax.y, ax.x), window, overlap)
    if min(grid) < 4:
        raise ValueError(
            f"{node}: a {window} px window with {overlap:.0%} overlap gives a {grid[0]}×{grid[1]} "
            "vector grid, and a deformation pass needs at least 4 × 4 (its predictor is a cubic "
            "spline over the grid) — use a smaller window, more overlap, or a larger field")
    old_grid = tuple(int(v) for v in pv["piv_grid_shape"])
    n_old = old_grid[0] * old_grid[1]
    c = int(pv["piv_channel"])
    roi_name = str(pv["piv_roi"] or "")
    plan = _piv_plan(ax, str(pv["piv_reference_mode"]), int(pv["piv_reference_frame"]))
    dy = np.asarray(cols["disp_y"], float)
    dx = np.asarray(cols["disp_x"], float)
    rows: List[Dict[str, np.ndarray]] = []
    cells: Dict[Any, Dict[str, np.ndarray]] = {}
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)),
                   note=f"pass at {window} px")
    for (m, z, t_out, t_ref) in _piv_blocks(ax, plan):
        idx = _piv_rows(cols, m, t_out, z, c, n_old, node)
        x_old = np.asarray(cols["x"], float)[idx].reshape(old_grid)
        y_old = np.asarray(cols["y"], float)[idx].reshape(old_grid)
        u_old = (dx[idx] / px).reshape(old_grid)
        v_old = (dy[idx] / px).reshape(old_grid)
        roi = _piv_roi_plane(ds, roi_name, m, z, c)
        r = piv_stage_deform_pass(_piv_plane(prov, m, t_ref, z, c), _piv_plane(prov, m, t_out, z, c), s,
                                  x_old, y_old, u_old, v_old, roi)
        rows.append(_piv_field_rows(r["x"], r["y"], r["u"], r["v"], r["s2n"], r["excluded"],
                                px, m, t_out, z, c))
        if (m, z) not in cells:                      # the finer lattice replaces the grid
            cells[(m, z)] = _piv_cells(r, m, z, c)
        bar.finish_unit(note=f"m={m} z={z} t={t_out}: {grid[0]}×{grid[1]} vectors at {window} px")
    if rows:
        out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    else:
        out = {k: np.zeros(0) for k in ("m", "t", "c", "z", "y", "x", "disp_y", "disp_x",
                                         "disp_mag_um", "qfactor", "excluded")}
    out = {"id": np.arange(len(out["m"]), dtype=np.int64), **out}
    md = {"piv_window": window, "piv_overlap": overlap, "piv_overlap_px": int(ov_px),
          "piv_grid_shape": (int(grid[0]), int(grid[1])), "piv_pass": int(pv["piv_pass"]) + 1,
          "piv_deformation": deform, "piv_stage": "deform_pass"}
    res = with_mesh(_piv_rewrite(ds, layer, out), layer, concat_rows(list(cells.values())),
                    kind="quad", z_kind="plane_index", window=(window, window))
    return res.with_metadata(**md)


def _columns_piv_deform_pass(params, modes, incoming):
    name = str((params or {}).get("source") or _DEFAULT_LAYER)
    return (tuple(on_layer(Domain.POINT, name, POINT_INVARIANT + FIELD_COLUMNS))
            + tuple(on_layer(Domain.POINT, mesh_layer_name(name), mesh_columns("quad"))))


def _layers_piv_deform_pass(params, modes):
    """The rewritten field layer and its finer lattice's cell table."""
    name = str((params or {}).get("source") or _DEFAULT_LAYER)
    return ((Domain.POINT, name), (Domain.POINT, mesh_layer_name(name)))


# ── registration ─────────────────────────────────────────────────────────────────

_SOURCE = InString(
    "source", "Grid layer", field=False, default=_DEFAULT_LAYER, layer_in=Domain.POINT,
    description=
    "Which Point table holds the PIV pass this stage works on — the one PIV First Pass wrote "
    "(`piv` unless renamed there), or the last Deform Pass's. Validate and Smooth write their "
    "columns back onto the SAME rows; a Deform Pass replaces the table with its finer grid. "
    "With a single Point table on the wire the name is resolved for you.")

_STAGE_GRAN = Granularity.WHOLE_SERIES
_IMG_KAX = frozenset({"t", "y", "x"})

register_node(
    _compute_piv_first_pass, op_key="piv.first_pass", label="PIV First Pass", category="analysis",
    adds_columns=_columns_piv_first_pass, extra_layers=_layers_piv_first_pass,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InInt("channel", "Channel", unit="", field=False, default=0, pick_kind="channel",
              description=
              "Which channel every pass correlates (0-based) — decided here and inherited. "
              "Displacement comes from image texture: pick the channel with stable, "
              "high-contrast speckle, not a reporter whose brightness changes biologically."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame", available_in={"reference_mode": frozenset({"fixed_frame"})},
              description=
              "Which timepoint every frame is paired with when the reference mode is a fixed "
              "frame (0-based); its own pair is the zero field. Inert under `previous_frame`. "
              "For a separate reference acquisition, chain it in front with Timeseries Builder "
              "and keep this at 0."),
        InInt("window_size", "Window size", unit="px", field=False, default=64,
              description=
              "Edge of the interrogation window of THIS pass in pixels — the coarsest of the "
              "ladder, so the motion it has to bracket is the whole displacement: keep it "
              "above about twice the largest expected shift. Each Deform Pass after it "
              "defaults to half this. Must leave at least 4 × 4 windows on the image for a "
              "later deform pass to interpolate over."),
        InFloat("overlap", "Overlap", unit="", field=False, default=0.5,
                description=
                "Window overlap as a fraction of the window (0.5 = half-overlapping). It sets "
                "the grid DENSITY of every pass (inherited unless a Deform Pass overrides it), "
                "not per-vector accuracy; more overlap costs proportionally more windows."),
        InString("roi", "ROI layer", field=False, default="roi_mask", layer_in=Domain.VOXEL,
                 description=
                 "Optional Voxel mask (from ROI Mask or a Threshold) restricting WHERE vectors "
                 "are computed: windows centred outside it are marked `excluded` and carry no "
                 "vector, on every pass. Static in time (the t=0 plane is used). Absent → the "
                 "whole frame."),
        InString("name", "Grid layer", field=False, default=_DEFAULT_LAYER,
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point table this pass writes — one row per window centre with "
                 "`disp_y`/`disp_x` (µm), `disp_mag_um`, `qfactor` (signal-to-noise) and "
                 "`excluded`. Every later stage selects it by this name."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("reference_mode", ["previous_frame", "fixed_frame"], default="previous_frame",
             label="Reference",
             description="How frames pair up — decided here and inherited by every pass.",
             choice_docs={
                 "previous_frame": "Each frame against the one before it: a velocity-like "
                                   "field per step, the PIV default; t=0 has no row.",
                 "fixed_frame": "Every frame against Reference frame: cumulative "
                                "displacement since that state, comparable across time "
                                "until the motion outgrows the window.",
             }),
        Mode("correlation", ["circular", "linear"], default="circular", label="Correlation",
             description="How each window pair is cross-correlated.",
             choice_docs={
                 "circular": "FFT circular correlation: OpenPIV's default and about twice as "
                             "fast; wraps at the window edge, fine for motions well inside it.",
                 "linear": "Zero-padded (linear) correlation, normalised: no wrap-around "
                           "bias near the window edge, at about twice the cost.",
             }),
        Mode("subpixel", ["gaussian", "centroid", "parabolic"], default="gaussian",
             label="Sub-pixel",
             description="How the correlation peak is located between integer pixels.",
             choice_docs={
                 "gaussian": "A three-point Gaussian fit through the peak: the standard, "
                             "lowest-bias estimator for particle images (~0.1 px).",
                 "centroid": "The intensity-weighted centroid around the peak: robust on a "
                             "broad peak, more peak-locking bias than the Gaussian fit.",
                 "parabolic": "A three-point parabola through the peak: cheapest, and the "
                              "most biased toward integer values.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=_IMG_KAX,
    description="OpenPIV's first (coarsest) pass as a node: FFT cross-correlation of every "
                "window on the raw frames → a Point table of raw vectors with signal-to-noise. "
                "First card of the staged PIV bundle; decides the channel, reference policy, "
                "correlation settings and ROI for every pass after it.")

register_node(
    _compute_piv_validate, op_key="piv.validate", label="PIV Validate", category="analysis",
    adds_columns=_columns_piv_validate, extra_layers=_piv_layers_on_source,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InFloat("s2n_threshold", "Min signal-to-noise", unit="", field=False, default=1.0,
                description=
                "Minimum ratio of the correlation's first peak to the plane mean for a vector "
                "to pass. HIGHER refuses more weak matches (then filled from neighbours or left "
                "NaN); 1.0 refuses essentially nothing — raise toward 1.3-2 on noisy images."),
        InFloat("max_disp", "Max displacement", unit="px", field=False, default=0.0,
                description=
                "Hard limit on either displacement component in pixels; vectors beyond it are "
                "flagged. 0 = half the pass's window, the physical limit a window can resolve."),
        InFloat("std_threshold", "Std threshold", unit="", field=False, default=10.0,
                description=
                "Global filter: flag vectors more than this many standard deviations from the "
                "field mean. 10 catches only gross outliers; lower it on a field that should be "
                "nearly uniform, never on one with real strong gradients."),
        InFloat("median_threshold", "Median threshold", unit="", field=False, default=2.0,
                description=
                "Threshold of the median test: in `universal` mode the normalised residual r* "
                "(the literature default 2), in `local` mode an absolute pixel difference from "
                "the neighbourhood median. LOWER flags more; on a field with genuine sharp "
                "structure the test flags real vectors — raise it or turn the test off."),
        InBool("replace", "Replace flagged", default=True,
               description=
               "Fill the flagged vectors from their valid neighbours (iterative local mean) and "
               "mark them `replaced`. ON is what a following Deform Pass needs (a hole poisons "
               "its predictor; it will fill holes itself if you leave them). OFF leaves NaN at "
               "the refused vectors — the honest final field."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("median_test", ["universal", "local", "off"], default="universal", label="Median test",
             description="The neighbourhood-consistency check, PIV's workhorse outlier detector.",
             choice_docs={
                 "universal": "Westerweel-Scarano normalised median test: the residual is "
                              "scaled by the neighbourhood's own scatter, so one threshold "
                              "works across the field.",
                 "local": "Plain local median test: an absolute pixel difference from the "
                          "3×3 neighbourhood median; simpler, scale-dependent.",
                 "off": "No median test — only signal-to-noise, limits and the std test run. "
                        "For fields with genuine sharp structure the median test would flatten.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=frozenset(),
    description="OpenPIV's validation as a node: signal-to-noise, global limit, std and median "
                "tests flag vectors; flagged ones are filled from neighbours (or left NaN). "
                "Writes flags / replaced on the pass's rows.")

register_node(
    _compute_piv_smooth, op_key="piv.smooth", label="PIV Smooth", category="analysis",
    adds_columns=_columns_piv_smooth, extra_layers=_piv_layers_on_source,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InFloat("smooth_strength", "Strength", unit="", field=False, default=0.05,
                description=
                "The smoother's s parameter — LARGER flattens the field harder. OpenPIV's "
                "default 0.05 steadies a predictor between passes without erasing structure; "
                "on the FINAL pass it is a filter on the measurement, so use it knowingly."),
    ],
    outputs=[OutDataset()],
    granularity=_STAGE_GRAN, kernel_axes=frozenset(),
    description="Garcia-2010 smoothn on the pass's field, as a node — the between-pass smoothing "
                "of the multipass driver, placeable wherever you want it.")

register_node(
    _compute_piv_deform_pass, op_key="piv.deform_pass", label="PIV Deform Pass",
    category="analysis", adds_columns=_columns_piv_deform_pass,
    extra_layers=_layers_piv_deform_pass,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InInt("window_size", "Window size", unit="px", field=False, default=0,
              description=
              "Edge of THIS pass's interrogation window in pixels. 0 = half the previous "
              "pass's window (the coarse-to-fine ladder; never under 8). Finer resolves more "
              "structure and gives four times the vectors per halving; the previous pass must "
              "have left at least a 4 × 4 grid for the predictor."),
        InFloat("overlap", "Overlap", unit="", field=False, default=0.0,
                description=
                "Window overlap for this pass as a fraction; 0 = inherit the first pass's. "
                "Sets this pass's grid density."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("deformation", ["symmetric", "second image"], default="symmetric", label="Deformation",
             description="How the predictor field deforms the frames before re-correlating.",
             choice_docs={
                 "symmetric": "Both frames are deformed by half the predictor toward each "
                              "other: the second-order accurate scheme, OpenPIV's default.",
                 "second image": "Only the second frame is deformed by the full predictor: "
                                 "cheaper, first-order; for a quick look.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=_IMG_KAX,
    description="One window-deformation pass as a node: the previous field predicts, the frames "
                "are deformed and re-correlated at a finer window, and a NEW finer grid replaces "
                "the table. Chain one per rung of the ladder.")


# ── the bundle: the default 64 / 32 / 16 px ladder, validated after every pass ─────

_DX = 250.0
register_bundle(Bundle(
    key="bundle:piv", label="PIV (OpenPIV) · staged", category="analysis",
    role="motion_and_deformation_fields",
    description=
    "OpenPIV's multipass ladder as six cards in one region: First Pass (64 px) → Validate → "
    "Deform Pass (32 px) → Validate → Deform Pass (16 px) → Validate. Pull any card to see "
    "the field as that pass leaves it and what validation refused; change a window and only "
    "the passes after it re-run. Same correlation, validation and replacement as the single "
    "PIV (OpenPIV) node, pass for pass.",
    nodes=(
        BundleNode("piv_pass1", "piv.first_pass", 0.0, 0.0, params={"window_size": 64}),
        BundleNode("piv_check1", "piv.validate", _DX, 0.0),
        BundleNode("piv_pass2", "piv.deform_pass", 2 * _DX, 0.0),
        BundleNode("piv_check2", "piv.validate", 3 * _DX, 0.0),
        BundleNode("piv_pass3", "piv.deform_pass", 4 * _DX, 0.0),
        BundleNode("piv_check3", "piv.validate", 5 * _DX, 0.0),
    ),
    edges=(
        BundleEdge("piv_pass1", "out", "piv_check1", "data"),
        BundleEdge("piv_check1", "out", "piv_pass2", "data"),
        BundleEdge("piv_pass2", "out", "piv_check2", "data"),
        BundleEdge("piv_check2", "out", "piv_pass3", "data"),
        BundleEdge("piv_pass3", "out", "piv_check3", "data"),
    ),
    input=("piv_pass1", "data"), output=("piv_check3", "out"),
    frame_title="PIV (OpenPIV)",
    # optional between-pass smoothing: part of the family, placed only where wanted
    extra_ops=("piv.smooth",),
))
