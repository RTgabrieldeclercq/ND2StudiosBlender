"""Staged DIC (pyALDIC) — one node per section of the AL-DIC pipeline, placed together as
the ``bundle:dic`` node bundle (2026-10-08).

``analysis.dic_correlate`` runs pyALDIC's Sections 2b-8 as one card. These five nodes are
those sections, one each, the way ``al_dic/core/pipeline.py::run_aldic`` is organised:

    dic.grid           §2b the correlation lattice + the reference policy          (no correlation)
    dic.initial_guess  §3  the FFT integer seed per lattice node (auto-expanding search)
    dic.local_match    §4  the 6-DOF IC-GN subset match                            (the cost centre)
    dic.global_step    §5-6 the global FEM compatibility solve + ADMM iterations
    dic.strain         §8  strain from the displacement field                      (no image read)

**The field between stages is a Point table** on one layer (``dic`` by default): the grid
writes the rows (one per lattice node per correlated frame, in pyALDIC's node order —
``node = ix·ny + iy``), every later stage reads it by layer name and adds or overwrites
columns on those same rows (``disp_*`` in µm, ``grad_*``, ``n_iter``, ``hole``, …,
``strain_*``). pyALDIC's own conventions stay inside the kernel; on the wire everything is
this repo's ``(y, x)``: ``disp_y`` / ``disp_x`` in µm as ``analysis.dic_correlate`` writes
them, ``grad_<i><j>`` = ∂(disp_i)/∂(axis_j) in image axes (the two cross terms sign-corrected
from pyALDIC's y-up world frame, as the single node does), ``strain_*`` the same four
components after pyALDIC's strain post-processing.

**Decided once at the grid and inherited** (§7b): subset size / step / minimum element,
channel, ROI layer and the reference policy are stamped as ``dic_*`` provenance; the
local stage stamps its tolerance and iteration cap. **Covered paths**: the FFT seed (what
the single node uses), fixed and previous-frame references, a static ROI mask per
``(m, z)`` line; not covered (the single node keeps them): pre-solve mesh refinement,
seed propagation, an external reference Dataset — chain a separate reference in front
with Timeseries Builder instead.
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

_DEFAULT_LAYER = "dic"
_AX = ("y", "x")
DISP_COLUMNS = ("disp_y", "disp_x", "disp_mag_um")
GRAD_COLUMNS = tuple(f"grad_{i}{j}" for i in _AX for j in _AX)       # yy, yx, xy, xx
STRAIN_COLUMNS = tuple(f"strain_{i}{j}" for i in _AX for j in _AX)
SEED_COLUMNS = DISP_COLUMNS + ("seed_cc",)
LOCAL_COLUMNS = DISP_COLUMNS + GRAD_COLUMNS + ("n_iter", "hole")
GLOBAL_COLUMNS = DISP_COLUMNS + GRAD_COLUMNS
STRAIN_OUT_COLUMNS = STRAIN_COLUMNS + ("strain_valid",)

_GRID_PROV = ("dic_winsize", "dic_winstepsize", "dic_winsize_min", "dic_channel",
              "dic_reference_mode", "dic_reference_frame", "dic_roi", "dic_n_nodes")
_LOCAL_PROV = ("dic_tol", "dic_icgn_max_iter")
_STRAIN_METHOD = {"plane_fit": 2, "fem": 3}
_STRAIN_TYPE = {"infinitesimal": 0, "euler_almansi": 1, "green_lagrange": 2}


# ── shared plumbing ───────────────────────────────────────────────────────────────

def _dic_px(ctx: EvalContext) -> float:
    return float(ctx.calib("pixel_size_um") or 0.1)


def _dic_plan(ax, ref_mode: str, ref_frame: int) -> List[Tuple[int, int]]:
    if ref_mode == "previous_frame":
        return [(t, t - 1) for t in range(1, ax.t)]
    r = min(max(0, int(ref_frame)), max(0, ax.t - 1))
    return [(t, r) for t in range(ax.t)]


def _dic_blocks(ax, plan) -> List[Tuple[int, int, int, int]]:
    return [(m, z, t_out, t_ref) for m in range(ax.m) for z in range(ax.z) for (t_out, t_ref) in plan]


def _dic_plane(prov, m: int, t: int, z: int, c: int) -> np.ndarray:
    pax = prov.axes
    return np.asarray(prov.get_region(0, m, t, z, c, 0, pax.y, 0, pax.x), dtype=np.float64)


def _dic_roi_plane(ds: Dataset, roi_name: str, m: int, z: int, c: int) -> Optional[np.ndarray]:
    if not roi_name:
        return None
    attr = ds.get(Domain.VOXEL, roi_name)
    if attr is None:
        return None
    v = attr.values
    s = v.shape
    return np.asarray(v[min(m, s[0] - 1), 0, min(z, s[2] - 1), min(c, s[3] - 1)]) > 0


def _dic_table(ds: Dataset, layer: str, node: str) -> Dict[str, np.ndarray]:
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT) if a.layer == layer}
    for req in ("id", "m", "t", "c", "z", "y", "x"):
        if req not in cols:
            raise ValueError(f"{node}: Point layer {layer!r} lacks the column {req!r} — run DIC "
                             "Grid upstream (it writes the lattice rows every stage fills)")
    return cols


def _dic_prov(md, node: str, keys=_GRID_PROV) -> Dict[str, Any]:
    out = {}
    for k in keys:
        if k not in md:
            raise ValueError(
                f"{node}: the input carries no {k!r} provenance — this node reads its lattice "
                "and reference policy from DIC Grid's output; wire DIC Grid upstream (the "
                "bundle places the chain in order)")
        out[k] = md[k]
    return out


def _dic_rows(cols, m: int, t: int, z: int, c: int, n: int, node: str) -> np.ndarray:
    idx = np.flatnonzero((cols["m"] == m) & (cols["t"] == t) & (cols["z"] == z) & (cols["c"] == c))
    if len(idx) != n:
        raise ValueError(
            f"{node}: the layer holds {len(idx)} rows for (m={m}, t={t}, z={z}, c={c}) but the "
            f"lattice has {n} nodes — the table was filtered or re-ordered between stages; wire "
            "the stages directly after DIC Grid")
    return idx


def _dic_write(ds: Dataset, layer: str, cols: Dict[str, np.ndarray]) -> Dataset:
    for name, vals in cols.items():
        ds = ds.with_layer(Domain.POINT, name, np.asarray(vals), layer=layer)
    return ds


def _dic_source(ctx: EvalContext, ds: Dataset, node: str, want: str) -> str:
    layer, _note = _resolve_layer(
        _point_layers(ds), want, node=node, socket="source",
        what="DIC lattice table", where="the `data` input",
        remedy="run DIC Grid upstream — it writes the Point table the stages fill", ctx=ctx)
    return layer


def _dic_kernel_params(gp: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    p = {"winsize": int(gp["dic_winsize"]), "winstepsize": int(gp["dic_winstepsize"]),
         "winsize_min": int(gp["dic_winsize_min"]), "init_guess_mode": "fft"}
    p.update(extra)
    return p


def _dic_al_mode(ref_mode: str) -> str:
    return "incremental" if ref_mode == "previous_frame" else "accumulative"


def _dic_grad_image_axes(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """pyALDIC's interleaved ``F`` → ``(dvdy, dvdx, dudy, dudx)`` = ``grad_yy, grad_yx,
    grad_xy, grad_xx`` in image axes; the raw IC-GN gradient is already in image axes
    (no world flip — that happens only inside pyALDIC's strain post-processing)."""
    from nodegraph.kernels.dic_correlate import dic_split_f
    dudx, dvdx, dudy, dvdy = dic_split_f(F)
    return dvdy, dvdx, dudy, dudx


class _dic_RefCache:
    def __init__(self, build):
        self._build = build
        self._d: Dict[int, Any] = {}

    def get(self, t_ref: int):
        if t_ref not in self._d:
            while len(self._d) >= 2:
                self._d.pop(next(iter(self._d)))
            self._d[t_ref] = self._build(t_ref)
        return self._d[t_ref]


def _dic_layers_on_source(params, modes):
    return ((Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER)),)


# ── §2b  DIC Grid ─────────────────────────────────────────────────────────────────

def _compute_dic_grid(ctx: EvalContext) -> Dataset:
    """pyALDIC Section 2b as a node: the correlation lattice for this image under the
    subset size / step (the FE mesh's nodes, in pyALDIC's node order), written as a Point
    table — one row per node per frame to be correlated — plus the channel, ROI and
    reference policy, stamped as ``dic_*`` provenance for every later stage. No image is
    read; the card is instant and shows the measurement density on the Viewer.

    Resolved spec: category analysis; op ``dic.grid``; reads VOXEL (axes + calibration
    only), adds POINT (``z_kind`` plane_index); 2-D per plane; footprint WHOLE_SERIES /
    {t,y,x}. ``winstepsize`` and ``winsize_min`` are snapped to powers of two and
    ``winsize`` to even, as the single node does. Lattice via
    :func:`nodegraph.kernels.dic_correlate.dic_stage_grid` (``build_grid_for_roi``, the same
    lattice ``integer_search`` lays its windows on) and ``mesh_setup`` for the node order."""
    from nodegraph.kernels.dic_correlate import dic_stage_lattice_nodes, dic_stage_para
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("DIC Grid needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    ref_mode = str(modes.get("reference_mode", "fixed_frame"))
    # under the quick-analysis scope a source frame index is not a payload index; the
    # stamped reference is what the later stages index the payload with
    ref_frame = scoped_frame(ds.metadata, int(ctx.params.get("reference_frame", 0)), ax.t,
                             mode=ref_mode, node="DIC Grid")
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    gp = {"winsize": max(2, int(ctx.params.get("winsize", 40))),
          "winstepsize": max(2, int(ctx.params.get("winstepsize", 16))),
          "winsize_min": max(2, int(ctx.params.get("winsize_min", 8)))}
    roi_name = ctx.layer("roi") or ""
    layer = ctx.layer("name")
    shape = (ax.y, ax.x)
    para = dic_stage_para(dict(gp, init_guess_mode="fft"), shape,
                          reference_mode=_dic_al_mode(ref_mode))
    nodes_xy = dic_stage_lattice_nodes(para, shape)                 # (N, 2) [x, y]
    n = int(nodes_xy.shape[0])
    plan = _dic_plan(ax, ref_mode, ref_frame)
    blocks = _dic_blocks(ax, plan)
    total = n * len(blocks)
    cols: Dict[str, np.ndarray] = {
        "id": np.arange(total, dtype=np.int64),
        "m": np.repeat([b[0] for b in blocks], n).astype(np.int64),
        "t": np.repeat([b[2] for b in blocks], n).astype(np.int64),
        "c": np.full(total, c, dtype=np.int64),
        "z": np.repeat([float(b[1]) for b in blocks], n),
        "y": np.tile(nodes_xy[:, 1], len(blocks)), "x": np.tile(nodes_xy[:, 0], len(blocks)),
    }
    if not blocks:
        cols = {k: v[:0] for k, v in cols.items()}
    md = {"dic_winsize": int(para.winsize), "dic_winstepsize": int(para.winstepsize),
          "dic_winsize_min": int(para.winsize_min), "dic_channel": c,
          "dic_reference_mode": ref_mode, "dic_reference_frame": ref_frame,
          "dic_roi": roi_name, "dic_n_nodes": n, "dic_stage": "grid"}
    ctx.progress(1, 1, f"{n} lattice nodes per frame at step {para.winstepsize} px, "
                       f"{len(plan)} frame(s) per plane", frames=ax.t)
    return (ds.with_structure(StructureTable(Domain.POINT, cols, layer=layer, z_kind="plane_index"))
            .with_metadata(**md))


def _columns_dic_grid(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("name") or _DEFAULT_LAYER), POINT_INVARIANT)


# ── §3  DIC Initial Guess ─────────────────────────────────────────────────────────

def _compute_dic_initial_guess(ctx: EvalContext) -> Dataset:
    """pyALDIC Section 3 as a node: the integer seed of every lattice node by FFT
    cross-correlation of its window inside ``search_range`` px, the search widened and
    retried while peaks sit on its edge (pyALDIC's auto-expand, the radius that finally
    covered the motion remembered per reference), cleaned by ``init_disp`` (inpainting,
    three outlier passes). Writes ``disp_*`` (µm) and ``seed_cc`` (the correlation peak).
    Pull it before the match: a seed that points the wrong way is never recovered by
    the local refinement.

    Resolved spec: category analysis; op ``dic.initial_guess``; reads VOXEL + POINT, adds
    POINT; footprint WHOLE_SERIES / {t,y,x}. Inherits lattice, channel, ROI and reference
    policy. Kernel: :func:`nodegraph.kernels.dic_correlate.dic_stage_seed`."""
    from nodegraph.kernels.dic_correlate import (dic_split_u, dic_stage_mesh, dic_stage_normalize,
                                                 dic_stage_para, dic_stage_reference,
                                                 dic_stage_seed, dic_stage_seed_on_mesh)
    node = "DIC Initial Guess"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    layer = _dic_source(ctx, ds, node, ctx.layer("source"))
    cols = _dic_table(ds, layer, node)
    gp = _dic_prov(ds.metadata, node)
    search = max(1, int(ctx.params.get("search_range", 20)))
    px = _dic_px(ctx)
    shape = (ax.y, ax.x)
    n = int(gp["dic_n_nodes"])
    c = int(gp["dic_channel"])
    roi_name = str(gp["dic_roi"] or "")
    ref_mode = str(gp["dic_reference_mode"])
    plan = _dic_plan(ax, ref_mode, int(gp["dic_reference_frame"]))
    n_rows = len(cols["id"])
    disp = np.full((n_rows, 2), np.nan)
    seed_cc = np.full(n_rows, np.nan)
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)), note="seeding")
    search_used = search
    for m in range(ax.m):
        for z in range(ax.z):
            roi = _dic_roi_plane(ds, roi_name, m, z, c)
            para = dic_stage_para(_dic_kernel_params(gp, {"size_of_fft_search_region": search}), shape,
                                  reference_mode=_dic_al_mode(ref_mode), roi_mask=roi)
            refs = _dic_RefCache(lambda t_ref, _m=m, _z=z, _p=para, _r=roi: dic_stage_reference(
                dic_stage_normalize(_dic_plane(prov, _m, t_ref, _z, c), _p), _p, shape, roi_mask=_r))
            learned: Dict[int, int] = {}
            for (t_out, t_ref) in plan:
                ref = refs.get(t_ref)
                idx = _dic_rows(cols, m, t_out, z, c, n, node)
                g = dic_stage_normalize(_dic_plane(prov, m, t_out, z, c), para)
                sd = dic_stage_seed(ref, g, search=learned.get(t_ref))
                learned[t_ref] = int(sd["search"])
                search_used = int(sd["search"])
                mesh = dic_stage_mesh(ref, sd["x0"], sd["y0"])
                u0 = dic_stage_seed_on_mesh(ref, mesh, sd["U0"], sd["x0"], sd["y0"])
                ux, vy = dic_split_u(u0)
                disp[idx, 0] = vy * px
                disp[idx, 1] = ux * px
                cc = np.asarray(sd["cc"], float).reshape(-1)
                seed_cc[idx] = cc if cc.size == n else np.nan
                bar.finish_unit(note=f"m={m} z={z} t={t_out}: search ±{sd['search']} px")
    out = {"disp_y": disp[:, 0], "disp_x": disp[:, 1],
           "disp_mag_um": np.sqrt(np.nansum(disp ** 2, axis=1)), "seed_cc": seed_cc}
    md = {"dic_stage": "initial_guess", "dic_search_range": search, "dic_search_used": search_used}
    return _dic_write(ds, layer, out).with_metadata(**md)


def _columns_dic_initial_guess(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER), SEED_COLUMNS)


# ── §4  DIC Local Match ───────────────────────────────────────────────────────────

def _compute_dic_local_match(ctx: EvalContext) -> Dataset:
    """pyALDIC Section 4 as a node: the 6-DOF inverse-compositional Gauss-Newton match of
    every subset, seeded by ``disp_*`` — the cost centre. Overwrites ``disp_*`` with the
    sub-pixel result, writes the deformation gradient ``grad_*`` (image axes), ``n_iter``
    and ``hole`` (a subset mostly outside the mask, whose match is not trusted). Bad
    points are filled from their neighbours inside the solver, as pyALDIC does.

    Resolved spec: category analysis; op ``dic.local_match``; reads VOXEL + POINT, adds
    POINT; footprint WHOLE_SERIES / {t,y,x}. Stamps ``dic_tol`` / ``dic_icgn_max_iter`` so
    the global step rebuilds the identical local context. Kernel:
    :func:`nodegraph.kernels.dic_correlate.dic_stage_local`."""
    from nodegraph.kernels.dic_correlate import (dic_merge_u, dic_split_u, dic_stage_local,
                                                 dic_stage_mesh, dic_stage_normalize,
                                                 dic_stage_para, dic_stage_reference, dic_stage_grid)
    node = "DIC Local Match"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    layer = _dic_source(ctx, ds, node, ctx.layer("source"))
    cols = _dic_table(ds, layer, node)
    for dn in DISP_COLUMNS[:2]:
        if dn not in cols:
            raise ValueError(f"{node}: the lattice layer {layer!r} carries no {dn!r} — run DIC "
                             "Initial Guess first (the local match refines a seed)")
    gp = _dic_prov(ds.metadata, node)
    tol = float(ctx.params.get("tol", 1e-2))
    max_iter = max(1, int(ctx.params.get("icgn_max_iter", 100)))
    px = _dic_px(ctx)
    shape = (ax.y, ax.x)
    n = int(gp["dic_n_nodes"])
    c = int(gp["dic_channel"])
    roi_name = str(gp["dic_roi"] or "")
    ref_mode = str(gp["dic_reference_mode"])
    plan = _dic_plan(ax, ref_mode, int(gp["dic_reference_frame"]))
    n_rows = len(cols["id"])
    disp_in = np.stack([np.asarray(cols["disp_y"], float), np.asarray(cols["disp_x"], float)], axis=1)
    disp = np.full((n_rows, 2), np.nan)
    grad = np.full((n_rows, 4), np.nan)
    n_iter = np.zeros(n_rows, dtype=np.int64)
    hole = np.zeros(n_rows, dtype=np.int64)
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)), note="matching")
    for m in range(ax.m):
        for z in range(ax.z):
            roi = _dic_roi_plane(ds, roi_name, m, z, c)
            para = dic_stage_para(_dic_kernel_params(gp, {"tol": tol, "icgn_max_iter": max_iter}), shape,
                                  reference_mode=_dic_al_mode(ref_mode), roi_mask=roi)
            x0, y0 = dic_stage_grid(para, shape)
            refs = _dic_RefCache(lambda t_ref, _m=m, _z=z, _p=para, _r=roi: dic_stage_reference(
                dic_stage_normalize(_dic_plane(prov, _m, t_ref, _z, c), _p), _p, shape, roi_mask=_r))
            for (t_out, t_ref) in plan:
                ref = refs.get(t_ref)
                idx = _dic_rows(cols, m, t_out, z, c, n, node)
                mesh = dic_stage_mesh(ref, x0, y0)
                g = dic_stage_normalize(_dic_plane(prov, m, t_out, z, c), para)
                u0 = dic_merge_u(disp_in[idx, 1] / px, disp_in[idx, 0] / px)
                r = dic_stage_local(ref, mesh, g, u0)
                ux, vy = dic_split_u(r["U"])
                disp[idx, 0] = vy * px
                disp[idx, 1] = ux * px
                gyy, gyx, gxy, gxx = _dic_grad_image_axes(r["F"])
                grad[idx] = np.stack([gyy, gyx, gxy, gxx], axis=1)
                ci = np.asarray(r["conv_iter"]).reshape(-1)
                n_iter[idx] = ci.astype(np.int64) if ci.size == n else 0
                h = np.zeros(n, dtype=np.int64)
                h[np.asarray(r["mark_hole_strain"], dtype=np.int64)] = 1
                hole[idx] = h
                bar.finish_unit(note=f"m={m} z={z} t={t_out}: {n} subsets in {r['solve_time']:.1f} s, "
                                     f"{len(r['mark_hole_strain'])} holes")
    out = {"disp_y": disp[:, 0], "disp_x": disp[:, 1], "disp_mag_um": np.sqrt(np.nansum(disp ** 2, axis=1))}
    for k, name in enumerate(GRAD_COLUMNS):
        out[name] = grad[:, k]
    out["n_iter"] = n_iter
    out["hole"] = hole
    md = {"dic_stage": "local_match", "dic_tol": tol, "dic_icgn_max_iter": max_iter}
    return _dic_write(ds, layer, out).with_metadata(**md)


def _columns_dic_local_match(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER), LOCAL_COLUMNS)


# ── §5-6  DIC Global Step ─────────────────────────────────────────────────────────

def _compute_dic_global_step(ctx: EvalContext) -> Dataset:
    """pyALDIC Sections 5-6 as a node: the global FEM compatibility solve and the ADMM
    iterations that make AL-DIC more than local DIC — the matched subsets (``disp_*`` /
    ``grad_*`` / ``hole``) reconciled into one kinematically compatible field, β by the
    L-curve once per reference unless pinned, the displacement and gradient smoothed as
    ``disp_smoothness`` / ``strain_smoothness`` ask (pyALDIC's own post-solve corrections
    included). Overwrites ``disp_*`` and ``grad_*``.

    Resolved spec: category analysis; op ``dic.global_step``; reads VOXEL + POINT, adds
    POINT; footprint WHOLE_SERIES / {t,y,x} (the local re-solves read the frames again).
    Inherits the lattice, reference AND local provenance. Kernel:
    :func:`nodegraph.kernels.dic_correlate.dic_stage_global` — the driver's
    `pipeline.py:1465-1673` line for line."""
    from nodegraph.kernels.dic_correlate import (dic_merge_f, dic_merge_u, dic_split_u,
                                                 dic_stage_global, dic_stage_grid, dic_stage_mesh,
                                                 dic_stage_normalize, dic_stage_para,
                                                 dic_stage_reference)
    from nodegraph.kernels.dic_correlate import _al_dic_modules
    node = "DIC Global Step"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    layer = _dic_source(ctx, ds, node, ctx.layer("source"))
    cols = _dic_table(ds, layer, node)
    for dn in DISP_COLUMNS[:2] + GRAD_COLUMNS + ("hole",):
        if dn not in cols:
            raise ValueError(f"{node}: the lattice layer {layer!r} carries no {dn!r} — run DIC "
                             "Local Match first (the global step reconciles its subsets)")
    gp = _dic_prov(ds.metadata, node)
    lp = _dic_prov(ds.metadata, node, _LOCAL_PROV)
    p_extra = {"tol": float(lp["dic_tol"]), "icgn_max_iter": int(lp["dic_icgn_max_iter"]),
               "mu": float(ctx.params.get("mu", 1e-3)),
               "admm_max_iter": max(1, int(ctx.params.get("admm_max_iter", 3))),
               "disp_smoothness": max(0.0, float(ctx.params.get("disp_smoothness", 5e-4))),
               "strain_smoothness": max(0.0, float(ctx.params.get("strain_smoothness", 1e-5))),
               "use_global_step": True}
    beta_param = max(0.0, float(ctx.params.get("beta", 0.0)))
    px = _dic_px(ctx)
    shape = (ax.y, ax.x)
    n = int(gp["dic_n_nodes"])
    c = int(gp["dic_channel"])
    roi_name = str(gp["dic_roi"] or "")
    ref_mode = str(gp["dic_reference_mode"])
    plan = _dic_plan(ax, ref_mode, int(gp["dic_reference_frame"]))
    n_rows = len(cols["id"])
    disp_in = np.stack([np.asarray(cols["disp_y"], float), np.asarray(cols["disp_x"], float)], axis=1)
    grad_in = np.stack([np.asarray(cols[g], float) for g in GRAD_COLUMNS], axis=1)
    hole_in = np.asarray(cols["hole"]).astype(bool)
    disp = disp_in.copy()
    grad = grad_in.copy()
    last_beta = None
    regions_mod = _al_dic_modules()["regions"]
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)), note="ADMM")
    for m in range(ax.m):
        for z in range(ax.z):
            roi = _dic_roi_plane(ds, roi_name, m, z, c)
            para = dic_stage_para(_dic_kernel_params(gp, p_extra), shape,
                                  reference_mode=_dic_al_mode(ref_mode), roi_mask=roi)
            if beta_param > 0:
                from dataclasses import replace
                para = replace(para, beta=beta_param)
            x0, y0 = dic_stage_grid(para, shape)
            refs = _dic_RefCache(lambda t_ref, _m=m, _z=z, _p=para, _r=roi: dic_stage_reference(
                dic_stage_normalize(_dic_plane(prov, _m, t_ref, _z, c), _p), _p, shape, roi_mask=_r))
            beta_by_ref: Dict[int, float] = {}
            for (t_out, t_ref) in plan:
                ref = refs.get(t_ref)
                idx = _dic_rows(cols, m, t_out, z, c, n, node)
                mesh = dic_stage_mesh(ref, x0, y0)
                g = dic_stage_normalize(_dic_plane(prov, m, t_out, z, c), para)
                u1 = dic_merge_u(np.nan_to_num(disp_in[idx, 1]) / px, np.nan_to_num(disp_in[idx, 0]) / px)
                gyy, gyx, gxy, gxx = (np.nan_to_num(grad_in[idx, k]) for k in range(4))
                f1 = dic_merge_f(gxx, gyx, gxy, gyy)        # (dudx, dvdx, dudy, dvdy)
                mhs = np.flatnonzero(hole_in[idx]).astype(np.int64)
                regions = regions_mod.precompute_node_regions(mesh.coordinates_fem, ref.f_mask, shape)
                r = dic_stage_global(ref, mesh, g, u1, f1, mhs, regions, beta=beta_by_ref.get(t_ref))
                beta_by_ref[t_ref] = float(r["beta"])
                last_beta = float(r["beta"])
                ux, vy = dic_split_u(r["U"])
                disp[idx, 0] = vy * px
                disp[idx, 1] = ux * px
                gyy, gyx, gxy, gxx = _dic_grad_image_axes(r["F"])
                grad[idx] = np.stack([gyy, gyx, gxy, gxx], axis=1)
                bar.finish_unit(note=f"m={m} z={z} t={t_out}: β={r['beta']:.3g}, {r['n_steps']} ADMM step(s)")
    out = {"disp_y": disp[:, 0], "disp_x": disp[:, 1], "disp_mag_um": np.sqrt(np.nansum(disp ** 2, axis=1))}
    for k, name in enumerate(GRAD_COLUMNS):
        out[name] = grad[:, k]
    md = {"dic_stage": "global_step", "dic_mu": p_extra["mu"],
          "dic_beta": (last_beta if last_beta is not None else beta_param),
          "dic_admm_max_iter": p_extra["admm_max_iter"],
          "dic_disp_smoothness": p_extra["disp_smoothness"],
          "dic_strain_smoothness": p_extra["strain_smoothness"], "dic_solver": "aldic"}
    return _dic_write(ds, layer, out).with_metadata(**md)


def _columns_dic_global_step(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER), GLOBAL_COLUMNS)


# ── §8  DIC Strain ────────────────────────────────────────────────────────────────

def _compute_dic_strain(ctx: EvalContext) -> Dataset:
    """pyALDIC Section 8 as a node: the nodal displacement gradient from the field on the
    lattice — plane fit over ``strain_plane_fit_rad`` px or the FEM gradient — under the
    chosen strain measure, with pyALDIC's edge trimming (``strain_valid``). Writes the four
    ``strain_<i><j>`` columns in image axes (the cross terms sign-corrected from pyALDIC's
    y-up world frame, exactly as the single node does) and ``strain_valid``. No image is
    read beyond the ROI mask: the lattice is rebuilt from provenance, so a new measure or
    radius re-runs in a second with the correlation untouched.

    Resolved spec: category analysis; op ``dic.strain``; reads POINT (+ the optional ROI
    Voxel mask), adds POINT; footprint WHOLE_SERIES with empty kernel_axes. Stamps
    ``dic_strain_method`` / ``dic_strain_type``. Kernel:
    :func:`nodegraph.kernels.dic_correlate.dic_stage_strain`."""
    from nodegraph.kernels.dic_correlate import (_strain_components, dic_merge_u, dic_stage_grid,
                                                 dic_stage_mesh, dic_stage_para,
                                                 dic_stage_reference, dic_stage_strain)
    node = "DIC Strain"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset (for its axes)")
    ax = prov.axes
    layer = _dic_source(ctx, ds, node, ctx.layer("source"))
    cols = _dic_table(ds, layer, node)
    for dn in DISP_COLUMNS[:2]:
        if dn not in cols:
            raise ValueError(f"{node}: the lattice layer {layer!r} carries no {dn!r} — run DIC "
                             "Local Match (or at least DIC Initial Guess) first")
    gp = _dic_prov(ds.metadata, node)
    modes = ctx.params.get("__modes__", {})
    method = str(modes.get("strain_method", "plane_fit"))
    stype = str(modes.get("strain_type", "infinitesimal"))
    rad = max(1.0, float(ctx.params.get("strain_plane_fit_rad", 20.0)))
    smooth = max(0.0, float(ctx.params.get("smoothness", 0.0)))
    px = _dic_px(ctx)
    shape = (ax.y, ax.x)
    n = int(gp["dic_n_nodes"])
    c = int(gp["dic_channel"])
    roi_name = str(gp["dic_roi"] or "")
    ref_mode = str(gp["dic_reference_mode"])
    plan = _dic_plan(ax, ref_mode, int(gp["dic_reference_frame"]))
    n_rows = len(cols["id"])
    disp_in = np.stack([np.asarray(cols["disp_y"], float), np.asarray(cols["disp_x"], float)], axis=1)
    strain = np.full((n_rows, 4), np.nan)
    valid = np.zeros(n_rows, dtype=np.int64)
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=max(1, len(plan)), note="strain")
    for m in range(ax.m):
        for z in range(ax.z):
            roi = _dic_roi_plane(ds, roi_name, m, z, c)
            para = dic_stage_para(_dic_kernel_params(gp, {
                "method_to_compute_strain": _STRAIN_METHOD.get(method, 2),
                "strain_type": _STRAIN_TYPE.get(stype, 0), "strain_plane_fit_rad": rad,
                "smoothness": smooth}), shape, reference_mode=_dic_al_mode(ref_mode), roi_mask=roi)
            from dataclasses import replace
            para = replace(para, method_to_compute_strain=_STRAIN_METHOD.get(method, 2),
                           strain_type=_STRAIN_TYPE.get(stype, 0), strain_plane_fit_rad=rad,
                           smoothness=smooth)
            x0, y0 = dic_stage_grid(para, shape)
            # the strain reads no frame: a blank, mask-only reference carries the mask + para
            blank = np.zeros(shape, dtype=np.float64)
            ref = dic_stage_reference(blank, para, shape, roi_mask=roi)
            mesh = dic_stage_mesh(ref, x0, y0)
            for (t_out, t_ref) in plan:
                idx = _dic_rows(cols, m, t_out, z, c, n, node)
                u = dic_merge_u(np.nan_to_num(disp_in[idx, 1]) / px, np.nan_to_num(disp_in[idx, 0]) / px)
                sr = dic_stage_strain(ref, mesh, u, deformed_mask=roi)
                comps = _strain_components(sr)
                if comps is not None:
                    dudx, dudy, dvdx, dvdy = comps
                    strain[idx] = np.stack([dvdy, dvdx, dudy, dudx], axis=1)
                sv = np.asarray(getattr(sr, "strain_valid", np.ones(n, bool)), dtype=bool).reshape(-1)
                valid[idx] = sv.astype(np.int64) if sv.size == n else 1
                bar.finish_unit(note=f"m={m} z={z} t={t_out}: {int(valid[idx].sum())}/{n} nodes with strain")
    out: Dict[str, np.ndarray] = {}
    for k, name in enumerate(STRAIN_COLUMNS):
        out[name] = strain[:, k]
    out["strain_valid"] = valid
    md = {"dic_stage": "strain", "dic_strain_method": method, "dic_strain_type": stype,
          "dic_strain_plane_fit_rad": rad}
    return _dic_write(ds, layer, out).with_metadata(**md)


def _columns_dic_strain(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER),
                    STRAIN_OUT_COLUMNS)


# ── registration ─────────────────────────────────────────────────────────────────

_SOURCE = InString(
    "source", "Lattice layer", field=False, default=_DEFAULT_LAYER, layer_in=Domain.POINT,
    description=
    "Which Point table holds the DIC lattice this stage fills in — the one DIC Grid wrote "
    "(`dic` unless renamed there). The stage reads its rows and the previous stages' columns "
    "and writes its own columns BACK ONTO THE SAME ROWS. With a single Point table on the wire "
    "the name is resolved for you.")

_STAGE_GRAN = Granularity.WHOLE_SERIES
_IMG_KAX = frozenset({"t", "y", "x"})

register_node(
    _compute_dic_grid, op_key="dic.grid", label="DIC Grid", category="analysis",
    adds_columns=_columns_dic_grid,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InInt("channel", "Channel", unit="", field=False, default=0, pick_kind="channel",
              description=
              "Which channel every stage correlates (0-based) — decided here and inherited. "
              "Displacement comes from image texture: pick the channel with stable, "
              "high-contrast speckle, not a reporter whose brightness changes biologically."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame", available_in={"reference_mode": frozenset({"fixed_frame"})},
              description=
              "Which timepoint is the undeformed state (0-based) under a fixed reference: every "
              "frame is correlated against it, displacements are cumulative, its own rows are "
              "the zero field. Inert under `previous_frame`. For a separate reference "
              "acquisition chain it in front with Timeseries Builder and keep this at 0."),
        InInt("winsize", "Subset size", unit="px", field=False, default=40,
              pick_kind="grid", pick_peer="winstepsize",
              description=
              "Edge of the correlation subset in pixels — the speckle patch matched between "
              "frames (snapped to even). LARGER lowers noise and smooths; SMALLER resolves "
              "gradients but needs enough speckle inside (3-5 speckles across). Inherited by "
              "every stage."),
        InInt("winstepsize", "Subset step", unit="px", field=False, default=16,
              pick_kind="grid", pick_peer="winsize",
              description=
              "Spacing of the lattice nodes in pixels — THE grid density and the dominant cost "
              "lever (snapped to a power of two: 4, 8, 16, 32, 64). The FE mesh for the global "
              "step is built on these nodes, so it is also the element size of the compatibility "
              "solve."),
        InInt("winsize_min", "Min element", unit="px", field=False, default=8,
              description=
              "The smallest element pyALDIC's mesh may use (a power of two, at most the step); "
              "only matters when a refinement policy subdivides the mesh, which the staged chain "
              "does not run — kept so the lattice matches the single node's."),
        InString("roi", "ROI layer", field=False, default="roi_mask", layer_in=Domain.VOXEL,
                 description=
                 "Optional Voxel mask (ROI Mask / a Threshold) restricting correlation: elements "
                 "mostly outside it are dropped from the mesh, their nodes seeded NaN and left to "
                 "the inpainting, and the lattice is tightened to the mask's bounding box. Static "
                 "in time (the t=0 plane). Absent → the whole frame."),
        InString("name", "Lattice layer", field=False, default=_DEFAULT_LAYER,
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point table this node writes: one row per lattice node per "
                 "correlated frame, positions only until the later stages fill `disp_*`, "
                 "`grad_*`, `strain_*`. Every stage after this one selects it by this name."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("reference_mode", ["fixed_frame", "previous_frame"], default="fixed_frame",
             label="Reference",
             description="How frames pair up — decided once here and inherited by every stage.",
             choice_docs={
                 "fixed_frame": "Every timepoint against Reference frame: cumulative "
                                "displacement from that state, comparable across time, "
                                "until the deformation outgrows what a subset can follow.",
                 "previous_frame": "Each timepoint against the one before it: small "
                                   "increments that follow large total deformation; no "
                                   "row for t=0; compose increments downstream if totals "
                                   "are wanted.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=_IMG_KAX,
    description="pyALDIC §2b as a node: the correlation lattice (subset size / step) as a Point "
                "table + the channel, ROI and reference policy every later DIC stage inherits. "
                "Instant — nothing is correlated. First card of the staged DIC bundle.")

register_node(
    _compute_dic_initial_guess, op_key="dic.initial_guess", label="DIC Initial Guess",
    category="analysis", adds_columns=_columns_dic_initial_guess, extra_layers=_dic_layers_on_source,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InInt("search_range", "Search range", unit="px", field=False, default=20,
              description=
              "± half-range in pixels of the FFT integer search that seeds every node. It must "
              "cover the largest per-pair displacement or the match starts on the wrong speckle; "
              "pyALDIC widens it automatically while peaks land on its edge and remembers the "
              "radius that worked, and shrinks it when the image cannot hold it. Arrows pointing "
              "the wrong way on THIS card are the symptom of a radius too small."),
    ],
    outputs=[OutDataset()],
    granularity=_STAGE_GRAN, kernel_axes=_IMG_KAX,
    description="pyALDIC §3 as a node: the FFT integer seed of every lattice node (auto-expanding "
                "search, outlier-cleaned) written as disp_* with its correlation peak. Pull it "
                "first: a wrong seed is never recovered by the refinement.")

register_node(
    _compute_dic_local_match, op_key="dic.local_match", label="DIC Local Match",
    category="analysis", adds_columns=_columns_dic_local_match, extra_layers=_dic_layers_on_source,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InInt("icgn_max_iter", "Max iterations", unit="", field=False, default=100,
              description=
              "Hard cap on inverse-compositional Gauss-Newton iterations per subset — a safety "
              "valve: a subset that hits it is treated as a bad point and filled from its "
              "neighbours. Raise it only when many `n_iter` sit at the cap."),
        InFloat("tol", "Tolerance", unit="", field=False, default=1e-2,
                description=
                "Convergence tolerance of the IC-GN update (pixels): the iteration stops once "
                "the step is below it. SMALLER is more precise and slower; pyALDIC's default "
                "0.01 is below the method's own precision floor on real speckle."),
    ],
    outputs=[OutDataset()],
    granularity=_STAGE_GRAN, kernel_axes=_IMG_KAX,
    description="pyALDIC §4 as a node: the 6-DOF IC-GN match of every subset from the seed — the "
                "cost centre. Writes sub-pixel disp_*, the gradient grad_*, n_iter and hole.")

register_node(
    _compute_dic_global_step, op_key="dic.global_step", label="DIC Global Step",
    category="analysis", adds_columns=_columns_dic_global_step, extra_layers=_dic_layers_on_source,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InFloat("mu", "AL penalty μ", unit="", field=False, default=1e-3,
                description=
                "Augmented-Lagrangian penalty weighting the constraint that each subset's own "
                "displacement equals the globally compatible field. HIGHER enforces agreement "
                "harder (smoother; a real discontinuity is smeared); LOWER lets subsets stay "
                "independent. pyALDIC's default is 1e-3; change it by factors of ten."),
        InFloat("beta", "Compatibility β", unit="", field=False, default=0.0,
                description=
                "Weight of the compatibility term of the global FEM solve. 0 = AUTO: pyALDIC "
                "sweeps candidates and fits the L-curve once per reference, reusing it for every "
                "frame against that reference. A positive value pins it."),
        InInt("admm_max_iter", "ADMM iterations", unit="", field=False, default=3,
              description=
              "Passes of the outer loop alternating the local re-solve with the global solve; "
              "stops early once the update falls under pyALDIC's tolerance. 1 runs the global "
              "solve once with no re-solve. MORE costs one local pass each."),
        InFloat("disp_smoothness", "Displacement smoothing", unit="", field=False, default=5e-4,
                description=
                "pyALDIC's displacement smoothing weight applied to the local result before and "
                "after each global solve (σ scales with the local node spacing); 0 is off. "
                "LARGER flattens the field. The single node's default is 5e-4."),
        InFloat("strain_smoothness", "Gradient smoothing", unit="", field=False, default=1e-5,
                description=
                "The same for the deformation gradient handed to the global solve; 0 is off. "
                "The single node's default is 1e-5."),
    ],
    outputs=[OutDataset()],
    granularity=_STAGE_GRAN, kernel_axes=_IMG_KAX,
    description="pyALDIC §5-6 as a node: the global FEM compatibility solve + ADMM iterations that "
                "reconcile the matched subsets into one compatible field, β auto-tuned per "
                "reference. Overwrites disp_* and grad_*.")

register_node(
    _compute_dic_strain, op_key="dic.strain", label="DIC Strain", category="analysis",
    adds_columns=_columns_dic_strain, extra_layers=_dic_layers_on_source,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InFloat("strain_plane_fit_rad", "Fit radius", unit="px", field=False, default=20.0,
                description=
                "Radius in pixels of the neighbourhood the plane-fit method differentiates over. "
                "WIDER averages more nodes (lower strain noise, gradients smeared over it); keep "
                "it above the lattice step so every node has neighbours. Only read by `plane_fit`."),
        InFloat("smoothness", "Pre-smoothing", unit="", field=False, default=0.0,
                description=
                "pyALDIC's displacement smoothing applied three times before differentiation; "
                "0 is off and right by default — the displacement is what was measured."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("strain_method", ["plane_fit", "fem"], default="plane_fit", label="Method",
             description="How the displacement gradient is obtained from the lattice field.",
             choice_docs={
                 "plane_fit": "A least-squares plane through each node's neighbourhood (Fit "
                              "radius wide): robust to a noisy node, pyALDIC's default.",
                 "fem": "The FEM gradient on the lattice's elements, area-averaged with an "
                        "outlier pass: consistent with the global solve, sharper near edges.",
             }),
        Mode("strain_type", ["infinitesimal", "green_lagrange", "euler_almansi"],
             default="infinitesimal", label="Measure",
             description="Which strain measure pyALDIC applies to the gradient before reporting.",
             choice_docs={
                 "infinitesimal": "The small-strain measure, linear in the gradient; right "
                                  "under a few percent of strain.",
                 "green_lagrange": "Finite strain referred to the undeformed frame — the "
                                   "Lagrangian choice for large deformations.",
                 "euler_almansi": "Finite strain referred to the DEFORMED frame — the "
                                  "Eulerian choice, reading strain where the material "
                                  "now is rather than where it started.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=frozenset(),
    description="pyALDIC §8 as a node: the nodal displacement gradient under the chosen measure — "
                "four strain_* columns in image axes + strain_valid — without reading a frame, so "
                "a new measure or radius re-runs in a second.")


# ── the bundle: what the palette places ─────────────────────────────────────────

_DX = 270.0
register_bundle(Bundle(
    key="bundle:dic", label="DIC (pyALDIC) · staged", category="analysis",
    role="motion_and_deformation_fields",
    description=
    "pyALDIC as five cards, wired in order and framed as one region: Grid → Initial Guess → "
    "Local Match → Global Step → Strain. Pull any card to run the pipeline up to that section "
    "and see the field as it stands; change a later card and only that section re-runs. The "
    "same solver as the single DIC (pyALDIC) node, section for section (FFT seed, no mesh "
    "refinement).",
    nodes=(
        BundleNode("dic_grid", "dic.grid", 0.0, 0.0),
        BundleNode("dic_seed", "dic.initial_guess", _DX, 0.0),
        BundleNode("dic_local", "dic.local_match", 2 * _DX, 0.0),
        BundleNode("dic_global", "dic.global_step", 3 * _DX, 0.0),
        BundleNode("dic_strain", "dic.strain", 4 * _DX, 0.0),
    ),
    edges=(
        BundleEdge("dic_grid", "out", "dic_seed", "data"),
        BundleEdge("dic_seed", "out", "dic_local", "data"),
        BundleEdge("dic_local", "out", "dic_global", "data"),
        BundleEdge("dic_global", "out", "dic_strain", "data"),
    ),
    input=("dic_grid", "data"), output=("dic_strain", "out"),
    frame_title="DIC (pyALDIC)",
))
