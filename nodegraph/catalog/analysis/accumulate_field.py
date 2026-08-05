"""Accumulate DVC Field (``analysis.accumulate_field``) — Compose a previous-frame DVC increment series into cumulative displacement + strain fields by Lagrangian point-tracking (ported ALDVC build_accumulated_results);."""

from __future__ import annotations

import numpy as np

from typing import Dict, Optional

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, OutDataset
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dvc import _dvc_rows
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer

def _layers_accumulate_field(params, modes):
    """Output name is DERIVED: empty `name` means `f"{source}_cumulative"`."""
    src = params.get("source") or "dvc"
    return ((Domain.POINT, params.get("name") or "%s_cumulative" % src),)
# ── Accumulate incremental DVC increments → cumulative Lagrangian fields ───────
#
# Ports the vendored ALDVC accumulation (`build_accumulated_results` / `accumulate_
# incremental`). A `analysis.dvc_field` run in `previous_frame` mode emits per-step
# increment fields (frame t−1 → t) as a Point series. FranckLab's incremental workflow
# then COMPOSES them into cumulative displacement-from-reference by Lagrangian point-
# tracking (advect the reference grid points through the increments, interpolating each
# increment at the points' current drifted position) and RECOMPUTES strain from the
# cumulative displacement — NOT a naive per-grid-point sum, and NOT the same result as a
# direct fixed-frame correlation. This node is that post-process. Metadata intelligence:
# it inherits the reference config (mode / dim / strain measure) from the upstream DVC
# node's stamped provenance — no redundant reference-frame control — and refuses a field
# that is already cumulative (fixed_frame / external reference).


def _reconstruct_grid(coords: np.ndarray, ndim: int):
    """Rebuild the ALDVC :class:`Grid` from a Point layer's subset-center coords
    ``(N, ndim)`` (voxels, one timepoint). The DVC grid is regular, so the per-axis
    unique sorted coordinates recover ``axes``/``grid_shape``/``step``; ``coords`` is the
    ``ij`` meshgrid. Requires a full grid (``N == prod(grid_shape)``)."""
    from nodegraph.kernels.aldvc_field import Grid
    axes = [np.unique(coords[:, a]).astype(np.float64) for a in range(ndim)]
    grid_shape = tuple(int(len(a)) for a in axes)
    if int(np.prod(grid_shape)) != coords.shape[0]:
        raise ValueError(
            f"accumulate field: {coords.shape[0]} points do not form a full regular "
            f"{grid_shape} subset grid — is the source a DVC field?")
    mesh = np.meshgrid(*axes, indexing="ij")
    coords_arr = np.stack(mesh, axis=-1).astype(np.float64)          # (*grid_shape, ndim)
    step = np.asarray([float(np.median(np.diff(a))) if len(a) > 1 else 1.0
                       for a in axes], dtype=np.float64)
    return Grid(axes=axes, coords=coords_arr, grid_shape=grid_shape, step=step, ndim=ndim)
def _place_on_grid(coords: np.ndarray, values: np.ndarray, grid) -> np.ndarray:
    """Scatter per-point ``values`` (``(N,)`` or ``(N, k)``) onto the grid in the kernel's
    C-order, mapping each point to its cell via a per-axis ``searchsorted`` (robust to row
    ordering). Returns ``(*grid_shape[, k])``."""
    idx = tuple(np.searchsorted(grid.axes[a], coords[:, a]) for a in range(grid.ndim))
    if values.ndim == 1:
        out = np.zeros(grid.grid_shape, dtype=np.float64)
        out[idx] = values
        return out
    out = np.zeros((*grid.grid_shape, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        out[(*idx, k)] = values[:, k]
    return out
def _compute_accumulate_field(ctx: EvalContext) -> Dataset:
    """Compose a `previous_frame` DVC increment series into cumulative displacement +
    strain fields (Lagrangian point-tracking). Reads the increment Point layer, rebuilds
    the subset grid per correlation series (``(m,c)`` in 3D; ``(m,c,z-plane)`` in 2D),
    runs :func:`accumulate_incremental` over ascending ``t``, recomputes strain from the
    cumulative displacement with :func:`compute_strain`, and emits a cumulative Point
    layer with the same schema (disp µm / strain / qfactor).

    Resolved spec (V2.06 addendum): category analysis; op ``analysis.accumulate_field``;
    reads POINT, adds POINT; footprint WHOLE_SERIES (crosses T to compose). **Metadata
    intelligence** — dim / reference-mode / strain measure are inherited from the upstream
    DVC node's stamped provenance (``dvc_dim`` / ``dvc_reference_mode`` / ``dvc_strain_*``),
    so there is no redundant reference control and an already-cumulative field (fixed_frame
    or external reference) is refused. Displacement voxels↔µm via ``ctx.calib`` (the DVC
    output carries the same series calibration). Kernel:
    :func:`nodegraph.kernels.aldvc_field.accumulate_incremental` / ``compute_strain``."""
    from types import SimpleNamespace
    from nodegraph.kernels.aldvc_field import accumulate_incremental, compute_strain
    ds = ctx.inputs[0]
    # the one Point field on the wire, whatever it is called (`_resolve_layer`) — the literal
    # default agreed only with `analysis.dvc_field`'s own default name
    source, _note = _resolve_layer(
        _point_layers(ds), ctx.layer("source"), node="accumulate field", socket="source",
        what="Point table", where="the `data` input",
        remedy="run analysis.dvc_field in previous_frame mode upstream — that is the "
               "increment series this composes", ctx=ctx)
    out_layer = ctx.layer("name") or f"{source}_cumulative"
    md = ds.metadata
    ref_mode = md.get("dvc_reference_mode")
    if ref_mode is None:
        raise ValueError(
            "accumulate field expects a DVC-produced Point field (missing dvc provenance); "
            "run analysis.dvc_field in previous_frame mode upstream")
    if ref_mode != "previous_frame":
        raise ValueError(
            f"accumulate field needs previous_frame increments, but the input DVC field was "
            f"correlated in {ref_mode!r} mode (already cumulative — nothing to accumulate)")
    # Dimensionality inherited from the field's generic z_kind provenance (§7b), not a lever.
    is_3d = ds.structure_zkind(Domain.POINT, source) == "subpixel"
    ndim = 3 if is_3d else 2
    strain_type = md.get("dvc_strain_type", "infinitesimal")
    strain_smooth = float(md.get("dvc_strain_smooth", 0.0))
    px = ctx.calib("pixel_size_um") or 0.1
    if is_3d:
        zs = ctx.calib("z_step_um") or 0.5
        vox = np.asarray([zs, px, px], dtype=np.float64)
    else:
        vox = np.asarray([px, px], dtype=np.float64)

    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:
        raise ValueError(f"accumulate field needs a Point layer {source!r} "
                         "(run a previous_frame DVC node first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    disp_cols = ("disp_z", "disp_y", "disp_x") if is_3d else ("disp_y", "disp_x")
    for dn in disp_cols:
        if dn not in col:
            raise ValueError(f"Point layer {source!r} missing displacement column {dn!r} "
                             f"(is it a {ndim}D DVC field?)")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all = col["z"].astype(float); y_all = col["y"].astype(float); x_all = col["x"].astype(float)
    qf_all = col.get("qfactor")

    def _series(sel_series: np.ndarray, z_plane: Optional[int]) -> list:
        """Accumulate one correlation series (a boolean mask over all rows sharing the same
        (m,c[,z-plane])), emitting cumulative rows per timepoint. ``z_plane`` stamps the 2D
        plane index (None in 3D, where z is a grid coordinate)."""
        ts = sorted(set(t_all[sel_series].tolist()))
        if not ts:
            return []
        # coords of the (identical) subset grid — take the earliest timepoint present.
        first = sel_series & (t_all == ts[0])
        gc = (np.column_stack([z_all[first], y_all[first], x_all[first]]) if is_3d
              else np.column_stack([y_all[first], x_all[first]]))
        grid = _reconstruct_grid(gc, ndim)
        m_i = int(m_all[first][0]); c_i = int(c_all[first][0])
        # increments (voxels) per t, placed onto the grid in kernel C-order.
        incr = []
        qf_by_t: Dict[int, np.ndarray] = {}
        for t in ts:
            selt = sel_series & (t_all == t)
            ct = (np.column_stack([z_all[selt], y_all[selt], x_all[selt]]) if is_3d
                  else np.column_stack([y_all[selt], x_all[selt]]))
            disp_um = np.column_stack([col[dn][selt] for dn in disp_cols])   # (n, ndim)
            disp_vox = disp_um / vox                                          # µm → voxels
            u_grid = np.moveaxis(_place_on_grid(ct, disp_vox, grid), -1, 0)   # (ndim,*grid)
            incr.append((int(t), u_grid))
            if qf_all is not None:
                qf_by_t[int(t)] = _place_on_grid(ct, qf_all[selt].astype(float),
                                                 grid).reshape(-1)
        accum = accumulate_incremental(grid, incr)                            # cumulative
        gc_flat = grid.coords_flat()
        out_rows = []
        for t, u_acc in accum:                                               # (ndim,*grid)
            _F, strain = compute_strain(u_acc, grid.step, voxel_size=vox,
                                        strain_type=strain_type,
                                        smooth_sigma=strain_smooth)
            r = SimpleNamespace(
                dim=ndim, grid_coords=gc_flat,
                displacement_field=np.moveaxis(u_acc, 0, -1),                 # (*grid,ndim) vox
                strain_field=np.moveaxis(strain, (0, 1), (-2, -1)),          # (*grid,ndim,ndim)
                qfactor=qf_by_t.get(t))
            out_rows.append(_dvc_rows(r, vox, m=m_i, t=t, c=c_i, z_plane=z_plane))
        return out_rows

    rows: list = []
    for m in sorted(set(m_all.tolist())):
        for c in sorted(set(c_all[m_all == m].tolist())):
            base = (m_all == m) & (c_all == c)
            if is_3d:
                rows.extend(_series(base, None))
            else:
                for z in sorted(set(np.rint(z_all[base]).astype(int).tolist())):
                    rows.extend(_series(base & (np.rint(z_all).astype(int) == z), z))

    zk = "subpixel" if is_3d else "plane_index"        # → auto-stamped z_kind provenance
    prov_md = {"dvc_reference_mode": "cumulative",
               "dvc_strain_type": strain_type, "dvc_strain_smooth": strain_smooth}
    if not rows:
        empty = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=out_layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)
    return (ds.with_structure(StructureTable(Domain.POINT, merged, layer=out_layer, z_kind=zk))
            .with_metadata(**prov_md))
register_node(
    _compute_accumulate_field, op_key="analysis.accumulate_field",
    label="Accumulate DVC Field", category="analysis",
    extra_layers=_layers_accumulate_field,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(),
            InString("source", "Increment layer", field=False, default="dvc",
                     layer_in=Domain.POINT,
                     description=
                     "Which per-frame DVC INCREMENT field to compose into a cumulative one. It "
                     "must come from a DVC node run against the PREVIOUS frame — incremental "
                     "measurements are what this node sums. Feeding it a fixed-reference field "
                     "is refused rather than silently double-counting, since those values are "
                     "already cumulative. The reference configuration is inherited from the "
                     "upstream DVC node's provenance, not re-specified here."),
            InString("name", "Output layer", field=False, default="",
                     description=
                     "Name of the cumulative Point field this node writes, with the same schema "
                     "as the increment it came from. EMPTY (the default) derives it from the "
                     "source layer's name so the pair stays recognisable. The values are "
                     "totals-since-the-start obtained by Lagrangian point tracking — each point "
                     "is followed through the series, so they are NOT a plain sum of the "
                     "per-frame numbers at fixed grid positions.")],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="Compose a previous-frame DVC increment series into cumulative "
                "displacement + strain fields by Lagrangian point-tracking (ported ALDVC "
                "build_accumulated_results); inherits the reference config from the upstream "
                "DVC node; emits a cumulative Point layer with the same schema.")
