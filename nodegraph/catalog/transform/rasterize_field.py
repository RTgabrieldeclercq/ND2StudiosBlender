"""Rasterize Field (``transform.rasterize_field``) — Interpolate a Point vector/scalar field…"""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer

# ── Rasterize a Point vector field → Voxel layers (interpolate the coarse grid) ─
#
# The DVC output (and any Point structure carrying scalar attribute columns) lives on a
# COARSE grid. This node interpolates every non-coordinate attribute column up to the
# full voxel grid → one Voxel layer per column (`<source>_<column>`), so an image-space
# heatmap / composite is available. Scattered-data interpolation (scipy `griddata`) with
# a nearest-fill for out-of-hull voxels (matches the kernel's own inpaint philosophy).
# The 2D/3D lever picks per-plane (y,x) vs volumetric (z,y,x) interpolation — z_kind is
# not preserved through `with_structure`, so the lever (not the table) drives routing.


def _compute_rasterize_field(ctx: EvalContext) -> Dataset:
    """Interpolate a Point structure's attribute columns onto the full voxel grid → Voxel
    layers (one per column, named ``<source>_<column>``). 2D interpolates each ``(m,t,z,c)``
    plane from the points on that integer z-plane; 3D interpolates the ``(z,y,x)`` scatter
    onto the whole volume per ``(m,t,c)``. Out-of-hull voxels get the nearest point value
    (dense/finite output). ``method`` ∈ linear / nearest.

    Resolved spec (V2.06 + §7b): category transform; op ``transform.rasterize_field``; reads
    POINT + VOXEL (the image defines the target grid), adds VOXEL. **Dimensionality is
    INHERITED from the source Point layer's ``z_kind`` provenance** (``subpixel`` → 3D
    volumetric interpolation; ``plane_index`` → 2D per-plane) — NOT an independent 2D/3D
    lever, which could disagree with how the field was produced and silently misread a 2D
    per-plane field's plane-index ``z`` as a 3D coordinate (metadata-intelligence directive,
    `wire-node-v2` §7b). No calibration (pure grid geometry). Backend:
    :func:`scipy.interpolate.griddata` (lazily imported)."""
    from scipy.interpolate import griddata
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("rasterize field needs an image provider to define the voxel grid")
    ax = prov.axes
    # the one Point field on the wire, whatever it is called (`_resolve_layer`)
    source, _note = _resolve_layer(
        _point_layers(ds), ctx.layer("source"), node="rasterize field", socket="source",
        what="Point table", where="the `data` input",
        remedy="this interpolates a point field's attribute columns onto the voxel grid, so "
               "run a DVC / DIC / point-field node upstream", ctx=ctx)
    prefix = ctx.layer("prefix") or source
    method = ctx.params.get("__modes__", {}).get("method", "linear")
    # Inherit the field's dimensionality from its stamped z_kind (§7b); fall back to the
    # image's own volume-ness only for a hand-built field with no structure provenance.
    zk = ds.structure_zkind(Domain.POINT, source)
    is_3d = (zk == "subpixel") if zk is not None else (ax.z > 1)
    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:                                      # pragma: no cover - _resolve_layer
        raise ValueError(f"rasterize field needs a Point layer {source!r} "
                         "(run a DVC / point-field node first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    coord = {"id", "m", "t", "c", "z", "y", "x"}
    attr_names = [n for n in col if n not in coord]
    if not attr_names:
        raise ValueError(f"Point layer {source!r} has no attribute columns to rasterize")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all, y_all, x_all = col["z"], col["y"], col["x"]
    out = {n: np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float) for n in attr_names}

    def _interp(points: np.ndarray, values: np.ndarray, grid_pts: np.ndarray):
        if len(points) == 0:
            return None
        try:
            if method == "nearest" or len(points) < points.shape[1] + 1:
                return griddata(points, values, grid_pts, method="nearest")
            g = griddata(points, values, grid_pts, method=method)
            nan = np.isnan(g)
            if nan.any():                                   # out-of-hull → nearest fill
                g[nan] = griddata(points, values, grid_pts[nan], method="nearest")
            return g
        except Exception:                                   # degenerate hull → nearest
            return griddata(points, values, grid_pts, method="nearest")

    if is_3d:
        gz, gy, gx = np.mgrid[0:ax.z, 0:ax.y, 0:ax.x]
        grid_pts = np.column_stack([gz.ravel(), gy.ravel(), gx.ravel()]).astype(float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    sel = (m_all == m) & (t_all == t) & (c_all == c)
                    if not sel.any():
                        continue
                    P = np.column_stack([z_all[sel], y_all[sel], x_all[sel]])
                    for n in attr_names:
                        g = _interp(P, col[n][sel], grid_pts)
                        if g is not None:
                            out[n][m, t, :, c] = g.reshape(ax.z, ax.y, ax.x)
    else:
        gy, gx = np.mgrid[0:ax.y, 0:ax.x]
        grid_pts = np.column_stack([gy.ravel(), gx.ravel()]).astype(float)
        zi_all = np.rint(z_all).astype(int)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    for z in range(ax.z):
                        sel = ((m_all == m) & (t_all == t) & (c_all == c) & (zi_all == z))
                        if not sel.any():
                            continue
                        P = np.column_stack([y_all[sel], x_all[sel]])
                        for n in attr_names:
                            g = _interp(P, col[n][sel], grid_pts)
                            if g is not None:
                                out[n][m, t, z, c] = g.reshape(ax.y, ax.x)
    res = ds
    for n in attr_names:
        res = res.with_layer(Domain.VOXEL, f"{prefix}_{n}", out[n])
    return res
register_node(
    _compute_rasterize_field, op_key="transform.rasterize_field", label="Rasterize Field",
    category="transform",
    reads_domains=frozenset({Domain.POINT, Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InString("source", "Point layer", field=False, default="dvc",
                     layer_in=Domain.POINT,
                     description=
                     "Which Point field to interpolate onto the voxel grid — a DVC or DIC "
                     "output, whose points are the correlation subset centres. Rasterizing "
                     "turns that sparse field into dense per-voxel layers you can display, "
                     "threshold, or measure per region. Its dimensionality is inherited from the "
                     "source field, so there is no 2D/3D control here."),
            # empty => the layer prefix follows `source` (it names a FAMILY of output
            # layers, one per field component, not a single layer)
            InString("prefix", "Output prefix", field=False, default="",
                     description=
                     "Prefix for the FAMILY of Voxel layers this node writes — one per field "
                     "component, so a displacement field yields a set like `<prefix>_u`, `_v`, "
                     "`_w` rather than a single layer. EMPTY (the default) follows the source "
                     "layer's name, which keeps the two visibly paired; set it when two "
                     "rasterized fields would otherwise collide.")],
    outputs=[OutDataset()],
    # No DimMode lever — dimensionality is INHERITED from the source field's z_kind (§7b),
    # so a fixed volumetric footprint (a superset of the per-plane case) is the honest
    # declaration; the compute loops (m,t,z,c) internally and writes the whole raster.
    modes=[Mode("method", ["linear", "nearest"], default="linear", label="Interpolation",
                description=
                "How the sparse measurement points are interpolated onto every voxel. The "
                "points are scattered (a DVC/DIC grid, a detection cloud), so this is what "
                "fills the space BETWEEN them — the bulk of the output. Voxels outside the "
                "points' convex hull are always nearest-filled, whichever option is chosen, "
                "so the result is finite everywhere; it is also extrapolation, and should not "
                "be read as measured.",
                choice_docs={
                    "linear":
                        "Barycentric interpolation over a triangulation of the points: values "
                        "vary smoothly and continuously between neighbours, which is what a "
                        "displacement or strain field physically does. The right default for "
                        "any continuous field, and it needs enough points to triangulate — "
                        "with too few it falls back to nearest.",
                    "nearest":
                        "Each voxel takes its closest point's value, producing visible cells "
                        "of constant value. Honest about the sampling — it never invents an "
                        "intermediate value — so it is the choice for a CATEGORICAL column "
                        "(an id, a cluster index) and for inspecting where the measurements "
                        "actually are.",
                })],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Interpolate a Point vector/scalar field (e.g. DVC) up to full-resolution "
                "Voxel layers (one per attribute column); 2D-per-plane vs volumetric is "
                "inherited from the field's z_kind (§7b); nearest-filled out-of-hull.")
