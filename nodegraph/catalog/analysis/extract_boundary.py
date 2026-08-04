"""Extract Boundary (``analysis.extract_boundary``) — A label region's outline → boundary Points (2D contours / 3D surface vertices); wraps boundary.extract_boundary."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, Granularity, InDataset, InString, OutDataset
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX

def _layers_extract_boundary(params, modes):
    """Output name is DERIVED: empty `name` means `f"{labels}_boundary"`."""
    src = params.get("labels") or "labels"
    return ((Domain.POINT, params.get("name") or "%s_boundary" % src),)
# ── Extract Boundary (Label raster → boundary Points) ───────────────────────────

def _compute_extract_boundary(ctx: EvalContext) -> Dataset:
    """Extract each Label region's outline as boundary Points (contours in 2D via
    ``find_contours``; surface vertices in 3D via ``marching_cubes``) — wraps
    :func:`nodegraph.boundary.extract_boundary` per (m,t,c[,z]) and threads the
    acquisition coordinates the compute-level helper leaves at 0. Ids are global-unique.

    Resolved spec (2026-07-28 socket/domain fix): the compute had always read a ``labels``
    (source Voxel layer) and an output-name param, but ``register_node`` declared **only**
    ``InDataset()`` — so neither had a socket and both were permanently pinned to their
    defaults: this node could not outline a Label raster that wasn't named ``labels``.
    Both are now real sockets. The output-name param is ``name`` (the catalog-wide
    convention — 16 other nodes) rather than the unreachable ``out`` it replaces, and
    follows the ``analysis.accumulate_field`` pattern where **empty means auto-derive**, so
    the default output layer stays ``f"{labels}_boundary"`` and existing graphs are
    unaffected. ``reads_domains``/``adds_domains`` were likewise empty despite the node
    hard-requiring a Voxel raster and adding POINT — so the GUI domain rail showed no chips
    and no red missing-domain validation here, unlike its ``detect.spots`` neighbour whose
    contract is identical.

    NOT changed: the 3D path's ``marching_cubes`` faces stay discarded. That is deliberate,
    not an oversight — this node's contract is a Point table, and it meshes the **union** of
    all non-zero labels as one surface (two touching labels give one shell), whereas
    ``analysis.tessellate``'s ``label_surface`` mode meshes **per region** and keeps the
    faces. label_surface is the MESH route; see :mod:`nodegraph.boundary`."""
    from dataclasses import replace as _dc_replace

    from nodegraph.boundary import extract_boundary
    ds = ctx.inputs[0]
    ax = ds.axes
    is_3d = ctx.is_volume
    layer = ctx.layer("labels")
    out_layer = ctx.layer("name") or f"{layer}_boundary"
    raster_attr = ds.get(Domain.VOXEL, layer)
    if raster_attr is None:
        raise ValueError(f"extract boundary needs a Label raster {layer!r}")
    raster6 = raster_attr.values

    def located(tbl, m, t, c, z=None):
        cols = dict(tbl.columns)
        cols["m"] = np.full(tbl.n, m, np.int64)
        cols["t"] = np.full(tbl.n, t, np.int64)
        cols["c"] = np.full(tbl.n, c, np.int64)
        if z is not None:
            cols["z"] = np.full(tbl.n, float(z))
        return _dc_replace(tbl, columns=cols)

    tables = []
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                if is_3d:
                    tbl = extract_boundary(raster6[m, t, :, c].astype(float),
                                           dim="3D", layer=out_layer)
                    if tbl.n:
                        tables.append(located(tbl, m, t, c))
                else:
                    for z in range(ax.z):
                        tbl = extract_boundary(raster6[m, t, z, c].astype(float),
                                               dim="2D", layer=out_layer)
                        if tbl.n:
                            tables.append(located(tbl, m, t, c, z=z))
    zk = "subpixel" if is_3d else "plane_index"
    if not tables:
        merged = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=out_layer)
    else:
        keys = list(tables[0].columns)
        cols = {k: np.concatenate([tb.columns[k] for tb in tables]) for k in keys}
        cols["id"] = np.arange(len(cols["id"]), dtype=np.int64)   # global-unique ids
        merged = StructureTable(Domain.POINT, cols, layer=out_layer, z_kind=zk)
    return ds.with_structure(merged)
register_node(
    _compute_extract_boundary, op_key="analysis.extract_boundary",
    label="Extract Boundary", category="analysis",
    extra_layers=_layers_extract_boundary,
    # identical contract to detect.spots / detect.particles: a Voxel raster in, Points out
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     description=
                     "Which label raster to outline — a Connected Components or Segmentation "
                     "output. Every region is traced separately, so the number of boundary "
                     "points scales with total object PERIMETER (2D) or SURFACE AREA (3D), "
                     "not with object count: a few hundred ragged cells can produce a very "
                     "large Point table."),
            # empty = auto-derive f"{labels}_boundary" (the accumulate_field convention),
            # so the default output layer name is unchanged
            InString("name", "Output layer", field=False, default="",
                     description=
                     "Name of the boundary Point table this node writes. EMPTY (the default) "
                     "auto-derives it as the label layer's name plus `_boundary`, so a graph "
                     "that renames its labels keeps a matching boundary name without a second "
                     "edit. Set it explicitly only when you need a specific name — the "
                     "auto-derived one is what the layer picker will offer downstream either "
                     "way.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="A label region's outline → boundary Points (2D contours / 3D surface "
                "vertices); wraps boundary.extract_boundary. For a MESH of a label raster "
                "use analysis.tessellate (label_surface), which meshes per region.")
