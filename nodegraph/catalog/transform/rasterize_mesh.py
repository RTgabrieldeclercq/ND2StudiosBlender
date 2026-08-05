"""Rasterize Mesh (``transform.rasterize_mesh``) — Rasterize a MESH → one filled Voxel Label region per element (interior test inherited from the mesh provenance: convex hull, or faces-based parity so concavity survives) + a per-element Label…"""

from __future__ import annotations

import numpy as np

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
    OutDataset,
)
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers

# ── Rasterize a MESH → a Voxel Label volume + geometry table (3D) ──────────────
#
# The downstream half of the v1 split (ports granule_volume_mask). Each mesh element is
# voxelized and the regions are painted into one combined Voxel **Label raster** (higher
# density wins on overlap), with a per-element **Label table** carrying the mesh's ANALYTIC
# geometry alongside the realized voxel_count.


def _compute_rasterize_mesh(ctx: EvalContext) -> Dataset:
    """Rasterize a **MESH** into a Voxel **Label** volume + a per-element geometry table
    (ports v1 ``granule_volume_mask``).

    Resolved spec (V2.08): category transform (beside ``transform.rasterize_field``); op
    ``transform.rasterize_mesh``; reads MESH + VOXEL (the image defines the target grid),
    adds VOXEL + LABEL — the same output contract the fused node had, so ``boundary_band``
    / ``measure`` chains are unchanged. ``WHOLE_VOLUME``, ``kernel_axes={z,y,x}``.

    **The interior test is INHERITED, not a lever** (`wire-node-v2` §7b, as
    ``rasterize_field`` inherits its dimensionality): ``convex_hull`` provenance keeps the
    cheap ``Delaunay.find_simplex`` path, while a possibly-concave mesh (``alpha_shape``
    with a finite alpha, ``label_surface``) uses the faces-based even-odd parity test — so
    the voxel mask finally agrees with the ``volume_um3`` / ``surface_area_um2`` the
    tessellation reports. A mesh with no provenance falls back to its own derived
    ``closed`` topology, so a hand-built mesh still resolves. This fixes a real latent
    bug: the alpha-shape producer stores the *unfiltered* Delaunay, whose simplices union
    to the convex hull, so every mode used to rasterize convex.

    ids are painted **verbatim** from the mesh element's ``id`` — no renumbering — so an
    element that voxelizes empty leaves a gap rather than shifting every id after it."""
    from nodegraph.kernels.mesh_raster import interior_test_for, voxelize_mesh
    from nodegraph.mesh import mesh_element, mesh_provenance, read_mesh

    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("rasterize mesh needs an image provider (defines the voxel grid)")
    ax = prov.axes
    if ax.z < 2:
        raise ValueError("rasterize mesh needs a 3D volume (z>1)")
    # The one mesh on the wire, whatever it is called (`_resolve_layer`). A mesh occupies
    # THREE structure buckets — `<name>`, `<name>/vert`, `<name>/face` — so the candidates
    # are the element tables only; the strata are not separately selectable meshes.
    from nodegraph.mesh import MESH_SEP
    layer, _note = _resolve_layer(
        [n for n in _structure_layers(ds, Domain.MESH) if MESH_SEP not in n],
        ctx.layer("mesh"), node="rasterize mesh", socket="mesh", what="mesh",
        where="the `data` input",
        remedy="run analysis.tessellate (which emits `mesh`) or analysis.voronoi with its "
               "mesh output on (`voronoi_mesh`) upstream", ctx=ctx)
    name = ctx.layer("name")
    smooth = max(0.0, float(ctx.params.get("smooth_um", 0.0)))
    fill = bool(ctx.params.get("fill_holes", False))
    min_vox = max(1, int(ctx.params.get("min_voxels", 1)))
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)

    tables = read_mesh(ds, layer)
    prv = mesh_provenance(ds, layer)
    el = tables.element.columns
    e_id = np.asarray(el["id"], dtype=np.int64)
    e_m = np.asarray(el["m"], dtype=np.int64)
    e_t = np.asarray(el["t"], dtype=np.int64)
    e_c = np.asarray(el["c"], dtype=np.int64)
    e_dens = np.asarray(el["density"], dtype=float)
    e_closed = np.asarray(el["closed"], dtype=np.int64)

    raster = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    rows: list = []
    for key in sorted({(int(a), int(b), int(d)) for a, b, d in zip(e_m, e_t, e_c)}):
        m, t, c = key
        sel = np.flatnonzero((e_m == m) & (e_t == t) & (e_c == c))
        painted: list = []                             # (density, -id, id, mask)
        for row in sel.tolist():
            verts, faces = mesh_element(tables, row)
            test = interior_test_for(prv, bool(e_closed[row]))
            mask = voxelize_mesh(verts, faces, (ax.z, ax.y, ax.x), vox, mode=test,
                                 density=float(e_dens[row]), smooth_um=smooth,
                                 fill_holes=fill, min_voxels=min_vox)
            n_vox = int(mask.sum())
            if n_vox == 0:
                continue                               # did not rasterize — id stays absent
            painted.append((float(e_dens[row]), -int(e_id[row]), int(e_id[row]), mask))
            rows.append({
                "id": int(e_id[row]), "m": m, "t": t, "c": c,
                "z": float(np.asarray(el["z"])[row]),
                "y": float(np.asarray(el["y"])[row]),
                "x": float(np.asarray(el["x"])[row]),
                "volume_um3": float(np.asarray(el["volume_um3"])[row]),
                "density": float(e_dens[row]),
                "n_points": int(np.asarray(el["n_points"])[row]),
                "surface_area_um2": float(np.asarray(el["surface_area_um2"])[row]),
                "voxel_count": n_vox,
            })
        combined = np.zeros((ax.z, ax.y, ax.x), dtype=np.int64)
        for _d, _negid, lid, mask in sorted(painted, key=lambda p: (p[0], p[1])):
            combined[mask] = lid                       # higher density wins (last write)
        raster[m, t, :, c] = combined

    res = ds.with_layer(Domain.VOXEL, name, raster)
    if not rows:
        return res
    merged = {k: np.array([r[k] for r in rows],
                          dtype=(np.int64 if k in ("id", "m", "t", "c", "n_points",
                                                   "voxel_count") else float))
              for k in rows[0]}
    return res.with_structure(StructureTable(Domain.LABEL, merged, layer=name,
                                             z_kind="subpixel"))
register_node(
    _compute_rasterize_mesh, op_key="transform.rasterize_mesh", label="Rasterize Mesh",
    category="transform",
    reads_domains=frozenset({Domain.MESH, Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[InDataset(),
            InString("mesh", "Mesh layer", field=False, default="mesh",
                     layer_in=Domain.MESH,
                     description=
                     "Which Mesh to convert back into voxels — a Tessellate output. Each mesh "
                     "element becomes one filled label region, so this is how a geometric "
                     "surface re-enters the voxel world to be measured or masked. Whether a "
                     "voxel counts as inside is decided from the mesh's own stamped "
                     "provenance, not by a control here."),
            InString("name", "Output layer", field=False, default="labels",
                     layer_out=(Domain.VOXEL, Domain.LABEL),
                     description=
                     "Name of the rasterized output — one name into two domains: a Voxel label "
                     "raster with one id per mesh element, plus a Label table. The default "
                     "`labels` matches what Connected Components and Segmentation emit, so "
                     "downstream nodes need no rewiring; rename it if a real label raster is "
                     "already on the wire under that name."),
            InFloat("smooth_um", "Smoothing", unit="um", field=True, default=0.0,
                    pick_kind="radius",
                    description=
                    "Smooth the region boundaries by this much after filling, in microns — it "
                    "removes the stair-stepping that voxelising a triangle mesh leaves behind. "
                    "0 keeps the exact rasterization. LARGER rounds off genuine surface detail "
                    "and shifts the boundary, so it CHANGES the volume it reports; keep it at 0 "
                    "when the volume is the measurement."),
            InInt("min_voxels", "Min voxels/region", unit="", field=False, default=1,
                  description=
                  "Discard rasterized regions smaller than this many voxels. 1 keeps "
                  "everything. Its use is dropping the slivers that appear when a mesh element "
                  "is thinner than a voxel — those have essentially no volume but still occupy "
                  "an id and a table row, which distorts per-object statistics and object "
                  "counts."),
            InBool("fill_holes", "Fill holes", field=False, default=False,
                   description=
                   "Fill enclosed interior voids after rasterizing. ON when a mesh is watertight "
                   "in principle but voxelising it left pinholes, which is common for thin or "
                   "highly-curved surfaces. It only ever INCREASES the reported volume — and it "
                   "will also fill a void that was genuinely part of the geometry, so leave it "
                   "off if the mesh legitimately encloses cavities.")],
    outputs=[OutDataset()],
    # No interior-test lever — it is derived from the mesh's stamped provenance (§7b).
    modes=[],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Rasterize a MESH → one filled Voxel Label region per element (interior "
                "test inherited from the mesh provenance: convex hull, or faces-based "
                "parity so concavity survives) + a per-element Label table (centroid, "
                "analytic volume_um3/density/n_points/surface_area, realized voxel_count); "
                "3D (ports v1 granule_volume_mask).")
