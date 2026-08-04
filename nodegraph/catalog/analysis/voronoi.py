"""Voronoi Cells (``analysis.voronoi``) — Voronoi tessellation LINKING dots to areas: one cell per seed Point, nearest-seed in µm space, clipped to a Label region (per-region arenas) or a mask → a territory Label raster + table…"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    DimMode,
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
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.labels import _label_centroids, _label_raster

# ── Voronoi cells: dots + areas → territory raster (+ a MESH in 3D) ────────────
#
# One cell per DOT, clipped to the AREAS — the two inputs "linked" by a nearest-seed
# partition of the voxel grid in µm space. This is the standard territory analysis (nuclei
# → cell/tissue territories, bead → local-density cells) and it is a different operation
# from `analysis.tessellate`'s `voronoi` boundary, which groups a cloud by a cluster column
# and emits ONE surface per cluster with no notion of a bounding area.
#
# Raster-first rather than `scipy.spatial.Voronoi` polytopes on purpose: an exact polytope
# can only be clipped to a convex boundary, while a real area — a tissue mask, a cell body,
# anything with a concavity or a hole — is the whole reason the second input exists. The
# nearest-seed partition is voxel-accurate, honours arbitrary area shape in both 2D and 3D,
# and comes out already registered to the image grid every downstream node reads.


#: Arena voxels per KD-tree query. Bounds the peak coordinate array (this many rows × ndim
#: float64) independently of arena size, so a full-frame 3-D Voronoi cannot try to
#: materialize a (26M, 3) point array.
_VORONOI_BLOCK = 1 << 18
def _vox_lookup(arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``arr`` sampled at the nearest voxel of each row of ``pts`` (clipped to bounds)."""
    idx = tuple(np.clip(np.rint(pts[:, i]).astype(np.int64), 0, arr.shape[i] - 1)
                for i in range(arr.ndim))
    return arr[idx]
def _voronoi_assign(seeds_vox: np.ndarray, arena: np.ndarray, seed_arena: np.ndarray,
                    scale: Sequence[float], upper: float) -> np.ndarray:
    """Nearest-seed (Voronoi) partition of ``arena``, measured in **µm**.

    ``arena`` is an integer array whose non-zero values are *arenas*: a voxel competes only
    among the seeds whose ``seed_arena`` matches its own value, which is what stops a cell
    leaking across an area boundary. A single arena (a mask, or the whole frame) is just the
    degenerate case with one value.

    Returns an array shaped like ``arena`` holding the winning seed's **row index + 1** —
    not its Point id, because id 0 is a legitimate Point id while 0 here has to mean
    *unclaimed*: an arena with no seed of its own, or a voxel farther than ``upper`` µm from
    every seed it may compete for.

    Coordinates are scaled to µm before the query rather than converted to pixels with
    :func:`to_pixels_v2`: the comparison is between distances along different axes, so the
    honest space to do it in is the isotropic physical one — otherwise a coarse z step would
    make each plane count as one unit of distance and stretch every cell along z."""
    from scipy.spatial import cKDTree
    out = np.zeros(arena.size, dtype=np.int64)
    flat = np.asarray(arena).reshape(-1)
    nz = np.flatnonzero(flat)
    if nz.size == 0 or len(seed_arena) == 0:
        return out.reshape(arena.shape)
    keys = flat[nz]
    order = np.argsort(keys, kind="stable")
    nz, keys = nz[order], keys[order]
    sc = np.asarray(scale, dtype=float)
    rows = np.arange(1, len(seed_arena) + 1, dtype=np.int64)
    for a in np.unique(keys):
        pick = seed_arena == a
        if not pick.any():
            continue                                  # arena with no seed → unclaimed
        lo, hi = (int(v) for v in np.searchsorted(keys, [a, a + 1]))
        tree = cKDTree(seeds_vox[pick] * sc)
        mine = rows[pick]
        for s in range(lo, hi, _VORONOI_BLOCK):
            # `min(..., hi)` is load-bearing: `nz` is the CONCATENATION of every arena's
            # voxels, so a block that overruns this arena's run would hand the next arena's
            # voxels to this arena's seeds. Uncapped, an arena's spillover was only masked
            # by the next arena overwriting it — i.e. silently correct only where every
            # arena has a seed, and wrong exactly where `per_region` differs from `mask`.
            blk = nz[s:min(s + _VORONOI_BLOCK, hi)]
            crd = np.unravel_index(blk, arena.shape)
            pts = np.column_stack([c.astype(float) * sc[i] for i, c in enumerate(crd)])
            dist, j = tree.query(pts, k=1, distance_upper_bound=upper)
            ok = np.isfinite(dist)                    # miss ⇒ inf distance, j == n
            if ok.any():
                out[blk[ok]] = mine[j[ok]]
    return out.reshape(arena.shape)
def _voronoi_mesh_elements(assign: np.ndarray, present: np.ndarray, gids: np.ndarray,
                           cen: np.ndarray, *, m: int, t: int, c: int,
                           vox: Tuple[float, float, float], step: int) -> list:
    """One **closed** marching-cubes surface per Voronoi cell of a 3-D ``assign`` volume.

    Meshed inside each cell's own bounding box (one ``find_objects`` pass, not one full-volume
    comparison per cell) and padded by a background voxel on every side. The padding is
    load-bearing, not tidiness: a region touching its box edge marches to an OPEN surface,
    and an open surface makes ``enclosed_volume_um3`` meaningless *and* drops
    ``transform.rasterize_mesh`` onto its convex interior test — which for a cell clipped to
    a concave area would fill straight back through the boundary the clip established.

    ``density`` is the Voronoi local number density, ``1 / cell volume`` in seeds/µm³ — the
    classical estimator this tessellation exists to produce, and what the rasterizer uses to
    break ties where two voxelized surfaces overlap."""
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes
    from nodegraph.mesh import MeshElement, enclosed_volume_um3, surface_area_um2
    boxes = ndi.find_objects(assign)
    out: list = []
    for k, lab in enumerate(present.tolist()):
        sl = boxes[lab - 1] if 0 <= lab - 1 < len(boxes) else None
        if sl is None:
            continue
        sub = np.zeros(tuple(s.stop - s.start + 2 for s in sl), dtype=float)
        sub[1:-1, 1:-1, 1:-1] = (assign[sl] == lab)
        if sub.sum() < 8:                       # thinner than any closed surface can be
            continue
        try:
            verts, faces, _n, _v = marching_cubes(sub, level=0.5, step_size=step)
        except (RuntimeError, ValueError):
            continue                            # degenerate cell — skip, never fail a pull
        verts = verts + np.array([s.start - 1 for s in sl], dtype=float)
        vol3 = enclosed_volume_um3(verts, faces, vox)
        out.append(MeshElement(
            m=m, t=t, c=c, src_label=int(gids[k]), verts_zyx=verts, faces=faces,
            centroid_zyx=(float(cen[k, 0]), float(cen[k, 1]), float(cen[k, 2])),
            volume_um3=vol3, surface_area_um2=surface_area_um2(verts, faces, vox),
            density=(1.0 / vol3 if vol3 > 0 else 0.0), n_points=1))
    return out
def _compute_voronoi(ctx: EvalContext) -> Dataset:
    """**Voronoi cells from dots, clipped to areas** — one territory per Point, bounded by a
    Voxel area layer, as a Label raster + table (+ a MESH in 3D).

    Resolved spec: category analysis; op ``analysis.voronoi``; reads POINT (and VOXEL, in the
    two bounded modes), adds VOXEL + LABEL + MESH. 2D/3D lever: ``{"2D": WHOLE_PLANE,
    "3D": WHOLE_VOLUME}`` — 2D partitions each plane among the dots on that plane, 3D
    partitions the volume among all of its dots.

    The ``bound`` Mode is what links the two inputs:

    * ``per_region`` — the area layer is a **Label** raster and each region is its own arena:
      a cell may only claim voxels of the region its seed sits in, so territories never cross
      a compartment boundary. Seeds outside every region are dropped; regions with no seed
      stay background.
    * ``mask`` — every non-zero voxel of the area layer is one arena and all dots compete for
      it. A seed just outside the mask still owns the mask voxels nearest it, which is the
      right behaviour when the mask is a field of view rather than an object.
    * ``frame`` — no area input; the whole plane/volume is tessellated.

    Ids are painted **globally unique** across every (m,t,c) — the ``analysis.label``
    convention, so a downstream measurement can join on id without collisions — and the Label
    table records which dot and which area each cell came from (``point_id`` / ``region``)
    plus ``area`` (voxels) and ``density`` (1 / area in µm², or µm³ in 3D: the Voronoi local
    number density). The MESH carries the same id in ``src_label``.

    The 3D-only ``mesh`` Mode chooses whether the cell surfaces are built at all — meshing
    every cell is the expensive half of this node and the raster plus its table are complete
    without it. It is a Mode rather than an emptied ``mesh_name``, because
    :func:`nodegraph.registry.layer_value` treats an empty layer name as *unset* and resolves
    it back to the socket default, so "clear it to skip" is not expressible; and being a Mode
    lets ``mesh_name``/``decimate`` gate on it (`wire-node-v2` §5b/§5c) instead of sitting
    live-looking and inert.

    Distances are compared in µm (``ctx.calib`` pixel/z size), so an anisotropic z step
    cannot stretch cells along z; ``max_distance_um`` caps a cell's reach so an isolated dot
    cannot claim an unbounded amount of empty area."""
    from nodegraph.mesh import build_mesh_tables, with_mesh

    ds = ctx.inputs[0]
    ax = ds.axes
    volumetric = ctx.is_volume
    modes = ctx.params.get("__modes__", {})
    bound = modes.get("bound", "per_region")
    pts_layer = ctx.layer("points")
    name = ctx.layer("name")
    # 2D cells are flat polygons with no enclosing surface, so the Mesh half is 3D-only
    # regardless of the (3D-gated) `mesh` Mode's resolved value.
    mesh_name = (ctx.layer("mesh_name")
                 if volumetric and modes.get("mesh", "build") == "build" else "")
    step = max(1, int(ctx.params.get("decimate", 1)))
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)
    scale = vox if volumetric else (px, px)
    max_um = max(0.0, float(ctx.params.get("max_distance_um", 0.0)))
    upper = max_um if max_um > 0.0 else np.inf

    seeds = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
             if a.layer == pts_layer}
    if not seeds:
        have = sorted({k[1] for k in ds.attributes if k[0] is Domain.POINT and k[1]})
        raise ValueError(
            f"voronoi: no Point layer {pts_layer!r} on the input Dataset"
            + (f" (it carries {have})" if have else " (it carries no Point layers)")
            + " — these are the seed dots, so wire a detection (detect.spots / "
              "detect.particles) or transform.label_to_points upstream.")
    for req in ("id", "m", "t", "c", "z", "y", "x"):
        if req not in seeds:
            raise ValueError(f"voronoi: Point layer {pts_layer!r} is missing the "
                             f"coordinate column {req!r}")

    region6 = None
    if bound == "per_region":
        region6, _zk = _label_raster(ds, ctx.layer("region"), node="voronoi (per_region)")
    elif bound == "mask":
        attr = ds.get(Domain.VOXEL, ctx.layer("region"))
        if attr is None:
            raise ValueError(
                f"voronoi (mask): no Voxel layer {ctx.layer('region')!r} to bound the cells "
                "— wire a threshold/ROI mask, or set Bound to `frame` to tessellate the "
                "whole image.")
        region6 = np.asarray(attr.values)

    s_m = seeds["m"].astype(np.int64)
    s_t = seeds["t"].astype(np.int64)
    s_c = seeds["c"].astype(np.int64)
    s_z = seeds["z"].astype(float)
    s_id = seeds["id"].astype(np.int64)
    shape = (ax.z, ax.y, ax.x) if volumetric else (ax.y, ax.x)
    raster = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    elements: list = []
    offset = 0
    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if volumetric else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    # voxel volume/area, for the local-density column (µm³ in 3D, µm² in 2D)
    unit_um = float(np.prod(np.asarray(scale, dtype=float)))
    ctx.progress(0, len(units), "tessellating", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        sel = (s_m == m) & (s_t == t) & (s_c == c)
        if not volumetric:
            sel = sel & (np.rint(s_z).astype(np.int64) == z)
        if sel.any():
            if region6 is None:
                arena = np.ones(shape, dtype=np.int64)
            else:
                sub = region6[m, t, :, c] if volumetric else region6[m, t, z, c]
                arena = np.asarray(sub, dtype=np.int64)
                if bound == "mask":
                    arena = (arena != 0).astype(np.int64)
            pos = (np.column_stack([s_z[sel], seeds["y"][sel].astype(float),
                                    seeds["x"][sel].astype(float)]) if volumetric else
                   np.column_stack([seeds["y"][sel].astype(float),
                                    seeds["x"][sel].astype(float)]))
            seed_arena = (_vox_lookup(arena, pos) if bound == "per_region"
                          else np.ones(len(pos), dtype=np.int64))
            assign = _voronoi_assign(pos, arena, seed_arena, scale, upper)
            present = np.unique(assign)
            present = present[present > 0]
            if present.size:
                cen, counts = _label_centroids(assign, present)
                gids = offset + 1 + np.arange(present.size, dtype=np.int64)
                offset += present.size
                lut = np.zeros(int(present.max()) + 1, dtype=np.int64)
                lut[present] = gids
                painted = lut[assign]
                if volumetric:
                    raster[m, t, :, c] = painted
                else:
                    raster[m, t, z, c] = painted
                mine = s_id[sel]
                if mesh_name:
                    elements.extend(_voronoi_mesh_elements(
                        assign, present, gids, cen, m=m, t=t, c=c, vox=vox, step=step))
                # `cen` is (z,y,x) on a volume and (y,x) on a plane — on a plane the row's
                # z is the plane index, matching the z_kind="plane_index" it is stamped with
                cz = cen[:, 0] if volumetric else np.full(present.size, float(z))
                cy = cen[:, 1] if volumetric else cen[:, 0]
                cx = cen[:, 2] if volumetric else cen[:, 1]
                for k, srow in enumerate((present - 1).tolist()):
                    area = int(counts[k])
                    rows.append({
                        "id": int(gids[k]), "m": m, "t": t, "c": c, "area": area,
                        "z": float(cz[k]), "y": float(cy[k]), "x": float(cx[k]),
                        "point_id": int(mine[srow]),
                        "region": int(seed_arena[srow]),
                        "density": (1.0 / (area * unit_um)) if area and unit_um else 0.0,
                    })
        ctx.progress(i + 1, len(units), "tessellating", frames=ax.t)

    out = ds.with_layer(Domain.VOXEL, name, raster)
    if rows:
        merged = {k: np.array([r[k] for r in rows],
                              dtype=(np.int64 if k in ("id", "m", "t", "c", "area",
                                                       "point_id", "region") else float))
                  for k in rows[0]}
        out = out.with_structure(StructureTable(
            Domain.LABEL, merged, layer=name,
            z_kind=("subpixel" if volumetric else "plane_index")))
    if mesh_name:
        out = with_mesh(out, build_mesh_tables(elements, layer=mesh_name,
                                              z_kind="subpixel"),
                        provenance={"boundary": "voronoi_cells", "source": "points",
                                    "src_layer": str(pts_layer), "bound": str(bound),
                                    "voxel_size_um": [float(v) for v in vox]})
    return out
register_node(
    _compute_voronoi, op_key="analysis.voronoi", label="Voronoi Cells",
    category="analysis",
    # POINT is required in every mode; the VOXEL area layer is mode-gated, which is what
    # exempts it from the reads_domains consistency check (`frame` reads no Voxel layer, and
    # reads_domains has no per-mode form — the tessellate / track.link precedent).
    reads_domains=frozenset({Domain.POINT}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL, Domain.MESH}),
    inputs=[InDataset(),
            InString("points", "Seed points", field=False, default="particles",
                     layer_in=Domain.POINT,
                     description=
                     "The DOTS — one Voronoi cell per point of this table. A detection output, "
                     "or Label → Points if the seeds are segmented nuclei. Points are compared "
                     "in MICRONS, so an anisotropic z step cannot stretch the cells along z. "
                     "In 2D each dot only seeds the plane its z rounds to; in 3D every dot in "
                     "the volume competes."),
            InString("region", "Area layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     available_in={"bound": frozenset({"per_region", "mask"})},
                     description=
                     "The AREAS the cells are clipped to. Under `per_region` this must be a "
                     "LABEL raster and each region becomes its own arena — a cell can only "
                     "claim voxels of the region its own dot sits in, so territories never "
                     "cross a compartment boundary and a region with no dot stays empty. Under "
                     "`mask` any non-zero voxel layer works and every dot competes for all of "
                     "it. Hidden under `frame`, which tessellates the whole image."),
            InFloat("max_distance_um", "Max reach", unit="um", field=True, default=0.0,
                    pick_kind="distance",
                    description=
                    "How far a cell may extend from its own dot, in microns. 0 means "
                    "unlimited, so every voxel of the area is claimed by someone. A positive "
                    "value leaves voxels farther than this from every eligible dot as "
                    "background, which is how you stop one isolated dot from claiming a large "
                    "empty area and reporting an enormous territory. It therefore caps the "
                    "reported `area` and raises `density` for sparse cells; set it near the "
                    "biological cell radius, not the image size."),
            InString("name", "Output layer", field=False, default="voronoi",
                     layer_out=(Domain.VOXEL, Domain.LABEL),
                     description=
                     "Name of the territory output — one name into two domains: a Voxel label "
                     "raster with one id per cell, plus a Label table carrying `area` (voxels), "
                     "`density` (1 / area in µm² or µm³ — the Voronoi local number density), "
                     "and `point_id` / `region` recording which dot and which area each cell "
                     "came from. Ids are globally unique across every (m,t,c), so Measure can "
                     "join on id."),
            InString("mesh_name", "Output mesh", field=False, default="voronoi_mesh",
                     layer_out=(Domain.MESH,),
                     available_in={"dim": frozenset({"3D"}),
                                   "mesh": frozenset({"build"})},
                     description=
                     "Name of the Mesh of cell surfaces (one closed marching-cubes surface per "
                     "cell) that Rasterize Mesh and the 3D viewer read. Each surface is padded "
                     "before meshing so it comes out watertight, which is what makes its "
                     "enclosed volume meaningful and keeps the rasterizer on its concavity-"
                     "preserving interior test. 3D only, and only when Surfaces is `build`."),
            InInt("decimate", "Vertex step", unit="", field=False, default=1,
                  available_in={"dim": frozenset({"3D"}),
                                "mesh": frozenset({"build"})},
                  description=
                  "Keep only every Nth vertex of each cell surface. 1 keeps the full mesh. A "
                  "Voronoi of many cells produces a lot of triangles, so raising this shrinks "
                  "the mesh sharply (roughly with the square) — but decimation cuts corners, "
                  "so it systematically REDUCES the enclosed volume the mesh reports. Raise it "
                  "for display; leave it at 1 when the mesh is being measured. The raster and "
                  "its `area`/`density` columns are unaffected either way.")],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("bound", ["per_region", "mask", "frame"], default="per_region",
                label="Bound",
                description=
                "What the territories are clipped to — i.e. what \"available space\" means. It "
                "decides whether a cell can leak past an object boundary, and whether the areas "
                "reported are areas of real space or of the whole field. It also selects "
                "whether the area-layer socket is read at all.",
                choice_docs={
                    "per_region":
                        "Each region of a Label raster is its own separate arena: dots inside a "
                        "region compete only with each other, and no territory can cross into a "
                        "neighbouring region. The right choice when the areas are real objects "
                        "— nuclei inside cells, granules inside a cell body — and the default.",
                    "mask":
                        "Every non-zero voxel of the layer is ONE shared arena and all dots "
                        "compete for it. A seed just outside the mask still owns the mask voxels "
                        "nearest it, which is correct when the mask is a field of view or a "
                        "tissue extent rather than an object — and wrong when it is a set of "
                        "objects that should not share.",
                    "frame":
                        "No area layer at all: the whole plane (or volume) is tessellated, so "
                        "every voxel belongs to some dot. Gives the classic Voronoi diagram and "
                        "the largest areas, since territories run to the image edge — which "
                        "makes the border cells' areas an artefact of the field of view, not a "
                        "measurement.",
                }),
           # 3D-only, and it in turn gates `mesh_name`/`decimate` — meshing every cell is
           # the expensive half of this node, and skipping it must not leave two live-looking
           # controls behind (`wire-node-v2` §5c)
           Mode("mesh", ["build", "skip"], default="build", label="Surfaces",
                available_in={"dim": frozenset({"3D"})},
                description=
                "Whether to also build a closed marching-cubes surface per 3-D cell. Meshing is "
                "the expensive half of this node, and it is what the 3-D viewer and Rasterize "
                "Mesh consume; the territory raster and its area/density table are produced "
                "either way. 3D only.",
                choice_docs={
                    "build":
                        "Build one watertight surface per cell, so the cells can be drawn in 3D, "
                        "rasterized back, or measured as enclosed volumes. Each surface is "
                        "padded before meshing to keep it closed. Costs the bulk of the node's "
                        "runtime and memory on a crowded volume.",
                    "skip":
                        "Raster and table only — no Mesh output, and the mesh-related controls "
                        "disappear with it. Much faster and much lighter, and all you need when "
                        "the question is about areas, densities and which dot owns which voxel "
                        "rather than about surfaces.",
                })],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Voronoi tessellation LINKING dots to areas: one cell per seed Point, "
                "nearest-seed in µm space, clipped to a Label region (per-region arenas) or "
                "a mask → a territory Label raster + table (area, local density, source dot "
                "and area) and, in 3D, a MESH of the cell surfaces.")
