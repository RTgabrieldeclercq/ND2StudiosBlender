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
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import MESH_ELEMENT, on_layer
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.labels import (
    _label_centroids,
    _label_instances,
    _label_raster,
    _point_layers,
    _resolve_layer,
    _voxel_layers,
)
from nodegraph.catalog._shared.sampling import _require_same_grid

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


def _layers_voronoi(params, modes):
    """The AREAS raster is carried onto the output as ``f"{name}_areas"`` (2026-08-04).

    No socket can describe it — its name is derived from the output name, and it is a *copy*
    of a layer a READ socket named — so it needs declaring here or the layer catalog, the
    downstream pickers and the Viewer's overlay list would never know it exists.

    **Why copy it at all.** The output Dataset is built on the ``data`` input, so it inherits
    that branch's layers: with the seeds coming from one chain and the areas from another,
    viewing this node showed the SEEDS' labels and points and no trace of the areas the cells
    were actually clipped to (reported 2026-08-04) — and, worse, the inherited raster is
    usually *also* called ``labels``, so the overlay looked like it was showing the area layer
    while showing a different branch's. There was no way to see a territory against the region
    that bounded it, which is the one picture this node exists to produce.

    Must never raise (it runs on every keystroke), and it is skipped under ``frame``, which
    reads no area layer at all."""
    nm = (params or {}).get("name") or "voronoi"
    # `<name>_seeds` — the dots that actually WON a territory (2026-08-04). The Points overlay
    # draws every Point table on the payload, so the seed cloud it shows includes the ones this
    # node dropped; a caller who wants "only what participated" needs them as their own layer
    # to pick. Emitted in every bound, since a dot can be dropped by the reach cap under
    # `frame` too.
    out = ((Domain.POINT, "%s_seeds" % nm),)
    if (modes or {}).get("bound", "per_region") == "frame":
        return out
    return out + ((Domain.VOXEL, "%s_areas" % nm),)
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

    Resolved spec: category analysis; op ``analysis.voronoi``; reads POINT (and VOXEL +
    LABEL, in the two bounded modes), adds VOXEL + LABEL + MESH. 2D/3D lever:
    ``{"2D": WHOLE_PLANE, "3D": WHOLE_VOLUME}`` — 2D partitions each plane among the dots on
    that plane, 3D partitions the volume among all of its dots.

    **Two Dataset inputs.** The dots are a Point table and the areas are a raster, and they
    are routinely produced by different branches — nuclei detected on one channel, cell
    bodies segmented on another — which one wire cannot carry. So ``areas`` is an optional
    second input; the area layer is read from it when wired and from the main input when
    not, and the two must address the same voxels (``_require_same_grid``) because the
    partition is computed on one shared grid. ``data`` stays the primary: it is the
    calibration and domain source, and the seeds always come from it.

    Which is also why the area raster is COPIED onto the output as ``f"{name}_areas"``: the
    payload is built on ``data``, so the areas branch would otherwise leave no trace and the
    result could not be viewed against the regions that produced it (see
    :func:`_layers_voronoi`).

    The ``bound`` Mode is what links the two:

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

    # The seeds are whichever Point table is on `data`; the socket only has to speak up when
    # there is more than one. See `_resolve_layer` for why this is inferred rather than typed.
    notes: List[str] = []
    pts_layer, _note = _resolve_layer(
        _point_layers(ds), pts_layer, node="voronoi", socket="points",
        what="Point table", where="the `data` input",
        remedy="these are the seed dots, so wire a detection (detect.spots / "
               "detect.particles) or transform.label_to_points into `data`")
    if _note:
        notes.append("seeds: " + _note)

    seeds = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
             if a.layer == pts_layer}
    for req in ("id", "m", "t", "c", "z", "y", "x"):
        if req not in seeds:
            raise ValueError(f"voronoi: Point layer {pts_layer!r} is missing the "
                             f"coordinate column {req!r}")

    # ── where the AREAS come from: the optional second Dataset, else this one ──
    #
    # The dots and the areas are routinely produced by two DIFFERENT branches — nuclei
    # detected on one channel, cell bodies segmented on another — and a Point table and
    # someone else's Label raster cannot be put on one wire. Unwired, `areas` falls back to
    # the main input, which is exactly the single-wire behaviour every existing graph has.
    areas = ctx.input("areas")
    if areas is None:
        areas = ds
    elif bound == "frame":
        raise ValueError(
            "voronoi: an `areas` Dataset is wired but Bound is `frame`, which tessellates "
            "the whole image and reads no area layer at all — so those areas would be "
            "silently ignored. Set Bound to `per_region` (Label regions as separate arenas) "
            "or `mask` (one shared arena), or unwire `areas`.")
    else:
        # read voxel-for-voxel against this node's grid: a cropped/shifted areas branch
        # would clip every cell to the wrong part of the image
        _require_same_grid(ds, areas, socket="areas",
                           consequence="clip each cell to the wrong region")

    # ...and the areas are whichever candidate is on the AREAS wire. `per_region` needs a
    # whole Label instance (a raster plus the table that divides it into objects); `mask`
    # takes any non-zero raster, so there its candidate set is wider and more often ambiguous.
    region_layer = ctx.layer("region")
    wire = "the `areas` input" if areas is not ds else "the `data` input"
    region6 = None
    if bound == "per_region":
        region_layer, _note = _resolve_layer(
            _label_instances(areas), region_layer, node="voronoi (per_region)",
            socket="region", what="Label instance", where=wire,
            remedy="`per_region` clips each cell to the region its own dot sits in, so it "
                   "needs a label raster AND its table — run analysis.segment / "
                   "analysis.label, or set Bound to `mask` to use a plain mask instead")
        if _note:
            notes.append("areas: " + _note)
        region6, _zk = _label_raster(areas, region_layer, node="voronoi (per_region)")
    elif bound == "mask":
        region_layer, _note = _resolve_layer(
            _voxel_layers(areas), region_layer, node="voronoi (mask)",
            socket="region", what="Voxel layer", where=wire,
            remedy="wire a threshold/ROI mask, or set Bound to `frame` to tessellate the "
                   "whole image")
        if _note:
            notes.append("areas: " + _note)
        region6 = np.asarray(areas.get(Domain.VOXEL, region_layer).values)

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
    if notes:                       # say which layers were inferred, before the long loop
        ctx.progress(0, len(units), "using " + "; ".join(notes), frames=ax.t)
    # Seed accounting. A dot that gets no territory is DROPPED, and until 2026-08-04 it was
    # dropped in silence — which is how a real graph came to report 25 cells for 141 nuclei
    # and read as "the area layer isn't being used". Two independent causes, both counted so
    # the message can name the actionable one:
    #   * `outside` — under `per_region` the seed's own voxel is BACKGROUND, so it belongs to
    #     no arena and never competes. This is the one that bites when the two branches do
    #     not spatially agree (nuclei from a Z-PROJECTION against an area layer from a single
    #     Z-PLANE: the projected centroid sits wherever the nucleus is in any plane, while
    #     the areas only cover that one plane).
    #   * `n_sel` vs the table length — a seed whose (m,t,c[,z]) addresses no unit at all.
    n_seeds = int(len(s_id))
    n_sel = 0
    outside = 0
    ctx.progress(0, len(units), "tessellating", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        sel = (s_m == m) & (s_t == t) & (s_c == c)
        if not volumetric:
            sel = sel & (np.rint(s_z).astype(np.int64) == z)
        n_sel += int(sel.sum())
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
            outside += int((seed_arena == 0).sum())
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

    # ── account for every seed, out loud (2026-08-04) ──────────────────────────
    #
    # `len(rows)` is the number of dots that actually got a territory. Anything missing was
    # dropped, and a silently shorter table is the worst possible way to say so: the node
    # succeeds, the raster looks plausible, and the only symptom is a cell count nobody
    # cross-checks. Reported on the progress rail (the one user-visible channel a compute
    # has) and stamped on the output so it survives the run and a downstream node can read
    # it — the §7b namespaced non-calibration pattern.
    kept = len(rows)
    dropped = max(0, n_seeds - kept)
    if dropped:
        why = []
        if outside:
            why.append(f"{outside} sat on BACKGROUND of {region_layer!r} "
                       f"(no arena, so they never compete)")
        if n_sel < n_seeds:
            why.append(f"{n_seeds - n_sel} addressed no (m,t,c"
                       f"{'' if volumetric else ',z'}) unit of this image")
        residual = dropped - outside - (n_seeds - n_sel)
        if residual > 0:
            why.append(f"{residual} won no voxel at all (another dot was nearer to every "
                       f"voxel they could claim — coincident or near-coincident dots, or "
                       f"`Max reach` set below the spacing)")
        ctx.progress(len(units), len(units),
                     f"{kept}/{n_seeds} seeds got a cell — {dropped} dropped"
                     + (f" ({'; '.join(why)})" if why else ""), frames=ax.t)
    # Deliberately NOT a raise, even when EVERY seed is dropped. An empty result has to stay
    # a structurally valid one (`build_mesh_tables` with zero elements is a pinned contract),
    # and a refusal would in any case have been silent about the case that actually bit: a
    # PARTIAL loss, where the node succeeds and only the row count is short.
    out = ds.with_layer(Domain.VOXEL, name, raster).with_metadata(
        voronoi_seeds=n_seeds, voronoi_cells=kept, voronoi_seeds_outside=outside)

    # ── the seeds that WON a territory, as their own Point table ──────────────
    #
    # The Points overlay draws every Point table on the payload, so the dots on screen are the
    # input cloud — including the ones dropped here. Filtering has to be a LAYER the caller can
    # select, not a flag on the original table, because the original belongs to the upstream
    # node and this node has no business editing what it means. `cell_id` closes the loop back
    # to the Label row each surviving dot produced.
    _kept_ids = {int(r["point_id"]) for r in rows}
    _sel = np.isin(s_id, np.fromiter(_kept_ids, dtype=np.int64, count=len(_kept_ids))) \
        if _kept_ids else np.zeros(len(s_id), dtype=bool)
    _cell_of = {int(r["point_id"]): int(r["id"]) for r in rows}
    seed_cols = {k: np.asarray(seeds[k])[_sel] for k in ("id", "m", "t", "c", "z", "y", "x")}
    seed_cols["cell_id"] = np.array([_cell_of[int(i)] for i in seed_cols["id"]],
                                    dtype=np.int64)
    out = out.with_structure(StructureTable(
        Domain.POINT, seed_cols, layer="%s_seeds" % name,
        z_kind=(ds.structure_zkind(Domain.POINT, pts_layer)
                or ("subpixel" if volumetric else "plane_index"))))
    if region6 is not None:
        # Carry the arenas through under a name of our own (`_layers_voronoi`). The payload is
        # built on `data`, so without this the areas branch leaves NO trace on the output and
        # a territory cannot be viewed against the region that bounded it. Deliberately not
        # under its original name: that is usually `labels`, which the seeds' branch has
        # already put on this Dataset meaning something else entirely.
        #
        # Only the arenas that PRODUCED a cell survive (2026-08-04). A region no dot landed in
        # contributed nothing to the result, so drawing it alongside the territories invites
        # exactly the misreading this layer exists to prevent — that the empty region is a
        # territory, or that a territory is missing from it. `region` on the Label table is the
        # arena id each cell came from, which is the authoritative "was this one used" set.
        used = np.zeros(int(np.asarray(region6).max()) + 2, dtype=bool)
        for r in rows:
            rid = int(r["region"])
            if 0 <= rid < used.size:
                used[rid] = True
        keep = np.asarray(region6, dtype=np.int64).copy()
        if bound == "mask":
            # every non-zero voxel is ONE arena here, so "used" is all-or-nothing and the
            # per-id mask above would zero a raster whose ids are not arena ids at all
            if not rows:
                keep[:] = 0
        else:
            keep[~used[np.clip(keep, 0, used.size - 1)]] = 0
        out = out.with_layer(Domain.VOXEL, "%s_areas" % name, keep)
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

def _columns_voronoi(params, modes, incoming):
    """All THREE tables this node writes, because they answer different questions.

    The territory **Label** table carries ``point_id`` (back to the seed that won it),
    ``region`` (the arena it was clipped to) and ``density``; the ``<name>_seeds`` **Point**
    table is the seeds that actually won one, carrying ``cell_id`` forward; and the optional
    **MESH** carries the element schema. Declaring only the Label half would leave the seed
    filter — the node's whole point, since the surviving dots are a different set from the
    input cloud — with an empty condition menu."""
    try:
        name = str((params or {}).get("name") or "voronoi")
        base = ("id", "m", "t", "c", "area", "z", "y", "x")
        out = on_layer(Domain.LABEL, name, base + ("point_id", "region", "density"))
        out += on_layer(Domain.POINT, "%s_seeds" % name,
                        ("id", "m", "t", "c", "z", "y", "x", "cell_id"))
        out += on_layer(Domain.MESH,
                        str((params or {}).get("mesh_name") or "voronoi_mesh"),
                        MESH_ELEMENT)
        return out
    except Exception:                        # pragma: no cover - defensive
        return ()

register_node(
    batch_aware(_compute_voronoi), op_key="analysis.voronoi", label="Voronoi Cells",
    adds_columns=_columns_voronoi,
    category="analysis",
    extra_layers=_layers_voronoi,
    # POINT is required in every mode — they are the seeds. The AREA is conditional, and
    # what it demands differs per bound: `per_region` goes through `_label_raster`, which
    # refuses a raster carrying no Label table, so it needs a whole Label INSTANCE (the
    # VOXEL raster and the LABEL table under one name); `mask` reads any non-zero Voxel
    # layer and never looks for the table; `frame` reads no layer at all. Declared
    # statically this had to be wrong in two states out of three, and the shipped
    # `frozenset({POINT})` was the one that warned about nothing — the node told you it
    # reads Points while silently also needing your labels (reported 2026-08-03).
    reads_domains=frozenset({Domain.POINT}),
    reads_domains_by_mode={"bound": {
        "per_region": frozenset({Domain.VOXEL, Domain.LABEL}),
        "mask": frozenset({Domain.VOXEL}),
        "frame": frozenset(),
    }},
    # POINT because of `<name>_seeds` — the surviving dots, so an overlay can show what
    # participated rather than the whole input cloud
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL, Domain.MESH, Domain.POINT}),
    inputs=[InDataset(description=
                      "The SEEDS branch, and the primary: calibration, axes and the domain "
                      "envelope all come from here, and so do the dots — one Voronoi cell per "
                      "point of its Point table. The output image is this branch's, which is "
                      "why the areas wire is composited on top rather than replacing it."),
            # The AREAS branch. Declared AFTER `data` so `data` stays the primary —
            # `graph.dataset_preds` sorts by declared socket position, so calibration and
            # domain propagation keep flowing from the seeds' chain no matter which edge the
            # user wired first (`wire-node-v2` §7d).
            #
            # It exists because this node is the one that genuinely COMBINES two domains:
            # the dots are a Point table and the areas are a Label/Voxel raster, and those
            # are routinely produced by different branches (nuclei on one channel, cell
            # bodies on another). With one input there was no way to converge them.
            # Optional — unwired, the area layer is read off the main input exactly as
            # before, so every graph that predates this socket is unaffected.
            # `view_source`: its image is composited into the Viewer under the seeds' one, so
            # viewing this node shows BOTH channels. Without it the payload carries only the
            # primary's image and the areas branch is invisible — "I can only see the UV
            # channel" (2026-08-04). It qualifies on the rule in `SocketSpec.view_source`:
            # this wire carries genuinely different content (another channel, segmented on
            # its own), not a second version of the primary's pixels.
            InDataset("areas", label="Areas", view_source=True, description=
                      "The AREAS branch — the layer the cells are clipped to, which is "
                      "routinely segmented from a DIFFERENT channel than the seeds. Optional: "
                      "leave it unwired and the area layer is read off the main wire instead. "
                      "Its image is composited into the Viewer so both channels are visible, "
                      "and its raster reaches the output as `<Output layer>_areas`. It must "
                      "address the same voxels as the main wire (same crop / resample / drift), "
                      "or the cells would be clipped to the wrong part of the image."),
            InString("points", "Seed points", field=False, default="",
                     layer_in=Domain.POINT,
                     description=
                     "The DOTS — one Voronoi cell per point of this table. A detection output, "
                     "or Label → Points if the seeds are segmented nuclei. LEAVE IT EMPTY and "
                     "the only Point table on the `data` wire is used, which is the usual case; "
                     "it only has to be set when two are present. Points are compared in "
                     "MICRONS, so an anisotropic z step cannot stretch the cells along z. In 2D "
                     "each dot only seeds the plane its z rounds to; in 3D every dot in the "
                     "volume competes."),
            InString("region", "Area layer", field=False, default="",
                     layer_in=Domain.VOXEL, layer_from="areas",
                     available_in={"bound": frozenset({"per_region", "mask"})},
                     description=
                     "Which layer on the AREAS wire the cells are clipped to — or on the main "
                     "wire when nothing is plugged into Areas. LEAVE IT EMPTY and the only "
                     "candidate on that wire is used (under `per_region` that means the only "
                     "Label INSTANCE, ignoring plain masks and distance fields; under `mask`, "
                     "the only Voxel layer); set it when several are present. Under "
                     "`per_region` this must be a LABEL raster and each region becomes its own "
                     "arena: a cell can only "
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
                     "join on id. A SECOND Voxel layer `<name>_areas` carries the area raster "
                     "the cells were clipped to — overlay that to see a territory against the "
                     "region that bounded it, since the output otherwise inherits only the "
                     "SEEDS' branch layers (whose raster is usually also called `labels`)."),
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
