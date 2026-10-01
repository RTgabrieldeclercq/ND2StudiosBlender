"""Grow (``transform.grow_points``) — Seeds grown into regions: each Point becomes a disc (2D) / ellipsoid (3D) of a physical radius, or each Label region is grown from its OWN shape (outward from its border, scaled about its centroid, or replaced by a disc) → a Voxel label raster + its Label table; overlap resolved by union / nearest / merge."""

from __future__ import annotations

import numpy as np

from typing import Dict, List, Optional, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    DimMode,
    Granularity,
    InBool,
    InDataset,
    InFloat,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.structure import COORD_COLUMNS, StructureTable, label_components

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.labels import (
    _label_centroids,
    _label_raster,
    _point_layers,
    _resolve_label_instance,
    _resolve_layer,
)
from nodegraph.catalog._shared.units import to_pixels_v2

#: Neighbourhood used to fuse touching stamps under ``overlap=merge``. Deliberately NOT a
#: socket: a grown region is a thick blob, so 4- vs 8- (or 6- vs 26-) connectivity can only
#: differ where two regions touch at a single diagonal voxel, and a control that is live in
#: one of three overlap modes and changes almost nothing is the "live-looking knob" the socket
#: contract exists to prevent. Full connectivity is also the fusing answer the mode's name
#: promises — under `merge` the user has asked for touching discs to become one object.
_MERGE_CONNECTIVITY = {2: 8, 3: 26}

#: Which ``grow`` values read a Label raster rather than a Point table. The Mode is ONE
#: dropdown over the four states that actually exist (a point has no shape of its own, so
#: `points` admits only the stamp geometry) — see :func:`_compute_grow_points`.
_LABEL_GROW = frozenset({"surface", "scale", "stamp"})

#: The geometry-specific column each label branch writes, and the socket it reports. One
#: column rather than a shared `grow_um`, because a scale factor is not a µm distance and a
#: column whose unit depends on a dropdown is a join waiting to be read wrong.
_GEOM_COLUMN = {"surface": "distance_um", "stamp": "radius_um", "scale": "scale"}
def _ellipsoid_stamp(shape: Tuple[int, ...], c: np.ndarray, r: np.ndarray,
                     scale: Sequence[float], want_d2um: bool
                     ) -> Optional[Tuple[tuple, np.ndarray, Optional[np.ndarray]]]:
    """One ellipsoid footprint as ``(slices, inside, d2um)``, or ``None`` if it misses the
    unit entirely.

    ``c`` is the centre in **voxel** coordinates, ``r`` the per-axis radius in **voxels**, and
    ``scale`` the µm each voxel spans on each axis.

    **Two distances, and they are not the same distance.** Membership uses the *normalized*
    ellipsoid form ``Σ (dᵢ/rᵢ)² ≤ 1``, which is what makes an anisotropic stack grow a real
    physical sphere rather than a cube: a coarse z step gives a smaller ``r_z`` in voxels and
    the ellipsoid reaches fewer planes, all from the same µm radius. The ``nearest`` tie-break
    then uses *physical* µm distance ``Σ (dᵢ·scaleᵢ)²``, because "which seed is closer" is a
    question about the specimen, and comparing voxel counts across an anisotropic grid would
    hand a contested voxel to a seed several planes away over one in the same plane.

    **Bounded by the stamp, never by the frame** — that is the whole reason this node exists
    beside ``analysis.voronoi``. Each centre touches only the bounding box of its own
    ellipsoid, so the cost is O(N·R^ndim) rather than O(image): measured 15 ms against
    voronoi's 2071 ms for 2000 points at r=5 px on 2048², and the ratio widens as coverage
    falls (i.e. on exactly the sparse detections this is for)."""
    ndim = len(shape)
    lo = np.maximum(np.floor(c - r).astype(np.int64), 0)
    hi = np.minimum(np.ceil(c + r).astype(np.int64) + 1, np.asarray(shape, dtype=np.int64))
    if np.any(hi <= lo):                           # the ellipsoid misses this unit entirely
        return None
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(ndim))
    grids = np.ogrid[sl]
    norm = np.zeros((1,) * ndim)
    d2um = np.zeros((1,) * ndim)
    for i in range(ndim):
        d = grids[i].astype(float) - float(c[i])
        norm = norm + (d / max(float(r[i]), 1e-12)) ** 2
        if want_d2um:
            d2um = d2um + (d * float(scale[i])) ** 2
    return sl, norm <= 1.0, (d2um if want_d2um else None)
def _resolve_stamps(shape: Tuple[int, ...], stamps: Sequence, ids: np.ndarray,
                    overlap: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Paint pre-built footprints into one unit (a ``(Y,X)`` plane in 2D, a ``(Z,Y,X)``
    volume in 3D) under the ``overlap`` rule.

    ``stamps`` is aligned with ``ids`` — one ``(slices, inside, d2um)`` triple per region, or
    ``None`` for a region with no footprint here — and ``ids`` MUST be ascending, because
    that ordering *is* ``union``'s documented tie-break.

    Returns ``(labels, mask)``. ``mask`` is the boolean union and is only built for
    ``overlap="merge"``, whose region ids come from a connected-components pass the caller
    runs instead of from the stamps.

    Every geometry this node offers — an ellipsoid around a detection, an ellipsoid around a
    label's centroid, a label dilated outward from its own border, a label scaled about its
    centroid — reduces to a bounded boolean block plus a tie-break distance, so the three
    overlap rules are written ONCE here rather than re-derived per geometry. That is what
    makes `union`/`nearest`/`merge` mean the same thing on points and on labels, which is the
    whole argument for these being one node.

    ``nearest`` reproduces ``analysis.voronoi(bound=frame, max_distance_um=R)``'s partition
    exactly for point seeds, and that is not a coincidence to be re-derived: a voxel goes to
    the nearest seed among those whose ellipsoid covers it, and a seed farther away than R
    cannot cover it, so the local winner is the global one. The selftest pins the
    equivalence."""
    lab = np.zeros(shape, dtype=np.int64)
    best = np.full(shape, np.inf) if overlap == "nearest" else None
    mask = np.zeros(shape, dtype=bool) if overlap == "merge" else None
    for k in range(len(ids)):
        st = stamps[k]
        if st is None:
            continue
        sl, inside, d2um = st
        if overlap == "merge":
            mask[sl] |= inside
        elif overlap == "union":
            blk = lab[sl]                          # basic slicing ⇒ a view, writes through
            blk[inside & (blk == 0)] = ids[k]      # ascending id order ⇒ lower id wins
        else:
            blk, bb = lab[sl], best[sl]
            take = inside & (d2um < bb)
            blk[take] = ids[k]
            bb[take] = np.broadcast_to(d2um, take.shape)[take]
    return lab, mask
def _stamp_rep(stamp) -> Optional[tuple]:
    """One voxel of ``stamp``'s footprint, in FULL-unit coordinates — the representative
    used to ask "which fused clump did this source region end up in?".

    A footprint voxel, not the region's centroid: under ``stamp`` the disc need not cover the
    original centroid's neighbourhood at all, under ``scale`` with a factor below 1 the
    shrunk copy may miss it, and a concave region's centroid is routinely outside itself. Any
    of those would look up background and undercount ``n_regions`` silently."""
    if stamp is None:
        return None
    sl, inside, _d2 = stamp
    flat = np.asarray(inside).reshape(-1)
    if not flat.any():
        return None
    idx = np.unravel_index(int(np.argmax(flat)), np.asarray(inside).shape)
    return tuple(int(idx[d]) + sl[d].start for d in range(len(sl)))
def _label_footprints(lab0: np.ndarray, ids: np.ndarray, cen: np.ndarray, *, geometry: str,
                      radii: np.ndarray, reach_vox: np.ndarray, reach_um: np.ndarray,
                      factor: float, scale: Sequence[float], want_d2um: bool,
                      protect: bool) -> List[Optional[tuple]]:
    """The grown footprint of every region of ``lab0``, aligned with ascending ``ids``.

    ``cen`` is the per-region centroid in voxel coordinates (:func:`_label_centroids`), and
    each footprint is computed inside that region's own bounding box grown by the reach — the
    same "bounded by the stamp, never by the frame" cost curve the point path has, which is
    what keeps a 3000-cell segmentation from paying one whole-volume pass per object.

    The three geometries:

    * ``surface`` — every voxel within the reach of the region's own border. The distance is
      a Euclidean transform sampled with the real voxel size and NORMALIZED per axis by that
      axis's reach, so the structuring element is a µm ellipsoid: with the axial reach equal
      to the lateral one this is a true physical sphere on an anisotropic stack, and lowering
      the axial reach flattens it. Concavities, holes and elongation all survive — the object
      keeps its outline and simply gets thicker, which is the shape-conserving growth
      ``skimage.segmentation.expand_labels`` performs (there without the µm ellipsoid, and
      there fused with the nearest-label rule this node keeps separate as ``overlap``).
    * ``scale`` — the region's own outline resampled about its centroid by ``factor``. Growth
      is PROPORTIONAL to size rather than a fixed distance, so a large cell gains more µm
      than a small one; it is the literal "same shape, bigger" reading. Computed by inverse
      mapping (each candidate voxel asks where it came from) rather than forward mapping,
      because a forward map leaves a lattice of unfilled holes for any factor above 1.
    * ``stamp`` — the region is REPLACED by an ellipsoid at its centroid, i.e. the point
      geometry applied to label centroids. The original outline is discarded on purpose; this
      is the branch for "give every object the same nominal extent".

    ``protect`` intersects every footprint with *background plus the region's own voxels*, so
    no region can be grown over another region's interior. It is applied here rather than
    after the overlap rule because it is a statement about the SOURCE raster, not about which
    of two grown regions wins: under ``stamp`` a region's own voxels outside its disc are
    released to background (the outline was discarded, as promised) but still cannot be taken
    by a neighbour."""
    from scipy import ndimage as ndi
    ndim = lab0.ndim
    shape = lab0.shape
    boxes = ndi.find_objects(lab0)
    # µm per voxel divided by the µm reach on that axis ⇒ an EDT whose 1.0 iso-surface IS the
    # requested ellipsoid. `_ellipsoid_stamp` uses the same normalization for the same reason.
    norm_sampling = tuple(float(scale[i]) / max(float(reach_um[i]), 1e-12)
                          for i in range(ndim))
    isotropic = bool(np.allclose(reach_um, reach_um[0]))
    out: List[Optional[tuple]] = []
    for k in range(len(ids)):
        gid = int(ids[k])
        sl0 = boxes[gid - 1] if 0 <= gid - 1 < len(boxes) else None
        if sl0 is None:                            # id counted but absent (never, in practice)
            out.append(None)
            continue
        if geometry == "stamp":
            st = _ellipsoid_stamp(shape, cen[k], radii, scale, want_d2um)
            if st is None:
                out.append(None)
                continue
            sl, inside, d2um = st
        else:
            if geometry == "surface":
                pad = np.ceil(reach_vox).astype(np.int64) + 1
                lo = np.array([max(0, sl0[d].start - int(pad[d])) for d in range(ndim)])
                hi = np.array([min(shape[d], sl0[d].stop + int(pad[d]))
                               for d in range(ndim)])
            else:                                   # scale — the outline resampled about `c`
                lo = np.empty(ndim, dtype=np.int64)
                hi = np.empty(ndim, dtype=np.int64)
                for d in range(ndim):
                    a = cen[k][d] + (sl0[d].start - cen[k][d]) * factor
                    b = cen[k][d] + (sl0[d].stop - cen[k][d]) * factor
                    lo[d] = max(0, int(np.floor(min(a, b))) - 1)
                    hi[d] = min(shape[d], int(np.ceil(max(a, b))) + 2)
            if np.any(hi <= lo):
                out.append(None)
                continue
            sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(ndim))
            sub = lab0[sl]
            mine = sub == gid
            if geometry == "surface":
                dn = ndi.distance_transform_edt(~mine, sampling=norm_sampling)
                inside = dn <= 1.0
                # the physical tie-break: free when the reach is isotropic (the normalized
                # transform is then just µm/reach), one more pass when it is not
                d2um = ((dn * float(reach_um[0])) ** 2 if isotropic else
                        ndi.distance_transform_edt(~mine, sampling=scale) ** 2) \
                    if want_d2um else None
            else:
                grids = np.ogrid[sl]
                idx: List[np.ndarray] = []
                valid = np.ones((1,) * ndim, dtype=bool)
                for d in range(ndim):
                    u = cen[k][d] + (grids[d].astype(float) - cen[k][d]) / factor
                    valid = valid & (u >= -0.5) & (u <= shape[d] - 0.5)
                    idx.append(np.clip(np.rint(u).astype(np.int64), 0, shape[d] - 1))
                # `valid` matters: without it an out-of-array source coordinate is CLIPPED
                # onto the border voxel, which for a region touching the edge would smear it
                # along the whole edge instead of leaving it unscaled.
                inside = (lab0[tuple(idx)] == gid) & valid
                d2um = (ndi.distance_transform_edt(~mine, sampling=scale) ** 2
                        if want_d2um else None)
        if protect:
            here = lab0[sl]
            keep = (here == 0) | (here == gid)
            inside = inside & keep
            # a footprint emptied by `protect` is a real outcome (a region wholly enclosed by
            # its neighbours), not an error — it simply wins no voxels
        out.append((sl, inside, d2um))
    return out
def _grow_from_points(ctx: EvalContext, ds: Dataset, *, volumetric: bool, overlap: str,
                      scale: Sequence[float], base: np.ndarray, r_um: float
                      ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], tuple]:
    """The ``grow=points`` branch — one ellipsoid per detection. See
    :func:`_compute_grow_points` for the resolved spec."""
    ax = ds.axes
    ndim = len(scale)
    # the ONE Point table on the wire, whatever it is called (§4g)
    layer, note = _resolve_layer(
        _point_layers(ds), ctx.layer("points"), node="grow points", socket="points",
        what="Point table", where="the `data` input",
        remedy="these are the centres to grow, so wire a detection (detect.spots / "
               "detect.particles) or transform.label_to_points into `data`")
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
            if a.layer == layer}
    missing = sorted(k for k in COORD_COLUMNS if k not in cols)
    if missing:
        raise ValueError(
            f"grow points: the Point layer {layer!r} is missing the invariant column(s) "
            f"{missing}, so its detections have no position to grow from.")
    n = int(len(cols["id"]))
    ragged = sorted(k for k, v in cols.items() if len(v) != n)
    if ragged:
        raise ValueError(
            f"grow points: column(s) {ragged} on Point layer {layer!r} disagree in length "
            f"with 'id' ({n}) — every region would be grown from a misaligned row.")

    # a per-point radius COLUMN scales the whole ellipsoid, so an anisotropic 3D stamp keeps
    # its shape and only changes size — the column is a µm radius like the socket.
    rcol = str(ctx.params.get("radius_column", "") or "").strip()
    fell_back = 0
    if rcol:
        if rcol not in cols:
            raise ValueError(
                f"grow points: `radius_column` names {rcol!r}, which the Point layer "
                f"{layer!r} does not have (it carries {sorted(cols)}). Leave the socket "
                f"empty to use the `radius` value for every point, or point it at a real "
                f"column — analysis.measure can write one onto this table.")
        per = np.asarray(cols[rcol], dtype=float)
        usable = np.isfinite(per) & (per > 0.0)
        fell_back = int((~usable).sum())
        # a NaN or non-positive entry falls back to the SOCKET rather than dropping the
        # detection: the socket value is declared and visible, where a skipped row is an
        # object that quietly stopped existing.
        factor = np.where(usable, per / max(r_um, 1e-12), 1.0)
    else:
        factor = np.ones(n, dtype=float)
    radii = base[None, :] * factor[:, None]
    radius_um_row = np.where(factor != 1.0, factor * r_um, r_um)

    # ── addressing ───────────────────────────────────────────────────────────────
    pid = np.asarray(cols["id"], dtype=np.int64)
    if n and pid.min() < 0:
        raise ValueError(
            f"grow points: Point layer {layer!r} carries negative id(s) (min {int(pid.min())}), "
            f"and a label raster's ids must be positive — 0 is its background. Point ids are "
            f"written 0-based by every detector in this catalog, so a negative one means the "
            f"table was edited or joined by something that used -1 as a 'missing' marker.")
    mm = np.asarray(cols["m"], dtype=np.int64)
    tt = np.asarray(cols["t"], dtype=np.int64)
    cc = np.asarray(cols["c"], dtype=np.int64)
    zf = np.asarray(cols["z"], dtype=float)
    yf = np.asarray(cols["y"], dtype=float)
    xf = np.asarray(cols["x"], dtype=float)
    finite = np.isfinite(zf) & np.isfinite(yf) & np.isfinite(xf)
    inside_axes = (finite & (mm >= 0) & (mm < ax.m) & (tt >= 0) & (tt < ax.t)
                   & (cc >= 0) & (cc < ax.c))
    # In 2D a point grows on ITS OWN plane, so that plane index has to exist. In 3D `z` is a
    # coordinate inside the volume and the ellipsoid is simply clipped to it, exactly as y/x
    # are — hence the check is 2D-only.
    zi = np.where(finite, np.rint(np.where(finite, zf, 0.0)), 0).astype(np.int64)
    if not volumetric:
        inside_axes &= (zi >= 0) & (zi < ax.z)
    live = np.flatnonzero(inside_axes)
    if n and live.size == 0:
        raise ValueError(
            f"grow points: not one of the {n} rows on Point layer {layer!r} addresses a "
            f"(position, frame, channel"
            + (", plane" if not volumetric else "")
            + f") this Dataset has — its axes are m={ax.m}, t={ax.t}, z={ax.z}, c={ax.c}. The "
            f"points and this Dataset are not from the same chain: a frame slice, a channel "
            f"tap, a temporal stack or a Z-projection between the detector and here would do "
            f"it.")

    raster = np.zeros(ax.shape_for(Domain.VOXEL), dtype=np.int64)
    units = _units(ax, volumetric)
    notes: List[str] = []
    if note:
        notes.append("centres: " + note)
    if fell_back:
        notes.append(f"{fell_back} of {n} points had no usable {rcol!r} value and used the "
                     f"`radius` socket")
    skipped = int(n - live.size)
    if skipped:
        notes.append(f"{skipped} of {n} points address a frame/channel"
                     + ("" if volumetric else "/plane") + " this Dataset has not and were "
                     "skipped")
    for msg in notes:
        ctx.progress(0, max(1, len(units)), msg, frames=ax.t)

    parts: List[Dict[str, np.ndarray]] = []
    merge_next = 1                     # merge mints its own ids, so they need a global offset
    ctx.progress(0, max(1, len(units)), "growing regions", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        sel = live[(mm[live] == m) & (tt[live] == t) & (cc[live] == c)
                   & (True if volumetric else (zi[live] == z))]
        if sel.size:
            shape = ((ax.z, ax.y, ax.x) if volumetric else (ax.y, ax.x))
            pts = (np.column_stack([zf[sel], yf[sel], xf[sel]]) if volumetric
                   else np.column_stack([yf[sel], xf[sel]]))
            order = np.argsort(pid[sel], kind="stable")     # ascending id ⇒ union's tie-break
            sel, pts = sel[order], pts[order]
            stamps = [_ellipsoid_stamp(shape, pts[k], radii[sel][k], scale,
                                       overlap == "nearest")
                      for k in range(len(sel))]
            lab, mask = _resolve_stamps(shape, stamps, pid[sel] + 1, overlap)
            if overlap == "merge":
                lab = label_components(mask, _MERGE_CONNECTIVITY[ndim])[0]
                present = np.unique(lab)
                present = present[present > 0]
                if present.size:
                    lab = np.where(lab > 0, lab + (merge_next - 1), 0)
                    ids_here = present + (merge_next - 1)
                    merge_next += int(present.size)
                    # which detections landed in which clump: look each centre up in the
                    # fused raster. A centre can miss (two discs fused around it) only if it
                    # is off-grid, which `live` already excluded.
                    idx = tuple(np.clip(np.rint(pts[:, d]).astype(np.int64), 0,
                                        shape[d] - 1) for d in range(ndim))
                    owner = lab[idx]
                    counts = np.array([int((owner == g).sum()) for g in ids_here],
                                      dtype=np.int64)
                    rmean = np.array(
                        [float(np.mean(radius_um_row[sel][owner == g]))
                         if (owner == g).any() else float(r_um) for g in ids_here],
                        dtype=float)
                    cen, area = _label_centroids(lab, ids_here)
                    cz = cen[:, 0] if volumetric else np.full(len(ids_here), float(z))
                    cy = cen[:, 1] if volumetric else cen[:, 0]
                    cx = cen[:, 2] if volumetric else cen[:, 1]
                    parts.append({
                        "id": ids_here.astype(np.int64),
                        "m": np.full(len(ids_here), m, np.int64),
                        "t": np.full(len(ids_here), t, np.int64),
                        "c": np.full(len(ids_here), c, np.int64),
                        "area": area.astype(np.int64),
                        "n_points": counts, "radius_um": rmean,
                        "z": cz, "y": cy, "x": cx,
                    })
            else:
                present = np.unique(lab)
                present = present[present > 0]
                if present.size:
                    area = np.bincount(lab.ravel(),
                                       minlength=int(present.max()) + 1)[present]
                    # a point whose whole ellipsoid was claimed by a lower id (union) or a
                    # nearer seed (nearest) has NO region, so the table is built from the
                    # ids actually painted rather than from the row list.
                    slot = np.searchsorted(pid[sel] + 1, present)
                    parts.append({
                        "id": present.astype(np.int64),
                        "m": np.full(present.size, m, np.int64),
                        "t": np.full(present.size, t, np.int64),
                        "c": np.full(present.size, c, np.int64),
                        "area": area.astype(np.int64),
                        "point_id": pid[sel][slot].astype(np.int64),
                        "n_points": np.ones(present.size, np.int64),
                        "radius_um": radius_um_row[sel][slot],
                        "z": zf[sel][slot] if volumetric else np.full(present.size, float(z)),
                        "y": yf[sel][slot], "x": xf[sel][slot],
                    })
            if volumetric:
                raster[m, t, :, c] = lab
            else:
                raster[m, t, z, c] = lab
        ctx.progress(i + 1, max(1, len(units)), "growing regions", frames=ax.t)

    keys = (("id", "m", "t", "c", "area", "n_points", "radius_um", "z", "y", "x")
            + (() if overlap == "merge" else ("point_id",)))
    return raster, parts, keys
def _grow_from_labels(ctx: EvalContext, ds: Dataset, *, volumetric: bool, overlap: str,
                      geometry: str, scale: Sequence[float], base: np.ndarray,
                      reach_um: np.ndarray, factor: float, value: float
                      ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], tuple]:
    """The ``grow=surface|scale|stamp`` branch — each Label region grown from its OWN shape.

    ``base`` is the per-axis stamp radius in voxels (used by ``stamp``), ``reach_um`` the
    per-axis µm reach (used by ``surface``), ``factor`` the scale factor, and ``value`` the
    number written into the geometry column of every row.

    **Output ids are minted globally unique, and `source_id` is the join.** Not the source
    ids passed through, for a reason the 3D case makes unavoidable: with the lever on 2D a
    z-connected source region spans several planes, each plane is its own unit, and passing
    the id through would put several rows with the SAME id in one Label table — an id that no
    longer identifies a row is worse than a renumbering. So the resolution runs on source ids
    (which is what makes ``union``'s "lower id wins" mean the lower *source* region) and a
    lookup table renumbers at the end, exactly as ``analysis.label`` and ``analysis.voronoi``
    do. ``source_id`` carries the original, and it is the column to join back on."""
    ax = ds.axes
    ndim = len(scale)
    src, note = _resolve_label_instance(
        ds, ctx.layer("labels"), node=f"grow labels ({geometry})", socket="labels",
        remedy="these are the objects to grow, so run analysis.label (connected components), "
               "analysis.segment or a mask-volume node first — growing needs the raster AND "
               "its table, since a plain mask is one undivided region, not objects")
    lab6, _zk = _label_raster(ds, src, node=f"grow labels ({geometry})")
    lab6 = np.asarray(lab6)
    if not (lab6 > 0).any():
        raise ValueError(
            f"grow labels: the Label raster {src!r} is entirely background — there is no "
            f"region to grow. Check the node that produced it (a threshold that selected "
            f"nothing, or a minimum-size filter that removed everything).")

    raster = np.zeros(ax.shape_for(Domain.VOXEL), dtype=np.int64)
    units = _units(ax, volumetric)
    if note:
        ctx.progress(0, max(1, len(units)), "regions: " + note, frames=ax.t)
    reach_vox = np.asarray(reach_um, dtype=float) / np.asarray(scale, dtype=float)
    gcol = _GEOM_COLUMN[geometry]
    protect = bool(ctx.params.get("protect", True))

    parts: List[Dict[str, np.ndarray]] = []
    next_id = 1                      # ids are minted globally unique — see the docstring
    ctx.progress(0, max(1, len(units)), "growing regions", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        lab0 = (lab6[m, t, :, c] if volumetric else lab6[m, t, z, c]).astype(np.int64)
        ids = np.unique(lab0)
        ids = ids[ids > 0]                          # ascending ⇒ union's tie-break
        if ids.size:
            shape = lab0.shape
            cen, _area0 = _label_centroids(lab0, ids)
            stamps = _label_footprints(
                lab0, ids, cen, geometry=geometry, radii=base, reach_vox=reach_vox,
                reach_um=np.asarray(reach_um, dtype=float), factor=factor, scale=scale,
                want_d2um=(overlap == "nearest"), protect=protect)
            lab, mask = _resolve_stamps(shape, stamps, ids, overlap)
            if overlap == "merge":
                lab = label_components(mask, _MERGE_CONNECTIVITY[ndim])[0]
                present = np.unique(lab)
                present = present[present > 0]
                if present.size:
                    ids_here = np.arange(next_id, next_id + present.size, dtype=np.int64)
                    lut = np.zeros(int(present.max()) + 1, dtype=np.int64)
                    lut[present] = ids_here
                    lab = lut[lab]
                    next_id += int(present.size)
                    # which source regions landed in which clump — looked up at a voxel of
                    # each region's own FOOTPRINT (`_stamp_rep`), never at its centroid
                    reps = [_stamp_rep(st) for st in stamps]
                    owner = np.array([int(lab[r]) if r is not None else 0 for r in reps],
                                     dtype=np.int64)
                    counts = np.array([int((owner == g).sum()) for g in ids_here],
                                      dtype=np.int64)
                    cenm, area = _label_centroids(lab, ids_here)
                    cz = cenm[:, 0] if volumetric else np.full(present.size, float(z))
                    cy = cenm[:, 1] if volumetric else cenm[:, 0]
                    cx = cenm[:, 2] if volumetric else cenm[:, 1]
                    parts.append({
                        "id": ids_here,
                        "m": np.full(present.size, m, np.int64),
                        "t": np.full(present.size, t, np.int64),
                        "c": np.full(present.size, c, np.int64),
                        "area": area.astype(np.int64),
                        "n_regions": counts,
                        gcol: np.full(present.size, float(value)),
                        "z": cz, "y": cy, "x": cx,
                    })
            else:
                present = np.unique(lab)
                present = present[present > 0]
                if present.size:
                    ids_here = np.arange(next_id, next_id + present.size, dtype=np.int64)
                    lut = np.zeros(int(present.max()) + 1, dtype=np.int64)
                    lut[present] = ids_here
                    lab = lut[lab]
                    next_id += int(present.size)
                    cenm, area = _label_centroids(lab, ids_here)
                    cz = cenm[:, 0] if volumetric else np.full(present.size, float(z))
                    cy = cenm[:, 1] if volumetric else cenm[:, 0]
                    cx = cenm[:, 2] if volumetric else cenm[:, 1]
                    # a region whose whole footprint was claimed by a lower id (union), a
                    # nearer region (nearest) or `protect` has NO output row — the table is
                    # built from the ids actually painted, not from the source id list
                    parts.append({
                        "id": ids_here,
                        "m": np.full(present.size, m, np.int64),
                        "t": np.full(present.size, t, np.int64),
                        "c": np.full(present.size, c, np.int64),
                        "area": area.astype(np.int64),
                        "source_id": present.astype(np.int64),
                        "n_regions": np.ones(present.size, np.int64),
                        gcol: np.full(present.size, float(value)),
                        "z": cz, "y": cy, "x": cx,
                    })
            if volumetric:
                raster[m, t, :, c] = lab
            else:
                raster[m, t, z, c] = lab
        ctx.progress(i + 1, max(1, len(units)), "growing regions", frames=ax.t)

    keys = (("id", "m", "t", "c", "area", "n_regions", gcol, "z", "y", "x")
            + (() if overlap == "merge" else ("source_id",)))
    return raster, parts, keys
def _units(ax, volumetric: bool) -> List[tuple]:
    """The (m, t, z, c) units this node grows independently — one volume per (m,t,c) in 3D,
    one plane per (m,t,z,c) in 2D."""
    return ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
             for c in range(ax.c)] if volumetric else
            [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
             for z in range(ax.z) for c in range(ax.c)])
def _compute_grow_points(ctx: EvalContext) -> Dataset:
    """Grow seeds into measurable regions → a Voxel label raster and its Label table.

    Resolved spec (grilled 2026-08-04; labels branch 2026-08-04): category ``transform``,
    beside the other cross-domain constructors and the exact inverse of
    ``transform.label_to_points``; adds VOXEL + LABEL; a ``DimMode`` lever, a ``grow`` Mode
    choosing what is grown and how, and an ``overlap`` Mode resolving contested voxels;
    footprint WHOLE_PLANE (2D) / WHOLE_VOLUME (3D).

    **``grow`` is ONE dropdown over four states, not two orthogonal ones.** The obvious
    factoring — a `source` (points / labels) crossed with a `shape` (surface / scale / stamp)
    — describes six states of which only four exist: a Point has no outline to conserve or
    resample, so it admits the stamp geometry and nothing else. A product Mode would have
    left two impossible combinations selectable, and — worse — no ``available_in`` expression
    could then have gated `radius` correctly, since a gate is a conjunction over modes and
    `radius` is live under `points` OR `stamp`. One four-choice Mode makes every gate exact:

    * ``points`` — each detection becomes a disc (2D) / ellipsoid (3D) of a µm radius. The
      default, and the node's original behaviour, so every saved graph predating the labels
      branch resolves to exactly what it did before.
    * ``surface`` — each Label region grows OUTWARD FROM ITS OWN BORDER by a µm distance. The
      outline is conserved: concavities, holes and elongation all survive and the object
      simply gets thicker.
    * ``scale`` — each Label region's outline is resampled about its centroid by a factor, so
      growth is proportional to size rather than a fixed distance.
    * ``stamp`` — each Label region is REPLACED by an ellipsoid at its centroid, discarding
      its outline. Reachable as ``transform.label_to_points`` → this node's ``points``, and
      offered here because the two-node detour is not obvious and the round trip loses the
      per-region provenance this branch keeps in ``source_id``.

    **Why this is not ``analysis.voronoi``.** Voronoi with ``bound=frame`` and a positive
    ``max_distance_um`` already grows dots into discs, so the points branch is not new
    capability — it is a different cost curve and two semantics voronoi does not have.
    Voronoi tessellates every voxel of the frame and then clips, i.e. O(image) whatever the
    radius; this stamps bounding boxes, i.e. O(N·R^ndim). Measured: 2000 points at r=5 px on
    2048² takes 2071 ms through voronoi and ~15 ms here (139×), and the ratio grows as the
    detections get sparser. On top of that, voronoi always splits contested space on the
    perpendicular bisector and its reach is a single scalar, where this offers
    ``union``/``merge`` as well and takes a **per-point radius from a column**. Bounded growth
    (inside a mask or one region per arena) is deliberately NOT duplicated here: that IS
    voronoi.

    **Why the labels branch is not ``enhance.morphology`` or ``analysis.boundary_band``
    either.** Grey `dilate` moves every boundary in an IMAGE and knows nothing about objects —
    two neighbouring cells merge into one blob and no table comes out. `boundary_band` grows a
    band and then EXCLUDES the interior: it produces a shell around each object, which is the
    opposite output. This branch grows the object itself, keeps one id per object, resolves
    collisions with a stated rule, and emits the Label table that makes the result measurable.

    **Raster ids.** The points branch paints ``point_id + 1``: Point tables are 0-based
    (``structure.point_table`` numbers from ``np.arange(n)``) and 0 is a label raster's
    background, so using the ids directly would silently delete the first detection of every
    cloud. The table carries both ``id`` (the raster value) and ``point_id`` (the source row),
    which is the ``analysis.voronoi`` convention and the column every downstream join should
    use. The labels branches mint ids globally unique across every (m,t,c) and carry
    ``source_id`` instead — see :func:`_grow_from_labels` for why passing the source id
    through is not an option once the 2D lever splits a z-connected region across planes.

    **``merge`` breaks that correspondence on purpose**, and it is a contract difference
    rather than a detail: fused regions are one object, so there are fewer of them than
    sources, there is no ``point_id``/``source_id``, and ``n_points``/``n_regions`` records
    how many sources went into each clump.

    **Membership is an ellipsoid in µm, resolved into voxels per axis** — under ``points`` and
    ``stamp`` around a centre (:func:`_ellipsoid_stamp`), under ``surface`` around the
    region's own border via a normalized distance transform (:func:`_label_footprints`). Both
    make one µm value grow a real physical sphere on an anisotropic stack rather than a cube.

    Refusals rather than silent degradation: no point cloud (or no Label instance) on the
    wire; two candidates (the socket has to choose); a raster that carries no Label table, so
    its foreground is one undivided region; an all-background raster; a positive reach that
    the pixel size rounds below one voxel (the ``_require_window`` failure, in the place it
    actually bites here); a non-positive scale factor; a ``radius_column`` naming a column the
    table lacks; a negative point id; and a Point table whose rows all address a
    ``(position, frame, channel)`` this Dataset does not have. Rows individually out of range
    are skipped and counted on the rail, which is deliberately the same rule
    ``analysis.measure``'s point branch applies so the two cannot disagree."""
    ds = ctx.inputs[0]
    volumetric = ctx.is_volume
    modes = ctx.params.get("__modes__", {})
    grow = modes.get("grow", "points")
    overlap = modes.get("overlap", "union")
    name = ctx.layer("name")

    # ── reach: µm in, per-axis voxels out ────────────────────────────────────────
    px = ctx.calib("pixel_size_um") or 0.1
    stamped = grow in ("points", "stamp")
    r_um = float(ctx.params.get("radius", 5.0)) if stamped else 0.0
    d_um = float(ctx.params.get("distance", 2.0)) if grow == "surface" else 0.0
    lead, lead_name, lead_socket = ((r_um, "radius", "Radius") if stamped
                                    else (d_um, "distance", "Distance"))
    lead_lat = to_pixels_v2(lead, "um", pixel_size_um=px)
    if lead > 0.0 and lead_lat < 1.0:
        raise ValueError(
            f"grow: `{lead_name}` is {lead:g} µm, which is {lead_lat:.3g} px at "
            f"pixel_size_um={float(px):g} — below one voxel, so "
            + ("every point would grow into a single voxel instead of a region"
               if stamped else "no region would grow at all")
            + f". Raise {lead_socket} above {float(px):g} µm"
            + (", or use transform.rasterize_field if a one-voxel splat is really what you "
               "want" if stamped else "")
            + ". (Your data may be coarser than you think: this catalog's µm defaults are "
              "sized for ~0.1-0.3 µm/px, and a 10x plate is nearer 1.7.)")
    # z_step_um is read ONLY on the 3D path (R1: never fence a pull on a key the selected
    # branch cannot use) — in 2D each seed stays on its own plane and no axial reach exists.
    if volumetric:
        zs = ctx.calib("z_step_um") or 0.5
        scale: Tuple[float, ...] = (float(zs), float(px), float(px))
        rz_um = float(ctx.params.get("radius_z", r_um)) if stamped else 0.0
        dz_um = float(ctx.params.get("distance_z", d_um)) if grow == "surface" else 0.0
        base = np.array([to_pixels_v2(rz_um, "um_axial", z_step_um=zs), lead_lat, lead_lat]
                        if stamped else [0.0, 0.0, 0.0], dtype=float)
        reach_um = np.array([dz_um, d_um, d_um], dtype=float)
    else:
        scale = (float(px), float(px))
        base = np.array([lead_lat, lead_lat] if stamped else [0.0, 0.0], dtype=float)
        reach_um = np.array([d_um, d_um], dtype=float)

    if grow == "points":
        raster, parts, keys = _grow_from_points(
            ctx, ds, volumetric=volumetric, overlap=overlap, scale=scale, base=base,
            r_um=r_um)
    else:
        factor = float(ctx.params.get("scale", 1.5)) if grow == "scale" else 1.0
        if grow == "scale" and not (factor > 0.0):
            raise ValueError(
                f"grow labels (scale): `scale` is {factor:g}, and a region's outline can only "
                f"be resampled about its centroid by a POSITIVE factor — 0 or a negative "
                f"value has no geometric meaning. Use a value above 1 to grow (1.5 makes each "
                f"object half again as wide), exactly 1 to leave the shapes alone, or between "
                f"0 and 1 to shrink them.")
        value = {"surface": d_um, "stamp": r_um, "scale": factor}[grow]
        raster, parts, keys = _grow_from_labels(
            ctx, ds, volumetric=volumetric, overlap=overlap, geometry=grow, scale=scale,
            base=base, reach_um=reach_um, factor=factor, value=value)

    _FLOAT = ("radius_um", "distance_um", "scale", "z", "y", "x")
    if parts:
        table = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    else:
        table = {k: np.zeros(0, dtype=(float if k in _FLOAT else np.int64)) for k in keys}
    # The lever decides the output's dimensionality, as `analysis.label`'s does: 2D grows
    # each plane's regions as their own objects (plane_index), 3D grows z-connected volumes
    # (subpixel). It is NOT inherited from the source table/raster, because both combinations
    # are meaningful — a 2D-detected bead may legitimately want a spherical mask, and a
    # 3D-segmented cell may legitimately want a per-plane halo.
    zk = "subpixel" if volumetric else "plane_index"
    return (ds.with_layer(Domain.VOXEL, name, raster)
            .with_structure(StructureTable(Domain.LABEL, table, layer=name, z_kind=zk)))

def _columns_grow_points(params, modes, incoming):
    """The grown-region Label table. Its schema depends on which SOURCE was grown, exactly as
    the compute's ``keys`` tuple does: growing points counts how many fell in each region
    (``n_points``, with the seed ``radius_um``), growing labels counts how many regions merged
    (``n_regions``, with the geometry column the ``grow`` lever names). Both are declared —
    the branch is decided by which layer socket is wired, which is not a param this pass can
    read, and under-declaring one would make a real column unpickable."""
    try:
        name = str((params or {}).get("name") or "grown")
        base = ("id", "m", "t", "c", "area", "z", "y", "x")
        return on_layer(Domain.LABEL, name,
                        base + ("n_points", "radius_um", "n_regions", "distance_um",
                                "scale"))
    except Exception:                        # pragma: no cover - defensive
        return ()

register_node(
    batch_aware(_compute_grow_points), op_key="transform.grow_points", label="Grow",
    adds_columns=_columns_grow_points,
    category="transform",
    # Nothing is required unconditionally: the point branch reads no pixels at all (just
    # `ds.axes`, to size the raster it writes) while the three label branches read a whole
    # Label instance and no Point table. Declared statically this had to be wrong in three
    # states out of four — see `analysis.voronoi` for the same repair.
    reads_domains=frozenset(),
    reads_domains_by_mode={"grow": {
        "points": frozenset({Domain.POINT}),
        "surface": frozenset({Domain.VOXEL, Domain.LABEL}),
        "scale": frozenset({Domain.VOXEL, Domain.LABEL}),
        "stamp": frozenset({Domain.VOXEL, Domain.LABEL}),
    }},
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[
        InDataset(),
        # ships EMPTY: no literal is right for detect.spots (`spots`), detect.particles
        # (`particles`) and transform.label_to_points (`<labels>_points`) at once (§4g)
        InString("points", "Point layer", field=False, default="",
                 layer_in=Domain.POINT,
                 available_in={"grow": frozenset({"points"})},
                 description=
                 "Which detections to grow — the output of Spot Detection, Particle "
                 "Detection or Label to Points. One region comes out per point, so this "
                 "fixes the object COUNT; the radius below fixes their size. Leave it EMPTY "
                 "and the only Point table on `data` is used, which is what you want on a "
                 "single-branch graph; name one explicitly when the wire carries two. Only "
                 "read under Grow = `points`."),
        InString("labels", "Label layer", field=False, default="",
                 layer_in=Domain.VOXEL,
                 available_in={"grow": frozenset({"surface", "scale", "stamp"})},
                 description=
                 "Which segmentation to grow — a Label raster from Connected Components, "
                 "Segmentation or Watershed. It must be a real LABEL instance (a raster AND "
                 "its table): a plain threshold mask is one undivided region, so growing it "
                 "would report a single object for the whole foreground, and that is refused "
                 "rather than done. One region comes out per source region, joined back "
                 "through `source_id`. Leave it EMPTY and the only Label instance on `data` "
                 "is used. Hidden under Grow = `points`."),
        InFloat("radius", "Radius", unit="um", field=True, default=5.0,
                pick_kind="radius",
                available_in={"grow": frozenset({"points", "stamp"})},
                description=
                "How far each centre grows, in microns — the radius of the disc (2D) or "
                "ellipsoid (3D) it becomes. It sets every reported `area`/volume directly "
                "(area goes as the SQUARE of this in 2D), so it is the number that decides "
                "your measurements, not a cosmetic one. The default 5 µm is roughly a cell "
                "radius, the scale at which the usual job — growing nuclei into cell-sized "
                "territories — works; use ~1 µm for a halo around a spot. A positive value "
                "that the pixel size rounds below one voxel is REFUSED rather than silently "
                "producing one-voxel dots. Read under Grow = `points` (around each detection) "
                "and `stamp` (around each region's centroid); the other two branches grow "
                "from the region's own outline and use Distance or Scale instead."),
        InFloat("radius_z", "Radius Z", unit="um_axial", field=True, default=5.0,
                available_in={"dim": frozenset({"3D"}),
                              "grow": frozenset({"points", "stamp"})},
                description=
                "The same reach measured along Z, in microns, kept separate from Radius "
                "because microscope voxels are anisotropic — one shared value in VOXELS would "
                "cover far more physical distance through z than across a plane. Set it equal "
                "to Radius (the default) for a true sphere; lower it for a flattened, "
                "pancake-shaped region, which is often what a thin specimen actually warrants. "
                "With a coarse z step even a correct value here may span only one plane, and "
                "that is not an error — unlike the lateral radius it is not refused, because "
                "a single-plane extent through z is a legitimate physical answer. 3D only; in "
                "2D each seed grows within its own plane and nothing crosses z."),
        InFloat("distance", "Distance", unit="um", field=True, default=2.0,
                pick_kind="distance",
                available_in={"grow": frozenset({"surface"})},
                description=
                "How far outward each region grows FROM ITS OWN BORDER, in microns — a true "
                "Euclidean distance, so it means the same physical length in every direction "
                "even on an anisotropic stack. The outline is conserved: a concave or "
                "elongated cell stays concave or elongated, it just gets thicker by this "
                "much everywhere. It adds roughly `2 x distance` to every object's width and "
                "so RAISES every area/volume measured downstream — on a small object the "
                "area can easily double, which is the difference between a halo and a "
                "different measurement. 0 leaves the shapes exactly as they are. A positive "
                "value the pixel size rounds below one voxel is refused. Only read under "
                "Grow = `surface`."),
        InFloat("distance_z", "Distance Z", unit="um_axial", field=True, default=2.0,
                available_in={"dim": frozenset({"3D"}),
                              "grow": frozenset({"surface"})},
                description=
                "The outward reach along Z, in microns, kept separate from Distance for the "
                "same reason Radius Z is: one shared value in voxels would grow much farther "
                "physically through a coarse z step than across a plane. Equal to Distance "
                "(the default) grows a true isotropic shell; lower it to thicken each object "
                "mostly within its planes, which is usually right for a thin specimen whose "
                "z sampling is coarse. Set it to 0 and each object grows only within its own "
                "planes. 3D only."),
        InFloat("scale", "Scale", unit="", field=True, default=1.5,
                available_in={"grow": frozenset({"scale"})},
                description=
                "How much bigger each region's own outline is made, as a multiple about its "
                "centroid. Unlike Distance this growth is PROPORTIONAL to size — a large cell "
                "gains more microns than a small one, so the population's size ordering and "
                "relative shapes are preserved exactly. 1.5 makes every object half again as "
                "wide, which multiplies a 2D area by ~2.25 and a 3D volume by ~3.4; 1 leaves "
                "the shapes untouched; below 1 SHRINKS them (it is a resample, not a "
                "one-way grow). 0 or negative is refused. Only read under Grow = `scale`."),
        InString("radius_column", "Radius column", field=False, default="",
                 available_in={"grow": frozenset({"points"})},
                 description=
                 "Optional Point column holding a PER-DETECTION radius in microns, so a "
                 "size-varying population grows correctly instead of every object getting the "
                 "same extent. EMPTY (the default) uses the Radius value for every point. A "
                 "row whose value is missing or not positive falls back to that socket and the "
                 "count is reported on the progress bar — it is never dropped. In 3D the "
                 "column scales the whole ellipsoid, so the Radius/Radius Z ratio (the shape) "
                 "is preserved and only the size changes. analysis.measure can write a column "
                 "onto a Point table for this. Only read under Grow = `points`."),
        InBool("protect", "Protect regions", field=False, default=True,
               available_in={"grow": frozenset({"surface", "scale", "stamp"})},
               description=
               "Whether a growing region may only take voxels that were BACKGROUND in the "
               "source raster. ON (the default) no object can be grown over a neighbour's "
               "interior, so every source region survives and areas only ever go up — the "
               "right setting when the output is a measurement and objects must not "
               "disappear. OFF, the overlap rule decides everywhere, so a low-id or nearer "
               "region can consume a smaller neighbour outright and that neighbour then has "
               "NO row in the output table — closer to a raw dilation, and the setting to "
               "use when the source labels are seeds rather than objects. It only matters "
               "where two regions are closer together than the growth; on well-separated "
               "objects both give identical output. Hidden under Grow = `points`, which has "
               "no source raster to protect."),
        InString("name", "Output layer", field=False, default="grown",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "Name of the label raster AND its Label table — both are written under this "
                 "one name, so downstream nodes see a complete Label instance and Measure, "
                 "Track Objects and Object Metrics all accept it. Under `points` the raster's "
                 "ids are the source point id PLUS ONE (0 is a raster's background and point "
                 "ids start at 0) and the table keeps a `point_id` column; on the label "
                 "branches the ids are minted fresh and globally unique and the table keeps "
                 "`source_id`. Either way that column is the one to join back on — under "
                 "`merge` there is no such column, because fused regions come from several "
                 "sources."),
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("grow", ["points", "surface", "scale", "stamp"], default="points",
             label="Grow",
             description=
             "What is grown, and how — the choice that decides which input this node reads "
             "(a Point table or a Label raster) and what each output region's SHAPE comes "
             "from. `points` is the only branch that invents shape from nothing; the other "
             "three all start from an existing segmentation and differ in how much of its "
             "outline they keep.",
             choice_docs={
                 "points":
                     "Grow each detection of a Point table into a disc (2D) or ellipsoid "
                     "(3D) of the given radius. The only branch that reads Points, and the "
                     "only one that can take a per-detection radius from a column. Use it to "
                     "turn a detection-only graph into measurable objects — every object "
                     "gets the same nominal extent, because a dot carries no shape.",
                 "surface":
                     "Grow each Label region OUTWARD FROM ITS OWN BORDER by a µm distance, "
                     "measured as a true Euclidean distance. The outline is conserved — "
                     "concavities, holes and elongation all survive and the object simply "
                     "gets thicker everywhere. The branch to use for a halo, a dilated "
                     "measurement region, or closing a gap between segmented objects; every "
                     "object grows by the SAME distance regardless of its size.",
                 "scale":
                     "Resample each Label region's outline about its own centroid by a "
                     "factor. Also shape-conserving, but growth is PROPORTIONAL: a large "
                     "cell gains more microns than a small one, so relative sizes are "
                     "preserved exactly where `surface` compresses them. Use it when the "
                     "question is about a fixed fraction of each object (a 50%-larger "
                     "territory) rather than a fixed physical reach. A factor below 1 "
                     "shrinks.",
                 "stamp":
                     "REPLACE each Label region with an ellipsoid at its centroid, "
                     "discarding its outline entirely. Use it to normalize a ragged "
                     "segmentation to one nominal size, or to sample a fixed-radius "
                     "neighbourhood around each object's centre. It is the `points` geometry "
                     "applied to label centroids, and it is the one label branch whose "
                     "output areas do not depend on the input shape at all.",
             }),
        Mode("overlap", ["union", "nearest", "merge"], default="union", label="Overlap",
             description=
             "What happens where two grown regions would cover the same voxel — the choice "
             "that decides whether you get overlapping stamps, a tessellation, or fused "
             "clumps. It only matters when sources are closer together than twice the "
             "growth; below that density all three give identical output. It changes the "
             "reported `area` of crowded objects and, under `merge`, the object COUNT itself.",
             choice_docs={
                 "union":
                     "Each region keeps its full growth and the LOWER source id wins the "
                     "contested voxels. The right answer for a fixed-size halo or a "
                     "sampling ring per source, where every object should have the same "
                     "nominal extent; crowded objects then report an area smaller than the "
                     "full shape, since the overlap is charged to one of them.",
                 "nearest":
                     "Contested voxels go to whichever source is physically closer — the "
                     "nearest centre under `points`/`stamp`, the nearest region SURFACE under "
                     "`surface`/`scale` — so neighbours meet on a clean boundary and the "
                     "result is a tessellation capped at the growth. On points this is "
                     "identical to Voronoi Cells with Bound=frame and Max reach set to the "
                     "radius, but bounded by the stamps rather than the whole frame — "
                     "measured ~139x faster on a sparse 2048x2048 field.",
                 "merge":
                     "Touching regions FUSE into one object with one id, so a cluster of "
                     "sources becomes a single region and the object count drops. Use it "
                     "to turn scattered objects into a connected mask of occupied territory. "
                     "It is the one option that breaks the source↔region correspondence: "
                     "there is no `point_id`/`source_id` column, and `n_points`/`n_regions` "
                     "records how many sources each fused region came from instead.",
             }),
    ],
    # The footprint describes the raster this node WRITES, and — on the label branches — the
    # one it reads. Never TILEABLE, for two independent reasons: a grown region straddles
    # tile boundaries, and `merge`'s connected-components pass is a whole-unit operation by
    # definition (§9 — structure is never computed per tile).
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Grows seeds into measurable objects — a Voxel label raster AND its Label "
                "table. `points` turns each detection of a Point table into a disc (2D) or "
                "ellipsoid (3D) of a physical µm radius; `surface`, `scale` and `stamp` grow "
                "an existing Label segmentation from its OWN shape — outward from each "
                "region's border by a µm distance (outline conserved), by resampling that "
                "outline about its centroid (growth proportional to size), or by replacing "
                "it with an ellipsoid. `overlap` decides contested voxels: `union` "
                "(fixed-size stamps, lower id wins), `nearest` (a tessellation capped at the "
                "growth, ~139x cheaper than the equivalent Voronoi Cells) or `merge` "
                "(touching regions fuse into one), and `protect` keeps growth out of a "
                "neighbour's interior. The inverse of transform.label_to_points.")
