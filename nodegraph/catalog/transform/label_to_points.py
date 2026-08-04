"""Label → Points (``transform.label_to_points``) — Every label region → one Point at its centre (centroid / guaranteed-inside / intensity-weighted), keeping a `label` column back to the region."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from typing import Dict, List, Sequence

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _label_centroids, _label_raster
from nodegraph.catalog._shared.raw_measure import _InRaw, _intensity_provider

def _snap_inside(vol: np.ndarray, ids: np.ndarray, centroids: np.ndarray,
                 scale: Sequence[float]) -> np.ndarray:
    """Move each centroid to the voxel of its **own** label nearest it (µm distance).

    The point this guarantees is membership: a C-shaped, annular or forked region has its
    centre of mass in background, so a plain centroid is a dot that does not lie in the
    object it names — which silently breaks every consumer that looks the point back up in
    the raster (``bridges.containing_label``, and ``analysis.voronoi``'s per-region arena
    assignment, which would drop the seed entirely).

    Distance is measured in µm via ``scale`` so an anisotropic z step cannot make a voxel
    several planes away look closer than one in the same plane. Ties resolve to the first
    voxel in raster order, which keeps the result deterministic."""
    k = len(ids)
    flat = np.asarray(vol).reshape(-1)
    nz = np.flatnonzero(flat > 0)                # see _label_centroids on why not `!= 0`
    if k == 0 or nz.size == 0:
        return centroids
    slot = np.searchsorted(ids, flat[nz])
    coords = np.unravel_index(nz, np.asarray(vol).shape)
    d2 = np.zeros(nz.size, dtype=float)
    for i, (c, sc) in enumerate(zip(coords, scale)):
        d2 += ((c.astype(float) - centroids[slot, i]) * float(sc)) ** 2
    # lexsort by (d2 within slot) then take each slot's first row — the same answer as
    # `np.minimum.at` + a second pass, in one sort and without ufunc.at's known slowness.
    order = np.lexsort((d2, slot))
    s = slot[order]
    first = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    pick = np.full(k, -1, dtype=np.int64)
    pick[s[first]] = order[first]
    out = centroids.copy()
    have = pick >= 0
    for i, c in enumerate(coords):
        out[have, i] = c[pick[have]].astype(float)
    return out
def _layers_label_to_points(params, modes):
    """The Point layer this node writes when ``name`` is left empty (V2.11 ``extra_layers``).

    ``layer_out`` covers the explicit case; it cannot express the auto-derived one, whose
    name comes from ANOTHER param (the ``analysis.extract_boundary`` precedent). Runs inside
    ``propagate_meta`` on every keystroke, so it must never raise."""
    try:
        if str((params or {}).get("name") or "").strip():
            return ()                       # the socket's own layer_out announced it
        labels = str((params or {}).get("labels") or "") or "labels"
        return ((Domain.POINT, f"{labels}_points"),)
    except Exception:                        # pragma: no cover - defensive
        return ()
def _compute_label_to_points(ctx: EvalContext) -> Dataset:
    """One **Point** per region of a Label raster — the constructive Label→Point hop.

    Resolved spec: category transform (beside the other cross-domain constructors); op
    ``transform.label_to_points``; reads VOXEL + LABEL, adds POINT. The output table carries
    the invariant ``id,m,t,c,z,y,x`` schema plus a **``label``** column holding the source
    region id, so the dots stay joinable to the region table they came from. ``id`` is a
    fresh globally-unique index (the ``detect.particles`` / ``extract_boundary`` convention),
    because Point ids are what ``track.link``'s point target and ``analysis.voronoi`` key on
    and a per-unit region id is not unique across (m,t,c).

    **No 2D/3D lever** (`wire-node-v2` §7b): the Label instance's own ``z_kind`` decides.
    ``plane_index`` labels are per-plane, so each region yields a point on ITS plane with
    ``z`` = the plane index; ``subpixel`` labels are volumetric and yield a true 3D centroid.
    A lever here could contradict the segmentation that produced the raster, which is the
    failure ``transform.rasterize_field`` avoids the same way. Footprint is a fixed
    ``WHOLE_VOLUME``/``{z,y,x}`` superset; the compute loops the right unit internally.

    The ``position`` Mode picks WHICH point: the region's ``centroid``; the region voxel
    nearest that centroid (``inside`` — guaranteed to lie in a concave/annular region); or
    the intensity-weighted centre of mass (``weighted``, which reads pixels, from the
    optional ``raw`` input when one is wired). ``raw`` is refused on the two geometric modes
    rather than silently ignored."""
    ds = ctx.inputs[0]
    ax = ds.axes
    layer = ctx.layer("labels")
    out_layer = ctx.layer("name") or f"{layer}_points"
    mode = ctx.params.get("__modes__", {}).get("position", "centroid")
    raster6, zk = _label_raster(ds, layer, node="label to points")
    volumetric = zk != "plane_index"
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    scale = (zs, px, px) if volumetric else (px, px)

    prov = None
    if mode == "weighted":
        prov, _on_raw = _intensity_provider(ctx, ds)
        if prov is None:
            raise ValueError(
                "label to points (position=weighted) needs an image to weight by — wire an "
                "image Dataset into `data` or `raw`, or switch position to `centroid`.")
    elif ctx.input("raw") is not None:
        raise ValueError(
            f"label to points: the `raw` input is only read by position=weighted, and this "
            f"node is set to {mode!r} — a purely geometric centroid does not look at "
            "pixels. Unwire `raw`, or set position to `weighted`.")

    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if volumetric else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    parts: List[Dict[str, np.ndarray]] = []
    ctx.progress(0, len(units), "locating regions", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        vol = np.asarray(raster6[m, t, :, c] if z is None else raster6[m, t, z, c])
        ids = np.unique(vol)
        ids = ids[ids > 0]
        if ids.size:
            weights = None
            if prov is not None:
                weights = np.asarray(
                    prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                    if z is None else
                    prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), dtype=float)
            cen, _counts = _label_centroids(vol, ids, weights=weights)
            if mode == "inside":
                cen = _snap_inside(vol, ids, cen, scale)
            tbl = (point_table(cen, m=m, t=t, c=c, z_kind="subpixel", layer=out_layer)
                   if z is None else
                   point_table(cen, z=z, m=m, t=t, c=c, z_kind="plane_index",
                               layer=out_layer))
            cols = dict(tbl.columns)
            cols["label"] = ids.astype(np.int64)
            parts.append(cols)
        ctx.progress(i + 1, len(units), "locating regions", frames=ax.t)

    zk_out = "subpixel" if volumetric else "plane_index"
    if not parts:
        empty = point_table(np.zeros((0, 3 if volumetric else 2)), z_kind=zk_out,
                            layer=out_layer)
        merged = replace(empty, columns={**empty.columns,
                                        "label": np.zeros(0, dtype=np.int64)})
    else:
        cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        cols["id"] = np.arange(len(cols["label"]), dtype=np.int64)   # global-unique
        merged = StructureTable(Domain.POINT, cols, layer=out_layer, z_kind=zk_out)
    return ds.with_structure(merged)
register_node(
    _compute_label_to_points, op_key="transform.label_to_points",
    label="Label → Points", category="transform",
    extra_layers=_layers_label_to_points,
    reads_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(), _InRaw(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     description=
                     "Which label raster to reduce to dots — a Connected Components or "
                     "Segmentation output. One point comes out per region, so the Point table "
                     "is as long as the object COUNT (unlike Extract Boundary, which scales "
                     "with perimeter). It must be a real label raster: a plain binary mask "
                     "carries no Label table and is refused, because its whole foreground "
                     "would collapse to a single dot."),
            InString("name", "Output layer", field=False, default="",
                     layer_out=(Domain.POINT,),
                     description=
                     "Name of the Point table this node writes. EMPTY (the default) "
                     "auto-derives it as the label layer's name plus `_points`, so renaming "
                     "the segmentation keeps a matching dot-cloud name without a second edit. "
                     "Each point also keeps a `label` column with the id of the region it came "
                     "from, so the dots stay joinable to that region's measurements.")],
    outputs=[OutDataset()],
    modes=[Mode("position", ["centroid", "inside", "weighted"], default="centroid",
                label="Position",
                description=
                "WHERE in each region its point is placed. Two options are purely geometric "
                "and read no pixels; `weighted` reads intensity (from the optional raw input "
                "when one is wired), and a raw input is refused on the other two rather than "
                "silently ignored. The choice matters most for concave, annular or forked "
                "regions, where the centre of mass can fall outside the object.",
                choice_docs={
                    "centroid":
                        "The region's geometric centre of mass — the average of its voxel "
                        "positions. Exact and cheap, and the natural representative for a "
                        "convex blob. For a C-shaped or annular region it lands in the HOLE, "
                        "i.e. on a voxel that is not part of the object.",
                    "inside":
                        "The region's own voxel nearest to that centroid, measured in µm so "
                        "an anisotropic z step cannot pick a plane too far away. Guaranteed "
                        "to lie INSIDE the region, which is what every consumer that looks "
                        "the point back up in the raster needs — Voronoi's per-region arena "
                        "and the containing-label bridge both drop a seed that misses.",
                    "weighted":
                        "The intensity-weighted centre of mass: bright voxels pull the point "
                        "toward them. The right answer when the point should mark where the "
                        "SIGNAL is rather than where the mask's middle is — a polarized cell, "
                        "an off-centre nucleolus. It reads pixels, so it is affected by any "
                        "enhancement upstream unless you wire the raw input.",
                })],
    # a fixed superset: the unit (plane vs volume) is inherited from the Label instance's
    # own z_kind provenance, not from a lever, so there is no per-dim map to resolve
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Every label region → one Point at its centre (centroid / guaranteed-inside "
                "/ intensity-weighted), keeping a `label` column back to the region. 2D vs "
                "3D is inherited from the Label instance's own provenance. The seed cloud "
                "for Cluster Points, Tessellate, point tracking and Voronoi Cells.")
