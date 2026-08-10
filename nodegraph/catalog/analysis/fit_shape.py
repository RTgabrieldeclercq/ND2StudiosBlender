"""Fit Shape (``analysis.fit_shape``) — fit a convex body to each labelled object → a shape table of half-spaces, so a signed distance to each object's own SURFACE becomes available."""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, List

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    DimMode,
    Granularity,
    InDataset,
    InInt,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.labels import _label_raster
from nodegraph.catalog._shared.shapes import (
    SHAPE_MODELS,
    face_layer,
    fit_object_polytope,
    with_shape_provenance,
)

# ── Fit Shape: labelled regions → a convex body per object ─────────────────────
#
# What this adds that nothing in the catalog had: a **surface** per object, and therefore a
# SIGNED DISTANCE to it. Every existing per-object measure describes a region — its area, its
# centroid, its intensity — and the closest thing to a boundary was a centroid, which is a
# point. A point cannot say "how far outside this object are you, and in which direction",
# and that question is the leading term of the energy every downstream shape node minimises.
#
# The body is a convex POLYTOPE, stored as half-spaces `n_i . x + d_i <= 0`, because for that
# form
#
#     s(x) = max_i (n_i . x + d_i)
#
# is negative inside, zero on the boundary, exact on the faces, and has |grad s| = 1 almost
# everywhere. That last property is what makes an energy imbalance of dE microns move a
# boundary by exactly dE microns, which is the only reason a term weight can be CALIBRATED
# against a one-voxel budget rather than tuned until the picture looks right.
#
# Two tables come out, and the split is forced by the representation rather than chosen: the
# facet count varies per object, so it cannot live in a fixed-width per-object table. See
# `_shared/shapes.py` for why metadata and the MESH domain were both rejected.


def _compute_fit_shape(ctx: EvalContext) -> Dataset:
    """Fit a convex body to every object of a label raster → a shape table + a facet table."""
    ds: Dataset = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    model = modes.get("model", SHAPE_MODELS[0])
    if model not in SHAPE_MODELS:
        raise ValueError(f"fit_shape: unknown model {model!r} (expected one of {SHAPE_MODELS})")
    src = ctx.layer("labels")
    name = ctx.layer("name")
    min_voxels = max(1, int(ctx.params.get("min_voxels", 8)))
    is_3d = ctx.is_volume
    dims = 3 if is_3d else 2
    px = ctx.calib("pixel_size_um") or 1.0
    # Read z_step ONLY in 3D, so a 2D pull is not memo-fenced on a key it never consumes.
    zs = (ctx.calib("z_step_um") or 1.0) if is_3d else None
    vox = (zs, px, px) if is_3d else (px, px)

    raster6, zk = _label_raster(ds, src, node="fit_shape")
    if is_3d and zk == "plane_index":
        raise ValueError(
            f"fit_shape: the label instance {src!r} is per-PLANE (z_kind='plane_index') but "
            f"this node is on the 3D lever, so each object would be fitted as a flat slab and "
            f"every facet normal in z would be meaningless. Set dim=2D, or segment in 3D.")

    from scipy import ndimage as ndi

    rows: List[Dict[str, Any]] = []
    frows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    ctx.progress(0, len(units), "fitting shapes", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        lab = raster6[m, t, :, c] if is_3d else raster6[m, t, z, c]
        lab = np.asarray(lab)
        if lab.max() <= 0:
            ctx.progress(i + 1, len(units), "fitting shapes", frames=ax.t)
            continue
        # one find_objects pass, then each object is fitted inside its OWN bounding box —
        # never a full-field comparison per object, which is what makes this linear in the
        # foreground rather than in K x field
        boxes = ndi.find_objects(lab)
        for gid, sl in enumerate(boxes, start=1):
            if sl is None:
                continue
            sub = (lab[sl] == gid)
            n_vox = int(sub.sum())
            if n_vox < min_voxels:
                skipped["min_voxels"] = skipped.get("min_voxels", 0) + 1
                continue
            org = [float(s.start) * v for s, v in zip(sl, vox)]
            poly, extras = fit_object_polytope(sub, voxel_um=vox, model=model, origin_um=org)
            if poly is None:
                # Never fail a pull for one bad object: a coplanar or one-voxel-thick region
                # has no hull, and a bed has hundreds of objects. Count it and move on.
                skipped[str(extras)] = skipped.get(str(extras), 0) + 1
                continue
            cen = np.asarray(extras["centre_vox"], dtype=float) + \
                np.array([s.start for s in sl], dtype=float)
            cz = float(cen[0]) if is_3d else float(z)
            cy = float(cen[-2])
            cx = float(cen[-1])
            rows.append({"id": int(gid), "m": m, "t": t, "c": c,
                         "z": cz, "y": cy, "x": cx,
                         "n_support": int(poly.n_support), "n_faces": int(poly.n_faces),
                         "content_um": float(poly.content_um),
                         "boundary_um": float(poly.boundary_um),
                         "fill": float(extras["fill"]),
                         "rms_residual_um": float(extras["rms_residual_um"]),
                         "max_residual_um": float(extras["max_residual_um"])})
            # a representative point ON each facet: the foot of the perpendicular from the
            # body's centroid. Keeps the facet table coordinate-conformant AND is genuinely
            # useful — it is a point the facet passes through.
            cen_um = np.array([cz * vox[0], cy * vox[-2], cx * vox[-1]]) if is_3d else \
                np.array([cy * vox[0], cx * vox[1]])
            sc = poly.A @ cen_um + poly.b                        # signed distance per facet
            foot = cen_um[None, :] - sc[:, None] * poly.A
            for k in range(poly.n_faces):
                nz = float(poly.A[k, 0]) if is_3d else 0.0
                fz = (float(foot[k, 0] / vox[0]) if is_3d else float(z))
                frows.append({"id": len(frows), "m": m, "t": t, "c": c,
                              "z": fz,
                              "y": float(foot[k, -2] / vox[-2]),
                              "x": float(foot[k, -1] / vox[-1]),
                              "object": int(gid),
                              "n_z": nz, "n_y": float(poly.A[k, -2]),
                              "n_x": float(poly.A[k, -1]),
                              "offset_um": float(poly.b[k])})
        ctx.progress(i + 1, len(units), "fitting shapes", frames=ax.t)

    if skipped:
        ctx.log(f"fit_shape: {sum(skipped.values())} object(s) not fitted "
                + ", ".join(f"{v}x {k}" for k, v in sorted(skipped.items())))
    if not rows:
        raise ValueError(
            f"fit_shape: no object of {src!r} had at least min_voxels={min_voxels} voxels and a "
            f"non-degenerate hull, so there is nothing to fit. Lower min_voxels, or check that "
            f"the upstream segmentation produced regions rather than a single mask.")

    def _tab(rr: List[Dict[str, Any]], layer: str) -> StructureTable:
        int_cols = {"id", "m", "t", "c", "n_support", "n_faces", "object"}
        cols = {k: np.array([r[k] for r in rr],
                            dtype=(np.int64 if k in int_cols else float))
                for k in rr[0]}
        return StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk)

    out = ds.with_structure(_tab(rows, name))
    out = out.with_structure(_tab(frows, face_layer(name)))
    return with_shape_provenance(out, name, {
        "model": model, "dims": dims,
        "voxel_size_um": [float(v) for v in vox],
        "source": str(src), "n_objects": len(rows), "n_faces": len(frows)})


def _layers_fit_shape(node, env):
    """Predict both tables. Total and never raises — `propagate_meta` runs on every keystroke."""
    try:
        name = str((node.params or {}).get("name") or "shapes")
    except Exception:                                              # pragma: no cover
        name = "shapes"
    return ((Domain.LABEL, name), (Domain.LABEL, face_layer(name)))


register_node(
    _compute_fit_shape, op_key="analysis.fit_shape", label="Fit Shape", category="analysis",
    extra_layers=_layers_fit_shape,
    reads_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_domains=frozenset({Domain.LABEL}),
    inputs=[InDataset(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     description=
                     "Which label raster to fit — the output of Segmentation or Connected "
                     "Components. One body is fitted per label id, inside that id's own "
                     "bounding box, so cost scales with the foreground rather than with the "
                     "number of objects times the field size. A raster with no Label table is "
                     "refused: its non-zero voxels are one undivided region, and fitting it "
                     "would return a single body spanning the whole foreground."),
            InString("name", "Output shape layer", field=False, default="shapes",
                     layer_out=(Domain.LABEL,),
                     description=
                     "Base name for the two Label tables this node writes. `<name>` carries one "
                     "row per object (centre, facet count, the fitted body's own volume and "
                     "surface, and the fit residual), and `<name>_faces` carries one row per "
                     "FACET of the half-space form. The split is not a style choice: the facet "
                     "count varies per object, so it cannot fit a fixed-width per-object table. "
                     "Downstream shape nodes read both under this one name."),
            InInt("min_voxels", "Minimum voxels", default=8,
                  description=
                  "Objects with fewer voxels than this are skipped rather than fitted, and the "
                  "count is logged. A convex body in D dimensions needs at least D+1 points to "
                  "exist at all, and a handful more before its facet normals mean anything, so "
                  "a 3-voxel speck would otherwise produce a body whose surface is an artefact "
                  "of which voxels happened to be lit. Skipping leaves that object absent from "
                  "the shape table while it stays present in the label raster, which is the "
                  "honest outcome — no shape was measurable.")],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("model", list(SHAPE_MODELS), default="convex_hull", label="Model",
                description=
                "Which convex body to fit. Both choices are POLYTOPES, stored as half-spaces "
                "and evaluated by the same `max(n.x + d)` — that uniformity is the point, "
                "because it is what gives every downstream node one signed distance to reason "
                "about. A sphere or an ellipsoid is deliberately NOT offered: neither is a "
                "polytope, so each would need its own field evaluator, and declaring a choice "
                "that half-works is worse than not offering it.",
                choice_docs={
                    "convex_hull":
                        "The tightest convex body containing the object, fitted to its BOUNDARY "
                        "voxels (the hull of a solid equals the hull of its surface, so the "
                        "interior is free to discard). Exact on the faces, so a granule with "
                        "flat facets is described to within the sampling. Where the object is "
                        "genuinely concave the hull bridges the dent, which shows up as `fill` "
                        "below 1 and a large `rms_residual_um` — that is a measurement, not a "
                        "failure, and it is how a merged pair betrays itself.",
                    "obb":
                        "The oriented bounding box: principal axes from the inertia tensor, "
                        "then two planes per axis at the cloud's extent. Six facets in 3D, four "
                        "in 2D, regardless of how ragged the surface is. Coarser than the hull "
                        "and far more robust to a noisy boundary or a few stray voxels, which "
                        "makes it the right choice when the segmentation is known to be rough "
                        "or when a stable orientation matters more than a tight surface.",
                })],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Fit a convex body to each labelled object → a shape table of half-spaces, so "
                "a signed distance to each object's own SURFACE becomes available.")
