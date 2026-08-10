"""shapes — the fitted-shape schema shared by the shape nodes.

A fitted convex body is stored as **two** Label tables under one base name, and that split is
forced by the representation rather than chosen:

* ``<name>`` — **one row per object**: where it is, how big, and how well the fit describes it.
  Fixed width, so it behaves like every other instance table.
* ``<name>_faces`` — **one row per FACET**: the half-space form ``nᵢ·x + dᵢ ≤ 0``.

The facet count is not fixed. A convex hull of a granule carries anywhere from 5 to 40 planes
depending on how many support points survive, so it cannot live in a per-object table without
either a wasteful fixed budget or a variable column set. The two alternatives were both worse:

* **Dataset metadata.** ``memo.py:166`` folds ``payload.metadata`` into the memo digest, so a
  few thousand floats per layer would be hashed on every pull — and the metadata dict is the
  calibration schema's home, which is not where a geometry belongs.
* **The MESH domain.** Carrying per-element shape parameters there means editing
  ``nodegraph/mesh.py``, the one file holding the CSR / object-dtype / memo-hash invariants, to
  smuggle a model through a topology domain.

A second table needs **zero core changes**: ``Domain.TRACK``'s membership table already ships
with a non-instance column set, so a table whose rows are facets rather than objects is an
established, tested shape, and it inherits the spreadsheet, the content hash, the Arrow export
and the ``z_kind`` provenance for free.

## Units, and the one convention that is load-bearing

``z, y, x`` on the object table are **voxel** coordinates, because every other structure table
in the repo is and the Viewer depends on it. Facet normals are dimensionless and facet offsets
are in **µm**, because ``s(x) = maxᵢ(nᵢ·x + dᵢ)`` is only a physical distance — and only has
``|∇s| = 1`` — when it is evaluated in µm. Mixing the two would give a "distance" that means
40 µm along z and 1.72 µm along x on this acquisition.

``__shape_provenance__`` stamps the ``voxel_size_um`` and ``dims`` the fit was made with, keyed
by layer name, following :data:`nodegraph.mesh.MESH_PROVENANCE_KEY`. A consumer inherits the
voxel size the fit used instead of re-reading ``ctx.calib`` and silently disagreeing after a
recalibration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain

__all__ = ["SHAPE_PROVENANCE_KEY", "SHAPE_COLUMNS", "SHAPE_FACE_COLUMNS", "FACE_SUFFIX",
           "SHAPE_MODELS", "face_layer", "shape_columns", "face_columns",
           "with_shape_provenance", "shape_provenance", "read_shape_planes",
           "fit_object_polytope"]

#: metadata key holding the per-shape-layer provenance stamp. Namespaced and
#: non-calibration, so ``strict_reads`` passes a consumer's read straight through.
SHAPE_PROVENANCE_KEY = "__shape_provenance__"

#: suffix of the companion facet table
FACE_SUFFIX = "_faces"

#: the shape models this schema can represent. BOTH are convex polytopes, which is the whole
#: point — the field evaluator is ``max(A·x + b)`` and nothing else. A sphere or an ellipsoid
#: is NOT a polytope and would need a different evaluator, so neither is offered here rather
#: than declared and half-supported (the charter's dead-control rule).
SHAPE_MODELS: Tuple[str, ...] = ("convex_hull", "obb")

#: one row per object
SHAPE_COLUMNS: Tuple[str, ...] = (
    "id", "m", "t", "c", "z", "y", "x",          # centre, in VOXELS (Viewer convention)
    "n_support",           # support points that survived the fit
    "n_faces",             # facets in the companion table
    "content_um",          # µm³ in 3D, µm² in 2D — the fitted body's own measure
    "boundary_um",         # µm² in 3D, µm in 2D
    "fill",                # object voxels / fitted body measure. 1.0 = convex; < 1 = concave
    "rms_residual_um",     # RMS |s| over the object's boundary voxels — this is σ_rough
    "max_residual_um",
)

#: one row per facet
SHAPE_FACE_COLUMNS: Tuple[str, ...] = (
    "id", "m", "t", "c", "z", "y", "x",          # a representative point ON the facet, VOXELS
    "object",              # the object id this facet bounds
    "n_z", "n_y", "n_x",   # outward UNIT normal (n_z is 0.0 on a 2D fit)
    "offset_um",           # d, in µm: the interior satisfies n·x + d <= 0 with x in µm
)


def face_layer(name: str) -> str:
    """The companion facet table's layer name for a shape layer ``name``."""
    return f"{name}{FACE_SUFFIX}"


def shape_columns() -> Tuple[str, ...]:
    return SHAPE_COLUMNS


def face_columns() -> Tuple[str, ...]:
    return SHAPE_FACE_COLUMNS


def with_shape_provenance(ds: Dataset, layer: str, stamp: Dict[str, Any]) -> Dataset:
    """Stamp ``layer``'s fit provenance, merging rather than replacing other layers' stamps."""
    pmap = dict(ds.metadata.get(SHAPE_PROVENANCE_KEY, {}))
    pmap[layer] = dict(stamp)
    return ds.with_metadata(**{SHAPE_PROVENANCE_KEY: pmap})


def shape_provenance(ds: Dataset, layer: str) -> Dict[str, Any]:
    """``layer``'s fit provenance, or ``{}``. Total and never raises."""
    return dict(ds.metadata.get(SHAPE_PROVENANCE_KEY, {}).get(layer, {}))


def read_shape_planes(ds: Dataset, name: str, *, node: str) -> Dict[int, np.ndarray]:
    """``{object_id: (F, dims+1) planes}`` from a shape layer's companion facet table.

    The inverse of what :mod:`nodegraph.catalog.analysis.fit_shape` writes, and the one route
    a consumer should use — it validates that the facet table is present and that the
    dimensionality agrees with the provenance stamp, rather than letting a 2D fit be evaluated
    on a 3D grid with a silently-dropped axis.
    """
    fl = face_layer(name)
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.LABEL)
            if a.layer == fl}
    if "object" not in cols:
        have = sorted({a.layer for a in ds.layers_on(Domain.LABEL) if a.layer})
        raise ValueError(
            f"{node}: no facet table {fl!r} for shape layer {name!r} — run "
            f"analysis.fit_shape upstream. Label layers present: {have}.")
    prov = shape_provenance(ds, name)
    dims = int(prov.get("dims") or 3)
    obj = cols["object"].astype(np.int64)
    if dims == 2:
        A = np.stack([cols["n_y"], cols["n_x"]], axis=1)
    else:
        A = np.stack([cols["n_z"], cols["n_y"], cols["n_x"]], axis=1)
    pl = np.c_[A, cols["offset_um"]]
    return {int(g): pl[obj == g] for g in np.unique(obj)}


def fit_object_polytope(mask: np.ndarray, *, voxel_um: Sequence[float], model: str,
                        origin_um: Optional[Sequence[float]] = None):
    """Fit one object's voxel mask, returning ``(Polytope, extras)`` or ``(None, reason)``.

    ``mask`` is a boolean array of the object's own bounding box; ``origin_um`` is that box's
    corner in µm so the returned body is in the full field's coordinates.

    Only the object's **boundary** voxels are hulled, not its interior. The hull of a solid and
    the hull of its surface are identical, and a granule is mostly interior — dropping it cuts
    the point count by roughly an order of magnitude for free.

    ``extras["rms_residual_um"]`` is the RMS of ``|s|`` over those boundary voxels, which is
    exactly the ``σ_rough`` the shape-slack hinge's ``τ = 3σ_rough`` is defined from — measured
    from the data rather than set as a knob. For a genuinely convex body it is a fraction of a
    voxel; for one with a concave face it is large, and that is the signal.
    """
    from nodegraph.kernels.convex_polytope import (POLY_OK, fit_polytope,
                                                   polytope_from_planes, signed_distance)
    d = int(mask.ndim)
    vox = np.asarray(voxel_um, dtype=float).reshape(d)
    org = (np.zeros(d) if origin_um is None
           else np.asarray(origin_um, dtype=float).reshape(d))
    idx = np.array(np.nonzero(mask), dtype=float).T
    if idx.shape[0] < d + 1:
        return None, "too_few_voxels"
    pts_all = idx * vox[None, :] + org[None, :]

    # boundary voxels only: a voxel with a background face-neighbour
    try:
        from scipy.ndimage import binary_erosion
        shell = mask & ~binary_erosion(mask)
    except ImportError:                                            # pragma: no cover
        shell = mask
    sidx = np.array(np.nonzero(shell), dtype=float).T
    if sidx.shape[0] < d + 1:
        sidx = idx
    # Hull the voxel CORNERS, not the centres. A voxel is a box, and the object is the union of
    # its boxes; hulling centres loses half a voxel all the way round, so the fitted body comes
    # out systematically too small. Measured, that put `fill` (object measure / fitted measure)
    # at a median of 1.016 — above 1, which is impossible for a body that contains its object,
    # and fatal for a split proposal that triggers on `fill` dropping below a threshold. With
    # corners the hull provably CONTAINS the union of boxes, so `fill <= 1` by construction.
    # The cost is 2**d times the boundary points into an O(n log n) hull, on the shell only.
    off = np.array(np.meshgrid(*[[-0.5, 0.5]] * d, indexing="ij")).reshape(d, -1).T
    pts = ((sidx[:, None, :] + off[None, :, :]).reshape(-1, d) * vox[None, :]
           + org[None, :])

    pts_all = ((idx[:, None, :] + off[None, :, :]).reshape(-1, d) * vox[None, :]
               + org[None, :])
    if model == "obb":
        # An oriented bounding box IS a polytope, so it uses the same evaluator: principal
        # axes from the inertia tensor of the voxel cloud, then two planes per axis at the
        # cloud's extent along it. Coarser than a hull and far more robust to a ragged
        # surface, which is why it is offered.
        cen = pts_all.mean(axis=0)
        cov = np.cov((pts_all - cen).T)
        cov = np.atleast_2d(cov)
        _w, V = np.linalg.eigh(cov)
        V = V[:, ::-1]                                   # descending eigenvalue
        if np.linalg.det(V) < 0:
            V[:, -1] *= -1.0                             # keep a right-handed frame
        proj = (pts_all - cen) @ V
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        rows = []
        for j in range(d):
            rows.append(np.r_[V[:, j], -(float(hi[j]) + float(V[:, j] @ cen))])
            rows.append(np.r_[-V[:, j], (float(lo[j]) + float(V[:, j] @ cen))])
        poly = polytope_from_planes(np.array(rows)[:, :d], np.array(rows)[:, d], dims=d,
                                   content_um=float(np.prod(hi - lo)),
                                   centroid_um=cen, n_support=int(pts_all.shape[0]),
                                   flags=POLY_OK)
    else:
        poly = fit_polytope(pts)
        if poly is None:
            return None, "degenerate"

    s = signed_distance(pts, poly)
    n_vox = float(mask.sum())
    measure = n_vox * float(np.prod(vox))
    extras = {
        "rms_residual_um": float(np.sqrt(np.mean(s ** 2))),
        "max_residual_um": float(np.max(np.abs(s))),
        "fill": float(measure / poly.content_um) if poly.content_um > 0 else float("nan"),
        "centre_vox": idx.mean(axis=0),
        "n_voxels": n_vox,
    }
    return poly, extras
