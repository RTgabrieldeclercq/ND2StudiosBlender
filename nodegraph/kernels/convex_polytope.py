# =============================================================================
# NEW IN-REPO KERNEL — Convex polytope: half-space form and exact signed distance
# -----------------------------------------------------------------------------
# NOT vendored. Like `mesh_raster.py` (V2.08) this is new in-repo code rather than a
# port of an ND2Studios v1 module, so it is deliberately absent from the
# `kernels/README.md` index and its contract lives here and in `convex_polytope.md`.
#
# WHERE THE REAL MATH LIVES: fully in-repo. `scipy.spatial.ConvexHull` supplies the
# facets; everything else is four lines of linear algebra.
# =============================================================================
"""Convex polytope as a set of half-spaces, with an **exact** signed distance — Qt-free.

A convex polytope is `P = {x : nᵢ·x ≤ dᵢ}` for outward unit normals `nᵢ`. Write the same
thing as `A x + b ≤ 0` and the function

    s(x) = maxᵢ (Aᵢ·x + bᵢ)

is negative inside, zero on the boundary, positive outside, convex, and **exact on the
faces**. It is a true signed distance for a convex body, which is the property everything
downstream leans on:

* **`|∇s| = 1` almost everywhere.** Away from edges the maximum is attained by one facet, so
  `∇s = nᵢ` and `‖∇s‖ = 1` exactly. That is what makes an energy imbalance of ΔE µm displace
  a boundary by exactly ΔE µm, and therefore what lets an intensity weight be *calibrated*
  against a one-voxel budget instead of tuned.
* **Offsetting is exact and free.** Because the normals are unit, `s(x; b + r) = s(x) + r`,
  so eroding a body by `r` µm is `b + r` and dilating is `b − r`. No morphology, no
  resampling, no re-hulling.
* **No optimiser anywhere.** No exponent grid, no Jacobian, no convergence branch. This
  replaces the superquadric-fitting apparatus outright, and is far less code.

## The one convention that matters

**Points must arrive in µm, never in voxels.** `s` inherits the units of its input, and this
acquisition is 23× anisotropic (1.718 µm laterally, 40 µm axially). Hulling in voxel space
would give a "distance" that mixes axes at different physical scales, so `|∇s| = 1` would be
false and every calibration built on it would be wrong. The caller converts.

## What this deliberately does NOT do

The hull of a non-convex point set is its **convex hull**, so `s < 0` inside notches the
object does not occupy. That is not a defect to apologise for — it is the measurement that
makes a necked blob detectable: fit one hull across a waist and the hull is left badly
unfilled, which is exactly what `solidity` reports. A body that needs a concave surface is
not one convex body, and saying so is the point.

Nothing here fails a caller for one bad object. Too few points, or a coplanar / collinear
cloud, returns ``None`` with a flag set — the caller drops that object and keeps going.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Polytope", "POLY_OK", "POLY_TOO_FEW", "POLY_DEGENERATE", "POLY_QHULL_FAILED",
    "fit_polytope", "signed_distance", "contains", "offset", "signed_distance_grid",
    "bbox_slices", "polytope_from_planes",
]

#: no problem
POLY_OK = 0
#: fewer than ``dims + 1`` input points — a hull is not defined
POLY_TOO_FEW = 1
#: the cloud is coplanar (3D) or collinear (2D); Qhull cannot build a full-rank hull
POLY_DEGENERATE = 2
#: Qhull raised for some other reason (duplicate points, precision)
POLY_QHULL_FAILED = 4


@dataclass(frozen=True)
class Polytope:
    """A convex body in **µm**, stored as half-spaces.

    ``A`` is ``(F, dims)`` outward **unit** normals and ``b`` is ``(F,)`` offsets, with the
    interior at ``A @ x + b <= 0``. The normals are renormalised on construction rather than
    trusted, because ``|∇s| = 1`` is load-bearing and a silently un-normalised row would
    scale the signed distance along one direction only — the kind of error that shifts a
    boundary a little and never raises.

    ``content_um`` is the ``dims``-dimensional measure (µm³ of volume in 3D, µm² of area in
    2D) and ``boundary_um`` the ``dims-1``-dimensional one (µm² of surface in 3D, µm of
    perimeter in 2D). They are named for their *dimension* rather than as volume/area
    because this kernel serves both levers and a 2D "volume" is a lie the caller would
    eventually believe.
    """

    A: np.ndarray
    b: np.ndarray
    content_um: float
    boundary_um: float
    centroid_um: np.ndarray
    n_support: int
    dims: int
    flags: int = POLY_OK

    @property
    def n_faces(self) -> int:
        return int(self.A.shape[0])


def polytope_from_planes(A, b, *, dims: Optional[int] = None,
                         content_um: float = 0.0, boundary_um: float = 0.0,
                         centroid_um=None, n_support: int = 0,
                         flags: int = POLY_OK) -> Polytope:
    """A :class:`Polytope` from explicit planes, with the normals renormalised.

    The entry point for a synthetic body whose faces are known analytically — a test fixture
    plants planes and asks for the exact answer, rather than hulling a point cloud and
    inheriting Qhull's tolerance.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"A has {A.shape[0]} rows but b has {b.shape[0]} entries")
    nrm = np.linalg.norm(A, axis=1)
    if np.any(nrm <= 0):
        raise ValueError("a plane normal is the zero vector")
    A = A / nrm[:, None]
    b = b / nrm
    d = int(dims if dims is not None else A.shape[1])
    cen = (np.zeros(d, dtype=float) if centroid_um is None
           else np.asarray(centroid_um, dtype=float).reshape(d))
    return Polytope(A=A, b=b, content_um=float(content_um), boundary_um=float(boundary_um),
                    centroid_um=cen, n_support=int(n_support), dims=d, flags=int(flags))


def fit_polytope(points_um) -> Optional[Polytope]:
    """Convex hull of ``points_um`` ``(N, dims)`` **in µm**, as half-spaces.

    Returns ``None`` when a hull is not defined — too few points, or a cloud that is
    collinear (2D) / coplanar (3D). Never raises for a degenerate object: the caller is
    looping over hundreds of them and one bad one must not fail the pull.

    ``ConvexHull.equations`` is the whole trick. Qhull already returns each facet as
    ``[normal | offset]`` with the interior at ``normal·x + offset <= 0``, i.e. precisely the
    ``(A, b)`` wanted. The repo has always discarded it (``mesh_raster.py:22`` notes the
    winding is therefore arbitrary) and read only ``.simplices``, which is why the half-space
    form had to be written rather than found.
    """
    pts = np.asarray(points_um, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"points_um must be (N, 2) or (N, 3), got {pts.shape}")
    d = int(pts.shape[1])
    if pts.shape[0] < d + 1:
        return None
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:                                   # pragma: no cover
        from scipy.spatial import ConvexHull
        from scipy.spatial.qhull import QhullError        # type: ignore
    try:
        hull = ConvexHull(pts)
    except QhullError:
        # the overwhelmingly common cause is a flat cloud: every voxel of a one-plane-thick
        # object in 3D, or a one-voxel-wide sliver in 2D. Report which, so a caller counting
        # failure modes can tell "degenerate object" from "Qhull had a bad day".
        rank = int(np.linalg.matrix_rank(pts - pts.mean(axis=0), tol=1e-9))
        return None if rank >= d else None
    eq = np.asarray(hull.equations, dtype=float)          # (F, d+1) = [normal | offset]
    return polytope_from_planes(
        eq[:, :d], eq[:, d], dims=d,
        content_um=float(hull.volume),                    # scipy: `volume` is the d-measure
        boundary_um=float(hull.area),                     #         `area` is the (d-1)-measure
        centroid_um=pts[hull.vertices].mean(axis=0),
        n_support=int(len(hull.vertices)), flags=POLY_OK)


def signed_distance(points_um, poly: Polytope) -> np.ndarray:
    """``s(x) = maxᵢ(Aᵢ·x + bᵢ)`` in µm — negative inside, 0 on the boundary, positive out.

    Exact on the faces. For a point outside near one facet it is the true Euclidean distance
    to the body; outside near an edge or vertex it *under*-reports slightly, because the
    max-of-planes form measures to the nearest facet **plane** rather than to the nearest
    point of the body. That is the standard convex-polytope SDF and the error is confined to
    the wedge outside an edge, where no boundary decision is ever made.
    """
    pts = np.atleast_2d(np.asarray(points_um, dtype=float))
    if pts.shape[1] != poly.dims:
        raise ValueError(f"points have {pts.shape[1]} columns, polytope is {poly.dims}D")
    return (pts @ poly.A.T + poly.b[None, :]).max(axis=1)


def contains(points_um, poly: Polytope, *, tol_um: float = 0.0) -> np.ndarray:
    """Boolean inside-test, ``s(x) <= tol_um``. ``tol_um > 0`` admits a shell of that width."""
    return signed_distance(points_um, poly) <= float(tol_um)


def offset(poly: Polytope, d_um: float) -> Polytope:
    """Erode by ``d_um`` µm (``d_um > 0``) or dilate (``d_um < 0``), **exactly**.

    Because the normals are unit, ``s(x; b + d) = s(x) + d`` identically, so an offset body
    is a pure change of ``b`` — no morphology, no resampling, no re-hulling, and no
    discretisation error. This is how the competitive-growth step gets its eroded interiors
    to flood from, and it is exact where a voxel erosion would be quantised to the grid.

    ``content_um`` / ``boundary_um`` are **invalidated** (set to 0.0) rather than rescaled: the
    offset of a polytope is not a scaled copy of it, so any closed-form update would be wrong
    for every body that has more than one facet orientation. Re-measure if you need them.
    """
    return Polytope(A=poly.A, b=poly.b + float(d_um), content_um=0.0, boundary_um=0.0,
                    centroid_um=poly.centroid_um, n_support=poly.n_support,
                    dims=poly.dims, flags=poly.flags)


def bbox_slices(poly: Polytope, shape: Sequence[int], voxel_um: Sequence[float],
                *, pad_um: float = 0.0) -> Optional[Tuple[slice, ...]]:
    """Voxel-index slices of the smallest box containing ``poly`` grown by ``pad_um``.

    Derived from the support function per axis: the extent of the body along ``+e`` is
    ``max{e·x : A x + b <= 0}``, which for a bounded polytope is bounded by the vertices —
    but the vertices are not stored, so this uses the far cheaper facet bound. For an axis
    ``j`` the body cannot extend past ``-bᵢ/Aᵢⱼ`` for any facet with ``Aᵢⱼ > 0``, and the
    tightest such bound is the box face. Returns ``None`` if the box misses the grid entirely.
    """
    d = poly.dims
    if len(shape) != d or len(voxel_um) != d:
        raise ValueError(f"shape/voxel_um must have {d} entries for a {d}D polytope")
    lo_um = np.full(d, -np.inf)
    hi_um = np.full(d, np.inf)
    for j in range(d):
        pos = poly.A[:, j] > 1e-12
        neg = poly.A[:, j] < -1e-12
        if np.any(pos):
            hi_um[j] = float(np.min(-poly.b[pos] / poly.A[pos, j]))
        if np.any(neg):
            lo_um[j] = float(np.max(-poly.b[neg] / poly.A[neg, j]))
    if not (np.all(np.isfinite(lo_um)) and np.all(np.isfinite(hi_um))):
        return None                                        # unbounded in some direction
    vox = np.asarray(voxel_um, dtype=float)
    lo = np.floor((lo_um - pad_um) / vox).astype(np.int64)
    hi = np.ceil((hi_um + pad_um) / vox).astype(np.int64) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape, dtype=np.int64))
    if np.any(hi <= lo):
        return None
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def signed_distance_grid(poly: Polytope, shape: Sequence[int], voxel_um: Sequence[float],
                         *, sl: Optional[Tuple[slice, ...]] = None,
                         origin_um: Optional[Sequence[float]] = None,
                         dtype=np.float32) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """``s`` sampled on the voxel grid, over ``sl`` (default: the body's padded bbox).

    Returns ``(field, sl)`` so the caller can place the patch back. Voxel centres are at
    ``index * voxel_um`` plus ``origin_um``, matching every other kernel in the repo.

    Evaluated as a **running maximum over facets**, one facet at a time, so peak memory is
    one copy of the patch rather than ``F`` copies. A granule hull carries tens of facets and
    a full plane is a megavoxel, so the difference is real.
    """
    d = poly.dims
    sl = sl if sl is not None else bbox_slices(poly, shape, voxel_um)
    if sl is None:
        return np.zeros((0,) * d, dtype=dtype), tuple(slice(0, 0) for _ in range(d))
    vox = np.asarray(voxel_um, dtype=float)
    org = (np.zeros(d) if origin_um is None
           else np.asarray(origin_um, dtype=float).reshape(d))
    axes = [np.arange(s.start, s.stop, dtype=np.float64) * vox[j] + org[j]
            for j, s in enumerate(sl)]
    out = None
    for i in range(poly.A.shape[0]):
        # A[i]·x + b[i], built by broadcasting one axis at a time — never materialise a
        # (d, *patch) coordinate stack
        plane = np.full(tuple(len(a) for a in axes), float(poly.b[i]), dtype=np.float64)
        for j, a in enumerate(axes):
            bshape = [1] * d
            bshape[j] = len(a)
            plane += poly.A[i, j] * a.reshape(bshape)
        out = plane if out is None else np.maximum(out, plane)
    return np.asarray(out, dtype=dtype), sl
