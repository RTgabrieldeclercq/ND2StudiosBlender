"""field_math -- the correlation-field maths shared by the correlation nodes.

WHAT THIS IS
    The dimension-agnostic half of what used to live inside ``aldvc_field``: the
    ``DVCResult`` output contract, the regular subset ``Grid``, harmonic NaN
    inpainting, the displacement-gradient/strain-measure stack, and Lagrangian
    accumulation of an increment series. None of it is AL-DVC-specific -- it is
    what every correlation node in this repo needs once a solver has produced a
    displacement field on a regular grid.

WHY IT IS ITS OWN MODULE (2026-09-25)
    ``aldvc_field`` was a 2998-line clean-room port of FranckLab's MATLAB ALDVC.
    That port was replaced wholesale by a thin adapter around the official
    upstream package (``al-dvc`` / pyALDVC, github.com/zachtong/pyALDVC), which
    owns the solver now. But two other nodes never used the solver at all --
    ``analysis.accumulate_field`` imports ``Grid`` / ``accumulate_incremental`` /
    ``compute_strain``, and ``analysis.track_field`` imports
    ``strain_from_gradient`` -- so this maths outlives the port it shipped in.
    Every symbol below is byte-verbatim from the module it came out of; only the
    imports were consolidated into this header.

PROVENANCE
    ``DVCResult``                       nd2studios/core/dvc_registry.py
    ``Grid`` / ``build_grid``           nd2studios/backend/dvc/mesh.py
    ``inpaint_nans`` / ``_vector``      nd2studios/backend/dvc/outliers.py
    ``displacement_gradient`` ..
      ``compute_strain``                nd2studios/backend/dvc/strain.py
    ``accumulate_incremental`` /
      ``build_accumulated_results``     nd2studios/backend/dvc/tracking.py

LOAD-BEARING CONVENTIONS
    * Every field is **slowest-axis-first**: ``(z, y, x)`` in 3D, ``(y, x)`` in
      2D. ``grid_coords[..., 0]`` and ``displacement_field[..., 0]`` are the
      slowest axis, NOT x. A solver whose native order is ``[x, y, z]`` (pyALDVC,
      pyALDIC) must reverse before it builds a ``DVCResult``.
    * ``G[i, j] = d(u_i)/d(x_j)`` in that same axis order; ``strain_from_gradient``
      returns the SYMMETRIC strain tensor of the requested measure, so
      ``strain[i, j] == strain[j, i]`` and the shear terms are TENSOR shear
      (half of engineering shear).
    * ``displacement_field`` is in VOXELS. ``voxel_size_um`` records the scale but
      is not applied -- ``displacement_um()`` applies it, and the catalog's
      ``_shared.dvc._dvc_rows`` does the same multiply when it flattens to Points.
    * ``displacement_gradient``'s ``voxel_size`` argument applies the
      ``voxel_i / voxel_j`` cross-axis rescale that non-cubic voxels (a confocal
      z-step against an xy pixel) require. Skipping it makes every off-diagonal
      strain term wrong by the anisotropy ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter


# ==== vendored from nd2studios/core/dvc_registry.py ====

@dataclass
class DVCResult:
    """Structured output from a :meth:`DVCMethod.run` call.

    Shapes (``d`` = 2 for 2D DIC, 3 for 3D DVC):

    - ``grid_coords``: ``(*grid, d)`` subset-center coordinates **in voxels**,
      where ``grid`` is ``(Gy, Gx)`` (2D) or ``(Gz, Gy, Gx)`` (3D).
    - ``displacement_field``: ``(*grid, d)`` displacement **in voxels**, axis
      order matching ``grid_coords`` (i.e. ``[..., 0]`` is the slowest spatial
      axis: y in 2D, z in 3D).
    - ``strain_field``: ``(*grid, n_components)`` or ``None`` until computed.

    ``voxel_size_um`` is ``(y, x)`` (2D) or ``(z, y, x)`` (3D); use
    :meth:`displacement_um` to convert the displacement field to micrometers.
    """
    dim: int
    grid_coords: np.ndarray
    displacement_field: np.ndarray
    voxel_size_um: Tuple[float, ...] = ()

    strain_field: Optional[np.ndarray] = None
    strain_type: str = ""

    qfactor: Optional[np.ndarray] = None        # (*grid,) correlation confidence
    converged: bool = False
    iterations: int = 0
    mu: float = 0.0
    beta: float = 0.0

    method: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ── convenience accessors ──
    @property
    def magnitude(self) -> np.ndarray:
        """``(*grid,)`` displacement magnitude in voxels."""
        return np.sqrt(np.sum(np.square(self.displacement_field), axis=-1))

    def displacement_um(self) -> np.ndarray:
        """Displacement field converted to micrometers (per-axis scaling)."""
        if not self.voxel_size_um or len(self.voxel_size_um) != self.dim:
            return self.displacement_field
        scale = np.asarray(self.voxel_size_um, dtype=np.float64)
        return self.displacement_field * scale

    def magnitude_um(self) -> np.ndarray:
        return np.sqrt(np.sum(np.square(self.displacement_um()), axis=-1))


# ==== vendored from nd2studios/backend/dvc/mesh.py ====

@dataclass
class Grid:
    """A regular grid of subset centers over an image/volume.

    Attributes
    ----------
    axes : list of ``ndim`` 1-D arrays
        Per-axis center coordinates (in voxels), slowest axis first.
    coords : (*grid_shape, ndim) float64
        Center coordinate of every node, ``coords[..., c]`` the ``c``-th axis
        (``ij`` meshgrid → axis order matches ``grid_shape``).
    grid_shape : tuple[int, ...]
        Number of centers along each axis.
    step : (ndim,) float64
        Spacing between adjacent centers, per axis (voxels). Equals
        ``subset_spacing`` except on axes too small to hold >1 center.
    ndim : int
    """
    axes: List[np.ndarray]
    coords: np.ndarray
    grid_shape: Tuple[int, ...]
    step: np.ndarray
    ndim: int

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.grid_shape)) if self.grid_shape else 0

    def coords_flat(self) -> np.ndarray:
        """``(n_nodes, ndim)`` C-order flattened center coordinates."""
        return self.coords.reshape(-1, self.ndim)


def build_grid(shape: Tuple[int, ...], subset_size: int, subset_spacing: int,
               border_margin: int = 0) -> Grid:
    """Build a :class:`Grid` of subset centers for a volume of ``shape``.

    Centers are ``subset_spacing`` apart and inset by ``subset_size // 2 +
    border_margin`` so every subset window lies fully inside the volume — and,
    with ``border_margin > 0``, so the *deformed* search window has room too.

    ``border_margin`` exists because ALDVC insets its mesh by substantially more
    than half a subset: ``funIntegerSearch3Multigrid`` uses ``1 + round(winsize)``
    and ``funIntegerSearch3`` trims by ``winsize/2 + 3 (+ searchRadius)``. With a
    bare ``subset_size // 2`` inset, the outermost ring of centers sits exactly on
    the edge of the legal region, so its FFT search window gets clipped on one
    side and the seed there cannot represent displacement of the clipped sign
    (measured: 18–40 % of nodes on a stretch field). :func:`run_aldvc` therefore
    passes a margin derived from the search radius.

    The inset is never allowed to collapse below the point where the subset
    window still fits: an axis shorter than ``subset_size + 1`` gets a *reduced
    window* on that axis (see :func:`subset_shape_for`) rather than centers whose
    window would hang off the end, which used to yield a silently all-zero field
    on thin-Z stacks.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    half = max(1, int(subset_size) // 2)
    margin = max(0, int(border_margin))
    step = max(1, int(subset_spacing))
    axes: List[np.ndarray] = []
    steps: List[float] = []
    for n in shape:
        # Inset by half+margin where the axis allows it; shrink the margin first
        # (it is only a search-window courtesy), then the half-window (which
        # forces a reduced window on this axis via subset_shape_for).
        half_a = min(half, max(0, (n - 1) // 2))
        pad = min(margin, max(0, (n - 1) // 2 - half_a))
        start = half_a + pad
        stop = max(start + 1, n - half_a - pad)    # exclusive upper bound
        c = np.arange(start, stop, step, dtype=np.float64)
        if c.size == 0:
            c = np.asarray([n / 2.0], dtype=np.float64)
        elif c.size == 1 and (stop - 1) > start:
            # Force >=2 centers where the extent allows: the finite-difference
            # gradient operator + np.gradient (strain) are undefined on a size-1
            # axis, so a single grid node along an axis would break the global
            # solve. Two evenly-placed centers keep a thin (e.g. shallow-Z) grid
            # well-formed.
            c = np.round(np.linspace(start, stop - 1, 2)).astype(np.float64)
        axes.append(c)
        steps.append(float(np.median(np.diff(c))) if c.size > 1 else float(step))
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack(mesh, axis=-1).astype(np.float64)
    return Grid(
        axes=axes,
        coords=coords,
        grid_shape=tuple(a.size for a in axes),
        step=np.asarray(steps, dtype=np.float64),
        ndim=ndim,
    )


# ==== vendored from nd2studios/backend/dvc/outliers.py ====

def inpaint_nans(field: np.ndarray, *, iterations: int = 60) -> np.ndarray:
    """Fill NaNs in a scalar ``(*grid,)`` array by a discrete harmonic (Laplace) fill.

    ALDVC uses ``inpaint_nans3``, which solves the discrete Laplace equation over
    the NaN set — the fill is therefore **exact for any locally linear field**,
    which is precisely the regime a displacement field is in over one grid step.
    The previous nearest-finite-value (EDT) fill is exact only for a *constant*
    field: it produces piecewise-constant blocks whose internal gradient is zero
    and whose boundary gradient is a step, and those blocks feed straight into the
    finite-difference operator ``D`` and the strain gradient.

    Implemented as a nearest-value seed followed by Jacobi smoothing restricted to
    the NaN set (Dirichlet data = the finite nodes). That converges to the same
    harmonic solution as a sparse solve but costs a handful of ``uniform_filter``
    passes instead of factorizing a matrix per component per ADMM iteration; the
    NaN regions here are small and the iteration count is capped accordingly.
    """
    arr = np.asarray(field, dtype=np.float64)
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    if nan_mask.all():
        return np.zeros_like(arr)
    idx = distance_transform_edt(nan_mask, return_distances=False,
                                 return_indices=True)
    out = arr[tuple(idx)]                     # nearest-value seed
    n_iter = int(iterations)
    if n_iter > 0:
        # Jacobi sweeps toward the harmonic solution. `uniform_filter` averages a
        # 3^ndim box; restricting the write to nan_mask pins the known values, so
        # the iteration is a Dirichlet Laplace solve on the unknown set.
        for _ in range(n_iter):
            sm = uniform_filter(out, size=3, mode="nearest")
            out[nan_mask] = sm[nan_mask]
    return out


def inpaint_vector(u_grid: np.ndarray) -> np.ndarray:
    """Apply :func:`inpaint_nans` to each component of a ``(ndim, *grid)`` field."""
    out = np.array(u_grid, dtype=np.float64, copy=True)
    for c in range(out.shape[0]):
        out[c] = inpaint_nans(out[c])
    return out


# ==== vendored from nd2studios/backend/dvc/strain.py ====

def displacement_gradient(
    u_grid: np.ndarray, grid_step: np.ndarray,
    voxel_size: Optional[np.ndarray] = None, smooth_sigma: float = 0.0,
) -> np.ndarray:
    """``(ndim, *grid) → (ndim, ndim, *grid)`` displacement gradient ``∂u_i/∂x_j``.

    ``grid_step`` is the node spacing (voxels) per axis. If ``voxel_size`` is
    given, the result is in physical (dimensionless-strain) units with the
    ``voxel_i/voxel_j`` cross-axis rescaling applied.
    """
    ndim = u_grid.shape[0]
    step = np.asarray(grid_step, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)
    if smooth_sigma and smooth_sigma > 0:
        u = np.stack([gaussian_filter(u[i], sigma=float(smooth_sigma))
                      for i in range(ndim)])
    G = np.zeros((ndim, ndim, *u.shape[1:]), dtype=np.float64)
    gshape = u.shape[1:]
    for i in range(ndim):
        for j in range(ndim):
            # np.gradient needs >=2 samples along the axis; a singleton axis
            # (e.g. a 1-node-deep Z grid) contributes zero gradient there.
            if gshape[j] < 2:
                continue
            G[i, j] = np.gradient(u[i], float(step[j]), axis=j)
    if voxel_size is not None:
        v = np.asarray(voxel_size, dtype=np.float64)
        for i in range(ndim):
            for j in range(ndim):
                G[i, j] *= v[i] / v[j]
    return G


def strain_from_gradient(G: np.ndarray, strain_type: str = "infinitesimal") -> np.ndarray:
    """Strain tensor ``(ndim, ndim, *grid)`` from the displacement gradient ``G``.

    ``strain_type`` ∈ {infinitesimal, green-lagrange, almansi, hencky}.
    """
    ndim = G.shape[0]

    def _t(A):  # transpose the two tensor axes, keep grid axes
        return A.transpose(1, 0, *range(2, A.ndim))

    st = str(strain_type).lower().replace("_", "-")
    if st in ("infinitesimal", "small", "engineering"):
        return 0.5 * (G + _t(G))

    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (G.ndim - 2)))
    F = G + eye                                       # deformation gradient
    if st in ("green-lagrange", "green", "lagrange"):
        FtF = np.einsum("ki...,kj...->ij...", F, F)   # Fᵀ F
        return 0.5 * (FtF - eye)
    if st in ("almansi", "euler-almansi", "eulerian-almansi"):
        Finv = _inv_tensor_field(F)
        FiTFi = np.einsum("ki...,kj...->ij...", Finv, Finv)   # F⁻ᵀ F⁻¹
        return 0.5 * (eye - FiTFi)
    if st in ("hencky", "log", "logarithmic"):
        return _hencky(F)
    raise ValueError(f"unknown strain_type: {strain_type!r}")


def _inv_tensor_field(F: np.ndarray) -> np.ndarray:
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    inv = np.linalg.inv(moved)
    return np.moveaxis(inv, (-2, -1), (0, 1))


def _hencky(F: np.ndarray) -> np.ndarray:
    """Hencky (logarithmic) strain ``½ ln(FᵀF)`` via eigen-decomposition."""
    ndim = F.shape[0]
    C = np.einsum("ki...,kj...->ij...", F, F)         # right Cauchy-Green
    Cm = np.moveaxis(C, (0, 1), (-2, -1))
    w, V = np.linalg.eigh(Cm)
    w = np.clip(w, 1e-12, None)
    logw = 0.5 * np.log(w)
    E = np.einsum("...ik,...k,...jk->...ij", V, logw, V)
    return np.moveaxis(E, (-2, -1), (0, 1))


def compute_strain(
    u_grid: np.ndarray, grid_step: np.ndarray,
    voxel_size: Optional[np.ndarray] = None, *,
    strain_type: str = "infinitesimal", smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(F_def, strain)`` — deformation gradient ``I+∇u`` and the strain
    tensor of the requested measure, both ``(ndim, ndim, *grid)``."""
    G = displacement_gradient(u_grid, grid_step, voxel_size, smooth_sigma)
    ndim = G.shape[0]
    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (G.ndim - 2)))
    strain = strain_from_gradient(G, strain_type)
    return G + eye, strain


# ==== vendored from nd2studios/backend/dvc/tracking.py ====

def accumulate_incremental(
    grid: Grid, increments: List[Tuple[int, np.ndarray]],
) -> List[Tuple[int, np.ndarray]]:
    """Compose ordered per-step increment fields into cumulative displacements.

    Parameters
    ----------
    grid : the (shared) subset grid the increments live on.
    increments : ``[(t, u_grid), ...]`` in ascending frame order, each ``u_grid``
        an ``(ndim, *grid)`` incremental displacement (frame ``t-1`` → ``t``), in
        voxels.

    Returns ``[(t, u_accum_grid), ...]`` — the cumulative displacement from the
    reference frame at each ``t`` (``(ndim, *grid)``), by tracking the reference
    grid points through the increments.
    """
    ndim = grid.ndim
    axes = [np.asarray(a, dtype=np.float64) for a in grid.axes]
    coords0 = grid.coords_flat().astype(np.float64)     # (N, ndim), reference
    cur = coords0.copy()
    out: List[Tuple[int, np.ndarray]] = []
    for t, u_grid in increments:
        u_grid = np.asarray(u_grid, dtype=np.float64)
        disp = np.zeros_like(cur)
        for c in range(ndim):
            interp = RegularGridInterpolator(
                axes, u_grid[c], method="linear",
                bounds_error=False, fill_value=None)   # extrapolate at borders
            disp[:, c] = np.nan_to_num(interp(cur), nan=0.0,
                                       posinf=0.0, neginf=0.0)
        cur = cur + disp
        u_accum = (cur - coords0).reshape(*grid.grid_shape, ndim)
        out.append((int(t), np.moveaxis(u_accum, -1, 0)))   # (ndim, *grid)
    return out


def build_accumulated_results(
    grid: Grid, increment_results: List[Tuple[int, DVCResult]],
    voxel_size_um: Tuple[float, ...], *, strain_type: str = "infinitesimal",
    strain_smooth: float = 0.0,
) -> Dict[int, DVCResult]:
    """Turn ordered incremental :class:`DVCResult`s into cumulative ones.

    Accumulates the increments (:func:`accumulate_incremental`), then rebuilds a
    :class:`DVCResult` per frame with the cumulative displacement + strain
    recomputed from it — the field ALDVC's incremental mode actually reports.
    """
    ndim = grid.ndim
    incr = [(t, np.moveaxis(np.asarray(r.displacement_field), -1, 0))
            for t, r in increment_results]     # (ndim,*grid) each
    accum = accumulate_incremental(grid, incr)
    voxel = (np.asarray(voxel_size_um, dtype=np.float64)
             if voxel_size_um and len(voxel_size_um) == ndim else np.ones(ndim))
    out: Dict[int, DVCResult] = {}
    base_by_t = dict(increment_results)
    for t, u_grid in accum:
        base = base_by_t[t]
        _F, strain = compute_strain(u_grid, grid.step, voxel_size=voxel,
                                    strain_type=strain_type,
                                    smooth_sigma=strain_smooth)
        disp = np.moveaxis(u_grid, 0, -1)                    # (*grid, ndim)
        strain_field = np.moveaxis(strain, (0, 1), (-2, -1))
        diag = dict(base.diagnostics)
        diag["accumulated_from_incremental"] = True
        out[t] = DVCResult(
            dim=base.dim, grid_coords=base.grid_coords, displacement_field=disp,
            voxel_size_um=base.voxel_size_um, strain_field=strain_field,
            strain_type=strain_type, qfactor=base.qfactor,
            converged=base.converged, iterations=base.iterations,
            mu=base.mu, beta=base.beta, method="ALDVC (cumulative from incremental)",
            notes=base.notes + " · accumulated to cumulative", diagnostics=diag)
    return out
