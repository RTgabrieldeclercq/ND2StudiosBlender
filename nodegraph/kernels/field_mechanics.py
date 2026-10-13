"""field_mechanics — kinematics and constitutive laws on a sampled displacement or velocity field.

The Field nodes (``field.gradient`` / ``field.strain`` / ``field.rotation`` /
``field.volume_change`` / ``field.stress``) are thin wrappers around this module. It takes
plain numpy arrays and returns plain numpy arrays: one row per field node, every tensor laid
out ``(N, D, D)`` with the axes **slowest first** — ``(y, x)`` in 2-D, ``(z, y, x)`` in 3-D —
the order every other field in this repo uses (``field_math``'s LOAD-BEARING CONVENTIONS).

Where the maths already exists it is reused, not copied:

* the four strain measures are :func:`nodegraph.kernels.field_math.strain_from_gradient`,
  so ``dvc.strain``, ``analysis.track_field`` and ``field.strain`` cannot disagree about
  what "Green-Lagrange" means;
* the scattered-point gradient is :func:`nodegraph.kernels.track_field.
  mls_displacement_gradient` (SerialTrack's strain gauge, pinned to it by selftest);
* Lamé parameters, Hooke's law under plane strain / plane stress, principal values and the
  stress invariants are ``track_field``'s.

What is new here: regular-lattice detection and a NaN-aware finite-difference gradient (the
correlation grids of DIC, PIV and DVC are lattices with holes), the rotation / curl and
volume-change quantities, and three more constitutive laws (Saint Venant-Kirchhoff, a
compressible neo-Hookean solid, a Newtonian fluid).

SIGN CONVENTIONS — stated once, used everywhere
    * ``G[p, i, j] = d u_i / d X_j`` — the MATERIAL gradient, taken in the reference
      configuration, so ``F = I + G`` is the deformation gradient.
    * Image axes: ``x`` to the right, ``y`` DOWN the rows, ``z`` along the plane index.
    * 2-D ``curl`` and ``rotation_deg`` are **positive counter-clockwise as seen on
      screen**: ``curl = du_x/dy - du_y/dx`` in image axes, which is ``2 x`` the small
      rotation angle (and the vorticity, for a velocity field).
    * 3-D curl components are in the frame (x right, y down, z = increasing plane index),
      which is right-handed when z runs AWAY from the viewer; ``curl_view = -curl_z`` is
      the on-screen in-plane part with the 2-D sign.
    * Stress is Cauchy (true) stress in the unit of the modulus; ``pressure = -tr(s)/3``
      (compression positive).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from nodegraph.kernels.field_math import strain_from_gradient
from nodegraph.kernels.track_field import (
    _principal,
    lame_parameters,
    linear_elastic_stress,
    mls_displacement_gradient,
    stress_invariants,
)

#: the strain measures ``strain`` accepts (``field_math`` spellings, underscored)
MEASURES: Tuple[str, ...] = ("infinitesimal", "green_lagrange", "euler_almansi", "hencky")
#: the constitutive laws ``stress`` accepts
MODELS: Tuple[str, ...] = ("linear_elastic", "saint_venant_kirchhoff", "neo_hookean",
                           "newtonian_fluid")
#: a lattice must fill at least this fraction of its bounding grid — otherwise scattered
#: points that merely share integer coordinates (detections on whole pixels) would be read
#: as a mostly-empty lattice and differentiated across gaps that are not holes
MIN_FILL = 0.5


# ═══════════════════════════════════════════════════════════════
#  Lattice detection
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Lattice:
    """A regular grid the rows sit on: ``shape`` nodes per axis, ``step`` in the
    coordinates' own unit per axis (``nan`` for a one-node axis), and each row's node
    ``index`` as a tuple of integer arrays (one per axis) for fancy indexing."""
    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    index: Tuple[np.ndarray, ...]


def lattice_of(coords: np.ndarray, *, tol: float = 1e-3) -> Optional[Lattice]:
    """The regular lattice ``coords`` ``(N, D)`` sit on, or ``None``.

    Per axis: the distinct values, the smallest gap as the step, and every value within
    ``tol`` steps of a whole number of steps from the first. Missing nodes are fine — DIC
    and PIV drop the vectors they could not measure, and those become holes — but two rows
    on one node, a fill below :data:`MIN_FILL`, or a value off the lattice is not a lattice.
    """
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 2 or not np.isfinite(c).all():
        return None
    n, d = c.shape
    shape, step, index = [], [], []
    for j in range(d):
        v = c[:, j]
        u = np.unique(np.round(v, 6))
        if u.size == 1:
            shape.append(1)
            step.append(float("nan"))
            index.append(np.zeros(n, dtype=np.intp))
            continue
        h = float(np.diff(u).min())
        if not h > 0:
            return None
        k = np.round((v - u[0]) / h)
        if np.abs((v - u[0]) - k * h).max() > tol * h:
            return None
        shape.append(int(k.max()) + 1)
        step.append(h)
        index.append(k.astype(np.intp))
    size = int(np.prod(shape))
    if size <= 0 or n / size < MIN_FILL:
        return None
    flat = np.ravel_multi_index(tuple(index), tuple(shape))
    if np.unique(flat).size != n:
        return None
    return Lattice(tuple(shape), tuple(step), tuple(index))


def node_spacing(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance of ``coords`` ``(N, D)`` — the field's own
    resolution, in the coordinates' unit; ``nan`` for fewer than two finite rows."""
    c = np.asarray(coords, dtype=np.float64)
    c = c[np.isfinite(c).all(axis=1)] if c.ndim == 2 else c
    if c.ndim != 2 or c.shape[0] < 2:
        return float("nan")
    from scipy.spatial import cKDTree      # lazy: the core stays scipy-free
    dd, _ = cKDTree(c).query(c, k=2)
    dd = np.asarray(dd)[:, 1]
    dd = dd[np.isfinite(dd) & (dd > 0)]
    return float(np.median(dd)) if dd.size else float("nan")


# ═══════════════════════════════════════════════════════════════
#  The gradient
# ═══════════════════════════════════════════════════════════════

def _smooth_holes(g: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized-convolution Gaussian smoothing of one component on the lattice: holes
    (NaN) neither contribute nor get filled, so a hole does not drag its neighbours toward
    zero and a smoothed field reports nothing where nothing was measured."""
    from scipy.ndimage import gaussian_filter   # lazy
    w = np.isfinite(g).astype(np.float64)
    num = gaussian_filter(np.where(w > 0, g, 0.0), sigma, mode="nearest")
    den = gaussian_filter(w, sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(w > 0, out, np.nan)


def _diff_axis(g: np.ndarray, axis: int, h: float) -> np.ndarray:
    """``dg/dx_axis`` on the lattice: central where both neighbours exist, one-sided where
    only one does (edges and hole borders), NaN where neither does or ``g`` itself is NaN.
    One-sided differences are first-order, but the alternative — NaN on every node next to
    a hole — would erase a ring of measurements around each unmeasured subset."""
    if g.shape[axis] < 2 or not np.isfinite(h):
        return np.full(g.shape, np.nan)
    pad = [(0, 0)] * g.ndim
    pad[axis] = (1, 1)
    p = np.pad(g, pad, mode="constant", constant_values=np.nan)
    sl = [slice(None)] * g.ndim

    def take(a, b):
        s = list(sl)
        s[axis] = slice(a, b)
        return p[tuple(s)]

    n = g.shape[axis]
    prev, here, nxt = take(0, n), take(1, n + 1), take(2, n + 2)
    cen = (nxt - prev) / (2.0 * h)
    fwd = (nxt - here) / h
    bwd = (here - prev) / h
    out = np.where(np.isfinite(cen), cen, np.where(np.isfinite(fwd), fwd, bwd))
    return np.where(np.isfinite(here), out, np.nan)


def lattice_gradient(vec: np.ndarray, lat: Lattice, step_phys: np.ndarray,
                     *, smooth: float = 0.0) -> np.ndarray:
    """``G`` ``(N, D, D)`` of ``vec`` ``(N, D)`` by finite differences on ``lat``.

    ``step_phys`` is the physical node step per axis (µm); ``smooth`` an optional Gaussian
    pre-smoothing in NODE units (0 = off), applied hole-aware (:func:`_smooth_holes`)."""
    v = np.asarray(vec, dtype=np.float64)
    n, d = v.shape
    G = np.full((n, d, d), np.nan)
    for i in range(d):
        grid = np.full(lat.shape, np.nan)
        grid[lat.index] = v[:, i]
        if smooth and smooth > 0:
            grid = _smooth_holes(grid, float(smooth))
        for j in range(d):
            G[:, i, j] = _diff_axis(grid, j, float(step_phys[j]))[lat.index]
    return G


@dataclass
class GradientResult:
    G: np.ndarray                 # (N, D, D), NaN where undefined
    valid: np.ndarray             # (N,) bool
    method: str                   # "lattice" | "local_fit" — what actually ran
    radius_um: float              # the local-fit radius used (nan on the lattice path)
    spacing_um: float             # the field's node spacing


def shell_distance(coords: np.ndarray) -> Tuple[float, int]:
    """``(r_k, k)``: the median distance to each node's ``k = 3^D - 1``-th nearest neighbour
    — the full first shell of a lattice (8 in 2-D, 26 in 3-D), z neighbours included.

    This, not the nearest-neighbour spacing, is what a local fit's radius has to reach: on
    an anisotropic stack (0.5 µm pixels, 2 µm planes) every nearest neighbour is in-plane,
    and a radius sized from it holds no node above or below — every 3-D fit is degenerate."""
    c = np.asarray(coords, dtype=np.float64)
    c = c[np.isfinite(c).all(axis=1)]
    d = c.shape[1] if c.ndim == 2 else 0
    k = min(3 ** d - 1, c.shape[0] - 1) if d else 0
    if k < 1:
        return float("nan"), 0
    from scipy.spatial import cKDTree      # lazy
    dd, _ = cKDTree(c).query(c, k=k + 1)
    dk = np.asarray(dd)[:, -1]
    dk = dk[np.isfinite(dk) & (dk > 0)]
    return (float(np.median(dk)) if dk.size else float("nan")), k


def auto_neighbors(radius: float, shell: float, k_shell: int, d: int) -> int:
    """The KNN ceiling for a ``radius`` neighbourhood: the first shell holds ``k_shell``
    nodes within ``shell``, so a radius holds about ``k_shell (radius / shell)^D``; 50 %
    headroom, at least ``D + 2``, at most 400 (the batched fit's memory is N x K)."""
    if not (np.isfinite(radius) and np.isfinite(shell) and shell > 0 and k_shell > 0):
        return 4 * (d + 1)
    est = k_shell * (float(radius) / shell) ** d
    return int(np.clip(np.ceil(1.5 * est), d + 2, 400))


def gradient(coords: np.ndarray, scale: np.ndarray, vec: np.ndarray, *,
             method: str = "auto", radius_um: float = 0.0, neighbors: int = 0,
             smooth: float = 0.0) -> GradientResult:
    """The gradient of ``vec`` ``(N, D)`` over the nodes ``coords`` ``(N, D)``.

    ``coords`` are in index units (px, plane index) and ``scale`` is µm per index unit per
    axis, so lattice detection works on clean index values while the derivative comes out
    per µm. ``method``:

    * ``lattice`` — finite differences on the regular grid (refused when there is none);
    * ``local_fit`` — a weighted plane fit over ``radius_um`` (0 = auto: 1.5 x the distance
      to the first full neighbour shell, :func:`shell_distance` — about 2 node spacings on
      an isotropic lattice), for any layout: refined DIC meshes, scattered tracks;
    * ``auto`` — lattice when the rows form one, otherwise local fit.
    """
    c = np.asarray(coords, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    sc = np.asarray(scale, dtype=np.float64).reshape(-1)
    n, d = c.shape
    phys = c * sc
    lat = lattice_of(c) if method in ("auto", "lattice") else None
    if method == "lattice" and lat is None:
        raise ValueError(
            "the field's nodes do not form a regular lattice (a refined DIC mesh, tracked "
            "particles, or a grid with more than half its nodes missing), so there are no "
            "neighbours to difference between — choose `local fit`, which fits a plane "
            "through the nodes within a radius and works on any layout")
    if lat is not None:
        steps = np.array([h * s if np.isfinite(h) else np.nan
                          for h, s in zip(lat.step, sc)])
        spacing = float(np.nanmin(steps)) if np.isfinite(steps).any() else float("nan")
        G = lattice_gradient(v, lat, steps, smooth=smooth)
        valid = np.isfinite(G).all(axis=(1, 2))
        return GradientResult(G, valid, "lattice", float("nan"), spacing)
    spacing = node_spacing(phys)
    shell, k_shell = shell_distance(phys)
    r = float(radius_um) if radius_um and radius_um > 0 else 1.5 * shell
    if not np.isfinite(r) or r <= 0:
        G = np.full((n, d, d), np.nan)
        return GradientResult(G, np.zeros(n, bool), "local_fit", r, spacing)
    k = (int(neighbors) if neighbors and neighbors > 0
         else auto_neighbors(r, shell, k_shell, d))
    _u, G, valid = mls_displacement_gradient(phys, v, radius=r, n_neighbors=k)
    return GradientResult(G, valid, "local_fit", r, spacing)


# ═══════════════════════════════════════════════════════════════
#  Quantities derived from G
# ═══════════════════════════════════════════════════════════════

def _finite_rows(G: np.ndarray) -> np.ndarray:
    return np.isfinite(G).all(axis=tuple(range(1, G.ndim)))


def deformation(G: np.ndarray) -> np.ndarray:
    """``F = I + G`` ``(N, D, D)``."""
    d = G.shape[-1]
    return G + np.eye(d)[None]


def strain(G: np.ndarray, measure: str = "infinitesimal") -> np.ndarray:
    """The symmetric strain tensor ``(N, D, D)`` of ``measure`` (tensor shear). Rows with
    a non-finite gradient stay NaN (a NaN reaching ``eigh`` would fail the whole batch)."""
    m = str(measure).lower().replace("-", "_")
    if m not in MEASURES:
        raise ValueError(f"unknown strain measure {measure!r}; one of {', '.join(MEASURES)}")
    out = np.full(G.shape, np.nan)
    ok = _finite_rows(G)
    if ok.any():
        out[ok] = np.moveaxis(
            strain_from_gradient(np.moveaxis(G[ok], 0, -1), m.replace("_", "-")), -1, 0)
    return out


def symmetric_part(G: np.ndarray) -> np.ndarray:
    return 0.5 * (G + np.swapaxes(G, 1, 2))


def principal(T: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Principal values of a symmetric ``(N, D, D)`` tensor, **largest first** ``(N, D)``,
    and in 2-D the angle of the largest one's axis in degrees, counter-clockwise on screen
    from +x, folded into (-90, 90]; ``None`` in 3-D."""
    a = np.asarray(T, dtype=np.float64)
    n, d = a.shape[0], a.shape[-1]
    vals = _principal(np.moveaxis(a, 0, -1))[:, ::-1]
    if d != 2:
        return vals, None
    ang = np.full(n, np.nan)
    ok = np.isfinite(a).all(axis=(1, 2))
    if ok.any():
        sym = 0.5 * (a[ok] + np.swapaxes(a[ok], 1, 2))
        _w, V = np.linalg.eigh(sym)
        vy, vx = V[:, 0, -1], V[:, 1, -1]          # eigenvector of the largest value
        th = np.degrees(np.arctan2(-vy, vx))       # screen-up is -y
        th = np.where(th > 90.0, th - 180.0, th)
        ang[ok] = np.where(th <= -90.0, th + 180.0, th)
    return vals, ang


def equivalent_strain(T: np.ndarray) -> np.ndarray:
    """Von Mises equivalent strain ``sqrt(2/3 e':e')`` of ``(N, D, D)``.

    In 2-D the out-of-plane normal strain is not measured; it is taken from incompressibility
    (``e_zz = -(e_yy + e_xx)``), the convention commercial DIC packages report, which makes
    the formula ``(2/sqrt 3) sqrt(e1^2 + e1 e2 + e2^2)`` over the in-plane principal values."""
    a = np.asarray(T, dtype=np.float64)
    d = a.shape[-1]
    sym = 0.5 * (a + np.swapaxes(a, 1, 2))
    if d == 2:
        e11, e22, e12 = sym[:, 0, 0], sym[:, 1, 1], sym[:, 0, 1]
        # e1^2 + e1 e2 + e2^2 in invariant form: tr^2 - det over the in-plane tensor
        tr = e11 + e22
        det = e11 * e22 - e12 * e12
        return (2.0 / np.sqrt(3.0)) * np.sqrt(np.clip(tr * tr - det, 0.0, None))
    tr = np.trace(sym, axis1=1, axis2=2)
    dev = sym - (tr / 3.0)[:, None, None] * np.eye(3)[None]
    return np.sqrt(np.clip(2.0 / 3.0 * np.einsum("nij,nij->n", dev, dev), 0.0, None))


def max_shear(T: np.ndarray) -> np.ndarray:
    """``(e_max - e_min) / 2`` over the principal values (the largest Mohr circle)."""
    w = _principal(np.moveaxis(np.asarray(T, dtype=np.float64), 0, -1))
    return 0.5 * (w[:, -1] - w[:, 0])


def curl(G: np.ndarray) -> Dict[str, np.ndarray]:
    """The curl of the field (vorticity for a velocity) — see the module's sign conventions.

    2-D → ``{"curl"}``; 3-D → ``{"curl_z", "curl_y", "curl_x", "curl_mag", "curl_view"}``."""
    d = G.shape[-1]
    if d == 2:                                     # axes (y, x): index 0 = y, 1 = x
        return {"curl": G[:, 1, 0] - G[:, 0, 1]}
    # axes (z, y, x): index 0 = z, 1 = y, 2 = x; right-handed (x, y, z)
    wx = G[:, 0, 1] - G[:, 1, 0]                   # du_z/dy - du_y/dz
    wy = G[:, 2, 0] - G[:, 0, 2]                   # du_x/dz - du_z/dx
    wz = G[:, 1, 2] - G[:, 2, 1]                   # du_y/dx - du_x/dy
    return {"curl_z": wz, "curl_y": wy, "curl_x": wx,
            "curl_mag": np.sqrt(wx * wx + wy * wy + wz * wz), "curl_view": -wz}


def rotation_deg(G: np.ndarray) -> np.ndarray:
    """The finite rotation of the polar decomposition ``F = R U``, in degrees.

    2-D: signed, counter-clockwise on screen positive, exact
    (``atan2(F_xy - F_yx, F_xx + F_yy)`` in image axes). 3-D: the unsigned rotation angle
    ``arccos((tr R - 1) / 2)``, ``R`` from a batched SVD with the reflection removed."""
    n, d = G.shape[0], G.shape[-1]
    out = np.full(n, np.nan)
    ok = _finite_rows(G)
    if not ok.any():
        return out
    F = deformation(G[ok])
    if d == 2:
        out[ok] = np.degrees(np.arctan2(F[:, 1, 0] - F[:, 0, 1], F[:, 0, 0] + F[:, 1, 1]))
        return out
    W, _s, Vt = np.linalg.svd(F)
    R = W @ Vt
    flip = np.linalg.det(R) < 0
    if flip.any():
        W[flip, :, -1] *= -1.0
        R[flip] = W[flip] @ Vt[flip]
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    out[ok] = np.degrees(np.arccos(c))
    return out


def volume_change(G: np.ndarray) -> Dict[str, np.ndarray]:
    """``jacobian`` = det F (area ratio in 2-D, volume ratio in 3-D), ``dilatation`` =
    J - 1, ``divergence`` = tr G (the small-strain volumetric strain; for a velocity the
    rate of volume change, 1/s)."""
    n = G.shape[0]
    J = np.full(n, np.nan)
    ok = _finite_rows(G)
    if ok.any():
        J[ok] = np.linalg.det(deformation(G[ok]))
    return {"jacobian": J, "dilatation": J - 1.0,
            "divergence": np.trace(G, axis1=1, axis2=2)}


# ═══════════════════════════════════════════════════════════════
#  Constitutive laws → Cauchy stress (N, 3, 3) in (z, y, x)
# ═══════════════════════════════════════════════════════════════

def _embed3(F2: np.ndarray, fzz: np.ndarray) -> np.ndarray:
    """A 2-D ``(N, 2, 2)`` tensor in (y, x) into ``(N, 3, 3)`` in (z, y, x) with ``fzz``
    in the z slot and no coupling to z."""
    n = F2.shape[0]
    F = np.zeros((n, 3, 3))
    F[:, 1:, 1:] = F2
    F[:, 0, 0] = fzz
    return F


def _neo_hookean_fzz(F2: np.ndarray, lam: float, mu: float, iters: int = 60) -> np.ndarray:
    """The out-of-plane stretch ``s`` that makes a compressible neo-Hookean plane-stress
    state traction-free: ``mu (s^2 - 1) + lam ln(J2 s) = 0``. The left side increases
    monotonically in ``s > 0`` (derivative ``2 mu s + lam / s``), so the root is unique;
    Newton from ``s = 1`` with step-halving whenever it would leave ``s > 0``."""
    J2 = np.linalg.det(F2)
    s = np.ones(F2.shape[0])
    ok = J2 > 0
    s[~ok] = np.nan
    for _ in range(iters):
        f = mu * (s * s - 1.0) + lam * np.log(J2 * s, where=ok, out=np.zeros_like(s))
        fp = 2.0 * mu * s + lam / s
        step = f / fp
        new = s - step
        bad = ok & (new <= 0)
        new[bad] = 0.5 * s[bad]
        new = np.where(ok, new, np.nan)
        moved = np.abs(new - s) / np.maximum(np.abs(s), 1e-300)
        s = new
        if not np.any(moved[ok] > 1e-14):
            break
    return s


def stress(G: np.ndarray, model: str, *, youngs_modulus: float = 1.0,
           poisson_ratio: float = 0.3, plane: str = "strain",
           viscosity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """``(sigma (N, 3, 3), energy (N,))`` — Cauchy stress in (z, y, x) and the strain-energy
    density (NaN for the fluid), from the gradient ``G`` of a displacement (the elastic laws)
    or of a velocity (``newtonian_fluid``).

    * ``linear_elastic`` — Hooke on the small strain ``sym(G)`` (``track_field``'s law).
    * ``saint_venant_kirchhoff`` — ``S = lam tr(E) I + 2 mu E`` on Green-Lagrange ``E``,
      pushed forward ``sigma = F S F^T / J``; plane stress in closed form
      (``E_zz = -lam / (lam + 2 mu) tr(E_in)``).
    * ``neo_hookean`` — compressible, ``sigma = mu/J (B - I) + lam ln J / J I``; plane stress
      by :func:`_neo_hookean_fzz`.
    * ``newtonian_fluid`` — the DEVIATORIC viscous stress ``2 eta dev(D)`` of the strain
      rate ``D = sym(L)``; a fluid's pressure is not a function of its velocity gradient, so
      only the deviator is reported. A 2-D flow is taken as planar (``D_zz = 0``).

    2-D ``plane``: ``strain`` (``F_zz = 1`` — a thick block) or ``stress`` (``sigma_zz = 0``
    — a thin sheet). A 3-D field measures everything and ignores it."""
    g = np.asarray(G, dtype=np.float64)
    n, d = g.shape[0], g.shape[-1]
    sig = np.full((n, 3, 3), np.nan)
    energy = np.full(n, np.nan)
    ok = _finite_rows(g)
    mdl = str(model)
    if mdl not in MODELS:
        raise ValueError(f"unknown constitutive model {model!r}; one of {', '.join(MODELS)}")
    pl = str(plane)
    if d == 2 and pl not in ("strain", "stress"):
        raise ValueError(f"plane must be 'strain' or 'stress', got {plane!r}")
    if mdl == "newtonian_fluid":
        eta = float(viscosity)
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError(f"the viscosity must be a positive, finite value in Pa·s "
                             f"(got {viscosity!r})")
        if ok.any():
            D = symmetric_part(g[ok])
            D3 = D if d == 3 else _embed3(D, np.zeros(D.shape[0]))
            tr = np.trace(D3, axis1=1, axis2=2)
            sig[ok] = 2.0 * eta * (D3 - (tr / 3.0)[:, None, None] * np.eye(3)[None])
        return sig, energy
    lam, mu = lame_parameters(youngs_modulus, poisson_ratio)
    if not ok.any():
        return sig, energy
    gk = g[ok]
    if mdl == "linear_elastic":
        eps = symmetric_part(gk)
        s = np.moveaxis(linear_elastic_stress(np.moveaxis(eps, 0, -1),
                                              youngs_modulus=youngs_modulus,
                                              poisson_ratio=poisson_ratio,
                                              plane=pl), -1, 0)
        sig[ok] = s
        # 2-D: sigma_zz * eps_zz = 0 under both plane assumptions, so the in-plane sum is W
        sin = s if d == 3 else s[:, 1:, 1:]
        energy[ok] = 0.5 * np.einsum("nij,nij->n", sin, eps)
        return sig, energy
    F2 = deformation(gk)
    if d == 3:
        F = F2
    elif mdl == "saint_venant_kirchhoff":
        if pl == "strain":
            F = _embed3(F2, np.ones(F2.shape[0]))
        else:
            Ein = 0.5 * (np.swapaxes(F2, 1, 2) @ F2 - np.eye(2)[None])
            ezz = -lam / (lam + 2.0 * mu) * np.trace(Ein, axis1=1, axis2=2)
            arg = 1.0 + 2.0 * ezz
            fzz = np.sqrt(np.where(arg > 0, arg, np.nan))
            F = _embed3(F2, fzz)
    else:
        F = _embed3(F2, np.ones(F2.shape[0]) if pl == "strain"
                    else _neo_hookean_fzz(F2, lam, mu))
    J = np.linalg.det(F)
    good = np.isfinite(J) & (J > 0)
    eye = np.eye(3)[None]
    out = np.full((F.shape[0], 3, 3), np.nan)
    en = np.full(F.shape[0], np.nan)
    if good.any():
        Fg, Jg = F[good], J[good]
        if mdl == "saint_venant_kirchhoff":
            E = 0.5 * (np.swapaxes(Fg, 1, 2) @ Fg - eye)
            trE = np.trace(E, axis1=1, axis2=2)
            S = lam * trE[:, None, None] * eye + 2.0 * mu * E
            out[good] = Fg @ S @ np.swapaxes(Fg, 1, 2) / Jg[:, None, None]
            en[good] = 0.5 * lam * trE ** 2 + mu * np.einsum("nij,nij->n", E, E)
        else:                                          # neo_hookean
            B = Fg @ np.swapaxes(Fg, 1, 2)
            lnJ = np.log(Jg)
            out[good] = (mu / Jg)[:, None, None] * (B - eye) \
                + (lam * lnJ / Jg)[:, None, None] * eye
            I1 = np.trace(B, axis1=1, axis2=2)
            en[good] = 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2
    sig[ok] = out
    energy[ok] = en
    return sig, energy


def stress_summary(sig: np.ndarray) -> Dict[str, np.ndarray]:
    """``stress_p1..p3`` (largest first), ``pressure`` (-tr/3, compression positive),
    ``von_mises`` and ``max_shear`` (Tresca) of a ``(N, 3, 3)`` Cauchy stress."""
    s = np.moveaxis(np.asarray(sig, dtype=np.float64), 0, -1)
    press, vm, tau = stress_invariants(s)
    w = _principal(s)[:, ::-1]
    return {"stress_p1": w[:, 0], "stress_p2": w[:, 1], "stress_p3": w[:, 2],
            "pressure": press, "von_mises": vm, "max_shear": tau}


__all__ = [
    "MEASURES", "MODELS", "MIN_FILL", "Lattice", "lattice_of", "node_spacing",
    "shell_distance", "lattice_gradient", "GradientResult", "auto_neighbors", "gradient",
    "deformation",
    "strain", "symmetric_part", "principal", "equivalent_strain", "max_shear", "curl",
    "rotation_deg", "volume_change", "stress", "stress_summary",
]
