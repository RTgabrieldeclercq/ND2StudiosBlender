"""Synthetic DVC volume generator + exact analytic deformations — ALDVC Appendix D.

WHY THIS EXISTS
    `aldvc_field.run_aldvc` is a port of FranckLab's MATLAB ALDVC. To know the port
    is *numerically* right (not merely "runs green"), it has to be measured against
    a field whose answer is known to machine precision. This module builds exactly
    the benchmark the paper does:

      Appendix D — isolated spherical beads seeded by Poisson-disc sampling at
      0.006 beads/voxel, minimum separation = one bead diameter, each rendered as
      a 3D Gaussian PSF  ``A·exp(-Σ xᵢ²/2σ²)`` with ``σ = 1`` (≈5-voxel diameter),
      quantized to 8-bit.

    The deformed volume is built by **re-placing every bead at its analytically
    deformed centre**, not by resampling the reference. That matters: the paper's
    own warp-based protocol injects an interpolation bias of O(10⁻³) voxels
    (Bornert 2017), which is the sinusoidal ripple visible in the paper's Fig. 2(a)
    and is the *generator's* error, not the solver's. Analytic re-placement removes
    it, so the RMS error this module reports is the algorithm's alone. Pass
    ``warp=True`` to reproduce the paper's warp protocol (bias included) instead.

DEFORMATIONS (paper "Performance Assessment via Homogeneous Deformations")
    translation  — uniform shift along x, 0…1 voxel
    stretch      — uniaxial x-stretch λ, zero Poisson ratio
    rotation     — rigid in-plane rotation about the z-axis

    All are expressed as a displacement ``u(X) = y(X) − X`` about a chosen centre,
    so the exact field is available at any point (the DVC grid nodes included).

Axis order is numpy/kernel order throughout: ``(z, y, x)``, slowest first. The
"x-direction" of the paper is therefore axis **2** and displacement component 2.

Manual (not part of nodegraph.selftest — a measurement tool):
    python scripts/_aldvc_synth.py --demo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  Bead field
# ──────────────────────────────────────────────────────────────────────

def poisson_disc_3d(
    shape: tuple[int, int, int],
    density: float = 0.006,
    min_sep: float = 5.0,
    seed: int = 0,
    margin: float = 3.0,
) -> np.ndarray:
    """``(n, 3)`` bead centres in ``(z, y, x)`` voxels, ≥ ``min_sep`` apart.

    Bridson-style dart throwing on a uniform grid hash: O(n) rather than the
    O(n²) all-pairs check, which matters because the paper's density puts ~300 k
    beads in a 512×512×192 volume. ``density`` is beads per voxel (paper: 0.006);
    the returned count is whatever the separation constraint admits, which is
    slightly below the target once the volume saturates.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    target = int(round(density * float(np.prod(shape))))
    lo = np.asarray([margin] * 3, dtype=np.float64)
    hi = np.asarray(shape, dtype=np.float64) - 1.0 - margin
    if np.any(hi <= lo):
        raise ValueError(f"volume {shape} too small for margin {margin}")

    cell = float(min_sep) / np.sqrt(3.0)          # ≤1 point per cell
    occupied: dict[tuple[int, int, int], int] = {}
    pts = np.empty((target, 3), dtype=np.float64)
    n = 0
    # 30 darts per accepted point is enough to fill to the packing limit; beyond
    # that the rejection rate says the volume is saturated and we stop.
    max_darts = 30 * target
    sep2 = float(min_sep) ** 2
    for _ in range(max_darts):
        if n >= target:
            break
        p = lo + rng.random(3) * (hi - lo)
        c = tuple(((p - lo) / cell).astype(int))
        clash = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = occupied.get((c[0] + dz, c[1] + dy, c[2] + dx))
                    if q is not None and float(np.sum((pts[q] - p) ** 2)) < sep2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        occupied[c] = n
        pts[n] = p
        n += 1
    return pts[:n]


def render_beads(
    shape: tuple[int, int, int],
    centres: np.ndarray,
    sigma: float = 1.0,
    amplitude: float = 255.0,
    support: float = 4.0,
    dtype=np.uint8,
) -> np.ndarray:
    """Rasterize Gaussian-PSF beads — ALDVC Appendix D Eq. (33).

    ``PSF(x) = A·exp(-Σ xᵢ²/2σ²)``. Each bead is splatted only over the
    ``±support·σ`` box around its centre (the tail below that is < 3·10⁻⁴·A, i.e.
    sub-LSB for 8-bit), which is what makes a 300 k-bead volume tractable.
    Overlapping beads add, then the volume is clipped and quantized.
    """
    shape = tuple(int(s) for s in shape)
    vol = np.zeros(shape, dtype=np.float32)
    r = int(np.ceil(support * sigma))
    off = np.arange(-r, r + 1, dtype=np.float64)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for c in np.asarray(centres, dtype=np.float64):
        base = np.rint(c).astype(np.int64)
        lo = base - r
        hi = base + r + 1
        clo = np.maximum(lo, 0)
        chi = np.minimum(hi, shape)
        if np.any(chi <= clo):
            continue
        # Separable Gaussian on the clipped box, evaluated at the *exact*
        # sub-voxel centre — this is what makes the bead position (and hence the
        # imposed displacement) exact rather than rounded to the voxel lattice.
        gs = []
        for ax in range(3):
            a = off[(clo[ax] - lo[ax]):(2 * r + 1 - (hi[ax] - chi[ax]))]
            d = (base[ax] + a) - c[ax]
            gs.append(np.exp(-(d * d) * inv2s2))
        blob = amplitude * gs[0][:, None, None] * gs[1][None, :, None] * gs[2][None, None, :]
        vol[clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]] += blob.astype(np.float32)
    cap = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else amplitude
    np.clip(vol, 0.0, cap, out=vol)
    return vol.astype(dtype)


# ──────────────────────────────────────────────────────────────────────
#  Analytic deformations   u(X) = y(X) − X,  X and u in (z, y, x)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Deformation:
    """An exact deformation: maps reference points to deformed points."""
    name: str
    label: str
    fn: Callable[[np.ndarray], np.ndarray]          # X (n,3) → y (n,3)
    exact_strain: np.ndarray | None = None          # (3,3) infinitesimal, if uniform

    def displacement(self, X: np.ndarray) -> np.ndarray:
        """``u = y(X) − X`` at points ``X`` ``(n, 3)`` in ``(z, y, x)`` voxels."""
        X = np.asarray(X, dtype=np.float64)
        return self.fn(X) - X


def translation(dx: float) -> Deformation:
    """Uniform shift of ``dx`` voxels along x (paper case i)."""
    def fn(X):
        y = X.copy()
        y[:, 2] += dx
        return y
    return Deformation("translation", f"ux={dx:g} vox", fn,
                       exact_strain=np.zeros((3, 3)))


def uniaxial_stretch(lam: float, centre: Sequence[float]) -> Deformation:
    """Uniaxial x-stretch ratio ``lam``, zero Poisson ratio (paper case ii).

    ``y_x = c_x + λ(X_x − c_x)`` about ``centre`` (``(z, y, x)``); y and z fixed.
    Infinitesimal strain is exactly ``e_xx = λ − 1``, all other components 0.
    """
    c = np.asarray(centre, dtype=np.float64)
    e = np.zeros((3, 3)); e[2, 2] = lam - 1.0

    def fn(X):
        y = X.copy()
        y[:, 2] = c[2] + lam * (X[:, 2] - c[2])
        return y
    return Deformation("stretch", f"lambda={lam:g}", fn, exact_strain=e)


def rotation_z(deg: float, centre: Sequence[float]) -> Deformation:
    """Rigid in-plane rotation of ``deg`` degrees about the z-axis (paper case iii).

    Rotation acts in the (y, x) plane about ``centre``; z is unchanged. A rigid
    rotation has exactly **zero** Green-Lagrange strain, but its *infinitesimal*
    strain is ``(cosθ − 1)·I₂`` — O(θ²), not 0. The paper reports that residual as
    the method's strain error, so we record the exact value rather than 0.
    """
    c = np.asarray(centre, dtype=np.float64)
    t = np.deg2rad(deg)
    ct, st = np.cos(t), np.sin(t)
    e = np.zeros((3, 3)); e[1, 1] = e[2, 2] = ct - 1.0

    def fn(X):
        y = X.copy()
        dy = X[:, 1] - c[1]
        dx = X[:, 2] - c[2]
        y[:, 1] = c[1] + ct * dy - st * dx
        y[:, 2] = c[2] + st * dy + ct * dx
        return y
    return Deformation("rotation", f"theta={deg:g}deg", fn, exact_strain=e)


# ──────────────────────────────────────────────────────────────────────
#  Volume pair construction
# ──────────────────────────────────────────────────────────────────────

def make_pair(
    shape: tuple[int, int, int],
    deform: Deformation,
    *,
    density: float = 0.006,
    sigma: float = 1.0,
    seed: int = 0,
    warp: bool = False,
    dtype=np.uint8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ref, defm, centres)`` for one deformation.

    ``warp=False`` (default, and the *better* benchmark) re-renders the beads at
    their analytically deformed centres — no interpolation anywhere, so the only
    error measured is the solver's. ``warp=True`` reproduces the paper's protocol
    (tri-cubic resample of the reference), which additionally carries the
    generator's O(10⁻³)-voxel interpolation bias.
    """
    centres = poisson_disc_3d(shape, density=density, seed=seed,
                              min_sep=max(3.0, 5.0 * sigma))
    ref = render_beads(shape, centres, sigma=sigma, dtype=dtype)
    if warp:
        from scipy.ndimage import map_coordinates
        # g(p) = f(y⁻¹(p)): sample the reference at the *inverse*-mapped output
        # grid. Every deformation here is affine, so the inverse is exact.
        grids = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape],
                            indexing="ij")
        P = np.stack([g.ravel() for g in grids], axis=1)
        Xsrc = _invert_affine(deform, P)
        defm = map_coordinates(ref.astype(np.float64), Xsrc.T.reshape(3, *shape),
                               order=3, mode="nearest")
        cap = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 255.0
        defm = np.clip(defm, 0, cap).astype(dtype)
    else:
        defm = render_beads(shape, deform.fn(centres), sigma=sigma, dtype=dtype)
    return ref, defm, centres


def _invert_affine(deform: Deformation, P: np.ndarray) -> np.ndarray:
    """Invert an affine ``y(X)`` at points ``P`` by recovering ``(A, b)`` from four
    probe points (exact for every deformation in this module)."""
    probes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Y = deform.fn(probes)
    b = Y[0]
    A = (Y[1:] - b).T                     # columns = images of the basis vectors
    return np.linalg.solve(A, (P - b).T).T


# ──────────────────────────────────────────────────────────────────────
#  Error metrics — paper Eq. (13)
# ──────────────────────────────────────────────────────────────────────

def rms_error(numeric: np.ndarray, exact: np.ndarray) -> float:
    """``sqrt( Σ|numeric − exact|² / N )`` over finite entries — paper Eq. (13)."""
    a = np.asarray(numeric, dtype=np.float64).ravel()
    b = np.asarray(exact, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def interior_mask(grid_coords: np.ndarray, shape: Sequence[int],
                  subset_size: int, pad: float = 0.0) -> np.ndarray:
    """``(*grid,)`` bool — nodes whose subset window lies fully inside the volume.

    The paper solves the whole VOI except "a small portion of the domain near the
    image borders … due to the loss of information"; comparing RMS on border nodes
    where the reference itself is undefined inflates every method's error, so every
    benchmark here masks them out identically.
    """
    half = max(1, int(subset_size) // 2) + float(pad)
    c = np.asarray(grid_coords, dtype=np.float64)
    ok = np.ones(c.shape[:-1], dtype=bool)
    for ax, n in enumerate(shape):
        ok &= (c[..., ax] >= half) & (c[..., ax] <= (n - 1 - half))
    return ok


def exact_field_on_grid(deform: Deformation, grid_coords: np.ndarray) -> np.ndarray:
    """``(*grid, 3)`` exact displacement at the DVC subset centres."""
    c = np.asarray(grid_coords, dtype=np.float64)
    flat = c.reshape(-1, c.shape[-1])
    u = deform.displacement(flat)
    return u.reshape(c.shape)


def _demo() -> int:
    shape = (64, 64, 64)
    for d in (translation(0.5), uniaxial_stretch(1.05, [32, 32, 32]),
              rotation_z(5.0, [32, 32, 32])):
        ref, dfm, ctr = make_pair(shape, d, seed=1)
        u = d.displacement(ctr)
        print(f"{d.name:<12} {d.label:<16} beads={len(ctr):>6} "
              f"ref[{ref.min()},{ref.max()}] "
              f"max|u|={np.abs(u).max():.3f} vox  "
              f"mean|def-ref|={np.abs(dfm.astype(float) - ref.astype(float)).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
