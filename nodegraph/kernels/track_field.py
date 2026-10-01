"""track_field — per-particle kinematics from a TRACKED point field.

The post-processing half of SerialTrack, lifted out of the tracker.

:mod:`nodegraph.kernels.track_objects` ends at the correspondence problem: it decides
which detection in frame *t* is which detection in frame *t-1*. Everything a
particle-tracking experiment is actually *for* — the displacement field, its gradient,
the strain, and (with a constitutive law) the stress — is downstream of that answer and
needs nothing from the tracker but the track ids. That is why this lives in its own
module: the maths here works on ANY tracked field, whether the ids came from
SerialTrack, from ``track.link``'s nearest-neighbour linker, or from a hand-built
correspondence.

Upstream parentage, function by function:

``mls_displacement_gradient``
    SerialTrack's own scattered strain gauge — ``funCompDefGrad3.m``, ported once in
    :func:`track_objects.compute_strain_mls` and re-derived here in vectorised form.
    Fits ``u(x) = u0 + G·(x - x0)`` by weighted least squares over each particle's
    neighbourhood. The loop version does one ``np.linalg.lstsq`` per particle; this one
    batches the normal equations, measured at **6x** on 8000 points (0.11 s vs 0.7 s) —
    worth having on a 50-frame bead field, but the real reasons for the rewrite are the
    two things the loop does not do: it refuses a degenerate neighbourhood instead of
    returning ``lstsq``'s least-norm answer for it, and it excludes a particle whose own
    displacement is unmeasured instead of interpolating one from its neighbours.
    :func:`nodegraph.selftest` pins the two together on a fixture, so the port cannot
    drift from the code it was derived from.

``linear_elastic_stress`` / ``stress_invariants``
    NOT from SerialTrack. SerialTrack computes displacement and strain and stops there —
    stress needs a material model, which is a claim about the specimen rather than about
    the images. The isotropic linear-elastic law here is the standard one
    (``sigma = 2*mu*eps + lambda*tr(eps)*I``) and is exposed as an explicit, named choice
    with its own moduli, so nobody reads a stress map without having stated what they
    believe the gel is made of.

``strain_invariants``
    Plain tensor algebra, here so the strain and stress halves report their scalars
    through one code path.

CONVENTIONS — get these wrong and every number is meaningless.

*Axis order is slowest-first*, ``(z, y, x)`` in 3-D and ``(y, x)`` in 2-D — the Dataset's
own order, so a caller never permutes. A 2-D stress tensor is still returned as a full
``3x3`` in ``(z, y, x)`` with the out-of-plane component in slot 0, because both plane
assumptions produce one and von Mises needs it.

*Everything is physical on the way in.* ``coords`` and ``disp`` are in the SAME length
unit (the node passes micrometres), so the gradient comes out dimensionless with no
anisotropy correction to forget. This deliberately differs from
:func:`track_objects.compute_strain_mls`, which takes voxel coordinates and rescales
afterwards through ``pixel_steps``; that path exists for the tracker's own internal grid
and is the one place the two functions are allowed to disagree.

*Gradient, not deformation gradient.* ``G[i, j] = d u_i / d x_j``. The deformation
gradient is ``F = I + G``. :func:`nodegraph.kernels.field_math.strain_from_gradient`
takes ``G`` and adds the identity itself, which is why the four strain measures are
shared with ``analysis.dvc_field`` rather than re-derived.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "mls_displacement_gradient",
    "linear_elastic_stress",
    "stress_invariants",
    "strain_invariants",
    "lame_parameters",
]


# ═══════════════════════════════════════════════════════════════
#  Moving-least-squares strain gauge (vectorised funCompDefGrad3)
# ═══════════════════════════════════════════════════════════════

def mls_displacement_gradient(
    coords: np.ndarray,
    disp: np.ndarray,
    *,
    radius: float,
    n_neighbors: int,
    rcond: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving-least-squares displacement gradient at every particle.

    Fits ``u(x) = u0 + G @ (x - x0)`` around each particle over its ``n_neighbors``
    nearest neighbours, keeping only those within ``radius``, and returns the fitted
    displacement and the gradient.

    Parameters
    ----------
    coords : (N, D) float
        Particle positions, ``(z, y, x)`` or ``(y, x)``, in a physical length unit.
    disp : (N, D) float
        Displacement at each particle, in the SAME unit as ``coords``. A non-finite row
        is an UNMEASURED particle: it is dropped from every neighbourhood it appears in,
        and gets no gradient of its own (see ``valid``).
    radius : float
        Neighbourhood radius in that unit. Neighbours beyond it get zero weight; the
        particle itself is always in its own fit, so a lone particle is *invalid* rather
        than silently fitted to nothing.
    n_neighbors : int
        How many nearest neighbours to consider before the radius cut. Clamped to
        ``N - 1``.
    rcond : float
        Relative singular-value floor. A neighbourhood that is degenerate — all
        neighbours collinear in 2-D, coplanar in 3-D — has no unique gradient, and the
        least-norm answer ``pinv`` would return looks entirely plausible while being
        arbitrary in the unconstrained direction. Those particles are marked invalid
        instead.

    Returns
    -------
    u_fit : (N, D) — the smoothed displacement, NaN where the fit failed.
    G : (N, D, D) — ``G[p, i, j] = d u_i / d x_j``, NaN where the fit failed.
    valid : (N,) bool — which particles got a well-conditioned fit from their own
        measured displacement plus at least ``D`` measured neighbours.

    Notes
    -----
    The weighted normal equations are formed directly rather than through ``lstsq``
    because the design matrix is at most ``(K+1, 4)``: forming ``A^T W A`` costs nothing
    and lets all N systems be solved in one batched call. Conditioning is checked
    explicitly (see ``rcond``) rather than left to ``lstsq``'s internal cutoff, which
    would silently truncate and report success.
    """
    coords = np.asarray(coords, dtype=np.float64)
    disp = np.asarray(disp, dtype=np.float64)
    if coords.ndim != 2 or disp.shape != coords.shape:
        raise ValueError(
            f"mls_displacement_gradient: coords {coords.shape} and disp {disp.shape} "
            f"must both be (N, D) and agree")
    n, d = coords.shape
    u_fit = np.full((n, d), np.nan)
    G = np.full((n, d, d), np.nan)
    valid = np.zeros(n, dtype=bool)
    if n < d + 1:                      # fewer points than the fit has unknowns
        return u_fit, G, valid

    from scipy.spatial import cKDTree     # lazy: the core stays scipy-free

    k = int(min(max(1, int(n_neighbors)), n - 1))
    dd, ii = cKDTree(coords).query(coords, k=k + 1)
    dd = np.atleast_2d(np.asarray(dd, dtype=np.float64))
    ii = np.atleast_2d(np.asarray(ii, dtype=np.intp))

    w = (dd <= float(radius)).astype(np.float64)
    w[:, 0] = 1.0                       # column 0 is the particle itself (distance 0)
    count = w.sum(axis=1)

    dx = coords[ii] - coords[:, None, :]                       # (N, K+1, D)
    A = np.concatenate([np.ones((n, k + 1, 1)), dx], axis=2)    # (N, K+1, D+1)
    b = disp[ii]                                                # (N, K+1, D)
    # A neighbour with a non-finite displacement must not poison the fit: zero its weight
    # rather than propagating NaN through the whole neighbourhood.
    measured = np.isfinite(disp).all(axis=1)
    w = np.where(measured[ii], w, 0.0)
    count = w.sum(axis=1)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    Aw = A * w[:, :, None]
    M = np.einsum("pki,pkj->pij", Aw, A)        # (N, D+1, D+1)
    rhs = np.einsum("pki,pkj->pij", Aw, b)      # (N, D+1, D)

    sv = np.linalg.svd(M, compute_uv=False)     # (N, D+1), descending
    top = np.maximum(sv[:, 0], np.finfo(float).tiny)
    # A particle whose OWN displacement is unmeasured gets no gradient, even where its
    # neighbours would support a perfectly well-conditioned fit. That fit is a legitimate
    # INTERPOLATION of the field at that location, and reporting it in the same column as
    # measured particles would be fabrication wearing a measurement's name — the node
    # feeds this exactly such rows (a track's first appearance has no displacement).
    ok = measured & (count >= d + 1) & (sv[:, -1] > float(rcond) * top)
    if ok.any():
        sol = np.linalg.solve(M[ok], rhs[ok])   # (Nok, D+1, D)
        u_fit[ok] = sol[:, 0, :]
        # sol[p, 1 + j, i] is the coefficient of dx_j in the fit of u_i, i.e. du_i/dx_j.
        G[ok] = np.swapaxes(sol[:, 1:, :], 1, 2)
        valid[ok] = True
    return u_fit, G, valid


# ═══════════════════════════════════════════════════════════════
#  Isotropic linear elasticity (NOT part of SerialTrack — see module docstring)
# ═══════════════════════════════════════════════════════════════

def lame_parameters(youngs_modulus: float, poisson_ratio: float) -> Tuple[float, float]:
    """``(lambda, mu)`` from ``(E, nu)``, refusing the two singular ends.

    ``nu = 0.5`` is perfectly incompressible and sends ``lambda`` to infinity — the
    displacement field then determines only the deviatoric stress and the pressure is a
    Lagrange multiplier this local law cannot supply. ``nu <= -1`` is thermodynamically
    impossible. Both are refused rather than returned as ``inf``, which would flow
    downstream as a map of infinities.
    """
    e = float(youngs_modulus)
    nu = float(poisson_ratio)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError(
            f"Young's modulus must be a positive, finite stiffness (got {e!r}). It sets "
            f"the units of every stress column, so there is no sensible default.")
    if not np.isfinite(nu) or nu <= -1.0 or nu >= 0.5:
        raise ValueError(
            f"Poisson's ratio must lie in (-1, 0.5) (got {nu!r}). At exactly 0.5 the "
            f"material is incompressible and the bulk modulus is infinite, so the "
            f"pressure is not determined by the strain at all — use 0.49 or 0.499 for a "
            f"nearly-incompressible gel, which is what the TFM literature does.")
    mu = e / (2.0 * (1.0 + nu))
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def linear_elastic_stress(
    strain: np.ndarray,
    *,
    youngs_modulus: float,
    poisson_ratio: float,
    plane: str = "strain",
) -> np.ndarray:
    """Cauchy stress from the isotropic linear-elastic law, as a full ``(3, 3, N)``.

    Parameters
    ----------
    strain : (D, D, N)
        The strain tensor per particle, ``D`` = 2 or 3, axes slowest-first.
    youngs_modulus, poisson_ratio :
        The material. The stress carries whatever pressure unit ``youngs_modulus`` is in
        (the node's socket says Pa).
    plane : ``"strain"`` | ``"stress"``
        Read only for a 2-D field, where the third dimension has to be assumed:

        * ``"strain"`` — ``eps_zz = 0``. The right assumption for an in-plane slice of a
          THICK specimen, which a TFM gel is: the surrounding material prevents
          out-of-plane relaxation, and ``sigma_zz = nu * (sigma_yy + sigma_xx)`` comes out
          non-zero.
        * ``"stress"`` — ``sigma_zz = 0``. The right assumption for a THIN free-standing
          film, where nothing resists out-of-plane thinning.

        A 3-D field measures ``eps_zz`` and needs neither assumption, so this is inert
        there.

    Returns
    -------
    sigma : (3, 3, N) — always the full tensor in ``(z, y, x)``, so that a 2-D field's
    out-of-plane component (which both assumptions produce, one of them as zero) is
    present for the invariants.
    """
    eps = np.asarray(strain, dtype=np.float64)
    if eps.ndim != 3 or eps.shape[0] != eps.shape[1] or eps.shape[0] not in (2, 3):
        raise ValueError(
            f"linear_elastic_stress: strain must be (D, D, N) with D in (2, 3), "
            f"got {eps.shape}")
    lam, mu = lame_parameters(youngs_modulus, poisson_ratio)
    d, n = eps.shape[0], eps.shape[2]
    eye = np.eye(d)[:, :, None]

    if d == 3:
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        return 2.0 * mu * eps + lam * tr * eye

    mode = str(plane).lower()
    if mode not in ("strain", "stress"):
        raise ValueError(
            f"linear_elastic_stress: plane must be 'strain' or 'stress', got {plane!r}")
    tr2 = eps[0, 0] + eps[1, 1]
    # Plane stress replaces lambda by the reduced 2lam*mu/(lam+2mu) = E*nu/(1-nu^2), which
    # is what enforcing sigma_zz = 0 and eliminating eps_zz leaves behind.
    lam_eff = lam if mode == "strain" else 2.0 * lam * mu / (lam + 2.0 * mu)
    in_plane = 2.0 * mu * eps + lam_eff * tr2 * eye

    sigma = np.zeros((3, 3, n), dtype=np.float64)
    sigma[1:, 1:] = in_plane                         # (y, x) occupy slots 1 and 2
    sigma[0, 0] = lam * tr2 if mode == "strain" else 0.0
    return sigma


# ═══════════════════════════════════════════════════════════════
#  Tensor invariants — the scalars a map is actually drawn from
# ═══════════════════════════════════════════════════════════════

def _principal(t: np.ndarray) -> np.ndarray:
    """Ascending eigenvalues of a symmetric ``(D, D, N)`` field, NaN-safe.

    ``eigvalsh`` does not tolerate a non-finite entry — it raises for the whole batch —
    so the non-finite particles are excluded and filled with NaN. The tensor is
    symmetrised first: every strain measure here is symmetric by construction, and
    forcing it makes that an assertion rather than an assumption."""
    a = np.asarray(t, dtype=np.float64)
    d, n = a.shape[0], a.shape[2]
    out = np.full((n, d), np.nan)
    sym = 0.5 * (a + a.transpose(1, 0, 2))
    m = np.moveaxis(sym, (0, 1), (-2, -1))           # (N, D, D)
    good = np.isfinite(m).all(axis=(1, 2))
    if good.any():
        out[good] = np.linalg.eigvalsh(m[good])
    return out


def strain_invariants(strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(volumetric, max_shear)`` from a ``(D, D, N)`` strain field.

    ``volumetric`` is the trace — the local dilatation, positive in expansion. For a
    small strain it is the fractional volume change directly; for the finite measures it
    is the first invariant of that measure and no longer exactly that, which is the usual
    price of asking one scalar to summarise a tensor.

    ``max_shear`` is ``(eps_1 - eps_min) / 2`` over the principal strains — the radius of
    the largest Mohr circle, and the scalar that actually shows where a field is shearing
    rather than translating.
    """
    a = np.asarray(strain, dtype=np.float64)
    tr = np.trace(a, axis1=0, axis2=1)
    w = _principal(a)
    return tr, 0.5 * (w[:, -1] - w[:, 0])


def stress_invariants(sigma: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(pressure, von_mises, max_shear)`` from a ``(3, 3, N)`` stress field.

    ``pressure`` is ``-tr(sigma) / 3``, signed so that COMPRESSION is positive — the
    convention in every gel-mechanics paper, and the opposite of the raw trace.

    ``von_mises`` is ``sqrt(1.5 * s:s)`` over the deviator ``s`` — the single scalar that
    says how close a point is to yielding, and the one usually plotted as "the stress
    map".

    ``max_shear`` is ``(sigma_1 - sigma_3) / 2`` over the principal stresses (Tresca).
    """
    a = np.asarray(sigma, dtype=np.float64)
    if a.ndim != 3 or a.shape[:2] != (3, 3):
        raise ValueError(f"stress_invariants: expected (3, 3, N), got {a.shape}")
    tr = a[0, 0] + a[1, 1] + a[2, 2]
    dev = a - (tr / 3.0) * np.eye(3)[:, :, None]
    vm = np.sqrt(np.clip(1.5 * np.einsum("ij...,ij...->...", dev, dev), 0.0, None))
    w = _principal(a)
    return -tr / 3.0, vm, 0.5 * (w[:, -1] - w[:, 0])
