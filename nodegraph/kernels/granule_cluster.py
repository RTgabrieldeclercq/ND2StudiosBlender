# ============================================================================
# VENDORED KERNEL — Granule clustering
# ============================================================================
# Purpose: assign each bead centroid in a 3-D point cloud to a granule cluster,
#   picking the model order (number of granules k) by BIC over a relaxed range
#   around a user-seeded count.
#
# WHERE THE REAL MATH LIVES: in-repo. The clustering (BIC sweep, GMM/KMeans fit,
#   X-means-style KMeans BIC, µm scaling, degeneracy fallbacks) is implemented
#   here in ND2Studios itself. The heavy lifting of the actual GMM/KMeans fits is
#   delegated to scikit-learn (sklearn.mixture.GaussianMixture / sklearn.cluster
#   .KMeans), lazily imported inside the fit helpers.
#
# PROVENANCE: vendored from
#   nd2studios/backend/analysis/granule_cluster.py  (branch: Version-1.45)
#   The single cross-module symbol it referenced — NOISE_LABEL from
#   nd2studios/backend/analysis/granule_types.py — is inlined below (= -1); it is
#   unused on the compute path.
#
# Vendored verbatim; imports nothing from nd2studios; caller owns all prep
#   (no file I/O, no per-multipoint/per-timepoint looping, no bead detection —
#   the caller supplies an already-extracted (N,3) voxel point cloud).
#
# DROPPED UI/registry-only members: NONE (source is already a pure math module).
# RENAMED for collision: NONE.
# ============================================================================
"""Granule clustering (V1.70 · P2) — pure, Qt-free.

Answers *"which granule does each bead belong to"* for a 3-D cloud of bead
centroids. The user seeds a granule count ``n_granules``; we relax it by ±``p``%
and let **BIC** pick the best model order over the resulting ``k`` range. The
default model is a full-covariance **Gaussian Mixture** (so an elongated /
anisotropic granule stays a single cluster instead of being split); a spherical
**KMeans** fallback is exposed for sparse / unstable clouds.

Optional dependency: ``scikit-learn`` is lazily imported and gated by
``importlib.util.find_spec`` (mirrors ``backend/serialtrack/prediction.py`` and
the cellpose/stardist precedent). It is deliberately *not* in
``requirements.txt``; a friendly :class:`ImportError` is raised when absent.

Coordinate / unit convention (P0 contract):

* ``points_zyx`` is ``(N, 3)`` in **voxel** units, ordered ``(z, y, x)``.
* ``voxel_size_um`` is ``(dz, dy, dx)`` µm. Coordinates are scaled to **µm**
  *before* fitting so anisotropic Z does not bias the Gaussians / distances.
* Returned ``means`` are in the fitted **µm** space.
* ``-1`` (:data:`NOISE_LABEL`) marks a rejected point. It is emitted **only** when
  ``params["noise_resp"] > 0``; with the default of ``0.0`` every point is
  assigned, which is this stage's historical behaviour. Consumers that iterate
  clusters must skip the negative label.

Beyond the labels, the winning fit also reports its **soft responsibilities** and
(for the full-covariance GMM) its **covariances** and the principal axes derived
from them — see :func:`cluster_granules` and :func:`_covariance_orientation`.
Those were previously computed by scikit-learn and then thrown away.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np

# Inlined from nd2studios.backend.analysis.granule_types (import removed for
# self-containment). Emitted only when ``noise_resp > 0`` — see
# :func:`cluster_granules`; with the default of 0.0 every point is assigned and
# this value never appears, which is the historical behaviour.
NOISE_LABEL: int = -1

# Fixed for reproducibility — the node exposes no per-run seed (P2 §Params).
_RANDOM_STATE: int = 0

# A uniformly-filled solid ellipsoid with semi-axis ``a`` has second moment
# ``lambda = a**2 / 5`` along that axis, so ``a = sqrt(5*lambda)``. Fitting a
# GAUSSIAN to points drawn from such a solid therefore under-reports the extent
# by ``sqrt(5) = 2.236x`` unless this factor is applied. See
# :func:`_covariance_orientation`.
_UNIFORM_SOLID_MOMENT = 5.0

_SKLEARN_MISSING_MSG = (
    "Granule clustering needs scikit-learn — `pip install scikit-learn`"
)

# reg_covar bump ladder for degenerate / ill-defined covariance retries.
_REG_COVAR_START = 1e-6
_REG_COVAR_FACTOR = 100.0
_REG_COVAR_TRIES = 5


def _require_sklearn() -> None:
    """Raise a friendly :class:`ImportError` if scikit-learn is unavailable."""
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError(_SKLEARN_MISSING_MSG)


class _Fit(NamedTuple):
    """One candidate model order, plus the estimator that produced it.

    The estimator is carried so the caller can ask the *winning* fit for its
    soft responsibilities and covariances without either refitting or paying for
    them at every ``k`` in the sweep.
    """

    bic: float
    labels: np.ndarray
    means: np.ndarray
    model: Any
    kind: str


def _fit_gmm(
    x_um: np.ndarray, k: int, n_init: int, random_state: int
) -> Optional[_Fit]:
    """Fit a full-covariance GMM with ``k`` components.

    Retries with a progressively larger ``reg_covar`` when the fit hits an
    ill-defined (singular / collapsed) covariance. Returns a :class:`_Fit` in the
    µm space, or ``None`` if every retry failed.
    """
    from sklearn.mixture import GaussianMixture

    reg = _REG_COVAR_START
    for _ in range(_REG_COVAR_TRIES):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                reg_covar=reg,
                random_state=random_state,
            )
            gm.fit(x_um)
            bic = float(gm.bic(x_um))
            if not np.isfinite(bic):
                reg *= _REG_COVAR_FACTOR
                continue
            labels = gm.predict(x_um).astype(int)
            return _Fit(bic, labels, np.asarray(gm.means_, dtype=float), gm, "gmm")
        except (ValueError, FloatingPointError):
            reg *= _REG_COVAR_FACTOR
    return None


def _kmeans_bic(
    x_um: np.ndarray, labels: np.ndarray, centers: np.ndarray, inertia: float
) -> float:
    """Lower-is-better BIC for a hard KMeans partition (X-means style).

    Models the partition as an equal-variance spherical Gaussian mixture with
    hard assignments, so the score is directly comparable to
    :meth:`GaussianMixture.bic` (also lower-is-better). Free parameters:
    ``k`` d-dim means + one shared variance + ``k-1`` mixing weights.
    """
    n, d = x_um.shape
    k = centers.shape[0]
    counts = np.bincount(labels, minlength=k).astype(float)

    denom = max(n - k, 1) * d
    variance = max(inertia / denom, 1e-9)

    # LL = Σ_k[ n_k log(n_k/N) - (n_k d / 2) log(2πσ²) ] - (N-k) d / 2
    nonzero = counts > 0
    log_lik = float(
        np.sum(counts[nonzero] * np.log(counts[nonzero] / n))
        - 0.5 * d * np.log(2.0 * np.pi * variance) * np.sum(counts)
        - 0.5 * max(n - k, 0) * d
    )
    n_params = k * d + 1 + (k - 1)
    return -2.0 * log_lik + n_params * np.log(max(n, 1))


def _fit_kmeans(
    x_um: np.ndarray, k: int, n_init: int, random_state: int
) -> Optional[_Fit]:
    """Fit spherical KMeans with ``k`` clusters; returns a :class:`_Fit`."""
    from sklearn.cluster import KMeans

    try:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(x_um).astype(int)
        centers = np.asarray(km.cluster_centers_, dtype=float)
        bic = _kmeans_bic(x_um, labels, centers, float(km.inertia_))
        if not np.isfinite(bic):
            return None
        return _Fit(bic, labels, centers, km, "kmeans")
    except (ValueError, FloatingPointError):
        return None


# The whole model menu, in one place. This is deliberately a dict-and-raise
# rather than the ``_fit_kmeans if method == "kmeans" else _fit_gmm`` ternary it
# replaces: that ternary silently ran the GMM for ANY unrecognised string, so a
# caller passing a method this kernel does not implement got plausible numbers
# from the wrong model instead of an error. With more than two entries that is a
# correctness bug no test can see.
_FITTERS: Dict[str, Callable[[np.ndarray, int, int, int], Optional[_Fit]]] = {
    "gmm": _fit_gmm,
    "kmeans": _fit_kmeans,
}


def _covariance_orientation(
    covariances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cluster principal axes and full axis lengths from covariances.

    ``covariances`` is ``(k, d, d)``. Returns ``(axes, lengths)`` where ``axes``
    is ``(k, d, d)`` with **columns** the unit eigenvectors ordered by descending
    eigenvalue and flipped to a right-handed frame (``det = +1``), and ``lengths``
    is ``(k, d)`` the **full** axis lengths in the same order and the same units
    as the fitted space.

    Two caveats the caller must respect, both of which make this a good
    *initialiser* and a poor final answer:

    * The ``sqrt(5)`` factor (:data:`_UNIFORM_SOLID_MOMENT`) converts a Gaussian
      second moment into the extent of a uniformly-filled solid. Omit it and the
      axes come out 2.24x too small.
    * Where two objects touch, the point cloud belonging to each is *truncated*
      at the contact, so its covariance is biased and the Gaussian systematically
      under-reports that object's size.
    """
    cov = np.asarray(covariances, dtype=float)
    if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
        raise ValueError(
            f"covariances must be (k, d, d); got {cov.shape}"
        )
    k, d = cov.shape[0], cov.shape[1]
    axes = np.zeros((k, d, d), dtype=float)
    lengths = np.zeros((k, d), dtype=float)
    for i in range(k):
        # eigh: ascending eigenvalues, orthonormal eigenvectors as COLUMNS.
        vals, vecs = np.linalg.eigh(cov[i])
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # A reflection is not a rotation; downstream pose code needs det=+1.
        if np.linalg.det(vecs) < 0:
            vecs[:, -1] = -vecs[:, -1]
        axes[i] = vecs
        lengths[i] = 2.0 * np.sqrt(
            np.maximum(vals, 0.0) * _UNIFORM_SOLID_MOMENT
        )
    return axes, lengths


def _fit_extras(fit: _Fit, x_um: np.ndarray) -> Dict[str, Any]:
    """Soft responsibilities, covariances and orientation for the winning fit.

    Every value here is either read straight off the fitted estimator or is one
    eigendecomposition away from it — the GMM already computes its covariances
    and can already report ``predict_proba``, so this costs essentially nothing
    and was previously being discarded.

    KMeans is spherical and hard-assigning by construction, so it reports a
    one-hot ``resp`` and ``covariances = None`` rather than faking either.
    """
    k = int(fit.means.shape[0])
    if fit.kind == "gmm":
        resp = np.asarray(fit.model.predict_proba(x_um), dtype=np.float32)
        cov = np.asarray(fit.model.covariances_, dtype=float)
        axes, lengths = _covariance_orientation(cov)
        return {"resp": resp, "covariances": cov,
                "axes": axes, "axis_lengths_um": lengths}
    resp = np.zeros((x_um.shape[0], k), dtype=np.float32)
    if x_um.shape[0]:
        resp[np.arange(x_um.shape[0]), fit.labels] = 1.0
    return {"resp": resp, "covariances": None,
            "axes": None, "axis_lengths_um": None}


def cluster_granules(
    points_zyx: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cluster a bead point cloud into granules by a BIC-selected mixture model.

    Parameters
    ----------
    points_zyx : (N, 3) float
        Bead centroids in **voxel** units, ordered ``(z, y, x)`` (P0 convention).
    voxel_size_um : (dz, dy, dx)
        Physical voxel size in µm. Coordinates are scaled to µm before fitting so
        anisotropic Z does not bias the model. ``None`` ⇒ isotropic ``(1, 1, 1)``.
    params : dict
        ``n_granules`` (int, seed count), ``relax_pct`` (float 0–100, the ±p%),
        ``method`` (one of :data:`_FITTERS`), ``n_init`` (int, restarts),
        ``noise_resp`` (float 0–1, default ``0.0`` = off): when positive, any
        point whose top responsibility falls below it is relabelled
        :data:`NOISE_LABEL`. Off by default, so the historical
        every-point-is-assigned behaviour is unchanged unless asked for.

    Returns
    -------
    labels : (N,) int
        Per-point granule id in ``0 .. k-1``, or :data:`NOISE_LABEL` where
        ``noise_resp`` rejected the point. **Downstream consumers that iterate
        clusters must skip the negative label.**
    info : dict
        ``k``, ``bic_by_k`` (``{k: bic}``), ``means`` ((k, 3) µm), ``resp``
        ((N, k) float32 soft responsibilities), ``covariances`` ((k, 3, 3) or
        ``None`` for a spherical model), ``axes`` ((k, 3, 3), unit eigenvectors as
        columns, descending eigenvalue, ``det = +1``, or ``None``) and
        ``axis_lengths_um`` ((k, 3) full axis lengths, or ``None``).
        See :func:`_covariance_orientation` for the two caveats that make ``axes``
        a good initialiser and a poor final answer.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If ``method`` is not one of :data:`_FITTERS`.
    """
    _require_sklearn()

    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    n = int(pts.shape[0])

    if voxel_size_um is None:
        dz = dy = dx = 1.0
    else:
        dz, dy, dx = (float(voxel_size_um[0]),
                      float(voxel_size_um[1]),
                      float(voxel_size_um[2]))
    x_um = pts * np.array([dz, dy, dx], dtype=float)

    params = params or {}
    n_granules = max(1, int(params.get("n_granules", 1) or 1))
    relax_pct = max(0.0, float(params.get("relax_pct", 0.0) or 0.0))
    method = str(params.get("method", "gmm") or "gmm").strip().lower()
    n_init = max(1, int(params.get("n_init", 1) or 1))
    noise_resp = min(1.0, max(0.0, float(params.get("noise_resp", 0.0) or 0.0)))

    try:
        fit_one = _FITTERS[method]
    except KeyError:
        raise ValueError(
            f"unknown clustering method {method!r}; "
            f"expected one of {sorted(_FITTERS)}"
        ) from None

    def _empty_extras(kk: int) -> Dict[str, Any]:
        return {"resp": np.zeros((n, kk), dtype=np.float32),
                "covariances": None, "axes": None, "axis_lengths_um": None}

    # Empty cloud → nothing to cluster.
    if n == 0:
        return (np.zeros((0,), dtype=int),
                {"k": 0, "bic_by_k": {},
                 "means": np.zeros((0, 3), dtype=float), **_empty_extras(0)})

    # BIC sweep bounds, clamped so k never exceeds the number of points.
    p = relax_pct / 100.0
    k_lo = max(1, int(round(n_granules * (1.0 - p))))
    k_hi = max(k_lo, int(round(n_granules * (1.0 + p))))
    k_hi = min(k_hi, n)
    k_lo = max(1, min(k_lo, k_hi))

    bic_by_k: Dict[int, float] = {}
    fitted: Dict[int, _Fit] = {}
    for k in range(k_lo, k_hi + 1):
        result = fit_one(x_um, k, n_init, _RANDOM_STATE)
        if result is None:
            continue
        bic_by_k[k] = result.bic
        fitted[k] = result

    # Every fit failed (extreme degeneracy) → collapse to a single cluster.
    if not fitted:
        one = _empty_extras(1)
        one["resp"][:] = 1.0
        return (np.zeros((n,), dtype=int),
                {"k": 1, "bic_by_k": {},
                 "means": x_um.mean(axis=0, keepdims=True), **one})

    best_k = min(bic_by_k, key=lambda kk: bic_by_k[kk])
    best = fitted[best_k]
    # Extras come from the WINNER only — computing them for every k in the sweep
    # would allocate an (N, k) responsibility matrix per candidate for nothing.
    extras = _fit_extras(best, x_um)

    labels = best.labels
    if noise_resp > 0.0 and labels.size:
        rejected = extras["resp"].max(axis=1) < noise_resp
        if rejected.any():
            labels = labels.copy()
            labels[rejected] = NOISE_LABEL

    info: Dict[str, Any] = {
        "k": int(best_k),
        "bic_by_k": bic_by_k,
        "means": best.means,
        **extras,
    }
    return labels, info
