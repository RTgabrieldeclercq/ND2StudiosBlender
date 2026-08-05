"""ZS-DeconvNet — zero-shot denoising + deconvolution (Qiao et al. 2024).

WHERE THE REAL MATH LIVES
    Here. This module is the numeric port of the authors' reference implementation:
    the two-stage network graphs, the self-supervised losses, the re-corruption /
    axial-sampling data augmentation, the training loops, and the tiled inference +
    fusion. The catalog node (:mod:`nodegraph.catalog.enhance.zs_deconvnet`) owns only
    Dataset/metadata plumbing and hands this module plain arrays.

    Method: C. Qiao et al., "Zero-shot learning enables instant denoising and
    super-resolution in optical fluorescence microscopy", *Nature Communications*
    15:4180 (2024), doi:10.1038/s41467-024-48575-9.

PROVENANCE — github.com/TristaZeng/ZS-DeconvNet @ main, fetched 2026-08-04
    Python_MATLAB_Codes/train_inference_python/
      models/twostage_Unet.py        (Unet)          -> :func:`build_unet2d`
      models/twostage_Unet3D.py      (Unet)          -> :func:`build_unet3d`
      models/twostage_RCAN3D.py      (RCAN3D)        -> :func:`build_rcan3d`
      utils/utils.py                 (prctile_norm)  -> :func:`prctile_norm`
      utils/loss.py    (create_psf_loss)             -> :func:`make_psf_loss_2d`
                       (create_NBR2NBR_loss)         -> :func:`make_nbr2nbr_loss_3d`
                       (create_psf_loss_3D_NBR2NBR)  -> :func:`make_psf_loss_3d`
                       (psf_estimator_2d/3d)         -> :func:`psf_sigma`
      Infer_2D.py                                    -> :func:`infer_2d`
      Infer_3D.py                                    -> :func:`infer_3d`, :func:`fourier_damp`
      Train_ZSDeconvNet_2D.py                        -> :func:`train_2d`
      Train_ZSDeconvNet_3D.py                        -> :func:`train_3d`
    data_augment_recorrupt_matlab/
      GenData4ZS-DeconvNet/DataAugmFor2D.m           -> :func:`recorrupt`, :func:`estimate_beta2`
      GenData4ZS-DeconvNet/DataAugmFor3D.m           -> :func:`axial_split`, :func:`sample_patches_3d`
      XxUtils/SegTool/XxCalMask.m                    -> :func:`foreground_mask`
      XxUtils/SegTool/XxDataSeg_ForTrain.m           -> :func:`sample_patches_2d`
    Paper equations: Eq. 3-5 (2D losses), Eq. 6-8 (3D losses + GAR),
    Eq. 9-12 (re-corruption), Methods "Implementation of 2D/3D ZS-DeconvNet".

DELIBERATE DEVIATIONS FROM THE REFERENCE  (each one is a decision, not an accident)
  1. **Keras 3, not Keras 2.** The reference targets TF 2.5 (`tensorflow.keras` == Keras
     2). This environment has TF 2.21 / Keras 3.14, so `keras.backend.conv2d` and
     `optimizers.Adam(decay=...)` no longer exist. `K.conv2d/conv3d` become
     `tf.nn.conv2d/conv3d` (same op, same SAME padding) and the optimizer's `decay=1e-5`
     is dropped -- it was legacy per-step decay that Keras 3 removed, and the reference
     ALSO applies an explicit ×0.5 step decay, which is kept and is the one that matters.
     `LeakyReLU(alpha=)` becomes `negative_slope=` (renamed, same maths).
  2. **The layer ORDER is preserved exactly**, because a legacy `.h5` checkpoint is keyed
     by topological order of weight-bearing layers
     (`keras.src.legacy.saving.legacy_h5_format.load_weights_from_hdf5_group`). Do not
     "tidy" the builders: reordering or inserting a weighted layer silently loads the
     authors' published weights into the wrong tensors. That failure is invisible without
     the golden-output check (:mod:`nodegraph.selftest`).
  3. **No uint16 round trip.** The reference writes augmented pairs and outputs through
     `uint16(65535*x)` / `uint16(1e4*x)` files. Arrays stay float32 in memory here, so the
     quantization those casts imposed is simply absent.
  4. **Dihedral augmentation, not bilinear rotation.** `XxDataSeg_ForTrain` rotates each
     patch by a random 0-360 deg with bilinear interpolation. Interpolation spatially
     CORRELATES the per-pixel noise, and the re-corruption scheme's whole premise is that
     the two corrupted copies differ by pixel-wise independent noise (Eq. 9-12). The
     8-element dihedral group (`np.rot90` + flips) is exact, allocation-cheap, and
     preserves that independence; it is also the standard augmentation for denoising nets.
  5. **PSF resampling by `scipy.ndimage.zoom`.** Both training scripts re-grid the PSF from
     its own pixel size to the data's with ~60 lines of hand-rolled `interp1d` walks
     outward from the centre. `zoom(order=1)` is the same piecewise-linear resampling in
     one call; the result is re-centred to an odd size and renormalized to sum 1, which is
     what those walks were arranging.
  6. **`.mrc` OTF input is refused, not silently mishandled.** `cal_psf_2d`/`cal_psf_3d`
     reconstruct a PSF from an SIM-style radially-averaged OTF whose radial sampling comes
     from hard-coded constants (`dkzotf = 0.0497`, `dkrotf = 0.0302`) that are properties of
     the authors' own microscopes. Porting that without their OTF files to check against
     would be a confident guess. `.tif` PSFs (what `create_PSF.m` emits) and the
     metadata-derived Gaussian are supported; `.mrc` raises and says so.
  7. **No `np.flip(axis=1)`.** `Infer_3D` flips its input on load and un-flips on save.
     That is a TIFF row-order quirk of the authors' files, not part of the algorithm, and
     it cancels out; porting it would mirror every volume this engine feeds in.
  8. **SIM variants are out of scope.** ZS-DeconvNet-SIM needs raw structured-illumination
     frames plus a SIM reconstruction algorithm, neither of which exists in this repo.
     `RCAN3D_SIM*` and `Train_ZSDeconvNet_*SIM.py` are not ported.

Backends (tensorflow/keras, scipy) are imported lazily INSIDE the functions that need
them, so importing this module stays cheap and the core stays framework-free.
"""
from __future__ import annotations

import os
import threading

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "prctile_norm", "foreground_mask", "recorrupt", "estimate_beta2", "axial_split",
    "psf_sigma", "crop_psf", "resample_psf", "load_psf_tif", "gaussian_psf_from_sigmas",
    "build_unet2d", "build_unet3d", "build_rcan3d", "ARCHITECTURES",
    "net_padding", "get_model", "clear_model_cache",
    "make_psf_loss_2d", "make_psf_loss_3d", "make_nbr2nbr_loss_3d",
    "sample_patches_2d", "sample_patches_3d", "train_2d", "train_3d",
    "infer_2d", "infer_3d", "fourier_damp",
    "DEFAULTS",
]

#: The reference's own hyper-parameter defaults, by dim. Collected here so the catalog
#: node's socket defaults and this module's fallbacks cannot drift apart, and so a reader
#: can see at a glance which number came from which demo script.
#:
#: 2D: ``Train_ZSDeconvNet_2D.py`` argparse defaults + ``DataAugmFor2D.m`` signature.
#: 3D: ``Train_ZSDeconvNet_3D.py`` argparse defaults + ``DataAugmFor3D.m`` signature.
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "2D": {"iterations": 50000, "batch_size": 4, "patch": 128, "insert_xy": 16,
           "start_lr": 5e-5, "lr_decay_factor": 0.5, "lr_decay_every": 10000,
           "hess_weight": 0.02, "tv_weight": 0.0, "denoise_weight": 0.5,
           "background": 100.0, "alpha": 1.0, "beta1": 1.0},
    "3D": {"iterations": 10000, "batch_size": 3, "patch": 64, "patch_z": 13,
           "insert_xy": 8, "insert_z": 2, "start_lr": 1e-4, "lr_decay_factor": 0.5,
           "hess_weight": 0.1, "tv_weight": 0.0, "gar_weight": 1.0,
           "background": 100.0},
}


# ── normalization ─────────────────────────────────────────────────────────────

def prctile_norm(x: np.ndarray, min_prc: float = 0.0,
                 max_prc: float = 100.0) -> np.ndarray:
    """Percentile min-max normalization to ``[0, 1]`` — ``utils/utils.py::prctile_norm``.

    ``(x - p_lo) / (p_hi - p_lo + 1e-7)`` then clipped to ``[0, 1]``. The ``1e-7`` guard and
    the clip are the reference's, and both matter: a flat array would divide by zero without
    the guard, and the clip is what makes the low percentile a genuine *cut* rather than a
    shift (with ``min_prc=3`` the dimmest 3 % of the image is pinned to exactly 0).

    Non-finite input is excluded from the percentiles — a single NaN otherwise makes
    ``np.percentile`` return NaN for the whole array and the result is uniformly NaN. NaNs
    are then mapped to 0, which is the same answer the clip gives every other under-range
    sample.
    """
    a = np.asarray(x, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.float32)
    lo = float(np.percentile(finite, min_prc))
    hi = float(np.percentile(finite, max_prc))
    y = (a - lo) / (hi - lo + 1e-7)
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0),
                   0.0, 1.0).astype(np.float32)


# ── foreground mask + patch sampling ──────────────────────────────────────────

def foreground_mask(img: np.ndarray, ksize: int = 3,
                    thresh: float = 5e-2) -> np.ndarray:
    """Band-pass foreground mask — ``XxUtils/SegTool/XxCalMask.m``.

    ``(G_ksize * img) - (G_50 * img) >= thresh``: a difference of Gaussians, i.e. structure
    at the fine scale minus a broad background estimate. Used only to decide WHERE to cut
    training patches, so that a 128 px patch lands on cells rather than on empty coverslip.

    MATLAB's ``fspecial('gaussian',[k,k],k)`` is a ``k×k`` window whose **sigma is also k**,
    so ``ksize=3`` is a 3×3 window at sigma 3 — heavily truncated, very nearly a 3×3 box
    blur. The background kernel is ``[100,100]`` at sigma 50, likewise truncated at one
    sigma. `scipy.ndimage.gaussian_filter` is sigma-parameterized with its own radius, so
    the truncation is reproduced explicitly via ``radius``; using the untruncated default
    would make the background term far broader than the reference's and shift the mask.
    ``mode="nearest"`` is MATLAB's ``'replicate'``.
    """
    from scipy.ndimage import gaussian_filter
    a = np.asarray(img, dtype=np.float32)
    k = max(1, int(ksize))
    fine = gaussian_filter(a, sigma=float(k), radius=k // 2, mode="nearest")
    back = gaussian_filter(a, sigma=50.0, radius=50, mode="nearest")
    return (fine - back) >= float(thresh)


def _mask_points(ref: np.ndarray, min_points: int = 1000) -> np.ndarray:
    """Candidate patch-centre coordinates — the mask loop of ``XxDataSeg_ForTrain.m``.

    ``thresh_mask = 0.07 * max(ref)`` with ``ksize=3``, relaxed by ×0.8 until the mask holds
    at least ``min_points`` pixels (the reference's ``1e3``). The relaxation is capped so a
    blank plane cannot spin forever; if the mask never fills, every in-bounds pixel becomes a
    candidate, which is the honest fallback for an image with no detectable structure.

    Returns an ``(N, ndim)`` integer array of coordinates.
    """
    a = np.asarray(ref, dtype=np.float32)
    thresh = 0.07 * float(a.max() if a.size else 0.0)
    for _ in range(200):
        if thresh <= 0:
            break
        mask = foreground_mask(a, ksize=3, thresh=thresh)
        if int(mask.sum()) >= min_points:
            return np.argwhere(mask)
        thresh *= 0.8
    return np.argwhere(np.ones(a.shape, dtype=bool))


def _dihedral(patch: np.ndarray, k: int, *, axes: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """One of the 8 dihedral transforms of ``patch`` in the plane spanned by ``axes``.

    Deviation 4 in the module docstring: this replaces the reference's random-angle bilinear
    rotation. Exact, interpolation-free, and it preserves the pixel-wise independence of the
    re-corruption noise that the self-supervised loss assumes.
    """
    out = np.rot90(patch, k % 4, axes=axes)
    if k >= 4:
        out = np.flip(out, axis=axes[0])
    return np.ascontiguousarray(out)


def sample_patches_2d(inp: np.ndarray, gt: np.ndarray, n: int, size: int,
                      rng: np.random.Generator,
                      ref: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` co-registered ``size×size`` patch pairs — ``XxDataSeg_ForTrain.m``.

    Centres are drawn uniformly from the band-pass foreground of ``ref`` (default: ``gt``),
    which is what keeps patches on structure. Both members of a pair get the SAME crop and
    the SAME dihedral transform, because the pair is the network's (input, target) and any
    relative geometric change between them would be a systematic error the loss would
    happily fit.

    The reference's `thresh_ar` / `thresh_sum_gt` / `thresh_ar_gt` rejection loop is NOT
    ported, and nothing is lost: all three thresholds are initialized to literal ``0`` in
    ``XxDataSeg_ForTrain.m`` (lines 51-53) and only ever *decrease*, so its ``while``
    condition is false on the first test and the loop body never executes.

    Returns ``(inputs, gts)``, each ``(n, size, size)`` float32.
    """
    a = np.asarray(inp, dtype=np.float32)
    b = np.asarray(gt, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"patch pair shape mismatch: {a.shape} vs {b.shape}")
    h, w = a.shape
    if size > h or size > w:
        raise ValueError(
            f"zs-deconvnet: training patch {size}x{size} does not fit the "
            f"{h}x{w} image. Lower the patch size, or crop less upstream.")
    pts = _mask_points(b if ref is None else ref)
    half = size // 2
    lo_y, hi_y = half, h - (size - half)
    lo_x, hi_x = half, w - (size - half)
    keep = ((pts[:, 0] >= lo_y) & (pts[:, 0] <= hi_y) &
            (pts[:, 1] >= lo_x) & (pts[:, 1] <= hi_x))
    pts = pts[keep]
    if pts.size == 0:                     # no in-bounds foreground: centre every patch
        pts = np.array([[h // 2, w // 2]], dtype=np.int64)
    pick = rng.integers(0, len(pts), size=int(n))
    ks = rng.integers(0, 8, size=int(n))
    xs = np.empty((int(n), size, size), dtype=np.float32)
    ys = np.empty((int(n), size, size), dtype=np.float32)
    for i, (p, k) in enumerate(zip(pick, ks)):
        cy, cx = int(pts[p, 0]), int(pts[p, 1])
        sy, sx = cy - half, cx - half
        xs[i] = _dihedral(a[sy:sy + size, sx:sx + size], int(k))
        ys[i] = _dihedral(b[sy:sy + size, sx:sx + size], int(k))
    return xs, ys


def axial_split(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a ``(Z,Y,X)`` volume into its two axial parities — the parameter-free 3D
    augmentation of ``DataAugmFor3D.m`` (``data(...,z1+1:2:z2)`` vs ``data(...,z1:2:z2)``).

    This is why 3D ZS-DeconvNet needs no noise hyper-parameters at all: two interleaved
    half-stacks of the same specimen are two independent noisy measurements of (very nearly)
    the same signal, so one can serve as the other's target — the spatially-interleaved
    self-supervision of the paper's Eq. 6-8. The cost is that each half is sampled at twice
    the z step, which is exactly the expectation gap the GAR term corrects.

    Returns ``(even_parity, odd_parity)``, both ``(floor(Z/2), Y, X)`` and equal length.
    """
    v = np.asarray(vol)
    n = v.shape[0] // 2
    if n < 1:
        raise ValueError(
            f"zs-deconvnet: 3D zero-shot training needs at least 2 z planes to split "
            f"into two axial parities; this volume has {v.shape[0]}. Use the 2D lever, "
            f"or the `pretrained` mode.")
    return v[0:2 * n:2], v[1:2 * n:2]


def sample_patches_3d(vol: np.ndarray, n: int, size: int, size_z: int,
                      rng: np.random.Generator
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` co-registered ``(size_z, size, size)`` axial-parity patch pairs —
    ``DataAugmFor3D.m``.

    A window of ``2*size_z`` consecutive planes is cut, then split by parity, so input and
    target interleave in z exactly as the reference's ``z1+1:2:z2`` / ``z1:2:z2`` pair does.
    Lateral centres come from the same band-pass foreground as 2D, computed on the volume's
    max projection so that a patch lands on structure somewhere in its depth rather than on
    whichever plane happened to be sampled.

    Returns ``(inputs, gts)``, each ``(n, size_z, size, size)`` float32.
    """
    v = np.asarray(vol, dtype=np.float32)
    nz, h, w = v.shape
    need_z = 2 * int(size_z)
    if need_z > nz:
        raise ValueError(
            f"zs-deconvnet: 3D training needs {need_z} planes per patch (2 x patch_z="
            f"{size_z}, one per axial parity) but the volume has {nz}. Lower `patch_z` "
            f"to at most {nz // 2}.")
    if size > h or size > w:
        raise ValueError(
            f"zs-deconvnet: training patch {size}x{size} does not fit the {h}x{w} "
            f"volume. Lower the patch size.")
    pts = _mask_points(v.max(axis=0))
    half = size // 2
    keep = ((pts[:, 0] >= half) & (pts[:, 0] <= h - (size - half)) &
            (pts[:, 1] >= half) & (pts[:, 1] <= w - (size - half)))
    pts = pts[keep]
    if pts.size == 0:
        pts = np.array([[h // 2, w // 2]], dtype=np.int64)
    pick = rng.integers(0, len(pts), size=int(n))
    z0s = rng.integers(0, nz - need_z + 1, size=int(n))
    ks = rng.integers(0, 8, size=int(n))
    xs = np.empty((int(n), size_z, size, size), dtype=np.float32)
    ys = np.empty((int(n), size_z, size, size), dtype=np.float32)
    for i, (p, z0, k) in enumerate(zip(pick, z0s, ks)):
        cy, cx = int(pts[p, 0]), int(pts[p, 1])
        sub = v[int(z0):int(z0) + need_z,
                cy - half:cy - half + size, cx - half:cx - half + size]
        even, odd = axial_split(sub)
        # dihedral in the LATERAL plane only (axes 1,2): z is physically distinct from y/x
        # (anisotropic sampling, and the PSF is anisotropic), so mixing z into the symmetry
        # group would present the network with volumes no microscope can produce.
        xs[i] = _dihedral(even, int(k), axes=(1, 2))
        ys[i] = _dihedral(odd, int(k), axes=(1, 2))
    return xs, ys


# ── re-corruption (2D) — paper Eq. 9-12 ───────────────────────────────────────

def estimate_beta2(images: Sequence[np.ndarray], bg: float = 100.0,
                   thresh: float = 0.005) -> float:
    """Estimate the Gaussian noise variance ``beta2`` from the images themselves — the
    ``est_beta2`` block of ``DataAugmFor2D.m``.

    ``beta2`` is the camera's read-noise variance (paper Eq. 12), the one re-corruption
    hyper-parameter that is NOT theoretically 1 and must come from the data or a calibration.
    The recipe: build a mask of pixels that are *background* (normalize 0.1-99.9 %, smooth at
    sigma 5, renormalize, keep everything at or below ``thresh``), take the variance of those
    pixels per image, discard images more than one standard deviation from the mean variance,
    and average what is left.

    The per-image trim is what makes it robust: one field with a bright out-of-focus blob
    inflates its "background" variance enormously, and averaging that in would over-estimate
    the read noise for every other field.

    Returns 0.0 when no image yields a usable background population — the caller then has to
    decide, since a silently-zero ``beta2`` would make ``sigma`` purely Poissonian.
    """
    from scipy.ndimage import gaussian_filter
    variances: List[float] = []
    for img in images:
        a = np.asarray(img, dtype=np.float32) - float(bg)
        m = prctile_norm(a, 0.1, 99.9)
        m = gaussian_filter(m, sigma=5.0, mode="nearest")
        # MIN-MAX, matching MATLAB's `XxNorm(mask)` — not a divide-by-max. The difference is
        # not cosmetic: after the 0.1-99.9 % stretch the background sits at some small but
        # NON-ZERO level (it is signal minus a pedestal, not zero), so dividing by the max
        # alone leaves it just above a 0.005 cut and the mask comes back EMPTY — the
        # estimator then reports 0.0, i.e. "no read noise", and re-corruption silently
        # degrades to a pure-Poisson model. Subtracting the min is what puts the dimmest
        # background at 0 so the threshold means what it says.
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / ((hi - lo) or 1.0)
        keep = m <= float(thresh)
        vals = np.asarray(img, dtype=np.float64)[keep]
        if vals.size >= 2:
            variances.append(float(np.var(vals)))
    if not variances:
        return 0.0
    v = np.asarray(variances, dtype=np.float64)
    if v.size > 2:
        mu, sd = float(v.mean()), float(v.std())
        trimmed = v[(v >= mu - sd) & (v <= mu + sd)]
        if trimmed.size:
            v = trimmed
    return float(v.mean())


def recorrupt(y: np.ndarray, bg: float, beta1: float, beta2: float, alpha: float,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Split one noisy image into two noise-independent copies — paper Eq. 9-12, as
    implemented by ``DataAugmFor2D.m`` lines 128-137.

    .. code::

        sigma = sqrt( max( beta1 * max(H(y) - bg, 0) + beta2, 0 ) )     # Eq. 12
        input = y + alpha * sigma * g                                   # Eq. 9   (y_hat)
        gt    = y - sigma * g / alpha                                   # Eq. 10  (y_tilde)

    with ``H`` a 5x5 box filter and ``g ~ N(0, 1)`` per pixel. ``D = alpha*I`` from the
    paper, so ``D*g`` and ``D^-1*g`` are the two scalings above. The construction is what
    makes the pair usable as (input, target): ``E[gt | y]`` is the clean signal and the two
    noise terms are *negatively* correlated in exactly the way that makes the cross term
    vanish in expectation (Supplementary Note 1), which is why a network trained to map
    ``input -> gt`` learns to denoise rather than to reproduce the noise.

    Both ``max(...)`` clamps are the reference's and both are load-bearing: the inner one
    stops a background-subtracted dim pixel from contributing a negative Poisson variance,
    and the outer one guards the sqrt when ``beta2`` itself is negative or absent.

    The theoretically optimal values are ``beta1 = alpha = 1`` (paper, Supplementary Note 1
    and Supplementary Figs. 3-4); ``beta2`` is the camera's read-noise variance and is
    estimated by :func:`estimate_beta2`. The MATLAB *randomizes* all three over ranges
    (``beta1=[0.5,1.5]``, ``alpha=[0.5,1.5]``, ``beta2=[0.8,1.2]*est``) as an extra
    augmentation across its 100 re-corruptions per cell; the caller here draws per call,
    which is the same thing done per training step.

    Both outputs are jointly min-max normalized to ``[0,1]`` (MATLAB lines 134-137) — one
    shared scale, so the pair stays radiometrically consistent.
    """
    from scipy.ndimage import uniform_filter
    a = np.asarray(y, dtype=np.float32)
    smooth = uniform_filter(a, size=5, mode="nearest")           # H: 5x5 average
    var = float(beta1) * np.maximum(smooth - float(bg), 0.0) + float(beta2)
    sigma = np.sqrt(np.maximum(var, 0.0), dtype=np.float32)
    g = rng.standard_normal(a.shape).astype(np.float32)
    al = float(alpha) or 1.0
    inp = a + sigma * g * al
    gt = a - sigma * g / al
    lo = float(min(inp.min(), gt.min()))
    hi = float(max(inp.max(), gt.max()))
    span = (hi - lo) or 1.0
    return ((inp - lo) / span).astype(np.float32), ((gt - lo) / span).astype(np.float32)


# ── PSF ───────────────────────────────────────────────────────────────────────

def gaussian_psf_from_sigmas(sigmas: Sequence[float],
                             radius_factor: float = 3.0) -> np.ndarray:
    """A normalized n-D Gaussian PSF from per-axis sigmas in PIXELS.

    Identical construction to ``enhance.deconvolve``'s ``gaussian_psf`` — same clamp
    (``max(0.5, sigma)``) and same radius (``round(3*sigma)``) — so the two deconvolution
    nodes in this catalog model the same optics the same way, and the σ derivation itself
    stays single-sourced in ``catalog.enhance.deconvolve.diffraction_sigmas``.
    """
    sig = tuple(max(0.5, float(s)) for s in sigmas)
    radii = [max(1, int(round(radius_factor * s))) for s in sig]
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    out = np.ones_like(grids[0], dtype=np.float64)
    for coord, s in zip(grids, sig):
        out = out * np.exp(-(coord.astype(np.float64) ** 2) / (2.0 * s * s))
    total = out.sum()
    return (out / total if total else out).astype(np.float32)


def load_psf_tif(path: str) -> np.ndarray:
    """Read a measured/simulated PSF from a TIFF — 2D ``(Y,X)`` or 3D ``(Z,Y,X)``.

    ``.mrc`` is refused (deviation 6): those files are radially-averaged OTFs whose radial
    sampling lives in constants specific to the authors' microscopes, so reconstructing a
    PSF from one without their files to check against would be guesswork.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mrc", ".otf"):
        raise ValueError(
            f"zs-deconvnet: {os.path.basename(path)} is an .mrc OTF, which this port does "
            "not read — reconstructing a PSF from a radially-averaged OTF needs the radial "
            "sampling constants of the microscope that produced it. Supply a PSF as a TIFF "
            "(what the authors' create_PSF.m writes), or leave the PSF path empty to derive "
            "a Gaussian PSF from the file's own NA / emission / pixel size.")
    import tifffile
    psf = np.asarray(tifffile.imread(path), dtype=np.float32)
    psf = np.squeeze(psf)
    if psf.ndim not in (2, 3):
        raise ValueError(
            f"zs-deconvnet: PSF {os.path.basename(path)} has shape {psf.shape}; a 2D (Y,X) "
            "or 3D (Z,Y,X) PSF is required.")
    psf = np.maximum(psf, 0.0)
    total = float(psf.sum())
    return (psf / total if total else psf).astype(np.float32)


def psf_sigma(psf: np.ndarray) -> Tuple[float, ...]:
    """Per-axis Gaussian sigma of ``psf`` in pixels — ``psf_estimator_2d`` /
    ``psf_estimator_3d`` in ``utils/loss.py``.

    A 1-D Gaussian is least-squares fitted to the profile through the PSF's maximum along
    each axis. The reference uses this only to choose the PSF crop radius
    (:func:`crop_psf`), and it is used for the same purpose here plus as the comparison
    statistic in the validation bench.

    A `curve_fit` failure falls back to the second-moment (RMS) width of the same profile
    rather than raising: an un-fittable profile is a reason to widen the crop, never a
    reason to fail a pull.
    """
    a = np.asarray(psf, dtype=np.float64)
    peak = np.unravel_index(int(np.argmax(a)), a.shape)
    out: List[float] = []
    for ax in range(a.ndim):
        idx: List[Any] = list(peak)
        idx[ax] = slice(None)
        prof = np.asarray(a[tuple(idx)], dtype=np.float64)
        prof = prof - prof.min()
        top = prof.max()
        if top <= 0:
            out.append(0.5)
            continue
        prof = prof / top
        x = np.arange(prof.size, dtype=np.float64)
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(
                lambda t, amp, mu, sd: amp * np.exp(-((t - mu) ** 2) / (2.0 * sd * sd)),
                x, prof, p0=[1.0, float(peak[ax]), 2.0], maxfev=10000)
            sd = abs(float(popt[2]))
            if np.isfinite(sd) and 0 < sd < prof.size:
                out.append(sd)
                continue
        except Exception:                             # noqa: BLE001 — documented fallback
            pass
        w = prof / prof.sum()
        mu = float((x * w).sum())
        out.append(float(np.sqrt(max(((x - mu) ** 2 * w).sum(), 0.25))))
    return tuple(out)


def crop_psf(psf: np.ndarray, max_z: Optional[int] = None) -> np.ndarray:
    """Crop a PSF to ``+/- 4*sigma`` laterally — the "crop PSF for faster computation" block
    of both training scripts (``ksize = int(sigma_y * 4)``).

    The PSF enters the loss as a convolution kernel evaluated every training step, so its
    footprint is a direct multiplier on training cost; 4 sigma keeps >99.99 % of a Gaussian's
    mass. ``max_z`` mirrors the 3D script's ``halfz = min(psf_depth//2, input_z-1)`` clamp,
    which stops an axially long PSF from exceeding the patch depth. Always renormalized to
    sum 1 afterwards, because cropping removes mass and an unnormalized kernel would rescale
    the whole degradation term.
    """
    a = np.asarray(psf, dtype=np.float32)
    sig = psf_sigma(a)
    if a.ndim == 3:
        # (Z,Y,X): lateral sigma is axis 1 (== sigma_y in the reference's YXZ layout)
        half_z = a.shape[0] // 2
        if max_z is not None:
            half_z = min(half_z, max(0, int(max_z)))
        ks = max(1, int(sig[1] * 4))
        cz, cy, cx = (s // 2 for s in a.shape)
        z0, z1 = cz - half_z, cz + half_z + 1
        if ks <= min(cy, cx):
            a = a[z0:z1, cy - ks:cy + ks + 1, cx - ks:cx + ks + 1]
        else:
            a = a[z0:z1]
    else:
        ks = max(1, int(sig[0] * 4))
        cy, cx = (s // 2 for s in a.shape)
        if ks <= min(cy, cx):
            a = a[cy - ks:cy + ks + 1, cx - ks:cx + ks + 1]
    total = float(a.sum())
    return (a / total if total else a).astype(np.float32)


def resample_psf(psf: np.ndarray, src_um: float, dst_um: float,
                 src_z_um: Optional[float] = None,
                 dst_z_um: Optional[float] = None) -> np.ndarray:
    """Re-grid a PSF from its own sampling to the data's — deviation 5.

    A PSF is only meaningful together with the pixel size it was sampled at. `create_PSF.m`
    writes one at ``dxypsf`` (0.0313 µm in the demos) and the training scripts stretch it to
    the data's ``dx`` before it ever enters the loss; skipping that step is the "improper
    training with mismatched PSFs" the paper warns about (Supplementary Fig. 28c), which
    shows up as ringing or as no resolution gain at all.

    Scale factor ``src_um / dst_um``: a PSF sampled FINER than the data (src < dst) shrinks
    in pixel count. The result is forced to an odd size about its centre — the 3D training
    script raises on an even-depth PSF outright, and an even-sized kernel has no centre
    sample, so a `SAME` convolution with one shifts the image by half a pixel.
    """
    a = np.asarray(psf, dtype=np.float32)
    if not src_um or not dst_um:
        return a
    lat = float(src_um) / float(dst_um)
    if a.ndim == 3:
        axial = ((float(src_z_um) / float(dst_z_um))
                 if (src_z_um and dst_z_um) else 1.0)
        zoom = (axial, lat, lat)
    else:
        zoom = (lat, lat)
    if all(abs(z - 1.0) < 1e-9 for z in zoom):
        return a
    from scipy.ndimage import zoom as ndzoom
    out = ndzoom(a, zoom, order=1, mode="constant", cval=0.0)
    out = _to_odd_centered(out)
    total = float(out.sum())
    return (out / total if total else out).astype(np.float32)


def _to_odd_centered(a: np.ndarray) -> np.ndarray:
    """Trim each axis to an ODD length about the array's centre of mass-ish middle."""
    out = np.asarray(a)
    for ax in range(out.ndim):
        if out.shape[ax] % 2 == 0 and out.shape[ax] > 1:
            idx: List[Any] = [slice(None)] * out.ndim
            idx[ax] = slice(0, out.shape[ax] - 1)
            out = out[tuple(idx)]
    return np.ascontiguousarray(out)


# ── network graphs (layer order is a WEIGHT-FILE CONTRACT — see deviation 2) ───

def build_unet2d(input_shape: Sequence[int], upsample_flag: bool, insert_x: int,
                 insert_y: int, conv_block_num: int = 4, conv_num: int = 3):
    """The 2D two-stage U-Net — ``models/twostage_Unet.py::Unet``, layer-for-layer.

    Stage I is a U-Net denoiser whose single-channel output is cropped by
    ``insert_x``/``insert_y`` (the zero-padding margin the caller added, which the loss and
    the target never see). Stage II is a second, structurally identical U-Net that takes
    stage I's output and deconvolves it, optionally 2x upsampled, through two 128-channel
    convolutions to a single channel.

    ``output1`` is cropped INSIDE the graph and ``output2`` is not — that asymmetry is the
    reference's, and every caller (`Infer_2D.py`, `Validate`, `Test`) crops ``output2``
    itself by ``insert * (1 + upsample)``. :func:`infer_2d` does the same.

    Every activation is ``relu``, including both output layers, so the network cannot emit a
    negative intensity.
    """
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                         concatenate)
    from tensorflow.keras.models import Model

    def conv_block(x, ch):
        for _ in range(conv_num):
            x = Conv2D(ch, kernel_size=3, activation="relu", padding="same")(x)
        return MaxPooling2D(pool_size=(2, 2))(x), x

    def concat_block(a, b, ch):
        up = concatenate([UpSampling2D(size=(2, 2))(a), b], axis=3)
        c = Conv2D(ch, kernel_size=3, activation="relu", padding="same")(up)
        for _ in range(conv_num - 1):
            c = Conv2D(ch // 2, kernel_size=3, activation="relu", padding="same")(c)
        return c

    inputs = Input(tuple(input_shape))
    _, h, w, _ = inputs.shape
    pool = inputs
    output1 = None
    conv = None
    for stage in range(2):
        conv_list = []
        channels = 32
        for n in range(conv_block_num):
            channels = 2 ** (n + 5)
            pool, c = conv_block(pool, channels)
            conv_list.append(c)
        mid = Conv2D(channels * 2, kernel_size=3, activation="relu",
                     padding="same")(pool)
        mid = Conv2D(channels, kernel_size=3, activation="relu", padding="same")(mid)
        init_channels, conv = channels, mid
        for n in range(conv_block_num):
            conv = concat_block(conv, conv_list[-(n + 1)], init_channels // (2 ** n))
        if stage == 0:
            output1 = Conv2D(1, kernel_size=3, activation="relu", padding="same")(conv)
            pool = output1
    if upsample_flag:
        conv = UpSampling2D(size=(2, 2))(conv)
    conv = Conv2D(128, kernel_size=3, activation="relu", padding="same")(conv)
    conv = Conv2D(128, kernel_size=3, activation="relu", padding="same")(conv)
    output2 = Conv2D(1, kernel_size=3, activation="relu", padding="same")(conv)
    return Model(inputs=inputs,
                 outputs=[output1[:, insert_x:h - insert_x, insert_y:w - insert_y, :],
                          output2])


def build_unet3d(input_shape: Sequence[int], upsample_flag: bool, insert_z: int = 2,
                 insert_xy: int = 8, conv_block_num: int = 4, conv_num: int = 3):
    """The 3D two-stage U-Net — ``models/twostage_Unet3D.py::Unet``, layer-for-layer.

    Data layout is ``(X, Y, Z, 1)`` — z LAST, which is the reference's convention throughout
    the 3D path and the reason :func:`infer_3d` transposes. Pooling and upsampling are
    ``(2, 2, 1)``: **lateral only, never axial**, because a microscope volume is already
    coarsely sampled in z and halving it again would destroy what little axial detail the
    stack has. That is also why only the lateral extent must be divisible by
    ``2**conv_block_num``.
    """
    from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, UpSampling3D,
                                         concatenate)
    from tensorflow.keras.models import Model

    def conv_block(x, ch):
        for _ in range(conv_num):
            x = Conv3D(ch, kernel_size=3, activation="relu", padding="same")(x)
        return MaxPooling3D(pool_size=(2, 2, 1))(x), x

    def concat_block(a, b, ch):
        up = concatenate([UpSampling3D(size=(2, 2, 1))(a), b], axis=4)
        c = Conv3D(ch, kernel_size=3, activation="relu", padding="same")(up)
        for _ in range(conv_num - 1):
            c = Conv3D(ch // 2, kernel_size=3, activation="relu", padding="same")(c)
        return c

    inputs = Input(tuple(input_shape))
    _, h, w, d, _ = inputs.shape
    pool = inputs
    output1 = None
    conv = None
    for stage in range(2):
        conv_list = []
        channels = 32
        for n in range(conv_block_num):
            channels = 2 ** (n + 5)
            pool, c = conv_block(pool, channels)
            conv_list.append(c)
        mid = Conv3D(channels * 2, kernel_size=3, activation="relu",
                     padding="same")(pool)
        mid = Conv3D(channels, kernel_size=3, activation="relu", padding="same")(mid)
        init_channels, conv = channels, mid
        for n in range(conv_block_num):
            conv = concat_block(conv, conv_list[-(n + 1)], init_channels // (2 ** n))
        if stage == 0:
            output1 = Conv3D(1, kernel_size=3, activation="relu", padding="same")(conv)
            pool = output1
    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(64, kernel_size=3, activation="relu", padding="same")(conv)
    conv = Conv3D(64, kernel_size=3, activation="relu", padding="same")(conv)
    output2 = Conv3D(1, kernel_size=3, activation="relu", padding="same")(conv)
    output1 = output1[:, insert_xy:h - insert_xy, insert_xy:w - insert_xy,
                      insert_z:d - insert_z, :]
    return Model(inputs=inputs, outputs=[output1, output2])


def build_rcan3d(input_shape: Sequence[int], upsample_flag: bool, insert_z: int = 2,
                 insert_xy: int = 8, n_ResGroup: int = 2, n_RCAB: int = 2):
    """The 3D two-stage RCAN — ``models/twostage_RCAN3D.py::RCAN3D``, layer-for-layer.

    The paper's 3D backbone (Fig. 3a) and the reference's default ``--model``. Residual
    channel-attention blocks with **no pooling anywhere**, which has two consequences worth
    knowing: the lateral extent needs no divisibility (unlike :func:`build_unet3d`), and the
    receptive field is small, so it is far cheaper per voxel than the 2D U-Net despite being
    volumetric.

    ``CALayer``'s global average pool is over ``(1,2,3)`` — the whole volume — via a ``Lambda``
    so the reduction stays a graph op. ``add`` and ``multiply`` are Keras layers with no
    weights, so they do not affect the topological weight order that a legacy ``.h5`` is
    keyed by.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Conv3D, LeakyReLU, Lambda, UpSampling3D,
                                         add, multiply)
    from tensorflow.keras.models import Model

    def ca_layer(x, channel, reduction=16):
        w = Lambda(lambda t: tf.reduce_mean(t, axis=(1, 2, 3), keepdims=True))(x)
        w = Conv3D(channel // reduction, kernel_size=1, activation="relu",
                   padding="same")(w)
        w = Conv3D(channel, kernel_size=1, activation="sigmoid", padding="same")(w)
        return multiply([x, w])

    def rcab(x, ch):
        c = Conv3D(ch, kernel_size=3, padding="same")(x)
        c = LeakyReLU(negative_slope=0.2)(c)
        c = Conv3D(ch, kernel_size=3, padding="same")(c)
        c = LeakyReLU(negative_slope=0.2)(c)
        return add([ca_layer(c, ch, reduction=16), x])

    def residual_group(x, ch):
        for _ in range(n_RCAB):
            x = rcab(x, ch)
        return x

    inputs = Input(tuple(input_shape))
    _, h, w, d, _ = inputs.shape
    conv = Conv3D(64, kernel_size=3, padding="same")(inputs)
    res = conv
    for _ in range(n_ResGroup):
        conv = residual_group(conv, 64)
    conv = res + conv
    conv = Conv3D(64, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(negative_slope=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding="same")(conv)
    output1 = LeakyReLU(negative_slope=0.2)(conv)

    conv = Conv3D(64, kernel_size=3, padding="same")(output1)
    res = conv
    for _ in range(n_ResGroup):
        conv = residual_group(conv, 64)
    conv = res + conv
    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(64, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(negative_slope=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding="same")(conv)
    output2 = LeakyReLU(negative_slope=0.2)(conv)
    output1 = output1[:, insert_xy:h - insert_xy, insert_xy:w - insert_xy,
                      insert_z:d - insert_z, :]
    return Model(inputs=inputs, outputs=[output1, output2])


#: architecture name -> (builder, dim, pooling levels). ``levels`` is how many factors of 2
#: the lateral extent must be divisible by (0 = no constraint), which is what
#: :func:`net_padding` needs and the one place the RCAN/U-Net difference is recorded.
ARCHITECTURES: Dict[str, Tuple[Callable[..., Any], str, int]] = {
    "unet2d": (build_unet2d, "2D", 4),
    "rcan3d": (build_rcan3d, "3D", 0),
    "unet3d": (build_unet3d, "3D", 4),
}


def net_window(extent: int, tile: int, levels: int) -> int:
    """The tile extent to actually use on one axis.

    For a pooling architecture the window must be **even**, and that is not a stylistic
    preference — it is forced by the geometry. The network's ``output1`` crop is symmetric
    (``[insert : size - insert]``), so the padding must be symmetric, so the padded extent is
    ``extent + 2*insert`` and has the SAME PARITY as ``extent``. An odd extent can therefore
    never be a multiple of 16, and the reference's
    ``insert_x = int((16*n - seg_window_x)/2)`` (``Infer_2D.py`` line 84) silently truncates
    when it is: the padded tile comes out one pixel short of the multiple, and the U-Net's
    pooling and upsampling paths then return different sizes and the skip ``concatenate``
    raises. It never bites upstream only because their demo windows happen to be even.

    Shrinking an odd window by one costs nothing: :func:`_tile_starts` pins the final tile
    flush against the far edge, so an even window over an odd extent is still covered
    completely — the last two tiles simply overlap by one more row.
    """
    win = max(1, min(int(tile), int(extent)))
    if levels > 0 and win % 2:
        win = max(2, win - 1)
    return win


def net_padding(extent: int, insert: int, levels: int = 4) -> int:
    """The symmetric zero-padding margin one axis needs — the ``insert_x``/``insert_y`` loop
    in ``Infer_2D.py`` lines 80-88.

    Two requirements at once: the margin must be at least ``insert`` (the network was trained
    with that much context outside the region it is scored on), and the padded extent must be
    divisible by ``2**levels`` or the U-Net's pooling and upsampling paths return different
    sizes and the skip concatenation fails. The reference grows the multiple until both hold;
    so does this, and then it CHECKS the result instead of truncating (see
    :func:`net_window` for why an odd extent cannot be satisfied).

    ``levels=0`` (RCAN, which never pools) reduces to just ``insert``.
    """
    ins, ext = max(0, int(insert)), int(extent)
    if levels <= 0:
        return ins
    step = 2 ** int(levels)
    n = -(-ext // step)                                   # ceil
    while step * n - ext < 2 * ins:
        n += 1
    pad = step * n - ext
    if pad % 2:
        raise ValueError(
            f"zs-deconvnet: a tile of {ext} px cannot be symmetrically padded to a multiple "
            f"of {step} (needs {pad} px, which is odd). Use an even tile extent — "
            f"`net_window` does this automatically.")
    return pad // 2


# ── model cache ───────────────────────────────────────────────────────────────

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple, Any] = {}


def clear_model_cache() -> None:
    """Drop every cached Keras model. Called by the selftest between cases so one case's
    weights can never satisfy another's cache key."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


def get_model(arch: str, input_shape: Sequence[int], *, upsample: bool,
              insert_xy: int, insert_z: int = 2, insert_x: Optional[int] = None,
              insert_y: Optional[int] = None, weights_path: str = "",
              weights: Optional[List[np.ndarray]] = None):
    """Build (or fetch) the model for one input shape, optionally loading weights.

    Cached because tiled inference builds ONE graph and reuses it for every tile of every
    unit: constructing the 13 M-parameter 2D U-Net is not free, and a time series has
    hundreds of tiles. The key includes the weights identity — a path plus its
    ``st_mtime_ns``/size, so retraining to the same filename still re-keys — and the shape,
    since a differently-shaped tile needs its own graph.

    ``weights`` (an in-memory list, from a training run that was never written to disk)
    bypasses the cache entirely: the arrays are the freshest thing there is and there is no
    stable identity to key them on.

    Loading a legacy Keras-2 ``.h5`` goes through Keras 3's
    ``legacy_h5_format.load_weights_from_hdf5_group``, which matches by TOPOLOGICAL ORDER of
    weight-bearing layers. There is no name check and no shape-mismatch error for a
    same-shaped-but-wrong layer, so a checkpoint from a different architecture can load
    "successfully" and predict plausible nonsense — which is why the arch is a socket the
    user sets, not something guessed from the file.
    """
    builder, _dim, _levels = _arch(arch)
    shape = tuple(int(s) for s in input_shape)
    kw: Dict[str, Any] = dict(upsample_flag=bool(upsample))
    if _dim == "2D":
        # The 2D graph bakes a SEPARATE crop per axis into `output1`, and a non-square tile
        # (or a non-square image smaller than one tile) genuinely needs two different
        # margins — 16-divisibility is solved per axis. Defaulting both to `insert_xy`
        # keeps the square call site unchanged.
        kw.update(insert_x=int(insert_xy if insert_x is None else insert_x),
                  insert_y=int(insert_xy if insert_y is None else insert_y))
    else:
        kw.update(insert_xy=int(insert_xy), insert_z=int(insert_z))

    if weights is not None:
        model = builder(shape, **kw)
        model.set_weights(weights)
        return model

    stamp: Tuple = ()
    if weights_path:
        try:
            st = os.stat(weights_path)
            stamp = (int(st.st_mtime_ns), int(st.st_size))
        except OSError as exc:
            raise ValueError(
                f"zs-deconvnet: cannot read weights {weights_path!r}: {exc}") from exc
    key = (arch, shape, bool(upsample), tuple(sorted(kw.items())), int(insert_z),
           os.path.abspath(weights_path) if weights_path else "", stamp)
    with _MODEL_LOCK:
        hit = _MODEL_CACHE.get(key)
        if hit is not None:
            return hit
    model = builder(shape, **kw)
    if weights_path:
        try:
            model.load_weights(weights_path)
        except Exception as exc:                       # noqa: BLE001 — one clear message
            raise ValueError(
                f"zs-deconvnet: {os.path.basename(weights_path)} would not load into the "
                f"{arch!r} graph at input shape {shape}: {exc}. Legacy .h5 weights are "
                f"matched by layer ORDER, so this usually means the checkpoint is for a "
                f"different architecture (2D U-Net vs 3D RCAN vs 3D U-Net) or was trained "
                f"with a different `upsample` setting.") from exc
    with _MODEL_LOCK:
        _MODEL_CACHE[key] = model
    return model


def _arch(arch: str) -> Tuple[Callable[..., Any], str, int]:
    try:
        return ARCHITECTURES[arch]
    except KeyError:
        raise ValueError(f"zs-deconvnet: unknown architecture {arch!r} — one of "
                         f"{sorted(ARCHITECTURES)}") from None


def save_weights(model, path: str) -> None:
    """Persist a trained model's weights.

    Keras 3 REFUSES ``save_weights`` to a bare ``.h5`` ("The filename must end in
    `.weights.h5`"), so a cache path is normalized to that suffix. Reading legacy ``.h5`` is
    still supported and unaffected — this asymmetry is Keras 3's, not ours.
    """
    out = path if path.endswith(".weights.h5") else (
        path[:-3] + ".weights.h5" if path.endswith(".h5") else path + ".weights.h5")
    d = os.path.dirname(os.path.abspath(out))
    if d:
        os.makedirs(d, exist_ok=True)
    model.save_weights(out)


# ── losses ────────────────────────────────────────────────────────────────────

def make_psf_loss_2d(psf: np.ndarray, hess_weight: float, tv_weight: float,
                     mse: bool, upsample: bool, insert_xy: int):
    """The 2D deconvolution loss — ``utils/loss.py::create_psf_loss`` with
    ``deconv_flag=1``, paper Eq. 5.

    ``L = |y_true - down(conv(y_pred, PSF))| + lambda*R_Hessian(y_pred) + TV*R_TV(y_pred)``

    The *degradation* term is the whole trick: the network's sharp output is blurred by the
    PSF and downsampled back onto the measured grid, and only THEN compared to the
    re-corrupted target. So the loss never needs a sharp ground truth — it asks "would this
    sharp image, imaged by this microscope, have produced what I measured?".

    Order of operations is the reference's and matters: convolve at the network's (possibly
    2x) resolution, resize to half, then crop the padding margin. Convolving after
    downsampling would apply a PSF sampled at the wrong pixel size.

    The Hessian regularizer (paper's ``R_Hessian``, from Huang et al. 2018) penalizes the
    four second differences ``xx, yy, xy, yx``, each as ``l2_loss / size``; it suppresses the
    ringing and speckle an unregularized deconvolution amplifies. ``lambda = 0.02`` for 2D.

    ``laplace_weight`` and ``l1_rate`` are not ported: both demo scripts pass 0, and the 2D
    trainer hard-codes ``laplace_weight=0`` at the call site.
    """
    import tensorflow as tf
    k = np.asarray(psf, dtype=np.float32)
    if k.ndim != 2:
        raise ValueError(f"2D psf loss needs a 2D PSF, got shape {k.shape}")
    kernel = tf.constant(k.reshape(k.shape[0], k.shape[1], 1, 1))
    ins = int(insert_xy)

    def psf_loss(y_true, y_pred):
        _, height, width, _ = y_pred.shape
        y_true = tf.cast(y_true, tf.float32)
        y_conv = tf.nn.conv2d(y_pred, kernel, strides=1, padding="SAME")
        if upsample:
            y_conv = tf.image.resize(y_conv, [height // 2, width // 2])
        y_conv = y_conv[:, ins:y_conv.shape[1] - ins, ins:y_conv.shape[2] - ins, :]
        base = (tf.reduce_mean(tf.square(y_true - y_conv)) if mse
                else tf.reduce_mean(tf.abs(y_true - y_conv)))
        dy = y_pred[:, :height - 1] - y_pred[:, 1:]
        dx = y_pred[:, :, :width - 1] - y_pred[:, :, 1:]
        total = base
        if tv_weight > 0:
            total = total + tv_weight * (
                tf.nn.l2_loss(dx) / tf.cast(tf.size(dx), tf.float32) +
                tf.nn.l2_loss(dy) / tf.cast(tf.size(dy), tf.float32))
        if hess_weight > 0:
            xx = dx[:, :, :width - 2] - dx[:, :, 1:]
            yy = dy[:, :height - 2] - dy[:, 1:]
            xy = dy[:, :, :width - 1] - dy[:, :, 1:]
            yx = dx[:, :height - 1] - dx[:, 1:]
            hess = tf.add_n([tf.nn.l2_loss(t) / tf.cast(tf.size(t), tf.float32)
                             for t in (xx, yy, yx, xy)])
            total = total + hess_weight * hess
        return total

    return psf_loss


def make_nbr2nbr_loss_3d(tv_weight: float, mse: bool):
    """The 3D **denoising** loss with gap-amending regularization —
    ``utils/loss.py::create_NBR2NBR_loss``, paper Eq. 6.

    ``y_true`` arrives as a 2-channel stack: channel 0 is the other axial parity (the
    target), channel 1 is ``output_G``, the *expectation gap* estimate the caller computed by
    running a frozen copy of the network over an interleaved double-z volume.

    ``L = |gt - out| + |out - gt - output_G|``

    The second term is the GAR (``gamma = 1``). It exists because the two axial parities are
    NOT samples of the same signal — they are half a z step apart — so a plain
    neighbour2neighbour loss would train the network to also interpolate that offset, blurring
    z. GAR subtracts the network's own estimate of that systematic difference, leaving only
    the noise for the loss to remove.
    """
    import tensorflow as tf

    def nbr2nbr_loss(y_true, y_pred):
        output_g = y_true[:, :, :, :, 1]
        gt = y_true[:, :, :, :, 0]
        out = tf.squeeze(y_pred, axis=4)
        if out.shape[1] is not None and gt.shape[1] is not None \
                and out.shape[1] > gt.shape[1]:
            out = tf.image.resize(out, (out.shape[1] // 2, out.shape[2] // 2))
        if mse:
            loss = tf.reduce_mean(tf.square(gt - out))
            reg = tf.reduce_mean(tf.square(out - gt - output_g))
        else:
            loss = tf.reduce_mean(tf.abs(gt - out))
            reg = tf.reduce_mean(tf.abs(out - gt - output_g))
        total = loss + reg
        if tv_weight > 0:
            _, h, w, d = out.shape
            dy = out[:, :h - 1] - out[:, 1:]
            dx = out[:, :, :w - 1] - out[:, :, 1:]
            dz = (out[:, :, :, :d - 1] - out[:, :, :, 1:]) if d and d > 1 else 0.0
            total = total + tv_weight * (tf.reduce_mean(tf.square(dx)) +
                                         tf.reduce_mean(tf.square(dy)) +
                                         tf.reduce_mean(tf.square(dz)))
        return total

    return nbr2nbr_loss


def make_psf_loss_3d(psf: np.ndarray, hess_weight: float, tv_weight: float, mse: bool,
                     upsample: bool, insert_z: int, insert_xy: int):
    """The 3D deconvolution loss with GAR — ``utils/loss.py::create_psf_loss_3D_NBR2NBR``,
    paper Eq. 7-8.

    Same degradation-term idea as :func:`make_psf_loss_2d` (blur the sharp prediction with
    the PSF, compare on the measured grid) plus the GAR term of
    :func:`make_nbr2nbr_loss_3d`, and a 3D Hessian penalizing all nine second differences
    when the patch has depth (six when it does not). ``lambda = 0.1`` for 3D, ``gamma = 1``.

    Operation order differs from the 2D loss and is again the reference's: 3D convolves,
    then CROPS, then resizes; 2D convolves, resizes, then crops. The 2024.7 update note in
    the authors' ReadMe records them deliberately moving the 3D crop after the OTF
    multiplication to match training, so this ordering is intentional upstream.

    Data layout is ``(batch, X, Y, Z, 1)``, so ``tf.image.resize`` — which acts on axes 1-2 —
    downsamples LATERALLY, which is the only axis ``upsample`` ever scales.
    """
    import tensorflow as tf
    k = np.asarray(psf, dtype=np.float32)
    if k.ndim != 3:
        raise ValueError(f"3D psf loss needs a 3D PSF, got shape {k.shape}")
    kernel = tf.constant(k.reshape(k.shape[0], k.shape[1], k.shape[2], 1, 1))
    ins_z = int(insert_z)
    ins_xy = int(insert_xy) * (2 if upsample else 1)

    def psf_loss(y_true, y_pred):
        output_g = y_true[:, :, :, :, 1]
        target = y_true[:, :, :, :, 0]
        h, w, d = y_pred.shape[1], y_pred.shape[2], y_pred.shape[3]
        conv = tf.nn.conv3d(y_pred, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        conv = conv[:, ins_xy:h - ins_xy, ins_xy:w - ins_xy, ins_z:d - ins_z, :]
        pred = tf.squeeze(y_pred, axis=4)[:, ins_xy:h - ins_xy, ins_xy:w - ins_xy,
                                          ins_z:d - ins_z]
        conv = tf.squeeze(conv, axis=4)
        if upsample:
            conv = tf.image.resize(conv, [conv.shape[1] // 2, conv.shape[2] // 2])
        if mse:
            loss = tf.reduce_mean(tf.square(target - conv))
            reg = tf.reduce_mean(tf.square(conv - target - output_g))
        else:
            loss = tf.reduce_mean(tf.abs(target - conv))
            reg = tf.reduce_mean(tf.abs(conv - target - output_g))
        total = loss + reg
        ph, pw, pd = pred.shape[1], pred.shape[2], pred.shape[3]
        if tv_weight > 0:
            dy = pred[:, :ph - 1] - pred[:, 1:]
            dx = pred[:, :, :pw - 1] - pred[:, :, 1:]
            dz = (pred[:, :, :, :pd - 1] - pred[:, :, :, 1:]) if pd and pd > 1 else 0.0
            total = total + tv_weight * (tf.reduce_mean(tf.square(dx)) +
                                         tf.reduce_mean(tf.square(dy)) +
                                         tf.reduce_mean(tf.square(dz)))
        if hess_weight > 0:
            first = [pred[:, 1:] - pred[:, :ph - 1], pred[:, :, 1:] - pred[:, :, :pw - 1]]
            if pd and pd > 1:
                first.append(pred[:, :, :, 1:] - pred[:, :, :, :pd - 1])
            hess = 0.0
            for tv in first:
                hess = hess + tf.reduce_mean(tf.square(tv[:, 1:] - tv[:, :-1]))
                hess = hess + tf.reduce_mean(tf.square(tv[:, :, 1:] - tv[:, :, :-1]))
                if pd and pd > 1:
                    hess = hess + tf.reduce_mean(tf.square(tv[:, :, :, 1:] -
                                                           tv[:, :, :, :-1]))
            total = total + hess_weight * hess
        return total

    return psf_loss


# ── training ──────────────────────────────────────────────────────────────────

def _adam(lr: float):
    """Adam at ``lr``, betas 0.9/0.999 — the reference's optimizer minus its ``decay=1e-5``.

    Keras 3 removed the legacy ``decay`` argument. It was the small per-step ``1/(1+decay*t)``
    schedule; the ×0.5 step decay both trainers apply explicitly is the one that carries the
    learning-rate schedule, and it is kept in :func:`train_2d` / :func:`train_3d`.
    """
    import keras
    return keras.optimizers.Adam(learning_rate=float(lr), beta_1=0.9, beta_2=0.999)


def _set_lr(model, lr: float) -> None:
    model.optimizer.learning_rate.assign(float(lr))


def train_2d(planes: Sequence[np.ndarray], psf: np.ndarray, *, iterations: int,
             batch_size: int = 4, patch: int = 128, insert_xy: int = 16,
             upsample: bool = True, start_lr: float = 5e-5,
             lr_decay_factor: float = 0.5, lr_decay_every: int = 10000,
             hess_weight: float = 0.02, tv_weight: float = 0.0, mse: bool = False,
             denoise_weight: float = 0.5, background: float = 100.0,
             beta1: float = 1.0, beta2: float = 0.0, alpha: float = 1.0,
             seed: int = 0,
             progress: Optional[Callable[[int, int, float], None]] = None
             ) -> List[np.ndarray]:
    """Train a 2D ZS-DeconvNet on ``planes`` themselves — ``Train_ZSDeconvNet_2D.py``.

    Zero-shot: there is no ground truth anywhere in here. Each step draws a plane, splits it
    into two noise-independent copies by re-corruption (:func:`recorrupt`), cuts foreground
    patches, and trains the two-stage network with ``[MAE, psf_loss]`` weighted
    ``[mu, 1-mu]`` (``mu = denoise_weight = 0.5``). Stage I learns to denoise (its target is
    the second corrupted copy); stage II learns to deconvolve (its output is re-blurred by
    the PSF before comparison), which is the paper's Eq. 3.

    Both output heads get the SAME target patch, and that is correct rather than a shortcut:
    ``psf_loss`` internally blurs and downsamples stage II's output back onto the target's
    grid, so a single low-resolution target scores both heads.

    The learning rate is multiplied by ``lr_decay_factor`` every ``lr_decay_every`` steps.

    Returns the trained weights as a list of arrays (``model.get_weights()``) rather than the
    model, because the caller re-instantiates the graph at each inference tile shape.
    """
    if not len(planes):
        raise ValueError("zs-deconvnet: no planes to train on")
    rng = np.random.default_rng(int(seed))
    side = int(patch) + 2 * int(insert_xy)
    model = _arch("unet2d")[0]((side, side, 1), upsample_flag=bool(upsample),
                               insert_x=int(insert_xy), insert_y=int(insert_xy))
    model.compile(
        loss=["mean_squared_error" if mse else "mean_absolute_error",
              make_psf_loss_2d(psf, hess_weight, tv_weight, mse, bool(upsample),
                               int(insert_xy))],
        loss_weights=[float(denoise_weight), 1.0 - float(denoise_weight)],
        optimizer=_adam(start_lr))
    total = max(1, int(iterations))
    lr = float(start_lr)
    for step in range(total):
        plane = np.asarray(planes[int(rng.integers(0, len(planes)))], dtype=np.float32)
        # beta1/alpha are randomized +/-50% around their value exactly as DataAugmFor2D.m
        # does across its 100 re-corruptions per cell; beta2 gets the +/-20% band the
        # estimator hands back. Doing it per STEP is the same augmentation, streamed.
        inp, gt = recorrupt(plane, background,
                            float(beta1) * float(rng.uniform(0.5, 1.5)),
                            float(beta2) * float(rng.uniform(0.8, 1.2)),
                            float(alpha) * float(rng.uniform(0.5, 1.5)), rng)
        xb, yb = sample_patches_2d(inp, gt, int(batch_size), int(patch), rng)
        xb = np.pad(xb, ((0, 0), (insert_xy, insert_xy), (insert_xy, insert_xy)))
        xb = xb[..., None]
        yb = yb[..., None]
        logs = model.train_on_batch(xb, [yb, yb], return_dict=True)
        if (step + 1) % max(1, int(lr_decay_every)) == 0:
            lr *= float(lr_decay_factor)
            _set_lr(model, lr)
        if progress is not None:
            progress(step + 1, total, float(logs.get("loss", 0.0)))
    return model.get_weights()


def train_3d(volumes: Sequence[np.ndarray], psf: np.ndarray, *, iterations: int,
             arch: str = "rcan3d", batch_size: int = 3, patch: int = 64,
             patch_z: int = 13, insert_xy: int = 8, insert_z: int = 2,
             upsample: bool = False, start_lr: float = 1e-4,
             lr_decay_factor: float = 0.5, hess_weight: float = 0.1,
             tv_weight: float = 0.0, mse: bool = False, seed: int = 0,
             progress: Optional[Callable[[int, int, float], None]] = None
             ) -> List[np.ndarray]:
    """Train a 3D ZS-DeconvNet on ``volumes`` themselves — ``Train_ZSDeconvNet_3D.py``.

    Parameter-free augmentation: instead of re-corruption's noise model, the two training
    copies are the volume's two axial parities (:func:`axial_split`), so none of
    ``alpha``/``beta1``/``beta2`` exists in the 3D path at all — this is the paper's
    "totally parameter-free data augmentation strategy".

    **The GAR forward pass.** Each step also runs a frozen copy of the network
    (``g_copy``, built at ``2 * patch_z`` depth with ``upsample_flag=0``) over a volume in
    which the input and target parities are re-interleaved plane by plane. The difference
    between its consecutive output planes estimates the systematic half-z-step gap between
    the two parities, and both losses subtract it (``gamma = 1``). Weights are copied from
    the live model every step, which is what "frozen copy" means here — it contributes no
    gradient, only a target correction.

    The learning rate halves at 5000 and 7500 steps (the reference's two explicit
    milestones), scaled proportionally when ``iterations`` is smaller so a short run still
    gets its decay schedule instead of none at all.

    Returns ``model.get_weights()``.
    """
    if not len(volumes):
        raise ValueError("zs-deconvnet: no volumes to train on")
    builder, dim, levels = _arch(arch)
    if dim != "3D":
        raise ValueError(f"zs-deconvnet: {arch!r} is not a 3D architecture")
    rng = np.random.default_rng(int(seed))
    side = int(patch) + 2 * int(insert_xy)
    if levels and side % (2 ** levels):
        raise ValueError(
            f"zs-deconvnet: {arch!r} pools laterally {levels} times, so patch + 2*insert_xy "
            f"= {side} must be divisible by {2 ** levels}. Adjust the patch size "
            f"(64 + 2*8 = 80 is not; the reference uses the RCAN for 3D, which never pools).")
    depth = int(patch_z) + 2 * int(insert_z)
    model = builder((side, side, depth, 1), upsample_flag=bool(upsample),
                    insert_xy=int(insert_xy), insert_z=int(insert_z))
    gap = builder((side, side, 2 * int(patch_z) + 2 * int(insert_z), 1),
                  upsample_flag=False, insert_xy=int(insert_xy), insert_z=int(insert_z))
    model.compile(loss=[make_nbr2nbr_loss_3d(tv_weight, mse),
                        make_psf_loss_3d(psf, hess_weight, tv_weight, mse,
                                         bool(upsample), int(insert_z), int(insert_xy))],
                  optimizer=_adam(start_lr))
    total = max(1, int(iterations))
    # the reference's milestones are absolute (5000, 7500 of 10000); scale them so a short
    # interactive run still decays rather than training flat at the initial rate
    milestones = {max(1, int(round(total * 0.5))), max(1, int(round(total * 0.75)))}
    lr = float(start_lr)
    for step in range(total):
        vol = np.asarray(volumes[int(rng.integers(0, len(volumes)))], dtype=np.float32)
        xb, yb = sample_patches_3d(vol, int(batch_size), int(patch), int(patch_z), rng)
        # (n,Z,Y,X) -> the reference's (n,X,Y,Z,1)
        xin = np.transpose(xb, (0, 2, 3, 1))[..., None]
        gtv = np.transpose(yb, (0, 2, 3, 1))[..., None]
        xpad = np.pad(xin, ((0, 0), (insert_xy, insert_xy), (insert_xy, insert_xy),
                            (insert_z, insert_z), (0, 0)))
        # ── GAR: interleave target/input parities into one 2x-depth volume ──────
        n = xpad.shape[0]
        inter = np.zeros((n, side, side, 2 * int(patch_z) + 2 * int(insert_z), 1),
                         dtype=np.float32)
        for z in range(int(insert_z), 2 * int(patch_z) + int(insert_z), 2):
            inter[:, insert_xy:insert_xy + patch, insert_xy:insert_xy + patch, z, 0] = \
                gtv[:, :, :, (z - insert_z) // 2, 0]
            inter[:, :, :, z + 1, 0] = xpad[:, :, :, (z + insert_z) // 2, 0]
        gap.set_weights(model.get_weights())
        og = np.asarray(gap.predict(inter, verbose=0)[0])
        output_g = og[:, :, :, 1::2, :] - og[:, :, :, 0::2, :]
        target = np.concatenate((gtv, output_g), axis=4)
        logs = model.train_on_batch(xpad, [target, target], return_dict=True)
        if (step + 1) in milestones:
            lr *= float(lr_decay_factor)
            _set_lr(model, lr)
        if progress is not None:
            progress(step + 1, total, float(logs.get("loss", 0.0)))
    return model.get_weights()


# ── inference ─────────────────────────────────────────────────────────────────

def _forward(model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One forward pass, in GRAPH mode — ``(denoised, deconvolved)`` as numpy.

    ``predict_on_batch`` rather than an eager ``model(x)``, and the reason is memory rather
    than speed. Eager keeps every intermediate alive for as long as something references it,
    and the 3D RCAN is nothing but residual chains — 8 RCABs, each holding a conv, two
    LeakyReLUs, a channel-attention product and a skip addend, all at FULL tile resolution
    with 64 channels. At the authors' own 3D inference tile that is ~50 live tensors of
    2.25 GiB, and TensorFlow's ``mklcpu`` BFC allocator caps its CPU arena at **64 GiB**
    regardless of installed RAM, so it dies at ~58 GiB in use on a machine with 256 GiB free.
    ``predict_on_batch`` runs the compiled graph, whose memory planner reuses buffers, and the
    same tile then completes in 33 s.

    Graph and eager agree to ~1e-6 relative (float summation order), which is four orders of
    magnitude below the 1-count quantization of the reference's own uint16 output.
    """
    out = model.predict_on_batch(x)
    return np.asarray(out[0]), np.asarray(out[1])


def tile_plan(extent: int, window: int, overlap: int) -> Tuple[List[int], int, int]:
    """``(starts, window, overlap)`` for one axis — THE single source of truth for tiling.

    Returns the clamped window and overlap alongside the offsets precisely so that
    :func:`_seam` cannot be handed a different overlap than :func:`_tile_starts` used. It was
    two independent computations, and they disagreed whenever ``overlap >= window``: the
    stride floors at 1, but the seam still cut at half the *requested* overlap, which leaves
    an uncovered BAND in the middle of the output (extent 17, window 16, overlap 20 left
    ``[6,11)`` unwritten — a stripe of zeros through the result). Clamping the overlap to
    ``window - 1`` keeps the stride positive and makes the requested and actual overlap the
    same number again.
    """
    win = max(1, min(int(window), int(extent)))
    ov = max(0, min(int(overlap), win - 1))
    return _tile_starts(int(extent), win, ov), win, ov


def _tile_starts(extent: int, window: int, overlap: int) -> List[int]:
    """Tile start offsets — the ``rr_list``/``cc_list``/``zz_list`` construction shared by
    ``Infer_2D.py`` and ``Infer_3D.py``: a regular stride of ``window - overlap``, with the
    final tile pinned flush to the far edge so the last strip is never short.

    The flush-pinned last tile may overlap its predecessor by MORE than ``overlap``; that is
    the reference's behaviour and it is safe, because a wider overlap only means the later
    tile overwrites a few more samples that the earlier one had already answered validly.
    A NARROWER-than-assumed overlap is the unsafe direction, which :func:`tile_plan` rules
    out."""
    win = min(int(window), int(extent))
    step = max(1, win - int(overlap))
    starts = list(range(0, max(1, extent - win + 1), step))
    if not starts:
        starts = [0]
    if starts[-1] != extent - win:
        starts.append(max(0, extent - win))
    return starts


def _seam(start: int, window: int, extent: int, overlap: int) -> Tuple[int, int, int, int]:
    """Where one tile's contribution begins and ends — the fusion arithmetic of both Infer
    scripts.

    An interior seam is cut at the MIDDLE of the overlap (``ceil(o/2)`` in, ``floor(o/2)``
    off the far end), so each output sample comes from whichever tile saw the most context
    around it. A tile flush against an edge contributes all the way to that edge.

    Returns ``(dst_lo, dst_hi, src_lo, src_hi)``.
    """
    o = int(overlap)
    lo_cut = 0 if start == 0 else -(-o // 2)                       # ceil(o/2)
    hi_cut = 0 if start + window >= extent else o // 2             # floor(o/2)
    return (start + lo_cut, start + window - hi_cut, lo_cut, window - hi_cut)


def infer_2d(plane: np.ndarray, *, arch: str = "unet2d", weights_path: str = "",
             weights: Optional[List[np.ndarray]] = None, tile: int = 256,
             overlap: int = 20, upsample: bool = True, insert_xy: int = 16,
             norm_low: float = 3.0, tile_y: int = 0, tile_x: int = 0,
             progress: Optional[Callable[[int, int], None]] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Run a trained 2D ZS-DeconvNet over one plane — ``Infer_2D.py``.

    Pipeline, in order: clip negatives, percentile-normalize to ``[0,1]``, cut overlapping
    tiles, zero-pad each by :func:`net_padding`, predict both heads, crop stage II's output
    by ``insert * (1 + upsample)``, fuse on half-overlap seams, and percentile-normalize the
    two results with ``norm_low`` as the low cut.

    The input normalization is not cosmetic: the network was trained on ``[0,1]`` patches, so
    feeding raw counts would put every activation far outside the range it learned.

    Returns ``(denoised, deconvolved)`` — the deconvolved plane is ``2x`` larger on both
    lateral axes when ``upsample`` is set.
    """
    a = np.asarray(plane, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"infer_2d expects a 2D plane, got shape {a.shape}")
    a = prctile_norm(np.maximum(a, 0.0))
    h, w = a.shape
    _b, _d, levels = _arch(arch)
    # `tile_y`/`tile_x` exist for the parity bench, which must reproduce the reference's
    # per-axis tile COUNTS (its knob is `num_seg_window_*`, ours is a tile size); the node
    # passes the single square `tile`.
    win_y = net_window(h, int(tile_y) or int(tile), levels)
    win_x = net_window(w, int(tile_x) or int(tile), levels)
    ins_y = net_padding(win_y, insert_xy, levels)
    ins_x = net_padding(win_x, insert_xy, levels)
    up = 2 if upsample else 1
    model = get_model(arch, (win_y + 2 * ins_y, win_x + 2 * ins_x, 1),
                      upsample=bool(upsample), insert_xy=ins_y,
                      insert_x=ins_y, insert_y=ins_x,
                      weights_path=weights_path, weights=weights)
    ys, win_y, ov_y = tile_plan(h, win_y, overlap)
    xs, win_x, ov_x = tile_plan(w, win_x, overlap)
    den = np.zeros((h, w), dtype=np.float32)
    dec = np.zeros((h * up, w * up), dtype=np.float32)
    n_tiles = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            patch = a[y0:y0 + win_y, x0:x0 + win_x]
            padded = np.pad(patch, ((ins_y, ins_y), (ins_x, ins_x)))[None, ..., None]
            o_den, o_dec = _forward(model, padded)
            p_den = o_den[0, ..., 0]
            p_dec = o_dec[0, ..., 0]
            p_dec = p_dec[ins_y * up:(win_y + ins_y) * up,
                          ins_x * up:(win_x + ins_x) * up]
            dy0, dy1, sy0, sy1 = _seam(y0, win_y, h, ov_y)
            dx0, dx1, sx0, sx1 = _seam(x0, win_x, w, ov_x)
            den[dy0:dy1, dx0:dx1] = p_den[sy0:sy1, sx0:sx1]
            dec[dy0 * up:dy1 * up, dx0 * up:dx1 * up] = \
                p_dec[sy0 * up:sy1 * up, sx0 * up:sx1 * up]
            done += 1
            if progress is not None:
                progress(done, n_tiles)
    return (prctile_norm(den, norm_low, 100.0), prctile_norm(dec, norm_low, 100.0))


def infer_3d(vol: np.ndarray, *, arch: str = "rcan3d", weights_path: str = "",
             weights: Optional[List[np.ndarray]] = None, tile: int = 256,
             tile_z: int = 0, overlap: int = 20, overlap_z: int = 4,
             upsample: bool = False, insert_xy: int = 8, insert_z: int = 2,
             background: float = 100.0, norm_low: float = 3.0,
             damping_length: int = 0, damping_width: int = 1,
             tile_y: int = 0, tile_x: int = 0,
             progress: Optional[Callable[[int, int], None]] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Run a trained 3D ZS-DeconvNet over one ``(Z,Y,X)`` volume — ``Infer_3D.py``.

    Same shape of pipeline as :func:`infer_2d`, with three differences that are the 3D
    script's: a ``background`` offset is subtracted before normalizing (LLSM/confocal data
    carries a camera pedestal the network was trained without), tiles are cut in z as well,
    and the deconvolved result may be Fourier-damped (:func:`fourier_damp`) to suppress the
    sCMOS fixed-pattern stripe that the deconvolution stage amplifies.

    ``upsample`` scales the LATERAL axes only — ``UpSampling3D(size=(2,2,1))`` — so z is never
    resampled and ``z_step_um`` never changes.

    Returns ``(denoised, deconvolved)`` as ``(Z,Y,X)``.
    """
    v = np.asarray(vol, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"infer_3d expects a (Z,Y,X) volume, got shape {v.shape}")
    v = prctile_norm(np.maximum(v - float(background), 0.0))
    nz, h, w = v.shape
    _b, _d, levels = _arch(arch)
    win_z = min(int(tile_z) or nz, nz)          # z is never pooled: no parity constraint
    win_y = net_window(h, int(tile_y) or int(tile), levels)
    win_x = net_window(w, int(tile_x) or int(tile), levels)
    ins_y = net_padding(win_y, insert_xy, levels)
    ins_x = net_padding(win_x, insert_xy, levels)
    if ins_y != ins_x:
        # Unlike the 2D graph, the 3D builders bake ONE lateral margin for both axes
        # (`output1[..., insert_xy:h-insert_xy, insert_xy:w-insert_xy, ...]`), so unequal
        # margins cannot be represented. Only reachable with `unet3d`, the pooling variant.
        raise ValueError(
            f"zs-deconvnet: {arch!r} pools laterally, and tiles of {win_y}x{win_x} need "
            f"different padding margins ({ins_y} vs {ins_x}) which the 3D graph shares "
            f"between both axes. Use the `rcan3d` architecture (never pools, no constraint) "
            f"or a tile size that divides both lateral extents the same way.")
    up = 2 if upsample else 1
    model = get_model(arch, (win_y + 2 * ins_y, win_x + 2 * ins_x,
                             win_z + 2 * int(insert_z), 1),
                      upsample=bool(upsample), insert_xy=ins_y, insert_z=int(insert_z),
                      weights_path=weights_path, weights=weights)
    zs, win_z, ov_z = tile_plan(nz, win_z, overlap_z)
    ys, win_y, ov_y = tile_plan(h, win_y, overlap)
    xs, win_x, ov_x = tile_plan(w, win_x, overlap)
    den = np.zeros((nz, h, w), dtype=np.float32)
    dec = np.zeros((nz, h * up, w * up), dtype=np.float32)
    n_tiles = len(zs) * len(ys) * len(xs)
    done = 0
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                sub = v[z0:z0 + win_z, y0:y0 + win_y, x0:x0 + win_x]
                # (Z,Y,X) -> the model's (X,Y,Z)
                block = np.transpose(sub, (1, 2, 0))
                padded = np.pad(block, ((ins_y, ins_y), (ins_x, ins_x),
                                        (insert_z, insert_z)))[None, ..., None]
                o_den, o_dec = _forward(model, padded)
                p_den = np.transpose(o_den[0, ..., 0], (2, 0, 1))
                p_dec = o_dec[0, ..., 0]
                p_dec = p_dec[ins_y * up:(win_y + ins_y) * up,
                              ins_x * up:(win_x + ins_x) * up,
                              insert_z:win_z + insert_z]
                p_dec = np.transpose(p_dec, (2, 0, 1))
                dz0, dz1, sz0, sz1 = _seam(z0, win_z, nz, ov_z)
                dy0, dy1, sy0, sy1 = _seam(y0, win_y, h, ov_y)
                dx0, dx1, sx0, sx1 = _seam(x0, win_x, w, ov_x)
                den[dz0:dz1, dy0:dy1, dx0:dx1] = \
                    p_den[sz0:sz1, sy0:sy1, sx0:sx1]
                dec[dz0:dz1, dy0 * up:dy1 * up, dx0 * up:dx1 * up] = \
                    p_dec[sz0:sz1, sy0 * up:sy1 * up, sx0 * up:sx1 * up]
                done += 1
                if progress is not None:
                    progress(done, n_tiles)
    if int(damping_length) > 0:
        dec = fourier_damp(dec, int(damping_length), int(damping_width))
    return (prctile_norm(den, norm_low, 100.0), prctile_norm(dec, norm_low, 100.0))


def fourier_damp(vol: np.ndarray, length: int, width: int) -> np.ndarray:
    """Zero a vertical stripe through the lateral Fourier transform of every plane — the
    ``Fourier_damping`` post-process of ``Infer_3D.py``.

    sCMOS sensors have per-column gain non-uniformity ("fixed pattern noise"), which
    noise2noise-style schemes cannot remove because it is not random — it is identical in
    both training copies. The deconvolution stage then AMPLIFIES it into visible stripes.
    Their energy lies on one line through the frequency origin, so zeroing a band
    ``2*width+1`` wide and ``length`` tall on each side of DC removes the stripes while
    leaving the rest of the spectrum untouched.

    The reference notes this is a cosmetic fallback: pre-calibrating the camera removes the
    pattern properly. Because it deletes real spatial frequencies along one axis, it is off
    by default here (``length=0``).
    """
    a = np.asarray(vol, dtype=np.float32)
    if int(length) <= 0:
        return a
    f = np.fft.fftshift(np.fft.fft2(a, axes=(-2, -1)), axes=(-2, -1))
    half_x = f.shape[-1] // 2
    lo, hi = half_x - int(width), half_x + int(width) + 1
    n = min(int(length), f.shape[-2])
    f[..., :n, lo:hi] = 0
    f[..., f.shape[-2] - n:, lo:hi] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(f, axes=(-2, -1)),
                                axes=(-2, -1))).astype(np.float32)
