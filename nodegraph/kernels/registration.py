"""registration.py — self-contained image-registration (drift-correction) kernel.

PURPOSE
-------
Estimate per-frame spatial transforms that align each frame of a ``(T, H, W)``
time series onto a common anchor, then resample any channel through those
transforms ("register once on a reference channel, apply to all channels").
Supports pure sub-pixel translation (phase correlation), rigid/affine (ECC),
and large-motion feature alignment (ORB + RANSAC).

WHERE THE REAL MATH LIVES
-------------------------
IN-REPO. The algorithm is plain Python over numpy + established library
primitives — there is no external math package. It calls into:
  * scipy.ndimage       — sub-pixel resampling (``shift``, ``gaussian_filter``)
  * skimage.registration — ``phase_cross_correlation`` (sub-pixel translation)
  * skimage.feature/measure/transform — ORB, ``match_descriptors``, ``ransac``
  * cv2 (OpenCV)        — ``findTransformECC``, ``warpAffine``/``warpPerspective``
The windowing / high-pass / NCC scoring primitives (``_hann2d``, ``_highpass``,
``_ncc``) are the same ones the stitcher uses.

PROVENANCE (branch: Version-1.45)
---------------------------------
  * nd2studios/backend/registration/estimate.py   — the whole kernel body
  * nd2studios/backend/stitch/register.py          — helpers _highpass, _hann2d,
        _ncc (copied verbatim; the stitch module was NOT imported because its
        top level drags in StitchConfig / Dataset)
  * nd2studios/backend/analysis/manual_mask.py     — rasterize_shapes + its
        private helper _rasterize (copied verbatim; needed for the freeform
        "shapes" ROI branch of roi_to_mask)

Vendored verbatim; imports nothing from nd2studios; caller owns all prep
(no file I/O, no per-multipoint / per-timepoint looping, no crop / downsample /
registration-source selection / exclusion — the caller supplies a prepared
(T, H, W) single-channel array).

DROPPED UI / REGISTRY-ONLY MEMBERS
----------------------------------
  * From manual_mask.py: the ManualMaskPipeline class, its get_params()
    ParamSpec list, the @AnalysisPipeline.register decorator, and every helper
    that is NOT on the rasterize_shapes -> _rasterize compute path
    (generate_tiled_label_boxes, generate_tiled_mask, _normalise_frame_slot,
    _ordered_shapes, _voxel_counts_for_frame, shape_to_editable_polygon,
    expand_polygon_uniformly, _resample_closed_polygon). None of these are
    reachable from this kernel's entry points.
  * From register.py: everything except the three primitives above
    (_overlap_boxes, _pair_shift, _candidate_pairs, _solve_axis,
    refine_positions) — those are stitcher-only, off this kernel's path.
  * method.py / RegistrationResult were intentionally NOT vendored; callers use
    estimate_series / apply_series / apply_frame directly.

EDITS MADE (per the extraction rules; see registration.md "issues")
-------------------------------------------------------------------
  (a) Removed ``from nd2studios.backend.stitch.register import ...`` and
      ``from nd2studios.backend.analysis.manual_mask import rasterize_shapes``;
      both symbols are now defined locally in this file.
  No private-helper renames were needed (no name collisions between the three
  concatenated sources).

REFINEMENTS (2026-10-07 — ``scripts/registration_synthetic_bench.py`` is the evidence)
--------------------------------------------------------------------------------------
The kernel is no longer verbatim. Four synthetic worlds with exact ground truth (a star
field, a star volume, a slowly deforming textured mass, an affine-moving mass) measured
the recovered transform against the true motion, and these are the changes they forced:
  * ``ecc_align`` seeded ECC at ``+shift`` where the warp convention needs ``-shift`` — the
    mirror image of the answer. Invisible on a big smooth texture (ECC still pulls in),
    catastrophic on a sparse field (9–21 px error vs 0.02 px). Fixed; see its docstring.
  * phase-normalized correlation (skimage ``normalization="phase"``) was the default. On
    band-limited, noisy microscopy images plain cross-correlation of the band-passed images
    is 3–15× more precise (0.18→0.06 px on sparse stars, 0.33→0.03 px on a textured mass),
    and it holds under bleaching and vignetting once the high-pass is on. Phase whitening
    also CANCELS any linear pre-filter applied to both images, so ``highpass_sigma`` had
    been a dead parameter. ``correlation="cross"`` is the default; ``"phase"`` remains.
  * a matched-filter ``lowpass_sigma`` before the high-pass: at peak-SNR 8 a sparse field
    goes from 1.9 px (and 104 px with the old σ=2 high-pass alone) to 0.1 px.
  * ECC is seeded with the feature (ORB+RANSAC) warp when that fit is trusted, so a
    rotation beyond ECC's ~10° capture range registers in ``first``/``mean``/``template``
    mode (0.02 px up to 75° on the bench; previously ``first`` mode failed past ~10°), and
    a converged warp is sanity-checked (singular values, translation) — a failed ECC is
    always gated instead of applied.
  * n-D: ``estimate_translation`` / ``estimate_series`` / ``apply_volume`` accept a
    ``(T, Z, H, W)`` series and return a 3-component ``(dz, dy, dx)`` shift, so a z-stack's
    axial drift is corrected instead of ignored.
  * ``estimate_from_landmarks`` fits user-clicked correspondences, says which transform
    family the motion needs (and when NONE does), and seeds the series estimate.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage.draw import ellipse as sk_ellipse
from skimage.draw import polygon as sk_polygon


# ============================================================================
# Vendored verbatim from nd2studios/backend/stitch/register.py
# (helpers _highpass, _hann2d, _ncc — copied byte-for-byte)
# ============================================================================
def _highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    img = img.astype(np.float32)
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        img = img - gaussian_filter(img, sigma=sigma)
    return img


def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.outer(wy, wx).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    da = float(np.sqrt((a * a).sum())); db = float(np.sqrt((b * b).sum()))
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return float((a * b).sum() / (da * db))


# ============================================================================
# 2026-10-07 helpers (not vendored) — band-pass, n-D window, warp sanity
# ============================================================================
def _hann_nd(shape: Sequence[int]) -> np.ndarray:
    """Separable Hann window over every axis of ``shape`` (2-D or 3-D)."""
    out = None
    for i, n in enumerate(shape):
        w = (np.hanning(n) if n > 1 else np.ones(1)).astype(np.float32)
        view = [1] * len(shape)
        view[i] = int(n)
        w = w.reshape(view)
        out = w if out is None else out * w
    return out


def _bandpass(img: np.ndarray, lowpass_sigma: float, highpass_sigma: float) -> np.ndarray:
    """Matched-filter smoothing (``lowpass_sigma``, 0 = off) then background removal
    (``highpass_sigma``, 0 = off): the band the correlation runs on. The low-pass is what
    rescues a sparse, noisy field — a Gaussian of about the PSF width is the matched filter
    for point-like features, and without it the high-pass whitens the noise floor into the
    correlation peak (bench: 104 px → 0.1 px at peak-SNR 8 with σ_lp = 1–1.5)."""
    a = np.asarray(img, dtype=np.float32)
    if lowpass_sigma and lowpass_sigma > 0:
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, sigma=float(lowpass_sigma))
    return _highpass(a, highpass_sigma)


def _unit_range(img: np.ndarray) -> np.ndarray:
    """Percentile-stretch to ``[0, 1]`` float32. skimage's FAST corner threshold is an
    absolute intensity difference, so ORB on raw 12/16-bit counts treats every noisy pixel
    as a corner; on a ``[0, 1]`` image the library default means what it was tuned for."""
    a = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(a, (0.5, 99.5))
    return np.clip((a - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0).astype(np.float32)


def _warp_is_sane(warp: np.ndarray, shape: Sequence[int],
                  scale_range: Tuple[float, float] = (0.5, 2.0)) -> bool:
    """Is this 2×3 warp a plausible specimen motion? Finite, every singular value of its
    linear part within ``scale_range`` (no collapse, no 2× stretch), translation inside the
    frame. A diverged ECC can report a confident warp with ``det`` 1.05 that lands the
    field 1600 px away — this is the fence that stops it from being applied."""
    w = np.asarray(warp, dtype=np.float64)
    if w.shape[0] < 2 or w.shape[1] < 3 or not np.all(np.isfinite(w)):
        return False
    sv = np.linalg.svd(w[:2, :2], compute_uv=False)
    if sv.min() < scale_range[0] or sv.max() > scale_range[1]:
        return False
    H, W = int(shape[-2]), int(shape[-1])
    return bool(abs(w[0, 2]) <= W and abs(w[1, 2]) <= H)


# ============================================================================
# Vendored verbatim from nd2studios/backend/analysis/manual_mask.py
# (rasterize_shapes + its private helper _rasterize — copied byte-for-byte;
#  needed for the freeform "shapes" ROI branch of roi_to_mask)
# ============================================================================
def rasterize_shapes(
    frame: np.ndarray,
    shapes: List[Dict[str, Any]],
    H: int,
    W: int,
    label_offset: int = 0,
) -> None:
    """Paint each shape into ``frame`` with label_id (offset+1)..(offset+N) in place.

    ``label_offset`` lets callers reserve lower IDs for tiled masks so drawn
    shapes never collide with tiled-strip label IDs.
    """
    for rel_id, shape in enumerate(shapes, start=1):
        _rasterize(frame, shape, label_offset + rel_id, H, W)


def _rasterize(frame: np.ndarray, shape: Dict[str, Any],
               label_id: int, H: int, W: int) -> None:
    """Paint a single shape into a frame at the given label id (in place)."""
    s_type = shape.get("type")
    raw_verts = shape.get("vertices") or []
    if not raw_verts:
        return
    verts = np.asarray(raw_verts, dtype=float)

    if s_type == "rect" and verts.shape[0] == 2:
        y0, y1 = sorted([verts[0, 0], verts[1, 0]])
        x0, x1 = sorted([verts[0, 1], verts[1, 1]])
        iy0 = max(0, int(round(y0)))
        ix0 = max(0, int(round(x0)))
        iy1 = min(H, int(round(y1)) + 1)
        ix1 = min(W, int(round(x1)) + 1)
        if iy1 > iy0 and ix1 > ix0:
            frame[iy0:iy1, ix0:ix1] = label_id
        return

    if s_type == "ellipse" and verts.shape[0] == 2:
        cy = (verts[0, 0] + verts[1, 0]) / 2.0
        cx = (verts[0, 1] + verts[1, 1]) / 2.0
        ry = abs(verts[1, 0] - verts[0, 0]) / 2.0
        rx = abs(verts[1, 1] - verts[0, 1]) / 2.0
        if ry <= 0 or rx <= 0:
            return
        rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(H, W))
        frame[rr, cc] = label_id
        return

    if s_type == "polygon" and verts.shape[0] >= 3:
        rr, cc = sk_polygon(verts[:, 0], verts[:, 1], shape=(H, W))
        frame[rr, cc] = label_id
        return


# ============================================================================
# Vendored from nd2studios/backend/registration/estimate.py (the two nd2studios
# imports removed — those symbols are defined above) and refined 2026-10-07; see
# the module docstring for what changed and why.
# ============================================================================
# Reference-frame selection modes. ``template`` is a robust two-pass anchor (see
# :func:`estimate_series`) recommended for long / photobleaching series.
REFERENCE_MODES: Tuple[str, ...] = ("first", "previous", "mean", "template")
# Transform models (increasing DOF). ``feature`` is ORB+RANSAC (large motion).
MODELS: Tuple[str, ...] = ("translation", "euclidean", "affine", "feature")


# ── dtype helpers ────────────────────────────────────────────────────────────

def _cast_like(arr_float: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float result back to ``dtype``, clipping integer ranges to avoid
    wrap-around from interpolation over/undershoot."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.round(arr_float), info.min, info.max).astype(dtype)
    return arr_float.astype(dtype, copy=False)


# ── translation (phase correlation) ──────────────────────────────────────────

def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    window: bool = True,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    lowpass_sigma: float = 0.0,
    correlation: str = "cross",
) -> Tuple[np.ndarray, float]:
    """Estimate the sub-pixel translation aligning ``moving`` onto ``reference``.

    Works on a 2-D plane ``(H, W)`` or a 3-D volume ``(Z, H, W)`` alike: the shift has one
    component per axis, ``(row, col)`` or ``(z, row, col)``, in the :func:`apply_shift`
    convention (``reference ≈ apply_shift(moving, shift)``).

    Pre-filter: :func:`_bandpass` (``lowpass_sigma`` then ``highpass_sigma``) and a Hann
    window against spectral leakage. ``correlation="cross"`` correlates the band-passed
    images as they are; ``"phase"`` whitens the spectrum first (skimage's
    ``normalization="phase"``). Cross is the default since 2026-10-07: on band-limited,
    noisy images whitening amplifies the noise-dominated high frequencies into the peak
    (3–15× worse on the bench), and because whitening cancels any linear filter applied to
    both images it also made the high-pass a no-op. ``disambiguate=True`` resolves the
    modulo-image wrap so large shifts are recovered without aliasing. Returns
    ``(shift, ncc)`` where ``ncc`` is the normalized cross-correlation of the aligned
    overlap — a confidence in ``[-1, 1]`` for gating near-blank / failed frames.

    Region of interest (estimate on a sub-region, apply full-frame):
    - ``bbox=(y0, y1, x0, x1)`` crops both images (the last two axes) first — **subpixel**,
      used for a rectangular ROI; a pure shift is invariant to the crop origin.
    - ``mask`` (bool ``(H,W)``, broadcast over z for a volume) restricts a freeform region
      via masked phase correlation (Padfield); masked correlation is **integer-pixel** only.
    """
    from scipy.ndimage import shift as nd_shift
    from skimage.registration import phase_cross_correlation

    ref = np.asarray(reference)
    mov = np.asarray(moving)
    if bbox is not None:
        y0, y1, x0, x1 = (int(v) for v in bbox)
        ref = ref[..., y0:y1, x0:x1]
        mov = mov[..., y0:y1, x0:x1]
        mask = None  # region already isolated by the crop
    fa = _bandpass(ref, lowpass_sigma, highpass_sigma)
    fb = _bandpass(mov, lowpass_sigma, highpass_sigma)
    if window:
        win = _hann_nd(fa.shape)
        fa_w, fb_w = fa * win, fb * win
    else:
        fa_w, fb_w = fa, fb
    if fa_w.std() < 1e-6 or fb_w.std() < 1e-6:
        # No texture to correlate — a peak would be spurious.
        return np.zeros(ref.ndim, dtype=np.float64), 0.0
    m = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim < fa.ndim:
            m = np.broadcast_to(m, fa.shape)
    if m is not None:
        out = phase_cross_correlation(fa, fb, reference_mask=m, moving_mask=m)
        shift = np.asarray(out[0] if isinstance(out, tuple) else out,
                           dtype=np.float64)
    else:
        shift, _err, _phase = phase_cross_correlation(
            fa_w, fb_w, upsample_factor=max(1, int(upsample)),
            normalization=("phase" if correlation == "phase" else None),
            disambiguate=True,
        )
        shift = np.asarray(shift, dtype=np.float64)
    b_aligned = nd_shift(fb, shift=shift, order=1, mode="constant", cval=0.0)
    if m is not None:
        return shift, (float(_ncc(fa[m], b_aligned[m])) if m.any() else 0.0)
    return shift, float(_ncc(fa, b_aligned))


def apply_shift(image: np.ndarray, shift: np.ndarray, order: int = 1) -> np.ndarray:
    """Resample ``image`` by ``shift`` ``(row, col)``, preserving dtype.

    ``order`` — 0 nearest, 1 bilinear (safe default for uint16), 3 cubic.
    """
    from scipy.ndimage import shift as nd_shift

    arr = np.asarray(image)
    moved = nd_shift(arr.astype(np.float32), shift=np.asarray(shift, dtype=np.float64),
                     order=int(order), mode="constant", cval=0.0)
    return _cast_like(moved, arr.dtype)


# ── rigid / affine (ECC) ──────────────────────────────────────────────────────

def _cv_motion(model: str):
    import cv2
    return {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }[model]


def ecc_align(
    reference: np.ndarray,
    moving: np.ndarray,
    model: str = "euclidean",
    init_shift: Optional[np.ndarray] = None,
    iters: int = 200,
    eps: float = 1e-6,
    gauss: int = 5,
    interp_order: int = 1,
    mask: Optional[np.ndarray] = None,
    init_warp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Align ``moving`` onto ``reference`` with ECC (subpixel, brightness/contrast
    invariant). ``model`` ∈ {translation, euclidean, affine, homography}.

    Seeding: ``init_warp`` — a full 2×3 reference→moving warp from
    :func:`estimate_features` or :func:`estimate_from_landmarks` — wins over
    ``init_shift``, a phase-correlation translation ``(row, col)`` in the
    :func:`apply_shift` convention. ``mask`` (bool ``(H,W)``) restricts the estimate via
    ECC's ``inputMask`` — kept in full-frame coordinates (not cropped) so the rotation
    centre stays correct.

    **Sign of the seed (fixed 2026-10-07).** The warp maps reference → moving: under
    ``WARP_INVERSE_MAP`` the aligned image samples ``moving`` at ``p + d``, where ``d`` is
    how far the content MOVED. ``init_shift`` is the shift that moves the content BACK
    (``reference ≈ apply_shift(moving, shift)``), so the seed translation is
    ``-init_shift``. The vendored code wrote ``+init_shift`` and started ECC at the mirror
    image of the answer, ``2·|shift|`` away. A large smooth texture still pulled it in,
    which is why the bug was invisible on textured specimens; a sparse field did not, and
    since the fallback on non-convergence is the seed itself, the returned warp was wrong by
    exactly twice the drift (bench: 9–21 px on star fields, 0.02 px with the sign right).

    Returns ``(warp_matrix, cc, aligned)``. On non-convergence (``cv2.error``), or when the
    converged warp is not a plausible motion (:func:`_warp_is_sane`), returns the SEED warp,
    ``cc=0.0`` and the *unaligned* ``moving`` — callers gate on ``cc``.
    """
    import cv2

    motion = _cv_motion(model)
    ref = np.asarray(reference).astype(np.float32)
    mov = np.asarray(moving).astype(np.float32)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
    if init_warp is not None:
        w0 = np.asarray(init_warp, dtype=np.float32)
        warp[:2, :] = w0[:2, :3]
    elif init_shift is not None:
        warp[0, 2] = -float(init_shift[1])      # x: minus the apply_shift col component
        warp[1, 2] = -float(init_shift[0])      # y: minus the apply_shift row component
    seed = warp.copy()

    input_mask = None
    if mask is not None:
        input_mask = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    g = int(gauss) | 1  # gaussFiltSize must be odd
    try:
        cc, warp = cv2.findTransformECC(ref, mov, warp, motion, criteria, input_mask, g)
    except cv2.error:
        return seed, 0.0, np.asarray(moving)
    if not (np.isfinite(cc) and _warp_is_sane(warp, ref.shape)):
        return seed, 0.0, np.asarray(moving)

    aligned = apply_warp(moving, warp, motion=motion, output_shape=ref.shape,
                         interp_order=interp_order)
    return warp, float(cc), aligned


def apply_warp(
    image: np.ndarray,
    warp_matrix: np.ndarray,
    motion: Optional[int] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    interp_order: int = 1,
) -> np.ndarray:
    """Resample ``image`` through an ECC ``warp_matrix`` (reference→moving map),
    using ``WARP_INVERSE_MAP``. Preserves dtype. ``motion`` selects affine vs
    perspective (inferred from the matrix shape if ``None``)."""
    import cv2

    arr = np.asarray(image)
    h, w = output_shape if output_shape is not None else arr.shape[:2]
    is_homography = (motion == cv2.MOTION_HOMOGRAPHY if motion is not None
                     else warp_matrix.shape[0] == 3)
    interp = cv2.INTER_LINEAR if int(interp_order) >= 1 else cv2.INTER_NEAREST
    flags = cv2.WARP_INVERSE_MAP | interp
    src = arr.astype(np.float32)
    if is_homography:
        out = cv2.warpPerspective(src, warp_matrix, (int(w), int(h)), flags=flags,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        out = cv2.warpAffine(src, warp_matrix, (int(w), int(h)), flags=flags,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _cast_like(out, arr.dtype)


# ── (T, H, W) series driver ───────────────────────────────────────────────────

def stabilize(
    volume: np.ndarray,
    model: str = "translation",
    reference: str = "previous",
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    interp_order: int = 1,
    min_confidence: float = 0.0,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stabilize a ``(T, H, W)`` series onto a reference frame.

    Args:
        volume: ``(T, H, W)`` series (single channel).
        model: ``translation`` (phase correlation) | ``euclidean`` | ``affine``
            (ECC, seeded from a phase-correlation translation).
        reference: ``first`` (register every frame to frame 0), ``previous``
            (cumulative — each frame to the one before), or ``mean`` (to the
            series mean). Frame 0 is always the fixed anchor.
        upsample: phase-correlation sub-pixel factor (accuracy ≈ 1/upsample px).
        highpass_sigma: band-pass whitening sigma before correlation (0 = off).
        interp_order: resampling order (0 nearest, 1 bilinear, 3 cubic).
        min_confidence: frames whose registration confidence (NCC / ECC cc) is
            below this fall back to identity (guards spurious peaks on near-blank
            frames). 0 = never gate.

    Returns:
        ``(aligned (T,H,W) same dtype, shifts (T,2) row/col, confidence (T,))``.
        For non-translation models ``shifts`` holds the warp translation part.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"stabilize expects a (T,H,W) array, got shape {vol.shape}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    if reference not in REFERENCE_MODES:
        raise ValueError(f"unknown reference {reference!r}; expected {REFERENCE_MODES}")

    T = int(vol.shape[0])
    out = np.empty_like(vol)
    shifts = np.zeros((T, 2), dtype=np.float64)
    confidence = np.ones(T, dtype=np.float64)
    cum_shift = np.zeros(2, dtype=np.float64)

    # `first` / `previous` anchor on frame 0 (left untouched). `mean` has no
    # natural anchor, so *every* frame — including frame 0 — is registered to the
    # series mean, otherwise frame 0 would sit at a constant offset from the rest.
    if reference == "mean":
        mean_ref = vol.mean(axis=0)
        start = 0
    else:
        mean_ref = None
        out[0] = vol[0]
        start = 1

    for t in range(start, T):
        if cancelled_cb is not None and cancelled_cb():
            # Leave the remaining (unprocessed) frames as-is.
            out[t:] = vol[t:]
            break

        if reference == "first":
            ref = vol[0]
        elif reference == "mean":
            ref = mean_ref
        else:  # previous
            ref = vol[t - 1]

        if model == "translation":
            sh, conf = estimate_translation(ref, vol[t], upsample=upsample,
                                            highpass_sigma=highpass_sigma)
            if conf < float(min_confidence):
                sh = np.zeros(2, dtype=np.float64)  # gate spurious peak
            if reference == "previous":
                cum_shift = cum_shift + sh
                eff = cum_shift.copy()
            else:
                eff = sh
            out[t] = apply_shift(vol[t], eff, order=interp_order)
            shifts[t] = eff
            confidence[t] = conf
        else:
            # ECC path. In `previous` mode reference the already-stabilized prior
            # output so any model accumulates without composing warp matrices.
            ecc_ref = out[t - 1] if reference == "previous" else ref
            seed, _ = estimate_translation(ecc_ref, vol[t], upsample=upsample,
                                           highpass_sigma=highpass_sigma)
            warp, cc, aligned = ecc_align(ecc_ref, vol[t], model=model,
                                          init_shift=seed, interp_order=interp_order)
            if cc < float(min_confidence):
                aligned = vol[t]  # gate: leave frame unaligned
                warp = np.eye(2, 3, dtype=np.float32)
            out[t] = aligned
            shifts[t] = (float(warp[1, 2]), float(warp[0, 2]))  # (row, col)
            confidence[t] = cc

        if progress_cb is not None:
            progress_cb(int(100 * t / max(T - 1, 1)))

    return out, shifts, confidence


# ── register once, apply to all (cross-channel) ───────────────────────────────

def _homog(mat: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine to a 3x3 homogeneous matrix."""
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = np.asarray(mat, dtype=np.float64)
    return h


def common_translation_crop(
    shifts: np.ndarray, shape: Tuple[int, int], inset_edges: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """Largest axis-aligned rectangle that is real (non-padded) data in **every**
    frame after :func:`apply_shift`, for the pure-translation model.

    ``apply_shift`` moves content by ``shift=(dy,dx)`` and zero-pads the vacated
    border, so frame ``t``'s valid region is ``[max(0,dy):H+min(0,dy),
    max(0,dx):W+min(0,dx)]``. The intersection over all frames (pass every series'
    shifts stacked as ``(N,2)`` row/col) is the common region — cropping to it makes
    all frames equal-size, recentred, and border-free.

    Returns ``(y0, y1, x0, x1)`` (row/col half-open), or ``None`` if the drift
    leaves no common overlap. ``inset_edges`` trims 1 px off each padded side to
    avoid bilinear edge fuzz. Translation only (rotation/affine footprints are not
    axis-aligned rectangles).
    """
    s = np.asarray(shifts, dtype=np.float64).reshape(-1, 2)
    H, W = int(shape[0]), int(shape[1])
    if s.size == 0:
        return (0, H, 0, W)
    dy, dx = s[:, 0], s[:, 1]
    y0 = int(np.ceil(max(0.0, float(dy.max()))))
    y1 = int(np.floor(min(float(H), float(H) + float(dy.min()))))
    x0 = int(np.ceil(max(0.0, float(dx.max()))))
    x1 = int(np.floor(min(float(W), float(W) + float(dx.min()))))
    if inset_edges:
        if float(dy.max()) > 0.0:
            y0 += 1
        if float(dy.min()) < 0.0:
            y1 -= 1
        if float(dx.max()) > 0.0:
            x0 += 1
        if float(dx.min()) < 0.0:
            x1 -= 1
    y0, y1 = max(0, min(y0, H)), max(0, min(y1, H))
    x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


def _normalize_series(vol: np.ndarray, normalize: str) -> np.ndarray:
    """Per-frame intensity normalization for *estimation only* (geometry is
    unaffected). ``"zscore"`` counters photobleaching / intensity decay so late,
    dim frames still correlate; ``"none"`` passes the series through."""
    if normalize == "zscore":
        out = np.empty(vol.shape, dtype=np.float32)
        for t in range(vol.shape[0]):
            f = vol[t].astype(np.float32)
            out[t] = (f - float(f.mean())) / (float(f.std()) + 1e-6)
        return out
    return vol


def roi_to_mask(
    roi: Optional[Dict[str, Any]], shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Turn a serializable ROI spec into ``(mask (H,W) bool | None, bbox
    (y0,y1,x0,x1) | None)``.

    - ``{"kind":"rect","x","y","w","h"}`` → a box mask **and** a bbox (the bbox
      drives the subpixel crop for translation; the mask drives ECC/feature).
    - ``{"kind":"shapes","shapes":[{"type","vertices":[[y,x]…]}…]}`` → a rasterized
      mask (no bbox; freeform uses the masked / ``inputMask`` paths).
    - ``{"kind":"mask","mask":bool (H,W)}`` → that mask as given (no bbox) — for a caller
      that already rasterized a richer shape vocabulary (circle / brush / cut ops).
    - falsy / unknown → ``(None, None)`` (whole frame).
    """
    if not roi:
        return None, None
    H, W = int(shape[0]), int(shape[1])
    kind = roi.get("kind")
    if kind == "mask":
        m = np.asarray(roi.get("mask"), dtype=bool)
        return (m, None) if (m.shape == (H, W) and m.any()) else (None, None)
    if kind == "rect":
        x, y = int(roi.get("x", 0)), int(roi.get("y", 0))
        w, h = int(roi.get("w", 0)), int(roi.get("h", 0))
        y0, y1 = max(0, y), min(H, y + h)
        x0, x1 = max(0, x), min(W, x + w)
        if y1 <= y0 or x1 <= x0:
            return None, None
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask, (y0, y1, x0, x1)
    if kind == "shapes":
        # rasterize_shapes is vendored above in this file (was a lazy
        # `from nd2studios.backend.analysis.manual_mask import rasterize_shapes`).
        frame = np.zeros((H, W), dtype=np.int32)
        rasterize_shapes(frame, list(roi.get("shapes", [])), H, W)
        mask = frame > 0
        return (mask, None) if mask.any() else (None, None)
    return None, None


def _build_template(
    est_vol: np.ndarray, upsample: int, highpass_sigma: float,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    lowpass_sigma: float = 0.0,
    correlation: str = "cross",
) -> np.ndarray:
    """Two-pass reference: roughly stabilize the series with a cheap
    translation-``previous`` pass, then average the stabilized frames into a
    single template. Registering every frame to this template (in
    :func:`estimate_series`) avoids both cumulative-error drift and appearance
    divergence from a single (possibly bleached / atypical) anchor frame."""
    T = int(est_vol.shape[0])
    if T == 0:
        return np.zeros(est_vol.shape[1:], dtype=np.float32)
    cum = np.zeros(2, dtype=np.float64)
    rough = np.zeros((T, 2), dtype=np.float64)
    for t in range(1, T):
        if cancelled_cb is not None and cancelled_cb():
            break
        sh, _ = estimate_translation(
            est_vol[t - 1], est_vol[t], upsample=upsample,
            highpass_sigma=highpass_sigma,
            mask=(mask if bbox is None else None), bbox=bbox,
            lowpass_sigma=lowpass_sigma, correlation=correlation)
        cum = cum + sh
        rough[t] = cum
    acc = np.zeros(est_vol.shape[1:], dtype=np.float64)
    for t in range(T):
        acc += apply_shift(est_vol[t].astype(np.float32), rough[t], order=1)
    return (acc / max(T, 1)).astype(np.float32)


def estimate_features(
    reference: np.ndarray,
    moving: np.ndarray,
    transform: str = "affine",
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    n_keypoints: int = 800,
    min_inliers: int = 8,
    residual_threshold: float = 2.0,
) -> Tuple[np.ndarray, float]:
    """Feature-based alignment: ORB keypoints + descriptor matching + RANSAC fit of
    a Euclidean / Similarity / Affine model. Handles large displacement / rotation
    / scale that phase correlation and ECC cannot.

    Returns ``(warp 2x3, confidence)`` where the warp maps **reference → moving**
    (the same convention :func:`apply_warp` expects under ``WARP_INVERSE_MAP``) and
    confidence is the RANSAC inlier fraction. On too few keypoints / matches /
    inliers returns ``(eye(2,3), 0.0)`` so the caller falls back to identity — the
    graceful failure mode on near-blank / sparse fields. ``bbox`` crops keypoint
    detection to a rectangular ROI; ``mask`` keeps only matches inside a freeform ROI.
    """
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import (
        AffineTransform, EuclideanTransform, SimilarityTransform,
    )

    tclass = {"euclidean": EuclideanTransform, "similarity": SimilarityTransform,
              "affine": AffineTransform}.get(transform, AffineTransform)
    ref = np.asarray(reference).astype(np.float32)
    mov = np.asarray(moving).astype(np.float32)
    identity = np.eye(2, 3, dtype=np.float32)

    if bbox is not None:
        y0, y1, x0, x1 = (int(v) for v in bbox)
        rr, mm = ref[y0:y1, x0:x1], mov[y0:y1, x0:x1]
        off = np.array([y0, x0], dtype=np.float64)
    else:
        rr, mm, off = ref, mov, np.zeros(2, dtype=np.float64)
    # FAST's threshold is an absolute intensity step (skimage default 0.08 on a [0,1]
    # image); on raw camera counts it is ~0 and every noisy pixel is a corner.
    rr, mm = _unit_range(rr), _unit_range(mm)

    def _kp(img):
        orb = ORB(n_keypoints=int(n_keypoints), fast_threshold=0.05)
        try:
            orb.detect_and_extract(img)
        except Exception:  # noqa: BLE001 — blank / too-small patch
            return None, None
        return orb.keypoints, orb.descriptors

    kp_r, des_r = _kp(rr)
    kp_m, des_m = _kp(mm)
    if des_r is None or des_m is None or len(kp_r) < 3 or len(kp_m) < 3:
        return identity, 0.0
    matches = match_descriptors(des_r, des_m, cross_check=True)
    if len(matches) < max(3, int(min_inliers)):
        return identity, 0.0
    # keypoints are (row, col); to full-frame (x, y) = (col, row).
    src = (kp_r[matches[:, 0]] + off)[:, ::-1]
    dst = (kp_m[matches[:, 1]] + off)[:, ::-1]
    if mask is not None and bbox is None:
        m = np.asarray(mask, dtype=bool)
        keep = [i for i, (x, y) in enumerate(src)
                if 0 <= int(round(y)) < m.shape[0] and 0 <= int(round(x)) < m.shape[1]
                and m[int(round(y)), int(round(x))]]
        if len(keep) < max(3, int(min_inliers)):
            return identity, 0.0
        src, dst = src[keep], dst[keep]
    try:
        model, inliers = ransac(
            (src, dst), tclass, min_samples=3,
            residual_threshold=float(residual_threshold), max_trials=2000)
    except Exception:  # noqa: BLE001
        return identity, 0.0
    if model is None or inliers is None or int(np.sum(inliers)) < int(min_inliers):
        return identity, 0.0
    # model maps src(reference, x,y) → dst(moving, x,y): exactly reference→moving.
    warp = np.asarray(model.params, dtype=np.float32)[:2, :]
    return warp, float(np.sum(inliers)) / max(1, int(inliers.size))


def _ecc_seeded(
    ref_img: np.ndarray, cur: np.ndarray, model: str, *,
    seed_warp: Optional[np.ndarray], mask, bbox, tmask, upsample: int,
    highpass_sigma: float, lowpass_sigma: float, correlation: str,
    min_inliers: int, seed_from_features: bool,
) -> Tuple[np.ndarray, float]:
    """One ECC estimate from the best available seed.

    Order of trust: a caller-supplied ``seed_warp`` (landmarks), else the ORB+RANSAC warp
    when at least half the cross-checked matches agree on it (``conf ≥ 0.5`` — a
    geometrically verified vote, which is why it outranks the translation seed), else the
    phase-correlation translation. Whichever seed ECC fails from, the other is tried. ECC
    from a translation seed converges to a wrong local optimum — with a HIGH cc — once the
    true rotation passes ~10°, so cc alone cannot pick the seed; the feature vote can.
    Returns ``(warp, cc)``; ``cc == 0`` means every attempt failed (callers gate it)."""
    if seed_warp is None and seed_from_features:
        fw, fconf = estimate_features(
            ref_img, cur, transform=("euclidean" if model == "euclidean" else "affine"),
            mask=mask, bbox=bbox, min_inliers=min_inliers)
        if fconf >= 0.5 and _warp_is_sane(fw, ref_img.shape):
            seed_warp = fw
    if seed_warp is not None:
        warp, cc, _ = ecc_align(ref_img, cur, model=model, init_warp=seed_warp, mask=mask)
        if cc > 0.0:
            return warp, cc
    sh, _ = estimate_translation(
        ref_img, cur, upsample=upsample, highpass_sigma=highpass_sigma, mask=tmask,
        bbox=bbox, lowpass_sigma=lowpass_sigma, correlation=correlation)
    warp, cc, _ = ecc_align(ref_img, cur, model=model, init_shift=sh, mask=mask)
    if cc > 0.0 or seed_warp is None:
        return warp, cc
    return np.asarray(seed_warp, dtype=np.float32), 0.0


def estimate_series(
    series: np.ndarray,
    model: str = "translation",
    reference: str = "previous",
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    min_confidence: float = 0.0,
    normalize: str = "none",
    roi: Optional[Dict[str, Any]] = None,
    feature_transform: str = "affine",
    min_inliers: int = 8,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    lowpass_sigma: float = 0.0,
    correlation: str = "cross",
    landmarks: Optional[Dict[str, Any]] = None,
    seed_from_features: bool = True,
    click_sigma: float = 1.0,
) -> Dict[str, Any]:
    """Estimate per-frame **effective** transforms aligning each frame of ``series``
    onto the anchor, *without applying them*.

    ``series`` is ``(T, H, W)`` — or ``(T, Z, H, W)``, in which case the translation model
    returns a 3-component ``(dz, dy, dx)`` shift from a 3-D correlation of the whole
    volume, and a warp model estimates its planar warp on the mid-z plane and takes only
    the axial component from the 3-D correlation (:func:`apply_volume` applies both).

    The returned transforms are **absolute** (each maps a frame directly onto the
    anchor), so they can be applied identically to any other channel via
    :func:`apply_series` — the "register once on a reference channel, apply to all
    channels" rule that preserves colocalization.

    Robustness (V1.60):
    - ``reference="template"`` — two-pass anchor (rough-stabilize → mean), robust to
      cumulative drift and to a bleached / atypical single anchor frame.
    - ``min_confidence`` gates a failed frame by **holding the last good transform**
      (absolute modes) or **skipping the increment** (``previous``) instead of snapping to
      identity — so one low-SNR frame never poisons the series. A warp estimate that FAILED
      outright (``cc == 0``: ECC did not converge, or converged to an implausible warp, or
      the feature fit found no consensus) is gated whatever ``min_confidence`` is — there is
      no estimate to apply.
    - ``normalize="zscore"`` per-frame normalization counters photobleaching.
    - ``roi`` (see :func:`roi_to_mask`) restricts the estimate to a region.
    - ``model="feature"`` uses ORB+RANSAC (``feature_transform`` / ``min_inliers``).

    2026-10-07:
    - ``lowpass_sigma`` / ``correlation`` — see :func:`estimate_translation`.
    - ``landmarks`` — ``{"t": k, "src": [[y, x], …], "dst": [[y, x], …]}``: the same
      physical places clicked on frame 0 and on frame ``k``. :func:`estimate_from_landmarks`
      fits them; with ``model="auto"`` the family it picks becomes the model (without
      landmarks ``auto`` is ``translation``), and for the warp models the fitted warp —
      interpolated to each frame's fraction of the clicked motion — seeds ECC. The fit is
      returned under ``"landmarks"`` so a caller can show the residual per family.
      ``click_sigma`` is the per-click precision the non-rigid verdict is judged against —
      declare it honestly: 2 px clicks judged at 1 px flag a pure drift as non-rigid.
    - ``seed_from_features`` — seed ECC from the ORB+RANSAC warp when it is trusted (see
      :func:`_ecc_seeded`), which is what lets ``first`` mode register a rotation beyond
      ECC's own ~10° capture range.

    Returns ``{"model","reference","shifts" (T,2|3),"warps" (T,2,3)|None,
    "confidence" (T,),"gated" (T,) bool,"landmarks": dict|None}``. ``warps`` is ``None``
    for pure translation (use ``shifts``); otherwise ``shifts[t]`` is the warp's
    translation part (for plotting; its leading entry is the axial shift for a volume).
    """
    vol = np.asarray(series)
    if vol.ndim not in (3, 4):
        raise ValueError(f"estimate_series expects (T,H,W) or (T,Z,H,W), got {vol.shape}")
    if reference not in REFERENCE_MODES:
        raise ValueError(f"unknown reference {reference!r}; expected {REFERENCE_MODES}")

    # Landmarks: fit once; they can choose the model and they seed every warp estimate.
    lm: Optional[Dict[str, Any]] = None
    lm_t = 1
    if landmarks:
        lm = estimate_from_landmarks(landmarks["src"], landmarks["dst"],
                                     model=("auto" if model == "auto" else model),
                                     click_sigma=click_sigma)
        lm_t = max(1, int(landmarks.get("t", 1) or 1))
    if model == "auto":
        model = lm["model"] if lm is not None else "translation"
    if model == "similarity":
        model = "affine"              # ECC has no similarity motion; affine contains it
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS} or 'auto'")

    T = int(vol.shape[0])
    nd = vol.ndim - 1

    if nd == 3 and model != "translation":
        # A volume under a warp model: the planar warp on the mid-z plane, the axial shift
        # from the 3-D correlation. Rotation about z is in-plane, so the two separate.
        common = dict(reference=reference, upsample=upsample, highpass_sigma=highpass_sigma,
                      min_confidence=min_confidence, normalize=normalize, roi=roi,
                      cancelled_cb=cancelled_cb, lowpass_sigma=lowpass_sigma,
                      correlation=correlation)
        planar = estimate_series(vol[:, vol.shape[1] // 2], model=model,
                                 feature_transform=feature_transform,
                                 min_inliers=min_inliers, progress_cb=progress_cb,
                                 landmarks=landmarks, seed_from_features=seed_from_features,
                                 click_sigma=click_sigma, **common)
        axial = estimate_series(vol, model="translation", **common)
        shifts3 = np.zeros((T, 3), dtype=np.float64)
        shifts3[:, 0] = axial["shifts"][:, 0]
        shifts3[:, 1:] = planar["shifts"]
        return {"model": model, "reference": reference, "shifts": shifts3,
                "warps": planar["warps"], "confidence": planar["confidence"],
                "gated": planar["gated"] | axial["gated"],
                "landmarks": planar.get("landmarks")}

    shifts = np.zeros((T, nd), dtype=np.float64)
    confidence = np.ones(T, dtype=np.float64)
    gated = np.zeros(T, dtype=bool)
    use_warp = model != "translation"
    warps = (np.tile(np.eye(2, 3, dtype=np.float32), (T, 1, 1)) if use_warp else None)

    # ROI: rect → bbox (subpixel crop for translation) + box mask (ECC/feature);
    # freeform shapes → mask only. Estimate on the region, apply full-frame.
    mask, bbox = roi_to_mask(roi, vol.shape[-2:])
    tmask = mask if bbox is None else None  # translation uses the crop for a rect

    # Normalization affects correlation only (not geometry); estimate on ``est``.
    est = _normalize_series(vol, normalize)

    # Anchor image for the absolute reference modes.
    if reference == "template":
        anchor = _build_template(est, upsample, highpass_sigma, mask, bbox, cancelled_cb,
                                 lowpass_sigma=lowpass_sigma, correlation=correlation)
    elif reference == "mean":
        anchor = est.mean(axis=0)
    elif reference == "first":
        anchor = est[0]
    else:
        anchor = None  # ``previous`` (cumulative)
    absolute = reference != "previous"

    cum_shift = np.zeros(nd, dtype=np.float64)
    h_abs = np.eye(3, dtype=np.float64)          # composed absolute warp (cumulative)
    last_shift = np.zeros(nd, dtype=np.float64)  # last good (absolute translation)
    last_warp = np.eye(2, 3, dtype=np.float32)   # last good (absolute warp)
    frame_shape = vol.shape[-2:]

    start = 0 if reference in ("mean", "template") else 1
    for t in range(start, T):
        if cancelled_cb is not None and cancelled_cb():
            break
        ref_img = anchor if absolute else est[t - 1]
        cur = est[t]

        if model == "translation":
            sh, conf = estimate_translation(
                ref_img, cur, upsample=upsample, highpass_sigma=highpass_sigma,
                mask=tmask, bbox=bbox, lowpass_sigma=lowpass_sigma,
                correlation=correlation)
            g = bool(conf < float(min_confidence))
            if absolute:
                eff = last_shift if g else sh
                shifts[t] = eff
                last_shift = eff
            else:                                # previous: skip a bad increment
                if not g:
                    cum_shift = cum_shift + sh
                shifts[t] = cum_shift.copy()
            confidence[t] = conf
            gated[t] = g
        else:
            if model == "feature":
                warp, conf = estimate_features(
                    ref_img, cur, transform=feature_transform, mask=mask,
                    bbox=bbox, min_inliers=min_inliers)
            else:
                seed_warp = None
                if lm is not None and lm.get("warp") is not None:
                    frac = (t / lm_t) if absolute else (1.0 / lm_t)
                    seed_warp = _interp_warp(lm["warp"], frac, frame_shape)
                warp, conf = _ecc_seeded(
                    ref_img, cur, model, seed_warp=seed_warp, mask=mask, bbox=bbox,
                    tmask=tmask, upsample=upsample, highpass_sigma=highpass_sigma,
                    lowpass_sigma=lowpass_sigma, correlation=correlation,
                    min_inliers=min_inliers, seed_from_features=seed_from_features)
            # a failed estimate (cc 0) is gated whatever the threshold — nothing to apply
            g = bool(conf <= 0.0 or conf < float(min_confidence))
            if absolute:
                eff = last_warp if g else np.asarray(warp, dtype=np.float32)
                warps[t] = eff
                last_warp = eff
            else:                                # previous: skip a bad increment
                if not g:
                    h_abs = _homog(warp) @ h_abs
                warps[t] = h_abs[:2, :].astype(np.float32)
            shifts[t] = (float(warps[t][1, 2]), float(warps[t][0, 2]))
            confidence[t] = conf
            gated[t] = g

        if progress_cb is not None:
            progress_cb(int(100 * (t - start + 1) / max(T - start, 1)))

    return {"model": model, "reference": reference, "shifts": shifts,
            "warps": warps, "confidence": confidence, "gated": gated,
            "landmarks": lm}


def apply_frame(
    frame: np.ndarray,
    transforms: Dict[str, Any],
    t: int,
    interp_order: int = 1,
) -> np.ndarray:
    """Apply frame ``t``'s effective transform (from :func:`estimate_series`) to ONE
    ``(H, W)`` plane, preserving dtype.

    The per-frame unit of :func:`apply_series`, factored out because the apply half of
    register-once is **separable per plane**: frame ``t``'s output depends only on frame
    ``t``'s pixels and frame ``t``'s transform, never on its neighbours. That is what lets
    ``registration.stabilize`` hand the engine a lazy per-plane provider instead of
    materializing the whole ``(M,T,Z,C,Y,X)`` volume, so a z-stack costs only the planes
    actually looked at.

    :func:`apply_series` delegates here rather than keeping its own copy of the
    shift-vs-warp branch: two implementations of the same arithmetic is exactly how a
    lazy path and an eager path come to disagree in the last bits, and the node asserts
    they do not. (This module used to define ``apply_frame`` TWICE, the second shadowing
    the first; merged 2026-10-07.)

    An all-zero shift (or an identity warp) returns the plane untouched instead of
    resampling by nothing — a no-op interpolation still costs a pass and, at order 1,
    still perturbs the values. An out-of-range ``t`` or an absent transform returns the
    frame unchanged, so it never breaks the read path. A 3-component ``(dz, dy, dx)`` shift
    from a volume estimate applies its planar part here; :func:`apply_volume` is the
    whole-volume form.
    """
    img = np.asarray(frame)
    shifts = transforms.get("shifts")
    warps = transforms.get("warps")
    ti = int(t)
    if warps is None:
        if shifts is None or ti < 0 or ti >= len(shifts):
            return img
        sh = np.asarray(shifts[ti], dtype=np.float64)[-2:]
        return img if not np.any(sh) else apply_shift(img, sh, order=interp_order)
    if ti < 0 or ti >= len(warps):
        return img
    W = np.asarray(warps[ti], dtype=np.float32)
    if np.allclose(W, np.eye(2, 3)):
        return img
    return apply_warp(img, W, output_shape=img.shape[:2], interp_order=interp_order)


def apply_volume(
    volume: np.ndarray,
    transforms: Dict[str, Any],
    t: int,
    interp_order: int = 1,
) -> np.ndarray:
    """Frame ``t``'s transform applied to one ``(Z, H, W)`` volume, preserving dtype — the
    3-D unit ``registration.stabilize`` streams in its 3D mode.

    A 3-component ``(dz, dy, dx)`` shift resamples the volume in one pass (a 2-component
    one is a planar shift on every z). With a planar warp, every plane is warped and the
    axial component of ``shifts[t]`` (its leading entry) then shifts the stack along z —
    the two commute, since the warp is in-plane. Identity short-circuits as in
    :func:`apply_frame`."""
    arr = np.asarray(volume)
    shifts = transforms.get("shifts")
    warps = transforms.get("warps")
    ti = int(t)
    if warps is None:
        if shifts is None or ti < 0 or ti >= len(shifts):
            return arr
        sh = np.asarray(shifts[ti], dtype=np.float64)
        if sh.size == 2:
            sh = np.concatenate([[0.0], sh])
        return arr if not np.any(sh) else apply_shift(arr, sh, order=interp_order)
    if ti < 0 or ti >= len(warps):
        return arr
    W = np.asarray(warps[ti], dtype=np.float32)
    out = arr
    if not np.allclose(W, np.eye(2, 3)):
        out = np.stack([apply_warp(arr[z], W, output_shape=arr.shape[-2:],
                                   interp_order=interp_order) for z in range(arr.shape[0])])
    dz = 0.0
    if shifts is not None and ti < len(shifts) and np.asarray(shifts[ti]).size == 3:
        dz = float(np.asarray(shifts[ti])[0])
    if dz:
        from scipy.ndimage import shift as nd_shift
        moved = nd_shift(np.asarray(out).astype(np.float32), shift=(dz, 0.0, 0.0),
                         order=int(interp_order), mode="constant", cval=0.0)
        out = _cast_like(moved, arr.dtype)
    return out


# ── landmarks: fit user-clicked correspondences, name the family, seed the series ─────

#: The global families a landmark set can be fitted with, simplest first, with the number
#: of parameters each one has — ``2·n`` equations from ``n`` pairs must EXCEED it for the
#: residual to say anything (an exact fit has no residual to judge).
LANDMARK_FAMILIES: Tuple[Tuple[str, int], ...] = (
    ("translation", 2), ("euclidean", 3), ("similarity", 4), ("affine", 6))


def estimate_from_landmarks(
    src_yx: Sequence[Sequence[float]],
    dst_yx: Sequence[Sequence[float]],
    model: str = "auto",
    click_sigma: float = 1.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Fit a global transform to clicked correspondences and say which family the motion
    needs — or that none does.

    ``src_yx[i]`` is a point on the REFERENCE frame, ``dst_yx[i]`` the same physical place
    on the moving frame, both ``(row, col)`` pixels. Every family in
    :data:`LANDMARK_FAMILIES` with more equations than parameters (``2n > p``) is fitted by
    least squares (skimage ``estimate_transform``) and scored by its **dof-corrected** RMS
    residual ``sqrt(RSS / (2n − p))`` — the unbiased estimate of the per-pair click error
    under that family, so a parameter is not rewarded for merely absorbing noise.

    ``model="auto"`` chooses by a **nested F-test**: walking from the simplest family up,
    the first one the richest informative family does NOT improve on significantly
    (``p > alpha``) is chosen. That is what makes five hand-placed pairs enough to tell a
    pure drift from a rotation from a shear without any guess about how precise the clicks
    were — the clicks' own scatter is the yardstick. An explicit ``model`` is fitted as
    asked (``"feature"`` means affine) and must have enough pairs.

    **Non-rigid verdict.** Two clicks per pair, each ``click_sigma`` px off, leave a pure-noise
    pair residual of ``√2·click_sigma``; the richest informative family's residual is tested
    against that with a χ² test at ``alpha``. When it fails, the motion at the clicked points
    is not a global transform: ``nonrigid=True``, and the residual says how far off ANY global
    registration will be at best there. (With the minimum number of pairs the richest family
    fits exactly and nothing can be judged — ``nonrigid`` stays False and the summary says so.)

    Why this answers "which variables make the registration behave": the fit IS the
    transform's parameters, the per-family residual table is the model-selection evidence,
    and the chosen warp seeds :func:`ecc_align` (through :func:`estimate_series`) inside its
    capture range — a 42° rotation that ECC cannot find from a translation seed registers to
    0.02 px from a five-click seed with 1 px click jitter (bench §5).

    Returns ``{"model", "warp" (2,3) float32 reference→moving (x, y), "residual",
    "residuals": {family: corrected rms}, "raw_residuals": {family: rms}, "pvalues":
    {family: p vs the richest}, "n", "nonrigid", "threshold", "summary"}``.
    """
    from skimage.transform import estimate_transform

    src = np.asarray(src_yx, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_yx, dtype=np.float64).reshape(-1, 2)
    n = int(len(src))
    if n == 0 or dst.shape != src.shape:
        raise ValueError("landmarks: need the same number of reference and moving points, "
                         "at least one pair")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("landmarks: every coordinate must be a finite number")
    sx, dx = src[:, ::-1], dst[:, ::-1]                    # (x, y) for skimage
    # family → (warp 2×3, rss, dof, p)
    fits: Dict[str, Tuple[np.ndarray, float, int, int]] = {}
    for fam, p in LANDMARK_FAMILIES:
        if 2 * n < p:
            continue
        if fam == "translation":
            d = (dx - sx).mean(axis=0)
            W = np.array([[1.0, 0.0, d[0]], [0.0, 1.0, d[1]]])
            fit = sx + d
        else:
            tf = estimate_transform(fam, sx, dx)
            params = np.asarray(getattr(tf, "params", np.full((3, 3), np.nan)), dtype=float)
            if not np.all(np.isfinite(params)):
                continue
            W = params[:2, :]
            fit = tf(sx)
        rss = float(((fit - dx) ** 2).sum())
        fits[fam] = (W.astype(np.float32), rss, 2 * n - p, p)

    def _corrected(fam: str) -> float:
        _w, rss, dof, _p = fits[fam]
        return math.sqrt(rss / dof) if dof > 0 else float("inf")

    def _raw(fam: str) -> float:
        return math.sqrt(fits[fam][1] / (2 * n))

    informative = [f for f, _p in LANDMARK_FAMILIES if f in fits and fits[f][2] > 0]
    richest = informative[-1] if informative else list(fits)[-1]
    pvalues: Dict[str, float] = {}
    if informative:
        from scipy.stats import f as f_dist
        _wr, rss_r, dof_r, p_r = fits[richest]
        for fam in informative:
            _wf, rss_f, _dof_f, p_f = fits[fam]
            if fam == richest or p_r == p_f:
                pvalues[fam] = 1.0
                continue
            if rss_r <= 1e-12:
                pvalues[fam] = 1.0 if rss_f <= 1e-12 else 0.0
                continue
            F = ((rss_f - rss_r) / (p_r - p_f)) / (rss_r / dof_r)
            pvalues[fam] = float(f_dist.sf(max(F, 0.0), p_r - p_f, dof_r))

    if model == "auto":
        chosen = richest
        for fam in informative:
            if pvalues.get(fam, 0.0) > alpha:
                chosen = fam
                break
        judge = richest
    else:
        fam = {"feature": "affine"}.get(model, model)
        need = dict(LANDMARK_FAMILIES).get(fam)
        if need is None:
            raise ValueError(f"landmarks: unknown model {model!r}; expected one of "
                             f"{[f for f, _ in LANDMARK_FAMILIES]} or 'auto'")
        if fam not in fits:
            raise ValueError(f"landmarks: {fam} needs at least {-(-need // 2)} point pairs, "
                             f"got {n}")
        chosen = judge = fam

    # the non-rigid verdict: can the richest (or the requested) family explain the clicks
    # given their precision? χ² on RSS / (2 σ_click²) with that family's dof.
    nonrigid = False
    threshold = float("nan")
    _wj, rss_j, dof_j, _pj = fits[judge]
    judged = dof_j > 0
    if judged:
        from scipy.stats import chi2
        sig2 = 2.0 * float(click_sigma) ** 2
        threshold = math.sqrt(sig2 * chi2.ppf(1.0 - alpha, dof_j) / dof_j)   # in corrected-rms px
        nonrigid = bool(_corrected(judge) > threshold)

    W = fits[chosen][0]
    parts = [f"{f}: {_corrected(f):.2f} px" if fits[f][2] > 0 else f"{f}: exact fit"
             for f, _ in LANDMARK_FAMILIES if f in fits]
    if not judged:
        verdict = (f"{chosen} fitted exactly — {n} pair{'s' if n != 1 else ''} cannot judge "
                   f"whether a global transform fits; add pairs (5 well-spread ones tell the "
                   f"families apart).")
    elif nonrigid:
        verdict = (f"NON-RIGID at the clicked points: even {judge} leaves "
                   f"{_corrected(judge):.2f} px against {threshold:.2f} px expected from "
                   f"{click_sigma:g} px clicks. No global transform registers this field; "
                   f"restrict the estimate to a rigid region or expect that much residual.")
    else:
        verdict = (f"{chosen} chosen ({_corrected(chosen):.2f} px residual; richer families "
                   f"do not improve on it at p<{alpha:g}).")
    summary = f"{n} landmark pairs — " + "; ".join(parts) + ". " + verdict
    return {"model": chosen, "warp": W, "residual": float(_corrected(chosen)),
            "residuals": {f: float(_corrected(f)) for f in fits},
            "raw_residuals": {f: float(_raw(f)) for f in fits},
            "pvalues": pvalues, "n": n, "nonrigid": bool(nonrigid),
            "threshold": float(threshold), "summary": summary}


def parse_landmarks(text: Any) -> Optional[Dict[str, Any]]:
    """The ``landmarks`` socket text → ``{"t", "src", "dst"}`` for :func:`estimate_series`,
    or ``None`` when blank.

    Two forms. JSON: ``{"t": 5, "src": [[y, x], …], "dst": [[y, x], …]}`` (or ``"pairs":
    [[[y, x], [y, x]], …]``) — what a click gesture writes. Compact text for typing:
    ``t=5; 10.5,20 -> 12,21.5; 40,60 -> 41.2,60.9`` — pairs separated by ``;`` or newlines,
    reference point before the arrow (``->`` or ``>``), moving point after, every point
    ``row,col``. The frame index is REQUIRED: a point pair means nothing without knowing
    which frame the moving points were clicked on."""
    if text is None:
        return None
    if isinstance(text, dict):
        raw = text
    else:
        s = str(text).strip()
        if not s:
            return None
        raw = None
        if s[0] in "{[":
            import json as _json
            try:
                raw = _json.loads(s)
            except ValueError as exc:
                raise ValueError(f"landmarks: not valid JSON ({exc})") from exc
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("landmarks: JSON must be an object with t, src and dst")
        if "pairs" in raw and "src" not in raw:
            pairs = list(raw.get("pairs") or [])
            src = [p[0] for p in pairs]
            dst = [p[1] for p in pairs]
        else:
            src, dst = list(raw.get("src") or []), list(raw.get("dst") or [])
        if "t" not in raw:
            raise ValueError("landmarks: say which frame the moving points were clicked "
                             "on (\"t\")")
        t = int(raw["t"])
    else:
        t = None
        src, dst = [], []
        for chunk in re.split(r"[;\n]+", s):
            chunk = chunk.strip()
            if not chunk:
                continue
            m_t = re.fullmatch(r"t\s*[=:]\s*(\d+)", chunk, flags=re.I)
            if m_t:
                t = int(m_t.group(1))
                continue
            halves = re.split(r"\s*(?:->|>|→)\s*", chunk)
            if len(halves) != 2:
                raise ValueError(f"landmarks: cannot read {chunk!r}; expected "
                                 f"'row,col -> row,col'")
            try:
                a = [float(v) for v in halves[0].split(",")]
                b = [float(v) for v in halves[1].split(",")]
            except ValueError as exc:
                raise ValueError(f"landmarks: cannot read {chunk!r}: {exc}") from exc
            if len(a) != 2 or len(b) != 2:
                raise ValueError(f"landmarks: each point is 'row,col', got {chunk!r}")
            src.append(a)
            dst.append(b)
        if t is None:
            raise ValueError("landmarks: say which frame the moving points were clicked "
                             "on, e.g. 't=5; 10,20 -> 12,21'")
    if not src or len(src) != len(dst):
        raise ValueError("landmarks: need at least one pair, and as many reference points "
                         "as moving points")
    if t < 1:
        raise ValueError("landmarks: t is the moving frame's index and must be ≥ 1 (frame 0 "
                         "is the reference)")
    return {"t": t, "src": [[float(p[0]), float(p[1])] for p in src],
            "dst": [[float(p[0]), float(p[1])] for p in dst]}


def _decompose_warp(warp: np.ndarray, shape: Sequence[int]):
    """2×3 reference→moving warp → ``(angle, P, drift, centre)`` about the frame centre:
    linear part ``L = R(angle) @ P`` by polar decomposition (``P`` symmetric positive —
    scale / shear), translation split into the part a rotation about the centre produces
    and a pure ``drift``."""
    A = np.asarray(warp, dtype=np.float64)
    L, t = A[:2, :2], A[:2, 2]
    c = np.array([(shape[-1] - 1) / 2.0, (shape[-2] - 1) / 2.0])       # (x, y)
    U, S, Vt = np.linalg.svd(L)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    R = U @ Vt
    P = Vt.T @ np.diag(S) @ Vt
    theta = math.atan2(R[1, 0], R[0, 0])
    drift = t - (c - L @ c)
    return theta, P, drift, c


def _interp_warp(warp: np.ndarray, frac: float, shape: Sequence[int]) -> np.ndarray:
    """The warp at ``frac`` of a motion: angle and drift scale linearly, the symmetric
    part by a matrix power — so a landmark warp measured at frame ``k`` seeds frame ``t``
    with ``frac = t / k`` (or one increment with ``1 / k``) and a 40° turn interpolates as
    a 20° turn, not as a shrunken 40° one."""
    theta, P, drift, c = _decompose_warp(warp, shape)
    w, V = np.linalg.eigh(P)
    Ps = V @ np.diag(np.clip(w, 1e-6, None) ** float(frac)) @ V.T
    th = theta * float(frac)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    L = R @ Ps
    t = c - L @ c + float(frac) * drift
    out = np.zeros((2, 3), dtype=np.float32)
    out[:, :2] = L
    out[:, 2] = t
    return out


def apply_series(
    series: np.ndarray,
    transforms: Dict[str, Any],
    interp_order: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Apply per-frame effective ``transforms`` (from :func:`estimate_series`) to a
    ``(T, H, W)`` series — any channel — returning the aligned series (dtype
    preserved). This is the "apply to all channels" half of register-once.

    Per-frame work lives in :func:`apply_frame`; this is the loop over ``t``."""
    vol = np.asarray(series)
    if vol.ndim != 3:
        raise ValueError(f"apply_series expects (T,H,W), got {vol.shape}")
    T = int(vol.shape[0])
    out = np.empty_like(vol)
    for t in range(T):
        out[t] = apply_frame(vol[t], transforms, t, interp_order=interp_order)
        if progress_cb is not None:
            progress_cb(int(100 * (t + 1) / max(T, 1)))
    return out


