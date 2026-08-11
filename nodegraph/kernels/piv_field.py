"""piv_field — 2D multipass window-deformation PIV, a thin driver around **OpenPIV**.

The real correlation math is EXTERNAL: FFT cross-correlation, subpixel peak fits,
signal-to-noise, validation and vector replacement all live in the third-party
``openpiv`` package (GPLv3, github.com/OpenPIV/openpiv-python), imported **lazily**
inside :func:`_require_openpiv`. What this module owns is the in-repo glue:

- a **multipass driver** that mirrors ``openpiv.windef.piv`` / ``simple_multipass``
  pass-for-pass (first pass → validate → replace → optional smoothn → N window-
  deformation passes) but (a) keeps the FINAL pass's signal-to-noise field, which
  ``windef`` computes and throws away, (b) never calls ``transform_coordinates``,
  so everything stays in image coordinates — the probe in ``scripts/piv_synthetic_bench.py``
  established empirically that OpenPIV's raw ``u`` is +x (columns right) and raw
  ``v`` is +y (ROWS DOWN), i.e. exactly this repo's ``(dy, dx)`` convention with no
  axis swap, and (c) survives the all-vectors-invalid case that makes
  ``windef.multipass_img_deform`` raise ``ValueError``.
- a **pass-ladder builder** with honest degradation: window sizes that do not fit
  the image, or whose vector grid is too small for the cubic predictor spline
  (< 4 nodes per axis), are dropped rather than crashing scipy mid-pass.
- the :class:`PIVResult` container, shaped so the catalog's shared ``_dvc_rows``
  Point flattener consumes it exactly like a ``DVCResult``.

Everything here follows the kernel README conventions: arrays are ``(H, W)`` =
``(y, x)``, ``voxel_size_um`` is ``(dy, dx)``, displacements are in PIXELS with
component order ``[dy, dx]`` (+dy = down rows), and the CALLER owns all data prep
(channel choice, z/m/t looping, crop, downsample, registration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_INSTALL_HINT = "pip install openpiv"


def openpiv_available() -> bool:
    """True when the external ``openpiv`` package is importable (no import side effects)."""
    from importlib.util import find_spec

    return find_spec("openpiv") is not None


def pivuq_available() -> bool:
    """True when the external ``pivuq`` package is importable (no import side effects)."""
    from importlib.util import find_spec

    return find_spec("pivuq") is not None


def _require_openpiv():
    """Import the openpiv modules this driver needs, or raise a friendly ImportError."""
    try:
        from openpiv import filters, pyprocess, smoothn, validation, windef
        from openpiv.settings import PIVSettings
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "PIV needs the external 'openpiv' package (GPLv3), which is imported "
            f"lazily at run time: {_INSTALL_HINT}"
        ) from exc
    return pyprocess, windef, validation, filters, smoothn, PIVSettings


@dataclass
class PIVResult:
    """One frame pair's PIV field on the interrogation-window grid.

    Field names and layouts deliberately match ``DVCResult`` where they overlap, so
    the catalog's shared ``_dvc_rows`` flattener consumes either. All coordinates
    and displacements are in (possibly downsampled) PIXELS, image convention:
    component 0 = y (+down rows), component 1 = x (+right columns). µm conversion
    is the caller's job via ``voxel_size_um``."""

    dim: int                                  # always 2
    grid_coords: np.ndarray                   # (Gy, Gx, 2) [y, x] window centers, px
    displacement_field: np.ndarray            # (Gy, Gx, 2) [dy, dx] px; NaN = unsolved
    strain_field: Optional[np.ndarray]        # None (PIV measures displacement only)
    qfactor: Optional[np.ndarray]             # (Gy, Gx) final-pass signal-to-noise
    voxel_size_um: Tuple[float, float]        # (dy, dx) µm/px, echoed from the caller
    flags: np.ndarray                         # (Gy, Gx) bool True = failed validation
    excluded: np.ndarray                      # (Gy, Gx) bool True = outside the ROI mask
    method: str = "OpenPIV"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _as_frame(a: np.ndarray, what: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"PIV {what} must be a single (H, W) plane, got shape {arr.shape}")
    return arr


def _overlap_px(window: int, frac: float) -> int:
    """Overlap in px for one pass: fraction of the window, clamped to [0, window-1]."""
    return int(min(window - 1, max(0, round(window * frac))))


def _ladder(
    shape: Tuple[int, int],
    windowsizes: Sequence[int],
    overlap_frac: float,
    get_field_shape,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], bool]:
    """Resolve the effective pass ladder for this image size.

    Rules (see module docstring): every pass of a MULTIPASS run needs >= 4 vector
    rows and columns, because both the predictor interpolation and the deformation
    field use cubic ``RectBivariateSpline`` over the vector grid. Windows larger
    than the image are dropped outright. If no multipass-capable ladder survives,
    fall back to a single pass on the smallest requested window (any grid size);
    if even that window does not fit the image, raise."""
    req = tuple(int(w) for w in windowsizes)
    if not req:
        raise ValueError("PIV needs at least one window size")
    h, w = int(shape[0]), int(shape[1])
    final = req[-1]
    if final > min(h, w):
        raise ValueError(
            f"PIV window size {final} px does not fit a {h}x{w} px image; "
            "reduce the window size or crop less aggressively"
        )

    def _grid_ok(win: int) -> bool:
        if win > min(h, w):
            return False
        rows, cols = get_field_shape((h, w), win, _overlap_px(win, overlap_frac))
        return rows >= 4 and cols >= 4

    eff = tuple(win for win in req if _grid_ok(win))
    if len(eff) < 2 or eff[-1] != final:
        eff = (final,)                        # honest degradation: single pass
    ov = tuple(_overlap_px(win, overlap_frac) for win in eff)
    return eff, ov, eff != req


def _build_settings(params: Dict[str, Any], eff_ws, eff_ov, PIVSettings):
    """A FRESH PIVSettings per call — ``windef`` mutates the settings object it is
    handed (``multipass_img_deform`` nulls ``sig2noise_method`` when validation is
    off), so sharing one across calls leaks state between frame pairs."""
    s = PIVSettings()
    s.windowsizes = tuple(eff_ws)
    s.overlap = tuple(eff_ov)
    s.num_iterations = len(eff_ws)
    s.correlation_method = str(params.get("correlation_method", "circular"))
    # openpiv contract: 'linear' correlation requires normalized correlation.
    s.normalized_correlation = bool(
        params.get("normalized_correlation", False)
    ) or s.correlation_method == "linear"
    s.subpixel_method = str(params.get("subpixel_method", "gaussian"))
    s.deformation_method = str(params.get("deformation_method", "symmetric"))
    s.interpolation_order = int(params.get("interpolation_order", 3))
    s.use_vectorized = False
    s.dt = 1.0
    s.scaling_factor = 1.0
    s.sig2noise_method = str(params.get("sig2noise_method", "peak2mean"))
    s.sig2noise_mask = int(params.get("sig2noise_mask", 2))
    s.sig2noise_threshold = float(params.get("sig2noise_threshold", 1.0))
    s.sig2noise_validate = bool(params.get("sig2noise_validate", True))
    s.validation_first_pass = True
    lim = float(params.get("max_disp_px", 0.0))
    if lim <= 0.0:
        lim = float(eff_ws[0]) / 2.0          # auto: half the coarsest window
    s.min_max_u_disp = (-lim, lim)
    s.min_max_v_disp = (-lim, lim)
    s.std_threshold = float(params.get("std_threshold", 10.0))
    median_test = str(params.get("median_test", "universal"))
    s.median_normalized = median_test == "universal"
    s.median_threshold = (
        float(params.get("median_threshold", 2.0)) if median_test != "off" else 1e12
    )
    s.median_size = int(params.get("median_size", 1))
    s.replace_vectors = bool(params.get("replace_vectors", True))
    s.smoothn = bool(params.get("smoothn", False))
    s.smoothn_p = float(params.get("smoothn_p", 0.05))
    s.filter_method = str(params.get("filter_method", "localmean"))
    s.max_filter_iteration = int(params.get("max_filter_iteration", 4))
    s.filter_kernel_size = int(params.get("filter_kernel_size", 2))
    s.static_mask = None                      # ROI handled here, not by windef
    s.show_plot = s.show_all_plots = s.save_plot = False
    return s


def _roi_on_grid(roi_mask: Optional[np.ndarray], x, y) -> np.ndarray:
    """True where a grid point falls OUTSIDE the ROI (nearest-neighbour sampling —
    windef spline-interpolates its boolean static mask, order 0 is the sane choice
    for a mask and a deliberate deviation)."""
    if roi_mask is None:
        return np.zeros(x.shape, dtype=bool)
    from scipy.ndimage import map_coordinates

    inside = map_coordinates(
        np.asarray(roi_mask, dtype=np.float32), [y, x], order=0, mode="nearest"
    )
    return inside < 0.5


def _deform_pass(frame_a, frame_b, i, x_old, y_old, u_old, v_old, s,
                 roi_mask, pyprocess, windef):
    """One window-deformation pass — ``windef.multipass_img_deform`` re-implemented
    on openpiv's public pieces so the pass RETURNS its signal-to-noise field and
    never raises on an all-invalid grid. Kept line-for-line parallel to upstream
    (predictor spline -> symmetric/second-image deformation -> correlate -> add
    predictor); the bench asserts parity against ``simple_multipass``."""
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import map_coordinates

    window_size = s.windowsizes[i]
    overlap = s.overlap[i]
    x, y = pyprocess.get_rect_coordinates(frame_a.shape, window_size, overlap)

    ip_u = RectBivariateSpline(y_old[:, 0], x_old[0, :], np.ma.filled(u_old, 0.0))
    ip_v = RectBivariateSpline(y_old[:, 0], x_old[0, :], np.ma.filled(v_old, 0.0))
    u_pre = ip_u(y[:, 0], x[0, :])
    v_pre = ip_v(y[:, 0], x[0, :])

    if s.deformation_method == "symmetric":
        x_new, y_new, ut, vt = windef.create_deformation_field(
            frame_a, x, y, u_pre, v_pre, interpolation_order=s.interpolation_order)
        fa = map_coordinates(frame_a, ((y_new - vt / 2, x_new - ut / 2)),
                             order=s.interpolation_order, mode="nearest")
        fb = map_coordinates(frame_b, ((y_new + vt / 2, x_new + ut / 2)),
                             order=s.interpolation_order, mode="nearest")
    elif s.deformation_method == "second image":
        fa = frame_a
        fb = windef.deform_windows(frame_b, x, y, u_pre, -v_pre,
                                   interpolation_order=s.interpolation_order)
    else:
        raise ValueError(f"unknown deformation_method {s.deformation_method!r}")

    u, v, s2n = pyprocess.extended_search_area_piv(
        fa, fb, window_size=window_size, overlap=overlap,
        width=s.sig2noise_mask, subpixel_method=s.subpixel_method,
        sig2noise_method=s.sig2noise_method,
        correlation_method=s.correlation_method,
        normalized_correlation=s.normalized_correlation,
        use_vectorized=s.use_vectorized)
    shape = pyprocess.get_field_shape(frame_a.shape, window_size, overlap)
    u = u.reshape(shape) + u_pre
    v = v.reshape(shape) + v_pre
    s2n = s2n.reshape(shape)

    excluded = _roi_on_grid(roi_mask, x, y)
    u = np.ma.masked_array(u, mask=excluded)
    v = np.ma.masked_array(v, mask=excluded)
    return x, y, u, v, s2n, excluded


def _piv_one_pair(frame_a, frame_b, params, voxel_size_um, roi_mask,
                  pyprocess, windef, validation, filters, smoothn_mod, PIVSettings
                  ) -> PIVResult:
    """The multipass driver for ONE frame pair. Mirrors ``windef.piv`` semantics:
    first pass -> mask -> validate -> replace (mandatory before any deform pass,
    holes poison the predictor) -> optional smoothn on every non-final pass ->
    deform passes -> final validation -> replacement only if ``replace_vectors``."""
    frame_a = _as_frame(frame_a, "frame A")
    frame_b = _as_frame(frame_b, "frame B")
    if frame_a.shape != frame_b.shape:
        raise ValueError(
            f"PIV frame pair shapes differ: {frame_a.shape} vs {frame_b.shape}")
    if roi_mask is not None and np.asarray(roi_mask).shape != frame_a.shape:
        raise ValueError(
            f"PIV roi_mask shape {np.asarray(roi_mask).shape} != frame {frame_a.shape}")

    overlap_frac = float(params.get("overlap", 0.5))
    eff_ws, eff_ov, trimmed = _ladder(
        frame_a.shape, params.get("windowsizes", (64, 32, 16)), overlap_frac,
        pyprocess.get_field_shape)
    s = _build_settings(params, eff_ws, eff_ov, PIVSettings)
    n_passes = s.num_iterations

    def _smooth(u, v, excluded):
        u, *_ = smoothn_mod.smoothn(u, s=s.smoothn_p)
        v, *_ = smoothn_mod.smoothn(v, s=s.smoothn_p)
        # smoothn returns plain arrays; re-mask (upstream does the same re-mask)
        return (np.ma.masked_array(u, mask=excluded),
                np.ma.masked_array(v, mask=excluded))

    # ── first pass ────────────────────────────────────────────────────────────
    x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, s)
    excluded = _roi_on_grid(roi_mask, x, y)
    u = np.ma.masked_array(u, mask=excluded)
    v = np.ma.masked_array(v, mask=excluded)
    flags = validation.typical_validation(u, v, s2n, s)
    replaced_final = False                    # did the FINAL pass's field get inpainted?

    if n_passes == 1:
        if s.replace_vectors and flags.any() and not flags.all():
            u, v = filters.replace_outliers(
                u, v, flags, method=s.filter_method,
                max_iter=s.max_filter_iteration, kernel_size=s.filter_kernel_size)
            replaced_final = True
        if s.smoothn:
            u, v = _smooth(u, v, excluded)
    else:
        # multipass: the predictor may not contain holes -> replace unconditionally
        if flags.any() and not flags.all():
            u, v = filters.replace_outliers(
                u, v, flags, method=s.filter_method,
                max_iter=s.max_filter_iteration, kernel_size=s.filter_kernel_size)
        if s.smoothn:
            u, v = _smooth(u, v, excluded)
        for i in range(1, n_passes):
            x, y, u, v, s2n, excluded = _deform_pass(
                frame_a, frame_b, i, x, y, u, v, s, roi_mask, pyprocess, windef)
            flags = validation.typical_validation(u, v, s2n, s)
            final = i == n_passes - 1
            if (not final or s.replace_vectors) and flags.any() and not flags.all():
                u, v = filters.replace_outliers(
                    u, v, flags, method=s.filter_method,
                    max_iter=s.max_filter_iteration, kernel_size=s.filter_kernel_size)
                if final:
                    replaced_final = True
            if not final and s.smoothn:
                u, v = _smooth(u, v, excluded)

    return _package_result(x, y, u, v, s2n, flags, excluded, replaced_final,
                           eff_ws, eff_ov, trimmed, s, voxel_size_um)


def _package_result(x, y, u, v, s2n, flags, excluded, replaced_final,
                    eff_ws, eff_ov, trimmed, s, voxel_size_um,
                    extra_diag: Optional[Dict[str, Any]] = None) -> PIVResult:
    """Shared final packaging (pair and ensemble paths — one copy so they cannot drift).

    Contract: the final field carries only vectors that PASSED validation or were
    inpainted. If replacement did not run (turned off, or every vector failed and
    there was nothing to inpaint from), flagged vectors come back NaN, and `flags`
    on the survivors means "this value was filled from its neighbours"."""
    dy = np.ma.filled(np.ma.masked_array(v, mask=excluded), np.nan)
    dx = np.ma.filled(np.ma.masked_array(u, mask=excluded), np.nan)
    if not replaced_final and flags.any():
        dy = dy.copy(); dx = dx.copy()
        dy[flags] = np.nan
        dx[flags] = np.nan
    disp = np.stack([dy, dx], axis=-1)        # [dy, dx], +dy = down rows (probed)
    grid = np.stack([np.asarray(y, float), np.asarray(x, float)], axis=-1)
    flags = np.asarray(flags, dtype=bool)
    diag: Dict[str, Any] = {
        "windowsizes": tuple(eff_ws), "overlaps": tuple(eff_ov),
        "n_passes": len(eff_ws), "ladder_trimmed": bool(trimmed),
        "correlation": s.correlation_method,
        "normalized_correlation": bool(s.normalized_correlation),
        "subpixel": s.subpixel_method, "deformation": s.deformation_method,
        "n_flagged": int(flags.sum()), "n_excluded": int(excluded.sum()),
    }
    if extra_diag:
        diag.update(extra_diag)
    return PIVResult(
        dim=2,
        grid_coords=grid,
        displacement_field=disp,
        strain_field=None,
        qfactor=np.asarray(s2n, dtype=float),
        voxel_size_um=(float(voxel_size_um[0]), float(voxel_size_um[1])),
        flags=flags,
        excluded=excluded,
        diagnostics=diag,
    )


def run_piv_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    voxel_size_um: Tuple[float, float],
    params: Dict[str, Any],
    *,
    roi_mask: Optional[np.ndarray] = None,
) -> PIVResult:
    """PIV one already-prepared ``(H, W)`` frame pair; see module docstring for params."""
    mods = _require_openpiv()
    return _piv_one_pair(frame_a, frame_b, params, voxel_size_um, roi_mask, *mods)


def run_piv_series(
    images: Sequence[np.ndarray],
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, float],
    *,
    pairing: str = "previous",
    roi_mask: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> List[PIVResult]:
    """PIV an ordered series. ``images`` is indexed lazily (``__len__``/``__getitem__``
    suffice), each frame is pulled exactly once, and peak memory stays at ~two planes.

    pairing='previous': correlate (images[i-1], images[i]) — a velocity field per step.
    pairing='fixed_head': correlate (images[0], images[i]) — displacement from a fixed
    reference (the caller prepends the reference plane). Both return ``len(images)-1``
    results, in order. A truthy ``cancelled_cb`` stops between pairs and returns the
    partial list."""
    n = len(images)
    if n < 2:
        raise ValueError("PIV needs at least 2 images (a reference and a current frame)")
    if pairing not in ("previous", "fixed_head"):
        raise ValueError(f"unknown pairing {pairing!r}")
    mods = _require_openpiv()
    out: List[PIVResult] = []
    head = _as_frame(images[0], "frame 0")
    prev = head
    for i in range(1, n):
        if cancelled_cb is not None and cancelled_cb():
            break
        cur = _as_frame(images[i], f"frame {i}")
        a = head if pairing == "fixed_head" else prev
        out.append(_piv_one_pair(a, cur, params, voxel_size_um, roi_mask, *mods))
        prev = cur
        if progress_cb is not None:
            progress_cb(int(round(100.0 * i / (n - 1))))
    return out


def disparity_uncertainty(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    res: PIVResult,
    *,
    window_size: Optional[int] = None,
    grid_size: int = 4,
    min_peaks: int = 2,
) -> np.ndarray:
    """Per-vector displacement uncertainty by IMAGE MATCHING (Sciacchitano, Wieneke &
    Scarano 2013), via the external ``pivuq`` package → ``(Gy, Gx, 2)`` ``[unc_y, unc_x]``
    in PIXELS on ``res``'s grid.

    Both frames are warped toward each other by the measured field; the residual
    per-particle position disparities are pooled per window and Eq. (3)'s
    ``sqrt(mu^2 + (sigma/sqrt(N))^2)`` is the standard uncertainty of the vector. Probed
    empirically here: a correct field returns δ at the actual error scale (~0.03–0.06 px
    on the bench fixture, where the true RMSE is 0.031), and a planted +1 px x-error
    comes back as δx ≈ 1.06 px in the right component. Conventions verified from pivuq
    source: its ``U`` is ``(u=+x, v=+y ROWS-DOWN)`` and its ``delta`` is ``[x, y]`` —
    this adapter swaps both to this repo's ``[y, x]``.

    Windows where fewer than ``min_peaks`` particle disparities were found come back
    **NaN** — no honest estimate exists there (pivuq would report a hard 0). NaN in
    ``res.displacement_field`` is also NaN here. ``window_size`` defaults to the final
    interrogation window from ``res.diagnostics``.

    The dense field handed to pivuq is built HERE (``RectBivariateSpline``), because
    pivuq's own sparse-field upsampler still calls ``scipy.interpolate.interp2d``,
    which modern scipy has removed — a dense ``(2, H, W)`` input skips that path.

    Install: ``pip install pivuq --no-deps`` (its pinned numba tries to build from
    source on this Python; the packages already present satisfy it)."""
    try:
        from pivuq import disparity
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "PIV uncertainty needs the external 'pivuq' package: "
            "pip install pivuq --no-deps  (its pinned numba fails to build on this "
            "Python; numpy/scipy/scikit-image/numba already present satisfy it)"
        ) from exc
    from scipy.interpolate import RectBivariateSpline

    frame_a = _as_frame(frame_a, "frame A")
    frame_b = _as_frame(frame_b, "frame B")
    h, w = frame_a.shape
    g = res.grid_coords
    if window_size is None:
        window_size = int(res.diagnostics.get("windowsizes", (32,))[-1])

    y1, x1 = g[:, 0, 0], g[0, :, 1]
    dy = np.nan_to_num(res.displacement_field[..., 0])
    dx = np.nan_to_num(res.displacement_field[..., 1])
    kx = int(min(3, len(x1) - 1))
    ky = int(min(3, len(y1) - 1))
    yy, xx = np.arange(h), np.arange(w)
    u_dense = RectBivariateSpline(y1, x1, dx, kx=kx, ky=ky)(yy, xx)
    v_dense = RectBivariateSpline(y1, x1, dy, kx=kx, ky=ky)(yy, xx)

    _X, _Y, delta, n_peaks, _mu, _sigma = disparity.sws(
        np.stack([frame_a, frame_b]), np.stack([u_dense, v_dense]),
        window_size=int(window_size), grid_size=int(grid_size))
    delta = np.where(n_peaks[None, ...] >= min_peaks, delta, np.nan)

    # sample the grid_size-pitch delta map at the PIV window centres (nearest)
    n_gy, n_gx = delta.shape[1], delta.shape[2]
    iy = np.clip(np.round((g[..., 0] - grid_size / 2) / grid_size).astype(int), 0, n_gy - 1)
    ix = np.clip(np.round((g[..., 1] - grid_size / 2) / grid_size).astype(int), 0, n_gx - 1)
    unc = np.stack([delta[1][iy, ix], delta[0][iy, ix]], axis=-1)   # → [y, x]
    unc[~np.isfinite(res.displacement_field)] = np.nan
    return unc


def run_piv_ensemble(
    images: Sequence[np.ndarray],
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, float],
    *,
    pairing: str = "previous",
    roi_mask: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> PIVResult:
    """Ensemble (correlation-averaged) PIV: ONE time-averaged field for the whole series.

    The micro-PIV method of Meinhart, Wereley & Santiago (2000): at every pass, the
    CORRELATION PLANES of all frame pairs are summed before peak-finding — signal at the
    true displacement accumulates over pairs while random cross-particle peaks average
    out, so a field is recoverable from seeding far too sparse for any single pair
    (upstream's measured case: <1% bad vectors at ~2.5 particles per window over 8 pairs,
    where image- or vector-averaging leave 5–12%). Assumes the flow is statistically
    STEADY over the series (or the deformation static, for a `fixed_head` bead series);
    real temporal variation is averaged away, not detected.

    Multipass window deformation composes with the averaging exactly as in PIVlab's
    ensemble mode: each pass deforms EVERY pair by the ensemble predictor (deformation
    is always symmetric here — `deformation_method` is ignored), re-correlates, and
    averages again. `qfactor` is the signal-to-noise of the AVERAGED plane — the growth
    of this number with series length is the method working. Frames are re-read lazily
    once per pass (`n_passes × (len(images) - 1)` pair reads); peak memory is ~two
    planes plus one correlation stack. A truthy ``cancelled_cb`` raises RuntimeError."""
    n = len(images)
    if n < 2:
        raise ValueError("PIV ensemble needs at least 2 images (>= 1 pair)")
    if pairing not in ("previous", "fixed_head"):
        raise ValueError(f"unknown pairing {pairing!r}")
    pyprocess, windef, validation, filters, smoothn_mod, PIVSettings = _require_openpiv()
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import map_coordinates

    head = _as_frame(images[0], "frame 0")
    shape = head.shape
    if roi_mask is not None and np.asarray(roi_mask).shape != shape:
        raise ValueError(
            f"PIV roi_mask shape {np.asarray(roi_mask).shape} != frame {shape}")
    overlap_frac = float(params.get("overlap", 0.5))
    eff_ws, eff_ov, trimmed = _ladder(
        shape, params.get("windowsizes", (64, 32, 16)), overlap_frac,
        pyprocess.get_field_shape)
    s = _build_settings(params, eff_ws, eff_ov, PIVSettings)
    n_passes = s.num_iterations
    n_pairs = n - 1
    total_units = n_passes * n_pairs
    done = 0

    def _pairs():
        first = _as_frame(images[0], "frame 0")
        prev = first
        for i in range(1, n):
            if cancelled_cb is not None and cancelled_cb():
                raise RuntimeError("PIV ensemble cancelled")
            cur = _as_frame(images[i], f"frame {i}")
            if cur.shape != shape:
                raise ValueError(
                    f"frame {i} shape {cur.shape} != frame 0 shape {shape}")
            yield (first, cur) if pairing == "fixed_head" else (prev, cur)
            prev = cur

    def _mean_corr(window_size, overlap, deform_coords):
        """Sum correlation planes over all pairs (optionally pre-deformed) → mean."""
        nonlocal done
        ws2 = (window_size, window_size)
        ov2 = (overlap, overlap)
        corr_sum = None
        for fa, fb in _pairs():
            if deform_coords is not None:
                ca, cb = deform_coords
                fa = map_coordinates(fa, ca, order=s.interpolation_order, mode="nearest")
                fb = map_coordinates(fb, cb, order=s.interpolation_order, mode="nearest")
            aa = pyprocess.sliding_window_array(fa, ws2, ov2)
            bb = pyprocess.sliding_window_array(fb, ws2, ov2)
            c = pyprocess.fft_correlate_images(
                aa, bb, correlation_method=s.correlation_method,
                normalized_correlation=s.normalized_correlation)
            corr_sum = c if corr_sum is None else corr_sum + c
            done += 1
            if progress_cb is not None:
                progress_cb(int(round(100.0 * done / total_units)))
        return corr_sum / n_pairs

    def _peaks(corr_mean, n_rows, n_cols):
        u, v = pyprocess.correlation_to_displacement(
            corr_mean, n_rows, n_cols, subpixel_method=s.subpixel_method)
        s2n = pyprocess.sig2noise_ratio(
            corr_mean, sig2noise_method=s.sig2noise_method,
            width=s.sig2noise_mask).reshape(n_rows, n_cols)
        return u, v, s2n

    def _smooth(u, v, excluded):
        u, *_ = smoothn_mod.smoothn(u, s=s.smoothn_p)
        v, *_ = smoothn_mod.smoothn(v, s=s.smoothn_p)
        return (np.ma.masked_array(u, mask=excluded),
                np.ma.masked_array(v, mask=excluded))

    def _replace(u, v, flags):
        return filters.replace_outliers(
            u, v, flags, method=s.filter_method,
            max_iter=s.max_filter_iteration, kernel_size=s.filter_kernel_size)

    # ── pass 0: plain windows, planes averaged over pairs ─────────────────────
    window_size, overlap = eff_ws[0], eff_ov[0]
    x, y = pyprocess.get_rect_coordinates(shape, window_size, overlap)
    n_rows, n_cols = pyprocess.get_field_shape(shape, window_size, overlap)
    u, v, s2n = _peaks(_mean_corr(window_size, overlap, None), n_rows, n_cols)
    excluded = _roi_on_grid(roi_mask, x, y)
    u = np.ma.masked_array(u, mask=excluded)
    v = np.ma.masked_array(v, mask=excluded)
    flags = validation.typical_validation(u, v, s2n, s)
    replaced_final = False

    if n_passes == 1:
        if s.replace_vectors and flags.any() and not flags.all():
            u, v = _replace(u, v, flags)
            replaced_final = True
        if s.smoothn:
            u, v = _smooth(u, v, excluded)
    else:
        if flags.any() and not flags.all():
            u, v = _replace(u, v, flags)
        if s.smoothn:
            u, v = _smooth(u, v, excluded)
        for i in range(1, n_passes):
            window_size, overlap = s.windowsizes[i], s.overlap[i]
            x_new, y_new = pyprocess.get_rect_coordinates(shape, window_size, overlap)
            ip_u = RectBivariateSpline(y[:, 0], x[0, :], np.ma.filled(u, 0.0))
            ip_v = RectBivariateSpline(y[:, 0], x[0, :], np.ma.filled(v, 0.0))
            u_pre = ip_u(y_new[:, 0], x_new[0, :])
            v_pre = ip_v(y_new[:, 0], x_new[0, :])
            # the deformation coordinates depend only on the ENSEMBLE predictor, so
            # they are built once per pass and re-used for every pair (symmetric split)
            xg, yg, ut, vt = windef.create_deformation_field(
                head, x_new, y_new, u_pre, v_pre,
                interpolation_order=s.interpolation_order)
            coords = ((yg - vt / 2, xg - ut / 2), (yg + vt / 2, xg + ut / 2))
            n_rows, n_cols = pyprocess.get_field_shape(shape, window_size, overlap)
            du, dv, s2n = _peaks(_mean_corr(window_size, overlap, coords),
                                 n_rows, n_cols)
            u = du + u_pre
            v = dv + v_pre
            x, y = x_new, y_new
            excluded = _roi_on_grid(roi_mask, x, y)
            u = np.ma.masked_array(u, mask=excluded)
            v = np.ma.masked_array(v, mask=excluded)
            flags = validation.typical_validation(u, v, s2n, s)
            final = i == n_passes - 1
            if (not final or s.replace_vectors) and flags.any() and not flags.all():
                u, v = _replace(u, v, flags)
                if final:
                    replaced_final = True
            if not final and s.smoothn:
                u, v = _smooth(u, v, excluded)

    return _package_result(x, y, u, v, s2n, flags, excluded, replaced_final,
                           eff_ws, eff_ov, trimmed, s, voxel_size_um,
                           extra_diag={"ensemble_pairs": n_pairs,
                                       "pairing": pairing})
