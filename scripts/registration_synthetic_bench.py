"""registration_synthetic_bench — known-answer validation of ``nodegraph.kernels.registration``.

Four synthetic worlds, each with an EXACT per-frame ground-truth motion:

  1. **star field**   (2-D)  sparse Gaussian points, rendered analytically from the moved
                              catalogue — no interpolation anywhere in the truth
  2. **star volume**  (3-D)  the same in (Z, Y, X) with a 3-D drift, to probe what a node
                              that estimates on ONE mid-z plane and applies (dy, dx) to every
                              plane does when the sample also drifts in z — and what the 3-D
                              estimate recovers
  3. **deforming mass**      a textured soft disk pushed by a smooth, growing, NON-affine
                              displacement field (a one-sided bulge) plus optional isotropic
                              growth and rigid drift
  4. **affine mass**         the same disk under rotation / translation / shear / scale
  5. **landmarks**           simulated user clicks (true correspondences + click jitter) →
                              ``estimate_from_landmarks`` picks the family and seeds the series

One metric throughout: the **RMS endpoint error (EPE)** of the recovered per-frame transform
against the true displacement field, in pixels, over the support (the frame interior for
stars, the disk for masses).  Next to it the **family floor** — the EPE of the best
least-squares fit of that transform family to the true field — so "this model cannot
represent the motion" (floor ≫ 0) is kept apart from "the algorithm failed to find the
parameters" (achieved ≫ floor).

``--legacy`` re-creates the pre-2026-10-07 kernel behaviour (phase-whitened correlation, no
low-pass, the mirrored ECC seed, no feature seeding) so the before/after can be reproduced.

Run:  python scripts/registration_synthetic_bench.py [--quick] [--legacy] [--only 1,4] [--out DIR]
Green line:  "ALL REGISTRATION BENCH CASES PASSED"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodegraph.kernels import registration as reg  # noqa: E402

FAILS: List[str] = []
ROWS: List[dict] = []
LEGACY = False


# ═══════════════════════════════════════════════════════════════════════════════════
# 1. Motion models — forward maps reference → frame t, in (y, x) pixel coordinates
# ═══════════════════════════════════════════════════════════════════════════════════

def affine_yx(angle_deg: float = 0.0, shift: Tuple[float, float] = (0.0, 0.0),
              shear: float = 0.0, scale: float = 1.0,
              centre: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """3×3 homogeneous forward map in (y, x): rotate about ``centre`` by ``angle_deg``
    (positive = +x toward +y, i.e. clockwise on screen), shear x by ``shear``·(y−cy), scale
    about the centre, then translate by ``shift``."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, s], [-s, c]])
    Sh = np.array([[1.0, 0.0], [shear, 1.0]])      # x' = x + shear·y
    L = scale * (R @ Sh)
    cy, cx = centre
    cvec = np.array([cy, cx])
    t = cvec - L @ cvec + np.asarray(shift, dtype=float)
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = t
    return M


def apply_h(M: np.ndarray, pts_yx: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts_yx, np.ones((len(pts_yx), 1))], axis=1)
    return (M @ p.T).T[:, :2]


_P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def yx_to_cv(M_yx: np.ndarray) -> np.ndarray:
    """(y, x) homogeneous → the 2×3 (x, y) warp the kernel / cv2 use (reference → moving)."""
    return (_P @ M_yx @ _P)[:2, :].astype(np.float32)


def cv_to_yx(W: np.ndarray) -> np.ndarray:
    H3 = np.eye(3)
    H3[:2, :] = W
    return _P @ H3 @ _P


@dataclass
class Truth:
    disp: np.ndarray          # (T, H, W, 2) forward displacement, (dy, dx)
    support: np.ndarray       # (H, W) bool
    affine: Optional[List[np.ndarray]] = None   # (T) 3×3 (y,x) when the motion IS affine
    shift3d: Optional[np.ndarray] = None        # (T, 3) for the volume scenario
    drift: Optional[np.ndarray] = None          # (T, 2) the rigid component (deforming mass)


def grid_yx(H: int, W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return np.stack([yy, xx], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════════
# 2. Generators
# ═══════════════════════════════════════════════════════════════════════════════════

def _noise(rng, img: np.ndarray, read_sigma: float) -> np.ndarray:
    """Poisson shot noise (image is in photo-electrons) + Gaussian read noise."""
    out = rng.poisson(np.clip(img, 0, None)).astype(np.float32)
    if read_sigma > 0:
        out += rng.normal(0.0, read_sigma, img.shape).astype(np.float32)
    return out


def render_stars(pos_yx: np.ndarray, amp: np.ndarray, shape: Tuple[int, int],
                 sigma: float, bg: float) -> np.ndarray:
    """Analytic Gaussian PSFs at sub-pixel positions (exact — no resampling)."""
    H, W = shape
    img = np.full((H, W), bg, dtype=np.float64)
    r = int(math.ceil(4 * sigma))
    for (py, px), a in zip(pos_yx, amp):
        y0, y1 = int(math.floor(py)) - r, int(math.floor(py)) + r + 1
        x0, x1 = int(math.floor(px)) - r, int(math.floor(px)) + r + 1
        if y1 <= 0 or x1 <= 0 or y0 >= H or x0 >= W:
            continue
        ya, yb = max(0, y0), min(H, y1)
        xa, xb = max(0, x0), min(W, x1)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        img[ya:yb, xa:xb] += a * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
    return img


def star_field(T: int, H: int, W: int, n_stars: int, motions: Sequence[np.ndarray],
               *, sigma: float = 1.5, amp_range=(300.0, 3000.0), bg: float = 100.0,
               read_sigma: float = 10.0, seed: int = 0, margin: int = 40,
               ) -> Tuple[np.ndarray, Truth]:
    """``motions[t]`` is the 3×3 (y,x) forward map of frame t (motions[0] = identity).
    The catalogue extends ``margin`` px past the frame so stars enter and leave."""
    rng = np.random.default_rng(seed)
    area = (H + 2 * margin) * (W + 2 * margin) / (H * W)
    n_cat = int(round(n_stars * area))
    pos0 = np.stack([rng.uniform(-margin, H + margin, n_cat),
                     rng.uniform(-margin, W + margin, n_cat)], axis=1)
    amp = np.exp(rng.uniform(math.log(amp_range[0]), math.log(amp_range[1]), n_cat))
    series = np.empty((T, H, W), dtype=np.float32)
    g = grid_yx(H, W).reshape(-1, 2)
    disp = np.zeros((T, H, W, 2))
    for t in range(T):
        pos_t = apply_h(motions[t], pos0)
        series[t] = _noise(rng, render_stars(pos_t, amp, (H, W), sigma, bg), read_sigma)
        disp[t] = (apply_h(motions[t], g) - g).reshape(H, W, 2)
    sup = np.zeros((H, W), bool)
    sup[16:H - 16, 16:W - 16] = True
    return series, Truth(disp=disp, support=sup, affine=list(motions))


def render_stars_3d(pos_zyx, amp, shape, sig_z, sig_xy, bg):
    Z, H, W = shape
    vol = np.full((Z, H, W), bg, dtype=np.float64)
    rz, r = int(math.ceil(3 * sig_z)), int(math.ceil(3.5 * sig_xy))
    for (pz, py, px), a in zip(pos_zyx, amp):
        z0, z1 = int(math.floor(pz)) - rz, int(math.floor(pz)) + rz + 1
        y0, y1 = int(math.floor(py)) - r, int(math.floor(py)) + r + 1
        x0, x1 = int(math.floor(px)) - r, int(math.floor(px)) + r + 1
        za, zb = max(0, z0), min(Z, z1)
        ya, yb = max(0, y0), min(H, y1)
        xa, xb = max(0, x0), min(W, x1)
        if za >= zb or ya >= yb or xa >= xb:
            continue
        zz, yy, xx = np.mgrid[za:zb, ya:yb, xa:xb]
        vol[za:zb, ya:yb, xa:xb] += a * np.exp(
            -((zz - pz) ** 2) / (2 * sig_z ** 2) - ((yy - py) ** 2 + (xx - px) ** 2) / (2 * sig_xy ** 2))
    return vol


def star_volume(T: int, Z: int, H: int, W: int, n_stars: int, step_zyx: Tuple[float, float, float],
                *, sig_z: float = 2.0, sig_xy: float = 1.5, amp_range=(300.0, 3000.0),
                bg: float = 100.0, read_sigma: float = 10.0, seed: int = 0, margin: int = 24,
                ) -> Tuple[np.ndarray, Truth]:
    """(T, Z, H, W) with a pure 3-D translation of ``step_zyx`` per frame."""
    rng = np.random.default_rng(seed)
    mz = max(4, int(abs(step_zyx[0]) * T) + 3)
    n_cat = int(round(n_stars * ((Z + 2 * mz) / Z) * ((H + 2 * margin) * (W + 2 * margin) / (H * W))))
    pos0 = np.stack([rng.uniform(-mz, Z + mz, n_cat),
                     rng.uniform(-margin, H + margin, n_cat),
                     rng.uniform(-margin, W + margin, n_cat)], axis=1)
    amp = np.exp(rng.uniform(math.log(amp_range[0]), math.log(amp_range[1]), n_cat))
    series = np.empty((T, Z, H, W), dtype=np.float32)
    sh = np.zeros((T, 3))
    for t in range(T):
        sh[t] = np.asarray(step_zyx) * t
        series[t] = _noise(rng, render_stars_3d(pos0 + sh[t], amp, (Z, H, W), sig_z, sig_xy, bg), read_sigma)
    disp = np.zeros((T, H, W, 2))
    disp[:, :, :, 0] = sh[:, 1][:, None, None]
    disp[:, :, :, 1] = sh[:, 2][:, None, None]
    sup = np.zeros((H, W), bool)
    sup[16:H - 16, 16:W - 16] = True
    return series, Truth(disp=disp, support=sup, shift3d=sh)


def textured_disk(H: int, W: int, radius: float, *, seed: int = 0, bg: float = 100.0,
                  peak: float = 1200.0, texture_sigma: float = 2.0, edge: float = 4.0,
                  centre: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """A soft-edged disk filled with speckle texture; returns (image, support mask)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    cy, cx = centre if centre is not None else ((H - 1) / 2.0, (W - 1) / 2.0)
    g = grid_yx(H, W)
    r = np.hypot(g[..., 0] - cy, g[..., 1] - cx)
    disk = 1.0 / (1.0 + np.exp((r - radius) / edge))
    tex = gaussian_filter(rng.random((H, W)), texture_sigma)
    tex = (tex - tex.min()) / np.ptp(tex)
    img = bg + peak * disk * (0.35 + 0.65 * tex)
    return img, (r < radius)


def _inverse_displacement(fwd: Callable[[np.ndarray], np.ndarray], H: int, W: int,
                          iters: int = 12) -> np.ndarray:
    """For an output pixel p, find q with q + D(q) = p (fixed-point) → source coordinates."""
    g = grid_yx(H, W).reshape(-1, 2)
    q = g.copy()
    for _ in range(iters):
        q = g - fwd(q)
    return q.reshape(H, W, 2)


def render_through(base: np.ndarray, src_yx: np.ndarray, bg: float) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    return map_coordinates(base, [src_yx[..., 0], src_yx[..., 1]], order=3,
                           mode="constant", cval=bg)


def mass_series(T: int, H: int, W: int, *, radius: float = 70.0, seed: int = 0,
                affine_per_frame: Optional[Callable[[int], np.ndarray]] = None,
                bulge_amp: float = 0.0, bulge_sigma: float = 28.0,
                bulge_dir: Tuple[float, float] = (0.0, 1.0),
                bulge_centre_frac: Tuple[float, float] = (0.0, 0.55),
                growth: float = 0.0, drift: Tuple[float, float] = (0.0, 0.0),
                read_sigma: float = 10.0, bg: float = 100.0, peak: float = 1200.0,
                ) -> Tuple[np.ndarray, Truth]:
    """Textured disk under either an exact affine (``affine_per_frame(t)`` → 3×3 (y,x))
    or a smooth non-rigid field: ``bulge_amp``·t px Gaussian push, ``growth``·t isotropic
    dilation about the centre, ``drift``·t rigid translation."""
    rng = np.random.default_rng(seed + 1000)
    base, sup = textured_disk(H, W, radius, seed=seed, bg=bg, peak=peak)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    series = np.empty((T, H, W), dtype=np.float32)
    g = grid_yx(H, W)
    disp = np.zeros((T, H, W, 2))
    affs: Optional[List[np.ndarray]] = [] if affine_per_frame is not None else None
    drifts = np.zeros((T, 2))
    bc = np.array([cy + bulge_centre_frac[0] * radius, cx + bulge_centre_frac[1] * radius])
    bd = np.asarray(bulge_dir, float)
    bd = bd / (np.linalg.norm(bd) + 1e-12)
    for t in range(T):
        if affine_per_frame is not None:
            M = affine_per_frame(t)
            affs.append(M)
            disp[t] = (apply_h(M, g.reshape(-1, 2)) - g.reshape(-1, 2)).reshape(H, W, 2)
            src = apply_h(np.linalg.inv(M), g.reshape(-1, 2)).reshape(H, W, 2)
        else:
            a, e, d = bulge_amp * t, growth * t, np.asarray(drift) * t
            drifts[t] = d

            def fwd(p, a=a, e=e, d=d):
                r2 = ((p - bc) ** 2).sum(axis=1)
                bul = a * np.exp(-r2 / (2 * bulge_sigma ** 2))[:, None] * bd[None, :]
                return bul + e * (p - np.array([cy, cx])) + d[None, :]
            disp[t] = fwd(g.reshape(-1, 2)).reshape(H, W, 2)
            src = _inverse_displacement(fwd, H, W) if t > 0 else g
        img = base if t == 0 else render_through(base, src, bg)
        series[t] = _noise(rng, img, read_sigma)
    return series, Truth(disp=disp, support=sup, affine=affs,
                         drift=(None if affine_per_frame is not None else drifts))


# ═══════════════════════════════════════════════════════════════════════════════════
# 3. Metric: endpoint error of a recovered transform against the true field
# ═══════════════════════════════════════════════════════════════════════════════════

def disp_from_bundle(tf: dict, t: int, H: int, W: int) -> np.ndarray:
    """The forward displacement field (dy, dx) the kernel's frame-t transform implies."""
    g = grid_yx(H, W)
    if tf.get("warps") is None:
        sh = np.asarray(tf["shifts"][t], float)[-2:]      # applied to moving → aligns onto ref
        return np.broadcast_to(-sh, (H, W, 2)).copy()       # content moved by −shift
    M = cv_to_yx(np.asarray(tf["warps"][t], float))        # reference → moving, (y,x)
    return (apply_h(M, g.reshape(-1, 2)) - g.reshape(-1, 2)).reshape(H, W, 2)


def epe(d_est: np.ndarray, d_true: np.ndarray, support: np.ndarray) -> float:
    e = d_est[support] - d_true[support]
    return float(np.sqrt((e ** 2).sum(axis=1).mean()))


def family_floor(d_true: np.ndarray, support: np.ndarray, family: str, n: int = 4000,
                 seed: int = 0) -> float:
    """EPE of the best least-squares member of ``family`` fitted to the true field."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(support)
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)
    H, W = support.shape
    g = grid_yx(H, W).reshape(-1, 2)
    src = g[pick]
    dst = src + d_true.reshape(-1, 2)[pick]
    if family == "translation":
        fit = src + (dst - src).mean(axis=0)
    else:
        from skimage.transform import estimate_transform
        name = {"euclidean": "euclidean", "similarity": "similarity",
                "affine": "affine", "feature": "affine"}[family]
        tf = estimate_transform(name, src[:, ::-1], dst[:, ::-1])
        fit = tf(src[:, ::-1])[:, ::-1]
    e = fit - dst
    return float(np.sqrt((e ** 2).sum(axis=1).mean()))


# ═══════════════════════════════════════════════════════════════════════════════════
# 4. Runners — the kernel, with an optional re-creation of its pre-fix behaviour
# ═══════════════════════════════════════════════════════════════════════════════════

_ecc_orig = reg.ecc_align


def _ecc_mirrored(reference, moving, model="euclidean", init_shift=None, **kw):
    """The vendored seed: +shift where the warp convention needs −shift."""
    if init_shift is not None and kw.get("init_warp") is None:
        init_shift = -np.asarray(init_shift, float)
    return _ecc_orig(reference, moving, model=model, init_shift=init_shift, **kw)


def run_kernel(series: np.ndarray, model: str, reference: str, **kw) -> dict:
    if LEGACY:
        kw.setdefault("correlation", "phase")
        kw.setdefault("lowpass_sigma", 0.0)
        kw.setdefault("seed_from_features", False)
        reg.ecc_align = _ecc_mirrored
        try:
            return reg.estimate_series(series, model=model, reference=reference, **kw)
        finally:
            reg.ecc_align = _ecc_orig
    kw.setdefault("lowpass_sigma", 1.0)          # the node's default
    return reg.estimate_series(series, model=model, reference=reference, **kw)


def score(name: str, series: np.ndarray, truth: Truth, tf: dict, family: str,
          tol: float, *, note: str = "", t_sec: float = 0.0, assert_pass: bool = True,
          tags: Optional[dict] = None) -> dict:
    T, H, W = series.shape[0], series.shape[-2], series.shape[-1]
    errs = []
    for t in range(1, T):
        errs.append(epe(disp_from_bundle(tf, t, H, W), truth.disp[t], truth.support))
    errs = np.asarray(errs)
    floor = float(np.mean([family_floor(truth.disp[t], truth.support, family) for t in range(1, T)]))
    conf = float(np.mean(tf["confidence"][1:]))
    row = dict(name=name, epe_mean=float(errs.mean()), epe_max=float(errs.max()),
               floor=floor, frac_ok=float((errs < 0.5).mean()), conf=conf,
               gated=int(tf["gated"].sum()), t_sec=t_sec, note=note, tol=tol,
               legacy=LEGACY, **(tags or {}))
    # the rigid-drift bias: how far the recovered translation at the mass centre is from
    # the true drift (deforming mass only) — the error that matters for drift correction
    if truth.drift is not None:
        cy, cx = H // 2, W // 2
        bias = [np.linalg.norm(disp_from_bundle(tf, t, H, W)[cy, cx] - truth.drift[t])
                for t in range(1, T)]
        row["drift_bias"] = float(np.mean(bias))
    ok = row["epe_mean"] <= tol
    status = "ok" if ok else ("FAIL" if assert_pass else "info")
    if not ok and assert_pass:
        FAILS.append(name)
    extra = f"  drift-bias {row['drift_bias']:.3f}" if "drift_bias" in row else ""
    print(f"[{status:>4}] {name:<60} EPE {row['epe_mean']:7.3f} px (max {row['epe_max']:7.3f}, "
          f"floor {floor:6.3f}, tol {tol:g})  ok<0.5px {row['frac_ok']:4.0%}  conf {conf:5.2f}"
          f"{'  gated=' + str(row['gated']) if row['gated'] else ''}{extra}"
          f"{'  ' + note if note else ''}")
    ROWS.append(row)
    return row


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def motions_from(T: int, fn: Callable[[int], np.ndarray]) -> List[np.ndarray]:
    return [fn(t) for t in range(T)]


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 100 - len(title)))


# ═══════════════════════════════════════════════════════════════════════════════════
# 5. Scenarios
# ═══════════════════════════════════════════════════════════════════════════════════

def scenario_star_field(quick: bool) -> None:
    section("1. STAR FIELD (2-D) — translation drift, sparse → dense")
    T, H, W = (8 if quick else 12), 256, 256
    step = np.array([1.3, -0.7])
    for n_stars in ([12, 60] if quick else [6, 12, 30, 60, 250]):
        series, truth = star_field(T, H, W, n_stars, motions_from(T, lambda t: affine_yx(shift=tuple(step * t))),
                                   seed=n_stars)
        for model, ref, tol in (("translation", "first", 0.15), ("translation", "previous", 0.35),
                                ("translation", "template", 0.35)):
            tf, dt = timed(run_kernel, series, model, ref)
            score(f"stars n={n_stars:<3} {model}/{ref}", series, truth, tf, "translation",
                  tol=tol, t_sec=dt, assert_pass=(n_stars >= 12), tags=dict(fig="stars", x=n_stars))
        for model, ref in (("euclidean", "first"), ("affine", "first"), ("feature", "first")):
            tf, dt = timed(run_kernel, series, model, ref)
            score(f"stars n={n_stars:<3} {model}/{ref}", series, truth, tf, model,
                  tol=0.3, t_sec=dt, assert_pass=(model != "feature" and n_stars >= 12),
                  tags=dict(fig="stars", x=n_stars))

    section("1b. STAR FIELD — SNR ladder (n=40, read-noise σ ↑, star amplitude ↓)")
    for amp_hi, rs in ([(300, 30)] if quick else [(3000, 10), (800, 20), (300, 30), (150, 40), (80, 60), (50, 80)]):
        series, truth = star_field(T, H, W, 40, motions_from(T, lambda t: affine_yx(shift=tuple(step * t))),
                                   amp_range=(amp_hi / 4, amp_hi), read_sigma=rs, seed=7)
        peak_snr = amp_hi / math.sqrt(100 + amp_hi + rs ** 2)
        for ref in ("first", "previous"):
            tf, dt = timed(run_kernel, series, "translation", ref)
            score(f"stars SNR≈{peak_snr:5.1f} translation/{ref}", series, truth, tf, "translation",
                  tol=0.3, t_sec=dt, assert_pass=False, tags=dict(fig="snr", x=peak_snr))

    section("1c. STAR FIELD — big jumps (fraction of the frame per frame)")
    Tj = 5
    for frac in ([0.25] if quick else [0.1, 0.25, 0.4, 0.55]):
        st = np.array([frac * H, -0.3 * frac * W])
        series, truth = star_field(Tj, H, W, 80, motions_from(Tj, lambda t: affine_yx(shift=tuple(st * t))),
                                   seed=3, margin=int(0.6 * H * Tj))
        for ref in ("previous", "first"):
            tf, dt = timed(run_kernel, series, "translation", ref)
            score(f"stars jump {frac:.2f}·frame/frame translation/{ref}", series, truth, tf,
                  "translation", tol=0.3, t_sec=dt, assert_pass=False, tags=dict(fig="jump", x=frac))


def scenario_star_volume(quick: bool) -> None:
    section("2. STAR VOLUME (3-D) — the mid-z-plane (2D) estimate vs the 3-D estimate under z drift")
    T, Z, H, W = (6 if quick else 10), 16, 160, 160
    for step in ([(0.0, 1.0, -0.6), (0.5, 1.0, -0.6)] if quick
                 else [(0.0, 1.0, -0.6), (0.25, 1.0, -0.6), (0.5, 1.0, -0.6), (1.0, 1.0, -0.6)]):
        series, truth = star_volume(T, Z, H, W, 120, step, seed=11)
        mid = series[:, Z // 2]
        # (a) the 2D node path: estimate on the mid plane, apply (dy,dx) everywhere
        tf, dt = timed(run_kernel, mid, "translation", "first")
        score(f"volume dz={step[0]:.2f}/frame  2D lever (mid-plane) translation/first",
              mid, truth, tf, "translation", tol=0.3, t_sec=dt, assert_pass=False,
              tags=dict(fig="volume", x=step[0], path="2D"))
        z_err = float(np.abs(truth.shift3d[1:, 0]).mean())
        print(f"        ↳ axial drift left uncorrected by the 2D lever: mean |dz| = {z_err:.2f} planes")
        # (b) the 3D lever: estimate_series on the (T, Z, H, W) volume → (dz, dy, dx)
        tf3, dt3 = timed(run_kernel, series, "translation", "first")
        e3 = np.sqrt(((-tf3["shifts"][1:] - truth.shift3d[1:]) ** 2).sum(axis=1))
        ez = np.abs(-tf3["shifts"][1:, 0] - truth.shift3d[1:, 0])
        row = score(f"volume dz={step[0]:.2f}/frame  3D lever translation/first (lateral part)",
                    mid, truth, tf3, "translation", tol=0.3, t_sec=dt3, assert_pass=True,
                    tags=dict(fig="volume", x=step[0], path="3D"))
        row["z_err"] = float(ez.mean())
        row["epe3d"] = float(e3.mean())
        # Judged only while the cumulative axial drift stays within a quarter of the stack:
        # past that, content has left the volume at one end and no estimator can see it —
        # that is the non-registrable condition, reported rather than asserted.
        cum_dz = abs(float(truth.shift3d[-1, 0]))
        within = cum_dz <= 0.25 * Z
        ok = ez.mean() < 0.6
        status = "ok" if ok else ("FAIL" if within else "info")
        print(f"[{status:>4}]         ↳ axial error of the 3D lever: mean |dz err| = {ez.mean():.2f} planes "
              f"(3-D EPE {e3.mean():.2f}; z precision is set by {Z} planes vs σz=2"
              f"{'' if within else f'; cumulative dz {cum_dz:.1f} of {Z} planes — content left the stack'})")
        if not ok and within:
            FAILS.append(f"volume 3D dz={step[0]}")


def scenario_deforming_mass(quick: bool) -> None:
    section("3. SLOWLY DEFORMING MASS — one-sided bulge (non-affine) + growth + drift")
    T, H, W = (8 if quick else 12), 256, 256
    cases = [
        dict(bulge_amp=0.4, growth=0.0, drift=(0.0, 0.0), label="bulge 0.4px/f"),
        dict(bulge_amp=0.4, growth=0.0, drift=(0.6, 0.3), label="bulge 0.4px/f + drift (0.6,0.3)"),
        dict(bulge_amp=0.0, growth=0.004, drift=(0.6, 0.3), label="growth 0.4%/f + drift"),
        dict(bulge_amp=1.2, growth=0.004, drift=(0.6, 0.3), label="bulge 1.2px/f + growth + drift"),
        dict(bulge_amp=3.0, growth=0.0, drift=(0.6, 0.3), label="bulge 3.0px/f + drift"),
    ]
    if quick:
        cases = cases[1:2] + cases[3:4]
    for c in cases:
        label = c.pop("label")
        series, truth = mass_series(T, H, W, seed=5, **c)
        for model, ref in (("translation", "first"), ("translation", "previous"),
                           ("euclidean", "first"), ("affine", "first"), ("affine", "previous"),
                           ("feature", "first")):
            tf, dt = timed(run_kernel, series, model, ref)
            score(f"mass {label:<34} {model}/{ref}", series, truth, tf, model,
                  tol=0.5, t_sec=dt, assert_pass=False, tags=dict(fig="deform", x=label))
        # the remedy: estimate in a region that does NOT deform (the half of the disk away
        # from the bulge) — a rect ROI, as the node's `region` socket supplies
        roi = {"kind": "rect", "x": 40, "y": 40, "w": 70, "h": 176}
        tf, dt = timed(run_kernel, series, "translation", "first", roi=roi)
        score(f"mass {label:<34} translation/first + rigid-region ROI", series, truth, tf,
              "translation", tol=0.5, t_sec=dt, assert_pass=False, tags=dict(fig="deform", x=label))


def scenario_affine_mass(quick: bool) -> None:
    section("4. MASS ROTATING / TRANSLATING / SKEWING — exact affine motion")
    T, H, W = (8 if quick else 12), 256, 256
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    cases = [
        ("translate (1.0,-0.5)/f", lambda t: affine_yx(shift=(1.0 * t, -0.5 * t), centre=(cy, cx))),
        ("rotate 0.5°/f", lambda t: affine_yx(angle_deg=0.5 * t, centre=(cy, cx))),
        ("rotate 2°/f", lambda t: affine_yx(angle_deg=2.0 * t, centre=(cy, cx))),
        ("rotate 2°/f + translate", lambda t: affine_yx(angle_deg=2.0 * t, shift=(1.0 * t, -0.5 * t), centre=(cy, cx))),
        ("rotate 6°/f", lambda t: affine_yx(angle_deg=6.0 * t, centre=(cy, cx))),
        ("rotate 12°/f", lambda t: affine_yx(angle_deg=12.0 * t, centre=(cy, cx))),
        ("shear 0.01/f", lambda t: affine_yx(shear=0.01 * t, centre=(cy, cx))),
        ("scale 1%/f + rotate 1°/f", lambda t: affine_yx(angle_deg=1.0 * t, scale=1.01 ** t, centre=(cy, cx))),
        ("rotate 1°/f + shear 0.01/f + translate", lambda t: affine_yx(angle_deg=1.0 * t, shear=0.01 * t, shift=(0.8 * t, 0.4 * t), centre=(cy, cx))),
    ]
    if quick:
        cases = [cases[0], cases[2], cases[4], cases[8]]
    for label, fn in cases:
        series, truth = mass_series(T, H, W, seed=9, affine_per_frame=fn)
        for model, ref in (("translation", "first"), ("euclidean", "first"), ("euclidean", "previous"),
                           ("affine", "first"), ("affine", "previous"), ("feature", "first")):
            tf, dt = timed(run_kernel, series, model, ref)
            score(f"mass {label:<40} {model}/{ref}", series, truth, tf, model,
                  tol=0.5, t_sec=dt, assert_pass=False, tags=dict(fig="affine", x=label))


def scenario_landmarks(quick: bool) -> None:
    section("5. LANDMARKS — simulated clicks (true correspondences + jitter) → family, warp, seeded series")
    T, H, W = (8 if quick else 12), 256, 256
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rng = np.random.default_rng(2024)

    def clicks(truth: Truth, t: int, n: int, jitter: float):
        """n random points inside the support on frame 0 and their true places on frame t,
        each perturbed by a Gaussian click error of ``jitter`` px."""
        idx = np.flatnonzero(truth.support)
        pick = rng.choice(idx, size=n, replace=False)
        src = grid_yx(H, W).reshape(-1, 2)[pick]
        dst = src + truth.disp[t].reshape(-1, 2)[pick]
        return src + rng.normal(0, jitter, src.shape), dst + rng.normal(0, jitter, dst.shape)

    worlds = [
        ("rotate 6°/f (42° by t=7)", mass_series(T, H, W, seed=9, affine_per_frame=lambda t: affine_yx(angle_deg=6.0 * t, centre=(cy, cx))), "euclidean"),
        ("rotate 1°/f + shear + translate", mass_series(T, H, W, seed=9, affine_per_frame=lambda t: affine_yx(angle_deg=1.0 * t, shear=0.01 * t, shift=(0.8 * t, 0.4 * t), centre=(cy, cx))), "affine"),
        ("translate only", mass_series(T, H, W, seed=9, affine_per_frame=lambda t: affine_yx(shift=(1.0 * t, -0.5 * t), centre=(cy, cx))), "translation"),
        ("deforming: bulge 1.2px/f + drift", mass_series(T, H, W, seed=5, bulge_amp=1.2, growth=0.004, drift=(0.6, 0.3)), "nonrigid"),
    ]
    tk = T - 1
    grid = [(5, 1.0)] if quick else [(3, 1.0), (5, 0.3), (5, 1.0), (5, 2.0), (8, 1.0)]
    for label, (series, truth), expect in worlds:
        for n, jitter in grid:
            src, dst = clicks(truth, tk, n, jitter)
            lm = reg.estimate_from_landmarks(src, dst, model="auto")
            d_lm = (apply_h(cv_to_yx(lm["warp"]), grid_yx(H, W).reshape(-1, 2)) - grid_yx(H, W).reshape(-1, 2)).reshape(H, W, 2)
            e_direct = epe(d_lm, truth.disp[tk], truth.support)
            picked_ok = (lm["model"] == expect) or (expect == "nonrigid" and lm["nonrigid"]) \
                or (expect == "affine" and lm["model"] in ("similarity", "affine"))
            # What the clicks can be expected to resolve (the F-test is honest about noise):
            # drift vs rotation from 5 pairs at 1 px; shear / scale against rotation only from
            # sub-pixel clicks or ~8 pairs — 5 pairs at 1 px leave 4 dof for a 2-parameter
            # difference and (correctly) decline to call it. Asserted inside that envelope.
            resolvable = (n >= 5 and jitter <= 0.5) or (n >= 8 and jitter <= 1.0) or \
                (n >= 5 and jitter <= 1.0 and expect in ("translation", "euclidean"))
            if picked_ok:
                mark = "✓"
            elif expect == "nonrigid":
                mark = (f"– not flagged: the {n} random clicks did not land on the bulge, and "
                        f"{jitter:.1f} px clicks can only reveal deformation above "
                        f"~{lm['threshold']:.1f} px")
            elif not resolvable:
                mark = (f"– {expect} not resolvable from {n} clicks at {jitter:.1f} px (needs "
                        f"sub-pixel clicks or ~8 pairs); the simpler family is the honest call")
            else:
                mark = "✗ expected " + expect
            print(f"  {label:<34} n={n} click σ={jitter:.1f}: auto → {lm['model']:<11} "
                  f"{'NON-RIGID ' if lm['nonrigid'] else ''}res {lm['residual']:5.2f} px  "
                  f"direct-warp EPE@t={tk} {e_direct:6.3f}  {mark}")
            if not picked_ok and expect != "nonrigid" and resolvable:
                FAILS.append(f"landmarks {label} n={n} σ={jitter} picked {lm['model']}")
            # the seeded series: model=auto, first reference
            if expect != "nonrigid":
                land = {"t": tk, "src": src.tolist(), "dst": dst.tolist()}
                tf, dt = timed(run_kernel, series, "auto", "first", landmarks=land)
                fam = tf["model"]
                score(f"  ↳ series model=auto({fam})/first seeded by {n} clicks σ={jitter:.1f}", series, truth, tf,
                      fam, tol=0.3, t_sec=dt, assert_pass=bool(resolvable),
                      tags=dict(fig="landmarks", x=f"{label} n={n} σ={jitter}"))
        if expect == "nonrigid":
            print(f"  {label:<34} ↳ a NON-RIGID verdict is the useful answer here: no global model will do better "
                  f"than the family floor (see §3); register a rigid region instead.")


# ═══════════════════════════════════════════════════════════════════════════════════
# 6. Figures + main
# ═══════════════════════════════════════════════════════════════════════════════════

def make_figures(out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print("no matplotlib; skipping figures:", exc)
        return
    rows = [r for r in ROWS if "fig" in r]
    tag = "legacy" if LEGACY else "fixed"

    def _sub(fig_key):
        return [r for r in rows if r["fig"] == fig_key]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # stars: EPE vs n
    ax = axes[0, 0]
    for key, mk in (("translation/first", "o"), ("translation/previous", "s"), ("euclidean/first", "^"),
                    ("affine/first", "v"), ("feature/first", "x")):
        pts = sorted((r["x"], r["epe_mean"]) for r in _sub("stars") if r["name"].endswith(key))
        if pts:
            ax.plot([p[0] for p in pts], [max(p[1], 1e-3) for p in pts], marker=mk, label=key)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("stars per frame"); ax.set_ylabel("EPE (px)")
    ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_title(f"1. star field — {tag}"); ax.legend(fontsize=8)
    # snr
    ax = axes[0, 1]
    for key, mk in (("translation/first", "o"), ("translation/previous", "s")):
        pts = sorted((r["x"], r["epe_mean"]) for r in _sub("snr") if r["name"].endswith(key))
        if pts:
            ax.plot([p[0] for p in pts], [max(p[1], 1e-3) for p in pts], marker=mk, label=key)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("peak SNR of the brightest stars"); ax.set_ylabel("EPE (px)")
    ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_title("1b. SNR ladder (40 stars)"); ax.legend(fontsize=8)
    # affine mass: bar per case × model
    ax = axes[1, 0]
    aff = _sub("affine")
    labels = []
    for r in aff:
        if r["x"] not in labels:
            labels.append(r["x"])
    models = ["translation/first", "euclidean/first", "euclidean/previous", "affine/first", "affine/previous", "feature/first"]
    wbar = 0.8 / len(models)
    for i, mdl in enumerate(models):
        vals = []
        for lab in labels:
            v = [r["epe_mean"] for r in aff if r["x"] == lab and r["name"].endswith(mdl)]
            vals.append(max(v[0], 1e-3) if v else np.nan)
        ax.bar(np.arange(len(labels)) + i * wbar, vals, width=wbar, label=mdl)
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(labels)) + 0.4); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_ylabel("EPE (px)"); ax.set_title("4. affine mass"); ax.legend(fontsize=7, ncol=2)
    # deform: achieved vs floor
    ax = axes[1, 1]
    dfm = _sub("deform")
    labels = []
    for r in dfm:
        if r["x"] not in labels:
            labels.append(r["x"])
    models = ["translation/first", "translation/previous", "euclidean/first", "affine/first",
              "translation/first + rigid-region ROI"]
    wbar = 0.8 / len(models)
    for i, mdl in enumerate(models):
        vals = []
        for lab in labels:
            v = [r for r in dfm if r["x"] == lab and r["name"].endswith(mdl)]
            vals.append(v[0].get("drift_bias", np.nan) if v else np.nan)
        ax.bar(np.arange(len(labels)) + i * wbar, vals, width=wbar, label=mdl)
    ax.set_xticks(np.arange(len(labels)) + 0.4); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_ylabel("error of the recovered DRIFT at the mass centre (px)")
    ax.set_title("3. deforming mass — how much the deformation pulls the drift estimate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, f"registration_bench_{tag}.png")
    fig.savefig(path, dpi=130)
    print("figure:", path)


def main() -> int:
    global LEGACY
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="re-create the pre-2026-10-07 kernel behaviour")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="comma list of scenario numbers, e.g. 1,4")
    args = ap.parse_args()
    LEGACY = bool(args.legacy)
    if LEGACY:
        print("LEGACY MODE: phase-whitened correlation, no low-pass, mirrored ECC seed, no feature seeding")
    t0 = time.time()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = {"1": scenario_star_field, "2": scenario_star_volume,
            "3": scenario_deforming_mass, "4": scenario_affine_mass, "5": scenario_landmarks}
    for k, fn in todo.items():
        if not only or k in only:
            fn(args.quick)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"registration_bench_rows_{'legacy' if LEGACY else 'fixed'}.json"), "w") as f:
            json.dump(ROWS, f, indent=1, default=str)
        make_figures(args.out)
    print(f"\n{len(FAILS)} failures in {time.time() - t0:.1f} s")
    if FAILS:
        print("FAILED:", ", ".join(FAILS))
        return 1 if not LEGACY else 0
    print("ALL REGISTRATION BENCH CASES PASSED" if not LEGACY else "(legacy run — failures expected and not asserted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
