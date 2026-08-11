"""piv_synthetic_bench — known-answer validation of ``nodegraph.kernels.piv_field``.

Replicates the character of openpiv's own shipped suite (``openpiv/test/test_process.py``:
dense random-noise texture, scipy sub-pixel shifts, 0.25 px tolerance) with FIXED seeds,
then goes past it the same way ``scripts/dic_synthetic_bench.py`` does for pyALDIC —
analytic rotation/shear/sinusoid fields, a noise probe, and a parity assertion against
``openpiv.windef.simple_multipass`` (the upstream reference driver our kernel re-implements
in order to keep the final-pass signal-to-noise field).

Upstream's tolerances are LOOSE (0.25 px, majority-of-trials logic) — per this repo's
policy the bench reports the measured error and asserts against tighter bounds chosen
from the measured floor, not upstream's.

Run:  python scripts/piv_synthetic_bench.py            (~20 s CPU)
Green line:  "ALL PIV BENCH CASES PASSED"
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodegraph.kernels.piv_field import openpiv_available, run_piv_pair  # noqa: E402

H = W = 256
MARGIN = 32                       # interior margin, px (windows near edges excluded)
PARAMS = {"windowsizes": (64, 32), "overlap": 0.5}
FAILS: list = []


def _speckle(seed: int = 42, sigma: float = 1.5) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.random((H, W)), sigma)
    return (a - a.min()) / np.ptp(a) * 250.0


def _warp(frame: np.ndarray, u_fn, v_fn) -> np.ndarray:
    """Eulerian warp: B(y, x) = A(y - v(y,x), x - u(y,x)) — a feature at (y, x) in A
    lands at ~(y + v, x + u) in B for smooth, small fields."""
    from scipy.ndimage import map_coordinates

    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return map_coordinates(frame, [yy - v_fn(yy, xx), xx - u_fn(yy, xx)],
                           order=3, mode="nearest")


def _interior(res):
    g = res.grid_coords
    keep = ((g[..., 0] >= MARGIN) & (g[..., 0] <= H - MARGIN)
            & (g[..., 1] >= MARGIN) & (g[..., 1] <= W - MARGIN))
    return g[keep], res.displacement_field[keep]


def _case(name: str, res, truth_fn, tol: float) -> None:
    """Worst-component RMSE over interior windows vs the analytic truth."""
    g, d = _interior(res)
    ty, tx = truth_fn(g[:, 0], g[:, 1])
    ok = np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])
    rmse_y = float(np.sqrt(np.mean((d[ok, 0] - ty[ok]) ** 2)))
    rmse_x = float(np.sqrt(np.mean((d[ok, 1] - tx[ok]) ** 2)))
    worst = max(rmse_y, rmse_x)
    status = "ok" if worst < tol else "FAIL"
    if status == "FAIL":
        FAILS.append(name)
    print(f"[{status:>4}] {name:<34} worst RMSE {worst:8.4f} px  (tol {tol})  "
          f"n={int(ok.sum())} dropped={int((~ok).sum())}")


def main() -> int:
    if not openpiv_available():
        print("SKIP: openpiv not installed (pip install openpiv)")
        return 0
    t0 = time.time()
    A = _speckle()
    vox = (1.0, 1.0)

    # ── 1. zero (self-pair): the noise floor ───────────────────────────────────
    r = run_piv_pair(A, A.copy(), vox, PARAMS)
    _case("zero / self-pair", r, lambda y, x: (0 * y, 0 * x), tol=0.01)

    # ── 2. integer roll (3, 5) ─────────────────────────────────────────────────
    r = run_piv_pair(A, np.roll(A, (3, 5), axis=(0, 1)), vox, PARAMS)
    _case("integer translation (3, 5)", r, lambda y, x: (0 * y + 3, 0 * x + 5), tol=0.05)

    # ── 3. sub-pixel translation (2.5, -1.25) ──────────────────────────────────
    from scipy.ndimage import shift as ndshift

    r = run_piv_pair(A, ndshift(A, (2.5, -1.25), order=3, mode="nearest"), vox, PARAMS)
    _case("sub-pixel translation (2.5, -1.25)", r,
          lambda y, x: (0 * y + 2.5, 0 * x - 1.25), tol=0.06)

    # ── 4. large translation (0, 11) through a 3-pass ladder ───────────────────
    r = run_piv_pair(A, ndshift(A, (0.0, 11.0), order=3, mode="nearest"), vox,
                     {"windowsizes": (128, 64, 32), "overlap": 0.5})
    _case("large translation (0, 11), 3-pass", r,
          lambda y, x: (0 * y, 0 * x + 11), tol=0.06)

    # ── 5. rigid rotation 1 deg about the centre ───────────────────────────────
    th = np.deg2rad(1.0)
    cy = cx = (H - 1) / 2.0

    def _rot_u(y, x):
        return (np.cos(th) - 1) * (x - cx) - np.sin(th) * (y - cy)

    def _rot_v(y, x):
        return np.sin(th) * (x - cx) + (np.cos(th) - 1) * (y - cy)

    r = run_piv_pair(A, _warp(A, _rot_u, _rot_v), vox, PARAMS)
    _case("rotation 1 deg", r, lambda y, x: (_rot_v(y, x), _rot_u(y, x)), tol=0.08)

    # ── 6. simple shear du_x/dy = 0.02 ─────────────────────────────────────────
    r = run_piv_pair(A, _warp(A, lambda y, x: 0.02 * (y - cy), lambda y, x: 0 * x), vox,
                     PARAMS)
    _case("shear du_x/dy = 0.02", r, lambda y, x: (0 * y, 0.02 * (y - cy)), tol=0.08)

    # ── 7. sinusoidal shear, lambda = 64 px, amp 2 px (spatial-resolution probe) ─
    lam, amp = 64.0, 2.0

    def _sin_u(y, x):
        return amp * np.sin(2 * np.pi * y / lam)

    # Two sub-cases, both documented physics rather than defects (cf. dic bench case 12):
    # (a) with the median test OFF, the measured RMSE is the pure window-averaging
    #     attenuation — a 32 px top-hat window on lambda=64 px passes sinc(pi*32/64)=0.637
    #     of the amplitude, predicting RMSE = (1-0.637)*amp/sqrt(2) = 0.51 px, and the
    #     64 px coarse pass sees exactly ZERO of it (sinc(pi)=0), measured 0.64;
    # (b) with the DEFAULT universal median test, steep REAL gradients get flagged
    #     (~58 of 225 vectors on clean data) and replacement flattens them further —
    #     the exact failure mode the median_test="off" choice exists for.
    B_sin = _warp(A, _sin_u, lambda y, x: 0 * x)
    r = run_piv_pair(A, B_sin, vox, dict(PARAMS, median_test="off"))
    _case("sinusoid lam=64 amp=2 (median off)", r,
          lambda y, x: (0 * y, _sin_u(y, x)), tol=0.75)
    r = run_piv_pair(A, B_sin, vox, PARAMS)
    _case("sinusoid lam=64 amp=2 (default val.)", r,
          lambda y, x: (0 * y, _sin_u(y, x)), tol=1.35)

    # ── 8. sub-pixel + 5% gaussian noise on both frames ────────────────────────
    rng = np.random.default_rng(3)
    An = A + rng.normal(0, 12.5, A.shape)
    Bn = ndshift(A, (2.5, -1.25), order=3, mode="nearest") + rng.normal(0, 12.5, A.shape)
    r = run_piv_pair(An, Bn, vox, PARAMS)
    _case("sub-pixel + 5% noise", r, lambda y, x: (0 * y + 2.5, 0 * x - 1.25), tol=0.15)

    # ── 9. upstream test_process.py replica: dense noise, wrap shift, their tol ─
    from scipy.ndimage import shift as _sh

    rng = np.random.default_rng(1234)
    fa = (rng.random((64, 64)) * 255)
    fb = _sh(fa, (2.5, -3.5), mode="wrap")                 # their (v, u) convention
    r = run_piv_pair(fa, fb, vox, {"windowsizes": (32,), "overlap": 0.5,
                                   "median_test": "off", "std_threshold": 1e9})
    d = r.displacement_field.reshape(-1, 2)
    err_y = float(np.nanmax(np.abs(d[:, 0] - 2.5)))
    err_x = float(np.nanmax(np.abs(d[:, 1] + 3.5)))
    status = "ok" if max(err_y, err_x) < 0.25 else "FAIL"   # upstream's own THRESHOLD
    if status == "FAIL":
        FAILS.append("upstream replica")
    print(f"[{status:>4}] {'upstream create_pair replica':<34} worst |err| "
          f"{max(err_y, err_x):8.4f} px  (tol 0.25 = upstream THRESHOLD)")

    # ── 10. parity vs openpiv.windef.simple_multipass (the reference driver) ────
    from openpiv import windef
    from openpiv.settings import PIVSettings

    Bs = ndshift(A, (2.5, -1.25), order=3, mode="nearest")
    s = PIVSettings()
    s.windowsizes = (64, 32)
    s.overlap = (32, 16)
    s.num_iterations = 2
    x3, y3, u3, v3, _fl = windef.simple_multipass(A, Bs, s)
    ours = run_piv_pair(A, Bs, vox, dict(PARAMS, median_test="classic",
                                         median_threshold=3, max_disp_px=30))
    du = float(np.nanmax(np.abs(ours.displacement_field[..., 1] - u3)))
    dv = float(np.nanmax(np.abs(ours.displacement_field[..., 0] - (-v3))))
    status = "ok" if max(du, dv) < 1e-9 else "FAIL"
    if status == "FAIL":
        FAILS.append("windef parity")
    print(f"[{status:>4}] {'windef.simple_multipass parity':<34} max |delta| "
          f"{max(du, dv):8.2e} px  (tol 1e-9; y-flip/v-negation undone)")

    print(f"\n{len(FAILS)} failures in {time.time() - t0:.1f} s")
    if FAILS:
        print("FAILED:", ", ".join(FAILS))
        return 1
    print("ALL PIV BENCH CASES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
