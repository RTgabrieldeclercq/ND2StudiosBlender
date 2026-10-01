"""ALDVC adapter conformance — does `kernels/aldvc_field.py` hand pyALDVC's answer
back in THIS repo's axis order, units and schedule?

WHY THIS EXISTS, AND WHY IT IS NOT WHAT IT USED TO BE
    Until 2026-09-25 this script validated a clean-room in-repo port of FranckLab's
    MATLAB ALDVC against the paper's benchmark, FranckLab's distributed
    `results_*.mat`, and an x-only invariant. That port is gone: the solver is now
    the official pyALDVC package (`al-dvc`, github.com/zachtong/pyALDVC), which
    carries its own numerical validation against the same MATLAB reference. Re-running
    the paper's benchmark here would be measuring UPSTREAM'S code, which upstream
    already measures.

    What is unvalidated by upstream — and is the only new code on the compute path —
    is the ADAPTER: the `[x,y,z] -> [z,y,x]` reversal, the strain relabelling, the
    physical-units pass-through, and the frame schedule. Every one of those is a
    silent-wrong-answer failure: they produce a field of exactly the right shape,
    full of finite numbers, that is transposed, negated or mis-scaled. `selftest`
    catches a gross axis swap because its fixture shifts all three axes by different
    amounts; it does NOT catch a strain transpose (the fixture's strain is ~0), nor a
    voxel-size rescale that is off by the anisotropy ratio.

    So this script checks the adapter against analytic truth, using pyALDVC's own
    `al_dvc.synthetic` generators, with fixtures chosen so that each failure mode is
    individually observable:

    A. AXIS ISOLATION — translate along exactly one axis; the other two components
       of the returned displacement must be zero. Catches any reversal error.
    B. STRAIN RELABELLING, ASYMMETRIC — a displacement gradient whose `du/dy` and
       `dv/dx` DIFFER, so a transpose of the tensor is visible. A symmetric fixture
       (a pure stretch, or the selftest's rigid shift) cannot see it. This is the
       fixture that proves `_strain_tensor_zyx`'s hand-written relabelling.
    C. SIGN, ANTISYMMETRIC — `du/dy = +g`, `dv/dx = -g`. pyALDIC negates its two 2D
       cross-terms and `dic_correlate` has to undo it; this asserts that pyALDVC does
       NOT, so nobody "fixes" a bug that is not there by analogy with the sibling.
    D. ANISOTROPIC VOXELS — the same deformation solved with cubic and with
       non-cubic voxels. Strain must pick up the `voxel_i/voxel_j` cross-axis
       rescale; displacement must NOT (it stays in voxels). Catches a voxel_size
       that is dropped, applied twice, or applied in `(z,y,x)` order to a `DVCPara`
       that wants `(x,y,z)`.
    E. SCHEDULE — accumulative vs incremental `ref_indices` over a 4-frame series.
       Catches an off-by-one in the frame pairing and proves the reference is reused
       rather than re-solved per pair.

    A tolerance here is NOT upstream's tolerance. Upstream's suite asks "is the
    solver accurate?"; this one asks "did the adapter move the numbers?", so the
    bars are tight — an adapter error is O(1), never O(1e-3).

RUN
    PYTHONUTF8=1 python scripts/_aldvc_validate.py             # all five groups
    PYTHONUTF8=1 python scripts/_aldvc_validate.py --quick     # A, C only (fast)
    PYTHONUTF8=1 python scripts/_aldvc_validate.py --size 96   # bigger volume

Manual — NOT part of `nodegraph.selftest`: it runs a real solver several times and
takes tens of seconds. The selftest's own DVC group covers the wiring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodegraph.kernels.aldvc_field import (           # noqa: E402
    al_dvc_available, run_aldvc, run_aldvc_series,
)

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "[ok] " if ok else "[FAIL] "
    print(tag + name + (("  — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def _close(a: float, b: float, tol: float) -> bool:
    return bool(abs(float(a) - float(b)) <= tol)


def _fixture(size: int, seed: int = 0):
    """A bead volume + the ``(x, y, z)`` centre to deform about."""
    from al_dvc import synthetic as syn
    shape = (size, size, size)
    ref = syn.generate_bead_volume(shape, n_beads=int(0.024 * size ** 3),
                                   radius=1.8, seed=seed)
    centre = (size / 2.0, size / 2.0, size / 2.0)
    return ref, shape, centre


def _params(**over):
    p = dict(subset_size=24, subset_spacing=12, admm_iterations=3,
             init_guess="pyramid", n_threads=0)
    p.update(over)
    return p


def _interior(res, trim: int = 1):
    """Boolean mask over the node grid with ``trim`` nodes peeled off every face.

    The outermost ring of nodes is where the global step's Neumann boundary and the
    strain window's incomplete support both live, so edge nodes carry a real,
    documented bias. Judging an ADAPTER on them would measure upstream's edge
    handling instead.
    """
    g = res.grid_coords.shape[:3]
    m = np.zeros(g, dtype=bool)
    if min(g) <= 2 * trim:
        m[...] = True
        return m
    m[trim:-trim, trim:-trim, trim:-trim] = True
    return m


def _med(arr, mask):
    return float(np.nanmedian(np.asarray(arr)[mask]))


# ── A. axis isolation ────────────────────────────────────────────────────────────

def group_axis_isolation(size: int) -> None:
    from al_dvc import synthetic as syn
    print("\n=== A. axis isolation — a one-axis shift must land on one component ===")
    ref, shape, _c = _fixture(size)
    # (x, y, z) translation -> which (dz, dy, dx) component must carry it
    cases = [("x", (2.5, 0.0, 0.0), 2), ("y", (0.0, 2.0, 0.0), 1),
             ("z", (0.0, 0.0, 1.5), 0)]
    for axis, t, comp in cases:
        dfm = syn.warp_volume_lagrangian(ref, syn.affine_displacement(t=t))
        r = run_aldvc(ref, dfm, voxel_size_um=(1.0, 1.0, 1.0), params=_params())
        m = _interior(r)
        d = r.displacement_field
        got = [_med(d[..., k], m) for k in range(3)]
        truth = max(t)  if max(t) > 0 else min(t)
        ok = _close(got[comp], truth, 0.05) and all(
            abs(got[k]) < 0.05 for k in range(3) if k != comp)
        check(f"A/{axis}: shift {truth:+.1f} vox on {axis} -> (dz,dy,dx)", ok,
              f"got {np.round(got, 4).tolist()}, expected component {comp}")


# ── B. strain relabelling, asymmetric ────────────────────────────────────────────

def group_strain_relabel(size: int) -> None:
    from al_dvc import synthetic as syn
    print("\n=== B. strain relabelling — an ASYMMETRIC gradient exposes a transpose ===")
    ref, shape, centre = _fixture(size)
    # du_x/dy = 0.030 and du_y/dx = 0.010: different, so [y,x] != [x,y] would show,
    # and the symmetric tensor entry is their mean.
    F = np.zeros((3, 3)); F[0, 1] = 0.030; F[1, 0] = 0.010
    dfm = syn.warp_volume_lagrangian(ref, syn.affine_displacement(F=F, centre=centre))
    r = run_aldvc(ref, dfm, voxel_size_um=(1.0, 1.0, 1.0), params=_params())
    m = _interior(r)
    s = r.strain_field
    # this repo indexes (z, y, x): x is 2, y is 1, z is 0
    e_xy = _med(s[..., 2, 1], m)
    e_yx = _med(s[..., 1, 2], m)
    e_xx = _med(s[..., 2, 2], m)
    e_zz = _med(s[..., 0, 0], m)
    expect = 0.5 * (0.030 + 0.010)
    check("B/shear: strain[x,y] == mean(du_x/dy, du_y/dx)",
          _close(e_xy, expect, 0.002), f"got {e_xy:.5f}, expected {expect:.5f}")
    check("B/symmetry: strain[x,y] == strain[y,x]",
          _close(e_xy, e_yx, 1e-9), f"{e_xy:.6f} vs {e_yx:.6f}")
    check("B/no-leak: normal terms stay ~0 for a pure shear",
          abs(e_xx) < 0.003 and abs(e_zz) < 0.003,
          f"exx={e_xx:.5f} ezz={e_zz:.5f}")
    # a uniaxial x-stretch must land on [x,x], NOT on [z,z] — the reversal's own test
    F2 = np.zeros((3, 3)); F2[0, 0] = 0.040
    dfm2 = syn.warp_volume_lagrangian(ref, syn.affine_displacement(F=F2, centre=centre))
    r2 = run_aldvc(ref, dfm2, voxel_size_um=(1.0, 1.0, 1.0), params=_params())
    m2 = _interior(r2)
    sxx = _med(r2.strain_field[..., 2, 2], m2)
    szz = _med(r2.strain_field[..., 0, 0], m2)
    check("B/stretch: a +4% x-stretch lands on strain[x,x], not strain[z,z]",
          _close(sxx, 0.040, 0.004) and abs(szz) < 0.004,
          f"exx={sxx:.5f} ezz={szz:.5f}")


# ── C. sign, antisymmetric ───────────────────────────────────────────────────────

def group_sign(size: int) -> None:
    from al_dvc import synthetic as syn
    print("\n=== C. sign — pyALDVC does NOT negate cross-terms (pyALDIC does) ===")
    ref, shape, centre = _fixture(size)
    g = 0.020
    F = np.zeros((3, 3)); F[0, 1] = g; F[1, 0] = -g       # antisymmetric = rotation
    dfm = syn.warp_volume_lagrangian(ref, syn.affine_displacement(F=F, centre=centre))
    r = run_aldvc(ref, dfm, voxel_size_um=(1.0, 1.0, 1.0), params=_params())
    m = _interior(r)
    # A rigid rotation has ZERO infinitesimal shear strain: the antisymmetric parts
    # cancel. If the adapter negated ONE cross-term, they would ADD instead and this
    # would read +/- g. That is the whole point of the fixture.
    e_xy = _med(r.strain_field[..., 2, 1], m)
    check("C/rotation: antisymmetric gradient gives ~0 shear strain (no negation)",
          abs(e_xy) < 0.003, f"strain[x,y] = {e_xy:.6f}, must be ~0 not {g:+.3f}")
    # ...and the displacement field itself still rotates the right way: at a point
    # +X of centre, u_y = -g*dx (from F[1,0] = -g).
    coords = r.grid_coords                                  # [z, y, x] voxels
    dxs = coords[..., 2] - centre[0]
    dy_pred = -g * dxs
    dy_got = r.displacement_field[..., 1]
    err = float(np.nanmax(np.abs((dy_got - dy_pred)[m])))
    check("C/handedness: u_y == -g*(x-cx) across the grid",
          err < 0.05, f"max |err| = {err:.4f} vox")


# ── D. anisotropic voxels ────────────────────────────────────────────────────────

def group_anisotropic(size: int) -> None:
    from al_dvc import synthetic as syn
    print("\n=== D. anisotropic voxels — strain rescales, displacement does not ===")
    ref, shape, centre = _fixture(size)
    # du_x/dz = 0.030 only: a cross-axis term between the two axes whose voxel sizes
    # differ, so the v_x/v_z rescale is directly observable.
    F = np.zeros((3, 3)); F[0, 2] = 0.030
    dfm = syn.warp_volume_lagrangian(ref, syn.affine_displacement(F=F, centre=centre))
    iso = run_aldvc(ref, dfm, voxel_size_um=(1.0, 1.0, 1.0), params=_params())
    # (z, y, x) = (0.5, 0.2, 0.2) um: a confocal-like 2.5x axial anisotropy
    vz, vy, vx = 0.5, 0.2, 0.2
    ani = run_aldvc(ref, dfm, voxel_size_um=(vz, vy, vx), params=_params())
    m = _interior(iso)

    # displacement stays in VOXELS regardless of voxel_size
    for k, nm in enumerate(("dz", "dy", "dx")):
        a, b = _med(iso.displacement_field[..., k], m), _med(ani.displacement_field[..., k], m)
        check(f"D/disp {nm} is voxel-valued (unchanged by voxel_size)",
              _close(a, b, 0.02), f"iso {a:.4f} vs ani {b:.4f}")

    # strain[x,z] picks up v_x/v_z; strain[z,x] picks up v_z/v_x. For a gradient with
    # only du_x/dz nonzero, the symmetric entry is half of each contribution, so the
    # ratio of the anisotropic to the isotropic entry is exactly v_x/v_z.
    e_iso = _med(iso.strain_field[..., 2, 0], m)            # [x, z]
    e_ani = _med(ani.strain_field[..., 2, 0], m)
    ratio = e_ani / e_iso if abs(e_iso) > 1e-6 else float("nan")
    check("D/strain[x,z] scales by v_x/v_z",
          _close(ratio, vx / vz, 0.08),
          f"iso {e_iso:.5f} ani {e_ani:.5f} ratio {ratio:.3f}, expected {vx / vz:.3f}")
    # a LATERAL-only shear must be untouched by an axial voxel size
    F2 = np.zeros((3, 3)); F2[0, 1] = 0.030
    d2 = syn.warp_volume_lagrangian(ref, syn.affine_displacement(F=F2, centre=centre))
    i2 = run_aldvc(ref, d2, voxel_size_um=(1.0, 1.0, 1.0), params=_params())
    a2 = run_aldvc(ref, d2, voxel_size_um=(vz, vy, vx), params=_params())
    m2 = _interior(i2)
    check("D/strain[x,y] is unchanged (both axes lateral, v_x/v_y == 1)",
          _close(_med(i2.strain_field[..., 2, 1], m2),
                 _med(a2.strain_field[..., 2, 1], m2), 0.002),
          f"iso {_med(i2.strain_field[..., 2, 1], m2):.5f} "
          f"ani {_med(a2.strain_field[..., 2, 1], m2):.5f}")


# ── E. frame schedule ────────────────────────────────────────────────────────────

def group_schedule(size: int) -> None:
    from al_dvc import synthetic as syn
    print("\n=== E. frame schedule — accumulative vs incremental pairing ===")
    ref, shape, _c = _fixture(size)
    step = 0.9                                              # voxels per frame, along x
    vols = [ref] + [syn.warp_volume_lagrangian(
        ref, syn.affine_displacement(t=(step * k, 0.0, 0.0))) for k in (1, 2, 3)]

    acc = run_aldvc_series(lambda i: vols[i], 4, shape,
                           voxel_size_um=(1.0, 1.0, 1.0), params=_params(),
                           ref_indices=(0, 0, 0))
    got = [_med(r.displacement_field[..., 2], _interior(r)) for r in acc]
    want = [step, 2 * step, 3 * step]
    check("E/accumulative: every frame against frame 0 -> k*step",
          len(acc) == 3 and all(_close(g, w, 0.06) for g, w in zip(got, want)),
          f"got {np.round(got, 3).tolist()}, expected {want}")
    check("E/accumulative: all three pairs report ref_frame 0",
          all(r.diagnostics.get("ref_frame") == 0 for r in acc),
          str([r.diagnostics.get("ref_frame") for r in acc]))

    inc = run_aldvc_series(lambda i: vols[i], 4, shape,
                           voxel_size_um=(1.0, 1.0, 1.0), params=_params(),
                           ref_indices=(0, 1, 2))
    goti = [_med(r.displacement_field[..., 2], _interior(r)) for r in inc]
    check("E/incremental: each frame against its predecessor -> step each time",
          len(inc) == 3 and all(_close(g, step, 0.06) for g in goti),
          f"got {np.round(goti, 3).tolist()}, expected {[step] * 3}")
    check("E/incremental: ref_frame walks 0,1,2",
          [r.diagnostics.get("ref_frame") for r in inc] == [0, 1, 2],
          str([r.diagnostics.get("ref_frame") for r in inc]))
    # the grid is shared across the series (one mesh built once)
    same_grid = all(np.array_equal(acc[0].grid_coords, r.grid_coords) for r in acc[1:])
    check("E/shared grid: every frame lands on the same node coordinates", same_grid,
          f"{acc[0].grid_coords.shape[:3]} nodes, identical across all "
          f"{len(acc)} frames" if same_grid else "grid_coords DIFFER between frames")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=80,
                    help="cube edge in voxels (default 80; must be >= 48)")
    ap.add_argument("--quick", action="store_true",
                    help="only groups A and C")
    args = ap.parse_args()

    if not al_dvc_available():
        print("al-dvc is not installed — nothing to validate.\n"
              "  pip install al-dvc")
        return 2
    size = max(48, int(args.size))
    import al_dvc
    print(f"pyALDVC {al_dvc.__version__} | fixture {size}^3 bead volume")

    group_axis_isolation(size)
    group_sign(size)
    if not args.quick:
        group_strain_relabel(size)
        group_anisotropic(size)
        group_schedule(size)

    print()
    if FAILS:
        print(f"ALDVC ADAPTER VALIDATION FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print("   " + f)
        return 1
    print("ALL ALDVC ADAPTER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
