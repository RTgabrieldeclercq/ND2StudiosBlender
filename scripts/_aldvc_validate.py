"""ALDVC validation suite — does `aldvc_field.run_aldvc` reproduce the paper?

WHY THIS EXISTS
    `nodegraph.selftest` proves the DVC node *wires up*. It does not prove the
    kernel measures displacement correctly. This script does, against three
    independent kinds of truth:

    A. SYNTHETIC (exact truth, machine precision) — the paper's own homogeneous
       benchmark (Fig. 2 / Table 4): x-translation 0…1 vox, uniaxial stretch
       λ = 1.00…1.30, in-plane rotation 0…25°, on Appendix-D Gaussian-PSF bead
       volumes built by `_aldvc_synth`. Reports RMS displacement + strain error
       exactly as paper Eq. (13), so the numbers are directly comparable to the
       paper's published curves.

    B. REFERENCE-DATASET PARITY (truth = FranckLab's own MATLAB output) — the
       distributed ALDVC example data (SEM Challenge Sample 14 volumes and the
       uniaxial-stretch volumes) together with the `results_*.mat` files the
       MATLAB code produced from them. Same input, same nominal parameters, so
       the Python field should land on the MATLAB field. This is the only test
       that can catch a *self-consistent* port that is uniformly wrong.

    C. INVARIANT (truth known without any reference) — for a deformation imposed
       purely along x, the y- and z-displacements are identically zero, so
       RMS(u_y), RMS(u_z) is an absolute accuracy measure on the real datasets
       even where no analytic field is available.

CONVENTIONS
    MATLAB ALDVC indexes volumes and fields as (x, y, z) with column-major
    flattening; this kernel uses numpy (z, y, x). Every MATLAB array crossing into
    this script is converted once, in `matlab_vol` / `matlab_disp`, and never again.

Usage
    python scripts/_aldvc_validate.py --suite synth          # A (self-contained)
    python scripts/_aldvc_validate.py --suite parity         # B (needs the data)
    python scripts/_aldvc_validate.py --suite all
    python scripts/_aldvc_validate.py --suite synth --quick  # smaller/faster
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _aldvc_synth import (  # noqa: E402
    Deformation, exact_field_on_grid, interior_mask, make_pair, rms_error,
    rotation_z, translation, uniaxial_stretch,
)

DATA = Path(r"C:\Users\McGheeLab - Analysis\Downloads")
ALDVC_REPO = DATA / "ALDVC-master" / "ALDVC-master"
DVC_IMAGES = ALDVC_REPO / "DVC_images"
EXTRACTED = DATA / "aldvc_data"


# ──────────────────────────────────────────────────────────────────────
#  MATLAB ↔ numpy bridges
# ──────────────────────────────────────────────────────────────────────

def matlab_vol(path: Path) -> np.ndarray:
    """Load a MATLAB ALDVC `vol{1}` matfile → numpy ``(z, y, x)``.

    MATLAB stores the stack as (x, y, z) (see `DVC_images/GenerateVolMatfile.m`:
    ``vol{1} = uint8(permute(voltemp,[2,1,3]))``), so the transpose to kernel
    order is (2, 1, 0).
    """
    import scipy.io as sio
    d = sio.loadmat(str(path))
    v = d["vol"]
    a = v[0, 0] if v.dtype == object else v
    return np.ascontiguousarray(np.transpose(np.asarray(a), (2, 1, 0)))


def tiff_stack(folder: Path, pattern: str = "*.tif") -> np.ndarray:
    """Load a z-ordered TIFF slice folder → numpy ``(z, y, x)``.

    `DVC_images/GenerateImgTiff.m` writes ``squeeze(permute(vol,[2,1,3])(:,:,z))``,
    i.e. each slice is already (y, x), so stacking on a new leading axis lands
    directly in kernel order.
    """
    import tifffile
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {folder}")
    return np.stack([tifffile.imread(str(f)) for f in files], axis=0)


def matlab_disp(U: np.ndarray, mnl_xyz: tuple[int, int, int]) -> np.ndarray:
    """MATLAB ``U`` (interleaved ``[ux,uy,uz]`` per node, x-fastest grid) →
    ``(3, Gz, Gy, Gx)`` in kernel component order ``(z, y, x)``.

    ``mnl_xyz`` is the MATLAB grid size ``(Mx, My, Mz)``. MATLAB nodes are ordered
    column-major over (x, y, z), so reshaping with order="F" recovers the (x,y,z)
    grid; transposing to (z, y, x) and reversing the component order gives kernel
    layout.
    """
    U = np.asarray(U, dtype=np.float64).ravel()
    comps = [U[c::3].reshape(mnl_xyz, order="F") for c in range(3)]  # each (Mx,My,Mz)
    out = [np.transpose(c, (2, 1, 0)) for c in comps]                # → (Mz,My,Mx)
    return np.stack(out[::-1], axis=0)                               # (uz,uy,ux)


def load_reference(path: Path) -> dict:
    """Read a MATLAB ALDVC ``results_*.mat`` into a plain dict of numpy arrays."""
    import scipy.io as sio
    d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    para = d["DVCpara"]
    mesh = d["DVCmesh"]
    xyz0 = mesh.xyz0
    mnl = tuple(int(s) for s in np.asarray(xyz0.x).shape)   # (Mx, My, Mz)
    disp = np.atleast_1d(d["ResultDisp"])
    grad = np.atleast_1d(d["ResultDefGrad"])
    mubeta = np.atleast_1d(d["Resultmubeta"])

    def _pick(rec, *names):
        for nm in names:
            if hasattr(rec, nm):
                return np.asarray(getattr(rec, nm), dtype=np.float64)
        return np.asarray(rec.U, dtype=np.float64)

    return dict(
        winsize=np.asarray(para.winsize, dtype=int),
        winstep=np.asarray(para.winstepsize, dtype=int),
        img_size_xyz=tuple(int(s) for s in np.asarray(para.ImgSize)),
        mnl_xyz=mnl,
        coords_xyz=np.asarray(mesh.coordinatesFEM, dtype=np.float64),
        files=[str(f) for f in np.atleast_1d(d["file_name"])],
        U=[np.asarray(r.U, dtype=np.float64) for r in disp],
        U_local=[_pick(r, "ULocalICGN", "U_local_ICGN") for r in disp],
        U0=[_pick(r, "U0", "U0_crosscorr") for r in disp],
        F=[np.asarray(r.F, dtype=np.float64) for r in grad],
        beta=[float(m.beta) for m in mubeta],
        mu=[float(m.mu) for m in mubeta],
        icgn_tol=float(getattr(para, "ICGNtol", 1e-2)),
    )


def resample_to(u_src: np.ndarray, coords_src: list[np.ndarray],
                coords_dst: np.ndarray) -> np.ndarray:
    """Trilinearly resample ``(3, *src_grid)`` onto ``coords_dst`` ``(*dst, 3)``.

    The Python grid and the MATLAB grid need not coincide node-for-node (different
    border insetting), so parity is measured after putting both on the *Python*
    nodes. Linear interpolation of an already-smooth compatible field adds error
    well below the differences being measured.
    """
    from scipy.interpolate import RegularGridInterpolator
    pts = np.asarray(coords_dst, dtype=np.float64).reshape(-1, 3)
    out = np.empty((3, pts.shape[0]), dtype=np.float64)
    for c in range(3):
        f = RegularGridInterpolator(coords_src, u_src[c], method="linear",
                                    bounds_error=False, fill_value=None)
        out[c] = f(pts)
    return out.reshape(3, *coords_dst.shape[:-1])


# ──────────────────────────────────────────────────────────────────────
#  Suite A — synthetic, exact truth (paper Fig. 2 / Table 4)
# ──────────────────────────────────────────────────────────────────────

def run_case(ref, dfm, deform: Deformation, params: dict, voxel=(1.0, 1.0, 1.0)):
    """Run the kernel on one pair and score it against the exact field."""
    from nodegraph.kernels.aldvc_field import run_aldvc
    t0 = time.perf_counter()
    res = run_aldvc(ref, dfm, voxel_size_um=voxel, params=params)
    dt = time.perf_counter() - t0

    u_num = np.asarray(res.displacement_field)                 # (*grid, 3)
    u_ex = exact_field_on_grid(deform, res.grid_coords)        # (*grid, 3)
    mask = interior_mask(res.grid_coords, ref.shape, params["subset_size"])

    per_axis = [rms_error(u_num[mask][..., c], u_ex[mask][..., c]) for c in range(3)]
    rms_u = rms_error(u_num[mask], u_ex[mask])

    rms_e = float("nan")
    e_meas = float("nan")
    if res.strain_field is not None and deform.exact_strain is not None:
        s_num = np.asarray(res.strain_field)[mask]              # (n, 3, 3)
        s_ex = np.broadcast_to(deform.exact_strain, s_num.shape)
        rms_e = rms_error(s_num, s_ex)
        e_meas = float(np.mean(s_num[..., 2, 2]))               # measured e_xx
    return dict(
        case=deform.name, label=deform.label, seconds=round(dt, 2),
        grid=list(res.grid_coords.shape[:-1]), n_nodes=int(mask.sum()),
        rms_u=rms_u, rms_ux=per_axis[2], rms_uy=per_axis[1], rms_uz=per_axis[0],
        rms_strain=rms_e, exx_measured=e_meas,
        exx_exact=(float(deform.exact_strain[2, 2])
                   if deform.exact_strain is not None else float("nan")),
        median_zncc=res.diagnostics.get("median_zncc"),
        beta=res.beta, admm_iters=res.iterations, converged=res.converged,
    )


def suite_synth(shape, params, *, quick: bool, warp: bool, seed: int = 3) -> list[dict]:
    """Paper's homogeneous-deformation sweep on synthetic bead volumes."""
    c = [s / 2.0 for s in shape]
    if quick:
        cases = [translation(0.0), translation(0.5), translation(1.0),
                 uniaxial_stretch(1.05, c), uniaxial_stretch(1.20, c),
                 rotation_z(5.0, c), rotation_z(15.0, c)]
    else:
        cases = ([translation(t) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
                 + [uniaxial_stretch(l, c)
                    for l in (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30)]
                 + [rotation_z(a, c) for a in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)])
    rows = []
    for d in cases:
        ref, dfm, _ = make_pair(shape, d, seed=seed, warp=warp)
        r = run_case(ref, dfm, d, params)
        rows.append(r)
        print(f"  {r['case']:<11} {r['label']:<16} "
              f"RMS|u|={r['rms_u']:.4f}  (ux={r['rms_ux']:.4f} uy={r['rms_uy']:.4f} "
              f"uz={r['rms_uz']:.4f})  RMSe={r['rms_strain']:.2e}  "
              f"exx={r['exx_measured']:+.5f}/{r['exx_exact']:+.5f}  "
              f"zncc={r['median_zncc']:.3f}  {r['seconds']}s", flush=True)
    return rows


# ──────────────────────────────────────────────────────────────────────
#  Suite B — parity against FranckLab's own MATLAB output
# ──────────────────────────────────────────────────────────────────────

def suite_parity(*, quick: bool, crop: int | None, workers: int) -> list[dict]:
    rows = []
    for spec in _parity_specs():
        if not spec["ok"]:
            print(f"  SKIP {spec['name']}: {spec['why']}")
            continue
        rows.append(_run_parity(spec, quick=quick, crop=crop, workers=workers))
    return rows


def _parity_specs() -> list[dict]:
    s14_ref = DVC_IMAGES / "vol_Sample14_1001.mat"
    s14_def = DVC_IMAGES / "vol_Sample14_1002.mat"
    s14_res = EXTRACTED / "Sample14_noise5" / "results_S14_ws20_st10.mat"
    st_ref = EXTRACTED / "vol_stretch_1001_tiff"
    st_def = EXTRACTED / "vol_stretch_1002_tiff"
    st_res = EXTRACTED / "Uniaxial_stretch" / "results_uniaxial_stretch_ws30_st10.mat"
    return [
        dict(name="Sample14-L1 (1001->1002)", kind="mat",
             ref=s14_ref, defm=s14_def, results=s14_res, frame=0,
             compare_matlab=True, x_only=True,
             ok=all(p.exists() for p in (s14_ref, s14_def, s14_res)),
             why="missing vol_Sample14_100{1,2}.mat or results_S14_ws20_st10.mat"),
        # The distributed uniaxial results file is 1001-vs-1006; we hold the
        # 1001/1002 TIFF stacks. So MATLAB-field parity is not applicable here —
        # but the deformation is a *known uniform* x-stretch, which makes the
        # x-only invariant plus a constant-e_xx check a strong absolute test.
        dict(name="Uniaxial stretch (1001->1002)", kind="tiff",
             ref=st_ref, defm=st_def, results=st_res, frame=0,
             compare_matlab=False, x_only=True,
             ok=st_ref.is_dir() and st_def.is_dir() and st_res.exists(),
             why="missing vol_stretch_100{1,2}_tiff or the results matfile"),
    ]


def _run_parity(spec: dict, *, quick: bool, crop: int | None,
                workers: int) -> dict:
    from nodegraph.kernels.aldvc_field import run_aldvc

    meta = load_reference(spec["results"])
    ws = int(meta["winsize"][0])
    st = int(meta["winstep"][0])
    ref = matlab_vol(spec["ref"]) if spec["kind"] == "mat" else tiff_stack(spec["ref"])
    dfm = matlab_vol(spec["defm"]) if spec["kind"] == "mat" else tiff_stack(spec["defm"])
    print(f"  {spec['name']}: vol{ref.shape} (z,y,x)  MATLAB ws={ws} st={st} "
          f"beta={meta['beta'][spec['frame']]:.4g} mu={meta['mu'][spec['frame']]:.4g}",
          flush=True)

    full_shape = ref.shape
    if crop:
        # Centre crop along the LONG axis only, so the deformation content and the
        # voxel statistics are preserved; used to keep a smoke run tractable.
        ax = int(np.argmax(full_shape))
        n = full_shape[ax]
        k = min(crop, n)
        lo = (n - k) // 2
        sl = [slice(None)] * 3
        sl[ax] = slice(lo, lo + k)
        ref, dfm = ref[tuple(sl)], dfm[tuple(sl)]
        print(f"    cropped axis {ax} to {k} -> {ref.shape}", flush=True)

    params = dict(subset_size=ws, subset_spacing=st, admm_iterations=4,
                  mu=meta["mu"][spec["frame"]], icgn_tol=meta["icgn_tol"],
                  icgn_max_iter=100, admm_tol=1e-2, seed_levels=2 if quick else 3,
                  n_workers=workers, strain_type="infinitesimal")
    t0 = time.perf_counter()
    res = run_aldvc(ref, dfm, voxel_size_um=(1.0, 1.0, 1.0), params=params)
    dt = time.perf_counter() - t0

    u = np.asarray(res.displacement_field)
    mask = interior_mask(res.grid_coords, ref.shape, ws)
    row = dict(case=spec["name"], seconds=round(dt, 1),
               grid=list(res.grid_coords.shape[:-1]), n_nodes=int(mask.sum()),
               beta_py=res.beta, beta_matlab=meta["beta"][spec["frame"]],
               admm_iters=res.iterations, converged=res.converged,
               median_zncc=res.diagnostics.get("median_zncc"), cropped=bool(crop))

    if spec["x_only"]:
        # C — invariant: the imposed deformation is x-only, so u_y = u_z = 0.
        zeros = np.zeros(int(mask.sum()))
        row["rms_uy_should_be_0"] = rms_error(u[mask][..., 1], zeros)
        row["rms_uz_should_be_0"] = rms_error(u[mask][..., 0], zeros)
        s = np.asarray(res.strain_field)[mask]
        row["exx_mean"] = float(np.mean(s[..., 2, 2]))
        row["exx_std"] = float(np.std(s[..., 2, 2]))

    if spec["compare_matlab"] and not crop:
        # B — field-level parity against MATLAB, on the Python nodes.
        mx, my, mz = meta["mnl_xyz"]
        cxyz = meta["coords_xyz"]
        axes = [np.unique(cxyz[:, 2]) - 1.0,      # z (MATLAB 1-based -> 0-based)
                np.unique(cxyz[:, 1]) - 1.0,      # y
                np.unique(cxyz[:, 0]) - 1.0]      # x
        for tag, key in (("aldvc", "U"), ("local", "U_local"), ("seed", "U0")):
            m = matlab_disp(meta[key][spec["frame"]], (mx, my, mz))
            m_on_py = resample_to(m, axes, res.grid_coords)
            for c, nm in ((2, "ux"), (1, "uy"), (0, "uz")):
                row[f"d_{tag}_{nm}"] = rms_error(u[mask][..., c], m_on_py[c][mask])
            row[f"matlab_{tag}_ux_range"] = [float(np.nanmin(m_on_py[2][mask])),
                                             float(np.nanmax(m_on_py[2][mask]))]
        row["py_ux_range"] = [float(np.nanmin(u[mask][..., 2])),
                              float(np.nanmax(u[mask][..., 2]))]
    for k, v in row.items():
        if isinstance(v, float):
            print(f"      {k:<26} {v:.5g}", flush=True)
        elif k != "case":
            print(f"      {k:<26} {v}", flush=True)
    return row


# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("synth", "parity", "all"), default="synth")
    ap.add_argument("--quick", action="store_true", help="fewer/smaller cases")
    ap.add_argument("--warp", action="store_true",
                    help="generate deformed volumes by resampling (paper protocol, "
                         "adds the generator's own interpolation bias) instead of "
                         "analytic bead re-placement")
    ap.add_argument("--shape", default=None, help="synthetic volume shape, e.g. 96,96,96")
    ap.add_argument("--subset", type=int, default=20)
    ap.add_argument("--spacing", type=int, default=10)
    ap.add_argument("--admm", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--crop", type=int, default=None,
                    help="crop the parity volumes' long axis to N voxels (smoke run)")
    ap.add_argument("--json", default=None, help="write all rows to this JSON file")
    args = ap.parse_args()

    shape = (tuple(int(v) for v in args.shape.split(","))
             if args.shape else ((64, 64, 64) if args.quick else (96, 96, 96)))
    params = dict(subset_size=args.subset, subset_spacing=args.spacing,
                  admm_iterations=args.admm, n_workers=args.workers,
                  seed_levels=2, strain_type="infinitesimal")

    out: dict[str, list[dict]] = {}
    if args.suite in ("synth", "all"):
        print(f"-- Suite A: synthetic, exact truth  shape={shape} "
              f"subset={args.subset} spacing={args.spacing} admm={args.admm} "
              f"{'(warp protocol)' if args.warp else '(analytic bead placement)'}",
              flush=True)
        out["synth"] = suite_synth(shape, params, quick=args.quick, warp=args.warp)
    if args.suite in ("parity", "all"):
        print("-- Suite B/C: FranckLab reference datasets", flush=True)
        out["parity"] = suite_parity(quick=args.quick, crop=args.crop,
                                     workers=args.workers)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
