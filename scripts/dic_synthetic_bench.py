"""Synthetic AL-DIC validation bench for the `analysis.dic_correlate` node.

WHAT THIS IS
    A ground-truth accuracy + wall-clock bench for our 2D DIC stack, built by
    replicating **pyALDIC's own synthetic verification suite** (upstream
    ``tests/test_integration/test_synthetic.py`` + ``tests/conftest.py`` in
    github.com/zachtong/pyALDIC) case-for-case, then adding the sub-pixel-bias
    and noise sweeps that suite does not carry.

    It exercises OUR code — ``nodegraph.kernels.dic_correlate`` and, with
    ``--node``, the wired ``analysis.dic_correlate`` node through the engine —
    not the upstream solver in isolation. ``--upstream`` additionally runs the
    same case through a *direct* ``al_dic.core.pipeline.run_aldic`` call using
    upstream's own settings, which separates "our adapter loses accuracy" from
    "AL-DIC is simply less accurate under these settings".

WHY THE UPSTREAM TOLERANCES ARE NOT DIRECTLY COMPARABLE
    Every upstream synthetic case passes ``U0=`` the EXACT ground-truth
    displacement at every mesh node as the initial guess, plus a pre-built
    ``mesh=``. Their quoted RMSEs (~0.005 px) are therefore a *converged-from-
    truth* floor. A real acquisition has no such oracle: our node must find the
    field from an FFT integer search. ``--u0`` reproduces the upstream oracle
    path for comparison; the default (no ``--u0``) is the honest number.

USAGE
    python scripts/dic_synthetic_bench.py                    # kernel, all cases
    python scripts/dic_synthetic_bench.py --quick            # 3 fast cases
    python scripts/dic_synthetic_bench.py --node             # + engine/node path
    python scripts/dic_synthetic_bench.py --upstream --u0    # + upstream oracle
    python scripts/dic_synthetic_bench.py --cases translation,subpixel
    python scripts/dic_synthetic_bench.py --json out.json    # machine-readable

Requires: al-dic (``pip install al-dic``), scipy, numpy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth image synthesis — ported from pyALDIC tests/conftest.py
# ═══════════════════════════════════════════════════════════════════════════════
IMG_H = IMG_W = 256
CX = CY = 127.0                      # 0-based centre of a 256² image (upstream)
EDGE_MARGIN = 32                     # upstream compute_disp_rmse_interior default


def generate_speckle(height: int = IMG_H, width: int = IMG_W,
                     sigma: float = 3.0, seed: int = 42) -> np.ndarray:
    """Gaussian-filtered noise speckle in [20, 235] (upstream ``generate_speckle``)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    filtered = gaussian_filter(rng.standard_normal((height, width)), sigma=sigma,
                               mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement_lagrangian(ref: np.ndarray, u_func: Callable,
                                  v_func: Callable, n_iter: int = 20) -> np.ndarray:
    """Deform ``ref`` under the LAGRANGIAN convention x = X + u(X).

    Upstream ``apply_displacement_lagrangian``: fixed-point inversion of the
    reference->deformed map, sampled with order=5 quintic B-splines so the
    solver's own order=3 cubic sampling is not fighting a matched-order
    interpolation artifact (which would floor the achievable residual)."""
    from scipy.ndimage import map_coordinates
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    warped = map_coordinates(ref, np.array([Y.ravel(), X.ravel()]), order=5,
                             mode="nearest")
    return warped.reshape(h, w)


# ═══════════════════════════════════════════════════════════════════════════════
# Case catalog
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Case:
    """One synthetic deformation with an analytic ground truth."""
    name: str
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]     # x-displacement u(x, y)
    v: Callable[[np.ndarray, np.ndarray], np.ndarray]     # y-displacement v(x, y)
    tol: float                                            # upstream RMSE tol (px)
    note: str = ""
    params: Dict[str, Any] = field(default_factory=dict)  # param overrides
    noise: float = 0.0                                    # additive Gaussian sigma (grey levels)
    upstream: bool = True                                 # is this an upstream case?


_ROT = lambda deg: (                                       # noqa: E731
    (lambda x, y, a=np.deg2rad(deg): (x - CX) * (np.cos(a) - 1) - (y - CY) * np.sin(a)),
    (lambda x, y, a=np.deg2rad(deg): (x - CX) * np.sin(a) + (y - CY) * (np.cos(a) - 1)),
)

CASES: List[Case] = [
    # ── upstream test_synthetic.py cases (frame-2 field of each) ──────────────
    Case("zero", lambda x, y: np.zeros_like(x), lambda x, y: np.zeros_like(x),
         tol=0.01, note="upstream case1 — the noise floor"),
    Case("translation", lambda x, y: np.full_like(x, 2.5),
         lambda x, y: np.full_like(x, -1.8),
         tol=0.03, note="upstream case2 — rigid (u=2.5, v=-1.8) px"),
    Case("affine", lambda x, y: 0.02 * (x - CX), lambda x, y: 0.02 * (y - CY),
         tol=0.05, note="upstream case3 — 2% biaxial stretch"),
    Case("shear", lambda x, y: 0.015 * (y - CY), lambda x, y: np.zeros_like(x),
         tol=0.05, note="upstream case5 — 1.5% simple shear"),
    Case("rotation", _ROT(2.0)[0], _ROT(2.0)[1],
         tol=0.05, note="upstream case10 — 2 deg rigid rotation"),
    Case("large_deform", lambda x, y: 0.10 * (x - CX) + 0.05 * (y - CY),
         lambda x, y: 0.05 * (x - CX) + 0.10 * (y - CY),
         tol=1.0, params={"winsize": 48},
         note="upstream case6 — 10% stretch + 5% shear (winsize 48)"),
    Case("local_only", lambda x, y: 0.02 * (x - CX), lambda x, y: 0.02 * (y - CY),
         tol=0.05, params={"use_global_step": False},
         note="upstream case9 — Local DIC only (ADMM off)"),

    # ── our additions: the classic DIC sub-pixel + robustness probes ──────────
    Case("subpixel_050", lambda x, y: np.full_like(x, 0.5),
         lambda x, y: np.full_like(x, 0.5), tol=0.03, upstream=False,
         note="half-pixel shift — peak of the sub-pixel bias S-curve"),
    Case("subpixel_025", lambda x, y: np.full_like(x, 0.25),
         lambda x, y: np.full_like(x, 0.25), tol=0.03, upstream=False,
         note="quarter-pixel shift"),
    Case("large_translation", lambda x, y: np.full_like(x, 11.0),
         lambda x, y: np.full_like(x, -7.0), tol=0.05, upstream=False,
         note="11 px shift — stresses the FFT integer search range"),
    Case("noise_2pct", lambda x, y: np.full_like(x, 2.5),
         lambda x, y: np.full_like(x, -1.8), tol=0.10, noise=4.3, upstream=False,
         note="translation + 2% (4.3 grey level) Gaussian sensor noise"),
    Case("sinusoid", lambda x, y: 1.0 * np.sin(2 * np.pi * x / 64.0),
         lambda x, y: 1.0 * np.sin(2 * np.pi * y / 64.0),
         tol=0.15, upstream=False,
         note="amp 1 px, wavelength 64 px — spatial-resolution probe"),
]
CASES_BY_NAME = {c.name: c for c in CASES}
QUICK = ["zero", "translation", "affine"]

#: pyALDIC's own synthetic-suite solver settings (tests/test_integration/
#: test_synthetic.py::_case_para). NOTE the two smoothness weights are ZEROED
#: there but default to 5e-4 / 1e-5 in DICPara — the suite is measuring the
#: unregularized solver.
UPSTREAM_PARA = dict(winsize=32, winstepsize=16, winsize_min=8, admm_max_iter=3,
                     icgn_max_iter=50, tol=1e-2, mu=1e-3, disp_smoothness=0.0,
                     strain_smoothness=0.0, gauss_pt_order=2, alpha=0.0)

#: What OUR node ships as defaults (nodegraph/nodes.py analysis.dic_correlate).
NODE_DEFAULTS = dict(winsize=40, winstepsize=16, winsize_min=8, admm_max_iter=3,
                     icgn_max_iter=100, tol=1e-2, mu=1e-3, disp_smoothness=5e-4,
                     strain_smoothness=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Score:
    n: int
    rmse_u: float
    rmse_v: float
    bias_u: float
    bias_v: float
    max_err: float
    n_nan: int
    seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("n", "rmse_u", "rmse_v", "bias_u", "bias_v", "max_err", "n_nan",
                 "seconds")}


def score_grid(grid_yx: np.ndarray, disp_dydx: np.ndarray, case: Case,
               seconds: float, edge_margin: int = EDGE_MARGIN) -> Score:
    """RMSE / bias of a (Gy, Gx, 2) [dy, dx] field vs the case's analytic truth.

    Ground truth is evaluated at the grid's own REFERENCE coordinates, which is
    exactly what a Lagrangian DIC solve reports. Nodes within ``edge_margin`` of
    an image border are excluded (upstream ``compute_disp_rmse_interior``)."""
    y = grid_yx[..., 0].ravel()
    x = grid_yx[..., 1].ravel()
    dy = disp_dydx[..., 0].ravel()
    dx = disp_dydx[..., 1].ravel()
    gt_dx = case.u(x, y)
    gt_dy = case.v(x, y)
    interior = ((x >= edge_margin) & (x <= IMG_W - 1 - edge_margin)
                & (y >= edge_margin) & (y <= IMG_H - 1 - edge_margin))
    finite = np.isfinite(dx) & np.isfinite(dy)
    sel = interior & finite
    n_nan = int(np.count_nonzero(interior & ~finite))
    if not sel.any():
        return Score(0, np.inf, np.inf, np.inf, np.inf, np.inf, n_nan, seconds)
    eu = dx[sel] - gt_dx[sel]
    ev = dy[sel] - gt_dy[sel]
    return Score(int(sel.sum()),
                 float(np.sqrt(np.mean(eu ** 2))), float(np.sqrt(np.mean(ev ** 2))),
                 float(np.mean(eu)), float(np.mean(ev)),
                 float(max(np.abs(eu).max(), np.abs(ev).max())), n_nan, seconds)


# ═══════════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════════
def build_pair(case: Case, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=seed)
    deformed = apply_displacement_lagrangian(ref, case.u, case.v)
    if case.noise > 0:
        rng = np.random.default_rng(seed + 1000)
        ref = ref + rng.normal(0.0, case.noise, ref.shape)
        deformed = deformed + rng.normal(0.0, case.noise, deformed.shape)
    return ref, deformed


def run_kernel(case: Case, base_params: Dict[str, Any]) -> Score:
    """Our vendored adapter, the way the node calls it."""
    from nodegraph.kernels.dic_correlate import run_pyaldic_pair
    ref, deformed = build_pair(case)
    params = dict(base_params)
    params.update(case.params)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_pyaldic_pair(ref, deformed, (1.0, 1.0), params)
    dt = time.perf_counter() - t0
    return score_grid(res.grid_coords, res.displacement_field, case, dt)


def run_upstream(case: Case, use_u0: bool) -> Score:
    """A direct ``run_aldic`` call in upstream's own idiom (mesh + optional U0)."""
    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange, merge_uv, DICMesh
    from al_dic.core.pipeline import run_aldic

    step, margin = 16, 16
    xs = np.arange(margin, IMG_W - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, IMG_H - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    ny, nx = len(ys), len(xs)
    elems = np.array([[iy * nx + ix, iy * nx + ix + 1, (iy + 1) * nx + ix + 1,
                       (iy + 1) * nx + ix, -1, -1, -1, -1]
                      for iy in range(ny - 1) for ix in range(nx - 1)],
                     dtype=np.int64)
    mesh = DICMesh(coordinates_fem=coords, elements_fem=elems, x0=xs, y0=ys)

    over = dict(UPSTREAM_PARA)
    over.update({k: v for k, v in case.params.items()})
    para = dicpara_default(img_size=(IMG_H, IMG_W),
                           gridxy_roi_range=GridxyROIRange(gridx=(0, IMG_W - 1),
                                                           gridy=(0, IMG_H - 1)),
                           reference_mode="accumulative", show_plots=False,
                           **over)
    ref, deformed = build_pair(case)
    U0 = merge_uv(case.u(coords[:, 0], coords[:, 1]),
                  case.v(coords[:, 0], coords[:, 1])) if use_u0 else None
    ones = np.ones((IMG_H, IMG_W), dtype=np.float64)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_aldic(para, [ref, deformed], [ones, ones], mesh=mesh, U0=U0,
                        compute_strain=False)
    dt = time.perf_counter() - t0
    fr = out.result_disp[0]
    U = fr.U_accum if fr.U_accum is not None else fr.U
    c = out.dic_mesh.coordinates_fem
    grid = np.stack([c[:, 1], c[:, 0]], axis=-1)[None, :, :]        # (1, N, 2) [y,x]
    disp = np.stack([np.asarray(U)[1::2], np.asarray(U)[0::2]], axis=-1)[None, :, :]
    return score_grid(grid, disp, case, dt)


def run_node(case: Case, base_params: Dict[str, Any]) -> Score:
    """The wired node, pulled through the engine (2-frame graph, fixed_frame)."""
    from nodegraph.dataset import Dataset
    from nodegraph.engine import Engine
    from nodegraph.graph import Graph, NodeInstance
    from nodegraph.metadata import AxisSizes, MetaEnvelope
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodegraph.registry import NODES, OutDataset, define_node
    from nodegraph.structure import Domain

    if "io.dicbench" not in NODES:
        define_node("io.dicbench", "S", outputs=[OutDataset()])
    ref, deformed = build_pair(case)
    img = np.zeros((1, 2, 1, 1, IMG_H, IMG_W))
    img[0, 0, 0, 0], img[0, 1, 0, 0] = ref, deformed
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=IMG_H, x=IMG_W)
    meta = {"pixel_size_um": 1.0}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
    params = {k: v for k, v in {**base_params, **case.params}.items()
              if k not in ("use_global_step",)}
    g = Graph()
    g.add(NodeInstance("S", "io.dicbench"))
    g.add(NodeInstance("D", "analysis.dic_correlate", params=params))
    g.connect("S", "D", dst_socket="data")
    e = Engine(g, computes=COMPUTES, seeds={"S": ds},
               meta_seeds={"S": MetaEnvelope(axes=ax, metadata=meta)})
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = e.pull("D")
    dt = time.perf_counter() - t0
    col = lambda n: out.get(Domain.POINT, n, layer="dic").values      # noqa: E731
    t = col("t")
    sel = t == 1
    grid = np.stack([col("y")[sel], col("x")[sel]], axis=-1)[None, :, :]
    disp = np.stack([col("disp_y")[sel], col("disp_x")[sel]], axis=-1)[None, :, :]
    return score_grid(grid, disp, case, dt)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def _fmt(label: str, s: Score, tol: float) -> str:
    worst = max(s.rmse_u, s.rmse_v)
    flag = "PASS" if worst < tol else "FAIL"
    return (f"    {label:<10s} rmse_u={s.rmse_u:8.4f}  rmse_v={s.rmse_v:8.4f}  "
            f"bias=({s.bias_u:+.4f},{s.bias_v:+.4f})  max={s.max_err:7.4f}  "
            f"n={s.n:<5d} nan={s.n_nan:<3d} {s.seconds:6.2f}s  [{flag} vs {tol}]")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="", help="comma-separated case names (default: all)")
    ap.add_argument("--quick", action="store_true", help=f"only {QUICK}")
    ap.add_argument("--node", action="store_true", help="also run the wired node via the engine")
    ap.add_argument("--upstream", action="store_true", help="also run a direct al_dic call")
    ap.add_argument("--u0", action="store_true", help="give --upstream the ground-truth U0 oracle")
    ap.add_argument("--upstream-params", action="store_true",
                    help="drive our kernel with pyALDIC's suite settings instead of node defaults")
    ap.add_argument("--json", default="", help="write results to this JSON path")
    a = ap.parse_args(argv)

    from nodegraph.kernels.dic_correlate import al_dic_available
    if not al_dic_available():
        print("al-dic is not installed — `pip install al-dic`", file=sys.stderr)
        return 2

    names = ([n.strip() for n in a.cases.split(",") if n.strip()] if a.cases
             else (QUICK if a.quick else [c.name for c in CASES]))
    unknown = [n for n in names if n not in CASES_BY_NAME]
    if unknown:
        print(f"unknown case(s): {unknown}\nknown: {list(CASES_BY_NAME)}", file=sys.stderr)
        return 2

    base = dict(UPSTREAM_PARA if a.upstream_params else NODE_DEFAULTS)
    print(f"AL-DIC synthetic bench — {IMG_H}x{IMG_W} speckle (sigma=3, seed=42), "
          f"Lagrangian order-5 warp")
    print(f"  solver params: {'pyALDIC suite' if a.upstream_params else 'our node defaults'} "
          f"{ {k: base[k] for k in ('winsize', 'winstepsize', 'disp_smoothness')} }")
    print(f"  interior only (edge margin {EDGE_MARGIN} px); tolerances are upstream's own\n")

    out: Dict[str, Any] = {"config": {"base_params": base, "u0": a.u0}, "cases": {}}
    worst_fail = 0
    for name in names:
        case = CASES_BY_NAME[name]
        print(f"  {name}  — {case.note}")
        rec: Dict[str, Any] = {"note": case.note, "tol": case.tol,
                               "upstream_case": case.upstream}
        try:
            s = run_kernel(case, base)
            rec["kernel"] = s.as_dict()
            print(_fmt("kernel", s, case.tol))
            worst_fail += int(max(s.rmse_u, s.rmse_v) >= case.tol)
        except Exception as ex:                              # noqa: BLE001 — bench
            rec["kernel"] = {"error": f"{type(ex).__name__}: {ex}"}
            print(f"    kernel     ERROR {type(ex).__name__}: {ex}")
            worst_fail += 1
        if a.upstream:
            try:
                s = run_upstream(case, a.u0)
                rec["upstream"] = s.as_dict()
                print(_fmt("upstream" + ("+U0" if a.u0 else ""), s, case.tol))
            except Exception as ex:                          # noqa: BLE001 — bench
                rec["upstream"] = {"error": f"{type(ex).__name__}: {ex}"}
                print(f"    upstream   ERROR {type(ex).__name__}: {ex}")
        if a.node:
            try:
                s = run_node(case, base)
                rec["node"] = s.as_dict()
                print(_fmt("node", s, case.tol))
            except Exception as ex:                          # noqa: BLE001 — bench
                rec["node"] = {"error": f"{type(ex).__name__}: {ex}"}
                print(f"    node       ERROR {type(ex).__name__}: {ex}")
        out["cases"][name] = rec
        print()

    print(f"{len(names) - worst_fail}/{len(names)} kernel cases within upstream tolerance")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
