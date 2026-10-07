"""_bead_finder_validate.py — score the Bead Finder (``detect.beads``) on synthetic data.

Runs the shipped node (through the Engine, exactly as a graph would) on the ``beads3d``
phantom — a confocal z-stack of 1 µm fluorescent spheres imaged through a skewed sinc⁴
axial PSF, with aggregates, fibres, a scan line, a haze, hot voxels and out-of-stack beads
planted — across a bead-density sweep and a left-to-right density gradient, and scores it
against the planted truth: recall, precision, lateral and axial localisation error, the
axial bias, and what every missed bead and every false positive was.

    python scripts/_bead_finder_validate.py                      # table on stdout
    python scripts/_bead_finder_validate.py out.png              # + a figure
    python scripts/_bead_finder_validate.py out.png --densities 20,60,150,400,800 --seed 0

Exit status 1 if precision drops below 0.98 or recall at the sparsest density below 0.95
(the headline guarantee this node makes), so it can gate a change.

Interpreter: ``.venv\\Scripts\\python.exe`` (numpy/scipy; matplotlib only for the figure).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nodegraph.nodes  # noqa: E402,F401 — registers the catalog
from nodegraph.catalog._base import COMPUTES  # noqa: E402
from nodegraph.dataset import Dataset  # noqa: E402
from nodegraph.domains import Domain  # noqa: E402
from nodegraph.engine import Engine  # noqa: E402
from nodegraph.graph import Graph, NodeInstance  # noqa: E402
from nodegraph.kernels.bead_slabs import expected_sigmas_px, find_beads  # noqa: E402
from nodegraph.metadata import MetaEnvelope  # noqa: E402
from nodegraph.phantom import FULL_SCALE, Phantom, phantom  # noqa: E402
from nodegraph.provider import ArrayProvider  # noqa: E402
from nodegraph.registry import OutDataset, define_node  # noqa: E402

define_node("io.bead_validate_seed", "Seed", outputs=[OutDataset()])

MATCH_UM = 0.6          # a detection within this (anisotropic µm) distance of a planted bead is it


# ── running the node ────────────────────────────────────────────────────────────

def run_node(ds: Dataset, env: MetaEnvelope, *, params: Optional[Dict[str, Any]] = None,
             modes: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """Pull ``detect.beads`` on one Dataset → ``(points_zyx voxels, columns, seconds)``."""
    g = Graph()
    g.add(NodeInstance("S", "io.bead_validate_seed"))
    g.add(NodeInstance("B", "detect.beads", params=dict(params or {}), modes=dict(modes or {})))
    g.connect("S", "B")
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
    t0 = time.perf_counter()
    out = eng.pull("B")
    dt = time.perf_counter() - t0
    get = lambda k: np.asarray(out.get(Domain.POINT, k, layer="beads").values)  # noqa: E731
    pts = np.stack([get("z"), get("y"), get("x")], axis=1) if len(get("z")) else np.zeros((0, 3))
    cols = {k: get(k) for k in ("amplitude", "snr", "sigma_xy_um", "sigma_z_um", "skew_z", "slab")}
    return pts, cols, dt


def inverted(ph: Phantom) -> Tuple[Dataset, MetaEnvelope]:
    """The same field as DARK beads on a bright background (for ``projection=min``)."""
    a = (FULL_SCALE - ph.array.astype(np.int64)).astype(np.uint16)
    md = dict(ph.envelope.metadata)
    ds = Dataset(axes=ph.axes, metadata=dict(md)).with_image(ArrayProvider(a, tile=512))
    return ds, MetaEnvelope(axes=ph.axes, metadata=dict(md), domains=frozenset({Domain.VOXEL}))


# ── scoring ─────────────────────────────────────────────────────────────────────

def score(pts: np.ndarray, truth: np.ndarray, vox: np.ndarray) -> Dict[str, Any]:
    """Unique nearest-neighbour matching inside ``MATCH_UM`` → the headline numbers plus
    the index lists a diagnosis needs."""
    from scipy.spatial import cKDTree
    n_t, n_p = len(truth), len(pts)
    out: Dict[str, Any] = {"n_truth": n_t, "n_found": n_p}
    if n_p == 0 or n_t == 0:
        out.update(tp=0, fp=n_p, recall=0.0 if n_t else 1.0, precision=0.0 if n_p else 1.0,
                   err_xy=float("nan"), err_z=float("nan"), bias_z=float("nan"),
                   missed=list(range(n_t)), false=list(range(n_p)))
        return out
    tq, pq = truth * vox, pts * vox
    d, j = cKDTree(tq).query(pq, distance_upper_bound=MATCH_UM)
    best: Dict[int, Tuple[int, float]] = {}
    for i in np.flatnonzero(np.isfinite(d)):
        if j[i] not in best or d[i] < best[j[i]][1]:
            best[int(j[i])] = (int(i), float(d[i]))
    det_idx = np.array([v[0] for v in best.values()], dtype=int)
    tru_idx = np.array(list(best.keys()), dtype=int)
    tp = len(best)
    dxy = np.hypot(pq[det_idx, 1] - tq[tru_idx, 1], pq[det_idx, 2] - tq[tru_idx, 2])
    dz = pq[det_idx, 0] - tq[tru_idx, 0]
    matched_det = set(det_idx.tolist())
    out.update(tp=tp, fp=n_p - tp, recall=tp / n_t, precision=tp / n_p,
               err_xy=float(np.median(dxy)), err_z=float(np.median(np.abs(dz))),
               bias_z=float(np.mean(dz)),
               missed=sorted(set(range(n_t)) - set(tru_idx.tolist())),
               false=[i for i in range(n_p) if i not in matched_det])
    return out


def _dist_to_segment(p_yx: np.ndarray, c_yx: np.ndarray, length_um: float, ang: float,
                     vox_yx: np.ndarray) -> float:
    d = (p_yx - c_yx) * vox_yx
    u = np.array([math.sin(ang), math.cos(ang)])
    along = float(np.clip(d @ u, -length_um / 2, length_um / 2))
    return float(np.linalg.norm(d - along * u))


def attribute(pt: np.ndarray, truth: Dict[str, Any], vox: np.ndarray) -> str:
    """Which planted thing a false positive sits on, if any."""
    where = truth.get("artifact_positions") or {}
    z, y, x = pt
    hits: List[Tuple[float, str]] = []
    if "aggregate" in where:
        for (cz, cy, cx, rz, ry, rx) in where["aggregate"]:
            r = math.sqrt(((z - cz) * vox[0] / rz) ** 2 + ((y - cy) * vox[1] / ry) ** 2
                          + ((x - cx) * vox[2] / rx) ** 2)
            hits.append((r, "aggregate"))
    if "fibre" in where:
        for (cz, cy, cx, length, ang) in where["fibre"]:
            dd = _dist_to_segment(np.array([y, x]), np.array([cy, cx]), length, ang, vox[1:])
            hits.append((max(dd, abs(z - cz) * vox[0]) / 0.6, "fibre"))
    if "scan_line" in where:
        zl, yl = where["scan_line"]
        hits.append((math.hypot((y - yl) * vox[1], (z - zl) * vox[0]) / 0.6, "scan line"))
    if "haze" in where:
        hz, hy, hx = where["haze"]
        hits.append((math.sqrt(((z - hz) * vox[0] / 2.0) ** 2 + ((y - hy) * vox[1] / 5.0) ** 2
                               + ((x - hx) * vox[2] / 5.0) ** 2), "haze"))
    if "hot_voxel" in where and len(where["hot_voxel"]):
        hv = where["hot_voxel"]
        dd = np.sqrt((((hv - pt) * vox) ** 2).sum(1))
        hits.append((float(dd.min()) / 0.6, "hot voxel"))
    if "outside_bead" in where and len(where["outside_bead"]):
        ob = where["outside_bead"]
        dd = np.hypot((ob[:, 1] - y) * vox[1], (ob[:, 2] - x) * vox[2])
        hits.append((float(dd.min()) / 0.6, "out-of-stack bead"))
    if not hits:
        return "unexplained"
    r, name = min(hits)
    return name if r <= 1.5 else "unexplained"


def explain_miss(i: int, truth: Dict[str, Any], vox: np.ndarray, dropped: List[Tuple],
                 found: np.ndarray) -> str:
    """Why a planted bead was not reported: merged with a neighbour, refused (and how),
    under an artefact, or never a candidate."""
    from scipy.spatial import cKDTree
    beads = truth["beads"]
    me = beads[i] * vox
    others = np.delete(beads, i, axis=0) * vox
    near_bead = float(cKDTree(others).query(me)[0]) if len(others) else float("inf")
    reasons = []
    if near_bead < 1.3 * truth["diameter_um"]:
        reasons.append(f"touching another bead ({near_bead:.2f} µm)")
    art = attribute(beads[i], truth, vox)
    if art != "unexplained":
        reasons.append(f"under a planted {art}")
    if dropped:
        dq = np.array([[r[0], r[1], r[2]] for r in dropped]) * vox
        d, k = cKDTree(dq).query(me)
        if d < 0.8:
            det = dropped[int(k)][4] if len(dropped[int(k)]) > 4 else {}
            reasons.append(f"candidate refused as {dropped[int(k)][3]}"
                           + (f" ({', '.join(f'{a}={b:.2f}' for a, b in det.items())})" if det else ""))
    if not reasons:
        reasons.append("never became a candidate")
    return "; ".join(reasons)


def node_sigmas(ph: Phantom) -> Tuple[float, float]:
    """The expected sigmas the node computes for this phantom — so the kernel can be
    re-run for diagnostics with the same expectations."""
    md = ph.envelope.metadata
    lam = float(md["channel_emission_nm"][0])
    na = float(md["objective_na"])
    n_imm = 1.515 if na > 1.0 else (1.33 if na > 0.8 else 1.0)
    fwhm = 0.88 * (lam / 1000.0) / (n_imm - math.sqrt(n_imm ** 2 - min(na, 0.98 * n_imm) ** 2))
    return expected_sigmas_px(1.0, pixel_size_um=md["pixel_size_um"], z_step_um=md["z_step_um"],
                              psf_sigma_xy_um=0.21 * lam / 1000.0 / na, psf_sigma_z_um=fwhm / 2.355)


# ── the cases ───────────────────────────────────────────────────────────────────

def evaluate(label: str, ph: Phantom, *, ds: Optional[Dataset] = None,
             env: Optional[MetaEnvelope] = None, params: Optional[Dict[str, Any]] = None,
             modes: Optional[Dict[str, str]] = None, diagnose: bool = True) -> Dict[str, Any]:
    tr = ph.truth
    vox = np.asarray(tr["voxel_size_um"], dtype=float)
    pts, cols, dt = run_node(ds or ph.dataset, env or ph.envelope, params=params, modes=modes)
    sc = score(pts, tr["beads"], vox)
    sc.update(label=label, seconds=dt, n_beads=tr["n_beads"], pts=pts, cols=cols,
              sigma_xy_um=float(np.median(cols["sigma_xy_um"])) if len(pts) else float("nan"),
              sigma_z_um=float(np.nanmedian(cols["sigma_z_um"])) if len(pts) else float("nan"),
              skew_z=float(np.nanmedian(cols["skew_z"])) if len(pts) else float("nan"))
    sc["false_what"] = [attribute(pts[i], tr, vox) for i in sc["false"]]
    sc["missed_why"] = []
    if diagnose and sc["missed"]:
        vol = (ds or ph.dataset).image.get_region_volume(0, 0, 0, 0, 0, ph.axes.z, 0, ph.axes.y,
                                                         0, ph.axes.x)
        s_xy, s_z = node_sigmas(ph)
        kp = {"sigma_xy_px": s_xy, "sigma_z_px": s_z,
              "projection": (modes or {}).get("projection", "max")}
        if (modes or {}).get("slabs") == "manual":
            zs = float(ph.envelope.metadata["z_step_um"])
            kp["slab_px"] = max(1, int(round(float((params or {}).get("slab_thickness", 2.0)) / zs)))
            kp["overlap_px"] = max(0, int(round(float((params or {}).get("slab_overlap", 0.8)) / zs)))
        _p, _c, info = find_beads(vol, vox, kp)
        sc["missed_why"] = [explain_miss(i, tr, vox, info["dropped"], pts) for i in sc["missed"]]
        sc["info"] = {k: info[k] for k in ("slab_px", "overlap_px", "n_slabs", "rejected",
                                          "passes", "despiked_voxels")}
    return sc


def fmt_row(sc: Dict[str, Any]) -> str:
    return (f"{sc['label']:<34s} {sc['n_truth']:5d} {sc['n_found']:5d} {sc['tp']:4d} {sc['fp']:3d} "
            f"{sc['recall']:6.3f} {sc['precision']:6.3f} {sc['err_xy'] * 1000:7.1f} "
            f"{sc['err_z'] * 1000:7.1f} {sc['bias_z'] * 1000:+7.1f} {sc['sigma_xy_um']:6.3f} "
            f"{sc['seconds']:5.1f}")


HEADER = (f"{'case':<34s} {'truth':>5s} {'found':>5s} {'tp':>4s} {'fp':>3s} {'recall':>6s} "
          f"{'prec':>6s} {'xy nm':>7s} {'|z| nm':>7s} {'z bias':>7s} {'σxy µm':>6s} {'sec':>5s}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figure", nargs="?", help="write a PNG figure here")
    ap.add_argument("--densities", default="20,60,150,400,800",
                    help="bead counts for the sweep (the field is 41.6 × 41.6 × 9.6 µm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="sweep only; skip the extra cases")
    args = ap.parse_args(argv)
    densities = [int(v) for v in args.densities.split(",") if v.strip()]

    print(HEADER)
    print("-" * len(HEADER))
    sweep: List[Dict[str, Any]] = []
    for n in densities:
        ph = phantom("beads3d", seed=args.seed, n_beads=n)
        sc = evaluate(f"{n} beads, artefacts, auto slabs", ph)
        sweep.append(sc)
        print(fmt_row(sc))
    extra: List[Dict[str, Any]] = []
    if not args.quick:
        ph = phantom("beads3d", seed=args.seed, n_beads=60, artifacts=False)
        extra.append(evaluate("60 beads, NO artefacts (control)", ph))
        print(fmt_row(extra[-1]))
        ph = phantom("beads3d", seed=args.seed, n_beads=300, gradient=True)
        extra.append(evaluate("300 beads, density GRADIENT", ph))
        print(fmt_row(extra[-1]))
        ph = phantom("beads3d", seed=args.seed, n_beads=400)
        extra.append(evaluate("400 beads, MANUAL slabs 2.0/0.8 µm", ph,
                              params={"slab_thickness": 2.0, "slab_overlap": 0.8},
                              modes={"slabs": "manual"}))
        print(fmt_row(extra[-1]))
        ph = phantom("beads3d", seed=args.seed, n_beads=60)
        ds_i, env_i = inverted(ph)
        extra.append(evaluate("60 DARK beads, projection=min", ph, ds=ds_i, env=env_i,
                              modes={"projection": "min"}))
        print(fmt_row(extra[-1]))
        ph = phantom("beads3d", seed=args.seed, n_beads=60)
        extra.append(evaluate("60 beads, projection=mean", ph, modes={"projection": "mean"}))
        print(fmt_row(extra[-1]))

    print()
    for sc in sweep + extra:
        if sc["fp"]:
            print(f"[{sc['label']}] false positives: " + ", ".join(sc["false_what"]))
        if sc["missed"]:
            print(f"[{sc['label']}] missed {len(sc['missed'])}:")
            for i, why in zip(sc["missed"], sc["missed_why"]):
                print(f"    bead {i:4d}: {why}")
        if "info" in sc:
            inf = sc["info"]
            print(f"[{sc['label']}] slabs {inf['slab_px']} planes ×{inf['n_slabs']} (overlap "
                  f"{inf['overlap_px']}), passes {[(p['slab_px'], p['n']) for p in inf['passes']]}, "
                  f"despiked {inf['despiked_voxels']} voxels, refused {inf['rejected']}")
    ph0 = phantom("beads3d", seed=args.seed, n_beads=densities[0] if densities else 60)
    print(f"\nphantom: {ph0.caption}")

    if args.figure:
        draw(sweep, extra, Path(args.figure), seed=args.seed)
        print(f"figure: {args.figure}")

    worst_prec = min(sc["precision"] for sc in sweep + extra) if sweep + extra else 1.0
    first_recall = sweep[0]["recall"] if sweep else 1.0
    ok = worst_prec >= 0.98 and first_recall >= 0.95
    print(f"\n{'BEAD FINDER VALIDATION PASSED' if ok else 'BEAD FINDER VALIDATION FAILED'}: "
          f"worst precision {worst_prec:.3f}, recall at {sweep[0]['n_beads'] if sweep else '-'} "
          f"beads {first_recall:.3f}")
    return 0 if ok else 1


# ── the figure ──────────────────────────────────────────────────────────────────

# the dataviz reference palette (light surface): slots 1–3 validate all-pairs
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_TEXT, C_TEXT2, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"


def draw(sweep: List[Dict[str, Any]], extra: List[Dict[str, Any]], path: Path, *,
         seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 4.6), facecolor=C_SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.32)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.grid(True, color=C_GRID, linewidth=0.8)
        ax.tick_params(colors=C_TEXT2, labelsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("planted beads in 41.6 × 41.6 × 9.6 µm", color=C_TEXT2, fontsize=9)
    n = [sc["n_beads"] for sc in sweep]
    from matplotlib.ticker import FixedLocator, NullLocator
    for ax in (ax1, ax2):                      # the tested densities ARE the ticks
        ax.xaxis.set_major_locator(FixedLocator(n))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(v) for v in n])

    # (a) recall & precision vs density — rates share one axis
    ax1.plot(n, [sc["recall"] for sc in sweep], color=C_BLUE, lw=1.6, marker="o", ms=6,
             label="recall")
    ax1.plot(n, [sc["precision"] for sc in sweep], color=C_ORANGE, lw=1.6, marker="o", ms=6,
             label="precision")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("found / planted, and found / reported", color=C_TEXT, fontsize=10, loc="left")
    ax1.text(n[-1], sweep[-1]["recall"] - 0.06, "recall", color=C_TEXT2, fontsize=9, ha="right")
    ax1.text(n[-1], sweep[-1]["precision"] + 0.03, "precision", color=C_TEXT2, fontsize=9,
             ha="right")
    ax1.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=C_TEXT2)

    # (b) localisation error vs density — one unit (nm), lateral and axial
    ax2.plot(n, [sc["err_xy"] * 1000 for sc in sweep], color=C_BLUE, lw=1.6, marker="o", ms=6,
             label="lateral, median")
    ax2.plot(n, [sc["err_z"] * 1000 for sc in sweep], color=C_AQUA, lw=1.6, marker="o", ms=6,
             label="axial, median |Δz|")
    ax2.plot(n, [sc["bias_z"] * 1000 for sc in sweep], color=C_AQUA, lw=1.6, ls="--",
             marker="o", ms=6, label="axial bias (PSF skew)")
    ax2.set_ylim(0, None)
    ax2.set_ylabel("nm", color=C_TEXT2, fontsize=9)
    ax2.set_title("localisation error of matched beads", color=C_TEXT, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_TEXT2)

    # (c) one field: the max projection with the truth, the found and the missed
    sc = next((s for s in sweep if s["n_beads"] >= 60), sweep[len(sweep) // 2])
    ph = phantom("beads3d", seed=seed, n_beads=sc["n_beads"])
    mip = ph.array[0, 0, :, 0].max(axis=0).astype(float)
    lo, hi = np.percentile(mip, [1, 99.7])
    ax3.imshow(mip, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    tr = ph.truth["beads"]
    found = sc["pts"]
    missed = sc["missed"]
    ax3.scatter(tr[:, 2], tr[:, 1], s=70, facecolors="none", edgecolors=C_BLUE, lw=1.2,
                label=f"planted ({len(tr)})")
    if len(found):
        ax3.scatter(found[:, 2], found[:, 1], s=12, color=C_BLUE, lw=0,
                    label=f"found ({len(found)})")
    if missed:
        ax3.scatter(tr[missed, 2], tr[missed, 1], s=130, facecolors="none", edgecolors=C_ORANGE,
                    lw=1.6, label=f"missed ({len(missed)})")
    if sc["false"]:
        fp = found[sc["false"]]
        ax3.scatter(fp[:, 2], fp[:, 1], s=90, marker="x", color=C_ORANGE, lw=1.6,
                    label=f"false ({len(fp)})")
    ax3.set_title(f"{sc['n_beads']} beads: max projection, every plane", color=C_TEXT,
                  fontsize=10, loc="left")
    ax3.set_xticks([])
    ax3.set_yticks([])
    for side in ax3.spines.values():
        side.set_color(C_GRID)
    ax3.legend(frameon=False, fontsize=8, labelcolor=C_TEXT2, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, columnspacing=1.2, handletextpad=0.4)
    fig.suptitle("Bead Finder (detect.beads) on the confocal bead phantom — artefacts planted: "
                 "aggregates, fibres, scan line, haze, hot voxels, out-of-stack beads",
                 color=C_TEXT, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=130, facecolor=C_SURF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
