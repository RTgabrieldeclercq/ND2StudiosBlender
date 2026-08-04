"""SerialTrack validation suite — does the vendored tracker reproduce the paper?

WHY THIS EXISTS
    `nodegraph.selftest` proves the Track Objects node *wires up*. It does not
    prove the SerialTrack backend localises or links particles correctly. This
    script does, against three independent kinds of truth taken from FranckLab's
    own distributed example data:

    A. PER-PARTICLE SYNTHETIC TRUTH (exact, machine precision) — every
       `imgFolder/img_syn_hardpar/*` case ships `imposed_disp.mat` holding the
       Poisson-disc seed coordinates `x0`, the deformed coordinates `x1{k}`, and
       the imposed displacement `u{k}` for **each individual bead**. So we know
       not only the true displacement field but the true *correspondence*: bead
       i in the reference is bead i in every deformed frame. That turns the
       tracking ratio and the displacement RMS error of paper Fig. 3 into
       directly checkable numbers, and adds one the paper cannot report — the
       fraction of links that go to the *right* particle.

    B. GOLDEN MATLAB PARITY (truth = FranckLab's own MATLAB output) — two 3-D
       cases ship `results_3D_hardpar.mat` with the `parCoord_prev`,
       `track_A2B_prev` and `uvw_B2A_prev` that the MATLAB code produced from
       these exact volumes with the documented parameters. Same input, same
       nominal parameters, so the Python detector should land on the MATLAB
       detector. This is the only test that can catch a *self-consistent* port
       that is uniformly wrong — and it is what caught the radial-symmetry
       origin bug (0.34 px → 0.064 px, matching MATLAB's 0.064 px).

    C. INVARIANT (truth known without any reference) — the imposed fields are
       homogeneous, so the displacement is *linear* in position; a correct
       global step reproduces a linear field exactly, and the transverse
       components of a pure x-translation are identically zero.

CONVENTIONS  (get these wrong and every number below is meaningless)
    3-D volumes. `GenerateSynVol.m` seeds beads with `seedBeadsN(sigma,x0,sizeI)`
    which writes `I(idx1,idx2,idx3)` from `x0(:,1..3)`, and saves the volume
    unrotated. So `x0` column j indexes MATLAB array dim j, and the numpy array
    scipy loads has the same axis order: **ground truth = x0 - 1** (1-based →
    0-based), no permutation.

    2-D images. `GenerateSynImg_hardpar.m` writes `imwrite(uint8(ImgCurr)')` —
    transposed — and `funReadImage2.m` reads it back with `Img = double(...)'`,
    transposing again. The two cancel, so MATLAB tracks in the generator's frame.
    Reading the TIFF in Python gives the *transposed* array, so `.T` recovers the
    generator's frame and again **ground truth = x0 - 1** with no permutation.
    (`load_image` does this once; nothing downstream re-permutes.)

    Displacement sign. The tracker returns `disp_b2a`: the displacement that
    warps the deformed (B) particles back onto the reference (A). The imposed
    field `u` is A→B. So the expected value is **-u**, which is also what
    MATLAB's `uvw_B2A_prev` contains.

WHAT `ratio` IS, AND IS NOT
    `ratio` here is per-frame links surviving outlier removal, over the reference
    frame's particle count. That is deliberately stricter than the ratio the
    paper's Fig. 3 plots, in two ways, so do not read a shortfall as a defect:

      * The ADMM loop exits on the displacement-update norm and returns whatever
        `track_A2B` the *last* local step produced — so on an easy field (a 0.1 px
        translation) it converges at iteration 2, while `n_neighbors` is still 16,
        and never reaches the `n_neighbors = 2` pass that links everything. That
        is upstream's control flow verbatim (`f_track_serial_match2D.m:265-270`),
        not a port artefact: instrumenting the 2-D translation case shows the raw
        match count climbing 679 → 780 of 780 as K decays, at **1.000** precision
        the whole way.
      * The paper reports the ratio *after* trajectory-segment merging
        (§2.4: "extrapolation and finding", 3–5 passes), which recovers links no
        single frame pair made.

    `correct` and `rms_px` have no such ambiguity, which is why they are the
    numbers to watch: `correct` is the share of links that went to the physically
    correct particle, and `rms_px` is the displacement error over those links.

Usage
    python scripts/_serialtrack_validate.py --suite detect     # A + B, detection only
    python scripts/_serialtrack_validate.py --suite link       # A, linker on exact coords
    python scripts/_serialtrack_validate.py --suite pipeline   # A, images -> tracks
    python scripts/_serialtrack_validate.py --suite parity     # B, vs golden MATLAB
    python scripts/_serialtrack_validate.py --suite all
    python scripts/_serialtrack_validate.py --suite all --quick
    python scripts/_serialtrack_validate.py --suite all --json out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodegraph.kernels import track_objects as T  # noqa: E402

DATA = Path(r"C:\Users\McGheeLab - Analysis\Downloads")
D3 = DATA / "SerialTrack3D_data" / "SerialTrack3D_data" / "imgFolder"
D2 = DATA / "SerialTrack2D_data" / "SerialTrack2D_data" / "imgFolder"
SYN3 = D3 / "img_syn_hardpar"
SYN2 = D2 / "img_syn_hardpar"


# ──────────────────────────────────────────────────────────────────────
#  Upstream parameter sets, copied from the shipped Example_main files
# ──────────────────────────────────────────────────────────────────────

# Example_main_3D_hardpar_inc_syn.m
BEAD3 = dict(method=T.DetectionMethod.TPT, threshold=0.5, bead_radius=3.0,
             min_size=4, max_size=100, win_size=(5, 5, 5), color="white")
MPT3 = dict(f_o_s=60.0, n_neighbors_max=25, n_neighbors_min=1,
            solver=T.GlobalSolver.REGULARIZATION, smoothness=1e-1,
            outlier_threshold=5.0, max_iter=20, iter_stop_threshold=1e-3,
            dist_missing=5.0)

# Example_main_hardpar_inc_syn.m  (2-D)
BEAD2 = dict(method=T.DetectionMethod.TRACTRAC, threshold=0.5, bead_radius=3.0,
             min_size=2, max_size=20, color="white")
MPT2 = dict(f_o_s=30.0, n_neighbors_max=25, n_neighbors_min=1,
            solver=T.GlobalSolver.ADMM, smoothness=1e-2,
            outlier_threshold=2.0, max_iter=20, iter_stop_threshold=1e-2,
            dist_missing=2.0)

# The paper's own test matrix (Table 3), with the per-case parameters taken from
# the matching shipped `Example_main_*` file. Neither the mode nor the two knobs
# below are a free choice:
#
#   * `f_o_s` is the radius the topology matcher draws candidates from, and the
#     rigid-body examples set it to **Inf** ("search the whole field"). A large
#     *finite* value is NOT equivalent — candidates are then the `n_neighbors`
#     nearest *within* the radius (`f_track_neightopo_match3.m:122-128`), and
#     under a 100 deg rotation a particle's true partner is nowhere near its 25
#     nearest neighbours.
#   * `use_prev` is `MPTPara.usePrevResults`, which the stretch/shear accum
#     examples set to 1. Cumulative mode compares frame k against frame 0
#     directly, so without a warm start every frame is solved cold from a
#     deformation the scale/rotation-invariant descriptor cannot see through
#     (a uniaxial stretch is not a similarity transform). The paper says as much:
#     "For large deformations, we employ the tracked results from the previous
#     frames ... to estimate a displacement predictor."
#
# Table 3 lists f_o_s = 50 for stretch/shear where the shipped code uses 60 (3-D)
# / 30 (2-D); the shipped value wins here, since that is what actually ran.
#
#   name                        mode     f_o_s   use_prev  paper fig
CASES_3D = [
    ("img_trans_hardpar_sd2",    "inc",   np.inf, False, "3e"),
    ("img_rot_hardpar_sd2",      "inc",   np.inf, False, "3f"),
    ("img_str_hardpar_sd2",      "accum", 60.0,   True,  "3g"),
    ("img_simshear_hardpar_sd2", "accum", 60.0,   True,  "3h"),
]
CASES_2D = [
    ("img_trans_hardpar_sd1",    "inc",   np.inf, False, "3a"),
    ("img_rot_hardpar_sd1",      "inc",   np.inf, False, "3b"),
    ("img_str_hardpar_sd1",      "accum", 30.0,   True,  "3c"),
    ("img_simshear_hardpar_sd1", "accum", 30.0,   True,  "3d"),
]

MODE_OF = {"inc": T.TrackingMode.INCREMENTAL, "accum": T.TrackingMode.CUMULATIVE}

GOLDEN_3D = ["img_trans_hardpar_sd2", "img_simshear_hardpar_sd2"]


# ──────────────────────────────────────────────────────────────────────
#  MATLAB ↔ numpy bridges  (the only place a convention is applied)
# ──────────────────────────────────────────────────────────────────────

def ground_truth(case_dir: Path) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """``imposed_disp.mat`` → ``(x0, [x1_k], [u_k])`` in 0-based numpy coords.

    See the CONVENTIONS block: column j ↔ numpy axis j for both 2-D and 3-D, so
    the whole conversion is a ``- 1``.
    """
    d = sio.loadmat(str(case_dir / "imposed_disp.mat"))
    x0 = np.asarray(d["x0"][0, 0], dtype=np.float64) - 1.0
    n = d["x1"].shape[1]
    x1 = [np.asarray(d["x1"][0, k], dtype=np.float64) - 1.0 for k in range(n)]
    u = [np.asarray(d["u"][0, k], dtype=np.float64) for k in range(n)]
    return x0, x1, u


def load_volume(case_dir: Path, k: int) -> np.ndarray:
    """``vol_100k.mat`` → numpy volume, axis order already matching `x0`."""
    return sio.loadmat(str(case_dir / f"vol_{1000 + k}.mat"))["vol"][0, 0]


def load_image(case_dir: Path, k: int) -> np.ndarray:
    """``img_100k.tif`` → numpy image in the *generator's* frame.

    The `.T` undoes `imwrite(ImgCurr')`, reproducing what `funReadImage2.m`
    hands the MATLAB tracker.
    """
    from PIL import Image
    im = np.asarray(Image.open(str(case_dir / f"img_{1000 + k}.tif")),
                    dtype=np.float64)
    return np.ascontiguousarray(im.T)


def golden(case_dir: Path) -> Optional[Dict[str, list]]:
    """``results_3D_hardpar.mat`` → MATLAB's own detections / links / displacements."""
    p = case_dir / "results_3D_hardpar.mat"
    if not p.exists():
        return None
    r = sio.loadmat(str(p))
    return {
        # 1-based MATLAB coords → 0-based
        "parCoord": [np.asarray(a[0], float) - 1.0 for a in r["parCoord_prev"]],
        # MATLAB link indices are 1-based with 0 = "no match" → 0-based with -1
        "track_A2B": [np.asarray(a[0], np.int64).ravel() - 1 for a in r["track_A2B_prev"]],
        "uvw_B2A": [np.asarray(a[0], float) for a in r["uvw_B2A_prev"]],
    }


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────

def localisation_error(found: np.ndarray, truth: np.ndarray,
                       tol: float = 2.0) -> Dict[str, float]:
    """Detection quality: how many true beads were found, and how precisely.

    `tol` (px) is the radius within which a detection counts as *that* bead; it
    only affects the recall count, never the error statistics.
    """
    if len(found) == 0:
        return dict(n=0, recall=0.0, med=float("nan"), rms=float("nan"),
                    p90=float("nan"))
    dist, idx = cKDTree(truth).query(found)
    hit = dist < tol
    return dict(
        n=int(len(found)),
        recall=float(len(set(idx[hit].tolist())) / len(truth)),
        med=float(np.median(dist)),
        # clip so a handful of spurious detections can't dominate the RMS
        rms=float(np.sqrt(np.mean(np.minimum(dist, tol) ** 2))),
        p90=float(np.percentile(dist, 90)),
    )


def link_quality(track_a2b: np.ndarray, disp_b2a: np.ndarray,
                 coords_a: np.ndarray, coords_b: np.ndarray,
                 ids_a: np.ndarray, ids_b: np.ndarray,
                 disp_expected: np.ndarray) -> Dict[str, float]:
    """Linker quality against a known correspondence.

    `ids_a` / `ids_b` map each row of `coords_a` / `coords_b` to the ground-truth
    bead id, so a link is *correct* when the two ids agree. `disp_expected` is
    the true B→A displacement at each B row.
    """
    m = track_a2b >= 0
    n_a = len(track_a2b)
    if not m.any():
        return dict(ratio=0.0, correct=0.0, rms=float("nan"), p95=float("nan"))
    ib = track_a2b[m]
    right = ids_a[m] == ids_b[ib]
    err = np.linalg.norm(disp_b2a[ib] - disp_expected[ib], axis=1)
    e = err[right]
    return dict(
        ratio=float(m.sum() / n_a),
        correct=float(right.sum() / m.sum()),
        rms=float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
        p95=float(np.percentile(e, 95)) if e.size else float("nan"),
    )


# ──────────────────────────────────────────────────────────────────────
#  Suite A/B — detection
# ──────────────────────────────────────────────────────────────────────

def suite_detect(quick: bool) -> dict:
    print("\n" + "=" * 92)
    print("SUITE detect — particle localisation vs per-bead truth (A) and MATLAB (B)")
    print("=" * 92)
    out: dict = {"3d": {}, "2d": {}}

    cases3 = [c[0] for c in (CASES_3D[:1] if quick else CASES_3D)]
    print(f"\n3-D  TPT (locateParticles → radialcenter3dvec), {BEAD3['win_size']} window")
    print(f"  {'case':30s} {'frame':>5s} {'N':>6s} {'recall':>7s} {'med':>7s} "
          f"{'rms':>7s} {'p90':>7s}   {'MATLAB med/rms':>16s}")
    for case in cases3:
        cd = SYN3 / case
        if not cd.exists():
            print(f"  {case:30s} -- missing --"); continue
        x0, x1, _ = ground_truth(cd)
        gold = golden(cd)
        det = T.ParticleDetector(T.DetectionConfig(**BEAD3))
        frames = [0] if quick else [0, 1, len(x1)]
        rows = []
        for fi in frames:
            vol = load_volume(cd, fi + 1)
            truth = x0 if fi == 0 else x1[fi - 1]
            t0 = time.perf_counter()
            found = T.ParticleDetector.clip_to_bounds(det.detect(vol), vol.shape)
            dt = time.perf_counter() - t0
            m = localisation_error(found, truth)
            m["seconds"] = round(dt, 3)
            gm = ""
            if gold and fi < len(gold["parCoord"]):
                g = localisation_error(gold["parCoord"][fi], truth)
                m["matlab"] = g
                gm = f"{g['med']:.4f}/{g['rms']:.4f}"
            print(f"  {case:30s} {fi:5d} {m['n']:6d} {m['recall']:7.3f} "
                  f"{m['med']:7.4f} {m['rms']:7.4f} {m['p90']:7.4f}   {gm:>16s}")
            rows.append(m)
        out["3d"][case] = rows

    cases2 = [c[0] for c in (CASES_2D[:1] if quick else CASES_2D)]
    print(f"\n2-D  LoG / TracTrac (f_detect_particles)")
    print(f"  {'case':30s} {'frame':>5s} {'N':>6s} {'recall':>7s} {'med':>7s} "
          f"{'rms':>7s} {'p90':>7s}")
    for case in cases2:
        cd = SYN2 / case
        if not cd.exists():
            print(f"  {case:30s} -- missing --"); continue
        x0, x1, _ = ground_truth(cd)
        det = T.ParticleDetector(T.DetectionConfig(**BEAD2))
        rows = []
        for fi in ([0] if quick else [0, 1, len(x1)]):
            img = load_image(cd, fi + 1)
            truth = x0 if fi == 0 else x1[fi - 1]
            found = T.ParticleDetector.clip_to_bounds(det.detect(img), img.shape)
            m = localisation_error(found, truth)
            print(f"  {case:30s} {fi:5d} {m['n']:6d} {m['recall']:7.3f} "
                  f"{m['med']:7.4f} {m['rms']:7.4f} {m['p90']:7.4f}")
            rows.append(m)
        out["2d"][case] = rows
    return out


# ──────────────────────────────────────────────────────────────────────
#  Suite A — the linker, on exact coordinates
# ──────────────────────────────────────────────────────────────────────

def _run_coords(coords: List[np.ndarray], mode, mpt: dict, ndim: int):
    cfg = T.TrackingConfig(mode=mode, strain_n_neighbors=0, **mpt)
    cfg.roi_x = (0, 1)
    cfg.roi_y = (0, 1)
    cfg.roi_z = (0, 1) if ndim == 3 else None
    tracker = T.SerialTracker(T.DetectionConfig(), cfg)
    t0 = time.perf_counter()
    session = tracker.track_coordinates(coords)
    return session, time.perf_counter() - t0


def frame_shape(case_dir: Path, dim: int) -> Tuple[int, ...]:
    """Pixel/voxel extent of a case's frames."""
    if dim == 3:
        return load_volume(case_dir, 1).shape
    return load_image(case_dir, 1).shape


def in_bounds(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Mask of coords inside ``[0, shape)`` on every axis.

    ``fun_SerialTrack_3D_HardPar.m:98-99`` deletes out-of-frame detections
    outright, so a particle the imposed field has pushed past the edge is not in
    MATLAB's denominator either. Leaving them in makes the tracking ratio
    unachievable by construction: at λ = 1.5 uniaxial stretch a third of the
    beads are outside a 192³ volume, capping any correct tracker at 0.67.
    """
    m = np.ones(len(coords), dtype=bool)
    for d in range(coords.shape[1]):
        m &= (coords[:, d] >= 0) & (coords[:, d] < shape[d])
    return m


def suite_link(quick: bool) -> dict:
    print("\n" + "=" * 92)
    print("SUITE link — ADMM linker fed EXACT coordinates (detection error = 0)")
    print("  Isolates the linker. Correspondence is the identity, so `correct` is")
    print("  the fraction of links that went to the *right* bead, not just to one.")
    print("=" * 92)
    out: dict = {}
    n_steps = 4 if quick else 10

    for dim, root, cases, mpt in ((3, SYN3, CASES_3D, MPT3),
                                  (2, SYN2, CASES_2D, MPT2)):
        print(f"\n{dim}-D   solver={mpt['solver'].name}, "
              f"smoothness={mpt['smoothness']}   (paper Table 3 per-case f_o_s)")
        print(f"  {'case':30s} {'fig':>4s} {'mode':>6s} {'f_o_s':>7s} {'steps':>5s} "
              f"{'ratio':>7s} {'effcy':>7s} {'correct':>8s} {'rms_px':>9s} "
              f"{'p95_px':>9s} {'max|u|':>8s} {'sec':>7s}")
        for case, mode_key, fos, use_prev, fig in (cases[:1] if quick else cases):
            cd = root / case
            if not cd.exists():
                print(f"  {case:30s} -- missing --"); continue
            mode = MODE_OF[mode_key]
            x0, x1, u = ground_truth(cd)
            ns = min(n_steps, len(x1))
            m = dict(mpt, f_o_s=float(fos), use_prev_results=bool(use_prev))

            # Clip each frame to the frame bounds, as the real detector would,
            # and carry the surviving bead ids so `correct` stays meaningful.
            shape = frame_shape(cd, dim)
            seq, seq_ids = [], []
            for c in [x0] + x1[:ns]:
                keep = in_bounds(c, shape)
                seq.append(c[keep])
                seq_ids.append(np.flatnonzero(keep))

            session, dt = _run_coords(seq, mode, m, dim)
            agg = []
            for k, fr in enumerate(session.frame_results):
                ref = 0 if mode == T.TrackingMode.CUMULATIVE else k
                # true B->A displacement, looked up by bead id
                expected = -(x1[k][seq_ids[k + 1]]
                             - (x0 if ref == 0 else x1[ref - 1])[seq_ids[k + 1]])
                q = link_quality(fr.track_a2b, fr.disp_b2a,
                                 seq[ref], seq[k + 1],
                                 seq_ids[ref], seq_ids[k + 1], expected)
                # Geometric ceiling: A beads whose partner is still in frame.
                # ratio can never exceed this, so ratio/ceiling is the share of
                # *trackable* particles the linker actually recovered.
                shared = len(np.intersect1d(seq_ids[ref], seq_ids[k + 1],
                                            assume_unique=True))
                q["ceiling"] = shared / max(len(seq_ids[ref]), 1)
                q["efficiency"] = q["ratio"] / max(q["ceiling"], 1e-9)
                agg.append(q)
            worst_ratio = min(a["ratio"] for a in agg)
            worst_eff = min(a["efficiency"] for a in agg)
            worst_corr = min(a["correct"] for a in agg)
            worst_rms = max(a["rms"] for a in agg)
            worst_p95 = max(a["p95"] for a in agg)
            umax = float(np.abs(u[ns - 1]).max())
            print(f"  {case:30s} {fig:>4s} {mode_key:>6s} {fos:7.0f} {ns:5d} "
                  f"{worst_ratio:7.4f} {worst_eff:7.4f} {worst_corr:8.4f} "
                  f"{worst_rms:9.2e} {worst_p95:9.2e} {umax:8.2f} {dt:7.2f}")
            out[f"{dim}d/{mode_key}/{case}"] = dict(
                figure=fig, f_o_s=float(fos), steps=ns,
                worst_ratio=worst_ratio, worst_efficiency=worst_eff,
                worst_correct=worst_corr, worst_rms=worst_rms,
                worst_p95=worst_p95, seconds=round(dt, 2), per_step=agg)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Suite A — the whole pipeline, images in / tracks out
# ──────────────────────────────────────────────────────────────────────

def suite_pipeline(quick: bool) -> dict:
    print("\n" + "=" * 92)
    print("SUITE pipeline — images/volumes in, tracks out (paper Fig. 3)")
    print("  Links are scored against the nearest true bead of each detection, so")
    print("  `correct` folds in detection error as well as linking error.")
    print("=" * 92)
    out: dict = {}
    n_steps = 3 if quick else 8

    for dim, root, cases, bead, mpt, loader in (
            (3, SYN3, CASES_3D, BEAD3, MPT3, load_volume),
            (2, SYN2, CASES_2D, BEAD2, MPT2, load_image)):
        print(f"\n{dim}-D   (paper Table 3 per-case mode / f_o_s)")
        print(f"  {'case':30s} {'mode':>6s} {'f_o_s':>7s} {'steps':>5s} "
              f"{'ratio':>7s} {'correct':>8s} {'rms_px':>8s} {'p95_px':>8s} {'sec':>7s}")
        for case, mode_key, fos, use_prev, _fig in (cases[:1] if quick else cases):
            cd = root / case
            if not cd.exists():
                print(f"  {case:30s} -- missing --"); continue
            x0, x1, _ = ground_truth(cd)
            ns = min(n_steps, len(x1))
            frames = [loader(cd, k + 1) for k in range(ns + 1)]

            cfg = T.TrackingConfig(mode=MODE_OF[mode_key], strain_n_neighbors=0,
                                   **dict(mpt, f_o_s=float(fos),
                                                 use_prev_results=bool(use_prev)))
            tracker = T.SerialTracker(T.DetectionConfig(**bead), cfg)
            t0 = time.perf_counter()
            session = tracker.track_images(frames)
            dt = time.perf_counter() - t0

            # Assign each detection the id of its nearest true bead. Detections
            # further than 1 px from every bead get id -1 and can never be
            # "correct" — that is the honest accounting.
            truths = [x0] + x1[:ns]
            det_ids = []
            det_coords = [session.coords_ref] + [f.coords_b for f in session.frame_results]
            for c, tr in zip(det_coords, truths):
                d, i = cKDTree(tr).query(c)
                det_ids.append(np.where(d < 1.0, i, -1))

            cum = MODE_OF[mode_key] == T.TrackingMode.CUMULATIVE
            agg = []
            for k, fr in enumerate(session.frame_results):
                ref = 0 if cum else k
                ca, cb = det_coords[ref], det_coords[k + 1]
                # true B→A displacement of each detected B particle, from its bead
                ib = det_ids[k + 1]
                expected = np.zeros_like(cb)
                ok = ib >= 0
                expected[ok] = -(truths[k + 1][ib[ok]] - truths[ref][ib[ok]])
                agg.append(link_quality(fr.track_a2b, fr.disp_b2a, ca, cb,
                                        det_ids[ref], det_ids[k + 1], expected))
            wr = min(a["ratio"] for a in agg)
            wc = min(a["correct"] for a in agg)
            wrms = max(a["rms"] for a in agg)
            wp95 = max(a["p95"] for a in agg)
            print(f"  {case:30s} {mode_key:>6s} {fos:7.0f} {ns:5d} {wr:7.4f} "
                  f"{wc:8.4f} {wrms:8.4f} {wp95:8.4f} {dt:7.2f}")
            out[f"{dim}d/{case}"] = dict(mode=mode_key, f_o_s=float(fos),
                                         steps=ns, worst_ratio=wr,
                                         worst_correct=wc, worst_rms=wrms,
                                         worst_p95=wp95, seconds=round(dt, 2),
                                         per_step=agg)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Suite B — golden MATLAB parity
# ──────────────────────────────────────────────────────────────────────

def suite_parity(quick: bool) -> dict:
    print("\n" + "=" * 92)
    print("SUITE parity — vs FranckLab's own MATLAB output on the same volumes")
    print("=" * 92)
    out: dict = {}
    for case in (GOLDEN_3D[:1] if quick else GOLDEN_3D):
        cd = SYN3 / case
        gold = golden(cd) if cd.exists() else None
        if gold is None:
            print(f"  {case:30s} -- no results_3D_hardpar.mat --"); continue
        x0, x1, u = ground_truth(cd)
        det = T.ParticleDetector(T.DetectionConfig(**BEAD3))

        print(f"\n  {case}")
        print(f"    {'frame':>5s} {'N_py':>6s} {'N_ml':>6s} {'matched':>8s} "
              f"{'py-ml med':>10s} {'py err':>8s} {'ml err':>8s}")
        frames = [0, 1] if quick else [0, 1, 2, len(x1) // 2, len(x1)]
        rows = []
        for fi in frames:
            if fi >= len(gold["parCoord"]):
                continue
            vol = load_volume(cd, fi + 1)
            truth = x0 if fi == 0 else x1[fi - 1]
            py = T.ParticleDetector.clip_to_bounds(det.detect(vol), vol.shape)
            ml = gold["parCoord"][fi]
            # pair Python detections to MATLAB detections
            d, _ = cKDTree(ml).query(py)
            paired = d < 1.0
            mpy = localisation_error(py, truth)
            mml = localisation_error(ml, truth)
            print(f"    {fi:5d} {len(py):6d} {len(ml):6d} "
                  f"{paired.sum():8d} {np.median(d[paired]):10.4f} "
                  f"{mpy['med']:8.4f} {mml['med']:8.4f}")
            rows.append(dict(frame=fi, n_py=len(py), n_ml=len(ml),
                             paired=int(paired.sum()),
                             py_vs_ml_med=float(np.median(d[paired])),
                             py_err=mpy, ml_err=mml))
        out[case] = rows

        # MATLAB's own displacement error against the imposed field — the bar the
        # port has to clear. The shipped results files do not record which mode
        # produced them, so score against both and report the one that fits.
        rows_ml = []
        for k in range(min(3 if quick else 8, len(gold["uvw_B2A"]))):
            uv = gold["uvw_B2A"][k]
            d, i = cKDTree(x1[k]).query(gold["parCoord"][k + 1])
            ok = d < 1.0
            errs = {}
            for label, ref in (("inc", x0 if k == 0 else x1[k - 1]),
                               ("accum", x0)):
                step = -(x1[k] - ref)
                errs[label] = float(np.sqrt(np.mean(
                    np.sum((uv[ok] - step[i[ok]]) ** 2, axis=1))))
            rows_ml.append(errs)
        best = min(("inc", "accum"),
                   key=lambda m: max(r[m] for r in rows_ml)) if rows_ml else "inc"
        print(f"    MATLAB disp RMS vs imposed field ({best} mode fits): "
              f"{['%.4f' % r[best] for r in rows_ml]}")
        out[case + "/matlab_disp_rms"] = dict(mode=best,
                                              rms=[r[best] for r in rows_ml])
    return out


# ──────────────────────────────────────────────────────────────────────
#  Suite C — invariants (no reference data needed)
# ──────────────────────────────────────────────────────────────────────

def suite_invariant(quick: bool) -> dict:
    print("\n" + "=" * 92)
    print("SUITE invariant — a correct global step reproduces a LINEAR field exactly")
    print("=" * 92)
    rng = np.random.default_rng(0)
    out: dict = {}
    for ndim in (2, 3):
        pts = rng.uniform(0, 200, size=(800, ndim))
        M = rng.normal(scale=0.02, size=(ndim, ndim))
        b = rng.normal(size=ndim)
        disp = pts @ M + b
        step = np.full(ndim, 20.0)
        for sm in (0.0, 1e-2, 1e-1, 1.0):
            grids, dg = T.scatter_to_grid_multi(pts, disp, step, sm, None)
            back = T._interp_grid_to_points(grids, dg, pts)
            err = float(np.abs(back - disp).max())
            print(f"  {ndim}-D smoothness={sm:<5}  grid={dg.shape[1:]}  "
                  f"max|round-trip err| = {err:.3e}")
            out[f"{ndim}d/sm{sm}"] = err
            assert err < 1e-6, f"linear field not reproduced ({err:.3e})"
        # transverse invariance: displacement purely along axis 0
        d0 = np.zeros_like(pts)
        d0[:, 0] = 3.7
        grids, dg = T.scatter_to_grid_multi(pts, d0, step, 1e-1, None)
        back = T._interp_grid_to_points(grids, dg, pts)
        tr = float(np.abs(back[:, 1:]).max())
        print(f"  {ndim}-D pure-axis0 translation: max|transverse| = {tr:.3e}")
        out[f"{ndim}d/transverse"] = tr
        assert tr < 1e-6
    return out


SUITES = {
    "detect": suite_detect,
    "link": suite_link,
    "pipeline": suite_pipeline,
    "parity": suite_parity,
    "invariant": suite_invariant,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all",
                    choices=list(SUITES) + ["all"])
    ap.add_argument("--quick", action="store_true",
                    help="one case per group, fewer steps")
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    ap.add_argument("--verbose", action="store_true", help="show kernel logging")
    args = ap.parse_args()

    # Windows consoles default to cp1252; the report uses box drawing and arrows.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if not args.verbose:
        logging.disable(logging.CRITICAL)

    if not D3.exists() and not D2.exists():
        print(f"Neither dataset found under {DATA}.\n"
              f"  expected {D3}\n       and {D2}", file=sys.stderr)
        return 2

    names = list(SUITES) if args.suite == "all" else [args.suite]
    results: dict = {}
    t0 = time.perf_counter()
    for n in names:
        results[n] = SUITES[n](args.quick)
    print(f"\nTotal {time.perf_counter() - t0:.1f}s")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
