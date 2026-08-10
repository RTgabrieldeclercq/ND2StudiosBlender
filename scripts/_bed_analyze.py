"""Packed-bed analysis over a whole ND2 series — cluster-runnable, array-parallel, resumable.

See CodeLog/ClaudesPlan/V2.23_bed_dynamics_batch.md for the plan this implements and
CodeLog/live/evidence_log.md for why each parameter is what it is.

MODES
    convert   read the ND2 once and write a chunked zarr store (do this on a transfer node)
    analyze   segment one plane per (position, timepoint, z) and write one JSON each
    link      link granule instances through time per (position, z)
    reduce    aggregate every plane JSON into tables, optionally figures

USAGE
    export PYTHONUTF8=1
    export ND2_PATH=/scratch/$USER/bed.nd2

    python scripts/_bed_analyze.py convert --out results/ --z 0-3
    python scripts/_bed_analyze.py analyze --out results/ --task-id $SLURM_ARRAY_TASK_ID \
                                           --n-tasks 64 --skip-existing
    python scripts/_bed_analyze.py link    --out results/
    python scripts/_bed_analyze.py reduce  --out results/ --figures

    # check the partition without touching data
    python scripts/_bed_analyze.py analyze --out results/ --n-tasks 64 --dry-run

WHY ONE JSON PER PLANE
    Array-parallel with no coordination, resumable with --skip-existing, and a single corrupt
    plane fails alone instead of losing the run.

CALIBRATION -- every number here is measured, not assumed.  See the evidence log entry ids.
    granule seeding      intensity h-maxima, sigma 4.0 px, h 0.15 of the interior span
                         88.9% held-out exact count against 585 hand labels          [e043]
    single-granule area  S 2078, M 3599, L 4030 um^2, from label-confirmed singles    [e040]
    phantom-inert fix    keep components >= 0.30 x single-granule area                [e047]
    voxel size           XY 1.7182777601481225 um, Z step 40 um
    timepoint interval   47.3 min (the file declares periodMs = 0.0 and carries none)  [e044]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PX_UM = 1.7182777601481225
Z_STEP_UM = 40.0
DT_MIN = 47.3
SIGMA_PX, H_FRAC = 4.0, 0.15
MIN_FRAC = 0.30
SAT_DN = 4095
CH_GFP, CH_F, CH_I, CH_TD = 0, 1, 2, 3
CELL_MIN_PX, CELL_MAX_PX = 6, 600
VOID_FAR_UM = 60.0

# median area of a label-confirmed SINGLE granule, per size class.  This replaces every
# nominal/sieve size in the workflow -- see V2.23 section 0.
A_SINGLE = {"S": 2078.0, "M": 3599.0, "L": 4030.0}

# The 21-condition design.  P = M - 1, verified against three falsifiable predictions.
# (condition, F fraction, F size class, I size class or None)
DESIGN: List[Tuple[str, float, str, Optional[str]]] = [
    ("M01", 1.00, "S", None), ("M02", 0.50, "S", "S"), ("M03", 0.75, "S", "S"),
    ("M04", 0.50, "S", "M"),  ("M05", 0.75, "S", "M"), ("M06", 0.50, "S", "L"),
    ("M07", 0.75, "S", "L"),
    ("M08", 1.00, "M", None), ("M09", 0.50, "M", "S"), ("M10", 0.75, "M", "S"),
    ("M11", 0.50, "M", "M"),  ("M12", 0.75, "M", "M"), ("M13", 0.50, "M", "L"),
    ("M14", 0.75, "M", "L"),
    ("M15", 1.00, "L", None), ("M16", 0.50, "L", "S"), ("M17", 0.75, "L", "S"),
    ("M18", 0.50, "L", "M"),  ("M19", 0.75, "L", "M"), ("M20", 0.50, "L", "L"),
    ("M21", 0.75, "L", "L"),
]


# ───────────────────────────── helpers ─────────────────────────────

def parse_range(s: str) -> List[int]:
    """'0-20' | '0,3,7' | '0-4,9' | '5' -> sorted unique ints."""
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def nd2_path(explicit: Optional[str] = None) -> str:
    p = explicit or os.environ.get("ND2_PATH")
    if not p:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [f for f in os.listdir(here) if f.lower().endswith(".nd2")]
        if len(cands) == 1:
            p = os.path.join(here, cands[0])
    if not p or not os.path.exists(p):
        raise SystemExit("SKIP: no ND2 found. Set ND2_PATH or pass --nd2.")
    return p


def design_for(position: int) -> Tuple[str, float, str, Optional[str]]:
    if not 0 <= position < len(DESIGN):
        raise ValueError(f"position {position} outside the 21-condition design")
    return DESIGN[position]


def plane_json(out: str, p: int, t: int, z: int) -> str:
    return os.path.join(out, "planes", f"P{p:02d}_T{t:02d}_Z{z}.json")


# ───────────────────────── segmentation core ─────────────────────────
# Deliberately kept in one place so `analyze` and any node port share it exactly.

def detect_cells(gfp: np.ndarray):
    """Small compact bright objects in GFP.  These are cells, ~11 um, not passive markers."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    top = np.clip(gfp - ndi.median_filter(gfp, size=21), 0, None)
    pos = top[top > 0]
    if pos.size < 500:
        return np.zeros(gfp.shape, bool)
    thr = max(float(threshold_otsu(pos)),
              float(np.median(pos) + 6.0 * (np.percentile(pos, 84) - np.median(pos))))
    lab, n = ndi.label(top > thr)
    if n == 0:
        return np.zeros(gfp.shape, bool)
    area = np.bincount(lab.ravel())
    keep = (area >= CELL_MIN_PX) & (area <= CELL_MAX_PX)
    keep[0] = False
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if not keep[i]:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if area[i] / float(h * w) < 0.45 or max(h, w) / max(min(h, w), 1) > 3.0:
            keep[i] = False
    return keep[lab]


def size_filter(mask: np.ndarray, a_single_um2: float) -> np.ndarray:
    """Keep only components >= MIN_FRAC x a single granule.

    This is the phantom-inert fix.  A channel with no real signal has an Otsu cut a few DN
    above its own pedestal and yields scattered sub-granule specks; a real granule is
    granule-sized.  Two earlier attempts failed: plain Otsu invented 9% phantom inert at a
    pure-functional well, and a background+10-sigma threshold floor pushed the cut to 4791 DN,
    above the 4095 sensor ceiling, emptying the functional phase outright.  [e047]
    """
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    area = np.bincount(lab.ravel()) * PX_UM * PX_UM
    keep = area >= MIN_FRAC * a_single_um2
    keep[0] = False
    return keep[lab]


def solid_phase(ch: np.ndarray, a_single_um2: float, cells: Optional[np.ndarray]):
    """Otsu -> fill holes -> (optionally drop cells) -> size filter."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    thr = float(threshold_otsu(ch))
    raw = ndi.binary_fill_holes(ch > thr)
    if cells is not None:
        raw = raw & ~cells
    return size_filter(raw, a_single_um2), thr, float(raw.mean())


def granule_instances(ch: np.ndarray, a_single_um2: float):
    """The validated finder: intensity h-maxima seeds flooded on intensity.  [e043]

    `cells` is deliberately NOT subtracted.  Cell pixels are removed from the PHASE masks so
    the three phases partition the field, but a cell crawling on a granule is still ON it --
    punching its pixels out of the instance label makes every cell's centroid land in a hole
    and read as "void", which happened at all six conditions.  Two masks, two purposes. [e048]
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    thr = float(threshold_otsu(ch))
    fg = ndi.binary_fill_holes(ch > thr)
    if fg.sum() < 200:
        return np.zeros(ch.shape, np.int32), 0
    sm = ndi.gaussian_filter(ch, SIGMA_PX)
    span = float(np.percentile(sm[fg], 99) - thr)
    seeds = ndi.label(h_maxima(np.where(fg, sm, 0.0), H_FRAC * span) & fg)[0]
    lab = watershed(-sm, seeds, mask=fg)
    n_raw = int(lab.max())
    if n_raw == 0:
        return lab.astype(np.int32), 0
    area = np.bincount(lab.ravel(), minlength=n_raw + 1) * PX_UM * PX_UM
    keep = area >= MIN_FRAC * a_single_um2
    keep[0] = False
    out, _ = ndi.label(keep[lab])
    return out.astype(np.int32), n_raw


def instance_rows(lab: np.ndarray) -> List[Dict]:
    """Per-instance geometry.  Feret/solidity are the merge signatures scored in e041."""
    from scipy import ndimage as ndi
    from scipy.spatial import ConvexHull
    n = int(lab.max())
    if n == 0:
        return []
    rows = []
    edt = ndi.distance_transform_edt(lab > 0, sampling=(PX_UM, PX_UM))
    for oid, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == oid
        ys, xs = np.nonzero(m)
        if ys.size < 4:
            continue
        A = float(m.sum()) * PX_UM * PX_UM
        h = float(np.ptp(ys) + 1) * PX_UM
        w = float(np.ptp(xs) + 1) * PX_UM
        try:
            hull = ConvexHull(np.column_stack([xs, ys]).astype(float))
            solid = A / (float(hull.volume) * PX_UM * PX_UM)
        except Exception:
            solid = float("nan")
        rows.append(dict(
            id=int(oid),
            y=float(ys.mean() + sl[0].start), x=float(xs.mean() + sl[1].start),
            area_um2=round(A, 1),
            feret_um=round(max(h, w), 1), minor_um=round(min(h, w), 1),
            inscribed_d_um=round(2.0 * float((edt[sl] * m).max()), 1),
            solidity=(round(solid, 4) if np.isfinite(solid) else None)))
    return rows


def analyze_plane(vol: np.ndarray, position: int, timepoint: int, z: int) -> Dict:
    """One plane, all channels.  `vol` is (C, Y, X) float32.  Returns a JSON-safe dict."""
    from scipy import ndimage as ndi
    cond, f_frac, fcls, icls = design_for(position)
    aF = A_SINGLE[fcls]
    aI = A_SINGLE[icls] if icls else A_SINGLE["M"]
    gfp, chF, chI = vol[CH_GFP], vol[CH_F], vol[CH_I]

    cells = detect_cells(gfp)
    F, thrF, rawF = solid_phase(chF, aF, cells)
    I0, thrI, rawI = solid_phase(chI, aI, cells)
    I = I0 & ~F
    void = ~F & ~I & ~cells
    # second inert mask WITHOUT the cells punched out, for the parent lookup only  [e048]
    I_look, _, _ = solid_phase(chI, aI, None)
    I_look = I_look & ~F

    GL, n_raw = granule_instances(chF, aF)
    grows = instance_rows(GL)

    # cells: position, brightness, parent granule, parent-local coordinates
    clab, ncell = ndi.label(cells)
    crows = []
    if ncell:
        cen = np.atleast_2d(np.array(ndi.center_of_mass(cells, clab,
                                                        np.arange(1, ncell + 1))))
        gcen = {r["id"]: (r["y"], r["x"]) for r in grows}
        for i, (cy, cx) in enumerate(cen, start=1):
            yi = int(np.clip(round(cy), 0, cells.shape[0] - 1))
            xi = int(np.clip(round(cx), 0, cells.shape[1] - 1))
            gid = int(GL[yi, xi])
            on_i = bool(I_look[yi, xi]) and gid == 0
            pu = pv = None
            if gid and gid in gcen:
                pu = round((cy - gcen[gid][0]) * PX_UM, 2)
                pv = round((cx - gcen[gid][1]) * PX_UM, 2)
            m = clab == i
            crows.append(dict(
                id=i, y=round(float(cy), 2), x=round(float(cx), 2),
                area_um2=round(float(m.sum()) * PX_UM * PX_UM, 1),
                gfp=round(float(np.median(gfp[m])), 1),
                f_ch=round(float(np.median(chF[m])), 1),
                i_ch=round(float(np.median(chI[m])), 1),
                parent=gid, on_inert=on_i, parent_u_um=pu, parent_v_um=pv))

    # void attribution, only to phases that actually exist in this field
    hasF, hasI = F.mean() > 0.002, I.mean() > 0.002
    tot = max(float(void.sum()), 1.0)
    vF = vI = vfar = None
    if hasF or hasI:
        dF = (ndi.distance_transform_edt(~F, sampling=(PX_UM, PX_UM)) if hasF else None)
        dI = (ndi.distance_transform_edt(~I, sampling=(PX_UM, PX_UM)) if hasI else None)
        if hasF and hasI:
            near = void & (np.minimum(dF, dI) <= VOID_FAR_UM)
            vF = float((near & (dF <= dI)).sum()) / tot
            vI = float((near & (dI < dF)).sum()) / tot
        elif hasF:
            near = void & (dF <= VOID_FAR_UM)
            vF, vI = float(near.sum()) / tot, 0.0
        else:
            near = void & (dI <= VOID_FAR_UM)
            vF, vI = 0.0, float(near.sum()) / tot
        vfar = float((void & ~near).sum()) / tot

    # ── QC.  Level test here; the continuity test needs neighbours, so it runs in `reduce`.
    fF, fI = float(F.mean()), float(I.mean())
    qc = []
    if f_frac == 1.0 and fI > 0.01:
        qc.append("phantom_inert")                 # the gate this fix exists for
    if f_frac < 1.0 and rawI < 0.02:
        qc.append("level_no_inert_signal")         # M21's failure mode
    if fF < 0.01:
        qc.append("level_no_functional_signal")
    sat = float((chF >= SAT_DN).mean())
    if sat > 0.01:
        qc.append("f_channel_saturated")           # F intensity unusable, geometry fine

    return dict(
        schema=2, position=position, timepoint=timepoint, z=z,
        condition=cond, f_frac=f_frac, f_class=fcls, i_class=icls,
        px_um=PX_UM, z_step_um=Z_STEP_UM, dt_min=DT_MIN,
        thr_f=round(thrF, 1), thr_i=round(thrI, 1),
        frac_f=round(fF, 5), frac_i=round(fI, 5),
        frac_cells=round(float(cells.mean()), 6), frac_void=round(float(void.mean()), 5),
        frac_f_raw=round(rawF, 5), frac_i_raw=round(rawI, 5),
        sat_frac=round(sat, 5),
        n_granules=len(grows), n_watershed_raw=n_raw,
        n_cells=len(crows),
        void_to_f=(round(vF, 4) if vF is not None else None),
        void_to_i=(round(vI, 4) if vI is not None else None),
        void_unattributed=(round(vfar, 4) if vfar is not None else None),
        qc_flags=qc, granules=grows, cells=crows)


# ───────────────────────────── linking ─────────────────────────────

def link_instances(lab_a: np.ndarray, lab_b: np.ndarray, *, iou_min: float = 0.35):
    """Hungarian assignment on mask IoU.  Returns (a_to_b, iou) with 0 meaning unmatched.

    Centroid nearest-neighbour reached only 74-93% mutual matching, and where it was weakest
    the parent-frame migration inverted -- subtracting the parent added more noise than it
    removed.  Granules move 8-19 um per frame while being 50-70 um across, so consecutive
    masks overlap heavily and IoU is the far stronger cue: it uses shape and extent, and it
    degrades gracefully when a granule is split differently between frames.  [e049]
    """
    from scipy.optimize import linear_sum_assignment
    na, nb = int(lab_a.max()), int(lab_b.max())
    if na == 0 or nb == 0:
        return np.zeros(na, int), np.zeros(na)
    inter = np.zeros((na, nb), np.int64)
    both = (lab_a > 0) & (lab_b > 0)
    if both.any():
        np.add.at(inter, (lab_a[both] - 1, lab_b[both] - 1), 1)
    area_a = np.bincount(lab_a.ravel(), minlength=na + 1)[1:].astype(np.int64)
    area_b = np.bincount(lab_b.ravel(), minlength=nb + 1)[1:].astype(np.int64)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    ri, ci = linear_sum_assignment(-iou)
    out = np.zeros(na, int)
    got = np.zeros(na)
    for a, b in zip(ri, ci):
        if iou[a, b] >= iou_min:
            out[a] = b + 1
            got[a] = iou[a, b]
    return out, got


# ───────────────────────────── modes ─────────────────────────────

def mode_convert(a) -> int:
    import nd2
    try:
        import zarr
    except ImportError:
        raise SystemExit("convert needs zarr: pip install zarr")
    path = nd2_path(a.nd2)
    zs = parse_range(a.z)
    store = a.to or os.path.join(a.out, "bed.zarr")
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
    with nd2.ND2File(path) as f:
        dk = f.to_dask()
        nT, nP, nZ, nC = dk.shape[:4]
        Y, X = dk.shape[-2:]
        zs = [z for z in zs if z < nZ]
        print(f"source {path}\n  T={nT} P={nP} Z={nZ} C={nC} Y={Y} X={X}")
        print(f"  writing z={zs} -> {store}")
        g = zarr.open(store, mode="w")
        arr = g.create_dataset("plane", shape=(nT, nP, len(zs), nC, Y, X),
                               chunks=(1, 1, 1, 1, Y, X), dtype="f4", overwrite=True)
        arr.attrs.update(px_um=PX_UM, z_step_um=Z_STEP_UM, dt_min=DT_MIN,
                         z_indices=zs, source=os.path.basename(path))
        t0 = time.time()
        n = 0
        for p in range(nP):
            for t in range(nT):
                blk = np.asarray(dk[t, p][zs]).astype(np.float32)   # (len(zs), C, Y, X)
                for k in range(len(zs)):
                    arr[t, p, k] = blk[k]
                n += len(zs)
            el = time.time() - t0
            print(f"  P{p:02d} done  {n} planes  {el:6.1f}s  "
                  f"eta {el / max(p + 1, 1) * (nP - p - 1):6.1f}s", flush=True)
    print(f"wrote {store}")
    return 0


def _reader(a):
    """Returns (get(p,t,z) -> (C,Y,X), close, grid) preferring zarr over the ND2."""
    store = a.zarr or os.path.join(a.out, "bed.zarr")
    if os.path.exists(store):
        import zarr
        g = zarr.open(store, mode="r")
        arr = g["plane"]
        zidx = list(arr.attrs.get("z_indices", range(arr.shape[2])))
        print(f"reading {store} (zarr, z={zidx})")

        def get(p, t, z):
            return np.asarray(arr[t, p, zidx.index(z)])
        return get, (lambda: None), dict(nT=arr.shape[0], nP=arr.shape[1], zs=zidx)
    import nd2
    path = nd2_path(a.nd2)
    print(f"reading {path} (ND2 direct -- consider `convert` for parallel runs)")
    f = nd2.ND2File(path)
    dk = f.to_dask()

    def get(p, t, z):
        return np.asarray(dk[t, p, z]).astype(np.float32)
    return get, f.close, dict(nT=dk.shape[0], nP=dk.shape[1],
                              zs=list(range(dk.shape[2])))


def mode_analyze(a) -> int:
    tasks = [(p, t, z) for p in parse_range(a.positions)
             for t in parse_range(a.timepoints) for z in parse_range(a.z)]
    tasks.sort()
    mine = tasks
    if a.n_tasks and a.n_tasks > 1:
        if a.task_id is None:
            raise SystemExit("--n-tasks needs --task-id (use $SLURM_ARRAY_TASK_ID)")
        mine = tasks[a.task_id::a.n_tasks]
    os.makedirs(os.path.join(a.out, "planes"), exist_ok=True)
    if a.skip_existing:
        mine = [k for k in mine if not os.path.exists(plane_json(a.out, *k))]
    print(f"{len(tasks)} planes total; this task has {len(mine)}"
          + (f" (id {a.task_id} of {a.n_tasks})" if a.n_tasks else ""))
    if a.dry_run:
        for k in mine[:8]:
            print("   ", plane_json(a.out, *k))
        if len(mine) > 8:
            print(f"    ... and {len(mine) - 8} more")
        return 0
    if not mine:
        print("nothing to do")
        return 0
    get, close, grid = _reader(a)
    t0, done = time.time(), 0
    try:
        for (p, t, z) in mine:
            try:
                rec = analyze_plane(get(p, t, z), p, t, z)
            except Exception as e:                      # one bad plane must not kill the run
                rec = dict(schema=2, position=p, timepoint=t, z=z,
                           qc_flags=["analysis_failed"], error=repr(e))
                print(f"  FAILED P{p:02d} T{t:02d} Z{z}: {e!r}", flush=True)
            with open(plane_json(a.out, p, t, z), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, separators=(",", ":"), sort_keys=True)
            done += 1
            if done % 10 == 0 or done == len(mine):
                el = time.time() - t0
                print(f"  {done}/{len(mine)}  {el:6.1f}s  "
                      f"{el / done:4.1f}s/plane  eta {el / done * (len(mine) - done):6.1f}s",
                      flush=True)
    finally:
        close()
    return 0


def mode_link(a) -> int:
    """Link granule instances through time per (position, z), and gate the result."""
    get, close, grid = _reader(a)
    os.makedirs(os.path.join(a.out, "links"), exist_ok=True)
    rep = []
    try:
        for p in parse_range(a.positions):
            cond, f_frac, fcls, icls = design_for(p)
            aF = A_SINGLE[fcls]
            for z in parse_range(a.z):
                ts = parse_range(a.timepoints)
                prev = None
                pairs, ious, matched = [], [], []
                for t in ts:
                    lab, _ = granule_instances(get(p, t, z)[CH_F], aF)
                    if prev is not None:
                        m, io = link_instances(prev, lab, iou_min=a.iou_min)
                        pairs.append([int(x) for x in m])
                        ious.append([round(float(x), 3) for x in io])
                        matched.append(float((m > 0).mean()) if m.size else 0.0)
                    prev = lab
                frac = float(np.mean(matched)) if matched else 0.0
                ok = frac >= a.match_min
                rep.append(dict(position=p, z=z, condition=cond,
                                match_rate=round(frac, 4), gate=("PASS" if ok else "FAIL")))
                with open(os.path.join(a.out, "links",
                                       f"P{p:02d}_Z{z}.json"), "w", encoding="utf-8") as fh:
                    json.dump(dict(position=p, z=z, condition=cond, timepoints=ts,
                                   iou_min=a.iou_min, match_rate=frac,
                                   a_to_b=pairs, iou=ious), fh, separators=(",", ":"))
                print(f"  P{p:02d} Z{z} {cond}: match {frac:.1%} "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close()
    npass = sum(1 for r in rep if r["gate"] == "PASS")
    print(f"\nLINK GATE: {npass}/{len(rep)} (position,z) series at or above "
          f"{a.match_min:.0%} mutual match")
    print("  A series that FAILS must not be used for parent-frame migration or per-granule\n"
          "  velocity: the parent's identity is flickering and subtracting it adds noise.")
    with open(os.path.join(a.out, "link_gate.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return 0 if npass == len(rep) else 1


def mode_reduce(a) -> int:
    d = os.path.join(a.out, "planes")
    files = sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
    if not files:
        raise SystemExit(f"no plane JSONs in {d}")
    recs = []
    for f in files:
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            recs.append(json.load(fh))
    ok = [r for r in recs if "analysis_failed" not in r.get("qc_flags", [])]
    print(f"{len(recs)} planes, {len(recs) - len(ok)} failed analysis")

    # ── continuity QC: a phase fraction that STEPS between adjacent frames is an
    # acquisition event, not physics.  M04's void jumped +0.32 in one 47-minute step.
    by_pz: Dict[Tuple[int, int], List[Dict]] = {}
    for r in ok:
        by_pz.setdefault((r["position"], r["z"]), []).append(r)
    n_step = 0
    for k, rows in by_pz.items():
        rows.sort(key=lambda r: r["timepoint"])
        for i in range(1, len(rows)):
            if abs(rows[i]["frac_void"] - rows[i - 1]["frac_void"]) > a.step_max:
                rows[i].setdefault("qc_flags", []).append("continuity_step")
                n_step += 1
    print(f"continuity QC: {n_step} planes flagged with a void step > {a.step_max}")

    good = [r for r in ok if not any(q in r.get("qc_flags", []) for q in
                                     ("phantom_inert", "level_no_inert_signal",
                                      "level_no_functional_signal", "continuity_step"))]
    print(f"{len(good)} planes pass every QC test "
          f"({len(good) / max(len(recs), 1):.0%})")

    # per-condition table
    rows_out = []
    for cond, f_frac, fcls, icls in DESIGN:
        sel = [r for r in good if r["condition"] == cond]
        if not sel:
            rows_out.append(dict(condition=cond, f_frac=f_frac, f_class=fcls,
                                 i_class=icls, n_planes=0))
            continue
        first = [r for r in sel if r["timepoint"] == min(x["timepoint"] for x in sel)]
        last = [r for r in sel if r["timepoint"] == max(x["timepoint"] for x in sel)]

        def mu(rs, k):
            v = [x[k] for x in rs if x.get(k) is not None]
            return round(float(np.mean(v)), 5) if v else None
        rows_out.append(dict(
            condition=cond, f_frac=f_frac, f_class=fcls, i_class=icls,
            n_planes=len(sel),
            frac_f=mu(sel, "frac_f"), frac_i=mu(sel, "frac_i"),
            frac_void=mu(sel, "frac_void"), frac_cells=mu(sel, "frac_cells"),
            d_void=(None if not (first and last)
                    else round(mu(last, "frac_void") - mu(first, "frac_void"), 5)),
            void_to_f=mu(sel, "void_to_f"), void_to_i=mu(sel, "void_to_i"),
            void_unattributed=mu(sel, "void_unattributed"),
            n_granules=mu(sel, "n_granules"), n_cells=mu(sel, "n_cells"),
            # void share normalised by that phase's own solid share -- the phase-specific
            # packing measure.  Inert ran ~45% higher than functional on two conditions.
            void_per_solid_f=(None if not mu(sel, "frac_f") else
                              round((mu(sel, "void_to_f") or 0) / mu(sel, "frac_f"), 3)),
            void_per_solid_i=(None if not mu(sel, "frac_i") else
                              round((mu(sel, "void_to_i") or 0) / mu(sel, "frac_i"), 3))))

    with open(os.path.join(a.out, "by_condition.json"), "w", encoding="utf-8") as fh:
        json.dump(rows_out, fh, indent=1)
    cols = ["condition", "f_frac", "f_class", "i_class", "n_planes", "frac_f", "frac_i",
            "frac_void", "d_void", "void_to_f", "void_to_i", "void_unattributed",
            "void_per_solid_f", "void_per_solid_i", "n_granules", "n_cells"]
    with open(os.path.join(a.out, "by_condition.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows_out:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    print(f"wrote {os.path.join(a.out, 'by_condition.csv')}")

    print(f"\n{'cond':>5} {'F':>7}{'I':>7}{'void':>7}{'dVoid':>8} "
          f"{'v/solid F':>10}{'v/solid I':>10} {'planes':>7}")
    for r in rows_out:
        if not r["n_planes"]:
            print(f"{r['condition']:>5} {'-- no planes passed QC --':>50}")
            continue
        print(f"{r['condition']:>5} {r['frac_f']:7.3f}{r['frac_i']:7.3f}"
              f"{r['frac_void']:7.3f}{(r['d_void'] or 0):+8.3f} "
              f"{(r['void_per_solid_f'] or 0):10.2f}{(r['void_per_solid_i'] or 0):10.2f} "
              f"{r['n_planes']:7d}")

    if a.figures:
        _reduce_figures(a.out, rows_out, good)
    return 0


def _reduce_figures(out: str, rows_out: List[Dict], good: List[Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [r for r in rows_out if r["n_planes"]]
    if not have:
        print("no conditions passed QC; no figures")
        return
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    CL = {"S": "#38bdf8", "M": "#fbbf24", "L": "#fb7185"}
    for r in have:
        axs[0].scatter(r["f_frac"], r["frac_void"], s=110, c=CL[r["f_class"]],
                       edgecolor="k", zorder=3)
        axs[0].annotate(r["condition"], (r["f_frac"], r["frac_void"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    axs[0].set_xlabel("designed functional fraction"); axs[0].set_ylabel("void fraction")
    axs[0].set_title("Void against the design's dose axis\ncolour = F size class")
    axs[0].grid(alpha=.25)
    for r in have:
        if r["d_void"] is not None:
            axs[1].scatter(r["f_frac"], r["d_void"], s=110, c=CL[r["f_class"]],
                           edgecolor="k", zorder=3)
    axs[1].axhline(0, color="#94a3b8", lw=1.2)
    axs[1].set_xlabel("designed functional fraction")
    axs[1].set_ylabel("change in void over the series")
    axs[1].set_title("Which formulations compact and which dilate")
    axs[1].grid(alpha=.25)
    f_ = [r["void_per_solid_f"] for r in have if r.get("void_per_solid_i")]
    i_ = [r["void_per_solid_i"] for r in have if r.get("void_per_solid_i")]
    if f_ and i_:
        axs[2].scatter(f_, i_, s=110, c="#a78bfa", edgecolor="k", zorder=3)
        lim = [0, max(max(f_), max(i_)) * 1.1]
        axs[2].plot(lim, lim, "--", color="#94a3b8")
        axs[2].set_xlim(lim); axs[2].set_ylim(lim)
        axs[2].set_xlabel("void per unit solid, FUNCTIONAL")
        axs[2].set_ylabel("void per unit solid, INERT")
        axs[2].set_title("Above the line = inert is more loosely packed")
        axs[2].grid(alpha=.25)
    fig.suptitle(f"Bed dynamics over the 21-condition design "
                 f"({len(good)} planes passing QC)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, .93])
    p = os.path.join(out, "by_condition.png")
    plt.savefig(p, dpi=110); plt.close()
    print(f"wrote {p}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Packed-bed analysis over an ND2 series (cluster-runnable)")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, fn in (("convert", mode_convert), ("analyze", mode_analyze),
                     ("link", mode_link), ("reduce", mode_reduce)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--out", default="results")
        s.add_argument("--nd2", default=None, help="defaults to $ND2_PATH")
        s.add_argument("--zarr", default=None, help="defaults to <out>/bed.zarr if present")
        s.add_argument("--positions", default="0-20")
        s.add_argument("--timepoints", default="0-14")
        s.add_argument("--z", default="0-3", help="usable axial range; deeper is attenuated")
        if name == "convert":
            s.add_argument("--to", default=None)
        if name == "analyze":
            s.add_argument("--task-id", type=int, default=None)
            s.add_argument("--n-tasks", type=int, default=None)
            s.add_argument("--skip-existing", action="store_true")
            s.add_argument("--dry-run", action="store_true")
        if name == "link":
            s.add_argument("--iou-min", type=float, default=0.35)
            s.add_argument("--match-min", type=float, default=0.95,
                           help="gate: mutual match rate required per series")
        if name == "reduce":
            s.add_argument("--figures", action="store_true")
            s.add_argument("--step-max", type=float, default=0.10,
                           help="a void jump larger than this between adjacent frames is "
                                "flagged as an acquisition event")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
