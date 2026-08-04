"""Verify CellSAM's `fast` path against upstream — on the REAL model and REAL weights.

Run:  python scripts/_bench_cellsam_fast.py [image.tif]

WHY THIS IS A SCRIPT AND NOT A SELFTEST GROUP
    `nodegraph.selftest` stubs the whole `cellSAM` package (`_stub_cellsam`), because the
    real thing is a multi-hundred-MB checkpoint plus torch plus a GPU. That stub covers the
    GLUE — kwarg forwarding, the singleton, the no-cells repairs, the chunk-error watch —
    but it cannot say anything about NUMERICS, and numerics is the entire question about
    `fast`: it batches the mask decoder, so it is *not* bit-identical and the only honest
    claim is a measured one. This is the same split as `scripts/_bench_nms_numba.py`, which
    checks the numba↔numpy equivalence the fast gate cannot.

    So: run this after upgrading `cellSAM`, `torch` or the weights, and paste the numbers
    into `analysis.segment`'s `fast` socket description if they have moved.

WHAT IT ASSERTS
    * no cell is gained or lost, and no id is renumbered;
    * every per-cell AREA change is within `--max-area-delta` pixels (default 2);
    * `fast` is actually faster (else there is no reason to accept any difference).
    The reference numbers as of 2026-07-31, RTX 3090, 1024² block with 441 cells:
    10.14 s → 1.78 s (5.7x), 10 differing pixels (0.0010%), 11/443 cells changed area by
    exactly 1 px against a median cell of 556 px.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_IMAGE = os.path.expanduser(r"~\Downloads\CellSAM_test.tif")


def _load(path: str) -> np.ndarray:
    import tifffile
    img = tifffile.imread(path)
    while img.ndim > 2:                      # first plane of whatever this is
        img = img[0]
    return np.ascontiguousarray(img)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    ap.add_argument("--size", type=int, default=1024, help="square crop edge (px)")
    ap.add_argument("--max-area-delta", type=int, default=2,
                    help="largest per-cell area change, in px, this will accept")
    args = ap.parse_args(argv[1:])

    from nodegraph.kernels import cellsam_segment as CS
    if not CS.cellsam_available():
        print("cellSAM is not installed — nothing to verify")
        return 0
    try:
        import torch
    except ImportError:
        print("torch is not installed — nothing to verify")
        return 0
    if not torch.cuda.is_available():
        print("no CUDA device — `fast` is a GPU optimization, skipping")
        return 0
    if not os.path.exists(args.image):
        print(f"no test image at {args.image} (pass one as the first argument)")
        return 0

    img = _load(args.image)
    n = min(args.size, min(img.shape))
    crop = np.ascontiguousarray(img[:n, :n])
    dev = torch.cuda.get_device_properties(0).name
    print(f"{dev} | block {crop.shape} {crop.dtype} "
          f"[{int(crop.min())}, {int(crop.max())}]")

    model = CS.get_cellsam_model(device="cuda")

    def run(fast: bool):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lab = CS.segment_plane(crop, model=model, device="cuda", fast=fast)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, lab

    t_ref, ref = run(False)
    t_fast, fast = run(True)

    ids_ref, ids_fast = np.unique(ref), np.unique(fast)
    lost = np.setxor1d(ids_ref, ids_fast)
    npx = int((ref != fast).sum())
    top = int(max(ids_ref.max(), ids_fast.max()))
    a_ref = np.bincount(ref.ravel(), minlength=top + 1)
    a_fast = np.bincount(fast.ravel(), minlength=top + 1)
    live = ids_ref[ids_ref > 0]
    delta = np.abs(a_ref[live].astype(np.int64) - a_fast[live].astype(np.int64))
    median = int(np.median(a_ref[live])) if live.size else 0

    print(f"\n  upstream : {t_ref:6.2f} s   {len(ids_ref) - 1:5d} cells")
    print(f"  fast     : {t_fast:6.2f} s   {len(ids_fast) - 1:5d} cells   "
          f"({t_ref / max(t_fast, 1e-9):.2f}x)")
    print(f"  differing pixels : {npx} ({100 * npx / ref.size:.4f}%)")
    print(f"  cells whose AREA moved : {int((delta > 0).sum())}/{live.size}, "
          f"max {int(delta.max()) if live.size else 0} px "
          f"on a median cell of {median} px")

    ok = True
    if lost.size:
        print(f"\nFAIL: {lost.size} cell id(s) present in one result only: "
              f"{lost[:10].tolist()}")
        ok = False
    if live.size and int(delta.max()) > args.max_area_delta:
        print(f"\nFAIL: a cell's area moved by {int(delta.max())} px, over the "
              f"{args.max_area_delta} px this accepts. `fast` is supposed to be a "
              f"scheduling change, not a different segmentation — investigate before "
              f"shipping it.")
        ok = False
    if t_fast >= t_ref:
        print(f"\nFAIL: `fast` ({t_fast:.2f} s) is not faster than upstream "
              f"({t_ref:.2f} s) — then it is all cost and no benefit; check the GPU is "
              f"really being used.")
        ok = False
    print("\n" + ("CELLSAM FAST PATH VERIFIED" if ok else "CELLSAM FAST PATH FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
