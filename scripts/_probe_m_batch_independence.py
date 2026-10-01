"""Probe: is ``m`` an independent batch axis?

The multi-file design question. If N files are concatenated along the multipoint axis
``m``, then "one pipeline over N files" is free for the whole catalog **iff** every node
treats each ``m`` slice independently — i.e. the result at ``m=k`` in a stacked run equals
the result of running the same graph on file ``k`` alone.

This probe runs a real segmentation chain (gaussian -> threshold -> label) twice:
once over a 2-position stack whose two positions hold DIFFERENT content, and once over
each position on its own. It then compares the label rasters.

It also checks the known pooling lever: ``analysis.threshold``'s ``scope`` Mode. ``plane``
(the default) must stay independent; ``dataset`` is documented to pool over the whole
(m,t,z,c) series and is expected to BLEED across positions -- which, if files are stacked
on ``m``, means it pools across files.

Run:  .venv\\Scripts\\python.exe -B -u scripts\\_probe_m_batch_independence.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.graph import Graph, NodeInstance
from nodegraph.engine import Engine
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import ArrayProvider
from nodegraph.registry import define_node
from nodegraph.registry import OutDataset
from nodegraph.nodes import COMPUTES

SEED_OP = "probe.seedM"          # NOT a real op_key (INV-03)
H = W = 48
OPTICS = {"pixel_size_um": 0.5, "z_step_um": 1.0}


def _blobs(centres, sigma=3.0, amp=1.0) -> np.ndarray:
    """A (H,W) plane with a Gaussian blob at each centre."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    out = np.zeros((H, W), float)
    for (cy, cx) in centres:
        out += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    return out


#: two "files": position 0 has 2 objects, position 1 has 3 -- and position 1 is DIMMER,
#: which is what makes a pooled threshold visibly wrong.
FILE0 = _blobs([(12, 12), (34, 30)], amp=1.0)
FILE1 = _blobs([(10, 36), (30, 10), (38, 38)], amp=0.45)


def _volume(planes) -> np.ndarray:
    """Stack per-position planes into (M,T,Z,C,Y,X)."""
    return np.stack(planes, axis=0)[:, None, None, None, :, :]


def _run(vol: np.ndarray, *, scope: str) -> np.ndarray:
    """Run gaussian -> threshold -> label over ``vol``; return the label raster."""
    m = vol.shape[0]
    ax = AxisSizes(m=m, t=1, z=1, c=1, y=H, x=W)
    env = MetaEnvelope(axes=ax, metadata=dict(OPTICS))
    ds = Dataset(axes=ax, metadata=dict(OPTICS)).with_image(ArrayProvider(vol))

    g = Graph()
    g.add(NodeInstance("S", SEED_OP))
    g.add(NodeInstance("G", "enhance.gaussian", params={"sigma": 0.5}))
    g.add(NodeInstance("T", "analysis.threshold",
                       modes={"method": "otsu", "scope": scope}))
    g.add(NodeInstance("L", "analysis.label"))
    g.connect("S", "G")
    g.connect("G", "T")
    g.connect("T", "L")

    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
    out = eng.pull("L")
    lay = out.get(Domain.VOXEL, "labels")
    assert lay is not None, (
        "analysis.label produced no Voxel 'labels' raster; layers present: "
        f"{[(k) for k in out.attributes]}")
    return np.asarray(lay.values)


def _count(raster: np.ndarray) -> int:
    """Distinct non-zero region ids."""
    return int(len(set(np.unique(raster).tolist()) - {0}))


def main() -> int:
    define_node(SEED_OP, "Seed (probe)", outputs=[OutDataset()])

    stacked = _volume([FILE0, FILE1])
    solo0 = _volume([FILE0])
    solo1 = _volume([FILE1])

    failures = []

    # ---- 1. the default scope: per-plane -------------------------------------
    st = _run(stacked, scope="plane")
    s0 = _run(solo0, scope="plane")
    s1 = _run(solo1, scope="plane")

    n_st0, n_st1 = _count(st[0]), _count(st[1])
    n_s0, n_s1 = _count(s0[0]), _count(s1[0])
    print(f"[scope=plane] stacked: m0={n_st0} objects, m1={n_st1} objects")
    print(f"[scope=plane] solo   : file0={n_s0} objects, file1={n_s1} objects")

    # Region ids are global-unique across the stack, so compare the MASKS (a region
    # boundary is the segmentation; the id is just a name).
    same0 = np.array_equal(st[0] > 0, s0[0] > 0)
    same1 = np.array_equal(st[1] > 0, s1[0] > 0)
    print(f"[scope=plane] m0 mask == file0-alone mask : {same0}")
    print(f"[scope=plane] m1 mask == file1-alone mask : {same1}")
    if not (same0 and same1):
        failures.append("scope=plane did NOT keep positions independent")
    if (n_s0, n_s1) != (2, 3):
        failures.append(f"fixture wrong: expected 2 and 3 objects, got {n_s0} and {n_s1}")

    # ---- 2. the documented pooling lever -------------------------------------
    dt = _run(stacked, scope="dataset")
    d0 = _run(solo0, scope="dataset")
    d1 = _run(solo1, scope="dataset")
    pooled0 = np.array_equal(dt[0] > 0, d0[0] > 0)
    pooled1 = np.array_equal(dt[1] > 0, d1[0] > 0)
    print(f"[scope=dataset] m0 mask == file0-alone mask : {pooled0}")
    print(f"[scope=dataset] m1 mask == file1-alone mask : {pooled1}")
    print(f"[scope=dataset] stacked: m0={_count(dt[0])}, m1={_count(dt[1])} "
          f"| solo: file0={_count(d0[0])}, file1={_count(d1[0])}")
    bleeds = not (pooled0 and pooled1)
    print(f"[scope=dataset] pools across positions (expected True): {bleeds}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PROBE RESULT: m is an independent batch axis under the default scope.")
    print("  -> stacking files on m gives per-file-independent results for this chain,")
    print("     and scope='dataset' is the named, opt-in way to pool across them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
