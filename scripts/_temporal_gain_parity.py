"""Cell-Tracker **Temporal Fold Correction** parity bench for ``enhance.temporal_gain``.

The node ports CT's plugin as ``reference=rolling_mean`` x ``extent`` in
{``global``, ``tile``, ``gaussian``} (= CT's ``global`` / ``local-tile`` /
``local-gaussian``). ``nodegraph.selftest`` pins the rolling-mean KERNEL against CT's
``pad(edge)+convolve("same")+slice`` sandwich, but nothing pinned the two LOCAL extents
end to end — and a local extent is where a port silently diverges (tile edges, ragged
blocks, which array the blur is applied to).

So this runs CT's three ``_correct_*`` methods, transcribed verbatim from
``Cell-Tracker/CellTracker/plugins/enhancement/builtin.py`` (only the trailing
``clip(0, 65535).astype(volume.dtype)`` dropped, so the comparison is of the ARITHMETIC
and not of a uint16 round-trip), against the node driven through the real ``Engine``.

Run: ``python scripts/_temporal_gain_parity.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.engine import Engine
from nodegraph.graph import Graph, NodeInstance
from nodegraph.metadata import MetaEnvelope
from nodegraph.nodes import COMPUTES
from nodegraph.provider import ArrayProvider
from nodegraph.registry import OutDataset, define_node

# ── Cell-Tracker's TemporalFoldCorrectionPlugin, transcribed ────────────────────
# Verbatim from CellTracker/plugins/enhancement/builtin.py:245-334, except that each
# method returns float64 instead of clipping to 65535 and casting back to volume.dtype.


#: Closest any sample's fold deviation came to the threshold, over every reference run.
#: The dead band makes this node DISCONTINUOUS in its input — a sample sitting exactly on
#: the threshold flips between "gain applied" and "gain exactly 1", so the last bit of the
#: measured brightness decides a whole gain step. CT means each tile with ``.mean()`` while
#: the node block-sums with ``np.add.reduceat``, and those necessarily differ in that last
#: bit, so ON the boundary the two legitimately disagree by the full gain. That is a
#: property of Cell-Tracker's design, not a port defect, and a fixture that lands there
#: would report a bogus DIVERGES — hence this is tracked and printed.
_margin = [np.inf]


def _ratios(rolling, means, threshold):
    r = rolling / np.clip(means, 1e-10, None)
    _margin[0] = min(_margin[0], float(np.abs(np.abs(r - 1.0) - threshold).min()))
    return r


def ct_global(volume, window, threshold):
    T = volume.shape[0]
    means = np.array([float(volume[t].astype(np.float64).mean()) for t in range(T)])
    kernel = np.ones(window) / window
    padded = np.pad(means, window // 2, mode="edge")
    rolling = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]
    ratios = _ratios(rolling, means, threshold)
    result = np.zeros(volume.shape, dtype=np.float64)
    for t in range(T):
        if abs(ratios[t] - 1.0) > threshold:
            result[t] = volume[t].astype(np.float32) * ratios[t]
        else:
            result[t] = volume[t]
    return result


def ct_local_tile(volume, window, threshold, tile_x, tile_y):
    T, H, W = volume.shape
    result = volume.astype(np.float64).copy()
    ny = max(1, (H + tile_y - 1) // tile_y)
    nx = max(1, (W + tile_x - 1) // tile_x)
    for ty_i in range(ny):
        for tx_i in range(nx):
            y0, y1 = ty_i * tile_y, min(ty_i * tile_y + tile_y, H)
            x0, x1 = tx_i * tile_x, min(tx_i * tile_x + tile_x, W)
            tile = volume[:, y0:y1, x0:x1]
            means = np.array([float(tile[t].astype(np.float64).mean()) for t in range(T)])
            kernel = np.ones(window) / window
            padded = np.pad(means, window // 2, mode="edge")
            rolling = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]
            ratios = _ratios(rolling, means, threshold)
            for t in range(T):
                if abs(ratios[t] - 1.0) > threshold:
                    result[t, y0:y1, x0:x1] = tile[t].astype(np.float32) * ratios[t]
    return result


def ct_local_gaussian(volume, window, threshold, sigma, *, blur):
    """CT's per-pixel loop. ``blur`` is injected so the same arithmetic can be scored
    against CT's own ``cv2.GaussianBlur`` and against the node's ``scipy`` blur."""
    T, H, W = volume.shape
    result = volume.astype(np.float64).copy()
    blurred = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        blurred[t] = blur(volume[t].astype(np.float32), sigma)
    kernel = np.ones(window) / window
    for y in range(H):
        for x in range(W):
            ts = blurred[:, y, x]
            padded = np.pad(ts, window // 2, mode="edge")
            rolling = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]
            ratios = _ratios(rolling, ts, threshold)
            for t in range(T):
                if abs(ratios[t] - 1.0) > threshold:
                    result[t, y, x] = np.float32(volume[t, y, x]) * ratios[t]
    return result


def scipy_blur(frame, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(frame, sigma=sigma)


def cv2_blur(frame, sigma):
    import cv2
    return cv2.GaussianBlur(frame, (0, 0), sigma)


# ── the fixture ─────────────────────────────────────────────────────────────────
# A drifting, unevenly lit series with one aberrant frame: the rolling reference has
# something it must actually repair, and the tile/gaussian extents see spatial structure
# that a global gain cannot represent.

T, Y, X = 9, 37, 45          # deliberately NOT multiples of the tile size (ragged blocks)
yy, xx = np.mgrid[0:Y, 0:X]
rng = np.random.default_rng(7)
obj = 400.0 * np.exp(-(((yy - 12) ** 2 + (xx - 16) ** 2) / 32.0)) \
    + 300.0 * np.exp(-(((yy - 26) ** 2 + (xx - 31) ** 2) / 32.0))
shade = 1.0 + 0.8 * (xx / X)
vol = np.empty((T, Y, X))
for t in range(T):
    # a per-frame TILT as well as a decay, so a global gain is provably insufficient
    tilt = 1.0 + 0.25 * np.sin(0.7 * t) * (yy / Y)
    vol[t] = (200.0 + obj) * shade * tilt * (0.88 ** t)
vol[3] *= 1.55                                     # aberrant frame
vol[6] *= 0.65                                     # and one dim frame
vol += rng.normal(0.0, 3.0, vol.shape)
vol = np.clip(vol, 0.0, None)

PIXEL_UM = 1.0                # 1 µm/px ⇒ the node's µm params are px 1:1 with CT's
TILE_PX = 8
SIGMA_PX = 4.0
WINDOW = 5
THRESHOLD = 0.05

ax = AxisSizes(m=1, t=T, z=1, c=1, y=Y, x=X)
meta = {"pixel_size_um": PIXEL_UM, "dt_s": 60.0, "bit_depth": 16}
seed = Dataset(axes=ax, metadata=meta).with_image(
    ArrayProvider(vol.reshape(1, T, 1, 1, Y, X)))
env = MetaEnvelope(axes=ax, metadata=meta)
define_node("io.parity_seed", "S", outputs=[OutDataset()])


def run_node(extent, **params):
    g = Graph()
    g.add(NodeInstance("S", "io.parity_seed"))
    g.add(NodeInstance("G", "enhance.temporal_gain",
                       modes={"reference": "rolling_mean", "extent": extent},
                       params={"window": WINDOW, "threshold": THRESHOLD, **params}))
    g.connect("S", "G")
    e = Engine(g, computes=COMPUTES, seeds={"S": seed}, meta_seeds={"S": env})
    out = e.pull("G")
    return np.stack([out.image.get_region(0, 0, t, 0, 0, 0, Y, 0, X) for t in range(T)])


def score(name, got, ref):
    d = np.abs(got - ref)
    scale = max(float(np.abs(ref).max()), 1e-12)
    rel = float(d.max()) / scale
    n_off = int((d > 1e-6 * scale).sum())
    verdict = "MATCH" if rel < 1e-9 else ("close" if rel < 1e-3 else "DIVERGES")
    print(f"  {name:<34s} max abs diff={d.max():11.4e}  rel={rel:9.3e}  "
          f"pixels off={n_off:6d}/{got.size}  -> {verdict}")
    return rel


print(f"fixture: T={T} Y={Y} X={X}, tile={TILE_PX}px (ragged: {Y%TILE_PX}x{X%TILE_PX} "
      f"remainder), sigma={SIGMA_PX}px, window={WINDOW}, threshold={THRESHOLD}")
print(f"frame means: {np.array([vol[t].mean() for t in range(T)]).round(1)}")
print()

print("global extent  vs  CT _correct_global:")
score("temporal_gain[global]", run_node("global"), ct_global(vol, WINDOW, THRESHOLD))
print()

print("tile extent  vs  CT _correct_local:")
score("temporal_gain[tile]", run_node("tile", tile_size=float(TILE_PX)),
      ct_local_tile(vol, WINDOW, THRESHOLD, TILE_PX, TILE_PX))
print()

print("gaussian extent  vs  CT _correct_local_gaussian:")
got_g = run_node("gaussian", local_sigma=SIGMA_PX)
score("vs CT arithmetic, scipy blur", got_g,
      ct_local_gaussian(vol, WINDOW, THRESHOLD, SIGMA_PX, blur=scipy_blur))
try:
    score("vs CT verbatim, cv2 blur", got_g,
          ct_local_gaussian(vol, WINDOW, THRESHOLD, SIGMA_PX, blur=cv2_blur))
    print("     (^ expected: cv2's BORDER_REFLECT_101 vs scipy's reflect. The frame")
    print("        INTERIOR agrees to ~1e-7; only pixels within ~4 sigma of the edge differ.")
    print("        scipy-at-default is the convention across the whole catalog.)")
except ImportError:
    print("  (cv2 absent - skipped the blur-backend comparison)")

print()
print(f"dead-band margin: closest any sample's fold deviation came to threshold "
      f"{THRESHOLD} was {_margin[0]:.3e}")

# The criterion is NOT "some sample came close to the boundary" — the `gaussian` extent has
# one deviation PER PIXEL (T*Y*X ~ 15k here), so by density something always lands within
# ~1e-4..1e-8 of any threshold, and parity still holds at ~4e-8 because a near-boundary
# sample does not FLIP. A flip needs the deviation to be within the two implementations'
# actual disagreement about the ratio, which is ULP-level (~1e-16 relative) since they
# differ only in summation order. 1e-12 is that, with room to spare. The selftest fixture's
# analytic 0.95 gain sat 4e-17 from a 0.05 threshold — that is what this catches.
if _margin[0] <= 1e-12:
    print("  *** DEGENERATE FIXTURE: a sample sits ON the dead-band boundary, where the")
    print("      strict `>` is decided by the last bit of the measured mean and the two")
    print("      summation orders legitimately disagree by a WHOLE gain step. Any")
    print("      divergence reported above is a coin flip, not a parity defect. Move")
    print("      THRESHOLD off the boundary and re-run.")
