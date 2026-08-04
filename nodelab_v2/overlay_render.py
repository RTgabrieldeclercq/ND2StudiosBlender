"""Renderer maths for the scalar, vector and comparison overlays — Qt-free and testable.

Three overlay families landed together because they share nothing with the existing
Points/Labels/Tracks painters except the surface they draw on, and everything that decides
whether they are *correct* is arithmetic:

* **scalar** — a Voxel-domain layer (strain, EDT, density, a probability, a mask) as a
  colour-mapped wash with a transparency ramp. This is what finally implements
  :class:`~nodelab_v2.overlays.VoxelsOverlay`, which has been declared and never drawn.
* **vector** — a displacement field (DIC/DVC ``u``/``v``) as decimated arrows, optionally
  coloured by magnitude.
* **diff** — two object sets matched against each other and split into TP / FP / FN, the
  "did my new segmentation get better" view.

Keeping the maths here rather than inside the painter is what makes them checkable without
a screen: a colormap that is not monotonic, an alpha ramp that hides the wrong end, a
quiver whose arrows are scaled per-component instead of uniformly, or a matcher that pairs
one object twice are all defects you can assert on, and none of them are visible in a
screenshot until you already believe the picture.

Everything takes and returns plain numpy. The painters in :mod:`nodelab_v2.overlays` turn
the results into QImage/QPainterPath.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "COLORMAPS", "colormap_lut", "normalize_for_map", "scalar_rgba",
    "quiver_arrows", "match_points", "match_labels", "DIFF_COLORS",
]

#: Colormap control points, sampled to 256 entries by :func:`colormap_lut`.
#:
#: Hand-tabulated rather than pulled from matplotlib: this package must not gain a plotting
#: dependency for four gradients, and the sampling is a lerp. `viridis`/`magma` are the
#: perceptually-uniform sequential pair, `turbo` the high-contrast rainbow people expect for
#: displacement magnitude, and `coolwarm` the DIVERGING one — the only correct choice for a
#: signed field like strain, where zero must read as neutral and the two signs must be
#: distinguishable at a glance. A sequential map on signed data hides the sign.
COLORMAPS: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "viridis": ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)),
    "magma": ((0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)),
    "turbo": ((48, 18, 59), (28, 158, 218), (58, 224, 105), (233, 205, 57), (122, 4, 3)),
    "coolwarm": ((59, 76, 192), (144, 178, 246), (220, 220, 220), (245, 156, 125),
                 (180, 4, 38)),
    "grey": ((0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192), (255, 255, 255)),
}

#: Which maps are DIVERGING — they have a meaningful midpoint, so `center_zero` is only
#: honoured for these. Centring a sequential map would put its darkest end at zero and make
#: a symmetric field look like it had a hole in the middle.
_DIVERGING = frozenset({"coolwarm"})

#: TP / FP / FN colours for the comparison overlay. Green/red/blue rather than a red-green
#: pair on purpose: red-green is the most common colour-vision deficiency, and this overlay
#: exists to be *judged* by eye, so the two error classes must differ in hue AND lightness.
DIFF_COLORS: Dict[str, Tuple[int, int, int]] = {
    "matched": (80, 220, 120),      # in both
    "only_a": (235, 90, 90),        # in A only — a miss, if A is ground truth
    "only_b": (90, 150, 250),       # in B only — a false positive, if A is ground truth
}


def colormap_lut(name: str, n: int = 256) -> np.ndarray:
    """``(n, 3)`` uint8 lookup table for ``name`` (unknown → ``viridis``).

    Piecewise-linear through the control points. Monotonic in index by construction, which
    :func:`scalar_rgba` relies on: a value's colour must depend only on where it sits
    between the limits, or two pixels with the same number would read differently."""
    stops = np.array(COLORMAPS.get(name, COLORMAPS["viridis"]), dtype=float)
    src = np.linspace(0.0, 1.0, len(stops))
    dst = np.linspace(0.0, 1.0, int(n))
    out = np.empty((int(n), 3), dtype=float)
    for ch in range(3):
        out[:, ch] = np.interp(dst, src, stops[:, ch])
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_for_map(values: np.ndarray, lo: Optional[float] = None,
                      hi: Optional[float] = None, *, center_zero: bool = False,
                      cmap: str = "viridis") -> Tuple[np.ndarray, float, float]:
    """``values`` → ``(t in [0,1], lo, hi)``, with NaN preserved as NaN.

    ``lo``/``hi`` default to the finite 1st/99th percentiles: a scalar field from a solver
    routinely has a few extreme outliers at its boundary, and scaling to the true min/max
    would compress every real value into the middle of the colormap.

    ``center_zero`` makes the limits symmetric about zero so the midpoint colour lands
    exactly on zero — the thing a strain field needs, and honoured only for a diverging map
    (see :data:`_DIVERGING`), because centring a sequential one is meaningless.
    """
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    if lo is None or hi is None:
        if finite.size:
            auto_lo, auto_hi = np.percentile(finite, [1.0, 99.0])
        else:
            auto_lo, auto_hi = 0.0, 1.0
        lo = float(auto_lo) if lo is None else float(lo)
        hi = float(auto_hi) if hi is None else float(hi)
    lo, hi = float(lo), float(hi)
    if center_zero and cmap in _DIVERGING:
        r = max(abs(lo), abs(hi))
        lo, hi = -r, r
    if hi <= lo:
        hi = lo + 1e-9
    with np.errstate(invalid="ignore"):
        t = (a - lo) / (hi - lo)
    return np.clip(t, 0.0, 1.0), lo, hi


def scalar_rgba(values: np.ndarray, *, cmap: str = "viridis",
                lo: Optional[float] = None, hi: Optional[float] = None,
                center_zero: bool = False, alpha_mode: str = "ramp",
                threshold: float = 0.0, opacity: float = 0.55) -> np.ndarray:
    """A scalar field as ``(H, W, 4)`` uint8 RGBA, ready to draw over the image.

    ``alpha_mode`` is the setting that decides whether this overlay is usable:

    * ``flat`` — every finite sample at full ``opacity``. Correct for a MASK, where the
      value carries no magnitude and a ramp would just make the mask look faded.
    * ``ramp`` — alpha rises with the value, so the low end stays transparent and the image
      shows through where there is nothing to report. This is the default because a flat
      wash over a whole field hides exactly the pixels you are trying to judge the field
      against. For a diverging map the ramp is on |t − ½|, so **zero** is the transparent
      end rather than the most negative value — otherwise a symmetric strain field would
      paint compression solid and tension invisible.
    * ``gated`` — hard cut: samples below ``threshold`` (in DATA units) draw nothing at all.

    NaN is always fully transparent, never colour-mapped: a solver's undefined region is
    absence, and painting it as the colormap's low end would report a measurement there.
    """
    t, lo_r, hi_r = normalize_for_map(values, lo, hi, center_zero=center_zero, cmap=cmap)
    lut = colormap_lut(cmap)
    idx = np.clip(np.nan_to_num(t, nan=0.0) * 255.0, 0, 255).astype(np.uint8)
    rgb = lut[idx]

    base = float(min(max(opacity, 0.0), 1.0))
    if alpha_mode == "flat":
        a = np.full(t.shape, base, dtype=float)
    elif alpha_mode == "gated":
        a = np.where(np.asarray(values, dtype=float) >= float(threshold), base, 0.0)
    else:                                          # ramp
        if center_zero and cmap in _DIVERGING:
            a = base * np.clip(np.abs(t - 0.5) * 2.0, 0.0, 1.0)
        else:
            a = base * np.clip(t, 0.0, 1.0)
    a = np.where(np.isfinite(np.asarray(values, dtype=float)), a, 0.0)
    out = np.empty(t.shape + (4,), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return out


def quiver_arrows(y: np.ndarray, x: np.ndarray, u: np.ndarray, v: np.ndarray,
                  *, every: int = 1, scale: float = 1.0,
                  gate: float = 0.0, max_arrows: int = 4000
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate and scale a vector field → ``(tails (N,2) yx, heads (N,2) yx, magnitude)``.

    ``u``/``v`` are the displacement's y/x components in the same units as ``y``/``x``.
    ``scale`` multiplies BOTH components — never one — so an arrow's direction is the
    field's direction. Scaling per-axis is the classic quiver bug: it looks like a
    plausible field and every angle in it is wrong.

    ``every`` keeps 1 in N samples; ``gate`` drops vectors shorter than that magnitude
    (they are usually solver noise and they dominate the arrow count). ``max_arrows`` is a
    final uniform thinning so a dense field cannot make a pan unusable — applied AFTER the
    gate, so thinning never removes the large vectors the gate was keeping.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    n = min(y.size, x.size, u.size, v.size)
    y, x, u, v = y[:n], x[:n], u[:n], v[:n]
    keep = np.isfinite(y) & np.isfinite(x) & np.isfinite(u) & np.isfinite(v)
    if int(every) > 1:
        step = np.zeros(n, dtype=bool)
        step[::int(every)] = True
        keep &= step
    mag = np.hypot(u, v)
    if gate > 0:
        keep &= mag >= float(gate)
    idx = np.flatnonzero(keep)
    if idx.size > int(max_arrows) > 0:
        idx = idx[np.linspace(0, idx.size - 1, int(max_arrows)).astype(np.intp)]
    tails = np.stack([y[idx], x[idx]], axis=1)
    heads = np.stack([y[idx] + u[idx] * float(scale),
                      x[idx] + v[idx] * float(scale)], axis=1)
    return tails, heads, mag[idx]


def match_points(a_yx: np.ndarray, b_yx: np.ndarray, *, max_dist: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour pairing → ``(pairs (K,2), only_a, only_b)`` of indices.

    Greedy on ascending distance, and **one-to-one**: once an object on either side is
    paired it is out. That is the property that makes the TP/FP/FN counts mean something —
    a matcher that let two predictions claim the same ground-truth object would report
    twice the recall it earned.

    ``max_dist`` is a hard cut in the coordinates' own units; nothing further apart is a
    match however lonely the two objects are.
    """
    a = np.atleast_2d(np.asarray(a_yx, dtype=float))
    b = np.atleast_2d(np.asarray(b_yx, dtype=float))
    if a.size == 0 or b.size == 0:
        return (np.zeros((0, 2), dtype=np.intp),
                np.arange(len(a) if a.size else 0, dtype=np.intp),
                np.arange(len(b) if b.size else 0, dtype=np.intp))
    d = np.hypot(a[:, 0:1] - b[None, :, 0], a[:, 1:2] - b[None, :, 1])
    pairs = []
    used_a: set = set()
    used_b: set = set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] > float(max_dist):
            break                                  # sorted, so nothing later is closer
        if int(i) in used_a or int(j) in used_b:
            continue
        used_a.add(int(i)); used_b.add(int(j))
        pairs.append((int(i), int(j)))
    only_a = np.array([i for i in range(len(a)) if i not in used_a], dtype=np.intp)
    only_b = np.array([j for j in range(len(b)) if j not in used_b], dtype=np.intp)
    return np.array(pairs, dtype=np.intp).reshape(-1, 2), only_a, only_b


def match_labels(raster_a: np.ndarray, raster_b: np.ndarray, *, min_iou: float = 0.5
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two label rasters by IoU → ``(pairs (K,2) of label IDS, only_a, only_b)``.

    IoU rather than centroid distance because a segmentation comparison is about the
    *extent* an object claims, not where its middle is: two masks can share a centroid and
    disagree about half their area, and a distance matcher scores that as perfect.

    Computed from one joint histogram of the two rasters — the intersections of every pair
    at once — so the cost does not grow with the number of objects the way a per-pair mask
    comparison does. Background (0) is excluded from both sides.

    Greedy on descending IoU, one-to-one, exactly as :func:`match_points` is.
    """
    a = np.asarray(raster_a)
    b = np.asarray(raster_b)
    if a.shape != b.shape:
        raise ValueError(
            f"label rasters differ in shape ({a.shape} vs {b.shape}); an IoU comparison "
            f"reads them voxel-for-voxel, so they must be on the same grid")
    ids_a = np.unique(a[a > 0])
    ids_b = np.unique(b[b > 0])
    if ids_a.size == 0 or ids_b.size == 0:
        return (np.zeros((0, 2), dtype=np.intp), ids_a.astype(np.intp),
                ids_b.astype(np.intp))
    ia = np.searchsorted(ids_a, a.ravel())
    ib = np.searchsorted(ids_b, b.ravel())
    valid = (a.ravel() > 0) & (b.ravel() > 0)
    inter = np.zeros((ids_a.size, ids_b.size), dtype=np.int64)
    np.add.at(inter, (ia[valid], ib[valid]), 1)
    area_a = np.array([(a == i).sum() for i in ids_a], dtype=np.int64)
    area_b = np.array([(b == j).sum() for j in ids_b], dtype=np.int64)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    pairs = []
    used_a: set = set()
    used_b: set = set()
    order = np.dstack(np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape))[0]
    for i, j in order:
        if iou[i, j] < float(min_iou):
            break
        if int(i) in used_a or int(j) in used_b:
            continue
        used_a.add(int(i)); used_b.add(int(j))
        pairs.append((int(ids_a[i]), int(ids_b[j])))
    only_a = np.array([int(v) for k, v in enumerate(ids_a) if k not in used_a],
                      dtype=np.intp)
    only_b = np.array([int(v) for k, v in enumerate(ids_b) if k not in used_b],
                      dtype=np.intp)
    return np.array(pairs, dtype=np.intp).reshape(-1, 2), only_a, only_b
