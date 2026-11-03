"""Arrays derived from a pulled Dataset for representations no node makes (V5.00 step 1).

Each function takes plain numpy arrays and the few metadata numbers it needs and returns
plain arrays — orthogonal reslices, depth- and time-coded projections, kymographs,
intensity profiles, mean squared displacement, Kaplan–Meier survival, colocalisation
statistics. They are display derivations: nothing here reaches a downstream node, which is
why they live beside the figure document and not in the catalog.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.catalog._shared.overlay_math import colormap_lut


# ── reslice and coded projections ────────────────────────────────────────────────────
def reslice(stack: np.ndarray, axis: str, index: int, *, z_step_um: Optional[float] = None,
            pixel_size_um: Optional[float] = None) -> np.ndarray:
    """An orthogonal view of a ``(Z, Y, X)`` stack: ``xz`` at row ``index`` → ``(Z', X)``,
    ``yz`` at column ``index`` → ``(Z', Y)``. With both spacings the Z axis is resampled
    (linear) so a pixel is square — ``Z' = round(Z · z_step / pixel_size)`` — because an
    XZ view drawn at the raw slice count squashes the specimen by the sampling ratio."""
    a = np.asarray(stack, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"reslice: expected a (Z, Y, X) stack, got shape {a.shape}")
    nz, ny, nx = a.shape
    if axis == "xz":
        if not 0 <= int(index) < ny:
            raise ValueError(f"reslice: row {index} outside 0..{ny - 1}")
        sl = a[:, int(index), :]
    elif axis == "yz":
        if not 0 <= int(index) < nx:
            raise ValueError(f"reslice: column {index} outside 0..{nx - 1}")
        sl = a[:, :, int(index)]
    else:
        raise ValueError(f"reslice: axis must be 'xz' or 'yz', not {axis!r}")
    if z_step_um and pixel_size_um and nz > 1 and z_step_um > 0 and pixel_size_um > 0:
        ratio = float(z_step_um) / float(pixel_size_um)
        n_out = max(2, int(round((nz - 1) * ratio)) + 1)
        src = np.arange(nz, dtype=float)
        dst = np.linspace(0.0, nz - 1.0, n_out)
        out = np.empty((n_out, sl.shape[1]), dtype=float)
        for j in range(sl.shape[1]):
            out[:, j] = np.interp(dst, src, sl[:, j])
        return out
    return sl.copy()


def coded_projection(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(index of the maximum along axis 0, that maximum)`` for an ``(N, Y, X)`` stack —
    the depth code of a Z stack, the time code of a T series. NaN is ignored; an
    all-NaN pixel codes 0 with a NaN maximum."""
    a = np.asarray(stack, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"coded_projection: expected (N, Y, X), got shape {a.shape}")
    filled = np.where(np.isfinite(a), a, -np.inf)
    idx = np.argmax(filled, axis=0)
    mx = np.take_along_axis(filled, idx[None], axis=0)[0]
    mx = np.where(np.isfinite(mx), mx, np.nan)
    return idx.astype(np.int32), mx


def coded_rgb(code: np.ndarray, intensity: np.ndarray, *, n: Optional[int] = None,
              cmap: str = "turbo", lo: Optional[float] = None, hi: Optional[float] = None,
              gamma: float = 1.0) -> np.ndarray:
    """A coded projection as ``(Y, X, 3)`` uint8: hue from ``code / (n - 1)`` on ``cmap``,
    brightness from ``intensity`` windowed to ``[lo, hi]`` (the finite 0.5–99.8 percentiles
    by default), so depth or time reads as colour and signal as brightness — the
    Fiji "Temporal-Color Code" / "Z-code" look."""
    c = np.asarray(code)
    v = np.asarray(intensity, dtype=float)
    if n is None:
        n = int(c.max()) + 1 if c.size else 1
    t = np.clip(c.astype(float) / max(1, int(n) - 1), 0.0, 1.0)
    lut = colormap_lut(cmap)
    rgb = lut[np.clip((t * 255).astype(int), 0, 255)].astype(float)
    finite = v[np.isfinite(v)]
    if lo is None or hi is None:
        if finite.size:
            p_lo, p_hi = np.percentile(finite, [0.5, 99.8])
        else:
            p_lo, p_hi = 0.0, 1.0
        lo = float(p_lo) if lo is None else float(lo)
        hi = float(p_hi) if hi is None else float(hi)
    if hi <= lo:
        hi = lo + 1e-9
    with np.errstate(invalid="ignore"):
        b = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    b = np.where(np.isfinite(b), b, 0.0) ** float(gamma if gamma > 0 else 1.0)
    out = rgb * b[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── lines through an image ───────────────────────────────────────────────────────────
def _line_samples(p0: Sequence[float], p1: Sequence[float], *, samples: Optional[int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y0, x0 = float(p0[0]), float(p0[1])
    y1, x1 = float(p1[0]), float(p1[1])
    length = math.hypot(y1 - y0, x1 - x0)
    n = int(samples) if samples else max(2, int(round(length)) + 1)
    s = np.linspace(0.0, length, n)
    if length > 0:
        ys = y0 + (y1 - y0) * s / length
        xs = x0 + (x1 - x0) * s / length
    else:
        ys = np.full(n, y0)
        xs = np.full(n, x0)
    return s, ys, xs, length


def _sample_width(plane: np.ndarray, ys: np.ndarray, xs: np.ndarray, p0, p1,
                  width: int) -> np.ndarray:
    """Bilinear samples of ``plane`` along the line, averaged over ``width`` parallel
    offsets (perpendicular to the line, centred on it)."""
    from scipy.ndimage import map_coordinates
    y0, x0 = float(p0[0]), float(p0[1])
    y1, x1 = float(p1[0]), float(p1[1])
    length = math.hypot(y1 - y0, x1 - x0)
    if length > 0:
        ny, nx = -(x1 - x0) / length, (y1 - y0) / length       # unit normal (y, x)
    else:
        ny, nx = 1.0, 0.0
    w = max(1, int(width))
    offs = np.arange(w, dtype=float) - (w - 1) / 2.0
    acc = np.zeros(ys.shape, dtype=float)
    for o in offs:
        acc += map_coordinates(np.asarray(plane, dtype=float),
                               [ys + o * ny, xs + o * nx], order=1, mode="nearest")
    return acc / w


def profile(plane: np.ndarray, p0: Sequence[float], p1: Sequence[float], *,
            width: int = 1, samples: Optional[int] = None, pixel_size_um: Optional[float] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    """``(distance, values)`` along the line ``p0 → p1`` (``(y, x)`` pixels) through
    ``plane`` — ``(Y, X)`` → one row of values, ``(C, Y, X)`` → one row per channel.
    ``distance`` is in pixels, or µm with ``pixel_size_um``; ``width`` averages that many
    parallel samples across the line, as Fiji's Plot Profile does with a wide line."""
    a = np.asarray(plane, dtype=float)
    s, ys, xs, _ = _line_samples(p0, p1, samples=samples)
    if a.ndim == 2:
        vals = _sample_width(a, ys, xs, p0, p1, width)[None]
    elif a.ndim == 3:
        vals = np.stack([_sample_width(a[c], ys, xs, p0, p1, width) for c in range(a.shape[0])])
    else:
        raise ValueError(f"profile: expected (Y, X) or (C, Y, X), got shape {a.shape}")
    if pixel_size_um:
        s = s * float(pixel_size_um)
    return s, (vals[0] if a.ndim == 2 else vals)


def kymograph(stack: np.ndarray, p0: Sequence[float], p1: Sequence[float], *,
              width: int = 1, samples: Optional[int] = None) -> np.ndarray:
    """``(T, L)``: the intensity along the line ``p0 → p1`` of a ``(T, Y, X)`` series, one
    row per frame — time runs down, position along the line runs across."""
    a = np.asarray(stack, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"kymograph: expected (T, Y, X), got shape {a.shape}")
    s, ys, xs, _ = _line_samples(p0, p1, samples=samples)
    out = np.empty((a.shape[0], ys.size), dtype=float)
    for t in range(a.shape[0]):
        out[t] = _sample_width(a[t], ys, xs, p0, p1, width)
    return out


# ── tracks ───────────────────────────────────────────────────────────────────────────
def msd(track_id: Sequence[Any], t: Sequence[float], y: Sequence[float], x: Sequence[float],
        *, dt: float = 1.0, max_lag: Optional[int] = None, z: Optional[Sequence[float]] = None
        ) -> Dict[str, Any]:
    """Mean squared displacement from a track table's rows.

    Every pair of sightings of one track ``k`` frames apart contributes ``|Δr|²`` to lag
    ``k`` (time-averaged within the track, then averaged across tracks weighted by their
    pair counts — the ensemble-time average). Frames are matched by their ``t`` value, so
    a gap in a track is skipped, not bridged. Returns ``{"lag": [k·dt], "msd": [...],
    "n": [pairs], "per_track": {id: {"lag": …, "msd": …}}}``; ``max_lag`` defaults to
    half the longest track, past which the estimate is from too few pairs."""
    ids = np.asarray(track_id)
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    zz = None if z is None else np.asarray(z, dtype=float)
    keep = np.isfinite(tt) & np.isfinite(yy) & np.isfinite(xx)
    if zz is not None:
        keep &= np.isfinite(zz)
    per: Dict[Any, Dict[int, List[float]]] = {}
    longest = 0
    for uid in np.unique(ids[keep]):
        m = keep & (ids == uid)
        order = np.argsort(tt[m], kind="stable")
        ts = np.rint(tt[m][order]).astype(np.int64)
        ps = np.stack([yy[m][order], xx[m][order]] +
                      ([zz[m][order]] if zz is not None else []), axis=1)
        pos = {int(a): ps[i] for i, a in enumerate(ts)}
        if len(pos) < 2:
            continue
        frames = sorted(pos)
        span = frames[-1] - frames[0]
        longest = max(longest, span)
        acc: Dict[int, List[float]] = {}
        for i, f0 in enumerate(frames):
            for f1 in frames[i + 1:]:
                k = f1 - f0
                if max_lag is not None and k > int(max_lag):
                    break
                d = pos[f1] - pos[f0]
                acc.setdefault(k, []).append(float(np.dot(d, d)))
        per[uid] = acc
    cap = int(max_lag) if max_lag is not None else max(1, longest // 2)
    lags = list(range(1, cap + 1))
    ens_msd: List[float] = []
    ens_n: List[int] = []
    for k in lags:
        vals: List[float] = []
        for acc in per.values():
            vals.extend(acc.get(k, []))
        ens_n.append(len(vals))
        ens_msd.append(float(np.mean(vals)) if vals else float("nan"))
    per_track = {}
    for uid, acc in per.items():
        ks = sorted(k for k in acc if k <= cap)
        per_track[uid.item() if hasattr(uid, "item") else uid] = {
            "lag": [k * float(dt) for k in ks],
            "msd": [float(np.mean(acc[k])) for k in ks]}
    return {"lag": [k * float(dt) for k in lags], "msd": ens_msd, "n": ens_n,
            "per_track": per_track}


def power_fit(x: Sequence[float], y: Sequence[float]) -> Optional[Dict[str, float]]:
    """``y = A·x^alpha`` by least squares on log–log → ``{"A", "alpha", "r2"}``; ``None``
    with fewer than two positive points. For an MSD, ``alpha`` ≈ 1 is diffusive, ≈ 2
    ballistic, < 1 confined."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if keep.sum() < 2:
        return None
    lx, ly = np.log10(xs[keep]), np.log10(ys[keep])
    if np.ptp(lx) == 0:
        return None
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"A": float(10 ** a), "alpha": float(b), "r2": float(r2)}


def survival(durations: Sequence[float], *, observed: Optional[Sequence[bool]] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier ``(times, S)``: the fraction of tracks still alive after each duration,
    starting at ``(0, 1)``. ``observed`` marks durations that ended (``True``, the default
    for every row) against ones cut off by the end of the series (censored: they leave the
    risk set without an event)."""
    d = np.asarray(durations, dtype=float)
    obs = np.ones(d.shape, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    keep = np.isfinite(d) & (d >= 0)
    d, obs = d[keep], obs[keep]
    order = np.argsort(d, kind="stable")
    d, obs = d[order], obs[order]
    times = [0.0]
    surv = [1.0]
    s = 1.0
    at_risk = d.size
    i = 0
    while i < d.size:
        tcur = d[i]
        j = i
        events = 0
        while j < d.size and d[j] == tcur:
            events += int(obs[j])
            j += 1
        if events:
            s *= 1.0 - events / at_risk
            times.append(float(tcur))
            surv.append(float(s))
        at_risk -= (j - i)
        i = j
    return np.asarray(times), np.asarray(surv)


# ── colocalisation ───────────────────────────────────────────────────────────────────
def coloc(a: np.ndarray, b: np.ndarray, *, mask: Optional[np.ndarray] = None,
          thresh_a: Optional[float] = None, thresh_b: Optional[float] = None
          ) -> Dict[str, Any]:
    """Pearson's r over the (masked) pixels, Manders' M1 / M2 with thresholds (Otsu of each
    channel inside the mask by default), and the pixel count. Two channels of one plane."""
    from skimage.filters import threshold_otsu
    A = np.asarray(a, dtype=float).ravel()
    B = np.asarray(b, dtype=float).ravel()
    if A.shape != B.shape:
        raise ValueError(f"coloc: shapes differ ({A.shape} vs {B.shape})")
    keep = np.isfinite(A) & np.isfinite(B)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    A, B = A[keep], B[keep]
    n = int(A.size)
    if n < 2:
        return {"pearson": float("nan"), "m1": float("nan"), "m2": float("nan"), "n": n,
                "thresh_a": thresh_a, "thresh_b": thresh_b}
    sa, sb = A.std(), B.std()
    r = float(((A - A.mean()) * (B - B.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 \
        else float("nan")
    if thresh_a is None:
        thresh_a = float(threshold_otsu(A)) if np.ptp(A) > 0 else float(A.min())
    if thresh_b is None:
        thresh_b = float(threshold_otsu(B)) if np.ptp(B) > 0 else float(B.min())
    sum_a, sum_b = float(A.sum()), float(B.sum())
    m1 = float(A[B > thresh_b].sum() / sum_a) if sum_a > 0 else float("nan")
    m2 = float(B[A > thresh_a].sum() / sum_b) if sum_b > 0 else float("nan")
    return {"pearson": r, "m1": m1, "m2": m2, "n": n, "thresh_a": float(thresh_a),
            "thresh_b": float(thresh_b)}
