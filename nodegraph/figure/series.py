"""A chart's data — the series the ``plot.*`` nodes draw from, and the kinds they lack
(V5.00 step 1).

The existing helpers (grouping, mean ± spread, pooled SD, heatmap cells, fits) are
re-exported from :mod:`nodegraph.catalog._shared.figure` unchanged, so a chart on the page
and a Plot XY node summarise a table identically. New here: bars, strips, SuperPlots,
roses, survival, MSD and hexagonal binning. Every function takes columns (anything
array-like) and returns JSON-able lists, because a resolved series rides in the render
cache and in the export bundle.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.catalog._shared import figure as _F
from nodegraph.figure import derive as _D

xy_series = _F.xy_series
distribution_series = _F.distribution_series
heatmap_grid = _F.heatmap_grid
group_parts = _F.group_parts
pooled_stats = _F.pooled_stats
parse_range = _F.parse_range
table_columns = _F.table_columns
column = _F.column
axis_range = _F.axis_range
padded_range = _F.padded_range
color_limits = _F.color_limits
frame_times = _F.frame_times
xy_fit = _F.xy_fit
xy_langmuir = _F.xy_langmuir
xy_exponential = _F.xy_exponential
HEAT_REDUCERS = _F.HEAT_REDUCERS
MISSING_GROUP = _F.MISSING_GROUP

__all__ = ["xy_series", "distribution_series", "heatmap_grid", "group_parts", "pooled_stats",
           "parse_range", "table_columns", "column", "axis_range", "padded_range",
           "color_limits", "frame_times", "xy_fit", "xy_langmuir", "xy_exponential",
           "HEAT_REDUCERS", "MISSING_GROUP", "spread", "bar_series", "strip_series",
           "superplot_series", "rose_series", "survival_series", "msd_series",
           "hexbin_series", "normalize_series"]


def spread(values: Sequence[float], error: str = "sem") -> Tuple[float, float, int]:
    """``(mean, half-width, n)`` of the finite ``values`` under ``error`` — ``sd``, ``sem``,
    ``ci95`` (Student's t) or ``none`` (0). Fewer than two values give 0 spread."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), 0.0, 0
    mean = float(v.mean())
    if n < 2 or error == "none":
        return mean, 0.0, n
    sd = float(v.std(ddof=1))
    if error == "sd":
        return mean, sd, n
    sem = sd / math.sqrt(n)
    if error == "sem":
        return mean, sem, n
    if error == "ci95":
        from scipy import stats as S
        return mean, float(S.t.ppf(0.975, n - 1)) * sem, n
    raise ValueError(f"spread: unknown error {error!r}")


def bar_series(categories: Sequence[Any], values: Sequence[float],
               group: Optional[Sequence[Any]] = None, *, error: str = "sem"
               ) -> List[Dict[str, Any]]:
    """Bars: one series per ``group`` value (one when ``None``), each with every category
    in the data's stable order (numbers ascending, then text), ``y`` the mean of the rows in
    that (group, category), ``err`` its spread, ``n`` the rows; a category a group lacks is
    NaN with ``n = 0``. The category order is shared by every series, so grouped and
    stacked bars line up."""
    cats = np.asarray(categories, dtype=object)
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    order = [label for label, _idx in group_parts(cats, keep)]
    cat_of = {label: idx for label, idx in group_parts(cats, keep)}
    parts = [("", np.flatnonzero(keep))] if group is None else group_parts(group, keep)
    out: List[Dict[str, Any]] = []
    for label, idx in parts:
        rows = set(idx.tolist())
        ys: List[float] = []
        errs: List[float] = []
        ns: List[int] = []
        for cat in order:
            sel = [i for i in cat_of[cat].tolist() if i in rows]
            m, e, n = spread(v[sel], error) if sel else (float("nan"), 0.0, 0)
            ys.append(m)
            errs.append(e)
            ns.append(n)
        out.append({"label": label, "categories": list(order), "y": ys, "err": errs,
                    "n": ns})
    return out


def strip_series(values: Sequence[float], group: Optional[Sequence[Any]] = None, *,
                 jitter: float = 0.25, seed: int = 0) -> List[Dict[str, Any]]:
    """The distribution series with a deterministic horizontal ``offsets`` list per point
    (uniform in ``[-jitter, jitter]`` category units), so a strip plot redraws identically."""
    rng = np.random.default_rng(int(seed))
    out = []
    for s in distribution_series(values, group):
        n = len(s["values"])
        s = dict(s)
        s["offsets"] = (rng.uniform(-float(jitter), float(jitter), n).tolist() if n else [])
        out.append(s)
    return out


def superplot_series(values: Sequence[float], group: Sequence[Any],
                     replicate: Sequence[Any]) -> List[Dict[str, Any]]:
    """A SuperPlot (Lord et al. 2020): per ``group``, every value with its ``replicate``
    label, and one mean per replicate — the points are coloured by replicate and the
    replicate means are what the statistics compare."""
    v = np.asarray(values, dtype=float)
    rep = np.asarray(replicate, dtype=object)
    keep = np.isfinite(v)
    out = []
    for label, idx in group_parts(group, keep):
        reps = [(r, j) for r, j in group_parts(rep[idx], np.ones(idx.size, dtype=bool))]
        rep_label = np.empty(idx.size, dtype=object)
        means = []
        for r, j in reps:
            rep_label[j] = r
            means.append({"replicate": r, "mean": float(v[idx][j].mean()), "n": int(j.size)})
        out.append({"label": label, "values": v[idx].tolist(),
                    "replicate": [str(x) for x in rep_label.tolist()],
                    "means": means, "n": int(idx.size)})
    return out


def rose_series(angles_deg: Sequence[float], group: Optional[Sequence[Any]] = None, *,
                bins: int = 12, weights: Optional[Sequence[float]] = None
                ) -> List[Dict[str, Any]]:
    """Orientation histograms on the circle: ``bins`` equal sectors over 0–360°, angles
    taken modulo 360, counts (or summed ``weights``) per sector, one series per group."""
    a = np.asarray(angles_deg, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    keep = np.isfinite(a) & (np.isfinite(w) if w is not None else True)
    edges = np.linspace(0.0, 360.0, int(bins) + 1)
    parts = [("", np.flatnonzero(keep))] if group is None else group_parts(group, keep)
    out = []
    for label, idx in parts:
        ang = np.mod(a[idx], 360.0)
        counts, _ = np.histogram(ang, bins=edges, weights=None if w is None else w[idx])
        out.append({"label": label, "edges_deg": edges.tolist(),
                    "counts": [float(c) for c in counts], "n": int(idx.size)})
    return out


def survival_series(lengths: Sequence[float], group: Optional[Sequence[Any]] = None, *,
                    dt: float = 1.0, observed: Optional[Sequence[bool]] = None
                    ) -> List[Dict[str, Any]]:
    """Kaplan–Meier curves of track lifetimes (``lengths`` in frames × ``dt``), one per
    group: ``t`` the step times, ``s`` the surviving fraction."""
    L = np.asarray(lengths, dtype=float)
    obs = None if observed is None else np.asarray(observed, dtype=bool)
    keep = np.isfinite(L)
    parts = [("", np.flatnonzero(keep))] if group is None else group_parts(group, keep)
    out = []
    for label, idx in parts:
        t, s = _D.survival(L[idx] * float(dt), observed=None if obs is None else obs[idx])
        out.append({"label": label, "t": t.tolist(), "s": s.tolist(), "n": int(idx.size),
                    "median": _median_survival(t, s)})
    return out


def _median_survival(t: np.ndarray, s: np.ndarray) -> Optional[float]:
    below = np.flatnonzero(s <= 0.5)
    return float(t[below[0]]) if below.size else None


def msd_series(track_id: Sequence[Any], t: Sequence[float], y: Sequence[float],
               x: Sequence[float], group: Optional[Sequence[Any]] = None, *,
               dt: float = 1.0, max_lag: Optional[int] = None,
               z: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    """Ensemble MSD curves, one per group, each with its power-law fit
    (``A·lag^alpha``) over the lags with at least ten pairs."""
    ids = np.asarray(track_id)
    keep = np.ones(ids.shape, dtype=bool)
    parts = [("", np.flatnonzero(keep))] if group is None else group_parts(group, keep)
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    zz = None if z is None else np.asarray(z, dtype=float)
    out = []
    for label, idx in parts:
        r = _D.msd(ids[idx], tt[idx], yy[idx], xx[idx], dt=dt, max_lag=max_lag,
                   z=None if zz is None else zz[idx])
        lag = np.asarray(r["lag"], dtype=float)
        val = np.asarray(r["msd"], dtype=float)
        n = np.asarray(r["n"])
        good = n >= 10
        fit = _D.power_fit(lag[good], val[good]) if good.sum() >= 2 else None
        out.append({"label": label, "lag": r["lag"], "msd": r["msd"], "n": r["n"],
                    "fit": fit, "tracks": len(r["per_track"])})
    return out


def hexbin_series(x: Sequence[float], y: Sequence[float], *, gridsize: int = 30,
                  extent: Optional[Sequence[float]] = None, log_x: bool = False,
                  log_y: bool = False) -> Dict[str, Any]:
    """Hexagonal binning of the finite ``(x, y)`` pairs (matplotlib's two-lattice scheme):
    ``centers`` ``[[x, y], …]`` in data units, ``counts`` per hexagon, ``hex`` the six vertex
    offsets of one hexagon in data units, ``extent`` ``[xmin, xmax, ymin, ymax]``. On a log
    axis the binning is in log10 and the centres come back in data units."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    if log_x:
        keep &= xs > 0
    if log_y:
        keep &= ys > 0
    xs, ys = xs[keep], ys[keep]
    px = np.log10(xs) if log_x else xs
    py = np.log10(ys) if log_y else ys
    if extent is None:
        if px.size:
            xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        else:
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in extent)
        if log_x:
            xmin, xmax = math.log10(max(xmin, 1e-300)), math.log10(max(xmax, 1e-300))
        if log_y:
            ymin, ymax = math.log10(max(ymin, 1e-300)), math.log10(max(ymax, 1e-300))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    nx = max(1, int(gridsize))
    ny = max(1, int(round(nx / math.sqrt(3))))
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny
    ix = (px - xmin) / sx
    iy = (py - ymin) / sy
    ix1, iy1 = np.round(ix), np.round(iy)
    ix2, iy2 = np.round(ix - 0.5), np.round(iy - 0.5)
    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    first = d1 < d2
    cells: Dict[Tuple[int, int, int], int] = {}
    for lat, a, b, m in ((0, ix1, iy1, first), (1, ix2, iy2, ~first)):
        for i, j in zip(a[m].astype(int).tolist(), b[m].astype(int).tolist()):
            cells[(lat, i, j)] = cells.get((lat, i, j), 0) + 1
    centers = []
    counts = []
    for (lat, i, j), c in sorted(cells.items()):
        cx = xmin + (i + (0.5 if lat else 0.0)) * sx
        cy = ymin + (j + (0.5 if lat else 0.0)) * sy
        centers.append([float(10 ** cx) if log_x else float(cx),
                        float(10 ** cy) if log_y else float(cy)])
        counts.append(int(c))
    hexagon = [[0.0, -0.5 * sy], [0.5 * sx, -0.25 * sy], [0.5 * sx, 0.25 * sy],
               [0.0, 0.5 * sy], [-0.5 * sx, 0.25 * sy], [-0.5 * sx, -0.25 * sy]]
    return {"centers": centers, "counts": counts, "hex": hexagon,
            "extent": [float(xmin), float(xmax), float(ymin), float(ymax)],
            "log_x": bool(log_x), "log_y": bool(log_y), "n": int(xs.size)}


def normalize_series(series: Sequence[Dict[str, Any]], mode: str = "first"
                     ) -> List[Dict[str, Any]]:
    """XY series with ``y`` normalised per series: ``first`` divides by the first point
    (a FRAP / bleaching curve starting at 1), ``max`` by the maximum, ``range`` to 0–1
    between its min and max. Errors scale with the same factor; ``none`` returns copies."""
    out = []
    for s in series:
        y = np.asarray(s.get("y", []), dtype=float)
        e = None if s.get("err") is None else np.asarray(s["err"], dtype=float)
        if mode == "none" or y.size == 0:
            out.append(dict(s))
            continue
        if mode == "first":
            ref = y[np.isfinite(y)][0] if np.isfinite(y).any() else 1.0
            den, off = (ref if ref else 1.0), 0.0
        elif mode == "max":
            mx = np.nanmax(y)
            den, off = (mx if mx else 1.0), 0.0
        elif mode == "range":
            lo, hi = np.nanmin(y), np.nanmax(y)
            den, off = ((hi - lo) if hi > lo else 1.0), lo
        else:
            raise ValueError(f"normalize_series: unknown mode {mode!r}")
        t = dict(s)
        t["y"] = ((y - off) / den).tolist()
        if e is not None:
            t["err"] = (e / den).tolist()
        out.append(t)
    return out
