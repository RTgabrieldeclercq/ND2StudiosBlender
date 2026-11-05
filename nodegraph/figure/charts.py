"""A chart's GEOMETRY in millimetres (V5.00 step 1).

:func:`chart_geometry` takes a chart block of a figure document, its resolved series and
the item's rectangle, and returns every line, mark and label of the chart as page
coordinates — the axes rectangle, tick positions with their text, series polylines, marker
positions, error bars and bands, bars, boxes, violins, heatmap cells, hexagons, legend
entries, colour bar, statistics brackets. Nothing is painted: the GUI's chart item and the
offscreen renderer both paint *this*, so the preview and the export are one set of lines.

Tick placement uses ``matplotlib.ticker`` (pure Python, no backend); text sizes come from a
``measure(text, pt) -> (w_mm, h_mm)`` callback the caller supplies (``QFontMetricsF`` in the
GUI), with a width estimate as the default so the geometry is computable headless.

Page coordinates: x grows right, y grows DOWN (a page), so a y scale maps its low end to
the bottom of the axes. Sizes are points for strokes and markers, millimetres for layout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.catalog._shared.figure import PALETTES
from nodegraph.catalog._shared.overlay_math import colormap_lut
from nodegraph.figure import series as SR
from nodegraph.figure import stats as ST
from nodegraph.figure.layout import PT_MM

MeasureFn = Callable[[str, float], Tuple[float, float]]

MARKERS = ("circle", "square", "triangle", "diamond", "cross", "plus")

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def default_measure(text: str, pt: float) -> Tuple[float, float]:
    """A width estimate (0.55 em per character) and a line height, in mm."""
    return 0.55 * float(pt) * PT_MM * len(str(text)), 1.2 * float(pt) * PT_MM


# ── number and time formatting ───────────────────────────────────────────────────────
def fmt_number(v: float, step: Optional[float] = None) -> str:
    """``v`` as a tick label: as many decimals as ``step`` needs, no trailing zeros,
    scientific past 1e5 or under 1e-3."""
    if not math.isfinite(v):
        return ""
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        s = f"{v:.2g}"
        if "e" in s:
            mant, exp = s.split("e")
            exp_i = int(exp)
            mant = mant.rstrip("0").rstrip(".")
            return f"{mant}×10{str(exp_i).translate(_SUPER)}" if mant not in ("1", "-1") \
                else ("−" if mant == "-1" else "") + f"10{str(exp_i).translate(_SUPER)}"
        return s
    if step and step > 0:
        dec = max(0, int(math.ceil(-math.log10(step) - 1e-9)))
        return f"{v:.{dec}f}"
    s = f"{v:.4g}"
    if "." in s and "e" not in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_log(v: float) -> str:
    """``10³`` for an exact power of ten, else the plain number."""
    if v > 0:
        k = math.log10(v)
        if abs(k - round(k)) < 1e-9:
            return f"10{str(int(round(k))).translate(_SUPER)}"
    return fmt_number(v)


def fmt_time(seconds: float) -> str:
    """``m:ss`` under an hour, ``h:mm`` under two days, ``d`` days after."""
    s = float(seconds)
    sign = "−" if s < 0 else ""
    s = abs(s)
    if s < 3600:
        return f"{sign}{int(s // 60)}:{int(s % 60):02d}"
    if s < 2 * 86400:
        return f"{sign}{int(s // 3600)}:{int((s % 3600) // 60):02d} h"
    return f"{sign}{s / 86400:.1f} d"


_TIME_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 3 * 3600,
               6 * 3600, 12 * 3600, 86400, 2 * 86400, 7 * 86400, 14 * 86400, 30 * 86400)


# ── scales ───────────────────────────────────────────────────────────────────────────
@dataclass
class Scale:
    """A data → mm mapping along one axis. ``mm0`` is where ``lo`` lands, ``mm1`` where
    ``hi`` lands (for y, ``mm0`` is the bottom, the larger page coordinate)."""
    kind: str                       # linear | log | time | category
    lo: float
    hi: float
    mm0: float
    mm1: float
    categories: Optional[List[str]] = None

    def frac(self, v: float) -> float:
        if self.kind == "log":
            if v <= 0:
                return float("nan")
            return (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi)
                                                             - math.log10(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def to_mm(self, v: float) -> float:
        return self.mm0 + self.frac(v) * (self.mm1 - self.mm0)

    def from_mm(self, mm: float) -> float:
        f = (mm - self.mm0) / (self.mm1 - self.mm0)
        if self.kind == "log":
            return 10 ** (math.log10(self.lo) + f * (math.log10(self.hi) - math.log10(self.lo)))
        return self.lo + f * (self.hi - self.lo)

    def ticks(self, approx: int = 5) -> Tuple[List[float], List[str], List[float]]:
        """``(major positions, labels, minor positions)`` in data units."""
        if self.kind == "category":
            n = len(self.categories or [])
            return list(range(n)), list(self.categories or []), []
        if self.kind == "log":
            from matplotlib.ticker import LogLocator
            maj = [float(v) for v in LogLocator(base=10.0).tick_values(self.lo, self.hi)
                   if self.lo * (1 - 1e-9) <= v <= self.hi * (1 + 1e-9)]
            if len(maj) > 8:
                keep = max(1, int(math.ceil(len(maj) / 6)))
                maj = maj[::keep]
            minors = [float(v) for v in LogLocator(base=10.0, subs=tuple(range(2, 10)))
                      .tick_values(self.lo, self.hi)
                      if self.lo <= v <= self.hi and v not in maj]
            return maj, [fmt_log(v) for v in maj], minors
        if self.kind == "time":
            span = self.hi - self.lo
            step = _TIME_STEPS[-1]
            for s in _TIME_STEPS:
                if span / s <= approx + 1:
                    step = s
                    break
            start = math.ceil(self.lo / step) * step
            maj = []
            v = start
            while v <= self.hi + 1e-9:
                maj.append(float(v))
                v += step
            return maj, [fmt_time(v) for v in maj], []
        from matplotlib.ticker import MaxNLocator
        loc = MaxNLocator(nbins=approx, steps=[1, 2, 2.5, 5, 10])
        vals = [float(v) for v in loc.tick_values(self.lo, self.hi)]
        eps = (self.hi - self.lo) * 1e-9
        maj = [v for v in vals if self.lo - eps <= v <= self.hi + eps]
        step = (maj[1] - maj[0]) if len(maj) > 1 else None
        return maj, [fmt_number(v, step) for v in maj], []


def nice_range(values: Sequence[float], *, log: bool = False,
               fixed: Sequence[Optional[float]] = (None, None), pad: float = 0.05,
               include_zero: bool = False) -> Tuple[float, float]:
    """``(lo, hi)`` for an axis: the finite values' range padded by ``pad`` (multiplicative
    on a log axis), each end overridden by ``fixed``; a degenerate range is widened."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    f_lo, f_hi = (list(fixed) + [None, None])[:2]
    if v.size == 0:
        lo, hi = (1.0, 10.0) if log else (0.0, 1.0)
    else:
        lo, hi = float(v.min()), float(v.max())
        if include_zero and not log:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        if log:
            if hi <= lo:
                lo, hi = lo / 2, hi * 2
            lo, hi = lo / (1 + pad), hi * (1 + pad)
        else:
            if hi <= lo:
                d = abs(lo) if lo else 1.0
                lo, hi = lo - d, hi + d
            span = hi - lo
            lo, hi = lo - span * pad, hi + span * pad
            if include_zero and not log and v.min() >= 0:
                lo = 0.0
    if f_lo is not None:
        lo = float(f_lo)
    if f_hi is not None:
        hi = float(f_hi)
    if log:
        lo, hi = max(lo, 1e-300), max(hi, 1e-300)
    if hi <= lo:
        hi = lo + (abs(lo) if lo else 1.0) if not log else lo * 10
    return lo, hi


# ── helpers ──────────────────────────────────────────────────────────────────────────
def _color(colors: Sequence[str], i: int) -> str:
    return colors[i % len(colors)] if colors else "#000000"


def _hex_lut(cmap: str) -> np.ndarray:
    return colormap_lut(cmap)


def _rgb_hex(rgb: Sequence[int]) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _mapped_color(lut: np.ndarray, t: float) -> str:
    if not math.isfinite(t):
        return "#808080"
    return _rgb_hex(lut[int(round(min(max(t, 0.0), 1.0) * (len(lut) - 1)))])


def _kde(values: np.ndarray, grid: np.ndarray) -> Optional[np.ndarray]:
    v = values[np.isfinite(values)]
    if v.size < 2 or np.ptp(v) == 0:
        return None
    from scipy.stats import gaussian_kde
    try:
        return gaussian_kde(v)(grid)
    except Exception:                      # noqa: BLE001 — a singular sample
        return None


def _box_stats(v: np.ndarray) -> Optional[Dict[str, Any]]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q1, q2, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_f) & (v <= hi_f)]
    wlo = float(inside.min()) if inside.size else q1
    whi = float(inside.max()) if inside.size else q3
    out = v[(v < lo_f) | (v > hi_f)]
    return {"q1": q1, "median": q2, "q3": q3, "whisker_lo": wlo, "whisker_hi": whi,
            "outliers": out.tolist(), "n": int(v.size), "mean": float(v.mean())}


def _fit_curve(model: str, xs: Sequence[float], ys: Sequence[float], *, log_x: bool,
               log_y: bool, x_lo: float, x_hi: float, n: int = 64
               ) -> Optional[Tuple[List[float], List[float], str]]:
    """The fitted curve of ``model`` through one series, sampled across the axis."""
    if model == "linear":
        fit = SR.xy_fit(xs, ys, log_x=log_x, log_y=log_y)
    elif model == "power":
        fit = SR.xy_fit(xs, ys, log_x=True, log_y=True)
    elif model == "langmuir":
        fit = SR.xy_langmuir(xs, ys, log_x=log_x, log_y=log_y)
    elif model == "exponential":
        fit = SR.xy_exponential(xs, ys, log_x=log_x, log_y=log_y)
    else:
        return None
    if not fit:
        return None
    if log_x or model == "power":
        lo, hi = max(x_lo, 1e-300), max(x_hi, 1e-300)
        grid = np.geomspace(lo, hi, n)
    else:
        grid = np.linspace(x_lo, x_hi, n)
    if model in ("linear", "power"):
        fx = np.log10(grid) if (log_x or model == "power") else grid
        fy = fit["slope"] * fx + fit["intercept"]
        yv = 10 ** fy if (log_y or model == "power") else fy
    elif model == "langmuir":
        yv = fit["bmax"] * grid / (fit["kd"] + grid)
    else:
        yv = fit["a"] * np.exp(-fit["k"] * grid)
    keep = np.isfinite(yv)
    return grid[keep].tolist(), np.asarray(yv)[keep].tolist(), str(fit.get("text", ""))


def _text(kind: str, s: str, x: float, y: float, *, pt: float, anchor: str = "start",
          rotate: float = 0.0, color: str = "#111111", bold: bool = False) -> Dict[str, Any]:
    return {"kind": "text", "role": kind, "text": str(s), "x_mm": float(x), "y_mm": float(y),
            "pt": float(pt), "anchor": anchor, "rotate": float(rotate), "color": color,
            "bold": bool(bold)}


# ── the geometry ─────────────────────────────────────────────────────────────────────
def chart_geometry(chart: Mapping[str, Any], series: Any, *, rect_mm: Sequence[float],
                   style: Mapping[str, Any], measure: Optional[MeasureFn] = None,
                   units: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Everything to paint for ``chart`` (a normalized chart block) with its resolved
    ``series`` inside ``rect_mm`` ``[x, y, w, h]`` under ``style`` (``font_pt``,
    ``line_width``, ``marker_size``, ``palette``, ``grid``, ``dark``).

    ``series`` is what :mod:`nodegraph.figure.series` returns for the chart's kind — XY
    series for ``xy`` / ``timeseries``, bar series for ``bar``, distribution / strip /
    superplot series for ``distribution``, ``(rows, cols, matrix)`` for ``heatmap``, the
    hexbin dict for ``hexbin``, MSD series for ``msd``, survival series for ``survival``,
    rose series for ``rose``.

    Returns ``{"frame_mm", "axes_mm", "x", "y", "marks", "texts", "legend", "colorbar",
    "stats"}``. Marks carry ``kind`` (``polyline`` | ``markers`` | ``segments`` |
    ``polygon`` | ``rects`` | ``box`` | ``cells`` | ``wedge`` | ``bracket``) and page
    coordinates in mm; every mark that draws data rows also carries ``series`` (its
    index) so a hit can be traced back."""
    measure = measure or default_measure
    kind = str(chart.get("kind", "xy"))
    draw = dict(chart.get("draw") or {})
    axes_cfg = dict(chart.get("axes") or {})
    legend_cfg = dict(chart.get("legend") or {"show": True, "loc": "inside"})
    fit_cfg = dict(chart.get("fit") or {})
    stats_cfg = dict(chart.get("stats") or {})
    fx, fy, fw, fh = (float(v) for v in rect_mm)
    pt = float(style.get("font_pt", 8.0))
    lw = float(style.get("line_width", 1.0))
    ms = float(style.get("marker_size", 3.0))
    dark = bool(style.get("dark", False))
    fg = "#e6e6e6" if dark else "#111111"
    grid_on = bool(style.get("grid", False))
    colors = list(PALETTES.get(str(style.get("palette", "colorblind")), PALETTES["colorblind"]))
    log_x = bool(axes_cfg.get("log_x", False))
    log_y = bool(axes_cfg.get("log_y", False))
    units = units or {}

    marks: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    legend_entries: List[Dict[str, Any]] = []
    colorbar: Optional[Dict[str, Any]] = None
    stat_results: List[Dict[str, Any]] = []
    brackets: List[Tuple[int, int, str]] = []

    # ── 1. the data extents and the axis kinds ───────────────────────────────────
    x_kind, y_kind = ("log" if log_x else "linear"), ("log" if log_y else "linear")
    x_cats: Optional[List[str]] = None
    y_cats: Optional[List[str]] = None
    xs_all: List[float] = []
    ys_all: List[float] = []
    include_zero_y = False
    sub = str(chart.get("sub_kind") or "histogram")
    if kind == "timeseries":
        x_kind = "time" if not log_x else "log"
    if kind in ("xy", "timeseries", "msd", "survival"):
        for s in series:
            if kind == "msd":
                xv, yv = s.get("lag", []), s.get("msd", [])
            elif kind == "survival":
                xv, yv = s.get("t", []), s.get("s", [])
            else:
                xv, yv = s.get("x", []), s.get("y", [])
            xs_all.extend(float(v) for v in xv)
            ys_all.extend(float(v) for v in yv)
            err = s.get("err")
            if err:
                ys_all.extend(float(a) + float(b) for a, b in zip(yv, err))
                ys_all.extend(float(a) - float(b) for a, b in zip(yv, err))
        if kind == "msd" and bool(draw.get("loglog", True)):
            x_kind = y_kind = "log"
        if kind == "survival":
            include_zero_y = True
            ys_all.append(1.0)
    elif kind == "bar":
        x_kind = "category"
        x_cats = list(series[0]["categories"]) if series else []
        for s in series:
            if draw.get("stacked"):
                continue
            ys_all.extend(float(v) + float(e) for v, e in zip(s["y"], s["err"]))
            ys_all.extend(float(v) - float(e) for v, e in zip(s["y"], s["err"]))
        if draw.get("stacked") and series:
            tot = np.nansum(np.array([[float(v) for v in s["y"]] for s in series]), axis=0)
            ys_all.extend(tot.tolist())
        include_zero_y = True
    elif kind == "distribution":
        if sub in ("histogram", "kde", "ecdf"):
            for s in series:
                xs_all.extend(float(v) for v in s["values"])
            include_zero_y = True
        else:
            x_kind = "category"
            x_cats = [s["label"] or "all" for s in series]
            for s in series:
                ys_all.extend(float(v) for v in s["values"])
    elif kind == "heatmap":
        rows, cols, mat = series
        x_kind = y_kind = "category"
        x_cats, y_cats = list(cols), list(rows)
    elif kind == "hexbin":
        ex = series.get("extent", [0, 1, 0, 1])
        xs_all = [10 ** ex[0], 10 ** ex[1]] if series.get("log_x") else [ex[0], ex[1]]
        ys_all = [10 ** ex[2], 10 ** ex[3]] if series.get("log_y") else [ex[2], ex[3]]
        x_kind = "log" if series.get("log_x") else "linear"
        y_kind = "log" if series.get("log_y") else "linear"
    elif kind == "rose":
        pass
    else:
        raise ValueError(f"chart_geometry: unknown chart kind {kind!r}")

    # ── 2. histogram / kde / ecdf series → xy-like marks need their y extents first ──
    hist_bins: Optional[np.ndarray] = None
    hist_counts: List[np.ndarray] = []
    kde_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    if kind == "distribution" and sub == "histogram":
        allv = np.asarray(xs_all, dtype=float)
        allv = allv[np.isfinite(allv)]
        if log_x:
            allv = allv[allv > 0]
        xr = axes_cfg.get("x_range") or [None, None]
        lo = float(xr[0]) if xr[0] is not None else (float(allv.min()) if allv.size else 0.0)
        hi = float(xr[1]) if xr[1] is not None else (float(allv.max()) if allv.size else 1.0)
        if hi <= lo:
            hi = lo + 1.0
        nb = max(1, int(chart.get("bins", 20)))
        hist_bins = np.geomspace(max(lo, 1e-300), hi, nb + 1) if log_x else np.linspace(lo, hi, nb + 1)
        norm = str(chart.get("normalize", "count"))
        for s in series:
            v = np.asarray(s["values"], dtype=float)
            c, _ = np.histogram(v[np.isfinite(v)], bins=hist_bins)
            c = c.astype(float)
            if norm == "density":
                widths = np.diff(hist_bins)
                tot = c.sum()
                c = c / (tot * widths) if tot > 0 else c
            elif norm == "percent":
                tot = c.sum()
                c = 100.0 * c / tot if tot > 0 else c
            hist_counts.append(c)
            ys_all.extend(c.tolist())
        xs_all = hist_bins.tolist()
    elif kind == "distribution" and sub == "kde":
        allv = np.asarray(xs_all, dtype=float)
        allv = allv[np.isfinite(allv)]
        if allv.size:
            lo, hi = nice_range(allv, log=log_x, fixed=axes_cfg.get("x_range") or (None, None))
            grid = np.geomspace(lo, hi, 200) if log_x else np.linspace(lo, hi, 200)
            for s in series:
                d = _kde(np.asarray(s["values"], dtype=float), grid)
                if d is None:
                    kde_curves.append((grid, np.zeros_like(grid)))
                else:
                    kde_curves.append((grid, d))
                    ys_all.extend(d.tolist())
    elif kind == "distribution" and sub == "ecdf":
        ys_all = [0.0, 1.0]

    # ── 3. scales and layout ─────────────────────────────────────────────────────
    if x_kind == "category":
        x_lo, x_hi = -0.5, max(0.5, len(x_cats or []) - 0.5)
    else:
        x_lo, x_hi = nice_range(xs_all, log=(x_kind == "log"),
                                fixed=axes_cfg.get("x_range") or (None, None),
                                pad=0.0 if kind in ("hexbin",) or sub == "histogram" else 0.05)
    if y_kind == "category":
        y_lo, y_hi = -0.5, max(0.5, len(y_cats or []) - 0.5)
    else:
        y_lo, y_hi = nice_range(ys_all, log=(y_kind == "log"),
                                fixed=axes_cfg.get("y_range") or (None, None),
                                include_zero=include_zero_y)

    title = str(axes_cfg.get("title") or "")
    x_label = str(axes_cfg.get("x_label") or "")
    y_label = str(axes_cfg.get("y_label") or "")
    if not x_label and units.get("x"):
        x_label = units["x"]
    if not y_label and units.get("y"):
        y_label = units["y"]
    line_h = measure("Ag", pt)[1]
    tick_len = 1.0
    pad = 1.0

    # legend entries (decided early: the right/bottom legend reserves space)
    show_legend = bool(legend_cfg.get("show", True)) and legend_cfg.get("loc") != "none"
    if kind in ("xy", "timeseries", "msd", "survival", "bar"):
        for i, s in enumerate(series):
            if s.get("label"):
                legend_entries.append({"label": s["label"], "color": _color(colors, i),
                                       "marker": MARKERS[i % len(MARKERS)]
                                       if kind != "bar" else None,
                                       "line": kind == "bar" or draw.get("mode", "scatter")
                                       != "scatter", "fill": kind == "bar"})
    elif kind == "distribution":
        for i, s in enumerate(series):
            if s.get("label") and sub in ("histogram", "kde", "ecdf"):
                legend_entries.append({"label": s["label"], "color": _color(colors, i),
                                       "marker": None, "line": sub != "histogram",
                                       "fill": sub == "histogram"})
            if sub == "superplot":
                reps = [m["replicate"] for m in s.get("means", [])]
                for r in reps:
                    if r not in [e["label"] for e in legend_entries]:
                        legend_entries.append({"label": r, "color": _color(colors, len(legend_entries)),
                                               "marker": "circle", "line": False, "fill": False})
    elif kind == "rose":
        for i, s in enumerate(series):
            if s.get("label"):
                legend_entries.append({"label": s["label"], "color": _color(colors, i),
                                       "marker": None, "line": False, "fill": True})
    if not show_legend or len(legend_entries) > 12:
        legend_entries = [] if not show_legend else legend_entries[:12]
    leg_loc = str(legend_cfg.get("loc", "inside"))
    leg_w = (max((measure(e["label"], pt)[0] for e in legend_entries), default=0.0) + 6.0) \
        if legend_entries else 0.0
    leg_h = (len(legend_entries) * line_h + 2 * pad) if legend_entries else 0.0

    # provisional scales for tick text (ranges are known; mm come after the margins)
    def _tick_texts(kind_: str, lo: float, hi: float, cats: Optional[List[str]]
                    ) -> Tuple[List[float], List[str], List[float]]:
        return Scale(kind_, lo, hi, 0.0, 1.0, cats).ticks(approx=5)

    xt_pos, xt_lab, xt_min = _tick_texts(x_kind, x_lo, x_hi, x_cats)
    yt_pos, yt_lab, yt_min = _tick_texts(y_kind, y_lo, y_hi, y_cats)
    if kind == "rose":
        xt_pos, xt_lab, yt_pos, yt_lab = [], [], [], []
    y_tick_w = max((measure(s, pt)[0] for s in yt_lab), default=0.0)
    x_tick_h = line_h if xt_lab else 0.0
    # brackets reserve head-room
    if kind in ("distribution", "bar") and x_kind == "category" \
            and str(stats_cfg.get("test", "none")) != "none":
        groups: Dict[str, List[float]] = {}
        if kind == "bar":
            pass                                           # bars carry means only
        else:
            for s in series:
                groups[s["label"] or "all"] = list(s["values"])
        if groups:
            pairs_cfg = stats_cfg.get("pairs") or None
            pairs = [tuple(p) for p in pairs_cfg] if pairs_cfg else None
            try:
                stat_results = ST.compare(groups, test=str(stats_cfg["test"]), pairs=pairs)
            except ValueError:
                stat_results = []
            idx = {lab: i for i, lab in enumerate(x_cats or [])}
            for r in stat_results:
                if r["a"] in idx and r["b"] in idx and math.isfinite(r["p"]):
                    brackets.append((idx[r["a"]], idx[r["b"]],
                                     r["stars"] if stats_cfg.get("label", "stars") == "stars"
                                     else r["text"]))
    levels = ST.bracket_levels([(a, b) for a, b, _ in brackets]) if brackets else []
    n_levels = (max(levels) + 1) if levels else 0
    bracket_step = line_h + 1.5
    top_extra = n_levels * bracket_step + (1.0 if n_levels else 0.0)

    left = y_tick_w + tick_len + pad + ((line_h + pad) if y_label else 0.0) + pad
    bottom = x_tick_h + tick_len + pad + ((line_h + pad) if x_label else 0.0) + pad
    top = (line_h * 1.1 + pad if title else pad) + top_extra
    right = 1.5
    cbar_w = 0.0
    if kind in ("heatmap", "hexbin"):
        cbar_w = 3.0 + pad + max((measure("0.000", pt)[0]), 4.0) + pad
        right += cbar_w + pad
    if legend_entries and leg_loc == "right":
        right += leg_w + pad
    if legend_entries and leg_loc == "bottom":
        bottom += leg_h + pad
    aw = max(5.0, fw - left - right)
    ah = max(5.0, fh - top - bottom)
    ax, ay = fx + left, fy + top
    xs_scale = Scale(x_kind, x_lo, x_hi, ax, ax + aw, x_cats)
    if kind == "heatmap":                        # row 0 at the top, as a matrix reads
        ys_scale = Scale(y_kind, y_lo, y_hi, ay, ay + ah, y_cats)
    else:
        ys_scale = Scale(y_kind, y_lo, y_hi, ay + ah, ay, y_cats)

    X = xs_scale.to_mm
    Y = ys_scale.to_mm

    # ── 4. axes furniture ────────────────────────────────────────────────────────
    if kind != "rose":
        marks.append({"kind": "rect", "role": "axes", "rect_mm": [ax, ay, aw, ah],
                      "stroke": fg, "width_pt": 0.75, "fill": None})
        tick_segs: List[List[List[float]]] = []
        grid_segs: List[List[List[float]]] = []
        for v, lab in zip(xt_pos, xt_lab):
            mx = X(v)
            tick_segs.append([[mx, ay + ah], [mx, ay + ah + tick_len]])
            texts.append(_text("x_tick", lab, mx, ay + ah + tick_len + pad * 0.5, pt=pt,
                               anchor="middle_top", color=fg))
            if grid_on and x_kind != "category":
                grid_segs.append([[mx, ay], [mx, ay + ah]])
        for v in xt_min:
            mx = X(v)
            tick_segs.append([[mx, ay + ah], [mx, ay + ah + tick_len * 0.5]])
        for v, lab in zip(yt_pos, yt_lab):
            my = Y(v) if y_kind != "category" else Y(v)
            tick_segs.append([[ax - tick_len, my], [ax, my]])
            texts.append(_text("y_tick", lab, ax - tick_len - pad * 0.5, my, pt=pt,
                               anchor="end_middle", color=fg))
            if grid_on and y_kind != "category":
                grid_segs.append([[ax, my], [ax + aw, my]])
        for v in yt_min:
            my = Y(v)
            tick_segs.append([[ax - tick_len * 0.5, my], [ax, my]])
        if grid_segs:
            marks.append({"kind": "segments", "role": "grid", "segs_mm": grid_segs,
                          "stroke": "#80808060", "width_pt": 0.4})
        marks.append({"kind": "segments", "role": "ticks", "segs_mm": tick_segs,
                      "stroke": fg, "width_pt": 0.6})
        if x_label:
            texts.append(_text("x_label", x_label, ax + aw / 2, fy + fh - pad, pt=pt,
                               anchor="middle_bottom", color=fg))
        if y_label:
            texts.append(_text("y_label", y_label, fx + pad, ay + ah / 2, pt=pt,
                               anchor="middle_top", rotate=-90.0, color=fg))
    if title:
        texts.append(_text("title", title, fx + fw / 2, fy + pad * 0.5, pt=pt * 1.1,
                           anchor="middle_top", color=fg, bold=True))

    # ── 5. the data marks ────────────────────────────────────────────────────────
    def _pts(xv: Sequence[float], yv: Sequence[float]) -> List[List[float]]:
        out = []
        for a, b in zip(xv, yv):
            mx, my = X(float(a)), Y(float(b))
            if math.isfinite(mx) and math.isfinite(my):
                out.append([mx, my])
        return out

    if kind in ("xy", "timeseries", "msd", "survival"):
        mode = str(draw.get("mode", "scatter"))
        if kind == "msd":
            mode = "line_markers"
        if kind == "survival":
            mode = "line"
        for i, s in enumerate(series):
            col = _color(colors, i)
            if kind == "msd":
                xv, yv = s.get("lag", []), s.get("msd", [])
            elif kind == "survival":
                xv, yv = s.get("t", []), s.get("s", [])
            else:
                xv, yv = s.get("x", []), s.get("y", [])
            xv = [float(v) for v in xv]
            yv = [float(v) for v in yv]
            if kind == "survival" or draw.get("step"):
                sx, sy = [], []
                for k in range(len(xv)):
                    if k > 0:
                        sx.append(xv[k]); sy.append(yv[k - 1])
                    sx.append(xv[k]); sy.append(yv[k])
                if xv and x_hi > xv[-1]:
                    sx.append(x_hi); sy.append(yv[-1])
                xl, yl = sx, sy
            else:
                xl, yl = xv, yv
            if mode in ("line", "line_markers"):
                marks.append({"kind": "polyline", "role": "series", "series": i,
                              "pts_mm": _pts(xl, yl), "stroke": col, "width_pt": lw})
            err = s.get("err")
            if err and kind in ("xy", "timeseries"):
                if str(draw.get("error_style", "bars")) == "band":
                    up = _pts(xv, [a + b for a, b in zip(yv, err)])
                    dn = _pts(xv, [a - b for a, b in zip(yv, err)])
                    if up and dn:
                        marks.append({"kind": "polygon", "role": "band", "series": i,
                                      "pts_mm": up + dn[::-1], "fill": col + "40",
                                      "stroke": None})
                else:
                    segs = []
                    caps = []
                    cap = max(0.3, ms * PT_MM * 0.5)
                    for a, b, e in zip(xv, yv, err):
                        mx = X(a)
                        y0, y1 = Y(b - e), Y(b + e)
                        if all(math.isfinite(v) for v in (mx, y0, y1)):
                            segs.append([[mx, y0], [mx, y1]])
                            caps.append([[mx - cap, y0], [mx + cap, y0]])
                            caps.append([[mx - cap, y1], [mx + cap, y1]])
                    marks.append({"kind": "segments", "role": "errorbars", "series": i,
                                  "segs_mm": segs + caps, "stroke": col, "width_pt": lw * 0.8})
            if mode in ("scatter", "line_markers"):
                pts = _pts(xv, yv)
                mk: Dict[str, Any] = {"kind": "markers", "role": "series", "series": i,
                                      "pts_mm": pts, "shape": MARKERS[i % len(MARKERS)],
                                      "size_pt": ms, "fill": col, "stroke": None}
                if s.get("c") is not None:
                    cv = np.asarray(s["c"], dtype=float)
                    lo_c, hi_c = nice_range(cv, pad=0.0)
                    lut = _hex_lut(str(chart.get("cmap", "viridis")))
                    mk["fills"] = [_mapped_color(lut, (v - lo_c) / (hi_c - lo_c)
                                                 if hi_c > lo_c else 0.5) for v in cv]
                    colorbar = {"rect_mm": [ax + aw + pad, ay, 3.0, ah], "lo": lo_c,
                                "hi": hi_c, "cmap": str(chart.get("cmap", "viridis")),
                                "label": str(chart.get("color_label") or chart.get("color_by")
                                             or ""), "vertical": True}
                if s.get("text"):
                    mk["labels"] = [str(v) for v in s["text"]]
                marks.append(mk)
            model = str(fit_cfg.get("model", "none"))
            if kind == "msd" and model == "none":
                model = "power"                  # an MSD is read by its exponent
            if model != "none" and kind in ("xy", "timeseries", "msd") and len(xv) >= 2:
                span = fit_cfg.get("span") or [None, None]
                lo_s = float(span[0]) if span[0] is not None else (min(xv) if xv else x_lo)
                hi_s = float(span[1]) if span[1] is not None else (max(xv) if xv else x_hi)
                if kind == "msd":
                    model = "power"
                cur = _fit_curve(model, xv, yv, log_x=(x_kind == "log"),
                                 log_y=(y_kind == "log"), x_lo=lo_s, x_hi=hi_s)
                if cur:
                    cx_, cy_, txt = cur
                    marks.append({"kind": "polyline", "role": "fit", "series": i,
                                  "pts_mm": _pts(cx_, cy_), "stroke": col,
                                  "width_pt": lw, "dash": "dashed"})
                    for e in legend_entries:
                        if e["label"] == s.get("label") and txt:
                            e["label"] = f"{e['label']}: {txt}"
    elif kind == "bar":
        n_series = max(1, len(series))
        stacked = bool(draw.get("stacked", False))
        group_w = 0.8
        bw = group_w if stacked else group_w / n_series
        base = Y(0.0) if y_kind != "log" else Y(y_lo)
        running = np.zeros(len(x_cats or []), dtype=float)
        for i, s in enumerate(series):
            col = _color(colors, i)
            rects = []
            segs = []
            for j, (v, e) in enumerate(zip(s["y"], s["err"])):
                if not math.isfinite(float(v)):
                    continue
                if stacked:
                    x0 = j - group_w / 2
                    y_bot, y_top = running[j], running[j] + float(v)
                    running[j] = y_top
                else:
                    x0 = j - group_w / 2 + i * bw
                    y_bot, y_top = 0.0, float(v)
                mx0, mx1 = X(x0), X(x0 + bw)
                my0, my1 = Y(y_bot) if y_kind != "log" else base, Y(y_top)
                if not all(math.isfinite(q) for q in (mx0, mx1, my0, my1)):
                    continue
                rects.append([mx0, min(my0, my1), mx1 - mx0, abs(my1 - my0)])
                if e and not stacked:
                    cx = (mx0 + mx1) / 2
                    segs.append([[cx, Y(y_top - e)], [cx, Y(y_top + e)]])
            marks.append({"kind": "rects", "role": "bars", "series": i, "rects_mm": rects,
                          "fill": col, "stroke": fg, "width_pt": 0.5})
            if segs:
                marks.append({"kind": "segments", "role": "errorbars", "series": i,
                              "segs_mm": segs, "stroke": fg, "width_pt": lw * 0.8})
    elif kind == "distribution":
        if sub == "histogram" and hist_bins is not None:
            alpha = "" if len(series) == 1 else "80"
            for i, c in enumerate(hist_counts):
                col = _color(colors, i)
                rects = []
                for k in range(len(c)):
                    x0, x1 = X(float(hist_bins[k])), X(float(hist_bins[k + 1]))
                    y0, y1 = Y(0.0 if y_kind != "log" else y_lo), Y(float(c[k]))
                    if all(math.isfinite(q) for q in (x0, x1, y0, y1)) and c[k] > 0:
                        rects.append([x0, min(y0, y1), x1 - x0, abs(y1 - y0)])
                marks.append({"kind": "rects", "role": "bars", "series": i, "rects_mm": rects,
                              "fill": col + alpha, "stroke": None, "width_pt": 0.0})
        elif sub == "kde":
            for i, (grid, d) in enumerate(kde_curves):
                col = _color(colors, i)
                pts = _pts(grid.tolist(), d.tolist())
                marks.append({"kind": "polyline", "role": "series", "series": i,
                              "pts_mm": pts, "stroke": col, "width_pt": lw})
                if pts:
                    base = Y(0.0)
                    marks.append({"kind": "polygon", "role": "band", "series": i,
                                  "pts_mm": [[pts[0][0], base]] + pts + [[pts[-1][0], base]],
                                  "fill": col + "30", "stroke": None})
        elif sub == "ecdf":
            for i, s in enumerate(series):
                v = np.sort(np.asarray(s["values"], dtype=float))
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                n = v.size
                xs_ = [float(v[0])]
                ys_ = [0.0]
                for k in range(n):
                    xs_.append(float(v[k])); ys_.append(k / n)
                    xs_.append(float(v[k])); ys_.append((k + 1) / n)
                marks.append({"kind": "polyline", "role": "series", "series": i,
                              "pts_mm": _pts(xs_, ys_), "stroke": _color(colors, i),
                              "width_pt": lw})
        else:                                  # box | violin | strip | superplot
            half = 0.35
            for i, s in enumerate(series):
                col = _color(colors, i)
                v = np.asarray(s["values"], dtype=float)
                st = _box_stats(v)
                if sub in ("box", "violin") and st is not None:
                    if sub == "violin":
                        grid = np.linspace(max(y_lo, float(np.nanmin(v))),
                                           min(y_hi, float(np.nanmax(v))), 100) \
                            if np.isfinite(v).any() else np.array([])
                        d = _kde(v, grid) if grid.size else None
                        if d is not None and d.max() > 0:
                            w = d / d.max() * half
                            right_side = [[X(i + float(wk)), Y(float(g))] for g, wk in zip(grid, w)]
                            left_side = [[X(i - float(wk)), Y(float(g))] for g, wk in zip(grid, w)]
                            marks.append({"kind": "polygon", "role": "violin", "series": i,
                                          "pts_mm": right_side + left_side[::-1],
                                          "fill": col + "60", "stroke": col})
                        marks.append({"kind": "markers", "role": "median", "series": i,
                                      "pts_mm": [[X(i), Y(st["median"])]], "shape": "circle",
                                      "size_pt": ms, "fill": fg, "stroke": None})
                    else:
                        bx0, bx1 = X(i - half * 0.8), X(i + half * 0.8)
                        marks.append({"kind": "box", "role": "box", "series": i,
                                      "rect_mm": [bx0, Y(st["q3"]), bx1 - bx0,
                                                  Y(st["q1"]) - Y(st["q3"])],
                                      "median_mm": [[bx0, Y(st["median"])], [bx1, Y(st["median"])]],
                                      "whiskers_mm": [[[X(i), Y(st["q1"])], [X(i), Y(st["whisker_lo"])]],
                                                      [[X(i), Y(st["q3"])], [X(i), Y(st["whisker_hi"])]]],
                                      "caps_mm": [[[X(i - half * 0.4), Y(st["whisker_lo"])],
                                                   [X(i + half * 0.4), Y(st["whisker_lo"])]],
                                                  [[X(i - half * 0.4), Y(st["whisker_hi"])],
                                                   [X(i + half * 0.4), Y(st["whisker_hi"])]]],
                                      "fill": col + "40", "stroke": col, "width_pt": lw})
                        if st["outliers"]:
                            marks.append({"kind": "markers", "role": "outliers", "series": i,
                                          "pts_mm": _pts([i] * len(st["outliers"]), st["outliers"]),
                                          "shape": "circle", "size_pt": ms * 0.7,
                                          "fill": None, "stroke": col})
                elif sub == "strip":
                    offs = s.get("offsets") or [0.0] * int(v.size)
                    marks.append({"kind": "markers", "role": "series", "series": i,
                                  "pts_mm": _pts([i + float(o) for o in offs], v.tolist()),
                                  "shape": "circle", "size_pt": ms, "fill": col + "a0",
                                  "stroke": None})
                    if st is not None:
                        marks.append({"kind": "segments", "role": "mean", "series": i,
                                      "segs_mm": [[[X(i - half), Y(st["mean"])],
                                                   [X(i + half), Y(st["mean"])]]],
                                      "stroke": fg, "width_pt": lw * 1.2})
                elif sub == "superplot":
                    reps = [e["label"] for e in legend_entries] or \
                        [m["replicate"] for m in s.get("means", [])]
                    rep_of = s.get("replicate") or []
                    rng = np.random.default_rng(i)
                    offs = rng.uniform(-half * 0.8, half * 0.8, int(v.size))
                    fills = [_color(colors, reps.index(r) if r in reps else 0) + "80"
                             for r in rep_of] or [col + "80"] * int(v.size)
                    marks.append({"kind": "markers", "role": "series", "series": i,
                                  "pts_mm": _pts([i + float(o) for o in offs], v.tolist()),
                                  "shape": "circle", "size_pt": ms * 0.8, "fill": col,
                                  "fills": fills, "stroke": None})
                    mpts = []
                    mfills = []
                    for mrec in s.get("means", []):
                        mpts.append([X(i), Y(float(mrec["mean"]))])
                        r = mrec["replicate"]
                        mfills.append(_color(colors, reps.index(r) if r in reps else 0))
                    if mpts:
                        marks.append({"kind": "markers", "role": "means", "series": i,
                                      "pts_mm": mpts, "shape": "diamond", "size_pt": ms * 1.8,
                                      "fill": col, "fills": mfills, "stroke": fg})
            # brackets above the tallest mark
            if brackets:
                y_top_mm = ay
                for (a, b, label), lvl in zip(brackets, levels):
                    y_mm = y_top_mm - 1.0 - lvl * bracket_step
                    marks.append({"kind": "bracket", "role": "stats",
                                  "x0_mm": X(a), "x1_mm": X(b), "y_mm": y_mm,
                                  "drop_mm": 1.0, "text": label, "stroke": fg,
                                  "width_pt": 0.6})
                    texts.append(_text("bracket", label, (X(a) + X(b)) / 2, y_mm - 0.3,
                                       pt=pt, anchor="middle_bottom", color=fg))
    elif kind == "heatmap":
        rows, cols, mat = series
        arr = np.array([[np.nan if v is None else float(v) for v in r] for r in mat],
                       dtype=float) if rows and cols else np.zeros((0, 0))
        cr = chart.get("color_range") or [None, None]
        finite = arr[np.isfinite(arr)] if arr.size else np.array([])
        lo_c = float(cr[0]) if cr[0] is not None else (float(finite.min()) if finite.size else 0.0)
        hi_c = float(cr[1]) if cr[1] is not None else (float(finite.max()) if finite.size else 1.0)
        if hi_c <= lo_c:
            hi_c = lo_c + 1.0
        cmap = str(chart.get("cmap", "viridis"))
        lut = _hex_lut(cmap)
        cells = []
        for r_i in range(len(rows)):
            for c_i in range(len(cols)):
                v = arr[r_i, c_i]
                x0, x1 = X(c_i - 0.5), X(c_i + 0.5)
                y0, y1 = Y(r_i - 0.5), Y(r_i + 0.5)         # the y scale runs top-down
                cells.append({"rect_mm": [x0, min(y0, y1), x1 - x0, abs(y1 - y0)],
                              "fill": _mapped_color(lut, (v - lo_c) / (hi_c - lo_c))
                              if math.isfinite(v) else None, "value": None if not
                              math.isfinite(v) else float(v), "row": r_i, "col": c_i})
        marks.append({"kind": "cells", "role": "heatmap", "cells": cells})
        colorbar = {"rect_mm": [ax + aw + pad, ay, 3.0, ah], "lo": lo_c, "hi": hi_c,
                    "cmap": cmap, "label": str(chart.get("color_label") or chart.get("value")
                                               or ""), "vertical": True}
    elif kind == "hexbin":
        counts = np.asarray(series.get("counts", []), dtype=float)
        if bool(draw.get("log_count", False)):
            counts = np.log10(counts + 1.0)
        lo_c, hi_c = (0.0, float(counts.max())) if counts.size else (0.0, 1.0)
        if hi_c <= lo_c:
            hi_c = lo_c + 1.0
        cmap = str(chart.get("cmap", "viridis"))
        lut = _hex_lut(cmap)
        hexes = []
        hx = series.get("hex", [])
        for (cx, cy), cnt in zip(series.get("centers", []), counts.tolist()):
            pts = []
            for dx, dy in hx:
                px_ = (math.log10(cx) + dx) if series.get("log_x") else cx + dx
                py_ = (math.log10(cy) + dy) if series.get("log_y") else cy + dy
                xv = 10 ** px_ if series.get("log_x") else px_
                yv = 10 ** py_ if series.get("log_y") else py_
                pts.append([X(xv), Y(yv)])
            hexes.append({"pts_mm": pts, "fill": _mapped_color(lut, (cnt - lo_c) / (hi_c - lo_c)),
                          "count": float(cnt)})
        marks.append({"kind": "cells", "role": "hexbin", "cells": hexes})
        colorbar = {"rect_mm": [ax + aw + pad, ay, 3.0, ah], "lo": lo_c, "hi": hi_c,
                    "cmap": cmap, "label": "log₁₀(count + 1)" if draw.get("log_count")
                    else "count", "vertical": True}
    elif kind == "rose":
        cx, cy = ax + aw / 2, ay + ah / 2
        R = max(2.0, min(aw, ah) / 2 - line_h)
        radial = str(draw.get("radial", "sqrt"))
        all_counts = [float(c) for s in series for c in s.get("counts", [])]
        cmax = max(all_counts) if all_counts else 1.0
        scale_fn = (lambda c: math.sqrt(c / cmax)) if radial == "sqrt" else (lambda c: c / cmax)
        for i, s in enumerate(series):
            col = _color(colors, i)
            edges = s.get("edges_deg", [])
            wedges = []
            for k, cnt in enumerate(s.get("counts", [])):
                if cnt <= 0:
                    continue
                wedges.append({"cx_mm": cx, "cy_mm": cy, "r0_mm": 0.0,
                               "r1_mm": R * scale_fn(float(cnt)),
                               "a0_deg": float(edges[k]), "a1_deg": float(edges[k + 1]),
                               "count": float(cnt)})
            marks.append({"kind": "wedge", "role": "series", "series": i, "wedges": wedges,
                          "fill": col + ("80" if len(series) > 1 else ""), "stroke": fg,
                          "width_pt": 0.4})
        rings = []
        for f in (0.5, 1.0):
            rings.append({"cx_mm": cx, "cy_mm": cy, "r_mm": R * f,
                          "label": fmt_number(cmax * (f ** 2 if radial == "sqrt" else f))})
        marks.append({"kind": "rings", "role": "axes", "rings": rings, "stroke": "#808080",
                      "width_pt": 0.4})
        for a in (0, 90, 180, 270):
            rad = math.radians(a)
            texts.append(_text("x_tick", f"{a}°", cx + (R + line_h * 0.6) * math.cos(rad),
                               cy - (R + line_h * 0.6) * math.sin(rad), pt=pt,
                               anchor="middle_middle", color=fg))

    # ── 6. legend and colour bar geometry ────────────────────────────────────────
    legend: Optional[Dict[str, Any]] = None
    if legend_entries:
        if leg_loc == "right":
            lx, ly = ax + aw + pad + (cbar_w + pad if cbar_w else 0.0), ay
        elif leg_loc == "bottom":
            lx, ly = ax, fy + fh - leg_h
        else:
            lx, ly = ax + aw - leg_w - pad, ay + pad
        legend = {"rect_mm": [lx, ly, leg_w, leg_h], "entries": legend_entries,
                  "line_h_mm": line_h, "pt": pt, "loc": leg_loc,
                  "fill": ("#202020c0" if dark else "#ffffffc0") if leg_loc == "inside" else None}
    if colorbar is not None:
        lo_c, hi_c = colorbar["lo"], colorbar["hi"]
        cs = Scale("linear", lo_c, hi_c, colorbar["rect_mm"][1] + colorbar["rect_mm"][3],
                   colorbar["rect_mm"][1])
        pos, labs, _ = cs.ticks(approx=4)
        colorbar["ticks"] = [{"pos_mm": cs.to_mm(v), "label": l} for v, l in zip(pos, labs)]
        colorbar["pt"] = pt

    return {"frame_mm": [fx, fy, fw, fh], "axes_mm": [ax, ay, aw, ah],
            "x": {"scale": x_kind, "lim": [x_lo, x_hi], "ticks": [{"pos_mm": X(v), "label": l}
                                                                   for v, l in zip(xt_pos, xt_lab)],
                  "label": x_label, "categories": x_cats},
            "y": {"scale": y_kind, "lim": [y_lo, y_hi], "ticks": [{"pos_mm": Y(v), "label": l}
                                                                   for v, l in zip(yt_pos, yt_lab)],
                  "label": y_label, "categories": y_cats},
            "marks": marks, "texts": texts, "legend": legend, "colorbar": colorbar,
            "stats": stat_results, "fg": fg}


def data_at(geometry: Mapping[str, Any], x_mm: float, y_mm: float) -> Dict[str, Any]:
    """The data coordinates under a page point, from the geometry's axes and limits — for
    a hover readout. ``None`` values outside the axes."""
    ax, ay, aw, ah = geometry["axes_mm"]
    inside = ax <= x_mm <= ax + aw and ay <= y_mm <= ay + ah
    gx, gy = geometry["x"], geometry["y"]
    xs_ = Scale(gx["scale"] if gx["scale"] != "category" else "linear", *gx["lim"], ax, ax + aw)
    ys_ = Scale(gy["scale"] if gy["scale"] != "category" else "linear", *gy["lim"], ay + ah, ay)
    xv = xs_.from_mm(x_mm) if inside else None
    yv = ys_.from_mm(y_mm) if inside else None
    if inside and gx.get("categories"):
        k = int(round(xv))
        xv = gx["categories"][k] if 0 <= k < len(gx["categories"]) else None
    if inside and gy.get("categories"):
        k = int(round(yv))
        yv = gy["categories"][k] if 0 <= k < len(gy["categories"]) else None
    return {"inside": inside, "x": xv, "y": yv}


def nearest_mark(geometry: Mapping[str, Any], x_mm: float, y_mm: float, *,
                 within_mm: float = 2.0) -> Optional[Dict[str, Any]]:
    """The closest data marker to a page point within ``within_mm`` — ``{"series",
    "index", "distance_mm"}`` — or ``None``. Markers only; lines are not picked."""
    best: Optional[Dict[str, Any]] = None
    for mk in geometry.get("marks", []):
        if mk.get("kind") != "markers" or mk.get("role") not in ("series", "means"):
            continue
        for k, (px_, py_) in enumerate(mk.get("pts_mm", [])):
            d = math.hypot(px_ - x_mm, py_ - y_mm)
            if d <= within_mm and (best is None or d < best["distance_mm"]):
                best = {"series": mk.get("series"), "index": k, "distance_mm": d,
                        "role": mk.get("role")}
    return best
