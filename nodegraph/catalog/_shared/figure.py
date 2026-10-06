"""figure — the shared core of the ``plot.*`` nodes and ``io.write_figure`` (V4.00 step 7).

A plot node renders a **figure spec** into a **Picture dataset**:

* a FIGURE SPEC is a plain JSON-able dict — what to draw and how::

      {"kind": "xy",
       "series": [{"label": str, "x": [..], "y": [..], "err": [..] | None, "n": int}],
       "draw": {"mode": "line" | "scatter" | "line_markers", "error_style": "bars" | "band"},
       "axes": {"title", "x_label", "y_label", "log_x", "log_y",
                "x_range": [lo | None, hi | None], "y_range": [lo | None, hi | None]},
       "style": {"width_mm", "height_mm", "dpi", "font", "font_pt", "line_width",
                 "marker_size", "grid", "legend", "palette", "dark"}}

  It rides in the picture's metadata (``figure_spec``), so ``io.write_figure`` can render the
  SAME figure again at another resolution or as a vector file;
* a PICTURE DATASET is one RGB image — axes ``(m, t, z, c, y, x) = (1, 1, 1, 3, H, W)``,
  uint8 — with FRESH metadata: ``bit_depth = 8``, channels R/G/B with their colours and
  ``picture = "rgb"`` (the Viewer then shows it in true colour on a fixed 0–255 window) and
  NO calibration, so no scale bar or physical readout is ever drawn on a chart.

matplotlib is imported lazily, through its object-oriented API only (``Figure`` + the Agg
canvas) — never ``pyplot``, whose global state is not thread-safe — and a style is applied as
artist properties, never through ``rcParams``. The one exception is set ONCE, at first use,
under a lock: SVG text stays text and PDF fonts embed as TrueType, so an exported figure's
labels remain editable in Illustrator or Inkscape.

``figure_pixels`` is the ONE place the picture's size comes from — the edit-time prediction
(``metadata.figure_frame``) and the render both call it, which is what keeps the envelope and
the payload in lockstep (INV-04).
"""
from __future__ import annotations

import importlib.util
import json
from collections import OrderedDict
from dataclasses import replace
import logging
import math
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.provider import ArrayProvider, TileProvider
from nodegraph.registry import InBool, InFloat, InInt, InString, Mode

# ── styles ─────────────────────────────────────────────────────────────────────
#: the presets of the ``style`` mode — every number a ``detail = custom`` socket overrides
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "paper": {"width_mm": 85.0, "height_mm": 65.0, "dpi": 300, "font": "sans",
              "font_pt": 8.0, "line_width": 1.0, "marker_size": 3.0, "grid": False,
              "legend": True, "dark": False},
    "talk": {"width_mm": 160.0, "height_mm": 100.0, "dpi": 150, "font": "sans",
             "font_pt": 14.0, "line_width": 2.0, "marker_size": 6.0, "grid": True,
             "legend": True, "dark": False},
    "poster": {"width_mm": 250.0, "height_mm": 180.0, "dpi": 150, "font": "sans",
               "font_pt": 20.0, "line_width": 3.0, "marker_size": 8.0, "grid": True,
               "legend": True, "dark": False},
    "dark": {"width_mm": 160.0, "height_mm": 100.0, "dpi": 150, "font": "sans",
             "font_pt": 12.0, "line_width": 1.8, "marker_size": 5.0, "grid": True,
             "legend": True, "dark": True},
}

#: series colours. Okabe & Ito's eight stay distinguishable under the common colour-vision
#: deficiencies; tab10 is matplotlib's default cycle; greys print in black and white.
PALETTES: Dict[str, List[str]] = {
    "colorblind": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
                   "#F0E442", "#000000"],
    "tab10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "greys": ["#000000", "#555555", "#999999", "#cccccc"],
}

_FONTS = {"sans": "DejaVu Sans", "serif": "DejaVu Serif"}   # both ship with matplotlib
#: what one custom socket may hold — a figure is never smaller than a thumbnail or larger
#: than a poster, so a typo cannot ask for a gigapixel render
_LIMITS = {"width_mm": (20.0, 1200.0), "height_mm": (20.0, 1200.0), "dpi": (50, 1200),
           "font_pt": (3.0, 72.0), "line_width": (0.1, 20.0), "marker_size": (0.5, 40.0)}

MATPLOTLIB_MISSING = ("this node draws with matplotlib, which is not installed — "
                      "`pip install matplotlib` (it is in requirements.txt), then restart")


def require_matplotlib() -> None:
    """Refuse with an install hint, at pull time, when matplotlib is absent — the node still
    registers and shows on the palette without it."""
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(MATPLOTLIB_MISSING)


def _clamp(key: str, v: float) -> float:
    lo, hi = _LIMITS[key]
    return min(max(v, lo), hi)


def resolve_style(values: Mapping[str, Any], modes: Mapping[str, str]) -> Dict[str, Any]:
    """The style a plot draws with: the ``style`` preset; under ``detail = custom`` the
    custom fields instead — each one unset taking the PAPER preset's value, which is what its
    socket shows as its default — clamped to :data:`_LIMITS`, while the style still decides
    light or dark; plus the palette. TOTAL: a bad value falls back (the edit-time
    ``figure_frame`` calls this on every keystroke and must never raise)."""
    name = str(modes.get("style", "paper") or "paper")
    st = dict(STYLE_PRESETS.get(name, STYLE_PRESETS["paper"]))
    if str(modes.get("detail", "preset")) == "custom":
        base = STYLE_PRESETS["paper"]
        for key, cast in (("width_mm", float), ("height_mm", float), ("dpi", int),
                          ("font_pt", float), ("line_width", float), ("marker_size", float)):
            v = values.get(key)
            try:
                st[key] = (base[key] if v in (None, "")
                           else cast(_clamp(key, float(v))))
            except (TypeError, ValueError):
                st[key] = base[key]
        for key in ("grid", "legend"):
            v = values.get(key)
            st[key] = v if isinstance(v, bool) else base[key]
        st["font"] = str(modes.get("font", st["font"]) or st["font"])
    st["palette"] = str(modes.get("palette", "colorblind") or "colorblind")
    return st


def style_from_ctx(ctx) -> Dict[str, Any]:
    """:func:`resolve_style` on a compute's own params — read here by their literal names, so
    the socket-contract gate sees every style socket read."""
    values = {"width_mm": ctx.params.get("width_mm"), "height_mm": ctx.params.get("height_mm"),
              "dpi": ctx.params.get("dpi"), "font_pt": ctx.params.get("font_pt"),
              "line_width": ctx.params.get("line_width"),
              "marker_size": ctx.params.get("marker_size"),
              "grid": ctx.params.get("grid"), "legend": ctx.params.get("legend")}
    return resolve_style(values, ctx.params.get("__modes__", {}) or {})


def figure_pixels(width_mm: float, height_mm: float, dpi: float) -> Tuple[int, int]:
    """``(height, width)`` in pixels of a figure of that size and resolution."""
    w = max(16, int(round(float(width_mm) / 25.4 * float(dpi))))
    h = max(16, int(round(float(height_mm) / 25.4 * float(dpi))))
    return h, w


def picture_pixels(values: Mapping[str, Any], modes: Mapping[str, str]) -> Tuple[int, int]:
    st = resolve_style(values, modes)
    return figure_pixels(st["width_mm"], st["height_mm"], st["dpi"])


# ── the sockets and modes every plot node shares ──────────────────────────────────
_CUSTOM = {"detail": frozenset({"custom"})}

_STYLE_DOCS = {
    "paper": "A one-column journal figure: 85 x 65 mm at 300 dpi, 8 pt text, thin lines and "
             "small markers, no grid, on white. Sized to drop into a manuscript at 100 %.",
    "talk": "A slide: 160 x 100 mm at 150 dpi with 14 pt text and heavier lines, plus a "
            "light grid, so the chart reads from the back of a room on a projector.",
    "poster": "A poster panel: 250 x 180 mm at 150 dpi with 20 pt text and the heaviest "
              "lines, readable from a couple of metres; the largest picture of the four.",
    "dark": "The talk layout on a dark background with light text and grid — for slides and "
            "screens with a dark theme. Prints badly; pick another style for paper.",
}
_PALETTE_DOCS = {
    "colorblind": "Okabe & Ito's eight colours, chosen to stay distinguishable under the "
                  "common colour-vision deficiencies. The safe default for any audience.",
    "tab10": "matplotlib's ten-colour default cycle: familiar from other figures, but two "
             "of its pairs merge for red-green colour-blind readers.",
    "viridis": "Five steps of the perceptually uniform viridis map, dark purple to yellow — "
               "for groups that have an ORDER (doses, timepoints), so the colour reads as one.",
    "greys": "Four greys from black to light grey, for a figure printed in black and white; "
             "past four groups the shades repeat, so pair it with few groups.",
}


def style_modes(*, palette: bool = True) -> list:
    """The ``style`` / ``detail`` / ``palette`` / ``font`` modes of a plot node — without
    ``palette`` for a plot that colours by a colour map instead (a heatmap)."""
    modes = _style_modes()
    return modes if palette else [m for m in modes if m.name != "palette"]


def _style_modes() -> list:
    return [
        Mode("style", list(STYLE_PRESETS), default="paper", label="Style",
             description="The figure's size, resolution, type sizes and line weights as one "
                         "preset for where it is going. Each re-renders the picture; switch "
                         "Detail to custom to set any of them by hand.",
             choice_docs=_STYLE_DOCS),
        Mode("detail", ["preset", "custom"], default="preset", label="Detail",
             description="Whether the style preset decides every size and weight, or the "
                         "custom fields below do.",
             choice_docs={
                 "preset": "Use the chosen style exactly — size, dpi, type, lines, grid and "
                           "legend — and hide the fields that would override it.",
                 "custom": "Show width, height, dpi, type size, line width, marker size, "
                           "font, grid and legend, and draw with them. They start at the "
                           "paper preset's values; the style still picks light or dark.",
             }),
        Mode("palette", list(PALETTES), default="colorblind", label="Palette",
             description="The colours the series (the groups) are drawn in, in order.",
             choice_docs=_PALETTE_DOCS),
        Mode("font", ["sans", "serif"], default="sans", label="Font",
             available_in=_CUSTOM,
             description="The typeface of every label. Both ship with the app, so a figure "
                         "renders identically on every machine.",
             choice_docs={
                 "sans": "DejaVu Sans, a plain sans-serif like most journals' figure text "
                         "and every slide template — the default.",
                 "serif": "DejaVu Serif, for a document set in a serif face whose figures "
                          "should match its body text.",
             }),
    ]


def style_sockets(*, axes: bool = True) -> list:
    """The labels, ranges, scales and custom style fields of a plot node — without the
    ranges and log axes when ``axes`` is False (a heatmap has neither)."""
    socks = [
        InString("title", "Title", field=False, default="",
                 description="The text above the chart. Blank draws no title — the usual "
                             "choice for a manuscript, whose caption carries it."),
        InString("x_label", "X label", field=False, default="",
                 description="The horizontal axis label. Blank uses the x column's name; "
                             "add the unit here, e.g. 'time (min)'."),
        InString("y_label", "Y label", field=False, default="",
                 description="The vertical axis label. Blank uses the y column's name; add "
                             "the unit here, e.g. 'area (µm²)'."),
        InString("x_range", "X range", field=False, default="",
                 description="Fix the horizontal axis as 'low, high' (e.g. '0, 120'). Blank "
                             "lets the data decide, and either side may be left blank "
                             "('0,' fixes only the low end). Values outside are not drawn."),
        InString("y_range", "Y range", field=False, default="",
                 description="Fix the vertical axis as 'low, high' (e.g. '0, 500'). Blank "
                             "lets the data decide; '0,' starts the axis at zero, which keeps "
                             "a bar of differences honest."),
        InBool("log_x", "Log x", field=False, default=False,
               description="Draw the horizontal axis on a log scale. Rows whose x is zero or "
                           "negative cannot be placed on it and are left out."),
        InBool("log_y", "Log y", field=False, default=False,
               description="Draw the vertical axis on a log scale — for values spanning "
                           "orders of magnitude. Rows whose y is zero or negative are left "
                           "out."),
        InFloat("width_mm", "Width", unit="mm", field=False, default=85.0,
                available_in=_CUSTOM,
                description="The figure's printed width. Together with dpi it sets the "
                            "picture's pixel size; text and lines keep their point sizes, so "
                            "a wider figure gives the data more room, not bigger type."),
        InFloat("height_mm", "Height", unit="mm", field=False, default=65.0,
                available_in=_CUSTOM,
                description="The figure's printed height (see Width)."),
        InInt("dpi", "Resolution", unit="dpi", field=False, default=300,
              available_in=_CUSTOM,
              description="Pixels per inch of the picture. 300 is print quality; 150 is "
                          "plenty on screen and four times fewer pixels. Export Figure can "
                          "write another resolution without changing this."),
        InFloat("font_pt", "Text size", unit="pt", field=False, default=8.0,
                available_in=_CUSTOM,
                description="The axis-label size in points; tick labels are a little "
                            "smaller and the title a little larger."),
        InFloat("line_width", "Line width", unit="pt", field=False, default=1.0,
                available_in=_CUSTOM,
                description="The data lines' width in points; axes, ticks and error bars "
                            "scale with it."),
        InFloat("marker_size", "Marker size", unit="pt", field=False, default=3.0,
                available_in=_CUSTOM,
                description="The diameter of a point's marker in points."),
        InBool("grid", "Grid", field=False, default=False, available_in=_CUSTOM,
               description="Draw light grid lines behind the data at the major ticks."),
        InBool("legend", "Legend", field=False, default=True, available_in=_CUSTOM,
               description="Name each group in a legend. Only drawn when there are named "
                           "groups (a Group by column is set)."),
    ]
    if not axes:
        # a heatmap: no data axes to range or log; Grid outlines its cells and Legend is
        # its colour bar, so they say so
        socks = [s for s in socks
                 if s.name not in ("x_range", "y_range", "log_x", "log_y", "grid", "legend")]
        socks += [
            InBool("grid", "Grid", field=False, default=False, available_in=_CUSTOM,
                   description="Outline every cell of a table heatmap with a thin line in the "
                               "background colour, so neighbouring cells of a similar colour "
                               "stay apart. A Voxel plane has no cells to outline."),
            InBool("legend", "Legend", field=False, default=True, available_in=_CUSTOM,
                   description="Show the colour bar that says which value each colour "
                               "stands for. Off leaves the scale to a caption."),
        ]
    return socks


def parse_range(text: Any, *, socket: str) -> List[Optional[float]]:
    """``'lo, hi'`` → ``[lo, hi]`` with a blank side as ``None``; blank → ``[None, None]``."""
    s = str(text or "").strip()
    if not s:
        return [None, None]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{socket} must be 'low, high' (either side may be blank), "
                         f"not {s!r}")
    out: List[Optional[float]] = []
    for p in parts:
        if not p:
            out.append(None)
            continue
        try:
            out.append(float(p))
        except ValueError:
            raise ValueError(f"{socket}: {p!r} is not a number ({s!r})") from None
    return out


# ── reading a structure table ────────────────────────────────────────────────────
def table_columns(ds: Dataset, domain: Domain, layer: str) -> Dict[str, np.ndarray]:
    """The columns of the ``domain`` table ``layer`` on ``ds``, as numpy arrays."""
    return {a.name: np.asarray(a.values) for a in ds.layers_on(domain) if a.layer == layer}


def column(cols: Mapping[str, np.ndarray], name: str, *, node: str, socket: str,
           layer: str) -> np.ndarray:
    """``cols[name]``, or a refusal naming the columns the table does carry."""
    if name in cols:
        return cols[name]
    raise ValueError(
        f"{node}: the {layer!r} table has no column {name!r} for `{socket}` — it carries "
        f"{sorted(cols)}. Measure it first (analysis.measure adds intensity and shape "
        f"columns), or pick one of these from the {socket} dropdown.")


# ── columns → series ──────────────────────────────────────────────────────────
#: the legend entry of the rows whose group value is missing (NaN / None)
MISSING_GROUP = "(missing)"
_MISSING_KEY = object()
#: past this many named series a legend is noise (a track id per series): it is left off
LEGEND_MAX = 12


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return not math.isfinite(float(v))
    return False


def _series_key(v: Any) -> Any:
    """``v`` as a dictionary key: one key for every missing value, numpy scalars as Python."""
    if _is_missing(v):
        return _MISSING_KEY
    return v.item() if isinstance(v, np.generic) else v


def _series_order(key: Any):
    return (2, 0.0, "") if key is _MISSING_KEY else _sort_key(key)


def _series_label(key: Any) -> str:
    return MISSING_GROUP if key is _MISSING_KEY else _label(key)


def group_parts(group, keep: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """``[(label, row indices)]`` per distinct value of ``group`` among the kept rows, in ONE
    pass over the rows, in a stable order — numbers ascending, then text, then the rows with
    no value (NaN / None), which form ONE group labelled :data:`MISSING_GROUP`. Every value
    keeps its colour from run to run."""
    g = np.asarray(group, dtype=object).tolist()
    buckets: Dict[Any, List[int]] = {}
    for i, (v, k) in enumerate(zip(g, np.asarray(keep, dtype=bool).tolist())):
        if k:
            buckets.setdefault(_series_key(v), []).append(i)
    return [(_series_label(key), np.asarray(buckets[key], dtype=np.int64))
            for key in sorted(buckets, key=_series_order)]


def _label(v) -> str:
    """A group value as a legend entry: 1.0 → "1", a string as itself."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _sort_key(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def xy_series(x, y, group=None, error: str = "none", *, log_x: bool = False,
              log_y: bool = False) -> List[Dict[str, Any]]:
    """The series of an XY plot, one per value of ``group`` (one in all when ``None``).

    ``error = "none"`` keeps every row as a point, ordered by x so a line reads left to right.
    Otherwise the rows sharing an x are summarised: their mean, with ``sd``, ``sem`` or the
    95 % confidence half-width (Student's t, n - 1 degrees of freedom) as the error; a lone
    row has no spread and gets 0. Rows whose x or y is not finite — or not positive on a log
    axis — are left out."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if log_x:
        keep &= x > 0
    if log_y:
        keep &= y > 0
    if group is None:
        parts = [("", np.flatnonzero(keep))]
    else:
        parts = group_parts(group, keep)
    out: List[Dict[str, Any]] = []
    for label, m in parts:
        xs, ys = x[m], y[m]
        if error == "none":
            order = np.argsort(xs, kind="stable")
            out.append({"label": label, "x": xs[order].tolist(), "y": ys[order].tolist(),
                        "err": None, "n": int(m.size)})
            continue
        ux = np.unique(xs)
        mean: List[float] = []
        err: List[float] = []
        for v in ux:
            vals = ys[xs == v]
            n = int(vals.size)
            mean.append(float(vals.mean()))
            if n < 2:
                err.append(0.0)
                continue
            sd = float(vals.std(ddof=1))
            if error == "sd":
                err.append(sd)
            elif error == "sem":
                err.append(sd / math.sqrt(n))
            else:                                    # ci95
                from scipy.stats import t as _student_t
                err.append(float(_student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n))
        out.append({"label": label, "x": ux.tolist(), "y": mean, "err": err,
                    "n": int(m.size)})
    return out


def distribution_series(values, group=None) -> List[Dict[str, Any]]:
    """One series of the finite ``values`` per ``group`` value (one in all when ``None``)."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    if group is None:
        return [{"label": "", "values": v[keep].tolist(), "n": int(keep.sum())}]
    return [{"label": label, "values": v[idx].tolist(), "n": int(idx.size)}
            for label, idx in group_parts(group, keep)]


#: what a heatmap cell can summarise its rows with
HEAT_REDUCERS = ("mean", "median", "sum", "min", "max", "count", "std")


def heatmap_grid(rows, cols, values, reducer: str = "mean"
                 ) -> Tuple[List[str], List[str], List[List[Optional[float]]]]:
    """``(row labels, column labels, matrix)``: one cell per (row value, column value) — the
    ``reducer`` of the finite values that fall in it; an empty cell is ``None`` (drawn blank,
    not as zero)."""
    r = np.asarray(rows, dtype=object).tolist()
    c = np.asarray(cols, dtype=object).tolist()
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    r = [_series_key(x) for x in r]
    c = [_series_key(x) for x in c]
    rk = sorted({x for x, k in zip(r, keep.tolist()) if k}, key=_series_order)
    ck = sorted({x for x, k in zip(c, keep.tolist()) if k}, key=_series_order)
    ri = {x: i for i, x in enumerate(rk)}
    ci = {x: i for i, x in enumerate(ck)}
    cells: Dict[Tuple[int, int], List[float]] = {}
    for x, y, val, k in zip(r, c, v.tolist(), keep.tolist()):
        if k:
            cells.setdefault((ri[x], ci[y]), []).append(val)
    fns = {"mean": np.mean, "median": np.median, "sum": np.sum, "min": np.min,
           "max": np.max, "count": len,
           "std": lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0}
    fn = fns.get(reducer, np.mean)
    mat: List[List[Optional[float]]] = [[None] * len(ck) for _ in rk]
    for (i, j), vals in cells.items():
        mat[i][j] = float(fn(np.asarray(vals)))
    return [_series_label(x) for x in rk], [_series_label(x) for x in ck], mat


def frame_times(n_frames: int, *, frame_time_jd: Any = None, dt_s: Any = None,
                mode: str = "elapsed") -> Tuple[List[float], str, str]:
    """``(x value per frame, axis label, kind)`` for a time axis of ``mode`` elapsed | clock |
    frame. ``kind`` is ``"clock"`` when the values are wall-clock dates, else ``""``.

    elapsed — seconds since the first frame from the file's own per-frame clock
    (``frame_time_jd``), else ``t x dt_s``, else the frame index (the label says so); the
    unit is picked ONCE from the span (s / min / h). clock — each frame's wall-clock time
    (matplotlib date numbers), when the file has a per-frame clock; otherwise elapsed.
    frame — the frame index."""
    n = max(0, int(n_frames))
    idx = list(range(n))
    jd = None
    if frame_time_jd is not None:
        try:
            vals = [float(v) for v in list(frame_time_jd)[:n]]
            if len(vals) == n and n and all(math.isfinite(v) for v in vals):
                jd = vals
        except (TypeError, ValueError):
            jd = None
    if mode == "frame":
        return [float(i) for i in idx], "frame", ""
    if mode == "clock" and jd is not None:
        return [v - 2440587.5 for v in jd], "clock time", "clock"   # JD of 1970-01-01
    try:
        dt = float(dt_s) if dt_s not in (None, "") else 0.0
    except (TypeError, ValueError):
        dt = 0.0
    if jd is not None:
        secs = [(v - jd[0]) * 86400.0 for v in jd]
    elif dt > 0:
        secs = [i * dt for i in idx]
    else:
        return [float(i) for i in idx], "frame (no frame times in the file)", ""
    span = (max(secs) - min(secs)) if secs else 0.0
    if span < 90.0:
        return secs, "time (s)", ""
    if span < 5400.0:
        return [s / 60.0 for s in secs], "time (min)", ""
    return [s / 3600.0 for s in secs], "time (h)", ""


def padded_range(values: Sequence[float]) -> List[Optional[float]]:
    """``[lo, hi]`` of the finite ``values`` with 5 % padding — the fixed axis every frame of a
    per-frame figure shares, so the chart does not jump as the frames play."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return [None, None]
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * 0.05 or (abs(hi) * 0.05 or 1.0)
    return [lo - pad, hi + pad]


def frame_clock(md: Mapping[str, Any], frames: int,
                dt_s: Any = None) -> Tuple[Dict[str, Any], List[str]]:
    """The input's per-frame clock for a PER-FRAME picture: the metadata the Viewer's
    timestamp reads (``frame_time_jd`` / ``frame_datetime`` of those frames, ``dt_s``), and one
    title stamp per frame — the elapsed time when there is a clock, else ``frame k/T``."""
    from nodegraph.placement import elapsed_text
    n = max(1, int(frames))
    extra: Dict[str, Any] = {}
    jd, fd = (md or {}).get("frame_time_jd"), (md or {}).get("frame_datetime")
    if isinstance(jd, (list, tuple)) and len(jd) >= n:
        extra["frame_time_jd"] = list(jd)[:n]
    if isinstance(fd, (list, tuple)) and len(fd) >= n:
        extra["frame_datetime"] = list(fd)[:n]
    try:
        dt = float(dt_s) if dt_s not in (None, "") else 0.0
    except (TypeError, ValueError):
        dt = 0.0
    if dt > 0:
        extra["dt_s"] = dt
    secs: Optional[List[float]] = None
    if "frame_time_jd" in extra:
        try:
            v = [float(x) for x in extra["frame_time_jd"]]
            secs = [(x - v[0]) * 86400.0 for x in v]
        except (TypeError, ValueError):
            secs = None
    if secs is None and dt > 0:
        secs = [i * dt for i in range(n)]
    if secs is None:
        return extra, [f"frame {i + 1}/{n}" for i in range(n)]
    span = (max(secs) - min(secs)) if secs else 0.0
    return extra, [elapsed_text(s, span) for s in secs]


def json_value(v: Any) -> Any:
    """``v`` as a plain JSON value — a figure spec rides in metadata and in files."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def axis_range(values: Any, *, log: bool = False) -> List[Optional[float]]:
    """The fixed axis every frame of a per-frame figure shares: the finite ``values``' range
    with 5 % padding — multiplicative, and over the positive values only, on a log axis."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
        if not v.size:
            return [None, None]
        return [float(v.min()) / 1.1, float(v.max()) * 1.1]
    return padded_range(v)


#: the ``per`` mode every time-aware plot shares
PER_DOCS = {
    "all": "One figure of every row, drawn once — the whole experiment in a single picture.",
    "frame": "One figure per frame of the input, drawn when the frame is shown, on axes "
             "every frame shares — scrub or play it in the Viewer, or export it as a movie.",
}


def frame_spec(spec: Mapping[str, Any], t: int) -> Dict[str, Any]:
    """The figure of frame ``t`` of a PER-FRAME spec (``per = "frame"``): the rows of that frame
    (``xy``), or the course up to it with a cursor (``timeseries``), on the axes every frame
    shares. A spec drawn whole is returned as it is."""
    if spec.get("per") != "frame":
        return dict(spec)
    rows = spec.get("rows") or {}
    tt = np.asarray(rows.get("t", []), dtype=float)
    sel = (tt <= t) if spec.get("kind") == "timeseries" else (tt == t)
    pick = lambda key: (np.asarray(rows[key], dtype=object)[sel]       # noqa: E731
                        if rows.get(key) is not None else None)
    x = np.asarray(rows.get("x", []), dtype=float)[sel]
    y = np.asarray(rows.get("y", []), dtype=float)[sel]
    a = dict(spec.get("axes") or {})
    out = dict(spec)
    out["series"] = xy_series(x, y, pick("g"), str(spec.get("error", "none")),
                              log_x=bool(a.get("log_x")), log_y=bool(a.get("log_y")))
    if rows.get("g") is not None:
        g_all = rows["g"]
        order = {label: i for i, (label, _idx) in
                 enumerate(group_parts(g_all, np.ones(len(g_all), dtype=bool)))}
        for s in out["series"]:
            s["color_index"] = order.get(s["label"], 0)
    out["per"] = "all"
    out.pop("rows", None)
    times = spec.get("frame_x") or []
    if spec.get("kind") == "timeseries" and 0 <= t < len(times):
        out["cursor_x"] = times[t]
    label = (spec.get("frame_labels") or [])
    stamp = label[t] if 0 <= t < len(label) else f"frame {t + 1}"
    a["title"] = (f"{a['title']} — {stamp}" if a.get("title") else stamp)
    out["axes"] = a
    return out


# ── drawing ─────────────────────────────────────────────────────────────────────
_ONCE = threading.Lock()
_READY = False
#: ONE render at a time, process-wide: matplotlib's text layout (mathtext, used for a log
#: axis's tick labels) keeps shared parser state, and two figures laid out at once on two
#: threads fail with a ParseException. A figure takes a fraction of a second; the lock costs
#: nothing a user would see.
_RENDER_LOCK = threading.RLock()
#: the largest picture a render may make — a typo in a size or resolution (6000 dpi for 600)
#: is refused rather than allowed to ask for gigabytes
MAX_PIXELS = 100_000_000


def check_pixels(h: int, w: int) -> None:
    """Refuse a figure past :data:`MAX_PIXELS` — before anything is allocated or drawn."""
    if int(h) * int(w) > MAX_PIXELS:
        raise ValueError(f"the figure would be {w} x {h} px ({h * w / 1e6:.0f} Mpx) — more "
                         f"than {MAX_PIXELS // 1_000_000} Mpx; lower its size or resolution")


#: the drawing code's identity: a live reload of this module re-executes it, so a per-frame
#: figure built after the reload is a different provider (and memo entry) from the one
#: built before — its frames are never a mix of old and new drawing code
_DRAW_CODE = ""
try:
    import hashlib as _hashlib
    with open(__file__, "rb") as _fh:
        _DRAW_CODE = _hashlib.blake2b(_fh.read(), digest_size=8).hexdigest()
except Exception:                                       # noqa: BLE001 — frozen / no source
    _DRAW_CODE = "unknown"


def _matplotlib():
    """matplotlib's Figure and Agg canvas, imported lazily and set up once for editable
    exported text."""
    global _READY
    require_matplotlib()
    with _ONCE:
        if not _READY:
            logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
            import matplotlib
            matplotlib.rcParams["svg.fonttype"] = "none"   # SVG text stays text
            matplotlib.rcParams["pdf.fonttype"] = 42       # PDF fonts embed as TrueType
            _READY = True
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    return Figure, FigureCanvasAgg


def _ink(style: Mapping[str, Any]) -> Tuple[str, str, str]:
    """(background, foreground, grid) colours of a style."""
    if style.get("dark"):
        return "#1b1d21", "#e8e8e8", "#3a3d44"
    return "#ffffff", "#111111", "#d9d9d9"


def draw(spec: Mapping[str, Any]):
    """The matplotlib ``Figure`` of ``spec`` (laid out, not yet rendered) and its ``(h, w)``."""
    Figure, FigureCanvasAgg = _matplotlib()
    style = dict(STYLE_PRESETS["paper"])
    style.update(spec.get("style") or {})
    dpi = float(style["dpi"])
    h, w = figure_pixels(style["width_mm"], style["height_mm"], dpi)
    check_pixels(h, w)
    # +0.5 px: Agg truncates figsize*dpi, so an exact W/dpi can land on W-1 by a rounding ulp
    fig = Figure(figsize=((w + 0.5) / dpi, (h + 0.5) / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    bg, fg, grid = _ink(style)
    fam = _FONTS.get(str(style.get("font", "sans")), "DejaVu Sans")
    pt = float(style["font_pt"])
    lw, ms = float(style["line_width"]), float(style["marker_size"])
    fig.patch.set_facecolor(bg)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(bg)
    colors = PALETTES.get(str(style.get("palette", "colorblind")), PALETTES["colorblind"])
    kind = str(spec.get("kind", "xy"))
    series = list(spec.get("series") or [])
    if kind == "distribution":
        _draw_distribution(ax, spec, colors, fg, lw, ms)
    elif kind == "heatmap":
        _draw_heatmap(fig, ax, spec, fg, pt, fam, style=style, bg=bg, lw=lw)
    else:
        _draw_xy(ax, spec, colors, lw, ms)
    if spec.get("cursor_x") is not None:
        ax.axvline(float(spec["cursor_x"]), color=fg, lw=lw * 0.8, ls="--", alpha=0.7)
    _finish(fig, ax, spec, style, kind, series, fg, grid, fam, pt, lw)
    return fig, (h, w)


def _draw_xy(ax, spec, colors, lw, ms) -> None:
    dr = spec.get("draw") or {}
    mode = str(dr.get("mode", "line"))
    band = str(dr.get("error_style", "bars")) == "band"
    for i, s in enumerate(list(spec.get("series") or [])):
        col = colors[int(s.get("color_index", i)) % len(colors)]
        x = np.asarray(s.get("x", []), dtype=float)
        y = np.asarray(s.get("y", []), dtype=float)
        label = str(s.get("label", "")) or None
        if mode == "scatter":
            ax.scatter(x, y, s=ms ** 2, color=col, label=label, linewidths=0)
        else:
            ax.plot(x, y, color=col, lw=lw, label=label,
                    marker="o" if mode == "line_markers" else None, ms=ms)
        err = s.get("err")
        if err is not None and len(err) == len(x):
            e = np.asarray(err, dtype=float)
            if band:
                ax.fill_between(x, y - e, y + e, color=col, alpha=0.25, linewidth=0)
            else:
                ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col, elinewidth=lw * 0.8,
                            capsize=ms * 0.8)


def _hist_edges(allv: np.ndarray, n: int, axes: Mapping[str, Any]):
    """Histogram bin edges: ``n`` bins over the fixed X range when one is set (else the data's
    range), spaced geometrically on a log x axis — so the bins asked for are the bins seen."""
    v = allv[np.isfinite(allv)]
    log = bool(axes.get("log_x"))
    if log:
        v = v[v > 0]
    xl, xh = (list(axes.get("x_range") or [None, None]) + [None, None])[:2]
    lo = float(xl) if xl is not None else (float(v.min()) if v.size else None)
    hi = float(xh) if xh is not None else (float(v.max()) if v.size else None)
    if log and lo is not None and lo <= 0:
        lo = float(v.min()) if v.size else None
    if lo is None or hi is None or not hi > lo:
        return np.histogram_bin_edges(allv, bins=n) if allv.size else 10
    if log:
        return np.geomspace(lo, hi, n + 1)
    return np.linspace(lo, hi, n + 1)


def _draw_distribution(ax, spec, colors, fg, lw, ms) -> None:
    dr = spec.get("draw") or {}
    kind = str(dr.get("mode", "histogram"))
    series = [s for s in (spec.get("series") or []) if s.get("values")]
    multi = len(series) > 1
    allv = (np.concatenate([np.asarray(s["values"], float) for s in series])
            if series else np.zeros(0))
    if kind == "histogram":
        edges = _hist_edges(allv, max(1, int(dr.get("bins", 30))), spec.get("axes") or {})
        norm = str(dr.get("normalize", "count"))
        for i, s in enumerate(series):
            v = np.asarray(s["values"], float)
            col = colors[i % len(colors)]
            ax.hist(v, bins=edges, density=(norm == "density"),
                    weights=(np.full(v.size, 100.0 / v.size) if norm == "percent" else None),
                    histtype="stepfilled", alpha=0.45 if multi else 0.85, color=col,
                    edgecolor=col, linewidth=lw * 0.6, label=s.get("label") or None)
    elif kind == "kde":
        if allv.size:
            from scipy.stats import gaussian_kde
            lo, hi = float(allv.min()), float(allv.max())
            pad = (hi - lo) * 0.1 or 1.0
            grid = np.linspace(lo - pad, hi + pad, 256)
            for i, s in enumerate(series):
                v = np.asarray(s["values"], float)
                col = colors[i % len(colors)]
                if v.size < 2 or float(np.ptp(v)) == 0.0:
                    ax.axvline(float(v.mean()), color=col, lw=lw,
                               label=s.get("label") or None)
                    continue
                ax.plot(grid, gaussian_kde(v)(grid), color=col, lw=lw,
                        label=s.get("label") or None)
    elif kind == "ecdf":
        for i, s in enumerate(series):
            v = np.sort(np.asarray(s["values"], float))
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                    color=colors[i % len(colors)], lw=lw, label=s.get("label") or None)
        ax.set_ylim(0.0, 1.02)
    else:                                                     # box | violin
        data = [np.asarray(s["values"], float) for s in series]
        pos = list(range(1, len(data) + 1))
        if data and kind == "box":
            bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                            medianprops={"color": fg, "linewidth": lw},
                            whiskerprops={"color": fg, "linewidth": lw * 0.8},
                            capprops={"color": fg, "linewidth": lw * 0.8},
                            flierprops={"markersize": ms, "markeredgecolor": fg})
            for i, b in enumerate(bp["boxes"]):
                b.set_facecolor(colors[i % len(colors)])
                b.set_alpha(0.7)
                b.set_edgecolor(fg)
        elif data:
            vp = ax.violinplot(data, positions=pos, showmedians=True, widths=0.8)
            for i, b in enumerate(vp["bodies"]):
                b.set_facecolor(colors[i % len(colors)])
                b.set_edgecolor(fg)
                b.set_alpha(0.7)
            for k in ("cmedians", "cmins", "cmaxes", "cbars"):
                if k in vp:
                    vp[k].set_color(fg)
                    vp[k].set_linewidth(lw * 0.8)
        if pos:
            ax.set_xticks(pos, [s.get("label") or "all" for s in series])


def _draw_heatmap(fig, ax, spec, fg, pt, fam, *, style=None, bg="#ffffff",
                  lw=1.0) -> None:
    dr = spec.get("draw") or {}
    mat = np.array([[np.nan if v is None else v for v in row]
                    for row in (spec.get("matrix") or [])], dtype=float)
    if mat.ndim != 2 or not mat.size:
        mat = np.full((1, 1), np.nan)
    lo, hi = (list(dr.get("color_range") or [None, None]) + [None, None])[:2]
    im = ax.imshow(mat, cmap=str(dr.get("colormap", "viridis")), aspect="auto",
                   interpolation="nearest", vmin=lo, vmax=hi, origin="upper")
    rl, cl = spec.get("row_labels") or [], spec.get("col_labels") or []
    for labels, setter in ((rl, ax.set_yticks), (cl, ax.set_xticks)):
        if labels:
            # every k-th value past 40, so the ticks always name the real row / column
            k = max(1, -(-len(labels) // 40))
            idx = list(range(0, len(labels), k))
            setter(idx, [labels[i] for i in idx])
    style = style or {}
    if style.get("grid") and str((spec.get("source") or {}).get("source")) != "voxel":
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1.0), minor=True)
        ax.grid(which="minor", color=bg, linewidth=lw * 0.8)
        ax.tick_params(which="minor", length=0)
    if style.get("legend", True):
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=pt * 0.85, colors=fg)
        cb.set_label(str(dr.get("value_label", "")), fontsize=pt, color=fg, family=fam)
        cb.outline.set_edgecolor(fg)


def _finish(fig, ax, spec, style, kind, series, fg, grid, fam, pt, lw) -> None:
    """Axes, labels, ticks, spines, grid and legend — the same on every kind of figure."""
    a = spec.get("axes") or {}
    if a.get("log_x"):
        ax.set_xscale("log")
    if a.get("log_y"):
        ax.set_yscale("log")
    xl, xh = (list(a.get("x_range") or [None, None]) + [None, None])[:2]
    yl, yh = (list(a.get("y_range") or [None, None]) + [None, None])[:2]
    if xl is not None or xh is not None:
        ax.set_xlim(left=xl, right=xh)
    if yl is not None or yh is not None:
        ax.set_ylim(bottom=yl, top=yh)
    if a.get("title"):
        ax.set_title(str(a["title"]), fontsize=pt * 1.15, color=fg, family=fam)
    ax.set_xlabel(str(a.get("x_label", "")), fontsize=pt, color=fg, family=fam)
    ax.set_ylabel(str(a.get("y_label", "")), fontsize=pt, color=fg, family=fam)
    ax.tick_params(labelsize=pt * 0.9, colors=fg, width=lw * 0.6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(fam)
    for side, sp in ax.spines.items():
        sp.set_color(fg)
        sp.set_linewidth(lw * 0.6)
        if side in ("top", "right"):
            sp.set_visible(False)
    if a.get("x_kind") == "clock":
        import matplotlib.dates as mdates
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    if style.get("grid") and kind != "heatmap":
        ax.grid(True, color=grid, linewidth=lw * 0.5)
        ax.set_axisbelow(True)
    if style.get("legend") and kind not in ("heatmap",) and \
            not (kind == "distribution" and str((spec.get("draw") or {}).get("mode"))
                 in ("box", "violin")) and \
            1 <= len([s for s in series if str(s.get("label", ""))]) <= LEGEND_MAX:
        leg = ax.legend(frameon=False, prop={"family": fam, "size": pt * 0.9})
        for txt in leg.get_texts():
            txt.set_color(fg)
    empty = (not spec.get("matrix")) if kind == "heatmap" else \
        not any(s.get("values") or s.get("x") for s in series)
    if empty:
        ax.text(0.5, 0.5, "no rows to plot", transform=ax.transAxes, ha="center",
                va="center", color=fg, fontsize=pt, family=fam)
    fig.tight_layout(pad=0.4)


def render_rgb(spec: Mapping[str, Any]) -> np.ndarray:
    """``spec`` rendered to an ``(H, W, 3)`` uint8 array, ``(H, W)`` = :func:`figure_pixels`."""
    with _RENDER_LOCK:
        fig, (h, w) = draw(frame_spec(spec, 0) if spec.get("per") == "frame" else spec)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    out = np.zeros((h, w, 3), np.uint8)
    hh, ww = min(h, rgb.shape[0]), min(w, rgb.shape[1])
    out[:hh, :ww] = rgb[:hh, :ww]
    return out


#: the formats io.write_figure writes, as matplotlib names them
SAVE_FORMATS = {"png": "png", "svg": "svg", "pdf": "pdf", "tiff": "tiff"}


def part_path(path: str) -> str:
    """Where a write goes before it is complete: ``.part`` BEFORE the extension, so anything
    that reads the format off the suffix still sees the right one."""
    stem, ext = os.path.splitext(path)
    return f"{stem}.part{ext}" if ext else f"{path}.part"


def render_file(spec: Mapping[str, Any], path: str, fmt: str,
                dpi: Optional[float] = None) -> str:
    """Write ``spec`` to ``path`` as ``fmt`` (png | svg | pdf | tiff), atomically: into
    :func:`part_path` first, renamed over ``path`` only once complete. A raster ``dpi`` draws
    the figure afresh at that resolution — same millimetres and points, so the same layout —
    which makes its pixel size exactly :func:`figure_pixels` at that dpi."""
    if dpi:
        spec = dict(spec)
        spec["style"] = dict(spec.get("style") or {}, dpi=int(dpi))
    part = part_path(path)
    with _RENDER_LOCK:
        fig, _hw = draw(spec)
        kw: Dict[str, Any] = {"format": SAVE_FORMATS[fmt], "facecolor": fig.get_facecolor()}
        if dpi:
            kw["dpi"] = float(int(dpi))
        try:
            fig.savefig(part, **kw)
            os.replace(part, path)
        finally:
            if os.path.exists(part):
                os.remove(part)
    return path


# ── the Picture dataset ───────────────────────────────────────────────────────────
#: the metadata every Picture carries (the edit-time envelope predicts the same keys)
PICTURE_METADATA: Dict[str, Any] = {
    "bit_depth": 8,
    "channel_names": ["R", "G", "B"],
    "channel_colors": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    "picture": "rgb",
}


def picture_dataset(rgb: np.ndarray, spec: Mapping[str, Any],
                    extra: Optional[Mapping[str, Any]] = None) -> Dataset:
    """A Picture dataset of ``rgb`` (``(H, W, 3)`` uint8), carrying ``spec`` — fresh metadata,
    no calibration (but ``extra``: a time plot's ``dt_s``), no structure layers."""
    arr = np.ascontiguousarray(np.moveaxis(np.asarray(rgb, dtype=np.uint8), -1, 0)
                               [None, None, None])                 # (1, 1, 1, 3, H, W)
    prov = ArrayProvider(arr)
    st = dict(spec.get("style") or {})
    md = dict(PICTURE_METADATA)
    md["channel_colors"] = [list(c) for c in PICTURE_METADATA["channel_colors"]]
    md["channel_names"] = list(PICTURE_METADATA["channel_names"])
    md.update(figure_spec=dict(spec), figure_dpi=st.get("dpi"),
              figure_size_mm=[st.get("width_mm"), st.get("height_mm")])
    md.update(dict(extra or {}))
    return Dataset(axes=prov.axes, metadata=md).with_image(prov)


def picture_axes(h: int, w: int, t: int = 1) -> AxisSizes:
    return AxisSizes(m=1, t=max(1, int(t)), z=1, c=3, y=int(h), x=int(w))


class FigureProvider(TileProvider):
    """A PER-FRAME figure series: frame ``t`` is drawn on its first read (from
    :func:`frame_spec`) and the last few are kept. One pyramid level, axes
    ``(1, T, 1, 3, H, W)``, uint8. Its identity is the spec's digest, so two pulls of the same
    figure are one memo entry and an edit re-keys it."""

    def __init__(self, spec: Mapping[str, Any], frames: int, h: int, w: int, *,
                 keep: int = 8) -> None:
        from nodegraph.memo import digest
        self.spec = dict(spec)
        self.axes = AxisSizes(m=1, t=max(1, int(frames)), z=1, c=3, y=int(h), x=int(w))
        self.levels = 1
        self.tile = 512
        self._keep = int(keep)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        self._key = digest("figure-provider", _DRAW_CODE,
                           json.dumps(self.spec, sort_keys=True, default=str))
        self._busy: Dict[int, threading.Event] = {}
        self.dtype = np.dtype(np.uint8)

    #: a read RUNS a figure render (it is not a decompress): the runner's prefetch and
    #: playback policies treat it like a computing provider, not like bytes on disk
    computes_on_read = True

    @property
    def nbytes(self) -> int:
        """What this provider can hold resident: its frame cache at its fullest. The memo's
        byte budget reads it, so superseded per-frame figures are evicted like any image."""
        return int(min(self._keep, self.axes.t) * self.axes.y * self.axes.x * 3)

    def frame(self, t: int) -> np.ndarray:
        """Frame ``t``, drawn ONCE however many readers ask at the same time (the R, G and B
        planes of one frame are read on different threads during playback)."""
        t = int(t)
        while True:
            with self._lock:
                got = self._cache.get(t)
                if got is not None:
                    self._cache.move_to_end(t)
                    return got
                ev = self._busy.get(t)
                if ev is None:
                    ev = self._busy[t] = threading.Event()
                    break                                # this reader draws it
            ev.wait()                                    # another reader is drawing it
        rgb = None
        try:
            rgb = render_rgb(frame_spec(self.spec, t))
        finally:
            with self._lock:
                if rgb is not None:
                    self._cache[t] = rgb
                    while len(self._cache) > self._keep:
                        self._cache.popitem(last=False)
                self._busy.pop(t, None)
            ev.set()                                     # a failed draw lets a waiter retry
        return rgb

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        return np.ascontiguousarray(self.frame(t)[y0:y1, x0:x1, c])

    def fingerprint(self) -> tuple:
        return ("figure", self._key, self.axes.t, self.axes.y, self.axes.x)


def picture_series(spec: Mapping[str, Any], frames: int, h: int, w: int,
                   extra: Optional[Mapping[str, Any]] = None) -> Dataset:
    """A PER-FRAME Picture: ``frames`` figures drawn on demand, with the picture metadata,
    ``spec`` and ``extra`` (the input's per-frame clock, so the Viewer's timestamp works)."""
    check_pixels(h, w)
    prov = FigureProvider(spec, frames, h, w)
    st = dict(spec.get("style") or {})
    md = dict(PICTURE_METADATA)
    md["channel_colors"] = [list(c) for c in PICTURE_METADATA["channel_colors"]]
    md["channel_names"] = list(PICTURE_METADATA["channel_names"])
    md.update(figure_spec=dict(spec), figure_dpi=st.get("dpi"),
              figure_size_mm=[st.get("width_mm"), st.get("height_mm")])
    md.update(dict(extra or {}))
    return Dataset(axes=prov.axes, metadata=md).with_image(prov)


# ── the edit-time envelope of a plot node ────────────────────────────────────────
def figure_frame(env, params: Mapping[str, Any], modes: Mapping[str, str]):
    return _figure_env(env, params, modes, clock=False)


def figure_frame_clocked(env, params: Mapping[str, Any], modes: Mapping[str, str]):
    """:func:`figure_frame` for a plot on a TIME axis (``plot.timeseries``): the input's frame
    interval ``dt_s`` stays on the picture's envelope, because the plot reads it
    (``ctx.calib`` reads this node's own envelope) and stamps it on the picture."""
    return _figure_env(env, params, modes, clock=True)


def _figure_env(env, params: Mapping[str, Any], modes: Mapping[str, str], *,
                clock: bool):
    """The ``meta_transform`` of a ``plot.*`` node: ONE RGB picture of the size its style
    gives — axes ``(1, 1, 1, 3, H, W)`` from :func:`picture_pixels`, exactly what the compute
    renders — with no calibration but ``bit_depth = 8``. Total: never raises (it runs on every
    keystroke); the node also declares ``fresh_output`` so no input domain, layer or column
    reaches the envelope.

    Lives here rather than in ``nodegraph.metadata``: the core module importing the catalog,
    even lazily, would put these helpers in every node's fingerprint closure."""
    try:
        h, w = picture_pixels(params or {}, modes or {})
    except Exception:                                   # noqa: BLE001 — total by contract
        h, w = picture_pixels({}, {})
    per = str((modes or {}).get("per", "all")) == "frame"
    md: Dict[str, Any] = {
        "bit_depth": 8,
        "channel_names": list(PICTURE_METADATA["channel_names"]),
        "channel_colors": [list(c) for c in PICTURE_METADATA["channel_colors"]]}
    src = env.metadata or {}
    if (per or clock) and src.get("dt_s") is not None:
        md["dt_s"] = src["dt_s"]                       # the input's clock, kept
    if per:
        # the per-frame times the payload carries (frame_clock): the Viewer's timestamp and
        # Export Movie's read them, and the envelope says what the payload holds
        from nodegraph.metadata import PER_TIME_KEYS
        t_in = int(getattr(env.axes, "t", 1) or 1)
        for key in PER_TIME_KEYS:
            vals = src.get(key)
            if isinstance(vals, (list, tuple)) and len(vals) >= t_in:
                md[key] = list(vals)[:t_in]
        # one figure per frame of the input: its T, and a T it cannot know yet stays
        # unknown rather than guessed
        t_in = getattr(env.axes, "t", 1) or 1
        unknown = frozenset({"t"}) & frozenset(getattr(env, "unknown_axes", ()) or ())
        return replace(env, axes=picture_axes(h, w, t_in), metadata=md,
                       unknown_axes=unknown, layers=())
    return replace(env, axes=picture_axes(h, w), metadata=md,
                   unknown_axes=frozenset(), layers=())


__all__ = ["STYLE_PRESETS", "PALETTES", "MATPLOTLIB_MISSING", "PICTURE_METADATA",
           "require_matplotlib", "resolve_style", "style_from_ctx", "figure_pixels",
           "picture_pixels", "style_modes", "style_sockets", "parse_range", "table_columns",
           "column", "xy_series", "draw", "render_rgb", "render_file", "part_path",
           "picture_dataset", "picture_axes", "SAVE_FORMATS", "figure_frame",
           "distribution_series", "heatmap_grid", "HEAT_REDUCERS", "frame_times",
           "padded_range", "frame_spec", "FigureProvider", "picture_series",
           "frame_clock", "PER_DOCS", "json_value", "axis_range",
           "figure_frame_clocked", "check_pixels", "group_parts", "MISSING_GROUP", "LEGEND_MAX",
           "MAX_PIXELS"]
