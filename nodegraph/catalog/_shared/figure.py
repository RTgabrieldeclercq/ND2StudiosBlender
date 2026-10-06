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
from dataclasses import replace
import logging
import math
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.provider import ArrayProvider
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


def style_modes() -> list:
    """The ``style`` / ``detail`` / ``palette`` / ``font`` modes of a plot node."""
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


def style_sockets() -> list:
    """The labels, ranges, scales and custom style fields of a plot node."""
    return [
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


def group_parts(group, keep: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """``[(label, row indices)]`` per distinct value of ``group`` among the kept rows, in ONE
    pass over the rows, in a stable order — numbers ascending, then text, then the rows with
    no value (NaN / None), which form ONE group labelled :data:`MISSING_GROUP`. Every value
    keeps its colour from run to run."""
    g = np.asarray(group, dtype=object).tolist()
    buckets: Dict[Any, List[int]] = {}
    for i, (v, k) in enumerate(zip(g, np.asarray(keep, dtype=bool).tolist())):
        if not k:
            continue
        key = _MISSING_KEY if _is_missing(v) else (v.item() if isinstance(v, np.generic)
                                                   else v)
        buckets.setdefault(key, []).append(i)

    def order(key):
        return (2, 0.0, "") if key is _MISSING_KEY else _sort_key(key)

    return [(MISSING_GROUP if key is _MISSING_KEY else _label(key),
             np.asarray(buckets[key], dtype=np.int64))
            for key in sorted(buckets, key=order)]

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
    if h * w > MAX_PIXELS:
        raise ValueError(f"the figure would be {w} x {h} px ({h * w / 1e6:.0f} Mpx) — more "
                         f"than {MAX_PIXELS // 1_000_000} Mpx; lower its size or resolution")
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
    dr = spec.get("draw") or {}
    mode = str(dr.get("mode", "line"))
    band = str(dr.get("error_style", "bars")) == "band"
    series = list(spec.get("series") or [])
    for i, s in enumerate(series):
        col = colors[i % len(colors)]
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
    if style.get("grid"):
        ax.grid(True, color=grid, linewidth=lw * 0.5)
        ax.set_axisbelow(True)
    named = [s for s in series if str(s.get("label", ""))]
    if style.get("legend") and 1 <= len(named) <= LEGEND_MAX:
        leg = ax.legend(frameon=False, prop={"family": fam, "size": pt * 0.9})
        for t in leg.get_texts():
            t.set_color(fg)
    if not series:
        ax.text(0.5, 0.5, "no rows to plot", transform=ax.transAxes, ha="center",
                va="center", color=fg, fontsize=pt, family=fam)
    fig.tight_layout(pad=0.4)
    return fig, (h, w)


def render_rgb(spec: Mapping[str, Any]) -> np.ndarray:
    """``spec`` rendered to an ``(H, W, 3)`` uint8 array, ``(H, W)`` = :func:`figure_pixels`."""
    with _RENDER_LOCK:
        fig, (h, w) = draw(spec)
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


def picture_dataset(rgb: np.ndarray, spec: Mapping[str, Any]) -> Dataset:
    """A Picture dataset of ``rgb`` (``(H, W, 3)`` uint8), carrying ``spec`` — fresh metadata,
    no calibration, no structure layers."""
    arr = np.ascontiguousarray(np.moveaxis(np.asarray(rgb, dtype=np.uint8), -1, 0)
                               [None, None, None])                 # (1, 1, 1, 3, H, W)
    prov = ArrayProvider(arr)
    st = dict(spec.get("style") or {})
    md = dict(PICTURE_METADATA)
    md["channel_colors"] = [list(c) for c in PICTURE_METADATA["channel_colors"]]
    md["channel_names"] = list(PICTURE_METADATA["channel_names"])
    md.update(figure_spec=dict(spec), figure_dpi=st.get("dpi"),
              figure_size_mm=[st.get("width_mm"), st.get("height_mm")])
    return Dataset(axes=prov.axes, metadata=md).with_image(prov)


def picture_axes(h: int, w: int) -> AxisSizes:
    return AxisSizes(m=1, t=1, z=1, c=3, y=int(h), x=int(w))


# ── the edit-time envelope of a plot node ────────────────────────────────────────
def figure_frame(env, params: Mapping[str, Any], modes: Mapping[str, str]):
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
    md = {"bit_depth": 8,
          "channel_names": list(PICTURE_METADATA["channel_names"]),
          "channel_colors": [list(c) for c in PICTURE_METADATA["channel_colors"]]}
    return replace(env, axes=picture_axes(h, w), metadata=md,
                   unknown_axes=frozenset(), layers=())


__all__ = ["STYLE_PRESETS", "PALETTES", "MATPLOTLIB_MISSING", "PICTURE_METADATA",
           "require_matplotlib", "resolve_style", "style_from_ctx", "figure_pixels",
           "picture_pixels", "style_modes", "style_sockets", "parse_range", "table_columns",
           "column", "xy_series", "draw", "render_rgb", "render_file", "part_path",
           "picture_dataset", "picture_axes", "SAVE_FORMATS", "figure_frame",
           "group_parts", "MISSING_GROUP", "LEGEND_MAX", "MAX_PIXELS"]
