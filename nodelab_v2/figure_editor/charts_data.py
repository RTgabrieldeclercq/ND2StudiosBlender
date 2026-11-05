"""A chart's data from a payload, and its geometry (V5.00 step 4).

:func:`chart_tables` tabulates the payload a chart reads (a source's Dataset, or a data
entry's terminal node's) with :func:`nodegraph.tables.all_tables`; :func:`resolve_chart`
picks the table, reads the columns the chart names and builds the series through
:mod:`nodegraph.figure.series` — the same helpers the ``plot.*`` nodes use — then asks
:func:`nodegraph.figure.charts.chart_geometry` for the marks in page millimetres, with
the item's rect and the page's style, and a ``measure`` that uses Qt's font metrics in
page units. Every point of an XY-like chart also carries its ROW (``m, t, id, y, x``)
so a hovered marker can light its object on an image.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtGui import QFontMetricsF

from nodegraph.domains import Domain
from nodegraph.figure import charts as C
from nodegraph.figure import series as SR
from nodegraph.figure.layout import PT_MM
from nodegraph.tables import all_tables
from nodegraph.units import format_unit, unit_of_column
from nodelab_v2.figure_editor.items import page_font

Tables = Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]]


def chart_tables(payload: Any) -> Tables:
    """Every table of a payload, keyed ``(domain, layer)``."""
    try:
        return all_tables(payload)
    except Exception:                      # noqa: BLE001 — a payload with nothing to tabulate
        return {}


def pick_table(tables: Tables, domain: str, name: str = "") -> Optional[Tuple[Tuple[str, Optional[str]], Dict[str, np.ndarray]]]:
    """The ``(key, columns)`` of table ``name`` on ``domain`` (the only one on the domain
    when ``name`` is blank), or ``None``."""
    cands = [(k, v) for k, v in tables.items() if k[0] == domain]
    if name:
        for k, v in cands:
            if str(k[1]) == name:
                return k, v
        # a summary lands on the label domain whatever it was reduced from: by name, any domain
        for k, v in tables.items():
            if str(k[1]) == name:
                return k, v
        return None
    return cands[0] if cands else None


def table_columns(tables: Tables, domain: str, name: str = "") -> List[str]:
    got = pick_table(tables, domain, name)
    if got is None:
        return []
    return sorted(got[1])


def _numeric(col: np.ndarray) -> bool:
    return np.asarray(col).dtype.kind in "biuf"


def _rows_payload(cols: Mapping[str, np.ndarray], idx: Optional[np.ndarray] = None
                  ) -> List[str]:
    """``"m,t,id,y,x"`` per row — the ride-along a hovered point is traced back by."""
    n = len(next(iter(cols.values()))) if cols else 0
    keys = ("m", "t", "id", "y", "x")
    arrs = [np.asarray(cols[k]) if k in cols else None for k in keys]
    out = []
    rng = range(n) if idx is None else idx.tolist()
    for i in rng:
        out.append(",".join("" if a is None else str(a[i].item() if hasattr(a[i], "item")
                                                          else a[i]) for a in arrs))
    return out


def measure_for(style: Mapping[str, Any]):
    """A ``measure(text, pt) -> (w_mm, h_mm)`` using Qt's metrics of the page font."""
    cache: Dict[Tuple[str, float], Tuple[float, float]] = {}

    def measure(text: str, pt: float) -> Tuple[float, float]:
        key = (str(text), float(pt))
        got = cache.get(key)
        if got is None:
            f = page_font(style, float(pt))
            fm = QFontMetricsF(f)
            # the font is in page units (10 per mm); mm = units / 10
            got = (fm.horizontalAdvance(str(text)) / 10.0, fm.height() / 10.0)
            cache[key] = got
        return got
    return measure


def resolve_series(chart: Mapping[str, Any], tables: Tables, md: Optional[Mapping[str, Any]]
                   = None) -> Tuple[Any, Dict[str, str], str]:
    """``(series, units, note)`` for ``chart`` from ``tables``: the series in the shape
    :func:`nodegraph.figure.charts.chart_geometry` expects for the chart's kind, the axis
    units (``{"x": label, "y": label}``) and a note when nothing could be drawn."""
    kind = str(chart.get("kind", "xy"))
    domain = str(chart.get("domain") or "label")
    got = pick_table(tables, domain, str(chart.get("table") or ""))
    if got is None:
        have = sorted(f"{k[0]}:{k[1]}" for k in tables)
        return [], {}, (f"no {domain} table" + (f" named {chart['table']!r}" if chart.get("table")
                                                else "") + (f" — the payload carries {have}"
                                                            if have else " on the payload"))
    key, cols = got
    z_kind = None
    units: Dict[str, str] = {}

    def col(name: str, *, numeric: bool = True) -> Optional[np.ndarray]:
        if not name or name not in cols:
            return None
        a = np.asarray(cols[name])
        if numeric and not _numeric(a):
            return None
        return a

    def unit_label(name: str) -> str:
        u = unit_of_column(name, z_kind=z_kind) if name else None
        pretty = format_unit(u, pretty=True) if u else ""
        return f"{name} ({pretty})" if pretty else name

    gname = str(chart.get("group_by") or "")
    group = np.asarray(cols[gname]) if gname and gname in cols else None
    draw = dict(chart.get("draw") or {})
    axes = dict(chart.get("axes") or {})
    try:
        if kind in ("xy", "timeseries"):
            xname = str(chart.get("x") or ("t" if kind == "timeseries" else ""))
            yname = str(chart.get("y") or chart.get("value") or "")
            x = col(xname)
            y = col(yname)
            if x is None or y is None:
                return [], {}, (f"pick numeric X and Y columns (have {sorted(cols)[:12]}…)"
                                if len(cols) > 12 else f"pick numeric X and Y columns (have "
                                                        f"{sorted(cols)})")
            if kind == "timeseries" and xname == "t" and md:
                n = int(np.nanmax(x)) + 1 if x.size else 1
                mode = str(chart.get("time") or "elapsed")
                xs, xlabel, _tk = SR.frame_times(n, frame_time_jd=md.get("frame_time_jd"),
                                                 dt_s=md.get("dt_s"), mode=mode,
                                                 utc_offset_s=md.get("utc_offset_s") or 0)
                lut = np.asarray(xs, dtype=float)
                x = lut[np.clip(x.astype(int), 0, len(lut) - 1)]
                units["x"] = xlabel
            else:
                units["x"] = unit_label(xname)
            units["y"] = unit_label(yname)
            error = str(draw.get("error", "none"))
            rows = _rows_payload(cols) if error in ("none", "column") else None
            series = SR.xy_series(x, y, group, error, log_x=bool(axes.get("log_x")),
                                  log_y=bool(axes.get("log_y")),
                                  text=rows if rows is not None else None,
                                  err=col(str(chart.get("error_column") or "")) if error == "column"
                                  else None)
            for s in series:
                if "text" in s:
                    s["rows"] = s.pop("text")
            return series, units, ""
        if kind == "bar":
            cat = str(chart.get("x") or chart.get("group_by") or "")
            val = str(chart.get("y") or chart.get("value") or "")
            v = col(val)
            if not cat or cat not in cols or v is None:
                return [], {}, "pick a category column (X) and a numeric value (Y)"
            series_by = str(chart.get("color_by") or "")
            g2 = np.asarray(cols[series_by]) if series_by and series_by in cols else None
            units["y"] = unit_label(val)
            return SR.bar_series(np.asarray(cols[cat]), v, g2,
                                 error=str(draw.get("error", "sem")) or "sem"), units, ""
        if kind == "distribution":
            val = str(chart.get("value") or chart.get("y") or "")
            v = col(val)
            if v is None:
                return [], {}, "pick a numeric value column"
            sub = str(chart.get("sub_kind") or "histogram")
            units["x" if sub in ("histogram", "kde", "ecdf") else "y"] = unit_label(val)
            if sub == "strip":
                return SR.strip_series(v, group), units, ""
            if sub == "superplot":
                rep = str(chart.get("facet_by") or chart.get("color_by") or "")
                if not rep or rep not in cols:
                    return [], {}, "a SuperPlot needs a replicate column (Facet / colour by)"
                return SR.superplot_series(v, group if group is not None else
                                           np.array([""] * len(v), dtype=object),
                                           np.asarray(cols[rep])), units, ""
            return SR.distribution_series(v, group), units, ""
        if kind == "heatmap":
            r, c_, val = (str(chart.get("x") or "t"), str(chart.get("group_by") or "m"),
                          str(chart.get("value") or chart.get("y") or ""))
            v = col(val)
            if r not in cols or c_ not in cols or v is None:
                return [], {}, "pick row (X), column (Group by) and a numeric value"
            return SR.heatmap_grid(np.asarray(cols[r]), np.asarray(cols[c_]), v,
                                   str(draw.get("reducer", "mean"))), units, ""
        if kind == "hexbin":
            x = col(str(chart.get("x") or ""))
            y = col(str(chart.get("y") or chart.get("value") or ""))
            if x is None or y is None:
                return {}, {}, "pick numeric X and Y columns"
            units["x"], units["y"] = unit_label(str(chart.get("x"))), unit_label(str(chart.get("y") or chart.get("value")))
            return SR.hexbin_series(x, y, gridsize=int(chart.get("bins", 20) or 20),
                                    log_x=bool(axes.get("log_x")),
                                    log_y=bool(axes.get("log_y"))), units, ""
        if kind == "msd":
            tid = col("track_id", numeric=False)
            t = col("t")
            y = col("y")
            x = col("x")
            if tid is None or t is None or y is None or x is None:
                return [], {}, "an MSD needs track_id, t, y and x on the table (Track Objects)"
            dt = float((md or {}).get("dt_s") or 1.0)
            px = float((md or {}).get("pixel_size_um") or 1.0)
            units["x"], units["y"] = ("lag (s)" if (md or {}).get("dt_s") else "lag (frames)",
                                      "MSD (µm²)" if (md or {}).get("pixel_size_um") else "MSD (px²)")
            return SR.msd_series(tid, t, y * px, x * px, group, dt=dt,
                                 max_lag=int(chart.get("bins", 0) or 0) or None), units, ""
        if kind == "survival":
            L = col("track_length")
            if L is None:
                L = col(str(chart.get("value") or ""))
            if L is None:
                return [], {}, "a survival curve needs track_length (or a duration column)"
            dt = float((md or {}).get("dt_s") or 1.0)
            units["x"] = "time (s)" if (md or {}).get("dt_s") else "frames"
            units["y"] = "fraction surviving"
            return SR.survival_series(L, group, dt=dt), units, ""
        if kind == "rose":
            a = col(str(chart.get("value") or chart.get("x") or "direction"))
            if a is None:
                return [], {}, "a rose needs an angle column (degrees)"
            return SR.rose_series(a, group, bins=int(chart.get("bins", 12) or 12)), units, ""
    except (ValueError, KeyError) as exc:
        return [], {}, str(exc)
    return [], {}, f"unknown chart kind {kind!r}"


def chart_geometry(chart: Mapping[str, Any], series: Any, rect_mm: Sequence[float],
                   page: Mapping[str, Any], *, units: Optional[Mapping[str, str]] = None
                   ) -> Dict[str, Any]:
    """The geometry of ``chart`` inside ``rect_mm`` under the page's style, measured with
    Qt's font metrics."""
    style = {"font_pt": float(page.get("font_pt", 8.0)),
             "line_width": float(page.get("line_width", 1.0)),
             "marker_size": float(page.get("marker_size", 3.0)),
             "palette": str(page.get("palette", "colorblind")),
             "grid": bool(page.get("grid", False)), "dark": bool(page.get("dark", False)),
             "font": str(page.get("font", "sans"))}
    return C.chart_geometry(chart, series, rect_mm=rect_mm, style=style,
                            measure=measure_for(style), units=units)


def chart_csv(chart: Mapping[str, Any], series: Any) -> str:
    """The values a chart draws, as CSV — the export bundle's data file."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    kind = str(chart.get("kind", "xy"))
    if kind == "heatmap":
        rows, cols, mat = series
        w.writerow([""] + list(cols))
        for r, line in zip(rows, mat):
            w.writerow([r] + ["" if v is None else v for v in line])
    elif kind == "hexbin":
        w.writerow(["x", "y", "count"])
        for (x, y), c in zip(series.get("centers", []), series.get("counts", [])):
            w.writerow([x, y, c])
    elif kind == "bar":
        w.writerow(["series", "category", "mean", "err", "n"])
        for s in series:
            for cat, y, e, n in zip(s["categories"], s["y"], s["err"], s["n"]):
                w.writerow([s["label"], cat, y, e, n])
    elif kind == "distribution":
        w.writerow(["series", "value"])
        for s in series:
            for v in s["values"]:
                w.writerow([s["label"], v])
    elif kind == "msd":
        w.writerow(["series", "lag", "msd", "n"])
        for s in series:
            for l, m, n in zip(s["lag"], s["msd"], s["n"]):
                w.writerow([s["label"], l, m, n])
    elif kind == "survival":
        w.writerow(["series", "t", "s"])
        for s in series:
            for t, v in zip(s["t"], s["s"]):
                w.writerow([s["label"], t, v])
    elif kind == "rose":
        w.writerow(["series", "from_deg", "to_deg", "count"])
        for s in series:
            e = s["edges_deg"]
            for k, c in enumerate(s["counts"]):
                w.writerow([s["label"], e[k], e[k + 1], c])
    else:
        w.writerow(["series", "x", "y", "err"])
        for s in series:
            err = s.get("err") or [None] * len(s["x"])
            for x, y, e in zip(s["x"], s["y"], err):
                w.writerow([s["label"], x, y, "" if e is None else e])
    return buf.getvalue()


__all__ = ["chart_tables", "pick_table", "table_columns", "resolve_series", "chart_geometry",
           "measure_for", "chart_csv"]
