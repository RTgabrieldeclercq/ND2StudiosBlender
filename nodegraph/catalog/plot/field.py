"""plot.field — a measured field drawn on its image as a figure: the image plane, a column of
the field as a colour map with a colour bar, optional arrows and iso-contours, a µm scale bar
(2026-10-09). One view per position, frame and plane of the input, picked by the Viewer's
playback strip."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import nodegraph.catalog._shared.figure as FIG
import nodegraph.fieldmap as FM
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (Granularity, InBool, InDataset, InFloat, InInt, InString,
                                Mode, OutDataset)

_CMAPS = ("auto", "turbo", "viridis", "magma", "cividis", "coolwarm")
#: maps with a meaningful midpoint: blank limits are centred on zero for these
_DIVERGING = frozenset({"coolwarm"})


def _source_digest(path: str) -> str:
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.blake2b(fh.read(), digest_size=8).hexdigest()
    except OSError:
        return ""


#: this module's and the rasteriser's code: a view drawn by old code is never reused by new
_CODE = _source_digest(__file__) + _source_digest(FM.__file__)


def _background(ds: Dataset, m: int, t: int, z: int, c: int, side: int
                ) -> Tuple[Optional[np.ndarray], int]:
    """The image plane at ≤ ``side`` px a side (the coarsest pyramid level that still covers
    it, then mean-pooled), auto-contrasted 1–99.5 % to uint8; and the pooling factor relative
    to full resolution. ``(None, 1)`` without an image."""
    prov = getattr(ds, "image", None)
    if prov is None:
        return None, 1
    ax = ds.axes
    level, f = 0, 1
    for lv in range(int(getattr(prov, "levels", 1) or 1) - 1, -1, -1):
        la = prov.level_axes(lv)
        if max(la.y, la.x) >= side or lv == 0:
            level, f = lv, max(1, int(round(ax.x / max(1, la.x))))
            break
    la = prov.level_axes(level)
    plane = np.asarray(prov.get_region(level, m, t, z, c, 0, la.y, 0, la.x), dtype=float)
    k = max(1, -(-max(plane.shape) // side))
    if k > 1:
        hh, ww = plane.shape[0] // k * k, plane.shape[1] // k * k
        plane = plane[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
    fin = plane[np.isfinite(plane)]
    lo, hi = (np.percentile(fin, [1.0, 99.5]) if fin.size else (0.0, 1.0))
    img = np.clip((plane - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
    return img, f * k


#: The most arrows one view draws. Past a few thousand they merge into a texture that shows
#: no direction, and each one costs the spec and matplotlib time: a stitched 36-tile chip
#: mosaic has ~90,000 PIV nodes per view.
_MAX_ARROWS = 2500


def _thin_arrows(rows: np.ndarray, spacing: float, every: int) -> Tuple[np.ndarray, float]:
    """One arrow per square block ``every`` node spacings wide — the first node in each —
    the blocks widened until at most :data:`_MAX_ARROWS` remain. Returns the kept rows and
    the block size in pixels. On a lattice with ``every = 1`` every node is its own block,
    so every arrow stays.

    Spatial rather than every N-th row: rows come tile by tile and row by row, so a stride
    leaves stripes on a lattice and clumps on a mosaic."""
    rows = rows[np.isfinite(rows).all(axis=1)]
    s = float(spacing) if np.isfinite(spacing) and spacing > 0 else 1.0
    if rows.shape[0] == 0:
        return rows, s
    y, x = rows[:, 0], rows[:, 1]
    span_y = float(y.max() - y.min()) + s
    span_x = float(x.max() - x.min()) + s
    cell = s * max(1, int(every))
    if (span_y / cell) * (span_x / cell) > _MAX_ARROWS:
        cell = float(np.sqrt(span_y * span_x / _MAX_ARROWS))
    by = np.floor((y - y.min() + s / 2.0) / cell).astype(np.int64)
    bx = np.floor((x - x.min() + s / 2.0) / cell).astype(np.int64)
    _, first = np.unique(by * (int(bx.max()) + 1) + bx, return_index=True)
    return rows[np.sort(first)], cell


def _frame_entry(ds: Dataset, layer: str, col: str, m: int, t: int, z: int, c: int,
                 side: int, arrows: bool, every: int) -> Dict[str, Any]:
    img, _f = _background(ds, m, t, z, c, side)
    # a lattice field draws one cell per node whatever this says; a scattered one (a
    # stitched mosaic, tracked points) is resampled to about two figure pixels a cell —
    # finer than the Viewer's 128, which has the GUI thread's budget to keep
    fr = FM.raster(ds, layer, col, m, t, z, c, max_side=max(128, side // 2))
    entry: Dict[str, Any] = {"m": int(m), "t": int(t), "z": int(z),
                             "image": None if img is None else FIG.pack_plane(img)}
    if fr is not None:
        entry["grid"] = [[None if not np.isfinite(v) else float(v) for v in row]
                         for row in fr.grid]
        entry["extent"] = [float(v) for v in fr.extent]
        entry["lattice"] = bool(fr.lattice)
    if arrows:
        vf = FM.vectors(ds, m, t, z, c, layer=layer)
        if vf is not None:
            rows, cell = _thin_arrows(vf.rows, vf.spacing_px, every)
            entry["arrows"] = [[float(v) for v in r] for r in rows]
            # the 95th-percentile arrow as long as the gap between DRAWN arrows
            s = vf.spacing_px if np.isfinite(vf.spacing_px) and vf.spacing_px > 0 else cell
            entry["arrow_scale"] = float(FM.auto_scale(vf)) * cell / s
    return entry


def _scale_bar_um(width_um: float) -> float:
    """A round length near a fifth of the field's width: 1, 2 or 5 × 10^k µm."""
    if not width_um > 0:
        return 0.0
    target = width_um / 5.0
    k = 10.0 ** np.floor(np.log10(target))
    return float(min((1.0, 2.0, 5.0, 10.0), key=lambda f: abs(f * k - target)) * k)


def _compute_plot_field(ctx: EvalContext) -> Dataset:
    """A measured field drawn on its image, as a PICTURE.

    Resolved spec (§0, 2026-10-09):

    * **Kind** plot → ``op_key="plot.field"``, category ``plot``: a picture of the input like
      every plot node (``fresh_output``, the ``figure_frame`` envelope), whole table at once
      (WHOLE_SERIES, empty kernel_axes — the background plane is read through the provider).
    * **What it draws** the image plane of the viewed position / frame / plane and of
      ``channel`` (auto contrast, pooled to ≤ 512 px on screen, ≤ 1024 px when Export Figure
      writes it) in grey; the ``value`` column of the ``table``
      field as a colour map at ``opacity`` — rasterised by :mod:`nodegraph.fieldmap`, the
      Viewer's own Field-overlay rasteriser, so screen and figure agree — with a colour bar
      naming the column and its unit; optional arrows (auto length: the 95th-percentile
      vector one node spacing) and iso-contours; a µm scale bar.
    * **Views, not sockets** (2026-10-09, the user's call). The picture has the input's M, T
      and Z, one view each, drawn lazily the first time the Viewer shows it
      (:class:`FIG.ViewFigureProvider`). The Viewer's playback strip therefore picks
      position, frame and plane, as it does for the image; the node has no Frame, Position
      or Plane socket and no Per mode. Export Figure picks a view with its own Frame /
      Position / Plane, since a writer cannot follow the strip; Export Movie sweeps T or Z.
    * **Limits** ``color_range`` 'low, high'; blank = the column's 1st–99th percentile over
      the whole table, so every view (position, frame, plane) shares one scale. Blank limits on a
      DIVERGING map are made symmetric about zero, so its neutral colour sits on 0 (the
      Viewer Field overlay's ``center_zero`` default). Typed limits are kept as typed.
    """
    FIG.require_matplotlib()
    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    layer = _resolve_layer(
        FM.layer_names(ds), ctx.layer("table"), node="plot field", socket="table",
        what="field table", where="the `data` input",
        remedy="wire a DIC / PIV / DVC node (or a Field card after one) — they write the "
               "Point table this draws", ctx=ctx)[0]
    want = str(ctx.params.get("value", "") or "")
    layer, col = FM.auto_pick(ds, layer, want)
    if want and col != want:
        raise ValueError(f"plot field: the table {layer!r} has no numeric column {want!r}; "
                         f"it has {', '.join(FM.column_names(ds, layer)[:30])}")
    if not col:
        raise ValueError(f"plot field: the table {layer!r} carries no numeric column to "
                         f"draw")
    ax = ds.axes
    c = min(max(0, int(ctx.params.get("channel", 0) or 0)), max(0, int(ax.c) - 1))
    arrows = bool(ctx.params.get("arrows", True))
    every = max(1, int(ctx.params.get("arrow_every", 1) or 1))
    rng = FIG.parse_range(ctx.params.get("color_range", ""), socket="color_range")
    lo, hi = FM.series_limits(ds, layer, col)
    lo = rng[0] if rng[0] is not None else lo
    hi = rng[1] if rng[1] is not None else hi
    cm = str(modes.get("colormap", "auto"))
    cmap = cm if cm != "auto" else ("coolwarm" if lo < 0.0 < hi else "turbo")
    if cmap in _DIVERGING and rng[0] is None and rng[1] is None:
        # a diverging map's midpoint must BE zero, or an unrotated background reads as
        # negative rotation; the Viewer's Field overlay centres the same way by default
        r = max(abs(lo), abs(hi))
        lo, hi = -r, r
    unit = FM.column_unit(ds, layer, col)
    # the INPUT's pixel size, not ctx.calib: a plot's own envelope carries no calibration
    # (a chart has no µm, and the Viewer must not draw a scale bar on it), so the image's
    # scale is read where it lives; the input's identity already keys the memo
    md_in = ds.metadata or {}
    try:
        px_um = float(md_in.get("pixel_size_um") or 0.0)
    except (TypeError, ValueError):
        px_um = 0.0
    bar_um = float(ctx.params.get("scale_bar_um", 0.0) or 0.0)
    if bar_um <= 0 and px_um > 0:
        bar_um = _scale_bar_um(ax.x * px_um)
    style = FIG.style_from_ctx(ctx)
    title = str(ctx.params.get("title", "") or "")
    template: Dict[str, Any] = {
        "kind": "field",
        "per": "view",
        "frames": [],
        "image_hw": [int(ax.y), int(ax.x)],
        "draw": {"colormap": cmap, "range": [float(lo), float(hi)],
                 "opacity": float(min(1.0, max(0.0, float(ctx.params.get("opacity", 0.6))))),
                 "contours": max(0, int(ctx.params.get("contours", 0) or 0)),
                 "value_label": f"{col} ({unit})" if unit else col,
                 "scale_bar": ({"um": bar_um, "px": bar_um / px_um}
                               if bar_um > 0 and px_um > 0 else None)},
        "axes": {"title": title,
                 "x_label": str(ctx.params.get("x_label", "") or ""),
                 "y_label": str(ctx.params.get("y_label", "") or "")},
        "style": style,
        "source": {"table": layer, "value": col, "c": c},
    }
    extra = FIG.view_metadata(md_in, ax)
    _clock, t_labels = FIG.frame_clock(md_in, int(ax.t), ctx.calib("dt_s"))
    names = extra.get("position_name")

    def stamp(m: int, t: int, z: int) -> str:
        """The view's place in the title, counted from 1 like the frame clock's own labels
        (``frame 3/6``): the position's name (else ``position k/M``), the frame's time (else
        ``frame k/T``) and ``plane k/Z``, each only when that axis has more than one."""
        parts = []
        if ax.m > 1:
            nm = names[m] if isinstance(names, list) and m < len(names) and names[m] else ""
            parts.append(str(nm) or f"position {m + 1}/{ax.m}")
        if ax.t > 1:
            parts.append(t_labels[t] if t < len(t_labels) else f"frame {t + 1}/{ax.t}")
        if ax.z > 1:
            parts.append(f"plane {z + 1}/{ax.z}")
        return " · ".join(parts)

    def spec_at(m: int, t: int, z: int, side: int) -> Dict[str, Any]:
        """The whole spec of view ``(m, t, z)``, its background ≤ ``side`` px a side."""
        out = dict(template)
        out["frames"] = [_frame_entry(ds, layer, col, m, t, z, c, side, arrows, every)]
        out["per"] = "all"
        out["source"] = dict(template["source"], m=int(m), t=int(t), z=int(z))
        st = stamp(m, t, z)
        a = dict(template["axes"])
        a["title"] = f"{title} — {st}" if (title and st) else (title or st)
        out["axes"] = a
        return out

    h, w = FIG.figure_pixels(style["width_mm"], style["height_mm"], style["dpi"])
    prov = FIG.ViewFigureProvider(template, spec_at, ax.m, ax.t, ax.z, h, w,
                                  key=_view_key(template, ds, layer, col))
    return FIG.picture_views(prov, extra)


def _view_key(template: Dict[str, Any], ds: Dataset, layer: str, col: str) -> str:
    """Everything a view's pixels depend on: the template (settings, limits, style), the
    drawing code, the input image's identity, the pixel size and frame interval, and the
    field columns the raster and the arrows read. It keys the provider's cached views."""
    import json
    from nodegraph.memo import digest
    cols = FM.point_tables(ds).get(layer) or {}
    names = [k for k in ("m", "t", "z", "c", "y", "x", col, "disp_y", "disp_x", "disp_z",
                         "vy", "vx", "vz") if k in cols]
    data = digest(*[np.ascontiguousarray(np.asarray(cols[k], dtype=float)).tobytes()
                    for k in names]) if names else ""
    img = getattr(ds, "image", None)
    try:
        img_id = repr(img.fingerprint()) if img is not None else "no image"
    except Exception:                                  # noqa: BLE001 — identity only
        img_id = f"unfingerprinted {id(img)}"
    md = ds.metadata or {}
    return digest("plot-field-views", FIG._DRAW_CODE, _CODE,
                  json.dumps(template, sort_keys=True, default=str), img_id,
                  repr((md.get("pixel_size_um"), md.get("dt_s"))), layer, col,
                  ",".join(names), data)


register_node(
    _compute_plot_field, op_key="plot.field", label="Plot Field", category="plot",
    inputs=[
        InDataset("data", description="The Dataset carrying the image and the field table — "
                                       "a DIC / PIV / DVC node's output, or a Field card's."),
        InString("table", "Table", field=False, default="", layer_in=Domain.POINT,
                 description="The field table to draw. Blank uses the only one on the wire "
                             "(correlation grids are not fields and are skipped)."),
        InString("value", "Value", field=False, default="", column_in=Domain.POINT,
                 column_from="table",
                 description="The column the colour map shows. Blank picks the most derived "
                             "one on the table — von_mises, then strain_eq, jacobian, curl, "
                             "then disp_mag_um. The plot changes no measurement."),
        InInt("channel", "Channel", unit="", field=False, default=0,
              description="Which channel the grey background shows. The field itself is "
                          "drawn whichever channel it was measured on."),
        InString("color_range", "Colour range", field=False, default="",
                 description="Fix the colour scale as 'low, high'. Blank uses the column's "
                             "1st–99th percentile over the whole table, so every position, "
                             "frame and plane shares one scale; on the diverging coolwarm map, "
                             "blank "
                             "is also made symmetric about zero so white means 0. A peak "
                             "beyond the range saturates: type a wider range to see it."),
        InFloat("opacity", "Opacity", unit="", field=False, default=0.6,
                description="How opaque the colour map is over the image, 0 to 1. Lower "
                            "shows more of the image; 1 hides it under the field."),
        InBool("arrows", "Arrows", field=False, default=True,
               description="Draw the displacement as arrows, auto-lengthened so the "
                           "95th-percentile vector spans the gap between drawn arrows. At "
                           "most 2500 are drawn per view: a bigger field (a stitched "
                           "mosaic) is thinned further, evenly."),
        InInt("arrow_every", "Arrows every", unit="", field=False, default=1,
              description="Draw one arrow per block of N × N nodes — for a dense field "
                          "whose arrows would overlap. HIGHER draws fewer, longer arrows. "
                          "Only read when Arrows is on."),
        InInt("contours", "Contours", unit="", field=False, default=0,
              description="How many iso-lines to draw over the colour map; 0 draws none."),
        InFloat("scale_bar_um", "Scale bar", unit="um", field=False, default=0.0,
                description="The scale bar's length in µm; 0 picks a round length about a "
                            "fifth of the field. No bar without a pixel size."),
        *FIG.style_sockets(axes=False),
    ],
    outputs=[OutDataset(label="Figure")],
    modes=[
        Mode("colormap", list(_CMAPS), default="auto", label="Colour map",
             description="The colours the field column is drawn in.",
             choice_docs={
                 "auto": "coolwarm (diverging, zero neutral) for a column with both signs — "
                         "strain, curl, a stress component — and turbo otherwise: the Viewer "
                         "Field overlay's own choice.",
                 "turbo": "A high-contrast rainbow — what displacement magnitude is usually "
                          "shown in; not perceptually uniform, so judge levels from the bar.",
                 "viridis": "Dark purple to yellow, perceptually uniform and colour-blind "
                            "safe — the accessible choice for a one-signed quantity.",
                 "magma": "Black through red to pale yellow, perceptually uniform — strong "
                          "contrast where the field peaks.",
                 "cividis": "Blue to yellow, designed to read the same for the common "
                            "colour-vision deficiencies.",
                 "coolwarm": "Blue through white to red, diverging — for a signed quantity "
                             "whose zero must read as neutral. Blank limits are centred on "
                             "zero.",
             }),
        *FIG.style_modes(palette=False),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains=frozenset({Domain.POINT}),
    adds_domains=frozenset({Domain.VOXEL}),       # a picture
    fresh_output=True, meta_transform=FIG.figure_views,
    description="A measured field drawn on its image as a figure: the image plane in grey, a "
                "field column as a colour map with a colour bar, optional arrows and "
                "contours, a µm scale bar. The Viewer's playback strip picks the position, "
                "frame and plane shown, as it does for the image; Export Figure writes one "
                "view, Export Movie sweeps time or planes.")
