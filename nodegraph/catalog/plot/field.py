"""plot.field — a measured field drawn on its image as a figure: the image plane, a column of
the field as a colour map with a colour bar, optional arrows and iso-contours, a µm scale bar
(2026-10-09)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

#: the background's longest side in the figure spec: one figure, or each frame of a series
_SIDE_ONE, _SIDE_FRAME = 1024, 512
_CMAPS = ("auto", "turbo", "viridis", "magma", "cividis", "coolwarm")
#: maps with a meaningful midpoint: blank limits are centred on zero for these
_DIVERGING = frozenset({"coolwarm"})
_ONE = {"per": frozenset({"all"})}


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


def _frame_entry(ds: Dataset, layer: str, col: str, m: int, t: int, z: int, c: int,
                 side: int, arrows: bool, every: int) -> Dict[str, Any]:
    img, _f = _background(ds, m, t, z, c, side)
    fr = FM.raster(ds, layer, col, m, t, z, c)
    entry: Dict[str, Any] = {"t": int(t),
                             "image": None if img is None else FIG.pack_plane(img)}
    if fr is not None:
        entry["grid"] = [[None if not np.isfinite(v) else float(v) for v in row]
                         for row in fr.grid]
        entry["extent"] = [float(v) for v in fr.extent]
        entry["lattice"] = bool(fr.lattice)
    if arrows:
        vf = FM.vectors(ds, m, t, z, c, layer=layer)
        if vf is not None:
            rows = vf.rows[::max(1, int(every))]
            rows = rows[np.isfinite(rows).all(axis=1)]
            entry["arrows"] = [[float(v) for v in r] for r in rows]
            entry["arrow_scale"] = float(FM.auto_scale(vf))
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
    * **What it draws** the image plane of ``position`` / ``frame`` / ``plane`` / ``channel``
      (auto contrast, pooled to ≤ 1024 px) in grey; the ``value`` column of the ``table``
      field as a colour map at ``opacity`` — rasterised by :mod:`nodegraph.fieldmap`, the
      Viewer's own Field-overlay rasteriser, so screen and figure agree — with a colour bar
      naming the column and its unit; optional arrows (auto length: the 95th-percentile
      vector one node spacing) and iso-contours; a µm scale bar.
    * **Per** ``all`` = the one frame chosen; ``frame`` = one figure per frame of the input
      (backgrounds ≤ 512 px), drawn when shown — scrub it in the Viewer or export a movie.
    * **Limits** ``color_range`` 'low, high'; blank = the column's 1st–99th percentile over
      the whole series, so every frame of a movie shares one scale. Blank limits on a
      DIVERGING map are made symmetric about zero, so its neutral colour sits on 0 (the
      Viewer Field overlay's ``center_zero`` default). Typed limits are kept as typed.
    """
    FIG.require_matplotlib()
    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    per = str(modes.get("per", "all"))
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
    clamp = lambda v, n: min(max(0, int(v or 0)), max(0, int(n) - 1))   # noqa: E731
    m = clamp(ctx.params.get("position", 0), ax.m)
    z = clamp(ctx.params.get("plane", 0), ax.z)
    c = clamp(ctx.params.get("channel", 0), ax.c)
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
    # (figure_frame — a chart has no µm, and the Viewer must not draw a scale bar on it), so
    # the image's scale is read where it lives; the input's identity already keys the memo
    try:
        px_um = float((ds.metadata or {}).get("pixel_size_um") or 0.0)
    except (TypeError, ValueError):
        px_um = 0.0
    bar_um = float(ctx.params.get("scale_bar_um", 0.0) or 0.0)
    if bar_um <= 0 and px_um > 0:
        bar_um = _scale_bar_um(ax.x * px_um)
    style = FIG.style_from_ctx(ctx)
    frames: List[int] = (list(range(max(1, int(ax.t)))) if per == "frame"
                         else [clamp(ctx.params.get("frame", 0), ax.t)])
    side = _SIDE_FRAME if per == "frame" else _SIDE_ONE
    entries = []
    for k, t in enumerate(frames):
        entries.append(_frame_entry(ds, layer, col, m, t, z, c, side, arrows, every))
        ctx.progress(k + 1, len(frames), f"{col}: frame {t}")
    spec: Dict[str, Any] = {
        "kind": "field",
        "frames": entries,
        "image_hw": [int(ax.y), int(ax.x)],
        "draw": {"colormap": cmap, "range": [float(lo), float(hi)],
                 "opacity": float(min(1.0, max(0.0, float(ctx.params.get("opacity", 0.6))))),
                 "contours": max(0, int(ctx.params.get("contours", 0) or 0)),
                 "value_label": f"{col} ({unit})" if unit else col,
                 "scale_bar": ({"um": bar_um, "px": bar_um / px_um}
                               if bar_um > 0 and px_um > 0 else None)},
        "axes": {"title": str(ctx.params.get("title", "") or ""),
                 "x_label": str(ctx.params.get("x_label", "") or ""),
                 "y_label": str(ctx.params.get("y_label", "") or "")},
        "style": style,
        "source": {"table": layer, "value": col, "m": m, "z": z, "c": c},
    }
    if per == "frame":
        extra, labels = FIG.frame_clock(ds.metadata or {}, len(frames), ctx.calib("dt_s"))
        spec.update(per="frame", frame_labels=labels)
        h, w = FIG.figure_pixels(style["width_mm"], style["height_mm"], style["dpi"])
        return FIG.picture_series(spec, len(frames), h, w, extra)
    return FIG.picture_dataset(FIG.render_rgb(spec), spec)


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
        InInt("frame", "Frame", unit="", field=False, default=0, available_in=_ONE,
              description="Which timepoint is drawn (the first is 0); beyond the last, the "
                          "last. With Per frame every timepoint gets its own figure."),
        InInt("position", "Position", unit="", field=False, default=0,
              description="Which stage position is drawn (the first is 0)."),
        InInt("plane", "Plane", unit="", field=False, default=0,
              description="Which Z plane: a per-plane 2-D field is drawn on its plane; a 3-D "
                          "field is cut there (interpolated between its node planes)."),
        InInt("channel", "Channel", unit="", field=False, default=0,
              description="Which channel the grey background shows. The field itself is "
                          "drawn whichever channel it was measured on."),
        InString("color_range", "Colour range", field=False, default="",
                 description="Fix the colour scale as 'low, high'. Blank uses the column's "
                             "1st–99th percentile over the whole series, so every frame of a "
                             "movie shares one scale; on the diverging coolwarm map, blank "
                             "is also made symmetric about zero so white means 0. A peak "
                             "beyond the range saturates: type a wider range to see it."),
        InFloat("opacity", "Opacity", unit="", field=False, default=0.6,
                description="How opaque the colour map is over the image, 0 to 1. Lower "
                            "shows more of the image; 1 hides it under the field."),
        InBool("arrows", "Arrows", field=False, default=True,
               description="Draw the displacement as arrows, auto-lengthened so the "
                           "95th-percentile vector is one node spacing long."),
        InInt("arrow_every", "Arrows every", unit="", field=False, default=1,
              description="Draw one arrow in N — for a dense field whose arrows would "
                          "overlap. Only read when Arrows is on."),
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
        Mode("per", ["all", "frame"], default="all", label="Per",
             description="One figure of the chosen frame, or one figure per frame of the "
                         "input.",
             choice_docs={
                 "all": "One figure of the timepoint Frame names — what Export Figure writes "
                        "as PNG, SVG or PDF.",
                 "frame": "One figure per frame of the input, drawn when shown, on one colour "
                          "scale — scrub it in the Viewer or export it with Export Movie.",
             }),
        *FIG.style_modes(palette=False),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains=frozenset({Domain.POINT}),
    adds_domains=frozenset({Domain.VOXEL}),       # a picture
    fresh_output=True, meta_transform=FIG.figure_frame,
    description="A measured field drawn on its image as a figure: the image plane in grey, a "
                "field column as a colour map with a colour bar, optional arrows and "
                "contours, a µm scale bar — one frame, or one figure per frame for a movie.")
