"""plot.heatmap — a colour grid of a table summary or of a Voxel layer (V4.00 step 8)."""
from __future__ import annotations

import numpy as np

import nodegraph.catalog._shared.figure as FIG
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import (_resolve_layer, _structure_layers,
                                              _voxel_layers)
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, Mode, OutDataset

_HEAT_SOURCES = ("label", "point", "track", "voxel")
_TABLES = {"source": frozenset({"label", "point", "track"})}
_VOXEL = {"source": frozenset({"voxel"})}
#: the longest side a Voxel plane is drawn at — the figure is ~1000 px wide, so more is
#: pixels nobody sees, carried in the figure spec
_MAX_SIDE = 512


def _compute_plot_heatmap(ctx: EvalContext) -> Dataset:
    """A heatmap: a table summarised on a grid, or a Voxel layer's plane.

    Resolved spec (V4.00 step 8, `build-node-v2`):

    * ``source`` label / point / track: one cell per (``row`` value, ``column`` value), the
      ``reducer`` of the ``value`` column's finite rows in it — e.g. mean area per frame and
      position; an empty cell is blank, not zero;
    * ``source`` voxel: the ``table`` socket names a Voxel layer (a mask, a distance field,
      labels); the plane of ``position`` and ``frame`` is drawn, Z max-projected, its first
      channel, mean-pooled to at most 512 px a side;
    * ``colormap`` and ``color_range`` (``low, high``; blank = the data's range); a colour bar
      names the value; a PICTURE output like every plot; whole table or layer at once.
    """
    FIG.require_matplotlib()
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    source = str(modes.get("source", "label"))
    domain = Domain(source)
    want = ctx.layer("table")
    if source == "voxel":
        layer = _resolve_layer(
            _voxel_layers(ds), want, node="plot heatmap", socket="table",
            what="Voxel layer", where="the `data` input",
            remedy="wire a node that makes a Voxel layer (analysis.threshold makes a mask, "
                   "analysis.label a label raster), or switch Source", ctx=ctx)[0]
        arr = np.asarray(ds.get(Domain.VOXEL, layer).values)
        if arr.ndim == 7:
            arr = arr[0]
        m = min(max(0, int(ctx.params.get("position", 0) or 0)), arr.shape[0] - 1)
        t = min(max(0, int(ctx.params.get("frame", 0) or 0)), arr.shape[1] - 1)
        plane = np.asarray(arr[m, t, :, 0], dtype=float).max(axis=0)
        f = max(1, -(-max(plane.shape) // _MAX_SIDE))
        if f > 1:
            hh, ww = plane.shape[0] // f * f, plane.shape[1] // f * f
            plane = plane[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
        matrix = [[None if not np.isfinite(v) else float(v) for v in row] for row in plane]
        rl, cl, value_label = [], [], layer
        xl, yl = (f"x (px{f' / {f}' if f > 1 else ''})",
                  f"y (px{f' / {f}' if f > 1 else ''})")
        reducer = ""
    else:
        layer = _resolve_layer(
            _structure_layers(ds, domain), want, node="plot heatmap", socket="table",
            what=f"{domain.value} table", where="the `data` input",
            remedy=f"wire a node that makes a {domain.value} table, or switch Source",
            ctx=ctx)[0]
        cols = FIG.table_columns(ds, domain, layer)
        rname = str(ctx.params.get("row", "t") or "t")
        cname = str(ctx.params.get("column", "m") or "m")
        vname = str(ctx.params.get("value", "area") or "area")
        reducer = str(modes.get("reducer", "mean"))
        rl, cl, matrix = FIG.heatmap_grid(
            FIG.column(cols, rname, node="plot heatmap", socket="row", layer=layer),
            FIG.column(cols, cname, node="plot heatmap", socket="column", layer=layer),
            FIG.column(cols, vname, node="plot heatmap", socket="value", layer=layer, numeric=True),
            reducer)
        value_label = f"{reducer} of {vname}" if reducer != "count" else "rows"
        xl, yl = cname, rname
    spec = {
        "kind": "heatmap",
        "matrix": matrix, "row_labels": rl, "col_labels": cl,
        "draw": {"colormap": str(modes.get("colormap", "viridis")),
                 "color_range": FIG.parse_range(ctx.params.get("color_range", ""),
                                                socket="color_range"),
                 "value_label": value_label},
        "axes": {"title": str(ctx.params.get("title", "") or ""),
                 "x_label": str(ctx.params.get("x_label", "") or "") or xl,
                 "y_label": str(ctx.params.get("y_label", "") or "") or yl},
        "style": FIG.style_from_ctx(ctx),
        "source": {"source": source, "table": layer, "reducer": reducer},
    }
    ctx.progress(0, 1, "drawing the heatmap")
    return FIG.picture_dataset(FIG.render_rgb(spec), spec)


register_node(
    _compute_plot_heatmap, op_key="plot.heatmap", label="Plot Heatmap", category="plot",
    inputs=[
        InDataset("data", description="The Dataset whose table or Voxel layer is drawn."),
        InString("table", "Table", field=False, default="", layer_in_mode="source",
                 description="The table (or, with Source voxel, the Voxel layer) to draw. "
                             "Blank uses the only one on the wire, and asks when there are "
                             "several."),
        InString("row", "Rows", field=False, default="t", column_in_mode="source",
                 column_from="table", available_in=_TABLES,
                 description="The column whose distinct values make the grid's ROWS — `t` "
                             "for one row per frame."),
        InString("column", "Columns", field=False, default="m", column_in_mode="source",
                 column_from="table", available_in=_TABLES,
                 description="The column whose distinct values make the grid's COLUMNS — "
                             "`m` for one column per position."),
        InString("value", "Value", field=False, default="area", column_in_mode="source",
                 column_from="table", available_in=_TABLES,
                 description="The column each cell summarises with the Reducer. The plot "
                             "changes no measurement."),
        InInt("frame", "Frame", unit="", field=False, default=0, available_in=_VOXEL,
              description="Which timepoint of the Voxel layer is drawn (the first is 0); "
                          "beyond the last, the last."),
        InInt("position", "Position", unit="", field=False, default=0,
              available_in=_VOXEL,
              description="Which stage position of the Voxel layer is drawn (the first is "
                          "0); beyond the last, the last."),
        InString("color_range", "Colour range", field=False, default="",
                 description="Fix the colour scale as 'low, high'. Blank stretches it over "
                             "the data; fix it to compare two heatmaps by colour."),
        *FIG.style_sockets(axes=False),
    ],
    outputs=[OutDataset(label="Figure")],
    modes=[
        Mode("source", list(_HEAT_SOURCES), default="label", label="Source",
             description="What the heatmap draws: a table summarised on a grid, or a Voxel "
                         "layer's plane.",
             choice_docs={
                 **domain_docs(("label", "point", "track")),
                 "voxel": "A Voxel layer's plane (a mask, a distance field, a label raster) "
                          "at one frame and position, Z max-projected — the layer as an "
                          "image with a colour bar, not a table summary.",
             }),
        Mode("reducer", list(FIG.HEAT_REDUCERS), default="mean", label="Reducer",
             available_in=_TABLES,
             description="How the rows that fall in one cell become its colour.",
             choice_docs={
                 "mean": "The average value of the cell's rows — the usual summary; pulled "
                         "by outliers in a small cell.",
                 "median": "The middle value of the cell's rows — robust to a few outlying "
                           "objects, at the cost of ignoring their size.",
                 "sum": "The total of the cell's rows — e.g. the total area covered per "
                        "frame and position; grows with the number of objects.",
                 "min": "The smallest value in the cell — where the floor of a population "
                        "lies, e.g. the smallest object per frame.",
                 "max": "The largest value in the cell — the extreme object per frame and "
                        "position; one outlier sets it.",
                 "count": "The number of rows in the cell — how many objects each frame "
                          "and position holds (a row whose Value is missing is not counted).",
                 "std": "The spread (sample standard deviation) of the cell's rows — where "
                        "a population is heterogeneous; 0 for a lone row.",
             }),
        Mode("colormap", ["viridis", "magma", "cividis", "gray", "coolwarm"],
             default="viridis", label="Colour map",
             description="The colours values are drawn in.",
             choice_docs={
                 "viridis": "Dark purple to yellow, perceptually uniform and readable in "
                            "greyscale and by colour-blind readers — the safe default.",
                 "magma": "Black through red to pale yellow, perceptually uniform — "
                          "strong contrast at the high end for sparse bright cells.",
                 "cividis": "Blue to yellow, designed to read the same for the common "
                            "colour-vision deficiencies — for figures that must be accessible.",
                 "gray": "Black to white — prints exactly in greyscale; the eye separates "
                         "fewer levels than with a coloured map.",
                 "coolwarm": "Blue through white to red, diverging — for values with a "
                             "meaningful middle, e.g. a change around zero.",
             }),
        *FIG.style_modes(palette=False),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"source": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK}),
                                      "voxel": frozenset({Domain.VOXEL})}},
    adds_domains=frozenset({Domain.VOXEL}),       # a picture
    fresh_output=True, meta_transform=FIG.figure_frame,
    description="A colour grid: a table column summarised per row and column value (e.g. "
                "mean area per frame and position), or a Voxel layer's plane — drawn as a "
                "true-colour picture with a colour bar.")
