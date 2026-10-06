"""plot.xy — a line or scatter chart of two columns of a structure table (V4.00 step 7)."""
from __future__ import annotations

import numpy as np

from nodegraph.catalog._base import register_node
import nodegraph.catalog._shared.figure as FIG
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

_XY_DOMAINS = ("label", "point", "track")
_XY_ERRORS = ("none", "sd", "sem", "ci95")


def _compute_plot_xy(ctx: EvalContext) -> Dataset:
    """Draw ``y`` against ``x`` from one structure table, one series per ``group_by`` value.

    Resolved spec (V4.00 step 7, `build-node-v2`):

    * reads one table — ``domain`` label | point | track, its name from ``table`` (the only
      one on the wire when blank) — and its ``x``, ``y`` and optional ``group_by`` columns;
    * ``error = none`` draws every row; ``sd`` / ``sem`` / ``ci95`` summarise the rows that
      share an x as mean ± that spread, as bars or a band (``error_style``);
    * renders through matplotlib's Agg canvas (``_shared/figure``) into a PICTURE dataset —
      one RGB plane, ``(1, 1, 1, 3, H, W)`` uint8, fresh metadata with no calibration — whose
      size the style decides and ``figure_frame`` predicts at edit time;
    * footprint: the whole table at once, no image axes read (``WHOLE_SERIES``, no kernel
      axes); no 2D/3D lever — a chart has no Z.
    """
    FIG.require_matplotlib()
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    domain = Domain(str(modes.get("domain", "label")))
    layer = _resolve_layer(
        _structure_layers(ds, domain), ctx.layer("table"), node="plot xy", socket="table",
        what=f"{domain.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {domain.value} table (analysis.label + "
               f"analysis.measure for regions, detect.spots for points, track.objects for "
               f"tracks), or switch Domain", ctx=ctx)[0]
    cols = FIG.table_columns(ds, domain, layer)
    xname = str(ctx.params.get("x", "t") or "t")
    yname = str(ctx.params.get("y", "area") or "area")
    gname = str(ctx.params.get("group_by", "") or "")
    x = FIG.column(cols, xname, node="plot xy", socket="x", layer=layer, numeric=True)
    y = FIG.column(cols, yname, node="plot xy", socket="y", layer=layer, numeric=True)
    g = FIG.column(cols, gname, node="plot xy", socket="group_by", layer=layer) \
        if gname else None
    error = str(modes.get("error", "none"))
    log_x = bool(ctx.params.get("log_x", False))
    log_y = bool(ctx.params.get("log_y", False))
    series = FIG.xy_series(x, y, g, error, log_x=log_x, log_y=log_y)
    style = FIG.style_from_ctx(ctx)
    per = str(modes.get("per", "all"))
    x_range = FIG.parse_range(ctx.params.get("x_range", ""), socket="x_range")
    y_range = FIG.parse_range(ctx.params.get("y_range", ""), socket="y_range")
    spec = {
        "kind": "xy",
        "series": series,
        "draw": {"mode": str(modes.get("kind", "line_markers")),
                 "error_style": str(modes.get("error_style", "bars"))},
        "axes": {"title": str(ctx.params.get("title", "") or ""),
                 "x_label": str(ctx.params.get("x_label", "") or "") or xname,
                 "y_label": str(ctx.params.get("y_label", "") or "") or yname,
                 "log_x": log_x, "log_y": log_y,
                 "x_range": x_range, "y_range": y_range},
        "style": style,
        "source": {"domain": domain.value, "table": layer, "x": xname, "y": yname,
                   "group_by": gname, "error": error},
    }
    if per == "frame":
        # one figure per frame of the input: the rows of that frame, on axes every frame
        # shares (the data's whole range unless a range is fixed), drawn when shown
        tcol = FIG.column(cols, "t", node="plot xy", socket="per", layer=layer, numeric=True)
        frames = max(1, int(ds.axes.t))
        extra, labels = FIG.frame_clock(ds.metadata or {}, frames, ctx.calib("dt_s"))
        a = spec["axes"]
        if a["x_range"] == [None, None]:
            a["x_range"] = FIG.axis_range(x, log=log_x)
        if a["y_range"] == [None, None]:
            vals = list(y)
            if error != "none":
                vals = []
                garr = None if g is None else np.asarray(g, dtype=object)
                tt = np.asarray(tcol, dtype=float)
                for f in range(frames):
                    sel = tt == f
                    for s in FIG.xy_series(x[sel], y[sel],
                                           None if garr is None else garr[sel], error,
                                           log_x=log_x, log_y=log_y):
                        vals += list(s["y"])
                        if s.get("err"):
                            vals += [v + e for v, e in zip(s["y"], s["err"])]
                            vals += [v - e for v, e in zip(s["y"], s["err"])]
            a["y_range"] = FIG.axis_range(vals or list(y), log=log_y)
        spec.update(per="frame", error=error, frame_labels=labels,
                    rows={"x": [FIG.json_value(v) for v in x],
                          "y": [FIG.json_value(v) for v in y],
                          "g": (None if g is None else [FIG.json_value(v) for v in g]),
                          "t": [FIG.json_value(v) for v in tcol]})
        spec["series"] = []
        h, w = FIG.figure_pixels(style["width_mm"], style["height_mm"], style["dpi"])
        ctx.progress(0, 1, f"{frames} frames, each drawn when shown")
        return FIG.picture_series(spec, frames, h, w, extra)
    ctx.progress(0, 1, f"drawing {sum(s['n'] for s in series)} rows")
    return FIG.picture_dataset(FIG.render_rgb(spec), spec)


register_node(
    _compute_plot_xy, op_key="plot.xy", label="Plot XY", category="plot",
    inputs=[
        InDataset("data", description="The Dataset whose table is plotted — anything "
                                       "carrying a Label, Point or Track table."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen domain to plot. Blank uses the only "
                             "one on the wire, and asks when there are several."),
        InString("x", "X", field=False, default="t", column_in_mode="domain",
                 column_from="table",
                 description="The column on the horizontal axis — `t` (the frame) for a "
                             "time course, or any measured column against another."),
        InString("y", "Y", field=False, default="area", column_in_mode="domain",
                 column_from="table",
                 description="The column on the vertical axis. The plot reads the values "
                             "as they are in the table; it changes no measurement."),
        InString("group_by", "Group by", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="Draw one series per distinct value of this column — a "
                             "`condition` from a Page Output, a track id, a channel. Blank "
                             "draws every row as one series, with no legend."),
        *FIG.style_sockets(),
    ],
    outputs=[OutDataset(label="Figure")],
    modes=[
        Mode("domain", list(_XY_DOMAINS), default="label", label="Domain",
             description="Which kind of table the rows come from: one row per region, "
                         "per detection, or per tracked object.",
             choice_docs=domain_docs(_XY_DOMAINS)),
        Mode("kind", ["line", "scatter", "line_markers"], default="scatter",
             label="Kind",
             description="How each series is drawn.",
             choice_docs={
                 "line": "Join the points of a series by a line ordered by x, with no "
                         "markers — a clean time course when x is dense (with several rows "
                         "per x, average them with Error first).",
                 "scatter": "Markers only, no line — for two measurements against each "
                            "other, where joining neighbours in x would mean nothing.",
                 "line_markers": "A line through markers: a time course whose individual "
                                 "points still show, so a gap or an outlier is visible.",
             }),
        Mode("error", list(_XY_ERRORS), default="none", label="Error",
             description="Whether rows that share an x are summarised, and with what spread.",
             choice_docs={
                 "none": "Draw every row as its own point; nothing is averaged. With many "
                         "objects per frame this shows the whole population as a cloud.",
                 "sd": "Average the rows at each x and show ± one standard deviation — "
                       "the spread of the population itself, which does not shrink as n grows.",
                 "sem": "Average the rows at each x and show ± the standard error of the "
                        "mean (sd/√n) — how well the mean is known; narrows as n grows.",
                 "ci95": "Average the rows at each x and show the 95 % confidence interval "
                         "of the mean (Student's t, n-1 degrees of freedom) — wider than "
                         "sem, the honest bar for comparing two groups.",
             }),
        Mode("error_style", ["bars", "band"], default="bars", label="Error style",
             available_in={"error": frozenset({"sd", "sem", "ci95"})},
             description="How the spread is drawn around each mean.",
             choice_docs={
                 "bars": "An error bar with caps at every x — reads best with few x "
                         "values, and on a scatter.",
                 "band": "A translucent band between the lower and upper bounds — reads "
                         "best for a dense time course, where bars would overlap.",
             }),
        Mode("per", ["all", "frame"], default="all", label="Per",
             description="One figure of every row, or one figure per frame of the input.",
             choice_docs=FIG.PER_DOCS),
        *FIG.style_modes(),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_domains=frozenset({Domain.VOXEL}),       # a picture: its image IS a Voxel layer
    fresh_output=True, meta_transform=FIG.figure_frame,
    description="A line or scatter chart of two columns of a Label, Point or Track table, "
                "one series per group, optionally averaged with error bars — drawn as a "
                "true-colour picture the Viewer shows and Export Figure writes.")
