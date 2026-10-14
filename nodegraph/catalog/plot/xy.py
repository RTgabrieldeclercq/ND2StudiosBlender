"""plot.xy — a line or scatter chart of two columns of a structure table (V4.00 step 7)."""
from __future__ import annotations

import numpy as np

from nodegraph.catalog._base import register_node
import nodegraph.catalog._shared.figure as FIG
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, Mode, OutDataset
from nodegraph.units import format_unit, unit_of

_XY_DOMAINS = ("label", "point", "track")
_XY_ERRORS = ("none", "sd", "sem", "ci95", "pooled_sd")
#: the sockets and modes that read one value per point: per-point plots only
_PER_POINT = {"error": frozenset({"none"})}


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

    **Trendline and x ≤ 0** (2026-10-13, for a standard curve: concentration on a log x axis,
    blanks at 0). ``fit = linear`` fits each series a least-squares line IN THE PLOTTED
    COORDINATES (:func:`~nodegraph.catalog._shared.figure.xy_fit`) — log10 of an axis that
    is logarithmic — so turning on log x and log y fits a power law and the line is straight
    on the figure; it is fitted to the points the series draws (the means when ``error``
    summarises rows), and its equation and R² join the series' legend entry. ``zero_x =
    reference_line`` keeps a log-x plot's rows at x <= 0 instead of dropping them: each
    series' mean there is a dashed horizontal line in its colour, outside the fit. Both
    default to the old behaviour (``none`` / ``drop``), so a saved graph draws as before.

    **Langmuir, and a blank on linear axes** (2026-10-13, the same standard curve once its
    signal was read on linear axes). ``fit = langmuir`` fits each series the one-site binding
    curve ``y = Bmax·x / (Kd + x)`` in DATA units (:func:`~nodegraph.catalog._shared.figure.
    xy_langmuir`) on its x > 0 points, Bmax, Kd and R² in the legend. ``zero_x =
    reference_line`` now acts on a linear x axis too: the x <= 0 rows leave the points and
    become the dashed line, as on a log axis. A graph saved with ``reference_line`` on a
    linear x axis therefore draws its blank as a line where it drew points; ``drop`` on a
    linear axis is unchanged (those rows are ordinary points).

    **Colour by and Label by** (2026-10-14, for IL-8 capture beads around a spheroid: x the
    distance from it, y the height, each bead lit by its MFI and labelled with its true 3-D
    distance). Both read one value per POINT, so both live under ``error = none`` only
    (hidden otherwise — a summarised point is several rows). ``color_by`` names a numeric
    column; every marker takes its value on ``colormap`` (``color_scale`` linear | log), with
    a colour bar labelled ``color_label`` (blank: the column and its unit). The range is
    ``color_range`` where fixed, else the data's — over EVERY row, so the frames of a
    per-frame figure share one scale and a colour means the same number in each; values
    past it (zero and below on a log scale) take the end colours and the bar shows an
    arrow; a missing value is grey. With Colour by the series keep apart by marker shape,
    not hue. ``label_by`` names any column printed beside each point — a number rounded to
    ``label_decimals`` with the column's unit, text as it is — placed after the layout,
    clear of the other labels and points (``_shared/figure._place_point_labels``). Neither
    changes a measurement; both default blank, so a saved graph draws as before.
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
    fit = str(modes.get("fit", "none"))
    zero_x = str(modes.get("zero_x", "drop"))
    pn = psd = None
    if error == "pooled_sd":
        pn = FIG.column(cols, str(ctx.params.get("pool_n", "area") or "area"),
                        node="plot xy", socket="pool_n", layer=layer, numeric=True)
        sd_name = str(ctx.params.get("pool_sd", "") or "")
        if not sd_name:
            # blank = Measure's `std` column when the table has it (a factory default naming
            # it would be an orphan entry on every table without one)
            if "std_intensity" not in cols:
                raise ValueError(
                    f"plot xy: Error = pooled SD needs each row's SD — pick Pool SD, or tick "
                    f"`std` in Measure's Statistics (it writes `std_intensity`). The "
                    f"{layer!r} table carries {sorted(cols)}.")
            sd_name = "std_intensity"
        psd = FIG.column(cols, sd_name, node="plot xy", socket="pool_sd", layer=layer,
                         numeric=True)
    cvals = texts = colour = None
    if error == "none":
        cname = str(ctx.params.get("color_by", "") or "")
        lname = str(ctx.params.get("label_by", "") or "")
        if cname:
            cvals = np.asarray(FIG.column(cols, cname, node="plot xy", socket="color_by",
                                          layer=layer, numeric=True), dtype=float)
            scale = str(modes.get("color_scale", "linear"))
            rng = FIG.color_limits(cvals, FIG.parse_range(ctx.params.get("color_range", ""),
                                                          socket="color_range"), scale=scale)
            fin = cvals[np.isfinite(cvals)]
            below = bool(rng[0] is not None and fin.size and (fin < rng[0]).any())
            above = bool(rng[1] is not None and fin.size and (fin > rng[1]).any())
            unit = format_unit(unit_of(ds, domain, cname, layer), pretty=True)
            colour = {"colormap": str(modes.get("colormap", "viridis")), "scale": scale,
                      "range": rng,
                      "label": str(ctx.params.get("color_label", "") or "")
                      or (f"{cname} ({unit})" if unit else cname),
                      "extend": {(True, True): "both", (True, False): "min",
                                 (False, True): "max"}.get((below, above), "neither")}
        if lname:
            raw = FIG.column(cols, lname, node="plot xy", socket="label_by", layer=layer)
            lunit = "" if np.asarray(raw).dtype.kind in "USO" else \
                format_unit(unit_of(ds, domain, lname, layer), pretty=True)
            texts = FIG.point_text(raw, decimals=int(ctx.params.get("label_decimals", 0) or 0),
                                   unit=lunit)
    series = FIG.xy_series(x, y, g, error, log_x=log_x, log_y=log_y, pool_n=pn, pool_sd=psd,
                           color=cvals, text=texts)
    FIG.decorate_xy(series, x, y, g, fit=fit, zero_x=zero_x, log_x=log_x, log_y=log_y,
                    pool_n=pn)
    style = FIG.style_from_ctx(ctx)
    per = str(modes.get("per", "all"))
    x_range = FIG.parse_range(ctx.params.get("x_range", ""), socket="x_range")
    y_range = FIG.parse_range(ctx.params.get("y_range", ""), socket="y_range")
    spec = {
        "kind": "xy",
        "series": series,
        "draw": {"mode": str(modes.get("kind", "line_markers")),
                 "error_style": str(modes.get("error_style", "bars")),
                 "fit": fit, "zero_x": zero_x},
        "axes": {"title": str(ctx.params.get("title", "") or ""),
                 "x_label": str(ctx.params.get("x_label", "") or "") or xname,
                 "y_label": str(ctx.params.get("y_label", "") or "") or yname,
                 "log_x": log_x, "log_y": log_y,
                 "x_range": x_range, "y_range": y_range},
        "style": style,
        "source": {"domain": domain.value, "table": layer, "x": xname, "y": yname,
                   "group_by": gname, "error": error},
    }
    if colour is not None:
        spec["draw"]["color"] = colour
    x_lines = _x_lines(ctx, ds, domain, layer, cols, xname)
    if x_lines:
        spec["draw"]["x_lines"] = x_lines
        if per == "all" and None in x_range:
            # the lines are part of what the axis must show: an open end is taken over the
            # rows AND the lines, as the per-frame figures' shared axis is
            auto = FIG.axis_range(list(x) + [ln["x"] for ln in x_lines], log=log_x)
            spec["axes"]["x_range"] = [auto[i] if x_range[i] is None else x_range[i]
                                       for i in (0, 1)]
    if per in ("frame", "column"):
        # one figure per frame of the input — or per value of a column (2026-10-14: three
        # timepoints from two files, one scale) — the rows of that figure, on axes every
        # figure shares (the data's whole range unless a range is fixed), drawn when shown
        if per == "column":
            tcol, labels = _per_column(cols, str(ctx.params.get("per_by", "") or ""), layer)
            frames = max(1, len(labels))
            extra: dict = {}
        else:
            tcol = FIG.column(cols, "t", node="plot xy", socket="per", layer=layer,
                              numeric=True)
            frames = max(1, int(ds.axes.t))
            extra, labels = FIG.frame_clock(ds.metadata or {}, frames, ctx.calib("dt_s"))
        a = spec["axes"]
        if None in a["x_range"]:
            # an open end too: half-fixed ('0,') must not leave the other end to each frame
            auto = FIG.axis_range(list(x) + [ln["x"] for ln in x_lines], log=log_x)
            a["x_range"] = [auto[i] if a["x_range"][i] is None else a["x_range"][i]
                            for i in (0, 1)]
        if None in a["y_range"]:
            fixed_y = list(a["y_range"])
            vals = list(y)
            if error != "none":
                vals = []
                garr = None if g is None else np.asarray(g, dtype=object)
                tt = np.asarray(tcol, dtype=float)
                for f in range(frames):
                    sel = tt == f
                    for s in FIG.xy_series(x[sel], y[sel],
                                           None if garr is None else garr[sel], error,
                                           log_x=log_x, log_y=log_y,
                                           pool_n=None if pn is None else pn[sel],
                                           pool_sd=None if psd is None else psd[sel]):
                        vals += list(s["y"])
                        if s.get("err"):
                            vals += [v + e for v, e in zip(s["y"], s["err"])]
                            vals += [v - e for v, e in zip(s["y"], s["err"])]
            auto = FIG.axis_range(vals or list(y), log=log_y)
            a["y_range"] = [auto[i] if fixed_y[i] is None else fixed_y[i] for i in (0, 1)]
        spec.update(per="frame", error=error, frame_labels=labels,
                    rows={"x": [FIG.json_value(v) for v in x],
                          "y": [FIG.json_value(v) for v in y],
                          "g": (None if g is None else [FIG.json_value(v) for v in g]),
                          "t": [FIG.json_value(v) for v in tcol],
                          "pn": (None if pn is None else [FIG.json_value(v) for v in pn]),
                          "psd": (None if psd is None
                                  else [FIG.json_value(v) for v in psd]),
                          "c": (None if cvals is None
                                else [FIG.json_value(v) for v in cvals]),
                          "lab": texts})
        spec["series"] = []
        h, w = FIG.figure_pixels(style["width_mm"], style["height_mm"], style["dpi"])
        ctx.progress(0, 1, f"{frames} {'figures' if per == 'column' else 'frames'}, "
                           f"each drawn when shown")
        return FIG.picture_series(spec, frames, h, w, extra)
    ctx.progress(0, 1, f"drawing {sum(s['n'] for s in series)} rows")
    return FIG.picture_dataset(FIG.render_rgb(spec), spec)


def _x_lines(ctx: EvalContext, ds: Dataset, domain: Domain, layer: str, cols, xname: str):
    """``[{x, label, unit}]`` — the vertical reference lines of ``x_lines`` (2026-10-14):
    each comma entry a number (in X's unit) or a column whose MEDIAN places the line (a
    column holding one value per row — a spheroid's radius beside every bead — lands on that
    value). Labels pair with entries by position; a blank one is the column's name."""
    spec = str(ctx.params.get("x_lines", "") or "")
    toks = [t.strip() for t in spec.split(",") if t.strip()]
    if not toks:
        return []
    labels = [t.strip() for t in str(ctx.params.get("x_line_label", "") or "").split(",")]
    x_unit = format_unit(unit_of(ds, domain, xname, layer), pretty=True)
    out = []
    for i, tok in enumerate(toks):
        try:
            v, unit, name = float(tok), x_unit, ""
        except ValueError:
            col = np.asarray(FIG.column(cols, tok, node="plot xy", socket="x_lines",
                                        layer=layer, numeric=True), dtype=float)
            fin = col[np.isfinite(col)]
            if not fin.size:
                continue                       # a column with no value draws no line
            v = float(np.median(fin))
            unit, name = format_unit(unit_of(ds, domain, tok, layer), pretty=True), tok
        if not np.isfinite(v):
            continue
        lab = labels[i] if i < len(labels) and labels[i] else name
        out.append({"x": v, "label": lab, "unit": unit})
    return out


def _per_column(cols, name: str, layer: str):
    """``(figure index per row, figure titles)`` for ``per = column``: one figure per distinct
    value of ``name``, in the order the values first appear in the table (a Table Concat's
    inputs, a timepoint label), a missing value as its own last figure."""
    if not name:
        raise ValueError("plot xy: Per = column needs a column in `Per column` — the one "
                         "whose values each get a figure (a Table Concat's `condition`).")
    vals = FIG.column(cols, name, node="plot xy", socket="per_by", layer=layer)
    keys: list = []
    idx = np.zeros(len(vals), dtype=float)
    for i, v in enumerate(vals):
        k = FIG.MISSING_GROUP if FIG._is_missing(v) else FIG.json_value(v)
        if k not in keys:
            keys.append(k)
        idx[i] = keys.index(k)
    if FIG.MISSING_GROUP in keys and keys[-1] != FIG.MISSING_GROUP:
        last = keys.index(FIG.MISSING_GROUP)
        idx = np.where(idx == last, len(keys), np.where(idx > last, idx - 1, idx))
        keys = [k for k in keys if k != FIG.MISSING_GROUP] + [FIG.MISSING_GROUP]
    return idx, [str(k) for k in keys]


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
        InString("pool_n", "Pool count", field=False, default="area",
                 column_in_mode="domain", column_from="table",
                 available_in={"error": frozenset({"pooled_sd"})},
                 description="How many raw measurements each row's Y is the mean of — "
                             "Measure's `area` (its `count` stat), the pixels in each bead. "
                             "A row with more pixels weighs more in the pooled mean, so a "
                             "bead cut by the frame edge counts for less than a whole one."),
        InString("pool_sd", "Pool SD", field=False, default="",
                 column_in_mode="domain", column_from="table",
                 available_in={"error": frozenset({"pooled_sd"})},
                 description="The standard deviation of each row's raw measurements — "
                             "Measure's `std_intensity` (tick its `std` stat), the spread of "
                             "a bead's own pixels; blank uses `std_intensity` when the table "
                             "has it. It adds the spread WITHIN each row to the spread "
                             "between rows, which is what makes the bar the SD of every "
                             "pixel rather than of the bead means."),
        InString("color_by", "Colour by", field=False, default="", column_in_mode="domain",
                 column_from="table", available_in=_PER_POINT,
                 description="A numeric column that colours every point on a continuous "
                             "scale, with a colour bar — the MFI of each bead, say, over its "
                             "position. Blank colours each series in its own hue. With it the "
                             "series are told apart by marker shape. Only per-point plots "
                             "(Error = none) have it; it changes no measurement."),
        InString("color_range", "Colour range", field=False, default="",
                 available_in=_PER_POINT,
                 description="Fix the colour scale as 'low, high' (either end may be left "
                             "blank). Blank stretches it over EVERY row — all frames of a "
                             "per-frame figure share it, so one colour is one value in each. "
                             "Values past a fixed end take the end colour and the bar shows "
                             "an arrow. Fix it to compare two figures by colour. Only read "
                             "with Colour by set."),
        InString("color_label", "Colour bar label", field=False, default="",
                 available_in=_PER_POINT,
                 description="The colour bar's title. Blank uses the Colour by column's name "
                             "and its unit. Only read with Colour by set."),
        InString("label_by", "Label by", field=False, default="", column_in_mode="domain",
                 column_from="table", available_in=_PER_POINT,
                 description="A column printed as text beside every point — a number with "
                             "its unit (a distance, an id), or a text column as it is. Labels "
                             "are placed clear of each other and of the points where there "
                             "is room; a crowded one moves out on a thin leader line. Blank "
                             "prints none. Only per-point plots (Error = none) have it."),
        InString("x_lines", "Reference lines (x)", field=False, default="",
                 description="Vertical dotted reference lines, comma-separated: a number (in "
                             "X's unit) or a column whose MEDIAN places the line — a column "
                             "that holds one value beside every row, like a spheroid's radius, "
                             "lands exactly on it. Each line is labelled with its value and "
                             "unit, and the X axis widens to show it. Blank draws none. "
                             "Changes no measurement."),
        InString("x_line_label", "Reference line labels", field=False, default="",
                 description="Words for each reference line, comma-separated in the same "
                             "order as the lines ('spheroid edge'). Blank uses the column's "
                             "name (a typed number gets no words, only its value)."),
        InString("per_by", "Per column", field=False, default="", column_in_mode="domain",
                 column_from="table", available_in={"per": frozenset({"column"})},
                 description="With Per = column, the column whose values each get a figure "
                             "— a Table Concat's `condition` (one per input: three timepoints "
                             "from two files), a well, a position group. Figures follow the "
                             "order the values first appear and are titled by the value; all "
                             "share their axes and one colour scale."),
        InInt("label_decimals", "Label decimals", unit="", field=False, default=0,
              available_in=_PER_POINT,
              description="How many decimal places a numeric label keeps: 0 prints 1234 µm, "
                          "1 prints 1234.5 µm. Rounding only — the table keeps every digit. "
                          "Only read with Label by set to a numeric column."),
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
                 "pooled_sd": "Read each row as a SUMMARY of raw measurements (a bead's "
                              "pixels: Y their mean, Pool count how many, Pool SD their "
                              "spread) and pool them: the point is the mean of every pixel "
                              "at that x and the bar ± the SD of all of them together — "
                              "exactly what listing every pixel would give. Wider than sd "
                              "over the row means, because the spread inside each object "
                              "counts too. Needs Measure's count and std stats.",
             }),
        Mode("error_style", ["bars", "band"], default="bars", label="Error style",
             available_in={"error": frozenset({"sd", "sem", "ci95", "pooled_sd"})},
             description="How the spread is drawn around each mean.",
             choice_docs={
                 "bars": "An error bar with caps at every x — reads best with few x "
                         "values, and on a scatter.",
                 "band": "A translucent band between the lower and upper bounds — reads "
                         "best for a dense time course, where bars would overlap.",
             }),
        Mode("fit", list(FIG.XY_FITS), default="none", label="Trendline",
             description="Whether each series gets a fitted trendline, and which model. "
                         "Linear is fitted in the PLOTTED coordinates, so it follows Log X / "
                         "Log Y; Langmuir is fitted in data units whatever the axes. Either "
                         "reads the plot's points and changes no measurement.",
             choice_docs={
                 "none": "No trendline — the points (and their error bars) only. The "
                         "default, and what every saved graph from before trendlines draws.",
                 "linear": "A least-squares straight line per series through the points it "
                           "draws (the means when Error summarises rows), with its equation "
                           "and R² in the legend: y = b·x + a on linear axes, a power law "
                           "y = A·x^b when both axes are log. Needs two distinct x values; a "
                           "series with fewer gets no line.",
                 "langmuir": "A one-site binding curve y = Bmax·x / (Kd + x) per series, "
                             "least squares in data units on its x > 0 points (the curve "
                             "passes through the origin, so a blank stays out of it), with "
                             "Bmax, Kd and R² in the legend. For a saturating standard curve "
                             "on linear axes; Kd is in x's units. Needs two distinct x > 0.",
             }),
        Mode("zero_x", list(FIG.XY_ZERO_X), default="drop", label="x ≤ 0 rows",
             description="What happens to rows at x of zero or below — a blank or a zero "
                         "dose, which a log axis cannot place and a standard curve should "
                         "not fit through.",
             choice_docs={
                 "drop": "On a log x axis leave them out, as every log axis does; on a "
                         "linear one they stay ordinary points (and enter the trendline). "
                         "The default.",
                 "reference_line": "Draw each series' mean at x ≤ 0 as a dashed horizontal "
                                   "line in the series' colour — the blank level of a "
                                   "standard curve — on linear and log x axes alike, and "
                                   "keep those rows out of the points and the trendline.",
             }),
        Mode("per", ["all", "frame", "column"], default="all", label="Per",
             description="One figure of every row, one per frame of the input, or one per "
                         "value of a column.",
             choice_docs={**FIG.PER_DOCS,
                          "column": "One figure per value of `Per column` — small multiples "
                                    "of conditions, wells or timepoints from different "
                                    "files, which a frame axis cannot hold — on axes every "
                                    "figure shares and ONE colour scale over all their rows. "
                                    "Titled by the value; scrub them in the Viewer or export "
                                    "each with Export Figure's `frame`."}),
        Mode("colormap", list(FIG.XY_COLORMAPS), default="viridis", label="Colour map",
             available_in=_PER_POINT,
             description="The colours Colour by's values are drawn in. Only read with "
                         "Colour by set.",
             choice_docs={
                 "viridis": "Dark purple to yellow, perceptually uniform and readable in "
                            "greyscale and by colour-blind readers — the safe default.",
                 "inferno": "Black through red and orange to pale yellow — a heat scale "
                            "where high values glow; its pale top end reads best with the "
                            "points' dark outline on a white page.",
                 "magma": "Black through purple and pink to pale yellow — like inferno with "
                          "a cooler middle, perceptually uniform.",
                 "plasma": "Blue through magenta to yellow — perceptually uniform with no "
                           "near-black end, so low values stay visible as colour.",
                 "cividis": "Blue to yellow, designed to read the same for the common "
                            "colour-vision deficiencies — for figures that must be "
                            "accessible.",
                 "turbo": "A rainbow from dark blue through green to dark red — many "
                          "distinguishable levels, but not perceptually uniform, so equal "
                          "steps in value do not look equal.",
                 "coolwarm": "Blue through white to red, diverging — for values with a "
                             "meaningful middle, e.g. a change around zero; a point at the "
                             "middle is near-white.",
             }),
        Mode("color_scale", list(FIG.XY_COLOR_SCALES), default="linear",
             label="Colour scale", available_in=_PER_POINT,
             description="How Colour by's values map onto the colour map. Only read with "
                         "Colour by set.",
             choice_docs={
                 "linear": "Equal steps in value are equal steps in colour — honest for a "
                           "narrow range; a few bright outliers leave the rest in the "
                           "bottom colours.",
                 "log": "Equal RATIOS are equal steps in colour, so values spanning decades "
                        "(a few bright beads among dim ones) all show. Zero and below have "
                        "no place on it: they take the bottom colour and the bar shows an "
                        "arrow.",
             }),
        *FIG.style_modes(),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_domains=frozenset({Domain.VOXEL}),       # a picture: its image IS a Voxel layer
    fresh_output=True, meta_transform=FIG.figure_frame,
    description="A line or scatter chart of two columns of a Label, Point or Track table, "
                "one series per group, optionally averaged with error bars and fitted with "
                "a trendline (straight in the plotted axes, so log-log fits a power law) — "
                "drawn as a true-colour picture the Viewer shows and Export Figure writes.")
