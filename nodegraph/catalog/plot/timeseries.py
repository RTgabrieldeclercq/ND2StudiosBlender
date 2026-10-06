"""plot.timeseries — a table column over the experiment's own time axis (V4.00 step 8)."""
from __future__ import annotations

import numpy as np

import nodegraph.catalog._shared.figure as FIG
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

_TS_DOMAINS = ("label", "point", "track")
_TS_ERRORS = ("none", "sd", "sem", "ci95")


def _compute_plot_timeseries(ctx: EvalContext) -> Dataset:
    """A column against TIME, one series per ``group_by`` value.

    Resolved spec (V4.00 step 8, `build-node-v2`):

    * reads one table (``domain``, ``table``) and its ``value``, ``t`` and optional
      ``group_by`` columns; each row's frame ``t`` is placed on the time axis;
    * ``time`` elapsed: seconds since the first frame from the file's own per-frame clock
      (``frame_time_jd``), else ``t x dt_s``, else the frame index (and the axis label says
      so) — in s, min or h by the span; clock: wall-clock time when the file has a clock;
      frame: the frame index;
    * ``error`` (default sem): the rows of a frame summarised as mean ± spread, bars or band;
    * ``per`` frame: one figure per frame — the course up to that frame, a cursor at it;
    * a PICTURE output like every plot; the whole table at once.
    """
    FIG.require_matplotlib()
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    domain = Domain(str(modes.get("domain", "label")))
    layer = _resolve_layer(
        _structure_layers(ds, domain), ctx.layer("table"), node="plot timeseries",
        socket="table", what=f"{domain.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {domain.value} table, or switch Domain",
        ctx=ctx)[0]
    cols = FIG.table_columns(ds, domain, layer)
    yname = str(ctx.params.get("value", "area") or "area")
    gname = str(ctx.params.get("group_by", "") or "")
    y = np.asarray(FIG.column(cols, yname, node="plot timeseries", socket="value",
                              layer=layer, numeric=True), dtype=float)
    tcol = np.asarray(FIG.column(cols, "t", node="plot timeseries", socket="table",
                                 layer=layer, numeric=True), dtype=float)
    g = FIG.column(cols, gname, node="plot timeseries", socket="group_by",
                   layer=layer) if gname else None
    tfin = tcol[np.isfinite(tcol)]
    frames = max(1, int(ds.axes.t), int(tfin.max()) + 1 if tfin.size else 1)
    md = ds.metadata or {}
    dt = ctx.calib("dt_s")
    times, xl, xkind = FIG.frame_times(frames, frame_time_jd=md.get("frame_time_jd"),
                                       dt_s=dt, mode=str(modes.get("time", "elapsed")))
    x = np.array([times[int(i)] if np.isfinite(i) and 0 <= int(i) < len(times) else np.nan
                  for i in tcol], dtype=float)
    if str(modes.get("time", "elapsed")) == "elapsed" and \
            str(modes.get("per", "all")) != "frame" and "time_s" in cols:
        # a Table Concat wrote each row's time on its OWN input's clock: the inputs may have
        # been imaged at different intervals, and this Dataset carries input 0's clock only
        secs = np.asarray(cols["time_s"], dtype=float)
        fin = secs[np.isfinite(secs)]
        span = float(fin.max() - fin.min()) if fin.size else 0.0
        div, unit = ((1.0, "s") if span < 90.0 else
                     (60.0, "min") if span < 5400.0 else (3600.0, "h"))
        x, xl = secs / div, f"time ({unit})"
    elif str(modes.get("time", "elapsed")) == "clock" and \
            str(modes.get("per", "all")) != "frame" and "time_jd" in cols:
        # the same, for wall-clock time: each row's own input's clock
        x = np.asarray(cols["time_jd"], dtype=float) - 2440587.5    # JD of 1970-01-01
        xl, xkind = "clock time", "clock"
    error = str(modes.get("error", "sem"))
    log_y = bool(ctx.params.get("log_y", False))
    log_x = bool(ctx.params.get("log_x", False))
    series = FIG.xy_series(x, y, g, error, log_x=log_x, log_y=log_y)
    style = FIG.style_from_ctx(ctx)
    spec = {
        "kind": "timeseries",
        "series": series,
        "draw": {"mode": str(modes.get("kind", "line")),
                 "error_style": str(modes.get("error_style", "band"))},
        "axes": {"title": str(ctx.params.get("title", "") or ""),
                 "x_label": str(ctx.params.get("x_label", "") or "") or xl,
                 "y_label": str(ctx.params.get("y_label", "") or "") or yname,
                 "log_x": log_x, "log_y": log_y, "x_kind": xkind,
                 "x_range": FIG.parse_range(ctx.params.get("x_range", ""), socket="x_range"),
                 "y_range": FIG.parse_range(ctx.params.get("y_range", ""), socket="y_range")},
        "style": style,
        "source": {"domain": domain.value, "table": layer, "value": yname,
                   "group_by": gname, "time": str(modes.get("time", "elapsed")),
                   "error": error},
    }
    if str(modes.get("per", "all")) == "frame":
        n_out = max(1, int(ds.axes.t))
        extra, labels = FIG.frame_clock(md, n_out, dt)
        a = spec["axes"]
        if a["x_range"] == [None, None]:
            a["x_range"] = FIG.axis_range(times, log=log_x)
        if a["y_range"] == [None, None]:
            a["y_range"] = FIG.axis_range(
                [v for s in series for v in s["y"]] +
                [v + e for s in series if s.get("err") for v, e in zip(s["y"], s["err"])] +
                [v - e for s in series if s.get("err") for v, e in zip(s["y"], s["err"])],
                log=log_y)
        spec.update(per="frame", error=error, frame_labels=labels, frame_x=list(times),
                    rows={"x": [FIG.json_value(v) for v in x],
                          "y": [FIG.json_value(v) for v in y],
                          "g": (None if g is None else [FIG.json_value(v) for v in g]),
                          "t": [FIG.json_value(v) for v in tcol]})
        spec["series"] = []
        h, w = FIG.figure_pixels(style["width_mm"], style["height_mm"], style["dpi"])
        ctx.progress(0, 1, f"{n_out} frames, each drawn when shown")
        return FIG.picture_series(spec, n_out, h, w, extra)
    ctx.progress(0, 1, f"drawing {sum(s['n'] for s in series)} rows over {frames} frames")
    return FIG.picture_dataset(FIG.render_rgb(spec), spec,
                               {"dt_s": float(dt)} if dt else None)


register_node(
    _compute_plot_timeseries, op_key="plot.timeseries", label="Plot Time Series",
    category="plot",
    inputs=[
        InDataset("data", description="The Dataset whose table is plotted over time — its "
                                       "frame clock comes from the file."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen domain to plot. Blank uses the only "
                             "one on the wire, and asks when there are several."),
        InString("value", "Value", field=False, default="area", column_in_mode="domain",
                 column_from="table",
                 description="The column drawn against time. The plot reads it as it is; it "
                             "changes no measurement."),
        InString("group_by", "Group by", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="One curve per distinct value of this column — a `condition`, a "
                             "position, a track id. Blank draws every row as one curve."),
        *FIG.style_sockets(),
    ],
    outputs=[OutDataset(label="Figure")],
    modes=[
        Mode("domain", list(_TS_DOMAINS), default="label", label="Domain",
             description="Which kind of table the rows come from.",
             choice_docs=domain_docs(_TS_DOMAINS)),
        Mode("time", ["elapsed", "clock", "frame"], default="elapsed", label="Time",
             description="What the horizontal axis counts.",
             choice_docs={
                 "elapsed": "Time since the first frame from the file's own per-frame clock, "
                            "else frame x interval, else the frame number — in s, min or h "
                            "chosen from the span.",
                 "clock": "The wall-clock time each frame was taken, from the file's clock — "
                          "for lining up with events in the lab; elapsed when the file has no "
                          "clock.",
                 "frame": "The frame index — when the frames are not evenly timed or the "
                          "time does not matter, only the order.",
             }),
        Mode("kind", ["line", "scatter", "line_markers"], default="line", label="Kind",
             description="How each curve is drawn.",
             choice_docs={
                 "line": "Join a curve's points by a line — the usual time course when "
                         "frames are dense.",
                 "scatter": "Markers only — every row of every frame as a point, the whole "
                            "population (best with Error none).",
                 "line_markers": "A line through markers, so each frame's point stays "
                                 "visible — for sparse time courses.",
             }),
        Mode("error", list(_TS_ERRORS), default="sem", label="Error",
             description="Whether the rows of a frame are summarised, and with what spread.",
             choice_docs={
                 "none": "Draw every row as its own point; nothing is averaged — the cloud "
                         "of all objects per frame.",
                 "sd": "The mean per frame ± one standard deviation — the spread of the "
                       "population itself, which does not shrink as n grows.",
                 "sem": "The mean per frame ± the standard error (sd/√n) — how well the mean "
                        "is known; the usual time-course band.",
                 "ci95": "The mean per frame with its 95 % confidence interval (Student's t) "
                         "— the honest band for comparing two curves.",
             }),
        Mode("error_style", ["bars", "band"], default="band", label="Error style",
             available_in={"error": frozenset({"sd", "sem", "ci95"})},
             description="How the spread is drawn around each frame's mean.",
             choice_docs={
                 "bars": "An error bar with caps at every frame — reads best with few frames.",
                 "band": "A translucent band between the bounds — reads best for a dense "
                         "time course, where bars would overlap.",
             }),
        Mode("per", ["all", "frame"], default="all", label="Per",
             description="One figure of the whole course, or one per frame with the course "
                         "drawn up to it.",
             choice_docs=FIG.PER_DOCS),
        *FIG.style_modes(),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_domains=frozenset({Domain.VOXEL}),       # a picture
    fresh_output=True, meta_transform=FIG.figure_frame_clocked,
    description="A Label, Point or Track table column over the experiment's own time axis "
                "(elapsed, wall clock or frame), one curve per group with its mean and "
                "spread — or one figure per frame, drawn as a true-colour picture.")
