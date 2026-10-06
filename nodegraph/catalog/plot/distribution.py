"""plot.distribution — the distribution of one table column, per group (V4.00 step 8)."""
from __future__ import annotations

import numpy as np

import nodegraph.catalog._shared.figure as FIG
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, Mode, OutDataset

_DIST_DOMAINS = ("label", "point", "track")
_DIST_KINDS = ("histogram", "kde", "box", "violin", "ecdf")
_HIST = {"kind": frozenset({"histogram"})}


def _compute_plot_distribution(ctx: EvalContext) -> Dataset:
    """How one column's values are spread, one series per ``group_by`` value.

    Resolved spec (V4.00 step 8, `build-node-v2`):

    * reads one table (``domain``, ``table`` — the only one when blank) and its ``value`` and
      optional ``group_by`` columns; non-finite values are left out;
    * ``kind``: histogram (``bins`` over the range of ALL groups, so the groups share bins;
      ``normalize`` count / density / percent), kde (a Gaussian kernel density estimate per
      group — scipy, Scott's bandwidth), box, violin, ecdf;
    * a log axis drops the values it cannot place (non-positive);
    * a PICTURE output like every plot (``_shared/figure``); whole table, no image axes.
    """
    FIG.require_matplotlib()
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    domain = Domain(str(modes.get("domain", "label")))
    layer = _resolve_layer(
        _structure_layers(ds, domain), ctx.layer("table"), node="plot distribution",
        socket="table", what=f"{domain.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {domain.value} table, or switch Domain",
        ctx=ctx)[0]
    cols = FIG.table_columns(ds, domain, layer)
    vname = str(ctx.params.get("value", "area") or "area")
    gname = str(ctx.params.get("group_by", "") or "")
    v = np.asarray(FIG.column(cols, vname, node="plot distribution", socket="value",
                              layer=layer, numeric=True), dtype=float)
    g = FIG.column(cols, gname, node="plot distribution", socket="group_by",
                   layer=layer) if gname else None
    kind = str(modes.get("kind", "histogram"))
    log_x = bool(ctx.params.get("log_x", False))
    log_y = bool(ctx.params.get("log_y", False))
    keep = np.isfinite(v)
    if (log_x and kind in ("histogram", "kde", "ecdf")) or \
            (log_y and kind in ("box", "violin")):
        keep &= v > 0
    series = FIG.distribution_series(v[keep], None if g is None else np.asarray(g)[keep])
    norm = str(modes.get("normalize", "count"))
    bins = max(1, min(500, int(ctx.params.get("bins", 30) or 30)))
    if kind in ("box", "violin"):
        xl, yl = gname, vname
    else:
        xl = vname
        yl = {"kde": "density", "ecdf": "fraction of rows <= x"}.get(
            kind, {"count": "count", "density": "density", "percent": "% of rows"}[norm])
    spec = {
        "kind": "distribution",
        "series": series,
        "draw": {"mode": kind, "bins": bins, "normalize": norm},
        "axes": {"title": str(ctx.params.get("title", "") or ""),
                 "x_label": str(ctx.params.get("x_label", "") or "") or xl,
                 "y_label": str(ctx.params.get("y_label", "") or "") or yl,
                 "log_x": log_x, "log_y": log_y,
                 "x_range": FIG.parse_range(ctx.params.get("x_range", ""), socket="x_range"),
                 "y_range": FIG.parse_range(ctx.params.get("y_range", ""), socket="y_range")},
        "style": FIG.style_from_ctx(ctx),
        "source": {"domain": domain.value, "table": layer, "value": vname,
                   "group_by": gname},
    }
    ctx.progress(0, 1, f"drawing {int(keep.sum())} values")
    return FIG.picture_dataset(FIG.render_rgb(spec), spec)


register_node(
    _compute_plot_distribution, op_key="plot.distribution", label="Plot Distribution",
    category="plot",
    inputs=[
        InDataset("data", description="The Dataset whose table is plotted — anything "
                                       "carrying a Label, Point or Track table."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen domain to plot. Blank uses the only "
                             "one on the wire, and asks when there are several."),
        InString("value", "Value", field=False, default="area", column_in_mode="domain",
                 column_from="table",
                 description="The column whose spread is drawn. The plot reads the values as "
                             "they are; it changes no measurement."),
        InString("group_by", "Group by", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="One distribution per distinct value of this column — a "
                             "`condition`, a channel, a position. Blank draws all rows as one."),
        InInt("bins", "Bins", unit="", field=False, default=30, available_in=_HIST,
              description="How many bins span the range of ALL groups (they share bins, so "
                          "bar heights compare). More bins show detail and more noise."),
        *FIG.style_sockets(),
    ],
    outputs=[OutDataset(label="Figure")],
    modes=[
        Mode("domain", list(_DIST_DOMAINS), default="label", label="Domain",
             description="Which kind of table the rows come from.",
             choice_docs=domain_docs(_DIST_DOMAINS)),
        Mode("kind", list(_DIST_KINDS), default="histogram", label="Kind",
             description="How the spread is drawn.",
             choice_docs={
                 "histogram": "Bars counting the rows in each bin, the groups overlaid "
                              "translucently — the most literal view, sensitive to Bins.",
                 "kde": "A smooth density curve per group (Gaussian kernel, Scott's "
                        "bandwidth) — easy to compare shapes, but smooths away fine structure.",
                 "box": "Median, quartiles and whiskers per group side by side, outliers as "
                        "points — compact for many groups, hides a two-peaked shape.",
                 "violin": "A mirrored density per group with its median — the box plot's "
                           "comparison with the shape left in.",
                 "ecdf": "The fraction of rows at or below each value, as a step curve per "
                         "group — no bins, no bandwidth; shifts and spreads read directly.",
             }),
        Mode("normalize", ["count", "density", "percent"], default="count",
             label="Normalize", available_in=_HIST,
             description="What a histogram bar's height means.",
             choice_docs={
                 "count": "The number of rows in the bin — honest about sample size, but a "
                          "bigger group simply draws taller bars.",
                 "density": "Scaled so each group's bars enclose an area of 1 — compares "
                            "shapes across groups of different sizes and bin widths.",
                 "percent": "Each bar as a percentage of its own group's rows — compares "
                            "groups of different sizes in units anyone reads.",
             }),
        *FIG.style_modes(),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_domains=frozenset({Domain.VOXEL}),       # a picture
    fresh_output=True, meta_transform=FIG.figure_frame,
    description="The distribution of one column of a Label, Point or Track table — "
                "histogram, density, box, violin or cumulative — one per group, drawn as a "
                "true-colour picture.")
