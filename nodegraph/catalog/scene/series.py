"""Scene: Series (``scene.series``) — a table column against time as the scene's chart."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InBool, InDataset, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _label_tables, _point_layers, _resolve_layer
from nodegraph.catalog._shared.scene import SCENE_LAYERS_KEY, appended_layers, spec_meta_transform

_REDUCERS = ("mean", "median", "sum", "count", "max", "min")


def _series_spec(params: Mapping[str, Any], modes: Mapping[str, str], *, point_layer: str = "",
                 label_layer: str = "") -> Dict[str, Any]:
    table = str(modes.get("table", "point") or "point")
    return {
        "kind": "series", "name": str(params.get("name", "") or ""), "table": table,
        "layer": str((point_layer if table == "point" else label_layer) or ""),
        "column": str(params.get("column", "mean_intensity") or "mean_intensity"),
        "reducer": str(modes.get("reducer", "mean") or "mean"),
        "per_position": bool(params.get("per_position", False)),
        "color": str(params.get("color", "#31d0c6") or "#31d0c6"),
    }


def _compute_scene_series(ctx: EvalContext) -> Dataset:
    """Add a table column, reduced per timepoint, to the scene's chart panel and hand the
    Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** scene → ``op_key="scene.series"``, category ``scene``.
    * **Data contract** a TAP on one structure table (Point or Label by the ``table`` mode,
      ``reads_domains_by_mode``): ``column`` is reduced over the rows of each timepoint
      (``reducer``), as one curve, or one curve per position. The chart's x axis is time in
      the scene's unit, and its cursor follows the page's timeline — e.g. the mean ELISA-bead
      intensity against the cell dynamics drawn above it. Names are checked here; rows are
      read by ``io.write_scene_viewer``. Edit-time and pull-time halves share
      :func:`_series_spec`.
    * **Footprint** ``TILEABLE`` — only the layer catalog is read here.
    * **Params read** ``name``, ``point_layer`` (table=point), ``label_layer`` (table=label),
      ``column``, ``per_position``, ``color``; modes ``table``, ``reducer``.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    table = str(modes.get("table", "point") or "point")
    if table == "point":
        layer, _ = _resolve_layer(
            _point_layers(ds), ctx.layer("point_layer"), node="Scene: Series", socket="point_layer",
            what="Point table", where="the `data` input",
            remedy="run a detector + Measure upstream, or set Table to 'label'", ctx=ctx)
        dom = Domain.POINT
    else:
        layer, _ = _resolve_layer(
            _label_tables(ds), ctx.layer("label_layer"), node="Scene: Series", socket="label_layer",
            what="Label table", where="the `data` input",
            remedy="run Label / Segment + Measure upstream, or set Table to 'point'", ctx=ctx)
        dom = Domain.LABEL
    column = str(ctx.params.get("column", "mean_intensity") or "mean_intensity")
    cols = {a.name for a in ds.layers_on(dom) if a.layer == layer}
    if column not in cols:
        raise ValueError(f"Scene: Series: {layer!r} has no column {column!r}; it has {sorted(cols)}")
    spec = _series_spec({
        "name": ctx.params.get("name", ""),
        "column": column, "per_position": ctx.params.get("per_position", False),
        "color": ctx.params.get("color", "#31d0c6"),
    }, modes, point_layer=layer, label_layer=layer)
    if not spec["name"]:
        spec["name"] = f"{column} ({layer})"
    return ds.with_metadata(**{SCENE_LAYERS_KEY: appended_layers(ds.metadata, spec)})


register_node(
    _compute_scene_series, op_key="scene.series", label="Scene: Series", category="scene",
    reads_domains_by_mode={"table": {"point": frozenset({Domain.POINT}),
                                     "label": frozenset({Domain.LABEL})}},
    meta_transform=spec_meta_transform(_series_spec, "scene_series", op_key="scene.series",
                                       layer_keys=("point_layer", "label_layer")),
    inputs=[
        InDataset("data", description="The stream carrying the table whose column is charted; "
                                      "handed on unchanged with the layer spec appended."),
        InString("point_layer", "Point layer", field=False, default="", layer_in=Domain.POINT,
                 available_in={"table": frozenset({"point"})},
                 description="Which Point table holds the column (e.g. detected ELISA beads "
                             "with `mean_intensity` from Measure). Empty = the only one on the "
                             "wire. Only read when Table is 'point'."),
        InString("label_layer", "Label layer", field=False, default="", layer_in=Domain.LABEL,
                 available_in={"table": frozenset({"label"})},
                 description="Which Label table holds the column (segmented objects with "
                             "Measure's columns). Empty = the only one on the wire. Only read "
                             "when Table is 'label'."),
        InString("column", "Column", field=False, default="mean_intensity",
                 description="The table column to chart against time: `mean_intensity`, "
                             "`area`, `speed`, any Measure output. Refused when the table has "
                             "no such column."),
        InBool("per_position", "One curve per position", field=False, default=False,
               description="ON draws one curve per multipoint position (the reducer over "
                           "that position's rows at each timepoint); OFF pools every "
                           "position into one curve."),
        InString("name", "Layer name", field=False, default="",
                 description="The chart's title in the page. Empty uses `column (table)`. "
                             "Cosmetic."),
        InString("color", "Colour", field=False, default="#31d0c6",
                 description="Curve colour (`#rrggbb` or a name) when the chart shows a single "
                             "curve; several curves cycle hues. Cosmetic."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("table", ["point", "label"], default="point", label="Table",
             description="Which kind of structure table holds the column.",
             choice_docs={
                 "point": "A Point table (detections such as beads or spots, with the columns "
                          "Measure added); needs the Point domain on the wire.",
                 "label": "A Label table (segmented regions with `area` and Measure's "
                          "columns); needs the Label domain on the wire.",
             }),
        Mode("reducer", list(_REDUCERS), default="mean", label="Reducer",
             description="How the rows of one timepoint collapse to one chart value.",
             choice_docs={
                 "mean": "Arithmetic mean over the rows at that timepoint — the usual "
                         "readout for an intensity; sensitive to a few bright outliers.",
                 "median": "Median over the rows: robust to outliers and dropouts, slightly "
                           "lower than the mean on a skewed bead population.",
                 "sum": "Total over the rows: scales with how many objects were found, so "
                        "detection count and per-object value are mixed.",
                 "count": "Number of rows at that timepoint, ignoring the column's value: "
                          "cell density or bead count against time.",
                 "max": "The largest value among the rows: the brightest bead / biggest "
                        "cell per timepoint.",
                 "min": "The smallest value among the rows: the dimmest bead / smallest cell per "
                        "timepoint, a floor that drops when a dropout or a dark object appears.",
             }),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Chart a table column against time in the scene viewer's panel (one value "
                "per timepoint by a reducer, optionally per position), with a cursor tied to "
                "the timeline. A tap: the Dataset passes through, the spec rides its metadata.")
