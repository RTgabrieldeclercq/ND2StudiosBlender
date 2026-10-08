"""Scene: Objects (``scene.objects``) — cells / beads / particles as spheres in the scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _label_tables, _point_layers, _resolve_layer
from nodegraph.catalog._shared.scene import SCENE_LAYERS_KEY, appended_layers, spec_meta_transform

_COLORMAPS = ("viridis", "turbo", "inferno", "magma", "plasma", "cool", "gray")
_CMAP_DOCS = {
    "viridis": "Perceptually uniform purple → green → yellow; the safe default for a "
               "measured quantity such as intensity or area.",
    "turbo": "Rainbow-like with an even luminance ramp: the most hue steps per unit, "
             "separates close values, reads badly for a mass-like quantity.",
    "inferno": "Black → red → yellow: the brightest objects blaze, the dim tail fades "
               "toward the dark page.",
    "magma": "Like inferno with a softer, pinker top end; dim objects stay a little more "
             "visible.",
    "plasma": "Blue → magenta → yellow with no near-black end, so low values stay legible "
              "on a dark page.",
    "cool": "Blue → cyan → magenta, no dark end; for sparse objects on a dark page.",
    "gray": "Black to white: the value as luminance, for a figure that reserves colour for "
            "another layer.",
}


def _objects_spec(params: Mapping[str, Any], modes: Mapping[str, str], *, point_layer: str = "",
                  label_layer: str = "") -> Dict[str, Any]:
    table = str(modes.get("table", "point") or "point")
    color_by = str(modes.get("color_by", "uniform") or "uniform")
    spec = {
        "kind": "objects", "name": str(params.get("name", "") or ""), "table": table,
        "layer": str((point_layer if table == "point" else label_layer) or ""),
        "radius_um": float(params.get("radius", 0.0) or 0.0),
        "size_scale": float(params.get("size_scale", 1.0) or 1.0),
        "max_objects": int(params.get("max_objects", 200_000) or 200_000),
        "color_by": color_by, "colormap": str(modes.get("colormap", "viridis") or "viridis"),
        "color": str(params.get("color", "#ffd43b") or "#ffd43b"),
    }
    if color_by == "value":
        spec["value_column"] = str(params.get("value_column", "") or "")
    return spec


def _compute_scene_objects(ctx: EvalContext) -> Dataset:
    """Add the objects of a Point or Label table to the scene as spheres and hand the
    Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** scene → ``op_key="scene.objects"``, category ``scene``.
    * **Data contract** a TAP on one structure table: a Point table (detections, particles,
      beads) or a Label table (segmented + measured cells), chosen by the ``table`` mode and
      its gated layer socket (``reads_domains_by_mode``). The sphere radius comes from
      ``radius`` (µm) when set, else from the table's ``area`` column (equivalent circle for
      a per-plane table, equivalent sphere for a 3-D one), else 1 µm; the colour from
      ``value_column`` through the colourmap when Colour by is 'value'. One frame per
      timepoint, so a measured timelapse is a dynamic layer. Rows are read by
      ``io.write_scene_viewer``; the table name is resolved here so it fails on this card.
      Edit-time and pull-time halves share :func:`_objects_spec`.
    * **2D/3D** none — the table's ``z_kind`` decides the radius formula.
    * **Footprint** ``TILEABLE`` — only the layer catalog is read here.
    * **Params read** ``name``, ``point_layer`` (table=point), ``label_layer``
      (table=label), ``radius``, ``size_scale``, ``max_objects``, ``color``, ``value_column``
      (color_by=value); modes ``table``, ``color_by``, ``colormap``.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    table = str(modes.get("table", "point") or "point")
    if table == "point":
        layer, _ = _resolve_layer(
            _point_layers(ds), ctx.layer("point_layer"), node="Scene: Objects",
            socket="point_layer", what="Point table", where="the `data` input",
            remedy="run a detector (spots, particles, beads) upstream, or set Table to 'label'",
            ctx=ctx)
        dom = Domain.POINT
    else:
        layer, _ = _resolve_layer(
            _label_tables(ds), ctx.layer("label_layer"), node="Scene: Objects",
            socket="label_layer", what="Label table", where="the `data` input",
            remedy="run Label / Segment (+ Measure) upstream, or set Table to 'point'", ctx=ctx)
        dom = Domain.LABEL
    color_by = str(modes.get("color_by", "uniform") or "uniform")
    value_column = str(ctx.params.get("value_column", "") or "")
    if color_by == "value":
        cols = {a.name for a in ds.layers_on(dom) if a.layer == layer}
        if not value_column:
            raise ValueError("Scene: Objects: Colour by is 'value' but `value_column` is empty "
                             f"— name a column of {layer!r}: {sorted(cols)}")
        if value_column not in cols:
            raise ValueError(f"Scene: Objects: {layer!r} has no column {value_column!r}; it has "
                             f"{sorted(cols)}")
    spec = _objects_spec({
        "name": ctx.params.get("name", ""),
        "radius": ctx.params.get("radius", 0.0), "size_scale": ctx.params.get("size_scale", 1.0),
        "max_objects": ctx.params.get("max_objects", 200_000),
        "color": ctx.params.get("color", "#ffd43b"), "value_column": value_column,
    }, modes, point_layer=layer, label_layer=layer)
    if not spec["name"]:
        spec["name"] = layer
    return ds.with_metadata(**{SCENE_LAYERS_KEY: appended_layers(ds.metadata, spec)})


register_node(
    _compute_scene_objects, op_key="scene.objects", label="Scene: Objects", category="scene",
    reads_domains_by_mode={"table": {"point": frozenset({Domain.POINT}),
                                     "label": frozenset({Domain.LABEL})}},
    meta_transform=spec_meta_transform(_objects_spec, "scene_objects", op_key="scene.objects",
                                       layer_keys=("point_layer", "label_layer")),
    inputs=[
        InDataset("data", description="The stream carrying the objects' table; handed on "
                                      "unchanged with the layer spec appended."),
        InString("point_layer", "Point layer", field=False, default="", layer_in=Domain.POINT,
                 available_in={"table": frozenset({"point"})},
                 description="Which Point table (detections, particles, beads) to draw. Empty "
                             "= the only one on the wire; two need the name. Only read when "
                             "Table is 'point'."),
        InString("label_layer", "Label layer", field=False, default="", layer_in=Domain.LABEL,
                 available_in={"table": frozenset({"label"})},
                 description="Which Label table (segmented objects with `area`, centroid and "
                             "any Measure columns) to draw. Empty = the only one on the wire. "
                             "Only read when Table is 'label'."),
        InString("name", "Layer name", field=False, default="",
                 description="The label in the page's layer panel. Empty uses the table's "
                             "name. Cosmetic."),
        InFloat("radius", "Radius", unit="um", field=False, default=0.0,
                description="Sphere radius in µm for every object. 0 = from each row's `area` "
                            "(equivalent circle per plane, equivalent sphere for a 3-D table) "
                            "and 1 µm when there is no area column. Display only."),
        InFloat("size_scale", "Size ×", unit="", field=False, default=1.0,
                description="Multiplier on the radius as drawn (the page's Size slider "
                            "changes it live). >1 exaggerates small objects so they read "
                            "from afar; it moves no measurement."),
        InString("value_column", "Value column", field=False, default="",
                 available_in={"color_by": frozenset({"value"})},
                 description="The table column that colours the spheres through the "
                             "colourmap (e.g. `mean_intensity`, `area`, `speed`). Refused when "
                             "empty or absent. Only read when Colour by is 'value'."),
        InString("color", "Colour", field=False, default="#ffd43b",
                 description="The uniform sphere colour (`#rrggbb` or a name) used when Colour "
                             "by is 'uniform'. Cosmetic."),
        InInt("max_objects", "Max objects per frame", unit="", field=False, default=200_000,
              description="Objects kept per timepoint (a random subset beyond this, fixed "
                          "seed). The page's size and frame rate scale with it."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("table", ["point", "label"], default="point", label="Table",
             description="Which kind of structure table holds the objects.",
             choice_docs={
                 "point": "A Point table: one row per detection with sub-pixel `z, y, x` "
                          "(spots, particles, beads, PIV grid points); needs the Point domain "
                          "on the wire.",
                 "label": "A Label table: one row per segmented region with its centroid, "
                          "`area` and Measure's columns, so the radius can follow the real "
                          "size; needs the Label domain on the wire.",
             }),
        Mode("color_by", ["uniform", "value", "z"], default="uniform", label="Colour by",
             description="What the sphere colour encodes.",
             choice_docs={
                 "uniform": "One colour (`color`) for every object; the colourmap is unused.",
                 "value": "A table column (`value_column`) through the colourmap between the "
                          "page's range sliders — intensity, area, speed, anything Measure "
                          "produced.",
                 "z": "Depth in the scene through the colourmap: tells the layers of a "
                      "stack apart in a top view.",
             }),
        Mode("colormap", list(_COLORMAPS), default="viridis", label="Colourmap",
             description="The scalar → colour map for value and depth colouring.",
             choice_docs=_CMAP_DOCS),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Add the objects of a Point or Label table to the scene viewer as spheres "
                "sized by radius or area and coloured by a column, one frame per timepoint. A "
                "tap: the Dataset passes through, the layer spec rides its metadata.")
