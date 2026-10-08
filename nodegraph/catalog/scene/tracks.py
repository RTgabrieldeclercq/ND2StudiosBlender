"""Scene: Tracks (``scene.tracks``) — a Track table as polylines through time in the scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import (_label_tables, _point_layers, _resolve_layer,
                                              _structure_layers)
from nodegraph.catalog._shared.scene import SCENE_LAYERS_KEY, appended_layers, spec_meta_transform

_COLORMAPS = ("turbo", "viridis", "inferno", "magma", "plasma", "cool", "gray")
_CMAP_DOCS = {
    "turbo": "Rainbow-like with an even luminance ramp: many distinct hues, so neighbouring "
             "tracks (or speeds) tell apart — the default.",
    "viridis": "Perceptually uniform purple → green → yellow; the safe choice for a figure "
               "colouring by time or speed.",
    "inferno": "Black → red → yellow: the fast / late end blazes, the slow / early end "
               "fades toward the dark page.",
    "magma": "Like inferno with a softer, pinker top end, so the slow / early segments stay a "
             "little more visible against the dark page.",
    "plasma": "Blue → magenta → yellow with no near-black end, so early or slow segments "
              "stay legible on a dark page.",
    "cool": "Blue → cyan → magenta, no dark end; for few tracks on a dark page.",
    "gray": "Black to white: the scalar as luminance, for a figure that reserves colour for "
            "another layer.",
}


def _tracks_spec(params: Mapping[str, Any], modes: Mapping[str, str], *, source: str = "",
                 point_layer: str = "", label_layer: str = "") -> Dict[str, Any]:
    members = str(modes.get("members", "point") or "point")
    return {
        "kind": "tracks", "name": str(params.get("name", "") or ""),
        "source": str(source or ""), "members": members,
        "member_layer": str((point_layer if members == "point" else label_layer) or ""),
        "tail": float(params.get("tail", 0.0) or 0.0),
        "max_tracks": int(params.get("max_tracks", 5000) or 5000),
        "color_by": str(modes.get("color_by", "track") or "track"),
        "colormap": str(modes.get("colormap", "turbo") or "turbo"),
        "color": str(params.get("color", "#e599f7") or "#e599f7"),
    }


def _compute_scene_tracks(ctx: EvalContext) -> Dataset:
    """Add a Track table to the scene as polylines and hand the Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** scene → ``op_key="scene.tracks"``, category ``scene``.
    * **Data contract** a TAP on a Track table (``reads_domains = {TRACK}``: rows of
      ``track_id, t, member_id``) joined to the member table that holds the positions — a
      Point table (``track.link`` on detections) or a Label table (``track.objects`` on
      segmented cells), chosen by ``members`` and its gated layer socket
      (``reads_domains_by_mode``). Each track becomes one polyline in µm with a speed per
      vertex; the page reveals it up to the time cursor with a comet tail of ``tail`` time
      units. Names are resolved here; rows are read by ``io.write_scene_viewer``. Edit-time
      and pull-time halves share :func:`_tracks_spec`.
    * **2D/3D** none — the member table's ``z_kind`` decides whether z is a plane or sub-pixel.
    * **Footprint** ``TILEABLE`` — only the layer catalog is read here.
    * **Params read** ``name``, ``source``, ``point_layer`` (members=point), ``label_layer``
      (members=label), ``tail``, ``max_tracks``, ``color``; modes ``members``, ``color_by``,
      ``colormap``.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    source, _ = _resolve_layer(
        _structure_layers(ds, Domain.TRACK), ctx.layer("source"), node="Scene: Tracks",
        socket="source", what="Track table", where="the `data` input",
        remedy="run Link Tracks / Track Objects upstream", ctx=ctx)
    members = str(modes.get("members", "point") or "point")
    if members == "point":
        member_layer, _ = _resolve_layer(
            _point_layers(ds), ctx.layer("point_layer"), node="Scene: Tracks", socket="point_layer",
            what="Point table", where="the `data` input",
            remedy="the tracks' positions live in the detections they link; keep that table on "
                   "the wire, or set Members to 'label'", ctx=ctx)
    else:
        member_layer, _ = _resolve_layer(
            _label_tables(ds), ctx.layer("label_layer"), node="Scene: Tracks", socket="label_layer",
            what="Label table", where="the `data` input",
            remedy="the tracks' positions live in the segmented objects they link; keep that "
                   "table on the wire, or set Members to 'point'", ctx=ctx)
    spec = _tracks_spec({
        "name": ctx.params.get("name", ""),
        "tail": ctx.params.get("tail", 0.0), "max_tracks": ctx.params.get("max_tracks", 5000),
        "color": ctx.params.get("color", "#e599f7"),
    }, modes, source=source, point_layer=member_layer, label_layer=member_layer)
    if not spec["name"]:
        spec["name"] = source
    return ds.with_metadata(**{SCENE_LAYERS_KEY: appended_layers(ds.metadata, spec)})


register_node(
    _compute_scene_tracks, op_key="scene.tracks", label="Scene: Tracks", category="scene",
    reads_domains=frozenset({Domain.TRACK}),
    reads_domains_by_mode={"members": {"point": frozenset({Domain.POINT}),
                                       "label": frozenset({Domain.LABEL})}},
    meta_transform=spec_meta_transform(_tracks_spec, "scene_tracks", op_key="scene.tracks",
                                       layer_keys=("source", "point_layer", "label_layer")),
    inputs=[
        InDataset("data", description="The stream carrying the Track table and the table its "
                                      "members come from; handed on unchanged with the layer "
                                      "spec appended."),
        InString("source", "Track layer", field=False, default="", layer_in=Domain.TRACK,
                 description="Which Track table (`track_id, t, member_id` rows) to draw. Empty "
                             "= the only one on the wire; two need the name."),
        InString("point_layer", "Member points", field=False, default="", layer_in=Domain.POINT,
                 available_in={"members": frozenset({"point"})},
                 description="The Point table the track members index (the detections that "
                             "were linked). Empty = the only one on the wire. Only read when "
                             "Members is 'point'."),
        InString("label_layer", "Member labels", field=False, default="", layer_in=Domain.LABEL,
                 available_in={"members": frozenset({"label"})},
                 description="The Label table the track members index (the segmented objects "
                             "that were linked). Empty = the only one on the wire. Only read "
                             "when Members is 'label'."),
        InString("name", "Layer name", field=False, default="",
                 description="The label in the page's layer panel. Empty uses the track "
                             "table's name. Cosmetic."),
        InFloat("tail", "Tail", unit="", field=False, default=0.0,
                description="Comet length behind the time cursor, in the scene's time unit "
                            "(seconds when the file has a frame interval, else frames). 0 = "
                            "the whole track up to the cursor. The page's Tail slider changes "
                            "it live. Display only."),
        InInt("max_tracks", "Max tracks", unit="", field=False, default=5000,
              description="Tracks kept, in table order, beyond which the rest are dropped. "
                          "The page's size scales with the total vertex count."),
        InString("color", "Colour", field=False, default="#e599f7",
                 description="The uniform track colour (`#rrggbb` or a name) used when Colour "
                             "by is 'uniform'. Cosmetic."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("members", ["point", "label"], default="point", label="Members",
             description="Which kind of table the track members live in.",
             choice_docs={
                 "point": "A Point table of detections (`track.link` on spots / particles / "
                          "beads): positions are the sub-pixel detection centres; needs the "
                          "Point domain on the wire.",
                 "label": "A Label table of segmented objects (`track.objects`): positions "
                          "are the region centroids; needs the Label domain on the wire.",
             }),
        Mode("color_by", ["track", "time", "speed", "uniform"], default="track", label="Colour by",
             description="What the polyline colour encodes.",
             choice_docs={
                 "track": "One hue per track id (hashed, so neighbouring ids differ): the "
                          "easiest way to follow one cell through a crowd.",
                 "time": "Time along each track through the colourmap, early to late: shows "
                         "direction of travel at a glance in a still frame.",
                 "speed": "Instantaneous speed (µm per time unit, forward difference) through "
                          "the colourmap: where cells run and where they stall.",
                 "uniform": "One colour (`color`) for every track; reserve the colourmap for "
                            "another layer.",
             }),
        Mode("colormap", list(_COLORMAPS), default="turbo", label="Colourmap",
             description="The scalar → colour map for track, time and speed colouring.",
             choice_docs=_CMAP_DOCS),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Add a Track table to the scene viewer as polylines revealed along the "
                "timeline with a comet tail, coloured by track, time or speed. A tap: the "
                "Dataset passes through, the layer spec rides its metadata.")
