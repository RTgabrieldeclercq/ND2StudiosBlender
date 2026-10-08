"""Scene: Vectors (``scene.vectors``) — a PIV / DVC point field as a glyph layer."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer
from nodegraph.catalog._shared.scene import SCENE_LAYERS_KEY, appended_layers, spec_meta_transform

_COLOR_BY = ("magnitude", "vertical", "uniform")
_COLORMAPS = ("turbo", "viridis", "inferno", "magma", "plasma", "cool", "gray")
_CMAP_DOCS = {
    "turbo": "Rainbow-like with an even luminance ramp: the most hue steps per unit of "
             "speed, so close speeds separate — the flow-viewer default.",
    "viridis": "Perceptually uniform purple → green → yellow; the safe choice for a figure "
               "and for colour-blind readers.",
    "inferno": "Black → red → yellow: fast vectors blaze, the slow tail fades to near-black "
               "on the dark page (use a white page with care).",
    "magma": "Like inferno with a softer, pinker top end; slow vectors stay a little more "
             "visible.",
    "plasma": "Blue → magenta → yellow with no near-black end, so slow vectors remain "
              "legible on a dark page.",
    "cool": "Blue → cyan → magenta, no dark end; for a sparse field on a dark page.",
    "gray": "Black to white: the magnitude as plain luminance, for a figure that reserves "
            "colour for another layer.",
}


def _vectors_spec(params: Mapping[str, Any], modes: Mapping[str, str], *, source: str = "") -> Dict[str, Any]:
    lines = str(modes.get("lines", "off") or "off")
    spec = {
        "kind": "vectors", "name": str(params.get("name", "") or ""),
        "source": str(source or ""),
        "min_sn": float(params.get("min_sn", 0.0) or 0.0),
        "scale": float(params.get("scale", 0.0) or 0.0),
        "max_vectors": int(params.get("max_vectors", 200_000) or 200_000),
        "color_by": str(modes.get("color_by", "magnitude") or "magnitude"),
        "colormap": str(modes.get("colormap", "turbo") or "turbo"),
        "color": str(params.get("color", "#31d0c6") or "#31d0c6"),
        "lines": lines == "on",
    }
    if lines == "on":
        spec["max_lines"] = int(params.get("max_lines", 800) or 800)
    return spec


def _compute_scene_vectors(ctx: EvalContext) -> Dataset:
    """Add a point velocity / displacement field to the scene as glyphs and hand the Dataset
    through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** scene → ``op_key="scene.vectors"``, category ``scene``.
    * **Data contract** a TAP on a Point table (``reads_domains = {POINT}``): the table
      ``source`` must carry ``vx``/``vy`` (µm/s, PIV with Velocity on) or
      ``disp_x``/``disp_y`` (µm: PIV without velocity, DIC, DVC — whose ``disp_z`` makes the
      glyphs three-dimensional). Resolved here so a wrong name fails on this card, not three
      nodes later in the exporter; the rows are read by ``io.write_scene_viewer``, one frame
      per timepoint, placed by the stream's ``scene_frame``. Edit-time and pull-time halves
      share :func:`_vectors_spec`.
    * **How it will be drawn** one tapered ribbon per vector from its grid point, length =
      field value × ``scale`` (0 = auto: the median glyph spans one grid pitch), coloured by
      magnitude / vertical component / uniform; optional streamlines through the first
      frame's field (RK2 on the binned grid).
    * **2D/3D** none — derived from the columns.
    * **Footprint** ``TILEABLE`` — only the layer catalog is read here.
    * **Params read** ``name``, ``source``, ``min_sn``, ``scale``, ``max_vectors``,
      ``color``, ``max_lines`` (lines on); modes ``color_by``, ``colormap``, ``lines``.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    source, _ = _resolve_layer(
        _point_layers(ds), ctx.layer("source"), node="Scene: Vectors", socket="source",
        what="Point table", where="the `data` input",
        remedy="this draws a PIV / DIC / DVC vector field, so run one of those upstream", ctx=ctx)
    cols = {a.name for a in ds.layers_on(Domain.POINT) if a.layer == source}
    if not (({"vx", "vy"} <= cols) or ({"disp_x", "disp_y"} <= cols)):
        raise ValueError(
            f"Scene: Vectors: Point table {source!r} carries no vectors — it needs `vx`/`vy` "
            f"(µm/s, turn on Velocity in PIV) or `disp_x`/`disp_y` (µm). Columns here: "
            f"{sorted(cols)}")
    spec = _vectors_spec({
        "name": ctx.params.get("name", ""),
        "min_sn": ctx.params.get("min_sn", 0.0), "scale": ctx.params.get("scale", 0.0),
        "max_vectors": ctx.params.get("max_vectors", 200_000),
        "color": ctx.params.get("color", "#31d0c6"), "max_lines": ctx.params.get("max_lines", 800),
    }, modes, source=source)
    if not spec["name"]:
        spec["name"] = source
    return ds.with_metadata(**{SCENE_LAYERS_KEY: appended_layers(ds.metadata, spec)})


register_node(
    _compute_scene_vectors, op_key="scene.vectors", label="Scene: Vectors", category="scene",
    reads_domains=frozenset({Domain.POINT}),
    meta_transform=spec_meta_transform(_vectors_spec, "scene_vectors", op_key="scene.vectors",
                                       layer_keys=("source",)),
    inputs=[
        InDataset("data", description="The stream carrying the point field (a PIV, DIC or DVC "
                                      "output); handed on unchanged with the layer spec appended."),
        InString("source", "Point layer", field=False, default="", layer_in=Domain.POINT,
                 description="Which Point table holds the vectors: one row per interrogation "
                             "window with `vx`/`vy` (µm/s) or `disp_x`/`disp_y`[/`disp_z`] "
                             "(µm). Empty = the only point table on the wire; two need the name."),
        InString("name", "Layer name", field=False, default="",
                 description="The label in the page's layer panel. Empty uses the table's "
                             "name. Cosmetic."),
        InFloat("min_sn", "Min S/N", unit="", field=False, default=0.0,
                description="Drop vectors whose correlation signal-to-noise (`qfactor`) is "
                            "below this. 0 keeps every vector; inert on a table without a "
                            "`qfactor` column. RAISING it removes junk from featureless "
                            "windows but thins sparse regions."),
        InFloat("scale", "Glyph scale", unit="um", field=False, default=0.0,
                description="Drawn length in µm per unit of the field (per µm/s, or per µm "
                            "of displacement). 0 = auto: the median vector spans one grid "
                            "pitch. The page's Length slider multiplies this live. Display only."),
        InInt("max_vectors", "Max vectors per frame", unit="", field=False, default=200_000,
              description="Vectors kept per timepoint (a random subset beyond this, fixed "
                          "seed). The page's size and frame rate scale with it."),
        InString("color", "Colour", field=False, default="#31d0c6",
                 description="The uniform glyph colour (`#rrggbb` or a name) used when "
                             "Colour by is 'uniform'; otherwise the colourmap applies. Cosmetic."),
        InInt("max_lines", "Max streamlines", unit="", field=False, default=800,
              available_in={"lines": frozenset({"on"})},
              description="Upper bound on streamlines integrated through the first frame's "
                          "field. MORE is denser and slower. Only read when Streamlines is on."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("color_by", list(_COLOR_BY), default="magnitude", label="Colour by",
             description="What the glyph colour encodes.",
             choice_docs={
                 "magnitude": "Speed / displacement magnitude through the colourmap between "
                              "the page's range sliders (1st–99th percentile to start).",
                 "vertical": "The vertical (z) component as a fraction of the magnitude, "
                             "mid-map = in-plane: shows where a DVC field moves up or down.",
                 "uniform": "One colour (`color`) for every glyph; reserve the colourmap for "
                            "another layer.",
             }),
        Mode("colormap", list(_COLORMAPS), default="turbo", label="Colourmap",
             description="The scalar → colour map for magnitude and vertical colouring.",
             choice_docs=_CMAP_DOCS),
        Mode("lines", ["off", "on"], default="off", label="Streamlines",
             description="Whether streamlines are integrated through the field as well.",
             choice_docs={
                 "off": "Glyphs only — the cheapest layer, and every drawn vector is a "
                        "measurement.",
                 "on": "Also integrate streamlines (RK2 on the first frame's field binned to "
                       "its grid) and draw them as ribbons: shows the flow's topology at the "
                       "cost of page size and a second of compute; they are an interpolation, "
                       "not data.",
             }),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Add a PIV / DIC / DVC point field to the scene viewer as vector glyphs (one "
                "frame per timepoint, coloured by magnitude), optionally with streamlines. A "
                "tap: the Dataset passes through, the layer spec rides its metadata.")
