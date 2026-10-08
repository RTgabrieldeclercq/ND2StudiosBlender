"""Scene: Volume (``scene.volume``) — an image channel as a volume layer of the scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.scene import SCENE_LAYERS_KEY, appended_layers, spec_meta_transform

_RENDERS = ("mip", "cloud", "slices", "iso")
_COLORMAPS = ("gray", "viridis", "inferno", "magma", "plasma", "turbo", "cool")
_CMAP_DOCS = {
    "gray": "Black to white — the plain microscope look; reads intensity most faithfully and "
            "takes any tint from the layer colour on the page.",
    "viridis": "Perceptually uniform purple → green → yellow; the safe default for a "
               "quantitative scalar and for colour-blind readers.",
    "inferno": "Black → red → yellow: bright structure on a dark field pops, the dim tail is "
               "nearly invisible — good for a sparse bead field.",
    "magma": "Like inferno with a softer, pinker top end; dim structure stays a little "
             "more visible than inferno's.",
    "plasma": "Blue → magenta → yellow; high contrast through the mid-range, no near-black "
              "end, so a dense volume stays legible.",
    "turbo": "Rainbow-like with an even luminance ramp; the most hue steps per unit, which "
             "separates close values but reads badly for a mass-like quantity.",
    "cool": "Blue → cyan → magenta, no dark end: for a faint field drawn on a dark page "
            "where a map with a black end would vanish.",
}


def _volume_spec(params: Mapping[str, Any], modes: Mapping[str, str]) -> Dict[str, Any]:
    render = str(modes.get("render", "mip") or "mip")
    spec = {
        "kind": "volume", "name": str(params.get("name", "") or ""),
        "channel": int(params.get("channel", 0) or 0), "render": render,
        "colormap": str(modes.get("colormap", "gray") or "gray"),
        "color": str(params.get("color", "#ffffff") or "#ffffff"),
        "opacity": float(params.get("opacity", 0.8)),
        "window_low": float(params.get("window_low", 0.5)),
        "window_high": float(params.get("window_high", 99.8)),
        "voxel_budget": int(params.get("voxel_budget", 2_500_000) or 2_500_000),
        "max_frames": int(params.get("max_frames", 24) or 24),
    }
    if render == "iso":
        spec["iso_level"] = float(params.get("iso_level", 90.0))
    return spec


def _compute_scene_volume(ctx: EvalContext) -> Dataset:
    """Add an image channel to the scene as a volume layer and hand the Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** scene → ``op_key="scene.volume"``, category ``scene``.
    * **Data contract** a TAP that needs the image on the wire (``reads_domains = {VOXEL}``):
      it appends ``{kind: "volume", channel, render, …}`` to the ``scene_layers`` metadata and
      changes nothing else. The pixels are read later, by ``io.write_scene_viewer``, one
      ``(m, t, c)`` volume at a time through ``get_region`` and pooled to the voxel budget;
      this node only has to know the channel exists. Edit-time and pull-time halves share
      :func:`_volume_spec`.
    * **How it will be drawn** ``mip`` / ``cloud`` / ``slices`` share one uint8 brick per
      timepoint (a WebGL2 3-D texture, intensity windowed by two percentiles over every frame
      so brightness is comparable along the timeline); ``iso`` is a marching-cubes mesh at a
      speed… intensity percentile. Positions are laid out by the stream's ``scene_frame``.
    * **2D/3D** none: a single plane is a one-plane brick (MIP = the image).
    * **Footprint** ``TILEABLE`` — the compute reads no pixels.
    * **Params read** ``name``, ``channel``, ``color``, ``opacity``, ``window_low``,
      ``window_high``, ``voxel_budget``, ``max_frames``, ``iso_level`` (iso only); modes
      ``render``, ``colormap``.
    """
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Scene: Volume needs the image on its input — this Dataset carries "
                         "no image provider (a table-only wire). Wire the loader, or the node "
                         "whose output still has the image.")
    modes = ctx.params.get("__modes__", {}) or {}
    c = int(ctx.params.get("channel", 0) or 0)
    if not (0 <= c < max(1, prov.axes.c)):
        raise ValueError(f"Scene: Volume: channel {c} is out of range for a {prov.axes.c}-channel image")
    spec = _volume_spec({
        "name": ctx.params.get("name", ""), "channel": c,
        "color": ctx.params.get("color", "#ffffff"), "opacity": ctx.params.get("opacity", 0.8),
        "window_low": ctx.params.get("window_low", 0.5), "window_high": ctx.params.get("window_high", 99.8),
        "voxel_budget": ctx.params.get("voxel_budget", 2_500_000),
        "max_frames": ctx.params.get("max_frames", 24), "iso_level": ctx.params.get("iso_level", 90.0),
    }, modes)
    if not spec["name"]:
        names = list((ds.metadata or {}).get("channel_names") or [])
        spec["name"] = str(names[c]) if c < len(names) and names[c] else f"channel {c}"
    return ds.with_metadata(**{SCENE_LAYERS_KEY: appended_layers(ds.metadata, spec)})


register_node(
    _compute_scene_volume, op_key="scene.volume", label="Scene: Volume", category="scene",
    reads_domains=frozenset({Domain.VOXEL}),
    meta_transform=spec_meta_transform(_volume_spec, "scene_volume"),
    inputs=[
        InDataset("data", description="The stream whose image becomes a volume layer; handed on "
                                      "unchanged with the layer spec appended."),
        InString("name", "Layer name", field=False, default="",
                 description="The label in the page's layer panel. Empty uses the channel's "
                             "name from the file (else `channel N`). Cosmetic."),
        InInt("channel", "Channel", unit="", field=False, default=0, pick_kind="channel",
              description="Which image channel (0-based) is drawn. Out of range is refused."),
        InString("color", "Colour", field=False, default="#ffffff",
                 description="Tint of the layer as `#rrggbb` or a name (white, cyan, red, …): "
                             "the iso-surface's colour, and the top end the gray colourmap "
                             "blends toward. Cosmetic."),
        InFloat("opacity", "Opacity", unit="", field=False, default=0.8,
                description="Starting opacity of the layer in the page (0–1); the panel's "
                            "slider changes it live. LOWER lets layers behind show through. "
                            "Display only."),
        InFloat("window_low", "Window low (percentile)", unit="", field=False, default=0.5,
                description="Intensity percentile mapped to black, taken over every exported "
                            "frame so brightness is comparable along the timeline. HIGHER "
                            "suppresses background haze in MIP/cloud renders. Display only — the "
                            "viewer's window sliders move within this range."),
        InFloat("window_high", "Window high (percentile)", unit="", field=False, default=99.8,
                description="Intensity percentile mapped to full brightness. LOWER saturates "
                            "bright beads so dimmer structure shows; 100 keeps the brightest "
                            "voxel unsaturated. Display only."),
        InFloat("iso_level", "Iso level (percentile)", unit="", field=False, default=90.0,
                available_in={"render": frozenset({"iso"})},
                description="The intensity percentile the iso-surface encloses. LOWER wraps "
                            "more of the volume (bigger, blobbier surface), HIGHER keeps only "
                            "the brightest cores. Only read when Render is 'iso'."),
        InInt("voxel_budget", "Voxel budget", unit="", field=False, default=2_500_000,
              description="Most voxels one frame of this layer may hold across all positions; "
                          "the volume is mean-pooled by integer blocks (near-isotropic in µm) "
                          "until it fits. 2.5 M ≈ 3 MB of page per frame. LOWER = smaller, "
                          "faster page, coarser volume; the report lists the block size used."),
        InInt("max_frames", "Max frames", unit="", field=False, default=24,
              description="Most timepoints exported for this layer (evenly spaced over T when "
                          "the series is longer). Each frame costs one voxel budget of page "
                          "size, so 24 frames at the default budget is ~75 MB. 1 = the first "
                          "timepoint only (a static volume)."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("render", list(_RENDERS), default="mip", label="Render",
             description="How the page draws this volume.",
             choice_docs={
                 "mip": "Maximum-intensity projection along each view ray, additive on the "
                        "page: the classic fluorescence look, bright structure visible from any "
                        "angle, no depth cue beyond the fog.",
                 "cloud": "Emission–absorption ray march: denser voxels occlude what is "
                          "behind them, so the volume reads as a solid with depth; the page's "
                          "Density slider sets how quickly it turns opaque.",
                 "slices": "Three orthogonal planes through the brick, each movable on the "
                           "page: the exact voxel values, best for reading positions against "
                           "a vector or object layer.",
                 "iso": "A marching-cubes surface at `iso_level`, lit and tinted with "
                        "`color`: the smallest page per frame (triangles, not voxels) and the "
                        "clearest shape; loses everything dimmer than the level.",
             }),
        Mode("colormap", list(_COLORMAPS), default="gray", label="Colourmap",
             description="The intensity → colour map for mip, cloud and slices (the iso "
                         "surface uses `color`).",
             choice_docs=_CMAP_DOCS),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Add an image channel to the scene viewer as a volume layer (MIP, cloud, "
                "slices or iso-surface; one brick per timepoint under a voxel budget). A tap: "
                "the Dataset passes through, the layer spec rides its metadata.")
