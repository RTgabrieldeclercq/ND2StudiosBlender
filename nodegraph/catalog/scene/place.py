"""Scene: Place (``scene.place``) — where this stream sits in the scene's world frame."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InBool, InDataset, InFloat, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.scene import SCENE_FRAME_KEY, frame_meta_transform

_LAYOUTS = ("stage", "origin")


def _place_frame(params: Mapping[str, Any], modes: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "layout": str(modes.get("layout", "stage") or "stage"),
        "flip_x": bool(params.get("flip_x", True)),
        "flip_y": bool(params.get("flip_y", False)),
        "offset_um": [float(params.get("offset_x", 0.0) or 0.0),
                      float(params.get("offset_y", 0.0) or 0.0),
                      float(params.get("offset_z", 0.0) or 0.0)],
    }


def _compute_scene_place(ctx: EvalContext) -> Dataset:
    """Set the stream's placement for the scene viewer and hand the Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** scene (presentation bookkeeping, no pixels touched) → ``op_key="scene.place"``.
    * **Data contract** a TAP: the Dataset passes through byte-identical plus one metadata
      key, ``scene_frame`` = ``{layout, flip_x, flip_y, offset_um}``. Every ``scene.*`` layer
      downstream on this stream is placed by it in ``io.write_scene_viewer``: positions laid
      out by the stage log (Stitch's mapping, the flags are its handedness) or stacked at
      the origin, then the whole stream shifted by the offsets. Edit-time and pull-time halves
      share :func:`_place_frame`, so the inspector's prediction and the payload agree.
    * **2D/3D** no lever. **Footprint** ``TILEABLE`` — nothing is read.
    * **Params read** ``flip_x``, ``flip_y``, ``offset_x``, ``offset_y``, ``offset_z``; mode
      ``layout``.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    frame = _place_frame({
        "flip_x": ctx.params.get("flip_x", True),
        "flip_y": ctx.params.get("flip_y", False),
        "offset_x": ctx.params.get("offset_x", 0.0),
        "offset_y": ctx.params.get("offset_y", 0.0),
        "offset_z": ctx.params.get("offset_z", 0.0),
    }, modes)
    return ds.with_metadata(**{SCENE_FRAME_KEY: frame})


register_node(
    _compute_scene_place, op_key="scene.place", label="Scene: Place", category="scene",
    meta_transform=frame_meta_transform(_place_frame),
    inputs=[
        InDataset("data", description="The stream to place. Handed on unchanged; only the "
                                      "`scene_frame` metadata the scene exporter reads is set."),
        InFloat("offset_x", "Offset X", unit="um", field=False, default=0.0,
                description="Shift everything on this stream along world x (µm) in the scene — "
                            "to register a second acquisition that was taken at a different "
                            "stage origin, or to set two one-position files side by side. "
                            "0 leaves the stage/origin placement as it is. Display only."),
        InFloat("offset_y", "Offset Y", unit="um", field=False, default=0.0,
                description="As Offset X, along world y (image +y, which the page draws "
                            "pointing away from the viewer in its default view). Display only."),
        InFloat("offset_z", "Offset Z", unit="um", field=False, default=0.0,
                description="Shift along z (µm): the plane spacing is the file's `z_step_um`, "
                            "so this is where a stack's first plane sits — e.g. the height of "
                            "a cell layer above a bead volume acquired separately. Display only."),
        InBool("flip_x", "Flip X", field=False, default=True,
               description="Whether a position's image +x runs along stage −x when several "
                           "positions are laid out by their stage coordinates (the same flag "
                           "and default as Stitch — this lab's scopes). Wrong = tiles mirrored "
                           "left-right. Inert for one position and for layout 'origin'."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               description="As Flip X for the y axis: image +y along stage −y when ON. Default "
                           "OFF matches Stitch. Inert for one position and for layout 'origin'."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("layout", list(_LAYOUTS), default="stage", label="Layout",
             description="How this stream's positions are laid out in the scene.",
             choice_docs={
                 "stage": "Each position at its stage coordinates (`stage_xy_um`, mirrored "
                          "per the flip flags), one at the origin — the honest layout for a "
                          "multi-position file; refused when the stage log is missing.",
                 "origin": "Every position at (0, 0): a one-position file, or positions that "
                           "really are the same place at different times. Overlapping tiles "
                           "are drawn on top of each other.",
             }),
    ],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Place this stream in the scene viewer's world frame: positions by stage log "
                "or at the origin, Stitch's handedness flags, and a µm offset. A tap — the "
                "Dataset passes through unchanged.")
