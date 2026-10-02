"""The experiment CANVAS — one blank field over the union of several files' fields.

Two nodes build one: ``view.canvas`` (N wired files at once) and ``view.overlay`` with
``canvas=union`` (grows the canvas one file at a time along a chain). They must build the
SAME canvas from the same files, so the builder lives here — the catalog forbids a node
module importing another (importing executes it, which registers that node early and welds
the two fingerprints together).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence, Tuple

from nodegraph.dataset import Dataset

#: The value :func:`nodegraph.placement.union_canvas` stamps as ``stage_layout_source`` — how
#: an Overlay recognises that its primary is ALREADY a canvas, so a chain of
#: ``canvas=union`` Overlays grows one canvas instead of nesting canvases.
CANVAS_LAYOUT = "canvas"


def is_canvas(md: Mapping[str, Any]) -> bool:
    """Whether a Dataset's metadata describes an experiment canvas (blank backdrop)."""
    return str((md or {}).get("stage_layout_source", "")) == CANVAS_LAYOUT


def build_canvas(sources: Sequence[Tuple[Mapping[str, Any], Any]], ref: Dataset, *,
                 pixel_size_um: float = 0.0, margin_um: float = 0.0,
                 flip: Tuple[bool, bool] = (False, False)) -> Dataset:
    """The blank canvas holding every ``(metadata, axes)`` source's fields, as a Dataset.

    ``ref`` is the Dataset the canvas takes its non-placement metadata and clock from (input
    0 / the Overlay's primary), through :func:`nodegraph.metadata.canvas_changes` — the same
    changes the edit-time pass applies, so payload and prediction agree (INV-04). Raises
    ``ValueError`` naming every source that cannot be placed.
    """
    from nodegraph.metadata import canvas_changes
    from nodegraph.placement import union_canvas
    from nodegraph.provider import ConstantProvider

    uc = union_canvas(list(sources), pixel_size_um=pixel_size_um, margin_um=margin_um,
                      flip=flip)
    if uc["refusals"]:
        raise ValueError("cannot build the experiment canvas:\n  - "
                         + "\n  - ".join(uc["refusals"]))
    axes = replace(ref.axes, m=1, t=int(uc["t"]), z=int(uc["z"]), c=1,
                   y=int(uc["y"]), x=int(uc["x"]))
    # a pyramid deep enough that the coarsest level fits a 1024 px display: the backdrop
    # is generated, so every level is free and the Viewer can always take a cheap overview
    levels, side = 1, max(axes.y, axes.x)
    while (side >> (levels - 1)) > 1024:
        levels += 1
    # A FRESH Dataset, not `ref` re-imaged: its layers (a mask, a label raster) describe
    # ref's grid, which this canvas is not.
    out = Dataset(axes=axes, metadata=dict(ref.metadata)).with_image(
        ConstantProvider(axes, levels=levels))
    return out.with_metadata(**canvas_changes(ref.metadata, uc))


__all__ = ["CANVAS_LAYOUT", "is_canvas", "build_canvas"]
