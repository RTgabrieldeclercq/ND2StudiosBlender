"""Experiment Canvas (``view.canvas``) — One blank field spanning every wired file's fields at their true stage positions: the primary that lets Overlays put files which share no field into one view."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import canvas_union
from nodegraph.registry import Granularity, InBool, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.canvas import build_canvas


def _compute_canvas(ctx: EvalContext) -> Dataset:
    """Build the blank canvas that holds every input's fields where the stage put them.

    Resolved spec (§0 grill, 2026-10-01)
    ------------------------------------
    * **Why** three files of one well — two 4x8 grids of the same position list 8.84 mm
      apart in X and a 7-tile column between them — share no field, so an Overlay onto any
      ONE of them draws the others nowhere: the primary defines the canvas, and none of
      these primaries' fields contains the others. The fix is not a second placement rule;
      it is a primary whose field is the UNION, after which every Overlay stage-places its
      file onto it with the existing, honest machinery (coverage, Z, T, pins, LUTs, Play
      all, zoom detail).
    * **Kind** view → ``op_key="view.canvas"``, category ``"view"``.
    * **Data contract** N Datasets (one ``multi`` socket) → ONE field (M=1, C=1) of blank
      pixels whose ``origin_um`` is the union corner, at the finest input's pixel size (or
      ``pixel_size_um``), on a uniform focus grid spanning every input's stack at the finest
      Z step, with input 0's T and clock. Axis-changing, so ``meta_transform=canvas_union``
      — which opts into every input's envelope (``wants_inputs``) and calls the same
      :func:`~nodegraph.placement.union_canvas`, so the card's size is the pull's.
    * **Pixels** none read, none held: :class:`~nodegraph.provider.ConstantProvider`
      generates the backdrop on demand at any pyramid level. A 15 x 23 mm canvas at
      1.7 µm/px is 9000 x 13500 px and costs nothing.
    * **2D/3D** no lever — the Z grid is derived from the inputs' own focus logs.
    * **Footprint** ``TILEABLE`` / ``kernel_axes=frozenset()``: the compute is metadata
      arithmetic.
    * **Refuses** an input that cannot be placed (no pixel size, no origin / stage log —
      a TIFF) and a canvas past :data:`~nodegraph.placement.CANVAS_MAX_PX` px a side,
      naming the pixel size that fits.
    """
    files = tuple(ctx.input("data") or ())
    if not files:
        raise ValueError(
            "Experiment Canvas needs at least one file: wire every acquisition you want on "
            "one canvas into `data`, then chain an Overlay per file onto this node's output.")
    # the SAME builder `view.overlay`'s canvas=union uses (`_shared/canvas.py`)
    return build_canvas([(d.metadata, d.axes) for d in files], files[0],
                        pixel_size_um=float(ctx.params.get("pixel_size_um", 0.0) or 0.0),
                        margin_um=float(ctx.params.get("margin_um", 0.0) or 0.0),
                        flip=(bool(ctx.params.get("flip_x", True)),
                              bool(ctx.params.get("flip_y", False))))


register_node(
    _compute_canvas, op_key="view.canvas", label="Experiment Canvas", category="view",
    inputs=[
        InDataset("data", multi=True, label="Files", passes_domains=False,
                  description=
                  "Every acquisition that belongs on one canvas — wire them all here. Only "
                  "their PLACEMENT is read (pixel size, stage position, focus, clock); no "
                  "pixel of theirs is copied. Input 0's clock becomes the canvas's T, so "
                  "wire first the file whose timeline the others should be paired against."),
        InFloat("pixel_size_um", "Pixel size", unit="um", field=False, default=0.0,
                description=
                "The canvas's sampling. 0 (the default) takes the FINEST input's, so no file "
                "is minified onto the canvas and zooming in resolves each file's own pixels. "
                "Raise it to make a huge canvas cheaper to look at — files finer than this "
                "are then averaged down. Display geometry only: no measurement of any "
                "Overlay's 'display' output moves; a 'resample' bake is expressed on this "
                "grid."),
        InFloat("margin_um", "Margin", unit="um", field=False, default=0.0,
                description=
                "Blank border added on every side of the union of the files' fields, so the "
                "outermost fields are not drawn flush against the edge of the view. Grows "
                "the canvas; changes no file's placement."),
        InBool("flip_x", "Flip X", field=False, default=True,
               description=
               "The canvas's left-right orientation. On (the default, matching Stitch and "
               "Overlay) runs stage +X to the RIGHT with every tile sampled in this lab's "
               "camera handedness — the rendering in which tiles join at their seams. "
               "Toggling it mirrors the WHOLE picture: every file, every tile position and "
               "every tile's pixels together, so the seams stay intact. Display geometry "
               "only; no file's placement relative to another moves."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               description=
               "The canvas's top-bottom orientation, as Flip X: off (the default) runs stage "
               "+Y down; on mirrors the whole picture top-to-bottom — positions and pixels "
               "together, seams intact."),
    ],
    outputs=[OutDataset()],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=canvas_union,
    description="One blank field spanning every wired file's fields at their true stage "
                "positions — the primary to chain Overlays onto when the files share no "
                "field (adjacent regions of a well, several wells of a plate), so all of them "
                "appear in ONE view, each exactly where the stage put it. Reads no pixels and "
                "holds none.")
