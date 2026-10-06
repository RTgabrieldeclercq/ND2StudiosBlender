"""io.write_figure — write a plot node's figure to PNG, SVG, PDF or TIFF (V4.00 step 7)."""
from __future__ import annotations

import os

import nodegraph.catalog._shared.figure as FIG
from nodegraph.catalog._base import register_node
from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, Mode, OutDataset

_EXT = {"png": ".png", "svg": ".svg", "pdf": ".pdf", "tiff": ".tif"}
_KNOWN = {".png": "png", ".svg": "svg", ".pdf": "pdf", ".tif": "tiff", ".tiff": "tiff"}


def _target(path: str, fmt: str) -> str:
    """``path`` with the format's extension: added when it has none (or a foreign one);
    refused when it names ANOTHER figure format — a file called ``.svg`` holding a PNG is a
    trap for whoever opens it next."""
    if os.path.isdir(path) or path.endswith(("/", "\\")):
        raise ValueError(f"Export Figure: {path!r} is a folder — add a file name")
    stem, ext = os.path.splitext(path)
    have = _KNOWN.get(ext.lower())
    if have is None:
        return path + _EXT[fmt]
    if have != fmt:
        raise ValueError(f"Export Figure: the file {os.path.basename(path)!r} ends in {ext} "
                         f"but Format is {fmt} — change one so they agree")
    return path


def _compute_write_figure(ctx: EvalContext) -> Dataset:
    """Write the figure a ``plot.*`` node drew, and hand the picture on unchanged.

    Resolved spec (V4.00 step 7, `build-node-v2`):

    * the input must be a PICTURE carrying ``figure_spec`` (what the plot node drew); the
      figure is RENDERED AGAIN from that spec, so a raster export can use another resolution
      than the picture on screen, and SVG / PDF are true vector files with editable text;
    * ``path`` is required (an empty one is refused, as Export TIFF does); the format's
      extension is added when missing; ``existing`` decides about a file already there;
    * written atomically through ``<stem>.part<ext>``; a pass-through tap like Export TIFF.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    spec = (ds.metadata or {}).get("figure_spec")
    if not spec:
        raise ValueError("Export Figure writes the figure a Plot node drew — wire a plot.* "
                         "node (Plot XY) into it. To save an image, use Export TIFF.")
    raw = str(ctx.params.get("path", "") or "").strip().strip('"')
    if not raw:
        raise ValueError("Export Figure has no destination: set `path` (the Browse… button "
                         "beside it picks one)")
    fmt = str(modes.get("format", "png"))
    target = _target(raw, fmt)
    if str(modes.get("existing", "overwrite")) == "refuse" and os.path.exists(target):
        raise ValueError(f"Export Figure: {target} already exists and Existing is 'refuse' — "
                         f"move it, pick another name, or set Existing to overwrite")
    FIG.require_matplotlib()
    if spec.get("per") == "frame":
        # a per-frame figure: write the chosen frame (the last one when past the end)
        frames = max(1, int(ds.axes.t))
        spec = FIG.frame_spec(spec, min(max(0, int(ctx.params.get("frame", 0) or 0)),
                                        frames - 1))
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    ctx.progress(0, 1, f"writing {os.path.basename(target)}")
    # the same 50-1200 the plot's own resolution is held to (a typo is not a gigapixel render)
    dpi = (min(max(int(ctx.params.get("dpi", 300) or 300), 50), 1200)
           if fmt in ("png", "tiff") else None)
    FIG.render_file(spec, target, fmt, dpi=dpi)
    ctx.progress(1, 1, f"wrote {os.path.basename(target)}")
    return ds


register_node(
    _compute_write_figure, op_key="io.write_figure", label="Export Figure", category="io",
    inputs=[
        InDataset("data", description="The figure to write — the output of a plot node. "
                                      "It is handed on unchanged."),
        InString("path", "File", field=False, default="", path_kind="save_file",
                 path_filter="PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tif *.tiff);;"
                             "All files (*)",
                 path_hint="Browse… to choose where the figure is written",
                 description="Where the figure is written. There is no default and an empty "
                             "value is refused. Without an extension the format's is added; "
                             "an extension naming another figure format is refused."),
        InInt("frame", "Frame", unit="", field=False, default=0,
              description="Which frame of a PER-FRAME figure to write (the first is 0; past "
                          "the last, the last). A figure drawn whole has one frame and "
                          "ignores it; a movie of all frames is Export Movie's job."),
        InInt("dpi", "Resolution", unit="dpi", field=False, default=300,
              available_in={"format": frozenset({"png", "tiff"})},
              description="Pixels per inch of a PNG or TIFF. The figure is drawn again at "
                          "this resolution, so its printed size, type and line weights stay "
                          "what the plot's style set; 300 is print quality, 600 for line "
                          "art; held to 50-1200. Vector formats have no resolution."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("format", ["png", "svg", "pdf", "tiff"], default="png", label="Format",
             description="The file format. Raster formats have a resolution; vector "
                         "formats scale to any size and keep their text editable.",
             choice_docs={
                 "png": "A lossless raster image at the chosen resolution — for slides, "
                        "web pages and anything that just needs a picture.",
                 "svg": "A vector drawing whose text stays text, so labels can be edited in "
                        "Inkscape or Illustrator and it scales without blur.",
                 "pdf": "A vector page with embedded TrueType fonts — what journals ask for, "
                        "and what drops cleanly into a LaTeX document.",
                 "tiff": "An uncompressed raster image at the chosen resolution, for a "
                         "journal or a layout tool that asks for TIFF specifically.",
             }),
        Mode("existing", ["overwrite", "refuse"], default="overwrite", label="Existing",
             description="What to do when a file is already at the path.",
             choice_docs={
                 "overwrite": "Replace it — the default, since this node only runs again "
                              "when the figure changed. The old file is replaced only once "
                              "the new one is complete.",
                 "refuse": "Raise instead of touching an existing file, so a figure already "
                           "sent for review is never silently replaced.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Write the figure a plot node drew to PNG, SVG, PDF or TIFF — drawn again at "
                "the export's resolution, vector formats with editable text — and hand the "
                "picture on unchanged.")
