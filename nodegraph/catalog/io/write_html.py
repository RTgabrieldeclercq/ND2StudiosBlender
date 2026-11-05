"""io.write_html — write a standalone interactive HTML view of the Dataset (2026-11-05)."""
from __future__ import annotations

import os
from typing import Optional

from nodegraph.catalog._base import register_node
from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import (Granularity, InBool, InDataset, InInt, InString, Mode,
                                OutDataset)


def _html_target(path: str) -> str:
    """``path`` with ``.html`` (``.htm`` kept), refused when it names a folder."""
    if os.path.isdir(path) or path.endswith(("/", "\\")):
        raise ValueError(f"Export HTML: {path!r} is a folder — add a file name")
    stem, ext = os.path.splitext(path)
    if ext.lower() in (".html", ".htm"):
        return path
    return path + ".html"


def _budget(raw: object) -> float:
    """The byte cap in MB: the socket's value, 250 when unset, no cap at all for 0."""
    try:
        v = float(raw) if raw not in (None, "") else 250.0
    except (TypeError, ValueError):
        v = 250.0
    return float("inf") if v <= 0 else v


def _compute_write_html(ctx: EvalContext) -> Dataset:
    """Write a standalone HTML view of the Dataset, and hand the Dataset on unchanged.

    Resolved spec (2026-11-05, `build-node-v2`):

    * **Kind** io / side effect, a tap like Export Movie and Export Table: ``Dataset -> the
      SAME Dataset``, byte-identical, no ``meta_transform``, nothing downstream can tell.
      ``WHOLE_SERIES`` with no kernel axes, because the page holds every shown frame.
    * **What the file is** — one ``.html`` that opens in any browser with no server and no
      network: the frames embedded as packed PNGs and the Viewer's controls rebuilt in
      JavaScript (:mod:`nodegraph.htmlview`): M / T / Z strips with ▶ and an fps spinner,
      a toggle and a histogram LUT per channel with lo / hi handles and a gamma dot, Auto /
      Fit / Split, the overlays popup (points, labels, tracks, a scalar wash, a field
      raster, vectors, in the Viewer's looks), the hover readout, the scale bar and the
      timestamp. The Viewer's compositing arithmetic, line for line.
    * **Size** — every plane is read at ``max_px`` through the pyramid (or strided down,
      never interpolated) and the embedded bytes are counted as they are made; past
      ``budget_mb`` the node refuses with the numbers rather than writing a page a browser
      cannot open. ``frames`` / ``positions`` / ``channels`` narrow what is embedded;
      ``planes`` can collapse Z to one projection.
    * ``path`` is required (an empty one is refused, as every export node does); the
      extension ``.html`` is added when missing; ``existing`` decides about a file already
      there; written atomically through ``<stem>.part.html``.
    """
    from nodegraph.htmlview import BudgetExceeded, build_view, view_html, write_html
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    raw = str(ctx.params.get("path", "") or "").strip().strip('"')
    if not raw:
        raise ValueError("Export HTML has no destination: set `path` (the Browse… button "
                         "beside it picks one)")
    target = _html_target(raw)
    if str(modes.get("existing", "overwrite")) == "refuse" and os.path.exists(target):
        raise ValueError(f"Export HTML: {os.path.basename(target)!r} already exists and "
                         f"Existing is 'refuse' — move it, pick another name, or set "
                         f"Existing to overwrite")
    planes = str(modes.get("planes", "all"))
    z: object = "all" if planes == "all" else planes
    title = str(ctx.params.get("title", "") or "").strip()
    if not title:
        title = str((getattr(ds, "metadata", None) or {}).get("source_name") or
                    os.path.splitext(os.path.basename(target))[0])
    md = getattr(ds, "metadata", None) or {}
    provenance = {"node": "io.write_html", "source": str(md.get("source_name") or ""),
                  "files": list(md.get("source_files") or [])[:8]}

    def progress(i: int, n: int, text: str) -> None:
        ctx.progress(i, max(n, 1), text)

    try:
        doc = build_view(
            ds, title=title, max_px=int(ctx.params.get("max_px", 1024) or 1024),
            channels=str(ctx.params.get("channels", "") or ""),
            t_frames=str(ctx.params.get("frames", "") or ""),
            m_frames=str(ctx.params.get("positions", "") or ""),
            z=z, overlays=str(modes.get("overlays", "auto")) != "none",
            budget_mb=_budget(ctx.params.get("budget_mb", 250)),
            scalebar=bool(ctx.params.get("show_scalebar", True)),
            timestamp="elapsed" if bool(ctx.params.get("show_time", True)) else "off",
            provenance=provenance, progress=progress)
    except BudgetExceeded as exc:
        raise ValueError(f"Export HTML: {exc}") from None
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    out = write_html(target, view_html(doc))
    ctx.progress(1, 1, f"wrote {os.path.basename(out)} "
                       f"({len(doc['frames'])} planes, {doc['bytes'] / 1e6:.1f} MB embedded)")
    return ds


register_node(
    _compute_write_html, op_key="io.write_html", label="Export HTML", category="io",
    inputs=[
        InDataset("data", description="The Dataset the page shows — its image at every "
                                      "shown frame, and its Point, Label, Track, scalar and "
                                      "field layers as overlays. It is handed on unchanged."),
        InString("path", "File", field=False, default="", path_kind="save_file",
                 path_filter="HTML page (*.html);;All files (*)",
                 path_hint="Browse… to choose where the page is written",
                 description="Where the page is written, on the machine that runs the graph. "
                             "There is no default and an empty value is refused. `.html` is "
                             "added when the name has no extension. The file is complete in "
                             "itself: send it, open it from a USB stick, drop it on a wiki."),
        InString("title", "Title", field=False, default="",
                 description="The heading of the page. Blank takes the source file's name, "
                             "else the page's file name. Display only; nothing else reads it."),
        InString("channels", "Channels", field=False, default="",
                 description="Which channels to embed, as indices — `0`, `0,2`, `1-3`. "
                             "EMPTY (the default) means all of them. Each embedded channel "
                             "gets its own toggle and LUT on the page; a channel left out is "
                             "not in the file at all, which is the way to halve its size."),
        InString("frames", "Frames (T)", field=False, default="",
                 description="Which timepoints to embed — `0-99`, `0-99:3` (every third), "
                             "`5,9,12`, `last`; EMPTY means every frame. The page's T strip "
                             "runs over the embedded frames only. The file grows linearly "
                             "with this count: 200 frames of a 1024 px, 2-channel series are "
                             "roughly 150 MB."),
        InString("positions", "Positions (M)", field=False, default="",
                 description="Which multipoint positions to embed, the same grammar as "
                             "Frames; EMPTY means all. A plate of 96 fields at 1024 px is "
                             "too big for one page — pick the fields the reader needs."),
        InInt("max_px", "Max px", unit="px", field=False, default=1024,
              description="Each plane is read so its longest edge is at most this many "
                          "pixels — through the pyramid, or strided down (never "
                          "interpolated, so a label raster's ids survive). 1024 is the "
                          "Viewer's own display budget and reads crisp on a screen; 2048 "
                          "doubles the detail and quadruples the file. It never upscales. "
                          "The hover readout stays in the source's own pixel grid whatever "
                          "this is."),
        InInt("budget_mb", "Max file MB", unit="", field=False, default=250,
              description="The most the page may embed, in megabytes of image data. Past it "
                          "the node refuses — with the count of planes it had read — "
                          "instead of writing a page that a browser opens slowly or not at "
                          "all. Lower it for a file that goes by e-mail; raise it for an "
                          "archive on disk; 0 removes the cap (as 0 does for Max px on "
                          "Export Movie). Nothing is written when it refuses."),
        InBool("show_scalebar", "Scale bar", field=False, default=True,
               description="Draw the scale bar, a round 1-2-5 length near a fifth of the "
                           "frame width, in the corner of the picture. Omitted when the "
                           "Dataset carries no `pixel_size_um`, as the Viewer and Export "
                           "Movie do: a bar without a calibration would be a fabrication. "
                           "The reader can switch it off on the page."),
        InBool("show_time", "Elapsed time", field=False, default=True,
               description="Show the elapsed experiment time of the shown frame, from each "
                           "frame's own acquisition clock when the source carries one, else "
                           "from the frame interval. Off, the page starts without a "
                           "timestamp; the reader can still pick elapsed, clock or frame on "
                           "the page."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("planes", ["all", "max", "mean"], default="all", label="Z",
             description="What the page holds along Z.",
             choice_docs={
                 "all": "Every plane, with a Z strip to scrub them — the Viewer's view of "
                        "a stack. The file grows with the plane count.",
                 "max": "One maximum-intensity projection per frame: a stack reads as one "
                        "picture, and the file is one plane's size per frame. Labels of a "
                        "3-D raster are not drawn on a projection.",
                 "mean": "One mean projection per frame, for a stack whose signal is "
                         "diffuse rather than punctate. Same size as max.",
             }),
        Mode("overlays", ["auto", "none"], default="auto", label="Overlays",
             description="Whether the Dataset's structure layers travel with the picture.",
             choice_docs={
                 "auto": "Every Point, Label and Track layer, a scalar wash, a measured "
                         "field's column and its vectors — each a tab in the page's "
                         "Overlays popup, in the Viewer's looks, switchable by the reader. "
                         "Mesh cross-sections and correlation grids are not drawn.",
                 "none": "The picture alone. Smaller, and the only choice for a result whose "
                         "tables you do not want to hand out with it.",
             }),
        Mode("existing", ["overwrite", "refuse"], default="overwrite", label="Existing",
             description="What to do when a file is already at the path.",
             choice_docs={
                 "overwrite": "Replace it — the default, since this node only runs again "
                              "when the data changed. The old file is replaced only once the "
                              "new one is complete.",
                 "refuse": "Raise instead of touching an existing file — for a destination "
                           "that must be written exactly once; the graph stops at this node "
                           "and nothing is written.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Write a standalone interactive HTML page of this Dataset — its frames with "
                "the Viewer's strips, channel LUTs, overlays, hover readout and scale bar "
                "rebuilt in the browser, no server needed — and hand the Dataset on unchanged.")
