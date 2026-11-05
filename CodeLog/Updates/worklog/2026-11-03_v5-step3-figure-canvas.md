# V5 step 3: the figure canvas

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step3-figure-canvas
- **Base:** 88881ba (`v5-step2-figure-page`, stacked on step 1 and `nd2-acquisition-clock`)

## What changed

**An Analysis page opens the Figure Editor.** `nodelab_v2/figure_editor/` (new): a
journal-sized page on a Qt scene in page units (`scene.py`: 10 units per mm, zoom, pan,
rubber-band selection, snapping, PNG / TIFF / SVG / PDF export of the same scene);
one item per document item (`items.py`: image, montage, colour bar, histogram with the
display window marked, scale bar, text, letter, arrow, line, shape, measurement line, the
leader lines of a zoom link), movable and resizable with handles, images keeping their
aspect; the picture of an image from its source's pulled Dataset (`render.py`: the Viewer's
own compositing, one window per channel — auto percentiles, the Viewer's window, typed,
or shared — and the overlays painted by the Viewer's own `OverlayRenderer`); the Sources
panel (`sources.py`: every feeder page's Outputs with what they carry, read from their
envelopes; drag a source onto the page, a table or column onto an image); the Properties
panel (`properties.py`: declarative forms for the page and every item kind, channels and
layers as sub-forms); the start card (`start_card.py`: the goals, what each needs and what
is upstream); the link behaviours (`links.py`); and the panel that owns the working
document, the renders on a worker thread (latest request per item wins), undo / redo, the
page cursor, and the host protocol (`panel.py`).

The editing model: every edit commits the whole document through the host
(`FigureDocument.replace` validates it), the scene is rebuilt from the document, and the
panel keeps its own undo list. Alt-drag on an image makes a **popout** (a zoom link: the
box on the parent, leaders that follow the child); **Twin** shares region, frame and
channels; **Share display range** on a multi-selection gives every image one window per
channel through a LUT-share link; the toolbar's M / T / Z cursor drives every image whose
frame reads it.

**The Viewer's overlay geometry became a shared module.** `nodelab_v2/overlay_geometry.py`
(new) holds what the Viewer's `_point_marks`, `_track_paths`, `_label_source`,
`_label_plane`, `_label_keys`, `_scalar_plane`, `_mesh_sections`, `_vector_info`,
`_field_info`, `_glyph_info`, `_deformed_lines`, `_grid_rects`, `_diff_sets`,
`_member_layers`, `_track_layers`, `_best_member_layer`, `_build_palette` and
`_drop_synth` did, as functions of ``(dataset, coords, settings, palette)``; the Viewer's
methods are one-line delegations (the ragged-table flag the probe reads is kept through a
hook). `OverlayFrame` gained `stroke_scale`: 0 is the Viewer (cosmetic screen-pixel pens),
> 0 a figure (plain pens in page units × that scale, so strokes are vector in an export);
every painter's pen goes through one helper, point glyphs are drawn scaled, ids sized by
the scale and culled against the painter's clip.

`nodegraph/figure/schema.py` + `bind.py`: a `rect_frac` region (a fraction of the parent's
own plane) and the zoom box of a same-source child straight from its window.
`nodelab_v2/canvas.py`: a stacked widget shows the editor for a figure page and the graph
view otherwise; `window.py`: `_FigureHost` (the editor's whole dependency surface),
fetch routing, document-change and Viewer-look hooks, `show_node` selecting a figure
item, render-thread shutdown. MANUAL §2c describes the editor. The GUI probe gains
`_probe_figure_editor` (with `NDFIG_STOP_AT=<k>` early-return checkpoints and
`NDFIG_NO_SVG=1`, two bisect hooks used to clear the editor of the exit crash below).

Not in this step: charts (step 4), tables and *Capture view → page* (step 5), figure
recipes and the headless renderer (step 6), native-resolution image re-reads on export
(images are embedded at the display resolution, step 7).

## Why

The design record's step 3: direct manipulation of images and their overlays on a real
page, with the links the user's example needs (a metadata-placed popout, a twin with a
normalised heat layer, one display range across compared panels), and one painter for
preview and export. Reusing the Viewer's compositing and overlay renderer — rather than a
second painter — is what keeps a figure's colours, windows and looks identical to what the
Viewer shows; moving the geometry extraction out of the Viewer was the price of that, and
it also makes those functions testable without a window.

**Gate note.** The GUI probe prints `ALL PHASE-5 GUI PROBES PASSED` with every probe
passing (172 checks, the new one included), and the process then segfaults at its own
`os._exit`. The same happens on an untouched checkout of step 2 in this session (two runs
of two), with the editors' render threads torn down first, and with the SVG export (the
one new library the section loads) skipped — so it is a teardown race of the offscreen
process in this environment, not a regression of this step; earlier runs today exited 0.
The probe now also tears the editors' render threads down before exiting, the discipline
it already applies to the Movie Editor's.

## Files

- `CodeLog/Updates/worklog/2026-11-03_v5-step3-figure-canvas.md`
- `MANUAL.md` (§2c, new)
- `codemap/STATE.md`, `codemap/gen/*` (regenerated — hotspot)
- `nodegraph/figure/bind.py`, `nodegraph/figure/schema.py` (`rect_frac`)
- `nodelab_v2/canvas.py`, `nodelab_v2/window.py`
- `nodelab_v2/figure_editor/{__init__,scene,items,render,links,panel,sources,properties,start_card}.py` (new)
- `nodelab_v2/overlay_geometry.py` (new), `nodelab_v2/overlays.py`, `nodelab_v2/viewer.py`
- `scripts/_nodelab_v2_phase5_probe.py` (`_probe_figure_editor`)

## How to verify

`PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` — the line beginning
`[ok] Figure Editor (V5.00 step 3)`. In the app: File → New, load an image on Image Input,
segment on Image Processing, open the Analysis page, press *Overlay*, drag `cells` onto the
page, *Compute sources*, alt-drag a box on the image, *Export ▾ SVG*.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (printed; the process then segfaults at `os._exit` — also on the untouched step-2 tree in this session, see the gate note)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC at session start (origin fetch refused by the machine's SSL setup; local refs used)
