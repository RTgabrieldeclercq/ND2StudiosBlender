# V5.00 step 5 — composite and paper-readiness of the Figure Editor

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step5-figure-paper (stacked on v5-step4-figure-charts → step 3 → step 2 → step 1 → nd2-acquisition-clock)
- **Base:** 34e5831 (step 4)

## What changed

The figure page gains the pieces that turn a draft into a submission. **Tables**: a `table`
item (`TableItem`) shows a summary of a source's table or a data entry — the columns named
(blank = every measured column) reduced per value of *Group by* with n, mean, SD, SEM,
median, min, max or sum, formatted to significant digits or fixed decimals, booktabs rules,
*Fit to content*; the *Table* goal on the start card places one. `nodegraph/figure/tables.py`
(new, Qt-free) builds the rows. **Shared legend**: a `legend` item (`LegendItem`) of an
image lists its channels as swatches in their colours and its layers as outline / marker /
line glyphs or a colour-map ramp with its limits; *Legend* on an image's form adds one.
**Re-letter panels** letters every panel in reading order (rows top to bottom, left to
right; a zoom popout is part of its parent's panel and gets none) in the page's letter
style; **Style sync** applies a page preset (`paper` 8 pt / 1 pt, `talk` 14 pt, `poster`
18 pt, `dark`) to the page and clears every item's own type size and line width back to it
(`layout.STYLE_PRESETS`, `reletter`, `style_sync`, `reading_order`). The page form lists
the **paper-readiness checks** with a button that selects the item. **Palette validation**:
`layout.palette_report` (CIE76 ΔE: series pairs ≥ 12, series vs page ≥ 25) and
`effective_palette` — on a dark page black is lifted to a legible lightness with its hue
kept and a grey ramp is re-spaced white → mid-grey (black → mid-grey on a light page), so
every series stays visible; `charts.chart_geometry` draws the effective palette, and the
selftest validates every palette on both page colours. **Export bundle** (Export ▾ →
Bundle): the PNG and an SVG plus `<stem>.figure.json`, `<stem>.legend.txt` (the generated
legend paragraph), `<stem>.provenance.json` (each source as `<page>:<name>` with run id,
session-stable lineage from the runner's engine, the file it came from with size and
mtime, channel names, the display windows every image was drawn with, the app version, the
figure digest) and `<stem>.data/` with a CSV per chart, table and heat layer —
`nodegraph/figure/bundle.py` (new, Qt-free) composes and writes them. **Capture view →
page**: a *→ Page* button in the Viewer's display tools places what the Viewer shows — its
frame, its visible region (a `rect_frac` region when it is not the whole plane), its active
channels with the windows and colours the Viewer holds (manual mode where the user set a
window) — as an image item on the first figure page that can read the node's Page Output,
adding an Analysis page when the workspace has none; the node must be published, and the
status line says so otherwise (`MainWindow.capture_view_to_page`, `FigureEditorPanel.
capture_view`).

## Why

Steps 3–4 made a figure that draws; a figure that *ships* needs the letters in order, one
type size across the page, a legend a reader can decode, the numbers behind the plots, and
a record of where every panel came from — the things the community checklists ask for and
that are lost when a figure is assembled by hand in a drawing program. The bundle is
Qt-free so the headless renderer (step 6) and LabLink can write the same files; provenance
reads the runner's engine lineage rather than re-deriving identity. The palette fix came out
of the validator: the colorblind palette's black is invisible on a dark page, and the grey
ramp's lightest step was swallowed by white paper — substituting ink broke the ramp, so a
ramp is re-spaced instead.

## Files

- `nodegraph/figure/tables.py` — **new**: `summary_table`, `table_csv`, `STATS`
- `nodegraph/figure/bundle.py` — **new**: `provenance`, `bundle_files`, `write_bundle`
- `nodegraph/figure/layout.py` — `STYLE_PRESETS`, `PANEL_KINDS`, `reading_order`, `reletter`, `style_sync`, `delta_e`, `palette_report`, `effective_palette`
- `nodegraph/figure/charts.py` — draws the effective palette
- `nodegraph/figure/__init__.py` — the two new modules
- `nodelab_v2/figure_editor/items.py` — `TableItem`, `LegendItem`
- `nodelab_v2/figure_editor/panel.py` — `add_table`, `fit_table`, `add_legend_for`, `reletter`, `sync_style`, `export_bundle`, `capture_view`, `_refresh_tables`, `_legend_entries`, `_lut_windows`, `_layer_csv`; reads-block helpers shared by charts and tables; Table goal; Bundle menu entry
- `nodelab_v2/figure_editor/properties.py` — table and legend forms, the checks list with jumps, Re-letter / Style sync / Legend / Fit-to-content buttons, `list` control
- `nodelab_v2/figure_editor/start_card.py` — Table goal enabled
- `nodelab_v2/viewer.py` — `capture_page` signal, *→ Page* button
- `nodelab_v2/window.py` — `capture_view_to_page`; `_FigureHost.lineage` / `app_version`; the viewer slot
- `nodegraph/selftest.py` — **hotspot**: `test_figure_paper` appended, registered after `test_figure_data`
- `scripts/_nodelab_v2_phase5_probe.py` — table, legend, re-letter, style sync, bundle and capture-view sections in `_probe_figure_editor`
- `codemap/gen/*`, `codemap/STATE.md` — regenerated
- `MANUAL.md` — §2c items 7–9

## How to verify

On a figure page with an image of a labelled source: Properties → *Legend* on the image;
start card *Table* (or add a table item, Reads = the source, Group by `t`, *Fit to
content*); page form → *Re-letter panels*, *Style sync* with the preset set to *talk* (the
type grows to 14 pt everywhere); the Checks group names what is missing and jumps to it.
Export ▾ → Bundle: beside the PNG land the SVG, `.figure.json`, `.legend.txt`,
`.provenance.json` and `.data/*.csv`. In the Viewer, show a published node, zoom into a
region, press *→ Page*: the figure page opens with the region as a new image item.

```
PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest     # test_figure_paper
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png
```

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the process still segfaults at `os._exit` *after* that line — the environmental exit crash documented in the step-3 entry; not a regression)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 145 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC — pushed 2026-11-05 on top of origin/Blender @572312a (ahead only, no rebase needed). The push block was Norton re-signing TLS, fixed with `git config --global http.sslBackend schannel`; see 2026-11-05_v5-stack-pushed.md
