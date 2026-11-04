# V5.00 step 7 — polish: pyramid windows, montage budgets, keyboard, Report, page → Movie

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step7-figure-polish (stacked on v5-step6-figure-recipes → step 5 → step 4 → step 3 → step 2 → step 1 → nd2-acquisition-clock)
- **Base:** c4fe74d (step 6)

## What changed

**Reads through the pyramid.** `render.read_plane` now sizes its pixel budget to the
WINDOW it reads (a popout of a tenth of a mosaic asks for a level ten times finer than the
whole plane would) and, when the finest level still hands back more than twice the budget
— a provider without a pyramid, a synthetic stack, an array — strides the plane down (never
interpolates) with the level shape and the window box following the stride, so overlays
still land. The panel passes a budget per render: the display's (1024 px) on screen; for an
**export** every image is re-read at the budget its page width and dpi need (capped at 4096
px) and the display renders come back on the worker once the export has returned
(`render_all_now(dpi)`, `_export_px`, `_restore_display_renders`); a **montage** splits its
budget over its columns, so a 500-frame montage composites 500 thumbnails.

**Keyboard.** On the page: Ctrl+A selects all, Esc deselects, Ctrl+D duplicates beside (fresh
ids, no letter, no links), Ctrl+Z / Ctrl+Y undo and redo, Home fits, 0 shows 1:1, Ctrl+plus /
Ctrl+minus zoom (`FigureScene.keyPressEvent` → `key_action`, `FigureEditorPanel._on_key`,
`duplicate_items`); *Duplicate* is also a button on every item's form.

**Report.** `nodegraph/figure/report.py` (new, Qt-free): `panel_captions` writes a caption
per lettered panel in reading order from the document — an image's source, channels,
frame and projection, overlays, scale bar; a chart's kind, columns, grouping, fit and test
and what it reads (a data entry's pool, filter and summary spelled out); a table's columns,
statistics and grouping — and `report_html` lays the picture, the captions, the legend
paragraph, the provenance table and the file links out as one self-contained HTML page.
`FigureEditorPanel.export_report` writes the bundle, then `<stem>.html` and
`<stem>.report.pdf` (`QTextDocument.print_`); *Report* is a goal (`schema.GOALS`) and an
entry of Export ▾.

**Page → Movie.** *Movie…* on an image (or the *Movie* goal) — `FigureEditorPanel.make_movie`
→ `_FigureHost.make_movie` — adds a Free page holding a Page Input of the image's source
wired into an Export Movie whose timeline (`movie_timeline.flat_spec`) carries the image's
channels with the windows the figure drew them with (absolute contrast per channel), its
first overlay layer, its position and projection, and its decor as annotations; the page is
shown and the Movie Editor opened on the node. The figure page is untouched.

**Perf numbers** (synthetic 512 × 512 stack, 500 frames, one channel, no pyramid, this
machine, `scratchpad/bench_montage.py`): 500 montage tiles took **12.2 s** at the display
budget before the stride (24 ms per tile, all of it compositing a full plane) and **2.7 s**
at the per-tile budget of 128 px after it (5.4 ms per tile, of which ~4 ms is the synthetic
provider generating the plane — a pyramid provider skips that). Exporting the probe's page
at 300 dpi re-reads its images at ≤ 1000 px each; no item on it exceeds the display budget.

## Why

A figure of a mosaic read a popout at the level the whole plane picked, so the closeup was
soft; a 500-frame montage composited 500 full planes for tiles a few millimetres wide; and an
export at 300 dpi drew the display render. The keyboard, the report and the movie hand-off
close the loop the design record promised: a page you can work without the mouse, a page
you can hand a reader with its captions and legend, and a page scrubbed over T that becomes
a movie without rebuilding it. Striding rather than interpolating keeps a label raster's
ids intact, the same rule LabLink's quicklooks follow.

## Files

- `nodelab_v2/figure_editor/render.py` — window-aware level pick, stride fallback
- `nodelab_v2/figure_editor/panel.py` — `MAX_EXPORT_PX`, `_export_px`, `_render_key` / `_render_fn` / `render_now` / `render_all_now` with budgets, `_restore_display_renders`, `export` at the page dpi, `_on_key` actions, `duplicate_items`, `make_movie`, `export_report`, Report / Movie goals, the Report menu entry
- `nodelab_v2/figure_editor/scene.py` — the key chords
- `nodelab_v2/figure_editor/start_card.py` — Report and Movie goals, a four-column grid
- `nodelab_v2/figure_editor/properties.py` — *Movie…* and *Duplicate* buttons
- `nodelab_v2/window.py` — `_FigureHost.make_movie`
- `nodegraph/figure/report.py` — **new**; `nodegraph/figure/schema.py` — `report` goal
- `nodegraph/selftest.py` — **hotspot**: `test_figure_polish` appended, registered after `test_figure_headless_render`
- `scripts/_nodelab_v2_phase5_probe.py` — keyboard, report and page → Movie sections in `_probe_figure_editor`
- `codemap/gen/*`, `codemap/STATE.md` — regenerated
- `MANUAL.md` — §2c item 10
- `CodeLog/ClaudesPlan/V5.00_figure_editor.md` — a status paragraph under the steps table: what landed, what deviated from the table (format stays 3.0; profile / kymograph / ortho items schema-only; highlight rings the figure's images, not the Viewer)

## How to verify

On a figure page with an image: Ctrl+D duplicates it beside; Export ▾ → Report writes the
HTML with a caption per panel and the PDF; *Movie…* on the image opens the Movie Editor on a
new page's Export Movie with the image's windows. Export a page holding a popout of a large
mosaic at 300 dpi: the popout is read at the level the page needs, not the whole plane's.

```
PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest     # test_figure_polish
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png
```

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the process still segfaults at `os._exit` *after* that line — the environmental exit crash documented in the step-3 entry; not a regression)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 145 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC — fetch and push are blocked on this machine ("SSL peer certificate or SSH remote key was not OK"); the branch is stacked locally, to be rebased and pushed once the certificate issue is fixed
