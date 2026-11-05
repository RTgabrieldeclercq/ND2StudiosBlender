# Standalone interactive HTML views: Export HTML node, the Viewer's → HTML, the figure's Interactive HTML

- **Date:** 2026-11-05
- **Author:** Gabriel Declercq
- **Branch:** html-export (on nd2-acquisition-clock @445232c — the V5.00 stack and origin/Blender)
- **Base:** 445232c

## What changed

**A result can be written as one `.html` file that holds the pixels and the Viewer.** Any
browser opens it with no server and no network: the shown planes are embedded as packed
PNGs and the Viewer's controls are rebuilt in JavaScript — the M / T / Z strips with a ▶ and
an fps spinner each (a T box's hover names its acquisition time), a toggle and a histogram
LUT per channel with the lo / hi handles, the gamma dot, typed lo / hi, the colour menu
(the additive presets, a custom colour, back to the emission colour), Auto, Fit, ⊞ Split,
the ◈ Overlays popup with a tab per layer the result carries (points, labels, tracks, a
scalar wash, a measured field with its colour bar, vectors — in `nodelab_v2.overlays`'
looks, each switchable), the hover readout in the node's own pixel grid (`x 511, y 511 px ·
878.9, 878.9 µm · GFP 663 · cells 12`), the scale bar, the timestamp, wheel zoom, drag pan,
the keyboard and Save PNG. The compositing is `composite_with_clim` ported line for line;
Auto is `_auto_clim`; the histogram is `HistogramLUT`; the readout is `_hover_text`.

Three ways in, one implementation (`nodegraph/htmlview/`, Qt-free): the **Export HTML
node** (`io.write_html`, a tap like Export Movie: `path`, `title`, `channels`, `frames`,
`positions` in the frames grammar `0-99:3` / `5,9,12` / `last`, `max_px`, `budget_mb`,
`show_scalebar`, `show_time`; modes `planes` all / max / mean, `overlays` auto / none,
`existing`); the **Viewer's → HTML button** beside → Page (a dialog for the file, the
longest side, the T frames, the Z planes, the overlays and the cap; the Viewer's windows,
colours, overlay settings and frame become the page's start); and the **figure editor's
Export ▾ → Interactive HTML** (the editor's own SVG export as the page, and over every
image item a live viewer with the item's channels, windows, colours and layers; twins,
zoom popouts and cursor followers scrub together; charts answer a hover with their data;
⚙ opens a panel's controls; montages, pictures, tables and the decor stay as drawn). The
headless renderer writes the same page from `--out figure.html`.

`build_view` decides everything the page shows where the Dataset is: planes read at a
pixel budget through the pyramid or strided (never interpolated), quantised to 16 bits with
an affine back to data units, packed as two stacked grey halves (high bytes over low bytes —
a canvas returns 8-bit samples, and alpha would be premultiplied, so no alpha); the embedded
bytes are counted as they are made and the build refuses past the budget with the numbers.
The identity palette, the label source / plane and the scalar plane **moved Qt-free** from
`nodelab_v2/overlay_geometry.py` to `nodegraph/catalog/_shared/identity_palette.py`
(re-exported under the old names, nothing in the GUI changed) so the engine colours objects
the way the Viewer does.

Also: `scripts/_sync_check.py` no longer flags GitHub's own merge commits for lacking a
worklog (`git log --no-merges`).

## Why

Asked for: output figures or analysis outputs as standalone HTML files to view and
interact with — a video gaining sliders, images with overlays gaining LUTs — all in the
native ND2Studios way. A movie is a picture of the result at one window; a TIFF needs the
app; this is the result with the Viewer attached, for a collaborator, a thesis appendix, a
wiki, a reviewer. The arithmetic is ported rather than approximated because a page that
re-windows differently from the Viewer would be a page nobody could trust.

Design calls, and why: **lossless 16-bit** rather than 8-bit frames, because the LUT
handles and the hover readout work in the data's own units and an 8-bit page would lie
about both; the price is size, which is why every entry point asks about frames, planes
and the longest side and refuses past a cap. **Data overlays, not baked pixels**: the
labels travel as the raster (so fill, outline, ids and per-object colour are decided in
the page), points and tracks as rows — that is what makes them switchable and hoverable.
**The figure's page as the editor's SVG** rather than a second renderer: the static layer
is guaranteed to look like the export, and only the image items become live. **Stacked
grey halves** over interleaved RGB: PNG's row filter runs along one byte plane, measured
1.20 MB in 136 ms against 1.51 MB in 315 ms per 1024² 12-bit plane with shot noise.
**The pixel budget as a ceiling** here (the figure render tolerates twice it before
striding) because the user picks max_px for the file's size.

**Numbers** (this machine, synthetic 1024² 12-bit planes with shot noise, 50 frames ×
2 channels): build 17.4 s, 161 MB embedded (1.61 MB per plane; 2.10 raw); at 512 px:
4.4 s, 41 MB. The fixture page (32 planes of 64²): 162 KB. The JavaScript core was checked
against the Python originals with node (composite bytes within 1, `elapsed_text`,
`_scalebar`, the golden-angle hues, percentiles, the flat window); the view and the figure
pages were opened in QtWebEngine offscreen: the DOM builds, the composite lights, ArrowRight
steps T, the hover reads `x 32, y 32 px · 15.8, 15.8 µm · DAPI 2267 · GFP 62 · cells 1`.
These checks live in the scratchpad (`js_harness.js`, `web_smoke.py`), not in the gates —
the gates stay node-free and WebEngine-free.

**Not drawn** (said in the page's Overlays popup): mesh cross-sections and correlation
grids; the labels of a 3-D raster on a Z projection; a flipped image item is exported
unflipped (its `flip` travels in the data for later).

## Files

- `nodegraph/htmlview/{__init__,document,planes,png,page}.py`, `viewer.js`, `viewer.css` — **new**: the view document, plane reading / quantising / packing, the page, the browser viewer
- `nodegraph/catalog/_shared/identity_palette.py` — **new**: the identity palette, member / track tables, label and scalar planes, moved from `nodelab_v2/overlay_geometry.py` (which re-exports)
- `nodegraph/catalog/io/write_html.py` — **new** node; `nodegraph/catalog/__init__.py` — **hotspot**: `io.write_html` appended to `MODULES`
- `nodelab_v2/html_export.py` — **new**: the Viewer's state → page, the export dialog
- `nodelab_v2/viewer.py` — `export_html` signal and the → HTML button; `nodelab_v2/window.py` — `export_viewer_html`
- `nodelab_v2/figure_editor/panel.py` — `export_html`, the Export ▾ entry; `nodelab_v2/figure_render.py` — `--out x.html`
- `nodegraph/selftest.py` — **hotspot**: `test_htmlview` appended, registered after `test_figure_data`
- `scripts/_nodelab_v2_phase5_probe.py` — → HTML and Interactive HTML in `_probe_figure_editor`
- `scripts/_sync_check.py` — `--no-merges`
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` — **regenerated** (the node; the moved symbols)
- `codemap/node_roles.json` — `io.write_html` (input_output, analyze); `codemap/concepts.md` — CON-30; `codemap/workflows.md` — WF-11; `codemap/curated.lock.json`
- `MANUAL.md` — §2c item 9, §7 Channels (→ HTML), §11 a section, §15 a row

## How to verify

Wire **Export HTML** after any node, set File, run: the page opens in a browser with the
strips, LUTs, overlays and readout. Or press **→ HTML** on a Viewer showing a labelled
time series and scrub T in the page; or on a figure page **Export ▾ → Interactive HTML**
and hover a chart.

```
PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest     # test_htmlview
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png
python -m nodelab_v2.figure_render ws.nd2graph.json --page Figures --out fig.html
```

## Gates

- [x] `PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the process then segfaults at `os._exit`, the known environmental exit crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 146 ops (baseline re-saved for the new node)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-30 and WF-11 blessed 2026-11-05)
- [x] `python scripts/_sync_check.py` -> IN SYNC (on nd2-acquisition-clock @445232c, PR #3's head)
