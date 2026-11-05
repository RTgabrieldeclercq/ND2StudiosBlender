# V5 step 1: the figure core

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step1-figure-core
- **Base:** 572312ab (stacked on `nd2-acquisition-clock`, which it needs for
  `nodegraph/tables.py`; land that PR first)

## What changed

A new Qt-free package, `nodegraph/figure/`, holds everything about a **figure document**
that is not painting — step 1 of `CodeLog/ClaudesPlan/V5.00_figure_editor.md`:

- `schema.py` — the document (`nd2studios-figure/1`): sources, items (image, montage,
  chart, colorbar, histogram, scale bar, text, arrow, line, shape, letter, ortho,
  kymograph, profile …) and links (zoom, twin, lut_share, cursor, highlight, profile,
  kymograph). `normalize` fills every default and refuses a bad value by its spec path
  (`items[1].image.channels[0].rgb: …`); `canonical_json` / `digest` give the one
  spelling the render cache keys on; `effective_image` resolves twin inheritance.
- `bind.py` — read plans: the frame a page cursor resolves to, the fractional window of a
  source plane a µm rect or an object covers, and `zoom_box`: where a popout sits on its
  parent, from the two fields' stage metadata through `nodegraph.placement` (the maths
  `view.overlay` already uses). A montage tiled by position is searched across tiles.
- `derive.py` — orthogonal reslice (Z resampled to square pixels), depth- and time-coded
  projections with their RGB, kymograph and intensity profile along a line (bilinear,
  wide-line averaging), ensemble MSD with its power-law exponent, Kaplan–Meier survival,
  Pearson / Manders colocalisation.
- `series.py` — re-exports the plot nodes' helpers (same objects) and adds bar, strip,
  SuperPlot, rose, survival, MSD, hexbin series and FRAP-style normalisation.
- `stats.py` — Welch / Student / Mann–Whitney / paired / Wilcoxon / Kruskal / ANOVA /
  Tukey, correlations, stars and p text, non-crossing bracket levels.
- `charts.py` — a chart's geometry in page millimetres: scales (linear, log, time,
  category) with `matplotlib.ticker` placement, axes furniture, every mark of xy /
  timeseries / bar / distribution (histogram, kde, box, violin, ecdf, strip, superplot) /
  heatmap / hexbin / rose / survival / msd, fits, stat brackets, legend and colour-bar
  geometry; `data_at` and `nearest_mark` for hover and pick.
- `layout.py` — page presets, panel letters, snapping, align / distribute, and the
  paper-readiness `page_checks` (scale bar, inset box, grey single channel, red/green
  merge, shared display range on twins, labelled colour bar, type size, dpi, letters).
- `legend_text.py` — the legend sentence generated from the sources' envelopes: objective,
  NA, immersion RI, pinhole, zoom, pixel size, z step, interval, channels' excitation /
  emission, projections, scale bars, display ranges, overlays, charts.

`nodelab_v2/overlay_render.py` moved into the engine as
`nodegraph/catalog/_shared/overlay_math.py` and is now a re-export shim (the
`nodegraph.tables` precedent). `nodegraph/selftest.py` gains `test_figure_core`. No node
changed: the catalog snapshot is identical.

## Why

The Analysis page is to become an interactive figure canvas (the design record, rev. 2).
The canvas needs a document it can validate and hash, read plans it can hand to the
runner, geometry it can paint identically on screen and in an export, and derivations for
the representations confocal figures use that no node produces. Keeping all of it Qt-free
in the engine is what lets the LabLink worker and scripts render a figure page headless,
and what lets the selftest prove the maths without a window. The overlay maths moved down
for the same reason the tabulation did: a node cannot import the GUI package.

## Files

- `CodeLog/Updates/worklog/2026-11-03_v5-step1-figure-core.md`
- `codemap/STATE.md`, `codemap/gen/MANIFEST.json`, `codemap/gen/imports.jsonl`,
  `codemap/gen/modules.jsonl`, `codemap/gen/symbols.jsonl` (regenerated — hotspot)
- `nodegraph/catalog/_shared/overlay_math.py` (new, moved from the GUI)
- `nodegraph/figure/__init__.py`, `bind.py`, `charts.py`, `derive.py`, `layout.py`,
  `legend_text.py`, `schema.py`, `series.py`, `stats.py` (new)
- `nodegraph/selftest.py` (hotspot — one test appended, one call in `main`)
- `nodelab_v2/overlay_render.py` (shim)

## How to verify

`PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "from nodegraph import selftest; selftest.test_figure_core()"`
prints eight `[ok]` lines; the fixture is the design record's §4.2 example (a plate
montage, a popout placed by stage metadata on tile 1 at pixels 10–15 × 20–25, a twin with a
normalised speed layer, an MSD chart).

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC at session start (origin fetch refused by the machine's SSL setup; local refs used)
