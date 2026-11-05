# V5.00 step 4 — charts and the Data step of the Figure Editor

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step4-figure-charts (stacked on v5-step3-figure-canvas → step 2 → step 1 → nd2-acquisition-clock)
- **Base:** 5f4d199 (step 3)

## What changed

The Analysis page's figure can now hold **charts**, drawn on the same canvas as its images, from
the tables its sources carry. A chart item (`nodelab_v2/figure_editor/items.py: ChartItem`)
paints the geometry `nodegraph.figure.charts.chart_geometry` computes in page millimetres —
axes, ticks, markers, lines, bars, boxes, violins, heat-map cells, hexagons, rose wedges,
brackets, legend and colour bar — with plain pens and the page font, so screen and export are
the same drawing. `nodelab_v2/figure_editor/charts_data.py` (new) is the bridge from a payload
to that geometry: it tabulates the payload with `nodegraph.tables.all_tables`, resolves the
chart's columns into series through `nodegraph.figure.series` (XY, time series with the frame
clock, bars, distributions incl. strip and SuperPlot, heat map, hexbin, MSD, survival, rose),
measures text with Qt's metrics of the page font, and writes a chart's values as CSV. Every
point of an XY series carries its row (`m,t,id,y,x`), so hovering a marker rings that object on
every image of the chart's source (or on the targets of a `highlight` link) and the status
line reads the data under the pointer. The panel resolves charts synchronously inside its
render refresh, keyed on the chart block, its rect, the page style and the payload, and caches
the payload's tables per run id.

The **Data step** of the design (V5.00 §4.4) is live: a figure document's `data` entries
compile into the page's hidden graph by `FigureDocument.sync_shadow` — `table.concat` when an
entry pools several sources, `analysis.if_else` for its `where` terms, `table.aggregate` for
its summary — wired in order after the shadow inputs; `run_ids` names an entry's terminal
stage so *Compute sources* fetches it like a source, and the window host adds the stages'
envelopes to the panel's catalog. `nodegraph/figure/schema.py` validates the entries
(`WHERE_OPS`, `AGG_REDUCERS`, sources named, a chart naming a missing entry refused). In the
GUI: `add_chart` (a histogram of the table's first measured column by default — something
drawn at once, to edit from), `add_data`, `promote_chart_data` (the chart form's *Filter /
summarise* button gives a chart a data entry of its own source), `set_chart_reads`,
`set_chart_table`, column pickers that list the payload's columns when in hand, else the
envelope's, else what an aggregate will produce; a column dropped on a chart becomes its Y
(or value); the start card's *Plot* and *Plot + Overlay* goals are enabled and seed a chart.
The Properties panel has the chart form (kind-dependent fields: columns, draw mode, error,
fit, test, axes, legend, colour map, typed ranges) and the data-entry form (domain, table,
`where` as `column op value; …`, match, summarise on/off with group-by, value, reducers, name).
`charts.py` re-measures the legend box after fits append their equations to the labels (the
box was measured before the fit text existed and an inside legend ran off the frame).

## Why

Step 3 drew images, overlays and decorations; a figure without a chart next to the overlay is
half a figure, and the design's whole point is that the plot and the image read the same
result in one document, with the plot's point traceable back to the object on the image. The
Data step compiles into real catalog nodes rather than filtering in the editor so that the
memo, the revision fence, LabLink and the shadow-graph sync all work unchanged — a summary is
a run id like any other and is never computed until asked. Charts resolve synchronously (not
on the render worker) because they read tables already in memory and produce a few thousand
floats; the image renders stay on the worker.

## Files

- `nodelab_v2/figure_editor/charts_data.py` — **new**: payload → tables → series → geometry, CSV
- `nodelab_v2/figure_editor/items.py` — `ChartItem`; `ImageItem.set_highlight` / `structure_to_item` / `paint_highlight`
- `nodelab_v2/figure_editor/panel.py` — `add_chart`, `add_data`, `edit_data`, `set_chart_reads`, `set_chart_table`, `promote_chart_data`, `chart_tables_of`, `chart_columns`, `_refresh_charts`, `chart_csv`, `_on_hover`, `_apply_chart_payload`; plot goals
- `nodelab_v2/figure_editor/properties.py` — `CHART_FIELDS`, `CHART_FIELDS_OF`, `_build_chart`, `_build_data`, `column` / `chart_source` / `chart_table` / `range` controls, `_parse_where`, `_parse_range`
- `nodelab_v2/figure_editor/scene.py` — `FigureView.hovered` (mouse tracking)
- `nodelab_v2/figure_editor/start_card.py` — plot / composite goals enabled
- `nodelab_v2/figure_document.py` — data stages in `sync_shadow`, `data_id`, `data_stages`, `data_terminal`, `data_table_name`, `run_ids` with data ids, `data_envelopes`
- `nodelab_v2/window.py` — `_FigureHost.envelopes` adds data envelopes; `run_ids` with data-only figures
- `nodegraph/figure/schema.py` — `WHERE_OPS`, `AGG_REDUCERS`, data-entry validation
- `nodegraph/figure/charts.py` — legend re-measured after fit labels
- `nodegraph/selftest.py` — **hotspot**: `test_figure_data` appended, registered after `test_figure_page`
- `scripts/_nodelab_v2_phase5_probe.py` — chart section inside `_probe_figure_editor`; undo/redo check now adds its own item
- `codemap/gen/*`, `codemap/STATE.md` — regenerated; `codemap/concepts.md` CON-17 names the data stages
- `MANUAL.md` — §2c charts and the Data step

## How to verify

Open an Analysis page below a page whose Output carries a Label table. On the start card pick
*Plot*: a histogram of the table's `area` lands. In Properties set Kind = XY, X = `x`, Y = `y`:
hover a marker — the status line reads its coordinates and `object <id>`, and the object is
ringed on every image of the same source on the page. Click *Filter / summarise*, tick
*Summarise*, Group by `t`, Value `area`; *Compute sources* fetches the summary and the chart
(Kind = bars, X = `t`, Y = `area_mean`) draws one bar per frame. Export PNG / SVG / PDF: the
chart is in the page.

```
PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest     # test_figure_data
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png
```

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the process still segfaults at `os._exit` *after* that line — the environmental exit crash documented in the step-3 entry, reproduced on a pristine step-2 tree; not a regression)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 145 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC — fetch and push are blocked on this machine ("SSL peer certificate or SSH remote key was not OK"); the branch is stacked locally, to be rebased and pushed once the certificate issue is fixed
