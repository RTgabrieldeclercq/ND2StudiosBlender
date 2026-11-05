# Measure pixels, pixel strip, table export

- **Date:** 2026-11-02
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Measure** (`analysis.measure`) has a third **Members** choice, `pixel`. It writes every voxel of chosen regions as a row of a new Point table (`pixels`, `z_kind = plane_index`) and marks the layer under `__pixel_tables__`. Each row carries:

- the voxel's `m t c z y x` address and `label_id`;
- µm and absolute stage position;
- `intensity` plus `<channel>_intensity` per channel;
- `time_s` when the file has a clock;
- opt-in `edge_dist_um` / `centroid_dist_um`.

The controls:

- **Pixels of** `regions` / `volume`.
- `ids` (`12, 40, 100-120`) narrows the regions.
- `labels` may name a plain mask or a drawn region; the label branch still refuses one.
- `max_rows` (default 5 M, 0 = none) refuses before anything is read.

Other user-visible changes:

- **Spreadsheet.** A pixel table opens as a **strip of its pixels** (`nodelab_v2/pixel_strip.py`). Pixels wrap line by line to the panel width; **Sort by** takes any column ↑/↓; **Colour** is one numeric column in grey; **Size** sets the square; hovering a pixel shows a rounded card with every column of its row. A Pixels/Table toggle switches to the grid. The strip scrolls virtually, so cost does not grow with row count.
- **Export.** A pixel table, or any table past 1,048,576 rows, goes to a file of its own (`<name>_<table>.<ext>`), written column-wise. Other tables keep the one long file exactly as before. CSV writing is now streamed in chunks. The GUI dialog offers Parquet first when a pixel table is present. Files are written via `.part` and renamed into place.
- **New node `io.write_table` (Export Table).** It writes the same files from inside a graph, as a pass-through tap: Format auto/csv/parquet/arrow, Layout auto/per_table, Existing overwrite/refuse.
- **Table Join** has **On** `label`: a pixel's `label_id` against its region's `id`, plus `m`, `t`.
- **Viewer.**
  - The hover line now names the region id under the pointer on each label raster.
  - A pixel table is never drawn as point markers unless the Points picker names it.
  - Pixel tables stay out of the two-set comparison and the track palette.
- **Module moves.** `nodelab_v2/tables.py` → `nodegraph/tables.py` and `nodelab_v2/export.py` → `nodegraph/table_export.py`; the old paths are re-export shims. The per-row `file`/`group` columns and the spreadsheet's file tabs are vectorized.

## Why

Requested: Measure should gather single-pixel data for a given label or volume, show it in the Measure table as a strip of pixels sortable by any column with a clean rounded hover card, and export it. The proposed export fix and the optional node were asked for too.

- **A Point table, not a new domain** (the user's call). A voxel's address is exactly the invariant `id,m,t,c,z,y,x` schema, and Plot XY, Table Aggregate, Join, Concat, the spreadsheet and export already take Point tables. `id` is a row number, not the region id, because every Point consumer reads an id as one object.
- **A metadata mark rather than a column signature.** The Viewer (one marker per pixel would stall it), the spreadsheet (strip) and the export (own file) must all agree on what a pixel table is.
- **The export had to change.** It refused Voxel layers outright. Its row-wise writer built one Python value per cell (~250 M objects at 12.6 M × 20), and its single long sheet would have padded object rows and pixel rows with each other's blank columns.
- **The tabulation moved into the engine** because `io.write_table` is a catalog node, `nodegraph` may not import `nodelab_v2`, and the node must write what the spreadsheet shows.

Hotspot files touched: `nodegraph/selftest.py` (two tests appended), `nodegraph/catalog/__init__.py` (one module appended), `codemap/gen/*` (regenerated), `scripts/catalog_baseline.json` (re-saved; the diff was exactly this change's seven entries).

## Files

- `nodegraph/catalog/analysis/measure.py` — the `pixel` branch, sockets, edit-time layer/column declarations
- `nodegraph/catalog/io/write_table.py` — new node `io.write_table`
- `nodegraph/catalog/table/join.py` — On `label`
- `nodegraph/tables.py` — moved from `nodelab_v2/tables.py`; pixel-table mark; vectorized per-M columns
- `nodegraph/table_export.py` — moved from `nodelab_v2/export.py`; own-file split, column-wise writer, streamed CSV, atomic writes
- `nodelab_v2/tables.py`, `nodelab_v2/export.py` — re-export shims
- `nodelab_v2/pixel_strip.py` — new: the strip, its sort/colour model, the hover card
- `nodelab_v2/spreadsheet.py` — strip for pixel tables, Pixels/Table toggle, vectorized file tabs
- `nodelab_v2/viewer.py` — label id in the hover line; pixel tables out of point marks / pair / palette
- `nodelab_v2/window.py` — Export… defaults to Parquet with a pixel table and reports every file
- `nodegraph/selftest.py` — `test_measure_pixels`, `test_table_export_split`
- `scripts/_nodelab_v2_phase5_probe.py` — E1c: the strip, sort, wrap, hover card, toggle
- `nodegraph/catalog/__init__.py`, `codemap/node_roles.json`, `codemap/node_demos.json`, `MANUAL.md` (§11, §15 Measure / Table Join / Export Table)
- `CLAUDE.md` — a footgun found on the way: a module-level function name duplicated across catalog modules (this change briefly had two `_target`s) makes `_codemap.py write` silently drop `params_read` from every node
- regenerated: `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json`

The same commit carries earlier uncommitted work from this branch, each part with its own worklog entry: IL-8 Run 4 two configs, the paper's Monte Carlo, overlay native resolution, bead QC / Curate Objects (point-source fit, standard curve, `table.curate_objects`, overlay, placement, three graphs). The two share `MANUAL.md`, `nodegraph/selftest.py`, `nodegraph/catalog/__init__.py`, the role/demo files and every generated file, so the gates were run on the combined tree. `CodeLog/live/crc-leading-edge/` (288 MB of evidence-log media from another session) is deliberately left untracked.

## Measured

Scale check, 2048² with 2,500 regions (2,102,500 pixel rows), this workstation, offscreen:

- Measure (pixel, with edge + centroid distance): 1.1 s.
- Spreadsheet show, sort and colour: 0.31 s; re-sort 0.13 s; a scroll repaint ~1.5 ms.
- Export: Parquet 0.8 s / 21 MB, CSV 2.6 s / 193 MB.

Two GUI defects were found and fixed on the way:

- The strip's one-row header raised the Spreadsheet's minimum width to 690 px. Tabbed with Properties, that widened the column from 376 to 439 px on a fresh window, which the GUI probe caught. Fixed with two rows and shrinkable dropdowns (now 334 px).
- The parentless hover card could be deleted before the view at shutdown. Fixed: the card is owned by the view and the hide is guarded.

## How to verify

- `PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as S; S.test_measure_pixels(); S.test_table_export_split()"`
- In the app:
  1. Segment, then Measure with Members = pixel (try `ids` read off the Viewer's hover line).
  2. Pull it; the Spreadsheet's `point · pixels` table opens as the strip.
  3. Sort by `edge_dist_um` (with Pixel geometry `edge_dist`) and hover a pixel.
  4. File → Export table… writes `<name>.parquet` plus `<name>_pixels.parquet`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` before push: 0 behind `origin/Blender` @572312a, so the rebase was a no-op; PR #1 is stale (2026-08-10) and touches none of these files
