# Unique Objects, Langmuir fit, pooled-pixel SD, whole-frame regions, Measure stage handedness

- **Date:** 2026-10-13
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock (same working tree as the clock, node-queue and standard-curve-trendline changes; separate entry)
- **Base:** 572312ab

## What changed

**New node `table.unique_objects` (Unique Objects).** It finds the rows of a Label or Point
table that are one physical object seen more than once: from overlapping positions, or again
in later frames.
- **Linking.** Rows whose X/Y columns (default `stage_x_um` / `stage_y_um`) agree within
  `tolerance` µm are one object, by single linkage. Rows of different batch members never
  link. A row with no finite position is an object of its own.
- **Numbering.** Each row gets an `object_uid` (1…K by first sighting) and a 0-based
  `sighting` in `order_by` order (default `t,m`). Rows of one object in one frame share a
  sighting. It also gets `sightings`, the total.
- **Keeping.** `keep_first` N drops later sightings into a NEW table (default `unique`). The
  input table keeps every row, so its raster and overlays still match.

**`analysis.measure`: `flip_x` / `flip_y`** (default on / off = `placement.DEFAULT_FLIP`,
the Stitch / Overlay default). `stage_x_um` used to be `field corner + x_um`, i.e.
unflipped. On this scope that mirrored every stage x inside its field, so one bead seen from
two overlapping fields got two stage positions. It is now `x1 − x_um` by default.

**`analysis.label` (Connected Components): `regions` mode, `components` (default) / `whole`.**
`whole` makes all of a plane's (2D) or volume's (3D) mask ONE region.
`structure.label_components` gained the matching `whole=` kwarg. Connectivity is gated to
`components`.

**`plot.xy`:**
- **`fit = langmuir`** fits y = Bmax·x / (Kd + x) per series, by least squares in data
  units, on its x > 0 points. Bmax, Kd and R² go in the legend
  (`figure.xy_langmuir`; drawn as a curve from the origin).
- **`zero_x = reference_line` now applies on a linear x axis too.** The x ≤ 0 rows leave the
  points and the fit and become the dashed line. Its label is now "x ≤ 0 rows".
- `drop` on a linear axis is unchanged.
- **`error = pooled_sd`** with two column pickers, **Pool count** (default `area`) and
  **Pool SD** (default `std_intensity`). Each row is read as a summary of raw measurements
  (a bead's pixels). Per x, the point is the mean of every pooled measurement and the bar
  their SD:
  - M = Σnᵢmᵢ/N;
  - var = [Σ(nᵢ−1)sᵢ² + Σnᵢ(mᵢ−M)²]/(N−1);
  - exactly what listing every pixel gives (`figure.pooled_stats`);
  - `n_pooled` per point counts the measurements;
  - the blank reference line is pixel-weighted too.

**`analysis.measure` `stats`: `std` → `std_intensity`**, the per-region sample SD
(n − 1, NaN for one voxel). This comes from a new `std` reducer in the Voxel→Label bridge
(`bridges._group_reduce`, two-pass, so a bright region with a small spread keeps it).

**Demo harness (`nodelab_v2/demo_recipes.py`).** `_written_layers` now includes the names a
node announces through `extra_layers`. Table Aggregate / Join / Concat demos showed the
larger INPUT table instead of their own output. All four table nodes now show their own.

## Why

The user asked to:
- redo the IL-8 bead standard curves with a Langmuir fit instead of log–log;
- detect beads on Cy5 (their "Cy5 Bead Threshold" graph);
- record each bead's centroid;
- keep each bead's first 5 imaging occurrences and discard the rest, "as some beads get
  scanned in multiple M positions";
- fix the y axis at 4095.

Each change came out of that request:
- **Unique Objects.** Nothing in the catalog could say two rows are the same object across
  positions. Track Objects links frame to frame within one position.
- **Measure flip.** The dedupe could not work while one bead's two views disagreed by up to a
  field width. Measured on the IL-8 data with +x → −x: cross-field pairs coincide within 3 µm
  (median 0.74 µm); unflipped, they didn't.
- **Regions = whole.** The background (frame minus the bead discs ∪ the Cy5 mask) was split
  into pockets in 4–7 frames per configuration. Connected Components then gave two background
  rows for that frame, and Table Join On `m_t` refuses an ambiguous key. Aggregating two
  tables and dividing would also work, but takes four extra nodes for what is one region by
  definition.
- **Langmuir in data units.** The model is a statement about the numbers, so logging an axis
  must not change Bmax and Kd.
- **Pooled SD.** The user then corrected the error bars. They were the SEM of bead means
  (between-bead variance). The user wants the spread of every pixel of every bead, pooled:
  "if a bead has 9 pixels then you collect 10 beads worth of data to get 90 measurements".
  - A per-pixel table would do this literally, but it is ~400k rows per configuration
    flowing through Join / Select Group / Concat.
  - Pooling (count, mean, SD) per bead is exact and costs two columns.
  - On the IL-8 data, 81–98 % of the pooled variance is within beads.

**Why tolerance 5 µm in the graph.** Two distinct beads in one frame are never closer than
9 µm, and 6 µm already chains two beads into one. Below 5 µm, edge-clipped views of one bead
split off as extra beads (config 2: 413 beads at 3 µm, 399 at 5 µm, 394 at 7 µm).

## How to verify

- **Automated:** `selftest.test_unique_objects_langmuir` covers linking, sighting order,
  keep_first, the refusals, Regions = whole through the Engine, Langmuir recovery and the
  linear-axis blank. `selftest.test_measure_stage_position` covers the flip both ways.
  `selftest.test_pooled_pixel_spread` covers:
  - the bridge's `std` against numpy;
  - `pooled_stats` against literally concatenated values;
  - Plot XY `pooled_sd` through the Engine.
- **In the app:** open `graphs/IL8_standard_curves.nd2graph.json` and pull a page's
  **Standard curve** node. You should see five Langmuir curves (imaging 0–4), a dashed blank
  line, and a y axis of 0–4095. The **One bead across positions** node's `beads` table
  carries `object_uid` / `sighting` / `sightings` and the centroid columns.
- **Independent check:** an independent NumPy/SciPy re-implementation (scratch, not
  committed) agreed with the graph on all 15,095 detections: centroids, R-PE, background,
  bead partition and sighting numbers, with 0 difference. Every plotted point, bar and
  pixel count equals the mean, n − 1 SD and size of the prototype's concatenated pixels.

## Files

- `nodegraph/catalog/table/unique_objects.py` (new): `table.unique_objects`
- `nodegraph/catalog/__init__.py` (hotspot): module registered after `table.aggregate`
- `nodegraph/catalog/analysis/measure.py`: `flip_x` / `flip_y`, flip-aware stage columns;
  `std` stat
- `nodegraph/bridges.py`: `std` group reducer
- `nodegraph/catalog/analysis/label.py`: `regions` mode
- `nodegraph/structure.py`: `label_components(..., whole=)`
- `nodegraph/catalog/_shared/figure.py`: `xy_langmuir`, `XY_FITS += langmuir`,
  `decorate_xy` on linear x, Langmuir drawing, reference-line legend text, `pooled_stats`,
  `xy_series` / `xy_zero_refs` / `frame_spec` pooled rows
- `nodegraph/catalog/plot/xy.py`: `fit` / `zero_x` docs and choices; `error = pooled_sd`
  with `pool_n` / `pool_sd`
- `nodelab_v2/demo_recipes.py`: `extra_layers` names count as written
- `nodegraph/selftest.py` (hotspot): `test_unique_objects_langmuir`,
  `test_pooled_pixel_spread`; `test_measure_stage_position` updated for the flip
- `codemap/node_roles.json`, `codemap/node_demos.json`: the new node's role and demo
- `codemap/node_synopsis.json`, `codemap/gen/*`, `codemap/STATE.md` (hotspot): regenerated
- `scripts/catalog_baseline.json` (hotspot): re-saved for the new op, modes and sockets
- `MANUAL.md`:
  - §15 rows for Unique Objects, Connected Components, Measure and Plot XY;
  - §18 Table Join row: the split-background case
- `graphs/IL8_standard_curves.nd2graph.json` (data, untracked): rebuilt with the Cy5 /
  unique-bead / Langmuir workflow

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 0 on this run; 139 after the pass on earlier runs, as on the untouched base 889d9d4)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-saved: the new op, `analysis.label` regions, Measure flip sockets and `std` stat, Plot XY modes and pool sockets)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
