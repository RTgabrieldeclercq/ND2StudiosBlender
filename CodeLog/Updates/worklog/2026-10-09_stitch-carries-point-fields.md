# Stitch carries Point fields; Plot Field draws the whole mosaic

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

**Stitch (M→1) accepts Point tables and moves them onto the mosaic.**
- **Moving.** Each row's `m` becomes 0. Its position columns shift by its tile's canvas
  offset: `y`/`x`, the grid-cell box (`y_lo`/`y_hi`/`x_lo`/`x_hi`), `centroid_y`/`x` and the
  µm positions `y_um`/`x_um` (by the offset × pixel size). Every other column is a property
  of the row and is kept as it is (displacement, velocity, strain, stress, ids).
- **Overlaps.** Where tiles overlap, a row is kept only from the tile whose centre is
  nearest, among the tiles covering its canvas point. Nothing is counted twice.
- **Refusals.**
  - A Point table with no `m`/`y`/`x` is refused by name.
  - Label, Track and Mesh tables are still refused: a label raster, a track crossing a seam
    and a mesh's links cannot be moved row by row.
- **Provenance.** The sampling stamp records each table's row count and how many overlap
  rows were dropped.

**Plot Field reads a mosaic.**
- **Arrows** are thinned on a spatial grid: `arrow_every` N means one per N × N block of
  nodes. At most 2500 are drawn per view, each as long as the gap between drawn arrows.
  Before, it took every Nth row and drew arrows one node spacing long. That left stripes on
  a lattice, and on a 36-tile mosaic about 70,000 arrows of 2 px each.
- **A scattered field** is resampled to `side / 2` cells (256 on screen, 512 for export)
  instead of the Viewer's 128. A merged mosaic is rarely one lattice, so it is drawn this
  way.

**Playground.** The chip page now has `Stitch (M→1)` (stage layout, Flip X on, feather)
between Stress and the four figures. Each figure shows the whole 6 × 6 chip, with a
500 µm scale bar. The page's frame label and the script's docstring say so.

**Docs:**
- MANUAL: the Stitch row and the Plot Field row;
- concept card CON-28: a note on Stitch;
- the descriptions of `util.stitch`, `arrows` and `arrow_every`.

## Why

The user: "right before the plot field, i want to be able to use the stitch M->1 node to
make the full file".

**What was in the way.** Stitch refused any Dataset carrying a structure table, and the
field Plot Field draws is a Point table. The refusal's reasons:
- rows address the per-tile grid;
- detections in overlaps would be duplicated.

Both have answers for a Point table: move the address with the tile, and keep one row per
overlap. Neither has an answer for labels, tracks or meshes, which stay refused.

**Why move the field, not stitch first and run PIV on the mosaic.** The second would re-run
PIV on a 6144 × 6144 × 11-plane image. It would also correlate across tile seams that are
not one continuous acquisition. The field is already measured per tile; drawing it on the
mosaic only needs its coordinates moved.

**Why the arrow change.** Without it the whole-chip figure is a black texture: about 70,000
arrows, each 2 figure px long.

## Files

- `nodegraph/catalog/util/stitch.py`: `_STITCH_PX_COLUMNS`, `_STITCH_UM_COLUMNS`,
  `_stitch_owner`, `_stitch_points`; the refusal narrowed to non-Point structure; docstring
  and description.
- `nodegraph/catalog/plot/field.py`: `_MAX_ARROWS`, `_thin_arrows`; scattered raster at
  `side // 2`; the arrow scale follows the drawn spacing; `arrows` / `arrow_every`
  descriptions.
- `nodegraph/selftest.py`:
  - `test_stitch`: the refusal cases are now a Track table and an address-less Point table,
    plus a new §5b Point-field case;
  - `test_plot_field`: arrow thinning and the cap.
- `scripts/make_field_playground.py`, `graphs/field_mechanics_playground.nd2graph.json`.
- `MANUAL.md`, `codemap/concepts.md` (CON-28).
- Regenerated: `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`,
  `scripts/catalog_baseline.json` (three descriptions).

Shared files both developers edit: `nodegraph/selftest.py`, `codemap/gen/*`,
`scripts/catalog_baseline.json`. (The two `graphs/*alia*` files the scaffold listed are the
user's own untracked graphs and are not part of this change.)

## How to verify

1. **`test_stitch` §5b** builds a 2 × 3 mosaic of 96 px tiles with 32 px overlap. Each
   tile has a 16 px lattice, so the overlaps hold exact duplicates. After the stitch:
   - one row per canvas node (280 of 432);
   - `m` = 0;
   - each row's `y`/`x` = its tile coordinates + its tile's offset, and the box and µm
     columns agree;
   - the four-tile corner (72, 72) comes from tile 0, (88, 88) from tile 4, and a
     single-cover node from its tile;
   - the stamp counts the rows.

   A Track table and an address-less Point table are refused.
2. **`test_plot_field`:**
   - a 40 × 40 lattice keeps every arrow at N = 1 and one in four at N = 2, evenly
     (20 rows × 20 columns);
   - a 90,000-node field is capped at 2500.
3. **Real data** (scratchpad `chip_fig.py`, through `EngineRunner`):
   - `fig_speed` of the chip page at plane 3 loaded the saved PIV trial, ran the chain, the
     stitch and the view in 34 s;
   - the picture has m 1, t 10, z 11, with a 256² scattered field, 2490 arrows and a
     500 µm bar;
   - the speed map is continuous across the tile seams.
4. **Handedness** (scratchpad `chip_mosaic.py`): the stitched chip image with Flip X on is
   seamless; with it off, vertical seams show. On is the default.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: 192 passing, then the long-standing
  `test_write_movie` failure (same assertion and numbers); all 38 tests after it pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
  PROBES PASSED, exit 0. One earlier run, made while the full selftest ran alongside,
  printed the same PASSED and then crashed on exit (segfault), the teardown race the probe's
  own comments describe. Re-run alone, it exited 0.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (baseline re-saved: the
  three descriptions).
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT.
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed.
