# IL-8 gradient maps and the standard curve, in-graph

- **Date:** 2026-10-15
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**New node: Gradient Map** (`analysis.gradient_map`). It turns a value measured at scattered objects into a map over an image.

- **Inputs:** `data` is the image drawn on: one field and one plane with a stage placement, such as an Experiment Canvas with a baked background. `points` carries the objects. `source` carries the source's table.
- **Methods:**
  - `trend_idw` (default): the exponential A·e^(−d/λ), the same fit Plot XY draws, times the inverse-distance-weighted log residuals. The map is exact at an object in the plane, follows the decay's shape between objects, and relaxes to the trend past `reach`.
  - `trend`: the exponential alone.
  - `idw`: plain IDW of ln value, with no source.
- **Distances:** with a Z column, every distance is 3-D, measured to the plane `height` above the source.
- **Frames:** `per_by` gives one frame per value. T becomes the count (unknown at edit time), the clock is dropped and `frame_names` carries the values.
- **Outputs:**
  - a lattice Point table (`gradient`) that Plot Field and the Viewer's Field tab colour;
  - `gradient_objects`, the objects placed on this image, with each frame's fit.

**New node: Standard Curve** (`table.standard_curve`). It turns a signal column into a concentration, adding `<name>` and `<name>_flag` in place.

- **Curve = fit:** reads a standards table filtered by `std_where` (`sighting=0`). Each concentration's point is the mean weighted by `std_weight`, the pixel count. Langmuir is fitted through those points (Plot XY's own fit); linear is fitted through them and the blank. The blank gives the LOD as mean + 3 SD.
- **Curve = typed:** takes the parameters as typed.
- **Flags:** in range, below or above range, below LOD, above Bmax, no signal.

**Plot Field** gains four things:
- `marks`, `mark_label` and `mark_decimals`: rings with labels placed clear of each other;
- `color_scale` linear / log;
- `color_label`;
- an extend arrow when the table passes the colour range.

It also gains the `inferno` colormap, and frame titles take `frame_names`.

**Plot XY:** `fit_weight` relative / absolute for the exponential trendline. `absolute` fits in data units, for linear y axes.

**Measure:** ~~Flip X/Y are ignored on a stage-Stitched mosaic~~ — **reverted 2026-10-16, this change was wrong** (see `2026-10-16_il8-session3-landmark-registration.md`). A stitched mosaic keeps the camera's handedness, so Measure's flip applies to it; Measure's behaviour is back to what it was before this entry.

**The IL-8 graph** (`graphs/IL8_bead_spheroid_distance.nd2graph.json`, 148 nodes) adds:
- the MFI vs 3-D distance plot on linear y, plus the same plot in pg/mL;
- the Config 1 page of `graphs/IL8_standard_curves` copied in, from raw ND2s, with its figure exported;
- Standard Curve: Langmuir on the first imagings, pixel-weighted;
- an Experiment Canvas (10 µm) spanning the four bead files and the spheroid, with the spheroid's max projection baked onto it by Overlay `resample`;
- Gradient Map + Plot Field for MFI and for pg/mL, five timepoints each;
- an Overlay (display) putting the Seq0004 ELISA fields over the map in the Viewer.

## Why

The request:
- plot the decay as an exponential without a log y axis;
- overlay the stitched spheroid with the ELISA field and a bead field spanning it;
- interpolate between beads along the exponential decay, with the source at the spheroid's centre;
- convert to IL-8 pg/mL with the Config 1 standard curve from `analysis_2026-10-13`;
- do all of it in ND2Studios.

No node made a field from scattered objects round a source, and none converted a signal through a standard curve, hence the two nodes.

**Standard curve from raw pixels.** The curve is fitted in-graph from the standard-curve graph's own Config 1 page, not typed from the report. The result matches the report: Bmax 5,186.9 (report 5,187), Kd 5,158.1 pg/mL (5,158), R² 0.9953, blank 5.78. So the conversion is reproducible and follows the standards.

**Why the interpolation is 3-D.** The first map interpolated residuals in XY only. At T1, B4_0012 (MFI 47, 665 µm above the spheroid's plane) then punched a dark hole beside the spheroid. 3-D distances weigh it as far.

**The Measure change (reverted 2026-10-16).** The compositor reported that it would not flip a stitched mosaic, and Measure was changed to match, moving the spheroid centroid 46 µm in x. The "independent" check — the centroid of the canvas-baked footprint — was not independent: Overlay's bake makes the same no-flip assumption. On 2026-10-16 a whole-well bead mosaic placed the 20 beads of the T1 bead run (median 5 µm) only WITH the flip, and Stitch's code puts the highest-x tile on the left. So the change was reverted, and the numbers this entry quoted after it (λ T1 873 … T5 891 µm, distances moved ±46 µm) were computed with the wrong source; the pre-change numbers stand.

**Plot XY fit weighting.** On a linear axis the ln-y fit sits under the brightest beads. The absolute weighting is offered, but the graph keeps relative: λ is stable, about 0.87–1.10 mm, whereas the absolute fit swings between about 0.5 and 2 mm with a few bright beads.

**Why the canvas.** The Experiment Canvas + Overlay `resample` bake is the engine's existing way to put two files on one grid. Plot Field already draws a field over an image, so the figure needed marks and a log scale rather than a new plot node.

## Files

- `nodegraph/catalog/analysis/gradient_map.py` **(new)**.
- `nodegraph/catalog/table/standard_curve.py` **(new)**.
- `nodegraph/catalog/__init__.py` **(hotspot)**: both appended after `table.register_positions`.
- `nodegraph/catalog/_shared/figure.py`:
  - `xy_exponential(weight=)`, `XY_EXP_WEIGHTS`, and `decorate_xy` / `frame_spec` carrying `fit_weight`;
  - `frame_clock` honours `frame_names`;
  - `_draw_field` gains a log scale, marks, labels and an extend arrow.
- `nodegraph/catalog/plot/field.py`: marks, `color_scale`, `color_label`, `inferno`.
- `nodegraph/catalog/plot/xy.py`: the `fit_weight` mode.
- `nodegraph/catalog/analysis/measure.py`: `_flip(ctx, ds)` via `handedness_for` — reverted 2026-10-16.
- `nodegraph/selftest.py` **(hotspot)**: `test_gradient_map_standard_curve`, appended and registered at the end of `main()`.
- `codemap/node_roles.json`: `analysis.gradient_map` under measurement_and_statistics, `table.standard_curve` under table_synthesis.
- `codemap/node_demos.json`: demos for both.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` **(hotspot, regenerated)**:
  - the two ops, Plot Field / Plot XY sockets and modes, and Measure's flip_x text;
  - `fp` only on the other plot nodes and `io.write_figure`, which share `figure.py`.
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: the two ops, plot.field, plot.xy and measure's flip_x description.
- `MANUAL.md` §15:
  - new Gradient Map and Standard Curve rows;
  - Plot Field (marks, log, label), Plot XY (weighting) and Measure (stitched mosaic).
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the 148-node graph.

Not part of this change: the working tree also holds earlier uncommitted work. That is the 2026-10-15 bead-timepoints entry (Register Positions, the exponential trendline) and another session's Properties-panel files (`nodelab_v2/inspector.py`, `nodelab_v2/window.py`, `scripts/_nodelab_v2_phase5_probe.py`, three `graphs/10.6_*`). The worklog tool pre-listed some of them; this change does not touch them.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_gradient_map_standard_curve()"`
- Open the graph:
  - pull **Std curve: Langmuir** (STD): its rail reads Bmax 5187, Kd 5158, R² 0.995.
  - pull **Export map T1** … **T5**: they write `Downloads\IL8_bead_spheroid_figures\bead_mfi_map_T*.png` and `bead_il8_pgml_map_T*.png`.
  - **Export MFI vs distance** and **Export pg/mL vs distance** write the linear-axis decay plots.
- View **OVF** to see the Seq0004 ELISA fields over the MFI map.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (248 ok). The first run failed `test_column_catalog_complete`: Standards X defaulted to `condition_value`, which no producer declares statically. It now defaults to blank, which means that column.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED, exit 0. One earlier run, chained straight after the selftest, printed the same line and then exited 139 at teardown; a rerun alone did not reproduce it.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL after `save` (141 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The tree holds other uncommitted work, so committing is left to the owner.
