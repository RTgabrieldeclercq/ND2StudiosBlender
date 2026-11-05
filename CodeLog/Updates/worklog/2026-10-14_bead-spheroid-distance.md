# Bead spheroid distance

- **Date:** 2026-10-14
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Plot XY gains Colour by and Label by.** These live under Error = none only, since they need one row per point:

- **Colour by** names a numeric column. Every marker takes its value on a continuous colour map (viridis, inferno, magma, plasma, cividis, turbo, coolwarm) on a linear or log scale, with a colour bar.
- **Colour range** fixes either end. Left blank, the range spans every row, so the frames of a Per = frame figure share one scale. Values past the range take the end colour and the bar shows an arrow. Missing values are grey, and series are then told apart by marker shape.
- **Label by** prints a column beside each point: numbers rounded to Label decimals with the column's unit, text as it is. Labels are placed after layout, clear of other labels and points, with a leader line when crowded.
- **Half-fixed axis ranges now hold across frames.** In Per = frame, a range with one end fixed (`0,`) has its open end taken from all rows. Before, each frame autoscaled that end, which broke the "axes every frame shares" promise.

**Measure: stage Z after a Stitch.** `analysis.measure` now gives a stitched mosaic's rows an absolute `stage_z_um`, where it used to return NaN. Stitch drops the per-position focus log, so Z now comes from the mosaic's `origin_um` Z corner, the same transform-maintained corner `field_box` already uses for X/Y:

- slice k at z0 + k·step for bottom-to-top stacks;
- z1 − k·step for top-to-bottom stacks;
- z0 for a single plane.

Where a focus log exists it still wins.

**New graph:** `graphs/IL8_bead_spheroid_distance.nd2graph.json` (38 nodes). It plots IL-8 capture-bead MFI against XY distance from a GFP spheroid's centroid and height above Z = 5022 µm, one figure per timepoint, each bead labelled with its true 3D distance.

## Why

The request was this figure for `ChannelR-PE,Cy5_Seq0004.nd2` (beads) around the spheroid in `WellB4_ChannelGFP_Seq0001.nd2`: x = XY distance from the point source, y = Z above 5022 µm, points coloured by MFI on a heat scale, the true distance as text beside each bead, and separate plots for the two timepoints.

Plot XY could colour only by series and print nothing per point. A dedicated plot node was the obvious alternative, but it would have duplicated Plot XY's whole style, frame and export machinery.

The spheroid is a 2×2 tile Z-stack, so its centroid has to be measured on the stitched mosaic. There every row's `stage_z_um` was NaN, even though `origin_um` still held the stack's Z. That is a real gap for any stitched volume, not just this one.

Distances use eleven Math Values nodes rather than a new "distance to point" node. The arithmetic stays visible and editable on the canvas, and no new op was needed.

Three graph choices were forced by this data. All are recorded in the node titles and the report:

1. **Cy5 threshold 150, not Config 1's 500.** At 500, 6 of 24 positions went undetected, including the bead nearest the spheroid (R-PE MFI 46). Cy5 background never exceeds 75 across all 48 frames.
2. **Field B4_0020 is cropped out.** It is the same bead as B4_0021 (stage XY within 0.5 µm), imaged 17.5 µm out of focus.
3. **Unique Objects keeps the FIRST imaging of a doubly imaged bead.** The same-focus re-image B4_0013 → B4_0014 lost 33 % of its R-PE (142 → 96), so the second imaging is bleached.

## Files

- `nodegraph/catalog/plot/xy.py`: the five sockets and two modes, half-open per-frame ranges.
- `nodegraph/catalog/_shared/figure.py`: `color_limits`, `point_text`, colour and text through `xy_series` / `decorate_xy` / `frame_spec`, the colour bar, `_place_point_labels`.
- `nodegraph/catalog/analysis/measure.py`: `_stage_position_columns` origin_um Z fallback.
- `nodegraph/selftest.py` **(hotspot)**: `test_bead_distance_plot`, appended and registered at the end of `main()`.
- `codemap/gen/*`, `codemap/STATE.md` **(hotspot, regenerated)**; `codemap/node_synopsis.json` (regenerated).
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: only `plot.xy`'s sockets and modes differ.
- `MANUAL.md` §15: the Plot XY and Measure rows.
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the graph.

The working tree also carries five earlier uncommitted changes (nd2-acquisition-clock, node-queue-popup-fix, standard-curve-trendline, unique-beads-langmuir, overlay-pixel-centre), each with its own worklog entry. `figure.py`, `xy.py`, `measure.py`, `selftest.py` and the generated files mix their hunks with this one.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_bead_distance_plot()"`
- Open `graphs/IL8_bead_spheroid_distance.nd2graph.json` and pull **Export timepoint 0** and **Export timepoint 1**. They write `Downloads\IL8_bead_spheroid_figures\bead_mfi_vs_distance_t0.png` and `_t1.png`.
- **Point source Z** should show `spheroid_z_um` = 5118.5 µm, not NaN. X/Y should read −10832.5 / −10135.6 µm, which matches a numpy centroid of the same mask to 0.1 µm.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The tree holds other uncommitted work, so committing is left to the owner.
