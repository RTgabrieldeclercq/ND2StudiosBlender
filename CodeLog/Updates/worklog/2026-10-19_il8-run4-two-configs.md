# IL-8 Run 4: two wells, two imaging configurations

- **Date:** 2026-10-19
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Fit fix (`_shared/point_source.py`).** Under `sqrt` weighting, `_solve` now sets every coefficient whose lower bound `least_squares` reports active (`active_mask == -1`) to exactly 0. Before, a rate pinned at its bound came back as ~1e-20 ('trf' keeps its iterates strictly inside the bounds), so Plot XY's legend printed "q = 3.54e-20 fg/s" at C6's flat late passes. On a noiseless flat profile the old code returned 5.6e-10, and on C6's last pass 2.3e-17, both with the bound flagged active. `test_point_source_fit` now checks that a flat profile returns q == 0 and its level.

**New graph.** `graphs/IL8_run4_two_configs.nd2graph.json` has one page per well of the Run 4 plate (`Downloads\Run 4`, 16–19 Oct):

- **C5 · Config 1** (gain 35, 561 nm 40 %, pinhole 18.3 µm): `ChannelR-PE,Cy5_Seq0005.nd2`, 150 fields × 7 passes about 8.5 h apart. 145 nodes.
- **C6 · Config 2** (gain 1, 561 nm 2.5 %, pinhole 99 µm): `ChannelR-PE,Cy5_Seq0006.nd2`, 74 fields × 52 hourly passes. 479 nodes.

Each page runs the WellB4 workflow (`IL8_bead_spheroid_distance`):

- **Spheroid group.** The centroid comes from `ChannelGFP_Seq0000.nd2` (16 Oct 21:34). Each well uses its own 2 × 2 tiles: C5 is `m4-7`, C6 is `m0-3`.
- **Boundary.** The group's footprint, ≥ 5000 µm² so single cells drop out, taken as an equivalent circle.
- **Beads.** The standard-curve bead chain, one row per bead per pass from its first field.
- **pg/mL.** The configuration's own standard curve (`IL8_standard_curves` page 1 or 2), first imaging.
- **Fit.** Crank point source + uniform well background on the beads ≤ 400 µm above the floor.
- **Figures.** Every figure type of the earlier run, plus a pg/mL twin of the XY-distance-vs-height panels.

Figures are in `Downloads\IL8_Run4_figures\C5_config1` (48) and `\C6_config2` (318).

## Why

The user asked for the earlier plots, plus their pg/mL equivalents, on this new dataset. The new data forced these choices:

- **Spheroid Z.** The GFP file's XY points log z = 0. Each frame's own stage z (5016.84) is the stack's bottom plane: in WellB4's GFP file the points log 5126.3 and the frames 5066.3, which is 5126.3 − 4 × 15. A labelled `math.values` adds the home focus, 5016.84 + 60 µm. The floor reference is the stack's bottom, Z = 5016.84.
- **Background = frame median, not mean.** 32 of the 150 C5 fields carry cell signal in R-PE. Those fields are 1.2–5 mm from the group, so cells sit all over the well. That signal raises the frame mean by up to 731 counts. In clean fields, mean and median agree within 1–3 counts.
- **Config 2's combined plots.** 52 series can't be told apart, so these plots show every 8th pass plus the last, through a parallel concat → curve → filter → fit. With D fixed and one rate per pass, each pass's fit is independent, so these curves are the same as in the 52-pass fit.
- **Time origin (an assumption).** C5's first pass is taken as seeding + 15 h, as in WellB4. The Day 3 overview's name and clock (19 Oct 07:40) agree. C6's first pass is then at +15.83 h.

Results:

- **C5.** q is 0.02 → 0.65 → 0.26 fg/s. The well background climbs from 88 to 1240 pg/mL, and R² stays ≤ 0.26.
- **C6.** The gradient is clear early (R² 0.80 at the first pass). It flattens by about 40 h, and the background plateaus at about 1,100–1,200 pg/mL from about 55 h.

## Files

- `nodegraph/catalog/_shared/point_source.py`: `_solve`'s sqrt branch snaps an active lower bound to 0.
- `nodegraph/selftest.py` **(hotspot)**: the flat-profile case in `test_point_source_fit`.
- `codemap/gen/*`, `codemap/STATE.md` **(regenerated)**: line shifts, plus new fingerprints on the 9 nodes that import `point_source` (through `figure.py`). Nothing else.
- `graphs/IL8_run4_two_configs.nd2graph.json` (new). Its builder and the figure scripts (one bead lighting up, the two-well time course, the bleaching check) are in the session scratchpad. The figure scripts are copied beside the figures.
- The other paths the worklog tool pre-listed (`CodeLog/live/crc-leading-edge/**`) are another session's untracked work and not part of this change.

## How to verify

Open the graph, then pull **Crank point source + well background per pass** on each page:

- **C5** reads q 0.022 … 0.26 fg/s, with background 88 … 1240 pg/mL.
- **C6** reads 0.076 … 0 fg/s, with background 2.7 … 1168 pg/mL.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (252 ok, exit 0)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 0)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 143 ops
- [x] `python scripts/_codemap.py` -> written, diff read; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] The graph loads and every figure node pulls headlessly: 48/48 (C5) and 318/318 (C6) `io.write_figure` nodes ran without error.
- [x] Offscreen GUI load and render of both pages succeeded. The page 0 process then hit the known exit-139 at `os._exit`, after every image was written.
- [ ] `python scripts/_sync_check.py`: not pushed. Committing is left to the owner.
