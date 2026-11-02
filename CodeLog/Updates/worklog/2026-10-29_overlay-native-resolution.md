# Overlay native resolution

- **Date:** 2026-10-29
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

Overlay's `resample` bake now draws the secondary at its own resolution wherever the primary is finer than it. `view.overlay`'s `_bake_resample` reads only the window of each secondary tile that the field covers (the `want` the compositor already computed, snapped out to whole pixels by the new `placement.pixel_window`), not the whole tile. `placement.compose_secondary_plane` also cuts a whole-tile read down to that window before its minification test, so every caller samples the same way: the Viewer, the lazy overlay provider and the bake, whether the reader returns a whole tile or a window. Area-averaging still happens, but only where the output really is coarser than the source pixels it draws. `test_overlay_resample` checks a 128 µm, 2 µm/px tile behind an 8 µm, 1 µm/px field. All 16 window pixels must be drawn, each as a 2 × 2 block. Only that window may be read. A whole-tile reader and a windowed reader must give identical planes.

## Why

The user asked for Overlay to draw at full resolution. The TFM_10_20 and 10_28 strain-over-cells movies drew the CRC nuclei (0.86 µm/px) at about 1.7 µm/px. Each CRC tile (~1760 µm) sits behind a 294 µm TFM field. The compositor compared the whole tile's pixel count with the number of output rows that land in it, judged that minification, and block-averaged the whole tile 2× before magnifying the small window. The window is what was drawn, so the window's pixel count is the right comparison. Cutting to the window fixes the sampling for every reader without a second code path. Reading only the window in the bake also stops the bake reading roughly 30× the pixels it draws per tile, frame and plane. That was a large part of why each cells movie took hours. The memo keys each op on its transitive import closure (`hotreload`), so old half-resolution bakes are not served after this change.

## Files

- `nodegraph/placement.py`: `pixel_window`; the whole-tile cut in `compose_secondary_plane`
- `nodegraph/catalog/view/overlay.py`: windowed `read_tile` in `_bake_resample`
- `nodegraph/selftest.py`: native-resolution case in `test_overlay_resample` (hotspot file)
- `codemap/gen/*`, `codemap/STATE.md`: regenerated (new symbol `pixel_window`; op fingerprints)

The working tree also holds the uncommitted IL-8 Run 4 change (`_shared/point_source.py`, its `test_point_source_fit` hunk, worklog `2026-10-19_il8-run4-two-configs.md`). It is a separate change.

## How to verify

`PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_overlay_resample()"`. In the GUI, put an Overlay (Output = resample) of a CRC tile file on a TFM frame and zoom in. The nuclei show their 0.86 µm pixels as 3 × 3 blocks of the 0.287 µm primary grid, not 6 × 6.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (run with
      `NODEGRAPH_MIN_FREE_RAM=1G` while a DVC batch held the RAM). Another session was editing the
      IL-8 files at the same time, so this run vouches for the overlay change, not for theirs.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
      (second run; the first failed a wall-clock assertion, "running 1:05" read as 1:13, because the
      machine was loaded. Unrelated to this change.)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py write` -> regenerated; diff read (only `pixel_window` added, op `fp`s moved, touched modules)
- [ ] `python scripts/_sync_check.py` -> IN SYNC (not pushed: the commit waits for the user)
