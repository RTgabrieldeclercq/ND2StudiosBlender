# Overlay union axes

- **Date:** 2026-10-14
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Per-source axes on a union canvas.** On an Overlay chain whose every Overlay is `canvas = union`, the Viewer hides the canvas's own M/T/Z strips. Each overlaid file instead gets its own M, T and Z strips, each with its own ▶ and fps. Moving one strip moves only that source.

- **Mechanism.** A display-only runner selection (`EngineRunner.set_source_select`), keyed by `_ovr_gen` exactly like the ◀▶ steppers.
- **Defaults.** A source starts on field 0, frame 0, at its stack's home plane.
- **The M strip.** It picks the field the source's LUT histogram and Auto read. That field is outlined on the canvas in the source's colour (`beads · M 15`), never smaller than 10 px. The outline geometry is `placement.field_rect01`, the compositor's own.

**Playback.**

- Every ▶ plays only its own axis. T's ▶ no longer doubles as Play all.
- A separate **▶ Play all** button shows whenever sources are overlaid, on the Overlay or downstream of it.
  - On a primary it is the old sub-ticked T playback.
  - On a union canvas it steps every source's T on one counter; a source with fewer frames holds its last.

**Overlay LUTs read the source, not the composite.**

- **Histogram.** Comes from the source's own plane at the field and frame it shows. The worker reads it beside each composite (`_stash_lut_refs`, ≤ 256 px, LRU), and the window passes it to the Viewer before each delivery.
- **Slider extent.** The source's bit depth.
- **Auto.** Now re-scales every overlay channel too, each from its own source plane.
- **Default window.** Pooled over up to 16 of the source's fields, not field 0 alone.

**Bead fields no longer vanish (engine).** `placement.tiles_covering` judges a "sliver" against the smaller of the field and the tile. A small tile wholly inside a large field is kept; a large tile grazing a field corner is still dropped.

## Why

User report on a spheroid (GFP, 2×2 tiles × 9 z) + ELISA bead (R-PE + Cy5, 24 fields × 2 t) union Overlay:

- **No per-source axes.** The union method should show as many axes as there are overlays, with the spheroid's and the beads' T/M/Z side by side.
- **LUTs.** They should follow each source's selected M, and Auto should scale all of them; the bead overlay "does not process the LUTs correctly".
- **Playback.** Playing an axis should play only that axis, and there should be a Play all button.

Reproducing it in the real window found three causes:

1. **The beads were never drawn.** All 24 bead fields were dropped as slivers: each 64 µm field is 0.0055 % of the 8.8 mm canvas, under the 0.01 % cut, which was measured against the field only. Channels 2/3 got no plane at all.
2. **The canvas carried only its first file's axes.** Canvas T = 1 (the spheroid's), so the beads' T1 was reachable only through ◀▶. The spheroid's plane followed a 53-plane canvas Z.
3. **The LUTs were built from the wrong pixels.** The overlay LUT seed came from field 0 (the dimmest bead: R-PE 43–77, Cy5 43–237, while real beads reach 2000+/4095). The histogram was taken from the composite (a speck on blank). Auto skipped overlay channels entirely.

**Why scope independence to all-union chains.** A union canvas has no primary to pair against. An ordinary Overlay onto a `view.canvas` keeps the paired model: input 0's clock, pins and rates are what those graphs rely on.

**Why reuse the stepper keying (`_ovr_gen`).** It is already proven display-only and stale-safe through the prefetch and preload, so it was reused rather than adding a second override path.

## Files

- `nodegraph/placement.py`: `tiles_covering` sliver rule; new `field_rect01`.
- `nodegraph/catalog/view/overlay.py`: the `union` option text names the Viewer behaviour. This changes `view.overlay`'s catalog entry.
- `nodelab_v2/runner.py`:
  - `_src_select`, `set_source_select`, `_independent_overlay` and `_default_source_frame`;
  - selection-aware `_source_frame`, `_compose_overlay` and `overlay_frame_readout`;
  - `_lut_ref_frame`, `_stash_lut_refs`, `overlay_lut_refs` and `overlay_field_rects`;
  - pooled `_source_lut`.
- `nodelab_v2/window.py`: `overlay_select` slot, LUT references before each delivery, field rects, Play all preloads like T.
- `nodelab_v2/viewer.py`: per-source axis rows, the Play all row, key-generic playback, LUT logic from source planes, field outlines.
- `nodegraph/selftest.py` **(hotspot)**: `test_overlay_union_axes`, appended and registered at the end of `main()`.
- `scripts/_nodelab_v2_phase5_probe.py`: O11 updated. T's ▶ steps T only and Play all owns the sub-ticks. A union block was added.
- `codemap/gen/*`, `codemap/STATE.md` **(hotspot, regenerated)**: every node's `fp` moves because `placement.py` is in the engine closure. `codemap/node_synopsis.json` was regenerated.
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: only `view.overlay`'s `canvas` option text differs.
- `MANUAL.md`: the Viewer's playback section and the Overlay how-to.

The working tree also carries earlier uncommitted work (five changes with their own entries, plus 2026-10-14_bead-spheroid-distance). Several of the files above mix hunks from them.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_overlay_union_axes()"`
- Load the spheroid GFP and bead ND2s, wire them into an Overlay (spheroid → data, beads → secondary) with Canvas = union, and view the Overlay.
- **Axes.** Playback shows "Spheroid GFP: M 0/3, Z 4/8" and "beads: M 0/23, T 0/1", with no canvas strips.
- **Fields and LUTs.** Bead M = 15 outlines that field on the canvas and moves the R-PE/Cy5 histograms. Auto sets every overlay channel from its picked field.
- **Playback.** The beads' T ▶ moves only the beads, and ▶ Play all steps every source's T.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The tree holds other uncommitted work, so committing is left to the owner.
