# Overlay, Merge and Crop To place by the camera's handedness on every primary

- **Date:** 2026-10-16
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**The placement compositor walks a non-canvas primary the camera's way.** Every placement entry now carries `frame_flip`, the primary's handedness, which is the node's Flip X/Y (both files came off one scope). The new `placement.frame_mapping(pri_md, entry)` decides how the output is walked:
- an Experiment Canvas keeps its own orientation (`canvas_mapping`, unchanged);
- any other primary is walked in `frame_flip`;
- an entry stamped before this change still gets the old ascending walk.

`compose_secondary_plane`, `field_rect01` and `nudge_delta_um` all read the walk from it. So the Overlay display, the `resample` bake, Merge (C), the Viewer's field outline and the two-click Nudge agree.

**A stage-Stitched input takes Flip X/Y like a raw field.** The 2026-08-05 rule that forced the flips off for a stitched input is removed, along with `placement_entry.already_stage_placed` / `handedness_for`. This applies to Overlay, Crop To, Merge and the runner's display re-plan. The "Flip inert (stitched secondary)" note is gone.

**What moves:**
- Overlays onto a raw field or a stitched mosaic with Flip X on (the default) now land where the primary shows the same feature. Before, they were mirrored about the primary's centre.
- Stitched secondaries on an Experiment Canvas are no longer mirrored about the mosaic's centre.
- Crop To of a stitched mosaic takes the requested window, not its mirror.
- A hand-set `flip_x: False` behaves exactly as before.

## Why

**The reported bug.** Overlay mirrored a stitched secondary baked onto an Experiment Canvas (IL-8 spheroid: IoU 0.73 against its own raw tiles; see `2026-10-16_il8-session3-landmark-registration.md`). The cause was `handedness_for`, which forced the flip off for a stitched input on the theory that a mosaic is "in stage coordinates". It is not. Stitch keeps every tile's pixels as taken, so a mosaic runs the camera's way (proved the same day on the whole-well bead mosaic, 20/20 beads within 5 µm only when flipped).

**The deeper fault the rule had been patching.** The compositor walked every non-canvas primary toward stage +x. On this scope that is the mirror of the primary's own pixels. With two stitched files of nearly one centre (the 2026-08-05 WellA3 pair, 784 µm off), forcing the secondary's flip off happened to cancel the mirror, which is why the rule looked right.

**Ground truth.** A synthetic stage image was cut into camera-handed tiles and placed through the real nodes. The table gives the correlation with the truth in the primary's own frame:

| Case (Flip X on) | Before | After |
|---|---|---|
| raw field ← raw | 0.36 | 1.000 |
| stitched ← stitched | 0.05 | 0.998 |
| stitched ← raw | −0.17 | 0.998 |
| canvas ← stitched | −0.23 | 0.998 |
| canvas ← raw | 0.998 | 0.998 |
| Crop To (stitched) | — | 0.991 |
| Merge, stitched + raw | — | 0.998 |
| Merge, raw + stitched | — | 1.000 |

**Real data, IL-8 spheroid on the T4 canvas.** The stitched bake now matches the raw-tile bake: IoU 0.972 (was 0.733) and centroid x −10845.1 vs −10844.1 µm (the stitched bake was at −10869.2).

**Why the frame is taken from the node's flips** rather than stamped on each Dataset: no file records handedness, Stitch keeps the camera's, and union mode already assumed the primary shares the node's handedness. A canvas, which does state its orientation, keeps it.

**Saved graphs this changes.** `graphs/TFM_10_12.nd2graph.json` (TFM_1…TFM_5) is the only one.
- Its stage placements, `cells_place_T1_*` / `cells_place_T2_*`, are titled "(drawn mirrored in x)", which independently confirms the old fault on real data. They now place correctly.
- Its same-grid `cells_overlay_*` ("Flip X un-mirrors them") no longer mirrors either, so the final cells-under-strain picture is still right side up.
- **However**, `cells_shift_*` applies the gel drift between the two. In the old chain that shift was applied to mirrored cells, so by my reading its x sense came out reversed; it is now applied in the camera frame.
- Both titles are now stale. I did not edit that graph; whoever owns the TFM analysis should re-pull it and retitle.

The IL-8 graph keeps its raw-tile spheroid bake, which was correct before and is still correct.

## Files

- `nodegraph/placement.py`: `frame_mapping`; `compose_secondary_plane`, `field_rect01` and `nudge_delta_um` (new `frame_flip=` keyword) walk by it; docstrings.
- `nodegraph/catalog/_shared/placement_entry.py`: `frame_flip` in every entry; `already_stage_placed` and `handedness_for` removed.
- `nodegraph/catalog/view/overlay.py`, `nodegraph/catalog/util/merge.py`, `nodegraph/catalog/util/crop_to.py`: flips applied directly. The Flip X/Y descriptions say they apply to both files, stitched ones included.
- `nodegraph/catalog/analysis/measure.py`: one docstring line.
- `nodelab_v2/runner.py`: `_replan` uses the node's flips.
- `nodelab_v2/picker.py`, `nodelab_v2/window.py`: `PickRequest.frame`, so a two-click Nudge off a canvas is signed by the primary's frame.
- `nodegraph/selftest.py` **(hotspot)**:
  - new `test_placement_handedness_ground_truth`: Overlay (raw, stitched and canvas primaries), Merge, Crop To, `frame_mapping`, `field_rect01` against the painted rect, and the Nudge sign;
  - `test_util_merge` loses the stitched-inert assertions.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` **(hotspot, regenerated)**; `scripts/catalog_baseline.json` **(hotspot, re-saved)**: the six Flip descriptions.
- `MANUAL.md` §15: the Overlay, Crop To and Merge rows.
- `CodeLog/Architecture/ENGINEERING_NOTES.md`: the 2026-08-05 entry marked superseded, with the reason.
- `CodeLog/Updates/worklog/2026-10-16_il8-session3-landmark-registration.md`: its "finding not fixed here" now points to this entry.

Not part of this change: the earlier uncommitted work (14–16 Oct entries) and other sessions' files, which the worklog tool pre-listed.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_placement_handedness_ground_truth()"`
- Overlay a raw high-mag field onto a raw low-mag field of the same well, with Flip X on. A feature now lands on itself rather than at its mirror across the primary's centre.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (249 ok, exit 0)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 0; one earlier run passed then segfaulted on exit, as seen once before this session; the rerun exited 0)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (141 ops) after `save` of the six Flip descriptions
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed; committing is left to the owner.
