# Staged cards and SerialTrack on the unit runner: process lanes, saved trials, resume

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** 3af51b2

## What changed

1. **Every frame-reading staged card runs unit by unit**, across the governor's process lanes,
   saved beside the source and resumed like the single nodes:
   - **DIC:** `dic.initial_guess`, `dic.local_match`, `dic.global_step`, with (m, z) units.
   - **PIV:** `piv.first_pass`, `piv.deform_pass`, with (m, z) units.
   - **DVC:** `dvc.initial_guess`, `dvc.local_match`, `dvc.global_step`, with (m,) units.

   Each card's per-line loop moved, unchanged, into a module-level unit worker (`_dic_seed_unit`,
   `_dic_local_unit`, `_dic_global_unit`, `_piv_first_unit`, `_piv_deform_unit`,
   `_dvc_seed_unit`, `_dvc_local_unit`, `_dvc_global_unit`). New `_shared/analysis_units`
   pieces:
   - `UnitFrames` / `unit_frames`: a unit's frames by source t, read lazily in-process, or
     shipped stacked or memory-mapped to a worker.
   - `run_stage_units`: rows, payload, runner, store and parts back in unit order.
   - `stage_result`, `finish_stage`.

   A staged trial saves the unit's FULL rows (input columns + the card's new ones) and its grid
   rows, so Load Analysis reads a stage's trial as the complete table at that stage. The cards
   gained the `results` / `trial_label` sockets.
2. **SerialTrack runs and saves per group** (`track.objects`, `method = serialtrack`).
   - **Units:** (m, channel, z) in 2-D, (m, channel) in 3-D. Each group is linked in a worker
     with a fresh id counter, then offset into the shared ids in group order. The node already
     renumbers tracks through `build_membership`, so this gives the same tracks the shared
     counter did.
   - **Saving:** each group's links are saved as they finish. At the end, a `final` unit saves
     the Track table and the member table with its `track_id`, so Load Analysis restores the
     whole tracking result.
   - The store sockets are gated to `method = serialtrack`, since the other linkers are fast.
     `store_sockets(available_in=…)` is new.
3. **Load Analysis skips a trial's working tables** that have no placement (SerialTrack's
   per-group links).
4. **The GUI probe shuts the parallel pools down before its final `os._exit`.** With the
   analysis lanes now spawning workers inside the probe, the exit-139 teardown crash after
   "ALL PHASE-5 GUI PROBES PASSED" became frequent: 3 of 5 runs. With the pool shut down first,
   3 of 3 runs exited 0.
5. Docs: MANUAL §13b (what saves, including staged cards and SerialTrack); CON-26 text.

## Why

The user asked for "any analysis method" to run across the cores (m, and m × z for 2-D, never
t), and for any DIC / DVC / PIV / serialtrack analysis to save as it is produced and load back.
The single nodes were converted in fbc29a1. The staged chains are the workflow the user asked
for when splitting DVC/DIC/PIV, and SerialTrack is named in the request.

Design calls:
- **Unit workers are the old loop bodies moved verbatim.** Frames are converted exactly as the
  serial reader did (float64 planes, float32 volumes), so a one-lane run is the old code path
  with the same numbers.
- **A staged trial stores full rows, not just its new columns**, so it stands alone in Load
  Analysis. This duplicates some disk across stages (tables are small next to the images).

Proof (selftests):
- The staged DIC / PIV / DVC chains still equal their single nodes (7.1e-15, 1.4e-13, 1e-6 µm).
- On 2 positions, every staged chain on process lanes equals its serial run (≤1e-12 DIC/PIV,
  ≤1e-10 DVC).
- SerialTrack over 4 groups gives identical tracks on lanes and serially, reloads 4/4, and Load
  Analysis restores the Track table and the members' `track_id`.

## Files

- `nodegraph/catalog/_shared/analysis_units.py` (`UnitFrames`, `run_stage_units`,
  `stage_result`, `finish_stage`, gated `store_sockets`)
- `nodegraph/catalog/analysis/dic_stages.py`, `piv_stages.py`, `dvc_stages.py` (unit workers,
  store sockets)
- `nodegraph/catalog/track/objects.py` (per-group SerialTrack, final unit, gated store sockets)
- `nodegraph/catalog/io/load_analysis.py`
- `scripts/_nodelab_v2_phase5_probe.py` (pool shutdown before `os._exit`)
- `MANUAL.md`, `codemap/concepts.md`
- **Hotspots:** `nodegraph/selftest.py` (`test_analysis_parallel`: staged chains + SerialTrack),
  `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`,
  `scripts/catalog_baseline.json` (re-blessed: store sockets on 9 nodes)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_analysis_parallel(); s.test_dic_stages(); s.test_piv_stages(); s.test_dvc_stages()"`
- In the app, pull a staged DVC chain on a multi-position ND2:
  - the progress note shows lanes;
  - `<file>.analysis/` holds `dvc_local_match-…` and the other stage trials.

## Gates

- [x] Focused selftests: codemap, node demos, socket contract / docs / option docs, column
      catalog, import hygiene, DIC/PIV/DVC catalog, store, quick scope, mesh, the three staged
      chains, track objects, track field, tracking, catalog. Plus `test_analysis_parallel`.
      Full snapshot run: next entry
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
      PROBES PASSED, exit 0 on 3 of 3 runs after the pool-shutdown fix
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (131 ops; re-blessed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
- 3af51b2 on its snapshot:
  - selftest: 186 ok up to the pre-existing `test_write_movie` failure;
  - the 38 tests after it passed.
