# Gate flakes

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

Two timing races that failed the full selftest on snapshot runs are fixed.

**1. LabLink beat thread (`nodelab_v2/lablink/worker.py`).** `_stop_beats` now joins the beat
thread (5 s timeout) after setting its stop event.
- Before, a pulse that had just woken could finish its `emit` after the command's reply: a beat
  for a command that was over.
- `test_lablink` caught this as "the beat thread outlived the command it was reporting on".
- It only failed under CPU load.

**2. `test_parallel_governor`'s "ran in ≥ 2 workers" check.**
- The check now warms two workers of the governor's pool width before the fold.
- Each test unit now takes 50 ms, so the units really spread over those workers. Before, they were
  instant, and one freshly spawned worker could drain all 12 before the second had started
  ("one worker process only").

## Why

Both failures were seen on snapshot gate runs:
- `test_lablink` on `3a919ff`;
- `test_parallel_governor` on `8790bd7`.

Neither reproduces when the test runs alone; both are scheduling races, not product logic.

**Why fix the worker, not the test.** The LabLink race is real behaviour: a stray beat after a
reply could confuse the hub's per-command bookkeeping. So the worker is fixed, rather than adding
a sleep to the test.

**Why fix the test, not the code.** The governor race is in the test's own assumption (spawn
timing), so the test is fixed and the code is left alone.

## Files

- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/modules.jsonl`
- `codemap/gen/symbols.jsonl`
- `nodegraph/selftest.py`
- `nodelab_v2/lablink/worker.py`

## How to verify

- `test_lablink`: 4 runs in parallel beside an 8-thread BLAS burner, all passed.
- `test_parallel_governor`: 3 runs in a row, all passed.
- The full selftest on a snapshot of this commit is in the next report.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [ ] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->

**Snapshot gates for `8790bd7` (Plot Field),** run in a worktree:
- **Full selftest:** stopped at `test_parallel_governor`, the race fixed here.
- **Rest of the selftest:** passed.
- **GUI probe:** printed ALL PASSED, then exited 139 at teardown — the intermittent post-PASS crash.
- **Catalog and codemap:** CATALOG IDENTICAL (137 ops), CODEMAP CURRENT.
