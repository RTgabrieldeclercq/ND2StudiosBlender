# Parallel governor

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

A resource governor for the expensive analysis nodes. New `nodegraph/governor.py` reads the machine live (logical AND physical cores, available RAM, system CPU load; dependency-free ctypes on Windows, /proc on Linux, psutil when present) and plans process lanes so lanes x inner threads never exceed physical cores minus a reserve (max(2, physical//10)) and lanes x worker RAM stays above a free-RAM floor (max(4 GiB, 10 %)). Its live half (`Governor`) starts at 2 lanes, re-plans on the first unit's MEASURED peak working set, holds new units while free RAM (minus what just-started units will still grow into) is under the floor, gives a lane back while other programs saturate the CPU, and restores it with hysteresis. New `parallel.adaptive_fold` is the rolling process-pool fold that consults it: a unit starts as soon as a slot frees and the governor allows it, results fold strictly in order through a reorder buffer, every task pins its worker's numba/BLAS threads and reports time + peak memory (`governor.timed_call`), compute errors re-raise as `UnitError` with the worker traceback, and a broken pool or an unpicklable compute finishes the rest serially in the calling thread. Every process-pool worker now runs at below-normal priority (`_worker_init`). New env overrides `NODEGRAPH_RESERVE_CORES`, `NODEGRAPH_MIN_FREE_RAM`. No node uses it yet; the analysis nodes move onto it in the next commits.

## Why

The user asked for the DIC/DVC/PIV/SerialTrack analyses to run across all cores, and then: "the parallel processing of these data should understand the computers architecture and availability for processing and adapt to it to ensure the computer doesnt freeze". The existing pools size once from cpu_count and installed RAM (capped at 8 for GIL reasons), which is right for cheap per-plane work and wrong for these units: a unit is a whole position holding gigabytes for minutes, numba inside each worker defaults to every core (8 workers x 80 threads on this box), workers ran at normal priority, and nothing watched free RAM - the recipe for paging and a frozen desktop. Below-normal priority is the main freeze guard (the OS always serves the GUI and input first); the core reserve and RAM floor are the second. Calibrating on a measured peak beats guessing a per-kernel memory model. Kept separate from `parallel.py`'s fixed-width maps, which the cheap nodes still use unchanged.

## Files

- `codemap/STATE.md` (**hotspot**, regenerated; every node's `fp` moves because `parallel.py` is a transitive dependency of the catalog)
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/symbols.jsonl`
- `nodegraph/parallel.py`
- `nodegraph/selftest.py` (**hotspot**: `test_parallel_governor` + `_pw_gov_unit`)
- `nodegraph/governor.py`

## How to verify

`PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_parallel_governor()"` - fake-machine plans and admission, then a real adaptive_fold over the spawn pool. `python -c "from nodegraph import governor; print(governor.describe())"` prints this machine's reading (here: cores=40p/80l reserve=4 ram=483/511GiB free floor=51GiB).

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (129 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
