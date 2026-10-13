# Lanes fixes staged metrics

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

Four fixes to the analysis process lanes, plus quality metrics on the staged cards.

- **Worker priority actually applies.** Analysis workers now really run at below-normal priority.
  The ctypes call passed the current-process pseudo-handle as a 32-bit int, so `SetPriorityClass`
  failed silently and every worker stayed at Normal. It is now HANDLE-typed. The new
  `test_parallel_governor` check reads `0x4000` back from each worker.
- **pyALDVC import is serialized.** Each worker imports `al_dvc._numba_compat` one at a time,
  under a machine-wide file lock: `governor.serialized_import`, `ALDVC_PREIMPORT`, and the
  `preimport` argument on `adaptive_fold`, `run_units` and `run_stage_units`.
  - Reason: pyALDVC's import-time cache probe writes and deletes a file in its `__pycache__`.
    Workers starting together collide on that file, and the losers can turn numba's disk cache
    off and recompile every kernel.
  - Used by: the DVC single node and its three frame-reading staged cards.
- **A pool failure is reported.** When the process pool cannot start or breaks,
  `adaptive_fold` sets `Governor.fallback`. That text becomes the run's progress note and the
  `analysis_trial_units` stamp ("process lanes failed (...); finished in this process"). Before,
  the run silently finished serially.
- **Docs: guard engine scripts.** A script that drives the engine must keep its top level under
  `if __name__ == "__main__":`. Recorded in the `adaptive_fold` docstring and CON-27.
- **Staged-card quality metrics.** The frame-reading staged cards now write the quick-analysis
  quality metrics as Global scalars, so an Iterate card can tune a staged chain section by
  section:
  - DIC Initial Guess: `dic_median_cc`.
  - DIC Local Match: `dic_solved_frac`, `dic_hole_frac`, `dic_median_iter`.
  - DIC Global Step: `dic_solved_frac`.
  - PIV First Pass and Deform Pass: `piv_valid_frac`, `piv_median_s2n`.
  - DVC Initial Guess: `dvc_seed_ok_frac`.
  - DVC Local Match and Global Step: `dvc_valid_frac`, `dvc_median_q`.

  They also write `analysis_est_full_s`. The MANUAL's quick-analysis table lists them.

## Why

The request was for analysis parallelism that "adapts to the machine so the computer doesn't
freeze". An earlier synthetic DVC benchmark seemed to show 8 lanes running 8× slower than
serial. Investigating it turned up two genuine bugs, the priority and the import race. But the
slowdown itself was an artefact of the benchmark script: it had no `__main__` guard, so every
spawned worker re-ran the whole benchmark when it imported the script.

Re-measured with the guard (8 synthetic DVC positions, 48×256×256, 3 frames, 40-core box):

| Run | Wall time |
|---|---|
| serial | 12.9–14.9 s (1.6 s per unit on every thread) |
| 8 lanes × 4 threads, cold (includes spawn and the 2-lane warm-up) | 16.0 s |
| 8 lanes × 4 threads, warm | 8.1–8.2 s |

Each warm unit takes 3.6 s on 4 threads, at 365 MB peak per worker. The steady state is about
3.6× the serial throughput; these units are tiny, so pool start-up dominates this benchmark.

The silent fallback is what made the artefact look like a scheduler problem. That is why the
fallback is now stated in the note rather than only logged.

## Files

- `MANUAL.md`
- `codemap/STATE.md`
- `codemap/concepts.md`
- `codemap/curated.lock.json`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/node_synopsis.json`
- `nodegraph/catalog/_shared/analysis_units.py`
- `nodegraph/catalog/analysis/dic_stages.py`
- `nodegraph/catalog/analysis/dvc_field.py`
- `nodegraph/catalog/analysis/dvc_stages.py`
- `nodegraph/catalog/analysis/piv_stages.py`
- `nodegraph/governor.py`
- `nodegraph/parallel.py`
- `nodegraph/selftest.py`
- `scripts/catalog_baseline.json`

## How to verify

`PYTHONUTF8=1 .venv/Scripts/python.exe -B -c "import nodegraph.selftest as s; s.test_parallel_governor()"`
checks four things:
- every worker reports below-normal priority;
- an unpicklable compute falls back;
- the fallback is stated in `Governor.fallback` and `note()`;
- the fold stays ordered.

`test_dic_stages`, `test_piv_stages` and `test_dvc_stages` cover the staged metrics.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->

Focused selftests run on this tree all passed: `test_parallel_governor`, `test_analysis_store`,
`test_analysis_parallel`, `test_quick_scope`, `test_analysis_mesh`, `test_dic_stages`,
`test_piv_stages`, `test_dvc_stages`, `test_socket_docs`, `test_option_docs`,
`test_param_socket_contract`, `test_codemap`. The full selftest and the GUI probe run on a
snapshot worktree of this commit afterwards; their results are in the next worklog entry.
