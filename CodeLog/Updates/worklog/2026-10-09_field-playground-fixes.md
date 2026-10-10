# Four fixes found by running the field playground on real data

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

Four fixes. Each was found by pulling the field-mechanics playground graph through the GUI's
own runner on real data, and each has a regression test.

**1. PIV no longer segfaults when a pass leaves a NaN vector** (`nodegraph/kernels/piv_field.py`).
- **Where.** Both drivers, the ensemble and the per-pair `_deform_pass`, used by `analysis.piv`
  and the staged PIV cards.
- **Fix.** The next pass's predictor now comes from the new `_predictor()`, which reads masked
  AND non-finite windows as 0 px. Before, only masked windows were zeroed.
- **Behaviour.** A window with no correlation peak now just predicts no shift. It used to:
  - make the global spline all NaN;
  - make the deformation coordinates all NaN;
  - segfault `scipy.ndimage.map_coordinates`, first in the pool worker and then in the serial
    fallback, which runs in the GUI process.
- **Test.** `test_piv_nan_predictor`.

**2. A chained series saves its analysis results** (`nodegraph/metadata.py`
`chained_metadata`).
- **What it was.** Chaining files onto T, C or Z (Timeseries Builder) dropped the per-position
  `source_path`, because the files' paths disagree. That left `open_node_trial` with "no source
  file to save beside", so PIV, DIC and DVC on any chained series silently saved nothing and
  recomputed on every launch.
- **Fix.** `source_path` now keeps the FIRST file's path, in chain order, so results sit beside
  the series' first file. Every other per-position key still drops when the files disagree.
- **Test.** An assertion in `test_chain_files`.

**3. The analysis store writes on drives that refuse big writes, and a failed save no longer
fails the analysis** (`nodegraph/analysis_store.py`, `catalog/_shared/analysis_units.py`).
- **Chunked writes.**
  - Every store file is written in `WRITE_CHUNK` (4 MiB) pieces, then renamed into place.
  - Parquet units are built in memory first, instead of pyarrow's one-call file write.
  - A refused write removes its temp file.
- **Saves never fail a run.**
  - A unit whose save raises `OSError` keeps its result and carries `save_error`, in the
    worker path and the serial path. The serial path used to swallow the error silently.
  - `UnitRun.save_errors` lists them.
  - `stamp_trial` writes "N unit(s) NOT saved (results are complete, the trial is partial):
    <reason>" into the output's `analysis_trial_note`.
- **Test.** `test_analysis_store_writes`.

**4. Plot Field centres a diverging map on zero** (`nodegraph/catalog/plot/field.py`).
- **Fix.** With blank Colour range limits and the coolwarm map (Auto picks it for a signed
  column: a strain component, curl, pressure, rotation), the limits are now symmetric about
  zero. Typed limits are kept as typed.
- **Why it matters.** This is the Viewer Field overlay's `center_zero` default; the figure
  used to disagree with the screen.
- **Docs.** The socket and option docs say so.
- **Test.** Assertions in `test_plot_field`.

## Why

The user asked for a node graph of the field-mechanics work to play with. Before handing it
over, every page was pulled headlessly through `EngineRunner`, exactly what opening the file
and pulling the last card does, and these fell out.

**1. The PIV crash.**
- **Symptom.** The chip-flow page (ensemble PIV over 36 positions × 11 planes) died with exit
  139 after about 75 s.
- **Diagnosis.**
  - Faulthandler put both crashes, the worker's and then the serial fallback's in the GUI
    process, in `map_coordinates`.
  - Trapping its inputs showed 2,097,152 of 2,097,152 coordinates non-finite.
  - Replaying the dumped unit (position 0, plane 5) traced it to 3 NaN vectors in plain-
    background corner windows. Nothing was flagged, so replacement never ran, and
    `RectBivariateSpline`'s global fit spread the 3 NaNs over all 3,969 predictor values.
- **Why fix the predictor.** Sanitising there is the narrowest fix that covers both drivers,
  and it matches openpiv's own `np.ma.filled(u, 0)` intent ("no measurement → no shift").

**2. The lost results.**
- **Symptom.** The chip page recomputed 7 minutes of PIV on every pull, while the synthetic
  pages reloaded in 0.3 s.
- **Why keeping the first path is safe.** The trial's identity (`Engine.lineage`) already
  identifies every chained file at its own source card. `source_path` only has to say WHERE
  the results go, and `_store_root` already documented "beside the first position's source
  file".

**3. The store writes.**
- **Symptom.** Once fix 2 let the chip run save, every unit's save failed: pyarrow raised
  `OSError: [Errno 22] Invalid argument`. Because the worker re-raised it, the whole PIV pull
  failed after its compute.
- **Measurement.** S: (13 TB, 7 % used) accepts a single 16 MiB write and refuses 24 MiB, from
  Python and from pyarrow, even with small row groups.
- **Why chunk and soften.** Chunking makes the store work there. Not failing the run
  restores the store's contract that saving is a convenience: the design note already said a
  crash "loses nothing that finished", but a failed save was discarding a finished analysis.

**4. The plot limits.**
- **Symptom.** The playground's rotation figure ran from −1.1° to 4.5°, so its white point sat
  at 1.7° and the unrotated gel read as negative rotation. That broke the option doc's own
  promise, "zero neutral".

## Files

- `nodegraph/kernels/piv_field.py`: `_predictor()` in `_deform_pass` and `run_piv_ensemble`.
- `nodegraph/metadata.py`: `chained_metadata` keeps `source_path` off M; `SOURCE_PATH_KEY` doc.
- `nodegraph/analysis_store.py`: `WRITE_CHUNK`; chunked `_atomic_write_bytes`; in-memory
  `write_table`.
- `nodegraph/catalog/_shared/analysis_units.py`: `save_error` on a unit,
  `UnitRun.save_errors`, the note in `stamp_trial`.
- `nodegraph/catalog/plot/field.py`: zero-centred limits on a diverging map; docs.
- `nodegraph/selftest.py`: new `test_piv_nan_predictor` and `test_analysis_store_writes`;
  assertions in `test_chain_files` and `test_plot_field`.
- Regenerated: `scripts/catalog_baseline.json` (two plot.field descriptions), `codemap/gen/*`,
  `codemap/STATE.md`, `codemap/node_synopsis.json`.

Shared files both developers edit: `nodegraph/selftest.py`, `codemap/gen/*`,
`scripts/catalog_baseline.json`.

## How to verify

- Run `run_named.py . test_piv_nan_predictor test_piv_stages test_catalog_piv
  test_chain_files test_plot_field test_analysis_store_writes test_analysis_store
  test_analysis_parallel`.
- **Proof the PIV test catches the bug.** With `_predictor` swapped back to
  `np.ma.filled(u, 0.0)`, the 256² case returns no finite vector (the 1024² real unit
  segfaults).
- **Real data.** The playground's chip-flow page should complete, save
  `Count00000_ChannelFITC_Seq0001.analysis/` beside the first ND2 on S:, and reload it on a
  second pull. The timings are in `2026-10-09_field-playground-graph.md`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: gated together with the playground
  commit; see that worklog.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`: as above.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (baseline re-saved for the
  two plot.field descriptions).
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The branch is ahead of origin;
  PR #1 is open.
