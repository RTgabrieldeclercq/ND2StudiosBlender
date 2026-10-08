# UI lag + troubleshooting-mode slowness: thread caps, palette, spreadsheet, dot grid

- **Date:** 2026-10-08
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** graph-node-sets @ 8fc61af

## What changed

Four GUI-thread costs that made the window lag during any pull, and made every cursor
move under troubleshooting mode (`F9`) take seconds, are bounded or removed:

1. **Thread caps.** `nodegraph.parallel.cpu_budget()` defaults to `min(cpu_count-2, 8)`
   (was `cpu_count-2`, i.e. 78 workers on the lab's 80-core workstation), and the runner's
   shared display pool (plane decode, prefetch, playback warm-up, viewport detail) is capped
   at `display_workers()` = `min(8, cpu_count-2)` (was Qt's default of one thread per core,
   and `preload_series` fanned out to `maxThreadCount-1` of them). `NODEGRAPH_WORKERS` and
   the new `NODELAB_DISPLAY_WORKERS` override. The ingest lane keeps its old sizing (it is
   now derived from the core count directly, so the cap does not shrink it).
2. **Label palette.** `label_paint.deconflict_slots` does the per-object neighbour check in
   plain Python over a memoised hue table and scores the clearing-slot search in chunks of
   64 candidates, nearest first — ~4× faster, and verified slot-for-slot identical to the
   old implementation on random, label-like, dense (>32 hues) and full-wheel inputs. The
   Viewer's `MAX_PALETTE_OBJECTS` drops from 40 000 to 10 000, which bounds the
   per-delivery palette build at about a quarter of a second.
3. **Spreadsheet.** The panel's grid is a `QTableView` over a virtual
   `QAbstractTableModel` that formats a cell only when it is painted (was a `QTableWidget`
   filled with one item per cell). Column widths are measured over the header and the first
   200 rows. Per-file tabs and the original-row-number header behave as before; the GUI
   probe reads the table through the model.
4. **Canvas dot grid.** `GraphView.drawBackground` fills the exposed rect with one cached
   STEP×STEP texture brush (one dot, rendered at the device pixel ratio) instead of one
   `drawEllipse` per dot.

MANUAL: the env-var table (`NODEGRAPH_WORKERS` default and why, the new
`NODELAB_DISPLAY_WORKERS`, `NODELAB_INGEST_WORKERS` sizing) and two new §18 rows ("under
F9 every cursor move still takes seconds" and "the whole interface lags while a pull
runs"). The codemap is regenerated (only the volatile closure fingerprints move).

## Why

Users reported that troubleshooting mode had become very slow and that the whole UI was
laggy. Reproduced offscreen with a real ND2 (66 positions × 10 z × 1024², direct access)
through Load → Select → Gaussian → Threshold → Label, sampling the GUI thread's stack
whenever one `processEvents()` ran long:

- **The scope itself works.** Under `F9` the pin reaches every source seed on the flat demo,
  on the four-page example workspace and on real data; a scoped Threshold computes in
  ~0.25 s. The seconds went elsewhere, all of it on the GUI thread.
- **78 engine threads starved the GUI of the GIL.** A normal Threshold pull took 53.6 s
  with the window frozen for 31.6 s at a stretch. With 8 workers the same pull took 49.7 s
  (no slower) and the longest freeze was 0.3 s; 16 workers were slower than both (69 s).
  The workers are Python threads in the GUI's own process, numpy releases the GIL but the
  per-unit glue does not, and past a handful of threads the GIL is the bottleneck — so the
  extra 70 threads bought nothing and every Qt→Python callback (each paint, each queued
  signal) waited behind them. PySide also releases the GIL around every painter call, so a
  loop of 5 000 `drawEllipse` calls that paints in 5 ms on an idle machine did not finish in
  five minutes beside eight CPU-bound Python threads — which is why the dot grid became one
  fill. Capping threads rather than moving the engine out of process is the fix that fits in
  a day; the process-pool lane already exists for CNN inference and nothing else.
- **A Label result cost ~12 s of GUI thread per delivery, after the 0.25 s compute.** A
  noisy threshold labels 21 620 specks. The Viewer built its identity palette on the GUI
  thread (`deconflict_slots`, ~90 µs an object: ~5 s) and the Spreadsheet created a
  `QTableWidgetItem` for every cell (~7 s). Under `F9` every cursor move delivers a new
  payload, so every move paid both. After: palette 22 ms, spreadsheet ~100 ms for the same
  table. The palette cap is lowered rather than removed because de-confliction is a
  courtesy for the eye comparing a few hundred neighbouring cells, not for ten thousand.
- The "requires a visual output node to crop the signal" part of the report could NOT be
  reproduced: the region box arms on whichever node the active Viewer is bound to (any
  double-clicked or previewed card), and the pin carried the window on Threshold, Label and
  Measure alike. Most likely what was seen is the Viewer only showing (and so only boxing) a
  node once it has been pulled into it; with *Preview clicked node* off, selecting a plain
  card does not do that while selecting a Viewer node does.

## How to verify

- `F9`, drag the T/M cursor on a Label result of a noisy threshold: the Viewer follows in
  well under a second (was 6–10 s per move), the Spreadsheet fills instantly.
- Start a long pull and keep clicking around: the window stays responsive (the pull itself
  is no slower; on the 80-core box it is slightly faster).
- `NODEGRAPH_WORKERS=78 python run.py` brings the old behaviour back for comparison.
- Offscreen: the stack-sampling probes used for the measurements live in the session's
  scratchpad, not the repo; the method is a watchdog thread that reads
  `sys._current_frames()[main]` whenever one `processEvents()` exceeds 80 ms. (`cProfile`
  cannot do this on the venv's Python 3.14 — it profiles every thread, so engine work on
  the worker shows up as if it ran on the GUI thread.)

## Files

- `nodegraph/parallel.py` — `DEFAULT_WORKER_CAP`, `cpu_budget()` capped, docstring
- `nodelab_v2/runner.py` — `display_workers()`, shared pool capped, ingest sizing from cores
- `nodegraph/catalog/_shared/label_paint.py` — `deconflict_slots` rewrite, `_clearing_slot`
- `nodelab_v2/viewer.py` — `MAX_PALETTE_OBJECTS` 40 000 → 10 000
- `nodelab_v2/spreadsheet.py` — `_ColumnsModel`, `QTableView`, `_size_columns`
- `nodelab_v2/scene.py` — `drawBackground` texture brush, `_grid_brush`
- `scripts/_nodelab_v2_phase5_probe.py` — spreadsheet assertions read the model
- `MANUAL.md` — env-var table, §18 rows
- `codemap/STATE.md`, `codemap/gen/*` — regenerated (hotspot files; not hand-merged)

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every test passes EXCEPT
  `test_write_movie`, which fails with the identical assertion (the same six frame means)
  on the base commit 8fc61af in a clean worktree — pre-existing, unrelated to this change.
  Because `main()` stops at the first failure, the 42 tests after it were run individually
  in order (`test_movie_timeline` … `test_canvas_pill_and_reorganize`): all pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
  (twice; one run exited 139 after printing PASSED, which the base commit does too — the
  known offscreen teardown crash, not a probe failure)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (115 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> branch is on top of `graph-node-sets`, which is
  ahead of `origin/Blender` and behind by 0, so there was nothing to rebase
