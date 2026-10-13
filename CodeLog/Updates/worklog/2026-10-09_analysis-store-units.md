# Analysis units: saved trials beside the .nd2, process lanes, Load Analysis

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** 9509827

## What changed

1. **The single DIC, PIV and DVC nodes run unit by unit.** A unit is `(m, z)` for DIC/PIV and
   `(m,)` for DVC, and always covers the whole T series. `nodegraph/catalog/_shared/analysis_units.py`
   (`run_units`) drives them:
   - **Saved units load from disk.** Units the node's trial already holds are read back.
   - **The rest run across worker processes** through `parallel.adaptive_fold` (the governor
     from 9509827). With one lane they run in this thread, and the kernel's own progress still
     drives the bar.
   - **Results come back in unit order.** Ids are identical to the old serial loop.
   - **Each node's work moved into a module-level unit worker:** `_dic_correlate_unit`,
     `_piv_unit`, `_dvc_field_unit`. Big unit inputs (a DVC volume series) go to a worker as a
     memory-mapped temp file (`ship`). In-process, they are read lazily as before.
2. **Saved trials beside the source** (`nodegraph/analysis_store.py`).
   - **Layout:** `<stem>.analysis/<op>-<key8>/trial.json` + `data/<unit>.<kind>.parquet`, and
     a `<unit>.done.json` marker written last. Workers save their own unit, so a cancelled or
     crashed run resumes. `index.json` lists the trials and is rebuilt from `trial.json`.
     `NODEGRAPH_STORE_DIR` redirects the folder.
   - **Two new sockets on each node.** `results` (`reuse` / `recompute` / `off`) and
     `trial_label`; neither is part of the trial key.
   - **Output stamps** `analysis_trial`, `analysis_trial_note` and `analysis_trial_units`.
3. **`Engine.lineage(node, exclude=())`**, exposed as `ctx.lineage()` and
   `ctx.upstream_chain()`. It is an identity that is the same in every session: op, params,
   modes, wiring and upstream lineages, and each source's FILE (path + size + mtime). The trial
   key hashes it.
4. **`source_path`** is a new per-position metadata key, an absolute path, stamped beside
   `source_file` by `stamp_source_file`, the bundle envelope and the bundle card's seed. It is a
   `PER_POSITION_KEYS` member.
5. **New node `io.load_analysis` (Load Analysis).**
   - It reads a trial back: tables on their original layers, original stamps.
   - `onto`: `image` refuses an M/T/Z mismatch; `tables_only` needs no wire.
   - Its `trial` dropdown is a new presentation-only `SocketSpec.options_from="analysis_trials"`,
     filled by `GraphDocument.trial_choices` and shown by the inspector's `_options_box`.
6. **Docs.** MANUAL §13b and a §15 row; CON-26 (unit / trial / store) and CON-27 (governor)
   blessed; roles and demos for the new node.

## Why

The user asked for three things:
- "for any analysis method, we want to be able to parallelize the analysis across the cores
  ... m splits for 2D and 3D data, and m and z splits if its 2D ... time is the only one that
  cant be split";
- "save this output as they are produced into a temporary file ... named and placed along side
  the nd2 file it came from. We want to be able to load back any analysis";
- and then: reuse + resume, "keep as an option", "we may want to run multiple trials with
  various settings, and being able to switch between them", one folder (not files strewn
  beside the .nd2) with a JSON explaining types and settings plus a data sub-folder.

Design calls, recorded so they can be vetoed:
- **The unit is the save point and the parallel grain.** Saving per unit means one atomic file
  per worker with no locking. Folding in unit order keeps ids deterministic.
- **The trial key is a new session-stable lineage, not the memo's recipe hash.** The recipe hash
  folds per-session revisions, so it cannot find yesterday's file. A source is identified by
  its file rather than its provider, so ND2-direct and the ingest store agree. Upstream *code* is
  deliberately not in the key; `recompute` covers that case.
- **Parquet, not HDF5.** Each unit is its own atomically written file, so a crash cannot corrupt
  a run, and pandas / MATLAB / R read it directly.
- **A scoped (F9) pull never saves**, so quick looks stay out of the folder. The scope stamps
  arrive with the quick-analysis step.

Proof:
- **Process lanes ≡ serial.** DIC 2 units ≤1e-12, PIV 3 units ≤1e-12, DVC 2 units ≤1e-10, same
  rows and ids.
- **Reload.** A fresh Engine reloads 2/2 units with identical columns.
- **Resume.** A deleted unit is recomputed alone.

## Files

- New: `nodegraph/analysis_store.py`, `nodegraph/catalog/_shared/analysis_units.py`,
  `nodegraph/catalog/io/load_analysis.py`
- `nodegraph/catalog/analysis/dic_correlate.py`, `piv.py`, `dvc_field.py` (unit workers + store
  sockets)
- `nodegraph/engine.py` (lineage), `nodegraph/metadata.py` (`source_path`),
  `nodegraph/registry.py` (`options_from`), `nodegraph/catalog/util/merge.py` (comment)
- `nodelab_v2/runner.py`, `window.py` (`source_path` stamps), `document.py`
  (`trial_choices`), `inspector.py` (`_options_box`)
- `MANUAL.md` §13b + §15 row; `codemap/concepts.md` (CON-26, CON-27), `codemap/curated.lock.json`
- **Hotspots:** `nodegraph/selftest.py` (`test_analysis_store`, `test_analysis_parallel`),
  `nodegraph/catalog/__init__.py` (`io.load_analysis` appended), `codemap/gen/*`,
  `codemap/STATE.md`, `codemap/node_synopsis.json`, `codemap/node_roles.json`,
  `codemap/node_demos.json`, `scripts/catalog_baseline.json` (re-blessed: one new op, and every
  socket gains the `options_from` field)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_analysis_store(); s.test_analysis_parallel()"`
- In the app, pull a DIC/PIV/DVC node on an ND2:
  - A `<file>.analysis/` folder appears beside it.
  - Re-pull in a new session: the note reads `n/n loaded`.
  - Add **Load Analysis**, wire the image, and pick the trial.

## Gates

- [x] Focused selftests: the store, parallel, DIC/PIV/DVC catalog, socket contract / docs /
      option docs, registry, metadata pass, engine, memo, solo/subset scope, bundle
      source_file, chain files, split positions, catalog, codemap, node demos.
      Full run on the commit snapshot: see the next entry.
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`: run on the commit
      snapshot (next entry)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (130 ops; re-blessed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-26, CON-27 blessed)
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
- 9509827 (previous entry) on its snapshot:
  - selftest: 182 ok up to the pre-existing `test_write_movie` failure, then 39 more ok;
  - GUI probe: ALL PHASE-5 GUI PROBES PASSED.
