# Saved analysis trials get a spare folder

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

**A unit the source's folder refuses is saved in a spare folder, and every run looks there.**
- **Where.** The spare is `~/.nd2studios/analysis/<file>.analysis.<digest>/`, one folder per
  source path; the digest keeps two same-named files apart. It holds the same trial folder
  layout as beside the source.
- **Saving.** A worker (`_unit_task`) and the in-process path (`Trial.save`) both try the
  folder beside the source first, then the spare (`analysis_store.save_unit_to`). Only when
  both refuse is the unit "NOT saved", as before.
- **Finding.** `Trial.has` / `load` / `done_units` search both folders, so a re-run loads
  whatever either holds.
- **A source folder that cannot hold a trial at all** (it cannot even be created):
  `open_node_trial` opens the trial in the spare and keeps the source's folder as the
  second place it reads. Before, this case saved nothing.
- **Headers.** Once the spare holds a unit, both halves' `trial.json` name each other
  (`also_saved_in`). `list_trials`, `read_trial`, the new `list_source_trials` /
  `find_trial`, `trial_choices` and the inspector's trial list therefore see the whole
  trial from either folder. Load Analysis looks a trial id up in both.
- **Telling the user.**
  - The card's progress line adds `· saved in the spare folder` or `· NOT saved` per unit.
  - The output's `analysis_trial_note` counts both cases.
  - Before, a failed save was recorded only in that metadata note, which nothing in the
    window shows.

**Docs:**
- the `results` description on every saving node, and Load Analysis's `trial` description;
- MANUAL §13b;
- concept card CON-26.

## Why

The user ran the field playground's chip-flow page, restarted the app, and re-ran it. PIV
started again from the beginning.

**The cause.**
- The re-run did find the trial (`S:\alia chip flow data\Count00000_ChannelFITC_Seq0001.analysis\piv-b2dd260f`).
- That trial held 1 of its 396 units. S: is failing: `iaStorE` 4155 and NTFS 140 have been
  logged about twice a second all evening.
- So every unit's Parquet write failed, while the small `trial.json` / `index.json` writes
  succeeded.
- Since dc55c6d a failed save no longer fails the run. But the result then lived only in
  memory, and the restart dropped it.

**Why not just tell the user to move the data.** They should, and the reply says so. But a
read-only share or a full disk hits the same wall, and the design of the store is "an hour
of DVC survives a restart". A per-user local folder is the one place that is always
writable.

**Alternatives rejected:**
- `NODEGRAPH_STORE_DIR` already moves the store, but it is an environment variable set
  before launch, and it moves ingest stores too.
- A write probe at open time cannot predict a flaky disk: here the header writes passed and
  the unit writes failed.

## Files

- `nodegraph/analysis_store.py`:
  - `SPARE_DIR`, `spare_root`, `trial_roots`, `save_unit_to`;
  - `Trial.alt_root` and its two-folder `has` / `load` / `save` / `update` / `clear`, plus
    `alt_units`;
  - `open_trial(alt_root=)`;
  - `list_trials` and `read_trial` follow `also_saved_in`;
  - `list_source_trials`, `find_trial`; `trial_choices` uses them.
- `nodegraph/catalog/_shared/analysis_units.py`:
  - `_spare_root`;
  - `open_node_trial` passes the spare, and swaps roots when the source's folder refuses
    the trial;
  - the worker payload carries the spare data dir;
  - `saved_spare` on a result, `UnitRun.spare_saves`, `_save_note`, `stamp_trial`;
  - the `results` description.
- `nodegraph/catalog/io/load_analysis.py`: trial lookup via `find_trial`; the `trial`
  description.
- `nodelab_v2/document.py`: `trial_choices` via `list_source_trials`.
- `nodegraph/selftest.py`: `test_analysis_store_writes` extended.
- `MANUAL.md` §13b, `codemap/concepts.md` CON-26.
- Regenerated: `codemap/gen/*`, `codemap/STATE.md`, `codemap/curated.lock.json` (CON-26
  blessed), `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` (two descriptions).

Shared files both developers edit: `nodegraph/selftest.py`, `codemap/gen/*`,
`scripts/catalog_baseline.json`. (The two `graphs/*alia*` files the scaffold listed are the
user's own untracked graphs and are not part of this change.)

## How to verify

1. `test_analysis_store_writes` checks:
   - a source folder that refuses every unit sends both a worker's unit and
     `Trial.save`'s unit to the spare;
   - the trial loads them and counts them done;
   - both headers name each other, so `list_trials` from either root reports 2/2,
     `read_trial` from either returns both units, and `trial_choices` shows "2/2 complete";
   - a FILE where the source's `.analysis` folder would go makes `open_node_trial` open the
     trial in the spare, with the source's folder as its second root;
   - the output note counts the units saved in the spare.
2. `test_analysis_store` (DIC through the Engine on process lanes) is unchanged and passes.
   It now runs the five-part worker payload.
3. In the app: run the chip-flow page of `graphs/field_mechanics_playground.nd2graph.json`.
   The PIV card's progress line says `saved in the spare folder` per unit, and
   `~/.nd2studios/analysis/Count00000_ChannelFITC_Seq0001.analysis.<digest>/piv-b2dd260f/`
   fills. Restart and pull again: the units load instead of computing.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: see the commit report.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`: see the commit report.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (baseline re-saved: the
  `results` and Load Analysis `trial` descriptions).
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-26 re-read, extended, blessed).
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed.
