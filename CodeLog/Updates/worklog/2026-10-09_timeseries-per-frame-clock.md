# Timeseries per frame clock

- **Date:** 2026-10-09
- **Author:** Gabriel Declercq
- **Branch:** timeseries-per-frame-clock
- **Base:** e1ae0e28

## What changed

Timeseries Builder (`util.timeseries`), `File order = time`, onto T now reads EVERY
frame's timestamp in EVERY attached file and merges the files frame by frame
(`metadata.chain_frame_order`, a k-way merge on each frame's `frame_time_jd`). Files
whose frames alternate in time now interleave in the output (t0 from file 1, t1 from file
2, t2 from file 1, …), and each file keeps its own frames in their recorded order. Files
that do not overlap in time come out exactly as before, on the same provider. When frames
interleave, the block chain is re-read through the existing `FrameRemapProvider`, which
is still a pure index remap and still TILEABLE. The per-T clock (`frame_time_jd`,
`frame_datetime`) is reindexed by the same order in the shared `chained_metadata`, so the
pull and the edit-time envelope agree. The card's `__chain__` note gains
`interleaved` and, when true, `frame_file` (each output frame's file). A multi-file card
now carries each member file's own clock as a new per-position key,
`source_frame_time_jd` (stamped in `EngineRunner._resolve_bundle` via
`runner.bundle_source_clock`). It joins `PER_POSITION_KEYS` and is reindexed by
`time_subset`, so position or frame narrowing keeps it aligned. Files inside one card are
therefore also ordered and interleaved by their own clocks, and the built series gets
their real per-frame times instead of a dropped clock. `sequence` and `loaded` are
unchanged.

**Hotspots touched:** `nodegraph/selftest.py` (one test appended before `main`, called
last), `codemap/gen/*`, `scripts/catalog_baseline.json` (regenerated; the only catalog diff
is `util.timeseries`'s slot and `time`/`T` choice docs).

## Why

User request. Some microscopes write one run as a numbered sequence of files, and
neighbouring timepoints can sit in the NEXT file of the sequence. The old `time` order
sorted whole files by their first frame's clock and played each file as one contiguous
block, so such a run came out of time order with no error. Inside a multi-file card it
was worse: the card only carried its first file's clock, so the other files could not be
placed by time at all (they quietly kept name order under a note that still said
"acquisition-clock order"), and the built series lost its clock. Merging rather than
globally sorting frames was a deliberate choice. Within one file, frame order IS
acquisition order, so a clock glitch inside a file should never shuffle that file. The
merge degrades exactly to the old file sort when files don't overlap, so existing graphs
keep their memo identity. The per-file clock rides as a per-position key because a
bundle's other axes are already shared and its per-T `frame_time_jd` can only describe
one file.

Not fixed, found on the way: a bundle's payload takes member 0's whole display dict, so
its per-M `stage_xy_um`/`stage_z_um` are file 0's list (length n0, not K·n0) and get
rejected by every length check downstream. Bundles therefore lose their stage log, which
matters for Stitch-by-stage after a bundle. Worth a separate change.

## Files

- `MANUAL.md` (the Timeseries Builder row)
- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/node_synopsis.json`
- `nodegraph/catalog/util/timeseries.py`
- `nodegraph/metadata.py`
- `nodegraph/selftest.py`
- `nodelab_v2/runner.py`
- `scripts/catalog_baseline.json`

## How to verify

`PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as S; S.test_timeseries_frame_interleave()"`
pins alternating files (separate cards and one multi-file card), a within-file clock
glitch, non-overlapping files staying in blocks, `sequence`/`loaded` never interleaving,
the legacy unclocked-bundle drop, and the family helpers. In the GUI, load a sequence of
ND2s whose timepoints alternate between files as one multi-file card, add a Timeseries
Builder (Onto axis = T, File order = time), and the card's note reads "N frames from K
files, each placed by its own acquisition clock — the files interleave"; the Viewer's
timestamp overlay is then monotone across the series.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (219 `[ok]`, incl. the new `test_timeseries_frame_interleave`)
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> **FAILS, pre-existing.** It exits abnormally right after `PG14` (110 `[ok]` lines, no traceback, even with `-X faulthandler`), inside the LK1 linked-pages section. The base commit `8fc61af`, run from a clean worktree without this change, fails identically at the same line, so this change did not cause it. Needs its own fix.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (115 ops), re-blessed with `save`; the only diff was `util.timeseries`'s slot and `time`/`T` choice docs
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT after `write`. The diff was reviewed: every node's `fp` moved because `fp` is the transitive closure fingerprint and `nodegraph.metadata` is in every closure; 5 new symbols; the other moves are line shifts in the 4 touched files
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT after `write` (only `util.timeseries` changed)
- [ ] `python scripts/_sync_check.py` -> not yet: branch not pushed; it was cut from `graph-node-sets`, which is itself 44 commits ahead of `origin/Blender` and not merged

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
