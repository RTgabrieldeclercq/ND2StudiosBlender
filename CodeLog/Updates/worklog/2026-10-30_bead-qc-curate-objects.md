# Bead QC: Curate Objects, applied to the IL-8 Run 4 graph

- **Date:** 2026-10-30
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

A new table node, **Curate Objects** (`table.curate_objects`), applies a review to a table of objects by id: `Drop` removes every row of the listed ids (matched against the ids before any merge; `a-b` ranges allowed), `Merge` takes `from>to` pairs and gives the rows of `from` the id `to` (chains resolve; cycles, two-way merges and a merge into a dropped id are refused), and every kept row keeps its old id in `<id column>_before`. With a blank `Name` it rewrites the table it reads, which it allows only for a table a table node wrote (Table Concat, Unique Objects), never a measured table whose raster indexes the rows; with a name it writes a new table. The IL-8 Run 4 graph (`graphs/IL8_run4_two_configs.nd2graph.json`) now has one after each Table Concat (`CUR` on both pages, `CURM` on the C6 eight-pass chain): C5 drops 31 beads and merges 10 split tracks, C6 drops 10 beads and merges bead 32 into 31. Everything downstream (standard curve, the 400 um split, the Crank fit and its Monte Carlo, every plot and map) now sees the curated table.

## Why

The user reviewed every Run 4 bead in a QC panel (crops of every pass, flags from the raw 12-bit images) and chose which to discard: beads measured against bright structures, scan-line streaks, debris, field-edge fragments and out-of-focus beads. While checking C6 bead 32 we found that Unique Objects (10 um tolerance) gives a bead a new id when its whole field jumps further than that between passes (the JOBS plate drift is common-mode, 5-30 um): field 37 holds one Cy5 bead in all 52 passes, tracked as 31 except at T10-T11 (a 13 um jump) as 32, with a continuous MFI (63, 81, 90, 94). Every merge in the graph was checked the same way (one bead in that field in every pass, continuous signal). A list-based node was chosen over chained If/Else nodes (four conditions each, about eleven nodes for C5, and no way to remap an id) and over raising the Unique Objects tolerance (it would renumber every object and invalidate the reviewed id lists, and single linkage could chain neighbouring beads). Effect: C5's rate rises early (T2 0.22 -> 0.46 fg/s) and its well level falls (T2 188 -> 102 pg/mL); peak 1.22 fg/s (T5). C6 is unchanged to within Monte Carlo noise.

## Files

- `nodegraph/catalog/table/curate_objects.py` (new node)
- `nodegraph/catalog/__init__.py` (hotspot: one import line, after Unique Objects)
- `nodegraph/selftest.py` (hotspot: `test_curate_objects`, registered after `test_unique_objects_langmuir`)
- `scripts/catalog_baseline.json` (hotspot: regenerated with `save`)
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` (regenerated), `codemap/node_roles.json` (table_synthesis), `codemap/node_demos.json` (demo after Unique Objects)
- `graphs/IL8_run4_two_configs.nd2graph.json` (CUR / CURM inserted; saved in the app's own format, so the diff is the new nodes only)
- `MANUAL.md` (§15 Tables: a Curate Objects row — what it is for and what bites)

`id_column` defaults to blank (= `object_uid`), not to `object_uid` itself: no structure producer declares that column (Unique Objects is a table node), so `test_column_catalog_complete` refuses it as a closed-dropdown default that would show as an orphan entry.

The other modified files in this working tree belong to earlier, separately logged work (2026-10-19 / 10-29 entries).

## How to verify

`PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as T; T.test_curate_objects()"`. For the graph: open `graphs/IL8_run4_two_configs.nd2graph.json`, select `CUR` on either page and read its progress line ("508 rows -> 396 ..." on C5), or run the paper dump (`PAGE=0 dump.py <graph> c5q`): C5 396 rows / 71 beads, C6 2,253 rows / 47 beads, rates matching a standalone refit to 0.0024 fg/s.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (253 ok, with `NODEGRAPH_MIN_FREE_RAM=1G`; a first run stopped at `test_parallel_governor` while a 318-figure export shared the machine, and passed alone; a second caught the `id_column` default, fixed above, and the missing MANUAL row)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (144 ops, after `save`)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (not pushed: no commit requested)
