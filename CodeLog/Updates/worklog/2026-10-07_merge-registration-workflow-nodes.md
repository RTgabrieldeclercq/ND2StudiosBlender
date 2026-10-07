# Merge registration workflow nodes

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** graph-node-sets
- **Base:** e1ae0e28

## What changed

Merge of `registration-workflow-nodes` (four commits: the registration workflow as nodes — Select Plane / Split Z / Shift; range groups on the split cards; Split T / Select Frame with the fan-out cap and `every N`) into `graph-node-sets`, so the nodes are in the app the owner runs. Conflicts were all in the generated files plus two appended lists: `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` and `codemap/curated.lock.json` were taken from the `graph-node-sets` side and REGENERATED (`_codemap.py write`, `_node_synopsis.py write`, `_catalog_snapshot.py save`), never hand-merged; `nodegraph/catalog/__init__.py` (both sides appended to `MODULES`: `detect.beads` and the five workflow nodes) and `nodegraph/phantom.py` (both sides added phantoms: `beads3d` with its confocal kernel, and the four registration worlds) keep both sides, `__all__` as one list. The `node_roles.json` `pages` change (a page kind's PRIMARY set, everything else under “More nodes”) merged cleanly with the new nodes' `op_pages` entries. INV-08 / INV-10 / INV-11 re-read and re-blessed: the regenerated lock from the other side had dropped the verification the branch carried, and all three are still true.

## Why

The owner could not find Split Z or Split T: the app runs from the main checkout, which was on `graph-node-sets` and did not contain the branch the work lived on (a worktree). Merging here rather than rebasing the branch keeps the four feature commits and their worklogs intact and lands them where the app loads its catalog; the dry run (`git merge-tree`) had shown the conflicts were confined to regenerated files and two appended lists, which the team-sync recipe resolves by regeneration and keep-both.

## Files

- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/node_synopsis.json`
- `scripts/catalog_baseline.json`
- `CLAUDE.md`
- `MANUAL.md`
- `codemap/gen/imports.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/invariants.md`
- `codemap/node_demos.json`
- `codemap/node_roles.json`
- `nodegraph/catalog/__init__.py`
- `nodegraph/catalog/_shared/drift_layers.py`
- `nodegraph/catalog/align/shift.py`
- `nodegraph/catalog/channel/split.py`
- `nodegraph/catalog/registration/align_to.py`
- `nodegraph/catalog/registration/stabilize.py`
- `nodegraph/catalog/util/select_frame.py`
- `nodegraph/catalog/util/select_plane.py`
- `nodegraph/catalog/util/split_positions.py`
- `nodegraph/catalog/util/split_t.py`
- `nodegraph/catalog/util/split_z.py`
- `nodegraph/codemap.py`
- `nodegraph/kernels/README.md`
- `nodegraph/kernels/registration.md`
- `nodegraph/kernels/registration.py`
- `nodegraph/metadata.py`
- `nodegraph/phantom.py`
- `nodegraph/registry.py`
- `nodegraph/selftest.py`
- `nodelab_v2/demo_recipes.py`
- `nodelab_v2/demo_window.py`
- `nodelab_v2/document.py`
- `nodelab_v2/ops.py`
- `nodelab_v2/picker.py`
- `scripts/_nodelab_v2_phase5_probe.py`
- `scripts/registration_synthetic_bench.py`

## How to verify

`python run.py` from this checkout → the palette offers Split Z, Split T, Select Plane, Select Frame (Input / Refine pages) and Shift (Refine) alongside Bead Finder; the gates below.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (encoder; fails on both parents); the groups scheduled after it were run separately and pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the exit code after the PASSED line is the probe's documented teardown crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 114 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> not pushed; `graph-node-sets` is now ahead of origin by this merge and the push is the owner's call

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
