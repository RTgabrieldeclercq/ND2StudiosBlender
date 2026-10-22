# Cyto nuc ratio node

- **Date:** 2026-10-22
- **Author:** Gabriel Declercq
- **Branch:** cyto-ring-node
- **Base:** 572312ab

## What changed

A new node, **Cyto / Nuc Ratio** (`analysis.cyto_nuc_ratio`), builds a cytoplasm region for
each nucleus and reports, for every wired signal channel, each cell's nucleus and cytoplasm
intensity and their ratio.

- **`ring`:** the cytoplasm is the voxels `gap`…`gap + ring_width` µm from the nearest
  nucleus edge, measured as true Euclidean distance on the anisotropic grid. Neighbouring
  rings meet halfway.
- **`cells`:** the cytoplasm comes from another channel's cell segmentation (the `cells`
  input), with the nuclei cut out of it, so no nuclear voxel is ever counted as cytoplasm.
  - A label raster is matched to nuclei by overlap. A binucleate cell is split by nearest
    nucleus, and an anucleate cell is dropped.
  - A plain mask is split by nearest nucleus.
- **Signals:** `signal` … `signal_8` are growing slots; the card shows one empty slot after
  the last wired one.
- **Output:** columns land on the nucleus table, named after each signal's channel:
  `<ch>_nuc_<stat>`, `<ch>_cyto_<stat>`, the optional `<ch>_background`, and
  `<ch>_cyto_nuc_ratio` (or `_nuc_cyto_ratio`), plus `nuc_voxels` / `cyto_voxels`. The
  cytoplasm is kept as a label instance whose ids are the nucleus ids.
- **Modes:** Statistic `mean`/`median`, Background `none`/`outside` (the median outside
  every region), and Ratio direction.
- **Dimensionality** is inherited from the nuclei table's `z_kind` (§7b): a 3D nucleus gets
  a 3D shell.

**Hotspot files touched:** `nodegraph/selftest.py` (`test_cyto_nuc_ratio`, placed after
`test_load_table`), `nodegraph/catalog/__init__.py` (one `MODULES` line), `codemap/gen/*`
(regenerated; `nodes.jsonl` diff is the one added op), and `scripts/catalog_baseline.json`
(saved: op added).

## Why

The user asked for a node that takes a nuclear segmentation and extends a cytoplasmic ring
around it. The ring is either artificial or a separate segmentation from another channel. Where
the cell segmentation overlaps a nucleus, the nucleus removes the overlap, so the
cytoplasm-to-nucleus ratio is preserved. Signals come in through signal sockets, and more can
be added.

Existing nodes did not compose into this:
- Boundary Band gives a band label raster but no table and no ratio.
- Measure gives per-region statistics, but on one label layer and one image per node.

A graph of Grow + arithmetic + Measure ×2 + Join per signal would have been six nodes per
signal, with no nucleus-overlap removal.

Design choices:
- **One channel per signal socket**, because `_require_same_grid` (the second-input rule every
  measuring node follows) requires equal axes. That matches the canonical channel-tap
  workflow.
- **Dimensionality inherited, not a lever,** for the reason Measure gives: a lever could
  disagree with how the nuclei were segmented.
- **Physical distance through EDT `sampling`**, so one µm ring width is a sphere on an
  anisotropic stack. There is no `_z` companion socket.

## How to verify

1. Load a two-channel ND2 and Channel Select DAPI. Run Segmentation on it, then this node.
2. Channel Select the reporter (for example p65) and wire it into `Signal 1`. A `Signal 2`
   slot appears.
3. Pull the node.
   - The Spreadsheet's nucleus table gains `p65_nuc_mean`, `p65_cyto_mean` and
     `p65_cyto_nuc_ratio`.
   - Draw the `cytoplasm` layer to check the rings.
   - For `cells` mode, wire a membrane-channel segmentation into `cells`.

Headless: `test_cyto_nuc_ratio` in the selftest, on a synthetic plate whose ratio is exactly
3.

## Files

- `MANUAL.md`
- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/node_roles.json`
- `codemap/node_synopsis.json`
- `nodegraph/catalog/__init__.py`
- `nodegraph/selftest.py`
- `scripts/catalog_baseline.json`
- `nodegraph/catalog/analysis/cyto_nuc_ratio.py`

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (254 `[ok]`, incl. `test_cyto_nuc_ratio`, node demos, socket contract/docs; `NODEGRAPH_MIN_FREE_RAM=1G`)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED, exit 0 (172 `[ok]`; no node-specific probe — the grow-slot behaviour is covered Qt-free in the selftest)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 145 ops (baseline saved for the new op)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> 0 behind origin/Blender; stacked on `load-table-node` (both touch the same hotspot files, so stacking avoids a second merge of them)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
