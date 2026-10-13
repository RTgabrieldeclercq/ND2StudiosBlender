# Lattice layers unrolled by their own axes (Registration after a Timeseries Builder)

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** registration-timeseries-table
- **Base:** 572312ab

## What changed

Three places now index a lattice attribute by its array's real axes,
`AxisSizes.axis_list(domain)`. That list elides the batch axis `b` at `b == 1`, so a
Frame layer is `(M, T)`. Before, they used `axes_of(domain)` ∩ `AXIS_ORDER`, which
always includes `b`.
- `nodelab_v2/tables.py` `lattice_tables`: the Spreadsheet's (and the CSV export's) index
  columns for Frame / Plane / Timepoint / Multipoint layers.
- `nodegraph/catalog/zone/frame.py`: which axis of a lattice layer is T when a
  Simulation zone slices one frame.
- `nodegraph/transfer.py` `execute_bridge_plan`: the axes of a mid-route lattice carrier.
  It follows the same rule as `execute_transfer`, but keeps the full list if a carrier
  still holds a singleton `b`.

## Why

User report: Registration on a Timeseries Builder's T axis failed with
`ValueError: cannot reshape array of size 76 into shape (3,newaxis)` in
`lattice_tables`. Stabilize's per-frame drift is a Frame layer `(1, 38)` (the CRC
`_Seq` files build 8+8+8+7+7 = 38 frames), and the table built three index rows (`b`,
`m`, `t`) for a two-axis array.

This was not specific to registration or the builder. Since `b` joined the lattice
(2026-10-01, V3.01, elided from arrays at `b == 1`), every Frame table did this:
- **crash** when `2·M·T` is not a multiple of 3;
- otherwise run, but **silently** show wrong index columns. A 39-frame series listed
  `t` starting at 13, so drift rows were labelled with the wrong frames, on screen and in
  any export.

`zone.frame` used the same list and would slice the wrong axis of a Frame layer.

`axis_list` is the shape every lattice array actually has (`shape_for` is built from
it), and `execute_transfer` already used it. The other option, re-adding a singleton `b`
to every array, would undo the V3.01 rule.

## How to verify

1. Load a sequence, then Timeseries Builder → Registration (Stabilize).
2. Double-click the Stabilize node. The Spreadsheet shows the Frame table with `m`, `t`
   (0..T-1), `drift_y`, `drift_x`, with no traceback.

Automated: selftest `test_lattice_tables_elided_batch` covers:
- 38- and 39-frame Frame tables;
- a Plane table;
- a real batch, which keeps `b`;
- `zone.frame` on a Frame layer;
- builder → Stabilize → `all_tables`.

On the pre-fix code it fails with the user's exact `ValueError`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC (based on origin/Blender 572312a)

Hotspot files touched: `nodegraph/selftest.py` (one test appended plus its call in
`main`) and `codemap/gen/*` (regenerated).
