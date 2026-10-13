# Merge hyper's ui-lag-troubleshooting with Gabriel's stack

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** sync-ui-lag-merge
- **Base:** e1ae0e28

## What changed

One tree now holds both lines of work that grew from `8fc61af` (the canvas action pill),
and `Blender` is fast-forwarded to it:
- **Gabriel's stack:** the timeseries per-frame clock, the nd2 `_Seq` reader, the bundle
  stage log, page inputs from the previous page, mirror/merge, frame timing, the Node
  queue, and the troubleshooting cut after a Timeseries Builder.
- **hyper's `ui-lag-troubleshooting`:** the governor and display-worker caps, field
  mechanics, staged DIC/DVC, analysis lanes and store, Quick analysis, overlays, and the
  spreadsheet.

Conflicts, all resolved by keeping both sides:
- **Bundle metadata.** Ours adds the per-position file clock (`SOURCE_CLOCK_KEY`) and
  concatenates each file's stage log and clock (`bundle_display`). Theirs adds the
  per-position absolute path (`SOURCE_PATH_KEY`, via `bundle_envelope(..., paths)`). Both
  keys are in `PER_POSITION_KEYS` and `__all__`. The runner's bundle resolve calls
  `bundle_display` and `bundle_envelope(..., paths)`. The window's bundle seed goes through
  `bundle_envelope` (ours) with the paths, so it stamps `source_path` exactly as the pull
  does (theirs).
- **Tests and probe.** Both sides appended tests at the end of the selftest; both kept,
  and every `test_*` is called once in `main`. In the probe, both sides' blocks are kept,
  and hyper's analysis-lane teardown stays last, before the pass line.
- **Imports.** The `scene.py` imports are the union of both sides.
- **Generated files.** `codemap/gen/*` and `STATE.md` were regenerated, not merged. The
  auto-merged `catalog_baseline.json` already checks IDENTICAL (137 ops), and
  `node_synopsis.json` checks CURRENT.

One behaviour change made for the merge, in `util.crop`'s frames mode (both the compute
and the meta transform, through the new `metadata.scope_stamp_subset`). The `scope_m` /
`scope_t` / `scope_z` stamps that `_pin_frames` puts on a scoped seed now follow a frames
cut:
- a stamp that still describes the axis is remapped to the survivors' source frames;
- a stale one (a Timeseries Builder upstream rebuilt the axis) is replaced by the kept
  indices, and `scope_of` takes that axis's extent.

## Why

Gabriel asked to "push and merge/sync", choosing to merge hyper's latest branch in as
well rather than fast-forward `Blender` to his stack alone. Both lines built on the same
unmerged base, so landing either alone would leave the other to integrate later.

The `util.crop` change exists because the two features meet there. Below a Timeseries
Builder, troubleshooting now cuts after the builder (a frames crop in the run graph). The
seed's `scope_t` then names one source file's frames while the payload holds picked
rebuilt frames. `analysis_units.scoped_frame` saw a length mismatch and fell back to
reading the reference frame as a payload index, so a DIC/PIV/DVC with a fixed reference
frame below a builder would silently correlate against the wrong frame.

**For hyper to review:** `_pin_frames`' comment says no node that narrows M or T should
re-index these stamps. Crop's frames mode now does: the stamps' consumer
(`scoped_frame`) reads them as "the source index of each payload frame", and only a
re-indexed stamp keeps that true after a cut. If you intended otherwise, the alternative
is to restamp only the troubleshooting cut that `nodelab_v2.solo_scope` injects.

## Files

- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/symbols.jsonl`
- `nodegraph/catalog/util/crop.py`
- `nodegraph/metadata.py`
- `nodegraph/selftest.py`
- `nodelab_v2/runner.py`
- `nodelab_v2/scene.py`
- `nodelab_v2/window.py`
- `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`
- `codemap/concepts.md`
- `codemap/curated.lock.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/invariants.md`
- `codemap/node_demos.json`
- `codemap/node_roles.json`
- `codemap/node_synopsis.json`
- `graphs/bead_field_crc_reference.nd2graph.json`
- `graphs/dvc_crc_early_run.nd2graph.json`
- `graphs/dvc_crc_early_run_staged.nd2graph.json`
- `graphs/field_mechanics_playground.nd2graph.json`
- `nodegraph/analysis_store.py`
- `nodegraph/bundles.py`
- `nodegraph/catalog/__init__.py`
- `nodegraph/catalog/_shared/analysis_mesh.py`
- `nodegraph/catalog/_shared/analysis_units.py`
- `nodegraph/catalog/_shared/field_nodes.py`
- `nodegraph/catalog/_shared/figure.py`
- `nodegraph/catalog/_shared/label_paint.py`
- `nodegraph/catalog/_shared/labels.py`
- `nodegraph/catalog/analysis/dic_correlate.py`
- `nodegraph/catalog/analysis/dic_stages.py`
- `nodegraph/catalog/analysis/dvc_field.py`
- `nodegraph/catalog/analysis/dvc_stages.py`
- `nodegraph/catalog/analysis/piv.py`
- `nodegraph/catalog/analysis/piv_stages.py`
- `nodegraph/catalog/detect/beads.py`
- `nodegraph/catalog/field/__init__.py`
- `nodegraph/catalog/field/gradient.py`
- `nodegraph/catalog/field/rotation.py`
- `nodegraph/catalog/field/strain.py`
- `nodegraph/catalog/field/stress.py`
- `nodegraph/catalog/field/volume_change.py`
- `nodegraph/catalog/io/load_analysis.py`
- `nodegraph/catalog/io/write_figure.py`
- `nodegraph/catalog/io/write_movie.py`
- `nodegraph/catalog/plot/field.py`
- `nodegraph/catalog/track/objects.py`
- `nodegraph/catalog/util/merge.py`
- `nodegraph/catalog/util/stitch.py`
- `nodegraph/engine.py`
- `nodegraph/fieldmap.py`
- `nodegraph/governor.py`
- `nodegraph/kernels/README.md`
- `nodegraph/kernels/aldvc_field.md`
- `nodegraph/kernels/aldvc_field.py`
- `nodegraph/kernels/bead_slabs.md`
- `nodegraph/kernels/bead_slabs.py`
- `nodegraph/kernels/dic_correlate.md`
- `nodegraph/kernels/dic_correlate.py`
- `nodegraph/kernels/field_mechanics.md`
- `nodegraph/kernels/field_mechanics.py`
- `nodegraph/kernels/piv_field.md`
- `nodegraph/kernels/piv_field.py`
- `nodegraph/parallel.py`
- `nodegraph/registry.py`
- `nodegraph/units.py`
- `nodelab_v2/demo_recipes.py`
- `nodelab_v2/document.py`
- `nodelab_v2/inspector.py`
- `nodelab_v2/lablink/worker.py`
- `nodelab_v2/overlay_dialog.py`
- `nodelab_v2/overlays.py`
- `nodelab_v2/palette.py`
- `nodelab_v2/quick.py`
- `nodelab_v2/spreadsheet.py`
- `nodelab_v2/theme.py`
- `nodelab_v2/viewer.py`
- `scripts/catalog_baseline.json`
- `scripts/make_field_playground.py`

## How to verify

The full selftest and GUI probe on the merged tree cover:
- both sides' tests and probe blocks;
- `test_solo_scope_plan` part 5: rebuilt frames {1, 3} below a builder carry `scope_t`
  (1, 3), and reference frame 3 maps to payload frame 1;
- a current stamp is remapped and an unscoped one is untouched.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (137 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> 0 behind origin/Blender; `Blender` fast-forwarded to this merge

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
