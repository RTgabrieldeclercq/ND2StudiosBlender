# Multi-file cards keep every file's stage log, point names and clock

- **Date:** 2026-10-09
- **Author:** Gabriel Declercq
- **Branch:** timeseries-per-frame-clock
- **Base:** e1ae0e28

## What changed

A multi-file card's payload display dict is now built by `runner.bundle_display`. Everything
the files share (channel names, colours, emission, bit depth, the shared per-T clock) comes
from file 0, as before. Every per-position list in it (`stage_xy_um`, `stage_z_um`,
`position_name`) is concatenated across the files, all-or-nothing, and the per-position
file clock (`source_frame_time_jd`) is folded into the same function. The card's edit-time
seed (`window._add_bundle_node`) is now built by `bundle_envelope`, the same function the
pull uses, so its per-M calibration (`origin_um`) spans every file from the moment the
card is dropped, not only after the first pull.

## Why

Found while building the per-frame clock. `_resolve_bundle` took member 0's whole display
dict, which is where the ND2 reader puts the stage log and the point names. A card of K
files therefore carried file 0's n-long lists on a K·n-long M axis, every positional
reader rejected them as the wrong length, and a multi-file card had no stage log at all.
Stitch by stage refused, the Viewer's hover readout had no stage position, and a
Timeseries Builder after the card had no log to keep. On the user's own files (Seq0000 +
Seq0003 as one card) the card now carries 80 stage entries where it carried 40, and the
log survives a Timeseries Builder onto T (0.3 µm spread against a 176 µm tolerance), so
Stitch by stage works downstream. The edit-time seed had the same flaw for `origin_um` and
self-corrected only on the first pull. A `view.canvas` or anything placing by
`origin_um` at edit time saw a list of the wrong length until then.

## Files

- `nodelab_v2/runner.py`: `bundle_display`; `_resolve_bundle` uses it
- `nodelab_v2/window.py`: `_add_bundle_node` seeds through `bundle_envelope`
- `nodegraph/selftest.py`: `test_timeseries_frame_interleave` §8 (hotspot)
- `MANUAL.md`: "Each file keeps its own positions and clock" (Grouping files into one bundle)
- `codemap/STATE.md`, `codemap/gen/*`: regenerated for the whole branch tip. The diff was read: 3 new symbols (`chain_uniform_step_s`, `sequence_indexer`, `bundle_display`), 2 new lazy import edges (`window → catalog.util.timeseries` for the slot ceiling; `selftest → nd2_meta`), line shifts in the 5 touched files, and every node's closure fingerprint (`fp`) because `nodegraph.metadata` is in every closure

## How to verify

`test_timeseries_frame_interleave` §8. Real data: Seq0000 + Seq0003 as one card through
`bundle_display`/`bundle_envelope` gave `stage_xy_um`/`stage_z_um`/`position_name`/`origin_um`
all 80 long and an 80 × 8 per-position clock. A Timeseries Builder onto T then interleaved
the two files by clock (16 frames, pixels byte-equal to the named frames) and kept the stage
log.

## Gates

Run once on the branch tip (this commit, which contains the two before it):

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (161 `[ok]`, LK1 and the new SQ2 included; the access violation at `os._exit(0)` AFTER the pass line is the probe's documented teardown crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (115 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`; diff read, see Files)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> not IN SYNC by design: the branch sits on `graph-node-sets`, which is not merged into `origin/Blender`
