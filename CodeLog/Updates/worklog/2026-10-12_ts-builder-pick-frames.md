# Troubleshooting picks frames of a Timeseries Builder's rebuilt T axis

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** ts-builder-pick-frames
- **Base:** e1ae0e28

## What changed

In troubleshooting mode (F9), viewing a Timeseries Builder or anything below one now
works like this:
- The Viewer's T strip spans the whole rebuilt series (every file's frames, in the built
  order).
- Ctrl+clicking frames on it computes exactly those rebuilt frames, from whichever file
  they come from.
- With nothing picked, the cursor's own rebuilt frame runs.

The new module `nodelab_v2/solo_scope.py` decides where the picks are applied:
- `scope_plan` cuts after the last node that re-addresses an axis. The sources behind it
  stay whole on that axis.
- `scoped_graph` adds a frames-mode `util.crop` after that node in the run graph only. It
  is named `<node>#solo`, so its progress shows on the builder's card. The canvas and the
  saved file never see it.
- An axis is left whole everywhere (`unscoped`, which was the old behaviour) only where a
  node reaches the viewed node both through a re-addressing node and around it.

The runner applies this plan:
- `_pin_for` keeps the picks in the viewed node's own indices.
- Each source takes its own pin (`source_pin`).
- The cached engine is keyed by the scoped graph as well as the revision.
- `readdressed_axes` delegates to the plan.

On the document side, `source_scope_totals` spans an axis cut after a re-addressing node
using that node's output extent.

Two fixes to the re-addressing table:
- The builder re-addresses **M** as well as its chain axis.
- The table reads the **resolved** mode state, so a builder at its default axis counts.

## Why

User report: "In Troubleshooting mode, the Timeseries Builder node should allow you to
select the frames you want from the rebuilt T axis, but right now it automatically cuts it
once entering the mode." Three causes:
1. **Default-axis builders were invisible.** The guard read `chain_axis` from the
   instance's explicit modes, so a builder placed by hand (onto T by default) was missed.
   The sources were cut to the cursor's frame, the builder built one frame per file, and
   the strip shrank to one file's frame count.
2. **The strips spanned the sources.** Under the scope, the strips took their length from
   the source roots: one file's frames, not the rebuilt series.
3. **Picks were dropped.** Even when the builder was detected (explicit "onto T"), T was
   left whole and the picks were thrown away, so there was nothing to pick from. M was
   still cut at the source, and a bundle card's files are positions, so that kept one file
   and the builder passed it through.

Cutting after the re-addressing node keeps the picks in the viewed node's own indices.
Inverting each node's index map back onto source frames would also work, but it means one
inverse per op (builder modes, clock interleaving, Time Shift, Merge) that can drift from
the node. Crop's frames mode already is the troubleshooting scope's own view
(`FrameSubsetProvider`, with the per-T/per-M metadata carried), and the builder is a lazy
index remap, so a cut after it reads only the picked frames' source frames.

## How to verify

1. Load two sequence files as separate cards into a Timeseries Builder.
2. Double-click the builder, then press F9. The T strip spans every built frame.
3. Ctrl+click two boxes; the Viewer shows exactly those rebuilt frames.

Automated:
- probe block `TSB1`, which fails on the pre-fix code at the pick check;
- selftest `test_solo_scope_plan` (plan, scoped graph, engine end-to-end on a bundle and on
  two cards of 3 and 2 frames, document strip extents);
- `test_frame_timing`'s assertions updated (builder → `{t, m}`, the default-axis builder
  included).

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

Hotspot files touched: `nodegraph/selftest.py` (one test appended plus its call in
`main`; `test_frame_timing`'s builder assertions updated) and `codemap/gen/*`
(regenerated).
