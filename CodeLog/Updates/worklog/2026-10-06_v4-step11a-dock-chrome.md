# V4.00 step 11a (part C): dock chrome, viewer visible by default, layout v2

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** origin/Blender
- **Commit:** aa2ef1c (entry written after the fact on 2026-10-07, from the commit's message
  and diff; the sync check flagged the commit as having no worklog file)

## What changed

The dock shell's chrome is painted by the theme everywhere a panel can be — docked, floated,
and in the frame gutter between them — and the Viewer panel is visible from the first launch
rather than appearing on the first pull. The saved layout format moved to v2 so an arrangement
saved by the old chrome does not restore with the old gaps. The theme gained the tokens the
chrome needed; console, palette, spreadsheet and viewer got the one-line fixes that let them
take the themed fill; the GUI probe asserts the new defaults.

## Why

From the commit: a floated panel showed Qt's light palette through the dock's frame gutter and
the panel bodies' transparent margins; the Viewer was hidden until the first pull, so a new
user saw an empty window and did not know where results would appear.

## Files

- codemap/STATE.md, codemap/gen/*
- nodegraph/selftest.py
- nodelab_v2/console.py, layout_store.py, palette.py, shell.py, spreadsheet.py, theme.py,
  viewer.py, window.py
- scripts/_nodelab_v2_phase5_probe.py, scripts/_nodelab_v2_shot.py

## How to verify

Launch NodeLab: the Viewer dock is visible before anything is pulled. Float any panel (⇱):
its frame and body are the theme's dark fill with no light gutter.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (at the time of the commit)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (at the time of the commit)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC — the branch was ahead of origin/Blender and not yet pushed
