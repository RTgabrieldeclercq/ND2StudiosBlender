# V4.00 step 11a (part A): standard workspace, auto page flow

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** origin/Blender
- **Commit:** ee4621d (entry written after the fact on 2026-10-07, from the commit's diff;
  the sync check flagged the commit as having no worklog file)

## What changed

A new document opens as the **standard workspace** — the typed pages (input, refine,
process, …) laid out in pipeline order with the page flow wired automatically — instead of
one free page. The palette offers each page's node kinds, the readiness block learned the
page-flow problems and their fixes, and the inspector, node cards, scene and window gained
the page-aware behaviour that goes with it. The selftest grew the headless coverage for the
workspace's auto page flow; the GUI probe drives the new flow.

## Why

Step 11 of V4.00 is the "standard workflow": a user should not have to invent the page
structure every time — the pipeline stages are known, so the workspace should start as
them, with the data already flowing from one page to the next. Marked WIP because parts B
and C (chrome, layout, viewer defaults) followed in separate commits.

## Files

- MANUAL.md, codemap/STATE.md, codemap/concepts.md, codemap/curated.lock.json, codemap/gen/*
- nodegraph/selftest.py
- nodelab_v2/document.py, inspector.py, node_item.py, palette.py, readiness.py, scene.py,
  window.py, workspace.py
- scripts/_nodelab_v2_phase5_probe.py

## How to verify

Open NodeLab with no file: the Pages outline shows the standard pages in order, each Page
Input already sourced from the page before it. `PYTHONUTF8=1 python -B -m nodegraph.selftest`
and the phase-5 probe exercise the flow headless.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (at the time of the commit)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (at the time of the commit)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC — the branch was ahead of origin/Blender and not yet pushed
