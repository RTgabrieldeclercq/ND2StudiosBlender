# Commit the working-tree leftovers: `.nd3` in two LabLink recipes, a granule graph, the bead brief

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** bead-finder
- **Base:** 6298377 (the Bead Finder recall commit, same branch)

## What changed

Four files that sat uncommitted in the working tree from before the Bead Finder session
are now in git, unchanged:

- `lablink_recipes/nd2studios/cell-segmentation/recipe.json` and
  `lablink_recipes/nd2studios/selftest-synthetic/recipe.json` — each recipe's `image` input
  now also matches `*.nd3` files, beside `*.nd2`, `*.tif`, `*.tiff`.
- `graphs/granule_detection.nd2graph.json` — a saved workspace (app 4.0.0, format 3.0), one
  free page holding a single `io.load` node: four channels (GFP, R-B, Nile Blue, TD),
  1.718 µm/px, 40 µm z step, pointing at a local file under
  `D:/GELS Experimental/20260715_115850_227/`. A starting point for a granule-detection
  graph; it opens only on a machine that has that file. New top-level `graphs/` folder.
- `node_idea.md` — the design brief the Bead Finder (`detect.beads`) was built from: slab
  projection with size `S` and overlap `G`, 2-D detection per slab, z localisation on the
  raw stack near the slab, a skewed Gaussian in z, size filtering. The bead worklogs quote it.

## Why

The request was "push this repo to git": the `bead-finder` commits were already on
`origin`, and these four were the only things still local. Why the recipes gained `*.nd3`
is **not recorded** — the edit predates this session and nothing in `nodegraph/` or
`nodelab_v2/` reads an `.nd3` file today, so the pattern only widens which files a LabLink
hub offers to the recipe; a file it matches but cannot load would fail at `io.load`. Whoever
made the edit should add the reason here. They are committed separately from the bead work
so the bead commits stay reviewable on their own, and so this one can be dropped or moved to
its own branch without touching them.

## Files

- `lablink_recipes/nd2studios/cell-segmentation/recipe.json`
- `lablink_recipes/nd2studios/selftest-synthetic/recipe.json`
- `graphs/granule_detection.nd2graph.json` (new)
- `node_idea.md` (new)
- `CodeLog/Updates/worklog/2026-10-07_leftover-recipes-graph-brief.md` — this entry

## How to verify

```
PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as s; s.test_lablink()"
```
`test_lablink` reads both recipe folders and validates the shipped recipes against the live
catalog; it passes with the `.nd3` patterns in place.

## Gates

- [x] `test_lablink` (the only selftest that reads these files) -> all `[ok]`, run 2026-10-07
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (no change under `nodegraph/` or `nodelab_v2/`)
- [ ] full `nodegraph.selftest` and the GUI probe — not re-run for a data/docs-only commit;
      both were run on 6298377 an hour earlier (green apart from the pre-existing
      `test_write_movie` failure)
- [x] `python scripts/_sync_check.py` -> origin/Blender unchanged (behind 0)
