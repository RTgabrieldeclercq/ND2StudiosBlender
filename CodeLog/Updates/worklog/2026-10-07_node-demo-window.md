# Node demo window — "What does this node do?"

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** node-demo-window
- **Base:** e1ae0e28 (v4-step11-standard-workflow head; PR #1 from the other machine is
  open and does not touch these files beyond the usual selftest/window hotspots)

## What changed

Every node can now show itself. A **?** beside the node's title at the top of the Properties
panel, and a **What does this node do?** button under the palette's Overview, open a
non-modal window that runs that node *type* on a small deterministic synthetic image (a
*phantom*: nuclei on an uneven background, a Z-stack, a drifting or moving time series, two
channels, a speckle pair, a 2×2 stage mosaic) with the result beside the input and one
control per parameter — a slider + spin box for every number (span from the unit, the pick
gesture and the default; a level slider spans the phantom's own histogram; a derived
parameter starts at its derived value), dropdowns for choices and modes (modes hide and show
parameters exactly as the inspector does), ticks for booleans, grey chips for values the demo
fixes, nothing for presentation-only sockets. Every change recomputes on a worker thread
(latest wins, 80 ms debounce, the After view dims and is never blanked); the upstream chain
(a threshold → label before Measure, a tracker before Track Field, a spot detector before
Voronoi) is memoised so only the node moves. What *After* shows follows the node's kind:
image (+ Wipe / Checkerboard / Difference compare), mask or scalar heat map, label outlines,
points, tracks (with a Frame t slider), vectors, mesh vertices, a table, or a figure — all
painted by the Viewer's own OverlayRenderer on the Viewer's own image surface. Slow nodes
(StarDist/CellSAM methods, ZS-DeconvNet, DIC/DVC, mesh rasterization) get a Run button, and a
node that measures slow twice switches itself. 25 nodes that do not transform pixels
(loaders, writers, docks, page boundaries, zones, groups, reroute, sweep/simulation controls,
the Viewer tap, the batch organisers) open as a **guide**: curated key features / how to use,
sockets with the gesture each offers, modes, how it works. Draw Regions is NOT a guide: its
demo rasterizes a built-in shape list (rect with a cut-out, circle, polygon) so Apply's result
is visible, and its features walk the Draw → Tool → Operation → Apply flow. The palette
Overview is now built by the same HTML builder as the window's guide (one source; it gained
the key features and the per-socket gestures). The window is a sandbox: it never reads or
writes the canvas node; one window per op type, closed windows leave the cache.

New modules: `nodegraph/phantom.py` (Qt-free phantoms), `nodelab_v2/demo_recipes.py`
(Qt-free: recipes, control classification, slider-range heuristic, guide HTML, `DemoSession`
running seed → prelude → node through `headless_engine` on its own memo + tile cache),
`nodelab_v2/demo_window.py` (Qt), `nodelab_v2/value_steps.py` (the scrub-step helpers moved
out of `node_item.py`, which re-exports them, so the Qt-free modules can size a slider the
way the inspector sizes a spin box), `codemap/node_demos.json` (per-op curation: phantom,
prelude, fixed values, slider spans, slow marks, guide features — 108 entries).
Tests: `selftest.test_phantoms`, `selftest.test_node_demos` (validates the JSON against the
registry, then runs EVERY op's demo once and asserts the evidence its kind promises — 78 live,
23 guides, 5 slow-with-reason, 2 need `openpiv`), a phase-5 probe section. Docs: MANUAL §8e,
INV-17, WF-09, a CLAUDE.md lookup row, a build-node-v2 checklist item. Also two after-the-fact
worklog entries for the step-11a commits the sync check flagged.

## Why

A node today explains itself only in prose (the palette Overview, hover text); nothing shows
what it does to pixels and nothing lets a user move a parameter and watch. The request: a
popout per node, opened from the node's header box, with synthetic before/after and live
sliders for every editable variable, and for nodes like Draw Regions a guide to their key
features. Design choices and why: phantoms are Qt-free and in `nodegraph/` because the
selftest must run every demo headless (SyntheticProvider is the precedent); the recipe module
is in `nodelab_v2/` because it needs `headless_engine` and the picker's gesture text (the
selftest reaches it through the sanctioned seam); curation is a JSON validated by the gate
rather than Python so a renamed socket fails loudly; slider ranges are a heuristic because
the registry carries no min/max on purpose; the demo seeds an `io.load` exactly as the runner
does and sends only touched params so the engine keeps deriving the rest (INV-03 honoured, no
`define_node`); one shared `TileCache` + `Memo` per session because the engine's cache note
requires it of anyone rebuilding engines; the overlays go through the real renderer so the
demo cannot disagree with the Viewer. The window writes nothing back to the node by the
user's decision (sandbox only).

## Files

- `nodegraph/phantom.py` (new), `nodelab_v2/demo_recipes.py` (new),
  `nodelab_v2/demo_window.py` (new), `nodelab_v2/value_steps.py` (new),
  `codemap/node_demos.json` (new)
- `nodelab_v2/node_item.py` (value-step helpers moved out, re-exported),
  `nodelab_v2/inspector.py` (`demo_requested`, `_demo_button`), `nodelab_v2/palette.py`
  (`demo_requested`, the button, Overview via the shared builder), `nodelab_v2/window.py`
  (`open_node_demo`, cache, shutdown) — **hotspots:** `window.py`, `inspector.py`
- `nodegraph/selftest.py` (**hotspot**: two tests appended before `main`, two calls at the
  end of `main`), `scripts/_nodelab_v2_phase5_probe.py` (one section before the final print)
- `MANUAL.md` (§8e + contents), `codemap/invariants.md` (INV-17), `codemap/workflows.md`
  (WF-09), `CLAUDE.md` (one lookup row), `.claude/skills/build-node-v2/SKILL.md` (checklist)
- `codemap/STATE.md`, `codemap/gen/*` (**hotspot**, regenerated, never hand-merged)
- `CodeLog/Updates/worklog/2026-10-06_v4-step11a-standard-workspace.md`,
  `CodeLog/Updates/worklog/2026-10-06_v4-step11a-dock-chrome.md` (after-the-fact entries)

## How to verify

`python run.py`, select any node (e.g. Gaussian Blur) → press **?** beside its title in the
Properties panel → drag *Sigma*: the After view recomputes; switch Compare to *Wipe*. In the
palette click *Draw Regions* → **What does this node do?** → three numbered regions and the
guide. `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_phantoms();
S.test_node_demos()"` runs every demo headless in ~20 s.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every test passes EXCEPT
      `test_write_movie`, which fails identically on the base commit e1ae0e28 (frame means
      `[53.14, 53.14, 52.40, 83.3, …]` not monotone — the encoder on this machine, or the
      step-12 commit; not touched by this change). Verified by running the suite with that
      one test skipped: ALL NODEGRAPH SELF-TESTS PASSED, including the two new tests.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (154 checks)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (108 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (`write --verified 2026-10-07`)
- [x] `python scripts/_sync_check.py` -> branch `node-demo-window` off `v4-step11-standard-workflow`
      (24 ahead of origin/Blender, 0 behind, so no rebase was needed); PR #1 open, no overlap
