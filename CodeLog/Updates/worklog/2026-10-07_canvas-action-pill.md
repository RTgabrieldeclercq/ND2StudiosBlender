# Canvas action pill

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** canvas-action-pill (stacked on region-tabs)
- **Base:** 48326c3

## What changed

Every canvas now has an **action pill** at its top centre. Its text names the context —
*Graph*, *2 nodes*, *Region “Cells”*, *2 regions*, *Region tab · “Cells”* — and it turns
cyan while the region box is armed. Clicking it opens a menu built from the selection at
that moment. With loose nodes selected it offers *Group into a region*. With nodes already
in a region it offers *Remove from region*, and *Add to region* when loose nodes are also
selected. One selected region gets *Duplicate as a linked tab*, *Rename* and *Ungroup*
(the frame goes, the cards stay). Several regions get *Group N regions into one* and
*Ungroup N regions*. A node-group card gets *Ungroup node group*, and a region tab gets *Go
to the region on the master*. *Draw a region* (or *Put the region box away*), *Reorganize
graph*, *Undo reorganize* and *Fit graph* are always offered. An action that cannot run is
listed greyed out with the reason as its tooltip; a linked page's region edits are its
master's. **Reorganize graph** (also in the Graph menu) is a new layered left-to-right
layout that keeps every region in one block with room for its title and port labels, so no
card lands inside a region it is not in. On a region tab only the tab's own Page Inputs and
Outputs move. *Undo reorganize* restores every card until one is moved by hand.

## Why

Asked for: a pill in the middle of the node graph whose options depend on what is going on,
for example draw a region, group regions, ungroup a region and reorganize the graph. The
offer is computed by a Qt-free module (`canvas_actions`) from the document and the
selection, rather than wired into the widget, so the selftest can check every context
headless and the Graph menu and frame menu stay one code path with it. "Group regions" is
read two ways and both are offered: selected nodes into a region, and several regions into
one. Reorganize is a dedicated Qt-free layout (`graph_layout`) rather than a generic
force-directed one because the pipeline reads left to right and the regions must stay
whole, which a compound layered layout gives directly. A region tab freezes and anchors its
region, because the cards inside belong to the master and moving them would rearrange the
master page. The new frame edits (`merge_frames`, `remove_from_frames`) and the
`set_frame_members` added with region tabs are now refused on a linked page, which the
region-tabs change had missed.

## Files

- `nodelab_v2/canvas_actions.py` — new: the pill's text and its context-dependent actions
- `nodelab_v2/graph_layout.py` — new: the left-to-right layout with regions as blocks
- `nodelab_v2/document.py` — `merge_frames`, `remove_from_frames`
- `nodelab_v2/linked_document.py` — refuses the frame edits
- `nodelab_v2/scene.py` — the pill on `GraphView` (placement, style, armed state)
- `nodelab_v2/canvas.py` — the pill's menu, filled as it opens
- `nodelab_v2/window.py` — `fill_action_menu`, `run_canvas_action`, the region edits, reorganize + undo, Graph → Reorganize graph
- `nodegraph/selftest.py` — `test_canvas_pill_and_reorganize` (hotspot file: appended at the end)
- `scripts/_nodelab_v2_phase5_probe.py` — probe RG2
- `MANUAL.md` — §12 *The action pill and Reorganize graph*
- `codemap/STATE.md`, `codemap/gen/*` — regenerated

## How to verify

Run the app and load the example graph. The pill at the top centre reads *Graph*. Select
two cards: it reads *2 nodes*; click it → *Group 2 nodes into a region*. Click the pill again
with the region selected: *Duplicate region as a linked tab*, *Rename region…*, *Ungroup
region*. Drag a few cards around, click the pill → *Reorganize graph*: the cards line up left
to right with the region whole; the pill now offers *Undo reorganize*.
Headless: `python -B -c "from nodegraph.selftest import test_canvas_pill_and_reorganize as t; t()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (with `test_write_movie` stubbed: it fails identically on the untouched base, see region-tabs entry)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> no incoming commits; stacked on region-tabs, which is not merged yet
