# Mirror/merge: off by default, ticks are the selection, region boxes, new copies selected

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** node-mirror-merge
- **Base:** e1ae0e28

## What changed

- **Three modes, off by default.** The button beside the action pill steps **off → Mirror →
  Merge → off** (`multilink.OFF`, `next_mode`). While off, no card or region shows a
  checkbox, there's no border, and every wire is ordinary. Turning a mode off unticks
  everything. Mirror ↔ Merge keeps the ticks.
- **While a mode is on, the ticks ARE the selection.** `GraphScene` follows
  `selectionChanged` (`_ticks_follow_selection` → `multilink.set_ticks`): every selected
  card, plus every card of every selected region. Turning a mode on ticks what is already
  selected. A card's box adds it to the selection or takes it out without clearing the rest
  (`toggle_tick`); a card ticked only through its selected region unticks alone, and the
  region gives way to its other cards. A plain click on empty canvas deselects, and so
  unticks, everything. **Clear selected** also deselects, so nothing re-ticks; the mode
  stays on.
- **Regions have their own checkbox** at the right of the title bar
  (`FrameItem.mm_check_rect`/`mm_ticked`, `GraphScene.toggle_region_tick`). It ticks every
  card in the region (the region is selected) or unticks them all, and reads ticked while
  the region is selected or all its cards are ticked.
- **A mirror that made nodes selects them** (the placed node and every copy,
  `GraphScene.select_only`), and the cards they came from drop out of the selection. So
  switching to Merge next merges the new copies. Ticks that are exactly one mirror group
  keep that group's lanes (`lanes_for`), so later connections still line up lane by lane.
- The border shows whenever a mode is on, ticked or not. Mirror edits are shared whenever
  the mode is Mirror, whether or not anything is ticked.

## Why

User request: mirror/merge should be off by default (a third button state), and "activating
it will make your dragging and selecting nodes automatically check the boxes", with regions
having a checkbox that works the same way, and the new mirrored nodes becoming selected (the
previous ones unselected) so a switch to merge merges them.

The request treats "selected" and "checked" as one thing in these modes, so they are kept
identical (two-way) rather than letting selection only add ticks. One-way would pile up
ticks as the user clicks around, and would make "the previous ones become unselected"
meaningless. Panning (a plain drag on empty canvas) doesn't touch the selection, so it
doesn't untick. Checkboxes are hidden while off because a tick there would do nothing. A
region's box ticks its cards, which is how a region "functions the same way" as a card's box.

## Files

- `nodelab_v2/multilink.py`: `OFF`, `MODES` order, `set_mode` (off unticks), `next_mode`, `set_ticks`, `active` (= a mode on), `lanes_for` (keeps a ticked group's lanes; nothing while off), `mirror_siblings` (mirror mode only)
- `nodelab_v2/scene.py`: the selection↔ticks sync, `toggle_tick`, `toggle_region_tick`, `select_only`, the post-mirror selection; the mode button's three states and styling; Clear selected deselects
- `nodelab_v2/node_item.py`: the box and ring only while a mode is on; a box click goes through the scene
- `nodelab_v2/frame_item.py`: the region's box
- `nodegraph/selftest.py`: `test_multilink` updated for off-by-default, `set_ticks`, a ticked group keeping its lanes, and propagation by mode (hotspot: inside the test appended earlier today)
- `scripts/_nodelab_v2_phase5_probe.py`: MM1 rewritten for the new flow; the footprint section's "a collapsed card has no controls" is back to `[]` (the box is off by default)
- `MANUAL.md`: *Mirror and merge* rewritten
- `codemap/STATE.md`, `codemap/gen/*`: regenerated. The diff was read: 11 new symbols, 9 changed, one new import edge (`frame_item` → `multilink`)

## How to verify

Probe MM1 drives the whole flow on the canvas:
- off: no box, no border, selecting ticks nothing
- turning Mirror on ticks the current selection; more selecting adds ticks
- a card's box adds and removes; a region's box ticks and unticks its two cards
- white, then yellow borders, kept apart from the amber troubleshooting frame
- merge order and the single-input refusal
- merge → off unticks; off → Mirror re-ticks the selection
- a mirror selects the new Gammas, not the cards they came from
- an edit on one copy reaches all, and the mirror continues down two more steps
- switching to Merge merges the newest copies into one Timeseries Builder
- Clear selected keeps the mode, and off ends it

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the new MM1)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> not IN SYNC by design (stacked branches, none merged); 0 behind `origin/Blender`
