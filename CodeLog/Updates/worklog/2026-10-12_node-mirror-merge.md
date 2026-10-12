# Mirror and merge: a checkbox on every card, one wire for many nodes

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** node-mirror-merge
- **Base:** e1ae0e28

## What changed

Every card (not reroute or batch dots) has a **checkbox at the right of its header**, beside
the 2D/3D switch when it has one; the title is elided short of it. Ticking cards builds the
page's multi-link set. A wire drawn OUT of a ticked card, while at least one other is
ticked, goes through the canvas's mode:

- **Merge** wires every ticked card into the target. The dragged-from card goes into the
  primary (first) input it fits, and the others go in alphabetical order of title, one per
  following input. A multi-input socket takes them all, and growing slots fill in order. If
  they don't all fit, it is refused with nothing changed (a node with one input always
  refuses), and a node just made from the link-drag search for it is removed again.
- **Mirror** makes the drawn wire, then gives every other ticked card its own copy of the
  target: same op, params, modes, mute/collapse, the same wire from its card, the same
  offset, and the target's other incoming wires. The copies and the original form a mirror
  group. While the page is in mirror mode (mirror selected and something ticked), an edit
  to any member applies to all. This happens in `GraphDocument.touch`, which every editor
  calls, so the card and the inspector both propagate. A Page Output's name and a card's
  title stay each node's own.

Beside the action pill (top centre) are the **mode button** (⧉ Mirror / ⤞ Merge; click to
switch; filled in the mode colour while live) and **Clear selected (N)**, which unticks
everything. While anything is ticked the canvas is **outlined at its very edge**: white for
mirror, yellow for merge (new theme tokens `MIRROR`/`MERGE`). The outline is a square 3 px
band with a dark hairline inside. Ticked cards get a ring in the same colour. A pan now
repaints the whole viewport while the outline or the troubleshooting frame is up, so
neither smears.

State is per document and per session, in the new Qt-free `nodelab_v2/multilink.py`:
ticks, mode and groups, plus the merge plan, mirror copies and propagation. It isn't saved
to the file. A tick isn't a graph edit: it notifies its own listeners (the page's scenes),
never `doc._notify`.

## Why

User request. Wiring the same step after several nodes, or several nodes into one, meant one
drag per node, and keeping N copies of a step in step meant N edits.

Some choices the request left open:
- The modes act only on wires dragged out of a ticked card's output, so every other wire
  stays ordinary.
- "Primary input" means the target's first input that fits, as the request says, even if
  the drop landed on another socket.
- A refusal is all-or-nothing (planned first, rolled back if a connect fails mid-way), not a
  partial merge.
- Propagation is tied to the visible mode: border up and Mirror selected. Clearing the ticks
  is how a user makes the copies independent again.

The border sits at the viewport edge, square, with a hairline, while the troubleshooting
frame is rounded at a 12 px inset with its glow. That way yellow and amber can't merge
visually: the probe samples the pixels and checks for a plain-canvas gap between them, and
that the two hues differ by at least 12°. White alone would vanish on the light theme
(`#eef1f5`), hence the hairline.

## Files

- `nodelab_v2/multilink.py` (new): state, `merge_plan`/`merge`, `mirror`, `mirror_siblings`/`copy_values`
- `nodelab_v2/document.py`: `touch` → `_mirror_edit`
- `nodelab_v2/node_item.py`: the header checkbox (`Ctl` kind `mm`, `_mm_check_rect`, `_paint_mm`), the ring, the narrower title beside a switch
- `nodelab_v2/scene.py`: `GraphScene._multilink`, `link_new_node` (the link-search pick, now callable), `multilink_note`, the state listener; `GraphView` mode/clear buttons, `refresh_multilink`, the border, `setScene`/`scrollContentsBy` overrides
- `nodelab_v2/window.py`: `multilink_note` → status bar
- `nodelab_v2/theme.py`: `MIRROR`, `MERGE` tokens (both palettes)
- `nodegraph/selftest.py`: `test_multilink` (hotspot: appended before `main`, called last)
- `scripts/_nodelab_v2_phase5_probe.py`: probe MM1. The footprint section's "a collapsed card has no controls" now reads "only the mm checkbox"
- `MANUAL.md`: *Mirror and merge* after the action-pill section
- `codemap/STATE.md`, `codemap/gen/*`: regenerated. The diff was read: the new module and its 40 symbols, 8 import edges (`multilink` ← node_item/scene/selftest/probe, document lazily), line shifts in the touched files

## How to verify

`test_multilink` (Qt-free: ticks, merge into growing slots, a multi socket, two inputs, a
refused third, a single-input refusal, mirror copies, propagation on and off). Probe MM1
drives the real canvas: header clicks, the buttons beside the pill, border pixels (white,
yellow, and yellow and amber with a gap between), a merge drag into a Timeseries Builder
(Z first, then A, M), the single-input refusal, a refused merge into a NEW node leaving none
behind, a mirror from the link search with an edit on one copy reaching all, and *Clear
selected* ending it. In the app: tick three cards, Merge, drag from one into a Timeseries
Builder's input.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (MM1 included; then the documented teardown crash at `os._exit`)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (115 ops; no node changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> not IN SYNC by design: stacked on `page-inputs-from-previous` → `timeseries-per-frame-clock` → `graph-node-sets`, none merged into `origin/Blender`
