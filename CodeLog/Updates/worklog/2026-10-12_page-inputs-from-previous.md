# Page menu: Load all outputs from the previous page as Page Inputs

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** page-inputs-from-previous
- **Base:** e1ae0e28

## What changed

Every page's dropdown has a new entry, **Load all outputs from “<previous page>”  (N)**.
That dropdown is the top-left page switcher, and the same menu opens on a right-click on a
page tab. One click adds a Page Input on the shown page for each named Output of the previous
page that no Input there reads yet, bound to it, in the order the Outputs were added. The new
Inputs stack in a column at the left of the page (under its existing Inputs, else left of its
nodes), placed by each card's real height. They are selected and brought into view, and the
status bar names them.

The entry names the source page and the count. It greys out with the reason when there is
nothing to add: Image Input has no previous page, no earlier page publishes yet, or every
Output is already read. Run again after the earlier page gains an Output, it adds just that
one.

"Previous page" is `Workspace.previous_feeder`: the nearest page that may feed this one
(`feeder_pages`: nearest kind, latest page, Free last) AND publishes a named Output. That is
the same page a hand-placed Input's `default_source` binds to. The list is
`Workspace.unread_outputs`. On a linked page the action goes through `_topology_ok`, so it
asks how the edit applies, like every structural edit.

## Why

User request: wiring a later page meant placing and binding one Page Input per output by
hand. The standard four-page workflow publishes several Outputs per page (one per loaded
file, plus masks and tables), so that was many repeated steps per page.

Using `feeder_pages` rather than the literal previous tab is deliberate. The tab before an
Image Processing page can be a second Processing page, which a Page Input may not read (only
strictly earlier kinds). The previous tab in the strip can also have no Outputs. The rule
`default_source` already uses gives "previous" one meaning across the app. Already-read
Outputs are skipped, not duplicated, so the entry is safe to use again.

## Files

- `nodelab_v2/workspace.py`: `previous_feeder`, `unread_outputs`
- `nodelab_v2/window.py`: `_add_load_outputs_action` (in `fill_page_menu`), `load_previous_outputs`
- `nodegraph/selftest.py`: `test_page_inputs_from_previous` (hotspot: appended before `main`, called last)
- `scripts/_nodelab_v2_phase5_probe.py`: probe PI1 (appended before the final print)
- `MANUAL.md` §2 Pages; `codemap/concepts.md` CON-17 (one sentence; the gate stays CURRENT)
- `codemap/STATE.md`, `codemap/gen/*`: regenerated. The diff was read: 5 new symbols, `fill_page_menu`'s docstring, line shifts in the 3 touched files

## How to verify

In the app: File → New, load two files (Image Input publishes one Output per file), show
Image Refinement. Its first visit seeds one Input. The switcher's menu then offers *Load all
outputs from “Image Input”  (1)*, and one click adds the other Input under the first.
Headless: `test_page_inputs_from_previous`, probe PI1 (menu text and count, binding order, one
column without overlap, selection, the greyed states, and the linked-page ask: cancel adds
nothing, "keep on this page" adds there).

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (PI1 included; then the documented teardown crash at `os._exit`)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (115 ops; no node changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> not IN SYNC by design: stacked on `timeseries-per-frame-clock` (needs its LK1 probe fix) on `graph-node-sets`, neither merged into `origin/Blender`
