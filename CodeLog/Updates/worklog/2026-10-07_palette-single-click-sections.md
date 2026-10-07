# Palette single click sections

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** graph-node-sets
- **Base:** e1ae0e28

## What changed

In the Nodes palette, ONE left click on a section heading now opens it and another closes it — a stage band, the Pages and More nodes bands, and a role row alike. Qt's own double-click expand is switched off on the palette tree (`setExpandsOnDoubleClick(False)`), so a quick double-click on a heading toggles once rather than opening and shutting again; a right click does not fold; a click on a node row still only selects it and shows its overview, and a double-click on a node row still adds it to the canvas. The handler is `_PaletteTree._toggle_section` on `itemClicked`, acting only on rows that have children. The MANUAL's palette lines and the palette's own legend say so; the GUI probe drives it with real mouse events through the viewport.

## Why

The owner: "on the node menu, i want a single click to open up sections not double click". The rows draw no branch arrow (root decoration is off, for the full-width band look), so Qt's default double-click expand was the only way to open a section, and nothing on the row hinted at it — every band starts collapsed, so the first thing a user met was a list that seemed not to respond to a click. Toggling on `itemClicked` (emitted on release, left button only) rather than on press keeps a drag from a heading inert and means Qt's double-click bookkeeping — it resets the pressed index after a double-click — leaves exactly one toggle per double-click once its own expand is off.

## Files

- GUI: `nodelab_v2/palette.py` (`_PaletteTree`: `setExpandsOnDoubleClick(False)`, `itemClicked` → `_toggle_section`; module docstring and legend text)
- Tests: `scripts/_nodelab_v2_phase5_probe.py` (G2b: real clicks on a band and a role, a right click, a double-click on a band, a double-click on a node)
- Records: `MANUAL.md` (the palette bullet and the page-kind paragraph), `codemap/gen/*` + `codemap/STATE.md` (regenerated)

## How to verify

`python run.py` → Nodes palette: click a band heading (it opens), click it again (it closes); click a role heading inside an open band to fold its nodes; double-click a node row to add it. Headless: the phase-5 probe's G2b block.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (encoder); the groups scheduled after it were run separately and pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 115 ops (no node changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> not pushed; on `graph-node-sets`, the push is the owner's call

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
