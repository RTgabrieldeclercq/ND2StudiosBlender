# V4.00 step 11a — the standard workflow: four pages, loads onto Image Input, bound Page Inputs; dock chrome

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs)

## What changed

The V4.00 pieces now add up to one workflow a user can follow without hunting:

- **The standard workspace.** A fresh window and File → New hold the four standard pages —
  Image Input, Image Refinement, Image Processing, Analysis — Image Input active
  (`Workspace.standard`, `standard_kinds()` from the roles file; `reset()` returns to them
  keeping the active page's document as Image Input's). The Free kind stays for pre-V4 files
  and *New page ▸ Free*.
- **Loading an image goes to the Image Input page**, whatever page is active: Ctrl+L, a file
  sequence, a desktop drop (a drop ON a Batch point stays with the point) and a LabLink result
  all land there, the canvas switches to it, the card is published as a **Page Output named
  after the file** (`blobs`, `blobs2`, …) and pulled so the Viewer shows it. A TIFF card starts
  on the `ingest` access mode — `direct` is ND2-only, so a loaded TIFF could not even be
  previewed before. A Free-only (legacy) workspace keeps loading onto the active page.
- **Pages connect themselves.** One choke point, `GraphDocument.node_defaults` (installed by
  `Workspace._attach`, merged UNDER explicit params in `add_node`), names a hand-placed Page
  Output (`out`, `out2`, …) and binds a hand-placed Page Input to the nearest earlier page's
  newest named Output (`Workspace.default_source`, `feeder_pages`). A new downstream page
  (`add_page(seed_input=True)`) and an empty one shown for the first time (`seed_input` from
  `_activate_canvas`) start with that bound Page Input; the welcome card stays while a page
  holds only its seeded Input. The load, duplicate and group paths keep their own values.
- **One-click fixes.** `readiness.Suggestion` gained an `action` (`add` / `append` /
  `set_param`) and `Problem` a `severity`: an unbound Page Input offers *Bind to <page · name>*
  (the default first), an unnamed or duplicate Output *Name it “…”*, and a new `unpublished`
  HINT (never blocking) offers *+ Page Output* after a page's terminal node or a loader nothing
  publishes; the inspector applies them, with a hint line and *Go to <page>* under the Source
  menu. The card reads `Input · Image Input · raw`, `Input · (unbound)`, `Output · (unnamed)`
  in the error colour. The palette leads with a **Pages** band on typed pages; a wire dragged
  into empty canvas offers Page Output first.
- **The Example graph** (welcome card) builds one analysis across the four pages
  (`workspace.build_example`); `build_demo` stays as the probe's flat fixture.
- **Dock chrome and layout** (part C, built in a worktree and cherry-picked): a floated panel
  no longer shows Qt's light palette — `PanelDock.paintEvent` fills the dock and draws a 1 px
  border when floating, the `QDockWidget` rule carries a background, the panel bodies paint
  (`WA_StyledBackground`), and the application palette follows the theme (`theme.palette`).
  The ✕ is disabled when a close would be vetoed; the inactive title edge is a real border; the
  Movie Editor's scroll wrapper and the title glyph fonts are styled. The **Viewer is visible
  from launch** above the canvas (`VIEWER_SHARE` 0.45, `_apply_default_sizes` on first show
  and on reset), the palette 330 and Properties 376 wide; LabLink sits in a scroll area so its
  Send tab's natural width no longer holds the right column open (a blank strip beside
  Properties, pre-existing). The layout file is format `nd2studios.layout/2`: an older build's
  file is set aside once, and the status bar says so.
- **Tests**: `test_workspace_standard`, `test_page_defaults_hook`, `test_default_source_rule`,
  `test_readiness_fixes`, `test_example_workspace`, a v1 layout fixture in `test_layout_store`;
  probe SW1 / LD1 / LD2 / SW3 / RF1 / EX1, SH1b / CH1, and the WS1 / PG1 / PG7 / PG2 / PG8 / RD1
  / SH2 / SH3 sections updated for the standard pages. MANUAL §2 (window, pages), §2b, §18;
  codemap CON-17 (blessed).

## Why

The user tested V4.00 steps 0–10 and found the pieces did not add up: the app opened on one
Free page named "Graph", every loader put its card on whatever page was active, nothing ever
created or bound a Page Output / Page Input (the Source menu was empty until the user had
placed and named everything by hand, with the explanation in a tooltip), and a popped-out
panel had a white frame. The decisions (2026-10-06): keep explicit boundary nodes — several
named outputs per page and the `condition` column depend on them — but have the app create and
bind them; route every load to Image Input and switch there; show the Viewer from the start.
A single document hook rather than per-call-site patches, because eleven places create nodes
and the load/duplicate paths must keep their values. The dock frame was Qt's: with a custom
title bar `QDockWidget::paintEvent` paints nothing, so a floating dock showed the default
palette through its frame gutter, and the panel bodies (QWidget subclasses) never painted their
stylesheet background.

## Files

- `nodelab_v2/workspace.py` — `standard`, `standard_kinds`, `reset`, `node_defaults`,
  `unique_output_name`, `sanitize_output_name`, `default_source`, `feeder_pages`,
  `seed_input`, `add_page(seed_input)`, `build_example`
- `nodelab_v2/document.py` — `node_defaults` / `page_feeders` hooks, the merge in `add_node`
- `nodelab_v2/window.py` — `_input_page`, `_source_load_allowed`, `_begin_source_load`,
  `_publish_source`, `_published_note`, `_needs_source_page`, loader routing, `new_page`,
  `build_example_workspace`, `_on_append_requested`, seeding on activation, welcome sync;
  part C: `_window_qss`, `_panel_specs`, `_column_height`, `_apply_default_sizes`,
  `showEvent`, `reset_layout`, `set_theme`, `set_maximized`, `_allow_panel_close` /
  `_prune_canvases`
- `nodelab_v2/readiness.py`, `nodelab_v2/inspector.py`, `nodelab_v2/node_item.py`,
  `nodelab_v2/palette.py`, `nodelab_v2/scene.py`
- `nodelab_v2/shell.py`, `nodelab_v2/theme.py`, `nodelab_v2/layout_store.py`,
  `nodelab_v2/viewer.py`, `nodelab_v2/console.py`, `nodelab_v2/spreadsheet.py`
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`,
  `scripts/_nodelab_v2_shot.py`
- `MANUAL.md`, `codemap/concepts.md` (CON-17), `codemap/gen/*`, `codemap/STATE.md`,
  `codemap/curated.lock.json` — regenerated / blessed

## How to verify

`python run.py`: the window opens on **Image Input** with the Viewer above the canvas and the
palette's first band **Pages**. Switch to Analysis (page button, top left) and press Ctrl+L on
any TIFF: the canvas jumps to Image Input, the card arrives wired into `Output · <file>` and
the Viewer shows the image. Ctrl+PgDn to Image Refinement: a card `Input · Image Input ·
<file>` is already there; add Gaussian Blur from it — its Ready-to-run block offers
*+ Page Output*. Pop the Viewer out with ⇱: a thin dark edge, no white. Welcome card ▸
*Example graph*: four pages, every boundary named; double-click the Analysis Viewer.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [ ] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [ ] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
