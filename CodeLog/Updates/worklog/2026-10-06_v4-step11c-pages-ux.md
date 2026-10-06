# V4.00 step 11c — keeping track of pages; a start card per page kind; what a page reads is what the Viewer shows

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a and 11b

## What changed

Three reports from the user's first session with the step-11 build:

- **The start card can be dismissed and asks for what each page needs.** `welcome.WelcomeCard`
  is configured per page kind (`configure`): an **Image Input** page offers *Load image…*,
  *Load sequence…*, *Example graph*; a **Refinement / Processing / Analysis** page shows a
  BANNER along the canvas's bottom edge — clear of the Page Input it was seeded with — that
  names what that Input reads, offers the kind's page recipes as buttons (one click fills the
  page, its Input reading what the seed read), *More recipes…*, *Link to a master…* and
  *Start empty*, or *Go to Image Input* while nothing upstream is published; a **Free** page
  keeps the original card. A ✕ (and *Start empty*) hides the card for that page for the
  session; File → New brings the cards back. `&` in a recipe name is literal. The seeded
  Page Input is framed at the canvas's top left (`_frame_seed`).
- **Pages are easy to keep track of.** Every canvas has a **page tab strip**
  (`canvas.PageTabs`): one tab per page in page order with its kind's dot; click to show,
  drag to reorder (the page order follows), double-click to rename, right-click for the page
  menu, **+** for *New page…*. A **Pages** panel (`pages_panel.PagesPanel`, left, above
  Nodes) lists every page grouped by kind in pipeline order with what its Page Inputs read
  and which Outputs it publishes (★ master, linked pages name their master); click to show,
  double-click to rename, right-click for the menu. New pages are inserted after the last
  page of their kind or an earlier one (`Workspace.insert_index_for`), so the pages stay in
  pipeline order however they are added. `Workspace.page_summary` feeds both.
- **What a page reads is what the Viewer shows.** Split Positions → a Page Output of one
  position → on Image Refinement the Page Input read that one position, but the Viewer still
  showed every position: click-to-preview is off by default, so nothing was pulled on
  arriving and the Viewer kept the load's preview from Image Input. Now arriving on a page
  whose Viewer shows another page's node previews the page's first bound Page Input
  (`_preview_page_reads`), and selecting a Page Input or Page Output card previews it like a
  Viewer or plot card (`_previews_on_select`). The data path itself was already right (the
  `posK` socket is materialized into a `util.select_position` tap before composition).
- **Also fixed:** a page's Page Input is seeded ONCE per session (`_seeded_pages`) — a
  deleted seed no longer came back on the next click (review finding); a **bundle** of TIFFs
  starts on the `ingest` access mode like a single TIFF card (its first pull failed on
  `direct`).
- **Tests**: `test_page_order_and_summary`; probe WC1 (start card per kind, dismissal,
  recipe button, seeded once), PT1 (tabs), PP1 (Pages panel), SP2 (split position seen on
  the next page; boundary cards preview on select), NP4 updated for the banner. **Docs**:
  MANUAL §2 window / Pages / Page recipes, §2b step 2, §18 two rows; codemap CON-17.

## Why

The user, testing the build: "when I make a new node graph page, it opens with a 'start your
graph' overlay that cannot be dismissed. This should ask for different things for various new
node graph pages"; "it's hard to keep track of all the various node graph pages that we have
open, we need to make a way to organize these"; "when I split an image's locations and output
that, I go to image refinement and the first node inputs the one location that I split, but
the visual is still displaying all locations." A tab strip keeps every page one click away on
the canvas itself (Blender's workspace tabs are the model); the panel adds what a strip cannot
show — what connects the pages. Previewing on arrival rather than turning click-to-preview on
by default keeps the existing preference intact while answering the actual confusion: the
Viewer showing another page's data.

## Files

- `nodelab_v2/welcome.py` — rewritten: per-kind card / banner, dismissal, recipe buttons
- `nodelab_v2/pages_panel.py` — new
- `nodelab_v2/canvas.py` — `PageTabs`, `sync_tabs`, the tab strip's requests
- `nodelab_v2/window.py` — the Pages panel and its dock spec / default size, tab support
  (`page_tab_items`, `move_page`, `next_page_kind`), `_refresh_page_views`, the start card's
  wiring and `_sync_welcome`, `_start_page_from_recipe`, `_dismiss_welcome`,
  `_goto_input_page`, `_frame_seed`, seeding once, `_previews_on_select`,
  `_preview_page_reads`, TIFF bundles on ingest
- `nodelab_v2/workspace.py` — `insert_index_for`, `page_summary`
- `nodelab_v2/page_recipes.py` — new pages in pipeline order
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`, `codemap/concepts.md`, `codemap/gen/*`, `codemap/STATE.md`

## How to verify

`python run.py`: tabs run along the top of the canvas and the Pages panel sits above Nodes.
Load three TIFFs as one bundle (or an ND2 with several positions), add Split Positions and wire
`pos1` into a Page Output; click the *Image Refinement* tab — the Viewer shows ONE position
(what the page's Page Input reads) and the start banner offers *Smooth & threshold*; click it
— the page fills in view. Click ✕ on a start card: it stays hidden on that page until
File → New.

## Gates

- [x] selftest — 177 tests run; only `test_write_movie` fails (pre-existing: H.264 frame means
  non-monotone on this machine's encoder, unrelated)
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 140 checks (new: WC1, PT1, PP1, SP2). The
  probe's final `os._exit` can crash in process teardown with ~40 worker threads alive (seen
  with `-X faulthandler`, after the PASSED line); not a failed check
- [x] catalog snapshot — CATALOG IDENTICAL, 103 ops
- [x] codemap — CODEMAP CURRENT; synopsis CURRENT
- [ ] sync check — run before the push
