# V4.00 step 11b — New page…, page recipes and master pages

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of step 11a

## What changed

A new page can now start from something, not only from an empty canvas:

- **New page…** (the page switcher) opens a dialog (`nodelab_v2/new_page_dialog.py`) that
  settles the page before it exists: its kind and name, how it **starts** — *Empty page* (one
  Page Input, bound), *Page recipe*, or *Linked to a master page* — and which earlier
  **Output** its Page Input reads (the nearest one preselected). The quick *New page ▸ <kind>*
  submenu stays.
- **Page recipes** (`nodelab_v2/page_recipes.py`, Qt-free): a page's graph under a name, a
  kind and a description. Eight built-ins ship in `nodelab_v2/builtin_page_recipes/<kind>/`:
  *Load & name*; *Smooth & threshold*, *Background & deconvolve*; *Label & measure*, *Track
  objects*; *Plot over time*, *Distribution*, *Summary table*. *Save as page recipe…* (the
  switcher, and Graph ▸ *Save page as a page recipe…*) writes the page to
  `~/.nd2studios/page_recipes/<kind>/<slug>.json`; a saved recipe with a built-in's name
  replaces it in the list; a file that is not a recipe is skipped. `NODELAB_PAGE_RECIPES_DIR`
  redirects the folder, `NODELAB_PAGE_RECIPES=0` switches it off. Instantiating loads the body
  through `Workspace.load_page_body` (op renames apply), binds the Page Inputs to the chosen
  Output and keeps the Output names unique.
- **Master pages**: `Page.is_master` — *Set as master page* in the switcher, ★ on the row and
  the page button, offered first in the dialog's *Master* menu; any plain page can still be
  chosen, a linked page cannot be one. In the 3.0 file only when set (older files stay
  byte-identical); a linked page flagged as master is refused on load. A linked page's label
  reads `(linked · N overrides)`; a source chosen for it is recorded as that page's override.
- **Entry points**: a named Page Output's right-click *New page from this output…* (a page of
  the next kind reading it); the welcome card of an empty Refinement, Processing or Analysis
  page offers *Start from a page recipe…*, which fills that page.
- `Workspace`: `set_master`, `masters`, `feeders_for_kind`, `sources_for_kind`,
  `default_source_for_kind`, `load_page_body` (the unique-duplicate path now goes through it).
- **Tests**: `test_page_recipes_builtin_load` (every built-in instantiates and composes; two
  of them pulled end to end), `test_page_recipe_roundtrip`, `test_page_master_flag`; probe
  NP1–NP4 and the LK1 label. **Docs**: MANUAL §2 *Page recipes and masters*, §2b steps 2 and
  4 (step 4 also corrected: the Processing page's Input arrives bound to the newest Refinement
  page, which is the linked copy), §13 page record; codemap CON-18 addition and CON-21
  (blessed).

## Why

The user asked that a new page can be "inherited from recipe graphs, or graphs that have
already been built as a master/slave": recipes as prebuilt node graphs that do something
standard to the data and can be modified, a page the user can set as master, linked copies
whose value changes are customisations rather than graph edits, and any new page still free
to read any earlier Output. Linked pages already existed (step 6); what was missing was a way
to start a page from a prebuilt graph, to mark a master, and one place that offers all three.
"Page recipe" is kept distinct from the existing "LabLink recipe" (a whole graph published to
a hub) in every label and in code, and the built-in data folder is not named like its module.

## Files

- `nodelab_v2/page_recipes.py`, `nodelab_v2/new_page_dialog.py` — new
- `nodelab_v2/builtin_page_recipes/<kind>/*.json` — eight built-in recipes (new)
- `nodelab_v2/workspace.py` — `Page.is_master`, the master and new-page helpers, the file
- `nodelab_v2/window.py` — page menu, `new_page_dialog`, `_apply_new_page`,
  `_new_page_from_output`, `set_master_page`, `save_page_as_recipe`, labels, Graph menu
- `nodelab_v2/scene.py` — `new_page_from_output`; `nodelab_v2/welcome.py` — recipe mode
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`, `codemap/concepts.md` (CON-18, CON-21), `codemap/gen/*`, `codemap/STATE.md`,
  `codemap/curated.lock.json`

## How to verify

`python run.py` → File ▸ Load a TIFF (it lands on Image Input) → page switcher ▸ *New page…*
→ Kind *Image Refinement*, *Page recipe* ▸ *Smooth & threshold* → *Add page*: the page holds
Page Input → Gaussian Blur → Threshold → `Output · mask`, reading the loaded image.
Switcher ▸ *Set as master page* (★), then *New page… ▸ Linked to a master page*: the new page
reads `(linked · 0 overrides)`. Right-click `Output · mask` ▸ *New page from this output…*.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [ ] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [ ] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
