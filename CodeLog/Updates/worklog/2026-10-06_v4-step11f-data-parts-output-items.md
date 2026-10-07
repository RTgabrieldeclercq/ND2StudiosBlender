# V4.00 step 11f — a node's output one kind of data at a time (part sockets), a Page Output that collects several items

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a–11e

## What changed

- **Part sockets: each kind of data separately, or combined.** A node's `out` carries its whole
  Dataset — after a Threshold the image and its mask, after Label the image, the mask and the
  labels (raster and table). Any card whose output carries more than one kind of data now also
  offers each one on a socket of its own under `out`: `image only`, `mask only`, `labels only`,
  a point or track table by its name (`GraphDocument.data_parts`, from the edit-time envelope's
  layer catalog; synthetic `part:<name>` sockets in `output_specs`, appearing only when there
  are two or more kinds). `out` still passes them combined. A part wire is rewritten at graph
  build into a hidden `data.part` tap (`ops.materialize_part_taps`, one per node and part,
  shared), a GUI-layer op beside `page.*`, hidden from the palette, whose compute keeps that
  part ALONE: `image` — the image with no layers or tables; a layer — that layer (a label
  raster and its table under one name, `ops.part_of`) with its raster AS THE IMAGE (a mask a
  0/1 uint8 image, labels an integer one; the voxel raster already has the image's shape, so
  nothing is copied) and `bit_depth` dropped; a table-only part (points, tracks) has no image.
  A part that is no longer on the wire refuses by name. A switched-off node keeps a part wire
  pointing at its part (`_bypass_muted`).
- **The engine learns `keep_layers`.** A `NodeSpec` may declare `keep_layers(params, modes)
  -> frozenset | None`: the edit-time pass then keeps only the input's layers of those names,
  their columns, and the structure domains they live on (`metadata._kept_only`; the lattice
  always stays). Every catalog node leaves it `None`; `data.part` is its user. Downstream of a
  part, layer and column menus offer only that part.
- **A Page Output collects several items.** `page.output` takes `data` plus `data_2` …
  `data_8` (one growable group: a slot appears under the last one wired) and an `Items` field
  naming them in wiring order; a blank entry takes a name from its wire — a part's name, a Page
  Input's variable, or the source node's title — unique within the Output
  (`GraphDocument.output_items`). The card's slots read as the item names and its title counts
  them (`Output · work · 3 items`; the empty slot reads `+ item`, `NodeItem._input_row_text`). Items are kept SEPARATE, never merged: at graph build each
  extra item becomes a `page.output` node of its own (`ops.materialize_output_items`,
  `__item__<output>__data_2`), so pulling the Output previews its first item and computes no
  other. A Page Input reading such a variable offers each item on an `item · <name>` socket
  beside `out` (the first item; `doc.page_items` → `Workspace.input_items`); a wire from one
  becomes a Page Input tap reading that item (`ops.materialize_input_items`), which the
  Workspace seeds with that item's envelope (`_input_seeds`) and splices onto the item's node
  at compose (`Workspace.resolve_item`). The Pages panel lists the items on the Output's row.
- **A flat plane shows as what it is** (`viewer._flat_window`): the Viewer's percentile window
  for a plane whose values are all equal used to be `[v, v+1]`, which draws a constant 1 —
  an all-ones mask, the first thing a mask part showed on the example graph — as black; a
  positive flat plane is now shown bright (`[0, v]`).
- **Selecting a card is ~5x faster** (found while the GUI probe's debounce check went
  flaky): `catalog.module_order` re-evaluated `set(discover())` for every listed module —
  about a hundred walks of the catalog folder per call — and Properties asks it on every
  selection (`hotreload.module_of_op`, for the Reload button), so each click cost ~0.4 s; one
  walk now (0.07–0.15 s per selection). Long enough a click straddled the 170 ms preview
  debounce, which is what made `Mini-map: a multi-node selection debounces to a single pull`
  queue two pulls.
- **Gates' baselines**: `scripts/catalog_baseline.json` re-saved (op added `data.part`;
  `page.output` gained `data_2`…`data_8` and `items`), `codemap/node_synopsis.json` regenerated
  and `data.part` given the "Dataset & channel organization" role in `node_roles.json`.
- **Tests**: `test_data_parts`, `test_page_output_items`; probe IP1 (the Threshold card's part
  sockets, a two-item Output, the next page's item sockets and a pull through one). **Docs**:
  MANUAL §4 *One kind of data at a time*, §2 Pages (several items), the Page Output / Page
  Input reference rows, three §18 rows; codemap CON-22 (new), CON-12, WF-04.

## Why

The user, right after step 11e: "I need the page output node to be able to collect multiple
data", and then: "for any node that has multiple data types as an output, we should be able to
pass each data type separately or combined." Asked how, they chose (1) the items of a Page
Output kept SEPARATE — one socket per item on the next page's Page Input, like Split Positions'
positions or the Load card's channels — rather than merged along an axis or batched; and (2) a
data type taken on its own to carry that piece ALONE, a raster shown as the image, rather than
the image plus that piece or a layer with no image.

Why this shape: both follow the pattern the app already uses for channels, positions, groups
and batch members — a synthetic socket on the card, a real tap node at graph build — so the
engine runs ordinary nodes, the memo shares one tap among every wire, and nothing new is
needed in the runner. The item Outputs are split at build time rather than given a
multi-output compute because the engine is one payload per node; splitting also keeps a
preview of the Output from computing every item. Edit-time accuracy for a part needed one
generic engine declaration (`keep_layers`) — the pass could add layers but never keep only
some — rather than over-claiming the input's layers and domains, which would make every menu
downstream of a part offer layers that are not there.

## Files

- `nodegraph/registry.py` — `NodeSpec.keep_layers`, `define_node(keep_layers=)`
- `nodegraph/metadata.py` — `_kept_only` in `propagate_meta`
- `nodegraph/catalog/__init__.py` — `module_order` walks the folder once
- `nodelab_v2/ops.py` — `data.part` (compute, meta, keep), part / item constants and helpers,
  `page.output`'s item slots and `items`, `materialize_part_taps`, `materialize_output_items`,
  `materialize_input_items`, `prepare_run_graph` order
- `nodelab_v2/document.py` — `data_parts`, `output_items`, part and item sockets in
  `output_specs`, item labels in `input_specs`, the `page_items` hook, `_bypass_muted` keeps
  part wires
- `nodelab_v2/workspace.py` — `input_items`, `resolve_item`, item taps in `_input_seeds` and
  `compose`, the outline lists items
- `nodelab_v2/node_item.py` — the Output card's item count and slot names
- `nodelab_v2/viewer.py` — `_flat_window`: a flat positive plane is shown bright
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `scripts/catalog_baseline.json` (hotspot, re-saved), `codemap/node_roles.json`,
  `codemap/node_synopsis.json`
- `MANUAL.md`, `codemap/concepts.md`, `codemap/workflows.md`, `codemap/curated.lock.json`,
  `codemap/gen/*`, `codemap/STATE.md`

## How to verify

`python run.py` → *Example graph*. On Image Refinement the Threshold card has `image only` and
`mask only` under `out`. Drag `mask only` onto the Page Output (`mask`): a second slot fills,
the title reads `Output · mask · 2 items`. On Image Processing the Page Input card now has
`item · threshold` and `item · mask`; wire `item · mask` into a Median and pull it — the Viewer
shows the 0/1 mask. Type `smooth, cells` into the Output's *Items* field and the next page's
sockets follow.

## Gates

- [x] selftest — 187 tests run; only `test_write_movie` fails (pre-existing codec rounding).
  `test_codemap` failed once (MANUAL §15 lacked the hand-written `data.part` row) and passes
  with it; after the last edits (the card's slot names, the viewer window, `module_order`)
  `test_codemap`, the ten catalog / registry tests and `test_live_reload_contract` were re-run
  and pass
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 151 checks (new: IP1), on the final code,
  run alone
- [x] catalog snapshot — CATALOG IDENTICAL, 104 ops (baseline re-saved for the intended
  change); synopsis CURRENT, 104 nodes
- [x] codemap — CODEMAP CURRENT (CON-22 new; CON-01, CON-12, INV-03, WF-04 re-read and
  blessed — `define_node` gained `keep_layers`)
- [x] native launch — Windows platform with a COPY of the user's layout: a part socket wired
  into a two-item Output, the next page's item socket pulled into the Viewer (the mask shown),
  a clean close; the real `~/.nd2studios/layout.json` untouched
- [ ] sync check — run before the push
