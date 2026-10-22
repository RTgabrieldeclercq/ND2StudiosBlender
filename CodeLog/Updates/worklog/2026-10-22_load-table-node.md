# Load table node

- **Date:** 2026-10-22
- **Author:** Gabriel Declercq
- **Branch:** load-table-node
- **Base:** 572312ab

## What changed

A table exported with *File → Export table…* can be loaded back. The new **Load Table** node
(`io.load_table`) reads a Parquet, Arrow or CSV export and puts every table back on its
original domain and layer: structure tables with their `z_kind`, and per-frame, per-plane and
global attributes back into their arrays. With `Onto = tables_only` (the default) it needs no
wire and carries the export's axes and calibration. With `Onto = image` it attaches the tables
to a wired image and refuses one whose B/M/T/Z differ.

Every export now writes an **origin record** (`nodegraph/table_file.py`): when and on which
machine, the graph file, the page and node, the chain of nodes in front of that node, the
source image file(s), the axes, the JSON-safe metadata, and a placement for each table. It
goes inside the file for Parquet and Arrow, and beside it as `<file>.origin.json` for CSV. The
node shows the record on its card (hover text) and in a new inspector section, **Origin**. Its
output carries the record as `table_file_origin`.

The card has a **Load file** button, implemented through a new GUI-side mechanism,
`nodelab_v2/card_actions.py` (a row of buttons on a card, mirrored in the inspector). It loads
the recorded source file on the Image Input page, or reuses it if that page already reads the
file, and publishes it. It then adds or reuses a Page Input on the Load Table's page reading
that output, wires it into the node, and sets `Onto` to `image`. On the Image Input page
itself, or in a Free-only workspace, the source card is wired in directly. A source that has
moved is looked for beside the table file, then asked for.

**Engine:** `NodeSpec.file_identity` is a new hook. The engine folds it into the memo key on
every lookup, and into the lineage. Load Table declares it as the file's path, size and mtime,
so a re-export to the same file name is recomputed instead of served stale.

**Hotspot files touched:** `nodegraph/selftest.py` (new `test_load_table`, placed after the
analysis-store tests rather than at the end), `nodegraph/catalog/__init__.py` (one `MODULES`
line), and `codemap/gen/*` (regenerated). Every node's `fp` in `nodes.jsonl` changed, because
`engine.py` and `registry.py` are in every node module's import closure. `catalog_baseline.json`
gains `io.load_table`.

## Why

The user asked: "Can you currently reload parquet files to look and play with the results?"
Only DIC/DVC/PIV/SerialTrack trials could be reloaded (Load Analysis). Export table was one-way,
and `io.load` reads only ND2/TIFF. They asked for a loader that is tied to its original file,
describes where the file came from, and has a button that loads that file on Image Input and
links it as a page input.

The record had to travel with the file, because the two machines exchange exports. For CSV a
sidecar was chosen over `#` comment lines, because comment lines break Excel and pandas'
default readers.

The memo hook was needed because a root node's key was only its op and params. Without it,
the most ordinary workflow — tweak the analysis, re-export to the same name, re-pull — served
the old table. The hook is generic, so another node that reads a file (Load Analysis's growing
partial trials, for one) can adopt it.

The button lives in the GUI layer (`card_actions`), not on `NodeSpec`, because nothing below
the GUI can carry it out.

## Files

New:
- `nodegraph/table_file.py` — the format: origin record writer and reader, the long-form splitter, CSV typing, axis inference, the origin description
- `nodegraph/catalog/io/load_table.py` — the node
- `nodelab_v2/card_actions.py` — card action buttons (Load Table's Load file)

Changed:
- `nodegraph/registry.py`, `nodegraph/engine.py` — `NodeSpec.file_identity`, folded into the memo key and the lineage
- `nodelab_v2/export.py` — origin record into Parquet/Arrow metadata / CSV sidecar; `export_context`
- `nodelab_v2/window.py` — export passes the context; `load_table_source` (Load file); `_load_source_paths` returns its Page Outputs and takes `preview`
- `nodelab_v2/node_item.py`, `nodelab_v2/scene.py`, `nodelab_v2/inspector.py` — the action row, its signal relay, the Origin section
- `nodelab_v2/spreadsheet.py` — remembers the page of the node it shows (the export names it)
- `MANUAL.md` — Export table section + §15 row
- `codemap/node_roles.json`, `codemap/node_demos.json`, `codemap/node_synopsis.json` — the new op
- `codemap/curated.lock.json` — CON-01, CON-12, INV-03, WF-04 re-read and blessed (`define_node` gained `file_identity`; none needed a wording change)
- **hotspots:** `nodegraph/selftest.py` (`test_load_table`), `nodegraph/catalog/__init__.py` (MODULES), `codemap/gen/*` + `codemap/STATE.md` (regenerated), `scripts/catalog_baseline.json` (saved: `io.load_table` added)
- `scripts/_nodelab_v2_phase5_probe.py` — LT1 / LT2. LT2 drains the runner before moving on: a pull still in flight at the probe's `os._exit` segfaulted it (exit 139) after it printed PASSED.

## How to verify

1. Pull any node with structure tables (Measure, Detect Spots). File → Export table… →
   `cells.parquet`.
2. On the Analysis page, add **Load Table**, Browse… to `cells.parquet`. The card's note reads
   `source <file>`; hover it, or open the inspector's Origin section, for the full record.
   Double-click: the Spreadsheet shows the same tables.
3. Click **Load file**. Image Input gains the source card and a Page Output, the Analysis page
   gains a Page Input wired into Load Table, `Onto` reads `image`, and the Viewer shows the
   points over the image.

Headless: `test_load_table` in the selftest, and probes LT1/LT2 in the phase-5 GUI probe.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (253 `[ok]`, run with `NODEGRAPH_MIN_FREE_RAM=1G`)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED, exit 0 (172 `[ok]`, incl. LT1/LT2)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 144 ops (baseline saved for the new op)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> branch is 0 behind origin/Blender (fetched 2026-10-22); stacked on `nd2-acquisition-clock`, whose 3 commits are already on origin

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
