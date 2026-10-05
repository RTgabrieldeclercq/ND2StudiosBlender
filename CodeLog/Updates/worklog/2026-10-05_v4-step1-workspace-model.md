# V4 step1 workspace model

- **Date:** 2026-10-05
- **Author:** hyper
- **Branch:** v4-step1-workspace-model
- **Base:** e1ae0e28

## What changed

V4.00 step 1 of `CodeLog/ClaudesPlan/V4.00_workspaces.md`: the **Workspace model**, with the GUI
still showing one page. A saved file is now a workspace — `format_version` **3.0**,
`{format_version, app_version, workspace: {active, next_page_seq, pages: [...]}}` — holding
several node graphs ("pages") of a typed kind (`input` → `refine` → `process` → `analyze`, plus
`free`). A pre-V4 `2.0` single-graph file still opens, as one Free page named after the file, and
`nodegraph.serialize.from_dict` on a 3.0 file returns its active page so every 2.0-era caller
keeps working. Two new ops, `page.output` (names the Dataset wired in as a variable of its page,
optionally stamping a `condition`) and `page.input` (reads `<page id>:<name>` from a page of a
strictly earlier kind, or any page when either side is Free), are the cross-page wiring;
`Workspace.compose(page)` splices every page a target reads from into ONE run graph with node ids
qualified `<page>/<node>`, so a refinement chain pulled from three processing pages is one memo
entry. `nodelab_v2/workspace.py` is Qt-free (`Page`, `PageKind`, `Workspace`, `ComposedGraph`,
`qualify`/`split_run_id`); `GraphDocument` gains the additive hooks it needs (`seed_hooks`,
`page_sources`/`source_choices`, `repropagate`, `rebase_path`, `off_change`, `page_kind`,
`editable_topology`); `codemap/node_roles.json` moves to schema 2 with a `pages` table and
`op_pages` overrides, read through `nodegraph.roles.pages_of/ops_for_page`; readiness reports
`unbound` (an Input whose source resolves to nothing) and `duplicate_output`; the LabLink worker
opens a 3.0 file, picks the recipe's `"page"` (id or name, absent = active) and runs it composed,
and `hello` advertises `graph_formats`. `window.py` routes open/save/save-as through
`Workspace.single(doc)`. Six selftests (`test_workspace_model`, `test_page_composition_memo_reuse`,
`test_workspace_format_v3`, `test_page_kind_catalog`, `test_lablink_page_select`, the
`test_readiness` extension) and probe section `WS1` cover it; MANUAL §file format and the node
table describe the two ops. Touches the four merge hotspots (`selftest.py`, `catalog_baseline.json`,
`codemap/gen/*`, `node_synopsis.json`) — regenerated, not hand-edited.

Two things found while closing the gates, fixed in the same change. (1) The GUI probe's
dock-store assertion still expected `<graph>.docks/<node>`; a page's docks now live under
`<graph>.docks/<page id>/<node>` (the MANUAL text above), so the probe asserts the page-qualified
path. (2) **The selftest had been silently running only its first half.** `test_codemap()` sits
mid-way through `main()` and raises on curated codemap prose whose pinned file changed; nine
entries (CON-02, CON-11, CON-13, INV-02, INV-07, INV-08, INV-10, INV-11, WF-07) had drifted
between 2026-08-05 and 2026-10-01, so every run since aborted there and the **42 tests after it**
(`test_lablink`, `test_nd2_direct_access`, the overlay and batch suites, …) never ran. Each entry
was re-read against its file's actual change (all nine still true; WF-07 gained the placement
vocabulary the ingest now carries) and blessed individually — never `--all`. (3) With the
run reaching its second half, `test_codemap`'s MANUAL §15 phantom check fired for the first
time: the Timeseries Builder row still back-ticked `util.chain` (renamed 2026-10-02) in its
"replaces Chain Files" aside, which the check reads as a documented op — now plain text.

## Why

One canvas per file has become the ceiling: a lab workflow is "bring the dishes on, clean and
segment, measure and track, then plot and export", and today that is one graph of forty cards or
four files that cannot see each other's results, re-ingesting the same ND2 in each. The V4.00
record (decisions confirmed 2026-10-05) answers with typed pages wired by NAMED outputs rather
than, say, graph groups or sub-graph nodes: a name is something a later page can pick from a
menu, survives the upstream page being re-laid-out, and — because ids are qualified per page at
run time while a node's recipe hash carries no node id — lets the memo treat the shared upstream
chain as one computation however many pages pull it. Step 1 lands the model and the file format
FIRST, with no GUI change beyond save/open, so that steps 2–6 (runner, dock shell, viewers, page
switcher, dependent pages) build on a format and a composition that are already tested headless,
and so a file saved today is already a V4 file. Deviation from the record's Step 1 text: the
single-graph `FORMAT_VERSION` stays `"2.0"` and the new constant is `WORKSPACE_FORMAT_VERSION =
"3.0"`, so the many 2.0-era writers (recipes, probes, `to_json`) keep writing the format they
test against; only `Workspace.save_file` writes 3.0. The probe's G6 keeps asserting 2.0 for a
single-graph save and WS1 asserts 3.0 for a workspace save, which is the actual contract.

## Files

- `MANUAL.md`
- `codemap/node_roles.json`
- `codemap/node_synopsis.json`
- `lablink_recipes/nd2studios/cell-segmentation/recipe.json`
- `lablink_recipes/nd2studios/selftest-synthetic/recipe.json`
- `nodegraph/roles.py`
- `nodegraph/selftest.py`
- `nodegraph/serialize.py`
- `nodelab_v2/document.py`
- `nodelab_v2/lablink/__init__.py`
- `nodelab_v2/lablink/protocol.py`
- `nodelab_v2/lablink/worker.py`
- `nodelab_v2/ops.py`
- `nodelab_v2/readiness.py`
- `nodelab_v2/theme.py`
- `nodelab_v2/window.py`
- `scripts/_node_synopsis.py`
- `scripts/_nodelab_v2_phase5_probe.py`
- `scripts/catalog_baseline.json`
- `nodelab_v2/workspace.py` (new)
- `CodeLog/ClaudesPlan/V4.00_beta_tests.md` (new) — the by-hand checks to run after each V4 step, plus the regression sweep
- `codemap/gen/*`, `codemap/STATE.md` (regenerated: `page.input`, `page.output`, `workspace.py`)
- `codemap/workflows.md` (WF-07: placement vocabulary), `codemap/gen/MANIFEST.json` (nine curated entries re-pinned)

## How to verify

- `PYTHONUTF8=1 python -B -m nodegraph.selftest` — the `[ok]` lines for `workspace model`,
  `page composition memo reuse`, `workspace format v3`, `page kind catalog`, `lablink page select`,
  `readiness`.
- `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` — `WS1 workspace file: Save
  writes format 3.0 (one Free page, app_version stamped)`; G6 still writes 2.0 for a single graph.
- By hand: `python run.py`, build anything, **Ctrl+S** → the file's first line is
  `"format_version": "3.0"` with one page of kind `free`; **Ctrl+O** on any pre-V4
  `*.nd2graph.json` → opens as one Free page named after the file. Place a **Page Output**, name
  it, then a **Page Input**: its Source menu is empty on the same page (no earlier page exists yet
  — the switcher is step 5) and the readiness panel says `unbound`.
- `python scripts/_node_synopsis.py show page.input` — `pages: refine, process, analyze`.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: 146 `[ok]` lines (was ~100 — the run
  now passes `test_codemap` and the phantom check), then red at the ONE pre-existing failure,
  `test_write_movie`'s codec-rounding monotonicity (means 53.14, 53.14, 52.40 …), which aborts
  the tests registered after it; those were run directly and pass (see the step's reply).
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (WS1 included)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 95 ops (re-blessed for `page.input`/`page.output`)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (no curated entry unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT — 95 nodes
- [x] `python scripts/_worklog.py check` -> WORKLOG OK
- [x] `python scripts/_sync_check.py` -> feature branch, 0 behind origin/Blender

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
