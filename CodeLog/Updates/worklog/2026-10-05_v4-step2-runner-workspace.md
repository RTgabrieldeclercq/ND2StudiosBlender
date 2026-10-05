# V4 step2 runner workspace

- **Date:** 2026-10-05
- **Author:** hyper
- **Branch:** v4-step2-runner-workspace
- **Base:** e1ae0e28

## What changed

V4.00 step 2 of `CodeLog/ClaudesPlan/V4.00_workspaces.md`: **the runner runs the workspace, not
one document.** `EngineRunner(source)` is bound to the window's `Workspace` (anything that
implements `nodelab_v2.workspace.GraphSource`; a bare `GraphDocument` is still accepted and
wrapped into a one-page workspace, so every pre-V4 caller and probe keeps working). Every run id
the runner stores, hands to the engine or emits on a signal is now **page-qualified**,
`"pg1/n3"` (`qualify`/`split_run_id`): a pull of any node composes its page with every page it
reads from (`Workspace.compose`) and the engine, the memo, the held views, the finished-results
map, the run cones, the ingest table and the plane cache are all keyed in those terms, with the
page's composed revision digest (`Workspace.revision_of`) where the document revision used to be
— so the cached engine is rebuilt exactly when any page in the dependency closure changes, and
only then. A public method takes a bare document id to mean "on the active page" (`run_id()`);
`planned_nodes`, `held`, `sweep_all`, `ingesting()`, `queued_nodes()` return qualified ids. The
grammar that maps a run-graph id back to its card (`#it@`, `%inst`) keeps the page prefix
(`workspace.doc_id_of`, Qt-free; `runner._doc_id_of` delegates). The window splits every runner
signal first and touches the canvas, the viewer and the inspector only for the page it shows:
a result for another page is remembered by the runner and leaves the Viewer alone, while the
cards that page's pull computed on the shown page (its upstream chain) still read queued →
running → done under the run's qualified plan key. Invalidation moves from the document's
touched set to the Workspace's qualified one (`_on_workspace_changed`), so an edit on one page
cancels exactly the runs on other pages whose cone reads it. Bakes and holds record on the page
the run id names. New: `Workspace.page_ids/all_records/meta_seed/iterate_aliases`, module
functions `doc_id_of`/`local_ids`, `revision_of` of an unknown page is `""`; selftest
`test_runner_qualified_ids`; probe section `WS2` (a pull on a page the canvas is not showing
composes the upstream page, finishes under its qualified id, the shared Output is a memo hit
from its own page, an edit on the shown page cancels only the other page's run that reads it);
the C4/C5 queue and cancel probes, the iterate and dock probes and two standalone probes
(`_dock_hold_probe`, `_probe_file_bundle_e2e`) now speak run ids. Hotspots touched:
`nodegraph/selftest.py`, `codemap/gen/*`.

**Found before commit, fixed here.** Gate runs and an adversarial review (one reviewer per
failure mode, each finding put to independent verifiers who tried to refute it) turned up:

1. **A page's run identity could repeat after File → Open.** Fresh page documents restart their
   revision counters, so a second file's pull could reuse the first file's cached engine and
   results. The identity now digests each document's process-wide `GraphDocument.uid`.
2. **Double-clicking a bound Page Input card raised a bare `KeyError`**: the Input has no node in
   the composed graph. It is now served by the upstream Output, and the card joins the run's cone.
3. **Re-pointing a Page Input's Source mid-run cancelled nothing**: a resolved Input is in no
   plan. `ComposedGraph.inputs` lists each Input's consumers, and `EngineRunner._cone_of` adds the
   Input to every run that plans one.
4. **Touched sets for page-level changes.** Adding or duplicating a page published "unknown",
   which cancelled every pull; it now publishes "nothing a run can see". A rename, or an edit to
   an Output that can re-bind a reader, touches exactly the Outputs and the Inputs that read them
   (`Workspace._reader_inputs`). `remove_page` also re-describes its readers, which it used to
   leave on the removed page's envelopes.
5. **A page-reference cycle** (Free pages, a hand-edited file) made every pull, cursor move and
   repaint raise. A reference against the kind order is no longer a dependency, and the run
   identity and composition use a cycle-tolerant closure; the Inputs on a cycle stay unbound and
   say so.
6. **An overlay could lose its display context.** The context now travels with its result, both
   in the worker's delivery packet and beside each remembered result, so an edit that clears the
   shared map cannot strip a delivered or re-served result of its overlay channels.
7. **Listener order after File → Open.** The Workspace now hears its pages' edits first
   (`GraphDocument.on_change(first=True)`), even after a load re-attaches it. The runner therefore
   cancels stale runs before the Movie Editor's change handler re-fetches its sources.
8. **Pin T/Z silently did nothing**: the window's handler now splits the strip's run id. A bake
   on another page releases that page's dormant chain rather than the active page's.

The test fakes that call runner methods unbound now carry the `GraphSource` interface. One
finding is deferred to step 5: a Dock fed through a Page Input still signs only its own page's
upstream chain, so it would not read "stale" after an upstream page changes. No GUI path can
edit an upstream page before step 5, and the plan now lists it there. One finding was refuted:
the hover readout composing per mouse move costs about 0.3 ms extra on a path that was already
uncached, with no wrong output.

Also in this change: the beta-test checklist (`CodeLog/ClaudesPlan/V4.00_beta_tests.md`) is
re-cut so each step's block lists **only that step's new behaviour** (user decision 2026-10-05);
the regression sweep becomes a separate optional pass before a merge. The Step 2 block opens a
hand-made two-page file, `CodeLog/ClaudesPlan/beta/beta2.nd2graph.json` (Input: synthetic Load →
Output `raw`; Refine, active: Input `pg1:raw` → Gaussian → Viewer). `CHANGELOG.md` gains the
Step 1 line it was missing and the Step 2 line.

## Why

Steps 4 and 5 put several viewers and several pages on screen at once, and a viewer bound to
`(page, node)` can only be fed by a runner that knows which page a result belongs to; the
window's single `self.doc` was the one assumption every one of the runner's 40 document reads
made. Qualifying **every** run id — the target page's own nodes included — rather than only the
spliced-in upstream ones is what makes the memo work across pages: a root's `__source__` is its
node id, so a Load reached as `pg1/L` from its own page and from three processing pages is one
seed and one recipe hash, whereas a bare-id shortcut for "my own page" would have given the same
chain two identities and recomputed it on the first cross-page pull (`test_page_composition_
memo_reuse` pins this). Keying results and the cached engine on the composed revision digest
instead of the document revision is the same fact from the other side: an edit on the Input
page changes what the Refine page computes without touching the Refine document, and the old
key would have served the stale result. The window splits ids at the boundary rather than the
runner stripping them, because the viewer bindings of step 4 need the page; the shown page's
cards keep claiming the nodes another page's run computes because those nodes really did run,
and a card that says nothing while its chain is being evaluated for a neighbouring page reads
as a hang. The beta checklist change is the user's call: a step's testers should check what
that step added, and the "did anything else break" question is a merge-time question.

## Files

- `nodelab_v2/runner.py` — bound to a `GraphSource`; qualified run ids throughout; `source`/`document`/`run_id`/`_rev`/`_compose`; `_cone_of`; overlay context in the delivery packet and `_result_ctx`; `_pull_id` serves a bound Page Input
- `nodelab_v2/window.py` — splits run ids (`_local`/`_local_ids`), `_on_workspace_changed` invalidation, non-active-page handlers, `_on_run_started`/`_on_detail_ready`, Pin T/Z split, per-page dormant release
- `nodelab_v2/workspace.py` — `doc_id_of`, `local_ids`, `GraphSource` surface (`page_ids`, `all_records`, `meta_seed`, `iterate_aliases`, `off_change`), `revision_of` guard and uid digest, `ComposedGraph.inputs`, `_reader_inputs`/`_kinds_allow`/`_downstream_of`, cycle-tolerant closure, run-accurate notify scopes
- `nodelab_v2/document.py` — `on_change(first=True)`, `uid`
- `nodegraph/selftest.py` — `test_runner_qualified_ids`; the unbound-call fakes in six older tests (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — WS2; run-id forms in G7, E8c, Mini, Compare, dock, V1, detail, iterate, channel, C4, C5
- `scripts/_dock_hold_probe.py`, `scripts/_probe_file_bundle_e2e.py` — run-id forms
- `CodeLog/ClaudesPlan/V4.00_beta_tests.md` — per-step blocks cover new features only; Step 2 block rewritten; sweep moved to the end as optional
- `CodeLog/ClaudesPlan/beta/beta2.nd2graph.json` (new) — the two-page file the Step 2 checks open
- `CodeLog/ClaudesPlan/V4.00_workspaces.md` — delivery table row
- `CodeLog/Updates/CHANGELOG.md` — Step 1 and Step 2 lines
- `codemap/gen/*`, `codemap/STATE.md` (regenerated)

## How to verify

- `PYTHONUTF8=1 python -B -m nodegraph.selftest` — the `[ok]` line `runner qualified ids: …`.
- `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` — `WS2 runner on the
  workspace: …` plus the unchanged C4/C5 queue-and-cancel sections, now in run ids.
- By hand (the Step 2 block of `V4.00_beta_tests.md`): `python run.py`, Ctrl+O
  `CodeLog/ClaudesPlan/beta/beta2.nd2graph.json` → the Refine page shows; double-click the Viewer
  → `(3 computed, 0 cached)` then `(0 computed, 3 cached)` on a re-pull; change the Gaussian's
  sigma → `(1 computed, 2 cached)`; set the Page Input's source to `pg1:nope` → `unbound` and a
  `Page Input is not bound…` refusal in the console.
- `PYTHONPATH=. python scripts/_dock_hold_probe.py` → `ALL 11 DOCK HOLD-TIER PROBES PASSED`.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: red only at the ONE pre-existing failure,
  `test_write_movie`'s codec-rounding monotonicity (means 53.14, 53.14, 52.40 …, identical on
  `origin/Blender`), which aborts the tests registered after it. All 144 registered tests were
  also run one by one, continuing past failures: 143 pass, `test_write_movie` fails.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (90 checks, WS2 included)
- [x] `scripts/_dock_hold_probe.py` -> ALL 11 DOCK HOLD-TIER PROBES PASSED. Two non-gate probes
  fail IDENTICALLY on `origin/Blender` (`e1ae0e2`) before any V4 work, so not this change:
  `_hotreload_probe.py` ("palette lost the node") and `_probe_file_bundle_e2e.py` (a TIFF card
  under the newer default `access=direct`).
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 95 ops (no catalog change)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT — 95 nodes
- [x] `python scripts/_worklog.py check` -> WORKLOG OK
- [x] `python scripts/_sync_check.py` -> feature branch, 0 behind origin/Blender

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
