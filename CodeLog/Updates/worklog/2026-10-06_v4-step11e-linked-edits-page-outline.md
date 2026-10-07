# V4.00 step 11e — a Source menu coloured by page kind, unique variable names, three answers to a linked-page edit, on/off per node, the Pages panel as an outline

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a–11d

## What changed

Five requests from the user's third session with the step-11 build:

- **A Page Input's Source is a dropdown on its card, coloured by graph type.** The `source`
  pill on a Page Input card used to show the stored `pg1:raw` and open a text editor; it now
  reads what the Input reads — `Image Input · raw ▾` — edged in the colour of that page's
  KIND, and a click opens a menu of every Output the Input may read, `<page> · <variable>`,
  each with a dot in its page kind's colour (the page tabs' colours) and the current one
  ticked (`NodeItem._open_source_menu`). The inspector's Source combo carries the same dots.
  The colour comes from a new document hook, `doc.source_kind` (`Workspace.source_kind`).
- **Output variable names are unique across the workspace.** A Page Output's name used to be
  unique only by default, and only within its page; a duplicate was a readiness warning. Now
  `GraphDocument._settle_output_name` runs on every `add_node` and `touch` — the two ways a
  name arrives — and asks the new hook `doc.claim_output_name`
  (`Workspace.claim_output_name`): a name another Output already has, on any page, becomes
  `name2` (case-insensitive, cleaned of `:` `/` `\`; a blank stays blank), and the window
  says so on the status bar (`doc.renamed_output`). The pages linked to one master share its
  names by design. A copied page, or a linked page made unique, renames its taken names
  (`Workspace.dedupe_outputs`; a page recipe placed twice publishes `mask` and `mask2`) and every Page Input reading them follows
  (`_follow_renames`). `unique_output_name` is workspace-wide too, so a hand-placed Output,
  a loader and a page recipe all get a free name. The name helpers moved to `ops.py`
  (`sanitize_output_name`, `next_free_name`, `DEFAULT_OUTPUT_BASE`; workspace re-exports).
- **Changing a linked page's graph asks how.** A structural edit on a linked page used to be
  refused with a hint. The first one now opens a dialog (`nodelab_v2/linked_edit_dialog.py`,
  shown from `MainWindow.ask_linked_edit` through the window's `_topology_ok` and the scene's
  `topology_gate`) with three answers:
  *Make unique* (as before, then the gesture runs on the page's new scene);
  *Keep the change on this page* — a **modified linked page** (`LinkedDocument.edit_mode ==
  "modified"`, saved): the edit runs on the live mirror and `_record` diffs it against the
  master into the page's STRUCTURE — own nodes (ids `nL1`…), master nodes removed, wires
  added and removed (`structure_dict`, saved as the page record's `structure`) — which
  `_mirror` lays back over the master on every change, so the master's other edits keep
  arriving; where both wire one single input this page wins; an own id the master later takes
  moves aside; the page menu's *Drop this page's own changes* reverts it;
  *Add it to the master, switched off* (`"master"`, this session only): the edit is made ON
  the master in a form that changes nothing it computes — an added node arrives there muted,
  on every other linked page too, and switched ON here; a wire or a removal goes through only
  when the master's bypassed run graph is unchanged (`_keeps_master`), a muted node's removal
  heals its chain, and a node that cannot be switched off is refused; the page menu's *Stop
  sending edits to the master* asks again. Frames, groups and zones stay the master's
  (`_needs_shape`: only *Make unique* is offered); a load never offers the master answer (a
  Load card is where data starts). Cancel changes nothing. A refusal part-way through a
  gesture reaches the status bar (`scene._refusable`, the window's `_needs_topology`). The
  canvas menu's structural entries say they will ask (`ASK_HINT`) rather than being greyed.
  Titles read `(linked · modified · N overrides)` / `(linked · edits → master · …)`.
- **On / off per node, from the Pages panel.** Switching a node off (mute) on a linked page
  is now that page's own setting — an override (`"muted"`) counted, kept through master edits,
  saved and reset like a value (`LinkedDocument.set_muted`; `check_overrides` accepts it).
  The Pages panel has an on/off switch for every node that may be switched off, and the
  status bar names what was switched.
- **Only a node that keeps the kind of data may be switched off.** `document.
  pass_through_reason(spec, params, modes)` decides it from the node's declarations: a node
  that adds a domain (mask, labels, points, tracks, mesh, per-frame values…), names a layer
  (`layer_out`), adds layers (`extra_layers`) or table columns (`adds_columns`), makes a
  picture (`fresh_output`), has no Dataset input or output, or is the graph's wiring (page,
  group, zone, Iterate, reroute) cannot — 52 of the 103 ops can (every image enhancement,
  crops, projections, resampling, channel picks, writers). `set_muted` refuses switching such
  a node OFF with the reason, never switching it back ON (a file from before may hold one);
  the canvas menu's *Muted* is greyed with the reason; `M` leaves it on and says why.
- **The Pages panel shows each page's graph as a hierarchy.** Under each page, its nodes in
  data-flow order (`Workspace.page_outline` → `OutlineRow`, Qt-free): its Page Inputs first
  (`⇤ Image Input · raw`, dotted in the colour of the page kind they read), then its Load
  cards, and from each the chain it feeds — a chain at one depth, a branch nested one level
  under the node it leaves, a node fed by several listed after the last, reroutes left out.
  **Outputs stand out**: tinted in the page kind's colour, the variable name in bold mono
  (`⇥ mask`) and which pages read it. A switched-off node is struck through, a modified linked
  page's own node marked `+`. Clicking a node shows its page with the node selected and in
  view (`MainWindow.show_node`); a folded page stays folded across rebuilds.
- **Tests**: `test_pass_through_rule`, `test_output_names_unique`, `test_linked_on_off`,
  `test_linked_modified_structure`, `test_linked_send_to_master`, `test_page_outline`; probe
  SP1 (the card's Source menu, the inspector's dots, a taken name), PO1 (the outline and
  switches), LE1 (the three answers, shapes, the dialog, the canvas menu's Muted, M on a linked
  page); G3, LK1, LK3, PP1, `test_readiness_fixes`, `test_workspace_model` and
  `test_linked_document_mirrors_master` follow the new rules. **Docs**: MANUAL §2 (Pages panel,
  Page Output/Input, Linked pages), §5 (Mute), the node reference rows, four §18 rows; codemap
  CON-17, CON-18.

## Why

The user, testing the step-11d build: "For each input node, I want it to have a dropdown
selection for all output nodes that it can use colored by graph type. Output nodes should
ensure variable names are unique. When making modifications to the node graph of a slaved
node graph I want a popup to ask if it should be made unique, made into a modified slave (any
changes to the master downstream of the mod is still accepted), or modify the master from the
slave (master gets the node but it is turned off). On the page window, I want the ability to
turn any node on or off via the passthrough node ability. Passthrough is only valid for a node
if it does not transform the datatype from input to output. The page window should show a
better hierarchy of the node graph. Outputs from a hierarchy should be highlighted in some way
and have their variable name easily identified."

Decisions taken, and why:
- **Unique names are workspace-wide, except within a linked family.** A Source value is
  page-qualified, so per-page uniqueness would have been enough for addressing — but the user
  asked for unique VARIABLE names, and a menu listing `Image Input · out` beside `Refinement ·
  out` is what that prevents. Linked copies exist to run one workflow per condition with the
  same variables, so they share their master's names. Enforcing at the document (`add_node`,
  `touch`) rather than in each editor means the card, the inspector, a recipe, a loader and a
  script all get it.
- **"Turned off" is mute, and on a linked page it is a per-page value.** The request's third
  answer only works if the slave can have ON what the master has OFF, so on/off had to become
  an override. Muting was a structural edit before; it is not one any more.
- **"Send to the master" is checked against what the master COMPUTES**, not against a list of
  allowed gestures: an edit goes through when the master's bypassed run graph is unchanged,
  which accepts inserting a node and wiring it in or out (in that order) and refuses anything
  that would change the master or its other pages — the property the request names.
- **A modified page stores a diff, recomputed after each edit**, rather than a log of
  operations: the master's later edits are merged by re-mirroring and re-applying the diff,
  which is how "any changes to the master downstream of the mod are still accepted" holds; a
  conflict on a single input goes to the slave, whose change was deliberate.
- **The datatype rule reads declarations, not a pull.** "Does not transform the datatype"
  means what flows out is the kind of data that flowed in; every node already declares what it
  adds (domains, layers, columns, a picture), so the rule is total, instant, and explains
  itself in the refusal.
- **The hierarchy follows the data flow, nesting only at branches.** Nesting every step would
  push a ten-node chain off the panel; a flat list would hide where a page forks.

## Files

- `nodelab_v2/linked_edit_dialog.py` — new: the three-answer dialog
- `nodelab_v2/linked_document.py` — edit modes, own structure (`_record`, `_mirror`,
  `structure_dict`, `check_structure`), sending to the master, per-page on/off
- `nodelab_v2/document.py` — `pass_through_reason`, `set_muted` refuses, `title_of`, the
  `claim_output_name` / `source_kind` hooks, `_settle_output_name`; `_bypass_muted(muted=)`
- `nodelab_v2/ops.py` — `sanitize_output_name`, `next_free_name`, `DEFAULT_OUTPUT_BASE`
- `nodelab_v2/page_recipes.py` — `unique_output_names` checks every page
  (`Workspace.dedupe_outputs`)
- `nodelab_v2/workspace.py` — workspace-wide names, `_dedupe_outputs` / `_follow_renames`,
  `source_kind`, `readers_of`, `page_outline` / `OutlineRow`, `structure` in the file
- `nodelab_v2/window.py` — `_topology_ok` asks, `ask_linked_edit`, `_scene_topology_gate`,
  `_needs_shape`, `show_node`, `set_node_muted`, page labels and menu entries, the
  renamed-name status message
- `nodelab_v2/scene.py` — `_needs_topology` asks through `topology_gate`, `toggle_muted`,
  the menu's Muted follows the rule
- `nodelab_v2/node_item.py` — the Source pill and menu; `nodelab_v2/inspector.py` — the
  combo's dots
- `nodelab_v2/pages_panel.py` — the outline, switches, node rows
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`, `codemap/concepts.md`, `codemap/curated.lock.json`, `codemap/gen/*`,
  `codemap/STATE.md`

## How to verify

`python run.py`, then *Example graph* on the welcome card. Pages panel (left): each page's
nodes under it, Outputs tinted with their names; untick Gaussian Blur under Image Refinement —
it is struck through and the page runs without it; Threshold has no switch (hover it). On the
Refinement page click the Page Input card's Source pill: a menu of Outputs with coloured
dots. Add a Page Output on Image Refinement and name it `raw` — it becomes `raw2`. Page menu
▸ *Duplicate as linked page*, then drop a Gamma onto the linked page: the dialog asks;
*Keep the change on this page* — the title says *modified*, the master is untouched; on
another linked copy *Add it to the master, switched off* — the master shows it struck through.

## Gates

- [x] selftest — 185 tests run; only `test_write_movie` fails (pre-existing: H.264 frame means
  non-monotone on this machine's encoder, unrelated). Three recipe tests in that run failed
  only because `workspace.py` was edited WHILE it ran (the module loaded before the edit, the
  one importing it after); the 31 page / linked / workspace / recipe / readiness / codemap
  tests were re-run on the final code and all pass, the six new ones among them
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 150 checks (new: SP1, PO1, LE1; G3, LK1,
  LK3, PP1 updated), on the final code
- [x] catalog snapshot — CATALOG IDENTICAL, 103 ops
- [x] codemap — CODEMAP CURRENT (CON-17, CON-18 blessed); synopsis CURRENT
- [x] native launch — the Windows platform with a COPY of the user's saved layout: the
  layout restores, the Pages panel's switch and node click, a linked page modified and one
  sending to the master, the dialog shown and answered, a clean close; the real
  `~/.nd2studios/layout.json` untouched
- [ ] sync check — run before the push
