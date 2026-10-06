# V4 step6 linked

- **Date:** 2026-10-05
- **Author:** hyper
- **Branch:** v4-step6-linked
- **Base:** e1ae0e28

## What changed

V4.00 step 6 of `CodeLog/ClaudesPlan/V4.00_workspaces.md`: **linked (dependent) pages.**

- **The model.** `nodelab_v2/linked_document.py` `LinkedDocument(GraphDocument)` mirrors a
  master page's document IN PLACE — every node keeps its `NodeRecord` (and its `params` /
  `modes` dicts) while the master keeps that node with the same type, so the canvas updates
  cards instead of rebuilding them — and keeps per-node **overrides**
  `{node: {"params": {name: value | UNSET}, "modes": {...}}}`. An edit on the linked page
  (`touch`) records only what differs from the master; a parameter the master pins and the
  page resets to its default is the plain-JSON marker `UNSET`. Pins (`__locked__`) are not
  overrides: they are derived — the master's, minus what the page resets, plus every value
  it sets. Overrides of a node the master deletes or re-types go with it. Every structural
  edit (`add_node`, `remove_node`, `connect`, `disconnect`, frames, mute, zones, groups,
  Iterate targets, `clear`, loads) raises `LinkedPageError(TOPOLOGY_HINT)`; `can_connect` says
  no with the hint; moving and folding a card act on the master. `make_unique()` returns a
  plain document holding the page's graph and values.
- **The workspace.** `Workspace.duplicate_page(pid, dependent=True)` makes `<name> (linked)`
  right after it (a page linked to a linked page links to that page's master and starts from
  its overrides); `dependents_of`, `make_unique(pid)` (swaps the page's document);
  `remove_page` refuses a master that still has linked pages. A file stores a linked page as
  `{"master", "overrides"}` without a `"graph"`; `load_dict` loads plain pages first, then the
  linked pages over their masters, and checks each master is a plain page of the file.
  `File ▸ New` on a linked page starts a fresh page rather than emptying the linked one.
- **The window.** The switcher's menu adds *Duplicate as linked page*, marks linked pages
  `· linked`, and on a linked page offers *Go to master page* and *Make unique*. Deleting a
  master asks (naming its linked pages), then makes them unique. Every structural gesture is
  refused before it starts, with the hint in the status bar: the window's actions
  (`_needs_topology` — palette drop/double-click, Ready-to-run suggestions, region nodes,
  wrap/group/ungroup/frame, flatten, file and desktop loads, the example graph) and the
  canvas's (wire drags, delete, dissolve, mute, reroute, fan-out; its context menu greys
  those entries out with the hint as tooltip). A `LinkedPageError` escaping any other slot
  reaches the status bar through a `sys.excepthook` backstop instead of a traceback. When a
  page's document is swapped (Make unique), Properties re-reads the new canvas's selection.
- **Properties.** On a linked page a **Linked page** banner says `Linked to "X" · N overrides`
  (and how many on this node) with **Go to master** (shows the master, this node selected)
  and **Make unique**. An overridden parameter or mode row carries an accent bar and the
  master's value in its tooltip; right-click → *Reset to master*.
- **Cards follow.** `GraphDocument.set_pos` now reports a real position change to GUI-only
  move listeners (`on_moved` / `off_moved`; no revision, no run disturbed). A scene puts its
  card where the record says, and a linked document follows its master's moves, so a card
  dragged on either page moves on the other, live.

Tests: selftest `test_linked_document_mirrors_master` (in-place mirror, the touched set,
refusals, moves both ways without a revision), `test_linked_overrides_roundtrip` (overrides,
`UNSET`, derived pins, reset, a link to a linked page, the file, page order, a master with
dependents), `test_linked_make_unique`, `test_linked_page_shares_prefix` (one memo: an
un-overridden linked page is a full cache hit; overriding one node re-runs that node alone);
`test_workspace_format_v3` updated (linked pages load; a bad master is refused). GUI probe:
LK1 (duplicate, the menu, refusals incl. the context menu and the backstop, the banner, an
override marked and reset, a master edit and a card move reaching it, a pull) and LK2 (save
and reopen, Make unique from the banner, deleting a master).

**Review (an adversarial pass: three lenses, every finding reproduced, then verified).**
Fourteen confirmed, all fixed here; regression: selftest `test_linked_page_state`, probe LK3.

- *A Dock's checkpoint was mirrored*: a linked page served the master's bake, a Bake there
  overwrote the master's folder, a Hold on the master broke the linked page's pulls, and Save
  As did not re-anchor a linked page's own folder. A Dock's `store`, bake record and `state`
  are now the page's own (`DOCK_LOCAL_*`, kept under an override's `"local"` entry — not an
  override, never counted or marked); `rebase_path` records the re-anchored folder; a new
  linked page has the workspace's path.
- *File seeds were shared by reference*: a linked page whose Load reads another file
  re-described the master and every page reading it. A linked page has its own seeds; a node
  it does not override takes the master's.
- *Envelopes were all-unknown* on a new linked page and after Make unique until the next edit
  (propagation ran before the Page Input hook was installed; loading cleared the copied seeds).
  Both re-propagate once attached; Make unique carries the seeds, path and held set.
- *File ▸ New on a linked page* left `active` naming the deleted page.
- *A file with malformed overrides* overwrote the canvas page, left a listener that made later
  edits raise, and stopped the restored linked pages following. Overrides are checked before
  anything changes (`check_overrides`); a linked document subscribes only after a successful
  mirror and re-attaches on a rolled-back load; the canvas page's content is restored.
- *Edit ▸ Dissolve* was bound at start-up to the FIRST page's scene (a step-5 regression), so on
  a linked page it dissolved the master's selected node. *Edit ▸ Delete* reported a deletion
  that was refused. *LabLink's load into graph* on a linked page showed a read error, then a
  false success. *A region request* was refused even with the Draw Regions node already wired.
  *Make unique* dropped the selection. All fixed (guards moved to the right methods; the
  selection is re-applied on the rebuilt scene).
- Found while checking the fixes: an override equal to a later master value was dropped by an
  unrelated edit, so the next master change overwrote it. An override now stays one until
  *Reset to master*.

## Why

V4's pages exist to tune one workflow per position or condition without losing the earlier
attempts. Copying a page forks it: an improvement to the chain then has to be made on every
copy. A linked page shares the master's graph and owns only its values, so the chain is
edited once and every variant follows, and the shared memo re-uses everything up to the first
node a variant changes. Mirroring in place rather than rebuilding the document on each master
edit keeps record identity, which is what lets the canvas update cards instead of tearing
them down on every keystroke. Structural edits are refused rather than silently applied to
the master: an edit made on the page in front of you that changes another page is a surprise,
while "edit the master, or Make unique" is a choice the user makes knowingly.

## Files

- `nodelab_v2/linked_document.py` — new: `LinkedDocument`, `LinkedPageError`, `TOPOLOGY_HINT`,
  `UNSET`, `check_overrides`, `DOCK_LOCAL_PARAMS` / `DOCK_LOCAL_MODES`
- `nodelab_v2/workspace.py` — linked duplicate, `dependents_of`, `make_unique`, the file
  format's linked records, `reset`, the master-name hook
- `nodelab_v2/document.py` — `on_moved` / `off_moved`; `set_pos` reports real moves
- `nodelab_v2/scene.py` — structural gestures guarded (`_needs_topology`,
  `topology_refused`), the context menu greyed, cards follow moves
- `nodelab_v2/window.py` — page menu entries, `duplicate_page(linked=)`, `make_unique`,
  delete-master flow, action guards, the `LinkedPageError` backstop, Properties re-pointed on
  a document swap
- `nodelab_v2/inspector.py` — the Linked page banner, override marks, Reset to master
- `nodegraph/selftest.py` — the linked tests incl. `test_linked_page_state` (hotspot file:
  appended at the tail of `main()`), `test_workspace_format_v3` updated
- `scripts/_nodelab_v2_phase5_probe.py` — LK1, LK2, LK3
- `MANUAL.md` (§2 Linked pages), `CodeLog/ClaudesPlan/V4.00_beta_tests.md` (Step 6 block),
  `CodeLog/ClaudesPlan/V4.00_workspaces.md` (delivery row), `CodeLog/Updates/CHANGELOG.md`
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

`python run.py` → Example graph → the switcher (top-left) → *Duplicate as linked page* → click
the Threshold card, change its threshold: the row gets a bar and Properties says `1 override`;
right-click the row → Reset to master. Try to drop a node from the palette or press Delete on
a card: refused, with the hint in the status bar. Switcher → *Go to master page*, change the
Gaussian's sigma, come back: the linked page has it. Properties → *Make unique*. The beta
block B6.1–B6.11 lists the rest.

## Gates

- [x] selftest: all 152 registered tests run (continuing past failures); one fails,
  `test_write_movie` — the pre-existing codec-rounding failure every V4 step has carried. The
  linked tests (`test_linked_*`, `test_linked_page_state`) and `test_workspace_format_v3` pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
  PROBES PASSED (119 checks, LK1-LK3 among them)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 95 ops (no node changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> 0 behind origin/Blender (e1ae0e2); stacked on
  `v4-step5-pages`
