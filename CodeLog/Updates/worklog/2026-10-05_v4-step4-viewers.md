# V4 step4 viewers

- **Date:** 2026-10-05
- **Author:** hyper
- **Branch:** v4-step4-viewers
- **Base:** e1ae0e28

## What changed

V4.00 step 4 of `CodeLog/ClaudesPlan/V4.00_workspaces.md`: **the Viewer is a dock, and there
can be several.** Each Viewer is a `viewer:<n>` panel of the step-3 `DockShell` (a multi kind),
docked above the canvas by default — the canvas (`GraphView`) is now the window's central widget,
and the old Viewer/canvas `QSplitter`, the compare `QSplitter` and its `_CompareBox` frame are
gone. **+** on a Viewer's title bar, **View ▸ New ▸ Viewer** or Compare opens another; any one
pops out, tabs or closes like every panel, the last one included (the next pull opens a fresh
one), and the set of viewers survives a restart with the layout.

- **Binding-based routing.** A viewer is BOUND to the node it was asked to show
  (`ViewerPanel.binding = (page, node)`; its title bar names it, `Viewer · n3 · Gaussian Blur`).
  A result lands in every viewer bound to its node (`MainWindow._viewers_for`); failing that, in
  the viewer that last asked for it (re-bound since — the first result to land stays viewable
  while the next computes, as before); failing that, in the active viewer, which shows whatever
  finishes (V2's primary pane).
- **Several viewers on one node.** Every plane delivery now says which request it answers:
  `EngineRunner.finished` / `plane_ready` gained a trailing `request = (coords, channels)` (a
  slot that takes fewer arguments is unaffected). When two viewers show the same node, a frame
  goes only to the one whose cursor and channel set asked for it
  (`runner.request_answers`); the other asks for its own.
- **The active viewer.** The one the user last worked in — a press on its image
  (`ViewerPanel.activated`) or title bar (`PanelDock.pressed`), focus inside it, its strips or
  region box — carries the accent. Pulls, click-to-preview, picks and drawing, the
  troubleshooting scope (frame picks and region box), the spreadsheet and the sweep table
  follow it; each viewer keeps its own iteration strip, LUTs, channels and play gate. Handlers
  take a trailing `viewer=` (wired with `functools.partial`; the window keeps each viewer's slots
  in `_window_slots`, see `_viewer_slot`), so a call without one means the active viewer.
- **Click a visual node.** Selecting a card whose op is a visual output
  (`ops.VISUAL_OUTPUT_PREFIXES = ("view.", "plot.")`) shows it in the active viewer at once, even
  with click-to-preview off — like a preview it never starts an ingest or queues.
- **Compare (V2.28) is a dock.** F8 opens the active viewer's Compare viewer beside it (or
  re-targets the existing one); `_links` maps it to its leader and the cursor link is re-derived
  from the payloads' M/T/Z after every delivery, as before; its title bar says linked / own
  cursor. Shift+F8 closes it; deleting its node closes it.
- **Maximize** hides the docked viewers (floating ones stay) and re-homes the active one into the
  mini-map; restoring puts it back into its dock and brings the hidden docks back at their old
  heights. The mini-map viewer's dock cannot be closed meanwhile (`allow_close`).
- Compatibility names for single-viewer code and the probe: `viewer` (the active viewer, created
  when every one was closed), `viewer2` (the newest Compare viewer), `_viewed` (getter/setter),
  `_viewed2`.
- `glview.GLImageView` counts its GL contexts (`_context_gen`) and asks for a fresh detail patch
  after each rebuild; `scripts/_nodelab_v2_gl_float_probe.py` (desktop only) floats a GL viewer
  and docks it back, asserting a new context each time and a painted picture after.
- The side columns now own the TOP corners too (`DockShell`), so a top panel spans the canvas,
  not the full width.

**Found while testing, fixed here.** The viewer's scale bar ran past the right edge of any image
narrower than 140 px (a fixed 14 px inset plus a 90%-wide bar). The dock's title bar made the
probe's tiled Viewer node just small enough to show it; the inset now shrinks with the image.
View ▸ Reset layout hid every Viewer, the ones holding a picture included (the default layout
keeps the Viewer hidden only because a fresh install has nothing to show); those now stay on
screen at the Viewer's share of the column.

**Found by the review, fixed here.** An adversarial review in three lenses (result routing,
dock lifecycle, active-viewer semantics) wrote a reproduction for each suspicion. The account's
session limit stopped the three reviewers before they could report, so their twelve scripts
were re-run by hand; every finding below was reproduced, fixed, and re-run clean.

1. **Two viewers of one node under F9 pulled forever** — 151 pulls in 6 s. Under the solo
   scope every T is its own pull: a delivery to one viewer blanked the other, which re-requested
   its own T — a pull — whose delivery did the same to the first. A viewer that did not ask for
   a frame now keeps its own on screen and asks again WITHOUT pulling
   (`EngineRunner.request_plane(pull=False)`).
2. **A result re-served while the runner was busy blanked every other viewer of that node**
   ("no image on this output") for good, since nothing asked for their frames again. A viewer
   already showing the node keeps its frame (`ViewerPanel.show_result(keep_image=True)`,
   redrawn so its status line is right again) and asks for its own, pull-free.
3. **Clicking a viewer under F9 cancelled the scoped pull in flight.** Activation pushed that
   viewer's frame picks to the runner, and a changed scope invalidates. Activation now moves
   only the scope's indicators (region box, chip); the next pull takes the active viewer's.
4. **Quitting with the canvas maximized** saved every viewer dock hidden and empty (the
   mini-map held the viewer), so they came back as hidden 30 px docks. The window restores
   the docked layout before saving it.
5. **View ▸ Panels could open the mini-map viewer's dock as an empty panel.** Its entry is
   disabled while the viewer lives in the mini-map.
6. **A linked Compare viewer moved into the mini-map** kept its cursor strips hidden — its
   leader, which drives them, was hidden — and deleting its node while maximized left it bound
   to a node that no longer exists (its dock cannot close then). It now scrubs alone in the
   mini-map, and when its node goes it is unlinked and unbound instead.
7. **F8 on a card the selection had just previewed compared it beside itself**: selecting a
   Viewer node (or any card, with click-to-preview on) had already pulled it into the active
   viewer. F8 now puts the active viewer back on what it showed before the click.

Refuted: a double-click on a Viewer-node card does not pull twice (one start, one delivery).

Known and not fixed: the runner's decode lane keeps ONE pending request (the latest wins, which
is what keeps scrubbing responsive). With three or more viewers of one node all scrubbed to cold
frames at once, one of them can miss a frame until its cursor moves again.

Tests: selftest `test_viewer_routing` (the request a delivery carries, from the warm and the
decode path; `request_answers`; a pull-free request never pulls; `is_visual_output`). GUI
probe: E9 (maximize; the Panels entry disabled) and Compare rewritten onto docks; SH1 knows the
viewer kind; SH3 restores a second, floating Viewer; new VW1–VW7 — View ▸ New ▸ Viewer, a
click on a Viewer node, F8 after that click, activation by image or title press with the sheet
and a pick following it, two viewers on one node at different z (a re-served result keeps both
frames; F9 at different T does not loop), closing every viewer then pulling, floating a viewer,
quitting while maximized.

## Why

The V4 plan puts several viewers and canvases on screen at once ("every panel can pop out or
dock; multiple viewers"), so the Viewer had to stop being one fixed pane in a splitter. Binding
rather than "the primary shows everything": with N viewers, "where does this result go" needs an
answer per viewer, and the one the user can predict is "where I asked for it". The request echo
was not optional: two viewers on one node at different frames — the obvious way to compare two
time points — otherwise each drew the other's frames, because the runner's deliveries named only
the node. A coordinate echo on the existing signals was chosen over a requester token: it is
stateless (no table of in-flight requests to keep in step with the decode lane's latest-wins
drops) and old slots keep working. The active-viewer rule keeps every single-viewer behaviour
(picks, F9, the sheet) unambiguous with several viewers open.

## Files

- `nodelab_v2/window.py` — viewers as docks, routing, Compare, maximize, active viewer
- `nodelab_v2/viewer.py` — `binding`/`bind`/`unbind`/`shows`/`showing`, `activated`; scale-bar inset
- `nodelab_v2/runner.py` — `finished`/`plane_ready` carry the request; `request_answers`
- `nodelab_v2/shell.py` — top corners; `PanelDock.pressed` (a title-bar press activates)
- `nodelab_v2/glview.py` — `_context_gen`; detail re-request after a context rebuild
- `nodelab_v2/ops.py` — `VISUAL_OUTPUT_PREFIXES`, `is_visual_output`
- `nodegraph/selftest.py` — `test_viewer_routing` (hotspot file: appended at the tail of `main()`)
- `scripts/_nodelab_v2_phase5_probe.py` — E9, Compare, SH1, VW1–VW6
- `scripts/_nodelab_v2_gl_float_probe.py` — new, desktop only
- `MANUAL.md` — §2 Several Viewers, §7 Compare and the mini-map, shortcuts
- `CodeLog/ClaudesPlan/V4.00_beta_tests.md`, `CodeLog/ClaudesPlan/V4.00_workspaces.md`,
  `CodeLog/Updates/CHANGELOG.md`
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

`python run.py` → Example graph → double-click a card (the Viewer opens above the canvas) →
**+** on the Viewer's title bar → double-click another card: it shows in the new Viewer, the
first keeps its image → click the first Viewer's image, then select the Viewer node: it shows in
the first → F8 on another card: a Compare viewer docks beside the active one → Ctrl+Space and
back. The beta block B4.1–B4.12 lists the rest.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: every registered test run one by one, continuing past failures: 146 run, 145 pass (the new `test_viewer_routing` among them), `test_write_movie` fails — the pre-existing codec-rounding failure, identical on `origin/Blender`
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (101 checks; VW1–VW7 and SH1–SH4 included). The last run then exited 139 inside the final `os._exit(0)` — the Windows teardown race recorded in the step 2 worklog, after every check had passed; the run before exited 0.
- [x] `scripts/_nodelab_v2_gl_float_probe.py` (desktop, real GPU) -> ALL GL FLOAT PROBES PASSED: OpenGL 4.6, GL context #1 docked → #2 floating → #3 docked back, the picture painted each time
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 95 ops (no catalog change)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (no curated entry flagged; the cards describe no viewer internals)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT — 95 nodes
- [x] `python scripts/_worklog.py check` -> WORKLOG OK
- [x] `python scripts/_sync_check.py` -> feature branch, stacked on step 3, 0 behind origin/Blender
