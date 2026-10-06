# V4 step3 dock shell

- **Date:** 2026-10-05
- **Author:** hyper
- **Branch:** v4-step3-dock-shell
- **Base:** e1ae0e28

## What changed

V4.00 step 3 of `CodeLog/ClaudesPlan/V4.00_workspaces.md`: **every side panel is a dock that pops
out, docks back, closes and comes back, and the arrangement survives a restart.** The Nodes,
Properties, Spreadsheet, LabLink, Console and Movie Editor panels are now `PanelDock`s of a
`DockShell` (`nodelab_v2/shell.py`). Each is movable, floatable and closable, named
`"<kind>:<index>"` (what Qt's `saveState` keys positions by), and has its own `PanelTitleBar`:
the kind glyph, the title (elided, and later naming what an instance shows), `⇱`/`⇲` to pop out
or dock back (a double-click on the title does the same), and `✕`. A single panel's close
only hides it. **View ▸ Panels** ticks every panel; the Console keeps its Ctrl+` action as its
entry. **View ▸ Reset layout** puts everything back as a fresh install has it. The machinery the
next steps need is in place but not yet used by any shipped panel: multi-instance kinds
(`"viewer:1"`, `+` on the title bar, the lowest free index reused, **View ▸ New**, which stays
hidden until such a kind exists), the active instance per kind (accented title, `activated`
signals, focus-driven), and a close veto (`DockShell.allow_close`, for step 5's "never the last
canvas"). `nodelab_v2/layout_store.py` (Qt-free) keeps the layout in
`~/.nd2studios/layout.json`: Qt's geometry and state blobs plus the list of panels, so extra
instances are re-created before Qt is handed the state that places them. A damaged, foreign or
other-generation file is ignored, never fatal, and writes are atomic.
`MainWindow(persist_layout=None)` restores at start and saves on close unless `NODELAB_LAYOUT=0`;
`NODELAB_LAYOUT_FILE` points elsewhere. A floating panel restored off every screen is moved
back onto one. The Viewer/canvas splitter stays the central widget until steps 4–5 make those
panels too.

**Found before commit, fixed here.** An adversarial review (Qt docking semantics, layout
persistence) and reproductions of each finding turned up:

1. **`+` beside a panel in a tab group, or a default layout with a tabbed multi kind, lost
   panels off screen.** Qt's `splitDockWidget` on a tabbed dock takes both docks out of the
   group. A new instance now joins the group (`DockShell._place_beside`) and comes to the
   front. Instances line up beside the previous one, in index order.
2. **The title bar's border and the active-instance accent never painted.** Stylesheets set
   `WA_StyledBackground` only on a plain `QWidget`, so the title bar now sets it itself; SH2
   asserts the painted pixel.
3. **A close vetoed from View ▸ Panels left the panel's tick off.** It is put back on.
4. **The launcher discarded the restored window.** `nodelab_v2/app.py` resized and maximized
   unconditionally; it now shows the saved geometry and window state when a layout was
   restored (`MainWindow._layout_restored`).
5. **Layout-file robustness.** `load_layout` never raises: a non-list `docks` field means no
   docks, and over-nested JSON means no layout. With `quarantine` (the window uses it) an
   existing but unusable file is set aside as `layout.json.rejected`, so a newer build's
   layout survives a session of an older checkout. Saves use a per-process temporary name and
   remove it after a failed write.
6. **Tests that did not prove their claims.** SH3 now saves a size a fresh window cannot have;
   the selftest simulates a write that fails midway; the probes force `NODELAB_LAYOUT=0`
   instead of `setdefault`.

Known and accepted: each **View ▸ Reset layout** leaves Qt one more internal tab bar (Qt keeps
the old ones). The cost is a few widgets per reset, and resets are rare.

Tests: selftest `test_layout_store` (names, environment switches, the atomic round trip, and
every way a file can be untrustworthy). Probe sections: `SH1` (every panel is a shell dock;
pop-out, dock-back, close and reopen from the title bar and from Panels; theme restyles a
floating panel; the Console action; Browse nodes reopens a closed palette); `SH2` (a
probe-only multi kind: `+`, index reuse, the active instance, focus activation, destroy on
close, the veto); `SH3`/`SH4` (two more windows on a TEMP layout file: floating, shown and
closed panels and the window size come back; Reset layout; a damaged file opens on the default
layout and is replaced on close). The probe and the window-building scripts set
`NODELAB_LAYOUT=0`, so no test run reads or writes a user's layout. MANUAL §2 gains a Panels
paragraph; the Step 3 beta block lists only this step's new behaviour.

## Why

The V4 shell has to put several viewers and several canvases side by side, on one screen or
across two, and that is not possible while panels are fixed in place. Making every panel a dock
first, with nothing else changing, lets the next two steps add instances to a shell that is
already tested, instead of building docking and multi-instance routing in the same change. Three
choices differ from the plan's text, each to cut risk:

- **The shell is an object the window owns, not a QMainWindow mixin.** PySide's
  multiple-inheritance rules make a QObject mixin fragile, and composition is testable on a bare
  QMainWindow; the offscreen smoke against one ran before the window was touched.
- **No `GroupedDragging`.** It turns a tab group into one floating window and complicates
  `saveState`. Dragging one tab out now floats just that panel, which is the more common
  expectation.
- **The layout is restored at the end of `__init__`, not in the first `showEvent`.** Every
  panel exists by then, and a restore before the first show is what Qt documents. It also keeps
  the probe deterministic.

Layout memory is opt-out through one environment variable because a test that reads a user's
floating panels, or overwrites them, is the kind of cross-talk that wastes an afternoon.

## Files

- `nodelab_v2/shell.py` (new) — `PanelSpec`, `PanelTitleBar`, `PanelDock`, `DockShell`
- `nodelab_v2/layout_store.py` (new, Qt-free) — the layout file
- `nodelab_v2/window.py` — panels through the shell (`_panel_specs`), View ▸ Panels / New / Reset layout, restore in `__init__` and save in `closeEvent`, `focus_palette` via the shell, menu actions resolve the canvas and viewer at trigger time
- `nodelab_v2/app.py` — the launcher shows a restored window as saved
- `nodegraph/selftest.py` — `test_layout_store` (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — SH1–SH4; `NODELAB_LAYOUT=0`
- `scripts/_hotreload_probe.py`, `scripts/_dock_hold_probe.py`, `scripts/_nodelab_v2_shot.py` — `NODELAB_LAYOUT=0`
- `MANUAL.md` — §2 Panels paragraph
- `CodeLog/ClaudesPlan/V4.00_beta_tests.md` — Step 3 block (new features only)
- `CodeLog/ClaudesPlan/V4.00_workspaces.md` — delivery table row
- `CodeLog/Updates/CHANGELOG.md` — Step 3 line
- `codemap/gen/*`, `codemap/STATE.md` (regenerated: two new modules)

## How to verify

- `PYTHONUTF8=1 python -B -m nodegraph.selftest` — the `[ok] layout store: …` line.
- `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` — `SH1`…`SH4`.
- By hand: the Step 3 block of `CodeLog/ClaudesPlan/V4.00_beta_tests.md` (B3.1–B3.13): `⇱` on
  Spreadsheet pops it out and `⇲` docks it back; `✕` on LabLink then View ▸ Panels brings it
  back; pop out Spreadsheet, show the Console, close LabLink, quit, relaunch → the same
  arrangement; View ▸ Reset layout restores the default.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: all 145 registered tests run one by one, continuing past failures: 144 pass, `test_write_movie` fails (the pre-existing codec-rounding failure, identical on `origin/Blender`). `test_codemap` passes in a fresh LF worktree — the line-ending fix in the step 2 follow-up at work.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (93 checks, SH1–SH4 included); the probe's final `os._exit` still hits the pre-existing Windows teardown race (see the step 2 worklog), exit code 0 this run
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 95 ops (no catalog change)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT — 95 nodes
- [x] `python scripts/_worklog.py check` -> WORKLOG OK
- [x] `python scripts/_sync_check.py` -> feature branch, stacked on step 2, 0 behind origin/Blender

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
