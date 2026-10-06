# V4.00 step 11d — the Viewer's controls as panels; two-row page tabs; a Page Input's channels; the mode switch; fit to nodes

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a–11c

## What changed

Four reports from the user's second session with the step-11 build:

- **A Page Input offers the file's channels, like the Load card.** `page.input` joins
  `CHANNEL_TAP_OPS`: its card grows one `chK` output per channel, named from the upstream
  Output's descriptors — the Load card's captured names and colours — through any number of
  pages (`Workspace.input_channels`, installed on every page document as the new
  `GraphDocument.page_channels` hook; `channel_descriptors` and the inherited-name walk cross
  the boundary through it). A `chK` edge materializes into a `channel.select` tap like any
  other, and `compose` splices the tap onto the upstream Output with the Input itself. An
  unbound Input has no channel outputs.
- **"The image refinement is only working on the first channel" — not reproducible, now
  gated.** Every node allowed on a Refinement page was pulled through Split Positions → a
  Page Output → a Page Input on a two-channel fixture, and on the user's own 4-channel ND2
  in the GUI: every node that runs changes every channel. The new selftest pins it for the
  common refinement nodes, and MANUAL §18 says what to check (a channel output carries ONE
  channel; a channel switched off in the Channels panel is left out of the composite).
- **The Viewer's controls are panels of their own.** A `ViewerPanel` now builds two
  sections — PLAYBACK (M/T/Z strips, play, the overlay-source and iteration strips) and
  CHANNELS (Auto / Fit / Split / Overlays over one column per channel). The window moves
  both out of every Viewer it makes (`detach_controls`) into a single **Playback** and a
  single **Channels** dock (`nodelab_v2/viewer_controls.py`), which show the ACTIVE Viewer's
  sections (a linked Compare viewer shows its leader's strips) and name it once several are
  open. Default place: Playback under the Viewer, Channels beside it. The channel columns
  WRAP onto rows when the panel is narrow (`viewer.ChannelColumns`; the histogram minimum
  went from 160 to 100 px so four channels sit side by side under a normal Viewer), and so
  does the Auto / Fit / Split / Overlays row above them — otherwise its ~270 px would be the
  panel's floor and two columns could never stack.
  Maximized, the mini-map Viewer takes its sections back, compact (`attach_controls`), and the
  docked control panels hide until it goes home. The sections stay the Viewer's widgets, so
  every behaviour is the Viewer's code unchanged; they carry its stylesheet.
- **Page tabs are two rows, and a tab can be closed without losing the page.** The top row
  has a tab per page KIND (with a count when a kind holds several pages); under it, the
  pages of the kind shown as sub-tabs (drag to reorder, double-click to rename, right-click
  for the page menu, + for a new page of that kind, ✕ to close). A closed tab
  (`MainWindow.close_page_tab`, session state `_closed_pages`) leaves the tab row only: the
  Pages panel lists the page *tab closed* in italics and a click there opens it again, the
  kind tab reopens the last page shown when all of a kind's tabs are closed, the page menu
  has *Close tab*, and the last open tab has no ✕.
- **Panels grouped as tabs carry their tabs on top** (`setTabPosition(…, North)`; Qt puts
  dock tab groups' tabs along the bottom by default).
- **A Normal | Troubleshooting switch on the menu bar** (`nodelab_v2/mode_switch.py`, the
  menu bar's top-right corner): picking a segment sets the troubleshooting scope (the F9
  action stays the one source of truth); F9, Run ▸ and a programmatic change move the switch;
  Troubleshooting is lit in the same amber as the canvas frame and the SOLO chip.
- **A fit-to-nodes button beside the canvas's maximize button** (`HudButton` kind `"fit"`,
  painted like the maximize glyph): what `Home` does, one click away.
- **Fixes found on the way.** (a) *The app closed by itself on launch* with the user's saved
  layout: restoring a step-11c layout re-added the new panels after `restoreState` had laid
  them out without knowing them — an access violation inside Qt on the Windows platform (the
  offscreen platform every gate runs on survived it). The unknown docks now leave the window
  BEFORE the restore and are placed after it. (b) *"Could not parse stylesheet of object
  QLabel"*: the LabLink tab's muted labels wrote a QColor's Python repr into their
  stylesheet (`lablink/tuning.py _dim`); `.name()` now. (c) *Home / Fit left a far-away card
  off screen*: the view scrolls only inside the scene rect, which starts as a fixed area round
  the origin; `fit_all` grows it to the nodes first.
- Shell: a `PanelSpec` can name `split_from`/`split` for its default place, and
  `restore_layout` puts a dock the saved layout never heard of in that place
  (`DockShell.place_default`) — the user's step-11a layout keeps its arrangement and gains
  the two new panels under the Viewer, with no layout-format bump.
- **Tests**: `test_page_input_channel_taps`, `test_refine_ops_all_channels`; probe CT1 (Page
  Input channel outputs, ch1 → Gaussian pulls one channel), DT1 (tabs on top), MS1 (the
  mode switch), FB1 (fit to nodes), VC2 (a pre-11d layout restores with the new panels), VC1 (the
  control panels: placement, follow the active Viewer, wrap, theme, maximize round trip,
  default placement), PT1 rewritten for two rows, PT2 (closing tabs); the Compare, SH1 and
  CH1 assertions moved with the controls. **Docs**: MANUAL §2 (window, diagram, panels,
  pages, the mode switch, the fit button), §7 (the controls as panels, maximize),
  troubleshooting mode, the shortcuts table, the Page Input row, four §18 rows; codemap
  CON-17, CON-20.

## Why

The user, testing the step-11c build: (1) "when I pass image data into the image refinement
from a split positions node, the image refinement is only working on the first channel …
any image refinement node should be applied to all channels. Also on the page input, the
channels should also be options just as if it was the original IO node"; (2) "on the viewer,
there is the M,T,Z and play buttons, as well as the histograms and color channel options. I
want these sections to be their own windows that can be placed wherever the user wants. For
the channel and histogram, we should also stack these if needed"; (3) "there should be sub
tabs for each node graph type. If the tab is closed, it isn't gone, you can still access it
from the pages window"; (4) "when grouping windows, the tabs for each window should be on the
top"; and, mid-step, "on the top bar, I want a toggle button for the mode (normal,
troubleshooting)" and "next to the maximize button on the node graph I want a fit to nodes
button"; and the build closing by itself on launch, its terminal showing only "Could not
parse stylesheet of object QLabel".

On (1): the engine already applies every refinement node to every channel — swept over the
whole Refinement palette and on the real ND2 — so the change is the part that was missing
(channel choice on the Page Input) plus a gate that keeps the property true, rather than a
fix for a defect that could not be found. On (2): ONE Playback and ONE Channels panel that
follow the active Viewer, rather than a pair per Viewer — the same rule the Spreadsheet
follows, and it keeps the panel list from growing with every Compare pane. Re-using the
Viewer's own widgets (moved, not rebuilt) keeps every existing behaviour and test of those
controls valid. Placing unknown docks at their default spot on restore, rather than bumping
the layout format, keeps the arrangement the user just made under step 11a.

## Files

- `nodelab_v2/viewer_controls.py` — new: the Playback and Channels panels
- `nodelab_v2/mode_switch.py` — new: the menu bar's Normal | Troubleshooting switch
- `nodelab_v2/viewer.py` — the two control sections, `ChannelColumns`, `detach_controls` /
  `attach_controls`, compact mode leaves detached controls whole, histogram minimum 100 px
- `nodelab_v2/window.py` — the control panels (specs, `_home_controls`,
  `_sync_viewer_controls`, maximize, default sizes, `VIEWER_SHARE` 0.30, `CONTROLS_H`), page
  tab kinds / closed tabs (`page_tab_kinds`, `show_kind`, `reorder_pages`, `close_page_tab`,
  *Close tab*), the mode switch, File ▸ Open resets the per-page session state
- `nodelab_v2/canvas.py` — `PageTabs` rewritten: kind row + closable sub-tabs
- `nodelab_v2/pages_panel.py` — closed pages listed *tab closed*
- `nodelab_v2/shell.py` — tabs on top, `PanelSpec.split_from`/`split`, `place_default`,
  unknown docks out of the window during `restoreState`
- `nodelab_v2/document.py` — `page.input` in `CHANNEL_TAP_OPS`, the `page_channels` hook
- `nodelab_v2/workspace.py` — `input_channels`, the hook installed / removed
- `nodelab_v2/scene.py`, `nodelab_v2/minimap.py` — the fit-to-nodes button (`HudButton`
  kind `"fit"`), `fit_all` grows the scene rect
- `nodelab_v2/lablink/tuning.py` — `_dim` writes a colour name, not a QColor repr
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`, `codemap/concepts.md`, `codemap/curated.lock.json`, `codemap/gen/*`,
  `codemap/STATE.md`

## How to verify

`python run.py`. Under the Viewer: **Playback** (M/T/Z, ▶) and **Channels** (a column per
channel) — drag either anywhere; narrow Channels and its columns stack. Above the canvas: a
row of page kinds and, under it, that kind's pages — ✕ a sub-tab, the page is still in the
Pages panel (*tab closed*), click it there to bring the tab back. Load a multi-channel image,
open Image Refinement: the Page Input card has `0 · <channel>` outputs like the Load card.
Top right of the menu bar: *Normal | Troubleshooting* (= F9). Beside the canvas's ⛶: fit to
nodes. Grouped panels (Properties / Spreadsheet / LabLink) show their tabs on top.

## Gates

- [x] selftest — 179 tests run; only `test_write_movie` fails (pre-existing: H.264 frame means
  non-monotone on this machine's encoder, unrelated). After the last edits (mode switch, fit,
  restore order, LabLink label) `test_codemap`, `test_layout_store` and the two new tests were
  re-run and pass; the rest touch none of those files
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 147 checks (new: CT1, DT1, MS1, FB1, VC1, VC2,
  PT2; PT1 rewritten). The process exit after the PASSED line is the known teardown fault
  (worker threads alive at `os._exit`), not a failed check
- [x] catalog snapshot — CATALOG IDENTICAL, 103 ops
- [x] codemap — CODEMAP CURRENT (CON-17, CON-20 blessed); synopsis CURRENT
- [x] native launch — `run.py`'s window on the Windows platform with a COPY of the user's
  saved layout, five runs under `-X faulthandler`: no crash, no stylesheet warning with the
  LabLink tab raised, a clean close (the crash fix is native-only; offscreen never crashed)
- [ ] sync check — run before the push
