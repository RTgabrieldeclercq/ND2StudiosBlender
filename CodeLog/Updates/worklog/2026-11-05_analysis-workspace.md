# The Analysis workspace: the page owns the centre, Figure widgets in Properties, Sources in Nodes, the HTML preview inside the tab

- **Date:** 2026-11-05
- **Author:** Gabriel Declercq
- **Branch:** analysis-workspace (on origin/Blender @5ff516d, with the HTML view commit of PR #11 rebased under it)
- **Base:** 5ff516d

## What changed

**Showing an Analysis page reshapes the window into a figure workspace.** The docked
Viewer, Playback and Channels panels step aside (hidden, not closed — a floating one stays
where the user put it), so the page has the whole centre; the **Nodes** dock turns to the
figure's **Sources** panel; the **Properties** dock turns to the new **Figure widgets**
palette over the selected item's properties form. Switching to any graph page brings
every dock back at the size it had (the same bookkeeping as the maximized canvas:
`_set_figure_mode` remembers the hidden docks and their heights; `_sync_figure_mode`
runs on every page / canvas switch, idempotent). View ▸ Reset layout, the window's close
(the layout saved is the docked one) and a canvas that closes all know about it. The
Figure Editor is built with `side_panels=False` in the window, so its Sources and
Properties panels are no longer in its own splitter but homed in the docks
(`figure_editor/docks.py`: `ModeStack` — one dock, two faces; `SourcesHost` and
`FigureSidePanel` — the ACTIVE canvas's editor's panels, the others parked, a closed
canvas's forgotten). The window keeps its `inspector` and `palette` objects as the first
face of each stack, so `shell.dock_of(window.palette)` and the probe's `win.inspector`
checks are unchanged.

**Figure widgets** (`figure_editor/widgets.py`): one button per thing the page can hold —
*Images*: Image, Montage, **Zoom-in** (a popout of the selected image's centre, a box on
the parent and leaders), Twin; *Annotate*: Text, Letter, Scale bar, Box, Line, Arrow,
Measure; *Keys*: Colour bar, Histogram, Legend; *Data*: Chart, Table, Data block. Each is
enabled when what it needs is there (an image selected; a published image or table) and
says why in its tooltip otherwise; Image / Montage / Chart / Table / Data block open a
menu when several sources qualify. They call the editor's existing entry points
(`add_image`, `add_popout`, `add_twin`, `add_item`, `add_chart` …), so a placed widget is
the item a drag, the Properties form or a template would have made. The palette follows
the editor's `selection_changed`, which the panel now also emits on bind and when the
sources catalog changes.

**The HTML export lives in the Analysis tab.** The figure toolbar has its own
**Interactive HTML** button (one click to the file dialog; still under Export ▾ too) and a
**Preview** toggle: the page's interactive export built into a temporary file at a preview
budget (`PREVIEW_PX` 512 per image, `PREVIEW_MB` 150) and shown **in the page's place** in a
QtWebEngine view (`figure_editor/preview.py`), with a bar saying when and at what budget
it was built, going amber when the page is edited under it (**↻ Rebuild**), **Open in
browser**, **Save as…** (the full-resolution export) and **✕ Back to the page**. Binding
the editor to another page drops the preview; shutdown removes the temporary file.
Without QtWebEngine (or with `NODELAB_WEBVIEW=0`) the file opens in the system browser
instead through the new host call `open_url`, and the page stays. `app.py` sets
`AA_ShareOpenGLContexts` before the QApplication, which Chromium needs.

The figure toolbar (and the preview's bar) are now widgets whose width the layout may
ignore: the canvas stack takes the figure panel's minimum width as the window centre's,
and with the two new buttons that minimum (1017 px) squeezed the side docks to their
minimums on a 1600 px window — the probe's palette click test found it (a role row whose
centre lay outside a 147 px viewport). Measured after: the centre's minimum is 284 px and
the Nodes dock gets its 330 again.

## Why

Asked for, after the HTML view landed: *the Analysis tab should contain the HTML export
directly within it; the entire viewer space needs to become the figure canvas space to
navigate it better; the parameter tab should be replaced with figure widgets (the zoom-in
component and many more).* Until now the figure page sat under the Viewer docks and
between its own Sources and Properties panels — a page of 85 × 65 mm in the smallest
rectangle of the window — while the Nodes palette on the left offered nodes a figure page
cannot take and the Properties dock sat empty. The Viewer is not needed on a figure page:
every image item is live, rendered from the same payloads with the Viewer's own
arithmetic.

Design calls: **the docks turn their other face rather than new docks appearing**, so the
layout the user saved is untouched and nothing new has to be placed — the figure's Sources
are its nodes, its widgets and properties are its parameters, in the same places. **The
Viewer is hidden, never closed**, with the maximize's bookkeeping, so it comes back exactly
where it was. **The preview is the export itself** (the same `export_html`, at a smaller
budget), not a second renderer, so what the preview shows is what a reader gets; it is
built on demand because a 50-frame page costs seconds, and marked stale rather than
rebuilt on every edit. **The widgets go through the editor's existing add_* calls** so
there is one way to make each item.

## Files

- `nodelab_v2/figure_editor/widgets.py` — **new**: the Figure widgets palette
- `nodelab_v2/figure_editor/docks.py` — **new**: `ModeStack`, `SourcesHost`, `FigureSidePanel`
- `nodelab_v2/figure_editor/preview.py` — **new**: `HtmlPreview` (QtWebEngine, optional)
- `nodelab_v2/figure_editor/panel.py` — `side_panels`, the body stack (page / preview), the Interactive HTML and Preview buttons, `set_preview`, the toolbar's ignorable width
- `nodelab_v2/window.py` — the stacks in the Nodes and Properties docks, `_sync_figure_mode` / `_set_figure_mode`, the hooks (page switch, canvas change, prune, open viewer, close, reset layout, theme), `_FigureHost.open_url`
- `nodelab_v2/app.py` — `AA_ShareOpenGLContexts`
- `nodelab_v2/figure_editor/__init__.py` — the three modules listed
- `scripts/_nodelab_v2_phase5_probe.py` — the workspace, the widgets, the preview fallback
- `MANUAL.md` — §2c (the Analysis workspace, Sources / Properties locations, item 9 Preview), §11 (the figure entry), §18 (a row); `codemap/workflows.md` — WF-11 step 1
- `codemap/gen/*`, `codemap/STATE.md` — **regenerated**

## How to verify

Open any workspace with a labelled result, show the Analysis page: the Viewer docks fold
away, the Nodes panel reads *Nodes · sources*, the Properties panel *Properties · figure*
with the widget buttons. Select an image, press **Zoom-in**, then **Preview**: the page
comes up live in the centre; edit a letter, the bar turns amber; ↻ Rebuild. Switch to the
Processing page: the Viewer is back.

```
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png
PYTHONPATH=. python -u preview_smoke.py fig.html      # the QtWebEngine path, in the scratchpad
```

## Gates

- [x] `PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (270 checks)
- [x] `PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (172 checks; the process then segfaults at `os._exit`, the known environmental exit crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 146 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (the three new modules, 50 symbols)
- [x] `python scripts/_sync_check.py` -> IN SYNC (rebased onto origin/Blender @5ff516d, PR #3's merge)
