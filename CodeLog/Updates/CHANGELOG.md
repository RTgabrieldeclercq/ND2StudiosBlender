# Changelog — NodeLab

All notable changes to the standalone NodeLab GUI. The vendored `nd2studios/`
backend is a copy and is **not** modified here (see `pipeline_kit` parity).

## [4.0.0-dev] — Workspaces (ND2Studios V4.00) — opened 2026-10-05

The V4 generation turns the single canvas into a **workspace** of typed node-graph pages
(Image Input → Image Refinement → Image Processing → Analysis, plus Free for legacy graphs)
with named outputs flowing between pages, linked pages that share a master's nodes with
their own parameter values, every panel poppable into its own window with several
instances, and an analysis toolkit of plot and table nodes. Design record and step list:
`CodeLog/ClaudesPlan/V4.00_workspaces.md`. Each step lands as its own PR and adds a line here.

### Step 0 — version and plan record (2026-10-05)
- `nodelab_v2/version.py`: `__version__ = "4.0.0"`, `APP_NAME`, `VERSION_LABEL`, `PRODUCT` —
  the one place the number is written. Window title and Help → Capabilities read it; the
  LabLink `hello` reports `SOFTWARE_NAME`/`SOFTWARE_VERSION` from it (was a stale `"2.21"`).
- Package name `nodelab_v2` is unchanged by decision: the version is a label, not a path.

### Step 1 — workspace model, file format 3.0, page kinds (2026-10-05)
- `nodelab_v2/workspace.py`: a saved file is a **workspace** of typed pages (`input` →
  `refine` → `process` → `analyze`, plus `free`); `format_version` 3.0 with `workspace:
  {active, next_page_seq, pages}`; a 2.0 single-graph file opens as one Free page named after
  the file. `page.output` names a Dataset as a variable of its page (and stamps `condition`);
  `page.input` reads `<page id>:<name>` from an earlier kind. `Workspace.compose` splices the
  pages a target reads from into one run graph under page-qualified ids, so a shared upstream
  chain is one memo entry however many pages pull it. LabLink opens 3.0 files and runs the
  recipe's `"page"`. The window still shows one page.

### Step 2 — the runner on the workspace (2026-10-05)
- `EngineRunner` is bound to the Workspace (any `GraphSource`) instead of one document. Every
  run id it stores, hands to the engine or emits is **page-qualified** (`pg1/n3`); a bare id
  passed to a public method means the active page. A pull on any page composes its upstream
  pages in; results, held views, cones and the cached engine are keyed on the page's composed
  revision digest. The window splits run ids and touches the canvas only for the page it
  shows; a run on another page still marks the cards it computes on the shown page.
  Invalidation goes through the Workspace's qualified touched set, so an edit on one page
  cancels exactly the runs on other pages that read it.

### Step 3 — the dock shell (2026-10-05)
- `nodelab_v2/shell.py` (`PanelSpec`, `PanelDock`, `PanelTitleBar`, `DockShell`): every side
  panel is a dock with its own title bar — pop out / dock back / close, View ▸ Panels to bring
  a closed one back, View ▸ Reset layout. Multi-instance machinery (`'<kind>:<n>'`, `+`,
  the active instance, close vetoes) is in place for the viewers and canvases of steps 4–5.
- `nodelab_v2/layout_store.py`: the layout survives a restart in `~/.nd2studios/layout.json`
  (`NODELAB_LAYOUT=0` off, `NODELAB_LAYOUT_FILE` elsewhere); a damaged or foreign file is
  ignored, never fatal.
- Fixed before release (review): a page's run identity no longer repeats after File → Open;
  a bound Page Input card can be pulled (it shows its upstream Output); re-pointing an Input
  or re-binding it through an Output rename cancels the runs reading through it; a
  page-reference cycle leaves its Inputs unbound instead of making every pull raise; an
  overlay keeps its channels when its result is re-served after an unrelated edit; Pin T/Z
  works again.

### Step 4 — viewers as docks (2026-10-05)
- The Viewer is a dock (`viewer:<n>`), and there can be several: `+` on its title bar,
  View ▸ New ▸ Viewer, or Compare. Each viewer is bound to the node it was asked to show (its
  title bar names it) and receives that node's results; the active viewer — the one last
  clicked in — gets pulls, click-to-preview, picks, the troubleshooting scope and the
  Spreadsheet. The last viewer can be closed; the next pull opens a fresh one.
- Selecting a Viewer node (any `view.*` / `plot.*` card) shows it in the active viewer at
  once, even with click-to-preview off.
- Two viewers can show one node at different frames: `EngineRunner.finished` / `plane_ready`
  now carry the `(coords, channels)` they answer, and a frame lands only in the viewer whose
  cursor asked for it.
- Compare (F8) opens a Compare viewer docked beside the active one, linked to one cursor when
  the M/T/Z extents match; maximize hides the docked viewers and restores them at their old
  sizes; a floating viewer keeps its picture (`scripts/_nodelab_v2_gl_float_probe.py`, desktop).
- Fixed: the scale bar ran past the right edge of an image narrower than 140 px.
- Fixed before release (review): two viewers of one node under F9 no longer pull each other's
  frame forever; a result re-served while the runner is busy no longer blanks the other viewers
  of that node; clicking a viewer no longer cancels a scoped pull in flight; quitting while
  maximized saves the docked layout; F8 on a card that the click just previewed compares it
  beside what was viewed before; a Compare viewer in the mini-map scrubs on its own.

## [0.1.0] — NodeLab initial build (ND2Studios V1.90)

### Added
- **Standalone project** `ND2Studios_Blender` with `run.py` launcher and the
  `nodelab` package, built on a **vendored copy** of `nd2studios` (backend
  reached via `nodelab.bootstrap.ensure_backend`, override `ND2STUDIOS_ROOT`).
- **Theme** (`nodelab/theme.py`) — full design-token set + single configurable
  accent (`set_accent`, `accent_glow`, `glow_blur_px`) and `build_stylesheet()`
  which templates `nodelab.qss` (`@ACCENT@ / @ACCENT_12@ / @ACCENT_22@`).
- **Canvas** (`nodelab/canvas/`):
  - `NodeGraphView` — manual pan, wheel zoom-to-cursor (clamp 0.3–2.2×),
    dot-grid `drawBackground`, fit-to-view, delete/backspace, right-click
    quick-add & node context menus, palette drag-drop (`application/x-nodelab-op`).
  - `NodeScene` — bound to a `pipeline_kit` `GraphSlice`; connect mirroring
    `can_connect` + `would_create_cycle`, one-wire-per-input, wire pick-up,
    splice-on-wire, duplicate/bypass/disconnect/delete, selection lift.
  - `NodeItem` — card with category header tint, shape glyph, tag chip,
    output/param/input rows, live progress bar, bypass opacity, selection/running
    glow; channel-source **pills**; rainbow `CHANNEL` header ports.
  - `SocketItem` — `PortType` glyphs (circle/square/diamond/dashed/accent) with
    hover scale + connected glow. `EdgeItem` — cubic-Bézier wires with
    horizontal control handles, channel dashing, and Run flow animation.
- **Panels** (`nodelab/panels/`): registry-driven **Add-Node palette**
  (grouped, searchable, click/drag add), **Properties** (typed widgets from
  `param_specs_for`, honoring `visible_when`, Bypass/Preview), **Results** (stat
  cards + notes), **Viewport** placeholder.
- **Chrome** (`nodelab/chrome/`): menu bar (logo + wordmark + File/Edit/Node/
  View/Help + file readout), toolbar (Open/Save/RUN GRAPH/Stop/zoom/counters),
  console+batch dock (SYS/IO/GRAPH/RUN levels), status bar.
- **Run** (`nodelab/run/`): `RunController` + `RunWorker` (QThread) drive a
  `GraphRunner` topological walk off the UI thread; enhancement recipes execute
  via `apply_recipe`; if-else branch pruning via `evaluate_simple_condition`;
  **`SpecialContext` handler seam** (`register_handler`) for special/results/
  logic nodes (simulated by default).
- **Image I/O** (`nodelab/loading.py`) — **Open TIFF…** loads `.nd2` (via the
  vendored `backend.nd2_loader`) or `.tif/.tiff` (tifffile) into
  `{channel: (T,H,W)}` off the UI thread (`LoadWorker`); the viewport shows the
  frame, the palette's CHANNELS group updates to the file's real channel names,
  the file readout shows `W×H · dtype · frames · ch`, and **Run executes on the
  loaded pixels** (with a processed-frame preview back to the viewport).
- **Save / Load** — `.nd2s_pipeline.json` via `pipeline_kit.io` (schema 6).
- **Verification scripts** copied into `scripts/` (`_pipeline_kit_parity.py`,
  `_pipeline_graph_selftest.py`).

### Notes
- Backend parity preserved: `scripts/_pipeline_kit_parity.py --check` reports no
  drift; all `pipeline_graph` self-tests pass against the vendored copy.
- Run input is a synthetic `(T,H,W)` volume until real TIFF loading is wired.
