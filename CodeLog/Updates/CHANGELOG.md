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

### Step 5 — pages in the GUI (2026-10-05)
- Every page of a file has its own canvas scene; the **page switcher** in a canvas's top-left
  corner lists the pages by kind and adds, duplicates, renames, deletes or opens one in a new
  canvas; Ctrl+PgDn / Ctrl+PgUp step through them. More canvases are docks (View ▸ New ▸
  Canvas); the main canvas stays the window's centre. The canvas clicked last decides the
  active page — palette, Properties, edits and the title bar follow it.
- The palette, the link-drag search and the Ready-to-run suggestions offer the nodes of the
  page's kind; a later page's missing source is suggested as a Page Input.
- A Page Input's Source is a dropdown of the named Page Outputs it may read; both cards say
  what they carry (`Output · raw`, `Input · raw`).
- Runs on any page update that page's cards, and viewers key results by page, so two pages'
  `n3` never share a picture or a contrast setting.
- A dock fed through a Page Input goes stale when the upstream page changes (a single-page
  dock keeps the signature its bake recorded); an edit drops cached planes and views only on
  the pages it can reach.
- Work started on a page stays with that page when another canvas is clicked: a pick is
  applied to the page it was armed on, the Movie Editor keeps the page it was opened on, and
  Shift+F5, F9 and the Hold/Bake/Release re-pulls reach a Viewer's node on another page. A
  Hold or a Bake retires its card claims on every page; opening a file rebinds every page's
  canvas; an edit on one page keeps another page's overlay channels and preload; renaming a
  page stales a dock under its blank-condition Page Output; Properties refreshes when a page
  is renamed or deleted from another canvas's switcher.

### Step 6 — linked pages (2026-10-05)
- **Duplicate as linked page**: a page that follows its master's graph — cards, wires,
  positions and every later edit — with parameter and mode values of its own (overrides). A
  linked page re-uses everything its master computed up to the first node it overrides.
- Properties shows a **Linked page** banner (master, override count, Go to master, Make
  unique); an overridden row carries an accent bar, the master's value on hover and *Reset to
  master* on right-click.
- Structural edits on a linked page are refused with a status-bar hint (its context menu
  greys them out); moving or folding a card acts on both pages, live.
- **Make unique** turns a linked page into a page of its own; deleting a master does that for
  its linked pages after a confirmation. Files store a linked page as master + overrides.
- A Dock's checkpoint (folder, bake, held state) is per page; a Load reading another file
  describes only its page; an override stays until reset. Edit ▸ Dissolve acts on the page
  shown (it acted on the first page since step 5).

### Step 7 — plots (2026-10-05)
- **Plot XY** (`plot.xy`): one column of a Label, Point or Track table against another, one
  series per group, optionally averaged with sd / sem / 95 % CI as bars or a band; style
  presets (paper, talk, poster, dark) or custom sizes, palette, labels, ranges, log axes.
- A plot's output is a **Picture**: one RGB image with no calibration, shown in the Viewer in
  true colour on a fixed 0–255 window; its input's tables do not pass on.
- **Export Figure** (`io.write_figure`) writes it as PNG / TIFF at any resolution, or as SVG /
  PDF with editable text, drawn again from the figure's own spec.
- `matplotlib` is a core requirement (`pip install -r requirements.txt`).
- Review fixes: Plot XY scatters by default; `group_by` offers `(none)` and groups rows in one
  pass, blank values as one `(missing)` series; renders are serialised (log axes were not
  thread-safe) and capped at 100 Mpx; Export Figure clamps 50–1200 dpi; Export Movie, the
  Movie Editor and LabLink quicklooks keep a picture's colours; F9 re-pull no longer recurses.

### Step 8 — plots II (2026-10-05)
- **Plot Distribution** (`plot.distribution`): histogram, kde, box, violin or ecdf of one
  column, one distribution per group.
- **Plot Heatmap** (`plot.heatmap`): a table summarised per row and column value with a
  reducer, or a Voxel layer's plane, with a colour bar.
- **Plot Time Series** (`plot.timeseries`): a column over the file's own clock (elapsed, wall
  clock or frame), mean ± spread per frame, one curve per group.
- **Per frame** on Plot XY and Plot Time Series: one figure per input frame, drawn when shown
  and carrying the input's clock; the Viewer scrubs it, Export Movie animates it, and Export
  Figure's new `frame` socket writes one of them.
- Review fixes: a group keeps its colour in every frame; heatmap ticks past 40 rows name
  the rows; histogram bins span a fixed X range (geometric on log); per-frame XY error bars
  fit; a per-frame figure counts in the memory budget, draws each frame once, is re-keyed by a
  reload, is refused past 100 Mpx at pull time and is prefetched sparingly; no channel legend
  on a chart's movie; the heatmap's Legend / Grid work and it has no Palette.

### Step 9 — tables (2026-10-06)
- **Table Concat** (`table.concat`): the tables of several inputs (pages, variants) stacked into
  one, ids kept unique, each row labelled with its `condition`, `input` and own position group.
- **Table Join** (`table.join`): another table's columns added on id+position+frame, id,
  position+frame or track, left or inner, prefixed; ambiguous keys refused.
- **Table Aggregate** (`table.aggregate`): one row per group with mean, median, sum, min, max,
  count, std or sem of a column — a Label table every plot reads.
- `std` and `sem` reducers; text columns (numpy unicode, memo- and dock-safe); plots refuse a
  text column as X, Y or Value.
- A condition TYPED on a Page Output survives later blank Outputs (blank still means the
  page's name).
- Engine: a column declaration may see every input (`adds_columns.wants_inputs`), so joined and
  concatenated columns are offered by the column menus downstream.
- Review fixes: unique ids for 0-based tables; a nested concat keeps each row's provenance and
  clears the scalar condition; per-row clocks (`time_s`, `time_jd`) after a concat; summary rows carry
  the file / group they share; std / sem / count ignore inf; Names that would replace the table
  read are refused; the Viewer leaves synthesized tables off the image; no blank file tab.

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
