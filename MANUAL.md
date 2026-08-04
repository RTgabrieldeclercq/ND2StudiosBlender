# NodeLab — User Manual

**ND2Studios_Blender** is a Blender-geometry-nodes-style node editor for microscopy image
analysis. You build an **acquire → enhance → segment → measure → track** pipeline by wiring
nodes on a canvas, tune each node in the inspector, and *pull* a node to see its result in a
live multi-channel viewer or as a table.

The engine is [`nodegraph/`](nodegraph/) (Qt-free); the editor is
[`nodelab_v2/`](nodelab_v2/). For how it works internally, see
[CodeLog/Architecture/ENGINEERING_NOTES.md](CodeLog/Architecture/ENGINEERING_NOTES.md).

> **State as of 2026-08-03:** catalog **69 node types** — 72 registered ops, counting the
> three the GUI layer adds (`io.load`, `io.dock`, `view.viewer`). V2.12 folded Watershed +
> StarDist into the one **Segmentation** node and added CellSAM; V2.13 ported the last of
> Cell-Tracker's processing — Flatten Illumination, Temporal Gain, Remove Blobs, Object
> Metrics, Object Field — and **Stitch (M→1)** put the first node behind the `MULTI_VIEW`
> footprint; V2.14–V2.21 added the machine-aware run and the optional CUDA path
> ([§6](#tuning-the-run-to-your-machine-v214)), parameter picking
> ([§8b](#8b-picking-parameters-off-the-image-v216)), docking
> ([§12b](#12b-docking-bake-a-chain-to-disk-and-free-the-memory-v218)), parameter iteration
> ([§12c](#12c-iterating-a-parameter-sweeps-and-searches-v219)) and live node editing
> ([§12d](#12d-editing-a-node-while-nodelab-is-running-v220)). Headless gate
> `python -m nodegraph.selftest` → **85 `[ok]` lines green**; driven GUI gate
> `scripts/_nodelab_v2_phase5_probe.py` → **ALL PASS** (23 `[ok]`); catalog gate
> `python scripts/_catalog_snapshot.py check` → **CATALOG IDENTICAL**. On Windows, run all
> three under `PYTHONUTF8=1` — they print `σ`/`→` and a cp1252 console raises
> `UnicodeEncodeError` inside the reporting line itself, which reads like a failure but is
> not one. There is exactly one editor — the first-generation `nodelab`/`pipeline_kit` app
> was removed on 2026-07-29 and `--legacy` no longer exists.

---

## Contents

1. [Install & launch](#1-install--launch)
2. [The window](#2-the-window)
3. [Your first graph in five minutes](#3-your-first-graph-in-five-minutes)
4. [Loading data](#4-loading-data)
5. [Building graphs](#5-building-graphs)
6. [Running: pull, progress, errors](#6-running-pull-progress-errors)
7. [The Viewer](#7-the-viewer)
8. [The Inspector — parameters, units, auto/pinned](#8-the-inspector--parameters-units-autopinned)
   - [Picking parameters off the image](#8b-picking-parameters-off-the-image-v216)
   - [Zooming into a big image — detail on demand](#8c-zooming-into-a-big-image--detail-on-demand)
9. [The 2D/3D lever](#9-the-2d3d-lever)
10. [Domains, layers and the layer picker](#10-domains-layers-and-the-layer-picker)
11. [Spreadsheet & export](#11-spreadsheet--export)
12. [Organising a big graph: frames, reroutes, groups, zones](#12-organising-a-big-graph-frames-reroutes-groups-zones)
    - [Docking: bake a chain to disk and free the memory](#12b-docking-bake-a-chain-to-disk-and-free-the-memory-v218)
    - [Iterating a parameter: sweeps and searches](#12c-iterating-a-parameter-sweeps-and-searches-v219)
    - [Editing a node while NodeLab is running](#12d-editing-a-node-while-nodelab-is-running-v220)
13. [Saving & loading](#13-saving--loading)
14. [Keyboard & mouse reference](#14-keyboard--mouse-reference)
15. [Node reference](#15-node-reference)
16. [Worked workflows](#16-worked-workflows)
17. [Headless / scripted use](#17-headless--scripted-use)
18. [Troubleshooting](#18-troubleshooting)
19. [Verifying a build](#19-verifying-a-build)

---

## 1. Install & launch

```bash
pip install -r requirements.txt
python run.py
```

Python 3.13 is the tested interpreter. Required: PySide6, numpy, scipy, scikit-image,
PyWavelets, opencv-python, nd2 + dask, tifffile, blosc2, pyarrow.

**Optional, feature-gated at import.** The app launches and the *whole palette registers*
without these; the node raises a friendly install hint only when you pull it:

| Extra | Unlocks |
|---|---|
| `scikit-learn` | `analysis.cluster_points` (GMM / KMeans + BIC) |
| `numba`, `pandas` | `track.objects` (the 5-method tracker) |
| `al-dic` | `analysis.dic_correlate` (2D DIC, pyALDIC) |
| `stardist`, `tensorflow`, `csbdeep` | `analysis.segment` → method **stardist** |
| `cellSAM` (+ `torch`) | `analysis.segment` → method **cellsam**; the weights also need
  `DEEPCELL_ACCESS_TOKEN` from <https://users.deepcell.org> (non-commercial academic
  licence), or point `model_path` at a local `.pt` |
| `zarr` | only `scripts/_bench_provider_granularity.py` |
| `cupy` | optional CUDA path for the ndimage filters — install with `python scripts/setup_cupy.py`, which picks the build matching your driver + GPU. Off unless `NODEGRAPH_GPU=auto`; see [§6](#tuning-the-run-to-your-machine-v214) |

### Setting up CellSAM (once per machine)

The weights are ~1.7 GB and licensed for **non-commercial academic use** through an
authenticated endpoint, so they cannot ship with the repo — every user fetches their own:

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
python scripts/setup_cellsam.py     # downloads + verifies the weights
python scripts/_cellsam_smoke.py    # proves the whole path: model -> node -> labels
```

`setup_cellsam.py` asks for a token from <https://users.deepcell.org> the first time and
tells you exactly how to set it. **After that, loading is fully offline** — the token is
never consulted again, including from the GUI, because `get_model()` returns early once
`~/.deepcell/models/cellsam_v<ver>/` exists.

> **If your machine intercepts HTTPS** (corporate proxy, or antivirus with HTTPS scanning),
> Python's downloader will fail with `CERTIFICATE_VERIFY_FAILED` even though your browser
> and `pip` work — the interceptor's root is trusted by the OS but absent from certifi, and
> on Python 3.13 adding it to a CA bundle does **not** help either (verification is strict
> and rejects most antivirus-generated roots). `setup_cellsam.py` detects this and tells you
> to `pip install truststore`, which routes verification through the OS trust store; it then
> retries automatically. This was the actual experience on the development machine (Norton
> HTTPS scanning) — see [`nodegraph/kernels/cellsam_segment.md`](nodegraph/kernels/cellsam_segment.md) §8.

**Once the weights are on disk, CellSAM needs neither a token nor a network** — verified
with the token removed from the environment and a socket-level tripwire armed: 0 network
calls, model load + segmentation unaffected. That applies to the GUI too, which runs the
same kernel.

**Second machine, no token at all.** The weights directory is portable. Copy
`~/.deepcell/models/cellsam_v<ver>/` (or just `cellsam_general.pt`) to the other machine
and either drop it at the same path, or point the node's **`model_path`** socket at the
file — that calls `get_local_model` and skips the download, the token and the network
entirely. Verified to give bit-identical results to the downloaded copy. Respect the
non-commercial academic licence when you copy them.

Reference timings on a CPU-only torch build: model load ~1.5 s, then ~10 s per image
(matching the paper's benchmark; cost scales with cell count because the mask decoder runs
once per detected cell). A CUDA torch build is picked up automatically
(`NODELAB_CELLSAM_DEVICE=cpu|cuda|auto` overrides).

**Environment switches**

| Variable | Effect |
|---|---|
| `NODELAB_GL=0` | force the CPU viewer path (skip requesting a GL context) |
| `NODELAB_OVERLAYS=<path>` | read/write overlay settings from one explicit JSON file |
| `PYTHONUTF8=1` | needed on a cp1252 Windows console for the gates' `µ`/`σ`/`↔` glyphs |

---

## 2. The window

NodeLab opens on a **blank canvas** with a welcome card: *Load image… / Browse nodes /
Example graph*. The Viewer pane starts collapsed — the canvas owns the whole centre until
your first pull returns an image, which unfolds the Viewer to a ~2.5:1 split.

```
┌──────────────────────────────────────────────────────────────────────┐
│ File  Edit  Run  Graph  View  Help                                   │
├──────────┬───────────────────────────────────────────┬───────────────┤
│ Palette  │              VIEWER                       │  Properties   │
│ (search, │  image · channel strip · LUT · overlays   │  (inspector)  │
│  grouped │───────────────────────────────────────────┤               │
│  by      │              NODE CANVAS              [⛶] │  Spreadsheet  │
│  category│  cards · wires · frames · mini-map        │  (tables)     │
├──────────┴───────────────────────────────────────────┴───────────────┤
│ Console (log + full tracebacks, Copy all / Clear)                    │
├──────────────────────────────────────────────────────────────────────┤
│ status:  ● LED   node · progress · wall time                          │
└──────────────────────────────────────────────────────────────────────┘
```

* **Palette** — searchable, grouped by category. Double-click a row to place the node at
  the view centre, or drag it onto the canvas.
* **Node canvas** — pan by dragging empty space, zoom with the wheel, `Home` fits the graph.
  The `⛶` button in the top-right (or `Ctrl+Space`) maximises it.
* **Viewer** — the pulled node's image. See [§7](#7-the-viewer).
* **Properties (Inspector)** — the selected node's parameters. See [§8](#8-the-inspector--parameters-units-autopinned).
* **Spreadsheet** — the pulled node's structure tables (Label / Point / Track / Mesh).
* **Console** — a selectable log; every failed pull lands here with its **full traceback**
  and a *Copy all* button. This is where you look when a node goes red.
* **Status bar** — a pulsing LED (idle / busy / error) plus the current node and timing.

Light theme: **View → Light theme**.

---

## 3. Your first graph in five minutes

1. **File → Load ND2/TIFF file…** (`Ctrl+L`), pick a file (or Ctrl/Shift-select several).
   A source card appears per file, titled with the file name, carrying one output socket
   **per channel** plus an *All channels* output. No pixels have been read yet — only
   metadata.
2. Drag **Gaussian Blur** from the palette onto the canvas. Wire the source's `ch0`
   output into its `data` input.
3. Select the Gaussian node. In the Inspector, note that `sigma` is in **µm** and shows an
   **auto** value derived from the file's own calibration. Type over it to pin it.
4. Add **Threshold** → wire it after the blur. Leave `method` at `otsu`.
5. Add **Connected Components** → wire it after the threshold.
6. Add **Measure** → wire it after the labels.
7. **Double-click the Measure card** (or select it and press `F5`). The engine walks
   backwards, computes only what is needed, and the Viewer opens with the result; the
   Spreadsheet fills with one row per label.
8. **Ctrl+E** exports that table to CSV / Parquet / Arrow.
9. **Ctrl+S** saves the graph as `*.nd2graph.json`.

Now change `sigma`. Only the blur and everything downstream of it recompute — the source
ingest is memoized. That incremental behaviour is the whole point of the engine.

---

## 4. Loading data

### File → Load ND2/TIFF file… (`Ctrl+L`)

Reads **metadata only** and drops a pre-configured `io.load` source node — **one per file
you pick**. The dialog is multi-select: Ctrl-click or Shift-click (or `Ctrl+A`) to load a
whole set at once, and the cards stack in a column at the centre of the view, each one
selected and ready to wire. A file the reader refuses doesn't cost the rest of the
selection — the readable ones load and the failures are listed together afterwards.

Per source node:

* **Axes** are normalised to canonical `(M, T, Z, C, Y, X)`, inserting size-1 axes for
  absent dimensions.
* **Calibration** is parsed into the engine's vocabulary: `pixel_size_um`, `z_step_um`,
  `dt_s`, `channel_emission_nm`, `objective_na`, `objective_magnification`, `bit_depth`.
  Every metadata-intelligent parameter in the graph re-seeds from this immediately.
* **Channel display** (names, emission λ, colours) drives the per-channel output sockets,
  their wire tints, and the Viewer's channel buttons.
* ND2 pixels are read through `nd2.ND2File.to_dask()`; TIFF through `tifffile` (first
  series, best-effort calibration from the TIFF tags).

**Pixels ingest lazily on the first pull.** The first pull writes a planar-block `.b2nd`
store next to the file and re-opens it lazily thereafter, so subsequent runs skip the
conversion. Expect the first pull on a large file to be slower than the rest.

### Ingesting: double-click a source card

You don't have to wait for a pull to discover that cost. **Double-click a source card** and
it ingests that one file right away — on its own worker, showing its own progress on the
card, with the rest of the app still usable. When it lands, that card opens in the Viewer.

**Several ingest at once.** Double-click each of the files you just loaded and they all go
together (up to `NODELAB_INGEST_WORKERS`, by default 4 or fewer on a small machine; the rest
queue and start as slots free). **Run → Ingest all source files** (`Ctrl+I`) does the whole
graph in one command, and a source card's right-click menu has *Ingest this file now*.

This is a separate lane from pulling. A pull is one-at-a-time and latest-wins — the right
model for "show me this node", and the wrong one for putting five ND2s on disk, because the
second file could not start until the first had finished. An ingest is per-file and
order-free: its only result is a store on disk plus a cached provider, both keyed by the
file's path, so it keeps whatever you do to the graph meanwhile.

Notes:

* Only the card you double-clicked **last** opens in the Viewer when its file lands. The
  others just finish — five files landing minutes apart would otherwise each grab the pane.
* A single *click* (click-to-preview, always on while the canvas is maximized) never starts
  an ingest. It says the file isn't ingested yet and leaves it alone.
* Two cards naming the same file share one ingest and finish together — two writers on one
  `.b2nd` store is exactly how a torn store gets made.
* A pull that reaches a file already ingesting in the background waits for it and takes the
  result rather than starting a second copy.
* A failed ingest marks its card red, reports the reason, and stays retryable.

**A pull only ingests the files it actually needs** — the sources in the pulled node's
upstream closure. Files sitting on the canvas that this chain doesn't touch cost nothing.

### Several files in one graph

A graph can hold as many source nodes as you like, and they're independent all the way
down: each `io.load` keys its own provider and its own `.b2nd` store by the file's absolute
path (so two files sharing a basename in different folders never fight over one store), and
each is memoized separately — re-pulling a branch fed by file A doesn't touch file B.

Where they come back together:

* **`view.overlay`** composites a second file onto the one you're viewing, placed by stage
  coordinates and focus rather than by pixel index (it needs the placement metadata ND2
  carries and TIFF doesn't — it says so rather than guessing).
* **Correlation / DVC reference inputs** take a separate file as the undeformed reference
  frame.
* Otherwise the branches stay independent — the Viewer and Spreadsheet show whichever node
  you last pulled, so you compare files by pulling each branch in turn (or by exporting
  each one and joining the tables outside).

The Viewer's hover readout deliberately withholds its "raw" column when a node has **two**
sources upstream — there is no single raw file to attribute the value to.

### Per-channel output sockets

`io.load` and `channel.split` grow one synthetic output socket per channel (`ch0`, `ch1`, …).
These are a GUI convenience: at run time each `chK` wire is rewritten into a real
`channel.select` tap with `channels=[K]`, so the engine sees an ordinary node. The `out`
socket carries the full multi-channel bundle.

### No file? You still get a working app

An `io.load` node with an empty `path` falls back to a deterministic synthetic source
(`1×1×5×2×512×512`) with real-looking calibration — so the whole GUI, every node, and the
derive pills work out of the box. The welcome card's *Example graph* button builds an
8-node demo chain on it.

---

## 5. Building graphs

### Placing nodes

* Palette **double-click** → placed at the view centre.
* Palette **drag & drop** → placed where you drop it (a drop that lands on the welcome
  card is forwarded to the canvas beneath).
* **Drop a node onto an existing wire** → it **splices** in (the wire is rerouted through
  it). Refused cleanly on value/field wires.
* Drag from a socket and **release on empty canvas** → the **link-drag search** popup opens,
  filtered to ops with a compatible socket; picking one places it there and connects it.

### Wiring

* Press a socket and drag; the target socket rings **green** (valid) or **red** (invalid —
  wrong type, wrong direction, or it would create a cycle).
* `DATASET` pairs only with `DATASET` (the thick main wire). Value sockets connect on an
  exact type match or an implicit conversion (bool→int→float, scalar→vector broadcast,
  vector→colour). Vector arity **widens only** — 2→3 pads the axial component with 0, and
  3→2 is rejected.
* A second wire into a **non-multi** input **replaces** the existing one (Blender behaviour).
* Pressing a **connected** non-multi input **detaches** the wire and re-drags it from its
  source.
* Field sockets draw as a **diamond**, plain value sockets as a circle, Dataset as a large dot.

### Wire colours and domain chips

Dataset sockets carry a **domain rail** — small chips (`VOX`, `LBL`, `PT`, `TRK`, `MSH`, …)
naming the attribute domains present on that wire. Wires are tinted by domain, and a node
whose required domain is **missing upstream** paints a red validation chip. That is your
at-a-glance answer to "why does Measure complain?" — it needs `LABEL`, and nothing upstream
produced labels.

### Editing the graph

| Action | How |
|---|---|
| Delete | `Del` / `Backspace`, the ✕ badge that appears when you hover a card, or right-click → Delete |
| **Dissolve** (delete but keep the chain) | `Ctrl+X`, right-click → Dissolve, or Edit menu |
| Mute (bypass, pass-through) | select + `M` |
| Collapse a card to a compact strip | select + `C` |
| Select all nodes | `Ctrl+A` |
| Insert a reroute dot on a wire | **double-click the wire** |

Muted nodes dim and are bypassed at run time — the engine never sees them.

---

## 6. Running: pull, progress, errors

Nothing computes until you **pull** a node. The engine walks *backwards* from the node you
asked for and computes only what that request needs.

| Action | How |
|---|---|
| Pull the selected node | `F5`, or **Run → Pull selected node** |
| Pull & view a node | **double-click its card** |
| Re-pull the currently viewed node | `Shift+F5` |
| Preview whatever you click | **View → Preview clicked node** (always on while the canvas is maximised) |
| Analyse only the frame(s) you pick | `F9` — see [Troubleshooting mode](#troubleshooting-mode-analyse-the-frames-you-pick--f9) |

One pull runs at a time (latest-wins queueing) on a worker thread, so the UI never blocks.
Edits invalidate in-flight results by epoch, so a stale result can never land in the Viewer.
A re-pull with nothing changed is a **full memo hit** — every card reports `cached` and
nothing recomputes.

### Troubleshooting mode: analyse the frames you pick — `F9`

Tuning a threshold on a 200-frame acquisition should not cost 200 frames of segmentation.
Press **`F9`** (**Run → Troubleshoot: picked frames only**) and every pull is evaluated over
**just what the Viewer's M/T/Z strips scope to** — the boxes you have *picked*, or the
single frame the cursor is on if you have picked nothing.

**You cannot miss that it is on.** The node canvas takes an **amber frame** all the way
round, a **● TROUBLESHOOTING MODE** badge appears in its top-left corner naming the current
scope (`t7`, `3T[1, 3–4]·2Z`) with a slowly pulsing dot, and an amber **`SOLO t57`** chip
sits in the status bar. That loudness is deliberate: a scoped pull leaves the graph, the
cards and the progress rails looking *identical* to a full run, so the canvas itself is the
only thing that can stop you reading one-frame numbers as series numbers. With the canvas
maximised the badge steps to the right of the mini-map rather than hiding behind it.

* **Move the T cursor to analyse another frame.** With nothing picked the strip is a
  *chooser*, not a scrubber: moving it re-runs the graph on that frame (Z and the channel
  toggles stay instant, as always).
* **Ctrl+click (or ctrl+drag) T boxes to run several frames.** This is what gives a
  temporal node something real to do — a tracker linking three picked timepoints instead
  of nothing. Shift+click extends the pick from the last one; right-click offers
  *pick all / invert / clear*; **Run → Clear picked frames** drops them all.
* **Ctrl+click Z boxes to cut the volume too.** A z pick applies *inside* every scoped
  frame, so checking a 3D pipeline on 4 of 60 planes is another 15× on top of the frame
  saving. The three axes are independent and combine as a cross product: 2 positions ×
  3 timepoints × 5 planes runs 6 frames of 5 planes each. The two defaults differ on
  purpose — an unpicked M or T means *the frame the cursor is on*, an unpicked Z means
  *the whole volume*, because a 3D node needs one.
* **Selections you have already run come back instantly** — the scoped set is part of every
  node's memo key, so flipping between two frames after the first look costs nothing.
* **Nothing about your graph changes.** Same cards, same run plan, same per-node progress,
  same saved file. It is the *run* that is scoped, not the pipeline — so what you tune here
  is exactly what the full series will use.
* Measured on a 24-frame, 4-plane 512² source through *Gaussian 3D → Segmentation →
  Measure*: **8.97 s → 0.25 s** (36×), a fresh frame 0.23 s, a revisit 0.00 s. The saving
  scales with how little you scope to, and applies to memory as well as time — the eager
  analysis nodes allocate the picked frames' raster instead of the series'.

**Nodes see only the scoped frames and planes**, which is inherent to the mode rather than a
defect. With one frame a tracker has nothing to link across and a time reduction reduces a
single sample; picking several fixes that. The caveat worth holding on to is that skipped
indices are *absent*, not empty — links, per-frame rates and any 3D measurement are computed
across the **picked** neighbours rather than the acquired ones, and `dt_s` / `z_step_um`
keep describing the source (exact for a contiguous pick, an approximation for a strided
one). A 3D filter is the clearest case: a Gaussian 3D over 5 picked planes does not produce
the same pixels as the same filter over all 60, so use z picks to check *shape and speed*,
not to read off numbers. The Spreadsheet likewise shows the scoped rows, numbered from 0.
Turn `F9` off for real results — the full-series pull is a normal pull and reuses everything
the lazy chain already cached.

The one cost it cannot avoid is the **first** pull of a new file, which still pays the
one-time ingest of the whole volume (below) — that is a property of the file, not of the
graph.

### Reading per-node progress on the canvas

Each card wears up to **two** rails under its header plus a status dot:

| Look | Meaning |
|---|---|
| hollow dot | queued |
| **orange rail (upper)** | frames finished out of the total frame count — where the node is in the series |
| **blue rail (lower)** | the work inside the frame being processed right now; it refills once per frame |
| **blue rail sweeping, orange holding** | inside one step this frame that cannot report its own progress — a CellSAM / StarDist inference. It is working; how far along is genuinely unknown |
| pulsing dot, one blue rail **filling** | running, but the node has no frame axis to report (a whole-series fold, a file read) |
| pulsing dot, blue rail **sweeping** | running, cost not fractionally knowable |
| solid glowing dot, no rails | done |
| hollow ring | **memo hit** — nothing was recomputed |
| red | error |

Read them together: the orange one is the slow, coarse answer to *how far through the movie
am I*, and the blue one is the fast answer to *is this frame still moving*. That split is
what tells a stall apart from a slow frame — a blue rail creeping across while orange sits
still is one expensive timepoint, whereas both frozen is something wrong. The blue rail is
the glowing one because it is the one worth watching.

Both bars apply to every node that does its work eagerly. Nodes with a long solve *inside* a
single frame — DVC and DIC correlation especially — drive the blue rail from the solver's own
iteration count, so it keeps moving through a correlation that takes tens of seconds.

**Segmentation is the case where the distinction earns its keep.** A CellSAM or StarDist plane is
a single call into someone else's model, and how far through it is is not observable — so:

- **CellSAM with `tile` on** reports each tile as it finishes, and the blue rail fills for real.
  If you want a moving bar on a big FOV, this is the setting that gives you one (it is also the
  setting upstream recommends above roughly 3000 cells per image).
- **CellSAM with `tile` off, and StarDist**, are one opaque inference per plane, so the blue rail
  **sweeps** for the duration instead of pretending to a percentage. Orange still tells you which
  frame you are on, and the rail refills the moment the plane lands.
- On CPU these run several planes at once, and the blue rail then counts *planes finished within
  the frame* — with four in flight, a sub-fraction for any one of them would be meaningless.

A working card also takes an accent border and its outgoing wires flow. Exact numbers —
`frame 13/40 · 40% of frame · 31% total` — live in the **tooltip** and the status bar (which
stacks the same two bars the same way), so a busy canvas stays readable.

### The first pull of a file shows a determinate bar

The **first** pull of an ND2/TIFF pays a one-time **ingest**: the volume is read, then
compressed into a planar-block `.b2nd` store (a `<file>.b2nd_store` directory beside it,
with one `level_<l>.b2nd` per pyramid level). For a large series that is minutes, so the
status bar shows a real **progress bar** with the phase and the source card's rail fills in
step. The bar appears only while a genuine fraction is known; indeterminate work keeps the
sweeping rail.

An **ND2** reads and compresses in one streamed pass — the phase reads
`ingesting <file>.nd2` throughout — and never holds more than one slab of the series in
memory, so a file far bigger than your RAM ingests fine. A TIFF still reads whole first
(`reading TIFF`, then `writing <file>.b2nd_store`), because `tifffile` has no lazy read.

**Every later pull re-opens that store lazily and is effectively instant** (milliseconds —
it decompresses only the blocks you look at). So don't delete `.b2nd_store` directories:
they are the difference between a sub-second open and re-paying the whole ingest. They are
also portable — copy one to another machine rather than re-ingesting there.

**An interrupted ingest repairs itself.** The store holds a small pyramid of
mean-downsampled levels (`level_0` is your data; the levels above it are what the Viewer
reads when you are zoomed out). Each level is stamped on disk before its first byte and
again when it is finished, so an ingest killed partway — a crash, a cancel, a full disk —
leaves a record of exactly how far it got instead of a file that *looks* complete and reads
as zeros where nothing was written. On the next pull:

* a store missing levels above `level_0` is **completed in place**, from `level_0` alone.
  The source file is not read, or even needed, and nothing you have already computed is
  invalidated — so this costs one pass over the store and nothing else. The status bar says
  `completing pyramid for <file>`. (On a 12-position × 16-timepoint × 210-plane 1024² series:
  a couple of minutes, once.)
* a store whose `level_0` itself is unfinished is **re-ingested** from the source, because
  no other level can rebuild it. If the source file is gone, so is that store — delete the
  directory and start from the data.

**Stores written before the stamps shipped are checked a second way.** They carry no record
of how far they got, so on open the program asks the store itself which of its chunks were
ever written — a fast header scan, no pixels read. If the unwritten ones form a run at the
*end*, that is a write that stopped partway, and the store is re-ingested from the source.
Scattered blank chunks are left alone: those are real, legitimately empty data.

This is not hypothetical. It is how a 12-position × 16-timepoint × 210-plane 1024² series in
this lab was found to be showing solid black for every position from the third onward: its
`level_0` had only ever received 20% of its chunks, the ND2 was perfect, and nothing on disk
said otherwise. If you see a store re-ingest itself with a message about chunks, that is why —
and the frames you were missing will be there afterwards.

**A lazy node finishing in microseconds is correct, not a bug.** Most enhancement nodes
return a *lazy provider*, not pixels; the real cost appears on whichever node actually reads
planes (often the Viewer's decode). The bars report the truth rather than a comforting
fiction.

### Tuning the run to your machine (V2.14)

Out of the box the engine now **sizes itself from the machine**: cache budgets are a
fraction of installed RAM, and node unit loops (per plane in 2D, per volume in 3D) run
across your cores. Nothing needs configuring — but every knob has an environment override
for a headless run, a smaller box, or a bisect.

| Variable | Default | What it does |
|---|---|---|
| `NODEGRAPH_PARALLEL` | `auto` | `auto` / `thread` / `process` / **`off`** (the old single-threaded engine) |
| `NODEGRAPH_WORKERS` | `cpu_count-2` | Max concurrent units |
| `NODEGRAPH_PROC_WORKERS` | `min(workers, 8)` | Process-pool size (per-frame CNN inference) |
| `NODEGRAPH_CACHE_BYTES` | 20% of RAM, ≤64 GiB | Tile/unit cache — **also the lazy-vs-eager gate**, see below |
| `NODEGRAPH_MEMO_BYTES` | 12% of RAM, ≤32 GiB | Persistent memo across edits |
| `NODEGRAPH_PLANE_CACHE_BYTES` | 4% of RAM, ≤8 GiB | Decoded display planes (scrub/playback smoothness) |
| `NODEGRAPH_FLOAT32` | off | Stream in float32 — faster and half the memory, **but changes numerics** |
| `NODEGRAPH_GPU` | `off` | `auto` / `on` enable CUDA dispatch for the ndimage filters (needs CuPy) |
| `NODEGRAPH_STORE_DIR` | beside the file | Put every `.b2nd_store` in one directory |
| `NODELAB_INGEST_WORKERS` | `min(4, workers/4)` | How many source files may ingest at once (§4) |

Budgets accept a suffix: `NODEGRAPH_CACHE_BYTES=48g`.

**Why the cache budget is the important one.** A 3D node streams *one touched volume at a
time* only while that volume fits `budget / 2`; past that it falls back to realizing the
entire series eagerly. Displaying one plane of an 80-plane, 3-timepoint series measured
**81 s** at the old fixed 1 GiB budget versus **26 s** once the node stayed lazy — and the
gap grows with the number of timepoints, because the eager path computes every one of them.
Rule of thumb: keep the budget above `2 × z × y × x × 8` bytes for your largest volume.

**Results do not depend on any of this.** Parallel and serial runs are byte-identical,
including the nodes whose object ids depend on unit order (Segmentation, Connected
Components, Spot Detection number their objects by a running counter). The expensive part
runs in parallel; the counter-advancing fold stays serial and ordered. `nodegraph.selftest`
asserts that equality directly, and passes under `off` / `thread` / `process` / `float32`
alike. The one exception is `NODEGRAPH_FLOAT32`, which is genuinely a different
computation — ~7 significant digits instead of ~16.

**If the data is on slow media, move the store.** The `.b2nd_store` lives beside the source
file by default. On a USB SSD here it wrote at 27 MB/s against 56 MB/s on the internal
NVMe; point `NODEGRAPH_STORE_DIR` at fast local disk and the store follows (keyed by a
path digest, so same-named files in different folders never collide).

### What a 3D whole-volume node costs while you look at it

A 3D node whose maths spans z — **Deconvolve** above all — cannot answer for one plane
without doing the whole stack. Its work is therefore *deferred to the Viewer*: the pull
itself returns in milliseconds and the wait you actually see is the first displayed plane.
Measured on a 12-position × 16-timepoint × 210-plane × 1024² series (3D Deconvolve, 10
iterations, real optics — NA 0.8, 663 nm, 0.287/0.288 µm sampling):

| What you do | Cost |
|---|---|
| Pull the node | ~0 s (nothing is computed yet) |
| Display the first plane | **~130 s** — one whole 210-plane volume |
| Scrub z inside that volume | **0.00 s** — every plane of it was cached |
| Step T or position by one | **~130 s** — a different volume |
| Come back to a volume you have seen | 0.00 s while it is still in the tile cache |

So the shape to expect is: one wait per `(position, timepoint)`, then free z-scrubbing
inside it. The window stays responsive throughout — the decode runs on a worker and the
status bar reads *reading planes* — and results are cached across edits, so an unrelated
change to the graph does not re-run a volume you already computed. The cache holds
`NODEGRAPH_CACHE_BYTES / (z·y·x·8)` volumes (≈31 on a 256 GB box for the series above);
past that the oldest are evicted and revisiting them recomputes.

If that is too slow to work with, the levers, cheapest first:

* **Pick your frames — `F9`.** Troubleshooting mode evaluates only the M/T frames you tick,
  so you iterate on one position instead of 192. (A **z** pick also cuts the volume, which
  for a 3D deconvolution *changes the answer* — the PSF then couples only the planes you
  kept. Use it to preview, not to produce numbers.)
* **Fewer iterations.** Runtime is linear in the Deconvolve `iterations` socket. 10 is the
  default; 4–5 is often enough to judge whether the PSF is right.
* **`NODEGRAPH_FLOAT32=1`.** On the volume above: 127 s → **101 s** and 8.5 GB → **4.3 GB**,
  at ~7 significant digits instead of ~16.
* **The 2D lever.** A per-plane deconvolution of the same series is ~1 s per displayed plane
  instead of ~130 s per volume — a different (purely lateral) correction, not an
  approximation of the 3D one, so choose it on the optics, not on the clock.

One number worth knowing before you start: a 3D RL pass on a 210×1024² volume peaks around
**8.5 GB** of working set in float64. It is one volume at a time by design; do not raise
`NODEGRAPH_WORKERS` expecting several at once to be free.

> **Why it is ~130 s and not ~230 s (V2.19).** The PSF this node derives is a Gaussian, and a
> Gaussian is *separable* — a 3D blur is three 1-D passes, not one 3-D convolution — and it is
> its own mirror, so both convolutions inside a Richardson–Lucy step are the same separable
> blur. The old path handed the full 23×5×5 kernel to an FFT convolution twice per iteration
> (226.5 s, 20.9 GB peak); the current one does the axial pass as a BLAS reduction over whole
> planes (127.3 s, 8.5 GB). The two agree to ~1e-15 relative and `nodegraph.selftest` asserts
> that against skimage directly — this is the same deconvolution, computed the way its own
> kernel allows, not a faster approximation of it.

### Using a CUDA card (optional, off by default)

With CuPy installed, `NODEGRAPH_GPU=auto` routes the `scipy.ndimage` kernels to the card.

**Install it with the setup script, not a pip line from a README.** CuPy ships a separate
build per CUDA major, and the correct one depends on two things about *your* machine: the
driver's CUDA version (a wheel's runtime cannot exceed it) and the GPU's compute capability
(each CUDA major drops old architectures — so on, say, a GTX 1080 behind a current driver the
*newest* wheel is the wrong answer even though the driver would load it). The script probes
both from the driver library directly — no `nvidia-smi` needed, which is often missing from
`PATH` on Windows even with a healthy driver:

```bash
python scripts/setup_cupy.py             # detect → install the match → verify each kernel
python scripts/setup_cupy.py --check     # report what it would do, install nothing
python scripts/setup_cupy.py --bench     # also time every kernel against scipy
```

It always installs the `[ctk]` extra. That is load-bearing rather than tidy: CuPy
JIT-compiles these kernels, so **without the CUDA headers every single op is rejected** and
the engine silently runs everything on the CPU — which looks exactly like success. The
script ends by printing each kernel's verdict and its measured difference from scipy, so you
can see what is actually running on the card.

**It is off by default because the honest answer is "it depends on the node."** The raw
kernels are dramatic — verified *bit-exact* against scipy on an RTX 3090, a 4096² median at
79× and a 64×512² gaussian at 94× — but the CPU alternative here is twelve cores, not one.
Measured through whole node pulls:

| node shape | GPU vs 12-core CPU |
|---|---|
| 3D volume filters (gaussian, median) | **9–31×** |
| tiled 2D filters | ~1× — a 512² tile is below the dispatch threshold, so it stays on the CPU |
| `analysis.edt` | ~1× — deliberately never dispatched; cupyx's 3D transform is only ~2× *one* core |

So: turn it on when the graph does heavy **3D volumetric** filtering; leave it off otherwise.
Two safety properties make `auto` safe to try on real data:

* **Every kernel must earn its place.** The first time an op is dispatched it is run on both
  backends over a fixture and the results compared; a mismatch permanently retires that op
  to the CPU for the session. An op that cannot reproduce scipy is never used on your data.
* **One device, one queue.** Device work is serialized. Without that, worker threads each
  push their own transfer + kernel at a single card and it thrashes — measured as a **33×
  regression** before the queue was added.

### When something fails

The card turns red, the status LED goes red, and the **Console** gets the full traceback
with *Copy all*. Dependency-gated nodes raise a one-line install hint (e.g. "install
scikit-learn") rather than a mysterious `ImportError` deep in a kernel.

---

## 7. The Viewer

The Viewer renders the pulled Dataset as a **multi-channel colour composite**. There is no
title bar — the image fills the top; all controls sit in one strip beneath it, and the
viewed node's name is folded into the status line at the bottom.

### Navigation

* **M / T / Z frame strips**, one row each, with a **▶ play/pause** button and an fps
  spinner per axis. Each strip is **one box per frame** at a fixed size, compressing to fit
  once they would run past the end of the row — so the row is a readable index of the axis,
  not just a groove. Press and drag anywhere on it to scrub; the drag keeps working after
  the pointer leaves the strip, clamping at whichever end you passed. The box you are on is
  filled with the accent colour; the wheel and the arrow keys step one frame. Playback
  advances one axis frame-by-frame, self-throttled to the target fps, and reports the
  achieved rate beside the strip.
* **Ctrl+click a strip to *pick* boxes** (ctrl+drag paints a run, shift+click extends from
  the last one, right-click offers pick all / invert / clear). Picked boxes are outlined in
  the accent colour and counted in the `4/199 ·3` readout. Picks are the run scope for
  [troubleshooting mode](#troubleshooting-mode-analyse-the-frames-you-pick--f9) (`F9`) and
  do nothing while it is off. All three axes pick, and they compose as a cross product:
  2 positions × 3 timepoints × 5 planes runs 6 frames of 5 planes each.
* **Scroll to zoom, drag to pan.** The view re-fits on resize unless you have zoomed or
  panned yourself.
* Under [troubleshooting mode](#troubleshooting-mode-analyse-the-frames-you-pick--f9) (`F9`)
  the M/T strips keep spanning the whole series, but they select **which frames get
  analysed** rather than scrubbing already-computed ones.

### Hover readout — what is under the pointer

Move the pointer over the image and the line just above the status strip answers, live:

```
x 511, y 511 px  ·  878.9, 878.9 µm  ·  stage -19702.7, -28602.2 µm  ·  GFP 663.1 (raw 1279)
```

* **Position** in the viewed node's own pixel grid, and the same point in **microns** from
  the top-left of the field (using that node's propagated `pixel_size_um`).
* **Absolute stage position** in the microscope's own coordinates, from the ND2's
  per-position log. Nikon records the stage coordinate of the **field centre**, so hovering
  the middle of the frame reads back exactly the logged position for that M. Image `+x`/`+y`
  (right / down) are taken to run along stage `+x`/`+y`.
* **One value per shown channel** — every channel currently toggled into the composite,
  not just the one whose histogram is selected.
* **The raw file value in brackets** when the viewed node changed it: `663.1 (raw 1279)`
  is the enhancement's number and the untouched microscope count at the same pixel. Viewing
  the load node itself shows one number, because there is nothing in between.

The stage coordinate and the raw value are **withheld, not guessed**, whenever the viewed
node re-addresses the pixel grid — a crop, a resample, a z-project, a channel-select, or a
graph with two source files merged. After those, `(x, y)` no longer indexes the field the
stage log describes, and the source pixel at the same index is a different pixel. Position
and microns stay, because they are still true in the node's own grid.

TIFFs carry no stage log, so the stage part is simply absent there.

### Channels and contrast

* One **toggle button per channel**, tinted by its emission colour (or the file's native
  channel colour when it is unambiguous). Toggle channels in and out of the composite.
* **Right-click a channel button to change its colour** — the additive presets (red, green,
  blue, cyan, magenta, yellow, orange, grey), **Custom colour…** for the full picker, or
  *Reset to emission colour*. The choice is remembered **per channel name**, so it follows
  that channel through every re-pull and every node you view, and it re-tints the button,
  its histogram and the composite at once.
* Per channel: a **histogram with draggable LUT handles**. Drag the handles to set the
  window; drag the middle dot up/down for gamma; or type `lo`/`hi` values directly.
* **Auto** re-applies percentile auto-contrast; **Fit** zooms every histogram to its window.
* **⊞ Split** is the split-channel view (as in NIS Elements): the same frame laid out as
  panes — the composite plus one pane per active channel, each labelled in its own colour.
  All panes share one zoom/pan, so they stay registered while you inspect a feature, and
  the grid picks the arrangement that shows the image largest. It needs 2+ active channels,
  costs no re-pull (a display mode only), and overlays stay on the composite pane.

On the GPU path (default), LUT/gamma/colour changes are **uniform updates** — instant, with
zero decode and zero re-upload. Moving T or Z only binds a different texture (uploading a
new plane once). `NODELAB_GL=0` forces the CPU path; a GL failure falls back automatically.

### Overlays

The **◈ Overlays** button (also `Ctrl+Shift+O`, **View → Overlays…**) opens a popup with a
tab per domain: **Points**, **Labels**, **Tracks**, **Mesh** (and a reserved Voxels tab).
Quick on/off checkboxes for Points / Labels / Tracks sit in the channel strip.

* **Points** — golden-star glyphs, per-point palette, configurable arm spread.
* **Labels** — outlines and/or fills, per-object palette (fill hue matches outline hue).
* **Tracks** — one hued polyline per track through its members' `(y,x)` ordered by t, with
  the vertex at the **currently viewed T enlarged** — so scrubbing T walks each track's
  current position.
* **Mesh** — the **cross-section at the viewed Z**: closed loops where the surface cuts the
  plane, in three styles.

Every overlay size is a **screen** size, so an outline keeps its thickness as you zoom into
a label instead of being magnified with the image. Region *fills* scale, because a fill is
the region. A tab's look can be **spread** to the other tabs by role (opacity travels;
"arm spread" does not).

#### One colour per object, not per id

A segmentation re-issues its label ids from 1 on every frame, so "a colour per label id"
makes the same cell flash a different colour at every T step. Colour therefore follows the
**object**, not the id: once a tracker (`track.link` / `track.objects`) has linked the
regions, every member of a track paints in one colour for the whole series — and the track's
polyline is drawn in that same colour, so a trajectory and the cell it belongs to always
match. Untracked regions keep their own id's colour, exactly as before.

Colours are also **de-conflicted between neighbours**. The golden-angle palette spreads
consecutive indices well and says nothing about distant ones — ids 5 and 39 land 4.7° apart
and are indistinguishable — which is harmless across the field and a misreading when they
are two touching cells. So each object's few nearest neighbours are found per frame, and an
object is nudged to another slot only when a neighbour already sits within ~30° of it. An
object that has no conflict keeps the colour it would have had, so turning a tracker on
does not reshuffle the whole field. Above 200 000 structure rows the neighbour pass is
skipped (noted on stderr); the one-colour-per-object rule still holds.

Settings persist as JSON in three layers, later wins: built-in defaults → the project file
shipped in the package (meant to be committed) → this machine's file. Partial files merge,
so a file written by an older build still loads.

### Maximised canvas + mini-map

`Ctrl+Space` (or the `⛶` button, or **View → Maximize node canvas**) gives the whole centre
to the graph and re-homes **the same Viewer** into a mini-map HUD pinned to the canvas's
top-left corner — channels, LUT, playback and overlays all carry across. While maximised,
**clicking any node previews it live** (debounced, so marquee-selecting a chain is one pull),
and the previewed card wears an accent spine. Drag the mini-map header to move it (it
re-anchors to the nearest corner), drag the bottom-right grip to resize, and double-click the
header — or press its dock button, or `Esc` — to put the Viewer back at its old split size.

---

## 8. The Inspector — parameters, units, auto/pinned

Selecting a node builds an editable form from its definition: the 2D/3D switch, the resolved
data-access footprint, and one row per **active** parameter, each with its unit label.

### Physical units, not pixels

Spatial and temporal parameters are authored in **physical units** and converted for you:

| Unit | Meaning |
|---|---|
| `um` / `nm` | lateral, divided by `pixel_size_um` |
| `um_axial` | axial, divided by `z_step_um` (anisotropic sampling) |
| `um2`, `um3` | areas / volumes |
| `s` | seconds, divided by `dt_s` |
| `px` | already pixels |

So a `radius = 0.5 µm` becomes the right pixel count for *this* objective, and the same
graph run on differently calibrated data does the right thing without edits.

### auto vs pinned (the sticky override)

A parameter that can be **derived from metadata** shows an **auto / pinned** toggle and an
`ƒmd` pill on the node card:

* **auto** — the value is derived from the incoming edge's metadata right now. Edit an
  upstream node and the pill updates live, before anything is computed.
* **pinned** — typing a value, or clicking the pin, fixes it. It stays fixed across
  calibration changes and file swaps. Unpinning reverts to auto.

Examples: Deconvolve's PSF derives from emission λ, NA, and pixel/z size (**per channel**);
Spot Detection's radii derive from the diffraction limit; DoG's low σ derives the same way
and its high σ auto-fills at 1.6× low.

### Mode-gated parameters

A parameter that the currently selected mode would ignore is **hidden**, not shown and
discarded. Switch `analysis.histogram_threshold`'s method and you see 1–2 live fields
instead of all eight; switch `analysis.threshold` to `otsu` and the manual `threshold` field
disappears (the method self-derives it). If a control is visible, the selected code path
reads it — that is enforced by a build gate, not by convention.

### Path parameters browse (V2.15)

Any parameter whose value is a path on this machine shows a **Browse…** button beside the
field, so you never type one by hand: `io.load`'s **`path`** (an ND2/TIFF file picker),
`analysis.segment`'s **`sd_model_path`** (a *folder* picker — a locally trained StarDist
model is a directory of `config.json` + `weights_best.h5`) and **`model_path`** (a `*.pt`
CellSAM checkpoint). The dialog starts at the last folder you browsed, so weights and the
image they run on are one click apart. The greyed placeholder says what an **empty** field
means (`io.load` → the synthetic demo; a model path → use the pretrained/published name).
Typing still works, and a pasted Windows *Copy as path* value has its quotes stripped on
commit. Which sockets qualify is a declaration on the socket (`path_kind`), checked by a
build gate — a new path parameter cannot ship browse-less.

A control whose relevance depends on a *wire* rather than a mode stays visible on purpose
(DVC/DIC's `reference_frame` is dead under `previous_frame` but live again the moment you
wire an external `reference` Dataset).

### Choices and tick lists (V2.16)

Where a parameter's legal values are a fixed, published set, it is a **dropdown**, not a
text field — the StarDist and CellSAM checkpoint names, for instance. A value loaded from an
older graph that is no longer offered stays selected rather than snapping to the first entry.

Where a parameter is a *menu of several*, it is a **tick list** that writes the
comma-separated string underneath: Measure's `stats` and `shape`, Object Metrics' `metrics`,
Object Field's `fields`. A misspelt token in those lists used to produce no column and no
error; you can no longer misspell one. Select Channel's `channels` ticks the real channel
**names** from the file and writes the 0-based indices for you (the text field stays above
it, because typing `2,0` is also how you *reorder* channels).

**Every option in every dropdown explains itself (V2.21).** Hovering a number tells you what
it does; a dropdown used to tell you only what the control *was*, and then offer six bare
words — `otsu`, `li`, `yen`, … — whose differences are the entire reason you opened it. So
each option now carries its own line, written *relative to its siblings*: what it assumes
about your data, which way the result moves, what it costs, and what it needs that the others
do not (a `track_id` column, a label raster, a model download) — including when it is refused
outright. They appear on four surfaces, all fed from one source so they cannot disagree:

* the **Inspector row** for the dropdown — the whole option list at once, under a line saying
  what the control selects (a Mode row had no hover text at all before this);
* each **item of the open list**, as you arrow down it — which is when you are actually asking;
* each **tick box** of a tick list, since `feret`, `solidity` and `self_fold` are column names,
  not explanations;
* the **popup menu on the node card**, so the card is not the poor relation.

The **2D / 3D switch** is documented the same way, having previously said nothing except that
3D was unavailable — which left the card's most consequential control (it reconfigures the
node's parameters and caches 2D and 3D separately) as its least explained. Three vocabularies
are written once centrally and reused: the 2D/3D lever, the eleven **domains**, and the
**reducers** — so `median` means the same thing on Stack, Z-Project, Reduce → Scalar and both
Transfer nodes. A build gate (`test_option_docs`) fails if any dropdown in the catalog leaves
an option unexplained, and, like the parameter hovers, the prose is presentation-only: writing
it can never invalidate a cached result.

---

## 8b. Picking parameters off the image (V2.16)

Many parameters answer a question you can *see* — how big is that nucleus, how far does a
cell move between frames, where is the region worth correlating. Sixty of them across thirty
nodes therefore carry a **Pick** button, under the value in the Inspector, and a small
**ring (○) on the node card** beside the same parameter. Both arm the same gesture on the
Viewer.

| Gesture | What you do | Parameters |
|---|---|---|
| **Drag the radius** | press at the centre of a feature, drag to its edge | every µm radius/σ — Gaussian, Median, Morphology, Top-Hat, Unsharp, DoG, Bilateral, Local Threshold's block, Boundary Band, Tessellate's alpha, Flatten Illumination, … |
| **Two rings** | drag the inner, then the outer, over one real spot | Spot Detection & Remove Blobs `min_radius`/`max_radius`, DoG's two σ |
| **Click or draw the size** | click a segmented object to take its measured area, **or** drag to draw a blob and use that | Segmentation & Histogram Threshold `min_area`/`max_area`, `min_volume`/`max_volume`, `min_hole_size` |
| **Eyedropper** | click a pixel; its intensity becomes the level | Threshold, Segmentation's fixed level, Histogram Threshold's low/high, Spot/Particle Detection, Remove Blobs |
| **Measure on the image** | click two points; the value is the distance between them | Track Linking / Track Objects `max_distance`, Particle Detection & watershed seed `min_distance`, Voronoi Cells' max reach |
| **Drag the crop rectangle** | drag one box; **all four bounds** are set at once and everything outside it dims | Crop `y0`/`y1`/`x0`/`x1` |
| **Use the picked Z planes** | tick planes on the Z strip; their span becomes the Z window | Crop `z0`/`z1` (3D only) |
| **Drag the grid** | drag one subset box; the lattice previews at that spacing | DVC/DIC subset size + stride, Segmentation's tiling, Object Field's grid, Temporal Gain's tile |
| **Draw the region** | rect / ellipse / circle / polygon / freehand, with Add, Cut, Invert, Clear and Undo | ROI Mask's `shapes` (which DIC then consumes as its ROI) |
| **Take the histogram** | put the LUT handles where you want them, then Apply | Normalize's percentiles, Histogram Threshold's percentiles, Gamma |
| **Use what's on screen** | commits immediately, no aiming | the viewed channel (Registration, DVC, DIC), the current timepoint, the viewer's channel set |

**While a pick is armed** a bar appears above the image with the instruction, a live readout
of the value as you aim, and Apply / Cancel. **Esc** backs out — first the stroke in
progress, then the pick. **Enter** applies. Single-value gestures commit on release, so a
radius costs one drag; the ROI region does not, because its value is a *list* you keep
building. Zooming still works mid-pick.

**Crop is the clearest case.** Six pixel spin boxes become one drag: the kept rectangle stays
bright, the discard dims, corner ticks mark the extent, and the readout reads
`y 60–260 · x 100–340  (240 × 200 px kept)`. The bounds are a slice with an exclusive end, and
the pick rounds so `[start:end]` keeps **exactly** the pixels the rectangle covered — verified
against real pulls, not just the numbers in the pills. Because one gesture writes four
params, the Pick button appears once (on the first bound) rather than four identical times;
every bound's hover text still names the gesture. Z is separate and 3D-only: a rectangle
drawn on a plane says nothing about depth, so the Z window comes from the planes you tick on
the **Z strip** (their span; all planes if none are ticked).

Everything the pick produces is an **ordinary parameter value**: it is written pinned,
exactly as if you had typed it, it can be unpinned back to auto, and it saves and caches
identically. Values are converted using **this node's** propagated calibration, so a Crop or
Resample upstream is accounted for. On an uncalibrated file the readout says so out loud
rather than quietly reporting pixels labelled µm.

> Picking aims at the image, so pull a node first. If nothing is displayed the status bar
> says so instead of arming a gesture with nothing under it.

### Editing on the canvas

The value pills on a node card are live controls, so you can work without the Inspector open:

| On a card | Does |
|---|---|
| **Drag a number sideways** | scrubs it; **Shift** = fine (÷10), **Ctrl** = coarse (×10) |
| **Click a number** | opens a box to type it (**Esc** abandons, **Enter** commits) |
| **Click a checkbox pill** | toggles it |
| **Click a mode pill** (`▾`) | opens its menu |
| **Click the `ƒmd` badge** | pins the derived value, or unpins back to auto |
| **Click the ring (○)** | arms that parameter's pick |

The pointer tells you which: a horizontal-resize cursor over a scrubbable number, a hand over
everything else, and the pill lights up under the cursor. A scrub re-propagates metadata once
on release, not on every mouse move, so dragging a value on a large graph stays smooth.

---

## 8c. Zooming into a big image — detail on demand

The Viewer draws a whole plane as one texture capped at **`MAX_DISPLAY_DIM`** (4096 px;
override with `NODELAB_MAX_DISPLAY_DIM`). For a camera frame that cap is never reached, so
what you see is every pixel. For a **stitched mosaic** it is: a 13106² canvas overviews at
3276², a quarter of the linear resolution.

**Zoom in and it re-reads.** About 0.1 s after the wheel or the drag settles, the visible
rectangle is read again at the finest pyramid level that fits the same 4096 px budget and
drawn over the overview. On a 49-tile mosaic that means **true 1:1 — every source pixel —
in 0.2–0.45 s**, because a stitched provider serves a window by stitching only the tiles
that window touches rather than rebuilding the whole 172 Mpx canvas.

Three things worth knowing:

* the overview stays underneath, so there is always a complete picture: a patch that is
  late or superseded can only ever make a sub-rect sharper, never blank it;
* the read happens **off the GUI thread**, and a patch for a rect (or a frame) you have
  already left is discarded rather than painted somewhere wrong;
* **it is a display artefact and nothing else.** A node always reads its input at full
  extent and full resolution — zoom state cannot reach a measurement, an export or a
  memo. The GUI gate asserts exactly that (probe §V3: after a detail patch, the node's
  payload still has its full extent and byte-identical pixels).

---

## 9. The 2D/3D lever

Nodes whose kernel has a spatial extent carry a **2D / 3D switch in the card header**. It is
a genuine dimensionality choice, not a hint:

* **2D** — the op runs per `(Y, X)` plane.
* **3D** — the op runs on the whole `(Z, Y, X)` volume, with **anisotropic** axial
  parameters (`sigma_z`, `radius_z`, … appear only in 3D mode, defaulting to the lateral
  value).
* The default is **metadata-adaptive**: `z > 1 ⇒ 3D`.
* The switch is **greyed out when the incoming z is known to be 1** (unknown ≠ 1, so an
  unresolved source never greys it). A saved graph locked to 3D on `z == 1` paints a red
  validation badge.
* 2D and 3D memoize **distinctly**, so flipping the lever recomputes rather than serving the
  other dimensionality's cached result.

Some nodes deliberately have **no** lever:

* Pointwise ops (Gamma, Threshold) are dimension-agnostic.
* A **scope** choice is not a lever: Normalize's `plane / volume / series` is a plain mode.
* A node whose dimensionality is **inherited from its input geometry** has no lever by
  design — `transform.rasterize_field` and `analysis.accumulate_field` read it from the
  field's own provenance, `transform.label_to_points` from the Label instance's own `z_kind`
  (per-plane labels must give per-plane dots), and `analysis.boundary_band` /
  `analysis.tessellate` are 3D-only. A lever there could disagree with the data and silently
  corrupt it.
* A node declaring `stack_of_2d` honesty (Wavelet Denoise, Bilateral) still shows a 3D mode
  but tells you it stacks 2D results — it never pretends to be volumetric.

---

## 10. Domains, layers and the layer picker

A Dataset on the wire is not just pixels: it carries **attribute layers** on **domains**.

**Acquisition-lattice domains** (coarsenings of the image hypercube):
`Voxel (m,t,z,c,y,x)` → `Plane (m,t,z)` → `Frame (m,t)` → `Timepoint (t)` / `Multipoint (m)`
→ `Global ()`, plus the orthogonal `Channel (c)`.

**Detected-structure domains** (produced by analysis): `Label`, `Point`, `Track`, `Mesh`.

That is what the domain chips on Dataset sockets mean. Practical consequences:

* A mask, a distance field, a band raster are **Voxel** layers.
* Segmentation results are a Voxel raster **and** a `Label` table (one row per region).
* Detections are a `Point` table; tracking adds a `Track` table **and** a `track_id` column
  written back onto the member layer.
* `analysis.tessellate` produces a `Mesh` — the one domain carrying topology (which vertices
  form which face) — which `transform.rasterize_mesh` turns back into a filled Label volume.
  `analysis.voronoi` writes one too (in 3D), alongside its territory raster.
* `transform.label_to_points` goes the other way across the structure spine: it **creates**
  Point rows from `Label` regions, which is what the attribute-moving bridges cannot do.
* `transform.transfer_domain` moves a **lattice** attribute between lattice domains
  (coarsening reduces over the dropped axes; refining broadcasts). Structure hops use the
  built-in bridges inside the relevant nodes.

### The layer picker

Nodes that consume a named layer (`mask`, `labels`, `points`, `source`, `mesh`, …) present an
**editable combo offering exactly the layer names actually present upstream** — you pick
instead of retyping. Free text is still accepted for hand-authored graphs. A node is never
offered its own output.

Two behaviours worth knowing:

* The catalog is **not monotone**. An axis-changing node (crop, resample, z-project, stack,
  channel select) **drops** lattice layers whose shape no longer matches. If a mask
  disappears after a crop, that is the rule working — re-derive it after the crop.
* Some producers write layers no socket names (drift's `drift_y`/`drift_x`, extract-boundary's
  `<labels>_boundary`, measure's columns on the label table). Those appear in the picker too.

---

## 11. Spreadsheet & export

The Spreadsheet groups the pulled Dataset's structure layers into **one table per
`(domain, layer)`** — rows are elements ordered by id, columns are attributes with the
coordinate columns (`id,m,t,c,z,y,x`) first. A mesh shows as **three** tables (elements /
vertices / faces), which is exactly how it is stored.

Voxel/lattice attributes are per-voxel rasters — those belong in the Viewer, not here.

**File → Export table…** (`Ctrl+E`):

| Format | Notes |
|---|---|
| `.csv` | one file; several tables are written in long form with leading `domain` / `layer` columns |
| `.parquet` / `.arrow` | one table, same long form, via pyarrow |

---

## 12. Organising a big graph: frames, reroutes, groups, zones

### Reroute dots

**Double-click a wire** to insert a reroute — a real identity pass-through node rendered as
a compact 22 px dot. Use them to keep long wires tidy. Refused on value/field wires (without
dropping the wire).

### Labelled frames — `Ctrl+J`

**Graph → Frame selection…** draws a titled, tinted rectangle *behind* the selected nodes.
It auto-sizes to enclose them, reflows when a member moves, and dragging the frame moves all
members. Frames are **GUI-only** — they never enter the run graph. A frame always has ≥1
member (emptying it removes it); deleting a frame keeps its nodes.

### Node groups — `Ctrl+G` / `Ctrl+Shift+G`

**Graph → Group selection…** collapses a selected **linear** sub-chain (exactly one external
Dataset input and one output) into a reusable group **definition** plus a single purple
`group:<name>` **instance** card wired to the same frontier. **Ungroup** reverses it.

* A source node cannot be grouped — its seed would be buried.
* At run time the instance is **expanded inline**, so the engine, memo and metadata pass need
  no group awareness; downstream domain rails read correctly **through** the opaque instance.
* Groups are nestable, and a group that contains itself is rejected with a clear error.
* Both the instance and its definition are saved, and round-trip.

### Repeat / Simulation zones

**Graph → Wrap selection in Repeat zone…** bounds a body between paired `In`/`Out` markers
with a feedback back-edge, then **unrolls** it into a flat per-iteration chain. Because the
result is an ordinary acyclic graph, memoization works per iteration: re-pulling an unchanged
graph is fully cached, editing the seed or a body param invalidates from that point on, and a
downstream edit leaves the zone cached.

* Iteration 0 is fed by the external Dataset wired into `In`; later iterations get the
  feedback instead. A loop-invariant input wired straight into a *body* node feeds every
  iteration.
* **Simulation** zones are the same mechanism with temporal intent. Put a **`zone.frame`**
  node in the body and iteration *t* processes frame *t* while the `In`/`Out` feedback carries
  cross-frame **state** — a scan over frames.
* A genuinely stateful/random body can be flagged impure, which makes it non-cacheable across
  pulls rather than silently wrong.

---

## 12b. Docking: bake a chain to disk and free the memory (V2.18)

On a large series, a long chain gets expensive in two ways at once. Every realized
intermediate is **held** — a mask or a label raster is a full `(M,T,Z,C,Y,X)` array in RAM,
not a lazy provider — and every time one is evicted, getting a plane back means **re-running
the whole chain above it**. Past a certain point, adding one more node means waiting for
twenty again.

A **Dock Data** node (Nodes palette → `io`) cuts that. Drop it mid-chain, press **Bake**, and:

* everything above it is computed **once** and written to a checkpoint folder on disk;
* the dock then behaves like a freshly loaded file — downstream nodes read decompressed
  pixels off disk instead of recomputing;
* the nodes it replaced **grey out**, are dropped from the run graph entirely, and their
  cached results are released from memory.

You keep building downstream at the cost of a fresh load, not the cost of the whole pipeline.

### What gets baked

The whole Dataset, not just the picture: the image pyramid, every **Voxel** layer (masks,
label rasters), the coarse lattice attributes (Frame/Plane/Channel/Global), every structure
table (Label / Point / Track / Mesh) with its `z_kind` provenance, and the calibration. So a
dock placed *after* Segmentation or Track Objects keeps its masks and tables — the
spreadsheet and export read exactly what they read before.

Full-size Voxel layers come back **memory-mapped**, so a docked segmentation costs no RAM;
the pages you touch are paged in by the OS and are not counted against the memo budget.

### Precision — you have to choose

There is no default, on purpose. Filter nodes compute in **float64**, so baking "what was
computed" is 4× the bytes of the original uint16 file.

| Choice | When |
|---|---|
| `float32` | The usual answer after a filter chain. Half the size of float64, ~7 significant digits — beyond what a 12-bit sensor recorded. |
| `float64` | Bit-identical to running live. Pick it when you need exactness more than disk. |
| `uint16` | Smallest, but only correct while the values are still camera counts. Normalized `[0,1]` or negative data is **refused**, not quantized. |

Integer and boolean data — label rasters, masks, a raw camera frame — is always stored as
itself, whatever you choose.

### Baking

**Run → Bake selected dock… (`F6`)**, the **Bake** button in the Inspector, or the node's
right-click menu. You are told the range, the precision and the folder before it starts.

* **Range** is the whole series by default. *Bake selection only* bakes just the frames the
  Viewer's M/T/Z strips have picked — a shortcut for checking a chain, since the checkpoint
  then holds a **truncated** series that every downstream node will run on.
* **Where** is `<your-graph>.docks/<node-id>/`, beside the saved file. `NODEGRAPH_STORE_DIR`
  relocates it exactly as it relocates an ingest store. Set the *Dock folder* parameter to
  point at an existing bake instead — to share one between graphs, or to put a big dock on
  another drive.

### After it is baked

The dock's card reads **DOCKED** and the chain behind it dims to **DORMANT**. A node feeding
both a docked dock *and* a live branch is **not** greyed — it still runs for that branch.
Dormant nodes stay fully visible and editable, and you can still select one and pull it; it
just recomputes on demand, since nothing else is asking for it.

**Edit something upstream and nothing happens behind your back.** The dock turns amber and
says *stale*; it keeps serving the old bake until you press **Re-bake**. Changing the
precision marks it stale the same way. **Un-dock** runs the chain live again and leaves the
checkpoint on disk, so re-docking is instant.

Docks chain: bake one, add ten nodes, bake a second. Editing above the first dock does not
stale the second — the first one's frozen bake stands between them.

### It survives save/load

The folder is stored **relative** to the graph whenever it sits beside it, so a project
directory can be moved or handed to someone else and the docks stay attached. A loaded graph
comes back already docked, and the axes, calibration and layer catalog every downstream node
reads are seeded from the checkpoint's manifest. Batch runs of a saved graph
(`headless_engine`) skip the docked chains too, rather than silently recomputing hours of
work.

If a bake is interrupted, the folder reads as *absent* rather than half-valid — the dock says
so and offers to bake again.

---

## 12c. Iterating a parameter: sweeps and searches (V2.19)

A **Repeat zone** iterates *data*. The **Iterate** node (`flow.iterate`, the `flow` category)
iterates *parameters*: it runs the chain in front of it once per value and keeps one result.

### Wiring one up

1. Drop an **Iterate** card after the chain you want to tune.
2. Drag from its **V0** output onto the parameter you want to sweep — any spin box, text
   field or **dropdown** on any node downstream of it. (Mode dropdowns grow a port as soon as
   an Iterate node exists in the graph; the 2D/3D lever deliberately does not, because
   sweeping it would produce errors rather than comparisons.)
3. Wire the **end** of that chain back into the Iterate node's **Collect** input.

Those two wires close a loop on the canvas, which is intended — the driver wire is not a data
route, so it never makes the graph cyclic. Everything between the driven node and Collect is
the **iterated chain**, and it is what gets run once per value.

### Choosing the values

Each variable picks a **source**:

| Source | Fields | Use it for |
|---|---|---|
| `list` | comma text | explicit values, and the only source that can sweep **text** (`otsu, li, yen`) |
| `linear` / `log` | from · to · steps | an even range; `log` for anything spanning decades |
| `around` | centre · ±% · steps | **leave the centre empty** and it anchors on the target parameter's own value — its override if you pinned one, otherwise the value its metadata rule derives from *this* file's optics. That is what keeps a saved sweep meaningful on the next file. |

Raise **Variables** to sweep up to four at once. **Combine** is `grid` (every combination) or
`zip` (the i-th value of each). The panel shows the resolved iteration count before you run
anything, and refuses past 64.

### Keeping one

**Preserve** decides what flows downstream:

* **picked** — the iteration you chose. This is the *cheap* one: only that iteration is
  computed, so a finished graph costs one run, not N.
* **first** / **last** — the ends of the sweep.
* **best** — the one with the highest (or lowest) **metric**. A metric is a `Global` scalar,
  which is what the **Reduce → Scalar** node (`analysis.reduce_scalar`) exists to produce:
  put one inside the iterated chain, point it at a domain + column + reducer (`count` on any
  Label column gives the object count), name it, and name that same name in the Iterate
  node's Metric field.

The output also carries `sweep_<param>` and `sweep_metric` as Global scalars, so a later node
can read which value won.

### Comparing them

Press **Run sweep** in the Iterate panel. Every iteration is computed, the results table
fills in, and the Viewer grows an **ITER** strip below M/T/Z — drag it to flip between the
results (instantly; they are all cached), and stop on the one you want. **Stop sweeping**
returns to computing only the kept one; nothing is thrown away.

> **Scope it first.** Every iteration is a full run of the chain. Turn on **F9
> troubleshooting** and pick the frame you are tuning against, or a six-point sweep on a
> 49-position series is six full analyses. The panel says so in amber when the scope is off.

### Searching instead of sweeping

Set **Mode** to `feedback` and the node stops enumerating and starts *searching* one numeric
variable inside its range:

* **golden** — golden-section search for the value that maximizes (or minimizes) the metric.
* **secant** — drive the metric *to* a **target**: "which threshold gives 500 objects".
  **Tolerance** decides the `sweep_converged` flag it stamps; it does not stop the search
  early (every probe always runs).

Feedback is one variable only, needs a range rather than a list, and cannot drive a dropdown
or a parameter that changes the image's axes — those values have to be known before the run.

### What it refuses, and why

Each of these would otherwise produce a plausible, wrong answer rather than an error:

* a driven node that is not upstream of Collect (the sweep would change nothing);
* a **docked** Dock inside the chain (every iteration would read the same baked pixels);
* another Iterate node inside, or overlapping, this one's chain;
* one parameter driven by two Iterate cards;
* a branch leaving the chain to somewhere outside it (which iteration would it read?);
* the 2D/3D lever, and more than 64 iterations.

---

## 12d. Editing a node while NodeLab is running (V2.20)

You do not have to restart to change a node. Edit the node's own file under
`nodegraph/catalog/`, or the kernel behind it under `nodegraph/kernels/`, save, and the open
window runs the new code on the next pull — same graph, same window, same already-ingested
file.

**Each node is its own module.** `enhance.gamma` is `nodegraph/catalog/enhance/gamma.py`; the
op key *is* the module path. Shared helpers live in `nodegraph/catalog/_shared/`, grouped by
concern, and `nodegraph/nodes.py` is now a 70-line facade that just re-exports the public
names. This is what makes the reload worth using: a node's memo key carries a fingerprint of
its own dependencies, so **editing one node re-keys one node**. Measured on the real catalog:
69 nodes, 65 distinct fingerprints — one edit recomputes one node and leaves the other 68
sitting on their cached results. Before the split there was one module and one fingerprint,
so any edit threw away everything.

**The ⟳ button in the Nodes palette** — beside the search box — re-reads the whole node list
from disk. Use it when the *set* of nodes changed: it picks up a node whose `.py` you created
while NodeLab was open (which nothing else can — an un-imported file is in no list and simply
never appears), retires one whose file you deleted, and reloads any that changed. A new node
arrives as a full citizen: registered, in the palette, placeable, and code-fingerprinted, so
editing it afterwards re-keys only itself. A new file that does not compile is reported rather
than half-registered.

**The ⟳ button in the Properties panel** — beside the node's title — re-reads *that node's*
`.py` from disk, re-registers it, and rebuilds both the panel and the card. Use it whenever
the panel disagrees with your editor: it is unconditional, so unlike the menu action it never
answers "nothing changed". Only that node recomputes on the next pull; anything sharing its
code reloads with it. It is disabled, with the reason in its tooltip, for a group instance or
a GUI-layer op (`io.load`, `view.viewer`, `io.dock`), which are not reloadable.

Note that **re-selecting a node does not re-read anything** — selection just points the panel
at the card you clicked. That is what the ⟳ button is for.

**Auto-reload is also on by default.** Saving is usually enough: the status bar reports
*"Reloaded 1 module, 12 re-keyed"* a moment later. Turn it off under
**Run → Auto-reload on file change** if you would rather choose the moment, reload everything
that changed with **`Ctrl+R`**, or reload one node with ⟳.

What a reload does, in order:

* **Only the modules whose contents changed** are re-executed, plus anything that depends on
  them. A touched-but-unchanged file is skipped, so an editor that writes on a timer costs
  nothing. Edit a shared helper and its users are re-executed too — their parameter sockets
  are *built* from that helper, so nothing less would refresh them.
* **Nodes whose code changed recompute; everything else keeps its cached results.** This is
  the part that makes it worth using: a source's decoded pixels, its store, and every
  untouched upstream node stay warm, so you pay only for the node you edited. Ingesting a
  large ND2 is *not* repeated.
* **The canvas catches up.** Placed cards re-read their definition in place, so a socket you
  added appears on the card you already wired, at the same position, with its wires intact.
  The palette and the Inspector rebuild too — new params, new units, new hover text.
* **A node type you deleted leaves the palette.** A card already placed for it stays on the
  canvas with its wires and cannot be pulled, so putting the definition back restores it
  rather than costing you the graph.

Its limits, which are worth knowing before you rely on it:

* **A syntax error is reported and ignored.** Files are compiled before anything is swapped,
  so a half-typed save leaves the session on the code it already had. `Ctrl+R` shows the file
  and line; an auto-reload only writes to the status bar.
* **A reload is held back while a pull is running** and applied when it finishes — swapping a
  compute out from under a working thread is not safe. The status bar says
  *"Node reload deferred"*.
* **A kernel or shared-helper edit re-keys exactly its users** — the nodes that import it.
  `_shared/map_image.py` (the tiling/halo policy) is the widest at 14 of 69 nodes; most are
  far narrower.
* **Editing the engine re-keys everything, and does not reload.** `nodegraph/streaming.py`,
  `field.py` and friends are genuine dependencies of every compute, so an edit correctly
  invalidates the whole memo — but making it *live* still needs a restart. Only node modules,
  the shared prelude, and kernels reload.
* **A baked dock is not invalidated.** A dock is a frozen artefact you asked for by name;
  re-bake it (`F6`) if the code behind it changed.
* If a file fails *while executing* (not a syntax error — an exception in its module body),
  Python cannot fully undo that. The catalog is put back, but the message says to restart,
  and you should.

### Adding or moving a node

* One node, one file: `nodegraph/catalog/<category>/<name>.py`, where the path matches the op
  key. A module may register a *family* (the four `zone.*` boundary markers share one file
  because they share one compute).
* **Dropping the file in is enough** — the catalog is resolved against the filesystem, so the
  node appears at the next launch, or immediately via the palette's ⟳. Nothing else to edit,
  and nothing to forget.
* Optionally add it to `MODULES` in `nodegraph/catalog/__init__.py` to **place** it. That list
  is the curated registration order — which the link-drag search menu enumerates and
  `scripts/_catalog_snapshot.py` gates. A node that is not listed still loads; it just
  registers last, because its position is genuinely unknown and inventing one would be a
  guess presented as a fact.
* Four import rules, enforced by `test_catalog_import_hygiene`: no importing
  `nodegraph.nodes`; no bare `from nodegraph import X` (use the leaf module); import
  `_shared` **submodules**, never the `_shared` package; and keep `__init__.py` files
  logic-free. Each one, if broken, silently collapses per-node granularity back to
  catalog-wide — the nodes keep working and the memo quietly throws everything away.

Headless equivalent:

```python
from nodegraph import hotreload
hotreload.prime()                     # baseline, right after importing nodegraph.nodes
...
report = hotreload.reload_nodes()     # never raises for a bad node file
print(report.summary(), report.removed, report.added)
```

---

## 13. Saving & loading

`*.nd2graph.json` (`Ctrl+S` / `Ctrl+Shift+S` / `Ctrl+O`, **File → New** re-welcomes).

* The **structure** — nodes (op_key, params, modes), every edge including zone back-edges,
  zones, and group definitions (recursively) — is written by the engine's serializer.
  Output is deterministic and key-sorted, so saved files diff cleanly.
* **GUI-only state** rides in a top-level `ui` object the headless loader ignores: canvas
  positions, mute/collapse, frames, the source node's display title, its captured channel
  descriptors, and the sticky `__locked__` pin list.
* A **Dock**'s state, its checkpoint folder and its bake record are ordinary node
  params/modes, so they are part of the structure, not the `ui` extras — a batch run of the
  saved file skips the docked chains exactly as the GUI does. The folder is written
  **relative** to the graph whenever it sits beside it, so the project folder can be moved.
* Loading validates: an unknown or absent `format_version`, or malformed structure, is a
  clear error rather than a half-loaded graph.
* Files from the retired first-generation editor (`.nd2s_pipeline.json`) **cannot** be
  opened, and no importer will be built.

---

## 14. Keyboard & mouse reference

### Menus

| Shortcut | Action |
|---|---|
| `Ctrl+N` / `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` | New / Open / Save / Save As |
| `Ctrl+L` | Load ND2/TIFF file(s)… — multi-select drops one source card per file |
| `Ctrl+I` | **Ingest all source files** — every loaded ND2/TIFF to disk, several at a time |
| `Ctrl+E` | Export table… |
| `Del` / `Backspace` | Delete selected nodes / wires / frames |
| `Ctrl+X` | **Dissolve** — delete and reconnect the chain |
| `Ctrl+A` | Select all nodes |
| `F5` | Pull selected node |
| `Shift+F5` | Pull viewed node again |
| `F6` | **Bake selected dock** — freeze everything above a Dock Data node to disk |
| `F9` | **Troubleshoot: picked frames only** — scope every pull to the picked M/T/Z boxes (or the viewed frame) |
| `Ctrl+R` | **Reload node code** — run the node/kernel files as they are on disk now, without restarting |
| `Ctrl+J` | Frame selection |
| `Ctrl+G` / `Ctrl+Shift+G` | Group / Ungroup selection |
| `Home` | Fit graph |
| `Ctrl+Space` | Maximize node canvas (`Esc` to leave) |
| `Ctrl+Shift+O` | Overlays… |

### Canvas

| Input | Action |
|---|---|
| Double-click a **node** | pull it and show it in the Viewer |
| Double-click a **not-yet-ingested source** | ingest that file (concurrently with any others), then show it |
| Double-click a **wire** | insert a reroute dot |
| Drag a socket → socket | connect (green ring = valid, red = invalid) |
| Drag a **connected** non-multi input | detach and re-drag from its source |
| Drag a socket → empty canvas | link-drag search popup |
| Drop a palette node **onto a wire** | splice it in |
| Hover a card → `✕` | delete that node |
| Drag a value pill sideways | scrub it (`Shift` fine ÷10, `Ctrl` coarse ×10) |
| Click a value pill | type the value (`Esc` abandons, `Enter` commits) |
| Click a mode pill (`▾`) / checkbox pill | open its menu / toggle it |
| Click the `ƒmd` badge | pin the derived value, or unpin back to auto |
| Click the ring (`○`) | pick that parameter off the image |
| Right-click a node / wire / frame | context menu (Delete, Dissolve, …) |
| `M` | mute / unmute selected |
| `C` | collapse / expand selected |
| `Esc` | cancel a wire drag; leave maximised canvas |
| Drag empty space / wheel | pan / zoom |

### Viewer

| Input | Action |
|---|---|
| Wheel / drag | zoom / pan the image |
| Hover a pixel | read its position, microns, absolute stage coordinate and every shown channel's value (plus the raw file value when the node changed it) |
| M/T/Z strip drag, wheel, `←`/`→`, `▶` | scrub / play that axis (fps spinner per axis) |
| `Ctrl`+click / drag a box | pick frames (M/T) or planes (Z) for the `F9` run scope (`Shift`+click extends, right-click for all / invert / clear) |
| Channel button | toggle that channel in the composite |
| Right-click a channel button | set that channel's colour (presets / picker / back to emission) |
| `⊞ Split` | split-channel view: composite + one pane per channel, shared zoom/pan |
| Drag LUT handles | set the window; drag the middle dot for gamma; or type `lo`/`hi` |
| `Auto` / `Fit` | re-apply percentile auto-contrast / zoom histograms |
| While a **pick** is armed | click / drag as the bar instructs; `Esc` cancels (stroke first, then the pick), `Enter` applies |
| `◈ Overlays` | overlay settings popup |

---

## 15. Node reference

Notation: **lever** = has the 2D/3D header switch · **modes** = in-body dropdowns ·
`µm`/`µm²`/`µm³` = the parameter's unit · *(dep)* = needs an optional package.

### Source & channel

| Node | `op_key` | What it does |
|---|---|---|
| Load (GUI source) | `io.load` | The pipeline source. Resolves `path` to a lazy provider (ND2/TIFF → planar-block `.b2nd`), or a synthetic fallback when empty. Grows per-channel output sockets. |
| Select Channel | `channel.select` | Subset/reorder the channel axis; the per-channel emission list follows in lockstep. |
| Split Channels | `channel.split` | Fan a multi-channel Dataset into per-channel outputs; `out` still carries the full bundle. |
| Dock Data | `io.dock` | Bake everything upstream to a disk checkpoint, then serve it as a new source — the chain behind it greys out and is released from memory. See [§12b](#12b-docking-bake-a-chain-to-disk-and-free-the-memory-v218). |
| Reroute | `rr.reroute` | Identity pass-through for wire tidiness (created by double-clicking a wire; hidden from the palette). |
| Viewer tap | `view.viewer` | Pure pass-through inspection tap. |

### Enhancement (18)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| Gaussian Blur | `enhance.gaussian` | `sigma` µm, `sigma_z` µm_axial (3D) | lever; tiled in 2D |
| Median | `enhance.median` | `radius` µm, `radius_z` | lever; edge-preserving |
| Morphology | `enhance.morphology` | `radius` µm, mode `erode/dilate/open/close` | lever |
| Top-Hat | `enhance.tophat` | `radius` µm, `variant white/black` | lever; background flattening |
| Difference of Gaussians | `enhance.dog` | `low_sigma` µm (diffraction-derived), `high_sigma` (0 ⇒ 1.6× low), `rescale` | lever; blob band-pass. Raw output is a **difference** image with negatives; `rescale` clips at 0 and stretches to the declared full range (Cell-Tracker's DoG) |
| Unsharp Mask | `enhance.unsharp` | `radius` µm, `amount` | lever |
| Morphological Gradient | `enhance.morphological_gradient` | `radius` µm, `blend`, `rescale` | lever; edge map. `blend` mixes the original back over the edges (Cell-Tracker's plugin = `rescale` on, `blend` 0.5) |
| TV Denoise | `enhance.tv_denoise` | `weight` | lever; **true 3D** |
| Wavelet Denoise | `enhance.wavelet_denoise` | — | lever; **stack-of-2D** (declared) |
| Non-Local Means | `enhance.nlm` | `h`, `patch_size` px, `patch_distance` px | lever; **true 3D** |
| Bilateral Denoise | `enhance.bilateral` | `sigma_spatial` µm, `sigma_color` | lever; **stack-of-2D** (declared) |
| CLAHE | `enhance.clahe` | `clip_limit`, `tile_grid` | lever; rescales back into the input range. `tile_grid` = contextual regions per axis (8 = Cell-Tracker's OpenCV default; was pinned to 4 before V2.13) |
| Gamma | `enhance.gamma` | `gamma`, scale `full_range/plane_max` | no lever (pointwise). `full_range` (default) takes the power law over `2**bit_depth-1` — one fixed transfer curve for the series, Cell-Tracker's behaviour; `plane_max` is the adaptive pre-V2.13 per-plane normalisation |
| Normalize | `enhance.normalize` | `low_pct`, `high_pct`, scope `plane/volume/series` | **drops `bit_depth`** — output is no longer integer counts |
| Deconvolve | `enhance.deconvolve` | `na`, `emission_nm` nm, `z_step_um`, `iterations` | lever; Richardson–Lucy with a PSF derived **per channel** from optics |
| **Flatten Illumination** | `enhance.flatten_field` | method `subtract/subtract_mean/divide_mean/ratio`, reference `per_plane/time_averaged`, `sigma` µm, `bg_floor` | no lever (lateral background). Ports Cell-Tracker's Background Subtract + Spatial Flatness + Local Contrast; `ratio` **drops `bit_depth`** |
| **Temporal Gain** | `enhance.temporal_gain` | reference `series_mean/rolling_mean/exponential_fit`, extent `global/tile/gaussian`, `window`, `threshold`, `tile_size` µm, `local_sigma` µm | no lever; per (m,z,c) T-series. Ports Bleach Correction + Temporal Fold Correction (+ the exponential decay model CT's UI promised but never ran). Refuses T=1 |
| **Remove Blobs** | `enhance.remove_blobs` | detector `log/dog`, action `zero/interpolate/median`, `min_radius`/`max_radius` µm (per channel), `threshold`, `expand`, `fill_radius` µm | per-plane, no lever (sensor/glass artifacts are per-plane). Ports Blob Subtract |

### Segmentation & analysis (24)

| Node | `op_key` | Key controls | Produces |
|---|---|---|---|
| Threshold | `analysis.threshold` | method `fixed/otsu/li/yen/triangle/mean`, `threshold` (fixed only), `name` | Voxel mask |
| Multi-Otsu | `analysis.multiotsu` | `classes`, `name` | Voxel class raster (0..K-1) |
| Local Threshold | `analysis.threshold_local` | `block_size` µm, `offset`, `name` | Voxel mask (uneven illumination) |
| Connected Components | `analysis.label` | `mask`, `connectivity` (8 / 26 default), `name` | Voxel raster **+ Label table** |
| **Segmentation** | `analysis.segment` | method **threshold** (`level` otsu/li/yen/triangle/mean/fixed, `connectivity`) · **watershed** (optional `mask` layer, `min_distance` µm) · **stardist** (`prob_thresh`, `nms_thresh`, `scale`, `model_name`) · **cellsam** (`bbox_threshold`, `cellsam_model`, `model_path`, `normalize`, `postprocess`, `remove_boundaries`, `tile`+`tile_size`/`tile_overlap` px, `fast`); shared: `name`, `fill_holes`, min/max **area µm²** (2D) or **volume µm³** (3D) | Voxel label raster **+ Label table**; lever: 2D = per-plane instances, 3D = z-connected *(the two learned methods refuse 3D)*. CellSAM `fast` = **~5× on a GPU** (batched mask decoder); not bit-identical, so it is off by default — see below |
| Distance Transform | `analysis.edt` | `mask`, `name` | µm distance field (anisotropic in 3D) |
| Measure | `analysis.measure` | `labels`, `stats`, `shape`, optional **`raw`** Dataset | per-label stats on the Label table. `shape` adds µm-aware regionprops geometry (`eccentricity`, `perimeter`, `solidity`, `extent`, `axis_major`, `axis_minor`, `orientation`); the 2-D-only three are refused on a 3D Label table |
| Histogram Threshold | `analysis.histogram_threshold` | method `single/hysteresis/percentile/relative` × direction `below/above/between/outside`, morphology cleanup, area filters µm², optional `raw` | mask + Label raster + region table (2D) |
| Spot Detection | `detect.spots` | `min/max_radius` µm (+ `_z`), method `log/dog`, polarity `bright/dark` | Point table; lever |
| Particle Detection | `detect.particles` | `min_distance` µm, `threshold`, `min_intensity`, `min_size`, `subpixel`, mode `log/components` | Point table; lever |
| Extract Boundary | `analysis.extract_boundary` | `labels`, `name` | boundary Points (2D contours / 3D surface verts) |
| Boundary Band | `analysis.boundary_band` | method `dilation/edt`, `band_voxels`, `band_um` µm, `include_neighbors` | outward band Label raster; **3D only** |
| Cluster Points | `analysis.cluster_points` | method `gmm/kmeans`, `n_clusters`, `relax_pct`, `n_init` | per-point cluster-id column *(dep: scikit-learn)* |
| **Object Metrics** | `analysis.object_metrics` | members `label/point`, `metrics` (velocity, speed, neighbors, divergence, curl, frame_fold, self_fold), `n_neighbors`, `intensity` | per-object columns onto the member layer: µm/s velocity+speed, µm neighbour distances, 1/s local divergence+curl, frame- and self-normalised intensity fold. Ports Cell-Tracker's `compute_spatial_metrics` + self-fold + `mean_velocity`; **2D** |
| **Object Field** | `analysis.object_field` | members `label/point`, `fields` (density, mean_area, intensity, fold_change, velocity, speed, divergence, curl), `grid_step` µm, `smooth` µm, `intensity`, `name` | a **Point layer on a grid** (objects/µm², µm², µm/s, 1/s) → feed `transform.rasterize_field`. Ports `compute_spatial_fields`; **2D** |
| Tessellate | `analysis.tessellate` | boundary `convex_hull/alpha_shape/voronoi/label_surface`, `alpha_um` µm, `min_points`, `labels`, `iso_level`, `decimate` | **Mesh** (one closed surface per label + analytic volume/area/density); 3D |
| **Voronoi Cells** | `analysis.voronoi` | `points` (the dots), `region` (the areas), bound `per_region/mask/frame`, `max_distance_um` µm, `name`; 3D-only `mesh` `build/skip` + `mesh_name`, `decimate` | **One territory per dot, clipped to the areas** → Voxel Label raster + table (`area`, `density` = 1/µm² or 1/µm³, `point_id`, `region`), plus a **Mesh** of the cell surfaces in 3D. Nearest-seed in **µm** space, so anisotropic z cannot stretch a cell; lever |
| ROI Mask | `analysis.roi_mask` | `shapes` (**drawn** on the image — see [§8b](#8b-picking-parameters-off-the-image-v216)), `name` | boolean Voxel ROI mask; empty ⇒ whole frame |
| DVC (ALDVC) | `analysis.dvc_field` | reference `fixed_frame/previous_frame` + optional external `reference` Dataset, `subset_size/spacing` px, `search_radius`, correlation `zncc/phase`, strain `infinitesimal/green-lagrange/almansi/hencky`, `newFFTSearch` | Point displacement+strain field (µm); lever |
| Accumulate DVC Field | `analysis.accumulate_field` | `source`, `name` | cumulative Lagrangian disp+strain; **inherits** its config from the upstream DVC (refuses `fixed_frame`/no provenance) |
| DIC (pyALDIC) | `analysis.dic_correlate` | reference lever + optional `reference` Dataset, solver `aldic/local`, `winsize/winstepsize/search_range` px, ICGN/ADMM iterations, smoothness, `compute_strain`, optional `roi` mask | 2D Point displacement (+ optional strain) field *(dep: al-dic)*. Solves the whole T stack in **one** solver call per (m, z) — 2.3–3.2× faster than the per-frame pairing it replaced, for bit-identical displacements. Validated against pyALDIC's own synthetic suite (`scripts/dic_synthetic_bench.py`) |
| Track Linking | `track.link` | target `label/point`, `max_distance` µm, `iou_threshold` | Track membership (IoU overlap / nearest neighbour) |
| **Reduce → Scalar** | `analysis.reduce_scalar` | `domain` (voxel…track), `source` column/layer, `table` (only when a name is ambiguous), `reducer` `count/mean/sum/max/min/median`, `name` | ONE number on the **Global** domain — the score an Iterate node's `best` mode compares ([§12c](#12c-iterating-a-parameter-sweeps-and-searches-v219)), and the only node in the catalog that writes Global. An empty domain is a result (`count` → 0), not an error; a wrong column name still is |
| Track Objects | `track.objects` | target `label/point`, method `centroid/serialtrack/topology/fingerprint/overlap` + per-method params | Track table **+ `track_id` write-back**; *(dep: numba/pandas)*. `serialtrack` is 1–2 orders of magnitude slower than the rest |

### Registration (2)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| Drift Correction | `align.drift` | — | phase cross-correlation across T; stores per-frame shift as Frame attrs (`drift_y`/`drift_x`) |
| Registration | `registration.stabilize` | model `translation/euclidean/affine/feature`, reference `first/previous/mean/template`, `ref_channel`, `upsample`, `highpass_sigma` px, `min_confidence` | **register once on a reference channel, apply to all** — preserves colocalisation |

### Transform & utility (10)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| **Transform** | `transform.rigid` | `channels` (tick by name), `shift_x`/`shift_y` µm (+ `shift_z` µm-axial in 3D), `angle` degrees, interp `linear/nearest/cubic`, lever | Move **selected channels** inside a **fixed frame** — translate and rotate in-plane about the centre; what leaves the frame is cropped and what it vacates is zero. Channels not named pass through at their raw position **on the same wire**, so a corrected and an untouched channel arrive downstream addressable voxel-for-voxel — the reason this is one node rather than a split/transform/merge chain. Axes, pixel size and `origin_um` are unchanged (only content moves), but the move IS stamped as sampling provenance, so `analysis.measure`'s `raw` guard sees it. Positive: X right, Y down, angle clockwise as displayed (ImageJ's sign). Voxel layers ride along (integer rasters always nearest, so ids survive); a Label/Point/Track table on the wire is **refused** — put Transform before segmentation |
| Z-Project | `util.zproject` | method `max/mean/sum/min/median/none` | Z→1; drops `z_step_um`, marks `z_collapsed`; `sum` widens `bit_depth`. **`none` is the reset** — the node becomes a pass-through and the full Z stack comes back with `z_step_um` intact, `z_collapsed` unstamped and the pixels untouched, so a downstream lever can go back to 3D without rewiring the graph (it also stamps no sampling provenance and costs nothing — the footprint drops to `tileable`). Streams as a tree-reduce (no plane is ever realized) and forwards the source's display pyramid, so it scrubs on a stitched mosaic; on screen a coarse level of `max`/`min`/`median` is very slightly smoothed (~1% of the display range, measured), while level 0 — everything a node reads, measures or exports — is exact |
| Crop | `util.crop` | `y0/y1/x0/x1` px — **drag one rectangle** ([§8b](#8b-picking-parameters-off-the-image-v216)) (+ `z0/z1` from the Z strip in 3D) | pixel size preserved; lever |
| Resample | `util.resample` | `scale_xy`, `scale_z` | pixel size scales inversely; lever |
| Stack (T→1) | `util.stack` | method `mean/median/sigma_clip/trimmed_mean/max/sum` | SNR stacking; drops `dt_s`; `sum` widens `bit_depth` |
| Stitch (M→1) | `util.stitch` | layout `stage/stage+refine/grid`, blend `feather/max/mean/overwrite`, `flip_x`/`flip_y`, `refine_*`, `grid_cols` | M→1 mosaic from the file's stage log; Y/X **grow to an extent the header reports UNKNOWN** (it depends on the position log, which rides the payload); one canvas plane streamed at a time; refuses a missing/short stage log, an M axis of repeat visits, and any Dataset carrying a structure table |
| Transfer Domain | `transform.transfer_domain` | `from_domain`, `to_domain`, `reducer`, `attr` | lattice ↔ lattice only |
| **Label → Points** | `transform.label_to_points` | `labels`, position `centroid/inside/weighted`, `name` (empty ⇒ `<labels>_points`), optional **`raw`** Dataset | every region → **one dot** at its centre, keeping a `label` column back to the region. `inside` snaps to the nearest region voxel so a C-shaped or annular object still gets a point *inside* it; `weighted` is the intensity-weighted centre of mass. **No lever** — 2D vs 3D is inherited from the Label instance's own provenance |
| Rasterize Mesh | `transform.rasterize_mesh` | `mesh`, `smooth_um` µm, `min_voxels`, `fill_holes`, `name` | Mesh → filled Voxel Label + per-element table (concavity survives) |
| Rasterize Field | `transform.rasterize_field` | method `linear/nearest`, `source`, `prefix` | Point field → full-res Voxel layers; **no lever** (dim inherited from the field) |

### Flow (1)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| Iterate | `flow.iterate` | mode `sweep/feedback`, `variables` 1–4, combine `grid/zip`, preserve `picked/first/last/best`, per-variable source `list/linear/log/around`, `metric`, `index`, search `golden/secant`, `target`, `tol` | Runs the chain in front of it once per parameter value and keeps one result. Drive any param or dropdown, collect the chain's end. See [§12c](#12c-iterating-a-parameter-sweeps-and-searches-v219). |

### Abstraction (7)

`zone.repeat_in` / `zone.repeat_out` / `zone.sim_in` / `zone.sim_out` — zone boundary markers
(pass-through). `zone.frame` — inside a zone body, iteration *t* yields frame *t*.
`group.input` / `group.output` — the group interface markers. `flow.advance` — one step of an
Iterate node's feedback search, minted by the rewrite between two iterations (never placed by
hand, hidden from the palette).

---

## 16. Worked workflows

### A. Segment and measure (the bread and butter)

```
io.load ─ch0→ enhance.tophat ─→ enhance.gaussian
        ─→ analysis.segment(method=watershed) ─→ analysis.measure
                                                  ↑ raw (optional)
```

* Top-hat flattens the background; Gaussian smooths at a µm scale.
* **One Segmentation node does the whole job**: it cuts its own foreground (Otsu by
  default) and splits touching objects, emitting the label raster *and* the region table.
  Swap `method` to `stardist` or `cellsam` and nothing downstream changes — that is the
  point of the single node. Point its `mask` socket at an existing mask layer instead
  (from Threshold / ROI Mask / Histogram Threshold) when you have already made one.
* **Wire the raw source into Measure's optional `raw` input.** Then segmentation happens on
  the enhanced chain you tuned, but the reported intensities come from the pixels the camera
  actually recorded. The mask, labels and thresholds always stay on the main input.
* Geometry must match: a cropped/resampled/z-projected `raw` is refused with both shapes
  named, because reading them voxel-for-voxel would report a neighbour's intensity.
* Pull Measure → Spreadsheet → `Ctrl+E`.

### B. Detect and track puncta

```
io.load ─ch0→ enhance.dog ─→ detect.spots ─→ track.link (point, max_distance µm)
```

Turn on the **Tracks** overlay and play T: each track draws its whole trail with the current
timepoint's vertex enlarged. For heavy-duty tracking swap in `track.objects` and choose a
linker; it also writes `track_id` back onto the member layer, which is what makes the result
usable downstream (nothing consumes the Track domain directly).

### C. Drift, then analyse

```
io.load ─→ registration.stabilize (ref_channel=0, model=translation) ─→ …
```

Register **once** on one reference channel and apply the same transform to every channel and
z — that is what preserves colocalisation. `align.drift` is the lighter phase-correlation
variant; both store the per-frame shift as Frame attributes you can inspect.

### D. Strain fields (DVC / DIC)

```
io.load ─→ analysis.dvc_field (previous_frame) ─→ analysis.accumulate_field
                                              └─→ transform.rasterize_field
```

* `dvc_field` returns a **Point** field at subset centres with displacement in µm plus
  strain; 2D per-plane or 3D volumetric via the lever.
* `accumulate_field` composes `previous_frame` increments into a **cumulative Lagrangian**
  series. It does not ask you to restate the reference mode — it **inherits** it from the
  upstream node's stamped provenance, and refuses a `fixed_frame` field (already cumulative)
  or a non-DVC input.
* `rasterize_field` interpolates the point field up to full-resolution Voxel layers for
  display. It has no 2D/3D lever: it reads the field's own dimensionality, so a 2D per-plane
  field on a `z > 1` image is never misread as a 3D grid.
* For 2D image pairs use `analysis.dic_correlate` instead (it can also take an ROI mask from
  `analysis.roi_mask`). Note the first al-dic call JIT-compiles numba (~5–6 s, once per
  process); every later solve in the session is fast.

  Four things worth knowing before you tune it:
  * **Subset size is the consequential knob, and it is nearly free.** Bigger subsets average
    deformation over more area — measured on synthetic fields, going 16 → 48 px cuts noise
    error 3.4× (0.100 → 0.029 px) and costs 5.3× in spatial resolution (0.097 → 0.516 px on a
    64 px-wavelength field). Wall-clock barely moves. **Grid step** is what costs time: it
    alone sets how many points you get.
  * **Solver `local` is ~2–4× faster and, on smooth fields, just as accurate** — the global
    ADMM step earns its cost when you need kinematic compatibility across a noisy or
    discontinuous field, not on gentle motion. The ADMM-only controls (μ, ADMM iterations,
    displacement smoothing) hide themselves in this mode.
  * **Displacement smoothing is inert below ~1e-3.** The default 5e-4 is indistinguishable
    from 0; 5e-3 cuts noise error 3× but flattens a real 2% strain field 50× worse. Move it
    in decades, and only for visibly speckled fields.
  * **An ROI mask now also shrinks the work**, tightening the correlation grid to the mask's
    bounding box. Points the solver cannot resolve are dropped rather than written as NaN
    rows — the count lands in the `dic_unsolved_points` metadata key.

  `python scripts/dic_synthetic_bench.py` re-runs the ground-truth accuracy suite (pyALDIC's
  own synthetic cases plus sub-pixel/noise/resolution probes) against the live code.

### E. Point cloud → mesh → label volume

```
detect.particles ─→ analysis.cluster_points ─→ analysis.tessellate (alpha_shape)
                                            ─→ transform.rasterize_mesh ─→ analysis.boundary_band
```

`tessellate` builds one closed surface per label with analytic volume/area/density;
`rasterize_mesh` fills it into a Label volume — and because the interior test is derived from
the mesh's own provenance, a concave alpha-shape fills **less** than its convex hull (which is
a real bug this split fixed). `label_surface` boundary mode meshes an existing label raster
via marching cubes instead of a point cloud.

### E2. Voronoi territories — dots inside areas

The other tessellation: instead of one surface per *cluster*, **one cell per dot**, bounded by
an area layer. The classic use is cell territories from nuclei.

```
analysis.segment (nuclei)  ─→ transform.label_to_points (inside) ─┐
analysis.threshold (tissue) ─→ analysis.label ────────────────────┴─→ analysis.voronoi
                                                                       (bound = per_region)
                                                                    ─→ analysis.measure
```

The two inputs are what the node links. `points` are the seeds; `region` is the area the cells
are cut out of, and `bound` decides how strongly:

| bound | each cell may claim… | a region with no dot | use it for |
|---|---|---|---|
| `per_region` | only voxels of the region **its own dot sits in** | stays background | nuclei splitting a cell body / compartment — territories can never cross a boundary |
| `mask` | any non-zero voxel of the layer; every dot competes | is claimed by the nearest dot elsewhere | a field-of-view or tissue mask, where the boundary is the edge of valid data |
| `frame` | the whole plane/volume (no area input) | — | a plain Voronoi of the image |

Three things worth knowing:

* **Distances are microns, not pixels.** On an anisotropic stack a pixel-space partition would
  count one z step as one unit of distance and stretch every cell along z. Cells are cut at the
  true physical midplane instead.
* **`max_distance_um` caps a cell's reach.** At 0 every voxel of the area belongs to someone,
  so one isolated dot in a large empty region reports an enormous territory. Set it near the
  biological cell radius and the surplus stays background.
* **The table records the link.** `point_id` is the seed that owns the cell and `region` is the
  area it came from, so the territory joins back to both. `density` is `1 / area` in µm² (2D) or
  µm³ (3D) — the Voronoi local number density.

`transform.label_to_points` is the usual way to get the dots when the seeds are segmented
objects rather than detected puncta. Prefer its **`inside`** position for Voronoi seeding: a
concave or annular nucleus has its centre of mass in background, and a seed that does not lie
in any region is dropped by `per_region`.

In 3D the node also writes a **Mesh** of the cell surfaces (one closed marching-cubes surface
per cell), so the cells render in the viewer and feed `transform.rasterize_mesh`. Set the
3D-only **Surfaces** mode to `skip` while tuning — meshing every cell is the expensive half,
and the raster plus its table are complete without it.

### F. Rebuilding a Cell-Tracker recipe

Cell-Tracker's `.ctrecipe.json` pipelines map one-for-one onto nodes. Its two shipped
recipes become:

```
First_pass:   enhance.temporal_gain (series_mean, global)      ← Bleach Correction
            ─→ enhance.normalize (plane, low 1 / high 90)      ← Normalize
            ─→ enhance.flatten_field (divide_mean, per_plane)   ← Spatial Flatness
            ─→ enhance.temporal_gain (rolling_mean, global)     ← Temporal Fold Correction

Second_pass: enhance.clahe ─→ enhance.unsharp ─→ enhance.flatten_field (subtract)
            ─→ enhance.median ─→ enhance.unsharp
```

| Cell-Tracker plugin | node | mode |
|---|---|---|
| Normalize | `enhance.normalize` | scope `plane` |
| CLAHE | `enhance.clahe` | — (`tile_grid` 8; **`clip_limit` is on scikit-image's 0–1 scale**, not OpenCV's) |
| Gaussian Blur / Median / Top-Hat / DoG / Unsharp Mask / Bilateral / NLM / Wavelet / TV | the same-named `enhance.*` node | — |
| Gamma Correction | `enhance.gamma` | scale `full_range` |
| Morphological Gradient | `enhance.morphological_gradient` | `rescale` on, `blend` 0.5 |
| Background Subtract | `enhance.flatten_field` | method `subtract` |
| Spatial Flatness | `enhance.flatten_field` | method `divide_mean` / `subtract_mean`, reference `per_plane` / `time_averaged` |
| Local Contrast | `enhance.flatten_field` | method `ratio` |
| Bleach Correction | `enhance.temporal_gain` | reference `series_mean`, extent `global` |
| Temporal Fold Correction | `enhance.temporal_gain` | reference `rolling_mean`, extent `global` / `tile` / `gaussian` |
| Blob Subtract | `enhance.remove_blobs` | — |
| StarDist detection | `analysis.segment` | method `stardist` |
| Topology / Fingerprint / Mask-Overlap tracking | `track.objects` | method `topology` / `fingerprint` / `overlap` |

**Recipe numbers do not transfer verbatim.** Every spatial control here is in **µm**, not
pixels, so multiply Cell-Tracker's value by that dataset's pixel size (a σ of 100 px at
0.65 µm/px is 65 µm). CLAHE's clip limit is on a different scale entirely.

### G. Per-cell and per-region dynamics

```
… ─→ analysis.segment ─→ analysis.measure (stats + shape) ─→ track.objects
   ─→ analysis.object_metrics (velocity, speed, neighbors, divergence, curl, self_fold)
   ─→ analysis.object_field (density, speed, divergence) ─→ transform.rasterize_field
```

`object_metrics` writes per-cell columns back onto the Label layer, so they appear in the
spreadsheet next to area and intensity — velocities in **µm/s**, neighbour distances in
**µm**, divergence and curl in **1/s**. `object_field` answers the Eulerian question
instead ("what is happening *here*"), emitting a coarse Point grid that
`transform.rasterize_field` renders as an image layer for overlay.

Both need what they read: anything motion-derived needs a `track_id` column (so
`track.objects` or `track.link` first), and the fold changes need a measured intensity
column. Both are **2D** and refuse a 3D structure table, exactly as `track.objects` does.

### H. Per-frame simulation with carried state

Wrap the body in a **Sim zone**, put a `zone.frame` node inside it, and set iterations = T.
Iteration *t* processes frame *t* while the `In`/`Out` feedback carries state across frames.
Each iteration memoizes independently, so editing frame 40's parameters does not recompute
frames 1–39.

---

## 17. Headless / scripted use

The engine is Qt-free, and a graph saved by the GUI can be run without PySide6:

```python
import nodegraph.nodes                       # importing registers the 59-node catalog
from nodegraph.serialize import from_json
from nodegraph.groups import expand
from nodelab_v2.ops import headless_engine   # Qt-free by design

graph, zones, groups = from_json(open("my.nd2graph.json").read())
if groups:
    graph = expand(graph, groups)            # inline group instances
if zones:
    from nodegraph.zones import unroll
    graph = unroll(graph, zones)             # flatten Repeat/Sim zones (iterations ride the Zone)

engine = headless_engine(graph, seeds={"load1": my_dataset},
                         meta_seeds={"load1": my_envelope})
result = engine.pull("node_id")              # → a Dataset
```

Notes:

* You supply the source: a headless consumer provides its own seed `Dataset` for each
  `io.load` node (`nodelab_v2.ingest.ingest_nd2(path, store_path)` gives you a provider +
  a metadata envelope). `headless_engine` calls `ensure_ops()` and materializes the
  per-channel taps for you; group expansion and zone unrolling are yours to do (the GUI's
  `document.to_graph` does both).
* `Engine(strict_reads=True)` turns any un-fenced calibration read into a hard error — use it
  when developing nodes.
* `Engine(memo_bytes=…)` caps the memo with a byte-budget LRU (the GUI uses 1 GiB);
  `cache_bytes=…` sizes the streaming tile cache.
* `Engine(observer=fn)` gives you `start` / `progress` / `done` / `cached` / `error` events
  per node.

---

## 18. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "install scikit-learn / al-dic / stardist / numba" | that node is dependency-gated; install the extra from [§1](#1-install--launch) |
| A node shows a **red domain chip** | it requires a domain nothing upstream produced (e.g. Measure needs `LABEL` — add Connected Components) |
| The **3D switch is greyed out** | the incoming `z` is known to be 1. Z-project or a `z==1` file will do that. If a Z-Project upstream is the cause, set its method to **`none`** — the stack (and the 3D switch) come back without deleting the node or rewiring |
| Red validation badge on a card | the graph is locked to 3D but `z == 1` |
| A **mask vanished** after Crop/Resample/Z-Project | the layer catalog is not monotone: an axis change drops lattice layers whose shape no longer matches. Re-derive the mask after the geometry change |
| `analysis.histogram_threshold` refuses the input | it needs raw integer counts; something upstream (Normalize, CLAHE) produced `[0,1]` floats and honestly dropped `bit_depth` |
| Raw-input geometry error on Measure | the optional `raw` Dataset must match voxel-for-voxel; both shapes are named in the message |
| A node finishes instantly with a full bar | it returned a lazy provider; the real cost lands on whichever node reads planes |
| **CellSAM is very slow** (tens of minutes on a big/stitched image) | upstream calls the SAM mask decoder **once per detected cell**, so a 24 GB GPU sits idle on launch overhead. Turn on the CellSAM **`fast`** socket: batched decoder + GPU mask upsample, measured **5.2×** (a 1024² block of 441 cells goes 11.6 s → 2.25 s; a 26-minute stitched mosaic → ~5 min). It is **not bit-identical** — 11 of 441 cells shifted by exactly 1 px, none gained or lost — so leave it off to reproduce a published number. Also check `analysis.segment` is on CUDA (`NODELAB_CELLSAM_DEVICE`), and put a **Dock** after it so tuning downstream nodes does not re-run it |
| **A stitched mosaic looks lower-resolution than the tiles when zoomed OUT** | expected, and only on screen — the pulled data, every measurement and every export are full-resolution. The whole-image overview is decimated once to `MAX_DISPLAY_DIM` (4096 px, `NODELAB_MAX_DISPLAY_DIM`), so a 13106² mosaic previews at 3276². **Zoom in and it sharpens**: the visible rect is re-read at full detail ~0.1 s after the wheel/drag settles (§8c). If it stays soft after zooming, the detail read is failing — check the status line |
| **Stitch scrubs slowly**, as if each frame loads on the spot | a stitched canvas is far bigger than a source plane, so the Viewer serves it from the source's pyramid. Put **Stitch early** — directly after the load, before any enhancement. Most computed intermediates have no pyramid, so a stitch downstream of one must build the full-resolution canvas for every frame (measured 0.9 s/frame `overwrite`, 2.9 s `feather` at 49×2048², versus 0.09/0.22 s straight off the load). `overwrite` is also ~2.5× cheaper than `feather` if you are only previewing. **Z-Project and Stack are the exceptions** (V2.20): an axis reduce forwards the pyramid, so `load → Z-Project → Stitch` keeps its coarse levels and scrubs |
| **Which order — Stitch→Z-Project, or Z-Project→Stitch?** | Both stream at the ideal cost (every source voxel read once) and both keep the pyramid, so this is a question about the *numbers*, not speed. With blend `overwrite` or `max` the two are **bit-identical**, always. With `feather`/`mean` they agree exactly as long as overlapping tiles differ only by a **z-independent** factor — vignetting, exposure, gain all cancel — and diverge when they disagree *as a function of z*: measured on a ±0.45-z-step per-tile focus offset, 8% of overlap pixels differ (peak 3% of range); on 5% independent per-tile noise, 41% differ (peak 1.6%). The direction is fixed by convexity — `Z-Project(Stitch) ≤ Stitch(Z-Project)` — so projecting **last** averages the tiles before taking the max and suppresses independent noise in the seams, while projecting **first** takes each tile's best focus and then averages, which is brighter there. Outside the overlaps they are identical either way. Prefer **Stitch → Z-Project** unless you specifically want per-tile best-focus |
| **Z-Project on a Stitch hangs** (fixed in V2.20) | it did, and badly. The projection tiled the mosaic, and each of its tiles re-stitched the whole canvas from the source planes without caching — `n_output_tiles × Z × n_positions` full plane reads where `Z × n_positions` was needed, a **~350× multiplier** on a 26×26-tile mosaic. Two smaller faults rode along: a windowed stitch also read every tile lying *before* the window (only the far edge was checked), and the projection had no pyramid, so the Viewer could only read level 0. All three are fixed and the numbers are unchanged — level 0 is bit-identical. If you are on an older build, project **before** stitching (Z→1 per tile, then one mosaic) as the workaround |
| Viewer went black after docking/undocking | the GL context was recreated; it rebuilds and replays automatically. `NODELAB_GL=0` forces the CPU path |
| First pull on a big ND2 is slow | the one-time `.b2nd` planar-block ingest next to the file; later pulls reopen it lazily |
| Tuning a parameter on a long series is painfully slow | press `F9` — pulls then analyse only the frames you picked, or the one you are on ([§6](#troubleshooting-mode-analyse-the-frames-you-pick--f9)), typically T× faster |
| Tracks/time reductions look empty or trivial, and the canvas has an amber frame / `TROUBLESHOOTING MODE` badge | troubleshooting mode is on and only one frame is scoped. Ctrl+click a few T boxes so the tracker has a series, or `F9` to leave the mode |
| A tracker under `F9` links across the wrong gaps | the picked frames are the whole series as far as the graph is concerned — unpicked ones are absent, not empty. Pick a contiguous run (ctrl+drag), or leave the mode for real numbers |
| A table's `t` column reads 0 for a frame you know is 57 | same — under `F9` the payload holds that one frame and numbers it from 0. The status bar names the real frame |
| Console shows a `mojibake`/`UnicodeEncodeError` running the gates | prefix `PYTHONUTF8=1` (Windows cp1252 vs the `µ`/`σ`/`↔` glyphs) |
| Re-pulling recomputes a chain you expected cached | an ancestor was evicted by the memo's byte budget; eviction only costs a recompute, never correctness. Raise `NODEGRAPH_MEMO_BYTES` if you have RAM |
| A 3D node is far slower than expected, and slows further as you add timepoints | its volume exceeded `NODEGRAPH_CACHE_BYTES / 2`, so it fell back to realizing the whole series instead of the one volume you are looking at. Raise the budget above `2 × z × y × x × 8` bytes (§6) |
| The machine is pegged / you want the old single-threaded behaviour | `NODEGRAPH_PARALLEL=off`, or cap it with `NODEGRAPH_WORKERS=4`. Results are identical either way |
| Suspect a parallelism bug in a result | re-run with `NODEGRAPH_PARALLEL=off` and compare — they are asserted byte-identical, so any difference is a real bug worth reporting |
| Numbers shifted in the last ~7 digits after a config change | `NODEGRAPH_FLOAT32=1` is set; it is a genuinely different (single-precision) computation, not just a faster one |
| `serialtrack` seems hung | it is 1–2 orders of magnitude slower than the other linkers (≈8–18 s on 1k–15k detections) |
| An old `.nd2s_pipeline.json` will not open | that is the retired first-generation format; it is not supported and no importer exists |

---

## 19. Verifying a build

```bash
PYTHONUTF8=1 python -m nodegraph.selftest                        # headless core → 58 groups
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png  # driven GUI probe
PYTHONUTF8=1 python scripts/_hotreload_probe.py                  # live per-node reload (§12d)
PYTHONUTF8=1 python scripts/_catalog_snapshot.py check           # the catalog is unchanged
```

The first three must end in `ALL NODEGRAPH SELF-TESTS PASSED` / `ALL PHASE-5 GUI PROBES
PASSED` / `All live per-node reload probes passed.`; the fourth in `CATALOG IDENTICAL`.

**The catalog snapshot** is the gate for anything that moves node definitions between modules.
It compares every op key, in registration order, against `scripts/catalog_baseline.json`:
socket-by-socket and field-by-field, including units, derives, defaults, layer/path/pick
annotations and hover prose, plus each node's modes, footprint, domain declarations and wired
compute. A refactor that drops one socket's `unit`, reorders a pair, or attaches the wrong
compute is invisible to a test that only runs a few nodes — this catches it. Re-bless it with
`save` **only** when a catalog change is intended.

**The reload probe edits real node source on disk** (`nodegraph/catalog/enhance/gamma.py` and
a shared helper) to prove an edit goes live, and restores both byte-for-byte from a `finally`,
including after a failed assertion. If it reports anything other than `every edited file
restored byte-for-byte`, check the working tree before running anything else.

**The run-policy matrix.** The engine's execution policy is configurable (§6), so the
selftest is expected to pass under each setting — and `test_parallel` asserts serial and
parallel results are byte-identical, so a divergence fails the gate rather than becoming a
silent numerical difference:

```bash
for cfg in "" "NODEGRAPH_PARALLEL=off" "NODEGRAPH_PARALLEL=thread" \
           "NODEGRAPH_PARALLEL=process" "NODEGRAPH_FLOAT32=1"; do
  env $cfg PYTHONUTF8=1 python -m nodegraph.selftest >/dev/null && echo "ok: ${cfg:-default}"
done
```

Heavier, not in the fast gate:

```bash
python scripts/_ingest_nd2_smoke.py            # a real ND2 end-to-end
python scripts/_bench_provider_granularity.py  # the storage-layout keystone benchmark
python scripts/_bench_ccl_watershed.py         # CCL / watershed cost
python scripts/_nodelab_v2_shot.py out.png     # one offscreen render (--welcome for launch state)
```

**Adding a node?** Read the **`wire-node-v2`** skill for the concepts, then follow
**`build-node-v2`** for the procedure. Both gates above are that procedure's exit criteria.
