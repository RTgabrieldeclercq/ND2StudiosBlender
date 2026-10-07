# Workflows

How a request actually travels through this codebase. Each entry is a numbered trace naming
the real `module.symbol` at every step — grep any of them in `gen/symbols.jsonl` for its live
signature — and ends with an ordered **read next** list of at most two items, with a rough
cost, so following a thread is a decision rather than a browse.

These are routes, not authority. A step that says "X hands to Y" is checked by the gate only
insofar as X and Y still exist with the same signature. The reasoning is in
`ENGINEERING_NOTES.md`; the truth is in the code.

---

### WF-01 — a pull, end to end
anchors: sym:nodegraph.engine.Engine.pull, sym:nodelab_v2.runner.EngineRunner, sym:nodegraph.memo.node_recipe_hash

What happens between clicking a node and seeing pixels.

1. `nodelab_v2.runner.EngineRunner` takes the canvas's `GraphDocument`, builds a
   `nodegraph.graph.Graph`, and schedules the work on **one persistent, Python-created
   thread** (`_PullThread`) off the UI thread — never a QThreadPool, whose recycled Qt thread
   CPython re-adopts per pull, which corrupted torch's per-thread state (INV: one pull thread
   identity for the process lifetime). Further requests are QUEUED, each with its own run id,
   and an edit cancels only the runs whose cone contains what it touched — so a finished
   branch stays viewable and editable while another is still computing. A DELETE narrows the
   same way (`GraphDocument.remove_node` names the node), and additionally latches the
   running job's `cancelled` flag, which the engine polls through `Engine.should_stop` —
   the pull unwinds with `PullCancelled` and the queue advances instead of waiting it out.
2. `nodegraph.engine.Engine.pull(node_id)` asks for the node's payload. It delegates to
   `Engine.entry`, which recurses up the graph through `Engine._entry`, so upstream nodes are
   computed depth-first on demand. Nothing outside the pulled node's ancestry runs at all.
3. For each node, the engine resolves the mode state, then the footprint
   (`NodeSpec.resolve_granularity`) — this is where TILEABLE/WHOLE_VOLUME routing is decided,
   see WF-02.
4. It computes the recipe hash (`nodegraph.memo.node_recipe_hash`) and asks the memo. A hit is
   only accepted if `Engine._reads_valid` confirms every calibration value the node declared
   reading still digests the same (CON-09).
5. On a miss it calls the compute with a `ReadContext` — `ctx.params`, `ctx.calib`, `ctx.layer`
   — which *records* what was read, so the memo can fence on it next time. The compute
   returns a lazy provider chain, not an array (CON-11).
6. `Memo.put` stores the entry under a byte budget; the runner emits progress events that the
   console, the framestrip and the viewer consume.

**Where it goes wrong:** a compute that constructs its streaming provider *before* recording a
read leaves that read out of the provider's fingerprint, and the tile cache then serves stale
tiles. Construct the provider last (INV-02).

read next: `grep '"path":"nodegraph/engine.py"' gen/modules.jsonl` (~150 tok, the exports) ·
ENGINEERING_NOTES §9 "the pull engine" (~2k tok, the why)

---

### WF-02 — where tiling is decided
anchors: sym:nodegraph.registry.NodeSpec.resolve_granularity, sym:nodegraph.streaming.TileCache, sym:nodegraph.registry.Granularity

The question "why did this pull eat 12 GB" almost always ends here.

1. The node declares a footprint per mode value (CON-03). `resolve_granularity(state)` picks
   the one for the current 2D/3D lever position; `resolve_kernel_axes(state)` says which axes
   the maths reaches across.
2. `TILEABLE` → `nodegraph.streaming` wraps the compute in a per-tile provider
   (`MapComputeProvider` and friends) and plans tiles against a byte budget. The halo comes
   from the kernel's own reach — a Gaussian's is `int(4σ+0.5)` — and overlapping tiles are
   recomputed rather than shared, which is why a halo that is too small is silently wrong at
   seams rather than loudly broken.
3. `WHOLE_VOLUME` → the compute is handed one lazy ZYX volume per (t, c). It still does not
   materialize the series, but it *does* materialize a stack, which is the usual answer to
   "why is 3D so much heavier than 2D on the same node".
4. `WHOLE_SERIES` / `MULTI_VIEW` → everything, or every position. Rare and deliberate
   (normalisation over a series; stitching).
5. `nodegraph.streaming.TileCache` holds realized tiles under a byte budget, keyed by the
   provider fingerprint. `nodegraph.spill` takes over when a single dense output will not fit.

**For the viewer specifically:** display reads are the only ones that ask for a reduced
resolution level, and a `WHOLE_VOLUME` node cannot serve one — it has to build the stack
before anything can be shown. That is the mechanism behind "the viewer got slow when I flipped
the lever to 3D".

read next: `grep '"path":"nodegraph/streaming.py"' gen/modules.jsonl` (~200 tok) ·
[MANUAL §6](../MANUAL.md) "what a 3D whole-volume node costs while you look at it" (~1k tok)

---

### WF-03 — what invalidates a cached result
anchors: sym:nodegraph.memo.node_recipe_hash, sym:nodegraph.memo.output_fingerprint, sym:nodegraph.hotreload.closure_fingerprint

1. **Params, modes, upstream shape** fold into `node_recipe_hash` — the cache key. Change one,
   get a new key, recompute. Sockets marked `presentation` are deliberately excluded, which is
   why dragging an opacity slider repaints instead of re-running.
2. **Code** folds in too, via `hotreload.closure_fingerprint(owner_module)`: the digest of the
   node's whole first-party dependency closure. Edit a shared helper and every node that
   forwards through it re-keys — correctly, because that helper *is* node behaviour.
   Package `__init__` files are excluded from the closure on purpose, or adding one node would
   re-key all of them.
3. **Calibration** is fenced separately: the entry records `(key, digest)` for every
   calibration value the compute read, and `Engine._reads_valid` re-checks them on every hit.
4. **Upstream results** are compared by `output_fingerprint`, so a node that re-ran and
   produced identical output does not force its dependents to re-run.
5. Identity is `nodegraph.revision.next_revision`, never `id()` (INV-01).

read next: `grep '"path":"nodegraph/memo.py"' gen/modules.jsonl` (~150 tok) ·
ENGINEERING_NOTES §10 "memoization" (~1.5k tok)

---

### WF-04 — how a node comes to exist
anchors: sym:nodegraph.catalog.module_order, sym:nodegraph.catalog._base.register_node, sym:nodegraph.registry.define_node

1. `import nodegraph.nodes` — an 89-line **facade** — calls the catalog's `load()`.
2. `nodegraph.catalog.module_order()` yields the ordered module list (the static `MODULES`
   tuple first, then anything found on disk). **Order is load-bearing**: it is the order the
   link-drag search menu enumerates.
3. Importing each module executes one `nodegraph.catalog._base.register_node(compute, op_key=…,
   **spec)` per node, which calls `registry.define_node` and stores the compute in `COMPUTES`.
4. `define_node` runs the registration validators (layer sockets, interaction/pick kinds,
   bounds, modes, conditional domain rails, choice docs) and records **which module** defined
   the node. That provenance is what `hotreload.is_catalog_op` uses to decide whether a node is
   shipped-catalog or GUI-layer, and therefore whether the catalog-wide gates cover it.
5. `nodelab_v2.ops.ensure_ops()` separately registers the GUI-layer ops (`io.load`,
   `view.viewer`, `io.dock`, and since V4.00 `page.input` / `page.output` and the hidden
   `data.part` tap a card's part socket becomes, CON-22). They are real
   nodes but are not hot-reloadable and are outside the catalog gates — the map marks them
   `gui_only`.

**Do not** go looking for node definitions in `nodegraph/nodes.py`. Several older documents
still say that; the split to one-file-per-node happened in V2.20.

read next: the `build-node-v2` skill (procedure) · `wire-node-v2` (concepts) — invoke them,
do not read them as files

---

### WF-05 — metadata propagation, edit time
anchors: sym:nodegraph.metadata.propagate_meta, sym:nodegraph.metadata.MetaEnvelope, sym:nodegraph.metadata.eval_derive

Runs on **every keystroke**, with no pixels.

1. Seeds: each source node contributes a `MetaEnvelope` (axis sizes, calibration dict,
   domains, layer names) built from the file's own header at ingest.
2. `propagate_meta` walks the graph forward topologically. Each node's declared
   `meta_transform` predicts its output envelope (CON-07); `extra_layers` adds layers that have
   no output socket; `reads_domains` / `reads_domains_by_mode` decide whether the wiring is
   even legal. Domains, layers and columns otherwise flow on from the primary input — except
   past a node declaring `fresh_output` (a plot's Picture, V4.00), whose output is a new
   Dataset and carries only what it adds. A column declaration marked `wants_inputs`
   (`table.concat`, `table.join`, V4.00) is also handed every input's envelope, so the
   columns only a later input carries are named too.
3. The resulting envelope drives the GUI: axis sizes in the header, the layer picker's
   choices, the domain rail, and `derive` defaults resolved through
   `envelope_symbols` + `eval_derive` (CON-05).
4. Anything genuinely unknowable before a pull stays in `unknown_axes` and renders as `?`. It
   is never guessed — a guessed axis size becomes a wrong default that a user then trusts.

**Invariant:** this pass must never raise. Its caller catches only `ValueError`, and even a
caught error blanks every node's envelope graph-wide (INV-06).

read next: `grep '"path":"nodegraph/metadata.py"' gen/modules.jsonl` (~200 tok) ·
ENGINEERING_NOTES §8 "metadata intelligence" (~2k tok)

---

### WF-06 — live node editing (hot reload)
anchors: sym:nodegraph.hotreload.node_modules, sym:nodegraph.hotreload.dependency_closure

1. `node_modules()` is **derived** from the catalog's module order plus the `_shared` prelude —
   never hand-maintained, because a node module missing from that list is un-watched,
   un-fingerprinted, and its memo would serve results from code that is no longer on disk.
2. `_direct_imports` parses each module's **source** with AST rather than inspecting the
   runtime, specifically so a kernel imported inside a compute body is seen before that node
   has ever run.
3. `dependency_closure` walks it transitively (cycle-safe, package initialisers excluded);
   `closure_fingerprint` digests the result. That fingerprint is what re-keys the memo (WF-03).
4. On reload, kernels are refreshed first, then node modules re-execute so their specs are
   rebuilt. `nodelab_v2.ops` is deliberately excluded — the window and the runner hold
   references into it, so re-executing it would swap plumbing the GUI is standing on.

read next: `grep '"path":"nodegraph/hotreload.py"' gen/modules.jsonl` (~200 tok) ·
`scripts/_hotreload_probe.py` (the end-to-end proof, ~400 L)

---

### WF-07 — a file becomes a Dataset
anchors: sym:nodelab_v2.runner.EngineRunner, file:nodelab_v2/ingest.py

1. The File menu resolves a path. **Never `import nd2` directly** — go through
   `nodelab_v2.nd2_compat.import_nd2()`, which wraps SDK bugs that make healthy files
   unopenable and fire at file-pick time, before a pixel is read (INV-07).
2. `nodelab_v2.ingest` converts ND2/TIFF into a planar-block `.b2nd` store and hands back a
   `nodegraph.provider.B2ndProvider` plus a `MetaEnvelope`. `nodelab_v2.nd2_meta` parses the
   optics and calibration — pixel size, z step, frame interval, emission per channel, NA,
   magnification — which is what makes every `unit`/`derive` socket downstream work (CON-05).
   Beside the calibration rides the **placement vocabulary**, `ingest.PLACEMENT_KEYS`:
   `z_home_index`, `z_bottom_to_top`, the per-T `frame_time_jd` and its readable twin
   `frame_datetime` (2026-10-02; the Viewer's timestamp overlay and the Timeseries Builder
   read it), `stage_layout_source`, `acquisition_start`, and the per-M `position_name` —
   each best-effort, and dropped whole rather than padded when the file carries it short.
   The per-channel **names and native colours** ride the source envelope too
   (`ingest.channel_display_seed`, V4.00 step 11g — the one function the file-pick seed and
   the runner's resolved envelope both go through), so a stream downstream of a channel tap
   can say which channel it is (CON-23).
2b. **There are two routes, chosen by `io.load`'s `access` mode**, and the store is only
   one of them. Under `access="direct"` — **the default** (`ops.ACCESS_DEFAULT`), for a
   freshly placed card and for every graph saved before this mode existed alike — the
   runner skips the ingest entirely and builds a `nodelab_v2.nd2_direct.Nd2DirectProvider`
   over the ND2's own memory-mapped frames: no store, nothing written, usable the instant
   the file is picked. The calibration half is unchanged (same `read_calibration`), so
   only the provider differs; everything downstream is identical. It exists because the
   ingest's second copy is not always affordable: the lab's 453 GB series needs ~340+ GB of
   store on a 931 GB drive that already holds it.

   `access="auto"` resolves the choice instead of committing to it, via
   `EngineRunner._effective_access`: an existing valid store on disk always wins outright
   (no space check — it is already paid for); otherwise `nodelab_v2.nd2_direct.decide_access`
   estimates a fresh ingest's size — pessimistically, no compression credited — against the
   destination drive's free space (a 25%-or-5 GiB margin) and picks `ingest` if it fits,
   `direct` if it does not. Reach for it on a file that already has a store from an earlier
   session (`direct` on its own never looks for one), or whenever the size trade-off should
   just be decided rather than defaulted. The resolution is decided once per path and
   cached for the runner's session (`EngineRunner._auto_access`) — `source_state()` polls
   it continuously and must not re-open the file or re-stat the drive on every tick.
   `access="ingest"` forces the store unconditionally.

   `access` (resolved, never the literal `"auto"`) is part of the provider-cache key
   (`runner.source_key`), and the `ingest` key is deliberately byte-identical to the
   pre-mode tuple so every saved graph and on-disk store still hits under `access="auto"`
   or `access="ingest"`.
2c. **The per-M grouping is resolved here too**, by `runner._with_position_groups`, and
   stamped onto the envelope as `position_group` / `position_name`
   (`metadata.PER_POSITION_KEYS`). A multipoint axis is often several specimens rather than
   one flat list — the lab's CRC file is six 3×3 mosaics a millimetre apart — and
   `nodelab_v2.position_groups.resolve_plan` decides which: a hand-written
   `<file>.groups.json` sidecar if there is one, else
   `nodegraph.placement.position_groups` clustering the field centres at one field width.
   It is applied on the same path as `_with_card_calib`, which is the ONE place the payload
   and the engine's meta-seed both come from — stamping anywhere else would put the
   grouping in one of them and not the other, and `util.select_group`'s edit-time header
   would disagree with its pull.
3. An empty `io.load` path falls back to a synthetic stack, so a graph is always runnable.
4. A value the *file* carries that looks wrong is passed through, not repaired: a microscope
   PC with a bad clock produces an absurd acquisition date, and a downstream node refusing to
   place that data is the correct outcome. Shims only ever act on a raised exception.

read next: `grep '"path":"nodelab_v2/ingest.py"' gen/modules.jsonl` (~200 tok) ·
ENGINEERING_NOTES §12 "providers & storage layout" (~2.5k tok)

---

### WF-08 — a pull across pages
anchors: sym:nodelab_v2.workspace.Workspace.compose, sym:nodelab_v2.runner.EngineRunner, sym:nodelab_v2.workspace.split_run_id

1. A pull names a node on a page — `MainWindow.pull_node` qualifies it with the page it came
   from (`workspace.qualify`, the canvas or Viewer's page, else the active one) and hands the
   runner the run id `<page>/<node>`; `EngineRunner.run_id` also takes a bare id to be on
   the active page. Every runner signal carries qualified ids; the window splits them
   (`split_run_id`) to route a result to the Viewer bound to that page and node.
2. `EngineRunner._compose` asks the `Workspace` (the runner's `GraphSource`) for
   `compose(page)`: the page and its dependency closure as ONE graph (CON-17), each page
   materialised through its own document (`to_graph(for_run=True)`), resolved Page Inputs
   rewired to the upstream Outputs, meta seeds qualified.
3. One engine runs it on the runner's ONE persistent memo and tile cache. The recipe hash
   carries no node id, and a root's `__source__` is its qualified run id, so a refinement
   chain computed for one processing page is a memo hit for every other page that reads it,
   and for a linked page what it reads through a Page Input, up to its first override
   (CON-18) — a root on the linked page itself is its own.
4. `ComposedGraph.revision` (and `Workspace.revision_of`) is the runner's identity for a page:
   an edit on any page in the closure changes it and nothing else, so only the pages
   downstream of an edit rebuild.

read next: CON-17 (2 min) · `nodegraph.selftest.test_page_composition_memo_reuse` (grep, 3 min)

### WF-09 — a demo pull (What does this node do?)
anchors: sym:nodelab_v2.window.MainWindow.open_node_demo, sym:nodelab_v2.demo_window.NodeDemoWindow, sym:nodelab_v2.demo_recipes.DemoSession.run

1. The inspector's `?` (or the palette's button) emits `demo_requested(op_key)`;
   `MainWindow.open_node_demo` shows the one `NodeDemoWindow` per op type, creating it from
   `demo_recipes.recipe_for(op)` — the role default from `node_roles.json` merged with the
   `codemap/node_demos.json` entry (phantom, prelude, fixed values, slider spans, kind).
2. `DemoSession.build_graph` makes a fresh `Graph`: `src` (an `io.load` seeded with the
   phantom, as the runner seeds a file) -> `pre0..preN` -> `demo`, plus any extra Dataset
   inputs (a second phantom, a prelude node). Only touched params are sent; modes are the
   full state, so `active_inputs` gates the sliders the way it gates the inspector.
3. `_DemoWorker` (one Python thread, latest request wins, 80 ms debounce) calls
   `DemoSession.run`: `headless_engine(memo, tiles)` -> `pull("demo")` -> `realize` ->
   a `DemoResult` reduced to the viewed `(t, z, c)` plane: the before and after planes, the
   label / mask / scalar plane, points, tracks, vectors, mesh vertices, structure tables.
4. On the GUI thread a stale generation is dropped; otherwise both planes go through
   `composite_with_clim` (the before's window shared when the after lives in the same range)
   onto two `_ImageView`s and the kind's overlay is painted by the Viewer's `OverlayRenderer`
   through `overlay_cb` with an `OverlayFrame`. Slow recipes (or two runs over 1.5 s) switch
   to a Run button.

read next: INV-17 (1 min) · `nodegraph.selftest.test_node_demos` (grep, 3 min) · MANUAL §8e
