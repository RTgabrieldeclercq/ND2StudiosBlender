# Concepts

The vocabulary this repo talks in. Each term is defined **once** here so that everywhere else
— node cards, socket prose, workflow traces, commit messages — can use it bare and cost four
tokens instead of a sentence.

Read an entry when a term in a record does not explain itself. Do not read this file end to
end; it is a dictionary.

Every entry declares `anchors:` — the code it describes. When that code's contract changes,
`nodegraph.selftest.test_codemap` flags the entry for re-reading. The mechanism cannot check
that prose is *true*; it checks that nobody moved the thing it describes.

---

### CON-01 — op_key
anchors: sym:nodegraph.registry.define_node, sym:nodegraph.catalog._base.register_node

A node type's permanent identity: `<category>.<name>`, e.g. `enhance.gaussian`. It is the key
in the global `NODES` registry, the key in a saved `*.nd2graph.json`, the filename convention
for its module, and the primary key of `codemap/gen/nodes.jsonl`. **It is frozen once
shipped** — renaming one silently breaks every saved graph that used it, with no error at
load, because the node simply is not there any more.

see: [WF-04](workflows.md) for how registration happens · `grep '"op":"<op_key>"' gen/nodes.jsonl`

---

### CON-02 — Domain
anchors: sym:nodegraph.domains.Domain, file:nodegraph/domains.py

The twelve kinds of thing an attribute can be attached to: the acquisition lattice
(`voxel`, `plane`, `frame`, `timepoint`, `multipoint`, `batch`, `global`, `channel`) and the
structures built on it (`label`, `point`, `track`, `mesh`). A Dataset is not "an image" — it is
a bundle of attribute layers, each living on one domain, and a node declares which domains it
*reads* and which it *adds*.

This is why "segment then measure" needs no special case: segmentation adds a `label` domain,
measurement reads it. The prose for each domain is in `gen/vocab.jsonl` (`"group":"Domain"`).

**`batch` (V3.01) is per FILE, and `global` is across the whole batch.** The `b` axis joins
every place-or-time domain — `voxel` is `{b,m,t,z,c,y,x}`, `multipoint` `{b,m}`, `timepoint`
`{b,t}` — because a position index and a frame index only mean anything within one file.
`channel` does not gain it (a batch shares its stains) and neither does `global`.

`batch` is **load-bearing, not a convenience**: `meet(Multipoint, Timepoint)` is now `{b}`,
and `meet` indexes the axis-set→domain map directly, so without a name for `{b}` that call is
a `KeyError`. It is what keeps the lattice closed under intersection.

The axis is **elided from arrays at `b == 1`** — see `AxisSizes.axis_list`. A one-member
Dataset has exactly the shapes it always had, so a batch axis costs existing computes
nothing; `b` materializes only on a real `b > 1` Dataset at the batch boundary.

**The structure spine carries it as a COLUMN**, `domains.BATCH_COLUMN` (re-exported by
`structure`), and deliberately *not* as a member of `COORD_COLUMNS` — that tuple is a hard
requirement two nodes check, and a batch column exists only when there is a batch. It lives
in `domains` rather than beside `COORD_COLUMNS` because `dataset.with_structure` must check
it and `structure → memo → dataset` would be a cycle. `with_structure` refuses a table that
lacks it on a `b > 1` Dataset, so **absent always means "there is no batch"**, never "nobody
filled it" — without that rule two specimens' objects are indistinguishable rows in one
table, which reads as a measurement averaging two samples.

Producers get the column for free: `_shared.batch.batch_aware` wraps a compute so a batched
input runs **once per member** and the join stamps the column and offsets the ids. That is
why an eager node needs no edit to its `(m,t,z,c)` loop, and why a data-derived level is
per-file by construction — each member's compute only ever sees one file's population.

see: CON-08 transfer · CON-11 (the `b` read contract) · [MANUAL §10](../MANUAL.md)

---

### CON-03 — Granularity, and the data-access footprint
anchors: sym:nodegraph.registry.Granularity, sym:nodegraph.registry.NodeSpec.resolve_granularity

How much data a compute must see at once — the single most consequential thing a node
declares. Five values, cheapest first: `TILEABLE` (a tile at a time, with a halo),
`WHOLE_PLANE`, `WHOLE_VOLUME` (one ZYX stack), `WHOLE_SERIES`, `MULTI_VIEW` (every position).

The engine routes on it: a `TILEABLE` node streams and a 6554² plane never lands in RAM
whole; a `WHOLE_VOLUME` node gets a lazy volume instead of tiles. So a footprint that
**over-claims** costs memory the node never needed, and one that **under-claims** produces
wrong pixels at tile seams — silently, because each tile is individually plausible.

Usually declared per mode value: `{"2D": TILEABLE, "3D": WHOLE_VOLUME}`, keyed by the node's
`footprint_mode` (normally the 2D/3D lever). `kernel_axes` says which axes the maths reaches
across, which is what sizes the halo.

see: [WF-02](workflows.md) · the `build-node-v2` skill's footprint gate · CON-04

---

### CON-04 — the 2D/3D lever (DimMode)
anchors: sym:nodegraph.registry.NodeSpec.resolve_kernel_axes, op:enhance.gaussian

One `Mode` with `role="dim_lever"`, rendered as a switch in the node's header rather than a
dropdown in its body, because it is the question you ask most. It selects between running
plane-by-plane and running truly volumetrically, and it is the key that resolves `granularity`
and `kernel_axes` — so the same graph runs 2D or 3D without rewiring.

It greys out on data where `z == 1`. Sockets can be gated to one side of it via
`available_in` (an axial σ exists only in 3D).

see: CON-03 · [MANUAL §9](../MANUAL.md)

---

### CON-05 — unit and derive: metadata-intelligent parameters
anchors: sym:nodegraph.metadata.eval_derive, sym:nodegraph.metadata.envelope_symbols

A spatial or temporal socket does not carry pixels — it carries a physical quantity plus a
`unit` (`um`, `um_axial`, `s`, …), and often a `derive` expression that computes its default
from the image's own calibration. A 2 µm radius becomes the right pixel count for *this*
objective; a deconvolution PSF derives from emission wavelength and NA per channel. Override
it and the override sticks.

`derive` is evaluated against a small symbol table (`pixel_size_um`, `z_step_um`, `dt_s`,
`emission_nm`, `na`, `mag`, `n_z`, `is_3d`, …) with empty builtins.

**The rule that catches people:** calibration describes the data *on this wire*, not the file
it came from. After a resample, `ctx.calib` already returns the post-transform value — using
it and *also* re-dividing by the scale double-counts.

see: [WF-05](workflows.md) · INV-05 · the `wire-node-v2` skill §7

---

### CON-06 — MetaEnvelope and the edit-time pass
anchors: sym:nodegraph.metadata.MetaEnvelope, sym:nodegraph.metadata.propagate_meta

The pixel-free shadow of a Dataset: axis sizes, the calibration dict, which domains exist,
which attribute layers exist by name. It is propagated forward through the graph on **every
keystroke**, which is what lets the GUI show axis sizes, offer a layer picker, resolve
`derive` defaults and grey out impossible wiring — all without reading a pixel.

Because it runs on every keystroke, it must never raise. Sizes that are genuinely unknowable
before a pull stay in `unknown_axes` and are shown as `?` rather than guessed.

see: CON-07 · [WF-05](workflows.md) · INV-06

---

### CON-07 — meta_transform
anchors: sym:nodegraph.metadata.propagate_meta, op:util.crop

A node that changes axes or calibration declares a named function saying *how*, so the
edit-time pass can predict the output envelope without computing it. A z-projection collapses
z; a crop changes extent; a resample changes pixel size.

**Both halves must be declared and must agree exactly**: the `meta_transform` (edit time) and
the payload the compute actually returns (pull time). A disagreement is a real bug class here
— the GUI shows one geometry and the data has another — so the selftest asserts
`engine.env(node).axes == pulled.axes` for axis-changing nodes.

The signature is `(env_in, params, modes)`, and `env_in` is **input 0's envelope only**. So
an N-input node normally cannot predict the axis it grows and marks it *unknown* instead —
`merge_grow` and `batch_grow` both do, and the GUI shows `?` until a pull resolves it.

**A transform that needs every input opts in** by setting `wants_inputs = True` on the
function (`chain_grow` is the only one today); `propagate_meta` then passes the full list of
dataset-predecessor envelopes as a fourth argument. It is opt-in rather than a protocol
change so a node that only ever reads its primary input cannot start depending on a second
one's envelope by accident — and so adding it cost no edit to the other fourteen transforms.

see: INV-04 · `grep '"group":"meta_transform"' gen/vocab.jsonl` for the named transforms

---

### CON-08 — domain transfer and structure bridges
anchors: sym:nodegraph.transfer.plan_transfer, sym:nodegraph.transfer.register_bridge

Moving an attribute from one domain to another. Within the acquisition lattice it is
generated: coarsening applies a reducer (mean, sum, …), refining broadcasts. Between the
lattice and a structure domain it needs a **bridge** — a registered, explicit conversion
(voxel mask → labels, labels → points, points → mesh) — because there is no single right
answer and the choice is the user's.

see: CON-02 · ENGINEERING_NOTES §13

---

### CON-09 — the two-hash memo
anchors: sym:nodegraph.memo.node_recipe_hash, sym:nodegraph.memo.output_fingerprint

Two different hashes doing two different jobs. The **recipe hash** is structural — this node,
its params and modes, its upstream chain, its code fingerprint — and it is the cache key. The
**output fingerprint** is a digest of what came out, and it is what lets a downstream node
notice that its input did not actually change even though something upstream re-ran.

Result: an unrelated edit recomputes only the invalidated chain. Entries are evicted under a
byte budget, LRU.

Calibration reads are *fenced*: a hit is only valid if every calibration value the node
declared reading still digests the same. That is why a compute must read calibration through
`ctx.calib` and not out of the payload.

see: [WF-03](workflows.md) · CON-10 · INV-01

---

### CON-10 — revision, the memo's identity
anchors: sym:nodegraph.revision.next_revision

A process-monotonic counter. Identity in the memo is the revision, **never** `id()` (freed
objects get their address reused, and the memo then serves the wrong payload) and never a
content hash (too expensive to take on every lookup). This is a one-line rule with a long
debugging history behind it.

see: INV-01

---

### CON-11 — provider, and lazy streaming
anchors: sym:nodegraph.streaming.TileCache, file:nodegraph/provider.py

Computes do not return realized 6-D arrays; they return chained lazy providers. A pull walks
the chain per tile, with overlap-recompute halos and a byte-budget tile/field cache, so the
peak memory of a pipeline is a few tiles rather than a series.

A provider's fingerprint must fold in the declared calibration reads and any field
expressions, or the tile cache serves a stale tile that looks perfectly valid.

A provider may also compose *several* sources rather than wrap one: `ChannelMergeProvider`
re-addresses a second acquisition onto the first's grid, `MultiSourceProvider` lays K files
end to end on `m` (the **file bundle** — pure index arithmetic, nothing resampled), and
`AxisConcatProvider` is that same index-remap generalised to `t`/`m`/`z` — whichever axis
`util.merge` was told to grow — for its literal-concatenation branch (`util.merge`'s `C`
branch instead chains `ChannelMergeProvider`, since a channel merge needs placement/
resampling that a pure index remap cannot express).
A composite folds each member's `version`, not its `fingerprint`, into its own: a
disk-backed member carries its store's mtime in `version` alone, so folding fingerprints
would serve a bundle whose file changed underneath it.

A provider need not wrap anything at all: `ConstantProvider` (`view.canvas`, 2026-10-01)
GENERATES a uniform window on demand at any pyramid level, so an experiment canvas tens of
thousands of pixels a side — the blank primary that several files are overlaid onto — holds
no pixels. Its fingerprint is structural (geometry + value): two canvases of one geometry
are the same pixels.

`AxisRespreadProvider` (`util.timeseries`, formerly `util.chain`, V3.02) is the one that erases the difference between
the two shapes above. It takes **`(provider, m start, position count)` triples — one per
source FILE** — and lays them end to end on `t`, `z`, `c` or `m`, by the same exclusive
prefix sum `AxisConcatProvider` uses: `m = start_i + j, axis = a` becomes `m = j,
axis = off_i + a` (onto `m`, `m = off_i + j`). A *bundle* is K members sharing one provider
and differing in `start`; K *separately loaded cards* are K providers all starting at 0; a
mixture is both at once. Same pure index arithmetic as the rest of the family, so a chained
read costs an un-chained read.

**Members may differ on the axis being chained — that is the point**, and the output is
their sum. A series exported in unequal chunks (5 frames, then 3) is still one series, and
the first cut of this provider refused it by requiring all six axes to match and sizing the
result `k * a0`. What must agree is every axis EXCEPT `m` and the chained one; `m` is exempt
because each member is read through its own `[start, start+count)` window. The one extra
rule: onto anything but `m`, the members' position counts must match, because the result has
a single `m` and a file with more has nowhere to put them.

Members are in OUTPUT order, so the filename-sequence ordering (`nodegraph.file_sequence`)
is expressed there and nowhere else, and folds into `version` — the same files chained the
other way round can never be served from each other's memo entry.

**Ordering needs the files to be NAMED, which is why every source card now stamps
`source_file`** (`metadata.stamp_source_file`, V3.02), not only a bundle's members. Before
that a Dataset loaded from its own card carried its filename nowhere a compute could reach —
the path lived in the node's params. The visible consequence, accepted deliberately: a table
from a single-file graph carries a `file` column too now.

**The read contract carries the batch axis as a keyword (V3.01).** `read_region` and its
`get_*` wrappers take `b` **keyword-only, defaulting to 0**, so all 206 pre-existing
positional call sites keep meaning "the one member" and every batch-aware read has to say
`b=` where it can be read. Threading it positionally would have been 206 edits in which one
mis-ordered argument reads `m` as `b` — the wrong position's pixels, no error.

A wrapper provider must **forward** `b` (`FrameSubsetProvider`, `MultiSourceProvider`,
`AxisConcatProvider`, `AxisRespreadProvider`, `ChannelMergeProvider`, `WindowView`, the
channel/frame views), and
every *compute* provider in `streaming.py` must carry it in its **unit cache key**. That key
was `(m,t,z,c)`, which was total before `b` existed and is not now: without `b` in it,
member 1 is served member 0's cached unit — the right shape, plausible pixels, the wrong
file, and no error anywhere. All five (`MapComputeProvider`, `VolumeComputeProvider`,
`_AxisReduceProvider`, `PlaneRealizeProvider`, `MultiViewProvider`) key and forward it.

**Two providers own the batch itself.** `BatchProvider` stacks K files on `b` — a pure
re-address like `MultiSourceProvider`, but growing a new axis instead of lengthening `m`,
which is what stops a scope pooling a level across files. `BatchSliceProvider` is its exact
inverse, pinning one member back to `b == 1`; it wraps *any* provider, not just a
`BatchProvider`, which is why unbatching a fully-built pipeline costs no copy and no
recompute.

**Nothing folds across `b`, and that is geometry rather than a special case**:
`_AxisReduceProvider` only ever reduces `z` or `t`, so a projection folds inside one member;
`MultiViewProvider` collapses `m`, so a batch of K files stitches to K canvases, never one
composite.

see: [WF-02](workflows.md) · INV-02 · CON-02 (the `Batch` domain) · ENGINEERING_NOTES §11–12

---

### CON-12 — layer sockets
anchors: sym:nodegraph.registry.SocketSpec, sym:nodegraph.registry.define_node

A socket whose value is the *name* of an attribute layer rather than data: `layer_in` picks an
existing one (the GUI offers a picker populated from the envelope's layer catalog),
`layer_out` declares one this node creates. It is how "measure the `distance` layer" is
expressed without a socket per possible layer.

A layer a node creates that has no output socket must still be declared, via `extra_layers`,
or the picker downstream will not offer it. The opposite — a node that passes on only SOME of
its input's layers (the GUI's `data.part` tap) — is declared with `keep_layers` (CON-22).

`column_in` is the same idea one level down — the name of a COLUMN on a structure table
rather than the table itself. See CON-15.

see: CON-06 · CON-15 · the `wire-node-v2` skill §4c

---

### CON-13 — the socket contract
anchors: file:.claude/skills/wire-node-v2/SKILL.md

Deliberately not restated here. One law governs every socket — every param reachable, every
socket read, defaults declared once, hover prose on every non-Dataset input — and it has
three homes already: **stated** for builders in the `wire-node-v2` skill §4b, **enforced** by
`nodegraph.selftest.test_param_socket_contract`, and **explained** (the why) in
ENGINEERING_NOTES §7. A fourth copy would just be a fourth thing to drift.

If you are adding or changing a socket, invoke `build-node-v2`; it cites the clause you need.

---

### CON-14 — the GUI seam
anchors: sym:nodelab_v2.runner.EngineRunner, sym:nodegraph.registry.NodeSpec

`nodegraph/` never imports Qt. Several `nodelab_v2` modules are deliberately Qt-free too —
`document.py` (the editing model: wiring rules, cycle rejection, metadata propagation),
`ops.py`, `tables.py`, `overlay_render.py` — which is exactly why the headless selftest can
test GUI behaviour at all. It is also the one place the layering arrow is allowed to reverse:
`nodegraph/selftest.py` imports `nodelab_v2` to test that seam.

The runner drives the engine off the UI thread through a QThreadPool with one worker and an
epoch registry, so an edit mid-run cancels cleanly instead of racing.

see: [WF-01](workflows.md) · INV-08

---

### CON-15 — the column catalog
anchors: sym:nodegraph.metadata.MetaEnvelope.columns_in, sym:nodegraph.metadata._column_names_out, sym:nodelab_v2.document.GraphDocument.column_choices

CON-12's layer catalog answers "which Label tables are on this wire". The column catalog
answers the next question down: "and what has anyone MEASURED onto them yet". A producer
declares the columns it writes with `NodeSpec.adds_columns(params, modes, incoming)`;
`propagate_meta` accumulates them into `MetaEnvelope.column_names` as `(domain, layer,
column)` triples; a socket declaring `column_in` (or `column_in_mode`, for a node whose
member domain is a lever) gets a picker offering them.

It takes a third argument where its sibling `extra_layers` takes two, because columns FLOW:
a node re-emitting a table under a new name carries every column with it, and could not name
one without being told what is already there.

**It is CLOSED, and therefore mandatory to declare.** A `column_in` socket renders as a
non-editable dropdown, so a producer that declares nothing makes its columns *unpickable*,
not merely unsuggested. `selftest::test_column_catalog_complete` fails the build for any
node that adds a structure domain without an `adds_columns`. That is the opposite
obligation from CON-12's layer catalog, which must stay open because a couple of producers
name layers the edit-time pass cannot predict — the column catalog can be closed because
what a table carries really is determined by the nodes upstream, which is the whole reason
the condition menu is worth having.

`column_join` lets a socket also offer columns from another domain that the compute reaches
by a JOIN — `analysis.if_else` tests `track_length`, which lives on the Track table and
reaches the Label rows by `member_id`. The picker filters that domain's join KEYS back out
(`document._JOIN_KEYS`), because a name the menu offers and the pull then refuses is the
live-control-that-does-nothing defect wearing a picker's clothes.

The `incoming` catalog is INPUT 0's, like every edit-time catalog. A node that builds a table
out of several inputs (V4.00: `table.concat`'s many, `table.join`'s `other`) marks its
declaration `adds_columns.wants_inputs = True` and receives a fourth argument,
`((socket, envelope), ...)` for every Dataset input, so it can name the columns only a later
input carries — otherwise they would be unpickable downstream.

see: CON-06 · CON-12 · `grep '"key":"analysis.if_else' gen/sockets.jsonl`

---

### CON-16 — the movie timeline
anchors: sym:nodegraph.catalog._shared.movie_timeline.Timeline, sym:nodegraph.catalog._shared.movie_timeline.normalize_spec, sym:nodelab_v2.runner.EngineRunner.fetch, sym:nodelab_v2.movie_editor.MovieEditorPanel

`io.write_movie` with `sweep = timeline` renders a JSON **timeline spec** held in its
`timeline` STRING socket: `segments` (a `clip`, or a depth-1 `loop` over t whose body clips
bind `t = "auto"` to the loop variable), each clip playing `t`, `z` or nothing and showing
**panels**, a panel optionally `tile`d by m / c / z / t into a grid. Sources are letters:
`A` = `data`, `B`/`C` = `source_b`/`source_c`. `normalize_spec` refuses unknown keys and
names the spec path of every bad value; `canonical_json` is the one spelling the editor
writes, because the memo hashes the string.

**There is one renderer.** A flat movie (`sweep = time|z`) is the one-clip timeline
`flat_spec` builds, per position, so the node, the editor's monitor and a converted movie
all draw through `Timeline.render(k)`. The display window is ONE decision per (source,
raster, z-mode, contrast settings, channel) for the whole movie, shared across tiles.

`source_b`/`source_c` are declared `passes_domains=False`: read, never merged, so
`propagate_meta` does not promise their layers downstream of a node that passes `data`
through. The editor gets their payloads from `EngineRunner.fetch`, a payload-only pull
(modelled on the Dock bake) that never reaches a Viewer pane. A panel channel with
`link: viewer` is a GUI hint: the window stamps the Viewer's LUT for that source node into
the concrete `lo`/`hi`/`gamma`/`rgb` at commit points (a settled LUT edit, Capture, Export,
Save), never per drag tick, and the renderer reads only the concrete values.

see: [MANUAL §16 I](../MANUAL.md) · INV-09 · CON-14

---

### CON-17 — workspace and pages
anchors: sym:nodelab_v2.workspace.Workspace, sym:nodelab_v2.workspace.Workspace.compose, sym:nodelab_v2.workspace.ComposedGraph, sym:nodelab_v2.workspace.Workspace.node_defaults, sym:nodelab_v2.workspace.Workspace.default_source, sym:nodelab_v2.workspace.Workspace.input_channels

A V4.00 file holds a **workspace**: ordered **pages**, each a node graph (`GraphDocument`) of a
**kind** — `input` < `refine` < `process` < `analyze`, plus `free` (any node, any wiring; a
pre-V4 file opens as one Free page). The kind decides what the palette and the link search
LEAD WITH — its PRIMARY set, `nodegraph.roles.ops_for_page`; every other op is its SECONDARY
set, `secondary_ops`, offered after it (the palette's collapsed *More nodes* band,
`scene.visible_specs` = `primary_specs` + `secondary_specs`, 2026-10-07; step 5 had hidden
it). The kind locks nothing and hides nothing. Pages hand data on BY NAME: a
`page.output` node names its input as a variable of its page, and a `page.input` on a page of
a strictly later kind (or across a Free page, while acyclic) reads it — its `source` param is
`"<page id>:<name>"`. Page ids (`pg1`, ...) come from a counter stored in the file and are
never re-used, so a reference cannot silently rebind.

**A page never runs alone.** `Workspace.compose(page)` builds ONE run graph: the page plus every
page in its dependency closure, upstream first, node ids qualified `<page>/<node>`
(INV-15), each resolved `page.input` dropped and its consumers rewired to the upstream
Output's run id. An unresolved Input stays a root whose pull says it is unbound.
`ComposedGraph.revision` folds every page in the closure, so the runner rebuilds its engine
exactly when one of them changes. `page.input` / `page.output` are GUI-layer ops (INV-16).

**The standard workspace (step 11).** `Workspace.standard()` — what a fresh window and File →
New hold — is one page per typed kind (`standard_kinds()`, read from the roles file), Image
Input active; `reset()` returns to it, keeping the active page's document as Image Input's.
A loaded image is routed to the Image Input page (`MainWindow._input_page`; the canvas
switches there first, since the loaders read the active scene) and published as a
`page.output` named after the file. The pages stay in sight on every canvas's tab strip
(`canvas.PageTabs`) and in the Pages panel (`pages_panel.PagesPanel`), both fed by
`Workspace.page_summary` (what a page reads and publishes); `insert_index_for` keeps a new
page in pipeline order. Since step 11d the strip is two rows — the page KINDS, then the pages
of the kind shown as sub-tabs — and a sub-tab can be CLOSED (`MainWindow.close_page_tab`,
session state `_closed_pages`): the page stays, listed "tab closed" in the Pages panel, and
any route that shows it again (the panel, the page menu, its kind tab) opens its tab again —
a page on a canvas is always open. A `page.input` grows one synthetic `chK` output per channel
like a Load card (`CHANNEL_TAP_OPS`), named from the upstream Output's descriptors
(`Workspace.input_channels`, installed as `doc.page_channels`; the file's channel TOTAL
crosses the same way, `Workspace.input_channel_scope`, CON-23); a `chK` edge materializes into
a `channel.select` tap that `compose` splices onto the upstream Output with the Input itself. A page's Input is seeded once per session (`_seeded_pages`), and its
start card (`welcome.WelcomeCard.configure`) is worded for its kind and dismissible (`_publish_source`; a TIFF card starts on `ingest`, the only
access mode that can read it). Page-op DEFAULTS have one choke point: `Workspace._attach`
installs `doc.node_defaults`, which `GraphDocument.add_node` merges UNDER explicit params — a
hand-placed Output gets `unique_output_name` (`out`, `out2`, …), a hand-placed Input gets
`default_source` (the most recently added named Output of the nearest feeder page,
`feeder_pages`: nearest kind, latest page, a Free feeder last); the load, duplicate and
group paths build records directly and keep their values. `add_page(seed_input=True)` (the
switcher's *New page*) and the first visit to an empty downstream page add one bound Input.
Readiness (`readiness.problems`) states the fixes as `Suggestion.action` — `set_param`
(bind, name) and `append` (a Page Output after a terminal node or an unpublished loader, the
`unpublished` hint, severity `hint`, never blocking `ready()`).

**Variables, colours, the outline (step 11e).** A Page Output's name is unique ACROSS THE
WORKSPACE: `GraphDocument._settle_output_name` (run by `add_node` and `touch`, the two ways a
name arrives) asks `doc.claim_output_name`, which the Workspace installs as
`Workspace.claim_output_name` — a taken name becomes `name2`, reported through
`doc.renamed_output`; only the pages linked to one master share its names. A page copied, or
made unique, renames its taken names (`dedupe_outputs`) and `_follow_renames` re-points every
Page Input that read them. `doc.source_kind` (`Workspace.source_kind`) gives each Source-menu
entry the colour of the page kind it reads. The Pages panel draws `Workspace.page_outline`:
each page's nodes as `OutlineRow`s in data-flow order — entries first, a chain at one depth, a
branch one level under the node it leaves, reroutes left out — with why a node cannot be
switched off (`document.pass_through_reason`, CON-18).

see: WF-08 · CON-18 · [MANUAL §2 Pages, §2b](../MANUAL.md)

---

### CON-18 — linked page
anchors: sym:nodelab_v2.linked_document.LinkedDocument, sym:nodelab_v2.linked_document.LinkedDocument.touch, sym:nodelab_v2.linked_document.LinkedDocument.set_edit_mode, sym:nodelab_v2.linked_document.LinkedDocument.structure_dict, sym:nodelab_v2.linked_document.LinkedDocument.set_muted, sym:nodelab_v2.document.pass_through_reason

A **linked page** is a page whose document is a `LinkedDocument`: a live mirror of its
**master** page — the same nodes, wires, positions, frames, zones and groups, rebuilt IN PLACE
on every master change so record identity survives and the canvas does not rebuild its cards.
What it owns are **overrides** `{node id: {"params", "modes", "local"}}`: a param or mode set
on the linked page differs from the master's and stays the page's own (sticky, even if the
master later reaches the same value) until *Reset to master*. Whether a node is switched off
is a value of the same kind since step 11e: `set_muted` records `"muted"` in its override.
One level deep: a link to a linked page links to its master. A file stores a linked page as
`master` + `overrides` (+ `structure`, below), no `graph`.

**Its topology answers to `edit_mode` (step 11e).** Unset, add/remove/connect raise
`LinkedPageError` — the window asks first (`MainWindow._topology_ok` → `ask_linked_edit`, the
`linked_edit_dialog`) and sets the answer: *Make unique* swaps in a plain document;
`EDIT_MODIFIED` (saved) keeps the edit on the page — the base `GraphDocument` op runs on the
live mirror and `_record` diffs it against the master into the page's STRUCTURE (own nodes,
ids `nL1`…; master nodes removed; wires added and removed; `structure_dict`), which `_mirror`
lays back over the master on every master change, this page's wire winning a single input;
`EDIT_MASTER` (this session only) makes the edit ON the master in a form that changes nothing
it computes — an added node arrives there muted (and on here), a wire or a removal is allowed
only if `_keeps_master` finds the master's bypassed run graph unchanged, a muted node's removal
heals its chain. Frames, groups and zones stay the master's in every mode (`SHAPE_HINT`).
Muting at all is bounded by `document.pass_through_reason`: only a node that keeps the kind of
data (it adds no domain, named layer, extra layer, column or picture; not a boundary, Iterate
control or reroute) may be switched off — the same rule makes a node eligible to go to the
master switched off.

The memo's recipe hash carries no node id, so a linked page shares with its master every
cached result it READS through a Page Input from a shared upstream page, and every node it does
not override that is fed only by those. A root on the linked page itself (an `io.load`, a Dock,
any seeded source) carries that page's own run id as `__source__`, so a linked Image Input
page computes its own chain.

Since step 11 the switcher labels a linked page `(linked · N overrides)`, and *New page… ▸
Linked to a master page* makes one with its Page Input's `source` already written as an
override — `source` is an ordinary param, so a copy reads a different Output than its master
without touching the master (CON-21).

see: CON-17 · CON-09 · CON-21 · [MANUAL §2 Linked pages](../MANUAL.md)

---

### CON-19 — picture dataset
anchors: sym:nodegraph.catalog._shared.figure.picture_dataset, sym:nodegraph.catalog._shared.figure.FigureProvider, sym:nodegraph.catalog._shared.figure.figure_frame

A `plot.*` node returns a **Picture**: a Dataset whose image is the rendered figure — axes
`(1, T, 1, 3, H, W)` uint8, channels R/G/B with their own colours, `bit_depth` 8, no
calibration, `picture = "rgb"`, and the JSON `figure_spec` it was drawn from (Export Figure
redraws that spec at any resolution, or as SVG/PDF). The node declares
`NodeSpec.fresh_output`: the envelope inherits no domain, layer or column of its input, and
its own Voxel domain comes from `adds_domains`. Viewer, Export Movie and LabLink quicklooks
show a picture on a fixed 0–255 window in its channel colours.

`T` is 1, except with `per = frame` (Plot XY, Plot Time Series): one figure per input frame,
drawn on read by a `FigureProvider` (a `TileProvider` that renders frame t, keeps 8, draws
each frame once for concurrent readers, counts its resident bytes for the memo budget and
`computes_on_read` for the runner's prefetch), on axes every frame shares, with the input's
per-frame clock on payload and envelope alike. Renders are serialised process-wide
(matplotlib's text layout is not thread-safe) and refused past 100 Mpx.

see: CON-11 · [MANUAL §15 Plotting](../MANUAL.md)

---

### CON-20 — dock shell and layout memory
anchors: sym:nodelab_v2.shell.DockShell, sym:nodelab_v2.shell.PanelDock.paintEvent, sym:nodelab_v2.layout_store.load_layout, sym:nodelab_v2.window.MainWindow._apply_default_sizes, sym:nodelab_v2.viewer_controls.ViewerControlsPanel, sym:nodelab_v2.viewer.ViewerPanel.detach_controls

Every panel is a `PanelDock` (a `QDockWidget` with a custom `PanelTitleBar`: kind glyph, title,
`+` on a multi kind, float/dock, ✕), registered by kind in the `DockShell` from the window's
`PanelSpec`s. The main canvas is the window's centre; everything else floats, closes and tabs.
**A dock with a custom title bar is painted by nobody** — `QDockWidget::paintEvent` returns
without drawing — so a floating dock showed Qt's default light palette through its frame
gutter until step 11: `PanelDock.paintEvent` fills the dock with `BG` and draws a 1 px `BORDER`
edge when floating, the `QDockWidget` stylesheet rule carries a background and a 1 px border
(the frame width is derived from it; `border:0` would hide the painted edge under the panel),
the panel bodies set `WA_StyledBackground` (a QWidget SUBCLASS does not auto-paint its own
stylesheet background), and `theme.palette()` is pushed as the application palette so
dialogs, popups and empty viewports follow the theme. The ✕ is disabled when the window's
`_allow_panel_close` would veto the close (`sync_close_buttons`), which is a pure predicate;
the un-maximize that a closing mini-map canvas needs happens in `_prune_canvases`. Panels
grouped as tabs carry their tabs on top (`setTabPosition(…, North)`; Qt's default is South).

**The Viewer's controls are panels (step 11d).** A `ViewerPanel` builds two SECTIONS
(`axes_section`: M/T/Z strips, play, the overlay-source and iteration strips;
`channel_section`: the display tools over `ChannelColumns`, which wraps the per-channel
columns onto rows when narrow). The window takes both out of every Viewer it makes
(`detach_controls`) into the single `playback` and `channels` docks
(`viewer_controls.ViewerControlsPanel`, a stack of every Viewer's section), which show the
ACTIVE Viewer's (`MainWindow._sync_viewer_controls`; a linked Compare viewer shows its
leader's strips). The sections stay the Viewer's widgets — every slot and attribute is
unchanged — and carry its stylesheet. Maximized, the mini-map Viewer takes them back
(`attach_controls`, compact) and the docked control panels hide. A `PanelSpec` may name
`split_from`/`split` for its default place, and `restore_layout` puts a dock the saved layout
never heard of in that place (`DockShell.place_default`) rather than wherever Qt left it.

**Layout memory.** `layout_store` keeps `~/.nd2studios/layout.json` — Qt's `saveState`
bytes plus the multi docks' bindings — under `LAYOUT_FORMAT` / `LAYOUT_VERSION`
(`nd2studios.layout/2` since step 11, when the Viewer became visible from launch); a file of
another version is set aside as `layout.json.rejected` once and the default layout applies
(`_apply_default_sizes` on the first show: Viewer `VIEWER_SHARE` of the canvas column, Nodes
`PALETTE_W`, Properties `INSPECTOR_W`). `NODELAB_LAYOUT=0` disables the file (every probe),
`NODELAB_LAYOUT_FILE` redirects it.

see: [MANUAL §2 Panels](../MANUAL.md) · CON-17

---

### CON-21 — page recipe
anchors: sym:nodelab_v2.page_recipes.PageRecipe, sym:nodelab_v2.page_recipes.apply_new_page, sym:nodelab_v2.page_recipes.instantiate, sym:nodelab_v2.workspace.Workspace.set_master

A **page recipe** is one page's graph — its `to_page_dict` body (nodes, wires, zones, groups,
positions) — under a name, a kind and a line of description, as a starting point for a new
page: *New page… ▸ Page recipe*. Built-ins ship in `nodelab_v2/builtin_page_recipes/<kind>/<slug>.json`
("Smooth & threshold", "Label & measure", …); the user's own are written by *Save as page
recipe…* to `~/.nd2studios/page_recipes/` (`NODELAB_PAGE_RECIPES_DIR` redirects,
`NODELAB_PAGE_RECIPES=0` disables) and shadow a built-in of the same kind and name. A file is
`{"format": "nd2studios.page-recipe/1", name, kind, description, app_version, page}`; a file
that is not one is skipped, never trusted. `instantiate` adds a page (or takes over an empty
one holding only its seeded Page Input), loads the body through `Workspace.load_page_body`
(so `_OP_RENAMES` apply), binds its Page Inputs to the chosen Output or the page's default
source, and keeps its Output names unique. It is **not a LabLink recipe** (a whole graph
published for a hub, `nodelab_v2.lablink.recipe`): both labels carry their qualifier.

The *New page…* dialog settles kind, name, start — empty / page recipe / **linked to a master
page** — and the Output the page reads; `apply_new_page` applies the `NewPageSpec`: a linked
start is `Workspace.duplicate_page(dependent=True)` with the chosen source written as that
page's override. **Masters:** `Page.is_master` (★ in the switcher, in the file only when set;
never on a linked page) puts a page first in the dialog's *Master* menu — any plain page may
still be chosen. `Workspace.sources_for_kind` / `default_source_for_kind` answer for a page
that does not exist yet.

see: CON-17 · CON-18 · [MANUAL §2 Pages](../MANUAL.md)

---

### CON-22 — data part and several-item Output
anchors: sym:nodelab_v2.ops.materialize_part_taps, sym:nodelab_v2.ops.materialize_output_items, sym:nodelab_v2.ops.materialize_input_items, sym:nodelab_v2.document.GraphDocument.data_parts, sym:nodelab_v2.document.GraphDocument.output_items, sym:nodegraph.metadata._kept_only

**Parts (V4.00 step 11f).** A node's `out` carries its whole Dataset — the image and every
named layer (a mask raster; a label raster and its structure table, ONE part under one name,
`ops.part_of`; a point or track table). `GraphDocument.data_parts` lists them from the
node's envelope (`image` + each `layer_names` name) when there are two or more, and
`output_specs` offers each on a synthetic `part:<name>` socket. Like the channel taps, a part
wire is a GUI fiction: `prepare_run_graph` rewrites it (`materialize_part_taps`, last) through a
hidden `data.part` node (`PART_OP`, a GUI-layer op beside `page.*`, hidden from the palette)
fed by the node's real Dataset output — one tap per (node, part), shared. Its compute keeps
that part alone: `image` → the image with no attributes; a layer → that layer's attributes,
its VOXEL raster (same shape as the image on the voxel lattice, so no copy) becoming the
image (bool viewed as uint8), `bit_depth` dropped; a table-only part has no image. At edit
time the engine's `NodeSpec.keep_layers` (`metadata._kept_only`) keeps only that name's
layers, their columns and the structure domains they live on — the general form, total like
`extra_layers`. `_bypass_muted` keeps a part wire pointing at its part through a muted node.

**Several items.** `page.output` declares `data` plus `data_2` … `data_8` in one
`grow_group` (one empty slot shows) with `passes_domains=False`, and an `items` presentation
param naming them; `GraphDocument.output_items` resolves `(socket, name)` per wired slot —
the entry for that slot, else from the wire (a part's name, a Page Input's variable, the
source node's title), unique within the Output. `materialize_output_items` (first) gives every
extra item a `page.output` node of its own (`__item__<output>__data_2`), so the Output keeps
item one only — a pull previews it and computes nothing else — and each item is a run node
with the Output's condition stamp. A `page.input` reading a several-item Output gets
`item:<name>` sockets (`doc.page_items` → `Workspace.input_items`); `materialize_input_items`
turns each read into a `page.input` tap (`__tap__<input>__item_<name>`, `PAGE_ITEM_KEY`) that
`Workspace._input_seeds` seeds with that item's envelope and `compose` splices onto the item's
node (`Workspace.resolve_item`). The cross-page signature walks the UNmaterialized graph, where
every item wire still enters the Output, so editing any item's chain stales its readers.

see: CON-17 · CON-12 · CON-15 · [MANUAL §4 One kind of data at a time, §2 Pages](../MANUAL.md)


---

### CON-23 — stream identity: channel, position, the name it was given
anchors: sym:nodelab_v2.ingest.channel_display_seed, sym:nodelab_v2.document.GraphDocument.channel_subset, sym:nodelab_v2.document.GraphDocument.stream_identity, sym:nodelab_v2.document.GraphDocument._name_output_from_wire, sym:nodelab_v2.document.GraphDocument.source_channel_total, sym:nodelab_v2.workspace.Workspace.input_channel_scope

**The names ride the data (V4.00 step 11g).** A source envelope — the Load card's seed at
file-pick (`window._add_source_node` / `_add_bundle_node`) and the runner's resolved
envelope (`_ingest_locked`, `_open_direct`, `_resolve_bundle`, cached in `_providers` so
`_fresh_envs`' re-seed says the same) — carries `channel_names` and `channel_colors` through
ONE function, `ingest.channel_display_seed` (`with_channel_display`): one spelling on both
sides, because `set_meta_seed` re-seeds only on a CHANGED envelope and two spellings would
cost a re-pull after every first pull. Both keys are in `metadata.PER_CHANNEL_KEYS`, so
`channel_select` (the `chK` taps, `channel.select`) narrows them in lockstep with the axis;
`position_name` is in `PER_POSITION_KEYS` and the position taps narrow it the same way —
and `position_subset` INVENTS `position_index` on the first narrowing (the indices it kept;
step 11i), so a position with no point name is named by its index in the FILE (`m1`), not in
the one-long stream it has become (`m0`, which is what every single position used to read).
Until this the edit-time envelope carried only `channel_emission_nm`: past a `chK` wire a
card could tint by emission but could only call its stream `Ch0`. The engine's "calibration
schema" rule for the seed stands otherwise — `ctx.calib` still refuses a non-calibration key;
these are display lists the engine never reads.

**One resolution, three readers.** `GraphDocument.socket_channels(node, socket, io)` — a
`chK` output is channel K of `channel_descriptors`; a Page Input's `item:<name>` is what that
item's wire carries (`page_channel_scope`); any other output the node's own list; an input its
wire's source socket. `channel_subset` keeps them only when `1 <= len <
source_channel_total` — the wire tint's long-standing rule — and a `fresh_output` node (a
plot's picture) has none of the file's. `socket_positions` / `position_subset` /
`source_position_total` are the same three for M (step 11h; a stream's position names off
its envelope, `_env_position_names`). `stream_name` is the name a stream was given at the
Page Output it last came through — the Page Input's variable, or the item for an
`item:<name>` socket — followed back along the PRIMARY Dataset wire (the first Dataset input,
the one the envelope follows) through any node. `stream_identity` composes them: the name,
then the positions and channels the name does not already mention, each joined as `a · b` or
`a +N`. `socket_text` prints it — a generic `data`/`out` becomes the identity, a named `raw`
becomes `raw · <identity>`, a synthetic socket keeps its label (`_SYNTHETIC_SOCKET_RE`) —
and the dot (`NodeItem._tint_channel_socket`, inputs too, re-applied in `refresh`'s cheap
branch because a wire arrives after the card is laid out), the wire
(`EdgeItem._channel_colors`) and the hover (`NodeItem._stream_facts`) read the same
resolution, coloured by `node_item.desc_qcolor` (native colour, else emission). The totals
cross a page: a root `page.input` asks `Workspace.input_channel_scope` /
`input_position_scope` (hooks `doc.page_channel_scope` / `page_position_scope`) for the
FILE's totals on the page it reads from, recursively, so a one-channel, one-position stream
three pages down still knows it is one of three. `_inherited_channel_descriptors` narrows
through a `chK` wire instead of stopping at it (an older seed without names still yields the
file's name past a tap) and crosses a page by the ITEM's wire, never the Output's — whose
list is its first item's, which is how three masks of a split all came through as "GFP".

**An Output names what it carries.** `_item_default` names an item from its wire: a part or
item name; a tap socket's channel, position, group key or batch member (`_tap_name`); a Page
Input's variable; else the source title, with the one position and one channel it does not
say appended (`gaussian_blur_b03_cy5`). `connect` hands a Page Output the same answer for its
VARIABLE while its name is still the placeholder (`_PLACEHOLDER_NAME_RE`: `out`, `out2`) or
blank — `_name_output_from_wire`, first slot only, settled unique by `_settle_output_name`,
never over a typed name; a reader already bound to the old name follows through the
`output_renamed` hook (`Workspace._follow_renames`). A `LinkedDocument` sets
`AUTO_NAMES_OUTPUTS = False`: its names are its master's.

see: CON-17 · CON-22 · CON-12 · [MANUAL §4 A stream knows where it came from](../MANUAL.md)

---

### CON-24 — units, math and the region crop
anchors: sym:nodegraph.units.conversion_factor, sym:nodegraph.units.unit_of, sym:nodegraph.catalog.math.values._compute_math_values, sym:nodegraph.catalog.util.crop_region._compute_crop_region, sym:nodelab_v2.ops.materialize_outside_taps, sym:nodegraph.metadata.crop_region

**Units (V4.00 step 12).** A number a Dataset carries — a Global scalar, a structure-table
column, a lattice layer — has a UNIT, read by `nodegraph.units.unit_of` in two steps: the
record a producer wrote under `ds.metadata["units"]` (key `<domain>:<layer>:<name>`, written
by `with_unit`; the Math nodes and `analysis.reduce_scalar` record what they produce), else the
catalog's NAMING convention (`unit_of_column`: `area` is `px2` on a 2D table and `vox` on a 3D
one by the table's `z_kind`, `x`/`y` are `px`, `z` a plane step, `t` a `frame`, `*_um` is
`um`, `*_intensity` is `counts`, Object Metrics' velocities `um/s`; `n_*`, ids and fractions
dimensionless). Unknown is `None` and STAYS unknown through arithmetic — never silently
dimensionless. A unit is a dict `{base: exponent}` over `um px zpx s frame counts rad` (+
scaled spellings `nm mm ms min h deg`; `vox` = `px²·zpx`), with one canonical spelling
(`format_unit`, pretty `µm²`, slug `um_per_s`). `conversion_factor` converts: a fixed factor
between scaled spellings, the CALIBRATION between a pixel and a micron (`pixel_size_um`
laterally, `z_step_um` axially — hence `vox → um3`), between a frame and a second (`dt_s`);
a conversion that needs a key the data lacks is refused BY NAME, different dimensions as
such. Why metadata + convention rather than a field on `AttributeLayer`/`StructureTable`: the
catalog already encodes units in names and no engine type changes; the record covers what a
name cannot say, and both travel with the payload through every node that keeps the numbers.

**The Math nodes (category `math`, role `arithmetic`, every page).** `math.mask` — set
algebra on Voxel rasters (any raster is a mask, nonzero = inside): `subtract/union/intersect/
xor/invert`; an EMPTY A is the whole frame, so `subtract` with B = the mask is the background;
B from `other` broadcasts over m/t/z/c where it is size 1 (`_shared.rasters.broadcast_raster`);
writes a uint8 mask recorded dimensionless. `math.image` — pixel arithmetic with another image
or a constant, LAZY per plane (its own `MapComputeProvider`, fp folding the other provider's
fingerprint), `other` may be a projection / one timepoint / one channel of the same field
(broadcast; sampling provenance compared on the axes it does not collapse); the
`metadata.image_math` transform widens `bit_depth` by a bit for `add`, drops it for
`multiply`/`divide`, keeps it otherwise, and the payload syncs through `ctx.calib`.
`math.values` — arithmetic on a domain lever's attributes (`reduce_scalar`'s shape): A on
`data`; B a layer/column on `other` or `data`, a Global scalar (broadcast), or a constant with
`value_unit`; same-unit ops convert B into A's unit, products/ratios compose, `power`/`sqrt`/
`log10` follow dimensional analysis, `to_physical`/`to_pixels` convert A; the result is a new
column beside A (table rebuilt through `with_structure`) or a layer/scalar, auto-named from the
operands (`_out_name`, shared by `extra_layers`/`adds_columns` so the edit-time catalog predicts
the pull), unit recorded.

**Crop by Region (`util.crop_region`).** Crop to a raster, not a rectangle: the region is any
Voxel layer on `data` or on the `regions` wire. `extent = frame` masks in place (identity
axes, no stamp); `fit` windows to the box around everything inside (+ `margin` on y/x), per
position, padded to a common size; `each` makes every OBJECT a position — connected components
of the footprint over all frames (8-conn on the Z-collapsed plane in 2D, 26-conn in 3D) or one
per label id — each in a common-size window at its own corner, named `obj1…`/`<pos>_obj1…`
with `position_index` retired. One `widx6` raster (which window owns each kept voxel) serves
the pixel fill, the Voxel-layer windowing and the row test alike; Label/Point rows outside are
dropped and the rest shifted onto their window and position. `origin_um` is restated per
window from `placement.field_box` + `sub_field_box` (z by the Z step when planes were cut) and
the stage keys retired, so placement reads the Dataset and stays exact; `metadata.crop_region`
says Y/X (fit; +Z in 3D; +M for each) are UNKNOWN and drops the geometry keys at edit time.
`keep = outside` is the complement at the frame's extent whatever the extent mode.

**The `outside` socket.** The card carries a second Dataset output, `outside`
(`GraphDocument.output_specs` for `CROP_REGION_OP`; a synthetic socket, so it keeps its label
in `socket_text`). The engine is one-payload-per-node, so `ops.materialize_outside_taps` —
first among the tap passes in `prepare_run_graph`, so the wires it copies are then tapped like
the original's — turns a wire from it into a SIBLING node `__tap__<node>__outside`: the same
op, params and modes with `keep` flipped, fed by copies of every wire the crop node receives.
The sibling is an ordinary node to the engine, the memo and `propagate_meta`, which is why the
consumer's envelope is the frame-sized identity while the inside's is the unknown box.

see: CON-22 · CON-23 · [MANUAL §10b Math with units, and cropping by a region](../MANUAL.md)
