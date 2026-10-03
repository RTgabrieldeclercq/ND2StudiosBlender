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
or the picker downstream will not offer it.

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
