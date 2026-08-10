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

The eleven kinds of thing an attribute can be attached to: the acquisition lattice
(`voxel`, `plane`, `frame`, `timepoint`, `multipoint`, `global`, `channel`) and the structures
built on it (`label`, `point`, `track`, `mesh`). A Dataset is not "an image" — it is a bundle
of attribute layers, each living on one domain, and a node declares which domains it *reads*
and which it *adds*.

This is why "segment then measure" needs no special case: segmentation adds a `label` domain,
measurement reads it. The prose for each domain is in `gen/vocab.jsonl` (`"group":"Domain"`).

see: CON-08 transfer · [MANUAL §10](../MANUAL.md)

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

see: [WF-02](workflows.md) · INV-02 · ENGINEERING_NOTES §11–12

---

### CON-12 — layer sockets
anchors: sym:nodegraph.registry.SocketSpec, sym:nodegraph.registry.define_node

A socket whose value is the *name* of an attribute layer rather than data: `layer_in` picks an
existing one (the GUI offers a picker populated from the envelope's layer catalog),
`layer_out` declares one this node creates. It is how "measure the `distance` layer" is
expressed without a socket per possible layer.

A layer a node creates that has no output socket must still be declared, via `extra_layers`,
or the picker downstream will not offer it.

see: CON-06 · the `wire-node-v2` skill §4c

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
