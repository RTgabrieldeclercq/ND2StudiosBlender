# Invariants

Rules that are cheap to break and expensive to notice. Each one below cost somebody real
debugging time, and each is one an agent can violate while every test still passes.

**This is a selection, not the list.** `ENGINEERING_NOTES.md` §19 is the full inventory —
registry, memo, streaming, metadata, the `nd2` seam, kernel-port conventions, determinism,
numba, GUI. What is extracted here is the subset that (a) fits in a paragraph, (b) can be
violated silently, and (c) can be pinned to code so the gate keeps this page honest. When you
are about to touch one of those subsystems, read §19's block for it — it is a few dozen lines.

---

### INV-01 — memo identity is `revision`, never `id()`
anchors: sym:nodegraph.revision.next_revision, sym:nodegraph.memo.node_recipe_hash

Use the process-monotonic revision counter. `id()` is reused the moment an object is freed, so
a memo keyed on it serves a *different* object's payload — intermittently, and only under
memory pressure, which is the worst possible failure schedule. A content hash is correct but
too expensive to take on every lookup.

---

### INV-02 — construct a streaming provider LAST in a compute
anchors: sym:nodegraph.streaming.TileCache, file:nodegraph/streaming.py

A provider's fingerprint folds in the calibration reads and field-expression hashes recorded
*at the moment it is built*. Anything the compute reads afterwards is not in that fingerprint,
so the tile cache will happily serve a tile computed under a different calibration. The tile
looks perfectly valid; nothing errors.

Related, same section of §19: provider fingerprints must be **flat digest strings** (nested
tuples blow up `_canon` on deep unrolled chains), and no `ctx.calib` / `ctx.meta` may be read
from inside a lazy closure — the frozen-`ReadContext` guard exists to catch exactly that.

---

### INV-03 — a test fixture must never `define_node` a real `op_key`
anchors: sym:nodegraph.registry.define_node

`NODES` is process-global. A fixture registering a real key clobbers the shipped node for
every test that runs after it, producing the "passes alone, fails in the suite" failure —
which then gets blamed on whichever unrelated test happened to run next. Use `test.*` /
`eng.*` keys. Likewise, do not poke `runner._providers[("synthetic",)]`; build a throwaway
`EngineRunner`.

---

### INV-04 — declare both halves of a calibration or axis change
anchors: sym:nodegraph.metadata.propagate_meta, op:util.crop

A node that changes geometry must declare the `meta_transform` (edit time) **and** stamp the
payload (pull time), and the two must agree exactly — including one-sided and out-of-range
inputs, where `span()`/bounds semantics differ. If they disagree, the GUI displays one
geometry and the data has another, and every downstream µm value is quietly wrong.

The selftest asserts `engine.env(node).axes` equals the pulled payload's axes for
axis-changing nodes. Add that assertion for any new one.

---

### INV-05 — do not double-count a relative calibration change
anchors: sym:nodegraph.metadata.eval_derive, sym:nodegraph.metadata.envelope_symbols

`ctx.calib` already returns the **post-transform** value. A resample must *sync* the payload's
pixel size to the envelope value, not divide by the scale again. And never reach into
`ctx.inputs[i].metadata[<calibration key>]` — that bypasses the read fence, so the memo will
not invalidate when the calibration changes. It is a hard error under `strict_reads`.
(Non-calibration provenance keys are the deliberate exception.)

---

### INV-06 — the edit-time pass must never raise
anchors: sym:nodegraph.metadata.propagate_meta

`propagate_meta` and `extra_layers` run on **every keystroke**. Their caller catches only
`ValueError`, and even a caught error blanks every node's envelope graph-wide — so a typo in
one node's resolver makes the whole canvas look broken. Write them total: no unguarded
indexing, no assumption that an upstream layer exists.

---

### INV-07 — go through `nd2_compat.import_nd2()`, never `import nd2`
anchors: file:nodelab_v2/nd2_compat.py

The SDK has bugs that make a *healthy* file unopenable, and they fire from `ND2File.sizes` —
i.e. at file-pick time in the File menu, before a pixel is read. The shim wraps upstream and
acts **only on the raised exception**; it never reimplements the calculation, which is what
makes patching a third-party parser safe and lets the shim self-sunset when a fixed release
stops raising. The import also stays lazy and in-function so `ingest.py` remains importable,
and its TIFF half usable, with no SDK installed.

---

### INV-08 — `nodegraph/` never imports Qt
anchors: sym:nodelab_v2.runner.EngineRunner, file:nodegraph/codemap.py

One arrow: `scripts/` → `nodelab_v2/` → `nodegraph/`. The single sanctioned reversal is
`nodegraph/selftest.py`, which imports `nodelab_v2` to test the Qt-free GUI seam. Adding a
second would mean the engine can no longer run headless, and the headless gate is the one that
runs everywhere.

Corollary for GUI code: `document.py`, `ops.py`, `tables.py`, `overlay_render.py` and the
headless half of `picker.py` must stay import-clean of Qt, because that is what lets the
selftest cover them at all.

---

### INV-09 — declare an auxiliary Dataset socket AFTER `data`
anchors: sym:nodegraph.registry.SocketSpec, sym:nodegraph.registry.NodeSpec

The engine takes `dataset_preds[0]` as the calibration/envelope source and walks inputs in
**canonical socket order** — never edge-connect order. A second Dataset socket declared before
`data` silently feeds the wrong envelope to the whole node. This was a real latent bug in
`graph.py`.

---

### INV-10 — `voxel_size_um` is always `(dz, dy, dx)`, slowest-first
anchors: file:nodegraph/kernels/README.md

Every kernel in `nodegraph/kernels/` takes it that way: `(z_step_um, pixel_size_um,
pixel_size_um)`. A `(dx, dy, dz)` swap corrupts anisotropy and every µm column downstream
without raising anything. It is the single most-repeated warning across the per-kernel `.md`
contracts, which is why it is here rather than only there.

Each kernel's own `.md` lists the rest of its conventions — 2D fallbacks that put z in
column 0, threshold kernels that need integer input and a 2-tuple, registration shifts as
`(row, col)`. Read the contract before wiring a kernel; that is what it is for.

---

### INV-11 — vendored kernels are vendored verbatim
anchors: file:nodegraph/kernels/README.md

The only sanctioned deviations from upstream are **version-compatibility fixes**. A semantic
preference — "this default seems wrong" — is not a licence to edit a kernel; the *node* refuses
or exposes a param instead. Otherwise the kernel stops being comparable to the paper it
implements, and the validation suites that score it against published results stop meaning
anything.

---

### INV-12 — anything feeding the memo must be repeat-stable
anchors: sym:nodegraph.memo.output_fingerprint

Non-determinism in a compute makes the output fingerprint change on identical input, which
invalidates every downstream node on every pull. Where a linker's row order affects only id
numbering, a canonical `lexsort` plus a renumber makes it total. Unseeded RNG paths are pinned
off rather than left to chance.

---

### INV-13 — writing onto an existing structure instance must not re-emit `id`
anchors: sym:nodegraph.dataset.Dataset.with_structure

A structure instance is stored as **one array per column** under one layer name, and nothing
revalidates that the columns still agree in length once a second node has written onto it. So a
node that adds per-row columns to a table it did not create must align them to that table's
EXISTING `id` column and leave `id` alone. Emitting `id` from its own row set — the regions
present in the raster, say — leaves the instance **ragged**, which is silent at the seam and
crashes somewhere else entirely: `nodelab_v2.viewer._build_palette` sizes its row map from `id`
and its frame selection from `m`/`t`, so a short `id` beside a long `m` indexes past the end
(`IndexError: index 208 is out of bounds for axis 0 with size 208`, reported 2026-08-06 —
`analysis.histogram_threshold`'s per-parent write-back, fixed by aligning to the table).

The two row sets are genuinely different things: a table can list a region the raster no longer
holds, and a raster can hold ids the table never measured. A parent that was never visited gets
**NaN**, which is why the count columns are float — 0 would claim "we looked and found none".

`analysis.measure` re-emits `id` from `voxel_to_label`'s ids and carries the same latent hazard
wherever its raster and its incoming table disagree. The viewer now clips a ragged layer to the
rows every column agrees on and says so once, because a colour is never worth a crash — but that
is a backstop, not a licence.
