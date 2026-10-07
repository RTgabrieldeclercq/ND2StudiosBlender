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
i.e. at file-pick time in the File menu, before a pixel is read. Two so far: a zero-range
Z-stack that divides by zero, and picture metadata keyed `ImageMetadataSeqLV|N!` with `N != 0`
(upstream hard-codes `|0!`). Each shim wraps upstream and acts **only on the raised
exception**, so a file that opens today opens bit-for-bit identically. That is what makes
patching a third-party parser safe, and it lets each shim self-sunset once a fixed release
stops raising. Where a shim has to redo upstream's work, it copies upstream's own
lines (shim 2 decodes a different key exactly the way upstream decodes `|0!`) and does not
derive anything new. The import also stays lazy and in-function so `ingest.py` remains importable,
and its TIFF half usable, with no SDK installed.

---

### INV-08 — `nodegraph/` never imports Qt
anchors: sym:nodelab_v2.runner.EngineRunner, file:nodegraph/codemap.py

One arrow: `scripts/` → `nodelab_v2/` → `nodegraph/`. The single sanctioned reversal is
`nodegraph/selftest.py`, which imports `nodelab_v2` to test the Qt-free GUI seam. Adding a
second would mean the engine can no longer run headless, and the headless gate is the one that
runs everywhere.

Corollary for GUI code: `document.py`, `ops.py`, `tables.py`, `overlay_render.py`,
`workspace.py` (V4.00) and the headless half of `picker.py` must stay import-clean of Qt,
because that is what lets the selftest cover them at all.

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

Every kernel in `nodegraph/kernels/` that takes a voxel size takes it that way:
`(z_step_um, pixel_size_um, pixel_size_um)`. A `(dx, dy, dz)` swap corrupts anisotropy and
every µm column downstream without raising anything. It is the single most-repeated warning
across the per-kernel `.md` contracts, which is why it is here rather than only there.

One kernel takes no voxel size at all: `track_field` requires its `coords` and `disp` to
arrive in ONE physical length unit already, so its gradient is dimensionless with no
anisotropy factor left to apply. That is the same invariant reached from the other side, and
the caller (`analysis.track_field`) is where the `(dz, dy, dx)` scaling happens.

The invariant is the KERNEL's boundary, not the upstream package's. `aldvc_field` wraps
pyALDVC, whose `DVCPara` triples are `(x, y, z)`, and `dic_correlate` wraps pyALDIC, which
works in `(x, y)` — both still take `voxel_size_um` slowest-first and reverse it *inside*.
Passing `(x, y, z)` to match the upstream package's own order is the exact silent corruption
this entry exists to prevent.

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

Not every kernel is vendored, and this rule says nothing about the ones that are not:
`cellsam_segment`, `piv_field` and `aldvc_field` (rewritten 2026-09-25) are adapters around
third-party packages, and `track_field` (2026-09-17) is new in-repo math. `field_math`
(2026-09-25) is a third case: it is byte-verbatim code that was *lifted out of* a vendored
kernel when the official pyALDVC package replaced the in-repo ALDVC port, so the verbatim
rule still governs it — it just no longer lives where it was vendored to.

An adapter is not exempt from the spirit of this entry. Its own risk is not editing the
maths — it cannot, the maths is in the package — but silently reordering, negating or
rescaling the answer on the way out. That is why `scripts/_aldvc_validate.py` scores the
ADAPTER against analytic truth rather than re-running upstream's benchmark.

Where new code is *derived* from a vendored kernel, the
equivalent guarantee is an explicit equivalence test rather than byte-identity —
`track_field.mls_displacement_gradient` is a vectorised re-derivation of
`track_objects.compute_strain_mls` and `selftest::test_track_field` pins the two together on
a fixture, so the derivation cannot silently drift from the code it came from.

**`registration` left verbatim status on 2026-10-07**, and is the worked example of the only
way a kernel may: it is the lab's OWN v1 code (vendored from this project's predecessor, not
from a paper, so there is no published result to stay comparable to), a known-answer bench
(`scripts/registration_synthetic_bench.py`, with `--legacy` re-creating the old behaviour)
measured the old code against exact synthetic truth, found a sign bug in the ECC seed and a
3–15× precision loss from phase whitening, and the departures are listed with their numbers
in `nodegraph/kernels/registration.md` §12 and pinned by
`selftest::test_registration_refinement`. A departure without all three — in-house
provenance, a bench that reproduces before and after, a selftest fence — is still forbidden.

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

---

### INV-14 — a compute must never return fewer batch members than it was given
anchors: sym:nodegraph.engine._check_batch_kept, sym:nodegraph.streaming.realize

Roughly 25 realizing nodes allocate a plain `(m,t,z,c,y,x)` raster and fill it from
`get_region(...)`. The batch index is **keyword-only with a default of 0**
([CON-11](concepts.md)), so such a loop reads the FIRST file, writes it into an array
shaped like the whole batch, and hands back a Dataset whose `b` has quietly become 1.
Every other file is gone.

Nothing about the result looks wrong. The pixels are real, the axes are self-consistent,
the row count is plausible, no exception is raised — the single witness is that two files
went in and one came out. `realize()` had exactly this bug (fixed 2026-09-27) and it sat on
the export path, so a batched write-out produced file 1 under the batch's name.

Two guards, deliberately independent, because each catches cases the other does not:

* a node that adds a **lattice layer** is caught by the shape check in
  `Dataset.with_attribute`, which names the missing batch axis;
* a node that returns only an **image** is caught by `_check_batch_kept` in the engine,
  the one place every compute passes through.

Both **refuse** rather than repair: the correct per-member result is something the node has
to compute, and one member's output carries nothing to reconstruct the others from.

`nodegraph.selftest.test_batch_never_silently_dropped` runs the whole catalog against a
two-file batch and fails if *any* node returns fewer files. As of 2026-09-27: 28 carry both
through, 11 refuse loudly, 0 drop. Lifting one of the 11 means giving its unit loop the
batch axis — never widening the guard.

---

### INV-15 — run ids are ALWAYS page-qualified
anchors: sym:nodelab_v2.workspace.qualify, sym:nodelab_v2.workspace.split_run_id

Every id that reaches the engine, the memo or a runner signal in the GUI is `<page>/<node>` —
the target page's own nodes included. Cross-page reuse depends on it: a root's `__source__` is
its run id, and two pages that both have an `n1` would otherwise share a source identity (and
a cached result) they do not share in fact. Two places read a bare id, each deliberately:
`EngineRunner.run_id` takes it to be on the active page, and `Workspace.compose` on the page
being composed. Everything past them carries the qualified id. A Page Input's reference uses `:` (`pg1:raw`), never the run separator.

---

### INV-16 — `page.*` ops are GUI-layer and never live in `nodegraph/`
anchors: sym:nodelab_v2.ops.ensure_ops

`page.input` and `page.output` are registered by `nodelab_v2.ops.ensure_ops`, beside
`io.load`, `view.viewer` and `io.dock`: they only mean something inside a `Workspace`, which is
a `nodelab_v2` object. Moving them into the catalog would make the engine depend on the
workspace model (INV-08) and put two ops into the catalog sweeps that cannot run headless on
their own (an unbound Input refuses by design). The selftest's catalog sweeps
(`hotreload.is_catalog_op`) therefore skip them, and `selftest.test_workspace_*` and the
LabLink page tests cover them. They ARE in the catalog snapshot baseline and the synopsis
(both call `ensure_ops`, role `page_boundary`): a change to their sockets or descriptions
needs `_catalog_snapshot.py save` and `_node_synopsis.py write` like any node's.

---

### INV-17 — a node demo defines no op, touches no live runner, and every op has one
anchors: sym:nodelab_v2.demo_recipes.DemoSession, sym:nodegraph.phantom.phantom, sym:nodelab_v2.demo_recipes.validate_curation

The *What does this node do?* window runs a node on a phantom through a **throwaway engine**:
`DemoSession.build_graph` seeds an `io.load` with the phantom's Dataset and MetaEnvelope
exactly as the runner seeds one, and `headless_engine` runs `src -> prelude -> demo` on the
session's own `Memo` and `TileCache` (shared across rebuilt engines, the cache rule in
`Engine.__init__`). Nothing calls `define_node` (INV-03), nothing reads the runner's providers,
and a param the user has not touched is **absent** from the node so the engine derives it
itself. Synthetic datasets live only in `nodegraph/phantom.py` (deterministic by seed; the
cached object is the identity); the per-op curation only in `codemap/node_demos.json`, which
`selftest.test_node_demos` validates against the live registry and then runs for EVERY op —
a guide must carry curated features, a live recipe must produce the evidence its kind
promises, a `live=false` one must say why. A node that cannot demonstrate itself does not pass.
A recipe may offer several synthetic worlds (`scenarios`, 2026-10-07 — Registration's eight are
the registration bench's worlds at demo size); the gate runs every one of them, and the
Frame-domain values a node writes are read back at the viewed frame so the demo can be
checked against the phantom's stated truth, not merely seen to run.
