---
name: wire-node-v2
description: >-
  How to define, wire, and metadata-adapt a pipeline node in the nodegraph v2 engine
  (the greenfield, Blender-geometry-nodes rebuild under `nodegraph/`). Read this BEFORE
  adding any v2 node type, changing a node's sockets/modes, touching the 2D/3D lever,
  declaring a data-access footprint (Granularity/kernel_axes), adding a meta_transform,
  or wiring domain transfer / structure bridges. This is the ONLY node-concepts reference
  (the legacy v1 `wire-node` was deleted with v1 on 2026-07-29). Eleven detail sections are
  summarised in the skill and held in full under `references/`, read only when needed.
  Triggers: "add a node", "new pipeline node",
  "nodegraph node", "wire a node", "node sockets", "2D/3D lever", "DimMode", "Granularity",
  "meta_transform", "domain transfer", "structure bridge", "metadata-intelligent param",
  "socket contract", "layer_in", "layer_out", "layer picker", "ctx.layer",
  "unreachable param", "dead socket".
---

# Wiring a node in nodegraph v2 (NodeLab v2)

The v2 engine is a **Blender geometry-nodes engine adapted to microscope image
analysis**, rebuilt greenfield under [`nodegraph/`](../../../nodegraph/) (Qt-free;
GUI lives only in `nodelab_v2/`). Nodes are pure, typed functions of a lazy
**Dataset**; a two-hash memo + lazy pull scheduler decides what actually computes.

> **This is the concepts/contracts reference.** To actually build or modify a v2
> node, use the **`build-node-v2`** skill — it runs the procedure (grill → write →
> footprint/metadata gate → verify gate) and links back here.

**Eleven detail sections live in [`references/`](references/)** and are summarised in place
below — the rule stays here, the worked examples and the failures that motivated it moved
out. Most node work never needs them, and loading 830 lines to reach the socket contract was
costing about 14k tokens a task. **Section numbers are frozen**: `§4d` is still `§4d`
whether you read the summary here or the leaf, because roughly fifteen places across the skills
and the design record cite them, and `selftest::test_codemap` now fails on a reference that
stops resolving.

| leaf | what it covers |
|---|---|
| [`4d`](references/4d-path-sockets.md) | filesystem-path sockets — `path_kind` |
| [`4e`](references/4e-choice-docs.md) | `choice_docs` — one explanation per dropdown option |
| [`4f`](references/4f-reads-domains-by-mode.md) | the domain rail declared per branch |
| [`4g`](references/4g-derivable-layer-names.md) | inferring a layer name instead of defaulting to one |
| [`5b`](references/5b-available-in-any-mode.md) | `available_in` gates any mode, not just `dim` |
| [`5c`](references/5c-mode-gated-modes.md) | a Mode gated on another Mode |
| [`7b`](references/7b-provenance-inheritance.md) | stamp-and-inherit for non-calibration config |
| [`7c`](references/7c-calibration-current-data.md) | calibration describes the current data |
| [`7d`](references/7d-raw-socket.md) | the optional `raw` socket — measure unenhanced pixels |
| [`7e`](references/7e-layer-from.md) | a second input for a second domain — `layer_from` |
| [`12`](references/12-numba-vs-numpy.md) | when numba beats numpy, and `cache=True` |

---

## 1. Mental model — Blender → microscopy (v2 mapping)

| Blender geometry nodes | nodegraph v2 | Where |
|---|---|---|
| Geometry flowing through the tree | one lazy **`Dataset`** (image provider + attribute layers) | [`dataset.py`](../../../nodegraph/dataset.py) |
| Typed sockets; implicit conversions | `SocketType` {DATASET, FLOAT, INT, BOOL, VECTOR, COLOR, STRING} + `can_connect` | [`sockets.py`](../../../nodegraph/sockets.py) |
| Named-attribute layer over geometry | **AttributeLayer**s keyed `(domain, layer, name)` over domains | `dataset.py` |
| Attribute **domains** (point/edge/face) | **domains**: lattice (Global/Frame/Timepoint/Channel/Voxel/…) + structure (Label/Point/Track) | [`domains.py`](../../../nodegraph/domains.py) |
| Field / value inputs adapting to the mesh | **value sockets** carrying `unit`/`derive` (metadata-intelligence) + `is_field` | [`registry.py`](../../../nodegraph/registry.py) |
| A modifier's evaluation dimension | the **2D/3D lever** (`DimMode`, a header ModeSpec) | `registry.py` |
| Viewer node / lazy eval | **lazy pull** — nothing computes until `Engine.pull(node)` | [`engine.py`](../../../nodegraph/engine.py) |
| Node = pure function of inputs + params | Same. No Qt, no global state in a compute body | `nodegraph/` is Qt-free |

**The one rule:** a node declares typed sockets + unit-tagged params + its
data-access footprint; the engine does the routing, the px↔µm math, the memoization,
and the domain transfers. A node never hardcodes a pixel constant and never invents
graph state.

---

## 2. The seam — where a v2 node lives

A node type = a **`NodeSpec`** (registered in the global `NODES` registry) + a
**`compute(ctx) -> Dataset`** function (recorded in `COMPUTES`, keyed by `op_key`).
Both are created by one call in the node's **own module**,
`nodegraph/catalog/<category>/<name>.py` (the per-node split, V2.20 — that is what gives each
node its own memo fingerprint). [`nodes.py`](../../../nodegraph/nodes.py) is now just a facade
whose import loads the catalog; nothing is defined there any more:

```python
register_node(compute_fn, op_key="enhance.median", label="Median",
              category="enhancement", inputs=[...], outputs=[OutDataset()],
              modes=[DimMode()], granularity=..., kernel_axes=..., meta_transform=...)
```

`register_node` (`catalog/_base.py`) = `define_node(**spec)` (registry.py, builds + registers the
`NodeSpec`) + `COMPUTES[op_key] = compute_fn`. The `Engine` looks up the compute by
`op_key` and runs it inside the pull/memo/ReadContext harness. Importing `nodegraph.nodes`
registers every node (its module-level `register_node` calls fire).

---

## 3. Anatomy of a v2 node (the pieces)

1. **`op_key`** — the frozen string identity (`enhance.gaussian`, `analysis.label`,
   `util.crop`). A saved-file contract; never rename. **Test/demo fixtures MUST use a
   fake op_key** (`test.*`) or they clobber a real node in the global registry.
2. **`NodeSpec`** ([registry.py](../../../nodegraph/registry.py)) — `op_key`, `label`,
   `category`, `inputs`/`outputs` (`SocketSpec`s), `modes` (`ModeSpec`s), `description`,
   and the v2 declarations: `granularity`, `kernel_axes`, `meta_transform`,
   `supports_2d`/`supports_true_3d`/`three_d_fallback`.
3. **Sockets** (`SocketSpec`) — `InDataset()`/`OutDataset()` for the main wire;
   `InFloat/InInt/InBool/InVector/InColor/InString` for value/field inputs. Value
   sockets carry `unit`, `derive`, `is_field`, `default`, `dims` (VECTOR arity),
   `available_in` (mode-gated presence).
4. **Modes** (`ModeSpec`, via `Mode(...)` / `DimMode()`) — in-body dropdowns; may
   reconfigure sockets. Fold into the recipe hash (as `params["__modes__"]`).
5. **`compute(ctx)`** — reads `ctx.inputs`, `ctx.calib`/`ctx.meta`, `ctx.params`
   (incl. `__modes__`), `ctx.granularity`/`ctx.is_volume`, and returns a new `Dataset`.

---

## 4. Typed sockets & wiring (`sockets.can_connect`)

- **DATASET↔DATASET** is the main wire. `InDataset(multi=True)` accepts multiple.
- **Value/field sockets** convert implicitly per the conversion table
  (`can_convert`): e.g. FLOAT→INT, scalar→VECTOR (broadcast).
- **VECTOR** arity is **widen-only** (2→3 pads the axial 0; 3→2 rejected).
- A socket may be **field-driven** (`is_field=True`) — a value that varies over a
  domain, evaluated by the field IR ([`field.py`](../../../nodegraph/field.py)).
- Cycles are rejected outside zones (`graph.topo_order`); the engine walks inputs in
  **canonical socket order** so a multi-input node's args land in declared slots
  regardless of edge-connect order.

Choose output by what downstream consumes: a transform → `OutDataset()`; the Dataset
carries new attribute layers / structure tables added by the compute.


### 4b. THE SOCKET CONTRACT — one law for every socket (V2.10/V2.11)

**Read this before adding or editing any socket.** These rules are not style; each one is
enforced by `nodegraph.selftest::test_param_socket_contract`, which fails the build. They
exist because socket declarations and compute code drifted apart repeatedly — a 2026-07-28
sweep found **18 of 55 nodes** with a defect of this class.

**The root hazard: the engine does NOT filter params against the socket list.** It hands
the compute `{**node.params, "__modes__": …}` — params are *overrides*, never
default-filled. So a compute can happily read a param no socket declares: it works
headlessly (a test just passes it) and is **unreachable in the GUI**, which builds its
widgets from `NodeSpec.inputs`. No functional test can see it — the node returns the right
answer for the only value it can ever have. This is a *structural* defect, so the contract
is checked structurally.

**The six clauses:**

1. **Every param the compute reads has a socket.** If you write `ctx.params.get("x")`,
   declare an `x` socket. The only exemptions are documented in `_PARAM_NO_SOCKET_OK`
   (currently one back-compat alias) and *machine-set* params written by the graph builder
   rather than a user (`channel.select`'s `channels`, written by the per-channel tap
   materializer). Prove machine-set by finding the writer; do not assume.
2. **Every socket the node declares is read.** A control that does nothing is
   charter-forbidden (the `track.objects` review). The remedies, in order:
   **(a) gate it** with `available_in` so it only appears in modes that consume it;
   **(b) refuse** — raise if it is explicitly set on a path that ignores it; **(c) delete**.
   Prefer (a). Use (b) when the condition is not a rectangle of mode values —
   `transform.transfer_domain` refuses a non-default `reducer` on a pure broadcast because
   "src drops an axis vs dst" cannot be expressed in `available_in`.
3. **`available_in` must name real modes and real values.** A typo silently hides the
   socket in every state.
4. **A layer-name socket declares its direction and domain** (§4c).
5. **A default is declared exactly ONCE — in the `SocketSpec`.**
6. **Every param socket carries a `description`** — the GUI's hover text (§4h).

**Clause 5 is the subtle one.** Because params are raw overrides, computes used to repeat
their socket's default inline: `ctx.params.get("mask", "mask")`. That is a *second* copy —
and `propagate_meta` needs the same name at edit time to predict the layer catalog, making
a *third*. Nothing kept them in step. Read layer params through **`ctx.layer("mask")`**,
which resolves override → `SocketSpec.default` via the one shared
`registry.layer_value`, the same function `propagate_meta` calls. Never re-spell the
default in the compute.

### 4c. Layer-name sockets — `layer_in` / `layer_out` (V2.11)

A STRING socket naming an attribute **layer** is not free text; declare which way it points:

```python
InString("mask",  "Mask layer",   field=False, default="mask",
         layer_in=Domain.VOXEL)              # SELECTS an existing layer → GUI picker
InString("name",  "Output layer", field=False, default="labels",
         layer_out=(Domain.VOXEL, Domain.LABEL))   # NAMES a layer this node CREATES
```

* `layer_out` is a **tuple** because one name can land in two domains — `analysis.label`
  emits both a Voxel raster *and* a Label table called `labels`.
* `layer_in_mode="from_domain"` instead of `layer_in` when the domain is a **mode value**
  (only `transform.transfer_domain`).
* **Domain consistency is enforced:** `layer_in` domain ⊆ `reads_domains`, and `layer_out`
  domains ⊆ `adds_domains`. The `reads_domains` half is checked in **every mode state the
  socket is active in**, against `spec.resolve_reads_domains(state)` — so a conditional
  requirement is stated with **`reads_domains_by_mode`** (§4f) rather than waived. The
  exemption is an **empty default**: the socket then does not name a required layer *literally*
  — either the layer is optional (`analysis.segment`'s watershed `mask`, `flow.iterate`'s
  `metric`) or the node **infers** it (`analysis.voronoi`, §4g). An inferred layer is still
  required, so declare its domain anyway; the structural check cannot tell the two cases apart
  from the default alone, which is why `voronoi`'s declaration is pinned directly in
  `selftest::test_conditional_reads_domains`.
* Producers no socket can describe use **`NodeSpec.extra_layers(params, modes)`**: literal
  names with no socket (`align.drift`'s `drift_y`/`drift_x`), names derived from another
  param (`extract_boundary` → `f"{labels}_boundary"`), or writes into the layer a *read*
  socket names (`measure`). It runs on every keystroke — **it must never raise**.

**What the declarations buy.** `metadata.propagate_meta` builds `MetaEnvelope.layer_names`
— the `(domain, name)` catalog for each edge — and the inspector turns a `layer_in` socket
into an **editable combo** offering exactly those names (`document.layer_choices`, which
follows the *primary* Dataset edge only, so a `reference`/`raw` second input never leaks in).

### 4d. Filesystem-path sockets — `path_kind` (V2.15)

A STRING socket holding a filesystem path declares
`path_kind="open_file"|"save_file"|"directory"` (plus `path_filter` / `path_hint`) and the
inspector gives it a **Browse…** button. `path_kind` is the **only** thing the GUI keys on, so
a typo or a non-STRING socket is refused at registration — a silently-ignored annotation would
look exactly like a plain text field. Presentation-only, never hashed. Use `path_hint` to say
what an *empty* value means.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/4d-path-sockets.md](references/4d-path-sockets.md).

### 4h. `description` — the hover text (clause 6, V2.13)

<!-- Numbered 4h, out of sequence, because this section and §4d (path sockets) were BOTH
     called 4d until 2026-08-05, and each was cited from a different file meaning a different
     thing. §4d keeps its number because build-node-v2 §0 cites it for path sockets; this one
     moved to the next free letter so both citations resolve. `selftest::test_codemap` now
     fails if any two sections in a skill share a number again. -->


Every param socket carries prose saying **what it does and how it moves the result**. It is
`SocketSpec.description`, and the GUI renders it as the tooltip on both hover surfaces — the
inspector's parameter row and the node card's port dot — through the one shared builder
`nodelab_v2.node_item.socket_hover_text`, which prepends the `name — type · unit` identity
line and hard-wraps the prose. Enforced by `selftest::test_socket_docs`.

```python
InFloat("mask_threshold", "Mask cut", unit="", field=False, default=0.4,
        available_in={"method": frozenset({"cellsam"})},
        description=
        "The per-pixel sigmoid cut on the mask decoder's output — in effect, how far each "
        "cell's mask extends. LOWER grows every mask, HIGHER shrinks it, so this is the "
        "knob that moves reported µm² areas without changing WHICH cells are found. "
        "Default 0.4 is what CellSAM ships; the paper states 0.5.")
```

**It is presentation-only and must stay that way.** `node_recipe_hash` keys on op_key +
params + upstream, never on the `SocketSpec`, so editing a description cannot invalidate a
memo entry or a saved graph — `test_socket_docs` asserts exactly this. That is what makes
documentation a zero-risk edit; do not let anything start reading `description` at runtime.

**What earns its space** (the reader already sees name/type/unit for free):
* the DIRECTION — "lower finds more cells and more false positives";
* whether it moves a **measurement** a downstream table reports (area/volume/intensity);
* when it is **inert** — "only read when Tiled inference is on";
* a load-bearing value: a library default, a paper's value, and any disagreement between
  them.

What does not: restating the label, restating the type, or explaining a unit conversion that
`unit=` already displays.

**The one exemption** is a param whose meaning lives in an external paper or repo — an
augmented-Lagrangian solver's `mu`/`tol`, a published tracker's topology weights, another
project's model thresholds. A confident wrong description is worse than none, so those are
listed in `selftest::_SOCKET_DOC_EXEMPT` **with a reason**, and get documented only against
the algorithm's own docs. Code vendored from *this project's* history is **not** exempt: it
is in the repo, so it is readable.

### 4e. `choice_docs` — one explanation per DROPDOWN OPTION (V2.21)

A `description` describes the control; a dropdown must also
explain **every option**, because the differences between `otsu`, `li` and `yen` are the whole
reason the menu was opened. Both `ModeSpec` and `SocketSpec` take `choice_docs={choice: prose}`
(and a `Mode` carries its own `description`). Write each option **relative to its siblings**:
what it assumes about the data, and which way the result moves if you pick it. Three
vocabularies are canned centrally — `registry.DIM_CHOICE_DOCS`, `domains.domain_docs`,
`reducers.reducer_docs` — do not rewrite them per node. Presentation-only and memo-neutral;
enforced by `selftest::test_option_docs`.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/4e-choice-docs.md](references/4e-choice-docs.md).


### 4f. `reads_domains_by_mode` — the domain rail per BRANCH (V2.22)

`reads_domains` is static, so a node whose
**branches read different structure** cannot state its requirement once — and both failures are
silent, because nothing at pull time reads the rail. Over-claiming paints a red missing-domain
chip on a graph that is fine; under-claiming advertises no requirement at all. Declare the
conditional half as `reads_domains_by_mode={mode: {value: frozenset(...)}}`. It is a **UNION
over every listed mode**, unlike the single-key mapping `granularity` uses. Resolve it with
`spec.resolve_reads_domains(state)` / `missing_domains(incoming, state)`, never by reading the
raw field. Do not reach for it when the domain is simply the image.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/4f-reads-domains-by-mode.md](references/4f-reads-domains-by-mode.md).

### 4g. A layer name is DERIVABLE — infer it (`_resolve_layer`, 2026-08-04)

A layer-name socket's **literal** default is right
for exactly one upstream producer, and wrong for every other graph — which then fails with an
error that prints the right answer one clause after declining to use it. Treat a layer name the
way you treat a spatial param: **derivable from the incoming data ⇒ the node derives it.**
Default to `""`, resolve with `_resolve_layer(...)` (`_shared/labels.py`), and **report the
inference on the progress rail** — inferring silently is how a node quietly analyses the wrong
layer. Never guess between two candidates or across a domain boundary: raise and list them.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/4g-derivable-layer-names.md](references/4g-derivable-layer-names.md).

---

## 5. The 2D/3D lever + variant sockets (V2.03 directive B)

Every 2D/3D-capable node gets a **header lever** — `DimMode()`, a distinguished
`ModeSpec` with `role="dim_lever"`, `presentation="header"`, and a metadata-adaptive
default (`derive="'3D' if (n_z or 1) > 1 else '2D'"` — z>1 ⇒ 3D). It is a **Mode, not a
socket**, so it folds into the recipe hash as an in-body option → 2D and 3D memoize
distinctly, and serialization is unchanged.

- **Variant sockets:** `SocketSpec.available_in={"dim": frozenset({"3D"})}` marks a
  socket present only in that mode state (e.g. an anisotropic axial `sigma_z`,
  `unit="um_axial"`, 3D-only). `NodeSpec.active_inputs(state)` / `active_sockets(state)`
  resolve the live set. The paired-float pattern: a lateral param + a 3D-only `_z`
  companion that defaults to the lateral value.
- **`supports_true_3d` / `three_d_fallback`:** declare honesty — a genuinely
  volumetric backend sets `supports_true_3d=True`; a **stack-of-2D** backend (e.g.
  `denoise_wavelet`, `denoise_bilateral`) sets `supports_true_3d=False`,
  `three_d_fallback="stack_of_2d"`, and keeps its 3D `granularity` at `WHOLE_PLANE` so
  it stays on the per-plane path (H15). Never silently loop 2D over Z while claiming 3D.
- **z==1 guard (GUI):** the lever greys 3D when incoming z==1; a serialized 3D-on-z==1
  is a validation error (H11).
- **Not every node has a lever.** Pointwise/dimension-agnostic ops (gamma, threshold)
  omit it (`kernel_axes=frozenset()`). A **scope** choice (Normalize: plane/volume/series)
  is a plain `Mode`, **not** the dim lever (H24).

### 5b. `available_in` gates ANY mode, not just `dim` (2026-07-28)

**A socket the selected mode never reads must be
HIDDEN, not shown and discarded.** `available_in` keys on *any* mode name, not just `dim`, and
one socket may gate on several modes at once. Gating is **edit-time only**: the engine still
passes every declared param, so a compute's `ctx.params.get(name, default)` fallback must stay
correct, and the mode already folds into the recipe hash, so hiding a socket never alters a
memo key. `available_in` cannot see whether an optional socket is *wired*, so a param whose
liveness depends on a wire stays ungated on purpose — hiding a control that still has an effect
is the same bug in reverse.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/5b-available-in-any-mode.md](references/5b-available-in-any-mode.md).

---

### 5c. A MODE can be gated on another Mode too — `ModeSpec.available_in` (V2.12)

`available_in` is **not socket-only**. A `ModeSpec` takes
the same mapping, so a second Mode that only some methods read is hidden rather than shown and
ignored — the dead control §4b clause 2 forbids for sockets, one level up. A Mode must never
gate on itself. **What gating cannot fix:** a lever whose *value* the compute still resolves —
do not gate the `dim` lever away for a 2D-only method, because the state still resolves `dim`
from its `derive` and hands the compute `WHOLE_VOLUME`. Refuse instead, naming the fix.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/5c-mode-gated-modes.md](references/5c-mode-gated-modes.md).

## 6. Data-access footprint — `Granularity` + `kernel_axes`

Declares what the node must consume *whole* vs per-element; gates tiling/halo/memo in
the scheduler and selects the provider read path (engine granularity routing).

| `Granularity` | Meaning | Provider read |
|---|---|---|
| `TILEABLE` | pointwise / small-stencil 2D | tile / region at one z |
| `WHOLE_PLANE` | a full (Y,X) plane per (m,t,z,c) | region at one z |
| `WHOLE_VOLUME` | a full (Z,Y,X) volume per (m,t,c) | `get_region_volume` |
| `WHOLE_SERIES` | the full T series per (m,c) | across-T reads |
| `MULTI_VIEW` | several M (stitching / fusion) | across-M reads |

- For a lever node, pass a **per-dim map**: `granularity={"2D": TILEABLE, "3D": WHOLE_VOLUME}`,
  `kernel_axes={"2D": frozenset({"y","x"}), "3D": frozenset({"z","y","x"})}` — resolved
  by mode state via `spec.resolve_granularity(state)` / `resolve_kernel_axes(state)`.
- In the compute, `ctx.is_volume` (== resolved `WHOLE_VOLUME`) routes the 2D vs 3D path.
  The shared helper `_map_image(ctx, ds, plane_fn, volume_fn)` (nodes.py) applies a
  per-plane fn in 2D and a per-volume fn in 3D and re-wraps the result as an
  `ArrayProvider`.
- **Declare the footprint honestly** — it must match what the compute actually reads.
  A `WHOLE_VOLUME` node must supply a `volume_fn`; a stack-of-2D node keeps 3D at
  `WHOLE_PLANE`.

---

## 7. Metadata intelligence — `unit` / `derive` on value sockets

Every spatial/temporal number is authored in **physical units**; the framework
resolves it to px/frames from the *incoming edge's* metadata (per-edge, V2.03 §A).

- `unit` ∈ `{"px","um","um_axial","um2","um3","nm","s",""}`. `um_axial` divides by
  `z_step_um` (anisotropic axial sampling); `um`/`nm` divide by `pixel_size_um`.
- `derive` is a pure-arithmetic expression over the **envelope symbol table**
  (`metadata.envelope_symbols`): `pixel_size_um, z_step_um, dt_s, emission_nm, na, mag,
  n_m/n_t/n_z/n_c, z_collapsed, is_3d`. Guard every symbol with `or <fallback>`
  (e.g. `"0.61*(emission_nm or 520)/(na or 1.4)/1000"` → diffraction-limited µm).
- **In the compute**, convert with `to_pixels_v2(value, unit, pixel_size_um=..., z_step_um=...)`
  and read calibration through **`ctx.calib(key)`** (validated against `CALIBRATION_KEYS`;
  a typo is a hard error) — the read is *recorded* and the memo re-validates it on a hit,
  so a metadata change the node actually read invalidates exactly that node.
- The compute reads params with an **inline fallback default**
  (`ctx.params.get("sigma", 0.5)`) — the socket `derive`/`default` seed the GUI widget;
  headless the compute's fallback applies.
- **Per-channel derive — `ctx.channel(c)` (C8/H12).** A **c-iterating** node (its `for c`
  loop processes each channel: Deconvolve's PSF, Spot Detection's radii) must resolve any
  `emission_nm`-derived param **per channel** — channel c's λ, not a collapsed channel-0
  value. Use **`ctx.channel(c).param(name)`**: a user override (param set) wins as one value
  for all channels; else the socket's `derive` is re-evaluated with channel c's optics
  (`envelope_symbols(ctx.env, c)`), memo-fenced automatically; else the socket default.
  `ctx.channel(c).emission_nm()` gives just that channel's λ. Resolve **eagerly** in the loop
  (before returning a lazy provider — a late read trips the frozen-ReadContext guard). This
  also makes `derive` work headless (not only as a GUI seed) for these nodes.
- **Never read `ctx.inputs[i].metadata[calib_key]` directly** — that bypasses the memo
  fence. In debug (`Engine(strict_reads=True)`) it is a hard error (C7).

### 7b. Provenance inheritance — a node stamps context a downstream node reads

When a downstream node's behaviour is determined by
**how an upstream node ran** — not by a value the user should re-type — the upstream node
stamps that config and the downstream node inherits it, rather than exposing a control that
could disagree with the data. Structure dimensionality (`z_kind`) is preserved automatically by
`with_structure`; read it back with `ds.structure_zkind(domain, layer)`, which is why
`transform.rasterize_field` needs no `DimMode` lever. For anything else: stamp with
`ds.with_metadata(...)` under a **namespaced non-calibration** key, and inherit by reading
`ctx.inputs[0].metadata.get(key)` directly. **Derive, don't ask.**

**Full section** — examples, the enforcement details and the cases that motivated it: [references/7b-provenance-inheritance.md](references/7b-provenance-inheritance.md).

### 7c. Calibration describes the CURRENT data, not the file (2026-07-28)

The envelope is a **running description of the
data on this wire**, never a record of the acquisition — so a node that changes what its
numbers MEAN must restamp the affected key in lockstep. `bit_depth` is the intensity-domain
instance: a node that widens the value scale restamps it, one whose output is no longer integer
counts **drops** it, and one that merely redistributes intensities inside the same range does
not restamp at all. Declare **both halves** (the `meta_transform` for edit time, the payload
stamp for pull time), and do not re-derive a relative change in the compute — `ctx.calib`
already returns the post-transform value, so sync to it or a re-pull widens twice.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/7c-calibration-current-data.md](references/7c-calibration-current-data.md).

### 7d. The optional `raw` socket — measure on unenhanced pixels (2026-07-28)

Enhancement is for *finding* objects; the numbers you report
should come from the pixels the camera recorded. A node that MEASURES intensities takes an
optional second Dataset, resolved by `_intensity_provider(ctx, ds)`. Four rules generalise to
any second-Dataset input: **override measurement only, never segmentation**; **declare it after
`data`**, so `dataset_preds[0]` stays the calibration/env source whichever edge was wired
first; **refuse a geometry mismatch** with `_require_same_grid(...)` rather than hand-rolling
one (it checks the sampling provenance as well as the shape, which is what catches a drift
correction that moved the content without changing it); and the memo needs nothing, because a
new predecessor re-keys the node on its own.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/7d-raw-socket.md](references/7d-raw-socket.md).

#### 7e. A second input for a second DOMAIN — `layer_from` (V2.22)

The other reason to take a second Dataset is that the node
**combines two domains that different branches produce** — seed points from one wire, the label
raster to clip them to from another. Same rules as §7d, plus: the layer socket must name the
input it reads (`layer_from="areas"`), or the picker lists names off a wire the compute never
reads; it falls back to the primary when unwired; refuse the input in a mode that ignores it;
never `available_in`-gate a Dataset input (hiding a socket that already has a wire leaves the
edge dangling — refuse instead); copy what it used onto the output under a name of the node's
own choosing via `extra_layers`; and set `view_source=True` only when the wire carries
genuinely different content.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/7e-layer-from.md](references/7e-layer-from.md).

---

## 8. Axis-changing nodes — `meta_transform` (V2.03 §A2)

A node that changes `AxisSizes` or calibration (Z-Project, Crop, Resample, Stack,
Stitch, Channel Split/Merge) MUST declare a **`meta_transform`** — a function in
[`metadata.py`](../../../nodegraph/metadata.py) (`z_project`/`crop`/`resample`/
`stack_time`/`channel_select`/`stitch`) that maps the incoming `MetaEnvelope` to the
outgoing one **touching no pixels**. The edit-time `propagate_meta` pass runs it so the
GUI's widget re-seed, the lever default, and the memo `OutputHeader.axes` are correct
before any pull.

**The payload MUST agree with the meta_transform prediction:**
- Update axes+calibration *in lockstep* in the compute — `reshaped_axes(new_axes)` +
  `with_metadata(...)` (a `None` value removes a key, e.g. z-project drops `z_step_um`).
- **Do not double-count a relative calibration change.** `ctx.calib` reads the node's
  **post-transform** envelope (the meta_transform already ran in `propagate_meta`). So a
  resample must **sync** the payload pixel size to the env's post value, *not* re-divide
  by the scale (that would apply the scale twice). Mirror the meta_transform; let it be
  the single source of truth. Match `span()`/bounds semantics between the compute and the
  meta_transform exactly (one-sided/out-of-range inputs included) or header≠payload.

---

## 9. Structure domains & transfer/bridges

- **Producing structure** (Label/Point/Track): build a `StructureTable`
  ([structure.py](../../../nodegraph/structure.py)) with the invariant
  `id,m,t,c,z,y,x` schema (`z` never NaN; `z_kind` = "plane_index" 2D / "subpixel" 3D)
  and attach via **`Dataset.with_structure(table)`** (columns → per-domain
  `AttributeLayer`s keyed by source `layer`). CCL → `label_components`; watershed →
  `seeded_watershed` + a region-props table; detection → `point_table`.
- **Voxel layers** (masks, distance fields) are lattice attributes:
  `Dataset.with_layer(Domain.VOXEL, name, values6d)` (shape validated against axes).
- **Domain transfer** ([transfer.py](../../../nodegraph/transfer.py)): lattice↔lattice is
  *generated* (reduce over dropped axes / broadcast) and executes via `execute_transfer`;
  structure hops use **explicit bridges** ([bridges.py](../../../nodegraph/bridges.py):
  `voxel_to_label`, `label_to_voxel`, `voxel_to_point`, `point_to_voxel`, `points_in_label`,
  `containing_label`, `gather_by_track`, `broadcast_track`, …). An **indirect route**
  (Voxel→Track = Voxel→Label→Track) is planned by `plan_transfer` and chained per
  frame/volume by **`execute_bridge_plan`** (C2) with the structure inputs (label raster /
  points / track membership).
- **Boundary** ([boundary.py](../../../nodegraph/boundary.py)): the constructive
  Point↔Label spine (`fill_boundary`/`extract_boundary`) — an explicit data-layer op,
  **not** an auto-routed bridge; the input geometry is the dimensionality authority.

---

## 10. The memo invariant (why identity is `revision`, not `id()`)

- **`recipe_hash`** (lookup): op + params (incl. `__modes__` mode/lever state) + upstream
  recipe_hashes + upstream **revisions**. A cheap proxy — never hashes pixels.
- **Declared reads** (`ctx.calib`) are re-validated on a hit — a metadata change the node
  read invalidates it; one it didn't read does not.
- **`output_fingerprint`** (content) drives cutoff/dedup and **includes the image
  provider's `fingerprint()`** (two datasets differing only by image must not collide).
- **Source identity:** a source node folds its provider's `.version` (C5) into the key so
  two providers with different data don't collide on one cached payload.
- Layer content identity is the monotonic **`revision`** — never `id()`, never a content
  hash for lookup.

---

## 11. Verify gate (see build-node-v2 §4 for the full procedure)

```bash
python -m nodegraph.selftest                          # all groups green (core + catalog)
python scripts/_nodelab_v2_phase5_probe.py out.png    # the driven GUI probe
```
On Windows, prefix `PYTHONUTF8=1` (some `[ok]` lines carry `↔`/`σ` glyphs). A v2 node's
gate is `nodegraph.selftest` — add an end-to-end pull through the `Engine` that asserts:
the footprint resolves per dim, 2D vs 3D produce distinct recipe hashes, an axis-changing
node's payload axes/calibration equal its `meta_transform` prediction, and structure output
lands on the right domain/layer.

---

## 12. Compute performance — numba vs numpy/scipy (when to reach for it)

A compute body is fast **by default** when it bottoms out in
one vectorized numpy op or a scipy/skimage/cv2 call — those are already compiled C/MKL and
numba cannot beat them. Reach for numba **only** when the hot cost is a Python loop of many
small array ops, or a sequential data-dependent algorithm that will not vectorize. **`cache=True`
is mandatory in this engine**: per-tile streaming plus Windows `spawn` process pools mean every
worker re-JITs otherwise (~0.3–2 s each), which can go net-negative on a short interactive pull.
Keep the kernel a pure numeric helper — arrays in, arrays out, no `ctx`, no `Dataset`, no scipy
inside `njit`. Do not stack `parallel=True` against a process pool.

**Full section** — examples, the enforcement details and the cases that motivated it: [references/12-numba-vs-numpy.md](references/12-numba-vs-numpy.md).
