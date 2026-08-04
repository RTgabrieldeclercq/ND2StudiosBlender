# NodeLab — Engineering Notes

How the whole system fits together: the data model, node contracts, sockets, domains and
bridges, metadata intelligence, the pull engine, memoization, streaming evaluation, and the
GUI seam. Written for someone about to change the code.

Companion documents:

* [../../MANUAL.md](../../MANUAL.md) — the user manual (what the software does).
* **`wire-node-v2`** skill — the authoritative *node-concepts* reference (socket contract,
  units/derive, provenance patterns). Read it before adding or editing a node.
* **`build-node-v2`** skill — the *procedure* (grill → write → footprint/metadata gate →
  verify gate).
* [../ClaudesPlan/](../ClaudesPlan/) — the dated design record: `V2.00` (11 locked
  decisions) + addenda `V2.03` (per-edge metadata + the 2D/3D lever), `V2.04` (streaming
  eval), `V2.06` (DVC), `V2.07` (general-node directive), `V2.08` (mesh domain), `V2.09`
  (overlays), `V2.10` (param↔socket contract), `V2.11` (layer picker).
* `nodegraph/kernels/<name>.md` — the integration contract for each vendored kernel.

> `ARCHITECTURE.md` in this folder describes the **removed** first-generation app and is
> retained only as history.

**State:** catalog 55 node types; `python -m nodegraph.selftest` → 55 groups green;
`scripts/_nodelab_v2_phase5_probe.py` → ALL PASS (both verified 2026-07-29).

---

## Contents

1. [Layering](#1-layering)
2. [The one-sentence model](#2-the-one-sentence-model)
3. [Data model — Dataset, axes, layers, revision](#3-data-model--dataset-axes-layers-revision)
4. [Domains](#4-domains)
5. [Sockets & wiring](#5-sockets--wiring)
6. [A node — NodeSpec + compute](#6-a-node--nodespec--compute)
7. [The socket contract](#7-the-socket-contract)
8. [Metadata intelligence](#8-metadata-intelligence)
9. [The pull engine](#9-the-pull-engine)
10. [Memoization](#10-memoization)
11. [Streaming evaluation](#11-streaming-evaluation)
12. [Providers & storage layout](#12-providers--storage-layout)
13. [Structure, transfer, bridges](#13-structure-transfer-bridges)
14. [The mesh domain](#14-the-mesh-domain)
15. [Fields](#15-fields)
16. [Zones & groups](#16-zones--groups)
17. [Serialization](#17-serialization)
18. [The GUI](#18-the-gui)
19. [Invariants & landmines](#19-invariants--landmines)
20. [Gates, and how to add a node](#20-gates-and-how-to-add-a-node)
21. [Known gaps](#21-known-gaps)

---

## 1. Layering

```
run.py ─→ nodelab_v2.app.run
┌───────────────────────── nodelab_v2/  (PySide6 — ALL Qt lives here) ─────────────┐
│ window.py      chrome, menus, splitter, maximize/mini-map orchestration          │
│ document.py    GraphDocument — Qt-FREE editing model + wiring rules + save/load  │
│ scene/node_item/edge_item/frame_item/minimap/welcome   the canvas (view only)    │
│ inspector.py   NodeSpec → editable form (units, auto/pinned, mode gating)        │
│ viewer.py + glview.py + overlays.py + overlay_dialog.py   image + overlays       │
│ spreadsheet.py + export.py    structure tables + CSV/Parquet/Arrow               │
│ runner.py      QThreadPool + epoch registry + persistent Memo + plane render     │
│ ops.py         Qt-FREE io.load / view.viewer + channel-tap materialization       │
│ ingest.py + nd2_meta.py   ND2/TIFF → provider + MetaEnvelope (the nd2 seam)      │
│ theme.py console.py palette.py app.py                                            │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                 │ imports nodegraph only
┌───────────────────────── nodegraph/  (Qt-free engine, 25 modules) ───────────────┐
│ domains · dataset · revision · sockets · registry · graph                        │
│ metadata (edit-time envelope pass)   memo (two-hash + GC)   engine (lazy pull)   │
│ provider (tiled sources)   streaming (per-tile lazy providers + TileCache)       │
│ structure · bridges · transfer · boundary · mesh · tracking · reducers · field   │
│ zones · groups · serialize · nodes (the 55-node catalog) · selftest              │
│ kernels/  vendored pure-compute analysis kernels + their .md contracts           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Hard rules that keep this honest:

* **`nodegraph/` never imports Qt, and never imports `nd2`.** ND2 coupling lives in
  `nodelab_v2/ingest.py` + `nd2_meta.py`, and every one of their `nd2` imports goes through
  `nodelab_v2/nd2_compat.py` (§19, *Ingest / the `nd2` seam*).
* **`nodelab_v2/document.py` and `ops.py` are Qt-free**, so the editing model and the
  GUI-introduced ops are testable headless and a saved graph runs without PySide6.
* **Heavy backends are lazily imported inside a compute** (scipy, skimage, sklearn, numba,
  tensorflow, al-dic, pyarrow, blosc2). Importing `nodegraph.nodes` must stay cheap — the
  whole palette registers with none of the optional deps installed.

---

## 2. The one-sentence model

> A node declares **typed sockets + unit-tagged params + its data-access footprint**; the
> engine does the routing, the px↔µm math, the memoization, the domain transfers, and the
> tiling.

A node never hardcodes a pixel constant, never invents graph state, never reaches outside its
inputs, and never reads calibration except through the recording context. Everything else in
this document is machinery that only works because of those four abstentions.

The Blender mapping, which is not decoration — it is why the pieces compose:

| Blender geometry nodes | here |
|---|---|
| Geometry flowing through the tree | one lazy **`Dataset`** (image provider + attribute layers) |
| Attribute domains (point/edge/face) | **11 domains**: the acquisition lattice + detected structures |
| Named attribute over geometry | **`AttributeLayer`** keyed `(domain, layer, name)` |
| Typed sockets + implicit conversion | `SocketType` + `can_connect` |
| Field inputs adapting to the mesh | **value sockets** with `unit`/`derive` + `is_field` |
| A modifier's evaluation dimension | the **2D/3D lever** (`DimMode`) |
| Viewer node / lazy eval | **lazy pull** — nothing computes until `Engine.pull` |
| Node groups, simulation zones | `groups.expand`, `zones.unroll` |

---

## 3. Data model — Dataset, axes, layers, revision

[`nodegraph/dataset.py`](../../nodegraph/dataset.py)

```python
Dataset(axes: AxisSizes,
        metadata: Dict[str, Any],           # calibration + provenance
        image: Optional[TileProvider],      # the lazy voxel source
        attributes: Dict[LayerKey, AttributeLayer])
```

* **`AxisSizes(m, t, z, c, y, x)`** — canonical axis order `AXIS_ORDER = (m,t,z,c,y,x)`.
  `c` is a **first-class store/tile/memo axis**, not a channel loop bolted on.
  `is_volumetric` (`z > 1`) is the metadata-adaptive default for the 2D/3D lever.
* **`LayerKey = (Domain, layer|None, name)`**. `with_layer` leaves `layer=None`
  (`(VOXEL, None, "mask")`); `with_structure` files **each column** under the source layer
  (`(POINT, "spots", "y")`). *Never key user-facing logic on `LayerKey` slot 1* — the
  user-facing name lives in a different slot per family. That is exactly why
  `MetaEnvelope.layer_names` exists.
* **`AttributeLayer`** is frozen with `eq=False` (identity semantics — an ndarray field makes
  generated `__eq__`/`__hash__` raise), its array is made **read-only after owning its
  buffer** (copy if it is a view or a caller-owned writeable array), and it carries a
  **`revision`**: a fresh monotonic integer per constructed layer.

  **`revision` is content identity.** Never `id()` (recycled), never a content hash (too
  expensive for a lookup key). Replacing a layer mints a new revision, so any memo key
  embedding the old one auto-invalidates.
* **Derivation is structural sharing.** `with_attribute` / `with_layer` / `with_metadata` /
  `with_image` / `with_structure` / `reshaped_axes` all copy-on-write and share the upstream
  store, so a node returning a "new" Dataset is cheap.
* **`with_metadata` is the only sanctioned calibration write path.** A `None` value *removes*
  a key (z-project dropping `z_step_um`).
* **`reshaped_axes(new_axes, drop_stale=True)`** resolves layers orphaned by an axis change:
  a **lattice** layer whose shape no longer matches its domain's shape is dropped; structure
  layers pass through. This is the single source of the "layer catalog is not monotone" rule.
* **`CALIBRATION_KEYS`** is a validated key *schema*, not a typed model:
  `pixel_size_um, z_step_um, dt_s, channel_emission_nm, objective_na,
  objective_magnification, bit_depth`. A typo in `ctx.calib("pixle_size_um")` is a hard
  error, not a silent `None`.

---

## 4. Domains

[`nodegraph/domains.py`](../../nodegraph/domains.py)

**(a) The acquisition lattice** — coarsenings of the image hypercube, each domain being the
finer one with an axis group aggregated away:

```
Voxel {m,t,z,c,y,x} → Plane {m,t,z} → Frame {m,t} ─┬─ Multipoint {m} ─┐
                                                   └─ Timepoint {t} ──┴─ Global {}
                                       Channel {c}  (orthogonal)
```

Because this is a **lattice**, transfer between any two lattice domains is *generated* —
coarsen = reduce over the dropped axes, refine = broadcast — so no per-pair rule is written.
`meet` (∩) is total; **`join` (∪) is partial**, because Channel is orthogonal
(`join(Channel, Frame) = {m,t,c}` is unnamed). The transfer generator never needs `join`, so
the partiality is harmless there.

Consequence worth knowing: coarsening Voxel to a spatial/temporal domain **reduces over `c`**
(channel-mean). For per-channel results, transfer to Channel or select a channel first.

**(b) Detected structures** — `Label`, `Point`, `Track`, `Mesh`. Defined by analysis rather
than by coarsening, so their transfers are **explicit bridges**. They are *multi-instance*:
one Dataset may carry several, keyed by source layer.

`_LATTICE_AXES` is the single declaration: a domain's **absence** from that map is what makes
it a structure domain, and that is why `is_lattice` / `axes_of` / `shape_for` / `axis_list`
all give the right answer or a clean raise for free.

Two presentation maps live here deliberately (`DOMAIN_COLOR`, `DOMAIN_ABBR`) because domain
identity belongs to the data model, not to a theme — the GUI wraps them as QColors. **A new
`Domain` member must land its colour in the same commit**: the GUI builds its map by
iterating `Domain`, and a missing entry used to `KeyError` at import (now hardened, still bad
practice).

---

## 5. Sockets & wiring

[`nodegraph/sockets.py`](../../nodegraph/sockets.py)

One data type (`DATASET`, the thick main wire) plus field-able value types
(`FLOAT/INT/BOOL/VECTOR/COLOR/STRING/MENU`).

`can_connect(src, dst)`:

1. `src` must be an output, `dst` an input.
2. `DATASET` pairs **only** with `DATASET`.
3. `VECTOR→VECTOR` connects on equal `dims` or **widens** (`src.dims <= dst.dims`; the axial
   component pads with 0). Narrowing is rejected — it needs an explicit swizzle node.
4. Otherwise: exact type or a registered implicit conversion (`bool→int→float`,
   `float/int→vector` broadcast, `float/vector→color`). **Data-layer transforms are never
   implicit conversions** — those stay explicit nodes.

Field-ness never blocks a value connection: an input value socket accepts a value *or* a
field.

Cardinality is the graph layer's business: `multi=True` accepts many wires in canonical
order; a second wire into a non-multi input **replaces** it in the GUI, and is a **hard
error** in the engine (never a silent last-wins).

---

## 6. A node — NodeSpec + compute

[`nodegraph/registry.py`](../../nodegraph/registry.py) + [`nodegraph/nodes.py`](../../nodegraph/nodes.py)

A node **type** is a `NodeSpec` in the global `NODES` registry plus a
`compute(ctx) -> Dataset` in `COMPUTES`, both created by one call:

```python
register_node(compute_fn, op_key="enhance.median", label="Median",
              category="enhancement",
              inputs=[InDataset(), InFloat("radius", unit="um", derive=..., kernel_param=True)],
              outputs=[OutDataset()],
              modes=[DimMode()],
              granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
              kernel_axes={"2D": frozenset("yx"), "3D": frozenset("zyx")},
              meta_transform=None,
              reads_domains=frozenset({Domain.VOXEL}),
              adds_domains=frozenset({Domain.LABEL, Domain.VOXEL}),
              extra_layers=lambda params, modes: ((Domain.VOXEL, "drift_y"),))
```

`register_node` = `define_node(**spec)` (builds + registers the `NodeSpec`) plus
`COMPUTES[op_key] = compute_fn`. Importing `nodegraph.nodes` fires every module-level
registration.

### The declarations, and what each one buys

| Declaration | Consumed by | Effect |
|---|---|---|
| `op_key` | serialization, `COMPUTES`, memo | the frozen identity. **Never rename.** Test fixtures MUST use fake keys (`test.*`) |
| `inputs`/`outputs` (`SocketSpec`) | GUI form, wiring, memo | typed sockets + units/derive/defaults/`available_in`/layer direction |
| `modes` (`ModeSpec`) | GUI dropdowns, memo | in-body enums; fold into the recipe hash as `params["__modes__"]` |
| `DimMode()` | GUI header switch | the 2D/3D lever: `role="dim_lever"`, `presentation="header"`, `derive="'3D' if (n_z or 1) > 1 else '2D'"` |
| `granularity` / `kernel_axes` | engine read routing, streaming unit | the data-access footprint, per dim |
| `supports_true_3d` / `three_d_fallback` | GUI honesty | a stack-of-2D backend says so and keeps 3D at `WHOLE_PLANE` |
| `meta_transform` | edit-time envelope pass | axes/calibration prediction, pixel-free |
| `reads_domains` / `adds_domains` | domain rail, wire tint, validation | required vs produced domains |
| `layer_in` / `layer_out` / `extra_layers` | layer catalog + GUI picker | which named layers a node reads / writes |

### The footprint

| `Granularity` | Meaning | Provider read path |
|---|---|---|
| `TILEABLE` | pointwise / small stencil, 2D | tile or region at one z |
| `WHOLE_PLANE` | a full `(Y,X)` plane per `(m,t,z,c)` | region at one z |
| `WHOLE_VOLUME` | a full `(Z,Y,X)` volume per `(m,t,c)` | `get_region_volume` / `get_subvolume` |
| `WHOLE_SERIES` | the whole T series per `(m,c)` | across-T reads |
| `MULTI_VIEW` | several M (stitching / fusion) | across-M reads |

**Declare it honestly.** It is not a hint — it selects the provider read path and the
streaming unit. A plane-global statistic or solver declared `TILEABLE` sees the wrong
population per tile; six nodes were caught mis-declared during C1 and re-declared
(gamma/tv/wavelet/clahe/bilateral/nlm). A `WHOLE_VOLUME` node must supply a `volume_fn`; a
stack-of-2D node keeps 3D at `WHOLE_PLANE`.

In the compute, `ctx.is_volume` (== resolved `WHOLE_VOLUME`) routes 2D vs 3D, and the shared
helper `_map_image(ctx, ds, plane_fn, volume_fn, halo=…)` applies the right one lazily.

### The general-node directive (V2.07)

**Nodes are general primitives, not workflow-specific.** "Granule-ness" or "DIC-ness" lives
in default params and docs, not in the node's identity. Hence `boundary_band` (any labels),
`cluster_points`, `roi_mask`, `detect.particles` — while genuine algorithm names are kept
(`dvc_field`, `dic_correlate`, `stardist_nuclei`).

---

## 7. The socket contract

Five clauses, each **enforced structurally** by `selftest::test_param_socket_contract`
(V2.10/V2.11). A 2026-07-28 sweep found 18 of 55 nodes with a defect of this class.

**The root hazard: the engine does not filter params against the socket list.** It hands the
compute `{**node.params, "__modes__": …}` — params are *overrides*, never default-filled. So
a compute can read a param no socket declares: it works headlessly (a test just passes it)
and is **unreachable in the GUI**, which builds widgets from `NodeSpec.inputs`. No functional
test can see it, because the node returns the right answer for the only value it can ever
have. That is why the check is structural.

1. **Every param the compute reads has a socket.** Exemptions are documented in
   `_PARAM_NO_SOCKET_OK` (one back-compat alias) plus genuinely *machine-set* params — prove
   the writer exists, don't assume.
2. **Every socket the node declares is read.** A control that does nothing is
   charter-forbidden. Remedies in order: **(a) gate it** with `available_in`; **(b) refuse** —
   raise if it is explicitly set on a path that ignores it; **(c) delete**. Prefer (a). Use
   (b) when the condition is not a rectangle of mode values.
3. **`available_in` must name real modes and real values** — a typo hides the socket in every
   state.
4. **A layer-name socket declares its direction and domain** (`layer_in` / `layer_out`), and
   a **filesystem-path socket declares `path_kind`** so the GUI can offer a file/folder
   dialog instead of a hand-typed path (V2.15).
5. **A default is declared exactly once — in the `SocketSpec`.** Read layer params through
   **`ctx.layer("mask")`**, which resolves override → socket default via the one shared
   `registry.layer_value` — the same function `propagate_meta` calls. Never re-spell a default
   inline (`ctx.params.get("mask", "mask")`): that made the default a second copy, and the
   edit-time prediction a third, with nothing keeping them in step.

### `available_in` gates any mode

The key is any mode name (`{"method": frozenset({"fixed"})}`), and a socket may gate on
several modes at once (`histogram_threshold` gates each threshold on method × direction, so
1–2 live fields show instead of eight). Two things it does **not** change: the engine still
passes every declared param to the compute (gating is edit-time only, so the compute's
fallback must stay correct), and the mode already folds into the recipe hash, so hiding a
socket never alters a memo key.

`available_in` can only see **mode state**, not whether an optional socket is wired — so a
param whose liveness depends on a wire (DVC/DIC `reference_frame`) stays ungated on purpose.
Hiding a control that still has an effect is the same bug in reverse.

### Layer sockets and the catalog (V2.11)

```python
InString("mask", "Mask layer", field=False, default="mask", layer_in=Domain.VOXEL)
InString("name", "Output layer", field=False, default="labels",
         layer_out=(Domain.VOXEL, Domain.LABEL))     # one name, two domains
```

* `layer_out` is a **tuple** because one name can land in two domains (`analysis.label` emits
  both a Voxel raster and a Label table called `labels`).
* `layer_in_mode="from_domain"` when the domain is itself a mode value (only
  `transform.transfer_domain`).
* Domain consistency is enforced: `layer_in` ⊆ `reads_domains`, `layer_out` ⊆ `adds_domains`.
  The `reads_domains` half is **exempt for a mode-gated socket** — a conditional requirement,
  and `reads_domains` has no per-mode form.
* Producers no socket can describe use **`NodeSpec.extra_layers(params, modes)`** — literal
  names (`drift_y`/`drift_x`), names derived from another param
  (`f"{labels}_boundary"`), or writes into the layer a *read* socket names (`measure`). It
  runs on **every keystroke**: it must never raise.

`propagate_meta` turns these into `MetaEnvelope.layer_names` — the `(domain, name)` catalog
per edge — and the inspector turns a `layer_in` socket into an editable combo over exactly
those names (`document.layer_choices`, which follows the **primary** Dataset edge only, so a
`reference`/`raw` second input never leaks in).

### Path sockets (V2.15)

A STRING socket holding a **filesystem path** declares
`path_kind="open_file" | "save_file" | "directory"` (+ `path_filter`, `path_hint`), and the
inspector puts a **Browse…** button on the row. Same shape of bug as an undeclared layer
socket: the button used to key on the literal socket *name* `"path"`, so `analysis.segment`'s
`sd_model_path` / `model_path` were reachable only by typing an absolute path from memory.
`directory` gets `getExistingDirectory` — a local StarDist model is a folder, and no file
dialog can return one. Enforced from both ends: `NodeRegistry.register` refuses an unknown
kind or a non-STRING socket (a silently-ignored declaration would just look like a text
field), and the socket-contract gate flags any name-shaped path socket (`path`/`dir`/`file`/
`folder`) that declares none. All three fields are presentation-only and **never hashed**,
like `description`.

---

## 8. Metadata intelligence

Three cooperating mechanisms: an **edit-time forward pass** that predicts metadata, a
**recorded read fence** that makes eval-time reads memo-correct, and a set of **provenance
conventions** for structural (non-physical) config.

### 8.1 The edit-time envelope pass

[`nodegraph/metadata.py`](../../nodegraph/metadata.py)

Evaluation is lazy pull, so at edit time there is no materialized Dataset to read. So a cheap
forward topological walk propagates **metadata only** — `AxisSizes` + the calibration dict +
the domain set + the layer catalog — through each node's declared `meta_transform`, **touching
no pixels**. It runs on load and after every edit, and it feeds:

* the GUI's widget re-seed (the `ƒmd` pills — edit an upstream node and every downstream auto
  value updates before anything computes),
* the 2D/3D lever's default and its z==1 greying,
* the domain rail + wire tint + red missing-domain validation,
* the layer picker's suggestions,
* `OutputHeader.axes` on the memo entry.

It is **advisory**: the memo key is driven by the *eval-time* recording context. Statically
unknowable sizes (a stitched Y,X extent) are marked `unknown_axes` rather than guessed.

`MetaEnvelope` is `(axes, metadata, layers, unknown_axes, domains, layer_names)`. A node's
input envelope is its **first Dataset predecessor's** output — which is why declared socket
order matters (§8.5). Roots use `seeds[node_id]`.

Named meta-transforms: `identity`, `resample`, `z_project`, `stack_time`, `frame_slice`,
`channel_select`, `crop`, `stitch`, `value_rescaled`.

### 8.2 Units and derive

* `unit ∈ {"px","um","um_axial","um2","um3","nm","s",""}`. `um`/`nm` divide by
  `pixel_size_um`; **`um_axial` divides by `z_step_um`** (anisotropic axial sampling).
  Convert with `to_pixels_v2(value, unit, pixel_size_um=…, z_step_um=…, dt_s=…)`; a missing
  calibration degrades to 1:1 and the caller decides whether that is acceptable.
* `derive` is a pure-arithmetic expression over the **envelope symbol table**
  (`envelope_symbols`): `pixel_size_um, z_step_um, dt_s, bit_depth, emission_nm, na, mag,
  n_m/n_t/n_z/n_c, z_collapsed, is_3d`, evaluated with empty builtins. **Guard every symbol
  with `or <fallback>`** — e.g. `"0.61*(emission_nm or 520)/(na or 1.4)/1000"`.
* The v2 convention: a param is **auto/derived iff it is absent from `ctx.params`** (the GUI
  stores a value only when the user edits or pins it). Headless, the compute's inline
  fallback applies.

### 8.3 The read fence — `ctx.calib`

A compute reads calibration **only** through `ctx.calib(key)` (validated against
`CALIBRATION_KEYS`). Each read is recorded as `(key, value_digest)` onto the memo entry and
**re-validated on a hit**: a metadata change the node actually read invalidates exactly that
node; one it did not read does not.

Two guards make this enforceable:

* `Engine(strict_reads=True)` wraps input payload metadata in `_StrictCalibMetadata`, so an
  un-contexted `ds.metadata["pixel_size_um"]` is a hard error (C7). It subclasses `dict`, so
  `{**d}` / `items()` / `dataclasses.replace` take the C fast path and are unaffected — only
  an explicit calibration `__getitem__`/`get` trips. The engine also **launders** the wrapper
  out of any returned payload so it never enters the memo.
* `ReadContext.freeze()` fires the moment a compute returns. A `ctx.calib` call from inside a
  lazy closure at tile-pull time would silently escape the fence, so it raises: **resolve
  every calibration value before constructing a streaming provider.**

`_RecordingMetadata` additionally records reads made through `ctx.env.metadata` directly, so
even that route stays fenced.

**What gets recorded is the ENVELOPE's value, never the caller's default** (fixed
2026-07-30). Re-validation compares each stored digest against `metadata.get(key)` with no
default, so recording a *defaulted* value for an **absent** key stored a digest that could
never match again — the entry re-validated false on every single pull and the node
recomputed forever, silently. It bit every node that resolves a param through
`envelope_symbols`, which reads `md.get("z_collapsed", False)`: i.e. most of the
metadata-intelligent catalog, since real sources rarely carry that key. Measured after the
fix: an unchanged re-pull of *Gaussian 3D → Segmentation → Measure* went from recomputing
the two eager analysis nodes every time to a full `cached` chain. The fence is unweakened —
the digest still moves whenever the envelope's value moves, absent → present included, and
the caller's default is a code constant rather than data.

### 8.4 Per-channel derive — `ctx.channel(c)` (C8)

A **c-iterating** node (its `for c` loop processes each channel: Deconvolve's PSF, Spot
Detection's radii) must resolve `emission_nm`-derived params **per channel**.
`ctx.channel(c).param(name)` resolves override → derive re-evaluated with channel `c`'s
optics (`envelope_symbols(env, c)`, auto memo-fenced) → socket default.
`ctx.channel(c).emission_nm()` gives just that λ. Resolve **eagerly in the loop**, before
returning a lazy provider. This also makes `derive` work headless for these nodes, not just as
a GUI seed.

### 8.5 Provenance inheritance — stamp and inherit (§7b)

When a downstream node's behaviour is determined by **how an upstream node ran** — not by a
value the user should retype — the upstream node **stamps** it and the downstream node
**inherits** it, rather than exposing a redundant, conflictable control.

* **The built-in generic case: `z_kind`.** A structure table's `z_kind`
  (`plane_index` = 2D per-plane, `subpixel` = 3D) would be lost when `with_structure`
  explodes the table into per-column layers, so `with_structure` **automatically preserves it**
  into a namespaced `__struct_zkind__` map. Every structure producer self-describes its
  dimensionality for free; a consumer reads `ds.structure_zkind(domain, layer)`.

  This is why `transform.rasterize_field` has **no DimMode lever** — it routes off the
  field's own `z_kind`, so a 2D per-plane field on a `z>1` image is never misread as a 3D
  grid (a real default-silent-corruption bug this fixed). `accumulate_field` inherits the
  same way.
* **The general pattern:** stamp with `ds.with_metadata(key=value)` using a **namespaced
  non-calibration key** (`dvc_reference_mode`, `dvc_strain_type`, `dic_reference_mode`);
  inherit by reading `ctx.inputs[0].metadata.get(key)` **directly**. That direct read is
  allowed *because the key is not a calibration key* (`_StrictCalibMetadata` passes
  non-calibration reads through) and is memo-safe without recording: the marker rides the
  upstream payload, so an upstream change bumps the upstream **revision**, which already folds
  into the consumer's recipe hash.
* **Derive, don't ask** — a consumer that inherits dim/reference needs no lever and no
  reference param; fix a plain superset `granularity`/`kernel_axes` and loop the units
  internally.
* **Refuse the wrong input.** `accumulate_field` raises if the marker is absent (not a DVC
  field) or says `fixed_frame` (already cumulative), and re-stamps
  `dvc_reference_mode="cumulative"` so a second accumulate is refused too. DIC stamps
  `dic_*` deliberately *distinct* from `dvc_*` so a DIC field is never fed to the ALDVC-only
  accumulator.

### 8.6 Calibration describes the CURRENT data, not the file (§7c)

Every node reads its **input** envelope, so calibration is a running description of the data
on this wire — never a record of the acquisition. A node that changes what its numbers *mean*
must restamp the affected key in lockstep. `bit_depth` is the intensity-domain instance:

* **Widening the scale restamps it.** Summing `n` samples of a `b`-bit signal needs
  `b + ceil(log2 n)` bits → `metadata.bit_depth_after_sum(env, n)`, used by `z_project` /
  `stack_time` for their `sum` combiner only (mean/median/max/min stay in range). A 12-bit
  series summed over T=8 becomes 15-bit **for every node after it**, and chains.
* **Output that is no longer integer counts DROPS it** — `metadata.value_rescaled` is the
  ready-made transform (percentile Normalize → `[0,1]` floats). Absent `bit_depth` is the
  honest signal "no declared integer scale", and consumers handle it:
  `analysis.threshold`'s fixed level falls back to 0.5; `histogram_threshold` refuses `[0,1]`
  input outright.
* **Redistribution inside the same range does NOT restamp** — CLAHE rescales back to the
  input's `[min,max]`; γ preserves the plane max. Check the backend's actual output range
  before deciding.

### 8.7 The optional `raw` socket (§7d)

Enhancement is for *finding* objects; reported numbers should come from recorded pixels. A
measuring node takes an optional second Dataset input (`_InRaw()`), resolved by
`_intensity_provider(ctx, ds) -> (provider, is_raw)`. On `analysis.measure` (all stats) and
`analysis.histogram_threshold` (region intensity columns). Four rules generalize to any
second-Dataset input:

1. **Override MEASUREMENT only, never segmentation.** The mask, label raster and thresholds
   stay on the main input, so geometry and calibration remain one consistent story.
2. **Declare it AFTER `data`.** `graph.dataset_preds` sorts by declared socket position, so
   `data` stays `dpreds[0]` — the calibration/domain env source no matter which edge the user
   wired first. Never let an auxiliary input become the env. *(A latent bug of exactly this
   shape was fixed in `graph.py` when DVC introduced its second Dataset socket.)*
3. **Refuse a geometry mismatch.** They are read voxel-for-voxel, so compare `provider.axes`
   and raise, naming both shapes. `AxisSizes` is frozen — `==` works, but it is **not
   iterable**; format it field by field.
4. **The memo needs nothing** — a new predecessor folds into `recipe_hash` automatically.

---

## 9. The pull engine

[`nodegraph/engine.py`](../../nodegraph/engine.py)

`Engine.pull(node_id)` → `entry(node_id).payload`. The engine walks **backward**, computing
only what that request needs, memoizing every node output.

`_entry(node_id, stack)`, in order:

1. **Cycle check** against the walk stack (cycles are rejected outside zones).
2. **Recurse upstream in canonical socket order** — `preds` sorted by declared socket
   position, then edge order. Without this, a multi-input node's args land in the wrong slots
   and the memo key depends on edge-insertion order.
3. **Build `params = {**node.params, "__modes__": dict(state)}`** so mode/lever state re-keys
   the memo.
4. **Source identity.** A node with no predecessors folds its provider's `version` (or a seed
   image's `version`) plus `__source__ = node_id` into the key, so two distinct sources with
   the same op+params cannot collide on one cached payload, and a swapped seed cannot serve a
   stale chain.
5. **`recipe_hash = node_recipe_hash(op_key, params, up_hashes, up_revisions)`**; look it up;
   a hit is returned only if `_reads_valid` (every recorded calibration digest still matches
   the current envelope).
6. **Resolve the footprint** from mode state, build a `ReadContext` + a recording env view,
   assemble `inputs` (positional) and `by_name` (socket → payload; a `multi` socket collects a
   tuple, a second edge into a non-multi socket raises).
7. **Run the compute** inside `start`/`done`/`error` observer events, then `rc.freeze()`.
8. **Store** with an `OutputHeader(axes, metadata_digest, layers)` from the envelope.

`EvalContext` is the whole surface a compute sees: `params`, `env`, `granularity`,
`kernel_axes`, `inputs`, `by_name` via `ctx.input(name)`, `provider`, `tiles`, `fields`,
`spec`, plus `calib` / `meta` / `layer` / `channel` / `is_volume` / `progress`.

**`ctx.progress(done, total, note)`** is a no-op when nothing observes. Only eager per-unit
computes should call it — a compute returning a lazy provider must **not** fake a fraction;
staying silent is what tells the UI its cost is deferred. Observer exceptions are swallowed:
a run must never depend on who is watching.

**Two-level progress (V2.17).** The UI draws *two* stacked bars per node — frames finished
out of total frames (orange, on top) and the work inside the frame in flight (accent,
below) — so `progress` reports both levels. There are three ways in, and a node should use
the cheapest one that is honest:

| how | when | what happens |
|---|---|---|
| `ctx.progress(i+1, n, note, frames=ax.t)` | any eager per-unit loop | the flat count is *split* by arithmetic: `frames_done = floor(done·frames/total)`, sub = the remainder within one frame |
| `…, sub=k, sub_total=K` | the work inside one unit reports its own percentage | the derived split is overridden, so the sub bar moves *through* a single unit |
| `ctx.progress_frame(frame, frames, sub, sub_total, note)` | the compute's own structure is a frame loop with inner steps | the flat `done`/`total`/`fraction` are synthesized from the two |
| `…, sub_unknown=True` | the frame is known but the work inside it is **one opaque call** | `sub_fraction` is an explicit `None` → the UI *sweeps* that bar |

That last row is the one to reach for on a learned segmenter. `sub_unknown` is not a way to avoid
counting something countable — it is the honest report for a single CNN inference with no upstream
hook, and it exists because the alternatives are both lies: a determinate bar frozen for thirty
seconds reads as a hang, and omitting the field reads as 0% ("nothing has started"). An explicit
`None` is distinct from a missing key for exactly this reason, so a sink must test
`"sub_fraction" in info and info["sub_fraction"] is None`, not truthiness.

The helpers do this for you: `_each_plane_p` / `_each_volume_p` / `_parallel_progress` in
`nodes.py` pass `frames=ax.t`, which is why every eager node gets both bars without its loop
being touched. `_UnitBar` covers the third case — it measures the sub axis in *centi-units*
(`sub_total = units_per_frame × 100`) and is what wires the ALDVC/DIC kernels' own
`progress_cb` through, so a multi-second solve no longer freezes the bar.

**Where the ticks come from matters as much as the arithmetic.** `fold_units` folds each result
**as it lands** rather than materialising the batch (`_imap`, V2.17). Output and order are
identical either way — but the eager catalog nodes tick from the fold, so a materialised batch
collapsed every tick in a batch onto the instant its *slowest* unit finished. On a per-plane
learned segmenter with N lanes that is a bar standing still for N planes and then jumping N.
The two throttle exemptions in `nodelab_v2.runner` exist for the same class of reason: a **frame
step** and a **flip between determinate and sweeping** are each the only event that carries a
visible change, and the next one may be a whole unit of work away, so neither may be dropped.

Invariants a sink may rely on (gated in `test_engine_observer`): `frames_done` is
**monotone**, `sub_fraction` stays in `[0,1]` and restarts at each frame boundary, the final
tick lands **both** at 1.0, and the frame fields are **absent entirely** when no frame count
was reported — a node with no frame axis must not be given a fake one. `fraction` alone is
always valid, so pre-V2.17 sinks keep working unchanged.

`Engine(...)` knobs: `memo` (or `memo_bytes`), `seeds`, `providers`, `meta_seeds`,
`strict_reads`, `cache_bytes` (tile cache, default 1 GiB), `observer`.
`reseed_meta(meta_seeds)` re-runs the envelope pass while keeping the memo, so a subsequent
pull re-validates declared reads.

`entry()` wraps the recursion in `recursion_headroom(8 * len(graph.nodes))` — the C1 stopgap
for deep unrolled-zone chains (an iterative rewrite is a follow-up).

---

## 10. Memoization

[`nodegraph/memo.py`](../../nodegraph/memo.py)

Two hashes with different jobs:

* **`recipe_hash`** — the **lookup** key, computed *before* compute from structure only:
  op + params (incl. `__modes__`) + upstream recipe_hashes + upstream **revisions**. A cheap
  proxy: it never reads pixels (you cannot hash an 85 MB plane to look it up).
* **`output_fingerprint`** — a **content** hash computed *after* compute (per-tile for images,
  full for KB–MB structure tables), used for **cutoff** (a recompute that yields identical
  bytes does not dirty downstream) and **dedup** (identical outputs share one blob). It
  **includes the image provider's `fingerprint()`**, so two Datasets differing only by image
  cannot collide.

Plus the **declared-reads re-validation** on every hit (§8.3). Hashing is stdlib
`blake2b` — the proxy hash is over small metadata, so no third-party hash dependency enters
the core.

**`_canon` refuses object-dtype ndarrays.** It hashes arrays via
`ascontiguousarray(...).tobytes()`, which for object dtype hashes **pointer bytes** — so
identical content would hash differently (permanent miss) and distinct content could collide
through a reused CPython pointer slot (a *wrong payload served*). This is what makes the mesh
domain's flat/fixed-width contract enforceable.

### The byte-budget LRU GC

`Memo(budget_bytes=…)`: retained realized bytes (`payload_bytes`: a Dataset's in-memory
`ArrayProvider` image + attribute arrays; lazy/disk providers count 0) over budget evict LRU
entries. Accounting is keyed by the **unique deduped blob** (`fp → refcount`), so a shared
blob frees only at its last reference.

Eviction is **correctness-safe** — it can only force a later recompute, because output is
deterministic and identity rides the monotonic `revision`. So nothing is pinned, except
`_last_fp` (the Salsa cutoff marker), which is never GC'd. `budget_bytes=None` (the headless
default) is byte-identical to pre-GC behaviour. The GUI's persistent memo caps at 1 GiB
(`runner.MEMO_BUDGET_BYTES`).

Honest trade: an evicted ancestor recomputes with a fresh revision, so the descendant chain
recomputes too. A budget that fits never evicts, and a re-pull is fully memoized.

---

## 11. Streaming evaluation

[`nodegraph/streaming.py`](../../nodegraph/streaming.py) · design record `V2.04`

**Provider chaining.** A `TILEABLE` / `WHOLE_PLANE` / `WHOLE_VOLUME` node returns a Dataset
whose `image` is a *computing* lazy provider, not a realized array. `MapComputeProvider`
serves `read_region` on demand: it decomposes the request into **canonical tiles of its own
grid**, computes each missing tile by reading the tile extent **+ halo** from its base
provider (itself possibly lazy — the chain *is* the dataflow), applies the kernel, and crops.
Only canonical tiles enter the shared `TileCache`, so every level of the chain gets reuse.

The provider family:

| Provider | Role |
|---|---|
| `MapComputeProvider` | per-tile (with halo) or per-plane kernel application |
| `VolumeComputeProvider` | per-volume kernel, cached as per-z planar slabs |
| `_AxisReduceProvider` → `ZReduceProvider` / `TReduceProvider` | tree-reduced projections over z or t (monoid combiners fold per tile; median/sigma-clip/trimmed gather the column) |
| `PlaneRealizeProvider` | geometry-changing per-unit lazy realize (resample) |
| `WindowView` | a crop view over a base provider |
| `realize(payload)` | force a lazy chain (used by `assert_zone_pure`) |

### The correctness pillars

* **Flat fingerprints.** Every streaming provider's identity is a **single digest string
  computed once at construction** from `(op_key, params, declared calibration reads, field
  expression hashes, base fingerprint)`.
  * Folding **declared reads** closes the `reseed_meta` staleness hole — the node-level reads
    fence does not protect the tile cache.
  * Folding **field expression hashes** (which embed layer revisions) closes the
    attribute-layer hole.
  * **Flatness** avoids a nested-tuple `_canon` recursion blow-up on deep unrolled chains.
  * Corollary: **construct the streaming provider LAST in a compute** — reads recorded after
    construction never enter its fingerprint.
* **Halo = overlap-recompute.** Windows are clipped at the *immediate base's* true extents,
  reproducing scipy `mode='reflect'` edge behaviour (probe-verified byte-identical to eager).
* **cum-halo fence.** When `2·cum_halo ≥ tile`, accumulated windows make tiling pointless and
  the provider silently switches to the plane unit.
* **Uniform freeze.** Every array a streaming provider returns is read-only, so an in-place
  kernel fails loudly and deterministically instead of corrupting a cached tile.
* **Kernel-param field gate.** `SocketSpec.kernel_param` marks radius/σ sockets. When such a
  socket is wired to a **non-Const Field**, the kernel is spatially varying, which breaks tile
  translation-invariance and halo sizing — so `_map_image` downgrades `TILEABLE` to the plane
  unit. A future per-pixel kernel-Field consumer must also fold `field_expr_hash` into
  `stream_fp`.

**Deferred (user decision, 2026-07-27):** computed-provider pyramids. `StreamProvider.levels`
stays 1 and the Viewer stride-decimates full-res. Downsample-then-compute is wrong for
non-linear ops, compute-then-downsample buys no savings, and the GPU viewer + prefetch path
already covers viewer smoothness.

> **Amended 2026-07-30.** That last clause was wrong, and the correction is in
> `runner.py`, not here: the prefetch path *assumed* a plane read is a decompress. On a
> `volume_unit` provider it is a whole-unit compute, so prefetching made things worse rather
> than smoother, and the GUI thread's own inline decode of a cold plane froze the app for the
> length of one unit. See the `runner.py` §18 bullets on the warm-inline/cold-async fast path
> and `_prefetch_span`. The pyramid deferral itself stands.

> **One scoped exception, 2026-07-31 — `MultiViewProvider` (`util.stitch`).** The deferral's
> reasoning is a statement about *kernels*: downsample-then-compute is wrong for a non-linear
> op, and compute-then-downsample saves nothing. A **paste** is neither — stitching
> mean-downsampled tiles IS the mean-downsampled stitch (exactly, away from the seams), so
> the coarse level is real work avoided rather than an approximation smuggled in. It is also
> the case where "the Viewer stride-decimates full-res" stopped being affordable: the canvas
> is far larger than any source plane, so displaying one frame of a 49×2048² mosaic stitched
> 172 Mpx to show 3.5 Mpx — **measured 0.92 s/frame (overwrite) and 2.9 s/frame (feather)**,
> paid again on every frame scrubbed to. Forwarding the base's pyramid took that to 89 ms and
> 220 ms. So `StreamProvider.levels` stays 1 for every *computing* provider; only the fusion
> provider overrides it, and only because its kernel commutes with the downsample. The
> general deferral stands.

---

## 12. Providers & storage layout

[`nodegraph/provider.py`](../../nodegraph/provider.py)

`Dataset.image` holds a `TileProvider`. Concrete providers implement `read_region` (a
single-z 2D window) and report `axes` / `levels` / `tile`; the base derives every other read
(`get_tile`, `get_region`, `get_subvolume`, `get_region_volume`) from it.

**The keystone benchmark (real 6554² ND2 plane, 2026-07-21) settled the layout**, and this is
load-bearing for the whole streaming design:

* A 512² block ROI costs **1.5–4.8%** of a whole-plane read → **tiled**.
* For 3D, **planar `(1,512,512)` blocks win**: a z-range subvolume reads in ~11% of
  whole-volume with per-z blocks vs ~47% with a fat z-spanning block, because blosc2
  decompresses whole blocks. So **block = one 2D tile per z**, and a subvolume is *gathered*
  from planar blocks across the z-range.

Providers: `SyntheticProvider` (deterministic formula, numpy only), `B2ndProvider` (Blosc2
b2nd store with planar blocks, in-memory or on disk; blosc2 lazily imported), `ArrayProvider`
(a realized array, reports `nbytes` for the memo GC), and `FrameSubsetProvider` /
`FrameSliceProvider` — pure `(m,t,z)` index remaps that shrink a source to a chosen set of
frames (or one) and, inside each, a chosen set of planes. They compute nothing, so they
inherit `depth`; they are how the GUI scopes a *run* without editing the graph (§18).

`TileProvider.version` (C5) defaults to `fingerprint()`; disk providers fold `mtime_ns`. That
is what the engine's `__provider_version__` hook uses so a file changing on disk cannot serve
a stale chain. For a b2nd store the mtime is **level 0's only** (V2.19) — every compute reads
level 0 and the pyramid above it is a display convenience, so *completing a pyramid* must not
re-key the source and discard every memoized result under it.

### The pyramid: streamed, marked, repairable (V2.19)

Levels above 0 are mean-pooled 2× per level, and three things about how they are written are
load-bearing:

* **Level *l* streams out of level *l-1*'s store**, one chunk-aligned z-slab at a time
  (`_append_level`), with the per-plane reduction in one place (`_plane_mean_2x`), **on the
  worker pool**. The old writer downsampled a whole level in RAM via `_mean_downsample_2x`
  and handed the array to blosc2 — for the lab's 84.7 GB ND2 that is a 21 GB allocation made
  *while* the 84.7 GB source is still live, and it is exactly why that file's store on disk
  had a complete `level_0` and no pyramid at all. `_mean_downsample_2x` stays as the in-RAM
  reference the selftest asserts the streamed levels against, bit for bit.

  The pool part was a **V2.20 repair, and the omission was expensive**: V2.19's first cut
  looped the slab's planes serially, silently giving up the fan-out `_mean_downsample_2x`
  already had. Streaming never required that. Measured on a 1.64 GiB slice of the 640 series,
  24 cores, `levels=3` — decode 2.0 s, level 0 compress 2.3 s, and **level 1 alone 6.0 s**:
  a quarter of the data for nearly three times level 0's cost, one core mean-pooling 840
  planes while 23 idled. On the pool level 1 is 1.8 s and level 2 is 0.2 s, taking the whole
  ingest **11.9 s → 6.3 s (141 → 266 MB/s), a 1.9× speedup for bit-identical output**.
* **Every level is marked twice** in its `vlmeta`: `complete: False` right after
  `blosc2.empty` declares the geometry, `complete: True` once the last chunk lands. Marking
  the *start* is the part that matters — a b2nd array declares its full shape up front and
  unwritten blocks read as **zeros**, so stamping only on success would leave a half-written
  level indistinguishable from a pre-V2.19 one. No marker at all means *legacy*, which the
  marker check trusts — but see V2.20 below: that trust is no longer the last word.
* **`open` takes the leading sound run.** A torn level above 0 is dropped (the store opens
  short, and is repairable); a torn `level_0` is a hard error naming the fix, because nothing
  but the source file can rebuild it. A *gap* (`level_0` + `level_2`) is refused outright —
  `levels` is a count and every reader indexes by position, so accepting it would serve
  level 2's pixels to a level-1 read.

**`B2ndProvider.ensure_levels(urlpath, levels)`** is the repair: level *l* is a pure function
of level *l-1*, so a store with an intact `level_0` regains its pyramid **without the source
file being read, or even existing**. Idempotent, memo-neutral, and progress-reporting. The GUI
runner calls it on every resolve where `prov.levels < PYRAMID_LEVELS` (§18), so a store torn
by an earlier crash heals on the next pull instead of silently serving full-res planes to
every zoom level forever. Measured on the lab's 640 series (12×16×210×1024²): **463 s** to
build levels 1–2 from level 0, no ND2 touched, fingerprint unchanged — that figure predates
the V2.20 parallel reduce above, which cut the same two levels by **3.8×** on a slice of that
series, so expect ~2 minutes now.

### "Legacy is trusted" was wrong, and the census replaces it (V2.20)

V2.19 read the lab's 640 store as *a complete `level_0` with no pyramid* and repaired it on
that basis. It was not complete. `level_0` held **7 899 of its 40 320 chunks** — the write had
stopped at m=2, t=5, z=129 — so positions m≥2 loaded as solid black while the ND2 itself was
perfect. The marker could not say so (the store predates it), the geometry looked right, and
the 463 s repair then read those zeros and wrote two more levels of them, *marked complete*.
The pyramid fix worked exactly as designed on a premise that was false.

Two changes, one for each half of that:

* **Detection — `B2ndProvider._blank_tail` (a chunk census, no marker required).** blosc2 files
  a never-written chunk as a special run-length value rather than as data, so *which chunks
  were written* is a fact on disk readable from 32-byte headers: 2.7 s for those 40 320, zero
  decompression. The verdict is deliberately narrow — a **pure trailing run** of blanks with a
  fully-written body. blosc2 files a genuinely-written constant chunk the same way, so a blank
  chunk alone proves nothing (a mask is legitimately blank almost everywhere); only a write
  that *stopped* leaves blanks exclusively at the end. Scattered blanks read as sparse and are
  not flagged, which is the right way round for a verdict that costs a re-ingest.
* **Policy lives at the ingest seam, not in `open`.** `nodelab_v2.ingest.verify_store` gates the
  census on `level_state(0) == "legacy"` (so it self-sunsets — a marked store skips it) and
  raises `ValueError`, which `_resolve_source` already catches and answers with a real
  re-ingest. It is *not* in `B2ndProvider.open`, because density is a claim about ND2/TIFF data
  specifically: `nodegraph.checkpoint` opens the same class over a raster that may be a mask,
  and keeps its own manifest-written-last rule. `ensure_store_levels` verifies too — repairing
  a torn level 0 is precisely how the zeros got propagated the first time.

**And the cause, not just the symptom: ND2 ingest no longer materializes.** `ingest_image` fed
`B2ndProvider.write` a realized 6-D numpy array — for this file an 84.7 GB `np.empty` filled
before the first byte was compressed, then held live while blosc2 read out of it. Any death in
those tens of minutes leaves a torn store. `_build_levels`' level-0 loop already reads exactly
one z-slab per write, so it only ever needed a *lazy* source: `lazy_nd2()` hands it nd2's
`ResourceBackedDaskArray` (valid after the file handle closes — it re-opens per block) and peak
memory becomes one slab. Byte-identical output, verified against the eager reader on a 3-channel
5300² plane and a 51×2 z-stack. TIFF and the in-memory path keep the two-phase read (`tifffile`
has no lazy form), which is what `_READ_SHARE` still splits the bar for.

**It costs nothing in wall time** — the concern to check, since the eager read got one big
parallel dask compute per (m,t) and the streamed one gets a compute per z-slab. Benchmarked
a-b-b-a on a 1.64 GiB slice of the 640 series so the cold page cache lands on one run of each,
best-of-two: eager 6.7 s vs streamed 6.2 s, **0.94×**. The write dominates either way (decode
is 2.0 s of it at 821 MB/s), so removing the materialize pass trades 84.7 GB of RAM for nothing.

---

## 13. Structure, transfer, bridges

### Structure tables

[`nodegraph/structure.py`](../../nodegraph/structure.py)

Detected structures are computed **whole** (a whole plane in 2D, a whole volume in 3D — never
per-tile CCL) and stored as **columnar tables** (Arrow-target):

```python
StructureTable(domain, columns, layer=None,
               z_kind="subpixel"|"plane_index", channel_kind="single"|"per_point")
```

The schema is **invariant**: `COORD_COLUMNS = (id, m, t, c, z, y, x)` always present, `z`
**never NaN** (2D uses the integer plane index). That invariance is what lets bridges avoid
branching on mode. `content_hash()` is a cheap full hash; `to_arrow()` lazily imports pyarrow
and stamps `z_kind`/`channel_kind`/`layer` into the schema metadata.

Producers: `label_components` (CCL), `seeded_watershed`, `point_table`. Connectivity: 2D
`{4,8}`, 3D `{6,18,26}` via a shared `_connectivity_rank`. `label_components` uses
`scipy.ndimage.label` + a **raster-canonical relabel** + `bincount` areas + `center_of_mass`
centroids, and is **byte-identical** to the pure-numpy flood-fill kept as
`_label_components_flood` (the reference and the scipy-absent fallback) — asserted equal
across 2D/3D connectivities in the selftest. ~40× faster at 512²; the ~191 s cliff at 6554² is
gone.

`TrackMembership(track_id, t, member_id, member_domain)` **defines** the Track domain: one row
per (track, timepoint) occupancy. `tracking.py` supplies the two dep-free deterministic
linkers (`link_labels` max-IoU, `link_points` nearest-neighbour) plus the public
`build_membership` (contiguous first-appearance ids, rows sorted `(track_id, t, member_id)`).
Both assume member ids are **globally unique across timepoints** — which is exactly what
`analysis.label` and `detect.spots` emit, and what the bridges require.

**Nothing in v2 consumes `Domain.TRACK`** (the bridges need a live `TrackMembership` object no
node can rebuild from a Dataset), which is why `track.objects`' **`track_id` write-back onto
the member layer** is the load-bearing half of its output, not a nicety.

### Transfer & bridges

[`nodegraph/transfer.py`](../../nodegraph/transfer.py) · [`bridges.py`](../../nodegraph/bridges.py)

`plan_transfer(A, B)` → a `TransferPlan`:

* **Lattice ↔ lattice** is *generated* (coarsen = reduce over dropped axes, refine =
  broadcast) and executes here on numpy arrays (`execute_transfer` / `lattice_transfer`).
* **Structure hops** use **explicit registered bridges** with metadata: `voxel_to_label`
  (mean-in-mask / sum/count/max/min/median), `label_to_voxel` (paint-by-label),
  `voxel_to_point` (sample-at-position: nearest numpy / linear via lazy scipy),
  `point_to_voxel` (splat), `containing_label`, `points_in_label`, `gather_by_track`,
  `broadcast_track`, …
* **Indirect pairs** are **routed by BFS** over the domain graph (the lattice mega-edge +
  registered bridges) into a multi-hop plan, chained per frame/volume by
  **`execute_bridge_plan(carrier, plan, label_raster=…, points=…, membership=…, shape=…)`**
  (C2). Voxel→Track resolves as Voxel→Label→Track.
* Track↔Timepoint (member×t) and Frame→structure broadcast still raise — call the `bridges`
  functions directly there.

### The constructive boundary spine

[`nodegraph/boundary.py`](../../nodegraph/boundary.py)

`fill_boundary` / `extract_boundary` are **explicit data-layer operations, deliberately NOT
auto-routed bridges** (locked decision 6) — they are absent from `transfer._BRIDGES` so a
constructive fill never becomes an implicit routing edge. **The input geometry is the
dimensionality authority**: points on a single z fill as a planar contour on that plane (never
extruded); points spanning z fill per-plane. The header 2D/3D toggle is a **consistency
assertion only** — it hard-errors on contradiction, it never overrides the geometry.

---

## 14. The mesh domain

[`nodegraph/mesh.py`](../../nodegraph/mesh.py) · design record `V2.08`

Mesh is the 11th domain, added because **topology** (which vertices form which face) is the
one thing no other domain encodes: vertices ≈ Point, the filled region ≈ Voxel Label,
"which vertices belong to which region" ≈ Label-over-Points. Faces do not exist elsewhere.

Blender splits mesh data across POINT/EDGE/FACE/CORNER precisely because a mesh has several
different array **lengths**. Here one new domain absorbs that by riding the `layer` sub-key —
a mesh named `L` occupies **three internally-uniform buckets**:

```
(MESH, "L",      *) → K   rows, one per mesh ELEMENT (one closed surface)
(MESH, "L/vert", *) → Nv  rows, the vertex pool
(MESH, "L/face", *) → Nf  rows, the face pool (triangles)
```

Each bucket is a legal `StructureTable` carrying the invariant coordinate schema, so `.n`
stays meaningful, `to_arrow` works per bucket, and the Spreadsheet renders three clean tables
instead of one ragged one.

**Every array is flat and fixed-width (int64/float64)** — no object dtype, no ragged column,
no 2-D array, no live `scipy.spatial.Delaunay`. That is a *correctness* requirement, not
tidiness: see the `_canon` object-dtype hazard in §10.

**Element → pool addressing is CSR**: each element row carries `vert_start/vert_count` and
`face_start/face_count`. `start`+`count` rather than a Blender-style `(K+1)` offsets array
specifically so *every* element column is exactly length K.

**Vertices are voxel `(z,y,x)`**, deliberately against the vendored kernels' own
`vertices_um` convention: every other structure table's `z,y,x` are voxel coords and
`COORD_COLUMNS` carries no unit tag, so a µm mesh would be the only geometry whose *meaning*
depends on calibration not captured in the layer — a recalibrated upstream would silently
reinterpret it. Producers convert on the way in, consumers on the way out, and the µm↔voxel
flip stays inside computes behind `ctx.calib` where the memo fence lives.

Triangles only for now (every producer emits `(Nf,3)`); n-gons later mean a new `L/corner`
bucket, not a migration.

`analysis.tessellate` → MESH → `transform.rasterize_mesh` replaced the deleted fused
`analysis.tessellate_volume`, and fixed a real bug: alpha-shape always rasterized its *convex
hull*, because the interior test had nothing to read. Now it derives from the mesh's own
provenance, so concavity survives.

---

## 15. Fields

[`nodegraph/field.py`](../../nodegraph/field.py)

A **field** is a value socket carrying a *deferred function*: evaluated per-element on the
**consuming** node's domain, only where consumed. The IR is deliberately minimal: `Const`
(→ a non-materializing `VirtualArray`), `Attr` (read a layer; a layer on another **lattice**
domain is transferred to the consuming domain by the default rule shown on the wire),
`Input`, `BinOp`, `UnaryOp`, `Where`.

The field memo key is `("field", field_expr_hash, domain, kernel_axes, token)` — a namespace
**disjoint** from the image tile key. `field_expr_hash` folds operator identity + params +
referenced layer **revisions** (so a changed layer invalidates) + `kernel_axes` (so a 2D vs 3D
stencil field never collides). `FieldCache` is backed by the same byte budget as the tile
cache. Structure-domain field transfer is deferred.

---

## 16. Zones & groups

### Zones — unroll + revision-fold

[`nodegraph/zones.py`](../../nodegraph/zones.py)

A zone bounds a body subgraph between paired `In`/`Out` nodes with a **back-edge**
`Out → In` (`Edge.kind == "back"`) carrying the feedback. `unroll` expands it into a **flat
per-iteration chain**: iteration `i` is a distinct copy of every body/boundary node, and the
back-edge becomes a *forward* edge `Out@(i-1) → In@i`.

Because the result is an ordinary acyclic graph, **the engine and memo run unchanged**, and
iteration `i`'s recipe hash naturally folds iteration `i-1`'s `Out` revision → correct
incremental invalidation. Re-pulling an unchanged graph is fully cached; editing the seed or a
body param invalidates from that point on; a downstream edit leaves the zone cached.

* **Iteration 0 re-initialises**: the external Dataset wired into `In` feeds only iteration 0.
  A loop-invariant external input wired straight to a *body* node feeds every iteration.
* **Purity** is assumed in the hot path. `assert_zone_pure` double-computes and compares
  fingerprints (debug-verify). A zone flagged `impure` folds a per-unroll `epoch` salt into
  its copies so it is non-cacheable across pulls — the escape hatch for genuinely
  stateful/random bodies.
* **Per-frame-T** is a Simulation specialization with **no schema change**: a `zone.frame`
  node in the body is stamped `__frame__ = <iteration index>` by `unroll`, and its compute
  slices that frame of its T-stacked input. Iteration *t* processes frame *t* while the
  `In`/`Out` feedback carries state — exactly the Blender split. The caller sets
  `iterations = T`.

### Groups — inline expand

[`nodegraph/groups.py`](../../nodegraph/groups.py)

A `Group` is a reusable subgraph *definition* bounded by `group.input` / `group.output`. An
*instance* is one node whose `op_key` is `"group:<name>"`. `expand` inlines every instance
into its parent (body node `N` → `f"{N}%{instance_id}"`) and stitches the interface: parent
edges into the instance rewire onto the copied `Group Input`; the copied `Group Output` feeds
the parent edges out. Boundary nodes **persist** as identity pass-throughs so
`group_output` can name a group's external output.

**Nestable with a fixed point**: a body is fully expanded *before* it is copied, so no
residual `group:*` node survives. Recursion is guarded by an expansion-path stack; a
self-containing group, an unknown reference, or a malformed definition raises clearly.

**Ordering that matters:** `document.to_graph(materialize=True)` calls `groups.expand`
**before** channel-tap materialization, so the engine/memo/metadata pass need no group
awareness. `propagate_meta` patches each instance's envelope from its body OUTPUT
(`grp.out%inst`), which is how downstream domain rails read correctly *through* the opaque
instance.

---

## 17. Serialization

[`nodegraph/serialize.py`](../../nodegraph/serialize.py) — the headless core of
`*.nd2graph.json`.

Round-trips the **structural** model only: `Graph` / `NodeInstance` / `Edge` (**including
`kind="back"` zone-feedback edges**, which are load-bearing), `Zone`s, and `Group`s (whose
body is a nested `Graph`, serialized recursively).

* **JSON-native values only.** `params`/`modes` hold plain scalars/containers. No numpy, no
  Qt, no `nodegraph.nodes` import — the saved model is pure structure.
* **Deterministic output**: node/edge/zone/group lists in a stable sorted order plus
  `sort_keys=True`, so saved files diff cleanly.
* **Validated on load**: an unknown/absent `format_version` or malformed structure raises.
* **Extension point**: unknown extra keys on a node object are tolerated on load, so a
  forward-written file still reads back.

GUI-only per-node state (canvas position, mute/collapse, frame membership, the source's title
and channel descriptors, the `__locked__` pin list) rides in a **top-level `ui` object the
headless loader ignores**.

---

## 18. The GUI

### `document.py` is the source of truth; the canvas is a view

`GraphDocument` (Qt-free) owns node records (`op_key` + the **live** `params`/`modes` dicts
that both the graphics items and the inspector mutate), edges, and GUI-only extras. It:

* validates every connection through `nodegraph.sockets.can_connect` on the two sockets'
  **active** specs + cycle rejection + non-multi replace,
* re-runs `propagate_meta` after every structural edit (the live widget re-seed),
* builds a real `nodegraph.Graph` on demand — `to_graph(for_run=True)` strips UI-only params
  (`__locked__`, `__title__`, `__channels__`) and **bypasses muted nodes**, so the engine never
  sees them,
* round-trips the file,
* owns `make_group` / `ungroup`, zone wrapping, frames, splice, dissolve.

`scene.sync()` (wired to `document.on_change`) reconciles cards and rebuilds edge items from
the model, so the canvas **cannot drift** from what will be saved or run.

### `runner.py` — the engine bridge (G7)

A `QThreadPool` worker + an **epoch registry**; no qasync, because the engine is synchronous
CPU work.

* **Snapshot at submit** — the headless `Graph` is built on the GUI thread, so the worker
  never touches live GUI state.
* **A persistent `Memo` across runs** — engines are rebuilt when the document revision
  changes but the memo carries over, which is what makes "an unrelated edit recomputes only
  the invalidated chain" user-visible. Capped at 1 GiB (`MEMO_BUDGET_BYTES`).
* **A persistent `TileCache` too, and it is not optional** (2026-07-30). The runner owns one
  and passes it into every `Engine` it builds (`Engine(tiles=…)`). A `StreamProvider` holds
  its cache by **weakref** so an orphaned engine's cache can be collected — but the memo
  hands out lazy Datasets *built by an earlier engine*, so with a per-engine cache every
  memo-hit lazy chain reads through `_NO_CACHE` from the first revision bump onwards: no
  tile is ever stored again and **every plane re-runs the whole unit**. Measured on the
  WellA3 640 series (12×16×210×1024², 3D Deconvolve, one unit = 210×1024² ⇒ ~130 s with the
  V2.19 kernel, ~230 s when this was found):
  z-scrubbing went from 0.00 s per step (cached per-z slabs) to a fresh whole-volume
  Richardson–Lucy per step, permanently, after any edit — and the revision bumps by itself
  on the first pull, via the `set_meta_seed` re-seed. Sharing is as sound as sharing the
  memo: the tile key embeds the provider fingerprint (op + params + declared reads + field
  hashes + base fp + grid), so it is content-addressed, not engine-scoped.
* **Source resolution** — an `io.load` root resolves `path` through `ingest` (ingested once to
  an on-disk b2nd store next to the file, re-opened lazily); an empty path falls back to a
  deterministic `SyntheticProvider` with real calibration. The resolved envelope is delivered
  back to the document (`set_meta_seed`) — the live re-seed.
* **Plane rendering** — a job may ask for a display plane at `(m,t,z,c)`, read in the worker
  through the tile cache and decimated to `max_dim`, so the GUI thread never blocks on a lazy
  chain.
* **The scrub fast path is warm-inline / cold-async** (2026-07-30). A coords-only request
  (`request_plane`) bypasses the graph snapshot and serves the `PlaneCache`. A **warm** frame
  is emitted right there on the GUI thread — the pixels exist, so it costs a dict lookup and
  scrubbing keeps zero-hop latency. A **cold** one goes to `_DecodeJob` on the pool, because
  a `PlaneCache` miss on a *computing* provider does not decode a plane, it **runs the
  node**: the same 3D Deconvolve read that costs 0.00 s warm costs ~130 s cold, and doing it
  inline froze the whole application for that long with no repaint and no status. One decode
  in flight with a single latest-wins pending slot (the `pull` shape) — a `volume_unit`
  provider carries a whole unit's working set per concurrent job (8.5 GB for that volume), so
  one job per pool thread is not an option. Staleness is judged on `_decode_gen` (cursor moved) and `_epoch` (edit or
  real pull); a dropped result still populated the cache, so a cursor that comes back is
  warm. The card + status bar say *reading planes* for the duration.
* **The prefetcher is cost-gated** (`_prefetch_span`, 2026-07-30). ±8 T-neighbours is right
  for a store-backed provider (a decompress each) and catastrophic for a computing one:
  warming ±8 T across a `volume_unit` Deconvolve queued **sixteen whole-volume RL computes**
  — over half an hour of CPU behind a cursor that had moved one frame, starving the Viewer's own
  decode and pushing the volume being *looked at* down the cache LRU. Span is now 8 on real
  bytes, 2 on a per-plane/per-tile compute, **0** across frames of a whole-unit compute
  (there the useful neighbours are the other z of the unit, already cached by the read that
  displayed it).
* **Latest-wins queueing**, and stale results are dropped by epoch on arrival.

### `ops.py` — the two GUI-introduced ops, deliberately Qt-free

* `io.load` — the source. **It has no compute**: the engine gets pixels from a seed Dataset.
  A headless consumer supplies its own seed per `io.load` node (`headless_engine`).
* `view.viewer` — a pass-through inspection tap, registered into `COMPUTES` *here* (not in the
  Qt runner) so `Engine` can run a GUI-authored graph without importing PySide6.
* **`materialize_channel_taps`** — `io.load`/`channel.split` expose *synthetic* per-channel
  output sockets `ch0…chN-1`. The engine is one-payload-per-node, so these cannot be distinct
  engine outputs; instead every `chK` edge is rewritten into a real `channel.select` tap
  (`params={"channels":[K]}`) at graph-build time, reusing the tested select compute and its
  lockstep `channel_select` meta_transform. This runs for the run graph, the edit-time
  envelope pass, and any headless consumer alike. **Do not add multi-output to the engine for
  this.**

### Viewer, GL path, overlays

* `viewer.py` composites ready per-channel float planes into an RGB `QImage`; decoding stays
  in the worker.
* `glview.py` is the GPU path (default; `NODELAB_GL=0` forces CPU, and a GL failure emits
  `gl_failed` once and swaps in the CPU view). One textured quad per frame; each channel's
  plane is uploaded **once** as `R32F` (the uint16→float cast happens at upload, so the shader
  uses a plain `sampler2D` for every channel); contrast/gamma/colour/compositing are all
  fragment-shader **uniforms**. Moving T/Z rebinds textures; dragging a LUT is a uniform
  change — zero decode, zero re-upload.
  Reparenting (docking into the mini-map and back) destroys and recreates the context, so
  `_release_gl` frees textures/buffers/program while the dying context is current and
  `initializeGL` rebuilds and replays `_last_planes`.
* `overlays.py` owns **one** renderer for every domain. Three structural properties:
  everything is painted in **widget space in screen pixels** (given a `plane px → widget px`
  mapping), so thickness is zoom-invariant — region *fills* are the one thing that scales,
  because a fill *is* the region; **both backends share this code** (each calls back with a
  `QPainter` in widget coordinates and exposes `plane_to_widget`); and **field specs drive the
  UI** (`FIELDS` describes each setting's kind/range/dependencies and its cross-domain
  *role*, which is how "spread" copies a look by role). Settings persist in three merging JSON
  layers: defaults → the committed project file → this machine's file, or one explicit file via
  `NODELAB_OVERLAYS`.
* **The identity palette.** Per-item colour keys off a **palette slot**, not a raw id, because
  a segmentation re-issues label ids from 1 every frame — colouring by the id makes one cell
  flash a new colour at every T step. `viewer._build_palette()` walks the dataset's structure
  tables ONCE per dataset (not per frame) and produces, per layer, a dense `id → slot` LUT:
  a row's object is its track when one links it, itself otherwise, and a track *prefers* the
  slot of its first member so an untracked field keeps exactly the colours it already had.
  `overlays.deconflict_slots()` then moves an object off its preference only when a neighbour
  — from a per-frame k-NN over member centroids, deduplicated across frames — sits within
  `MIN_HUE_SEP` (30°) of it; the golden angle spreads *consecutive* indices, so without this
  slots 5 and 39 land 4.7° apart and two touching cells read as one. The LUT rides on
  `OverlayFrame.label_keys` (dense, because the fill indexes it per pixel), on
  `PointMark.key`, and on `TrackPath.color_key` — which is what makes a trajectory paint in
  the same colour as the regions it threads. `None` anywhere means "colour by raw id", the
  pre-palette behaviour, and is what the preview and every direct renderer call use.

### The hover readout (`ViewerPanel._hover_text`)

Pointer position → one line: pixel `(x, y)` in the viewed node's grid, that point in µm, the
**absolute stage coordinate**, and each shown channel's value with the **raw** file value
beside it. Three seams make it work, and each is a refusal boundary:

* **Stage geometry is display metadata, not calibration.** `stage_xy_um[m]` (the ND2
  XYPosLoop's per-position field CENTRE) rides on the seed `Dataset` via
  `ingest.STAGE_KEYS`, alongside the channel names — *not* in `CALIBRATION_KEYS`. It is
  per-M geometry, no `meta_transform` knows how to keep it true across a crop or a
  resample, and putting it in the locked schema would imply the engine maintains it.
* **Raw pixels come from the runner, through the window.** The Viewer holds no provider, so
  `window` hands it `EngineRunner.raw_plane` as `raw_plane_cb` — the same division as
  `arm_pick`'s calibration. `raw_source` gates on **exactly one** `io.load` in the pull's
  ancestor closure AND the node's propagated axes **equalling** the source's on all six.
  Enhancements pass; anything that re-addresses `(m,t,z,c,y,x)` does not, and the readout
  then withholds both the raw value and the stage coordinate rather than naming the wrong
  pixel. `raw_source` is memoized per `(node, document.revision)` — the caller is a
  mouse-move handler and the uncached answer costs a run-graph build — but a **refusal is
  never cached**, because the commonest refusal is "the source has not resolved yet" and
  that resolves at an unchanged revision.
* **The event filter is permanent.** `_install_surface_filter` installs the panel on the
  surface (and its viewport) for good and forces mouse tracking; `_install_pick_filter` now
  only swaps the cursor. Wiring the readout to the pick-time filter would have left it dead
  in ordinary use — and a GL→CPU fallback replaces the filtered widget, so both
  `_do_fallback_to_cpu` and construction re-install.

Cost per move: two array index reads and a string build. The raw plane is cached in the same
`PlaneCache` as the display planes, and when the viewed node *is* the load (no pin) it reuses
the display key outright, so hovering a source costs no extra read.

Five PySide6/GL landmines are documented in `glview.py` — VAO handling, float-array uniforms
via `QVector2D`, the RGBA8 upload constraint, no sampler-in-function, deferred fallback. Read
them before editing that file.

### Frame strips and the run scope (F9)

`framestrip.py` replaces the Viewer's M/T/Z sliders with **one box per frame** (fixed size,
compressing to fit once the row runs out of width). It carries two independent pieces of
state: the **cursor** — what is displayed, dragged like the slider it replaced (the pointer
is grabbed on press, so a drag past either end keeps scrubbing against the clamp) — and a
**selection**, entered with Ctrl / Shift / the context menu precisely so that plain dragging
stays scrubbing. The widget is axis-agnostic; what an empty selection *means* is a scope
policy the Viewer states per strip in `unpicked_note`.

The selection is the **run scope** the troubleshooting mode evaluates, and the whole seam is
three hops with nothing in between:

* `ViewerPanel.frame_selection()` → `EngineRunner.set_frame_selection()`, as `(ms, ts, zs)`.
  The axes pick independently, so what runs is their **cross product**. The two fallbacks
  differ and the asymmetry is deliberate: an empty M or T means *the frame the cursor is on*
  (so untouched strips reproduce the original one-frame scope exactly), an empty Z means
  *the whole volume* — z is inside a frame, and cutting a 3D node to one plane because a
  cursor happened to sit there would be a trap.
* `_pin_frames` wraps every resolved **source seed** in a `FrameSubsetProvider` and shortens
  its envelope to match. **No node participates**: the catalog keeps looping "every frame",
  there are just few, each shorter in z. That is why this is a seed concern and not a graph
  edit — the canvas, the run plan and the saved file are untouched.
* The picks ride the provider `fingerprint` → `version` → `__seed_version__` → every
  downstream `recipe_hash`, so one selection can never serve another and a revisit is a memo
  hit. An unset `zs` is left out of the fingerprint entirely, so a frame-only scope keys the
  memo exactly as it did before z became pickable.

**Landmine 1 — addressing.** Under the scope a payload holds only the picked frames and
planes and numbers them from 0, while every strip, label and request stays in *global*
indices. `provider.subset_index` maps one to the other (an unpicked cursor resolves to its
nearest picked neighbour; an unpicked axis passes through) and it must be applied **exactly
once**, at the boundary where a global cursor first meets a payload —
`ViewerPanel._payload_coords` on the display side, `EngineRunner._payload_coords` before any
plane decode or prefetch. Re-applying it remaps an index that is already an index.

**Landmine 2 — a z pick changes what 3D nodes compute.** Skipped planes are *absent*, not
empty, so a Gaussian 3D over 5 picked planes is not the same filter as over all 60, and
`z_step_um` still describes the source. That is the mode, not a bug — but it means a scoped
result can only be compared against an unscoped one on a node with no z (or t) context. The
GUI probe's T3 checks pixel identity on the **source** node for exactly this reason.

### Inspector

Builds the form from the `NodeSpec`: the 2D/3D switch (disabled when incoming z is *known* to
be 1 — unknown ≠ 1), the resolved footprint, one row per **active** parameter with its unit
label, and the **auto / pinned** toggle backed by the sticky `__locked__` list. Changing a mode
rebuilds via a deferred `_rebuild`, and the card re-lays-out via `scene.sync → item.refresh` —
**both halves must be verified** when you add mode gating.

### Interactive parameters (V2.16) — `picker.py`

Four presentation-only `SocketSpec` fields drive the whole feature, following the `path_kind`
precedent exactly: **never hashed**, validated at registration (a typo in any of them would
otherwise fail *silently*, by simply not drawing a control).

| Field | Effect |
|---|---|
| `pick_kind` | the gesture this param offers (one of `PICK_KINDS`); draws the Inspector's **Pick** button and the card's `○` glyph |
| `pick_peer` | another input on the same node that the same gesture also sets (a min/max interval, a grid's box + stride) — an interval, aimed in **two phases** |
| `pick_bounds` | the complete ordered GROUP one gesture writes at once (`BOUND_PICK_KINDS`: `rect` → `("y0","y1","x0","x1")`, `zrange` → `("z0","z1")`) — a rectangle *is* four numbers, so there is no ordering and no second phase |
| `choices` | a STRING param's closed value set → a dropdown |
| `vocab` | a STRING param's multi-select set → a tick list, stored as the comma string the compute already parses |

**`nodelab_v2/picker.py` is Qt-free and holds the state machine and the maths** — what a
gesture means, how plane pixels become microns, what value gets committed. The viewer owns
only mouse events, painting and *sampling*, feeding sampled numbers in as a `probe`. That
split is why every conversion is checked headlessly by `selftest.test_picking` rather than
only by eye.

**The image-surface contract is `getattr`-probed, so an omission is silent.** A surface must
provide `plane_to_widget` / `widget_to_plane` / `refresh` / `overlay_cb`
(`viewer.SURFACE_CONTRACT`), and the panel reaches all four through `getattr` because the
backend is swapped at runtime (a GL failure falls back mid-session). `widget_to_plane` was
first added to the CPU `_ImageView` only — so on the **GPU path, the default in a windowed
session**, `_pick_plane_pt` returned `None` for every click and *every* on-image gesture did
nothing at all, with no error. The offscreen tests all force `NODELAB_GL=0` and passed
throughout. `check_surface_contract()` now gates both classes in the GUI probe, and the probe
additionally drives one gesture with **real `QMouseEvent`s through the widget** — driving
`session.press()` directly, which every other check does, is exactly what let a dead event
path look healthy.

`widget_to_plane` returns `None` **outside** the image rather than extrapolating: the picture
is letterboxed in the viewport, and accepting margin clicks produced ROI shapes with negative
vertices that rasterized to nothing.

Three surfaces, keyed off `PickRequest.surface`:

* **canvas** — an event filter on the image widget *and* its viewport (the two backends
  deliver mouse events to different objects; consuming the press is also what stops the CPU
  view's `ScrollHandDrag` panning, with no mode to set and restore). Wheel is deliberately
  **not** consumed — zooming is part of aiming.
* **histogram** — reads the live LUT window/gamma on Apply.
* **instant** — commits from the viewer's cursor with no bar at all.

Landmines this cost:

* **Two pixel spaces.** A payload can arrive decimated, so `_disp_scale()` converts between
  *displayed* and *axes* pixels. Every committed value is in the node's own full-resolution
  space; the same `sx/sy` the overlay renderer uses, for the same reason.
* **Calibration is the target node's propagated envelope**, not the source file's — a Crop or
  Resample upstream changes what a pixel is worth. Uncalibrated files report pixels and the
  readout **says so**; silently relabelling px as µm is the one failure mode here that
  produces a wrong number with nothing on screen to show it.
* **A peer with no declared unit inherits its partner's** (`PickRequest.unit_for`). Defaulting
  to px made phase 2 of a pair silently wrong by the pixel size.
* **A peer must be gated identically to its partner.** `analysis.histogram_threshold`'s
  low/high are gated apart under `direction=below|above`, so they are *not* peers — co-picking
  there would write a param the user cannot see. `test_picking` enforces this across every
  mode-state product, for `pick_peer` pairs **and** `pick_bounds` groups.
* **A bound group must be homogeneous** (one SocketType, one unit), enforced at registration.
  That is what lets `PickSession` convert the whole group using the armed socket's own
  type/unit instead of threading a per-name table through every call — the invariant makes the
  simple code *correct*, not merely lucky. It is also why crop's lateral rect and its axial Z
  range are two groups, not one: they are gated apart on the 2D/3D lever, and a rectangle
  drawn on a plane says nothing about z.
* **A group draws its affordance once**, on `bounds[0]` (`PickRequest.leads_group`) — four
  identical "Drag the crop rectangle" buttons down the panel is worse than one. Every member
  still advertises the gesture in hover text.
* **`rect`'s release position wins** over the last mouse-move. With a real mouse they
  coincide, but relying on that makes the committed window depend on how Qt coalesced the
  drag. The bounds are a slice with an **exclusive** end, so the pick floors the starts and
  ceils the ends: `[start:end]` then covers exactly the pixels drawn. An off-by-one here still
  looks plausible in the pills, which is why the probe compares **pulled pixels** against the
  source window rather than checking the four numbers.
* **A pick is an ordinary edit**: `window._on_pick_committed` writes value + sticky pin, the
  same two lines as `inspector._set_param`. There is no second kind of parameter value.

### On-canvas parameter editing (V2.16) — `node_item.py`

Value pills are painted, hit-tested controls (drag-scrub / click-to-type / toggle / mode menu
/ `ƒmd` pin / `○` pick), not embedded widgets — a `QGraphicsProxyWidget` per parameter would
cost a real widget per row on a canvas that repaints at 60 fps while wires animate.

* `NodeItem.controls()` is **the single source of both the hit rects and the painted ones**;
  geometry must therefore be derivable outside `paint()`. Add a control in one place only.
* `mousePressEvent` consumes the event when it hits a control, which is also what stops
  `ItemIsMovable` dragging the card — there is no drag mode to suppress and nothing that can
  disagree.
* A live scrub writes `rec.params` and repaints but **does not** call `doc.touch()`; the
  release does it once. `touch()` re-propagates metadata across the whole graph, which is not
  a per-mouse-move operation.
* Pills are width-capped and elide (`_PILL_MAX_W`); before that a long string value ran
  underneath its own row label.
* Glyph coverage: the shipped Windows UI fonts have no `◎`/`▭`/`⬠`/`✎` — they render as tofu.
  `PICK_GLYPH` is a plain `○`, the tool buttons are words, and the card's ring is painted with
  `QPainter` rather than drawn from a font.

---

## 19. Invariants & landmines

Each of these cost real debugging time. They are listed in the order they tend to bite.

**Registry / tests**

* **Test fixtures MUST use fake op_keys** (`test.*` / `io.*` / `eng.*`). A fixture
  `define_node`-ing a *real* op_key clobbers it in the global `NODES` registry → an
  order-dependent "passes alone, fails in the suite" failure.
* A test must **not** poke `runner._providers[("synthetic",)]` — that corrupts live `io.load`
  source resolution. Use a throwaway `EngineRunner` with fake keys.

**Memo / streaming**

* Identity is the monotonic **`revision`** — never `id()`, never a content hash for lookup.
* A streaming provider's fp **must** fold declared calibration reads + field expression
  hashes, or the tile cache serves stale tiles.
* Provider fingerprints are **flat digest strings** — nested tuples blow up `_canon` on deep
  unrolled chains.
* **Construct a streaming provider LAST** in a compute; reads recorded after construction do
  not enter its fp.
* No `ctx.calib` / `ctx.meta` from inside a lazy closure (the frozen-`ReadContext` guard).
* `_canon` refuses object-dtype arrays — keep every stored array flat and fixed-width.

**Metadata**

* A `meta_transform`'s prediction and the compute's payload must agree **exactly** — match
  `span()`/bounds semantics including one-sided and out-of-range inputs.
* **Do not double-count a relative calibration change.** `ctx.calib` already returns the
  **post-transform** value, so a resample must *sync* the payload's pixel size to the env
  value, not re-divide by the scale.
* Declare **both halves** of any calibration change: the `meta_transform` (edit time) and the
  payload stamp (pull time).
* Never read `ctx.inputs[i].metadata[<calibration key>]` — that bypasses the fence and is a
  hard error under `strict_reads`. (Non-calibration provenance keys are the deliberate
  exception, §8.5.)

**Ingest / the `nd2` seam**

* **Never `import nd2` directly — go through `nodelab_v2.nd2_compat.import_nd2()`.** The SDK
  has bugs that make a *healthy* file unopenable, and they fire from `ND2File.sizes`, i.e. at
  file-pick time in the File menu, before a pixel is read. Live example (2026-08-03): a
  Z-stack configured in ND Acquisition but acquired at a **single plane** has
  `dZLow == dZHigh` and `dZStep == 0`, and `nd2 <= 0.11.3` divides by that zero range in
  `_calc_zstack_home_index` → `ZeroDivisionError` on a 6.08 GB file that is otherwise perfect.
  0.11.3 is the newest release, so there was nothing to upgrade to.
* **A shim wraps upstream and only acts on the raised exception** — it never reimplements the
  calculation. That is what makes patching a third-party parser safe: a file that loads today
  is bit-for-bit unaffected, and the shim self-sunsets when a fixed release stops raising.
  `nodegraph.selftest.test_nd2_zstack_home_index_guard` pins both halves.
* **The import stays lazy and in-function.** `ingest.py` must remain importable, and its TIFF
  half usable, with no SDK installed.
* **A garbage value that the FILE carries is not a shim's business.** The same file reports an
  `acquisition_start` of 7/31/2609 with a matching Julian day — the microscope PC's clock, not
  a parse error. It is passed through, so `frame_time_jd` stays honest and `view.overlay`
  refuses to place it rather than placing it wrongly. Relative `dt_s` is unaffected.

**Graph / sockets**

* Declare an auxiliary Dataset socket **after** `data`, so `dataset_preds[0]` stays the env
  source. (This was a latent `graph.py` bug: two-Dataset-socket nodes fed the wrong
  calibration env.)
* `AxisSizes` is frozen and comparable but **not iterable** — format it field by field.
* The engine walks inputs in **canonical socket order**; never rely on edge-connect order.

**Kernel ports** (recurring, from `nodegraph/kernels/*.md`)

* **`voxel_size_um` is ALWAYS `(dz, dy, dx)` slowest-first** = `(z_step_um, pixel_size_um,
  pixel_size_um)`. A `(dx,dy,dz)` swap silently corrupts anisotropy and every µm column.
* Bead/particle detect's **2D fallback puts z in column 0** (=0) with y,x in 1,2 — take
  `pts[:,1:3]` and stamp the true plane index.
* Histogram threshold is **2D-only, needs INTEGER input** and `voxel_size` as a **2-tuple**
  (it multiplies all elements; a 3-tuple folds z into area).
* Registration shifts are **`(row, col) = (y, x)`**; estimate on `ref_z = z//2` of the
  reference channel and apply the **same** bundle to every c,z (register-once/apply-all is
  what preserves colocalization).
* Granule kernels flip **`(z,y,x)` voxel → `(x,y,z)` micron** internally, driven by
  `voxel_size_um` — do **not** pre-convert points to µm.
* Vendoring is **verbatim**; the only sanctioned deviations are **version-compat fixes** (the
  al-dic ≥0.7 `gridxy_roi_range` fix, which must be set explicitly or the solver reports "no
  grid points generated"). A *semantic* preference is not a licence to edit a kernel — the
  node refuses instead (`ct_max_gap=0`).
* A dep-gated kernel is lazily imported **inside** the compute and raises a friendly
  `ImportError`.

**Determinism**

* Anything feeding the memo must be repeat-stable. Where a linker's row order affects only id
  *numbering*, a canonical `lexsort` + a renumber makes it total anyway. An unseeded RNG path
  (`st_use_prev_results`' POD-GPR warm start) is **pinned off**.

**Numba** (`wire-node-v2` §12)

* Reach for numba only when the hot path is a Python loop of many small array ops or a
  sequential/data-dependent algorithm. One big `ndimage`/`skimage`/`cv2`/FFT call already wins.
* **`cache=True` is mandatory** — per-tile streaming and Windows `spawn` process pools mean
  every worker re-JITs otherwise (~0.3–2 s each), which can go net-negative on short pulls.
* Don't stack redundant parallelism against a process pool.

**GUI**

* Deleting/re-wiring: `_endpoints` takes the drag anchor as an argument because `_end_wire`
  clears `_drag_fixed` before resolving the drop (that ordering crashed **every** mouse
  connect once).
* A `QComboBox` wheel over an unfocused combo must stay inert.
* `extra_layers` and `propagate_meta` run on **every keystroke** and must never raise —
  `propagate_meta`'s caller catches only `ValueError`, and even a caught error blanks every
  node's envelope graph-wide.

---

## 20. Gates, and how to add a node

```bash
PYTHONUTF8=1 python -m nodegraph.selftest                        # headless core → 55 groups
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png  # driven GUI probe
```

Heavier, not in the fast gate: `scripts/_ingest_nd2_smoke.py` (real ND2),
`_bench_provider_granularity.py` (the storage keystone), `_bench_ccl_watershed.py`,
`_bench_aldvc_profile.py`, `_bench_nms_numba.py`, `_nodelab_v2_shot.py` (one render).

Offscreen GUI gotchas: register Windows TTFs (else tofu), `os._exit(0)` to bypass the exit-5
teardown crash, `setParent(None)` before an offscreen grab.

### The node procedure

Read **`wire-node-v2`** (concepts) then follow **`build-node-v2`** (procedure). In short:

1. `register_node(compute, op_key=…, **spec)` in `nodes.py`.
2. Declare per-dim `granularity`/`kernel_axes` **honestly**.
3. Put a `unit` + `derive` on every spatial/temporal param; read calibration via `ctx.calib`
   and convert with `to_pixels_v2`; resolve per-channel params via `ctx.channel(c)`.
4. Add a `meta_transform` if it changes axes or calibration — and stamp the payload in
   lockstep.
5. Satisfy the socket contract (§7); use `ctx.layer(name)` for layer params; declare
   `reads_domains`/`adds_domains`/`layer_in`/`layer_out`/`extra_layers`.
6. Wrap a realized array in `ArrayProvider`, a Voxel layer via `with_layer`, structure via
   `with_structure`.
7. Land an end-to-end pull in `selftest.py` asserting: the footprint resolves per dim, 2D vs
   3D produce **distinct recipe hashes**, an axis-changing node's payload axes/calibration
   equal its `meta_transform` prediction, the right calibration key appears in
   `dict(engine.entry(node).reads)`, and structure lands on the right domain/layer.
8. Run both gates.

**Porting a vendored kernel:** the `.md` next to it is the authoritative contract — read it
first. The node owns the m/t/c loop; the kernel acts on one frame/volume.

### Working norms that produced this codebase

* **Grill before big design; present before locking.** Big autonomous decisions (representation
  choices, forks) are the user's.
* **Adversarial verification after building** — fan out review lenses, then verify each finding
  by attempting to *refute* it, and fix confirmed findings with regression guards. This has
  repeatedly caught real bugs the happy path missed: C1 staleness, GUI data loss, kernel-port
  hardening, and the four `track.objects` defects (all one class — *a live-looking control the
  selected path silently ignores*, which is what §7 clause 2 now forbids structurally).
* **Parallel subagents only when non-conflicting** — each owns one new file; the orchestrator
  does shared-file edits (`nodes.py` registration, `selftest.py` wiring) sequentially.
* **Never `pip install` unprompted; re-verify a backend's signature in-env before writing
  against it.**
* **Validate a vendored external solver against ITS OWN test suite, then distrust that
  suite's tolerances** (V2.18, `dic_correlate`/pyALDIC). Cloning upstream and replicating its
  synthetic cases as a repo-side bench (`scripts/dic_synthetic_bench.py`) is what turned
  "the node runs" into a number — and it settled three questions no amount of reading could:
  which of our defaults are inert (displacement smoothing below 1e-3), which are the real
  trade-off (subset size: 3.4× noise vs 5.3× resolution, at flat wall-clock), and that our
  adapter was already as accurate as a direct upstream call. Two traps that generalize:
  * **Upstream tolerances can be looser than the signal they test.** pyALDIC's shear and
    rotation strain cases assert against tolerances *exceeding twice* the quantity measured,
    so a fully sign-inverted strain cross term passed its suite for both of us. We found it
    only by checking an **antisymmetric** field (a rotation, where `∂u/∂y` and `∂v/∂x` must
    have opposite signs) against analytic truth. Pick discriminating fixtures, not just
    fixtures that pass.
  * **Upstream benchmarks may seed the answer.** Every pyALDIC synthetic case passes `U0=`
    the exact ground truth as the initial guess, so its quoted ~0.005 px is a
    converged-from-truth floor, not a field number. Read what the harness *gives* the solver
    before quoting what the solver achieves.
  * And **an audit's severity ratings need the same verification as its findings**: of the
    salvaged reports here, the "high severity, expose this param" item turned out to be
    largely self-healing in the library (the FFT search radius auto-expands and is remembered
    per reference), and the recommended `qfactor` fix was not implementable at all —
    `cc_max` never reaches `PipelineResult`. Both were caught by opening the source, not by
    re-reading the report.

---

## 21. Known gaps

Nothing is blocking; these are the honest edges.

* **No v1 importer.** `.nd2s_pipeline.json` files cannot be opened, by decision (2026-07-29).
* **Computed-provider pyramids** deferred (`StreamProvider.levels == 1`) with **two
  exceptions**, each allowed for its own reason and each keeping level 0 exact:
  `MultiViewProvider`, because a paste of downsampled tiles *is* the downsampled paste
  (§11 amendment), and `_AxisReduceProvider` (V2.20), because a z/t fold is *orthogonal* to
  an xy mean-pool — `sum`/`mean` commute with it exactly, `max`/`min`/`median` deviate only
  by the xy variation inside one `2**L` block and only in the smoother direction (measured
  on bead fixtures: ~0–1% of the display range mean, ≤8% peak at level 3, at point-source
  cores). Both are safe on the same invariant, verified repo-wide: the only callers that pass
  `level > 0` are the three display reads in `nodelab_v2.runner`. Every node compute,
  `realize`, the checkpoint writer and every export read level 0.
  Either way, a computed intermediate *upstream* still has no pyramid, so a stitch or a
  projection placed after an enhancement loses the coarse levels and its full-res display
  cost returns.
* **`plane_unit` is a cost contract, not an optimization** (V2.20). A level whose compute unit
  is a whole plane must say so, and a level that decomposes its output into tiles must ask its
  base for whole planes when the base says that. Getting this wrong is invisible in every
  correctness test — the pixels stay right — and shows up only as a multiplier: `util.zproject`
  over `util.stitch` cost `n_output_tiles × Z × n_positions` source-plane reads instead of
  `Z × n_positions`, i.e. ~350× on a 26×26-tile mosaic, which is why "Max Z onto a stitch"
  read as a hang. `test_zproject_over_stitch` asserts the read COUNT, not just the bytes.
* **Viewport detail-on-demand** (2026-07-31) closes the other half: the Viewer's overview is
  capped at `runner.MAX_DISPLAY_DIM` and zoom is a pure view transform over that texture, so
  the cap used to be the only resolution a mosaic ever got. Zooming now re-reads the visible
  rect off the GUI thread (`EngineRunner.request_detail` → `_DetailJob` → `detail_ready`) and
  both surfaces draw it over the overview (`SURFACE_CONTRACT`: `set_detail`/`clear_detail`/
  `visible_rect01`/`view_changed`). `MultiViewProvider` serves such a window by stitching
  ONLY the tiles it touches, which is what makes a level-0 patch affordable. Display-only by
  construction and asserted so (probe §V3). **Known gap:** the offscreen probe runs the CPU
  surface, so the GL second-quad draw is structurally checked but not pixel-verified.
* **`Engine._entry` is recursive**, papered over with `recursion_headroom` for deep unrolled
  chains; an iterative rewrite is a follow-up.
* **Some bridge hops still raise**: Track↔Timepoint (member×t) and Frame→structure broadcast —
  call `bridges` directly.
* **Structure-domain field transfer** is deferred (the field IR transfers lattice attributes
  only).
* **Stitching / multi-view fusion**: closed — `util.stitch` is the first `MULTI_VIEW` node
  (M→1 mosaic from the file's stage log, optional phase-correlation refinement, four blends;
  streams one canvas plane at a time via `MultiViewProvider`). Still open beneath it:
  *non-rigid* multi-view fusion (BigStitcher/multiview-stitcher-class deformable warping)
  and re-addressing a structure table across a stitch — `util.stitch` refuses a Dataset
  carrying one rather than leaving its objects at tile-grid coordinates.
* **No conditional-branch node** (`if_else`); zones and groups exist, branching does not.
* **Capability declared obsolete with v1** (`V2.05` §6): four enhancement extras (bleach,
  blob-subtract, spatial-flatness, temporal-fold), cell-tracker spatial maps, DIC mesh
  refinement, Cellpose nuclei. Vendored kernels + `.md` contracts for several of these still
  sit in `nodegraph/kernels/`, so a future port starts from the contract, not archaeology.
* **Per-axis `(y,x)` pixel size and `origin_um`** are deferred (`V2.00` §16); crop preserves
  pixel size and does not track an origin.
