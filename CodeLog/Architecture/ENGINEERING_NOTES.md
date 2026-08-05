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
| `reads_domains` / `reads_domains_by_mode` / `adds_domains` | domain rail, wire tint, validation | required vs produced domains; the `_by_mode` half states the requirement **per branch** |
| `layer_in` / `layer_out` / `extra_layers` | layer catalog + GUI picker | which named layers a node reads / writes |
| `layer_from` | GUI picker | which **Dataset input** a layer socket picks from (default: the primary) |

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
  The `reads_domains` half is checked in **every mode state the socket is active in**, against
  `spec.resolve_reads_domains(state)` — see **`reads_domains_by_mode`** below. The only
  exemption left is an **empty default**, which means the layer is optional (`analysis.segment`'s
  watershed `mask` falls back to cutting the image; `flow.iterate`'s `metric` to not scoring).
* Producers no socket can describe use **`NodeSpec.extra_layers(params, modes)`** — literal
  names (`drift_y`/`drift_x`), names derived from another param
  (`f"{labels}_boundary"`), or writes into the layer a *read* socket names (`measure`). It
  runs on **every keystroke**: it must never raise.

#### `reads_domains_by_mode` — the rail resolved per branch (V2.22)

`{mode: {value: frozenset(Domain)}}`, **unioned** onto the static `reads_domains` and resolved
by `spec.resolve_reads_domains(state)` / `missing_domains(incoming, state)`.

A node whose branches read different structure could previously state only one static answer,
so the catalog split into nodes that over-claimed (`analysis.object_field` declared
`{LABEL, POINT}` and demanded Points of a pure-Label graph) and nodes that declared
`frozenset()` and warned about nothing (`analysis.voronoi` needs a whole Label instance under
`bound=per_region`, and its rail said POINT — reported 2026-08-03). Both were silent: the rail
is advisory, so nothing failed, it just told you the wrong thing.

A **union over modes**, not the single-key `Mapping` that `granularity` uses, because the real
requirements are not keyed by one dropdown: `transform.transfer_structure` reads what
`from_domain` names AND what `to_domain` names, and `flow.iterate` needs a Global under
`preserve=best` **or** under `mode=feedback` — a disjunction (and the same limit that makes that
socket's own `available_in` approximate). An unlisted value contributes nothing, which is how a
branch says it requires no structure at all.

Registration refuses all three ways it can silently do nothing: a mode name that does not exist,
a value that mode cannot take, and a non-`Domain` entry. Eight nodes carry it; `analysis.segment`
deliberately stays static (its VOXEL is the image domain every source supplies, not a
per-method layer). Covered by `selftest::test_conditional_reads_domains`.

#### `layer_from` — a second Dataset input for a second DOMAIN (V2.22)

Declaring the requirement is not the same as being able to satisfy it. `analysis.voronoi`
needs a Point table AND a Label raster, and those come from different branches — dots detected
on one channel, areas segmented on another — which one wire cannot carry. Every other
multi-Dataset node uses its second input for *pixels* (`raw`, `reference`) or *display*
(`secondary`); none of them brings in a domain, and there is no merge/join node.

So `analysis.voronoi` grew an optional **`areas`** input (declared after `data`, so `data` stays
`dataset_preds[0]` = the calibration/domain env source), and the `region` layer socket declares
**`layer_from="areas"`** — because `document.layer_choices` follows the *primary* edge on
purpose, and without the declaration the picker would list names off a wire the compute never
reads. Unwired, both the compute and the picker fall back to the primary, so every pre-existing
single-wire graph is unaffected.

The geometry guard is now shared: **`_require_same_grid`** (`_shared/sampling.py`), lifted out
of `_intensity_provider` when the second consumer arrived — `AxisSizes` *plus* the `__sampling__`
provenance. An `areas` wire under `bound=frame` is refused rather than ignored (a Dataset input
is not `available_in`-gated: hiding a socket that has a wire would dangle the edge). The rail
composes for free, since `input_domains` unions every Dataset predecessor. Covered in
`selftest::test_label_to_points_voronoi`.

#### Axis-confined sampling stamps (2026-08-04)

A `__sampling__` stamp may declare the axes its effect is **confined** to, as a `"<axes>:"`
prefix — `CHANNEL_STAMP` (`"c:"`), `Z_STAMP` (`"z:"`). One rule follows: *a stamp confined to
axes that are singleton on both sides being compared cannot misalign anything.* The channel tap
was the first instance; the second came from a real graph and was a **false refusal** of correct
work — Voronoi seeds from a `zproject[max]` of one channel, areas from a single-plane
`crop[z5:6,y0:2048,x0:2048]` of another. Both end at `z == 1` and neither moves a `(y,x)`
address, so they address identical voxels, but strict stamp equality rejected them.

`util.zproject` always marks; `util.crop` marks **per call** (a pure z-crop is lateral identity;
a windowed one moves the corner) — that conditional is what the exemption rests on and is tested
against the real compute. An unmarked stamp is never dropped, so the parse fails safe.
`m`/`t` are deliberately not exempt: collapsing them picks a position or timepoint, and calling
frame 3 and frame 7 one grid is a content decision this guard cannot make for the user.

#### The card's layer picker (2026-08-04)

`inspector._layer_box` has offered a layer combo since V2.11, but the node CARD had none: a
`layer_in` pill fell through to the inline text editor, so on the canvas the only way to change
it was to know the name. A node holding its factory-default layer name against a wire that
carried another therefore failed at **pull** time, several nodes downstream, with the right
answer one click away. `node_item._open_layer_menu` closes it — the layers the wire actually
carries (via `layer_choices`, so `layer_from` is honoured), each with hover prose, plus a
`Type a name…` entry because the edit-time prediction is honest but incomplete. Driven in the
phase-5 probe via the `_PeekMenu` pattern.

#### A second input needs an OUTPUT trace (2026-08-04)

A node's payload is built on `dataset_preds[0]`, so it inherits the **primary** wire's layers and
nothing from the auxiliary one. For `raw`/`reference` that is right — those bring in pixels to
measure, not data to keep. For a second input that brings in a **domain** it is a hole: viewing
`analysis.voronoi` showed the *seeds'* branch labels and points and no trace of the areas the
cells were clipped to, so a territory could not be seen against the region that bounded it — and
the inherited raster is usually *also* named `labels`, so the overlay looked like the area layer
while showing a different branch's.

So the compute copies the area raster onto its output as **`f"{name}_areas"`**, declared through
`NodeSpec.extra_layers` (`_layers_voronoi`) — no socket can describe it, since the name derives
from another param and the layer is a copy of one a *read* socket named. Deliberately not under
its original name, which usually collides with what the seeds' branch already means by it. Empty
under `bound=frame`, which reads no area layer. **Rule for any future second-domain input: if the
node reads structure off an auxiliary wire, put what it used on the output, under a name of the
node's own choosing.**

#### A layer socket's DEFAULT is a guess about the graph (2026-08-04)

A `layer_in` socket must ship some default, and a literal one is a bet on which upstream node
you wired. Every such bet in the catalog was wrong for every producer but one, and it failed at
**pull** time with an error that printed the right answer one clause after refusing to use it:

```
ValueError: label to points: no Voxel layer 'labels' on the input Dataset (it carries
['CELLS']) — run analysis.segment / analysis.label upstream, or point the layer socket at
the right name.
```

Two nodes, one Label instance on the wire, nothing to decide. And renaming was not even needed
to reach it: the shipped defaults of sockets that are *routinely wired to each other* disagree —
`detect.particles` emits `particles` while `track.link`/`track.objects` ask for `spots`,
`analysis.tessellate` emits `mesh` while `analysis.voronoi`'s mesh is `voronoi_mesh`,
`analysis.dvc_field` emits `dvc` and nothing else does.

`catalog/_shared/labels._resolve_layer` had already fixed this for `analysis.voronoi`; it is now
the route for **every required input-layer socket** (17 nodes). Three-way rule, unchanged from
voronoi's: a name that IS on the wire wins literally; zero-or-stale with exactly ONE candidate
resolves to it and says so on the progress rail; **several candidates raise**, listing them —
it never chooses between them. So this can only widen what runs, and it replaces an error with
either an answer or a better error.

The candidate SET is per-node and is the interesting part, because it encodes what each node
actually needs:

| set | who | why |
|---|---|---|
| `_label_instances` (raster **and** table) | measure, label_to_points, transfer_structure's `via_label` | one row/dot per REGION — a bare mask would collapse the foreground to one |
| `_voxel_layers` (any raster) | edt, label, extract_boundary, boundary_band, tessellate's `label_surface`, track.link's label branch | groups by id, never reads a Label table, so a plain threshold mask is legal |
| `_label_tables` / `_point_layers` | track.objects, object_metrics, object_field | read the TABLE only; a raster dropped by an axis change is no obstacle |
| element tables only | rasterize_mesh | a mesh occupies three buckets (`m`, `m/vert`, `m/face`) and the strata are not separately selectable |

Two refinements the first cut got wrong:

* **`_resolve_label_instance`** hands back a name that is a Voxel layer but not a whole Label
  instance, unchanged, so `_label_raster` can give its own specific refusal ("carries no Label
  table, so its non-zero voxels are one undivided region"). Inferring past a name the user
  pointed at something real with would answer a question they did not ask and lose the one
  message that names the problem.
* **Optional sockets are excluded**, and a non-empty default is not proof of required. Kept out
  with their reasons in `selftest._LAYER_RESOLVE_EXEMPT`: `dic_correlate.roi` (unresolvable ⇒
  `roi6 = None` ⇒ correlate the whole frame — inference would silently MASK it) and
  `reduce_scalar.source` (names an attribute COLUMN, where on a structure domain every
  invariant coordinate is a "candidate", and an empty domain must stay a RESULT, not an error).

Gated structurally as **clause (g) of `test_param_socket_contract`**: any required `layer_in` /
`layer_in_mode` socket whose compute does not call a resolver fails the build. Three selftest
assertions that encoded the OLD contract were inverted with their reasoning recorded in place —
"a missing source layer must raise" was the defect, not the rule. Catalog declarations are
byte-identical (`_catalog_snapshot check` green): this is a compute-level change, so no saved
graph is affected and no memo key moves.

One known seam: `NodeSpec.extra_layers` sees params only, not the incoming layer catalog, so
`label_to_points`/`extract_boundary`/`accumulate_field` announce `<socket>_points` at edit time
while the run writes `<resolved>_points`. Harmless in practice — a downstream socket pointed at
the announced name resolves by the same only-candidate rule — and fixable only by widening the
`extra_layers` contract.

#### The Labels overlay had no layer selector (2026-08-04)

`viewer._label_plane` chose its raster by a heuristic — *the integer Voxel layer with the most
regions in the viewed plane* — and there was no control anywhere to override it. Several label
rasters on one Dataset is the normal case (Voronoi alone contributes three, and two segmentation
nodes both default their output name to `labels`), so the overlay drew whichever happened to be
the most fragmented. Reported as *"it pulls the wrong label — I select the segmentation from the
Red channel but it still shows the UV labels"*: there was nothing to select.

Now `LabelsOverlay.layer` (`""` = the old guess, kept as the fallback), resolved once in
**`viewer._label_source`** and shared by `_label_plane` (what is painted) and
`_label_layer_values` (what the size-probe counts) — they could otherwise disagree about which
layer is on screen. Exposed as a new `FieldSpec` kind `"layer"`: a combo the dialog repopulates
from the **live payload** on every `reload` (via an injected `layer_names` callable, so the
dialog stays constructible without a Viewer), Auto first, and a pick the current payload lacks is
kept visible as *"… — not on this payload"* rather than snapping silently back to Auto. A stale
pick falls back to the guess instead of drawing nothing. Driven in the phase-5 probe.

#### `view_source` — an auxiliary input's IMAGE reaches the Viewer (2026-08-04)

One payload, one image. So a node reading structure off a second branch could only ever display
the primary's channel — reported as *"I can only see the UV channel"* on a graph whose seeds came
from one channel and its areas from another. The compositing machinery already existed (it is what
`view.overlay` drives); nothing could tell it a socket other than `secondary` was a source.

`SocketSpec.view_source` says so. `EngineRunner.overlay_chain` now yields those edges alongside
`view.overlay`'s, and `_resolve_overlay` no longer bails when the payload carries no
`OVERLAY_KEY` recipe — because these nodes stamp none, and must not: display config in a payload
would ride the memo key, and `view.overlay` exists so that "how it looks" is a graph decision.
`_view_source_entry` synthesizes the placement through the same `plan_placement` +
`overlay_entry` pair a real Overlay uses, with **`on_unplaceable="index"`**: field-for-field at
scale 1, no stage log required (a sibling branch needs none and a TIFF never has one), and no
`flip_x=True` — that default is for a secondary from a different acquisition. The degrade path's
"alignment is NOT verified" warnings are replaced, since `_require_same_grid` already verified it.

**Opt-in per socket, deliberately.** `analysis.voronoi`'s `areas` qualifies; `measure`/`
label_to_points`' `raw` and DVC/DIC/`align_to`'s `reference` do not — those are a second version
of the *same* pixels or another timepoint of the same channel, and compositing them by default
would draw the field twice and read as a bug. The test that separates them is whether the node
reads a *domain* off the wire rather than intensities.

#### Two Dataset inputs, told apart (2026-08-04)

The painted domain rail is a **node-level** answer repeated beside every Dataset input, so
`analysis.voronoi`'s seeds wire and areas wire showed identical chips. Attributing the
requirement per wire is **not derivable**: `reads_domains_by_mode` says a `per_region` transfer
needs `{POINT, VOXEL, LABEL}`, and while `layer_from` maps the `region` socket's VOXEL onto the
areas wire, nothing maps LABEL there — a Label instance's table half is wanted by that branch but
no socket names it. Splitting the declaration per socket would give the node total and the
per-socket sets two sources of truth, which is the defect `reads_domains_by_mode` was written to
end. **So the chips stay node-level and the per-socket fact goes in the hover**, where it is
honest: each Dataset socket's tip names what *its own* edge carries, beside (not instead of) the
node's requirement, and says `nothing wired` when unplugged. `InDataset` also takes a
`description` now — a value socket has carried prose since V2.13, a Dataset socket could not, so
`areas`/`raw`/`secondary`/`reference` all hovered as bare names.

Found while testing it: `NodeItem.refresh()` only re-applied domain tips through `_layout`, which
it skips when the socket set is unchanged — so **every tip was frozen at the last relayout** and a
wiring change never updated it. That was invisible while the tips were node-level and constant.

**Still open:** `transform.transfer_structure` is the other node that combines two domains (its
`From` and `To` endpoints) and still has one input. It is NOT the same shape as
`analysis.voronoi`'s `areas` and should not be copied from it — see below.

#### Why `transfer_structure` did not get a second input

Its compute reads both endpoints off one `ds` across ~250 lines and fifteen pair-specific paths,
and the blocker is the **output contract**, not the plumbing: this node writes the transferred
column *onto the target table*. If the target lives on a second wire, the output — built on the
primary — has no such table to write to, so the target structure would have to be copied onto the
output wholesale. At which point the honest reading is that the **target** is what the result is
about and should be the primary input, with the *source* arriving auxiliary — the opposite of
`voronoi`, where the primary is unambiguously the seeds.

That is a design decision about the node's identity, not a mechanical extension, so it wants its
own grill rather than an analogy to `areas`.

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

### The checkpoint writer, and why it drifted (V2.26)

`nodegraph/checkpoint.py` is the Dock's persistence layer and it writes the **same** layout the
ingest does — manifest-last, `image/level_*.b2nd` planar-block pyramid, `voxel/*.npy` memmapped,
one `tables.npz`. It had its own copy of the geometry and the pyramid arithmetic, and both copies
had drifted. This section exists because there was no section: §12 documented the provider and
the ingest and never mentioned the second writer, which is plausibly *how* it drifted.

Three faults, in order of what they cost:

1. **The chunk was framed off Z alone.** `cz = min(ax.z, target // plane_bytes)` — right for a
   z-stack, and on a `z == 1` series it pins **one plane per chunk** however small the plane is.
   That is not an edge case for a Dock: a timelapse, a Z-projection, a stitched mosaic and a
   channel merge are all `z == 1`, which is to say everything the feature exists to freeze. A
   2048² uint16 plane is an 8 MiB chunk and writes at 143 MiB/s against 552 at cz=16.
   `_store_kwargs(..., batch_thin_z=True)` now batches over **T** until the chunk clears
   `_CHUNK_FLOOR_BYTES` (32 MiB — a floor to escape per-chunk overhead, not a target, since
   every byte past it costs full-plane read latency for nothing). Opt-in, so the ingest store's
   read profile — load-bearing for every live chain, and benchmarked at its current framing —
   is unchanged until someone measures it.
2. **The pyramid was a private copy of the mean-pool**, run serially on the calling thread
   through a `float64` intermediate, while `B2ndProvider._append_level` did the identical
   reduction pooled and chunk-aligned. `provider._plane_mean_2x`'s own docstring claims "every
   pyramid path routes its per-plane reduction through this function so … `nodegraph.checkpoint`
   cannot drift apart" — it could, because nothing enforced it and this module never called it.
   The coarse levels are now streamed out of the finished `level_0` by `_append_level`, so the
   bake's pyramid is bit-identical to an ingest store's *by construction*. `_append_level` had
   the mirror-image of fault 1: with `z == 1` it looped once per `(m,t,c)` over a single plane,
   paying a pool dispatch with a fan-out of **one** and writing a small chunk, 160 times for a
   160-plane series — 17.4 s against 3.3 s for all of level 0.
3. **`clevel` was never set**, so all three call sites inherited blosc2's dataclass default of
   **5** while its own docstring says 1. ~1.5× the write time for ~1% of the ratio.
   `pyramid_cparams()` is now the single definition and names `clevel=1` explicitly.

Measured end-to-end on a 1.25 GiB uint16 fixture, 160 planes, 3 levels: **thin-Z 23.0 s → 6.6 s
(3.5×), deep-Z 9.4 s → 5.7 s (1.7×)**. The same bytes took 2.4× longer as thin-Z than as deep-Z
before, which is the shape of fault 1.

**The fourth fault was not in the writer at all — it was that nothing asked what the input
WAS.** `_write_image` touched `prov` through `get_region` only, so a dock straight after
`io.load` decompressed a b2nd store and compressed it straight back, to produce bytes already
on disk. `_copy_level0` now copies the store's pyramid files when the destination bytes would
be identical: **12.4 s → 2.6 s** on a 1.25 GiB store.

Two things about it are worth keeping straight. First, **every guard is exact equality**,
because "close enough" here means silently wrong pixels — disk-backed `B2ndProvider`, axes
equal to the payload's, `target_dtype` a no-op at this precision (so integer data copies and a
real float conversion does not), `tile` equal, and the source level sound by both its marker
and the blank-chunk census. That last one is the guard that matters: copying is the only path
that would propagate a half-written source verbatim, and a marker-free interrupted write is
exactly what `open` trusts by age.

Second, **all sound levels are copied, not just level 0.** The first cut copied level 0 and
rebuilt the pyramid, which left the rebuild dominating — it decompresses all of level 0 again
to make coarse levels that were already beside it, and measured just 1.3× against a full
re-encode. Copying them is sound on `open`'s own terms (it takes the leading complete-or-legacy
run and drops a torn level, so `prov.levels` already excludes what it could not vouch for),
and each is put through the census here too. The copied levels keep the SOURCE's framing, which
differs from this writer's because an ingest store does not batch thin Z — a framing
difference, not a pixel one, and the manifest reports what is on disk (`chunks`) plus how many
levels came over verbatim (`copied_levels`) rather than what was requested.

Two correctness additions alongside: every level is `_mark`ed (`_LEVEL_META`) before its first
byte and again when complete — manifest-last made a torn *checkpoint* read as absent but could
not make a torn *level* read as torn, and `B2ndProvider.open` globs the leading run of level
files rather than trusting the manifest's count, so a re-bake producing fewer levels left a
stale higher level to be adopted as trusted-`legacy`. And the writer takes a `should_cancel`
token polled at each block boundary, returning `None` with no manifest — the state this module
already documents as correct for an interrupted bake.

### The `held` tier — a frozen dock that wrote nothing (V2.26)

`io.dock`'s `state` is `[live, held, docked]`. `held` cuts the upstream edge exactly as `docked`
does and serves the already-computed payload from memory. It is Cell-Tracker's *Set as raw data*
with the copy removed — and worth noting that Cell-Tracker's version is instant only because its
data model has **no M and no Z axis** (`nd2_loader` indexes position 0), so on the lab's
49-position file it keeps 1/49 of the acquisition and never says so. The latency was the target;
the technique was not available.

Everything it needs already existed, which is why it is small:

* **`Engine.seeds` is the right slot**, not the memo. `Memo._evict_to_budget` drops entries on
  recency alone and is documented as "always correctness-safe" precisely because dropping one
  only costs a recompute — but a held dock's upstream edge is **cut**, so an eviction would not
  cost a recompute, it would lose the data. Nothing evicts a seed.
* **`EvalContext.seed` was the one gap.** The engine's compute branch wins over its seed branch
  (`if fn is not None: payload = fn(ctx)`), so a node with both never saw its own seed. That was
  invisible while the only such node was `io.dock`, whose `docked` state re-opens a store from a
  path in its params and needs nothing from the engine.
* **`is_docked` had to split into two predicates.** It meant both "frozen" and "has a disk
  store". Every graph rewrite wants the first (`cut_docked_inputs`, `dormant_nodes`,
  `upstream_signature`'s walk stop) and every disk path wants the second. Conflating them is how
  a new frozen state ends up cut-but-still-evaluated, or evaluated-but-not-cut — which is worse,
  because the seed is then ignored and the chain the user froze runs anyway.
  `nodegraph/iterate.py` keeps its own duplicate of the state list on purpose (importing
  `nodelab_v2.ops` would invert the dependency); `selftest::test_iterate` asserts both states
  refuse, which is the only thing forcing the two copies to keep pace.

**The three honest limits, all stated on the card and in `choice_docs`:** it does not free
memory, it is uncounted by every budget, and it does not survive a reload. On reload the node
reports **`released`** in red and names the fix — never auto-re-held (that would re-run the
frozen chain on file open with no progress bar anyone asked for) and never silently downgraded
to `live` (which looks like it worked). Cell-Tracker ships the silent version of exactly this:
its baseline is never serialized, so after a reload the page renders empty with no message while
downstream pages still look loaded.

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

**The Iterate target picker (V2.22).** `flow.iterate`'s panel grows one "iterate on" dropdown
per variable slot, filled by `nodegraph.iterate.candidate_targets` — the active params and
Modes of every node upstream of `collect`, in `topo_order`. Two properties are the whole
design. (1) The menu is filtered by `_check_target`, the *same* predicate the rewrite refuses
on, so it can never offer a target the run would then reject; if you add a refusal there, the
menu narrows for free. (2) Choosing writes a **driver edge** through
`GraphDocument.set_iterate_target` and nothing else — the wire stays the single storage, so
the canvas, save/load, the "driven by" note on the target's own editor and `unroll` need to
know nothing about the control. The slot's `v{k}_type` follows the target (a Mode or a string
param is swept from the slot's STRING output) and the spare `+` row raises `variables`,
because both are settings the user would otherwise have to discover before the wire would
connect at all.

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
* **A detail patch has to carry EVERY channel the composite draws** (V2.23, reported
  2026-08-04 as "the overlay disappears if you zoom in"). The patch is drawn opaquely over the
  overview inside its rect, so a channel missing from it is not dimmed there, it is *erased* —
  and `view.overlay`'s composed channels live at indices above the payload's own, which
  `_plane_addrs` was **clamping** into the primary's range. Clamping is the shape of bug worth
  naming: it makes a request for something that does not exist look satisfied, so nothing
  anywhere reports a problem. The same clamp was dropping the overlay from every *warm* frame
  (composed planes now have cache slots of their own, so a re-visited frame keeps its
  secondary instead of blinking it off). Three consequences to hold onto:
  - the compositor takes the output's `region` of the primary field
    (`placement.sub_field_box`), so the overview and the patch go through **one** function.
    Two would drift, and the way they would drift is by placing the same overlay differently
    at two zoom levels — which reads as a registration error, not a rendering one;
  - its reader is offered the fractional **window** of the tile the picture needs
    (`placement.source_window`, the inverse of `axis_map`) and may return `(plane, window)`
    instead of a whole plane. Reading the whole plane decimated to `MAX_DISPLAY_DIM` was
    reading the *source's own overview*: with a stitched secondary the part covering one
    primary field is a few hundred of those pixels stretched over the display, which is
    exactly what "not the raw channel data" meant. Measured on the WellA3 pair with the
    7168² mosaic as the secondary: 1085 distinct values against 681 at the overview and
    120 against 47 at a 10 % zoom, and 0.02 s against 0.26 s — more data for less work,
    because the old read decoded 12.8 Mpx to sample 30 k of them;
  - the spatial comparators (`checkerboard`, `wipe`) are defined on the **image**, so the
    quad has to know where it sits: `u_rect` in the shader, `region=` in
    `composite_with_clim`. Off the quad's own uv a patch grew a fresh checkerboard and slid
    the divider to the middle of the zoom — a registration check that moves when you look
    closer is worse than none.
* **The display cap was a constant where it should have been a decision** (V2.23, asked
  2026-08-04: "I want the stitched and overlay to be full resolution and instantly
  loaded/playable"). `MAX_DISPLAY_DIM = 4096` pinned a 7168² stitched canvas to pyramid level 1 —
  half resolution everywhere outside a zoom patch — and the reason was nothing but the number.
  Measured before changing anything: the whole-canvas read is 0.78 s live and 0.06 s from a baked
  store, one frame narrows to 98 MiB, the whole 16-frame series is 1.53 GiB against a 64 GiB
  budget, and this GPU reports `GL_MAX_TEXTURE_SIZE` 32768. Nothing was ever protecting anything.
  It is now `display_cap(axes, texture_limit, bytes_per_px, planes)` — the decision NIS-Elements
  makes once per image against `MaxMemoryImageSize`, against the same 25%-of-RAM default
  (`DISPLAY_RAM_SHARE`, configurable because the answer is a property of the machine). Four
  things this taught, in descending order of how much they will bite again:
  - **a shape-deciding parameter must ride in the cache key.** The cap decides a plane's shape,
    and *three* writers put planes under that key (the displayed-frame decode, the prefetcher,
    the warm probe). Two of them disagreeing would serve a plane at the wrong size — so the cap
    is in the key, which turns any disagreement into a miss instead of a picture. The limit that
    feeds it also arrives *late* (a surface has to come up first), so this is not hypothetical.
  - **narrowing a display copy is free; converting for its own sake is not.** `util.stitch` fuses
    in float64 because a feather blend is a weighted mean, so a mosaic frame reached the display
    as 392 MiB carrying 12-bit data. The fix belongs inside `_fit_plane`, which *already*
    materializes a copy on both paths — riding that costs nothing, whereas a standalone
    `astype` pass would have cost as much as the read it was saving RAM for. Gated on the
    payload's own `bit_depth` (every node that makes values continuous drops it) and it
    **refuses** rather than wraps out-of-range values.
  - **level 0 is not display-only, so it must not be narrowed at the source.** Casting the fuse
    result would have been simpler and is wrong: every compute reads `get_region(0, …)`, and
    rounding a blend there moves a measured intensity behind the user's back. That is what the
    Dock's explicit `precision` is for.
  - **play is a licence the cursor does not grant.** `preload_series` reads the whole T range;
    `prefetch` deliberately will not (±2 frames on a computing provider — warming ±8 T of the
    WellA3 640 series once queued sixteen whole-volume deconvolutions). Pressing play is the user
    saying every frame is wanted in order. It stops at the budget rather than evicting its own
    head, because a preload that wraps the LRU re-decodes every lap.
* **A live mosaic has a floor no display trick reaches** (V2.23). Full resolution was a display
  decision; *cheap* is not — a stitched canvas is recomputed per displayed frame. NIS-Elements
  does not keep one live either (its Large Image mode carries the pyramid in the file and
  disables much of the processing menu), and the repo already had the artifact: a Dock's
  `write_checkpoint` writes a chunked 3-level pyramid. `Run → Flatten to Large Image…` is that
  path with the setup done — insert the Dock, pick `precision` off the propagated envelope's
  `bit_depth`, bake. 0.78 s → 0.06 s per full-resolution frame, read-ahead 2 → 8 (a store is not
  a `StreamProvider`), for 29 s and 0.97 GiB once.
  **The refusal is the interesting part.** Baking a chain containing a `display`-mode overlay
  wrote the primary alone: docking cuts in-edges in the run graph, so the chain walk finds no
  `secondary`, and `display` mode stores no pixels *by contract*. The overlay did not degrade, it
  vanished, while the graph still showed it — so the bake now offers to switch the node to
  `resample` and cancels if declined. Two ceilings remain, both documented in the manual rather
  than hidden: `resample` builds its output eagerly (6.1 GiB for 16 frames of this mosaic, ~77
  GiB for 200), and it reports no `bit_depth`, so its frames stay float32 and half as many fit.
* **`channel.merge`: two files on one channel axis, lazily** (V2.23, asked 2026-08-05 — "no need
  of an overlay, but instead a true metadata/image overlay … each channel can go through their
  individual Z frames since they are overlaid based on registration of metadata"). `view.overlay`
  could not be it: `display` mode changes nothing a node reads, and `resample` bakes ONE channel
  EAGERLY. The new node returns a real merged Dataset behind `ChannelMergeProvider` — the first
  provider that reads from two sources — at 0.025 s and +0 MiB against `resample`'s 25 s and
  +12.4 GiB for the same pair. Four things worth keeping:
  - **the meta_transform is shown only the PRIMARY's envelope**, and here BOTH grown axes depend
    on the secondary (`c` by its channel count, `z` by its focus range and step). So both are
    marked unknown, `stitch`-style, rather than guessed — and `z_step_um` is dropped with them,
    because leaving the primary's step standing would put every downstream µm→plane conversion
    on the wrong spacing. Overlay solves the same blindness by baking exactly one channel, which
    is right for something you measure one channel of and wrong for a merge.
  - **"union of both Z grids" is not literally representable.** The calibration vocabulary
    describes Z as origin + step + count, so an irregular list of focus positions has nowhere to
    live. The union is therefore the finest-step *uniform* grid spanning both — every acquired
    focus addressable to within half a step, collapsing to the finer stack's own grid when one
    range nests inside the other. Claiming a non-uniform grid in metadata that says uniform
    would be a lie about where the pixels are. On the real pair it is 297 planes rather than the
    640's own 210, because that file's twelve fields were each focused at a different height —
    correct, and worth reading the note for before assuming it is a bug.
  - **`primary` mode has to keep the primary's STEP, not just its range.** The first cut used the
    finer step for both modes, which silently multiplied the primary's plane count — the one
    thing that mode promises not to do. Caught by asserting the plane count, not the extent.
  - **the fixture step is 3 µm, not 2, on purpose.** At 2 µm the union's integer planes land
    exactly halfway between primary planes, where "nearest" is decided by round-half-to-even —
    a dead-band fixture tests the tie-break instead of the placement.
  Also: the recipe-entry builder moved to `catalog/_shared/placement_entry.py`, because the
  catalog forbids one node module importing another (importing executes it, registering that node
  early and welding the two fingerprints together) and two copies of a placement record is exactly
  the drift the single-builder rule exists to prevent.
* **Three co-registration defects found by reading the LIVE graph** (V2.23, 2026-08-05: "the 640
  channel still has no Z and looks like a lower quality"). Worth recording as a method as much as
  a fix: the saved workflow (`TFM_test_fullZ`) named the actual wiring, and one of its params —
  a hand-set `flip_x: False` — was the clue that cracked the third.
  - **"No Z" was not a bug.** `view.overlay`'s output axes ARE the primary's, and the primary was
    the stitched GFP mosaic at `z=1`. The 640 secondary carried 210 planes, none of them
    reachable, and the Z slider had one position. `channel.merge` is the fix; nothing in the
    overlay could have been.
  - **A MINIFIED secondary was point-sampled.** At 0.287 µm/px into the mosaic's 1.718, the 640
    lands 36 source pixels per output pixel and `axis_map` kept exactly one — the defect
    `_fit_plane` documents at length, and for once the *same* one: it discards `1 - 1/36` of the
    data while keeping the noise at full amplitude, so the output is noisier than the source. Now
    area-averaged when the source outnumbers the output samples by >1.5x, measured at 71% of the
    point-sampled noise at an unchanged mean. Magnification is deliberately left
    nearest-neighbour — "blocky is the honest rendering of real pixels stretched" depends on it.
  - **An already-stitched secondary must not be flipped again.** `flip_x`/`flip_y` describe how
    the camera is mounted, which no file records — but `util.stitch` had to answer that to place
    its tiles, so its canvas is already in stage coordinates. Applying the flip again mirrored the
    mosaic inside its own footprint: **784 µm, 456 px** out in x on the WellA3 pair. Derived from
    the sampling provenance (`stitch[stage…]`, and deliberately not `stitch[grid…]`, which makes
    no stage claim) in `_shared/placement_entry.handedness_for`, applied by the overlay, the merge
    AND the runner's display re-plan — a display path that flipped while the compute did not would
    draw the overlay 456 px from where a bake put it. The general lesson: a param that answers a
    question an upstream node has already answered is not a preference, it is inapplicable, and
    the honest form is to derive it and say so rather than ship a default the user finds by eye.
* **Per-channel display state now survives** (V2.23, same report). Contrast is keyed
  `(node_id, channel)` and was *also* being cleared on every node switch — redundant, since the
  key already namespaces it, and it threw away the one thing that cannot be recomputed: a window
  set by hand. And a composed overlay/merged channel is auto-enabled the first time it appears
  rather than on every pull, so one the user switched off stays off instead of its toggle button
  looking broken.
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
