# V4.00 step 12 — math with units (Mask Math, Image Math, Math), and Crop by Region with an `outside` socket

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b874ffd (step 11i), on top of steps 11a–11i

## What changed

- **Units are a thing the engine knows** (`nodegraph/units.py`, new, Qt-free). A number on
  the wire — a Global scalar, a table column, a lattice layer — has a unit: the one its
  producer RECORDED (`ds.metadata["units"]`, written by `with_unit`; the Math cards and
  `analysis.reduce_scalar` now record what they produce) or, failing that, the one the
  catalog's naming convention says (`unit_of_column`: `area` is px² on a 2D table and voxels
  on a 3D one, `x`/`y` px, `t` frame, `*_um` µm, `*_intensity` counts, velocities µm/s).
  Unknown stays unknown. The algebra composes units (µm × µm = µm²), converts between them
  (`conversion_factor`: fixed factors; the CALIBRATION between px and µm, zpx and µm, frame
  and s), and refuses a conversion that needs a calibration the data lacks BY NAME, or two
  different dimensions as such.
- **Three Math nodes, category `math`, role "Math & units", offered on EVERY page.**
  `math.mask` (Mask Math): `subtract/union/intersect/xor/invert` on Voxel rasters; an EMPTY A
  is the whole frame, so `subtract` with B = the mask is the background in one card; B may
  come from a second wire and broadcast over m/t/z/c. `math.image` (Image Math): pixel
  arithmetic with another image (a projection / one channel / one timepoint of the same
  field) or a constant, lazy per plane; `bit_depth` follows the operation (`add` widens,
  `multiply`/`divide` drop, the rest keep — `metadata.image_math`). `math.values` (Math):
  arithmetic on the attributes of a domain lever — a column, a per-plane layer, a Global
  scalar — against another one, a Global scalar or a constant with a unit; like units for
  sums (B converted into A's unit), composed for products/ratios, `power`/`sqrt`/`log10` by
  dimensional analysis, `to_physical`/`to_pixels`; the result is auto-named
  (`total_intensity_per_area`, `area_physical`), predicted by the edit-time column catalog
  (`extra_layers`/`adds_columns`), and recorded with its unit.
- **Crop by Region (`util.crop_region`)**: crop to a mask, drawn regions or a label raster
  (on the wire or on a `regions` wire). `extent = frame` masks in place; `fit` shrinks to
  the box around everything inside (+ `margin`), restating `origin_um` at the corner; `each`
  cuts EVERY object out into a position of its own (`obj1…`, `B03_obj1…`), each in a
  common-size window at its own corner with its origin — "objects always keep their relative
  position" — so Split Positions fans them out and Stitch/Canvas put them back. Objects are
  connected components of the footprint over all frames, or label ids. `keep = outside` is
  the complement at the frame's extent; `fill` zero or NaN; the dim lever shrinks Z in 3D.
  Masks, Label/Point rows and per-M metadata follow the pixels. Edit time: `fit` → Y/X
  unknown, `each` → M too (`metadata.crop_region`).
- **The card's `outside` socket.** The engine is one-payload-per-node, so the Crop by
  Region card carries a synthetic second output (`document.output_specs`) and a wire from it
  is materialized into a SIBLING of the node with `keep` flipped, fed by copies of the crop's
  own wires (`ops.materialize_outside_taps`, first among the tap passes). One card, both
  outputs, each independently memoized.
- **Tests**: `test_units`, `test_math_nodes`, `test_crop_region` (incl. the GUI seam: the
  `outside` socket → sibling → envelopes and pulls). **Docs**: MANUAL §10b (new), §15 rows
  (a Utility row, a "Math & units" table), six §18 rows; codemap CON-24 (new);
  `node_roles.json` (role `arithmetic`, `util.crop_region` in geometric transformation,
  `op_pages`); catalog baseline re-blessed (104 → 108 ops); synopsis regenerated.

## Why

The user (2026-10-07): "I need a node set to do math on. to do math units need to be kept
track of. math should be able to be done in any node graph. I want to be able to do things
like take the area or volume of an image and subtract masks from it to make a new mask for
the background for example. I should also be able to create a mask and feed it into a crop
node. Or i should be able to use the draw regions node and feed the regions into the crop
node. the crop node outputs two images the cropped part and the inverse of the cropped part.
cropping does not need to be continuous, cropping can either keep the origional frame
extents or shrink the frame to fit all objects, or shrink to individual objects and output
those objects as an array. objects always keep their relative position in an image. regions
need not be square."

Design choices, and why: **units as metadata + convention** rather than a new field on
`AttributeLayer`/`StructureTable` — the catalog already encodes units in column names, no
engine type changes, and a record covers what a name cannot say; both travel with the
payload. **One Math node for every domain** (the `reduce_scalar` shape) rather than a Scalar
Math and a Column Math: the arithmetic is identical and only where the numbers live differs.
**The frame as an operand** (A empty) is how "the area of the image minus the masks" reads.
**Objects as positions** (M) for `each` rather than a batch: a position carries `origin_um`
and a name, which is exactly "keeps its relative position in the image", and the whole
position machinery (Split Positions, stream identity, Stitch, Canvas) works on it unchanged;
a batch means independent files. **A sibling node for `outside`** rather than a second
payload: the engine is one-payload-per-node by design, and the synthetic-socket → real-tap
pattern (`chK`, `posK`, `part:`) already exists; a sibling is an ordinary node to the memo and
the envelope pass. **Unknown units stay unknown**: calling an unlabeled column dimensionless
would make a wrong answer look right, the failure the whole module exists to prevent.

## Files

- `nodegraph/units.py` (new), `nodegraph/catalog/math/{__init__,mask,image,values}.py` (new),
  `nodegraph/catalog/util/crop_region.py` (new), `nodegraph/catalog/_shared/rasters.py` (new)
- `nodegraph/metadata.py` — `image_math`, `crop_region` meta transforms + registry entries
- `nodegraph/catalog/analysis/reduce_scalar.py` — records its scalar's unit
- `nodegraph/catalog/__init__.py` (hotspot: four MODULES appended)
- `nodegraph/selftest.py` (hotspot: tail of `main()` — three new tests + their registration)
- `nodelab_v2/ops.py` — `materialize_outside_taps` in `prepare_run_graph`, constants;
  `nodelab_v2/document.py` — `output_specs` adds `outside`; `_SYNTHETIC_SOCKET_RE`
- `codemap/node_roles.json`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json`
  (hotspot: re-blessed), `codemap/concepts.md` (CON-24), `codemap/curated.lock.json`,
  `codemap/gen/*`, `codemap/STATE.md`, `MANUAL.md`

## How to verify

`python run.py`, load an image, Threshold it. Drop **Mask Math** (any page): leave A empty,
B = `mask`, `subtract` → the `mask_math` layer is the background. Drop **Crop by Region**
after the threshold: `extent = fit` shrinks to the cells, `each` gives one position per cell
(Split Positions after it fans them out as `obj1…`), the card's `outside` socket wired to a
Viewer shows the field with the cells removed. Drop **Math**: On = label, A = `area`,
`to_physical` → `area_physical` in µm² appears in the Spreadsheet and in the next card's
column picker; A = `total_intensity`, B = `area`, `divide` → counts/px². Reduce → Scalar
(sum of area) then Math on Global `to_physical` → the total area in µm².

## Gates

- [x] selftest — 192 tests run (new: `test_units`, `test_math_nodes`, `test_crop_region`);
  only `test_write_movie` fails (pre-existing H.264 rounding). `main()` stops at that
  failure, so the tail was run with the scratch runner that continues past it
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 153 checks, run alone, exit 0 (no new
  section; the GUI seam of the `outside` socket is covered headless in `test_crop_region`,
  and an offscreen check built the four cards and their inspector panels)
- [x] catalog snapshot — re-blessed (104 → 108 ops), then CATALOG IDENTICAL; synopsis
  regenerated, SYNOPSIS CURRENT
- [x] codemap — CODEMAP CURRENT (CON-24 blessed; every node's `fp` moved because
  `nodegraph/metadata.py` is in every fingerprint — expected); WORKLOG OK
- [ ] sync check — run before the push
