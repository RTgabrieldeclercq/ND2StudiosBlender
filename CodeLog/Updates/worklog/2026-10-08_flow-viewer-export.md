# Flow viewer export + the composable scene viewer

- **Date:** 2026-10-08
- **Author:** hyper
- **Branch:** flow-viewer-export
- **Base:** e1ae0e28

## What changed

**1. Export Flow Viewer (3D HTML)** (`io.write_flow_viewer`, category io) writes a PIV / DIC
point velocity field as the lab's self-contained 3-D WebGL flow viewer: one `.html` that
opens in any browser with no server, the page deployed on mcgheelab.com for the
granular-channel data (streamlines as speed-scaled ribbons, animated flow particles, the
high-speed channel iso-surface, rough super-ellipsoid grains, the vasculature-analogy
skeleton, DTI-style bundles, view cube, presets, paper-figure mode, PNG snapshot,
`?embed=1`, the saved-view config API). A tap: the Dataset passes through plus a
`flow_viewer_report`. The maths are a Qt-free kernel, `nodegraph/kernels/flow_viewer.py`
(+ `.md`, + `flow_viewer_template.html` = the deployed page with its dataset strings turned
into placeholders): vectors from every position on one grid whose cell is the PIV window
pitch (positions by `stage_xy_um` as Stitch, with its `flip_x`/`flip_y`), timepoints
averaged per cell, the out-of-plane velocity from mass continuity (regularised least
squares / trapezoid / off) at the plane spacing the calibration gives
(`s = z_step_um / cell_um`, never a guessed z-stretch), PCHIP z upsampling, the five
geometry layers. Grains from a Voxel mask, from Otsu + watershed on an image channel, or
off; their size floor is a micron socket.

**2. The composable scene viewer (V4.10)** — six `scene.*` taps and one sink, so any
mixture of volumes and analyses from one experiment can be merged into one 3-D page from
the graph (`CodeLog/ClaudesPlan/V4.10_scene_viewer.md` is the design record):

- `scene.place` — where a stream sits in the µm world frame (stage layout or origin,
  Stitch's flip flags, an offset).
- `scene.volume` — an image channel as MIP / cloud / slices (one uint8 brick per timepoint,
  a WebGL2 3-D texture) or an iso-surface, under a voxel budget and a frame cap.
- `scene.vectors` — a PIV / DIC / DVC point field as glyphs, one frame per timepoint,
  optional streamlines.
- `scene.objects` — a Point or Label table as spheres sized by radius or `area`, coloured
  by a column, one frame per timepoint (cell density over time).
- `scene.tracks` — a Track table joined to its member table as polylines revealed along
  the timeline with a comet tail.
- `scene.series` — a table column reduced per timepoint as a chart with a cursor tied to
  the timeline (ELISA-bead intensity against time).
- `io.write_scene_viewer` — up to eight streams through a grow group; each layer converted
  to µm with its own input's calibration and stage log; one page with a layer panel,
  timeline, camera, µm axes, legends, chart, snapshot, embed mode and the saved-view API.

Each tap appends a small layer spec to the Dataset's `scene_layers` metadata, at edit time
(`meta_transform`) and at pull time from one shared function, and hands the payload through
untouched; the sink builds the geometry. Kernel `nodegraph/kernels/scene_viewer.py` (+ `.md`,
+ `scene_viewer_template.html`, a new WebGL2 page: ray-marched volumes, slice quads,
instanced glyphs and spheres, ribbon polylines, a 2-D chart canvas). The placement helpers
moved to `catalog/_shared/scene.py` so the flow viewer and the scene sink share one
mapping. New category `scene` (its own palette colour), role `scene_composition`.

**3. PIV kernel: a segfault fixed.** `analysis.piv` on the alia-chip data (position 17,
plane 9, windows 32→16, a 65 %-excluded pore ROI) killed the interpreter with exit 139 and
no traceback, inside `scipy.ndimage.map_coordinates`. `replace_outliers` leaves a flagged
vector NaN when every neighbour in its kernel is flagged too; `np.ma.filled(u, 0)` only
fills the mask, so the NaN reached `RectBivariateSpline`, every coefficient went NaN, the
whole deformation field was NaN, and `map_coordinates` casts a NaN coordinate to an
integer index without a finiteness check. `_predictor_field` (NaN → 0: "nothing measured
near here" = no shift) now feeds both the pair and the ensemble predictors, and
`_finite_deformation` sanitises the pixel shifts before any `map_coordinates` call.

Shipped with selftests (`test_write_flow_viewer`, `test_scene_viewer`), MANUAL §15 rows,
the roles, `graphs/piv_alia_chip_flow.nd2graph.json` (the alia-chip PIV workflow with the
flow viewer) and `graphs/alia_chip_scene.nd2graph.json` (the same PIV field + the granule
channel as a scene). Those graph files use `math.mask` and `plot.heatmap`, which exist on
the V4.00 branches but not yet on `Blender`; they are data and load once those merge.

## Why

The lab wants the interactive 3-D viewers hand-built for the paper
(`microfluidic-LLS-Paper/DataAnalysis/granular_flow_viewer`) to be an output of the node
graph — first the PIV page as-is (2026-10-05 alia-chip bead flow, 36 positions × 11 planes,
ensemble PIV at dt = 0.2 s), then, per the follow-up request, any composite of the lab's
volumetric and dynamic data: a bead volume with its DVC field, a cell channel's PIV /
objects / density / tracks on top, reporter beads' intensity against time. One exporter per
data kind cannot show two of them in one frame, so the second step is a scene: layer specs
ride the wire (the one slot that flows through every node without their knowledge, the
`__struct_zkind__` precedent) and the sink does the geometry, because a display mesh has
no honest home in the Dataset model. The flow viewer stays as the paper's specialised page.
`s` and every placement come from calibration and the stage log because the original
pipeline's single most uncertain number was a guessed plane spacing. The watershed floor
became a micron socket because pixel thresholds tuned to one mosaic split angular granules
into thousands of fragments on the chip. The PIV fix is in this branch because the saved
workflow runs that node on that data: without it the app dies on pull. Two pre-existing gate
failures on a clean `Blender` base were cleared on the way: CON-13 / WF-07 drift (re-read,
still hold, blessed) and the MANUAL §15 `util.chain` phantom (now plain text).

## Files

- `MANUAL.md` — §15 rows for the eight nodes
- `codemap/STATE.md`, `codemap/gen/*` — regenerated (`_codemap.py write`)
- `codemap/curated.lock.json` — CON-13 and WF-07 blessed (pre-existing drift, re-read)
- `codemap/node_roles.json` — `input_output` + the new `scene_composition` role (hotspot)
- `codemap/node_synopsis.json` — regenerated
- `nodegraph/catalog/__init__.py` — eight modules appended (hotspot)
- `nodegraph/selftest.py` — `test_write_flow_viewer`, `test_scene_viewer` (hotspot)
- `scripts/catalog_baseline.json` — re-saved, +8 ops (hotspot)
- `nodegraph/catalog/io/write_flow_viewer.py`, `nodegraph/catalog/io/write_scene_viewer.py` — the sinks
- `nodegraph/catalog/scene/{__init__,place,volume,vectors,objects,tracks,series}.py` — the taps
- `nodegraph/catalog/_shared/scene.py` — placement + layer-spec helpers
- `nodegraph/kernels/flow_viewer.{py,md}`, `flow_viewer_template.html`,
  `nodegraph/kernels/scene_viewer.{py,md}`, `scene_viewer_template.html`,
  `nodegraph/kernels/README.md` — the kernels, contracts, pages, index rows
- `nodegraph/kernels/piv_field.py` — `_predictor_field` / `_finite_deformation`
- `nodelab_v2/theme.py` — the `scene` category colour
- `CodeLog/ClaudesPlan/V4.10_scene_viewer.md` — the design record
- `graphs/piv_alia_chip_flow.nd2graph.json`, `graphs/alia_chip_scene.nd2graph.json`
- `CodeLog/Updates/worklog/2026-10-08_flow-viewer-export.md` — this entry

## How to verify

```
PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as s; s.test_write_flow_viewer(); s.test_scene_viewer()"
```
prints two `[ok]` lines. For the real thing: open either graph in the app (paths point at
`S:\alia chip flow data\`), pull the export node, open the page it writes. Pages produced
by the same nodes from the cached PIV results are at `S:\alia chip flow data\piv_results\
flow_viewer3d.html` (790 streamlines, 3240 grains, 13.5 MB) and `…\scene_viewer3d.html`.
Synthetic pages from both templates and the scene page itself were loaded in headless
Chrome with an error trap: no JS errors, a live WebGL2 context (the 13.5 MB flow page
exceeds SwiftShader — the software GPU process crashes — so check it in a real browser).
The PIV fix was reproduced on position 17 of the chip with the graph's parameters; every
plane now completes, `scripts/piv_synthetic_bench.py` still passes bit-for-bit.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every test `[ok]` except the known
  pre-existing `test_write_movie` failure (frame means not monotone; fails identically on
  the clean `e1ae0e28` base, see the 2026-10-07_graph-node-sets entry); the nine tests
  after it in `main()` were run in order from a runner script and all passed
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`: +8 ops, intended)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/piv_synthetic_bench.py` -> ALL PIV BENCH CASES PASSED (after the kernel fix)
- [x] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
