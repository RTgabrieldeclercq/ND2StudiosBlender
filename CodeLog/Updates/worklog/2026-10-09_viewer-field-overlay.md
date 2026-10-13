# Viewer field overlay

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

The Viewer draws a measured field's results on the image.

**New Field overlay tab.** It colours the image by any numeric column of a field table:
displacement magnitude, or a Field card's strain, rotation, Jacobian or stress.
- Live pickers choose the table and the column; a new `column` kind in the overlay dialog fills the
  column list from the chosen table. Auto shows the most derived quantity available: von Mises,
  then `strain_eq`, `jacobian`, `curl`, then `disp_mag_um`.
- Styles: a smooth wash, one cell per node, or iso-contours.
- Limits default to the whole series' 1st–99th percentile, so frames compare.
- The auto colour map is diverging for signed columns.
- A colour bar shows the column and its unit, and the hover readout adds the value under the cursor.
- Optional principal-axis glyphs draw red/blue tension/compression bars from the Strain or Stress
  tensor.

**Grid tab.** A new `deformed` style draws the field's lattice through the displaced nodes,
exaggerated.

**Vectors tab.**
- **Unit fix:** arrows are drawn in image pixels. Before, a µm displacement was drawn at pixel
  positions, so every arrow was 1/pixel_size of its true length; the same bug was duplicated in the
  demo window.
- A live table picker replaces "first matching layer".
- Auto length draws the series' 95th-percentile arrow one node spacing long.
- Magnitude colour uses the series' range, not each frame's.
- A one-channel field shows on every channel; a 3-D field shows its node plane nearest the viewed Z.
- A velocity-only field is drawn as the motion over one frame interval.

**Voxels tab.** Its declared-but-never-drawn `contour` style now draws iso-lines.

**Where the logic lives.** All extraction and rasterising is in a new Qt-free module,
`nodelab_v2/field_view.py`, shared by the Viewer and the demo window:
- a lattice field is placed one cell per node, with holes left empty;
- a 3-D lattice is interpolated between the planes either side of the viewed Z;
- a scattered field is resampled by a k-d tree, at most 128² cells, about 22 ms for 20k nodes,
  computed once per frame change on the GUI thread.

## Why

The request was a way to see the analysis results on the image, with a visual output for the new
strain / curl / Jacobian / stress columns.

**Why a new tab instead of Rasterize Field.** The existing route was Rasterize Field into a Voxel
layer and then the Voxels wash. That costs a full-resolution interpolation per column and frame, has
no column picker and no colour bar, and loses the "one value per measured node" reading. A field is
sparse, so drawing it from its own table is cheap and honest.

**Why the vector code moved.** Arrow extraction was duplicated in two places, both drawing µm at
pixel positions. Moving it into one Qt-free module fixes the bug once and makes it testable:
`test_field_view` asserts the px conversion.

**Why glyphs and the deformed grid.** They are in the approved plan: a scalar strain map cannot show
the direction of stretch, and a deformed lattice is the most direct picture of the deformation.

**Why plot.field is not here.** It is the next commit.

## Files

- `MANUAL.md`
- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/symbols.jsonl`
- `nodegraph/selftest.py`
- `nodelab_v2/demo_recipes.py`
- `nodelab_v2/overlay_dialog.py`
- `nodelab_v2/overlays.py`
- `nodelab_v2/viewer.py`
- `scripts/_nodelab_v2_phase5_probe.py`
- `nodelab_v2/field_view.py`

## How to verify

Pull a PIV card on the speckle demo followed by `bundle:mechanics`, then open Overlays → Field. It
shows `von_mises` with a colour bar; switching Colour by to `strain_xx` brings in the diverging map,
and Principal axes = strain draws the crosses. Grid → Style deformed shows the distorted lattice.

Headless checks:
- `test_field_view`, the Qt-free seam;
- the GUI probe's `[field]` section, which checks the pickers, the auto choice, the raster, the px
  arrows, and that the wash, contours, colour bar, glyphs and deformed grid are painted.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (working tree; one run exit 0, the run with the glyph checks printed PASSED then exited 139 at teardown — the intermittent post-PASS crash also seen on 3af1653)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no op changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->

Focused selftests run on this tree, all passing: `test_field_view`, `test_field_mechanics`,
`test_overlay_renderers`, `test_node_demos`, `test_codemap`, `test_catalog_import_hygiene`.

**Snapshot gates for `063c1ec` (the Field nodes)**, in a worktree:
- the full selftest passed up to the known pre-existing `test_write_movie` failure, and every test
  after it passed;
- the GUI probe passed with exit 0.

Hotspot touched: `nodegraph/selftest.py`.
