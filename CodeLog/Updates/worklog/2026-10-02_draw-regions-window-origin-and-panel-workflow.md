# Draw Regions: shapes placed correctly under a troubleshooting window; Apply re-runs; panel in workflow order

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** team-sync-roles-palette-viewer
- **Base:** fd5c7a5 (origin/Blender) + the eleven commits of this branch

## What changed

**1. A shape drawn under a troubleshooting window now rasterizes where it was drawn.** The
runner stamps the window's top-left corner in source pixels onto a windowed seed
(`window_origin_px`, `[y0, x0]`, new `catalog/_shared/regions.py`), and every node that
rasterizes a `shapes` socket — Draw Regions, ROI Mask, Subtract Background's own sample —
shifts the stored full-frame shapes into the frame it is running on before rasterizing
(`shapes_in_frame`). Shapes outside the window rasterize to nothing, as they should.

**2. Apply re-runs the drawing node.** Committing a shapes pick now pulls the node whose
shapes changed, so the regions appear (as the Labels overlay's own layer) the moment Apply
is pressed. Before, the shapes landed in the param and nothing on screen changed.

**3. The Draw Regions panel reads as the workflow**, top to bottom in one section, *Draw
regions*: a numbered hint; the **Draw regions** button; then the shape (rect / ellipse /
circle / polygon / brush), **Add / Cut**, and the brush size — the node's presentation
params, moved here out of the general Parameters list; then Undo / Clear / Close polygon
with the live count; then **✓ Apply** / Cancel. Apply writes the regions into the node's
data (its Label layer + table), which is what the `Background regions` input of Subtract
Background, Measure, Filter Labels and the overlay consume.

Tests: `test_draw_regions` gained the window case for all three consumers (unit shift,
Draw Regions and ROI Mask rasters at window coordinates, Subtract Background still zeroes
the field from a shifted sample). Manual: Draw Regions row rewritten for the workflow and
the window rule.

## Why

Reported: "when I draw a region in troubleshooting mode with the ROI made smaller, the
shape I draw is not applied correctly" and "the drawing node does not work properly at
all; there should be a workflow in the panel: a Draw regions button, then select shapes,
add, subtract, then Apply; Apply saves the region into the data, which is fed as a region
to other nodes."

The first is a coordinate-frame bug: the viewer correctly stores shapes in full-frame
pixels (so a drawing survives the window being moved or cleared), but under the window the
compute runs on a `WindowView` whose frame starts at `(y0, x0)` — it had no way to know
that, so it rasterized full-frame numbers into a window-sized raster. The fix tells it,
through metadata on the seed rather than a new socket, because every shapes consumer needs
the same fact and none of them should have to be rewired to get it. The second was mostly
the missing re-pull: the pipeline did work, but a node whose only visible effect is an
overlay, and which is never re-run after Apply, looks dead. The panel reorder puts the
controls in the order they are used and keeps the tool settings beside the Draw button
instead of in a parameter list the eye has already passed.

## Files

- `nodegraph/catalog/_shared/regions.py` — new: `WINDOW_ORIGIN_KEY`, `window_origin_px`, `shapes_in_frame`
- `nodelab_v2/runner.py` — `_pin_frames` stamps the window origin
- `nodegraph/catalog/analysis/draw_regions.py`, `nodegraph/catalog/analysis/roi_mask.py`, `nodegraph/catalog/enhance/subtract_background.py` — shift shapes into the frame
- `nodelab_v2/window.py` — a committed shapes pick re-pulls its node
- `nodelab_v2/inspector.py` — *Draw regions* section in workflow order; tool params moved into it
- `nodegraph/selftest.py` — `test_draw_regions` window case (hotspot)
- `MANUAL.md`
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_draw_regions(); s.test_region_scope()"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png
```

In the GUI: F9, drag the amber box to a quarter of the frame; select a Draw Regions node,
press Draw regions, drag a rectangle inside the window, Apply. The region appears exactly
under where you drew it; clear the window (Run → Clear troubleshooting region) and it stays
at the same place in the full frame.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole: still red at HEAD for the
  pre-existing reasons in the team-sync entry. Run directly: `test_draw_regions`,
  `test_region_scope`, `test_subtract_background_zero_regions`, `test_registry`,
  `test_live_reload_contract` pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (91 ops, no spec change)
- [x] `python scripts/_codemap.py` -> gen/ CURRENT (the same 9 curated entries unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> on the pushed feature branch, level with origin
