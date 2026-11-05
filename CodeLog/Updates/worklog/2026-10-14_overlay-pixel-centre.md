# Overlay pixel centre

- **Date:** 2026-10-14
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

Every Viewer overlay drawn from coordinates (points, tracks, vectors, the Field wash and
contours, principal-axis glyphs, the deformed and element grids, subset windows, the
comparison rings, mesh sections, and label ids) now lands on the **centre** of the pixel it
names. Before this change it landed on the pixel's top-left corner, half an image pixel
up-left. They all go through one new conversion, `OverlayFrame.img_pt(x, y)` =
`map_pt((x + 0.5)·sx, (y + 0.5)·sy)`; nothing in `overlays.py` writes `map_pt(x * sx, …)`
any more. Label ids get the same half pixel on their own label-plane scale.

The hover readout and the two pick probes had the same seam and are fixed with it:

- The hover line now names the pixel whose value it prints. Before, it printed `int(pt)`
  but sampled `round(pt)`, so the right half of pixel 3 read "x 3" beside pixel 4's value.
  Its stage and µm figures also move by half a pixel, to the pixel-centre convention the
  stage arithmetic was already written in.
- The level eyedropper and the object-size click now read the pixel under the cursor
  (`floor`) instead of rounding to a neighbour.

## Why

Reported: "the centroids appearing as overlays don't seem to represent their true location
— a grow node from the centroid does not match the origin of the centroid overlay." The
engine is self-consistent. A centroid is a mean of voxel indices, and `transform.grow_points`
stamps its ellipsoid around that same index-space point (checked: a disc grown at (10, 13)
has a mean index of exactly (10, 13)). Both put a pixel's centre on the integer. The
displayed plane puts a pixel's edge there (pixel `i` is drawn over `[i, i+1)`, the pixmap
sits at the scene origin 1:1), and the renderer mapped one onto the other with no offset.
The grown disc, drawn as a label raster on pixel edges, was therefore correct; the marker
was off.

The fix sits in one conversion on the renderer, not in the engine. Shifting every table's
coordinates would move what every downstream measurement means, and the stage readout
(`(x − (W−1)/2)·psize`) already proves the engine's convention is pixel-centre.
`_pick_plane_pt` deliberately stays edge-on-the-integer: the crop rectangle's
`floor`/`ceil` to slice bounds is written for it, so only the hover converts.

**Not fixed here (follow-up):** drawn ROI shapes (`analysis.draw_regions`). The picker
stores their vertices in edge coordinates, and `_rasterize_region` rasterizes with
skimage's pixel-centre convention, so a drawn shape's mask lands half a pixel down-right of
the outline that was drawn. Fixing it changes the meaning of coordinates already saved in
graphs, so it is left as its own decision.

## Files

- `nodelab_v2/overlays.py` — `OverlayFrame.img_pt`; every structure-coordinate painter routed through it; label-id centroids +½ label pixel
- `nodelab_v2/viewer.py` — `_on_hover` converts to centre coords; `_hover_text` names the nearest centre; `_sample` floors `(pt + ½)·scale`; `_probe_intensity` / `_probe_object_size` floor; `_pick_plane_pt` documents its edge convention
- `scripts/_nodelab_v2_phase5_probe.py` — new O4b (marker and region fill share a centre at 10×); the P3 hover assertion now expects `x 120, y 100` at the exact field centre (the pixel `_v0` always read)
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

`QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 python -B -u scripts/_nodelab_v2_phase5_probe.py out.png`
prints `O4b: a point at its region's centroid draws on the region's centre`. With the old
mapping monkeypatched back in, the marker measured 5.5 px off the region centre at 10×.

In the app: Label to Points → Grow (points, small radius), view the Grow node with Points
and Labels overlays on, and zoom in. Each star's centre pixel now sits in the middle of its
disc.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (re-run on the full tree before the push, with every other 2026-10-13/14 change in it)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED, 170 `[ok]` incl. the new O4b and the updated P3. Exit 0 on the full-tree run before the push; two earlier runs exited 139 at `os._exit`, the pre-existing teardown fault recorded on the untouched base 889d9d4 in the 2026-10-13 entries
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (and `_node_synopsis.py` SYNOPSIS CURRENT)
- [x] `python scripts/_sync_check.py` -> 0 behind origin/Blender; PR #1 does not touch these files. Committed together with the seven other uncommitted 2026-10-13/14 changes, whose edits share `viewer.py`, the probe and the codemap
