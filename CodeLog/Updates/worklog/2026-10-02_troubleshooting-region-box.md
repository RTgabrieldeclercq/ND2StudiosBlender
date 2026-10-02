# Troubleshooting mode: a draggable region box scopes the run to part of the frame

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (built before the team-sync integration step; see the team-sync entry)
- **Base:** 6fbd719

## What changed

While troubleshooting mode (F9) is on, the Viewer shows an amber rectangle with eight handles
that frames the whole image. Dragging an edge or corner resizes it, dragging the interior
moves it, and on release the viewed node re-runs on that window of the **source** frame only.
The payload is the window, so the Viewer then shows just that region with the box around it;
dragging the box past the image's edge grows the window again. The status chip and the
Viewer's status line name the window (`SOLO t4·64×64px`, `region 64×64@y32,x16`), a new
**Run → Clear troubleshooting region** action resets it, and the region combines with the
existing M/T/Z picks as one more axis of the same scope.

Implementation, by layer:

- `nodelab_v2/region_box.py` (new, Qt-free): the window as `(y0, y1, x0, x1)` in source
  pixels; `clamp` (clips into the source, floors at 8 px, and returns `None` for a
  whole-frame window), `hit` (corner, then edge, then interior), `drag` (a move slides and
  stops at the frame edge; a side never crosses its opposite), `label`.
- `nodelab_v2/runner.py`: the pin grows a fourth element, the region. `_pin_frames` wraps the
  frame subset in a `WindowView` (the same lazy translated view `util.crop` returns), shrinks
  the envelope's Y/X and moves `origin_um` by the cut. New `set_region` / `region` on the
  runner; a region change invalidates held views like a pick change does. The window rides
  in the seed provider's fingerprint, so results are keyed per window and a whole-frame
  window is the identity the pre-region pin had.
- `nodelab_v2/document.py`: `source_scope_extent(node_id)` — the `(Y, X)` of the source
  roots, the same walk as `source_scope_totals`.
- `nodelab_v2/viewer.py`: region state, `set_region_extent` / `set_region` / `clear_region`
  / `region`, a `region_changed` signal, the painter (`_paint_region`, under the pick
  gesture), the mouse handling (`_region_event`, consuming only the box's own handles so a
  click elsewhere still pans), an extrapolating widget-to-source mapping so edges can be
  dragged into the margin, resize cursors on hover, and the status-line note.
- `nodelab_v2/window.py`: wires the signal, pushes the region to the runner on F9 and on
  node change, ranges the box to the source extent in `_sync_solo`, names the window in the
  chip and the scope phrase, and adds the Clear action.

Tests: `test_region_scope` in the selftest (the maths plus a `WindowView` over a frame
subset serving exactly the window of each picked frame and keying apart from the unwindowed
subset); probe section T1b in the GUI probe (box extent, a 64×64 window pulls a 64×64 payload
whose interior equals the full frame's slice, chip and status text, clear restores the full
frame). Manual: the troubleshooting section and the key table.

## Why

The request: pick a region of the image in troubleshooting mode and run the node on only
that subset, with a draggable yellow boundary that fills the frame at first. Tuning a
parameter on a 2048² field still costs a 2048² compute per frame even with one frame picked;
a window makes the loop interactive on the part of the field that matters.

It is a scope applied at the source seed, not a Crop node inserted into the graph, for the
same three reasons the frame picks are: the catalog is untouched (every node simply sees a
smaller frame and the eager computes allocate for it), the user's graph, run plan and saved
file are untouched, and the memo stays honest because the window rides in the seed
fingerprint. A whole-frame window is stored as `None` so that arming the box never re-keys
results the user already has.

Because the payload is the window, the box always frames the whole displayed image and the
drag changes where that window sits in the source. That is why the viewer needs a mapping
that extrapolates past the image: growing the window means dragging an edge into the margin.
The region is expressed in source pixels and the box spans the source extent, so a chain
with a Crop, Resample or Stitch upstream of the viewed node still selects a window of the
source; the manual says so.

Kernels clip their halos at the window's edge, exactly as below a Crop node, so a windowed
result's outermost halo of pixels differs from a full run's. The probe compares the interior
and the manual states the caveat: the box is for checking parameters, not producing results.

## Files

- `nodelab_v2/region_box.py` (new)
- `nodelab_v2/runner.py`, `nodelab_v2/viewer.py`, `nodelab_v2/window.py`, `nodelab_v2/document.py`
- `nodegraph/selftest.py` — `test_region_scope`, registered in `main()` (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — section T1b
- `MANUAL.md` — troubleshooting section, key table
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_region_scope()"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # T1b
```

In the GUI: load a file, view a node, press F9. Drag the amber box's corner inward; the node
re-runs on that window and the Viewer shows it. Drag an edge outward into the margin to grow
it. Run → Clear troubleshooting region to reset.

## Gates

- [x] `test_region_scope` passes; `test_solo_frame_scope` and `test_frame_subset_scope` still pass.
- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole is still red at HEAD for the
  pre-existing reasons recorded in the team-sync entry. Not touched by this change.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no catalog change)
- [x] `python scripts/_codemap.py` -> gen/ current
