# Draw Regions node + Subtract Background `regions` input; Ready-to-run block; region locator map

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** team-sync-roles-palette-viewer
- **Base:** fd5c7a5 (origin/Blender) + the eight commits of this branch

## What changed

**1. A new node, `analysis.draw_regions` (Draw Regions).** Hand-drawn labelled regions —
rectangle, circle, ellipse, closed polygon, freehand, with Add and Cut — each **pinned to
the frame (m, t, z) it was drawn on**. The viewer stamps every finished shape with the
frame being viewed when the pick is committed (and shifts a shape drawn inside a
troubleshooting window back into full-frame coordinates). Under `scope = drawn_frame` a
shape appears on its frame only and an unstamped shape on every frame; `all_frames` puts
every shape everywhere. Output: a Label raster (one id per region, global-unique across
frames, draw order; a Cut carves out of the regions before it; Clear resets) plus the
invariant Label table (`id, m, t, c, area, z, y, x`), so the Labels overlay draws them with
ids, Measure reports on them, and Subtract Background samples them.

**2. Subtract Background gained an optional `Background regions` Dataset input** (visible
under `zero_regions` / `sampled_region`). Wire a Draw Regions node into it, fed from the
same image: every non-zero pixel is the background sample, **per frame** — a frame with no
patch of its own uses the union of everything drawn anywhere — and the wire wins over the
socket's own `Background sample` drawing. A wired Dataset with nothing drawn, or of a
different (Y, X), is refused with the fix named. The shared per-unit mapper
`_map_image` gained a `with_coords=True` form (`plane_fn(a, m, t, z, c)` /
`volume_fn(v, m, t, c)`) so a compute whose answer depends on which frame it is on can know;
the default one-argument form is unchanged for every other caller.

**3. The inspector's *Ready to run* block** (`nodelab_v2/readiness.py`, Qt-free, plus the
panel section). Directly under the node's title: a green tick when every input is present;
otherwise one red block per problem with the input in question painted red below (the
parameter row or the `in · data` connection line), and **`+ Add <node>`** buttons for the
nodes that would supply the gap. Four checks, edit-time only: unwired primary image input
(the only problem reported when it holds; offers Load when the graph has no source); a
read domain nothing upstream produces (producers ranked, the usual one first — Connected
Components for Label, Threshold for a mask, Spot Detection for points, Link Tracks for
tracks); a drawn-shape socket the node refuses empty (declared per (op, socket) in
`EMPTY_PICKS`: Subtract Background's sample → Draw Regions into `regions`; Draw Regions'
own shapes); the 3D lever on one-plane data. Pressing a button adds the node **and wires
it**: inserted on the primary wire when it feeds `data`, or fed from the same image and
wired into the side input. `HIDDEN_OP_PREFIXES` moved from `scene.py` (Qt) to `ops.py`
(Qt-free) so the ranking can skip hidden ops; `scene` still exports it.

**4. The troubleshooting region's locator map.** Once the amber box is narrower than the
frame, the Viewer paints a small map top-left: the whole source field — its picture as
captured from the live surface at the moment the window was first narrowed (overlays
suppressed for the grab; GL via `grabFramebuffer`, CPU via `grab`), or a plain panel if
the capture failed — with everything outside the window dimmed, the window in amber, a
dashed candidate while dragging, and a caption with the field's size. It clears when the
box is back to the whole frame and the picture is dropped when the field changes.

Manual: Draw Regions row, Subtract Background row, §8 *Ready to run* paragraph, §6 locator
map bullet. Roles: `analysis.draw_regions` under Segmentation & labeling. Catalog baseline
re-blessed (91 ops). Codemap and synopsis regenerated (the `fp` of every node importing
`_shared/map_image.py` changed with it, as the dependency digest should).

## Why

The user reported three things: Subtract Background "doesn't work", and wanted a node to
draw labelled regions specific to an (x, y) location and frame as the sample for it; the
Properties panel should check whether a node has every input it needs, highlight what is
missing in red and suggest the nodes that go with it; and in troubleshooting mode, a map
top-left showing where the cropped region sits in the full field.

On the first: the `zero_regions` compute passed its tests, so "doesn't work" was the
experience — switch the approach and the node refuses at once ("no background region is
drawn") with the only route a hidden Pick button on a socket; nothing on the panel said
what to do. Draw Regions is the proper input: a node of its own that is visible, per
frame, measurable, and offered by the readiness block at the exact moment the sample is
empty. Frame pinning is done at commit time in the viewer because only the viewer knows
the cursor and the window at that moment; the stamp is an extra JSON key that consumers
without a use for it (ROI Mask, the socket drawing) ignore. The regions input takes any
mask, not only Draw Regions, because "these pixels are background" has no reason to be
tied to how the mask was made. The union fallback for an undrawn frame is the honest
reading of a per-frame annotation on a series: it never invents a sample and never
refuses a frame the user simply did not annotate.

Readiness is a Qt-free module with an explicit `EMPTY_PICKS` table rather than an inferred
rule, because whether an empty drawing means "whole frame" or "nothing to compare against"
is each node's own semantics and guessing it would be wrong for one of them. The panel
reuses the document's existing `missing_domains` so the block and the card's red chip can
never disagree. The locator map's picture is a grab of the live surface rather than a
re-read of the planes so it shows what the user was looking at, with their LUTs, for one
grab; a failure falls back to a schematic because a wrong locator is worse than a plain one.

## Files

- `nodegraph/catalog/analysis/draw_regions.py` — the new node (hotspot: `nodegraph/catalog/__init__.py` gained its import line)
- `nodegraph/catalog/enhance/subtract_background.py` — `regions` input, `_regions_lookup`, per-frame sampling
- `nodegraph/catalog/_shared/map_image.py` — `with_coords`
- `nodelab_v2/readiness.py` — new, Qt-free
- `nodelab_v2/inspector.py` — *Ready to run* section, red highlights, `add_requested`
- `nodelab_v2/window.py` — `_on_add_requested` (insert on the wire / feed a side input)
- `nodelab_v2/viewer.py` — `_stamp_shapes`, `_capture_region_thumb`, `_paint_region_map`
- `nodelab_v2/ops.py`, `nodelab_v2/scene.py` — `HIDDEN_OP_PREFIXES` moved
- `nodegraph/selftest.py` — `test_draw_regions`, `test_readiness` (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — T1b locator map + stamping, RD1 readiness
- `codemap/node_roles.json`, `MANUAL.md`
- `scripts/catalog_baseline.json` (hotspot, generated), `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_draw_regions(); s.test_readiness()"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # T1b, RD1
```

In the GUI: drop Subtract Background, set Approach to `zero_regions`: the panel's *Ready to
run* block says the sample is empty and offers `+ Draw Regions`; press it, the node appears
wired; pull the Draw Regions node, Pick → draw a patch of background on frame 0, step to
frame 3, draw another; pull Subtract Background. Press F9, drag the amber box smaller: the
locator map appears top-left.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole: still red at HEAD for the
  pre-existing reasons recorded in the team-sync entry (`test_codemap` aborts on 9 stale
  curated entries; `test_write_movie` codec rounding). The tests after the abort were run
  directly with it skipped: all pass except that one, including the two new tests.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`, 91 ops)
- [x] `python scripts/_codemap.py` -> gen/ CURRENT (the same 9 curated entries unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> on the pushed feature branch, level with origin
