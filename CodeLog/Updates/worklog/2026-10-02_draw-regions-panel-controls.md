# Draw Regions: controls in the node's panel; region-wanting nodes draw in line and return

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** team-sync-roles-palette-viewer
- **Base:** fd5c7a5 (origin/Blender) + the nine commits of this branch

## What changed

**Nothing about a drawn region is configured on the image any more.** The viewer's pick
bar (its tool / Add / Cut / Invert / Clear / Undo strip and its Apply) is hidden for a
drawing whose request says `tools_in_panel`; the viewer only takes the mouse and shows the
live shape. Draw Regions gained three **presentation** sockets — `Tool` (rect / ellipse /
circle / polygon / brush), `Operation` (add / cut) and `Brush size` px — ordinary node
settings in the Parameters section, outside the recipe hash, never read by the compute,
read live by the gesture (the inspector emits `sync` when one changes, the window pushes
them into the armed session). Below Parameters a **Draw** section hosts the rest: **Draw**
(arm), then Undo / Clear / Close polygon, a live readout ("1 shape · rect", or "3 shapes
on 2 frames" at rest from the socket's JSON), **✓ Apply** and Cancel. ROI Mask, which also
IS a drawing, gets the same section.

**A node that wants a region goes and draws it on a Draw Regions node in line, then comes
back.** On Subtract Background the `Background sample` row's button now reads `Draw
Background sample on a Draw Regions node…` (the card's ◎ takes the same route): the window
drops a Draw Regions node on the `regions` input — or uses the one already wired there —
fed from the same image, **selects it, pulls it so its image is on screen, and arms its
drawing** with the controls in its panel. **Apply (or Enter) writes the stamped shapes,
re-selects the original node and pulls it**, so the corrected result is what you see;
Cancel / Esc returns too. Declared, not hardcoded: the route is driven by
`readiness.EMPTY_PICKS` — the same table the *Ready to run* block uses — so the node that
wants a region and the node that supplies it are named in one place.

New viewer surface for the window to relay to: `set_pick_tool` / `set_pick_op` /
`set_pick_brush`, `pick_action` (undo / clear / close / invert), `apply_pick`,
`pick_node_id`, `pick_shape_count`, and a `pick_readout_changed` signal. New inspector
signals `draw_control(node_id, what, value)` and `region_requested(node_id, socket)`, and
`set_draw_state(node_id, armed, readout)` for the window to drive the section. New window
handlers `_arm_draw` (arms at once when the viewer already shows the node, else pulls and
arms when the result lands), `_sync_draw_session`, `_on_draw_control`,
`_on_region_requested`, `_return_from_draw`, `_select_only`. `PickRequest.tools_in_panel`.

Manual: Draw Regions row, Subtract Background row, the §8b gesture table row. Catalog
baseline re-blessed (three new sockets). Probe section RD2 covers the whole round trip
with a real mouse drag on the surface.

## Why

The user: "I don't want the options for the drawing to be on the image viewer. I want all
the options for drawing or defining parameters to be within the Draw Regions node. If I am
in another node that asks for a region to be input, drop a Draw Regions node in line to
the input and switch to the Draw Regions node; when I hit Apply go back to the node we
were working on."

The tool/op/brush are presentation sockets rather than Modes or plain params because they
configure the gesture, not the result: a Mode would re-key the memo and re-run the node on
every tool change, and a plain param would be recorded as a read the compute never makes
(the socket-contract gate would rightly refuse it). Presentation sockets are exactly the
"GUI reads it live, the recipe ignores it" class the Viewer's scale bar already uses. The
return is deferred by one event-loop turn because the viewer disarms before it commits
(Enter → `pick_armed(False)` → `pick_committed`): returning synchronously would pull the
original node against the shapes it had before Apply. The ROI Mask drawing moved into the
panel too, for one reason only: two drawing surfaces for one gesture would be the drift the
viewer's old bar and this panel are now guaranteed not to have.

## Files

- `nodegraph/catalog/analysis/draw_regions.py` — `tool` / `op` / `brush_px` presentation sockets
- `nodelab_v2/picker.py` — `PickRequest.tools_in_panel`
- `nodelab_v2/viewer.py` — bar hidden for panel-hosted picks; public setters; `pick_readout_changed`
- `nodelab_v2/inspector.py` — region button, Draw section, `draw_control` / `region_requested` / `set_draw_state`
- `nodelab_v2/window.py` — `_arm_draw`, `_sync_draw_session`, `_on_draw_control`, `_on_region_requested`, `_return_from_draw`, `_select_only`; `_arm_pick` routes shapes picks
- `scripts/_nodelab_v2_phase5_probe.py` — RD2
- `MANUAL.md`
- `scripts/catalog_baseline.json` (hotspot, generated), `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

```
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # RD2
```

In the GUI: Subtract Background, Approach `zero_regions`, press `Draw Background sample on a
Draw Regions node…`. A Draw Regions node appears wired into `Background regions`, the
panel switches to it and the image is live; set Tool in the panel, drag on the image, press
✓ Apply in the panel. You are back on Subtract Background, re-run with the sample.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole: still red at HEAD for the
  pre-existing reasons in the team-sync entry. Run directly: `test_registry`,
  `test_live_reload_contract`, `test_catalog_import_hygiene`, `test_socket_docs`,
  `test_option_docs`, `test_draw_regions`, `test_readiness` pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`, 91 ops)
- [x] `python scripts/_codemap.py` -> gen/ CURRENT (the same 9 curated entries unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> on the pushed feature branch, level with origin
