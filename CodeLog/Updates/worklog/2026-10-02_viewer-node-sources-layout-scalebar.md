# Viewer node: growable image inputs, no output, layout modes, scale bar

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (built before the team-sync integration step; see the team-sync entry)
- **Base:** 6fbd719

## What changed

`view.viewer` is now a display sink. Its inputs are `Image` (the primary, which is the
payload the Viewer panel shows and describes) plus `Source 2` to `Source 6`. The extra slots
appear one at a time: the card always shows exactly one empty slot after the last wired
stream, so there is always an input to add and never a column of six. Each extra slot is a
`view_source` input that passes no domains downstream; the runner composites every wired one
onto the primary's frame as extra display channels named after the socket, through the same
path Overlay and the Voronoi node's `areas` input already use. The node has **no output**.

A `Layout` mode chooses how several streams are shown: `merged` (one composite, every stream
toggleable on the channel strip), `tiles` (one pane per stream side by side, shared cursor
and contrast), or `both` (the merged pane first, then the per-stream panes). The Viewer panel
builds the panes by grouping composed channels by their source label.

Four presentation-only sockets add a scale bar: `Scale bar` on/off (off by default),
`Bar length` in microns (0 = auto, a round 1-2-5 value near a fifth of the frame, the same
rule Export Movie uses), `Bar corner`, and `Bar colour`. The bar is drawn over the image in
widget pixels from the payload's `pixel_size_um` and omitted when the payload has none. It
follows the visible corner under zoom and pan. The window reads these from the document and
pushes them to the Viewer with each result; because they are `presentation` sockets they are
outside the recipe hash, so changing them repaints rather than re-runs.

Engine support: `SocketSpec.grow_group` (new field) names a group of inputs that reveal
themselves one at a time; `GraphDocument.input_specs` applies it (the document is the one
place that knows the wires) and `connect` validates against the unfiltered spec so a saved
graph or a script can wire a later slot directly, after which the filter reveals it and the
slots before it. `EngineRunner.overlay_chain` now collects every wired `view_source` socket
on a node in declaration order instead of the first only, and the strip label is resolved
per source.

Tests: `test_viewer_node_inputs` (Qt-free, through the document seam: spec shape, the reveal
rule including a skipped slot, presentation flags, payload = primary) and probe section V1
(two streams composited with `source_2:` labels, the layout modes producing 0 / 2 / 3 panes,
the scale bar's geometry inside the image, growth of the input column in the live document).
Manual row rewritten. Catalog baseline re-blessed (the new field appears on every socket).

## Why

The request: the Viewer should take an image, offer another input whenever one is filled so
there is always a slot free, have no output because its purpose is to display, offer tile /
merged / both viewing modes, and offer a scale bar with size and position options.

The extra streams are display sources rather than a channel merge because the engine is
one-payload-per-node and the Viewer's payload must stay the primary: measurements, hover
readouts and overlay tabs describe one Dataset, and the composite is a picture, not data.
Reusing the Overlay compositing path means the streams get the same per-source LUTs, channel
strip entries and Play-all pairing an Overlay gets, for free. Field-for-field placement is
the honest default for a tap; a user who needs stage placement already has the Overlay node.

The growable column is an engine-neutral field on the socket spec rather than GUI
convention, so the inspector, the card, the link-drag menu and the palette's overview all
agree on what is visible, and the synopsis records it. The scale bar is presentation because
a bar is about the window, not the result; a recipe-hashed bar would re-pull every node on
every toggle for no new pixels.

## Files

- `nodegraph/registry.py` — `SocketSpec.grow_group`, `InDataset(grow_group=)` (hotspot: engine)
- `nodelab_v2/ops.py` — the `view.viewer` definition and `_compute_viewer`
- `nodelab_v2/document.py` — `_grow_filter`, `connect` validates unfiltered
- `nodelab_v2/runner.py` — every `view_source` wire composited; per-source socket name
- `nodelab_v2/viewer.py` — `set_source_layout` / `_source_tiles`, `set_scalebar` / painter
- `nodelab_v2/window.py` — pushes layout and scale bar per result
- `nodegraph/selftest.py` — `test_viewer_node_inputs` (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — section V1
- `scripts/catalog_baseline.json` — re-blessed (hotspot, generated)
- `MANUAL.md` — Viewer row
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_viewer_node_inputs()"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # V1
```

In the GUI: drop a Viewer, wire an image into `Image`; a `Source 2` slot is there — wire a
second stream and `Source 3` appears. Set Layout to `tiles`; turn on Scale bar and pick a
corner.

## Gates

- [x] `test_viewer_node_inputs`, `test_registry`, `test_live_reload_contract`, `test_catalog_import_hygiene` pass.
- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole is still red at HEAD for the
  pre-existing reasons recorded in the team-sync entry.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`)
- [x] `python scripts/_codemap.py` -> gen/ current
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
