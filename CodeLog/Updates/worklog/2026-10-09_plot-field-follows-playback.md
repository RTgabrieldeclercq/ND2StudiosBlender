# Plot Field follows the Viewer's playback strip

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

**Plot Field no longer picks its own position, frame or plane.**
- **Removed.** Its `frame`, `position` and `plane` sockets and its `per` mode are gone.
- **The picture.** It now has the INPUT's M, T and Z axes, one view each. The Viewer's playback
  strip moves through position, frame and plane exactly as it does for the image.
- **Drawing.** A view is drawn the first time it is shown, and the last eight are kept. It
  takes about 0.2 s for a new view at the paper style's 1004 × 768 px.
- **Colour limits** are the whole table's 1st–99th percentile, so every view shares one scale.
- **Titles** name the view's position (its `position_name`, else `position k/M`), its frame
  (the clock's elapsed time, else `frame k/T`) and `plane k/Z`, each only when that axis has
  more than one.

**Mechanism** (`catalog/_shared/figure.py`):
- **`FigureProvider`** caches per `(m, t, z)` and has `spec_for(m, t, z)`. Per-frame plots are
  unchanged.
- **New `ViewFigureProvider`** takes a `spec_at(m, t, z, side)` function and an identity key.
  Plot Field's key covers the template, the drawing code, the input image's fingerprint, the
  pixel size, the frame interval and the field columns read.
- **Edit-time envelope and payload.** The new `figure_views` meta_transform and the
  `picture_views` payload builder agree through the shared `view_metadata`: the frame clock
  plus position names. So the card shows the real M / T / Z while you wire.

**Export Figure** (`io.write_figure`) gains Position and Plane sockets beside Frame. A writer
cannot follow the playback strip, so for a Plot Field figure these three pick the view written,
drawn at a 1024 px background instead of the screen's 512.

**Export Movie** (`io.write_movie`) refuses Z projection `max` or `mean` on a figure with
several planes: they would blend separate pictures into one. Its message points to `mid` or
Sweep `z`.

**Also updated:**
- MANUAL: the Plot Field and Export Figure rows, and §16 D4;
- concept card CON-28: a note on view pictures;
- the Plot Field demo: it now opens on frame 1 instead of setting a Frame param;
- the playground graph: its figure cards no longer pin frame, plane or position, and the page
  frames say which views to scrub to.

## Why

The user: "the plot field nodes should not control the frame, position, or plane that should
be controlled by the normal playback area".

**Why lazy views.**
- The old Plot Field drew ONE view chosen by its own sockets; `per = frame` added T only.
- Building every `(m, t, z)` view up front would have meant 396 entries on the chip-flow data
  (36 positions × 11 planes), each carrying a packed background, inside one JSON spec.
- Lazy views cost nothing until shown, and they reuse the provider path the Viewer already
  reads planes through (`computes_on_read`, so prefetch treats them as a compute).

**Why the key includes the input's data.** Two different inputs with identical settings must
never share a cached view.

## Files

- `nodegraph/catalog/_shared/figure.py`: `FigureProvider` per `(m, t, z)`;
  `ViewFigureProvider`; `picture_views`; `view_metadata`; `figure_views`.
- `nodegraph/catalog/plot/field.py`: views instead of sockets; `_view_key`; title stamps.
- `nodegraph/catalog/io/write_figure.py`: Position and Plane sockets; writes a view.
- `nodegraph/catalog/io/write_movie.py`: refuses max or mean on a multi-plane figure.
- `nodegraph/selftest.py`: `test_plot_field` rewritten for views.
- `scripts/make_field_playground.py`, `graphs/field_mechanics_playground.nd2graph.json`.
- `MANUAL.md`, `codemap/concepts.md` (CON-28), `codemap/node_demos.json`.
- Regenerated: `scripts/catalog_baseline.json`, `codemap/gen/*`, `codemap/STATE.md`,
  `codemap/node_synopsis.json`.

Shared files both developers edit: `nodegraph/selftest.py`, `codemap/gen/*`,
`scripts/catalog_baseline.json`.

## How to verify

1. Open `graphs/field_mechanics_playground.nd2graph.json`, page "3-D gel", and pull `fig_J`.
   The Viewer's strip shows 3 frames and 40 planes. Scrub to frame 3 / plane 21 for the cell's
   centre.
2. `test_plot_field` checks:
   - the picture's M, T and Z equal both the input's and the envelope's;
   - two frames give different views;
   - the title names position, frame and plane;
   - Export Figure writes view (position 1, frame 1, plane 2);
   - Write Movie refuses to max-project three planes.
3. Headlessly through `EngineRunner`, the GUI's runner (scratchpad `strip_check.py`): pulled
   at the strip's coordinates (0, 2, 20), (0, 2, 5) and (0, 0, 20), the returned planes equal
   the provider's views exactly, and moving the plane or the frame changes the picture.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: see the commit report.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`: see the commit report.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (baseline re-saved: plot.field
  lost `frame`, `position`, `plane` and `per`; io.write_figure gained `position` and `plane`;
  io.write_movie's Z-projection docs).
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed.
