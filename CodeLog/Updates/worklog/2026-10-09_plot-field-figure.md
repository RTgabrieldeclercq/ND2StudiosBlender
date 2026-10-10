# Plot field figure

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

New `plot.field` (Plot Field): a measured field drawn on its image as a figure (a PICTURE, like
every `plot.*` node).

**What it draws:**
- the image plane, auto-contrasted grey;
- a field column as a colour map, with a colour bar naming the column and its unit;
- optional arrows (auto length) and iso-contours;
- a µm scale bar.

**Behaviour:**
- Auto picks the most derived column.
- Limits default to the series' 1st–99th percentile.
- `per = frame` gives one figure per frame on one scale, so Export Movie can animate it.
- Export Figure writes PNG, SVG or PDF through the same spec. The spec carries the background as
  zlib + base64 (`figure.pack_plane` / `unpack_plane`).
- `figure.py` learns the `field` kind: drawing it, `frame_spec`, and `_finish`.

**Module move.** The Qt-free rasteriser behind the Viewer's Field and Vectors overlays moved from
`nodelab_v2/field_view.py` to `nodegraph/fieldmap.py`, and the Viewer, demo window, selftest and
probe now import it from there.

## Why

The request asked for "a visual output capability" for the fields and their derived quantities.
The Viewer overlay (previous commit) covers interactive viewing; this gives an exportable figure
and movie.

**Why the module moved.** It is the same rasteriser the Viewer uses, so a figure and the screen
cannot disagree, and a catalog node may not import the GUI package (the seam rule in CLAUDE.md).
The rasteriser therefore moved into the engine, as `nodegraph/placement.py` serves both
`view.overlay` and the Viewer.

**Why the scale bar reads `ds.metadata`.** It reads the input's `pixel_size_um` from metadata
rather than `ctx.calib`: a plot's own envelope deliberately carries no calibration (a chart has no
µm, and the Viewer must not draw a scale bar on it). The input's identity already keys the memo.

## Files

- `MANUAL.md`
- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/node_demos.json`
- `codemap/node_roles.json`
- `codemap/node_synopsis.json`
- `nodegraph/catalog/__init__.py`
- `nodegraph/catalog/_shared/figure.py`
- `nodegraph/fieldmap.py`
- `nodegraph/selftest.py`
- `nodelab_v2/demo_recipes.py`
- `nodelab_v2/overlays.py`
- `nodelab_v2/viewer.py`
- `scripts/_nodelab_v2_phase5_probe.py`
- `scripts/catalog_baseline.json`
- `nodegraph/catalog/plot/field.py`

## How to verify

`test_plot_field` checks:
- the picture is the style's size and matches the envelope;
- Auto picks `strain_eq` on the series range;
- arrows are 2 px for 1 µm at 0.5 µm/px;
- the scale bar comes from the input's pixel size;
- per-frame gives 2 figures on one scale;
- PNG and SVG are written;
- an unknown column is refused.

The Plot Field demo window draws the speckle pair's strain.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (working tree, exit 0)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (baseline saved: plot.field added)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->

Focused selftests run on this tree, all passing:
- every `test_plot_*` and `test_write_figure`;
- `test_field_view`, `test_socket_docs`, `test_option_docs`, `test_param_socket_contract`,
  `test_column_catalog_complete`, `test_codemap`, `test_node_demos`, `test_catalog_import_hygiene`.

**Snapshot gates for `3a919ff` (the Viewer commit),** run in a worktree:
- **GUI probe:** passed, exit 0.
- **Rest of the selftest:** passed.
- **Full selftest:** stopped earlier than usual, at `test_lablink`, with "the beat thread outlived
  the command it was reporting on". That run overlapped other test processes. The test passes
  3 of 3 when run alone; the module is untouched here; the cause is a timing race (the beat thread
  is stopped, not joined, before the 0.3 s silence check).

Hotspots touched: `nodegraph/selftest.py`, `nodegraph/catalog/__init__.py`, `codemap/gen/*`,
`scripts/catalog_baseline.json`.

Also new here: `test_field_end_to_end` runs the Field cards on real correlations of known
deformations.

| Input | Chain | Result |
|---|---|---|
| speckle image warped by F = I + A (strains of 1 %) | `analysis.piv` → Field Gradient → Green-Lagrange Strain → Volume Change | median gradient within 1.2e-4 of A; Green-Lagrange strain within about 1e-4 of the analytic E; J 1.0019 (true 1.0019) |
| speckle volume dilated by 1.02 | `analysis.dvc_field` → Field Gradient → Volume Change → neo-Hookean Stress (E 5 kPa, ν 0.45) | J 1.0620 (true 1.0612); pressure −946 Pa, which is tension, close to −K·ln J ≈ −990 Pa; von Mises 6.8 Pa, about zero for a pure dilation |

The test takes 3.8 s.
