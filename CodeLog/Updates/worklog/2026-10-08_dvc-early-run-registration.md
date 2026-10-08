# Register-then-DVC graph for the CRC early run, and the Viewer draws DVC displacement arrows

- **Date:** 2026-10-08
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** 30a1d4c

## What changed

1. **The Viewer's arrow overlay now reads DVC's displacement columns.** `Viewer._VECTOR_COLUMNS`
   gains `("disp_y", "disp_x")` — the pair `analysis.dvc_field` writes (µm) — after PIV/DIC's
   `(u, v)`, their µm twins and tracking's `(dy, dx)`. `demo_recipes.py` already listed it; the
   Viewer did not, so a pulled DVC field drew no arrows at all.
2. **A graph for the dataset:** `graphs/dvc_crc_early_run.nd2graph.json` =
   `io.load` (`…20260901_023728_582_early run\Channel640_Seq0001.nd2`, 11 timepoints × 96
   positions × 71 planes) → `util.select_position(0)` → `util.crop` in Frames mode (empty =
   keep all; type `t0-2` for a quick look) → `registration.stabilize` (3D, translation,
   reference = first, low-pass 1.5 px / high-pass 2 px, upsample 20) → `analysis.dvc_field`
   (pyALDVC via `al-dvc` 1.2.0, fixed frame 0, subset 32 / 24 along Z, spacing 32 / 8 along Z,
   prefilter σ 1.0, global shift on, subset stride 2, 4 ADMM iterations, all cores).
3. `al-dvc` 1.2.0 (the official pyALDVC, github.com/zachtong/pyALDVC) is now installed in the
   project venv; it was absent, so the DVC node could only raise its install hint.

## Why

The user's call after the Bead Finder work: this field is too dense for centroids to be the
representation, so go to DVC — but the series "drifts a lot" and has to be registered first.
Measured on position 0 by phase correlation of each frame against frame 0 (2-D max
projections at full resolution and 3-D at quarter resolution agree): the stage walks
diagonally, −18/+27 px at t=1, −78/+78 px (−22/+22 µm) at t=5, saturating at −124/+98 px
(−36/+28 µm) by t=10, with no measurable axial drift. That is an order of magnitude beyond the
DVC seed search (8 voxels at the coarsest level), so the drift is removed by the Registration
node first — 3-D so an axial drift would be caught too, translation because the stage, not the
gel, is what moved — and the DVC's own `global_shift` then has nothing left to do. Why the
Viewer change is in this commit: the whole point is to SEE the field over the data, and
without it the Vectors overlay silently drew nothing for a DVC table.

Result of the 3-frame headless test (position 0, t0–t2, pulled through the GUI's
EngineRunner from the same document): Registration took 207 s (three 71-plane direct ND2
reads + 3-D correlation) and wrote `drift_y` = 0 / −18.4 / −34.65 px, `drift_x` = 0 / 27.15 /
42.3 px, `drift_z` = 0 / 0.55 / −1.3 planes, confidence 1.0 / 0.89 / 0.87 — within a quarter
pixel of the independent probe. DVC (CPU numba kernels; no CUDA device on this box) took
119 s for the two pairs on a 31 × 31 × 5 node grid (4805 nodes per frame, node planes z = 19
… 51). Residual field after registration, t=2 vs t=0: `disp_x` p10/50/90 = −0.18 / −0.04 /
+0.11 µm, `disp_y` −0.20 / −0.06 / +0.08 µm, `disp_z` −0.74 / −0.15 / +0.21 µm, qfactor
median 0.97 with 314 of 4805 nodes under 0.5 (the dark cell cluster mid-field, where there
are no beads to correlate, and the border strip the shift uncovered); median strain ~1e-4.
The lateral field is smooth and organised around that cluster — a deformation, not noise.

## Files

- `nodelab_v2/viewer.py` — `_VECTOR_COLUMNS` (one line + comment)
- `graphs/dvc_crc_early_run.nd2graph.json` — the dataset's graph
- `codemap/gen/*`, `codemap/STATE.md` — regenerated (hotspots)
- (environment, not a file) `.venv`: `pip install al-dvc` → 1.2.0, with its pyvista/vtk/h5py deps

## How to verify

- Open `graphs/dvc_crc_early_run.nd2graph.json`, set the Crop node's Frames to `t0-2`, pull
  DVC (pyALDVC); the Viewer's Vectors overlay draws the displacement arrows over the
  registered frames; the Registration card's `drift_y`/`drift_x` Frame layers show the
  removed stage walk.
- Headless: the scratch runner used here pulls `reg` then `dvc` through `EngineRunner` and
  prints the drift layers and per-frame median displacement / qfactor.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> 178 `[ok]` up to `test_write_movie`, which fails on
      this branch's base too (pre-existing, movie writing); the 38 tests after it run in order by a
      one-off runner and all pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
