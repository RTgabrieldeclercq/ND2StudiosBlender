# pyALDIC mesh refinement, in the single DIC node and as a staged Refine Mesh card

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** f3c7e3d

## What changed

1. **The single DIC node can refine** (`analysis.dic_correlate`).
   - New sockets: `refine_holes`, `refine_roi_edge` and `refine_brush` (an optional Voxel mask,
     its t=0 plane). The existing `winsize_min` is the minimum element size.
   - Each (m, z) line's spec goes to `run_pyaldic_series(refinement=…)`. The kernel's
     `_refinement_policy` was already plumbed, but no node passed it.
   - Off by default, so existing graphs are unchanged.
2. **New staged card `dic.refine_mesh` (DIC Refine Mesh)**, placed in `bundle:dic` between
   Initial Guess and Local Match. That is where pyALDIC refines, after the seed
   (`pipeline.py:1374-1404`).
   - **What it does:** it splits the elements at ROI holes (default on), the ROI edge or a
     brush mask, down to `min_element_size` (0 = DIC Grid's). It interpolates each frame's seed
     onto the new nodes and REPLACES the lattice table and the `dic_mesh` grid with the refined
     ones.
   - **New kernel entry point** `dic_stage_refine(ref, mesh, U0, refinement)`.
   - **How the later stages follow:** refinement depends only on the mesh, the mask and the
     criteria, so Local Match, Global Step and Strain re-derive the identical refined mesh
     from the stamped `dic_refine` (`_dic_unit_mesh`). They take each line's node count from
     that mesh. pyALDIC's mesh object is never carried down the wire.
   - With nothing on, or no ROI, the card passes the data through.
3. **Docs and gates.** Roles and demos list the card; MANUAL §15 has rows for the card and the
   single node's refinement sockets. CON-25 now says six DIC cards. The staged module's
   docstring and `dic_correlate.md` are updated. `test_analysis_mesh` gains the refinement
   checks, and `test_dic_stages` expects the six-card bundle.

## Why

The user's grid request read "including any mesh refinement". In planning, refinement turned
out to be implemented in pyALDIC but unreachable from any node. The user then chose
"Single node + staged card".

Measured on a 160² speckle pair with a 22 px ROI hole, 24 px subsets, 16 px step, min 4 px:
- **Single node:** 60 → 125 elements at levels 0-2 (16 / 8 / 4 px).
- **Staged chain** (Grid → Seed → Refine → Local → Global):
  - the SAME 125 elements;
  - displacements equal to 3e-15 µm (median) on every node the single node solved, at most
    0.002 µm at a few by the hole edge, where the seed is interpolated onto the new nodes;
  - the single node blanks its 52 in-hole nodes, where the staged chain keeps neighbour-filled
    values (unchanged staged behaviour).

The re-derive design was chosen over serialising the mesh into the table. A table would lose
the hole-edge marks and boundary sets pyALDIC's solver uses. Re-deriving is exact and cheap
next to a correlation.

## Files

- `nodegraph/kernels/dic_correlate.py` + `.md` (`dic_stage_refine`)
- `nodegraph/catalog/analysis/dic_correlate.py` (refinement sockets + spec)
- `nodegraph/catalog/analysis/dic_stages.py` (`dic.refine_mesh`, `_dic_refine_spec`,
  `_dic_unit_mesh`, later stages on the refined mesh, `bundle:dic` six cards)
- `MANUAL.md` §15; `codemap/concepts.md` (CON-25), `codemap/node_roles.json`,
  `codemap/node_demos.json`
- **Hotspots:** `nodegraph/selftest.py` (refinement in `test_analysis_mesh`, six-card bundle in
  `test_dic_stages`), `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`,
  `scripts/catalog_baseline.json` (re-blessed: +`dic.refine_mesh`, the single node's 3 sockets)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_analysis_mesh(); s.test_dic_stages()"`
- In the app: give DIC Grid an ROI mask with a hole, add the DIC bundle and pull Local Match.
  The Grid overlay shows smaller elements around the hole, coloured by level.

## Gates

- [x] Focused selftests: codemap, node demos, socket contract / docs / option docs, column
      catalog, import hygiene, DIC catalog, store, quick scope, mesh (with refinement), the
      three staged chains, catalog, registry. Full snapshot run: next entry
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
      PROBES PASSED ("3 bundle rows … each hold their 15 stage nodes"). One run segfaulted in
      the final `os._exit` after the pass line (the known teardown flake); the re-run exited 0
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (131 ops; re-blessed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
- f3c7e3d on its snapshot:
  - selftest: 186 ok up to the pre-existing `test_write_movie` failure;
  - the 38 tests after it passed.
