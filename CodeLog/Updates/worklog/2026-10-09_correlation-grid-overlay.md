# The correlation grid on the image: DIC, PIV and DVC write their mesh; the Viewer's Grid tab draws it

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** ef96337

## What changed

1. **One mesh table shape for all three analyses.** New `catalog/_shared/analysis_mesh.py`:
   - Every element here is an axis-aligned box in the reference configuration, so one table
     fits all three: invariant schema (centre) + `y_lo/y_hi/x_lo/x_hi/z_lo/z_hi` + `size_px` +
     `level` (refinement depth), and for DIC `n0…n7` (pyALDIC's Q8 node indices).
   - It sits on its own Point layer, `<field>_mesh`, marked in `__analysis_mesh__` with its
     kind and subset / window size.
   - `t = -1` means shared by every frame of its unit. A mesh is written once per position,
     not once per frame.
2. **Producers:**
   - **`analysis.dic_correlate`:** pyALDIC's own elements. The kernel now returns each frame's
     mesh (`"mesh"` in the series result; contract updated).
   - **`analysis.piv`:** the final pass's window lattice; cells touching an excluded window
     drop.
   - **`analysis.dvc_field`:** the hex grid; hexes touching an unsolved node drop.
   - **`dic.grid`:** the uniform lattice's cells.
   - **`piv.first_pass` / `piv.deform_pass`:** the pass's lattice. A deform pass REPLACES the
     grid with its finer one; `with_mesh` always drops the old table first.
   - **`dvc.grid`:** the hexes.

   Each producer declares the layer (`extra_layers`) and its columns (`adds_columns`). The
   single nodes now also stamp their window size (`dic_winsize` / `dic_winstepsize`,
   `piv_window`, `dvc_subset_size(_z)`).
3. **Viewer Grid tab** (`nodelab_v2/overlays.py`, `viewer.py`).
   - `GridOverlay` with style `elements` | `subsets` (the window around every Nth field node),
     colour by refinement `level` | `flat`, colour, width, every, opacity.
   - `_grid_rects()` keeps the viewed position and frame (`t = -1` included) and plane. A 2-D
     element needs its plane index to match; a DVC hex is kept when its z extent spans the
     plane.
   - `_paint_grid` draws one cosmetic-pen path per colour, under the vectors.
4. **Grids are kept out of field logic.**
   - The Points overlay skips marked grid layers, which hold element centres.
   - `_shared.labels._point_layers` excludes them too, so a lone field still auto-resolves in
     every consumer that picks "the" Point table.
5. **Load Analysis** restores a trial's grid with its mark and window. A trial records each
   table's columns, so Load Analysis can declare `adds_columns` when given a trial path.
   That also fixes the `test_column_catalog_complete` failure the ef96337 snapshot showed.
6. Docs: MANUAL (Overlays: the Grid tab; tab list corrected), `dic_correlate.md`.

## Why

The user: "on DIC, PIV, DVC i want the grid to show overlayed on the image. including any mesh
refinement." The element arrays existed inside the solvers but were dropped in the kernel
adapters, so only vectors could be drawn.

Design calls, recorded so they can be vetoed:
- **Boxes (bounds), not polygons.** Every element kind here is axis-aligned, so drawing needs
  no join against node rows and survives the single nodes dropping unsolved rows.
- **DIC also keeps the node indices** for the staged refinement card (next step).
- **A grid is an annotation, not a field.** It is excluded from auto-resolution and from the
  Points overlay rather than given a new domain.

Refinement itself (the single node's sockets and the staged Refine Mesh card) is the next
commit. The grid already shows refined elements by level once refinement runs.

## Files

- New: `nodegraph/catalog/_shared/analysis_mesh.py`
- `nodegraph/kernels/dic_correlate.py` + `.md` (mesh in series results)
- `nodegraph/catalog/analysis/dic_correlate.py`, `piv.py`, `dvc_field.py`, `dic_stages.py`,
  `piv_stages.py`, `dvc_stages.py`; `nodegraph/catalog/io/load_analysis.py`;
  `nodegraph/catalog/_shared/analysis_units.py` (columns in the trial header),
  `_shared/labels.py`
- `nodelab_v2/overlays.py`, `nodelab_v2/viewer.py`; `scripts/_nodelab_v2_phase5_probe.py`
  (Grid checks)
- `MANUAL.md`
- **Hotspots:** `nodegraph/selftest.py` (`test_analysis_mesh`, store-test grid round trip),
  `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`,
  `scripts/catalog_baseline.json` (re-blessed: extra_layers on the 4 grid cards)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_analysis_mesh()"`
- The GUI probe prints `[grid] DIC/PIV/DVC grid overlay: …`.
- In the app, pull a DIC, PIV or DVC node: the grid draws over the image (Overlays → Grid;
  `subsets` for the windows).

## Gates

- [x] Focused selftests: codemap, node demos, socket contract / docs / option docs, column
      catalog, import hygiene, DIC/PIV/DVC catalog, the three staged chains, store, parallel,
      quick scope, mesh, iterate, catalog. Full snapshot run: next entry
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
      PROBES PASSED (exit 0), with the new Grid block
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (130 ops; re-blessed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
- ef96337 on its snapshot:
  - selftest: 137 ok, then `test_column_catalog_complete` failed on `io.load_analysis` (fixed
    here);
  - the 38 tests after `test_write_movie` passed;
  - the GUI probe passed.
