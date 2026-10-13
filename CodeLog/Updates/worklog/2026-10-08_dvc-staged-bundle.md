# Staged DVC: pyALDVC as five cards, placed together by the first node BUNDLE

- **Date:** 2026-10-08
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** 2b23ca5

## What changed

1. **Five new nodes, one per section of the AL-DVC pipeline**, in
   `nodegraph/catalog/analysis/dvc_stages.py`, mirroring how `al_dvc/core/pipeline.py` is
   organised: `dvc.grid` (§2: the node grid as a Point table + the channel and reference
   policy, no correlation), `dvc.initial_guess` (§3: the whole-volume shift + pyramid/NCC
   seed → `disp_*`, `seed_cc`, `seed_ok`), `dvc.local_match` (§4: the 12-DOF IC-GN match →
   `disp_*`, `grad_*`, `qfactor`, `status`, `n_iter`, `bad`, `valid`, `disp_std_*`),
   `dvc.global_step` (§5-6: the compatibility solve + ADMM loop → `disp_*`, `grad_*`,
   `outlier`) and `dvc.strain` (§8: the nine `strain_*` + `strain_valid`, no voxel read).
   The field between stages is ONE Point table (`dvc`): the grid writes the rows in
   pyALDVC's node order, every later stage reads it by layer name and adds or overwrites
   columns on those same rows (`with_layer`, ids untouched). Everything decided once — the
   subset size/spacing, channel, pre-filter, reference policy, then the local stage's
   interpolation/backend/stride — is stamped as `dvc_*` provenance and READ by the stages
   after it, so no control is asked twice and a stage fed a table without its predecessor
   refuses by name. The single node `analysis.dvc_field` is unchanged.
2. **The kernel gained staged entry points** (`nodegraph/kernels/aldvc_field.py`:
   `stage_para`, `stage_grid`, `stage_reference`, `stage_deformed`, `stage_initial_guess`,
   `stage_local`, `stage_global`, `stage_strain`, the `DVCReference` value), each
   reproducing its slice of pyALDVC's `run_aldvc` driver verbatim — the ADMM loop, which is
   not a function upstream, is re-implemented line for line in `stage_global`. The contract
   (`aldvc_field.md` §2b) documents them.
3. **Node bundles** (`nodegraph/bundles.py`, Qt-free; CON-25): a palette entry that places
   SEVERAL pre-wired nodes — a recipe of members with relative positions, wires, an input and
   an output socket for splice-on-wire, and a frame title. `GraphDocument.place_bundle`
   adds the members with fresh ids, wires them, splices the chain into a wire it was dropped
   on, and frames them as a region (border ports, linked tab on demand). The palette lists a
   bundle under its role with a ⧉ and an overview naming its members; double-click and drag
   place it (`window._add_at_center` / `_on_op_dropped`). The file holds ordinary nodes; the
   engine never sees a bundle. `bundle:dvc` is the first.
4. Roles, demos (every stage demos on `speckle_pair` 3-D behind its predecessors), MANUAL §15
   rows, the selftest (`test_dvc_stages`), the GUI probe (a bundle row is placed and framed;
   its row collectors count ops only), codemap/synopsis/baseline regenerated.
5. `graphs/dvc_crc_early_run_staged.nd2graph.json`: the early run registered, then the
   bundle as placed by the palette, with the grid set to 32/24 px, spacing 32/8, pre-filter
   σ 1, stride 2 on the match.

## Why

The user: "alDVC is currently a single node, and this makes it really hard to use. What we
need to do is split it up just like the pyALDVC code does so that the user can run it by
parts. I think this node when placed, should be a group of nodes. This should be the same
concept for other complex analysis nodes like AlDIC, PIV, and others."

Design calls, made without the interactive grill (the user was not at the keyboard) and
recorded here so they can be vetoed:

- **Stage boundaries are pyALDVC's own sections**, found by reading `al_dvc` 1.2.0: every
  section is a public module-level function except the ADMM loop; the boundary objects
  (`DVCMesh`, `ReferenceBundle`, `LocalContext`, `GlobalOperators`, `FrameResult`) are all
  rebuildable from the volumes and the parameters. The one object that is NOT handed between
  cards is the reference context (gradients, Hessians, operators): a few seconds to rebuild,
  and nothing opaque may ride the memo, the spill or a saved file.
- **The field between stages is a Point table on one layer, not a new domain**: the
  columns are the ones `analysis.dvc_field` already writes (`disp_*` in µm, `strain_*`,
  `qfactor`), so Accumulate DVC Field, Rasterize Field and the Viewer's arrows work after any
  stage with no change. New per-stage columns (`seed_cc`, `grad_*`, `status`, `bad`,
  `valid`, `outlier`, `disp_std_*`) ride beside them.
- **Provenance, not sockets, for what is decided once** (§7b metadata intelligence): the
  grid, channel and reference policy live on `dvc.grid`; the local context's settings on
  `dvc.local_match`. The global step therefore has no interpolation/backend controls of its
  own — it must rebuild the identical context, and a second copy of those knobs would be a
  lie waiting to diverge.
- **No external `reference` Dataset input on the stage nodes.** Four cards that must all be
  wired to the same reference is the kind of thing that goes wrong; Timeseries Builder
  chains a separate undeformed stack in front with Reference frame 0 — one mechanism the
  catalog already has. The single node keeps its socket.
- **A bundle, not a `nodegraph.groups` group, not a page recipe**: a group instance is one
  opaque card with one Dataset in and out, run as a unit — it hides exactly the stages this
  exists to expose; a page recipe replaces a whole page. A bundle is a *gesture*: after
  placement the user holds ordinary nodes in a frame, which is what "run it by parts" needs.
- **Bundles are not ops**: they carry a `role` for the palette and are not listed in
  `node_roles.json` (the synopsis gate refuses an op name it cannot find in the registry).

Proof that the split is faithful: on the `test_catalog_dvc` fixture the staged chain, pulled
through the Engine, equals `analysis.dvc_field` to 1e-8 µm on displacement, 1e-9 on strain and
qfactor (the kernel alone: 0.0); a strain-measure change re-computes exactly one node. On the
real early run (position 0, t0-t2, registered, the graph above pulled card by card through the
GUI's EngineRunner): grid 4805 nodes × 3 frames; seed 32 s (all 4805 nodes accepted, t=2
median −0.036/−0.055/−0.225 µm); local match 17 s (ZNCC median 0.978, 4311 converged);
global step 35 s (β auto, 805 median-test outliers replaced); strain < 1 s. The final field at
t=2 — median disp −0.040/−0.056/−0.145 µm, p90 0.766 µm, ZNCC 0.974 — is what the single node
reported on the same frames to three decimals, in 84 s of stage time against its 119 s.

DIC (pyALDIC) and PIV (OpenPIV) are NOT split in this change. The bundle mechanism is generic
and the recipe is the same (sections → stage nodes writing columns on one Point table, a
bundle to place them); pyALDIC follows pyALDVC's structure closely and is the natural next
one — `al-dic` is not installed in the venv, which is why it was not started here.

## Files

- `nodegraph/kernels/aldvc_field.py`, `.md` — the staged entry points (+ §2b)
- `nodegraph/catalog/analysis/dvc_stages.py` — the five nodes + `bundle:dvc`
- `nodegraph/bundles.py` — the bundle registry (new, Qt-free)
- `nodegraph/catalog/__init__.py` — MODULES (**hotspot**)
- `nodelab_v2/document.py` (`place_bundle`), `nodelab_v2/palette.py` (bundle rows + overview),
  `nodelab_v2/window.py` (place / drop / splice)
- `nodegraph/selftest.py` (`test_dvc_stages`, **hotspot**), `scripts/_nodelab_v2_phase5_probe.py`
- `codemap/node_roles.json`, `codemap/node_demos.json`, `codemap/concepts.md` (CON-25, blessed),
  `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json`
  (**hotspots**, regenerated), `MANUAL.md` §15
- `graphs/dvc_crc_early_run_staged.nd2graph.json`

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_dvc_stages()"` → `[ok] dvc stages: …`
- In the app: the palette's *Motion & deformation fields* role lists `⧉ DVC (pyALDVC) · staged`;
  double-click it → five wired cards in a "DVC (pyALDVC)" region; pull *DVC Initial Guess* and the
  Viewer's arrows show the seed field; pull *DVC Strain* and the rest runs; change the strain
  measure and only that card re-runs.
- Open `graphs/dvc_crc_early_run_staged.nd2graph.json`, set Crop's Frames to `t0-2`, pull the
  cards left to right.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> 179 `[ok]` up to `test_write_movie`, which fails on
      the branch base too (pre-existing, movie writing); the 38 tests after it run in order by a
      one-off runner and all pass (`test_dvc_stages` is among the 179)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
      (with the new bundle-placement probe)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-blessed: five new ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-25 blessed 2026-10-08)
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
