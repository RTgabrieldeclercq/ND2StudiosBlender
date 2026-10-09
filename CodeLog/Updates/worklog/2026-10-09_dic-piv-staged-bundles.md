# Staged DIC and PIV, and the palette lists every bundle's stages as a level under it

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** 2cd19b9

## What changed

1. **Staged DIC (pyALDIC)** — `nodegraph/catalog/analysis/dic_stages.py`, five nodes mirroring
   `al_dic/core/pipeline.py::run_aldic`'s sections: `dic.grid` (§2b, the lattice as a Point
   table + channel / ROI / reference policy, no correlation), `dic.initial_guess` (§3, the
   FFT integer seed with pyALDIC's auto-expanding search → `disp_*`, `seed_cc`),
   `dic.local_match` (§4, the 6-DOF IC-GN → `disp_*`, `grad_*`, `n_iter`, `hole`),
   `dic.global_step` (§5-6, the global FEM solve + post-solve corrections + ADMM),
   `dic.strain` (§8, four `strain_*` + `strain_valid`, no frame read). Placed together by
   `bundle:dic`. The kernel gained `dic_stage_*` entry points
   (`nodegraph/kernels/dic_correlate.py`, contract §2b).
2. **Staged PIV (OpenPIV)** — `nodegraph/catalog/analysis/piv_stages.py`, four nodes, one per
   piece of the multipass driver: `piv.first_pass`, `piv.validate`, `piv.deform_pass` (a NEW
   finer grid replaces the table — its columns from the coarser pass are dropped),
   `piv.smooth`. `bundle:piv` places the 64 / 32 / 16 px ladder with a Validate after every
   pass (six cards); `piv.smooth` is in the family as an optional extra. The kernel gained
   `piv_stage_*` entry points (`nodegraph/kernels/piv_field.py`, contract §2b).
3. **The palette's new level** (`nodelab_v2/palette.py`): stage band → role → **⧉ bundle →
   its stages**. Every stage of a bundle family is listed ONCE under its bundle row, in
   pipeline order, optional extras after (`Bundle.extra_ops`, `Bundle.family_ops` in
   `nodegraph/bundles.py`) — and no longer beside the ordinary nodes of the role. A stage row is
   still a node: double-click or drag places that one card. A bundle row shows only where its
   family survived the search / page filter. The bundle overview lists the placed cards in
   order plus "Also in this family".
4. Roles, demos (every stage demos on `speckle_pair`), MANUAL §15 rows, CON-25 updated and
   re-blessed, selftests `test_piv_stages` / `test_dic_stages`, the GUI probe (the hierarchy:
   each bundle row's children are its family once each; no stage repeated at role level),
   codemap/synopsis/baseline regenerated. `al-dic` 0.9.0 installed in the venv.

## Why

The user, after the staged DVC: "this same thing needs to be done for the ALDIC node and PIV
nodes. Then all the subnodes should be grouped into a new hirearchy layer under the regions."
Read as: the palette's role sections ("regions") gain a level, so the fourteen stage cards do
not sit unexplained beside the ordinary nodes — they list under the bundle that places them.

Design calls, recorded so they can be vetoed:

- **pyALDIC stages follow its own Sections**, read from al-dic 0.9.0 (every called function is
  public; the ADMM loop and the β tuning are reproduced from the driver line for line, the
  driver's private `_smooth_disp` / `_apply_post_solve_corrections` / `_auto_tune_beta` called
  directly so the corrections are pyALDIC's, not a copy). Covered paths: the FFT seed (what the
  single node uses), fixed and previous-frame references, one static ROI per plane. Not
  covered — the single node keeps them: pre-solve mesh refinement, seed propagation, an
  external reference Dataset.
- **PIV's unit of staging is the pass, not a solver section**, because that is how OpenPIV is
  built and how a user tunes it (a window per rung, validation between). A deform pass changes
  the grid, so it REPLACES the table rather than adding columns; Validate and Smooth add
  columns on the pass's rows. Not covered (the single node keeps them): ensemble averaging,
  velocity columns, the disparity uncertainty, an external reference.
- **One field on the wire per family** — the same `disp_*` µm columns the single nodes write,
  so the Viewer's arrows, Rasterize Field and every consumer read any stage's output.
- **Helper names are module-unique** (`_piv_*`, `_dic_*`): the socket-contract scanner keys
  forwarded params by function name across the catalog, and two `_plan`s shadowed each other.

Proof that the splits are faithful (the selftests pin these):

| chain | vs single node | where |
|---|---|---|
| DIC grid → seed → local → global → strain | 7e-15 µm on displacement and the four gradient columns, all 289 nodes | Engine, synthetic 160² speckle pair, (1.5, 2.0) px shift recovered |
| PIV first pass (32 px) → validate → deform (16 px) → validate | 1.4e-13 µm on all 225 vectors | Engine, `speckle_pair` phantom |
| DIC / PIV kernels alone | 0.0 | the same fixtures |

## Files

- `nodegraph/catalog/analysis/dic_stages.py`, `piv_stages.py` (new)
- `nodegraph/kernels/dic_correlate.py`, `.md`; `nodegraph/kernels/piv_field.py`, `.md`
- `nodegraph/bundles.py` (`extra_ops`, `family_ops`), `nodelab_v2/palette.py` (the level)
- `nodegraph/catalog/__init__.py` (**hotspot**), `nodegraph/selftest.py` (**hotspot**: two tests,
  two `_LAYER_RESOLVE_EXEMPT` entries with the single nodes' reason)
- `scripts/_nodelab_v2_phase5_probe.py`, `MANUAL.md` §15
- `codemap/node_roles.json`, `node_demos.json`, `concepts.md` (CON-25), `codemap/gen/*`,
  `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` (**hotspots**,
  regenerated)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_piv_stages(); s.test_dic_stages()"`
- In the app: the palette's *Motion & deformation fields* role shows DIC (pyALDIC), DVC (pyALDVC),
  PIV (OpenPIV), Optical Flow, Accumulate DVC Field as nodes, then three ⧉ bundle rows; expand
  `⧉ PIV (OpenPIV) · staged` → First Pass, Validate, Deform Pass, Smooth. Double-click the bundle
  → six wired cards in a "PIV (OpenPIV)" region; double-click a stage → that one card.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> 181 `[ok]` up to `test_write_movie`, which fails on
      the branch base too (pre-existing, movie writing); the 38 tests after it, run in order by a one-off
      runner, all pass — `test_dvc_stages`, `test_piv_stages`, `test_dic_stages` among them. The first run
      caught a one-sided picker pairing on PIV First Pass's window (removed: the single PIV node has none)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED, with the
      new hierarchy check. One of three runs crashed (exit 139) inside the final `os._exit(0)` AFTER printing
      the pass line; the re-run exited 0 — a teardown flake, not a failed check
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-blessed: nine new ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-25 re-blessed)
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
