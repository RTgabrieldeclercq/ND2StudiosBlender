# Quick analysis: tune DIC/PIV/DVC settings in seconds, then the full run

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** fbc29a1

## What changed

1. **A Quick analysis section in the inspector** for every DIC, PIV, DVC and SerialTrack node,
   staged cards included.
   - **Quick analysis** arms the F9 troubleshooting scope with the picks that analysis needs,
     then pulls the node (`MainWindow._on_quick_action`). The picks come from
     `nodelab_v2/quick.py` (`quick_picks`, Qt-free):
     - **M:** the position on screen.
     - **Z:** the plane on screen for planar methods; the whole volume for DVC and 3-D
       tracking.
     - **T:** {reference, current} for a fixed reference; {current−1, current} for
       previous-frame and ensemble; 5 frames for tracking.
     - **Region:** a centred box ~16 subsets across, unless one is drawn.
   - **A staged card reads the reference policy off its chain's grid card.**
   - **One pull, not one per pick.** Picks and box are set with the Viewer's signals blocked
     and handed to the runner directly (`ViewerPanel.set_frame_selection` is new).
   - **Full run** turns F9 off.
2. **The reference frame keeps its meaning under the scope.**
   - `runner._pin_frames` now stamps the scoped seed with `scope_m`, `scope_t`, `scope_z`,
     `scope_of` (the source's full M,T,Z,Y,X) and `scope_region`.
   - `_shared.analysis_units.scoped_frame` maps a fixed reference frame (a SOURCE index)
     through `scope_t`. It is applied in the single DIC/PIV/DVC nodes and in the staged grid
     cards, which stamp the index later stages use.
   - A reference left out of the picks is refused with the fix named.
   - The same stamps make the store skip scoped pulls. The lineage already folds them in.
3. **Quality metrics as Global scalars**, the shape `analysis.reduce_scalar` writes, so an
   Iterate card's Metric can name them:
   - `dic_solved_frac`;
   - `piv_valid_frac`, `piv_median_s2n`;
   - `dvc_valid_frac`, `dvc_median_q`.

   They are announced through `extra_layers`, and `adds_domains` gains GLOBAL. Full runs also
   record them in `trial.json` / `index.json`.
4. **A full-run estimate.** On a scoped pull, `analysis_est_full_s` (Global, seconds) is the
   measured unit time scaled by frames and area to a full unit, spread over the full unit
   count on the governor's lane plan. It also appears as an `analysis_estimate` sentence.
5. **Fix:** `_shared/analysis_units.py` imported `parallel` / `analysis_store` as
   `from nodegraph import …`, which breaks catalog import-hygiene rule 2. The full selftest on
   fbc29a1 stopped there, at `test_catalog_import_hygiene`. It is now `import
   nodegraph.<leaf> as …`.
6. MANUAL §13b quick-analysis paragraphs.

## Why

The user, mid-plan: "we also need a way to do a quick analysis with the settings chosen. this
will allow users to find the best set of variables for their data set befor the full run."

Design calls, recorded so they can be vetoed:
- **Built on the two features that already exist:** the F9 scope (memo-keyed, so revisiting a
  setting is instant) and Iterate (sweep / best-by-metric / golden search). A third mechanism
  would compete with them.
- **What plain F9 gets wrong for these nodes:**
  - one frame gives a correlator nothing to compare;
  - a picked set without the reference makes a fixed-reference run correlate against the wrong
    image, silently, because the payload renumbers frames.

  Hence analysis-aware picks plus the `scope_t` mapping.
- **Quick runs never save,** so trial folders hold full runs only.

## Files

- New: `nodelab_v2/quick.py`
- `nodelab_v2/inspector.py` (Quick analysis section, `quick_action` signal),
  `nodelab_v2/window.py` (`_on_quick_action`), `nodelab_v2/viewer.py`
  (`set_frame_selection`), `nodelab_v2/runner.py` (scope stamps)
- `nodegraph/catalog/_shared/analysis_units.py` (`scoped_frame`, `QUALITY`, `with_quality`,
  `full_run_estimate`, import fix)
- `nodegraph/catalog/analysis/dic_correlate.py`, `piv.py`, `dvc_field.py` (mapping, metrics,
  `extra_layers`); `dic_stages.py`, `piv_stages.py`, `dvc_stages.py` (grid cards map the
  reference)
- `MANUAL.md` §13b
- **Hotspots:** `nodegraph/selftest.py` (`test_quick_scope`), `codemap/gen/*`,
  `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json`
  (re-blessed: adds_domains + extra_layers of the three single nodes)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_quick_scope()"`
- In the app:
  - Select a DIC node and press **Quick analysis**. F9 turns on with the reference and current
    frame picked and an amber box in the middle.
  - Change Subset size: each run takes seconds, and going back is instant.
  - Press **Full run**.

## Gates

- [x] Focused selftests: registry, metadata pass, engine, solo/subset scope, import hygiene,
      catalog, codemap, node demos, socket contract / docs / option docs, DIC/PIV/DVC catalog,
      the three staged chains, store, parallel, quick scope, iterate. Full snapshot run: next
      entry
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`: on the commit snapshot
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (130 ops; re-blessed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
- fbc29a1 on its snapshot:
  - the main selftest stopped at `test_catalog_import_hygiene` (fixed here);
  - the 38 tests after `test_write_movie` passed;
  - the GUI probe printed ALL PHASE-5 GUI PROBES PASSED, then exit 139 in the final
    `os._exit` (the known teardown flake).
