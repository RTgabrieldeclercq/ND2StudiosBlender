# Registration bench and fixes

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** registration-bench
- **Base:** e1ae0e28

## What changed

A synthetic known-answer bench for the registration nodes (`scripts/registration_synthetic_bench.py`: a 2-D star field, a 3-D star volume, a slowly deforming textured mass, a mass that rotates / translates / shears, and simulated landmark clicks — each with the exact true motion, scored as the RMS endpoint error of the recovered transform next to the best any model family could do) and the kernel fixes it forced. `nodegraph/kernels/registration.py`: the ECC seed sign is fixed (it started ECC at the mirror image of the answer); the correlation default is plain cross-correlation of the band-passed images instead of phase whitening (which also made `highpass_sigma` a no-op); a matched-filter `lowpass_sigma`; ECC is seeded from the feature match when ≥50 % of matches agree, and a converged warp is sanity-checked, with a failed warp gated rather than applied; `estimate_translation` / `estimate_series` / a new `apply_volume` accept a `(T, Z, H, W)` series and return `(dz, dy, dx)`; new `estimate_from_landmarks` (per-family least-squares fit, nested F-test model choice, χ² non-rigid verdict) and `parse_landmarks`; the duplicate `apply_frame` is merged. `registration.stabilize` gains the 2D/3D lever (3D corrects axial drift and stores `drift_z`), `lowpass_sigma`, a drawable `region` the estimate is restricted to, a `landmarks` socket with its `click_precision`, the `auto` model choice, and a per-frame `drift_confidence` layer; `registration_model` / `registration_landmarks` ride the metadata. `registration.align_to` correlates without phase whitening so its `highpass_um` is live. New selftest group `test_registration_refinement`; kernel contract, manual rows and CLAUDE.md trap updated.

## Why

The user asked for a full test of the registration nodes on synthetic data and for the conditions that cannot be registered. Measured before the fixes: euclidean/affine on a 12–60-star field were wrong by 2–14 px (twice the drift — ECC's seed had the wrong sign and its non-convergence fallback IS the seed, so a sparse field could never recover; a big smooth texture hid the bug because ECC still pulled in); translation on sparse stars was 0.18 px and on a textured mass 0.33 px where cross-correlation gives 0.007 / 0.000 (phase whitening amplifies the noise-dominated high frequencies on band-limited microscopy images and cancels any linear pre-filter, so the high-pass knob did nothing); 40 faint stars at peak SNR 8 were unregistrable (156 px) and are 0.09 px with a 1 px matched filter; `first`-mode rotation failed past ~10° because ECC converges to a wrong optimum with a HIGH cc (no threshold catches it — only a better seed does); a z-stack's axial drift was ignored entirely. The alternatives rejected: a pyramid ECC (rotation capture is scale-invariant, it would not have helped), picking the ECC seed by cc (the wrong optimum scores 0.99 vs 1.00), and hiding the model Mode in 3D (a saved euclidean graph would silently become translation — the planar warp + axial shift composite keeps every model valid). Landmarks answer the user's question about clicking corresponding points: five jittered pairs are enough for a nested F-test to name the family, the fit seeds ECC inside its capture range (a 42° turn registers to 0.02 px), and the residual says when NO global transform fits. Deformation itself is not fixable by any global model — the bench reports the family floor and the drift bias instead, and the drawable region is the remedy (0.45 → 0.04 px drift bias).

## Files

- `CLAUDE.md`
- `MANUAL.md`
- `codemap/STATE.md`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/node_synopsis.json`
- `nodegraph/catalog/_shared/drift_layers.py`
- `nodegraph/catalog/registration/align_to.py`
- `nodegraph/catalog/registration/stabilize.py`
- `nodegraph/kernels/README.md`
- `nodegraph/kernels/registration.md`
- `nodegraph/kernels/registration.py`
- `nodegraph/selftest.py`
- `scripts/catalog_baseline.json`
- `scripts/registration_synthetic_bench.py` (new)
- `CodeLog/Updates/worklog/2026-10-07_registration-bench-and-fixes.md` (this entry)

Hotspot files touched: `nodegraph/selftest.py` (one test group appended before `main()`, one call added after `test_align_to()`), `codemap/gen/*` + `codemap/STATE.md` (regenerated, never hand-merged), `scripts/catalog_baseline.json` (re-saved: `registration.stabilize` gained three sockets, the `dim` lever and the `auto` model choice). `nodegraph/catalog/__init__.py` is untouched.

## How to verify

`PYTHONUTF8=1 .venv\Scripts\python.exe scripts/registration_synthetic_bench.py --quick` → `ALL REGISTRATION BENCH CASES PASSED` (add `--legacy` to reproduce the pre-fix numbers; `--out DIR` writes the rows and a figure). In the app: Registration on a z-stack time-lapse with the lever on 3D shows `drift_z` in the Frame table; type `t=5; y,x -> y,x; …` into Landmarks with Model = auto and read `registration_landmarks` in the output metadata.

## Gates

Also: `PYTHONUTF8=1 python scripts/registration_synthetic_bench.py` -> ALL REGISTRATION BENCH CASES PASSED (full run 22 min; `--quick` 7 min; `--legacy` reproduces the pre-fix numbers). `MANUAL.md` §15 also needed `util.chain` un-backticked — the Timeseries Builder row named its predecessor as a code span, which the §15 phantom-key check reads as a registered node; that made the selftest red on origin/Blender already.

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes EXCEPT `test_write_movie`, which fails identically (frame means not monotone: 53.1, 53.1, 52.4, …) on the untouched v4-step11 checkout on this machine — a pre-existing encoder-environment failure, not from this change (assertion from commit 108833e, 2026-10-01). The nine groups after it in `main()` were run individually and all pass; the new `test_registration_refinement` passes.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (offscreen, on the final tree)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-saved on purpose: `registration.stabilize` gained `lowpass_sigma`, `region`, `landmarks`, `click_precision`, the `dim` lever and the `auto` model)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (INV-08/INV-10/INV-11 re-read and blessed — INV-11's text now records the registration departure; CON-13 and WF-07 were already flagged on origin/Blender, re-read and blessed; the generator now prefers a node's own kernel doc over a borrowed helper's)
- [x] `python scripts/_sync_check.py` -> branch level with origin/Blender @e1ae0e2 at commit time; PR #1 open, overlapping only on `nodegraph/selftest.py` and `scripts/catalog_baseline.json` (the standard recipes apply)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
