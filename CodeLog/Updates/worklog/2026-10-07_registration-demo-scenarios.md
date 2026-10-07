# Registration demo scenarios

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** registration-demo
- **Base:** e1ae0e28

## What changed

The *What does this node do?* window for **Registration** now runs on the registration bench's own synthetic worlds instead of the generic drifting-nuclei phantom, and lets the user play with them. Four new phantoms in `nodegraph/phantom.py` port `scripts/registration_synthetic_bench.py`'s generators to demo size — `star_field` (beads drifting and optionally turning, with a `faint` low-SNR variant), `moving_mass` (a textured body under an exact rotation / drift / shear / scale), `deforming_mass` (a one-sided bulge plus drift — motion no global transform represents) and `star_volume` (a 12-plane bead stack drifting in z) — each captioned with its TRUE motion. Demo recipes gain **scenarios** (`codemap/node_demos.json`: `scenarios: [{label, phantom, phantom_kw, view, fixed_params, fixed_modes, note}]`, parsed, validated and resolved by `demo_recipes.Scenario` / `scenario_recipe`); the window shows them as a **Synthetic data** dropdown above the modes with the scenario's note, rebuilds the session and worker on a switch, carries the user's settings over (the 2D/3D lever re-derives from the new stack), and the header caption follows. `DemoResult.frame_values` reads every Frame-domain layer at the viewed frame and the status line prints it — for Registration `drift_y / drift_x (/ drift_z) / drift_confidence` next to the caption's true drift. Registration gets eight worlds (drifting beads, sparse faint beads, a turning bead field, a turning body, a shearing body, a deforming body with and without a drawn estimate region, a bead stack drifting in z) and rewritten key-features text that says what to try; Drift Correction gets three (its limits beside Registration). `registration.stabilize` routes a single drawn rectangle to the kernel's sub-pixel rect path instead of the whole-pixel masked path. The demo gate runs every scenario and asserts the registration readouts against the true motion; the GUI probe switches the Registration demo to the z-stack world. This branch also merges `registration-bench` (the kernel fixes, the 3D lever, region + landmarks) onto `node-demo-window`, which is where the demo window lives.

## Why

The request: redo the Registration node's help Overview / *What does this node do?* so users can play with the node on the synthetic data the bench established its behaviour on. A demo that only shows nuclei drifting cannot show the things the bench found — that faint sparse beads need the matched filter, that a rotation needs euclidean, that a deforming body pulls the drift estimate and a drawn region fixes it, that a z-stack's axial drift needs the 3D lever. One phantom per recipe could not carry that, so recipes grew scenarios rather than the registration node getting eight separate demo entries (the window is one per op type by design). The readout of the Frame layers in the status line is what turns the demo from a picture into a measurement: the phantom's caption states the true drift and the node's own number sits beside it, so a user sees the correction land (or, on the deforming body, miss by the bulge's pull). The scenario switch keeps the user's settings because the point of switching worlds is to see the SAME settings succeed on one and fail on another; only the dim lever re-derives, since it is a property of the data. The single-rectangle → rect-roi routing came from the demo itself: the drawn still-half region, rasterized to a mask, took the integer-only masked correlation and landed 2 px off, where the bench's rect roi lands within 0.04 px; the rectangle is the common drawn region, and the mask path remains for cuts, ellipses and strokes, as the kernel contract documents.

## Files

Hotspots: `nodegraph/selftest.py` (two existing demo tests extended in place), `codemap/gen/*` + `codemap/STATE.md` (regenerated), `scripts/_nodelab_v2_phase5_probe.py` (one block added to the demo section). The merge commit before this one (`d4cec7f`) resolved `nodegraph/selftest.py` (both appended test groups kept) and `MANUAL.md` (both rows kept) by hand and regenerated the generated files.

- `MANUAL.md`
- `codemap/node_demos.json`
- `nodegraph/catalog/registration/stabilize.py`
- `nodegraph/phantom.py`
- `nodelab_v2/demo_recipes.py`
- `nodelab_v2/demo_window.py`
- `scripts/_nodelab_v2_phase5_probe.py`

## How to verify

`python run.py` → select a Registration node (or click *Registration* in the palette) → **?** / **What does this node do?** → the Synthetic data dropdown lists eight worlds; on *Drifting beads* at Frame t = 5 the status line reads `drift_y -7.50, drift_x +10.00`; switch to *Sparse, faint beads* and drag Lowpass σ to 0; switch to *Beads drifting in z* and read `drift_z`. Headless: `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_phantoms(); S.test_node_demos()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (encoder on this machine; fails identically on the base, as the node-demo-window entry records). The full run also tripped `test_lablink` once ("the beat thread outlived the command") while the GUI probe was loading the CPU; alone and in a quiet re-run of every group from `test_lablink` to the end it passes. `test_phantoms` and `test_node_demos` (78 live ops + 9 scenario worlds) and `test_registration_refinement` pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (offscreen; includes the new Registration scenario-switch check)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (108 ops; re-saved once in the merge commit for the registration sockets, unchanged by this commit)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (INV-17 extended for scenarios; INV-08/10/11 re-read and blessed in the merge commit)
- [x] `python scripts/_sync_check.py` -> branch `registration-demo` off `node-demo-window` (your local line, which is where the demo window lives), with `registration-bench` merged in; not pushed — the push is yours to make. PR #1 overlaps only on the usual hotspots.

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
