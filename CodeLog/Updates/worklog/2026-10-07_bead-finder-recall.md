# Bead Finder recall: beads stacked in z, touching pairs, and a physical phantom

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** bead-finder
- **Base:** aa5681d (the Bead Finder node commit, same branch)

## What changed

`detect.beads` finds the beads a dense field hides. Three mechanisms, all in the kernel
`nodegraph/kernels/bead_slabs.py`, switchable there (`multi_peak`, `deblend`):

1. **Several beads per column.** A LoG hit's z-profile is no longer reduced to its
   `argmax`: every prominent peak (≥ 3 profile-noise sigmas and 10 % of its own height,
   ≥ 15 % of the window's tallest, ≥ 2 planes apart; `scipy.signal.find_peaks`) becomes a
   bead to fit, each on its own peak plane, with an axial fit bounded by the valleys to its
   neighbours or joint (two Gaussians, one shared sigma) with a neighbour closer than 4 σz.
   Beads stacked in z under one (y, x) were the second-largest loss.
2. **The two-bead deblend.** A compact blob that fails the width or aspect test is offered
   the two-bead model (two round Gaussians, one shared sigma, seeded along the blob's
   principal axis at the separation the excess width implies). Accepted only when the
   shared sigma lies inside HALF the size tolerance, both amplitudes clear the raw-noise
   floor, the pair is at least ¾ of a bead diameter apart, and its SSE is ≤ 0.6 × the better
   of a single axis-aligned Gaussian and the best ROTATED elongated Gaussian on the same
   pixels — the latter being what a fibre segment at any angle is. A deblended pair whose
   masking neighbours were all refused is dropped outright.
3. **Rigid spheres.** The node now passes the bead diameter in voxels (`diameter_px`,
   `diameter_zpx`): two beads can never be closer than one diameter, so the deblend floor is
   ¾ of it and the duplicate-merge radius is ½ of it per axis (it was 2 σ, then 1 σ in
   sigma-normalised distance; a fixed sigma radius either swallowed deblended pairs or let
   two fits of one bead through).

Also: the `beads3d` phantom places **hard spheres** (rejection sampling, centres ≥ one
diameter apart, touching allowed; `truth["n_beads"]` says how many fit); the node writes two
more columns, `deblended` and `stacked`, marking positions that rest on a model choice;
`sigma_z_px`/`skew_z` are NaN for a joint axial fit; fits use `_curve_fit` (LM, analytic
Jacobians for every model incl. the rotated Gaussian, capped evaluations: a model that has
not converged in 200 steps is wrong, not slow). The kernel contract, the node docstring and
socket text, MANUAL §15 and the phantom docstring describe all of this.

Measured (`scripts/_bead_finder_validate.py`, same seed, hard-sphere phantom), committed
kernel → this one: recall 0.950 → 0.967 at 60 beads, 0.933 → 0.960 at 150, 0.925 → 0.970 at
400, 0.851 → 0.958 at 800 (a fifth touching), gradient field 0.973, manual 2 µm slabs 0.975;
precision 1.000 everywhere except 800 (0.999 — one fibre point beside a bead). Ablation at
800: multi-peak alone +3.3 points, deblend alone +0.8, both +4.2 over neither (0.916 with the
new merge radius). Median error 6–11 nm lateral, 64 nm axial (the +0.06 µm PSF-skew bias is
unchanged). Remaining misses at 800: 13 under planted artefacts, ~17 z-stacked with the
dimmer bead's profile still dominated by the brighter one, 8 pairs. Cost: 1.7–3 s per
field at 60–400 beads (was 1–2), 9 s at 800.

## Why

The request: "let's work on strategies to increase the recall". A miss taxonomy on the
previous phantom put the loss squarely in two places — at 800 beads, 116 of 195 misses were
laterally touching pairs and 61 were beads stacked in z under the same (x, y); artefacts
and everything else were 18 — and showed thinner slabs alone buy only 3–5 points at 2–3× the
cost. So the two mechanisms above, aimed at exactly those two classes, rather than a slab
policy change.

The first versions cost precision (0.986–0.992): every false positive was a deblended pair
lying on a fibre, in slabs where a genuine bead sat near enough to the fibre that the
"some masking neighbour was accepted" rule trusted the pair. Hence the rotated-Gaussian
comparison (a mask cannot fool a model the pair has to beat) and the half-tolerance band on
the pair's sigma. The remaining duplicates (two fits of one bead 1.03 σ apart; one bead
found from two slabs a plane apart in z) and ghost splits all pointed at one missing fact,
that beads are rigid — which also exposed the phantom allowing interpenetrating spheres that
no detector could separate. Making the phantom physical is where most of the raw recall
jump came from (the committed kernel scores 0.851 at 800 on the new phantom against 0.756 on
the old); the mechanisms' own contribution is the A/B above, which is why both numbers are
reported.

## Files

- `nodegraph/kernels/bead_slabs.py` — multi-peak z, two-bead deblend, rotated-Gaussian ridge model, diameter floors, `flags`, switches
- `nodegraph/kernels/bead_slabs.md` — contract updated (params, columns, §6 conventions, §11 numbers)
- `nodegraph/catalog/detect/beads.py` — passes `diameter_px`/`diameter_zpx`; `deblended`, `stacked` columns; docs
- `nodegraph/phantom.py` — hard-sphere placement in `beads3d`
- `MANUAL.md` — §15 row
- `scripts/catalog_baseline.json` — **hotspot**: regenerated (socket descriptions changed)
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — **hotspot**: regenerated
- `CodeLog/Updates/worklog/2026-10-07_bead-finder-recall.md` — this entry

## How to verify

```
PYTHONUTF8=1 .venv\Scripts\python.exe -B -u scripts\_bead_finder_validate.py out.png
```
→ the table above and `BEAD FINDER VALIDATION PASSED`. For the attribution, run the kernel
with `{"multi_peak": False, "deblend": False}` on the same phantom (the ablation in the
numbers above was produced that way).

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> stops at the pre-existing
      **`test_write_movie`** failure (fails on the base commit too, 2026-10-07; a movie-frame
      assertion, nothing to do with beads). Every test before it passed, and the 36 after it —
      `test_detect_beads`, `test_phantoms`, `test_node_demos` (79 ops live, `detect.beads` the
      slowest at 1.8 s) included — were run in main()'s order from that point: **36 passed**.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
      PROBES PASSED (154 `[ok]`; the process then exited 139 at Qt teardown, after the pass line)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (109 ops, after `save`)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT (after `write`)
- [x] `python scripts/_bead_finder_validate.py` -> BEAD FINDER VALIDATION PASSED (worst precision
      0.999, recall at 20 beads 1.000)
- [x] `python scripts/_sync_check.py` -> branch `bead-finder`; origin/Blender unchanged (behind 0)
