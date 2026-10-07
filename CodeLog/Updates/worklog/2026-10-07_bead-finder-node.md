# Bead Finder node (`detect.beads`) — slab-projected 3-D bead centroids, validated on a confocal phantom

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** bead-finder
- **Base:** 7f3f6cf (node-demo-window head; PR #1 from the other machine is open and touches
  only the usual generated hotspots in common)

## What changed

A new analysis node, **Bead Finder** (`detect.beads`, 3-D only), finds the sub-voxel
centroids of fluorescent beads of a known diameter in a Z-stack and writes a Point table.
It cuts the stack into overlapping Z slabs (thickness `S`, overlap `G`), flattens each
(max / mean, or min for dark beads), finds bright spots in 2-D on every flattened image
(a scale-normalised LoG at the bead's expected apparent size, peaks above `min_snr` robust
noise sigmas), then refines every hit on the **raw** stack: a matched-filter z-profile
searched near the slab that found it and fitted with a **split Gaussian** (one sigma below
the peak, another above — a confocal axial profile is skewed), and a 2-D Gaussian on the
planes around that z for the sub-pixel (y, x) and the apparent sigmas, with pixels nearer to
a neighbouring hit left out so touching beads do not widen each other. Only what is
bead-shaped survives: a blob rather than a ridge (Hessian anisotropy of the projection), a
lateral sigma inside `size_tolerance` of the expectation for `diameter`, an aspect under
`max_aspect`, an axial sigma inside twice the tolerance, a fitted amplitude above the raw
noise floor, a convergent axial fit, a peak not on the first/last plane or on a slab window's
edge. The same bead found in two slabs is merged. `Slabs = auto` derives `S` from the bead
density (a whole-stack pass counts beads; `S` is set so ≤ 10 % share a projected footprint,
between two axial sigmas and half the stack, re-estimated up to twice); `manual` takes
`slab_thickness` / `slab_overlap` in µm along Z. The per-bead columns `amplitude`, `snr`,
`sigma_xy_um`, `sigma_z_um`, `skew_z`, `slab` ride on the Point table and are declared in the
column catalog. `axial_fwhm` derives the confocal axial PSF width from λ, NA and an immersion
index inferred from the NA.

The maths is a new in-repo kernel, `nodegraph/kernels/bead_slabs.py` (+ its `.md` contract,
indexed in the kernels README). A new phantom, `beads3d` (`nodegraph/phantom.py`), renders
`n_beads` 1 µm spheres on a padded Z grid, blurs them with an Airy-like lateral PSF and a
**skewed sinc⁴ confocal axial PSF** (far side 1.5× wider, peak at the bead centre), adds depth
dimming, an uneven background, shot and read noise, and plants the things a bead finder must
refuse: three aggregates, two fibres, a scan line, a haze, 25 hot voxels and four beads whose
centres lie just outside the stack. `Phantom` gained a `truth` slot carrying every planted
position and the artefact positions. `scripts/_bead_finder_validate.py` runs the shipped
node through the Engine across a density sweep (20 → 800 beads in 41.6 × 41.6 × 9.6 µm), a
left-to-right density gradient, a no-artefact control, manual slabs, dark beads with
`projection=min` and `projection=mean`, scores recall / precision / localisation against the
truth, names what every false positive sat on and why every missed bead was missed, and
draws a figure. Result at this commit: **precision 1.000 in every case**; recall 1.000 at 20
beads, 0.967 at 60, 0.927 at 150, 0.807 at 400, 0.756 at 800 (every miss is a bead touching
another or lying under a planted aggregate / scan line); median error 7 nm lateral, 66 nm
axial with a constant +59 nm axial bias from the skewed PSF; 1–6 s per field.
`selftest::test_detect_beads` pins the 60-bead headline, the table schema and catalog, the
memo fence (px / z step / NA / λ), mode re-keying, manual slabs, dark beads and the z = 2
refusal. The node demos itself on `beads3d` (`codemap/node_demos.json`), belongs to the
`object_detection` role, has a MANUAL §15 row, and is appended to the catalog `MODULES`.

## Why

The request (the design brief that was `node_idea.md`): a bead finder for a volume stack
that (1) takes multiple Z sub-stacks of size `S`, equally spaced with overlap `G`, flattens
each by max / min / mean — the spacing depending on the bead density, from which `G` and `S`
can be estimated; (2) finds circles / bright spots on each flattened image; (3) crops each
bead from the raw stack at its (x, y) and finds its z near the slab's region; (4) backs out
a sub-pixel (x, y, z) from a Gaussian of the intensity, skewed in z by the convolution; (5)
outputs the centroid list; (6) filters for beads of the correct size — and to test it on
synthetic bead data with the correct z intensity distribution of a fluorescent bead under a
confocal microscope, varying the density, with non-bead artefacts thrown in, finding only
the beads.

Why a new node rather than a mode on `detect.particles` (the vendored v1 bead kernel) or
`detect.spots`: both report every bright thing and know nothing about bead size; this one
assumes a bead of known diameter and uses that to refuse everything else, which is a
different contract, and the slab projection is a different detection geometry. Why the
maths is a kernel + contract: it is 500 lines of numpy/scipy that a validation script and the
node both call, which is what `nodegraph/kernels/` is for. Design calls made on the way, each
from a measured failure on the phantom: expected sigmas are the **least-squares-fit** sigma
of a solid sphere (0.272 d lateral, 0.28 d axial, computed numerically), not its FWHM — the
FWHM constant put every real bead at the bottom of the size band; **hot voxels are despiked**
by a 26-neighbour MEDIAN test (a max test let a spike beside a bright bead through, and a
spike's LoG response swallows the neighbouring bead's peak); the auto slab is **capped at
half the stack** (a whole-stack projection lets one fibre hide every bead beneath it);
touching beads get **Voronoi-masked fits**, and a fibre's end — which that masking disguises
as a bead — is caught by a fixed **Hessian ridge test** (2.5; 99 % of true beads read < 2.25
even when a fifth touch) plus an **unmasked re-fit** when none of a candidate's masking
neighbours became a bead; thin manual slabs planted ghosts from a bead's axial tail and let
noise blips and aggregate caps through, fixed by refusing a z peak on the slab window's edge
and by requiring the fitted amplitude to clear the raw noise floor and the axial fit to
converge. Fits moved from bounded trust-region with numeric Jacobians to Levenberg–Marquardt
with analytic Jacobians: 22 ms → ~1 ms per fit, 7.8 s → 0.9 s per field at 60 beads.

Known limits, deliberately not hidden: two beads closer than one diameter in (x, y) AND in
the same slab merge and are refused as too wide — the resolution limit of a projection, and
exactly what thinner manual slabs are for; a bead directly under an aggregate, a fibre or a
scan line in every slab that contains it is lost; the fitted z carries the PSF's own skew
bias (+0.06 µm on the phantom, constant across a stack, cancels in displacements); the
immersion index is inferred from the NA because no file carries it.

## Files

- `nodegraph/catalog/detect/beads.py` — the node (new)
- `nodegraph/kernels/bead_slabs.py`, `nodegraph/kernels/bead_slabs.md` — the kernel + contract (new)
- `nodegraph/kernels/README.md` — index row
- `nodegraph/phantom.py` — `beads3d`, `confocal_axial_kernel`, `Phantom.truth`
- `nodegraph/catalog/__init__.py` — **hotspot**: `detect.beads` appended to `MODULES`
- `nodegraph/selftest.py` — **hotspot**: `test_detect_beads` (+ registered in `main()`)
- `scripts/_bead_finder_validate.py` — the synthetic validation (new)
- `scripts/catalog_baseline.json` — **hotspot**: regenerated (`_catalog_snapshot.py save`)
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — **hotspot**: regenerated
- `codemap/node_roles.json` — `detect.beads` under `object_detection`
- `codemap/node_demos.json` — the `detect.beads` demo recipe
- `MANUAL.md` — §15 row
- `CodeLog/Updates/worklog/2026-10-07_bead-finder-node.md` — this entry

## How to verify

```
PYTHONUTF8=1 .venv\Scripts\python.exe -B -u scripts\_bead_finder_validate.py out.png
```
prints the recall / precision / error table per case, the attribution of every miss, and
ends `BEAD FINDER VALIDATION PASSED`; `out.png` shows the sweep and one field with the
planted, found and missed beads. In the app: open the Bead Finder's **?** demo window — it
runs on the `beads3d` phantom; drag Bead diameter away from 1 µm and the size filter refuses
the real beads. `PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as
s; s.test_detect_beads()"` runs the pinned assertions alone.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> stops at **`test_write_movie`**, a
      failure that predates this branch (it fails on the base commit too, 2026-10-07; the
      assertion is on movie frame values and touches nothing here). Every test before it
      passed, and the 36 tests after it — `test_detect_beads`, `test_phantoms` (8 phantoms),
      `test_node_demos` (79 ops live, the bead demo among them) included — were run in
      main()'s order from that point and **all 36 passed**. Not "ALL NODEGRAPH SELF-TESTS
      PASSED" until `test_write_movie` is fixed on `Blender`.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
      PROBES PASSED (154 `[ok]` lines; the process then exited 139 at Qt teardown, after the
      pass line)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (109 ops, after `save`)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`; INV-10 and INV-11 re-read,
      extended for `bead_slabs`, and blessed)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT (after `write`)
- [x] `python scripts/_bead_finder_validate.py` -> BEAD FINDER VALIDATION PASSED (worst precision
      1.000; recall at 20 beads 1.000)
- [x] `python scripts/_sync_check.py` -> branch `bead-finder` off `node-demo-window`; origin/Blender
      unchanged since (behind 0), so no rebase was needed
