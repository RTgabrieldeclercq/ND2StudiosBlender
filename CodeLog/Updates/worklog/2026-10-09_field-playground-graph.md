# Field playground graph

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

A ready-made graph for exploring the Field cards:
`graphs/field_mechanics_playground.nd2graph.json`, with three pages.
`scripts/make_field_playground.py` writes the graph and the synthetic data it needs.

1. **2-D gel (synthetic).** A bead-gel movie (512² px at 0.5 µm/px, 6 frames).
   - **The deformation:** a cell contracting 12 % at its centre (upper left), a vortex turning
     10° counter-clockwise (lower right), and a 1 % stretch along x. All three ramp in from
     frame 0 to frame 5.
   - **The chain:** DIC (fixed frame) → Field Gradient → Strain (Green–Lagrange) → Rotation &
     Curl → Volume Change → Stress (neo-Hookean, plane stress, E = 1 kPa, ν = 0.45).
   - **Figures:** four Plot Fields on frame 5 (strain_eq, rotation_deg, jacobian with contours,
     von_mises).
2. **Chip flow PIV (real data).** The user's chip-flow pipeline: 10 FITC burst files chained on
   T, Otsu granules on the R-B file, pore = not-granules, then ensemble PIV with velocity, only
   inside the pores.
   - **The chain:** Field Gradient (velocity) → Strain (rate) → Rotation (vorticity) → Volume
     Change (divergence) → Stress (Newtonian, η = 1 mPa·s).
   - **Figures:** four Plot Fields at position 24, plane 3, the fastest flow in the earlier PIV
     summary (speed, curl, strain_eq, divergence).
3. **3-D gel (synthetic).** A bead-gel volume (40 × 160² at 0.5 µm/px, 1 µm z step, 3
   frames).
   - **The deformation:** a central cell contracting 10 % and twisting 6° about z, over a 1.5 %
     x stretch.
   - **The chain:** DVC → the same chain (neo-Hookean).
   - **Figures:** four Plot Fields on the mid plane (jacobian, rotation_deg, pressure,
     von_mises).

Frames on each page name what the data contain.

**The script** renders the beads at their EXACTLY displaced positions; the image is not warped.
So the only error is the correlation's. It also writes `truth.json`, the analytic gradient,
strain, J, curl and rotation at each feature centre and in the far field.

**The data** go to `sample_data/field_mechanics/`. That folder is git-ignored, so the other
machine regenerates them with the script, which also re-points the graph at its own copy. The
TIFF Load cards use Access = ingest, because direct reading is ND2-only.

**MANUAL §16 D4** gains a "graph to try it on" bullet, and a sentence on Plot Field's
zero-centred diverging limits (fixed in `dc55c6d`).

## Why

The user asked: "set me up a node graph of what you just did, so i can play with the results".

**Why synthetic gels as well as real data.** A known deformation is the only way to tell
whether a number is right. The real datasets each cover only part of the chain:
- the CRC early-run ND2 still faults on its very first frame ("Windows fatal exception: page
  error" from the memory map);
- the CRC reference file has no time axis, so it has no deformation;
- the chip flow is a velocity field, so it exercises the strain-rate, vorticity and viscous
  branches.

Synthetic 2-D and 3-D gels cover the displacement → strain → J → stress path against ground
truth. The chip page covers the real velocity path.

**Why the bug-fix commits came first.** Pulling every page headlessly before handing the graph
over found four bugs; see `2026-10-09_field-playground-fixes.md`.

## Files

- `scripts/make_field_playground.py` (new): data, truth, and the graph.
- `graphs/field_mechanics_playground.nd2graph.json` (new).
- `MANUAL.md`: §16 D4.
- Regenerated: `codemap/gen/*`, `codemap/STATE.md` (the new script module).

Shared files both developers edit: `codemap/gen/*`.

## How to verify

1. Run `.venv\Scripts\python.exe scripts/make_field_playground.py`. It writes the data in about
   5 s.
2. Run `python run.py`, then File → Open `graphs/field_mechanics_playground.nd2graph.json`.
3. Pull any `fig_*` card, or select Stress and open the Viewer's Field overlay.

**Pulled headlessly through `EngineRunner`** (scratchpad `verify_playground.py`):

**Page 1 (2-D):** stress in 5.5 s; 0.3 s on re-pull, from the saved DIC trial. At frame 5:

| Point | Recovered | True |
|---|---|---|
| Far field | J 1.0097, strain_xx 0.0100 | 1.0100, 0.0100 |
| Vortex | rotation 8.8°, curl 0.31, Green–Lagrange strain ≈ 0 | 9.95°, 0.347, 0 |
| Cell | J 0.82 | 0.783 |

- The DIC subset smooths the peaks, so the cell and vortex values come out lower.
- The far-field stress matches plane-stress Hooke: von Mises 10.7 Pa against 10.8, pressure
  −5.8 Pa against −6.0.

**Page 3 (3-D):** stress in 6.5 s. At frame 2:

| Point | Recovered | True |
|---|---|---|
| Far field | J 1.018, strain_xx 0.019 | 1.016, 0.016 |
| Cell | J 0.815, rotation 4.3° | 0.741, 5.95° |

- **Far field:** that point sits about 10 µm from the volume edge, where DVC subsets are
  one-sided.
- **Cell:** a 12 µm subset averages over a 12 µm feature.
- **Stress:** the far field matches uniaxial-strain Hooke (von Mises 13.7 Pa against 13.3), and
  the cell's pressure is +832 Pa (compression).

**Page 2 (chip):**
- PIV over 36 positions × 11 planes gives 955,087 vectors in about 430 s.
- Medians: speed 2.4 µm/s (95th percentile 12.7), strain rate 0.13 /s, vorticity ±0.33 /s
  (5th–95th percentile), viscous von Mises about 0.3 mPa.
- **Save and reload: not achieved, because the S: drive is failing.** After the chain fix, the
  trial is created beside the first ND2. But only 1 of 396 units saved before the drive began
  refusing writes; see `2026-10-09_store-chunking-reverted.md` for the event-log evidence.
  Since `dc55c6d` the run still completes and its note says how many units were not saved. So
  each pull of this page recomputes about 8 minutes of PIV and reads about 8 GB from S:. Copy
  the chip data to a healthy drive and repoint the 11 Load cards; reloads then work.

## Gates

Run on the working tree of this commit, which also contains `dc55c6d` and `75e4342`:
- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`:
  - 192 `[ok]`, then stops at `test_write_movie`, the known failure that also fails on the
    base commit;
  - everything after it passes (`selftest_rest_at.py`: REST OF SELFTEST PASSED);
  - the new tests all pass: `test_piv_nan_predictor`, `test_analysis_store_writes`,
    `test_plot_field`, `test_chain_files`.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
  PROBES PASSED, exit 0.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (137 ops).
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT.
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The branch is ahead of origin;
  PR #1 is open.
