# Bead Finder on a dense, low-NA field: axial tolerances scale with σz, the auto slab stops voting for itself

- **Date:** 2026-10-08
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

`detect.beads` (the slab-projected Bead Finder) now finds the beads in a real 0.45 NA
confocal bead gel — `S:\9.1.26_CRC_Gradient\20260904_135958_104_reference\Channel640_Seq0000.nd2`,
54 positions × 54 planes × 1024², 0.286 µm px / 0.288 µm z, thousands of beads per field —
where before it reported 19 beads in a 256 × 256 crop that holds about 780. Five changes
inside `nodegraph/kernels/bead_slabs.py`, no socket or footprint change on the node:

1. **Every axial tolerance scales with the expected axial sigma.** The split-Gaussian z fit
   may wander `max(1.5, σz)` planes from its argmax (was a fixed 1.5), the re-profiled peak
   at the refined (y, x) may sit `max(2, ⌈σz⌉)` planes from the one the component came
   from (was 2), and two z peaks closer than `max(2, round σz)` planes are one peak (was 2).
   At 1.4 NA (σz ≈ 1.5 planes, the phantom) these are the numbers the kernel shipped with.
2. **A joint two-bead axial fit that does not hold falls back to the single valley-bounded
   fit** instead of refusing the bead. Still a Gaussian fit that can fail — never the
   parabola.
3. **The slab that holds a bead's peak plane owns its fit.** A z peak found in the margin
   beyond the slab's own planes is left to the neighbouring slab (`other_slab`), which sees
   the bead at full brightness. Before, every bead on a low-NA stack was fitted from three
   slabs and merged afterwards: three times the work for the same answer.
4. **The auto slab's density estimate is a raw LoG peak count on bead-thin projections**
   (`_probe_bead_count`, top/middle/bottom of the stack, no fits), and a re-run only ever
   goes THINNER. Before, pass 0 projected the whole stack, counted the beads it accepted,
   and derived the slab from that count.
5. **The auto overlap is capped at half the slab** (was `slab − 1`).

Plus the contract (`bead_slabs.md` §5/§6/§9), the node's docstring and its `auto` / `min_snr`
descriptions, the MANUAL §15 row, and a graph for the dataset:
`graphs/bead_field_crc_reference.nd2graph.json` = `io.load` → `util.select_position(0)` →
`detect.beads(diameter 1 µm, min_snr 3, slabs auto, max projection)`.

## Why

The user could not get the Bead Finder to see the beads in this dataset although it passes
its synthetic validation. The validation phantom is a 1.4 NA stack (axial FWHM 0.5 µm,
σz 1.5 planes) with at most 800 well-separated 1 µm beads; the real data is a 0.45 NA air
objective (derived axial FWHM 5.5 µm — a bead spans ~19 planes of a 54-plane stack) with
beads nearly touching in every projection. Measured on position 0, a 256² crop, with the
node's own derived sigmas:

- `_fit_z` refused **half of all converged axial fits** because the fitted centre of a
  19-plane-wide noisy bump legitimately sits 2–6 planes from its discrete argmax, and the
  wander bound was a fixed 1.5 planes. The re-profile step then refused a quarter more (the
  argmax moves > 2 planes for a half-pixel lateral shift).
- Of the rest, `_z_peaks` (min separation 2 planes) found two "peaks" 2–9 planes apart on
  most columns — noise shoulders on one σz = 8 bump — and offered them to the two-bead joint
  fit, which refused 632 of 863 calls, each one a lost bead.
- The auto slab's pass 0 (whole-stack projection) accepted 4 beads in the crop; that count
  argued for the thickest slab (27 of 54 planes), on which beads overlap everywhere. The
  undercount justified the projection that caused it — a self-defeating loop no re-estimate
  could escape, since the re-estimate read the same undercount.
- At the slab floor `round(2σz) = 16` the overlap rule gave 15 planes: 39 projections of
  the stack stepping one plane at a time.

Why scale the tolerances rather than loosen them: the fixed numbers are right at the NA the
phantom pins, and the phantom must keep passing unchanged (it does: 58/60, precision 1.0,
6 nm / 64 nm). Why a peak-count probe rather than a thinner pass 0: a count that needs no
fit cannot be starved by the fits' refusals, it over-counts noise — which only argues for a
thinner slab, bounded by the floor — and it costs three `_detect_2d` calls instead of a full
pass. Why not also change `min_snr`'s meaning: in a packed field the response MAD IS the
clutter level, which is the right thing to beat; the node text now says to start at 3 there.
Result on the crop (min_snr 3): 19 → 781 beads (fit_failed 1883 → 91 of ~3300 candidates);
at the defaults 19 → 92. The whole 1024² field of position 0, pulled through the GUI's
EngineRunner from the saved graph: 10,441 beads in 1103 s (auto slab 16 planes × 6, 51k
candidates; the kernel alone on the same volume: 10,412 in 1122 s). Whether centroids are
the right representation of a field this dense is a separate question (the user's call on
2026-10-08 was to go to DVC instead); the finder now does what it claims on it. On the phantom sweep (`scripts/_bead_finder_validate.py`):
precision 1.000 in every case as before; recall 400 beads 0.807 → 0.970, 800 beads
0.756 → 0.943, the others within ±0.01.

## Files

- `nodegraph/kernels/bead_slabs.py` — the five changes above
- `nodegraph/kernels/bead_slabs.md` — contract §5 (`probe_beads`), §6 (auto slab, ownership, tolerances), §9
- `nodegraph/catalog/detect/beads.py` — docstring + `auto` / `min_snr` descriptions (no socket change)
- `MANUAL.md` — §15 Bead Finder row
- `graphs/bead_field_crc_reference.nd2graph.json` — the dataset's graph
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` — regenerated (hotspots)

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as s; s.test_detect_beads()"` → `[ok] detect.beads: 58/60 …`
- `PYTHONUTF8=1 python scripts/_bead_finder_validate.py` → `BEAD FINDER VALIDATION PASSED`, recall ≥ 0.94 at 800 beads
- Open `graphs/bead_field_crc_reference.nd2graph.json`, pull Bead Finder, enable the Points
  overlay with *Show points from every Z*: the field fills with centroids (position 0 only —
  the Select Position node keeps the pull to one field; clear it to run all 54).

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> 166 tests pass up to `test_write_movie`, which fails
      on this branch's base too (pre-existing, movie writing, imports nothing from the bead code);
      the 38 tests after it were run in order by a one-off runner and all pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-blessed: descriptions changed, no declarations)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
