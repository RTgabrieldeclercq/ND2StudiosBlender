# IL-8 session 3 (T6, T7): whole-well landmark registration; a stitched mosaic is camera-handed

- **Date:** 2026-10-16
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Register Positions gains a Landmarks input** (`landmarks` + `landmark_table`). Wired, the vote, pairs and rigid fit run on the landmark table against the Reference. The transform is then applied to every row of `data`'s table, which needs to share no object with the reference. `reg_residual_um` is NaN on those rows, and the landmark residuals go to the status line. Unwired, the node behaves exactly as before.

**Measure's 2026-10-15 change is reverted.** Measure flips a stage-Stitched mosaic like a camera field again; only its docstring and Flip X description differ from the last commit.

**Gradient Map treats a non-canvas image as camera-handed**, a stitched mosaic included. It previously ignored the flip for stitched mosaics.

**The IL-8 graph now has seven timepoints** (`graphs/IL8_bead_spheroid_distance.nd2graph.json`, 198 nodes):
- **T6 and T7:** `E:\ELISA_M_logo\20261015_172408_022\ChannelR-PE,Cy5_Seq0004.nd2` (15 Oct 18:24, +39.2 h) and `Seq0005.nd2` (16 Oct 02:25, +47.2 h), both Configuration 1.
- **Labels:** T5's label now reads "day 2" instead of "next day".
- **Landmarks:** one branch per session: the WellB4 Cy5 whole-well scan (session 1 `M_logo_Overnight1\…Seq0002`, session 3 `20261015_172408_022\…Seq0002`) → first frame → max over Z → stage Stitch → smooth 2 µm → Cy5 > 350, ≥ 10 µm² → Measure.
- **Registration:** REG5 and REG6 register T6/T7 by those landmarks (Search 2.5 mm, blank Z).
- **Map background:** the spheroid is now baked from its four raw camera tiles' max projections, not from the stitched mosaic.
- **Canvas:** the margin is 1.3 mm, so the registered T6/T7 beads fit.
- **Figures:** 24 in all, re-exported for seven timepoints.

## Why

**Session 3 imaged new beads after the plate moved.** No T6/T7 bead pairs with a T1 bead. A rotation scan of ±6° pairs at most 4 of 18, which is what chance gives, and fields pair only 6 of 21. So Register Positions on shared beads (T3–T5's method) cannot place them.

**The beads in the gel register sessions instead.** Each session has a 12 × 12-tile whole-well Cy5 scan, and the beads fixed in the gel show in both.

| Validation | Known move (shared beads) | Landmark move |
|---|---|---|
| Session 2 (T5's session) | (−57, +46) µm, −0.12° | (−57, +43) µm, −0.09° (36 pairs) |
| Session 3 | — | (−1,392, +1,297) µm, −0.09° (9 pairs, residual 9.6 µm) |

The session-3 result agrees with an offline count over many more spots, (−1,392…−1,417, +1,298…+1,317) µm. Sparser detection recipes (> 1,200 raw, > 700 smoothed) found too few session-3 pairs and locked onto chance peaks.

**Two other landmarks were rejected:**
- The M-logo scans show a +610 µm move: that sample did not move with this plate.
- The session-3 GFP tiles are offset by the B4 grid's re-definition, (+1,767, −1,247) µm. They cut the spheroid at their edge, and their Z log reads 0.

**A stage-Stitched mosaic is camera-handed. The 15 Oct Measure change was wrong.** With Flip X on, Stitch places the highest-x tile on the left (`u = xs.max() − xs`) and keeps each tile's pixels as taken. Evidence from the whole-well mosaic against the T1 bead run:
- read as "+x right", none of T1's 20 beads is within 30 µm of a mosaic bead (median 268 µm);
- read flipped, all 20 are within a median 5 µm.

The 15 Oct "confirmation" came from Overlay's bake, which makes the same no-flip assumption, so it was not independent. Reverting moves the spheroid source back to (−10,832.5, −10,135.6, 5,118.5) µm. All figures before 15 Oct used this value; the 15 Oct re-exports were about 46 µm off and are now replaced.

**A finding not fixed here: Overlay mirrors a stitched secondary baked onto an Experiment Canvas.** `placement_entry.handedness_for` forces Flip X/Y off for an "already stage-placed" (stitched) secondary. On a canvas this mirrors the mosaic about its own centre. Baked from its raw tiles, the spheroid matches the stitched bake mirrored back (IoU 0.97); as baked, the IoU is 0.73.

The rule is shared with Crop To and Merge, and it rests on a WellA3 measurement (784 µm off with the flip on) that I could not re-check. So it is left as is, and the graph avoids it by baking the tiles. **It needs its own investigation before Overlay, Crop To or Merge are trusted with a stitched input on a canvas.**

*Fixed the same day:* see `2026-10-16_overlay-handedness-frame.md`. The rule was a patch over a compositor fault that mirrored every non-canvas overlay, and both are now fixed. The tile bake stays in the graph, and it is correct either way.

**Science note.** Bead MFI tracks height above the floor more strongly than distance at every timepoint:

| Timepoints | ρ(MFI, height) | ρ(MFI, distance) | ρ(distance, height) |
|---|---|---|---|
| T1–T5 | −0.80 to −0.90 | −0.63 to −0.71 | +0.24 to +0.37 |
| T6–T7 | −0.80 to −0.81 | −0.36 to −0.38 | 0.00 |

At T1–T5 the far beads also sat higher, so the two effects were confounded. T6/T7's new beads decouple distance from height, and their exponential fit is weak (R² 0.18–0.20, λ ≈ 1.1 mm). Their brightest beads, 1,525, 1,421 and 871 at 3.3–4.3 mm, all sit 145–191 µm above the floor. The earlier depth-attenuation finding (Cy5 falls with Z, ρ −0.96) means this mixes optics with biology. A Cy5-based depth correction needs Cy5 unsaturated, so a re-image at lower Cy5 gain remains the fix.

## Files

- `nodegraph/catalog/table/register_positions.py`: the `landmarks` / `landmark_table` sockets and the fit-on-landmarks path.
- `nodegraph/catalog/analysis/measure.py`: `_flip` reverted to always honour Flip X/Y; docstring and Flip X description carry today's evidence.
- `nodegraph/catalog/analysis/gradient_map.py`: `image_geometry` without the stitched-mosaic exception; Flip X description.
- `nodegraph/selftest.py` **(hotspot)**:
  - `test_register_positions_exponential` gains a Landmarks case: new objects are moved by the transform fitted on shared ones, and are refused without landmarks;
  - `test_gradient_map_standard_curve` now asserts that Measure flips a mosaic.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` **(hotspot, regenerated)**: the two sockets, the two Flip X descriptions, and `fp` for the three modules.
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: register_positions' socket list, and the two Flip X descriptions.
- `MANUAL.md` §15:
  - Register Positions: Landmarks, with the validation numbers;
  - Measure: the corrected stitched-mosaic text.
- `CodeLog/Updates/worklog/2026-10-15_il8-gradient-maps-standard-curve.md`: its Measure claims are marked reverted, with the reason.
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the seven-timepoint graph.

Not part of this change: the earlier uncommitted work (14–15 Oct entries) and other sessions' files, which the worklog tool pre-listed.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_register_positions_exponential(); S.test_gradient_map_standard_curve()"`
- Open the graph and pull **T6 beads into the T1 stage frame (rigid fit on whole-well bead landmarks)**. The rail reads 9 landmark pairs, a turn of −0.086° and a shift of about (−1,392, +1,297).
- The 24 PNGs in `Downloads\IL8_bead_spheroid_figures\` cover T1–T7.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (248 ok)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 0)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL after `save`
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed; committing is left to the owner.
