# IL-8: Crank point-source fit, production rate, T8, and the final figure set

- **Date:** 2026-10-16
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**New node `table.point_source_fit`, "Point-Source Fit (Crank)".** It fits McGhee et al. 2022's Eq. 2, Crank's continuous point source `C = q/(4πDr)·erfc(r/2√(Dt))`, to a concentration column measured round a source.

- **Rate:** one rate per group (`constant`, each timepoint's average since the source started), or a rate history by superposition (`piecewise`).
- **D:** fixed (default 20 µm²/s, the paper's value) or profiled.
- **Boundary:** `infinite`, or `half_space` (a source on an impermeable floor).
- **Background:** `none`, or `uniform` (one level per group: the far field of a finite well).
- **Weighting:** `relative` (ln), `sqrt` or `absolute`.
- **Time:** each row's clock (Table Concat's `time_jd`) plus `start_h`.
- **Bootstrap:** 95 % intervals on the rates.
- **Outputs:**
  - the model's value per row;
  - a `<name>_rates` Label table (`q_fg_s` with its interval, pg/h, released pg with its interval, cumulative pg, background pg/mL, D, R², n);
  - a `point_source_fit` metadata record.

The maths is in `_shared/point_source.py`.

**Plot XY.**
- `fit = point_source` draws the upstream fit's curve per series, or per panel under Per = column. It draws in the signal's units through the recorded standard curve when Y is the signal, and refuses an X that is not the fitted distance.
- `error = column` with `error_column` draws per-row half-widths.
- `legend_loc` (inside / right / below) is a Plot-XY-only custom-style socket.

**Gradient Map:** `point_source_idw` and `point_source` take the trend's shape from the fit and refit only its amplitude to the mapped objects.

**Standard Curve** records its `signal` column in the curve record. **If / Else** splits a Label table that has no raster as a table only.

**IL-8 graph** (`graphs/IL8_bead_spheroid_distance.nd2graph.json`, 242 nodes):
- **T8 added:** `E:\ELISA_M_logo\20261016_065053_006\ChannelR-PE,Cy5_Seq0004.nd2`, 16 Oct 07:52.
- **The fit:** If/Else keeps beads ≤ 400 µm above Z = 5,022, then the Point-Source Fit runs on them: constant rate, D = 20, `uniform` background, `sqrt` weighting, T1 = seeding + 15 h, 400 bootstraps.
- **Combined plots:** MFI and pg/mL vs distance with the Crank curves.
- **Per-timepoint panels:** every bead coloured by height, both MFI and pg/mL.
- **Over time:** production rate, IL-8 released, and the well background.
- **Maps:** built on the near-floor beads with `point_source_idw`.

47 figures are in `Downloads\IL8_bead_spheroid_figures_final\`. The bead-lighting-up figure was made by a script on the graph's bead table; the script is saved beside the figures. The report is published as an artifact.

## Why

**The fit.** The user asked for the Crank point-source fit "per the PDF attached". No PDF arrived. The paper's Eq. 2 is the equation the lab's 9 Sep analysis (`Desktop\9.5-9.7_IL-8 PC3 ELISA\crank_model.py`) used, and its q is the production rate the user also asked for.

**The background.** The user then asked for curves that rise to the brighter far beads and flatten out ("a fit that reflects real data"). Physically, a finite well (about 6 mm across, √(Dt) ≈ 2 mm by 60 h) returns what an unbounded Eq. 2 lets escape. The `uniform` background models that, and raises overall R² from 0.27 to 0.43.

**Near-floor beads only.** MFI tracks bead height (ρ −0.8 to −0.9 at every timepoint, depth attenuation), so the high beads read dim at any distance and the fit uses beads ≤ 400 µm. That is also the subset the user asked for in the maps.

**D fixed at 20.** Profiling D on that subset gives 5.9 µm²/s, and the point source vanishes at T6–T8. Fitting all beads with a ln fit gives 35. D is therefore fixed at the paper's measured 20.

**Sqrt weighting.** ln lets dim beads pin the tail; absolute lets a few bright beads decide, and the point-source term collapses at T1–T3.

**Piecewise rates.** These swung between 0 and several fg/s (7–9 beads per timepoint; T1–T3 are the same beads re-imaged), so `constant` is used.

**T8 registration.** T8's own whole-well landmark fit (+335, +40 µm) was a chance alignment: only 2 of its beads then agreed with T1. Its raw beads match T7's 18 of 18 after a 5 µm shift, so it registers onto the registered T7 by shared beads (residual 1.1 µm).

**The bead figure.** No single bead spans all 8 timepoints. Sessions 3–4 imaged new fields, and the nearest T1–T5 bead is 1.3 mm from the T6–T8 one. The figure shows bead A (T1–T5) and bead B (T6–T8), both about 3.3–3.7 mm out and about 140 µm up, and says so. The `ChannelGFP,R-B,Cy5,TD` time-lapses image the M-logo sample, not WellB4, so they cannot bridge the gap.

**Result.** Production rate (average since seeding) is 1.40, 1.16, 0.44, 0.78, 1.23, 0.29, 0.46 and 0.40 fg/s at T1–T8. The well background is 31, 41, 31, 57, 92, 506, 522 and 418 pg/mL.

## Files

- `nodegraph/catalog/_shared/point_source.py` (new): the model, fit, predict, rate rows and the standard-curve forward map.
- `nodegraph/catalog/table/point_source_fit.py` (new): the node.
- `nodegraph/catalog/_shared/figure.py`: `xy_point_source`, the `point_source` fit in `decorate_xy` and `frame_spec`, the `curve` draw, the D legend note, `error = column` in `xy_series`, and `LEGEND_LOCS` with `style_sockets(legend_loc=)`.
- `nodegraph/catalog/plot/xy.py`: the `point_source` fit, `error_column`, `legend_loc`, and `_point_source`.
- `nodegraph/catalog/analysis/gradient_map.py`:
  - `point_source_trend`, the two methods, `_point_source_record`;
  - `_CLOCK_KEYS` now built from `metadata.PER_TIME_KEYS` + `SOURCE_CLOCK_KEY`, so a map of groups also drops `frame_time_s` and a bundle's per-file clock. The other session added those keys the same day, and its review caught the gap. `test_gradient_map_standard_curve` asserts this on the payload and the envelope.
- `nodegraph/catalog/table/standard_curve.py`: `curve["signal"]`.
- `nodegraph/catalog/analysis/if_else.py`: the table-only Label split.
- `nodegraph/catalog/__init__.py` **(hotspot)**: `table.point_source_fit` after `analysis.gradient_map`.
- `nodegraph/selftest.py` **(hotspot)**: `test_point_source_fit` after `test_gradient_map_standard_curve`, plus its `main()` line.
- `codemap/node_roles.json`, `codemap/node_demos.json`: the new node's role and demo.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` **(hotspots, regenerated)**.
- `MANUAL.md` §15: the Point-Source Fit row, plus Plot XY, Gradient Map and If / Else notes.
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the eight-timepoint graph.

Not part of this change: the Overlay handedness fix (its own worklog, `2026-10-16_overlay-handedness-frame.md`), the other session's `util.trim` and time-metadata work, and the earlier uncommitted IL-8 entries, which the worklog tool pre-listed.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_point_source_fit()"`
- Open the graph and pull **Crank point source + well background per timepoint**. The rail reads D 20 µm²/s (fixed) and q 1.4 … 0.4 fg/s, with background 31 … 418 pg/mL.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (251 ok, exit 0; run on the tree as regenerated, before the other session's time-metadata edits)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 0; on the shared tree while the other session was editing)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (143 ops) after `save`
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT (at my last regen)
- [ ] `python scripts/_sync_check.py` -> not pushed; committing is left to the owner.
