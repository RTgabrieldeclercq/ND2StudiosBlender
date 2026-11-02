# IL-8 production rate the paper's way: Monte Carlo uncertainty, fit span, calibration SD

- **Date:** 2026-10-29
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**`table.point_source_fit` (Point-Source Fit)**
- **New Mode `uncertainty`.**
  - `bootstrap`: the default, unchanged.
  - `monte_carlo`: McGhee et al. 2022, Appendix B.3. It refits each timepoint `mc_runs` times (default 10⁵) on perturbed inputs: each value (by the `value_sd` column, or `value_rel_sd`·value, default 0.117), each distance (`r_sd`, default 50 µm) and each time (`t_sd`, default 600 s). Each run's D has mean D and SD `D_sd` (default 10).
  - The reported rate and level are the runs' MEAN, with `q_lo`/`q_hi` at mean ∓ SD and `q_ci` = SD. The plain least-squares fit stays in the record as `q_nominal_fg_s`, and R² is recomputed for the model drawn.
  - `monte_carlo` needs Weighting = absolute and Rate = constant (the paper's fit); anything else is refused.
- **New socket `fit_span`.** Takes `low, high`, each a number or a column's median, read like Plot XY's Trendline span. Rows outside are not fitted.
- **New rate columns** `q_sd_fg_s` and `background_sd_pg_ml`, written in both modes.

**`table.standard_curve` (Standard Curve)**
- **New Mode `uncertainty`** (none / monte_carlo; only with Curve = fit), with sockets `mc_runs` (default 2000) and `x_rel_sd`.
- With monte_carlo it writes `<name>_sd`: each row's concentration SD from refits of the curve on perturbed standards (Appendix B.1). The SD is half the runs' 16–84 % range. The record also carries each standard's SD (`point_sd`).

**`_shared/point_source.py`**
- `monte_carlo()`: an exact two-variable NNLS per run (`_ls_runs`), batched.
- `unit_response` now accepts an array D.

**`graphs/IL8_run4_two_configs.nd2graph.json`, both pages**
- **Standard Curve** runs its Monte Carlo (standards' concentrations ± 1.6 %, the paper's Table 1).
- **New column** `spheroid_diameter_um`.
- **Point-Source Fit settings:**
  - absolute weighting, uniform well level, monte_carlo with 10⁵ runs;
  - D 20 ± 10, t ± 600 s;
  - r ± 155 µm (C5) / 176 µm (C6), the measured SD of each group outline's distance from its centroid;
  - concentrations ± 16 % (Config 1) / 43 % (Config 2), each the median per-bead SD from that curve's Monte Carlo;
  - `fit_span = spheroid_diameter_um,`, fit floor 0.
- **Plots** draw the curves from one diameter and mark both the boundary and the diameter. The rate plots show ± 1 SD.

## Why

The user asked for the Crank equation and production rate to follow the attached paper (McGhee et al. 2022) and to be verified. The equation was already right: the code's Eq. 2 matches an independent implementation to 12 digits, units included. The fitting was not:
- it used √ weighting where the paper uses least squares;
- it fitted beads inside one spheroid diameter;
- it dropped values under 1 pg/mL;
- its uncertainty was a bootstrap rather than the paper's Monte Carlo.

A preview of two passes per well, the paper as written vs the paper plus a uniform well level vs the old fit, was shown. The user chose "paper + well level" (B). Three departures from the paper's text were forced by Run 4's data:
- **D's draw is lognormal**, with the paper's mean and SD. The paper's normal 20 ± 10 puts 2 % of runs at D ≤ 0 and some just above. Eq. 2 then says nothing has reached the beads, a run needs ~10¹⁰⁰ fg/s, and the mean was 10¹²⁰ on Run 4.
- **The calibration SD is half the 16–84 % range, not the plain SD.** Near a Langmuir's top, a refit whose Bmax lands just above a bead's signal gives a near-infinite concentration, and Run 4's Config 2 rows had plain SDs of 10⁴–10⁶ pg/mL.
- **The graph uses one relative SD per configuration**, as the paper does with its 11.7 %, rather than per-bead SDs. Even the robust per-bead SD runs to 10⁵ for a few Config 2 beads near Bmax, and those beads took over the least squares in some runs (C6 T46: 1.7 ± 4.3 fg/s and a level of 3,700 pg/mL; with one 43 %: 0.03 ± 0.11 and 1,580). Both agree to the second digit wherever the per-bead SDs behaved.

Each timepoint's random stream is seeded by its label, so C6's 8-pass combined fit reproduces the 52-pass fit's numbers.

Results:
- **C5:** q goes 0 → 1.17 ± 0.55 fg/s (58 h) → 0.65 ± 0.33 (66 h); the level 90 → 1,416 pg/mL.
- **C6:** q goes 0.09 ± 0.04 (16 h) → 0.40 ± 0.26 (38 h) → ~0 from 54 h; the level 2 → ~1,560 pg/mL.
- **The paper**, for scale: 0.16–0.18 fg/s for one 500 µm spheroid.

## Files

- `nodegraph/catalog/_shared/point_source.py`: `monte_carlo`, `_ls_runs`, `MC_SEED`, array D in `unit_response`.
- `nodegraph/catalog/table/point_source_fit.py`: the uncertainty Mode and its sockets, `fit_span` (`_psf_fit_span`; a first `_fit_span` collided with Plot XY's helper name and the selftest's param index refused it), `_apply_monte_carlo`, bootstrap draws now also give SDs.
- `nodegraph/catalog/table/standard_curve.py`: the uncertainty Mode, `point_spreads`, `calibration_sd`.
- `nodegraph/selftest.py` **(hotspot)**: new cases inside `test_point_source_fit` and `test_gradient_map_standard_curve` only:
  - zero-spread MC equals the fit;
  - the mean holds and the run is seeded;
  - the lognormal D stays finite;
  - a timepoint alone matches the full fit;
  - the value-SD column;
  - the fit span, by number and by column;
  - refusals;
  - the calibration SD: 0 for spread-free standards, growing toward the top, seeded, announced at edit time.
- `MANUAL.md` §15: the Standard Curve and Point-Source Fit rows.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json` **(regenerated)**. The catalog diff is these two nodes only. The regenerated map also carries session nd2studiosblender-45's `placement.py` / `overlay.py` edits from the same tree (its own worklog, `2026-10-29_overlay-native-resolution.md`).
- `graphs/IL8_run4_two_configs.nd2graph.json`: rebuilt (146 + 481 nodes). The builder and the check scripts (`preview.py`, `margins.py`, `dtest.py`, `reltest.py`) are in `Downloads\IL8_Run4_figures\scripts\`.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_point_source_fit(); S.test_gradient_map_standard_curve()"`
- Open the graph and pull the fit node on page C6. The rail reads q 0.092 … 0.001 fg/s (Monte Carlo mean of 100000 runs), with D 20 fixed.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (252 ok, exit 0). After it, one socket description and the MANUAL row changed (a warning about per-row SDs); `test_socket_docs`, `test_option_docs`, `test_param_socket_contract`, `test_codemap` and `test_point_source_fit` were rerun and pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED. The process then exited 139 in its final `os._exit`, the known intermittent teardown crash.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 143 ops, after `save`; the diff saved was these two nodes only.
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT.
- [ ] `python scripts/_sync_check.py`: not pushed. Committing is left to the owner.
