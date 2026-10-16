# Decay curves carried in to the spheroid edge

- **Date:** 2026-10-15
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Plot XY gains a Trendline span socket** (`fit_span`, shown with any trendline). It sets where each fitted curve is DRAWN, as 'low, high' in X's unit:
- each side is a number, or a column whose median is the value;
- a blank side keeps the series' own extent;
- an end a log x axis cannot place is ignored.

The fit itself does not move: A, λ, R² and every point are unchanged. The span flows through `decorate_xy`, so per-frame and per-column figures get it too.

**The IL-8 graph** sets `fit_span = "spheroid_edge_um,"` on both decay plots, **PLOTD** (MFI) and **PLOTDC** (IL-8 pg/mL). Every timepoint's exponential now runs from the spheroid edge (557.8 µm) out to its farthest bead. At the edge the curves reach:

| | T1 | T5 |
|---|---|---|
| MFI | 788 | 2,627 |
| IL-8 (pg/mL) | 856 | 3,372 |

## Why

The request was to "extend the plot for MFI and protein v. distance to the spheroid edge".

Each curve used to span only its own beads. Those start at 0.9 mm at T1–T3 and 1.7 mm at T4–T5, so the space between the beads and the dotted spheroid-edge line was empty.

A drawing span rather than a refit: the fit is still made on the beads, and the picture shows where it leads.

The values at the edge are an extrapolation. No bead was measured closer than 0.9 mm.

## Files

- `nodegraph/catalog/_shared/figure.py`: `decorate_xy(fit_span=)` overrides the fit's drawn `x0` / `x1`, and `frame_spec` carries `fit_span`.
- `nodegraph/catalog/plot/xy.py`: the `fit_span` socket, `_fit_span`, and the draw spec's `fit_span`.
- `nodegraph/selftest.py` **(hotspot)**: `test_gradient_map_standard_curve` asserts that the span moves the drawn extent and not the fit, for both the number form and the column-median form.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` **(hotspot, regenerated)**: the `plot.xy:in:fit_span` socket; `fp` only for the modules that import `figure.py`.
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: only plot.xy's socket list.
- `MANUAL.md` §15: Trendline span in the Plot XY row.
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the span on PLOTD and PLOTDC.

Not part of this change: the rest of today's uncommitted work (the gradient maps and standard curve, the five timepoints) and other sessions' files. The worklog tool pre-listed them; they have their own entries or are someone else's.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_gradient_map_standard_curve()"`
- Open the graph and pull **Export MFI vs distance** and **Export pg/mL vs distance**. In the two PNGs in `Downloads\IL8_bead_spheroid_figures\`, every curve starts at the dotted spheroid-edge line.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (248 ok)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 0)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL after `save` (141 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed; committing is left to the owner.
