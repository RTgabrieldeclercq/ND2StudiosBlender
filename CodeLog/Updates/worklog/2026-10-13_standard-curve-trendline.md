# Standard curves: Plot XY trendline + blank reference line, Table Concat numeric condition

- **Date:** 2026-10-13
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock (same working tree as the clock and node-queue changes; separate entry)
- **Base:** 889d9d4

## What changed

**`plot.xy` has two new modes.** Both default to the previous behaviour, so saved graphs draw
as before.
- **`fit`** (Trendline): `none` / `linear`. Each series gets a least-squares line fitted
  **in the plotted coordinates** (log10 of a log axis), so log-log fits a power law
  y = A·x^b.
  - The line is fitted to the points the series draws, which are the means when `error`
    summarises.
  - The equation and R² join the series' legend entry.
- **`zero_x`** (x ≤ 0 on log x): `drop` / `reference_line`. On a log x axis, rows at x ≤ 0 (a
  blank, a zero dose) become each series' mean drawn as a dashed horizontal line, never
  fitted.
  - The mean is taken over **every** blank row.
  - If that mean is ≤ 0 on a log y axis, no line is drawn and the legend says so.
- **Short log axes** (under 2.5 decades, e.g. 500–5000) are now labelled in plain numbers at
  1-2-5.
- **Minor tick labels** now get the same font size as the majors. Before, they were larger
  and overlapped.

**`table.concat` writes `condition_value`.** When **every** row's condition reads as a number
(`names = 5000,2500,1000,500,0`), the table also gets that numeric column. It is declared at
edit time too, so Plot XY's X picker offers it.

**Shared helpers.** `xy_fit`, `xy_zero_refs` and `decorate_xy` live in
`_shared/figure.py` and are used by both the whole-table and the per-frame figure.

## Why

The user asked for IL-8 bead standard curves:
- R-PE background-subtracted signal against concentration;
- one line per photobleaching count;
- a trendline on each;
- log scale where the data are not linear;
- blanks as a dashed line, not fitted.

The catalog had no fit in any plot node, and Plot XY refuses a text column on an axis. So the
condition labels Table Concat writes could not be the x axis of a concentration curve.

**Why fit in the plotted axes.** Fitting in the space the user is looking at makes the
straightness the user sees and the R² the same claim. That answers "did the log scale reach a
linear regime" directly.

**Why the blank mean uses every row.** The first version averaged only the positive blank
rows on a log y axis. That drew configuration 2's blank at about +5 counts when its true
mean is −2. Averaging every row and refusing to draw a non-positive level is the honest
version.

## How to verify

Open `graphs/IL8_standard_curves.nd2graph.json` and pull a page's **Standard curve** node.
You should see:
- one line per count, each with its equation and R²;
- a dashed blank line where the blank mean is positive.

Automated: `selftest.test_standard_curve_trendline`.

## Files

- `nodegraph/catalog/_shared/figure.py`: `xy_fit`, `xy_zero_refs`, `decorate_xy`, trendline
  and reference drawing, plain short-log ticks, minor tick font
- `nodegraph/catalog/plot/xy.py`: `fit` and `zero_x` modes
- `nodegraph/catalog/_shared/table_ops.py`: `CONDITION_VALUE_COLUMN`, `numeric_labels`
- `nodegraph/catalog/table/concat.py`: `condition_value` (compute and edit-time columns)
- `nodegraph/selftest.py` (hotspot): `test_standard_curve_trendline`
- `MANUAL.md`: Plot XY and Table Concat rows
- `scripts/catalog_baseline.json` (hotspot): re-saved for plot.xy's modes
- `codemap/gen/*` (hotspot): regenerated
- `graphs/IL8_standard_curves.nd2graph.json`: the user's 4-page standard-curve workflow
  (data, untracked)

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (exit 139 afterwards, as on the untouched base 889d9d4)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-saved for plot.xy's modes)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
