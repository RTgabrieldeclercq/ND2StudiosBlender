# Bead plots three timepoints

- **Date:** 2026-10-14
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**Plot XY gains three capabilities.**

1. **Per = column.** One figure per distinct value of `per_by`, in first-seen order and titled by the value. All figures share their axes and ONE colour range taken over every row. The figure count is in the data, so the edit-time picture leaves T unknown. A blank `per_by` is refused with a fix.
2. **Reference lines (x).** `x_lines` / `x_line_label` draw dotted vertical lines at numbers, or at a column's median (a constant column lands on its value). Each line is labelled with its words, value and unit, the open X range widens to show it, and point labels treat it as an obstacle.
3. **Colour bar top.** The bar's top is now its own tick, labelled with its value.

**The IL-8 bead graph is rebuilt for three timepoints** (`graphs/IL8_bead_spheroid_distance.nd2graph.json`, 75 nodes).

- **Inputs.** Seq0004 gives T1 and T2. Seq0002, a third pass at 11:46, is a second bead chain, and its stage coordinates are registered into Seq0004's frame. The registration pairs beads by field and takes the median shift, (−22.2, +97.3, +26.2) µm.
- **Combining.** The three timepoint tables are concatenated with labels and plotted per column on one MFI scale. The scale runs from 1 to the maximum over all timepoints (918.7). Three PNGs are exported.
- **Spheroid edge.** A dotted line marks the spheroid edge: the equivalent radius of the max-projected GFP footprint, 557.8 µm, computed in the graph.
- **Detection.** The Cy5 rule is now > 120 with a 10 µm² floor (it was > 150).

## Why

The request was to:

- redo the graphs with the spheroid perimeter as a vertical dotted line;
- normalise the MFI scale across timepoints, with the highest intensity of all timepoints as the upper bound;
- add a third plot for `ChannelR-PE,Cy5_Seq0002.nd2`.

Seq0002 cannot be appended as a third frame of Seq0004. Its position list sits about (−22, +97, +26) µm off Seq0004's, while every bead stays centred in its field: the plate moved, the beads did not. A T-merge would carry one stage log for both files and misplace every T3 bead by about 100 µm. So each file is measured with its own stage log, and T3 is registered to T1 by the beads themselves.

One colour scale across figures from two files needs one table. Plot XY's per-frame mode keys on the T axis, which a concatenated table does not have, hence **Per = column**. A typed colour range was rejected: its top would not follow the data.

**Detection threshold.** At > 150, B4_0012, the bead nearest the spheroid, dropped out of T3 (smoothed peak 145). At > 120 it is a 15 px blob. The defocused B4_0019 at T1 and T2 only fragments into pieces of 4 px or fewer, so the 10 µm² floor separates the two cleanly.

**Colour floor.** The log scale's floor is pinned at 1. The data's smallest positive MFI (0.01) had stretched two decades of the bar over noise.

## Files

- `nodegraph/catalog/plot/xy.py`: `per = column` + `per_by`, `x_lines` / `x_line_label`, `_x_lines`, `_per_column`.
- `nodegraph/catalog/_shared/figure.py`:
  - `_figure_env` gives `per = column` an unknown T;
  - dotted reference lines with labels;
  - existing words are obstacles for point labels;
  - a tick at the colour bar's top.
- `nodegraph/selftest.py` **(hotspot)**: `test_plot_per_column_and_lines`, appended and registered at the end of `main()`.
- `codemap/gen/*`, `codemap/STATE.md` **(hotspot, regenerated)**: `plot.xy`'s new sockets and mode; `fp` only for the other plot nodes and `io.write_figure`. `codemap/node_synopsis.json` was regenerated.
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: only `plot.xy` differs.
- `MANUAL.md` §15: the Plot XY row.
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the rebuilt graph.

The working tree also carries earlier uncommitted work, including this session's bead-spheroid-distance and overlay-union-axes entries. `figure.py`, `xy.py`, `selftest.py` and the generated files mix hunks with it.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_plot_per_column_and_lines()"`
- Open the graph and pull **Export T1** to **Export T3**. They write `Downloads\IL8_bead_spheroid_figures\bead_mfi_vs_distance_T1.png` to `_T3.png`, each with the dotted "spheroid edge 558 µm" line and a colour bar topped at 918.7.
- **Run shift X/Y/Z** should read −22.2 / 97.3 / 26.2 µm.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The tree holds other uncommitted work, so committing is left to the owner.
