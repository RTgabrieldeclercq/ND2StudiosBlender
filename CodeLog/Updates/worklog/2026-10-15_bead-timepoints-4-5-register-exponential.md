# Bead timepoints 4 and 5: Register Positions, exponential trendline

- **Date:** 2026-10-15
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**New node: Register Positions** (`table.register_positions`, category table). It moves a table's stage positions into the stage frame of a table on its required Reference input, fitted on the objects both tables contain.

1. The coarse shift is voted. Every row-to-reference vector no longer than `search` votes, and the shift most vectors agree on within `tolerance` wins.
2. Pairs are mutual nearest neighbours within `tolerance`.
3. `model` fits the pairs:
   - `rigid` (the default): a turn + shift (Kabsch, never a mirror);
   - `shift`: the median shift.

   Pairing and fitting are then refined twice under the fit.
4. Z moves by the pairs' median Z difference.

Every row moves, paired or not. X / Y / Z are replaced in place, or written to a named copy.

- **In place** keeps the table's identity. A Label table may keep its raster's name, and the table is not marked as a table node's output, so the Viewer still draws its rows over their image.
- **A named copy** goes through `write_table`: it is marked, and it is refused under a Voxel layer's name.

The node demo caught the in-place case: the first version refused writing to `labels` in place. The node adds:

- `reg_dx_um`, `reg_dy_um`, `reg_dz_um`: what was added to each row;
- `reg_residual_um`: distance to its twin after the fit (NaN when unpaired);
- `reg_rotation_deg`: the fitted turn.

**Plot XY gains `fit = exponential`** (y = A·e^(−x/λ)).

- It is fitted as a line to ln y whatever the axes, on y > 0 points.
- The legend shows A, λ (in x's unit) and R² of ln y.
- A new **Fit floor** socket (`fit_floor`, gated to exponential) keeps rows below it out of the fit while they stay on the figure.

**The IL-8 bead graph now covers five timepoints** (`graphs/IL8_bead_spheroid_distance.nd2graph.json`, 100 nodes).

- **New inputs:**
  - T4: `ChannelR-PE,Cy5_Seq0003.nd2`, 14 Oct 18:50 (+15.6 h);
  - T5: `ChannelR-PE,Cy5_Seq0004 (1).nd2`, 15 Oct 04:13 (+25.0 h).
- **Registration:** T3, T4 and T5 are registered onto T1 with Register Positions (rigid, tolerance 30 µm). This replaces T3's join + median-shift chain.
- **Figures:**
  - five per-timepoint figures on one MFI scale (top 1474);
  - a new MFI vs 3D distance figure over all five timepoints, with an exponential fit per timepoint (Fit floor 1 count).

## Why

The two new runs could not join the T3 recipe.

- **Different field lists.** Each run has 46–48 fields: every bead is imaged twice in a row, and new beads were added. Field m no longer matches T1's field m, so pairing by `m_t` fails.
- **A rigid move, not a shift.** The plate was put back on the stage turned 0.1–0.3° and moved ~80 µm. A shift-only registration leaves errors of up to 46 µm across the well; the rigid fit cuts that to 22 µm.
- **No other way to pair in-graph.** The catalog had no spatial pairing between two tables. A node keeps the registration inside the graph and reusable for the next re-mounted pass. The alternative was constants fitted offline and typed into Math nodes, which would not be reproducible in the app.
- **Why the shift is voted.** The move (~80 µm) is larger than some bead spacings, so nearest-neighbour pairing alone would mis-pair. The vote finds the shift most pairs agree on before pairing.

Measured on the data:

| Pass | Turn | Paired beads | Residual median / max |
|---|---|---|---|
| T3 | +0.009° | 18 | 5.5 / 12 µm |
| T4 | −0.075° | 12 | 16.5 / 20 µm |
| T5 | −0.12° | 11 | 17 / 22 µm |

- Re-registering T3 moves it ≤ 1.9 µm from the old median-shift result.
- An affine fit does not explain the ~15 µm left after a re-mount (median 16.7 → 14.9 µm), so it is per-bead noise, under 2 % of any bead's distance.

**Exponential trendline.** A source's gradient is expected to fall exponentially with distance. The fit is on ln y because MFI spans three decades: a data-unit fit would follow the brightest few beads.

**Fit floor.** Background-subtracted readings near zero (0.01) are noise, and on ln y each weighs like a real hundred-fold drop. With every y > 0 included, T1's λ is 700 µm; with the floor at 1 count it is 862 µm. That is the same reason the colour scale's floor is pinned at 1.

Fits, Fit floor 1:

| Pass | A | λ | R² |
|---|---|---|---|
| T1 | 1570 | 862 µm | 0.60 |
| T2 | 1333 | 958 µm | 0.54 |
| T3 | 532 | 1084 µm | 0.49 |
| T4 | 3201 | 862 µm | 0.55 |
| T5 | 5152 | 879 µm | 0.52 |

**B4_0012 is missing from T4 and T5.** This is the bead nearest the spheroid (0.94 mm). Its fields were imaged four and two times, but its Cy5 peak fell from 357 at T1 to 98 (T4) and 76 (T5) after smoothing. That is under the 120 detection cut and, at T5, inside the Cy5 background (max 75). The detection rule is kept the same for all five passes, so T4's and T5's fits start at 1.77 mm.

## Files

- `nodegraph/catalog/table/register_positions.py` **(new)**: the node and its helpers:
  - `rp_vote_shift`, `rp_pairs`, `rp_fit`, `register_xy`.
- `nodegraph/catalog/__init__.py` **(hotspot)**: `table.register_positions` slotted after `table.unique_objects`.
- `nodegraph/catalog/_shared/figure.py`:
  - `xy_exponential` (`min_y`, `unit`), and `XY_FITS` + `exponential`;
  - `decorate_xy` takes `fit_floor` / `x_unit`, and `frame_spec` passes them through;
  - the curve is drawn in `_draw_xy`.
- `nodegraph/catalog/plot/xy.py`:
  - the `exponential` choice and its docs;
  - the `fit_floor` socket;
  - `fit_floor` / `x_unit` in the draw spec.
- `nodegraph/selftest.py` **(hotspot)**: `test_register_positions_exponential`, appended and registered at the end of `main()`.
- `codemap/node_roles.json`: the op under `table_synthesis`.
- `codemap/node_demos.json`: a plate-mosaic demo whose reference is its own measured table.
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` **(hotspot, regenerated)**:
  - the new op;
  - `plot.xy`'s socket and choice;
  - `fp` only for the other plot nodes and `io.write_figure`, which share `figure.py`.
- `scripts/catalog_baseline.json` **(hotspot, re-saved)**: the new op and `plot.xy` only.
- `MANUAL.md` §15: a Register Positions row, plus Exponential and Fit floor in the Plot XY row.
- `CLAUDE.md` Footguns: with the app open, `test_analysis_store` fails on the governor's 6 GB free-RAM floor. Run the selftest with `NODEGRAPH_MIN_FREE_RAM=1G`. This bit the gate run for this change: `run.py` held ~427 GB and left 5.3 GB free.
- `graphs/IL8_bead_spheroid_distance.nd2graph.json`: the five-timepoint graph.

Not part of this change: the working tree also holds another session's uncommitted Properties-panel work. Those files are `nodelab_v2/inspector.py`, `nodelab_v2/window.py`, `scripts/_nodelab_v2_phase5_probe.py`, its worklog `2026-10-14_properties-fill-column.md`, and three `graphs/10.6_*` files. The worklog tool pre-listed them here; this change does not touch them. The regenerated `codemap/gen/*` also reflects their edits.

## How to verify

- `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_register_positions_exponential()"`
- Open the graph and pull **Export MFI vs distance** and **Export T1** to **Export T5**. They write these files to `Downloads\IL8_bead_spheroid_figures\`:
  - `bead_mfi_vs_3d_distance_all_timepoints.png`;
  - `bead_mfi_vs_distance_T1.png` to `bead_mfi_vs_distance_T5.png`.
- **T4 / T5 beads into the T1 stage frame** each report their turn, shift and residual in the node's status line.

## Gates

- [x] `NODEGRAPH_MIN_FREE_RAM=1G PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED. The 1G floor was needed because the open app left 5.3 GB free; see CLAUDE.md Footguns.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (139 ops)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed. The tree holds another session's uncommitted work, so committing is left to the owner.
