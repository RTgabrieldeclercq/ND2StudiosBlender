# §7d. The optional `raw` socket — measure on unenhanced pixels (2026-07-28)

*A leaf of the **wire-node-v2** skill. Cited as `§7d`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


Enhancement is for *finding* objects; the numbers you report should come from the pixels
the camera recorded. A node that MEASURES intensities therefore takes an optional second
Dataset input, `_InRaw()` (`catalog/_shared/raw_measure.py`), resolved by
**`_intensity_provider(ctx, ds)` →
`(provider, is_raw)`**. Wired: those pixels are measured. Unwired: its own image, exactly
as before. On `analysis.measure` (all stats) and `analysis.histogram_threshold` (the region
table's intensity columns, re-measured via the vendored `_measure_regions`).

Four rules make it safe, and they generalize to any "second Dataset input" node:

- **Override MEASUREMENT only, never segmentation.** The mask, the label raster and every
  threshold stay on the main input — the chain the user tuned. So geometry and calibration
  remain one consistent story and only the *reported* numbers change. A node that wants to
  *threshold* raw pixels needs no lever: wire the raw Dataset into its main input.
- **Declare it AFTER `data`.** `graph.dataset_preds` sorts by declared socket position, so
  `data` stays `dpreds[0]` = the calibration/domain env source no matter which edge the
  user wired first. Never let an auxiliary input become the env.
- **Refuse a geometry mismatch** with **`_require_same_grid(main, other, socket=…,
  consequence=…)`** (`_shared/sampling.py`) — do NOT hand-roll it. Two branches read
  voxel-for-voxel must agree on BOTH halves: `AxisSizes` (catches a crop/resample/project)
  *and* the sampling provenance `__sampling__` (catches everything shape-preserving that
  moves the content — `align.drift` and `registration.stabilize` are exactly that, and a
  shape-only guard waved them through while the node measured each object where it used to
  be). `consequence` is the only per-node part: say what goes wrong *here*.
- **A stamp may declare the AXES its effect is confined to**, as a `"<axes>:"` prefix —
  `CHANNEL_STAMP` (`"c:"`) for a channel tap, `Z_STAMP` (`"z:"`) for a Z-collapse. The rule
  `_sampling_of`/`_require_same_grid` apply is one sentence: **a stamp confined to axes that
  are singleton on BOTH sides cannot misalign anything.** That is what lets "segment ch0,
  measure ch1" and "dots from a Z-projection, areas from a single-plane Z-crop" through
  while a lateral crop, a drift and a resample stay refused. If you write a node that
  collapses or reindexes ONE axis and leaves every other address alone, mark its stamp;
  `util.crop` decides **per call** (a pure z-crop is lateral identity, a windowed one moves
  the corner), which is the discrimination the exemption rests on. An unmarked stamp is
  never dropped — that is the safe default, so an unparsable prefix costs correctness
  nothing. `m`/`t` are deliberately NOT exempt: collapsing them picks a position or
  timepoint, and calling frame 3 and frame 7 one grid is a content decision the guard has no
  basis to make.
- **The memo needs nothing.** A new predecessor folds into `recipe_hash` automatically, so
  wiring or unwiring the second input re-keys the node on its own.

Why not a Mode toggle: a lever cannot say *which* raw Dataset, and a hidden "read the
source" path would break the "pure function of its inputs" law the memo depends on. An
explicit socket is visible in the graph, diffable, and serializes for free. Same shape as
`analysis.dvc_field`/`dic_correlate`'s optional `reference` input.
