# Time axis rebuild

- **Date:** 2026-10-16
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

Every node that changes T now rebuilds the series' time metadata by one shared rule, and
every node that turns frames into time reads it the same way.

**Producers.**

- `nodegraph/metadata.py` gains `retime(metadata, keep)`, the one subset rule. Crop's frames
  mode, Select Frame, Trim, Batch's `trim` alignment and `zone.frame` (compute and
  meta_transform alike) now call it. The rule:
  - The per-frame lists follow the kept frames.
  - A contiguous run keeps `dt_s`.
  - A pick that skips frames is checked against the kept frames' own clock: an even stride
    multiplies `dt_s` when the clock bears it out and takes the clock's step when the source
    paused.
  - Unevenly spaced kept times drop `dt_s`.
  - A **clockless** series picked unevenly keeps each frame's time in a new per-T key,
    `frame_time_s`. It is added to `PER_TIME_KEYS`, so later subsets carry it.
- `concat_interval` is the concatenation rule. Timeseries Builder (`chained_metadata`, shared
  with its meta_transform) and Merge onto T both use it:
  - An agreed in-file `dt_s` survives only if every step, the gaps between files included,
    matches the clock.
  - A series built from single-frame files gains the clock's step.
  - `frame_time_s` is never concatenated.
  - The builder's card note now says when a gap between files dropped `dt_s`.
- `util.stack` / `stack_time` now drop the per-frame clock together with `dt_s`
  (`drop_time_lists`).
- `zone.frame` / `frame_slice` subset the clock to the sliced frame, instead of leaving it
  full length.

**Consumers.**

- The per-object rate nodes (`analysis.object_metrics`, `object_field`, `track_field`) and
  `analysis.piv` resolve their time base through `_frame_timebase`: the `frame_interval`
  override, then `dt_s`, then each frame's own time (`series_seconds`: the clock, else
  `frame_time_s`), and only then refuse.
- `_object_velocity_nd` divides each step by its real gap when given per-frame times. PIV's
  `_pair_dt_s` does the same per pair, and uses the mean gap for the ensemble.
- The display readers fall back to `frame_time_s` when there is no clock: `figure.frame_times`
  (Plot Time Series), the movie timeline, the Viewer timestamp and T strip, and the Info
  window's time rows.

**Docs.** The MANUAL gains a new section, "The time axis after a T change", and the Trim and
Stack rows are updated. The Trim, Builder and Stack docstrings and option docs are updated to
match.

**Hotspots touched:**

- `nodegraph/selftest.py`: a new `test_time_metadata` after `test_trim`, plus its `main()`
  line. The `test_crop_frames` fixture clock was made consistent with its `dt_s` (it stepped
  one day per frame under a 2 s interval, which the stricter rule correctly rejects).
- `test_trim`'s fixture clock was also made consistent.
- `codemap/gen/*`, `codemap/node_synopsis.json`, `scripts/catalog_baseline.json`:
  regenerated.

## Why

The request was that Trim, and any other node that manipulates the T axis, should rebuild
the time metadata accurately, so that other nodes read it accurately.

Before this change:

- An uneven Trim (`span`, the default) or a hand-picked Crop dropped `dt_s`. Every speed and
  velocity node then **refused**, even though the per-frame clock still held the exact time
  of every kept frame.
- On a clockless series the same cut lost the timing entirely.
- Timeseries Builder in block order kept the first file's in-file `dt_s` across every gap
  between files. That **silently** mis-scaled every rate at the seams, and Merge onto T did
  the same.
- A builder made of one-frame files (a per-frame export) got no interval at all.
- `util.stack` and `zone.frame` left a full-length clock on a one-frame result, so the
  result showed frame 0's time.

One shared rule replaces per-node `time_subset` + `respaced` pairs, so the next T node cannot
diverge.

Rates keep `dt_s` first, so every evenly spaced series computes exactly what it did before:
the interval path is unchanged and bit-identical. The per-frame branch only runs where the
interval is honestly absent. The alternative was to always use the real clock, which is more
accurate for jitter. It would shift every existing speed slightly and break bit-parity, so it
was not done. That decision is the user's call to revisit.

## How to verify

- `PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "import nodegraph.selftest as S; S.test_time_metadata()"`
- In the app: Load an ND2 time series → Trim (Fraction 0.33, span) → segment / track →
  Object Metrics (speed). Speeds come out in µm/s instead of being refused for "no frame
  interval".

## Gates

- [x] Targeted: `test_time_metadata`, `test_trim`, `test_crop_frames`, the builder / merge /
      batch / zone / PIV / track-field / plot / movie tests, plus the four contract tests:
      29/29 pass.
- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (252
      tests incl. `test_time_metadata`, run with `NODEGRAPH_MIN_FREE_RAM=1G`)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (170)
      **Open, undiagnosed:** on the final pushed tree (this change plus the IL-8 / handedness
      work), all 170 probes pass on every run. But the process then hit a Windows access
      violation INSIDE the probe's closing `os._exit(0)`, on the main thread, after the
      PASSED line was flushed: exit 139 on 3 of 5 runs, and a 4th run under
      `-X faulthandler` logged it. Ten idle `runner._loop` threads were still alive at the
      time. At 0ed96d0 the same probe exited cleanly 2 of 2, so the crash may be new in this
      tree, but the sample is small. It was not seen in either earlier probe run with this
      change alone. Measured while `run.py` held ~369 GB, with 55 GB free.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (143 ops, after `save`; the
      only differences were this change's doc text)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT; `_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC — not yet: nothing is committed, and the
      tree also holds other uncommitted work (see `2026-10-16_trim-node.md`).

Not covered, by design:

- An ND2 whose own acquisition paused still gets `dt_s` = the median of its timestamps at
  ingest (`nodelab_v2/ingest.py`), which is a source rule, not a T node.
- The troubleshooting scope pin (`runner._pin_frames`) still passes `dt_s` through unchanged,
  by its documented design.
- `units.py` seconds↔frames conversions still need `dt_s`.
- `util.time_shift` deliberately keeps its grid.
- `analysis.gradient_map` (another session's file) is updating its own clock-key list to
  include `frame_time_s`.
