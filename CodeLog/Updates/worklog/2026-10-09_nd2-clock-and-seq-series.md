# ND2 per-frame clock fixed; the CRC_TF_ELISA _Seq series builds in time order

- **Date:** 2026-10-09
- **Author:** Gabriel Declercq
- **Branch:** timeseries-per-frame-clock
- **Base:** e1ae0e28

## What changed

Three things, all needed for the user's files
(`G:\McGhee Lab\CRC_TF_ELISA\20261006_194721_215\Channelp53-GFP,ERK-mRuby2,H2B-iRFP670_Seq0000/0003/0006/0009/0012.nd2`)
to become one series in acquisition order:

1. **The ND2 reader's frame index** (`nodelab_v2/nd2_meta.py`, new `sequence_indexer`). It
   now looks up the sequence index in the file's own `loop_indices`, falling back to
   row-major arithmetic over the LOOP axes only. It used to fold `C` into the index as if
   channels were the innermost loop. On T=8/P=40/Z=3/C=3 that read timepoint `t` at frame
   `t*360` instead of `t*120`, ran off the end at t=3, and dropped the whole per-T clock as
   partial. So these files, and any long enough multichannel time-lapse, carried no
   `frame_time_jd`, no `frame_datetime` and no `dt_s`. Fixing it restores all three.
   The same index feeds the per-frame stage-position fallback.
2. **`dt_s` on an interleaved build** (`metadata.chained_metadata`, new
   `chain_uniform_step_s`). When the Timeseries Builder interleaves files, `dt_s` becomes
   the merged clock's single step if every step is within 10% of the median, and is dropped
   otherwise (the `respaced` rule). The card's note says when it was dropped. Block order
   still leaves `dt_s` alone.
3. **The sequence loader's separate-cards fallback keeps the builder**
   (`window._load_source_paths`). When a sequence cannot share one card, "Load as separate
   cards" now wires each card into a Timeseries Builder slot in pick order and publishes the
   builder's output, not each card. More files than the builder has slots (12) load
   unwired, and the status bar says why.

## Why

User report: "It didn't work" on the five `_Seq` files after the per-frame interleave
landed. A metadata survey showed every file's reader clock was EMPTY (`frame_time_jd`
length 0), although every frame carries `absoluteJulianDayNumber`. With no clock, `time`
order fell back to filenames and played the files as blocks. The files are a textbook
interleave: each is its own 7–8 point time-lapse (~163 min apart), each starting ~65 min
after the previous, so the true order weaves through all five. The files hold 8, 8, 8, 7, 7
frames and cannot share one card, and the loader's fallback then deliberately dropped the
builder. That rule predates the builder's per-file slots (2026-10-02), so it was stale.
`dt_s` had to change with the reader fix. With the clock restored, each file's `dt_s`
(~9790 s) appears for the first time, and an interleaved series built from them is 32–65 min
apart frame to frame. Leaving the in-file `dt_s` on it would make every speed downstream 3–5x
too slow, so it is dropped as uneven.

Known, accepted: the edit-time envelope carries no clock (it never has; the clock rides
the payload's display dict), so a builder's CARD can show the in-file `dt_s` until the
first pull replaces it.

## Files

- `nodelab_v2/nd2_meta.py`: `sequence_indexer`; both `frame_metadata` lookups guard a `None` index
- `nodegraph/metadata.py`: `chain_uniform_step_s`, `CHAIN_UNIFORM_STEP_TOL`, the `dt_s` rule in `chained_metadata`
- `nodegraph/catalog/util/timeseries.py`: the note says when `dt_s` was dropped
- `nodelab_v2/window.py`: the separate-cards fallback wires a Timeseries Builder
- `nodegraph/selftest.py`: `test_timeseries_frame_interleave` sections 6–7 (hotspot: inside the test appended earlier today)
- `scripts/_nodelab_v2_phase5_probe.py`: probe SQ2 (appended before the final print)
- `MANUAL.md`: the sequence-fallback sentence and the builder's `dt_s` sentence

## How to verify

Real data (read-only; uses the no-copy ND2 provider): a scratch script fed the five files,
wired Seq0000 first, through `util.timeseries` (T, time). It gave T=38, `order=time`,
`interleaved=true`, and a monotone clock 2026-10-07 03:27:32 → 2026-10-08 00:40:13 in a
steady Seq0000 → 0009 → 0003 → 0012 → 0006 rotation. `dt_s` was dropped (steps 32–65 min),
and output pixels at t = 0–6, 17 and 37 were byte-equal to the frames the note named.
Synthetic: `test_timeseries_frame_interleave` §6–7; probe SQ2.

## Gates

See `2026-10-09_bundle-stage-log.md`: the gates ran once on the branch tip, which
contains this change.
