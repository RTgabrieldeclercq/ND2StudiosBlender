# Built-series twitch diagnosed; per-frame acquisition timing on hover and in the movie

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** series-frame-times
- **Base:** e1ae0e28

## What changed

- **Hovering a T frame** in the Viewer's T strip shows when it was acquired (frame 0 = 0),
  the gap from the frame before, and the mean gap over the series. The source is the
  payload's own `frame_time_jd` (`placement.frame_seconds`/`frame_timing`/`frame_timing_lines`,
  `FrameStrip.setFrameTimes`, `ViewerPanel._apply_frame_times`). A clock that doesn't cover
  the strip shows nothing.
- **The Movie Editor uses the real per-frame clock.** Its readout and its timeline's hover
  show the same three lines for the frame (`Timeline.frame_timing`,
  `_TimelineStrip.event`). The burned-in time (both the corner clock and a T-tile caption)
  is `clock[t] - clock[0]` whenever the source carries a full clock (`Timeline._time_at`,
  `MovieSource.frame_s`). `t × Frame interval` is only the fallback. Export Movie's
  `show_time` and `interval_s` descriptions say so.
- **Troubleshooting scope:** the scope no longer cuts the sources on an axis that a node
  upstream of the pulled one re-addresses: Timeseries Builder onto T/M/Z, Time Shift, Select
  Frame, or Merge onto T/M/Z (`runner.readdressed_axes`; `_pin_for(coords, node_id)`). The
  pin carries `None` for that axis and `_pin_frames` reads it as every index of that
  source. The result is cached per page revision for the held-view fast path.
- **Docs.** The builder's `time` doc and MANUAL say an interleaved series can jitter, and
  how to fix it.

## Why

User report: "the timeseries builder appears to make the frames twitch". Checked on
`CRC_1_T1..T5.nd2` (the renamed Seq0000/0003/0006/0009/0012):

- **The order is correct.** Neighbours in the built series (about 33 min apart) are more
  similar than a file's own consecutive frames (163 min apart). Earlier pixel checks
  matched each frame to its source frame.
- **The twitch is in the acquisition.** All five files list the same XY positions
  (within 0.3 µm), but each file's visit to a position lands at its own spot. Each file's
  first frame sits at a different offset from T1's, e.g. (+24, −2) px for T2 at position 20,
  and ~(+20, +25) at positions 1 and 7. So frames from neighbouring files sit 20–30 px apart
  (0.86 µm/px), and the image jumps on every frame. Position 0 of T4 is also focused 68 µm
  below the others.
- **Registration removes it.** Builder → Select Position (20) → Select Plane (1) →
  Stabilize (H2B, previous) cut the consecutive-frame shift from median 22.7 px (max 37.9)
  to 0.5 px (max 2.1).

No code produced the twitch, so the builder's fix is a pointer to Stabilize, not a change to
its order.

- **Real bug found on the way.** The troubleshooting scope cut every source to the Viewer's
  cursor index, but below a re-addressing node that index means something else (frame 7 of
  the series is some file's frame 1), so the shown frame was built from the wrong source
  frames. It is fixed by leaving that axis whole, which is slower to scope and never wrong.
- **Timing.** The hover and movie timing are the user's request. Real per-frame times matter
  here because the merged series is uneven (32–65 min gaps). With `dt_s` dropped, the movie
  showed only "t N"; with an interval typed in, it showed wrong times.

## Files

- `nodegraph/placement.py`: `frame_seconds`, `frame_timing`, `frame_timing_lines`
- `nodelab_v2/framestrip.py`: `setFrameTimes`/`frameTimes`; the T tooltip; a re-range drops the clock
- `nodelab_v2/viewer.py`: `_apply_frame_times`
- `nodegraph/catalog/_shared/movie_timeline.py`: `MovieSource.frame_s`, `Timeline._time_at`, `Timeline.frame_timing`
- `nodelab_v2/movie_editor.py`: readout timing, timeline hover
- `nodelab_v2/runner.py`: `readdressed_axes`, `_READDRESSING`, `_pin_for(node_id)`, `_readdressed`, `_pin_frames` `None` axes
- `nodegraph/catalog/io/write_movie.py`, `nodegraph/catalog/util/timeseries.py`: descriptions
- `nodegraph/selftest.py`: `test_frame_timing` (hotspot: appended before `main`); `test_multilink`'s summary corrected
- `scripts/_nodelab_v2_phase5_probe.py`: FT1
- `MANUAL.md`: the strip hover, the troubleshooting bullet, the Movie Editor's real acquisition time, and the builder jitter note
- `scripts/catalog_baseline.json` (hotspot, `save`: 3 description diffs), `codemap/node_synopsis.json`, `codemap/gen/*` + `STATE.md` (regenerated). The diff was read: 13 new symbols, `_pin_for` changed, 2 lazy `placement` imports, and every node's closure fingerprint (placement is in every closure)

## How to verify

`test_frame_timing`; probe FT1. On the real files: build CRC_1_T1..T5 with the Timeseries
Builder (T, time) and hover T 3: "acquired 02:43:19 after frame 0 · Δ 00:33:15 after
frame 2 · mean Δ 00:34:24 over 37 intervals". Add Registration → Stabilize after the builder
and the twitch is gone. The scratch scripts behind the numbers are `crc_twitch.py`,
`crc_offsets.py` and `crc_stabilize.py`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (FT1 included)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`; 3 description diffs)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT (after `write`)
- [ ] `python scripts/_sync_check.py` -> not IN SYNC by design (stacked on node-mirror-merge and the branches under it); 0 behind `origin/Blender`
