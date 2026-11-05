# ND2 acquisition clock, time zone, and the full acquisition record

- **Date:** 2026-10-13
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock
- **Base:** 572312ab

## What changed

**The per-frame clock.** Every ND2 frame's time (`frame_time_jd`) is now rebuilt from the
file's one stored absolute stamp: `jd(frame k) = jd0 + (relativeTimeMs[k] - relativeTimeMs[0])`.
The new module `nodelab_v2/nd2_acquisition.py` does this, called from the vendored reader. A
Timeseries Builder over an NIS JOBS series (CRC_1_T1..T5) now plays the files round robin
(T1 t0, T2 t0 … T5 t0, T1 t1 …) in true time order, every step ~32.7 min forward.

**Local time.** All wall-clock text is the microscope's local time:
- `frame_datetime`, `acquisition_start`, the Viewer's clock overlay, the Builder's note and
  the plot clock axes.
- The offset is read from the file's own `date`, stored as `utc_offset_s`
  (`placement.UTC_OFFSET_KEY`), and applied through `jd_to_datetime_text(jd, offset)`.
- `acquisition_start` is now the first frame's time. It used to be the file's close date.

**The full acquisition record.** Each file's settings ride the payload as `acquisition_info`
(sections), `channel_acquisition` (per channel, a `PER_CHANNEL_KEYS` member) and
`acquisition_text` (raw NIS text). The record covers laser line and power %, detector, gain,
offset, pinhole, emission range, dwell time, scanner, objective, loops, per-frame recorded
data and software version. In the Viewer:
- hovering a channel button lists that channel's settings;
- the new **Info** button opens a window with the series' own frame times and everything
  the file recorded, plus the raw text and *Copy all*.

**Export TIFF writes the same times.**
- OME: `AcquisitionDate` with its UTC offset, a per-plane `DeltaT` from the real clock (no
  longer `t x dt_s`), every frame's local time and the acquisition settings as
  `MapAnnotation`s, and per-channel excitation wavelength.
- ImageJ: each slice's label is its frame's time, and `Info` lists all frame times and the
  settings.

**Viewer hover.**
- A failed pull no longer sets its traceback as the image's tooltip. That tooltip was never
  cleared, so it popped up on every later hover.
- The pixel readout and the T-strip tooltip can no longer raise.
- The T-strip frame times and the channel tooltips refresh on every payload. They used to
  refresh only when axis sizes or channel names changed.

**Troubleshooting pin.** A troubleshooting T pin now subsets the seed's per-T clock with its
frames (`runner.pinned_seed_metadata`), and the overlay readout requires a clock of exactly T
entries.

## Why

The user reported the CRC_1 monolayer "twitching": growing, then going back to an earlier
state. Reading the five files' raw metadata showed the cause.

**What the files store.**
- An ND2 has one absolute stamp: frame 0's `dTimeAbsolute`, in UTC.
- Every frame has a `relativeTimeMs`, measured from the start of the NIS **run**. For a JOBS
  run that writes several files, that start is shared by all of them.
- Proof: on all five files, frame 0's time minus the JOBS folder's name (19:47:21.215)
  equals frame 0's `relativeTimeMs` to the millisecond.

**Why the library is wrong.** `nd2` 0.11.3 builds each frame's `absoluteJulianDayNumber` as
`jd0 + relativeTimeMs`, adding frame 0's offset into the run twice. That is 20–151 min,
different per file. The Builder sorted on it and played true frames 0,1,2,5,3,6,4,7,10,8…,
fourteen steps back in time.

**Independent checks of the corrected clock.**
- Each file's own `text_info["date"]` (local, written at close) equals the corrected time
  of its last frame to the second.
- The files' modification times (16:18) match the corrected end of the run.
- On the WellA3 pair the library's 2414.6 s gap would put the GFP sweep inside the 640's
  *next* sweep. The corrected 1207.3 s puts it 11 s after the 640 sweep ends.
- The old selftest fixture and code comments had enshrined the doubled number as proof
  that the JD was the shared clock. They are corrected.

**Why fix it at the reader.** Fixing it once in the reader, rather than patching the
Builder, puts every consumer (Viewer, plots, tables, export, overlay pairing) on the true
clock. The fix is a no-op on ordinary single-file acquisitions, where `relativeTimeMs[0]`
is ~0.

**The time zone.** The JD is UTC, and the code had printed it as local time (7 h ahead in
Arizona).

**The rest of the request.** The user asked for:
- all file metadata (laser %, gain, optics per channel) to be visible, with channel
  metadata on hover;
- TIFF exports to carry time exactly as the app shows it;
- no errors or tracebacks when hovering the viewer;
- the Builder to keep working when frame 37 is cropped out. The stopped run's last
  timepoint is acquired only through position 9; positions 10–39 are zeros.

## How to verify

- Load `G:\McGhee Lab\CRC_TF_ELISA\20261006_194721_215\CRC_1_T1..T5.nd2` into one
  Timeseries Builder.
- Scrub T: the monolayer only grows.
- The T-strip hover and Info window show 2026-10-06 20:07:27 → 2026-10-07 16:15:05
  (UTC-07:00), ~32.7 min apart.
- Add **Crop** (frames, `t0-36`) after the builder: 37 frames, last 2026-10-07 15:42:26.
- Hover a channel button for its laser/gain rows.
- Export TIFF and open the file in Fiji: Image ▸ Show Info.
- `test_nd2_true_clock` gates all of this, against the real files when this machine can see
  them.

## Files

Hotspots: `nodegraph/selftest.py` (new `test_nd2_true_clock`; WellA3 fixture corrected),
`codemap/gen/*`, `scripts/catalog_baseline.json` (write_tiff mode docs only).

- `nodelab_v2/nd2_acquisition.py` (new): the clock, zone, settings parser, acquisition record
- `nodelab_v2/acquisition_view.py` (new, Qt-free): channel tooltip, series time, Info sections
- `nodelab_v2/acquisition_dialog.py` (new): the Viewer's Info window
- `nodelab_v2/nd2_meta.py`: true clock, zone, local `frame_datetime`/`acquisition_start`, record
- `nodelab_v2/ingest.py`: `ACQUISITION_KEYS` whitelist; corrected clock docs
- `nodelab_v2/runner.py`: `pinned_seed_metadata` (T pin subsets the clock)
- `nodelab_v2/viewer.py`: Info button, channel tips, no traceback tooltip, strict clock, safe hover
- `nodelab_v2/framestrip.py`: tooltip can no longer raise
- `nodelab_v2/window.py`: comment only
- `nodegraph/placement.py`: `UTC_OFFSET_KEY`, `utc_offset_of`, `jd_to_datetime_text(jd, offset)`
- `nodegraph/metadata.py`: `channel_acquisition` in `PER_CHANNEL_KEYS`; Builder's local `frame_datetime`
- `nodegraph/catalog/util/timeseries.py`: note and `first_frame` in local time
- `nodegraph/catalog/io/write_tiff.py`: real-clock DeltaT, AcquisitionDate, annotations, ImageJ labels/Info
- `nodegraph/catalog/_shared/figure.py`, `nodegraph/catalog/plot/timeseries.py`: local clock axes, offset rides pictures
- `scripts/_nodelab_v2_phase5_probe.py`: ACQ probe
- `MANUAL.md`, `codemap/workflows.md` (WF-07 re-blessed), `codemap/*`, `scripts/catalog_baseline.json`

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (exit 0)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED, incl. the new ACQ probe. The process then exits 139 (a native fault after the final print, at os._exit) — identical on the untouched base commit 889d9d4, so pre-existing and not this change
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (re-saved for the write_tiff mode docs)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (WF-07 re-read, extended, blessed)
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

## Not done / caveats

- A Dock checkpoint baked before this change keeps the old order and clock in its manifest
  (`dock_signature` is graph-only). Re-bake any dock taken from a JOBS series.
- `channel_colors` still comes back empty on modern ND2s: the vendored reader asks for
  `colorRGB`, which nd2 0.11 replaced with a `Color` object. It was deliberately left alone
  because fixing it would change every channel's default tint. The NIS colour is shown in
  the channel tooltip.
- The plain TIFF model still writes no metadata by its own contract.
- Upstream: the double-count is in `nd2` (`_parse.py` `load_frame_metadata`) and worth
  reporting to tlambert03/nd2.
