# Time shift node

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** graph-node-sets
- **Base:** e1ae0e28

## What changed

A new node, **Time Shift** (`util.time_shift`, utility): move a data stream along its own time axis by a whole number of frames, later or earlier — frame k of the output is frame k − Δ of the input. T, `dt_s` and the per-frame clock are untouched (the grid is the same; the content moved on it). `Delta` is a magnitude in frames; the sign is the `direction` Mode (`later` delays, `earlier` advances) because the card's numeric controls are non-negative; the `edges` Mode says what the frames with no source show (`hold` repeats the nearest edge frame, `blank` zeroes them). Masks and label rasters move frame for frame with the image; Point / Label / Track rows move their t and rows shifted off either end are dropped, never duplicated; a mesh that would lose rows is dropped whole. Δ = 0 and a single frame are the identity; a Δ past the length holds the edge everywhere. It is lazy: a new `FrameRemapProvider` (an index remap that, unlike `FrameSubsetProvider`, allows repeats and blanks) plus two shared helpers, `remap_lattice_layers` and `shift_structure_rows`. Registered in the registration & alignment role beside Shift, with a demo on the drifting star field (Δ = 2, viewed at frame 3), a MANUAL row and a selftest group.

## Why

The owner: "i want a node that modifies a data stream by adjusting time + or − a delta". Shift (2026-10-07) moves the content of every frame under a fixed pixel grid; this is the same operation one axis over, under a fixed time grid — what lines up two streams of one acquisition for a Merge or a comparison when one lags the other by a few frames (a second camera or channel starting late). Whole frames only: a fractional shift would mean interpolating in time, which is a different node. The grid metadata is deliberately NOT moved: a frame's slot keeps its time, the content in the slot changed, so downstream pairing by index and by `frame_time_jd` stay consistent. A direction Mode rather than a signed number because every numeric control in the GUI clamps at 0 (spin boxes and the card's scrub alike), so a negative Delta could not be entered. `hold` is the default because padding frames that look like the series read naturally when scrubbing; `blank` exists for measurements that must not count a repeated frame's objects. `FrameSubsetProvider` could not carry it: it sorts and de-duplicates its picks on purpose (a subset is a sampling), and a held edge frame is frame 0 three times.

## Files

- New: `nodegraph/catalog/util/time_shift.py` (the node; `time_shift_sources`, `signed_delta` shared with its test)
- Engine: `nodegraph/provider.py` (`FrameRemapProvider`), `nodegraph/catalog/_shared/frame_subset.py` (`remap_lattice_layers`, `shift_structure_rows`), `nodegraph/catalog/__init__.py` (MODULES)
- Records: `codemap/node_roles.json` (registration_and_alignment ops + description; op_pages refine/process), `codemap/node_demos.json`, `MANUAL.md` (Transform & utility row), `codemap/gen/*` + `codemap/STATE.md` + `codemap/node_synopsis.json` + `scripts/catalog_baseline.json` (regenerated)
- Tests: `nodegraph/selftest.py` (`test_time_shift`, registered after `test_select_frame_split_t`)
- Hotspots: `nodegraph/catalog/__init__.py` (one appended MODULES entry), `nodegraph/selftest.py` (one new group), `codemap/gen/*` and `scripts/catalog_baseline.json` (regenerated, never hand-merged)

## How to verify

`python run.py` → Load a time series → Time Shift (Refine page, under Registration & alignment) → Delta 2, Direction later: scrub Frame t — the series is two frames late, the first two frames repeat frame 0 (Edges hold) or are dark (blank). Its `?` demo shows the drifting beads two frames back at frame 3. Headless: `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_time_shift()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (encoder); the groups scheduled after it were run separately and pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (no GUI code changed; run as the gate)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 115 ops (baseline re-saved for the new node)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> not pushed; on `graph-node-sets`, the push is the owner's call

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
