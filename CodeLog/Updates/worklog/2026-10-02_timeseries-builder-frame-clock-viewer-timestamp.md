# Per-frame timestamps on ND2 import; Chain Files → Timeseries Builder (slots, clock order); Viewer timestamp

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** team-sync-roles-palette-viewer
- **Base:** fd5c7a5 (origin/Blender) + the ten commits of this branch

## What changed

**1. The ND2 reader writes a readable per-frame timestamp.** Beside the per-T Julian-day
clock it already read (`frame_time_jd`), the extended reader now writes `frame_datetime`:
one `YYYY-MM-DD HH:MM:SS.mmm` per timepoint, derived once from the Julian day by the new
Qt-free `placement.jd_to_datetime_text` (J2000.0 → `2000-01-01 12:00:00.000`; the wall
clock as the microscope wrote it, no time-zone shift). It is a `PLACEMENT_KEYS` member so it
reaches the Dataset, and a `PER_TIME_KEYS` member so every T subset, scope and chain
reindexes it in lockstep with the clock — a frame can never show one time and pair by
another. `placement.elapsed_text` is the one span-aware elapsed formatter; Export Movie's
counter now calls it.

**2. Chain Files is now the Timeseries Builder** (`util.timeseries`, module
`catalog/util/timeseries.py`). Its single multi-input socket became **slots that reveal one
at a time** — `File 1` (`data`) then `File 2…File 12` via the Viewer's `grow_group` rule —
so files are collected one per slot with always exactly one empty slot waiting. `File
order` gained **`time`, now the default**: files are sorted by their first frame's absolute
clock (`metadata.chain_member_clock`), whatever order they were wired in and whatever their
names say; where any file has no clock it falls back to the filename sequence, then to
wiring order, and `metadata.chain_order_used` names the rule that applied. The stamped
`__chain__` note now carries `order` as the rule actually used, each file's `first_frame`
timestamp and a sentence (`N files in acquisition-clock order, <first> → <last>`). The
output carries the concatenated `frame_time_jd` and `frame_datetime`, monotone under
`time`. A saved graph naming `util.chain` loads as `util.timeseries`
(`GraphDocument._OP_RENAMES`, beside the existing socket renames); the Load-file-sequence
dialog drops the new node; `util.stitch`'s diagnostic names the new node.

**3. The Viewer node gained a timestamp overlay.** Four presentation sockets: `Timestamp`
on/off (off by default), `Timestamp shows` — `elapsed` (default: seconds since the series'
first frame from the payload's own `frame_time_jd`, so a built series shows real gaps;
falls back to `dt_s × frame`, then to the frame number), `clock` (`frame_datetime[t]`,
blank when the payload has no absolute clock), `frame` (`t 7/120`), `elapsed_clock` (both)
— `Timestamp corner` (top-left default; shares the bottom-right corner with the scale bar
by moving up a line) and `Timestamp colour`. The window reads them from the document and
pushes them with each result; the viewer paints the text with a shadow under the overlays.
Indexed by the PAYLOAD's T, so the solo scope's subset lists line up.

Tests: `test_timeseries_clock_order` (formatters; clock order against names and wiring;
monotone concatenated clocks; `frame_datetime` in lockstep incl. `time_subset`; `sequence`
follows names; a clockless file falls back and the note says so; envelope == pull; slots
reveal one at a time; `util.chain` loads as `util.timeseries`; refusal). `test_chain_files`
rewired to the slots. Probe VN1 extended with the timestamp's four modes and its geometry.
Manual: Timeseries Builder row rewritten, Viewer row extended. Catalog baseline re-blessed
(91 ops). Roles and the concepts card renamed the op.

## Why

The user: ensure the time stamp is added to the T-frame metadata in full
year/month/day/hour/second/millisecond form when importing ND2 files; rescope and rename
Chain Files into a Timeseries Builder that takes multiple file inputs the way the Viewer
collects sources, and — since the frames' timestamps are known — auto-sorts everything
placed in it by time so the series is right; and give the Viewer node a timestamp with a
choice of what to show, elapsed time by default.

The timestamp is derived once in the reader rather than wherever it is displayed because
two consumers deriving it separately would eventually disagree by a rounding; it is a
per-T list so the existing lockstep machinery (`time_subset`, `chained_metadata`) carries it
for free. Clock order is the default because it is the only order that is true by
construction — names and wiring are conventions — and the fallbacks are explicit and
stamped because a silently wrong frame order is the failure this node exists to prevent.
The slots replace the multi-input socket so the card shows what is wired, one line per
file, and so the readiness/document rules written for the Viewer apply unchanged. The
op key was renamed rather than relabelled because the node's contract changed (default
order, input shape); the loader alias keeps old files opening. The Viewer's elapsed time
reads the per-frame clock rather than `dt_s × t` because a Timeseries Builder's output is
exactly the case where the interval is not constant.

## Files

- `nodegraph/placement.py` — `JD_UNIX_EPOCH`, `jd_to_datetime_text`, `elapsed_text`
- `nodelab_v2/nd2_meta.py` — writes `frame_datetime`
- `nodelab_v2/ingest.py` — `PLACEMENT_KEYS` + `frame_datetime`
- `nodegraph/metadata.py` — `PER_TIME_KEYS` + `frame_datetime`; `CHAIN_ORDERS`, `chain_member_clock`, `chain_order_used`; `chain_members(order=)`; `chain_grow` passes the order
- `nodegraph/catalog/util/timeseries.py` — renamed from `chain.py`, slots + clock order (hotspot: `nodegraph/catalog/__init__.py` entry renamed)
- `nodegraph/catalog/util/stitch.py` — message names the new node
- `nodegraph/catalog/_shared/movie_draw.py` — `_time_text` calls `elapsed_text`
- `nodelab_v2/document.py` — `_OP_RENAMES` applied on load
- `nodelab_v2/ops.py` — Viewer timestamp sockets
- `nodelab_v2/window.py` — `_viewer_timestamp`, pushed per result; sequence loader drops `util.timeseries`
- `nodelab_v2/viewer.py` — `set_timestamp`, `timestamp_text`, `_timestamp_geometry`, `_paint_timestamp`
- `nodelab_v2/sequence_dialog.py`, `nodelab_v2/runner.py` — comments
- `nodegraph/selftest.py` — `test_timeseries_clock_order`, `test_chain_files` rewired (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — VN1 timestamp checks
- `codemap/node_roles.json`, `codemap/concepts.md`, `MANUAL.md`
- `scripts/catalog_baseline.json` (hotspot, generated), `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_timeseries_clock_order(); s.test_chain_files()"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # VN1
```

In the GUI: load three ND2 chunks of one timelapse in any order, drop a Timeseries Builder,
wire each into the next free slot; the card's note reads `3 files in acquisition-clock
order, <first> → <last>`. Wire a Viewer after it, turn `Timestamp` on: the top-left reads
the elapsed time of the viewed frame, with real gaps between chunks; switch it to `clock`
for the acquisition date and time.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole: still red at HEAD for the
  pre-existing reasons in the team-sync entry. Run directly: `test_registry`,
  `test_live_reload_contract`, `test_catalog_import_hygiene`, `test_socket_docs`,
  `test_option_docs`, `test_chain_files`, `test_viewer_node_inputs`,
  `test_timeseries_clock_order` pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`, 91 ops)
- [x] `python scripts/_codemap.py` -> gen/ CURRENT (the same 9 curated entries unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> on the pushed feature branch, level with origin
