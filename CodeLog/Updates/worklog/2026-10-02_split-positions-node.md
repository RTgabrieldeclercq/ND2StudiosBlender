# Split Positions + Select Position: fan a multipoint Dataset out per stage position

- **Date:** 2026-10-02
- **Author:** hyper (Claude Fable 5.1 session)
- **Branch:** split-positions-node
- **Base:** e1ae0e2 (origin/Blender)

## What changed

Two new utility nodes, the M-axis twin of Split Channels / Select Channel:

- **Split Positions** (`util.split_positions`) is a pass-through whose card grows one
  synthetic output per stage position (`pos0…`) once the wire carries two or more. The
  sockets are labelled with the acquisition's point names when the file has them
  (`2 · C3`, with the specimen group key appended when known), else `mK`. They are
  resolved from the edit-time envelope (`GraphDocument.position_descriptors`), so the
  count is exact before anything is pulled and a rewire moves the sockets.
- **Select Position** (`util.select_position`) keeps one position by 0-based index or
  point name: M → 1, with the per-M metadata family, lattice layers and structure rows
  following through the same helpers Select Group uses; `meta_transform=select_position`
  predicts it; the sampling stamp records the index. Empty passes through; an unknown
  position is refused with the list; a single-position input is the identity.
- At graph build each wired `posK` is materialized into one `util.select_position` tap
  carrying `position=K`, shared by every branch leaving that socket
  (`ops.materialize_position_taps`, in `prepare_run_graph` between the group and channel
  taps). The tap carries the INDEX, unlike the group/batch taps: a position's index in its
  file is its identity and names are optional labels.

Tests: `test_split_positions` (selection by index/name with layers and rows following,
envelope == payload, refusals, pass-through, document sockets and labels, one shared tap
per socket, a pull per branch). Probe SP1 on the live canvas. Manual rows for both nodes;
both placed under the dataset/channel-organization role; catalog baseline re-blessed
(93 ops).

## Why

The user: "I need a location splitting node to split M channels" — one branch per stage
position (well, dish, field), the way Split Channels gives one branch per channel. The
existing position tools answer different questions: Select Group keeps a specimen (a set
of positions found from the stage log) and Crop's frames mode takes a typed spec. Neither
puts one socket per position on a card. Mirroring the channel-tap arrangement rather than
inventing a multi-output compute keeps the engine's one-payload-per-node rule and reuses
the materializer, the shared-tap memo behaviour and the document's socket validation
unchanged. The tap carries the index because, unlike a batch member or a group, a
position has no guaranteed name, and its index is how every per-M list already addresses
it.

## Files

- `nodegraph/catalog/util/split_positions.py`, `nodegraph/catalog/util/select_position.py` — new (hotspot: `nodegraph/catalog/__init__.py` entries)
- `nodegraph/metadata.py` — `position_pick`, `select_position`
- `nodelab_v2/document.py` — `POSITION_TAP_OPS`, `position_descriptors`, `output_specs` branch
- `nodelab_v2/ops.py` — `POS_SOCKET_RE`, `materialize_position_taps`, `prepare_run_graph` chain
- `nodegraph/selftest.py` — `test_split_positions` (hotspot)
- `scripts/_nodelab_v2_phase5_probe.py` — SP1
- `codemap/node_roles.json`, `MANUAL.md`
- `scripts/catalog_baseline.json` (hotspot, generated), `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_split_positions()"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # SP1
```

In the GUI: load a multipoint ND2, drop Split Positions after it: one output per position,
named. Wire `pos1` into a Viewer: it shows position 1 alone; `out` still carries them all.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole: still red at HEAD for the
  pre-existing reasons in the team-sync entry. Run directly: `test_split_positions`,
  `test_channel_split`, `test_registry`, `test_live_reload_contract`,
  `test_catalog_import_hygiene`, `test_socket_docs`, `test_option_docs`,
  `test_param_socket_contract` pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`, 93 ops)
- [x] `python scripts/_codemap.py` -> gen/ CURRENT (the same 9 curated entries unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> feature branch, level with origin/Blender
