# Split groups

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** registration-workflow-nodes
- **Base:** e1ae0e28

## What changed

Every split card — **Split Channels**, **Split Positions**, **Split Z** — gains a `Groups` text: ranges of the axis to split into, `0-3; 4-7; 8-11`, `;` (or a newline) between groups, each a list or inclusive range of 0-based indices, with an optional name in front (`top: 0-3`). With it set the card grows one output **per group** (`chg0…`, `posg0…`, `zg0…`) in place of the per-index sockets, labelled by the typed name or the members as the axis knows them — channel names joined with `+`, first–last position name, `z 4-7` with its height span — and a group that reaches past the end says `(past the end)`. A wired group materializes at graph-build into one shared tap per group: `channel.select` with the list for a channel group, `util.crop` in frames mode (`m0-1` / `z4-7`) for a position or plane group — the node that already carries every per-axis list, spacing and origin rule a frame subset needs and refuses past the end with the real length. The socket is **presentation-only**: it shapes the card and the taps, never the split's own `out`, so it stays out of the recipe hash and the pass-through memoizes as before. The grammar is `metadata.parse_groups` (Qt-free, total, out-of-range kept rather than clipped), beside `parse_indices`; the GUI half is `GraphDocument.split_groups`, three branches in `output_specs`, three `_tap_name` rows, and `ops.materialize_channel_groups` / `materialize_position_groups` / `materialize_plane_groups` (`_materialize_taps` learnt an optional `modes`). MANUAL §15 rows and the §16 C recipe, the three demos' features, a selftest and a GUI probe block.

## Why

The request: for any of the split nodes, select a RANGE of the data to split into groups — a 60-plane stack as six sub-stacks of ten, a plate as its wells, a channel pair kept together — rather than one socket per index, which on a deep stack is unusable and on a plate is wrong-grained. Groups REPLACE the per-index sockets when set (rather than sitting beside them) because the point of grouping is a card you can read; blank brings the singles back. A position or plane group is a `util.crop` frames tap, not a widened Select Position / Select Plane, because Crop's frames mode already implements the whole frame-subset contract — per-M lists, `z_step_um`, `z_home_index`, `origin_um`, structure rows renumbered — and a second implementation of that rule is how two nodes come to disagree about where a plane sits. A channel group is a `channel.select` list, which that node already took. The group socket carries the group INDEX and the materializer re-reads the card's text, so the run graph reads exactly as the card does and retyping a group re-keys only its tap. Out-of-range indices are kept, labelled and refused at the pull rather than clipped: `8-11` on a 10-plane stack must not quietly become `8-9`. Not covered: Unbatch (`util.unbatch`) is not a Split card and its tap, Select Batch, takes one member; a batch group would need a multi-member select first.

## Files

- Engine: `nodegraph/metadata.py` (`parse_groups`), `nodegraph/catalog/channel/split.py`, `nodegraph/catalog/util/split_positions.py`, `nodegraph/catalog/util/split_z.py` (the `groups` presentation socket each)
- GUI: `nodelab_v2/ops.py` (`CHG_/POSG_/ZG_SOCKET_RE`, `split_group`, three materializers, `_materialize_taps(modes=)`, the run-graph order), `nodelab_v2/document.py` (`split_groups`, `output_specs`, `_tap_name`)
- Records: `MANUAL.md`, `codemap/node_demos.json`, `codemap/gen/*` + `codemap/STATE.md` + `codemap/node_synopsis.json` + `scripts/catalog_baseline.json` (regenerated)
- Tests: `nodegraph/selftest.py` (`test_split_groups`, one line in `main()`), `scripts/_nodelab_v2_phase5_probe.py` (SG1)
- Hotspots: `nodegraph/selftest.py` (one group appended before `test_shift_node`), `codemap/gen/*` and `scripts/catalog_baseline.json` (regenerated, never hand-merged)

## How to verify

`python run.py` → Load a z-stack → **Split Z** → type `0-3; 4-7; 8-11` into **Groups**: the card shows `zg0…zg2` labelled with the plane heights; wire one into a Viewer and it shows a 4-plane sub-stack; `top: 0-3` names a socket. The same on Split Positions (`wells: 0-5; 6-11`) and Split Channels (`0,2; 1`). Headless: `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_split_groups()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (fails identically on the base commit, encoder); the groups after it run and pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (SG1 included; the exit code after the PASSED line is the probe's documented teardown crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 111 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> not pushed: on `registration-workflow-nodes`; the push and the rebase onto `origin/Blender` are the owner's call

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
