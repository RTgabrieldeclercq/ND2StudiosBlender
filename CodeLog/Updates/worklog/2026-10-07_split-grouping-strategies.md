# Split grouping strategies

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** graph-node-sets
- **Base:** e1ae0e28

## What changed

Grouping on the four split cards (Split Channels / Positions / Z / T) is now a **Mode** with a visible strategy and its own parameter, declared once in `nodegraph/catalog/_shared/split_grouping.py` and forwarded by the four nodes: `grouping` = `none` (one output per index, up to the fan-out cap — the family's original behaviour), `every` (consecutive groups of `Group size`; defaults 2 / 2 / 4 / 10 for channels / positions / planes / frames) or `ranges` (the groups typed into `Ranges`, `0-3; 4-7`, optionally named, `every 4` accepted). Each strategy shows only its own socket (`available_in`), so the Properties panel offers a dropdown and one field rather than a bare `groups` text box; the Mode and both sockets are presentation-only, as the text was. One shared resolver, `metadata.split_plan` (with the axis length, for the card) and `split_plan_group` (without it, for the run-graph build), replaces the two ad-hoc readings in `document.split_groups` and `ops.split_group`, and both resolve the `group_size` socket's declared default when the user has not touched it — a node's params hold only what changed. `ops.split_group` accepts a document `NodeRecord` as well as a run-graph `NodeInstance`. Tests choose the Mode where they typed the text and gain assertions for the `every` strategy through the dropdown and the default; the probe drives the Properties panel and asserts the dropdown; MANUAL rows, the §16 C recipe and the four demos' features describe the strategies.

## Why

The owner: "none of the group strategies are possible in any of the splitting nodes — it's just not built in the node properties panel; we need grouping strategies." The previous commits exposed grouping as one STRING socket named `groups`, which the panel showed as an unlabelled text box with no hint of what to type, and `every N` only as a grammar inside it — a strategy hidden in a syntax. A Mode is the catalog's way of offering a closed set of approaches: it renders as a dropdown on the card and in the panel, each choice carries its own explanation (`choice_docs`), and `available_in` shows only the parameter the chosen strategy reads, so the panel reads `Grouping: every · Group size: 10` instead of a box. Declaring it once in `_shared/` keeps the four cards identical, and one resolver on both the card and the run-graph side is what keeps a socket and the tap it becomes from ever disagreeing. `count` (N equal groups) was considered and left out: the run-graph build has no axis length, so it cannot compute chunk k without an envelope-aware materializer; `every` with a size covers the same need.

## Files

- `nodegraph/catalog/channel/split.py`
- `nodegraph/catalog/util/split_positions.py`
- `nodegraph/catalog/util/split_t.py`
- `nodegraph/catalog/util/split_z.py`
- `nodegraph/metadata.py`
- `nodelab_v2/document.py`
- `nodelab_v2/ops.py`
- `nodegraph/catalog/_shared/split_grouping.py`

## How to verify

`python run.py` → a Split Z (or T / Positions / Channels) card → Properties → **Grouping**: `none` shows per-index sockets (up to 24); `every` shows **Group size** and the card regrows as groups of that many; `ranges` shows **Ranges** — type `top: 0-3; 4-11`. Headless: `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_split_groups(); S.test_select_frame_split_t()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (encoder); the groups scheduled after it were run separately and pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the exit code after the PASSED line is the probe's documented teardown crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 114 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> not pushed; on `graph-node-sets`, the push is the owner's call

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
