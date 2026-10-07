# Split group selected

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** graph-node-sets
- **Base:** e1ae0e28

## What changed

Grouping on the four split cards (Split Channels / Positions / Z / T) is now done by SELECTION: every data output on the card has a checkbox, and a **Group selected** button under the outputs turns the ticked ones into ONE group socket carrying that subset — sitting where the first ticked member was, beside the outputs left ungrouped (`z0 z1 z2 z3 zg0 z8 … z11`); the ticked sockets vanish and every wire that left one of them now leaves the new socket. Ticking a group offers **Ungroup**, which dissolves it (the groups after it renumber, and the wires follow). The Properties panel shows the same list with checkboxes — every member, even past the 24-socket fan-out cap — the two buttons with an optional name, **Group every N** for a long axis, and the `groups` text itself for typing. The `grouping` Mode (`none` / `every` / `ranges`) and the `group_size` socket from the previous commit are gone; `groups` is the one socket, and it is what the gestures write (`metadata.format_groups`, canonical and round-tripping through `parse_groups`). The document owns the whole thing Qt-free (`split_axis`, `split_items`, `split_pick`/`split_picked`, `split_group_selected`, `split_ungroup`, `split_group_every`, `_split_rewrite` for the edge moves) so the selftest covers it; the card hides the raw `groups` pill, since the checkboxes are the control there.

## Why

The owner, on the strategies commit: "you are really bad at making a grouping strategy. the range idea doesn't work, the every mode does, but it is not intuitive to use. Make grouping easy to use. make a check box next to each data out, and make a button called group selected." Typing ranges failed for the reason a headless replay shows: the grammar is `;` between groups and `,` inside one, so `0-3, 4-7` — what anyone types — is ONE group, and `z0-3; z4-7` is nothing; the box gave no hint either way. A strategy chosen from a dropdown, then a size or a text, is indirect twice over; the thing being grouped is already on the card as sockets, so the control belongs on them. Checkbox + button is direct manipulation: what you see is what you group, the wires you already drew come along, and the text becomes storage rather than interface. `every` survives as a bulk button in the panel because it is still the only sane way to window 300 frames, but it now writes explicit groups, so there is one representation and one resolver on both the card and the run-graph side. Mixed sockets (groups beside ungrouped members) are what make the gesture incremental — group four planes without losing the other eight.

## Files

- Engine: `nodegraph/catalog/_shared/split_grouping.py` (one `groups` socket, no Mode), the four split nodes, `nodegraph/metadata.py` (`format_groups`; `split_plan`/`split_plan_group` removed)
- GUI model: `nodelab_v2/document.py` (`_split_picks`, `split_axis`, `split_items`, `split_pick`, `split_picked`, `split_group_selected`, `split_ungroup`, `split_group_every`, `_split_rewrite`, `_split_outputs`), `nodelab_v2/ops.py` (`split_group` reads the text)
- GUI: `nodelab_v2/node_item.py` (`check` / `group` / `ungroup` controls, the split row, `_card_inputs`), `nodelab_v2/inspector.py` (`_split_grouping_box`)
- Tests: `nodegraph/selftest.py` (`test_split_pick_group` new; `test_split_groups`, `test_select_frame_split_t` without the Mode), `scripts/_nodelab_v2_phase5_probe.py` (SG1 drives the card and the panel; ST1 uses Group every)
- Records: `MANUAL.md`, `codemap/node_demos.json`, `codemap/gen/*` + `codemap/STATE.md` + `codemap/node_synopsis.json` + `scripts/catalog_baseline.json` (regenerated)
- Hotspots: `nodegraph/selftest.py` (edits inside the 2026-10-07 groups + one new group), `codemap/gen/*` and `scripts/catalog_baseline.json` (regenerated, never hand-merged)

## How to verify

`python run.py` → a Split Z (or T / Positions / Channels) card wired to a source → tick two or more plane checkboxes on the card → **Group selected**: the ticked sockets become one, labelled `0 · z 4-7 · 1.60–2.80 µm`, the rest stay. Tick the group → **Ungroup**. Properties panel: the same list, a name box, Group every N → Apply. Headless: `PYTHONUTF8=1 python -B -c "import nodegraph.selftest as S; S.test_split_pick_group()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> every group passes except the pre-existing `test_write_movie` (encoder); the groups scheduled after it were run separately and pass
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the exit code after the PASSED line is the probe's documented teardown crash)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 114 ops (baseline re-saved: the split cards lost a Mode and a socket)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> not pushed; on `graph-node-sets`, the push is the owner's call

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
