# Mirror continues down the chain, lane by lane

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** node-mirror-merge
- **Base:** e1ae0e28

## What changed

Mirror groups are now **lanes**: `{lane: node}`, where a lane is named after the ticked card
it started from. The copy made for ticked card `C` is lane `C`'s member. In mirror mode
(mirror selected, something ticked), a wire drawn out of ANY member of a mirror group, which
needs no tick of its own, is now mirrored across that group's lanes
(`multilink.lanes_for`; `applies` uses it):

- **To a fresh node**, or one outside any group: each other lane gets its own copy, wired from
  that lane's member, at the same offset. The new nodes form a group with the same lanes, so
  the node placed after THAT is mirrored again, for any number of steps. The copied node's
  other inputs follow their lane too: from a group member, each lane's own member; from
  anything else, the same shared node.
- **To the same lane's member of another group** (e.g. from Z's Gamma into Z's Overlay): the
  wire is drawn in every lane between that lane's members, and no node is made. Lanes the
  target group lacks get a copy and join it.
- **Across lanes** (Z's Gamma into A's Overlay): drawn once, not mirrored, and the status bar
  says so.

Merge is unchanged and still acts on the ticked set only. In merge mode, a wire from a
mirrored copy is an ordinary wire.

## Why

User report: after mirroring a node onto every ticked card, a node placed after the
just-placed one was added only once, so the mirror stopped at the first step. The request
asks that "all mirrored nodes need to have that connection as well".

Lanes were chosen over moving the ticks onto the new copies. Moving ticks would have
stopped the original ticked cards from mirroring again, and the lanes also let a wire
between two parallel groups be repeated lane-wise. Mere group membership can't do that: it
has no notion of which copy goes with which.

## Files

- `nodelab_v2/multilink.py`: lane groups (`groups: List[Dict[lane, node]]`), `lanes_of`, `lane_in`, `lanes_for`, `applies`, `mirror` (copies, lane-wise connections, cross-lane refusal), `_set_group`, `group_of`
- `nodelab_v2/scene.py`: the mode button's tooltip
- `nodegraph/selftest.py`: `test_multilink` §3b (hotspot: inside the test appended earlier today)
- `scripts/_nodelab_v2_phase5_probe.py`: MM1 continues from a mirrored Gamma by the link search and by a wire drag
- `MANUAL.md`: *Mirror and merge* — the chain paragraph and the ordinary-wire sentence
- `codemap/STATE.md`, `codemap/gen/*`: regenerated. The diff was read: 4 new symbols, `_join_group` gone, `applies`/`mirror`/`group_of` changed

## How to verify

`test_multilink` §3b: a three-step chain from a mirrored node, a parallel input, a lane-wise
connection with no node made, and a cross-lane refusal. Probe MM1: on the canvas, the node
placed after the mirrored Gamma (link search) and the one after that (wire drag) both appear
in all three lanes. In the app: tick three cards, Mirror, add a node from one card, then
drag on from the new node.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (MM1 with the chain steps)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> not IN SYNC by design (stacked branches, none merged into `origin/Blender`); 0 behind it
