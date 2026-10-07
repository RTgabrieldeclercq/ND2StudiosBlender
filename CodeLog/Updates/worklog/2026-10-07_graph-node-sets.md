# Graph node sets — a page kind orders the palette, it no longer filters it

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** graph-node-sets
- **Base:** e1ae0e28

## What changed

Every node can now be placed on every page. A page's kind (Image Input, Refinement,
Processing, Analysis) decides which nodes the palette, the link-drag search and the
Ready-to-run suggestions *lead with* — the kind's **primary set**, still the `pages` /
`op_pages` lists in `codemap/node_roles.json` — and every other node is the kind's
**secondary set**, offered after it. In the palette the secondary set sits under one
collapsed **More nodes** band at the bottom, grouped stage → role → node exactly like the
bands above it; a Free page has no secondary set (everything is primary). The link-drag
search lists primary matches first, then the rest; readiness suggestions rank the same way
and leave nothing out. Every band of the palette now starts **collapsed** (role rows inside
a band stay open, so one click on a band shows its nodes), two buttons beside the page-kind
chip — **Expand all** / **Collapse all** — fold the whole tree, *Expand all* sticks across
refills until *Collapse all*, and a search opens whatever it finds (including a hit under
More nodes). The chip reads `<Kind> nodes first`, its tooltip
says every node is placeable. The Image Input page's
primary set gains the **Projection & stitching** role — Stitch (M→1), Stack (T→1), Z Project —
beside Split Channels, Split Positions and Merge (any axis: C/T/M/Z), so splitting, merging,
stitching M, stacking T and projecting Z are all in the first band on the input graph.
`nodegraph.roles.secondary_ops(kind)` and `nodelab_v2.scene.primary_specs` /
`secondary_specs` are the new API; `op_in_page` / `ops_for_page` / `pages_of` keep their
names and now mean *primary* (their docstrings say so); `scene.visible_specs(kind)` returns
every visible node, primary first, instead of the primary set alone.

## Why

Request (2026-10-07): on the image-input graph, be able to split and merge any axis, stitch
M, stack T and stack Z, and in general build any combination — and stop limiting which node
types a graph can hold just because it was given a set of selected nodes. The V4.00 step-5
design made a page kind a *filter*: the palette hid every node outside the kind's list, so
a Stitch or a Z Project could not even be dragged onto an Image Input page, and anything
unusual (a Gaussian before merging, a Viewer on an input page) needed a Free page. The
cheap fix — add more ops to the Input list — would have kept the lock and reopened the
argument for every node. Keeping the per-kind lists as the *leading* set and putting the
rest under a collapsed band keeps the curated guidance (the usual nodes are still the first
thing in reach, and the welcome cards, recipes and page-boundary logic that use
`op_in_page` structurally keep working unchanged) without the lock. Collapsed-by-default
with Expand/Collapse all was asked for alongside; with ~100 nodes and a second set below,
an all-open tree no longer fits a dock.

## Files

- `codemap/node_roles.json` — `projection_and_stitching.pages` += `input`; the Image Input
  description; the `about` text now defines primary / secondary
- `codemap/node_synopsis.json` — regenerated (`scripts/_node_synopsis.py write`): three nodes
  gain `input` in their pages
- `nodegraph/roles.py` — `secondary_ops()`; primary/secondary wording in the docstrings
- `nodelab_v2/scene.py` — `visible_specs` = `primary_specs` + `secondary_specs`
- `nodelab_v2/readiness.py` — `producers_of` ranks primary first instead of filtering
- `nodelab_v2/palette.py` — More nodes band, nested stage bands, fold buttons, collapsed
  default, overview text for the Pages / More bands
- `nodegraph/selftest.py` — **hotspot**: `test_page_kind_catalog` (secondary set, stitch /
  stack / project primary on input), `test_pages_seam` (producers ordered, not filtered)
- `scripts/_nodelab_v2_phase5_probe.py` — new G2b block
- `MANUAL.md` §2 Pages, `codemap/concepts.md` CON-17 — prose
- `codemap/STATE.md`, `codemap/gen/*` — regenerated (`scripts/_codemap.py write`)

## How to verify

Run the app, open a new file (four standard pages), stay on **Image Input**: the palette
shows the Pages band and four stage bands, all collapsed, then **More nodes**; open
*Prepare the image* and Stitch (M→1) / Stack (T→1) / Z Project are there; open *More nodes*
and every other node is placeable (drag Gaussian Blur onto the input canvas — it lands).
Click **Expand all** / **Collapse all** beside the chip; type `gauss` in the search and the
hit under More nodes is open without a click. Drag a wire into empty canvas: the link
search lists the page's usual nodes first, then the rest. Headless:
`PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` prints the `G2b` line.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> 171 [ok] up to `test_write_movie`, which
  fails on the base commit too (pre-existing, unrelated: the movie's frame means are not
  monotone); the 36 tests after it were run separately in order and ALL PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no node files changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
