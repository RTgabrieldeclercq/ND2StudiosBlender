# Region tabs

- **Date:** 2026-10-07
- **Author:** hyper
- **Branch:** region-tabs
- **Base:** a49deca

## What changed

A frame on the canvas is now a **region** with **ports**, and a region can be duplicated as
a **linked tab**. *Graph → Draw a region…* (`R` on the canvas, or Alt+drag) arms a box;
dragging it around a set of cards makes a frame around the cards whose centre is inside
(`Region`, `Region 2`, …). Every wire crossing the frame's border is drawn as a port on the
edge — an input port per signal entering, an output port per socket a wire leaves from —
with the counts in the title bar; `Ctrl+J` frames get the same ports. The frame's menu
(and *Graph → Duplicate region as a linked tab*) makes a new **Free page linked to the
region alone**: the frame's nodes follow the master page (grow the frame and the tab
grows), their values are overrides there like any linked page, and at each port the tab
owns a Page Input (bound to a Page Output the master gains at that socket, reused by the
next tab) or a Page Output — nodes the tab may re-point, rename, add or delete freely, so
every tab of one region can take different inputs, publish different outputs and hold
different values. The frame's title bar counts its tabs; the page menu says `(linked ·
region · N overrides)`, the Pages panel `region tab`, the inspector names the region. The
file carries `region` beside `master`; *Duplicate as linked page* of a tab is another tab;
*Make unique* drops the link. Refused with the reason: a box enclosing no card, a value
wire or a zone across the border, a tab sending edits to the master.

## Why

Asked for: group nodes by drawing a region, give the wires crossing it connection points,
duplicate the region into a new tab, let every tab differ in inputs, outputs and variables
while the nodes inside stay one graph. Everything but the drawing and the ports already
existed as *linked pages* (shared structure, own values) and *Page Input / Output* (named
boundaries), so the tab is a `LinkedDocument` restricted to one frame's members rather than
a new group engine: the engine's `groups.py` is one-in / one-out and would not have given
per-tab inputs and outputs. The tab is a Free page because only a Free page may read from
its own master's page without the kind order refusing it. The tab's boundary nodes are its
*structure* (a modified linked page from birth) rather than mirrored nodes, which is exactly
what makes "a different set of inputs and outputs per tab" true without new machinery. The
members are remembered on the tab so deleting the frame on the master never empties the
tabs.

## Files

- `MANUAL.md` — §12 *Regions* subsection, §2 pointer
- `codemap/STATE.md`, `codemap/gen/*` — regenerated
- `codemap/concepts.md` — CON-18 extended, anchors re-pinned (6) and blessed
- `nodegraph/selftest.py` — `test_region_tabs` (hotspot file: appended at the end)
- `nodelab_v2/document.py` — `region_ports`, `region_interface`, `set_frame_members`
- `nodelab_v2/frame_item.py` — ports, `title_text`, `shape`
- `nodelab_v2/inspector.py` — the banner names the region
- `nodelab_v2/linked_document.py` — `region`, `_master_nodes`, member-restricted mirror
- `nodelab_v2/scene.py` — region box on the view, `frame_from_rect`, frame menu entries
- `nodelab_v2/window.py` — Graph menu entries, `duplicate_region`, labels
- `nodelab_v2/workspace.py` — `Page.region`, `duplicate_region`, `region_tabs`, file format
- `scripts/_nodelab_v2_phase5_probe.py` — probe RG1

## How to verify

Run the app, load the example graph (welcome card), press `R` on the canvas and drag a box
around two wired cards: a frame appears with a dot on its left and right edge and
`1 in · 1 out` in its title. Right-click the frame → *Duplicate region as a linked tab*: a
new tab opens showing the two cards between a Page Input and a Page Output; change a value
there and the master keeps its own; add a card to the frame on the master and the tab grows.
Headless: `python -B -c "from nodegraph.selftest import test_region_tabs; test_region_tabs()"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED (216 ok, with `test_write_movie` stubbed: it fails identically on the untouched base a49deca — monotone-means assertion in the movie encoder — so it is not this change's)
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)
