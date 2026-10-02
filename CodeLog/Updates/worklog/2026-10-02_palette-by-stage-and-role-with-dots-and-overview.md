# Palette: grouped by stage and role, in/out dots, and a click-to-read overview

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (built before the team-sync integration step; see the team-sync entry)
- **Base:** 6fbd719

## What changed

The node palette (left dock) is reorganised into the taxonomy from `codemap/node_roles.json`:
five pipeline stages (Acquire & organize, Prepare the image, Find structure, Quantify,
Control & present) as top-level headers, nineteen functional roles beneath them, nodes
beneath the roles. Search still filters, and only stages and roles with a hit remain.

Every node row carries coloured dots. On the left, what flows in: one dot per attribute
domain the node reads from its Dataset (voxel, label, point, …), or a plain dataset dot when
it requires nothing, then one per parameter type. On the right, what flows out: one dot per
domain the node adds, or a plain dataset dot when it passes its input through, then one per
value output. The colours are the canvas's own socket and domain-rail colours, so a dot in
the palette means what a wire means on the canvas. Hovering a dot column spells it out.

The bottom third of the panel is a scrolling overview card. Click a node and it shows the
label, op key, stage › role, the one-line description, every active input and output with
type, unit, default and layer-picker role, the modes with their choices and defaults, the
footprint and 2D/3D support, and the long-form "how it works" from the compute's docstring
(or the module's when the compute has none). Click a stage or role header and it shows that
group's description and members. Before any click it shows a legend of the dot colours.

`nodegraph/roles.py` (new, Qt-free) is the single runtime reader of the roles file: stages
and roles in pipeline order, `role_of(op)` with an `other/other` fallback for an unassigned
op, and `reload()` which the palette's ⟳ button triggers so a newly written role appears
without a restart. The GUI probe's G2 section now asserts the stage order, that every
visible op is placed, that both dot icons exist with domain-aware tooltips, that the
overview follows the click, and that a filter leaves no empty group.

## Why

The request, verbatim in intent: reorganise the left menu into the 19-role / 5-stage
taxonomy we built, give each node coloured dots for its input and output data types, and
show an overview of the node and its data structure in the bottom third when clicked.

The grouping is read from the same file the synopsis check validates, so the palette and the
documentation cannot disagree about where a node lives, and a node added without a role
shows up under "Other" in the GUI while the synopsis check fails loudly. The dots encode
domains rather than only socket types because nearly every node takes and returns a
Dataset; "reads voxel, adds label" is the information a user choosing between two
segmentation nodes actually needs, and it is exactly what the canvas's domain rails already
colour. The overview reads the live registry rather than `node_synopsis.json` so it is
current after a hot reload and never a second copy of the facts.

## Files

- `nodegraph/roles.py` (new)
- `nodelab_v2/palette.py` (rewritten)
- `scripts/_nodelab_v2_phase5_probe.py` — G2 extended
- `MANUAL.md` — the palette paragraph
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

```
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png     # G2
```

In the GUI: the palette shows ACQUIRE & ORGANIZE … CONTROL & PRESENT with roles beneath;
hover a node's dots; click Connected Components and read the overview; type "otsu" in the
search and only Find structure › Segmentation & labeling remains.

## Gates

- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole is still red at HEAD for the
  pre-existing reasons recorded in the team-sync entry. No engine change here beyond the
  new Qt-free reader module.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no catalog change)
- [x] `python scripts/_codemap.py` -> gen/ current
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT (roles file unchanged)
