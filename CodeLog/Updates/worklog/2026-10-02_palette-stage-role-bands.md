# Palette: stage and role rows as left-justified bands with a filled background

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** team-sync-roles-palette-viewer
- **Base:** fd5c7a5 (origin/Blender) + the seven commits of this branch

## What changed

In the node palette the five stage rows ("Acquire & organize", "Prepare the image", …)
and the role rows under them are now **bands**: the text sits in the first column and
spans the row, so it starts at the panel's left edge instead of after the dots column;
the stage band has a filled accent-dim background with bold ink text at normal size, and
the role band a lighter tint with accent text, one indent step in. Node rows are unchanged
(input dots | label | output dots). The bands follow the theme (dark and light) because
the palette is refilled on restyle.

The bands are painted by a small item delegate (`_BandDelegate`) rather than left to the
view, because the panel's stylesheet has a `QTreeWidget::item` rule and once such a rule
exists Qt's stylesheet style ignores the model's background brush — `setBackground` alone
produced no band at all (verified in an off-screen render before the delegate was added).

## Why

The user: "on the left node menu it's too hard to see each section like 'Prepare the
image'; make this left justified with a coloured background for contrast, and each
sub-section left justified too." The first cut put the stage label in the middle column in
small muted caps, which made it both indented and the faintest thing on the panel —
exactly backwards for a section heading.

## Files

- `nodelab_v2/palette.py` — stage/role rows in column 0, spanned, with background +
  foreground brushes; `_BandDelegate` paints the band
- `scripts/_nodelab_v2_phase5_probe.py` — G2 reads stage labels from column 0, checks the
  rows are spanned and carry a band brush
- `codemap/gen/*`, `codemap/STATE.md` — regenerated

## How to verify

Open NodeLab: the palette's stage rows are filled bands flush left; roles are lighter bands
one step in; nodes keep their dots. Toggle the theme (View ▸ Theme) and the bands recolour.
Off-screen: `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` (G2).

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` — not re-run: GUI-only change, the
  selftest never imports `palette.py`. Still red at HEAD for the pre-existing reasons in
  the team-sync entry.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no node changed)
- [x] `python scripts/_codemap.py` -> gen/ CURRENT
- [x] `python scripts/_sync_check.py` -> on the pushed feature branch, level with origin
