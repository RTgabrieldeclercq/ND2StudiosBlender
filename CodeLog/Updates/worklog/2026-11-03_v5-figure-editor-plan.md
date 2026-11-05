# V5 figure editor plan

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step1-figure-core
- **Base:** 572312ab

## What changed

A new design record, `CodeLog/ClaudesPlan/V5.00_figure_editor.md`: the Analysis page
stops being a node canvas and becomes an interactive **figure canvas** — a journal-sized
page of linked items (images, montages, zoom popouts bound by stage metadata, heat and
vector layers, native charts, colour bars, histograms, scale bars, annotations) fed by the
named Page Outputs of the upstream pages and rendered by one Qt painter for preview and
export. The record maps thirty representations confocal figures use to the nodes and
metadata that already feed them, names the gaps (orthogonal reslice, coded projections,
kymograph, line profile, MSD, rose and survival charts, colour-by-column overlays), fixes
nine decisions, and lays out seven PR-sized steps with their tests. No code changed.

## Why

The user asked for the Analysis tab to ask what output they want to work towards — plots,
overlays, both, more — built from results metadata rather than wired nodes, with editing
"extreme" enough to compose a montage, a metadata-placed popout, a twin with a user-
normalised speed heat layer and a documented display range on one page, paper-ready. A
first revision proposed a properties panel over a matplotlib render; the user's examples
need direct manipulation of both charts and overlays, so the record now builds on a Qt
scene graph (the node canvas's own idiom), keeps matplotlib only inside the existing
`plot.*` nodes, and reuses the Viewer's overlay painter rather than writing a second one.
Written down here so the other developer can read the plan cold before step 1 lands.

## Files

- `CodeLog/ClaudesPlan/V5.00_figure_editor.md`

## How to verify

Read the record; §4.2 holds the user's own example as the document the editor would save.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> skipped: documentation only
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> skipped: documentation only
- [ ] `python scripts/_catalog_snapshot.py` -> skipped: documentation only
- [ ] `python scripts/_codemap.py` -> skipped: no file under nodegraph/ or nodelab_v2/ touched
- [x] `python scripts/_sync_check.py` -> IN SYNC at session start (origin fetch refused by the machine's SSL setup; local refs used)
