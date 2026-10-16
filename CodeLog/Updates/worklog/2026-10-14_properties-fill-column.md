# Properties fill column

- **Date:** 2026-10-14
- **Author:** Gabriel Declercq
- **Branch:** properties-fill-column
- **Base:** 572312ab

## What changed

The Properties panel now fills the full width of its tab column, which it shares with
Spreadsheet and LabLink. That holds both at the default layout and when the column is
dragged wider. 376 px is now the panel's **minimum** width (`setMinimumWidth`) rather than
its fixed width (`setFixedWidth`), so the right-hand ƒ-auto buttons still never clip.

Measured offscreen at 1500 × 880 with the demo graph:

| | Tab column | Properties dock |
|---|---|---|
| Default layout | 424 px | 424 px |
| Column widened to 620 px | 620 px | 620 px |
| Before this change | — | 376 px, whatever the column's width |

## Why

Reported: "The properties tab does not span its whole length." In the probe's own window
screenshot, the right column ran from x 1070 to 1500 (430 px) but the Properties title bar
and content stopped at x ≈ 1445, leaving a dark strip down the right.

The cause is `InspectorPanel.setFixedWidth(376)`. Its comment states the intent: wide enough
that the richest row does not clip. That is a floor. A fixed width is also a ceiling, and in
a tab group Qt sizes the column to its widest member or to the user's drag, while the capped
dock stops at 376.

A minimum keeps the original guarantee and drops the cap. `INSPECTOR_W` (the default-layout
request) is unchanged. The `_apply_default_sizes` docstring that described the panel as
fixed now says it is a floor.

## Files

- `nodelab_v2/inspector.py` — `setFixedWidth(376)` → `setMinimumWidth(376)`, with the reason
- `nodelab_v2/window.py` — `_apply_default_sizes` docstring: the panel's 376 is a floor
- `scripts/_nodelab_v2_phase5_probe.py` — SH1b: on a fresh 1920×1400 window, Properties raised and widened by 200 px must take the width, with the panel filling the dock (the fixed width held it at 376)
- `codemap/gen/*`, `codemap/STATE.md` — regenerated; the diff is only line/loc shifts below the edits

## How to verify

`QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 python -B -u scripts/_nodelab_v2_phase5_probe.py out.png`
prints SH1b with "filling its column when it is widened (576 px)".

In the app: Properties reaches the window's right edge. Drag the splitter to its left
wider, and the panel's rows stretch with it.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (170 ok, exit 0; SH1b: 376 px default, 576 px widened)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> 0 behind origin/Blender when the branch was cut; not yet committed or pushed
