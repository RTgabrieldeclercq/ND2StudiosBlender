# V5.00 stack pushed: the step 4–7 Sync lines ticked

- **Date:** 2026-11-05
- **Author:** Gabriel Declercq
- **Branch:** v5-step7-figure-polish (top of the V5.00 stack)
- **Base:** ee06665 (step 7)

## What changed

The Gates block of the step 4, 5, 6 and 7 entries said fetch and push were blocked and the
branch waited locally. Those four Sync lines are now ticked and say when and onto what the
stack was pushed and what the block was. No code changed.

## Why

The stack — `nd2-acquisition-clock`, then V5.00 steps 1–7, twelve commits over
`origin/Blender` @572312a — reached origin on 2026-11-05 once git was pointed at the Windows
certificate store: Norton Web/Mail Shield re-signs every TLS connection and git's bundled
OpenSSL CA file refused its root. The diagnosis, the fix (`http.sslBackend=schannel`) and
the sync-check change that came out of it are in `2026-11-05_sync-check-open-prs.md` on its
own branch (PR #2), so they reach `Blender` without waiting for this stack. A reader of the
step 4–7 entries on `Blender` would otherwise find four unticked gates claiming the branches
were never pushed. Pushed commits are not amended, so the correction is its own commit on
top of the stack. The stack was ahead of `origin/Blender` and not behind it, so no rebase
and no re-run of the heavy gates was needed; each step's gates stand as recorded.

## Files

- `CodeLog/Updates/worklog/2026-11-03_v5-step4-figure-charts.md` — Sync line
- `CodeLog/Updates/worklog/2026-11-03_v5-step5-figure-paper.md` — Sync line
- `CodeLog/Updates/worklog/2026-11-03_v5-step6-figure-recipes.md` — Sync line
- `CodeLog/Updates/worklog/2026-11-03_v5-step7-figure-polish.md` — Sync line
- this entry

## How to verify

`git log --oneline origin/Blender..v5-step7-figure-polish` lists thirteen commits, and
`python scripts/_sync_check.py` on the branch reports ahead 13, behind 0, with the stack's
own PRs tagged `[this branch]` or `[in your stack]` and no PR action in the verdict.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` — skipped: worklog text only; step 7's ALL PASSED (269) stands
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` — skipped: worklog text only
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC — on top of origin/Blender @572312a, nothing incoming; pushed 2026-11-05
