# V5.00 stack integration: PR #3 carries steps 1–7 and origin/Blender

- **Date:** 2026-11-05
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock (the head of PR #3)
- **Base:** ac2868c (PR #4 merged into this branch) + 577c43a (the stack's top) + origin/Blender @72bf0b1

## What changed

`nd2-acquisition-clock`, the head of PR #3, now carries the whole V5.00 figure-editor stack
(steps 1–7 and the worklog tick commit) and `origin/Blender`'s PR #2 (the sync-check fix).
Two merge commits, no code of its own: the first merges the stack's linear top, the second
merges `origin/Blender`; the three generated codemap files that both sides had regenerated
were taken from `Blender` and regenerated again with `scripts/_codemap.py write`.

## Why

PRs #4–#10 were merged in order on GitHub, but each PR's base was the branch below it, so
step 1 landed on `nd2-acquisition-clock`, step 2 on `v5-step1-figure-core`, … step 7 on
`v5-step6-figure-recipes` — and nothing past step 1 reached `Blender`. One more merge of
PR #3 lands everything, which is what those seven clicks meant. A rebase and force-push of
the stack would have given a straighter line, but the worklog entries cite the stack's
commit SHAs, so the commits are kept and merged instead. `Blender` had moved by PR #2
only (a stdlib script and the skill); the merge conflicted on nothing but the generated
codemap, resolved the way the team-sync skill says: take either side, regenerate.

## Files

- `CodeLog/Updates/worklog/2026-11-05_v5-stack-integration.md` — this entry
- `codemap/STATE.md`, `codemap/gen/MANIFEST.json`, `codemap/gen/modules.jsonl` — regenerated after the second merge

## How to verify

`git log --oneline origin/Blender..nd2-acquisition-clock` lists the stack and the two merges;
`git diff --stat origin/Blender...nd2-acquisition-clock` shows the figure editor; PR #3 on
GitHub reports no conflicts with `Blender`.

## Gates

(run on the merged tree, after the second merge)

- [x] `PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 145 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC (ahead of origin/Blender, behind 0)
