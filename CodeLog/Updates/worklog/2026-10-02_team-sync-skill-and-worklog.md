# Team sync skill and worklog

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (should have been a feature branch; this is the change that makes the rule enforceable)
- **Base:** 6fbd719

## What changed

Two developers on two machines now share one procedure. A `SessionStart` hook in
`.claude/settings.json` runs `scripts/_sync_check.py --hook` at the start of every Claude
session and prints a report with a numbered VERDICT: git identity, current branch (warning if
it is `Blender`), dirty tree, fetch, ahead/behind against `origin/Blender`, open pull-request
heads on origin and whether each is merged, incoming commits and which known conflict
hotspots they touch, and any local commits that carry no worklog entry. The `team-sync` skill
is the procedure around it: integrate first, feature branch, conflict recipes for the files
that always collide, and a close-out that writes What/Why before pushing.
`scripts/_worklog.py new <slug>` scaffolds one Markdown file per change under
`CodeLog/Updates/worklog/` with the changed-file list pre-filled; `check` fails on leftover
placeholders. CLAUDE.md gained a short section pointing at all of it.

Found while running the gates on this change: `nodegraph/hotreload.py:_digest_path` hashed raw
bytes, so every node `fp` in `codemap/gen/nodes.jsonl` depended on the clone's line endings.
This clone has `core.autocrlf=true` (CRLF working copy, LF blobs); the committed fingerprints
came from an LF checkout, so each regeneration here flipped 86 fingerprints that the other
machine would flip back. The digest now folds CRLF to LF before hashing, and a new
`.gitattributes` pins `eol=lf` so both clones see identical bytes.

## Why

The two of us pushed to `Blender` from both machines on the same day; one side was a 25-file
commit that touched `selftest.py`, `codemap/gen/*`, `catalog_baseline.json` and
`catalog/__init__.py`, the same four files the other side had touched 13 times. Neither
commit said why it was made. Both machines run Claude, so the fix had to be something Claude
executes rather than a convention people remember: a hook that fires before any edit, a
script that says exactly what to do next, and a per-change log file that cannot conflict
because no two changes share a file. The fingerprint fix is in the same change because it is
the first thing the new procedure found: a generated-file conflict that would have recurred
on every single PR regardless of how careful either person was.

## Files

- `.claude/settings.json` (new) — SessionStart hook
- `.claude/skills/team-sync/SKILL.md` (new) — the procedure
- `scripts/_sync_check.py` (new) — the report, stdlib only
- `scripts/_worklog.py` (new) — worklog scaffold / list / check
- `CodeLog/Updates/worklog/` (new) — this entry is the first
- `CLAUDE.md` — new section "Two people, two Claudes"
- `nodegraph/hotreload.py` — `_digest_path` folds CRLF before hashing (hotspot: engine)
- `.gitattributes` (new) — `* text=auto eol=lf` plus binary markers
- `codemap/gen/*`, `codemap/STATE.md` — regenerated (two new script modules; node `fp`
  values now match the committed LF-based ones)

## How to verify

```
python scripts/_sync_check.py            # prints the report; exit 1 while actions remain
python scripts/_worklog.py check         # WORKLOG OK
git diff --stat codemap/gen/nodes.jsonl  # after `_codemap.py write`: no fp churn on a CRLF clone
```

Start a new Claude session in the repo: the `=== TEAM SYNC CHECK ===` block appears in context.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> NOT GREEN, and not because of this change:
  `test_codemap` asserts that no curated entry is stale, and 9 entries (CON-02, CON-11, CON-13,
  INV-02, INV-07, INV-08, INV-10, INV-11, WF-07; pinned to files changed Aug–Sep 2026, none of
  them `hotreload.py`) were already stale at HEAD `6fbd719`. The runner is a straight sequence, so
  the 40 tests after it never ran. Verified instead: the 89 tests before it pass (including
  `test_live_reload_contract`, which covers `hotreload.py`), and the 40 after it were run directly
  with `test_codemap` skipped: 40 pass, 1 fails. The failure is `test_write_movie` at
  `selftest.py:21249` (decoded frame means not monotone: frame 3 mean 52.40 < frame 2 mean 53.14,
  a codec rounding artefact). It fails identically when run alone and touches nothing this change
  edits. The incoming origin commit `fd5c7a5 Overlay node fix` extends this test but leaves the
  monotone assertion as is, so the failure is most likely the video codec on this machine, not
  the code. Check whether it passes on the other machine; if it does, the assertion needs a
  tolerance, not the node. Whoever owns the 9 stale entries should
  re-read and `python scripts/_codemap.py bless <ID>` each one; never bulk-bless.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> gen/ CURRENT; the 9 "curated prose unverified" warnings predate this change (dated Aug–Sep) and were left for their authors
- [x] `python scripts/_sync_check.py` -> integrated 2026-10-02 on branch `team-sync-roles-palette-viewer`,
  created from `origin/Blender` at `fd5c7a5 Overlay node fix` with the whole day's working tree
  replayed onto it (`git stash -u` / `git stash pop`). Conflicts and how they were resolved, as a
  first exercise of the recipes in this skill: `MANUAL.md` node table (kept the new Viewer row,
  the incoming Overlay row and the incoming Experiment Canvas row); `codemap/gen/*` and
  `codemap/STATE.md` (took the base side, regenerated); `scripts/catalog_baseline.json` merged
  cleanly but was re-blessed anyway, 90 ops; `nodegraph/selftest.py`, `nodegraph/metadata.py`,
  `nodelab_v2/runner.py`, `viewer.py`, `window.py` merged automatically and were import-checked.
  The incoming commit added a node, `view.canvas`, with no entry in `codemap/node_roles.json`, so
  the synopsis gate blocked until it was given the `visualization` role — exactly the check that
  file exists for. The user's own uncommitted recipe edits (`*.nd3` glob) and `graphs/` folder
  were carried through and left uncommitted. Stale PR #1 (head `5084466`, 22 commits behind) is
  still open on origin and should be closed by its author.
