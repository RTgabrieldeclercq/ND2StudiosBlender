---
name: team-sync
description: >-
  MANDATORY FIRST STEP FOR ANY CODING WORK IN THIS REPO. Two people on two computers both
  drive Claude against the same GitHub repo, so every coding session begins by checking for
  incoming commits and open pull requests, and every change ends with a written WHAT and WHY
  that travels with the commit. This skill is the procedure: (1) run
  `python scripts/_sync_check.py` and act on its VERDICT before touching a file - never edit
  on the shared `Blender` branch, rebase onto `origin/Blender` first, resolve the known
  conflict hotspots with the recipes here; (2) work on a feature branch in small commits;
  (3) before pushing, scaffold a worklog entry with `python scripts/_worklog.py new <slug>`
  (one file per change under CodeLog/Updates/worklog/, so two writers never collide), fill
  What/Why, run the four gates, commit the entry WITH the change, push, open a PR; (4) end
  the reply with a "What changed / Why" summary. A SessionStart hook in
  .claude/settings.json already prints the sync report into context at startup - read it.
  Triggers: starting any task that edits files, "implement", "fix", "add a node", "refactor",
  "commit", "push", "merge", "pull request", "PR", "rebase", "conflict", "sync", "am I up to
  date", "what did the other person change", "what changed", "changelog", "worklog".
---

# team-sync — two people, two Claudes, one repo

**One rule: never start editing until `scripts/_sync_check.py` says IN SYNC, and never
push a commit that does not say what it changed and why.**

Why this exists: on 2026-10-02 one machine had 13 commits in a day on `Blender` while the
other pushed a 25-file "Overlay node fix" to the same branch. Both touched
`nodegraph/selftest.py`, `codemap/gen/*`, `scripts/catalog_baseline.json` and
`nodegraph/catalog/__init__.py`. Those four collide on every merge unless the integration
below is followed. Neither commit said why it was made.

## Phase 0 — read the sync report (automatic)

A `SessionStart` hook in [.claude/settings.json](../../settings.json) runs
`python scripts/_sync_check.py --hook` and its output is already in your context under
`=== TEAM SYNC CHECK ===`. If it is not there (hook disabled, session resumed from
compaction, or you are a subagent), run it yourself:

```
python scripts/_sync_check.py
```

It uses the `python` on PATH on purpose (stdlib only) so it never dies on the missing-numpy
trap. It prints a **VERDICT** with numbered actions. Do them in that order. Do not start the
user's task first and "sync later" — the later merge is exactly the thing this prevents.

## Phase 1 — integrate before you edit

| VERDICT says | Do |
|---|---|
| you are ON `Blender` | `git switch -c <topic>` (lower-kebab, what not who: `overlay-pin-fix`, not `gabriel-work`) |
| tree is dirty | If the edits are the user's unfinished work, `git stash -u`, integrate, `git stash pop`. If they are leftovers nobody claims, ask the user before discarding. |
| behind `origin/Blender` | On a feature branch: `git rebase origin/Blender`. On `Blender` itself (only to catch up, never to work): `git pull --rebase origin Blender`. |
| PR #N open | Tell the user. If the user's task overlaps the PR's files (`git fetch origin pull/N/head:pr-N && git diff --stat origin/Blender...pr-N`), stop and ask whether to build on the PR branch or wait for the merge. If it does not overlap, continue and say so. |
| commits lack a worklog entry | Phase 3 below, before anything new. |
| identity is the lab default | Ask the user for their name once and run `git config user.name "<Name>"`. Do not guess it. |

After a rebase that pulled in code, **run the gates** (Phase 3 list) before editing. A
green gate after integration is the only proof the merge did not silently break something.
There is no CI.

### Conflict recipes — the files that always collide

Resolve the *code* conflicts first, then the generated files, then regenerate:

| File | Resolution |
|---|---|
| `nodegraph/selftest.py` | Both sides appended tests at the end. Keep **both** hunks, in either order. Then run the selftest. |
| `nodegraph/catalog/__init__.py` | Both sides added import lines. Keep **both**. |
| `codemap/gen/*`, `codemap/STATE.md` | **Never hand-merge.** `git checkout --theirs codemap/gen codemap/STATE.md` then `.venv\Scripts\python.exe scripts/_codemap.py write`. Read `git diff codemap/` — an unexpected line is a finding. |
| `scripts/catalog_baseline.json` | **Never hand-merge.** After the code is resolved: `.venv\Scripts\python.exe scripts/_catalog_snapshot.py save`, then re-run with no argument and expect `CATALOG IDENTICAL`. |
| `MANUAL.md`, `codemap/*.md` | Keep both sides, then re-read the merged section once for duplicated paragraphs. |

Never use git's `union` merge driver on Python. Never `--force` push `Blender`.

## Phase 2 — work

- Feature branch only. Small commits, each one buildable. Push at least daily even if the
  branch is unfinished — an unpushed branch is invisible to the other person's sync check.
- Follow the repo's other rules unchanged: `build-node-v2` for nodes, `codemap` for lookup,
  `python scripts/_codemap.py write` after any change under `nodegraph/` or `nodelab_v2/`.
- Ownership heuristic that avoids most collisions: one person in `nodegraph/` (engine), the
  other in `nodelab_v2/` (GUI) for a given stretch. Adding a node always touches the four
  hotspot files, so say in the worklog entry that you did.

## Phase 3 — close: say what and why, then prove it, then push

1. **Scaffold the worklog entry** — one file per change, never a shared changelog:
   ```
   python scripts/_worklog.py new <slug>
   ```
   It writes `CodeLog/Updates/worklog/<YYYY-MM-DD>_<slug>.md` with the changed-file list
   pre-filled from `git diff`. Fill every `TODO:`:
   - **What changed** — the behaviour a reader diffing the commit will see. Not the lines.
   - **Why** — the symptom, request, or design pressure, and why this fix over the obvious
     alternative. This is the part git cannot reconstruct and the only reason the file exists.
   - **How to verify** — a command or click-path.
   - **Gates** — tick the ones you ran; say which you skipped and why.

   `python scripts/_worklog.py check` must print `WORKLOG OK` (no placeholders left).

2. **Run the gates** (the interpreter is `.venv\Scripts\python.exe`, see CLAUDE.md):
   ```
   PYTHONUTF8=1 python -B -m nodegraph.selftest
   PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png
   python scripts/_catalog_snapshot.py
   python scripts/_codemap.py
   ```
   Report failures as failures. A docs-only change may skip the GUI probe; say so.

3. **Commit the entry with the change.** Subject line = what, body = why, first line of
   the body names the worklog file. The sync check flags any commit on the branch that has
   no `CodeLog/Updates/worklog/` file in it.
   ```
   <area>: <what changed, imperative, under 70 chars>

   Why: <one or two sentences>
   Worklog: CodeLog/Updates/worklog/<date>_<slug>.md
   ```

4. **Rebase and push.** `git fetch origin && git rebase origin/Blender`, re-run the gates
   if anything came in, then `git push -u origin <topic>`.

5. **Open the PR.** There is no `gh` on these machines. Give the user the compare URL:
   `https://github.com/RTgabrieldeclercq/ND2StudiosBlender/compare/Blender...<topic>?expand=1`
   PR body = the worklog entry's What and Why, pasted. The other person's next sync check
   will list it as `PR #N OPEN`.

## Phase 4 — the reply

End every reply that changed files with a block the other person can read cold:

```
What changed: <one or two sentences, behaviour not lines>
Why: <one or two sentences>
Files: <the list, hotspot files called out>
Worklog: CodeLog/Updates/worklog/<file>.md
Gates: <which ran, which passed, which skipped and why>
Sync: <branch>, rebased on origin/Blender @<sha>, PR <url or "not yet pushed">
```

## Subagents

A subagent does not see this skill or the hook output. If you delegate editing work,
paste Phases 1 and 3 into its prompt, or keep the commit and worklog step in the parent.

## What to do when the other person's work breaks yours

Do not "fix forward" on `Blender`. Reproduce on a feature branch, write the worklog entry
first (the Why is the breakage), fix, gates, PR. The entry is how the other person learns
what their change did on your machine.
