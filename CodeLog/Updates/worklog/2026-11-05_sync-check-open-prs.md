# Sync check: ask GitHub which PRs are open; the TLS fix and `gh` in the team-sync skill

- **Date:** 2026-11-05
- **Author:** Gabriel Declercq
- **Branch:** sync-check-open-prs
- **Base:** 572312ab

## What changed

`scripts/_sync_check.py` now asks GitHub which pull requests are **open** — `gh pr list`
when the CLI is installed and logged in, else the REST API (a public repo, or a `GH_TOKEN`
/ `GITHUB_TOKEN` in the environment) — and lists each with its head branch and title. A PR
whose head is the branch you are on is tagged `[this branch]`; one whose head sits below
yours in your own stack is tagged `[in your stack]`; neither is an action. Only PRs outside
your stack reach the verdict, as one line naming them all. When GitHub does not answer the
old `refs/pull/*/head` listing runs, and its header says that a closed-but-unmerged PR looks
open there. The team-sync skill gains a Phase 1 row for the `SSL peer certificate or SSH
remote key was not OK` fetch failure and its fix, and Phase 3 step 5 creates the PR with
`gh` (a stacked branch takes its parent as `--base`) instead of asserting there is no `gh`.

## Why

Every session since PR #1 was closed unmerged (a pull of the McGheeLab fork's `Blender`,
opened 2026-08-10) the verdict demanded "review/merge PR #1": GitHub keeps
`refs/pull/N/head` for closed PRs, so the ref listing cannot tell them from open ones. Found
while pushing the V5.00 figure-editor stack on 2026-11-05, after fetch and push had failed on
this machine for two sessions with the SSL message above. The cause was Norton Web/Mail
Shield re-signing every TLS connection with a root that only the Windows certificate store
trusts; `curl.exe -I https://github.com` succeeding while `git fetch` failed was the tell,
and `git config --global http.sslBackend schannel` the fix. `gh` 2.91 turned out to be
installed and logged in, contrary to the skill. Why fix the PR section rather than drop it:
the open-PR warning is the one thing that stops two people building the same change, so it
had to be right, not gone. Why the stack tags: eight stacked PRs would otherwise put eight
"merge this PR" actions in their own author's verdict.

## Files

- `scripts/_sync_check.py` — `github_slug`, `open_prs`, section 5 rewritten
- `.claude/skills/team-sync/SKILL.md` — Phase 1 SSL row, Phase 3 step 5 with `gh`
- `codemap/STATE.md`, `codemap/gen/MANIFEST.json`, `codemap/gen/modules.jsonl` — regenerated (the script's LOC and docstring)

## How to verify

`python scripts/_sync_check.py` prints `pull requests OPEN on GitHub:` followed by the real
list (none when this was written; PR #1 no longer appears). With a PR open for the branch
you are on, its line ends `[this branch]` and the verdict carries no PR action. To see the
offline path, cut the network: the header becomes `pull requests on origin (refs/pull/*/head
— GitHub did not answer, …)` and PR #1 is back, flagged as possibly closed.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` — skipped: scripts + skill docs only; `_sync_check.py` is stdlib-only and nothing under `nodegraph/` or `nodelab_v2/` imports it
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` — skipped, same reason
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 137 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (after `write`; diff = scripts LOC + the script's module record)
- [x] `python scripts/_sync_check.py` -> IN SYNC (branched from origin/Blender @572312a, nothing incoming)
