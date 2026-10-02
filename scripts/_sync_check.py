#!/usr/bin/env python
"""Two-person sync check. Run at the start of every Claude coding session.

    python scripts/_sync_check.py            # full report, exit 1 if action needed
    python scripts/_sync_check.py --hook     # SessionStart hook: same report, always exit 0

Stdlib only, on purpose: the SessionStart hook runs with whatever `python` is on PATH,
which in this repo has no numpy. Never import nodegraph here.

What it reports, in order:
  1. who you are (git identity) and which branch you are on
  2. whether the branch is the shared integration branch (editing it directly is the
     thing this repo's team-sync skill forbids)
  3. the fetch result, and ahead/behind against origin/<integration> and origin/<branch>
  4. open pull-request heads on origin and whether each is already merged
  5. incoming commits and whether they touch the files that always conflict
  6. the dirty working tree
  7. commits on this branch with no worklog entry under CodeLog/Updates/worklog/
  8. a one-line verdict with the actions to take, in order
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

INTEGRATION_BRANCH = "Blender"
WORKLOG_DIR = "CodeLog/Updates/worklog"

# Paths that two people editing concurrently will collide on every single time.
# Each maps to the recipe for resolving it. Keep in step with the team-sync skill.
HOTSPOTS = {
    "nodegraph/selftest.py": "both sides append tests at the end: keep BOTH hunks",
    "nodegraph/catalog/__init__.py": "both sides add imports: keep BOTH lines",
    "codemap/gen/": "GENERATED: take either side, then `python scripts/_codemap.py write`",
    "codemap/STATE.md": "GENERATED: take either side, then `python scripts/_codemap.py write`",
    "scripts/catalog_baseline.json": "GENERATED: resolve code first, then `python scripts/_catalog_snapshot.py save`",
    "MANUAL.md": "keep both sides, re-read the merged section once",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args: str, check: bool = False, timeout: int = 60) -> str:
    r = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def count_lr(a: str, b: str) -> tuple[int, int] | None:
    out = git("rev-list", "--left-right", "--count", f"{a}...{b}")
    if not out:
        return None
    l, r = out.split()
    return int(l), int(r)


def ref_exists(ref: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref], cwd=ROOT, capture_output=True
    ).returncode == 0


def is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT, capture_output=True
    ).returncode == 0


def main(argv: list[str]) -> int:
    hook = "--hook" in argv
    lines: list[str] = []
    actions: list[str] = []
    say = lines.append

    say("=== TEAM SYNC CHECK (scripts/_sync_check.py) ===")

    # 1. identity + branch -------------------------------------------------------------
    name = git("config", "user.name") or "<unset>"
    email = git("config", "user.email") or "<unset>"
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "<detached>"
    say(f"identity : {name} <{email}>")
    say(f"branch   : {branch}")
    if name in ("<unset>", "McGheeLab"):
        actions.append(
            "git identity is the shared lab default; set a personal one so blame means "
            "something: git config user.name \"<Your Name>\""
        )

    # 2. on the integration branch? ----------------------------------------------------
    on_integration = branch == INTEGRATION_BRANCH
    if on_integration:
        say(f"WARNING  : you are ON '{INTEGRATION_BRANCH}'. Do not edit here. "
            f"Create a feature branch first: git switch -c <topic>")
        actions.append(f"switch off '{INTEGRATION_BRANCH}' onto a feature branch before editing")

    # 3. dirty tree (checked before the rebase advice so the stash comes first) ----------
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    if status:
        rows = status.splitlines()
        say(f"tree     : {len(rows)} path(s) modified/untracked")
        for row in rows[:20]:
            say("           " + row)
        if len(rows) > 20:
            say(f"           ... {len(rows) - 20} more")
        actions.append("the tree is dirty: commit it, or `git stash -u` before any rebase/pull "
                       "and `git stash pop` after")
    else:
        say("tree     : clean")

    # 4. fetch + ahead/behind ----------------------------------------------------------
    try:
        fetch_err = subprocess.run(
            ["git", "fetch", "origin", "--prune"], cwd=ROOT, capture_output=True,
            text=True, timeout=45,
        ).stderr.strip()
        say("fetch    : ok" + (f" ({fetch_err.splitlines()[-1]})" if fetch_err else ""))
        fetched = True
    except subprocess.TimeoutExpired:
        say("fetch    : TIMED OUT (offline?) - numbers below are from the last fetch")
        fetched = False
    except Exception as e:  # noqa: BLE001
        say(f"fetch    : FAILED ({e}) - numbers below are from the last fetch")
        fetched = False

    integ_remote = f"origin/{INTEGRATION_BRANCH}"
    if ref_exists(integ_remote):
        lr = count_lr("HEAD", integ_remote)
        if lr:
            ahead, behind = lr
            say(f"vs {integ_remote}: ahead {ahead}, behind {behind}")
            if behind:
                actions.append(
                    f"integrate {behind} incoming commit(s): "
                    + ("git pull --rebase origin " + INTEGRATION_BRANCH if on_integration
                       else f"git rebase {integ_remote}")
                    + "  (stash first if the tree is dirty)"
                )
    else:
        say(f"vs {integ_remote}: remote branch not found")

    branch_remote = f"origin/{branch}"
    if branch != INTEGRATION_BRANCH and ref_exists(branch_remote):
        lr = count_lr("HEAD", branch_remote)
        if lr:
            say(f"vs {branch_remote}: ahead {lr[0]}, behind {lr[1]}")
            if lr[1]:
                actions.append(f"your own branch moved on origin ({lr[1]} commits): "
                               f"git pull --rebase origin {branch}")

    # 5. open pull requests -------------------------------------------------------------
    say("")
    say("pull requests on origin:")
    try:
        prs = git("ls-remote", "origin", "refs/pull/*/head", timeout=45)
    except Exception:  # noqa: BLE001
        prs = ""
    pr_rows = []
    for row in prs.splitlines():
        m = re.match(r"^([0-9a-f]+)\s+refs/pull/(\d+)/head$", row.strip())
        if m:
            pr_rows.append((int(m.group(2)), m.group(1)))
    if not pr_rows:
        say("  none visible" if fetched else "  unknown (fetch failed)")
    for num, sha in sorted(pr_rows):
        merged = ref_exists(integ_remote) and is_ancestor(sha, integ_remote)
        state = "merged into " + INTEGRATION_BRANCH if merged else "OPEN / not yet merged"
        say(f"  PR #{num}  {sha[:8]}  {state}")
        if not merged:
            actions.append(
                f"PR #{num} is open: review/merge it on GitHub before starting work that "
                f"may overlap, or pull its head to look: git fetch origin pull/{num}/head:pr-{num}"
            )

    # 6. incoming commits + hotspot touches ----------------------------------------------
    if ref_exists(integ_remote):
        incoming = git("log", "--oneline", "--format=%h %an %ad %s", "--date=short",
                       f"HEAD..{integ_remote}")
        if incoming:
            say("")
            say("incoming commits (not in your HEAD):")
            for row in incoming.splitlines()[:15]:
                say("  " + row)
            touched = git("diff", "--name-only", f"HEAD...{integ_remote}")
            hot = {}
            for f in touched.splitlines():
                for prefix, recipe in HOTSPOTS.items():
                    if f.startswith(prefix):
                        hot[prefix] = recipe
            if hot:
                say("  these touch known conflict hotspots:")
                for prefix, recipe in hot.items():
                    say(f"    {prefix:<34} -> {recipe}")

    # 7. worklog coverage ----------------------------------------------------------------
    if ref_exists(integ_remote):
        local_commits = git("log", "--format=%h %s", f"{integ_remote}..HEAD")
        missing = []
        for row in local_commits.splitlines():
            sha = row.split()[0]
            files = git("show", "--name-only", "--format=", sha)
            if not any(f.startswith(WORKLOG_DIR + "/") for f in files.splitlines()):
                missing.append(row)
        if missing:
            say("")
            say(f"commits on this branch with NO entry under {WORKLOG_DIR}/:")
            for row in missing[:10]:
                say("  " + row)
            actions.append(
                f"{len(missing)} commit(s) lack a worklog entry: "
                "python scripts/_worklog.py new <slug>, fill What/Why, amend or add a commit"
            )

    # 8. verdict -------------------------------------------------------------------------
    say("")
    if actions:
        say("VERDICT: ACTION NEEDED before you edit code, in this order:")
        for i, a in enumerate(actions, 1):
            say(f"  {i}. {a}")
    else:
        say("VERDICT: IN SYNC - safe to start. Work on a feature branch, log what/why "
            "with scripts/_worklog.py, run the gates before you push.")
    say("Procedure: invoke the `team-sync` skill (.claude/skills/team-sync/SKILL.md).")

    out = "\n".join(lines)
    try:
        print(out)
    except UnicodeEncodeError:
        print(out.encode("ascii", "replace").decode("ascii"))
    if hook:
        return 0
    return 1 if actions else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
