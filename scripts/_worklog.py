#!/usr/bin/env python
"""Worklog: one file per change, so two people never conflict on a shared changelog.

    python scripts/_worklog.py new <slug>      # scaffold CodeLog/Updates/worklog/<date>_<slug>.md
    python scripts/_worklog.py list [N]        # newest N entries (default 10)
    python scripts/_worklog.py check           # exit 1 if any entry still has a TODO placeholder

The scaffold pre-fills the file list from `git diff` (staged + unstaged + untracked) so the
WHAT is accurate by construction; the author writes the WHY. The entry is committed in the
same commit as the change it describes, and `scripts/_sync_check.py` flags commits that
lack one.

Stdlib only. Never import nodegraph here.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOG_DIR = os.path.join(ROOT, "CodeLog", "Updates", "worklog")
PLACEHOLDER = "TODO"

TEMPLATE = """\
# {title}

- **Date:** {date}
- **Author:** {author}
- **Branch:** {branch}
- **Base:** {base}

## What changed

{what}

## Why

{why}

## Files

{files}

## How to verify

{verify}

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [ ] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [ ] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
"""


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout.strip()


def changed_files() -> list[str]:
    seen: list[str] = []
    for cmd in (("diff", "--name-only"), ("diff", "--name-only", "--cached"),
                ("ls-files", "--others", "--exclude-standard")):
        for f in git(*cmd).splitlines():
            f = f.strip()
            if f and f not in seen and not f.startswith("CodeLog/Updates/worklog/"):
                seen.append(f)
    return seen


def cmd_new(slug: str) -> int:
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    if not slug:
        print("usage: _worklog.py new <slug>", file=sys.stderr)
        return 2
    os.makedirs(WORKLOG_DIR, exist_ok=True)
    today = dt.date.today().isoformat()
    path = os.path.join(WORKLOG_DIR, f"{today}_{slug}.md")
    if os.path.exists(path):
        print(f"already exists: {os.path.relpath(path, ROOT)}")
        return 1

    author = git("config", "user.name") or "<unset>"
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "<detached>"
    base = git("merge-base", "HEAD", "origin/Blender")[:8] or "<unknown>"
    files = changed_files()
    files_md = "\n".join(f"- `{f}`" for f in files) or f"- {PLACEHOLDER}: no changes detected yet; list them when you commit"

    body = TEMPLATE.format(
        title=slug.replace("-", " ").capitalize(),
        date=today, author=author, branch=branch, base=base,
        what=f"{PLACEHOLDER}: one paragraph. What a reader diffing this commit will see. "
             "Name the behaviour that changed, not the lines.",
        why=f"{PLACEHOLDER}: one paragraph. The problem, symptom, or request that caused it, and "
            "why this fix rather than the obvious alternative. This is the part git cannot "
            "reconstruct.",
        files=files_md,
        verify=f"{PLACEHOLDER}: the command or click-path that shows the change working.",
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    print(f"created {os.path.relpath(path, ROOT)}")
    print(f"  {len(files)} changed file(s) pre-listed. Fill every {PLACEHOLDER}, then commit "
          "it WITH the change it describes.")
    return 0


def entries() -> list[str]:
    if not os.path.isdir(WORKLOG_DIR):
        return []
    return sorted(f for f in os.listdir(WORKLOG_DIR) if f.endswith(".md"))


def cmd_list(n: int) -> int:
    rows = entries()
    if not rows:
        print("no worklog entries yet")
        return 0
    for f in rows[-n:][::-1]:
        with open(os.path.join(WORKLOG_DIR, f), encoding="utf-8") as fh:
            first = fh.readline().strip().lstrip("# ")
        print(f"{f}  {first}")
    return 0


def cmd_check() -> int:
    bad = []
    for f in entries():
        with open(os.path.join(WORKLOG_DIR, f), encoding="utf-8") as fh:
            if PLACEHOLDER + ":" in fh.read():
                bad.append(f)
    if bad:
        print(f"{len(bad)} worklog entr{'y' if len(bad) == 1 else 'ies'} still contain "
              f"a {PLACEHOLDER} placeholder:")
        for f in bad:
            print("  CodeLog/Updates/worklog/" + f)
        return 1
    print(f"WORKLOG OK - {len(entries())} entries, no placeholders")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "new" and len(argv) >= 2:
        return cmd_new(" ".join(argv[1:]))
    if argv[0] == "list":
        return cmd_list(int(argv[1]) if len(argv) > 1 else 10)
    if argv[0] == "check":
        return cmd_check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
