# Codemap file pins eol

- **Date:** 2026-10-05
- **Author:** hyper
- **Branch:** v4-step2-runner-workspace
- **Base:** e1ae0e28

## What changed

A curated codemap entry that pins a whole file (`file:` anchor) now fingerprints the file with
its line endings normalized to LF (`nodegraph/codemap.py`, `anchor_hash`), so the same text has
the same pin on every checkout. The eight entries whose pins had been taken from CRLF working
copies were re-blessed one by one: CON-02, CON-11, CON-13, INV-02, INV-07, INV-08, INV-10 and
INV-11. For seven of them the pinned files are byte-identical to a fresh LF checkout apart
from line endings, so their prose was not re-read for content. INV-08 pins
`nodegraph/codemap.py`, which this change edits; it was re-read, still holds (the module
imports no Qt), and its Qt-free corollary now also lists `nodelab_v2/workspace.py`, which
V4.00 added.

## Why

The full selftest failed `test_codemap` in a fresh `git worktree` of this branch: INV-11
"CHANGED since this entry was written" for `nodegraph/kernels/README.md`, a file nobody had
touched. `.gitattributes` pins the repo to LF, but this machine's working copy still holds
CRLF files from before that rule. The eight curated entries blessed on 2026-10-05 had their
`file:` pins hashed from those CRLF bytes, so every LF checkout read all eight as changed:
the other developer's clone, a fresh clone, any worktree. The gate was green only on the one
machine that blessed them. Normalizing inside the fingerprint, rather than rewriting this
machine's working copy to LF, fixes it for every checkout at once and keeps the next stray
CRLF file from breaking it again.

## Files

- `nodegraph/codemap.py` — `file:` pins hash LF-normalized bytes
- `codemap/curated.lock.json` — the eight entries re-blessed (`--verified 2026-10-05`)
- `codemap/invariants.md` — INV-08's corollary names `nodelab_v2/workspace.py`
- `codemap/gen/*`, `codemap/STATE.md` (regenerated)

## How to verify

- `python scripts/_codemap.py` -> `CODEMAP CURRENT` in this working copy, and in a fresh
  `git worktree add <dir> HEAD` (LF files) — the second is what failed before.
- `PYTHONUTF8=1 python -B -m nodegraph.selftest` — the `[ok] codemap: …` line.

## Gates

- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 95 ops
- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> green except the pre-existing `test_write_movie`, as recorded in the step 2 worklog committed alongside
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> as recorded in the step 2 worklog committed alongside (no GUI code changed here)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->
