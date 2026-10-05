# V4.00 step 0 — the version lives in one file; the Workspaces plan is on record

- **Date:** 2026-10-05
- **Author:** hyper (Claude Fable 5.1 session)
- **Branch:** v4-step0-version
- **Base:** e1ae0e28 (origin/Blender) + 57cae63 (split-positions-node, unmerged)

## What changed

ND2Studios now has a version: `nodelab_v2/version.py` carries `__version__ = "4.0.0"`,
`APP_NAME`, `VERSION_LABEL` ("ND2Studios V4") and `PRODUCT` ("ND2Studios V4 — NodeLab"),
and nothing else in the repo writes the number. The main window title reads it
("ND2Studios V4 — NodeLab — nodegraph canvas"), Help → Capabilities opens with it, and the
LabLink `hello` reports `SOFTWARE_NAME = "ND2Studios V4 (NodeLab)"` /
`SOFTWARE_VERSION = "4.0.0"` from it (the literal there said "2.21", three minor versions
stale). The design record `CodeLog/ClaudesPlan/V4.00_workspaces.md` opens the V4 generation:
the decisions, the architecture, the ten steps and their tests. CHANGELOG gains a `[4.0.0-dev]`
section and MANUAL's header says what V4 is and where the plan lives. No behaviour of the
engine, the catalog or the graph file format changes in this step; `format_version` is still
"2.0" until step 1.

## Why

The user asked for the workspace restructure (typed node-graph pages with named outputs,
linked pages, pop-out panels, plots) to be broken into manageable steps and named as a new
version. Step 0 exists so the name and the plan land before any code does: there was no
`__version__` anywhere (the only version strings were a stale LabLink literal and the
`Vx.yy` plan-file convention), so a reader of a saved file or a hub log could not tell which
generation wrote it. Writing the number once, in a Qt-free module both the GUI and the
LabLink adapter import, is what lets step 1 stamp `app_version` into the 3.0 file format
without a second literal. The `nodelab_v2` package keeps its name by decision: the version is
a label, and renaming the package would churn every import, the selftest seam, the probes and
the codemap for no behaviour.

## Files

- `nodelab_v2/version.py` — new
- `nodelab_v2/window.py` — title and Help → Capabilities read `PRODUCT`/`__version__`
- `nodelab_v2/lablink/protocol.py` — `SOFTWARE_NAME`/`SOFTWARE_VERSION` from `version.py`
- `CodeLog/ClaudesPlan/V4.00_workspaces.md` — new design record (dated; code wins on conflict)
- `CodeLog/Updates/CHANGELOG.md` — `[4.0.0-dev]` section, step 0 entry
- `MANUAL.md` — H1 and the V4.00 paragraph in the header
- `codemap/gen/*`, `codemap/STATE.md` — regenerated (one new module, two import edges,
  window.py symbol lines shifted by the two inserted lines)

## How to verify

```
python -c "from nodelab_v2.lablink import protocol as P; print(P.SOFTWARE_NAME, P.SOFTWARE_VERSION)"
python run.py        # title bar: "ND2Studios V4 — NodeLab — nodegraph canvas"; Help → Capabilities… opens with "ND2Studios V4 — NodeLab 4.0.0."
```

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole: still red at HEAD for the two
  pre-existing reasons (`test_codemap` aborts on the 9 stale curated entries CON-02, CON-11,
  CON-13, INV-02, INV-07, INV-08, INV-10, INV-11, WF-07 that their owners should bless one by
  one; `test_write_movie` codec rounding). Run directly: `test_lablink` passes with the new
  strings.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 93 ops
- [x] `python scripts/_codemap.py` -> gen/ CURRENT (the same 9 curated entries unverified)
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] `python scripts/_sync_check.py` -> feature branch, 0 behind origin/Blender
