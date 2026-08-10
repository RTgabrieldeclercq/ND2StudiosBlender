# ND2Studios_Blender — orientation for an agent

A Blender-geometry-nodes-style **node editor for microscopy image analysis**: wire
acquire → enhance → segment → measure → track pipelines on a canvas, pull a node, see the
result. Python 3.13, PySide6. Run it with `python run.py`.

**The seam.** `nodegraph/` is the engine and is **Qt-free**. `nodelab_v2/` is the GUI and
imports the engine. `scripts/` is a leaf — nothing imports it. There is exactly one
sanctioned exception to the arrow: `nodegraph/selftest.py` reaches up into `nodelab_v2` to
test the Qt-free GUI seam. Do not add a second.

**Nodes do not live in `nodegraph/nodes.py`.** That is an 89-line facade whose import loads
the catalog. A node is one file: `nodegraph/catalog/<category>/<name>.py`.

## Gates — run before claiming done

```
PYTHONUTF8=1 python -B -m nodegraph.selftest                       # "ALL NODEGRAPH SELF-TESTS PASSED"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png    # "ALL PHASE-5 GUI PROBES PASSED"
python scripts/_catalog_snapshot.py                                # "CATALOG IDENTICAL — N ops …"
python scripts/_codemap.py                                         # "CODEMAP CURRENT — …"
```

`PYTHONUTF8=1` on Windows only: some `[ok]` lines carry `µ`/`σ`/`↔` and a cp1252 console
raises `UnicodeEncodeError` **inside the reporting line**, which reads like a failure and is
not one. `-B` because a stale `.pyc` from a moved module fabricates failures in tests you
never touched — the tell is a traceback whose source line does not match that line number.

## How to find things

> **Nouns route through the generated index. Systems route through the curated cards.
> Verbs route through the skills.**

Grep first; open a source file only after a record names it.

| You want | Do this |
|---|---|
| a node — sockets, footprint, what it does | `grep '"op":"<op_key>"' codemap/gen/nodes.jsonl` |
| what a param means, its unit, its default | `grep '"key":"<op>:in:<param>"' codemap/gen/sockets.jsonl` |
| where a symbol is defined, and its signature | `grep '"id":".*<Name>"' codemap/gen/symbols.jsonl` |
| what a module is for, what it exports | `grep '"path":"<path.py>"' codemap/gen/modules.jsonl` |
| who imports X / what X needs | `grep '"dst":"<mod>"' codemap/gen/imports.jsonl` (or `"src"`) |
| a term — `TILEABLE`, `voxel`, `meta_transform` | `grep '"t":"<TERM>"' codemap/gen/vocab.jsonl` |
| how a whole flow works (pull, tiling, memo, reload) | [codemap/workflows.md](codemap/workflows.md) — `WF-01`…`WF-07` |
| a concept, precisely | [codemap/concepts.md](codemap/concepts.md) — `CON-01`… |
| rules you must not break | [codemap/invariants.md](codemap/invariants.md) — `INV-01`… |
| a kernel's integration contract | [nodegraph/kernels/README.md](nodegraph/kernels/README.md) indexes one `.md` per kernel |
| **to build or modify a node** | invoke the **`build-node-v2`** skill. It cites the concepts you need |
| user-facing behaviour, symptoms, recipes | [MANUAL.md](MANUAL.md) — §18 troubleshooting, §16 worked workflows |
| the record shapes above | [codemap/_schema.md](codemap/_schema.md) |

**Budgets.** Orientation (this file) ~2k tokens. One symbol or node lookup <500. A
task-scoped dive: the index, **one** curated card, and at most two of the reads it names.

**Escape hatch.** After two failed lookups, stop searching the map and read the code — and
say in your response which lookup failed, so the gap gets filled. A third grep costs more
than the file would have.

## Do not trust the map for

It is checked for *existence and shape*, never for truth. The gate can prove that a node
exists, that its footprint is what the map says, that every path and anchor resolves, and
that prose has not drifted off the code it is pinned to. It cannot prove any *explanation* is
correct, and it knows nothing about performance numbers, GUI runtime behaviour, or "why".

**On conflict, code wins — and you fix the map in the same change.** A disagreement you only
report is one the next agent pays for again.

```
running code  >  codemap/gen/*.jsonl  >  codemap/*.md  >  the skills
              >  ENGINEERING_NOTES.md  >  MANUAL.md  >  CodeLog/**
```

## Traps

- **`CodeLog/Architecture/ARCHITECTURE.md` describes the v1 app, deleted 2026-07-29.**
  History only. Same for `CodeLog/ClaudesPlan/**` — dated design records, several superseded,
  and some name gate scripts that no longer exist.
- **`CodeLog/ClaudesPlan/V3.00_catalog_expansion_roadmap.md` is 717 KB (~180k tokens).**
  Never open it whole. Grep it for a phase name if you must.
- **`nodegraph/selftest.py` is ~17k lines.** Never read it whole; grep for the `test_*` name.
- **Never trust a node count you find in prose.** [codemap/STATE.md](codemap/STATE.md) is the
  only place in this repo allowed to carry one.
- `nodegraph/kernels/registration.py` defines `apply_frame` **twice**; the second shadows the
  first. If you edit one, check you edited the live one.

## Footguns

- **`scripts/_catalog_snapshot.py` used to re-bless its own baseline when run bare.** Fixed —
  the default is now `check`. Old docs may still tell you to pass `check` explicitly; harmless.
- **A test fixture must never `define_node` a real `op_key`.** `NODES` is process-global, so
  it clobbers the shipped node and causes failures that depend on test order.
- **Run the selftest with `-B`.** See the gates block above.
- The GUI gates run offscreen; they need `QT_QPA_PLATFORM=offscreen` on a headless box.
- CellSAM's *second* model pull in one process crashes the interpreter — a native heap fault
  no `except` can catch. One pull per process.

## Editing rules

- After changing anything under `nodegraph/` or `nodelab_v2/`:
  `python scripts/_codemap.py write`, then **read `git diff codemap/`** — an unexpected line
  in that diff is a finding, not noise — and commit it with your change.
- If the gate reports a curated entry, re-read that entry, fix it if it is now wrong, then
  `python scripts/_codemap.py bless <ID> --verified <today>`.
- Never hand-edit `codemap/gen/*`. Never bulk-bless.
- A fact that would help a teammate belongs in this repo, not in a private memory.

## Briefing a subagent — this is not optional

**A subagent does not receive this file.** Measured on 2026-08-05: an Explore agent's context
contained the environment block, the scratchpad path, and the *descriptions* of the available
skills — and nothing else. It had never heard of `codemap/`. Two benchmark subagents both
ignored the index and hand-grepped the repo; one burned 17 tool calls probing for the registry
API that `symbols.jsonl` states in one line.

Two things follow, and you have to do both:

1. **Skill descriptions are the only channel that reaches a subagent.** That is why the
   **`codemap`** skill exists — its description alone tells an unbriefed agent the index is
   there. Do not delete it as redundant with this file; it is the half that travels.
2. **Paste this into every exploratory subagent prompt**, filled in:

> This repo has a generated grep-first index — invoke the `codemap` skill, or go straight to
> `codemap/gen/*.jsonl` (one JSON record per line: `nodes`, `sockets`, `modules`, `symbols`,
> `imports`, `vocab`). Grep it before you grep source. For `<op_key | subsystem>` start at
> `<exact path>`. Do NOT read `CodeLog/ClaudesPlan/**` unless I name a file — one is 717 KB.
> Do not trust any node count in prose; `codemap/STATE.md` is the only current one. Report
> `file:line` and exact quotes, not summaries. Budget: `<N>` files.

## Layout

```
run.py                    launcher (nd2studios_worker.py = the headless LabLink worker)
nodegraph/                THE ENGINE — Qt-free
  registry.py             NodeSpec / SocketSpec / the 2D-3D DimMode lever / Granularity
  catalog/<cat>/<node>.py one file per node; _shared/ holds the helpers they forward through
  kernels/                portable analysis maths + one .md integration contract each
  engine.py streaming.py  lazy pull + granularity routing; per-tile providers + tile cache
  memo.py revision.py     two-hash memo + the revision fence that is its identity
  metadata.py domains.py  the edit-time MetaEnvelope pass; the eleven attribute domains
  hotreload.py            live per-node reload + the AST import graph
  codemap.py              this map's generator
  selftest.py             `python -m nodegraph.selftest` — the core gate
nodelab_v2/               THE GUI (PySide6; Qt lives only here)
  document.py ops.py      Qt-free editing model and node ops — importable headless
  runner.py               canvas → Engine, off the UI thread
  window.py scene.py viewer.py inspector.py …
  lablink/                run this app as a remote worker
codemap/                  this map. gen/ is generated; the .md files are hand-written
CodeLog/                  design records and changelog — history, not instructions
scripts/                  gates, benchmarks, one-off probes and validations
```
