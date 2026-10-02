# Node synopsis datafile

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (built before the team-sync integration step; see the previous entry)
- **Base:** 6fbd719

## What changed

A new generated document, `codemap/node_synopsis.json`, describes every shipped node in one
nested record: label, category, owning module, a one-line `summary`, a long-form `overview`
(the compute's docstring, or the module's when the compute has none), 2D/3D support, the
data-access footprint (granularity, kernel axes, which mode selects them), the attribute
domains read and added, every mode with its choices, default, derive expression and per-choice
prose, and every input and output socket with its data type, role (dataset wire vs parameter),
unit, default, derive expression, field/multi flags, layer and column picker bindings, path
kind, and hover description. A `how_to_read` block defines the eight socket types, the socket
fields, the five granularities and the twelve domains, so the file is self-describing.

`scripts/_node_synopsis.py` generates it from the live registry (`check` is the default and
exits 1 when the file is stale; `write` regenerates; `show <op>` prints one node). The codemap
was regenerated so `modules.jsonl` indexes the new script.

A second, hand-curated file, `codemap/node_roles.json`, classifies every node by functional
ROLE (19: image restoration, background and intensity correction, image filtering,
registration, geometric transformation, projection and stitching, segmentation and labeling,
object detection, shape and distance geometry, representation transfer, measurement,
object filtering, tracking, motion and deformation fields, input/output, dataset and channel
organization, control flow, graph structure, visualization) and groups the roles into five
pipeline STAGES (acquire and organize, prepare the image, find structure, quantify, control
and present). The generator merges `role` and `stage` into each node record and emits
`roles` and `stages` sections; its check is BLOCKED, not merely stale, if a shipped node has
no role, has two, or if the roles file names an op that no longer exists. The registry's
`category` is the GUI palette grouping and is kept alongside; the role is a finer,
purpose-based cut (the 20 `enhancement` nodes split three ways, the 12 `utility` nodes four).

## Why

The request was a per-node review of inputs, outputs, data types and purpose in a data
structure. The codemap already holds those facts, but split across two terse line-per-record
indexes with defaults elided, which is right for grep and wrong for reading. Generating from
the registry rather than writing prose by hand means the review cannot drift from the code:
`check` fails the moment a socket, mode or docstring changes, and the file never needs a
human to notice.

## Files

- `scripts/_node_synopsis.py` (new)
- `codemap/node_synopsis.json` (new, generated, ~1 MB)
- `codemap/node_roles.json` (new, curated: 19 roles, 5 stages, every op assigned once)
- `codemap/gen/modules.jsonl`, `codemap/gen/MANIFEST.json`, `codemap/STATE.md` (regenerated)

## How to verify

```
python scripts/_node_synopsis.py            # SYNOPSIS CURRENT - 89 nodes, 579 inputs, 97 outputs
python scripts/_node_synopsis.py show analysis.measure
python -c "import json; d=json.load(open('codemap/node_synopsis.json')); print(d['counts'])"
```

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> not run for this change: it adds a leaf
  script and a generated JSON, nothing under the engine or GUI changed. The suite is red at
  HEAD for the pre-existing reasons recorded in the team-sync entry.
- [ ] GUI probe -> not applicable (no GUI change).
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (unchanged by this change)
- [x] `python scripts/_codemap.py` -> gen/ current; the 9 curated warnings predate this change
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
