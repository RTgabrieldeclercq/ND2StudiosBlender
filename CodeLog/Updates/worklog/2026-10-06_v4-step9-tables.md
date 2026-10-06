# V4.00 step 9 — tables: concat, join, aggregate

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step9-tables
- **Base:** e1ae0e28

## What changed

Three table nodes (category `table`, role `table_synthesis`, offered on Processing and
Analysis pages):

- **Table Concat** (`table.concat`) stacks the same-kind (Label or Point) tables of any number
  of inputs into one table: the union of their columns (missing ones blank — NaN or empty
  text), ids kept unique (input 0's unchanged, later inputs shifted past the largest id before
  them), and provenance columns — `condition` (the input's own `condition` metadata, else its
  entry in Condition names, else `input_K`), `input`, and `group` / `position_name` / `file`
  read from each input's OWN per-position metadata.
- **Table Join** (`table.join`) adds another table's columns to a table row by row on
  id+m+t, id, m+t or track_id+t, left or inner, with a prefix; a repeated key on the other
  side is refused as ambiguous, a name clash is refused, the key and coordinates are not
  copied. The other table may be on a second wire or the same one.
- **Table Aggregate** (`table.aggregate`) summarises a column per group (`group_by`, a comma
  list) with mean, median, sum, min, max, count, std and sem into a Label table with `n` and
  `<value>_<reducer>`, keeping the invariant coordinates.

Supporting changes: `std` and `sem` join `nodegraph.reducers` (NaN under two samples); text
columns are numpy unicode throughout (memo- and dock-safe); the plots refuse a text column as
X / Y / Value with a node-worded message and treat a blank text group as missing; If/Else's
track join skips text columns; the spreadsheet orders `condition` / `input` /
`position_name` beside `file` / `group`; a `table` category colour. **Engine:** a column
declaration marked `wants_inputs` is also handed every input's envelope
(`metadata._column_names_out`), so the columns a later input or the join's other wire
carries are offered by the closed column menus downstream. **Page Output:** a condition TYPED
on an Output survives later blank Outputs (blank still stamps the page's name), marked in
metadata as `condition_set`.

Tests: `test_table_concat`, `test_table_join`, `test_table_aggregate`,
`test_table_declarations_total`, `test_page_condition_typed`, `test_table_review`. GUI probe: TB1.

## Why

The V4 plan's last analysis piece: once several pages each measure their own condition or
dish, the Analysis page needs ONE table to compare them — rows labelled by condition — plus
the two everyday table operations, joining measurements of the same objects and summarising
per frame or condition. A grilling pass (three readers over the table model, the existing
table nodes and the GUI/LabLink side) settled the design before any code:

- the edit-time column catalog flowed from input 0 only, so without the `wants_inputs`
  opt-in every joined or concatenated column would have been unpickable in the closed menus;
- the GUI's per-position rule (`nodelab_v2.tables._with_per_m_column`) builds object-dtype
  columns the memo refuses and fills `group` / `file` from the OUTPUT's metadata, which after
  a concatenation is input 0's — so the engine writes them per input under the same names;
- a blank Page Output used to overwrite a condition typed further upstream with its own
  page's name, so labelling dishes on the first page could never reach the table.

## Review

An adversarial review (three lenses: nodes, engine/workspace, GUI; every finding reproduced by
an independent verifier) confirmed 14 findings — nine distinct problems, several found by two
or three lenses. All are fixed, with a regression check (`test_table_review`):

- **Concat duplicated ids for 0-based tables** (every Point detector): input K was shifted to
  the previous maximum, not past it ([0,1,2] + [0,1] gave [0,1,2,2,3]); joins then refused
  the table as ambiguous. Each later input now starts past every id already emitted.
- **Summary rows were labelled with position 0's file and group**: Aggregate wrote m = 0 and
  the spreadsheet / export filled `file` / `group` from position 0. A row not grouped by `m`
  now carries the one file / group its rows share, else blank.
- **A nested concat overwrote per-row provenance** (condition, file, group) with input-level
  values, and a concat's output kept input 0's typed scalar `condition`, which a blank Page
  Output downstream then kept. Rows that already say where they came from keep it; concat
  clears the scalar condition on payload and envelope.
- **Concat on a batch wrote member 0's file name on every row** and blocked the spreadsheet's
  per-member naming; a batch's per-M lists are no longer read.
- **Plots after a concat put every input on input 0's clock** (a 30 s page drawn on a 60 s
  axis). Concat writes `time_s` / `time_jd` per row when every input has a per-frame clock,
  and Plot Time Series reads them (elapsed and clock). `dt_s` cannot be read from the other
  inputs (a calibration key, memo-fenced to the node's own envelope), so a file with only a
  frame interval still shares input 0's.
- **The Viewer drew a combined Point table over input 0's image** (another page's detections,
  and input 0's twice); a join repeated its left table's marks. Tables a table node writes are
  marked (`__table_synth__`) and left off the point / track overlays unless picked.
- **std / sem became inf with one inf value**, and count counted it: values are reduced over
  their finite entries.
- **A Name equal to the table read** left stale columns in the menus (Aggregate) or replaced
  the other table (Join); both are refused.
- **The spreadsheet showed a blank-titled file tab** for rows with no file; blanks name no tab.

## Files

- `nodegraph/catalog/table/{__init__,concat,join,aggregate}.py` — new nodes
- `nodegraph/catalog/_shared/table_ops.py` — new: the shared table core
- `nodegraph/metadata.py` — `CONDITION_KEY`, `CONDITION_SET_KEY`; `adds_columns.wants_inputs`
- `nodegraph/reducers.py` — `std`, `sem`; `nodegraph/catalog/transform/transfer_domain.py` —
  comment
- `nodegraph/catalog/_shared/figure.py`, `nodegraph/catalog/plot/*.py` — the text guard,
  blank text as missing
- `nodegraph/catalog/analysis/if_else.py` — track join skips text
- `nodelab_v2/ops.py`, `nodelab_v2/workspace.py` — the typed-condition rule
- `nodelab_v2/tables.py` — provenance column order; `nodelab_v2/theme.py` — `table` colour
- `nodegraph/catalog/__init__.py` — MODULES (hotspot: three appended entries)
- `codemap/node_roles.json` — role `table_synthesis`; `codemap/workflows.md` — WF-04
- `nodegraph/selftest.py` — six tests (hotspot: tail of `main()`)
- Review fixes: `nodelab_v2/viewer.py` (synthesized tables off the overlays),
  `nodelab_v2/spreadsheet.py` (no blank file tab), `nodegraph/catalog/plot/timeseries.py`
  (per-row clocks)
- `scripts/_nodelab_v2_phase5_probe.py` — TB1
- `scripts/catalog_baseline.json` — re-saved for the three nodes (hotspot)
- `MANUAL.md` (§15 Tables, Page Output row), `CodeLog/ClaudesPlan/V4.00_beta_tests.md`
  (Step 9 block), `CodeLog/ClaudesPlan/V4.00_workspaces.md` (delivery row),
  `CodeLog/Updates/CHANGELOG.md`
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

`python run.py` → Example graph → after Measure add **Table Aggregate** (Group by `t`) → pull
→ the spreadsheet shows `summary` with `n`, `area_mean`, `area_sem`, `area_count` → add
**Plot XY** (Table `summary`, X `t`, Y `area_mean`). For Concat: two Processing pages with
Page Outputs `cells`, an Analysis page with two Page Inputs into **Table Concat** →
`combined` carries `condition` = the page names. The beta block B9.1–B9.11 lists the rest.

## Gates

- [x] selftest: all 167 registered tests run (continuing past failures); one fails,
  `test_write_movie` — the pre-existing codec-rounding failure every V4 step has carried. The
  table tests (`test_table_*`, `test_page_condition_typed`) and the contract gates
  (`test_param_socket_contract`, `test_column_catalog_complete`, `test_option_docs`,
  `test_socket_docs`, `test_codemap`) pass.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI
  PROBES PASSED (123 checks, TB1 among them), exit 0
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 103 ops (baseline re-saved:
  the three table nodes)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-15 re-read, extended with
  `wants_inputs`, blessed); `_node_synopsis.py` -> SYNOPSIS CURRENT
- [x] every reviewer's reproduction script re-run on the fixed tree: each now shows the fix
- [x] `python scripts/_sync_check.py` -> 0 behind origin/Blender (e1ae0e2); stacked on
  `v4-step8-plots2`
