# V5 step 2: the figure page

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step2-figure-page
- **Base:** 287f707 (`v5-step1-figure-core`, itself stacked on `nd2-acquisition-clock`)

## What changed

An **Analysis page is a figure page**. `nodelab_v2/figure_document.py` (new, Qt-free)
holds a `FigureDocument`: the figure document of `nodegraph.figure.schema` with a
revision, listeners, `edit` / `replace` (validated, a no-op edit bumps nothing), the
canonical spelling and digest, and the **shadow-graph sync** — one hidden `page.input`
per figure source (`__fig_src__<id>`) on the page's graph document, added, re-bound and
removed as the figure's sources change, never touching a hand-placed node. `envelopes`,
`run_ids` and `source_labels` give the editor each source's upstream envelope, its run id in
a composed graph and the label the Source menus show.

`nodelab_v2/workspace.py`: `Page.figure` and `FIGURE_KIND = "analyze"`; every Analysis page
made by `add_page` (so `standard()` and `File → New`) carries a figure; the figure's edits
re-sync the shadow and publish a page-level change; `page_outline` lists the figure's
items after its sources; `to_dict` writes the figure on the page record and `load_dict`
validates it before anything changes (refused by page id and spec path, workspace
untouched); a V4 file's Analysis page holding nodes opens as a Free page `<name> (graph)`
and its linked pages follow, an empty one as a figure page; a figure page cannot be linked
or be a master, a unique copy carries its own figure, `seed_input` never seeds one, `reset`
drops the figure of the page it keeps. The file format stays `3.0`. Composition,
`_input_seeds`, summaries, the cross-page signature and the LabLink publisher are
unchanged — the shadow graph is an ordinary graph to them.

`nodegraph/selftest.py` gains `test_figure_page` (the whole list above, end to end on the
workspace fixture, including a pull through the composed graph). CON-17 and MANUAL §13 say
what a figure page is and how an old file opens.

## Why

The design record's step 2: the Figure Editor (step 3) needs a page body it can validate,
hash and save, whose sources compose and pull through the existing run path. Making the
graph document the figure's shadow graph — rather than a second kind of body — is what
keeps `compose`, the memo, staleness, the Node queue and LabLink unchanged. **Deviation
from the record:** the format version is NOT bumped to 3.1 here. The `figure` key is
additive (an older build ignores it, so a file is never refused), and a bump would also
move the LabLink protocol token the selftest keeps in step with it — for a figure that
stays empty until the editor lands. The bump belongs with step 3, when the content is
real. The example workspace and the page recipes still place nodes on the Analysis
page; they move to figure documents in step 3, when the editor can show them.

## Files

- `CodeLog/Updates/worklog/2026-11-03_v5-step2-figure-page.md`
- `MANUAL.md` (§13, one bullet)
- `codemap/concepts.md` (CON-17, one paragraph), `codemap/STATE.md`, `codemap/gen/*`
  (regenerated — hotspot)
- `nodegraph/selftest.py` (hotspot — one test appended, one call in `main`)
- `nodelab_v2/figure_document.py` (new)
- `nodelab_v2/workspace.py`

## How to verify

`PYTHONUTF8=1 .venv\Scripts\python.exe -B -c "from nodegraph import selftest; selftest.test_figure_page()"`
prints one `[ok]` line. In the app, `File → New` then `File → Save`: the fourth page's
record in the file carries `"figure"`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC at session start (origin fetch refused by the machine's SSL setup; local refs used)
