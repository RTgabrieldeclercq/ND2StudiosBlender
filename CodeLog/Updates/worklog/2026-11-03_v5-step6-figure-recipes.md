# V5.00 step 6 — figure templates, the headless renderer, LabLink, docs

- **Date:** 2026-11-03
- **Author:** Gabriel Declercq
- **Branch:** v5-step6-figure-recipes (stacked on v5-step5-figure-paper → step 4 → step 3 → step 2 → step 1 → nd2-acquisition-clock)
- **Base:** ff2c374 (step 5)

## What changed

**Figure templates.** `nodelab_v2/figure_recipes.py` (new, Qt-free) and five built-ins under
`nodelab_v2/builtin_figure_recipes/`: *Segment & measure summary* (field with outlines,
area histogram, per-frame table, legend), *Tracking summary* (trails, MSD with its
exponent, survival, lifetimes table), *Strain / speed field panel* (heat layer with its
colour bar and the values' distribution), *Standard curve* (two columns, linear fit,
table), *Plate overview with popout* (montage over M, a popout with its zoom box, a
shared display range). A recipe is a figure document with **slots** in place of sources
(`needs`: image / labels / points / tracks / fields / mesh / table) and placeholders —
`slot:cells` for the Page Output value, `slot:cells.labels` for its first label layer,
`slot:cells.labels.column` for that table's first measured column — that `bind_recipe`
fills from the page's catalog; an unfillable slot is reported and keeps its placeholder.
The start card lists the templates the catalog can fill (the others greyed with what they
need); `FigureEditorPanel.apply_recipe` commits the bound document as the page;
`save_as_recipe` writes the user's own (`~/.nd2studios/figure_recipes/`, shadowing a
built-in by name; `NODELAB_FIGURE_RECIPES_DIR` / `=0` for tests).

**Headless renderer.** `nodelab_v2/figure_render.py` (new): a **figure job** (`build_job`)
is the composed graph, the figure, its run ids and a seed spec per loader (a file path
ingested as the app ingests it, or the synthetic stack for a blank path); `render_job`
makes an offscreen `QApplication`, seeds a headless engine, binds the same
`FigureEditorPanel` to a `HeadlessHost` whose `fetch` pulls synchronously, and exports the
picture or the whole bundle. CLI: `python -m nodelab_v2.figure_render file.nd2graph.json
--out fig.png [--page] [--dpi] [--svg] [--bundle] [--write-job]` or `--job job.json`.
`figure_editor/fonts.py` registers matplotlib's DejaVu TTFs with Qt once per process, so
the editor, the renderer and the plot nodes share one face (offscreen Qt had no fonts at
all: text drew as boxes). Charts leave room for their panel letter above the axes.

**LabLink.** `recipe.draft_from_workspace` of a figure page puts the figure (document,
run ids after the readable renaming, labels) in the manifest, targets its first source and
declares a `figure` output (`protocol.ARTIFACT_KINDS` gains `figure`); the worker's
`_write_figure` writes a job from the graph as it runs (knobs applied) with the uploaded
files' paths and ingest stores (or the synthetic stack), runs the renderer as a
**subprocess** — Qt never loads in the worker — and returns the PNG as the artifact with
the bundle's files listed in `extra["bundle"]`; a render that fails is reported and
skipped, never faked.

**Docs.** MANUAL §2c items 1 and 10 (templates; the CLI and LabLink), §16 J (a figure
from a segmentation to the bundle); `codemap/node_roles.json` describes the Analysis page
as the Figure Editor; curated cards CON-29 (figure document, shadow graph, figure job,
recipe), WF-10 (a render in the window and headless, through LabLink), INV-18 (a figure
reads only Page Outputs; the shadow nodes belong to the sync). Charts thin x tick labels
that would overlap on a narrow axis (seen on the headless fixture: a degenerate range's
long labels).

## Why

A figure page nobody can start quickly is a blank page: the templates are the five figures
the design record's survey says a confocal pipeline earns, bound to whatever the pages
publish, so the first draft is one click and every later edit is the editor's. The
headless renderer exists so the picture does not depend on a window — a batch, a hub, a
re-render after a file changed — and it reuses the panel rather than a second painter so
the two can never drift; the job format is what lets the LabLink worker (which must stay
Qt-free) hand the render to a subprocess with everything it needs. The font registration
came out of the first offscreen render: Qt's offscreen platform finds no system font on
Windows, and DejaVu ships with matplotlib already.

## Files

- `nodelab_v2/figure_recipes.py` — **new**; `nodelab_v2/builtin_figure_recipes/*.json` — **new** (5)
- `nodelab_v2/figure_render.py` — **new**: `build_job`, `build_seeds`, `HeadlessHost`, `render_job`, `render_workspace`, CLI
- `nodelab_v2/figure_editor/fonts.py` — **new**: `ensure_page_fonts`
- `nodelab_v2/figure_editor/panel.py` — `apply_recipe`, `save_as_recipe`, fonts, letter inset for charts
- `nodelab_v2/figure_editor/start_card.py` — the templates row, `recipe_chosen`
- `nodelab_v2/lablink/recipe.py` — `RecipeDraft.figure`, manifest, `figure` output, figure-page drafts
- `nodelab_v2/lablink/worker.py` — `graph_path`, `_write_figure`
- `nodelab_v2/lablink/protocol.py` — `ARTIFACT_KINDS` + `figure`
- `nodegraph/selftest.py` — **hotspot**: `test_figure_recipes_builtin`, `test_figure_headless_render` appended, registered after `test_figure_paper`
- `scripts/_nodelab_v2_phase5_probe.py` — the template check at the start of `_probe_figure_editor`
- `nodegraph/figure/charts.py` — x tick labels thinned when they would overlap
- `codemap/node_roles.json`, `codemap/concepts.md` (CON-29), `codemap/workflows.md` (WF-10), `codemap/invariants.md` (INV-18), `codemap/curated.lock.json`, `codemap/gen/*`, `codemap/STATE.md`
- `MANUAL.md` — §2c, §16 J

## How to verify

On an empty figure page below a labelled output, press *Segment & measure summary* on the
start card: four items land, bound to that output; *Compute sources* draws them. Then:

```
python -m nodelab_v2.figure_render study.nd2graph.json --page Analysis --out fig.png --bundle --write-job job.json
python -m nodelab_v2.figure_render --job job.json --out again.png      # identical bytes
PYTHONUTF8=1 NODEGRAPH_MIN_FREE_RAM=1G python -B -m nodegraph.selftest  # test_figure_recipes_builtin, test_figure_headless_render
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen python -u scripts/_nodelab_v2_phase5_probe.py out.png
```

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (the process still segfaults at `os._exit` *after* that line — the environmental exit crash documented in the step-3 entry; not a regression)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL — 145 ops
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-29, WF-10, INV-18 blessed 2026-11-03; CON-28 re-blessed after a numbering slip)
- [ ] `python scripts/_sync_check.py` -> IN SYNC — fetch and push are blocked on this machine ("SSL peer certificate or SSH remote key was not OK"); the branch is stacked locally, to be rebased and pushed once the certificate issue is fixed
