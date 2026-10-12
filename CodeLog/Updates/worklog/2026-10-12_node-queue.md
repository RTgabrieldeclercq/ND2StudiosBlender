# Node queue

- **Date:** 2026-10-12
- **Author:** Gabriel Declercq
- **Branch:** node-queue
- **Base:** e1ae0e28

## What changed

Every canvas has a **Node queue** button under its top-right corner (`Node queue · N`).
Clicking it opens a list of what the runner holds:
- the running job (a pull, a bake or a Movie-Editor fetch), with how long it has run and
  which node of its chain it is computing (`running 1:05 · computing Gamma (3/8)`);
- each waiting pull and fetch, with its place in line;
- any ingest, which is listed but not cancellable, and says why.

Each row has **Show**, which goes to the node's page and selects and centres its card, and
**Cancel** (**Stop** for a bake). A waiting job leaves the line. A running pull latches its
job cancelled, so it stops at the next node boundary or progress tick, and the next job
starts. **Cancel all** clears the line before stopping the running job.

The runner side is `EngineRunner.queue_snapshot()`, `cancel(node, kind)` and
`cancel_all()`, using the existing cooperative-cancel path (`job.cancelled` →
`PullCancelled`, and the `cancelled` signal that clears the cards).

Separately, **a node that already ran is now served immediately while another job runs**,
even after edits elsewhere on its page. Finished results are filed with a
`cone_signature`: a digest of the target's upstream run graph (ids, ops, params, modes,
wires) and each source's resolved file key. A lookup that misses the current page
revision re-files a result whose signature still matches. The remembered set grew from
4 to 32 results, bounded by a quarter of the memo budget in realized bytes, and always
keeps the newest.

## Why

The request was "allow the user to cancel nodes that are running — a Node Queue button
that shows the list, where each node is, and an option to cancel". Until now a pull you
no longer wanted could only be stopped by editing or deleting a node inside its cone, and
nothing showed what was waiting.

The follow-up was "while a node is queuing, the user should be able to go to past ran
nodes and play a timelapse, adjust LUTs". Playback and LUTs were already off the pull
slot (decode lane, preload pool). They broke because finished results were keyed by
**page revision**, which moves on any edit to the page, and only 4 were kept. Any edit
while waiting, or a fifth branch, made an already-computed result unservable, so viewing
it queued behind the long run.

Keying by what the result computed (its cone) rather than by when (the revision) fixes
this without serving anything stale: an edit upstream of the node still changes its
signature and still queues the re-pull.

The signature is taken at **delivery**, not submit. The run itself resolves a source's
file key (`_node_source_key`), so a signature taken at submit lacked it and never matched
a later lookup. NQ1 caught that.

The queue button sits on its own row under the corner buttons. On a ~790 px canvas, the
centred action pill and mirror/merge buttons already reach the top-right corner.

## How to verify

Load a file, pull a slow node (e.g. a segmentation), then double-click two more cards: the
canvas button reads `Node queue · 3`. Open it: one RUNNING row with elapsed time and the
node being computed, and `#1` and `#2` waiting. **Cancel** on `#2` removes it. **Cancel**
on RUNNING stops it and `#1` starts. While something runs, double-click a card that
already ran, after editing a different branch: it shows at once and plays.

Automated: probe block `NQ1`, and selftest `test_cone_signature`.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [x] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

Hotspot files touched: `nodegraph/selftest.py` (one test appended at the end, plus its
call in `main`; `test_held_views`'s stub runner gained `_result_sig` and `_finished_key`,
which `_rearm_view` now goes through) and `codemap/gen/*` (regenerated).
