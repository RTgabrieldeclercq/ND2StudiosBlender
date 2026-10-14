# Node queue: the list no longer collapses, Cancel lands, a cancelled run shows STOPPING

- **Date:** 2026-10-13
- **Author:** Gabriel Declercq
- **Branch:** nd2-acquisition-clock (same working tree as the clock change; separate entry)
- **Base:** 889d9d4

## What changed

**The popup keeps its rows.** The Node queue popup (`nodelab_v2/node_queue.py`) keeps each
job's row, keyed by (node, kind), for as long as the job is in line, and updates its text,
badge and buttons in place on every refresh.
- The list is rebuilt only when the line itself changes, and never while a mouse button is
  held.
- Rebuilt rows are shown before the list is measured, so the popup stays as tall as its rows.
- A button acts on its row's current data, looked up by key at click time.
- The popup opens above its button when there is no room below.

**A cancelled run shows STOPPING.** When the running pull or fetch is cancelled, the queue
lists it as **STOPPING** until the worker actually unwinds: "cancelled — stopping at its next
step", not cancellable again. The runner's new `queue_changed` signal fires when the pull slot
goes idle, and the canvas button's count follows it.

## Why

User report: "the queue appears and disappears right after I click on the queue, and
cancelling the nodes appears to not work". Reproduced with real OS clicks on the Windows
platform. The offscreen GUI probe could not see it: it called the popup's methods and never
measured or clicked it.

1. **Measurement of hidden rows.** The popup rebuilt every row widget on each 500 ms tick and
   on every runner signal. A child added to an already-visible widget is shown only on a
   queued call, so the new rows were still hidden when measured, and the list fixed its
   height at 2 px.
2. **What the user saw.** Half a second after opening, the 3-row queue shrank from 190 px to
   an empty 100 px box. A click where a row had been landed outside the popup, which Qt
   closes, and nothing was cancelled.
3. **Deleted buttons.** A click whose press and release straddled a tick released over a
   deleted button, so Cancel was lost there too.
4. **Running cancels.** Cancelling the RUNNING pull dropped its row at once, but the engine
   only stops at a node boundary or progress tick. On a long node the slot stayed busy and
   the next job did not start, with nothing saying why.

Updating rows in place, rather than adding a deferred re-measure, also keeps the button under
the pointer alive. That is the second half of the bug.

## How to verify

Pull a slow node, then double-click two more cards. Click **Node queue · 3**. The list stays
open at full height while it refreshes. **Cancel** the last row: it leaves, and the popup stays
open. **Cancel** the running row: it shows STOPPING until its node reaches a step, then the
next job runs.

Automated: probe block `NQ1`, which now covers kept rows and height across refresh ticks, a
real `QTest` click on Cancel, and STOPPING followed by `queue_changed`. The real-OS repro
(SendInput, guarded to this process's window) is in the session scratchpad, not the repo.

## Files

- `nodelab_v2/node_queue.py`: kept rows, in-place refresh, measured-while-shown, upward placement
- `nodelab_v2/runner.py`: `queue_changed` signal; STOPPING entry in `queue_snapshot`
- `nodelab_v2/window.py`: STOPPING row text; refresh on `queue_changed`
- `scripts/_nodelab_v2_phase5_probe.py`: NQ1 extended
- `codemap/gen/*`: regenerated

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED (NQ1 extended; the process then exits 139, as it does on the untouched base 889d9d4)
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

## Not done / caveats

A node that never reports progress, for example one long native call, still runs to the end
of its step before it stops. The queue now says so (STOPPING) instead of hiding it. Making
such a node interruptible mid-call is outside this change.
