"""progress — shared catalog helpers."""

from __future__ import annotations

import itertools

from typing import Optional

from nodegraph.engine import EvalContext

def _parallel_progress(ctx: EvalContext, total: int, note: str = "",
                       frames: Optional[int] = None):
    """A completion counter safe to call from pool workers (V2.14).

    ``done += 1`` from several threads is a read-modify-write across three bytecodes, so
    counts get lost — and a LOST count matters here beyond cosmetics: the runner exempts
    the final ``done == total`` update from its 50 ms throttle, so an under-count leaves
    the node's bar visibly stuck short of full. ``itertools.count`` is the fix:
    ``next()`` on it is a single atomic C call.

    Pass ``frames`` (normally ``ax.t``) to get the two-level frame/sub report. Units
    finish out of order under a pool, so the frame bar tracks *how much frame-work is
    done*, not which frame a particular worker is on — the only reading of "frame
    progress" that stays monotone when four frames are in flight at once.

    Returns a ``tick()`` to call once per finished unit."""
    counter = itertools.count(1)
    ctx.progress(0, total, note, frames=frames)

    def tick() -> None:
        ctx.progress(next(counter), total, note, frames=frames)

    return tick
class _UnitBar:
    """Two-level progress for a **serial** compute whose units are finer than a frame and
    whose work *inside* a unit can report its own 0-100 percentage (an iterative solver, a
    kernel with a ``progress_cb``).

    Counting whole units only would leave the sub bar frozen for the whole of a long
    solve, which is exactly the case where a user most wants to see movement — a single
    IC-GN/ADMM correlation can run for tens of seconds. So the sub axis is measured in
    *centi-units*: ``sub_total = units_per_frame * 100``, and a unit in flight contributes
    its own percentage. The frame axis still steps exactly once per frame.

    Usage — ``bar.emit(pct)`` from the kernel callback, ``bar.finish_unit()`` once the unit
    lands, ``bar.skip(n)`` for units that advance the bar without work (a t0 that has no
    reference to correlate against)::

        bar = _UnitBar(ctx, frames=ax.t, units_per_frame=ax.z, note="correlating")
        res = run_kernel(..., progress_cb=bar.emit)
        bar.finish_unit(note=f"t={t} z={z}")
    """

    __slots__ = ("ctx", "frames", "upf", "total", "done", "note")

    def __init__(self, ctx: EvalContext, frames: int, units_per_frame: int,
                 note: str = "") -> None:
        self.ctx = ctx
        self.frames = max(1, int(frames))
        self.upf = max(1, int(units_per_frame))
        self.total = self.frames * self.upf
        self.done = 0
        self.note = note
        self.emit(0)

    def emit(self, pct: int = 0, note: Optional[str] = None) -> None:
        """Report ``pct`` (0-100) through the unit currently in flight. Safe as a kernel
        ``progress_cb`` — the engine drops the event when nothing observes, and the runner
        throttles the rest."""
        sub_total = self.upf * 100
        if self.done >= self.total:
            sub = sub_total                  # the last unit landed: end on a full bar
        else:
            # position inside the CURRENT frame, in centi-units. `done % upf` is 0 right
            # after a frame boundary, which is what restarts the sub bar per frame.
            sub = (self.done % self.upf) * 100 + max(0, min(100, int(pct)))
        self.ctx.progress(self.done, self.total, note or self.note,
                          frames=self.frames, sub=sub, sub_total=sub_total)

    def emit_unknown(self, note: Optional[str] = None) -> None:
        """The unit in flight is **one opaque call** — report the frame position and tell the
        UI to sweep the sub bar rather than freeze it. Use this instead of ``emit(0)`` when
        the unit has no internal progress to offer (a single CNN inference): 0% claims no
        work has started, and a determinate bar that then sits still for thirty seconds
        reads as a hang."""
        self.ctx.progress(self.done, self.total, note or self.note,
                          frames=self.frames, sub_unknown=True)

    def finish_unit(self, n: int = 1, note: Optional[str] = None) -> None:
        """One (or ``n``) units finished."""
        self.done = min(self.total, self.done + max(0, int(n)))
        if note is not None:
            self.note = note
        self.emit(0, note)

    #: units the compute is stepping over without doing their work.
    skip = finish_unit
