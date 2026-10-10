# GUI probe: LK1's backstop check driven by a real event loop

- **Date:** 2026-10-09
- **Author:** Gabriel Declercq
- **Branch:** timeseries-per-frame-clock
- **Base:** e1ae0e28

## What changed

The phase-5 GUI probe's LK1 check (an unguarded structural edit on a linked page must land
in the status bar through the `LinkedPageError` backstop) now delivers its timer slot from
a nested `QEventLoop.exec()`, which quits when the status bar's message changes or after 5 s.
It no longer spins `app.processEvents()` in a Python loop. No app code changed. The probe now
runs to the end again; on this machine it had stopped after PG14 and never reached the 40-odd
checks after LK1.

## Why

The probe died silently right after PG14 (exit 1 or 127 from Git Bash, `0xC0000409` from
PowerShell; no traceback even under `-X faulthandler`). A line tracer showed the cause:
`LinkedPageError` escaped `app.processEvents()` at the LK1 wait loop, propagated out of
`main()`, and the interpreter crashed tearing Qt down before it could print the traceback.
On PySide6 6.11.1 an exception raised in a slot that a **Python-driven**
`processEvents()` runs is re-raised to that caller. Only from inside a real event loop
(`app.exec()`, `QEventLoop.exec()`) does Qt hand it to `sys.excepthook`, the hook
`window._install_linked_hook` installs. Three isolated scripts confirmed it. The app
itself never calls `processEvents()` (grep `nodelab_v2/`: none), so the backstop is
correct where users meet it, and only the probe's way of driving events was wrong. The
fix makes the probe exercise the mechanism the app relies on, rather than wrapping the spin in
a try/except, which would have tested nothing. The base commit `8fc61af` failed identically
in a clean worktree, so this was pre-existing, not caused by the timeseries change. Whether
an older PySide6 on the other machine routed `processEvents()` slot errors to the hook
(which would explain the earlier green runs) is not established.

## Files

- `scripts/_nodelab_v2_phase5_probe.py` (the LK1 wait only)

## How to verify

`PYTHONUTF8=1 .venv\Scripts\python.exe -B -u scripts/_nodelab_v2_phase5_probe.py out.png`
reaches `ALL PHASE-5 GUI PROBES PASSED`. Before this change it stopped after `PG14`.

## Gates

- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> see the gates of `2026-10-09_bundle-stage-log.md`, run on the branch tip that contains this commit
- [ ] selftest / catalog / codemap: not affected by a probe-only change; run on the branch tip (same entry)
