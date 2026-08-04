#!/usr/bin/env python3
"""Launch NodeLab — the ND2Studios node editor.

    python run.py

NodeLab is a Blender-geometry-nodes-style editor on the greenfield ``nodegraph`` engine:
metadata-intelligent 2D/3D nodes, lazy per-tile streaming eval, zones/groups, a live
multi-channel Viewer + Spreadsheet, and CSV/Arrow export.

**History (2026-07-29).** There used to be a second, first-generation editor (``nodelab``
on a vendored ``nd2studios``/``pipeline_kit`` backend) reachable via ``--legacy``; it was
removed once its remaining capabilities were declared obsolete. See
``CodeLog/ClaudesPlan/V2.05_phase7_capability_matrix.md`` §6 for what went with it.
"""
from __future__ import annotations


def main() -> int:
    from nodelab_v2.app import run
    return run()


if __name__ == "__main__":
    # The engine's process pool (nodegraph.parallel) uses the "spawn" start method, so a
    # worker re-imports THIS module to resolve the function it was handed. The
    # `if __name__` guard above is what keeps that import from launching a second GUI, and
    # `freeze_support` is its equivalent for a frozen (PyInstaller/cx_Freeze) build, where
    # the child re-invokes the executable rather than the interpreter. Both are no-ops in
    # ordinary use and load-bearing the moment a node fans out onto processes.
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
