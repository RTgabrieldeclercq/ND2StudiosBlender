#!/usr/bin/env python3
"""Serve one LabLink session — the headless ND2Studios worker.

    python nd2studios_worker.py --session /path/to/session

You do not normally run this by hand. A LabLink **hub** spawns it, one process per
session, and talks LWP/1 to it over stdin/stdout; see
:mod:`nodelab_v2.lablink.worker` for the protocol and
``MANUAL.md`` §"LabLink mode" for the setup.

This file exists because a hub config names a *command*, and the name LabLink's own
starter config expects is ``nd2studios-worker``. There is no installed console script
in this repo (it is run from a checkout, like ``run.py``), so the two forms that work
are this launcher and the module:

    "command": ["python", "C:/path/to/ND2StudiosBlender/nd2studios_worker.py",
                "--session", "{session}"]
    "command": ["python", "-m", "nodelab_v2.lablink.worker", "--session", "{session}"]

Prefer whichever names an absolute interpreter path — the hub runs the command with no
shell and does not search a virtualenv for you.

To check this machine can serve LabLink at all, without a hub:

    python nd2studios_worker.py --print-hello
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    # Run from a checkout: make sure the repo root is importable even when the hub
    # launched us with a different working directory (it launches with `cwd` set to the
    # workflow's `workdir`, which an operator may point anywhere).
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    from nodelab_v2.lablink.worker import main as worker_main
    return worker_main()


if __name__ == "__main__":
    # No `freeze_support` and no process pool here on purpose. A worker serves ONE session
    # and the engine's fan-out (nodegraph.parallel) is opt-in per node; if a recipe enables
    # it, the pool's children re-import this module, and the `__name__` guard is what keeps
    # that from starting a second LWP conversation on the same pipe.
    raise SystemExit(main())
