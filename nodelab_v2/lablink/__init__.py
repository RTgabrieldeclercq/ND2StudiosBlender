"""LabLink mode — ND2Studios as a lab-wide compute service, and as a client of one.

`LabLink <https://github.com/McGheeLab/lablink>`_ moves files between lab machines and runs
workflows on them as they arrive. Its third piece, the **hub**, holds heavy analysis
software warm and runs it for the small machines: a microscope PC opens a session, streams
a file and a few whitelisted knobs, and the second run after a knob change costs
milliseconds instead of the whole pipeline. This package is both ends of that, for
ND2Studios:

:mod:`~nodelab_v2.lablink.worker`
    The **server** half. A headless, Qt-free process the hub spawns — one per session —
    speaking LWP/1 (newline-delimited JSON on stdin/stdout). It loads the recipe's
    ``graph.nd2graph.json`` through the same :mod:`nodegraph.serialize` the editor writes,
    runs it on :func:`nodelab_v2.ops.headless_engine`, and returns tables, quicklooks and
    metrics. This is what makes one capable analysis PC callable by the whole lab.

:mod:`~nodelab_v2.lablink.client`
    The **client** half. A standard-library-only session client so this machine can send
    its own work to somebody else's hub — the GPU box down the hall, or a 128 GB server —
    without either side importing the other's code.

:mod:`~nodelab_v2.lablink.panel`
    The Qt dock: which hub, which sessions, what each one is doing. The only module here
    that imports PySide6, deliberately.

:mod:`~nodelab_v2.lablink.protocol`
    The wire contract, mirrored from LabLink rather than imported — see that module's
    docstring for why the duplication is the right call and how the drift is caught.

:mod:`~nodelab_v2.lablink.artifacts`
    What comes back: CSV tables with previews, PNG quicklooks and thumbnails, metrics
    blobs. The hub is standard-library-only and must never decode an image, so producing
    every returnable byte is this side's job.

**The security boundary, restated because it is easy to lose.** A node names a recipe and
sets knobs the recipe whitelists, within ranges the recipe declares. It can never supply a
command, a path, an argument or a graph — recipes are hub-owned files, curated by the
operator who owns the machine. The worker enforces the half of that the hub structurally
cannot (see :meth:`nodelab_v2.lablink.worker.Worker.validate_recipe`), and refuses rather
than guesses.

**The token is anti-misdirection, not security.** LabLink traffic is plain HTTP; the token
stops you sending into the wrong machine on a shared subnet and does nothing else. Do not
put sensitive data through this link — if it needs to be private, put it on Tailscale or a
private hotspot, which is a transport decision this package cannot make for you.
"""
from __future__ import annotations

from nodelab_v2.lablink.protocol import (
    GRAPH_FORMAT, SOFTWARE_VERSION, WORKER_NAME, WORKER_PROTOCOL_VERSION, WORKER_VERSION,
)

__all__ = ["WORKER_NAME", "WORKER_VERSION", "WORKER_PROTOCOL_VERSION", "GRAPH_FORMAT",
           "SOFTWARE_VERSION"]
