"""Unbatch (``util.unbatch``) — fan a batch back out into one output per file."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node

# ── Unbatch (V3.01) ──────────────────────────────────────────────────────────────
#
# The closing half of the canvas' golden point: everything between `util.batch` and here
# ran once over K files, and this is where they become K separate results again.
#
# **It is a pass-through, and that is forced by the engine, not a shortcut.** A node yields
# ONE payload, so K distinct per-member payloads cannot come out of one node however many
# sockets it draws. The card therefore grows one SYNTHETIC output per member in the GUI,
# and each wired one is materialized at graph-build into a real `util.select_batch` tap —
# precisely the arrangement `channel.split` has with `channel.select`, and `io.load`'s
# group taps with `util.select_group`. The full batch also passes through `out`, so a
# downstream node that genuinely wants all K (an export that writes one file per member)
# can still have it.
#
# **The member list comes from the data, not from a param.** `util.batch` stamps
# `batch_file` — one name per `b` — so the card reads its member names off the incoming
# envelope. Nothing has to carry the count separately, which is what stops the sockets
# disagreeing with the wiring after a re-wire: rewire the Batch node and the names change
# with it, and a tap naming a member that is gone refuses instead of quietly selecting
# whichever file now sits at that index.


def _compute_unbatch(ctx: EvalContext) -> Dataset:
    """Unbatch — a pass-through of the whole batch.

    The per-member outputs are GUI-synthetic and become ``util.select_batch`` taps at
    graph-build time (see the module note), so there is nothing to compute here: no axis
    changes, no calibration changes, no layers added. ``TILEABLE`` with no kernel axes and
    no ``meta_transform`` says exactly that.

    Splitting is free at pull time too — a tap is a lazy pin
    (:class:`~nodegraph.provider.BatchSliceProvider`), so K results cost no copy and no
    recompute; each member's chain was addressed by ``b`` all along.
    """
    return ctx.inputs[0]


register_node(
    _compute_unbatch,
    op_key="util.unbatch", label="Unbatch", category="utility",
    inputs=[InDataset()],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Fan a batch back out into one output per file — each a single-file "
                "Dataset, named after its source. The whole batch also passes through "
                "'out'. Put per-object analysis (segment, measure, track) AFTER this: "
                "structure rows carry no batch column, so they cannot tell two files "
                "apart while the batch is still stacked.",
)
