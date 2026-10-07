"""Split T (``util.split_t``) — Fan a time series out into one output per timepoint (each a
single-frame Dataset) or, with Groups set, per RANGE of frames; the whole series also passes
through ``out``."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.split_grouping import grouping_mode, grouping_sockets


def _compute_split_t(ctx: EvalContext) -> Dataset:
    """Split T — a domain-transparent pass-through of the full series, the T-axis twin of
    ``util.split_z`` (2026-10-07).

    Its single real ``out`` socket carries the input unchanged. The GUI adds one synthetic
    per-frame output socket (``t0…tT-1``, labelled with the frame's index and, when the series
    carries a frame interval, its time) while the series is short enough to fan out
    (:data:`nodelab_v2.document.FANOUT_CAP`), and each wired per-frame tap is materialized into
    a real ``util.select_frame`` at graph-build time — the engine is one-payload-per-node, so
    distinct per-frame payloads come from real select taps, not from this node. A time series
    is usually hundreds of frames, which is what the ``groups`` text is for: ``every 10`` or
    ``0-99; 100-199`` gives one output per RANGE (a Crop in frames mode at run time). No
    calibration or axes change here (``TILEABLE``, no meta_transform).
    """
    return ctx.inputs[0]


register_node(
    _compute_split_t,
    op_key="util.split_t", label="Split T", category="utility",
    inputs=[InDataset(description=
                      "The time series to fan out. Its frames appear as one output socket each "
                      "on the card (`t0…`, with the frame's time when the series has a frame "
                      "interval) while there are few enough to show; type Groups to split a "
                      "long series into ranges. `out` still carries the whole series."),
            *grouping_sockets('frame', 'T',
                              'a Crop in frames mode keeping those frames', 10)],
    outputs=[OutDataset("out")],
    modes=[grouping_mode('frame', 'T')],
    granularity=Granularity.TILEABLE,
    description="Fan a time series out into one output per timepoint (each a single-frame "
                "Dataset, tK = frame K) — or, with Groups set, one output per RANGE of frames "
                "(`every 10`, `0-99; 100-199`); the whole series also passes through 'out'. "
                "The T axis's twin of Split Z.",
)
