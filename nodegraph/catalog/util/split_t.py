"""Split T (``util.split_t``) — Fan a time series out into one output per timepoint (each a
single-frame Dataset) or, with Groups set, per RANGE of frames; the whole series also passes
through ``out``."""

from __future__ import annotations

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node


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
            InString("groups", "Groups", field=False, default="", presentation=True,
                     description=
                     "Split into RANGES instead of one output per frame: type the groups as "
                     "0-based frame indices, inclusive ranges, `;` between groups — `0-3; 4-7; "
                     "8-11` — with an optional name in front of a group (`top: 0-3; mid: "
                     "4-7`), or `every 10` for consecutive chunks of ten. The card then grows "
                     "one output PER GROUP, each carrying exactly that subset of the T axis, "
                     "and the per-frame outputs are put away. Blank = one output per frame, up "
                     "to 24 of them; past that the card offers only `out`, and this is how to "
                     "split. A time series is usually hundreds of frames, so `every 10` (or "
                     "`0-99; 100-199`) is the normal way to split one. A wired group "
                     "materializes into a Crop in frames mode keeping those frames at run "
                     "time, shared by every branch that reads it, so this never changes `out` "
                     "or anything this node itself computes. A group reaching past the end is "
                     "labelled so on the card; what lies past the end is dropped at the pull, "
                     "and a group entirely past it is refused with the real T length.")],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE,
    description="Fan a time series out into one output per timepoint (each a single-frame "
                "Dataset, tK = frame K) — or, with Groups set, one output per RANGE of frames "
                "(`every 10`, `0-99; 100-199`); the whole series also passes through 'out'. "
                "The T axis's twin of Split Z.",
)
