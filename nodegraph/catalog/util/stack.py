"""Stack (T→1) (``util.stack``) — Stack the Timepoint axis → one frame (SNR↑) with a robust combiner (mean/median/sigma-clip/trimmed);."""

from __future__ import annotations

import numpy as np

from dataclasses import replace

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import stack_time as _meta_stack_time
from nodegraph.provider import ArrayProvider
from nodegraph.reducers import reduce as _reduce, reducer_docs
from nodegraph.registry import Granularity, InDataset, Mode, OutDataset
from nodegraph.streaming import TReduceProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.sampling import _sampled

# ── Stack (axis-changing: T→1 SNR stacking with robust fusion) ──────────────────

def _compute_stack(ctx: EvalContext) -> Dataset:
    """Fuse the Timepoint axis into a single frame (T→1), boosting SNR. ``method`` picks
    the combiner (mean/median + the robust ``sigma_clip``/``trimmed_mean`` fusion
    reducers, or max/sum). Drops ``dt_s`` in lockstep with the ``stack_time``
    meta_transform."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("stack needs an image provider on its input Dataset")
    ax = prov.axes
    method = ctx.params.get("__modes__", {}).get("method", "mean")
    new_axes = replace(ax, t=1)
    bd_out = ctx.calib("bit_depth")        # widened by the transform when method == sum
    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((ax.m, 1, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for z in range(ax.z):
                for c in range(ax.c):
                    series = np.stack(
                        [prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                         for t in range(ax.t)], axis=0).astype(float)
                    out[m, 0, z, c] = _reduce(series, (0,), method)  # reduce over T
        stacked = _sampled(ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes),
                           f"stack[{method}]")
        return stacked.with_metadata(dt_s=None, bit_depth=bd_out)
    # C1 (V2.04 §6b sliver): engine-driven tree-reduce over T — the mirror of Z-Project's
    # ZReduceProvider. Monoid combiners (mean/sum/max/min) fold the T series incrementally
    # per tile (memory O(window)); median/sigma_clip/trimmed_mean stack the t-column. No
    # whole plane is ever realized — the eager fallback's SAME nodegraph.reducers keep the
    # bytes identical (one NaN policy across both paths).
    fp = stream_fp("treduce", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    stacked = ds.with_image(
        TReduceProvider(prov, method, fp=fp, cache=cache)).reshaped_axes(new_axes)
    stacked = _sampled(stacked, f"stack[{method}]")
    return stacked.with_metadata(dt_s=None, bit_depth=bd_out)
register_node(
    _compute_stack, op_key="util.stack", label="Stack (T→1)", category="utility",
    inputs=[InDataset()], outputs=[OutDataset()],
    modes=[Mode("method", ["mean", "median", "sigma_clip", "trimmed_mean", "max", "sum"],
                default="mean", label="Combine",
                description=
                "How the timepoints of each (position, z, channel) are fused into the single "
                "output frame. This is the whole point of the node — the combiner decides how "
                "much noise is suppressed and what happens to samples that do not belong "
                "(cosmic rays, hot pixels, a frame where something moved). Anything that "
                "MOVED between frames is smeared by every option, so register first.",
                choice_docs=reducer_docs(
                    ("mean", "median", "sigma_clip", "trimmed_mean", "max", "sum")))],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t"}),
    meta_transform=_meta_stack_time,
    description="Stack the Timepoint axis → one frame (SNR↑) with a robust combiner "
                "(mean/median/sigma-clip/trimmed); drops dt_s.")
