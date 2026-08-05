"""Z-Project (``util.zproject``) — Project the Z axis (max/mean/sum/min/median) → a 2D (z==1) Dataset;."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from typing import Dict, FrozenSet

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import z_project as _meta_z_project
from nodegraph.provider import ArrayProvider
from nodegraph.reducers import reduce as _reduce
from nodegraph.registry import Granularity, InDataset, Mode, OutDataset
from nodegraph.streaming import ZReduceProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.sampling import Z_STAMP, _sampled

# ── Z-Project (axis-changing: z→1 — the meta_transform showcase) ────────────────

#: The reducers, plus ``none`` — the RESET that puts the full Z stack back (V2.21). It is a
#: value of the SAME Mode rather than a second "enabled" control because the node then has
#: exactly one thing to read: a separate switch would leave ``method`` sitting live and
#: inert whenever it was off, which the node charter forbids (`wire-node-v2` §5b/§5c).
_Z_NONE = "none"
_Z_METHODS = ("max", "mean", "sum", "min", "median", _Z_NONE)
#: Footprint per METHOD, not per dim — this node has no lever, so ``footprint_mode`` is
#: ``"method"`` (the ``analysis.threshold``/``scope`` precedent). Every reducer folds the
#: whole z-column of a window, so it reads WHOLE_VOLUME over ``z``; the reset reads nothing
#: at all and forwards the upstream provider, so it is pointwise TILEABLE with NO kernel
#: axis. Declaring WHOLE_VOLUME for the reset too would be safe but would pessimize the
#: scheduler into materializing volumes for a node that is doing nothing.
_ZPROJECT_GRAN: Dict[str, Granularity] = {
    m: (Granularity.TILEABLE if m == _Z_NONE else Granularity.WHOLE_VOLUME)
    for m in _Z_METHODS}
_ZPROJECT_KAX: Dict[str, FrozenSet[str]] = {
    m: (frozenset() if m == _Z_NONE else frozenset({"z"})) for m in _Z_METHODS}

def _compute_zproject(ctx: EvalContext) -> Dataset:
    """Collapse the Z axis with the chosen reducer → a z==1 Dataset, dropping
    ``z_step_um`` and stamping ``z_collapsed`` in lockstep with its ``z_project``
    meta_transform (V2.03 §2 A2). An axis-changing node MUST update calibration and
    axes together — :meth:`reshaped_axes` + :meth:`with_metadata`.

    ``method="none"`` is the RESET: the input Dataset is returned untouched, so Z comes
    back at its full extent with ``z_step_um`` intact. Deliberately NOT a z==1 identity —
    the point is to get the stack back, and the ``z_project`` meta_transform returns the
    envelope verbatim to match. It also must NOT ``_sampled``-stamp: nothing moved, so a
    downstream ``raw`` intensity wire has to keep comparing equal (V2.17 §7b), and it must
    not read calibration, so the memo entry fences on nothing it did not use."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        # checked for EVERY method, reset included: a Z-Project with no image upstream is a
        # wiring mistake either way, and a guard that only fires on some methods would let
        # the user discover it by toggling the picker rather than at the wire.
        raise ValueError("z-project needs an image provider on its input Dataset")
    ax = prov.axes
    method = ctx.params.get("__modes__", {}).get("method", "max")
    if method == _Z_NONE:
        return ds
    new_axes = replace(ax, z=1)
    # a `sum` projection widens the intensity scale (n_z summed samples): the
    # meta_transform already computed the new depth, so SYNC the payload to the env value
    # instead of re-deriving it (§8 — re-deriving would widen twice on a re-pull).
    bd_out = ctx.calib("bit_depth")
    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        # SAME reducers as the lazy ZReduceProvider path (nodegraph.reducers — one
        # NaN policy for both paths; review 2026-07-22: the old plain-op lambdas
        # propagated NaN while the lazy path ignores it → path-dependent bytes)
        out = np.zeros((ax.m, ax.t, 1, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                    out[m, t, 0, c] = _reduce(vol.astype(float), (0,), method)
        projected = _sampled(ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes),
                             f"{Z_STAMP}zproject[{method}]")
        return projected.with_metadata(z_step_um=None, z_collapsed=True,
                                      bit_depth=bd_out)
    # C1: the engine-driven tree-reduce — output tile (iy,ix) folds the base's
    # z-planes of that window via the PartialReducer monoid (median stacks the
    # window's z-column); no whole plane is ever realized (V2.04 §1).
    fp = stream_fp("zreduce", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                   (), prov)
    projected = ds.with_image(
        ZReduceProvider(prov, method, fp=fp, cache=cache)).reshaped_axes(new_axes)
    projected = _sampled(projected, f"{Z_STAMP}zproject[{method}]")
    return projected.with_metadata(z_step_um=None, z_collapsed=True,
                                   bit_depth=bd_out)
register_node(
    _compute_zproject, op_key="util.zproject", label="Z-Project", category="utility",
    inputs=[InDataset()], outputs=[OutDataset()],
    modes=[Mode("method", list(_Z_METHODS), default="max", label="Method",
                description=
                "How the Z planes of each (position, timepoint, channel) are folded into one "
                "plane — and whether they are folded at all. Every reducer but `none` "
                "produces a z == 1 Dataset, drops z_step_um and marks the result "
                "z_collapsed, so nothing downstream can still treat it as a stack. This also "
                "sets the node's footprint: a reducer must read the whole z column, `none` "
                "reads nothing.",
                choice_docs={
                    "max":
                        "The brightest value down each z column. Keeps every in-focus "
                        "feature at full intensity wherever it sits in the stack, which is "
                        "why it is the default for finding objects — but it also keeps the "
                        "brightest NOISE and bright out-of-focus haze, and it is not a "
                        "quantitative sum of signal.",
                    "mean":
                        "The average down each z column. Suppresses noise by roughly √n_z "
                        "and preserves relative intensity, so it is the quantitative choice "
                        "— at the cost of diluting a thin object by the number of empty "
                        "planes it does not occupy.",
                    "sum":
                        "The total down each z column: total signal per (Y,X) position, the "
                        "right reducer when the quantity of interest is integrated "
                        "fluorescence. Values grow with n_z, so the declared bit depth is "
                        "widened to match rather than clipping.",
                    "min":
                        "The dimmest value down each z column. Keeps only what is present at "
                        "EVERY depth, so it strips per-plane debris and is mostly useful for "
                        "estimating a background or for dark-feature (transmitted-light) "
                        "data.",
                    "median":
                        "The middle value down each z column. Like `mean` for noise but "
                        "immune to a single outlier plane — a bright dust speck or one "
                        "saturated slice cannot pull it — which makes it the robust choice on "
                        "stacks with sporadic artefacts.",
                    "none":
                        "No projection: the input passes through untouched, Z at full extent "
                        "and z_step_um intact. This is the RESET, not a z == 1 identity — "
                        "use it to get the stack back after trying a projection without "
                        "having to unwire the node, and note the memo key changes, so the "
                        "projected result you already computed is still cached.",
                })],
    granularity=_ZPROJECT_GRAN, footprint_mode="method", kernel_axes=_ZPROJECT_KAX,
    meta_transform=_meta_z_project,
    description="Project the Z axis (max/mean/sum/min/median) → a 2D (z==1) Dataset; "
                "drops z_step_um and marks z_collapsed. Method 'none' resets the node to "
                "a pass-through: the full Z stack comes back with z_step_um intact.")
