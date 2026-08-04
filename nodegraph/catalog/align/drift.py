"""Drift Correction (``align.drift``) — Rigid drift correction across Timepoints (phase cross-correlation on a reference plane);."""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, OutDataset
from nodegraph.streaming import MapComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.drift_layers import _layers_drift
from nodegraph.catalog._shared.sampling import _sampled

# ── Drift correction (registration — axis-preserving, stores the shift) ─────────

def _compute_drift(ctx: EvalContext) -> Dataset:
    """Rigid drift correction across Timepoints via phase cross-correlation on a
    reference channel/plane; the estimated per-frame (Δy, Δx) shift is applied to every
    z and channel and also **stored as Frame-domain attributes** (``drift_y``/
    ``drift_x``) — provenance, per V2.03 §2 (registration stores its transform).
    Axis-preserving (geometry unchanged)."""
    from scipy.ndimage import shift as ndi_shift
    from skimage.registration import phase_cross_correlation
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("drift correction needs an image provider on its input Dataset")
    ax = prov.axes
    ref_z = ax.z // 2
    # Estimate the per-frame (Δy,Δx) shift EAGERLY — a global FFT per frame on the
    # reference channel/plane is inherently whole-plane (and cheap: one plane per t). The
    # APPLY (register-once/apply-all: the SAME shift on every c,z preserves colocalization)
    # is deferred to a lazy per-plane provider (C1 / V2.04 §6b sliver).
    shifts: Dict[Tuple[int, int], Tuple[float, float]] = {}
    dy = np.zeros((ax.m, ax.t), dtype=float)
    dx = np.zeros((ax.m, ax.t), dtype=float)
    for m in range(ax.m):
        ref = prov.get_region(0, m, 0, ref_z, 0, 0, ax.y, 0, ax.x).astype(float)
        for t in range(ax.t):
            mov = prov.get_region(0, m, t, ref_z, 0, 0, ax.y, 0, ax.x).astype(float)
            sh = phase_cross_correlation(ref, mov, upsample_factor=10)[0]
            shifts[(m, t)] = (float(sh[0]), float(sh[1]))
            dy[m, t], dx[m, t] = shifts[(m, t)]

    def apply_shift(a: np.ndarray, m: int, t: int) -> np.ndarray:
        return ndi_shift(a, shift=shifts[(m, t)], order=1, mode="constant")

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    for z in range(ax.z):
                        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                        out[m, t, z, c] = apply_shift(plane, m, t)
        res = ds.with_image(ArrayProvider(out))
    else:
        # WHOLE_PLANE unit: ndi_shift needs the whole plane (mode='constant' fills the
        # vacated edge); the shifts are baked from the eager estimate, and fold into the
        # provider fp via the base fingerprint (a base change re-estimates + re-keys).
        fp = stream_fp("drift", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
        res = ds.with_image(MapComputeProvider(
            prov, lambda a, m, t, z, c, *_: apply_shift(a, m, t),
            unit="plane", fp=fp, cache=cache))
    res = _sampled(res, "align.drift")     # content shifts under a fixed index grid
    # ...and that is exactly why the spatial origin is DROPPED. The shift is per-(m, t),
    # so no single per-M corner can describe where the content now sits; absent is the
    # honest answer and the overlay reports "cannot be placed" instead of drawing a
    # convincing picture off by the drift. The transform is not lost — it is right there in
    # the drift_y/drift_x layers below, so a future per-frame origin could restore it.
    res = res.with_metadata(origin_um=None)
    res = res.with_layer(Domain.FRAME, "drift_y", dy)
    return res.with_layer(Domain.FRAME, "drift_x", dx)
register_node(
    _compute_drift, op_key="align.drift", label="Drift Correction", category="registration",
    extra_layers=_layers_drift,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.FRAME}),
    inputs=[InDataset()], outputs=[OutDataset()],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="Rigid drift correction across Timepoints (phase cross-correlation on a "
                "reference plane); stores the per-frame shift as Frame attributes.")
