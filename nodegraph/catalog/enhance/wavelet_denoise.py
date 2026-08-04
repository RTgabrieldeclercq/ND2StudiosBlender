"""Wavelet Denoise (``enhance.wavelet_denoise``) — Wavelet (BayesShrink) denoise — stack-of-2D even in 3D mode (skimage denoise_wavelet is 2D-only);."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.map_image import _map_image

# ── Wavelet denoise (the stack-of-2D trap — declared honestly, H15) ─────────────

def _compute_wavelet(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_wavelet as dw
    ds = ctx.inputs[0]

    def plane(a: np.ndarray) -> np.ndarray:
        # BayesShrink estimates the noise σ from the wavelet detail coefficients; on a
        # flat OR sparse plane that estimate degenerates to 0 and the shrink divides by
        # it, returning an all-NaN plane. A blank z-slice / empty channel / sparse mask
        # is a routine input, so: short-circuit the flat case, and if the backend still
        # returns non-finite (sparse), pass the plane through — there is nothing to
        # denoise when noise cannot be estimated.
        if float(a.max()) <= float(a.min()):
            return a.copy()
        out = dw(a, rescale_sigma=True)
        return out if np.all(np.isfinite(out)) else a.copy()

    # Always stack-of-2D: even in 3D mode the footprint stays WHOLE_PLANE, so
    # `ctx.is_volume` is False and this only ever takes the per-plane path.
    return _map_image(ctx, ds, plane_fn=plane)
register_node(
    _compute_wavelet, op_key="enhance.wavelet_denoise", label="Wavelet Denoise",
    category="enhancement",
    inputs=[InDataset()], outputs=[OutDataset()], modes=[DimMode()],
    granularity=Granularity.WHOLE_PLANE,      # full-plane DWT + global σ estimate (C1)
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"y", "x"})},
    supports_2d=True, supports_true_3d=False, three_d_fallback="stack_of_2d",
    description="Wavelet (BayesShrink) denoise — stack-of-2D even in 3D mode "
                "(skimage denoise_wavelet is 2D-only); declared as such (H15).")
