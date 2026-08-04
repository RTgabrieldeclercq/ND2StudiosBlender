"""Non-Local Means (``enhance.nlm``) — Non-local-means denoise — genuinely volumetric in 3D (denoise_nl_means is true n-D)."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InDataset, InFloat, InInt, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.map_image import _map_image

# ── Non-local means denoise (patch-based; genuinely n-D) ────────────────────────

def _compute_nlm(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_nl_means
    ds = ctx.inputs[0]
    h = float(ctx.params.get("h", 0.1))
    ps = max(1, int(ctx.params.get("patch_size", 3)))
    pd = max(1, int(ctx.params.get("patch_distance", 3)))

    def apply(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return a.copy()
        n01 = (a - mn) / (mx - mn)
        out = denoise_nl_means(n01, patch_size=ps, patch_distance=pd, h=h, fast_mode=True)
        return out * (mx - mn) + mn

    return _map_image(ctx, ds, plane_fn=apply, volume_fn=apply)
register_node(
    _compute_nlm, op_key="enhance.nlm", label="Non-Local Means", category="enhancement",
    inputs=[InDataset(),
            InFloat("h", "Cut-off h", unit="", field=True, default=0.1,
                    description=
                    "Filter strength — how similar two patches must be before they are "
                    "averaged together. HIGHER denoises harder and eventually smears "
                    "genuinely different structures into one another, because dissimilar "
                    "patches start counting as matches; LOWER preserves detail and removes "
                    "less noise. Measured on the block rescaled to its own [0,1] range, so it "
                    "is bit-depth independent. Roughly the noise standard deviation is the "
                    "principled choice; 0.1 is a safe default."),
            InInt("patch_size", "Patch size", unit="px", field=False, default=3,
                  description=
                  "Edge length of the square patch compared between locations, in PIXELS — "
                  "the unit of \"looks the same\". LARGER patches are more selective, so they "
                  "protect texture better but find fewer matches (hence less denoising) and "
                  "cost more; SMALLER patches match easily and can blur fine texture "
                  "together. 3–7 is the usual range, odd values centring cleanly. Minimum 1."),
            InInt("patch_distance", "Patch distance", unit="px", field=False, default=3,
                  description=
                  "How far away, in PIXELS, to search for matching patches. LARGER finds more "
                  "genuine repeats — which is where non-local means beats a purely local "
                  "filter — but the search area grows with the square of this in 2D and the "
                  "CUBE in 3D, making it by far the dominant runtime knob on this node. "
                  "Minimum 1.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,   # plane/volume min/max normalize
    supports_2d=True, supports_true_3d=True,
    description="Non-local-means denoise — genuinely volumetric in 3D "
                "(denoise_nl_means is true n-D).")
