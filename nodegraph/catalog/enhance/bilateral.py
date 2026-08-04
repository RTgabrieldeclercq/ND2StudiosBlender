"""Bilateral Denoise (``enhance.bilateral``) — Edge-preserving bilateral denoise — stack-of-2D even in 3D mode (skimage denoise_bilateral is 2D-only, H15)."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, Granularity, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.kernel_radius import _radius_px
from nodegraph.catalog._shared.map_image import _map_image

# ── Bilateral denoise (edge-aware; skimage is 2D → stack-of-2D, H15) ────────────

def _compute_bilateral(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_bilateral
    ds = ctx.inputs[0]
    sc = float(ctx.params.get("sigma_color", 0.1))
    ss = _radius_px(ctx, "sigma_spatial", 0.2)

    def plane(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return a.copy()
        n01 = (a - mn) / (mx - mn)                 # bilateral σ_color is range-relative
        return denoise_bilateral(n01, sigma_color=sc, sigma_spatial=ss) * (mx - mn) + mn

    return _map_image(ctx, ds, plane_fn=plane)
register_node(
    _compute_bilateral, op_key="enhance.bilateral", label="Bilateral Denoise",
    category="enhancement",
    inputs=[InDataset(),
            InFloat("sigma_spatial", "Sigma spatial", unit="um", field=True, default=0.2,
                    pick_kind="radius",
                    description=
                    "How far the filter reaches, in microns — the same role a Gaussian σ "
                    "plays, setting how much smoothing is available. LARGER smooths more. "
                    "Unlike a Gaussian, reach alone does not blur across boundaries: whether "
                    "a neighbour in range actually contributes is decided by Sigma color."),
            InFloat("sigma_color", "Sigma color", unit="", field=True, default=0.1,
                    description=
                    "How different in BRIGHTNESS a neighbour may be and still be averaged in "
                    "— the edge-preserving half, and the one that decides the character of "
                    "the result. SMALL means only near-identical pixels mix, so edges survive "
                    "but little noise goes away; LARGE lets dissimilar pixels mix and the "
                    "filter degenerates into an ordinary blur that crosses boundaries. "
                    "Measured on the plane rescaled to its own [0,1] range, so 0.1 means "
                    "\"within 10% of this plane's range\" and stays meaningful at any bit "
                    "depth — but it also means the effective threshold shifts between planes "
                    "of differing contrast.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=Granularity.WHOLE_PLANE,      # range-relative σ_color → plane min/max (C1)
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"y", "x"})},
    supports_2d=True, supports_true_3d=False, three_d_fallback="stack_of_2d",
    description="Edge-preserving bilateral denoise — stack-of-2D even in 3D mode "
                "(skimage denoise_bilateral is 2D-only, H15).")
