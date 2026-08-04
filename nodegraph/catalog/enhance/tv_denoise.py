"""TV Denoise (``enhance.tv_denoise``) — Total-variation (Chambolle) edge-preserving denoise — genuinely 3D in 3D mode (denoise_tv_chambolle is true n-D)."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.map_image import _map_image

# ── TV denoise (Chambolle — genuinely n-D) ──────────────────────────────────────

def _compute_tv(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_tv_chambolle as tv
    ds = ctx.inputs[0]
    weight = float(ctx.params.get("weight", 0.1))
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: tv(a, weight=weight),
        volume_fn=lambda v: tv(v, weight=weight))
register_node(
    _compute_tv, op_key="enhance.tv_denoise", label="TV Denoise",
    category="enhancement",
    inputs=[InDataset(), InFloat("weight", "Weight", unit="", field=True, default=0.1,
                                 description=
                                 "How hard the solver pushes toward a flat, cartoon-like "
                                 "image. HIGHER removes more noise and flattens genuine "
                                 "texture into uniform patches while keeping strong edges "
                                 "sharp — the characteristic look, and a real risk if you "
                                 "then measure texture or interior intensity variation. "
                                 "LOWER stays closer to the input. Around 0.1 is gentle; "
                                 "past ~0.5 flat regions dominate. Dimensionless, and "
                                 "relative to the image's own intensity range, so a value "
                                 "tuned on raw counts will behave differently after a "
                                 "Normalize.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,     # global iterative solver
    supports_2d=True, supports_true_3d=True,
    description="Total-variation (Chambolle) edge-preserving denoise — genuinely 3D in "
                "3D mode (denoise_tv_chambolle is true n-D).")
