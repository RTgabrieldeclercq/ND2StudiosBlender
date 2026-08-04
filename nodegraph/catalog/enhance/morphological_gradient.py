"""Morphological Gradient (``enhance.morphological_gradient``) — Dilation − erosion edge map; 2D per-plane vs anisotropic 3D."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InBool, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.full_scale import _declared_full_scale
from nodegraph.catalog._shared.kernel_radius import (
    _InRadius,
    _radius_px,
    _radius_z_px,
    _require_window,
    _win,
)
from nodegraph.catalog._shared.map_image import _map_image
from nodegraph.catalog._shared.ndimage import _ndi

# ── Morphological Gradient (edge map) ───────────────────────────────────────────

def _compute_morph_gradient(ctx: EvalContext) -> Dataset:
    """Dilation − erosion edge map, optionally mixed back over the original.

    ``rescale`` + ``blend`` together reproduce Cell-Tracker's plugin, which never returns
    a bare gradient: it stretches the edge magnitude across the full intensity range and
    then mixes ``blend·original + (1-blend)·edges`` (its UI default is 0.5). The stretch is
    what makes the mix meaningful — a raw morphological gradient of dim data is far darker
    than the image it would be blended with, so without it a 50/50 mix is visually all
    original. Both default to off/0, so the node still returns the pure edge magnitude
    unless asked otherwise.

    The stretch reads the **unit's maximum**, a plane/volume-global statistic, hence the
    ``WHOLE_PLANE``/``WHOLE_VOLUME`` footprint rather than ``TILEABLE`` (a tiled stretch
    would normalize each tile by its own local peak and stitch seams). ``granularity``
    resolves per *dim* only, so it cannot follow a socket — the declaration has to cover
    both settings (V2.04 §7)."""
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    ry = _radius_px(ctx, "radius", 0.3)
    # dilate(a,1) - erode(a,1) == 0 everywhere — an all-black output, exactly like top-hat
    _DEGEN = ("takes the difference of the image with itself and returns an entirely "
              "BLACK image")
    wy = _require_window(ctx, _win(ry), ry, node="morphological gradient",
                         degenerate=_DEGEN, zero_is_identity=False)
    if ctx.is_volume:
        rz = _radius_z_px(ctx, "radius", 0.3)
        wz = _require_window(ctx, _win(rz), rz, node="morphological gradient",
                             param="radius_z", axis="axial", degenerate=_DEGEN,
                             zero_is_identity=False)
    else:
        wz = wy
    blend = min(1.0, max(0.0, float(ctx.params.get("blend", 0.0))))
    rescale = bool(ctx.params.get("rescale", False))
    # eager — a lazy closure must not touch ctx (V2.04 §6b). None ⇒ no declared scale, so
    # the stretch targets the unit's own peak (a plain normalize to [0,1]).
    full = _declared_full_scale(ctx) if rescale else None

    def mix(src: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if rescale:
            mx = float(grad.max())
            if mx > 0.0:
                grad = grad / mx * (full if full is not None else 1.0)
        if blend <= 0.0:
            return grad
        return blend * src + (1.0 - blend) * grad

    return _map_image(
        ctx, ds,
        plane_fn=lambda a: mix(a, _ndi("morphological_gradient", a, size=(wy, wy))),
        volume_fn=lambda v: mix(v, _ndi("morphological_gradient", v,
                                        size=(wz, wy, wy))),
        halo=wy // 2)                # single dilate−erode pass: one-window reach
register_node(
    _compute_morph_gradient, op_key="enhance.morphological_gradient",
    label="Morphological Gradient", category="enhancement",
    inputs=[InDataset(), *_InRadius(description=
            "Width of the neighbourhood compared, in microns — effectively the THICKNESS of "
            "the edges produced, since each output voxel is the local maximum minus the local "
            "minimum over this window. SMALL gives thin, precise edges that also respond to "
            "noise; LARGER gives thick, smooth edges and merges boundaries that are closer "
            "together than the window. The output is an edge MAGNITUDE, not brightness — "
            "object interiors go to zero — so nothing downstream should read these values as "
            "intensity."),
            InFloat("blend", "Blend original", unit="", field=True, default=0.0,
                    description=
                    "How much of the ORIGINAL image to mix back over the edge map, in [0,1]: "
                    "the output is blend*original + (1-blend)*edges. 0 (the default) is the "
                    "pure gradient — object interiors black. 1 is the original untouched. "
                    "Around 0.5 you get the original with its boundaries pushed up, which is "
                    "what Cell-Tracker's plugin defaults to and what helps a downstream "
                    "watershed find borders it would otherwise miss. A mix only makes visual "
                    "sense with Rescale on, since a raw gradient is much darker than the "
                    "image it is being mixed with."),
            InBool("rescale", "Rescale to full range", unit="", field=False, default=False,
                   description=
                   "Stretch the edge magnitude across the declared full range "
                   "(2**bit_depth-1, or [0,1] when no bit depth is declared) before "
                   "blending — Cell-Tracker does this unconditionally. Switch it on to "
                   "reproduce a Cell-Tracker recipe or to make Blend useful. Leave it OFF "
                   "for measurement: the stretch is per plane and per volume, so the same "
                   "boundary gets a different number depending on the strongest edge in its "
                   "unit.")],
    outputs=[OutDataset()], modes=[DimMode()],
    # WHOLE_PLANE, not TILEABLE: `rescale` reads the unit's maximum — see the compute.
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,
    description="Dilation − erosion edge map; 2D per-plane vs anisotropic 3D. Optional "
                "full-range stretch + blend-with-original reproduce Cell-Tracker's plugin.")
