"""Gamma (``enhance.gamma``) — Power-law γ over a declared intensity scale: `full_range` uses 2**bit_depth-1 (a fixed transfer curve — Cell-Tracker's behaviour, with the significant sensor depth in place of the dtype width),…"""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.full_scale import _declared_full_scale
from nodegraph.catalog._shared.map_image import _map_image

# ── enhancement filters ───────────────────────────────────────────────────────

def _compute_gamma(ctx: EvalContext) -> Dataset:
    """γ correction over a declared intensity scale. ``scale`` (a Mode) picks the reference
    the power law is taken over — the two are genuinely different operations:

    * ``full_range`` (default, V2.13) — the image's **declared full scale**
      ``2**bit_depth - 1`` (:func:`_declared_full_scale`). This is Cell-Tracker's
      ``(a/dtype_max)**g * dtype_max`` with the *significant* sensor depth in place of the
      container width, and it makes γ a **fixed transfer curve**: one input value maps to
      one output value on every plane of a series.
    * ``plane_max`` — the pre-V2.13 behaviour, normalized by each plane's own maximum.
      That is *adaptive*, so the same γ maps the same intensity differently on a dim frame
      than on a bright one and a time series acquires flicker nothing downstream can
      undo. Kept because existing graphs were tuned against it.

    ``full_range`` **degrades to ``plane_max``** when no ``bit_depth`` is declared — after
    a percentile Normalize there is no integer scale, and the observed range IS the scale.
    That fallback is why the footprint stays ``WHOLE_PLANE`` for both settings: a
    plane-global max must be computed per plane, and a tile would normalize by its local
    one (the misdeclared-TILEABLE trap, V2.04 §7).

    Either way the transform is monotone on ``[0, scale]`` and lands back inside it, so
    the intensity scale is unchanged and no ``meta_transform`` is owed (§7c)."""
    ds = ctx.inputs[0]
    g = float(ctx.params.get("gamma", 1.0))
    # Read bit_depth ONLY on the path that consumes it, so a `plane_max` pull is not
    # memo-fenced on a key it ignores (the R1 rule that keeps a 2D gaussian off z_step_um).
    per_plane = ctx.params.get("__modes__", {}).get("scale", "full_range") == "plane_max"
    declared = None if per_plane else _declared_full_scale(ctx)

    def plane(a: np.ndarray) -> np.ndarray:
        mx = declared if declared is not None else (float(a.max()) or 1.0)
        return (a / mx) ** g * mx

    return _map_image(ctx, ds, plane_fn=plane)
register_node(
    _compute_gamma, op_key="enhance.gamma", label="Gamma", category="enhancement",
    inputs=[InDataset(), InFloat("gamma", "Gamma", default=1.0, field=True,
                                 pick_kind="gamma",
                                 description=
                                 "Power-law exponent applied after scaling each plane to "
                                 "its own maximum. BELOW 1 brightens the mid-tones and "
                                 "lifts dim structure out of the background; ABOVE 1 "
                                 "darkens them and increases apparent contrast between "
                                 "bright and dim. 1.0 is a no-op. The plane's peak value is "
                                 "preserved, so this redistributes intensities rather than "
                                 "rescaling the range — but it is still a NON-LINEAR change "
                                 "to the numbers, so any intensity you measure downstream "
                                 "is no longer proportional to photon count. What counts as "
                                 "\"full\" is set by the Scale mode: the declared bit depth "
                                 "(one fixed curve for the whole series) or each plane's own "
                                 "maximum (adaptive, and a plane whose brightest object is "
                                 "dim gets stretched more than its neighbours).")],
    outputs=[OutDataset()],
    # V2.13: the reference scale is a Mode, not the dim lever — γ is pointwise in every
    # dimension, and this chooses the statistical POPULATION the normalization is taken
    # over, which is the same distinction that makes Normalize's `scope` a Mode (H24).
    modes=[Mode("scale", ["full_range", "plane_max"], default="full_range",
                label="Scale",
                description=
                "What counts as \"full brightness\" for the power law — the reference the "
                "image is divided by before γ and multiplied by afterwards. It decides "
                "whether γ is one fixed transfer curve for the whole series or a curve that "
                "re-fits itself to every plane, which is the difference between a "
                "comparable time series and a flickering one.",
                choice_docs={
                    "full_range":
                        "The sensor's declared full scale, 2**bit_depth - 1. One input "
                        "value maps to one output value on every plane, so frames stay "
                        "comparable and a dim frame stays dim. This is Cell-Tracker's "
                        "behaviour. If no bit depth is declared upstream (after a "
                        "percentile Normalize there is no integer scale) it falls back to "
                        "the plane maximum.",
                    "plane_max":
                        "Each plane's own maximum, so every plane is stretched to its own "
                        "brightest pixel. The same γ then maps the same intensity "
                        "differently on a dim frame than on a bright one and a time series "
                        "acquires flicker nothing downstream can undo. Kept because it is "
                        "the pre-V2.13 behaviour and existing graphs were tuned against it.",
                })],
    granularity=Granularity.WHOLE_PLANE,
    kernel_axes=frozenset(),
    description="Power-law γ over a declared intensity scale: `full_range` uses "
                "2**bit_depth-1 (a fixed transfer curve — Cell-Tracker's behaviour, with "
                "the significant sensor depth in place of the dtype width), `plane_max` "
                "the pre-V2.13 per-plane maximum. full_range falls back to the plane max "
                "when no bit_depth is declared, which is why the footprint is WHOLE_PLANE "
                "either way.",
)
