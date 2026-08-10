"""Background Probability (``analysis.background_probability``) — per-voxel p(background) from intensity as a calibrated PROBABILITY rather than a threshold; 2D per-plane vs 3D volumetric anchors."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    DimMode,
    Granularity,
    InDataset,
    InFloat,
    InString,
    OutDataset,
)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.planes import _each_plane_p

# ── p(background): the soft complement of a threshold ──────────────────────────


def _compute_background_probability(ctx: EvalContext) -> Dataset:
    """Per-voxel probability that a voxel is **background**, in [0, 1], from intensity.

    Where a threshold node answers yes/no, this answers *how likely*. That difference is
    only worth a node where something downstream must WEIGH the evidence instead of
    inheriting a decision — deciding whether a detected object sits in the background or on
    a body, whether a boundary between two regions is real, or combining this evidence with
    another probability by multiplying them. For a mask, use ``analysis.histogram_threshold``
    or ``analysis.multiotsu``; a probability thresholded at 0.5 is just a worse threshold.

    Intensity is normalised per unit onto ``u = (I - p_lo) / (p_hi - p_lo)`` and mapped
    through a decreasing logistic — see :mod:`nodegraph.kernels.background_probability` for
    why three parameters and not a mixture model. **The output is one layer per channel**,
    with exactly the input's ``(M,T,Z,C,Y,X)`` shape: this node has no idea which of your
    channels are the same physical material, so combining them (multiply for "background in
    every channel", minimum for "background in the most confident one") is left to whatever
    knows that, and is a channel reduction rather than a thresholding decision.

    **The anchors are per unit, and that is what the 2D/3D lever selects.** In 2D each plane
    is normalised against its own percentiles, so depth attenuation drops out and a deep
    plane is not read as all-background. In 3D the whole volume shares one pair of anchors,
    which is right when the stack is uniformly illuminated and wrong when it is not — the 2D
    setting is the safe one on an attenuating stack, and the two produce different numbers,
    not merely different footprints.

    Resolved spec: category analysis; op ``analysis.background_probability``; reads the image
    only, adds one float Voxel layer; ``DimMode`` lever, 2D ``WHOLE_PLANE`` ``{y,x}`` /
    3D ``WHOLE_VOLUME`` ``{z,y,x}``; no calibration read — every parameter is dimensionless
    by construction, which is the point of normalising first. Kernel:
    :mod:`nodegraph.kernels.background_probability` (numpy only).
    """
    from nodegraph.kernels.background_probability import background_probability

    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(
            "background probability needs an image provider on its input Dataset — it reads "
            "intensity, not an existing layer.")
    ax = ds.axes
    mid = float(ctx.params.get("midpoint", 0.33))
    wid = float(ctx.params.get("width", 0.09))
    ceil = float(ctx.params.get("ceiling", 0.90))
    lo_pct = float(ctx.params.get("lo_pct", 5.0))
    hi_pct = float(ctx.params.get("hi_pct", 99.0))
    if not 0.0 <= lo_pct < hi_pct <= 100.0:
        raise ValueError(
            f"background probability: the anchors must satisfy 0 ≤ low ({lo_pct:g}) < high "
            f"({hi_pct:g}) ≤ 100. They are percentiles of the same unit, so an inverted or "
            "degenerate pair would divide by a negative or zero range and flip the "
            "probability rather than merely rescaling it.")
    if wid <= 0.0:
        raise ValueError(
            f"background probability: width={wid:g} must be positive — it is the softness of "
            "the transition, and zero would be a hard threshold, which is what this node "
            "exists NOT to be. Use analysis.histogram_threshold for a hard cut.")
    if not 0.0 < ceil <= 1.0:
        raise ValueError(
            f"background probability: ceiling={ceil:g} must be in (0, 1] — it is the most "
            "this evidence may ever claim, so a value above 1 is not a probability and 0 "
            "would make every voxel certainly material.")

    out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.float32)
    if ctx.is_volume:
        # one pair of anchors per (m,t,c) volume — see the docstring on when that is right
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    vol = np.stack([prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                                    for z in range(ax.z)], axis=0)
                    out[m, t, :, c] = background_probability(
                        vol, midpoint=mid, width=wid, ceiling=ceil,
                        lo_pct=lo_pct, hi_pct=hi_pct)
    else:
        for m, t, z, c in _each_plane_p(ctx, ax, "background probability"):
            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
            out[m, t, z, c] = background_probability(
                plane, midpoint=mid, width=wid, ceiling=ceil,
                lo_pct=lo_pct, hi_pct=hi_pct)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)


register_node(
    _compute_background_probability, op_key="analysis.background_probability",
    label="Background Probability", category="analysis",
    # reads the IMAGE only — no structure domain is required of the input — and adds one
    # float Voxel layer, exactly like analysis.roi_mask declares.
    adds_domains=frozenset({Domain.VOXEL}),
    inputs=[
        InDataset(),
        InString("name", "Output layer", field=False, default="p_background",
                 layer_out=(Domain.VOXEL,),
                 description=
                 "Name of the float Voxel layer this node writes, one value per voxel in "
                 "[0, ceiling], PER CHANNEL — it has the input's full shape including the "
                 "channel axis, because nothing here knows which channels image the same "
                 "material. Combine them downstream if you need one answer: multiply for "
                 "\"background in every channel\", take the minimum for \"background in the "
                 "most confident one\"."),
        InFloat("midpoint", "Midpoint", unit="", field=False, default=0.33,
                description=
                "Where a voxel is equally background and material, on the normalised "
                "intensity scale set by the two anchors below (0 = the low anchor, 1 = the "
                "high one). RAISING it calls more of the image background, so objects shrink "
                "and dim ones vanish; lowering it grows them and admits haze. This is the "
                "knob that moves reported areas and volumes. The default 0.33 was fitted by "
                "leave-one-field-out on hand-outlined confocal data of bright objects on a "
                "dark background — a reasonable start for similar data, not a constant."),
        InFloat("width", "Width", unit="", field=False, default=0.09,
                description=
                "How sharply the probability falls through the midpoint, in the same "
                "normalised units — the softness of the transition. SMALLER approaches a hard "
                "threshold and pushes almost every voxel to 0 or 1, which throws away the "
                "uncertainty this node exists to report; LARGER leaves a wide band of "
                "genuinely undecided voxels around every edge. It barely moves a mask "
                "thresholded at 0.5, and strongly moves anything that weighs the probability "
                "or multiplies it with another. Fitted at 0.09 on the same data; must be "
                "greater than zero."),
        InFloat("ceiling", "Ceiling", unit="", field=False, default=0.90,
                description=
                "The most this evidence may ever claim — the probability returned for a voxel "
                "far into the background. Below 1 on purpose: intensity alone cannot prove a "
                "voxel is background, because a genuinely dim object looks the same, so a "
                "ceiling of 1 would state certainty the measurement does not support and "
                "would make this layer impossible to overrule by combining it with anything "
                "else. Raise it only if you have checked that dark really does mean empty in "
                "your data."),
        InFloat("lo_pct", "Low anchor", unit="", field=False, default=5.0,
                description=
                "The percentile of each unit's intensities that becomes 0 on the normalised "
                "scale — the background reference. A percentile rather than the minimum so "
                "one dead pixel cannot set it. Raise it if the darkest few percent of your "
                "image is not background (a vignette, a dark artefact); it shifts what "
                "Midpoint means, so the two are read together."),
        InFloat("hi_pct", "High anchor", unit="", field=False, default=99.0,
                description=
                "The percentile that becomes 1 on the normalised scale — the bright "
                "reference. Deliberately under 100 so that saturated pixels do not stretch "
                "the scale and compress everything else toward background. LOWER it when a "
                "large fraction of the frame is saturated, raise it toward 100 only on data "
                "with no saturation and no hot pixels. Must exceed the low anchor."),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Per-voxel p(background) from intensity, as a calibrated PROBABILITY rather "
                "than a threshold — for when the next step must weigh the evidence instead "
                "of inheriting a decision (placing an object in background vs on a body, "
                "combining with another probability). One layer per channel. The 2D/3D lever "
                "selects the extent the normalising anchors are computed over: per plane, so "
                "depth attenuation drops out, or per volume.")
