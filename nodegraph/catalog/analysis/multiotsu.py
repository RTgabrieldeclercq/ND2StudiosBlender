"""Multi-Otsu (``analysis.multiotsu``) — Multi-level Otsu → a class-index Voxel raster (0..K-1);."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_volume_p

def _compute_multiotsu(ctx: EvalContext) -> Dataset:
    """Multi-level Otsu → a **class-index** Voxel raster (0..K-1) for multi-population
    segmentation (K = ``classes``). Thresholds are derived per (m,t,c) over that volume's
    histogram; a volume with too few distinct levels degrades to all-class-0 (no crash)."""
    from skimage.filters import threshold_multiotsu
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    k = max(2, int(ctx.params.get("classes", 3)))
    out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    for m, t, c in _each_volume_p(ctx, ax, "multi-otsu"):
        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x).astype(float)
        try:
            th = threshold_multiotsu(vol.ravel(), classes=k)
        except ValueError:
            continue                               # too few distinct levels → all class 0
        out[m, t, :, c] = np.digitize(vol, th)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)
register_node(
    _compute_multiotsu, op_key="analysis.multiotsu", label="Multi-Otsu",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(), InInt("classes", "Classes", default=3, field=False,
                               description=
                               "How many intensity populations to split the histogram into "
                               "— 3 gives background plus two brightness tiers. The output "
                               "is a CLASS INDEX per voxel (0 = dimmest class), not a "
                               "binary mask, so a downstream consumer must pick which "
                               "classes count as foreground. MORE classes means finer "
                               "tiers but each is estimated from less of the histogram, so "
                               "they get less stable; a volume with fewer distinct levels "
                               "than classes degrades to all-zero rather than failing. "
                               "Minimum 2. Thresholds are derived per (m,t,c) volume, so "
                               "the same class index can mean different intensities in "
                               "different frames."),
            InString("name", "Output layer", field=False, default="classes",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the Voxel class-index layer this node writes (values 0..K-1, "
                     "not 0/1). Named `classes` rather than `mask` by default as a reminder "
                     "that it is not directly a binary mask.")],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Multi-level Otsu → a class-index Voxel raster (0..K-1); needs the whole "
                "volume histogram.",
)
