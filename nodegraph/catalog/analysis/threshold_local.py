"""Local Threshold (``analysis.threshold_local``) — Adaptive local threshold → a Voxel mask (uneven-illumination robust);."""

from __future__ import annotations

import numpy as np

from typing import Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_plane_p

def _compute_threshold_local(ctx: EvalContext) -> Dataset:
    """Adaptive (local) threshold → a Voxel mask — robust to uneven illumination: each
    pixel is compared to a Gaussian-weighted local mean over a ``block_size`` (µm)
    neighbourhood minus ``offset``. Per-plane 2D (WHOLE_PLANE)."""
    from skimage.filters import threshold_local
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    px = ctx.calib("pixel_size_um") or 0.1
    block = max(3, int(round(float(ctx.params.get("block_size", 1.5)) / px)) | 1)  # odd ≥3
    offset = float(ctx.params.get("offset", 0.0))
    mask = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    for m, t, z, c in _each_plane_p(ctx, ax, "local threshold"):
        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
        loc = threshold_local(plane, block_size=min(block, _odd_leq(plane.shape)),
                              offset=offset)
        mask[m, t, z, c] = (plane > loc).astype(np.int64)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), mask)
def _odd_leq(shape: Tuple[int, ...]) -> int:
    """Largest odd block edge that fits a plane (threshold_local needs block ≤ dims)."""
    m = max(3, min(shape))
    return m if m % 2 else m - 1
register_node(
    _compute_threshold_local, op_key="analysis.threshold_local", label="Local Threshold",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InFloat("block_size", "Block size", unit="um", field=True,
                    pick_kind="radius",
                    default=1.5,
                    description=
                    "Diameter of the neighbourhood each pixel is compared against, in "
                    "microns. This is the scale the threshold ADAPTS over, so it must be "
                    "LARGER than the objects you want to keep and SMALLER than the "
                    "illumination unevenness you want to remove — set it near or below "
                    "object size and objects begin thresholding against their own interiors, "
                    "which hollows them into rings. Converted to an odd pixel count and "
                    "clamped to fit the plane."),
            InFloat("offset", "Offset", unit="", field=True, default=0.0,
                    description=
                    "Subtracted from the local mean before comparing, in the image's own "
                    "intensity units — the bias knob. POSITIVE raises the bar and shrinks "
                    "objects, keeping only pixels clearly above their surroundings; NEGATIVE "
                    "lets in pixels barely above the local mean and grows objects. Leave at "
                    "0 and every neighbourhood keeps roughly its brightest half, which is "
                    "why flat background regions come out speckled without a positive "
                    "offset."),
            InString("name", "Output layer", field=False, default="mask",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the Voxel mask layer this node writes (1 = foreground). "
                     "Downstream nodes select it by this name.")],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Adaptive local threshold → a Voxel mask (uneven-illumination robust); "
                "per-plane 2D, block size in µm.")
