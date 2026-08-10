"""Distance Transform (``analysis.edt``) — Euclidean distance transform of a Voxel mask → a µm distance field (anisotropic in 3D);."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import map_units
from nodegraph.registry import DimMode, Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.labels import _resolve_layer, _voxel_layers
from nodegraph.catalog._shared.progress import _parallel_progress

# ── EDT (distance transform of a mask → a physical-µm Voxel field) ──────────────

def _compute_edt(ctx: EvalContext) -> Dataset:
    """Euclidean distance transform of a Voxel mask → a ``distance`` Voxel layer in
    **µm** (anisotropic sampling from pixel/z size). 2D distances each plane; 3D the
    whole volume."""
    ds = ctx.inputs[0]                    # the transform itself goes through `_ndi`
    ax = ds.axes
    # the one raster on the wire, whatever it is called (`_resolve_layer`)
    mask_layer, _note = _resolve_layer(
        _voxel_layers(ds), ctx.layer("mask"), node="EDT", socket="mask",
        what="Voxel layer", where="the `data` input",
        remedy="distance is measured from a mask's foreground, so run Threshold "
               "(analysis.threshold / analysis.histogram_threshold) upstream", ctx=ctx)
    mask_attr = ds.get(Domain.VOXEL, mask_layer)
    if mask_attr is None:                            # pragma: no cover - _resolve_layer
        raise ValueError(f"EDT needs a Voxel mask {mask_layer!r} (run Threshold first)")
    mask6 = mask_attr.values
    px = ctx.calib("pixel_size_um") or 0.1
    out = np.zeros_like(mask6, dtype=float)
    is_3d = ctx.is_volume
    # `ctx.calib` must be read on THIS thread, before the fan-out: the ReadContext records
    # into a plain dict and freezes when the compute returns (V2.02 §8), so a read from a
    # pool worker would both race that dict and risk landing after the fence.
    zs = (ctx.calib("z_step_um") or 0.5) if is_3d else None
    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    tick = _parallel_progress(ctx, len(units), "distance transform", frames=ax.t)

    # Deliberately scipy, NOT the `_ndi` GPU dispatcher. cupyx's distance transform is the
    # one op measured here where the card LOSES: ~2× a single core in 3D, which twelve
    # parallel cores beat outright (measured 0.46× end-to-end even with device access
    # serialized). Its 2D transform is much stronger (61×), but an EDT is only reached
    # through this node, so routing per-dim would buy a win on the cheap half and keep the
    # loss on the expensive one. The thread fan-out below is where this node's speedup is.
    from scipy.ndimage import distance_transform_edt as _edt

    def _edt_one(unit):
        m, t, z, c = unit
        if z is None:
            out[m, t, :, c] = _edt(mask6[m, t, :, c] != 0, sampling=(zs, px, px))
        else:
            out[m, t, z, c] = _edt(mask6[m, t, z, c] != 0, sampling=(px, px))
        tick()

    map_units(_edt_one, units)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)
register_node(
    _compute_edt, op_key="analysis.edt", label="Distance Transform",
    category="analysis",
    # same contract as analysis.threshold: consumes a Voxel mask, adds a Voxel layer
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InString("mask", "Mask layer", field=False, default="mask",
                     layer_in=Domain.VOXEL,
                     description=
                     "Which Voxel mask to measure distances inside — typically a Threshold "
                     "output. Every non-zero voxel counts as foreground, and each one is "
                     "assigned its distance to the NEAREST background voxel; background "
                     "itself stays 0. Naming a layer that is not present is an error."),
            InString("name", "Output layer", field=False, default="distance",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the Voxel distance layer this node writes. Values are in "
                     "MICRONS, not voxels, computed with anisotropic sampling in 3D (pixel "
                     "size laterally, z step axially), so a distance means the same physical "
                     "length in every direction. The classic use is as a height map for "
                     "watershed seeding: a peak sits at the centre of each object, and its "
                     "value is that object's inscribed radius.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Euclidean distance transform of a Voxel mask → a µm distance field "
                "(anisotropic in 3D); 2D per-plane vs 3D volumetric.")
