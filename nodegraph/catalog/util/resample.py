"""Resample (``util.resample``) — Rescale Y,X (and Z in 3D) by a factor;."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from typing import Any, Dict

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import resample as _meta_resample
from nodegraph.provider import ArrayProvider
from nodegraph.registry import DimMode, Granularity, InDataset, InFloat, OutDataset
from nodegraph.streaming import PlaneRealizeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.sampling import _sampled

# ── Resample (axis-changing: rescale Y,X and, in 3D, Z) ─────────────────────────

def _compute_resample(ctx: EvalContext) -> Dataset:
    """Rescale the spatial extent by ``scale_xy`` (and ``scale_z`` in 3D mode). The
    output sizes and the inverse pixel-size update mirror the ``resample``
    meta_transform EXACTLY (``round(size·scale)``) so header==payload; ``resize`` to the
    computed shape guarantees the match regardless of ``zoom`` rounding."""
    from skimage.transform import resize
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("resample needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.is_volume
    sxy = float(ctx.params.get("scale_xy", ctx.params.get("scale", 1.0)) or 1.0)
    sz = float(ctx.params.get("scale_z", 1.0) or 1.0)
    ny, nx = max(1, round(ax.y * sxy)), max(1, round(ax.x * sxy))
    nz = max(1, round(ax.z * sz)) if is_3d else ax.z
    new_axes = replace(ax, z=nz, y=ny, x=nx)

    def resize_plane(a: np.ndarray, m: int, t: int, z: int, c: int) -> np.ndarray:
        return resize(a, (ny, nx), order=1, preserve_range=True)

    def resize_volume(v: np.ndarray, m: int, t: int, c: int) -> np.ndarray:
        return resize(v, (nz, ny, nx), order=1, preserve_range=True)

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((ax.m, ax.t, nz, ax.c, ny, nx), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    if is_3d:
                        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                        out[m, t, :, c] = resize_volume(vol.astype(float), m, t, c)
                    else:
                        for z in range(ax.z):
                            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                            out[m, t, z, c] = resize_plane(plane.astype(float), m, t, z, c)
        res = ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes)
    else:
        # C1 (V2.04 §6b sliver): lazy per-UNIT realize. Resample is non-tileable — the
        # fractional output→input grid + auto anti-aliasing gathers the whole unit — so
        # each output plane (2D) / volume (3D) is resized once on first touch and cached.
        # Geometry changes, so a dedicated PlaneRealizeProvider (MapComputeProvider is 1:1).
        fp = stream_fp("resample", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
        res = ds.with_image(PlaneRealizeProvider(
            prov, new_axes, plane_fn=resize_plane, volume_fn=resize_volume,
            is_volume=is_3d, fp=fp, cache=cache)).reshaped_axes(new_axes)
    # The `resample` meta_transform already produced the post-scale pixel/z size in the
    # envelope (single source of truth); SYNC the payload to it rather than re-deriving
    # (ctx.calib reads the already-transformed value — re-dividing would double-count).
    changes: Dict[str, Any] = {}
    px_out = ctx.calib("pixel_size_um")
    if px_out is not None:
        changes["pixel_size_um"] = px_out
    if is_3d:
        zs_out = ctx.calib("z_step_um")
        if zs_out is not None:
            changes["z_step_um"] = zs_out
    res = _sampled(res, f"resample[xy={sxy:g},z={sz:g}]" if is_3d
                        else f"resample[xy={sxy:g}]")
    return res.with_metadata(**changes) if changes else res
register_node(
    _compute_resample, op_key="util.resample", label="Resample", category="utility",
    inputs=[InDataset(),
            InFloat("scale_xy", "Scale XY", unit="", field=True, default=1.0,
                    description=
                    "Lateral resize factor. BELOW 1 shrinks the image (0.5 halves both Y and "
                    "X, quartering the data and the runtime of everything downstream) and "
                    "ABOVE 1 enlarges it. 1.0 is a no-op. Pixel size scales INVERSELY and "
                    "automatically, so a µm-based control downstream keeps meaning the same "
                    "physical distance — the reason a graph tuned at full resolution still "
                    "behaves correctly after a downsample. Downsampling destroys detail "
                    "permanently and upsampling invents none, so resample for SPEED, never to "
                    "gain resolution. Because the axes change, attribute layers that no "
                    "longer fit are dropped from the bundle."),
            InFloat("scale_z", "Scale Z", unit="", field=True, default=1.0,
                    available_in={"dim": frozenset({"3D"})},
                    description=
                    "Axial resize factor, separate from the lateral one so anisotropic data "
                    "can be made isotropic — the usual reason to touch it. The z step scales "
                    "inversely in step, so physical measurements stay correct. Setting it to "
                    "pixel_size_um / z_step_um yields cubic voxels, which is what makes a 3D "
                    "distance transform or a mesh come out undistorted. 3D only.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX, meta_transform=_meta_resample,
    description="Rescale Y,X (and Z in 3D) by a factor; pixel size scales inversely "
                "(finer when upsampling).")
