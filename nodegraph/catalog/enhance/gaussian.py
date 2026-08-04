"""Gaussian Blur (``enhance.gaussian``) — Gaussian blur with a metadata-intelligent σ (µm);."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, Granularity, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.map_image import _map_image
from nodegraph.catalog._shared.ndimage import _ndi
from nodegraph.catalog._shared.units import to_pixels_v2

def _compute_gaussian(ctx: EvalContext) -> Dataset:
    """Gaussian blur; a metadata-intelligent σ (µm → px). The 2D/3D lever picks a
    per-plane 2D blur vs an anisotropic 3D blur (σ_z from the axial unit). Streams
    per tile in 2D with the probe-verified tight halo ``int(4σ+0.5)`` (scipy
    ``truncate=4.0``), per lazy volume in 3D (C1 / V2.04)."""
    from scipy.ndimage import gaussian_filter
    ds = ctx.inputs[0]
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    sigma_um = float(ctx.params.get("sigma", 0.5))
    sxy = to_pixels_v2(sigma_um, "um", pixel_size_um=px)
    sz = to_pixels_v2(float(ctx.params.get("sigma_z", sigma_um)),
                      "um_axial", z_step_um=zs)
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: _ndi("gaussian_filter", a, sigma=sxy),
        volume_fn=lambda v: _ndi("gaussian_filter", v, sigma=(sz, sxy, sxy)),
        halo=int(4.0 * sxy + 0.5))
register_node(
    _compute_gaussian, op_key="enhance.gaussian", label="Gaussian Blur",
    category="enhancement",
    inputs=[
        InDataset(),
        InFloat("sigma", "Sigma", unit="um", field=True, default=0.5, kernel_param=True,
                pick_kind="radius",
                description=
                "Lateral blur width in MICRONS, so the same value means the same physical "
                "smoothing at any magnification. LARGER suppresses more noise and more real "
                "detail with it: features smaller than roughly 2σ are erased, so keep σ well "
                "below the size of what you intend to measure. Blurring redistributes "
                "intensity, so a measured peak value drops while total signal is preserved. "
                "Around 0.5 µm is a light denoise; 2 µm is a heavy one."),
        InFloat("sigma_z", "Sigma Z", unit="um_axial", field=True, default=0.5,
                available_in={"dim": frozenset({"3D"})}, kernel_param=True,
                description=
                "Axial blur width in microns, separate from the lateral σ because voxels are "
                "anisotropic — a single σ would blur far more physical distance through z "
                "than across the plane. Keep it below the z step times the number of planes "
                "you are willing to mix, and remember that with a coarse z step even a small "
                "σ_z spans several planes. 3D only; in 2D each plane is blurred "
                "independently and nothing leaks across z."),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    description="Gaussian blur with a metadata-intelligent σ (µm); 2D per-plane vs 3D.",
)
