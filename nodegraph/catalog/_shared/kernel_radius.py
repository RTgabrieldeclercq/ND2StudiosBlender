"""kernel radius — shared catalog helpers."""

from __future__ import annotations


from nodegraph.engine import EvalContext
from nodegraph.registry import InFloat

from nodegraph.catalog._shared.units import to_pixels_v2

def _win(radius_px: float) -> int:
    """A structuring-element / kernel edge length (odd, ≥1) for a pixel ``radius``."""
    return max(1, int(round(radius_px)) * 2 + 1)
def _require_window(ctx: EvalContext, win: int, r_px: float, *, node: str,
                    param: str = "radius", axis: str = "lateral",
                    degenerate: str = "returns the input unchanged",
                    zero_is_identity: bool = True) -> int:
    """Refuse a µm radius that quantized to a **1-voxel** window (2026-07-30).

    ``_win`` floors at 1, and a 1×1 structuring element makes every one of its consumers
    degenerate: ``white_tophat``/``black_tophat`` and ``morphological_gradient`` become
    *identically zero*, ``grey_opening``/``grey_closing``/``median_filter`` become the
    identity. Nothing failed — the node returned a blank or unchanged image and the user
    found out at the next threshold, as "no objects".

    It is a **calibration** failure, not a tuning one, and it is not exotic: every µm
    default in this catalog (0.3–0.5 µm) was chosen for a high-NA image at ~0.1–0.3 µm/px.
    On the lab's 10×/NA 0.45 plate at 1.7183 µm/px, 0.5 µm is 0.29 px — so `tophat` and
    `morphological_gradient` shipped an all-zero image at their DEFAULTS, and
    median/morphology were no-ops.

    The precedent for refusing rather than degrading is already in this file:
    ``analysis.segment`` and ``analysis.histogram_threshold`` both raise when an area
    filter quantizes to 0 px² ("it quantizes to 0, which is this filter's OFF value"). This
    is that rule applied to the kernel radii, which is where it actually bites.

    A radius of exactly **0** is not refused — that is the user explicitly asking for the
    degenerate case, and an explicit request is not a silent one. Only a *positive* radius
    that the pixel size rounds away is an error, because that is the only case where what
    was asked for and what will happen differ. ``zero_is_identity`` says whether 0 leaves
    the image alone (median/morphology) or blanks it (top-hat / morphological gradient),
    so the message does not offer "set it to 0 to disable" where 0 means a black frame."""
    if win > 1 or r_px <= 0:
        return win
    px_key = "pixel_size_um" if axis == "lateral" else "z_step_um"
    step = ctx.calib(px_key)
    need = 0.5 * float(step) if step else None
    sample = "pixel" if axis == "lateral" else "z step"
    raise ValueError(
        f"{node}: {param} is {r_px:.3g} px after converting from µm"
        + (f" at {px_key}={float(step):g} µm" if step else "")
        + f", which rounds to a 1-voxel {axis} window — at that size the operation "
        f"{degenerate}, silently. "
        + (f"Raise {param} above {need:.3g} µm (half a {sample}) "
           if need else f"Raise {param} until it exceeds half a {sample} ")
        + "so the kernel spans real voxels"
        + (", or set it to exactly 0 if you meant this node to pass the image through."
           if zero_is_identity else
           " — note that setting it to 0 does NOT disable this node, it returns a black "
           "image; delete or mute the node instead.")
        + " (Your data is coarser than the shipped default assumes: the defaults are "
        "sized for a high-NA image at ~0.1–0.3 µm/px.)")
def _radius_px(ctx: EvalContext, name: str, default_um: float) -> float:
    """A lateral radius param (µm) → pixels via ``pixel_size_um`` (recorded read)."""
    r_um = float(ctx.params.get(name, default_um))
    return to_pixels_v2(r_um, "um", pixel_size_um=ctx.calib("pixel_size_um") or 0.1)
def _radius_z_px(ctx: EvalContext, name: str, default_um: float) -> float:
    """The axial radius (``<name>_z``, µm_axial → px via ``z_step_um``); falls back to
    the lateral param when the 3D-only axial socket is unset (paired-float pattern)."""
    r_um = float(ctx.params.get(name + "_z", ctx.params.get(name, default_um)))
    return to_pixels_v2(r_um, "um_axial", z_step_um=ctx.calib("z_step_um") or 0.5)
#: Shared tail for an axial (``_z``) companion socket's hover text. The lateral half's
#: description is caller-specific — "radius" means a smoothing window to Median and a
#: background scale to Top-Hat — but what the AXIAL twin adds is identical every time, so it
#: is written once here and formatted with the lateral socket's label (`build-node-v2` §2:
#: the description is per-socket, but nothing says the prose cannot be composed).
_AXIAL_TWIN_DOC = (
    "The same extent measured along Z, in microns, as a separate control from {lateral} "
    "because microscope voxels are anisotropic: one shared value would cover far more "
    "physical distance through z than across a plane, so the kernel would reach through many "
    "more planes than it does pixels. With a coarse z step even a small value here spans "
    "several planes. 3D only — in 2D each plane is processed independently and nothing leaks "
    "across z.")
def _InRadius(name: str = "radius", label: str = "Radius", default: float = 0.3,
              description: str = ""):
    """A lateral radius socket + its 3D-only axial companion (``<name>_z``). Both are
    ``kernel_param`` — a non-Const Field wired here varies the kernel spatially, so the
    consumer must drop to the plane unit (the kernel-param field gate, V2.04 §6b).

    ``description`` is the LATERAL socket's hover text and is required in practice: five
    nodes share this helper and "radius" means something different in each, so a generic
    string here would be the "restates the type" failure the socket-doc rule forbids. The
    axial twin's text is composed from :data:`_AXIAL_TWIN_DOC`.
    """
    return [
        InFloat(name, label, unit="um", field=True, default=default, kernel_param=True,
                pick_kind="radius",
                description=description),
        InFloat(f"{name}_z", f"{label} Z", unit="um_axial", field=True, default=default,
                available_in={"dim": frozenset({"3D"})}, kernel_param=True,
                description=_AXIAL_TWIN_DOC.format(lateral=label)),
    ]
