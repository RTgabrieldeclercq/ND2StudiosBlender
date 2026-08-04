"""Difference of Gaussians (``enhance.dog``) — Band-pass (blob) enhancement; low σ derives from the diffraction limit; high σ 0 ⇒ auto 1.6× low."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import DimMode, InBool, InDataset, InFloat, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.full_scale import _declared_full_scale
from nodegraph.catalog._shared.kernel_radius import _AXIAL_TWIN_DOC, _radius_px, _radius_z_px
from nodegraph.catalog._shared.map_image import _map_image

# ── Difference of Gaussians (band-pass feature enhancement) ─────────────────────

def _compute_dog(ctx: EvalContext) -> Dataset:
    """Band-pass (difference-of-Gaussians) enhancement.

    ``rescale`` (V2.13) reproduces Cell-Tracker's DoG plugin, which does not return the
    band-pass itself: it clips the negative lobe to zero and stretches what is left across
    the full intensity range (``diff/diff.max() * dtype_max``). That makes the result a
    *display-ready* image whose absolute values carry no physical meaning, which is why it
    is off by default here — a raw DoG is a difference image centred near zero, and
    keeping it that way is what lets a threshold downstream mean something.

    The stretch reads the **unit's maximum**, a plane/volume-global statistic, so this node
    declares ``WHOLE_PLANE``/``WHOLE_VOLUME`` rather than ``TILEABLE``: a tiled DoG would
    stretch every tile by its own local peak and stitch visible seams. That costs 2D
    tiling even when ``rescale`` is off — ``granularity`` resolves per *dim* only
    (``NodeSpec.resolve_granularity``), so it cannot follow a socket, and the honest
    declaration is the one that covers both settings (V2.04 §7)."""
    from skimage.filters import difference_of_gaussians as dog
    ds = ctx.inputs[0]
    lo = _radius_px(ctx, "low_sigma", 0.15)
    px = ctx.calib("pixel_size_um") or 0.1
    hi_um = float(ctx.params.get("high_sigma", 0.0))
    hi = (hi_um / px) if hi_um > 0 else None
    rescale = bool(ctx.params.get("rescale", False))
    # eager (a lazy closure must not touch ctx — V2.04 §6b); None ⇒ no declared scale, so
    # the stretch targets the unit's own peak, which is a plain clip-and-normalize to [0,1]
    full = _declared_full_scale(ctx) if rescale else None

    def stretch(a: np.ndarray) -> np.ndarray:
        """CT's post-pass: drop the negative lobe, then scale the positive band to the
        declared full range (or to [0,1] when no bit depth is declared)."""
        out = np.clip(a, 0.0, None)
        mx = float(out.max())
        if mx <= 0.0:
            return out
        return out / mx * (full if full is not None else 1.0)
    # influence radius from the LARGER σ; high_sigma 0 ⇒ skimage's implicit 1.6·low
    # (a halo from low σ alone leaves border errors — C1 audit / V2.04 §2)
    hi_eff = hi if hi is not None else 1.6 * lo
    if hi is not None and hi < lo:
        # validate eagerly: under C1 the kernel runs at TILE-PULL time, so a backend
        # ValueError would otherwise be memoized into a poisoned lazy payload and
        # surface far from the misconfigured node (review 2026-07-22)
        raise ValueError(
            f"high_sigma ({hi_um} µm → {hi:.2f} px) must be ≥ low_sigma ({lo:.2f} px)")
    halo = int(4.0 * max(lo, hi_eff) + 0.5)
    post = stretch if rescale else (lambda a: a)
    if ctx.is_volume:
        zs = ctx.calib("z_step_um") or 0.5
        loz = _radius_z_px(ctx, "low_sigma", 0.15)
        hiz = (hi_um / zs) if hi_um > 0 else None
        low3 = (loz, lo, lo)
        high3 = (hiz, hi, hi) if hi is not None else None
        return _map_image(ctx, ds, plane_fn=lambda a: post(dog(a, lo, hi)),
                          volume_fn=lambda v: post(dog(v, low3, high3)), halo=halo)
    return _map_image(ctx, ds, plane_fn=lambda a: post(dog(a, lo, hi)), halo=halo)
register_node(
    _compute_dog, op_key="enhance.dog", label="Difference of Gaussians",
    category="enhancement",
    inputs=[
        InDataset(),
        InFloat("low_sigma", "Low σ", unit="um", field=True, default=0.15,
                pick_kind="radius", pick_peer="high_sigma",
                derive="0.5*0.61*(emission_nm or 520)/(na or 1.4)/1000", kernel_param=True,
                description=
                "The SMALL blur of the pair, in microns — it sets the fine end of the band "
                "that survives. Detail finer than this is smoothed away as noise, so raise "
                "it to reject more noise and lower it to keep sharper structure. Auto "
                "derives it from the diffraction limit of the current optics (emission λ "
                "and NA), which is the smallest real feature the microscope can form, and "
                "is usually the right choice. Together with High σ this makes a band-pass: "
                "output is the low-blur minus the high-blur, so it is a DIFFERENCE image "
                "centred near zero with negative values, not an intensity image."),
        InFloat("low_sigma_z", "Low σ Z", unit="um_axial", field=True, default=0.15,
                available_in={"dim": frozenset({"3D"})}, kernel_param=True,
                description=_AXIAL_TWIN_DOC.format(lateral="Low σ")),
        InFloat("high_sigma", "High σ", unit="um", field=True, default=0.0,
                pick_kind="radius", pick_peer="low_sigma",
                kernel_param=True,
                description=
                "The LARGE blur of the pair, in microns — it sets the coarse end of the "
                "band. Structure broader than this is treated as background and subtracted, "
                "so LOWER it to strip more background (and more of your object's broad "
                "core), raise it to keep more. 0 means AUTO: 1.6× Low σ, the ratio that "
                "best approximates a Laplacian-of-Gaussian and the standard choice for "
                "blob detection. Must be ≥ Low σ — an inverted pair is refused up front "
                "rather than failing later inside the filter."),
        InBool("rescale", "Rescale to full range", unit="", field=False, default=False,
               description=
               "Turn the band-pass into a display-ready image the way Cell-Tracker's DoG "
               "plugin does: clip the negative lobe to zero, then stretch what is left "
               "across the declared full range (2**bit_depth-1, or [0,1] when no bit depth "
               "is declared). Switch it on to REPRODUCE a Cell-Tracker recipe, or when the "
               "next node wants a positive intensity image. Leave it OFF for measurement: "
               "the stretch is per plane and per volume, so the same structure gets a "
               "different number depending on the brightest thing in its unit, and the "
               "clipped half of the signal is gone. Off, the output is the true difference "
               "image — centred near zero, negatives intact."),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    # WHOLE_PLANE, not TILEABLE: `rescale` reads the unit's maximum, and granularity
    # resolves per dim only, so it cannot follow the socket — see the compute docstring.
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,
    description="Band-pass (blob) enhancement; low σ derives from the diffraction limit; "
                "high σ 0 ⇒ auto 1.6× low. 2D per-plane vs anisotropic 3D. Optional "
                "clip-and-stretch to the full range reproduces Cell-Tracker's DoG.")
