"""Deconvolve (``enhance.deconvolve``) — Richardson–Lucy deconvolution with a PSF derived from optics metadata (NA, emission λ, pixel/z size); 2D lateral PSF vs 3D anisotropic PSF."""

from __future__ import annotations

import numpy as np

from typing import Any, Optional, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import value_rescaled as _meta_value_rescaled
from nodegraph.provider import ArrayProvider
from nodegraph.registry import DimMode, Granularity, InDataset, InFloat, InInt, OutDataset
from nodegraph.streaming import MapComputeProvider, VolumeComputeProvider, stream_fp

from nodegraph.catalog._base import register_node

# ── metadata-intelligent PSF (the directive A showcase) ───────────────────────

def diffraction_sigmas(emission_nm: Optional[float], na: Optional[float],
                       pixel_size_um: Optional[float], z_step_um: Optional[float],
                       is_3d: bool) -> Tuple[float, ...]:
    """Gaussian-approximation PSF sigmas **derived from optics metadata**: lateral
    ``σ_xy ≈ 0.21·λ/NA`` and axial ``σ_z ≈ 0.66·λ·n/NA²`` (n≈1.5 immersion), converted
    to pixels via ``pixel_size_um`` / ``z_step_um``. Returns ``(σ_y,σ_x)`` in 2D or
    ``(σ_z,σ_y,σ_x)`` in 3D. (A Gaussian PSF is the portable default; a Gibson–Lanni /
    measured PSF is a backend swap behind the same derived sampling.)"""
    lam_um = (emission_nm or 520.0) / 1000.0
    na = na or 1.4
    sxy_um = 0.21 * lam_um / na
    sxy = sxy_um / (pixel_size_um or 0.1)
    if not is_3d:
        return (sxy, sxy)
    sz_um = 0.66 * lam_um * 1.5 / (na * na)
    sz = sz_um / (z_step_um or 0.5)
    return (sz, sxy, sxy)
def gaussian_psf(sigmas: Sequence[float], *, radius_factor: float = 3.0) -> np.ndarray:
    """A normalized n-D Gaussian kernel with the given per-axis ``sigmas`` (pixels)."""
    sig = tuple(max(0.5, float(s)) for s in sigmas)
    radii = [max(1, int(round(radius_factor * s))) for s in sig]
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    g = np.ones_like(grids[0], dtype=float)
    for coord, s in zip(grids, sig):
        g = g * np.exp(-(coord.astype(float) ** 2) / (2.0 * s * s))
    total = g.sum()
    return g / total if total else g
# ── Deconvolve (the two-mode metadata-intelligent PSF flagship) ───────────────

def _rl(image: np.ndarray, psf: np.ndarray, iters: int) -> np.ndarray:
    """Richardson–Lucy against an explicit PSF **array** — the reference form.

    Kept because it is the definition the fast path is checked against
    (:func:`_rl_gaussian`), not because anything calls it in a run: skimage convolves with
    the full n-D kernel, which for a 3D volume means ``2·iters`` FFT convolutions of the
    whole thing — and scipy re-transforms the padded PSF on every one of them. On the lab's
    210×1024² unit at 10 iterations that measured **226.5 s and a 20.9 GB peak**."""
    from skimage.restoration import richardson_lucy
    mx = float(image.max()) or 1.0
    out = richardson_lucy(image / mx, psf, num_iter=iters, clip=False)
    return out * mx
def _gauss_kernel1d(sigma: float, radius: int, dtype: Any) -> np.ndarray:
    """One normalized 1-D Gaussian tap set — the same expression :func:`gaussian_psf`
    evaluates per axis, so the outer product of these IS its kernel (it normalizes the n-D
    product by its total sum, and for a separable kernel that equals the product of the
    per-axis sums)."""
    x = np.arange(-int(radius), int(radius) + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(dtype, copy=False)
def _gauss_blur_zero(a: np.ndarray, sig: Tuple[float, ...],
                     rad: Tuple[int, ...]) -> np.ndarray:
    """A **zero-padded** separable Gaussian blur — ``scipy.signal.convolve(a,
    gaussian_psf(σ), 'same')`` without ever forming the n-D kernel.

    Hand-rolled rather than a ``scipy.ndimage.gaussian_filter`` call for one measured
    reason: for a 3D volume the axial kernel is the tall one (23 taps at the lab's NA 0.8 /
    0.288 µm sampling) and ``correlate1d`` along axis 0 walks taps a whole plane apart, one
    element at a time. Reducing z with ``tensordot`` instead turns each output plane into a
    BLAS matrix-vector product over a contiguous stack of input planes: same arithmetic,
    cache-friendly, and no temporary bigger than one plane. Measured on a 24×1024² slab,
    10 RL iterations: ``gaussian_filter`` 129 s versus 13 s here.

    Sequential 1-D zero-padded passes equal the n-D zero-padded convolution: a shift along
    one axis cannot move a sample out of range on *another* axis, so every term the
    separable form drops at the boundary is a term the n-D form multiplies by a zero sample
    anyway. And the Gaussian is symmetric, so correlation and convolution agree.

    **Deliberately not routed through :func:`_ndi`.** The device call would have to be
    ``gaussian_filter(..., radius=…)`` to match ``gaussian_psf``'s kernel exactly, and
    ``cupyx``'s signature has no ``radius`` — so the call would raise, and
    :func:`nodegraph.gpu.ndimage` retires a failing op **for the whole session**, which
    would silently take the GPU away from ``enhance.gaussian`` and every other filter that
    shares the name. ``truncate=3.0`` is not a safe substitute either: it is
    ``floor(3σ+0.5)`` where ``gaussian_psf`` is banker's ``round(3σ)``, which disagree
    whenever 3σ lands on a half. A GPU RL wants the volume resident on the device across
    all ``2·iters`` blurs anyway, not a host round trip per blur; that is its own piece of
    work, not a one-line dispatch."""
    from scipy.ndimage import correlate1d
    lateral = range(1, a.ndim) if a.ndim >= 3 else range(a.ndim)
    for ax in lateral:                       # contiguous-ish axes: scipy is at its best
        a = correlate1d(a, _gauss_kernel1d(sig[ax], rad[ax], a.dtype), axis=ax,
                        mode="constant", cval=0.0)
    if a.ndim < 3:
        return a
    kz = _gauss_kernel1d(sig[0], rad[0], a.dtype)
    r, nz = int(rad[0]), a.shape[0]
    res = np.empty_like(a)
    for z in range(nz):                      # zero padding == simply clipping the window
        lo, hi = max(0, z - r), min(nz, z + r + 1)
        res[z] = np.tensordot(kz[lo - z + r:hi - z + r], a[lo:hi], axes=(0, 0))
    return res
def _rl_gaussian(image: np.ndarray, sigmas: Sequence[float], iters: int) -> np.ndarray:
    """Richardson–Lucy with a **separable** Gaussian PSF — the same computation as
    ``_rl(image, gaussian_psf(sigmas), iters)``, without the FFTs or their padded complex
    intermediates.

    **Why this is the same maths, not an approximation.** The PSF this node builds is
    always a Gaussian (:func:`gaussian_psf`), so it is separable, and its mirror equals
    itself — both convolutions in an RL step are therefore the identical separable blur
    (:func:`_gauss_blur_zero`, which documents the boundary and symmetry arguments). Matching
    ``gaussian_psf``'s σ clamp and radius (``max(0.5, σ)``, ``max(1, round(3σ))``) matches
    its kernel, and the iteration itself is copied verbatim from skimage — estimate
    initialized to 0.5, ``+1e-12`` guard on the forward blur only, no clipping.

    What is left is float summation order, so the two agree to ~1e-15 relative rather than
    bit-for-bit; ``nodegraph.selftest`` asserts that against :func:`_rl` directly.

    The **dtype is preserved** (float32 in ⇒ float32 throughout), so a run under
    ``NODEGRAPH_FLOAT32=1`` gets the narrower, faster arithmetic the flag promises instead
    of being silently upcast here.

    Measured on one 210×1024² volume of the lab's 640 series, 10 iterations, real optics
    (NA 0.8, 663 nm, 0.287/0.288 µm ⇒ a 23×5×5 kernel) — see the Deconvolve node docstring
    for the table."""
    sig = tuple(max(0.5, float(s)) for s in sigmas)           # == gaussian_psf's clamp
    rad = tuple(max(1, int(round(3.0 * s))) for s in sig)      # == its radius_factor=3.0
    a = np.asarray(image)
    dt = a.dtype if a.dtype in (np.dtype(np.float32), np.dtype(np.float64)) \
        else np.dtype(float)
    mx = float(a.max()) or 1.0
    obs = a.astype(dt, copy=True) if a.dtype == dt else a.astype(dt)
    obs /= dt.type(mx)
    est = np.full(obs.shape, 0.5, dtype=dt)
    for _ in range(int(iters)):
        est *= _gauss_blur_zero(obs / (_gauss_blur_zero(est, sig, rad) + 1e-12), sig, rad)
    est *= dt.type(mx)
    return est
def _compute_deconvolve(ctx: EvalContext) -> Dataset:
    """Richardson–Lucy deconvolution with a metadata-derived PSF.

    **It drops ``bit_depth`` (§7c).** RL redistributes flux by dividing by a blurred
    estimate, and concentrating a blurred object back into fewer voxels drives those
    voxels ABOVE the input range: on the lab's 12-bit ND2 (max 4095) the shipped 10
    iterations returned a max of 5869. The node used to leave ``bit_depth=12`` stamped, so
    everything downstream kept sizing itself to a 4095 ceiling the data no longer respects
    — ``analysis.threshold``'s fixed level derives 2047.5 from it, ``enhance.gamma``'s
    full-range formula divides by it, and the Viewer's LUT windows to it.

    Dropping rather than widening is the honest call, and the distinction is the one §7c
    draws: ``bit_depth_after_sum`` widens where the new bound is *known* (n summed samples
    of a b-bit signal cannot exceed n·(2^b−1)), whereas RL's output bound depends on the
    PSF, the iteration count and the image, so there is no depth to widen to. Absent means
    "no declared integer scale", which every consumer already handles — the same signal a
    percentile Normalize leaves behind (:func:`nodegraph.metadata.value_rescaled`, which is
    this node's ``meta_transform`` for exactly that reason).

    **What 3D costs, and where.** In 3D the unit is a whole ``(Z,Y,X)`` volume — RL is global
    in z, so one displayed plane cannot be answered without the stack — and the node returns
    a lazy :class:`~nodegraph.streaming.VolumeComputeProvider`, so the pull is instant and
    the cost lands on the first plane a consumer reads. That plane computes the volume once
    and caches it as per-z slabs, after which z-scrubbing it is free; a different
    ``(position, timepoint)`` is a different volume. Measured on one 210×1024² volume of the
    lab's 640 series, 10 iterations, real optics (NA 0.8, 663 nm, 0.287/0.288 µm ⇒ a 23×5×5
    kernel):

    ================================================  =========  ===========
    kernel                                            wall       peak RSS
    ================================================  =========  ===========
    skimage RL on the n-D PSF array (:func:`_rl`)      226.5 s    20.9 GB
    separable Gaussian RL (:func:`_rl_gaussian`)       127.3 s     8.5 GB
    ================================================  =========  ===========

    Identical output (both peak at 8816.1 on the same input; equal to ~1e-15 relative) — the
    PSF is a Gaussian, so it is separable and self-mirroring, and the FFTs were never
    necessary (see :func:`_rl_gaussian`)."""
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("deconvolve needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.granularity is Granularity.WHOLE_VOLUME
    iters = int(ctx.params.get("iterations", 10))
    px = ctx.calib("pixel_size_um") or 0.1
    # C8 taught this node to honour an `emission_nm`/`na` override but left `z_step_um`
    # behind: the 3D-only socket declares `derive="z_step_um"`, so an UNSET socket already
    # resolves to the calibration — but the compute read the calibration DIRECTLY, so a
    # SET socket was silently dropped (a live control the kernel ignores, which the node
    # charter forbids). The calib read stays unconditional so the memo fence on z_step_um
    # survives in the common unset case; an override then wins and re-keys through params.
    zs = ctx.calib("z_step_um") or 0.5
    _zs_override = ctx.params.get("z_step_um")
    if _zs_override not in (None, ""):
        zs = float(_zs_override)
    # Resolve EVERY per-channel PSF eagerly: a lazy closure must not call ctx at tile-pull
    # time (the ReadContext freezes when compute returns — C1 / V2.04 §6b), and the reads
    # recorded here fold into the streaming fingerprint below. C8/H12: emission λ and NA
    # resolve PER CHANNEL via ctx.channel — a user override of the socket wins (one value,
    # all channels), else each channel's own emission (derive "emission_nm or 520") and NA.
    #
    # The per-channel PSF is carried as its **σ tuple**, not as a kernel array: the kernel
    # is a separable Gaussian, so :func:`_rl_gaussian` reconstructs it per axis and never
    # builds the n-D array at all (see that function for why this is the same computation).
    sigmas = {c: diffraction_sigmas(ctx.channel(c).param("emission_nm"),
                                    ctx.channel(c).param("na"), px, zs, is_3d)
              for c in range(ax.c)}
    cache = ctx.tiles
    # The pre-C1 eager fallback realizes the whole 6-D raster. It was gated on "one unit
    # does not fit half the tile cache", which is exactly backwards at scale: realizing
    # costs M·T·C units of RAM, so it can only be the right answer when the whole raster is
    # affordable — and a unit that already exceeds half the cache guarantees it is not. On
    # the lab's 640 series (12·16·210·1024², a 1.6 GiB volume) at a 2 GiB budget the old
    # gate answered "realize", i.e. allocate **337 GB** to avoid re-running a 1.6 GiB
    # volume: a MemoryError with extra steps, and on a small-RAM box the only thing 3D
    # Deconvolve could do. Streaming is always correct and never allocates more than one
    # unit; an oversized unit merely caches poorly (its per-z slabs are 8 MB each and cache
    # individually, so even a small budget keeps many of them). So: stream whenever there
    # is a cache to stream into, and keep the eager path for the cache-less context only
    # (a hand-built EvalContext — Engine always supplies one).
    if cache is None:
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    if is_3d:
                        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                        out[m, t, :, c] = _rl_gaussian(vol.astype(float), sigmas[c], iters)
                    else:
                        for z in range(ax.z):
                            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                            out[m, t, z, c] = _rl_gaussian(plane.astype(float),
                                                           sigmas[c], iters)
        return ds.with_image(ArrayProvider(out)).with_metadata(bit_depth=None)
    fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    if is_3d:                                    # lazy per-(m,t,c) volume (RL is global)
        return ds.with_image(VolumeComputeProvider(
            prov, lambda v, m, t, c: _rl_gaussian(v, sigmas[c], iters),
            fp=fp, cache=cache)).with_metadata(bit_depth=None)
    return ds.with_image(MapComputeProvider(     # lazy per-plane (iterative solver)
        prov, lambda a, m, t, z, c, gy0, gy1, gx0, gx1: _rl_gaussian(a, sigmas[c], iters),
        unit="plane", fp=fp, cache=cache)).with_metadata(bit_depth=None)
register_node(
    _compute_deconvolve,
    op_key="enhance.deconvolve", label="Deconvolve", category="enhancement",
    inputs=[
        InDataset(),
        InFloat("na", "NA", unit="", field=True, derive="na or 1.4",
                description=
                "Numerical aperture of the objective — one half of the PSF width. Together "
                "with emission λ it sets the diffraction limit, so a HIGHER NA means a "
                "tighter PSF and a gentler, more conservative deconvolution; too LOW and "
                "the model assumes more blur than the optics produced, which over-sharpens "
                "and can ring around bright edges. Reads the file's own NA when left on "
                "auto; override only if the metadata is wrong or absent (falls back to "
                "1.4). One value applies to every channel."),
        InFloat("emission_nm", "Emission λ", unit="nm", field=True,
                derive="emission_nm or 520",
                description=
                "Emission wavelength — the other half of the PSF width. LONGER wavelengths "
                "diffract more, giving a wider PSF and a stronger correction. Resolved PER "
                "CHANNEL from the file's own emission list when left on auto, which is the "
                "right behaviour for multi-channel data; setting it here overrides ALL "
                "channels with one value, so prefer auto unless the metadata is wrong "
                "(falls back to 520 nm)."),
        # anisotropic axial sampling is 3D-only (paired-float pattern, V2.03 §3 aniso)
        InFloat("z_step_um", "Z step", unit="um_axial", field=True, derive="z_step_um",
                available_in={"dim": frozenset({"3D"})},
                description=
                "Axial spacing between planes, used to build an ANISOTROPIC 3D PSF — real "
                "microscope voxels are far taller than they are wide, and a PSF that "
                "ignores that smears structure along z. Auto reads the file's own z step; "
                "override when the metadata is wrong (falls back to 0.5 µm). 3D only — in "
                "2D each plane is deconvolved with a purely lateral PSF."),
        InInt("iterations", "Iterations", default=10, field=False,
              description=
              "Richardson–Lucy iteration count — the sharpness/noise trade-off, and the "
              "only knob here that is not physical. MORE iterations recover more detail and "
              "then start amplifying noise into speckle and ringing around bright objects; "
              "the algorithm does not converge to a fixed point, so there is no \"enough\". "
              "Runtime is linear in this. 10 is a safe starting point; raise it in steps and "
              "stop when noise appears."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    supports_2d=True, supports_true_3d=True,
    # RL can push voxels above the input's declared range (12-bit in -> 5869 out on the
    # lab ND2), so the outgoing data has no declared integer scale. `value_rescaled` is the
    # ready-made §7c transform that drops `bit_depth`; the compute drops it in lockstep.
    meta_transform=_meta_value_rescaled,
    description="Richardson–Lucy deconvolution with a PSF derived from optics metadata "
                "(NA, emission λ, pixel/z size); 2D lateral PSF vs 3D anisotropic PSF. "
                "Output is no longer raw integer counts, so the declared bit depth is "
                "dropped.",
)
