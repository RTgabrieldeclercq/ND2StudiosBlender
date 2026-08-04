"""Remove Blobs (``enhance.remove_blobs``) — Detect bright round artifacts (dust, debris, hot pixels) with LoG/DoG and paint them out — zero / ring-mean interpolate / local median."""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, InFloat, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_plane_p
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Remove Blobs (CT Blob Subtract — dust, debris, hot pixels) ──────────────────

#: ``action`` → what replaces the pixels inside a detected blob.
_BLOB_ACTIONS = ("zero", "interpolate", "median")
def _disk_footprint(radius: int) -> np.ndarray:
    """A boolean disk structuring element of the given pixel ``radius`` (≥1)."""
    r = max(1, int(radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r
def _compute_remove_blobs(ctx: EvalContext) -> Dataset:
    """Detect round bright artifacts — dust, debris, hot pixels, fluorescent beads — and
    paint them out. Ports Cell-Tracker's **Blob Subtract**.

    LoG (``method=log``) or DoG (``method=dog``) blob detection runs on the per-plane
    normalized image, every detection is grown into a disk, and ``action`` decides what
    goes in its place:

    * ``zero`` — set the pixels to 0. Unambiguous, and leaves a hole a later threshold
      will read as background.
    * ``interpolate`` — fill with the mean of a ``fill_radius`` ring around the blobs.
      Blends into the local background; Cell-Tracker computes ONE ring mean for the whole
      plane, and that is kept so a recipe reproduces.
    * ``median`` — fill from a ``fill_radius`` median-filtered copy, so each blob takes
      its own neighbourhood's value rather than a plane-wide constant.

    **Per plane, no 2D/3D lever, deliberately.** The artifacts this removes live on the
    sensor or on a glass surface, so they appear at the same ``(y, x)`` in every z plane
    rather than as a compact 3D object — a volumetric detector would model them as tall
    ellipsoids and a 3D fill would smear a real structure through them. The footprint is
    ``WHOLE_PLANE``: detection needs the plane's min/max to normalize and ``interpolate``
    needs a plane-wide ring mean, so a tile would use the wrong population.

    Radii are per-channel (C8/H12) and physical: ``min_radius``/``max_radius`` are µm and
    convert to the detector's σ as ``σ = r/√2``, the same convention as ``detect.spots``,
    with the same 1-voxel floor (scipy's discretized LoG is ill-conditioned below ~1 px
    and floods the detector with spurious uniform response). Cell-Tracker instead passed
    its 'blob size' straight through as σ **and** used σ alone as the mask radius, which
    under-sizes the mask by √2; here the mask radius is the real blob radius ``σ·√2``
    before ``expand`` is applied, so the same numbers cover more of the artifact.

    Intensity scale unchanged (pixels are replaced with values from the same image), so no
    ``meta_transform`` is owed."""
    from scipy import ndimage as ndi
    from skimage.feature import blob_dog, blob_log
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("remove blobs needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    detector = blob_dog if str(modes.get("method") or "log") == "dog" else blob_log
    action = str(modes.get("action") or "zero")
    if action not in _BLOB_ACTIONS:
        raise ValueError(f"remove blobs: unknown action {action!r} — one of "
                         f"{list(_BLOB_ACTIONS)}")
    px = ctx.calib("pixel_size_um") or 0.1
    thr = float(ctx.params.get("threshold", 0.05))
    expand = max(1.0, float(ctx.params.get("expand", 1.5)))
    fill_px = 0
    if action != "zero":
        fill_px = max(1, int(round(to_pixels_v2(
            float(ctx.params.get("fill_radius", 1.5)), "um", pixel_size_um=px))))
    fill_disk = _disk_footprint(fill_px) if action == "interpolate" else None

    # Per-channel σ band, resolved EAGERLY (a lazy closure must not touch ctx — V2.04
    # §6b): each channel's radii derive from its own emission λ, so a multi-channel stack
    # gets a per-channel diffraction-limited scale rather than channel 0's.
    sigmas: Dict[int, Tuple[float, float]] = {}
    for c in range(ax.c):
        ch = ctx.channel(c)
        rmin = float(ch.param("min_radius", 0.3))
        rmax = float(ch.param("max_radius", 3.0))
        if rmax < rmin:
            raise ValueError(
                f"remove blobs: max radius ({rmax:g} µm) is below min radius "
                f"({rmin:g} µm) on channel {c} — there is no size band to search.")
        root2 = float(np.sqrt(2.0))
        sigmas[c] = (max(1.0, (rmin / px) / root2), max(1.0, (rmax / px) / root2))

    def clean(plane: np.ndarray, c: int) -> np.ndarray:
        f = np.asarray(plane, dtype=float)
        mn, mx = float(f.min()), float(f.max())
        if mx <= mn:                                   # blank plane — nothing to detect
            return f.copy()
        smin, smax = sigmas[c]
        blobs = detector((f - mn) / (mx - mn), min_sigma=smin, max_sigma=smax,
                         threshold=thr)
        if not len(blobs):
            return f.copy()
        yy, xx = np.ogrid[:f.shape[0], :f.shape[1]]
        mask = np.zeros(f.shape, dtype=bool)
        for blob in np.asarray(blobs, dtype=float):
            cy, cx, sig = float(blob[0]), float(blob[1]), float(blob[2])
            r = sig * np.sqrt(2.0) * expand            # σ → blob radius → safety margin
            mask |= ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        if not mask.any():
            return f.copy()
        out = f.copy()
        if action == "zero":
            out[mask] = 0.0
        elif action == "interpolate":
            ring = ndi.binary_dilation(mask, structure=fill_disk) & ~mask
            src = f[ring] if ring.any() else f[~mask]
            out[mask] = float(src.mean()) if src.size else 0.0
        else:
            out[mask] = ndi.median_filter(f, size=2 * fill_px + 1)[mask]
        return out

    cache = ctx.tiles
    if cache is None:                       # pre-C1 eager fallback (a bare EvalContext)
        out6 = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m, t, z, c in _each_plane_p(ctx, ax, "removing blobs"):
            out6[m, t, z, c] = clean(
                prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), c)
        return ds.with_image(ArrayProvider(out6))
    fp = stream_fp("remove_blobs", ctx.op_key, ctx.params,
                   ctx.reads.declared_reads(), (), prov)
    return ds.with_image(MapComputeProvider(
        prov, lambda a, m, t, z, c, *_: clean(a, c), unit="plane", fp=fp, cache=cache))
register_node(
    _compute_remove_blobs, op_key="enhance.remove_blobs", label="Remove Blobs",
    category="enhancement",
    inputs=[
        InDataset(),
        InFloat("min_radius", "Min radius", unit="um", field=False, default=0.3,
                pick_kind="radius", pick_peer="max_radius",
                derive="0.61*(emission_nm or 520)/(na or 1.4)/1000",
                description=
                "Smallest artifact radius to look for, in microns — the bottom of the size "
                "band searched. Auto uses the current optics' diffraction limit (emission λ "
                "and NA), i.e. the smallest thing that can be a real point source, resolved "
                "PER CHANNEL. Lower it to catch hot pixels and single-pixel specks; note "
                "that anything at or below this size is indistinguishable from your dimmest "
                "real puncta, so an over-eager setting deletes signal. Converted to the "
                "detector's σ as r/√2, floored at one pixel."),
        InFloat("max_radius", "Max radius", unit="um", field=False, default=3.0,
                pick_kind="radius", pick_peer="min_radius",
                derive="6*0.61*(emission_nm or 520)/(na or 1.4)/1000",
                description=
                "Largest artifact radius to look for, in microns — the top of the band. "
                "This is the knob that separates debris from cells: set it BELOW your cell "
                "radius and a cell can never be detected as an artifact, however bright. "
                "Widening the band also costs runtime, since a separate filter scale is "
                "evaluated across it. Auto uses 6x the diffraction limit, which covers dust "
                "and beads while staying under a typical nucleus. Must exceed Min radius."),
        InFloat("threshold", "Threshold", unit="", field=False, default=0.05,
                pick_kind="level",
                description=
                "Minimum detector response for a blob to be accepted, measured on the "
                "plane's NORMALIZED intensity, so it does not scale with your data's units. "
                "LOWER catches fainter artifacts and eventually starts deleting real "
                "structure; HIGHER removes only the obvious ones. 0.05 is Cell-Tracker's "
                "default and a sane start — this is the first thing to lower if visible "
                "debris survives, and the first to raise if cells start disappearing."),
        InFloat("expand", "Expand radius", unit="", field=False, default=1.5,
                description=
                "Multiplier on each detected blob's radius before it is painted out, so the "
                "artifact's faint halo goes with its core. The detector localizes the "
                "bright centre, not the full extent of a scattering speck, so a value above "
                "1 is normally needed; 1.5 is Cell-Tracker's default. Raise it if rings of "
                "residual glow are left behind, lower it if neighbouring real structure is "
                "being eaten. Area grows as the square of this, so 2.0 removes four times "
                "the pixels 1.0 does."),
        InFloat("fill_radius", "Fill radius", unit="um", field=False, default=1.5,
                pick_kind="radius",
                available_in={"action": frozenset({"interpolate", "median"})},
                description=
                "How far outside each blob to look for replacement values, in microns. For "
                "`interpolate` it is the width of the ring whose mean fills the blobs; for "
                "`median` it is the radius of the median filter the fill is taken from. It "
                "should be comparable to the blob size — much smaller and the ring sits "
                "inside the artifact's halo, so the fill inherits the glow it was meant to "
                "remove; much larger and it averages in unrelated structure. Not read by "
                "the `zero` action, which needs no source pixels."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("method", ["log", "dog"], default="log", label="Detector",
                description=
                "Which scale-space blob filter locates the artifacts, over the radius band "
                "between Min and Max radius. Same trade as in Spot Detection: accuracy "
                "against speed. It decides WHAT is found; Action decides what happens to it.",
                choice_docs={
                    "log":
                        "Laplacian of Gaussian at a series of scales — the more accurate of "
                        "the two at both locating an artifact and estimating its radius, "
                        "which matters here because the radius sets how much gets painted "
                        "out. The slower option, and the default.",
                    "dog":
                        "Difference of Gaussians, a cheap approximation of the same response. "
                        "Noticeably faster over a long series and equivalent on obvious dust "
                        "and hot pixels; its coarser scale sampling makes radii rougher, so "
                        "pair it with a slightly larger Expand radius.",
                }),
           Mode("action", list(_BLOB_ACTIONS), default="zero", label="Action",
                description=
                "What replaces the pixels inside each detected artifact. All three overwrite "
                "the same disks, so this does not change WHICH pixels are removed, only what "
                "is left behind — and therefore how visible the repair is to a human and to "
                "whatever measures the image next.",
                choice_docs={
                    "zero":
                        "Set the pixels to 0. Unambiguous and dependency-free — a later "
                        "threshold reads the patch as background and a later measurement sees "
                        "no signal there. It leaves an obvious black hole, which is a feature "
                        "when you want the repair to be auditable and a problem if a filter "
                        "downstream will smear that edge.",
                    "interpolate":
                        "Fill with the MEAN of a ring of width Fill radius around the blobs. "
                        "Blends into the local background so the patch is visually invisible. "
                        "Cell-Tracker computes one ring mean for the whole plane and that is "
                        "kept here so a recipe reproduces — meaning every blob on the plane "
                        "gets the same value, which shows if the background is uneven.",
                    "median":
                        "Fill from a Fill-radius median-filtered copy, so each blob takes its "
                        "OWN neighbourhood's value instead of a plane-wide constant. The best "
                        "choice on an uneven background or when artifacts sit on structure; it "
                        "costs a full median filter of the plane, which is the slowest of the "
                        "three.",
                })],
    # WHOLE_PLANE, no dim lever: plane min/max normalization + a plane-wide ring mean, and
    # sensor/glass artifacts are per-plane by nature (see the compute docstring).
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Detect bright round artifacts (dust, debris, hot pixels) with LoG/DoG and "
                "paint them out — zero / ring-mean interpolate / local median. Ports "
                "Cell-Tracker's Blob Subtract; radii in µm, per channel, mask radius σ·√2.")
